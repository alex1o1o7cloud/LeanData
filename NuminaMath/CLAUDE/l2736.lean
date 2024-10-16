import Mathlib

namespace NUMINAMATH_CALUDE_special_sequences_general_terms_l2736_273613

/-- Two sequences of positive real numbers satisfying specific conditions -/
structure SpecialSequences where
  a : ℕ → ℝ
  b : ℕ → ℝ
  a_pos : ∀ n, a n > 0
  b_pos : ∀ n, b n > 0
  arithmetic : ∀ n, 2 * b n = a n + a (n + 1)
  geometric : ∀ n, (a (n + 1))^2 = b n * b (n + 1)
  initial_a1 : a 1 = 1
  initial_b1 : b 1 = 2
  initial_a2 : a 2 = 3

/-- The general terms of the special sequences -/
theorem special_sequences_general_terms (s : SpecialSequences) :
    (∀ n, s.a n = n * (n + 1) / 2) ∧
    (∀ n, s.b n = (n + 1)^2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_special_sequences_general_terms_l2736_273613


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_6342_l2736_273616

/-- Calculates the cost of white washing a room with given dimensions, door, windows, and cost per square foot. -/
def whitewashing_cost (room_length room_width room_height : ℝ)
                      (door_width door_height : ℝ)
                      (window_width window_height : ℝ)
                      (num_windows : ℕ)
                      (cost_per_sqft : ℝ) : ℝ :=
  let total_wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_width * door_height
  let window_area := num_windows * (window_width * window_height)
  let net_area := total_wall_area - door_area - window_area
  net_area * cost_per_sqft

/-- Theorem stating that the cost of white washing the given room is 6342 Rs. -/
theorem whitewashing_cost_is_6342 :
  whitewashing_cost 25 15 12 6 3 4 3 3 7 = 6342 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_6342_l2736_273616


namespace NUMINAMATH_CALUDE_floor_cube_negative_fraction_l2736_273621

theorem floor_cube_negative_fraction : ⌊(-7/4)^3⌋ = -6 := by
  sorry

end NUMINAMATH_CALUDE_floor_cube_negative_fraction_l2736_273621


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2736_273641

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * (x - 1)^3 = 24
def equation2 (x : ℝ) : Prop := (x - 3)^2 = 64

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 3 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x₁ x₂ : ℝ, equation2 x₁ ∧ equation2 x₂ ∧ x₁ = 11 ∧ x₂ = -5 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2736_273641


namespace NUMINAMATH_CALUDE_largest_odd_equal_cost_l2736_273604

/-- Calculates the sum of digits in decimal representation -/
def sumDigitsDecimal (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumDigitsDecimal (n / 10)

/-- Calculates the sum of digits in binary representation with two trailing zeros -/
def sumDigitsBinary (n : Nat) : Nat :=
  if n < 4 then 0 else (n % 2) + sumDigitsBinary (n / 2)

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Prop := n % 2 = 1

/-- Theorem statement -/
theorem largest_odd_equal_cost :
  ∃ (n : Nat), n < 2000 ∧ isOdd n ∧
    sumDigitsDecimal n = sumDigitsBinary n ∧
    ∀ (m : Nat), m < 2000 ∧ isOdd m ∧ sumDigitsDecimal m = sumDigitsBinary m → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_odd_equal_cost_l2736_273604


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l2736_273667

theorem smallest_n_with_common_factor : 
  ∃ (n : ℕ), n > 0 ∧ n = 10 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬∃ (k : ℕ), k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (5*m + 4)) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (5*n + 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l2736_273667


namespace NUMINAMATH_CALUDE_distribution_count_correct_l2736_273619

/-- The number of ways to distribute 5 indistinguishable objects into 4 distinguishable containers,
    where 2 containers are of type A and 2 are of type B,
    with at least one object in a type A container. -/
def distribution_count : ℕ := 30

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The total number of rooms -/
def num_rooms : ℕ := 4

/-- The number of rooms with a garden view -/
def num_garden_view : ℕ := 2

/-- The number of rooms without a garden view -/
def num_no_garden_view : ℕ := 2

/-- Theorem stating that the distribution count is correct -/
theorem distribution_count_correct :
  distribution_count = 30 ∧
  num_cousins = 5 ∧
  num_rooms = 4 ∧
  num_garden_view = 2 ∧
  num_no_garden_view = 2 ∧
  num_garden_view + num_no_garden_view = num_rooms :=
by sorry

end NUMINAMATH_CALUDE_distribution_count_correct_l2736_273619


namespace NUMINAMATH_CALUDE_watch_time_theorem_l2736_273610

/-- Represents a season of the TV show -/
structure Season where
  episodes : Nat
  minutesPerEpisode : Nat

/-- Calculates the total number of days needed to watch the show -/
def daysToWatchShow (seasons : List Season) (hoursPerDay : Nat) : Nat :=
  let totalMinutes := seasons.foldl (fun acc s => acc + s.episodes * s.minutesPerEpisode) 0
  let minutesPerDay := hoursPerDay * 60
  (totalMinutes + minutesPerDay - 1) / minutesPerDay

/-- The main theorem stating it takes 35 days to watch the show -/
theorem watch_time_theorem (seasons : List Season) (hoursPerDay : Nat) :
  seasons = [
    ⟨30, 22⟩, ⟨28, 25⟩, ⟨27, 29⟩, ⟨20, 31⟩, ⟨25, 27⟩, ⟨20, 35⟩
  ] →
  hoursPerDay = 2 →
  daysToWatchShow seasons hoursPerDay = 35 := by
  sorry

#eval daysToWatchShow [
  ⟨30, 22⟩, ⟨28, 25⟩, ⟨27, 29⟩, ⟨20, 31⟩, ⟨25, 27⟩, ⟨20, 35⟩
] 2

end NUMINAMATH_CALUDE_watch_time_theorem_l2736_273610


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l2736_273606

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (525 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (525 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l2736_273606


namespace NUMINAMATH_CALUDE_population_ratio_l2736_273614

/-- The population ratio problem -/
theorem population_ratio 
  (pop_x pop_y pop_z : ℕ) 
  (h1 : pop_x = 7 * pop_y) 
  (h2 : pop_y = 2 * pop_z) : 
  pop_x / pop_z = 14 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l2736_273614


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l2736_273628

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 90 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨27, 15, 6⟩ = 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l2736_273628


namespace NUMINAMATH_CALUDE_set_intersection_example_l2736_273699

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2, 4} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l2736_273699


namespace NUMINAMATH_CALUDE_population_growth_l2736_273601

/-- Given an initial population that increases by 10% annually for 2 years
    resulting in 14,520 people, prove that the initial population was 12,000. -/
theorem population_growth (P : ℝ) : 
  (P * (1 + 0.1)^2 = 14520) → P = 12000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l2736_273601


namespace NUMINAMATH_CALUDE_vector_operation_l2736_273698

/-- Given two 2D vectors a and b, prove that the result of the vector operation is (-1, 2) -/
theorem vector_operation (a b : Fin 2 → ℝ) (ha : a = ![1, 1]) (hb : b = ![1, -1]) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = ![(-1 : ℝ), 2] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2736_273698


namespace NUMINAMATH_CALUDE_football_team_handedness_l2736_273652

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : right_handed = 62)
  (h4 : throwers ≤ right_handed) :
  (total_players - right_handed : ℚ) / (total_players - throwers) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_football_team_handedness_l2736_273652


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2736_273682

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  c = 2 * b - a →          -- a, b, c form an arithmetic sequence
  a * b * c = 125 →        -- product condition
  b ≥ 5 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    c' = 2 * b' - a' ∧ a' * b' * c' = 125 ∧ b' = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2736_273682


namespace NUMINAMATH_CALUDE_smallest_3_4_cut_is_14_l2736_273630

/-- A positive integer n is m-cut if n-2 is divisible by m -/
def is_m_cut (n m : ℕ) : Prop :=
  n > 2 ∧ m > 2 ∧ (n - 2) % m = 0

/-- The smallest positive integer that is both 3-cut and 4-cut -/
def smallest_3_4_cut : ℕ := 14

/-- Theorem stating that 14 is the smallest positive integer that is both 3-cut and 4-cut -/
theorem smallest_3_4_cut_is_14 :
  (∀ n : ℕ, n < smallest_3_4_cut → ¬(is_m_cut n 3 ∧ is_m_cut n 4)) ∧
  (is_m_cut smallest_3_4_cut 3 ∧ is_m_cut smallest_3_4_cut 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_3_4_cut_is_14_l2736_273630


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2736_273623

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2736_273623


namespace NUMINAMATH_CALUDE_min_value_sum_l2736_273658

theorem min_value_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → a + b ≤ x + y ∧ a + b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2736_273658


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l2736_273679

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) (seconds_per_hour : ℕ) : ℝ :=
  speed_km_per_second * (seconds_per_hour : ℝ)

/-- Theorem: A space shuttle orbiting at 9 km/s is equivalent to 32400 km/h -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 9 3600 = 32400 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l2736_273679


namespace NUMINAMATH_CALUDE_extreme_value_in_interval_l2736_273647

/-- The function f(x) = x ln x + (1/2)x² - 3x has an extreme value in the interval (3/2, 2) -/
theorem extreme_value_in_interval :
  ∃ x : ℝ, (3/2 < x ∧ x < 2) ∧
    ∀ y : ℝ, (3/2 < y ∧ y < 2) →
      (x * Real.log x + (1/2) * x^2 - 3*x ≤ y * Real.log y + (1/2) * y^2 - 3*y ∨
       x * Real.log x + (1/2) * x^2 - 3*x ≥ y * Real.log y + (1/2) * y^2 - 3*y) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_in_interval_l2736_273647


namespace NUMINAMATH_CALUDE_possible_value_of_n_l2736_273622

theorem possible_value_of_n : ∃ n : ℕ, 
  3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_possible_value_of_n_l2736_273622


namespace NUMINAMATH_CALUDE_wine_without_cork_cost_l2736_273607

def bottle_with_cork : ℝ := 2.10
def cork : ℝ := 2.05

theorem wine_without_cork_cost (bottle_without_cork : ℝ) 
  (h1 : bottle_without_cork > cork) : 
  bottle_without_cork - cork > 0.05 := by
  sorry

end NUMINAMATH_CALUDE_wine_without_cork_cost_l2736_273607


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l2736_273663

/-- A quadratic function f(x) = x^2 + 2x + k has its vertex on the x-axis if and only if k = 1 -/
theorem vertex_on_x_axis (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k ≥ x^2 + 2*x + k) ↔ 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l2736_273663


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l2736_273650

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (new_decaf_percent : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.4)
  (h3 : new_decaf_percent = 0.6)
  (h4 : final_decaf_percent = 0.44)
  : ∃ (additional_coffee : ℝ),
    additional_coffee = 100 ∧
    (initial_stock * initial_decaf_percent + additional_coffee * new_decaf_percent) / (initial_stock + additional_coffee) = final_decaf_percent :=
by sorry


end NUMINAMATH_CALUDE_coffee_stock_problem_l2736_273650


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l2736_273648

/-- The equation of a line passing through (0, 2) with slope 2 is 2x - y + 2 = 0 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (y - 2 = 2 * (x - 0)) ↔ (2 * x - y + 2 = 0) := by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l2736_273648


namespace NUMINAMATH_CALUDE_distinct_elements_l2736_273694

theorem distinct_elements (x : ℕ) : 
  (5 ≠ x ∧ 5 ≠ x^2 - 4*x ∧ x ≠ x^2 - 4*x) ↔ (x ≠ 5 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_distinct_elements_l2736_273694


namespace NUMINAMATH_CALUDE_sara_movie_purchase_cost_l2736_273688

/-- The amount Sara spent on movie theater tickets -/
def theater_ticket_cost : ℚ := 10.62

/-- The number of movie theater tickets Sara bought -/
def number_of_tickets : ℕ := 2

/-- The cost of renting a movie -/
def rental_cost : ℚ := 1.59

/-- The total amount Sara spent on movies -/
def total_spent : ℚ := 36.78

/-- Theorem: Given the conditions, Sara spent $13.95 on buying the movie -/
theorem sara_movie_purchase_cost :
  total_spent - (theater_ticket_cost * number_of_tickets + rental_cost) = 13.95 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_purchase_cost_l2736_273688


namespace NUMINAMATH_CALUDE_fraction_equality_l2736_273693

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7^2) :
  d / a = 1 / 122.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2736_273693


namespace NUMINAMATH_CALUDE_min_removal_for_given_structure_l2736_273631

/-- Represents the structure of triangles made with toothpicks -/
structure TriangleStructure where
  totalToothpicks : ℕ
  baseTriangles : ℕ
  rows : ℕ

/-- Calculates the number of toothpicks needed to be removed to eliminate all triangles -/
def minRemovalCount (ts : TriangleStructure) : ℕ :=
  ts.rows

/-- Theorem stating that for the given structure, 5 toothpicks need to be removed -/
theorem min_removal_for_given_structure :
  let ts : TriangleStructure := {
    totalToothpicks := 50,
    baseTriangles := 5,
    rows := 5
  }
  minRemovalCount ts = 5 := by
  sorry

#check min_removal_for_given_structure

end NUMINAMATH_CALUDE_min_removal_for_given_structure_l2736_273631


namespace NUMINAMATH_CALUDE_prob_rain_weekend_is_correct_l2736_273637

-- Define the probabilities of rain for each day
def prob_rain_friday : ℝ := 0.30
def prob_rain_saturday : ℝ := 0.60
def prob_rain_sunday : ℝ := 0.40

-- Define the probability of rain on at least one day during the weekend
def prob_rain_weekend : ℝ := 1 - (1 - prob_rain_friday) * (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

-- Theorem statement
theorem prob_rain_weekend_is_correct : 
  prob_rain_weekend = 0.832 := by sorry

end NUMINAMATH_CALUDE_prob_rain_weekend_is_correct_l2736_273637


namespace NUMINAMATH_CALUDE_pizza_sharing_l2736_273620

theorem pizza_sharing (total pizza_jovin pizza_anna pizza_olivia : ℚ) : 
  total = 1 →
  pizza_jovin = 1/3 →
  pizza_anna = 1/6 →
  pizza_olivia = 1/4 →
  total - (pizza_jovin + pizza_anna + pizza_olivia) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sharing_l2736_273620


namespace NUMINAMATH_CALUDE_triangle_base_length_l2736_273617

theorem triangle_base_length (height : ℝ) (area : ℝ) : 
  height = 8 → area = 24 → (1/2) * 6 * height = area :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2736_273617


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2736_273676

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℝ), (x^2 + y^2)^2 ≤ n * (x^4 + y^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y : ℝ), (x^2 + y^2)^2 > m * (x^4 + y^4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2736_273676


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l2736_273684

theorem subtraction_puzzle :
  ∀ (A B C D E F H I J : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
     E ≠ F ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
     F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
     H ≠ I ∧ H ≠ J ∧
     I ≠ J) →
    (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧
    (1 ≤ D ∧ D ≤ 9) ∧ (1 ≤ E ∧ E ≤ 9) ∧ (1 ≤ F ∧ F ≤ 9) ∧
    (1 ≤ H ∧ H ≤ 9) ∧ (1 ≤ I ∧ I ≤ 9) ∧ (1 ≤ J ∧ J ≤ 9) →
    100 * A + 10 * B + C - (100 * D + 10 * E + F) = 100 * H + 10 * I + J →
    A + B + C + D + E + F + H + I + J = 45 →
    A + B + C = 18 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l2736_273684


namespace NUMINAMATH_CALUDE_regular_pencil_price_correct_l2736_273697

/-- The price of a regular pencil in a stationery store --/
def regular_pencil_price : ℝ :=
  let pencil_with_eraser_price : ℝ := 0.8
  let short_pencil_price : ℝ := 0.4
  let pencils_with_eraser_sold : ℕ := 200
  let regular_pencils_sold : ℕ := 40
  let short_pencils_sold : ℕ := 35
  let total_sales : ℝ := 194
  0.5

/-- Theorem stating that the regular pencil price is correct --/
theorem regular_pencil_price_correct :
  let pencil_with_eraser_price : ℝ := 0.8
  let short_pencil_price : ℝ := 0.4
  let pencils_with_eraser_sold : ℕ := 200
  let regular_pencils_sold : ℕ := 40
  let short_pencils_sold : ℕ := 35
  let total_sales : ℝ := 194
  pencil_with_eraser_price * pencils_with_eraser_sold +
  regular_pencil_price * regular_pencils_sold +
  short_pencil_price * short_pencils_sold = total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_regular_pencil_price_correct_l2736_273697


namespace NUMINAMATH_CALUDE_inequality_preservation_l2736_273672

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 1 > b - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2736_273672


namespace NUMINAMATH_CALUDE_volleyball_team_median_age_l2736_273608

/-- Represents the age distribution of the volleyball team --/
def AgeDistribution : List (Nat × Nat) :=
  [(18, 3), (19, 5), (20, 2), (21, 1), (22, 1)]

/-- The total number of team members --/
def TotalMembers : Nat := 12

/-- Calculates the median age of the team --/
def medianAge (dist : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median age of the team is 19 --/
theorem volleyball_team_median_age :
  medianAge AgeDistribution TotalMembers = 19 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_median_age_l2736_273608


namespace NUMINAMATH_CALUDE_average_of_first_and_last_l2736_273674

def numbers : List Int := [-3, 2, 5, 8, 11]

def is_valid_arrangement (arr : List Int) : Prop :=
  arr.length = 5 ∧
  arr.toFinset = numbers.toFinset ∧
  (arr.getD 1 0 = 11 ∨ arr.getD 2 0 = 11) ∧
  (arr.getD 2 0 = -3 ∨ arr.getD 3 0 = -3 ∨ arr.getD 4 0 = -3) ∧
  (arr.getD 0 0 = 5 ∨ arr.getD 2 0 = 5 ∨ arr.getD 4 0 = 5)

theorem average_of_first_and_last (arr : List Int) :
  is_valid_arrangement arr →
  (arr.getD 0 0 + arr.getD 4 0) / 2 = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_and_last_l2736_273674


namespace NUMINAMATH_CALUDE_horner_v₃_value_l2736_273686

def f (x : ℝ) : ℝ := 7 * x^5 + 5 * x^4 + 3 * x^3 + x^2 + x + 2

def horner_method (x : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let v₀ : ℝ := 7
  let v₁ : ℝ := v₀ * x + 5
  let v₂ : ℝ := v₁ * x + 3
  let v₃ : ℝ := v₂ * x + 1
  (v₀, v₁, v₂, v₃)

theorem horner_v₃_value :
  (horner_method 2).2.2.2 = 83 := by sorry

end NUMINAMATH_CALUDE_horner_v₃_value_l2736_273686


namespace NUMINAMATH_CALUDE_spinner_probability_l2736_273625

theorem spinner_probability : 
  let spinner_sections : ℕ := 4
  let e_section : ℕ := 1
  let spins : ℕ := 2
  let prob_not_e_single : ℚ := (spinner_sections - e_section) / spinner_sections
  (prob_not_e_single ^ spins) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2736_273625


namespace NUMINAMATH_CALUDE_selection_theorem_l2736_273681

def number_of_students : ℕ := 10
def number_to_choose : ℕ := 3
def number_of_specific_students : ℕ := 2

def selection_ways : ℕ :=
  Nat.choose (number_of_students - 1) number_to_choose -
  Nat.choose (number_of_students - 1 - number_of_specific_students) number_to_choose

theorem selection_theorem :
  selection_ways = 49 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2736_273681


namespace NUMINAMATH_CALUDE_unique_solution_cubic_rational_equation_l2736_273611

theorem unique_solution_cubic_rational_equation :
  ∃! x : ℝ, (x^3 - 3*x^2 + 2*x)/(x^2 + 2*x + 1) + 2*x = -8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_rational_equation_l2736_273611


namespace NUMINAMATH_CALUDE_max_cross_section_area_l2736_273680

-- Define the prism
def prism_base_side_length : ℝ := 6

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 5 * x - 3 * y + 2 * z = 20

-- Define the cross-section area function
noncomputable def cross_section_area : ℝ := 9

-- Theorem statement
theorem max_cross_section_area :
  ∀ (area : ℝ),
    (∃ (x y z : ℝ), cutting_plane x y z ∧ 
      x^2 + y^2 ≤ (prism_base_side_length / 2)^2) →
    area ≤ cross_section_area :=
by sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l2736_273680


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l2736_273646

theorem parametric_to_ordinary_equation :
  ∀ (x y t : ℝ), x = t + 1 ∧ y = 3 - t^2 → y = -x^2 + 2*x + 2 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l2736_273646


namespace NUMINAMATH_CALUDE_r_div_p_equals_1100_l2736_273605

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of different numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_cards drawn_cards

/-- The probability of drawing three cards with one number and two with another -/
def r : ℚ := (13200 : ℚ) / Nat.choose total_cards drawn_cards

/-- Theorem stating the ratio of r to p -/
theorem r_div_p_equals_1100 : r / p = 1100 := by sorry

end NUMINAMATH_CALUDE_r_div_p_equals_1100_l2736_273605


namespace NUMINAMATH_CALUDE_cube_plus_one_expansion_problem_solution_l2736_273692

theorem cube_plus_one_expansion (n : ℕ) : 
  n^3 + 3*(n^2) + 3*n + 1 = (n + 1)^3 :=
by sorry

theorem problem_solution : 
  98^3 + 3*(98^2) + 3*98 + 1 = 970299 :=
by sorry

end NUMINAMATH_CALUDE_cube_plus_one_expansion_problem_solution_l2736_273692


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l2736_273664

theorem arithmetic_square_root_of_nine :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 9 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l2736_273664


namespace NUMINAMATH_CALUDE_gcd_18_30_l2736_273696

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l2736_273696


namespace NUMINAMATH_CALUDE_length_AE_is_5_sqrt_5_div_3_l2736_273634

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (2, 0)

-- Define E as the intersection point of AB and CD
def E : ℝ × ℝ := sorry

-- Theorem statement
theorem length_AE_is_5_sqrt_5_div_3 :
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A E = (5 * Real.sqrt 5) / 3 := by sorry

end NUMINAMATH_CALUDE_length_AE_is_5_sqrt_5_div_3_l2736_273634


namespace NUMINAMATH_CALUDE_pet_store_cats_sold_l2736_273639

theorem pet_store_cats_sold (dogs : ℕ) (cats : ℕ) : 
  cats = 3 * dogs →
  cats = 2 * (dogs + 8) →
  cats = 48 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_sold_l2736_273639


namespace NUMINAMATH_CALUDE_volume_removed_tetrahedra_2x2x3_l2736_273602

/-- Represents a rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the volume of removed tetrahedra when corners are sliced to form regular hexagons -/
def volume_removed_tetrahedra (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem: The volume of removed tetrahedra for a 2x2x3 rectangular prism is (22 - 46√2) / 3 -/
theorem volume_removed_tetrahedra_2x2x3 :
  volume_removed_tetrahedra ⟨2, 2, 3⟩ = (22 - 46 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_tetrahedra_2x2x3_l2736_273602


namespace NUMINAMATH_CALUDE_scenario1_probability_scenario2_probability_l2736_273642

-- Define the probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots for each scenario
def shots_scenario1 : ℕ := 3
def shots_scenario2 : ℕ := 2

-- Theorem for scenario 1
theorem scenario1_probability : 
  (1 - prob_A_hit ^ shots_scenario1) = 19/27 := by sorry

-- Theorem for scenario 2
theorem scenario2_probability : 
  (Nat.choose shots_scenario2 shots_scenario2 * prob_A_hit ^ shots_scenario2) *
  (Nat.choose shots_scenario2 1 * prob_B_hit ^ 1 * (1 - prob_B_hit) ^ (shots_scenario2 - 1)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_scenario1_probability_scenario2_probability_l2736_273642


namespace NUMINAMATH_CALUDE_sphere_volume_l2736_273691

theorem sphere_volume (A : Real) (V : Real) :
  A = 9 * Real.pi →  -- area of the main view (circle)
  V = (4 / 3) * Real.pi * (3 ^ 3) →  -- volume formula with radius 3
  V = 36 * Real.pi :=  -- expected volume
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l2736_273691


namespace NUMINAMATH_CALUDE_opposite_areas_equal_l2736_273638

/-- Represents a rectangle with an interior point connected to midpoints of its sides --/
structure RectangleWithInteriorPoint where
  /-- The rectangle --/
  rectangle : Set (ℝ × ℝ)
  /-- The interior point --/
  interior_point : ℝ × ℝ
  /-- The midpoints of the rectangle's sides --/
  midpoints : Fin 4 → ℝ × ℝ
  /-- The areas of the four polygons formed --/
  polygon_areas : Fin 4 → ℝ

/-- The sum of opposite polygon areas is equal --/
theorem opposite_areas_equal (r : RectangleWithInteriorPoint) : 
  r.polygon_areas 0 + r.polygon_areas 2 = r.polygon_areas 1 + r.polygon_areas 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_areas_equal_l2736_273638


namespace NUMINAMATH_CALUDE_beijing_olympics_edition_l2736_273627

/-- The year of the first modern Olympic Games -/
def first_olympics : ℕ := 1896

/-- The frequency of Olympic Games in years -/
def olympics_frequency : ℕ := 4

/-- The year we're considering -/
def target_year : ℕ := 2008

/-- Calculate the edition number of the Olympic Games for a given year -/
def olympic_edition (year : ℕ) : ℕ :=
  (year - first_olympics) / olympics_frequency + 1

theorem beijing_olympics_edition :
  olympic_edition target_year = 29 := by
  sorry

end NUMINAMATH_CALUDE_beijing_olympics_edition_l2736_273627


namespace NUMINAMATH_CALUDE_quadratic_negative_roots_probability_l2736_273655

/-- The probability that a quadratic equation with a randomly selected coefficient has two negative roots -/
theorem quadratic_negative_roots_probability : 
  ∃ (f : ℝ → ℝ → ℝ → Prop) (P : Set ℝ → ℝ),
    (∀ p x₁ x₂, f p x₁ x₂ ↔ x₁^2 + 2*p*x₁ + 3*p - 2 = 0 ∧ x₂^2 + 2*p*x₂ + 3*p - 2 = 0 ∧ x₁ < 0 ∧ x₂ < 0) →
    (P (Set.Icc 0 5) = 5) →
    P {p ∈ Set.Icc 0 5 | ∃ x₁ x₂, f p x₁ x₂} / P (Set.Icc 0 5) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_roots_probability_l2736_273655


namespace NUMINAMATH_CALUDE_equation_solution_l2736_273685

theorem equation_solution : ∃ x : ℝ, 
  (Real.sqrt (7 * x - 3) + Real.sqrt (2 * x - 2) = 5) ∧ 
  (7 * x - 3 ≥ 0) ∧ 
  (2 * x - 2 ≥ 0) ∧ 
  (abs (x - 20.14) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2736_273685


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2736_273615

theorem arithmetic_geometric_inequality (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (arith_prog : ∃ r : ℝ, b = a + r ∧ c = a + 2*r ∧ d = a + 3*r)
  (geom_prog : ∃ q : ℝ, e = a * q ∧ f = a * q^2 ∧ d = a * q^3) :
  b * c ≥ e * f := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2736_273615


namespace NUMINAMATH_CALUDE_players_satisfy_distances_l2736_273640

/-- Represents the positions of four players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two player positions -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required distances between players -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem players_satisfy_distances : 
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_players_satisfy_distances_l2736_273640


namespace NUMINAMATH_CALUDE_bus_problem_l2736_273659

/-- The number of children on a bus after a stop, given the initial number,
    the number who got off, and the difference between those who got off and on. -/
def children_after_stop (initial : ℕ) (got_off : ℕ) (diff : ℕ) : ℤ :=
  initial - got_off + (got_off - diff)

/-- Theorem stating that given the initial conditions, 
    the number of children on the bus after the stop is 12. -/
theorem bus_problem : children_after_stop 36 68 24 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2736_273659


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_half_unit_l2736_273665

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  sorry

/-- The main theorem stating that the area of the quadrilateral is 0.5 square units -/
theorem quadrilateral_area_is_half_unit : 
  let l1 : Line := { a := 3, b := 4, c := -12 }
  let l2 : Line := { a := 6, b := -4, c := -12 }
  let l3 : Line := { a := 1, b := 0, c := -3 }
  let l4 : Line := { a := 0, b := 1, c := -1 }
  let p1 := intersectionPoint l1 l2
  let p2 := intersectionPoint l1 l3
  let p3 := intersectionPoint l2 l3
  let p4 := intersectionPoint l1 l4
  quadrilateralArea p1 p2 p3 p4 = 0.5 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_half_unit_l2736_273665


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_l2736_273600

theorem quadrilateral_diagonal (sides : Finset ℝ) 
  (h_sides : sides = {1, 2, 2.8, 5, 7.5}) : 
  ∃ (diagonal : ℝ), diagonal ∈ sides ∧
  (∀ (a b c : ℝ), a ∈ sides → b ∈ sides → c ∈ sides → 
   a ≠ diagonal → b ≠ diagonal → c ≠ diagonal → 
   a + b > diagonal ∧ b + c > diagonal ∧ a + c > diagonal) ∧
  diagonal = 2.8 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_l2736_273600


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_wind_l2736_273633

/-- Given probabilities of events A and B, and their intersection, prove the conditional probability P(A|B) -/
theorem conditional_probability_rain_given_wind 
  (P_A : ℚ) (P_B : ℚ) (P_A_and_B : ℚ)
  (h1 : P_A = 4/15)
  (h2 : P_B = 2/15)
  (h3 : P_A_and_B = 1/10)
  : P_A_and_B / P_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_wind_l2736_273633


namespace NUMINAMATH_CALUDE_union_equals_B_implies_a_range_l2736_273636

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x - 6 < 0}
def C : Set ℝ := {x | x^2 - 2*x - 15 < 0}

-- State the theorem
theorem union_equals_B_implies_a_range (a : ℝ) :
  A ∪ B a = B a → a ∈ Set.Icc (-5) (-1) :=
by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end NUMINAMATH_CALUDE_union_equals_B_implies_a_range_l2736_273636


namespace NUMINAMATH_CALUDE_popsicle_sticks_theorem_l2736_273661

def steve_sticks : ℕ := 12

def sid_sticks : ℕ := 2 * steve_sticks

def sam_sticks : ℕ := 3 * sid_sticks

def total_sticks : ℕ := steve_sticks + sid_sticks + sam_sticks

theorem popsicle_sticks_theorem : total_sticks = 108 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_theorem_l2736_273661


namespace NUMINAMATH_CALUDE_production_cost_at_most_80_l2736_273653

/-- Represents the monthly production and financial data for an electronic component manufacturer -/
structure ComponentManufacturer where
  shippingCost : ℝ
  fixedCosts : ℝ
  monthlyProduction : ℕ
  lowestSellingPrice : ℝ

/-- Theorem stating that the production cost per component is at most $80 -/
theorem production_cost_at_most_80 (m : ComponentManufacturer) 
  (h1 : m.shippingCost = 5)
  (h2 : m.fixedCosts = 16500)
  (h3 : m.monthlyProduction = 150)
  (h4 : m.lowestSellingPrice = 195) :
  ∃ (productionCost : ℝ), productionCost ≤ 80 ∧ 
    (m.monthlyProduction : ℝ) * productionCost + 
    (m.monthlyProduction : ℝ) * m.shippingCost + 
    m.fixedCosts ≤ (m.monthlyProduction : ℝ) * m.lowestSellingPrice :=
by sorry

end NUMINAMATH_CALUDE_production_cost_at_most_80_l2736_273653


namespace NUMINAMATH_CALUDE_range_of_a_l2736_273687

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2*x - 1)/(x - 1) < 0 → x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0 ∧ (2*x - 1)/(x - 1) ≥ 0) →
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2736_273687


namespace NUMINAMATH_CALUDE_interval_constraint_l2736_273656

theorem interval_constraint (x : ℝ) : (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) := by
  sorry

end NUMINAMATH_CALUDE_interval_constraint_l2736_273656


namespace NUMINAMATH_CALUDE_stevens_cards_l2736_273651

def number_of_groups : ℕ := 5
def cards_per_group : ℕ := 6

theorem stevens_cards : number_of_groups * cards_per_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_stevens_cards_l2736_273651


namespace NUMINAMATH_CALUDE_dress_price_is_seven_l2736_273618

def total_revenue : ℝ := 69
def num_dresses : ℕ := 7
def num_shirts : ℕ := 4
def price_shirt : ℝ := 5

theorem dress_price_is_seven :
  ∃ (price_dress : ℝ),
    price_dress * num_dresses + price_shirt * num_shirts = total_revenue ∧
    price_dress = 7 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_is_seven_l2736_273618


namespace NUMINAMATH_CALUDE_max_log_expression_l2736_273673

theorem max_log_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : 4 * a - 2 * b + 25 * c = 0) :
  (∀ x y z, x > 0 → y > 0 → z > 0 → 4 * x - 2 * y + 25 * z = 0 →
    Real.log x + Real.log z - 2 * Real.log y ≤ Real.log a + Real.log c - 2 * Real.log b) ∧
  Real.log a + Real.log c - 2 * Real.log b = -2 := by
sorry

end NUMINAMATH_CALUDE_max_log_expression_l2736_273673


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l2736_273635

theorem first_number_in_ratio (A B : ℕ+) : 
  (A : ℚ) / (B : ℚ) = 8 / 9 →
  Nat.lcm A B = 432 →
  A = 48 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l2736_273635


namespace NUMINAMATH_CALUDE_equal_to_2x_6_l2736_273603

theorem equal_to_2x_6 (x : ℝ) : 2 * x^7 / x = 2 * x^6 := by sorry

end NUMINAMATH_CALUDE_equal_to_2x_6_l2736_273603


namespace NUMINAMATH_CALUDE_number_difference_l2736_273683

theorem number_difference (x y : ℕ) : 
  x + y = 34 → 
  y = 22 → 
  y - x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2736_273683


namespace NUMINAMATH_CALUDE_calvins_weight_loss_l2736_273609

/-- Calvin's weight loss problem -/
theorem calvins_weight_loss 
  (initial_weight : ℕ) 
  (weight_loss_per_month : ℕ) 
  (months : ℕ) 
  (h1 : initial_weight = 250)
  (h2 : weight_loss_per_month = 8)
  (h3 : months = 12) :
  initial_weight - (weight_loss_per_month * months) = 154 :=
by sorry

end NUMINAMATH_CALUDE_calvins_weight_loss_l2736_273609


namespace NUMINAMATH_CALUDE_soccer_ball_price_proof_l2736_273695

/-- The unit price of B type soccer balls -/
def unit_price_B : ℝ := 60

/-- The unit price of A type soccer balls -/
def unit_price_A : ℝ := 2.5 * unit_price_B

/-- The total cost of A type soccer balls -/
def total_cost_A : ℝ := 7500

/-- The total cost of B type soccer balls -/
def total_cost_B : ℝ := 4800

/-- The quantity difference between B and A type soccer balls -/
def quantity_difference : ℕ := 30

theorem soccer_ball_price_proof :
  (total_cost_A / unit_price_A) + quantity_difference = total_cost_B / unit_price_B :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_price_proof_l2736_273695


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2736_273657

/-- The minimum value of a quadratic function -/
theorem quadratic_minimum_value 
  (p q r : ℝ) 
  (h1 : p > 0) 
  (h2 : q^2 - 4*p*r < 0) : 
  ∃ (x : ℝ), ∀ (y : ℝ), p*y^2 + q*y + r ≥ (4*p*r - q^2) / (4*p) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2736_273657


namespace NUMINAMATH_CALUDE_probability_non_perimeter_square_10x10_l2736_273626

/-- Represents a square chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Calculates the total number of squares on the chessboard --/
def total_squares (board : Chessboard) : ℕ :=
  board.size * board.size

/-- Calculates the number of squares on the perimeter of the chessboard --/
def perimeter_squares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of squares not on the perimeter of the chessboard --/
def non_perimeter_squares (board : Chessboard) : ℕ :=
  total_squares board - perimeter_squares board

/-- Theorem stating the probability of selecting a non-perimeter square on a 10x10 chessboard --/
theorem probability_non_perimeter_square_10x10 :
  let board : Chessboard := ⟨10⟩
  (non_perimeter_squares board : ℚ) / (total_squares board) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_perimeter_square_10x10_l2736_273626


namespace NUMINAMATH_CALUDE_only_25_is_five_times_last_digit_l2736_273675

theorem only_25_is_five_times_last_digit :
  ∀ n : ℕ, (n > 5 * (n % 10)) ↔ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_only_25_is_five_times_last_digit_l2736_273675


namespace NUMINAMATH_CALUDE_weight_replacement_l2736_273660

theorem weight_replacement (n : ℕ) (old_weight new_weight avg_increase : ℝ) :
  n = 8 ∧
  new_weight = 65 ∧
  avg_increase = 2.5 →
  old_weight = new_weight - n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l2736_273660


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2736_273668

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2736_273668


namespace NUMINAMATH_CALUDE_remainder_sum_of_three_l2736_273666

theorem remainder_sum_of_three (a b c : ℕ) :
  a % 14 = 5 → b % 14 = 5 → c % 14 = 5 → (a + b + c) % 14 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_three_l2736_273666


namespace NUMINAMATH_CALUDE_problem_solution_l2736_273690

theorem problem_solution (x y z : ℝ) 
  (h1 : |x| + x + y = 12)
  (h2 : x + |y| - y = 10)
  (h3 : x - y + z = 5) :
  x + y + z = 9/5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2736_273690


namespace NUMINAMATH_CALUDE_undefined_fraction_l2736_273654

theorem undefined_fraction (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b + 3) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 := by
sorry

end NUMINAMATH_CALUDE_undefined_fraction_l2736_273654


namespace NUMINAMATH_CALUDE_problem_1_l2736_273632

theorem problem_1 : 2 * Real.sqrt 28 + 7 * Real.sqrt 7 - Real.sqrt 7 * Real.sqrt (4/7) = 11 * Real.sqrt 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2736_273632


namespace NUMINAMATH_CALUDE_tan_difference_l2736_273644

theorem tan_difference (α β : Real) (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) :
  Real.tan (α - β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l2736_273644


namespace NUMINAMATH_CALUDE_remainder_1493827_div_4_l2736_273645

theorem remainder_1493827_div_4 : 1493827 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1493827_div_4_l2736_273645


namespace NUMINAMATH_CALUDE_always_true_inequality_l2736_273671

theorem always_true_inequality (x : ℝ) : x + 2 < x + 3 := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequality_l2736_273671


namespace NUMINAMATH_CALUDE_investment_growth_equation_l2736_273689

/-- Represents the average growth rate equation for a two-year investment period -/
theorem investment_growth_equation (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 20000 →
  final_investment = 25000 →
  20 * (1 + x)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_equation_l2736_273689


namespace NUMINAMATH_CALUDE_siena_bookmarks_theorem_l2736_273677

/-- The number of pages Siena bookmarks every day -/
def pages_per_day : ℕ := 30

/-- The number of pages Siena has at the start of March -/
def initial_pages : ℕ := 400

/-- The number of pages Siena will have at the end of March -/
def final_pages : ℕ := 1330

/-- The number of days in March -/
def days_in_march : ℕ := 31

theorem siena_bookmarks_theorem :
  initial_pages + pages_per_day * days_in_march = final_pages :=
by sorry

end NUMINAMATH_CALUDE_siena_bookmarks_theorem_l2736_273677


namespace NUMINAMATH_CALUDE_xy_term_vanishes_l2736_273649

/-- The polynomial in question -/
def polynomial (k x y : ℝ) : ℝ := x^2 + (k-1)*x*y - 3*y^2 - 2*x*y - 5

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (k : ℝ) : ℝ := k - 3

theorem xy_term_vanishes (k : ℝ) :
  xy_coefficient k = 0 ↔ k = 3 := by sorry

end NUMINAMATH_CALUDE_xy_term_vanishes_l2736_273649


namespace NUMINAMATH_CALUDE_joes_total_weight_l2736_273624

/-- Proves that the total weight of Joe's two lifts is 1800 pounds given the conditions -/
theorem joes_total_weight (first_lift second_lift : ℕ) : 
  first_lift = 700 ∧ 
  2 * first_lift = second_lift + 300 → 
  first_lift + second_lift = 1800 := by
sorry

end NUMINAMATH_CALUDE_joes_total_weight_l2736_273624


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2736_273678

theorem least_number_with_remainder (n : ℕ) : n = 256 ↔ 
  (∀ m, m < n → ¬(m % 7 = 4 ∧ m % 9 = 4 ∧ m % 12 = 4 ∧ m % 18 = 4)) ∧
  n % 7 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2736_273678


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2736_273629

theorem quadratic_inequality_always_nonnegative : ∀ x : ℝ, x^2 - x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2736_273629


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_16385_l2736_273670

/-- The number we're analyzing -/
def n : ℕ := 16385

/-- Function to get the greatest prime divisor of a natural number -/
def greatest_prime_divisor (m : ℕ) : ℕ :=
  sorry

/-- Function to sum the digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the sum of digits of the greatest prime divisor of 16385 is 19 -/
theorem sum_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor n) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_16385_l2736_273670


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2736_273662

theorem angle_measure_proof : 
  ∀ x : ℝ, 
    (90 - x = (1/7) * x + 26) → 
    x = 56 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2736_273662


namespace NUMINAMATH_CALUDE_selection_methods_count_l2736_273643

def num_type_a : ℕ := 3
def num_type_b : ℕ := 4
def total_selected : ℕ := 3

theorem selection_methods_count :
  (Finset.sum (Finset.range (total_selected + 1)) (λ k =>
    if k ≥ 1 ∧ (total_selected - k) ≥ 1 then
      (Nat.choose num_type_a k) * (Nat.choose num_type_b (total_selected - k))
    else
      0
  )) = 30 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l2736_273643


namespace NUMINAMATH_CALUDE_divisibility_property_l2736_273612

theorem divisibility_property (n : ℕ) : 
  (n - 1) ∣ (n^n - 7*n + 5*n^2024 + 3*n^2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2736_273612


namespace NUMINAMATH_CALUDE_parabola_distance_l2736_273669

theorem parabola_distance : ∀ (x_p x_q : ℝ),
  (x_p^2 - 2*x_p - 8 = 8) →
  (x_q^2 - 2*x_q - 8 = -4) →
  (∀ x, x^2 - 2*x - 8 = -4 → |x - x_p| ≥ |x_q - x_p|) →
  |x_q - x_p| = |Real.sqrt 17 - Real.sqrt 5| :=
by sorry

end NUMINAMATH_CALUDE_parabola_distance_l2736_273669
