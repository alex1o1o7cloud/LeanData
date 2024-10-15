import Mathlib

namespace NUMINAMATH_CALUDE_medal_award_combinations_l4099_409906

/-- The number of sprinters --/
def total_sprinters : ℕ := 10

/-- The number of American sprinters --/
def american_sprinters : ℕ := 4

/-- The number of non-American sprinters --/
def non_american_sprinters : ℕ := total_sprinters - american_sprinters

/-- The number of medals to be awarded --/
def medals : ℕ := 4

/-- The maximum number of Americans that can win medals --/
def max_american_winners : ℕ := 2

/-- The function to calculate the number of ways medals can be awarded --/
def ways_to_award_medals : ℕ := sorry

/-- Theorem stating that the number of ways to award medals is 6600 --/
theorem medal_award_combinations : ways_to_award_medals = 6600 := by sorry

end NUMINAMATH_CALUDE_medal_award_combinations_l4099_409906


namespace NUMINAMATH_CALUDE_gold_coins_distribution_l4099_409985

theorem gold_coins_distribution (x y : ℕ) (h : x * x - y * y = 81 * (x - y)) : x + y = 81 := by
  sorry

end NUMINAMATH_CALUDE_gold_coins_distribution_l4099_409985


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_roots_l4099_409934

def fibonacci_like_sequence (F : ℕ → ℝ) : Prop :=
  F 0 = 2 ∧ F 1 = 3 ∧ ∀ n, F (n + 1) * F (n - 1) - F n ^ 2 = (-1) ^ n * 2

def has_exponential_form (F : ℕ → ℝ) (r₁ r₂ : ℝ) : Prop :=
  ∃ a b : ℝ, ∀ n, F n = a * r₁ ^ n + b * r₂ ^ n

theorem fibonacci_like_sequence_roots 
  (F : ℕ → ℝ) (r₁ r₂ : ℝ) 
  (h₁ : fibonacci_like_sequence F) 
  (h₂ : has_exponential_form F r₁ r₂) : 
  |r₁ - r₂| = Real.sqrt 17 / 2 := by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_roots_l4099_409934


namespace NUMINAMATH_CALUDE_modulus_two_plus_i_sixth_l4099_409905

/-- The modulus of (2 + i)^6 is equal to 125 -/
theorem modulus_two_plus_i_sixth : Complex.abs ((2 : ℂ) + Complex.I) ^ 6 = 125 := by
  sorry

end NUMINAMATH_CALUDE_modulus_two_plus_i_sixth_l4099_409905


namespace NUMINAMATH_CALUDE_estimate_total_children_l4099_409924

theorem estimate_total_children (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0) (h4 : n ≤ m) (h5 : n ≤ k) :
  ∃ (total : ℚ), total = k * (m / n) :=
sorry

end NUMINAMATH_CALUDE_estimate_total_children_l4099_409924


namespace NUMINAMATH_CALUDE_billboard_count_l4099_409946

theorem billboard_count (B : ℕ) : 
  (B + 20 + 23) / 3 = 20 → B = 17 := by
  sorry

end NUMINAMATH_CALUDE_billboard_count_l4099_409946


namespace NUMINAMATH_CALUDE_exhibition_ticket_sales_l4099_409909

/-- Calculates the total worth of tickets sold over a period of days -/
def totalWorth (averageTicketsPerDay : ℕ) (numDays : ℕ) (ticketPrice : ℕ) : ℕ :=
  averageTicketsPerDay * numDays * ticketPrice

theorem exhibition_ticket_sales :
  let averageTicketsPerDay : ℕ := 80
  let numDays : ℕ := 3
  let ticketPrice : ℕ := 4
  totalWorth averageTicketsPerDay numDays ticketPrice = 960 := by
sorry

end NUMINAMATH_CALUDE_exhibition_ticket_sales_l4099_409909


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4099_409992

theorem min_value_quadratic (x y : ℝ) : 2 * x^2 + 3 * y^2 - 8 * x + 6 * y + 25 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4099_409992


namespace NUMINAMATH_CALUDE_trig_identity_proof_l4099_409960

/-- Proves that sin 42° * cos 18° - cos 138° * cos 72° = √3/2 -/
theorem trig_identity_proof : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l4099_409960


namespace NUMINAMATH_CALUDE_yangmei_sales_l4099_409925

/-- Yangmei sales problem -/
theorem yangmei_sales (total_weight : ℕ) (round_weight round_price square_weight square_price : ℕ) 
  (h_total : total_weight = 1000)
  (h_round : round_weight = 8 ∧ round_price = 160)
  (h_square : square_weight = 18 ∧ square_price = 270) :
  (∃ a : ℕ, a * round_price + a * square_price = 8600 → a = 20) ∧
  (∃ x y : ℕ, x * round_price + y * square_price = 16760 ∧ 
              x * round_weight + y * square_weight = total_weight →
              x = 44 ∧ y = 36) ∧
  (∃ b : ℕ, b > 0 ∧ 
            (∃ m n : ℕ, (m + b) * round_weight + n * square_weight = total_weight ∧
                        m * round_price + n * square_price = 16760) →
            b = 9 ∨ b = 18) := by
  sorry

end NUMINAMATH_CALUDE_yangmei_sales_l4099_409925


namespace NUMINAMATH_CALUDE_projection_magnitude_l4099_409930

def vector_a : ℝ × ℝ := (7, -4)
def vector_b : ℝ × ℝ := (-8, 6)

theorem projection_magnitude :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b := Real.sqrt (vector_b.1^2 + vector_b.2^2)
  let projection := dot_product / magnitude_b
  |projection| = 8 := by sorry

end NUMINAMATH_CALUDE_projection_magnitude_l4099_409930


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_one_l4099_409951

theorem fraction_meaningful_iff_not_one (x : ℝ) :
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_one_l4099_409951


namespace NUMINAMATH_CALUDE_sum_of_specific_triangles_l4099_409953

/-- Triangle operation that takes three integers and returns their sum minus the last -/
def triangle_op (a b c : ℤ) : ℤ := a + b - c

/-- The sum of two triangle operations -/
def sum_of_triangles (a₁ b₁ c₁ a₂ b₂ c₂ : ℤ) : ℤ :=
  triangle_op a₁ b₁ c₁ + triangle_op a₂ b₂ c₂

/-- Theorem stating that the sum of triangle operations (1,3,4) and (2,5,6) is 1 -/
theorem sum_of_specific_triangles :
  sum_of_triangles 1 3 4 2 5 6 = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_triangles_l4099_409953


namespace NUMINAMATH_CALUDE_exists_marked_points_with_distance_l4099_409958

/-- Represents a marked point on the segment -/
structure MarkedPoint where
  position : ℚ
  deriving Repr

/-- The process of marking points on a segment of length 3^n -/
def markPoints (n : ℕ) : List MarkedPoint :=
  sorry

/-- Theorem stating the existence of two marked points with distance k -/
theorem exists_marked_points_with_distance (n : ℕ) (k : ℕ) 
  (h : 1 ≤ k ∧ k ≤ 3^n) : 
  ∃ (p q : MarkedPoint), p ∈ markPoints n ∧ q ∈ markPoints n ∧ 
    |p.position - q.position| = k :=
  sorry

end NUMINAMATH_CALUDE_exists_marked_points_with_distance_l4099_409958


namespace NUMINAMATH_CALUDE_extra_money_is_seven_l4099_409915

/-- The amount of extra money given by an appreciative customer to Hillary at a flea market. -/
def extra_money (price_per_craft : ℕ) (crafts_sold : ℕ) (deposited : ℕ) (remaining : ℕ) : ℕ :=
  (deposited + remaining) - (price_per_craft * crafts_sold)

/-- Theorem stating that the extra money given to Hillary is 7 dollars. -/
theorem extra_money_is_seven :
  extra_money 12 3 18 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_extra_money_is_seven_l4099_409915


namespace NUMINAMATH_CALUDE_freshman_percentage_l4099_409967

theorem freshman_percentage (total_students : ℝ) (freshman : ℝ) 
  (h1 : freshman > 0) 
  (h2 : total_students > 0) 
  (h3 : freshman * 0.4 * 0.2 = total_students * 0.048) : 
  freshman / total_students = 0.6 := by
sorry

end NUMINAMATH_CALUDE_freshman_percentage_l4099_409967


namespace NUMINAMATH_CALUDE_first_month_sale_is_800_l4099_409912

/-- Calculates the first month's sale given the sales of the following months and the average -/
def first_month_sale (sales : List ℕ) (average : ℕ) : ℕ :=
  6 * average - sales.sum

/-- Proves that the first month's sale is 800 given the problem conditions -/
theorem first_month_sale_is_800 :
  let sales : List ℕ := [900, 1000, 700, 800, 900]
  let average : ℕ := 850
  first_month_sale sales average = 800 := by
    sorry

end NUMINAMATH_CALUDE_first_month_sale_is_800_l4099_409912


namespace NUMINAMATH_CALUDE_deposit_difference_approximately_219_01_l4099_409980

-- Constants
def initial_deposit : ℝ := 10000
def a_interest_rate : ℝ := 0.0288
def b_interest_rate : ℝ := 0.0225
def tax_rate : ℝ := 0.20
def years : ℕ := 5

-- A's total amount after 5 years
def a_total : ℝ := initial_deposit + initial_deposit * a_interest_rate * (1 - tax_rate) * years

-- B's total amount after 5 years (compound interest)
def b_total : ℝ := initial_deposit * (1 + b_interest_rate * (1 - tax_rate)) ^ years

-- Theorem statement
theorem deposit_difference_approximately_219_01 :
  ∃ ε > 0, ε < 0.005 ∧ |a_total - b_total - 219.01| < ε :=
sorry

end NUMINAMATH_CALUDE_deposit_difference_approximately_219_01_l4099_409980


namespace NUMINAMATH_CALUDE_expression_value_l4099_409995

theorem expression_value : -2^4 + 3 * (-1)^6 - (-2)^3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4099_409995


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l4099_409975

theorem joshua_bottle_caps (initial_caps : ℕ) (bought_caps : ℕ) : 
  initial_caps = 40 → bought_caps = 7 → initial_caps + bought_caps = 47 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l4099_409975


namespace NUMINAMATH_CALUDE_find_divisor_l4099_409919

def nearest_number : ℕ := 3108
def original_number : ℕ := 3105

theorem find_divisor : 
  (nearest_number - original_number = 3) →
  (∃ d : ℕ, d > 1 ∧ nearest_number % d = 0 ∧ 
   ∀ n : ℕ, n > original_number ∧ n < nearest_number → n % d ≠ 0) →
  (∃ d : ℕ, d = 3 ∧ nearest_number % d = 0) :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l4099_409919


namespace NUMINAMATH_CALUDE_scientific_notation_of_passenger_trips_l4099_409973

theorem scientific_notation_of_passenger_trips :
  let trips : ℝ := 56.99 * 1000000
  trips = 5.699 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_passenger_trips_l4099_409973


namespace NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l4099_409923

theorem square_sum_from_sum_and_product (x y : ℚ) 
  (h1 : x + y = 5/6) (h2 : x * y = 7/36) : x^2 + y^2 = 11/36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l4099_409923


namespace NUMINAMATH_CALUDE_cake_area_l4099_409920

/-- Represents the size of a piece of cake in inches -/
def piece_size : ℝ := 2

/-- Represents the number of pieces that can be cut from the cake -/
def num_pieces : ℕ := 100

/-- Calculates the area of a single piece of cake -/
def piece_area : ℝ := piece_size * piece_size

/-- Theorem: The total area of the cake is 400 square inches -/
theorem cake_area : piece_area * num_pieces = 400 := by
  sorry

end NUMINAMATH_CALUDE_cake_area_l4099_409920


namespace NUMINAMATH_CALUDE_maze_max_candies_l4099_409955

/-- Represents a station in the maze --/
structure Station where
  candies : ℕ  -- Number of candies given at this station
  entries : ℕ  -- Number of times Jirka can enter this station

/-- The maze configuration --/
def Maze : List Station :=
  [⟨5, 3⟩, ⟨3, 2⟩, ⟨3, 2⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩]

/-- The maximum number of candies Jirka can collect --/
def maxCandies : ℕ := 30

theorem maze_max_candies :
  (Maze.map (fun s => s.candies * s.entries)).sum = maxCandies := by
  sorry


end NUMINAMATH_CALUDE_maze_max_candies_l4099_409955


namespace NUMINAMATH_CALUDE_sqrt_sum_2160_l4099_409962

theorem sqrt_sum_2160 (a b : ℕ+) : 
  a < b → 
  (a.val : ℝ).sqrt + (b.val : ℝ).sqrt = Real.sqrt 2160 → 
  a ∈ ({15, 60, 135, 240, 375} : Set ℕ+) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_2160_l4099_409962


namespace NUMINAMATH_CALUDE_problem_statement_l4099_409937

theorem problem_statement (a b : ℝ) (h : 2 * a - b + 3 = 0) :
  2 * (2 * a + b) - 4 * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4099_409937


namespace NUMINAMATH_CALUDE_length_of_AB_l4099_409907

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def C₂ (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24 / 7 := by sorry

end NUMINAMATH_CALUDE_length_of_AB_l4099_409907


namespace NUMINAMATH_CALUDE_parallelogram_base_l4099_409998

/-- 
Given a parallelogram with area 612 square centimeters and height 18 cm, 
prove that its base is 34 cm.
-/
theorem parallelogram_base (area height : ℝ) (h1 : area = 612) (h2 : height = 18) :
  area / height = 34 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l4099_409998


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l4099_409994

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 11 → difference = 3 → friend_cost = total / 2 + difference / 2 → friend_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l4099_409994


namespace NUMINAMATH_CALUDE_inequality_solution_l4099_409952

/-- The solution set of the inequality x^2 - (a + a^2)x + a^3 > 0 for any real number a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 ∨ a > 1 then {x | x > a^2 ∨ x < a}
  else if a = 0 then {x | x ≠ 0}
  else if 0 < a ∧ a < 1 then {x | x > a ∨ x < a^2}
  else {x | x ≠ 1}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + a^2) * x + a^3 > 0} = solution_set a :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4099_409952


namespace NUMINAMATH_CALUDE_scooter_initial_value_l4099_409957

/-- The depreciation rate of the scooter's value each year -/
def depreciation_rate : ℚ := 3/4

/-- The number of years of depreciation -/
def years : ℕ := 4

/-- The value of the scooter after 4 years in rupees -/
def final_value : ℚ := 12656.25

/-- The initial value of the scooter in rupees -/
def initial_value : ℚ := 30000

/-- Theorem stating that given the depreciation rate, number of years, and final value,
    the initial value of the scooter can be calculated -/
theorem scooter_initial_value :
  initial_value * depreciation_rate ^ years = final_value := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l4099_409957


namespace NUMINAMATH_CALUDE_chord_length_squared_l4099_409903

/-- Given three circles with radii 6, 9, and 15, where the circles with radii 6 and 9
    are externally tangent to each other and internally tangent to the circle with radius 15,
    this theorem states that the square of the length of the chord of the circle with radius 15,
    which is a common external tangent to the other two circles, is equal to 692.64. -/
theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 9) (h₃ : R = 15)
  (h₄ : r₁ + r₂ = R - r₁ - r₂) : -- Condition for external tangency of smaller circles and internal tangency with larger circle
  (2 * R * ((r₁ * r₂) / (r₁ + r₂)))^2 = 692.64 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l4099_409903


namespace NUMINAMATH_CALUDE_rational_sqrt_equation_l4099_409966

theorem rational_sqrt_equation (a b : ℚ) : 
  a - b * Real.sqrt 2 = (1 + Real.sqrt 2)^2 → a = 3 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_equation_l4099_409966


namespace NUMINAMATH_CALUDE_population_increase_rate_l4099_409911

theorem population_increase_rate 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (increase_rate : ℚ) : 
  initial_population = 240 →
  final_population = 264 →
  increase_rate = (final_population - initial_population : ℚ) / initial_population * 100 →
  increase_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_l4099_409911


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l4099_409929

theorem inscribed_hexagon_area (circle_area : ℝ) (hexagon_area : ℝ) :
  circle_area = 100 * Real.pi →
  hexagon_area = 6 * (((Real.sqrt (circle_area / Real.pi))^2 * Real.sqrt 3) / 4) →
  hexagon_area = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l4099_409929


namespace NUMINAMATH_CALUDE_negative_sqrt_two_squared_equals_two_l4099_409989

theorem negative_sqrt_two_squared_equals_two :
  (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_two_squared_equals_two_l4099_409989


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4099_409961

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^3 - x + 4) % (x - 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4099_409961


namespace NUMINAMATH_CALUDE_algorithm_properties_l4099_409974

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the properties of algorithms
def yields_definite_result (a : Algorithm) : Prop := sorry
def multiple_algorithms_exist : Prop := sorry
def terminates_in_finite_steps (a : Algorithm) : Prop := sorry

-- Theorem stating the correct properties of algorithms
theorem algorithm_properties :
  (∀ a : Algorithm, yields_definite_result a) ∧
  multiple_algorithms_exist ∧
  (∀ a : Algorithm, terminates_in_finite_steps a) := by
  sorry

end NUMINAMATH_CALUDE_algorithm_properties_l4099_409974


namespace NUMINAMATH_CALUDE_max_distance_l4099_409991

theorem max_distance (x y z w v : ℝ) 
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  ∃ (x' y' z' w' v' : ℝ), 
    |x' - y'| = 1 ∧ 
    |y' - z'| = 2 ∧ 
    |z' - w'| = 3 ∧ 
    |w' - v'| = 5 ∧ 
    |x' - v'| = 11 ∧
    ∀ (a b c d e : ℝ), 
      |a - b| = 1 → 
      |b - c| = 2 → 
      |c - d| = 3 → 
      |d - e| = 5 → 
      |a - e| ≤ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_l4099_409991


namespace NUMINAMATH_CALUDE_vector_product_l4099_409983

theorem vector_product (m n : ℝ) : 
  let a : Fin 2 → ℝ := ![m, n]
  let b : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), a = k • b) → 
  (‖a‖ = 2 * ‖b‖) →
  m * n = -8 := by sorry

end NUMINAMATH_CALUDE_vector_product_l4099_409983


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4099_409964

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4099_409964


namespace NUMINAMATH_CALUDE_max_overlap_theorem_l4099_409996

/-- The area of the equilateral triangle -/
def triangle_area : ℝ := 2019

/-- The maximum overlap area when folding the triangle -/
def max_overlap_area : ℝ := 673

/-- The fold line is parallel to one of the triangle's sides -/
axiom fold_parallel : True

theorem max_overlap_theorem :
  ∀ (overlap_area : ℝ),
  overlap_area ≤ max_overlap_area :=
sorry

end NUMINAMATH_CALUDE_max_overlap_theorem_l4099_409996


namespace NUMINAMATH_CALUDE_soccer_team_games_l4099_409916

theorem soccer_team_games (win lose tie rain higher : ℚ) 
  (ratio : win = 5.5 ∧ lose = 4.5 ∧ tie = 2.5 ∧ rain = 1 ∧ higher = 3.5)
  (lost_games : ℚ) (h_lost : lost_games = 13.5) :
  (win + lose + tie + rain + higher) * (lost_games / lose) = 51 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_l4099_409916


namespace NUMINAMATH_CALUDE_basketball_team_selection_l4099_409942

def team_size : ℕ := 16
def lineup_size : ℕ := 5
def num_twins : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose (team_size - num_twins) lineup_size) +
  (num_twins * Nat.choose (team_size - num_twins) (lineup_size - 1)) +
  (Nat.choose (team_size - num_twins) (lineup_size - num_twins)) = 4368 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l4099_409942


namespace NUMINAMATH_CALUDE_milk_powder_cost_july_l4099_409963

/-- The cost of milk powder and coffee in July -/
def july_cost (june_cost : ℝ) : ℝ × ℝ :=
  (0.4 * june_cost, 3 * june_cost)

/-- The total cost of the mixture in July -/
def mixture_cost (june_cost : ℝ) : ℝ :=
  1.5 * (july_cost june_cost).1 + 1.5 * (july_cost june_cost).2

theorem milk_powder_cost_july :
  ∃ (june_cost : ℝ),
    june_cost > 0 ∧
    mixture_cost june_cost = 5.1 ∧
    (july_cost june_cost).1 = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_milk_powder_cost_july_l4099_409963


namespace NUMINAMATH_CALUDE_unique_four_digit_cube_divisible_by_16_and_9_l4099_409988

theorem unique_four_digit_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∃ m : ℕ, n = m^3) ∧ 
  n % 16 = 0 ∧ n % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_cube_divisible_by_16_and_9_l4099_409988


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l4099_409938

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (5/3, 7/3)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 10 * x - 5 * y = 5

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 18

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l4099_409938


namespace NUMINAMATH_CALUDE_max_value_expression_l4099_409941

theorem max_value_expression (x y : ℝ) : 
  (Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
  (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l4099_409941


namespace NUMINAMATH_CALUDE_final_brownie_count_l4099_409997

def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def additional_brownies : ℕ := 24

theorem final_brownie_count :
  initial_brownies - father_ate - mooney_ate + additional_brownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_final_brownie_count_l4099_409997


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4099_409949

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + 1 < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4099_409949


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l4099_409978

/-- Given three consecutive even numbers whose sum is 246, the first number is 80 -/
theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ a + b + c = 246 ∧ Even a ∧ Even b ∧ Even c) → 
  n = 80 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l4099_409978


namespace NUMINAMATH_CALUDE_sequence_sum_times_three_l4099_409904

theorem sequence_sum_times_three (seq : List Nat) : 
  seq = [82, 84, 86, 88, 90, 92, 94, 96, 98, 100] →
  3 * (seq.sum) = 2730 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_times_three_l4099_409904


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4099_409928

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = (n : ℚ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) →
  (∀ n : ℕ, n > 0 → T n = (n : ℚ) / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) →
  (∀ n : ℕ, n > 0 → S n / T n = (n : ℚ) / (2 * n + 1)) →
  (a 5 / b 5 = 9 / 19) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4099_409928


namespace NUMINAMATH_CALUDE_graph_equation_two_lines_l4099_409935

theorem graph_equation_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_equation_two_lines_l4099_409935


namespace NUMINAMATH_CALUDE_work_completion_time_l4099_409914

/-- The time taken by A, B, and C to complete a work given their pairwise completion times -/
theorem work_completion_time 
  (time_AB : ℝ) 
  (time_BC : ℝ) 
  (time_AC : ℝ) 
  (h_AB : time_AB = 8) 
  (h_BC : time_BC = 12) 
  (h_AC : time_AC = 8) : 
  (1 / (1 / time_AB + 1 / time_BC + 1 / time_AC)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4099_409914


namespace NUMINAMATH_CALUDE_no_base_for_131_square_l4099_409917

theorem no_base_for_131_square (b : ℕ) : b > 3 → ¬∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_131_square_l4099_409917


namespace NUMINAMATH_CALUDE_no_fourfold_digit_move_l4099_409945

theorem no_fourfold_digit_move :
  ∀ (N : ℕ), ∀ (a : ℕ), ∀ (n : ℕ), ∀ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) →
    (x < 10^n) →
    (N = a * 10^n + x) →
    (10 * x + a ≠ 4 * N) :=
by sorry

end NUMINAMATH_CALUDE_no_fourfold_digit_move_l4099_409945


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l4099_409969

theorem solve_quadratic_equation (x : ℝ) :
  (1/3 - x)^2 = 4 ↔ x = -5/3 ∨ x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l4099_409969


namespace NUMINAMATH_CALUDE_coal_burn_duration_l4099_409977

/-- Given a factory with 300 tons of coal, this theorem establishes the relationship
    between the number of days the coal can burn and the average daily consumption. -/
theorem coal_burn_duration (x : ℝ) (y : ℝ) (h : x > 0) :
  y = 300 / x ↔ y * x = 300 :=
by sorry

end NUMINAMATH_CALUDE_coal_burn_duration_l4099_409977


namespace NUMINAMATH_CALUDE_sector_angle_l4099_409950

/-- Given a circular sector with circumference 4 and area 1, prove that its central angle is 2 radians -/
theorem sector_angle (r : ℝ) (l : ℝ) (α : ℝ) 
  (h_circumference : 2 * r + l = 4)
  (h_area : (1 / 2) * l * r = 1) :
  α = 2 :=
sorry

end NUMINAMATH_CALUDE_sector_angle_l4099_409950


namespace NUMINAMATH_CALUDE_simplify_expression_l4099_409954

theorem simplify_expression (x : ℝ) (h : x = Real.tan (60 * π / 180)) :
  (x + 1 - 8 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - x)) * (3 - x) = -3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4099_409954


namespace NUMINAMATH_CALUDE_product_of_max_min_sum_l4099_409913

theorem product_of_max_min_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * 2^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0 → 
  ∃ (min_sum max_sum : ℝ), 
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * 2^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 → 
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_max_min_sum_l4099_409913


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l4099_409926

theorem discount_percentage_proof (num_people : ℕ) (savings_per_person : ℝ) (final_price : ℝ) :
  num_people = 3 →
  savings_per_person = 4 →
  final_price = 48 →
  let total_savings := num_people * savings_per_person
  let original_price := final_price + total_savings
  let discount_percentage := (total_savings / original_price) * 100
  discount_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l4099_409926


namespace NUMINAMATH_CALUDE_solution_pairs_l4099_409902

theorem solution_pairs (x y : ℝ) : 
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l4099_409902


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l4099_409900

theorem consecutive_odd_numbers_sum (n1 n2 n3 : ℕ) : 
  (n1 % 2 = 1) →  -- n1 is odd
  (n2 = n1 + 2) →  -- n2 is the next consecutive odd number
  (n3 = n2 + 2) →  -- n3 is the next consecutive odd number after n2
  (n3 = 27) →      -- the largest number is 27
  (n1 + n2 + n3 ≠ 72) :=  -- their sum cannot be 72
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l4099_409900


namespace NUMINAMATH_CALUDE_waiter_new_customers_l4099_409971

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (left_customers : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 14) 
  (h2 : left_customers = 3) 
  (h3 : final_customers = 50) : 
  final_customers - (initial_customers - left_customers) = 39 := by
  sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l4099_409971


namespace NUMINAMATH_CALUDE_train_passing_time_l4099_409981

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 80 → 
  train_speed_kmph = 36 → 
  (train_length / (train_speed_kmph * 1000 / 3600)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l4099_409981


namespace NUMINAMATH_CALUDE_seashell_collection_l4099_409965

/-- The number of seashells collected by Stefan, Vail, and Aiguo -/
theorem seashell_collection (stefan vail aiguo : ℕ) 
  (h1 : stefan = vail + 16)
  (h2 : vail = aiguo - 5)
  (h3 : aiguo = 20) :
  stefan + vail + aiguo = 66 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l4099_409965


namespace NUMINAMATH_CALUDE_ellipse_focal_length_implies_m_8_l4099_409993

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

-- Define the condition for major axis along y-axis
def major_axis_y (m : ℝ) : Prop :=
  m - 2 > 10 - m

-- Define the focal length
def focal_length (m : ℝ) : ℝ :=
  4

-- Theorem statement
theorem ellipse_focal_length_implies_m_8 :
  ∀ m : ℝ,
  (∀ x y : ℝ, ellipse_equation x y m) →
  major_axis_y m →
  focal_length m = 4 →
  m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_implies_m_8_l4099_409993


namespace NUMINAMATH_CALUDE_triangle_property_triangle_area_l4099_409948

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_property (t : Triangle) 
  (h : t.c - t.a * Real.cos t.B = (Real.sqrt 2 / 2) * t.b) : 
  t.A = π / 4 := by sorry

theorem triangle_area (t : Triangle) 
  (h1 : t.c - t.a * Real.cos t.B = (Real.sqrt 2 / 2) * t.b)
  (h2 : t.c = 4 * Real.sqrt 2)
  (h3 : Real.cos t.B = 7 * Real.sqrt 2 / 10) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_property_triangle_area_l4099_409948


namespace NUMINAMATH_CALUDE_min_value_theorem_l4099_409922

def f (x : ℝ) := 45 * |2*x - 1|

def g (x : ℝ) := f x + f (x - 1)

theorem min_value_theorem (a m n : ℝ) :
  (∀ x, g x ≥ a) →
  m > 0 →
  n > 0 →
  m + n = a →
  (∀ p q, p > 0 → q > 0 → p + q = a → 4/m + 1/n ≤ 4/p + 1/q) →
  4/m + 1/n = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4099_409922


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l4099_409947

theorem sandy_shopping_money (initial_amount : ℝ) : 
  (initial_amount * 0.7 = 210) → initial_amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l4099_409947


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l4099_409918

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (interest : ℝ)
  (h1 : principal = 1100)
  (h2 : time = 8)
  (h3 : interest = principal - 572) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l4099_409918


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l4099_409910

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l4099_409910


namespace NUMINAMATH_CALUDE_orange_stack_count_l4099_409933

/-- Calculates the number of oranges in a single layer of the pyramid -/
def layerOranges (baseWidth : ℕ) (baseLength : ℕ) (layer : ℕ) : ℕ :=
  (baseWidth - layer + 1) * (baseLength - layer + 1)

/-- Calculates the total number of oranges in the pyramid stack -/
def totalOranges (baseWidth : ℕ) (baseLength : ℕ) : ℕ :=
  let numLayers := min baseWidth baseLength
  (List.range numLayers).foldl (fun acc i => acc + layerOranges baseWidth baseLength i) 0

/-- Theorem stating that a pyramid-like stack of oranges with a 6x9 base contains 154 oranges -/
theorem orange_stack_count : totalOranges 6 9 = 154 := by
  sorry

end NUMINAMATH_CALUDE_orange_stack_count_l4099_409933


namespace NUMINAMATH_CALUDE_remainder_theorem_a_value_l4099_409939

/-- The polynomial function f(x) = x^6 - 8x^3 + 6 -/
def f (x : ℝ) : ℝ := x^6 - 8*x^3 + 6

/-- The remainder function R(x) = 7x - 8 -/
def R (x : ℝ) : ℝ := 7*x - 8

/-- Theorem stating that R(x) is the remainder when f(x) is divided by (x-1)(x-2) -/
theorem remainder_theorem :
  ∀ x : ℝ, ∃ q : ℝ → ℝ, f x = ((x - 1) * (x - 2)) * (q x) + R x :=
by
  sorry

/-- Corollary: The value of a in the remainder 7x - a is 8 -/
theorem a_value : ∃ a : ℝ, a = 8 ∧ 
  (∀ x : ℝ, ∃ q : ℝ → ℝ, f x = ((x - 1) * (x - 2)) * (q x) + (7*x - a)) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_a_value_l4099_409939


namespace NUMINAMATH_CALUDE_equation_solutions_parabola_properties_l4099_409936

-- Part 1: Equation solving
def equation (x : ℝ) : Prop := (x - 9)^2 = 2 * (x - 9)

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 9 ∧ x₂ = 11 ∧ equation x₁ ∧ equation x₂ ∧
  ∀ (x : ℝ), equation x → x = x₁ ∨ x = x₂ :=
sorry

-- Part 2: Parabola function
def parabola (x y : ℝ) : Prop := y = -x^2 - 6*x - 7

theorem parabola_properties :
  (parabola (-3) 2) ∧ (parabola (-1) (-2)) ∧
  ∀ (x y : ℝ), y = -(x + 3)^2 + 2 ↔ parabola x y :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_parabola_properties_l4099_409936


namespace NUMINAMATH_CALUDE_flash_interval_l4099_409959

/-- Proves that the time between each flash is 6 seconds, given that a light flashes 450 times in ¾ of an hour. -/
theorem flash_interval (flashes : ℕ) (time : ℚ) (h1 : flashes = 450) (h2 : time = 3/4) :
  (time * 3600) / flashes = 6 := by
  sorry

end NUMINAMATH_CALUDE_flash_interval_l4099_409959


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4099_409908

theorem polynomial_division_remainder
  (dividend : Polynomial ℤ)
  (divisor : Polynomial ℤ)
  (h_dividend : dividend = 3 * X^6 - 2 * X^4 + 5 * X^2 - 9)
  (h_divisor : divisor = X^2 + 3 * X + 2) :
  ∃ (q : Polynomial ℤ), dividend = q * divisor + (-174 * X - 177) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4099_409908


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_value_l4099_409927

theorem unique_solution_implies_a_value (a : ℝ) : 
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) → a = 2019 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_value_l4099_409927


namespace NUMINAMATH_CALUDE_missing_score_proof_l4099_409982

theorem missing_score_proof (known_scores : List ℝ) (mean : ℝ) : 
  known_scores = [81, 73, 86, 73] →
  mean = 79.2 →
  ∃ (missing_score : ℝ), 
    (List.sum known_scores + missing_score) / 5 = mean ∧
    missing_score = 83 := by
  sorry

end NUMINAMATH_CALUDE_missing_score_proof_l4099_409982


namespace NUMINAMATH_CALUDE_binomial_1300_2_l4099_409970

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by sorry

end NUMINAMATH_CALUDE_binomial_1300_2_l4099_409970


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l4099_409944

theorem fly_distance_from_ceiling (z : ℝ) : 
  3^2 + 4^2 + z^2 = 6^2 → z = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l4099_409944


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4099_409990

theorem absolute_value_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4099_409990


namespace NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l4099_409984

theorem largest_divisor_of_polynomial (n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (m : ℤ), (m^4 - 5*m^2 + 6) % k = 0) ∧ 
  (∀ (l : ℕ), l > k → ∃ (m : ℤ), (m^4 - 5*m^2 + 6) % l ≠ 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l4099_409984


namespace NUMINAMATH_CALUDE_koi_fish_multiple_l4099_409931

theorem koi_fish_multiple (num_koi : ℕ) (target : ℕ) : 
  num_koi = 39 → target = 64 → 
  ∃ m : ℕ, m * num_koi > target ∧ 
           ∀ k : ℕ, k * num_koi > target → k ≥ m ∧
           m * num_koi = 78 := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_multiple_l4099_409931


namespace NUMINAMATH_CALUDE_finley_age_l4099_409976

/-- Represents the ages of the individuals in the problem -/
structure Ages where
  jill : ℕ
  roger : ℕ
  alex : ℕ
  finley : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.roger = 2 * ages.jill + 5 ∧
  ages.roger + 15 - (ages.jill + 15) = ages.finley - 30 ∧
  ages.jill = 20 ∧
  ages.roger = (ages.jill + ages.alex) / 2 ∧
  ages.alex = 3 * (ages.finley + 10) - 5

/-- The theorem stating Finley's age -/
theorem finley_age (ages : Ages) (h : problem_conditions ages) : ages.finley = 15 := by
  sorry

#check finley_age

end NUMINAMATH_CALUDE_finley_age_l4099_409976


namespace NUMINAMATH_CALUDE_apple_arrangements_l4099_409940

def word : String := "APPLE"

def letter_count : Nat := word.length

def letter_frequencies : List (Char × Nat) := [('A', 1), ('P', 2), ('L', 1), ('E', 1)]

/-- The number of distinct arrangements of the letters in the word "APPLE" -/
def distinct_arrangements : Nat := 60

/-- Theorem stating that the number of distinct arrangements of the letters in "APPLE" is 60 -/
theorem apple_arrangements :
  distinct_arrangements = 60 :=
by sorry

end NUMINAMATH_CALUDE_apple_arrangements_l4099_409940


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4099_409921

theorem hyperbola_equation (a b : ℝ) (h1 : b > 0) (h2 : a > 0) (h3 : ∃ n : ℕ, a = n) 
  (h4 : (a^2 + b^2) / a^2 = 7/4) (h5 : a^2 + b^2 ≤ 20) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((x^2 - 4*y^2/3 = 1) ∨ (x^2/4 - y^2/3 = 1) ∨ (x^2/9 - 4*y^2/27 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4099_409921


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l4099_409979

/-- The number of unique handshakes in a family gathering with twins and triplets -/
theorem family_gathering_handshakes :
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 3
  let twins_per_set : ℕ := 2
  let triplets_per_set : ℕ := 3
  let total_twins : ℕ := twin_sets * twins_per_set
  let total_triplets : ℕ := triplet_sets * triplets_per_set
  let twin_handshakes : ℕ := total_twins * (total_twins - twins_per_set)
  let triplet_handshakes : ℕ := total_triplets * (total_triplets - triplets_per_set)
  let twin_triplet_handshakes : ℕ := total_twins * total_triplets
  let total_handshakes : ℕ := twin_handshakes + triplet_handshakes + twin_triplet_handshakes
  327 = total_handshakes / 2 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l4099_409979


namespace NUMINAMATH_CALUDE_triangle_existence_l4099_409901

theorem triangle_existence (k : ℕ) (a b c : ℝ) 
  (h_k : k ≥ 10) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ a = y + z ∧ b = z + x ∧ c = x + y :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l4099_409901


namespace NUMINAMATH_CALUDE_positive_roots_quadratic_l4099_409968

/-- For a quadratic equation (n-2)x^2 - 2nx + n + 3 = 0, both roots are positive
    if and only if n ∈ (-∞, -3) ∪ (2, 6] -/
theorem positive_roots_quadratic (n : ℝ) : 
  (∀ x : ℝ, (n - 2) * x^2 - 2 * n * x + n + 3 = 0 → x > 0) ↔ 
  (n < -3 ∨ (2 < n ∧ n ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_positive_roots_quadratic_l4099_409968


namespace NUMINAMATH_CALUDE_log_product_equality_l4099_409999

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log y^8) * (Real.log y^3 / Real.log x^7) *
  (Real.log x^4 / Real.log y^5) * (Real.log y^5 / Real.log x^4) *
  (Real.log x^7 / Real.log y^3) * (Real.log y^8 / Real.log x^2) =
  28/3 * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l4099_409999


namespace NUMINAMATH_CALUDE_original_number_of_girls_l4099_409987

theorem original_number_of_girls (b g : ℚ) : 
  b > 0 ∧ g > 0 →  -- Initial numbers are positive
  3 * (g - 20) = b →  -- After 20 girls leave, ratio is 3 boys to 1 girl
  4 * (b - 60) = g - 20 →  -- After 60 boys leave, ratio is 1 boy to 4 girls
  g = 460 / 11 := by
sorry

end NUMINAMATH_CALUDE_original_number_of_girls_l4099_409987


namespace NUMINAMATH_CALUDE_f_equals_g_l4099_409972

def N : Set ℕ := {n : ℕ | n > 0}

theorem f_equals_g
  (f g : ℕ → ℕ)
  (f_onto : ∀ y : ℕ, ∃ x : ℕ, f x = y)
  (g_one_one : ∀ x y : ℕ, g x = g y → x = y)
  (f_ge_g : ∀ n : ℕ, f n ≥ g n)
  : ∀ n : ℕ, f n = g n :=
sorry

end NUMINAMATH_CALUDE_f_equals_g_l4099_409972


namespace NUMINAMATH_CALUDE_find_x_l4099_409943

-- Define the # operation
def sharp (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

-- Theorem statement
theorem find_x : 
  ∃ (x : ℤ), 
    (∀ (p : ℤ), sharp (sharp (sharp p x) x) x = -4) ∧ 
    (sharp (sharp (sharp 18 x) x) x = -4) → 
    x = -21 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l4099_409943


namespace NUMINAMATH_CALUDE_unique_solution_3n_plus_1_equals_a_squared_l4099_409986

theorem unique_solution_3n_plus_1_equals_a_squared :
  ∀ a n : ℕ+, 3^(n : ℕ) + 1 = (a : ℕ)^2 → a = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3n_plus_1_equals_a_squared_l4099_409986


namespace NUMINAMATH_CALUDE_circle_angle_equality_l4099_409932

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the angle between two vectors
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_angle_equality (Γ : Circle) (O A B M N : ℝ × ℝ) 
  (hO : O = Γ.center)
  (hA : PointOnCircle Γ A)
  (hB : PointOnCircle Γ B)
  (hM : PointOnCircle Γ M)
  (hN : PointOnCircle Γ N) :
  angle (M.1 - A.1, M.2 - A.2) (M.1 - B.1, M.2 - B.2) = 
  (angle (O.1 - A.1, O.2 - A.2) (O.1 - B.1, O.2 - B.2)) / 2 ∧
  angle (N.1 - A.1, N.2 - A.2) (N.1 - B.1, N.2 - B.2) = 
  (angle (O.1 - A.1, O.2 - A.2) (O.1 - B.1, O.2 - B.2)) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_angle_equality_l4099_409932


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l4099_409956

theorem reciprocal_of_negative_fraction (n : ℕ) (h : n ≠ 0) :
  ((-1 : ℚ) / n)⁻¹ = -n := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l4099_409956
