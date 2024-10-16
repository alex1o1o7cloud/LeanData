import Mathlib

namespace NUMINAMATH_CALUDE_sphere_volume_l66_6657

theorem sphere_volume (R : ℝ) (x y : ℝ) : 
  R > 0 ∧ 
  x ≠ y ∧
  R^2 = x^2 + 5 ∧ 
  R^2 = y^2 + 8 ∧ 
  |x - y| = 1 →
  (4/3) * Real.pi * R^3 = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_l66_6657


namespace NUMINAMATH_CALUDE_functional_equation_solution_l66_6684

/-- A function f: ℝ⁺* → ℝ⁺* satisfying the functional equation f(x) f(y f(x)) = f(x + y) for all x, y > 0 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f (y * f x) > 0 → f x * f (y * f x) = f (x + y)

/-- The theorem stating that functions satisfying the given functional equation
    are either of the form f(x) = 1/(1 + ax) for some a > 0, or f(x) = 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f →
  (∃ a : ℝ, a > 0 ∧ ∀ x, x > 0 → f x = 1 / (1 + a * x)) ∨
  (∀ x, x > 0 → f x = 1) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l66_6684


namespace NUMINAMATH_CALUDE_square_perimeter_l66_6607

theorem square_perimeter (rectangleA_perimeter : ℝ) (squareB_area_ratio : ℝ) :
  rectangleA_perimeter = 30 →
  squareB_area_ratio = 1/3 →
  ∃ (rectangleA_length rectangleA_width : ℝ),
    rectangleA_length > 0 ∧
    rectangleA_width > 0 ∧
    2 * (rectangleA_length + rectangleA_width) = rectangleA_perimeter ∧
    ∃ (squareB_side : ℝ),
      squareB_side > 0 ∧
      squareB_side^2 = squareB_area_ratio * (rectangleA_length * rectangleA_width) ∧
      4 * squareB_side = 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l66_6607


namespace NUMINAMATH_CALUDE_knight_freedom_guaranteed_l66_6624

/-- Represents a pile of coins -/
structure Pile :=
  (total : ℕ)
  (magical : ℕ)

/-- Represents the state of the coins -/
structure CoinState :=
  (pile1 : Pile)
  (pile2 : Pile)

/-- Checks if the piles have equal magical or ordinary coins -/
def isEqualDistribution (state : CoinState) : Prop :=
  state.pile1.magical = state.pile2.magical ∨ 
  (state.pile1.total - state.pile1.magical) = (state.pile2.total - state.pile2.magical)

/-- Represents a division strategy -/
def DivisionStrategy := ℕ → CoinState

/-- The theorem to be proved -/
theorem knight_freedom_guaranteed :
  ∃ (strategy : DivisionStrategy),
    (∀ (n : ℕ), n ≤ 25 → 
      (strategy n).pile1.total + (strategy n).pile2.total = 100 ∧
      (strategy n).pile1.magical + (strategy n).pile2.magical = 50) →
    ∃ (day : ℕ), day ≤ 25 ∧ isEqualDistribution (strategy day) :=
sorry

end NUMINAMATH_CALUDE_knight_freedom_guaranteed_l66_6624


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l66_6601

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- Predicate to check if a polynomial is nonnegative on [0,1] -/
def IsNonnegativeOn01 (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → P x ≥ 0

/-- Predicate to check if a polynomial is nonnegative on ℝ -/
def IsNonnegativeOnReals (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P x ≥ 0

theorem polynomial_decomposition (P : RealPolynomial) (h : IsNonnegativeOn01 P) :
  ∃ (P₀ P₁ P₂ : RealPolynomial),
    (IsNonnegativeOnReals P₀) ∧
    (IsNonnegativeOnReals P₁) ∧
    (IsNonnegativeOnReals P₂) ∧
    (∀ x : ℝ, P x = P₀ x + x * P₁ x + (1 - x) * P₂ x) :=
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l66_6601


namespace NUMINAMATH_CALUDE_f_is_quadratic_l66_6640

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 3x² + 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l66_6640


namespace NUMINAMATH_CALUDE_sqrt_sum_theorem_l66_6659

theorem sqrt_sum_theorem (a : ℝ) (h : a + 1/a = 3) : 
  Real.sqrt a + 1 / Real.sqrt a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_theorem_l66_6659


namespace NUMINAMATH_CALUDE_adams_to_ricks_ratio_l66_6698

/-- Represents the cost of lunch for each person -/
structure LunchCost where
  adam : ℚ
  rick : ℚ
  jose : ℚ

/-- The conditions of the lunch scenario -/
def lunch_scenario (cost : LunchCost) : Prop :=
  cost.rick = cost.jose ∧ 
  cost.jose = 45 ∧
  cost.adam + cost.rick + cost.jose = 120

/-- The theorem stating the ratio of Adam's lunch cost to Rick's lunch cost -/
theorem adams_to_ricks_ratio (cost : LunchCost) :
  lunch_scenario cost → cost.adam / cost.rick = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_adams_to_ricks_ratio_l66_6698


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l66_6663

/-- The value of a triangle -/
def triangle_value : ℝ := sorry

/-- The value of a square -/
def square_value : ℝ := sorry

/-- The sum of three triangles and two squares equals 18 -/
axiom eq1 : 3 * triangle_value + 2 * square_value = 18

/-- The sum of two triangles and three squares equals 22 -/
axiom eq2 : 2 * triangle_value + 3 * square_value = 22

/-- The sum of three squares equals 18 -/
theorem sum_of_three_squares : 3 * square_value = 18 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l66_6663


namespace NUMINAMATH_CALUDE_no_solution_exists_l66_6638

theorem no_solution_exists : ¬∃ (m n : ℤ), 
  m ≠ n ∧ 
  988 < m ∧ m < 1991 ∧ 
  988 < n ∧ n < 1991 ∧ 
  ∃ (a : ℤ), m * n + n = a ^ 2 ∧ 
  ∃ (b : ℤ), m * n + m = b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l66_6638


namespace NUMINAMATH_CALUDE_average_decrease_rate_proof_optimal_price_reduction_proof_l66_6692

-- Define the initial price, final price, and years of decrease
def initial_price : ℝ := 200
def final_price : ℝ := 162
def years_of_decrease : ℕ := 2

-- Define the daily sales and profit parameters
def initial_daily_sales : ℕ := 20
def price_reduction_step : ℝ := 5
def sales_increase_per_step : ℕ := 10
def daily_profit : ℝ := 1150

-- Define the average yearly decrease rate
def average_decrease_rate : ℝ := 0.1

-- Define the optimal price reduction
def optimal_price_reduction : ℝ := 15

-- Theorem for the average yearly decrease rate
theorem average_decrease_rate_proof :
  initial_price * (1 - average_decrease_rate) ^ years_of_decrease = final_price :=
sorry

-- Theorem for the optimal price reduction
theorem optimal_price_reduction_proof :
  let new_price := initial_price - optimal_price_reduction
  let new_sales := initial_daily_sales + (optimal_price_reduction / price_reduction_step) * sales_increase_per_step
  (new_price - final_price) * new_sales = daily_profit :=
sorry

end NUMINAMATH_CALUDE_average_decrease_rate_proof_optimal_price_reduction_proof_l66_6692


namespace NUMINAMATH_CALUDE_inequality_solution_l66_6605

theorem inequality_solution (x : ℝ) : 
  x ≠ 2 → (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ x ∈ Set.Ici 1 ∩ Set.Iio 2 ∪ Set.Ioi (32/7) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l66_6605


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l66_6608

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l66_6608


namespace NUMINAMATH_CALUDE_b_investment_l66_6602

/-- Proves that B's investment is Rs. 12000 given the conditions of the problem -/
theorem b_investment (a_investment b_investment c_investment : ℝ)
  (b_profit : ℝ) (profit_difference : ℝ) :
  a_investment = 8000 →
  c_investment = 12000 →
  b_profit = 3000 →
  profit_difference = 1199.9999999999998 →
  (a_investment / b_investment) * b_profit =
    (c_investment / b_investment) * b_profit - profit_difference →
  b_investment = 12000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_l66_6602


namespace NUMINAMATH_CALUDE_sqrt_17_bounds_l66_6614

theorem sqrt_17_bounds : 4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_bounds_l66_6614


namespace NUMINAMATH_CALUDE_bottles_bought_l66_6674

theorem bottles_bought (initial bottles_drunk final : ℕ) : 
  initial = 42 → bottles_drunk = 25 → final = 47 → 
  final - (initial - bottles_drunk) = 30 := by sorry

end NUMINAMATH_CALUDE_bottles_bought_l66_6674


namespace NUMINAMATH_CALUDE_total_wallpaper_removal_time_l66_6631

-- Define the structure for a room
structure Room where
  name : String
  walls : Nat
  time_per_wall : List Float

-- Define the rooms
def dining_room : Room := { name := "Dining Room", walls := 3, time_per_wall := [1.5, 1.5, 1.5] }
def living_room : Room := { name := "Living Room", walls := 4, time_per_wall := [1, 1, 2.5, 2.5] }
def bedroom : Room := { name := "Bedroom", walls := 3, time_per_wall := [3, 3, 3] }
def hallway : Room := { name := "Hallway", walls := 5, time_per_wall := [4, 2, 2, 2, 2] }
def kitchen : Room := { name := "Kitchen", walls := 4, time_per_wall := [3, 1.5, 1.5, 2] }
def bathroom : Room := { name := "Bathroom", walls := 2, time_per_wall := [2, 3] }

-- Define the list of all rooms
def all_rooms : List Room := [dining_room, living_room, bedroom, hallway, kitchen, bathroom]

-- Function to calculate total time for a room
def room_time (room : Room) : Float :=
  room.time_per_wall.sum

-- Theorem: The total time to remove wallpaper from all rooms is 45.5 hours
theorem total_wallpaper_removal_time :
  (all_rooms.map room_time).sum = 45.5 := by
  sorry


end NUMINAMATH_CALUDE_total_wallpaper_removal_time_l66_6631


namespace NUMINAMATH_CALUDE_problem_polygon_area_l66_6650

/-- Polygon PQRSTU with given side lengths and properties -/
structure Polygon where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  ST : ℝ
  TU : ℝ
  PT_parallel_QR : Bool
  PU_divides : Bool

/-- Calculate the area of the polygon PQRSTU -/
def polygon_area (p : Polygon) : ℝ :=
  sorry

/-- The specific polygon from the problem -/
def problem_polygon : Polygon := {
  PQ := 4
  QR := 7
  RS := 5
  ST := 6
  TU := 3
  PT_parallel_QR := true
  PU_divides := true
}

/-- Theorem stating that the area of the problem polygon is 41.5 square units -/
theorem problem_polygon_area :
  polygon_area problem_polygon = 41.5 := by sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l66_6650


namespace NUMINAMATH_CALUDE_probability_five_diamond_ace_l66_6687

-- Define the structure of a standard deck
def StandardDeck : Type := Fin 52

-- Define card properties
def isFive (card : StandardDeck) : Prop := sorry
def isDiamond (card : StandardDeck) : Prop := sorry
def isAce (card : StandardDeck) : Prop := sorry

-- Define the probability of drawing three specific cards
def probabilityOfDraw (deck : Type) (pred1 pred2 pred3 : deck → Prop) : ℚ := sorry

-- Theorem statement
theorem probability_five_diamond_ace :
  probabilityOfDraw StandardDeck isFive isDiamond isAce = 85 / 44200 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_diamond_ace_l66_6687


namespace NUMINAMATH_CALUDE_rental_van_cost_increase_l66_6679

theorem rental_van_cost_increase 
  (total_cost : ℝ) 
  (initial_people : ℕ) 
  (withdrawing_people : ℕ) 
  (h1 : total_cost = 450) 
  (h2 : initial_people = 15) 
  (h3 : withdrawing_people = 3) : 
  let remaining_people := initial_people - withdrawing_people
  let initial_share := total_cost / initial_people
  let new_share := total_cost / remaining_people
  new_share - initial_share = 7.5 := by
sorry

end NUMINAMATH_CALUDE_rental_van_cost_increase_l66_6679


namespace NUMINAMATH_CALUDE_fruit_buckets_l66_6673

theorem fruit_buckets (bucketA bucketB bucketC : ℕ) : 
  bucketA = bucketB + 4 →
  bucketB = bucketC + 3 →
  bucketA + bucketB + bucketC = 37 →
  bucketC = 9 := by
sorry

end NUMINAMATH_CALUDE_fruit_buckets_l66_6673


namespace NUMINAMATH_CALUDE_julies_savings_l66_6628

theorem julies_savings (monthly_salary : ℝ) (savings_fraction : ℝ) : 
  monthly_salary > 0 →
  savings_fraction > 0 →
  savings_fraction < 1 →
  12 * monthly_salary * savings_fraction = 4 * monthly_salary * (1 - savings_fraction) →
  1 - savings_fraction = 3/4 := by
sorry

end NUMINAMATH_CALUDE_julies_savings_l66_6628


namespace NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l66_6611

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  δ : ℝ
  hδ_pos : δ > 0

/-- The probability that a random variable is less than a given value -/
noncomputable def prob_lt (ξ : NormalRandomVariable) (x : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable is greater than a given value -/
noncomputable def prob_gt (ξ : NormalRandomVariable) (x : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability_theorem (ξ : NormalRandomVariable) (p : ℝ) 
    (h1 : ξ.μ = 1)
    (h2 : prob_lt ξ 1 = 1/2)
    (h3 : prob_gt ξ 2 = p) :
  prob_between ξ 0 1 = 1/2 - p := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l66_6611


namespace NUMINAMATH_CALUDE_pet_store_cages_l66_6646

/-- Given a pet store scenario, calculate the number of cages used -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 120 → 
  sold_puppies = 108 → 
  puppies_per_cage = 6 → 
  (initial_puppies - sold_puppies) / puppies_per_cage = 2 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cages_l66_6646


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l66_6664

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 6 * Real.sqrt 2) ∧ 
  (θ = Real.arctan (Real.sqrt 2 / 4)) ∧
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l66_6664


namespace NUMINAMATH_CALUDE_isosceles_triangle_special_angles_l66_6666

/-- An isosceles triangle with vertex angle twice the base angle has a 90° vertex angle and 45° base angles. -/
theorem isosceles_triangle_special_angles :
  ∀ (vertex_angle base_angle : ℝ),
    vertex_angle > 0 →
    base_angle > 0 →
    vertex_angle = 2 * base_angle →
    vertex_angle + 2 * base_angle = 180 →
    vertex_angle = 90 ∧ base_angle = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_special_angles_l66_6666


namespace NUMINAMATH_CALUDE_block_edge_sum_l66_6639

/-- A rectangular block with a square base -/
structure Block where
  side : ℝ  -- side length of the square base
  height : ℝ  -- height of the block

/-- The volume of the block -/
def volume (b : Block) : ℝ := b.side^2 * b.height

/-- The surface area of the vertical sides of the block -/
def verticalSurfaceArea (b : Block) : ℝ := 4 * b.side * b.height

/-- The sum of the lengths of all edges of the block -/
def sumOfEdges (b : Block) : ℝ := 8 * b.side + 4 * b.height

theorem block_edge_sum (b : Block) 
  (h_volume : volume b = 576) 
  (h_area : verticalSurfaceArea b = 384) : 
  sumOfEdges b = 112 := by
  sorry


end NUMINAMATH_CALUDE_block_edge_sum_l66_6639


namespace NUMINAMATH_CALUDE_salty_cookies_eaten_correct_l66_6694

/-- The number of salty cookies Paco ate -/
def salty_cookies_eaten (initial_salty initial_sweet eaten_sweet salty_left : ℕ) : ℕ :=
  initial_salty - salty_left

/-- Theorem: The number of salty cookies Paco ate is the difference between
    the initial number of salty cookies and the number of salty cookies left -/
theorem salty_cookies_eaten_correct
  (initial_salty initial_sweet eaten_sweet salty_left : ℕ)
  (h1 : initial_salty = 26)
  (h2 : initial_sweet = 17)
  (h3 : eaten_sweet = 14)
  (h4 : salty_left = 17)
  (h5 : initial_salty ≥ salty_left) :
  salty_cookies_eaten initial_salty initial_sweet eaten_sweet salty_left = initial_salty - salty_left :=
by
  sorry

end NUMINAMATH_CALUDE_salty_cookies_eaten_correct_l66_6694


namespace NUMINAMATH_CALUDE_smallest_n_inequality_l66_6699

theorem smallest_n_inequality (x y z : ℝ) :
  (∃ (n : ℕ), ∀ (a b c : ℝ), (a^2 + b^2 + c^2) ≤ n * (a^4 + b^4 + c^4)) ∧
  (∀ (n : ℕ), (∀ (a b c : ℝ), (a^2 + b^2 + c^2) ≤ n * (a^4 + b^4 + c^4)) → n ≥ 3) ∧
  ((x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_inequality_l66_6699


namespace NUMINAMATH_CALUDE_sqrt_inequality_l66_6616

theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l66_6616


namespace NUMINAMATH_CALUDE_distance_between_bars_l66_6609

/-- The distance between two bars given the walking times and speeds of two people --/
theorem distance_between_bars 
  (pierrot_extra_distance : ℝ) 
  (pierrot_time_after : ℝ) 
  (jeannot_time_after : ℝ) 
  (pierrot_speed_halved : ℝ → ℝ) 
  (jeannot_speed_halved : ℝ → ℝ) :
  ∃ (d : ℝ),
    pierrot_extra_distance = 200 ∧
    pierrot_time_after = 8 ∧
    jeannot_time_after = 18 ∧
    (∀ x, pierrot_speed_halved x = x / 2) ∧
    (∀ x, jeannot_speed_halved x = x / 2) ∧
    d > 0 ∧
    (d - pierrot_extra_distance) / (pierrot_speed_halved (d - pierrot_extra_distance) / pierrot_time_after) = 
      d / (jeannot_speed_halved d / jeannot_time_after) ∧
    2 * d - pierrot_extra_distance = 1000 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_bars_l66_6609


namespace NUMINAMATH_CALUDE_vector_operation_l66_6619

/-- Given two 2D vectors a and b, prove that 3a - b equals (4, 2) -/
theorem vector_operation (a b : Fin 2 → ℝ) 
  (ha : a = ![1, 1]) 
  (hb : b = ![-1, 1]) : 
  (3 • a) - b = ![4, 2] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l66_6619


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l66_6603

theorem arithmetic_calculation : 2011 - (9 * 11 * 11 + 9 * 9 * 11 - 9 * 11) = 130 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l66_6603


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_graph_l66_6606

theorem max_y_coordinate_polar_graph :
  let r : ℝ → ℝ := λ θ ↦ 2 * Real.sin (2 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  (∀ θ, y θ ≤ (8 * Real.sqrt 3) / 9) ∧ 
  (∃ θ, y θ = (8 * Real.sqrt 3) / 9) := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_graph_l66_6606


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l66_6604

/-- Given two vectors in 2D Euclidean space with specific magnitudes and angle between them,
    prove that the magnitude of their difference is √7. -/
theorem vector_difference_magnitude
  (a b : ℝ × ℝ)  -- Two vectors in 2D real space
  (h1 : ‖a‖ = 2)  -- Magnitude of a is 2
  (h2 : ‖b‖ = 3)  -- Magnitude of b is 3
  (h3 : a • b = 3)  -- Dot product of a and b (equivalent to 60° angle)
  : ‖a - b‖ = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l66_6604


namespace NUMINAMATH_CALUDE_blue_socks_count_l66_6621

/-- The number of red socks -/
def red_socks : ℕ := 2

/-- The number of black socks -/
def black_socks : ℕ := 2

/-- The number of white socks -/
def white_socks : ℕ := 2

/-- The probability of drawing two socks of the same color -/
def same_color_prob : ℚ := 1/5

theorem blue_socks_count (x : ℕ) (hx : x > 0) :
  let total := red_socks + black_socks + white_socks + x
  (3 * 2 + x * (x - 1)) / (total * (total - 1)) = same_color_prob →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_blue_socks_count_l66_6621


namespace NUMINAMATH_CALUDE_right_triangle_legs_l66_6653

theorem right_triangle_legs (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 = b^2 + c^2 →
  (1/2) * b * c = 150 →
  a = 25 →
  (b = 20 ∧ c = 15) ∨ (b = 15 ∧ c = 20) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l66_6653


namespace NUMINAMATH_CALUDE_f_max_min_values_f_max_min_m_neg_f_max_min_m_0_to_4_f_max_min_m_gt_4_l66_6685

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m - 1

-- Define the domain
def domain : Set ℝ := Set.Icc 0 4

-- Theorem for the maximum and minimum values
theorem f_max_min_values (m : ℝ) :
  (∀ x ∈ domain, f m x ≥ (m - 1) ∧ f m x ≤ (15 - 7*m)) ∨
  ((∀ x ∈ domain, f m x ≥ (-m^2 + m - 1)) ∧
   ((0 ≤ m ∧ m ≤ 2 → ∀ x ∈ domain, f m x ≤ (15 - 7*m)) ∧
    (2 ≤ m ∧ m ≤ 4 → ∀ x ∈ domain, f m x ≤ (m - 1)))) ∨
  (∀ x ∈ domain, f m x ≥ (15 - 7*m) ∧ f m x ≤ (m - 1)) :=
by sorry

-- Helper theorems for each case
theorem f_max_min_m_neg (m : ℝ) (hm : m < 0) :
  ∀ x ∈ domain, f m x ≥ (m - 1) ∧ f m x ≤ (15 - 7*m) :=
by sorry

theorem f_max_min_m_0_to_4 (m : ℝ) (hm : 0 ≤ m ∧ m ≤ 4) :
  (∀ x ∈ domain, f m x ≥ (-m^2 + m - 1)) ∧
  ((0 ≤ m ∧ m ≤ 2 → ∀ x ∈ domain, f m x ≤ (15 - 7*m)) ∧
   (2 ≤ m ∧ m ≤ 4 → ∀ x ∈ domain, f m x ≤ (m - 1))) :=
by sorry

theorem f_max_min_m_gt_4 (m : ℝ) (hm : m > 4) :
  ∀ x ∈ domain, f m x ≥ (15 - 7*m) ∧ f m x ≤ (m - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_values_f_max_min_m_neg_f_max_min_m_0_to_4_f_max_min_m_gt_4_l66_6685


namespace NUMINAMATH_CALUDE_count_polygons_l66_6677

/-- The number of points placed on the circle -/
def n : ℕ := 15

/-- The number of distinct convex polygons with at least three sides -/
def num_polygons : ℕ := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)

/-- Theorem stating that the number of distinct convex polygons is 32647 -/
theorem count_polygons : num_polygons = 32647 := by
  sorry

end NUMINAMATH_CALUDE_count_polygons_l66_6677


namespace NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l66_6697

/-- The area of the triangle formed by the lines y = 3x - 3, y = -2x + 18, and the y-axis -/
theorem triangle_area : ℝ → Prop :=
  fun A => 
    let line1 := fun x : ℝ => 3 * x - 3
    let line2 := fun x : ℝ => -2 * x + 18
    let y_axis := fun x : ℝ => 0
    let intersection_x := (21 : ℝ) / 5
    let intersection_y := line1 intersection_x
    let base := line2 0 - line1 0
    let height := intersection_x
    A = (1 / 2) * base * height ∧ A = 441 / 10

/-- Proof of the theorem -/
theorem prove_triangle_area : ∃ A : ℝ, triangle_area A :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l66_6697


namespace NUMINAMATH_CALUDE_series_sum_equals_five_l66_6615

theorem series_sum_equals_five (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n + 2) / k^n = 5) : k = (7 + Real.sqrt 14) / 5 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_five_l66_6615


namespace NUMINAMATH_CALUDE_product_less_than_60_probability_l66_6658

def paco_range : Finset ℕ := Finset.range 5
def manu_range : Finset ℕ := Finset.range 20

def total_outcomes : ℕ := paco_range.card * manu_range.card

def favorable_outcomes : ℕ :=
  (paco_range.filter (fun p => p + 1 ≤ 2)).sum (fun p =>
    (manu_range.filter (fun m => (p + 1) * (m + 1) < 60)).card)
  +
  (paco_range.filter (fun p => p + 1 > 2)).sum (fun p =>
    (manu_range.filter (fun m => (p + 1) * (m + 1) < 60)).card)

theorem product_less_than_60_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 21 / 25 := by sorry

end NUMINAMATH_CALUDE_product_less_than_60_probability_l66_6658


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l66_6623

theorem prop_a_necessary_not_sufficient_for_prop_b :
  (∀ (a b : ℝ), (1 / b < 1 / a ∧ 1 / a < 0) → a * b > b ^ 2) ∧
  (∃ (a b : ℝ), a * b > b ^ 2 ∧ ¬(1 / b < 1 / a ∧ 1 / a < 0)) := by
  sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l66_6623


namespace NUMINAMATH_CALUDE_max_cables_theorem_l66_6667

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brandA : ℕ  -- Number of brand A computers
  brandB : ℕ  -- Number of brand B computers

/-- Calculates the maximum number of cables that can be used in the network. -/
def maxCables (network : ComputerNetwork) : ℕ :=
  network.brandA * network.brandB

/-- Theorem: The maximum number of cables in a network with 25 brand A and 15 brand B computers is 361. -/
theorem max_cables_theorem (network : ComputerNetwork) 
  (h1 : network.brandA = 25) 
  (h2 : network.brandB = 15) : 
  maxCables network = 361 := by
  sorry

#eval maxCables { brandA := 25, brandB := 15 }

end NUMINAMATH_CALUDE_max_cables_theorem_l66_6667


namespace NUMINAMATH_CALUDE_same_number_of_atoms_l66_6681

/-- The number of atoms in a mole of a substance -/
def atoms_per_mole (substance : String) : ℕ :=
  match substance with
  | "H₃PO₄" => 8
  | "H₂O₂" => 4
  | _ => 0

/-- The number of moles of a substance -/
def moles (substance : String) : ℚ :=
  match substance with
  | "H₃PO₄" => 1/5
  | "H₂O₂" => 2/5
  | _ => 0

/-- The total number of atoms in a given amount of a substance -/
def total_atoms (substance : String) : ℚ :=
  (moles substance) * (atoms_per_mole substance)

theorem same_number_of_atoms : total_atoms "H₃PO₄" = total_atoms "H₂O₂" := by
  sorry

end NUMINAMATH_CALUDE_same_number_of_atoms_l66_6681


namespace NUMINAMATH_CALUDE_distance_to_origin_l66_6693

/-- The distance from the point (3, -4) to the origin (0, 0) in the Cartesian coordinate system is 5. -/
theorem distance_to_origin : Real.sqrt (3^2 + (-4)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l66_6693


namespace NUMINAMATH_CALUDE_maxwells_walking_speed_l66_6671

/-- Proves that Maxwell's walking speed is 3 km/h given the problem conditions --/
theorem maxwells_walking_speed 
  (total_distance : ℝ) 
  (maxwell_distance : ℝ) 
  (brad_speed : ℝ) 
  (h1 : total_distance = 36) 
  (h2 : maxwell_distance = 12) 
  (h3 : brad_speed = 6) : 
  maxwell_distance / (total_distance - maxwell_distance) * brad_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_maxwells_walking_speed_l66_6671


namespace NUMINAMATH_CALUDE_fabian_shopping_cost_l66_6654

/-- Calculates the total cost of Fabian's shopping --/
def shopping_cost (apple_price : ℝ) (walnut_price : ℝ) (apple_quantity : ℝ) (sugar_quantity : ℝ) (walnut_quantity : ℝ) : ℝ :=
  let sugar_price := apple_price - 1
  apple_price * apple_quantity + sugar_price * sugar_quantity + walnut_price * walnut_quantity

/-- Proves that the total cost of Fabian's shopping is $16 --/
theorem fabian_shopping_cost :
  shopping_cost 2 6 5 3 0.5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fabian_shopping_cost_l66_6654


namespace NUMINAMATH_CALUDE_sqrt_inequality_l66_6636

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l66_6636


namespace NUMINAMATH_CALUDE_complex_modulus_example_l66_6617

theorem complex_modulus_example : Complex.abs (-5 - (8/3)*Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l66_6617


namespace NUMINAMATH_CALUDE_divisibility_property_l66_6670

theorem divisibility_property (n : ℕ) : 
  ∃ (a b : ℕ), (a * n + 1)^6 + b ≡ 0 [MOD (n^2 + n + 1)] :=
by
  use 2, 27
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l66_6670


namespace NUMINAMATH_CALUDE_function_inequality_l66_6648

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x ∈ (Set.Ioo 0 (π / 2)), HasDerivAt f (f' x) x) →
  (∀ x ∈ (Set.Ioo 0 (π / 2)), f x * tan x + f' x < 0) →
  Real.sqrt 3 * f (π / 3) < f (π / 6) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l66_6648


namespace NUMINAMATH_CALUDE_probability_two_even_toys_l66_6651

def total_toys : ℕ := 21
def even_toys : ℕ := 10

theorem probability_two_even_toys :
  let p1 := even_toys / total_toys
  let p2 := (even_toys - 1) / (total_toys - 1)
  p1 * p2 = 3 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_two_even_toys_l66_6651


namespace NUMINAMATH_CALUDE_monthly_parking_rate_l66_6669

/-- Proves that the monthly parking rate is $24 given the specified conditions -/
theorem monthly_parking_rate (weekly_rate : ℕ) (yearly_savings : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_rate = 10 →
  yearly_savings = 232 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  ∃ (monthly_rate : ℕ), monthly_rate = 24 ∧ weeks_per_year * weekly_rate - months_per_year * monthly_rate = yearly_savings :=
by sorry

end NUMINAMATH_CALUDE_monthly_parking_rate_l66_6669


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_l66_6629

/-- Given a triangle with side lengths a, b, and c, 
    prove the inequality and equality condition --/
theorem triangle_inequality_and_equality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0) ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_l66_6629


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l66_6643

theorem multiply_and_simplify (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x)^3) = 25 / 8 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l66_6643


namespace NUMINAMATH_CALUDE_max_books_borrowed_l66_6665

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat) 
  (avg_books : Nat) (h1 : total_students = 20) (h2 : zero_books = 3) (h3 : one_book = 9) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : ∃ (max_books : Nat), max_books = 14 ∧ 
  max_books = total_students * avg_books - (zero_books * 0 + one_book * 1 + two_books * 2 + 
  (total_students - zero_books - one_book - two_books - 1) * 3) := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l66_6665


namespace NUMINAMATH_CALUDE_semicircle_shaded_area_l66_6613

/-- Given two adjacent semicircles sharing a diameter of length 2, prove that the area of the
    rectangle formed by the diameter and the vertical line through the intersection of the
    midpoints of the semicircle arcs is 2 square units. -/
theorem semicircle_shaded_area (X Y W Z M N P : ℝ × ℝ) : 
  -- Diameter XY is 2 units long
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 4 →
  -- M is on semicircle WXY
  (M.1 - X.1)^2 + (M.2 - X.2)^2 = 1 →
  -- N is on semicircle ZXY
  (N.1 - X.1)^2 + (N.2 - X.2)^2 = 1 →
  -- M is midpoint of arc WX
  (M.1 - W.1)^2 + (M.2 - W.2)^2 = (X.1 - W.1)^2 + (X.2 - W.2)^2 →
  -- N is midpoint of arc ZY
  (N.1 - Z.1)^2 + (N.2 - Z.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 →
  -- P is on the vertical line from M
  P.1 = M.1 →
  -- P is on the vertical line from N
  P.1 = N.1 →
  -- P is the midpoint of MN
  2 * P.1 = M.1 + N.1 ∧ 2 * P.2 = M.2 + N.2 →
  -- The area of the rectangle is 2 square units
  (X.1 - Y.1) * (P.2 - X.2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_shaded_area_l66_6613


namespace NUMINAMATH_CALUDE_computer_operations_l66_6655

theorem computer_operations (additions_per_second multiplications_per_second : ℕ) 
  (h1 : additions_per_second = 12000)
  (h2 : multiplications_per_second = 8000) :
  (additions_per_second + multiplications_per_second) * (30 * 60) = 36000000 := by
  sorry

#check computer_operations

end NUMINAMATH_CALUDE_computer_operations_l66_6655


namespace NUMINAMATH_CALUDE_no_constant_term_implies_n_not_eight_l66_6672

theorem no_constant_term_implies_n_not_eight (n : ℕ) :
  (∀ r : ℕ, r ≤ n → n ≠ 4 / 3 * r) →
  n ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_constant_term_implies_n_not_eight_l66_6672


namespace NUMINAMATH_CALUDE_range_of_c_l66_6660

theorem range_of_c (c : ℝ) : 
  (∀ x > 0, c^2 * x^2 - (c * x + 1) * Real.log x + c * x ≥ 0) ↔ c ≥ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l66_6660


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l66_6696

theorem consecutive_product_not_power (n m : ℕ) (h : m > 1) :
  ¬ ∃ k : ℕ, (n - 1) * n * (n + 1) = k ^ m := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l66_6696


namespace NUMINAMATH_CALUDE_row_sum_equals_square_l66_6686

theorem row_sum_equals_square (k : ℕ) (h : k > 0) : 
  let n := 2 * k - 1
  let a := k
  let l := 3 * k - 2
  (n * (a + l)) / 2 = (2 * k - 1)^2 := by
sorry

end NUMINAMATH_CALUDE_row_sum_equals_square_l66_6686


namespace NUMINAMATH_CALUDE_binomial_sum_formula_l66_6635

def binomial_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => (k + 1) * (k + 2) * Nat.choose n (k + 1))

theorem binomial_sum_formula (n : ℕ) (h : n ≥ 4) :
  binomial_sum n = n * (n + 3) * 2^(n - 2) :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_formula_l66_6635


namespace NUMINAMATH_CALUDE_bug_probability_l66_6691

def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

theorem bug_probability : P 12 = 683/2048 := by sorry

end NUMINAMATH_CALUDE_bug_probability_l66_6691


namespace NUMINAMATH_CALUDE_ring_cost_l66_6649

theorem ring_cost (total_revenue : ℕ) (necklace_count : ℕ) (ring_count : ℕ) (necklace_price : ℕ) :
  total_revenue = 80 →
  necklace_count = 4 →
  ring_count = 8 →
  necklace_price = 12 →
  ∃ (ring_price : ℕ), ring_price = 4 ∧ total_revenue = necklace_count * necklace_price + ring_count * ring_price :=
by
  sorry

end NUMINAMATH_CALUDE_ring_cost_l66_6649


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l66_6618

/-- A triangle with sides a, b, and c is either isosceles or right-angled if (a - b) * (a² + b² - c²) = 0 --/
theorem triangle_isosceles_or_right_angled (a b c : ℝ) (h : (a - b) * (a^2 + b^2 - c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l66_6618


namespace NUMINAMATH_CALUDE_sqrt_simplification_l66_6652

theorem sqrt_simplification : 
  Real.sqrt 80 - Real.sqrt 20 + Real.sqrt 5 = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l66_6652


namespace NUMINAMATH_CALUDE_triangle_vector_relation_l66_6637

theorem triangle_vector_relation (A B C : ℝ × ℝ) (a b : ℝ × ℝ) :
  (B.1 - C.1, B.2 - C.2) = a →
  (C.1 - A.1, C.2 - A.2) = b →
  (A.1 - B.1, A.2 - B.2) = (b.1 - a.1, b.2 - a.2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_relation_l66_6637


namespace NUMINAMATH_CALUDE_smallest_power_l66_6676

theorem smallest_power (a b c d : ℕ) : 
  2^55 < 3^44 ∧ 2^55 < 5^33 ∧ 2^55 < 6^22 :=
by sorry

end NUMINAMATH_CALUDE_smallest_power_l66_6676


namespace NUMINAMATH_CALUDE_inequality_solution_set_l66_6634

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | 56 * x^2 + a * x - a^2 < 0} = {x : ℝ | a / 8 < x ∧ x < -a / 7} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l66_6634


namespace NUMINAMATH_CALUDE_combined_variance_is_100_l66_6675

/-- Calculates the combined variance of two classes given their individual statistics -/
def combinedVariance (nA nB : ℕ) (meanA meanB : ℝ) (varA varB : ℝ) : ℝ :=
  let n := nA + nB
  let pA := nA / n
  let pB := nB / n
  let combinedMean := pA * meanA + pB * meanB
  pA * (varA + (meanA - combinedMean)^2) + pB * (varB + (meanB - combinedMean)^2)

/-- The variance of the combined scores of Class A and Class B is 100 -/
theorem combined_variance_is_100 :
  combinedVariance 50 40 76 85 96 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_variance_is_100_l66_6675


namespace NUMINAMATH_CALUDE_richard_cleaning_time_l66_6682

/-- Richard's room cleaning time in minutes -/
def richard_time : ℕ := 45

/-- Cory's room cleaning time in minutes -/
def cory_time (r : ℕ) : ℕ := r + 3

/-- Blake's room cleaning time in minutes -/
def blake_time (r : ℕ) : ℕ := cory_time r - 4

/-- Total cleaning time for all three people in minutes -/
def total_time : ℕ := 136

theorem richard_cleaning_time :
  richard_time + cory_time richard_time + blake_time richard_time = total_time :=
sorry

end NUMINAMATH_CALUDE_richard_cleaning_time_l66_6682


namespace NUMINAMATH_CALUDE_power_15000_mod_1000_l66_6625

theorem power_15000_mod_1000 (h : 7^500 ≡ 1 [ZMOD 1000]) :
  7^15000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_15000_mod_1000_l66_6625


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l66_6647

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^3 / (a^2 * b) ≥ 27/4 ∧ ((a + b)^3 / (a^2 * b) = 27/4 ↔ a = 2*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l66_6647


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l66_6600

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  focus : ℝ
  equation : ℝ → ℝ → Prop

/-- A circle with center (a, 0) and radius r -/
structure Circle where
  center : ℝ
  radius : ℝ
  equation : ℝ → ℝ → Prop

/-- The theorem statement -/
theorem parabola_circle_tangency 
  (C : Parabola) 
  (M : Circle)
  (h1 : C.focus > 0)
  (h2 : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ C.equation 1 y₁ ∧ C.equation 1 y₂)
  (h3 : ∀ (y₁ y₂ : ℝ), y₁ ≠ y₂ → C.equation 1 y₁ → C.equation 1 y₂ → y₁ * y₂ = -1)
  (h4 : M.center = 2)
  (h5 : M.radius = 1) :
  (C.equation = fun x y ↦ y^2 = x) ∧ 
  (M.equation = fun x y ↦ (x - 2)^2 + y^2 = 1) ∧ 
  (∀ (A₁ A₂ A₃ : ℝ × ℝ), 
    C.equation A₁.1 A₁.2 → 
    C.equation A₂.1 A₂.2 → 
    C.equation A₃.1 A₃.2 → 
    (∃ (k₁ k₂ : ℝ), 
      (∀ x y, y = k₁ * (x - A₁.1) + A₁.2 → 
        ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius)) ∧
      (∀ x y, y = k₂ * (x - A₁.1) + A₁.2 → 
        ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius))) →
    ∃ (k : ℝ), ∀ x y, y = k * (x - A₂.1) + A₂.2 → 
      ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius)) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l66_6600


namespace NUMINAMATH_CALUDE_police_chase_distance_l66_6627

/-- Calculates the distance between a police station and a thief's starting location
    given their speeds and chase duration. -/
def police_station_distance (thief_speed : ℝ) (police_speed : ℝ) 
                             (head_start : ℝ) (chase_duration : ℝ) : ℝ :=
  police_speed * chase_duration - 
  (thief_speed * head_start + thief_speed * chase_duration)

/-- Theorem stating that given specific chase parameters, 
    the police station is 60 km away from the thief's starting point. -/
theorem police_chase_distance : 
  police_station_distance 20 40 1 4 = 60 := by sorry

end NUMINAMATH_CALUDE_police_chase_distance_l66_6627


namespace NUMINAMATH_CALUDE_average_of_five_integers_l66_6695

theorem average_of_five_integers (k m r s t : ℕ) : 
  k < m → m < r → r < s → s < t → 
  t = 20 → r = 13 → k ≥ 1 → m ≥ 2 →
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_integers_l66_6695


namespace NUMINAMATH_CALUDE_sector_area_l66_6633

theorem sector_area (θ : Real) (L : Real) (A : Real) :
  θ = π / 6 →
  L = 2 * π / 3 →
  A = 4 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l66_6633


namespace NUMINAMATH_CALUDE_tory_cookie_sales_l66_6668

theorem tory_cookie_sales (grandmother_packs uncle_packs neighbor_packs more_packs : ℕ) 
  (h1 : grandmother_packs = 12)
  (h2 : uncle_packs = 7)
  (h3 : neighbor_packs = 5)
  (h4 : more_packs = 26) :
  grandmother_packs + uncle_packs + neighbor_packs + more_packs = 50 := by
  sorry

end NUMINAMATH_CALUDE_tory_cookie_sales_l66_6668


namespace NUMINAMATH_CALUDE_smallest_batch_size_l66_6662

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) :
  N ≥ 80 ∧ ∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_batch_size_l66_6662


namespace NUMINAMATH_CALUDE_polynomial_factorization_l66_6642

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l66_6642


namespace NUMINAMATH_CALUDE_sin_alpha_minus_beta_l66_6630

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α - Real.cos β = -2/3)
  (h2 : Real.cos α + Real.sin β = 1/3) :
  Real.sin (α - β) = 13/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_beta_l66_6630


namespace NUMINAMATH_CALUDE_cats_in_center_l66_6626

/-- The number of cats that can jump -/
def jump : ℕ := 60

/-- The number of cats that can fetch -/
def fetch : ℕ := 35

/-- The number of cats that can spin -/
def spin : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and spin -/
def fetch_spin : ℕ := 15

/-- The number of cats that can jump and spin -/
def jump_spin : ℕ := 25

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 10

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 8

/-- The total number of cats in the center -/
def total_cats : ℕ := 93

theorem cats_in_center : 
  jump + fetch + spin - jump_fetch - fetch_spin - jump_spin + all_three + no_tricks = total_cats :=
by sorry

end NUMINAMATH_CALUDE_cats_in_center_l66_6626


namespace NUMINAMATH_CALUDE_triangle_toothpicks_l66_6683

/-- Calculates the number of toothpicks needed for a large equilateral triangle
    with a given base length and border. -/
def toothpicks_for_triangle (base : ℕ) (border : ℕ) : ℕ :=
  let interior_triangles := base * (base + 1) / 2
  let interior_toothpicks := 3 * interior_triangles / 2
  let boundary_toothpicks := 3 * base
  let border_toothpicks := 2 * border + 2
  interior_toothpicks + boundary_toothpicks + border_toothpicks

/-- Theorem stating that a triangle with base 100 and border 100 requires 8077 toothpicks -/
theorem triangle_toothpicks :
  toothpicks_for_triangle 100 100 = 8077 := by
  sorry

end NUMINAMATH_CALUDE_triangle_toothpicks_l66_6683


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l66_6644

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 / 4) * (p + q + r + s) = 15) : 
  (p + q + r + s) / 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l66_6644


namespace NUMINAMATH_CALUDE_distinct_role_selection_l66_6620

theorem distinct_role_selection (n : ℕ) (k : ℕ) : 
  n ≥ k → (n * (n - 1) * (n - 2) = (n.factorial) / ((n - k).factorial)) → 
  (8 * 7 * 6 = 336) :=
by sorry

end NUMINAMATH_CALUDE_distinct_role_selection_l66_6620


namespace NUMINAMATH_CALUDE_expression_evaluation_l66_6690

theorem expression_evaluation : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l66_6690


namespace NUMINAMATH_CALUDE_existence_of_small_triangle_l66_6610

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of points -/
def PointSet := Set Point

/-- Definition of a square with side length 20 -/
def is_square_20 (A B C D : Point) : Prop := sorry

/-- Check if three points are collinear -/
def are_collinear (P Q R : Point) : Prop := sorry

/-- Check if a point is inside a square -/
def is_inside_square (P : Point) (A B C D : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
def triangle_area (P Q R : Point) : ℝ := sorry

theorem existence_of_small_triangle 
  (A B C D : Point) 
  (T : Fin 2000 → Point)
  (h_square : is_square_20 A B C D)
  (h_inside : ∀ i, is_inside_square (T i) A B C D)
  (h_not_collinear : ∀ P Q R, P ≠ Q → Q ≠ R → P ≠ R → 
    P ∈ {A, B, C, D} ∪ (Set.range T) → 
    Q ∈ {A, B, C, D} ∪ (Set.range T) → 
    R ∈ {A, B, C, D} ∪ (Set.range T) → 
    ¬(are_collinear P Q R)) :
  ∃ P Q R, P ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           Q ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           R ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           triangle_area P Q R < 1/10 :=
sorry

end NUMINAMATH_CALUDE_existence_of_small_triangle_l66_6610


namespace NUMINAMATH_CALUDE_building_painting_cost_l66_6678

theorem building_painting_cost (room1_area room2_area room3_area : ℝ)
  (paint_price1 paint_price2 paint_price3 : ℝ)
  (labor_cost : ℝ) (tax_rate : ℝ) :
  room1_area = 196 →
  room2_area = 150 →
  room3_area = 250 →
  paint_price1 = 15 →
  paint_price2 = 18 →
  paint_price3 = 20 →
  labor_cost = 800 →
  tax_rate = 0.05 →
  let room1_cost := room1_area * paint_price1
  let room2_cost := room2_area * paint_price2
  let room3_cost := room3_area * paint_price3
  let total_painting_cost := room1_cost + room2_cost + room3_cost
  let total_cost_before_tax := total_painting_cost + labor_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + tax
  total_cost_after_tax = 12012 :=
by sorry

end NUMINAMATH_CALUDE_building_painting_cost_l66_6678


namespace NUMINAMATH_CALUDE_full_merit_scholarship_percentage_l66_6688

theorem full_merit_scholarship_percentage
  (total_students : ℕ)
  (half_merit_percentage : ℚ)
  (no_scholarship_count : ℕ)
  (h1 : total_students = 300)
  (h2 : half_merit_percentage = 1 / 10)
  (h3 : no_scholarship_count = 255) :
  (total_students - (half_merit_percentage * total_students).floor - no_scholarship_count) / total_students = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_full_merit_scholarship_percentage_l66_6688


namespace NUMINAMATH_CALUDE_some_number_value_l66_6661

theorem some_number_value (x : ℝ) : (85 + 32 / x) * x = 9637 → x = 113 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l66_6661


namespace NUMINAMATH_CALUDE_james_weekly_beats_l66_6689

/-- The number of beats James hears per week -/
def beats_per_week : ℕ :=
  let beats_per_minute : ℕ := 200
  let hours_per_day : ℕ := 2
  let minutes_per_hour : ℕ := 60
  let days_per_week : ℕ := 7
  beats_per_minute * hours_per_day * minutes_per_hour * days_per_week

/-- Theorem stating that James hears 168,000 beats per week -/
theorem james_weekly_beats : beats_per_week = 168000 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_beats_l66_6689


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l66_6622

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {4, 7, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l66_6622


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l66_6680

theorem sufficient_not_necessary (x y a m : ℝ) :
  (∀ x y a m : ℝ, (|x - a| < m ∧ |y - a| < m) → |x - y| < 2*m) ∧
  (∃ x y a m : ℝ, |x - y| < 2*m ∧ ¬(|x - a| < m ∧ |y - a| < m)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l66_6680


namespace NUMINAMATH_CALUDE_pascal_triangle_25th_row_5th_number_l66_6612

theorem pascal_triangle_25th_row_5th_number : 
  let n : ℕ := 24  -- The row number (0-indexed) for a row with 25 numbers
  let k : ℕ := 4   -- The index (0-indexed) of the 5th number
  Nat.choose n k = 12650 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_25th_row_5th_number_l66_6612


namespace NUMINAMATH_CALUDE_circle_placement_l66_6632

theorem circle_placement (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (num_squares : ℕ) (square_size : ℝ) (circle_diameter : ℝ) :
  rectangle_width = 20 ∧ 
  rectangle_height = 25 ∧ 
  num_squares = 120 ∧ 
  square_size = 1 ∧ 
  circle_diameter = 1 →
  ∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ rectangle_width ∧ 
    0 ≤ y ∧ y ≤ rectangle_height ∧ 
    ∀ (i : ℕ), i < num_squares →
      ∃ (sx sy : ℝ), 
        0 ≤ sx ∧ sx + square_size ≤ rectangle_width ∧
        0 ≤ sy ∧ sy + square_size ≤ rectangle_height ∧
        (x - sx)^2 + (y - sy)^2 ≥ (circle_diameter / 2 + square_size / 2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_placement_l66_6632


namespace NUMINAMATH_CALUDE_digit_150_is_3_l66_6641

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => Fin.ofNat ((10 * (10^n % 13)) / 13)

/-- The length of the repeating block in the decimal representation of 1/13 -/
def rep_length : ℕ := 6

/-- The 150th digit after the decimal point in the decimal representation of 1/13 -/
def digit_150 : Fin 10 := decimal_rep_1_13 149

theorem digit_150_is_3 : digit_150 = 3 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_3_l66_6641


namespace NUMINAMATH_CALUDE_largest_odd_integer_sum_30_l66_6645

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def consecutive_odd_integers (m : ℕ) : List ℕ := [m - 4, m - 2, m, m + 2, m + 4]

theorem largest_odd_integer_sum_30 :
  ∃ m : ℕ, 
    (sum_first_n_odd 30 = (consecutive_odd_integers m).sum) ∧
    (List.maximum (consecutive_odd_integers m) = some 184) := by
  sorry

end NUMINAMATH_CALUDE_largest_odd_integer_sum_30_l66_6645


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l66_6656

theorem smallest_area_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area := (1/2) * a * b
  area = 24 ∧ ∀ (x y : ℝ), (x = a ∧ y = b) ∨ (x = a ∧ y = b) ∨ (x^2 + y^2 = a^2 + b^2) → (1/2) * x * y ≥ area :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l66_6656
