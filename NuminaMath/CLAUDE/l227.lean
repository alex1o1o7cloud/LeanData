import Mathlib

namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l227_22784

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 10 + 15 = 17) → initial_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l227_22784


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l227_22778

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + 12*x = 6*x^2 + 35 :=
by
  -- The unique solution is x = 5
  use 5
  constructor
  · -- Prove that x = 5 satisfies the equation
    simp
    -- Additional steps to prove 5^3 + 12*5 = 6*5^2 + 35
    sorry
  · -- Prove that any solution must equal 5
    intro y hy
    -- Steps to show that if y satisfies the equation, then y = 5
    sorry


end NUMINAMATH_CALUDE_cubic_equation_solution_l227_22778


namespace NUMINAMATH_CALUDE_greatest_difference_l227_22797

theorem greatest_difference (x y : ℤ) 
  (hx : 5 < x ∧ x < 8) 
  (hy : 8 < y ∧ y < 13) 
  (hxdiv : x % 3 = 0) 
  (hydiv : y % 3 = 0) : 
  (∀ a b : ℤ, 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 ∧ a % 3 = 0 ∧ b % 3 = 0 → b - a ≤ y - x) ∧ y - x = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_l227_22797


namespace NUMINAMATH_CALUDE_race_length_is_1000_l227_22700

/-- The length of a race, given the positions of two runners at the end. -/
def race_length (jack_position : ℕ) (distance_apart : ℕ) : ℕ :=
  jack_position + distance_apart

/-- Theorem stating that the race length is 1000 meters given the conditions -/
theorem race_length_is_1000 :
  race_length 152 848 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_1000_l227_22700


namespace NUMINAMATH_CALUDE_no_consecutive_digit_products_exist_l227_22704

/-- Given a natural number, return the product of its digits -/
def digitProduct (n : ℕ) : ℕ := sorry

theorem no_consecutive_digit_products_exist (n : ℕ) : 
  let x := digitProduct n
  let y := digitProduct (n + 1)
  ¬∃ m : ℕ, digitProduct m = y - 1 ∧ digitProduct (m + 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_digit_products_exist_l227_22704


namespace NUMINAMATH_CALUDE_constant_term_quadratic_equation_l227_22703

theorem constant_term_quadratic_equation :
  ∀ (x : ℝ), x^2 - 5*x = 2 → ∃ (a b c : ℝ), a = 1 ∧ x^2 + b*x + c = 0 ∧ c = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_quadratic_equation_l227_22703


namespace NUMINAMATH_CALUDE_average_transformation_l227_22702

theorem average_transformation (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 2) :
  ((2 * x₁ + 4) + (2 * x₂ + 4) + (2 * x₃ + 4)) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l227_22702


namespace NUMINAMATH_CALUDE_fish_value_in_rice_l227_22723

/-- Represents the trade value of items in terms of bags of rice -/
structure TradeValue where
  fish : ℚ
  bread : ℚ

/-- Defines the trade rates in the distant realm -/
def trade_rates : TradeValue where
  fish := 5⁻¹ * 3 * 6  -- 5 fish = 3 bread, 1 bread = 6 rice
  bread := 6           -- 1 bread = 6 rice

/-- Theorem stating that one fish is equivalent to 3 3/5 bags of rice -/
theorem fish_value_in_rice : trade_rates.fish = 18/5 := by
  sorry

#eval trade_rates.fish

end NUMINAMATH_CALUDE_fish_value_in_rice_l227_22723


namespace NUMINAMATH_CALUDE_triangle_area_l227_22710

/-- Given a triangle with perimeter 24 cm and inradius 2.5 cm, its area is 30 cm². -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) : 
  P = 24 → r = 2.5 → A = r * (P / 2) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l227_22710


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l227_22740

theorem expression_simplification_and_evaluation (x : ℤ) 
  (h1 : 3 * x + 7 > 1) (h2 : 2 * x - 1 < 5) :
  let expr := (x / (x - 1)) / ((x^2 - x) / (x^2 - 2*x + 1)) - (x + 2) / (x + 1)
  (expr = -1 / (x + 1)) ∧ 
  (expr = -1/3 ∨ expr = -1/2 ∨ expr = -1) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l227_22740


namespace NUMINAMATH_CALUDE_ladder_in_alley_l227_22791

/-- In a narrow alley, a ladder of length b is placed between two walls.
    When resting against one wall, it makes a 60° angle with the ground and reaches height s.
    When resting against the other wall, it makes a 70° angle with the ground and reaches height m.
    This theorem states that the width of the alley w is equal to m. -/
theorem ladder_in_alley (w b s m : ℝ) (h1 : 0 < w) (h2 : 0 < b) (h3 : 0 < s) (h4 : 0 < m)
  (h5 : w = b * Real.sin (60 * π / 180))
  (h6 : s = b * Real.sin (60 * π / 180))
  (h7 : w = b * Real.sin (70 * π / 180))
  (h8 : m = b * Real.sin (70 * π / 180)) :
  w = m :=
sorry

end NUMINAMATH_CALUDE_ladder_in_alley_l227_22791


namespace NUMINAMATH_CALUDE_floor_of_2_7_l227_22720

theorem floor_of_2_7 :
  ⌊(2.7 : ℝ)⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_2_7_l227_22720


namespace NUMINAMATH_CALUDE_luke_trivia_rounds_l227_22777

/-- Given that Luke gained 46 points per round and scored 8142 points in total,
    prove that he played 177 rounds. -/
theorem luke_trivia_rounds (points_per_round : ℕ) (total_points : ℕ) 
    (h1 : points_per_round = 46) 
    (h2 : total_points = 8142) : 
  total_points / points_per_round = 177 := by
  sorry

end NUMINAMATH_CALUDE_luke_trivia_rounds_l227_22777


namespace NUMINAMATH_CALUDE_range_of_a_l227_22787

def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

def M (a : ℝ) : Set ℝ := {x | x^2 - (a+2)*x + 2*a ≤ 0}

theorem range_of_a (a : ℝ) : (M a ⊆ A) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l227_22787


namespace NUMINAMATH_CALUDE_multiple_compounds_with_same_oxygen_percentage_l227_22739

/-- Represents a chemical compound -/
structure Compound where
  elements : List String
  massPercentages : List Float
  deriving Repr

/-- Predicate to check if a compound has 57.14% oxygen -/
def hasCorrectOxygenPercentage (c : Compound) : Prop :=
  "O" ∈ c.elements ∧ 
  let oIndex := c.elements.indexOf "O"
  c.massPercentages[oIndex]! = 57.14

/-- Theorem stating that multiple compounds can have 57.14% oxygen -/
theorem multiple_compounds_with_same_oxygen_percentage :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ 
    hasCorrectOxygenPercentage c1 ∧ 
    hasCorrectOxygenPercentage c2 :=
sorry

end NUMINAMATH_CALUDE_multiple_compounds_with_same_oxygen_percentage_l227_22739


namespace NUMINAMATH_CALUDE_composite_sum_of_fourth_power_and_64_power_l227_22711

theorem composite_sum_of_fourth_power_and_64_power (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64^n = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_fourth_power_and_64_power_l227_22711


namespace NUMINAMATH_CALUDE_min_value_polynomial_l227_22785

theorem min_value_polynomial (x : ℝ) :
  ∃ (min : ℝ), min = 2022 - (5 + Real.sqrt 5) / 2 ∧
  ∀ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + y + 2023 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l227_22785


namespace NUMINAMATH_CALUDE_one_shot_each_probability_l227_22746

def yao_rate : ℝ := 0.8
def mcgrady_rate : ℝ := 0.7

theorem one_shot_each_probability :
  let yao_one_shot := 2 * yao_rate * (1 - yao_rate)
  let mcgrady_one_shot := 2 * mcgrady_rate * (1 - mcgrady_rate)
  yao_one_shot * mcgrady_one_shot = 0.1344 := by
sorry

end NUMINAMATH_CALUDE_one_shot_each_probability_l227_22746


namespace NUMINAMATH_CALUDE_fifth_closest_is_park_l227_22759

def buildings := ["bank", "school", "stationery store", "convenience store", "park"]

theorem fifth_closest_is_park :
  buildings.get? 4 = some "park" :=
sorry

end NUMINAMATH_CALUDE_fifth_closest_is_park_l227_22759


namespace NUMINAMATH_CALUDE_symmetric_circle_l227_22768

/-- Given a circle C1 and a line of symmetry, this theorem proves the equation of the symmetric circle C2. -/
theorem symmetric_circle (x y : ℝ) : 
  (∃ C1 : ℝ × ℝ → Prop, C1 = λ (x, y) ↦ (x - 3)^2 + (y + 1)^2 = 1) →
  (∃ L : ℝ × ℝ → Prop, L = λ (x, y) ↦ 2*x - y - 2 = 0) →
  (∃ C2 : ℝ × ℝ → Prop, C2 = λ (x, y) ↦ (x + 1)^2 + (y - 1)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l227_22768


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_equal_l227_22715

/-- Two lines are parallel -/
def parallel (a b : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def perpendicular (a b : Line) : Prop := sorry

/-- The corresponding interior angles on the same side of two lines cut by a third line -/
def corresponding_interior_angles (a b c : Line) : Angle × Angle := sorry

theorem parallel_lines_corresponding_angles_not_equal :
  ∀ (a b c : Line),
  parallel a b →
  ∃ (α β : Angle),
  (α, β) = corresponding_interior_angles a b c ∧
  α ≠ β :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_equal_l227_22715


namespace NUMINAMATH_CALUDE_solve_hash_equation_l227_22705

-- Define the # operation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem solve_hash_equation :
  ∀ X : ℝ, hash X 7 = 290 → X = 17 ∨ X = -17 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_hash_equation_l227_22705


namespace NUMINAMATH_CALUDE_remainder_theorem_l227_22722

theorem remainder_theorem (d : ℚ) : 
  (∃! d, ∀ x, (3 * x^3 + d * x^2 - 6 * x + 25) % (3 * x + 5) = 3) → d = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l227_22722


namespace NUMINAMATH_CALUDE_dan_marbles_l227_22756

/-- Given an initial quantity of marbles and a number of marbles given away,
    calculate the remaining number of marbles. -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that with 64 initial marbles and 14 given away,
    50 marbles remain. -/
theorem dan_marbles : remaining_marbles 64 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_l227_22756


namespace NUMINAMATH_CALUDE_polynomial_b_value_l227_22761

theorem polynomial_b_value (A B : ℤ) : 
  let p := fun z : ℝ => z^4 - 9*z^3 + A*z^2 + B*z + 18
  (∃ r1 r2 r3 r4 : ℕ+, (p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ p r4 = 0) ∧ 
                       (r1 + r2 + r3 + r4 = 9)) →
  B = -20 := by
sorry

end NUMINAMATH_CALUDE_polynomial_b_value_l227_22761


namespace NUMINAMATH_CALUDE_some_number_value_l227_22750

theorem some_number_value (x y : ℝ) 
  (h1 : (27 / 4) * x - 18 = 3 * x + y) 
  (h2 : x = 12) : 
  y = 27 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l227_22750


namespace NUMINAMATH_CALUDE_polynomial_sum_l227_22782

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum (a b c d : ℝ) :
  g a b c d (3*I) = 0 ∧ g a b c d (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l227_22782


namespace NUMINAMATH_CALUDE_problem_statements_l227_22707

theorem problem_statements :
  -- Statement 1
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) ∧
  -- Statement 2
  ∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d ∧
  -- Statement 3
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a - 3) * x + a = 0 ∧ y^2 + (a - 3) * y + a = 0) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l227_22707


namespace NUMINAMATH_CALUDE_square_area_is_four_l227_22789

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a division of the square
structure SquareDivision where
  square : Square
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  sum_areas : area1 + area2 + area3 + area4 = square.side ^ 2
  perpendicular_division : True  -- This is a placeholder for the perpendicular division condition

-- Theorem statement
theorem square_area_is_four 
  (div : SquareDivision) 
  (h1 : div.area1 = 1) 
  (h2 : div.area2 = 1) 
  (h3 : div.area3 = 1) : 
  div.square.side ^ 2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_four_l227_22789


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l227_22724

/-- Proves that given a round trip with total distance 72 miles, total time 7 hours,
    and return speed 18 miles per hour, the outbound speed is 7.2 miles per hour. -/
theorem round_trip_speed_calculation (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 72 ∧ total_time = 7 ∧ return_speed = 18 →
  ∃ outbound_speed : ℝ,
    outbound_speed = 7.2 ∧
    total_distance / 2 / outbound_speed + total_distance / 2 / return_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l227_22724


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l227_22745

theorem repeating_decimal_division (a b : ℚ) :
  a = 45 / 99 →
  b = 18 / 99 →
  a / b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l227_22745


namespace NUMINAMATH_CALUDE_equal_implies_parallel_l227_22701

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem equal_implies_parallel (a b : V) : a = b → parallel a b := by
  sorry

end NUMINAMATH_CALUDE_equal_implies_parallel_l227_22701


namespace NUMINAMATH_CALUDE_email_cleaning_l227_22798

/-- Represents the email cleaning process and proves the number of emails deleted in the first round -/
theorem email_cleaning (initial_emails : ℕ) : 
  -- After first round, emails remain the same (deleted some, received 15)
  ∃ (first_round_deleted : ℕ), initial_emails = initial_emails - first_round_deleted + 15 →
  -- After second round, 20 deleted, 5 received
  initial_emails - 20 + 5 = 30 →
  -- Final inbox has 30 emails (15 + 5 + 10 new ones)
  30 = 15 + 5 + 10 →
  -- Prove that first_round_deleted is 0
  first_round_deleted = 0 := by
  sorry

end NUMINAMATH_CALUDE_email_cleaning_l227_22798


namespace NUMINAMATH_CALUDE_eric_blue_marbles_l227_22793

theorem eric_blue_marbles :
  let total_marbles : ℕ := 20
  let white_marbles : ℕ := 12
  let green_marbles : ℕ := 2
  let blue_marbles : ℕ := total_marbles - white_marbles - green_marbles
  blue_marbles = 6 := by sorry

end NUMINAMATH_CALUDE_eric_blue_marbles_l227_22793


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l227_22733

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 5 x = Nat.choose 5 2) → (x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l227_22733


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l227_22716

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (a^3 - 4*a^2 + 50*a - 7 = 0) →
  (b^3 - 4*b^2 + 50*b - 7 = 0) →
  (c^3 - 4*c^2 + 50*c - 7 = 0) →
  (a+b+1)^3 + (b+c+1)^3 + (c+a+1)^3 = 991 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l227_22716


namespace NUMINAMATH_CALUDE_normal_block_volume_l227_22783

/-- The volume of a normal block of cheese -/
def normal_volume : ℝ := sorry

/-- The volume of a large block of cheese -/
def large_volume : ℝ := 36

/-- The relationship between large and normal block volumes -/
axiom volume_relationship : large_volume = 12 * normal_volume

theorem normal_block_volume : normal_volume = 3 := by sorry

end NUMINAMATH_CALUDE_normal_block_volume_l227_22783


namespace NUMINAMATH_CALUDE_count_valid_a_l227_22730

theorem count_valid_a : ∃! (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (X : Finset ℤ), (∀ x ∈ X, 6*x - 5 ≥ a ∧ x/4 - (x-1)/6 < 1/2) ∧ X.card = 2) ∧
             (∃ y : ℚ, y > 0 ∧ 4*y - 3*a = 2*(y-3))) ∧
  S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_a_l227_22730


namespace NUMINAMATH_CALUDE_bicycle_sampling_is_systematic_l227_22712

-- Define the sampling method
structure SamplingMethod where
  location : String
  selectionCriteria : String

-- Define systematic sampling
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.location = "main road" ∧ 
  method.selectionCriteria = "6-digit license plate numbers"

-- Define the specific sampling method used in the problem
def bicycleSamplingMethod : SamplingMethod :=
  { location := "main road"
  , selectionCriteria := "6-digit license plate numbers" }

-- Theorem statement
theorem bicycle_sampling_is_systematic :
  isSystematicSampling bicycleSamplingMethod :=
by sorry


end NUMINAMATH_CALUDE_bicycle_sampling_is_systematic_l227_22712


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l227_22775

theorem gcd_of_three_numbers : Nat.gcd 4560 (Nat.gcd 6080 16560) = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l227_22775


namespace NUMINAMATH_CALUDE_calories_per_dollar_difference_l227_22757

-- Define the given conditions
def burrito_count : ℕ := 10
def burrito_price : ℚ := 6
def burrito_calories : ℕ := 120
def burger_count : ℕ := 5
def burger_price : ℚ := 8
def burger_calories : ℕ := 400

-- Define the theorem
theorem calories_per_dollar_difference :
  (burger_count * burger_calories : ℚ) / burger_price -
  (burrito_count * burrito_calories : ℚ) / burrito_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_dollar_difference_l227_22757


namespace NUMINAMATH_CALUDE_max_omega_for_increasing_g_l227_22758

/-- Given a function f and its translation g, proves that the maximum value of ω is 2 
    when g is increasing on [0, π/4] -/
theorem max_omega_for_increasing_g (ω : ℝ) (f g : ℝ → ℝ) : 
  ω > 0 → 
  (∀ x, f x = 2 * Real.sin (ω * x - π / 8)) →
  (∀ x, g x = f (x + π / (8 * ω))) →
  (∀ x ∈ Set.Icc 0 (π / 4), Monotone g) →
  ω ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_omega_for_increasing_g_l227_22758


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l227_22717

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n) →
  (∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l227_22717


namespace NUMINAMATH_CALUDE_ceiling_product_sqrt_l227_22779

theorem ceiling_product_sqrt : ⌈Real.sqrt 3⌉ * ⌈Real.sqrt 12⌉ * ⌈Real.sqrt 120⌉ = 88 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_sqrt_l227_22779


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l227_22729

theorem smallest_integer_satisfying_inequality : 
  (∃ (x : ℤ), x / 4 + 3 / 7 > 2 / 3 ∧ ∀ (y : ℤ), y < x → y / 4 + 3 / 7 ≤ 2 / 3) ∧
  (∀ (x : ℤ), x / 4 + 3 / 7 > 2 / 3 → x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l227_22729


namespace NUMINAMATH_CALUDE_inequality_proof_l227_22718

theorem inequality_proof (m n : ℝ) (h : m > n) : 1 - 2*m < 1 - 2*n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l227_22718


namespace NUMINAMATH_CALUDE_class_gpa_theorem_l227_22776

/-- The grade point average (GPA) of a class -/
def classGPA (n : ℕ) (gpa1 : ℚ) (gpa2 : ℚ) : ℚ :=
  (1 / 3 : ℚ) * gpa1 + (2 / 3 : ℚ) * gpa2

/-- Theorem: The GPA of a class where one-third has a GPA of 45 and two-thirds has a GPA of 60 is 55 -/
theorem class_gpa_theorem :
  classGPA 3 45 60 = 55 := by
  sorry

end NUMINAMATH_CALUDE_class_gpa_theorem_l227_22776


namespace NUMINAMATH_CALUDE_steven_peach_count_l227_22752

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := 9

/-- The difference in peaches between Steven and Jake -/
def steven_jake_diff : ℕ := 7

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := jake_peaches + steven_jake_diff

theorem steven_peach_count : steven_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_count_l227_22752


namespace NUMINAMATH_CALUDE_seokmin_school_cookies_l227_22792

/-- The number of boxes of cookies needed for a given number of students -/
def cookies_boxes_needed (num_students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) : ℕ :=
  ((num_students * cookies_per_student + cookies_per_box - 1) / cookies_per_box : ℕ)

/-- Theorem stating the number of boxes needed for Seokmin's school -/
theorem seokmin_school_cookies :
  cookies_boxes_needed 134 7 28 = 34 := by
  sorry

end NUMINAMATH_CALUDE_seokmin_school_cookies_l227_22792


namespace NUMINAMATH_CALUDE_millet_majority_on_day_three_l227_22762

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Real
  other : Real

/-- Calculates the next day's feeder state based on the current state -/
def nextDay (state : FeederState) : FeederState :=
  let remainingMillet := state.millet * 0.8
  let newMillet := if state.day = 1 then 0.5 else 0.4
  { day := state.day + 1,
    millet := remainingMillet + newMillet,
    other := 0.6 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 0.4, other := 0.6 }

/-- Theorem stating that on Day 3, more than half of the seeds are millet -/
theorem millet_majority_on_day_three :
  let day3State := nextDay (nextDay initialState)
  day3State.millet / (day3State.millet + day3State.other) > 0.5 := by
  sorry


end NUMINAMATH_CALUDE_millet_majority_on_day_three_l227_22762


namespace NUMINAMATH_CALUDE_ducks_and_dogs_total_l227_22765

theorem ducks_and_dogs_total (d g : ℕ) : 
  d = g + 2 →                   -- number of ducks is 2 more than dogs
  4 * g - 2 * d = 10 →          -- dogs have 10 more legs than ducks
  d + g = 16 := by              -- total number of ducks and dogs is 16
sorry

end NUMINAMATH_CALUDE_ducks_and_dogs_total_l227_22765


namespace NUMINAMATH_CALUDE_problem_solution_l227_22708

theorem problem_solution :
  (195 * 205 = 39975) ∧
  (9 * 11 * 101 * 10001 = 99999999) ∧
  (∀ a : ℝ, a^2 - 6*a + 8 = (a - 2)*(a - 4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l227_22708


namespace NUMINAMATH_CALUDE_isosceles_in_26gon_l227_22760

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsoscelesTriangle (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- Main theorem: In a regular 26-gon, any 9 vertices contain an isosceles triangle -/
theorem isosceles_in_26gon (p : RegularPolygon 26) 
  (vertices : Finset (Fin 26)) (h : vertices.card = 9) :
  ∃ (v1 v2 v3 : Fin 26), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v3 ∈ vertices ∧
    IsIsoscelesTriangle p v1 v2 v3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_in_26gon_l227_22760


namespace NUMINAMATH_CALUDE_ripe_apples_theorem_l227_22726

-- Define the universe of discourse
variable (Basket : Type)
-- Define the property of being ripe
variable (isRipe : Basket → Prop)

-- Define the statement "All apples in this basket are ripe" is false
axiom not_all_ripe : ¬(∀ (apple : Basket), isRipe apple)

-- Theorem to prove
theorem ripe_apples_theorem :
  (∃ (apple : Basket), ¬(isRipe apple)) ∧
  (¬(∀ (apple : Basket), isRipe apple)) := by
  sorry

end NUMINAMATH_CALUDE_ripe_apples_theorem_l227_22726


namespace NUMINAMATH_CALUDE_binomial_60_3_l227_22754

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l227_22754


namespace NUMINAMATH_CALUDE_pizza_percentage_left_l227_22755

theorem pizza_percentage_left (ravindra_ate hongshu_ate pizza_left : ℚ) : 
  ravindra_ate = 2/5 →
  hongshu_ate = ravindra_ate/2 →
  pizza_left = 1 - (ravindra_ate + hongshu_ate) →
  pizza_left * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_pizza_percentage_left_l227_22755


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l227_22731

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 4 → a 3 = 2 ∨ a 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l227_22731


namespace NUMINAMATH_CALUDE_tangent_two_identities_l227_22749

open Real

theorem tangent_two_identities (α : ℝ) (h : tan α = 2) :
  (2 * sin α + 2 * cos α) / (sin α - cos α) = 8 ∧
  (cos (π - α) * cos (π / 2 + α) * sin (α - 3 * π / 2)) /
  (sin (3 * π + α) * sin (α - π) * cos (π + α)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_identities_l227_22749


namespace NUMINAMATH_CALUDE_meeting_participants_l227_22774

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 125 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_l227_22774


namespace NUMINAMATH_CALUDE_omega_range_l227_22786

/-- Given a function f(x) = sin(ωx + π/4) where ω > 0, 
    if f(x) is monotonically decreasing in the interval (π/2, π),
    then 1/2 ≤ ω ≤ 5/4 -/
theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 4)
  (∀ x ∈ Set.Ioo (π / 2) π, ∀ y ∈ Set.Ioo (π / 2) π, x < y → f x > f y) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_omega_range_l227_22786


namespace NUMINAMATH_CALUDE_max_product_843_l227_22753

def digits : List Nat := [1, 3, 4, 5, 7, 8]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit_number (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  (three_digit_number a b c) * (two_digit_number d e)

theorem max_product_843 :
  ∀ a b c d e,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 4 3 7 5 :=
sorry

end NUMINAMATH_CALUDE_max_product_843_l227_22753


namespace NUMINAMATH_CALUDE_sales_solution_l227_22770

def sales_problem (m1 m3 m4 m5 m6 avg : ℕ) : Prop :=
  ∃ m2 : ℕ, 
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg ∧
    m2 = 5744

theorem sales_solution :
  sales_problem 5266 5864 6122 6588 4916 5750 :=
by sorry

end NUMINAMATH_CALUDE_sales_solution_l227_22770


namespace NUMINAMATH_CALUDE_minimum_married_men_l227_22781

theorem minimum_married_men (total_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (ac_men : ℕ) (married_with_all : ℕ)
  (h_total : total_men = 100)
  (h_tv : tv_men = 75)
  (h_radio : radio_men = 85)
  (h_ac : ac_men = 70)
  (h_married_all : married_with_all = 11)
  (h_tv_le : tv_men ≤ total_men)
  (h_radio_le : radio_men ≤ total_men)
  (h_ac_le : ac_men ≤ total_men)
  (h_married_all_le : married_with_all ≤ tv_men ∧ married_with_all ≤ radio_men ∧ married_with_all ≤ ac_men) :
  ∃ (married_men : ℕ), married_men ≥ married_with_all ∧ married_men ≤ total_men := by
  sorry

end NUMINAMATH_CALUDE_minimum_married_men_l227_22781


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l227_22727

/-- 
Given a canoe that rows downstream at 12 km/hr and a stream with a speed of 4.5 km/hr,
prove that the speed of the canoe when rowing upstream is 3 km/hr.
-/
theorem canoe_upstream_speed : 
  ∀ (downstream_speed stream_speed : ℝ),
  downstream_speed = 12 →
  stream_speed = 4.5 →
  (downstream_speed - 2 * stream_speed) = 3 :=
by sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l227_22727


namespace NUMINAMATH_CALUDE_largest_prime_diff_144_l227_22744

/-- Two natural numbers are considered different if they are not equal -/
def Different (a b : ℕ) : Prop := a ≠ b

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself -/
def IsPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ p → d = 1 ∨ d = p

/-- The statement that the largest possible difference between two different primes summing to 144 is 134 -/
theorem largest_prime_diff_144 : 
  ∃ (p q : ℕ), Different p q ∧ IsPrime p ∧ IsPrime q ∧ p + q = 144 ∧ 
  (∀ (r s : ℕ), Different r s → IsPrime r → IsPrime s → r + s = 144 → s - r ≤ 134) ∧
  q - p = 134 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_diff_144_l227_22744


namespace NUMINAMATH_CALUDE_art_price_increase_theorem_l227_22709

/-- Calculates the price increase of an art piece given its initial price and a multiplier for its future price. -/
def art_price_increase (initial_price : ℕ) (price_multiplier : ℕ) : ℕ :=
  (price_multiplier * initial_price) - initial_price

/-- Theorem stating that for an art piece with an initial price of $4000 and a future price 3 times the initial price, the price increase is $8000. -/
theorem art_price_increase_theorem :
  art_price_increase 4000 3 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_art_price_increase_theorem_l227_22709


namespace NUMINAMATH_CALUDE_S_independent_of_position_l227_22799

/-- The sum of distances from vertices of an n-gon to the nearest vertices of an (n-1)-gon -/
def S (n : ℕ) (θ : ℝ) : ℝ := sorry

/-- The theorem stating that S depends only on n and not on the relative position θ -/
theorem S_independent_of_position (n : ℕ) (h1 : n ≥ 4) (h2 : Even n) (θ₁ θ₂ : ℝ) :
  S n θ₁ = S n θ₂ := by sorry

end NUMINAMATH_CALUDE_S_independent_of_position_l227_22799


namespace NUMINAMATH_CALUDE_exponential_function_point_l227_22737

theorem exponential_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^x) 2 = 9 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_point_l227_22737


namespace NUMINAMATH_CALUDE_range_of_t_l227_22743

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x ≤ t ∧ x^2 - 4*x + t ≤ 0) → 
  0 ≤ t ∧ t ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_t_l227_22743


namespace NUMINAMATH_CALUDE_parabola_intersection_l227_22741

/-- Parabola 1 function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 1

/-- Parabola 2 function -/
def g (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 1

/-- Theorem stating that (0, 1) and (-8, 233) are the only intersection points -/
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233) := by
  sorry

#check parabola_intersection

end NUMINAMATH_CALUDE_parabola_intersection_l227_22741


namespace NUMINAMATH_CALUDE_value_of_x_l227_22773

theorem value_of_x :
  ∀ (x y z w u : ℤ),
    x = y + 3 →
    y = z + 15 →
    z = w + 25 →
    w = u + 10 →
    u = 90 →
    x = 143 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l227_22773


namespace NUMINAMATH_CALUDE_berts_spending_l227_22769

theorem berts_spending (initial_amount : ℚ) : 
  initial_amount = 44 →
  let hardware_spent := (1 / 4) * initial_amount
  let after_hardware := initial_amount - hardware_spent
  let after_drycleaner := after_hardware - 9
  let grocery_spent := (1 / 2) * after_drycleaner
  let final_amount := after_drycleaner - grocery_spent
  final_amount = 12 := by sorry

end NUMINAMATH_CALUDE_berts_spending_l227_22769


namespace NUMINAMATH_CALUDE_original_number_is_point_three_l227_22738

theorem original_number_is_point_three : 
  ∃ x : ℝ, (10 * x = x + 2.7) ∧ (x = 0.3) := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_point_three_l227_22738


namespace NUMINAMATH_CALUDE_max_value_of_a_l227_22735

open Real

theorem max_value_of_a : 
  (∀ x > 0, Real.exp (x - 1) + 1 ≥ a + Real.log x) → 
  (∀ b, (∀ x > 0, Real.exp (x - 1) + 1 ≥ b + Real.log x) → b ≤ a) → 
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l227_22735


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_13_l227_22721

theorem unique_x_with_three_prime_divisors_including_13 :
  ∀ (x n : ℕ),
    x = 9^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
    13 ∣ x →
    x = 728 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_13_l227_22721


namespace NUMINAMATH_CALUDE_small_poster_price_is_six_l227_22732

/-- Represents Laran's poster business --/
structure PosterBusiness where
  total_posters_per_day : ℕ
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_cost : ℕ
  weekly_profit : ℕ
  days_per_week : ℕ

/-- Calculates the selling price of small posters --/
def small_poster_price (business : PosterBusiness) : ℕ :=
  let small_posters_per_day := business.total_posters_per_day - business.large_posters_per_day
  let daily_profit := business.weekly_profit / business.days_per_week
  let large_poster_profit := business.large_poster_price - business.large_poster_cost
  let daily_large_poster_profit := large_poster_profit * business.large_posters_per_day
  let daily_small_poster_profit := daily_profit - daily_large_poster_profit
  let small_poster_profit := daily_small_poster_profit / small_posters_per_day
  small_poster_profit + business.small_poster_cost

/-- Theorem stating that the small poster price is $6 --/
theorem small_poster_price_is_six (business : PosterBusiness) 
    (h1 : business.total_posters_per_day = 5)
    (h2 : business.large_posters_per_day = 2)
    (h3 : business.large_poster_price = 10)
    (h4 : business.large_poster_cost = 5)
    (h5 : business.small_poster_cost = 3)
    (h6 : business.weekly_profit = 95)
    (h7 : business.days_per_week = 5) :
  small_poster_price business = 6 := by
  sorry

#eval small_poster_price {
  total_posters_per_day := 5,
  large_posters_per_day := 2,
  large_poster_price := 10,
  large_poster_cost := 5,
  small_poster_cost := 3,
  weekly_profit := 95,
  days_per_week := 5
}

end NUMINAMATH_CALUDE_small_poster_price_is_six_l227_22732


namespace NUMINAMATH_CALUDE_calculate_expression_l227_22788

theorem calculate_expression : (3.6 * 0.5) / 0.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l227_22788


namespace NUMINAMATH_CALUDE_cos_2017_pi_thirds_l227_22736

theorem cos_2017_pi_thirds : Real.cos (2017 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2017_pi_thirds_l227_22736


namespace NUMINAMATH_CALUDE_exist_50_integers_with_equal_sum_l227_22751

/-- Sum of digits function -/
def S (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + S (n / 10)

/-- Theorem statement -/
theorem exist_50_integers_with_equal_sum :
  ∃ (n : Fin 50 → ℕ), (∀ i j, i < j → n i < n j) ∧
    (∀ i j, i < j → n i + S (n i) = n j + S (n j)) :=
sorry

end NUMINAMATH_CALUDE_exist_50_integers_with_equal_sum_l227_22751


namespace NUMINAMATH_CALUDE_product_990_sum_93_l227_22742

theorem product_990_sum_93 : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧ 
  (a * b = 990) ∧ 
  (x * y * z = 990) ∧ 
  (a + b + x + y + z = 93) := by
sorry

end NUMINAMATH_CALUDE_product_990_sum_93_l227_22742


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l227_22706

theorem abs_sum_inequality (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ x ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l227_22706


namespace NUMINAMATH_CALUDE_tournament_scheduling_correct_l227_22772

/-- Represents a team in the tournament --/
inductive Team (n : ℕ) where
  | num : Fin n → Team n
  | inf : Team n

/-- A match between two teams --/
structure Match (n : ℕ) where
  team1 : Team n
  team2 : Team n

/-- A round in the tournament --/
def Round (n : ℕ) := List (Match n)

/-- Generate the next round based on the current round --/
def nextRound (n : ℕ) (current : Round n) : Round n :=
  sorry

/-- Check if a round is valid (each team plays exactly once) --/
def isValidRound (n : ℕ) (round : Round n) : Prop :=
  sorry

/-- Check if two teams have played against each other --/
def havePlayedAgainst (n : ℕ) (team1 team2 : Team n) (rounds : List (Round n)) : Prop :=
  sorry

/-- The main theorem: tournament scheduling is correct --/
theorem tournament_scheduling_correct (n : ℕ) (h : n > 1) :
  ∃ (rounds : List (Round n)),
    (rounds.length = n - 1) ∧
    (∀ r ∈ rounds, isValidRound n r) ∧
    (∀ t1 t2 : Team n, t1 ≠ t2 → havePlayedAgainst n t1 t2 rounds) :=
  sorry

end NUMINAMATH_CALUDE_tournament_scheduling_correct_l227_22772


namespace NUMINAMATH_CALUDE_regular_polyhedra_symmetry_axes_l227_22766

-- Define the types of regular polyhedra
inductive RegularPolyhedron
  | Tetrahedron
  | Hexahedron
  | Octahedron
  | Dodecahedron
  | Icosahedron

-- Define a structure for symmetry axis information
structure SymmetryAxis where
  order : ℕ
  count : ℕ

-- Define a function that returns the symmetry axes for a given polyhedron
def symmetryAxes (p : RegularPolyhedron) : List SymmetryAxis :=
  match p with
  | RegularPolyhedron.Tetrahedron => [
      { order := 3, count := 4 },
      { order := 2, count := 3 }
    ]
  | RegularPolyhedron.Hexahedron => [
      { order := 4, count := 3 },
      { order := 3, count := 4 },
      { order := 2, count := 6 }
    ]
  | RegularPolyhedron.Octahedron => [
      { order := 4, count := 3 },
      { order := 3, count := 4 },
      { order := 2, count := 6 }
    ]
  | RegularPolyhedron.Dodecahedron => [
      { order := 5, count := 6 },
      { order := 3, count := 10 },
      { order := 2, count := 15 }
    ]
  | RegularPolyhedron.Icosahedron => [
      { order := 5, count := 6 },
      { order := 3, count := 10 },
      { order := 2, count := 15 }
    ]

-- Theorem stating that the symmetry axes for each polyhedron are correct
theorem regular_polyhedra_symmetry_axes :
  ∀ p : RegularPolyhedron, 
    (symmetryAxes p).length > 0 ∧
    (∀ axis ∈ symmetryAxes p, axis.order ≥ 2 ∧ axis.count > 0) :=
by sorry

end NUMINAMATH_CALUDE_regular_polyhedra_symmetry_axes_l227_22766


namespace NUMINAMATH_CALUDE_horner_v2_equals_22_l227_22728

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => x * acc + a) 0

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 -/
def f (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

/-- The coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 0, 6, 0, 9, 0, 208]

/-- Theorem: v₂ = 22 when evaluating f(x) at x = -4 using Horner's method -/
theorem horner_v2_equals_22 :
  let x := -4
  let v₀ := 208
  let v₁ := x * v₀ + 0
  let v₂ := x * v₁ + 9
  v₂ = 22 := by sorry

end NUMINAMATH_CALUDE_horner_v2_equals_22_l227_22728


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l227_22764

/-- Given that (m-2)x^|m| - bx - 1 = 0 is a quadratic equation in x, prove that m = -2 -/
theorem quadratic_equation_m_value (m b : ℝ) : 
  (∀ x, ∃ a c : ℝ, (m - 2) * x^(|m|) - b*x - 1 = a*x^2 + b*x + c) → 
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l227_22764


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l227_22725

theorem arithmetic_mean_difference (p q r : ℝ) 
  (mean_pq : (p + q) / 2 = 10)
  (mean_qr : (q + r) / 2 = 22) :
  r - p = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l227_22725


namespace NUMINAMATH_CALUDE_max_type_A_books_l227_22763

/-- Represents the unit price of type A books -/
def price_A : ℝ := 20

/-- Represents the unit price of type B books -/
def price_B : ℝ := 15

/-- Represents the total number of books to be purchased -/
def total_books : ℕ := 300

/-- Represents the discount factor for type A books -/
def discount_A : ℝ := 0.9

/-- Represents the maximum total cost -/
def max_cost : ℝ := 5100

/-- Theorem stating the maximum number of type A books that can be purchased -/
theorem max_type_A_books : 
  ∃ (n : ℕ), n ≤ total_books ∧ 
  discount_A * price_A * n + price_B * (total_books - n) ≤ max_cost ∧
  ∀ (m : ℕ), m > n → discount_A * price_A * m + price_B * (total_books - m) > max_cost :=
sorry

end NUMINAMATH_CALUDE_max_type_A_books_l227_22763


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l227_22780

/-- Proves that the cost per mile for a car rental is $0.20 given specific conditions --/
theorem car_rental_cost_per_mile 
  (daily_fee : ℝ) 
  (daily_budget : ℝ) 
  (max_distance : ℝ) 
  (h1 : daily_fee = 50) 
  (h2 : daily_budget = 88) 
  (h3 : max_distance = 190) : 
  ∃ (cost_per_mile : ℝ), 
    cost_per_mile = 0.20 ∧ 
    daily_fee + cost_per_mile * max_distance = daily_budget :=
by
  sorry

end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l227_22780


namespace NUMINAMATH_CALUDE_angle_triple_complement_l227_22771

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l227_22771


namespace NUMINAMATH_CALUDE_c_can_be_any_real_l227_22795

theorem c_can_be_any_real (a b c d : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : a / b + c / d < 0) :
  ∃ (a' b' d' : ℝ) (h1' : b' ≠ 0) (h2' : d' ≠ 0),
    (∀ c' : ℝ, ∃ (h3' : a' / b' + c' / d' < 0), True) :=
sorry

end NUMINAMATH_CALUDE_c_can_be_any_real_l227_22795


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l227_22719

theorem quadruple_equation_solutions :
  let equation (a b c d : ℕ) := 2*a + 2*b + 2*c + 2*d = d^2 - c^2 + b^2 - a^2
  ∀ (a b c d : ℕ), a < b → b < c → c < d →
  (
    (equation 2 4 5 7) ∧
    (∀ x : ℕ, equation (2*x) (2*x+2) (2*x+4) (2*x+6))
  ) := by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l227_22719


namespace NUMINAMATH_CALUDE_problem_1_l227_22796

theorem problem_1 (m : ℝ) : m * m^3 + (-m^2)^3 / m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l227_22796


namespace NUMINAMATH_CALUDE_abs_gt_iff_square_gt_l227_22714

theorem abs_gt_iff_square_gt (x y : ℝ) : |x| > |y| ↔ x^2 > y^2 := by
  sorry

end NUMINAMATH_CALUDE_abs_gt_iff_square_gt_l227_22714


namespace NUMINAMATH_CALUDE_equation_real_roots_range_l227_22767

-- Define the equation
def equation (x m : ℝ) : ℝ := 25 - |x + 1| - 4 * 5 - |x + 1| - m

-- Define the property of having real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, equation x m = 0

-- Theorem statement
theorem equation_real_roots_range :
  ∀ m : ℝ, has_real_roots m ↔ m ∈ Set.Ioo (-3 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_equation_real_roots_range_l227_22767


namespace NUMINAMATH_CALUDE_gcf_of_96_144_240_l227_22734

theorem gcf_of_96_144_240 : Nat.gcd 96 (Nat.gcd 144 240) = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_96_144_240_l227_22734


namespace NUMINAMATH_CALUDE_anna_phone_chargers_l227_22713

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

theorem anna_phone_chargers :
  (laptop_chargers = 5 * phone_chargers) →
  (phone_chargers + laptop_chargers = total_chargers) →
  phone_chargers = 4 := by
  sorry

end NUMINAMATH_CALUDE_anna_phone_chargers_l227_22713


namespace NUMINAMATH_CALUDE_students_playing_sports_l227_22747

theorem students_playing_sports (A B : Finset ℕ) : 
  A.card = 7 → B.card = 8 → (A ∩ B).card = 3 → (A ∪ B).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_sports_l227_22747


namespace NUMINAMATH_CALUDE_john_chips_consumption_l227_22794

/-- The number of bags of chips John eats for dinner -/
def dinner_chips : ℕ := 1

/-- The number of bags of chips John eats after dinner -/
def after_dinner_chips : ℕ := 2 * dinner_chips

/-- The total number of bags of chips John eats -/
def total_chips : ℕ := dinner_chips + after_dinner_chips

theorem john_chips_consumption :
  total_chips = 3 :=
sorry

end NUMINAMATH_CALUDE_john_chips_consumption_l227_22794


namespace NUMINAMATH_CALUDE_root_product_squared_plus_one_l227_22748

theorem root_product_squared_plus_one (a b c : ℂ) : 
  (a^3 + 20*a^2 + a + 5 = 0) →
  (b^3 + 20*b^2 + b + 5 = 0) →
  (c^3 + 20*c^2 + c + 5 = 0) →
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) = 229 := by
  sorry

end NUMINAMATH_CALUDE_root_product_squared_plus_one_l227_22748


namespace NUMINAMATH_CALUDE_dartboard_angle_l227_22790

/-- Given a circular dartboard, if the probability of a dart landing in a particular region is 1/4,
    then the measure of the central angle of that region is 90 degrees. -/
theorem dartboard_angle (probability : ℝ) (angle : ℝ) :
  probability = 1/4 →
  angle = probability * 360 →
  angle = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_angle_l227_22790
