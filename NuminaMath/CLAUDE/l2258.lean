import Mathlib

namespace NUMINAMATH_CALUDE_language_interview_probability_l2258_225807

theorem language_interview_probability 
  (total_students : ℕ) 
  (french_students : ℕ) 
  (spanish_students : ℕ) 
  (both_languages : ℕ) 
  (h1 : total_students = 28)
  (h2 : french_students = 20)
  (h3 : spanish_students = 23)
  (h4 : both_languages = 17)
  (h5 : both_languages ≤ french_students)
  (h6 : both_languages ≤ spanish_students)
  (h7 : french_students ≤ total_students)
  (h8 : spanish_students ≤ total_students) :
  (1 : ℚ) - (Nat.choose (french_students - both_languages + (spanish_students - both_languages)) 2 : ℚ) / (Nat.choose total_students 2) = 20 / 21 :=
sorry

end NUMINAMATH_CALUDE_language_interview_probability_l2258_225807


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2258_225814

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The difference between the probability of 4 heads in 5 flips and 5 heads in 5 flips -/
def prob_difference : ℚ :=
  prob_k_heads 5 4 - prob_k_heads 5 5

theorem coin_flip_probability : prob_difference = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2258_225814


namespace NUMINAMATH_CALUDE_jim_purchase_total_l2258_225829

/-- Calculate the total amount Jim paid for lamps and bulbs --/
theorem jim_purchase_total : 
  let lamp_cost : ℚ := 7
  let bulb_cost : ℚ := lamp_cost - 4
  let lamp_quantity : ℕ := 2
  let bulb_quantity : ℕ := 6
  let tax_rate : ℚ := 5 / 100
  let bulb_discount : ℚ := 10 / 100
  let total_lamp_cost : ℚ := lamp_cost * lamp_quantity
  let total_bulb_cost : ℚ := bulb_cost * bulb_quantity
  let discounted_bulb_cost : ℚ := total_bulb_cost * (1 - bulb_discount)
  let subtotal : ℚ := total_lamp_cost + discounted_bulb_cost
  let tax_amount : ℚ := subtotal * tax_rate
  let total_cost : ℚ := subtotal + tax_amount
  total_cost = 3171 / 100 := by sorry

end NUMINAMATH_CALUDE_jim_purchase_total_l2258_225829


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2258_225806

theorem quadratic_coefficient (b : ℝ) (m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/4 = (x + m)^2 + 1/8) → 
  b = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2258_225806


namespace NUMINAMATH_CALUDE_anne_weighs_67_pounds_l2258_225841

/-- Anne's weight in pounds -/
def anne_weight : ℕ := sorry

/-- Douglas' weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference between Anne's and Douglas' weights in pounds -/
def weight_difference : ℕ := 15

/-- Theorem: Anne weighs 67 pounds -/
theorem anne_weighs_67_pounds : anne_weight = 67 := by
  sorry

end NUMINAMATH_CALUDE_anne_weighs_67_pounds_l2258_225841


namespace NUMINAMATH_CALUDE_linear_system_solution_l2258_225852

theorem linear_system_solution (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (given : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2258_225852


namespace NUMINAMATH_CALUDE_exponential_inequality_l2258_225874

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  3^a + 2*a = 3^b + 3*b → a > b :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2258_225874


namespace NUMINAMATH_CALUDE_age_problem_l2258_225845

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 12 → 
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2258_225845


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l2258_225859

theorem sqrt_2x_minus_1_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 1) → x ≥ (1 / 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l2258_225859


namespace NUMINAMATH_CALUDE_grasshopper_position_l2258_225887

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point
| seven : Point

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true
  | Point.six => false
  | Point.seven => true

/-- Represents a single jump of the grasshopper -/
def jump (p : Point) : Point :=
  match p with
  | Point.one => Point.seven
  | Point.two => Point.seven
  | Point.three => Point.two
  | Point.four => Point.two
  | Point.five => Point.four
  | Point.six => Point.four
  | Point.seven => Point.six

/-- Represents multiple jumps of the grasshopper -/
def multi_jump (p : Point) (n : Nat) : Point :=
  match n with
  | 0 => p
  | Nat.succ m => jump (multi_jump p m)

theorem grasshopper_position : multi_jump Point.seven 2011 = Point.two := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_position_l2258_225887


namespace NUMINAMATH_CALUDE_negative_abs_negative_eight_l2258_225894

theorem negative_abs_negative_eight : -|-8| = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_eight_l2258_225894


namespace NUMINAMATH_CALUDE_additive_function_properties_l2258_225875

/-- A function f: ℝ → ℝ satisfying f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_function_properties (f : ℝ → ℝ) (hf : AdditiveFunction f) :
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by sorry

end NUMINAMATH_CALUDE_additive_function_properties_l2258_225875


namespace NUMINAMATH_CALUDE_max_distinct_substrings_l2258_225889

/-- Represents the length of the string -/
def stringLength : ℕ := 66

/-- Represents the number of distinct letters in the string -/
def distinctLetters : ℕ := 4

/-- Calculates the sum of an arithmetic series -/
def arithmeticSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem: The maximum number of distinct substrings in a string of length 66
    composed of 4 distinct letters is 2100 -/
theorem max_distinct_substrings :
  distinctLetters +
  distinctLetters^2 +
  (arithmeticSum (stringLength - 2) - arithmeticSum (distinctLetters - 1)) = 2100 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_substrings_l2258_225889


namespace NUMINAMATH_CALUDE_municipal_hiring_problem_l2258_225892

theorem municipal_hiring_problem (U P : Finset ℕ) 
  (h1 : U.card = 120)
  (h2 : P.card = 98)
  (h3 : (U ∩ P).card = 40) :
  (U ∪ P).card = 218 := by
sorry

end NUMINAMATH_CALUDE_municipal_hiring_problem_l2258_225892


namespace NUMINAMATH_CALUDE_simplify_expression_l2258_225815

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2258_225815


namespace NUMINAMATH_CALUDE_round_repeating_decimal_to_thousandth_l2258_225854

/-- Represents a repeating decimal where the whole number part is 67 and the repeating part is 836 -/
def repeating_decimal : ℚ := 67 + 836 / 999

/-- Rounding function to the nearest thousandth -/
def round_to_thousandth (x : ℚ) : ℚ := 
  (⌊x * 1000 + 1/2⌋ : ℚ) / 1000

theorem round_repeating_decimal_to_thousandth :
  round_to_thousandth repeating_decimal = 67837 / 1000 := by sorry

end NUMINAMATH_CALUDE_round_repeating_decimal_to_thousandth_l2258_225854


namespace NUMINAMATH_CALUDE_population_decrease_rate_l2258_225843

theorem population_decrease_rate (initial_population : ℝ) (final_population : ℝ) (years : ℕ) 
  (h1 : initial_population = 8000)
  (h2 : final_population = 3920)
  (h3 : years = 2) :
  ∃ (rate : ℝ), initial_population * (1 - rate)^years = final_population ∧ rate = 0.3 := by
sorry

end NUMINAMATH_CALUDE_population_decrease_rate_l2258_225843


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l2258_225838

theorem trig_identity_simplification (x : ℝ) : 
  Real.sin (x + Real.pi / 3) + 2 * Real.sin (x - Real.pi / 3) - Real.sqrt 3 * Real.cos (2 * Real.pi / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l2258_225838


namespace NUMINAMATH_CALUDE_sector_max_area_l2258_225817

/-- Given a sector with perimeter 40, its maximum area is 100 -/
theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 40) :
  (1 / 2) * l * r ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l2258_225817


namespace NUMINAMATH_CALUDE_number_puzzle_l2258_225891

theorem number_puzzle (N A : ℝ) : N = 295 ∧ N / 5 + A = 65 → A = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2258_225891


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l2258_225857

/-- Converts a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ := sorry

/-- Converts a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String := sorry

theorem binary_multiplication_division :
  let a := binary_to_nat "1011010"
  let b := binary_to_nat "1010100"
  let c := binary_to_nat "100"
  let result := binary_to_nat "110001111100"
  (a / c) * b = result := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l2258_225857


namespace NUMINAMATH_CALUDE_distinct_equals_odd_partitions_l2258_225839

/-- The number of partitions of n into distinct positive integers -/
def distinctPartitions (n : ℕ) : ℕ := sorry

/-- The number of partitions of n into positive odd integers -/
def oddPartitions (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of partitions of n into distinct positive integers
    equals the number of partitions of n into positive odd integers -/
theorem distinct_equals_odd_partitions (n : ℕ+) :
  distinctPartitions n = oddPartitions n := by sorry

end NUMINAMATH_CALUDE_distinct_equals_odd_partitions_l2258_225839


namespace NUMINAMATH_CALUDE_smallest_n_greater_than_20_l2258_225863

/-- g(n) is the sum of the digits of 1/(6^n) to the right of the decimal point -/
def g (n : ℕ+) : ℕ :=
  sorry

theorem smallest_n_greater_than_20 :
  (∀ k : ℕ+, k < 4 → g k ≤ 20) ∧ g 4 > 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_greater_than_20_l2258_225863


namespace NUMINAMATH_CALUDE_exam_students_count_l2258_225816

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 40) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) : 
  ∃ (N : ℕ), N = 25 ∧ 
  (N : ℝ) * total_average = (N - excluded_count : ℝ) * new_average + 
    (excluded_count : ℝ) * excluded_average :=
by
  sorry

#check exam_students_count

end NUMINAMATH_CALUDE_exam_students_count_l2258_225816


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2258_225884

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the y-axis -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

/-- The theorem stating that if A(2,5) is symmetric with B about the y-axis, then B(-2,5) -/
theorem symmetric_point_coordinates :
  let A : Point := ⟨2, 5⟩
  let B : Point := ⟨-2, 5⟩
  symmetricAboutYAxis A B → B = ⟨-2, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2258_225884


namespace NUMINAMATH_CALUDE_second_diff_is_arithmetic_sequence_l2258_225808

-- Define the cube function
def cube (n : ℕ) : ℕ := n^3

-- Define the first difference of cubes
def first_diff (n : ℕ) : ℕ := cube (n + 1) - cube n

-- Define the second difference of cubes
def second_diff (n : ℕ) : ℕ := first_diff (n + 1) - first_diff n

-- Theorem stating that the second difference is 6n + 6
theorem second_diff_is_arithmetic_sequence (n : ℕ) : second_diff n = 6 * n + 6 := by
  sorry

end NUMINAMATH_CALUDE_second_diff_is_arithmetic_sequence_l2258_225808


namespace NUMINAMATH_CALUDE_prize_probabilities_l2258_225881

/-- Represents the number of balls of each color in a box -/
structure BallBox where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a specific number of red balls from two boxes -/
def probability_draw_red (box_a box_b : BallBox) (red_count : Nat) : Rat :=
  sorry

/-- The first box containing 4 red balls and 6 white balls -/
def box_a : BallBox := { red := 4, white := 6 }

/-- The second box containing 5 red balls and 5 white balls -/
def box_b : BallBox := { red := 5, white := 5 }

theorem prize_probabilities :
  probability_draw_red box_a box_b 4 = 4 / 135 ∧
  probability_draw_red box_a box_b 3 = 26 / 135 ∧
  (1 - probability_draw_red box_a box_b 0) = 75 / 81 :=
sorry

end NUMINAMATH_CALUDE_prize_probabilities_l2258_225881


namespace NUMINAMATH_CALUDE_field_trip_van_occupancy_l2258_225826

/-- Proves the number of people in each van for a field trip --/
theorem field_trip_van_occupancy (num_vans : ℝ) (num_buses : ℝ) (people_per_bus : ℝ) (extra_people_in_buses : ℝ) :
  num_vans = 6.0 →
  num_buses = 8.0 →
  people_per_bus = 18.0 →
  extra_people_in_buses = 108 →
  num_buses * people_per_bus = num_vans * (num_buses * people_per_bus - extra_people_in_buses) / num_vans + extra_people_in_buses →
  (num_buses * people_per_bus - extra_people_in_buses) / num_vans = 6.0 := by
  sorry

#eval (8.0 * 18.0 - 108) / 6.0  -- Should output 6.0

end NUMINAMATH_CALUDE_field_trip_van_occupancy_l2258_225826


namespace NUMINAMATH_CALUDE_water_filter_capacity_l2258_225864

/-- The total capacity of a cylindrical water filter in liters. -/
def total_capacity : ℝ := 120

/-- The amount of water in the filter when it is partially filled, in liters. -/
def partial_amount : ℝ := 36

/-- The fraction of the filter that is filled when it contains the partial amount. -/
def partial_fraction : ℝ := 0.30

/-- Theorem stating that the total capacity of the water filter is 120 liters,
    given that it contains 36 liters when it is 30% full. -/
theorem water_filter_capacity :
  total_capacity * partial_fraction = partial_amount :=
by sorry

end NUMINAMATH_CALUDE_water_filter_capacity_l2258_225864


namespace NUMINAMATH_CALUDE_housing_units_with_vcr_l2258_225883

theorem housing_units_with_vcr (H : ℝ) (H_pos : H > 0) : 
  let cable_tv := (1 / 5 : ℝ) * H
  let vcr := F * H
  let both := (1 / 4 : ℝ) * cable_tv
  let neither := (3 / 4 : ℝ) * H
  ∃ F : ℝ, F = (1 / 10 : ℝ) ∧ cable_tv + vcr - both = H - neither :=
by sorry

end NUMINAMATH_CALUDE_housing_units_with_vcr_l2258_225883


namespace NUMINAMATH_CALUDE_square_plus_one_to_zero_is_one_l2258_225836

theorem square_plus_one_to_zero_is_one (m : ℝ) : (m^2 + 1)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_to_zero_is_one_l2258_225836


namespace NUMINAMATH_CALUDE_norma_bananas_l2258_225810

theorem norma_bananas (initial : ℕ) (lost : ℕ) (final : ℕ) :
  initial = 47 →
  lost = 45 →
  final = initial - lost →
  final = 2 :=
by sorry

end NUMINAMATH_CALUDE_norma_bananas_l2258_225810


namespace NUMINAMATH_CALUDE_movie_marathon_difference_l2258_225896

/-- The duration of a movie marathon with three movies. -/
structure MovieMarathon where
  first_movie : ℝ
  second_movie : ℝ
  last_movie : ℝ
  total_time : ℝ

/-- The conditions of the movie marathon problem. -/
def movie_marathon_conditions (m : MovieMarathon) : Prop :=
  m.first_movie = 2 ∧
  m.second_movie = m.first_movie * 1.5 ∧
  m.total_time = 9 ∧
  m.total_time = m.first_movie + m.second_movie + m.last_movie

/-- The theorem stating the difference between the combined time of the first two movies
    and the last movie is 1 hour. -/
theorem movie_marathon_difference (m : MovieMarathon) 
  (h : movie_marathon_conditions m) : 
  m.first_movie + m.second_movie - m.last_movie = 1 := by
  sorry

end NUMINAMATH_CALUDE_movie_marathon_difference_l2258_225896


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2258_225865

/-- Two vectors are parallel if and only if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two parallel vectors a = (2, 3) and b = (4, y + 1), prove that y = 5 -/
theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y + 1)
  parallel a b → y = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2258_225865


namespace NUMINAMATH_CALUDE_system_solution_l2258_225849

theorem system_solution : 
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℚ),
    (x₁ = 0 ∧ y₁ = -1 ∧ z₁ = 1) ∧
    (x₂ = 3 ∧ y₂ = 2 ∧ z₂ = 4) ∧
    (x₁ = (y₁ + 1) / (3 * y₁ - 5) ∧ 
     y₁ = (3 * z₁ - 2) / (2 * z₁ - 3) ∧ 
     z₁ = (3 * x₁ - 1) / (x₁ - 1)) ∧
    (x₂ = (y₂ + 1) / (3 * y₂ - 5) ∧ 
     y₂ = (3 * z₂ - 2) / (2 * z₂ - 3) ∧ 
     z₂ = (3 * x₂ - 1) / (x₂ - 1)) := by
  sorry


end NUMINAMATH_CALUDE_system_solution_l2258_225849


namespace NUMINAMATH_CALUDE_cubic_roots_expression_l2258_225873

theorem cubic_roots_expression (α β γ : ℂ) : 
  (α^3 - 3*α - 2 = 0) → 
  (β^3 - 3*β - 2 = 0) → 
  (γ^3 - 3*γ - 2 = 0) → 
  α*(β - γ)^2 + β*(γ - α)^2 + γ*(α - β)^2 = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_expression_l2258_225873


namespace NUMINAMATH_CALUDE_angle_problem_l2258_225848

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle4 →
  angle1 + 50 + 60 = 180 →
  angle4 = 35 := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l2258_225848


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l2258_225886

-- Define the pentagon and extended points
variable (A B C D E A' B' C' D' E' : ℝ × ℝ)

-- Define the conditions
axiom ext_A : A' = A + (A - B)
axiom ext_B : B' = B + (B - C)
axiom ext_C : C' = C + (C - D)
axiom ext_D : D' = D + (D - E)
axiom ext_E : E' = E + (E - A)

-- Define the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (5/31 : ℝ) • B' + (10/31 : ℝ) • C' + (15/31 : ℝ) • D' + (1/31 : ℝ) • E' := by
  sorry

end NUMINAMATH_CALUDE_pentagon_reconstruction_l2258_225886


namespace NUMINAMATH_CALUDE_average_stream_speed_theorem_l2258_225812

/-- Represents the swimming scenario with given parameters. -/
structure SwimmingScenario where
  swimmer_speed : ℝ  -- Speed of the swimmer in still water (km/h)
  upstream_time_ratio : ℝ  -- Ratio of upstream time to downstream time
  stream_speed_increase : ℝ  -- Increase in stream speed per 100 meters (km/h)
  upstream_distance : ℝ  -- Total upstream distance (meters)

/-- Calculates the average stream speed over the given distance. -/
def average_stream_speed (scenario : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating the average stream speed for the given scenario. -/
theorem average_stream_speed_theorem (scenario : SwimmingScenario) 
  (h1 : scenario.swimmer_speed = 1.5)
  (h2 : scenario.upstream_time_ratio = 2)
  (h3 : scenario.stream_speed_increase = 0.2)
  (h4 : scenario.upstream_distance = 500) :
  average_stream_speed scenario = 0.7 :=
sorry

end NUMINAMATH_CALUDE_average_stream_speed_theorem_l2258_225812


namespace NUMINAMATH_CALUDE_twins_age_problem_l2258_225860

theorem twins_age_problem (x : ℕ) : 
  (x + 1) * (x + 1) = x * x + 11 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_twins_age_problem_l2258_225860


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2258_225851

open Real

theorem trigonometric_expression_equals_one : 
  (sin (15 * π / 180) * cos (15 * π / 180) + cos (165 * π / 180) * cos (105 * π / 180)) /
  (sin (19 * π / 180) * cos (11 * π / 180) + cos (161 * π / 180) * cos (101 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2258_225851


namespace NUMINAMATH_CALUDE_sqrt_x_squared_plus_two_is_quadratic_radical_l2258_225828

-- Define what it means for an expression to be a quadratic radical
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, f x = y ∧ y ≥ 0

-- Theorem statement
theorem sqrt_x_squared_plus_two_is_quadratic_radical :
  is_quadratic_radical (λ x : ℝ => Real.sqrt (x^2 + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_plus_two_is_quadratic_radical_l2258_225828


namespace NUMINAMATH_CALUDE_negation_of_all_seated_l2258_225861

universe u

-- Define the predicates
variable (in_room : α → Prop)
variable (seated : α → Prop)

-- State the theorem
theorem negation_of_all_seated :
  ¬(∀ (x : α), in_room x → seated x) ↔ ∃ (x : α), in_room x ∧ ¬(seated x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_seated_l2258_225861


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2258_225869

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2258_225869


namespace NUMINAMATH_CALUDE_K_3_15_5_l2258_225879

def K (x y z : ℚ) : ℚ := x / y + y / z + z / x

theorem K_3_15_5 : K 3 15 5 = 73 / 15 := by
  sorry

end NUMINAMATH_CALUDE_K_3_15_5_l2258_225879


namespace NUMINAMATH_CALUDE_derivative_ln_plus_reciprocal_l2258_225801

theorem derivative_ln_plus_reciprocal (x : ℝ) (hx : x > 0) :
  deriv (λ x => Real.log x + x⁻¹) x = (x - 1) / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_ln_plus_reciprocal_l2258_225801


namespace NUMINAMATH_CALUDE_star_calculation_l2258_225885

/-- The custom operation ⋆ defined as x ⋆ y = (x² + y²)(x - y) -/
def star (x y : ℝ) : ℝ := (x^2 + y^2) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ 4) = 16983 -/
theorem star_calculation : star 2 (star 3 4) = 16983 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2258_225885


namespace NUMINAMATH_CALUDE_line_slope_l2258_225899

/-- The slope of the line defined by the equation x/4 + y/3 = 1 is -3/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2258_225899


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2258_225890

theorem equal_roots_quadratic (a : ℕ) : 
  (∀ x : ℝ, x^2 - a*x + (a + 3) = 0 → (∃! y : ℝ, y^2 - a*y + (a + 3) = 0)) → 
  a = 6 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2258_225890


namespace NUMINAMATH_CALUDE_problem_statement_l2258_225855

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = -1) :
  a^3 / (b - c)^2 + b^3 / (c - a)^2 + c^3 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2258_225855


namespace NUMINAMATH_CALUDE_gcd_299_621_l2258_225858

theorem gcd_299_621 : Nat.gcd 299 621 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_299_621_l2258_225858


namespace NUMINAMATH_CALUDE_sum_of_squares_minus_fourth_power_l2258_225870

theorem sum_of_squares_minus_fourth_power (a b : ℕ+) : 
  a^2 - b^4 = 2009 → a + b = 47 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_minus_fourth_power_l2258_225870


namespace NUMINAMATH_CALUDE_polynomial_Q_value_l2258_225888

-- Define the polynomial
def polynomial (P Q R : ℤ) (z : ℝ) : ℝ :=
  z^5 - 15*z^4 + P*z^3 + Q*z^2 + R*z + 64

-- Define the roots
def roots : List ℤ := [8, 4, 1, 1, 1]

-- Theorem statement
theorem polynomial_Q_value (P Q R : ℤ) :
  (∀ r ∈ roots, polynomial P Q R r = 0) →
  (List.sum roots = 15) →
  (List.prod roots = 64) →
  (∀ r ∈ roots, r > 0) →
  Q = -45 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_Q_value_l2258_225888


namespace NUMINAMATH_CALUDE_triangle_perimeter_32_l2258_225872

/-- Given a triangle ABC with vertices A(-3, 5), B(3, -3), and M(6, 1) as the midpoint of BC,
    prove that the perimeter of the triangle is 32. -/
theorem triangle_perimeter_32 :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (3, -3)
  let M : ℝ × ℝ := (6, 1)
  let C : ℝ × ℝ := (2 * M.1 - B.1, 2 * M.2 - B.2)  -- Derived from midpoint formula
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB + BC + AC = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_32_l2258_225872


namespace NUMINAMATH_CALUDE_bouncing_ball_original_height_l2258_225827

/-- Represents the behavior of a bouncing ball -/
def BouncingBall (originalHeight : ℝ) : Prop :=
  let reboundFactor := (1/2 : ℝ)
  let totalTravel := originalHeight +
                     2 * (reboundFactor * originalHeight) +
                     2 * (reboundFactor^2 * originalHeight)
  totalTravel = 250

/-- Theorem stating the original height of the ball -/
theorem bouncing_ball_original_height :
  ∃ (h : ℝ), BouncingBall h ∧ h = 100 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_original_height_l2258_225827


namespace NUMINAMATH_CALUDE_geometric_sequences_operations_l2258_225822

/-- A sequence is geometric if the ratio of consecutive terms is constant and non-zero -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequences_operations
  (a b : ℕ → ℝ)
  (ha : IsGeometricSequence a)
  (hb : IsGeometricSequence b) :
  IsGeometricSequence (fun n ↦ a n * b n) ∧
  IsGeometricSequence (fun n ↦ a n / b n) ∧
  ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b → IsGeometricSequence (fun n ↦ a n + b n)) ∧
  ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b → IsGeometricSequence (fun n ↦ a n - b n)) :=
by sorry


end NUMINAMATH_CALUDE_geometric_sequences_operations_l2258_225822


namespace NUMINAMATH_CALUDE_mushroom_ratio_l2258_225824

theorem mushroom_ratio (total : ℕ) (safe : ℕ) (uncertain : ℕ) 
  (h1 : total = 32) 
  (h2 : safe = 9) 
  (h3 : uncertain = 5) : 
  (total - safe - uncertain) / safe = 2 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_ratio_l2258_225824


namespace NUMINAMATH_CALUDE_thirteen_gumballs_needed_l2258_225819

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Calculates the least number of gumballs needed to ensure four of the same color -/
def leastGumballs (machine : GumballMachine) : ℕ :=
  sorry

/-- Theorem stating that for the given gumball machine, 13 is the least number of gumballs needed -/
theorem thirteen_gumballs_needed (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.white = 9)
  (h3 : machine.blue = 8)
  (h4 : machine.green = 6) :
  leastGumballs machine = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_thirteen_gumballs_needed_l2258_225819


namespace NUMINAMATH_CALUDE_incorrect_quotient_calculation_l2258_225825

theorem incorrect_quotient_calculation (dividend : ℕ) (correct_divisor incorrect_divisor correct_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : incorrect_divisor = 12)
  (h4 : correct_quotient = 40) :
  dividend / incorrect_divisor = 70 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_quotient_calculation_l2258_225825


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2258_225844

/-- The circumference of the base of a right circular cone with given volume and height -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (π : ℝ) :
  V = 36 * π →
  h = 3 →
  π > 0 →
  (2 * π * (3 * V / (π * h))^(1/2) : ℝ) = 12 * π := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2258_225844


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2258_225862

open Real

/-- A function f: ℝ₊ → ℝ₊ satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x > 0, f x < 2*x - x / (1 + x^(3/2))) ∧
  (∀ x > 0, f (f x) = (5/2) * f x - x)

/-- The theorem stating that the only function satisfying the conditions is f(x) = x/2 -/
theorem unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → (∀ x > 0, f x = x/2) :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2258_225862


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l2258_225830

theorem number_of_divisors_180 : Finset.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l2258_225830


namespace NUMINAMATH_CALUDE_angle_Q_measure_l2258_225840

-- Define the triangle PQR
structure Triangle (P Q R : ℝ) :=
  (sum_angles : P + Q + R = 180)
  (isosceles : Q = R)
  (angle_relation : R = 3 * P)

-- Theorem statement
theorem angle_Q_measure (P Q R : ℝ) (t : Triangle P Q R) :
  Q = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_Q_measure_l2258_225840


namespace NUMINAMATH_CALUDE_product_first_two_terms_l2258_225818

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem product_first_two_terms (a₁ : ℝ) (d : ℝ) :
  arithmetic_sequence a₁ d 7 = 25 ∧ d = 3 →
  a₁ * (a₁ + d) = 70 := by
  sorry

end NUMINAMATH_CALUDE_product_first_two_terms_l2258_225818


namespace NUMINAMATH_CALUDE_equation_solution_l2258_225805

theorem equation_solution :
  ∃ x : ℝ, (3 / (x + 2) - 1 / x = 0) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2258_225805


namespace NUMINAMATH_CALUDE_peter_completes_work_in_35_days_l2258_225856

/-- The number of days Matt and Peter take to complete the work together -/
def total_days_together : ℚ := 20

/-- The number of days Matt and Peter work together before Matt stops -/
def days_worked_together : ℚ := 12

/-- The number of days Peter takes to complete the remaining work after Matt stops -/
def peter_remaining_days : ℚ := 14

/-- The fraction of work completed when Matt and Peter work together for 12 days -/
def work_completed_together : ℚ := days_worked_together / total_days_together

/-- The fraction of work Peter completes after Matt stops -/
def peter_remaining_work : ℚ := 1 - work_completed_together

/-- Peter's work rate (fraction of work completed per day) -/
def peter_work_rate : ℚ := peter_remaining_work / peter_remaining_days

/-- The number of days Peter takes to complete the work separately -/
def peter_total_days : ℚ := 1 / peter_work_rate

theorem peter_completes_work_in_35_days :
  peter_total_days = 35 := by sorry

end NUMINAMATH_CALUDE_peter_completes_work_in_35_days_l2258_225856


namespace NUMINAMATH_CALUDE_girls_fraction_l2258_225876

theorem girls_fraction (T G B : ℝ) (x : ℝ) 
  (h1 : x * G = (1 / 5) * T)  -- Some fraction of girls is 1/5 of total
  (h2 : B / G = 1.5)          -- Ratio of boys to girls is 1.5
  (h3 : T = B + G)            -- Total is sum of boys and girls
  : x = 1 / 2 := by 
  sorry

end NUMINAMATH_CALUDE_girls_fraction_l2258_225876


namespace NUMINAMATH_CALUDE_average_difference_l2258_225878

def number_of_students : ℕ := 120
def number_of_teachers : ℕ := 4
def class_sizes : List ℕ := [60, 30, 20, 10]

def t : ℚ := (List.sum class_sizes) / number_of_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / number_of_students

theorem average_difference : t - s = -11663/1000 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2258_225878


namespace NUMINAMATH_CALUDE_car_meeting_speed_l2258_225834

/-- Proves that given the conditions of the problem, the speed of the second car must be 60 mph -/
theorem car_meeting_speed (total_distance : ℝ) (speed1 : ℝ) (start_time1 start_time2 : ℝ) (x : ℝ) : 
  total_distance = 600 →
  speed1 = 50 →
  start_time1 = 7 →
  start_time2 = 8 →
  (total_distance / 2) / speed1 + start_time1 = (total_distance / 2) / x + start_time2 →
  x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_car_meeting_speed_l2258_225834


namespace NUMINAMATH_CALUDE_wall_width_calculation_l2258_225880

theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) :
  mirror_side = 21 →
  wall_length = 31.5 →
  (mirror_side * mirror_side) * 2 = wall_length * (882 / wall_length) := by
  sorry

#check wall_width_calculation

end NUMINAMATH_CALUDE_wall_width_calculation_l2258_225880


namespace NUMINAMATH_CALUDE_book_cost_proof_l2258_225867

/-- The original cost of a book before discount -/
def original_cost : ℝ := sorry

/-- The number of books bought -/
def num_books : ℕ := 10

/-- The discount per book -/
def discount_per_book : ℝ := 0.5

/-- The total amount paid -/
def total_paid : ℝ := 45

theorem book_cost_proof :
  original_cost = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_book_cost_proof_l2258_225867


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l2258_225842

/-- Given two planes in 3D space, this theorem states that their line of intersection
    can be represented by specific canonical equations. -/
theorem intersection_line_canonical_equations
  (plane1 : x + y + z = 2)
  (plane2 : x - y - 2*z = -2)
  : ∃ (t : ℝ), x = -t ∧ y = 3*t + 2 ∧ z = -2*t :=
sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l2258_225842


namespace NUMINAMATH_CALUDE_rod_lengths_at_zero_celsius_l2258_225800

/-- Theorem: Rod Lengths at 0°C
Given:
- Total length at 0°C is 1 m
- Total length at 100°C is 1.0024 m
- Coefficient of linear expansion for steel is 0.000011
- Coefficient of linear expansion for zinc is 0.000031

Prove:
- Length of steel rod at 0°C is 0.35 m
- Length of zinc rod at 0°C is 0.65 m
-/
theorem rod_lengths_at_zero_celsius 
  (total_length_zero : Real) 
  (total_length_hundred : Real)
  (steel_expansion : Real)
  (zinc_expansion : Real)
  (h1 : total_length_zero = 1)
  (h2 : total_length_hundred = 1.0024)
  (h3 : steel_expansion = 0.000011)
  (h4 : zinc_expansion = 0.000031) :
  ∃ (steel_length zinc_length : Real),
    steel_length = 0.35 ∧ 
    zinc_length = 0.65 ∧
    steel_length + zinc_length = total_length_zero ∧
    steel_length * (1 + 100 * steel_expansion) + 
    zinc_length * (1 + 100 * zinc_expansion) = total_length_hundred :=
by sorry

end NUMINAMATH_CALUDE_rod_lengths_at_zero_celsius_l2258_225800


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2258_225847

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 2 * x + 5 * y = 20) 
  (eq2 : 5 * x + 2 * y = 26) : 
  20 * x^2 + 60 * x * y + 50 * y^2 = 59600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2258_225847


namespace NUMINAMATH_CALUDE_valid_words_count_l2258_225866

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def total_words (n : ℕ) (k : ℕ) : ℕ :=
  (n^1 + n^2 + n^3 + n^4 + n^5)

def words_without_specific_letter (n : ℕ) (k : ℕ) : ℕ :=
  ((n-1)^1 + (n-1)^2 + (n-1)^3 + (n-1)^4 + (n-1)^5)

theorem valid_words_count :
  total_words alphabet_size max_word_length - words_without_specific_letter alphabet_size max_word_length = 1678698 :=
by sorry

end NUMINAMATH_CALUDE_valid_words_count_l2258_225866


namespace NUMINAMATH_CALUDE_circle_tangent_and_bisecting_point_l2258_225837

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 4 * Real.sqrt 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the center of circle C
def center_C : ℝ × ℝ := (0, 0)

-- Define point M
def point_M : ℝ × ℝ := (2, 0)

-- Define point N
def point_N : ℝ × ℝ := (8, 0)

-- Theorem statement
theorem circle_tangent_and_bisecting_point :
  ∃ (N : ℝ × ℝ), N = point_N ∧
  (∀ (A B : ℝ × ℝ),
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 →
    ∃ (k : ℝ), 
      A.2 = k * (A.1 - point_M.1) ∧
      B.2 = k * (B.1 - point_M.1) →
      (A.2 / (A.1 - N.1)) + (B.2 / (B.1 - N.1)) = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_and_bisecting_point_l2258_225837


namespace NUMINAMATH_CALUDE_visited_neither_country_l2258_225832

theorem visited_neither_country (total : ℕ) (visited_iceland : ℕ) (visited_norway : ℕ) (visited_both : ℕ)
  (h1 : total = 50)
  (h2 : visited_iceland = 25)
  (h3 : visited_norway = 23)
  (h4 : visited_both = 21) :
  total - (visited_iceland + visited_norway - visited_both) = 23 :=
by sorry

end NUMINAMATH_CALUDE_visited_neither_country_l2258_225832


namespace NUMINAMATH_CALUDE_arrange_digits_eq_16_l2258_225882

/-- The number of ways to arrange the digits of 47,770 into a 5-digit number not beginning with 0 -/
def arrange_digits : ℕ :=
  let digits : List ℕ := [4, 7, 7, 7, 0]
  let total_digits : ℕ := 5
  let non_zero_digits : ℕ := 4
  let repeated_digit : ℕ := 7
  let repeated_count : ℕ := 3

  /- Number of ways to place 0 in the last 4 positions -/
  let zero_placements : ℕ := total_digits - 1

  /- Number of ways to arrange the remaining digits -/
  let remaining_arrangements : ℕ := Nat.factorial non_zero_digits / Nat.factorial repeated_count

  zero_placements * remaining_arrangements

theorem arrange_digits_eq_16 : arrange_digits = 16 := by
  sorry

end NUMINAMATH_CALUDE_arrange_digits_eq_16_l2258_225882


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l2258_225831

/-- If the solution set of the inequality -1/2x^2 + 2x > mx is {x | 0 < x < 2}, then m = 1 -/
theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (-1/2 * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l2258_225831


namespace NUMINAMATH_CALUDE_t_less_than_p_l2258_225895

theorem t_less_than_p (j p t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.8 * t) (h3 : t = 6.25) :
  (p - t) / p = 0.8 := by sorry

end NUMINAMATH_CALUDE_t_less_than_p_l2258_225895


namespace NUMINAMATH_CALUDE_continued_fraction_value_l2258_225821

theorem continued_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (1 + 5 / y) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l2258_225821


namespace NUMINAMATH_CALUDE_problem_statement_l2258_225897

theorem problem_statement (a b c : ℕ+) 
  (h : (18 ^ a.val) * (9 ^ (3 * a.val - 1)) * (c ^ (2 * a.val - 3)) = (2 ^ 7) * (3 ^ b.val)) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2258_225897


namespace NUMINAMATH_CALUDE_functional_equation_equivalence_l2258_225833

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_equivalence_l2258_225833


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2258_225809

theorem complex_fraction_equality (z : ℂ) (h : z = 1 - I) : 
  (z^2 - 2*z) / (z - 1) = -1 - I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2258_225809


namespace NUMINAMATH_CALUDE_hot_dog_stand_ketchup_bottles_l2258_225803

/-- Given a ratio of condiment bottles and the number of mayo bottles,
    calculate the number of ketchup bottles -/
def ketchup_bottles (ketchup_ratio mustard_ratio mayo_ratio mayo_count : ℕ) : ℕ :=
  (ketchup_ratio * mayo_count) / mayo_ratio

theorem hot_dog_stand_ketchup_bottles :
  ketchup_bottles 3 3 2 4 = 6 := by sorry

end NUMINAMATH_CALUDE_hot_dog_stand_ketchup_bottles_l2258_225803


namespace NUMINAMATH_CALUDE_minimum_value_inequality_minimum_value_achievable_l2258_225802

theorem minimum_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  1/x + 4/y + 9/z ≥ 36/5 := by
  sorry

theorem minimum_value_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 5 ∧ 1/x + 4/y + 9/z = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_inequality_minimum_value_achievable_l2258_225802


namespace NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l2258_225893

theorem cube_root_of_four_fifth_powers (x : ℝ) :
  x = (5^7 + 5^7 + 5^7 + 5^7)^(1/3) → x = 100 * 10^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l2258_225893


namespace NUMINAMATH_CALUDE_power_product_result_l2258_225898

theorem power_product_result : (-8)^20 * (1/4)^31 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_result_l2258_225898


namespace NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l2258_225868

theorem fourth_power_nested_sqrt : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l2258_225868


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2258_225820

/-- Given a triangle DEF where the measure of angle D is 75 degrees and 
    the measure of angle E is 18 degrees more than four times the measure of angle F,
    prove that the measure of angle F is 17.4 degrees. -/
theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 75 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2258_225820


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_folded_paper_l2258_225813

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem perimeter_ratio_of_folded_paper : 
  let original_side : ℝ := 6
  let large_rectangle : Rectangle := { length := original_side, width := original_side / 2 }
  let small_rectangle : Rectangle := { length := original_side / 2, width := original_side / 2 }
  (perimeter small_rectangle) / (perimeter large_rectangle) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_ratio_of_folded_paper_l2258_225813


namespace NUMINAMATH_CALUDE_sum_of_popsicle_sticks_l2258_225846

/-- The number of popsicle sticks Gino has -/
def gino_sticks : ℕ := 63

/-- The number of popsicle sticks I have -/
def my_sticks : ℕ := 50

/-- The sum of Gino's and my popsicle sticks -/
def total_sticks : ℕ := gino_sticks + my_sticks

theorem sum_of_popsicle_sticks : total_sticks = 113 := by sorry

end NUMINAMATH_CALUDE_sum_of_popsicle_sticks_l2258_225846


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2258_225835

theorem complex_sum_theorem (a b : ℂ) (h1 : a = 3 + 2*I) (h2 : b = 2 - I) :
  3*a + 4*b = 17 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2258_225835


namespace NUMINAMATH_CALUDE_negation_of_universal_quantification_l2258_225877

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x^2 + 2*x ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantification_l2258_225877


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2258_225811

/-- A domino is a 1x2 rectangle on the board -/
structure Domino where
  x : Fin 3000
  y : Fin 3000
  horizontal : Bool

/-- A color is represented by a number from 0 to 2 -/
def Color := Fin 3

/-- A coloring assigns a color to each domino -/
def Coloring := Domino → Color

/-- Two dominoes are neighbors if they share an edge -/
def are_neighbors (d1 d2 : Domino) : Prop :=
  sorry

/-- The number of dominoes with a given color in a coloring -/
def count_color (c : Coloring) (color : Color) : Nat :=
  sorry

/-- The number of neighbors of a domino with the same color -/
def same_color_neighbors (c : Coloring) (d : Domino) : Nat :=
  sorry

/-- The main theorem: there exists a valid coloring -/
theorem exists_valid_coloring :
  ∃ (c : Coloring),
    (∀ color : Color, count_color c color = 1500000) ∧
    (∀ d : Domino, same_color_neighbors c d ≤ 2) :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2258_225811


namespace NUMINAMATH_CALUDE_equation_represents_line_and_hyperbola_l2258_225823

-- Define the equation
def equation (x y : ℝ) : Prop := y^6 - 6*x^6 = 3*y^2 - 8

-- Define what it means for the equation to represent a line
def represents_line (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ, eq x y → y = a*x + b

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a*b < 0 ∧
    ∀ x y : ℝ, eq x y → a*x^2 + b*y^2 + c*x*y + d*x + e*y + f = 0

-- Theorem statement
theorem equation_represents_line_and_hyperbola :
  represents_line equation ∧ represents_hyperbola equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_line_and_hyperbola_l2258_225823


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2258_225871

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def binomial_expansion_coefficient (r : ℕ) : ℚ :=
  (-3)^r * binomial_coefficient 5 r

theorem coefficient_of_x_squared (expansion : ℕ → ℚ) :
  expansion = binomial_expansion_coefficient →
  (∃ r : ℕ, (10 - 3 * r) / 2 = 2 ∧ expansion r = 90) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2258_225871


namespace NUMINAMATH_CALUDE_contrapositive_truth_l2258_225853

theorem contrapositive_truth (p q : Prop) : 
  (q → p) → (¬p → ¬q) := by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_l2258_225853


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2258_225804

/-- The total capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 112.5

/-- Theorem stating that the tank capacity is correct given the problem conditions. -/
theorem tank_capacity_proof :
  tank_capacity = 112.5 ∧
  (0.5 * tank_capacity = 0.9 * tank_capacity - 45) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l2258_225804


namespace NUMINAMATH_CALUDE_paul_running_time_l2258_225850

/-- Given that Paul watches movies while running on a treadmill, prove that it takes him 12 minutes to run one mile. -/
theorem paul_running_time (num_movies : ℕ) (avg_movie_length : ℝ) (total_miles : ℝ) :
  num_movies = 2 →
  avg_movie_length = 1.5 →
  total_miles = 15 →
  (num_movies * avg_movie_length * 60) / total_miles = 12 := by
  sorry

end NUMINAMATH_CALUDE_paul_running_time_l2258_225850
