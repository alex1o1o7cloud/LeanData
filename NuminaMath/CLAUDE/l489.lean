import Mathlib

namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l489_48915

theorem infinite_solutions_condition (c : ℝ) : 
  (∀ x, 5 * (3 * x - c) = 3 * (5 * x + 20)) ↔ c = -12 := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l489_48915


namespace NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l489_48926

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 14 * a (n + 1) - a n - 4

theorem all_terms_are_perfect_squares :
  ∀ n : ℕ, ∃ s : ℤ, a n = s^2 := by
  sorry

end NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l489_48926


namespace NUMINAMATH_CALUDE_max_area_30_60_90_triangle_in_rectangle_l489_48985

/-- The maximum area of a 30-60-90 triangle inscribed in a 12x15 rectangle --/
theorem max_area_30_60_90_triangle_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ t : ℝ, t ≥ 0 ∧ t ≤ 12 → t^2 * Real.sqrt 3 / 2 ≤ A) ∧ 
    A = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_area_30_60_90_triangle_in_rectangle_l489_48985


namespace NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_a_eq_0_solution_set_inequality2_a_pos_solution_set_inequality2_a_neg_l489_48998

-- Define the inequalities
def inequality1 (x : ℝ) := -x^2 + 3*x + 4 ≥ 0
def inequality2 (x a : ℝ) := x^2 + 2*x + (1-a)*(1+a) ≥ 0

-- Theorem for the first inequality
theorem solution_set_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorems for the second inequality
theorem solution_set_inequality2_a_eq_0 :
  ∀ x : ℝ, inequality2 x 0 := by sorry

theorem solution_set_inequality2_a_pos (a : ℝ) (h : a > 0) :
  {x : ℝ | inequality2 x a} = {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} := by sorry

theorem solution_set_inequality2_a_neg (a : ℝ) (h : a < 0) :
  {x : ℝ | inequality2 x a} = {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_a_eq_0_solution_set_inequality2_a_pos_solution_set_inequality2_a_neg_l489_48998


namespace NUMINAMATH_CALUDE_tv_price_decrease_l489_48977

theorem tv_price_decrease (x : ℝ) : 
  (1 - x / 100) * (1 + 55 / 100) = 1 + 24 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_decrease_l489_48977


namespace NUMINAMATH_CALUDE_circle_equation_l489_48905

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line 4x + 3y = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 0}

-- Define the y-axis
def YAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

-- Theorem statement
theorem circle_equation :
  ∀ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ),
    -- Conditions
    C = Circle center 1 →  -- Radius is 1
    center.1 < 0 ∧ center.2 > 0 →  -- Center is in second quadrant
    ∃ (p : ℝ × ℝ), p ∈ C ∩ Line →  -- Tangent to 4x + 3y = 0
    ∃ (q : ℝ × ℝ), q ∈ C ∩ YAxis →  -- Tangent to y-axis
    -- Conclusion
    C = {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l489_48905


namespace NUMINAMATH_CALUDE_range_of_a_l489_48952

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 1 then Real.exp x - a * x^2 + x - 1 else sorry

-- State the theorem
theorem range_of_a :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∀ m : ℝ, m ≠ 0 → f a (1/m) * f a m = 1) →  -- property for non-zero m
  (∀ x, x > 1 → f a x = Real.exp x - a * x^2 + x - 1) →  -- definition for x > 1
  (∀ y : ℝ, ∃ x, f a x = y) →  -- range of f is R
  (∀ x, (x - 2) * Real.exp x - x + 4 > 0) →  -- given inequality
  a ∈ Set.Icc (Real.exp 1 - 1) ((Real.exp 2 + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l489_48952


namespace NUMINAMATH_CALUDE_closest_vector_to_origin_l489_48990

/-- The vector v is closest to the origin when t = 1/13 -/
theorem closest_vector_to_origin (t : ℝ) : 
  let v : ℝ × ℝ × ℝ := (1 + 3*t, 2 - 4*t, 3 + t)
  let a : ℝ × ℝ × ℝ := (0, 0, 0)
  let direction : ℝ × ℝ × ℝ := (3, -4, 1)
  (∀ s : ℝ, ‖v - a‖ ≤ ‖(1 + 3*s, 2 - 4*s, 3 + s) - a‖) ↔ t = 1/13 :=
by sorry


end NUMINAMATH_CALUDE_closest_vector_to_origin_l489_48990


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l489_48916

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l489_48916


namespace NUMINAMATH_CALUDE_part_one_part_two_l489_48925

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f a x ≤ 4) → 
  -1 ≤ a ∧ a ≤ 2 :=
sorry

-- Part II
theorem part_two (a : ℝ) :
  (∃ x : ℝ, f a (x - a) - f a (x + a) ≤ 2 * a - 1) → 
  a ≥ 1/4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l489_48925


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l489_48947

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let is_odd (n : ℕ) := n % 2 = 1
  let total_products := table_size * table_size
  let odd_products := (table_size / 2) * (table_size / 2)
  odd_products / total_products = (1 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l489_48947


namespace NUMINAMATH_CALUDE_total_candy_pieces_l489_48980

def chocolate_boxes : ℕ := 2
def caramel_boxes : ℕ := 5
def pieces_per_box : ℕ := 4

theorem total_candy_pieces : 
  (chocolate_boxes + caramel_boxes) * pieces_per_box = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_pieces_l489_48980


namespace NUMINAMATH_CALUDE_davids_trip_spending_l489_48994

theorem davids_trip_spending (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 1800 →
  remaining_amount = spent_amount - 800 →
  initial_amount = spent_amount + remaining_amount →
  remaining_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_davids_trip_spending_l489_48994


namespace NUMINAMATH_CALUDE_group_size_proof_l489_48955

/-- Proves that the number of members in a group is 54, given the conditions of the problem -/
theorem group_size_proof (n : ℕ) : 
  (n : ℚ) * n = 2916 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l489_48955


namespace NUMINAMATH_CALUDE_sum_first_100_odd_integers_l489_48918

/-- The sum of the first n positive odd integers -/
def sumFirstNOddIntegers (n : ℕ) : ℕ :=
  n * n

theorem sum_first_100_odd_integers :
  sumFirstNOddIntegers 100 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_100_odd_integers_l489_48918


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l489_48919

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (x₀, y₀)
  (∃ x₀ y₀, C (x₀, y₀) ∧ (x₀ * b)^2 = (y₀ * a)^2) →  -- A is on the asymptote
  (x₀^2 + y₀^2 = c^2 / 4) →  -- A is on the circle with diameter OF
  (Real.cos (π/6) * c = b) →  -- ∠AFO = π/6
  c / a = 2 :=  -- eccentricity is 2
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l489_48919


namespace NUMINAMATH_CALUDE_shaded_area_is_eleven_l489_48972

/-- Given a grid with rectangles of dimensions 2x3, 3x4, and 4x5, and two unshaded right-angled triangles
    with dimensions (base 12, height 4) and (base 3, height 2), the shaded area is 11. -/
theorem shaded_area_is_eleven :
  let grid_area := 2 * 3 + 3 * 4 + 4 * 5
  let triangle1_area := (12 * 4) / 2
  let triangle2_area := (3 * 2) / 2
  let shaded_area := grid_area - triangle1_area - triangle2_area
  shaded_area = 11 := by
sorry


end NUMINAMATH_CALUDE_shaded_area_is_eleven_l489_48972


namespace NUMINAMATH_CALUDE_weekly_toy_production_l489_48923

/-- A factory produces toys with the following conditions:
  * Workers work 5 days a week
  * Workers produce the same number of toys every day
  * Workers produce 1100 toys each day
-/
def toy_factory (days_per_week : ℕ) (toys_per_day : ℕ) : Prop :=
  days_per_week = 5 ∧ toys_per_day = 1100

/-- The number of toys produced in a week -/
def weekly_production (days_per_week : ℕ) (toys_per_day : ℕ) : ℕ :=
  days_per_week * toys_per_day

/-- Theorem: Under the given conditions, the factory produces 5500 toys in a week -/
theorem weekly_toy_production :
  ∀ (days_per_week toys_per_day : ℕ),
    toy_factory days_per_week toys_per_day →
    weekly_production days_per_week toys_per_day = 5500 :=
by
  sorry

end NUMINAMATH_CALUDE_weekly_toy_production_l489_48923


namespace NUMINAMATH_CALUDE_unique_w_exists_l489_48992

theorem unique_w_exists : ∃! w : ℝ, w > 0 ∧ 
  (Real.sqrt 1.5) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt w) = 3.0751133491652576 := by
  sorry

end NUMINAMATH_CALUDE_unique_w_exists_l489_48992


namespace NUMINAMATH_CALUDE_apples_given_to_neighbor_l489_48943

theorem apples_given_to_neighbor (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : remaining_apples = 39) :
  initial_apples - remaining_apples = 88 := by
  sorry

end NUMINAMATH_CALUDE_apples_given_to_neighbor_l489_48943


namespace NUMINAMATH_CALUDE_motor_lifespan_probability_l489_48989

variable (X : Real → Real)  -- Random variable representing motor lifespan

-- Define the expected value of X
def expected_value : Real := 4

-- Define the theorem
theorem motor_lifespan_probability :
  (∫ x, X x) = expected_value →  -- The expected value of X is 4
  (∫ x in {x | x < 20}, X x) ≥ 0.8 := by
  sorry

end NUMINAMATH_CALUDE_motor_lifespan_probability_l489_48989


namespace NUMINAMATH_CALUDE_warm_up_puzzle_time_l489_48910

/-- Represents the time taken for the warm-up puzzle in minutes -/
def warm_up_time : ℝ := 10

/-- Represents the total number of puzzles solved -/
def total_puzzles : ℕ := 3

/-- Represents the total time spent solving all puzzles in minutes -/
def total_time : ℝ := 70

/-- Represents the time multiplier for the longer puzzles compared to the warm-up puzzle -/
def longer_puzzle_multiplier : ℝ := 3

/-- Represents the number of longer puzzles solved -/
def longer_puzzles : ℕ := 2

theorem warm_up_puzzle_time :
  warm_up_time * (1 + longer_puzzle_multiplier * longer_puzzles) = total_time :=
by sorry

end NUMINAMATH_CALUDE_warm_up_puzzle_time_l489_48910


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l489_48909

theorem binomial_expansion_sum (n : ℕ) : 
  (∃ p q : ℕ, p = (3 + 1)^n ∧ q = 2^n ∧ p + q = 272) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l489_48909


namespace NUMINAMATH_CALUDE_average_weight_of_all_girls_l489_48936

theorem average_weight_of_all_girls (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  let total_count := group1_count + group2_count
  (total_weight / total_count) = 48.55 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_all_girls_l489_48936


namespace NUMINAMATH_CALUDE_polynomial_remainder_l489_48957

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l489_48957


namespace NUMINAMATH_CALUDE_janet_total_miles_l489_48964

/-- Represents Janet's running schedule for a week -/
structure WeekSchedule where
  days : ℕ
  milesPerDay : ℕ

/-- Calculates the total miles run in a week -/
def weeklyMiles (schedule : WeekSchedule) : ℕ :=
  schedule.days * schedule.milesPerDay

/-- Janet's running schedule for three weeks -/
def janetSchedule : List WeekSchedule :=
  [{ days := 5, milesPerDay := 8 },
   { days := 4, milesPerDay := 10 },
   { days := 3, milesPerDay := 6 }]

/-- Theorem: Janet ran a total of 98 miles over the three weeks -/
theorem janet_total_miles :
  (janetSchedule.map weeklyMiles).sum = 98 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_miles_l489_48964


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l489_48950

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l489_48950


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l489_48975

theorem partial_fraction_decomposition (a b c : ℤ) 
  (h1 : (1 : ℚ) / 2015 = a / 5 + b / 13 + c / 31)
  (h2 : 0 ≤ a ∧ a < 5)
  (h3 : 0 ≤ b ∧ b < 13) :
  a + b = 14 := by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l489_48975


namespace NUMINAMATH_CALUDE_stream_rate_l489_48983

/-- The rate of a stream given boat speed and downstream travel information -/
theorem stream_rate (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 16 →
  distance = 168 →
  time = 8 →
  (boat_speed + (distance / time - boat_speed)) * time = distance →
  distance / time - boat_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_rate_l489_48983


namespace NUMINAMATH_CALUDE_smallest_gcd_of_20m_25n_l489_48982

theorem smallest_gcd_of_20m_25n (m n : ℕ+) (h : Nat.gcd m.val n.val = 18) :
  ∃ (m₀ n₀ : ℕ+), Nat.gcd m₀.val n₀.val = 18 ∧
    Nat.gcd (20 * m₀.val) (25 * n₀.val) = 90 ∧
    ∀ (m' n' : ℕ+), Nat.gcd m'.val n'.val = 18 →
      Nat.gcd (20 * m'.val) (25 * n'.val) ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_20m_25n_l489_48982


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l489_48984

/-- The surface area of a sphere inscribed in a triangular pyramid with all edges of length a -/
theorem inscribed_sphere_surface_area (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ r = a / (2 * Real.sqrt 6) ∧ 
  4 * Real.pi * r^2 = Real.pi * a^2 / 6 :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l489_48984


namespace NUMINAMATH_CALUDE_marias_trip_l489_48908

theorem marias_trip (total_distance : ℝ) (h1 : total_distance = 360) : 
  let first_stop := total_distance / 2
  let remaining_after_first := total_distance - first_stop
  let second_stop := remaining_after_first / 4
  let distance_after_second := remaining_after_first - second_stop
  distance_after_second = 135 := by
sorry

end NUMINAMATH_CALUDE_marias_trip_l489_48908


namespace NUMINAMATH_CALUDE_second_month_sale_l489_48917

/-- Proves that the sale in the second month is 10500 given the conditions of the problem -/
theorem second_month_sale (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 2500)
  (h3 : sales 2 = 9855)
  (h4 : sales 3 = 7230)
  (h5 : sales 4 = 7000)
  (h6 : sales 5 = 11915)
  (avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7500) :
  sales 1 = 10500 := by
  sorry

#check second_month_sale

end NUMINAMATH_CALUDE_second_month_sale_l489_48917


namespace NUMINAMATH_CALUDE_gravel_cost_calculation_l489_48924

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 3

/-- The cost of the given volume of gravel in dollars -/
def total_cost : ℝ := gravel_volume_cubic_yards * cubic_feet_per_cubic_yard * gravel_cost_per_cubic_foot

theorem gravel_cost_calculation : total_cost = 648 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_calculation_l489_48924


namespace NUMINAMATH_CALUDE_division_equality_l489_48968

theorem division_equality : 815472 / 6630 = 123 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l489_48968


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l489_48958

theorem cube_sum_over_product (x y z : ℝ) :
  ((x - y)^3 + (y - z)^3 + (z - x)^3) / (15 * (x - y) * (y - z) * (z - x)) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l489_48958


namespace NUMINAMATH_CALUDE_parking_savings_l489_48945

-- Define the weekly and monthly rental rates
def weekly_rate : ℕ := 10
def monthly_rate : ℕ := 42

-- Define the number of weeks and months in a year
def weeks_per_year : ℕ := 52
def months_per_year : ℕ := 12

-- Define the yearly cost for weekly and monthly rentals
def yearly_cost_weekly : ℕ := weekly_rate * weeks_per_year
def yearly_cost_monthly : ℕ := monthly_rate * months_per_year

-- Theorem: The difference in yearly cost between weekly and monthly rentals is $16
theorem parking_savings : yearly_cost_weekly - yearly_cost_monthly = 16 := by
  sorry

end NUMINAMATH_CALUDE_parking_savings_l489_48945


namespace NUMINAMATH_CALUDE_min_shots_is_60_l489_48986

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : Nat
  shots_taken : Nat
  nora_lead : Nat
  nora_min_score : Nat

/-- Calculates the minimum number of consecutive 10-point shots needed for Nora to guarantee victory -/
def min_shots_for_victory (comp : ArcheryCompetition) : Nat :=
  let remaining_shots := comp.total_shots - comp.shots_taken
  let max_opponent_score := remaining_shots * 10
  let n := (max_opponent_score - comp.nora_lead + comp.nora_min_score * remaining_shots - 1) / (10 - comp.nora_min_score) + 1
  n

/-- Theorem stating that for the given competition scenario, the minimum number of 10-point shots needed is 60 -/
theorem min_shots_is_60 (comp : ArcheryCompetition) 
    (h1 : comp.total_shots = 150)
    (h2 : comp.shots_taken = 75)
    (h3 : comp.nora_lead = 80)
    (h4 : comp.nora_min_score = 5) : 
  min_shots_for_victory comp = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_shots_is_60_l489_48986


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l489_48965

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℚ := sorry

/-- The first term of the arithmetic sequence -/
def a₁ : ℚ := sorry

/-- The common difference of the arithmetic sequence -/
def d : ℚ := sorry

/-- Properties of the arithmetic sequence -/
axiom sum_formula (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Given conditions -/
axiom condition_1 : S 10 = 16
axiom condition_2 : S 100 - S 90 = 24

/-- Theorem to prove -/
theorem arithmetic_sequence_sum : S 100 = 200 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l489_48965


namespace NUMINAMATH_CALUDE_special_permutations_l489_48911

def word_length : ℕ := 7
def num_vowels : ℕ := 3
def num_consonants : ℕ := 4

theorem special_permutations :
  (word_length.choose num_vowels) * (num_consonants.factorial) = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_permutations_l489_48911


namespace NUMINAMATH_CALUDE_five_students_four_lectures_l489_48927

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 5 students choosing from 4 lectures results in 4^5 choices -/
theorem five_students_four_lectures :
  lecture_choices 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_four_lectures_l489_48927


namespace NUMINAMATH_CALUDE_lower_price_calculation_l489_48970

/-- The lower selling price of an article -/
def lower_price : ℚ := 348

/-- The higher selling price of an article -/
def higher_price : ℚ := 350

/-- The cost price of the article -/
def cost_price : ℚ := 40

/-- The percentage difference in profit between the two selling prices -/
def profit_difference_percentage : ℚ := 5 / 100

theorem lower_price_calculation :
  (higher_price - cost_price) = (lower_price - cost_price) + profit_difference_percentage * cost_price :=
by sorry

end NUMINAMATH_CALUDE_lower_price_calculation_l489_48970


namespace NUMINAMATH_CALUDE_jellybean_count_jellybean_problem_l489_48981

theorem jellybean_count (normal_class_size : ℕ) (absent_children : ℕ) 
  (jellybeans_per_child : ℕ) (remaining_jellybeans : ℕ) : ℕ :=
  let present_children := normal_class_size - absent_children
  let eaten_jellybeans := present_children * jellybeans_per_child
  eaten_jellybeans + remaining_jellybeans

theorem jellybean_problem : 
  jellybean_count 24 2 3 34 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_jellybean_problem_l489_48981


namespace NUMINAMATH_CALUDE_elliptical_orbit_distance_l489_48999

/-- Given an elliptical orbit with perigee 3 AU and apogee 15 AU, 
    the distance from the sun (at one focus) to a point on the minor axis is 3√5 + 6 AU -/
theorem elliptical_orbit_distance (perigee apogee : ℝ) (h1 : perigee = 3) (h2 : apogee = 15) :
  let semi_major_axis := (apogee + perigee) / 2
  let focal_distance := semi_major_axis - perigee
  let semi_minor_axis := Real.sqrt (semi_major_axis^2 - focal_distance^2)
  semi_minor_axis + focal_distance = 3 * Real.sqrt 5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_elliptical_orbit_distance_l489_48999


namespace NUMINAMATH_CALUDE_garden_area_l489_48939

theorem garden_area (total_posts : ℕ) (post_spacing : ℝ) (longer_side_ratio : ℕ) :
  total_posts = 24 →
  post_spacing = 3 →
  longer_side_ratio = 3 →
  ∃ (short_side_posts long_side_posts : ℕ),
    short_side_posts > 1 ∧
    long_side_posts > 1 ∧
    long_side_posts = longer_side_ratio * short_side_posts ∧
    total_posts = 2 * short_side_posts + 2 * long_side_posts - 4 ∧
    (short_side_posts - 1) * post_spacing * (long_side_posts - 1) * post_spacing = 297 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l489_48939


namespace NUMINAMATH_CALUDE_vector_problems_l489_48966

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

theorem vector_problems (x : ℝ) :
  (∃ k : ℝ, a x = k • b x → ‖a x - b x‖ = 2 ∨ ‖a x - b x‖ = 2 * Real.sqrt 5) ∧
  (0 < (a x).1 * (b x).1 + (a x).2 * (b x).2 → x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 3) ∧
  (‖a x‖ = 2 → ∃ c : ℝ × ℝ, ‖c‖ = 1 ∧ (a x).1 * c.1 + (a x).2 * c.2 = 0 ∧
    ((c.1 = Real.sqrt 3 / 2 ∧ c.2 = -1/2) ∨
     (c.1 = -Real.sqrt 3 / 2 ∧ c.2 = 1/2) ∨
     (c.1 = Real.sqrt 3 / 2 ∧ c.2 = 1/2) ∨
     (c.1 = -Real.sqrt 3 / 2 ∧ c.2 = -1/2))) :=
by sorry


end NUMINAMATH_CALUDE_vector_problems_l489_48966


namespace NUMINAMATH_CALUDE_factorization_problems_l489_48954

variable (m x y : ℝ)

theorem factorization_problems :
  (mx^2 - m*y = m*(x^2 - y)) ∧
  (2*x^2 - 8*x + 8 = 2*(x-2)^2) ∧
  (x^2*(2*x-1) + y^2*(1-2*x) = (2*x-1)*(x+y)*(x-y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l489_48954


namespace NUMINAMATH_CALUDE_x_div_y_value_l489_48906

theorem x_div_y_value (x y : ℝ) (h1 : |x| = 4) (h2 : |y| = 2) (h3 : x < y) :
  x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_div_y_value_l489_48906


namespace NUMINAMATH_CALUDE_projection_vector_l489_48949

/-- Given two lines k and n in 2D space, prove that the vector (-6, 9) satisfies the conditions for the projection of DC onto the normal of line n. -/
theorem projection_vector : ∃ (w1 w2 : ℝ), w1 = -6 ∧ w2 = 9 ∧ w1 + w2 = 3 ∧ 
  ∃ (t s : ℝ),
    let k := λ t : ℝ => (2 + 3*t, 3 + 2*t)
    let n := λ s : ℝ => (1 + 3*s, 5 + 2*s)
    let C := k t
    let D := n s
    let normal_n := (-2, 3)
    ∃ (c : ℝ), (w1, w2) = c • normal_n :=
by sorry

end NUMINAMATH_CALUDE_projection_vector_l489_48949


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l489_48934

/-- Given that the solution set of ax^2 + x + b > 0 with respect to x is (-1, 2), prove that a + b = 1 -/
theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ -1 < x ∧ x < 2) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l489_48934


namespace NUMINAMATH_CALUDE_geometric_progression_middle_term_l489_48969

theorem geometric_progression_middle_term 
  (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_geometric : b^2 = a * c) 
  (h_a : a = 5 + 2 * Real.sqrt 6) 
  (h_c : c = 5 - 2 * Real.sqrt 6) : 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_middle_term_l489_48969


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l489_48940

/-- Represents a quadrilateral ABCD with given side lengths and angles -/
structure Quadrilateral :=
  (AB BC CD DA : ℝ)
  (angleA angleD : ℝ)

/-- Calculates the area of the quadrilateral ABCD -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for the given quadrilateral, its area is (47√3)/4 -/
theorem quadrilateral_area_theorem (q : Quadrilateral) 
  (h1 : q.AB = 5) 
  (h2 : q.BC = 7) 
  (h3 : q.CD = 3) 
  (h4 : q.DA = 4) 
  (h5 : q.angleA = 2 * π / 3) 
  (h6 : q.angleD = 2 * π / 3) : 
  area q = (47 * Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l489_48940


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l489_48933

theorem reciprocal_sum_theorem :
  (∀ (a b c : ℕ+), (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≠ 9/11) ∧
  (∀ (a b c : ℕ+), (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ > 41/42 →
    (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≥ 1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l489_48933


namespace NUMINAMATH_CALUDE_second_class_average_l489_48903

theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 50 →
  avg_total = 56.25 →
  (n₁ * avg₁ + n₂ * (n₁ * avg₁ + n₂ * avg_total - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_total →
  (n₁ * avg₁ + n₂ * avg_total - n₁ * avg₁) / n₂ = 60 := by
sorry

end NUMINAMATH_CALUDE_second_class_average_l489_48903


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l489_48938

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  let a₇ : ℚ := geometricSequenceTerm a₁ r 7
  a₇ = 1/15552 :=
by sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l489_48938


namespace NUMINAMATH_CALUDE_friends_behind_yuna_l489_48928

theorem friends_behind_yuna (total_friends : ℕ) (friends_in_front : ℕ) : 
  total_friends = 6 → friends_in_front = 2 → total_friends - friends_in_front = 4 := by
  sorry

end NUMINAMATH_CALUDE_friends_behind_yuna_l489_48928


namespace NUMINAMATH_CALUDE_didi_fundraiser_total_l489_48956

/-- Calculates the total amount raised by Didi for her local soup kitchen --/
theorem didi_fundraiser_total (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
  (donation1 : ℚ) (donation2 : ℚ) (donation3 : ℚ) (donation4 : ℚ) :
  num_cakes = 20 →
  slices_per_cake = 12 →
  price_per_slice = 1 →
  donation1 = 3/4 →
  donation2 = 1/2 →
  donation3 = 1/4 →
  donation4 = 1/10 →
  (num_cakes * slices_per_cake * price_per_slice) + 
  (num_cakes * slices_per_cake * (donation1 + donation2 + donation3 + donation4)) = 624 := by
sorry

end NUMINAMATH_CALUDE_didi_fundraiser_total_l489_48956


namespace NUMINAMATH_CALUDE_complex_multiplication_problem_l489_48979

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the complex number multiplication -/
def complex_mult (a b c d : ℝ) : ℂ := Complex.mk (a * c - b * d) (a * d + b * c)

/-- The problem statement -/
theorem complex_multiplication_problem :
  complex_mult 4 (-3) 4 3 = 25 := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_problem_l489_48979


namespace NUMINAMATH_CALUDE_stating_probability_same_district_l489_48930

/-- Represents the four districts available for housing applications. -/
inductive District : Type
  | A
  | B
  | C
  | D

/-- The number of districts available. -/
def num_districts : ℕ := 4

/-- Represents an application scenario for two applicants. -/
def ApplicationScenario : Type := District × District

/-- The total number of possible application scenarios for two applicants. -/
def total_scenarios : ℕ := num_districts * num_districts

/-- Predicate to check if two applicants applied for the same district. -/
def same_district (scenario : ApplicationScenario) : Prop :=
  scenario.1 = scenario.2

/-- The number of scenarios where two applicants apply for the same district. -/
def num_same_district_scenarios : ℕ := num_districts

/-- 
Theorem stating that the probability of two applicants choosing the same district
is 1/4, given that there are four equally likely choices for each applicant.
-/
theorem probability_same_district :
  (num_same_district_scenarios : ℚ) / total_scenarios = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_stating_probability_same_district_l489_48930


namespace NUMINAMATH_CALUDE_counterexample_exists_l489_48963

theorem counterexample_exists : ∃ n : ℕ, 
  Nat.Prime n ∧ Even n ∧ ¬(Nat.Prime (n + 2)) := by
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l489_48963


namespace NUMINAMATH_CALUDE_crescent_area_implies_square_area_l489_48920

/-- Given a square with side length s, the area of 8 "crescent" shapes formed by
    semicircles on its sides and the sides of its inscribed square (formed by
    connecting midpoints) is equal to πs². If this area is 5 square centimeters,
    then the area of the original square is 10 square centimeters. -/
theorem crescent_area_implies_square_area :
  ∀ s : ℝ,
  s > 0 →
  π * s^2 = 5 →
  s^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_crescent_area_implies_square_area_l489_48920


namespace NUMINAMATH_CALUDE_polygon_with_150_degree_interior_angles_has_12_sides_l489_48931

theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 150 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_150_degree_interior_angles_has_12_sides_l489_48931


namespace NUMINAMATH_CALUDE_motel_room_rate_problem_l489_48988

theorem motel_room_rate_problem (total_rent : ℕ) (lower_rate : ℕ) (num_rooms_changed : ℕ) (rent_decrease_percent : ℚ) (higher_rate : ℕ) : 
  total_rent = 400 →
  lower_rate = 50 →
  num_rooms_changed = 10 →
  rent_decrease_percent = 1/4 →
  (total_rent : ℚ) * rent_decrease_percent = (num_rooms_changed : ℚ) * (higher_rate - lower_rate) →
  higher_rate = 60 := by
sorry

end NUMINAMATH_CALUDE_motel_room_rate_problem_l489_48988


namespace NUMINAMATH_CALUDE_weight_11_25m_l489_48971

/-- Represents the weight of a uniform rod given its length -/
def rod_weight (length : ℝ) : ℝ := sorry

/-- The rod is uniform, meaning its weight is proportional to its length -/
axiom rod_uniform (l₁ l₂ : ℝ) : l₁ * rod_weight l₂ = l₂ * rod_weight l₁

/-- The weight of 6 meters of the rod is 22.8 kg -/
axiom weight_6m : rod_weight 6 = 22.8

/-- Theorem: If 6 m of a uniform rod weighs 22.8 kg, then 11.25 m weighs 42.75 kg -/
theorem weight_11_25m : rod_weight 11.25 = 42.75 := by sorry

end NUMINAMATH_CALUDE_weight_11_25m_l489_48971


namespace NUMINAMATH_CALUDE_games_lost_l489_48921

theorem games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 16) 
  (h2 : won_games = 12) : 
  total_games - won_games = 4 := by
sorry

end NUMINAMATH_CALUDE_games_lost_l489_48921


namespace NUMINAMATH_CALUDE_adams_shelves_l489_48962

/-- The number of action figures that can fit on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- The total number of action figures that can be held by all shelves -/
def total_action_figures : ℕ := 44

/-- The number of shelves in Adam's room -/
def number_of_shelves : ℕ := total_action_figures / action_figures_per_shelf

/-- Theorem stating that the number of shelves in Adam's room is 4 -/
theorem adams_shelves : number_of_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_adams_shelves_l489_48962


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l489_48907

/-- Parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * c.p * x

/-- Circle tangent to y-axis and intersecting MF -/
structure TangentCircle (c : Parabola) (m : PointOnParabola c) where
  a : ℝ × ℝ  -- Point A
  tangent_to_y_axis : sorry
  intersects_mf : sorry

/-- Theorem: Given the conditions, p = 2 -/
theorem parabola_focus_theorem (c : Parabola) 
    (m : PointOnParabola c)
    (h_m : m.y = 2 * Real.sqrt 2)
    (circle : TangentCircle c m)
    (h_ratio : (Real.sqrt ((m.x - circle.a.1)^2 + (m.y - circle.a.2)^2)) / 
               (Real.sqrt ((c.p - circle.a.1)^2 + circle.a.2^2)) = 2) :
  c.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l489_48907


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l489_48912

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def last_two_digits (n : ℕ) : ℕ := n % 100

def six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

theorem divisible_by_twelve (square : ℕ) :
  is_divisible_by (six_digit_number 4 8 6 3 square 5) 12 ↔ square = 1 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l489_48912


namespace NUMINAMATH_CALUDE_zero_in_interval_l489_48929

/-- Given two positive real numbers a and b where a > b > 0 and |log a| = |log b|,
    there exists an x in the interval (-1, 0) such that a^x + x - b = 0 -/
theorem zero_in_interval (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : |Real.log a| = |Real.log b|) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ a^x + x - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l489_48929


namespace NUMINAMATH_CALUDE_unique_prime_in_sequence_l489_48935

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (A : ℕ) : ℕ := 205100 + A

theorem unique_prime_in_sequence :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) ∧ number A = 205103 := by sorry

end NUMINAMATH_CALUDE_unique_prime_in_sequence_l489_48935


namespace NUMINAMATH_CALUDE_range_of_m_l489_48901

/-- Set A is defined as the set of real numbers x where -3 ≤ x ≤ 4 -/
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

/-- Set B is defined as the set of real numbers x where 1 < x < m, and m > 1 -/
def B (m : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < m}

/-- The theorem states that if B is a subset of A and m > 1, then 1 < m ≤ 4 -/
theorem range_of_m (m : ℝ) (h1 : B m ⊆ A) (h2 : 1 < m) : 1 < m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l489_48901


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l489_48961

theorem stratified_sampling_problem (total : ℕ) (sample_size : ℕ) 
  (stratum_A : ℕ) (stratum_B : ℕ) (h1 : total = 1200) (h2 : sample_size = 120) 
  (h3 : stratum_A = 380) (h4 : stratum_B = 420) : 
  let stratum_C := total - stratum_A - stratum_B
  (sample_size * stratum_C) / total = 40 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l489_48961


namespace NUMINAMATH_CALUDE_jerry_shelf_items_after_changes_l489_48953

/-- Represents the items on Jerry's shelf -/
structure ShelfItems where
  action_figures : ℕ
  books : ℕ
  video_games : ℕ

/-- Calculates the total number of items on the shelf -/
def total_items (items : ShelfItems) : ℕ :=
  items.action_figures + items.books + items.video_games

/-- Represents the changes made to the shelf items -/
structure ItemChanges where
  action_figures_added : ℕ
  books_removed : ℕ
  video_games_added : ℕ

/-- Applies changes to the shelf items -/
def apply_changes (items : ShelfItems) (changes : ItemChanges) : ShelfItems where
  action_figures := items.action_figures + changes.action_figures_added
  books := items.books - changes.books_removed
  video_games := items.video_games + changes.video_games_added

theorem jerry_shelf_items_after_changes :
  let initial_items : ShelfItems := ⟨4, 22, 10⟩
  let changes : ItemChanges := ⟨6, 5, 3⟩
  let final_items := apply_changes initial_items changes
  total_items final_items = 40 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_items_after_changes_l489_48953


namespace NUMINAMATH_CALUDE_square_difference_hundred_ninetynine_l489_48993

theorem square_difference_hundred_ninetynine : 100^2 - 2*100*99 + 99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_hundred_ninetynine_l489_48993


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l489_48967

def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 1)) / 2
  if total_needed > initial_coins then
    total_needed - initial_coins
  else
    0

theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 90) :
  min_additional_coins num_friends initial_coins = 30 := by
  sorry

#eval min_additional_coins 15 90

end NUMINAMATH_CALUDE_alex_coin_distribution_l489_48967


namespace NUMINAMATH_CALUDE_randy_blocks_problem_l489_48913

theorem randy_blocks_problem (blocks_used blocks_left : ℕ) 
  (h1 : blocks_used = 36)
  (h2 : blocks_left = 23) :
  blocks_used + blocks_left = 59 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_problem_l489_48913


namespace NUMINAMATH_CALUDE_codes_lost_l489_48922

/-- The number of digits in each code -/
def code_length : Nat := 5

/-- The number of possible digits (0 to 9) -/
def digit_options : Nat := 10

/-- The number of non-zero digits (1 to 9) -/
def nonzero_digit_options : Nat := 9

/-- The number of codes with leading zeros allowed -/
def codes_with_leading_zeros : Nat := digit_options ^ code_length

/-- The number of codes without leading zeros -/
def codes_without_leading_zeros : Nat := nonzero_digit_options * (digit_options ^ (code_length - 1))

theorem codes_lost (code_length : Nat) (digit_options : Nat) (nonzero_digit_options : Nat) 
  (codes_with_leading_zeros : Nat) (codes_without_leading_zeros : Nat) :
  codes_with_leading_zeros - codes_without_leading_zeros = 10000 := by
  sorry

end NUMINAMATH_CALUDE_codes_lost_l489_48922


namespace NUMINAMATH_CALUDE_car_speed_ratio_l489_48904

-- Define the variables and constants
variable (v : ℝ) -- speed of car A
variable (k : ℝ) -- speed multiplier for car B
variable (AB CD AD : ℝ) -- distances

-- Define the theorem
theorem car_speed_ratio (h1 : k > 1) (h2 : AD = AB / 2) (h3 : CD / AD = 1 / 2) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_ratio_l489_48904


namespace NUMINAMATH_CALUDE_ascent_speed_l489_48973

/-- Given a journey with ascent and descent, calculate the average speed during ascent -/
theorem ascent_speed
  (total_time : ℝ)
  (overall_speed : ℝ)
  (ascent_time : ℝ)
  (h_total_time : total_time = 6)
  (h_overall_speed : overall_speed = 3.5)
  (h_ascent_time : ascent_time = 4)
  (h_equal_distance : ∀ d : ℝ, d = overall_speed * total_time / 2) :
  ∃ (ascent_speed : ℝ), ascent_speed = 2.625 ∧ ascent_speed = (overall_speed * total_time / 2) / ascent_time :=
by sorry

end NUMINAMATH_CALUDE_ascent_speed_l489_48973


namespace NUMINAMATH_CALUDE_scientific_notation_3650000_l489_48948

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_3650000 :
  toScientificNotation 3650000 = ScientificNotation.mk 3.65 6 sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3650000_l489_48948


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l489_48951

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x y = 210) (h2 : Nat.gcd x z = 770) :
  ∃ (y' z' : ℕ+), Nat.gcd x y' = 210 ∧ Nat.gcd x z' = 770 ∧ Nat.gcd y' z' = 10 ∧
  ∀ (y'' z'' : ℕ+), Nat.gcd x y'' = 210 → Nat.gcd x z'' = 770 → Nat.gcd y'' z'' ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l489_48951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l489_48996

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- a_n represents the nth term of the arithmetic sequence -/
def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_problem (h : S 9 a = 45) : a 5 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l489_48996


namespace NUMINAMATH_CALUDE_pen_rubber_length_difference_l489_48902

/-- Given a rubber, pen, and pencil with certain length relationships,
    prove that the pen is 3 cm longer than the rubber. -/
theorem pen_rubber_length_difference :
  ∀ (rubber_length pen_length pencil_length : ℝ),
    pencil_length = 12 →
    pen_length = pencil_length - 2 →
    rubber_length + pen_length + pencil_length = 29 →
    pen_length - rubber_length = 3 :=
by sorry

end NUMINAMATH_CALUDE_pen_rubber_length_difference_l489_48902


namespace NUMINAMATH_CALUDE_square_eq_product_sum_seven_l489_48941

theorem square_eq_product_sum_seven (a b : ℕ) : a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_eq_product_sum_seven_l489_48941


namespace NUMINAMATH_CALUDE_equal_variables_l489_48987

theorem equal_variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + 1/y = y + 1/x)
  (h2 : y + 1/z = z + 1/y)
  (h3 : z + 1/x = x + 1/z) :
  x = y ∨ y = z ∨ x = z := by
  sorry

end NUMINAMATH_CALUDE_equal_variables_l489_48987


namespace NUMINAMATH_CALUDE_figure_squares_l489_48960

-- Define the sequence function
def f (n : ℕ) : ℕ := 2 * n^2 + 2 * n + 1

-- State the theorem
theorem figure_squares (n : ℕ) : 
  f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25 → f 100 = 20201 := by
  sorry


end NUMINAMATH_CALUDE_figure_squares_l489_48960


namespace NUMINAMATH_CALUDE_median_interval_is_65_to_69_l489_48997

/-- Represents a score interval with its lower and upper bounds -/
structure ScoreInterval where
  lower : ℕ
  upper : ℕ

/-- Represents the distribution of scores -/
structure ScoreDistribution where
  intervals : List ScoreInterval
  counts : List ℕ

/-- Finds the interval containing the median score -/
def findMedianInterval (dist : ScoreDistribution) : Option ScoreInterval :=
  sorry

/-- The given score distribution -/
def testScoreDistribution : ScoreDistribution :=
  { intervals := [
      { lower := 50, upper := 54 },
      { lower := 55, upper := 59 },
      { lower := 60, upper := 64 },
      { lower := 65, upper := 69 },
      { lower := 70, upper := 74 }
    ],
    counts := [10, 15, 25, 30, 20]
  }

/-- Theorem: The median score interval for the given distribution is 65-69 -/
theorem median_interval_is_65_to_69 :
  findMedianInterval testScoreDistribution = some { lower := 65, upper := 69 } :=
  sorry

end NUMINAMATH_CALUDE_median_interval_is_65_to_69_l489_48997


namespace NUMINAMATH_CALUDE_sequence_sum_l489_48995

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that the sum of the first n terms (S_n) is equal to 2n / (n + 1). -/
theorem sequence_sum (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) (h1 : a 1 = 1)
    (h2 : ∀ n : ℕ+, S n = n^2 * a n) :
    ∀ n : ℕ+, S n = 2 * n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l489_48995


namespace NUMINAMATH_CALUDE_taxi_growth_equation_l489_48944

def initial_taxis : ℕ := 11720
def final_taxis : ℕ := 13116
def years : ℕ := 2

theorem taxi_growth_equation (x : ℝ) : 
  (initial_taxis : ℝ) * (1 + x)^years = final_taxis ↔ 
  x = ((final_taxis : ℝ) / initial_taxis)^(1 / years : ℝ) - 1 :=
by sorry

end NUMINAMATH_CALUDE_taxi_growth_equation_l489_48944


namespace NUMINAMATH_CALUDE_fraction_equality_l489_48942

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l489_48942


namespace NUMINAMATH_CALUDE_expected_digits_is_31_20_l489_48900

/-- A fair 20-sided die with numbers 1 through 20 -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The number of digits for a given number on the die -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum (λ i => numDigits (i + 1))) / icosahedralDie.card

theorem expected_digits_is_31_20 : expectedDigits = 31 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_31_20_l489_48900


namespace NUMINAMATH_CALUDE_largest_sum_is_1803_l489_48914

/-- The set of digits to be used -/
def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

/-- A function that computes the sum of two 3-digit numbers -/
def sum_3digit (a b c d e f : Nat) : Nat :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The theorem stating that 1803 is the largest possible sum -/
theorem largest_sum_is_1803 :
  ∀ a b c d e f : Nat,
    a ∈ digits → b ∈ digits → c ∈ digits →
    d ∈ digits → e ∈ digits → f ∈ digits →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f →
    c ≠ d → c ≠ e → c ≠ f →
    d ≠ e → d ≠ f →
    e ≠ f →
    sum_3digit a b c d e f ≤ 1803 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_1803_l489_48914


namespace NUMINAMATH_CALUDE_equilateral_triangle_point_distance_l489_48991

/-- Given an equilateral triangle ABC with side length a and a point P inside the triangle
    such that PA = u, PB = v, PC = w, and u^2 + v^2 = w^2, prove that w^2 + √3uv = a^2. -/
theorem equilateral_triangle_point_distance (a u v w : ℝ) :
  a > 0 →  -- Ensure positive side length
  u > 0 ∧ v > 0 ∧ w > 0 →  -- Ensure positive distances
  u^2 + v^2 = w^2 →  -- Given condition
  w^2 + Real.sqrt 3 * u * v = a^2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_point_distance_l489_48991


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l489_48937

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l489_48937


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l489_48978

/-- Represents a repeating decimal with a 3-digit repetend -/
def RepeatingDecimal (whole : ℕ) (repetend : ℕ) : ℚ :=
  whole + (repetend : ℚ) / 999

theorem repeating_decimal_ratio : 
  (RepeatingDecimal 0 833) / (RepeatingDecimal 1 666) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l489_48978


namespace NUMINAMATH_CALUDE_cordelia_hair_bleaching_l489_48959

/-- The time it takes to bleach Cordelia's hair. -/
def bleaching_time : ℝ := 3

/-- The total time for the hair coloring process. -/
def total_time : ℝ := 9

/-- The relationship between dyeing time and bleaching time. -/
def dyeing_time (b : ℝ) : ℝ := 2 * b

theorem cordelia_hair_bleaching :
  bleaching_time + dyeing_time bleaching_time = total_time ∧
  bleaching_time = 3 := by
sorry

end NUMINAMATH_CALUDE_cordelia_hair_bleaching_l489_48959


namespace NUMINAMATH_CALUDE_barbara_candies_l489_48976

/-- The number of candies Barbara bought -/
def candies_bought (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem barbara_candies :
  candies_bought 9 27 = 18 :=
by sorry

end NUMINAMATH_CALUDE_barbara_candies_l489_48976


namespace NUMINAMATH_CALUDE_solution_for_a_l489_48974

theorem solution_for_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (eq1 : a + 1/b = 5) (eq2 : b + 1/a = 10) : 
  a = (5 + Real.sqrt 23) / 2 ∨ a = (5 - Real.sqrt 23) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_a_l489_48974


namespace NUMINAMATH_CALUDE_unique_digit_solution_l489_48932

theorem unique_digit_solution :
  ∃! (digits : Fin 5 → Nat),
    (∀ i, digits i ≠ 0 ∧ digits i ≤ 9) ∧
    (digits 0 + digits 1 = (digits 2 + digits 3 + digits 4) / 7) ∧
    (digits 0 + digits 3 = (digits 1 + digits 2 + digits 4) / 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l489_48932


namespace NUMINAMATH_CALUDE_inequality_system_solution_l489_48946

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 2 < 2*m ∧ x - m < 0) ↔ x < 2*m - 2) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l489_48946
