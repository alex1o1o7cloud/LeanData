import Mathlib

namespace NUMINAMATH_CALUDE_research_budget_allocation_l1423_142374

theorem research_budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (industrial_lubricants : ℝ) (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 20 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 18 →
  ∃ (genetically_modified_microorganisms : ℝ),
    genetically_modified_microorganisms = 29 ∧
    microphotonics + home_electronics + food_additives + industrial_lubricants + 
    (basic_astrophysics_degrees / 360 * 100) + genetically_modified_microorganisms = 100 :=
by sorry

end NUMINAMATH_CALUDE_research_budget_allocation_l1423_142374


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1423_142367

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The total number of people that can ride the wheel at the same time -/
def total_people : ℕ := 20

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1423_142367


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1423_142314

theorem absolute_value_inequality (x : ℝ) : ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1423_142314


namespace NUMINAMATH_CALUDE_acute_angle_solution_l1423_142300

theorem acute_angle_solution : ∃ x : Real, 
  0 < x ∧ 
  x < π / 2 ∧ 
  2 * (Real.sin x)^2 + Real.sin x - Real.sin (2 * x) = 3 * Real.cos x ∧ 
  x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_solution_l1423_142300


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1423_142397

/-- Calculates the total distance covered by a car given its uphill and downhill speeds and times. -/
def total_distance (uphill_speed downhill_speed uphill_time downhill_time : ℝ) : ℝ :=
  uphill_speed * uphill_time + downhill_speed * downhill_time

/-- Theorem stating that under the given conditions, the total distance covered by the car is 400 km. -/
theorem car_distance_theorem :
  let uphill_speed : ℝ := 30
  let downhill_speed : ℝ := 50
  let uphill_time : ℝ := 5
  let downhill_time : ℝ := 5
  total_distance uphill_speed downhill_speed uphill_time downhill_time = 400 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1423_142397


namespace NUMINAMATH_CALUDE_total_soldiers_on_great_wall_l1423_142310

-- Define the parameters
def wall_length : ℕ := 7300
def tower_interval : ℕ := 5
def soldiers_per_tower : ℕ := 2

-- Theorem statement
theorem total_soldiers_on_great_wall :
  (wall_length / tower_interval) * soldiers_per_tower = 2920 :=
by sorry

end NUMINAMATH_CALUDE_total_soldiers_on_great_wall_l1423_142310


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_powers_l1423_142396

theorem infinitely_many_divisible_powers (a b c : ℤ) 
  (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, (a + b + c) ∣ (a^n + b^n + c^n) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_powers_l1423_142396


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1423_142318

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_theorem :
  ((U \ M) ∩ N) = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1423_142318


namespace NUMINAMATH_CALUDE_max_divisors_1_to_20_l1423_142344

def divisorCount (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).card

def maxDivisorCount : ℕ → ℕ
  | 0 => 0
  | n + 1 => max (maxDivisorCount n) (divisorCount (n + 1))

theorem max_divisors_1_to_20 :
  maxDivisorCount 20 = 6 ∧
  divisorCount 12 = 6 ∧
  divisorCount 18 = 6 ∧
  divisorCount 20 = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → divisorCount n ≤ 6 :=
by sorry

#eval maxDivisorCount 20
#eval divisorCount 12
#eval divisorCount 18
#eval divisorCount 20

end NUMINAMATH_CALUDE_max_divisors_1_to_20_l1423_142344


namespace NUMINAMATH_CALUDE_third_row_is_4213_l1423_142380

/-- Represents a 4x4 grid of numbers -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a number is the first odd or even in a list -/
def isFirstOddOrEven (n : Fin 4) (list : List (Fin 4)) : Prop :=
  n.val % 2 ≠ list.head!.val % 2 ∧ 
  ∀ m ∈ list, m.val < n.val → m.val % 2 = list.head!.val % 2

/-- The constraints of the grid puzzle -/
structure GridConstraints (grid : Grid) : Prop where
  unique_in_row : ∀ i j k, j ≠ k → grid i j ≠ grid i k
  unique_in_col : ∀ i j k, i ≠ k → grid i j ≠ grid k j
  top_indicators : ∀ j, isFirstOddOrEven (grid 0 j) [grid 1 j, grid 2 j, grid 3 j]
  left_indicators : ∀ i, isFirstOddOrEven (grid i 0) [grid i 1, grid i 2, grid i 3]
  right_indicators : ∀ i, isFirstOddOrEven (grid i 3) [grid i 2, grid i 1, grid i 0]
  bottom_indicators : ∀ j, isFirstOddOrEven (grid 3 j) [grid 2 j, grid 1 j, grid 0 j]

/-- The main theorem stating that the third row must be [4, 2, 1, 3] -/
theorem third_row_is_4213 (grid : Grid) (h : GridConstraints grid) :
  (grid 2 0 = 4) ∧ (grid 2 1 = 2) ∧ (grid 2 2 = 1) ∧ (grid 2 3 = 3) := by
  sorry

end NUMINAMATH_CALUDE_third_row_is_4213_l1423_142380


namespace NUMINAMATH_CALUDE_asymptote_sum_l1423_142386

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x, x^3 + A*x^2 + B*x + C = (x + 3)*(x - 1)*(x - 3)) → A + B + C = 15 :=
by sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1423_142386


namespace NUMINAMATH_CALUDE_students_without_glasses_l1423_142327

theorem students_without_glasses (total_students : ℕ) (percent_with_glasses : ℚ) 
  (h1 : total_students = 325)
  (h2 : percent_with_glasses = 40/100) : 
  ↑total_students * (1 - percent_with_glasses) = 195 := by
  sorry

end NUMINAMATH_CALUDE_students_without_glasses_l1423_142327


namespace NUMINAMATH_CALUDE_marian_needs_31_trays_l1423_142358

/-- The number of trays Marian needs to prepare cookies for classmates and teachers -/
def trays_needed (cookies_for_classmates : ℕ) (cookies_for_teachers : ℕ) (cookies_per_tray : ℕ) : ℕ :=
  (cookies_for_classmates + cookies_for_teachers + cookies_per_tray - 1) / cookies_per_tray

/-- Proof that Marian needs 31 trays to prepare cookies for classmates and teachers -/
theorem marian_needs_31_trays :
  trays_needed 276 92 12 = 31 := by
  sorry

end NUMINAMATH_CALUDE_marian_needs_31_trays_l1423_142358


namespace NUMINAMATH_CALUDE_inequality_proof_l1423_142368

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 / (x + y) + 4 / (y + z) + 9 / (x + z) ≥ 18 / (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1423_142368


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1423_142383

/-- Theorem about a triangle ABC with specific side lengths and angle properties -/
theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  -- Triangle inequality and positive side lengths
  a + b > c ∧ b + c > a ∧ c + a > b →
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- A, B, C form angles of a triangle
  A + B + C = π →
  -- Side lengths correspond to opposite angles
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Conclusions
  A = π / 6 ∧
  (1 / 2) * a * b * Real.sin C = (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1423_142383


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l1423_142317

-- Define a point in 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the given point P
def P : Point := (-1, 2)

-- Define symmetry with respect to the origin
def symmetricPoint (p : Point) : Point :=
  (-p.1, -p.2)

-- Theorem statement
theorem symmetric_point_of_P :
  symmetricPoint P = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l1423_142317


namespace NUMINAMATH_CALUDE_max_value_of_a_l1423_142349

theorem max_value_of_a (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1423_142349


namespace NUMINAMATH_CALUDE_routes_between_plains_cities_l1423_142361

theorem routes_between_plains_cities
  (total_cities : Nat)
  (mountainous_cities : Nat)
  (plains_cities : Nat)
  (total_routes : Nat)
  (mountainous_routes : Nat)
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : mountainous_cities + plains_cities = total_cities)
  (h5 : total_routes = 150)
  (h6 : mountainous_routes = 21) :
  ∃ (plains_routes : Nat),
    plains_routes = 81 ∧
    plains_routes + mountainous_routes + (total_routes - plains_routes - mountainous_routes) = total_routes :=
by sorry

end NUMINAMATH_CALUDE_routes_between_plains_cities_l1423_142361


namespace NUMINAMATH_CALUDE_paint_usage_l1423_142332

theorem paint_usage (mary_paint mike_paint sun_paint total_paint : ℝ) 
  (h1 : mike_paint = mary_paint + 2)
  (h2 : sun_paint = 5)
  (h3 : total_paint = 13)
  (h4 : mary_paint + mike_paint + sun_paint = total_paint) :
  mary_paint = 3 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l1423_142332


namespace NUMINAMATH_CALUDE_survey_sample_size_l1423_142395

/-- Represents a survey with its characteristics -/
structure Survey where
  surveyors : ℕ
  households : ℕ
  questionnaires : ℕ

/-- Definition of sample size for a survey -/
def sampleSize (s : Survey) : ℕ := s.questionnaires

/-- Theorem stating that the sample size is equal to the number of questionnaires -/
theorem survey_sample_size (s : Survey) : sampleSize s = s.questionnaires := by
  sorry

/-- The specific survey described in the problem -/
def cityCenterSurvey : Survey := {
  surveyors := 400,
  households := 10000,
  questionnaires := 30000
}

#eval sampleSize cityCenterSurvey

end NUMINAMATH_CALUDE_survey_sample_size_l1423_142395


namespace NUMINAMATH_CALUDE_max_daily_profit_l1423_142398

/-- Represents the daily profit function for a store selling football souvenir books. -/
def daily_profit (x : ℝ) : ℝ :=
  (x - 40) * (-10 * x + 740)

/-- Theorem stating the maximum daily profit and the corresponding selling price. -/
theorem max_daily_profit :
  let cost_price : ℝ := 40
  let initial_price : ℝ := 44
  let initial_sales : ℝ := 300
  let price_range : Set ℝ := {x | 44 ≤ x ∧ x ≤ 52}
  let sales_decrease_rate : ℝ := 10
  ∃ (max_price : ℝ), max_price ∈ price_range ∧
    ∀ (x : ℝ), x ∈ price_range →
      daily_profit x ≤ daily_profit max_price ∧
      daily_profit max_price = 2640 ∧
      max_price = 52 :=
by sorry


end NUMINAMATH_CALUDE_max_daily_profit_l1423_142398


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1423_142335

/-- A linear function with slope k and y-intercept b -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

/-- Predicate for a point (x, y) being in quadrant I -/
def InQuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Predicate for a point (x, y) being in quadrant II -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate for a point (x, y) being in quadrant IV -/
def InQuadrantIV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the graph of y = 2x + 1 passes through quadrants I, II, and IV -/
theorem linear_function_quadrants :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (y₁ = LinearFunction 2 1 x₁) ∧ InQuadrantI x₁ y₁ ∧
    (y₂ = LinearFunction 2 1 x₂) ∧ InQuadrantII x₂ y₂ ∧
    (y₃ = LinearFunction 2 1 x₃) ∧ InQuadrantIV x₃ y₃ :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_quadrants_l1423_142335


namespace NUMINAMATH_CALUDE_sequence_property_l1423_142359

/-- A sequence where all terms are distinct starting from index 2 -/
def DistinctSequence (x : ℕ → ℝ) : Prop :=
  ∀ i j, i ≥ 2 → j ≥ 2 → i ≠ j → x i ≠ x j

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (x : ℕ → ℝ) : Prop :=
  ∀ n, x n = (x (n - 1) + 98 * x n + x (n + 1)) / 100

theorem sequence_property (x : ℕ → ℝ) 
    (h1 : DistinctSequence x) 
    (h2 : SatisfiesRecurrence x) : 
  Real.sqrt ((x 2023 - x 1) / 2022 * (2021 / (x 2023 - x 2))) + 2021 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1423_142359


namespace NUMINAMATH_CALUDE_sticker_distribution_l1423_142379

/-- The number of ways to distribute indistinguishable objects into distinguishable groups -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 1365 ways to distribute 11 indistinguishable stickers into 5 distinguishable sheets of paper -/
theorem sticker_distribution : distribute 11 5 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1423_142379


namespace NUMINAMATH_CALUDE_red_balls_count_l1423_142323

theorem red_balls_count (total_balls : ℕ) (red_frequency : ℚ) (h1 : total_balls = 40) (h2 : red_frequency = 15 / 100) : 
  ⌊total_balls * red_frequency⌋ = 6 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1423_142323


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l1423_142360

def primes : List Nat := [11, 17, 19, 23, 29, 37, 41]

def is_divisible_by_all (n : Nat) (lst : List Nat) : Prop :=
  ∀ p ∈ lst, (n % p = 0)

theorem smallest_number_divisible_by_primes :
  ∀ n : Nat,
    (n < 3075837206 →
      ¬(is_divisible_by_all (n - 27) primes)) ∧
    (is_divisible_by_all (3075837206 - 27) primes) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l1423_142360


namespace NUMINAMATH_CALUDE_range_of_m_for_B_subset_A_l1423_142321

/-- The set B defined as {x | -m < x < 2} -/
def B (m : ℝ) : Set ℝ := {x | -m < x ∧ x < 2}

/-- Theorem stating the range of m for which B is a subset of A -/
theorem range_of_m_for_B_subset_A (A : Set ℝ) :
  (∀ m : ℝ, B m ⊆ A) ↔ (∀ m : ℝ, m ≤ (1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_B_subset_A_l1423_142321


namespace NUMINAMATH_CALUDE_book_purchase_ratio_l1423_142313

/-- The number of people who purchased both books A and B -/
def both : ℕ := 500

/-- The number of people who purchased only book A -/
def only_A : ℕ := 1000

/-- The number of people who purchased only book B -/
def only_B : ℕ := both / 2

/-- The total number of people who purchased book A -/
def total_A : ℕ := only_A + both

/-- The total number of people who purchased book B -/
def total_B : ℕ := only_B + both

/-- The ratio of people who purchased book A to those who purchased book B is 2:1 -/
theorem book_purchase_ratio : total_A / total_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_ratio_l1423_142313


namespace NUMINAMATH_CALUDE_remaining_distance_to_hotel_l1423_142315

def totalDistance : ℝ := 1200

def drivingSegments : List (ℝ × ℝ) := [
  (60, 2),   -- 60 miles/hour for 2 hours
  (40, 1),   -- 40 miles/hour for 1 hour
  (70, 2.5), -- 70 miles/hour for 2.5 hours
  (50, 4),   -- 50 miles/hour for 4 hours
  (80, 1),   -- 80 miles/hour for 1 hour
  (60, 3)    -- 60 miles/hour for 3 hours
]

def distanceTraveled : ℝ := (drivingSegments.map (fun (speed, time) => speed * time)).sum

theorem remaining_distance_to_hotel : 
  totalDistance - distanceTraveled = 405 := by sorry

end NUMINAMATH_CALUDE_remaining_distance_to_hotel_l1423_142315


namespace NUMINAMATH_CALUDE_vector_addition_proof_l1423_142378

def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

theorem vector_addition_proof : 
  (2 : ℝ) • a + b = (7, -3, 2) := by sorry

end NUMINAMATH_CALUDE_vector_addition_proof_l1423_142378


namespace NUMINAMATH_CALUDE_equation_proof_l1423_142391

theorem equation_proof : (5568 / 87 : ℝ)^(1/3) + (72 * 2 : ℝ)^(1/2) = (256 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1423_142391


namespace NUMINAMATH_CALUDE_eagle_types_total_l1423_142376

theorem eagle_types_total (types_per_section : ℕ) (num_sections : ℕ) (h1 : types_per_section = 6) (h2 : num_sections = 3) :
  types_per_section * num_sections = 18 := by
  sorry

end NUMINAMATH_CALUDE_eagle_types_total_l1423_142376


namespace NUMINAMATH_CALUDE_fraction_equality_l1423_142333

theorem fraction_equality (a : ℕ+) : (a : ℚ) / ((a : ℚ) + 36) = 775 / 1000 → a = 124 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1423_142333


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l1423_142331

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the axioms
variable (different_lines : ∀ a b l : Line, a ≠ b ∧ b ≠ l ∧ a ≠ l)
variable (non_coincident_planes : ∀ α β : Plane, α ≠ β)

-- State the theorem
theorem line_perpendicular_to_plane_and_parallel_line 
  (a b l : Line) (α : Plane) :
  parallel a b → perpendicular_line_plane l α → perpendicular_line_line l b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l1423_142331


namespace NUMINAMATH_CALUDE_b_min_at_3_l1423_142357

def a (n : ℕ+) : ℕ := n

def S (n : ℕ+) : ℕ := n * (n + 1) / 2

def b (n : ℕ+) : ℚ := (2 * S n + 7) / n

theorem b_min_at_3 :
  ∀ n : ℕ+, n ≠ 3 → b n > b 3 :=
sorry

end NUMINAMATH_CALUDE_b_min_at_3_l1423_142357


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_2057_l1423_142390

theorem tens_digit_of_23_pow_2057 : ∃ n : ℕ, 23^2057 ≡ 60 + n [ZMOD 100] ∧ n < 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_2057_l1423_142390


namespace NUMINAMATH_CALUDE_sector_central_angle_l1423_142363

theorem sector_central_angle (area : Real) (radius : Real) (central_angle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  area = 1 / 2 * central_angle * radius ^ 2 →
  central_angle = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1423_142363


namespace NUMINAMATH_CALUDE_triangle_perimeter_bounds_l1423_142350

theorem triangle_perimeter_bounds (a b c : ℝ) (h : a * b + b * c + c * a = 12) :
  let k := a + b + c
  6 ≤ k ∧ k ≤ 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bounds_l1423_142350


namespace NUMINAMATH_CALUDE_house_transaction_loss_l1423_142381

theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  gain_percent = 0.20 → 
  initial_value * (1 - loss_percent) * (1 + gain_percent) - initial_value = 240 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l1423_142381


namespace NUMINAMATH_CALUDE_percentage_problem_l1423_142347

theorem percentage_problem (x : ℝ) (h : 75 = 0.6 * x) : x = 125 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1423_142347


namespace NUMINAMATH_CALUDE_cubic_equation_ratio_l1423_142322

theorem cubic_equation_ratio (a b c d : ℝ) (h : a ≠ 0) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) →
  c / d = -1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_ratio_l1423_142322


namespace NUMINAMATH_CALUDE_optimal_discount_sequence_l1423_142338

/-- The price of the coffee bag before discounts -/
def initial_price : ℝ := 18

/-- The fixed discount amount -/
def fixed_discount : ℝ := 3

/-- The percentage discount as a decimal -/
def percentage_discount : ℝ := 0.15

/-- The price after applying fixed discount then percentage discount -/
def price_fixed_then_percentage : ℝ := (initial_price - fixed_discount) * (1 - percentage_discount)

/-- The price after applying percentage discount then fixed discount -/
def price_percentage_then_fixed : ℝ := initial_price * (1 - percentage_discount) - fixed_discount

theorem optimal_discount_sequence :
  price_fixed_then_percentage - price_percentage_then_fixed = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_sequence_l1423_142338


namespace NUMINAMATH_CALUDE_water_requirement_l1423_142348

/-- The number of households in the village -/
def num_households : ℕ := 10

/-- The total amount of water available in litres -/
def total_water : ℕ := 6000

/-- The number of months the water lasts -/
def num_months : ℕ := 4

/-- The amount of water required per household per month -/
def water_per_household_per_month : ℕ := total_water / (num_households * num_months)

/-- Theorem stating that the water required per household per month is 150 litres -/
theorem water_requirement : water_per_household_per_month = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_l1423_142348


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1423_142301

/-- The area of a triangle with base 18 and height 6 is 54 -/
theorem triangle_area : Real → Real → Real → Prop :=
  fun base height area =>
    base = 18 ∧ height = 6 → area = (base * height) / 2 → area = 54

-- The proof is omitted
theorem triangle_area_proof : triangle_area 18 6 54 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1423_142301


namespace NUMINAMATH_CALUDE_calculate_expression_l1423_142302

theorem calculate_expression : (2 * Real.sqrt 48 - 3 * Real.sqrt (1/3)) / Real.sqrt 6 = 7 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1423_142302


namespace NUMINAMATH_CALUDE_optimal_order_l1423_142362

variable (p1 p2 p3 : ℝ)

-- Probabilities are between 0 and 1
axiom prob_range1 : 0 ≤ p1 ∧ p1 ≤ 1
axiom prob_range2 : 0 ≤ p2 ∧ p2 ≤ 1
axiom prob_range3 : 0 ≤ p3 ∧ p3 ≤ 1

-- Ordering of probabilities
axiom prob_order : p3 < p1 ∧ p1 < p2

-- Function to calculate probability of winning two games in a row
def win_probability (p_first p_second p_third : ℝ) : ℝ :=
  p_first * p_second + (1 - p_first) * p_second * p_third

-- Theorem stating that playing against p2 (highest probability) second is optimal
theorem optimal_order :
  win_probability p1 p2 p3 > win_probability p2 p1 p3 ∧
  win_probability p3 p2 p1 > win_probability p2 p3 p1 :=
sorry

end NUMINAMATH_CALUDE_optimal_order_l1423_142362


namespace NUMINAMATH_CALUDE_star_three_neg_two_thirds_l1423_142307

-- Define the ☆ operation
def star (x y : ℚ) : ℚ := x^2 + x*y

-- State the theorem
theorem star_three_neg_two_thirds : star 3 (-2/3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_star_three_neg_two_thirds_l1423_142307


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l1423_142366

/-- A function that returns true if a number is a single-digit prime -/
def isSingleDigitPrime (p : ℕ) : Prop :=
  p < 10 ∧ Nat.Prime p

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    isSingleDigitPrime d ∧
    isSingleDigitPrime e ∧
    Nat.Prime (d^2 + e^2) ∧
    n = d * e * (d^2 + e^2) ∧
    (∀ (m : ℕ), m > n →
      ¬(∃ (p q r : ℕ), isSingleDigitPrime p ∧
                        isSingleDigitPrime q ∧
                        Nat.Prime r ∧
                        r = p^2 + q^2 ∧
                        m = p * q * r)) ∧
    sumOfDigits n = 11 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l1423_142366


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1423_142369

/-- Given a geometric sequence with first term 1, last term 9, and middle terms a, b, c, prove that b = 3 -/
theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h_sequence : ∃ (r : ℝ), r > 0 ∧ a = 1 * r ∧ b = a * r ∧ c = b * r ∧ 9 = c * r) : 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1423_142369


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1423_142319

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 + 4 * x + 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 + 6 * x = 5
  let sol1_1 : ℝ := -1 + Real.sqrt 2 / 2
  let sol1_2 : ℝ := -1 - Real.sqrt 2 / 2
  let sol2_1 : ℝ := -3 + Real.sqrt 14
  let sol2_2 : ℝ := -3 - Real.sqrt 14
  (eq1 sol1_1 ∧ eq1 sol1_2) ∧ (eq2 sol2_1 ∧ eq2 sol2_2) := by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1423_142319


namespace NUMINAMATH_CALUDE_arithmetic_sequence_intersection_l1423_142371

/-- Given two arithmetic sequences {a_n} and {b_n}, prove that they intersect at n = 5 -/
theorem arithmetic_sequence_intersection :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  let b : ℕ → ℤ := λ n => -2 + 4 * (n - 1)
  ∃! n : ℕ, a n = b n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_intersection_l1423_142371


namespace NUMINAMATH_CALUDE_range_of_a_l1423_142339

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → a > 2 * x - 1) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1423_142339


namespace NUMINAMATH_CALUDE_initial_type_x_plants_l1423_142345

def initial_total : ℕ := 50
def final_total : ℕ := 1042
def days : ℕ := 12
def x_growth_factor : ℕ := 2^4  -- Type X doubles 4 times in 12 days
def y_growth_factor : ℕ := 3^3  -- Type Y triples 3 times in 12 days

theorem initial_type_x_plants : 
  ∃ (x y : ℕ), 
    x + y = initial_total ∧ 
    x_growth_factor * x + y_growth_factor * y = final_total ∧ 
    x = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_type_x_plants_l1423_142345


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1423_142388

theorem arctan_equation_solution (y : ℝ) : 
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = π/4 → y = 31/9 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1423_142388


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_equation_l1423_142377

theorem product_of_roots_quadratic_equation :
  ∀ x₁ x₂ : ℝ, (x₁^2 + x₁ - 2 = 0) → (x₂^2 + x₂ - 2 = 0) → x₁ * x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_equation_l1423_142377


namespace NUMINAMATH_CALUDE_casey_stay_is_three_months_l1423_142326

/-- Calculates the number of months Casey stays at the motel --/
def casey_stay_duration (weekly_rate : ℕ) (monthly_rate : ℕ) (weeks_per_month : ℕ) (total_savings : ℕ) : ℕ :=
  let monthly_cost_weekly := weekly_rate * weeks_per_month
  let savings_per_month := monthly_cost_weekly - monthly_rate
  total_savings / savings_per_month

/-- Proves that Casey stays for 3 months given the specified rates and savings --/
theorem casey_stay_is_three_months :
  casey_stay_duration 280 1000 4 360 = 3 := by
  sorry

end NUMINAMATH_CALUDE_casey_stay_is_three_months_l1423_142326


namespace NUMINAMATH_CALUDE_triangular_grid_theorem_l1423_142384

/-- Represents an infinite triangular grid with black unit equilateral triangles -/
structure TriangularGrid where
  black_triangles : ℕ

/-- Represents an equilateral triangle whose sides align with grid lines -/
structure AlignedTriangle where
  -- Add necessary fields

/-- Checks if there's exactly one black triangle outside the given aligned triangle -/
def has_one_outside (grid : TriangularGrid) (triangle : AlignedTriangle) : Prop :=
  sorry

/-- The main theorem statement -/
theorem triangular_grid_theorem (N : ℕ) :
  N > 0 →
  (∃ (grid : TriangularGrid) (triangle : AlignedTriangle),
    grid.black_triangles = N ∧ has_one_outside grid triangle) ↔
  N = 1 ∨ N = 2 ∨ N = 3 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_theorem_l1423_142384


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1423_142305

theorem cubic_root_sum (d e f : ℕ) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  let x : ℝ := (Real.rpow d (1/3) + Real.rpow e (1/3) + 3) / f
  (27 * x^3 - 15 * x^2 - 9 * x - 3 = 0) →
  d + e + f = 126 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1423_142305


namespace NUMINAMATH_CALUDE_money_relation_l1423_142324

theorem money_relation (a b : ℝ) 
  (h1 : 8 * a - b = 98) 
  (h2 : 2 * a + b > 36) : 
  a > 13.4 ∧ b > 9.2 := by
sorry

end NUMINAMATH_CALUDE_money_relation_l1423_142324


namespace NUMINAMATH_CALUDE_cube_surface_area_l1423_142375

/-- The surface area of a cube with side length 8 centimeters is 384 square centimeters. -/
theorem cube_surface_area : 
  let side_length : ℝ := 8
  let surface_area : ℝ := 6 * side_length * side_length
  surface_area = 384 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1423_142375


namespace NUMINAMATH_CALUDE_bob_wins_for_S_l1423_142399

/-- A set of lattice points in the Cartesian plane -/
def LatticeSet := Set (ℤ × ℤ)

/-- The set S defined by m and n -/
def S (m n : ℕ) : LatticeSet :=
  {p : ℤ × ℤ | m ≤ p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 ≤ n}

/-- Count of points on a line -/
def LineCount := ℕ

/-- Information provided by Alice: counts of points on horizontal, vertical, and diagonal lines -/
structure AliceInfo :=
  (horizontal : ℤ → LineCount)
  (vertical : ℤ → LineCount)
  (diagonalPos : ℤ → LineCount)  -- y = x + k
  (diagonalNeg : ℤ → LineCount)  -- y = -x + k

/-- Generate AliceInfo from a given set -/
def getAliceInfo (s : LatticeSet) : AliceInfo :=
  sorry

/-- Bob's winning condition -/
def BobCanWin (s : LatticeSet) : Prop :=
  ∀ t : LatticeSet, getAliceInfo s = getAliceInfo t → s = t

/-- Main theorem -/
theorem bob_wins_for_S (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  BobCanWin (S m n) :=
sorry

end NUMINAMATH_CALUDE_bob_wins_for_S_l1423_142399


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l1423_142382

theorem parabola_tangent_line (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 2*x → x = 2) ∧ 
  (2*2 = 2^2 + b*2 + c) ∧
  (∀ x, 2*x + b = 2) →
  b = -2 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l1423_142382


namespace NUMINAMATH_CALUDE_function_equality_l1423_142304

theorem function_equality (a b : ℝ) : 
  (∀ x, (x^2 + 4*x + 3 = (a*x + b)^2 + 4*(a*x + b) + 3)) → 
  ((a + b = -8) ∨ (a + b = 4)) := by
sorry

end NUMINAMATH_CALUDE_function_equality_l1423_142304


namespace NUMINAMATH_CALUDE_horse_speed_l1423_142341

theorem horse_speed (field_area : ℝ) (run_time : ℝ) (horse_speed : ℝ) : 
  field_area = 576 →
  run_time = 8 →
  horse_speed = (4 * Real.sqrt field_area) / run_time →
  horse_speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_horse_speed_l1423_142341


namespace NUMINAMATH_CALUDE_silver_dollar_problem_l1423_142303

/-- The problem of calculating the total value of silver dollars -/
theorem silver_dollar_problem (x y : ℕ) : 
  -- Mr. Ha owns x silver dollars, which is 2/3 of Mr. Phung's amount
  x = (2 * y) / 3 →
  -- Mr. Phung has y silver dollars, which is 16 more than Mr. Chiu's amount
  y = 56 + 16 →
  -- The total value of all silver dollars is $483.75
  (x + y + 56 + (((x + y + 56) * 120) / 100)) * (5 / 4) = 96750 / 200 := by
  sorry

end NUMINAMATH_CALUDE_silver_dollar_problem_l1423_142303


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1423_142336

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1423_142336


namespace NUMINAMATH_CALUDE_complex_power_of_four_l1423_142393

theorem complex_power_of_four :
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 =
  -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_four_l1423_142393


namespace NUMINAMATH_CALUDE_problem_statement_l1423_142325

def n : ℕ := 2^2015 - 1

def s_q (q k : ℕ) : ℕ := sorry

def f_n (x : ℕ) : ℕ := sorry

def N : ℕ := sorry

theorem problem_statement : 
  N ≡ 382 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1423_142325


namespace NUMINAMATH_CALUDE_cupcakes_brought_is_correct_l1423_142343

/-- The number of cupcakes Dani brought to her 2nd-grade class. -/
def cupcakes_brought : ℕ := 30

/-- The total number of students in the class, including Dani. -/
def total_students : ℕ := 27

/-- The number of teachers in the class. -/
def teachers : ℕ := 1

/-- The number of teacher's aids in the class. -/
def teacher_aids : ℕ := 1

/-- The number of students who called in sick. -/
def sick_students : ℕ := 3

/-- The number of cupcakes left after distribution. -/
def leftover_cupcakes : ℕ := 4

/-- Theorem stating that the number of cupcakes Dani brought is correct. -/
theorem cupcakes_brought_is_correct :
  cupcakes_brought = 
    (total_students - sick_students + teachers + teacher_aids) + leftover_cupcakes :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_brought_is_correct_l1423_142343


namespace NUMINAMATH_CALUDE_fraction_simplification_l1423_142385

theorem fraction_simplification : (1 : ℚ) / 462 + 19 / 42 = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1423_142385


namespace NUMINAMATH_CALUDE_at_least_one_zero_negation_l1423_142351

theorem at_least_one_zero_negation (a b : ℝ) :
  ¬(a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_at_least_one_zero_negation_l1423_142351


namespace NUMINAMATH_CALUDE_ratio_equality_l1423_142312

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2018 / (n + 1) * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2020 / (n + 1) * b (n + 1) + b n

/-- Theorem stating the equality of the ratio of specific terms in sequences a and b -/
theorem ratio_equality : a 1010 / 1010 = b 1009 / 1009 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1423_142312


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l1423_142320

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x^2 + 1/x^2 = 6) : x^4 + 1/x^4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l1423_142320


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1423_142354

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) (h : f b c 0 = f b c 2) :
  f b c (3/2) < f b c 0 ∧ f b c 0 < f b c (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1423_142354


namespace NUMINAMATH_CALUDE_abc_equation_solutions_l1423_142365

theorem abc_equation_solutions (a b c : ℕ+) :
  a * b * c + a * b + c = a ^ 3 →
  ((b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1))) :=
sorry

end NUMINAMATH_CALUDE_abc_equation_solutions_l1423_142365


namespace NUMINAMATH_CALUDE_min_pizzas_for_johns_car_l1423_142353

/-- Calculates the minimum number of pizzas needed to recover car cost -/
def min_pizzas_to_recover_cost (car_cost : ℕ) (earnings_per_pizza : ℕ) (expenses_per_pizza : ℕ) : ℕ :=
  ((car_cost + (earnings_per_pizza - expenses_per_pizza - 1)) / (earnings_per_pizza - expenses_per_pizza))

/-- Theorem: Given the specified conditions, the minimum number of pizzas to recover car cost is 1667 -/
theorem min_pizzas_for_johns_car : 
  min_pizzas_to_recover_cost 5000 10 7 = 1667 := by
  sorry

#eval min_pizzas_to_recover_cost 5000 10 7

end NUMINAMATH_CALUDE_min_pizzas_for_johns_car_l1423_142353


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1423_142311

theorem sqrt_equation_solution (c : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 9*c)) + Real.sqrt (3 + Real.sqrt (3 + c)) = 3 + 3 * Real.sqrt 3 →
  c = 33 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1423_142311


namespace NUMINAMATH_CALUDE_ellipse_properties_l1423_142356

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the line l -/
def line_l (x : ℝ) : Prop := x = -3

theorem ellipse_properties :
  ∀ a b : ℝ,
  a > b ∧ b > 0 ∧
  2 * Real.sqrt 3 = 2 * b ∧
  Real.sqrt 2 / 2 = Real.sqrt (a^2 - b^2) / a →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 6 + y^2 / 3 = 1) ∧
  (∃ min_value : ℝ,
    min_value = 0 ∧
    ∀ x y : ℝ,
    ellipse_C x y a b ∧ y > 0 →
    (x + 3)^2 - y^2 ≥ min_value) ∧
  (∃ m : ℝ,
    m = 9/8 ∧
    ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ a b ∧
    ellipse_C x₂ y₂ a b ∧
    y₁ = 1/4 * x₁ + m ∧
    y₂ = 1/4 * x₂ + m ∧
    ∃ xg yg : ℝ,
    ellipse_C xg yg a b ∧
    xg - (-3) = x₂ - x₁ ∧
    yg - 3 = y₂ - y₁) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1423_142356


namespace NUMINAMATH_CALUDE_remainder_problem_l1423_142346

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 186 = 19)
  (hy : y % 248 = 23)
  (hz : z % 372 = 29) :
  ((x * y * z) + 47) % 93 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1423_142346


namespace NUMINAMATH_CALUDE_parallel_line_segment_length_l1423_142372

/-- Given a triangle with sides a, b, c, and lines parallel to the sides drawn through an interior point,
    if the segments of these lines within the triangle are equal in length x, then
    x = 2 / (1/a + 1/b + 1/c) -/
theorem parallel_line_segment_length (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  x > 0 → x = 2 / (1/a + 1/b + 1/c) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_segment_length_l1423_142372


namespace NUMINAMATH_CALUDE_sum_of_ages_l1423_142373

/-- Given the ages of Eunji, Yuna, and Eunji's uncle, prove that the sum of Eunji's and Yuna's ages is 35 years. -/
theorem sum_of_ages (uncle_age : ℕ) (eunji_age : ℕ) (yuna_age : ℕ)
  (h1 : uncle_age = 41)
  (h2 : uncle_age = eunji_age + 25)
  (h3 : yuna_age = eunji_age + 3) :
  eunji_age + yuna_age = 35 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1423_142373


namespace NUMINAMATH_CALUDE_triangle_cosine_values_l1423_142364

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a - c = (√6/6)b and sin B = √6 sin C, then cos A = √6/4 and cos(2A - π/6) = (√15 - √3)/8 -/
theorem triangle_cosine_values (a b c A B C : ℝ) 
  (h1 : a - c = (Real.sqrt 6 / 6) * b)
  (h2 : Real.sin B = Real.sqrt 6 * Real.sin C) :
  Real.cos A = Real.sqrt 6 / 4 ∧ 
  Real.cos (2 * A - π / 6) = (Real.sqrt 15 - Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_values_l1423_142364


namespace NUMINAMATH_CALUDE_longest_tape_measure_l1423_142352

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 2400) 
  (hb : b = 3600) 
  (hc : c = 5400) : 
  Nat.gcd a (Nat.gcd b c) = 300 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l1423_142352


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_main_theorem_l1423_142334

def numerator : ℕ := 22 * 23 * 24 * 25 * 26 * 27
def denominator : ℕ := 2000

theorem units_digit_of_fraction (n d : ℕ) (h : d ≠ 0) : 
  (n / d) % 10 = ((n % (d * 10)) / d) % 10 :=
sorry

theorem main_theorem : (numerator / denominator) % 10 = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_main_theorem_l1423_142334


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l1423_142389

/-- The Stewart farm scenario -/
structure StewartFarm where
  total_horse_food : ℕ
  horse_food_per_horse : ℕ
  num_sheep : ℕ

/-- Calculate the number of horses on the farm -/
def num_horses (farm : StewartFarm) : ℕ :=
  farm.total_horse_food / farm.horse_food_per_horse

/-- Calculate the ratio of sheep to horses -/
def sheep_to_horses_ratio (farm : StewartFarm) : ℚ :=
  farm.num_sheep / (num_horses farm)

/-- Theorem: The ratio of sheep to horses on the Stewart farm is 6:7 -/
theorem stewart_farm_ratio :
  let farm := StewartFarm.mk 12880 230 48
  sheep_to_horses_ratio farm = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l1423_142389


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l1423_142392

/-- Calculates the total charge for a taxi trip with given conditions -/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_two_fifths_mile : ℚ) 
  (trip_distance : ℚ) (non_peak_discount : ℚ) (standard_car_discount : ℚ) : ℚ :=
  let base_charge := initial_fee + (trip_distance / (2/5)) * charge_per_two_fifths_mile
  let discount := base_charge * (non_peak_discount + standard_car_discount)
  base_charge - discount

/-- The total charge for the taxi trip is $4.95 -/
theorem taxi_charge_calculation :
  let initial_fee : ℚ := 235/100
  let charge_per_two_fifths_mile : ℚ := 35/100
  let trip_distance : ℚ := 36/10
  let non_peak_discount : ℚ := 7/100
  let standard_car_discount : ℚ := 3/100
  calculate_taxi_charge initial_fee charge_per_two_fifths_mile trip_distance 
    non_peak_discount standard_car_discount = 495/100 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l1423_142392


namespace NUMINAMATH_CALUDE_wall_construction_l1423_142330

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 1800

/-- Rate of the first bricklayer in bricks per hour -/
def rate1 : ℚ := total_bricks / 12

/-- Rate of the second bricklayer in bricks per hour -/
def rate2 : ℚ := total_bricks / 15

/-- Combined rate reduction when working together -/
def rate_reduction : ℕ := 15

/-- Time taken to complete the wall together -/
def time_taken : ℕ := 6

theorem wall_construction :
  (time_taken : ℚ) * (rate1 + rate2 - rate_reduction) = total_bricks := by sorry

end NUMINAMATH_CALUDE_wall_construction_l1423_142330


namespace NUMINAMATH_CALUDE_smallest_maximizer_of_g_l1423_142316

/-- Sum of all positive divisors of n -/
def σ (n : ℕ) : ℕ := sorry

/-- Function g(n) = σ(n) / n -/
def g (n : ℕ) : ℚ := (σ n : ℚ) / n

/-- Theorem stating that 6 is the smallest N maximizing g(n) for 1 ≤ n ≤ 100 -/
theorem smallest_maximizer_of_g :
  ∃ (N : ℕ), N = 6 ∧ 
  (∀ n : ℕ, 1 ≤ n → n ≤ 100 → n ≠ N → g n < g N) ∧
  (∀ m : ℕ, 1 ≤ m → m < N → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 100 ∧ k ≠ m ∧ g m ≤ g k) :=
sorry

end NUMINAMATH_CALUDE_smallest_maximizer_of_g_l1423_142316


namespace NUMINAMATH_CALUDE_equivalent_expression_l1423_142308

theorem equivalent_expression (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x^2 - 1) / x)) = -x / Real.sqrt (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l1423_142308


namespace NUMINAMATH_CALUDE_number_of_employees_l1423_142309

/-- Proves the number of employees in an organization given salary information --/
theorem number_of_employees
  (avg_salary : ℝ)
  (new_avg_salary : ℝ)
  (manager_salary : ℝ)
  (h1 : avg_salary = 1500)
  (h2 : new_avg_salary = 1650)
  (h3 : manager_salary = 4650) :
  ∃ (num_employees : ℕ),
    (num_employees : ℝ) * avg_salary + manager_salary = (num_employees + 1) * new_avg_salary ∧
    num_employees = 20 := by
  sorry


end NUMINAMATH_CALUDE_number_of_employees_l1423_142309


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2016_l1423_142342

theorem imaginary_unit_power_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2016_l1423_142342


namespace NUMINAMATH_CALUDE_exists_arithmetic_progression_of_5_primes_exists_arithmetic_progression_of_6_primes_l1423_142340

/-- An arithmetic progression of primes -/
def ArithmeticProgressionOfPrimes (n : ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ k : Fin n, Prime (a + k * d)

/-- There exists an arithmetic progression of 5 primes -/
theorem exists_arithmetic_progression_of_5_primes :
  ArithmeticProgressionOfPrimes 5 := by
  sorry

/-- There exists an arithmetic progression of 6 primes -/
theorem exists_arithmetic_progression_of_6_primes :
  ArithmeticProgressionOfPrimes 6 := by
  sorry

end NUMINAMATH_CALUDE_exists_arithmetic_progression_of_5_primes_exists_arithmetic_progression_of_6_primes_l1423_142340


namespace NUMINAMATH_CALUDE_complement_of_union_l1423_142337

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_of_union :
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1423_142337


namespace NUMINAMATH_CALUDE_previous_day_visitors_count_l1423_142306

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 666

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := 566

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := current_day_visitors - visitor_difference

theorem previous_day_visitors_count : previous_day_visitors = 100 := by
  sorry

end NUMINAMATH_CALUDE_previous_day_visitors_count_l1423_142306


namespace NUMINAMATH_CALUDE_emma_room_coverage_l1423_142328

/-- Represents the dimensions of Emma's room --/
structure RoomDimensions where
  rectangleLength : ℝ
  rectangleWidth : ℝ
  triangleBase : ℝ
  triangleHeight : ℝ

/-- Represents the tiles used to cover the room --/
structure Tiles where
  squareTiles : ℕ
  triangularTiles : ℕ
  squareTileArea : ℝ
  triangularTileBase : ℝ
  triangularTileHeight : ℝ

/-- Calculates the fraction of the room covered by tiles --/
def fractionalCoverage (room : RoomDimensions) (tiles : Tiles) : ℚ :=
  sorry

/-- Theorem stating that the fractional coverage of Emma's room is 3/20 --/
theorem emma_room_coverage :
  let room : RoomDimensions := {
    rectangleLength := 12,
    rectangleWidth := 20,
    triangleBase := 10,
    triangleHeight := 8
  }
  let tiles : Tiles := {
    squareTiles := 40,
    triangularTiles := 4,
    squareTileArea := 1,
    triangularTileBase := 1,
    triangularTileHeight := 1
  }
  fractionalCoverage room tiles = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_emma_room_coverage_l1423_142328


namespace NUMINAMATH_CALUDE_max_equal_covering_is_three_l1423_142387

/-- Represents a square covering on a cube face -/
structure SquareCovering where
  position : Fin 6 × Fin 6
  folded : Bool

/-- Represents the cube and its covering -/
structure CubeCovering where
  squares : List SquareCovering

/-- Check if a cell is covered by a square -/
def covers (s : SquareCovering) (cell : Fin 6 × Fin 6) : Bool :=
  sorry

/-- Count how many squares cover a given cell -/
def coverCount (cc : CubeCovering) (cell : Fin 6 × Fin 6 × Fin 3) : Nat :=
  sorry

/-- Check if the covering is valid (no overlaps, all 2x2) -/
def isValidCovering (cc : CubeCovering) : Bool :=
  sorry

/-- Check if all cells are covered equally -/
def isEqualCovering (cc : CubeCovering) : Bool :=
  sorry

/-- The main theorem -/
theorem max_equal_covering_is_three :
  ∀ (cc : CubeCovering),
    isValidCovering cc →
    isEqualCovering cc →
    ∃ (n : Nat), (∀ (cell : Fin 6 × Fin 6 × Fin 3), coverCount cc cell = n) ∧ n ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_max_equal_covering_is_three_l1423_142387


namespace NUMINAMATH_CALUDE_combinations_equal_twelve_l1423_142370

/-- The number of wall color choices -/
def wall_colors : Nat := 4

/-- The number of flooring type choices -/
def flooring_types : Nat := 3

/-- The total number of combinations of wall color and flooring type -/
def total_combinations : Nat := wall_colors * flooring_types

/-- Theorem: The total number of combinations is 12 -/
theorem combinations_equal_twelve : total_combinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twelve_l1423_142370


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_l1423_142329

-- Define a type for regular polygons
inductive RegularPolygon
  | Hexagon
  | Pentagon
  | Square
  | Triangle

-- Function to calculate the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℝ :=
  match p with
  | .Hexagon => 120
  | .Pentagon => 108
  | .Square => 90
  | .Triangle => 60

-- Function to check if a polygon can tile the plane
def canTilePlane (p : RegularPolygon) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ n * interiorAngle p = 360

-- Theorem stating that only the pentagon cannot tile the plane
theorem pentagon_cannot_tile :
  ∀ (p : RegularPolygon), ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_l1423_142329


namespace NUMINAMATH_CALUDE_special_polyhedron_ratio_l1423_142355

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  x : ℝ
  y : ℝ
  all_faces_isosceles : Prop
  edge_lengths : Prop
  vertex_degrees : Prop
  equal_dihedral_angles : Prop

/-- The theorem statement -/
theorem special_polyhedron_ratio 
  (P : SpecialPolyhedron)
  (h_faces : P.faces = 12)
  (h_edges : P.edges = 18)
  (h_vertices : P.vertices = 8)
  (h_isosceles : P.all_faces_isosceles)
  (h_edge_lengths : P.edge_lengths)
  (h_vertex_degrees : P.vertex_degrees)
  (h_dihedral_angles : P.equal_dihedral_angles)
  : P.x / P.y = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_special_polyhedron_ratio_l1423_142355


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1423_142394

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2

-- Define the arithmetic sequence property
def arithmetic_sequence (P : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), Real.sqrt ((P.1 + 1)^2 + P.2^2) = 2 - d ∧
             Real.sqrt ((P.1 - 1)^2 + P.2^2) = 2 + d

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (3 * Real.sqrt 7 / 7) * (x - 1) ∨
  y = -(3 * Real.sqrt 7 / 7) * (x - 1)

-- Define the perpendicular property
def perpendicular_property (P Q : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (Q.1 - F₁.1) + (P.2 - F₁.2) * (Q.2 - F₁.2) = 0

theorem ellipse_and_line_theorem :
  ∀ (P : ℝ × ℝ),
    point_on_ellipse P →
    arithmetic_sequence P →
    ∀ (Q : ℝ × ℝ),
      point_on_ellipse Q →
      Q.1 = F₂.1 →
      perpendicular_property P Q →
      line_m P.1 P.2 ∧ line_m Q.1 Q.2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1423_142394
