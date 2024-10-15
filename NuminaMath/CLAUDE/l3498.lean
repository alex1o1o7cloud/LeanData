import Mathlib

namespace NUMINAMATH_CALUDE_segment_length_product_l3498_349861

theorem segment_length_product (a : ℝ) : 
  (∃ b : ℝ, b ≠ a ∧ 
   ((3 * a - 5)^2 + (2 * a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2) ∧
   ((3 * b - 5)^2 + (2 * b - 5 - (-2))^2 = (3 * Real.sqrt 13)^2)) →
  (a * b = -1080 / 169) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l3498_349861


namespace NUMINAMATH_CALUDE_circle_equation_and_intersection_points_l3498_349864

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 1)^2 = 2}

-- Define the line y = x
def line_tangent : Set (ℝ × ℝ) :=
  {p | p.1 = p.2}

-- Define the line l: x - y + a = 0
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + a = 0}

theorem circle_equation_and_intersection_points (a : ℝ) 
  (h1 : a ≠ 0)
  (h2 : ∃ p, p ∈ circle_C ∩ line_tangent)
  (h3 : ∃ A B, A ∈ circle_C ∩ line_l a ∧ B ∈ circle_C ∩ line_l a ∧ A ≠ B)
  (h4 : ∀ A B, A ∈ circle_C ∩ line_l a → B ∈ circle_C ∩ line_l a → A ≠ B → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :
  (∀ p, p ∈ circle_C ↔ (p.1 - 3)^2 + (p.2 - 1)^2 = 2) ∧
  (a = Real.sqrt 2 - 2 ∨ a = -Real.sqrt 2 - 2) := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_intersection_points_l3498_349864


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l3498_349868

theorem largest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 35 = 0 → n ≤ 9985 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l3498_349868


namespace NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_proof_l3498_349877

/-- The probability of cutting a 1-meter rope at a random point such that the longer piece is at least three times as large as the shorter piece is 1/2. -/
theorem rope_cutting_probability : Real → Prop :=
  fun p => p = 1/2 ∧ 
    ∀ c : Real, 0 ≤ c ∧ c ≤ 1 →
      (((1 - c ≥ 3 * c) ∨ (c ≥ 3 * (1 - c))) ↔ (c ≤ 1/4 ∨ c ≥ 3/4)) ∧
      p = (1/4 - 0) + (1 - 3/4)

/-- Proof of the rope cutting probability theorem -/
theorem rope_cutting_probability_proof : ∃ p, rope_cutting_probability p :=
  sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_proof_l3498_349877


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3498_349805

theorem quadratic_function_property (b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  (f 1 = 0) → (f 3 = 0) → (f (-1) = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3498_349805


namespace NUMINAMATH_CALUDE_min_volume_to_prevent_explosion_l3498_349891

/-- Represents the relationship between pressure and volume for a balloon -/
structure Balloon where
  k : ℝ
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  h1 : ∀ v, pressure v = k / v
  h2 : pressure 3 = 8000
  h3 : ∀ v, pressure v > 40000 → volume v < v

/-- The minimum volume to prevent the balloon from exploding is 0.6 m³ -/
theorem min_volume_to_prevent_explosion (b : Balloon) : 
  ∀ v, v ≥ 0.6 → b.pressure v ≤ 40000 :=
sorry

#check min_volume_to_prevent_explosion

end NUMINAMATH_CALUDE_min_volume_to_prevent_explosion_l3498_349891


namespace NUMINAMATH_CALUDE_vector_operation_result_l3498_349873

theorem vector_operation_result :
  let a : ℝ × ℝ × ℝ := (3, -2, 1)
  let b : ℝ × ℝ × ℝ := (-2, 4, 0)
  let c : ℝ × ℝ × ℝ := (3, 0, 2)
  a - 2 • b + 4 • c = (19, -10, 9) :=
by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3498_349873


namespace NUMINAMATH_CALUDE_max_value_is_70_l3498_349836

/-- Represents the types of rocks available --/
inductive RockType
  | Seven
  | Three
  | Two

/-- The weight of a rock in pounds --/
def weight : RockType → ℕ
  | RockType.Seven => 7
  | RockType.Three => 3
  | RockType.Two => 2

/-- The value of a rock in dollars --/
def value : RockType → ℕ
  | RockType.Seven => 20
  | RockType.Three => 10
  | RockType.Two => 4

/-- The maximum weight Carl can carry in pounds --/
def maxWeight : ℕ := 21

/-- The minimum number of each type of rock available --/
def minAvailable : ℕ := 15

/-- A function to calculate the total value of a combination of rocks --/
def totalValue (combination : RockType → ℕ) : ℕ :=
  (combination RockType.Seven * value RockType.Seven) +
  (combination RockType.Three * value RockType.Three) +
  (combination RockType.Two * value RockType.Two)

/-- A function to calculate the total weight of a combination of rocks --/
def totalWeight (combination : RockType → ℕ) : ℕ :=
  (combination RockType.Seven * weight RockType.Seven) +
  (combination RockType.Three * weight RockType.Three) +
  (combination RockType.Two * weight RockType.Two)

/-- The main theorem stating that the maximum value of rocks Carl can carry is $70 --/
theorem max_value_is_70 :
  ∃ (combination : RockType → ℕ),
    (∀ rock, combination rock ≤ minAvailable) ∧
    totalWeight combination ≤ maxWeight ∧
    totalValue combination = 70 ∧
    (∀ other : RockType → ℕ,
      (∀ rock, other rock ≤ minAvailable) →
      totalWeight other ≤ maxWeight →
      totalValue other ≤ 70) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_is_70_l3498_349836


namespace NUMINAMATH_CALUDE_circle_area_l3498_349807

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (∃ (center_x center_y radius : ℝ),
    ∀ (x' y' : ℝ), (x' - center_x)^2 + (y' - center_y)^2 = radius^2 ↔
    3 * x'^2 + 3 * y'^2 - 9 * x' + 12 * y' + 27 = 0) →
  (π * (1/2)^2 : ℝ) = π/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l3498_349807


namespace NUMINAMATH_CALUDE_f_of_one_l3498_349811

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_of_one (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 4)
  (h_value : f (-5) = 1) : 
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_l3498_349811


namespace NUMINAMATH_CALUDE_project_selection_count_l3498_349896

/-- The number of key projects -/
def num_key_projects : ℕ := 4

/-- The number of general projects -/
def num_general_projects : ℕ := 6

/-- The number of projects to be selected from each category -/
def projects_per_category : ℕ := 2

/-- Calculates the number of ways to select projects with the given conditions -/
def select_projects : ℕ :=
  Nat.choose num_key_projects projects_per_category *
  Nat.choose num_general_projects projects_per_category -
  Nat.choose (num_key_projects - 1) projects_per_category *
  Nat.choose (num_general_projects - 1) projects_per_category

theorem project_selection_count :
  select_projects = 60 := by sorry

end NUMINAMATH_CALUDE_project_selection_count_l3498_349896


namespace NUMINAMATH_CALUDE_bird_feeder_theorem_l3498_349803

/-- Given a bird feeder with specified capacity and feeding rate, and accounting for stolen seed, 
    calculate the number of birds fed weekly. -/
theorem bird_feeder_theorem (feeder_capacity : ℚ) (birds_per_cup : ℕ) (stolen_amount : ℚ) : 
  feeder_capacity = 2 → 
  birds_per_cup = 14 → 
  stolen_amount = 1/2 → 
  (feeder_capacity - stolen_amount) * birds_per_cup = 21 := by
  sorry

end NUMINAMATH_CALUDE_bird_feeder_theorem_l3498_349803


namespace NUMINAMATH_CALUDE_ludwig_weekly_earnings_l3498_349859

/-- Calculates Ludwig's total earnings for a week --/
def ludwig_earnings (weekday_rate : ℚ) (weekend_rate : ℚ) (total_hours : ℕ) : ℚ :=
  let weekday_earnings := 4 * weekday_rate
  let weekend_earnings := 3 * weekend_rate / 2
  let regular_earnings := weekday_earnings + weekend_earnings
  let overtime_hours := max (total_hours - 48) 0
  let overtime_rate := weekend_rate * 3 / 8
  let overtime_earnings := overtime_hours * overtime_rate * 3 / 2
  regular_earnings + overtime_earnings

/-- Theorem stating Ludwig's earnings for the given week --/
theorem ludwig_weekly_earnings :
  ludwig_earnings 12 15 52 = 115.5 := by
  sorry


end NUMINAMATH_CALUDE_ludwig_weekly_earnings_l3498_349859


namespace NUMINAMATH_CALUDE_difference_between_shares_l3498_349835

/-- Represents the distribution of money among three people -/
structure MoneyDistribution where
  ratio : Fin 3 → ℕ
  vasimInitialShare : ℕ
  farukTaxRate : ℚ
  vasimTaxRate : ℚ
  ranjithTaxRate : ℚ

/-- Calculates the final share after tax for a given initial share and tax rate -/
def finalShareAfterTax (initialShare : ℕ) (taxRate : ℚ) : ℚ :=
  (1 - taxRate) * initialShare

/-- Theorem stating the difference between Ranjith's and Faruk's final shares -/
theorem difference_between_shares (d : MoneyDistribution) 
  (h1 : d.ratio 0 = 3) 
  (h2 : d.ratio 1 = 5) 
  (h3 : d.ratio 2 = 8) 
  (h4 : d.vasimInitialShare = 1500) 
  (h5 : d.farukTaxRate = 1/10) 
  (h6 : d.vasimTaxRate = 3/20) 
  (h7 : d.ranjithTaxRate = 3/25) : 
  finalShareAfterTax (d.ratio 2 * d.vasimInitialShare / d.ratio 1) d.ranjithTaxRate -
  finalShareAfterTax (d.ratio 0 * d.vasimInitialShare / d.ratio 1) d.farukTaxRate = 1302 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_shares_l3498_349835


namespace NUMINAMATH_CALUDE_segment_multiplication_l3498_349874

-- Define a segment as a pair of points
def Segment (α : Type*) := α × α

-- Define the length of a segment
def length {α : Type*} (s : Segment α) : ℝ := sorry

-- Define the multiplication of a segment by a scalar
def scaleSegment {α : Type*} (s : Segment α) (n : ℕ) : Segment α := sorry

-- Theorem statement
theorem segment_multiplication {α : Type*} (AB : Segment α) (n : ℕ) :
  ∃ (AC : Segment α), length AC = n * length AB :=
sorry

end NUMINAMATH_CALUDE_segment_multiplication_l3498_349874


namespace NUMINAMATH_CALUDE_line_circle_distance_sum_l3498_349869

-- Define the lines and circle
def line_l1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line_l2 (x y a : ℝ) : Prop := 4 * x - 2 * y + a = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the distance sum condition
def distance_sum_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, circle_C x y →
    (|2*x - y + 1| / Real.sqrt 5 + |4*x - 2*y + a| / Real.sqrt 20 = 2 * Real.sqrt 5)

-- Theorem statement
theorem line_circle_distance_sum (a : ℝ) :
  distance_sum_condition a → (a = 10 ∨ a = -18) :=
sorry

end NUMINAMATH_CALUDE_line_circle_distance_sum_l3498_349869


namespace NUMINAMATH_CALUDE_opposite_gender_officers_l3498_349809

theorem opposite_gender_officers (boys girls : ℕ) (h1 : boys = 18) (h2 : girls = 12) :
  boys * girls + girls * boys = 432 := by
  sorry

end NUMINAMATH_CALUDE_opposite_gender_officers_l3498_349809


namespace NUMINAMATH_CALUDE_triangle_area_relation_l3498_349839

/-- Given a triangle T with area Δ, and two triangles T' and T'' formed by successive altitudes
    with areas Δ' and Δ'' respectively, prove that if Δ' = 30 and Δ'' = 20, then Δ = 45. -/
theorem triangle_area_relation (Δ Δ' Δ'' : ℝ) : Δ' = 30 → Δ'' = 20 → Δ = 45 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_relation_l3498_349839


namespace NUMINAMATH_CALUDE_unique_n_divisibility_l3498_349828

theorem unique_n_divisibility : ∃! n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_divisibility_l3498_349828


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3498_349812

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 40*x^2 + 144 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 40*y^2 + 144 = 0 → x ≤ y) ∧
  x = -6 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3498_349812


namespace NUMINAMATH_CALUDE_equation_solutions_l3498_349850

theorem equation_solutions : 
  (∃ (S₁ S₂ : Set ℝ), 
    S₁ = {x : ℝ | x * (x + 4) = -5 * (x + 4)} ∧ 
    S₂ = {x : ℝ | (x + 2)^2 = (2*x - 1)^2} ∧
    S₁ = {-4, -5} ∧
    S₂ = {3, -1/3}) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3498_349850


namespace NUMINAMATH_CALUDE_function_properties_l3498_349866

variable (a b : ℝ × ℝ)

def f (x : ℝ) : ℝ := (x * a.1 + b.1) * (x * b.2 - a.2)

theorem function_properties
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0)  -- perpendicular vectors
  (h4 : a.1^2 + a.2^2 ≠ b.1^2 + b.2^2)  -- different magnitudes
  : (∃ k : ℝ, ∀ x : ℝ, f a b x = k * x) ∧  -- first-order function
    (∀ x : ℝ, f a b x = -f a b (-x))  -- odd function
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l3498_349866


namespace NUMINAMATH_CALUDE_gangster_undetected_conditions_l3498_349889

/-- Configuration of streets and houses -/
structure StreetConfig where
  a : ℝ  -- Side length of houses
  street_distance : ℝ  -- Distance between parallel streets
  house_gap : ℝ  -- Distance between neighboring houses
  police_interval : ℝ  -- Interval between police officers

/-- Movement parameters -/
structure MovementParams where
  police_speed : ℝ  -- Speed of police officers
  gangster_speed : ℝ  -- Speed of the gangster
  gangster_direction : Bool  -- True if moving towards police, False otherwise

/-- Predicate to check if the gangster remains undetected -/
def remains_undetected (config : StreetConfig) (params : MovementParams) : Prop :=
  (params.gangster_direction = true) ∧ 
  ((params.gangster_speed = 2 * params.police_speed) ∨ 
   (params.gangster_speed = params.police_speed / 2))

/-- Main theorem: Conditions for the gangster to remain undetected -/
theorem gangster_undetected_conditions 
  (config : StreetConfig) 
  (params : MovementParams) :
  config.street_distance = 3 * config.a ∧ 
  config.house_gap = 2 * config.a ∧
  config.police_interval = 9 * config.a ∧
  params.police_speed > 0 →
  remains_undetected config params ↔ 
  (params.gangster_direction = true ∧ 
   (params.gangster_speed = 2 * params.police_speed ∨ 
    params.gangster_speed = params.police_speed / 2)) :=
by sorry

end NUMINAMATH_CALUDE_gangster_undetected_conditions_l3498_349889


namespace NUMINAMATH_CALUDE_sin_in_M_l3498_349800

/-- The set of functions f that satisfy f(x + T) = T * f(x) for some non-zero constant T -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (T : ℝ), T ≠ 0 ∧ ∀ x, f (x + T) = T * f x}

/-- Theorem stating the condition for sin(kx) to be in set M -/
theorem sin_in_M (k : ℝ) : 
  (fun x ↦ Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi :=
sorry

end NUMINAMATH_CALUDE_sin_in_M_l3498_349800


namespace NUMINAMATH_CALUDE_unique_remainder_mod_11_l3498_349817

theorem unique_remainder_mod_11 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_11_l3498_349817


namespace NUMINAMATH_CALUDE_multiplier_problem_l3498_349823

theorem multiplier_problem (x : ℝ) (h1 : x = 11) (h2 : 3 * x = (26 - x) + 18) :
  ∃ m : ℝ, m * x = (26 - x) + 18 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l3498_349823


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3498_349816

def has_exactly_two_integer_solutions (m : ℝ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧
    (x < 1 ∧ x > m - 1) ∧
    (y < 1 ∧ y > m - 1) ∧
    ∀ (z : ℤ), (z < 1 ∧ z > m - 1) → (z = x ∨ z = y)

theorem inequality_system_solutions (m : ℝ) :
  has_exactly_two_integer_solutions m ↔ -1 ≤ m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3498_349816


namespace NUMINAMATH_CALUDE_events_independent_l3498_349826

/-- Represents the outcome of a single coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents the outcome of tossing a coin twice -/
def DoubleToss := CoinToss × CoinToss

/-- Event A: the first toss is heads -/
def event_A (toss : DoubleToss) : Prop :=
  toss.1 = CoinToss.Heads

/-- Event B: the second toss is tails -/
def event_B (toss : DoubleToss) : Prop :=
  toss.2 = CoinToss.Tails

/-- The probability of an event occurring -/
def probability (event : DoubleToss → Prop) : ℝ :=
  sorry

/-- Theorem: Events A and B are mutually independent -/
theorem events_independent :
  probability (fun toss ↦ event_A toss ∧ event_B toss) =
  probability event_A * probability event_B :=
sorry

end NUMINAMATH_CALUDE_events_independent_l3498_349826


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3498_349890

theorem rectangular_to_polar_conversion :
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ x = -r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3498_349890


namespace NUMINAMATH_CALUDE_percentage_of_blue_flowers_l3498_349808

/-- Given a set of flowers with specific colors, calculate the percentage of blue flowers. -/
theorem percentage_of_blue_flowers (total : ℕ) (red : ℕ) (white : ℕ) (h1 : total = 10) (h2 : red = 4) (h3 : white = 2) :
  (total - red - white) / total * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_blue_flowers_l3498_349808


namespace NUMINAMATH_CALUDE_tank_depth_l3498_349892

/-- Proves that a tank with given dimensions and plastering cost has a depth of 6 meters -/
theorem tank_depth (length width : ℝ) (cost_per_sqm total_cost : ℝ) : 
  length = 25 → 
  width = 12 → 
  cost_per_sqm = 0.75 → 
  total_cost = 558 → 
  ∃ d : ℝ, d = 6 ∧ cost_per_sqm * (2 * (length * d) + 2 * (width * d) + (length * width)) = total_cost :=
by
  sorry

#check tank_depth

end NUMINAMATH_CALUDE_tank_depth_l3498_349892


namespace NUMINAMATH_CALUDE_congruence_solution_l3498_349806

theorem congruence_solution (m : ℤ) : 
  (13 * m) % 47 = 8 % 47 ↔ m % 47 = 20 % 47 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l3498_349806


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3498_349801

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < C → C < π →
  0 < A → A < π / 3 →
  (2 * a + b) / Real.cos B = -c / Real.cos C →
  (C = 2 * π / 3 ∧ 
   ∀ A' B', 0 < A' → A' < π / 3 → 
             Real.sin A' * Real.sin B' ≤ 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3498_349801


namespace NUMINAMATH_CALUDE_element_in_complement_l3498_349802

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {1, 5}

-- Define set P
def P : Set Nat := {2, 4}

-- Theorem statement
theorem element_in_complement : 3 ∈ (U \ (M ∪ P)) := by sorry

end NUMINAMATH_CALUDE_element_in_complement_l3498_349802


namespace NUMINAMATH_CALUDE_set_M_properties_l3498_349854

-- Define the set M
variable (M : Set ℝ)

-- Define the properties of M
variable (h_nonempty : M.Nonempty)
variable (h_two : 2 ∈ M)
variable (h_diff : ∀ x y, x ∈ M → y ∈ M → x - y ∈ M)

-- Theorem statement
theorem set_M_properties :
  (0 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → x + y ∈ M) ∧
  (∀ x, x ∈ M → x ≠ 0 → x ≠ 1 → (1 / (x * (x - 1))) ∈ M) :=
sorry

end NUMINAMATH_CALUDE_set_M_properties_l3498_349854


namespace NUMINAMATH_CALUDE_point_coordinates_l3498_349897

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the 2D plane -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating that a point in the second quadrant with given distances from axes has specific coordinates -/
theorem point_coordinates (A : Point) 
    (h1 : secondQuadrant A) 
    (h2 : distanceFromXAxis A = 5) 
    (h3 : distanceFromYAxis A = 6) : 
  A.x = -6 ∧ A.y = 5 := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l3498_349897


namespace NUMINAMATH_CALUDE_derivative_y_wrt_x_l3498_349858

noncomputable def x (t : ℝ) : ℝ := Real.log (1 / Real.tan t)

noncomputable def y (t : ℝ) : ℝ := 1 / (Real.cos t)^2

theorem derivative_y_wrt_x (t : ℝ) (h : Real.cos t ≠ 0) (h' : Real.sin t ≠ 0) :
  deriv y t / deriv x t = -2 * (Real.tan t)^2 :=
sorry

end NUMINAMATH_CALUDE_derivative_y_wrt_x_l3498_349858


namespace NUMINAMATH_CALUDE_negation_equivalence_l3498_349825

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3498_349825


namespace NUMINAMATH_CALUDE_james_basketball_score_l3498_349829

/-- Calculates the total points scored by James in a basketball game --/
def jamesScore (threePointers twoPointers freeThrows missedFreeThrows : ℕ) : ℤ :=
  3 * threePointers + 2 * twoPointers + freeThrows - missedFreeThrows

theorem james_basketball_score :
  jamesScore 13 20 5 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_james_basketball_score_l3498_349829


namespace NUMINAMATH_CALUDE_phone_number_fraction_l3498_349887

def is_valid_phone_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ n / 1000000 ≥ 3

def starts_with_3_to_9_ends_even (n : ℕ) : Prop :=
  is_valid_phone_number n ∧ n % 2 = 0

def count_valid_numbers : ℕ := 7 * 10^6

def count_start_3_to_9_end_even : ℕ := 7 * 10^5 * 5

theorem phone_number_fraction :
  (count_start_3_to_9_end_even : ℚ) / count_valid_numbers = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_phone_number_fraction_l3498_349887


namespace NUMINAMATH_CALUDE_exists_piecewise_linear_involution_l3498_349867

/-- A piecewise-linear function is a function whose graph is a union of a finite number of points and line segments. -/
def PiecewiseLinear (f : ℝ → ℝ) : Prop := sorry

theorem exists_piecewise_linear_involution :
  ∃ (f : ℝ → ℝ), PiecewiseLinear f ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x ∈ Set.Icc (-1) 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f (f x) = -x) :=
sorry

end NUMINAMATH_CALUDE_exists_piecewise_linear_involution_l3498_349867


namespace NUMINAMATH_CALUDE_union_A_B_l3498_349842

def A : Set ℕ := {1, 2}

def B : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem union_A_B : A ∪ B = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_A_B_l3498_349842


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3498_349884

theorem arithmetic_square_root_of_four : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 4 ∧ ∀ y : ℝ, y ≥ 0 ∧ y^2 = 4 → y = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3498_349884


namespace NUMINAMATH_CALUDE_number_of_girls_l3498_349895

theorem number_of_girls (total_pupils : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_pupils = 929 → boys = 387 → girls = total_pupils - boys → girls = 542 := by
sorry

end NUMINAMATH_CALUDE_number_of_girls_l3498_349895


namespace NUMINAMATH_CALUDE_josh_initial_money_l3498_349849

/-- The cost of the candy bar in dollars -/
def candy_cost : ℚ := 45 / 100

/-- The change Josh received in dollars -/
def change_received : ℚ := 135 / 100

/-- The initial amount of money Josh had -/
def initial_money : ℚ := candy_cost + change_received

theorem josh_initial_money : initial_money = 180 / 100 := by
  sorry

end NUMINAMATH_CALUDE_josh_initial_money_l3498_349849


namespace NUMINAMATH_CALUDE_smallest_solution_correct_l3498_349845

noncomputable def smallest_solution : ℝ := 4 - Real.sqrt 15 / 3

theorem smallest_solution_correct :
  let x := smallest_solution
  (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 5 / (y - 4)) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_correct_l3498_349845


namespace NUMINAMATH_CALUDE_group_collection_theorem_l3498_349820

/-- Calculates the total collection amount in rupees for a group where each member
    contributes as many paise as there are members in the group. -/
def total_collection (group_size : ℕ) : ℚ :=
  (group_size * group_size : ℚ) / 100

/-- Proves that for a group of 99 members, where each member contributes as many
    paise as there are members, the total collection amount is 98.01 rupees. -/
theorem group_collection_theorem :
  total_collection 99 = 98.01 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_theorem_l3498_349820


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3498_349856

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0 ↔ x ≤ 3 ∨ x ≥ 4) →
  a > 0 ∧ a + b + c > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3498_349856


namespace NUMINAMATH_CALUDE_jewelry_price_increase_is_10_l3498_349843

/-- Represents the increase in price of jewelry -/
def jewelry_price_increase : ℝ := sorry

/-- Original price of jewelry -/
def original_jewelry_price : ℝ := 30

/-- Original price of paintings -/
def original_painting_price : ℝ := 100

/-- New price of paintings after 20% increase -/
def new_painting_price : ℝ := original_painting_price * 1.2

/-- Total cost for 2 pieces of jewelry and 5 paintings -/
def total_cost : ℝ := 680

theorem jewelry_price_increase_is_10 :
  2 * (original_jewelry_price + jewelry_price_increase) + 5 * new_painting_price = total_cost ∧
  jewelry_price_increase = 10 := by sorry

end NUMINAMATH_CALUDE_jewelry_price_increase_is_10_l3498_349843


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l3498_349827

/-- Given a cafeteria with apples, prove the number of apples handed out to students. -/
theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (apples_per_pie : ℕ) 
  (pies_made : ℕ) 
  (h1 : initial_apples = 51)
  (h2 : apples_per_pie = 5)
  (h3 : pies_made = 2) :
  initial_apples - (apples_per_pie * pies_made) = 41 :=
by sorry

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l3498_349827


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_a_for_nonempty_solution_l3498_349863

-- Define the function f(x)
def f (x : ℝ) : ℝ := |4*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) < 8
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -9/5 < x ∧ x < 11/3} :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x + 5*|x + 2| < a^2 - 8*a) ↔ (a < -1 ∨ a > 9) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_a_for_nonempty_solution_l3498_349863


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l3498_349872

/-- Calculates the perimeter of a figure composed of a central square and four smaller squares attached to its sides. -/
def figure_perimeter (central_side_length : ℝ) (small_side_length : ℝ) : ℝ :=
  4 * central_side_length + 4 * (3 * small_side_length)

/-- Theorem stating that the perimeter of the specific figure is 140 -/
theorem specific_figure_perimeter :
  figure_perimeter 20 5 = 140 := by
  sorry

#eval figure_perimeter 20 5

end NUMINAMATH_CALUDE_specific_figure_perimeter_l3498_349872


namespace NUMINAMATH_CALUDE_expression_simplification_l3498_349883

theorem expression_simplification (m : ℝ) (h : m^2 + 3*m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6*m) / (m + 2 - 5 / (m - 2)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3498_349883


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_hyperbola_vertices_distance_proof_l3498_349847

/-- The distance between the vertices of a hyperbola with equation x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertices_distance : ℝ :=
  let hyperbola_equation (x y : ℝ) := x^2/16 - y^2/9 = 1
  let vertices_distance := 8
  vertices_distance

/-- Proof of the theorem -/
theorem hyperbola_vertices_distance_proof : hyperbola_vertices_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_hyperbola_vertices_distance_proof_l3498_349847


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3498_349875

/-- Given three points in 2D space -/
def point1 : ℝ × ℝ := (4, -3)
def point2 (b : ℝ) : ℝ × ℝ := (2*b + 1, 5)
def point3 (b : ℝ) : ℝ × ℝ := (-b + 3, 1)

/-- Function to check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

/-- Theorem stating that b = 1/4 when the given points are collinear -/
theorem collinear_points_b_value :
  collinear point1 (point2 b) (point3 b) → b = 1/4 := by sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3498_349875


namespace NUMINAMATH_CALUDE_perimeter_ABCDEHG_l3498_349851

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the conditions
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := sorry
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry
def distance (X Y : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem perimeter_ABCDEHG :
  is_equilateral A B C →
  is_equilateral A D E →
  is_equilateral E F G →
  is_midpoint D A C →
  is_midpoint H A E →
  distance A B = 6 →
  distance A B + distance B C + distance C D + distance D E +
  distance E F + distance F G + distance G H + distance H A = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDEHG_l3498_349851


namespace NUMINAMATH_CALUDE_side_a_is_one_max_perimeter_is_three_max_perimeter_when_b_equals_c_l3498_349833

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b * Real.cos t.A + t.a * Real.cos t.B = t.a * t.c

-- Theorem for part 1
theorem side_a_is_one (t : Triangle) (h : satisfiesCondition t) : t.a = 1 := by
  sorry

-- Theorem for part 2
theorem max_perimeter_is_three (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.A = Real.pi / 3) :
  t.a + t.b + t.c ≤ 3 := by
  sorry

-- Theorem for the maximum perimeter occurring when b = c
theorem max_perimeter_when_b_equals_c (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.A = Real.pi / 3) :
  ∃ (t' : Triangle), satisfiesCondition t' ∧ t'.A = Real.pi / 3 ∧ t'.b = t'.c ∧ t'.a + t'.b + t'.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_side_a_is_one_max_perimeter_is_three_max_perimeter_when_b_equals_c_l3498_349833


namespace NUMINAMATH_CALUDE_cost_price_of_cloth_l3498_349860

/-- Represents the cost price of one metre of cloth -/
def costPricePerMetre (totalMetres : ℕ) (sellingPrice : ℕ) (profitPerMetre : ℕ) : ℕ :=
  (sellingPrice - profitPerMetre * totalMetres) / totalMetres

/-- Theorem stating that the cost price of one metre of cloth is 85 rupees -/
theorem cost_price_of_cloth :
  costPricePerMetre 85 8925 20 = 85 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_cloth_l3498_349860


namespace NUMINAMATH_CALUDE_weekly_earnings_proof_l3498_349837

/-- Calculates the total earnings for a repair shop given the number of repairs and their costs. -/
def total_earnings (phone_repairs laptop_repairs computer_repairs : ℕ) 
  (phone_cost laptop_cost computer_cost : ℕ) : ℕ :=
  phone_repairs * phone_cost + laptop_repairs * laptop_cost + computer_repairs * computer_cost

/-- Theorem: The total earnings for the week is $121 given the specified repairs and costs. -/
theorem weekly_earnings_proof :
  total_earnings 5 2 2 11 15 18 = 121 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_proof_l3498_349837


namespace NUMINAMATH_CALUDE_event_probability_l3498_349834

theorem event_probability (P_B P_AB P_AorB : ℝ) 
  (hB : P_B = 0.4)
  (hAB : P_AB = 0.25)
  (hAorB : P_AorB = 0.6) :
  ∃ P_A : ℝ, P_A = 0.45 ∧ P_AorB = P_A + P_B - P_AB :=
sorry

end NUMINAMATH_CALUDE_event_probability_l3498_349834


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l3498_349894

-- Define the coordinates of the rectangle
def vertex1 : ℤ × ℤ := (1, 2)
def vertex2 : ℤ × ℤ := (1, 6)
def vertex3 : ℤ × ℤ := (7, 6)
def vertex4 : ℤ × ℤ := (7, 2)

-- Define the function to calculate the perimeter and area sum
def perimeterAreaSum (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let width := (v3.1 - v1.1).natAbs
  let height := (v2.2 - v1.2).natAbs
  2 * (width + height) + width * height

-- Theorem statement
theorem rectangle_perimeter_area_sum :
  perimeterAreaSum vertex1 vertex2 vertex3 vertex4 = 44 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l3498_349894


namespace NUMINAMATH_CALUDE_problem_statement_l3498_349846

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  (y / x = 1) ∧
  (∃ (A B C : ℝ), Real.tan C = y / x ∧ 
    (∀ A' B' C' : ℝ, Real.tan C' = y / x → 
      Real.sin (2*A') + 2 * Real.cos B' ≤ Real.sin (2*A) + 2 * Real.cos B) ∧
    Real.sin (2*A) + 2 * Real.cos B = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3498_349846


namespace NUMINAMATH_CALUDE_expression_evaluation_l3498_349870

theorem expression_evaluation : (4 + 6 + 7) * 2 - 2 + 3 / 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3498_349870


namespace NUMINAMATH_CALUDE_jerry_won_47_tickets_l3498_349888

/-- The number of tickets Jerry won later at the arcade -/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Theorem: Jerry won 47 tickets later at the arcade -/
theorem jerry_won_47_tickets :
  tickets_won_later 4 2 49 = 47 := by
  sorry

end NUMINAMATH_CALUDE_jerry_won_47_tickets_l3498_349888


namespace NUMINAMATH_CALUDE_max_b_value_l3498_349886

theorem max_b_value (a b c : ℕ) : 
  1 < c → c < b → b < a → a * b * c = 360 → b ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l3498_349886


namespace NUMINAMATH_CALUDE_krakozyabr_population_is_32_l3498_349824

structure Krakozyabr where
  hasHorns : Bool
  hasWings : Bool

def totalKrakozyabrs (population : List Krakozyabr) : Nat :=
  population.length

theorem krakozyabr_population_is_32 
  (population : List Krakozyabr) 
  (all_have_horns_or_wings : ∀ k ∈ population, k.hasHorns ∨ k.hasWings)
  (horns_with_wings_ratio : (population.filter (λ k => k.hasHorns ∧ k.hasWings)).length = 
    (population.filter (λ k => k.hasHorns)).length / 5)
  (wings_with_horns_ratio : (population.filter (λ k => k.hasHorns ∧ k.hasWings)).length = 
    (population.filter (λ k => k.hasWings)).length / 4)
  (population_range : 25 < totalKrakozyabrs population ∧ totalKrakozyabrs population < 35) :
  totalKrakozyabrs population = 32 := by
  sorry

end NUMINAMATH_CALUDE_krakozyabr_population_is_32_l3498_349824


namespace NUMINAMATH_CALUDE_solution_in_second_quadrant_l3498_349893

theorem solution_in_second_quadrant :
  ∃ (x y : ℝ), 
    (y = 2*x + 2) ∧ 
    (y = -x + 1) ∧ 
    (x < 0) ∧ 
    (y > 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_second_quadrant_l3498_349893


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3498_349830

theorem min_value_of_expression (a b c : ℝ) 
  (h : ∀ (x y : ℝ), 3*x + 4*y - 5 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ 3*x + 4*y + 5) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (v : ℝ), (v = a + b - c → v ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3498_349830


namespace NUMINAMATH_CALUDE_initial_crayons_l3498_349814

theorem initial_crayons (initial : ℕ) : initial + 3 = 12 → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_l3498_349814


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l3498_349848

theorem cos_five_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (5 * π / 6 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l3498_349848


namespace NUMINAMATH_CALUDE_dislike_both_tv_and_sports_l3498_349852

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 40 / 100
def sports_dislike_percentage : ℚ := 15 / 100

theorem dislike_both_tv_and_sports :
  ∃ (n : ℕ), n = (total_surveyed : ℚ) * tv_dislike_percentage * sports_dislike_percentage ∧ n = 90 :=
by sorry

end NUMINAMATH_CALUDE_dislike_both_tv_and_sports_l3498_349852


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3498_349882

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4 * Real.sqrt 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 2 * Real.sqrt 2) ∧ (y = 2 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3498_349882


namespace NUMINAMATH_CALUDE_smallest_stairs_l3498_349881

theorem smallest_stairs (n : ℕ) : 
  (n > 20 ∧ n % 6 = 4 ∧ n % 7 = 3) → n ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_l3498_349881


namespace NUMINAMATH_CALUDE_tom_initial_investment_l3498_349810

/-- Represents the business partnership between Tom and Jose -/
structure Partnership where
  tom_investment : ℕ
  jose_investment : ℕ
  tom_join_time : ℕ
  jose_join_time : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the partnership details -/
def calculate_tom_investment (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that Tom's initial investment is 3000 given the problem conditions -/
theorem tom_initial_investment :
  let p : Partnership := {
    tom_investment := 0,  -- We don't know this value yet
    jose_investment := 45000,
    tom_join_time := 0,  -- Tom joined at the start
    jose_join_time := 2,
    total_profit := 54000,
    jose_profit := 30000
  }
  calculate_tom_investment p = 3000 := by
  sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l3498_349810


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3498_349885

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3498_349885


namespace NUMINAMATH_CALUDE_sqrt_3_powers_l3498_349876

theorem sqrt_3_powers (n : ℕ) (h : n ≥ 1) :
  ∃ (k w : ℕ), 
    (((2 : ℝ) + Real.sqrt 3) ^ (2 * n) + ((2 : ℝ) - Real.sqrt 3) ^ (2 * n) = (2 * k : ℝ)) ∧
    (((2 : ℝ) + Real.sqrt 3) ^ (2 * n) - ((2 : ℝ) - Real.sqrt 3) ^ (2 * n) = (w : ℝ) * Real.sqrt 3) ∧
    w > 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_3_powers_l3498_349876


namespace NUMINAMATH_CALUDE_addition_subtraction_integers_l3498_349899

theorem addition_subtraction_integers : (1 + (-2) - 8 - (-9) : ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_integers_l3498_349899


namespace NUMINAMATH_CALUDE_store_profit_percentage_l3498_349844

/-- Given a store that purchases an item and the conditions of a potential price change,
    this theorem proves that the original profit percentage was 35%. -/
theorem store_profit_percentage
  (original_cost : ℝ)
  (cost_decrease_percentage : ℝ)
  (profit_increase_percentage : ℝ)
  (h1 : original_cost = 200)
  (h2 : cost_decrease_percentage = 10)
  (h3 : profit_increase_percentage = 15)
  (h4 : ∃ (sale_price : ℝ) (original_profit_percentage : ℝ),
        sale_price = original_cost * (1 + original_profit_percentage / 100) ∧
        sale_price = (original_cost * (1 - cost_decrease_percentage / 100)) *
                     (1 + (original_profit_percentage + profit_increase_percentage) / 100)) :
  ∃ (original_profit_percentage : ℝ), original_profit_percentage = 35 :=
sorry

end NUMINAMATH_CALUDE_store_profit_percentage_l3498_349844


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l3498_349832

theorem book_arrangement_count : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 3
  let spanish_books : ℕ := 5
  let arabic_group : ℕ := 1  -- Treat Arabic books as one unit
  let spanish_group : ℕ := 1  -- Treat Spanish books as one unit
  let german_group : ℕ := 1  -- Treat German books as one ordered unit
  let total_groups : ℕ := arabic_group + spanish_group + german_group
  let group_arrangements : ℕ := Nat.factorial total_groups
  let arabic_arrangements : ℕ := Nat.factorial arabic_books
  let spanish_arrangements : ℕ := Nat.factorial spanish_books

  group_arrangements * arabic_arrangements * spanish_arrangements

-- Prove that book_arrangement_count equals 4320
theorem book_arrangement_theorem : book_arrangement_count = 4320 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l3498_349832


namespace NUMINAMATH_CALUDE_repeating_decimal_4_8_equals_44_9_l3498_349898

/-- Represents a repeating decimal where the digit 8 repeats infinitely after the decimal point -/
def repeating_decimal_4_8 : ℚ := 4 + 8/9

theorem repeating_decimal_4_8_equals_44_9 : 
  repeating_decimal_4_8 = 44/9 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_4_8_equals_44_9_l3498_349898


namespace NUMINAMATH_CALUDE_positive_integer_sum_with_square_is_thirty_l3498_349857

theorem positive_integer_sum_with_square_is_thirty (P : ℕ+) : P^2 + P = 30 → P = 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_sum_with_square_is_thirty_l3498_349857


namespace NUMINAMATH_CALUDE_triangle_side_sum_l3498_349879

/-- Represents a triangle with side lengths a, b, and c, and angles A, B, and C. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if the angles of a triangle are in arithmetic progression -/
def anglesInArithmeticProgression (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.A + 2*d

/-- Represents a value that can be expressed as p + √q + √r where p, q, r are integers -/
structure SpecialValue where
  p : ℤ
  q : ℤ
  r : ℤ

/-- The theorem to be proved -/
theorem triangle_side_sum (t : Triangle) (x₁ x₂ : SpecialValue) :
  t.a = 6 ∧ t.b = 8 ∧
  anglesInArithmeticProgression t ∧
  t.A = 30 * π / 180 ∧
  (t.c = Real.sqrt (x₁.q : ℝ) ∨ t.c = (x₂.p : ℝ) + Real.sqrt (x₂.q : ℝ)) →
  (x₁.p : ℝ) + Real.sqrt (x₁.q : ℝ) + Real.sqrt (x₁.r : ℝ) +
  (x₂.p : ℝ) + Real.sqrt (x₂.q : ℝ) + Real.sqrt (x₂.r : ℝ) =
  7 + Real.sqrt 36 + Real.sqrt 83 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l3498_349879


namespace NUMINAMATH_CALUDE_kafelnikov_served_first_l3498_349821

/-- Represents a tennis player -/
inductive Player : Type
| Kafelnikov : Player
| Becker : Player

/-- Represents the result of a tennis match -/
structure MatchResult :=
  (winner : Player)
  (winner_games : Nat)
  (loser_games : Nat)

/-- Represents the serving pattern in a tennis match -/
structure ServingPattern :=
  (server_wins : Nat)
  (receiver_wins : Nat)

/-- Determines who served first in a tennis match -/
def first_server (result : MatchResult) (serving : ServingPattern) : Player :=
  sorry

/-- Theorem stating that Kafelnikov served first -/
theorem kafelnikov_served_first 
  (result : MatchResult) 
  (serving : ServingPattern) :
  result.winner = Player.Kafelnikov ∧
  result.winner_games = 6 ∧
  result.loser_games = 3 ∧
  serving.server_wins = 5 ∧
  serving.receiver_wins = 4 →
  first_server result serving = Player.Kafelnikov :=
by sorry

end NUMINAMATH_CALUDE_kafelnikov_served_first_l3498_349821


namespace NUMINAMATH_CALUDE_division_remainder_l3498_349878

theorem division_remainder : Int.mod 1234567 256 = 503 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3498_349878


namespace NUMINAMATH_CALUDE_factor_calculation_l3498_349841

theorem factor_calculation (x : ℝ) (factor : ℝ) : 
  x = 4 → (2 * x + 9) * factor = 51 → factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l3498_349841


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l3498_349855

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    calculate the cost of 12 pens -/
theorem pen_cost_calculation (total_cost : ℚ) (pen_cost : ℚ) (pencil_cost : ℚ) : 
  total_cost = 150 →
  3 * pen_cost + 5 * pencil_cost = total_cost →
  pen_cost = 5 * pencil_cost →
  12 * pen_cost = 450 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_calculation_l3498_349855


namespace NUMINAMATH_CALUDE_trajectory_equation_l3498_349831

/-- Given points A(-1,1) and B(1,-1) symmetrical about the origin,
    prove that a point P(x,y) with x ≠ ±1 satisfies x^2 + 3y^2 = 4
    if the product of slopes of AP and BP is -1/3 -/
theorem trajectory_equation (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = -1/3 →
  x^2 + 3*y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3498_349831


namespace NUMINAMATH_CALUDE_mathildas_debt_l3498_349822

/-- Mathilda's debt problem -/
theorem mathildas_debt (initial_payment : ℝ) (remaining_percentage : ℝ) (original_debt : ℝ) : 
  initial_payment = 125 ∧ 
  remaining_percentage = 75 ∧ 
  initial_payment = (100 - remaining_percentage) / 100 * original_debt →
  original_debt = 500 := by
  sorry

end NUMINAMATH_CALUDE_mathildas_debt_l3498_349822


namespace NUMINAMATH_CALUDE_bowl_game_points_ratio_l3498_349815

theorem bowl_game_points_ratio :
  ∀ (noa_points phillip_points : ℕ) (multiple : ℚ),
    noa_points = 30 →
    phillip_points = noa_points * multiple →
    noa_points + phillip_points = 90 →
    phillip_points / noa_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowl_game_points_ratio_l3498_349815


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3498_349880

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- The foci coordinates -/
def foci : Set (ℝ × ℝ) := {(0, -Real.sqrt 29), (0, Real.sqrt 29)}

/-- Theorem: The foci of the given hyperbola are located at (0, ±√29) -/
theorem hyperbola_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci ↔ 
    (∃ (x y : ℝ), hyperbola_equation x y ∧ 
      f = (x, y) ∧ 
      (∀ (x' y' : ℝ), hyperbola_equation x' y' → 
        (x - x')^2 + (y - y')^2 = ((Real.sqrt 29) + (Real.sqrt 29))^2 ∨
        (x - x')^2 + (y - y')^2 = ((Real.sqrt 29) - (Real.sqrt 29))^2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3498_349880


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l3498_349838

theorem average_of_three_numbers (M : ℝ) (h1 : 12 < M) (h2 : M < 25) : 
  ∃ k : ℝ, k = 5 ∧ (8 + 15 + (M + k)) / 3 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l3498_349838


namespace NUMINAMATH_CALUDE_goldbach_140_max_diff_l3498_349804

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_140_max_diff :
  ∀ p q : ℕ,
    is_prime p →
    is_prime q →
    p + q = 140 →
    p < q →
    p < 50 →
    q - p ≤ 134 :=
by sorry

end NUMINAMATH_CALUDE_goldbach_140_max_diff_l3498_349804


namespace NUMINAMATH_CALUDE_trapezoid_semicircle_area_l3498_349871

-- Define the trapezoid
def trapezoid : Set (ℝ × ℝ) :=
  {p | p = (5, 11) ∨ p = (16, 11) ∨ p = (16, -2) ∨ p = (5, -2)}

-- Define the semicircle
def semicircle : Set (ℝ × ℝ) :=
  {p | (p.1 - 10.5)^2 + (p.2 + 2)^2 ≤ 5.5^2 ∧ p.2 ≤ -2}

-- Define the area to be calculated
def bounded_area : ℝ := sorry

-- Theorem statement
theorem trapezoid_semicircle_area :
  bounded_area = 15.125 * Real.pi := by sorry

end NUMINAMATH_CALUDE_trapezoid_semicircle_area_l3498_349871


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l3498_349865

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_sequence_solution :
  ∀ x y z : ℝ,
  is_arithmetic_sequence x y z →
  x + y + z = -3 →
  is_geometric_sequence (x + y) (y + z) (z + x) →
  ((x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = -7 ∧ y = -1 ∧ z = 5)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l3498_349865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3498_349818

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_20th : a 20 = 15)
  (h_21st : a 21 = 18) :
  a 3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3498_349818


namespace NUMINAMATH_CALUDE_quadratic_point_value_l3498_349853

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x ≤ 4)
  (h2 : quadratic_function a b c 2 = 4)
  (h3 : quadratic_function a b c 0 = -7) :
  quadratic_function a b c 5 = -83/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_value_l3498_349853


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_l3498_349813

theorem opposite_reciprocal_abs (x : ℚ) (h : x = -1.5) : 
  (-x = 1.5) ∧ (1 / x = -2/3) ∧ (abs x = 1.5) := by
  sorry

#check opposite_reciprocal_abs

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_l3498_349813


namespace NUMINAMATH_CALUDE_penetrated_cubes_count_stating_penetrated_cubes_calculation_correct_l3498_349862

/-- 
Given a rectangular solid with dimensions 120 × 260 × 300,
the number of unit cubes penetrated by an internal diagonal is 520.
-/
theorem penetrated_cubes_count : ℕ → ℕ → ℕ → ℕ
  | 120, 260, 300 => 520
  | _, _, _ => 0

/-- Function to calculate the number of penetrated cubes -/
def calculate_penetrated_cubes (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- 
Theorem stating that the calculate_penetrated_cubes function 
correctly calculates the number of penetrated cubes for the given dimensions
-/
theorem penetrated_cubes_calculation_correct :
  calculate_penetrated_cubes 120 260 300 = penetrated_cubes_count 120 260 300 := by
  sorry

#eval calculate_penetrated_cubes 120 260 300

end NUMINAMATH_CALUDE_penetrated_cubes_count_stating_penetrated_cubes_calculation_correct_l3498_349862


namespace NUMINAMATH_CALUDE_B_is_closed_l3498_349819

def ClosedSet (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def B : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem B_is_closed : ClosedSet B := by
  sorry

end NUMINAMATH_CALUDE_B_is_closed_l3498_349819


namespace NUMINAMATH_CALUDE_function_properties_l3498_349840

-- Define the function f(x) = ax³ + bx
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_properties (a b : ℝ) :
  -- Condition 1: Tangent line at x=3 is parallel to 24x - y + 1 = 0
  f' a b 3 = 24 →
  -- Condition 2: Function has an extremum at x=1
  f' a b 1 = 0 →
  -- Condition 3: a = 1
  a = 1 →
  -- Conclusion 1: f(x) = x³ - 3x
  (∀ x, f a b x = x^3 - 3*x) ∧
  -- Conclusion 2: Interval of monotonic decrease is [-1, 1]
  (∀ x, x ∈ Set.Icc (-1) 1 → f' a b x ≤ 0) ∧
  -- Conclusion 3: For f(x) to be decreasing on [-1, 1], b ≤ -3
  (∀ x, x ∈ Set.Icc (-1) 1 → f' 1 b x ≤ 0) → b ≤ -3 :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l3498_349840
