import Mathlib

namespace NUMINAMATH_CALUDE_exponent_problem_l2902_290200

theorem exponent_problem (a : ℝ) (m n : ℤ) (h1 : a^m = 5) (h2 : a^n = 2) :
  a^(m-2*n) = 5/4 := by sorry

end NUMINAMATH_CALUDE_exponent_problem_l2902_290200


namespace NUMINAMATH_CALUDE_delphine_chocolates_day1_l2902_290210

/-- Represents the number of chocolates Delphine ate on each day -/
structure ChocolatesEaten where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Theorem stating the number of chocolates Delphine ate on the first day -/
theorem delphine_chocolates_day1 (c : ChocolatesEaten) : c.day1 = 4 :=
  by
  have h1 : c.day2 = 2 * c.day1 - 3 := sorry
  have h2 : c.day3 = c.day1 - 2 := sorry
  have h3 : c.day4 = c.day3 - 1 := sorry
  have h4 : c.day1 + c.day2 + c.day3 + c.day4 + 12 = 24 := sorry
  sorry

#check delphine_chocolates_day1

end NUMINAMATH_CALUDE_delphine_chocolates_day1_l2902_290210


namespace NUMINAMATH_CALUDE_power_of_two_condition_l2902_290226

theorem power_of_two_condition (a n : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : a ≥ n) :
  (∃ m : ℕ, (a + 1)^n + a - 1 = 2^m) ↔ 
  ((a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1)) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_condition_l2902_290226


namespace NUMINAMATH_CALUDE_inverse_contrapositive_equivalence_l2902_290218

theorem inverse_contrapositive_equivalence (a b c : ℝ) :
  (¬(a > b) → ¬(a + c > b + c)) ↔ (a + c ≤ b + c → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_equivalence_l2902_290218


namespace NUMINAMATH_CALUDE_park_outer_diameter_l2902_290234

/-- Given a park layout with a central statue, lawn, and jogging path, 
    this theorem proves the diameter of the outer boundary. -/
theorem park_outer_diameter 
  (statue_diameter : ℝ) 
  (lawn_width : ℝ) 
  (jogging_path_width : ℝ) 
  (h1 : statue_diameter = 8) 
  (h2 : lawn_width = 10) 
  (h3 : jogging_path_width = 5) : 
  statue_diameter / 2 + lawn_width + jogging_path_width = 19 ∧ 
  2 * (statue_diameter / 2 + lawn_width + jogging_path_width) = 38 :=
sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l2902_290234


namespace NUMINAMATH_CALUDE_special_integers_property_l2902_290254

/-- A function that reverses the hundreds and units digits of a three-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  units * 100 + tens * 10 + hundreds

/-- The theorem stating the property of the 90 special integers -/
theorem special_integers_property :
  ∃ (S : Finset ℕ), 
    Finset.card S = 90 ∧ 
    (∀ n ∈ S, 100 < n ∧ n < 1100) ∧
    (∀ n ∈ S, reverseDigits n = n + 99) := by
  sorry

#check special_integers_property

end NUMINAMATH_CALUDE_special_integers_property_l2902_290254


namespace NUMINAMATH_CALUDE_arc_length_parametric_curve_l2902_290225

/-- The arc length of the curve given by the parametric equations
    x = 4(t - sin t) and y = 4(1 - cos t), where π/2 ≤ t ≤ 2π -/
theorem arc_length_parametric_curve :
  let x : ℝ → ℝ := λ t ↦ 4 * (t - Real.sin t)
  let y : ℝ → ℝ := λ t ↦ 4 * (1 - Real.cos t)
  let arc_length : ℝ := ∫ t in Set.Icc (π / 2) (2 * π), Real.sqrt ((4 * (1 - Real.cos t))^2 + (4 * Real.sin t)^2)
  arc_length = 8 * (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_arc_length_parametric_curve_l2902_290225


namespace NUMINAMATH_CALUDE_rachel_adam_weight_difference_l2902_290215

/-- Given the weights of three people Rachel, Jimmy, and Adam, prove that Rachel weighs 15 pounds more than Adam. -/
theorem rachel_adam_weight_difference (R J A : ℝ) : 
  R = 75 →  -- Rachel weighs 75 pounds
  R = J - 6 →  -- Rachel weighs 6 pounds less than Jimmy
  R > A →  -- Rachel weighs more than Adam
  (R + J + A) / 3 = 72 →  -- The average weight of the three people is 72 pounds
  R - A = 15 :=  -- Rachel weighs 15 pounds more than Adam
by sorry

end NUMINAMATH_CALUDE_rachel_adam_weight_difference_l2902_290215


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l2902_290284

/-- Proves the minimum number of workers needed for profit --/
theorem min_workers_for_profit :
  let daily_maintenance : ℝ := 600
  let hourly_wage : ℝ := 20
  let widgets_per_hour : ℝ := 6
  let price_per_widget : ℝ := 3.50
  let work_hours : ℝ := 9

  let cost (n : ℝ) := daily_maintenance + hourly_wage * work_hours * n
  let revenue (n : ℝ) := price_per_widget * widgets_per_hour * work_hours * n

  ∀ n : ℕ, (n ≥ 67 ↔ revenue n > cost n) :=
by
  sorry

#check min_workers_for_profit

end NUMINAMATH_CALUDE_min_workers_for_profit_l2902_290284


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficients_sum_l2902_290290

/-- Given a quadratic inequality x² - ax + b < 0 with solution set {x | 1 < x < 2},
    prove that a + b = 5 -/
theorem quadratic_inequality_coefficients_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficients_sum_l2902_290290


namespace NUMINAMATH_CALUDE_parallel_lines_x_value_l2902_290236

/-- A line passing through two points -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- Check if a line is vertical -/
def Line.isVertical (l : Line) : Prop :=
  l.x₁ = l.x₂

/-- Two lines are parallel if they are both vertical or have the same slope -/
def parallelLines (l₁ l₂ : Line) : Prop :=
  (l₁.isVertical ∧ l₂.isVertical) ∨
  (¬l₁.isVertical ∧ ¬l₂.isVertical ∧ 
   (l₁.y₂ - l₁.y₁) / (l₁.x₂ - l₁.x₁) = (l₂.y₂ - l₂.y₁) / (l₂.x₂ - l₂.x₁))

theorem parallel_lines_x_value :
  ∀ (x : ℝ),
  let l₁ : Line := ⟨-1, -2, -1, 4⟩
  let l₂ : Line := ⟨2, 1, x, 6⟩
  parallelLines l₁ l₂ → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_x_value_l2902_290236


namespace NUMINAMATH_CALUDE_marble_probability_l2902_290272

theorem marble_probability (total_marbles : ℕ) (p_white p_green : ℚ) :
  total_marbles = 90 →
  p_white = 1/3 →
  p_green = 1/5 →
  (1 : ℚ) - (p_white + p_green) = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2902_290272


namespace NUMINAMATH_CALUDE_eight_power_zero_minus_log_hundred_l2902_290222

theorem eight_power_zero_minus_log_hundred : 8^0 - Real.log 100 / Real.log 10 = -1 := by sorry

end NUMINAMATH_CALUDE_eight_power_zero_minus_log_hundred_l2902_290222


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2902_290201

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 4 * y^2 - 5 * y - 21 = (4 * y + a) * (y + b)) →
  a - b = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2902_290201


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l2902_290276

def rotation_angle (α : Real) : Prop :=
  0 < α ∧ α < Real.pi / 2 ∧ Real.cos α = 4 / 5

def overlapping_area (α : Real) : Real :=
  -- Definition of the overlapping area function
  sorry

theorem overlapping_squares_area (α : Real) 
  (h : rotation_angle α) : overlapping_area α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l2902_290276


namespace NUMINAMATH_CALUDE_a_leq_0_necessary_not_sufficient_l2902_290265

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x^2 + a * x - 3/2
  else 2 * a * x^2 + x

-- Define what it means for a function to be monotonically decreasing
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

-- Theorem statement
theorem a_leq_0_necessary_not_sufficient :
  (∃ a : ℝ, a ≤ 0 ∧ ¬(MonotonicallyDecreasing (f a))) ∧
  (∀ a : ℝ, MonotonicallyDecreasing (f a) → a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_a_leq_0_necessary_not_sufficient_l2902_290265


namespace NUMINAMATH_CALUDE_consecutive_color_probability_value_l2902_290248

/-- Represents the number of green chips in the bag -/
def green_chips : ℕ := 4

/-- Represents the number of orange chips in the bag -/
def orange_chips : ℕ := 3

/-- Represents the number of blue chips in the bag -/
def blue_chips : ℕ := 5

/-- Represents the total number of chips in the bag -/
def total_chips : ℕ := green_chips + orange_chips + blue_chips

/-- The probability of drawing all chips such that each color group is drawn consecutively -/
def consecutive_color_probability : ℚ :=
  (Nat.factorial 3 * Nat.factorial green_chips * Nat.factorial orange_chips * Nat.factorial blue_chips) /
  Nat.factorial total_chips

theorem consecutive_color_probability_value :
  consecutive_color_probability = 1 / 4620 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_probability_value_l2902_290248


namespace NUMINAMATH_CALUDE_water_consumption_proof_l2902_290224

/-- Proves that drinking 500 milliliters every 2 hours for 12 hours results in 3 liters of water consumption. -/
theorem water_consumption_proof (liters_goal : ℝ) (ml_per_interval : ℝ) (hours_per_interval : ℝ) :
  liters_goal = 3 ∧ ml_per_interval = 500 ∧ hours_per_interval = 2 →
  (liters_goal * 1000) / ml_per_interval * hours_per_interval = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_proof_l2902_290224


namespace NUMINAMATH_CALUDE_range_of_m_l2902_290206

/-- Represents an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / m + y^2 / (6 - m) = 1 ∧ m > 6 - m ∧ m > 0

/-- Represents a hyperbola with given eccentricity range -/
def is_hyperbola_with_eccentricity (m : ℝ) : Prop :=
  ∃ x y e : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 
    e^2 = 1 + m / 5 ∧ 
    Real.sqrt 6 / 2 < e ∧ e < Real.sqrt 2 ∧
    m > 0

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) : 
  (is_ellipse_x_axis m ∨ is_hyperbola_with_eccentricity m) ∧ 
  ¬(is_ellipse_x_axis m ∧ is_hyperbola_with_eccentricity m) →
  (5/2 < m ∧ m ≤ 3) ∨ (5 ≤ m ∧ m < 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2902_290206


namespace NUMINAMATH_CALUDE_right_triangle_set_l2902_290221

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The theorem stating that only one set of numbers forms a right triangle --/
theorem right_triangle_set :
  ¬(is_right_triangle 0.1 0.2 0.3) ∧
  ¬(is_right_triangle 1 1 2) ∧
  is_right_triangle 10 24 26 ∧
  ¬(is_right_triangle 9 16 25) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2902_290221


namespace NUMINAMATH_CALUDE_no_2013_numbers_exist_l2902_290258

theorem no_2013_numbers_exist : ¬ ∃ (S : Finset ℕ), 
  (S.card = 2013) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y) ∧
  (∀ a ∈ S, (S.sum id - a) ≥ a^2) :=
by sorry

end NUMINAMATH_CALUDE_no_2013_numbers_exist_l2902_290258


namespace NUMINAMATH_CALUDE_cube_cut_volume_ratio_l2902_290245

theorem cube_cut_volume_ratio (x y : ℝ) (h_positive : x > 0 ∧ y > 0) 
  (h_cut : y < x) (h_surface_ratio : 2 * (x^2 + 2*x*y) = x^2 + 2*x*(x-y)) : 
  (x^2 * y) / (x^2 * (x - y)) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_cube_cut_volume_ratio_l2902_290245


namespace NUMINAMATH_CALUDE_pattern_equality_l2902_290241

theorem pattern_equality (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l2902_290241


namespace NUMINAMATH_CALUDE_function_properties_l2902_290283

/-- Given real numbers b and c, and a function f(x) = x^2 + bx + c that satisfies
    f(sin α) ≥ 0 and f(2 + cos β) ≤ 0 for any α, β ∈ ℝ, prove that f(1) = 0 and c ≥ 3 -/
theorem function_properties (b c : ℝ) 
    (f : ℝ → ℝ) 
    (f_def : ∀ x, f x = x^2 + b*x + c)
    (f_sin_nonneg : ∀ α, f (Real.sin α) ≥ 0)
    (f_cos_nonpos : ∀ β, f (2 + Real.cos β) ≤ 0) : 
  f 1 = 0 ∧ c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2902_290283


namespace NUMINAMATH_CALUDE_garland_arrangement_count_l2902_290212

def blue_bulbs : ℕ := 5
def red_bulbs : ℕ := 6
def white_bulbs : ℕ := 7

def total_non_white_bulbs : ℕ := blue_bulbs + red_bulbs
def total_spaces : ℕ := total_non_white_bulbs + 1

theorem garland_arrangement_count :
  (Nat.choose total_non_white_bulbs blue_bulbs) * (Nat.choose total_spaces white_bulbs) = 365904 :=
sorry

end NUMINAMATH_CALUDE_garland_arrangement_count_l2902_290212


namespace NUMINAMATH_CALUDE_triangle_inequality_l2902_290268

theorem triangle_inequality (a b c p q r : ℝ) 
  (triangle_cond : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sum_zero : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2902_290268


namespace NUMINAMATH_CALUDE_cordelia_hair_coloring_time_l2902_290213

/-- Represents the hair coloring process -/
structure HairColoring where
  bleaching_time : ℝ
  dyeing_time : ℝ

/-- The total time for the hair coloring process -/
def total_time (hc : HairColoring) : ℝ :=
  hc.bleaching_time + hc.dyeing_time

/-- Theorem stating the total time for Cordelia's hair coloring process -/
theorem cordelia_hair_coloring_time :
  ∃ (hc : HairColoring),
    hc.bleaching_time = 3 ∧
    hc.dyeing_time = 2 * hc.bleaching_time ∧
    total_time hc = 9 := by
  sorry

end NUMINAMATH_CALUDE_cordelia_hair_coloring_time_l2902_290213


namespace NUMINAMATH_CALUDE_cube_monotone_l2902_290230

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l2902_290230


namespace NUMINAMATH_CALUDE_tank_capacity_is_24_gallons_l2902_290231

/-- Represents the capacity of a tank and its contents over time -/
structure TankState where
  capacity : ℝ
  initialMixture : ℝ
  initialSodiumChloride : ℝ
  initialWater : ℝ
  evaporationRate : ℝ
  time : ℝ

/-- Calculates the final water volume after evaporation -/
def finalWaterVolume (state : TankState) : ℝ :=
  state.initialWater - state.evaporationRate * state.time

/-- Theorem stating the tank capacity given the conditions -/
theorem tank_capacity_is_24_gallons :
  ∀ (state : TankState),
    state.initialMixture = state.capacity / 4 →
    state.initialSodiumChloride = 0.3 * state.initialMixture →
    state.initialWater = 0.7 * state.initialMixture →
    state.evaporationRate = 0.4 →
    state.time = 6 →
    finalWaterVolume state = state.initialSodiumChloride →
    state.capacity = 24 := by
  sorry

#check tank_capacity_is_24_gallons

end NUMINAMATH_CALUDE_tank_capacity_is_24_gallons_l2902_290231


namespace NUMINAMATH_CALUDE_car_speed_problem_l2902_290211

theorem car_speed_problem (D : ℝ) (V : ℝ) : 
  D > 0 →
  (D / ((D/3)/60 + (D/3)/24 + (D/3)/V)) = 37.89473684210527 →
  V = 48 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2902_290211


namespace NUMINAMATH_CALUDE_chip_cost_is_fifty_cents_l2902_290263

/-- The cost of a bag of chips given the conditions in the problem -/
def chip_cost : ℚ :=
  let candy_cost : ℚ := 2
  let student_count : ℕ := 5
  let total_cost : ℚ := 15
  let candy_per_student : ℕ := 1
  let chips_per_student : ℕ := 2
  (total_cost - student_count * candy_cost) / (student_count * chips_per_student)

/-- Theorem stating that the cost of a bag of chips is $0.50 -/
theorem chip_cost_is_fifty_cents : chip_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_chip_cost_is_fifty_cents_l2902_290263


namespace NUMINAMATH_CALUDE_fraction_order_l2902_290242

theorem fraction_order : 
  let f1 := (4 : ℚ) / 3
  let f2 := (4 : ℚ) / 5
  let f3 := (4 : ℚ) / 6
  let f4 := (3 : ℚ) / 5
  let f5 := (6 : ℚ) / 5
  let f6 := (2 : ℚ) / 5
  (f6 < f4) ∧ (f4 < f3) ∧ (f3 < f2) ∧ (f2 < f5) ∧ (f5 < f1) := by
sorry

end NUMINAMATH_CALUDE_fraction_order_l2902_290242


namespace NUMINAMATH_CALUDE_extreme_value_at_zero_tangent_line_equation_decreasing_condition_l2902_290232

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

def f_prime (a : ℝ) (x : ℝ) : ℝ := (-3 * x^2 + (6 - a) * x + a) / Real.exp x

theorem extreme_value_at_zero (a : ℝ) :
  f_prime a 0 = 0 → a = 0 := by sorry

theorem tangent_line_equation (a : ℝ) :
  a = 0 → ∀ x y : ℝ, y = f a x → (3 * x - Real.exp 1 * y = 0 ↔ x = 1) := by sorry

theorem decreasing_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 3 → f_prime a x ≤ 0) ↔ a ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_zero_tangent_line_equation_decreasing_condition_l2902_290232


namespace NUMINAMATH_CALUDE_a_formula_l2902_290285

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ+) : ℤ := n^2 - n

/-- The nth term of the sequence a_n -/
def a (n : ℕ+) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_formula (n : ℕ+) : a n = 2*n - 2 := by
  sorry

end NUMINAMATH_CALUDE_a_formula_l2902_290285


namespace NUMINAMATH_CALUDE_intersection_complement_subset_condition_l2902_290246

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Part I: Prove A ∩ (C \ B) = (-3, 2]
theorem intersection_complement (a : ℝ) (h : a > 0) :
  A ∩ (Set.diff (C a) B) = Set.Ioc (-3) 2 :=
sorry

-- Part II: Prove the range of a for which C ⊇ (A ∩ B)
theorem subset_condition (a : ℝ) (h : a > 0) :
  C a ⊇ (A ∩ B) ↔ 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_subset_condition_l2902_290246


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_of_primes_l2902_290235

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem least_positive_linear_combination_of_primes :
  ∃ (x y z w : ℕ), 
    is_prime x ∧ is_prime y ∧ is_prime z ∧ is_prime w ∧
    24*x + 16*y - 7*z + 5*w = 13 ∧
    (∀ (a b c d : ℕ), is_prime a → is_prime b → is_prime c → is_prime d →
      24*a + 16*b - 7*c + 5*d > 0 → 24*a + 16*b - 7*c + 5*d ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_of_primes_l2902_290235


namespace NUMINAMATH_CALUDE_loop_iterations_count_l2902_290292

theorem loop_iterations_count (i : ℕ) : 
  i = 20 → (∀ n : ℕ, n < 20 → i - n > 0) ∧ (i - 20 = 0) := by sorry

end NUMINAMATH_CALUDE_loop_iterations_count_l2902_290292


namespace NUMINAMATH_CALUDE_survey_result_l2902_290253

def teachers_survey (total : ℕ) (high_bp : ℕ) (heart : ℕ) (diabetes : ℕ) 
  (high_bp_heart : ℕ) (diabetes_heart : ℕ) (diabetes_high_bp : ℕ) (all_three : ℕ) : Prop :=
  let teachers_with_condition := 
    high_bp + heart + diabetes - high_bp_heart - diabetes_heart - diabetes_high_bp + all_three
  let teachers_without_condition := total - teachers_with_condition
  (teachers_without_condition : ℚ) / total * 100 = 28

theorem survey_result : 
  teachers_survey 150 90 60 10 30 5 8 3 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_result_l2902_290253


namespace NUMINAMATH_CALUDE_cube_not_square_in_progression_l2902_290256

/-- An arithmetic progression is represented by its first term and common difference -/
structure ArithmeticProgression (α : Type*) [Add α] where
  first : α
  diff : α

/-- Predicate to check if a number is in an arithmetic progression -/
def inProgression (a : ArithmeticProgression ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = a.first + k * a.diff

/-- Predicate to check if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

theorem cube_not_square_in_progression (a : ArithmeticProgression ℕ) :
  (∃ n : ℕ, inProgression a n ∧ isPerfectCube n) →
  (∃ m : ℕ, inProgression a m ∧ isPerfectCube m ∧ ¬isPerfectSquare m) :=
by sorry

end NUMINAMATH_CALUDE_cube_not_square_in_progression_l2902_290256


namespace NUMINAMATH_CALUDE_sum_x_y_equals_six_l2902_290289

theorem sum_x_y_equals_six (x y : ℝ) : 
  (|x| + x + y = 16) → (x + |y| - y = 18) → (x + y = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_six_l2902_290289


namespace NUMINAMATH_CALUDE_jungkook_has_smallest_number_l2902_290223

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem jungkook_has_smallest_number :
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_smallest_number_l2902_290223


namespace NUMINAMATH_CALUDE_circle_tangency_l2902_290207

/-- Two circles are tangent internally if the distance between their centers
    is equal to the absolute difference of their radii -/
def are_tangent_internally (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (r1 - r2)^2

/-- The statement of the problem -/
theorem circle_tangency (m : ℝ) : 
  are_tangent_internally (m, -2) (-1, m) 3 2 ↔ m = -2 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l2902_290207


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_theorem_l2902_290202

/-- Given two positive integers m and n with specific HCF, LCM, and sum,
    prove that the sum of their reciprocals equals 2/31.5 -/
theorem sum_of_reciprocals_theorem (m n : ℕ+) : 
  Nat.gcd m.val n.val = 6 →
  Nat.lcm m.val n.val = 210 →
  m + n = 80 →
  (1 : ℚ) / m + (1 : ℚ) / n = 2 / 31.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_theorem_l2902_290202


namespace NUMINAMATH_CALUDE_tan_problem_l2902_290219

theorem tan_problem (α β : Real) 
  (h1 : Real.tan (π/4 + α) = 2) 
  (h2 : Real.tan β = 1/2) : 
  Real.tan α = 1/3 ∧ 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / 
  (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_problem_l2902_290219


namespace NUMINAMATH_CALUDE_kylie_made_five_bracelets_l2902_290209

/-- The number of beaded bracelets Kylie made on Wednesday -/
def bracelets_made_wednesday (
  monday_necklaces : ℕ)
  (tuesday_necklaces : ℕ)
  (wednesday_earrings : ℕ)
  (beads_per_necklace : ℕ)
  (beads_per_bracelet : ℕ)
  (beads_per_earring : ℕ)
  (total_beads_used : ℕ) : ℕ :=
  (total_beads_used - 
   (monday_necklaces + tuesday_necklaces) * beads_per_necklace - 
   wednesday_earrings * beads_per_earring) / 
  beads_per_bracelet

/-- Theorem stating that Kylie made 5 beaded bracelets on Wednesday -/
theorem kylie_made_five_bracelets : 
  bracelets_made_wednesday 10 2 7 20 10 5 325 = 5 := by
  sorry

end NUMINAMATH_CALUDE_kylie_made_five_bracelets_l2902_290209


namespace NUMINAMATH_CALUDE_problem_statement_l2902_290275

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- The statement to be proved -/
theorem problem_statement (m : ℝ) : 
  (∀ x : ℝ, f m x > 0) ∧ 
  (∃ x : ℝ, x^2 < 9 - m^2) ↔ 
  (m > -3 ∧ m ≤ -2) ∨ (m ≥ 2 ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2902_290275


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2902_290298

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | x < -3 ∨ x > 4}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, f a b c x > 0 ↔ x ∈ solution_set a b c) :
  a > 0 ∧
  (∀ x, c * x^2 - b * x + a < 0 ↔ x < -1/4 ∨ x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2902_290298


namespace NUMINAMATH_CALUDE_sum_squares_inequality_l2902_290251

theorem sum_squares_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 5) : 
  a^2 + 2*b^2 + c^2 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_l2902_290251


namespace NUMINAMATH_CALUDE_cole_math_classes_l2902_290294

/-- Represents the number of students in Ms. Cole's sixth-level math class -/
def sixth_level_students : ℕ := sorry

/-- Represents the number of students in Ms. Cole's fourth-level math class -/
def fourth_level_students : ℕ := sorry

/-- Represents the number of students in Ms. Cole's seventh-level math class -/
def seventh_level_students : ℕ := sorry

/-- The total number of students Ms. Cole teaches -/
def total_students : ℕ := 520

theorem cole_math_classes :
  (fourth_level_students = 4 * sixth_level_students) ∧
  (seventh_level_students = 2 * fourth_level_students) ∧
  (total_students = sixth_level_students + fourth_level_students + seventh_level_students) →
  sixth_level_students = 40 := by
  sorry

end NUMINAMATH_CALUDE_cole_math_classes_l2902_290294


namespace NUMINAMATH_CALUDE_girl_multiplication_mistake_l2902_290278

theorem girl_multiplication_mistake (x : ℝ) : 43 * x - 34 * x = 1215 → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_girl_multiplication_mistake_l2902_290278


namespace NUMINAMATH_CALUDE_system_solution_l2902_290250

theorem system_solution (x y z t : ℝ) : 
  (x * y - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18) → 
  ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2902_290250


namespace NUMINAMATH_CALUDE_abs_difference_sqrt_square_l2902_290229

theorem abs_difference_sqrt_square (x α : ℝ) (h : x < α) :
  |x - Real.sqrt ((x - α)^2)| = α - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_sqrt_square_l2902_290229


namespace NUMINAMATH_CALUDE_hammond_statue_weight_l2902_290297

/-- Given Hammond's marble carving scenario, prove the weight of each remaining statue. -/
theorem hammond_statue_weight :
  -- Total weight of marble block
  let total_weight : ℕ := 80
  -- Weight of first statue
  let first_statue : ℕ := 10
  -- Weight of second statue
  let second_statue : ℕ := 18
  -- Weight of discarded marble
  let discarded : ℕ := 22
  -- Number of statues
  let num_statues : ℕ := 4
  -- Weight of each remaining statue
  let remaining_statue_weight : ℕ := (total_weight - first_statue - second_statue - discarded) / (num_statues - 2)
  -- Proof that each remaining statue weighs 15 pounds
  remaining_statue_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_hammond_statue_weight_l2902_290297


namespace NUMINAMATH_CALUDE_number_between_fractions_l2902_290217

theorem number_between_fractions : 0.2012 > (1 : ℚ) / 5 ∧ 0.2012 < (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_between_fractions_l2902_290217


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_exists_y_for_min_max_abs_quadratic_minus_linear_l2902_290264

theorem min_max_abs_quadratic_minus_linear (y : ℝ) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| = 4) ∧
  (∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4) :=
by sorry

theorem exists_y_for_min_max_abs_quadratic_minus_linear :
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_exists_y_for_min_max_abs_quadratic_minus_linear_l2902_290264


namespace NUMINAMATH_CALUDE_f_minimum_when_a_is_one_f_nonnegative_iff_a_ge_one_l2902_290277

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 2 / x + a * x - a - 2

theorem f_minimum_when_a_is_one :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, f 1 x ≥ min :=
sorry

theorem f_nonnegative_iff_a_ge_one :
  ∀ a > 0, (∀ x ∈ Set.Icc 1 3, f a x ≥ 0) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_when_a_is_one_f_nonnegative_iff_a_ge_one_l2902_290277


namespace NUMINAMATH_CALUDE_problem_solution_l2902_290299

theorem problem_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  Real.log b / Real.log a = 3 → b - a = 1000 → a + b = 1010 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2902_290299


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2902_290244

def f (x : ℝ) : ℝ := -3 * x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2902_290244


namespace NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l2902_290288

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l2902_290288


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2902_290259

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a^2 + b^2 - c^2 = ab and 2cos(A)sin(B) = sin(C), then the triangle is equilateral. -/
theorem triangle_is_equilateral
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a^2 + b^2 - c^2 = a * b)
  (h2 : 2 * Real.cos A * Real.sin B = Real.sin C)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : A > 0 ∧ B > 0 ∧ C > 0)
  (h5 : A + B + C = π) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2902_290259


namespace NUMINAMATH_CALUDE_sports_competition_solution_l2902_290239

/-- Represents the number of medals distributed on day k --/
def medals_distributed (k : ℕ) (m_k : ℕ) : ℕ :=
  k + (m_k - k) / 7

/-- Represents the number of medals remaining after day k --/
def medals_remaining (k : ℕ) (m_k : ℕ) : ℕ :=
  m_k - medals_distributed k m_k

/-- The sports competition problem --/
theorem sports_competition_solution (n m : ℕ) : 
  (n > 1) →
  (∀ k, k ∈ Finset.range n → medals_distributed k (medals_remaining (k-1) m) = medals_distributed (k+1) (medals_remaining k m)) →
  (medals_distributed n (medals_remaining (n-1) m) = n) →
  (n = 6 ∧ m = 36) :=
by sorry

end NUMINAMATH_CALUDE_sports_competition_solution_l2902_290239


namespace NUMINAMATH_CALUDE_base_conversion_2023_l2902_290282

/-- Converts a number from base 10 to base 8 --/
def toBase8 (n : ℕ) : ℕ := sorry

theorem base_conversion_2023 :
  toBase8 2023 = 3747 := by sorry

end NUMINAMATH_CALUDE_base_conversion_2023_l2902_290282


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2902_290247

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2902_290247


namespace NUMINAMATH_CALUDE_subset_gcd_property_l2902_290271

theorem subset_gcd_property (A : Finset ℕ) 
  (h1 : A ⊆ Finset.range 2007)
  (h2 : A.card = 1004) :
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (Nat.gcd a b ∣ c)) ∧
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(Nat.gcd a b ∣ c)) := by
sorry

end NUMINAMATH_CALUDE_subset_gcd_property_l2902_290271


namespace NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l2902_290269

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 3 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l2902_290269


namespace NUMINAMATH_CALUDE_square_area_error_l2902_290252

theorem square_area_error (side_error : Real) (area_error : Real) : 
  side_error = 0.17 → area_error = 36.89 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l2902_290252


namespace NUMINAMATH_CALUDE_total_cost_proof_l2902_290266

def squat_rack_cost : ℕ := 2500
def barbell_cost_ratio : ℚ := 1 / 10

theorem total_cost_proof :
  squat_rack_cost + (squat_rack_cost : ℚ) * barbell_cost_ratio = 2750 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l2902_290266


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2902_290293

theorem cubic_root_sum_cubes (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (2 - 3*ω + 4*ω^2)^3 + (3 + 2*ω - ω^2)^3 = 1191 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2902_290293


namespace NUMINAMATH_CALUDE_cleaning_payment_l2902_290291

theorem cleaning_payment (payment_per_room : ℚ) (rooms_cleaned : ℚ) (discount_rate : ℚ) :
  payment_per_room = 13/3 →
  rooms_cleaned = 5/2 →
  discount_rate = 1/10 →
  (payment_per_room * rooms_cleaned) * (1 - discount_rate) = 39/4 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_payment_l2902_290291


namespace NUMINAMATH_CALUDE_composite_expression_l2902_290227

theorem composite_expression (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 3^(2*n+1) - 2^(2*n+1) - 6^n := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l2902_290227


namespace NUMINAMATH_CALUDE_ricks_sisters_cards_l2902_290262

/-- The number of cards Rick's sisters receive -/
def cards_per_sister (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (num_friends : ℕ) (cards_per_friend : ℕ) (num_sisters : ℕ) : ℕ :=
  let remaining_cards := total_cards - kept_cards - miguel_cards - (num_friends * cards_per_friend)
  remaining_cards / num_sisters

/-- Proof that each of Rick's sisters received 3 cards -/
theorem ricks_sisters_cards : 
  cards_per_sister 130 15 13 8 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ricks_sisters_cards_l2902_290262


namespace NUMINAMATH_CALUDE_equal_coverings_l2902_290205

/-- Represents a 1993 x 1993 grid -/
def Grid := Fin 1993 × Fin 1993

/-- Represents a 1 x 2 rectangle -/
def Rectangle := Set (Fin 1993 × Fin 1993)

/-- Predicate to check if two squares are on the same edge of the grid -/
def on_same_edge (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = 0 ∨ a.2 = 1992)) ∨
  (a.2 = b.2 ∧ (a.1 = 0 ∨ a.1 = 1992))

/-- Predicate to check if there's an odd number of squares between two squares -/
def odd_squares_between (a b : Grid) : Prop :=
  ∃ n : Nat, n % 2 = 1 ∧
  ((a.1 = b.1 ∧ abs (a.2 - b.2) = n + 1) ∨
   (a.2 = b.2 ∧ abs (a.1 - b.1) = n + 1))

/-- Type representing a covering of the grid with 1 x 2 rectangles -/
def Covering := Set Rectangle

/-- Predicate to check if a covering is valid (covers the entire grid except one square) -/
def valid_covering (c : Covering) (uncovered : Grid) : Prop := sorry

/-- The number of valid coverings that leave a given square uncovered -/
def num_coverings (uncovered : Grid) : Nat := sorry

theorem equal_coverings (A B : Grid)
  (h1 : on_same_edge A B)
  (h2 : odd_squares_between A B) :
  num_coverings A = num_coverings B := by sorry

end NUMINAMATH_CALUDE_equal_coverings_l2902_290205


namespace NUMINAMATH_CALUDE_inverse_g_90_l2902_290237

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_g_90 : g⁻¹ 90 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_90_l2902_290237


namespace NUMINAMATH_CALUDE_regular_survey_rate_l2902_290270

/-- Proves that the regular rate for completing a survey is Rs. 30 given the specified conditions. -/
theorem regular_survey_rate
  (total_surveys : ℕ)
  (cellphone_rate_factor : ℚ)
  (cellphone_surveys : ℕ)
  (total_earnings : ℕ)
  (h1 : total_surveys = 100)
  (h2 : cellphone_rate_factor = 1.20)
  (h3 : cellphone_surveys = 50)
  (h4 : total_earnings = 3300) :
  ∃ (regular_rate : ℚ),
    regular_rate = 30 ∧
    regular_rate * (total_surveys - cellphone_surveys : ℚ) +
    (regular_rate * cellphone_rate_factor) * cellphone_surveys = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_regular_survey_rate_l2902_290270


namespace NUMINAMATH_CALUDE_negative_x_implies_a_greater_than_five_thirds_l2902_290281

theorem negative_x_implies_a_greater_than_five_thirds
  (x a : ℝ) -- x and a are real numbers
  (h1 : x - 5 = -3 * a) -- given equation
  (h2 : x < 0) -- x is negative
  : a > 5/3 := by
sorry

end NUMINAMATH_CALUDE_negative_x_implies_a_greater_than_five_thirds_l2902_290281


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2902_290203

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2902_290203


namespace NUMINAMATH_CALUDE_veranda_area_l2902_290261

/-- The area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 140 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l2902_290261


namespace NUMINAMATH_CALUDE_range_of_a_l2902_290295

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2902_290295


namespace NUMINAMATH_CALUDE_problem_solution_l2902_290216

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}

theorem problem_solution (m : ℝ) (α : ℝ) :
  (∀ x, determinant (x + m) 2 1 x < 0 ↔ x ∈ Set.Ioo (-1) 2) →
  m * Real.cos α + 2 * Real.sin α = 0 →
  m = -1 ∧ Real.tan (2 * α - Real.pi / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2902_290216


namespace NUMINAMATH_CALUDE_enhanced_computer_price_difference_l2902_290255

/-- The price difference between an enhanced computer and a basic computer -/
def price_difference (total_basic : ℝ) (price_basic : ℝ) : ℝ :=
  let price_printer := total_basic - price_basic
  let price_enhanced := 6 * price_printer
  price_enhanced - price_basic

/-- Theorem stating the price difference between enhanced and basic computers -/
theorem enhanced_computer_price_difference :
  price_difference 2500 2000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_enhanced_computer_price_difference_l2902_290255


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2902_290257

/-- Given a hyperbola with equation (x²/a²) - (y²/b²) = 1, where a > 0 and b > 0,
    with eccentricity 2 and distance from focus to asymptote √3,
    prove that its focal length is 4. -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let d := Real.sqrt 3  -- distance from focus to asymptote
  let c := e * a  -- distance from center to focus
  let focal_length := 2 * c
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → 
    e = c / a ∧
    d = (b * c) / Real.sqrt (a^2 + b^2)) →
  focal_length = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2902_290257


namespace NUMINAMATH_CALUDE_helga_extra_hours_l2902_290267

/-- Represents Helga's work schedule and productivity --/
structure HelgaWork where
  articles_per_half_hour : ℕ := 5
  regular_hours_per_day : ℕ := 4
  regular_days_per_week : ℕ := 5
  extra_hours_thursday : ℕ := 2
  total_articles_week : ℕ := 250

/-- Calculates the number of extra hours Helga worked on Friday --/
def extra_hours_friday (hw : HelgaWork) : ℕ :=
  sorry

/-- Theorem stating that Helga worked 3 extra hours on Friday --/
theorem helga_extra_hours (hw : HelgaWork) : extra_hours_friday hw = 3 := by
  sorry

end NUMINAMATH_CALUDE_helga_extra_hours_l2902_290267


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2902_290280

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧
  (n / 10) % 20 = 0 ∧
  (n % 1000) % 21 = 0 ∧
  (n / 100 % 10) ≠ 0

theorem smallest_valid_number :
  is_valid 1609 ∧ ∀ m < 1609, ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2902_290280


namespace NUMINAMATH_CALUDE_mean_score_of_all_students_l2902_290274

theorem mean_score_of_all_students
  (avg_score_group1 : ℝ)
  (avg_score_group2 : ℝ)
  (ratio_students : ℚ)
  (h1 : avg_score_group1 = 90)
  (h2 : avg_score_group2 = 75)
  (h3 : ratio_students = 2/5) :
  let total_score := avg_score_group1 * (ratio_students * s) + avg_score_group2 * s
  let total_students := ratio_students * s + s
  total_score / total_students = 79 :=
by
  sorry

#check mean_score_of_all_students

end NUMINAMATH_CALUDE_mean_score_of_all_students_l2902_290274


namespace NUMINAMATH_CALUDE_larger_number_proof_l2902_290208

theorem larger_number_proof (x y : ℝ) (h1 : y > x) (h2 : 4 * y = 5 * x) (h3 : y - x = 10) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2902_290208


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l2902_290287

/-- For a triangle ABC with angles A, B, C satisfying A/B = B/C = 1/3, 
    the sum of cosines of these angles is (1 + √13) / 4 -/
theorem triangle_cosine_sum (A B C : Real) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  A / B = 1 / 3 →
  B / C = 1 / 3 →
  Real.cos A + Real.cos B + Real.cos C = (1 + Real.sqrt 13) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l2902_290287


namespace NUMINAMATH_CALUDE_number_where_one_seventh_is_five_l2902_290249

theorem number_where_one_seventh_is_five : 
  ∃ n : ℝ, (1 / 7 : ℝ) * n = 5 → n = 35 :=
by sorry

end NUMINAMATH_CALUDE_number_where_one_seventh_is_five_l2902_290249


namespace NUMINAMATH_CALUDE_grocery_cost_l2902_290214

/-- The cost of groceries problem -/
theorem grocery_cost (mango_cost rice_cost flour_cost : ℝ) : 
  (10 * mango_cost = 24 * rice_cost) →
  (flour_cost = 2 * rice_cost) →
  (flour_cost = 24) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 271.2) := by
sorry

end NUMINAMATH_CALUDE_grocery_cost_l2902_290214


namespace NUMINAMATH_CALUDE_amy_blue_balloons_l2902_290233

theorem amy_blue_balloons :
  let total_balloons : ℕ := 67
  let red_balloons : ℕ := 29
  let green_balloons : ℕ := 17
  let blue_balloons : ℕ := total_balloons - red_balloons - green_balloons
  blue_balloons = 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_blue_balloons_l2902_290233


namespace NUMINAMATH_CALUDE_cement_mixture_water_fraction_l2902_290204

theorem cement_mixture_water_fraction 
  (total_weight : ℝ) 
  (sand_fraction : ℝ) 
  (gravel_weight : ℝ) 
  (h1 : total_weight = 49.99999999999999)
  (h2 : sand_fraction = 1/2)
  (h3 : gravel_weight = 15) :
  (total_weight - sand_fraction * total_weight - gravel_weight) / total_weight = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_water_fraction_l2902_290204


namespace NUMINAMATH_CALUDE_ratios_neither_necessary_nor_sufficient_l2902_290273

-- Define the coefficients for the two quadratic inequalities
variable (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ)

-- Define the solution sets for the two inequalities
def SolutionSet1 (x : ℝ) := a₁ * x^2 + b₁ * x + c₁ > 0
def SolutionSet2 (x : ℝ) := a₂ * x^2 + b₂ * x + c₂ > 0

-- Define the equality of ratios condition
def RatiosEqual := (a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)

-- Define the property of having the same solution set
def SameSolutionSet := ∀ x, SolutionSet1 a₁ b₁ c₁ x ↔ SolutionSet2 a₂ b₂ c₂ x

-- Theorem stating that the equality of ratios is neither necessary nor sufficient
theorem ratios_neither_necessary_nor_sufficient :
  ¬(RatiosEqual a₁ b₁ c₁ a₂ b₂ c₂ → SameSolutionSet a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(SameSolutionSet a₁ b₁ c₁ a₂ b₂ c₂ → RatiosEqual a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_ratios_neither_necessary_nor_sufficient_l2902_290273


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2902_290243

theorem children_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ) : 
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  boys = 17 →
  girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 5 →
  total_children - (happy_children + sad_children) = 20 := by
sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2902_290243


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2902_290260

theorem least_number_divisible_by_five_primes : 
  ∀ n : ℕ, n > 0 → (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n ∧ p₅ ∣ n) → n ≥ 2310 :=
by sorry

#check least_number_divisible_by_five_primes

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2902_290260


namespace NUMINAMATH_CALUDE_milk_needed_for_cookies_l2902_290296

/-- The number of cups in a quart -/
def cups_per_quart : ℚ := 4

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℕ := 24

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 6

/-- The number of cups of milk needed for the target number of cookies -/
def milk_needed : ℚ := 3

theorem milk_needed_for_cookies :
  milk_needed = (target_cookies : ℚ) / cookies_per_three_quarts * (3 * cups_per_quart) :=
sorry

end NUMINAMATH_CALUDE_milk_needed_for_cookies_l2902_290296


namespace NUMINAMATH_CALUDE_gcd_217_155_l2902_290220

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_217_155_l2902_290220


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2902_290238

theorem largest_prime_factor_of_expression : 
  (Nat.factors (18^4 + 12^5 - 6^6)).maximum? = some 11 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2902_290238


namespace NUMINAMATH_CALUDE_highlighters_count_l2902_290279

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 3

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 7

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 5

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

theorem highlighters_count : total_highlighters = 15 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l2902_290279


namespace NUMINAMATH_CALUDE_at_least_one_meets_standard_l2902_290228

theorem at_least_one_meets_standard (pA pB pC : ℝ) 
  (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_meets_standard_l2902_290228


namespace NUMINAMATH_CALUDE_ternary_to_decimal_l2902_290240

/-- 
Given a positive integer k, if the ternary number 10k2 is equal to 35 in decimal, 
then k is equal to 2.
-/
theorem ternary_to_decimal (k : ℕ+) : 
  (1 * 3^3 + k * 3 + 2 = 35) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ternary_to_decimal_l2902_290240


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l2902_290286

-- Define the arithmetic progression
def arithmeticProgression (n : ℕ) : ℕ := 36 * n + 3

-- Define the property of not being a sum of two squares
def notSumOfTwoSquares (k : ℕ) : Prop := ∀ a b : ℕ, k ≠ a^2 + b^2

-- Define the property of not being a sum of two cubes
def notSumOfTwoCubes (k : ℕ) : Prop := ∀ a b : ℕ, k ≠ a^3 + b^3

theorem arithmetic_progression_properties :
  (∀ n : ℕ, arithmeticProgression n > 0) ∧  -- Positive integers
  (∀ n m : ℕ, n ≠ m → arithmeticProgression n ≠ arithmeticProgression m) ∧  -- Non-constant
  (∀ n : ℕ, notSumOfTwoSquares (arithmeticProgression n)) ∧  -- Not sum of two squares
  (∀ n : ℕ, notSumOfTwoCubes (arithmeticProgression n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l2902_290286
