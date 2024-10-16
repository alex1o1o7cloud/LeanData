import Mathlib

namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_range_of_m_when_B_subset_A_l693_69327

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- Theorem 1: When m = 1, A ∩ B = { x | 1 < x < 2 }
theorem intersection_when_m_is_one :
  A ∩ B 1 = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: If B ⊆ A, then m ∈ [-1, +∞)
theorem range_of_m_when_B_subset_A :
  (∀ m : ℝ, B m ⊆ A) → {m : ℝ | -1 ≤ m} = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_range_of_m_when_B_subset_A_l693_69327


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l693_69382

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_pos : ∀ n, a n > 0) 
    (h_prod : a 3 * a 7 = 64) : 
  a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l693_69382


namespace NUMINAMATH_CALUDE_sin_15_times_sin_75_l693_69391

theorem sin_15_times_sin_75 : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_times_sin_75_l693_69391


namespace NUMINAMATH_CALUDE_total_fishermen_l693_69369

theorem total_fishermen (total_fish : ℕ) (fish_per_group : ℕ) (group_size : ℕ) (last_fisherman_fish : ℕ) :
  total_fish = group_size * fish_per_group + last_fisherman_fish →
  total_fish = 10000 →
  fish_per_group = 400 →
  group_size = 19 →
  last_fisherman_fish = 2400 →
  group_size + 1 = 20 := by
sorry

end NUMINAMATH_CALUDE_total_fishermen_l693_69369


namespace NUMINAMATH_CALUDE_muffin_apples_count_l693_69324

def initial_apples : ℕ := 62
def refrigerated_apples : ℕ := 25

def apples_for_muffins : ℕ :=
  initial_apples - (initial_apples / 2 + refrigerated_apples)

theorem muffin_apples_count :
  apples_for_muffins = 6 :=
by sorry

end NUMINAMATH_CALUDE_muffin_apples_count_l693_69324


namespace NUMINAMATH_CALUDE_k_value_for_decreasing_function_l693_69377

theorem k_value_for_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)
  (h_domain : ∀ x, x ≤ 1 → ∃ y, f x = y)
  (h_inequality : ∀ x : ℝ, f (k - Real.sin x) ≥ f (k^2 - Real.sin x^2))
  : k = -1 :=
sorry

end NUMINAMATH_CALUDE_k_value_for_decreasing_function_l693_69377


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_positive_l693_69386

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem negation_of_squared_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_positive_l693_69386


namespace NUMINAMATH_CALUDE_gcd_3375_9180_l693_69338

theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3375_9180_l693_69338


namespace NUMINAMATH_CALUDE_efficiency_ratio_l693_69399

-- Define the efficiencies of workers a, b, c, and d
def efficiency_a : ℚ := 1 / 18
def efficiency_b : ℚ := 1 / 36
def efficiency_c : ℚ := 1 / 20
def efficiency_d : ℚ := 1 / 30

-- Theorem statement
theorem efficiency_ratio :
  -- a and b together have the same efficiency as c and d together
  efficiency_a + efficiency_b = efficiency_c + efficiency_d →
  -- The ratio of a's efficiency to b's efficiency is 2:1
  efficiency_a / efficiency_b = 2 := by
sorry

end NUMINAMATH_CALUDE_efficiency_ratio_l693_69399


namespace NUMINAMATH_CALUDE_alice_above_quota_l693_69320

def alice_sales (quota nike_price adidas_price reebok_price : ℕ) 
                (nike_sold adidas_sold reebok_sold : ℕ) : ℕ := 
  nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold

theorem alice_above_quota : 
  let quota : ℕ := 1000
  let nike_price : ℕ := 60
  let adidas_price : ℕ := 45
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let adidas_sold : ℕ := 6
  let reebok_sold : ℕ := 9
  alice_sales quota nike_price adidas_price reebok_price nike_sold adidas_sold reebok_sold - quota = 65 := by
  sorry

end NUMINAMATH_CALUDE_alice_above_quota_l693_69320


namespace NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l693_69350

theorem cos_pi_4_minus_alpha (α : Real) (h : Real.sin (α - 7 * Real.pi / 4) = 1 / 2) :
  Real.cos (Real.pi / 4 - α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l693_69350


namespace NUMINAMATH_CALUDE_union_M_N_equals_real_l693_69352

-- Define the sets M and N
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

-- State the theorem
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_real_l693_69352


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_fifth_powers_l693_69398

theorem divisibility_of_sum_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_fifth_powers_l693_69398


namespace NUMINAMATH_CALUDE_square_greater_than_self_when_less_than_negative_one_l693_69379

theorem square_greater_than_self_when_less_than_negative_one (x : ℝ) : 
  x < -1 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_self_when_less_than_negative_one_l693_69379


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l693_69380

/-- Two lines y = m₁x + b₁ and y = m₂x + b₂ are perpendicular if and only if m₁ * m₂ = -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The statement "a = 2 is a sufficient but not necessary condition for the lines
    y = -ax + 2 and y = (a/4)x - 1 to be perpendicular" -/
theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2 → perpendicular (-a) (a/4)) ∧ 
  ¬(perpendicular (-a) (a/4) → a = 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l693_69380


namespace NUMINAMATH_CALUDE_lars_production_l693_69372

/-- Represents the baking rates and working hours of Lars' bakeshop --/
structure BakeshopData where
  bread_rate : ℕ  -- loaves of bread per hour
  baguette_rate : ℕ  -- baguettes per 2 hours
  croissant_rate : ℕ  -- croissants per 75 minutes
  working_hours : ℕ  -- hours worked per day

/-- Calculates the daily production of baked goods --/
def daily_production (data : BakeshopData) : ℕ × ℕ × ℕ :=
  let bread := data.bread_rate * data.working_hours
  let baguettes := data.baguette_rate * (data.working_hours / 2)
  let croissants := data.croissant_rate * (data.working_hours * 60 / 75)
  (bread, baguettes, croissants)

/-- Theorem stating Lars' daily production --/
theorem lars_production :
  let data : BakeshopData := {
    bread_rate := 10,
    baguette_rate := 30,
    croissant_rate := 20,
    working_hours := 6
  }
  daily_production data = (60, 90, 80) := by
  sorry

end NUMINAMATH_CALUDE_lars_production_l693_69372


namespace NUMINAMATH_CALUDE_like_terms_exponent_equality_l693_69375

theorem like_terms_exponent_equality (a b : ℤ) : 
  (2 * a + b = 6 ∧ a - b = 3) → a + 2 * b = 3 := by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_equality_l693_69375


namespace NUMINAMATH_CALUDE_heavy_traffic_time_l693_69397

/-- Proves that the time taken to drive with heavy traffic is 5 hours -/
theorem heavy_traffic_time (distance : ℝ) (no_traffic_time : ℝ) (speed_difference : ℝ) :
  distance = 200 →
  no_traffic_time = 4 →
  speed_difference = 10 →
  (distance / (distance / no_traffic_time - speed_difference)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_heavy_traffic_time_l693_69397


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l693_69360

theorem simplify_and_ratio : ∀ m : ℝ, 
  (6 * m + 12) / 3 = 2 * m + 4 ∧ 2 / 4 = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l693_69360


namespace NUMINAMATH_CALUDE_martha_cakes_l693_69304

theorem martha_cakes (num_children : ℕ) (cakes_per_child : ℕ) 
  (h1 : num_children = 3)
  (h2 : cakes_per_child = 6) :
  num_children * cakes_per_child = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l693_69304


namespace NUMINAMATH_CALUDE_shared_focus_hyperbola_ellipse_l693_69311

/-- Given a hyperbola and an ellipse that share a common focus, prove that the parameter p of the ellipse is equal to 4 -/
theorem shared_focus_hyperbola_ellipse (p : ℝ) : 
  (∀ x y : ℝ, x^2/3 - y^2 = 1 → x^2/8 + y^2/p = 1) → 
  (0 < p) → 
  (p < 8) → 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_shared_focus_hyperbola_ellipse_l693_69311


namespace NUMINAMATH_CALUDE_distance_between_5th_and_30th_red_light_l693_69310

/-- Represents the color of a light in the sequence -/
inductive LightColor
  | Red
  | Green

/-- Calculates the position of a light in the sequence given its number and color -/
def lightPosition (n : Nat) (color : LightColor) : Nat :=
  match color with
  | LightColor.Red => (n - 1) / 3 * 7 + (n - 1) % 3 + 1
  | LightColor.Green => (n - 1) / 4 * 7 + (n - 1) % 4 + 4

/-- The spacing between lights in inches -/
def lightSpacing : Nat := 8

/-- The number of inches in a foot -/
def inchesPerFoot : Nat := 12

/-- Calculates the distance in feet between two lights given their positions -/
def distanceBetweenLights (pos1 pos2 : Nat) : Nat :=
  ((pos2 - pos1) * lightSpacing) / inchesPerFoot

theorem distance_between_5th_and_30th_red_light :
  distanceBetweenLights (lightPosition 5 LightColor.Red) (lightPosition 30 LightColor.Red) = 41 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_5th_and_30th_red_light_l693_69310


namespace NUMINAMATH_CALUDE_smallest_n_for_polynomial_roots_l693_69389

theorem smallest_n_for_polynomial_roots : ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, 0 < k → k < n →
    ¬∃ (a b : ℤ), ∃ (x y : ℝ),
      0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
      k * x^2 + a * x + b = 0 ∧
      k * y^2 + a * y + b = 0) ∧
  (∃ (a b : ℤ), ∃ (x y : ℝ),
    0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
    n * x^2 + a * x + b = 0 ∧
    n * y^2 + a * y + b = 0) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_polynomial_roots_l693_69389


namespace NUMINAMATH_CALUDE_hexagon_fills_ground_l693_69364

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def can_fill_ground (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * interior_angle n = 360

theorem hexagon_fills_ground :
  can_fill_ground 6 ∧
  ¬ can_fill_ground 10 ∧
  ¬ can_fill_ground 8 ∧
  ¬ can_fill_ground 5 := by sorry

end NUMINAMATH_CALUDE_hexagon_fills_ground_l693_69364


namespace NUMINAMATH_CALUDE_x_eleven_percent_greater_than_90_l693_69341

theorem x_eleven_percent_greater_than_90 :
  ∀ x : ℝ, x = 90 * (1 + 11 / 100) → x = 99.9 := by
  sorry

end NUMINAMATH_CALUDE_x_eleven_percent_greater_than_90_l693_69341


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l693_69353

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ,
  x^50 + x^40 + x^30 + x^20 + x^10 + 1 = 
  (x^5 + x^4 + x^3 + x^2 + x + 1) * q + (-2 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l693_69353


namespace NUMINAMATH_CALUDE_seven_people_circular_arrangement_l693_69361

/-- The number of ways to arrange n people around a circular table -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n people around a circular table,
    where k specific people must sit together -/
def circularArrangementsWithGroup (n k : ℕ) : ℕ :=
  circularArrangements (n - k + 1) * (k - 1).factorial

theorem seven_people_circular_arrangement :
  circularArrangementsWithGroup 7 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_circular_arrangement_l693_69361


namespace NUMINAMATH_CALUDE_expression_evaluation_l693_69376

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -1/2
  2 * (3 * x^3 - x + 3 * y) - (x - 2 * y + 6 * x^3) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l693_69376


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l693_69373

theorem least_k_cube_divisible_by_120 :
  ∀ k : ℕ, k > 0 → k^3 % 120 = 0 → k ≥ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l693_69373


namespace NUMINAMATH_CALUDE_empty_set_problem_l693_69366

-- Define the sets
def set_A : Set ℝ := {x | x^2 - 4 = 0}
def set_B : Set ℝ := {x | x > 9 ∨ x < 3}
def set_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}
def set_D : Set ℝ := {x | x > 9 ∧ x < 3}

-- Theorem statement
theorem empty_set_problem :
  (set_A.Nonempty) ∧
  (set_B.Nonempty) ∧
  (set_C.Nonempty) ∧
  (set_D = ∅) :=
sorry

end NUMINAMATH_CALUDE_empty_set_problem_l693_69366


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l693_69394

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 1) ∧ (∃ x, x > 1 ∧ x ≤ a) ↔ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l693_69394


namespace NUMINAMATH_CALUDE_min_max_y_l693_69334

/-- The function f(x) = 2 + x -/
def f (x : ℝ) : ℝ := 2 + x

/-- The function y = [f(x)]^2 + f(x) -/
def y (x : ℝ) : ℝ := (f x)^2 + f x

theorem min_max_y :
  (∀ x ∈ Set.Icc 1 9, y 1 ≤ y x) ∧
  (∀ x ∈ Set.Icc 1 9, y x ≤ y 9) ∧
  y 1 = 13 ∧
  y 9 = 141 := by sorry

end NUMINAMATH_CALUDE_min_max_y_l693_69334


namespace NUMINAMATH_CALUDE_unique_solution_system_l693_69335

theorem unique_solution_system (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧
  (b * c + d + a = 5) ∧
  (c * d + a + b = 2) ∧
  (d * a + b + c = 6) →
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l693_69335


namespace NUMINAMATH_CALUDE_cubic_three_roots_l693_69395

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem cubic_three_roots : ∃ (a b c : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_l693_69395


namespace NUMINAMATH_CALUDE_ocean_depth_l693_69316

/-- The depth of the ocean given echo sounder measurements -/
theorem ocean_depth (t : ℝ) (v : ℝ) (h : ℝ) : t = 5 → v = 1.5 → h = (t * v * 1000) / 2 → h = 3750 :=
by sorry

end NUMINAMATH_CALUDE_ocean_depth_l693_69316


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l693_69349

theorem magnitude_of_complex_power : 
  Complex.abs ((2/3 : ℂ) + (5/6 : ℂ) * Complex.I) ^ 8 = (41^4 : ℝ) / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l693_69349


namespace NUMINAMATH_CALUDE_max_value_polynomial_l693_69374

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z w : ℝ), z + w = 5 ∧ 
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≥ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4) ∧
  (∀ (z w : ℝ), z + w = 5 → 
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≤ 6084/17) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l693_69374


namespace NUMINAMATH_CALUDE_equation_solution_l693_69300

theorem equation_solution : 
  ∃ x : ℚ, (x - 1) / 2 = 1 - (x + 2) / 3 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l693_69300


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l693_69393

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l693_69393


namespace NUMINAMATH_CALUDE_complete_square_m_value_l693_69306

/-- Given the equation x^2 + 2x - 1 = 0, prove that when completing the square,
    the resulting equation (x+m)^2 = 2 has m = 1 -/
theorem complete_square_m_value (x : ℝ) :
  x^2 + 2*x - 1 = 0 → ∃ m : ℝ, (x + m)^2 = 2 ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_m_value_l693_69306


namespace NUMINAMATH_CALUDE_complement_implies_a_value_l693_69312

def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}
def P (a : ℝ) : Set ℝ := {2, a^2 + 2 - a}

theorem complement_implies_a_value (a : ℝ) : 
  (U a \ P a = {-1}) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complement_implies_a_value_l693_69312


namespace NUMINAMATH_CALUDE_sin_product_identity_l693_69333

theorem sin_product_identity : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (60 * π / 180) * Real.sin (72 * π / 180) = 
  ((Real.sqrt 5 + 1) * Real.sqrt 3) / 16 := by
sorry

end NUMINAMATH_CALUDE_sin_product_identity_l693_69333


namespace NUMINAMATH_CALUDE_function_properties_l693_69331

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem function_properties (f g : ℝ → ℝ) (h : ∀ x, f x + g x = (1/2)^x)
  (hf : IsOdd f) (hg : IsEven g) :
  (∀ x, f x = (1/2) * (2^(-x) - 2^x)) ∧
  (∀ x, g x = (1/2) * (2^(-x) + 2^x)) ∧
  (∃ x₀ ∈ Set.Icc (1/2) 1, ∃ a : ℝ, a * f x₀ + g (2*x₀) = 0 →
    a ∈ Set.Icc (2 * Real.sqrt 2) (17/6)) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l693_69331


namespace NUMINAMATH_CALUDE_quadratic_always_real_solution_l693_69332

theorem quadratic_always_real_solution (m : ℝ) : 
  ∃ x : ℝ, x^2 - m*x + (m - 1) = 0 :=
by
  sorry

#check quadratic_always_real_solution

end NUMINAMATH_CALUDE_quadratic_always_real_solution_l693_69332


namespace NUMINAMATH_CALUDE_max_area_triangle_ellipse_line_intersection_l693_69328

/-- The maximum area of a triangle formed by two intersection points of a line with an ellipse and the origin -/
theorem max_area_triangle_ellipse_line_intersection :
  ∃ (k : ℝ),
    let ellipse := {(x, y) : ℝ × ℝ | x^2 / 3 + y^2 = 1}
    let line := {(x, y) : ℝ × ℝ | y = k * x + 2}
    let intersection := ellipse ∩ line
    ∀ (A B : ℝ × ℝ),
      A ∈ intersection → B ∈ intersection → A ≠ B →
      let O := (0, 0)
      let area := abs (A.1 * B.2 - A.2 * B.1) / 2
      area ≤ Real.sqrt 3 / 2 ∧
      ∃ (A' B' : ℝ × ℝ),
        A' ∈ intersection ∧ B' ∈ intersection ∧ A' ≠ B' ∧
        abs (A'.1 * B'.2 - A'.2 * B'.1) / 2 = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_ellipse_line_intersection_l693_69328


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l693_69318

theorem complex_arithmetic_equality : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l693_69318


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l693_69387

theorem max_value_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

theorem max_value_achieved : 
  ∃ a : ℝ, (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) ∧ a = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l693_69387


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l693_69343

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  back_seat_capacity : Nat
  total_capacity : Nat

/-- Calculates the number of people each regular seat can hold -/
def seat_capacity (bus : BusSeating) : Nat :=
  sorry

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people -/
theorem bus_seat_capacity :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    back_seat_capacity := 10,
    total_capacity := 91
  }
  seat_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l693_69343


namespace NUMINAMATH_CALUDE_lcm_of_210_and_605_l693_69351

theorem lcm_of_210_and_605 :
  let a := 210
  let b := 605
  let hcf := 55
  Nat.lcm a b = 2310 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_210_and_605_l693_69351


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l693_69363

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * 
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l693_69363


namespace NUMINAMATH_CALUDE_jason_has_four_balloons_l693_69337

/-- The number of violet balloons Jason has now, given the initial count and the number lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason has 4 violet balloons now. -/
theorem jason_has_four_balloons :
  remaining_balloons 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_four_balloons_l693_69337


namespace NUMINAMATH_CALUDE_inequality_preservation_l693_69345

theorem inequality_preservation (m n : ℝ) (h : m > n) : 2 + m > 2 + n := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l693_69345


namespace NUMINAMATH_CALUDE_value_of_expression_l693_69307

theorem value_of_expression (x : ℝ) (h : x = 5) : 4 * x - 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l693_69307


namespace NUMINAMATH_CALUDE_largest_common_divisor_528_440_l693_69305

theorem largest_common_divisor_528_440 : Nat.gcd 528 440 = 88 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_528_440_l693_69305


namespace NUMINAMATH_CALUDE_deepak_age_l693_69325

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 →
  rahul_age + 6 = 50 →
  deepak_age = 33 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l693_69325


namespace NUMINAMATH_CALUDE_mike_work_time_l693_69340

-- Define the basic task times for sedans (in minutes)
def wash_time : ℝ := 10
def oil_change_time : ℝ := 15
def tire_change_time : ℝ := 30
def paint_time : ℝ := 45
def engine_service_time : ℝ := 60

-- Define the number of tasks for sedans
def sedan_washes : ℕ := 9
def sedan_oil_changes : ℕ := 6
def sedan_tire_changes : ℕ := 2
def sedan_paints : ℕ := 4
def sedan_engine_services : ℕ := 2

-- Define the number of tasks for SUVs
def suv_washes : ℕ := 7
def suv_oil_changes : ℕ := 4
def suv_tire_changes : ℕ := 3
def suv_paints : ℕ := 3
def suv_engine_services : ℕ := 1

-- Define the time multiplier for SUV washing and painting
def suv_time_multiplier : ℝ := 1.5

-- Theorem statement
theorem mike_work_time : 
  let sedan_time := 
    sedan_washes * wash_time + 
    sedan_oil_changes * oil_change_time + 
    sedan_tire_changes * tire_change_time + 
    sedan_paints * paint_time + 
    sedan_engine_services * engine_service_time
  let suv_time := 
    suv_washes * (wash_time * suv_time_multiplier) + 
    suv_oil_changes * oil_change_time + 
    suv_tire_changes * tire_change_time + 
    suv_paints * (paint_time * suv_time_multiplier) + 
    suv_engine_services * engine_service_time
  let total_time := sedan_time + suv_time
  (total_time / 60) = 17.625 := by sorry

end NUMINAMATH_CALUDE_mike_work_time_l693_69340


namespace NUMINAMATH_CALUDE_sandwich_optimization_l693_69355

/-- Represents the number of sandwiches of each type -/
structure SandwichCount where
  cheese : ℕ
  salami : ℕ

/-- Represents the available resources -/
structure Resources where
  bread : ℕ  -- in dkg
  butter : ℕ -- in dkg
  cheese : ℕ -- in dkg
  salami : ℕ -- in dkg

/-- Represents the ingredient requirements for each sandwich type -/
structure SandwichRequirements where
  cheese_bread : ℕ  -- in dkg
  cheese_butter : ℕ -- in dkg
  cheese_cheese : ℕ -- in dkg
  salami_bread : ℕ  -- in dkg
  salami_butter : ℕ -- in dkg
  salami_salami : ℕ -- in dkg

def is_valid_sandwich_count (count : SandwichCount) (resources : Resources) 
    (requirements : SandwichRequirements) : Prop :=
  count.cheese * requirements.cheese_bread + count.salami * requirements.salami_bread ≤ resources.bread ∧
  count.cheese * requirements.cheese_butter + count.salami * requirements.salami_butter ≤ resources.butter ∧
  count.cheese * requirements.cheese_cheese ≤ resources.cheese ∧
  count.salami * requirements.salami_salami ≤ resources.salami

def total_sandwiches (count : SandwichCount) : ℕ :=
  count.cheese + count.salami

def revenue (count : SandwichCount) (cheese_price salami_price : ℚ) : ℚ :=
  count.cheese * cheese_price + count.salami * salami_price

def preparation_time (count : SandwichCount) (cheese_time salami_time : ℕ) : ℕ :=
  count.cheese * cheese_time + count.salami * salami_time

theorem sandwich_optimization (resources : Resources) 
    (requirements : SandwichRequirements) 
    (cheese_price salami_price : ℚ) 
    (cheese_time salami_time : ℕ) :
    ∃ (max_count optimal_revenue_count optimal_time_count : SandwichCount),
      is_valid_sandwich_count max_count resources requirements ∧
      total_sandwiches max_count = 40 ∧
      (∀ count, is_valid_sandwich_count count resources requirements → 
        total_sandwiches count ≤ total_sandwiches max_count) ∧
      is_valid_sandwich_count optimal_revenue_count resources requirements ∧
      revenue optimal_revenue_count cheese_price salami_price = 63.5 ∧
      (∀ count, is_valid_sandwich_count count resources requirements → 
        revenue count cheese_price salami_price ≤ revenue optimal_revenue_count cheese_price salami_price) ∧
      is_valid_sandwich_count optimal_time_count resources requirements ∧
      total_sandwiches optimal_time_count = 40 ∧
      preparation_time optimal_time_count cheese_time salami_time = 50 ∧
      (∀ count, is_valid_sandwich_count count resources requirements ∧ total_sandwiches count = 40 → 
        preparation_time optimal_time_count cheese_time salami_time ≤ preparation_time count cheese_time salami_time) :=
  sorry

end NUMINAMATH_CALUDE_sandwich_optimization_l693_69355


namespace NUMINAMATH_CALUDE_maintenance_model_correct_l693_69301

/-- Linear regression model for device maintenance cost --/
structure MaintenanceModel where
  b : ℝ  -- Slope of the regression line
  a : ℝ  -- Y-intercept of the regression line

/-- Conditions for the maintenance cost model --/
class MaintenanceConditions (model : MaintenanceModel) where
  avg_point : 5.4 = 4 * model.b + model.a
  cost_diff : 8 * model.b + model.a - (7 * model.b + model.a) = 1.1

/-- Theorem stating the correctness of the derived model and its prediction --/
theorem maintenance_model_correct (model : MaintenanceModel) 
  [cond : MaintenanceConditions model] : 
  model.b = 0.55 ∧ model.a = 3.2 ∧ 
  (0.55 * 10 + 3.2 : ℝ) = 8.7 := by
  sorry

#check maintenance_model_correct

end NUMINAMATH_CALUDE_maintenance_model_correct_l693_69301


namespace NUMINAMATH_CALUDE_seven_place_value_difference_l693_69367

def number : ℕ := 54179759

def first_seven_place_value : ℕ := 10000
def second_seven_place_value : ℕ := 10

def first_seven_value : ℕ := 7 * first_seven_place_value
def second_seven_value : ℕ := 7 * second_seven_place_value

theorem seven_place_value_difference : 
  first_seven_value - second_seven_value = 69930 := by
  sorry

end NUMINAMATH_CALUDE_seven_place_value_difference_l693_69367


namespace NUMINAMATH_CALUDE_total_laundry_time_l693_69385

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalTime (lt : LaundryTime) : ℕ := lt.washing + lt.drying

/-- Given laundry times for whites, darks, and colors, proves that the total time is 344 minutes -/
theorem total_laundry_time (whites darks colors : LaundryTime)
    (h1 : whites = ⟨72, 50⟩)
    (h2 : darks = ⟨58, 65⟩)
    (h3 : colors = ⟨45, 54⟩) :
    totalTime whites + totalTime darks + totalTime colors = 344 := by
  sorry


end NUMINAMATH_CALUDE_total_laundry_time_l693_69385


namespace NUMINAMATH_CALUDE_factors_of_34020_l693_69370

/-- The number of positive factors of 34020 -/
def num_factors : ℕ := 72

/-- The prime factorization of 34020 -/
def prime_factorization : List (ℕ × ℕ) := [(3, 5), (5, 1), (2, 2), (7, 1)]

theorem factors_of_34020 : (Nat.divisors 34020).card = num_factors := by sorry

end NUMINAMATH_CALUDE_factors_of_34020_l693_69370


namespace NUMINAMATH_CALUDE_julia_trip_euros_l693_69378

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Exchange rate from USD to EUR -/
def exchange_rate : ℚ := 8 / 5

theorem julia_trip_euros (d' : ℕ) : 
  (exchange_rate * d' - 80 : ℚ) = d' → sum_of_digits d' = 7 := by
  sorry

end NUMINAMATH_CALUDE_julia_trip_euros_l693_69378


namespace NUMINAMATH_CALUDE_perpendicular_lines_l693_69383

/-- A line in 2D space defined by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A line in 2D space defined by a standard equation ax + by = c --/
structure StandardLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Convert a parametric line to its standard form --/
def parametricToStandard (l : ParametricLine) : StandardLine :=
  sorry

/-- Check if two lines are perpendicular --/
def arePerpendicular (l1 l2 : StandardLine) : Prop :=
  sorry

/-- The main theorem --/
theorem perpendicular_lines (k : ℝ) : 
  let l1 := ParametricLine.mk (λ t => 1 + 2*t) (λ t => 3 + 2*t)
  let l2 := StandardLine.mk 4 k 1
  arePerpendicular (parametricToStandard l1) l2 → k = 4 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l693_69383


namespace NUMINAMATH_CALUDE_line_translation_l693_69308

-- Define the original line
def original_line (x : ℝ) : ℝ := 2 * x + 1

-- Define the translation amount
def translation : ℝ := -2

-- Define the translated line
def translated_line (x : ℝ) : ℝ := 2 * x - 1

-- Theorem stating that the translation of the original line results in the translated line
theorem line_translation :
  ∀ x : ℝ, translated_line x = original_line x + translation :=
sorry

end NUMINAMATH_CALUDE_line_translation_l693_69308


namespace NUMINAMATH_CALUDE_problem_solution_l693_69319

theorem problem_solution (a b m n x : ℝ) 
  (h1 : a * b = 1)
  (h2 : m + n = 0)
  (h3 : |x| = 1) :
  2022 * (m + n) + 2018 * x^2 - 2019 * a * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l693_69319


namespace NUMINAMATH_CALUDE_largest_multiple_of_45_with_8_and_0_l693_69362

/-- A function that checks if a natural number consists only of digits 8 and 0 -/
def onlyEightAndZero (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The largest positive multiple of 45 consisting only of digits 8 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_of_45_with_8_and_0 :
  m % 45 = 0 ∧
  onlyEightAndZero m ∧
  (∀ k : ℕ, k > m → k % 45 = 0 → ¬onlyEightAndZero k) ∧
  m / 45 = 197530 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_45_with_8_and_0_l693_69362


namespace NUMINAMATH_CALUDE_elizabeth_haircut_l693_69314

theorem elizabeth_haircut (first_cut second_cut : ℝ) 
  (h1 : first_cut = 0.375)
  (h2 : second_cut = 0.5) :
  first_cut + second_cut = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_haircut_l693_69314


namespace NUMINAMATH_CALUDE_family_photo_arrangements_l693_69396

/-- The number of ways to arrange a family in a line for a picture. -/
def familyArrangements (totalMembers : ℕ) (parents : ℕ) (otherMembers : ℕ) : ℕ :=
  2 * Nat.factorial otherMembers

/-- Theorem stating that for a family of 5 with 2 parents, there are 12 possible arrangements. -/
theorem family_photo_arrangements :
  familyArrangements 5 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_family_photo_arrangements_l693_69396


namespace NUMINAMATH_CALUDE_sons_age_l693_69358

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l693_69358


namespace NUMINAMATH_CALUDE_zeros_after_decimal_point_l693_69371

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the decimal representation of (1 / (2^7 * 5^6)) * (3 / 5^2) is 7. -/
theorem zeros_after_decimal_point : ∃ (n : ℕ) (r : ℚ), 
  (1 / (2^7 * 5^6 : ℚ)) * (3 / 5^2 : ℚ) = 10^(-n : ℤ) * r ∧ 
  0 < r ∧ 
  r < 1 ∧ 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_point_l693_69371


namespace NUMINAMATH_CALUDE_arccos_cos_eq_double_x_solution_l693_69322

theorem arccos_cos_eq_double_x_solution :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → (Real.arccos (Real.cos x) = 2 * x ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_double_x_solution_l693_69322


namespace NUMINAMATH_CALUDE_remaining_books_l693_69388

/-- Given an initial number of books and a number of books sold,
    proves that the remaining number of books is equal to
    the difference between the initial number and the number sold. -/
theorem remaining_books (initial : ℕ) (sold : ℕ) (h : sold ≤ initial) :
  initial - sold = initial - sold :=
by sorry

end NUMINAMATH_CALUDE_remaining_books_l693_69388


namespace NUMINAMATH_CALUDE_smallest_number_l693_69368

theorem smallest_number (a b c d : ℤ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l693_69368


namespace NUMINAMATH_CALUDE_cos_2theta_value_l693_69356

theorem cos_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ) ^ (2 * n) = 3) :
  Real.cos (2 * θ) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l693_69356


namespace NUMINAMATH_CALUDE_mixed_fraction_calculation_l693_69329

theorem mixed_fraction_calculation : 
  (-4 - 2/3) - (1 + 5/6) - (-18 - 1/2) + (-13 - 3/4) = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_calculation_l693_69329


namespace NUMINAMATH_CALUDE_count_special_numbers_l693_69392

theorem count_special_numbers : ∃ (S : Finset Nat),
  (∀ n ∈ S, n < 500 ∧ n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 15 ≠ 0) ∧
  (∀ n < 500, n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 15 ≠ 0 → n ∈ S) ∧
  S.card = 33 :=
by
  sorry

#check count_special_numbers

end NUMINAMATH_CALUDE_count_special_numbers_l693_69392


namespace NUMINAMATH_CALUDE_smallest_side_difference_l693_69342

theorem smallest_side_difference (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2021 →
  PQ < QR →
  QR ≤ PR →
  (∀ PQ' QR' PR' : ℕ, 
    PQ' + QR' + PR' = 2021 → 
    PQ' < QR' → 
    QR' ≤ PR' → 
    QR' - PQ' ≥ 1) →
  QR - PQ = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_difference_l693_69342


namespace NUMINAMATH_CALUDE_smallest_N_for_probability_threshold_l693_69348

def P (N : ℕ) : ℚ := (⌈(3 * N : ℚ) / 5 + 1⌉) / (N + 1 : ℚ)

theorem smallest_N_for_probability_threshold :
  ∀ N : ℕ, 
    N % 5 = 0 → 
    N > 0 → 
    (∀ k : ℕ, k % 5 = 0 → k > 0 → k < N → P k ≥ 321/400) → 
    P N < 321/400 → 
    N = 480 :=
sorry

end NUMINAMATH_CALUDE_smallest_N_for_probability_threshold_l693_69348


namespace NUMINAMATH_CALUDE_egg_selling_problem_l693_69347

theorem egg_selling_problem (n x : ℕ) : 
  (0 < n) → 
  (0 < x) → 
  (x ≤ n) → 
  (120 * n = 206 * x) → 
  (∀ m : ℕ, 0 < m → m < n → 120 * m ≠ 206 * x) →
  (n = 103 ∧ x = 60) :=
by sorry

end NUMINAMATH_CALUDE_egg_selling_problem_l693_69347


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_one_l693_69339

theorem sum_of_x_and_y_is_one (x y : ℝ) (h : x^2 + y^2 + x*y = 12*x - 8*y + 2) : x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_one_l693_69339


namespace NUMINAMATH_CALUDE_system_solution_l693_69365

theorem system_solution : ∃ (x y : ℚ), 
  (x * (1/7)^2 = 7^3) ∧ 
  (x + y = 7^2) ∧ 
  (x = 16807) ∧ 
  (y = -16758) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l693_69365


namespace NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_l693_69317

-- Define the angles
variable (A B C D F G : ℝ)

-- Define the condition that these angles form a quadrilateral
variable (h : IsQuadrilateral A B C D F G)

-- State the theorem
theorem sum_of_angles_in_quadrilateral :
  A + B + C + D + F + G = 360 :=
sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_l693_69317


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l693_69330

theorem journey_speed_calculation (D : ℝ) (v : ℝ) (h1 : D > 0) (h2 : v > 0) : 
  (D / ((0.8 * D / 80) + (0.2 * D / v)) = 50) → v = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l693_69330


namespace NUMINAMATH_CALUDE_share_proportion_l693_69303

theorem share_proportion (c d : ℕ) (h1 : c = d + 500) (h2 : d = 1500) :
  c / d = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_share_proportion_l693_69303


namespace NUMINAMATH_CALUDE_line_plane_relationships_l693_69346

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (lies_on : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (on_different_planes : Line → Line → Prop)

-- Define the theorem
theorem line_plane_relationships 
  (l a : Line) (α : Plane)
  (h1 : parallel_line_plane l α)
  (h2 : lies_on a α) :
  perpendicular l a ∨ parallel_lines l a ∨ on_different_planes l a :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l693_69346


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l693_69359

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l693_69359


namespace NUMINAMATH_CALUDE_f_2006_equals_1_l693_69336

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2006_equals_1 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period (fun x ↦ f (3*x + 1)) 3)
  (h_f_1 : f 1 = -1) : 
  f 2006 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_2006_equals_1_l693_69336


namespace NUMINAMATH_CALUDE_fourth_year_area_l693_69315

def initial_area : ℝ := 10000
def increase_rate : ℝ := 0.2

def area_after_n_years (n : ℕ) : ℝ :=
  initial_area * (1 + increase_rate) ^ n

theorem fourth_year_area :
  area_after_n_years 3 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_area_l693_69315


namespace NUMINAMATH_CALUDE_robin_gum_total_l693_69357

/-- Calculate the total number of gum pieces Robin has after his purchases -/
theorem robin_gum_total (initial_packages : ℕ) (initial_pieces_per_package : ℕ)
  (local_packages : ℚ) (local_pieces_per_package : ℕ)
  (foreign_packages : ℕ) (foreign_pieces_per_package : ℕ)
  (exchange_rate : ℚ) (foreign_purchase_dollars : ℕ) :
  initial_packages = 27 →
  initial_pieces_per_package = 18 →
  local_packages = 15.5 →
  local_pieces_per_package = 12 →
  foreign_packages = 8 →
  foreign_pieces_per_package = 25 →
  exchange_rate = 1.2 →
  foreign_purchase_dollars = 50 →
  (initial_packages * initial_pieces_per_package +
   ⌊local_packages⌋ * local_pieces_per_package +
   foreign_packages * foreign_pieces_per_package) = 872 := by
  sorry

#check robin_gum_total

end NUMINAMATH_CALUDE_robin_gum_total_l693_69357


namespace NUMINAMATH_CALUDE_complex_cube_root_identity_l693_69309

theorem complex_cube_root_identity (z : ℂ) (h1 : z^3 + 1 = 0) (h2 : z ≠ -1) :
  (z / (z - 1))^2018 + (1 / (z - 1))^2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_identity_l693_69309


namespace NUMINAMATH_CALUDE_closest_multiple_of_17_to_3513_l693_69390

theorem closest_multiple_of_17_to_3513 :
  ∀ k : ℤ, |3519 - 3513| ≤ |17 * k - 3513| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_17_to_3513_l693_69390


namespace NUMINAMATH_CALUDE_problem_1_l693_69321

theorem problem_1 : (-16) - 25 + (-43) - (-39) = -45 := by sorry

end NUMINAMATH_CALUDE_problem_1_l693_69321


namespace NUMINAMATH_CALUDE_tranquility_essence_l693_69323

/-- Represents the philosophical concepts in the problem --/
structure PhilosophicalConcept where
  opposingAndUnified : Bool  -- The sides of a contradiction are both opposing and unified
  struggleWithinUnity : Bool -- The nature of struggle is embedded within unity
  differencesBasedOnUnity : Bool -- Differences and opposition are based on unity
  motionCharacteristic : Bool -- Motion is the only characteristic of matter

/-- Represents a painting with its elements --/
structure Painting where
  hasWaterfall : Bool
  hasTree : Bool
  hasBirdNest : Bool
  hasSleepingBird : Bool

/-- Defines the essence of tranquility based on philosophical concepts --/
def essenceOfTranquility (p : Painting) (c : PhilosophicalConcept) : Prop :=
  p.hasWaterfall ∧ p.hasTree ∧ p.hasBirdNest ∧ p.hasSleepingBird ∧
  c.opposingAndUnified ∧ c.struggleWithinUnity ∧
  ¬c.differencesBasedOnUnity ∧ ¬c.motionCharacteristic

/-- The theorem to be proved --/
theorem tranquility_essence (p : Painting) (c : PhilosophicalConcept) :
  p.hasWaterfall ∧ p.hasTree ∧ p.hasBirdNest ∧ p.hasSleepingBird →
  c.opposingAndUnified ∧ c.struggleWithinUnity →
  essenceOfTranquility p c := by
  sorry


end NUMINAMATH_CALUDE_tranquility_essence_l693_69323


namespace NUMINAMATH_CALUDE_cookie_cutter_problem_l693_69302

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := sorry

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := 46

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

theorem cookie_cutter_problem :
  num_squares = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookie_cutter_problem_l693_69302


namespace NUMINAMATH_CALUDE_at_least_one_negative_l693_69354

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1)
  (sum_cd : c + d = 1)
  (product_sum : a * c + b * d > 1) :
  ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l693_69354


namespace NUMINAMATH_CALUDE_sam_has_most_pages_l693_69381

/-- Represents a book collection --/
structure Collection where
  pagesPerInch : ℕ
  height : ℕ

/-- Calculates the total number of pages in a collection --/
def totalPages (c : Collection) : ℕ := c.pagesPerInch * c.height

theorem sam_has_most_pages (miles daphne sam : Collection)
  (h_miles : miles = ⟨5, 240⟩)
  (h_daphne : daphne = ⟨50, 25⟩)
  (h_sam : sam = ⟨30, 60⟩) :
  totalPages sam = 1800 ∧ 
  totalPages sam > totalPages miles ∧ 
  totalPages sam > totalPages daphne :=
by sorry

end NUMINAMATH_CALUDE_sam_has_most_pages_l693_69381


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l693_69313

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let firstDescent := initialHeight
  let firstAscent := initialHeight * reboundFactor
  let secondDescent := firstAscent
  let secondAscent := firstAscent * reboundFactor
  let thirdDescent := secondAscent
  firstDescent + firstAscent + secondDescent + secondAscent + thirdDescent

/-- The theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistance 90 0.5 2 = 225 := by
  sorry

#eval totalDistance 90 0.5 2

end NUMINAMATH_CALUDE_ball_bounce_distance_l693_69313


namespace NUMINAMATH_CALUDE_intersection_of_sets_l693_69344

theorem intersection_of_sets : 
  let P : Set ℤ := {-3, -2, 0, 2}
  let Q : Set ℤ := {-1, -2, -3, 0, 1}
  P ∩ Q = {-3, -2, 0} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l693_69344


namespace NUMINAMATH_CALUDE_sum_product_over_sum_squares_is_zero_l693_69384

theorem sum_product_over_sum_squares_is_zero 
  (x y z : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) 
  (hsum : x + y + z = 1) : 
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_product_over_sum_squares_is_zero_l693_69384


namespace NUMINAMATH_CALUDE_red_apples_count_l693_69326

/-- The number of apples produced by each tree -/
def applesPerTree : ℕ := 20

/-- The percentage of red apples on the first tree -/
def firstTreeRedPercentage : ℚ := 40 / 100

/-- The percentage of red apples on the second tree -/
def secondTreeRedPercentage : ℚ := 50 / 100

/-- The total number of red apples from both trees -/
def totalRedApples : ℕ := 18

theorem red_apples_count :
  ⌊(firstTreeRedPercentage * applesPerTree : ℚ)⌋ +
  ⌊(secondTreeRedPercentage * applesPerTree : ℚ)⌋ = totalRedApples :=
sorry

end NUMINAMATH_CALUDE_red_apples_count_l693_69326
