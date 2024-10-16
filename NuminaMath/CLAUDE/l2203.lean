import Mathlib

namespace NUMINAMATH_CALUDE_marble_ratio_l2203_220396

/-- Proves the ratio of Michael's marbles to Wolfgang and Ludo's combined marbles -/
theorem marble_ratio :
  let wolfgang_marbles : ℕ := 16
  let ludo_marbles : ℕ := wolfgang_marbles + wolfgang_marbles / 4
  let total_marbles : ℕ := 20 * 3
  let michael_marbles : ℕ := total_marbles - wolfgang_marbles - ludo_marbles
  let wolfgang_ludo_marbles : ℕ := wolfgang_marbles + ludo_marbles
  (michael_marbles : ℚ) / wolfgang_ludo_marbles = 2 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_marble_ratio_l2203_220396


namespace NUMINAMATH_CALUDE_square_of_negative_product_l2203_220368

theorem square_of_negative_product (a b : ℝ) : (-3 * a * b^2)^2 = 9 * a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l2203_220368


namespace NUMINAMATH_CALUDE_sphere_radii_ratio_l2203_220321

theorem sphere_radii_ratio (V1 V2 r1 r2 : ℝ) :
  V1 = 450 * Real.pi →
  V2 = 36 * Real.pi →
  V2 / V1 = (r2 / r1) ^ 3 →
  r2 / r1 = Real.rpow 2 (1/3) / 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_radii_ratio_l2203_220321


namespace NUMINAMATH_CALUDE_diesel_in_container_l2203_220354

/-- Represents the ratio of diesel to water in the final mixture -/
def diesel_water_ratio : ℚ := 3 / 5

/-- Amount of petrol in the container -/
def petrol_amount : ℚ := 4

/-- Amount of water added to the container -/
def water_added : ℚ := 2.666666666666667

/-- Calculates the amount of diesel in the container -/
def diesel_amount (ratio : ℚ) (petrol : ℚ) (water : ℚ) : ℚ :=
  ratio * (petrol + water)

theorem diesel_in_container :
  diesel_amount diesel_water_ratio petrol_amount water_added = 4 := by
  sorry

end NUMINAMATH_CALUDE_diesel_in_container_l2203_220354


namespace NUMINAMATH_CALUDE_inequality_proof_l2203_220360

theorem inequality_proof (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d)) + (c / (b + a)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2203_220360


namespace NUMINAMATH_CALUDE_public_foundation_share_l2203_220356

/-- Represents the distribution of charitable funds by a private company. -/
structure CharityFunds where
  X : ℝ  -- Total amount raised
  Y : ℝ  -- Percentage donated to public foundation
  Z : ℕ  -- Number of organizations in public foundation
  W : ℕ  -- Number of local non-profit groups
  A : ℝ  -- Amount received by each local non-profit group
  B : ℝ  -- Amount received by special project
  h1 : Y > 0 ∧ Y ≤ 100  -- Ensure Y is a valid percentage
  h2 : Z > 0  -- Ensure there's at least one organization in the public foundation
  h3 : W > 0  -- Ensure there's at least one local non-profit group
  h4 : X > 0  -- Ensure a positive amount is raised
  h5 : B = (1/3) * X * (1 - Y/100)  -- Amount received by special project
  h6 : A = (2/3) * X * (1 - Y/100) / W  -- Amount received by each local non-profit group

/-- Theorem stating the amount received by each organization in the public foundation. -/
theorem public_foundation_share (cf : CharityFunds) :
  (cf.Y / 100) * cf.X / cf.Z = (cf.Y / 100) * cf.X / cf.Z :=
by sorry

end NUMINAMATH_CALUDE_public_foundation_share_l2203_220356


namespace NUMINAMATH_CALUDE_can_collection_ratio_l2203_220332

theorem can_collection_ratio : 
  ∀ (solomon juwan levi : ℕ),
  solomon = 3 * juwan →
  solomon = 66 →
  solomon + juwan + levi = 99 →
  levi * 2 = juwan :=
by
  sorry

end NUMINAMATH_CALUDE_can_collection_ratio_l2203_220332


namespace NUMINAMATH_CALUDE_cream_cheese_cost_l2203_220357

/-- Cost of items for staff meetings -/
theorem cream_cheese_cost (bagel_cost cream_cheese_cost : ℝ) : 
  2 * bagel_cost + 3 * cream_cheese_cost = 12 →
  4 * bagel_cost + 2 * cream_cheese_cost = 14 →
  cream_cheese_cost = 2.5 := by
sorry

end NUMINAMATH_CALUDE_cream_cheese_cost_l2203_220357


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2203_220311

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (x - y)^2 - (x + y)*(x - y) = -2*x*y + 2*y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (12*a^2*b - 6*a*b^2) / (-3*a*b) = -4*a + 2*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2203_220311


namespace NUMINAMATH_CALUDE_simplify_fraction_l2203_220317

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2203_220317


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_prime_factor_constraint_l2203_220398

theorem arithmetic_progression_with_prime_factor_constraint :
  ∀ (a b c : ℕ), 
    0 < a → a < b → b < c →
    b - a = c - b →
    (∀ p : ℕ, Prime p → p > 3 → (p ∣ a ∨ p ∣ b ∨ p ∣ c) → False) →
    ∃ (k m n : ℕ), 
      (a = k ∧ b = 2*k ∧ c = 3*k) ∨
      (a = 2*k ∧ b = 3*k ∧ c = 4*k) ∨
      (a = 2*k ∧ b = 9*k ∧ c = 16*k) ∧
      k = 2^m * 3^n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_prime_factor_constraint_l2203_220398


namespace NUMINAMATH_CALUDE_unique_solution_system_l2203_220386

theorem unique_solution_system (x y : ℝ) : 
  (x - 2*y = 1 ∧ 3*x + 4*y = 23) ↔ (x = 5 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2203_220386


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2203_220335

/-- Given a geometric sequence with common ratio q > 0 and T_n as the product of the first n terms,
    if T_7 > T_6 > T_8, then 0 < q < 1 and T_13 > 1 > T_14 -/
theorem geometric_sequence_properties (q : ℝ) (T : ℕ → ℝ) 
  (h_q_pos : q > 0)
  (h_T : ∀ n : ℕ, T n = (T 1) * q^(n * (n - 1) / 2))
  (h_ineq : T 7 > T 6 ∧ T 6 > T 8) :
  (0 < q ∧ q < 1) ∧ (T 13 > 1 ∧ 1 > T 14) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2203_220335


namespace NUMINAMATH_CALUDE_vault_code_thickness_l2203_220303

/-- Thickness of an Alpha card in millimeters -/
def alpha_thickness : ℚ := 1.65

/-- Thickness of a Beta card in millimeters -/
def beta_thickness : ℚ := 2.05

/-- Thickness of a Gamma card in millimeters -/
def gamma_thickness : ℚ := 1.25

/-- Thickness of a Delta card in millimeters -/
def delta_thickness : ℚ := 1.85

/-- Total thickness of the stack in millimeters -/
def total_thickness : ℚ := 15.6

/-- The number of cards in the stack -/
def num_cards : ℕ := 8

theorem vault_code_thickness :
  num_cards * delta_thickness = total_thickness ∧
  ∀ (a b c d : ℕ), 
    a * alpha_thickness + b * beta_thickness + c * gamma_thickness + d * delta_thickness = total_thickness →
    a = 0 ∧ b = 0 ∧ c = 0 ∧ d = num_cards :=
by sorry

end NUMINAMATH_CALUDE_vault_code_thickness_l2203_220303


namespace NUMINAMATH_CALUDE_function_fixed_point_l2203_220352

theorem function_fixed_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_fixed_point_l2203_220352


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l2203_220384

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Theorem statement
theorem f_strictly_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 3 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l2203_220384


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_three_l2203_220325

theorem fraction_zero_implies_x_three (x : ℝ) :
  (x - 3) / (2 * x + 5) = 0 ∧ 2 * x + 5 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_three_l2203_220325


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l2203_220387

theorem rectangle_length_calculation (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 24 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_calculation_l2203_220387


namespace NUMINAMATH_CALUDE_S_tiles_integers_not_naturals_l2203_220374

def S : Set ℤ := {1, 3, 4, 6}

def tiles_integers (S : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃ s ∈ S, ∃ k : ℤ, n = s + 4 * k

def tiles_naturals (S : Set ℤ) : Prop :=
  ∀ n : ℕ, ∃ s ∈ S, ∃ k : ℤ, (n : ℤ) = s + 4 * k

theorem S_tiles_integers_not_naturals :
  tiles_integers S ∧ ¬tiles_naturals S := by sorry

end NUMINAMATH_CALUDE_S_tiles_integers_not_naturals_l2203_220374


namespace NUMINAMATH_CALUDE_range_of_a_l2203_220380

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x ≤ a^2 - a - 3

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*a)^x > (5 - 2*a)^y

theorem range_of_a : 
  (∀ a : ℝ, (p a ∨ q a)) ∧ (¬∃ a : ℝ, p a ∧ q a) → 
  {a : ℝ | a = 2 ∨ a ≥ 5/2} = {a : ℝ | ∃ x : ℝ, p x ∨ q x} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2203_220380


namespace NUMINAMATH_CALUDE_fraction_simplification_l2203_220347

theorem fraction_simplification :
  5 / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2203_220347


namespace NUMINAMATH_CALUDE_percentage_relation_l2203_220334

theorem percentage_relation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x / 100 * y = 12) (h2 : y / 100 * x = 9) : x = 400 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2203_220334


namespace NUMINAMATH_CALUDE_unique_n_reaches_two_l2203_220333

def g (n : ℤ) : ℤ := 
  if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

def iterateG (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k+1 => g (iterateG n k)

theorem unique_n_reaches_two :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 100 ∧ ∃ k : ℕ, iterateG n k = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_n_reaches_two_l2203_220333


namespace NUMINAMATH_CALUDE_train_length_calculation_l2203_220316

theorem train_length_calculation (platform_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) :
  platform_length = 400 →
  platform_crossing_time = 42 →
  pole_crossing_time = 18 →
  ∃ train_length : ℝ,
    train_length = 300 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2203_220316


namespace NUMINAMATH_CALUDE_largest_number_l2203_220366

theorem largest_number (S : Set ℝ) (hS : S = {-1, 0, 1, 1/3}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l2203_220366


namespace NUMINAMATH_CALUDE_no_arithmetic_mean_among_fractions_l2203_220302

theorem no_arithmetic_mean_among_fractions : 
  let a := 8 / 13
  let b := 11 / 17
  let c := 5 / 8
  ¬(a = (b + c) / 2 ∨ b = (a + c) / 2 ∨ c = (a + b) / 2) := by
sorry

end NUMINAMATH_CALUDE_no_arithmetic_mean_among_fractions_l2203_220302


namespace NUMINAMATH_CALUDE_inequality_proof_l2203_220306

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2203_220306


namespace NUMINAMATH_CALUDE_parabola_directrix_distance_l2203_220365

/-- Proves that for a parabola y = ax² (a > 0) with a point M(3, 2),
    if the distance from M to the directrix is 4, then a = 1/8 -/
theorem parabola_directrix_distance (a : ℝ) : 
  a > 0 → 
  (let M : ℝ × ℝ := (3, 2)
   let directrix_y : ℝ := -1 / (4 * a)
   let distance_to_directrix : ℝ := |M.2 - directrix_y|
   distance_to_directrix = 4) →
  a = 1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_distance_l2203_220365


namespace NUMINAMATH_CALUDE_perimeter_of_figure_C_l2203_220327

/-- Represents the dimensions of a rectangle in terms of small rectangles -/
structure RectangleDimension where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle given its dimensions and the size of small rectangles -/
def calculatePerimeter (dim : RectangleDimension) (x y : ℝ) : ℝ :=
  2 * (dim.width * x + dim.height * y)

theorem perimeter_of_figure_C (x y : ℝ) : 
  calculatePerimeter ⟨6, 1⟩ x y = 56 →
  calculatePerimeter ⟨4, 3⟩ x y = 56 →
  calculatePerimeter ⟨2, 3⟩ x y = 40 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_C_l2203_220327


namespace NUMINAMATH_CALUDE_zeros_when_b_neg_one_inequality_condition_max_value_on_interval_l2203_220348

-- Define the function f
def f (a b x : ℝ) : ℝ := x * |x - a| + b * x

-- Theorem 1
theorem zeros_when_b_neg_one (a : ℝ) :
  (∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ f a (-1) z₁ = 0 ∧ f a (-1) z₂ = 0) ↔ (a = 1 ∨ a = -1) :=
sorry

-- Theorem 2
theorem inequality_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a 1 x / x ≤ 2 * Real.sqrt (x + 1)) ↔ 
  (0 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

-- Define the piecewise function g
noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ 4 * Real.sqrt 3 - 5 then 6 - 2*a
  else if a < 3 then (a + 1)^2 / 4
  else 2*a - 2

-- Theorem 3
theorem max_value_on_interval (a : ℝ) (h : a > 0) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc 0 2, f a 1 x ≤ m ∧ ∃ y ∈ Set.Icc 0 2, f a 1 y = m) ∧
  (∀ (m : ℝ), (∀ x ∈ Set.Icc 0 2, f a 1 x ≤ m ∧ ∃ y ∈ Set.Icc 0 2, f a 1 y = m) → m = g a) :=
sorry

end NUMINAMATH_CALUDE_zeros_when_b_neg_one_inequality_condition_max_value_on_interval_l2203_220348


namespace NUMINAMATH_CALUDE_community_avg_age_l2203_220353

-- Define the ratio of women to men
def women_to_men_ratio : ℚ := 7 / 5

-- Define the average age of women
def avg_age_women : ℝ := 30

-- Define the average age of men
def avg_age_men : ℝ := 35

-- Theorem statement
theorem community_avg_age :
  let total_population := women_to_men_ratio + 1
  let weighted_age_sum := women_to_men_ratio * avg_age_women + avg_age_men
  weighted_age_sum / total_population = 385 / 12 :=
by sorry

end NUMINAMATH_CALUDE_community_avg_age_l2203_220353


namespace NUMINAMATH_CALUDE_max_boats_in_river_l2203_220315

theorem max_boats_in_river (river_width : ℝ) (boat_width : ℝ) (min_space : ℝ) :
  river_width = 42 →
  boat_width = 3 →
  min_space = 2 →
  ⌊(river_width - 2 * min_space) / (boat_width + 2 * min_space)⌋ = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_boats_in_river_l2203_220315


namespace NUMINAMATH_CALUDE_late_start_time_l2203_220355

-- Define the usual time to reach the office
def usual_time : ℝ := 60

-- Define the slower speed factor
def slower_speed_factor : ℝ := 0.75

-- Define the late arrival time
def late_arrival : ℝ := 50

-- Theorem statement
theorem late_start_time (actual_journey_time : ℝ) :
  actual_journey_time = usual_time / slower_speed_factor + late_arrival →
  actual_journey_time - (usual_time / slower_speed_factor) = 30 := by
  sorry

end NUMINAMATH_CALUDE_late_start_time_l2203_220355


namespace NUMINAMATH_CALUDE_right_triangle_area_l2203_220300

theorem right_triangle_area (h : ℝ) (h_positive : h > 0) :
  let a := h * Real.sqrt 2
  let b := h * Real.sqrt 2
  let c := 2 * h * Real.sqrt 2
  h = 4 →
  (1 / 2 : ℝ) * c * h = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2203_220300


namespace NUMINAMATH_CALUDE_star_equation_solution_l2203_220377

def star (a b : ℝ) : ℝ := a^2 * b + 2 * b - a

theorem star_equation_solution :
  ∀ x : ℝ, star 7 x = 85 → x = 92 / 51 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2203_220377


namespace NUMINAMATH_CALUDE_floor_equality_iff_interval_l2203_220305

theorem floor_equality_iff_interval (x : ℝ) :
  ⌊⌊2 * x⌋ - (1/2 : ℝ)⌋ = ⌊x + 3⌋ ↔ 3 ≤ x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_floor_equality_iff_interval_l2203_220305


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2203_220308

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 + 3 * a 8 + a 15 = 60) : 
  2 * a 9 - a 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2203_220308


namespace NUMINAMATH_CALUDE_g_evaluation_l2203_220399

def g (x : ℝ) : ℝ := 3 * x^3 + 5 * x^2 - 6 * x + 4

theorem g_evaluation : 3 * g 2 - 2 * g (-1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l2203_220399


namespace NUMINAMATH_CALUDE_square_side_length_average_l2203_220394

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l2203_220394


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l2203_220358

theorem quadratic_root_implies_k (k : ℝ) : 
  (1 : ℝ)^2 + k*(1 : ℝ) - 3 = 0 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l2203_220358


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2203_220381

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number 188 million -/
def number : ℝ := 188000000

/-- The scientific notation representation of 188 million -/
def scientificForm : ScientificNotation :=
  { coefficient := 1.88
    exponent := 8
    coeff_range := by sorry }

theorem scientific_notation_correct :
  number = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2203_220381


namespace NUMINAMATH_CALUDE_two_million_six_hundred_thousand_scientific_notation_l2203_220314

/-- Scientific notation representation -/
def scientific_notation (n : ℝ) (x : ℝ) (p : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ n = x * (10 : ℝ) ^ p

/-- Theorem: 2,600,000 in scientific notation -/
theorem two_million_six_hundred_thousand_scientific_notation :
  ∃ (x : ℝ) (p : ℤ), scientific_notation 2600000 x p ∧ x = 2.6 ∧ p = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_million_six_hundred_thousand_scientific_notation_l2203_220314


namespace NUMINAMATH_CALUDE_consecutive_integers_product_255_l2203_220341

theorem consecutive_integers_product_255 (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 255) :
  x + (x + 1) = 31 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_255_l2203_220341


namespace NUMINAMATH_CALUDE_division_problem_l2203_220369

/-- Given the conditions of the division problem, prove the values of the divisors -/
theorem division_problem (D₁ D₂ : ℕ) : 
  1526 = 34 * D₁ + 18 → 
  34 * D₂ + 52 = 421 → 
  D₁ = 44 ∧ D₂ = 11 := by
  sorry

#check division_problem

end NUMINAMATH_CALUDE_division_problem_l2203_220369


namespace NUMINAMATH_CALUDE_max_distance_to_upper_vertex_l2203_220322

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def upper_vertex (B : ℝ × ℝ) : Prop :=
  B.1 = 0 ∧ B.2 = 1 ∧ ellipse B.1 B.2

theorem max_distance_to_upper_vertex :
  ∃ (B : ℝ × ℝ), upper_vertex B ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_upper_vertex_l2203_220322


namespace NUMINAMATH_CALUDE_shopping_discount_theorem_l2203_220361

def shoe_price : ℝ := 60
def dress_price : ℝ := 120
def accessory_price : ℝ := 25

def shoe_discount : ℝ := 0.3
def dress_discount : ℝ := 0.15
def accessory_discount : ℝ := 0.5
def additional_discount : ℝ := 0.1

def shoe_quantity : ℕ := 3
def dress_quantity : ℕ := 2
def accessory_quantity : ℕ := 3

def discount_threshold : ℝ := 200

theorem shopping_discount_theorem :
  let total_before_discount := shoe_price * shoe_quantity + dress_price * dress_quantity + accessory_price * accessory_quantity
  let shoe_discounted := shoe_price * shoe_quantity * (1 - shoe_discount)
  let dress_discounted := dress_price * dress_quantity * (1 - dress_discount)
  let accessory_discounted := accessory_price * accessory_quantity * (1 - accessory_discount)
  let total_after_category_discounts := shoe_discounted + dress_discounted + accessory_discounted
  let final_total := 
    if total_before_discount > discount_threshold
    then total_after_category_discounts * (1 - additional_discount)
    else total_after_category_discounts
  final_total = 330.75 := by
  sorry

end NUMINAMATH_CALUDE_shopping_discount_theorem_l2203_220361


namespace NUMINAMATH_CALUDE_binomial_10_3_l2203_220375

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2203_220375


namespace NUMINAMATH_CALUDE_crayons_count_l2203_220388

/-- The number of crayons in a box with specific color relationships -/
def total_crayons (blue : ℕ) : ℕ :=
  let red := 4 * blue
  let green := 2 * red
  let yellow := green / 2
  blue + red + green + yellow

/-- Theorem stating that the total number of crayons is 51 when there are 3 blue crayons -/
theorem crayons_count : total_crayons 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l2203_220388


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l2203_220313

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 16) / (4 - x) = 0 ∧ x ≠ 4 → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l2203_220313


namespace NUMINAMATH_CALUDE_cost_increase_l2203_220330

theorem cost_increase (t b : ℝ) : 
  let original_cost := t * b^5
  let new_cost := (3*t) * (2*b)^5
  (new_cost / original_cost) * 100 = 9600 := by
sorry

end NUMINAMATH_CALUDE_cost_increase_l2203_220330


namespace NUMINAMATH_CALUDE_guthrie_market_souvenirs_cost_l2203_220379

/-- The total cost of souvenirs distributed at Guthrie Market's Grand Opening -/
theorem guthrie_market_souvenirs_cost :
  let type1_cost : ℚ := 20 / 100  -- 20 cents in dollars
  let type2_cost : ℚ := 25 / 100  -- 25 cents in dollars
  let total_souvenirs : ℕ := 1000
  let type2_quantity : ℕ := 400
  let type1_quantity : ℕ := total_souvenirs - type2_quantity
  let total_cost : ℚ := type1_quantity * type1_cost + type2_quantity * type2_cost
  total_cost = 220 / 100  -- $220 in decimal form
:= by sorry

end NUMINAMATH_CALUDE_guthrie_market_souvenirs_cost_l2203_220379


namespace NUMINAMATH_CALUDE_half_of_five_bananas_worth_l2203_220319

-- Define the worth of bananas in terms of oranges
def banana_orange_ratio : ℚ := 8 / (2/3 * 10)

-- Theorem statement
theorem half_of_five_bananas_worth (banana_orange_ratio : ℚ) :
  banana_orange_ratio = 8 / (2/3 * 10) →
  (1/2 * 5) * banana_orange_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_half_of_five_bananas_worth_l2203_220319


namespace NUMINAMATH_CALUDE_expected_different_faces_formula_l2203_220350

/-- The number of sides on a fair die -/
def numSides : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def probNotAppear : ℚ := (numSides - 1) / numSides

/-- The expected number of different faces that appear when rolling a fair die -/
def expectedDifferentFaces : ℚ := numSides * (1 - probNotAppear ^ numRolls)

/-- Theorem stating the expected number of different faces when rolling a fair die -/
theorem expected_different_faces_formula :
  expectedDifferentFaces = (numSides^numRolls - (numSides - 1)^numRolls) / numSides^(numRolls - 1) :=
sorry

end NUMINAMATH_CALUDE_expected_different_faces_formula_l2203_220350


namespace NUMINAMATH_CALUDE_sum_to_k_perfect_cube_l2203_220378

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem sum_to_k_perfect_cube :
  ∀ k : ℕ, k > 0 → k < 200 →
    (is_perfect_cube (sum_to_k k) ↔ k = 1 ∨ k = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_perfect_cube_l2203_220378


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2203_220324

-- Define the triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the perpendicular bisector of BC
def perpendicular_bisector_BC (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Define the area of triangle ABC
def area_ABC : ℝ := 7

-- Theorem statement
theorem triangle_ABC_properties :
  (perpendicular_bisector_BC (A.1 + B.1 + C.1) (A.2 + B.2 + C.2)) ∧
  (area_ABC = 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2203_220324


namespace NUMINAMATH_CALUDE_equation_solution_l2203_220392

theorem equation_solution :
  ∃ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + n / (n + 2) = 6 - 3 / (n + 2)) ∧ (n = -6/5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2203_220392


namespace NUMINAMATH_CALUDE_lucky_number_properties_l2203_220349

def is_lucky_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  let ab := n / 100
  let cd := n % 100
  ab ≠ cd ∧ cd % ab = 0 ∧ n % cd = 0

def count_lucky_numbers : ℕ := sorry

def largest_odd_lucky_number : ℕ := sorry

theorem lucky_number_properties :
  count_lucky_numbers = 65 ∧
  largest_odd_lucky_number = 1995 ∧
  is_lucky_number largest_odd_lucky_number ∧
  (∀ n, is_lucky_number n → n % 2 = 1 → n ≤ largest_odd_lucky_number) := by sorry

end NUMINAMATH_CALUDE_lucky_number_properties_l2203_220349


namespace NUMINAMATH_CALUDE_power_function_through_point_l2203_220301

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2203_220301


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2203_220383

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2203_220383


namespace NUMINAMATH_CALUDE_committee_selection_l2203_220371

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 35) : Nat.choose n 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2203_220371


namespace NUMINAMATH_CALUDE_odd_digits_365_base5_l2203_220309

/-- Counts the number of odd digits in the base-5 representation of a natural number -/
def countOddDigitsBase5 (n : ℕ) : ℕ :=
  sorry

theorem odd_digits_365_base5 : countOddDigitsBase5 365 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_365_base5_l2203_220309


namespace NUMINAMATH_CALUDE_additional_men_count_l2203_220382

theorem additional_men_count (initial_men : ℕ) (initial_days : ℕ) (final_days : ℕ) :
  initial_men = 600 →
  initial_days = 20 →
  final_days = 15 →
  ∃ (additional_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    additional_men = 200 := by
  sorry

end NUMINAMATH_CALUDE_additional_men_count_l2203_220382


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2203_220346

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = ax + 1 -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x + 3 -/
def slope2 (a : ℝ) : ℝ := a - 2

/-- The theorem stating that a = 1 is the necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition (a : ℝ) : 
  perpendicular (slope1 a) (slope2 a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2203_220346


namespace NUMINAMATH_CALUDE_hyperbola_iff_ab_neg_l2203_220323

/-- A curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Definition of a hyperbola -/
def is_hyperbola (c : Curve) : Prop := sorry

/-- The specific curve ax^2 + by^2 = 1 -/
def quadratic_curve (a b : ℝ) : Curve where
  equation := fun x y => a * x^2 + b * y^2 = 1

/-- Theorem stating that ab < 0 is both necessary and sufficient for the curve to be a hyperbola -/
theorem hyperbola_iff_ab_neg (a b : ℝ) :
  is_hyperbola (quadratic_curve a b) ↔ a * b < 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_ab_neg_l2203_220323


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_at_least_two_in_first_l2203_220328

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes -/
def total_distributions (n : ℕ) : ℕ := 2^n

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes
    where the first box contains exactly k balls -/
def distributions_with_k_in_first_box (n k : ℕ) : ℕ := n.choose k

theorem seven_balls_two_boxes_at_least_two_in_first : 
  total_distributions 7 - (distributions_with_k_in_first_box 7 0 + distributions_with_k_in_first_box 7 1) = 120 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_at_least_two_in_first_l2203_220328


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2203_220397

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 1 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | b * x^2 + a * x + c < 0} = Set.Ioo (-2/3) 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2203_220397


namespace NUMINAMATH_CALUDE_y_intersection_is_six_l2203_220376

/-- The quadratic function f(x) = -2(x-1)(x+3) -/
def f (x : ℝ) : ℝ := -2 * (x - 1) * (x + 3)

/-- The y-coordinate of the intersection point with the y-axis is 6 -/
theorem y_intersection_is_six : f 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_intersection_is_six_l2203_220376


namespace NUMINAMATH_CALUDE_remainder_theorem_l2203_220340

-- Define the polynomial p(x)
variable (p : ℝ → ℝ)

-- Define the conditions
axiom p_div_x_minus_2 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x + 2
axiom p_div_x_minus_3 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * q x + 6

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * (x - 3) * q x + (4 * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2203_220340


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2203_220364

/-- Given a principal amount and a simple interest rate,
    if the amount after 5 years is 7/6 of the principal,
    then the rate is 1/30 -/
theorem simple_interest_rate (P R : ℚ) (P_pos : 0 < P) :
  P + P * R * 5 = (7 / 6) * P →
  R = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2203_220364


namespace NUMINAMATH_CALUDE_painted_area_is_33_l2203_220320

/-- Represents the arrangement of cubes -/
structure CubeArrangement where
  width : Nat
  length : Nat
  height : Nat
  total_cubes : Nat

/-- Calculates the total painted area for a given cube arrangement -/
def painted_area (arr : CubeArrangement) : Nat :=
  let top_area := arr.width * arr.length
  let side_area := 2 * (arr.width * arr.height + arr.length * arr.height)
  top_area + side_area

/-- The specific arrangement described in the problem -/
def problem_arrangement : CubeArrangement :=
  { width := 3
  , length := 3
  , height := 1
  , total_cubes := 14 }

/-- Theorem stating that the painted area for the given arrangement is 33 square meters -/
theorem painted_area_is_33 : painted_area problem_arrangement = 33 := by
  sorry

end NUMINAMATH_CALUDE_painted_area_is_33_l2203_220320


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l2203_220342

/-- Given a set of worksheets with the following properties:
    - There are 9 worksheets in total
    - 5 worksheets have been graded
    - 16 problems remain to be graded
    This theorem proves that there are 4 problems on each worksheet. -/
theorem problems_per_worksheet (total_worksheets : Nat) (graded_worksheets : Nat) (remaining_problems : Nat)
    (h1 : total_worksheets = 9)
    (h2 : graded_worksheets = 5)
    (h3 : remaining_problems = 16) :
    (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_worksheet_l2203_220342


namespace NUMINAMATH_CALUDE_second_replaced_man_age_l2203_220343

theorem second_replaced_man_age 
  (n : ℕ) 
  (age_increase : ℝ) 
  (first_replaced_age : ℕ) 
  (new_men_avg_age : ℝ) 
  (h1 : n = 15)
  (h2 : age_increase = 2)
  (h3 : first_replaced_age = 21)
  (h4 : new_men_avg_age = 37) :
  ∃ (second_replaced_age : ℕ),
    (n : ℝ) * age_increase = 
      2 * new_men_avg_age - (first_replaced_age : ℝ) - (second_replaced_age : ℝ) ∧
    second_replaced_age = 23 :=
by sorry

end NUMINAMATH_CALUDE_second_replaced_man_age_l2203_220343


namespace NUMINAMATH_CALUDE_basketball_volleyball_cost_total_cost_proof_l2203_220336

/-- The cost of buying basketballs and volleyballs -/
theorem basketball_volleyball_cost (m n : ℝ) : ℝ :=
  3 * m + 7 * n

/-- Proof that the total cost of 3 basketballs and 7 volleyballs is 3m + 7n yuan -/
theorem total_cost_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  basketball_volleyball_cost m n = 3 * m + 7 * n :=
by sorry

end NUMINAMATH_CALUDE_basketball_volleyball_cost_total_cost_proof_l2203_220336


namespace NUMINAMATH_CALUDE_unsold_books_percentage_l2203_220389

-- Define the initial stock and daily sales
def initial_stock : ℕ := 620
def daily_sales : List ℕ := [50, 82, 60, 48, 40]

-- Define the theorem
theorem unsold_books_percentage :
  let total_sold := daily_sales.sum
  let unsold := initial_stock - total_sold
  let percentage_unsold := (unsold : ℚ) / (initial_stock : ℚ) * 100
  ∃ ε > 0, abs (percentage_unsold - 54.84) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_unsold_books_percentage_l2203_220389


namespace NUMINAMATH_CALUDE_circle_trajectory_and_intersection_l2203_220359

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the trajectory of the center of circle P
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the condition for P being externally tangent to C1 and internally tangent to C2
def tangency_condition (px py : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), C1 x y → (x - px)^2 + (y - py)^2 ≥ r^2) ∧
  (∃ (x y : ℝ), C1 x y ∧ (x - px)^2 + (y - py)^2 = r^2) ∧
  (∀ (x y : ℝ), C2 x y → (x - px)^2 + (y - py)^2 ≤ (3 + r)^2) ∧
  (∃ (x y : ℝ), C2 x y ∧ (x - px)^2 + (y - py)^2 = (3 + r)^2)

-- Theorem statement
theorem circle_trajectory_and_intersection :
  (∀ (px py : ℝ), tangency_condition px py → trajectory px py) ∧
  (∀ (a b : ℝ), trajectory a 0 ∧ trajectory b 0 → 3 ≤ |a - b| ∧ |a - b| ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_intersection_l2203_220359


namespace NUMINAMATH_CALUDE_exponent_calculation_l2203_220385

theorem exponent_calculation : (8^5 / 8^2) * 4^4 = 2^17 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l2203_220385


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l2203_220304

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (earnings_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 60)
  (h3 : fuel_efficiency = 30)
  (h4 : earnings_rate = 0.75)
  (h5 : gasoline_cost = 3) :
  let distance := travel_time * speed
  let fuel_used := distance / fuel_efficiency
  let earnings := distance * earnings_rate
  let fuel_expense := fuel_used * gasoline_cost
  let net_earnings := earnings - fuel_expense
  net_earnings / travel_time = 39 := by
sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l2203_220304


namespace NUMINAMATH_CALUDE_hexagon_area_ratio_l2203_220326

-- Define the regular hexagon
def RegularHexagon (a : ℝ) : Set (ℝ × ℝ) := sorry

-- Define points on the sides of the hexagon
def PointOnSide (hexagon : Set (ℝ × ℝ)) (side : Set (ℝ × ℝ)) : (ℝ × ℝ) := sorry

-- Define parallel lines with specific spacing ratio
def ParallelLinesWithRatio (l1 l2 l3 l4 : Set (ℝ × ℝ)) (ratio : ℝ × ℝ × ℝ) : Prop := sorry

-- Define area of a polygon
def AreaOfPolygon (polygon : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_area_ratio 
  (a : ℝ) 
  (ABCDEF : Set (ℝ × ℝ))
  (G H I J : ℝ × ℝ)
  (BC CD EF FA : Set (ℝ × ℝ)) :
  ABCDEF = RegularHexagon a →
  G = PointOnSide ABCDEF BC →
  H = PointOnSide ABCDEF CD →
  I = PointOnSide ABCDEF EF →
  J = PointOnSide ABCDEF FA →
  ParallelLinesWithRatio AB GJ IH ED (1, 2, 1) →
  (AreaOfPolygon {A, G, I, H, J, F} / AreaOfPolygon ABCDEF) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_ratio_l2203_220326


namespace NUMINAMATH_CALUDE_abs_sum_nonzero_iff_either_nonzero_l2203_220393

theorem abs_sum_nonzero_iff_either_nonzero (x y : ℝ) :
  (abs x + abs y ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_nonzero_iff_either_nonzero_l2203_220393


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2203_220351

-- Define the repeating decimal 4.666...
def repeating_decimal : ℚ :=
  4 + (2 / 3)

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2203_220351


namespace NUMINAMATH_CALUDE_battle_station_staffing_l2203_220344

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def permutations (n k : ℕ) : ℕ := 
  factorial n / factorial (n - k)

theorem battle_station_staffing :
  permutations 15 5 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l2203_220344


namespace NUMINAMATH_CALUDE_quadratic_roots_positive_implies_a_zero_l2203_220363

theorem quadratic_roots_positive_implies_a_zero 
  (a b c : ℝ) 
  (h : ∀ (p : ℝ), p > 0 → ∀ (x : ℝ), a * x^2 + b * x + c + p = 0 → x > 0) :
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_positive_implies_a_zero_l2203_220363


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2203_220390

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_cond : a 1 * a 5 = 4) :
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 ∨ a 1 * a 2 * a 3 * a 4 * a 5 = -32 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2203_220390


namespace NUMINAMATH_CALUDE_mom_in_middle_l2203_220345

-- Define the people in the lineup
inductive Person : Type
  | Dad : Person
  | Mom : Person
  | Brother : Person
  | Sister : Person
  | Me : Person

-- Define the concept of being next to someone in the lineup
def next_to (p1 p2 : Person) : Prop := sorry

-- Define the concept of being in the middle
def in_middle (p : Person) : Prop := sorry

-- State the theorem
theorem mom_in_middle :
  -- Conditions
  (next_to Person.Me Person.Dad) →
  (next_to Person.Me Person.Mom) →
  (next_to Person.Sister Person.Mom) →
  (next_to Person.Sister Person.Brother) →
  -- Conclusion
  in_middle Person.Mom := by sorry

end NUMINAMATH_CALUDE_mom_in_middle_l2203_220345


namespace NUMINAMATH_CALUDE_mod_equivalence_2023_l2203_220338

theorem mod_equivalence_2023 : ∃! n : ℕ, n ≤ 6 ∧ n ≡ -2023 [ZMOD 7] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2023_l2203_220338


namespace NUMINAMATH_CALUDE_first_triangle_isosceles_l2203_220339

theorem first_triangle_isosceles (α β γ δ ε : ℝ) :
  α + β + γ = 180 →
  α + β = δ →
  β + γ = ε →
  0 < α ∧ 0 < β ∧ 0 < γ →
  0 < δ ∧ 0 < ε →
  ∃ (θ : ℝ), (α = θ ∧ γ = θ) ∨ (α = θ ∧ β = θ) ∨ (β = θ ∧ γ = θ) :=
by sorry

end NUMINAMATH_CALUDE_first_triangle_isosceles_l2203_220339


namespace NUMINAMATH_CALUDE_intersection_set_equality_l2203_220329

theorem intersection_set_equality : 
  let S := {α : ℝ | ∃ k : ℤ, α = k * π / 2 - π / 5 ∧ 0 < α ∧ α < π}
  S = {3 * π / 10, 4 * π / 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_equality_l2203_220329


namespace NUMINAMATH_CALUDE_three_greater_than_negative_five_l2203_220367

theorem three_greater_than_negative_five : 3 > -5 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_negative_five_l2203_220367


namespace NUMINAMATH_CALUDE_true_propositions_count_l2203_220331

/-- Represents the four propositions about geometric solids -/
inductive GeometricProposition
| RegularPyramidLateralEdges
| RightPrismLateralFaces
| CylinderGeneratrix
| ConeSectionIsoscelesTriangles

/-- Determines if a given geometric proposition is true -/
def isTrue (prop : GeometricProposition) : Bool :=
  match prop with
  | .RegularPyramidLateralEdges => true
  | .RightPrismLateralFaces => false
  | .CylinderGeneratrix => true
  | .ConeSectionIsoscelesTriangles => true

/-- The list of all geometric propositions -/
def allPropositions : List GeometricProposition :=
  [.RegularPyramidLateralEdges, .RightPrismLateralFaces, .CylinderGeneratrix, .ConeSectionIsoscelesTriangles]

/-- Counts the number of true propositions -/
def countTruePropositions (props : List GeometricProposition) : Nat :=
  props.filter isTrue |>.length

/-- Theorem stating that the number of true propositions is 3 -/
theorem true_propositions_count :
  countTruePropositions allPropositions = 3 := by
  sorry


end NUMINAMATH_CALUDE_true_propositions_count_l2203_220331


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2203_220395

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 - Complex.I) / (1 + 3 * Complex.I)
  Complex.im z = -2/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2203_220395


namespace NUMINAMATH_CALUDE_derivative_at_one_l2203_220373

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2203_220373


namespace NUMINAMATH_CALUDE_soccer_team_combinations_l2203_220337

theorem soccer_team_combinations (n : ℕ) (k : ℕ) (h1 : n = 16) (h2 : k = 7) :
  Nat.choose n k = 11440 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_combinations_l2203_220337


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2203_220318

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the square perimeter condition
def square_perimeter_condition (a b : ℝ) : Prop := 
  ∃ (c : ℝ), a^2 = b^2 + c^2 ∧ 4 * a = 4 * Real.sqrt 2 ∧ b = c

-- Define the line l
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the symmetric point D
def symmetric_point (m : ℝ) (x y : ℝ) : Prop := x = 0 ∧ y = -m

-- Define the condition for D being inside the circle with EF as diameter
def inside_circle_condition (m : ℝ) : Prop :=
  ∀ k : ℝ, (m * Real.sqrt (4 * k^2 + 1))^2 < 2 * (1 + k^2) * (2 * k^2 + 1 - m^2)

-- Main theorem
theorem ellipse_m_range :
  ∀ a b m : ℝ,
  a > b ∧ b > 0 ∧ m > 0 ∧
  square_perimeter_condition a b ∧
  inside_circle_condition m →
  0 < m ∧ m < Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2203_220318


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2203_220370

theorem inequality_equivalence (x y : ℝ) : 
  (y + x > |x/2|) ↔ ((x ≥ 0 ∧ y > -x/2) ∨ (x < 0 ∧ y > -3*x/2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2203_220370


namespace NUMINAMATH_CALUDE_circle_family_properties_l2203_220391

-- Define the family of circles
def circle_family (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

-- Define the fixed circle
def fixed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_family_properties :
  (∀ a : ℝ, circle_family a 4 (-2)) ∧ 
  (circle_family (1 + Real.sqrt 5 / 5) = fixed_circle) ∧
  (circle_family (1 - Real.sqrt 5 / 5) = fixed_circle) :=
sorry

end NUMINAMATH_CALUDE_circle_family_properties_l2203_220391


namespace NUMINAMATH_CALUDE_arrangement_count_l2203_220312

theorem arrangement_count (boys girls : ℕ) (total_selected : ℕ) (girls_selected : ℕ) : 
  boys = 5 → girls = 3 → total_selected = 5 → girls_selected = 2 →
  (Nat.choose girls girls_selected) * (Nat.choose boys (total_selected - girls_selected)) * (Nat.factorial total_selected) = 3600 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l2203_220312


namespace NUMINAMATH_CALUDE_common_point_l2203_220310

/-- A function of the form f(x) = x^2 + ax + b where a + b = 2021 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + (2021 - a)

/-- Theorem: All functions f(x) = x^2 + ax + b where a + b = 2021 have a common point at (1, 2022) -/
theorem common_point : ∀ a : ℝ, f a 1 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_common_point_l2203_220310


namespace NUMINAMATH_CALUDE_square_of_complex_2_minus_i_l2203_220307

theorem square_of_complex_2_minus_i :
  let z : ℂ := 2 - I
  z^2 = 3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_2_minus_i_l2203_220307


namespace NUMINAMATH_CALUDE_log_eight_x_equals_three_point_two_five_l2203_220362

theorem log_eight_x_equals_three_point_two_five (x : ℝ) :
  Real.log x / Real.log 8 = 3.25 → x = 32 * (2 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_log_eight_x_equals_three_point_two_five_l2203_220362


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l2203_220372

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_wrt_origin a 1 5 b → a - b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l2203_220372
