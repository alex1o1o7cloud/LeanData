import Mathlib

namespace NUMINAMATH_CALUDE_f_monotone_increasing_l2489_248938

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_monotone_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun y ↦ f y) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l2489_248938


namespace NUMINAMATH_CALUDE_odot_one_four_odot_comm_l2489_248947

-- Define the ⊙ operation for rational numbers
def odot (a b : ℚ) : ℚ := a - a * b + b + 3

-- Theorem: 1 ⊙ 4 = 4
theorem odot_one_four : odot 1 4 = 4 := by sorry

-- Theorem: ⊙ is commutative
theorem odot_comm (a b : ℚ) : odot a b = odot b a := by sorry

end NUMINAMATH_CALUDE_odot_one_four_odot_comm_l2489_248947


namespace NUMINAMATH_CALUDE_cindy_same_color_probability_l2489_248978

def total_marbles : ℕ := 8
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 1
def yellow_marbles : ℕ := 1

def alice_draw : ℕ := 3
def bob_draw : ℕ := 2
def cindy_draw : ℕ := 2

def probability_cindy_same_color : ℚ := 1 / 35

theorem cindy_same_color_probability :
  probability_cindy_same_color = 1 / 35 :=
by sorry

end NUMINAMATH_CALUDE_cindy_same_color_probability_l2489_248978


namespace NUMINAMATH_CALUDE_community_service_selection_schemes_l2489_248941

theorem community_service_selection_schemes :
  let total_boys : ℕ := 4
  let total_girls : ℕ := 2
  let group_size : ℕ := 4
  let min_girls : ℕ := 1

  let selection_schemes : ℕ := 
    Nat.choose total_girls 1 * Nat.choose total_boys 3 +
    Nat.choose total_girls 2 * Nat.choose total_boys 2

  selection_schemes = 14 :=
by sorry

end NUMINAMATH_CALUDE_community_service_selection_schemes_l2489_248941


namespace NUMINAMATH_CALUDE_light_2011_is_green_l2489_248953

def light_pattern : ℕ → String
  | 0 => "green"
  | 1 => "yellow"
  | 2 => "yellow"
  | 3 => "red"
  | 4 => "red"
  | 5 => "red"
  | n + 6 => light_pattern n

theorem light_2011_is_green : light_pattern 2010 = "green" := by
  sorry

end NUMINAMATH_CALUDE_light_2011_is_green_l2489_248953


namespace NUMINAMATH_CALUDE_square_diagonals_perpendicular_l2489_248986

structure Rhombus where
  diagonals_perpendicular : Bool

structure Square extends Rhombus

theorem square_diagonals_perpendicular (rhombus_property : Rhombus → Bool)
    (square_is_rhombus : Square → Rhombus)
    (h1 : ∀ r : Rhombus, rhombus_property r = r.diagonals_perpendicular)
    (h2 : ∀ s : Square, rhombus_property (square_is_rhombus s) = true) :
  ∀ s : Square, s.diagonals_perpendicular = true := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_perpendicular_l2489_248986


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2489_248908

theorem smallest_whole_number_above_sum : ∃ (n : ℕ), 
  (n : ℚ) > (3 + 1/3 + 4 + 1/4 + 5 + 1/5 + 6 + 1/6) ∧ 
  ∀ (m : ℕ), (m : ℚ) > (3 + 1/3 + 4 + 1/4 + 5 + 1/5 + 6 + 1/6) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2489_248908


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2489_248996

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 5 = 2 ∧ 
  (x : ℤ) % 7 = 3 ∧ 
  (x : ℤ) % 9 = 4 ∧
  ∀ y : ℕ+, ((y : ℤ) % 5 = 2 ∧ (y : ℤ) % 7 = 3 ∧ (y : ℤ) % 9 = 4) → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2489_248996


namespace NUMINAMATH_CALUDE_inverse_cube_theorem_l2489_248966

-- Define the relationship between z and x
def inverse_cube_relation (z x : ℝ) : Prop :=
  ∃ k : ℝ, 7 * z = k / (x^3)

-- State the theorem
theorem inverse_cube_theorem :
  ∀ z₁ z₂ : ℝ,
  inverse_cube_relation z₁ 2 ∧ z₁ = 4 →
  inverse_cube_relation z₂ 4 →
  z₂ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_cube_theorem_l2489_248966


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2489_248907

/-- The slope of the line -/
def m : ℚ := -1/2

/-- A point on the line -/
def p : ℝ × ℝ := (2, -3)

/-- The equation of the line in the form ax + by + c = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + 2*y + 4 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -4

/-- The y-intercept of the line -/
def y_intercept : ℝ := -2

/-- The area of the triangle formed by the line and coordinate axes -/
def triangle_area : ℝ := 4

theorem triangle_area_proof :
  line_equation p.1 p.2 ∧
  (∀ x y : ℝ, line_equation x y → y - p.2 = m * (x - p.1)) →
  triangle_area = (1/2) * |x_intercept| * |y_intercept| :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2489_248907


namespace NUMINAMATH_CALUDE_storm_rainfall_l2489_248910

/-- Given a rainstorm with specific rainfall patterns, prove that the amount of rain in the first 30 minutes is 5 inches. -/
theorem storm_rainfall (first_30min : ℝ) (next_30min : ℝ) (next_hour : ℝ) 
  (h1 : next_30min = first_30min / 2)
  (h2 : next_hour = 1/2)
  (h3 : (first_30min + next_30min + next_hour) / 2 = 4) : 
  first_30min = 5 := by
sorry

end NUMINAMATH_CALUDE_storm_rainfall_l2489_248910


namespace NUMINAMATH_CALUDE_fifth_term_smallest_l2489_248994

/-- The sequence term for a given n -/
def sequence_term (n : ℕ) : ℤ := 3 * n^2 - 28 * n

/-- The 5th term is the smallest in the sequence -/
theorem fifth_term_smallest : ∀ k : ℕ, sequence_term 5 ≤ sequence_term k := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_smallest_l2489_248994


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l2489_248906

def is_valid_box (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1729

theorem min_sum_of_dimensions :
  ∃ (a b c : ℕ), is_valid_box a b c ∧
  ∀ (x y z : ℕ), is_valid_box x y z → a + b + c ≤ x + y + z ∧
  a + b + c = 39 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l2489_248906


namespace NUMINAMATH_CALUDE_danes_daughters_flowers_l2489_248954

theorem danes_daughters_flowers (total_baskets : Nat) (flowers_per_basket : Nat) 
  (growth : Nat) (died : Nat) (num_daughters : Nat) :
  total_baskets = 5 →
  flowers_per_basket = 4 →
  growth = 20 →
  died = 10 →
  num_daughters = 2 →
  (total_baskets * flowers_per_basket + died - growth) / num_daughters = 5 := by
  sorry

end NUMINAMATH_CALUDE_danes_daughters_flowers_l2489_248954


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2489_248958

theorem hyperbola_focal_length (b : ℝ) : 
  (b > 0) → 
  (∃ (x y : ℝ), x^2 - y^2/b^2 = 1) → 
  (∃ (c : ℝ), c = 2) → 
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2489_248958


namespace NUMINAMATH_CALUDE_smallest_c_for_cosine_zero_l2489_248943

theorem smallest_c_for_cosine_zero (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (∀ x : ℝ, x < 0 → a * Real.cos (b * x + c) ≠ 0) →
  a * Real.cos c = 0 →
  c ≥ π / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_cosine_zero_l2489_248943


namespace NUMINAMATH_CALUDE_sector_angle_l2489_248915

/-- Given a circular sector with area 1 and radius 1, prove that its central angle in radians is 2 -/
theorem sector_angle (area : ℝ) (radius : ℝ) (angle : ℝ) 
  (h_area : area = 1) 
  (h_radius : radius = 1) 
  (h_sector : area = 1/2 * radius^2 * angle) : angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2489_248915


namespace NUMINAMATH_CALUDE_degree_of_g_l2489_248944

def f (x : ℝ) : ℝ := -7 * x^4 + 3 * x^3 + x - 5

theorem degree_of_g (g : ℝ → ℝ) :
  (∃ (a b : ℝ), ∀ x, f x + g x = a * x + b) →
  (∃ (a b c d e : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e) :=
by sorry

end NUMINAMATH_CALUDE_degree_of_g_l2489_248944


namespace NUMINAMATH_CALUDE_cos_2theta_value_l2489_248932

theorem cos_2theta_value (θ : ℝ) :
  let a : ℝ × ℝ := (1, Real.cos (2 * x))
  let b : ℝ × ℝ := (Real.sin (2 * x), -Real.sqrt 3)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2
  f (θ / 2 + 2 * Real.pi / 3) = 6 / 5 →
  Real.cos (2 * θ) = 7 / 25 :=
by sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l2489_248932


namespace NUMINAMATH_CALUDE_trig_expression_max_value_trig_expression_max_achievable_l2489_248964

theorem trig_expression_max_value (A B C : Real) :
  (Real.sin A)^2 * (Real.cos B)^2 + (Real.sin B)^2 * (Real.cos C)^2 + (Real.sin C)^2 * (Real.cos A)^2 ≤ 1 :=
sorry

theorem trig_expression_max_achievable :
  ∃ (A B C : Real), (Real.sin A)^2 * (Real.cos B)^2 + (Real.sin B)^2 * (Real.cos C)^2 + (Real.sin C)^2 * (Real.cos A)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trig_expression_max_value_trig_expression_max_achievable_l2489_248964


namespace NUMINAMATH_CALUDE_peach_difference_l2489_248952

/-- Given a basket of peaches with specified quantities of red, yellow, and green peaches,
    prove that there are 8 more green peaches than yellow peaches. -/
theorem peach_difference (red yellow green : ℕ) 
    (h_red : red = 2)
    (h_yellow : yellow = 6)
    (h_green : green = 14) :
  green - yellow = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2489_248952


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2489_248999

open Real

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ sin (arccos (tan (arcsin x))) = x :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2489_248999


namespace NUMINAMATH_CALUDE_single_point_ellipse_l2489_248917

theorem single_point_ellipse (c : ℝ) : 
  (∃! p : ℝ × ℝ, 4 * p.1^2 + p.2^2 + 16 * p.1 - 6 * p.2 + c = 0) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_single_point_ellipse_l2489_248917


namespace NUMINAMATH_CALUDE_sundae_booth_packs_l2489_248976

/-- Calculates the number of packs needed for a given topping -/
def packs_needed (total_items : ℕ) (items_per_pack : ℕ) : ℕ :=
  (total_items + items_per_pack - 1) / items_per_pack

/-- Represents the sundae booth problem -/
theorem sundae_booth_packs (monday_sundaes tuesday_sundaes : ℕ)
  (monday_mms monday_gummy monday_marsh : ℕ)
  (tuesday_mms tuesday_gummy tuesday_marsh : ℕ)
  (mms_per_pack gummy_per_pack marsh_per_pack : ℕ)
  (h_monday : monday_sundaes = 40)
  (h_tuesday : tuesday_sundaes = 20)
  (h_monday_mms : monday_mms = 6)
  (h_monday_gummy : monday_gummy = 4)
  (h_monday_marsh : monday_marsh = 8)
  (h_tuesday_mms : tuesday_mms = 10)
  (h_tuesday_gummy : tuesday_gummy = 5)
  (h_tuesday_marsh : tuesday_marsh = 12)
  (h_mms_pack : mms_per_pack = 40)
  (h_gummy_pack : gummy_per_pack = 30)
  (h_marsh_pack : marsh_per_pack = 50) :
  (packs_needed (monday_sundaes * monday_mms + tuesday_sundaes * tuesday_mms) mms_per_pack = 11) ∧
  (packs_needed (monday_sundaes * monday_gummy + tuesday_sundaes * tuesday_gummy) gummy_per_pack = 9) ∧
  (packs_needed (monday_sundaes * monday_marsh + tuesday_sundaes * tuesday_marsh) marsh_per_pack = 12) :=
by sorry

end NUMINAMATH_CALUDE_sundae_booth_packs_l2489_248976


namespace NUMINAMATH_CALUDE_long_sleeved_jersey_cost_l2489_248995

/-- Represents the cost of jerseys and proves the cost of long-sleeved jerseys --/
theorem long_sleeved_jersey_cost 
  (long_sleeved_count : ℕ) 
  (striped_count : ℕ) 
  (striped_cost : ℕ) 
  (total_spent : ℕ) 
  (h1 : long_sleeved_count = 4)
  (h2 : striped_count = 2)
  (h3 : striped_cost = 10)
  (h4 : total_spent = 80) :
  ∃ (long_sleeved_cost : ℕ), 
    long_sleeved_count * long_sleeved_cost + striped_count * striped_cost = total_spent ∧ 
    long_sleeved_cost = 15 :=
by sorry

end NUMINAMATH_CALUDE_long_sleeved_jersey_cost_l2489_248995


namespace NUMINAMATH_CALUDE_largest_number_l2489_248985

theorem largest_number (a b c d : ℝ) 
  (sum1 : a + b + c = 180)
  (sum2 : a + b + d = 197)
  (sum3 : a + c + d = 208)
  (sum4 : b + c + d = 222) :
  max a (max b (max c d)) = 89 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l2489_248985


namespace NUMINAMATH_CALUDE_matrix_power_not_identity_l2489_248918

/-- Given a 5x5 complex matrix A with trace 0 and invertible I₅ - A, A⁵ ≠ I₅ -/
theorem matrix_power_not_identity
  (A : Matrix (Fin 5) (Fin 5) ℂ)
  (h_trace : Matrix.trace A = 0)
  (h_invertible : IsUnit (1 - A)) :
  A ^ 5 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_not_identity_l2489_248918


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2489_248983

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 6*a*x + 1 < 0) → a ∈ Set.Icc (-1/3) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2489_248983


namespace NUMINAMATH_CALUDE_rational_cube_root_sum_implies_rational_inverse_sum_l2489_248937

theorem rational_cube_root_sum_implies_rational_inverse_sum 
  (p q r : ℚ) 
  (h : ∃ (x : ℚ), x = (p^2*q)^(1/3) + (q^2*r)^(1/3) + (r^2*p)^(1/3)) : 
  ∃ (y : ℚ), y = 1/(p^2*q)^(1/3) + 1/(q^2*r)^(1/3) + 1/(r^2*p)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_root_sum_implies_rational_inverse_sum_l2489_248937


namespace NUMINAMATH_CALUDE_min_concerts_required_l2489_248989

/-- Represents a concert where some musicians play and others listen -/
structure Concert where
  players : Finset (Fin 6)

/-- Checks if a set of concerts satisfies the condition that 
    for every pair of musicians, each plays for the other in some concert -/
def satisfies_condition (concerts : List Concert) : Prop :=
  ∀ i j, i ≠ j → 
    (∃ c ∈ concerts, i ∈ c.players ∧ j ∉ c.players) ∧
    (∃ c ∈ concerts, j ∈ c.players ∧ i ∉ c.players)

/-- The main theorem: the minimum number of concerts required is 4 -/
theorem min_concerts_required : 
  (∃ concerts : List Concert, concerts.length = 4 ∧ satisfies_condition concerts) ∧
  (∀ concerts : List Concert, concerts.length < 4 → ¬satisfies_condition concerts) :=
sorry

end NUMINAMATH_CALUDE_min_concerts_required_l2489_248989


namespace NUMINAMATH_CALUDE_angle_CDE_is_right_angle_l2489_248942

theorem angle_CDE_is_right_angle 
  (angle_A angle_B angle_C : Real)
  (angle_AEB angle_BED angle_BDE : Real)
  (h1 : angle_A = 90)
  (h2 : angle_B = 90)
  (h3 : angle_C = 90)
  (h4 : angle_AEB = 50)
  (h5 : angle_BED = 40)
  (h6 : angle_BDE = 50)
  : ∃ (angle_CDE : Real), angle_CDE = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_CDE_is_right_angle_l2489_248942


namespace NUMINAMATH_CALUDE_cone_height_calculation_l2489_248916

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cone with a given base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Theorem: Given three spheres and a cone touching externally on a flat surface,
    the height of the cone is 28 -/
theorem cone_height_calculation (s₁ s₂ s₃ : Sphere) (c : Cone) :
  s₁.radius = 20 →
  s₂.radius = 40 →
  s₃.radius = 40 →
  c.baseRadius = 21 →
  (∃ (arrangement : ℝ → ℝ → ℝ), 
    arrangement s₁.radius s₂.radius = arrangement s₁.radius s₃.radius ∧
    arrangement s₂.radius s₃.radius = s₂.radius + s₃.radius ∧
    arrangement s₁.radius s₂.radius = Real.sqrt ((s₁.radius + s₂.radius)^2 - (s₂.radius - s₁.radius)^2)) →
  c.height = 28 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_calculation_l2489_248916


namespace NUMINAMATH_CALUDE_fishing_competition_result_l2489_248919

/-- The total number of days in the fishing season -/
def season_days : ℕ := 213

/-- The number of fish caught per day by the first fisherman -/
def first_fisherman_rate : ℕ := 3

/-- The number of days the second fisherman catches 1 fish per day -/
def second_fisherman_phase1_days : ℕ := 30

/-- The number of days the second fisherman catches 2 fish per day -/
def second_fisherman_phase2_days : ℕ := 60

/-- The number of fish caught per day by the second fisherman in phase 1 -/
def second_fisherman_phase1_rate : ℕ := 1

/-- The number of fish caught per day by the second fisherman in phase 2 -/
def second_fisherman_phase2_rate : ℕ := 2

/-- The number of fish caught per day by the second fisherman in phase 3 -/
def second_fisherman_phase3_rate : ℕ := 4

/-- The total number of fish caught by the first fisherman -/
def first_fisherman_total : ℕ := first_fisherman_rate * season_days

/-- The total number of fish caught by the second fisherman -/
def second_fisherman_total : ℕ :=
  second_fisherman_phase1_rate * second_fisherman_phase1_days +
  second_fisherman_phase2_rate * second_fisherman_phase2_days +
  second_fisherman_phase3_rate * (season_days - second_fisherman_phase1_days - second_fisherman_phase2_days)

theorem fishing_competition_result :
  second_fisherman_total - first_fisherman_total = 3 := by sorry

end NUMINAMATH_CALUDE_fishing_competition_result_l2489_248919


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l2489_248946

/-- A function that checks if a positive integer contains the consecutive digit sequence 2048 -/
def contains2048 (n : ℕ+) : Prop := sorry

/-- The set of all positive integers that do not contain the consecutive digit sequence 2048 -/
def S : Set ℕ+ := {n : ℕ+ | ¬contains2048 n}

/-- The theorem to be proved -/
theorem sum_reciprocals_bound (T : Set ℕ+) (h : T ⊆ S) :
  ∑' (n : T), (1 : ℝ) / n ≤ 400000 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l2489_248946


namespace NUMINAMATH_CALUDE_sin_30_minus_one_plus_pi_to_zero_l2489_248998

theorem sin_30_minus_one_plus_pi_to_zero (h1 : Real.sin (30 * π / 180) = 1 / 2) 
  (h2 : ∀ x : ℝ, x ^ (0 : ℝ) = 1) : 
  Real.sin (30 * π / 180) - (1 + π) ^ (0 : ℝ) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_minus_one_plus_pi_to_zero_l2489_248998


namespace NUMINAMATH_CALUDE_smallest_k_for_multiple_of_180_k_1080_is_multiple_of_180_k_1080_is_smallest_l2489_248956

def sum_of_squares (k : ℕ) : ℕ := k * (k + 1) * (2 * k + 1) / 6

theorem smallest_k_for_multiple_of_180 :
  ∀ k : ℕ, k > 0 → sum_of_squares k % 180 = 0 → k ≥ 1080 :=
by sorry

theorem k_1080_is_multiple_of_180 :
  sum_of_squares 1080 % 180 = 0 :=
by sorry

theorem k_1080_is_smallest :
  ∀ k : ℕ, k > 0 → sum_of_squares k % 180 = 0 → k = 1080 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_multiple_of_180_k_1080_is_multiple_of_180_k_1080_is_smallest_l2489_248956


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l2489_248979

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n : ℝ) * exterior_angle = 360 → exterior_angle = 30 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l2489_248979


namespace NUMINAMATH_CALUDE_bus_passengers_l2489_248961

theorem bus_passengers (initial_passengers : ℕ) (step_off_difference : ℕ) (final_passengers : ℕ) :
  initial_passengers = 38 →
  step_off_difference = 9 →
  final_passengers = initial_passengers - step_off_difference →
  final_passengers = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l2489_248961


namespace NUMINAMATH_CALUDE_rank_squared_inequality_l2489_248914

theorem rank_squared_inequality (A B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : Matrix.rank A > Matrix.rank B) : 
  Matrix.rank (A ^ 2) ≥ Matrix.rank (B ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_rank_squared_inequality_l2489_248914


namespace NUMINAMATH_CALUDE_probability_b_greater_than_a_l2489_248931

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (A.product B).filter (fun p => p.2 > p.1)

theorem probability_b_greater_than_a :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_b_greater_than_a_l2489_248931


namespace NUMINAMATH_CALUDE_total_onions_l2489_248939

theorem total_onions (sara sally fred jack : ℕ) 
  (h1 : sara = 4) 
  (h2 : sally = 5) 
  (h3 : fred = 9) 
  (h4 : jack = 7) : 
  sara + sally + fred + jack = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_onions_l2489_248939


namespace NUMINAMATH_CALUDE_exists_t_shape_l2489_248923

/-- Represents a grid of squares -/
structure Grid :=
  (size : ℕ)
  (removed : ℕ)

/-- Function that measures the connectivity of the grid -/
def f (g : Grid) : ℤ :=
  2 * g.size^2 - 4 * g.size - 10 * g.removed

/-- Theorem stating that after removing 1950 rectangles, 
    there always exists a square with at least three adjacent squares -/
theorem exists_t_shape (g : Grid) 
  (h1 : g.size = 100) 
  (h2 : g.removed = 1950) : 
  ∃ (square : Unit), f g > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_t_shape_l2489_248923


namespace NUMINAMATH_CALUDE_xyz_sum_l2489_248945

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 12)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 28) :
  x*y + y*z + x*z = 16 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l2489_248945


namespace NUMINAMATH_CALUDE_time_period_is_seven_days_l2489_248922

/-- The number of horses Minnie mounts per day -/
def minnie_daily_mounts : ℕ := sorry

/-- The number of days in the time period -/
def time_period : ℕ := sorry

/-- The number of horses Mickey mounts per day -/
def mickey_daily_mounts : ℕ := sorry

/-- Mickey mounts six less than twice as many horses per day as Minnie -/
axiom mickey_minnie_relation : mickey_daily_mounts = 2 * minnie_daily_mounts - 6

/-- Minnie mounts three more horses per day than there are days in the time period -/
axiom minnie_time_relation : minnie_daily_mounts = time_period + 3

/-- Mickey mounts 98 horses per week -/
axiom mickey_weekly_mounts : mickey_daily_mounts * 7 = 98

/-- The main theorem: The time period is 7 days -/
theorem time_period_is_seven_days : time_period = 7 := by sorry

end NUMINAMATH_CALUDE_time_period_is_seven_days_l2489_248922


namespace NUMINAMATH_CALUDE_problem_statement_l2489_248911

theorem problem_statement (P Q : Prop) (h_P : P ↔ (2 + 2 = 5)) (h_Q : Q ↔ (3 > 2)) :
  (P ∨ Q) ∧ ¬(¬Q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2489_248911


namespace NUMINAMATH_CALUDE_cube_edge_increase_l2489_248971

theorem cube_edge_increase (e : ℝ) (h : e > 0) :
  let A := 6 * e^2
  let A' := 2.25 * A
  let e' := Real.sqrt (A' / 6)
  (e' - e) / e = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_increase_l2489_248971


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_265_l2489_248973

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_101101_equals_octal_265 :
  let binary : List Bool := [true, false, true, true, false, true]
  let decimal : Nat := binary_to_decimal binary
  let octal : List Nat := decimal_to_octal decimal
  octal = [5, 6, 2] := by sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_265_l2489_248973


namespace NUMINAMATH_CALUDE_equation_solution_l2489_248988

theorem equation_solution : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2489_248988


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2489_248905

theorem floor_ceil_sum : ⌊(0.99 : ℝ)⌋ + ⌈(2.99 : ℝ)⌉ + 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2489_248905


namespace NUMINAMATH_CALUDE_range_sum_bounds_l2489_248997

/-- The function f(x) = -2x^2 + 4x -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

/-- The range of f is [m, n] -/
def m : ℝ := -6
def n : ℝ := 2

theorem range_sum_bounds :
  ∀ x, m ≤ f x ∧ f x ≤ n →
  0 ≤ m + n ∧ m + n ≤ 4 := by
  sorry

#check range_sum_bounds

end NUMINAMATH_CALUDE_range_sum_bounds_l2489_248997


namespace NUMINAMATH_CALUDE_midnight_temperature_l2489_248969

/-- Given an initial temperature, a temperature rise, and a temperature drop,
    calculate the final temperature. -/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/-- Theorem stating that given the specific temperature changes in the problem,
    the final temperature is 2°C. -/
theorem midnight_temperature :
  final_temperature (-2) 12 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l2489_248969


namespace NUMINAMATH_CALUDE_simplify_calculations_l2489_248960

theorem simplify_calculations :
  (3.5 * 10.1 = 35.35) ∧
  (0.58 * 98 = 56.84) ∧
  (3.6 * 6.91 + 6.4 * 6.91 = 69.1) ∧
  ((19.1 - (1.64 + 2.36)) / 2.5 = 6.04) := by
  sorry

end NUMINAMATH_CALUDE_simplify_calculations_l2489_248960


namespace NUMINAMATH_CALUDE_exists_bound_for_factorial_digit_sum_l2489_248933

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: Existence of a bound for factorial digit sum -/
theorem exists_bound_for_factorial_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10^100 := by
  sorry

end NUMINAMATH_CALUDE_exists_bound_for_factorial_digit_sum_l2489_248933


namespace NUMINAMATH_CALUDE_airplane_passengers_l2489_248993

theorem airplane_passengers (total : ℕ) (men : ℕ) : 
  total = 170 → men = 90 → 2 * (total - men - (men / 2)) = men → total - men - (men / 2) = 35 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l2489_248993


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2489_248901

/-- Given a quadratic function y = -3x^2 + 6x + 4, prove that its maximum value is 7 -/
theorem quadratic_max_value :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 6 * x + 4
  ∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max ∧ f x_max = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2489_248901


namespace NUMINAMATH_CALUDE_sandwich_menu_count_l2489_248935

theorem sandwich_menu_count (initial_count sold_out remaining : ℕ) : 
  sold_out = 5 → remaining = 4 → initial_count = sold_out + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_menu_count_l2489_248935


namespace NUMINAMATH_CALUDE_initial_order_cogs_initial_order_cogs_correct_l2489_248980

/-- Proves that the number of cogs in the initial order is 60 given the production rates and overall average --/
theorem initial_order_cogs : ℕ :=
  let initial_rate : ℚ := 15  -- cogs per hour
  let second_rate : ℚ := 60   -- cogs per hour
  let second_quantity : ℕ := 60
  let average_rate : ℚ := 24  -- cogs per hour
  
  -- Define a function to calculate the initial order size
  let initial_order (x : ℕ) : Prop :=
    (x + second_quantity : ℚ) / ((x : ℚ) / initial_rate + (second_quantity : ℚ) / second_rate) = average_rate

  -- The theorem states that the initial order is 60 cogs
  60

theorem initial_order_cogs_correct : initial_order_cogs = 60 := by
  sorry

#check initial_order_cogs
#check initial_order_cogs_correct

end NUMINAMATH_CALUDE_initial_order_cogs_initial_order_cogs_correct_l2489_248980


namespace NUMINAMATH_CALUDE_sample_capacity_l2489_248926

theorem sample_capacity (f : ℕ) (fr : ℚ) (h1 : f = 36) (h2 : fr = 1/4) :
  ∃ n : ℕ, f / n = fr ∧ n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l2489_248926


namespace NUMINAMATH_CALUDE_fraction_comparison_l2489_248962

theorem fraction_comparison : (10^1984 + 1) / (10^1985) > (10^1985 + 1) / (10^1986) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2489_248962


namespace NUMINAMATH_CALUDE_tv_conditional_probability_l2489_248955

theorem tv_conditional_probability 
  (p_10000 : ℝ) 
  (p_15000 : ℝ) 
  (h1 : p_10000 = 0.80) 
  (h2 : p_15000 = 0.60) : 
  p_15000 / p_10000 = 0.75 := by
sorry

end NUMINAMATH_CALUDE_tv_conditional_probability_l2489_248955


namespace NUMINAMATH_CALUDE_ella_video_game_spending_l2489_248984

/-- Proves that Ella spent $100 on video games last year given her current salary and spending habits -/
theorem ella_video_game_spending (new_salary : ℝ) (raise_percentage : ℝ) (video_game_percentage : ℝ) :
  new_salary = 275 →
  raise_percentage = 0.1 →
  video_game_percentage = 0.4 →
  (new_salary / (1 + raise_percentage)) * video_game_percentage = 100 := by
  sorry

end NUMINAMATH_CALUDE_ella_video_game_spending_l2489_248984


namespace NUMINAMATH_CALUDE_area_of_larger_rectangle_l2489_248934

/-- A rectangle with area 2 and length twice its width -/
structure SmallerRectangle where
  width : ℝ
  length : ℝ
  area_eq_two : width * length = 2
  length_eq_twice_width : length = 2 * width

/-- The larger rectangle formed by three smaller rectangles -/
def LargerRectangle (r : SmallerRectangle) : ℝ × ℝ :=
  (3 * r.length, r.width)

/-- The theorem to be proved -/
theorem area_of_larger_rectangle (r : SmallerRectangle) :
  (LargerRectangle r).1 * (LargerRectangle r).2 = 6 := by
  sorry

#check area_of_larger_rectangle

end NUMINAMATH_CALUDE_area_of_larger_rectangle_l2489_248934


namespace NUMINAMATH_CALUDE_negation_of_sum_equals_one_l2489_248900

theorem negation_of_sum_equals_one (a b : ℝ) :
  ¬(a + b = 1) ↔ (a + b > 1 ∨ a + b < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_sum_equals_one_l2489_248900


namespace NUMINAMATH_CALUDE_grid_division_theorem_l2489_248959

/-- Represents a grid division into squares and corners -/
structure GridDivision where
  squares : ℕ  -- number of 2x2 squares
  corners : ℕ  -- number of 3-cell corners

/-- Checks if a grid division is valid for a 7x14 grid -/
def is_valid_division (d : GridDivision) : Prop :=
  4 * d.squares + 3 * d.corners = 7 * 14

theorem grid_division_theorem :
  -- Part a: There exists a valid division where squares = corners
  (∃ d : GridDivision, is_valid_division d ∧ d.squares = d.corners) ∧
  -- Part b: There does not exist a valid division where squares > corners
  (¬ ∃ d : GridDivision, is_valid_division d ∧ d.squares > d.corners) := by
  sorry

end NUMINAMATH_CALUDE_grid_division_theorem_l2489_248959


namespace NUMINAMATH_CALUDE_max_piles_l2489_248974

/-- Represents a configuration of stone piles -/
structure StonePiles :=
  (piles : List Nat)
  (total_stones : Nat)
  (h_total : piles.sum = total_stones)
  (h_factor : ∀ (p q : Nat), p ∈ piles → q ∈ piles → p < 2 * q)

/-- Defines a valid split operation on stone piles -/
def split (sp : StonePiles) (i : Nat) (n : Nat) : Option StonePiles :=
  sorry

/-- Theorem: The maximum number of piles that can be formed is 30 -/
theorem max_piles (sp : StonePiles) (h_initial : sp.total_stones = 660) :
  (∀ sp' : StonePiles, ∃ (i j : Nat), split sp i j = some sp') →
  sp.piles.length ≤ 30 :=
sorry

end NUMINAMATH_CALUDE_max_piles_l2489_248974


namespace NUMINAMATH_CALUDE_parabola_points_l2489_248977

theorem parabola_points : 
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_l2489_248977


namespace NUMINAMATH_CALUDE_lucy_fish_count_lucy_fish_proof_l2489_248948

theorem lucy_fish_count : ℕ → Prop :=
  fun current_fish =>
    (current_fish + 68 = 280) → (current_fish = 212)

-- Proof
theorem lucy_fish_proof : lucy_fish_count 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_lucy_fish_proof_l2489_248948


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l2489_248912

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l2489_248912


namespace NUMINAMATH_CALUDE_remainder_101_103_div_11_l2489_248949

theorem remainder_101_103_div_11 : (101 * 103) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_103_div_11_l2489_248949


namespace NUMINAMATH_CALUDE_line_intercepts_sum_zero_l2489_248927

/-- Given a line l with equation 2x + (k - 3)y - 2k + 6 = 0, where k ≠ 3,
    if the sum of its x-intercept and y-intercept is 0, then k = 1. -/
theorem line_intercepts_sum_zero (k : ℝ) (h : k ≠ 3) :
  let l := {(x, y) : ℝ × ℝ | 2 * x + (k - 3) * y - 2 * k + 6 = 0}
  let x_intercept := (k - 3 : ℝ)
  let y_intercept := (2 : ℝ)
  x_intercept + y_intercept = 0 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_zero_l2489_248927


namespace NUMINAMATH_CALUDE_triangle_inequality_l2489_248930

theorem triangle_inequality (A B C m n l : ℝ) (h : A + B + C = π) : 
  (m^2 + Real.tan (A/2) * Real.tan (B/2))^(1/2) + 
  (n^2 + Real.tan (B/2) * Real.tan (C/2))^(1/2) + 
  (l^2 + Real.tan (C/2) * Real.tan (A/2))^(1/2) ≤ 
  (3 * (m^2 + n^2 + l^2 + 1))^(1/2) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2489_248930


namespace NUMINAMATH_CALUDE_largest_proper_fraction_and_ratio_l2489_248913

theorem largest_proper_fraction_and_ratio :
  let fractional_unit : ℚ := 1 / 5
  let largest_proper_fraction : ℚ := 4 / 5
  let reciprocal_of_ten : ℚ := 1 / 10
  (∀ n : ℕ, n < 5 → n / 5 ≤ largest_proper_fraction) ∧
  (largest_proper_fraction / reciprocal_of_ten = 8) := by
  sorry

end NUMINAMATH_CALUDE_largest_proper_fraction_and_ratio_l2489_248913


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2489_248967

theorem triangle_perimeter_bound (a b c : ℝ) : 
  a = 7 → b = 23 → (a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + c → ↑n > p ∧ ∀ (m : ℕ), ↑m > p → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2489_248967


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2489_248924

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 5 → total_games = 10 → (n * (n - 1)) / 2 = total_games → n - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2489_248924


namespace NUMINAMATH_CALUDE_doubling_base_and_exponent_l2489_248928

theorem doubling_base_and_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2*a)^(2*b) = a^b * y^b → y = 4*a :=
by sorry

end NUMINAMATH_CALUDE_doubling_base_and_exponent_l2489_248928


namespace NUMINAMATH_CALUDE_min_value_theorem_l2489_248968

theorem min_value_theorem (C : ℝ) (x : ℝ) (h1 : C > 0) (h2 : x^3 - 1/x^3 = C) :
  C^2 + 9 ≥ 6 * C ∧ ∃ (C₀ : ℝ) (x₀ : ℝ), C₀ > 0 ∧ x₀^3 - 1/x₀^3 = C₀ ∧ C₀^2 + 9 = 6 * C₀ :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2489_248968


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2489_248904

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) :
  x^3 + 1/x^3 = -18 := by sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2489_248904


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2489_248951

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 → 
  red = 14 → 
  blue = 3 * red → 
  yellow = total - (red + blue) → 
  yellow = 29 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2489_248951


namespace NUMINAMATH_CALUDE_river_straight_parts_length_l2489_248965

theorem river_straight_parts_length 
  (total_length : ℝ) 
  (straight_percentage : ℝ) 
  (h1 : total_length = 80) 
  (h2 : straight_percentage = 0.25) : 
  straight_percentage * total_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_river_straight_parts_length_l2489_248965


namespace NUMINAMATH_CALUDE_expression_equality_l2489_248992

theorem expression_equality : 49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1 = 254804368 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2489_248992


namespace NUMINAMATH_CALUDE_max_value_of_f_l2489_248987

-- Define the function f
def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 1, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2489_248987


namespace NUMINAMATH_CALUDE_intersection_empty_iff_union_equals_B_iff_l2489_248990

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem intersection_empty_iff (a : ℝ) : A a ∩ B = ∅ ↔ a ≤ -4 ∨ a ≥ 5 := by
  sorry

theorem union_equals_B_iff (a : ℝ) : A a ∪ B = B ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_union_equals_B_iff_l2489_248990


namespace NUMINAMATH_CALUDE_problem_solution_l2489_248940

theorem problem_solution (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) :
  (1/3 * x^7 * y^6) * 4 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2489_248940


namespace NUMINAMATH_CALUDE_right_triangle_area_l2489_248902

/-- The area of a right-angled triangle can be expressed in terms of its hypotenuse and one of its acute angles. -/
theorem right_triangle_area (c α : ℝ) (h_c : c > 0) (h_α : 0 < α ∧ α < π / 2) :
  let t := (1 / 4) * c^2 * Real.sin (2 * α)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = t :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2489_248902


namespace NUMINAMATH_CALUDE_percent_subtraction_problem_l2489_248970

theorem percent_subtraction_problem : ∃ x : ℝ, 0.12 * 160 - 0.38 * x = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_percent_subtraction_problem_l2489_248970


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l2489_248982

theorem junk_mail_distribution (total_mail : ℕ) (houses : ℕ) (mail_per_house : ℕ) : 
  total_mail = 14 → houses = 7 → mail_per_house = total_mail / houses → mail_per_house = 2 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l2489_248982


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2489_248921

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (eq1 : x^3 + y^3 = 98) 
  (eq2 : x^2*y + x*y^2 = -30) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2489_248921


namespace NUMINAMATH_CALUDE_equation_solution_l2489_248963

theorem equation_solution (x : ℝ) : (25 : ℝ) / 75 = (x / 75) ^ 3 → x = 75 / (3 : ℝ) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2489_248963


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2489_248981

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 10 * b = 5 * c) 
  (h2 : 3 * c = 120) : 
  b = 20 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2489_248981


namespace NUMINAMATH_CALUDE_division_of_fractions_l2489_248950

theorem division_of_fractions : (4 - 1/4) / (2 - 1/2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l2489_248950


namespace NUMINAMATH_CALUDE_scout_troop_profit_l2489_248909

/-- Calculates the profit of a scout troop selling candy bars -/
theorem scout_troop_profit
  (num_bars : ℕ)
  (buy_price : ℚ)
  (sell_price : ℚ)
  (h_num_bars : num_bars = 1500)
  (h_buy_price : buy_price = 3 / 4)  -- Price per bar when buying
  (h_sell_price : sell_price = 2 / 3)  -- Price per bar when selling
  : (num_bars : ℚ) * sell_price - (num_bars : ℚ) * buy_price = -125 := by
  sorry

#check scout_troop_profit

end NUMINAMATH_CALUDE_scout_troop_profit_l2489_248909


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l2489_248972

/-- A geometric sequence with a_3 = 16 and a_5 = 4 has a_7 = 1 -/
theorem geometric_sequence_a7 (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 3 = 16 →                                 -- given a_3 = 16
  a 5 = 4 →                                  -- given a_5 = 4
  a 7 = 1 :=                                 -- to prove a_7 = 1
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a7_l2489_248972


namespace NUMINAMATH_CALUDE_sum_of_xy_l2489_248903

theorem sum_of_xy (x y : ℝ) 
  (eq1 : x^2 + 3*x*y + y^2 = 909)
  (eq2 : 3*x^2 + x*y + 3*y^2 = 1287) :
  x + y = 27 ∨ x + y = -27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2489_248903


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2489_248957

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 - Real.cos (x * Real.sin (1 / x))
  else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2489_248957


namespace NUMINAMATH_CALUDE_range_of_even_power_function_l2489_248929

theorem range_of_even_power_function (k : ℕ) (hk : Even k) (hk_pos : k > 0) :
  Set.range (fun x : ℝ => x ^ k) = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_even_power_function_l2489_248929


namespace NUMINAMATH_CALUDE_friend_team_assignment_l2489_248975

theorem friend_team_assignment (n : ℕ) (k : ℕ) : 
  n = 6 → k = 3 → k ^ n = 729 := by sorry

end NUMINAMATH_CALUDE_friend_team_assignment_l2489_248975


namespace NUMINAMATH_CALUDE_total_fish_count_l2489_248920

/-- Given 261 fishbowls with 23 fish each, prove that the total number of fish is 6003. -/
theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2489_248920


namespace NUMINAMATH_CALUDE_total_birds_caught_l2489_248936

def birds_caught_day : ℕ := 8

def birds_caught_night (day : ℕ) : ℕ := 2 * day

theorem total_birds_caught :
  birds_caught_day + birds_caught_night birds_caught_day = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_birds_caught_l2489_248936


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_price_l2489_248925

/-- The price Ramesh paid for the refrigerator --/
def price_paid (labelled_price : ℝ) : ℝ :=
  0.8 * labelled_price + 125 + 250

/-- The theorem stating the price Ramesh paid for the refrigerator --/
theorem ramesh_refrigerator_price :
  ∃ (labelled_price : ℝ),
    1.2 * labelled_price = 19200 ∧
    price_paid labelled_price = 13175 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_price_l2489_248925


namespace NUMINAMATH_CALUDE_set_inclusion_iff_range_l2489_248991

/-- Given sets A and B, prove that (ℝ \ B) ⊆ A if and only if a ≤ -2 or 1/2 ≤ a < 1 -/
theorem set_inclusion_iff_range (a : ℝ) : 
  let A : Set ℝ := {x | x < -1 ∨ x ≥ 1}
  let B : Set ℝ := {x | x ≤ 2*a ∨ x ≥ a+1}
  (Set.univ \ B) ⊆ A ↔ a ≤ -2 ∨ (1/2 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_iff_range_l2489_248991
