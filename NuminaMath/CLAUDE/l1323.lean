import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_expression_l1323_132334

theorem trigonometric_expression (α : Real) (m : Real) 
  (h : Real.tan (5 * Real.pi + α) = m) : 
  (Real.sin (α - 3 * Real.pi) + Real.cos (-α)) / (Real.sin α - Real.cos (Real.pi + α)) = (m + 1) / (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_l1323_132334


namespace NUMINAMATH_CALUDE_direct_proportion_is_straight_line_direct_proportion_passes_through_origin_l1323_132331

/-- A direct proportion function -/
def direct_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ k * x

/-- The graph of a function -/
def graph (f : ℝ → ℝ) : Set (ℝ × ℝ) := {p | p.2 = f p.1}

theorem direct_proportion_is_straight_line (k : ℝ) :
  ∃ (a b : ℝ), ∀ x y, (x, y) ∈ graph (direct_proportion k) ↔ a * x + b * y = 0 :=
sorry

theorem direct_proportion_passes_through_origin (k : ℝ) :
  (0, 0) ∈ graph (direct_proportion k) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_is_straight_line_direct_proportion_passes_through_origin_l1323_132331


namespace NUMINAMATH_CALUDE_trapezoid_base_midpoint_relation_shorter_base_length_l1323_132337

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  midpoint_segment : ℝ
  short_base : ℝ

/-- The theorem stating the relationship between the bases and midpoint segment in a trapezoid -/
theorem trapezoid_base_midpoint_relation (t : Trapezoid) 
  (h1 : t.long_base = 113)
  (h2 : t.midpoint_segment = 4) :
  t.short_base = 105 := by
  sorry

/-- The main theorem proving the length of the shorter base -/
theorem shorter_base_length :
  ∃ t : Trapezoid, t.long_base = 113 ∧ t.midpoint_segment = 4 ∧ t.short_base = 105 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_base_midpoint_relation_shorter_base_length_l1323_132337


namespace NUMINAMATH_CALUDE_same_type_square_roots_l1323_132333

theorem same_type_square_roots :
  ∃ (k₁ k₂ : ℝ) (x : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ Real.sqrt 12 = k₁ * x ∧ Real.sqrt (1/3) = k₂ * x :=
by sorry

end NUMINAMATH_CALUDE_same_type_square_roots_l1323_132333


namespace NUMINAMATH_CALUDE_dorothy_interest_l1323_132393

/-- Calculates the interest earned on an investment with annual compound interest. -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- The interest earned on Dorothy's investment -/
theorem dorothy_interest : 
  let principal := 2000
  let rate := 0.02
  let years := 3
  ⌊interest_earned principal rate years⌋ = 122 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_interest_l1323_132393


namespace NUMINAMATH_CALUDE_min_value_of_f_l1323_132388

/-- The function f(n) = n^2 - 8n + 5 -/
def f (n : ℝ) : ℝ := n^2 - 8*n + 5

/-- The minimum value of f(n) is -11 -/
theorem min_value_of_f : ∀ n : ℝ, f n ≥ -11 ∧ ∃ n₀ : ℝ, f n₀ = -11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1323_132388


namespace NUMINAMATH_CALUDE_existence_of_solution_specific_solution_valid_l1323_132309

theorem existence_of_solution :
  ∃ (n m : ℝ), n ≠ 0 ∧ m ≠ 0 ∧ (n * 5^n)^n = m * 5^9 :=
by sorry

theorem specific_solution_valid :
  let n : ℝ := 3
  let m : ℝ := 27
  (n * 5^n)^n = m * 5^9 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_solution_specific_solution_valid_l1323_132309


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1323_132390

/-- Given a point P(a,b) on the line y = √3x - √3, 
    the minimum value of (a+1)^2 + b^2 is 3 -/
theorem min_distance_to_line (a b : ℝ) : 
  b = Real.sqrt 3 * a - Real.sqrt 3 → 
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (a + 1)^2 + b^2 ≤ (x + 1)^2 + y^2) → 
  (a + 1)^2 + b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1323_132390


namespace NUMINAMATH_CALUDE_equation_solution_l1323_132311

theorem equation_solution : 
  ∃! x : ℚ, (x - 100) / 3 = (5 - 3 * x) / 7 ∧ x = 715 / 16 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1323_132311


namespace NUMINAMATH_CALUDE_integral_sine_product_zero_and_no_beta_solution_l1323_132395

theorem integral_sine_product_zero_and_no_beta_solution 
  (m n : ℕ) (h_distinct : m ≠ n) (h_positive_m : m > 0) (h_positive_n : n > 0) :
  (∀ α : ℝ, |α| < 1 → ∫ x in -π..π, Real.sin ((m : ℝ) + α) * x * Real.sin ((n : ℝ) + α) * x = 0) ∧
  ¬ ∃ β : ℝ, (∫ x in -π..π, Real.sin ((m : ℝ) + β) * x ^ 2 = π + 2 / (4 * m - 1)) ∧
             (∫ x in -π..π, Real.sin ((n : ℝ) + β) * x ^ 2 = π + 2 / (4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_sine_product_zero_and_no_beta_solution_l1323_132395


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1323_132357

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  (a > 0 → S = {x | x < -a/4} ∪ {x | x > a/3}) ∧
  (a = 0 → S = {x | x ≠ 0}) ∧
  (a < 0 → S = {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1323_132357


namespace NUMINAMATH_CALUDE_female_students_like_pe_l1323_132364

def total_students : ℕ := 1500
def male_percentage : ℚ := 2/5
def female_dislike_pe_percentage : ℚ := 13/20

theorem female_students_like_pe : 
  (total_students : ℚ) * (1 - male_percentage) * (1 - female_dislike_pe_percentage) = 315 := by
  sorry

end NUMINAMATH_CALUDE_female_students_like_pe_l1323_132364


namespace NUMINAMATH_CALUDE_hexahedron_faces_l1323_132387

/-- A hexahedron is a polyhedron with six faces -/
structure Hexahedron where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces of a hexahedron -/
def num_faces (h : Hexahedron) : ℕ := sorry

/-- Theorem: The number of faces of a hexahedron is 6 -/
theorem hexahedron_faces (h : Hexahedron) : num_faces h = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexahedron_faces_l1323_132387


namespace NUMINAMATH_CALUDE_clock_resale_price_l1323_132302

theorem clock_resale_price (original_cost : ℝ) : 
  -- Conditions
  original_cost > 0 → 
  -- Store sold to collector for 20% more than original cost
  let collector_price := 1.2 * original_cost
  -- Store bought back at 50% of collector's price
  let buyback_price := 0.5 * collector_price
  -- Difference between original cost and buyback price is $100
  original_cost - buyback_price = 100 →
  -- Store resold at 80% profit on buyback price
  let final_price := buyback_price + 0.8 * buyback_price
  -- Theorem: The final selling price is $270
  final_price = 270 := by
sorry

end NUMINAMATH_CALUDE_clock_resale_price_l1323_132302


namespace NUMINAMATH_CALUDE_intersection_points_problem_l1323_132359

/-- The number of intersection points in the first quadrant given points on x and y axes -/
def intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- The theorem stating the number of intersection points for the given problem -/
theorem intersection_points_problem :
  intersection_points 15 10 = 4725 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_problem_l1323_132359


namespace NUMINAMATH_CALUDE_book_cost_proof_l1323_132355

/-- Given that Mark started with $85, bought 10 books, and was left with $35, prove that each book cost $5. -/
theorem book_cost_proof (initial_amount : ℕ) (books_bought : ℕ) (remaining_amount : ℕ) :
  initial_amount = 85 ∧ books_bought = 10 ∧ remaining_amount = 35 →
  (initial_amount - remaining_amount) / books_bought = 5 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_proof_l1323_132355


namespace NUMINAMATH_CALUDE_rotate180_unique_l1323_132314

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a rigid transformation (isometry) in 2D space -/
def RigidTransformation := Point2D → Point2D

/-- Clockwise rotation by 180° about the origin -/
def rotate180 : RigidTransformation :=
  fun p => Point2D.mk (-p.x) (-p.y)

/-- The given points -/
def C : Point2D := Point2D.mk 3 (-2)
def D : Point2D := Point2D.mk 4 (-5)
def C' : Point2D := Point2D.mk (-3) 2
def D' : Point2D := Point2D.mk (-4) 5

/-- Statement: rotate180 is the unique isometry that maps C to C' and D to D' -/
theorem rotate180_unique : 
  (rotate180 C = C') ∧ 
  (rotate180 D = D') ∧ 
  (∀ (f : RigidTransformation), (f C = C' ∧ f D = D') → f = rotate180) :=
sorry

end NUMINAMATH_CALUDE_rotate180_unique_l1323_132314


namespace NUMINAMATH_CALUDE_actual_average_height_l1323_132322

/-- The number of boys in the class -/
def num_boys : ℕ := 50

/-- The initially calculated average height in cm -/
def initial_avg : ℝ := 175

/-- The incorrectly recorded heights of three boys in cm -/
def incorrect_heights : List ℝ := [155, 185, 170]

/-- The actual heights of the three boys in cm -/
def actual_heights : List ℝ := [145, 195, 160]

/-- The actual average height of the boys in the class -/
def actual_avg : ℝ := 174.8

theorem actual_average_height :
  let total_incorrect := num_boys * initial_avg
  let height_difference := (List.sum incorrect_heights) - (List.sum actual_heights)
  let total_correct := total_incorrect - height_difference
  (total_correct / num_boys) = actual_avg :=
sorry

end NUMINAMATH_CALUDE_actual_average_height_l1323_132322


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1323_132354

theorem sin_50_plus_sqrt3_tan_10_equals_1 : 
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1323_132354


namespace NUMINAMATH_CALUDE_units_digit_factorial_50_l1323_132325

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem units_digit_factorial_50 :
  ∃ k : ℕ, factorial 50 = 10 * k :=
sorry

end NUMINAMATH_CALUDE_units_digit_factorial_50_l1323_132325


namespace NUMINAMATH_CALUDE_intersection_ordinate_l1323_132324

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 1)^2 - 3

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point of the parabola and the y-axis -/
def intersection_point (x y : ℝ) : Prop := parabola x y ∧ y_axis x

theorem intersection_ordinate :
  ∃ x y : ℝ, intersection_point x y ∧ y = -5 := by sorry

end NUMINAMATH_CALUDE_intersection_ordinate_l1323_132324


namespace NUMINAMATH_CALUDE_handshake_theorem_l1323_132342

theorem handshake_theorem (n : ℕ) (k : ℕ) (h : n = 30 ∧ k = 3) :
  (n * k) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l1323_132342


namespace NUMINAMATH_CALUDE_tens_digit_of_4032_pow_4033_minus_4036_l1323_132347

theorem tens_digit_of_4032_pow_4033_minus_4036 :
  (4032^4033 - 4036) % 100 / 10 = 9 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_4032_pow_4033_minus_4036_l1323_132347


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1323_132379

theorem unique_solution_for_equation : 
  ∃! (x y : ℕ+), 2 * (x : ℕ) ^ (y : ℕ) - (y : ℕ) = 2005 ∧ x = 1003 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1323_132379


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1323_132385

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x^(1/3) * (x^5)^(1/4))^(1/3) = 4 ∧ x = 2^(8/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1323_132385


namespace NUMINAMATH_CALUDE_fence_cost_l1323_132391

/-- The cost of fencing a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 49 → price_per_foot = 58 → cost = 1624 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l1323_132391


namespace NUMINAMATH_CALUDE_correct_factorization_l1323_132398

theorem correct_factorization (x : ℝ) : x^2 - 3*x + 2 = (x - 1)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1323_132398


namespace NUMINAMATH_CALUDE_coconut_juice_unit_electric_water_heater_unit_l1323_132370

-- Define the types of containers
inductive Container
| CoconutJuiceBottle
| ElectricWaterHeater

-- Define the volume units
inductive VolumeUnit
| Milliliter
| Liter

-- Define a function to get the appropriate volume unit for a container
def appropriateUnit (container : Container) (volume : ℕ) : VolumeUnit :=
  match container with
  | Container.CoconutJuiceBottle => VolumeUnit.Milliliter
  | Container.ElectricWaterHeater => VolumeUnit.Liter

-- Theorem for coconut juice bottle
theorem coconut_juice_unit : 
  appropriateUnit Container.CoconutJuiceBottle 200 = VolumeUnit.Milliliter :=
by sorry

-- Theorem for electric water heater
theorem electric_water_heater_unit : 
  appropriateUnit Container.ElectricWaterHeater 50 = VolumeUnit.Liter :=
by sorry

end NUMINAMATH_CALUDE_coconut_juice_unit_electric_water_heater_unit_l1323_132370


namespace NUMINAMATH_CALUDE_f_differentiable_at_sqrt_non_square_l1323_132356

/-- A function f: ℝ → ℝ defined as follows:
    f(x) = 0 if x is irrational
    f(p/q) = 1/q³ if p ∈ ℤ, q ∈ ℕ, and p/q is in lowest terms -/
def f : ℝ → ℝ := sorry

/-- Predicate to check if a natural number is not a perfect square -/
def is_not_perfect_square (k : ℕ) : Prop := ∀ n : ℕ, n^2 ≠ k

theorem f_differentiable_at_sqrt_non_square (k : ℕ) (h : is_not_perfect_square k) :
  DifferentiableAt ℝ f (Real.sqrt k) ∧ deriv f (Real.sqrt k) = 0 := by sorry

end NUMINAMATH_CALUDE_f_differentiable_at_sqrt_non_square_l1323_132356


namespace NUMINAMATH_CALUDE_saline_drip_duration_l1323_132348

/-- Calculates the duration of a saline drip treatment -/
theorem saline_drip_duration 
  (drop_rate : ℕ) 
  (drops_per_ml : ℚ) 
  (total_volume : ℚ) : 
  drop_rate = 20 →
  drops_per_ml = 100 / 5 →
  total_volume = 120 →
  (total_volume * drops_per_ml / drop_rate) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_saline_drip_duration_l1323_132348


namespace NUMINAMATH_CALUDE_candidate_a_votes_l1323_132335

def total_votes : ℕ := 560000
def invalid_vote_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 85 / 100

theorem candidate_a_votes : 
  ⌊(1 - invalid_vote_percentage) * candidate_a_percentage * total_votes⌋ = 404600 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l1323_132335


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1323_132366

/-- The standard equation of a hyperbola with given asymptotes and passing through a specific point -/
theorem hyperbola_standard_equation (x y : ℝ) :
  (∀ (t : ℝ), y = (2/3) * t ∨ y = -(2/3) * t) →  -- Asymptotes condition
  (x = Real.sqrt 6 ∧ y = 2) →                    -- Point condition
  (3 * y^2 / 4) - (x^2 / 3) = 1 :=               -- Standard equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1323_132366


namespace NUMINAMATH_CALUDE_arc_length_specific_sector_l1323_132338

/-- Arc length formula for a sector -/
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

/-- Theorem: The length of an arc in a sector with radius 2 and central angle π/3 is 2π/3 -/
theorem arc_length_specific_sector :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  arc_length r θ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_specific_sector_l1323_132338


namespace NUMINAMATH_CALUDE_marys_overtime_rate_increase_l1323_132374

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  regularRate : ℚ
  maxWeeklyEarnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate --/
def overtimeRateIncrease (w : WorkSchedule) : ℚ :=
  let regularEarnings := w.regularRate * w.regularHours
  let overtimeEarnings := w.maxWeeklyEarnings - regularEarnings
  let overtimeHours := w.maxHours - w.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - w.regularRate) / w.regularRate) * 100

/-- Mary's work schedule --/
def marysSchedule : WorkSchedule :=
  { maxHours := 50
  , regularHours := 20
  , regularRate := 8
  , maxWeeklyEarnings := 460 }

/-- Theorem stating that Mary's overtime rate increase is 25% --/
theorem marys_overtime_rate_increase :
  overtimeRateIncrease marysSchedule = 25 := by
  sorry


end NUMINAMATH_CALUDE_marys_overtime_rate_increase_l1323_132374


namespace NUMINAMATH_CALUDE_sqrt_fifth_root_of_five_sixth_power_l1323_132316

theorem sqrt_fifth_root_of_five_sixth_power :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 5 ^ (15 / 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_fifth_root_of_five_sixth_power_l1323_132316


namespace NUMINAMATH_CALUDE_target_breaking_sequences_l1323_132380

/-- The number of unique permutations of a string with repeated characters -/
def multinomial_permutations (char_counts : List Nat) : Nat :=
  Nat.factorial (char_counts.sum) / (char_counts.map Nat.factorial).prod

/-- The target arrangement represented as character counts -/
def target_arrangement : List Nat := [4, 3, 3]

theorem target_breaking_sequences :
  multinomial_permutations target_arrangement = 4200 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_sequences_l1323_132380


namespace NUMINAMATH_CALUDE_division_problem_l1323_132326

theorem division_problem : (70 / 4 + 90 / 4) / 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1323_132326


namespace NUMINAMATH_CALUDE_min_sum_of_product_1716_l1323_132321

theorem min_sum_of_product_1716 (a b c : ℕ+) (h : a * b * c = 1716) :
  ∃ (x y z : ℕ+), x * y * z = 1716 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 31 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1716_l1323_132321


namespace NUMINAMATH_CALUDE_all_gp_lines_pass_through_origin_l1323_132378

/-- A line in 2D space defined by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The set of all lines where a, b, c form a geometric progression -/
def GPLines : Set Line :=
  {l : Line | isGeometricProgression l.a l.b l.c}

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

theorem all_gp_lines_pass_through_origin :
  ∀ l ∈ GPLines, pointOnLine ⟨0, 0⟩ l :=
sorry

end NUMINAMATH_CALUDE_all_gp_lines_pass_through_origin_l1323_132378


namespace NUMINAMATH_CALUDE_qin_jiushao_v_1_l1323_132308

def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

def nested_f (x : ℝ) : ℝ := ((3 * x + 0) * x + 2) * x + 1 * x + 4

def v_1 (x : ℝ) : ℝ := 3 * x + 0

theorem qin_jiushao_v_1 : v_1 10 = 30 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v_1_l1323_132308


namespace NUMINAMATH_CALUDE_right_triangle_condition_l1323_132369

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a * Real.cos C + c * Real.cos A = b * Real.sin B) →
  (a * Real.sin A = b * Real.sin B) →
  (b * Real.sin B = c * Real.sin C) →
  (B = π / 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l1323_132369


namespace NUMINAMATH_CALUDE_winning_post_at_200m_l1323_132397

/-- Two runners A and B, where A is faster than B but gives B a head start -/
structure RaceScenario where
  /-- The speed ratio of runner A to runner B -/
  speed_ratio : ℚ
  /-- The head start given to runner B in meters -/
  head_start : ℚ

/-- The winning post distance for two runners to arrive simultaneously -/
def winning_post_distance (scenario : RaceScenario) : ℚ :=
  (scenario.speed_ratio * scenario.head_start) / (scenario.speed_ratio - 1)

/-- Theorem stating that for the given scenario, the winning post distance is 200 meters -/
theorem winning_post_at_200m (scenario : RaceScenario) 
  (h1 : scenario.speed_ratio = 5/3)
  (h2 : scenario.head_start = 80) :
  winning_post_distance scenario = 200 := by
  sorry

end NUMINAMATH_CALUDE_winning_post_at_200m_l1323_132397


namespace NUMINAMATH_CALUDE_package_size_l1323_132312

/-- The number of candies Shirley ate -/
def candies_eaten : ℕ := 10

/-- The number of candies Shirley has left -/
def candies_left : ℕ := 2

/-- The number of candies in one package -/
def candies_in_package : ℕ := candies_eaten + candies_left

theorem package_size : candies_in_package = 12 := by
  sorry

end NUMINAMATH_CALUDE_package_size_l1323_132312


namespace NUMINAMATH_CALUDE_chess_pawns_remaining_l1323_132332

theorem chess_pawns_remaining (initial_pawns : ℕ) 
  (sophia_lost : ℕ) (chloe_lost : ℕ) : 
  initial_pawns = 8 → 
  sophia_lost = 5 → 
  chloe_lost = 1 → 
  (initial_pawns - sophia_lost) + (initial_pawns - chloe_lost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_pawns_remaining_l1323_132332


namespace NUMINAMATH_CALUDE_system_solution_unique_l1323_132361

theorem system_solution_unique (x y : ℝ) : 
  (2 * x + 3 * y = -11) ∧ (6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1323_132361


namespace NUMINAMATH_CALUDE_rowing_speed_l1323_132300

/-- The speed of a man rowing in still water, given his downstream speed and current speed. -/
theorem rowing_speed (downstream_speed current_speed : ℝ) 
  (h_downstream : downstream_speed = 18)
  (h_current : current_speed = 3) :
  downstream_speed - current_speed = 15 := by
  sorry

#check rowing_speed

end NUMINAMATH_CALUDE_rowing_speed_l1323_132300


namespace NUMINAMATH_CALUDE_f_roots_and_monotonicity_imply_b_range_l1323_132367

/-- The function f(x) = -x^3 + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- Theorem: If all roots of f(x) = 0 are within [-2, 2] and f(x) is monotonically increasing in (0, 1), then 3 ≤ b ≤ 4 -/
theorem f_roots_and_monotonicity_imply_b_range (b : ℝ) :
  (∀ x, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f b x < f b y) →
  b ∈ Set.Icc 3 4 := by
  sorry

end NUMINAMATH_CALUDE_f_roots_and_monotonicity_imply_b_range_l1323_132367


namespace NUMINAMATH_CALUDE_football_team_members_l1323_132327

theorem football_team_members :
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 5 = 1 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  n = 251 := by
sorry

end NUMINAMATH_CALUDE_football_team_members_l1323_132327


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1323_132303

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_two
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 10)
  (h_fourth : a 4 = 7) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1323_132303


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l1323_132351

theorem fractional_inequality_solution_set (x : ℝ) :
  (x + 1) / (x + 2) ≥ 0 ↔ (x ≥ -1 ∨ x < -2) ∧ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l1323_132351


namespace NUMINAMATH_CALUDE_squares_four_greater_than_prime_l1323_132329

theorem squares_four_greater_than_prime :
  ∃! n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n^2 = p + 4 :=
sorry

end NUMINAMATH_CALUDE_squares_four_greater_than_prime_l1323_132329


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_l1323_132306

/-- The smallest positive integer x such that 1980x is a perfect fourth power -/
def smallest_x : ℕ := 6006250

/-- Predicate to check if a number is a perfect fourth power -/
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^4

theorem smallest_x_is_correct :
  (∀ y : ℕ, y < smallest_x → ¬ is_fourth_power (1980 * y)) ∧
  is_fourth_power (1980 * smallest_x) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_is_correct_l1323_132306


namespace NUMINAMATH_CALUDE_third_term_is_18_l1323_132328

def arithmetic_geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

theorem third_term_is_18 (a₁ q : ℝ) (h₁ : a₁ = 2) (h₂ : q = 3) :
  arithmetic_geometric_sequence a₁ q 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_18_l1323_132328


namespace NUMINAMATH_CALUDE_find_m_value_l1323_132346

/-- Given functions f and g, prove that m = 4 when 3f(4) = g(4) -/
theorem find_m_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 3*x + m
  let g : ℝ → ℝ := λ x => x^2 - 3*x + 5*m
  3 * f 4 = g 4 → m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l1323_132346


namespace NUMINAMATH_CALUDE_cookie_banana_price_ratio_l1323_132301

theorem cookie_banana_price_ratio :
  ∀ (cookie_price banana_price : ℝ),
  cookie_price > 0 →
  banana_price > 0 →
  6 * cookie_price + 5 * banana_price > 0 →
  3 * (6 * cookie_price + 5 * banana_price) = 3 * cookie_price + 27 * banana_price →
  cookie_price / banana_price = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_cookie_banana_price_ratio_l1323_132301


namespace NUMINAMATH_CALUDE_expression_evaluation_l1323_132353

theorem expression_evaluation : 1 - (-2) - 3 - (-4) - 5 - (-6) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1323_132353


namespace NUMINAMATH_CALUDE_bus_stop_problem_l1323_132336

theorem bus_stop_problem (girls boys : ℕ) : 
  (girls - 15 = 5 * (boys - 45)) →
  (boys = 2 * (girls - 15)) →
  (girls = 40 ∧ boys = 50) :=
by sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l1323_132336


namespace NUMINAMATH_CALUDE_units_produced_l1323_132376

def fixed_costs : ℕ := 15000
def variable_cost_per_unit : ℕ := 300
def total_cost : ℕ := 27500

def total_cost_function (n : ℕ) : ℕ :=
  fixed_costs + n * variable_cost_per_unit

theorem units_produced : ∃ (n : ℕ), n > 0 ∧ n ≤ 50 ∧ total_cost_function n = total_cost :=
sorry

end NUMINAMATH_CALUDE_units_produced_l1323_132376


namespace NUMINAMATH_CALUDE_flower_count_l1323_132384

theorem flower_count (yoojung_flowers namjoon_flowers : ℕ) : 
  yoojung_flowers = 32 → 
  yoojung_flowers = 4 * namjoon_flowers → 
  yoojung_flowers + namjoon_flowers = 40 := by
sorry

end NUMINAMATH_CALUDE_flower_count_l1323_132384


namespace NUMINAMATH_CALUDE_license_plate_count_l1323_132375

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of positions for letters on the license plate -/
def letter_positions : ℕ := 4

/-- The number of digits on the license plate -/
def digit_positions : ℕ := 2

/-- The number of possible digits (0-9) -/
def digit_options : ℕ := 10

/-- Calculates the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  alphabet_size * (alphabet_size - 1).choose 2 * letter_positions.choose 2 * 2 * digit_options * (digit_options - 1)

theorem license_plate_count :
  license_plate_combinations = 8424000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1323_132375


namespace NUMINAMATH_CALUDE_exists_22_same_age_l1323_132305

/-- Represents a villager in Roche -/
structure Villager where
  age : ℕ

/-- The village of Roche -/
structure Village where
  inhabitants : Finset Villager
  total_count : inhabitants.card = 2020
  knows_same_age : ∀ v ∈ inhabitants, ∃ w ∈ inhabitants, v ≠ w ∧ v.age = w.age
  three_same_age_in_192 : ∀ (group : Finset Villager), group ⊆ inhabitants → group.card = 192 →
    ∃ (a : ℕ) (v₁ v₂ v₃ : Villager), v₁ ∈ group ∧ v₂ ∈ group ∧ v₃ ∈ group ∧
      v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧ v₁.age = a ∧ v₂.age = a ∧ v₃.age = a

/-- There exists a group of at least 22 villagers of the same age in Roche -/
theorem exists_22_same_age (roche : Village) : 
  ∃ (a : ℕ) (group : Finset Villager), group ⊆ roche.inhabitants ∧ group.card ≥ 22 ∧
    ∀ v ∈ group, v.age = a :=
sorry

end NUMINAMATH_CALUDE_exists_22_same_age_l1323_132305


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1323_132368

theorem complex_fraction_simplification :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1323_132368


namespace NUMINAMATH_CALUDE_square_boundary_length_l1323_132382

/-- The total length of the boundary created by quarter-circle arcs and straight segments
    in a square with area 144, where each side is divided into thirds and quarters. -/
theorem square_boundary_length : ∃ (l : ℝ),
  l = 12 * Real.pi + 16 ∧ 
  (∃ (s : ℝ), s^2 = 144 ∧ 
    l = 4 * (2 * Real.pi * (s / 3) / 4 + Real.pi * (s / 6) / 4) + 4 * (s / 3)) :=
by sorry

end NUMINAMATH_CALUDE_square_boundary_length_l1323_132382


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1323_132318

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (k : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1323_132318


namespace NUMINAMATH_CALUDE_highway_length_is_500_l1323_132352

/-- The length of a highway where two cars meet after traveling from opposite ends -/
def highway_length (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem stating the length of the highway is 500 miles -/
theorem highway_length_is_500 :
  highway_length 40 60 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_is_500_l1323_132352


namespace NUMINAMATH_CALUDE_train_speed_l1323_132345

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 150) (h2 : time = 3) :
  length / time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1323_132345


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l1323_132358

theorem quadratic_equation_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l1323_132358


namespace NUMINAMATH_CALUDE_apple_theorem_l1323_132339

/-- Represents the number of apples each person has -/
structure Apples where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions of the apple distribution problem -/
def apple_distribution (a : Apples) : Prop :=
  a.A + a.B + a.C < 100 ∧
  a.A - a.A / 6 - a.A / 4 = a.B + a.A / 6 ∧
  a.B + a.A / 6 = a.C + a.A / 4

theorem apple_theorem (a : Apples) (h : apple_distribution a) :
  a.A ≤ 48 ∧ a.B = a.C + 4 := by
  sorry

#check apple_theorem

end NUMINAMATH_CALUDE_apple_theorem_l1323_132339


namespace NUMINAMATH_CALUDE_circus_dogs_count_l1323_132315

theorem circus_dogs_count :
  ∀ (total_dogs : ℕ) (paws_on_ground : ℕ),
    paws_on_ground = 36 →
    (total_dogs / 2 : ℕ) * 2 + (total_dogs / 2 : ℕ) * 4 = paws_on_ground →
    total_dogs = 12 :=
by sorry

end NUMINAMATH_CALUDE_circus_dogs_count_l1323_132315


namespace NUMINAMATH_CALUDE_complex_product_real_implies_m_equals_negative_one_l1323_132350

theorem complex_product_real_implies_m_equals_negative_one (m : ℂ) : 
  (∃ (r : ℝ), (m^2 + Complex.I) * (1 + m * Complex.I) = r) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_m_equals_negative_one_l1323_132350


namespace NUMINAMATH_CALUDE_initial_trees_count_l1323_132360

/-- The number of oak trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of oak trees cut down -/
def cut_trees : ℕ := 2

/-- The number of oak trees remaining after cutting -/
def remaining_trees : ℕ := 7

/-- Theorem stating that the initial number of trees is 9 -/
theorem initial_trees_count : initial_trees = 9 := by sorry

end NUMINAMATH_CALUDE_initial_trees_count_l1323_132360


namespace NUMINAMATH_CALUDE_income_calculation_l1323_132396

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →
  income - expenditure = savings →
  savings = 3400 →
  income = 17000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l1323_132396


namespace NUMINAMATH_CALUDE_walkers_speed_l1323_132372

/-- Proves that a walker's speed is 5 mph given specific conditions involving a cyclist --/
theorem walkers_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (walker_catchup_time : ℝ) : ℝ :=
  let walker_speed : ℝ :=
    (cyclist_speed * cyclist_travel_time) / walker_catchup_time
  by
    sorry

#check walkers_speed 20 (5/60) (20/60) = 5

end NUMINAMATH_CALUDE_walkers_speed_l1323_132372


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l1323_132310

theorem rhombus_diagonal_length (area : ℝ) (ratio : ℚ) (shorter_diagonal : ℝ) : 
  area = 144 →
  ratio = 4/3 →
  shorter_diagonal = 6 * Real.sqrt 6 →
  area = (1/2) * shorter_diagonal * (ratio * shorter_diagonal) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l1323_132310


namespace NUMINAMATH_CALUDE_dot_product_parallel_l1323_132371

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define parallel vectors
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b ∨ b = k • a

theorem dot_product_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (inner a b = ‖a‖ * ‖b‖ → parallel a b) ∧
  ¬(parallel a b → inner a b = ‖a‖ * ‖b‖) :=
sorry

end NUMINAMATH_CALUDE_dot_product_parallel_l1323_132371


namespace NUMINAMATH_CALUDE_extreme_point_range_l1323_132340

theorem extreme_point_range (m : ℝ) : 
  (∃! x₀ : ℝ, x₀ > 0 ∧ 1/2 ≤ x₀ ∧ x₀ ≤ 3 ∧
    (∀ x : ℝ, x > 0 → (x₀ + 1/x₀ + m = 0 ∧
      ∀ y : ℝ, y > 0 → y ≠ x₀ → y + 1/y + m ≠ 0))) →
  -10/3 ≤ m ∧ m < -5/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_point_range_l1323_132340


namespace NUMINAMATH_CALUDE_rectangle_vertex_numbers_l1323_132394

theorem rectangle_vertex_numbers (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : 2 * a ≥ b + d)
  (h2 : 2 * b ≥ a + c)
  (h3 : 2 * c ≥ b + d)
  (h4 : 2 * d ≥ a + c) :
  a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_CALUDE_rectangle_vertex_numbers_l1323_132394


namespace NUMINAMATH_CALUDE_line_intercepts_l1323_132320

/-- Given a line with equation x/4 - y/3 = 1, prove that its x-intercept is 4 and y-intercept is -3 -/
theorem line_intercepts (x y : ℝ) :
  x/4 - y/3 = 1 → (x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l1323_132320


namespace NUMINAMATH_CALUDE_no_intersection_points_l1323_132317

/-- Parabola 1 defined by y = 2x^2 + 3x - 4 -/
def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

/-- Parabola 2 defined by y = 3x^2 + 12 -/
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 12

/-- Theorem stating that the two parabolas have no real intersection points -/
theorem no_intersection_points : ∀ x : ℝ, parabola1 x ≠ parabola2 x := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_points_l1323_132317


namespace NUMINAMATH_CALUDE_cube_volume_doubling_l1323_132383

theorem cube_volume_doubling (v : ℝ) (h : v = 27) :
  let new_volume := (2 * v^(1/3))^3
  new_volume = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_doubling_l1323_132383


namespace NUMINAMATH_CALUDE_quadruplet_babies_l1323_132341

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1200)
  (h_triplets_quadruplets : ∃ (t q : ℕ), t = 5 * q)
  (h_twins_triplets : ∃ (w t : ℕ), w = 2 * t)
  (h_sum : ∃ (w t q : ℕ), 2 * w + 3 * t + 4 * q = total_babies) :
  ∃ (q : ℕ), 4 * q = 123 := by
sorry

end NUMINAMATH_CALUDE_quadruplet_babies_l1323_132341


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1323_132313

/-- The perimeter of a triangle with vertices A(1,2), B(1,5), and C(4,5) on a Cartesian coordinate plane is 6 + 3√2. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 5)
  let C : ℝ × ℝ := (4, 5)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := distance A B + distance B C + distance C A
  perimeter = 6 + 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1323_132313


namespace NUMINAMATH_CALUDE_fibonacci_polynomial_property_l1323_132307

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_polynomial_property (n : ℕ) (P : ℕ → ℕ) :
  (∀ k ∈ Finset.range (n + 1), P (k + n + 2) = fibonacci (k + n + 2)) →
  P (2 * n + 3) = fibonacci (2 * n + 3) - 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_polynomial_property_l1323_132307


namespace NUMINAMATH_CALUDE_solve_equations_and_sum_l1323_132323

/-- Given two equations involving x and y, prove the values of x, y, and their sum. -/
theorem solve_equations_and_sum :
  ∀ (x y : ℝ),
  (0.65 * x = 0.20 * 552.50) →
  (0.35 * y = 0.30 * 867.30) →
  (x = 170) ∧ (y = 743.40) ∧ (x + y = 913.40) := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_and_sum_l1323_132323


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_of_24_l1323_132363

theorem smallest_sum_of_factors_of_24 :
  (∀ a b : ℕ, a * b = 24 → a + b ≥ 10) ∧
  (∃ a b : ℕ, a * b = 24 ∧ a + b = 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_of_24_l1323_132363


namespace NUMINAMATH_CALUDE_jane_inspection_fraction_l1323_132392

theorem jane_inspection_fraction 
  (total_rejection_rate : ℝ)
  (john_rejection_rate : ℝ)
  (jane_rejection_rate : ℝ)
  (h_total : total_rejection_rate = 0.0075)
  (h_john : john_rejection_rate = 0.007)
  (h_jane : jane_rejection_rate = 0.008)
  (h_all_inspected : ∀ x y : ℝ, x + y = 1 → 
    x * john_rejection_rate + y * jane_rejection_rate = total_rejection_rate) :
  ∃ y : ℝ, y = 1/2 ∧ ∃ x : ℝ, x + y = 1 ∧
    x * john_rejection_rate + y * jane_rejection_rate = total_rejection_rate :=
by sorry

end NUMINAMATH_CALUDE_jane_inspection_fraction_l1323_132392


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1323_132349

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ k : ℕ+, k < n → ¬(1023 * k.val ≡ 2147 * k.val [ZMOD 30])) ∧ 
  (1023 * n.val ≡ 2147 * n.val [ZMOD 30]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1323_132349


namespace NUMINAMATH_CALUDE_gerald_chores_per_month_l1323_132319

/-- Represents the number of chores Gerald needs to do per month to save for baseball supplies. -/
def chores_per_month (monthly_expense : ℕ) (season_length : ℕ) (chore_price : ℕ) : ℕ :=
  let total_expense := monthly_expense * season_length
  let off_season_months := 12 - season_length
  let monthly_savings_needed := total_expense / off_season_months
  monthly_savings_needed / chore_price

/-- Theorem stating that Gerald needs to average 5 chores per month to save for his baseball supplies. -/
theorem gerald_chores_per_month :
  chores_per_month 100 4 10 = 5 := by
  sorry

#eval chores_per_month 100 4 10

end NUMINAMATH_CALUDE_gerald_chores_per_month_l1323_132319


namespace NUMINAMATH_CALUDE_angle_sum_eq_pi_fourth_l1323_132343

theorem angle_sum_eq_pi_fourth (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : β ∈ Set.Ioo 0 (π / 2))
  (h3 : Real.tan α = 1 / 7)
  (h4 : Real.tan β = 1 / 3) :
  α + 2 * β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_eq_pi_fourth_l1323_132343


namespace NUMINAMATH_CALUDE_even_function_symmetry_is_universal_l1323_132304

-- Define what a universal proposition is
def is_universal_proposition (p : Prop) : Prop :=
  ∃ (U : Type) (P : U → Prop), p = ∀ (x : U), P x

-- Define what an even function is
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define symmetry about y-axis for a function's graph
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem even_function_symmetry_is_universal :
  is_universal_proposition (∀ f : ℝ → ℝ, is_even_function f → symmetric_about_y_axis f) :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetry_is_universal_l1323_132304


namespace NUMINAMATH_CALUDE_not_all_n_squared_plus_n_plus_41_prime_l1323_132399

theorem not_all_n_squared_plus_n_plus_41_prime :
  ∃ n : ℕ, ¬(Nat.Prime (n^2 + n + 41)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_n_squared_plus_n_plus_41_prime_l1323_132399


namespace NUMINAMATH_CALUDE_more_spins_more_accurate_l1323_132362

/-- Represents a spinner used in random simulation -/
structure Spinner :=
  (radius : ℝ)

/-- Represents the result of a spinner simulation -/
structure SimulationResult :=
  (accuracy : ℝ)

/-- Represents a random simulation using a spinner -/
def SpinnerSimulation := Spinner → ℕ → SimulationResult

/-- Axiom: The spinner must be spun randomly for accurate estimation -/
axiom random_spinning_required (s : Spinner) (n : ℕ) (sim : SpinnerSimulation) :
  SimulationResult

/-- Axiom: The number of spins affects the estimation accuracy -/
axiom spins_affect_accuracy (s : Spinner) (n m : ℕ) (sim : SpinnerSimulation) :
  n ≠ m → sim s n ≠ sim s m

/-- Axiom: The spinner's radius does not affect the estimation accuracy -/
axiom radius_doesnt_affect_accuracy (s₁ s₂ : Spinner) (n : ℕ) (sim : SpinnerSimulation) :
  s₁.radius ≠ s₂.radius → sim s₁ n = sim s₂ n

/-- Theorem: Increasing the number of spins improves the accuracy of the estimation result -/
theorem more_spins_more_accurate (s : Spinner) (n m : ℕ) (sim : SpinnerSimulation) :
  n < m → (sim s m).accuracy > (sim s n).accuracy :=
sorry

end NUMINAMATH_CALUDE_more_spins_more_accurate_l1323_132362


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l1323_132365

/-- Given two varieties of rice mixed in a specific ratio to obtain a mixture with a known cost,
    this theorem proves that the cost of the first variety can be determined. -/
theorem rice_mixture_cost
  (cost_second : ℝ)  -- Cost per kg of the second variety of rice
  (ratio : ℝ)        -- Ratio of the first variety to the second in the mixture
  (cost_mixture : ℝ) -- Cost per kg of the resulting mixture
  (h1 : cost_second = 8.75)
  (h2 : ratio = 0.8333333333333334)
  (h3 : cost_mixture = 7.50)
  : ∃ (cost_first : ℝ), 
    cost_first * (ratio / (1 + ratio)) + cost_second * (1 / (1 + ratio)) = cost_mixture ∧ 
    cost_first = 7.25 := by
  sorry


end NUMINAMATH_CALUDE_rice_mixture_cost_l1323_132365


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1323_132389

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(2/3) + m * x + 1

-- State the theorem
theorem f_monotone_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →  -- f is an even function
  ∀ x ≥ 0, ∀ y ≥ x, f m x ≤ f m y :=  -- f is monotonically increasing on [0, +∞)
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1323_132389


namespace NUMINAMATH_CALUDE_tilly_star_count_l1323_132381

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) : 
  stars_east = 120 →
  stars_west = 6 * stars_east →
  stars_east + stars_west = 840 := by
sorry

end NUMINAMATH_CALUDE_tilly_star_count_l1323_132381


namespace NUMINAMATH_CALUDE_cubic_sum_in_terms_of_products_l1323_132344

theorem cubic_sum_in_terms_of_products (x y z p q r : ℝ) 
  (h_xy : x * y = p)
  (h_xz : x * z = q)
  (h_yz : y * z = r)
  (h_x_nonzero : x ≠ 0)
  (h_y_nonzero : y ≠ 0)
  (h_z_nonzero : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_in_terms_of_products_l1323_132344


namespace NUMINAMATH_CALUDE_third_test_score_l1323_132386

def maria_scores (score3 : ℝ) : List ℝ := [80, 70, score3, 100]

theorem third_test_score (score3 : ℝ) : 
  (maria_scores score3).sum / (maria_scores score3).length = 85 → score3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_third_test_score_l1323_132386


namespace NUMINAMATH_CALUDE_no_harmonic_point_reciprocal_unique_harmonic_point_range_of_m_l1323_132377

-- Definition of a harmonic point
def is_harmonic_point (x y : ℝ) : Prop := x = y

-- Part 1: No harmonic point for y = -4/x
theorem no_harmonic_point_reciprocal : ¬∃ x : ℝ, x ≠ 0 ∧ is_harmonic_point x (-4/x) := by sorry

-- Part 2: Quadratic function with one harmonic point
theorem unique_harmonic_point (a c : ℝ) :
  a ≠ 0 ∧ 
  (∃! x : ℝ, is_harmonic_point x (a * x^2 + 6 * x + c)) ∧
  is_harmonic_point (5/2) (a * (5/2)^2 + 6 * (5/2) + c) →
  a = -1 ∧ c = -25/4 := by sorry

-- Part 3: Range of m for quadratic function with given min and max
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 ≥ -1) ∧
  (∀ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 ≤ 3) ∧
  (∃ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 = -1) ∧
  (∃ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 = 3) →
  3 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_no_harmonic_point_reciprocal_unique_harmonic_point_range_of_m_l1323_132377


namespace NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1323_132373

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1323_132373


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l1323_132330

/-- The cost of decorations for a wedding reception --/
def decorationCost (numTables : ℕ) (tableclothCost : ℕ) (placeSettingsPerTable : ℕ) 
  (placeSettingCost : ℕ) (rosesPerCenterpiece : ℕ) (roseCost : ℕ) 
  (liliesPerCenterpiece : ℕ) (lilyCost : ℕ) : ℕ :=
  numTables * tableclothCost + 
  numTables * placeSettingsPerTable * placeSettingCost +
  numTables * rosesPerCenterpiece * roseCost +
  numTables * liliesPerCenterpiece * lilyCost

/-- The total cost of decorations for Nathan's wedding reception is $3500 --/
theorem wedding_decoration_cost : 
  decorationCost 20 25 4 10 10 5 15 4 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_l1323_132330
