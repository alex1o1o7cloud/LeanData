import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_properties_l2504_250445

-- Define the ellipse and its properties
def Ellipse (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = b^2 + c^2

-- Define the points and conditions
def EllipseConditions (a b c : ℝ) (A B E F₁ F₂ : ℝ × ℝ) : Prop :=
  Ellipse a b c ∧
  F₁ = (-c, 0) ∧
  F₂ = (c, 0) ∧
  E = (a^2/c, 0) ∧
  (∃ k : ℝ, A.2 = k * (A.1 - a^2/c) ∧ B.2 = k * (B.1 - a^2/c)) ∧
  (∃ t : ℝ, A.1 - F₁.1 = t * (B.1 - F₂.1) ∧ A.2 - F₁.2 = t * (B.2 - F₂.2)) ∧
  (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = 4 * ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2)

-- Theorem statement
theorem ellipse_properties (a b c : ℝ) (A B E F₁ F₂ : ℝ × ℝ) 
  (h : EllipseConditions a b c A B E F₁ F₂) :
  c / a = Real.sqrt 3 / 3 ∧ 
  (∃ k : ℝ, (A.2 - B.2) / (A.1 - B.1) = k ∧ (k = Real.sqrt 2 / 3 ∨ k = -Real.sqrt 2 / 3)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2504_250445


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2504_250474

/-- Given an angle whose complement is 7° more than five times the angle,
    prove that the angle measures 13.833°. -/
theorem angle_measure_proof (x : ℝ) : 
  x + (5 * x + 7) = 90 → x = 13.833 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2504_250474


namespace NUMINAMATH_CALUDE_triangle_base_length_l2504_250463

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 24 →
  height = 8 →
  area = (base * height) / 2 →
  base = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2504_250463


namespace NUMINAMATH_CALUDE_min_value_expression_l2504_250467

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 48) :
  x^2 + 4*x*y + 4*y^2 + 3*z^2 ≥ 144 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 48 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 3*z₀^2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2504_250467


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l2504_250409

theorem coefficient_x_squared (p q : Polynomial ℤ) (hp : p = 3 * X^2 + 4 * X + 5) (hq : q = 6 * X^2 + 7 * X + 8) :
  (p * q).coeff 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l2504_250409


namespace NUMINAMATH_CALUDE_yoe_speed_calculation_l2504_250416

/-- Yoe's speed in miles per hour -/
def yoe_speed : ℝ := 40

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Initial distance between Teena and Yoe in miles (Teena behind) -/
def initial_distance : ℝ := 7.5

/-- Time elapsed in hours -/
def time_elapsed : ℝ := 1.5

/-- Final distance between Teena and Yoe in miles (Teena ahead) -/
def final_distance : ℝ := 15

theorem yoe_speed_calculation : 
  yoe_speed = (teena_speed * time_elapsed - initial_distance - final_distance) / time_elapsed :=
by sorry

end NUMINAMATH_CALUDE_yoe_speed_calculation_l2504_250416


namespace NUMINAMATH_CALUDE_y_minimizer_l2504_250494

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3*x + 5

/-- The theorem stating that (2a + 2b - 3) / 4 minimizes y -/
theorem y_minimizer (a b : ℝ) :
  ∃ (x_min : ℝ), x_min = (2*a + 2*b - 3) / 4 ∧
    ∀ (x : ℝ), y x a b ≥ y x_min a b :=
by
  sorry

end NUMINAMATH_CALUDE_y_minimizer_l2504_250494


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2504_250476

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n, 
    prove that if S_n / T_n = (2n - 5) / (4n + 3) for all n, then a_6 / b_6 = 17 / 47 -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1))
  (h_sum_a : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h_sum_b : ∀ n, T n = (n / 2) * (b 1 + b n))
  (h_ratio : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
  a 6 / b 6 = 17 / 47 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2504_250476


namespace NUMINAMATH_CALUDE_systematic_sampling_50_5_l2504_250413

/-- Represents a list of product numbers selected using systematic sampling. -/
def systematicSample (totalProducts : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that systematic sampling of 5 products from 50 products
    results in the selection of products numbered 10, 20, 30, 40, and 50. -/
theorem systematic_sampling_50_5 :
  systematicSample 50 5 = [10, 20, 30, 40, 50] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_50_5_l2504_250413


namespace NUMINAMATH_CALUDE_brick_width_correct_l2504_250415

/-- The width of a brick used to build a wall with given dimensions -/
def brick_width : ℝ :=
  let wall_length : ℝ := 800  -- 8 m in cm
  let wall_height : ℝ := 600  -- 6 m in cm
  let wall_thickness : ℝ := 22.5
  let brick_count : ℕ := 1600
  let brick_length : ℝ := 100
  let brick_height : ℝ := 6
  11.25

/-- Theorem stating that the calculated brick width is correct -/
theorem brick_width_correct :
  let wall_length : ℝ := 800  -- 8 m in cm
  let wall_height : ℝ := 600  -- 6 m in cm
  let wall_thickness : ℝ := 22.5
  let brick_count : ℕ := 1600
  let brick_length : ℝ := 100
  let brick_height : ℝ := 6
  wall_length * wall_height * wall_thickness = 
    brick_count * (brick_length * brick_width * brick_height) :=
by
  sorry

#eval brick_width

end NUMINAMATH_CALUDE_brick_width_correct_l2504_250415


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l2504_250454

/-- A line passing through two points intersects the x-axis at a specific point -/
theorem line_intersection_x_axis (x₁ y₁ x₂ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  let x_intercept := b / m
  (x₁ = 8 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6) →
  x_intercept = 10 ∧ m * x_intercept + b = 0 :=
by
  sorry

#check line_intersection_x_axis

end NUMINAMATH_CALUDE_line_intersection_x_axis_l2504_250454


namespace NUMINAMATH_CALUDE_hotel_revenue_maximization_l2504_250405

/-- Represents the hotel revenue optimization problem -/
def HotelRevenueProblem (totalRooms : ℕ) (initialPrice : ℕ) (initialOccupancy : ℕ) 
  (priceReduction : ℕ) (occupancyIncrease : ℕ) : Prop :=
  ∃ (maxRevenue : ℕ),
    maxRevenue = 22500 ∧
    (∀ (x : ℕ),
      let newPrice := initialPrice - x * priceReduction
      let newOccupancy := initialOccupancy + x * occupancyIncrease
      newPrice * newOccupancy ≤ maxRevenue)

/-- Theorem stating that the hotel revenue problem has a solution -/
theorem hotel_revenue_maximization :
  HotelRevenueProblem 100 400 50 20 5 := by
  sorry

#check hotel_revenue_maximization

end NUMINAMATH_CALUDE_hotel_revenue_maximization_l2504_250405


namespace NUMINAMATH_CALUDE_polynomial_never_33_l2504_250472

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
sorry

end NUMINAMATH_CALUDE_polynomial_never_33_l2504_250472


namespace NUMINAMATH_CALUDE_brendas_age_l2504_250481

theorem brendas_age (addison janet brenda : ℕ) 
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 9)
  (h3 : addison = janet) : 
  brenda = 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l2504_250481


namespace NUMINAMATH_CALUDE_vehicle_value_theorem_l2504_250447

def vehicle_value_last_year : ℝ := 20000

def depreciation_factor : ℝ := 0.8

def vehicle_value_this_year : ℝ := depreciation_factor * vehicle_value_last_year

theorem vehicle_value_theorem : vehicle_value_this_year = 16000 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_theorem_l2504_250447


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2504_250486

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.r = 2 * Real.cos (p.θ - c.center.θ)

theorem circle_equation_correct (c : PolarCircle) :
  c.center.r = 1 ∧ c.center.θ = π/4 ∧ c.radius = 1 →
  ∀ p : PolarPoint, circleEquation c p ↔ (p.r * Real.cos p.θ - c.center.r)^2 + (p.r * Real.sin p.θ - c.center.r)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2504_250486


namespace NUMINAMATH_CALUDE_horner_method_multiplications_l2504_250492

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Number of multiplications in Horner's method -/
def horner_multiplications (coeffs : List ℝ) : ℕ :=
  coeffs.length - 1

/-- The polynomial f(x) = 3x^4 + 3x^3 + 2x^2 + 6x + 1 -/
def f_coeffs : List ℝ := [3, 3, 2, 6, 1]

theorem horner_method_multiplications :
  horner_multiplications f_coeffs = 4 :=
by
  sorry

#eval horner_eval f_coeffs 0.5
#eval horner_multiplications f_coeffs

end NUMINAMATH_CALUDE_horner_method_multiplications_l2504_250492


namespace NUMINAMATH_CALUDE_initial_books_l2504_250446

theorem initial_books (total : ℕ) (additional : ℕ) (initial : ℕ) : 
  total = 77 → additional = 23 → total = initial + additional → initial = 54 := by
sorry

end NUMINAMATH_CALUDE_initial_books_l2504_250446


namespace NUMINAMATH_CALUDE_lena_video_game_time_l2504_250410

/-- Proves that Lena played video games for 3.5 hours given the conditions of the problem -/
theorem lena_video_game_time (lena_time brother_time : ℕ) : 
  brother_time = lena_time + 17 →
  lena_time + brother_time = 437 →
  (lena_time : ℚ) / 60 = 3.5 := by
    sorry

end NUMINAMATH_CALUDE_lena_video_game_time_l2504_250410


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2504_250491

theorem point_in_second_quadrant (A B : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2) :
  let P : ℝ × ℝ := (Real.cos B - Real.sin A, Real.sin B - Real.cos A)
  P.1 < 0 ∧ P.2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2504_250491


namespace NUMINAMATH_CALUDE_candle_height_at_half_time_l2504_250427

/-- Calculates the total burning time for a candle of given height -/
def totalBurningTime (height : ℕ) : ℕ :=
  10 * (height * (height + 1) * (2 * height + 1)) / 6

/-- Calculates the height of the candle after a given time -/
def heightAfterTime (initialHeight : ℕ) (elapsedTime : ℕ) : ℕ :=
  initialHeight - (Finset.filter (fun k => 10 * k * k ≤ elapsedTime) (Finset.range initialHeight)).card

theorem candle_height_at_half_time (initialHeight : ℕ) (halfTimeHeight : ℕ) :
  initialHeight = 150 →
  halfTimeHeight = heightAfterTime initialHeight (totalBurningTime initialHeight / 2) →
  halfTimeHeight = 80 := by
  sorry

#eval heightAfterTime 150 (totalBurningTime 150 / 2)

end NUMINAMATH_CALUDE_candle_height_at_half_time_l2504_250427


namespace NUMINAMATH_CALUDE_percent_of_a_is_3b_l2504_250444

theorem percent_of_a_is_3b (a b : ℝ) (h : a = 1.5 * b) : (3 * b) / a * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_3b_l2504_250444


namespace NUMINAMATH_CALUDE_second_month_sale_l2504_250411

def sale_month1 : ℕ := 6235
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 5191
def desired_average : ℕ := 6500
def num_months : ℕ := 6

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = desired_average * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month2 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l2504_250411


namespace NUMINAMATH_CALUDE_intersection_M_N_l2504_250435

-- Define set M
def M : Set ℝ := {x | x^2 + 2*x - 8 < 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2504_250435


namespace NUMINAMATH_CALUDE_initial_value_proof_l2504_250453

theorem initial_value_proof : 
  (∃ x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0) → 
  (∃! x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0) ∧
  (∃ x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0 ∧ x = 162) :=
by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l2504_250453


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l2504_250493

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l2504_250493


namespace NUMINAMATH_CALUDE_stating_probability_all_types_proof_l2504_250440

/-- Represents the probability of finding all three types of dolls in 4 blind boxes -/
def probability_all_types (ratio_A ratio_B ratio_C : ℕ) : ℝ :=
  let total := ratio_A + ratio_B + ratio_C
  let p_A := ratio_A / total
  let p_B := ratio_B / total
  let p_C := ratio_C / total
  4 * p_C * 3 * p_B * p_A^2 + 4 * p_C * 3 * p_B^2 * p_A + 6 * p_C^2 * 2 * p_B * p_A

/-- 
Theorem stating that given the production ratio of dolls A:B:C as 6:3:1, 
the probability of finding all three types of dolls when buying 4 blind boxes at once is 0.216
-/
theorem probability_all_types_proof :
  probability_all_types 6 3 1 = 0.216 := by
  sorry

end NUMINAMATH_CALUDE_stating_probability_all_types_proof_l2504_250440


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2504_250489

theorem no_integer_solutions (n k m l : ℕ) : 
  l ≥ 2 → 4 ≤ k → k ≤ n - 4 → Nat.choose n k ≠ m^l := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2504_250489


namespace NUMINAMATH_CALUDE_horner_rule_example_l2504_250459

def horner_polynomial (x : ℝ) : ℝ :=
  (((((x + 2) * x) * x - 3) * x + 7) * x - 2)

theorem horner_rule_example :
  horner_polynomial 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_example_l2504_250459


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l2504_250452

/-- The minimum value of 1/a^2 + 1/b^2 for a line tangent to a circle -/
theorem tangent_line_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + 2 * b * y + 2 = 0 ∧ x^2 + y^2 = 2) :
  (1 / a^2 + 1 / b^2 : ℝ) ≥ 9/2 ∧ 
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a₀ * x + 2 * b₀ * y + 2 = 0 ∧ x^2 + y^2 = 2) ∧
    1 / a₀^2 + 1 / b₀^2 = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l2504_250452


namespace NUMINAMATH_CALUDE_scalene_triangle_bisector_inequality_l2504_250417

/-- Given a scalene triangle with longest angle bisector l₁, shortest angle bisector l₂, and area S,
    prove that l₁² > √3 S > l₂². -/
theorem scalene_triangle_bisector_inequality (a b c : ℝ) (h_scalene : a > b ∧ b > c ∧ c > 0) :
  ∃ (l₁ l₂ S : ℝ), l₁ > 0 ∧ l₂ > 0 ∧ S > 0 ∧
  (∀ l : ℝ, (l > 0 ∧ l ≠ l₁ ∧ l ≠ l₂) → (l < l₁ ∧ l > l₂)) ∧
  S = (1/2) * b * c * Real.sin ((2/3) * Real.pi) ∧
  l₁^2 > Real.sqrt 3 * S ∧ Real.sqrt 3 * S > l₂^2 :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_bisector_inequality_l2504_250417


namespace NUMINAMATH_CALUDE_contest_scores_order_l2504_250403

theorem contest_scores_order (A B C D : ℕ) 
  (eq1 : A + B = C + D)
  (eq2 : D + B = A + C + 10)
  (eq3 : C = A + D + 5) :
  B > C ∧ C > D ∧ D > A := by
  sorry

end NUMINAMATH_CALUDE_contest_scores_order_l2504_250403


namespace NUMINAMATH_CALUDE_die_product_divisible_by_48_l2504_250408

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem die_product_divisible_by_48 (S : Finset ℕ) (h : S ⊆ die_numbers) (h_card : S.card = 7) :
  48 ∣ S.prod id :=
sorry

end NUMINAMATH_CALUDE_die_product_divisible_by_48_l2504_250408


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l2504_250488

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 50)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l2504_250488


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l2504_250400

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum_zero
  (a b c : ℝ)
  (h1 : ∃ (x : ℝ), ∀ (y : ℝ), quadratic a b c y ≥ quadratic a b c x)
  (h2 : quadratic a b c 1 = 0)
  (h3 : quadratic a b c (-3) = 0)
  (h4 : ∃ (x : ℝ), quadratic a b c x = 45)
  : a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l2504_250400


namespace NUMINAMATH_CALUDE_expression_simplification_l2504_250432

theorem expression_simplification (a : ℝ) (h : a/2 - 2/a = 3) :
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2504_250432


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l2504_250430

/-- Calculates the number of students sampled from a specific grade in a stratified sampling. -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample : ℕ) : ℕ :=
  (grade_population * total_sample) / total_population

/-- Proves that in a stratified sampling of 40 students from a population of 1000 students,
    where 400 students are in the third grade, the number of students sampled from the third grade is 16. -/
theorem third_grade_sample_size :
  stratified_sample_size 1000 400 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_sample_size_l2504_250430


namespace NUMINAMATH_CALUDE_bus_speed_with_stops_l2504_250414

/-- The speed of a bus including stoppages, given its speed excluding stoppages and stop time -/
theorem bus_speed_with_stops (speed_without_stops : ℝ) (stop_time : ℝ) :
  speed_without_stops = 54 →
  stop_time = 20 →
  let speed_with_stops := speed_without_stops * (60 - stop_time) / 60
  speed_with_stops = 36 := by
  sorry

#check bus_speed_with_stops

end NUMINAMATH_CALUDE_bus_speed_with_stops_l2504_250414


namespace NUMINAMATH_CALUDE_range_of_a_l2504_250449

theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) →
  a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2504_250449


namespace NUMINAMATH_CALUDE_jerky_order_fulfillment_l2504_250433

def days_to_fulfill_order (ordered_bags : ℕ) (existing_bags : ℕ) (bags_per_batch : ℕ) : ℕ :=
  ((ordered_bags - existing_bags) + bags_per_batch - 1) / bags_per_batch

theorem jerky_order_fulfillment :
  days_to_fulfill_order 60 20 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_jerky_order_fulfillment_l2504_250433


namespace NUMINAMATH_CALUDE_coffee_machine_payoff_l2504_250443

/-- Calculates the number of days until a coffee machine pays for itself. --/
def coffee_machine_payoff_days (machine_price : ℕ) (discount : ℕ) (daily_cost : ℕ) (prev_coffees : ℕ) (prev_price : ℕ) : ℕ :=
  let actual_cost := machine_price - discount
  let prev_daily_expense := prev_coffees * prev_price
  let daily_savings := prev_daily_expense - daily_cost
  actual_cost / daily_savings

/-- Theorem stating that under the given conditions, the coffee machine pays for itself in 36 days. --/
theorem coffee_machine_payoff :
  coffee_machine_payoff_days 200 20 3 2 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_coffee_machine_payoff_l2504_250443


namespace NUMINAMATH_CALUDE_johns_total_distance_l2504_250457

/-- Calculates the total distance driven given the speed and time for each segment of a trip. -/
def total_distance (speed1 speed2 speed3 speed4 : ℝ) (time1 time2 time3 time4 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4

/-- Theorem stating that John's total distance driven is 470 miles. -/
theorem johns_total_distance :
  total_distance 45 55 60 50 2 3 1.5 2.5 = 470 := by
  sorry

#eval total_distance 45 55 60 50 2 3 1.5 2.5

end NUMINAMATH_CALUDE_johns_total_distance_l2504_250457


namespace NUMINAMATH_CALUDE_largest_integer_divisible_by_all_less_than_cube_root_l2504_250455

theorem largest_integer_divisible_by_all_less_than_cube_root : ∃ (N : ℕ), 
  (N = 420) ∧ 
  (∀ (k : ℕ), k > 0 ∧ k ≤ ⌊(N : ℝ)^(1/3)⌋ → N % k = 0) ∧
  (∀ (M : ℕ), M > N → ∃ (m : ℕ), m > 0 ∧ m ≤ ⌊(M : ℝ)^(1/3)⌋ ∧ M % m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_divisible_by_all_less_than_cube_root_l2504_250455


namespace NUMINAMATH_CALUDE_range_of_a_l2504_250421

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 3 * (a - 3) * x^2 + 1 / x = 0) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → 
  a ∈ Set.Iic 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2504_250421


namespace NUMINAMATH_CALUDE_ship_blown_westward_distance_l2504_250420

/-- Represents the ship's journey with given conditions -/
structure ShipJourney where
  travelTime : ℝ
  speed : ℝ
  obstaclePercentage : ℝ
  finalFraction : ℝ

/-- Calculates the distance blown westward by the storm -/
def distanceBlownWestward (journey : ShipJourney) : ℝ :=
  let plannedDistance := journey.travelTime * journey.speed
  let actualDistance := plannedDistance * (1 + journey.obstaclePercentage)
  let totalDistance := 2 * actualDistance
  let finalDistance := journey.finalFraction * totalDistance
  actualDistance - finalDistance

/-- Theorem stating that for the given journey conditions, the ship was blown 230 km westward -/
theorem ship_blown_westward_distance :
  let journey : ShipJourney := {
    travelTime := 20,
    speed := 30,
    obstaclePercentage := 0.15,
    finalFraction := 1/3
  }
  distanceBlownWestward journey = 230 := by sorry

end NUMINAMATH_CALUDE_ship_blown_westward_distance_l2504_250420


namespace NUMINAMATH_CALUDE_common_root_condition_l2504_250479

theorem common_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l2504_250479


namespace NUMINAMATH_CALUDE_function_increasing_intervals_l2504_250495

noncomputable def f (A ω φ x : ℝ) : ℝ := 2 * A * (Real.cos (ω * x + φ))^2 - A

theorem function_increasing_intervals
  (A ω φ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π / 2)
  (h_symmetry_axis : ∀ x, f A ω φ (π/3 - x) = f A ω φ (π/3 + x))
  (h_symmetry_center : ∀ x, f A ω φ (π/12 - x) = f A ω φ (π/12 + x))
  : ∀ k : ℤ, StrictMonoOn (f A ω φ) (Set.Icc (k * π - 2*π/3) (k * π - π/6)) :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_intervals_l2504_250495


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l2504_250462

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_helper (n : Nat) : List Bool :=
    if n = 0 then [] else (n % 2 = 1) :: to_binary_helper (n / 2)
  to_binary_helper n

def binary_110010 : List Bool := [false, true, false, false, true, true]
def binary_1101 : List Bool := [true, false, true, true]
def binary_101 : List Bool := [true, false, true]
def binary_11110100 : List Bool := [false, false, true, false, true, true, true, true]

theorem binary_multiplication_division :
  (binary_to_nat binary_110010 * binary_to_nat binary_1101) / binary_to_nat binary_101 =
  binary_to_nat binary_11110100 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l2504_250462


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_trajectory_l2504_250480

/-- The equation of the trajectory of the midpoint of a chord passing through the focus of the parabola y² = 4x is y² = 2x - 2 -/
theorem parabola_chord_midpoint_trajectory (x y : ℝ) :
  (∀ x₀ y₀, y₀^2 = 4*x₀ → -- Parabola equation
   ∃ a b : ℝ, (y - y₀)^2 = 4*(a^2 + b^2)*(x - x₀) ∧ -- Chord passing through focus
   x = (x₀ + a)/2 ∧ y = (y₀ + b)/2) -- Midpoint of chord
  → y^2 = 2*x - 2 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_trajectory_l2504_250480


namespace NUMINAMATH_CALUDE_marble_count_l2504_250468

/-- The number of marbles in Jar A -/
def jarA : ℕ := 56

/-- The number of marbles in Jar B -/
def jarB : ℕ := 3 * jarA / 2

/-- The number of marbles in Jar C -/
def jarC : ℕ := 2 * jarA

/-- The number of marbles in Jar D -/
def jarD : ℕ := 3 * jarC / 4

/-- The total number of marbles in all jars -/
def totalMarbles : ℕ := jarA + jarB + jarC + jarD

theorem marble_count : totalMarbles = 336 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2504_250468


namespace NUMINAMATH_CALUDE_calculate_y_l2504_250441

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.Z = 4 * t.X ∧ t.x = 35 ∧ t.z = 60

-- Define the Law of Sines
def law_of_sines (t : Triangle) : Prop :=
  t.x / Real.sin t.X = t.y / Real.sin t.Y ∧
  t.y / Real.sin t.Y = t.z / Real.sin t.Z ∧
  t.z / Real.sin t.Z = t.x / Real.sin t.X

-- Theorem statement
theorem calculate_y (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : law_of_sines t) : 
  ∃ y : Real, t.y = y :=
sorry

end NUMINAMATH_CALUDE_calculate_y_l2504_250441


namespace NUMINAMATH_CALUDE_minute_hand_catches_hour_hand_l2504_250466

/-- The speed of the hour hand in degrees per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The speed of the minute hand in degrees per minute -/
def minute_hand_speed : ℚ := 6

/-- The number of degrees in a full circle -/
def full_circle : ℚ := 360

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time after 12:00 when the minute hand first catches up to the hour hand -/
def catch_up_time : ℚ := 65 + 5/11

theorem minute_hand_catches_hour_hand :
  let relative_speed := minute_hand_speed - hour_hand_speed
  let catch_up_angle := catch_up_time * relative_speed
  catch_up_angle = full_circle ∧ 
  catch_up_time < minutes_per_hour := by
  sorry

#check minute_hand_catches_hour_hand

end NUMINAMATH_CALUDE_minute_hand_catches_hour_hand_l2504_250466


namespace NUMINAMATH_CALUDE_quadratic_radical_always_nonnegative_l2504_250438

theorem quadratic_radical_always_nonnegative (x : ℝ) : x^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_always_nonnegative_l2504_250438


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2504_250439

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.4 → p_black = 0.25 → p_red + p_black + p_white = 1 → p_white = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2504_250439


namespace NUMINAMATH_CALUDE_benny_comic_books_l2504_250464

theorem benny_comic_books (x : ℚ) : 
  (3/4 * (2/5 * x + 12) + 18 = 72) → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_benny_comic_books_l2504_250464


namespace NUMINAMATH_CALUDE_acid_mixture_concentration_l2504_250484

/-- Proves that mixing 1.2 L of 10% acid solution with 0.8 L of 5% acid solution 
    results in a 2 L solution with 8% acid concentration -/
theorem acid_mixture_concentration : 
  let total_volume : Real := 2
  let volume_10_percent : Real := 1.2
  let volume_5_percent : Real := total_volume - volume_10_percent
  let concentration_10_percent : Real := 10 / 100
  let concentration_5_percent : Real := 5 / 100
  let total_acid : Real := 
    volume_10_percent * concentration_10_percent + 
    volume_5_percent * concentration_5_percent
  let final_concentration : Real := (total_acid / total_volume) * 100
  final_concentration = 8 := by sorry

end NUMINAMATH_CALUDE_acid_mixture_concentration_l2504_250484


namespace NUMINAMATH_CALUDE_handkerchief_usage_per_day_l2504_250482

/-- Proves that given square handkerchiefs of 25 cm × 25 cm and total fabric usage of 3 m² over 8 days, 
    the number of handkerchiefs used per day is 6. -/
theorem handkerchief_usage_per_day 
  (handkerchief_side : ℝ) 
  (total_fabric_area : ℝ) 
  (days : ℕ) 
  (h1 : handkerchief_side = 25) 
  (h2 : total_fabric_area = 3) 
  (h3 : days = 8) : 
  (total_fabric_area * 10000) / (handkerchief_side ^ 2 * days) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_handkerchief_usage_per_day_l2504_250482


namespace NUMINAMATH_CALUDE_total_tickets_correct_l2504_250422

/-- The total number of tickets sold at University Theater -/
def total_tickets : ℕ := 510

/-- The price of an adult ticket -/
def adult_price : ℕ := 21

/-- The price of a senior citizen ticket -/
def senior_price : ℕ := 15

/-- The number of senior citizen tickets sold -/
def senior_tickets : ℕ := 327

/-- The total receipts from ticket sales -/
def total_receipts : ℕ := 8748

/-- Theorem stating that the total number of tickets sold is correct -/
theorem total_tickets_correct :
  ∃ (adult_tickets : ℕ),
    total_tickets = adult_tickets + senior_tickets ∧
    total_receipts = adult_tickets * adult_price + senior_tickets * senior_price :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_correct_l2504_250422


namespace NUMINAMATH_CALUDE_solution_implies_m_minus_n_equals_negative_three_l2504_250429

theorem solution_implies_m_minus_n_equals_negative_three :
  ∀ m n : ℤ,
  (3 * (-2) + 2 * 1 = m) →
  (n * (-2) - 1 = 1) →
  m - n = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_minus_n_equals_negative_three_l2504_250429


namespace NUMINAMATH_CALUDE_fraction_simplification_l2504_250471

theorem fraction_simplification (x : ℝ) : (2*x - 3) / 4 + (3*x + 4) / 3 = (18*x + 7) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2504_250471


namespace NUMINAMATH_CALUDE_road_repair_time_l2504_250496

/-- 
Theorem: Time to repair a road with two teams working simultaneously
Given:
- Team A can repair the entire road in 3 hours
- Team B can repair the entire road in 6 hours
- Both teams work simultaneously from opposite ends
Prove: The time taken to complete the repair is 2 hours
-/
theorem road_repair_time (team_a_time team_b_time : ℝ) 
  (ha : team_a_time = 3)
  (hb : team_b_time = 6) :
  (1 / team_a_time + 1 / team_b_time) * 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_time_l2504_250496


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l2504_250437

def S : Set ℝ := {x | -2 < x ∧ x < 5}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

theorem subset_implies_a_range (a : ℝ) (h : S ⊆ P a) : -5 ≤ a ∧ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l2504_250437


namespace NUMINAMATH_CALUDE_square_number_correct_l2504_250436

/-- The number in the square with coordinates (m, n) -/
def square_number (m n : ℕ) : ℕ :=
  ((m + n - 2) * (m + n - 1)) / 2 + n

/-- Theorem: The square_number function correctly calculates the number
    in the square with coordinates (m, n) for positive integers m and n -/
theorem square_number_correct (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  square_number m n = ((m + n - 2) * (m + n - 1)) / 2 + n :=
by sorry

end NUMINAMATH_CALUDE_square_number_correct_l2504_250436


namespace NUMINAMATH_CALUDE_bob_sandwich_combinations_l2504_250418

/-- Represents the number of sandwich combinations Bob can order -/
def bobSandwichCombinations : ℕ :=
  let totalBreads : ℕ := 5
  let totalMeats : ℕ := 7
  let totalCheeses : ℕ := 6
  let turkeyCombos : ℕ := totalBreads -- Turkey/Swiss combinations
  let roastBeefRyeCombos : ℕ := totalCheeses -- Roast beef/Rye combinations
  let roastBeefSwissCombos : ℕ := totalBreads - 1 -- Roast beef/Swiss combinations (excluding Rye)
  let totalCombinations : ℕ := totalBreads * totalMeats * totalCheeses
  let forbiddenCombinations : ℕ := turkeyCombos + roastBeefRyeCombos + roastBeefSwissCombos
  totalCombinations - forbiddenCombinations

/-- Theorem stating that Bob can order exactly 194 different sandwiches -/
theorem bob_sandwich_combinations : bobSandwichCombinations = 194 := by
  sorry

end NUMINAMATH_CALUDE_bob_sandwich_combinations_l2504_250418


namespace NUMINAMATH_CALUDE_price_restoration_percentage_l2504_250402

theorem price_restoration_percentage (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.85 * original_price
  let restoration_factor := original_price / reduced_price
  let percentage_increase := (restoration_factor - 1) * 100
  ∃ ε > 0, abs (percentage_increase - 17.65) < ε :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_percentage_l2504_250402


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_24_l2504_250461

/-- A number is a four-digit number if it's greater than or equal to 1000 and less than 10000 -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- A number is divisible by another number if the remainder of their division is zero -/
def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem smallest_four_digit_divisible_by_24 :
  is_four_digit 1104 ∧ 
  is_divisible_by 1104 24 ∧ 
  ∀ n : ℕ, is_four_digit n → is_divisible_by n 24 → 1104 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_24_l2504_250461


namespace NUMINAMATH_CALUDE_jakes_comic_books_l2504_250499

theorem jakes_comic_books (jake_books : ℕ) (brother_books : ℕ) : 
  brother_books = jake_books + 15 →
  jake_books + brother_books = 87 →
  jake_books = 36 := by
sorry

end NUMINAMATH_CALUDE_jakes_comic_books_l2504_250499


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2504_250434

theorem youngest_sibling_age (youngest_age : ℕ) : 
  (youngest_age + (youngest_age + 4) + (youngest_age + 5) + (youngest_age + 7)) / 4 = 21 →
  youngest_age = 17 := by
sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2504_250434


namespace NUMINAMATH_CALUDE_complex_point_on_line_l2504_250428

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 + Complex.I)
  let x : ℝ := z.re
  let y : ℝ := z.im
  (x - y + 1 = 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l2504_250428


namespace NUMINAMATH_CALUDE_stack_surface_area_l2504_250431

/-- Calculates the external surface area of a stack of cubes -/
def external_surface_area (volumes : List ℕ) : ℕ :=
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  let adjusted_areas := surface_areas.zip side_lengths
    |> List.map (fun (area, s) => area - s^2)
  adjusted_areas.sum + 6 * (volumes.head!^(1/3))^2

/-- The volumes of the cubes in the stack -/
def cube_volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

theorem stack_surface_area :
  external_surface_area cube_volumes = 1021 := by
  sorry

end NUMINAMATH_CALUDE_stack_surface_area_l2504_250431


namespace NUMINAMATH_CALUDE_special_function_at_one_l2504_250490

/-- A function satisfying certain properties on positive real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f (1 / x) = x * f x) ∧
  (∀ x > 0, ∀ y > 0, f x + f y = x + y + f (x * y))

/-- The value of f(1) for a function satisfying the special properties -/
theorem special_function_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_l2504_250490


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2504_250412

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 2*x - 7 → x ≥ 8 ∧ 8 < 2*8 - 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2504_250412


namespace NUMINAMATH_CALUDE_first_triple_winner_lcm_of_prizes_first_triple_winner_is_900_l2504_250475

theorem first_triple_winner (n : ℕ) : 
  (n % 25 = 0 ∧ n % 36 = 0 ∧ n % 45 = 0) → n ≥ 900 :=
by sorry

theorem lcm_of_prizes : Nat.lcm (Nat.lcm 25 36) 45 = 900 :=
by sorry

theorem first_triple_winner_is_900 : 
  900 % 25 = 0 ∧ 900 % 36 = 0 ∧ 900 % 45 = 0 :=
by sorry

end NUMINAMATH_CALUDE_first_triple_winner_lcm_of_prizes_first_triple_winner_is_900_l2504_250475


namespace NUMINAMATH_CALUDE_lcm_24_30_40_l2504_250442

theorem lcm_24_30_40 : Nat.lcm (Nat.lcm 24 30) 40 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_30_40_l2504_250442


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2504_250458

theorem complex_power_magnitude : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 2) ^ 6) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2504_250458


namespace NUMINAMATH_CALUDE_perimeter_increase_first_to_fourth_l2504_250406

/-- Calculates the perimeter of an equilateral triangle given its side length -/
def trianglePerimeter (side : ℝ) : ℝ := 3 * side

/-- Calculates the side length of the nth triangle in the sequence -/
def nthTriangleSide (n : ℕ) : ℝ :=
  3 * (1.6 ^ n)

/-- Theorem stating the percent increase in perimeter from the first to the fourth triangle -/
theorem perimeter_increase_first_to_fourth :
  let first_perimeter := trianglePerimeter 3
  let fourth_perimeter := trianglePerimeter (nthTriangleSide 3)
  (fourth_perimeter - first_perimeter) / first_perimeter * 100 = 309.6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_increase_first_to_fourth_l2504_250406


namespace NUMINAMATH_CALUDE_donnys_spending_l2504_250407

/-- Donny's spending on Thursday given his savings from Monday to Wednesday -/
theorem donnys_spending (monday_savings : ℕ) (tuesday_savings : ℕ) (wednesday_savings : ℕ)
  (h1 : monday_savings = 15)
  (h2 : tuesday_savings = 28)
  (h3 : wednesday_savings = 13) :
  (monday_savings + tuesday_savings + wednesday_savings) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_donnys_spending_l2504_250407


namespace NUMINAMATH_CALUDE_vector_at_t_6_l2504_250424

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem vector_at_t_6 (h0 : line_vector 0 = (2, -1, 3))
                      (h4 : line_vector 4 = (6, 7, -1)) :
  line_vector 6 = (8, 11, -3) := by sorry

end NUMINAMATH_CALUDE_vector_at_t_6_l2504_250424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2504_250426

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the sum of specific terms in the arithmetic sequence -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_roots : a 5 ^ 2 - 6 * a 5 - 1 = 0 ∧ a 13 ^ 2 - 6 * a 13 - 1 = 0) :
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2504_250426


namespace NUMINAMATH_CALUDE_board_numbers_transformation_impossibility_of_returning_to_original_numbers_l2504_250483

theorem board_numbers_transformation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a - b / 2) ^ 2 + (b + a / 2) ^ 2 > a ^ 2 + b ^ 2 := by
  sorry

theorem impossibility_of_returning_to_original_numbers :
  ∀ (numbers : List ℝ), 
  (∀ n ∈ numbers, n ≠ 0) →
  ∃ (new_numbers : List ℝ),
  (new_numbers.length = numbers.length) ∧
  (List.sum (List.map (λ x => x^2) new_numbers) > List.sum (List.map (λ x => x^2) numbers)) := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_transformation_impossibility_of_returning_to_original_numbers_l2504_250483


namespace NUMINAMATH_CALUDE_fraction_transformation_l2504_250478

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : -a / (a - b) = a / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2504_250478


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2504_250485

theorem cycle_selling_price (cost_price : ℝ) (gain_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 900 →
  gain_percentage = 22.22222222222222 →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  selling_price = 1100 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2504_250485


namespace NUMINAMATH_CALUDE_not_sum_of_two_squares_l2504_250487

theorem not_sum_of_two_squares (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ (a b : ℤ), n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_two_squares_l2504_250487


namespace NUMINAMATH_CALUDE_class_size_l2504_250470

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 24) :
  french + german - both + neither = 78 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2504_250470


namespace NUMINAMATH_CALUDE_sets_partition_integers_l2504_250451

theorem sets_partition_integers (A B : Set ℤ) 
  (h1 : A ∪ B = (Set.univ : Set ℤ))
  (h2 : ∀ x : ℤ, x ∈ A → x - 1 ∈ B)
  (h3 : ∀ x y : ℤ, x ∈ B → y ∈ B → x + y ∈ A) :
  A = {x : ℤ | ∃ k : ℤ, x = 2 * k} ∧ 
  B = {x : ℤ | ∃ k : ℤ, x = 2 * k + 1} :=
by sorry

end NUMINAMATH_CALUDE_sets_partition_integers_l2504_250451


namespace NUMINAMATH_CALUDE_sanctuary_swamps_count_l2504_250456

/-- The number of different reptiles in each swamp -/
def reptiles_per_swamp : ℕ := 356

/-- The total number of reptiles living in the swamp areas -/
def total_reptiles : ℕ := 1424

/-- The number of swamps in the sanctuary -/
def number_of_swamps : ℕ := total_reptiles / reptiles_per_swamp

theorem sanctuary_swamps_count :
  number_of_swamps = 4 :=
by sorry

end NUMINAMATH_CALUDE_sanctuary_swamps_count_l2504_250456


namespace NUMINAMATH_CALUDE_prob_at_least_7_heads_in_9_flips_is_correct_l2504_250404

/-- The probability of getting at least 7 heads in 9 flips of a fair coin -/
def prob_at_least_7_heads_in_9_flips : ℚ :=
  46 / 512

/-- Theorem stating that the probability of getting at least 7 heads in 9 flips of a fair coin is 46/512 -/
theorem prob_at_least_7_heads_in_9_flips_is_correct :
  prob_at_least_7_heads_in_9_flips = 46 / 512 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_7_heads_in_9_flips_is_correct_l2504_250404


namespace NUMINAMATH_CALUDE_duke_three_pointers_l2504_250448

/-- The number of additional three-pointers Duke scored in the final game compared to his normal amount -/
def additional_three_pointers (
  points_to_tie : ℕ
  ) (points_over_record : ℕ
  ) (old_record : ℕ
  ) (free_throws : ℕ
  ) (regular_baskets : ℕ
  ) (normal_three_pointers : ℕ
  ) : ℕ :=
  let total_points := points_to_tie + points_over_record
  let points_from_free_throws := free_throws * 1
  let points_from_regular_baskets := regular_baskets * 2
  let points_from_three_pointers := total_points - (points_from_free_throws + points_from_regular_baskets)
  let three_pointers_scored := points_from_three_pointers / 3
  three_pointers_scored - normal_three_pointers

theorem duke_three_pointers :
  additional_three_pointers 17 5 257 5 4 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_duke_three_pointers_l2504_250448


namespace NUMINAMATH_CALUDE_work_completion_l2504_250473

/-- Given a piece of work that requires 400 man-days to complete,
    prove that if it takes 26.666666666666668 days for a group of men to complete,
    then the number of men in that group is 15. -/
theorem work_completion (total_man_days : ℝ) (days_to_complete : ℝ) (num_men : ℝ) :
  total_man_days = 400 →
  days_to_complete = 26.666666666666668 →
  num_men * days_to_complete = total_man_days →
  num_men = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_l2504_250473


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l2504_250401

theorem sally_pokemon_cards (x : ℕ) : 
  x + 41 - 20 = 48 → x = 27 := by
sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l2504_250401


namespace NUMINAMATH_CALUDE_factorization_theorem_l2504_250425

theorem factorization_theorem (m n : ℝ) : 2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l2504_250425


namespace NUMINAMATH_CALUDE_shoe_pairs_problem_l2504_250477

theorem shoe_pairs_problem (n : ℕ) (h : n > 0) :
  (1 : ℚ) / (2 * n - 1 : ℚ) = 1 / 5 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_shoe_pairs_problem_l2504_250477


namespace NUMINAMATH_CALUDE_max_time_sum_of_digits_is_19_l2504_250498

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≤ 23
  minute_valid : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits for any Time24 -/
def maxTimeSumOfDigits : Nat :=
  19

theorem max_time_sum_of_digits_is_19 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxTimeSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_max_time_sum_of_digits_is_19_l2504_250498


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l2504_250497

theorem distance_between_complex_points : 
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l2504_250497


namespace NUMINAMATH_CALUDE_circle_graph_proportion_l2504_250423

theorem circle_graph_proportion (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) :
  (y = 3.6 * x) ↔ (y / 360 = x / 100) :=
by sorry

end NUMINAMATH_CALUDE_circle_graph_proportion_l2504_250423


namespace NUMINAMATH_CALUDE_garden_area_increase_l2504_250450

/-- Given a rectangular garden with dimensions 60 feet by 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)
  let square_side : ℝ := rectangle_perimeter / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let square_area : ℝ := square_side ^ 2
  square_area - rectangle_area = 400 := by
sorry

end NUMINAMATH_CALUDE_garden_area_increase_l2504_250450


namespace NUMINAMATH_CALUDE_accumulator_implies_limit_in_segment_l2504_250465

/-- A sequence is a function from natural numbers to real numbers -/
def Sequence := ℕ → ℝ

/-- A segment [a, b] is an accumulator for a sequence if infinitely many terms of the sequence lie within [a, b] -/
def IsAccumulator (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a ≤ s n ∧ s n ≤ b

/-- The limit of a sequence, if it exists -/
def HasLimit (s : Sequence) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |s n - L| < ε

theorem accumulator_implies_limit_in_segment (s : Sequence) (a b L : ℝ) :
  IsAccumulator s a b → HasLimit s L → a ≤ L ∧ L ≤ b :=
by sorry


end NUMINAMATH_CALUDE_accumulator_implies_limit_in_segment_l2504_250465


namespace NUMINAMATH_CALUDE_larger_integer_proof_l2504_250419

theorem larger_integer_proof (x y : ℕ) (h1 : y = 4 * x) (h2 : (x + 6) / y = 1 / 3) : y = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l2504_250419


namespace NUMINAMATH_CALUDE_factor_polynomial_l2504_250460

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 250 * x^13 = 25 * x^7 * (3 - 10 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2504_250460


namespace NUMINAMATH_CALUDE_impossibleToReachOpposite_l2504_250469

/-- Represents the color of a point -/
inductive Color
| White
| Black

/-- Represents a point on the circle -/
structure Point where
  position : Fin 2022
  color : Color

/-- The type of operation that can be performed -/
inductive Operation
| FlipAdjacent (i : Fin 2022)
| FlipWithGap (i : Fin 2022)

/-- The configuration of all points on the circle -/
def Configuration := Fin 2022 → Color

/-- Apply an operation to a configuration -/
def applyOperation (config : Configuration) (op : Operation) : Configuration :=
  sorry

/-- The initial configuration with one black point and others white -/
def initialConfig : Configuration :=
  sorry

/-- Check if a configuration is the opposite of the initial configuration -/
def isOppositeConfig (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to reach the opposite configuration -/
theorem impossibleToReachOpposite : 
  ∀ (ops : List Operation), 
    ¬(isOppositeConfig (ops.foldl applyOperation initialConfig)) :=
  sorry

end NUMINAMATH_CALUDE_impossibleToReachOpposite_l2504_250469
