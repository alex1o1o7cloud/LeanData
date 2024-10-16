import Mathlib

namespace NUMINAMATH_CALUDE_factors_of_1320_l3053_305334

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_1320 : number_of_factors 1320 = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l3053_305334


namespace NUMINAMATH_CALUDE_unique_solution_power_sum_l3053_305344

theorem unique_solution_power_sum (a b c : ℕ) :
  (∀ n : ℕ, a^n + b^n = c^(n+1)) → (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_power_sum_l3053_305344


namespace NUMINAMATH_CALUDE_five_percent_problem_l3053_305390

theorem five_percent_problem (x : ℝ) : (5 / 100) * x = 12.75 → x = 255 := by
  sorry

end NUMINAMATH_CALUDE_five_percent_problem_l3053_305390


namespace NUMINAMATH_CALUDE_coin_regrouping_l3053_305399

/-- The total number of coins remains the same after regrouping -/
theorem coin_regrouping (x : ℕ) : 
  (12 + 17 + 23 + 8 : ℕ) = 60 ∧ 
  x > 0 ∧
  60 % x = 0 →
  60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coin_regrouping_l3053_305399


namespace NUMINAMATH_CALUDE_cos_54_degrees_l3053_305347

theorem cos_54_degrees : Real.cos (54 * π / 180) = (3 - Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l3053_305347


namespace NUMINAMATH_CALUDE_line_segment_properties_l3053_305348

/-- Given a line segment with endpoints (1, 4) and (7, 18), prove properties about its midpoint and slope -/
theorem line_segment_properties :
  let x₁ : ℝ := 1
  let y₁ : ℝ := 4
  let x₂ : ℝ := 7
  let y₂ : ℝ := 18
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
  (midpoint_x + midpoint_y = 15) ∧ (slope = 7 / 3) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_properties_l3053_305348


namespace NUMINAMATH_CALUDE_sin_390_degrees_l3053_305310

theorem sin_390_degrees (h1 : ∀ θ, Real.sin (θ + 2 * Real.pi) = Real.sin θ) 
                        (h2 : Real.sin (Real.pi / 6) = 1 / 2) : 
  Real.sin (13 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l3053_305310


namespace NUMINAMATH_CALUDE_equation_solution_l3053_305356

theorem equation_solution : ∃ x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3053_305356


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l3053_305398

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- The focal distance -/
  c : ℝ
  /-- Assumption that a > b > 0 -/
  h₁ : a > b ∧ b > 0
  /-- Relationship between a, b, and c -/
  h₂ : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.b^2 + y^2 / e.a^2 = 1

/-- Theorem stating the equation of the ellipse given the conditions -/
theorem ellipse_equation_from_conditions
  (e : Ellipse)
  (focus_on_y_axis : e.c = e.a * (1/2))
  (focal_length : 2 * e.c = 8) :
  ellipse_equation e = fun x y ↦ x^2 / 48 + y^2 / 64 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l3053_305398


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3053_305306

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 4 → a^2 > 16) ∧ 
  (∃ a, a^2 > 16 ∧ ¬(a > 4)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3053_305306


namespace NUMINAMATH_CALUDE_gcd_324_135_l3053_305393

theorem gcd_324_135 : Nat.gcd 324 135 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_324_135_l3053_305393


namespace NUMINAMATH_CALUDE_roots_sum_squares_l3053_305332

theorem roots_sum_squares (p q r : ℝ) : 
  (p^3 - 24*p^2 + 50*p - 35 = 0) →
  (q^3 - 24*q^2 + 50*q - 35 = 0) →
  (r^3 - 24*r^2 + 50*r - 35 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 1052 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_squares_l3053_305332


namespace NUMINAMATH_CALUDE_point_A_location_l3053_305354

theorem point_A_location (A : ℝ) : 
  (A + 2 = -2 ∨ A - 2 = -2) → (A = 0 ∨ A = -4) := by
sorry

end NUMINAMATH_CALUDE_point_A_location_l3053_305354


namespace NUMINAMATH_CALUDE_matrix_product_proof_l3053_305378

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 1, 3, -1; 0, 5, 2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 4; -2, 0, 0; 3, 0, -2]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-7, -2, 14; -8, -1, 6; -4, 0, -4]

theorem matrix_product_proof : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_product_proof_l3053_305378


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3053_305316

/-- Definition of an ellipse with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- The major axis of the ellipse lies on the y-axis -/
def major_axis_on_y (m : ℝ) : Prop :=
  m - 2 > 10 - m

/-- The focal distance of the ellipse is 4 -/
def focal_distance_4 (m : ℝ) : Prop :=
  4^2 = 4 * (m - 2) - 4 * (10 - m)

/-- Theorem: For an ellipse with the given properties, m equals 8 -/
theorem ellipse_m_value (m : ℝ) 
  (h1 : is_ellipse m) 
  (h2 : major_axis_on_y m) 
  (h3 : focal_distance_4 m) : 
  m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3053_305316


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l3053_305302

def cost_price : ℝ := 12
def initial_price : ℝ := 20
def initial_quantity : ℝ := 240
def quantity_increase_rate : ℝ := 40

def profit (x : ℝ) : ℝ :=
  (initial_price - cost_price - x) * (initial_quantity + quantity_increase_rate * x)

theorem profit_and_max_profit :
  (∃ x : ℝ, profit x = 1920 ∧ x = 2) ∧
  (∃ x : ℝ, ∀ y : ℝ, profit y ≤ profit x ∧ x = 4) ∧
  (∃ x : ℝ, profit x = 2560 ∧ x = 4) := by sorry

end NUMINAMATH_CALUDE_profit_and_max_profit_l3053_305302


namespace NUMINAMATH_CALUDE_combined_selling_price_theorem_l3053_305319

/-- Calculates the selling price of an article including profit and tax -/
def sellingPrice (cost : ℚ) (profitPercent : ℚ) (taxRate : ℚ) : ℚ :=
  let priceBeforeTax := cost * (1 + profitPercent)
  priceBeforeTax * (1 + taxRate)

/-- Calculates the combined selling price of three articles -/
def combinedSellingPrice (cost1 cost2 cost3 : ℚ) (profit1 profit2 profit3 : ℚ) (taxRate : ℚ) : ℚ :=
  sellingPrice cost1 profit1 taxRate +
  sellingPrice cost2 profit2 taxRate +
  sellingPrice cost3 profit3 taxRate

theorem combined_selling_price_theorem (cost1 cost2 cost3 : ℚ) (profit1 profit2 profit3 : ℚ) (taxRate : ℚ) :
  combinedSellingPrice cost1 cost2 cost3 profit1 profit2 profit3 taxRate =
  sellingPrice 500 (45/100) (12/100) +
  sellingPrice 300 (30/100) (12/100) +
  sellingPrice 1000 (20/100) (12/100) := by
  sorry

end NUMINAMATH_CALUDE_combined_selling_price_theorem_l3053_305319


namespace NUMINAMATH_CALUDE_point_on_line_l3053_305323

/-- Given a line passing through points (3, 6) and (-4, 0), 
    prove that if (x, 10) lies on this line, then x = 23/3 -/
theorem point_on_line (x : ℚ) : 
  (∀ (t : ℚ), (3 + t * (-4 - 3), 6 + t * (0 - 6)) = (x, 10)) → x = 23 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3053_305323


namespace NUMINAMATH_CALUDE_store_price_reduction_l3053_305362

theorem store_price_reduction (original_price : ℝ) (first_reduction : ℝ) : 
  first_reduction > 0 →
  first_reduction < 100 →
  (original_price * (1 - first_reduction / 100) * (1 - 0.14) = 0.774 * original_price) →
  first_reduction = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_price_reduction_l3053_305362


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l3053_305314

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Theorem: The width of the foil-covered prism is 10 inches -/
theorem foil_covered_prism_width : 
  ∀ (inner : PrismDimensions),
    volume inner = 128 →
    inner.width = 2 * inner.length →
    inner.width = 2 * inner.height →
    ∃ (outer : PrismDimensions),
      outer.length = inner.length + 2 ∧
      outer.width = inner.width + 2 ∧
      outer.height = inner.height + 2 ∧
      outer.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l3053_305314


namespace NUMINAMATH_CALUDE_zeros_difference_quadratic_l3053_305372

theorem zeros_difference_quadratic (m : ℝ) : 
  (∃ α β : ℝ, 2 * α^2 - m * α - 8 = 0 ∧ 
              2 * β^2 - m * β - 8 = 0 ∧ 
              α - β = m - 1) ↔ 
  (m = 6 ∨ m = -10/3) := by
sorry

end NUMINAMATH_CALUDE_zeros_difference_quadratic_l3053_305372


namespace NUMINAMATH_CALUDE_total_posters_proof_l3053_305364

def mario_posters : ℕ := 18
def samantha_extra_posters : ℕ := 15

theorem total_posters_proof :
  mario_posters + (mario_posters + samantha_extra_posters) = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_posters_proof_l3053_305364


namespace NUMINAMATH_CALUDE_camel_cost_l3053_305324

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 130000

/-- The theorem stating that given the problem conditions, the cost of a camel is 5200 rupees -/
theorem camel_cost (costs : AnimalCosts) :
  problem_conditions costs → costs.camel = 5200 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l3053_305324


namespace NUMINAMATH_CALUDE_simplify_fourth_root_exponent_sum_l3053_305396

theorem simplify_fourth_root_exponent_sum (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ (k : ℝ) (n m : ℕ), (48 * a^5 * b^8 * c^14)^(1/4) = k * b^n * c^m ∧ n + m = 5 :=
sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_exponent_sum_l3053_305396


namespace NUMINAMATH_CALUDE_solution_set_f_gt_g_range_of_a_l3053_305359

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := |2*x - 2|

-- Theorem for the solution set of f(x) > g(x)
theorem solution_set_f_gt_g :
  {x : ℝ | f x > g x} = {x : ℝ | 2/3 < x ∧ x < 2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, 2 * f x + g x > a * x + 1} = {a : ℝ | -4 ≤ a ∧ a < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_g_range_of_a_l3053_305359


namespace NUMINAMATH_CALUDE_tuesday_monday_ratio_l3053_305315

/-- Represents the number of visitors to a library on different days of the week -/
structure LibraryVisitors where
  monday : ℕ
  tuesday : ℕ
  remainingDaysAverage : ℕ
  totalWeek : ℕ

/-- The ratio of Tuesday visitors to Monday visitors is 2:1 -/
theorem tuesday_monday_ratio (v : LibraryVisitors) 
  (h1 : v.monday = 50)
  (h2 : v.remainingDaysAverage = 20)
  (h3 : v.totalWeek = 250)
  (h4 : v.totalWeek = v.monday + v.tuesday + 5 * v.remainingDaysAverage) :
  v.tuesday / v.monday = 2 := by
  sorry

#check tuesday_monday_ratio

end NUMINAMATH_CALUDE_tuesday_monday_ratio_l3053_305315


namespace NUMINAMATH_CALUDE_bird_cage_problem_l3053_305307

theorem bird_cage_problem (B : ℚ) : 
  (B > 0) →                         -- Ensure positive number of birds
  (B * (2/3) * (3/5) * (1/3) = 60)  -- Remaining birds after three stages equal 60
  ↔ 
  (B = 450) :=                      -- Total initial number of birds is 450
by sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l3053_305307


namespace NUMINAMATH_CALUDE_root_equation_value_l3053_305376

theorem root_equation_value (m : ℝ) (h : m^2 - 3*m + 1 = 0) : 
  (m - 3)^2 + (m + 2)*(m - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3053_305376


namespace NUMINAMATH_CALUDE_intersection_point_l3053_305337

/-- The quadratic function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The point (0, -1) -/
def point : ℝ × ℝ := (0, -1)

/-- Theorem: The point (0, -1) is the intersection point of y = x^2 - 1 with the y-axis -/
theorem intersection_point :
  (point.1 = 0) ∧ 
  (point.2 = f point.1) ∧ 
  (∀ x : ℝ, x ≠ point.1 → (x, f x) ≠ point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l3053_305337


namespace NUMINAMATH_CALUDE_count_common_divisors_36_90_l3053_305305

def divisors_of_both (a b : ℕ) : Finset ℕ :=
  (Finset.range a).filter (fun x => x > 0 ∧ a % x = 0 ∧ b % x = 0)

theorem count_common_divisors_36_90 : (divisors_of_both 36 90).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_36_90_l3053_305305


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l3053_305361

theorem distribute_and_simplify (a : ℝ) : -3 * a^2 * (4*a - 3) = -12*a^3 + 9*a^2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l3053_305361


namespace NUMINAMATH_CALUDE_line_through_point_l3053_305336

theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 2) * (-5) = b + 3) → b = 7/3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_l3053_305336


namespace NUMINAMATH_CALUDE_tangent_slope_at_1_l3053_305326

/-- The function f(x) = (x-2)(x^2+c) has an extremum at x=2 -/
def has_extremum_at_2 (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2 ∨ f x ≥ f 2

/-- The main theorem -/
theorem tangent_slope_at_1 (c : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - 2) * (x^2 + c)
  has_extremum_at_2 f c → (deriv f) 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_1_l3053_305326


namespace NUMINAMATH_CALUDE_children_bridge_problem_l3053_305355

/-- The problem of three children crossing a bridge --/
theorem children_bridge_problem (bridge_capacity : ℝ) (kelly_weight : ℝ) :
  bridge_capacity = 100 →
  kelly_weight = 34 →
  ∃ (megan_weight : ℝ) (mike_weight : ℝ),
    kelly_weight = 0.85 * megan_weight ∧
    mike_weight = megan_weight + 5 ∧
    kelly_weight + megan_weight + mike_weight - bridge_capacity = 19 :=
by sorry

end NUMINAMATH_CALUDE_children_bridge_problem_l3053_305355


namespace NUMINAMATH_CALUDE_slope_and_intercept_of_3x_plus_2_l3053_305349

/-- Given a linear function y = mx + b, the slope is m and the y-intercept is b -/
def linear_function (m b : ℝ) : ℝ → ℝ := λ x ↦ m * x + b

theorem slope_and_intercept_of_3x_plus_2 :
  ∃ (f : ℝ → ℝ), f = linear_function 3 2 ∧ 
  (∀ x y : ℝ, f x - f y = 3 * (x - y)) ∧
  f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_and_intercept_of_3x_plus_2_l3053_305349


namespace NUMINAMATH_CALUDE_ned_games_problem_l3053_305365

theorem ned_games_problem (initial_games : ℕ) : 
  (3/4 : ℚ) * (2/3 : ℚ) * initial_games = 6 → initial_games = 12 := by
  sorry

end NUMINAMATH_CALUDE_ned_games_problem_l3053_305365


namespace NUMINAMATH_CALUDE_first_car_speed_l3053_305394

/-- Proves that the speed of the first car is 54 miles per hour given the conditions of the problem -/
theorem first_car_speed (total_distance : ℝ) (second_car_speed : ℝ) (time_difference : ℝ) (total_time : ℝ)
  (h1 : total_distance = 80)
  (h2 : second_car_speed = 60)
  (h3 : time_difference = 1/6)
  (h4 : total_time = 1.5) :
  ∃ (first_car_speed : ℝ), first_car_speed = 54 ∧
    second_car_speed * total_time = first_car_speed * (total_time + time_difference) := by
  sorry


end NUMINAMATH_CALUDE_first_car_speed_l3053_305394


namespace NUMINAMATH_CALUDE_stating_dual_polyhedra_equal_spheres_l3053_305343

/-- Represents a regular polyhedron with its associated sphere radii -/
structure RegularPolyhedron where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  p : ℝ  -- radius of half-inscribed sphere

/-- Represents a pair of dual regular polyhedra -/
structure DualPolyhedraPair where
  T1 : RegularPolyhedron
  T2 : RegularPolyhedron

/-- 
Theorem stating that for dual regular polyhedra with equal inscribed spheres,
their circumscribed spheres are also equal.
-/
theorem dual_polyhedra_equal_spheres (pair : DualPolyhedraPair) :
  pair.T1.r = pair.T2.r → pair.T1.R = pair.T2.R := by
  sorry


end NUMINAMATH_CALUDE_stating_dual_polyhedra_equal_spheres_l3053_305343


namespace NUMINAMATH_CALUDE_monotonicity_indeterminate_l3053_305374

theorem monotonicity_indeterminate 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, x ∈ Set.Icc (-1) 2 → f x ≠ 0) 
  (h_inequality : f (-1/2) < f 1) : 
  ¬ (∀ x y, x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → x < y → f x < f y) ∧ 
  ¬ (∀ x y, x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_indeterminate_l3053_305374


namespace NUMINAMATH_CALUDE_min_bad_work_percentage_l3053_305346

/-- Represents the grading system for student work -/
inductive Grade
  | Accepted
  | NotAccepted

/-- Represents the true quality of student work -/
inductive Quality
  | Good
  | Bad

/-- Neural network classification result -/
def neuralNetworkClassify (work : Quality) : Grade :=
  sorry

/-- Expert classification result -/
def expertClassify (work : Quality) : Grade :=
  sorry

/-- Probability of neural network error -/
def neuralNetworkErrorRate : ℝ := 0.1

/-- Probability of work being bad -/
def badWorkProbability : ℝ := 0.2

/-- Probability of work being good -/
def goodWorkProbability : ℝ := 1 - badWorkProbability

/-- Percentage of work rechecked by experts -/
def recheckedPercentage : ℝ :=
  badWorkProbability * (1 - neuralNetworkErrorRate) + goodWorkProbability * neuralNetworkErrorRate

/-- Theorem: The minimum percentage of bad works among those rechecked by experts is 66% -/
theorem min_bad_work_percentage :
  (badWorkProbability * (1 - neuralNetworkErrorRate)) / recheckedPercentage ≥ 0.66 := by
  sorry

end NUMINAMATH_CALUDE_min_bad_work_percentage_l3053_305346


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3053_305391

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

#check opposite_of_negative_fraction

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3053_305391


namespace NUMINAMATH_CALUDE_bus_trip_speed_l3053_305301

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 210 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (original_speed : ℝ),
    distance / original_speed - time_decrease = distance / (original_speed + speed_increase) ∧
    original_speed = 30 :=
by sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l3053_305301


namespace NUMINAMATH_CALUDE_two_x_plus_y_equals_five_l3053_305368

theorem two_x_plus_y_equals_five (x y : ℝ) 
  (eq1 : 7 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  2 * x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_two_x_plus_y_equals_five_l3053_305368


namespace NUMINAMATH_CALUDE_adult_meal_cost_l3053_305367

def restaurant_problem (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) : Prop :=
  let num_adults := total_people - num_kids
  let cost_per_adult := total_cost / num_adults
  cost_per_adult = 7

theorem adult_meal_cost :
  restaurant_problem 13 9 28 := by sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l3053_305367


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l3053_305309

/-- Given that Rachelle uses 4 pounds of meat to make 10 hamburgers,
    prove that she needs 12 pounds of meat to make 30 hamburgers. -/
theorem meat_for_hamburgers (meat_for_10 : ℝ) (hamburgers_for_10 : ℕ)
    (meat_for_30 : ℝ) (hamburgers_for_30 : ℕ) :
    meat_for_10 = 4 ∧ hamburgers_for_10 = 10 ∧ hamburgers_for_30 = 30 →
    meat_for_30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l3053_305309


namespace NUMINAMATH_CALUDE_lighthouse_signals_lighthouse_signals_minimum_l3053_305388

theorem lighthouse_signals (x : ℕ) : 
  (x % 15 = 2 ∧ x % 28 = 8) → x ≥ 92 :=
by sorry

theorem lighthouse_signals_minimum : 
  ∃ (x : ℕ), x % 15 = 2 ∧ x % 28 = 8 ∧ x = 92 :=
by sorry

end NUMINAMATH_CALUDE_lighthouse_signals_lighthouse_signals_minimum_l3053_305388


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3053_305328

/-- A point P with coordinates (2m, m+8) lies on the y-axis if and only if its coordinates are (0, 8) -/
theorem point_on_y_axis (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (2*m, m+8) ∧ P.1 = 0) ↔ (∃ (P : ℝ × ℝ), P = (0, 8)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3053_305328


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3053_305397

theorem opposite_of_negative_fraction :
  -(-(1 / 2024)) = 1 / 2024 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3053_305397


namespace NUMINAMATH_CALUDE_data_set_properties_l3053_305377

def data_set : List ℕ := [3, 5, 4, 5, 6, 7]

def mode (list : List ℕ) : ℕ := sorry

def median (list : List ℕ) : ℚ := sorry

def mean (list : List ℕ) : ℚ := sorry

theorem data_set_properties :
  mode data_set = 5 ∧ 
  median data_set = 5 ∧ 
  mean data_set = 5 := by sorry

end NUMINAMATH_CALUDE_data_set_properties_l3053_305377


namespace NUMINAMATH_CALUDE_solve_for_k_l3053_305375

theorem solve_for_k : ∃ k : ℚ, (4 * k - 3 * (-1) = 2) ∧ (k = -1/4) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l3053_305375


namespace NUMINAMATH_CALUDE_calculate_expression_l3053_305387

theorem calculate_expression : (2023 - Real.pi) ^ 0 - (1 / 4)⁻¹ + |(-2)| + Real.sqrt 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3053_305387


namespace NUMINAMATH_CALUDE_water_speed_calculation_l3053_305383

/-- 
Given a person who can swim at 4 km/h in still water and takes 7 hours to swim 14 km against a current,
prove that the speed of the water is 2 km/h.
-/
theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) :
  still_water_speed = 4 →
  distance = 14 →
  time = 7 →
  ∃ (water_speed : ℝ), water_speed = 2 ∧ still_water_speed - water_speed = distance / time :=
by sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l3053_305383


namespace NUMINAMATH_CALUDE_art_exhibition_tickets_l3053_305363

/-- Calculates the total number of tickets sold given the conditions -/
def totalTicketsSold (advancedPrice : ℕ) (doorPrice : ℕ) (totalCollected : ℕ) (advancedSold : ℕ) : ℕ :=
  let doorSold := (totalCollected - advancedPrice * advancedSold) / doorPrice
  advancedSold + doorSold

/-- Theorem stating that under the given conditions, 165 tickets were sold in total -/
theorem art_exhibition_tickets :
  totalTicketsSold 8 14 1720 100 = 165 := by
  sorry

#eval totalTicketsSold 8 14 1720 100

end NUMINAMATH_CALUDE_art_exhibition_tickets_l3053_305363


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3053_305322

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3053_305322


namespace NUMINAMATH_CALUDE_cube_root_plus_power_plus_abs_eq_zero_l3053_305313

theorem cube_root_plus_power_plus_abs_eq_zero :
  - Real.rpow 8 (1/3) + (2016 : ℝ)^0 + |1 - Real.sqrt 4| = 0 := by sorry

end NUMINAMATH_CALUDE_cube_root_plus_power_plus_abs_eq_zero_l3053_305313


namespace NUMINAMATH_CALUDE_min_value_symmetric_circle_l3053_305360

/-- Given a circle and a line, if the circle is symmetric about the line,
    then the minimum value of 1/a + 2/b is 3 -/
theorem min_value_symmetric_circle (x y a b : ℝ) :
  x^2 + y^2 - 2*x - 4*y + 3 = 0 →
  a > 0 →
  b > 0 →
  a*x + b*y = 3 →
  (∃ (c : ℝ), c > 0 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a'*x + b'*y = 3 → 1/a' + 2/b' ≥ c) →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀*x + b₀*y = 3 ∧ 1/a₀ + 2/b₀ = 3) :=
by sorry


end NUMINAMATH_CALUDE_min_value_symmetric_circle_l3053_305360


namespace NUMINAMATH_CALUDE_ellipse_equation_l3053_305335

/-- Given an ellipse sharing foci with the hyperbola 2x^2 - 2y^2 = 1 
    and passing through the point (1, -3/2), 
    prove that the equation of the ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x₀ y₀ : ℝ), (x₀^2 / a^2 + y₀^2 / b^2 = 1) ↔ 
      ((x₀ + 1)^2 + y₀^2)^(1/2) + ((x₀ - 1)^2 + y₀^2)^(1/2) = 2*a)) →
  (1^2 / a^2 + (-3/2)^2 / b^2 = 1) →
  (x^2 / 4 + y^2 / 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3053_305335


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3053_305338

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3053_305338


namespace NUMINAMATH_CALUDE_kyles_rose_expense_l3053_305321

def roses_last_year : ℕ := 12
def roses_this_year : ℕ := roses_last_year / 2
def roses_needed : ℕ := 2 * roses_last_year
def price_per_rose : ℕ := 3

theorem kyles_rose_expense : 
  (roses_needed - roses_this_year) * price_per_rose = 54 := by sorry

end NUMINAMATH_CALUDE_kyles_rose_expense_l3053_305321


namespace NUMINAMATH_CALUDE_true_proposition_l3053_305366

-- Define proposition p
def p : Prop := ∀ x : ℝ, (3 : ℝ) ^ x > 0

-- Define proposition q
def q : Prop := (∀ x : ℝ, x > 0 → x > 1) ∧ ¬(∀ x : ℝ, x > 1 → x > 0)

-- Theorem statement
theorem true_proposition : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_true_proposition_l3053_305366


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3053_305371

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3053_305371


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l3053_305380

theorem gcd_special_numbers :
  let m : ℕ := 55555555
  let n : ℕ := 111111111
  Nat.gcd m n = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l3053_305380


namespace NUMINAMATH_CALUDE_matrix_determinant_l3053_305341

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 4, 7]

theorem matrix_determinant :
  Matrix.det matrix = 47 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3053_305341


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3053_305352

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x / (x - 1) ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3053_305352


namespace NUMINAMATH_CALUDE_scientific_notation_1742000_l3053_305330

theorem scientific_notation_1742000 :
  ∃ (a : ℝ) (n : ℤ), 1742000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 :=
by
  use 1.742, 6
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1742000_l3053_305330


namespace NUMINAMATH_CALUDE_distance_to_school_l3053_305300

/-- The distance to school given travel conditions -/
theorem distance_to_school : 
  ∀ (total_time speed_to speed_from : ℝ),
  total_time = 1 →
  speed_to = 5 →
  speed_from = 25 →
  ∃ (distance : ℝ),
    distance / speed_to + distance / speed_from = total_time ∧
    distance = 25 / 6 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_l3053_305300


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3053_305351

theorem division_remainder_problem (dividend : Nat) (divisor : Nat) : 
  Prime dividend → Prime divisor → dividend = divisor * 7 + 1054 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3053_305351


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3053_305358

theorem solution_satisfies_system :
  let solutions : List (ℝ × ℝ) := [(5, -3), (5, 3), (-Real.sqrt 118 / 2, 3 * Real.sqrt 2 / 2), (-Real.sqrt 118 / 2, -3 * Real.sqrt 2 / 2)]
  ∀ (x y : ℝ), (x, y) ∈ solutions →
    (x^2 + y^2 = 34 ∧ x - y + Real.sqrt ((x - y) / (x + y)) = 20 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3053_305358


namespace NUMINAMATH_CALUDE_complement_of_union_l3053_305303

def U : Set Nat := {1, 2, 3, 4, 5}
def P : Set Nat := {1, 2, 3}
def Q : Set Nat := {2, 3, 4}

theorem complement_of_union :
  (U \ (P ∪ Q)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3053_305303


namespace NUMINAMATH_CALUDE_potatoes_already_cooked_l3053_305345

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 13
  - Each potato takes 6 minutes to cook
  - It will take 48 minutes to cook the remaining potatoes
  This theorem proves that the number of potatoes already cooked is 5. -/
theorem potatoes_already_cooked 
  (total_potatoes : ℕ) 
  (cooking_time_per_potato : ℕ) 
  (remaining_cooking_time : ℕ) 
  (h1 : total_potatoes = 13)
  (h2 : cooking_time_per_potato = 6)
  (h3 : remaining_cooking_time = 48) :
  total_potatoes - (remaining_cooking_time / cooking_time_per_potato) = 5 :=
by sorry

end NUMINAMATH_CALUDE_potatoes_already_cooked_l3053_305345


namespace NUMINAMATH_CALUDE_payment_calculation_l3053_305395

/-- Calculates the payment per safely delivered bowl -/
def payment_per_bowl (total_bowls : ℕ) (fee : ℚ) (cost_per_damaged : ℚ) 
  (lost_bowls : ℕ) (broken_bowls : ℕ) (total_payment : ℚ) : ℚ :=
  let safely_delivered := total_bowls - lost_bowls - broken_bowls
  (total_payment - fee) / safely_delivered

theorem payment_calculation : 
  let result := payment_per_bowl 638 100 4 12 15 1825
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/100) ∧ |result - (282/100)| < ε := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l3053_305395


namespace NUMINAMATH_CALUDE_parallelogram_height_l3053_305379

theorem parallelogram_height 
  (area : ℝ) 
  (base : ℝ) 
  (h1 : area = 308) 
  (h2 : base = 22) : 
  area / base = 14 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3053_305379


namespace NUMINAMATH_CALUDE_twenty_paise_coins_l3053_305392

theorem twenty_paise_coins (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 500 ∧ 
  total_value = 105 ∧ 
  ∃ (x y : ℕ), x + y = total_coins ∧ 
                (20 : ℚ)/100 * x + (25 : ℚ)/100 * y = total_value →
  x = 400 :=
by sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_l3053_305392


namespace NUMINAMATH_CALUDE_problem_solution_l3053_305320

theorem problem_solution (x : ℝ) (h : |x| = x + 2) :
  19 * x^99 + 3 * x + 27 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3053_305320


namespace NUMINAMATH_CALUDE_smallest_population_with_conditions_l3053_305339

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_population_with_conditions : 
  ∃ n : ℕ, 
    is_perfect_square n ∧ 
    (∃ k : ℕ, n + 150 = k^2 + 1) ∧ 
    is_perfect_square (n + 300) ∧
    n = 144 ∧
    ∀ m : ℕ, m < n → 
      ¬(is_perfect_square m ∧ 
        (∃ k : ℕ, m + 150 = k^2 + 1) ∧ 
        is_perfect_square (m + 300)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_population_with_conditions_l3053_305339


namespace NUMINAMATH_CALUDE_circle_radii_sum_l3053_305357

theorem circle_radii_sum (r R : ℝ) : 
  r > 0 → R > 0 →  -- Radii are positive
  R - r = 5 →  -- Distance between centers
  π * R^2 - π * r^2 = 100 * π →  -- Area between circles
  r + R = 20 := by
sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l3053_305357


namespace NUMINAMATH_CALUDE_animal_weight_comparison_l3053_305386

theorem animal_weight_comparison (chicken_weight duck_weight cow_weight : ℕ) 
  (h1 : chicken_weight = 3)
  (h2 : duck_weight = 6)
  (h3 : cow_weight = 624) :
  (cow_weight / chicken_weight = 208) ∧ (cow_weight / duck_weight = 104) := by
  sorry

end NUMINAMATH_CALUDE_animal_weight_comparison_l3053_305386


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3053_305373

theorem opposite_of_negative_2023 : 
  ∀ x : ℤ, (x + (-2023) = 0) → x = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3053_305373


namespace NUMINAMATH_CALUDE_square_sum_equals_six_l3053_305342

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_six_l3053_305342


namespace NUMINAMATH_CALUDE_trapezium_area_with_triangle_removed_l3053_305329

/-- The area of a trapezium with a right triangle removed -/
theorem trapezium_area_with_triangle_removed
  (e f g h : ℝ)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g)
  (h_pos : 0 < h) :
  let trapezium_area := (e + f) * (g + h)
  let triangle_area := h^2 / 2
  trapezium_area - triangle_area = (e + f) * (g + h) - h^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_with_triangle_removed_l3053_305329


namespace NUMINAMATH_CALUDE_candy_store_spending_correct_l3053_305318

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 345 / 100

/-- Fraction of allowance spent at the arcade -/
def arcade_fraction : ℚ := 3 / 5

/-- Fraction of remaining allowance spent at the toy store -/
def toy_store_fraction : ℚ := 1 / 3

/-- Amount spent at the candy store -/
def candy_store_spending : ℚ := 
  weekly_allowance * (1 - arcade_fraction) * (1 - toy_store_fraction)

theorem candy_store_spending_correct : 
  candy_store_spending = 92 / 100 := by sorry

end NUMINAMATH_CALUDE_candy_store_spending_correct_l3053_305318


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_x_gt_e_l3053_305340

theorem necessary_not_sufficient_condition_for_x_gt_e (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ¬(x > 1 → x > Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_x_gt_e_l3053_305340


namespace NUMINAMATH_CALUDE_circle_configuration_theorem_l3053_305308

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of three circles C, D, and E -/
structure CircleConfiguration where
  C : Circle
  D : Circle
  E : Circle
  C_radius_is_4 : C.radius = 4
  D_internally_tangent_to_C : True  -- This is a placeholder for the tangency condition
  E_internally_tangent_to_C : True  -- This is a placeholder for the tangency condition
  E_externally_tangent_to_D : True  -- This is a placeholder for the tangency condition
  E_tangent_to_diameter : True      -- This is a placeholder for the tangency condition
  D_radius_twice_E : D.radius = 2 * E.radius

theorem circle_configuration_theorem (config : CircleConfiguration) 
  (p q : ℕ) (h : config.D.radius = Real.sqrt p - q) : 
  p + q = 259 := by
  sorry

end NUMINAMATH_CALUDE_circle_configuration_theorem_l3053_305308


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_three_l3053_305385

theorem smallest_four_digit_multiple_of_three :
  ∃ n : ℕ, n = 1002 ∧ 
    (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 3 = 0 → n ≤ m) ∧
    1000 ≤ n ∧ n < 10000 ∧ n % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_three_l3053_305385


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l3053_305350

theorem direct_inverse_variation (c R₁ R₂ S₁ S₂ T₁ T₂ : ℚ) 
  (h1 : R₁ = c * (S₁ / T₁))
  (h2 : R₂ = c * (S₂ / T₂))
  (h3 : R₁ = 2)
  (h4 : T₁ = 1/2)
  (h5 : S₁ = 8)
  (h6 : R₂ = 16)
  (h7 : T₂ = 1/4) :
  S₂ = 32 := by
sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l3053_305350


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3053_305389

theorem trigonometric_inequality : 
  let a := (1/2) * Real.cos (6 * (π/180)) - (Real.sqrt 3 / 2) * Real.sin (6 * (π/180))
  let b := (2 * Real.tan (13 * (π/180))) / (1 + Real.tan (13 * (π/180)) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * (π/180))) / 2)
  a < c ∧ c < b := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3053_305389


namespace NUMINAMATH_CALUDE_unique_number_power_ten_sum_l3053_305327

theorem unique_number_power_ten_sum : ∃! (N : ℕ), 
  N > 0 ∧ 
  (∃ (k : ℕ), N + (Nat.factors N).foldl Nat.lcm 1 = 10^k) ∧
  N = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_power_ten_sum_l3053_305327


namespace NUMINAMATH_CALUDE_triangle_area_l3053_305370

/-- The area of the right triangle formed by the x-axis, the line y = 2, and the line x = 1 + √3y --/
theorem triangle_area : ℝ := by
  -- Define the lines
  let x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}
  let line_y2 : Set (ℝ × ℝ) := {p | p.2 = 2}
  let line_x1_sqrt3y : Set (ℝ × ℝ) := {p | p.1 = 1 + Real.sqrt 3 * p.2}

  -- Define the vertices of the triangle
  let origin : ℝ × ℝ := (0, 0)
  let vertex_on_x_axis : ℝ × ℝ := (1, 0)
  let vertex_on_y_axis : ℝ × ℝ := (0, 2)

  -- Calculate the area of the triangle
  let base : ℝ := vertex_on_x_axis.1 - origin.1
  let height : ℝ := vertex_on_y_axis.2 - origin.2
  let area : ℝ := (1 / 2) * base * height

  -- Prove that the area equals 1
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3053_305370


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3053_305331

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3053_305331


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3053_305369

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

/-- The binary representation of 1110010110₂ -/
def binary_num : List Bool := [true, true, true, false, false, true, false, true, true, false]

/-- The expected quaternary representation of 32112₄ -/
def expected_quaternary : List (Fin 4) := [3, 2, 1, 1, 2]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_num) = expected_quaternary := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3053_305369


namespace NUMINAMATH_CALUDE_acute_angle_tangent_ratio_l3053_305311

theorem acute_angle_tangent_ratio (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_tangent_ratio_l3053_305311


namespace NUMINAMATH_CALUDE_uniform_price_is_200_l3053_305384

/-- Represents the agreement between a man and his servant --/
structure Agreement where
  full_year_salary : ℕ
  service_duration : ℕ
  actual_duration : ℕ
  partial_payment : ℕ

/-- Calculates the price of the uniform given the agreement details --/
def uniform_price (a : Agreement) : ℕ :=
  let expected_payment := a.full_year_salary * a.actual_duration / a.service_duration
  expected_payment - a.partial_payment

/-- Theorem stating that the price of the uniform is 200 given the problem conditions --/
theorem uniform_price_is_200 : 
  uniform_price { full_year_salary := 800
                , service_duration := 12
                , actual_duration := 9
                , partial_payment := 400 } = 200 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_is_200_l3053_305384


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l3053_305312

theorem cycle_price_calculation (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percent = 20) :
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 900 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l3053_305312


namespace NUMINAMATH_CALUDE_max_vertex_product_sum_l3053_305381

/-- Represents the assignment of numbers to the faces of a cube -/
structure CubeAssignment where
  faces : Fin 6 → ℕ
  valid : ∀ i, faces i ∈ ({3, 4, 5, 6, 7, 8} : Set ℕ)
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- Calculates the sum of products at vertices for a given cube assignment -/
def vertexProductSum (assignment : CubeAssignment) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The maximum sum of vertex products is 1331 -/
theorem max_vertex_product_sum :
  ∃ (assignment : CubeAssignment), 
    vertexProductSum assignment = 1331 ∧
    ∀ (other : CubeAssignment), vertexProductSum other ≤ 1331 :=
  sorry

end NUMINAMATH_CALUDE_max_vertex_product_sum_l3053_305381


namespace NUMINAMATH_CALUDE_largest_integer_below_root_l3053_305304

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 6

theorem largest_integer_below_root :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧
  (∀ x > 0, x < x₀ → f x < 0) ∧
  (∀ x > x₀, f x > 0) ∧
  (∀ n : ℤ, (n : ℝ) ≤ x₀ → n ≤ 4) ∧
  ((4 : ℝ) ≤ x₀) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_below_root_l3053_305304


namespace NUMINAMATH_CALUDE_area_after_transformation_l3053_305317

/-- Given a 2x2 matrix and a planar region, this function returns the area of the transformed region --/
noncomputable def transformedArea (a b c d : ℝ) (originalArea : ℝ) : ℝ :=
  (a * d - b * c) * originalArea

/-- Theorem stating that applying the given matrix to a region of area 9 results in a region of area 126 --/
theorem area_after_transformation :
  let matrix := !![3, 2; -1, 4]
  let originalArea := 9
  transformedArea 3 2 (-1) 4 originalArea = 126 := by
  sorry

end NUMINAMATH_CALUDE_area_after_transformation_l3053_305317


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3053_305382

theorem quadratic_function_property (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -1/3 * x₁^2 + 5 →
  y₂ = -1/3 * x₂^2 + 5 →
  0 < x₁ →
  x₁ < x₂ →
  y₂ < y₁ ∧ y₁ < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3053_305382


namespace NUMINAMATH_CALUDE_power_sum_difference_l3053_305353

theorem power_sum_difference : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l3053_305353


namespace NUMINAMATH_CALUDE_essay_section_length_l3053_305325

theorem essay_section_length 
  (intro_length : ℕ) 
  (conclusion_multiplier : ℕ) 
  (num_body_sections : ℕ) 
  (total_length : ℕ) 
  (h1 : intro_length = 450)
  (h2 : conclusion_multiplier = 3)
  (h3 : num_body_sections = 4)
  (h4 : total_length = 5000)
  : (total_length - (intro_length + intro_length * conclusion_multiplier)) / num_body_sections = 800 := by
  sorry

end NUMINAMATH_CALUDE_essay_section_length_l3053_305325


namespace NUMINAMATH_CALUDE_base6_greater_than_base8_l3053_305333

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem base6_greater_than_base8 : base6ToBase10 403 > base8ToBase10 217 := by
  sorry

end NUMINAMATH_CALUDE_base6_greater_than_base8_l3053_305333
