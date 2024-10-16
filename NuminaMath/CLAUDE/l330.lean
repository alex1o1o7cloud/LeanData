import Mathlib

namespace NUMINAMATH_CALUDE_blue_water_bottles_l330_33033

theorem blue_water_bottles (red black : ℕ) (total removed remaining : ℕ) (blue : ℕ) : 
  red = 2 →
  black = 3 →
  total = red + black + blue →
  removed = 5 →
  remaining = 4 →
  total = removed + remaining →
  blue = 4 := by
sorry

end NUMINAMATH_CALUDE_blue_water_bottles_l330_33033


namespace NUMINAMATH_CALUDE_james_profit_20_weeks_l330_33077

/-- Calculates the profit from James' media empire over a given number of weeks. -/
def calculate_profit (movie_cost : ℕ) (dvd_cost : ℕ) (price_multiplier : ℚ) 
                     (daily_sales : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  let selling_price := dvd_cost * price_multiplier
  let profit_per_dvd := selling_price - dvd_cost
  let daily_profit := profit_per_dvd * daily_sales
  let weekly_profit := daily_profit * days_per_week
  let total_profit := weekly_profit * num_weeks
  (total_profit - movie_cost).floor.toNat

/-- Theorem stating that James' profit over 20 weeks is $448,000. -/
theorem james_profit_20_weeks : 
  calculate_profit 2000 6 (5/2) 500 5 20 = 448000 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_20_weeks_l330_33077


namespace NUMINAMATH_CALUDE_recipe_total_cups_l330_33054

/-- The total number of cups needed for a recipe with cereal, milk, and nuts. -/
def total_cups (cereal_servings milk_servings nuts_servings : ℝ)
               (cereal_cups_per_serving milk_cups_per_serving nuts_cups_per_serving : ℝ) : ℝ :=
  cereal_servings * cereal_cups_per_serving +
  milk_servings * milk_cups_per_serving +
  nuts_servings * nuts_cups_per_serving

/-- Theorem stating that the total cups needed for the given recipe is 57.0 cups. -/
theorem recipe_total_cups :
  total_cups 18.0 12.0 6.0 2.0 1.5 0.5 = 57.0 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l330_33054


namespace NUMINAMATH_CALUDE_square_area_from_rectangles_l330_33053

/-- Given a square divided into 5 identical rectangles, where each rectangle has a perimeter of 120
    and a length that is 5 times its width, the area of the original square is 2500 -/
theorem square_area_from_rectangles (perimeter width length : ℝ) : 
  perimeter = 120 →
  length = 5 * width →
  2 * (length + width) = perimeter →
  (5 * width)^2 = 2500 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangles_l330_33053


namespace NUMINAMATH_CALUDE_solve_for_b_l330_33048

/-- Given two functions p and q, prove that if p(q(3)) = 31, then b has two specific values. -/
theorem solve_for_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 2 * x^2 - 7)
  (hq : ∀ x, q x = 4 * x - b)
  (h_pq3 : p (q 3) = 31) :
  b = 12 + Real.sqrt 19 ∨ b = 12 - Real.sqrt 19 := by
  sorry

#check solve_for_b

end NUMINAMATH_CALUDE_solve_for_b_l330_33048


namespace NUMINAMATH_CALUDE_aunt_may_milk_problem_l330_33085

/-- Aunt May's milk problem -/
theorem aunt_may_milk_problem 
  (morning_milk : ℕ) 
  (evening_milk : ℕ) 
  (sold_milk : ℕ) 
  (leftover_milk : ℕ) 
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 := by
sorry

end NUMINAMATH_CALUDE_aunt_may_milk_problem_l330_33085


namespace NUMINAMATH_CALUDE_parking_fee_range_l330_33057

/-- Represents the parking fee function --/
def parking_fee (x : ℝ) : ℝ := -5 * x + 12000

/-- Theorem: The parking fee range is [6900, 8100] given the problem conditions --/
theorem parking_fee_range :
  ∀ x : ℝ,
  0 ≤ x ∧ x ≤ 1200 ∧
  1200 * 0.65 ≤ x ∧ x ≤ 1200 * 0.85 →
  6900 ≤ parking_fee x ∧ parking_fee x ≤ 8100 :=
by sorry

end NUMINAMATH_CALUDE_parking_fee_range_l330_33057


namespace NUMINAMATH_CALUDE_homework_problem_l330_33060

theorem homework_problem (total : ℕ) (ratio_incomplete : ℕ) (ratio_complete : ℕ) 
  (h_total : total = 15)
  (h_ratio : ratio_incomplete = 3 ∧ ratio_complete = 2) :
  ∃ (completed : ℕ), completed = 6 ∧ 
    ratio_incomplete * completed = ratio_complete * (total - completed) :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l330_33060


namespace NUMINAMATH_CALUDE_circle_center_l330_33021

/-- The equation of a circle in the x-y plane --/
def CircleEquation (x y : ℝ) : Prop :=
  16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 80 = 0

/-- The center of a circle --/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the circle with the given equation is (1, -2) --/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 1 ∧ c.y = -2 ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l330_33021


namespace NUMINAMATH_CALUDE_total_profit_calculation_l330_33026

/-- Given the capital ratios and R's share of the profit, calculate the total profit -/
theorem total_profit_calculation (P Q R : ℕ) (r_profit : ℕ) 
  (h1 : 4 * P = 6 * Q)
  (h2 : 6 * Q = 10 * R)
  (h3 : r_profit = 900) : 
  4650 = (31 * r_profit) / 6 :=
by sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l330_33026


namespace NUMINAMATH_CALUDE_article_count_l330_33011

theorem article_count (x : ℕ) (cost_price selling_price : ℝ) : 
  (cost_price * x = selling_price * 16) →
  (selling_price = 1.5 * cost_price) →
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_article_count_l330_33011


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l330_33097

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 → B = 80 → B = 2 * C → A = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l330_33097


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l330_33034

theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (2*x - 1/(2*x))^10
  ∃ c : ℝ, (∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x) ∧ c = -252 :=
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l330_33034


namespace NUMINAMATH_CALUDE_rotation_center_l330_33003

noncomputable def f (z : ℂ) : ℂ := ((-1 - Complex.I * Real.sqrt 3) * z + (-2 * Real.sqrt 3 + 18 * Complex.I)) / 2

theorem rotation_center :
  ∃ (c : ℂ), f c = c ∧ c = -2 * Real.sqrt 3 - 4 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_rotation_center_l330_33003


namespace NUMINAMATH_CALUDE_length_of_AE_l330_33038

/-- The length of segment AE in a 7x5 grid where AB meets CD at E -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (6, 0) →
  C = (6, 4) →
  D = (2, 0) →
  E = (4, 2) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • B) →
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (1 - s) • C + s • D) →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10 * Real.sqrt 13 / 9 := by
  sorry


end NUMINAMATH_CALUDE_length_of_AE_l330_33038


namespace NUMINAMATH_CALUDE_arithmetic_geometric_equivalence_l330_33063

def is_arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (b : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, q > 1 ∧ ∀ n, b (n + 1) = b n * q

def every_term_in (b a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, b n = a m

theorem arithmetic_geometric_equivalence
  (a b : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_seq a d →
  is_geometric_seq b →
  a 1 = b 1 →
  a 2 = b 2 →
  (d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) →
  every_term_in b a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_equivalence_l330_33063


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l330_33099

/-- The number of dogwood trees in the park after planting -/
def total_trees (current : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  current + planted_today + planted_tomorrow

/-- Theorem stating that the total number of dogwood trees after planting is 100 -/
theorem park_trees_after_planting :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l330_33099


namespace NUMINAMATH_CALUDE_fraction_simplification_l330_33093

theorem fraction_simplification :
  (1/2 + 1/3) / (3/4 - 1/5) = 50/33 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l330_33093


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base_5_l330_33068

theorem base_conversion_1729_to_base_5 :
  ∃ (a b c d e : ℕ),
    1729 = a * 5^4 + b * 5^3 + c * 5^2 + d * 5^1 + e * 5^0 ∧
    a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 0 ∧ e = 4 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_1729_to_base_5_l330_33068


namespace NUMINAMATH_CALUDE_mentor_fraction_l330_33032

theorem mentor_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  n = 2 * s / 3 → (n / 2 + s / 3) / (n + s) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_mentor_fraction_l330_33032


namespace NUMINAMATH_CALUDE_certain_amount_less_than_twice_l330_33024

theorem certain_amount_less_than_twice (n : ℤ) (x : ℤ) : n = 16 ∧ 2 * n - x = 20 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_less_than_twice_l330_33024


namespace NUMINAMATH_CALUDE_second_alloy_amount_calculation_l330_33083

/-- The amount of the second alloy used to create a new alloy with a specific chromium percentage -/
def second_alloy_amount : ℝ := by sorry

/-- The percentage of chromium in the first alloy -/
def first_alloy_chromium_percent : ℝ := 0.15

/-- The percentage of chromium in the second alloy -/
def second_alloy_chromium_percent : ℝ := 0.08

/-- The amount of the first alloy used -/
def first_alloy_amount : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def new_alloy_chromium_percent : ℝ := 0.101

theorem second_alloy_amount_calculation :
  first_alloy_chromium_percent * first_alloy_amount +
  second_alloy_chromium_percent * second_alloy_amount =
  new_alloy_chromium_percent * (first_alloy_amount + second_alloy_amount) ∧
  second_alloy_amount = 35 := by sorry

end NUMINAMATH_CALUDE_second_alloy_amount_calculation_l330_33083


namespace NUMINAMATH_CALUDE_josh_cheese_purchase_cost_l330_33015

/-- Calculates the total cost of string cheese purchase including tax -/
def total_cost_with_tax (packs : ℕ) (pieces_per_pack : ℕ) (cost_per_piece : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := packs * pieces_per_pack * cost_per_piece
  let tax := total_cost * tax_rate
  total_cost + tax

/-- The total cost of Josh's string cheese purchase including tax is $6.72 -/
theorem josh_cheese_purchase_cost :
  total_cost_with_tax 3 20 (10 / 100) (12 / 100) = 672 / 100 := by
  sorry

#eval total_cost_with_tax 3 20 (10 / 100) (12 / 100)

end NUMINAMATH_CALUDE_josh_cheese_purchase_cost_l330_33015


namespace NUMINAMATH_CALUDE_unique_equidistant_point_l330_33047

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  sideLength : ℝ

/-- Checks if a point is inside or on the diagonal face BDD₁B₁ of the cube -/
def isOnDiagonalFace (c : Cube) (p : Point3D) : Prop := sorry

/-- Calculates the distance from a point to a plane -/
def distToPlane (p : Point3D) (plane : Point3D → Prop) : ℝ := sorry

/-- The plane ABC of the cube -/
def planeABC (c : Cube) : Point3D → Prop := sorry

/-- The plane ABA₁ of the cube -/
def planeABA1 (c : Cube) : Point3D → Prop := sorry

/-- The plane ADA₁ of the cube -/
def planeADA1 (c : Cube) : Point3D → Prop := sorry

theorem unique_equidistant_point (c : Cube) : 
  ∃! p : Point3D, 
    isOnDiagonalFace c p ∧ 
    distToPlane p (planeABC c) = distToPlane p (planeABA1 c) ∧
    distToPlane p (planeABC c) = distToPlane p (planeADA1 c) :=
sorry

end NUMINAMATH_CALUDE_unique_equidistant_point_l330_33047


namespace NUMINAMATH_CALUDE_product_remainder_theorem_l330_33087

def numbers : List Nat := [445876, 985420, 215546, 656452, 387295]

def remainder_sum_squares (nums : List Nat) : Nat :=
  (nums.map (λ n => (n^2) % 8)).sum

theorem product_remainder_theorem :
  (remainder_sum_squares numbers) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_theorem_l330_33087


namespace NUMINAMATH_CALUDE_population_change_l330_33086

/-- The population change problem --/
theorem population_change (P : ℝ) : 
  P > 0 → 
  P * 1.15 * 0.90 * 1.20 * 0.75 = 7575 → 
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) ≤ P ∧ P < (n : ℝ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_population_change_l330_33086


namespace NUMINAMATH_CALUDE_squares_in_figure_50_l330_33052

/-- The number of squares in the nth figure -/
def f (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

/-- The sequence of squares follows the given pattern -/
axiom pattern_holds : f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37

/-- The number of squares in figure 50 is 7651 -/
theorem squares_in_figure_50 : f 50 = 7651 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_figure_50_l330_33052


namespace NUMINAMATH_CALUDE_vegetables_in_box_l330_33095

/-- Given a box with cabbages and radishes, we define the total number of vegetables -/
def total_vegetables (num_cabbages num_radishes : ℕ) : ℕ :=
  num_cabbages + num_radishes

/-- Theorem: In a box with 3 cabbages and 2 radishes, there are 5 vegetables in total -/
theorem vegetables_in_box : total_vegetables 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_in_box_l330_33095


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l330_33012

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 16) and (-2, -8) is 7. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (8, 16)
  let p2 : ℝ × ℝ := (-2, -8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 7 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l330_33012


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_1013_l330_33014

theorem inverse_of_3_mod_1013 : ∃ x : ℕ, 0 ≤ x ∧ x < 1013 ∧ (3 * x) % 1013 = 1 :=
by
  use 338
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_1013_l330_33014


namespace NUMINAMATH_CALUDE_seed_mixture_ryegrass_percentage_l330_33030

/-- Given two seed mixtures X and Y, prove that Y contains 25% ryegrass -/
theorem seed_mixture_ryegrass_percentage :
  -- Definitions based on the problem conditions
  let x_ryegrass : ℝ := 0.40  -- 40% ryegrass in X
  let x_bluegrass : ℝ := 0.60  -- 60% bluegrass in X
  let y_fescue : ℝ := 0.75  -- 75% fescue in Y
  let final_ryegrass : ℝ := 0.32  -- 32% ryegrass in final mixture
  let x_proportion : ℝ := 0.4667  -- 46.67% of final mixture is X
  
  -- The percentage of ryegrass in Y
  ∃ y_ryegrass : ℝ,
    -- Conditions
    x_ryegrass + x_bluegrass = 1 ∧  -- X components sum to 100%
    y_ryegrass + y_fescue = 1 ∧  -- Y components sum to 100%
    x_proportion * x_ryegrass + (1 - x_proportion) * y_ryegrass = final_ryegrass →
    -- Conclusion
    y_ryegrass = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_ryegrass_percentage_l330_33030


namespace NUMINAMATH_CALUDE_tangent_slope_circle_tangent_slope_specific_circle_l330_33029

theorem tangent_slope_circle (center : ℝ × ℝ) (tangent_point : ℝ × ℝ) : ℝ :=
  let center_x : ℝ := center.1
  let center_y : ℝ := center.2
  let tangent_x : ℝ := tangent_point.1
  let tangent_y : ℝ := tangent_point.2
  let radius_slope : ℝ := (tangent_y - center_y) / (tangent_x - center_x)
  let tangent_slope : ℝ := -1 / radius_slope
  tangent_slope

theorem tangent_slope_specific_circle : 
  tangent_slope_circle (2, 3) (7, 8) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_tangent_slope_specific_circle_l330_33029


namespace NUMINAMATH_CALUDE_teacher_zhang_age_after_five_years_l330_33079

/-- Given Xiao Li's age and the relation to Teacher Zhang's age, calculate Teacher Zhang's age after 5 years -/
theorem teacher_zhang_age_after_five_years (a : ℕ) : 
  (3 * a - 2 : ℕ) + 5 = 3 * a + 3 :=
by sorry

end NUMINAMATH_CALUDE_teacher_zhang_age_after_five_years_l330_33079


namespace NUMINAMATH_CALUDE_intersection_k_value_l330_33019

theorem intersection_k_value (k : ℝ) : 
  (∃ y : ℝ, 3 * (-6) - 2 * y = k ∧ -6 - 0.5 * y = 10) → k = 46 := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l330_33019


namespace NUMINAMATH_CALUDE_intersection_points_coincide_l330_33045

/-- Two circles in a plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionPointsDistanceSquared (circles : TwoCircles) : ℝ := sorry

/-- Theorem: The square of the distance between intersection points is zero for the given circles -/
theorem intersection_points_coincide (circles : TwoCircles) 
  (h1 : circles.center1 = (3, -2))
  (h2 : circles.radius1 = 5)
  (h3 : circles.center2 = (3, 6))
  (h4 : circles.radius2 = 3) :
  intersectionPointsDistanceSquared circles = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_points_coincide_l330_33045


namespace NUMINAMATH_CALUDE_both_correct_probability_l330_33041

-- Define the probabilities
def prob_first : ℝ := 0.75
def prob_second : ℝ := 0.55
def prob_neither : ℝ := 0.20

-- Theorem statement
theorem both_correct_probability : 
  prob_first + prob_second - (1 - prob_neither) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_both_correct_probability_l330_33041


namespace NUMINAMATH_CALUDE_function_form_l330_33056

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating the form of the function satisfying the equation -/
theorem function_form (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_function_form_l330_33056


namespace NUMINAMATH_CALUDE_angle_sum_from_tan_roots_l330_33039

theorem angle_sum_from_tan_roots (α β : Real) :
  (∃ x y : Real, x^2 + 6*x + 7 = 0 ∧ y^2 + 6*y + 7 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  α + β = -3*π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_from_tan_roots_l330_33039


namespace NUMINAMATH_CALUDE_problem_statement_l330_33059

theorem problem_statement :
  ((-2023)^0 : ℝ) - 4 * Real.sin (π/4) + |(-Real.sqrt 8)| = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l330_33059


namespace NUMINAMATH_CALUDE_bench_cost_proof_l330_33058

/-- The cost of the bench in dollars -/
def bench_cost : ℝ := 150

/-- The cost of the garden table in dollars -/
def table_cost : ℝ := 2 * bench_cost

/-- The combined cost of the bench and garden table in dollars -/
def combined_cost : ℝ := 450

theorem bench_cost_proof : bench_cost = 150 := by sorry

end NUMINAMATH_CALUDE_bench_cost_proof_l330_33058


namespace NUMINAMATH_CALUDE_age_problem_l330_33072

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 32 → 
  b = 12 := by sorry

end NUMINAMATH_CALUDE_age_problem_l330_33072


namespace NUMINAMATH_CALUDE_floor_add_two_floor_sum_inequality_floor_square_inequality_exists_l330_33010

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem 1
theorem floor_add_two (x : ℝ) : floor (x + 2) = floor x + 2 := by sorry

-- Theorem 2
theorem floor_sum_inequality (x y : ℝ) : floor (x + y) ≤ floor x + floor y := by sorry

-- Theorem 3
theorem floor_square_inequality_exists :
  ∃ x : ℝ, floor (x^2) ≠ (floor x)^2 := by sorry

end NUMINAMATH_CALUDE_floor_add_two_floor_sum_inequality_floor_square_inequality_exists_l330_33010


namespace NUMINAMATH_CALUDE_seating_theorem_l330_33074

/-- The number of seats in the row -/
def n : ℕ := 8

/-- The number of people to be seated -/
def k : ℕ := 2

/-- The number of different seating arrangements for k people in n seats,
    with empty seats required on both sides of each person -/
def seating_arrangements (n k : ℕ) : ℕ := sorry

/-- Theorem stating that the number of seating arrangements
    for 2 people in 8 seats is 20 -/
theorem seating_theorem : seating_arrangements n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l330_33074


namespace NUMINAMATH_CALUDE_max_cookies_without_ingredients_l330_33028

/-- Given a set of cookies with specific ingredient distributions, 
    prove the maximum number of cookies without any of the ingredients. -/
theorem max_cookies_without_ingredients (total_cookies : ℕ) 
    (h_total : total_cookies = 48)
    (h_choc_chips : (total_cookies / 2 : ℕ) = 24)
    (h_peanut_butter : (total_cookies * 3 / 4 : ℕ) = 36)
    (h_white_choc : (total_cookies / 3 : ℕ) = 16)
    (h_coconut : (total_cookies / 8 : ℕ) = 6) :
    ∃ (max_without : ℕ), max_without ≤ 12 ∧ 
    max_without = total_cookies - (total_cookies * 3 / 4 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_without_ingredients_l330_33028


namespace NUMINAMATH_CALUDE_specific_ohara_triple_l330_33023

/-- O'Hara triple condition -/
def is_ohara_triple (a b x : ℝ) : Prop := Real.sqrt a + Real.sqrt b = x

/-- Proof of the specific O'Hara triple -/
theorem specific_ohara_triple :
  let a : ℝ := 49
  let b : ℝ := 16
  ∃ x : ℝ, is_ohara_triple a b x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_specific_ohara_triple_l330_33023


namespace NUMINAMATH_CALUDE_tangent_line_slope_l330_33025

/-- The value of a for which the tangent line to y = ax - ln(x+1) at (0,0) is y = 2x -/
theorem tangent_line_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - Real.log (x + 1)) →
  (∃ m : ℝ, ∀ x y : ℝ, y = m * x ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
      |((a * h - Real.log (h + 1)) / h) - m| < ε)) →
  m = 2 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l330_33025


namespace NUMINAMATH_CALUDE_different_color_probability_l330_33006

def blue_chips : ℕ := 8
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

def prob_different_colors : ℚ :=
  (blue_chips * (red_chips + yellow_chips + green_chips) +
   red_chips * (blue_chips + yellow_chips + green_chips) +
   yellow_chips * (blue_chips + red_chips + green_chips) +
   green_chips * (blue_chips + red_chips + yellow_chips)) /
  (total_chips * total_chips)

theorem different_color_probability :
  prob_different_colors = 143 / 200 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l330_33006


namespace NUMINAMATH_CALUDE_nathans_earnings_186_l330_33090

/-- Calculates the total earnings from Nathan's harvest --/
def nathans_earnings (strawberry_plants : ℕ) (tomato_plants : ℕ) 
  (strawberries_per_plant : ℕ) (tomatoes_per_plant : ℕ) 
  (fruits_per_basket : ℕ) (strawberry_basket_price : ℕ) (tomato_basket_price : ℕ) : ℕ :=
  let total_strawberries := strawberry_plants * strawberries_per_plant
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let strawberry_baskets := total_strawberries / fruits_per_basket
  let tomato_baskets := total_tomatoes / fruits_per_basket
  let strawberry_earnings := strawberry_baskets * strawberry_basket_price
  let tomato_earnings := tomato_baskets * tomato_basket_price
  strawberry_earnings + tomato_earnings

/-- Theorem stating that Nathan's earnings from his harvest equal $186 --/
theorem nathans_earnings_186 :
  nathans_earnings 5 7 14 16 7 9 6 = 186 := by
  sorry

end NUMINAMATH_CALUDE_nathans_earnings_186_l330_33090


namespace NUMINAMATH_CALUDE_octahedron_faces_l330_33017

/-- An octahedron is a polyhedron with a specific number of faces -/
structure Octahedron where
  faces : ℕ

/-- The number of faces of an octahedron is 8 -/
theorem octahedron_faces (o : Octahedron) : o.faces = 8 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_faces_l330_33017


namespace NUMINAMATH_CALUDE_walking_problem_l330_33070

/-- The walking problem theorem -/
theorem walking_problem (total_distance : ℝ) (yolanda_rate : ℝ) (bob_distance : ℝ) : 
  total_distance = 24 →
  yolanda_rate = 3 →
  bob_distance = 12 →
  ∃ (bob_rate : ℝ), bob_rate = 12 ∧ bob_distance = bob_rate * 1 := by
  sorry

end NUMINAMATH_CALUDE_walking_problem_l330_33070


namespace NUMINAMATH_CALUDE_point_sqrt_6_away_from_origin_l330_33027

-- Define a point on the number line
def Point := ℝ

-- Define the distance function
def distance (p : Point) : ℝ := |p|

-- State the theorem
theorem point_sqrt_6_away_from_origin (M : Point) 
  (h : distance M = Real.sqrt 6) : M = Real.sqrt 6 ∨ M = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_point_sqrt_6_away_from_origin_l330_33027


namespace NUMINAMATH_CALUDE_martian_puzzle_l330_33061

-- Define the Martian type
inductive Martian
| Red
| Blue

-- Define the state of the Martians
structure MartianState where
  total : Nat
  initialRed : Nat
  currentRed : Nat

-- Define the properties of the Martians' answers
def validAnswerSequence (state : MartianState) : Prop :=
  state.total = 2018 ∧
  ∀ i : Nat, i < state.total → 
    (i + 1 = state.initialRed + i - state.initialRed + 1)

-- Define the theorem
theorem martian_puzzle :
  ∀ state : MartianState,
    validAnswerSequence state →
    (state.initialRed = 0 ∨ state.initialRed = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_martian_puzzle_l330_33061


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l330_33069

/-- An isosceles triangle with sides a, b, and c, where two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)
  positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem stating that an isosceles triangle with sides 3 and 4 has perimeter 10 or 11 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 3 ∧ t.b = 4) ∨ (t.a = 4 ∧ t.b = 3) ∨ 
   (t.b = 3 ∧ t.c = 4) ∨ (t.b = 4 ∧ t.c = 3) ∨ 
   (t.a = 3 ∧ t.c = 4) ∨ (t.a = 4 ∧ t.c = 3)) →
  (perimeter t = 10 ∨ perimeter t = 11) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l330_33069


namespace NUMINAMATH_CALUDE_work_left_theorem_l330_33071

def work_left (p_days q_days collab_days : ℚ) : ℚ :=
  1 - collab_days * (1 / p_days + 1 / q_days)

theorem work_left_theorem (p_days q_days collab_days : ℚ) 
  (hp : p_days = 15)
  (hq : q_days = 20)
  (hc : collab_days = 4) :
  work_left p_days q_days collab_days = 8 / 15 := by
  sorry

#eval work_left 15 20 4

end NUMINAMATH_CALUDE_work_left_theorem_l330_33071


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_expressions_l330_33042

theorem consecutive_odd_integers_expressions (p q : ℤ) 
  (h1 : ∃ k : ℤ, p = 2*k + 1 ∧ q = 2*k + 3) : 
  Odd (2*p + 5*q) ∧ Odd (5*p - 2*q) ∧ Odd (2*p*q + 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_expressions_l330_33042


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l330_33046

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l330_33046


namespace NUMINAMATH_CALUDE_triangle_area_l330_33051

theorem triangle_area (a b c : ℝ) (α : ℝ) (h1 : a = 14)
  (h2 : α = Real.pi / 3) (h3 : b / c = 8 / 5) :
  (1 / 2) * b * c * Real.sin α = 40 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l330_33051


namespace NUMINAMATH_CALUDE_smallest_b_value_l330_33081

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1 / a + 1 / b ≤ 2) →
  ∀ ε > 0, ∃ b₀ : ℝ, 2 < b₀ ∧ b₀ < 2 + ε ∧
    ∃ a₀ : ℝ, 2 < a₀ ∧ a₀ < b₀ ∧
    (2 + a₀ ≤ b₀) ∧
    (1 / a₀ + 1 / b₀ ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l330_33081


namespace NUMINAMATH_CALUDE_cos_two_beta_equals_one_l330_33091

theorem cos_two_beta_equals_one (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 0) : 
  Real.cos (2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_beta_equals_one_l330_33091


namespace NUMINAMATH_CALUDE_angle_measure_from_area_ratio_l330_33078

/-- Given three concentric circles and two lines passing through their center,
    prove that the acute angle between the lines is 12π/77 radians when the
    shaded area is 3/4 of the unshaded area. -/
theorem angle_measure_from_area_ratio :
  ∀ (r₁ r₂ r₃ : ℝ) (shaded_area unshaded_area : ℝ) (θ : ℝ),
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_area = (3/4) * unshaded_area →
  shaded_area + unshaded_area = π * (r₁^2 + r₂^2 + r₃^2) →
  shaded_area = θ * (r₁^2 + r₃^2) + (π - θ) * r₂^2 →
  θ = 12 * π / 77 :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_from_area_ratio_l330_33078


namespace NUMINAMATH_CALUDE_factorization_equality_l330_33073

theorem factorization_equality (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 = (x^4 + x^2 + 1) * (x^2 + 2) * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l330_33073


namespace NUMINAMATH_CALUDE_waiter_section_proof_l330_33007

/-- Calculates the number of customers who left a waiter's section. -/
def customers_who_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  initial_customers - (remaining_tables * people_per_table)

/-- Proves that 17 customers left the waiter's section given the initial conditions. -/
theorem waiter_section_proof :
  customers_who_left 62 5 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_waiter_section_proof_l330_33007


namespace NUMINAMATH_CALUDE_trig_identity_l330_33018

theorem trig_identity (α β : Real) 
  (h : (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1) :
  ∃ x, (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = x ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l330_33018


namespace NUMINAMATH_CALUDE_problem_solution_l330_33064

theorem problem_solution (x y : ℝ) (h : |x + 5| + (y - 4)^2 = 0) : (x + y)^99 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l330_33064


namespace NUMINAMATH_CALUDE_winningScoresCount_is_nineteen_l330_33049

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- Number of runners in each team -/
  runnersPerTeam : Nat
  /-- Total number of runners -/
  totalRunners : Nat
  /-- The sum of all positions -/
  totalPositionSum : Nat
  /-- Condition that there are 2 teams -/
  twoTeams : totalRunners = 2 * runnersPerTeam
  /-- Condition that the total sum of positions is correct -/
  validTotalSum : totalPositionSum = totalRunners * (totalRunners + 1) / 2

/-- The number of different winning scores possible in a cross country meet -/
def winningScoresCount (meet : CrossCountryMeet) : Nat :=
  meet.totalPositionSum / 2 - (meet.runnersPerTeam * (meet.runnersPerTeam + 1) / 2) + 1

/-- Theorem stating that the number of different winning scores is 19 -/
theorem winningScoresCount_is_nineteen :
  ∀ (meet : CrossCountryMeet),
    meet.runnersPerTeam = 6 →
    winningScoresCount meet = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_winningScoresCount_is_nineteen_l330_33049


namespace NUMINAMATH_CALUDE_cars_cannot_meet_l330_33076

-- Define the network structure
structure TriangleNetwork where
  -- Assume the network is infinite and regular
  -- Each vertex has exactly 6 edges connected to it
  vertex_degree : ℕ
  vertex_degree_eq : vertex_degree = 6

-- Define a car's position and movement
structure Car where
  position : ℕ × ℕ  -- Represent position as discrete coordinates
  direction : ℕ     -- 0, 1, or 2 representing the three possible directions

-- Define the movement options
inductive Move
  | straight
  | left
  | right

-- Function to update car position based on move
def update_position (c : Car) (m : Move) : Car :=
  sorry  -- Implementation details omitted for brevity

-- Theorem statement
theorem cars_cannot_meet 
  (network : TriangleNetwork) 
  (car1 car2 : Car) 
  (start_same_edge : car1.position.1 = car2.position.1 ∧ car1.direction = car2.direction)
  (t : ℕ) :
  ∀ (moves1 moves2 : List Move),
  moves1.length = t ∧ moves2.length = t →
  (moves1.foldl update_position car1).position ≠ (moves2.foldl update_position car2).position :=
sorry

end NUMINAMATH_CALUDE_cars_cannot_meet_l330_33076


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l330_33004

def geometric_series (n : ℕ) : ℚ := (7 / 3) * (7 / 3) ^ n

theorem common_ratio_of_geometric_series :
  ∀ n : ℕ, geometric_series (n + 1) / geometric_series n = 7 / 3 :=
by
  sorry

#check common_ratio_of_geometric_series

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l330_33004


namespace NUMINAMATH_CALUDE_max_value_2x_3y_l330_33020

theorem max_value_2x_3y (x y : ℝ) (h : x^2 + y^2 = 16*x + 8*y + 20) :
  ∃ (M : ℝ), M = 33 ∧ 2*x + 3*y ≤ M ∧ ∃ (x₀ y₀ : ℝ), 2*x₀ + 3*y₀ = M ∧ x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 20 :=
sorry

end NUMINAMATH_CALUDE_max_value_2x_3y_l330_33020


namespace NUMINAMATH_CALUDE_austonHeightCm_l330_33000

/-- Converts inches to centimeters -/
def inchesToCm (inches : ℝ) : ℝ := inches * 2.54

/-- Auston's height in inches -/
def austonHeightInches : ℝ := 60

/-- Theorem stating Auston's height in centimeters -/
theorem austonHeightCm : inchesToCm austonHeightInches = 152.4 := by
  sorry

end NUMINAMATH_CALUDE_austonHeightCm_l330_33000


namespace NUMINAMATH_CALUDE_cornelia_asian_countries_l330_33043

theorem cornelia_asian_countries (total : ℕ) (europe : ℕ) (south_america : ℕ) 
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : south_america = 10)
  (h4 : (total - europe - south_america) % 2 = 0) :
  (total - europe - south_america) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cornelia_asian_countries_l330_33043


namespace NUMINAMATH_CALUDE_square_of_negative_product_l330_33066

theorem square_of_negative_product (x y : ℝ) : (-x * y^2)^2 = x^2 * y^4 := by sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l330_33066


namespace NUMINAMATH_CALUDE_bisection_method_root_location_l330_33084

def f (x : ℝ) := x^3 - 2*x - 1

theorem bisection_method_root_location :
  (∃ r ∈ Set.Ioo 1 2, f r = 0) →
  (f 1 < 0) →
  (f 2 > 0) →
  (f 1.5 < 0) →
  ∃ r ∈ Set.Ioo 1.5 2, f r = 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_root_location_l330_33084


namespace NUMINAMATH_CALUDE_average_of_squares_first_11_even_l330_33089

/-- The first 11 consecutive even numbers -/
def first_11_even_numbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

/-- The average of squares of the first 11 consecutive even numbers -/
theorem average_of_squares_first_11_even : 
  (first_11_even_numbers.map (λ x => x^2)).sum / first_11_even_numbers.length = 184 := by
  sorry

#eval (first_11_even_numbers.map (λ x => x^2)).sum / first_11_even_numbers.length

end NUMINAMATH_CALUDE_average_of_squares_first_11_even_l330_33089


namespace NUMINAMATH_CALUDE_factorization_proof_l330_33008

theorem factorization_proof (y : ℝ) : 81 * y^19 + 162 * y^38 = 81 * y^19 * (1 + 2 * y^19) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l330_33008


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_2057_l330_33002

theorem tens_digit_of_23_pow_2057 : ∃ n : ℕ, 23^2057 ≡ 60 + n [ZMOD 100] ∧ n < 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_2057_l330_33002


namespace NUMINAMATH_CALUDE_water_price_this_year_l330_33036

-- Define the price of water last year
def price_last_year : ℝ := 1.6

-- Define the price increase rate
def price_increase_rate : ℝ := 0.2

-- Define Xiao Li's water bill in December last year
def december_bill : ℝ := 17

-- Define Xiao Li's water bill in January this year
def january_bill : ℝ := 30

-- Define the difference in water consumption between January and December
def consumption_difference : ℝ := 5

-- Theorem: The price of residential water this year is 1.92 yuan per cubic meter
theorem water_price_this_year :
  let price_this_year := price_last_year * (1 + price_increase_rate)
  price_this_year = 1.92 ∧
  january_bill / price_this_year - december_bill / price_last_year = consumption_difference :=
by sorry

end NUMINAMATH_CALUDE_water_price_this_year_l330_33036


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l330_33013

/-- Given m > 0, p: (x+2)(x-6) ≤ 0, and q: 2-m ≤ x ≤ 2+m -/
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

/-- If p is a necessary condition for q, then 0 < m ≤ 4 -/
theorem range_of_m (m : ℝ) (h : m > 0) :
  (∀ x, q m x → p x) → 0 < m ∧ m ≤ 4 := by sorry

/-- Given m = 2, if ¬p ∨ ¬q is false, then 0 ≤ x ≤ 4 -/
theorem range_of_x (x : ℝ) :
  ¬(¬(p x) ∨ ¬(q 2 x)) → 0 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l330_33013


namespace NUMINAMATH_CALUDE_residue_7_2023_mod_19_l330_33082

theorem residue_7_2023_mod_19 : 7^2023 % 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_7_2023_mod_19_l330_33082


namespace NUMINAMATH_CALUDE_f_2005_equals_2_l330_33005

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2005_equals_2 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 6) = f x + f 3)
  (h_f_1 : f 1 = 2) :
  f 2005 = 2 := by
sorry

end NUMINAMATH_CALUDE_f_2005_equals_2_l330_33005


namespace NUMINAMATH_CALUDE_prob_not_all_same_value_l330_33050

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that five fair 8-sided dice won't all show the same number -/
def prob_not_all_same : ℚ :=
  1 - (sides : ℚ) / sides ^ num_dice

theorem prob_not_all_same_value :
  prob_not_all_same = 4095 / 4096 :=
sorry

end NUMINAMATH_CALUDE_prob_not_all_same_value_l330_33050


namespace NUMINAMATH_CALUDE_product_quantity_relationship_l330_33040

/-- The initial budget in yuan -/
def initial_budget : ℝ := 1500

/-- The price increase of product A in yuan -/
def price_increase_A : ℝ := 1.5

/-- The price increase of product B in yuan -/
def price_increase_B : ℝ := 1

/-- The reduction in quantity of product A in the first scenario -/
def quantity_reduction_A1 : ℝ := 10

/-- The budget excess in the first scenario -/
def budget_excess : ℝ := 29

/-- The reduction in quantity of product A in the second scenario -/
def quantity_reduction_A2 : ℝ := 5

/-- The total cost in the second scenario -/
def total_cost_scenario2 : ℝ := 1563.5

theorem product_quantity_relationship (x y a b : ℝ) :
  (a * x + b * y = initial_budget) →
  ((a + price_increase_A) * (x - quantity_reduction_A1) + (b + price_increase_B) * y = initial_budget + budget_excess) →
  ((a + 1) * (x - quantity_reduction_A2) + (b + 1) * y = total_cost_scenario2) →
  (2 * x + y > 205) →
  (2 * x + y < 210) →
  (x + 2 * y = 186) := by
sorry

end NUMINAMATH_CALUDE_product_quantity_relationship_l330_33040


namespace NUMINAMATH_CALUDE_factors_of_81_l330_33062

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l330_33062


namespace NUMINAMATH_CALUDE_percent_relationship_l330_33098

theorem percent_relationship (x y z : ℝ) 
  (h1 : x = 1.2 * y) 
  (h2 : y = 0.4 * z) : 
  x = 0.48 * z := by
  sorry

end NUMINAMATH_CALUDE_percent_relationship_l330_33098


namespace NUMINAMATH_CALUDE_max_profit_at_16_l330_33088

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (k b x : ℝ) : ℝ := k * x + b

/-- Represents the daily profit as a function of selling price -/
def daily_profit (k b x : ℝ) : ℝ := (x - 12) * (sales_quantity k b x)

theorem max_profit_at_16 (k b : ℝ) :
  sales_quantity k b 15 = 50 →
  sales_quantity k b 17 = 30 →
  (∀ x, 12 ≤ x → x ≤ 18 → daily_profit k b x ≤ daily_profit k b 16) ∧
  daily_profit k b 16 = 160 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_16_l330_33088


namespace NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l330_33075

theorem inverse_trig_sum_equals_pi : 
  Real.arctan (Real.sqrt 3) - Real.arcsin (-1/2) + Real.arccos 0 = π := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l330_33075


namespace NUMINAMATH_CALUDE_centroid_tetrahedron_volume_ratio_l330_33055

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- Checks if a point is inside a tetrahedron -/
def isInterior (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Main theorem: volume ratio of centroids' tetrahedron to original tetrahedron -/
theorem centroid_tetrahedron_volume_ratio 
  (ABCD : Tetrahedron) (P : Point3D) 
  (h : isInterior P ABCD) : 
  let G1 := centroid ⟨P, ABCD.A, ABCD.B, ABCD.C⟩
  let G2 := centroid ⟨P, ABCD.B, ABCD.C, ABCD.D⟩
  let G3 := centroid ⟨P, ABCD.C, ABCD.D, ABCD.A⟩
  let G4 := centroid ⟨P, ABCD.D, ABCD.A, ABCD.B⟩
  volume ⟨G1, G2, G3, G4⟩ / volume ABCD = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_centroid_tetrahedron_volume_ratio_l330_33055


namespace NUMINAMATH_CALUDE_problem_statement_l330_33037

-- Define the function f
def f (a b c x : ℝ) : ℝ := a*x + b*x - c*x

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define an obtuse triangle
def is_obtuse (a b c : ℝ) : Prop :=
  a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2

theorem problem_statement 
  (a b c : ℝ) 
  (h1 : c > a ∧ a > 0) 
  (h2 : c > b ∧ b > 0) 
  (h3 : triangle_inequality a b c) :
  (∃ x : ℝ, ¬ triangle_inequality (a*x) (b*x) (c*x)) ∧ 
  (is_obtuse a b c → ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f a b c x = 0) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l330_33037


namespace NUMINAMATH_CALUDE_company_blocks_l330_33009

/-- Represents the number of workers in each block -/
def workers_per_block : ℕ := 200

/-- Represents the total budget for gifts in dollars -/
def total_budget : ℕ := 6000

/-- Represents the cost of each gift in dollars -/
def gift_cost : ℕ := 2

/-- Calculates the number of blocks in the company -/
def number_of_blocks : ℕ := total_budget / (workers_per_block * gift_cost)

/-- Theorem stating that the number of blocks in the company is 15 -/
theorem company_blocks : number_of_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_company_blocks_l330_33009


namespace NUMINAMATH_CALUDE_problem_statement_l330_33031

theorem problem_statement (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 10) :
  (Real.log a + Real.log b > 0) ∧
  (Real.log a - Real.log b > 0) ∧
  (Real.log a * Real.log b < 1/4) ∧
  (¬ ∀ x y : ℝ, x > y ∧ y > 0 ∧ x * y = 10 → Real.log x / Real.log y > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l330_33031


namespace NUMINAMATH_CALUDE_black_area_proof_l330_33035

theorem black_area_proof (white_area black_area : ℝ) : 
  white_area + black_area = 9^2 + 5^2 →
  white_area + 2 * black_area = 11^2 + 7^2 →
  black_area = 64 := by
  sorry

end NUMINAMATH_CALUDE_black_area_proof_l330_33035


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_bounds_l330_33092

/-- Given four points A, B, C, D in a plane, with distances AB = 2, BC = 7, CD = 5, and DA = 12,
    the minimum possible length of AC is 7 and the maximum possible length of AC is 9. -/
theorem quadrilateral_diagonal_bounds (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 2)
  (h2 : dist B C = 7)
  (h3 : dist C D = 5)
  (h4 : dist D A = 12) :
  (∃ (m M : ℝ), m = 7 ∧ M = 9 ∧ 
    (∀ (AC : ℝ), AC = dist A C → m ≤ AC ∧ AC ≤ M)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_bounds_l330_33092


namespace NUMINAMATH_CALUDE_expression_value_when_b_is_3_l330_33016

theorem expression_value_when_b_is_3 :
  let b : ℝ := 3
  let expr := (3 * b⁻¹ + b⁻¹ / 3) / b^2
  expr = 10 / 81 := by sorry

end NUMINAMATH_CALUDE_expression_value_when_b_is_3_l330_33016


namespace NUMINAMATH_CALUDE_remainder_sum_l330_33096

theorem remainder_sum (n : ℤ) (h : n % 24 = 10) : (n % 4 + n % 6 = 6) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l330_33096


namespace NUMINAMATH_CALUDE_task_completion_probability_l330_33022

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 2/3) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l330_33022


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l330_33044

theorem partial_fraction_decomposition :
  ∃! (A B : ℝ), ∀ (x : ℝ), x ≠ 5 → x ≠ 6 →
    (5 * x - 8) / (x^2 - 11 * x + 30) = A / (x - 5) + B / (x - 6) ∧ A = -17 ∧ B = 22 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l330_33044


namespace NUMINAMATH_CALUDE_lattice_fifth_number_ninth_row_l330_33067

/-- Given a lattice with 7 numbers in each row, continuing for 9 rows,
    the fifth number in the 9th row is 60. -/
theorem lattice_fifth_number_ninth_row :
  ∀ (lattice : ℕ → ℕ → ℕ),
    (∀ row col, col ≤ 7 → lattice row col = row * col) →
    lattice 9 5 = 60 := by
sorry

end NUMINAMATH_CALUDE_lattice_fifth_number_ninth_row_l330_33067


namespace NUMINAMATH_CALUDE_subway_speed_increase_l330_33094

-- Define the speed function
def speed (s : ℝ) : ℝ := s^2 + 2*s

-- State the theorem
theorem subway_speed_increase (s : ℝ) : 
  0 ≤ s ∧ s ≤ 7 → 
  speed s = speed 5 + 28 → 
  s = 7 := by
  sorry

end NUMINAMATH_CALUDE_subway_speed_increase_l330_33094


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l330_33001

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

end NUMINAMATH_CALUDE_stewart_farm_ratio_l330_33001


namespace NUMINAMATH_CALUDE_exists_special_function_l330_33065

def I : Set ℝ := Set.Icc (-1) 1

def is_piecewise_continuous (f : ℝ → ℝ) : Prop :=
  ∃ (s : Set ℝ), Set.Finite s ∧
  ∀ x ∈ I, x ∉ s → ∃ ε > 0, ∀ y ∈ I, |y - x| < ε → f y = f x

theorem exists_special_function :
  ∃ f : ℝ → ℝ,
    (∀ x ∈ I, f (f x) = -x) ∧
    (∀ x ∉ I, f x = 0) ∧
    is_piecewise_continuous f :=
sorry

end NUMINAMATH_CALUDE_exists_special_function_l330_33065


namespace NUMINAMATH_CALUDE_expression_bounds_l330_33080

theorem expression_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  let k : ℝ := 2
  let expr := Real.sqrt (k * a^2 + (2 - b)^2) + Real.sqrt (k * b^2 + (2 - c)^2) + Real.sqrt (k * c^2 + (2 - a)^2)
  6 * Real.sqrt 2 ≤ expr ∧ expr ≤ 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l330_33080
