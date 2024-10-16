import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2342_234208

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- The point P -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- A chord AB passing through F -/
def chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ∃ (t : ℝ), (1 - t) • A + t • B = F

/-- The angle equality condition -/
def angle_equality (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  let P := (p, 0)
  let AP := (A.1 - p, A.2)
  let BP := (B.1 - p, B.2)
  let PF := (2 - p, 0)
  AP.1 * PF.1 + AP.2 * PF.2 = BP.1 * PF.1 + BP.2 * PF.2

/-- The main theorem -/
theorem ellipse_focal_property :
  ∃! (p : ℝ), p > 0 ∧
    (∀ A B : ℝ × ℝ, chord A B → angle_equality p A B) ∧
    p = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2342_234208


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l2342_234263

/-- The total distance Spencer walked on his errands -/
def total_distance (house_to_library : ℝ) (library_to_post : ℝ) (post_to_grocery : ℝ) (grocery_to_coffee : ℝ) (coffee_to_house : ℝ) : ℝ :=
  house_to_library + library_to_post + post_to_grocery + grocery_to_coffee + coffee_to_house

/-- Theorem stating that Spencer walked 6.1 miles in total -/
theorem spencer_walk_distance :
  total_distance 1.2 0.8 1.5 0.6 2 = 6.1 := by
  sorry


end NUMINAMATH_CALUDE_spencer_walk_distance_l2342_234263


namespace NUMINAMATH_CALUDE_johns_final_weight_l2342_234222

/-- Calculates the final weight after a series of weight changes --/
def final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 0.9  -- 10% loss
  let weight2 := weight1 + 5           -- 5 pounds gain
  let weight3 := weight2 * 0.85        -- 15% loss
  let weight4 := weight3 + 8           -- 8 pounds gain
  weight4 * 0.8                        -- 20% loss

/-- Theorem stating that John's final weight is approximately 144.44 pounds --/
theorem johns_final_weight :
  ∃ ε > 0, |final_weight 220 - 144.44| < ε :=
sorry

end NUMINAMATH_CALUDE_johns_final_weight_l2342_234222


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2342_234253

theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 60 →
  B = 85 →
  C = D →
  E = 2 * C + 15 →
  A + B + C + D + E = 540 →
  max A (max B (max C (max D E))) = 205 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2342_234253


namespace NUMINAMATH_CALUDE_min_value_expression_l2342_234262

theorem min_value_expression (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) ≥ 3 ∧ (x + 4 / (x + 1) = 3 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2342_234262


namespace NUMINAMATH_CALUDE_num_arrangements_l2342_234205

/-- Represents the number of volunteers --/
def num_volunteers : ℕ := 4

/-- Represents the number of communities --/
def num_communities : ℕ := 3

/-- Calculates the number of ways to arrange volunteers into communities --/
def arrange_volunteers : ℕ := sorry

/-- Theorem stating that the number of arrangements is 36 --/
theorem num_arrangements : arrange_volunteers = 36 := by sorry

end NUMINAMATH_CALUDE_num_arrangements_l2342_234205


namespace NUMINAMATH_CALUDE_moving_circles_touch_times_l2342_234202

/-- The problem of two moving circles touching each other --/
theorem moving_circles_touch_times
  (r₁ : ℝ) (v₁ : ℝ) (d₁ : ℝ)
  (r₂ : ℝ) (v₂ : ℝ) (d₂ : ℝ)
  (h₁ : r₁ = 981)
  (h₂ : v₁ = 7)
  (h₃ : d₁ = 2442)
  (h₄ : r₂ = 980)
  (h₅ : v₂ = 5)
  (h₆ : d₂ = 1591) :
  ∃ (t₁ t₂ : ℝ),
    t₁ = 111 ∧ t₂ = 566 ∧
    (∀ t, (d₁ - v₁ * t)^2 + (d₂ - v₂ * t)^2 = (r₁ + r₂)^2 → t = t₁ ∨ t = t₂) :=
by sorry

end NUMINAMATH_CALUDE_moving_circles_touch_times_l2342_234202


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l2342_234281

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    Real.sqrt 2 * (2*a + 3) * Real.cos (θ - π/4) + 6 / (Real.sin θ + Real.cos θ) - 2 * Real.sin (2*θ) < 3*a + 6) 
  → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l2342_234281


namespace NUMINAMATH_CALUDE_work_related_emails_l2342_234285

theorem work_related_emails (total : ℕ) (spam_percent : ℚ) (promo_percent : ℚ) (social_percent : ℚ)
  (h_total : total = 1200)
  (h_spam : spam_percent = 27 / 100)
  (h_promo : promo_percent = 18 / 100)
  (h_social : social_percent = 15 / 100) :
  (total : ℚ) * (1 - (spam_percent + promo_percent + social_percent)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_work_related_emails_l2342_234285


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2342_234295

theorem complex_sum_equality : 
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5
  let R : ℂ := 3*I
  let T : ℂ := 1 + 5*I
  B - Q + R + T = -1 + 10*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2342_234295


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2342_234296

/-- The function f(x) = x^2 / (x^2 + 1) is increasing on the interval (0, +∞) -/
theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
    (x₁^2 / (x₁^2 + 1)) < (x₂^2 / (x₂^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2342_234296


namespace NUMINAMATH_CALUDE_product_over_sum_equals_1120_l2342_234264

theorem product_over_sum_equals_1120 : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) = 1120 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_equals_1120_l2342_234264


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2342_234272

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2342_234272


namespace NUMINAMATH_CALUDE_horner_v4_at_2_l2342_234219

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ := 
  let v1 := x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_v4_at_2 : horner_v4 2 = 240 := by sorry

end NUMINAMATH_CALUDE_horner_v4_at_2_l2342_234219


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2342_234245

theorem arithmetic_calculation : (28 * 9 + 18 * 19 + 8 * 29) / 14 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2342_234245


namespace NUMINAMATH_CALUDE_recipe_ratio_change_l2342_234294

-- Define the original recipe ratios
def original_flour : ℚ := 8
def original_water : ℚ := 4
def original_sugar : ℚ := 3

-- Define the new recipe quantities
def new_water : ℚ := 2
def new_sugar : ℚ := 6

-- Theorem statement
theorem recipe_ratio_change :
  let original_flour_sugar_ratio := original_flour / original_sugar
  let new_flour := (original_flour / original_water) * 2 * new_water
  let new_flour_sugar_ratio := new_flour / new_sugar
  original_flour_sugar_ratio - new_flour_sugar_ratio = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_recipe_ratio_change_l2342_234294


namespace NUMINAMATH_CALUDE_michaels_money_ratio_l2342_234289

/-- Given the following conditions:
    - Michael has $42 initially
    - Michael gives some money to his brother
    - His brother buys $3 worth of candy
    - His brother has $35 left after buying candy
    - His brother had $17 at first
    Prove that the ratio of money Michael gave to his brother to Michael's initial money is 1:2 -/
theorem michaels_money_ratio :
  ∀ (initial_money : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) (brother_final : ℕ),
    initial_money = 42 →
    brother_initial = 17 →
    candy_cost = 3 →
    brother_final = 35 →
    ∃ (money_given : ℕ),
      money_given = brother_final + candy_cost - brother_initial ∧
      2 * money_given = initial_money :=
by sorry

end NUMINAMATH_CALUDE_michaels_money_ratio_l2342_234289


namespace NUMINAMATH_CALUDE_cosine_angle_in_ellipse_l2342_234274

/-- The cosine of the angle F₁PF₂ in an ellipse with specific properties -/
theorem cosine_angle_in_ellipse (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let on_ellipse (p : ℝ × ℝ) := p.1^2 / 25 + p.2^2 / 9 = 1
  let triangle_area (p : ℝ × ℝ) := abs ((p.1 - F₁.1) * (p.2 - F₂.2) - (p.2 - F₁.2) * (p.1 - F₂.1)) / 2
  on_ellipse P ∧ triangle_area P = 3 * Real.sqrt 3 →
  let PF₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let PF₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let cos_angle := (PF₁^2 + PF₂^2 - 64) / (2 * PF₁ * PF₂)
  cos_angle = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_angle_in_ellipse_l2342_234274


namespace NUMINAMATH_CALUDE_clock_angle_at_3_30_l2342_234286

/-- The angle of the hour hand at 3:30 in degrees -/
def hour_hand_angle : ℝ := 105

/-- The angle of the minute hand at 3:30 in degrees -/
def minute_hand_angle : ℝ := 180

/-- The smaller angle between the hour and minute hands at 3:30 -/
def smaller_angle : ℝ := minute_hand_angle - hour_hand_angle

/-- The larger angle between the hour and minute hands at 3:30 -/
def larger_angle : ℝ := 360 - smaller_angle

theorem clock_angle_at_3_30 : larger_angle = 285 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_30_l2342_234286


namespace NUMINAMATH_CALUDE_nicholas_crackers_l2342_234259

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) : 
  marcus_crackers = 3 * mona_crackers →
  nicholas_crackers = mona_crackers + 6 →
  marcus_crackers = 27 →
  nicholas_crackers = 15 := by sorry

end NUMINAMATH_CALUDE_nicholas_crackers_l2342_234259


namespace NUMINAMATH_CALUDE_online_store_commission_percentage_l2342_234273

theorem online_store_commission_percentage 
  (cost : ℝ) 
  (online_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : cost = 17) 
  (h2 : online_price = 25.5) 
  (h3 : profit_percentage = 0.2) : 
  (online_price - (cost * (1 + profit_percentage))) / online_price = 0.2 := by
sorry

end NUMINAMATH_CALUDE_online_store_commission_percentage_l2342_234273


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2342_234237

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (h1 : total_clips = 81) (h2 : clips_per_box = 9) :
  total_clips / clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2342_234237


namespace NUMINAMATH_CALUDE_weight_of_a_l2342_234255

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 75 kg -/
theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l2342_234255


namespace NUMINAMATH_CALUDE_barbaras_allowance_l2342_234226

theorem barbaras_allowance 
  (watch_cost : ℕ) 
  (current_savings : ℕ) 
  (weeks_left : ℕ) 
  (h1 : watch_cost = 100)
  (h2 : current_savings = 20)
  (h3 : weeks_left = 16) :
  (watch_cost - current_savings) / weeks_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_barbaras_allowance_l2342_234226


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l2342_234280

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  y_eq_x_plus_3 : y = x + 3
  product_eq_2541 : x * y * z = 2541

/-- The sum of the dimensions of a box. -/
def sum_dimensions (d : BoxDimensions) : ℕ := d.x + d.y + d.z

/-- Theorem stating the minimum sum of dimensions for the given conditions. -/
theorem min_sum_dimensions :
  ∀ d : BoxDimensions, sum_dimensions d ≥ 38 := by sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l2342_234280


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_g_l2342_234233

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 5| + |x - 3| - |3*x - 15|

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }

-- Theorem statement
theorem sum_of_max_and_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x ∈ domain, g x ≤ max_g) ∧
    (∃ x ∈ domain, g x = max_g) ∧
    (∀ x ∈ domain, min_g ≤ g x) ∧
    (∃ x ∈ domain, g x = min_g) ∧
    (max_g + min_g = -2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_g_l2342_234233


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2342_234275

theorem rectangle_perimeter (l w : ℝ) :
  l > 0 ∧ w > 0 ∧                             -- Positive dimensions
  2 * (w + l / 6) = 40 ∧                       -- Perimeter of smaller rectangle
  6 * w = l →                                  -- Relationship between l and w
  2 * (l + w) = 280 :=                         -- Perimeter of original rectangle
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2342_234275


namespace NUMINAMATH_CALUDE_projection_implies_coplanar_and_parallel_l2342_234266

-- Define a type for 3D points
def Point3D := ℝ × ℝ × ℝ

-- Define a type for 2D points (projections)
def Point2D := ℝ × ℝ

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  d : ℝ

-- Define a projection function
def project (p : Point3D) (plane : Plane) : Point2D :=
  sorry

-- Define a predicate for points being on a line
def onLine (points : List Point2D) : Prop :=
  sorry

-- Define a predicate for points being coplanar
def coplanar (points : List Point3D) : Prop :=
  sorry

-- Define a predicate for points being parallel
def parallel (points : List Point3D) : Prop :=
  sorry

-- The main theorem
theorem projection_implies_coplanar_and_parallel 
  (points : List Point3D) (plane : Plane) :
  onLine (points.map (λ p => project p plane)) →
  coplanar points ∧ parallel points :=
sorry

end NUMINAMATH_CALUDE_projection_implies_coplanar_and_parallel_l2342_234266


namespace NUMINAMATH_CALUDE_revenue_maximizing_price_l2342_234240

/-- Revenue function for the bookstore --/
def revenue (p : ℝ) : ℝ := p * (200 - 6 * p)

/-- The maximum price constraint --/
def max_price : ℝ := 30

/-- Theorem stating the price that maximizes revenue --/
theorem revenue_maximizing_price :
  ∃ (p : ℝ), p ≤ max_price ∧ 
  ∀ (q : ℝ), q ≤ max_price → revenue q ≤ revenue p ∧
  p = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_revenue_maximizing_price_l2342_234240


namespace NUMINAMATH_CALUDE_benny_bought_two_cards_l2342_234213

/-- The number of Pokemon cards Benny bought -/
def cards_bought (initial_cards final_cards : ℕ) : ℕ :=
  initial_cards - final_cards

/-- Proof that Benny bought 2 Pokemon cards -/
theorem benny_bought_two_cards :
  let initial_cards := 3
  let final_cards := 1
  cards_bought initial_cards final_cards = 2 := by
sorry

end NUMINAMATH_CALUDE_benny_bought_two_cards_l2342_234213


namespace NUMINAMATH_CALUDE_min_sum_distances_l2342_234299

/-- Given a parabola and a line, prove the minimum sum of distances -/
theorem min_sum_distances (x y : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8*x}
  let line := {(x, y) : ℝ × ℝ | x - y + 2 = 0}
  let d1 (p : ℝ × ℝ) := |p.1|  -- distance from point to y-axis
  let d2 (p : ℝ × ℝ) := |p.1 - p.2 + 2| / Real.sqrt 2  -- distance from point to line
  ∃ (min : ℝ), ∀ (p : ℝ × ℝ), p ∈ parabola → d1 p + d2 p ≥ min ∧ 
  ∃ (q : ℝ × ℝ), q ∈ parabola ∧ d1 q + d2 q = min ∧ min = 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l2342_234299


namespace NUMINAMATH_CALUDE_brick_length_is_25_cm_l2342_234236

/-- Proves that the length of each brick is 25 cm, given the wall dimensions,
    brick dimensions (except length), and the number of bricks needed. -/
theorem brick_length_is_25_cm
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (num_bricks : ℝ)
  (h_wall_length : wall_length = 2)
  (h_wall_height : wall_height = 3)
  (h_wall_thickness : wall_thickness = 0.02)
  (h_brick_width : brick_width = 0.11)
  (h_brick_height : brick_height = 0.06)
  (h_num_bricks : num_bricks = 72.72727272727273)
  : ∃ (brick_length : ℝ), brick_length = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_25_cm_l2342_234236


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2342_234268

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2342_234268


namespace NUMINAMATH_CALUDE_system_solution_l2342_234218

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 2 * y = 8) ∧ 
  (x + 3 * y = 9) ∧ 
  (x = 42 / 11) ∧ 
  (y = 19 / 11) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2342_234218


namespace NUMINAMATH_CALUDE_candy_eating_problem_l2342_234256

/-- Represents the number of candies eaten by each person -/
structure CandyEaten where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the rates at which each person eats candies -/
structure EatingRates where
  andrey : ℚ
  boris : ℚ
  denis : ℚ

/-- The theorem statement based on the given problem -/
theorem candy_eating_problem (rates : EatingRates) 
  (h1 : rates.andrey * 4 = rates.boris * 3)
  (h2 : rates.denis * 6 = rates.andrey * 7)
  (h3 : rates.andrey + rates.boris + rates.denis = 70) :
  ∃ (eaten : CandyEaten), 
    eaten.andrey = 24 ∧ 
    eaten.boris = 18 ∧ 
    eaten.denis = 28 ∧
    eaten.andrey + eaten.boris + eaten.denis = 70 := by
  sorry

end NUMINAMATH_CALUDE_candy_eating_problem_l2342_234256


namespace NUMINAMATH_CALUDE_min_side_length_is_optimal_l2342_234229

/-- The minimum side length of a square satisfying the given conditions -/
def min_side_length : ℝ := 25

/-- The condition that the area of the square is at least 625 square feet -/
def area_condition (s : ℝ) : Prop := s^2 ≥ 625

/-- The condition that there exists a smaller square inside with side length equal to half the side length of the larger square -/
def inner_square_condition (s : ℝ) : Prop := ∃ (inner_s : ℝ), inner_s = s / 2

/-- Theorem stating that the minimum side length satisfies both conditions and is minimal -/
theorem min_side_length_is_optimal :
  area_condition min_side_length ∧
  inner_square_condition min_side_length ∧
  ∀ s, s < min_side_length → ¬(area_condition s ∧ inner_square_condition s) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_is_optimal_l2342_234229


namespace NUMINAMATH_CALUDE_treehouse_planks_l2342_234239

theorem treehouse_planks (initial_planks : ℕ) (total_planks : ℕ) (planks_from_forest : ℕ) :
  initial_planks = 15 →
  total_planks = 35 →
  total_planks = initial_planks + 2 * planks_from_forest →
  planks_from_forest = 10 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_planks_l2342_234239


namespace NUMINAMATH_CALUDE_subset_iff_a_in_range_l2342_234204

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 < 0}
def B : Set ℝ := {x | (x + 1)/(x - 2) < 0}

-- State the theorem
theorem subset_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_a_in_range_l2342_234204


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2342_234221

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  Set.Ioo (-3 : ℝ) (1/2) = {x | c * x^2 + b * x + a < 0} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2342_234221


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2342_234225

theorem complex_fraction_equality : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2342_234225


namespace NUMINAMATH_CALUDE_total_water_volume_is_10750_l2342_234252

def tank1_capacity : ℚ := 7000
def tank2_capacity : ℚ := 5000
def tank3_capacity : ℚ := 3000

def tank1_fill_ratio : ℚ := 3/4
def tank2_fill_ratio : ℚ := 4/5
def tank3_fill_ratio : ℚ := 1/2

def total_water_volume : ℚ := 
  tank1_capacity * tank1_fill_ratio + 
  tank2_capacity * tank2_fill_ratio + 
  tank3_capacity * tank3_fill_ratio

theorem total_water_volume_is_10750 : total_water_volume = 10750 := by
  sorry

end NUMINAMATH_CALUDE_total_water_volume_is_10750_l2342_234252


namespace NUMINAMATH_CALUDE_complex_number_range_l2342_234216

theorem complex_number_range (m : ℝ) :
  let z : ℂ := 1 + Complex.I + m / (1 + Complex.I)
  (0 < z.re ∧ 0 < z.im) ↔ -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l2342_234216


namespace NUMINAMATH_CALUDE_evaluate_expression_l2342_234284

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  Real.sqrt ((x - 1)^2) + Real.sqrt (x^2 + 4*x + 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2342_234284


namespace NUMINAMATH_CALUDE_kelly_nintendo_games_l2342_234271

/-- Proves that Kelly's initial number of Nintendo games is 121.0 --/
theorem kelly_nintendo_games :
  ∀ x : ℝ, (x - 99 = 22.0) → x = 121.0 := by
  sorry

end NUMINAMATH_CALUDE_kelly_nintendo_games_l2342_234271


namespace NUMINAMATH_CALUDE_sin_B_in_triangle_ABC_l2342_234261

theorem sin_B_in_triangle_ABC (a b : ℝ) (sin_A : ℝ) :
  a = 15 →
  b = 10 →
  sin_A = (Real.sqrt 3) / 2 →
  (b * sin_A) / a = (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_sin_B_in_triangle_ABC_l2342_234261


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2342_234235

theorem division_remainder_problem (D : ℕ) : 
  (D / 12 = 70) → (D / 21 = 40) → (D % 21 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2342_234235


namespace NUMINAMATH_CALUDE_trader_donations_l2342_234227

theorem trader_donations (total_profit : ℝ) (goal_amount : ℝ) (above_goal : ℝ) : 
  total_profit = 960 → 
  goal_amount = 610 → 
  above_goal = 180 → 
  (goal_amount + above_goal) - (total_profit / 2) = 310 := by
sorry

end NUMINAMATH_CALUDE_trader_donations_l2342_234227


namespace NUMINAMATH_CALUDE_x_power_y_equals_243_l2342_234228

theorem x_power_y_equals_243 (x y : ℝ) : 
  y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 5 → x^y = 243 := by sorry

end NUMINAMATH_CALUDE_x_power_y_equals_243_l2342_234228


namespace NUMINAMATH_CALUDE_initial_strawberry_jelly_beans_proof_initial_strawberry_jelly_beans_l2342_234200

theorem initial_strawberry_jelly_beans : ℕ → ℕ → Prop :=
  fun s g =>
    s = 3 * g ∧ (s - 15 = 4 * (g - 15)) → s = 135

-- The proof is omitted
theorem proof_initial_strawberry_jelly_beans :
  ∀ s g : ℕ, initial_strawberry_jelly_beans s g :=
sorry

end NUMINAMATH_CALUDE_initial_strawberry_jelly_beans_proof_initial_strawberry_jelly_beans_l2342_234200


namespace NUMINAMATH_CALUDE_max_value_w_l2342_234298

theorem max_value_w (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ (w_max : ℝ), w_max = 0 ∧ ∀ (w : ℝ), w = x^2 + y^2 - 8 * x → w ≤ w_max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_w_l2342_234298


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2342_234257

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0, 
    this function returns true if they are perpendicular. -/
def are_perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

/-- Theorem stating that for lines l₁: (m+2)x-(m-2)y+2=0 and l₂: 3x+my-1=0,
    if they are perpendicular, then m = 6 or m = -1 -/
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, are_perpendicular (m + 2) (-(m - 2)) 3 m → m = 6 ∨ m = -1 := by
  sorry

#check perpendicular_lines_m_values

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2342_234257


namespace NUMINAMATH_CALUDE_reciprocal_expression_l2342_234282

theorem reciprocal_expression (x y : ℝ) (h : x * y = 1) :
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l2342_234282


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2342_234210

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 24 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2342_234210


namespace NUMINAMATH_CALUDE_linear_iff_m_neq_neg_six_l2342_234290

/-- A function f is linear if there exist constants a and b such that f(x) = ax + b for all x, and a ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f_m for a given m -/
def f_m (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 2) * x + 4 * x - 5

theorem linear_iff_m_neq_neg_six (m : ℝ) :
  IsLinearFunction (f_m m) ↔ m ≠ -6 := by
  sorry

end NUMINAMATH_CALUDE_linear_iff_m_neq_neg_six_l2342_234290


namespace NUMINAMATH_CALUDE_interest_difference_l2342_234265

def principal : ℚ := 250
def rate : ℚ := 4
def time : ℚ := 8

def simple_interest (p r t : ℚ) : ℚ := (p * r * t) / 100

theorem interest_difference :
  principal - simple_interest principal rate time = 170 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l2342_234265


namespace NUMINAMATH_CALUDE_binary_1101011_equals_107_l2342_234242

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1101011_equals_107 :
  binary_to_decimal [true, true, false, true, false, true, true] = 107 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101011_equals_107_l2342_234242


namespace NUMINAMATH_CALUDE_proposition_relation_l2342_234270

theorem proposition_relation (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) →
  (0 < a ∧ a < 1) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → (0 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_l2342_234270


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2342_234223

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to reflect a point about the y-axis -/
def reflectAboutYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Function to get the equation of a line given two points -/
def lineFromPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p1.y * p2.x - p1.x * p2.y }

/-- Theorem stating the equation of the reflected ray -/
theorem reflected_ray_equation
  (start : Point)
  (slope : ℝ)
  (h_start : start = { x := 2, y := 3 })
  (h_slope : slope = 1/2) :
  let intersect : Point := { x := 0, y := 2 }
  let reflected_start : Point := reflectAboutYAxis start
  let reflected_line : Line := lineFromPoints intersect reflected_start
  reflected_line = { a := 1, b := 2, c := -4 } :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2342_234223


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2342_234238

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, -a^2 + 9*a - 14 ≥ 0 → a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2342_234238


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2342_234209

theorem sum_of_roots_quadratic (x : ℝ) (h : x^2 - 3*x = 12) : 
  ∃ y : ℝ, y^2 - 3*y = 12 ∧ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2342_234209


namespace NUMINAMATH_CALUDE_average_salary_l2342_234217

def salary_a : ℕ := 9000
def salary_b : ℕ := 5000
def salary_c : ℕ := 11000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e
def num_people : ℕ := 5

theorem average_salary :
  (total_salary : ℚ) / num_people = 8200 := by sorry

end NUMINAMATH_CALUDE_average_salary_l2342_234217


namespace NUMINAMATH_CALUDE_folded_polyhedron_volume_l2342_234247

/-- Represents a polyhedron formed by folding four squares and two equilateral triangles -/
structure FoldedPolyhedron where
  square_side_length : ℝ
  triangle_side_length : ℝ
  h_square_side : square_side_length = 2
  h_triangle_side : triangle_side_length = Real.sqrt 8

/-- Calculates the volume of the folded polyhedron -/
noncomputable def volume (p : FoldedPolyhedron) : ℝ :=
  (16 * Real.sqrt 2 / 3) * Real.sqrt (2 - Real.sqrt 2)

/-- Theorem stating that the volume of the folded polyhedron is correct -/
theorem folded_polyhedron_volume (p : FoldedPolyhedron) :
  volume p = (16 * Real.sqrt 2 / 3) * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_folded_polyhedron_volume_l2342_234247


namespace NUMINAMATH_CALUDE_profit_percentage_l2342_234203

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.75 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2342_234203


namespace NUMINAMATH_CALUDE_probability_no_shaded_correct_l2342_234269

/-- Represents a 2 by 1001 rectangle with middle squares shaded --/
structure ShadedRectangle where
  width : Nat
  height : Nat
  shaded_column : Nat

/-- The probability of choosing a rectangle without a shaded square --/
def probability_no_shaded (r : ShadedRectangle) : ℚ :=
  500 / 1001

/-- Theorem stating the probability of choosing a rectangle without a shaded square --/
theorem probability_no_shaded_correct (r : ShadedRectangle) 
  (h1 : r.width = 1001) 
  (h2 : r.height = 2) 
  (h3 : r.shaded_column = 501) : 
  probability_no_shaded r = 500 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_shaded_correct_l2342_234269


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2342_234258

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 324375 / 1000000 ∧
  (∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 324375 / 1000000 → a ≤ c) →
  (a : ℕ) + (b : ℕ) = 2119 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2342_234258


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2342_234288

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the radius of the circle
def radius : ℝ := 4

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem statement
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - 1)^2 + (y + 2)^2 = 16) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2342_234288


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2342_234249

/-- The line passing through points A(0, -5) and B(1, 0) has the equation y = 5x - 5 -/
theorem line_equation_through_points (x y : ℝ) : 
  (x = 0 ∧ y = -5) ∨ (x = 1 ∧ y = 0) → y = 5*x - 5 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2342_234249


namespace NUMINAMATH_CALUDE_lecture_hall_rows_l2342_234243

/-- Represents the number of seats in a row of the lecture hall. -/
def seatsInRow (n : ℕ) : ℕ := 12 + 2 * (n - 1)

/-- Represents the total number of seats in the first n rows of the lecture hall. -/
def totalSeats (n : ℕ) : ℕ := n * (seatsInRow 1 + seatsInRow n) / 2

/-- States that the number of rows in the lecture hall is 16, given the conditions. -/
theorem lecture_hall_rows :
  ∃ (n : ℕ),
    n > 0 ∧
    totalSeats n > 400 ∧
    totalSeats n ≤ 440 ∧
    seatsInRow 1 = 12 ∧
    ∀ (i : ℕ), i > 0 → seatsInRow (i + 1) = seatsInRow i + 2 ∧
    n = 16 :=
  sorry

end NUMINAMATH_CALUDE_lecture_hall_rows_l2342_234243


namespace NUMINAMATH_CALUDE_ticket_price_is_28_l2342_234232

/-- The price of a single ticket given the total money, number of tickets, and remaining money -/
def ticket_price (total_money : ℕ) (num_tickets : ℕ) (remaining_money : ℕ) : ℕ :=
  (total_money - remaining_money) / num_tickets

/-- Theorem stating that the ticket price is $28 given the problem conditions -/
theorem ticket_price_is_28 :
  ticket_price 251 6 83 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_is_28_l2342_234232


namespace NUMINAMATH_CALUDE_car_trip_distance_l2342_234297

theorem car_trip_distance (D : ℝ) : D - (1/2) * D - (1/4) * ((1/2) * D) = 135 → D = 360 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l2342_234297


namespace NUMINAMATH_CALUDE_rectangle_inscribed_area_bound_l2342_234211

/-- A triangle represented by three points in the plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A rectangle represented by four points in the plane -/
structure Rectangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle is inscribed in a triangle -/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Theorem: The area of a rectangle inscribed in a triangle does not exceed half of the area of the triangle -/
theorem rectangle_inscribed_area_bound (t : Triangle) (r : Rectangle) :
  isInscribed r t → rectangleArea r ≤ (1/2) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_rectangle_inscribed_area_bound_l2342_234211


namespace NUMINAMATH_CALUDE_expected_expenditure_2017_l2342_234260

/-- Represents the average income over five years -/
def average_income : ℝ := 10

/-- Represents the average expenditure over five years -/
def average_expenditure : ℝ := 8

/-- The slope of the regression line -/
def b_hat : ℝ := 0.76

/-- The y-intercept of the regression line -/
def a_hat : ℝ := average_expenditure - b_hat * average_income

/-- The regression function -/
def regression_function (x : ℝ) : ℝ := b_hat * x + a_hat

/-- The income in 10,000 yuan for which we want to predict the expenditure -/
def income_2017 : ℝ := 15

theorem expected_expenditure_2017 : 
  regression_function income_2017 = 11.8 := by sorry

end NUMINAMATH_CALUDE_expected_expenditure_2017_l2342_234260


namespace NUMINAMATH_CALUDE_exists_divisible_by_2022_l2342_234224

def concatenate_numbers (n m : ℕ) : ℕ :=
  sorry

theorem exists_divisible_by_2022 :
  ∃ n m : ℕ, n > m ∧ m ≥ 1 ∧ (concatenate_numbers n m) % 2022 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_2022_l2342_234224


namespace NUMINAMATH_CALUDE_quadratic_equation_c_value_l2342_234206

theorem quadratic_equation_c_value : 
  ∀ c : ℝ, 
  (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 40) / 4 ∨ x = (-8 - Real.sqrt 40) / 4) → 
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_c_value_l2342_234206


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_implies_k_ge_160_l2342_234251

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem f_monotonically_decreasing_implies_k_ge_160 :
  ∀ k : ℝ, monotonically_decreasing_on (f k) 5 20 → k ≥ 160 :=
sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_implies_k_ge_160_l2342_234251


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2342_234244

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ := seq.a 1 - seq.a 0

/-- Theorem: If (S_2020 / 2020) - (S_20 / 20) = 2000 for an arithmetic sequence,
    then its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : seq.S 2020 / 2020 - seq.S 20 / 20 = 2000) :
  commonDifference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2342_234244


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2342_234279

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1/2023 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2342_234279


namespace NUMINAMATH_CALUDE_sector_area_l2342_234215

/-- Given a sector with central angle 60° and arc length π, its area is 3π/2 -/
theorem sector_area (angle : Real) (arc_length : Real) (area : Real) :
  angle = 60 * (π / 180) →
  arc_length = π →
  area = (angle / (2 * π)) * arc_length * arc_length / angle →
  area = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2342_234215


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2342_234241

def expression : ℕ := 16^4 + 2 * 16^2 + 1 - 15^4

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2342_234241


namespace NUMINAMATH_CALUDE_seventeenth_replacement_in_may_l2342_234278

/-- Represents months of the year -/
inductive Month
| january | february | march | april | may | june | july 
| august | september | october | november | december

/-- Calculates the number of months after January for a given replacement number -/
def monthsAfterStart (replacementNumber : Nat) : Nat :=
  7 * (replacementNumber - 1)

/-- Converts a number of months after January to the corresponding Month -/
def monthsToMonth (months : Nat) : Month :=
  match months % 12 with
  | 0 => Month.january
  | 1 => Month.february
  | 2 => Month.march
  | 3 => Month.april
  | 4 => Month.may
  | 5 => Month.june
  | 6 => Month.july
  | 7 => Month.august
  | 8 => Month.september
  | 9 => Month.october
  | 10 => Month.november
  | _ => Month.december

theorem seventeenth_replacement_in_may : 
  monthsToMonth (monthsAfterStart 17) = Month.may := by
  sorry

end NUMINAMATH_CALUDE_seventeenth_replacement_in_may_l2342_234278


namespace NUMINAMATH_CALUDE_two_angles_in_fourth_quadrant_l2342_234283

def is_fourth_quadrant (angle : Int) : Bool :=
  let normalized := angle % 360
  normalized > 270 || normalized ≤ 0

def count_fourth_quadrant (angles : List Int) : Nat :=
  (angles.filter is_fourth_quadrant).length

theorem two_angles_in_fourth_quadrant :
  count_fourth_quadrant [-20, -400, -2000, 1600] = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_angles_in_fourth_quadrant_l2342_234283


namespace NUMINAMATH_CALUDE_moles_of_ki_formed_l2342_234246

/-- Represents the chemical reaction NH4I + KOH → NH3 + KI + H2O -/
structure ChemicalReaction where
  nh4i : ℝ  -- moles of NH4I
  koh : ℝ   -- moles of KOH
  nh3 : ℝ   -- moles of NH3
  ki : ℝ    -- moles of KI
  h2o : ℝ   -- moles of H2O

/-- The molar mass of NH4I in g/mol -/
def molar_mass_nh4i : ℝ := 144.95

/-- The total mass of NH4I in grams -/
def total_mass_nh4i : ℝ := 435

/-- Theorem stating that the number of moles of KI formed is 3 -/
theorem moles_of_ki_formed
  (reaction : ChemicalReaction)
  (h1 : reaction.koh = 3)
  (h2 : reaction.nh3 = 3)
  (h3 : reaction.h2o = 3)
  (h4 : reaction.nh4i = total_mass_nh4i / molar_mass_nh4i)
  (h5 : reaction.nh4i = reaction.koh) :
  reaction.ki = 3 := by
  sorry

end NUMINAMATH_CALUDE_moles_of_ki_formed_l2342_234246


namespace NUMINAMATH_CALUDE_exponential_inequality_l2342_234254

theorem exponential_inequality (x : ℝ) : 
  (1/4 : ℝ)^(x^2 - 8) > (4 : ℝ)^(-2*x) ↔ -2 < x ∧ x < 4 := by
sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2342_234254


namespace NUMINAMATH_CALUDE_initial_capital_calculation_l2342_234248

def profit_distribution_ratio : ℚ := 2/3
def income_increase : ℕ := 200
def initial_profit_rate : ℚ := 5/100
def final_profit_rate : ℚ := 7/100

theorem initial_capital_calculation (P : ℚ) : 
  P * final_profit_rate * profit_distribution_ratio - 
  P * initial_profit_rate * profit_distribution_ratio = income_increase →
  P = 15000 := by
sorry

end NUMINAMATH_CALUDE_initial_capital_calculation_l2342_234248


namespace NUMINAMATH_CALUDE_trap_is_feeder_l2342_234230

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An interval is a trap (cover) for a sequence if only finitely many terms lie outside it -/
def IsTrap (s : Sequence) (a b : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a ≤ s n ∧ s n ≤ b

/-- An interval is a feeder for a sequence if infinitely many terms lie inside it -/
def IsFeeder (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a ≤ s n ∧ s n ≤ b

/-- Theorem: Every trap is a feeder -/
theorem trap_is_feeder (s : Sequence) (a b : ℝ) :
  IsTrap s a b → IsFeeder s a b := by
  sorry

end NUMINAMATH_CALUDE_trap_is_feeder_l2342_234230


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2342_234231

theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 16) :
  let square_side := square_perimeter / 4
  let rectangle_side := 2 * square_side
  rectangle_side * 4 = 32 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2342_234231


namespace NUMINAMATH_CALUDE_midpoint_sum_l2342_234220

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, 4) and (10, 20) is 18.5 -/
theorem midpoint_sum : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := 4
  let x₂ : ℝ := 10
  let y₂ : ℝ := 20
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 18.5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_l2342_234220


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zero_point_l2342_234250

def f (t : ℝ) (x : ℝ) : ℝ := 4 * x^3 + 3 * t * x^2 - 6 * t^2 * x + t - 1

theorem f_monotonicity_and_zero_point :
  ∀ t : ℝ,
  (t > 0 →
    (∀ x y : ℝ, ((x < y ∧ y < -t) ∨ (t/2 < x ∧ x < y)) → f t x < f t y) ∧
    (∀ x y : ℝ, -t < x ∧ x < y ∧ y < t/2 → f t x > f t y)) ∧
  (t < 0 →
    (∀ x y : ℝ, ((x < y ∧ y < t/2) ∨ (-t < x ∧ x < y)) → f t x < f t y) ∧
    (∀ x y : ℝ, t/2 < x ∧ x < y ∧ y < -t → f t x > f t y)) ∧
  (t > 0 → ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f t x = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zero_point_l2342_234250


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2342_234293

theorem girls_to_boys_ratio (total : ℕ) (girls boys : ℕ) 
  (h1 : total = 36)
  (h2 : girls + boys = total)
  (h3 : girls = boys + 6) : 
  girls * 5 = boys * 7 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2342_234293


namespace NUMINAMATH_CALUDE_not_monotone_decreasing_f_maps_A_to_B_l2342_234267

-- Define the sets A and B
def A : Set ℝ := {1, 4}
def B : Set ℝ := {1, -1, 2, -2}

-- Define the function f
def f (x : ℝ) : ℝ := (x^2)^(1/7)

-- Theorem 1
theorem not_monotone_decreasing (f : ℝ → ℝ) (h : f 2 < f 3) :
  ¬(∀ x y : ℝ, x ≤ y → f x ≥ f y) :=
sorry

-- Theorem 2
theorem f_maps_A_to_B :
  ∀ x ∈ A, f x ∈ B :=
sorry

end NUMINAMATH_CALUDE_not_monotone_decreasing_f_maps_A_to_B_l2342_234267


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l2342_234291

/-- Given two orthonormal vectors e₁ and e₂, and vectors AC and BD defined in terms of e₁ and e₂,
    prove that the area of quadrilateral ABCD is 10. -/
theorem area_of_quadrilateral (e₁ e₂ : ℝ × ℝ) 
    (h_orthonormal : e₁ • e₁ = 1 ∧ e₂ • e₂ = 1 ∧ e₁ • e₂ = 0) 
    (AC : ℝ × ℝ) (h_AC : AC = 3 • e₁ - e₂)
    (BD : ℝ × ℝ) (h_BD : BD = 2 • e₁ + 6 • e₂) : 
  Real.sqrt ((AC.1^2 + AC.2^2) * (BD.1^2 + BD.2^2)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l2342_234291


namespace NUMINAMATH_CALUDE_certain_number_proof_l2342_234276

theorem certain_number_proof : ∃ x : ℝ, (0.7 * x = 0.4 * 1050) ∧ (x = 600) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2342_234276


namespace NUMINAMATH_CALUDE_bakery_revenue_l2342_234277

/-- Calculates the total revenue from selling pumpkin and custard pies --/
def total_revenue (pumpkin_slices_per_pie : ℕ) (custard_slices_per_pie : ℕ) 
                  (pumpkin_price_per_slice : ℕ) (custard_price_per_slice : ℕ) 
                  (pumpkin_pies_sold : ℕ) (custard_pies_sold : ℕ) : ℕ :=
  (pumpkin_slices_per_pie * pumpkin_pies_sold * pumpkin_price_per_slice) +
  (custard_slices_per_pie * custard_pies_sold * custard_price_per_slice)

theorem bakery_revenue : 
  total_revenue 8 6 5 6 4 5 = 340 := by
  sorry

end NUMINAMATH_CALUDE_bakery_revenue_l2342_234277


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2342_234201

theorem fraction_decomposition (x C D : ℚ) : 
  (7 * x - 15) / (3 * x^2 - x - 4) = C / (x - 1) + D / (3 * x + 4) →
  3 * x^2 - x - 4 = (3 * x + 4) * (x - 1) →
  C = -8/7 ∧ D = 73/7 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2342_234201


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l2342_234207

/-- The number of ways to arrange 4 boys and 3 girls in a line such that no two boys are adjacent -/
def boys_not_adjacent : ℕ := 144

/-- The number of ways to arrange 4 boys and 3 girls in a line such that no two girls are adjacent -/
def girls_not_adjacent : ℕ := 1440

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

theorem photo_lineup_arrangements :
  (boys_not_adjacent = 144) ∧ (girls_not_adjacent = 1440) := by
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l2342_234207


namespace NUMINAMATH_CALUDE_fraction_of_half_is_one_seventh_l2342_234292

theorem fraction_of_half_is_one_seventh : (1 : ℚ) / 7 / ((1 : ℚ) / 2) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_half_is_one_seventh_l2342_234292


namespace NUMINAMATH_CALUDE_initial_number_count_l2342_234214

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  n > 0 ∧ 
  S / n = 12 ∧ 
  (S - 20) / (n - 1) = 10 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_number_count_l2342_234214


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l2342_234212

/-- The radius of a circle inscribed in a quadrilateral with sides 3, 6, 5, and 8 is less than 3 -/
theorem inscribed_circle_radius_bound (r : ℝ) : 
  r > 0 → -- r is positive (radius)
  r * 11 = 12 * Real.sqrt 5 → -- area formula: S = r * s, where s = (3 + 6 + 5 + 8) / 2 = 11
  r < 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l2342_234212


namespace NUMINAMATH_CALUDE_sqrt_14_bounds_l2342_234287

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_bounds_l2342_234287


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2342_234234

theorem triangle_angle_c (a b : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 6 →
  b = 2 * Real.sqrt 3 →
  a * Real.sin B = b * Real.sin A →
  A + B + C = π →
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2342_234234
