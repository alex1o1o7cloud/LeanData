import Mathlib

namespace apple_arrangements_l3334_333448

/-- The number of distinct arrangements of letters in a word with repeated letters -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "APPLE" has 5 letters with 'P' repeating twice -/
def appleWord : (ℕ × List ℕ) := (5, [2])

theorem apple_arrangements :
  distinctArrangements appleWord.1 appleWord.2 = 60 := by
  sorry

end apple_arrangements_l3334_333448


namespace painting_payment_l3334_333447

theorem painting_payment (rate : ℚ) (rooms : ℚ) (h1 : rate = 13 / 3) (h2 : rooms = 8 / 5) :
  rate * rooms = 104 / 15 := by
sorry

end painting_payment_l3334_333447


namespace cake_cost_calculation_l3334_333471

/-- The cost of a cake given initial money and remaining money after purchase -/
def cake_cost (initial_money remaining_money : ℚ) : ℚ :=
  initial_money - remaining_money

theorem cake_cost_calculation (initial_money remaining_money : ℚ) 
  (h1 : initial_money = 59.5)
  (h2 : remaining_money = 42) : 
  cake_cost initial_money remaining_money = 17.5 := by
  sorry

#eval cake_cost 59.5 42

end cake_cost_calculation_l3334_333471


namespace max_distance_in_parallelepiped_l3334_333438

/-- The maximum distance between two points in a 3x4x2 rectangular parallelepiped --/
theorem max_distance_in_parallelepiped :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 2
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ),
    0 ≤ x₁ ∧ x₁ ≤ a ∧
    0 ≤ y₁ ∧ y₁ ≤ b ∧
    0 ≤ z₁ ∧ z₁ ≤ c ∧
    0 ≤ x₂ ∧ x₂ ≤ a ∧
    0 ≤ y₂ ∧ y₂ ≤ b ∧
    0 ≤ z₂ ∧ z₂ ≤ c ∧
    ∀ (x₃ y₃ z₃ x₄ y₄ z₄ : ℝ),
      0 ≤ x₃ ∧ x₃ ≤ a ∧
      0 ≤ y₃ ∧ y₃ ≤ b ∧
      0 ≤ z₃ ∧ z₃ ≤ c ∧
      0 ≤ x₄ ∧ x₄ ≤ a ∧
      0 ≤ y₄ ∧ y₄ ≤ b ∧
      0 ≤ z₄ ∧ z₄ ≤ c →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 ≥ (x₃ - x₄)^2 + (y₃ - y₄)^2 + (z₃ - z₄)^2 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 = 29 := by
  sorry

end max_distance_in_parallelepiped_l3334_333438


namespace adult_dogs_adopted_l3334_333430

/-- The number of adult dogs adopted given the costs and number of other animals -/
def num_adult_dogs (cat_cost puppy_cost adult_dog_cost total_cost : ℕ) 
                   (num_cats num_puppies : ℕ) : ℕ :=
  (total_cost - cat_cost * num_cats - puppy_cost * num_puppies) / adult_dog_cost

theorem adult_dogs_adopted :
  num_adult_dogs 50 150 100 700 2 2 = 3 := by
  sorry

end adult_dogs_adopted_l3334_333430


namespace parallelogram_rotational_symmetry_l3334_333413

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- A parallelogram is a quadrilateral with opposite sides parallel -/
def is_parallelogram (p : Polygon) : Prop :=
  p.vertices.length = 4 ∧
  ∃ (a b c d : ℝ × ℝ), p.vertices = [a, b, c, d] ∧
    (b.1 - a.1, b.2 - a.2) = (d.1 - c.1, d.2 - c.2) ∧
    (c.1 - b.1, c.2 - b.2) = (a.1 - d.1, a.2 - d.2)

/-- Rotation by 180 degrees around a point -/
def rotate_180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- A polygon coincides with itself after 180-degree rotation -/
def coincides_after_rotation (p : Polygon) : Prop :=
  ∃ (center : ℝ × ℝ), 
    ∀ v ∈ p.vertices, (rotate_180 v center) ∈ p.vertices

theorem parallelogram_rotational_symmetry :
  ∀ (p : Polygon), is_parallelogram p → coincides_after_rotation p :=
sorry

end parallelogram_rotational_symmetry_l3334_333413


namespace smallest_integer_square_triple_l3334_333485

theorem smallest_integer_square_triple (x : ℤ) : x^2 = 3*x + 75 → x ≥ -5 :=
by sorry

end smallest_integer_square_triple_l3334_333485


namespace y_range_l3334_333434

theorem y_range (a b y : ℝ) (h1 : a + b = 2) (h2 : b ≤ 2) (h3 : y - a^2 - 2*a + 2 = 0) :
  y ≥ -2 := by
  sorry

end y_range_l3334_333434


namespace joe_paint_usage_l3334_333404

theorem joe_paint_usage (total_paint : ℝ) (used_paint : ℝ) 
  (h1 : total_paint = 360)
  (h2 : used_paint = 225) : 
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint + 
    (1 / 2) * (total_paint - first_week_fraction * total_paint) = used_paint ∧
    first_week_fraction = 1 / 4 := by
  sorry

end joe_paint_usage_l3334_333404


namespace scientific_notation_of_million_l3334_333402

/-- Prove that 1.6369 million is equal to 1.6369 × 10^6 -/
theorem scientific_notation_of_million (x : ℝ) : 
  x * 1000000 = x * (10 ^ 6) :=
by sorry

end scientific_notation_of_million_l3334_333402


namespace mothers_age_l3334_333463

-- Define variables for current ages
variable (A : ℕ) -- Allen's current age
variable (M : ℕ) -- Mother's current age
variable (S : ℕ) -- Sister's current age

-- Define the conditions
axiom allen_younger : A = M - 30
axiom sister_older : S = A + 5
axiom future_sum : (A + 7) + (M + 7) + (S + 7) = 110
axiom mother_sister_diff : M - S = 25

-- Theorem to prove
theorem mothers_age : M = 48 := by
  sorry

end mothers_age_l3334_333463


namespace negation_equivalence_l3334_333456

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
by sorry

end negation_equivalence_l3334_333456


namespace dvd_average_price_l3334_333490

/-- Calculates the average price of DVDs bought from different boxes -/
theorem dvd_average_price (box1_count box1_price box2_count box2_price box3_count box3_price : ℚ) :
  box1_count = 10 →
  box1_price = 2 →
  box2_count = 5 →
  box2_price = 5 →
  box3_count = 3 →
  box3_price = 7 →
  (box1_count * box1_price + box2_count * box2_price + box3_count * box3_price) / 
  (box1_count + box2_count + box3_count) = 367/100 := by
  sorry

#eval (10 * 2 + 5 * 5 + 3 * 7) / (10 + 5 + 3)

end dvd_average_price_l3334_333490


namespace right_triangle_shorter_leg_l3334_333401

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- Shorter leg length
by sorry

end right_triangle_shorter_leg_l3334_333401


namespace problem_solution_l3334_333454

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ - 2 > 0

-- Theorem to prove
theorem problem_solution : ¬p ∧ q := by sorry

end problem_solution_l3334_333454


namespace ceiling_floor_calculation_l3334_333433

theorem ceiling_floor_calculation : 
  ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_calculation_l3334_333433


namespace perfect_square_preserver_iff_square_multiple_l3334_333464

/-- A function is a perfect square preserver if it preserves the property of
    the sum of three distinct positive integers being a perfect square. -/
def IsPerfectSquarePreserver (f : ℕ → ℕ) : Prop :=
  ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (∃ n : ℕ, x + y + z = n^2) ↔ (∃ m : ℕ, f x + f y + f z = m^2)

/-- A function is a square multiple if it's of the form f(x) = k²x for some k ∈ ℕ. -/
def IsSquareMultiple (f : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ x : ℕ, f x = k^2 * x

theorem perfect_square_preserver_iff_square_multiple (f : ℕ → ℕ) :
  IsPerfectSquarePreserver f ↔ IsSquareMultiple f := by
  sorry

end perfect_square_preserver_iff_square_multiple_l3334_333464


namespace quadrilateral_area_l3334_333466

/-- The area of a quadrilateral with one diagonal of length 50 cm and offsets of 10 cm and 8 cm is 450 cm². -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) 
  (h1 : diagonal = 50) 
  (h2 : offset1 = 10) 
  (h3 : offset2 = 8) : 
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 450 :=
by sorry

end quadrilateral_area_l3334_333466


namespace parabola_line_slope_l3334_333492

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a point on the latus rectum
def on_latus_rectum (x y : ℝ) : Prop := x = -1

-- Define a point in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the midpoint condition
def is_midpoint (x1 y1 x2 y2 x3 y3 : ℝ) : Prop := 
  x2 = (x1 + x3) / 2 ∧ y2 = (y1 + y3) / 2

theorem parabola_line_slope (k : ℝ) (x1 y1 x2 y2 x3 y3 : ℝ) : 
  parabola x1 y1 →
  parabola x2 y2 →
  line k x1 y1 →
  line k x2 y2 →
  line k x3 y3 →
  on_latus_rectum x3 y3 →
  in_first_quadrant x1 y1 →
  is_midpoint x1 y1 x2 y2 x3 y3 →
  k = 2 * Real.sqrt 2 := by
sorry

end parabola_line_slope_l3334_333492


namespace company_kw_price_percentage_l3334_333481

theorem company_kw_price_percentage (price kw : ℝ) (assets_a assets_b : ℝ) 
  (h1 : price = 2 * assets_b)
  (h2 : price = 0.75 * (assets_a + assets_b))
  (h3 : ∃ x : ℝ, price = assets_a * (1 + x / 100)) :
  ∃ x : ℝ, x = 20 ∧ price = assets_a * (1 + x / 100) :=
sorry

end company_kw_price_percentage_l3334_333481


namespace coefficient_x_cubed_l3334_333436

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 5x + 3)(4x^3 + 5x^2 + 6x + 8) is 61 -/
theorem coefficient_x_cubed (x : ℝ) : 
  let p₁ : Polynomial ℝ := 3 * X^3 + 2 * X^2 + 5 * X + 3
  let p₂ : Polynomial ℝ := 4 * X^3 + 5 * X^2 + 6 * X + 8
  (p₁ * p₂).coeff 3 = 61 := by
  sorry

end coefficient_x_cubed_l3334_333436


namespace combined_salaries_l3334_333409

/-- The combined salaries of four employees given the salary of the fifth and the average of all five -/
theorem combined_salaries (salary_C average_salary : ℕ) 
  (hC : salary_C = 14000)
  (havg : average_salary = 8600) :
  salary_C + 4 * average_salary - 5 * average_salary = 29000 := by
  sorry

end combined_salaries_l3334_333409


namespace max_sphere_radius_squared_max_sphere_radius_squared_achievable_l3334_333408

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of three cones and a sphere -/
structure ConeConfiguration where
  cone : Cone
  axisIntersectionDistance : ℝ
  sphereRadius : ℝ

/-- Checks if the configuration is valid -/
def isValidConfiguration (config : ConeConfiguration) : Prop :=
  config.cone.baseRadius = 4 ∧
  config.cone.height = 10 ∧
  config.axisIntersectionDistance = 5

/-- Theorem stating the maximum possible value of r^2 -/
theorem max_sphere_radius_squared (config : ConeConfiguration) 
  (h : isValidConfiguration config) : 
  config.sphereRadius ^ 2 ≤ 100 / 29 := by
  sorry

/-- Theorem stating that the maximum value is achievable -/
theorem max_sphere_radius_squared_achievable : 
  ∃ (config : ConeConfiguration), isValidConfiguration config ∧ config.sphereRadius ^ 2 = 100 / 29 := by
  sorry

end max_sphere_radius_squared_max_sphere_radius_squared_achievable_l3334_333408


namespace intersection_of_A_and_B_l3334_333455

def A : Set ℕ := {2, 4, 6, 8}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end intersection_of_A_and_B_l3334_333455


namespace problem_solution_l3334_333499

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 + 2) + 1 / (x - Real.sqrt (x^2 + 2)) = 15 →
  x^2 + Real.sqrt (x^4 + 2) + 1 / (x^2 + Real.sqrt (x^4 + 2)) = 47089 / 1800 :=
by sorry

end problem_solution_l3334_333499


namespace third_term_is_eight_thirds_l3334_333417

/-- The sequence defined by a_n = n - 1/n -/
def a (n : ℕ) : ℚ := n - 1 / n

/-- Theorem: The third term of the sequence a_n is 8/3 -/
theorem third_term_is_eight_thirds : a 3 = 8 / 3 := by
  sorry

end third_term_is_eight_thirds_l3334_333417


namespace octagon_non_intersecting_diagonals_l3334_333415

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of non-intersecting diagonals in a star pattern for a regular polygon -/
def nonIntersectingDiagonals (p : RegularPolygon n) : ℕ := n

/-- Theorem: For an octagon, the number of non-intersecting diagonals in a star pattern
    is equal to the number of sides -/
theorem octagon_non_intersecting_diagonals :
  ∀ (p : RegularPolygon 8), nonIntersectingDiagonals p = 8 := by
  sorry

end octagon_non_intersecting_diagonals_l3334_333415


namespace quadratic_conditions_l3334_333495

/-- The quadratic function f(x) = x^2 - 4x - 3 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 3 + a

/-- Theorem stating the conditions for the quadratic function -/
theorem quadratic_conditions :
  (∃ a : ℝ, f a 0 = 1 ∧ a = 4) ∧
  (∃ a : ℝ, (∀ x : ℝ, f a x = 0 → x = 0 ∨ x ≠ 0) ∧ (a = 3 ∨ a = 7)) :=
by sorry

end quadratic_conditions_l3334_333495


namespace rectangle_area_l3334_333426

def square_side : ℝ := 15
def rectangle_length : ℝ := 18

theorem rectangle_area (rectangle_width : ℝ) :
  (4 * square_side = 2 * (rectangle_length + rectangle_width)) →
  (rectangle_length * rectangle_width = 216) := by
  sorry

end rectangle_area_l3334_333426


namespace similar_triangles_side_length_l3334_333443

/-- Given two similar triangles with areas A₁ and A₂, where A₁ > A₂,
    prove that the corresponding side of the larger triangle is 12 feet. -/
theorem similar_triangles_side_length 
  (A₁ A₂ : ℝ) 
  (h_positive : A₁ > A₂) 
  (h_diff : A₁ - A₂ = 27) 
  (h_ratio : A₁ / A₂ = 9) 
  (h_small_side : ∃ (s : ℝ), s = 4 ∧ s * s / 2 ≤ A₂) : 
  ∃ (S : ℝ), S = 12 ∧ S * S / 2 ≤ A₁ := by
  sorry

end similar_triangles_side_length_l3334_333443


namespace optimal_solution_l3334_333477

-- Define the normal distribution parameters
def μ : ℝ := 800
def σ : ℝ := 50

-- Define the probability p₀
def p₀ : ℝ := 0.9772

-- Define vehicle capacities and costs
def capacity_A : ℕ := 36
def capacity_B : ℕ := 60
def cost_A : ℕ := 1600
def cost_B : ℕ := 2400

-- Define the optimization problem
def optimal_fleet (a b : ℕ) : Prop :=
  -- Total vehicles constraint
  a + b ≤ 21 ∧
  -- Type B vehicles constraint
  b ≤ a + 7 ∧
  -- Probability constraint (simplified)
  (a * capacity_A + b * capacity_B : ℝ) ≥ μ + σ * 2 ∧
  -- Minimizes cost
  ∀ a' b' : ℕ,
    (a' * capacity_A + b' * capacity_B : ℝ) ≥ μ + σ * 2 →
    a' + b' ≤ 21 →
    b' ≤ a' + 7 →
    a * cost_A + b * cost_B ≤ a' * cost_A + b' * cost_B

-- Theorem statement
theorem optimal_solution :
  optimal_fleet 5 12 :=
sorry

end optimal_solution_l3334_333477


namespace quarter_circle_sum_limit_l3334_333486

theorem quarter_circle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * D / (4 * n)) - (π * D / 4)| < ε :=
sorry

end quarter_circle_sum_limit_l3334_333486


namespace water_height_in_cone_l3334_333411

theorem water_height_in_cone (r h : ℝ) (water_ratio : ℝ) :
  r = 16 →
  h = 96 →
  water_ratio = 1/4 →
  (water_ratio * (1/3 * π * r^2 * h) = 1/3 * π * r^2 * (48 * Real.rpow 2 (1/3))) :=
by sorry

end water_height_in_cone_l3334_333411


namespace half_abs_diff_squares_20_15_l3334_333451

theorem half_abs_diff_squares_20_15 : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_20_15_l3334_333451


namespace similar_triangles_side_length_l3334_333410

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop :=
  ∃ k > 0, ∀ (s1 s2 : ℝ × ℝ), s1 ∈ t1 → s2 ∈ t2 → ‖s1.1 - s1.2‖ = k * ‖s2.1 - s2.2‖

theorem similar_triangles_side_length 
  (PQR STU : Set (ℝ × ℝ))
  (h_similar : similar_triangles PQR STU)
  (h_PQ : ∃ PQ ∈ PQR, ‖PQ.1 - PQ.2‖ = 7)
  (h_PR : ∃ PR ∈ PQR, ‖PR.1 - PR.2‖ = 9)
  (h_ST : ∃ ST ∈ STU, ‖ST.1 - ST.2‖ = 4.2)
  : ∃ SU ∈ STU, ‖SU.1 - SU.2‖ = 5.4 := by
  sorry

end similar_triangles_side_length_l3334_333410


namespace sum_of_reciprocals_negative_l3334_333460

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_eight : a * b * c = 8) : 
  1 / a + 1 / b + 1 / c < 0 := by
  sorry

end sum_of_reciprocals_negative_l3334_333460


namespace shop_profit_calculation_l3334_333478

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 34

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℝ := 192

/-- The difference in cost between a t-shirt and a jersey -/
def cost_difference : ℝ := 158

theorem shop_profit_calculation :
  jersey_profit = tshirt_profit - cost_difference :=
by sorry

end shop_profit_calculation_l3334_333478


namespace greatest_integer_satisfying_inequality_l3334_333425

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 6 * x > 23) → x ≤ -3 ∧ 7 - 6 * (-3) > 23 :=
by sorry

end greatest_integer_satisfying_inequality_l3334_333425


namespace fraction_inequality_not_sufficient_nor_necessary_sufficient_condition_implies_subset_l3334_333429

-- Statement B
theorem fraction_inequality_not_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, (1 / a > 1 / b → a < b) ∧ (a < b → 1 / a > 1 / b)) := by sorry

-- Statement C
theorem sufficient_condition_implies_subset (A B : Set α) :
  (∀ x, x ∈ A → x ∈ B) → A ⊆ B := by sorry

end fraction_inequality_not_sufficient_nor_necessary_sufficient_condition_implies_subset_l3334_333429


namespace equilateral_triangle_to_three_layered_quadrilateral_l3334_333421

/-- Represents a polygon with a specified number of sides -/
structure Polygon where
  sides : ℕ
  deriving Repr

/-- Represents a folded shape -/
structure FoldedShape where
  shape : Polygon
  layers : ℕ
  deriving Repr

/-- Represents an equilateral triangle -/
def EquilateralTriangle : Polygon :=
  { sides := 3 }

/-- Represents a quadrilateral -/
def Quadrilateral : Polygon :=
  { sides := 4 }

/-- Folding operation that transforms one shape into another -/
def fold (start : Polygon) (result : FoldedShape) : Prop :=
  sorry

/-- Theorem stating that an equilateral triangle can be folded into a three-layered quadrilateral -/
theorem equilateral_triangle_to_three_layered_quadrilateral :
  ∃ (result : FoldedShape), 
    result.shape = Quadrilateral ∧ 
    result.layers = 3 ∧ 
    fold EquilateralTriangle result :=
by sorry

end equilateral_triangle_to_three_layered_quadrilateral_l3334_333421


namespace one_of_each_color_probability_l3334_333439

/-- Probability of selecting one marble of each color -/
theorem one_of_each_color_probability
  (total_marbles : Nat)
  (red_marbles blue_marbles green_marbles : Nat)
  (h1 : total_marbles = red_marbles + blue_marbles + green_marbles)
  (h2 : red_marbles = 3)
  (h3 : blue_marbles = 3)
  (h4 : green_marbles = 2)
  (h5 : total_marbles = 8) :
  (red_marbles * blue_marbles * green_marbles : Rat) /
  (Nat.choose total_marbles 3 : Rat) = 9 / 28 := by
  sorry

end one_of_each_color_probability_l3334_333439


namespace equation_solutions_l3334_333406

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x + 1)^2 = 8 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 - x - 6 = 0 ↔ x = -3/2 ∨ x = 2) :=
by sorry

end equation_solutions_l3334_333406


namespace abs_neg_three_eq_three_l3334_333431

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_eq_three_l3334_333431


namespace inverse_proportion_m_value_l3334_333442

-- Define the function y as an inverse proportion function
def is_inverse_proportion (m : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → (m - 2) * x^(m^2 - 5) = k / x

-- State the theorem
theorem inverse_proportion_m_value :
  ∀ m : ℝ, is_inverse_proportion m → m - 2 ≠ 0 → m = -2 := by
  sorry

end inverse_proportion_m_value_l3334_333442


namespace tangent_line_minimum_sum_l3334_333414

theorem tangent_line_minimum_sum (m n : ℝ) : 
  m > 0 → 
  n > 0 → 
  (∃ x : ℝ, (1/Real.exp 1) * x + m + 1 = Real.log x - n + 2 ∧ 
             (1/Real.exp 1) = 1/x) → 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/a + 1/b ≥ 1/m + 1/n) →
  1/m + 1/n = 4 := by
sorry

end tangent_line_minimum_sum_l3334_333414


namespace interior_nodes_theorem_l3334_333462

/-- A point with integer coordinates -/
structure Node where
  x : ℤ
  y : ℤ

/-- A triangle with vertices at nodes -/
structure Triangle where
  a : Node
  b : Node
  c : Node

/-- Checks if a node is inside a triangle -/
def Node.isInside (n : Node) (t : Triangle) : Prop := sorry

/-- Checks if a line through two nodes contains a vertex of the triangle -/
def Line.containsVertex (p q : Node) (t : Triangle) : Prop := sorry

/-- Checks if a line through two nodes is parallel to a side of the triangle -/
def Line.isParallelToSide (p q : Node) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem interior_nodes_theorem (t : Triangle) 
  (h : ∃ (p q : Node), p.isInside t ∧ q.isInside t ∧ p ≠ q) :
  ∃ (x y : Node), 
    x.isInside t ∧ 
    y.isInside t ∧ 
    x ≠ y ∧
    (Line.containsVertex x y t ∨ Line.isParallelToSide x y t) := by
  sorry

end interior_nodes_theorem_l3334_333462


namespace running_match_participants_l3334_333403

theorem running_match_participants : 
  ∀ (n : ℕ), 
  (∃ (participant : ℕ), 
    participant ≤ n ∧ 
    participant > 0 ∧
    n - 1 = 25) →
  n = 26 :=
by
  sorry

end running_match_participants_l3334_333403


namespace min_vertical_distance_l3334_333416

-- Define the two functions
def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x : ℝ), vertical_distance x = 8 ∧ 
  ∀ (y : ℝ), vertical_distance y ≥ 8 := by
sorry

end min_vertical_distance_l3334_333416


namespace product_of_symmetric_complex_numbers_l3334_333472

def symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem product_of_symmetric_complex_numbers :
  ∀ z₁ z₂ : ℂ, 
    symmetric_about_imaginary_axis z₁ z₂ → 
    z₁ = 1 + 2*I → 
    z₁ * z₂ = -5 := by
  sorry

end product_of_symmetric_complex_numbers_l3334_333472


namespace shorter_diagonal_length_l3334_333470

theorem shorter_diagonal_length (a b : ℝ × ℝ) :
  ‖a‖ = 2 →
  ‖b‖ = 4 →
  a • b = 4 →
  ‖a - b‖ = 2 * Real.sqrt 3 :=
by sorry

end shorter_diagonal_length_l3334_333470


namespace arithmetic_sequence_angles_eq_solution_angles_l3334_333467

open Real

-- Define the set of angles that satisfy the condition
def ArithmeticSequenceAngles : Set ℝ :=
  {a | 0 < a ∧ a < 2 * π ∧ 2 * sin (2 * a) = sin a + sin (3 * a)}

-- Define the set of solution angles in radians
def SolutionAngles : Set ℝ :=
  {π/6, 5*π/6, 7*π/6, 11*π/6}

-- Theorem statement
theorem arithmetic_sequence_angles_eq_solution_angles :
  ArithmeticSequenceAngles = SolutionAngles := by sorry

end arithmetic_sequence_angles_eq_solution_angles_l3334_333467


namespace total_books_combined_l3334_333418

theorem total_books_combined (bryan_books_per_shelf : ℕ) (bryan_shelves : ℕ) 
  (alyssa_books_per_shelf : ℕ) (alyssa_shelves : ℕ) : 
  bryan_books_per_shelf = 56 → 
  bryan_shelves = 9 → 
  alyssa_books_per_shelf = 73 → 
  alyssa_shelves = 12 → 
  bryan_books_per_shelf * bryan_shelves + alyssa_books_per_shelf * alyssa_shelves = 1380 := by
  sorry

end total_books_combined_l3334_333418


namespace unique_geometric_progression_pair_l3334_333427

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (x y z w : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r ∧ w = z * r

/-- There exists exactly one pair of real numbers (a, b) such that 12, a, b, ab form a geometric progression. -/
theorem unique_geometric_progression_pair :
  ∃! (a b : ℝ), IsGeometricProgression 12 a b (a * b) := by
  sorry

#check unique_geometric_progression_pair

end unique_geometric_progression_pair_l3334_333427


namespace zero_is_monomial_l3334_333440

/-- Definition of a monomial -/
def is_monomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ (k : ℕ), expr k = if k = n then c else 0

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : is_monomial (λ _ => 0) := by
  sorry

end zero_is_monomial_l3334_333440


namespace singing_percentage_is_32_l3334_333424

/-- Represents the rehearsal schedule and calculates the percentage of time spent singing -/
def rehearsal_schedule (total_time warm_up_time notes_time words_time : ℕ) : ℚ :=
  let singing_time := total_time - warm_up_time - notes_time - words_time
  (singing_time : ℚ) / total_time * 100

/-- Theorem stating that the percentage of time spent singing is 32% -/
theorem singing_percentage_is_32 :
  ∃ (words_time : ℕ), rehearsal_schedule 75 6 30 words_time = 32 := by
  sorry


end singing_percentage_is_32_l3334_333424


namespace min_abs_z_plus_i_l3334_333444

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 16) = Complex.abs (z * (z + 4*I))) :
  ∃ (w : ℂ), Complex.abs (w + I) = 3 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 16) = Complex.abs (z * (z + 4*I)) → Complex.abs (z + I) ≥ 3 :=
by sorry

end min_abs_z_plus_i_l3334_333444


namespace fraction_difference_l3334_333453

theorem fraction_difference (p q : ℝ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (10 / 12 : ℝ) - (3 / 21 : ℝ) = 29 / 42 := by
  sorry

end fraction_difference_l3334_333453


namespace problem_statement_l3334_333473

theorem problem_statement (a b : ℝ) (h1 : a * b = -3) (h2 : a + b = 2) :
  a^2 * b + a * b^2 = -6 := by sorry

end problem_statement_l3334_333473


namespace total_guesses_l3334_333474

def digits : List ℕ := [1, 1, 1, 1, 2, 2, 2, 2]

def valid_partition (p : List ℕ) : Prop :=
  p.length = 4 ∧ p.sum = 8 ∧ ∀ x ∈ p, 1 ≤ x ∧ x ≤ 3

def num_arrangements : ℕ := Nat.choose 8 4

def num_partitions : ℕ := 35

theorem total_guesses :
  num_arrangements * num_partitions = 2450 :=
sorry

end total_guesses_l3334_333474


namespace solution_difference_l3334_333468

-- Define the equation
def equation (x : ℝ) : Prop := (4 - x^2 / 3)^(1/3) = -2

-- Define the set of solutions
def solutions : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 12 :=
sorry

end solution_difference_l3334_333468


namespace brennans_pepper_theorem_l3334_333437

/-- The amount of pepper remaining after using some from an initial amount -/
def pepper_remaining (initial : ℝ) (used : ℝ) : ℝ :=
  initial - used

/-- Theorem: Given 0.25 grams of pepper initially and using 0.16 grams, 
    the remaining amount is 0.09 grams -/
theorem brennans_pepper_theorem :
  pepper_remaining 0.25 0.16 = 0.09 := by
  sorry

end brennans_pepper_theorem_l3334_333437


namespace triangle_perimeter_l3334_333407

/-- A triangle with two sides of lengths 3 and 4, and the third side length being a root of x^2 - 12x + 35 = 0 has a perimeter of 12. -/
theorem triangle_perimeter : ∃ (a b c : ℝ), 
  a = 3 ∧ b = 4 ∧ c^2 - 12*c + 35 = 0 ∧ 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a + b + c = 12 := by
sorry

end triangle_perimeter_l3334_333407


namespace islet_cell_transplant_indicators_l3334_333498

/-- Represents the type of transplantation performed -/
inductive TransplantationType
| IsletCell

/-- Represents the possible indicators for determining cure and medication needed -/
inductive Indicator
| UrineSugar
| Insulin
| Antiallergics
| BloodSugar
| Immunosuppressants

/-- Represents a pair of indicators -/
structure IndicatorPair :=
  (first second : Indicator)

/-- Function to determine the correct indicators based on transplantation type -/
def correctIndicators (transplantType : TransplantationType) : IndicatorPair :=
  match transplantType with
  | TransplantationType.IsletCell => ⟨Indicator.BloodSugar, Indicator.Immunosuppressants⟩

/-- Theorem stating that for islet cell transplantation, the correct indicators are blood sugar and immunosuppressants -/
theorem islet_cell_transplant_indicators :
  correctIndicators TransplantationType.IsletCell = ⟨Indicator.BloodSugar, Indicator.Immunosuppressants⟩ :=
by sorry

end islet_cell_transplant_indicators_l3334_333498


namespace first_year_x_exceeds_y_l3334_333412

def commodity_x_price (year : ℕ) : ℚ :=
  420/100 + (year - 2001) * 30/100

def commodity_y_price (year : ℕ) : ℚ :=
  440/100 + (year - 2001) * 20/100

theorem first_year_x_exceeds_y :
  (∀ y : ℕ, 2001 < y ∧ y < 2004 → commodity_x_price y ≤ commodity_y_price y) ∧
  commodity_x_price 2004 > commodity_y_price 2004 :=
by sorry

end first_year_x_exceeds_y_l3334_333412


namespace power_product_simplification_l3334_333483

theorem power_product_simplification (a : ℝ) : (3 * a)^2 * a^5 = 9 * a^7 := by
  sorry

end power_product_simplification_l3334_333483


namespace complex_addition_l3334_333489

theorem complex_addition : (6 - 5*Complex.I) + (3 + 2*Complex.I) = 9 - 3*Complex.I := by
  sorry

end complex_addition_l3334_333489


namespace perimeter_quadrilateral_l3334_333441

/-- The perimeter of a quadrilateral PQRS with given coordinates can be expressed as x√3 + y√10, where x + y = 12 -/
theorem perimeter_quadrilateral (P Q R S : ℝ × ℝ) : 
  P = (1, 2) → Q = (3, 6) → R = (6, 3) → S = (8, 1) →
  ∃ (x y : ℤ), 
    (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) +
     Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) +
     Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) +
     Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) =
     x * Real.sqrt 3 + y * Real.sqrt 10) ∧
    x + y = 12 :=
by sorry

end perimeter_quadrilateral_l3334_333441


namespace mean_of_three_numbers_l3334_333449

theorem mean_of_three_numbers (x y z : ℝ) 
  (h1 : (x + y) / 2 = 5)
  (h2 : (y + z) / 2 = 9)
  (h3 : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 := by sorry

end mean_of_three_numbers_l3334_333449


namespace atomic_number_difference_l3334_333488

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ

/-- Represents a main group in the periodic table -/
structure MainGroup where
  elements : Set Element

/-- 
  Given two elements A and B in the same main group of the periodic table, 
  where the atomic number of A is x, the atomic number of B cannot be x+4.
-/
theorem atomic_number_difference (g : MainGroup) (A B : Element) (x : ℕ) :
  A ∈ g.elements → B ∈ g.elements → A.atomicNumber = x → 
  B.atomicNumber ≠ x + 4 := by
  sorry

end atomic_number_difference_l3334_333488


namespace parallel_vectors_sum_l3334_333480

/-- Given two vectors a and b in ℝ³, where a = (2, 4, 5) and b = (3, x, y),
    if a is parallel to b, then x + y = 27/2 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![2, 4, 5]
  let b : Fin 3 → ℝ := ![3, x, y]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  x + y = 27/2 := by
sorry

end parallel_vectors_sum_l3334_333480


namespace factorization_cubic_minus_linear_l3334_333446

theorem factorization_cubic_minus_linear (a x : ℝ) : 
  a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
sorry

end factorization_cubic_minus_linear_l3334_333446


namespace intersection_A_complement_B_l3334_333428

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x + 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Define the complement of B with respect to U
def C_U_B : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ C_U_B = {x : ℝ | x < 3} := by sorry

end intersection_A_complement_B_l3334_333428


namespace expression_evaluation_l3334_333494

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(3*x)) / (y^(2*y) * x^(3*x)) = x^(2*y - 3*x) * y^(3*x - 2*y) := by
  sorry

end expression_evaluation_l3334_333494


namespace min_students_theorem_l3334_333422

/-- The minimum number of students that can be divided into either 18 or 24 teams 
    with a maximum difference of 2 students between team sizes. -/
def min_students : ℕ := 70

/-- Checks if a number can be divided into a given number of teams
    with a maximum difference of 2 students between team sizes. -/
def can_divide (n : ℕ) (teams : ℕ) : Prop :=
  ∃ (base_size : ℕ), 
    (n ≥ base_size * teams) ∧ 
    (n ≤ (base_size + 2) * teams)

theorem min_students_theorem : 
  (can_divide min_students 18) ∧ 
  (can_divide min_students 24) ∧ 
  (∀ m : ℕ, m < min_students → ¬(can_divide m 18 ∧ can_divide m 24)) :=
sorry

end min_students_theorem_l3334_333422


namespace max_fraction_sum_l3334_333479

theorem max_fraction_sum (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  B ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  C ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  D ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (A : ℚ) / B + (C : ℚ) / D ≤ 13 :=
sorry

end max_fraction_sum_l3334_333479


namespace inscribed_circle_area_isosceles_trapezoid_l3334_333475

/-- The area of a circle inscribed in an isosceles trapezoid -/
theorem inscribed_circle_area_isosceles_trapezoid 
  (a : ℝ) 
  (h_positive : a > 0) 
  (h_isosceles : IsoscelesTrapezoid) 
  (h_angle : AngleAtSmallerBase = 120) : 
  AreaOfInscribedCircle = π * a^2 / 12 := by
  sorry

end inscribed_circle_area_isosceles_trapezoid_l3334_333475


namespace min_value_and_range_l3334_333419

theorem min_value_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ min) ∧
  (∀ (x : ℝ), (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ |2*x - 1| - |x + 1|) ↔ 
    -7 ≤ x ∧ x ≤ 11) := by
  sorry

end min_value_and_range_l3334_333419


namespace quadratic_factorization_sum_l3334_333420

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + d) * (x + e)) →
  (∀ x, x^2 + 11*x - 60 = (x + e) * (x - f)) →
  d + e + f = 23 := by
sorry

end quadratic_factorization_sum_l3334_333420


namespace seven_balls_four_boxes_l3334_333400

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 101 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 101 := by
  sorry

end seven_balls_four_boxes_l3334_333400


namespace sin_120_degrees_l3334_333423

theorem sin_120_degrees : 
  ∃ (Q : ℝ × ℝ) (E : ℝ × ℝ),
    (Q.1^2 + Q.2^2 = 1) ∧  -- Q is on the unit circle
    (Real.cos (2*π/3) = Q.1 ∧ Real.sin (2*π/3) = Q.2) ∧  -- Q is at 120°
    (E.2 = 0 ∧ (Q.1 - E.1) * (Q.1 - E.1) + Q.2 * Q.2 = (Q.1 - E.1)^2) →  -- E is the foot of perpendicular
    Real.sin (2*π/3) = Real.sqrt 3 / 2 :=
by sorry

end sin_120_degrees_l3334_333423


namespace tan_difference_identity_l3334_333405

theorem tan_difference_identity (n : ℝ) : 
  Real.tan ((n + 1) * π / 180) - Real.tan (n * π / 180) = 
  Real.sin (π / 180) / (Real.cos (n * π / 180) * Real.cos ((n + 1) * π / 180)) := by
sorry

end tan_difference_identity_l3334_333405


namespace trigonometric_identity_l3334_333476

theorem trigonometric_identity (α : ℝ) : 
  (Real.sin (7 * α) / Real.sin α) - 2 * (Real.cos (2 * α) + Real.cos (4 * α) + Real.cos (6 * α)) - 1 = 0 := by
  sorry

end trigonometric_identity_l3334_333476


namespace group_distribution_theorem_l3334_333450

def number_of_ways (n_men n_women : ℕ) (group_sizes : List ℕ) : ℕ :=
  sorry

theorem group_distribution_theorem :
  let n_men := 4
  let n_women := 5
  let group_sizes := [3, 3, 3]
  number_of_ways n_men n_women group_sizes = 1440 :=
by sorry

end group_distribution_theorem_l3334_333450


namespace next_perfect_square_l3334_333465

theorem next_perfect_square (n : ℤ) (x : ℤ) (h1 : Even n) (h2 : x = n^2) :
  (n + 1)^2 = x + 2*n + 1 := by
  sorry

end next_perfect_square_l3334_333465


namespace decimal_to_binary_89_l3334_333496

theorem decimal_to_binary_89 :
  ∃ (b : List Bool),
    b.reverse.map (λ x => if x then 1 else 0) = [1, 0, 1, 1, 0, 0, 1] ∧
    b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0 = 89 := by
  sorry

end decimal_to_binary_89_l3334_333496


namespace man_swimming_speed_l3334_333432

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h_downstream : downstream_distance = 36) 
  (h_upstream : upstream_distance = 48) 
  (h_time : time = 6) : 
  ∃ (v_man : ℝ) (v_stream : ℝ), 
    v_man + v_stream = downstream_distance / time ∧ 
    v_man - v_stream = upstream_distance / time ∧ 
    v_man = 7 := by
  sorry

#check man_swimming_speed

end man_swimming_speed_l3334_333432


namespace douglas_county_y_votes_l3334_333493

/-- Represents the percentage of votes Douglas won in county Y -/
def douglas_county_y_percentage : ℝ := 46

/-- Represents the total percentage of votes Douglas won in both counties -/
def total_percentage : ℝ := 58

/-- Represents the percentage of votes Douglas won in county X -/
def douglas_county_x_percentage : ℝ := 64

/-- Represents the ratio of voters in county X to county Y -/
def county_ratio : ℚ := 2 / 1

theorem douglas_county_y_votes :
  douglas_county_y_percentage = 
    (3 * total_percentage - 2 * douglas_county_x_percentage) := by sorry

end douglas_county_y_votes_l3334_333493


namespace symmetric_point_y_axis_l3334_333482

/-- Given a point A with coordinates (-3, 2), its symmetric point
    with respect to the y-axis has coordinates (3, 2). -/
theorem symmetric_point_y_axis :
  let A : ℝ × ℝ := (-3, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  symmetric_point A = (3, 2) := by
  sorry

end symmetric_point_y_axis_l3334_333482


namespace gym_spending_l3334_333497

theorem gym_spending (total_spent adidas_cost nike_cost skechers_cost clothes_cost : ℝ) : 
  total_spent = 8000 →
  nike_cost = 3 * adidas_cost →
  adidas_cost = (1 / 5) * skechers_cost →
  adidas_cost = 600 →
  total_spent = adidas_cost + nike_cost + skechers_cost + clothes_cost →
  clothes_cost = 2600 := by
sorry

end gym_spending_l3334_333497


namespace incorrect_inequality_transformation_l3334_333452

theorem incorrect_inequality_transformation :
  ¬(∀ (a b c : ℝ), a * c > b * c → a > b) := by
  sorry

end incorrect_inequality_transformation_l3334_333452


namespace independence_test_problems_l3334_333469

/-- A real-world problem that may or may not be solvable by independence tests. -/
inductive Problem
| DrugCureRate
| DrugRelation
| SmokingLungDisease
| SmokingGenderRelation
| InternetCrimeRate

/-- Determines if a problem involves examining the relationship between two categorical variables. -/
def involves_categorical_relationship (p : Problem) : Prop :=
  match p with
  | Problem.DrugRelation => True
  | Problem.SmokingGenderRelation => True
  | Problem.InternetCrimeRate => True
  | _ => False

/-- The definition of an independence test. -/
def is_independence_test (test : Problem → Prop) : Prop :=
  ∀ p, test p ↔ involves_categorical_relationship p

/-- The theorem stating which problems can be solved using independence tests. -/
theorem independence_test_problems (test : Problem → Prop) 
  (h : is_independence_test test) : 
  (test Problem.DrugRelation ∧ 
   test Problem.SmokingGenderRelation ∧ 
   test Problem.InternetCrimeRate) ∧
  (¬ test Problem.DrugCureRate ∧ 
   ¬ test Problem.SmokingLungDisease) :=
by sorry

end independence_test_problems_l3334_333469


namespace area_between_curves_l3334_333459

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^3 - x
def curve2 (a x : ℝ) : ℝ := x^2 - a

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 3 * x^2 - 1
def curve2_derivative (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem area_between_curves :
  ∃ (a : ℝ) (P : ℝ × ℝ),
    -- Conditions:
    -- 1. P lies on both curves
    curve1 P.1 = P.2 ∧
    curve2 a P.1 = P.2 ∧
    -- 2. The curves have a common tangent at P
    curve1_derivative P.1 = curve2_derivative P.1 →
    -- Conclusion:
    -- The area between the curves is 13/12
    (∫ x in (Real.sqrt 5 / 2 - 1 / 6)..(1 / 6 + Real.sqrt 5 / 2), |curve1 x - curve2 a x|) = 13 / 12 :=
by
  sorry

end area_between_curves_l3334_333459


namespace boat_purchase_payment_l3334_333461

theorem boat_purchase_payment (w x y z : ℝ) : 
  w + x + y + z = 60 ∧
  w = (1/2) * (x + y + z) ∧
  x = (1/3) * (w + y + z) ∧
  y = (1/4) * (w + x + z) →
  z = 13 := by sorry

end boat_purchase_payment_l3334_333461


namespace arithmetic_sequence_2023_l3334_333487

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_2023 (seq : ArithmeticSequence)
  (h1 : seq.a 2 + seq.a 7 = seq.a 8 + 1)
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ seq.a 4 = r * seq.a 2 ∧ seq.a 8 = r * seq.a 4) :
  seq.a 2023 = 2023 := by
  sorry

end arithmetic_sequence_2023_l3334_333487


namespace intersection_and_midpoint_trajectory_l3334_333457

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y-1)^2 = 1/4

theorem intersection_and_midpoint_trajectory :
  ∀ m : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2) ∧
  (∀ x y : ℝ, (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
    x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → trajectory_M x y) :=
by sorry

end intersection_and_midpoint_trajectory_l3334_333457


namespace ant_distance_theorem_l3334_333484

theorem ant_distance_theorem (n : ℕ) (points : Fin n → ℝ × ℝ) :
  n = 1390 →
  (∀ i, abs (points i).2 < 1) →
  (∀ i j, i ≠ j → dist (points i) (points j) > 2) →
  ∃ i j, dist (points i) (points j) ≥ 1000 :=
by sorry

#check ant_distance_theorem

end ant_distance_theorem_l3334_333484


namespace trailing_zeros_count_l3334_333491

def N : ℕ := 10^2018 + 1

theorem trailing_zeros_count (n : ℕ) : 
  ∃ k : ℕ, (N^2017 - 1) % 10^2018 = 0 ∧ (N^2017 - 1) % 10^2019 ≠ 0 := by
  sorry

end trailing_zeros_count_l3334_333491


namespace toris_growth_l3334_333445

theorem toris_growth (original_height current_height : Real) 
  (h1 : original_height = 4.4)
  (h2 : current_height = 7.26) :
  current_height - original_height = 2.86 := by
  sorry

end toris_growth_l3334_333445


namespace quadratic_real_root_condition_l3334_333435

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end quadratic_real_root_condition_l3334_333435


namespace emma_age_l3334_333458

def guesses : List Nat := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]

def is_prime (n : Nat) : Prop := Nat.Prime n

def off_by_one (guess : Nat) (age : Nat) : Prop :=
  guess = age - 1 ∨ guess = age + 1

def count_lower_guesses (age : Nat) : Nat :=
  guesses.filter (· < age) |>.length

theorem emma_age : ∃ (age : Nat),
  age ∈ guesses ∧
  is_prime age ∧
  (count_lower_guesses age : Rat) / guesses.length ≥ 6/10 ∧
  (∃ (g1 g2 : Nat), g1 ∈ guesses ∧ g2 ∈ guesses ∧ g1 ≠ g2 ∧ 
    off_by_one g1 age ∧ off_by_one g2 age) ∧
  age = 43 := by
  sorry

end emma_age_l3334_333458
