import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l43_4333

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) :
  (∃ k : ℕ, k ≥ 2 ∧
    Nat.choose n k * x^(n - k) * a^k = 210 ∧
    Nat.choose n (k + 1) * x^(n - k - 1) * a^(k + 1) = 504 ∧
    Nat.choose n (k + 2) * x^(n - k - 2) * a^(k + 2) = 1260) →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l43_4333


namespace NUMINAMATH_CALUDE_perfect_square_existence_l43_4372

theorem perfect_square_existence : ∃ n : ℕ, 
  (10^199 - 10^100 : ℕ) < n^2 ∧ n^2 < 10^199 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_existence_l43_4372


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l43_4378

/-- The hyperbola with equation x²/16 - y²/20 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focus_distance (P : ℝ × ℝ) :
  hyperbola P.1 P.2 → distance P F₁ = 9 → distance P F₂ = 17 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l43_4378


namespace NUMINAMATH_CALUDE_union_of_reduced_rectangles_l43_4326

-- Define a reduced rectangle as a set in ℝ²
def ReducedRectangle : Set (ℝ × ℝ) → Prop :=
  sorry

-- Define a family of reduced rectangles
def FamilyOfReducedRectangles : Set (Set (ℝ × ℝ)) → Prop :=
  sorry

-- The main theorem
theorem union_of_reduced_rectangles 
  (F : Set (Set (ℝ × ℝ))) 
  (h : FamilyOfReducedRectangles F) :
  ∃ (C : Set (Set (ℝ × ℝ))), 
    (C ⊆ F) ∧ 
    (Countable C) ∧ 
    (⋃₀ F = ⋃₀ C) :=
  sorry

end NUMINAMATH_CALUDE_union_of_reduced_rectangles_l43_4326


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l43_4356

-- Define the two curves
def curve1 (x y : ℝ) : Prop := y = 8 / (x^2 + 4)
def curve2 (x y : ℝ) : Prop := x + y = 2

-- Theorem stating that the x-coordinate of the intersection point is 0
theorem intersection_x_coordinate :
  ∃ y : ℝ, curve1 0 y ∧ curve2 0 y :=
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l43_4356


namespace NUMINAMATH_CALUDE_purchasing_power_increase_l43_4344

theorem purchasing_power_increase (original_price : ℝ) (money : ℝ) (h : money > 0) :
  let new_price := 0.8 * original_price
  let original_quantity := money / original_price
  let new_quantity := money / new_price
  new_quantity = 1.25 * original_quantity :=
by sorry

end NUMINAMATH_CALUDE_purchasing_power_increase_l43_4344


namespace NUMINAMATH_CALUDE_quadratic_properties_l43_4391

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_neg1 : a * (-1)^2 + b * (-1) + c = 0
  point_0 : c = -3
  point_1 : a * 1^2 + b * 1 + c = -4
  point_2 : a * 2^2 + b * 2 + c = -3
  point_3 : a * 3^2 + b * 3 + c = 0

/-- The theorem stating properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (∃ x y, f.a * x^2 + f.b * x + f.c = y ∧ ∀ t, f.a * t^2 + f.b * t + f.c ≥ y) ∧
  (f.a * x^2 + f.b * x + f.c = -4 ↔ x = 1) ∧
  (f.a * 5^2 + f.b * 5 + f.c = 12) ∧
  (∀ x > 1, ∀ y > x, f.a * y^2 + f.b * y + f.c > f.a * x^2 + f.b * x + f.c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l43_4391


namespace NUMINAMATH_CALUDE_semicircle_sum_limit_l43_4360

/-- Theorem: As the number of divisions approaches infinity, the sum of the lengths of semicircles
    constructed on equal parts of a circle's diameter approaches the semi-circumference of the original circle. -/
theorem semicircle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * (D / n) / 2) - π * D / 2| < ε :=
sorry

end NUMINAMATH_CALUDE_semicircle_sum_limit_l43_4360


namespace NUMINAMATH_CALUDE_max_regions_six_chords_l43_4318

/-- The number of regions created by drawing k chords in a circle -/
def num_regions (k : ℕ) : ℕ := 1 + k * (k + 1) / 2

/-- Theorem: The maximum number of regions created by drawing 6 chords in a circle is 22 -/
theorem max_regions_six_chords : num_regions 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_six_chords_l43_4318


namespace NUMINAMATH_CALUDE_first_player_advantage_l43_4367

/-- A game board configuration -/
structure BoardConfig where
  spaces : ℕ
  s₁ : ℕ
  s₂ : ℕ

/-- The probability of a player winning -/
def winProbability (player : ℕ) (config : BoardConfig) : ℝ :=
  sorry

/-- The theorem stating that the first player has a higher probability of winning -/
theorem first_player_advantage (config : BoardConfig) 
    (h : config.spaces ≥ 12) 
    (h_start : config.s₁ = config.s₂) : 
  winProbability 1 config > 1/2 :=
sorry

end NUMINAMATH_CALUDE_first_player_advantage_l43_4367


namespace NUMINAMATH_CALUDE_investment_of_c_l43_4305

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℚ
  profitShare : ℚ

/-- Represents a business partnership -/
def Partnership (a b c : Partner) : Prop :=
  -- Profit shares are proportional to investments
  a.profitShare / a.investment = b.profitShare / b.investment ∧
  b.profitShare / b.investment = c.profitShare / c.investment ∧
  -- Given conditions
  b.profitShare = 1800 ∧
  a.profitShare - c.profitShare = 720 ∧
  a.investment = 8000 ∧
  b.investment = 10000

theorem investment_of_c (a b c : Partner) 
  (h : Partnership a b c) : c.investment = 4000 := by
  sorry

end NUMINAMATH_CALUDE_investment_of_c_l43_4305


namespace NUMINAMATH_CALUDE_beth_crayons_l43_4317

/-- Given the number of crayon packs, crayons per pack, and extra crayons,
    calculate the total number of crayons Beth has. -/
def total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) : ℕ :=
  packs * crayons_per_pack + extra_crayons

/-- Prove that Beth has 46 crayons in total. -/
theorem beth_crayons : total_crayons 4 10 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l43_4317


namespace NUMINAMATH_CALUDE_min_distance_squared_l43_4313

theorem min_distance_squared (a b c d : ℝ) 
  (h : (b + 2 * a^2 - 6 * Real.log a)^2 + |2 * c - d + 6| = 0) :
  ∃ (m : ℝ), m = 20 ∧ ∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l43_4313


namespace NUMINAMATH_CALUDE_problem_solution_l43_4329

-- Define the expression as a function of a, b, and x
def expression (a b x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

theorem problem_solution :
  (∀ x, expression 7 (-1) x = 2 * x^2 - 4 * x + 2) ∧
  (∀ x, expression 5 (-3) x = -6 * x + 2) ∧
  (∃ a b, ∀ x, expression a b x = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l43_4329


namespace NUMINAMATH_CALUDE_pool_length_is_ten_l43_4315

/-- Proves that the length of a rectangular pool is 10 feet given its width, depth, and volume. -/
theorem pool_length_is_ten (width : ℝ) (depth : ℝ) (volume : ℝ) :
  width = 8 →
  depth = 6 →
  volume = 480 →
  volume = width * depth * (10 : ℝ) :=
by
  sorry

#check pool_length_is_ten

end NUMINAMATH_CALUDE_pool_length_is_ten_l43_4315


namespace NUMINAMATH_CALUDE_proposition_2_l43_4371

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem statement
theorem proposition_2 
  (m n : Line) (α β γ : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : perpendicular m β) 
  (h4 : parallel m α) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_proposition_2_l43_4371


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l43_4354

/-- Given a circle with equation x^2 + y^2 + 4x = 0, its center is at (-2, 0) and its radius is 2 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 + 4*x = 0 → ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 0) ∧ radius = 2 ∧
    (x + 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l43_4354


namespace NUMINAMATH_CALUDE_candy_ratio_is_three_l43_4374

/-- The ratio of Jennifer's candies to Bob's candies -/
def candy_ratio (emily_candies bob_candies : ℕ) : ℚ :=
  (2 * emily_candies) / bob_candies

/-- Theorem: The ratio of Jennifer's candies to Bob's candies is 3 -/
theorem candy_ratio_is_three :
  candy_ratio 6 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_is_three_l43_4374


namespace NUMINAMATH_CALUDE_complement_of_union_l43_4341

open Set

universe u

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def A : Finset ℕ := {1,2,3}
def B : Finset ℕ := {3,4,5,6}

theorem complement_of_union :
  (U \ (A ∪ B) : Finset ℕ) = {7,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l43_4341


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l43_4310

theorem quadratic_root_in_interval (a b : ℝ) (hb : b > 0) 
  (h_distinct : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 + a*r₁ + b = 0 ∧ r₂^2 + a*r₂ + b = 0)
  (h_one_in_interval : ∃! r : ℝ, r^2 + a*r + b = 0 ∧ r ∈ Set.Icc (-1) 1) :
  ∃ r : ℝ, r^2 + a*r + b = 0 ∧ r ∈ Set.Ioo (-b) b :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l43_4310


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l43_4346

/-- A sector that is one-third of a circle --/
structure ThirdCircleSector where
  /-- The radius of the full circle --/
  R : ℝ
  /-- Assumption that R is positive --/
  R_pos : 0 < R

/-- An inscribed circle in the sector --/
structure InscribedCircle (S : ThirdCircleSector) where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- Assumption that r is positive --/
  r_pos : 0 < r

/-- The theorem stating the radius of the inscribed circle --/
theorem inscribed_circle_radius (S : ThirdCircleSector) (C : InscribedCircle S) 
    (h : S.R = 6) : C.r = 6 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l43_4346


namespace NUMINAMATH_CALUDE_rectangle_with_three_tangent_circles_l43_4376

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Checks if two circles are tangent to each other -/
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the sides of a rectangle -/
def is_circle_tangent_to_rectangle (c : Circle) (r : Rectangle) : Prop :=
  c.radius ≤ r.width / 2 ∧ c.radius ≤ r.length / 2

/-- Main theorem: If a rectangle contains three tangent circles (two smaller equal ones and one larger),
    and the width of the rectangle is 4, then its length is 3 + √8 -/
theorem rectangle_with_three_tangent_circles 
  (r : Rectangle) 
  (c1 c2 c3 : Circle) : 
  r.width = 4 →
  c1.radius = c2.radius →
  c1.radius < c3.radius →
  are_circles_tangent c1 c2 →
  are_circles_tangent c1 c3 →
  are_circles_tangent c2 c3 →
  is_circle_tangent_to_rectangle c1 r →
  is_circle_tangent_to_rectangle c2 r →
  is_circle_tangent_to_rectangle c3 r →
  r.length = 3 + Real.sqrt 8 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_with_three_tangent_circles_l43_4376


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l43_4377

theorem consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) :
  ∃ a : ℤ, n * (n - 1)^(k - 1) = n * (2 * a + (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l43_4377


namespace NUMINAMATH_CALUDE_chemical_mixture_composition_l43_4397

/-- Given the composition of two chemical solutions and their mixture, 
    prove the percentage of chemical b in solution y -/
theorem chemical_mixture_composition 
  (x_a : Real) (x_b : Real) (y_a : Real) (y_b : Real) 
  (mix_a : Real) (mix_x : Real) : 
  x_a = 0.1 → 
  x_b = 0.9 → 
  y_a = 0.2 → 
  mix_a = 0.12 → 
  mix_x = 0.8 → 
  y_b = 0.8 := by
  sorry

#check chemical_mixture_composition

end NUMINAMATH_CALUDE_chemical_mixture_composition_l43_4397


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l43_4381

/-- The volume of a triangular prism with given dimensions -/
theorem triangular_prism_volume 
  (thickness : ℝ) 
  (side1 side2 side3 : ℝ) 
  (h_thickness : thickness = 2)
  (h_side1 : side1 = 7)
  (h_side2 : side2 = 24)
  (h_side3 : side3 = 25)
  (h_right_triangle : side1^2 + side2^2 = side3^2) :
  thickness * (1/2 * side1 * side2) = 168 := by
sorry

end NUMINAMATH_CALUDE_triangular_prism_volume_l43_4381


namespace NUMINAMATH_CALUDE_equation_is_linear_l43_4332

def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (A B C : ℝ), ∀ x y, f x y = A * x + B * y + C

def equation (x y : ℝ) : ℝ := 2 * x + 3 * y - 4

theorem equation_is_linear :
  is_linear_equation_in_two_variables equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_l43_4332


namespace NUMINAMATH_CALUDE_rectangle_x_value_l43_4340

/-- Given a rectangle with vertices (x, 1), (1, 1), (1, -2), and (x, -2) and area 12, prove that x = -3 -/
theorem rectangle_x_value (x : ℝ) : 
  let vertices := [(x, 1), (1, 1), (1, -2), (x, -2)]
  let width := 1 - (-2)
  let area := 12
  let length := area / width
  x = 1 - length := by
  sorry

#check rectangle_x_value

end NUMINAMATH_CALUDE_rectangle_x_value_l43_4340


namespace NUMINAMATH_CALUDE_min_value_product_l43_4323

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 9) :
  x^3 * y^3 * z^2 ≥ 1/27 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l43_4323


namespace NUMINAMATH_CALUDE_perpendicular_probability_l43_4388

/-- The set of positive integers less than 6 -/
def A : Set ℕ := {n | n < 6 ∧ n > 0}

/-- The line l: x + 2y + 1 = 0 -/
def l (x y : ℝ) : Prop := x + 2*y + 1 = 0

/-- The condition for the line from (a,b) to (0,0) being perpendicular to l -/
def perpendicular (a b : ℕ) : Prop := (b : ℝ) / (a : ℝ) = 2

/-- The number of ways to select 3 different elements from A -/
def total_outcomes : ℕ := Nat.choose 5 3

/-- The number of favorable outcomes -/
def favorable_outcomes : ℕ := 6

/-- The main theorem -/
theorem perpendicular_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_perpendicular_probability_l43_4388


namespace NUMINAMATH_CALUDE_triangle_medians_inequality_l43_4389

/-- Given a triangle with sides a, b, c, medians ta, tb, tc, and circumcircle diameter D,
    the sum of the ratios of the squared sides to their opposite medians
    is less than or equal to 6 times the diameter of the circumcircle. -/
theorem triangle_medians_inequality (a b c ta tb tc D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ta : 0 < ta) (h_pos_tb : 0 < tb) (h_pos_tc : 0 < tc)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_medians : ta^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧ 
               tb^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧ 
               tc^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_circumcircle : D = (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (h_s : s = (a + b + c) / 2) :
  (a^2 + b^2) / tc + (b^2 + c^2) / ta + (c^2 + a^2) / tb ≤ 6 * D :=
sorry

end NUMINAMATH_CALUDE_triangle_medians_inequality_l43_4389


namespace NUMINAMATH_CALUDE_ratio_simplification_and_increase_l43_4325

def original_ratio : List Nat := [4, 16, 20, 12]

def gcd_list (l : List Nat) : Nat :=
  l.foldl Nat.gcd 0

def simplify_ratio (l : List Nat) : List Nat :=
  let gcd := gcd_list l
  l.map (·/gcd)

def percentage_increase (first last : Nat) : Nat :=
  ((last - first) * 100) / first

theorem ratio_simplification_and_increase :
  let simplified := simplify_ratio original_ratio
  simplified = [1, 4, 5, 3] ∧
  percentage_increase simplified.head! simplified.getLast! = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_simplification_and_increase_l43_4325


namespace NUMINAMATH_CALUDE_logarithm_inequality_l43_4304

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log c^2 / Real.log (a + b) + Real.log a^2 / Real.log (b + c) + Real.log b^2 / Real.log (c + a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l43_4304


namespace NUMINAMATH_CALUDE_max_sum_distance_to_line_l43_4383

theorem max_sum_distance_to_line (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁^2 + y₁^2 = 1)
  (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₁*x₂ + y₁*y₂ = 1/2) :
  (|x₁ + y₁ - 1| / Real.sqrt 2) + (|x₂ + y₂ - 1| / Real.sqrt 2) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_sum_distance_to_line_l43_4383


namespace NUMINAMATH_CALUDE_value_of_expression_l43_4384

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l43_4384


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l43_4336

/-- The set T of points (x,y) in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2
    (x + 3 = 5 ∧ y - 2 ≤ 5) ∨
    (y - 2 = 5 ∧ x + 3 ≤ 5) ∨
    (x + 3 = y - 2 ∧ 5 ≤ x + 3)}

/-- The common point of the three rays -/
def common_point : ℝ × ℝ := (2, 7)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 7}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 2 ∧ p.2 = 7}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 2 ∧ p.2 = p.1 + 5}

/-- Theorem stating that T consists of three rays with a common point -/
theorem T_is_three_rays_with_common_point :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_point ∈ ray1 ∧ common_point ∈ ray2 ∧ common_point ∈ ray3 :=
sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l43_4336


namespace NUMINAMATH_CALUDE_profession_assignment_l43_4375

/-- Represents the three people mentioned in the problem -/
inductive Person
  | Kondratyev
  | Davydov
  | Fedorov

/-- Represents the three professions mentioned in the problem -/
inductive Profession
  | Carpenter
  | Painter
  | Plumber

/-- Represents the age relation between two people -/
def OlderThan (a b : Person) : Prop := sorry

/-- Represents that one person has never heard of another -/
def NeverHeardOf (a b : Person) : Prop := sorry

/-- Represents the assignment of professions to people -/
def ProfessionAssignment := Person → Profession

/-- The carpenter was repairing the plumber's house -/
def CarpenterRepairingPlumbersHouse (assignment : ProfessionAssignment) : Prop := sorry

/-- The painter needed help from the carpenter -/
def PainterNeededHelpFromCarpenter (assignment : ProfessionAssignment) : Prop := sorry

/-- Main theorem: Given the conditions, prove the correct profession assignment -/
theorem profession_assignment :
  ∀ (assignment : ProfessionAssignment),
    (∀ p : Profession, ∃! person : Person, assignment person = p) →
    OlderThan Person.Davydov Person.Kondratyev →
    NeverHeardOf Person.Fedorov Person.Davydov →
    CarpenterRepairingPlumbersHouse assignment →
    PainterNeededHelpFromCarpenter assignment →
    (∀ p1 p2 : Person, assignment p1 = Profession.Plumber ∧ assignment p2 = Profession.Painter → OlderThan p1 p2) →
    assignment Person.Kondratyev = Profession.Carpenter ∧
    assignment Person.Davydov = Profession.Painter ∧
    assignment Person.Fedorov = Profession.Plumber := by
  sorry


end NUMINAMATH_CALUDE_profession_assignment_l43_4375


namespace NUMINAMATH_CALUDE_some_mystical_creatures_are_enchanted_beings_l43_4342

-- Define the types
variable (U : Type) -- Universe of discourse
variable (Dragon : U → Prop)
variable (MysticalCreature : U → Prop)
variable (EnchantedBeing : U → Prop)

-- Define the premises
variable (h1 : ∀ x, Dragon x → MysticalCreature x)
variable (h2 : ∃ x, EnchantedBeing x ∧ Dragon x)

-- Theorem to prove
theorem some_mystical_creatures_are_enchanted_beings :
  ∃ x, MysticalCreature x ∧ EnchantedBeing x :=
sorry

end NUMINAMATH_CALUDE_some_mystical_creatures_are_enchanted_beings_l43_4342


namespace NUMINAMATH_CALUDE_tan_alpha_and_fraction_l43_4339

theorem tan_alpha_and_fraction (α : Real) 
  (h : Real.tan (α + π / 4) = 2) : 
  Real.tan α = 1 / 3 ∧ 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_fraction_l43_4339


namespace NUMINAMATH_CALUDE_perfect_square_property_l43_4322

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem perfect_square_property : 
  (is_perfect_square (factorial 101 * 102 * 102)) ∧ 
  (¬ is_perfect_square (factorial 102 * 103 * 103)) ∧
  (¬ is_perfect_square (factorial 103 * 104 * 104)) ∧
  (¬ is_perfect_square (factorial 104 * 105 * 105)) ∧
  (¬ is_perfect_square (factorial 105 * 106 * 106)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_property_l43_4322


namespace NUMINAMATH_CALUDE_negative_45_same_terminal_side_as_315_l43_4335

def has_same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem negative_45_same_terminal_side_as_315 :
  has_same_terminal_side (-45 : ℝ) 315 :=
sorry

end NUMINAMATH_CALUDE_negative_45_same_terminal_side_as_315_l43_4335


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l43_4300

theorem p_necessary_not_sufficient_for_q :
  ∀ x : ℝ,
  (∃ y : ℝ, y < 1 ∧ ¬((y + 2) * (y - 1) < 0)) ∧
  (∀ z : ℝ, (z + 2) * (z - 1) < 0 → z < 1) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l43_4300


namespace NUMINAMATH_CALUDE_fox_speed_l43_4363

/-- Given a constant speed where 100 kilometers are covered in 120 minutes, 
    prove that the speed in kilometers per hour is 50. -/
theorem fox_speed (distance : ℝ) (time_minutes : ℝ) (speed_km_per_hour : ℝ)
  (h1 : distance = 100)
  (h2 : time_minutes = 120)
  (h3 : speed_km_per_hour = distance / time_minutes * 60) :
  speed_km_per_hour = 50 := by
  sorry

end NUMINAMATH_CALUDE_fox_speed_l43_4363


namespace NUMINAMATH_CALUDE_ribbon_segment_length_l43_4364

theorem ribbon_segment_length :
  let total_length : ℚ := 4/5
  let num_segments : ℕ := 3
  let segment_fraction : ℚ := 1/3
  let segment_length : ℚ := total_length * segment_fraction
  segment_length = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_segment_length_l43_4364


namespace NUMINAMATH_CALUDE_fraction_equality_l43_4351

theorem fraction_equality : (2018 + 2018 + 2018) / (2018 + 2018 + 2018 + 2018) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l43_4351


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l43_4382

/-- Calculates the average speed for a round trip journey between two points -/
theorem average_speed_round_trip (d : ℝ) (uphill_speed downhill_speed : ℝ) 
  (h1 : uphill_speed > 0)
  (h2 : downhill_speed > 0)
  (h3 : uphill_speed = 60)
  (h4 : downhill_speed = 36) :
  (2 * d) / (d / uphill_speed + d / downhill_speed) = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l43_4382


namespace NUMINAMATH_CALUDE_beavers_help_l43_4320

theorem beavers_help (initial_beavers : Real) (current_beavers : Nat) 
  (h1 : initial_beavers = 2.0) 
  (h2 : current_beavers = 3) : 
  (current_beavers : Real) - initial_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_help_l43_4320


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l43_4368

def arrange_comics (spiderman : Nat) (archie : Nat) (garfield : Nat) : Nat :=
  Nat.factorial spiderman * (Nat.factorial archie * Nat.factorial garfield * Nat.factorial 2)

theorem comic_arrangement_count :
  arrange_comics 7 6 5 = 871219200 := by
  sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l43_4368


namespace NUMINAMATH_CALUDE_min_value_theorem_l43_4324

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y + 1) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/(b + 1) = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l43_4324


namespace NUMINAMATH_CALUDE_diana_statue_painting_l43_4380

/-- The number of statues that can be painted given a certain amount of paint and paint required per statue -/
def statues_paintable (paint_available : ℚ) (paint_per_statue : ℚ) : ℚ :=
  paint_available / paint_per_statue

/-- Theorem: Given 1/2 gallon of paint and 1/4 gallon required per statue, 2 statues can be painted -/
theorem diana_statue_painting :
  statues_paintable (1/2) (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_diana_statue_painting_l43_4380


namespace NUMINAMATH_CALUDE_intersection_count_l43_4337

/-- The number of intersections between the line 3x + 4y = 12 and the circle x^2 + y^2 = 16 -/
def num_intersections : ℕ := 2

/-- The line equation 3x + 4y = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 16 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Theorem stating that the number of intersections between the given line and circle is 2 -/
theorem intersection_count :
  ∃ (p q : ℝ × ℝ),
    line_equation p.1 p.2 ∧ circle_equation p.1 p.2 ∧
    line_equation q.1 q.2 ∧ circle_equation q.1 q.2 ∧
    p ≠ q ∧
    (∀ (r : ℝ × ℝ), line_equation r.1 r.2 ∧ circle_equation r.1 r.2 → r = p ∨ r = q) :=
by sorry

end NUMINAMATH_CALUDE_intersection_count_l43_4337


namespace NUMINAMATH_CALUDE_subtract_negative_numbers_l43_4311

theorem subtract_negative_numbers : -5 - 9 = -14 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_numbers_l43_4311


namespace NUMINAMATH_CALUDE_ellipse_k_range_l43_4314

-- Define the equation of the ellipse
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 4) + y^2 / (10 - k) = 1

-- Define the property of having foci on the x-axis
def foci_on_x_axis (k : ℝ) : Prop :=
  k - 4 > 0 ∧ 10 - k > 0 ∧ k - 4 > 10 - k

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, ellipse_equation x y k) ∧ foci_on_x_axis k ↔ 7 < k ∧ k < 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l43_4314


namespace NUMINAMATH_CALUDE_dan_has_five_limes_l43_4306

/-- The number of limes Dan has after giving some to Sara -/
def dans_remaining_limes (initial_limes : ℕ) (limes_given : ℕ) : ℕ :=
  initial_limes - limes_given

/-- Theorem stating that Dan has 5 limes after giving 4 to Sara -/
theorem dan_has_five_limes :
  dans_remaining_limes 9 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_has_five_limes_l43_4306


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l43_4392

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral :=
  (a : ℝ) -- Length of side a
  (b : ℝ) -- Length of side b
  (c : ℝ) -- Length of diagonal c
  (d : ℝ) -- Length of diagonal d
  (ha : a > 0) -- Side lengths are positive
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)

/-- In any cyclic quadrilateral, the sum of the squares of the sides 
    is equal to the sum of the squares of the diagonals. -/
theorem cyclic_quadrilateral_theorem (q : CyclicQuadrilateral) :
  q.c^2 + q.d^2 = 2 * (q.a^2 + q.b^2) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l43_4392


namespace NUMINAMATH_CALUDE_round_0_689_to_two_places_l43_4330

/-- Rounds a real number to the specified number of decimal places. -/
def round_to_decimal_places (x : ℝ) (places : ℕ) : ℝ := 
  sorry

/-- The given number to be rounded -/
def given_number : ℝ := 0.689

/-- Theorem stating that rounding 0.689 to two decimal places results in 0.69 -/
theorem round_0_689_to_two_places :
  round_to_decimal_places given_number 2 = 0.69 := by
  sorry

end NUMINAMATH_CALUDE_round_0_689_to_two_places_l43_4330


namespace NUMINAMATH_CALUDE_real_part_of_complex_expression_l43_4395

theorem real_part_of_complex_expression :
  Complex.re ((1 - 2 * Complex.I)^2 + Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_expression_l43_4395


namespace NUMINAMATH_CALUDE_f_properties_l43_4357

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties :
  (∀ x b, f x b 0 = -f (-x) b 0) ∧
  (∀ c, c > 0 → ∃! x, f x 0 c = 0) ∧
  (∀ x b c, f (x - 0) b c - c = -(f (-x - 0) b c - c)) ∧
  (∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l43_4357


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_90_l43_4352

theorem greatest_common_multiple_10_15_under_90 : 
  ∃ (n : ℕ), n = 60 ∧ 
  (∀ m : ℕ, m < 90 ∧ 10 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  10 ∣ n ∧ 15 ∣ n ∧ n < 90 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_90_l43_4352


namespace NUMINAMATH_CALUDE_ratio_equality_solution_l43_4308

theorem ratio_equality_solution (x : ℝ) : 
  (4 + 2*x) / (6 + 3*x) = (2 + x) / (3 + 2*x) → x = 0 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_solution_l43_4308


namespace NUMINAMATH_CALUDE_points_for_level_completion_l43_4307

/-- Given a game scenario, prove the points earned for completing the level -/
theorem points_for_level_completion 
  (enemies_defeated : ℕ) 
  (points_per_enemy : ℕ) 
  (total_points : ℕ) 
  (h1 : enemies_defeated = 6)
  (h2 : points_per_enemy = 9)
  (h3 : total_points = 62) :
  total_points - (enemies_defeated * points_per_enemy) = 8 :=
by sorry

end NUMINAMATH_CALUDE_points_for_level_completion_l43_4307


namespace NUMINAMATH_CALUDE_princess_pear_cherries_l43_4343

def jester_height (i : ℕ) : ℕ := i

def is_valid_group (group : Finset ℕ) : Prop :=
  group.card = 6 ∧ ∃ (n : ℕ), n ≤ 100 ∧
  (∃ (lower upper : Finset ℕ),
    lower.card = 3 ∧ upper.card = 3 ∧
    lower ∪ upper = group ∧
    ∀ i ∈ lower, ∀ j ∈ upper, jester_height i < jester_height j)

def number_of_cherries : ℕ := (Nat.choose 50 3) ^ 2 * 2

theorem princess_pear_cherries :
  number_of_cherries = 384160000 := by sorry

end NUMINAMATH_CALUDE_princess_pear_cherries_l43_4343


namespace NUMINAMATH_CALUDE_joker_prob_is_one_twentyseventh_l43_4394

/-- A standard deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (jokers : ℕ)
  (h_total : total_cards = 54)
  (h_jokers : jokers = 2)

/-- The probability of drawing a joker from the top of the deck -/
def joker_probability (d : Deck) : ℚ :=
  d.jokers / d.total_cards

/-- Theorem: The probability of drawing a joker from a standard 54-card deck with 2 jokers is 1/27 -/
theorem joker_prob_is_one_twentyseventh (d : Deck) : joker_probability d = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_joker_prob_is_one_twentyseventh_l43_4394


namespace NUMINAMATH_CALUDE_order_of_sqrt_differences_l43_4358

theorem order_of_sqrt_differences :
  let a := Real.sqrt 3 - Real.sqrt 2
  let b := Real.sqrt 6 - Real.sqrt 5
  let c := Real.sqrt 7 - Real.sqrt 6
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_sqrt_differences_l43_4358


namespace NUMINAMATH_CALUDE_initial_ratio_proof_l43_4348

/-- Proves that given a 30-liter mixture of liquids p and q, if adding 12 liters of liquid q
    results in a 3:4 ratio of p to q, then the initial ratio of p to q was 3:2. -/
theorem initial_ratio_proof (p q : ℝ) 
  (h1 : p + q = 30)  -- Initial mixture is 30 liters
  (h2 : p / (q + 12) = 3 / 4)  -- After adding 12 liters of q, the ratio becomes 3:4
  : p / q = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_proof_l43_4348


namespace NUMINAMATH_CALUDE_paige_goldfish_l43_4345

/-- The number of goldfish Paige initially raised -/
def initial_goldfish : ℕ := sorry

/-- The number of catfish Paige initially raised -/
def initial_catfish : ℕ := 12

/-- The number of fish that disappeared -/
def disappeared_fish : ℕ := 4

/-- The number of fish left -/
def remaining_fish : ℕ := 15

theorem paige_goldfish :
  initial_goldfish = 7 :=
by sorry

end NUMINAMATH_CALUDE_paige_goldfish_l43_4345


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l43_4309

theorem quadratic_roots_sum_and_product :
  let a : ℝ := 2
  let b : ℝ := -10
  let c : ℝ := 12
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots = 5 ∧ product_of_roots = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l43_4309


namespace NUMINAMATH_CALUDE_rectangle_area_with_circles_l43_4355

/-- The area of a rectangle containing 8 circles arranged in a 2x4 grid, 
    where each circle has a radius of 3 inches. -/
theorem rectangle_area_with_circles (radius : ℝ) (width_circles : ℕ) (length_circles : ℕ) :
  radius = 3 →
  width_circles = 2 →
  length_circles = 4 →
  (2 * radius * width_circles) * (2 * radius * length_circles) = 288 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_circles_l43_4355


namespace NUMINAMATH_CALUDE_period_and_trigonometric_function_l43_4393

theorem period_and_trigonometric_function (ω : ℝ) (α β : ℝ) : 
  ω > 0 →
  (∀ x, 2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x) = 
    Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)) →
  (∀ x, 2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x) = 
    2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x)) →
  Real.sqrt 2 * Real.sin (α - Real.pi / 4 + Real.pi / 4) = Real.sqrt 2 / 3 →
  Real.sqrt 2 * Real.sin (β - Real.pi / 4 + Real.pi / 4) = 2 * Real.sqrt 2 / 3 →
  α > -Real.pi / 2 →
  α < Real.pi / 2 →
  β > -Real.pi / 2 →
  β < Real.pi / 2 →
  Real.cos (α + β) = (2 * Real.sqrt 10 - 2) / 9 := by
sorry


end NUMINAMATH_CALUDE_period_and_trigonometric_function_l43_4393


namespace NUMINAMATH_CALUDE_gwen_spent_l43_4365

theorem gwen_spent (received : ℕ) (left : ℕ) (spent : ℕ) : 
  received = 7 → left = 5 → spent = received - left → spent = 2 := by
  sorry

end NUMINAMATH_CALUDE_gwen_spent_l43_4365


namespace NUMINAMATH_CALUDE_exists_monochromatic_trapezoid_l43_4359

/-- A color is represented as a natural number -/
def Color := ℕ

/-- A point on a circle -/
structure CirclePoint where
  angle : ℝ
  color : Color

/-- A circle with colored points -/
structure ColoredCircle where
  points : Set CirclePoint
  num_colors : ℕ
  color_bound : num_colors ≥ 2

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  p1 : CirclePoint
  p2 : CirclePoint
  p3 : CirclePoint
  p4 : CirclePoint
  trapezoid_condition : (p2.angle - p1.angle) = (p4.angle - p3.angle)

/-- The main theorem -/
theorem exists_monochromatic_trapezoid (c : ColoredCircle) :
  ∃ t : InscribedTrapezoid, 
    t.p1 ∈ c.points ∧ 
    t.p2 ∈ c.points ∧ 
    t.p3 ∈ c.points ∧ 
    t.p4 ∈ c.points ∧
    t.p1.color = t.p2.color ∧ 
    t.p2.color = t.p3.color ∧ 
    t.p3.color = t.p4.color :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_trapezoid_l43_4359


namespace NUMINAMATH_CALUDE_bus_ride_difference_l43_4398

theorem bus_ride_difference (oscar_ride : ℝ) (charlie_ride : ℝ) 
  (h1 : oscar_ride = 0.75) (h2 : charlie_ride = 0.25) :
  oscar_ride - charlie_ride = 0.50 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l43_4398


namespace NUMINAMATH_CALUDE_not_always_parallel_if_perpendicular_to_same_plane_l43_4366

-- Define a type for planes
axiom Plane : Type

-- Define a relation for perpendicularity between planes
axiom perpendicular : Plane → Plane → Prop

-- Define a relation for parallelism between planes
axiom parallel : Plane → Plane → Prop

-- State the theorem
theorem not_always_parallel_if_perpendicular_to_same_plane :
  ¬ (∀ (P Q R : Plane), perpendicular P R → perpendicular Q R → parallel P Q) :=
sorry

end NUMINAMATH_CALUDE_not_always_parallel_if_perpendicular_to_same_plane_l43_4366


namespace NUMINAMATH_CALUDE_range_of_b_l43_4353

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * Real.log (x + 1) - x - b
def g (x : ℝ) : ℝ := Real.exp x

-- Define the point P
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- State the theorem
theorem range_of_b (a : ℝ) (x₀ : ℝ) (h1 : 0 < x₀ ∧ x₀ < Real.exp 1 - 1) :
  ∃ b : ℝ, 0 < b ∧ b < 1 - 1 / Real.exp 1 ∧
  ∃ y₀ : ℝ, 
    -- P is on the curve f
    y₀ = f a b x₀ ∧
    -- OP is the tangent line of f
    (deriv (f a b) x₀ = y₀ / x₀) ∧
    -- OP is perpendicular to a tangent line of g passing through the origin
    ∃ m : ℝ, deriv g m * (y₀ / x₀) = -1 ∧ g m = m * (deriv g m) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l43_4353


namespace NUMINAMATH_CALUDE_julia_car_rental_cost_l43_4312

/-- Calculates the total cost of a car rental given the daily rate, per-mile charge, days rented, and miles driven. -/
def carRentalCost (dailyRate : ℝ) (perMileCharge : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  dailyRate * daysRented + perMileCharge * milesDriven

/-- Proves that Julia's car rental cost is $46.12 given the specific conditions. -/
theorem julia_car_rental_cost :
  let dailyRate : ℝ := 29
  let perMileCharge : ℝ := 0.08
  let daysRented : ℕ := 1
  let milesDriven : ℝ := 214.0
  carRentalCost dailyRate perMileCharge daysRented milesDriven = 46.12 := by
  sorry

end NUMINAMATH_CALUDE_julia_car_rental_cost_l43_4312


namespace NUMINAMATH_CALUDE_ratio_problem_l43_4390

theorem ratio_problem (x y : ℚ) (h : (8*x - 5*y) / (10*x - 3*y) = 4/7) : x/y = 23/16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l43_4390


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l43_4387

/-- The original polynomial before multiplication by 3 -/
def original_poly (x : ℝ) : ℝ := x^4 + 2*x^3 + 5*x^2 + x + 2

/-- The expanded polynomial after multiplication by 3 -/
def expanded_poly (x : ℝ) : ℝ := 3 * (original_poly x)

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [3, 6, 15, 3, 6]

/-- Theorem: The sum of the squares of the coefficients of the expanded form of 3(x^4 + 2x^3 + 5x^2 + x + 2) is 315 -/
theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l43_4387


namespace NUMINAMATH_CALUDE_minimum_crossing_time_l43_4321

/-- Represents an individual with their crossing time -/
structure Individual where
  name : String
  time : Nat

/-- Represents a crossing of the bridge -/
inductive Crossing
  | Single : Individual → Crossing
  | Pair : Individual → Individual → Crossing

/-- Calculates the time taken for a single crossing -/
def crossingTime (c : Crossing) : Nat :=
  match c with
  | Crossing.Single i => i.time
  | Crossing.Pair i j => max i.time j.time

/-- The problem statement -/
theorem minimum_crossing_time
  (a b c d : Individual)
  (ha : a.time = 2)
  (hb : b.time = 3)
  (hc : c.time = 8)
  (hd : d.time = 10)
  (crossings : List Crossing)
  (hcross : crossings = [Crossing.Pair a b, Crossing.Single a, Crossing.Pair c d, Crossing.Single b, Crossing.Pair a b]) :
  (crossings.map crossingTime).sum = 21 ∧
  ∀ (otherCrossings : List Crossing),
    (otherCrossings.map crossingTime).sum ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_minimum_crossing_time_l43_4321


namespace NUMINAMATH_CALUDE_project_nap_duration_l43_4347

theorem project_nap_duration 
  (project_days : ℕ) 
  (hours_per_day : ℕ) 
  (work_hours : ℕ) 
  (num_naps : ℕ) 
  (h1 : project_days = 4) 
  (h2 : hours_per_day = 24) 
  (h3 : work_hours = 54) 
  (h4 : num_naps = 6) : 
  (project_days * hours_per_day - work_hours) / num_naps = 7 := by
  sorry

end NUMINAMATH_CALUDE_project_nap_duration_l43_4347


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l43_4349

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a ≥ 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l43_4349


namespace NUMINAMATH_CALUDE_f_of_3_equals_9_l43_4334

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_9_l43_4334


namespace NUMINAMATH_CALUDE_second_polygon_sides_l43_4338

theorem second_polygon_sides (perimeter : ℝ) (sides_first : ℕ) (length_ratio : ℝ) (sides_second : ℕ) : 
  perimeter > 0 →
  sides_first = 50 →
  length_ratio = 3 →
  (perimeter = sides_first * (length_ratio * (perimeter / (sides_second * length_ratio)))) →
  (perimeter = sides_second * (perimeter / (sides_second * length_ratio))) →
  sides_second = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l43_4338


namespace NUMINAMATH_CALUDE_complement_of_60_degrees_l43_4316

def angle : ℝ := 60

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem complement_of_60_degrees :
  complement angle = 30 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_60_degrees_l43_4316


namespace NUMINAMATH_CALUDE_intersection_point_properties_l43_4328

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := ((-1 : ℝ), (2 : ℝ))

-- Define the given line for perpendicularity
def perp_line (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Theorem statement
theorem intersection_point_properties :
  l₁ M.1 M.2 ∧ l₂ M.1 M.2 →
  (∀ x y : ℝ, y = -2 * x ↔ ∃ t : ℝ, x = t * M.1 ∧ y = t * M.2) ∧
  (∀ x y : ℝ, x - 2 * y + 5 = 0 ↔ (y - M.2 = (1/2) * (x - M.1) ∧ 
    ∃ a b : ℝ, perp_line a b ∧ (b - M.2) = (-2) * (a - M.1))) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_properties_l43_4328


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l43_4370

/-- Given the cost of 3 pens and 5 pencils is Rs. 150, and the ratio of the cost of one pen
    to one pencil is 5:1, prove that the cost of one dozen pens is Rs. 450. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℝ) 
  (h1 : 3 * pen_cost + 5 * pencil_cost = 150)
  (h2 : pen_cost = 5 * pencil_cost) : 
  12 * pen_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l43_4370


namespace NUMINAMATH_CALUDE_optimal_candy_purchase_l43_4301

/-- Represents a set of candies with its cost and quantity. -/
structure CandySet where
  cost : ℕ
  quantity : ℕ

/-- The problem setup -/
def candy_problem :=
  let set1 : CandySet := ⟨50, 25⟩
  let set2 : CandySet := ⟨180, 95⟩
  let set3 : CandySet := ⟨150, 80⟩
  let total_budget : ℕ := 2200
  (set1, set2, set3, total_budget)

/-- Calculate the total cost of the purchase -/
def total_cost (x y z : ℕ) : ℕ :=
  let (set1, set2, set3, _) := candy_problem
  x * set1.cost + y * set2.cost + z * set3.cost

/-- Calculate the total number of candies -/
def total_candies (x y z : ℕ) : ℕ :=
  let (set1, set2, set3, _) := candy_problem
  x * set1.quantity + y * set2.quantity + z * set3.quantity

/-- Check if the purchase is within budget -/
def within_budget (x y z : ℕ) : Prop :=
  let (_, _, _, budget) := candy_problem
  total_cost x y z ≤ budget

/-- The main theorem stating that (2, 5, 8) is the optimal solution -/
theorem optimal_candy_purchase :
  within_budget 2 5 8 ∧
  (∀ x y z : ℕ, within_budget x y z → total_candies x y z ≤ total_candies 2 5 8) :=
sorry

end NUMINAMATH_CALUDE_optimal_candy_purchase_l43_4301


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l43_4350

theorem modular_arithmetic_problem :
  (3 * (7⁻¹ : ZMod 120) + 9 * (13⁻¹ : ZMod 120) + 4 * (17⁻¹ : ZMod 120)) = (86 : ZMod 120) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l43_4350


namespace NUMINAMATH_CALUDE_range_of_a_l43_4302

/-- The range of a for which ¬p is a necessary but not sufficient condition for ¬q -/
theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → 
    ((x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0))) →
  (∃ x : ℝ, ((x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)) ∧ 
    (x^2 - 4*a*x + 3*a^2 ≥ 0)) →
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l43_4302


namespace NUMINAMATH_CALUDE_equation_solution_exists_l43_4362

theorem equation_solution_exists : ∃ x : ℤ, 
  |x - ((1125 - 500 + 660 - 200) * (3/2) * (3/4) / 45)| ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l43_4362


namespace NUMINAMATH_CALUDE_cubic_function_properties_l43_4386

noncomputable def f (a b x : ℝ) := x^3 + 3*(a-1)*x^2 - 12*a*x + b

theorem cubic_function_properties (a b : ℝ) :
  let f := f a b
  ∃ (x₁ x₂ M N : ℝ),
    (∀ x, x ≠ x₁ → x ≠ x₂ → f x ≤ f x₁ ∨ f x ≥ f x₂) →
    (∃ m c, ∀ x, m*x - f x - c = 0 → x = 0 ∧ m = 24 ∧ c = 10) →
    (x₁ = 2 ∧ x₂ = 4 ∧ M = f x₁ ∧ N = f x₂ ∧ M = 10 ∧ N = 6) ∧
    (f 1 > f 2 → x₂ - x₁ = 4 → b = 10 →
      (∀ x, x ≤ -2 → f x ≤ f (-2)) ∧
      (∀ x, -2 ≤ x ∧ x ≤ 2 → f 2 ≤ f x) ∧
      (∀ x, 2 ≤ x → f x ≥ f 2) ∧
      M = 26 ∧ N = -6) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l43_4386


namespace NUMINAMATH_CALUDE_right_focus_coordinates_l43_4319

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 64 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (10, 0)

/-- Theorem: The right focus of the given hyperbola is (10, 0) -/
theorem right_focus_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → right_focus = (10, 0) := by
  sorry

end NUMINAMATH_CALUDE_right_focus_coordinates_l43_4319


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l43_4331

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area (a b h x : ℝ) (hb : b > a) (hh : h > 0) (hx : 0 < x ∧ x < h) :
  let rectangle_area := (b - a) * x * (h - x) / h
  rectangle_area = (b - a) * x * (h - x) / h := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l43_4331


namespace NUMINAMATH_CALUDE_general_equation_pattern_l43_4303

theorem general_equation_pattern (n : ℝ) : n ≠ 4 ∧ (8 - n) ≠ 4 →
  n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_general_equation_pattern_l43_4303


namespace NUMINAMATH_CALUDE_trig_expression_max_value_l43_4396

theorem trig_expression_max_value (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_max_value_l43_4396


namespace NUMINAMATH_CALUDE_vasya_number_exists_l43_4373

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = 8) ∧
  (digits.count 1 = 2) ∧ (digits.count 2 = 2) ∧ (digits.count 3 = 2) ∧ (digits.count 4 = 2) ∧
  (∃ i, digits.get? i = some 1 ∧ digits.get? (i + 2) = some 1) ∧
  (∃ i, digits.get? i = some 2 ∧ digits.get? (i + 3) = some 2) ∧
  (∃ i, digits.get? i = some 3 ∧ digits.get? (i + 4) = some 3) ∧
  (∃ i, digits.get? i = some 4 ∧ digits.get? (i + 5) = some 4)

theorem vasya_number_exists : ∃ n : ℕ, is_valid_number n := by
  sorry

end NUMINAMATH_CALUDE_vasya_number_exists_l43_4373


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l43_4361

theorem polar_to_rectangular_conversion (r : ℝ) (θ : ℝ) :
  r = 6 ∧ θ = π / 3 →
  (r * Real.cos θ = 3 ∧ r * Real.sin θ = 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l43_4361


namespace NUMINAMATH_CALUDE_ann_total_blocks_l43_4327

/-- Ann's initial number of blocks -/
def initial_blocks : ℕ := 9

/-- Number of blocks Ann finds -/
def found_blocks : ℕ := 44

/-- Theorem stating the total number of blocks Ann ends with -/
theorem ann_total_blocks : initial_blocks + found_blocks = 53 := by
  sorry

end NUMINAMATH_CALUDE_ann_total_blocks_l43_4327


namespace NUMINAMATH_CALUDE_expand_and_simplify_l43_4379

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l43_4379


namespace NUMINAMATH_CALUDE_largest_n_when_floor_sqrt_n_is_5_l43_4385

/-- Floor function: largest integer not greater than x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem largest_n_when_floor_sqrt_n_is_5 :
  ∀ n : ℕ, (floor (Real.sqrt n) = 5) → (∀ m : ℕ, m ≤ n → m ≤ 35) ∧ n ≤ 35 :=
sorry

end NUMINAMATH_CALUDE_largest_n_when_floor_sqrt_n_is_5_l43_4385


namespace NUMINAMATH_CALUDE_smallest_n_for_eq1_smallest_n_for_eq2_l43_4369

-- Define the properties for the equations
def satisfies_eq1 (n : ℕ) : Prop :=
  ∃ x y : ℕ, x * (x + n) = y^2

def satisfies_eq2 (n : ℕ) : Prop :=
  ∃ x y : ℕ, x * (x + n) = y^3

-- Define the smallest n for each equation
def smallest_n1 : ℕ := 3
def smallest_n2 : ℕ := 2

-- Theorem for the first equation
theorem smallest_n_for_eq1 :
  satisfies_eq1 smallest_n1 ∧
  ∀ m : ℕ, m < smallest_n1 → ¬(satisfies_eq1 m) :=
by sorry

-- Theorem for the second equation
theorem smallest_n_for_eq2 :
  satisfies_eq2 smallest_n2 ∧
  ∀ m : ℕ, m < smallest_n2 → ¬(satisfies_eq2 m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_eq1_smallest_n_for_eq2_l43_4369


namespace NUMINAMATH_CALUDE_not_all_even_P_true_l43_4399

/-- A proposition P on even natural numbers -/
def P : ℕ → Prop := sorry

/-- Theorem stating that we cannot conclude P holds for all even natural numbers -/
theorem not_all_even_P_true :
  (∀ n : ℕ, n ≤ 1001 → P (2 * n)) →
  ¬(∀ k : ℕ, Even k → P k) :=
by sorry

end NUMINAMATH_CALUDE_not_all_even_P_true_l43_4399
