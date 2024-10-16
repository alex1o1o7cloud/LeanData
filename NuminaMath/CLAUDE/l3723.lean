import Mathlib

namespace NUMINAMATH_CALUDE_sphere_xz_intersection_radius_l3723_372301

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A circle in 3D space -/
structure Circle where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Theorem: The radius of the circle where the sphere intersects the xz-plane is 6 -/
theorem sphere_xz_intersection_radius : 
  ∀ (s : Sphere),
  ∃ (c1 c2 : Circle),
  c1.center = (3, 5, 0) ∧ c1.radius = 3 ∧  -- xy-plane intersection
  c2.center = (0, 5, -6) ∧                 -- xz-plane intersection
  (∃ (x y z : ℝ), s.center = (x, y, z)) →
  c2.radius = 6 := by
sorry


end NUMINAMATH_CALUDE_sphere_xz_intersection_radius_l3723_372301


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3723_372387

theorem largest_digit_sum (a b c z : ℕ) : 
  (a < 10) → (b < 10) → (c < 10) → 
  (0 < z) → (z ≤ 12) → 
  (100 * a + 10 * b + c = 1000 / z) → 
  (a + b + c ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3723_372387


namespace NUMINAMATH_CALUDE_students_to_add_l3723_372314

theorem students_to_add (current_students : ℕ) (teachers : ℕ) (h1 : current_students = 1049) (h2 : teachers = 9) :
  ∃ (students_to_add : ℕ), 
    students_to_add = 4 ∧
    (current_students + students_to_add) % teachers = 0 ∧
    ∀ (n : ℕ), n < students_to_add → (current_students + n) % teachers ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_students_to_add_l3723_372314


namespace NUMINAMATH_CALUDE_sin_increasing_interval_l3723_372394

/-- The function f with given properties has (-π/12, 5π/12) as its strictly increasing interval -/
theorem sin_increasing_interval (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x + π / 6)
  (∀ x, f x > 0) →
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ π) →
  (∃ p, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = π) →
  (∀ x ∈ Set.Ioo (-π/12 : ℝ) (5*π/12), StrictMono f) :=
by
  sorry

end NUMINAMATH_CALUDE_sin_increasing_interval_l3723_372394


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l3723_372390

theorem product_of_roots_cubic (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + 4 * a - 12 = 0) ∧
  (3 * b^3 - 9 * b^2 + 4 * b - 12 = 0) ∧
  (3 * c^3 - 9 * c^2 + 4 * c - 12 = 0) →
  a * b * c = 4 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l3723_372390


namespace NUMINAMATH_CALUDE_point_transformation_l3723_372343

/-- Given a point P(a,b) in the xy-plane, this theorem proves that if P is first rotated
    clockwise by 180° around the origin (0,0) and then reflected about the line y = -x,
    resulting in the point (9,-4), then b - a = -13. -/
theorem point_transformation (a b : ℝ) : 
  (∃ (x y : ℝ), ((-a) = x ∧ (-b) = y) ∧ (y = x ∧ -x = 9 ∧ -y = -4)) → b - a = -13 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3723_372343


namespace NUMINAMATH_CALUDE_tile_arrangements_l3723_372339

/-- The number of distinguishable arrangements of tiles of different colors -/
def distinguishable_arrangements (blue red green : ℕ) : ℕ :=
  Nat.factorial (blue + red + green) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green)

/-- Theorem stating that the number of distinguishable arrangements
    of 3 blue tiles, 2 red tiles, and 4 green tiles is 1260 -/
theorem tile_arrangements :
  distinguishable_arrangements 3 2 4 = 1260 := by
  sorry

#eval distinguishable_arrangements 3 2 4

end NUMINAMATH_CALUDE_tile_arrangements_l3723_372339


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l3723_372385

/-- Given that for all k ∈ ℝ, the line y - kx - 1 = 0 always intersects 
    with the ellipse x²/4 + y²/m = 1, prove that the range of m is [1, 4) ∪ (4, +∞) -/
theorem ellipse_line_intersection_range (m : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y - k*x - 1 = 0 ∧ x^2/4 + y^2/m = 1) ↔ 
  (m ∈ Set.Icc 1 4 ∪ Set.Ioi 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l3723_372385


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3723_372375

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + 1/b) ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3723_372375


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3723_372302

/-- Represents the side lengths of the nine squares in the rectangle -/
structure SquareSides where
  a1 : ℕ
  a2 : ℕ
  a3 : ℕ
  a4 : ℕ
  a5 : ℕ
  a6 : ℕ
  a7 : ℕ
  a8 : ℕ
  a9 : ℕ

/-- Checks if the given SquareSides satisfy the conditions of the problem -/
def isValidSquareSides (s : SquareSides) : Prop :=
  s.a1 = 2 ∧
  s.a1 + s.a2 = s.a3 ∧
  s.a1 + s.a3 = s.a4 ∧
  s.a3 + s.a4 = s.a5 ∧
  s.a4 + s.a5 = s.a6 ∧
  s.a2 + s.a3 + s.a5 = s.a7 ∧
  s.a2 + s.a7 = s.a8 ∧
  s.a1 + s.a4 + s.a6 = s.a9 ∧
  s.a6 + s.a9 = s.a7 + s.a8

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the given RectangleDimensions satisfy the conditions of the problem -/
def isValidRectangle (r : RectangleDimensions) : Prop :=
  r.length > r.width ∧
  Even r.length ∧
  Even r.width ∧
  r.length = r.width + 2

theorem rectangle_perimeter (s : SquareSides) (r : RectangleDimensions) :
  isValidSquareSides s → isValidRectangle r →
  r.length = s.a9 → r.width = s.a8 →
  2 * (r.length + r.width) = 68 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l3723_372302


namespace NUMINAMATH_CALUDE_work_time_ratio_l3723_372368

/-- Given two workers A and B, this theorem proves the ratio of their individual work times
    based on their combined work time and B's individual work time. -/
theorem work_time_ratio (time_together time_B : ℝ) (h1 : time_together = 4) (h2 : time_B = 24) :
  ∃ time_A : ℝ, time_A / time_B = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l3723_372368


namespace NUMINAMATH_CALUDE_square_circle_octagon_l3723_372361

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Square type
structure Square :=
  (a b c d : Point)

-- Define a Circle type
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define an Octagon type
structure Octagon :=
  (vertices : List Point)

-- Function to check if a square can be circumscribed by a circle
def can_circumscribe (s : Square) (c : Circle) : Prop :=
  sorry

-- Function to check if an octagon is regular and inscribed in a circle
def is_regular_inscribed_octagon (o : Octagon) (c : Circle) : Prop :=
  sorry

-- Main theorem
theorem square_circle_octagon (s : Square) :
  ∃ (c : Circle) (o : Octagon),
    can_circumscribe s c ∧ is_regular_inscribed_octagon o c :=
  sorry

end NUMINAMATH_CALUDE_square_circle_octagon_l3723_372361


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3723_372399

/-- Represents the pricing strategy of a merchant --/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price given the list price and purchase discount --/
def purchase_price (mp : MerchantPricing) : ℝ :=
  mp.list_price * (1 - mp.purchase_discount)

/-- Calculates the selling price given the marked price and selling discount --/
def selling_price (mp : MerchantPricing) : ℝ :=
  mp.marked_price * (1 - mp.selling_discount)

/-- Checks if the pricing strategy satisfies the profit margin requirement --/
def satisfies_profit_margin (mp : MerchantPricing) : Prop :=
  selling_price mp - purchase_price mp = mp.profit_margin * selling_price mp

/-- The main theorem to prove --/
theorem merchant_pricing_strategy (mp : MerchantPricing) 
  (h1 : mp.purchase_discount = 0.3)
  (h2 : mp.selling_discount = 0.2)
  (h3 : mp.profit_margin = 0.2)
  (h4 : satisfies_profit_margin mp) :
  mp.marked_price / mp.list_price = 1.09375 := by
  sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3723_372399


namespace NUMINAMATH_CALUDE_total_stamps_l3723_372365

def stamps_problem (snowflake truck rose : ℕ) : Prop :=
  (snowflake = 11) ∧
  (truck = snowflake + 9) ∧
  (rose = truck - 13)

theorem total_stamps :
  ∀ snowflake truck rose : ℕ,
    stamps_problem snowflake truck rose →
    snowflake + truck + rose = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l3723_372365


namespace NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_fifteen_is_minimum_removal_l3723_372315

/-- Represents a triangular figure constructed with toothpicks -/
structure TriangularFigure where
  total_toothpicks : ℕ
  horizontal_toothpicks : ℕ
  has_upward_triangles : Bool
  has_downward_triangles : Bool

/-- Represents the number of toothpicks that need to be removed -/
def toothpicks_to_remove (figure : TriangularFigure) : ℕ := figure.horizontal_toothpicks

/-- Theorem stating that removing horizontal toothpicks eliminates all triangles -/
theorem remove_horizontal_eliminates_triangles (figure : TriangularFigure) 
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.horizontal_toothpicks = 15)
  (h3 : figure.has_upward_triangles = true)
  (h4 : figure.has_downward_triangles = true) :
  toothpicks_to_remove figure = 15 ∧ 
  (∀ n : ℕ, n < 15 → ∃ triangle_remains : Bool, triangle_remains = true) := by
  sorry

/-- Theorem stating that 15 is the minimum number of toothpicks to remove -/
theorem fifteen_is_minimum_removal (figure : TriangularFigure)
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.horizontal_toothpicks = 15)
  (h3 : figure.has_upward_triangles = true)
  (h4 : figure.has_downward_triangles = true) :
  ∀ n : ℕ, n < 15 → ∃ triangle_remains : Bool, triangle_remains = true := by
  sorry

end NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_fifteen_is_minimum_removal_l3723_372315


namespace NUMINAMATH_CALUDE_max_a_value_l3723_372313

theorem max_a_value (a b c d : ℕ+) 
  (h1 : a < 2 * b + 1)
  (h2 : b < 3 * c + 1)
  (h3 : c < 4 * d + 1)
  (h4 : d^2 < 10000) :
  a ≤ 2376 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3723_372313


namespace NUMINAMATH_CALUDE_only_crop_yield_fertilizer_correlational_l3723_372349

-- Define the types of relationships
inductive Relationship
| Functional
| Correlational

-- Define the variables for each relationship
def height_age_relation : Relationship := sorry
def cube_volume_edge_relation : Relationship := sorry
def pencils_money_relation : Relationship := sorry
def crop_yield_fertilizer_relation : Relationship := sorry

-- Theorem stating that only the crop yield and fertilizer relationship is correlational
theorem only_crop_yield_fertilizer_correlational :
  (height_age_relation = Relationship.Functional) ∧
  (cube_volume_edge_relation = Relationship.Functional) ∧
  (pencils_money_relation = Relationship.Functional) ∧
  (crop_yield_fertilizer_relation = Relationship.Correlational) := by sorry

end NUMINAMATH_CALUDE_only_crop_yield_fertilizer_correlational_l3723_372349


namespace NUMINAMATH_CALUDE_badminton_cost_comparison_l3723_372321

/-- Represents the cost calculation and comparison for purchasing badminton equipment from two supermarkets. -/
theorem badminton_cost_comparison 
  (x : ℝ) 
  (h_x : x ≥ 3) 
  (racket_price : ℝ) 
  (h_racket : racket_price = 40) 
  (shuttlecock_price : ℝ) 
  (h_shuttlecock : shuttlecock_price = 4) 
  (num_pairs : ℕ) 
  (h_num_pairs : num_pairs = 10) 
  (discount_A : ℝ) 
  (h_discount_A : discount_A = 0.8) 
  (free_shuttlecocks_B : ℕ) 
  (h_free_B : free_shuttlecocks_B = 3) :
  let y_A := (racket_price * discount_A * num_pairs) + (shuttlecock_price * discount_A * num_pairs * x)
  let y_B := (racket_price * num_pairs) + (shuttlecock_price * (num_pairs * x - num_pairs * free_shuttlecocks_B))
  ∃ (y_A y_B : ℝ),
    (y_A = 32 * x + 320 ∧ y_B = 40 * x + 280) ∧
    (x = 5 → y_A = y_B) ∧
    (3 ≤ x ∧ x < 5 → y_A > y_B) ∧
    (x > 5 → y_A < y_B) := by
  sorry

end NUMINAMATH_CALUDE_badminton_cost_comparison_l3723_372321


namespace NUMINAMATH_CALUDE_moon_speed_mph_approx_l3723_372318

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℝ := 0.621371

/-- Conversion factor from seconds to hours -/
def seconds_to_hours : ℝ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_s : ℝ := 1.02

/-- Converts a speed from kilometers per second to miles per hour -/
def convert_km_s_to_mph (speed_km_s : ℝ) : ℝ :=
  speed_km_s * km_to_miles * seconds_to_hours

/-- Theorem stating that the moon's speed in miles per hour is approximately 2281.34 -/
theorem moon_speed_mph_approx :
  ∃ ε > 0, |convert_km_s_to_mph moon_speed_km_s - 2281.34| < ε :=
sorry

end NUMINAMATH_CALUDE_moon_speed_mph_approx_l3723_372318


namespace NUMINAMATH_CALUDE_intersection_equality_l3723_372358

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_equality (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = -1 ∨ a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_intersection_equality_l3723_372358


namespace NUMINAMATH_CALUDE_expression_simplification_l3723_372384

theorem expression_simplification : 
  ((3 + 4 + 5 + 6 + 7) / 3) + ((3 * 6 + 12) / 4) = 95 / 6 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3723_372384


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3723_372376

theorem solution_set_of_inequality (x : ℝ) :
  (1 ≤ |x + 2| ∧ |x + 2| ≤ 5) ↔ ((-7 ≤ x ∧ x ≤ -3) ∨ (-1 ≤ x ∧ x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3723_372376


namespace NUMINAMATH_CALUDE_all_admissible_triangles_finite_and_generable_l3723_372348

-- Define an admissible angle
def AdmissibleAngle (n : ℕ) (m : ℕ) : ℚ := (m * 180) / n

-- Define an admissible triangle
structure AdmissibleTriangle (n : ℕ) where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  sum_180 : AdmissibleAngle n angle1 + AdmissibleAngle n angle2 + AdmissibleAngle n angle3 = 180
  angle1_pos : angle1 > 0
  angle2_pos : angle2 > 0
  angle3_pos : angle3 > 0

-- Define a function to check if two triangles are similar
def areSimilar (n : ℕ) (t1 t2 : AdmissibleTriangle n) : Prop :=
  (t1.angle1 = t2.angle1 ∧ t1.angle2 = t2.angle2 ∧ t1.angle3 = t2.angle3) ∨
  (t1.angle1 = t2.angle2 ∧ t1.angle2 = t2.angle3 ∧ t1.angle3 = t2.angle1) ∨
  (t1.angle1 = t2.angle3 ∧ t1.angle2 = t2.angle1 ∧ t1.angle3 = t2.angle2)

-- Define the set of all possible admissible triangles
def AllAdmissibleTriangles (n : ℕ) : Set (AdmissibleTriangle n) :=
  {t : AdmissibleTriangle n | True}

-- Define the process of cutting triangles
def CutTriangle (n : ℕ) (t : AdmissibleTriangle n) : 
  Option (AdmissibleTriangle n × AdmissibleTriangle n) :=
  sorry -- Implementation details omitted

-- The main theorem
theorem all_admissible_triangles_finite_and_generable 
  (n : ℕ) (h_prime : Nat.Prime n) (h_gt_3 : n > 3) :
  ∃ (S : Set (AdmissibleTriangle n)),
    Finite S ∧ 
    (∀ t ∈ AllAdmissibleTriangles n, ∃ s ∈ S, areSimilar n t s) ∧
    (∀ t ∈ S, CutTriangle n t = none) :=
  sorry


end NUMINAMATH_CALUDE_all_admissible_triangles_finite_and_generable_l3723_372348


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3723_372344

theorem polynomial_remainder (q : ℝ → ℝ) :
  (∃ r : ℝ → ℝ, ∀ x, q x = (x - 2) * r x + 3) →
  (∃ s : ℝ → ℝ, ∀ x, q x = (x + 3) * s x - 9) →
  ∃ t : ℝ → ℝ, ∀ x, q x = (x - 2) * (x + 3) * t x + (12/5 * x - 9/5) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3723_372344


namespace NUMINAMATH_CALUDE_triangle_abc_right_angled_l3723_372312

theorem triangle_abc_right_angled (A B C : ℝ) 
  (h1 : A = (1/2) * B) 
  (h2 : A = (1/3) * C) 
  (h3 : A + B + C = 180) : 
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_right_angled_l3723_372312


namespace NUMINAMATH_CALUDE_complement_union_equality_l3723_372323

universe u

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {2, 4, 6}

theorem complement_union_equality : 
  (U \ M) ∪ (U \ N) = U := by sorry

end NUMINAMATH_CALUDE_complement_union_equality_l3723_372323


namespace NUMINAMATH_CALUDE_only_one_correct_statement_l3723_372366

theorem only_one_correct_statement :
  (∃! n : Nat, n = 1 ∧
    (¬ (∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/b < 1/a)) ∧
    (¬ (∀ a b c : ℝ, a < b → a*c < b*c)) ∧
    ((∀ a b c : ℝ, a < b → a + c < b + c)) ∧
    (¬ (∀ a b : ℝ, a^2 < b^2 → a < b))) :=
by sorry

end NUMINAMATH_CALUDE_only_one_correct_statement_l3723_372366


namespace NUMINAMATH_CALUDE_worker_earnings_worker_earnings_proof_l3723_372328

/-- Calculates the total earnings of a worker based on regular and cellphone survey rates -/
theorem worker_earnings (regular_rate : ℕ) (total_surveys : ℕ) (cellphone_rate_increase : ℚ) 
  (cellphone_surveys : ℕ) (h1 : regular_rate = 30) (h2 : total_surveys = 100) 
  (h3 : cellphone_rate_increase = 1/5) (h4 : cellphone_surveys = 50) : ℕ :=
  let regular_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate + (regular_rate * cellphone_rate_increase).floor
  let regular_pay := regular_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  let total_pay := regular_pay + cellphone_pay
  3300

/-- The worker's total earnings for the week are Rs. 3300 -/
theorem worker_earnings_proof : worker_earnings 30 100 (1/5) 50 rfl rfl rfl rfl = 3300 := by
  sorry

end NUMINAMATH_CALUDE_worker_earnings_worker_earnings_proof_l3723_372328


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3723_372305

theorem quadratic_inequality_range (m : ℝ) (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) →
  (∀ b : ℝ, b > a → m > b) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3723_372305


namespace NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l3723_372307

variable {A B : Type}
variable (f : A → B)

theorem mapping_not_necessarily_injective : 
  ¬(∀ (x y : A), f x = f y → x = y) :=
sorry

end NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l3723_372307


namespace NUMINAMATH_CALUDE_sum_of_products_l3723_372379

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 75)
  (h2 : y^2 + y*z + z^2 = 4)
  (h3 : z^2 + x*z + x^2 = 79) :
  x*y + y*z + x*z = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3723_372379


namespace NUMINAMATH_CALUDE_bicycle_shop_optimal_plan_l3723_372337

/-- Represents the purchase plan for bicycles -/
structure BicyclePlan where
  modelA : ℕ
  modelB : ℕ

/-- The bicycle shop problem -/
theorem bicycle_shop_optimal_plan :
  ∀ (plan : BicyclePlan),
  plan.modelA + plan.modelB = 50 →
  plan.modelB ≥ plan.modelA →
  1000 * plan.modelA + 1600 * plan.modelB ≤ 68000 →
  ∃ (optimalPlan : BicyclePlan),
  optimalPlan.modelA = 25 ∧
  optimalPlan.modelB = 25 ∧
  ∀ (p : BicyclePlan),
  p.modelA + p.modelB = 50 →
  p.modelB ≥ p.modelA →
  1000 * p.modelA + 1600 * p.modelB ≤ 68000 →
  500 * p.modelA + 400 * p.modelB ≤ 500 * optimalPlan.modelA + 400 * optimalPlan.modelB ∧
  500 * optimalPlan.modelA + 400 * optimalPlan.modelB = 22500 :=
by
  sorry


end NUMINAMATH_CALUDE_bicycle_shop_optimal_plan_l3723_372337


namespace NUMINAMATH_CALUDE_ratio_M_N_l3723_372341

theorem ratio_M_N (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.75 * R)
  (hR : R = 0.6 * P)
  (hP : P ≠ 0) : 
  M / N = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_M_N_l3723_372341


namespace NUMINAMATH_CALUDE_intersecting_lines_determine_plane_l3723_372392

-- Define the concepts of point, line, and plane
variable (Point Line Plane : Type)

-- Define the concept of intersection for lines
variable (intersect : Line → Line → Prop)

-- Define the concept of a line lying on a plane
variable (lieOn : Line → Plane → Prop)

-- Define the concept of a plane containing two lines
variable (contains : Plane → Line → Line → Prop)

-- Theorem: Two intersecting lines determine a unique plane
theorem intersecting_lines_determine_plane
  (l1 l2 : Line)
  (h_intersect : intersect l1 l2)
  : ∃! p : Plane, contains p l1 l2 :=
sorry

end NUMINAMATH_CALUDE_intersecting_lines_determine_plane_l3723_372392


namespace NUMINAMATH_CALUDE_namjoons_position_proof_l3723_372371

def namjoons_position (seokjins_position : ℕ) (people_between : ℕ) : ℕ :=
  seokjins_position + people_between

theorem namjoons_position_proof (seokjins_position : ℕ) (people_between : ℕ) :
  namjoons_position seokjins_position people_between = seokjins_position + people_between :=
by
  sorry

end NUMINAMATH_CALUDE_namjoons_position_proof_l3723_372371


namespace NUMINAMATH_CALUDE_composite_function_equality_l3723_372317

theorem composite_function_equality (x : ℝ) (hx : x > 0) :
  Real.sin (Real.log (Real.sqrt x)) = Real.sin ((1 / 2) * Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_composite_function_equality_l3723_372317


namespace NUMINAMATH_CALUDE_crayon_purchase_worth_l3723_372324

/-- Calculates the total worth of crayons after a discounted purchase -/
theorem crayon_purchase_worth
  (initial_packs : ℕ)
  (additional_packs : ℕ)
  (regular_price : ℝ)
  (discount_percent : ℝ)
  (h1 : initial_packs = 4)
  (h2 : additional_packs = 2)
  (h3 : regular_price = 2.5)
  (h4 : discount_percent = 15)
  : ℝ := by
  sorry

#check crayon_purchase_worth

end NUMINAMATH_CALUDE_crayon_purchase_worth_l3723_372324


namespace NUMINAMATH_CALUDE_orange_bin_problem_l3723_372311

theorem orange_bin_problem (initial : ℕ) (thrown_away : ℕ) (final : ℕ) 
  (h1 : initial = 40)
  (h2 : thrown_away = 25)
  (h3 : final = 36) :
  final - (initial - thrown_away) = 21 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l3723_372311


namespace NUMINAMATH_CALUDE_bennys_card_collection_l3723_372359

theorem bennys_card_collection (original_cards : ℕ) (remaining_cards : ℕ) : 
  (remaining_cards = original_cards / 2) → (remaining_cards = 34) → (original_cards = 68) := by
  sorry

end NUMINAMATH_CALUDE_bennys_card_collection_l3723_372359


namespace NUMINAMATH_CALUDE_cos_pi_minus_2theta_l3723_372326

theorem cos_pi_minus_2theta (θ : Real) (h : ∃ (x y : Real), x = 3 ∧ y = -4 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) :
  Real.cos (π - 2*θ) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2theta_l3723_372326


namespace NUMINAMATH_CALUDE_smallest_circle_covering_region_line_intersecting_circle_l3723_372383

-- Define the planar region
def planar_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle (C)
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Define the line (l)
def line_l (x y : ℝ) : Prop :=
  y = x - 1 + Real.sqrt 5 ∨ y = x - 1 - Real.sqrt 5

-- Theorem for the smallest circle covering the region
theorem smallest_circle_covering_region :
  (∀ x y, planar_region x y → circle_C x y) ∧
  (∀ x' y', (∀ x y, planar_region x y → (x - x')^2 + (y - y')^2 ≤ r'^2) →
    r'^2 ≥ 5) :=
sorry

-- Theorem for the line intersecting the circle
theorem line_intersecting_circle :
  ∃ A B : ℝ × ℝ,
    A ≠ B ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    ((A.1 - 2) * (B.1 - 2) + (A.2 - 1) * (B.2 - 1) = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_covering_region_line_intersecting_circle_l3723_372383


namespace NUMINAMATH_CALUDE_clock_resale_price_l3723_372300

theorem clock_resale_price (original_cost : ℝ) : 
  original_cost > 0 →
  let initial_sale_price := 1.2 * original_cost
  let buy_back_price := 0.5 * initial_sale_price
  let profit_margin := 0.8
  original_cost - buy_back_price = 100 →
  buy_back_price * (1 + profit_margin) = 270 := by
sorry


end NUMINAMATH_CALUDE_clock_resale_price_l3723_372300


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3723_372388

theorem polar_to_rectangular_conversion :
  let r : ℝ := 10
  let θ : ℝ := 5 * π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 5 ∧ y = -5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3723_372388


namespace NUMINAMATH_CALUDE_subtraction_with_division_l3723_372320

theorem subtraction_with_division : 6000 - (105 / 21.0) = 5995 := by sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l3723_372320


namespace NUMINAMATH_CALUDE_combinatorial_identity_l3723_372377

theorem combinatorial_identity (n : ℕ) : 
  (n.choose 2) * 2 = 42 → n.choose 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l3723_372377


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3723_372310

/-- A pyramid with an equilateral triangular base and isosceles lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)
  (is_equilateral_base : base_side > 0)
  (is_isosceles_lateral : height^2 + (base_side/2)^2 = (base_side * Real.sqrt 3 / 2)^2)

/-- A cube inscribed in the pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (touches_base_center : side_length ≤ p.base_side * Real.sqrt 3 / 3)
  (touches_apex : 2 * side_length = p.height)

/-- The volume of the inscribed cube is 1/64 -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) 
    (h1 : p.base_side = 1) : c.side_length^3 = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3723_372310


namespace NUMINAMATH_CALUDE_corners_removed_cube_edges_l3723_372325

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents the solid formed by removing smaller cubes from corners of a larger cube -/
structure CornersRemovedCube where
  originalCube : Cube
  removedCubeSideLength : ℝ
  removedCubeSideLength_pos : removedCubeSideLength > 0
  validRemoval : removedCubeSideLength < originalCube.sideLength / 2

/-- Calculates the number of edges in the resulting solid after removing smaller cubes from corners -/
def edgesAfterRemoval (c : CornersRemovedCube) : ℕ :=
  sorry

/-- Theorem stating that removing cubes of side length 2 from corners of a cube with side length 6 results in a solid with 36 edges -/
theorem corners_removed_cube_edges :
  let originalCube : Cube := ⟨6, by norm_num⟩
  let cornersRemovedCube : CornersRemovedCube := ⟨originalCube, 2, by norm_num, by norm_num⟩
  edgesAfterRemoval cornersRemovedCube = 36 :=
sorry

end NUMINAMATH_CALUDE_corners_removed_cube_edges_l3723_372325


namespace NUMINAMATH_CALUDE_y_axis_intersection_x_axis_intersections_l3723_372335

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem for y-axis intersection
theorem y_axis_intersection : f 0 = 2 := by sorry

-- Theorem for x-axis intersections
theorem x_axis_intersections :
  (f 2 = 0 ∧ f 1 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 2 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_y_axis_intersection_x_axis_intersections_l3723_372335


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l3723_372334

/-- Given two points A and B in the plane, this theorem states that the equation of the circle
    for which the segment AB is a diameter is (x-1)^2+(y+3)^2=116. -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (-4, -5) →
  B = (6, -1) →
  ∀ x y : ℝ, (x - 1)^2 + (y + 3)^2 = 116 ↔
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    x = -4 * (1 - t) + 6 * t ∧
    y = -5 * (1 - t) - 1 * t :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l3723_372334


namespace NUMINAMATH_CALUDE_wage_increase_proof_l3723_372309

theorem wage_increase_proof (original_wage new_wage : ℝ) 
  (h1 : new_wage = 70)
  (h2 : new_wage = original_wage * (1 + 0.16666666666666664)) :
  original_wage = 60 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_proof_l3723_372309


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l3723_372364

def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (∃ k : ℕ, n = 6 * k)

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l3723_372364


namespace NUMINAMATH_CALUDE_fraction_equality_l3723_372332

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 4 / 3) 
  (h2 : r / t = 9 / 14) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3723_372332


namespace NUMINAMATH_CALUDE_other_number_proof_l3723_372397

theorem other_number_proof (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by sorry

end NUMINAMATH_CALUDE_other_number_proof_l3723_372397


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l3723_372345

/-- The number of sections created by n line segments in a rectangle -/
def sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else sections (n - 1) + (n - 1)

/-- The maximum number of sections created by 5 line segments in a rectangle -/
theorem max_sections_five_lines :
  sections 5 = 12 :=
sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l3723_372345


namespace NUMINAMATH_CALUDE_hyperbola_transformation_l3723_372393

/-- Given a hyperbola with equation x^2/4 - y^2/5 = 1, 
    prove that the standard equation of a hyperbola with the same foci as vertices 
    and perpendicular asymptotes is x^2/9 - y^2/9 = 1 -/
theorem hyperbola_transformation (x y : ℝ) : 
  (∃ (a b : ℝ), x^2/a - y^2/b = 1 ∧ a = 4 ∧ b = 5) →
  (∃ (c : ℝ), x^2/c - y^2/c = 1 ∧ c = 9) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_transformation_l3723_372393


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_l3723_372354

/-- The prime counting function -/
def prime_counting (n : ℕ) : ℕ := sorry

/-- π(n) is non-decreasing -/
axiom prime_counting_nondecreasing : ∀ m n : ℕ, m ≤ n → prime_counting m ≤ prime_counting n

/-- The set of integers n such that π(n) divides n -/
def divisible_set : Set ℕ := {n : ℕ | prime_counting n ∣ n}

/-- There are infinitely many integers n such that π(n) divides n -/
theorem infinitely_many_divisible : Set.Infinite divisible_set := by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_l3723_372354


namespace NUMINAMATH_CALUDE_x_squared_in_set_l3723_372350

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l3723_372350


namespace NUMINAMATH_CALUDE_expand_product_l3723_372398

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3723_372398


namespace NUMINAMATH_CALUDE_sector_radius_l3723_372356

/-- Given a sector with a central angle of 150° and an arc length of 5π/2 cm, its radius is 3 cm. -/
theorem sector_radius (θ : ℝ) (arc_length : ℝ) (radius : ℝ) : 
  θ = 150 → 
  arc_length = (5/2) * Real.pi → 
  arc_length = (θ / 360) * 2 * Real.pi * radius → 
  radius = 3 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l3723_372356


namespace NUMINAMATH_CALUDE_vasims_share_l3723_372353

/-- Represents the distribution of money among three people -/
structure Distribution where
  faruk : ℕ
  vasim : ℕ
  ranjith : ℕ

/-- Checks if the distribution follows the given ratio -/
def is_valid_ratio (d : Distribution) : Prop :=
  11 * d.faruk = 3 * d.ranjith ∧ 5 * d.faruk = 3 * d.vasim

/-- The main theorem to prove -/
theorem vasims_share (d : Distribution) :
  is_valid_ratio d → d.ranjith - d.faruk = 2400 → d.vasim = 1500 := by
  sorry


end NUMINAMATH_CALUDE_vasims_share_l3723_372353


namespace NUMINAMATH_CALUDE_archie_record_l3723_372369

/-- The number of games in a season -/
def season_length : ℕ := 16

/-- Richard's average touchdowns per game in the first 14 games -/
def richard_average : ℕ := 6

/-- The number of games Richard has played so far -/
def games_played : ℕ := 14

/-- The number of remaining games -/
def remaining_games : ℕ := season_length - games_played

/-- The average number of touchdowns Richard needs in the remaining games to beat Archie's record -/
def needed_average : ℕ := 3

theorem archie_record :
  let richard_total := richard_average * games_played + needed_average * remaining_games
  richard_total - 1 = 89 := by sorry

end NUMINAMATH_CALUDE_archie_record_l3723_372369


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3723_372304

theorem fixed_point_of_exponential_function 
  (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 1
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3723_372304


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l3723_372331

/-- Given two points A(-2, y₁) and B(-1, y₂) on the inverse proportion function y = 2/x,
    prove that y₁ > y₂ -/
theorem inverse_proportion_comparison :
  ∀ y₁ y₂ : ℝ,
  y₁ = 2 / (-2) →
  y₂ = 2 / (-1) →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l3723_372331


namespace NUMINAMATH_CALUDE_two_digit_product_sum_l3723_372380

theorem two_digit_product_sum (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8060 → 
  a + b = 127 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_sum_l3723_372380


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3723_372391

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x - 3 > 0) ↔ (x > 3/2 ∨ x < -1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3723_372391


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l3723_372373

theorem cube_painting_theorem (n : ℕ) (h : n > 4) :
  (6 * (n - 4)^2 : ℕ) = (n - 4)^3 ↔ n = 10 := by sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l3723_372373


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l3723_372386

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l3723_372386


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3723_372395

-- Define an isosceles triangle with side lengths 3 and 5
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 5 ∧ (a = c ∨ b = c)) ∨ (a = 5 ∧ b = 3 ∧ (a = c ∨ b = c))

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → (Perimeter a b c = 11 ∨ Perimeter a b c = 13) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3723_372395


namespace NUMINAMATH_CALUDE_min_intercept_sum_l3723_372319

/-- Given a line passing through (1, 2) with equation x/a + y/b = 1 where a > 0 and b > 0,
    the minimum value of a + b is 3 + 2√2 -/
theorem min_intercept_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 1 / a + 2 / b = 1) : 
  ∀ (x y : ℝ), x / a + y / b = 1 → x + y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_intercept_sum_l3723_372319


namespace NUMINAMATH_CALUDE_yellow_surface_fraction_proof_l3723_372338

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  yellow_cubes : ℕ
  blue_cubes : ℕ

/-- Calculates the fraction of yellow surface area -/
def yellow_surface_fraction (cube : LargeCube) : ℚ :=
  sorry

theorem yellow_surface_fraction_proof (cube : LargeCube) :
  cube.edge_length = 4 →
  cube.small_cubes = 64 →
  cube.yellow_cubes = 15 →
  cube.blue_cubes = 49 →
  yellow_surface_fraction cube = 1/6 :=
sorry

end NUMINAMATH_CALUDE_yellow_surface_fraction_proof_l3723_372338


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3723_372336

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + 1 / a^2)
  1 < e ∧ e < Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3723_372336


namespace NUMINAMATH_CALUDE_parabola_properties_l3723_372303

/-- Parabola passing through (-1, 0) -/
def parabola (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x - 3

theorem parabola_properties :
  ∃ (b : ℝ),
    (parabola b (-1) = 0) ∧
    (b = -4) ∧
    (∃ (h k : ℝ), h = -2 ∧ k = 1 ∧ ∀ x, parabola b x = -(x - h)^2 + k) ∧
    (∀ y₁ y₂ : ℝ, parabola b 1 = y₁ → parabola b (-1) = y₂ → y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3723_372303


namespace NUMINAMATH_CALUDE_harry_terry_calculation_harry_terry_calculation_proof_l3723_372346

theorem harry_terry_calculation : ℤ → ℤ → Prop :=
  fun (H T : ℤ) =>
    H = 8 - (2 + 5) →
    T = 8 - 2 + 5 →
    H - T = -10

-- Proof
theorem harry_terry_calculation_proof : harry_terry_calculation 1 11 := by
  sorry

end NUMINAMATH_CALUDE_harry_terry_calculation_harry_terry_calculation_proof_l3723_372346


namespace NUMINAMATH_CALUDE_mixed_solution_purity_l3723_372342

/-- Calculates the purity of a mixed solution given two initial solutions with different purities -/
theorem mixed_solution_purity
  (purity1 purity2 : ℚ)
  (volume1 volume2 : ℚ)
  (h1 : purity1 = 30 / 100)
  (h2 : purity2 = 60 / 100)
  (h3 : volume1 = 40)
  (h4 : volume2 = 20)
  (h5 : volume1 + volume2 = 60) :
  (purity1 * volume1 + purity2 * volume2) / (volume1 + volume2) = 40 / 100 := by
  sorry

#check mixed_solution_purity

end NUMINAMATH_CALUDE_mixed_solution_purity_l3723_372342


namespace NUMINAMATH_CALUDE_min_selling_price_theorem_l3723_372396

/-- Represents the fruit shop scenario with two batches of fruits. -/
structure FruitShop where
  batch1_price : ℝ  -- Price per kg of first batch
  batch1_quantity : ℝ  -- Quantity of first batch in kg
  batch2_price : ℝ  -- Price per kg of second batch
  batch2_quantity : ℝ  -- Quantity of second batch in kg

/-- Calculates the minimum selling price for remaining fruits to achieve the target profit. -/
def min_selling_price (shop : FruitShop) (target_profit : ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum selling price for the given scenario. -/
theorem min_selling_price_theorem (shop : FruitShop) :
  shop.batch1_price = 50 ∧
  shop.batch2_price = 55 ∧
  shop.batch1_quantity * shop.batch1_price = 1100 ∧
  shop.batch2_quantity * shop.batch2_price = 1100 ∧
  shop.batch1_quantity = shop.batch2_quantity + 2 ∧
  shop.batch2_price = 1.1 * shop.batch1_price →
  min_selling_price shop 1000 = 60 :=
by sorry

end NUMINAMATH_CALUDE_min_selling_price_theorem_l3723_372396


namespace NUMINAMATH_CALUDE_train_passing_platform_l3723_372362

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (time_to_cross_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_to_cross_point = 120)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / time_to_cross_point) = 190 :=
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l3723_372362


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3723_372389

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 21) : 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3723_372389


namespace NUMINAMATH_CALUDE_sum_of_possible_DE_values_l3723_372370

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A function that constructs the number D465E32 from digits D and E -/
def constructNumber (D E : ℕ) : ℕ := D * 100000 + 465000 + E * 100 + 32

/-- The theorem stating the sum of all possible values of D+E is 24 -/
theorem sum_of_possible_DE_values : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, ∃ D E : ℕ, isSingleDigit D ∧ isSingleDigit E ∧ 
      7 ∣ constructNumber D E ∧ n = D + E) ∧
    (∀ D E : ℕ, isSingleDigit D → isSingleDigit E → 
      7 ∣ constructNumber D E → D + E ∈ S) ∧
    S.sum id = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_DE_values_l3723_372370


namespace NUMINAMATH_CALUDE_smallest_x_for_digit_sum_50_l3723_372378

def sequence_sum (x : ℕ) : ℕ := 100 * x + 4950

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem smallest_x_for_digit_sum_50 :
  ∀ x : ℕ, x < 99950 → digit_sum (sequence_sum x) ≠ 50 ∧
  digit_sum (sequence_sum 99950) = 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_digit_sum_50_l3723_372378


namespace NUMINAMATH_CALUDE_group_size_is_factor_l3723_372347

def num_cows : ℕ := 24
def num_sheep : ℕ := 7
def num_goats : ℕ := 113

def total_animals : ℕ := num_cows + num_sheep + num_goats

theorem group_size_is_factor :
  ∀ (group_size : ℕ), 
    group_size > 1 →
    group_size < total_animals →
    (total_animals % group_size = 0) →
    ∃ (num_groups : ℕ), num_groups > 1 ∧ num_groups * group_size = total_animals :=
by sorry

end NUMINAMATH_CALUDE_group_size_is_factor_l3723_372347


namespace NUMINAMATH_CALUDE_championship_assignments_l3723_372327

theorem championship_assignments (n_students : ℕ) (n_titles : ℕ) :
  n_students = 4 → n_titles = 3 →
  (n_students ^ n_titles : ℕ) = 64 := by
  sorry

end NUMINAMATH_CALUDE_championship_assignments_l3723_372327


namespace NUMINAMATH_CALUDE_cookies_left_l3723_372381

def cookies_per_tray : ℕ := 12

def daily_trays : List ℕ := [2, 3, 4, 5, 3, 4, 4]

def frank_daily_consumption : ℕ := 2

def ted_consumption : List (ℕ × ℕ) := [(2, 3), (4, 5)]

def jan_consumption : ℕ × ℕ := (3, 5)

def tom_consumption : ℕ × ℕ := (5, 8)

def neighbours_consumption : ℕ × ℕ := (6, 20)

def total_baked (trays : List ℕ) (cookies_per_tray : ℕ) : ℕ :=
  (trays.map (· * cookies_per_tray)).sum

def total_eaten (frank_daily : ℕ) (ted : List (ℕ × ℕ)) (jan : ℕ × ℕ) (tom : ℕ × ℕ) (neighbours : ℕ × ℕ) : ℕ :=
  7 * frank_daily + (ted.map Prod.snd).sum + jan.snd + tom.snd + neighbours.snd

theorem cookies_left : 
  total_baked daily_trays cookies_per_tray - 
  total_eaten frank_daily_consumption ted_consumption jan_consumption tom_consumption neighbours_consumption = 245 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3723_372381


namespace NUMINAMATH_CALUDE_bruce_mangoes_purchase_l3723_372333

theorem bruce_mangoes_purchase :
  let grapes_kg : ℕ := 8
  let grapes_price : ℕ := 70
  let mango_price : ℕ := 55
  let total_paid : ℕ := 1165
  let mango_kg : ℕ := (total_paid - grapes_kg * grapes_price) / mango_price
  mango_kg = 11 := by sorry

end NUMINAMATH_CALUDE_bruce_mangoes_purchase_l3723_372333


namespace NUMINAMATH_CALUDE_segment_ratio_in_quadrilateral_l3723_372360

/-- Given four distinct points on a plane with segment lengths a, a, a, b, b, and c,
    prove that the ratio of c to a is √3/2 -/
theorem segment_ratio_in_quadrilateral (a b c : ℝ) :
  (∃ A B C D : ℝ × ℝ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    dist A B = a ∧ dist A C = a ∧ dist B C = a ∧
    dist A D = b ∧ dist B D = b ∧ dist C D = c) →
  c / a = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_segment_ratio_in_quadrilateral_l3723_372360


namespace NUMINAMATH_CALUDE_restaurant_combinations_l3723_372308

theorem restaurant_combinations (menu_items : ℕ) (special_dish : ℕ) : menu_items = 12 ∧ special_dish = 1 →
  (menu_items - special_dish) * (menu_items - special_dish) + 
  2 * special_dish * (menu_items - special_dish) = 143 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_combinations_l3723_372308


namespace NUMINAMATH_CALUDE_problem_solution_l3723_372352

theorem problem_solution (x y : ℚ) : 
  x / y = 15 / 3 → y = 27 → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3723_372352


namespace NUMINAMATH_CALUDE_film_festival_selection_l3723_372329

/-- Given a film festival selection process, prove that the fraction of color films
    selected by the subcommittee is 20/21. -/
theorem film_festival_selection (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let total_films := 30 * x + 6 * y
  let bw_selected := (y / x) * (30 * x) / 100
  let color_selected := 6 * y
  let total_selected := bw_selected + color_selected
  color_selected / total_selected = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_film_festival_selection_l3723_372329


namespace NUMINAMATH_CALUDE_simplify_fraction_l3723_372340

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  (18 * x^3 * y^2) / (9 * x^2 * y^4) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3723_372340


namespace NUMINAMATH_CALUDE_rahul_share_l3723_372351

/-- Calculates the share of a worker given the total payment and the time taken by both workers --/
def calculateShare (totalPayment : ℚ) (time1 : ℚ) (time2 : ℚ) : ℚ :=
  let combinedRate := 1 / time1 + 1 / time2
  let share := (1 / time1) / combinedRate
  share * totalPayment

/-- Proves that Rahul's share is $60 given the problem conditions --/
theorem rahul_share :
  let rahulTime : ℚ := 3
  let rajeshTime : ℚ := 2
  let totalPayment : ℚ := 150
  calculateShare totalPayment rahulTime rajeshTime = 60 := by
sorry

#eval calculateShare 150 3 2

end NUMINAMATH_CALUDE_rahul_share_l3723_372351


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l3723_372330

theorem cricket_team_captain_age (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) 
  (team_average : ℚ) (remaining_average : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  team_average = 22 →
  remaining_average = team_average - 1 →
  (team_size : ℚ) * team_average = 
    captain_age + wicket_keeper_age + (team_size - 2 : ℚ) * remaining_average →
  captain_age = 25 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l3723_372330


namespace NUMINAMATH_CALUDE_only_minute_hand_rotates_l3723_372374

-- Define the set of objects
inductive Object
  | MinuteHand
  | Boat
  | Car

-- Define the motion types
inductive Motion
  | Rotation
  | Translation
  | Combined

-- Function to determine the motion type of an object
def motionType (obj : Object) : Motion :=
  match obj with
  | Object.MinuteHand => Motion.Rotation
  | Object.Boat => Motion.Combined
  | Object.Car => Motion.Combined

-- Theorem statement
theorem only_minute_hand_rotates :
  ∀ (obj : Object), motionType obj = Motion.Rotation ↔ obj = Object.MinuteHand :=
by sorry

end NUMINAMATH_CALUDE_only_minute_hand_rotates_l3723_372374


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l3723_372316

/-- The equation 4(3x-b) = 3(4x+16) has infinitely many solutions x if and only if b = -12 -/
theorem infinite_solutions_iff_b_eq_neg_twelve :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l3723_372316


namespace NUMINAMATH_CALUDE_complex_exp_conversion_l3723_372372

theorem complex_exp_conversion : Complex.exp (13 * π * Complex.I / 4) * (Real.sqrt 2) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_conversion_l3723_372372


namespace NUMINAMATH_CALUDE_general_term_formula_l3723_372363

def S (n : ℕ) : ℤ := 3 * n^2 - 2 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2 else 6 * n - 5

theorem general_term_formula (n : ℕ) :
  (n = 1 ∧ a n = S n) ∨
  (n ≥ 2 ∧ a n = S n - S (n-1)) :=
sorry

end NUMINAMATH_CALUDE_general_term_formula_l3723_372363


namespace NUMINAMATH_CALUDE_prism_volume_l3723_372355

/-- The volume of a right rectangular prism with face areas 30, 40, and 60 is 120√5 -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 40) (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3723_372355


namespace NUMINAMATH_CALUDE_wall_length_is_800cm_l3723_372322

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

/-- Theorem stating that the length of the wall is 800 cm -/
theorem wall_length_is_800cm (brick : BrickDimensions)
    (wall : WallDimensions) (num_bricks : ℕ) :
    brick.length = 50 →
    brick.width = 11.25 →
    brick.height = 6 →
    wall.width = 600 →
    wall.height = 22.5 →
    num_bricks = 3200 →
    brickVolume brick * num_bricks = wallVolume wall →
    wall.length = 800 := by
  sorry


end NUMINAMATH_CALUDE_wall_length_is_800cm_l3723_372322


namespace NUMINAMATH_CALUDE_andrew_jeffrey_walk_l3723_372367

/-- Calculates the number of steps Andrew walks given Jeffrey's steps and their step ratio -/
def andrews_steps (jeffreys_steps : ℕ) (andrew_ratio jeffrey_ratio : ℕ) : ℕ :=
  (andrew_ratio * jeffreys_steps) / jeffrey_ratio

/-- Theorem stating that if Jeffrey walks 200 steps and the ratio of Andrew's to Jeffrey's steps is 3:4, then Andrew walks 150 steps -/
theorem andrew_jeffrey_walk :
  andrews_steps 200 3 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_andrew_jeffrey_walk_l3723_372367


namespace NUMINAMATH_CALUDE_line_intersection_with_x_axis_l3723_372306

/-- Given a line y = kx + b parallel to y = -3x + 1 and passing through (0, -2),
    prove that its intersection with the x-axis is at (-2/3, 0) -/
theorem line_intersection_with_x_axis
  (k b : ℝ) 
  (parallel : k = -3)
  (passes_through : b = -2) :
  let line := λ x : ℝ => k * x + b
  ∃ x : ℝ, line x = 0 ∧ x = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_with_x_axis_l3723_372306


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l3723_372357

theorem largest_x_sqrt_3x_eq_5x :
  ∃ (x_max : ℚ), x_max = 3/25 ∧
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3 * x) = 5 * x → x ≤ x_max) ∧
  Real.sqrt (3 * x_max) = 5 * x_max := by
  sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l3723_372357


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3723_372382

theorem expand_and_simplify (x : ℝ) : (2 * x - 3) * (4 * x + 5) = 8 * x^2 - 2 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3723_372382
