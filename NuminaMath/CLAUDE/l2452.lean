import Mathlib

namespace NUMINAMATH_CALUDE_valid_pairs_count_l2452_245284

/-- A function that checks if a positive integer has a zero digit. -/
def has_zero_digit (n : ℕ+) : Prop := sorry

/-- The count of ordered pairs (a,b) of positive integers where a + b = 500 and neither a nor b has a zero digit. -/
def count_valid_pairs : ℕ := sorry

/-- Theorem stating that the count of valid pairs is 329. -/
theorem valid_pairs_count : count_valid_pairs = 329 := by sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l2452_245284


namespace NUMINAMATH_CALUDE_first_day_over_500_l2452_245290

def marbles (k : ℕ) : ℕ := 5 * 3^k

theorem first_day_over_500 : (∃ k : ℕ, marbles k > 500) ∧ 
  (∀ j : ℕ, j < 5 → marbles j ≤ 500) ∧ 
  marbles 5 > 500 := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_l2452_245290


namespace NUMINAMATH_CALUDE_negation_of_implication_conjunction_implies_disjunction_disjunction_not_implies_conjunction_negation_of_universal_even_function_condition_l2452_245258

-- Define the propositions p and q
variable (p q : Prop)

-- Define the function f
variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- 1. Negation of implication
theorem negation_of_implication : ¬(p → q) ↔ (p ∧ ¬q) := by sorry

-- 2. Relationship between conjunction and disjunction
theorem conjunction_implies_disjunction : (p ∧ q) → (p ∨ q) := by sorry

theorem disjunction_not_implies_conjunction : ¬((p ∨ q) → (p ∧ q)) := by sorry

-- 3. Negation of universal quantifier
theorem negation_of_universal : 
  ¬(∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 2*x ≤ 0) := by sorry

-- 4. Even function condition
theorem even_function_condition : 
  (∀ x : ℝ, f x = f (-x)) → b = 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_conjunction_implies_disjunction_disjunction_not_implies_conjunction_negation_of_universal_even_function_condition_l2452_245258


namespace NUMINAMATH_CALUDE_cylindrical_cans_radius_l2452_245278

theorem cylindrical_cans_radius (h : ℝ) (h_pos : h > 0) :
  let r₁ : ℝ := 15 -- radius of the second can
  let h₁ : ℝ := h -- height of the second can
  let h₂ : ℝ := (4 * h^2) / 3 -- height of the first can
  let v₁ : ℝ := π * r₁^2 * h₁ -- volume of the second can
  let r₂ : ℝ := (15 * Real.sqrt 3) / 2 -- radius of the first can
  v₁ = π * r₂^2 * h₂ -- volumes are equal
  := by sorry

end NUMINAMATH_CALUDE_cylindrical_cans_radius_l2452_245278


namespace NUMINAMATH_CALUDE_sufficient_condition_for_not_p_l2452_245265

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 4, log_half x < 2*x + a

-- Theorem statement
theorem sufficient_condition_for_not_p (a : ℝ) :
  a < -11 → ∀ x ∈ Set.Icc 1 4, log_half x ≥ 2*x + a :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_not_p_l2452_245265


namespace NUMINAMATH_CALUDE_ellipse_equation_l2452_245272

/-- Given an ellipse and a line passing through its upper vertex and right focus,
    prove that the equation of the ellipse is x^2/5 + y^2/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c < a ∧ a^2 = b^2 + c^2 ∧
   2*0 + b - 2 = 0 ∧ 2*c + 0 - 2 = 0) →
  ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/5 + y^2/4 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_equation_l2452_245272


namespace NUMINAMATH_CALUDE_remaining_laps_is_58_l2452_245275

/-- Represents the number of laps swum on each day --/
structure DailyLaps where
  friday : Nat
  saturday : Nat
  sundayMorning : Nat

/-- Calculates the remaining laps after Sunday morning --/
def remainingLaps (totalRequired : Nat) (daily : DailyLaps) : Nat :=
  totalRequired - (daily.friday + daily.saturday + daily.sundayMorning)

/-- Theorem stating that the remaining laps after Sunday morning is 58 --/
theorem remaining_laps_is_58 (totalRequired : Nat) (daily : DailyLaps) :
  totalRequired = 198 →
  daily.friday = 63 →
  daily.saturday = 62 →
  daily.sundayMorning = 15 →
  remainingLaps totalRequired daily = 58 := by
  sorry

#eval remainingLaps 198 { friday := 63, saturday := 62, sundayMorning := 15 }

end NUMINAMATH_CALUDE_remaining_laps_is_58_l2452_245275


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_problem_l2452_245269

theorem lcm_gcd_sum_problem (a b : ℕ) (ha : a = 12) (hb : b = 20) :
  (Nat.lcm a b * Nat.gcd a b) + (a + b) = 272 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_problem_l2452_245269


namespace NUMINAMATH_CALUDE_volumes_and_cross_sections_l2452_245231

/-- Represents a geometric body -/
structure GeometricBody where
  volume : ℝ
  crossSectionArea : ℝ → ℝ  -- Function mapping height to cross-sectional area

/-- Zu Chongzhi's principle -/
axiom zu_chongzhi_principle (A B : GeometricBody) :
  (∀ h : ℝ, A.crossSectionArea h = B.crossSectionArea h) → A.volume = B.volume

/-- The main theorem to prove -/
theorem volumes_and_cross_sections (A B : GeometricBody) :
  (A.volume ≠ B.volume → ∃ h : ℝ, A.crossSectionArea h ≠ B.crossSectionArea h) ∧
  ∃ C D : GeometricBody, C.volume = D.volume ∧ ∃ h : ℝ, C.crossSectionArea h ≠ D.crossSectionArea h :=
sorry

end NUMINAMATH_CALUDE_volumes_and_cross_sections_l2452_245231


namespace NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l2452_245219

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus_of_parabola (y : ℝ) :
  y^2 = 8 →  -- Point (1, y) satisfies the parabola equation
  Real.sqrt ((1 - 2)^2 + y^2) = 3 :=  -- Distance from (1, y) to focus (2, 0) is 3
by sorry

end NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l2452_245219


namespace NUMINAMATH_CALUDE_sum_zero_from_absolute_value_inequalities_l2452_245209

theorem sum_zero_from_absolute_value_inequalities (a b c : ℝ) 
  (h1 : |a| ≥ |b+c|) 
  (h2 : |b| ≥ |c+a|) 
  (h3 : |c| ≥ |a+b|) : 
  a + b + c = 0 := by 
sorry

end NUMINAMATH_CALUDE_sum_zero_from_absolute_value_inequalities_l2452_245209


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2452_245267

universe u

def U : Finset ℕ := {1, 2, 3, 4}
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2452_245267


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2452_245253

theorem cone_lateral_surface_area 
  (r : ℝ) (h : ℝ) (l : ℝ) (S : ℝ) 
  (h_r : r = 2) 
  (h_h : h = 4 * Real.sqrt 2) 
  (h_l : l^2 = r^2 + h^2) 
  (h_S : S = π * r * l) : 
  S = 12 * π := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2452_245253


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l2452_245212

/-- Given an equation (x² + cx + d) / (2x - e) = (n - 2) / (n + 2),
    if the roots are numerically equal but have opposite signs,
    then n = (-4 - 2c) / (c - 2) -/
theorem roots_opposite_signs (c d e : ℝ) :
  let f (x : ℝ) := (x^2 + c*x + d) / (2*x - e)
  ∃ (n : ℝ), (∀ x, f x = (n - 2) / (n + 2)) →
  (∃ (r : ℝ), (f r = (n - 2) / (n + 2) ∧ f (-r) = (n - 2) / (n + 2))) →
  n = (-4 - 2*c) / (c - 2) := by
sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l2452_245212


namespace NUMINAMATH_CALUDE_intersection_points_equality_l2452_245235

/-- Theorem: For a quadratic function y = ax^2 and two parallel lines intersecting
    this function, the difference between the x-coordinates of the intersection points
    satisfies (x3 - x1) = (x2 - x4). -/
theorem intersection_points_equality 
  (a : ℝ) 
  (x1 x2 x3 x4 : ℝ) 
  (h1 : x1 < x2) 
  (h2 : x3 < x4) 
  (h_parallel : ∃ (k b c : ℝ), 
    a * x1^2 = k * x1 + b ∧ 
    a * x2^2 = k * x2 + b ∧ 
    a * x3^2 = k * x3 + c ∧ 
    a * x4^2 = k * x4 + c) :
  x3 - x1 = x2 - x4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_equality_l2452_245235


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l2452_245249

/-- Given two cubes of the same material, if one cube has sides twice as long as the other,
    and the smaller cube weighs 4 pounds, then the larger cube weighs 32 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight : ℝ → ℝ) (volume : ℝ → ℝ) :
  (∀ x, weight x = (weight s / volume s) * volume x) →  -- weight is proportional to volume
  volume s = s^3 →  -- volume of a cube is side length cubed
  weight s = 4 →  -- weight of original cube is 4 pounds
  weight (2*s) = 32 :=  -- weight of new cube with doubled side length
by
  sorry


end NUMINAMATH_CALUDE_weight_of_doubled_cube_l2452_245249


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l2452_245211

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n = 99 ∧ 9 ∣ n ∧ n < 100 ∧ ∀ (m : ℕ), 9 ∣ m → m < 100 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l2452_245211


namespace NUMINAMATH_CALUDE_ball_bounce_count_l2452_245225

/-- The number of bounces required for a ball to reach a height less than 2 feet -/
theorem ball_bounce_count (initial_height : ℝ) (bounce_ratio : ℝ) (target_height : ℝ) :
  initial_height = 20 →
  bounce_ratio = 2/3 →
  target_height = 2 →
  (∀ k : ℕ, k < 6 → initial_height * bounce_ratio^k ≥ target_height) ∧
  initial_height * bounce_ratio^6 < target_height :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_count_l2452_245225


namespace NUMINAMATH_CALUDE_hammer_wrench_ratio_l2452_245201

/-- Given that the weight of a wrench is twice the weight of a hammer,
    prove that the ratio of (weight of 2 hammers + 2 wrenches) to
    (weight of 8 hammers + 5 wrenches) is 1/3. -/
theorem hammer_wrench_ratio (h w : ℝ) (hw : w = 2 * h) :
  (2 * h + 2 * w) / (8 * h + 5 * w) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hammer_wrench_ratio_l2452_245201


namespace NUMINAMATH_CALUDE_minimum_score_proof_l2452_245242

theorem minimum_score_proof (C ω : ℕ) (S : ℝ) : 
  S = 30 + 4 * C - ω →
  S > 80 →
  C + ω = 26 →
  ω ≤ 3 →
  ∀ (C' ω' : ℕ) (S' : ℝ), 
    (S' = 30 + 4 * C' - ω' ∧ 
     S' > 80 ∧ 
     C' + ω' = 26 ∧ 
     ω' ≤ 3) → 
    S ≤ S' →
  S = 119 :=
sorry

end NUMINAMATH_CALUDE_minimum_score_proof_l2452_245242


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l2452_245234

theorem min_value_cyclic_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / b + b / c + c / a ≥ 3 ∧ 
  (a / b + b / c + c / a = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l2452_245234


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l2452_245282

/-- Proves that the total number of tickets sold is 45 given the specified conditions --/
theorem concert_ticket_sales : 
  let ticket_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_group_discount : ℚ := 40 / 100
  let second_group_discount : ℚ := 15 / 100
  let total_revenue : ℕ := 760
  ∃ (full_price_tickets : ℕ),
    (first_group_size * (ticket_price * (1 - first_group_discount)).floor + 
     second_group_size * (ticket_price * (1 - second_group_discount)).floor + 
     full_price_tickets * ticket_price = total_revenue) ∧
    (first_group_size + second_group_size + full_price_tickets = 45) :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l2452_245282


namespace NUMINAMATH_CALUDE_distance_between_given_lines_l2452_245283

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 3 = 0

def are_parallel (l1 l2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), l1 x y ↔ l2 (k * x) (k * y)

def distance_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem distance_between_given_lines :
  are_parallel line1 line2 →
  distance_between_lines line1 line2 = 1.5 := by sorry

end NUMINAMATH_CALUDE_distance_between_given_lines_l2452_245283


namespace NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l2452_245241

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Defines a similarity transformation with a given ratio -/
def similarityTransform (p : Point) (ratio : ℝ) : Set Point :=
  { p' : Point | p'.x = p.x * ratio ∨ p'.x = p.x * (-ratio) ∧ 
                  p'.y = p.y * ratio ∨ p'.y = p.y * (-ratio) }

theorem similar_triangle_point_coordinates 
  (ABC : Triangle) 
  (C : Point) 
  (h1 : C = ABC.C) 
  (h2 : C.x = 4 ∧ C.y = 1) 
  (ratio : ℝ) 
  (h3 : ratio = 3) :
  ∃ (C' : Point), C' ∈ similarityTransform C ratio ∧ 
    ((C'.x = 12 ∧ C'.y = 3) ∨ (C'.x = -12 ∧ C'.y = -3)) :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l2452_245241


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l2452_245200

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem: The minimum number of 10 cubic cm cubes required to build a box
    with dimensions 8 cm x 15 cm x 5 cm is 60 -/
theorem min_cubes_for_box : min_cubes 8 15 5 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l2452_245200


namespace NUMINAMATH_CALUDE_park_entrance_cost_is_5_l2452_245230

def park_entrance_cost : ℝ → Prop :=
  λ cost =>
    let num_children := 4
    let num_parents := 2
    let num_grandmother := 1
    let attraction_cost_kid := 2
    let attraction_cost_adult := 4
    let total_paid := 55
    let total_family_members := num_children + num_parents + num_grandmother
    let total_attraction_cost := num_children * attraction_cost_kid + 
                                 (num_parents + num_grandmother) * attraction_cost_adult
    total_paid = total_family_members * cost + total_attraction_cost

theorem park_entrance_cost_is_5 : park_entrance_cost 5 := by
  sorry

end NUMINAMATH_CALUDE_park_entrance_cost_is_5_l2452_245230


namespace NUMINAMATH_CALUDE_second_purchase_profit_less_than_first_l2452_245222

/-- Represents a type of T-shirt -/
structure TShirt where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the store's inventory and sales -/
structure Store where
  typeA : TShirt
  typeB : TShirt
  firstPurchaseQuantityA : ℕ
  firstPurchaseQuantityB : ℕ
  secondPurchaseQuantityA : ℕ
  secondPurchaseQuantityB : ℕ

/-- Calculate the profit from the first purchase -/
def firstPurchaseProfit (s : Store) : ℕ :=
  (s.typeA.sellingPrice - s.typeA.purchasePrice) * s.firstPurchaseQuantityA +
  (s.typeB.sellingPrice - s.typeB.purchasePrice) * s.firstPurchaseQuantityB

/-- Calculate the maximum profit from the second purchase -/
def maxSecondPurchaseProfit (s : Store) : ℕ :=
  let newTypeA := TShirt.mk (s.typeA.purchasePrice + 5) s.typeA.sellingPrice
  let newTypeB := TShirt.mk (s.typeB.purchasePrice + 10) s.typeB.sellingPrice
  (newTypeA.sellingPrice - newTypeA.purchasePrice) * s.secondPurchaseQuantityA +
  (newTypeB.sellingPrice - newTypeB.purchasePrice) * s.secondPurchaseQuantityB

/-- The theorem to be proved -/
theorem second_purchase_profit_less_than_first (s : Store) :
  s.firstPurchaseQuantityA + s.firstPurchaseQuantityB = 120 →
  s.typeA.purchasePrice * s.firstPurchaseQuantityA + s.typeB.purchasePrice * s.firstPurchaseQuantityB = 6000 →
  s.secondPurchaseQuantityA + s.secondPurchaseQuantityB = 150 →
  s.secondPurchaseQuantityB ≤ 2 * s.secondPurchaseQuantityA →
  maxSecondPurchaseProfit s < firstPurchaseProfit s :=
by
  sorry

#check second_purchase_profit_less_than_first

end NUMINAMATH_CALUDE_second_purchase_profit_less_than_first_l2452_245222


namespace NUMINAMATH_CALUDE_expand_product_l2452_245236

theorem expand_product (x : ℝ) : (x + 5) * (x - 4^2) = x^2 - 11*x - 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2452_245236


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_b_equal_three_l2452_245227

/-- Given two functions f and g with a common tangent at (1, c), prove a = b = 3 -/
theorem common_tangent_implies_a_b_equal_three
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h_f : ∀ x, f x = a * x^2 + 1)
  (h_a_pos : a > 0)
  (h_g : ∀ x, g x = x^3 + b * x)
  (h_intersection : f 1 = g 1)
  (h_common_tangent : (deriv f) 1 = (deriv g) 1) :
  a = 3 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_implies_a_b_equal_three_l2452_245227


namespace NUMINAMATH_CALUDE_david_cindy_walk_difference_l2452_245294

theorem david_cindy_walk_difference (AC CB : ℝ) (h1 : AC = 8) (h2 : CB = 15) :
  let AB : ℝ := Real.sqrt (AC^2 + CB^2)
  AC + CB - AB = 6 := by sorry

end NUMINAMATH_CALUDE_david_cindy_walk_difference_l2452_245294


namespace NUMINAMATH_CALUDE_triangle_properties_l2452_245223

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  -- a and b are roots of x^2 - 2√3x + 2 = 0
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  -- 2cos(A+B) = 1
  2 * Real.cos (t.A + t.B) = 1

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = 2 * π / 3 ∧  -- 120° in radians
  t.c = Real.sqrt 10 ∧
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2452_245223


namespace NUMINAMATH_CALUDE_abc_inequality_l2452_245210

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define a, b, and c
noncomputable def a : ℝ := log (3/4) (log 3 4)
noncomputable def b : ℝ := (3/4) ^ (1/2 : ℝ)
noncomputable def c : ℝ := (4/3) ^ (1/2 : ℝ)

-- Theorem statement
theorem abc_inequality : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2452_245210


namespace NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_l2452_245224

theorem rational_fraction_implies_integer_sum_squares (a b c : ℕ+) 
  (h : ∃ (q : ℚ), (a.val : ℝ) * Real.sqrt 3 + b.val = q * ((b.val : ℝ) * Real.sqrt 3 + c.val)) :
  ∃ (n : ℤ), (a.val^2 + b.val^2 + c.val^2 : ℝ) / (a.val + b.val + c.val) = n := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_l2452_245224


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2452_245291

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 50000 →
  (5 * (n - 3)^5 - 3 * n^2 + 20 * n - 35) % 7 = 0 →
  n ≤ 49999 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2452_245291


namespace NUMINAMATH_CALUDE_third_part_time_l2452_245292

/-- Represents the time taken for each part of the assignment -/
def timeTaken (k : ℕ) : ℕ := 25 * k

/-- The total time available for the assignment in minutes -/
def totalTimeAvailable : ℕ := 150

/-- The time taken for the first break -/
def firstBreak : ℕ := 10

/-- The time taken for the second break -/
def secondBreak : ℕ := 15

/-- Theorem stating that the time taken for the third part is 50 minutes -/
theorem third_part_time : 
  totalTimeAvailable - (timeTaken 1 + firstBreak + timeTaken 2 + secondBreak) = 50 := by
  sorry


end NUMINAMATH_CALUDE_third_part_time_l2452_245292


namespace NUMINAMATH_CALUDE_petyas_calculation_error_l2452_245270

theorem petyas_calculation_error :
  ¬∃ (a : ℕ), a > 3 ∧ 
  ∃ (n : ℕ), ((a - 3) * (a + 4) - a = n) ∧ 
  (∃ (digits : List ℕ), 
    digits.length = 6069 ∧
    digits.count 8 = 2023 ∧
    digits.count 0 = 2023 ∧
    digits.count 3 = 2023 ∧
    (∀ d, d ∈ digits → d ∈ [8, 0, 3]) ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0) :=
sorry

end NUMINAMATH_CALUDE_petyas_calculation_error_l2452_245270


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2452_245273

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 2 → x ≠ 0 → (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2452_245273


namespace NUMINAMATH_CALUDE_age_difference_l2452_245233

/-- Given four people A, B, C, and D with ages a, b, c, and d respectively,
    prove that C is 14 years younger than A under the given conditions. -/
theorem age_difference (a b c d : ℕ) : 
  (a + b = b + c + 14) →
  (b + d = c + a + 10) →
  (d = c + 6) →
  (a = c + 14) := by sorry

end NUMINAMATH_CALUDE_age_difference_l2452_245233


namespace NUMINAMATH_CALUDE_not_or_implies_both_false_l2452_245207

theorem not_or_implies_both_false (p q : Prop) : 
  ¬(p ∨ q) → ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_not_or_implies_both_false_l2452_245207


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2452_245281

theorem no_positive_integer_solutions : 
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 8 * x^2 * y^2 + 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2452_245281


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2452_245247

theorem quadratic_roots_range (θ : Real) (α β : Complex) : 
  (∃ x : Complex, x^2 + 2*(Real.cos θ + 1)*x + (Real.cos θ)^2 = 0 ↔ x = α ∨ x = β) →
  Complex.abs (α - β) ≤ 2 * Real.sqrt 2 →
  ∃ k : ℤ, (θ ∈ Set.Icc (2*k*Real.pi + Real.pi/3) (2*k*Real.pi + 2*Real.pi/3)) ∨
           (θ ∈ Set.Icc (2*k*Real.pi + 4*Real.pi/3) (2*k*Real.pi + 5*Real.pi/3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2452_245247


namespace NUMINAMATH_CALUDE_basketball_team_selection_l2452_245251

theorem basketball_team_selection (total_students : Nat) 
  (total_girls total_boys : Nat)
  (junior_girls senior_girls : Nat)
  (junior_boys senior_boys : Nat)
  (callback_junior_girls callback_senior_girls : Nat)
  (callback_junior_boys callback_senior_boys : Nat) :
  total_students = 56 →
  total_girls = 33 →
  total_boys = 23 →
  junior_girls = 15 →
  senior_girls = 18 →
  junior_boys = 12 →
  senior_boys = 11 →
  callback_junior_girls = 8 →
  callback_senior_girls = 9 →
  callback_junior_boys = 5 →
  callback_senior_boys = 6 →
  total_students - (callback_junior_girls + callback_senior_girls + callback_junior_boys + callback_senior_boys) = 28 := by
  sorry

#check basketball_team_selection

end NUMINAMATH_CALUDE_basketball_team_selection_l2452_245251


namespace NUMINAMATH_CALUDE_right_triangle_set_l2452_245208

theorem right_triangle_set : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = Real.sqrt 2 ∧ b = 5 ∧ c = 2 * Real.sqrt 7) ∨
   (a = 6 ∧ b = 9 ∧ c = 15) ∨
   (a = 4 ∧ b = 12 ∧ c = 13)) ∧
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2452_245208


namespace NUMINAMATH_CALUDE_constant_speed_running_time_l2452_245287

/-- Given a constant running speed, if it takes 30 minutes to run 5 miles,
    then it will take 18 minutes to run 3 miles. -/
theorem constant_speed_running_time
  (speed : ℝ)
  (h1 : speed > 0)
  (h2 : 5 / speed = 30) :
  3 / speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_constant_speed_running_time_l2452_245287


namespace NUMINAMATH_CALUDE_derivative_sum_at_one_l2452_245246

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem derivative_sum_at_one 
  (h1 : ∀ x, f x + x * g x = x^2 - 1) 
  (h2 : f 1 = 1) : 
  deriv f 1 + deriv g 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_derivative_sum_at_one_l2452_245246


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_for_polynomial_l2452_245288

theorem smallest_root_of_unity_for_polynomial : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → m < n → ∃ z : ℂ, z^5 - z^3 + 1 = 0 ∧ z^m ≠ 1) ∧
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_for_polynomial_l2452_245288


namespace NUMINAMATH_CALUDE_dog_hare_speed_ratio_l2452_245214

/-- The ratio of dog's speed to hare's speed given their leap patterns -/
theorem dog_hare_speed_ratio :
  ∀ (dog_leaps hare_leaps : ℕ) (dog_distance hare_distance : ℝ),
  dog_leaps = 10 →
  hare_leaps = 2 →
  dog_distance = 2 * hare_distance →
  (dog_leaps * dog_distance) / (hare_leaps * hare_distance) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_hare_speed_ratio_l2452_245214


namespace NUMINAMATH_CALUDE_x_squared_coefficient_zero_l2452_245228

/-- The coefficient of x^2 in the expansion of (x^2+ax+1)(x^2-3a+2) is zero when a = 1 -/
theorem x_squared_coefficient_zero (a : ℝ) : 
  (a = 1) ↔ ((-3 * a + 2 + 1) = 0) := by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_zero_l2452_245228


namespace NUMINAMATH_CALUDE_inequality_proof_l2452_245295

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2452_245295


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2452_245250

theorem quadratic_roots_property (d e : ℝ) : 
  (5 * d^2 - 4 * d - 1 = 0) → 
  (5 * e^2 - 4 * e - 1 = 0) → 
  (d - 2) * (e - 2) = 11/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2452_245250


namespace NUMINAMATH_CALUDE_polyhedron_sum_theorem_l2452_245296

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  euler_formula : V - E + F = 2

/-- Represents the face configuration of a polyhedron --/
structure FaceConfig where
  T : ℕ  -- number of triangles meeting at each vertex
  H : ℕ  -- number of hexagons meeting at each vertex

theorem polyhedron_sum_theorem (p : ConvexPolyhedron) (fc : FaceConfig)
  (h_faces : p.F = 50)
  (h_vertex_config : fc.T = 3 ∧ fc.H = 2) :
  100 * fc.H + 10 * fc.T + p.V = 230 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_theorem_l2452_245296


namespace NUMINAMATH_CALUDE_perfect_squares_l2452_245238

theorem perfect_squares (a b c : ℕ+) 
  (h_gcd : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (h_eq : a.val ^ 2 + b.val ^ 2 + c.val ^ 2 = 2 * (a.val * b.val + b.val * c.val + c.val * a.val)) :
  ∃ (x y z : ℕ), a.val = x ^ 2 ∧ b.val = y ^ 2 ∧ c.val = z ^ 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_l2452_245238


namespace NUMINAMATH_CALUDE_min_value_theorem_l2452_245297

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  3 / a + 2 / b ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ 3 / a₀ + 2 / b₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2452_245297


namespace NUMINAMATH_CALUDE_four_jumps_reduction_l2452_245248

def jump_reduction (initial : ℕ) (jumps : ℕ) (reduction : ℕ) : ℕ :=
  initial - jumps * reduction

theorem four_jumps_reduction : jump_reduction 320 4 10 = 280 := by
  sorry

end NUMINAMATH_CALUDE_four_jumps_reduction_l2452_245248


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_tan_l2452_245237

theorem sqrt_difference_equals_negative_two_tan (α : Real) 
  (h : α ∈ Set.Ioo (-Real.pi) (-Real.pi/2)) : 
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - 
  Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = 
  -2 * Real.tan α :=
by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_tan_l2452_245237


namespace NUMINAMATH_CALUDE_half_of_one_point_six_million_l2452_245266

theorem half_of_one_point_six_million (x : ℝ) : 
  x = 1.6 * (10 : ℝ)^6 → (1/2 : ℝ) * x = 8 * (10 : ℝ)^5 := by
  sorry

end NUMINAMATH_CALUDE_half_of_one_point_six_million_l2452_245266


namespace NUMINAMATH_CALUDE_subject_selection_ways_l2452_245286

/-- The number of ways to choose 1 subject from 2 options -/
def physics_history_choices : Nat := 2

/-- The number of subjects to choose from for the remaining two subjects -/
def remaining_subject_options : Nat := 4

/-- The number of subjects to be chosen from the remaining options -/
def subjects_to_choose : Nat := 2

/-- Calculates the number of ways to choose k items from n options -/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

theorem subject_selection_ways :
  physics_history_choices * choose remaining_subject_options subjects_to_choose = 12 := by
  sorry

end NUMINAMATH_CALUDE_subject_selection_ways_l2452_245286


namespace NUMINAMATH_CALUDE_david_average_marks_l2452_245220

def david_marks : List ℝ := [72, 45, 72, 77, 75]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℝ) = 68.2 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l2452_245220


namespace NUMINAMATH_CALUDE_count_injective_functions_count_non_injective_functions_no_surjective_function_l2452_245293

/-- Set A with 3 elements -/
def A : Type := Fin 3

/-- Set B with 4 elements -/
def B : Type := Fin 4

/-- The number of injective functions from A to B is 24 -/
theorem count_injective_functions : (A → B) → Nat :=
  fun _ => 24

/-- The number of non-injective functions from A to B is 40 -/
theorem count_non_injective_functions : (A → B) → Nat :=
  fun _ => 40

/-- There does not exist a surjective function from A to B -/
theorem no_surjective_function : ¬∃ (f : A → B), Function.Surjective f := by
  sorry

end NUMINAMATH_CALUDE_count_injective_functions_count_non_injective_functions_no_surjective_function_l2452_245293


namespace NUMINAMATH_CALUDE_class_overlap_difference_l2452_245240

theorem class_overlap_difference (total students_geometry students_biology : ℕ) 
  (h_total : total = 232)
  (h_geometry : students_geometry = 144)
  (h_biology : students_biology = 119) :
  let max_overlap := min students_geometry students_biology
  let min_overlap := students_geometry + students_biology - total
  max_overlap - min_overlap = 88 := by
sorry

end NUMINAMATH_CALUDE_class_overlap_difference_l2452_245240


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2452_245218

def tournament_games (n : ℕ) : ℕ := n * (n - 1)

theorem chess_tournament_games :
  tournament_games 17 * 2 = 544 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2452_245218


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l2452_245243

theorem probability_at_least_one_girl (total : ℕ) (boys : ℕ) (girls : ℕ) (select : ℕ) :
  total = boys + girls →
  boys = 4 →
  girls = 2 →
  select = 3 →
  (1 - (Nat.choose boys select / Nat.choose total select : ℚ)) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l2452_245243


namespace NUMINAMATH_CALUDE_revenue_decrease_l2452_245262

theorem revenue_decrease (T C : ℝ) (h1 : T > 0) (h2 : C > 0) :
  let new_tax := 0.8 * T
  let new_consumption := 1.1 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_revenue_decrease_l2452_245262


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l2452_245221

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Checks if a point satisfies the equation of a line -/
def on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The original line 2x - 3y + 2 = 0 -/
def original_line : Line :=
  { a := 2, b := -3, c := 2 }

/-- The symmetric line to be proven -/
def symmetric_line : Line :=
  { a := 2, b := 3, c := 2 }

theorem symmetric_line_correct :
  ∀ p : ℝ × ℝ, on_line symmetric_line p ↔ on_line original_line (reflect_x p) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l2452_245221


namespace NUMINAMATH_CALUDE_expression_evaluation_l2452_245257

theorem expression_evaluation (a b c : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 5) :
  2 * ((a^2 + b)^2 - (a^2 - b)^2) * c^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2452_245257


namespace NUMINAMATH_CALUDE_binary_multiplication_l2452_245232

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- The first binary number 11101₂ -/
def num1 : List Bool := [true, true, true, false, true]

/-- The second binary number 1101₂ -/
def num2 : List Bool := [true, true, false, true]

/-- The expected product 1001101101₂ -/
def expectedProduct : List Bool := [true, false, false, true, true, false, true, true, false, true]

theorem binary_multiplication :
  bitsToNat num1 * bitsToNat num2 = bitsToNat expectedProduct := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l2452_245232


namespace NUMINAMATH_CALUDE_remainder_less_than_divisor_l2452_245229

theorem remainder_less_than_divisor (a d : ℤ) (h : d ≠ 0) :
  ∃ (q r : ℤ), a = q * d + r ∧ 0 ≤ r ∧ r < |d| := by
  sorry

end NUMINAMATH_CALUDE_remainder_less_than_divisor_l2452_245229


namespace NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l2452_245216

theorem square_difference_divided_by_eleven : (121^2 - 110^2) / 11 = 231 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l2452_245216


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_circles_perimeter_l2452_245254

/-- Represents a triangle with circles inside --/
structure TriangleWithCircles where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  circle_radius : ℝ

/-- Calculates the perimeter of a triangle with circles inside --/
def perimeter_with_circles (t : TriangleWithCircles) : ℝ :=
  t.side1 + t.side2 + t.base - 4 * t.circle_radius

/-- Theorem: The perimeter of the specified isosceles triangle with circles is 24 --/
theorem isosceles_triangle_with_circles_perimeter :
  let t : TriangleWithCircles := {
    side1 := 12,
    side2 := 12,
    base := 8,
    circle_radius := 2
  }
  perimeter_with_circles t = 24 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_circles_perimeter_l2452_245254


namespace NUMINAMATH_CALUDE_systematic_sampling_l2452_245206

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (last_sample : Nat) 
  (h1 : total_students = 300)
  (h2 : sample_size = 60)
  (h3 : last_sample = 293) :
  ∃ (first_sample : Nat), first_sample = 3 ∧ 
  (first_sample + (sample_size - 1) * (total_students / sample_size) = last_sample) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l2452_245206


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2452_245289

theorem multiplicative_inverse_203_mod_301 : ∃ a : ℕ, 0 ≤ a ∧ a < 301 ∧ (203 * a) % 301 = 1 :=
by
  use 238
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2452_245289


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_solution_set_for_any_a_l2452_245213

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*a*x + 2*a^2

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem solution_set_for_any_a (a : ℝ) :
  ({x : ℝ | f a x < 0} = ∅ ∧ a = 0) ∨
  ({x : ℝ | f a x < 0} = {x : ℝ | a < x ∧ x < 2*a} ∧ a > 0) ∨
  ({x : ℝ | f a x < 0} = {x : ℝ | 2*a < x ∧ x < a} ∧ a < 0) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_solution_set_for_any_a_l2452_245213


namespace NUMINAMATH_CALUDE_total_legs_is_43_l2452_245204

/-- Represents the number of legs for different types of passengers -/
structure LegCount where
  cat : Nat
  human : Nat
  oneLeggedCaptain : Nat

/-- Calculates the total number of legs given the number of heads and cats -/
def totalLegs (totalHeads : Nat) (catCount : Nat) (legCount : LegCount) : Nat :=
  let humanCount := totalHeads - catCount
  let regularHumanCount := humanCount - 1 -- Subtract the one-legged captain
  catCount * legCount.cat + regularHumanCount * legCount.human + legCount.oneLeggedCaptain

/-- Theorem stating that given the conditions, the total number of legs is 43 -/
theorem total_legs_is_43 (totalHeads : Nat) (catCount : Nat) (legCount : LegCount)
    (h1 : totalHeads = 15)
    (h2 : catCount = 7)
    (h3 : legCount.cat = 4)
    (h4 : legCount.human = 2)
    (h5 : legCount.oneLeggedCaptain = 1) :
    totalLegs totalHeads catCount legCount = 43 := by
  sorry


end NUMINAMATH_CALUDE_total_legs_is_43_l2452_245204


namespace NUMINAMATH_CALUDE_max_spheres_in_frustum_l2452_245299

structure Frustum where
  height : ℝ

structure Sphere where
  radius : ℝ

def is_tangent_to_frustum (s : Sphere) (f : Frustum) : Prop := sorry

def is_tangent_to_sphere (s1 s2 : Sphere) : Prop := sorry

def can_fit_inside_frustum (s : Sphere) (f : Frustum) : Prop := sorry

theorem max_spheres_in_frustum (f : Frustum) (o1 o2 : Sphere) 
  (h_height : f.height = 8)
  (h_o1_radius : o1.radius = 2)
  (h_o2_radius : o2.radius = 3)
  (h_o1_tangent : is_tangent_to_frustum o1 f)
  (h_o2_tangent : is_tangent_to_frustum o2 f)
  (h_o1_o2_tangent : is_tangent_to_sphere o1 o2) :
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬∃ (spheres : Fin m → Sphere), 
      (∀ i, (spheres i).radius = 3 ∧ 
            can_fit_inside_frustum (spheres i) f ∧
            (∀ j, i ≠ j → is_tangent_to_sphere (spheres i) (spheres j)))) :=
sorry

end NUMINAMATH_CALUDE_max_spheres_in_frustum_l2452_245299


namespace NUMINAMATH_CALUDE_compute_fraction_power_l2452_245276

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l2452_245276


namespace NUMINAMATH_CALUDE_locus_and_tangent_l2452_245279

-- Define the points and lines
def A : ℝ × ℝ := (1, 0)
def B : ℝ → ℝ × ℝ := λ y ↦ (-1, y)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the locus E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

theorem locus_and_tangent :
  (∀ y : ℝ, ∃ M : ℝ × ℝ, 
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - (B y).1)^2 + (M.2 - (B y).2)^2 ∧
    M ∈ E) ∧
  (P ∈ E ∧ tangent_line ∩ E = {P}) := by sorry

end NUMINAMATH_CALUDE_locus_and_tangent_l2452_245279


namespace NUMINAMATH_CALUDE_usual_weekly_salary_proof_l2452_245264

/-- Calculates the weekly salary given daily rate and work days per week -/
def weeklySalary (dailyRate : ℚ) (workDaysPerWeek : ℕ) : ℚ :=
  dailyRate * workDaysPerWeek

/-- Represents a worker with a daily rate and standard work week -/
structure Worker where
  dailyRate : ℚ
  workDaysPerWeek : ℕ

theorem usual_weekly_salary_proof (w : Worker) 
    (h1 : w.workDaysPerWeek = 5)
    (h2 : w.dailyRate * 2 = 745) :
    weeklySalary w.dailyRate w.workDaysPerWeek = 1862.5 := by
  sorry

#eval weeklySalary (745 / 2) 5

end NUMINAMATH_CALUDE_usual_weekly_salary_proof_l2452_245264


namespace NUMINAMATH_CALUDE_symmetric_point_in_fourth_quadrant_l2452_245274

theorem symmetric_point_in_fourth_quadrant (a : ℝ) (P : ℝ × ℝ) :
  a < 0 →
  P = (-a^2 - 1, -a + 3) →
  (∃ P1 : ℝ × ℝ, P1 = (-P.1, -P.2) ∧ P1.1 > 0 ∧ P1.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_in_fourth_quadrant_l2452_245274


namespace NUMINAMATH_CALUDE_original_number_l2452_245215

theorem original_number (x : ℝ) : x * 1.4 = 1680 ↔ x = 1200 := by sorry

end NUMINAMATH_CALUDE_original_number_l2452_245215


namespace NUMINAMATH_CALUDE_rudy_running_time_l2452_245261

-- Define the running segments
def segment1_distance : ℝ := 5
def segment1_rate : ℝ := 10
def segment2_distance : ℝ := 4
def segment2_rate : ℝ := 9.5
def segment3_distance : ℝ := 3
def segment3_rate : ℝ := 8.5
def segment4_distance : ℝ := 2
def segment4_rate : ℝ := 12

-- Define the rest times
def rest1 : ℝ := 15
def rest2 : ℝ := 10
def rest3 : ℝ := 5

-- Define the total time function
def total_time : ℝ :=
  segment1_distance * segment1_rate +
  segment2_distance * segment2_rate +
  segment3_distance * segment3_rate +
  segment4_distance * segment4_rate +
  rest1 + rest2 + rest3

-- Theorem statement
theorem rudy_running_time : total_time = 167.5 := by
  sorry

end NUMINAMATH_CALUDE_rudy_running_time_l2452_245261


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l2452_245217

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | ∃ a ∈ A, x = 2*a}

theorem complement_of_A_union_B (h : Set ℕ) : 
  h = U \ (A ∪ B) → h = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l2452_245217


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_l2452_245263

/-- The equation of the tangent line to y = x sin x at (π, 0) is y = -πx + π² -/
theorem tangent_line_at_pi (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => t * Real.sin t
  let f' : ℝ → ℝ := λ t => Real.sin t + t * Real.cos t
  let tangent_line : ℝ → ℝ := λ t => -π * t + π^2
  (∀ t, HasDerivAt f (f' t) t) →
  HasDerivAt f (f' π) π →
  tangent_line π = f π →
  tangent_line = λ t => -π * t + π^2 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_at_pi_l2452_245263


namespace NUMINAMATH_CALUDE_roxanne_change_l2452_245260

def lemonade_price : ℚ := 2
def lemonade_quantity : ℕ := 4

def sandwich_price : ℚ := 2.5
def sandwich_quantity : ℕ := 3

def watermelon_price : ℚ := 1.25
def watermelon_quantity : ℕ := 2

def chips_price : ℚ := 1.75
def chips_quantity : ℕ := 1

def cookie_price : ℚ := 0.75
def cookie_quantity : ℕ := 4

def pretzel_price : ℚ := 1
def pretzel_quantity : ℕ := 5

def salad_price : ℚ := 8
def salad_quantity : ℕ := 1

def payment : ℚ := 100

theorem roxanne_change :
  payment - (lemonade_price * lemonade_quantity +
             sandwich_price * sandwich_quantity +
             watermelon_price * watermelon_quantity +
             chips_price * chips_quantity +
             cookie_price * cookie_quantity +
             pretzel_price * pretzel_quantity +
             salad_price * salad_quantity) = 63.75 := by
  sorry

end NUMINAMATH_CALUDE_roxanne_change_l2452_245260


namespace NUMINAMATH_CALUDE_inequality_proof_l2452_245244

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_two : a + b + c = 2) :
  (1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) ≥ 27/13 ∧
  ((1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) = 27/13 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2452_245244


namespace NUMINAMATH_CALUDE_manager_percentage_reduction_l2452_245259

theorem manager_percentage_reduction (total_employees : ℕ) (initial_percentage : ℚ) 
  (managers_leaving : ℕ) (target_percentage : ℚ) : 
  total_employees = 600 →
  initial_percentage = 99 / 100 →
  managers_leaving = 300 →
  target_percentage = 49 / 100 →
  (total_employees * initial_percentage - managers_leaving) / total_employees = target_percentage := by
  sorry

end NUMINAMATH_CALUDE_manager_percentage_reduction_l2452_245259


namespace NUMINAMATH_CALUDE_heart_op_ratio_l2452_245268

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : 
  (heart_op 3 5 : ℚ) / (heart_op 5 3) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_heart_op_ratio_l2452_245268


namespace NUMINAMATH_CALUDE_factors_of_81_l2452_245280

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l2452_245280


namespace NUMINAMATH_CALUDE_impossible_to_make_all_divisible_by_three_l2452_245252

/-- Represents the state of numbers on the vertices of a 2018-sided polygon -/
def PolygonState := Fin 2018 → ℤ

/-- The initial state of the polygon -/
def initial_state : PolygonState :=
  fun i => if i.val = 2017 then 1 else 0

/-- The sum of all numbers on the vertices -/
def vertex_sum (state : PolygonState) : ℤ :=
  (Finset.univ.sum fun i => state i)

/-- Represents a legal move on the polygon -/
inductive LegalMove
  | add_subtract (i j : Fin 2018) : LegalMove

/-- Apply a legal move to a given state -/
def apply_move (state : PolygonState) (move : LegalMove) : PolygonState :=
  match move with
  | LegalMove.add_subtract i j =>
      fun k => if k = i then state k + 1
               else if k = j then state k - 1
               else state k

/-- Predicate to check if all numbers are divisible by 3 -/
def all_divisible_by_three (state : PolygonState) : Prop :=
  ∀ i, state i % 3 = 0

theorem impossible_to_make_all_divisible_by_three :
  ¬∃ (moves : List LegalMove), 
    all_divisible_by_three (moves.foldl apply_move initial_state) :=
  sorry


end NUMINAMATH_CALUDE_impossible_to_make_all_divisible_by_three_l2452_245252


namespace NUMINAMATH_CALUDE_range_of_g_l2452_245245

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 2

-- Define the function g as the composition of f with itself
def g (x : ℝ) : ℝ := f (f x)

-- State the theorem about the range of g
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 1 ≤ g x ∧ g x ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l2452_245245


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l2452_245285

theorem six_digit_multiple_of_nine (n : ℕ) (h1 : n ≥ 734601 ∧ n ≤ 734691) 
  (h2 : n % 9 = 0) : 
  ∃ d : ℕ, (d = 6 ∨ d = 9) ∧ n = 734601 + d * 100 :=
sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l2452_245285


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2452_245205

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 9 ∧ b = 4 ∧
      (a + a > b) ∧  -- Triangle inequality
      perimeter = a + a + b ∧
      perimeter = 22
      
#check isosceles_triangle_perimeter

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 22 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2452_245205


namespace NUMINAMATH_CALUDE_parallel_vectors_result_obtuse_triangle_result_l2452_245298

noncomputable section

def m (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)
def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 / 2)

def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

def f (x : ℝ) : ℝ := (m x).1^2 + (m x).2^2 - ((n x).1^2 + (n x).2^2)

theorem parallel_vectors_result (x : ℝ) (h : parallel (m x) (n x)) :
  (Real.sin x + Real.sqrt 3 * Real.cos x) / (Real.sqrt 3 * Real.sin x - Real.cos x) = 3 * Real.sqrt 3 :=
sorry

theorem obtuse_triangle_result (A B : ℝ) (hA : A > π / 2) (hC : Real.sin A = 1 / 2) :
  f A = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_result_obtuse_triangle_result_l2452_245298


namespace NUMINAMATH_CALUDE_rectangle_toothpicks_l2452_245226

/-- Calculates the number of toothpicks needed to form a rectangle --/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  let horizontal_rows := width + 1
  let vertical_columns := length + 1
  horizontal_rows * length + vertical_columns * width

/-- Theorem: A rectangle with length 20 and width 10 requires 430 toothpicks --/
theorem rectangle_toothpicks :
  toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end NUMINAMATH_CALUDE_rectangle_toothpicks_l2452_245226


namespace NUMINAMATH_CALUDE_flower_shop_problem_l2452_245255

/-- Flower shop problem -/
theorem flower_shop_problem (roses_per_bouquet : ℕ) 
  (total_bouquets : ℕ) (rose_bouquets : ℕ) (daisy_bouquets : ℕ) 
  (total_flowers : ℕ) : 
  roses_per_bouquet = 12 →
  total_bouquets = 20 →
  rose_bouquets = 10 →
  daisy_bouquets = 10 →
  rose_bouquets + daisy_bouquets = total_bouquets →
  total_flowers = 190 →
  (total_flowers - roses_per_bouquet * rose_bouquets) / daisy_bouquets = 7 :=
by sorry

end NUMINAMATH_CALUDE_flower_shop_problem_l2452_245255


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2452_245277

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1836 * k) :
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2452_245277


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l2452_245271

theorem cubic_function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 1
  f (-2) = 1 → f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l2452_245271


namespace NUMINAMATH_CALUDE_max_robot_weight_is_270_l2452_245202

/-- Represents the weight constraints and components of a robot in the competition. -/
structure RobotWeightConstraints where
  standard_robot_weight : ℝ
  battery_weight : ℝ
  min_payload_weight : ℝ
  max_payload_weight : ℝ
  min_robot_weight_diff : ℝ

/-- Calculates the maximum weight of a robot in the competition. -/
def max_robot_weight (constraints : RobotWeightConstraints) : ℝ :=
  let min_robot_weight := constraints.standard_robot_weight + constraints.min_robot_weight_diff
  let min_total_weight := min_robot_weight + constraints.battery_weight + constraints.min_payload_weight
  2 * min_total_weight

/-- Theorem stating the maximum weight of a robot in the competition. -/
theorem max_robot_weight_is_270 (constraints : RobotWeightConstraints) 
    (h1 : constraints.standard_robot_weight = 100)
    (h2 : constraints.battery_weight = 20)
    (h3 : constraints.min_payload_weight = 10)
    (h4 : constraints.max_payload_weight = 25)
    (h5 : constraints.min_robot_weight_diff = 5) :
  max_robot_weight constraints = 270 := by
  sorry

#eval max_robot_weight { 
  standard_robot_weight := 100,
  battery_weight := 20,
  min_payload_weight := 10,
  max_payload_weight := 25,
  min_robot_weight_diff := 5
}

end NUMINAMATH_CALUDE_max_robot_weight_is_270_l2452_245202


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2452_245256

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point to be reflected -/
def original_point : ℝ × ℝ := (-2, -3)

/-- The expected result after reflection -/
def expected_reflection : ℝ × ℝ := (-2, 3)

theorem reflection_across_x_axis :
  reflect_x original_point = expected_reflection := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2452_245256


namespace NUMINAMATH_CALUDE_percentage_problem_l2452_245203

theorem percentage_problem (P : ℝ) : (P / 100) * 600 = (50 / 100) * 720 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2452_245203


namespace NUMINAMATH_CALUDE_yuna_has_greatest_sum_l2452_245239

/-- Yoojung's first number -/
def yoojung_num1 : ℕ := 5

/-- Yoojung's second number -/
def yoojung_num2 : ℕ := 8

/-- Yuna's first number -/
def yuna_num1 : ℕ := 7

/-- Yuna's second number -/
def yuna_num2 : ℕ := 9

/-- The sum of Yoojung's numbers -/
def yoojung_sum : ℕ := yoojung_num1 + yoojung_num2

/-- The sum of Yuna's numbers -/
def yuna_sum : ℕ := yuna_num1 + yuna_num2

theorem yuna_has_greatest_sum : yuna_sum > yoojung_sum := by
  sorry

end NUMINAMATH_CALUDE_yuna_has_greatest_sum_l2452_245239
