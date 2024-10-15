import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_inscribed_circle_perpendicular_projections_l432_43201

structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def UnitCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

def projection (M : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

theorem rectangle_inscribed_circle_perpendicular_projections
  (ABCD : Rectangle)
  (h_inscribed : {ABCD.A, ABCD.B, ABCD.C, ABCD.D} ⊆ UnitCircle)
  (M : ℝ × ℝ)
  (h_M_on_arc : M ∈ UnitCircle ∧ M ≠ ABCD.A ∧ M ≠ ABCD.B)
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ) (S : ℝ × ℝ)
  (h_P : P = projection M {x | x.1 = ABCD.A.1})
  (h_Q : Q = projection M {x | x.2 = ABCD.A.2})
  (h_R : R = projection M {x | x.1 = ABCD.C.1})
  (h_S : S = projection M {x | x.2 = ABCD.C.2}) :
  (P.1 - Q.1) * (R.1 - S.1) + (P.2 - Q.2) * (R.2 - S.2) = 0 :=
sorry

end NUMINAMATH_CALUDE_rectangle_inscribed_circle_perpendicular_projections_l432_43201


namespace NUMINAMATH_CALUDE_equal_distances_l432_43212

/-- Two circles in a plane -/
structure TwoCircles where
  Γ₁ : Set (ℝ × ℝ)
  Γ₂ : Set (ℝ × ℝ)

/-- Points of intersection of two circles -/
structure IntersectionPoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : A ∈ tc.Γ₁ ∧ A ∈ tc.Γ₂
  h₂ : B ∈ tc.Γ₁ ∧ B ∈ tc.Γ₂

/-- Common tangent line to two circles -/
structure CommonTangent (tc : TwoCircles) where
  Δ : Set (ℝ × ℝ)
  C : ℝ × ℝ
  D : ℝ × ℝ
  h₁ : C ∈ tc.Γ₁ ∧ C ∈ Δ
  h₂ : D ∈ tc.Γ₂ ∧ D ∈ Δ

/-- Intersection point of lines AB and CD -/
def intersectionPoint (ip : IntersectionPoints tc) (ct : CommonTangent tc) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distances PC and PD are equal -/
theorem equal_distances (tc : TwoCircles) (ip : IntersectionPoints tc) (ct : CommonTangent tc) :
  let P := intersectionPoint ip ct
  distance P ct.C = distance P ct.D := by sorry

end NUMINAMATH_CALUDE_equal_distances_l432_43212


namespace NUMINAMATH_CALUDE_inequality_solution_set_l432_43223

theorem inequality_solution_set (a b : ℝ) (h1 : a < 0) (h2 : b = a) 
  (h3 : ∀ x, ax + b ≤ 0 ↔ x ≥ -1) :
  ∀ x, (a*x + b) / (x - 2) > 0 ↔ -1 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l432_43223


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l432_43274

/-- The y-intercept of the line 6x - 4y = 24 is (0, -6) -/
theorem y_intercept_of_line (x y : ℝ) : 
  (6 * x - 4 * y = 24) → (x = 0 → y = -6) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l432_43274


namespace NUMINAMATH_CALUDE_exists_permutation_with_many_swaps_l432_43297

/-- 
Represents a permutation of cards numbered from 1 to n.
-/
def Permutation (n : ℕ) := Fin n → Fin n

/-- 
Counts the number of adjacent swaps needed to sort a permutation into descending order.
-/
def countSwaps (n : ℕ) (p : Permutation n) : ℕ := sorry

/-- 
Theorem: There exists a permutation of n cards that requires at least n(n-1)/2 adjacent swaps
to sort into descending order.
-/
theorem exists_permutation_with_many_swaps (n : ℕ) :
  ∃ (p : Permutation n), countSwaps n p ≥ n * (n - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_exists_permutation_with_many_swaps_l432_43297


namespace NUMINAMATH_CALUDE_lowest_price_after_discounts_l432_43235

/-- Calculates the lowest possible price of a product after applying two consecutive discounts -/
theorem lowest_price_after_discounts 
  (original_price : ℝ) 
  (max_regular_discount : ℝ) 
  (sale_discount : ℝ) : 
  original_price * (1 - max_regular_discount) * (1 - sale_discount) = 22.40 :=
by
  -- Assuming original_price = 40.00, max_regular_discount = 0.30, and sale_discount = 0.20
  sorry

#check lowest_price_after_discounts

end NUMINAMATH_CALUDE_lowest_price_after_discounts_l432_43235


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l432_43260

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  a 0 = 4096 ∧ a 1 = 1024 ∧ a 2 = 256 ∧
  a 5 = 4 ∧ a 6 = 1 ∧ a 7 = (1/4 : ℚ) ∧
  (∀ n : ℕ, a (n + 1) = a n * (1/4 : ℚ)) →
  a 3 + a 4 = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l432_43260


namespace NUMINAMATH_CALUDE_route_comparison_l432_43250

-- Define the circular tram line
structure TramLine where
  circumference : ℝ
  park_zoo : ℝ
  zoo_circus : ℝ
  circus_park : ℝ

-- Define the conditions
def valid_tram_line (t : TramLine) : Prop :=
  t.park_zoo + t.zoo_circus + t.circus_park = t.circumference ∧
  t.park_zoo + t.circus_park = 3 * t.park_zoo ∧
  t.zoo_circus = 2 * (t.park_zoo + t.circus_park)

-- Theorem statement
theorem route_comparison (t : TramLine) (h : valid_tram_line t) :
  t.circus_park = 11 * t.zoo_circus :=
sorry

end NUMINAMATH_CALUDE_route_comparison_l432_43250


namespace NUMINAMATH_CALUDE_surfers_count_l432_43254

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 20

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 2 * santa_monica_surfers

/-- The total number of surfers on both beaches -/
def total_surfers : ℕ := malibu_surfers + santa_monica_surfers

theorem surfers_count : total_surfers = 60 := by
  sorry

end NUMINAMATH_CALUDE_surfers_count_l432_43254


namespace NUMINAMATH_CALUDE_special_pentagon_angles_l432_43206

/-- A pentagon that is a cross-section of a parallelepiped with side ratio constraints -/
structure SpecialPentagon where
  -- The pentagon is a cross-section of a parallelepiped
  is_cross_section : Bool
  -- The ratio of any two sides is either 1, 2, or 1/2
  side_ratio_constraint : ∀ (s1 s2 : ℝ), s1 > 0 → s2 > 0 → s1 / s2 ∈ ({1, 2, 1/2} : Set ℝ)

/-- The interior angles of the special pentagon -/
def interior_angles (p : SpecialPentagon) : List ℝ := sorry

/-- Theorem stating the interior angles of the special pentagon -/
theorem special_pentagon_angles (p : SpecialPentagon) :
  ∃ (angles : List ℝ), angles = interior_angles p ∧ angles.length = 5 ∧
  (angles.count 120 = 4 ∧ angles.count 60 = 1) := by sorry

end NUMINAMATH_CALUDE_special_pentagon_angles_l432_43206


namespace NUMINAMATH_CALUDE_blake_grocery_change_l432_43268

/-- Calculates the change Blake receives after purchasing groceries with discounts and sales tax. -/
theorem blake_grocery_change (oranges apples mangoes strawberries bananas : ℚ)
  (strawberry_discount banana_discount sales_tax : ℚ)
  (blake_money : ℚ)
  (h1 : oranges = 40)
  (h2 : apples = 50)
  (h3 : mangoes = 60)
  (h4 : strawberries = 30)
  (h5 : bananas = 20)
  (h6 : strawberry_discount = 10 / 100)
  (h7 : banana_discount = 5 / 100)
  (h8 : sales_tax = 7 / 100)
  (h9 : blake_money = 300) :
  let discounted_strawberries := strawberries * (1 - strawberry_discount)
  let discounted_bananas := bananas * (1 - banana_discount)
  let total_cost := oranges + apples + mangoes + discounted_strawberries + discounted_bananas
  let total_with_tax := total_cost * (1 + sales_tax)
  blake_money - total_with_tax = 90.28 := by
sorry


end NUMINAMATH_CALUDE_blake_grocery_change_l432_43268


namespace NUMINAMATH_CALUDE_inequality_solution_set_l432_43264

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 5 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l432_43264


namespace NUMINAMATH_CALUDE_cosine_is_periodic_l432_43299

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the cosine function
def cos : ℝ → ℝ := sorry

theorem cosine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric cos →
  IsPeriodic cos := by
  sorry

end NUMINAMATH_CALUDE_cosine_is_periodic_l432_43299


namespace NUMINAMATH_CALUDE_storage_tubs_price_l432_43296

/-- Calculates the total price Alison paid for storage tubs after discount -/
def total_price_after_discount (
  large_count : ℕ
  ) (medium_count : ℕ
  ) (small_count : ℕ
  ) (large_price : ℚ
  ) (medium_price : ℚ
  ) (small_price : ℚ
  ) (small_discount : ℚ
  ) : ℚ :=
  let large_medium_total := large_count * large_price + medium_count * medium_price
  let small_total := small_count * small_price * (1 - small_discount)
  large_medium_total + small_total

/-- Theorem stating the total price Alison paid for storage tubs after discount -/
theorem storage_tubs_price :
  total_price_after_discount 4 6 8 8 6 4 (1/10) = 968/10 :=
by
  sorry


end NUMINAMATH_CALUDE_storage_tubs_price_l432_43296


namespace NUMINAMATH_CALUDE_geometric_series_sum_l432_43211

theorem geometric_series_sum : 
  let a : ℚ := 2/3
  let r : ℚ := 2/3
  let n : ℕ := 12
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 1054690/531441 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l432_43211


namespace NUMINAMATH_CALUDE_not_all_triangles_form_square_l432_43245

/-- A triangle is a set of three points in a plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A partition of a triangle is a division of the triangle into a finite number of smaller parts. -/
def Partition (T : Triangle) (n : ℕ) := Set (Set (ℝ × ℝ))

/-- A square is a regular quadrilateral with four equal sides and four right angles. -/
structure Square where
  side : ℝ

/-- A function that checks if a partition of a triangle can be reassembled into a square. -/
def can_form_square (T : Triangle) (p : Partition T 1000) (S : Square) : Prop :=
  sorry

/-- Theorem stating that not all triangles can be divided into 1000 parts to form a square. -/
theorem not_all_triangles_form_square :
  ∃ T : Triangle, ¬∃ (p : Partition T 1000) (S : Square), can_form_square T p S := by
  sorry

end NUMINAMATH_CALUDE_not_all_triangles_form_square_l432_43245


namespace NUMINAMATH_CALUDE_f_inequality_solution_range_l432_43226

-- Define the function f
def f (x m : ℝ) : ℝ := -x^2 + x + m + 2

-- Define the property of having exactly one integer solution
def has_exactly_one_integer_solution (m : ℝ) : Prop :=
  ∃! (n : ℤ), f n m ≥ |n|

-- State the theorem
theorem f_inequality_solution_range :
  ∀ m : ℝ, has_exactly_one_integer_solution m ↔ m ∈ Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_range_l432_43226


namespace NUMINAMATH_CALUDE_equal_cell_squares_l432_43202

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if a square in the grid has an equal number of black and white cells -/
def has_equal_cells (g : Grid) (top_left : Fin 5 × Fin 5) (size : Nat) : Bool :=
  sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_cell_squares (g : Grid) : count_equal_squares g = 16 := by
  sorry

end NUMINAMATH_CALUDE_equal_cell_squares_l432_43202


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_plane_perp_plane_l432_43208

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (plane_perp_plane : Plane → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_perp_plane_implies_plane_perp_plane
  (α β : Plane) (l : Line)
  (h1 : line_subset_plane l α)
  (h2 : line_perp_plane l β) :
  plane_perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_plane_perp_plane_l432_43208


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l432_43233

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  q > 1 →  -- common ratio > 1
  4 * (a 2010)^2 - 8 * (a 2010) + 3 = 0 →  -- a_2010 is a root
  4 * (a 2011)^2 - 8 * (a 2011) + 3 = 0 →  -- a_2011 is a root
  a 2012 + a 2013 = 18 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l432_43233


namespace NUMINAMATH_CALUDE_mixed_feed_cost_per_pound_l432_43225

theorem mixed_feed_cost_per_pound
  (total_weight : ℝ)
  (cheap_cost_per_pound : ℝ)
  (expensive_cost_per_pound : ℝ)
  (cheap_weight : ℝ)
  (h1 : total_weight = 35)
  (h2 : cheap_cost_per_pound = 0.18)
  (h3 : expensive_cost_per_pound = 0.53)
  (h4 : cheap_weight = 17)
  : (cheap_weight * cheap_cost_per_pound + (total_weight - cheap_weight) * expensive_cost_per_pound) / total_weight = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_mixed_feed_cost_per_pound_l432_43225


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l432_43222

theorem cos_x_plus_2y_equals_one (x y a : ℝ) 
  (x_in_range : x ∈ Set.Icc (-π/4) (π/4))
  (y_in_range : y ∈ Set.Icc (-π/4) (π/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry


end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l432_43222


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l432_43284

theorem max_product_sum_2000 : 
  (∃ (a b : ℤ), a + b = 2000 ∧ a * b = 1000000) ∧
  (∀ (x y : ℤ), x + y = 2000 → x * y ≤ 1000000) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l432_43284


namespace NUMINAMATH_CALUDE_sum_of_roots_even_function_l432_43200

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a function that has exactly 4 real roots
def HasFourRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

-- Theorem statement
theorem sum_of_roots_even_function
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_four_roots : HasFourRealRoots f) :
  ∃ a b c d : ℝ, (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧ a + b + c + d = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_function_l432_43200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l432_43210

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The main theorem -/
theorem arithmetic_sequence_increasing_iff_a1_lt_a3 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 < a 3 ↔ MonotonicallyIncreasing a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l432_43210


namespace NUMINAMATH_CALUDE_football_team_throwers_l432_43251

/-- Proves the number of throwers on a football team given specific conditions -/
theorem football_team_throwers 
  (total_players : ℕ) 
  (total_right_handed : ℕ) 
  (h_total : total_players = 70)
  (h_right : total_right_handed = 62)
  (h_throwers_right : ∀ t : ℕ, t ≤ total_players → t ≤ total_right_handed)
  (h_non_throwers_division : ∀ n : ℕ, n < total_players → 
    3 * (total_players - n) = 2 * (total_right_handed - n) + (total_players - total_right_handed)) :
  ∃ throwers : ℕ, throwers = 46 ∧ throwers ≤ total_players ∧ throwers ≤ total_right_handed :=
sorry

end NUMINAMATH_CALUDE_football_team_throwers_l432_43251


namespace NUMINAMATH_CALUDE_g_of_3_equals_12_l432_43236

def g (x : ℝ) : ℝ := x^3 - 2*x^2 + x

theorem g_of_3_equals_12 : g 3 = 12 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_12_l432_43236


namespace NUMINAMATH_CALUDE_circle_area_ratio_l432_43288

theorem circle_area_ratio (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 3) 
  (h₃ : (r₁ + r₂)^2 + 40^2 = 41^2) (h₄ : 20 * (r₁ + r₂) = 300) :
  (π * r₁^2) / (π * r₂^2) = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l432_43288


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l432_43220

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 1 ↔ (x : ℚ) / 3 + 7 / 4 < 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l432_43220


namespace NUMINAMATH_CALUDE_tennis_players_count_l432_43280

/-- The number of members who play tennis in a sports club -/
def tennis_players (total_members badminton_players neither_players both_players : ℕ) : ℕ :=
  total_members - neither_players - (badminton_players - both_players)

/-- Theorem stating the number of tennis players in the sports club -/
theorem tennis_players_count :
  tennis_players 30 17 3 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tennis_players_count_l432_43280


namespace NUMINAMATH_CALUDE_pages_read_sunday_l432_43294

def average_pages_per_day : ℕ := 50
def days_in_week : ℕ := 7
def pages_monday : ℕ := 65
def pages_tuesday : ℕ := 28
def pages_wednesday : ℕ := 0
def pages_thursday : ℕ := 70
def pages_friday : ℕ := 56
def pages_saturday : ℕ := 88

def total_pages_week : ℕ := average_pages_per_day * days_in_week
def pages_monday_to_friday : ℕ := pages_monday + pages_tuesday + pages_wednesday + pages_thursday + pages_friday
def pages_monday_to_saturday : ℕ := pages_monday_to_friday + pages_saturday

theorem pages_read_sunday : 
  total_pages_week - pages_monday_to_saturday = 43 := by sorry

end NUMINAMATH_CALUDE_pages_read_sunday_l432_43294


namespace NUMINAMATH_CALUDE_product_of_complex_sets_l432_43272

theorem product_of_complex_sets : ∃ (z₁ z₂ : ℂ), 
  (Complex.I * z₁ = 1) ∧ 
  (z₂ + Complex.I = 1) ∧ 
  (z₁ * z₂ = -1 - Complex.I) := by sorry

end NUMINAMATH_CALUDE_product_of_complex_sets_l432_43272


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l432_43282

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∃ (μ : ℝ), ∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + b*c + μ*c*d) ∧
  (∀ (μ : ℝ), (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + b*c + μ*c*d) → μ ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l432_43282


namespace NUMINAMATH_CALUDE_tenth_angle_measure_l432_43218

/-- The sum of interior angles of a decagon -/
def decagon_angle_sum : ℝ := 1440

/-- The number of angles in a decagon that are 150° -/
def num_150_angles : ℕ := 9

/-- The measure of each of the known angles -/
def known_angle_measure : ℝ := 150

theorem tenth_angle_measure (decagon_sum : ℝ) (num_known : ℕ) (known_measure : ℝ) 
  (h1 : decagon_sum = decagon_angle_sum) 
  (h2 : num_known = num_150_angles) 
  (h3 : known_measure = known_angle_measure) :
  decagon_sum - num_known * known_measure = 90 := by
  sorry

end NUMINAMATH_CALUDE_tenth_angle_measure_l432_43218


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l432_43287

theorem quadratic_solution_sum (a b : ℝ) : 
  (a^2 - 6*a + 11 = 23) → 
  (b^2 - 6*b + 11 = 23) → 
  (a ≥ b) → 
  (a + 3*b = 12 - 2*Real.sqrt 21) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l432_43287


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l432_43224

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 2024 ∧ n % (39 * sum_of_digits n) = 0

theorem special_numbers_theorem :
  {n : ℕ | satisfies_condition n} = {351, 702, 1053, 1404} := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l432_43224


namespace NUMINAMATH_CALUDE_fifth_power_sum_l432_43253

theorem fifth_power_sum (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 2) :
  a^5 + b^5 = 19/4 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l432_43253


namespace NUMINAMATH_CALUDE_ln_abs_even_and_increasing_l432_43281

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem ln_abs_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ln_abs_even_and_increasing_l432_43281


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l432_43283

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l432_43283


namespace NUMINAMATH_CALUDE_increasing_function_implies_m_range_l432_43216

/-- The function f(x) = 2x³ - 3mx² + 6x --/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

/-- f is increasing on the interval (2, +∞) --/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

/-- The theorem stating that if f is increasing on (2, +∞), then m ∈ (-∞, 5/2] --/
theorem increasing_function_implies_m_range (m : ℝ) :
  is_increasing_on_interval m → m ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_implies_m_range_l432_43216


namespace NUMINAMATH_CALUDE_min_value_of_expression_l432_43252

theorem min_value_of_expression (x : ℚ) : (2*x - 5)^2 + 18 ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l432_43252


namespace NUMINAMATH_CALUDE_triangle_inequality_bounds_l432_43247

theorem triangle_inequality_bounds (a b c : ℝ) 
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sum_two : a + b + c = 2) :
  1 ≤ a * b + b * c + c * a - a * b * c ∧ 
  a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_bounds_l432_43247


namespace NUMINAMATH_CALUDE_furniture_factory_solution_valid_furniture_factory_solution_optimal_l432_43257

/-- Represents the solution to the furniture factory worker allocation problem -/
def furniture_factory_solution (total_workers : ℕ) 
  (tabletops_per_worker : ℕ) (legs_per_worker : ℕ) 
  (legs_per_table : ℕ) : ℕ × ℕ :=
  (20, 40)

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem furniture_factory_solution_valid 
  (total_workers : ℕ) (tabletops_per_worker : ℕ) 
  (legs_per_worker : ℕ) (legs_per_table : ℕ) :
  let (tabletop_workers, leg_workers) := 
    furniture_factory_solution total_workers tabletops_per_worker legs_per_worker legs_per_table
  (total_workers = tabletop_workers + leg_workers) ∧ 
  (tabletops_per_worker * tabletop_workers * legs_per_table = legs_per_worker * leg_workers) ∧
  (total_workers = 60) ∧ 
  (tabletops_per_worker = 3) ∧ 
  (legs_per_worker = 6) ∧ 
  (legs_per_table = 4) :=
by
  sorry

/-- Theorem stating that the solution maximizes production -/
theorem furniture_factory_solution_optimal 
  (total_workers : ℕ) (tabletops_per_worker : ℕ) 
  (legs_per_worker : ℕ) (legs_per_table : ℕ) :
  let (tabletop_workers, leg_workers) := 
    furniture_factory_solution total_workers tabletops_per_worker legs_per_worker legs_per_table
  ∀ (x y : ℕ), 
    (x + y = total_workers) → 
    (tabletops_per_worker * x * legs_per_table = legs_per_worker * y) →
    (tabletops_per_worker * x ≤ tabletops_per_worker * tabletop_workers) :=
by
  sorry

end NUMINAMATH_CALUDE_furniture_factory_solution_valid_furniture_factory_solution_optimal_l432_43257


namespace NUMINAMATH_CALUDE_scientific_notation_of_34000_l432_43219

theorem scientific_notation_of_34000 :
  (34000 : ℝ) = 3.4 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_34000_l432_43219


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l432_43278

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 : ℂ) / ((1 + Complex.I)^2 + 1) + Complex.I^4
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l432_43278


namespace NUMINAMATH_CALUDE_estate_distribution_l432_43213

-- Define the estate distribution function
def distribute (total : ℕ) (n : ℕ) : ℕ → ℕ
| 0 => 0  -- Base case: no children
| (i+1) => 
  let fixed := 1000 * i
  let remaining := total - fixed
  fixed + remaining / 10

-- Theorem statement
theorem estate_distribution (total : ℕ) :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ i j : ℕ, i > 0 → j > 0 → i ≤ n → j ≤ n → 
      distribute total n i = distribute total n j) ∧
    (∀ i : ℕ, i > 0 → i ≤ n → distribute total n i > 0)) →
  (∃ n : ℕ, n = 9 ∧
    (∀ i j : ℕ, i > 0 → j > 0 → i ≤ n → j ≤ n → 
      distribute total n i = distribute total n j) ∧
    (∀ i : ℕ, i > 0 → i ≤ n → distribute total n i > 0)) :=
by sorry


end NUMINAMATH_CALUDE_estate_distribution_l432_43213


namespace NUMINAMATH_CALUDE_stratified_by_educational_stage_is_most_reasonable_l432_43290

-- Define the different sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define the educational stages
inductive EducationalStage
| Primary
| JuniorHigh
| HighSchool

-- Define the conditions
def significantDifferencesInEducationalStages : Prop := True
def noSignificantDifferencesBetweenGenders : Prop := True
def goalIsUnderstandVisionConditions : Prop := True

-- Define the most reasonable sampling method
def mostReasonableSamplingMethod : SamplingMethod := SamplingMethod.StratifiedByEducationalStage

-- Theorem statement
theorem stratified_by_educational_stage_is_most_reasonable :
  significantDifferencesInEducationalStages →
  noSignificantDifferencesBetweenGenders →
  goalIsUnderstandVisionConditions →
  mostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_by_educational_stage_is_most_reasonable_l432_43290


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_problem_1_l432_43204

theorem quadratic_roots_sum_and_product (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) :
  a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 →
  x₁ + x₂ = -b / a ∧ x₁ * x₂ = c / a :=
by sorry

theorem problem_1 (x₁ x₂ : ℝ) :
  5 * x₁^2 + 10 * x₁ - 1 = 0 ∧ 5 * x₂^2 + 10 * x₂ - 1 = 0 →
  x₁ + x₂ = -2 ∧ x₁ * x₂ = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_problem_1_l432_43204


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l432_43232

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l432_43232


namespace NUMINAMATH_CALUDE_sales_solution_l432_43267

def sales_problem (sales : List ℕ) (average : ℕ) : Prop :=
  sales.length = 4 ∧ 
  (sales.sum + (average * 5 - sales.sum)) / 5 = average

theorem sales_solution (sales : List ℕ) (average : ℕ) 
  (h : sales_problem sales average) : 
  average * 5 - sales.sum = (average * 5 - sales.sum) := by sorry

end NUMINAMATH_CALUDE_sales_solution_l432_43267


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l432_43237

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l432_43237


namespace NUMINAMATH_CALUDE_best_marksman_score_prove_best_marksman_score_l432_43243

/-- Calculates the best marksman's score in a shooting competition. -/
theorem best_marksman_score (team_size : ℕ) (hypothetical_best_score : ℕ) (hypothetical_average : ℕ) (actual_total_score : ℕ) : ℕ :=
  let hypothetical_total := (team_size - 1) * hypothetical_average + hypothetical_best_score
  hypothetical_best_score - (hypothetical_total - actual_total_score)

/-- Proves that the best marksman's score is 77 given the problem conditions. -/
theorem prove_best_marksman_score :
  best_marksman_score 8 92 84 665 = 77 := by
  sorry

end NUMINAMATH_CALUDE_best_marksman_score_prove_best_marksman_score_l432_43243


namespace NUMINAMATH_CALUDE_expression_simplification_l432_43259

theorem expression_simplification (x : ℝ) (h : x^2 - x - 1 = 0) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x^2 - x) / (x^2 + 2 * x + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l432_43259


namespace NUMINAMATH_CALUDE_girls_percentage_after_adding_boy_l432_43276

def initial_boys : ℕ := 11
def initial_girls : ℕ := 13
def added_boys : ℕ := 1

def total_students : ℕ := initial_boys + initial_girls + added_boys

def girls_percentage : ℚ := (initial_girls : ℚ) / (total_students : ℚ) * 100

theorem girls_percentage_after_adding_boy :
  girls_percentage = 52 := by sorry

end NUMINAMATH_CALUDE_girls_percentage_after_adding_boy_l432_43276


namespace NUMINAMATH_CALUDE_point_transformation_sum_l432_43271

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90° counterclockwise around (2, 3) -/
def rotate90 (p : Point) : Point :=
  { x := -p.y + 5, y := p.x + 1 }

/-- Reflects a point about the line y = -x -/
def reflectAboutNegativeX (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- The final transformation applied to point P -/
def finalTransform (p : Point) : Point :=
  reflectAboutNegativeX (rotate90 p)

/-- Theorem statement -/
theorem point_transformation_sum (a b : ℝ) :
  let p := Point.mk a b
  finalTransform p = Point.mk (-3) 2 → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_sum_l432_43271


namespace NUMINAMATH_CALUDE_glenville_population_l432_43240

theorem glenville_population (h p : ℕ) : 
  (∃ h p, 13 * h + 6 * p = 48) ∧
  (∃ h p, 13 * h + 6 * p = 52) ∧
  (∃ h p, 13 * h + 6 * p = 65) ∧
  (∃ h p, 13 * h + 6 * p = 75) ∧
  (∀ h p, 13 * h + 6 * p ≠ 70) :=
by sorry

end NUMINAMATH_CALUDE_glenville_population_l432_43240


namespace NUMINAMATH_CALUDE_sandy_shopping_percentage_l432_43262

/-- The percentage of money Sandy spent on shopping -/
def shopping_percentage (initial_amount spent_amount : ℚ) : ℚ :=
  (spent_amount / initial_amount) * 100

/-- Proof that Sandy spent 30% of her money on shopping -/
theorem sandy_shopping_percentage :
  let initial_amount : ℚ := 200
  let remaining_amount : ℚ := 140
  let spent_amount : ℚ := initial_amount - remaining_amount
  shopping_percentage initial_amount spent_amount = 30 := by
sorry

end NUMINAMATH_CALUDE_sandy_shopping_percentage_l432_43262


namespace NUMINAMATH_CALUDE_work_completion_time_l432_43273

theorem work_completion_time (p d : ℕ) (work_left : ℚ) : 
  p = 15 → 
  work_left = 0.5333333333333333 →
  (4 : ℚ) * ((1 : ℚ) / p + (1 : ℚ) / d) = 1 - work_left →
  d = 20 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l432_43273


namespace NUMINAMATH_CALUDE_sequence_formula_l432_43246

def S (n : ℕ) : ℕ := n^2 + 3*n

def a (n : ℕ) : ℕ := 2*n + 2

theorem sequence_formula (n : ℕ) : 
  (∀ k : ℕ, S k = k^2 + 3*k) → 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l432_43246


namespace NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l432_43298

theorem pascals_triangle_51st_row_third_number : 
  (Nat.choose 51 2) = 1275 := by sorry

end NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l432_43298


namespace NUMINAMATH_CALUDE_opposite_number_l432_43270

theorem opposite_number (x : ℤ) : (- x = 2016) → (x = -2016) := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_l432_43270


namespace NUMINAMATH_CALUDE_outfit_choices_l432_43229

/-- Represents the number of available items of each type -/
def num_items : ℕ := 7

/-- Represents the number of available colors -/
def num_colors : ℕ := 7

/-- Calculates the total number of possible outfits -/
def total_outfits : ℕ := num_items * num_items * num_items

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Calculates the number of valid outfits (not all items the same color) -/
def valid_outfits : ℕ := total_outfits - same_color_outfits

theorem outfit_choices :
  valid_outfits = 336 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l432_43229


namespace NUMINAMATH_CALUDE_weight_replacement_l432_43203

theorem weight_replacement (initial_count : Nat) (weight_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  new_weight = 105 →
  (new_weight - (initial_count * weight_increase)) = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l432_43203


namespace NUMINAMATH_CALUDE_pizza_sharing_l432_43244

theorem pizza_sharing (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) : 
  total_slices = 78 → 
  buzz_ratio = 5 → 
  waiter_ratio = 8 → 
  (waiter_ratio * total_slices) / (buzz_ratio + waiter_ratio) - 20 = 28 := by
sorry

end NUMINAMATH_CALUDE_pizza_sharing_l432_43244


namespace NUMINAMATH_CALUDE_vodka_alcohol_consumption_l432_43261

/-- Calculates the amount of pure alcohol consumed by one person when splitting vodka -/
theorem vodka_alcohol_consumption 
  (total_shots : ℕ) 
  (ounces_per_shot : ℚ) 
  (alcohol_percentage : ℚ) 
  (num_people : ℕ) 
  (h1 : total_shots = 8)
  (h2 : ounces_per_shot = 3/2)
  (h3 : alcohol_percentage = 1/2)
  (h4 : num_people = 2) :
  (total_shots : ℚ) * ounces_per_shot * alcohol_percentage / num_people = 3 := by
  sorry

end NUMINAMATH_CALUDE_vodka_alcohol_consumption_l432_43261


namespace NUMINAMATH_CALUDE_price_change_theorem_l432_43209

theorem price_change_theorem (initial_price : ℝ) (initial_price_positive : initial_price > 0) :
  let price_after_increase := initial_price * 1.31
  let price_after_first_discount := price_after_increase * 0.9
  let final_price := price_after_first_discount * 0.85
  (final_price - initial_price) / initial_price = 0.00215 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l432_43209


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l432_43231

def numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]

theorem arithmetic_mean_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 12 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l432_43231


namespace NUMINAMATH_CALUDE_champion_determination_races_l432_43255

/-- The number of races needed to determine a champion sprinter -/
def races_needed (initial_sprinters : ℕ) (sprinters_per_race : ℕ) (eliminated_per_race : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 50 races are needed for the given conditions -/
theorem champion_determination_races :
  races_needed 400 10 8 = 50 := by sorry

end NUMINAMATH_CALUDE_champion_determination_races_l432_43255


namespace NUMINAMATH_CALUDE_smallest_integer_above_sum_of_roots_l432_43249

theorem smallest_integer_above_sum_of_roots : ∃ n : ℕ, n = 2703 ∧ 
  (∀ m : ℕ, (m : ℝ) > (Real.sqrt 4 + Real.sqrt 3)^6 → m ≥ n) ∧
  ((n : ℝ) - 1 ≤ (Real.sqrt 4 + Real.sqrt 3)^6) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sum_of_roots_l432_43249


namespace NUMINAMATH_CALUDE_min_rectangles_to_cover_l432_43241

/-- Represents a corner in the shape --/
inductive Corner
| Type1
| Type2

/-- Represents the shape with its corner configuration --/
structure Shape where
  type1_corners : Nat
  type2_corners : Nat

/-- Represents a rectangle that can cover cells and corners --/
structure Rectangle where
  covered_corners : List Corner

/-- Defines the properties of the shape as given in the problem --/
def problem_shape : Shape :=
  { type1_corners := 12
  , type2_corners := 12 }

/-- Theorem stating the minimum number of rectangles needed to cover the shape --/
theorem min_rectangles_to_cover (s : Shape) 
  (h1 : s.type1_corners = problem_shape.type1_corners) 
  (h2 : s.type2_corners = problem_shape.type2_corners) :
  ∃ (rectangles : List Rectangle), 
    (rectangles.length = 12) ∧ 
    (∀ c : Corner, c ∈ Corner.Type1 :: List.replicate s.type1_corners Corner.Type1 ++ 
                   Corner.Type2 :: List.replicate s.type2_corners Corner.Type2 → 
      ∃ r ∈ rectangles, c ∈ r.covered_corners) :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_to_cover_l432_43241


namespace NUMINAMATH_CALUDE_officer_selection_count_l432_43277

/-- The number of members in the club -/
def club_members : ℕ := 12

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 4

/-- The number of ways to choose officers from the club members -/
def ways_to_choose_officers : ℕ := club_members * (club_members - 1) * (club_members - 2) * (club_members - 3)

theorem officer_selection_count :
  ways_to_choose_officers = 11880 :=
sorry

end NUMINAMATH_CALUDE_officer_selection_count_l432_43277


namespace NUMINAMATH_CALUDE_eggs_laid_per_chicken_l432_43217

theorem eggs_laid_per_chicken 
  (initial_eggs : ℕ) 
  (used_eggs : ℕ) 
  (num_chickens : ℕ) 
  (final_eggs : ℕ) 
  (h1 : initial_eggs = 10)
  (h2 : used_eggs = 5)
  (h3 : num_chickens = 2)
  (h4 : final_eggs = 11)
  : (final_eggs - (initial_eggs - used_eggs)) / num_chickens = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_laid_per_chicken_l432_43217


namespace NUMINAMATH_CALUDE_infiniteSeriesSum_l432_43269

/-- The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

/-- Theorem: The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ is equal to 3/4 -/
theorem infiniteSeriesSum : infiniteSeries = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_infiniteSeriesSum_l432_43269


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_11_to_51_l432_43242

theorem sum_of_odd_integers_11_to_51 (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 11 →
  aₙ = 51 →
  d = 2 →
  aₙ = a₁ + (n - 1) * d →
  (n : ℚ) / 2 * (a₁ + aₙ) = 651 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_11_to_51_l432_43242


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l432_43266

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Finite E] [Fact (finrank ℝ E = 3)]

-- Define unit vectors
variable (a b c d : E)

-- State the theorem
theorem max_sum_squared_distances (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l432_43266


namespace NUMINAMATH_CALUDE_age_difference_l432_43291

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 15) : A - C = 15 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l432_43291


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l432_43228

theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 11) (h3 : v1 = 11) (h4 : v2 = 9) :
  let t1 := d1 / v1
  let t2 := d2 / v2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let avg_speed := total_distance / total_time
  ∃ ε > 0, |avg_speed - 9.8| < ε :=
by sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l432_43228


namespace NUMINAMATH_CALUDE_sum_of_first_100_digits_l432_43295

/-- The decimal expansion of 1/10101 -/
def decimal_expansion : ℕ → ℕ
| n => sorry

/-- The sum of the first n digits in the decimal expansion of 1/10101 -/
def digit_sum (n : ℕ) : ℕ :=
  (List.range n).map decimal_expansion |>.sum

/-- Theorem: The sum of the first 100 digits after the decimal point in 1/10101 is 450 -/
theorem sum_of_first_100_digits : digit_sum 100 = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_digits_l432_43295


namespace NUMINAMATH_CALUDE_polar_line_theorem_l432_43238

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  rho : ℝ
  theta : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a line in polar coordinates -/
def lies_on (p : PolarPoint) (l : PolarLine) : Prop :=
  l.equation p.rho p.theta

/-- Checks if a line is parallel to the polar axis -/
def parallel_to_polar_axis (l : PolarLine) : Prop :=
  ∀ (rho theta : ℝ), l.equation rho theta ↔ l.equation rho 0

theorem polar_line_theorem (p : PolarPoint) (l : PolarLine) 
  (h1 : p.rho = 2 ∧ p.theta = π/3)
  (h2 : lies_on p l)
  (h3 : parallel_to_polar_axis l) :
  ∀ (rho theta : ℝ), l.equation rho theta ↔ rho * Real.sin theta = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_line_theorem_l432_43238


namespace NUMINAMATH_CALUDE_salary_expenses_l432_43205

theorem salary_expenses (S : ℝ) 
  (h1 : S - (2/5)*S - (3/10)*S - (1/8)*S = 1400) :
  (3/10)*S + (1/8)*S = 3400 :=
by sorry

end NUMINAMATH_CALUDE_salary_expenses_l432_43205


namespace NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_product_l432_43275

theorem no_triangle_with_cube_sum_equal_product : ¬∃ (x y z : ℝ), 
  (0 < x ∧ 0 < y ∧ 0 < z) ∧  -- positive real numbers
  (x + y > z ∧ y + z > x ∧ z + x > y) ∧  -- triangle inequality
  (x^3 + y^3 + z^3 = (x+y)*(y+z)*(z+x)) := by
  sorry


end NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_product_l432_43275


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_l432_43265

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the inscribed circle
structure InscribedCircle (t : EquilateralTriangle) where
  center : ℝ × ℝ  -- Representing the center point P
  radius : ℝ
  radius_pos : radius > 0
  touches_sides : True  -- Assumption that the circle touches all sides

-- Define the perpendicular distances
def perpendicular_distances (t : EquilateralTriangle) (c : InscribedCircle t) : ℝ × ℝ × ℝ :=
  (c.radius, c.radius, c.radius)

-- Theorem statement
theorem sum_of_perpendiculars (t : EquilateralTriangle) (c : InscribedCircle t) :
  let (d1, d2, d3) := perpendicular_distances t c
  d1 + d2 + d3 = (Real.sqrt 3 * t.side_length) / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_l432_43265


namespace NUMINAMATH_CALUDE_volunteer_selection_l432_43256

theorem volunteer_selection (n_boys n_girls n_selected : ℕ) 
  (h_boys : n_boys = 4)
  (h_girls : n_girls = 3)
  (h_selected : n_selected = 3) : 
  (Nat.choose n_boys 2 * Nat.choose n_girls 1) + 
  (Nat.choose n_girls 2 * Nat.choose n_boys 1) = 30 :=
sorry

end NUMINAMATH_CALUDE_volunteer_selection_l432_43256


namespace NUMINAMATH_CALUDE_shirt_cost_difference_l432_43248

/-- The difference in cost between two shirts -/
def cost_difference (total_cost first_shirt_cost : ℕ) : ℕ :=
  first_shirt_cost - (total_cost - first_shirt_cost)

/-- Proof that the difference in cost between two shirts is $6 -/
theorem shirt_cost_difference :
  let total_cost : ℕ := 24
  let first_shirt_cost : ℕ := 15
  first_shirt_cost > total_cost - first_shirt_cost →
  cost_difference total_cost first_shirt_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_difference_l432_43248


namespace NUMINAMATH_CALUDE_distinct_centroids_count_l432_43293

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- Represents the distribution of points on the perimeter of a rectangle -/
structure PerimeterPoints where
  total : ℕ
  long_side : ℕ
  short_side : ℕ

/-- Calculates the number of distinct centroid positions for triangles formed by
    any three non-collinear points from the specified points on the rectangle's perimeter -/
def count_distinct_centroids (rect : Rectangle) (points : PerimeterPoints) : ℕ :=
  sorry

/-- The main theorem stating that for a 12x8 rectangle with 48 equally spaced points
    on its perimeter, there are 925 distinct centroid positions -/
theorem distinct_centroids_count :
  let rect := Rectangle.mk 12 8
  let points := PerimeterPoints.mk 48 16 8
  count_distinct_centroids rect points = 925 := by
  sorry

end NUMINAMATH_CALUDE_distinct_centroids_count_l432_43293


namespace NUMINAMATH_CALUDE_part_one_part_two_l432_43285

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (b : ℝ) (x : ℝ) : ℝ := b*x + 5 - 2*b

-- Part 1
theorem part_one (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, f a x = 0) → -8 ≤ a ∧ a ≤ 0 :=
sorry

-- Part 2
theorem part_two (b : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Icc 1 4, g b x₁ = f 3 x₂) →
  b ∈ Set.Icc (-1) (1/2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l432_43285


namespace NUMINAMATH_CALUDE_brothers_baskets_count_l432_43258

/-- Represents the number of strawberries in each basket picked by Kimberly's brother -/
def strawberries_per_basket : ℕ := 15

/-- Represents the number of people sharing the strawberries -/
def number_of_people : ℕ := 4

/-- Represents the number of strawberries each person gets when divided equally -/
def strawberries_per_person : ℕ := 168

/-- Represents the number of baskets Kimberly's brother picked -/
def brothers_baskets : ℕ := 3

theorem brothers_baskets_count :
  ∃ (b : ℕ),
    b = brothers_baskets ∧
    (17 * b * strawberries_per_basket - 93 = number_of_people * strawberries_per_person) :=
by sorry

end NUMINAMATH_CALUDE_brothers_baskets_count_l432_43258


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l432_43234

theorem sin_negative_thirty_degrees :
  Real.sin (-(30 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l432_43234


namespace NUMINAMATH_CALUDE_triangle_theorem_l432_43286

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific acute triangle -/
theorem triangle_theorem (t : Triangle) 
  (h_acute : 0 < t.A ∧ t.A < π/2 ∧ 0 < t.B ∧ t.B < π/2 ∧ 0 < t.C ∧ t.C < π/2)
  (h_cos : Real.cos t.A / Real.cos t.C = t.a / (2 * t.b - t.c))
  (h_a : t.a = Real.sqrt 7)
  (h_c : t.c = 3)
  (h_D : ∃ D : ℝ × ℝ, D = ((t.b + t.c)/2, 0)) :
  t.A = π/3 ∧ Real.sqrt ((t.b^2 + t.c^2 + 2*t.b*t.c*Real.cos t.A) / 4) = Real.sqrt 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l432_43286


namespace NUMINAMATH_CALUDE_lauren_mail_problem_l432_43207

theorem lauren_mail_problem (x : ℕ) 
  (h : x + (x + 10) + (x + 5) + (x + 20) = 295) : 
  x = 65 := by
  sorry

end NUMINAMATH_CALUDE_lauren_mail_problem_l432_43207


namespace NUMINAMATH_CALUDE_infinite_impossible_d_l432_43227

theorem infinite_impossible_d : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (d : ℕ), d ∈ S →
    ¬∃ (t r : ℝ), t > 0 ∧ 3 * t - 2 * Real.pi * r = 500 ∧ t = 2 * r + d :=
sorry

end NUMINAMATH_CALUDE_infinite_impossible_d_l432_43227


namespace NUMINAMATH_CALUDE_system_solution_l432_43292

theorem system_solution (x y z : ℝ) : 
  (x^2 + y^2 - z*(x + y) = 2 ∧
   y^2 + z^2 - x*(y + z) = 4 ∧
   z^2 + x^2 - y*(z + x) = 8) ↔
  ((x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l432_43292


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l432_43214

/-- The cost of Grandfather Zhao's ticket -/
def grandfather_ticket_cost : ℝ := 10

/-- The discount rate for Grandfather Zhao's ticket -/
def grandfather_discount_rate : ℝ := 0.2

/-- The number of minor tickets -/
def num_minor_tickets : ℕ := 3

/-- The discount rate for minor tickets -/
def minor_discount_rate : ℝ := 0.4

/-- The number of regular tickets -/
def num_regular_tickets : ℕ := 2

/-- The number of senior tickets (excluding Grandfather Zhao) -/
def num_senior_tickets : ℕ := 1

/-- The discount rate for senior tickets (excluding Grandfather Zhao) -/
def senior_discount_rate : ℝ := 0.3

/-- The total cost of all tickets -/
def total_cost : ℝ := 66.25

theorem concert_ticket_cost :
  let regular_ticket_cost := grandfather_ticket_cost / (1 - grandfather_discount_rate)
  let minor_ticket_cost := regular_ticket_cost * (1 - minor_discount_rate)
  let senior_ticket_cost := regular_ticket_cost * (1 - senior_discount_rate)
  total_cost = num_minor_tickets * minor_ticket_cost +
               num_regular_tickets * regular_ticket_cost +
               num_senior_tickets * senior_ticket_cost +
               grandfather_ticket_cost :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l432_43214


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l432_43263

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/144
  let r : ℚ := -10/21
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l432_43263


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l432_43215

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 7 k) * (2^(7-k)) * (a^k) * (1^(7-2*k)) = -70 ∧ 7 - 2*k = 1) → 
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l432_43215


namespace NUMINAMATH_CALUDE_complex_exponential_form_theta_l432_43221

theorem complex_exponential_form_theta (z : ℂ) : 
  z = -1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (2 * Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_form_theta_l432_43221


namespace NUMINAMATH_CALUDE_probability_x2_y2_leq_1_probability_equals_pi_over_16_l432_43230

/-- The probability that x^2 + y^2 ≤ 1 when x and y are randomly chosen from [0,2] -/
theorem probability_x2_y2_leq_1 : ℝ :=
  let total_area : ℝ := 4 -- Area of the square [0,2] × [0,2]
  let circle_area : ℝ := Real.pi / 4 -- Area of the quarter circle x^2 + y^2 ≤ 1 in the first quadrant
  circle_area / total_area

/-- The main theorem stating that the probability is equal to π/16 -/
theorem probability_equals_pi_over_16 : probability_x2_y2_leq_1 = Real.pi / 16 := by
  sorry


end NUMINAMATH_CALUDE_probability_x2_y2_leq_1_probability_equals_pi_over_16_l432_43230


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l432_43279

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I
theorem solve_inequality (x : ℝ) :
  f 2 x < 4 ↔ -1/2 < x ∧ x < 7/2 := by sorry

-- Part II
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l432_43279


namespace NUMINAMATH_CALUDE_operation_result_l432_43239

theorem operation_result (x : ℕ) (h : x = 40) : (((x / 4) * 5) + 10) - 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l432_43239


namespace NUMINAMATH_CALUDE_cyclic_quadrilaterals_area_l432_43289

/-- Represents a convex cyclic quadrilateral --/
structure CyclicQuadrilateral where
  diag_angle : Real
  inscribed_radius : Real
  area : Real

/-- The theorem to be proved --/
theorem cyclic_quadrilaterals_area 
  (A B C : CyclicQuadrilateral)
  (h_radius : A.inscribed_radius = 1 ∧ B.inscribed_radius = 1 ∧ C.inscribed_radius = 1)
  (h_sin_A : Real.sin A.diag_angle = 2/3)
  (h_sin_B : Real.sin B.diag_angle = 3/5)
  (h_sin_C : Real.sin C.diag_angle = 6/7)
  (h_equal_area : A.area = B.area ∧ B.area = C.area)
  : A.area = 16/35 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilaterals_area_l432_43289
