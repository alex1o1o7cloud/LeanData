import Mathlib

namespace NUMINAMATH_CALUDE_prime_factors_count_l2901_290197

theorem prime_factors_count : 
  let expression := [(2, 25), (3, 17), (5, 11), (7, 8), (11, 4), (13, 3)]
  (expression.map (λ (p : ℕ × ℕ) => p.2)).sum = 68 := by sorry

end NUMINAMATH_CALUDE_prime_factors_count_l2901_290197


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2901_290125

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 11 : ℚ) = (n / 100 : ℕ) + ((n / 10) % 10 : ℕ) + (n % 10 : ℕ) ∧
  n = 198 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2901_290125


namespace NUMINAMATH_CALUDE_distance_to_midpoint_is_12_l2901_290196

/-- An isosceles triangle DEF with given side lengths -/
structure IsoscelesTriangleDEF where
  /-- The length of side DE -/
  de : ℝ
  /-- The length of side DF -/
  df : ℝ
  /-- The length of side EF -/
  ef : ℝ
  /-- DE and DF are equal -/
  de_eq_df : de = df
  /-- DE is 13 units -/
  de_is_13 : de = 13
  /-- EF is 10 units -/
  ef_is_10 : ef = 10

/-- The distance from D to the midpoint of EF in the isosceles triangle DEF -/
def distanceToMidpoint (t : IsoscelesTriangleDEF) : ℝ :=
  sorry

/-- Theorem stating that the distance from D to the midpoint of EF is 12 units -/
theorem distance_to_midpoint_is_12 (t : IsoscelesTriangleDEF) :
  distanceToMidpoint t = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_is_12_l2901_290196


namespace NUMINAMATH_CALUDE_min_value_theorem_l2901_290130

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  (x^2 + 6 - Real.sqrt (x^4 + 36)) / x ≥ 12 / (2 * (Real.sqrt 6 + Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2901_290130


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l2901_290140

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l2901_290140


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2901_290179

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
    (h_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n)
    (h_sum1 : a 1 + a 2 + a 3 = 2)
    (h_sum2 : a 3 + a 4 + a 5 = 8) :
  a 4 + a 5 + a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2901_290179


namespace NUMINAMATH_CALUDE_abc_ordering_l2901_290133

noncomputable def a : ℝ := (1/2)^(1/4 : ℝ)
noncomputable def b : ℝ := (1/3)^(1/2 : ℝ)
noncomputable def c : ℝ := (1/4)^(1/3 : ℝ)

theorem abc_ordering : b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_abc_ordering_l2901_290133


namespace NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2901_290190

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem interior_angle_regular_octagon :
  (sum_interior_angles octagon_sides) / octagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2901_290190


namespace NUMINAMATH_CALUDE_fraction_simplification_l2901_290184

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2901_290184


namespace NUMINAMATH_CALUDE_triangle_ap_tangent_relation_l2901_290137

/-- A triangle with sides in arithmetic progression satisfies 3 * tan(α/2) * tan(γ/2) = 1, 
    where α is the smallest angle and γ is the largest angle. -/
theorem triangle_ap_tangent_relation (a b c : ℝ) (α β γ : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a + c = 2 * b →  -- Arithmetic progression condition
  α ≤ β → β ≤ γ →  -- α is smallest, γ is largest
  3 * Real.tan (α / 2) * Real.tan (γ / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ap_tangent_relation_l2901_290137


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2901_290178

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_calculation (t : Triangle) :
  t.A = Real.pi / 3 ∧ 
  t.a = 4 * Real.sqrt 3 ∧ 
  t.b = 4 * Real.sqrt 2 →
  t.B = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2901_290178


namespace NUMINAMATH_CALUDE_alex_upside_down_hours_l2901_290191

/-- The number of hours Alex needs to hang upside down each month to be tall enough for the roller coaster --/
def hours_upside_down (
  required_height : ℚ)
  (current_height : ℚ)
  (growth_rate_upside_down : ℚ)
  (natural_growth_rate : ℚ)
  (months_in_year : ℕ) : ℚ :=
  let height_difference := required_height - current_height
  let natural_growth := natural_growth_rate * months_in_year
  let additional_growth_needed := height_difference - natural_growth
  (additional_growth_needed / growth_rate_upside_down) / months_in_year

/-- Theorem stating that Alex needs to hang upside down for 2 hours each month --/
theorem alex_upside_down_hours :
  hours_upside_down 54 48 (1/12) (1/3) 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_upside_down_hours_l2901_290191


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2901_290154

/-- Given a rectangle with dimensions (x - 3) by (2x + 3) and area 4x - 9, prove that x = 7/2 -/
theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (2 * x + 3) = 4 * x - 9 → x = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2901_290154


namespace NUMINAMATH_CALUDE_fraction_simplification_l2901_290164

theorem fraction_simplification (a : ℝ) (h : a ≠ 5) :
  (a^2 - 5*a) / (a - 5) = a := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2901_290164


namespace NUMINAMATH_CALUDE_circle_inequality_theta_range_l2901_290175

theorem circle_inequality_theta_range :
  ∀ θ : ℝ,
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x y : ℝ, (x - 2 * Real.cos θ)^2 + (y - 2 * Real.sin θ)^2 = 1 → x ≤ y) →
  (5 * Real.pi / 12 ≤ θ ∧ θ ≤ 13 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_theta_range_l2901_290175


namespace NUMINAMATH_CALUDE_smallest_y_squared_l2901_290181

/-- An isosceles trapezoid with a tangent circle --/
structure IsoscelesTrapezoidWithCircle where
  -- The length of the longer base
  pq : ℝ
  -- The length of the shorter base
  rs : ℝ
  -- The length of the equal sides
  y : ℝ
  -- Assumption that pq > rs
  h_pq_gt_rs : pq > rs

/-- The theorem stating the smallest possible value of y^2 --/
theorem smallest_y_squared (t : IsoscelesTrapezoidWithCircle) 
  (h_pq : t.pq = 120) (h_rs : t.rs = 25) :
  ∃ (n : ℝ), n^2 = 4350 ∧ ∀ (y : ℝ), y ≥ n → 
  ∃ (c : IsoscelesTrapezoidWithCircle), c.pq = 120 ∧ c.rs = 25 ∧ c.y = y :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_squared_l2901_290181


namespace NUMINAMATH_CALUDE_difference_of_ones_and_zeros_l2901_290182

-- Define the decimal number
def decimal_number : ℕ := 173

-- Define the binary representation as a list of bits
def binary_representation : List Bool := [true, false, true, false, true, true, false, true]

-- Define x as the number of zeros
def x : ℕ := binary_representation.filter (· = false) |>.length

-- Define y as the number of ones
def y : ℕ := binary_representation.filter (· = true) |>.length

-- Theorem to prove
theorem difference_of_ones_and_zeros : y - x = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_ones_and_zeros_l2901_290182


namespace NUMINAMATH_CALUDE_ratio_problem_l2901_290150

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2901_290150


namespace NUMINAMATH_CALUDE_extreme_points_count_a_range_l2901_290198

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

def has_extreme_points (n : ℕ) (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, (deriv f) x = 0 ∧
    ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → (deriv f) y ≠ 0

theorem extreme_points_count (a : ℝ) :
  (a ≤ 0 → has_extreme_points 1 (f a)) ∧
  (0 < a ∧ a < 1/2 → has_extreme_points 2 (f a)) ∧
  (a = 1/2 → has_extreme_points 0 (f a)) ∧
  (a > 1/2 → has_extreme_points 2 (f a)) :=
sorry

theorem a_range (a : ℝ) :
  (∀ x : ℝ, f a x + Real.exp x ≥ x^3 + x) → a ≤ Real.exp 1 - 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_count_a_range_l2901_290198


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_ratio_bound_l2901_290126

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem statement --/
theorem rectangle_perimeter_area_ratio_bound :
  ∀ l w : ℕ,
    l < 100 →
    w < 100 →
    l ≠ w →
    isPrime l →
    isPrime w →
    (2 * l + 2 * w)^2 / (l * w : ℚ) ≥ 82944 / 5183 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_area_ratio_bound_l2901_290126


namespace NUMINAMATH_CALUDE_pokemon_card_count_l2901_290119

/-- The number of people who have Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem pokemon_card_count : total_cards = 56 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_count_l2901_290119


namespace NUMINAMATH_CALUDE_beads_per_necklace_l2901_290157

theorem beads_per_necklace (members : ℕ) (necklaces_per_member : ℕ) (total_beads : ℕ) :
  members = 9 →
  necklaces_per_member = 2 →
  total_beads = 900 →
  total_beads / (members * necklaces_per_member) = 50 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l2901_290157


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l2901_290108

theorem arccos_equation_solution (x : ℝ) : 
  Real.arccos (2 * x - 1) = π / 4 → x = (Real.sqrt 2 + 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l2901_290108


namespace NUMINAMATH_CALUDE_game_lives_theorem_l2901_290114

/-- Calculates the total number of lives after two levels in a game. -/
def totalLives (initial : ℕ) (firstLevelGain : ℕ) (secondLevelGain : ℕ) : ℕ :=
  initial + firstLevelGain + secondLevelGain

/-- Theorem stating that with 2 initial lives, gaining 6 in the first level
    and 11 in the second level, the total number of lives is 19. -/
theorem game_lives_theorem :
  totalLives 2 6 11 = 19 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_theorem_l2901_290114


namespace NUMINAMATH_CALUDE_second_quadrant_trig_identity_l2901_290143

theorem second_quadrant_trig_identity (α : Real) 
  (h1 : π / 2 < α) (h2 : α < π) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_identity_l2901_290143


namespace NUMINAMATH_CALUDE_max_value_is_120_l2901_290172

def is_valid_assignment (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

def expression (a b c d : ℕ) : ℚ :=
  (a : ℚ) / ((b : ℚ) / ((c * d : ℚ)))

theorem max_value_is_120 :
  ∀ a b c d : ℕ, is_valid_assignment a b c d →
    expression a b c d ≤ 120 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_120_l2901_290172


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_390_l2901_290142

theorem least_multiple_of_25_greater_than_390 :
  ∀ n : ℕ, n > 0 → 25 ∣ n → n > 390 → n ≥ 400 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_390_l2901_290142


namespace NUMINAMATH_CALUDE_simplify_expression_l2901_290111

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2901_290111


namespace NUMINAMATH_CALUDE_company_picnic_teams_l2901_290169

theorem company_picnic_teams (managers : ℕ) (employees : ℕ) (teams : ℕ) :
  managers = 3 →
  employees = 3 →
  teams = 3 →
  (managers + employees) / teams = 2 := by
sorry

end NUMINAMATH_CALUDE_company_picnic_teams_l2901_290169


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l2901_290168

theorem positive_integer_solutions_of_inequality :
  {x : ℕ+ | (3 : ℝ) * x.val < x.val + 3} = {1} := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l2901_290168


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2901_290118

/-- A coloring function that assigns a color (represented by a Boolean) to each point with integer coordinates. -/
def Coloring := ℤ × ℤ → Bool

/-- Predicate to check if a rectangle satisfies the required properties. -/
def ValidRectangle (a b c d : ℤ × ℤ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  let (x₄, y₄) := d
  (x₁ = x₄ ∧ x₂ = x₃ ∧ y₁ = y₂ ∧ y₃ = y₄) ∧ 
  (∃ k : ℕ, (x₂ - x₁).natAbs * (y₃ - y₁).natAbs = 2^k)

/-- Theorem stating that there exists a coloring such that no valid rectangle has all vertices of the same color. -/
theorem exists_valid_coloring : 
  ∃ (f : Coloring), ∀ (a b c d : ℤ × ℤ), 
    ValidRectangle a b c d → 
    ¬(f a = f b ∧ f b = f c ∧ f c = f d) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2901_290118


namespace NUMINAMATH_CALUDE_equidistant_point_count_l2901_290166

/-- A quadrilateral is a polygon with four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A rectangle is a quadrilateral with four right angles. -/
def IsRectangle (q : Quadrilateral) : Prop := sorry

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
def IsTrapezoid (q : Quadrilateral) : Prop := sorry

/-- A trapezoid has congruent base angles if the angles adjacent to each parallel side are congruent. -/
def HasCongruentBaseAngles (q : Quadrilateral) : Prop := sorry

/-- A point is equidistant from all vertices of a quadrilateral if its distance to each vertex is the same. -/
def HasEquidistantPoint (q : Quadrilateral) : Prop := sorry

/-- The theorem states that among rectangles and trapezoids with congruent base angles, 
    exactly two types of quadrilaterals have a point equidistant from all four vertices. -/
theorem equidistant_point_count :
  ∃ (q1 q2 : Quadrilateral),
    (IsRectangle q1 ∨ (IsTrapezoid q1 ∧ HasCongruentBaseAngles q1)) ∧
    (IsRectangle q2 ∨ (IsTrapezoid q2 ∧ HasCongruentBaseAngles q2)) ∧
    q1 ≠ q2 ∧
    HasEquidistantPoint q1 ∧
    HasEquidistantPoint q2 ∧
    (∀ q : Quadrilateral,
      (IsRectangle q ∨ (IsTrapezoid q ∧ HasCongruentBaseAngles q)) →
      HasEquidistantPoint q →
      (q = q1 ∨ q = q2)) :=
by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_count_l2901_290166


namespace NUMINAMATH_CALUDE_max_pieces_is_18_l2901_290151

/-- Represents the size of a square cake piece -/
inductive PieceSize
  | Small : PieceSize  -- 2" x 2"
  | Medium : PieceSize -- 4" x 4"
  | Large : PieceSize  -- 6" x 6"

/-- Represents a configuration of cake pieces -/
structure CakeConfiguration where
  small_pieces : Nat
  medium_pieces : Nat
  large_pieces : Nat

/-- Checks if a given configuration fits within a 20" x 20" cake -/
def fits_in_cake (config : CakeConfiguration) : Prop :=
  2 * config.small_pieces + 4 * config.medium_pieces + 6 * config.large_pieces ≤ 400

/-- The maximum number of pieces that can be cut from the cake -/
def max_pieces : Nat := 18

/-- Theorem stating that the maximum number of pieces is 18 -/
theorem max_pieces_is_18 :
  ∀ (config : CakeConfiguration),
    fits_in_cake config →
    config.small_pieces + config.medium_pieces + config.large_pieces ≤ max_pieces :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_is_18_l2901_290151


namespace NUMINAMATH_CALUDE_equation_solution_l2901_290160

theorem equation_solution : ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2901_290160


namespace NUMINAMATH_CALUDE_ashley_amount_l2901_290141

theorem ashley_amount (ashley betty carlos dick elgin : ℕ) : 
  ashley + betty + carlos + dick + elgin = 86 →
  ashley = betty + 20 →
  (betty = carlos + 9 ∨ carlos = betty + 9) →
  (carlos = dick + 6 ∨ dick = carlos + 6) →
  (dick = elgin + 7 ∨ elgin = dick + 7) →
  elgin = ashley + 10 →
  ashley = 24 := by sorry

end NUMINAMATH_CALUDE_ashley_amount_l2901_290141


namespace NUMINAMATH_CALUDE_angle_ratio_is_one_fourth_l2901_290177

-- Define the triangle ABC
variable (A B C : Point) (ABC : Triangle A B C)

-- Define the points P and Q
variable (P Q : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ bisect angle ABC
axiom bp_bisects : angle A B P = angle P B C
axiom bq_bisects : angle A B Q = angle Q B C

-- BM is the bisector of angle PBQ
variable (M : Point)
axiom bm_bisects : angle P B M = angle M B Q

-- Theorem statement
theorem angle_ratio_is_one_fourth :
  (angle M B Q) / (angle A B Q) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_is_one_fourth_l2901_290177


namespace NUMINAMATH_CALUDE_bryans_books_l2901_290136

/-- Given that Bryan has 9 bookshelves and each bookshelf contains 56 books,
    prove that the total number of books Bryan has is 504. -/
theorem bryans_books (num_shelves : ℕ) (books_per_shelf : ℕ) 
    (h1 : num_shelves = 9) (h2 : books_per_shelf = 56) : 
    num_shelves * books_per_shelf = 504 := by
  sorry

end NUMINAMATH_CALUDE_bryans_books_l2901_290136


namespace NUMINAMATH_CALUDE_strip_covers_cube_l2901_290128

/-- A rectangular strip can cover a cube in two layers -/
theorem strip_covers_cube (strip_length : ℝ) (strip_width : ℝ) (cube_edge : ℝ) :
  strip_length = 12 →
  strip_width = 1 →
  cube_edge = 1 →
  strip_length * strip_width = 2 * 6 * cube_edge ^ 2 := by
  sorry

#check strip_covers_cube

end NUMINAMATH_CALUDE_strip_covers_cube_l2901_290128


namespace NUMINAMATH_CALUDE_calculation_proof_l2901_290186

theorem calculation_proof : (-3/4 - 5/9 + 7/12) / (-1/36) = 26 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2901_290186


namespace NUMINAMATH_CALUDE_sneakers_final_price_l2901_290110

/-- Calculates the final price of sneakers after applying discounts and sales tax -/
def finalPrice (originalPrice couponDiscount promoDiscountRate membershipDiscountRate salesTaxRate : ℚ) : ℚ :=
  let priceAfterCoupon := originalPrice - couponDiscount
  let priceAfterPromo := priceAfterCoupon * (1 - promoDiscountRate)
  let priceAfterMembership := priceAfterPromo * (1 - membershipDiscountRate)
  let finalPriceBeforeTax := priceAfterMembership
  finalPriceBeforeTax * (1 + salesTaxRate)

/-- Theorem stating that the final price of the sneakers is $100.63 -/
theorem sneakers_final_price :
  finalPrice 120 10 (5/100) (10/100) (7/100) = 10063/100 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_final_price_l2901_290110


namespace NUMINAMATH_CALUDE_log_and_exp_problem_l2901_290171

theorem log_and_exp_problem :
  (Real.log 9 / Real.log 3 = 2) ∧
  (∀ a : ℝ, a = Real.log 3 / Real.log 4 → 2^a = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_log_and_exp_problem_l2901_290171


namespace NUMINAMATH_CALUDE_kids_wearing_socks_and_shoes_l2901_290101

/-- Given a classroom with kids wearing socks, shoes, or barefoot, 
    prove the number of kids wearing both socks and shoes. -/
theorem kids_wearing_socks_and_shoes 
  (total : ℕ) 
  (socks : ℕ) 
  (shoes : ℕ) 
  (barefoot : ℕ) 
  (h1 : total = 22) 
  (h2 : socks = 12) 
  (h3 : shoes = 8) 
  (h4 : barefoot = 8) 
  (h5 : total = socks + barefoot) 
  (h6 : total = shoes + barefoot) :
  shoes = socks + shoes - total := by
sorry

end NUMINAMATH_CALUDE_kids_wearing_socks_and_shoes_l2901_290101


namespace NUMINAMATH_CALUDE_golden_ratio_roots_l2901_290170

theorem golden_ratio_roots (r : ℝ) : r^2 = r + 1 → r^6 = 8*r + 5 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_roots_l2901_290170


namespace NUMINAMATH_CALUDE_savings_theorem_l2901_290120

def initial_amount : ℚ := 2000

def wife_share (amount : ℚ) : ℚ := (2 / 5) * amount

def first_son_share (amount : ℚ) : ℚ := (2 / 5) * amount

def second_son_share (amount : ℚ) : ℚ := (2 / 5) * amount

def savings_amount (initial : ℚ) : ℚ :=
  let after_wife := initial - wife_share initial
  let after_first_son := after_wife - first_son_share after_wife
  after_first_son - second_son_share after_first_son

theorem savings_theorem : savings_amount initial_amount = 432 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l2901_290120


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2901_290159

theorem fraction_equation_solution (n : ℚ) : 
  (2 / (n + 1) + 3 / (n + 1) + n / (n + 1) = 4) → n = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2901_290159


namespace NUMINAMATH_CALUDE_completing_square_transformation_l2901_290105

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l2901_290105


namespace NUMINAMATH_CALUDE_league_games_l2901_290121

theorem league_games (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l2901_290121


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l2901_290188

/-- Calculates the annual rent per square foot of a shop given its dimensions and monthly rent. -/
theorem annual_rent_per_square_foot 
  (length width : ℝ) 
  (monthly_rent : ℝ) 
  (h1 : length = 18) 
  (h2 : width = 20) 
  (h3 : monthly_rent = 3600) : 
  (monthly_rent * 12) / (length * width) = 120 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l2901_290188


namespace NUMINAMATH_CALUDE_baseball_season_length_l2901_290135

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The total number of baseball games in a season -/
def games_in_season : ℕ := 14

/-- The number of months in a baseball season -/
def season_length : ℕ := games_in_season / games_per_month

theorem baseball_season_length :
  season_length = 2 := by sorry

end NUMINAMATH_CALUDE_baseball_season_length_l2901_290135


namespace NUMINAMATH_CALUDE_impossible_to_reach_in_six_moves_l2901_290106

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a single move of the ant -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Applies a move to a position -/
def apply_move (p : Position) (m : Move) : Position :=
  match m with
  | Move.Up    => ⟨p.x, p.y + 1⟩
  | Move.Down  => ⟨p.x, p.y - 1⟩
  | Move.Left  => ⟨p.x - 1, p.y⟩
  | Move.Right => ⟨p.x + 1, p.y⟩

/-- Applies a list of moves to a starting position -/
def apply_moves (start : Position) (moves : List Move) : Position :=
  moves.foldl apply_move start

/-- The sum of coordinates of a position -/
def coord_sum (p : Position) : Int := p.x + p.y

/-- Theorem: It's impossible to reach (2,1) or (1,2) from (0,0) in exactly 6 moves -/
theorem impossible_to_reach_in_six_moves :
  ∀ (moves : List Move),
    moves.length = 6 →
    (apply_moves ⟨0, 0⟩ moves ≠ ⟨2, 1⟩) ∧
    (apply_moves ⟨0, 0⟩ moves ≠ ⟨1, 2⟩) := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_reach_in_six_moves_l2901_290106


namespace NUMINAMATH_CALUDE_inequality_proof_l2901_290173

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 3 * (a * b + b * c + c * a) / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2901_290173


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l2901_290147

/-- A function f(x) = ax^2 - b that is decreasing on (-∞, 0) -/
def DecreasingQuadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - b

/-- The property of being decreasing on (-∞, 0) -/
def IsDecreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

theorem decreasing_quadratic_condition (a b : ℝ) :
  IsDecreasingOnNegatives (DecreasingQuadratic a b) → a > 0 ∧ b ∈ Set.univ := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l2901_290147


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2901_290193

theorem polynomial_simplification (x : ℝ) : 
  (5 * x^10 + 8 * x^9 + 2 * x^8) + (3 * x^10 + x^9 + 4 * x^8 + 7 * x^4 + 6 * x + 9) = 
  8 * x^10 + 9 * x^9 + 6 * x^8 + 7 * x^4 + 6 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2901_290193


namespace NUMINAMATH_CALUDE_simplify_expression_l2901_290100

theorem simplify_expression : Real.sqrt ((Real.pi - 4) ^ 2) + (Real.pi - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2901_290100


namespace NUMINAMATH_CALUDE_expression_evaluation_l2901_290124

theorem expression_evaluation (x y : ℚ) 
  (hx : x = 2 / 15) (hy : y = 3 / 2) : 
  (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2901_290124


namespace NUMINAMATH_CALUDE_f_of_f_of_2_l2901_290163

def f (x : ℝ) : ℝ := 4 * x^2 - 7

theorem f_of_f_of_2 : f (f 2) = 317 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_2_l2901_290163


namespace NUMINAMATH_CALUDE_range_of_f_l2901_290115

def P : Set ℕ := {1, 2, 3}

def f (x : ℕ) : ℕ := 2^x

theorem range_of_f :
  {y | ∃ x ∈ P, f x = y} = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2901_290115


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l2901_290199

def f (x : ℝ) : ℝ := x^3

theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l2901_290199


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2901_290134

theorem decimal_sum_to_fraction :
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 : ℚ) = 22839 / 50000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2901_290134


namespace NUMINAMATH_CALUDE_coin_order_l2901_290180

/-- Represents a coin in the arrangement -/
inductive Coin
| A | B | C | D | E | F

/-- Represents the relative position of two coins -/
inductive Position
| Above | Below

/-- Defines the relationship between two coins -/
def relation (c1 c2 : Coin) : Position := sorry

/-- Theorem stating the correct order of coins from top to bottom -/
theorem coin_order :
  (relation Coin.F Coin.E = Position.Above) ∧
  (relation Coin.F Coin.C = Position.Above) ∧
  (relation Coin.F Coin.D = Position.Above) ∧
  (relation Coin.F Coin.A = Position.Above) ∧
  (relation Coin.F Coin.B = Position.Above) ∧
  (relation Coin.E Coin.C = Position.Above) ∧
  (relation Coin.E Coin.D = Position.Above) ∧
  (relation Coin.E Coin.A = Position.Above) ∧
  (relation Coin.E Coin.B = Position.Above) ∧
  (relation Coin.C Coin.A = Position.Above) ∧
  (relation Coin.C Coin.B = Position.Above) ∧
  (relation Coin.D Coin.A = Position.Above) ∧
  (relation Coin.D Coin.B = Position.Above) ∧
  (relation Coin.A Coin.B = Position.Above) →
  ∀ (c : Coin), c ≠ Coin.F →
    (relation Coin.F c = Position.Above) ∧
    (∀ (d : Coin), d ≠ Coin.B →
      (relation c Coin.B = Position.Above)) :=
by sorry

end NUMINAMATH_CALUDE_coin_order_l2901_290180


namespace NUMINAMATH_CALUDE_midpoint_y_coordinate_l2901_290187

theorem midpoint_y_coordinate (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  let f := λ x : Real => Real.sin x
  let g := λ x : Real => Real.cos x
  let M := (a, f a)
  let N := (a, g a)
  abs (f a - g a) = 1/5 →
  (f a + g a) / 2 = 7/10 := by
sorry

end NUMINAMATH_CALUDE_midpoint_y_coordinate_l2901_290187


namespace NUMINAMATH_CALUDE_weight_difference_l2901_290103

theorem weight_difference (anne_weight douglas_weight : ℕ) 
  (h1 : anne_weight = 67) 
  (h2 : douglas_weight = 52) : 
  anne_weight - douglas_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l2901_290103


namespace NUMINAMATH_CALUDE_projectile_max_height_l2901_290156

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 161 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2901_290156


namespace NUMINAMATH_CALUDE_birdseed_mix_theorem_l2901_290132

/-- The percentage of millet in Brand B -/
def brand_b_millet : ℝ := 0.65

/-- The percentage of Brand A in the mix -/
def mix_brand_a : ℝ := 0.60

/-- The percentage of Brand B in the mix -/
def mix_brand_b : ℝ := 0.40

/-- The percentage of millet in the final mix -/
def final_mix_millet : ℝ := 0.50

/-- The percentage of millet in Brand A -/
def brand_a_millet : ℝ := 0.40

theorem birdseed_mix_theorem :
  mix_brand_a * brand_a_millet + mix_brand_b * brand_b_millet = final_mix_millet :=
by sorry

end NUMINAMATH_CALUDE_birdseed_mix_theorem_l2901_290132


namespace NUMINAMATH_CALUDE_specific_normal_distribution_two_std_devs_less_l2901_290127

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- The value that is exactly 2 standard deviations less than the mean --/
def twoStdDevsLessThanMean (nd : NormalDistribution) : ℝ :=
  nd.μ - 2 * nd.σ

/-- Theorem statement for the given problem --/
theorem specific_normal_distribution_two_std_devs_less (nd : NormalDistribution) 
  (h1 : nd.μ = 16.5) (h2 : nd.σ = 1.5) : 
  twoStdDevsLessThanMean nd = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_normal_distribution_two_std_devs_less_l2901_290127


namespace NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l2901_290112

theorem min_value_sum_of_distances (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2) ≥ 6 * Real.sqrt 2 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l2901_290112


namespace NUMINAMATH_CALUDE_april_roses_unsold_l2901_290129

/-- Calculates the number of roses left unsold given the initial number of roses,
    the price per rose, and the total amount earned from sales. -/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Proves that the number of roses left unsold is 4 given the problem conditions. -/
theorem april_roses_unsold :
  roses_left_unsold 9 7 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_roses_unsold_l2901_290129


namespace NUMINAMATH_CALUDE_second_storm_duration_l2901_290107

/-- Represents the duration and rainfall rate of a rainstorm -/
structure Rainstorm where
  duration : ℝ
  rate : ℝ

/-- Proves that the second rainstorm lasted 25 hours given the conditions -/
theorem second_storm_duration
  (storm1 : Rainstorm)
  (storm2 : Rainstorm)
  (h1 : storm1.rate = 30)
  (h2 : storm2.rate = 15)
  (h3 : storm1.duration + storm2.duration = 45)
  (h4 : storm1.rate * storm1.duration + storm2.rate * storm2.duration = 975) :
  storm2.duration = 25 := by
  sorry

#check second_storm_duration

end NUMINAMATH_CALUDE_second_storm_duration_l2901_290107


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2901_290117

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  (c - b) / (Real.sqrt 2 * c - a) = Real.sin A / (Real.sin B + Real.sin C) →
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2901_290117


namespace NUMINAMATH_CALUDE_initial_men_correct_l2901_290185

/-- The number of men initially working on the digging project. -/
def initial_men : ℕ := 55

/-- The number of hours worked per day in the initial condition. -/
def initial_hours : ℕ := 8

/-- The depth dug in meters in the initial condition. -/
def initial_depth : ℕ := 30

/-- The number of hours worked per day in the new condition. -/
def new_hours : ℕ := 6

/-- The depth to be dug in meters in the new condition. -/
def new_depth : ℕ := 50

/-- The additional number of men needed for the new condition. -/
def extra_men : ℕ := 11

/-- Theorem stating that the initial number of men is correct given the conditions. -/
theorem initial_men_correct :
  initial_men * initial_hours * initial_depth = (initial_men + extra_men) * new_hours * new_depth :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l2901_290185


namespace NUMINAMATH_CALUDE_more_red_polygons_l2901_290131

/-- Represents a set of points on a circle -/
structure PointSet where
  white : ℕ
  red : ℕ

/-- Counts the number of polygons that can be formed from a given set of points -/
def countPolygons (ps : PointSet) (includeRed : Bool) : ℕ :=
  sorry

/-- The given configuration of points -/
def circlePoints : PointSet :=
  { white := 1997, red := 1 }

theorem more_red_polygons :
  countPolygons circlePoints true > countPolygons circlePoints false :=
sorry

end NUMINAMATH_CALUDE_more_red_polygons_l2901_290131


namespace NUMINAMATH_CALUDE_complex_fraction_equals_four_l2901_290146

theorem complex_fraction_equals_four :
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_four_l2901_290146


namespace NUMINAMATH_CALUDE_range_of_m_l2901_290138

-- Define the equations
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) →
  (∀ m : ℝ, m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2901_290138


namespace NUMINAMATH_CALUDE_tina_pen_difference_l2901_290116

/-- Prove that Tina has 3 more blue pens than green pens -/
theorem tina_pen_difference : 
  ∀ (pink green blue : ℕ),
  pink = 12 →
  green = pink - 9 →
  blue > green →
  pink + green + blue = 21 →
  blue - green = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tina_pen_difference_l2901_290116


namespace NUMINAMATH_CALUDE_championship_and_expectation_l2901_290104

-- Define the probabilities of class A winning each event
def p_basketball : ℝ := 0.4
def p_soccer : ℝ := 0.8
def p_badminton : ℝ := 0.6

-- Define the points awarded for winning and losing
def win_points : ℕ := 8
def lose_points : ℕ := 0

-- Define the probability of class A winning the championship
def p_championship : ℝ := 
  p_basketball * p_soccer * p_badminton +
  (1 - p_basketball) * p_soccer * p_badminton +
  p_basketball * (1 - p_soccer) * p_badminton +
  p_basketball * p_soccer * (1 - p_badminton)

-- Define the distribution of class B's total score
def p_score (x : ℕ) : ℝ :=
  if x = 0 then (1 - p_basketball) * (1 - p_soccer) * (1 - p_badminton)
  else if x = win_points then 
    p_basketball * (1 - p_soccer) * (1 - p_badminton) +
    (1 - p_basketball) * p_soccer * (1 - p_badminton) +
    (1 - p_basketball) * (1 - p_soccer) * p_badminton
  else if x = 2 * win_points then
    p_basketball * p_soccer * (1 - p_badminton) +
    p_basketball * (1 - p_soccer) * p_badminton +
    (1 - p_basketball) * p_soccer * p_badminton
  else if x = 3 * win_points then
    p_basketball * p_soccer * p_badminton
  else 0

-- Define the expectation of class B's total score
def expectation_B : ℝ :=
  0 * p_score 0 +
  win_points * p_score win_points +
  (2 * win_points) * p_score (2 * win_points) +
  (3 * win_points) * p_score (3 * win_points)

theorem championship_and_expectation :
  p_championship = 0.656 ∧ expectation_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_championship_and_expectation_l2901_290104


namespace NUMINAMATH_CALUDE_complementary_events_l2901_290144

-- Define the sample space for a die throw
def DieThrow : Type := Fin 6

-- Define event A: upward face shows an odd number
def eventA : Set DieThrow := {x | x.val % 2 = 1}

-- Define event B: upward face shows an even number
def eventB : Set DieThrow := {x | x.val % 2 = 0}

-- Theorem stating that A and B are complementary events
theorem complementary_events : 
  eventA ∪ eventB = Set.univ ∧ eventA ∩ eventB = ∅ := by
  sorry

end NUMINAMATH_CALUDE_complementary_events_l2901_290144


namespace NUMINAMATH_CALUDE_difference_of_squares_divided_l2901_290195

theorem difference_of_squares_divided : (113^2 - 107^2) / 6 = 220 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divided_l2901_290195


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l2901_290167

theorem min_value_quadratic_expression (a b c : ℝ) :
  a < b →
  (∀ x, a * x^2 + b * x + c ≥ 0) →
  (a + b + c) / (b - a) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l2901_290167


namespace NUMINAMATH_CALUDE_route_time_difference_l2901_290113

-- Define the times for each stage of the first route
def first_route_uphill : ℕ := 6
def first_route_path : ℕ := 2 * first_route_uphill
def first_route_final (t : ℕ) : ℕ := t / 3

-- Define the times for each stage of the second route
def second_route_flat : ℕ := 14
def second_route_final : ℕ := 2 * second_route_flat

-- Calculate the total time for the first route
def first_route_total : ℕ := 
  first_route_uphill + first_route_path + first_route_final (first_route_uphill + first_route_path)

-- Calculate the total time for the second route
def second_route_total : ℕ := second_route_flat + second_route_final

-- Theorem stating the difference between the two routes
theorem route_time_difference : second_route_total - first_route_total = 18 := by
  sorry

end NUMINAMATH_CALUDE_route_time_difference_l2901_290113


namespace NUMINAMATH_CALUDE_function_zeros_l2901_290194

def has_at_least_n_zeros (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (S : Finset ℝ), S.card ≥ n ∧ (∀ x ∈ S, a < x ∧ x ≤ b ∧ f x = 0)

theorem function_zeros
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_symmetry : ∀ x, f (2 + x) = f (2 - x))
  (h_zero_in_interval : ∃ x, 0 < x ∧ x < 4 ∧ f x = 0)
  (h_zero_at_origin : f 0 = 0) :
  has_at_least_n_zeros f (-8) 10 9 :=
sorry

end NUMINAMATH_CALUDE_function_zeros_l2901_290194


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l2901_290158

theorem triangle_ratio_proof (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let DC := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  AB = 8 →
  BC = 13 →
  AC = 10 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))) →
  AD = 8 →
  BD / DC = 133 / 36 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l2901_290158


namespace NUMINAMATH_CALUDE_integral_x_squared_minus_x_l2901_290123

theorem integral_x_squared_minus_x : ∫ (x : ℝ) in (0)..(1), (x^2 - x) = -1/6 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_minus_x_l2901_290123


namespace NUMINAMATH_CALUDE_inverse_sum_mod_13_l2901_290174

theorem inverse_sum_mod_13 :
  (((2⁻¹ : ZMod 13) + (5⁻¹ : ZMod 13) + (9⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_13_l2901_290174


namespace NUMINAMATH_CALUDE_flower_bed_side_length_l2901_290153

/-- Given a rectangular flower bed with area 6a^2 - 4ab + 2a and one side of length 2a,
    the length of the other side is 3a - 2b + 1 -/
theorem flower_bed_side_length (a b : ℝ) :
  let area := 6 * a^2 - 4 * a * b + 2 * a
  let side1 := 2 * a
  area / side1 = 3 * a - 2 * b + 1 := by
sorry

end NUMINAMATH_CALUDE_flower_bed_side_length_l2901_290153


namespace NUMINAMATH_CALUDE_paco_cookies_l2901_290165

/-- Calculates the total number of cookies Paco has after buying cookies with a promotion --/
def total_cookies (initial : ℕ) (eaten : ℕ) (bought : ℕ) : ℕ :=
  let remaining := initial - eaten
  let free := 2 * bought
  let from_bakery := bought + free
  remaining + from_bakery

/-- Proves that Paco ends up with 149 cookies given the initial conditions --/
theorem paco_cookies : total_cookies 40 2 37 = 149 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2901_290165


namespace NUMINAMATH_CALUDE_sqrt_4_plus_2_inv_l2901_290189

theorem sqrt_4_plus_2_inv : Real.sqrt 4 + 2⁻¹ = 5/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_4_plus_2_inv_l2901_290189


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_l2901_290148

theorem sum_of_last_two_digits (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_l2901_290148


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l2901_290149

-- Define the given line
def given_line : ℝ → ℝ → Prop :=
  λ x y => x + 2 * y - 1 = 0

-- Define the point that the desired line passes through
def point : ℝ × ℝ := (2, 0)

-- Define the equation of the desired line
def desired_line : ℝ → ℝ → Prop :=
  λ x y => x + 2 * y - 2 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given :
  (desired_line point.1 point.2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, desired_line x y ↔ given_line (x + k) (y + k/2)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l2901_290149


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2901_290152

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 7 * π / 4
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π) ∧
  r = 2 * Real.sqrt 2 ∧
  θ = 7 * π / 4 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2901_290152


namespace NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_l2901_290192

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- The diagonals of a rhombus. -/
def Rhombus.diagonals (r : Rhombus) : Fin 2 → ℝ × ℝ := sorry

/-- Two lines are perpendicular if their dot product is zero. -/
def perpendicular (l1 l2 : ℝ × ℝ) : Prop :=
  l1.1 * l2.1 + l1.2 * l2.2 = 0

/-- The diagonals of a rhombus are always perpendicular to each other. -/
theorem rhombus_diagonals_perpendicular (r : Rhombus) :
  perpendicular (r.diagonals 0) (r.diagonals 1) := by sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_l2901_290192


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l2901_290102

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 7 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l2901_290102


namespace NUMINAMATH_CALUDE_no_integer_sqrt_difference_150_l2901_290162

theorem no_integer_sqrt_difference_150 :
  (∃ (x : ℤ), x - Real.sqrt x = 20) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 30) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 110) ∧
  (∀ (x : ℤ), x - Real.sqrt x ≠ 150) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 600) := by
  sorry

#check no_integer_sqrt_difference_150

end NUMINAMATH_CALUDE_no_integer_sqrt_difference_150_l2901_290162


namespace NUMINAMATH_CALUDE_perimeter_of_z_shape_l2901_290161

-- Define the complex number z
variable (z : ℂ)

-- Define the condition that z satisfies
def satisfies_condition (z : ℂ) : Prop :=
  Complex.arg z = Complex.arg (Complex.I * z + Complex.I)

-- Define the shape corresponding to z
def shape_of_z (z : ℂ) : Set ℂ :=
  {w : ℂ | ∃ t : ℝ, w = t * z ∧ 0 ≤ t ∧ t ≤ 1}

-- Define the perimeter of the shape
def perimeter_of_shape (s : Set ℂ) : ℝ := sorry

-- State the theorem
theorem perimeter_of_z_shape (h : satisfies_condition z) :
  perimeter_of_shape (shape_of_z z) = π / 2 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_z_shape_l2901_290161


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l2901_290122

theorem sin_thirty_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l2901_290122


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2901_290139

theorem remainder_divisibility (x : ℤ) : x % 72 = 19 → x % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2901_290139


namespace NUMINAMATH_CALUDE_triple_composition_even_l2901_290183

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- State the theorem
theorem triple_composition_even
  (g : ℝ → ℝ)
  (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
sorry

end NUMINAMATH_CALUDE_triple_composition_even_l2901_290183


namespace NUMINAMATH_CALUDE_ping_pong_table_distribution_l2901_290145

theorem ping_pong_table_distribution (total_tables : Nat) (total_players : Nat)
  (h_tables : total_tables = 15)
  (h_players : total_players = 38) :
  ∃ (singles_tables doubles_tables : Nat),
    singles_tables + doubles_tables = total_tables ∧
    2 * singles_tables + 4 * doubles_tables = total_players ∧
    singles_tables = 11 ∧
    doubles_tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_table_distribution_l2901_290145


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2901_290176

-- Define the vectors
def a : ℝ × ℝ := (2, 5)
def b : ℝ → ℝ × ℝ := λ y ↦ (1, y)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem parallel_vectors_y_value :
  parallel a (b y) → y = 5/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2901_290176


namespace NUMINAMATH_CALUDE_population_reaches_capacity_l2901_290109

-- Define the constants
def land_area : ℕ := 40000
def acres_per_person : ℕ := 1
def base_population : ℕ := 500
def years_to_quadruple : ℕ := 20

-- Define the population growth function
def population (years : ℕ) : ℕ :=
  base_population * (4 ^ (years / years_to_quadruple))

-- Define the maximum capacity
def max_capacity : ℕ := land_area / acres_per_person

-- Theorem to prove
theorem population_reaches_capacity : 
  population 60 ≥ max_capacity ∧ population 40 < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_capacity_l2901_290109


namespace NUMINAMATH_CALUDE_sine_function_period_l2901_290155

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.sin (b * x + c) + d) →
  (∃ x : ℝ, x = 4 * Real.pi ∧ (x / (2 * Real.pi / b) = 5)) →
  b = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l2901_290155
