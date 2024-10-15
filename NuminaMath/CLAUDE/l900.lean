import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l900_90093

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →  -- Each term is positive
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence definition
  (a 0 + a 1 = 6) →  -- Sum of first two terms is 6
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 126) →  -- Sum of first six terms is 126
  (a 0 + a 1 + a 2 + a 3 = 30) :=  -- Sum of first four terms is 30
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l900_90093


namespace NUMINAMATH_CALUDE_regular_polygon_radius_l900_90020

/-- A regular polygon with the given properties --/
structure RegularPolygon where
  -- Number of sides
  n : ℕ
  -- Side length
  s : ℝ
  -- Radius
  r : ℝ
  -- Sum of interior angles is twice the sum of exterior angles
  interior_sum_twice_exterior : (n - 2) * 180 = 2 * 360
  -- Side length is 2
  side_length_is_two : s = 2

/-- The radius of the regular polygon with the given properties is 2 --/
theorem regular_polygon_radius (p : RegularPolygon) : p.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_radius_l900_90020


namespace NUMINAMATH_CALUDE_max_additional_plates_l900_90005

/-- Represents the number of letters in each set for license plates -/
structure LicensePlateSets :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Calculates the total number of unique license plates -/
def totalPlates (sets : LicensePlateSets) : ℕ :=
  sets.first * sets.second * sets.third

/-- The initial configuration of letter sets -/
def initialSets : LicensePlateSets :=
  ⟨5, 3, 4⟩

/-- The number of new letters to be added -/
def newLetters : ℕ := 2

/-- Theorem: The maximum number of additional unique license plates is 30 -/
theorem max_additional_plates :
  ∃ (newSets : LicensePlateSets),
    (newSets.first + newSets.second + newSets.third = initialSets.first + initialSets.second + initialSets.third + newLetters) ∧
    (∀ (otherSets : LicensePlateSets),
      (otherSets.first + otherSets.second + otherSets.third = initialSets.first + initialSets.second + initialSets.third + newLetters) →
      (totalPlates newSets - totalPlates initialSets ≥ totalPlates otherSets - totalPlates initialSets)) ∧
    (totalPlates newSets - totalPlates initialSets = 30) :=
  sorry

end NUMINAMATH_CALUDE_max_additional_plates_l900_90005


namespace NUMINAMATH_CALUDE_f_symmetric_l900_90094

noncomputable def f (x : ℝ) : ℝ :=
  (6 * Real.cos (Real.pi + x) + 5 * (Real.sin (Real.pi - x))^2 - 4) / Real.cos (2 * Real.pi - x)

theorem f_symmetric (m : ℝ) (h : f m = 2) : f (-m) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_l900_90094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l900_90084

/-- Prove that an arithmetic sequence starting with 13, ending with 73, 
    and having a common difference of 3 has 21 terms. -/
theorem arithmetic_sequence_terms (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 13 → aₙ = 73 → d = 3 → 
  aₙ = a₁ + (n - 1) * d → n = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l900_90084


namespace NUMINAMATH_CALUDE_sequence_general_term_l900_90061

theorem sequence_general_term (a : ℕ+ → ℝ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ+, n * a (n + 1) = (n + 1) * a n + 2) : 
  ∀ n : ℕ+, a n = 4 * n - 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l900_90061


namespace NUMINAMATH_CALUDE_square_difference_l900_90085

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 := by sorry

end NUMINAMATH_CALUDE_square_difference_l900_90085


namespace NUMINAMATH_CALUDE_blue_cards_count_l900_90026

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℚ) (blue_cards : ℕ) : 
  red_cards = 8 →
  blue_prob = 6/10 →
  (blue_cards : ℚ) / (blue_cards + red_cards) = blue_prob →
  blue_cards = 12 := by
sorry

end NUMINAMATH_CALUDE_blue_cards_count_l900_90026


namespace NUMINAMATH_CALUDE_sphere_circular_cross_section_l900_90063

-- Define the types of solids
inductive Solid
| Cylinder
| Cone
| Sphere
| Frustum

-- Define the types of cross-section shapes
inductive CrossSectionShape
| Rectangular
| Triangular
| Circular
| IsoscelesTrapezoid

-- Function to get the cross-section shape of a solid through its axis of rotation
def crossSectionThroughAxis (s : Solid) : CrossSectionShape :=
  match s with
  | Solid.Cylinder => CrossSectionShape.Rectangular
  | Solid.Cone => CrossSectionShape.Triangular
  | Solid.Sphere => CrossSectionShape.Circular
  | Solid.Frustum => CrossSectionShape.IsoscelesTrapezoid

-- Theorem stating that only the Sphere has a circular cross-section through its axis of rotation
theorem sphere_circular_cross_section :
  ∀ s : Solid, crossSectionThroughAxis s = CrossSectionShape.Circular ↔ s = Solid.Sphere :=
by
  sorry


end NUMINAMATH_CALUDE_sphere_circular_cross_section_l900_90063


namespace NUMINAMATH_CALUDE_acme_soup_words_count_l900_90014

/-- Represents the number of times each vowel (A, E, I, O, U) appears -/
def vowel_count : ℕ := 5

/-- Represents the number of times Y appears -/
def y_count : ℕ := 3

/-- Represents the length of words to be formed -/
def word_length : ℕ := 5

/-- Represents the number of vowels (A, E, I, O, U) -/
def num_vowels : ℕ := 5

/-- Calculates the number of five-letter words that can be formed -/
def acme_soup_words : ℕ := 
  (num_vowels ^ word_length) + 
  (word_length * (num_vowels ^ (word_length - 1))) +
  (Nat.choose word_length 2 * (num_vowels ^ (word_length - 2))) +
  (Nat.choose word_length 3 * (num_vowels ^ (word_length - 3)))

theorem acme_soup_words_count : acme_soup_words = 7750 := by
  sorry

end NUMINAMATH_CALUDE_acme_soup_words_count_l900_90014


namespace NUMINAMATH_CALUDE_friendship_theorem_l900_90011

/-- A graph representing friendships in a class -/
structure FriendshipGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  symmetric : ∀ {i j}, (i, j) ∈ edges → (j, i) ∈ edges
  no_self_loops : ∀ i, (i, i) ∉ edges

/-- The degree of a vertex in the graph -/
def degree (G : FriendshipGraph) (v : ℕ) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- A clique in the graph -/
def is_clique (G : FriendshipGraph) (S : Finset ℕ) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → i ≠ j → (i, j) ∈ G.edges

theorem friendship_theorem (G : FriendshipGraph) 
  (h1 : G.vertices.card = 20)
  (h2 : ∀ v ∈ G.vertices, degree G v ≥ 14) :
  ∃ S : Finset ℕ, S.card = 4 ∧ is_clique G S :=
sorry

end NUMINAMATH_CALUDE_friendship_theorem_l900_90011


namespace NUMINAMATH_CALUDE_power_division_rule_l900_90087

theorem power_division_rule (a : ℝ) (ha : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l900_90087


namespace NUMINAMATH_CALUDE_mailing_cost_correct_l900_90025

/-- The cost function for mailing a document -/
def mailing_cost (P : ℕ) : ℕ :=
  if P ≤ 5 then
    15 + 5 * (P - 1)
  else
    15 + 5 * (P - 1) + 2

/-- Theorem stating the correctness of the mailing cost function -/
theorem mailing_cost_correct (P : ℕ) :
  mailing_cost P =
    if P ≤ 5 then
      15 + 5 * (P - 1)
    else
      15 + 5 * (P - 1) + 2 :=
by
  sorry

/-- Lemma: The cost for the first kilogram is 15 cents -/
lemma first_kg_cost (P : ℕ) (h : P > 0) : mailing_cost P ≥ 15 :=
by
  sorry

/-- Lemma: Each subsequent kilogram costs 5 cents -/
lemma subsequent_kg_cost (P : ℕ) (h : P > 1) :
  mailing_cost P - mailing_cost (P - 1) = 5 :=
by
  sorry

/-- Lemma: Additional handling fee of 2 cents for documents over 5 kg -/
lemma handling_fee (P : ℕ) (h : P > 5) :
  mailing_cost P - mailing_cost 5 = 5 * (P - 5) + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_mailing_cost_correct_l900_90025


namespace NUMINAMATH_CALUDE_topsoil_discounted_cost_l900_90079

-- Define constants
def price_per_cubic_foot : ℝ := 7
def purchase_volume_yards : ℝ := 10
def discount_threshold : ℝ := 200
def discount_rate : ℝ := 0.05

-- Define conversion factor
def cubic_yards_to_cubic_feet : ℝ := 27

-- Theorem statement
theorem topsoil_discounted_cost :
  let volume_feet := purchase_volume_yards * cubic_yards_to_cubic_feet
  let base_cost := volume_feet * price_per_cubic_foot
  let discounted_cost := if volume_feet > discount_threshold
                         then base_cost * (1 - discount_rate)
                         else base_cost
  discounted_cost = 1795.50 := by
sorry

end NUMINAMATH_CALUDE_topsoil_discounted_cost_l900_90079


namespace NUMINAMATH_CALUDE_second_part_interest_rate_l900_90080

def total_amount : ℝ := 2500
def first_part : ℝ := 500
def first_rate : ℝ := 0.05
def total_income : ℝ := 145

theorem second_part_interest_rate :
  let second_part := total_amount - first_part
  let first_income := first_part * first_rate
  let second_income := total_income - first_income
  let second_rate := second_income / second_part
  second_rate = 0.06 := by sorry

end NUMINAMATH_CALUDE_second_part_interest_rate_l900_90080


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l900_90029

/-- Given three numbers in an arithmetic sequence with a ratio of 3:4:5,
    if increasing the smallest number by 1 forms a geometric sequence,
    then the original three numbers are 15, 20, and 25. -/
theorem arithmetic_to_geometric_sequence (a d : ℝ) : 
  (a - d : ℝ) / 3 = a / 4 ∧ a / 4 = (a + d) / 5 →  -- arithmetic sequence with ratio 3:4:5
  ∃ r : ℝ, (a - d + 1) / a = a / (a + d) ∧ a / (a + d) = r →  -- geometric sequence after increasing smallest by 1
  a - d = 15 ∧ a = 20 ∧ a + d = 25 := by  -- original numbers are 15, 20, 25
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l900_90029


namespace NUMINAMATH_CALUDE_triangle_inequality_l900_90032

/-- Checks if three lengths can form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ¬(is_valid_triangle a b 15) ∧ is_valid_triangle a b 13 :=
by
  sorry

#check triangle_inequality 8 7

end NUMINAMATH_CALUDE_triangle_inequality_l900_90032


namespace NUMINAMATH_CALUDE_simplify_expression_l900_90030

theorem simplify_expression (r s : ℝ) : 
  (2 * r^2 + 5 * r - 6 * s + 4) - (r^2 + 9 * r - 4 * s - 2) = r^2 - 4 * r - 2 * s + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l900_90030


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l900_90047

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse defined by x^2/9 + y^2/4 = 1 is 6 -/
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, is_ellipse x y → major_axis_length = 6 := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l900_90047


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l900_90081

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse in 2D space -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Check if four points form a trapezoid with bases parallel to x-axis -/
def isTrapezoid (p1 p2 p3 p4 : Point) : Prop :=
  (p1.y = p2.y) ∧ (p3.y = p4.y) ∧ (p1.y ≠ p3.y)

/-- Check if a point lies on the vertical bisector of a trapezoid -/
def onVerticalBisector (p : Point) (p1 p2 p3 p4 : Point) : Prop :=
  p.x = (p1.x + p2.x) / 2

theorem ellipse_major_axis_length
  (p1 p2 p3 p4 p5 : Point)
  (h1 : p1 = ⟨0, 0⟩)
  (h2 : p2 = ⟨4, 0⟩)
  (h3 : p3 = ⟨1, 3⟩)
  (h4 : p4 = ⟨3, 3⟩)
  (h5 : p5 = ⟨-1, 3/2⟩)
  (h_trapezoid : isTrapezoid p1 p2 p3 p4)
  (h_bisector : onVerticalBisector p5 p1 p2 p3 p4)
  (e : Ellipse)
  (h_on_ellipse : pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ pointOnEllipse p3 e ∧ pointOnEllipse p4 e ∧ pointOnEllipse p5 e)
  (h_axes_parallel : e.center.x = (p1.x + p2.x) / 2 ∧ e.center.y = (p1.y + p3.y) / 2) :
  2 * e.a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l900_90081


namespace NUMINAMATH_CALUDE_milk_quality_theorem_l900_90022

/-- The probability of a single bottle of milk being qualified -/
def p_qualified : ℝ := 0.8

/-- The number of bottles bought -/
def n_bottles : ℕ := 2

/-- The number of days considered -/
def n_days : ℕ := 3

/-- The probability that all bought bottles are qualified -/
def prob_all_qualified : ℝ := p_qualified ^ n_bottles

/-- The probability of drinking unqualified milk in a day -/
def p_unqualified_day : ℝ := 1 - p_qualified ^ n_bottles

/-- The expected number of days drinking unqualified milk -/
def expected_unqualified_days : ℝ := n_days * p_unqualified_day

theorem milk_quality_theorem :
  prob_all_qualified = 0.64 ∧ expected_unqualified_days = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_milk_quality_theorem_l900_90022


namespace NUMINAMATH_CALUDE_gcd_of_72_120_180_l900_90051

theorem gcd_of_72_120_180 : Nat.gcd 72 (Nat.gcd 120 180) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_180_l900_90051


namespace NUMINAMATH_CALUDE_total_new_shoes_l900_90088

def pairs_of_shoes : ℕ := 3
def shoes_per_pair : ℕ := 2

theorem total_new_shoes : pairs_of_shoes * shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_new_shoes_l900_90088


namespace NUMINAMATH_CALUDE_justice_ferns_l900_90034

/-- Given the number of palms and succulents Justice has, the total number of plants she wants,
    and the number of additional plants she needs, prove that Justice has 3 ferns. -/
theorem justice_ferns (palms_and_succulents : ℕ) (desired_total : ℕ) (additional_needed : ℕ)
  (h1 : palms_and_succulents = 12)
  (h2 : desired_total = 24)
  (h3 : additional_needed = 9) :
  desired_total - additional_needed - palms_and_succulents = 3 :=
by sorry

end NUMINAMATH_CALUDE_justice_ferns_l900_90034


namespace NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l900_90097

/-- A linear function passing through (-1, 2) with y-intercept 4 -/
def f (x : ℝ) : ℝ := 2 * x + 4

theorem linear_function_x_axis_intersection :
  ∃ (x : ℝ), f x = 0 ∧ x = -2 := by
  sorry

#check linear_function_x_axis_intersection

end NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l900_90097


namespace NUMINAMATH_CALUDE_min_blue_beads_l900_90008

/-- Represents a necklace with red and blue beads. -/
structure Necklace :=
  (red_beads : ℕ)
  (blue_beads : ℕ)

/-- Checks if a necklace satisfies the condition that any segment
    containing 10 red beads also contains at least 7 blue beads. -/
def satisfies_condition (n : Necklace) : Prop :=
  ∀ (segment : List (Bool)), 
    segment.length ≤ n.red_beads + n.blue_beads →
    (segment.filter id).length = 10 →
    (segment.filter not).length ≥ 7

/-- The main theorem: The minimum number of blue beads in a necklace
    with 100 red beads that satisfies the given condition is 78. -/
theorem min_blue_beads :
  ∃ (n : Necklace), 
    n.red_beads = 100 ∧ 
    satisfies_condition n ∧
    n.blue_beads = 78 ∧
    (∀ (m : Necklace), m.red_beads = 100 → satisfies_condition m → m.blue_beads ≥ 78) :=
by sorry

end NUMINAMATH_CALUDE_min_blue_beads_l900_90008


namespace NUMINAMATH_CALUDE_two_prime_pairs_sum_to_100_l900_90092

def isPrime (n : ℕ) : Prop := sorry

theorem two_prime_pairs_sum_to_100 : 
  ∃! (count : ℕ), ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ S ↔ isPrime p ∧ isPrime q ∧ p + q = 100 ∧ p ≤ q) ∧
    S.card = count ∧
    count = 2 :=
sorry

end NUMINAMATH_CALUDE_two_prime_pairs_sum_to_100_l900_90092


namespace NUMINAMATH_CALUDE_equation_solution_l900_90018

theorem equation_solution : ∃ (x : ℚ), 5*x - 3*x = 420 - 10*(x + 2) ∧ x = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l900_90018


namespace NUMINAMATH_CALUDE_min_value_theorem_l900_90082

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 2) :
  (2 / a + 4 / b) ≥ 14 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 2 ∧ 2 / a₀ + 4 / b₀ = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l900_90082


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_ten_l900_90054

theorem product_of_four_consecutive_integers_divisible_by_ten (n : ℕ) (h : n % 2 = 1) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_ten_l900_90054


namespace NUMINAMATH_CALUDE_exists_bisecting_line_l900_90050

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We don't define the internal structure of ConvexFigure,
  -- as it's not necessary for the statement of the theorem

/-- A line in the plane -/
structure Line where
  -- We don't define the internal structure of Line,
  -- as it's not necessary for the statement of the theorem

/-- The perimeter of a convex figure -/
noncomputable def perimeter (F : ConvexFigure) : ℝ :=
  sorry

/-- The area of a convex figure -/
noncomputable def area (F : ConvexFigure) : ℝ :=
  sorry

/-- Predicate to check if a line bisects the perimeter of a convex figure -/
def bisects_perimeter (l : Line) (F : ConvexFigure) : Prop :=
  sorry

/-- Predicate to check if a line bisects the area of a convex figure -/
def bisects_area (l : Line) (F : ConvexFigure) : Prop :=
  sorry

/-- Theorem: For any convex figure in the plane, there exists a line that
    simultaneously bisects both its perimeter and area -/
theorem exists_bisecting_line (F : ConvexFigure) :
  ∃ l : Line, bisects_perimeter l F ∧ bisects_area l F :=
sorry

end NUMINAMATH_CALUDE_exists_bisecting_line_l900_90050


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l900_90013

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l900_90013


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l900_90073

/-- Given points A(-3,8) and B(2,2), prove that M(1,0) on the x-axis minimizes |AM| + |BM| -/
theorem minimize_sum_distances (A B M : ℝ × ℝ) : 
  A = (-3, 8) → 
  B = (2, 2) → 
  M.2 = 0 → 
  M = (1, 0) → 
  ∀ P : ℝ × ℝ, P.2 = 0 → 
    Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) + Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) ≤ 
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_minimize_sum_distances_l900_90073


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l900_90066

/-- Given a circle with circumference 100 meters, the length of an arc subtended by a central angle of 45° is 12.5 meters. -/
theorem arc_length_45_degrees (D : Real) (arc_EF : Real) :
  D = 100 → -- Circumference of circle D is 100 meters
  arc_EF = D * (45 / 360) → -- Arc length is proportional to the central angle
  arc_EF = 12.5 := by
sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l900_90066


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l900_90046

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 3 * Real.sin θ * Real.cos θ

-- Define the Cartesian equation of a circle
def is_circle (x y : ℝ) : Prop := ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∀ (x y : ℝ), (∃ (r θ : ℝ), polar_equation r θ ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  is_circle x y :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l900_90046


namespace NUMINAMATH_CALUDE_geometric_sequence_k_value_l900_90006

/-- Given a geometric sequence {a_n} with a₂ = 3, a₃ = 9, and a_k = 243, prove that k = 6 -/
theorem geometric_sequence_k_value (a : ℕ → ℝ) (k : ℕ) :
  (∀ n : ℕ, a (n + 1) / a n = a 3 / a 2) →  -- geometric sequence condition
  a 2 = 3 →
  a 3 = 9 →
  a k = 243 →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_k_value_l900_90006


namespace NUMINAMATH_CALUDE_number_composition_l900_90042

/-- A number composed of hundreds, tens, ones, and hundredths -/
def compose_number (hundreds tens ones hundredths : ℕ) : ℚ :=
  (hundreds * 100 + tens * 10 + ones : ℚ) + (hundredths : ℚ) / 100

theorem number_composition :
  compose_number 3 4 6 8 = 346.08 := by
  sorry

end NUMINAMATH_CALUDE_number_composition_l900_90042


namespace NUMINAMATH_CALUDE_sarahs_age_l900_90078

theorem sarahs_age (game_formula : ℕ → ℕ) (name_letters : ℕ) (marriage_age : ℕ) :
  game_formula name_letters = marriage_age →
  name_letters = 5 →
  marriage_age = 23 →
  ∃ current_age : ℕ, game_formula 5 = 23 ∧ current_age = 9 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_age_l900_90078


namespace NUMINAMATH_CALUDE_apples_pears_equivalence_l900_90070

-- Define the relationship between apples and pears
def apples_to_pears (apples : ℚ) : ℚ :=
  (10 / 6) * apples

-- Theorem statement
theorem apples_pears_equivalence :
  apples_to_pears (3/4 * 6) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_apples_pears_equivalence_l900_90070


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l900_90083

theorem fraction_equation_solution (x : ℚ) : 
  (x + 11) / (x - 4) = (x - 3) / (x + 6) → x = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l900_90083


namespace NUMINAMATH_CALUDE_male_sample_size_in_given_scenario_l900_90072

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  female_count : ℕ
  sample_size : ℕ
  h_female_count : female_count ≤ total_population
  h_sample_size : sample_size ≤ total_population

/-- Calculates the number of male students to be drawn in a stratified sample -/
def male_sample_size (s : StratifiedSample) : ℕ :=
  ((s.total_population - s.female_count) * s.sample_size) / s.total_population

/-- Theorem stating the number of male students to be drawn in the given scenario -/
theorem male_sample_size_in_given_scenario :
  let s : StratifiedSample := {
    total_population := 900,
    female_count := 400,
    sample_size := 45,
    h_female_count := by norm_num,
    h_sample_size := by norm_num
  }
  male_sample_size s = 25 := by
  sorry

end NUMINAMATH_CALUDE_male_sample_size_in_given_scenario_l900_90072


namespace NUMINAMATH_CALUDE_min_abs_z_plus_two_l900_90067

open Complex

theorem min_abs_z_plus_two (z : ℂ) (h : (z * (1 + I)).im = 0) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ (w : ℂ), (w * (1 + I)).im = 0 → Complex.abs (w + 2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_two_l900_90067


namespace NUMINAMATH_CALUDE_car_braking_distance_l900_90052

/-- Represents the distance traveled by a car during braking -/
def distance_traveled (initial_speed : ℕ) (deceleration : ℕ) : ℕ :=
  let stopping_time := initial_speed / deceleration
  (initial_speed * stopping_time) - (deceleration * stopping_time * (stopping_time - 1) / 2)

/-- Theorem: A car with initial speed 40 ft/s and deceleration 10 ft/s² travels 100 ft before stopping -/
theorem car_braking_distance :
  distance_traveled 40 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_car_braking_distance_l900_90052


namespace NUMINAMATH_CALUDE_decimal_521_to_octal_l900_90096

-- Define a function to convert decimal to octal
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

-- Theorem statement
theorem decimal_521_to_octal :
  decimal_to_octal 521 = [1, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_decimal_521_to_octal_l900_90096


namespace NUMINAMATH_CALUDE_fourth_term_value_l900_90076

def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

theorem fourth_term_value : ∃ (a : ℕ+ → ℤ), a 4 = 11 :=
  sorry

end NUMINAMATH_CALUDE_fourth_term_value_l900_90076


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l900_90007

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 3 / 4 ∧ 
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 4 / 5 → 
  n + k = 55 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l900_90007


namespace NUMINAMATH_CALUDE_inequality_proof_l900_90003

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 * (y*z + z*x + x*y)^2 ≤ 3*(y^2 + y*z + z^2)*(z^2 + z*x + x^2)*(x^2 + x*y + y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l900_90003


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l900_90095

/-- The length factor after n days of growth -/
def length_factor (n : ℕ) : ℚ :=
  (n + 2 : ℚ) / 2

theorem barry_sotter_magic (n : ℕ) :
  length_factor n = 50 ↔ n = 98 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l900_90095


namespace NUMINAMATH_CALUDE_range_of_a_given_max_value_l900_90041

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - a| + a

/-- The theorem stating the range of a given the maximum value of f(x) -/
theorem range_of_a_given_max_value :
  (∃ (a : ℝ), (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 3) ∧ 
   (∃ x ∈ Set.Icc (-1) 3, f a x = 3)) ↔ 
  (∀ a : ℝ, a ≤ -1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_given_max_value_l900_90041


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l900_90057

/-- The rock collecting contest between Sydney and Conner --/
theorem rock_collecting_contest 
  (sydney_start : ℕ) 
  (conner_start : ℕ) 
  (sydney_day1 : ℕ) 
  (conner_day1_multiplier : ℕ) 
  (sydney_day2 : ℕ) 
  (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ)
  (h1 : sydney_start = 837)
  (h2 : conner_start = 723)
  (h3 : sydney_day1 = 4)
  (h4 : conner_day1_multiplier = 8)
  (h5 : sydney_day2 = 0)
  (h6 : conner_day2 = 123)
  (h7 : sydney_day3_multiplier = 2) :
  ∃ conner_day3 : ℕ, 
    conner_day3 ≥ 27 ∧ 
    conner_start + conner_day1_multiplier * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_start + sydney_day1 + sydney_day2 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) :=
by sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l900_90057


namespace NUMINAMATH_CALUDE_tim_payment_proof_l900_90075

/-- Represents the available bills Tim has -/
structure AvailableBills where
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the minimum number of bills needed to pay a given amount -/
def minBillsNeeded (bills : AvailableBills) (amount : Nat) : Nat :=
  sorry

theorem tim_payment_proof (bills : AvailableBills) (amount : Nat) :
  bills.tens = 13 ∧ bills.fives = 11 ∧ bills.ones = 17 ∧ amount = 128 →
  minBillsNeeded bills amount = 16 := by
  sorry

end NUMINAMATH_CALUDE_tim_payment_proof_l900_90075


namespace NUMINAMATH_CALUDE_bob_has_winning_strategy_l900_90017

/-- Represents the state of the game board -/
structure GameState where
  value : Nat

/-- Represents a player's move -/
inductive Move
  | Bob (a : Nat)
  | Alice (k : Nat)

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Bob a => ⟨state.value - a^2⟩
  | Move.Alice k => ⟨state.value^k⟩

/-- Defines a winning sequence of moves for Bob -/
def WinningSequence (initialState : GameState) (moves : List Move) : Prop :=
  moves.foldl applyMove initialState = ⟨0⟩

/-- The main theorem stating Bob's winning strategy exists -/
theorem bob_has_winning_strategy :
  ∀ (initialState : GameState), initialState.value > 0 →
  ∃ (moves : List Move), WinningSequence initialState moves :=
sorry


end NUMINAMATH_CALUDE_bob_has_winning_strategy_l900_90017


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l900_90023

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : |a 1| = 1)
  (h_a5_a2 : a 5 = -8 * a 2)
  (h_a5_gt_a2 : a 5 > a 2) :
  ∃ r : ℝ, r = -2 ∧ ∀ n : ℕ, a n = r^(n-1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l900_90023


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l900_90074

theorem quadratic_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0) ↔ (a = -1 ∨ a = 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l900_90074


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l900_90044

-- Define a triangle as three points in a 2D plane
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the area of a triangle
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_inequality (A B C P Q R : ℝ × ℝ) :
  Triangle A B C →
  PointOnSegment P B C →
  PointOnSegment Q C A →
  PointOnSegment R A B →
  min (TriangleArea A Q R) (min (TriangleArea B R P) (TriangleArea C P Q)) ≤ (1/4) * TriangleArea A B C :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l900_90044


namespace NUMINAMATH_CALUDE_count_equal_to_one_l900_90035

theorem count_equal_to_one : 
  let numbers := [(-1)^2, (-1)^3, -(1^2), |(-1)|, -(-1), 1/(-1)]
  (numbers.filter (λ x => x = 1)).length = 3 := by
sorry

end NUMINAMATH_CALUDE_count_equal_to_one_l900_90035


namespace NUMINAMATH_CALUDE_negation_equivalence_l900_90038

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ x^2 > 3) ↔ (∀ x : ℝ, x ≥ 0 → x^2 ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l900_90038


namespace NUMINAMATH_CALUDE_greatest_ratio_bound_l900_90028

theorem greatest_ratio_bound (x y z u : ℕ+) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) :
  (x : ℝ) / y ≤ 3 + 2 * Real.sqrt 2 ∧ ∃ (x' y' z' u' : ℕ+), 
    x' + y' = z' + u' ∧ 
    2 * x' * y' = z' * u' ∧ 
    x' ≥ y' ∧ 
    (x' : ℝ) / y' = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_greatest_ratio_bound_l900_90028


namespace NUMINAMATH_CALUDE_tanker_fill_time_l900_90065

/-- Proves that two pipes with filling rates of 1/30 and 1/15 of a tanker per hour
    will fill the tanker in 10 hours when used simultaneously. -/
theorem tanker_fill_time (fill_time_A fill_time_B : ℝ) 
  (h_A : fill_time_A = 30) 
  (h_B : fill_time_B = 15) : 
  1 / (1 / fill_time_A + 1 / fill_time_B) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tanker_fill_time_l900_90065


namespace NUMINAMATH_CALUDE_sin_90_degrees_l900_90043

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l900_90043


namespace NUMINAMATH_CALUDE_journey_end_day_l900_90010

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the next day of the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

/-- Calculates the arrival day given a starting day and journey duration in hours -/
def arrivalDay (startDay : Day) (journeyHours : Nat) : Day :=
  let daysPassed := journeyHours / 24
  (List.range daysPassed).foldl (fun d _ => nextDay d) startDay

/-- Theorem: A 28-hour journey starting on Tuesday will end on Wednesday -/
theorem journey_end_day :
  arrivalDay Day.Tuesday 28 = Day.Wednesday := by
  sorry


end NUMINAMATH_CALUDE_journey_end_day_l900_90010


namespace NUMINAMATH_CALUDE_unique_prime_perfect_power_l900_90089

def is_perfect_power (x : ℕ) : Prop :=
  ∃ m n, m > 1 ∧ n ≥ 2 ∧ x = m^n

theorem unique_prime_perfect_power :
  ∀ p : ℕ, p ≤ 1000 → Prime p → is_perfect_power (2*p + 1) → p = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_power_l900_90089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l900_90039

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₄ = -4 and a₈ = 4, a₁₂ = 12 -/
theorem arithmetic_sequence_property (a : ℕ → ℤ) 
  (h_arith : ArithmeticSequence a) 
  (h_a4 : a 4 = -4) 
  (h_a8 : a 8 = 4) : 
  a 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l900_90039


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l900_90086

theorem algebraic_expression_value (x y : ℝ) : 
  x - 2*y + 1 = 3 → 2*x - 4*y + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l900_90086


namespace NUMINAMATH_CALUDE_pete_susan_speed_ratio_l900_90015

/-- Given the walking and cartwheel speeds of Pete, Susan, and Tracy, prove that the ratio of Pete's backward walking speed to Susan's forward walking speed is 3. -/
theorem pete_susan_speed_ratio :
  ∀ (pete_backward pete_hands tracy_cartwheel susan_forward : ℝ),
  pete_hands > 0 →
  pete_backward > 0 →
  tracy_cartwheel > 0 →
  susan_forward > 0 →
  tracy_cartwheel = 2 * susan_forward →
  pete_hands = (1 / 4) * tracy_cartwheel →
  pete_hands = 2 →
  pete_backward = 12 →
  pete_backward / susan_forward = 3 := by
sorry

end NUMINAMATH_CALUDE_pete_susan_speed_ratio_l900_90015


namespace NUMINAMATH_CALUDE_books_bought_at_yard_sale_l900_90009

def initial_books : ℕ := 35
def final_books : ℕ := 91

theorem books_bought_at_yard_sale :
  final_books - initial_books = 56 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_at_yard_sale_l900_90009


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l900_90000

theorem smaller_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => x^2 - 13*x + 36
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) ∧
  x₁ = 4 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l900_90000


namespace NUMINAMATH_CALUDE_area_of_problem_shape_l900_90024

/-- A composite shape with right-angled corners -/
structure CompositeShape :=
  (height1 : ℕ)
  (width1 : ℕ)
  (height2 : ℕ)
  (width2 : ℕ)
  (height3 : ℕ)
  (width3 : ℕ)

/-- Calculate the area of the composite shape -/
def area (shape : CompositeShape) : ℕ :=
  shape.height1 * shape.width1 +
  shape.height2 * shape.width2 +
  shape.height3 * shape.width3

/-- The specific shape from the problem -/
def problem_shape : CompositeShape :=
  { height1 := 8
  , width1 := 4
  , height2 := 6
  , width2 := 4
  , height3 := 5
  , width3 := 3 }

theorem area_of_problem_shape :
  area problem_shape = 71 :=
by sorry

end NUMINAMATH_CALUDE_area_of_problem_shape_l900_90024


namespace NUMINAMATH_CALUDE_distance_between_points_l900_90098

-- Define the complex numbers
def z_J : ℂ := 3 + 4 * Complex.I
def z_G : ℂ := 2 - 3 * Complex.I

-- Define the scaled version of Gracie's point
def scaled_z_G : ℂ := 2 * z_G

-- Theorem statement
theorem distance_between_points : Complex.abs (z_J - scaled_z_G) = Real.sqrt 101 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l900_90098


namespace NUMINAMATH_CALUDE_original_statement_converse_is_false_inverse_is_false_neither_converse_nor_inverse_true_l900_90090

-- Define the properties of triangles
def is_equilateral (t : Triangle) : Prop := sorry
def is_isosceles (t : Triangle) : Prop := sorry

-- The original statement
theorem original_statement (t : Triangle) : is_equilateral t → is_isosceles t := sorry

-- The converse is false
theorem converse_is_false : ¬(∀ t : Triangle, is_isosceles t → is_equilateral t) := sorry

-- The inverse is false
theorem inverse_is_false : ¬(∀ t : Triangle, ¬is_equilateral t → ¬is_isosceles t) := sorry

-- Main theorem: Neither the converse nor the inverse is true
theorem neither_converse_nor_inverse_true : 
  (¬(∀ t : Triangle, is_isosceles t → is_equilateral t)) ∧ 
  (¬(∀ t : Triangle, ¬is_equilateral t → ¬is_isosceles t)) := sorry

end NUMINAMATH_CALUDE_original_statement_converse_is_false_inverse_is_false_neither_converse_nor_inverse_true_l900_90090


namespace NUMINAMATH_CALUDE_ABCDE_binary_digits_l900_90071

-- Define the base-16 number ABCDE₁₆
def ABCDE : ℕ := 10 * 16^4 + 11 * 16^3 + 12 * 16^2 + 13 * 16^1 + 14

-- Theorem stating that ABCDE₁₆ has 20 binary digits
theorem ABCDE_binary_digits : 
  2^19 ≤ ABCDE ∧ ABCDE < 2^20 :=
by sorry

end NUMINAMATH_CALUDE_ABCDE_binary_digits_l900_90071


namespace NUMINAMATH_CALUDE_division_problem_l900_90036

theorem division_problem : 12 / (2 / (5 - 3)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l900_90036


namespace NUMINAMATH_CALUDE_division_remainder_l900_90091

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 15 →
  divisor = 3 →
  quotient = 4 →
  remainder = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l900_90091


namespace NUMINAMATH_CALUDE_zu_chongzhi_pi_calculation_l900_90040

/-- Represents a historical mathematician -/
structure Mathematician where
  name : String
  calculating_pi : Bool
  decimal_places : ℕ
  father_of_pi : Bool

/-- The mathematician who calculated π to the 9th decimal place in ancient China -/
def ancient_chinese_pi_calculator : Mathematician :=
  { name := "Zu Chongzhi",
    calculating_pi := true,
    decimal_places := 9,
    father_of_pi := true }

/-- Theorem stating that Zu Chongzhi calculated π to the 9th decimal place and is known as the "Father of π" -/
theorem zu_chongzhi_pi_calculation :
  ∃ (m : Mathematician), m.calculating_pi ∧ m.decimal_places = 9 ∧ m.father_of_pi ∧ m.name = "Zu Chongzhi" :=
by
  sorry

end NUMINAMATH_CALUDE_zu_chongzhi_pi_calculation_l900_90040


namespace NUMINAMATH_CALUDE_rug_coverage_l900_90016

theorem rug_coverage (rug_length : ℝ) (rug_width : ℝ) (floor_area : ℝ) 
  (h1 : rug_length = 2)
  (h2 : rug_width = 7)
  (h3 : floor_area = 64)
  (h4 : rug_length * rug_width ≤ floor_area) : 
  (floor_area - rug_length * rug_width) / floor_area = 25 / 32 := by
  sorry

end NUMINAMATH_CALUDE_rug_coverage_l900_90016


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l900_90001

/-- Represents the atomic weights of elements in g/mol -/
def atomic_weight : String → ℝ
  | "C" => 12.01
  | "H" => 1.008
  | "O" => 16.00
  | _ => 0

/-- Calculates the total mass of a given number of atoms of an element -/
def total_mass (element : String) (num_atoms : ℕ) : ℝ :=
  (atomic_weight element) * (num_atoms : ℝ)

/-- Represents the molecular composition of the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℝ

/-- Calculates the total mass of the compound based on its composition -/
def compound_mass (c : Compound) : ℝ :=
  total_mass "C" c.carbon + total_mass "H" c.hydrogen + total_mass "O" c.oxygen

/-- The theorem to be proved -/
theorem compound_oxygen_atoms (c : Compound) 
  (h1 : c.carbon = 3)
  (h2 : c.hydrogen = 6)
  (h3 : c.molecular_weight = 58) :
  c.oxygen = 1 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l900_90001


namespace NUMINAMATH_CALUDE_f_properties_l900_90037

/-- A function f that is constant for all x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1/a| + |x - a + 1|

theorem f_properties (a : ℝ) (h_a : a > 0) (h_const : ∀ x y, f a x = f a y) :
  (∀ x, f a x ≥ 1) ∧
  (f a 3 < 11/2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l900_90037


namespace NUMINAMATH_CALUDE_base_eight_sum_theorem_l900_90077

/-- Converts a three-digit number in base 8 to its decimal representation -/
def baseEightToDecimal (a b c : ℕ) : ℕ := 64 * a + 8 * b + c

/-- Checks if a number is a valid non-zero digit in base 8 -/
def isValidBaseEightDigit (n : ℕ) : Prop := 0 < n ∧ n < 8

theorem base_eight_sum_theorem (A B C : ℕ) 
  (hA : isValidBaseEightDigit A) 
  (hB : isValidBaseEightDigit B) 
  (hC : isValidBaseEightDigit C) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : baseEightToDecimal A B C + baseEightToDecimal B C A + baseEightToDecimal C A B = baseEightToDecimal A A A) : 
  A + B + C = 8 := by
sorry

end NUMINAMATH_CALUDE_base_eight_sum_theorem_l900_90077


namespace NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l900_90027

/-- A star polygon with n points, skipping k points between connections -/
structure StarPolygon where
  n : ℕ  -- number of points
  k : ℕ  -- number of points skipped

/-- The sum of angles at the tips of a star polygon -/
def sumOfTipAngles (star : StarPolygon) : ℝ :=
  sorry

/-- Theorem: The sum of angles at the tips of a 9-pointed star, skipping 3 points, is 720° -/
theorem nine_pointed_star_angle_sum :
  let star : StarPolygon := { n := 9, k := 3 }
  sumOfTipAngles star = 720 := by
  sorry

end NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l900_90027


namespace NUMINAMATH_CALUDE_total_squares_count_l900_90012

/-- Represents a point in the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a square in the grid --/
structure GridSquare where
  topLeft : GridPoint
  size : ℕ

/-- The set of all points in the grid, including additional points --/
def gridPoints : Set GridPoint := sorry

/-- Checks if a square is valid within the given grid --/
def isValidSquare (square : GridSquare) : Prop := sorry

/-- Counts the number of valid squares of a given size --/
def countValidSquares (size : ℕ) : ℕ := sorry

/-- The main theorem to prove --/
theorem total_squares_count :
  (countValidSquares 1) + (countValidSquares 2) = 59 := by sorry

end NUMINAMATH_CALUDE_total_squares_count_l900_90012


namespace NUMINAMATH_CALUDE_pool_cover_radius_increase_l900_90048

/-- Theorem: When a circular pool cover's circumference increases from 30 inches to 40 inches, 
    the radius increases by 5/π inches. -/
theorem pool_cover_radius_increase (r₁ r₂ : ℝ) : 
  2 * Real.pi * r₁ = 30 → 
  2 * Real.pi * r₂ = 40 → 
  r₂ - r₁ = 5 / Real.pi := by
sorry

end NUMINAMATH_CALUDE_pool_cover_radius_increase_l900_90048


namespace NUMINAMATH_CALUDE_exists_simultaneous_j_half_no_universal_j_half_l900_90045

/-- A number is a j-half if it leaves a remainder of j when divided by 2j+1 -/
def is_j_half (n j : ℕ) : Prop := n % (2 * j + 1) = j

/-- For any positive integer k, there exists a number that is simultaneously a j-half for j = 1, 2, ..., k -/
theorem exists_simultaneous_j_half (k : ℕ) : ∃ n : ℕ, ∀ j ∈ Finset.range k, is_j_half n j := by sorry

/-- There is no number which is a j-half for all positive integers j -/
theorem no_universal_j_half : ¬∃ n : ℕ, ∀ j : ℕ, j > 0 → is_j_half n j := by sorry

end NUMINAMATH_CALUDE_exists_simultaneous_j_half_no_universal_j_half_l900_90045


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l900_90064

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a) 
  (h_sum : a 2 + a 3 + a 7 = 6) : 
  a 1 + a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l900_90064


namespace NUMINAMATH_CALUDE_circle_common_chord_l900_90069

/-- Given two circles with equations x^2 + y^2 = a^2 and x^2 + y^2 + ay - 6 = 0,
    where the common chord length is 2√3, prove that a = ±2 -/
theorem circle_common_chord (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = a^2) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + a*y - 6 = 0) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = a^2 ∧ 
    x₁^2 + y₁^2 + a*y₁ - 6 = 0 ∧
    x₂^2 + y₂^2 = a^2 ∧ 
    x₂^2 + y₂^2 + a*y₂ - 6 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →
  a = 2 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_circle_common_chord_l900_90069


namespace NUMINAMATH_CALUDE_T_properties_l900_90031

-- Define the operation T
def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

-- State the theorem
theorem T_properties (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : T a b 2 1 = 2) (h2 : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧ 
  (∀ m n, n ≠ -2 → T a b m n = 0 → m = 4 / (n + 2)) ∧
  (∀ k x y, (∀ k', T a b (k' * x) y = T a b (k * x) y) → y = -2) ∧
  (∀ x y : ℚ, (∀ k, T a b (k * x) y = T a b (k * y) x) → k = 0) :=
by sorry

end NUMINAMATH_CALUDE_T_properties_l900_90031


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l900_90021

theorem simplify_nested_expression (x : ℝ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l900_90021


namespace NUMINAMATH_CALUDE_tops_and_chudis_problem_l900_90049

/-- The price of tops and chudis problem -/
theorem tops_and_chudis_problem (C T : ℚ) : 
  (3 * C + 6 * T = 1500) →  -- Price of 3 chudis and 6 tops
  (C + 12 * T = 1500) →     -- Price of 1 chudi and 12 tops
  (500 / T = 5) :=          -- Number of tops for Rs. 500
by
  sorry

#check tops_and_chudis_problem

end NUMINAMATH_CALUDE_tops_and_chudis_problem_l900_90049


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l900_90056

/-- The center coordinates of a circle with equation (x-h)^2 + (y-k)^2 = r^2 are (h, k) -/
theorem circle_center_coordinates (h k r : ℝ) :
  let circle_equation := fun (x y : ℝ) ↦ (x - h)^2 + (y - k)^2 = r^2
  circle_equation = fun (x y : ℝ) ↦ (x - 2)^2 + (y + 3)^2 = 1 →
  (h, k) = (2, -3) := by
  sorry

#check circle_center_coordinates

end NUMINAMATH_CALUDE_circle_center_coordinates_l900_90056


namespace NUMINAMATH_CALUDE_investment_return_is_25_percent_l900_90033

/-- Calculates the percentage return on investment for a given dividend rate, face value, and purchase price of shares. -/
def percentage_return_on_investment (dividend_rate : ℚ) (face_value : ℚ) (purchase_price : ℚ) : ℚ :=
  (dividend_rate * face_value / purchase_price) * 100

/-- Theorem stating that for the given conditions, the percentage return on investment is 25%. -/
theorem investment_return_is_25_percent :
  let dividend_rate : ℚ := 125 / 1000
  let face_value : ℚ := 60
  let purchase_price : ℚ := 30
  percentage_return_on_investment dividend_rate face_value purchase_price = 25 := by
sorry

#eval percentage_return_on_investment (125/1000) 60 30

end NUMINAMATH_CALUDE_investment_return_is_25_percent_l900_90033


namespace NUMINAMATH_CALUDE_solve_salary_problem_l900_90068

def salary_problem (salaries : List ℝ) (mean : ℝ) : Prop :=
  let n : ℕ := salaries.length + 1
  let total : ℝ := mean * n
  let sum_known : ℝ := salaries.sum
  let sixth_salary : ℝ := total - sum_known
  salaries.length = 5 ∧ 
  mean = 2291.67 ∧
  sixth_salary = 2000.02

theorem solve_salary_problem (salaries : List ℝ) (mean : ℝ) 
  (h1 : salaries = [1000, 2500, 3100, 3650, 1500]) 
  (h2 : mean = 2291.67) : 
  salary_problem salaries mean := by
  sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l900_90068


namespace NUMINAMATH_CALUDE_smallest_possible_d_l900_90004

theorem smallest_possible_d : ∃ d : ℝ,
  (∀ d' : ℝ, d' ≥ 0 → (4 * Real.sqrt 3) ^ 2 + (d' - 2) ^ 2 = (4 * d') ^ 2 → d ≤ d') ∧
  (4 * Real.sqrt 3) ^ 2 + (d - 2) ^ 2 = (4 * d) ^ 2 ∧
  d = 26 / 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l900_90004


namespace NUMINAMATH_CALUDE_existence_of_m_l900_90002

theorem existence_of_m (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h_cd : c * d = 1) :
  ∃ m : ℕ, (0 < m) ∧ (a * b ≤ m^2) ∧ (m^2 ≤ (a + c) * (b + d)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l900_90002


namespace NUMINAMATH_CALUDE_chocolate_savings_bernie_savings_l900_90059

/-- Calculates the savings when buying chocolates at a lower price over a given period -/
theorem chocolate_savings 
  (weeks : ℕ) 
  (chocolates_per_week : ℕ) 
  (price_local : ℚ) 
  (price_discount : ℚ) :
  weeks * chocolates_per_week * (price_local - price_discount) = 
  weeks * chocolates_per_week * price_local - weeks * chocolates_per_week * price_discount :=
by sorry

/-- Proves that Bernie saves $6 over three weeks by buying chocolates at the discounted store -/
theorem bernie_savings :
  let weeks : ℕ := 3
  let chocolates_per_week : ℕ := 2
  let price_local : ℚ := 3
  let price_discount : ℚ := 2
  weeks * chocolates_per_week * (price_local - price_discount) = 6 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_savings_bernie_savings_l900_90059


namespace NUMINAMATH_CALUDE_electric_guitars_sold_l900_90055

theorem electric_guitars_sold (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (acoustic_price : ℕ) 
  (h1 : total_guitars = 9)
  (h2 : total_revenue = 3611)
  (h3 : electric_price = 479)
  (h4 : acoustic_price = 339) :
  ∃ (x : ℕ), x = 4 ∧ 
    ∃ (y : ℕ), x + y = total_guitars ∧ 
    electric_price * x + acoustic_price * y = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_electric_guitars_sold_l900_90055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l900_90062

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_4 + a_8 = 16, a_2 + a_10 = 16 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 16) : 
  a 2 + a 10 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l900_90062


namespace NUMINAMATH_CALUDE_leap_year_classification_l900_90058

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

theorem leap_year_classification :
  (isLeapYear 2036 = true) ∧
  (isLeapYear 1996 = true) ∧
  (isLeapYear 1998 = false) ∧
  (isLeapYear 1700 = false) := by
  sorry

end NUMINAMATH_CALUDE_leap_year_classification_l900_90058


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l900_90099

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  (l1.a * l2.b = l1.b * l2.a) ∧ (l1.a ≠ 0 ∨ l1.b ≠ 0) ∧ (l2.a ≠ 0 ∨ l2.b ≠ 0)

/-- The main theorem -/
theorem parallel_lines_theorem (k : ℝ) :
  let l1 : Line := { a := k - 2, b := 4 - k, c := 1 }
  let l2 : Line := { a := 2 * (k - 2), b := -2, c := 3 }
  are_parallel l1 l2 ↔ k = 2 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l900_90099


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l900_90060

theorem smallest_positive_integer_with_given_remainders : ∃! n : ℕ+, 
  (n : ℤ) % 5 = 4 ∧
  (n : ℤ) % 7 = 6 ∧
  (n : ℤ) % 9 = 8 ∧
  (n : ℤ) % 11 = 10 ∧
  ∀ m : ℕ+, 
    (m : ℤ) % 5 = 4 ∧
    (m : ℤ) % 7 = 6 ∧
    (m : ℤ) % 9 = 8 ∧
    (m : ℤ) % 11 = 10 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l900_90060


namespace NUMINAMATH_CALUDE_unique_divisible_by_thirteen_l900_90053

theorem unique_divisible_by_thirteen :
  ∀ (B : Nat),
    B < 10 →
    (2000 + 100 * B + 34) % 13 = 0 ↔ B = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_thirteen_l900_90053


namespace NUMINAMATH_CALUDE_expression_evaluation_l900_90019

theorem expression_evaluation :
  (45 + 15)^2 - (45^2 + 15^2 + 2 * 45 * 5) = 900 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l900_90019
