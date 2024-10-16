import Mathlib

namespace NUMINAMATH_CALUDE_repeating_decimal_027_product_l3168_316802

/-- Represents a repeating decimal with a 3-digit repeating sequence -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_027_product : 
  let x := RepeatingDecimal 0 2 7
  let (n, d) := (x.num, x.den)
  (n.gcd d = 1) → n * d = 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_027_product_l3168_316802


namespace NUMINAMATH_CALUDE_rectangle_area_l3168_316862

def circle_inscribed_rectangle (r : ℝ) (l w : ℝ) : Prop :=
  2 * r = w

def length_width_ratio (l w : ℝ) : Prop :=
  l = 3 * w

theorem rectangle_area (r l w : ℝ) 
  (h1 : circle_inscribed_rectangle r l w) 
  (h2 : length_width_ratio l w) 
  (h3 : r = 7) : l * w = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3168_316862


namespace NUMINAMATH_CALUDE_area_inequality_l3168_316805

/-- Triangle type -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Point on a line segment -/
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- Area of a triangle -/
noncomputable def TriangleArea (T : Triangle) : ℝ :=
  abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2)) / 2

/-- Theorem statement -/
theorem area_inequality (ABC : Triangle) (X Y Z : ℝ × ℝ) 
  (hX : PointOnSegment X ABC.B ABC.C)
  (hY : PointOnSegment Y ABC.C ABC.A)
  (hZ : PointOnSegment Z ABC.A ABC.B)
  (hBX : dist ABC.B X ≤ dist X ABC.C)
  (hCY : dist ABC.C Y ≤ dist Y ABC.A)
  (hAZ : dist ABC.A Z ≤ dist Z ABC.B) :
  4 * TriangleArea ⟨X, Y, Z⟩ ≥ TriangleArea ABC :=
sorry

end NUMINAMATH_CALUDE_area_inequality_l3168_316805


namespace NUMINAMATH_CALUDE_davis_items_left_l3168_316833

/-- The number of items Miss Davis has left after distributing popsicle sticks and straws --/
def items_left (popsicle_sticks_per_group : ℕ) (straws_per_group : ℕ) (num_groups : ℕ) (total_items : ℕ) : ℕ :=
  total_items - (popsicle_sticks_per_group + straws_per_group) * num_groups

/-- Theorem stating that Miss Davis has 150 items left --/
theorem davis_items_left :
  items_left 15 20 10 500 = 150 := by
  sorry

end NUMINAMATH_CALUDE_davis_items_left_l3168_316833


namespace NUMINAMATH_CALUDE_commonly_used_charts_characterization_l3168_316897

/-- A type representing different types of charts -/
inductive Chart
  | ContingencyTable
  | ThreeDimensionalBarChart
  | TwoDimensionalBarChart
  | OtherChart

/-- The set of charts commonly used for analyzing relationships between two categorical variables -/
def commonly_used_charts : Set Chart := sorry

/-- The theorem stating that the commonly used charts are exactly the contingency tables,
    three-dimensional bar charts, and two-dimensional bar charts -/
theorem commonly_used_charts_characterization :
  commonly_used_charts = {Chart.ContingencyTable, Chart.ThreeDimensionalBarChart, Chart.TwoDimensionalBarChart} := by sorry

end NUMINAMATH_CALUDE_commonly_used_charts_characterization_l3168_316897


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l3168_316866

theorem rectangular_prism_width (l w h : ℕ) : 
  l * w * h = 128 → 
  w = 2 * l → 
  w = 2 * h → 
  w + 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l3168_316866


namespace NUMINAMATH_CALUDE_keith_books_l3168_316859

theorem keith_books (jason_books : ℕ) (total_books : ℕ) (h1 : jason_books = 21) (h2 : total_books = 41) :
  total_books - jason_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_keith_books_l3168_316859


namespace NUMINAMATH_CALUDE_wire_shapes_l3168_316870

/-- Given a wire of length 28 cm, prove properties about shapes formed from it -/
theorem wire_shapes (wire_length : ℝ) (h_wire : wire_length = 28) :
  let square_side : ℝ := wire_length / 4
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := wire_length / 2 - rectangle_length
  (square_side = 7 ∧ rectangle_width = 2) := by
  sorry

#check wire_shapes

end NUMINAMATH_CALUDE_wire_shapes_l3168_316870


namespace NUMINAMATH_CALUDE_range_of_a_l3168_316844

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, (1/2) * x^2 - Real.log (x - a) ≥ 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0

-- Define the range of a
def range_a : Set ℝ := Set.Ici (-4) ∪ Set.Icc (-2) (1/2)

-- Theorem statement
theorem range_of_a (a : ℝ) : p a ∧ q a → a ∈ range_a := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l3168_316844


namespace NUMINAMATH_CALUDE_new_sales_tax_percentage_l3168_316851

theorem new_sales_tax_percentage
  (original_tax : ℝ)
  (market_price : ℝ)
  (savings : ℝ)
  (h1 : original_tax = 3.5)
  (h2 : market_price = 8400)
  (h3 : savings = 14)
  : ∃ (new_tax : ℝ), new_tax = 10/3 ∧ 
    new_tax / 100 * market_price = original_tax / 100 * market_price - savings :=
sorry

end NUMINAMATH_CALUDE_new_sales_tax_percentage_l3168_316851


namespace NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l3168_316808

/-- An angle is in the first quadrant if it's between 0 and π/2 radians -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

/-- The main theorem stating the equivalence between an angle being in the first quadrant
    and the sum of its sine and cosine being greater than 1 -/
theorem first_quadrant_iff_sin_cos_sum_gt_one (α : ℝ) :
  is_first_quadrant α ↔ Real.sin α + Real.cos α > 1 :=
sorry

end NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l3168_316808


namespace NUMINAMATH_CALUDE_total_lunch_cost_l3168_316887

/-- The total cost of lunches for a field trip --/
theorem total_lunch_cost : 
  let num_children : ℕ := 35
  let num_chaperones : ℕ := 5
  let num_teacher : ℕ := 1
  let num_additional : ℕ := 3
  let cost_per_lunch : ℕ := 7
  let total_lunches : ℕ := num_children + num_chaperones + num_teacher + num_additional
  total_lunches * cost_per_lunch = 308 :=
by sorry

end NUMINAMATH_CALUDE_total_lunch_cost_l3168_316887


namespace NUMINAMATH_CALUDE_circle_area_outside_triangle_l3168_316861

/-- Given a right triangle ABC with ∠BAC = 90° and AB = 6, and a circle tangent to AB at X and AC at Y 
    with points diametrically opposite X and Y lying on BC, the area of the portion of the circle 
    that lies outside the triangle is 18π - 18. -/
theorem circle_area_outside_triangle (A B C X Y : ℝ × ℝ) (r : ℝ) : 
  -- Triangle ABC is a right triangle with ∠BAC = 90°
  (A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 6 ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = 6) →
  -- Circle is tangent to AB at X and AC at Y
  (X.1 = r ∧ X.2 = 0 ∧ Y.1 = 0 ∧ Y.2 = r) →
  -- Points diametrically opposite X and Y lie on BC
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 →
  -- The area of the portion of the circle outside the triangle
  π * r^2 - (B.1 * C.2 / 2) = 18 * π - 18 := by
sorry

end NUMINAMATH_CALUDE_circle_area_outside_triangle_l3168_316861


namespace NUMINAMATH_CALUDE_angle_C_is_45_degrees_l3168_316801

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ -- Area of the triangle

-- Define the vectors p and q
def p (t : Triangle) : ℝ × ℝ := (4, t.a^2 + t.b^2 - t.c^2)
def q (t : Triangle) : ℝ × ℝ := (1, t.S)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem angle_C_is_45_degrees (t : Triangle) 
  (h : parallel (p t) (q t)) : t.C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_45_degrees_l3168_316801


namespace NUMINAMATH_CALUDE_tangent_circles_area_ratio_l3168_316855

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle tangent to three sides of a regular hexagon -/
structure TangentCircle (h : RegularHexagon) :=
  (radius : ℝ)
  (tangent_to_parallel_sides : True)
  (tangent_to_other_side : True)

/-- The ratio of areas of two tangent circles to a regular hexagon is 1 -/
theorem tangent_circles_area_ratio (h : RegularHexagon) 
  (c1 c2 : TangentCircle h) : 
  (c1.radius^2) / (c2.radius^2) = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_area_ratio_l3168_316855


namespace NUMINAMATH_CALUDE_veridux_male_associates_l3168_316858

/-- Proves the number of male associates at Veridux Corporation --/
theorem veridux_male_associates :
  let total_employees : ℕ := 250
  let female_employees : ℕ := 90
  let total_managers : ℕ := 40
  let female_managers : ℕ := 40
  let male_employees : ℕ := total_employees - female_employees
  let male_associates : ℕ := male_employees
  male_associates = 160 := by
  sorry

end NUMINAMATH_CALUDE_veridux_male_associates_l3168_316858


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l3168_316854

theorem polynomial_sum_simplification (x : ℝ) : 
  (2*x^4 + 3*x^3 - 5*x^2 + 9*x - 8) + (-x^5 + x^4 - 2*x^3 + 4*x^2 - 6*x + 14) = 
  -x^5 + 3*x^4 + x^3 - x^2 + 3*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l3168_316854


namespace NUMINAMATH_CALUDE_stating_tournament_winners_l3168_316814

/-- Represents the number of participants in a tournament round -/
def participants : ℕ := 512

/-- Represents the number of wins we're interested in -/
def target_wins : ℕ := 6

/-- 
Represents the number of participants who finish with exactly k wins 
in a single-elimination tournament with n rounds
-/
def participants_with_wins (n k : ℕ) : ℕ := Nat.choose n k

/-- 
Theorem stating that in a single-elimination tournament with 512 participants,
exactly 84 participants will finish with 6 wins
-/
theorem tournament_winners : 
  participants_with_wins (Nat.log 2 participants) target_wins = 84 := by
  sorry

end NUMINAMATH_CALUDE_stating_tournament_winners_l3168_316814


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l3168_316831

theorem number_of_elements_in_set (initial_average : ℚ) (incorrect_number : ℚ) (correct_number : ℚ) (correct_average : ℚ) (n : ℕ) : 
  initial_average = 21 →
  incorrect_number = 26 →
  correct_number = 36 →
  correct_average = 22 →
  (n : ℚ) * initial_average + (correct_number - incorrect_number) = (n : ℚ) * correct_average →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l3168_316831


namespace NUMINAMATH_CALUDE_election_votes_total_l3168_316807

theorem election_votes_total (V A B C : ℝ) 
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V) :
  V = 60000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_total_l3168_316807


namespace NUMINAMATH_CALUDE_angleBisectorRatioNotDeterminesShape_twoAnglesAndSideDeterminesShape_angleBisectorRatiosDetermineShape_sideLengthRatiosDetermineShape_threeAnglesDetermineShape_l3168_316835

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of an angle bisector to its corresponding opposite side --/
def angleBisectorToOppositeSideRatio (t : Triangle) : ℝ := sorry

/-- Determines if two triangles have the same shape (are similar) --/
def sameShape (t1 t2 : Triangle) : Prop := sorry

/-- The theorem stating that the ratio of an angle bisector to its corresponding opposite side
    does not uniquely determine the shape of a triangle --/
theorem angleBisectorRatioNotDeterminesShape :
  ∃ t1 t2 : Triangle, 
    angleBisectorToOppositeSideRatio t1 = angleBisectorToOppositeSideRatio t2 ∧
    ¬ sameShape t1 t2 := by sorry

/-- The theorem stating that the ratio of two angles and the included side
    uniquely determines the shape of a triangle --/
theorem twoAnglesAndSideDeterminesShape (α β : ℝ) (s : ℝ) :
  ∀ t1 t2 : Triangle,
    (α = sorry) ∧ (β = sorry) ∧ (s = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that the ratios of the three angle bisectors
    uniquely determine the shape of a triangle --/
theorem angleBisectorRatiosDetermineShape (r1 r2 r3 : ℝ) :
  ∀ t1 t2 : Triangle,
    (r1 = sorry) ∧ (r2 = sorry) ∧ (r3 = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that the ratios of the three side lengths
    uniquely determine the shape of a triangle --/
theorem sideLengthRatiosDetermineShape (r1 r2 r3 : ℝ) :
  ∀ t1 t2 : Triangle,
    (r1 = sorry) ∧ (r2 = sorry) ∧ (r3 = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that three angles
    uniquely determine the shape of a triangle --/
theorem threeAnglesDetermineShape (α β γ : ℝ) :
  ∀ t1 t2 : Triangle,
    (α = sorry) ∧ (β = sorry) ∧ (γ = sorry) →
    sameShape t1 t2 := by sorry

end NUMINAMATH_CALUDE_angleBisectorRatioNotDeterminesShape_twoAnglesAndSideDeterminesShape_angleBisectorRatiosDetermineShape_sideLengthRatiosDetermineShape_threeAnglesDetermineShape_l3168_316835


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3168_316841

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧ 
  a 1 = 8 ∧
  a 4 = a 3 * a 5

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 7 = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3168_316841


namespace NUMINAMATH_CALUDE_inequality_holds_l3168_316878

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 - x*y - x*z - y*z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3168_316878


namespace NUMINAMATH_CALUDE_commodity_sales_profit_l3168_316812

/-- Profit function for a commodity sale --/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

/-- Sales quantity function --/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400

theorem commodity_sales_profit 
  (cost_price : ℝ) 
  (h_cost : cost_price = 10) 
  (h_domain : ∀ x, 0 < x → x ≤ 40 → sales_quantity x ≥ 0) :
  /- 1. Profit function is correct for the given domain -/
  (∀ x, 0 < x → x ≤ 40 → 
    profit_function x = (sales_quantity x) * (x - cost_price)) ∧
  /- 2. Selling price for $1250 profit that maximizes sales is $15 -/
  (∃ x, profit_function x = 1250 ∧ 
    sales_quantity x = (sales_quantity 15) ∧
    x = 15) ∧
  /- 3. Maximum profit when x ≥ 28 and y ≥ 50 is $2160 -/
  (∀ x, x ≥ 28 → sales_quantity x ≥ 50 → 
    profit_function x ≤ 2160) ∧
  (∃ x, x ≥ 28 ∧ sales_quantity x ≥ 50 ∧ 
    profit_function x = 2160) := by
  sorry

end NUMINAMATH_CALUDE_commodity_sales_profit_l3168_316812


namespace NUMINAMATH_CALUDE_road_trip_distances_l3168_316853

theorem road_trip_distances (total_distance : ℕ) 
  (tracy_distance michelle_distance katie_distance : ℕ) : 
  total_distance = 1000 →
  tracy_distance = 2 * michelle_distance + 20 →
  michelle_distance = 3 * katie_distance →
  tracy_distance + michelle_distance + katie_distance = total_distance →
  michelle_distance = 294 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distances_l3168_316853


namespace NUMINAMATH_CALUDE_animal_count_l3168_316847

theorem animal_count (frogs : ℕ) (h1 : frogs = 160) : ∃ (dogs cats : ℕ),
  frogs = 2 * dogs ∧
  cats = dogs - dogs / 5 ∧
  frogs + dogs + cats = 304 := by
sorry

end NUMINAMATH_CALUDE_animal_count_l3168_316847


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3168_316892

theorem simplify_and_evaluate (a b : ℤ) (ha : a = 2) (hb : b = -1) :
  (a + 3*b)^2 + (a + 3*b)*(a - 3*b) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3168_316892


namespace NUMINAMATH_CALUDE_divisibility_of_cubic_difference_l3168_316845

theorem divisibility_of_cubic_difference (x a b : ℝ) :
  ∃ P : ℝ → ℝ, (x + a + b)^3 - x^3 - a^3 - b^3 = P x * ((x + a) * (x + b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_cubic_difference_l3168_316845


namespace NUMINAMATH_CALUDE_base_difference_theorem_l3168_316825

/-- Convert a number from base 16 to base 10 -/
def base16_to_base10 (n : String) : ℕ :=
  match n with
  | "1A3" => 419
  | _ => 0

/-- Convert a number from base 7 to base 10 -/
def base7_to_base10 (n : String) : ℕ :=
  match n with
  | "142" => 79
  | _ => 0

/-- The main theorem stating that the difference between 1A3 (base 16) and 142 (base 7) in base 10 is 340 -/
theorem base_difference_theorem :
  base16_to_base10 "1A3" - base7_to_base10 "142" = 340 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_theorem_l3168_316825


namespace NUMINAMATH_CALUDE_magnitude_2a_minus_b_l3168_316871

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-1, 1)

theorem magnitude_2a_minus_b :
  Real.sqrt ((2 * vector_a.1 - vector_b.1)^2 + (2 * vector_a.2 - vector_b.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_2a_minus_b_l3168_316871


namespace NUMINAMATH_CALUDE_function_positivity_condition_equiv_a_range_l3168_316828

/-- The function f(x) = ax² - (2-a)x + 1 --/
def f (a x : ℝ) : ℝ := a * x^2 - (2 - a) * x + 1

/-- The function g(x) = x --/
def g (x : ℝ) : ℝ := x

/-- The theorem stating the equivalence of the condition and the range of a --/
theorem function_positivity_condition_equiv_a_range :
  ∀ a : ℝ, (∀ x : ℝ, max (f a x) (g x) > 0) ↔ (0 ≤ a ∧ a < 4 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_condition_equiv_a_range_l3168_316828


namespace NUMINAMATH_CALUDE_quadratic_factorization_and_perfect_square_discriminant_l3168_316895

/-- A quadratic expression of the form 15x^2 + ax + 15 can be factored into two linear binomial 
factors with integer coefficients, and its discriminant is a perfect square when a = 34 -/
theorem quadratic_factorization_and_perfect_square_discriminant :
  ∃ (m n p q : ℤ), 
    (15 : ℤ) * m * p = 15 ∧ 
    m * q + n * p = 34 ∧ 
    n * q = 15 ∧
    ∃ (k : ℤ), 34^2 - 4 * 15 * 15 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_and_perfect_square_discriminant_l3168_316895


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l3168_316869

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 23

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 21

/-- The number of extra apples -/
def extra_apples : ℕ := 35

/-- Theorem stating that the number of red apples ordered is 33 -/
theorem cafeteria_red_apples : 
  red_apples = 33 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l3168_316869


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3168_316817

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (simple_interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : simple_interest = 176)
  (h2 : time = 4)
  (h3 : rate = 5.5 / 100) :
  simple_interest = (800 : ℝ) * rate * time := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3168_316817


namespace NUMINAMATH_CALUDE_inequality_proof_l3168_316899

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3168_316899


namespace NUMINAMATH_CALUDE_work_completion_time_l3168_316836

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℚ
  hours : ℚ

/-- Represents a worker's capacity to complete work -/
structure Worker where
  completion_time : ℚ

/-- Represents a work period with multiple workers -/
structure WorkPeriod where
  duration : ℚ
  workers : List Worker

/-- Calculates the fraction of work completed in a day by a worker -/
def Worker.daily_work (w : Worker) : ℚ :=
  1 / w.completion_time

/-- Calculates the total work completed in a period -/
def WorkPeriod.work_completed (wp : WorkPeriod) : ℚ :=
  wp.duration * (wp.workers.map Worker.daily_work).sum

/-- Converts days to a WorkTime structure -/
def days_to_work_time (d : ℚ) : WorkTime :=
  ⟨d.floor, (d - d.floor) * 24⟩

theorem work_completion_time 
  (worker_a worker_b worker_c : Worker)
  (period1 period2 period3 : WorkPeriod)
  (h1 : worker_a.completion_time = 15)
  (h2 : worker_b.completion_time = 10)
  (h3 : worker_c.completion_time = 12)
  (h4 : period1 = ⟨2, [worker_a, worker_b, worker_c]⟩)
  (h5 : period2 = ⟨3, [worker_a, worker_c]⟩)
  (h6 : period3 = ⟨(1 - period1.work_completed - period2.work_completed) / worker_a.daily_work, [worker_a]⟩) :
  days_to_work_time (period1.duration + period2.duration + period3.duration) = ⟨5, 18⟩ :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3168_316836


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3168_316840

theorem arithmetic_sequence_common_difference_range (a : ℕ → ℝ) (d : ℝ) :
  (a 1 = -3) →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∀ n : ℕ, n ≥ 5 → a n > 0) →
  d ∈ Set.Ioo (3/4) 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3168_316840


namespace NUMINAMATH_CALUDE_root_sum_product_l3168_316894

theorem root_sum_product (a b : ℝ) : 
  (a^4 + 2*a^3 - 4*a - 1 = 0) →
  (b^4 + 2*b^3 - 4*b - 1 = 0) →
  (a ≠ b) →
  (a*b + a + b = Real.sqrt 3 - 2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l3168_316894


namespace NUMINAMATH_CALUDE_hcf_of_numbers_l3168_316896

theorem hcf_of_numbers (a b lcm : ℕ) (ha : a = 231) (hb : b = 300) (hlcm : lcm = 2310) :
  Nat.gcd a b = 30 := by sorry

end NUMINAMATH_CALUDE_hcf_of_numbers_l3168_316896


namespace NUMINAMATH_CALUDE_intersection_circles_angle_relation_l3168_316864

/-- Given two intersecting circles with radius R and centers separated by a distance greater than R,
    prove that the angle β formed at one intersection point is three times the angle α formed at the other intersection point. -/
theorem intersection_circles_angle_relation (R : ℝ) (center_distance : ℝ) (α β : ℝ) :
  R > 0 →
  center_distance > R →
  α > 0 →
  β > 0 →
  β = 3 * α :=
by sorry

end NUMINAMATH_CALUDE_intersection_circles_angle_relation_l3168_316864


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l3168_316829

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 35

/-- Calculates the total duration of all meetings in minutes -/
def total_meeting_minutes : ℕ := 
  first_meeting_minutes + 2 * first_meeting_minutes + (first_meeting_minutes + 2 * first_meeting_minutes)

/-- Theorem stating that the percentage of work day spent in meetings is 35% -/
theorem meetings_percentage_of_workday : 
  (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l3168_316829


namespace NUMINAMATH_CALUDE_smallest_result_l3168_316889

def S : Finset Nat := {2, 4, 6, 8, 10, 12}

def process (a b c : Nat) : Nat := (a + b) * c

def valid_triple (a b c : Nat) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : Nat), valid_triple a b c ∧
    (∀ (x y z : Nat), valid_triple x y z →
      min (process a b c) (min (process a c b) (process b c a)) ≤
      min (process x y z) (min (process x z y) (process y z x))) ∧
    min (process a b c) (min (process a c b) (process b c a)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l3168_316889


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l3168_316850

/-- Represents the chemical mixture problem --/
theorem chemical_mixture_problem (a w : ℝ) 
  (h1 : a / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 2) = 3 / 8) :
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l3168_316850


namespace NUMINAMATH_CALUDE_legacy_gold_bars_l3168_316824

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := sorry

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The value of one gold bar in dollars -/
def bar_value : ℕ := 2200

/-- The total value of gold bars Legacy and Aleena have together -/
def total_value : ℕ := 17600

theorem legacy_gold_bars :
  legacy_bars = 5 ∧
  aleena_bars = legacy_bars - 2 ∧
  bar_value = 2200 ∧
  total_value = 17600 ∧
  total_value = bar_value * (legacy_bars + aleena_bars) :=
sorry

end NUMINAMATH_CALUDE_legacy_gold_bars_l3168_316824


namespace NUMINAMATH_CALUDE_data_statistics_l3168_316868

def data : List ℝ := [6, 8, 8, 9, 8, 9, 8, 8, 7, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_statistics :
  mode data = 8 ∧
  median data = 8 ∧
  mean data = 8 ∧
  variance data ≠ 8 := by sorry

end NUMINAMATH_CALUDE_data_statistics_l3168_316868


namespace NUMINAMATH_CALUDE_middle_card_number_l3168_316813

theorem middle_card_number (a b c : ℕ) : 
  a < b → b < c → 
  a + b + c = 15 → 
  a + b < 10 → 
  (∀ x y z : ℕ, x < y → y < z → x + y + z = 15 → x + y < 10 → (x = a ∧ z = c) → y ≠ b) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_number_l3168_316813


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l3168_316846

theorem cone_rolling_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (2 * Real.pi * Real.sqrt (r^2 + h^2) = 30 * Real.pi * r) → h / r = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l3168_316846


namespace NUMINAMATH_CALUDE_polynomial_property_l3168_316815

def P (x : ℝ) (a₃ a₂ a₁ a₀ : ℝ) : ℝ := x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀

theorem polynomial_property (a₃ a₂ a₁ a₀ : ℝ) 
  (h1 : P 1 a₃ a₂ a₁ a₀ = 1)
  (h2 : P 2 a₃ a₂ a₁ a₀ = 2)
  (h3 : P 3 a₃ a₂ a₁ a₀ = 3)
  (h4 : P 4 a₃ a₂ a₁ a₀ = 4) :
  Real.sqrt (P 13 a₃ a₂ a₁ a₀ - 12) = 109 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l3168_316815


namespace NUMINAMATH_CALUDE_shifted_parabola_l3168_316886

/-- The equation of a parabola shifted 1 unit to the left -/
theorem shifted_parabola (x y : ℝ) :
  (y = -(x^2) + 1) → 
  (∃ x' y', y' = -(x'^2) + 1 ∧ x' = x + 1) →
  y = -((x + 1)^2) + 1 := by
sorry

end NUMINAMATH_CALUDE_shifted_parabola_l3168_316886


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_fifth_of_one_third_of_one_sixth_of_ninety_l3168_316830

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * (b * (c * d)) = (a * b * c) * d :=
by sorry

theorem one_fifth_of_one_third_of_one_sixth_of_ninety :
  (1 / 5 : ℚ) * ((1 / 3 : ℚ) * ((1 / 6 : ℚ) * 90)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_fifth_of_one_third_of_one_sixth_of_ninety_l3168_316830


namespace NUMINAMATH_CALUDE_brownie_theorem_l3168_316800

/-- The number of brownie pieces obtained from a rectangular tray -/
def brownie_pieces (tray_length tray_width piece_length piece_width : ℕ) : ℕ :=
  (tray_length * tray_width) / (piece_length * piece_width)

/-- Theorem stating that a 24-inch by 30-inch tray yields 60 brownie pieces of size 3 inches by 4 inches -/
theorem brownie_theorem :
  brownie_pieces 24 30 3 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_theorem_l3168_316800


namespace NUMINAMATH_CALUDE_cyrus_day4_pages_l3168_316874

/-- Represents the number of pages written on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the book writing problem --/
structure BookWritingProblem where
  totalPages : ℕ
  pagesWritten : DailyPages
  remainingPages : ℕ

/-- The specific instance of the book writing problem --/
def cyrusProblem : BookWritingProblem where
  totalPages := 500
  pagesWritten := {
    day1 := 25,
    day2 := 50,
    day3 := 100,
    day4 := 10  -- This is what we want to prove
  }
  remainingPages := 315

/-- Theorem stating that Cyrus wrote 10 pages on day 4 --/
theorem cyrus_day4_pages : 
  cyrusProblem.pagesWritten.day4 = 10 ∧
  cyrusProblem.pagesWritten.day2 = 2 * cyrusProblem.pagesWritten.day1 ∧
  cyrusProblem.pagesWritten.day3 = 2 * cyrusProblem.pagesWritten.day2 ∧
  cyrusProblem.totalPages = 
    cyrusProblem.pagesWritten.day1 + 
    cyrusProblem.pagesWritten.day2 + 
    cyrusProblem.pagesWritten.day3 + 
    cyrusProblem.pagesWritten.day4 + 
    cyrusProblem.remainingPages := by
  sorry

end NUMINAMATH_CALUDE_cyrus_day4_pages_l3168_316874


namespace NUMINAMATH_CALUDE_integral_of_exponential_l3168_316875

theorem integral_of_exponential (x : ℝ) :
  let f : ℝ → ℝ := λ x => (3^(7*x - 1/9)) / (7 * Real.log 3)
  (deriv f) x = 3^(7*x - 1/9) := by
  sorry

end NUMINAMATH_CALUDE_integral_of_exponential_l3168_316875


namespace NUMINAMATH_CALUDE_correct_distribution_probability_l3168_316848

/-- The number of rolls of each type -/
def rolls_per_type : ℕ := 3

/-- The number of types of rolls -/
def num_types : ℕ := 4

/-- The total number of rolls -/
def total_rolls : ℕ := rolls_per_type * num_types

/-- The number of guests -/
def num_guests : ℕ := 3

/-- The number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_types

/-- The probability of each guest getting one roll of each type -/
def probability_correct_distribution : ℚ := 27 / 1925

theorem correct_distribution_probability :
  (rolls_per_type : ℚ) * (rolls_per_type - 1) * (rolls_per_type - 2) /
  (total_rolls * (total_rolls - 1) * (total_rolls - 2) * (total_rolls - 3)) *
  ((rolls_per_type - 1) * (rolls_per_type - 1) * (rolls_per_type - 1) /
  ((total_rolls - 4) * (total_rolls - 5) * (total_rolls - 6) * (total_rolls - 7))) *
  1 = probability_correct_distribution := by
  sorry

end NUMINAMATH_CALUDE_correct_distribution_probability_l3168_316848


namespace NUMINAMATH_CALUDE_january_savings_l3168_316820

def savings_sequence (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => 2 * savings_sequence initial n

theorem january_savings (initial : ℕ) :
  savings_sequence initial 4 = 160 → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_january_savings_l3168_316820


namespace NUMINAMATH_CALUDE_janet_final_lives_l3168_316809

/-- Calculates the final number of lives for Janet in the video game --/
def final_lives (initial_lives : ℕ) (lives_lost : ℕ) (points_earned : ℕ) : ℕ :=
  let remaining_lives := initial_lives - lives_lost
  let lives_earned := (points_earned / 100) * 2
  let lives_lost_penalty := points_earned / 200
  remaining_lives + lives_earned - lives_lost_penalty

theorem janet_final_lives : 
  final_lives 47 23 1840 = 51 := by
  sorry

end NUMINAMATH_CALUDE_janet_final_lives_l3168_316809


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l3168_316865

/-- Given points A and B in 2D space, and a point C such that BC = 1/2 * AB,
    prove that C has specific coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (-3, 5) → 
  B = (9, -1) → 
  C - B = (1/2 : ℝ) • (B - A) → 
  C = (15, -4) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l3168_316865


namespace NUMINAMATH_CALUDE_p_properties_l3168_316872

/-- The product of digits function -/
def p (n : ℕ+) : ℕ := sorry

/-- Theorem stating the properties of p(n) -/
theorem p_properties (n : ℕ+) : 
  (p n ≤ n) ∧ (10 * p n = n^2 + 4*n - 2005 ↔ n = 45) := by sorry

end NUMINAMATH_CALUDE_p_properties_l3168_316872


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_and_girl_l3168_316849

theorem probability_at_least_one_boy_and_girl (p : ℝ) : 
  p = 1/2 → (1 - 2 * p^4) = 7/8 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_and_girl_l3168_316849


namespace NUMINAMATH_CALUDE_binomial_60_3_l3168_316842

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3168_316842


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l3168_316843

/-- The number of noodles Daniel gave away -/
def noodles_given : ℕ := 12

/-- The number of noodles Daniel has now -/
def noodles_left : ℕ := 54

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℕ := noodles_given + noodles_left

theorem daniel_initial_noodles :
  initial_noodles = 66 := by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l3168_316843


namespace NUMINAMATH_CALUDE_existence_of_functions_composition_inequality_l3168_316867

-- Part 1
theorem existence_of_functions :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f (g x) = g (f x)) ∧ 
    (∀ x, f (f x) = g (g x)) ∧ 
    (∀ x, f x ≠ g x) := by sorry

-- Part 2
theorem composition_inequality 
  (f₁ g₁ : ℝ → ℝ) 
  (h₁ : ∀ x, f₁ (g₁ x) = g₁ (f₁ x)) 
  (h₂ : ∀ x, f₁ x ≠ g₁ x) : 
  ∀ x, f₁ (f₁ x) ≠ g₁ (g₁ x) := by sorry

end NUMINAMATH_CALUDE_existence_of_functions_composition_inequality_l3168_316867


namespace NUMINAMATH_CALUDE_wall_length_proof_l3168_316821

def men_group1 : ℕ := 20
def men_group2 : ℕ := 86
def days : ℕ := 8
def wall_length_group2 : ℝ := 283.8

def wall_length_group1 : ℝ := 65.7

theorem wall_length_proof :
  (men_group1 * days * wall_length_group2) / (men_group2 * days) = wall_length_group1 := by
  sorry

end NUMINAMATH_CALUDE_wall_length_proof_l3168_316821


namespace NUMINAMATH_CALUDE_equation_solution_l3168_316823

theorem equation_solution (n : ℝ) : 
  1 / (n + 1) + 2 / (n + 2) + n / (n + 3) = 1 ↔ n = 2 + Real.sqrt 10 ∨ n = 2 - Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3168_316823


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l3168_316891

theorem unique_solution_quadratic_equation :
  ∃! x : ℝ, (2016 + 3*x)^2 = (3*x)^2 ∧ x = -336 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l3168_316891


namespace NUMINAMATH_CALUDE_perimeter_of_new_arrangement_l3168_316822

/-- Represents a square arrangement -/
structure SquareArrangement where
  rows : ℕ
  columns : ℕ

/-- Calculates the perimeter of a square arrangement -/
def perimeter (arrangement : SquareArrangement) : ℕ :=
  2 * (arrangement.rows + arrangement.columns)

/-- The original square arrangement -/
def original : SquareArrangement :=
  { rows := 3, columns := 5 }

/-- The new square arrangement with an additional row -/
def new : SquareArrangement :=
  { rows := original.rows + 1, columns := original.columns }

theorem perimeter_of_new_arrangement :
  perimeter new = 37 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_new_arrangement_l3168_316822


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3168_316827

/-- The quadratic function f(x) = x^2 + tx - t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + t*x - t

/-- Theorem stating that t ≥ 0 is a sufficient but not necessary condition for f to have a root -/
theorem sufficient_not_necessary_condition (t : ℝ) :
  (∀ t ≥ 0, ∃ x, f t x = 0) ∧
  (∃ t < 0, ∃ x, f t x = 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3168_316827


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3168_316882

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^4 - 4 * y^3) * (9 * x^8 + 12 * x^4 * y^3 + 16 * y^6) = 27 * x^12 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3168_316882


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3168_316826

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = 5) : z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3168_316826


namespace NUMINAMATH_CALUDE_no_triangle_solution_l3168_316852

theorem no_triangle_solution (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  c = 2 →
  C = π / 3 →
  ¬ (∃ (a : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (A > 0 ∧ B > 0 ∧ C > 0) ∧
    (A + B + C = π) ∧
    (a / Real.sin A = b / Real.sin B) ∧
    (b / Real.sin B = c / Real.sin C) ∧
    (c / Real.sin C = a / Real.sin A)) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_solution_l3168_316852


namespace NUMINAMATH_CALUDE_last_locker_theorem_l3168_316893

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction the student is walking -/
inductive Direction
| Forward
| Backward

/-- 
Simulates the student's locker-opening process and returns the number of the last locker opened.
n: The total number of lockers
-/
def lastLockerOpened (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that for 1024 lockers, the last one opened is number 854 -/
theorem last_locker_theorem : lastLockerOpened 1024 = 854 := by
  sorry

end NUMINAMATH_CALUDE_last_locker_theorem_l3168_316893


namespace NUMINAMATH_CALUDE_max_points_32_l3168_316883

-- Define the total number of shots
def total_shots : ℕ := 40

-- Define the success rates for three-point and two-point shots
def three_point_rate : ℚ := 1/4
def two_point_rate : ℚ := 2/5

-- Define the function that calculates the total points based on the number of three-point attempts
def total_points (three_point_attempts : ℕ) : ℚ :=
  3 * three_point_rate * three_point_attempts + 
  2 * two_point_rate * (total_shots - three_point_attempts)

-- Theorem: The maximum number of points Jamal could score is 32
theorem max_points_32 : 
  ∀ x : ℕ, x ≤ total_shots → total_points x ≤ 32 :=
sorry

end NUMINAMATH_CALUDE_max_points_32_l3168_316883


namespace NUMINAMATH_CALUDE_problem_solution_l3168_316884

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem problem_solution (a : ℝ) (h_a : a > 0) (h_a_neq_1 : a ≠ 1) :
  -- Part 1
  (∀ x, f a 2 x = -f a 2 (-x)) →
  -- Part 2
  f a 2 1 < 0 →
  (∀ x t, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0 ↔ -3 < t ∧ t < 5) ∧
  (∀ x y, x < y → f a 2 y < f a 2 x) →
  -- Part 3
  f a 2 1 = 3/2 →
  (∃ m, ∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2) →
  (∃! m, ∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2 ∧
               (∃ y, y ≥ 1 ∧ a^(2*y) + a^(-2*y) - 2*m*(f a 2 y) = -2)) →
  ∃ m, m = 2 ∧
    (∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2) ∧
    (∃ y, y ≥ 1 ∧ a^(2*y) + a^(-2*y) - 2*m*(f a 2 y) = -2) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3168_316884


namespace NUMINAMATH_CALUDE_sum_of_integer_solutions_is_zero_l3168_316863

theorem sum_of_integer_solutions_is_zero : 
  ∃ (S : Finset Int), 
    (∀ x ∈ S, x^4 - 49*x^2 + 576 = 0) ∧ 
    (∀ x : Int, x^4 - 49*x^2 + 576 = 0 → x ∈ S) ∧ 
    (S.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integer_solutions_is_zero_l3168_316863


namespace NUMINAMATH_CALUDE_scientific_notation_86000000_l3168_316837

theorem scientific_notation_86000000 : 
  86000000 = 8.6 * (10 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_86000000_l3168_316837


namespace NUMINAMATH_CALUDE_crate_height_difference_is_zero_l3168_316819

/-- Represents a cylindrical pipe -/
structure Pipe where
  diameter : ℝ

/-- Represents a crate filled with pipes -/
structure Crate where
  pipes : List Pipe
  stackingPattern : String

/-- Calculate the height of a crate -/
def calculateCrateHeight (c : Crate) : ℝ := sorry

/-- The main theorem statement -/
theorem crate_height_difference_is_zero 
  (pipeA pipeB : Pipe)
  (crateA crateB : Crate)
  (h1 : pipeA.diameter = 15)
  (h2 : pipeB.diameter = 15)
  (h3 : crateA.pipes.length = 150)
  (h4 : crateB.pipes.length = 150)
  (h5 : crateA.stackingPattern = "triangular")
  (h6 : crateB.stackingPattern = "inverted triangular")
  (h7 : ∀ p ∈ crateA.pipes, p = pipeA)
  (h8 : ∀ p ∈ crateB.pipes, p = pipeB) :
  |calculateCrateHeight crateA - calculateCrateHeight crateB| = 0 := by
  sorry

end NUMINAMATH_CALUDE_crate_height_difference_is_zero_l3168_316819


namespace NUMINAMATH_CALUDE_star_polygon_angle_sum_l3168_316873

/-- Represents a star polygon created from an n-sided convex polygon. -/
structure StarPolygon where
  n : ℕ
  h_n : n ≥ 6

/-- Calculates the sum of internal angles at the intersections of a star polygon. -/
def sum_of_internal_angles (sp : StarPolygon) : ℝ :=
  180 * (sp.n - 4)

/-- Theorem stating that the sum of internal angles at the intersections
    of a star polygon is 180(n-4) degrees. -/
theorem star_polygon_angle_sum (sp : StarPolygon) :
  sum_of_internal_angles sp = 180 * (sp.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_angle_sum_l3168_316873


namespace NUMINAMATH_CALUDE_calculation_one_l3168_316804

theorem calculation_one : (1) - 2 + (-3) - (-5) + 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_one_l3168_316804


namespace NUMINAMATH_CALUDE_orange_bin_calculation_l3168_316880

/-- Calculates the final number of oranges in a bin after a series of transactions -/
theorem orange_bin_calculation (initial : ℕ) (sold : ℕ) (new_shipment : ℕ) : 
  initial = 124 → sold = 46 → new_shipment = 250 → 
  (initial - sold - (initial - sold) / 2 + new_shipment) = 289 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_calculation_l3168_316880


namespace NUMINAMATH_CALUDE_digit_sum_difference_l3168_316832

/-- Represents a two-digit number -/
def TwoDigitNumber (tens ones : Nat) : Nat := 10 * tens + ones

theorem digit_sum_difference (A B C D : Nat) (E F : Nat) 
  (h1 : TwoDigitNumber A B + TwoDigitNumber C D = TwoDigitNumber A E)
  (h2 : TwoDigitNumber A B - TwoDigitNumber D C = TwoDigitNumber A F)
  (h3 : A < 10) (h4 : B < 10) (h5 : C < 10) (h6 : D < 10) : E = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_difference_l3168_316832


namespace NUMINAMATH_CALUDE_area_is_192_l3168_316881

/-- A right triangle with a circle tangent to its legs -/
structure RightTriangleWithTangentCircle where
  /-- The circle cuts the hypotenuse into segments of lengths 1, 24, and 3 -/
  hypotenuse_segments : ℝ × ℝ × ℝ
  /-- The middle segment (of length 24) is a chord of the circle -/
  middle_segment_is_chord : hypotenuse_segments.2.1 = 24

/-- The area of a right triangle with a tangent circle satisfying specific conditions -/
def area (t : RightTriangleWithTangentCircle) : ℝ := sorry

/-- Theorem: The area of the triangle is 192 -/
theorem area_is_192 (t : RightTriangleWithTangentCircle) 
  (h1 : t.hypotenuse_segments.1 = 1)
  (h2 : t.hypotenuse_segments.2.2 = 3) : 
  area t = 192 := by sorry

end NUMINAMATH_CALUDE_area_is_192_l3168_316881


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3168_316839

theorem price_decrease_percentage (original_price new_price : ℚ) :
  original_price = 1750 →
  new_price = 1050 →
  (original_price - new_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3168_316839


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3168_316816

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3168_316816


namespace NUMINAMATH_CALUDE_distance_between_points_l3168_316857

def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (2, 5)

theorem distance_between_points : Real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3168_316857


namespace NUMINAMATH_CALUDE_circle_ratio_l3168_316877

/-- Two circles touching externally -/
structure ExternallyTouchingCircles where
  R₁ : ℝ  -- Radius of the first circle
  R₂ : ℝ  -- Radius of the second circle
  h₁ : R₁ > 0
  h₂ : R₂ > 0

/-- Point of tangency between the circles -/
def pointOfTangency (c : ExternallyTouchingCircles) : ℝ := c.R₁ + c.R₂

/-- Distance from point of tangency to center of second circle -/
def tangentDistance (c : ExternallyTouchingCircles) : ℝ := 3 * c.R₂

theorem circle_ratio (c : ExternallyTouchingCircles) 
  (h : tangentDistance c = pointOfTangency c - c.R₁) : 
  c.R₁ = 4 * c.R₂ := by
  sorry

#check circle_ratio

end NUMINAMATH_CALUDE_circle_ratio_l3168_316877


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3168_316818

theorem inequality_equivalence :
  {x : ℝ | |x^2 - 5*x + 3| < 9} = {x : ℝ | (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3168_316818


namespace NUMINAMATH_CALUDE_equal_savings_l3168_316803

theorem equal_savings (your_initial : ℕ) (your_weekly : ℕ) (friend_initial : ℕ) (friend_weekly : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  your_weekly = 7 →
  friend_initial = 210 →
  friend_weekly = 5 →
  weeks = 25 →
  your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks :=
by sorry

end NUMINAMATH_CALUDE_equal_savings_l3168_316803


namespace NUMINAMATH_CALUDE_expression_equality_l3168_316890

theorem expression_equality : 12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3168_316890


namespace NUMINAMATH_CALUDE_solve_equation_l3168_316856

theorem solve_equation (p q x : ℚ) : 
  (3 / 4 : ℚ) = p / 60 ∧ 
  (3 / 4 : ℚ) = (p + q) / 100 ∧ 
  (3 / 4 : ℚ) = (x - q) / 140 → 
  x = 135 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3168_316856


namespace NUMINAMATH_CALUDE_chris_leftover_money_l3168_316888

theorem chris_leftover_money (
  video_game_cost : ℕ)
  (candy_cost : ℕ)
  (babysitting_rate : ℕ)
  (hours_worked : ℕ)
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  babysitting_rate * hours_worked - (video_game_cost + candy_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_chris_leftover_money_l3168_316888


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3168_316838

/-- Represents the number of cone types -/
def num_cone_types : ℕ := 2

/-- Represents the maximum number of scoops -/
def max_scoops : ℕ := 3

/-- Represents the number of ice cream flavors -/
def num_flavors : ℕ := 4

/-- Represents the number of topping choices -/
def num_toppings : ℕ := 4

/-- Represents the maximum number of toppings allowed -/
def max_toppings : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the total number of ice cream combinations -/
def total_combinations : ℕ := 
  let one_scoop := num_flavors
  let two_scoops := num_flavors + choose num_flavors 2
  let three_scoops := num_flavors + num_flavors * (num_flavors - 1) + choose num_flavors 3
  let scoop_combinations := one_scoop + two_scoops + three_scoops
  let topping_combinations := 1 + num_toppings + choose num_toppings 2
  num_cone_types * scoop_combinations * topping_combinations

/-- Theorem stating that the total number of ice cream combinations is 748 -/
theorem ice_cream_combinations : total_combinations = 748 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l3168_316838


namespace NUMINAMATH_CALUDE_nine_squared_minus_sqrt_nine_l3168_316834

theorem nine_squared_minus_sqrt_nine : 9^2 - Real.sqrt 9 = 78 := by
  sorry

end NUMINAMATH_CALUDE_nine_squared_minus_sqrt_nine_l3168_316834


namespace NUMINAMATH_CALUDE_unique_starting_digit_l3168_316876

def starts_with (x : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, d * 10^k ≤ x ∧ x < (d + 1) * 10^k

theorem unique_starting_digit :
  ∃! a : ℕ, a < 10 ∧ 
    (∃ n : ℕ, starts_with (2^n) a ∧ starts_with (5^n) a) ∧
    (a^2 < 10 ∧ 10 < (a+1)^2) :=
by sorry

end NUMINAMATH_CALUDE_unique_starting_digit_l3168_316876


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3168_316810

theorem quadratic_inequality_solution_range (q : ℝ) : 
  (q > 0) → 
  (∃ x : ℝ, x^2 - 8*x + q < 0) ↔ 
  (q > 0 ∧ q < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3168_316810


namespace NUMINAMATH_CALUDE_ellipse_with_same_foci_and_eccentricity_l3168_316885

/-- The standard equation of an ellipse with the same foci as another ellipse and a given eccentricity -/
theorem ellipse_with_same_foci_and_eccentricity 
  (a₁ b₁ : ℝ) 
  (h₁ : 0 < a₁ ∧ 0 < b₁) 
  (h₂ : a₁ > b₁) 
  (e : ℝ) 
  (he : e = Real.sqrt 5 / 5) :
  let c₁ := Real.sqrt (a₁^2 - b₁^2)
  let a := 5
  let b := Real.sqrt 20
  ∀ x y : ℝ, 
    (x^2 / a₁^2 + y^2 / b₁^2 = 1) → 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
     c₁ = Real.sqrt 5 ∧ 
     e = c₁ / a) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_with_same_foci_and_eccentricity_l3168_316885


namespace NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l3168_316806

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_specific_gp :
  let a := Real.sqrt 2
  let r := (Real.sqrt 2) ^ (1/4)
  let third_term := geometric_progression a r 3
  third_term = 2 ^ (1/8) →
  geometric_progression a r 4 = 1 / (Real.sqrt 2) ^ (1/4) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l3168_316806


namespace NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l3168_316811

def atomic_weight_Al : ℝ := 27
def atomic_weight_O : ℝ := 16
def atomic_weight_H : ℝ := 1

def num_Al : ℕ := 1
def num_O : ℕ := 3

def molecular_weight : ℝ := 78

theorem hydrogen_atoms_in_compound :
  ∃ (num_H : ℕ), 
    (num_Al : ℝ) * atomic_weight_Al + 
    (num_O : ℝ) * atomic_weight_O + 
    (num_H : ℝ) * atomic_weight_H = molecular_weight ∧
    num_H = 3 := by sorry

end NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l3168_316811


namespace NUMINAMATH_CALUDE_rachel_father_age_at_25_l3168_316879

/-- Rachel's current age -/
def rachel_age : ℕ := 12

/-- Rachel's grandfather's age in terms of Rachel's age -/
def grandfather_age_factor : ℕ := 7

/-- Rachel's mother's age in terms of grandfather's age -/
def mother_age_factor : ℚ := 1/2

/-- Age difference between Rachel's father and mother -/
def father_mother_age_diff : ℕ := 5

/-- Rachel's target age -/
def rachel_target_age : ℕ := 25

/-- Theorem stating Rachel's father's age when Rachel is 25 -/
theorem rachel_father_age_at_25 : 
  rachel_age * grandfather_age_factor * mother_age_factor + father_mother_age_diff + 
  (rachel_target_age - rachel_age) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rachel_father_age_at_25_l3168_316879


namespace NUMINAMATH_CALUDE_distance_between_parallel_points_l3168_316860

/-- Given two points A(4, a) and B(5, b) on a line parallel to y = x + m,
    prove that the distance between A and B is √2. -/
theorem distance_between_parallel_points :
  ∀ (a b m : ℝ),
  (b - a) / (5 - 4) = 1 →  -- Parallel condition
  Real.sqrt ((5 - 4)^2 + (b - a)^2) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_parallel_points_l3168_316860


namespace NUMINAMATH_CALUDE_arrive_at_beths_house_time_l3168_316898

/-- The time it takes for Tom and Beth to meet and return to Beth's house -/
def meeting_and_return_time (tom_speed beth_speed : ℚ) : ℚ :=
  let meeting_time := 1 / (tom_speed + beth_speed)
  let return_time := (1 / 2) / beth_speed
  meeting_time + return_time

/-- Theorem stating that Tom and Beth will arrive at Beth's house 78 minutes after noon -/
theorem arrive_at_beths_house_time :
  let tom_speed : ℚ := 1 / 63
  let beth_speed : ℚ := 1 / 84
  meeting_and_return_time tom_speed beth_speed = 78 / 1 := by
  sorry

#eval meeting_and_return_time (1 / 63) (1 / 84)

end NUMINAMATH_CALUDE_arrive_at_beths_house_time_l3168_316898
