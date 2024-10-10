import Mathlib

namespace expression_evaluation_l3702_370284

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -1) :
  (3 * x + 2 * y) - (3 * x - 2 * y) = -4 := by
sorry

end expression_evaluation_l3702_370284


namespace vector_at_negative_one_l3702_370231

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  point_at : ℝ → (ℝ × ℝ × ℝ)

/-- The vector at a given t value -/
def vector_at (line : ParametricLine) (t : ℝ) : (ℝ × ℝ × ℝ) :=
  line.point_at t

theorem vector_at_negative_one
  (line : ParametricLine)
  (h0 : vector_at line 0 = (2, 1, 5))
  (h1 : vector_at line 1 = (5, 0, 2)) :
  vector_at line (-1) = (-1, 2, 8) := by
  sorry

end vector_at_negative_one_l3702_370231


namespace quadratic_property_contradiction_l3702_370257

/-- Represents a quadratic function of the form y = ax² + bx - 6 --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Properties of the quadratic function --/
def QuadraticProperties (f : QuadraticFunction) : Prop :=
  ∃ (x_sym : ℝ) (y_min : ℝ),
    -- Axis of symmetry is x = 1
    x_sym = 1 ∧
    -- Minimum value is -8
    y_min = -8 ∧
    -- x = 3 is a root
    f.a * 3^2 + f.b * 3 - 6 = 0

/-- The main theorem to prove --/
theorem quadratic_property_contradiction (f : QuadraticFunction) 
  (h : QuadraticProperties f) : 
  f.a * 3^2 + f.b * 3 - 6 ≠ -6 := by
  sorry

end quadratic_property_contradiction_l3702_370257


namespace perimeter_ratio_from_area_ratio_l3702_370281

theorem perimeter_ratio_from_area_ratio (s1 s2 : ℝ) (h : s1 > 0 ∧ s2 > 0) 
  (h_area_ratio : s1^2 / s2^2 = 49 / 64) : 
  (4 * s1) / (4 * s2) = 7 / 8 := by
  sorry

end perimeter_ratio_from_area_ratio_l3702_370281


namespace ef_fraction_of_gh_l3702_370221

/-- Given a line segment GH with points E and F on it, prove that EF is 5/36 of GH -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  G < E → E < F → F < H →  -- E and F are on GH
  G - E = 3 * (H - E) →    -- GE is 3 times EH
  G - F = 8 * (H - F) →    -- GF is 8 times FH
  F - E = 5/36 * (H - G) := by
  sorry

end ef_fraction_of_gh_l3702_370221


namespace gear_r_rpm_calculation_l3702_370291

/-- The number of revolutions per minute for Gear L -/
def gear_l_rpm : ℚ := 20

/-- The time elapsed in seconds -/
def elapsed_time : ℚ := 6

/-- The additional revolutions made by Gear R compared to Gear L -/
def additional_revolutions : ℚ := 6

/-- Calculate the number of revolutions per minute for Gear R -/
def gear_r_rpm : ℚ :=
  (gear_l_rpm * elapsed_time / 60 + additional_revolutions) * 60 / elapsed_time

theorem gear_r_rpm_calculation :
  gear_r_rpm = 80 := by sorry

end gear_r_rpm_calculation_l3702_370291


namespace parabola_directrix_l3702_370228

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 + x + 3

-- Define the parabola g(x) = 2px^2
def g (p x : ℝ) : ℝ := 2 * p * x^2

-- Define the tangent line to f(x) at x = -1
def tangent_line (x : ℝ) : ℝ := 2 * x + 4

-- Theorem statement
theorem parabola_directrix :
  ∃ (p : ℝ),
    (∀ x, tangent_line x = g p x) ∧
    (∃ x, f x = tangent_line x ∧ x = -1) →
    (∀ y, y = 1 ↔ ∃ x, x^2 = -4 * y) :=
by sorry

end parabola_directrix_l3702_370228


namespace x_gt_3_sufficient_not_necessary_for_x_sq_gt_9_l3702_370253

theorem x_gt_3_sufficient_not_necessary_for_x_sq_gt_9 :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ 
  ¬(∀ x : ℝ, x^2 > 9 → x > 3) := by
  sorry

end x_gt_3_sufficient_not_necessary_for_x_sq_gt_9_l3702_370253


namespace subset_implies_b_equals_two_l3702_370289

theorem subset_implies_b_equals_two :
  (∀ x y : ℝ, x + y - 2 = 0 ∧ x - 2*y + 4 = 0 → y = 3*x + b) →
  b = 2 :=
by sorry

end subset_implies_b_equals_two_l3702_370289


namespace final_rope_length_l3702_370273

/-- Given a rope of length 100 feet, prove that after the described cutting process,
    the length of the final piece is 100 / (3 * 2 * 3 * 4 * 5 * 6) feet. -/
theorem final_rope_length (initial_length : ℝ := 100) : 
  initial_length / (3 * 2 * 3 * 4 * 5 * 6) = initial_length / 360 :=
by sorry

end final_rope_length_l3702_370273


namespace smallest_triangle_longer_leg_l3702_370270

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  longerLeg : ℝ
  shorterLeg : ℝ
  hyp_longer_ratio : longerLeg = hypotenuse * Real.sqrt 3 / 2
  hyp_shorter_ratio : shorterLeg = hypotenuse / 2

/-- Represents a sequence of four 30-60-90 triangles -/
def TriangleSequence (t₁ t₂ t₃ t₄ : Triangle30_60_90) : Prop :=
  t₁.hypotenuse = 8 ∧
  t₂.hypotenuse = t₁.longerLeg ∧
  t₃.hypotenuse = t₂.longerLeg ∧
  t₄.hypotenuse = t₃.longerLeg

theorem smallest_triangle_longer_leg 
  (t₁ t₂ t₃ t₄ : Triangle30_60_90)
  (h : TriangleSequence t₁ t₂ t₃ t₄) :
  t₄.longerLeg = 9 / 2 := by
  sorry

end smallest_triangle_longer_leg_l3702_370270


namespace polynomial_value_l3702_370241

theorem polynomial_value (a : ℝ) (h : a = Real.sqrt 17 - 1) : 
  a^5 + 2*a^4 - 17*a^3 - a^2 + 18*a - 17 = -1 := by sorry

end polynomial_value_l3702_370241


namespace square_roots_sum_equals_ten_l3702_370245

theorem square_roots_sum_equals_ten :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end square_roots_sum_equals_ten_l3702_370245


namespace box_volume_theorem_l3702_370219

/-- Represents the possible volumes of the box -/
def PossibleVolumes : Set ℕ := {80, 100, 120, 150, 200}

/-- Theorem: Given a rectangular box with integer side lengths in the ratio 1:2:5,
    the only possible volume from the set of possible volumes is 80 -/
theorem box_volume_theorem (x : ℕ) (hx : x > 0) :
  (∃ (v : ℕ), v ∈ PossibleVolumes ∧ v = x * (2 * x) * (5 * x)) ↔ (x * (2 * x) * (5 * x) = 80) :=
by sorry

end box_volume_theorem_l3702_370219


namespace hyperbola_min_focal_distance_l3702_370292

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove that the minimum semi-focal distance c is 4 -/
theorem hyperbola_min_focal_distance (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = c^2) →
  (a * b / c = c / 4 + 1) →
  c ≥ 4 := by
sorry

end hyperbola_min_focal_distance_l3702_370292


namespace largest_common_term_largest_common_term_exists_l3702_370207

theorem largest_common_term (n : ℕ) : n ≤ 200 ∧ 
  (∃ k : ℕ, n = 8 * k + 2) ∧ 
  (∃ m : ℕ, n = 9 * m + 5) →
  n ≤ 194 := by
  sorry

theorem largest_common_term_exists : 
  ∃ n : ℕ, n = 194 ∧ n ≤ 200 ∧ 
  (∃ k : ℕ, n = 8 * k + 2) ∧ 
  (∃ m : ℕ, n = 9 * m + 5) := by
  sorry

end largest_common_term_largest_common_term_exists_l3702_370207


namespace min_value_theorem_l3702_370216

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  ∃ (min : ℝ), min = 5 ∧ ∀ y, y > 1 → x + 4 / (x - 1) ≥ min :=
by sorry

end min_value_theorem_l3702_370216


namespace hare_wolf_distance_l3702_370205

def track_length : ℝ := 200
def hare_speed : ℝ := 5
def wolf_speed : ℝ := 3
def time_elapsed : ℝ := 40

def distance_traveled (speed : ℝ) : ℝ := speed * time_elapsed

theorem hare_wolf_distance : 
  ∃ (initial_distance : ℝ), 
    (initial_distance = 40 ∨ initial_distance = 60) ∧
    (
      (distance_traveled hare_speed - distance_traveled wolf_speed) % track_length = 0 ∨
      (distance_traveled hare_speed - distance_traveled wolf_speed + initial_distance) % track_length = initial_distance
    ) :=
by sorry

end hare_wolf_distance_l3702_370205


namespace arithmetic_sequence_formula_l3702_370293

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  (∀ n : ℕ, a n < a (n + 1)) →
  (a 2 / a 1 = a 4 / a 2) →
  ∀ n : ℕ, a n = 2 * n :=
by sorry

end arithmetic_sequence_formula_l3702_370293


namespace even_result_more_likely_l3702_370258

/-- Represents a calculator operation -/
inductive Operation
  | Add : Nat → Operation
  | Subtract : Nat → Operation
  | Multiply : Nat → Operation

/-- Represents a sequence of calculator operations -/
def OperationSequence := List Operation

/-- Applies a single operation to a number -/
def applyOperation (n : Int) (op : Operation) : Int :=
  match op with
  | Operation.Add m => n + m
  | Operation.Subtract m => n - m
  | Operation.Multiply m => n * m

/-- Applies a sequence of operations to an initial number -/
def applySequence (initial : Int) (seq : OperationSequence) : Int :=
  seq.foldl applyOperation initial

/-- Probability of getting an even result from a random operation sequence -/
noncomputable def probEvenResult (seqLength : Nat) : Real :=
  sorry

theorem even_result_more_likely (seqLength : Nat) :
  probEvenResult seqLength > 1 / 2 := by sorry

end even_result_more_likely_l3702_370258


namespace equation_solutions_l3702_370267

theorem equation_solutions :
  (∃ x : ℝ, (x + 2)^3 + 1 = 0 ↔ x = -3) ∧
  (∃ x : ℝ, (3*x - 2)^2 = 64 ↔ x = 10/3 ∨ x = -2) :=
by sorry

end equation_solutions_l3702_370267


namespace certain_fraction_problem_l3702_370201

theorem certain_fraction_problem (a b x y : ℚ) : 
  (a / b) / (2 / 5) = (3 / 8) / (x / y) →
  a / b = 3 / 4 →
  x / y = 1 / 5 :=
by sorry

end certain_fraction_problem_l3702_370201


namespace expand_expression_l3702_370298

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := by
  sorry

end expand_expression_l3702_370298


namespace orthocenter_preservation_l3702_370212

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : Point := sorry

-- Define the function to check if a point is inside a triangle
def is_inside (p : Point) (t : Triangle) : Prop := sorry

-- Define the function to check if a point is on a line segment
def on_segment (p : Point) (a b : Point) : Prop := sorry

-- Define the function to find the intersection of two line segments
def intersection (a b c d : Point) : Point := sorry

-- Main theorem
theorem orthocenter_preservation 
  (A B C H A₁ B₁ C₁ A₂ B₂ C₂ : Point) 
  (ABC : Triangle) :
  -- Given conditions
  (orthocenter ABC = H) →
  (is_inside A₁ (B, C, H)) →
  (is_inside B₁ (C, A, H)) →
  (is_inside C₁ (A, B, H)) →
  (orthocenter (A₁, B₁, C₁) = H) →
  (A₂ = intersection A H B₁ C₁) →
  (B₂ = intersection B H C₁ A₁) →
  (C₂ = intersection C H A₁ B₁) →
  -- Conclusion
  (orthocenter (A₂, B₂, C₂) = H) := by
  sorry

end orthocenter_preservation_l3702_370212


namespace total_students_correct_l3702_370213

/-- The total number of students in Misha's grade -/
def total_students : ℕ := 69

/-- Misha's position from the top -/
def position_from_top : ℕ := 30

/-- Misha's position from the bottom -/
def position_from_bottom : ℕ := 40

/-- Theorem stating that the total number of students is correct given Misha's positions -/
theorem total_students_correct :
  total_students = position_from_top + position_from_bottom - 1 :=
by sorry

end total_students_correct_l3702_370213


namespace total_percentage_increase_approx_l3702_370278

/-- Calculates the total percentage increase in USD for a purchase of three items with given initial and final prices in different currencies. -/
theorem total_percentage_increase_approx (book_initial book_final : ℝ)
                                         (album_initial album_final : ℝ)
                                         (poster_initial poster_final : ℝ)
                                         (usd_to_eur usd_to_gbp : ℝ)
                                         (h1 : book_initial = 300)
                                         (h2 : book_final = 480)
                                         (h3 : album_initial = 15)
                                         (h4 : album_final = 20)
                                         (h5 : poster_initial = 5)
                                         (h6 : poster_final = 10)
                                         (h7 : usd_to_eur = 0.85)
                                         (h8 : usd_to_gbp = 0.75) :
  ∃ ε > 0, abs (((book_final - book_initial + 
                 (album_final - album_initial) / usd_to_eur + 
                 (poster_final - poster_initial) / usd_to_gbp) / 
                (book_initial + album_initial / usd_to_eur + 
                 poster_initial / usd_to_gbp)) - 0.5937) < ε :=
by sorry


end total_percentage_increase_approx_l3702_370278


namespace max_marks_calculation_l3702_370204

/-- Proves that if a student scores 80% and receives 240 marks, the maximum possible marks in the examination is 300. -/
theorem max_marks_calculation (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) 
  (h1 : percentage = 0.80) 
  (h2 : scored_marks = 240) 
  (h3 : percentage * max_marks = scored_marks) : 
  max_marks = 300 := by
  sorry

end max_marks_calculation_l3702_370204


namespace x_approximates_one_l3702_370286

/-- The polynomial function P(x) = x^4 - 4x^3 + 4x^2 + 4 -/
def P (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x^2 + 4

/-- A small positive real number representing the tolerance for approximation -/
def ε : ℝ := 0.000000000000001

theorem x_approximates_one :
  ∃ x : ℝ, abs (P x - 4.999999999999999) < ε ∧ abs (x - 1) < ε :=
sorry

end x_approximates_one_l3702_370286


namespace quadratic_no_roots_positive_c_l3702_370277

/-- A quadratic polynomial with no real roots and positive sum of coefficients has a positive constant term. -/
theorem quadratic_no_roots_positive_c (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≠ 0) →  -- no real roots
  a + b + c > 0 →                   -- sum of coefficients is positive
  c > 0 :=                          -- constant term is positive
by sorry

end quadratic_no_roots_positive_c_l3702_370277


namespace banana_bunches_l3702_370255

theorem banana_bunches (total_bananas : ℕ) (bunches_of_seven : ℕ) (bananas_per_bunch_of_seven : ℕ) (bananas_per_bunch_of_eight : ℕ) :
  total_bananas = 83 →
  bunches_of_seven = 5 →
  bananas_per_bunch_of_seven = 7 →
  bananas_per_bunch_of_eight = 8 →
  ∃ (bunches_of_eight : ℕ), 
    total_bananas = bunches_of_seven * bananas_per_bunch_of_seven + bunches_of_eight * bananas_per_bunch_of_eight ∧
    bunches_of_eight = 6 :=
by sorry

end banana_bunches_l3702_370255


namespace system_solution_l3702_370294

theorem system_solution (x y z : ℝ) : 
  x + y + z = 2 ∧ 
  x^2 + y^2 + z^2 = 6 ∧ 
  x^3 + y^3 + z^3 = 8 ↔ 
  ((x = 1 ∧ y = 2 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 2) ∨
   (x = 2 ∧ y = 1 ∧ z = -1) ∨
   (x = 2 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 2) ∨
   (x = -1 ∧ y = 2 ∧ z = 1)) :=
by sorry

end system_solution_l3702_370294


namespace binomial_expansion_coefficient_l3702_370279

/-- Given a > 0 and the coefficient of the 1/x term in the expansion of (a√x - 1/√x)^6 is 135, prove that a = 3 -/
theorem binomial_expansion_coefficient (a : ℝ) (h1 : a > 0) 
  (h2 : (Nat.choose 6 4) * a^2 = 135) : a = 3 := by
  sorry

end binomial_expansion_coefficient_l3702_370279


namespace ceiling_sqrt_200_l3702_370240

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by
  sorry

end ceiling_sqrt_200_l3702_370240


namespace probability_no_adjacent_same_roll_probability_no_adjacent_same_roll_proof_l3702_370262

/-- The probability of no two adjacent people rolling the same number on an 8-sided die
    when 5 people sit around a circular table. -/
theorem probability_no_adjacent_same_roll : ℚ :=
  let num_people : ℕ := 5
  let die_sides : ℕ := 8
  let prob_same : ℚ := 1 / die_sides
  let prob_diff : ℚ := 1 - prob_same
  let prob_case1 : ℚ := prob_same * prob_diff^2 * (die_sides - 2) / die_sides
  let prob_case2 : ℚ := prob_diff^3 * (die_sides - 2) / die_sides
  302 / 512

/-- Proof of the theorem -/
theorem probability_no_adjacent_same_roll_proof :
  probability_no_adjacent_same_roll = 302 / 512 := by
  sorry

end probability_no_adjacent_same_roll_probability_no_adjacent_same_roll_proof_l3702_370262


namespace traci_flour_amount_l3702_370214

/-- The amount of flour Harris has in grams -/
def harris_flour : ℕ := 400

/-- The amount of flour needed for one cake in grams -/
def flour_per_cake : ℕ := 100

/-- The number of cakes Traci created -/
def traci_cakes : ℕ := 9

/-- The number of cakes Harris created -/
def harris_cakes : ℕ := 9

/-- The amount of flour Traci brought from her own house in grams -/
def traci_flour : ℕ := 1400

theorem traci_flour_amount :
  traci_flour = (flour_per_cake * (traci_cakes + harris_cakes)) - harris_flour :=
by sorry

end traci_flour_amount_l3702_370214


namespace largest_intersection_point_l3702_370296

/-- Polynomial P(x) -/
def P (a : ℝ) (x : ℝ) : ℝ := x^6 - 13*x^5 + 42*x^4 - 30*x^3 + a*x^2

/-- Line L(x) -/
def L (c : ℝ) (x : ℝ) : ℝ := 3*x + c

/-- The set of intersection points between P and L -/
def intersectionPoints (a c : ℝ) : Set ℝ := {x : ℝ | P a x = L c x}

theorem largest_intersection_point (a c : ℝ) :
  (∃ p q r : ℝ, intersectionPoints a c = {p, q, r} ∧ p < q ∧ q < r) →
  (∀ x : ℝ, x ∉ intersectionPoints a c → P a x < L c x) →
  (∃ x ∈ intersectionPoints a c, ∀ y ∈ intersectionPoints a c, y ≤ x) →
  (∃ x ∈ intersectionPoints a c, x = 4) :=
sorry

end largest_intersection_point_l3702_370296


namespace symmetric_point_tanDoubleAngle_l3702_370269

/-- Given a line l in the Cartesian plane defined by the equation 2x*tan(α) + y - 1 = 0,
    and that the symmetric point of the origin (0,0) with respect to l is (1,1),
    prove that tan(2α) = 4/3. -/
theorem symmetric_point_tanDoubleAngle (α : ℝ) : 
  (∀ x y : ℝ, 2 * x * Real.tan α + y - 1 = 0 → 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) → 
  Real.tan (2 * α) = 4 / 3 := by
  sorry

end symmetric_point_tanDoubleAngle_l3702_370269


namespace product_abcd_equals_162_over_185_l3702_370256

theorem product_abcd_equals_162_over_185 
  (a b c d : ℚ) 
  (eq1 : 3*a + 4*b + 6*c + 9*d = 45)
  (eq2 : 4*(d+c) = b + 1)
  (eq3 : 4*b + 2*c = a)
  (eq4 : 2*c - 2 = d) :
  a * b * c * d = 162 / 185 := by
sorry

end product_abcd_equals_162_over_185_l3702_370256


namespace system_of_equations_l3702_370202

theorem system_of_equations (x y : ℝ) 
  (eq1 : 2 * x + 4 * y = 5)
  (eq2 : x - y = 10) : 
  x + y = 5 := by sorry

end system_of_equations_l3702_370202


namespace craft_corner_sales_l3702_370261

/-- The percentage of sales that are neither brushes nor paints -/
def other_sales_percentage (total : ℝ) (brushes : ℝ) (paints : ℝ) : ℝ :=
  total - (brushes + paints)

/-- Theorem stating that given the total sales is 100%, and brushes and paints account for 45% and 28% of sales respectively, the percentage of sales that are neither brushes nor paints is 27% -/
theorem craft_corner_sales :
  let total := 100
  let brushes := 45
  let paints := 28
  other_sales_percentage total brushes paints = 27 := by
sorry

end craft_corner_sales_l3702_370261


namespace lisa_borrowed_chairs_l3702_370260

/-- The number of chairs Lisa borrowed from Rodrigo's classroom -/
def chairs_borrowed (red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after : ℕ) : ℕ :=
  total_chairs_before - total_chairs_after

/-- Theorem stating the number of chairs Lisa borrowed -/
theorem lisa_borrowed_chairs : 
  ∀ (red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after : ℕ),
  red_chairs = 4 →
  yellow_chairs = 2 * red_chairs →
  blue_chairs = yellow_chairs - 2 →
  total_chairs_before = red_chairs + yellow_chairs + blue_chairs →
  total_chairs_after = 15 →
  chairs_borrowed red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after = 3 :=
by sorry

end lisa_borrowed_chairs_l3702_370260


namespace problem_statement_l3702_370243

theorem problem_statement (x y : ℝ) (h : x + 2 * y - 3 = 0) : 2 * x * 4 * y = 8 := by
  sorry

end problem_statement_l3702_370243


namespace pizza_distribution_l3702_370223

/-- Given 12 coworkers sharing 3 pizzas with 8 slices each, prove that each person gets 2 slices. -/
theorem pizza_distribution (coworkers : ℕ) (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : coworkers = 12)
  (h2 : pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end pizza_distribution_l3702_370223


namespace gcd_143_117_l3702_370227

theorem gcd_143_117 : Nat.gcd 143 117 = 13 := by
  sorry

end gcd_143_117_l3702_370227


namespace addition_puzzle_l3702_370290

theorem addition_puzzle (E S X : Nat) : 
  E ≠ 0 → S ≠ 0 → X ≠ 0 →
  E ≠ S → E ≠ X → S ≠ X →
  E * 100 + E * 10 + E + E * 100 + E * 10 + E = S * 100 + X * 10 + S →
  X = 7 := by
sorry

end addition_puzzle_l3702_370290


namespace second_player_wins_l3702_370251

/-- Represents the game board -/
def Board := Fin 3 → Fin 101 → Bool

/-- The initial state of the board with the central cell crossed out -/
def initialBoard : Board :=
  fun i j => i = 1 && j = 50

/-- A move in the game -/
structure Move where
  start_row : Fin 3
  start_col : Fin 101
  length : Fin 4
  direction : Bool  -- true for down-right, false for down-left

/-- Checks if a move is valid on the given board -/
def isValidMove (b : Board) (m : Move) : Bool :=
  sorry

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (b : Board) : Bool :=
  sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Board → Move),
    ∀ (game : List Move),
      game.length % 2 = 0 →
      let final_board := game.foldl applyMove initialBoard
      isGameOver final_board →
      ∃ (m : Move), isValidMove final_board m :=
sorry

end second_player_wins_l3702_370251


namespace fraction_sum_evaluation_l3702_370259

theorem fraction_sum_evaluation (p q r : ℝ) 
  (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 9) :
  6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 2.4 := by
  sorry

end fraction_sum_evaluation_l3702_370259


namespace expression_evaluation_l3702_370285

theorem expression_evaluation :
  Real.sqrt (25 / 9) + (Real.log 5 / Real.log 10) ^ 0 + (27 / 64) ^ (-(1/3 : ℝ)) = 4 := by
  sorry

end expression_evaluation_l3702_370285


namespace razorback_total_profit_l3702_370232

/-- The Razorback shop's sales during the Arkansas and Texas Tech game -/
def razorback_sales : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun tshirt_profit jersey_profit hat_profit keychain_profit
      tshirts_sold jerseys_sold hats_sold keychains_sold =>
    tshirt_profit * tshirts_sold +
    jersey_profit * jerseys_sold +
    hat_profit * hats_sold +
    keychain_profit * keychains_sold

theorem razorback_total_profit :
  razorback_sales 62 99 45 25 183 31 142 215 = 26180 := by
  sorry

end razorback_total_profit_l3702_370232


namespace max_loss_is_9_l3702_370229

/-- Represents the ratio of money for each person --/
structure MoneyRatio :=
  (cara : ℕ)
  (janet : ℕ)
  (jerry : ℕ)
  (linda : ℕ)

/-- Represents the price range for oranges --/
structure PriceRange :=
  (min : ℚ)
  (max : ℚ)

/-- Calculates the maximum loss for Cara and Janet --/
def calculate_max_loss (ratio : MoneyRatio) (total_money : ℚ) (price_range : PriceRange) (sell_percentage : ℚ) : ℚ :=
  sorry

theorem max_loss_is_9 (ratio : MoneyRatio) (total_money : ℚ) (price_range : PriceRange) (sell_percentage : ℚ) :
  ratio.cara = 4 ∧ ratio.janet = 5 ∧ ratio.jerry = 6 ∧ ratio.linda = 7 ∧
  total_money = 110 ∧
  price_range.min = 1/2 ∧ price_range.max = 3/2 ∧
  sell_percentage = 4/5 →
  calculate_max_loss ratio total_money price_range sell_percentage = 9 :=
by sorry

end max_loss_is_9_l3702_370229


namespace average_and_difference_l3702_370236

theorem average_and_difference (y : ℝ) : 
  (35 + y) / 2 = 42 → |35 - y| = 14 := by sorry

end average_and_difference_l3702_370236


namespace xiaoli_estimation_l3702_370295

theorem xiaoli_estimation (x y : ℝ) (h : x > y ∧ y > 0) :
  (1.1 * x - (y - 2) = x - y + 0.1 * x + 2) ∧
  (1.1 * x * (y - 2) = 1.1 * x * y - 2.2 * x) := by
  sorry

end xiaoli_estimation_l3702_370295


namespace ancient_market_prices_l3702_370268

/-- The cost of animals in an ancient market --/
theorem ancient_market_prices :
  -- Define the costs of animals
  ∀ (camel_cost horse_cost ox_cost elephant_cost : ℚ),
  -- Conditions from the problem
  (10 * camel_cost = 24 * horse_cost) →
  (16 * horse_cost = 4 * ox_cost) →
  (6 * ox_cost = 4 * elephant_cost) →
  (10 * elephant_cost = 110000) →
  -- Conclusion: the cost of one camel is 4400
  camel_cost = 4400 := by
  sorry

end ancient_market_prices_l3702_370268


namespace f_negative_t_zero_l3702_370275

theorem f_negative_t_zero (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f x = 3 * x + Real.sin x + 1) →
  f t = 2 →
  f (-t) = 0 := by
sorry

end f_negative_t_zero_l3702_370275


namespace specific_field_area_l3702_370200

/-- Represents a rectangular field with partial fencing -/
structure PartiallyFencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing : ℝ

/-- Calculates the area of a rectangular field -/
def field_area (f : PartiallyFencedField) : ℝ :=
  f.length * f.width

/-- Theorem: The area of a specific partially fenced field is 390 square feet -/
theorem specific_field_area :
  ∃ (f : PartiallyFencedField),
    f.uncovered_side = 20 ∧
    f.fencing = 59 ∧
    f.length = f.uncovered_side ∧
    2 * f.width + f.uncovered_side = f.fencing ∧
    field_area f = 390 := by
  sorry

end specific_field_area_l3702_370200


namespace nes_sale_price_l3702_370254

/-- The sale price of the NES given the trade-in value of SNES, additional payment, change, and game value. -/
theorem nes_sale_price
  (snes_value : ℝ)
  (trade_in_percentage : ℝ)
  (additional_payment : ℝ)
  (change : ℝ)
  (game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : additional_payment = 80)
  (h4 : change = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + additional_payment - change - game_value = 160 :=
sorry


end nes_sale_price_l3702_370254


namespace intersection_point_l3702_370276

theorem intersection_point (x y : ℚ) : 
  (x = 40/17 ∧ y = 21/17) ↔ (3*x + 4*y = 12 ∧ 7*x - 2*y = 14) :=
by sorry

end intersection_point_l3702_370276


namespace closest_irrational_to_four_l3702_370222

theorem closest_irrational_to_four :
  let options : List ℝ := [Real.sqrt 11, Real.sqrt 13, Real.sqrt 17, Real.sqrt 19]
  let four : ℝ := Real.sqrt 16
  ∀ x ∈ options, x ≠ Real.sqrt 17 →
    |Real.sqrt 17 - four| < |x - four| :=
by sorry

end closest_irrational_to_four_l3702_370222


namespace select_five_from_eight_l3702_370274

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end select_five_from_eight_l3702_370274


namespace power_inequality_l3702_370203

theorem power_inequality (a b : ℝ) : a^6 + b^6 ≥ a^4*b^2 + a^2*b^4 := by
  sorry

end power_inequality_l3702_370203


namespace hexagram_ratio_is_three_l3702_370283

/-- A hexagram formed by overlapping two equilateral triangles -/
structure Hexagram where
  /-- The hexagram's vertices coincide with those of a regular hexagon -/
  vertices_coincide : Bool
  /-- The number of smaller triangles in the shaded region -/
  shaded_triangles : Nat
  /-- The number of smaller triangles in the unshaded region -/
  unshaded_triangles : Nat

/-- The ratio of shaded to unshaded area in a hexagram -/
def shaded_unshaded_ratio (h : Hexagram) : ℚ :=
  h.shaded_triangles / h.unshaded_triangles

/-- Theorem: The ratio of shaded to unshaded area in the specified hexagram is 3 -/
theorem hexagram_ratio_is_three (h : Hexagram) 
  (h_vertices : h.vertices_coincide = true)
  (h_shaded : h.shaded_triangles = 18)
  (h_unshaded : h.unshaded_triangles = 6) : 
  shaded_unshaded_ratio h = 3 := by
  sorry

end hexagram_ratio_is_three_l3702_370283


namespace count_students_in_line_l3702_370208

/-- The number of students standing in a line, given that Yoojeong is at the back and there are some people in front of her. -/
def studentsInLine (peopleInFront : ℕ) : ℕ :=
  peopleInFront + 1

/-- Theorem stating that the number of students in the line is equal to the number of people in front of Yoojeong plus one. -/
theorem count_students_in_line (peopleInFront : ℕ) :
  studentsInLine peopleInFront = peopleInFront + 1 := by
  sorry

end count_students_in_line_l3702_370208


namespace cubic_function_extrema_l3702_370225

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

theorem cubic_function_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0) →
  a ∈ Set.Iic (-3) ∪ Set.Ioi 6 :=
sorry

end cubic_function_extrema_l3702_370225


namespace min_value_z_l3702_370282

theorem min_value_z (x y : ℝ) :
  let z := 3 * x^2 + 4 * y^2 + 8 * x - 6 * y + 30
  ∀ a b : ℝ, z ≥ 3 * a^2 + 4 * b^2 + 8 * a - 6 * b + 30 → z ≥ 24.1 :=
by sorry

end min_value_z_l3702_370282


namespace square_root_divided_by_six_l3702_370210

theorem square_root_divided_by_six (x : ℝ) : x > 0 ∧ Real.sqrt x / 6 = 1 ↔ x = 36 := by
  sorry

end square_root_divided_by_six_l3702_370210


namespace typhoon_tree_difference_l3702_370218

def initial_trees : ℕ := 3
def dead_trees : ℕ := 13

theorem typhoon_tree_difference : dead_trees - (initial_trees - dead_trees) = 13 := by
  sorry

end typhoon_tree_difference_l3702_370218


namespace right_triangle_to_square_l3702_370249

theorem right_triangle_to_square (longer_leg : ℝ) (shorter_leg : ℝ) (square_side : ℝ) : 
  longer_leg = 10 →
  longer_leg = 2 * square_side →
  shorter_leg = square_side →
  shorter_leg = 5 :=
by
  sorry

end right_triangle_to_square_l3702_370249


namespace video_game_cost_l3702_370252

def september_savings : ℕ := 17
def october_savings : ℕ := 48
def november_savings : ℕ := 25
def amount_left : ℕ := 41

def total_savings : ℕ := september_savings + october_savings + november_savings

theorem video_game_cost : total_savings - amount_left = 49 := by
  sorry

end video_game_cost_l3702_370252


namespace farm_chicken_count_l3702_370246

theorem farm_chicken_count :
  ∀ (num_hens num_roosters : ℕ),
    num_hens = 52 →
    num_roosters = num_hens + 16 →
    num_hens + num_roosters = 120 :=
by
  sorry

end farm_chicken_count_l3702_370246


namespace absolute_value_inequality_solution_set_l3702_370242

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 3| > 1} = Set.Iio (-4) ∪ Set.Ioi (-2) := by sorry

end absolute_value_inequality_solution_set_l3702_370242


namespace initial_ratio_is_four_to_one_l3702_370234

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- The initial mixture before adding water -/
def initial_mixture : Mixture := sorry

/-- The final mixture after adding water -/
def final_mixture : Mixture := sorry

theorem initial_ratio_is_four_to_one :
  -- Initial mixture volume is 45 litres
  initial_mixture.milk + initial_mixture.water = 45 →
  -- 9 litres of water added
  final_mixture.water = initial_mixture.water + 9 →
  -- Final ratio of milk to water is 2:1
  final_mixture.milk / final_mixture.water = 2 →
  -- Prove that the initial ratio of milk to water was 4:1
  initial_mixture.milk / initial_mixture.water = 4 := by
  sorry

end initial_ratio_is_four_to_one_l3702_370234


namespace x_greater_than_y_l3702_370287

theorem x_greater_than_y (a b x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : x = a + 1/a) 
  (h4 : y = b + 1/b) : 
  x > y := by
sorry

end x_greater_than_y_l3702_370287


namespace linear_equation_equivalence_l3702_370266

theorem linear_equation_equivalence (x y : ℚ) :
  2 * x + 3 * y - 4 = 0 →
  (y = (4 - 2 * x) / 3 ∧ x = (4 - 3 * y) / 2) :=
by sorry

end linear_equation_equivalence_l3702_370266


namespace four_person_greeting_card_distribution_l3702_370233

def greeting_card_distribution (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * greeting_card_distribution (n - 1)

theorem four_person_greeting_card_distribution :
  greeting_card_distribution 4 = 9 :=
sorry

end four_person_greeting_card_distribution_l3702_370233


namespace cube_sphere_volume_ratio_l3702_370271

/-- The ratio of the volume of a cube with edge length 8 inches to the volume of a sphere with diameter 12 inches is 16 / (9 * π) -/
theorem cube_sphere_volume_ratio :
  let cube_edge : ℝ := 8
  let sphere_diameter : ℝ := 12
  let cube_volume := cube_edge ^ 3
  let sphere_radius := sphere_diameter / 2
  let sphere_volume := (4 / 3) * π * sphere_radius ^ 3
  cube_volume / sphere_volume = 16 / (9 * π) := by
sorry

end cube_sphere_volume_ratio_l3702_370271


namespace money_sharing_problem_l3702_370247

theorem money_sharing_problem (total : ℕ) (amanda ben carlos : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 2 * (total / 13) →
  ben = 3 * (total / 13) →
  carlos = 8 * (total / 13) →
  ben = 30 →
  total = 130 := by
sorry

end money_sharing_problem_l3702_370247


namespace average_marks_combined_classes_l3702_370220

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 80 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 65 * ((n1 : ℚ) + (n2 : ℚ)) := by
  sorry

end average_marks_combined_classes_l3702_370220


namespace amit_work_days_l3702_370299

theorem amit_work_days (ananthu_days : ℝ) (amit_worked : ℝ) (total_days : ℝ) :
  ananthu_days = 90 ∧ amit_worked = 3 ∧ total_days = 75 →
  ∃ x : ℝ, 
    x > 0 ∧
    (3 / x) + ((total_days - amit_worked) / ananthu_days) = 1 ∧
    x = 15 := by
  sorry

end amit_work_days_l3702_370299


namespace negation_of_universal_statement_l3702_370297

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end negation_of_universal_statement_l3702_370297


namespace fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_odd_fifteenth_odd_multiple_of_5_is_multiple_of_5_l3702_370248

/-- The nth positive odd multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

/-- The 15th positive odd multiple of 5 is odd -/
theorem fifteenth_odd_multiple_of_5_is_odd : Odd (oddMultipleOf5 15) := by
  sorry

/-- The 15th positive odd multiple of 5 is a multiple of 5 -/
theorem fifteenth_odd_multiple_of_5_is_multiple_of_5 : ∃ k : ℕ, oddMultipleOf5 15 = 5 * k := by
  sorry

end fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_odd_fifteenth_odd_multiple_of_5_is_multiple_of_5_l3702_370248


namespace tetrahedron_volume_l3702_370237

/-- Tetrahedron PQRS with given properties -/
structure Tetrahedron where
  /-- Angle between faces PQR and QRS -/
  angle : ℝ
  /-- Area of face PQR -/
  area_PQR : ℝ
  /-- Area of face QRS -/
  area_QRS : ℝ
  /-- Length of edge QR -/
  length_QR : ℝ

/-- The volume of a tetrahedron with the given properties -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem tetrahedron_volume (t : Tetrahedron) 
  (h1 : t.angle = 45 * π / 180)
  (h2 : t.area_PQR = 150)
  (h3 : t.area_QRS = 50)
  (h4 : t.length_QR = 10) : 
  volume t = 250 * Real.sqrt 2 := by
  sorry

end tetrahedron_volume_l3702_370237


namespace point_on_600_degree_angle_l3702_370206

/-- Prove that if a point (-4, a) lies on the terminal side of an angle measuring 600°, then a = -4√3. -/
theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * π / 180 ∧ Real.tan θ = a / (-4)) → a = -4 * Real.sqrt 3 := by
  sorry

end point_on_600_degree_angle_l3702_370206


namespace tangent_line_at_one_l3702_370239

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
  (∀ x : ℝ, x ≠ 0 → HasDerivAt f (1 / x + 2) x) ∧
  (m * 1 - f 1 + b = 0) ∧
  (m = 3) ∧ (b = -2) := by
  sorry

end tangent_line_at_one_l3702_370239


namespace sum_remainder_l3702_370264

theorem sum_remainder (x y : ℤ) (hx : x % 72 = 19) (hy : y % 50 = 6) : 
  (x + y) % 8 = 1 := by
  sorry

end sum_remainder_l3702_370264


namespace isosceles_triangle_perimeter_l3702_370263

/-- An isosceles triangle with two sides measuring 4 and 7 has a perimeter of either 15 or 18. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 4 ∧ b = 4 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 4) →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 15 ∨ a + b + c = 18 := by
sorry

end isosceles_triangle_perimeter_l3702_370263


namespace circle_intersection_angle_l3702_370288

theorem circle_intersection_angle (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 4 → r₂ = 3 → r₃ = 2 →
  shaded_ratio = 5 / 11 →
  ∃ θ : ℝ,
    θ > 0 ∧ θ < π / 2 ∧
    (θ * (r₁^2 + r₃^2) + (π - θ) * r₂^2) / (π * (r₁^2 + r₂^2 + r₃^2)) = shaded_ratio ∧
    θ = π / 176 :=
by sorry

end circle_intersection_angle_l3702_370288


namespace perpendicular_vectors_x_value_l3702_370250

/-- Given two vectors a and b in R², where a = (4,8) and b = (x,4),
    if a is perpendicular to b, then x = -8. -/
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -8 :=
by
  sorry

#check perpendicular_vectors_x_value

end perpendicular_vectors_x_value_l3702_370250


namespace bicycle_price_last_year_l3702_370217

theorem bicycle_price_last_year (total_sales_last_year : ℝ) (price_decrease : ℝ) 
  (sales_quantity : ℝ) (decrease_percentage : ℝ) :
  total_sales_last_year = 80000 →
  price_decrease = 200 →
  decrease_percentage = 0.1 →
  total_sales_last_year * (1 - decrease_percentage) = 
    sales_quantity * (total_sales_last_year / sales_quantity - price_decrease) →
  total_sales_last_year / sales_quantity = 2000 := by
sorry

end bicycle_price_last_year_l3702_370217


namespace sin_360_degrees_l3702_370230

theorem sin_360_degrees : Real.sin (2 * Real.pi) = 0 := by
  sorry

end sin_360_degrees_l3702_370230


namespace september_reading_plan_l3702_370265

theorem september_reading_plan (total_pages : ℕ) (total_days : ℕ) (busy_days : ℕ) (flight_pages : ℕ) :
  total_pages = 600 →
  total_days = 30 →
  busy_days = 4 →
  flight_pages = 100 →
  ∃ (standard_pages : ℕ),
    standard_pages * (total_days - busy_days - 1) + flight_pages = total_pages ∧
    standard_pages = 20 := by
  sorry

end september_reading_plan_l3702_370265


namespace part_a_part_b_part_c_l3702_370235

-- Define what it means for a number to be TOP
def is_TOP (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  (n / 10000) * (n % 10) = ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10)

-- Part a
theorem part_a : is_TOP 23498 := by sorry

-- Part b
theorem part_b : ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_TOP n ∧ n / 10000 = 1 ∧ n % 10 = 2) ∧ 
  (∀ n, is_TOP n ∧ n / 10000 = 1 ∧ n % 10 = 2 → n ∈ s) ∧
  Finset.card s = 6 := by sorry

-- Part c
theorem part_c : ∃ (s : Finset ℕ),
  (∀ n ∈ s, is_TOP n ∧ n / 10000 = 9) ∧
  (∀ n, is_TOP n ∧ n / 10000 = 9 → n ∈ s) ∧
  Finset.card s = 112 := by sorry

end part_a_part_b_part_c_l3702_370235


namespace infinitely_many_solutions_l3702_370211

theorem infinitely_many_solutions (k : ℝ) : 
  (∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)) ↔ k = -9 := by
  sorry

end infinitely_many_solutions_l3702_370211


namespace fraction_product_simplification_l3702_370226

theorem fraction_product_simplification :
  (21 : ℚ) / 16 * 48 / 35 * 80 / 63 = 48 := by
  sorry

end fraction_product_simplification_l3702_370226


namespace max_log_sum_l3702_370209

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  ∃ (max_val : ℝ), max_val = 6 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 5 → Real.log a + Real.log b ≤ max_val :=
by
  sorry

end max_log_sum_l3702_370209


namespace shorter_base_length_l3702_370272

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The length of the line segment joining the midpoints of the diagonals
    is half the difference between the lengths of the bases -/
axiom trapezoid_midpoint_segment (t : Trapezoid) :
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.longer_base = 115)
  (h2 : t.midpoint_segment = 5) :
  t.shorter_base = 105 := by
  sorry

#check shorter_base_length

end shorter_base_length_l3702_370272


namespace intersection_distance_l3702_370238

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : ℝ  -- x-coordinate of the vertical directrix

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Check if a point is on the parabola -/
def isOnParabola (p : Point) (pa : Parabola) : Prop :=
  p.x = 2 * (5 + 2 * Real.sqrt 3) * p.y^2 + (5 + 2 * Real.sqrt 3) / 2

/-- The main theorem -/
theorem intersection_distance (e : Ellipse) (pa : Parabola) 
    (p1 p2 : Point) : 
    e.center = Point.mk 0 0 →
    e.a = 4 →
    e.b = 2 →
    pa.directrix = 5 →
    pa.focus = Point.mk (2 * Real.sqrt 3) 0 →
    isOnEllipse p1 e →
    isOnEllipse p2 e →
    isOnParabola p1 pa →
    isOnParabola p2 pa →
    p1 ≠ p2 →
    ∃ (d : ℝ), d = 2 * |p1.y - p2.y| ∧ 
               d = Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2) :=
  sorry

end intersection_distance_l3702_370238


namespace circle_area_from_polar_equation_l3702_370280

theorem circle_area_from_polar_equation :
  ∀ (r : ℝ → ℝ) (θ : ℝ),
    (r θ = 3 * Real.cos θ - 4 * Real.sin θ) →
    (∃ (c : ℝ × ℝ) (R : ℝ), ∀ (x y : ℝ),
      (x - c.1)^2 + (y - c.2)^2 = R^2 ↔ 
      ∃ θ, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
    (π * (5/2)^2 = 25*π/4) :=
by sorry

end circle_area_from_polar_equation_l3702_370280


namespace sine_cosine_identity_l3702_370244

theorem sine_cosine_identity : Real.sin (20 * π / 180) * Real.cos (110 * π / 180) + Real.cos (160 * π / 180) * Real.sin (70 * π / 180) = -1 := by
  sorry

end sine_cosine_identity_l3702_370244


namespace no_real_arithmetic_progression_l3702_370224

theorem no_real_arithmetic_progression : ¬ ∃ (a b : ℝ), 
  (b - a = a - 15) ∧ (a * b - b = b - a) := by
  sorry

end no_real_arithmetic_progression_l3702_370224


namespace g_zero_values_l3702_370215

theorem g_zero_values (g : ℝ → ℝ) (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) :
  g 0 = 0 ∨ g 0 = 1 := by
sorry

end g_zero_values_l3702_370215
