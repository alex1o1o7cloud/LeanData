import Mathlib

namespace square_perimeters_sum_l3849_384950

theorem square_perimeters_sum (x y : ℝ) (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  4*x + 4*y = 44 := by
  sorry

end square_perimeters_sum_l3849_384950


namespace percentage_of_hindu_boys_l3849_384982

theorem percentage_of_hindu_boys (total_boys : ℕ) 
  (muslim_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percentage = 44 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 153 →
  (total_boys - (muslim_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 28 / 100 := by
  sorry

end percentage_of_hindu_boys_l3849_384982


namespace division_remainder_3005_98_l3849_384957

theorem division_remainder_3005_98 : ∃ q : ℤ, 3005 = 98 * q + 65 ∧ 0 ≤ 65 ∧ 65 < 98 := by
  sorry

end division_remainder_3005_98_l3849_384957


namespace max_ab_min_a2_b2_l3849_384952

theorem max_ab_min_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 2 * b = 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 2 → x * y ≤ a * b) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 2 → a^2 + b^2 ≤ x^2 + y^2) ∧
  a * b = 1/2 ∧ a^2 + b^2 = 4/5 := by
sorry

end max_ab_min_a2_b2_l3849_384952


namespace regular_polygon_interior_angle_sides_l3849_384954

theorem regular_polygon_interior_angle_sides : ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) / n = 150 →
  n = 12 := by
  sorry

end regular_polygon_interior_angle_sides_l3849_384954


namespace add_negative_numbers_l3849_384961

theorem add_negative_numbers : -10 + (-12) = -22 := by
  sorry

end add_negative_numbers_l3849_384961


namespace particular_propositions_count_l3849_384902

/-- A proposition is particular if it contains quantifiers like "some", "exists", or "some of". -/
def is_particular_proposition (p : Prop) : Prop := sorry

/-- The first proposition: Some triangles are isosceles triangles. -/
def prop1 : Prop := sorry

/-- The second proposition: There exists an integer x such that x^2 - 2x - 3 = 0. -/
def prop2 : Prop := sorry

/-- The third proposition: There exists a triangle whose sum of interior angles is 170°. -/
def prop3 : Prop := sorry

/-- The fourth proposition: Rectangles are parallelograms. -/
def prop4 : Prop := sorry

/-- The list of all given propositions. -/
def propositions : List Prop := [prop1, prop2, prop3, prop4]

/-- Count the number of particular propositions in a list. -/
def count_particular_propositions (props : List Prop) : Nat := sorry

theorem particular_propositions_count :
  count_particular_propositions propositions = 3 := by sorry

end particular_propositions_count_l3849_384902


namespace distinct_values_count_l3849_384914

def original_expression : ℕ → ℕ := λ n => 3^(3^(3^3))

def parenthesization1 : ℕ → ℕ := λ n => 3^((3^3)^3)
def parenthesization2 : ℕ → ℕ := λ n => 3^(3^(3^3 + 1))
def parenthesization3 : ℕ → ℕ := λ n => (3^(3^3))^3

theorem distinct_values_count :
  ∃ (S : Finset ℕ),
    S.card = 4 ∧
    (∀ x ∈ S, x ≠ original_expression 0) ∧
    (∀ x ∈ S, (x = parenthesization1 0) ∨ (x = parenthesization2 0) ∨ (x = parenthesization3 0) ∨
              (∃ y, x = 3^y ∧ y ≠ 3^(3^3))) :=
by sorry

end distinct_values_count_l3849_384914


namespace inequality_proof_l3849_384988

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1/2) :
  (1 - a^2 + c^2) / (c * (a + 2*b)) + 
  (1 - b^2 + a^2) / (a * (b + 2*c)) + 
  (1 - c^2 + b^2) / (b * (c + 2*a)) ≥ 6 := by
  sorry

end inequality_proof_l3849_384988


namespace units_digit_factorial_sum_l3849_384925

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum :
  unitsDigit (sumFactorials 15) = 3 :=
by
  sorry

/- Hint: You may want to use the following lemma -/
lemma units_digit_factorial_ge_5 (n : ℕ) (h : n ≥ 5) :
  unitsDigit (factorial n) = 0 :=
by
  sorry

end units_digit_factorial_sum_l3849_384925


namespace pencil_cartons_l3849_384963

/-- Given an order of pencils and erasers, prove the number of cartons of pencils -/
theorem pencil_cartons (pencil_cost eraser_cost total_cartons total_cost : ℕ) 
  (h1 : pencil_cost = 6)
  (h2 : eraser_cost = 3)
  (h3 : total_cartons = 100)
  (h4 : total_cost = 360) :
  ∃ (pencil_cartons eraser_cartons : ℕ),
    pencil_cartons + eraser_cartons = total_cartons ∧
    pencil_cost * pencil_cartons + eraser_cost * eraser_cartons = total_cost ∧
    pencil_cartons = 20 :=
by sorry

end pencil_cartons_l3849_384963


namespace natural_number_equation_solutions_l3849_384994

theorem natural_number_equation_solutions (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) := by
  sorry

end natural_number_equation_solutions_l3849_384994


namespace circplus_commutative_l3849_384923

/-- The ⊕ operation -/
def circplus (a b : ℝ) : ℝ := a^2 + a*b + b^2

/-- Theorem: x ⊕ y = y ⊕ x for all real x and y -/
theorem circplus_commutative : ∀ x y : ℝ, circplus x y = circplus y x := by
  sorry

end circplus_commutative_l3849_384923


namespace helium_pressure_change_l3849_384993

-- Define the variables and constants
variable (v₁ v₂ p₁ p₂ : ℝ)

-- State the given conditions
def initial_volume : ℝ := 3.6
def initial_pressure : ℝ := 8
def final_volume : ℝ := 4.5

-- Define the inverse proportionality relationship
def inverse_proportional (v₁ v₂ p₁ p₂ : ℝ) : Prop :=
  v₁ * p₁ = v₂ * p₂

-- State the theorem
theorem helium_pressure_change :
  v₁ = initial_volume →
  p₁ = initial_pressure →
  v₂ = final_volume →
  inverse_proportional v₁ v₂ p₁ p₂ →
  p₂ = 6.4 := by
  sorry

end helium_pressure_change_l3849_384993


namespace unique_three_digit_factorial_product_l3849_384922

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_of (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_three_digit_factorial_product :
  ∃! n : ℕ, is_three_digit n ∧
    let (a, b, c) := digits_of n
    2 * n = 3 * (factorial a * factorial b * factorial c) :=
by
  sorry

end unique_three_digit_factorial_product_l3849_384922


namespace plane_division_theorem_l3849_384917

/-- The number of regions formed by h horizontal lines and s non-horizontal lines -/
def num_regions (h s : ℕ) : ℕ := h * (s + 1) + 1 + s * (s + 1) / 2

/-- The set of valid solutions for (h, s) -/
def valid_solutions : Set (ℕ × ℕ) :=
  {(995, 1), (176, 10), (80, 21)}

theorem plane_division_theorem :
  ∀ h s : ℕ, h > 0 ∧ s > 0 →
    (num_regions h s = 1992 ↔ (h, s) ∈ valid_solutions) := by
  sorry

#check plane_division_theorem

end plane_division_theorem_l3849_384917


namespace isosceles_triangle_base_length_l3849_384910

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  perimeter : ℝ
  is_isosceles : side ≥ 0 ∧ base ≥ 0 ∧ perimeter = 2 * side + base

-- Theorem statement
theorem isosceles_triangle_base_length 
  (t : IsoscelesTriangle) 
  (h_perimeter : t.perimeter = 26) 
  (h_side : t.side = 11 ∨ t.base = 11) : 
  t.base = 11 ∨ t.base = 7.5 := by
  sorry


end isosceles_triangle_base_length_l3849_384910


namespace max_intersections_theorem_l3849_384968

/-- A polygon in a plane -/
structure Polygon where
  sides : ℕ

/-- A regular polygon -/
structure RegularPolygon extends Polygon

/-- An irregular polygon -/
structure IrregularPolygon extends Polygon

/-- Two polygons that overlap but share no complete side -/
structure OverlappingPolygons where
  P₁ : RegularPolygon
  P₂ : IrregularPolygon
  overlap : Bool
  no_shared_side : Bool

/-- The maximum number of intersection points between two polygons -/
def max_intersections (op : OverlappingPolygons) : ℕ :=
  op.P₁.sides * op.P₂.sides

/-- Theorem: The maximum number of intersections between a regular polygon P₁
    and an irregular polygon P₂, where they overlap but share no complete side,
    is the product of their number of sides -/
theorem max_intersections_theorem (op : OverlappingPolygons)
    (h : op.P₁.sides ≤ op.P₂.sides) :
    max_intersections op = op.P₁.sides * op.P₂.sides :=
  sorry

end max_intersections_theorem_l3849_384968


namespace strictly_increasing_interval_l3849_384932

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem strictly_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 3) (k * π + π / 6)) :=
sorry

end strictly_increasing_interval_l3849_384932


namespace special_polyhedron_volume_l3849_384998

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  -- The polyhedron is convex
  isConvex : Bool
  -- Number of square faces
  numSquareFaces : Nat
  -- Number of hexagonal faces
  numHexagonalFaces : Nat
  -- No two square faces share a vertex
  noSharedSquareVertices : Bool
  -- All edges have unit length
  unitEdgeLength : Bool

/-- The volume of the special polyhedron -/
noncomputable def specialPolyhedronVolume (p : SpecialPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the special polyhedron -/
theorem special_polyhedron_volume :
  ∀ (p : SpecialPolyhedron),
    p.isConvex = true ∧
    p.numSquareFaces = 6 ∧
    p.numHexagonalFaces = 8 ∧
    p.noSharedSquareVertices = true ∧
    p.unitEdgeLength = true →
    specialPolyhedronVolume p = 8 * Real.sqrt 2 :=
by
  sorry

end special_polyhedron_volume_l3849_384998


namespace samuel_has_twelve_apples_left_l3849_384948

/-- The number of apples Samuel has left after buying, eating, and making pie -/
def samuels_remaining_apples (bonnies_apples : ℕ) (samuels_extra_apples : ℕ) : ℕ :=
  let samuels_apples := bonnies_apples + samuels_extra_apples
  let after_eating := samuels_apples / 2
  let used_for_pie := after_eating / 7
  after_eating - used_for_pie

/-- Theorem stating that Samuel has 12 apples left -/
theorem samuel_has_twelve_apples_left :
  samuels_remaining_apples 8 20 = 12 := by
  sorry

#eval samuels_remaining_apples 8 20

end samuel_has_twelve_apples_left_l3849_384948


namespace smallest_x_and_y_l3849_384985

theorem smallest_x_and_y (x y : ℕ+) (h : (3 : ℚ) / 4 = y / (242 + x)) : 
  (x = 2 ∧ y = 183) ∧ ∀ (x' y' : ℕ+), ((3 : ℚ) / 4 = y' / (242 + x')) → x ≤ x' :=
sorry

end smallest_x_and_y_l3849_384985


namespace complex_expression_magnitude_l3849_384986

theorem complex_expression_magnitude : 
  Complex.abs ((18 - 5 * Complex.I) * (14 + 6 * Complex.I) - (3 - 12 * Complex.I) * (4 + 9 * Complex.I)) = Real.sqrt 146365 := by
  sorry

end complex_expression_magnitude_l3849_384986


namespace probability_theorem_l3849_384940

def total_cups : ℕ := 8
def white_cups : ℕ := 3
def red_cups : ℕ := 3
def black_cups : ℕ := 2
def selected_cups : ℕ := 5

def probability_specific_sequence : ℚ :=
  (white_cups * (white_cups - 1) * red_cups * (red_cups - 1) * black_cups) /
  (total_cups * (total_cups - 1) * (total_cups - 2) * (total_cups - 3) * (total_cups - 4))

def number_of_arrangements : ℕ := Nat.factorial selected_cups / 
  (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

theorem probability_theorem :
  probability_specific_sequence * number_of_arrangements = 9 / 28 := by
  sorry

end probability_theorem_l3849_384940


namespace max_a_value_l3849_384929

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- The condition for m -/
def MCondition (m a : ℚ) : Prop := 1/2 < m ∧ m < a

/-- The main theorem -/
theorem max_a_value :
  ∃ (a : ℚ), a = 75/149 ∧
  (∀ (m : ℚ), MCondition m a →
    ∀ (x y : ℤ), 0 < x → x ≤ 150 → LatticePoint x y → ¬LineEquation m x y) ∧
  (∀ (a' : ℚ), a < a' →
    ∃ (m : ℚ), MCondition m a' ∧
    ∃ (x y : ℤ), 0 < x ∧ x ≤ 150 ∧ LatticePoint x y ∧ LineEquation m x y) :=
sorry

end max_a_value_l3849_384929


namespace triangle_shape_l3849_384995

theorem triangle_shape (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a * Real.cos A = b * Real.cos B) →
  (A = B ∨ A + B = Real.pi / 2) :=
sorry

end triangle_shape_l3849_384995


namespace limit_of_sequence_l3849_384984

def a (n : ℕ) : ℚ := (5 * n + 1) / (10 * n - 3)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 1/2| < ε :=
sorry

end limit_of_sequence_l3849_384984


namespace range_of_H_l3849_384900

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2|^2 - |x - 2|^2

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, ∃ x : ℝ, H x = y :=
sorry

end range_of_H_l3849_384900


namespace max_inverse_sum_14th_power_l3849_384989

/-- A quadratic polynomial x^2 - tx + q with roots r1 and r2 -/
structure QuadraticPolynomial where
  t : ℝ
  q : ℝ
  r1 : ℝ
  r2 : ℝ
  is_root : r1^2 - t*r1 + q = 0 ∧ r2^2 - t*r2 + q = 0

/-- The condition that the sum of powers of roots are equal up to 13th power -/
def equal_sum_powers (p : QuadraticPolynomial) : Prop :=
  ∀ n : ℕ, n ≤ 13 → p.r1^n + p.r2^n = p.r1 + p.r2

/-- The theorem statement -/
theorem max_inverse_sum_14th_power (p : QuadraticPolynomial) 
  (h : equal_sum_powers p) : 
  (∀ p' : QuadraticPolynomial, equal_sum_powers p' → 
    1 / p'.r1^14 + 1 / p'.r2^14 ≤ 1 / p.r1^14 + 1 / p.r2^14) →
  1 / p.r1^14 + 1 / p.r2^14 = 2 :=
sorry

end max_inverse_sum_14th_power_l3849_384989


namespace smallest_a_is_eight_l3849_384977

-- Define the polynomial function
def f (a x : ℤ) : ℤ := x^4 + a^2 + 2*a*x

-- Define what it means for a number to be composite
def is_composite (n : ℤ) : Prop := ∃ m k : ℤ, m > 1 ∧ k > 1 ∧ n = m * k

-- State the theorem
theorem smallest_a_is_eight :
  (∀ x : ℤ, is_composite (f 8 x)) ∧
  (∀ a : ℤ, 0 < a → a < 8 → ∃ x : ℤ, ¬ is_composite (f a x)) :=
sorry

end smallest_a_is_eight_l3849_384977


namespace newspaper_photos_l3849_384911

/-- The total number of photos in a newspaper -/
def total_photos (pages_with_4 : ℕ) (pages_with_6 : ℕ) : ℕ :=
  pages_with_4 * 4 + pages_with_6 * 6

/-- Theorem stating that the total number of photos is 208 -/
theorem newspaper_photos : total_photos 25 18 = 208 := by
  sorry

end newspaper_photos_l3849_384911


namespace quadratic_equation_k_l3849_384975

/-- Given a quadratic equation x^2 - 3x + k = 0 with two real roots a and b,
    if ab + 2a + 2b = 1, then k = -5 -/
theorem quadratic_equation_k (a b k : ℝ) :
  (∀ x, x^2 - 3*x + k = 0 ↔ x = a ∨ x = b) →
  (a*b + 2*a + 2*b = 1) →
  k = -5 := by
  sorry

end quadratic_equation_k_l3849_384975


namespace smallest_n_divisible_by_24_and_864_l3849_384955

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(864 ∣ m^3))) ∧ 
  (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ n = 12 := by
  sorry

end smallest_n_divisible_by_24_and_864_l3849_384955


namespace stratified_sampling_proportion_l3849_384974

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the sample sizes for each grade --/
structure SampleSizes where
  grade10 : ℕ
  grade11 : ℕ

/-- Checks if the sampling is proportional across grades --/
def isProportionalSampling (dist : GradeDistribution) (sample : SampleSizes) : Prop :=
  (dist.grade10 : ℚ) / sample.grade10 = (dist.grade11 : ℚ) / sample.grade11

theorem stratified_sampling_proportion 
  (dist : GradeDistribution)
  (sample : SampleSizes)
  (h1 : dist.grade10 = 50)
  (h2 : dist.grade11 = 40)
  (h3 : dist.grade12 = 40)
  (h4 : sample.grade11 = 8)
  (h5 : isProportionalSampling dist sample) :
  sample.grade10 = 10 := by
  sorry

end stratified_sampling_proportion_l3849_384974


namespace inequality_proof_l3849_384976

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := by
  sorry

end inequality_proof_l3849_384976


namespace centipede_sock_shoe_orders_l3849_384966

/-- The number of legs a centipede has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid orders for a centipede to put on socks and shoes -/
def valid_orders : ℕ := Nat.factorial total_items / (2 ^ num_legs)

/-- Theorem stating the number of valid orders for a centipede to put on socks and shoes -/
theorem centipede_sock_shoe_orders :
  valid_orders = Nat.factorial total_items / (2 ^ num_legs) :=
by sorry

end centipede_sock_shoe_orders_l3849_384966


namespace circle_intersection_exists_l3849_384981

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (A B : Point)
variable (S : Circle)
variable (α : ℝ)

-- Define the intersection angle between two circles
def intersectionAngle (c1 c2 : Circle) : ℝ := sorry

-- Define a function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Theorem statement
theorem circle_intersection_exists :
  ∃ (C : Circle), isOnCircle A C ∧ isOnCircle B C ∧ intersectionAngle C S = α := by sorry

end circle_intersection_exists_l3849_384981


namespace cos_pi_half_plus_alpha_l3849_384978

theorem cos_pi_half_plus_alpha (α : Real) 
  (h : (Real.sin (π + α) * Real.cos (-α + 4*π)) / Real.cos α = 1/2) : 
  Real.cos (π/2 + α) = 1/2 := by
sorry

end cos_pi_half_plus_alpha_l3849_384978


namespace money_distribution_l3849_384980

theorem money_distribution (x : ℝ) (h : x > 0) :
  let moe_original := 6 * x
  let loki_original := 5 * x
  let kai_original := 2 * x
  let total_original := moe_original + loki_original + kai_original
  let ott_received := 3 * x
  ott_received / total_original = 3 / 13 := by
sorry

end money_distribution_l3849_384980


namespace joanne_earnings_theorem_l3849_384937

/-- Calculates Joanne's total weekly earnings based on her work schedule and pay rates -/
def joanne_weekly_earnings (main_job_hours_per_day : ℕ) (main_job_rate : ℚ) 
  (part_time_hours_per_day : ℕ) (part_time_rate : ℚ) (days_per_week : ℕ) : ℚ :=
  (main_job_hours_per_day * main_job_rate + part_time_hours_per_day * part_time_rate) * days_per_week

/-- Theorem stating that Joanne's weekly earnings are $775.00 -/
theorem joanne_earnings_theorem : 
  joanne_weekly_earnings 8 16 2 (27/2) 5 = 775 := by
  sorry

end joanne_earnings_theorem_l3849_384937


namespace eduardo_classes_l3849_384918

theorem eduardo_classes (x : ℕ) : 
  x + 2 * x = 9 → x = 3 := by sorry

end eduardo_classes_l3849_384918


namespace A_intersect_B_l3849_384901

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x < 3}

theorem A_intersect_B : A ∩ B = {1, 2} := by sorry

end A_intersect_B_l3849_384901


namespace sqrt_sum_upper_bound_l3849_384964

theorem sqrt_sum_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 := by
  sorry

end sqrt_sum_upper_bound_l3849_384964


namespace sum_of_binary_digits_315_l3849_384987

/-- The sum of the digits in the binary representation of 315 is 6 -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end sum_of_binary_digits_315_l3849_384987


namespace triangle_area_10_24_26_l3849_384913

/-- The area of a triangle with side lengths 10, 24, and 26 is 120 -/
theorem triangle_area_10_24_26 : 
  ∀ (a b c area : ℝ), 
    a = 10 → b = 24 → c = 26 →
    (a * a + b * b = c * c) →  -- Pythagorean theorem condition
    area = (1/2) * a * b →
    area = 120 := by sorry

end triangle_area_10_24_26_l3849_384913


namespace fifteen_apples_solution_l3849_384912

/-- The number of friends sharing the apples -/
def num_friends : ℕ := 5

/-- The function representing the number of apples remaining after each friend takes their share -/
def apples_remaining (initial_apples : ℚ) (friend : ℕ) : ℚ :=
  match friend with
  | 0 => initial_apples
  | n + 1 => (apples_remaining initial_apples n / 2) - (1 / 2)

/-- The theorem stating that 15 is the correct initial number of apples -/
theorem fifteen_apples_solution :
  ∃ (initial_apples : ℚ),
    initial_apples = 15 ∧
    apples_remaining initial_apples num_friends = 0 := by
  sorry

end fifteen_apples_solution_l3849_384912


namespace impossibility_of_measuring_one_liter_l3849_384905

theorem impossibility_of_measuring_one_liter :
  ¬ ∃ (k l : ℤ), k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1 := by
  sorry

end impossibility_of_measuring_one_liter_l3849_384905


namespace enclosed_area_circular_arcs_l3849_384967

/-- The area enclosed by a curve composed of 9 congruent circular arcs -/
theorem enclosed_area_circular_arcs (n : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) 
  (h1 : n = 9)
  (h2 : arc_length = 5 * π / 6)
  (h3 : hexagon_side = 3) :
  ∃ (area : ℝ), area = 13.5 * Real.sqrt 3 + 375 * π / 8 := by
  sorry

end enclosed_area_circular_arcs_l3849_384967


namespace unique_diametric_circle_l3849_384930

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ (i j : Fin 3), i ≠ j → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 
    Real.sqrt ((vertices 0).1 - (vertices 1).1)^2 + ((vertices 0).2 - (vertices 1).2)^2

/-- A circle defined by two points as its diameter -/
structure DiametricCircle (T : EquilateralTriangle) where
  endpoint1 : Fin 3
  endpoint2 : Fin 3
  is_diameter : endpoint1 ≠ endpoint2

/-- The theorem stating that there's only one unique diametric circle for an equilateral triangle -/
theorem unique_diametric_circle (T : EquilateralTriangle) : 
  ∃! (c : DiametricCircle T), True := by sorry

end unique_diametric_circle_l3849_384930


namespace complement_M_in_U_l3849_384997

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | 2 * x - x^2 > 0}

-- Statement to prove
theorem complement_M_in_U : 
  {x : ℝ | x ∈ U ∧ x ∉ M} = {x : ℝ | x ≥ 2} := by sorry

end complement_M_in_U_l3849_384997


namespace quadratic_root_l3849_384921

/-- A quadratic polynomial with coefficients a, b, and c. -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Predicate to check if a quadratic polynomial has exactly one root. -/
def has_one_root (a b c : ℝ) : Prop := b^2 = 4 * a * c

theorem quadratic_root (a b c : ℝ) (ha : a ≠ 0) :
  has_one_root a b c →
  has_one_root (-a) (b - 30*a) (17*a - 7*b + c) →
  ∃! x : ℝ, quadratic a b c x = 0 ∧ x = -11 := by sorry

end quadratic_root_l3849_384921


namespace algebraic_simplification_l3849_384941

theorem algebraic_simplification (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end algebraic_simplification_l3849_384941


namespace candy_distribution_l3849_384951

theorem candy_distribution (total : ℝ) (total_pos : total > 0) : 
  let initial_shares := [4/10, 3/10, 2/10, 1/10]
  let first_round := initial_shares.map (· * total)
  let remaining_after_first := total - first_round.sum
  let second_round := initial_shares.map (· * remaining_after_first)
  remaining_after_first - second_round.sum = 0 := by
sorry

end candy_distribution_l3849_384951


namespace range_of_a_l3849_384996

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l3849_384996


namespace parallel_vectors_y_value_l3849_384928

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (6, y)
  parallel a b → y = 4 := by
sorry

end parallel_vectors_y_value_l3849_384928


namespace pizza_slice_volume_l3849_384992

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) 
  (h1 : thickness = 1/4)
  (h2 : diameter = 16)
  (h3 : num_slices = 8) :
  (π * (diameter/2)^2 * thickness) / num_slices = 2 * π := by
  sorry

end pizza_slice_volume_l3849_384992


namespace exponent_simplification_l3849_384958

theorem exponent_simplification :
  (1 : ℝ) / ((5 : ℝ)^2)^4 * (5 : ℝ)^15 = (5 : ℝ)^7 := by sorry

end exponent_simplification_l3849_384958


namespace circle_radius_zero_l3849_384973

/-- The radius of a circle given by the equation 4x^2 + 8x + 4y^2 - 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  4*x^2 + 8*x + 4*y^2 - 16*y + 20 = 0 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 := by
sorry

end circle_radius_zero_l3849_384973


namespace triangle_trig_identity_l3849_384956

/-- Given a triangle ABC with sides AB = 6, AC = 5, and BC = 4,
    prove that (cos((A - B)/2) / sin(C/2)) - (sin((A - B)/2) / cos(C/2)) = 5/3 -/
theorem triangle_trig_identity (A B C : ℝ) (hABC : A + B + C = π) 
  (hAB : Real.cos A * 6 = Real.cos B * 5 + Real.cos C * 4)
  (hBC : Real.cos B * 4 = Real.cos C * 5 + Real.cos A * 6)
  (hAC : Real.cos C * 5 = Real.cos A * 6 + Real.cos B * 4) :
  (Real.cos ((A - B)/2) / Real.sin (C/2)) - (Real.sin ((A - B)/2) / Real.cos (C/2)) = 5/3 := by
  sorry

end triangle_trig_identity_l3849_384956


namespace arctan_sum_equals_pi_half_l3849_384943

theorem arctan_sum_equals_pi_half (n : ℕ+) :
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/n) = π/2 → n = 54 := by
  sorry

end arctan_sum_equals_pi_half_l3849_384943


namespace complex_exponential_sum_l3849_384962

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1/2 + (5/4) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1/2 - (5/4) * Complex.I :=
by sorry

end complex_exponential_sum_l3849_384962


namespace tangent_line_is_perpendicular_and_tangent_l3849_384953

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

-- Define the given curve
def given_curve (x y : ℝ) : Prop := y = x^3 + 3 * x^2 - 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x + y + 6 = 0

-- Theorem statement
theorem tangent_line_is_perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    given_curve x₀ y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is perpendicular to the given line
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ ∧ given_line x₂ y₂ ∧ x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * ((y₀ + 6) / (-3) - y₀) / (((y₀ + 6) / (-3)) - x₀) = -1) ∧
    -- The tangent line is indeed tangent to the curve
    (∀ (x : ℝ), x ≠ x₀ → ∃ (y : ℝ), given_curve x y ∧ ¬tangent_line x y) :=
sorry

end tangent_line_is_perpendicular_and_tangent_l3849_384953


namespace ratio_a_to_d_l3849_384947

theorem ratio_a_to_d (a b c d : ℚ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (hcd : c / d = 5 / 7) :
  a / d = 1 / 3 := by sorry

end ratio_a_to_d_l3849_384947


namespace jogger_distance_l3849_384924

theorem jogger_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  actual_speed = 12 →
  faster_speed = 16 →
  extra_distance = 10 →
  (∃ time : ℝ, time > 0 ∧ faster_speed * time = actual_speed * time + extra_distance) →
  actual_speed * (extra_distance / (faster_speed - actual_speed)) = 30 :=
by
  sorry

end jogger_distance_l3849_384924


namespace equation_solution_l3849_384903

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3 ↔ x = 1 := by
sorry

end equation_solution_l3849_384903


namespace min_integer_solution_l3849_384945

def is_solution (x : ℤ) : Prop :=
  (3 - x > 0) ∧ ((4 * x : ℚ) / 3 + 3 / 2 > -x / 6)

theorem min_integer_solution :
  is_solution 0 ∧ ∀ y : ℤ, y < 0 → ¬is_solution y :=
sorry

end min_integer_solution_l3849_384945


namespace sum_equals_270_l3849_384938

/-- The sum of the arithmetic sequence with first term a, common difference d, and n terms -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

/-- The sum of two arithmetic sequences, each with 5 terms and common difference 10 -/
def two_sequence_sum (a₁ a₂ : ℕ) : ℕ := arithmetic_sum a₁ 10 5 + arithmetic_sum a₂ 10 5

theorem sum_equals_270 : two_sequence_sum 3 11 = 270 := by
  sorry

end sum_equals_270_l3849_384938


namespace marks_age_relation_l3849_384949

/-- Proves that Mark's age will be 2 years more than twice Aaron's age in 4 years -/
theorem marks_age_relation (mark_current_age aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age - 3 = 3 * (aaron_current_age - 3) + 1 →
  (mark_current_age + 4) = 2 * (aaron_current_age + 4) + 2 := by
  sorry

end marks_age_relation_l3849_384949


namespace green_balls_count_l3849_384926

theorem green_balls_count (total : ℕ) (blue : ℕ) : 
  total = 40 → 
  blue = 11 → 
  ∃ (red green : ℕ), 
    red = 2 * blue ∧ 
    green = total - (red + blue) ∧ 
    green = 7 := by
  sorry

end green_balls_count_l3849_384926


namespace same_color_probability_l3849_384990

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def drawn_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

theorem same_color_probability : 
  (Nat.choose red_marbles drawn_marbles + 
   Nat.choose white_marbles drawn_marbles + 
   Nat.choose blue_marbles drawn_marbles : ℚ) / 
  (Nat.choose total_marbles drawn_marbles : ℚ) = 11 / 612 :=
sorry

end same_color_probability_l3849_384990


namespace circle_radius_given_area_circumference_ratio_l3849_384904

theorem circle_radius_given_area_circumference_ratio 
  (A C : ℝ) (h1 : A > 0) (h2 : C > 0) (h3 : A / C = 10) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 20 := by
  sorry

end circle_radius_given_area_circumference_ratio_l3849_384904


namespace line_slope_l3849_384931

/-- A straight line in the xy-plane with y-intercept 10 and passing through (100, 1000) has slope 9.9 -/
theorem line_slope (f : ℝ → ℝ) (h1 : f 0 = 10) (h2 : f 100 = 1000) :
  (f 100 - f 0) / (100 - 0) = 9.9 := by
  sorry

end line_slope_l3849_384931


namespace collinear_points_k_value_l3849_384908

/-- Given three vectors OA, OB, and OC in ℝ², where points A, B, and C are collinear,
    prove that the x-coordinate of OA is 18. -/
theorem collinear_points_k_value (k : ℝ) :
  let OA : ℝ × ℝ := (k, 12)
  let OB : ℝ × ℝ := (4, 5)
  let OC : ℝ × ℝ := (10, 8)
  (∃ (t : ℝ), OC - OA = t • (OB - OA)) →
  k = 18 := by
  sorry

end collinear_points_k_value_l3849_384908


namespace raised_beds_planks_l3849_384965

/-- Calculates the number of 8-foot long planks needed for raised beds --/
def planks_needed (num_beds : ℕ) (bed_height : ℕ) (bed_width : ℕ) (bed_length : ℕ) (plank_width : ℕ) (plank_length : ℕ) : ℕ :=
  let long_sides := 2 * bed_height
  let short_sides := 2 * bed_height * bed_width / plank_length
  let planks_per_bed := long_sides + short_sides
  num_beds * planks_per_bed

theorem raised_beds_planks :
  planks_needed 10 2 2 8 1 8 = 50 := by
  sorry

end raised_beds_planks_l3849_384965


namespace average_of_tenths_and_thousandths_l3849_384972

theorem average_of_tenths_and_thousandths :
  let a : ℚ := 4/10  -- 4 tenths
  let b : ℚ := 5/1000  -- 5 thousandths
  (a + b) / 2 = 2025/10000 := by
sorry

end average_of_tenths_and_thousandths_l3849_384972


namespace probability_solution_l3849_384942

def probability_equation (p q : ℝ) : Prop :=
  q = 1 - p ∧ 
  (Nat.choose 10 7 : ℝ) * p^7 * q^3 = (Nat.choose 10 6 : ℝ) * p^6 * q^4

theorem probability_solution :
  ∀ p q : ℝ, probability_equation p q → p = 7/11 := by
  sorry

end probability_solution_l3849_384942


namespace sum_ge_sum_sqrt_products_l3849_384979

theorem sum_ge_sum_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) := by
  sorry

end sum_ge_sum_sqrt_products_l3849_384979


namespace calculation_proof_l3849_384939

theorem calculation_proof : (27 * 0.92 * 0.85) / (23 * 1.7 * 1.8) = 0.3 := by
  sorry

end calculation_proof_l3849_384939


namespace temp_increase_proof_l3849_384935

-- Define the temperatures
def last_night_temp : Int := -5
def current_temp : Int := 3

-- Define the temperature difference function
def temp_difference (t1 t2 : Int) : Int := t2 - t1

-- Theorem to prove
theorem temp_increase_proof : 
  temp_difference last_night_temp current_temp = 8 := by
  sorry

end temp_increase_proof_l3849_384935


namespace geometric_sequence_product_l3849_384915

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 3 →
  a 6 = 1 / 9 →
  a 4 * a 5 = 1 / 3 := by
  sorry

end geometric_sequence_product_l3849_384915


namespace nigella_sold_three_houses_l3849_384909

/-- Represents a house with its cost -/
structure House where
  cost : ℝ

/-- Represents a realtor's earnings -/
structure RealtorEarnings where
  baseSalary : ℝ
  commissionRate : ℝ
  totalEarnings : ℝ

def calculateCommission (house : House) (commissionRate : ℝ) : ℝ :=
  house.cost * commissionRate

def nigellaEarnings : RealtorEarnings := {
  baseSalary := 3000
  commissionRate := 0.02
  totalEarnings := 8000
}

def houseA : House := { cost := 60000 }
def houseB : House := { cost := 3 * houseA.cost }
def houseC : House := { cost := 2 * houseA.cost - 110000 }

theorem nigella_sold_three_houses :
  let commission := calculateCommission houseA nigellaEarnings.commissionRate +
                    calculateCommission houseB nigellaEarnings.commissionRate +
                    calculateCommission houseC nigellaEarnings.commissionRate
  nigellaEarnings.totalEarnings = nigellaEarnings.baseSalary + commission ∧
  (houseA.cost > 0 ∧ houseB.cost > 0 ∧ houseC.cost > 0) →
  3 = 3 := by
  sorry

#check nigella_sold_three_houses

end nigella_sold_three_houses_l3849_384909


namespace division_remainder_zero_l3849_384916

theorem division_remainder_zero (dividend : ℝ) (divisor : ℝ) (quotient : ℝ) 
  (h1 : dividend = 57843.67)
  (h2 : divisor = 1242.51)
  (h3 : quotient = 46.53) :
  dividend - divisor * quotient = 0 := by
  sorry

end division_remainder_zero_l3849_384916


namespace multiple_of_nine_square_greater_than_144_less_than_30_l3849_384999

theorem multiple_of_nine_square_greater_than_144_less_than_30 (x : ℕ) :
  (∃ k : ℕ, x = 9 * k) →
  x^2 > 144 →
  x < 30 →
  x = 18 ∨ x = 27 := by
sorry

end multiple_of_nine_square_greater_than_144_less_than_30_l3849_384999


namespace dogs_in_park_l3849_384919

theorem dogs_in_park (total_legs : ℕ) (legs_per_dog : ℕ) (h1 : total_legs = 436) (h2 : legs_per_dog = 4) :
  total_legs / legs_per_dog = 109 := by
  sorry

end dogs_in_park_l3849_384919


namespace find_a_l3849_384969

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else -x

-- State the theorem
theorem find_a : ∃ (a : ℝ), f (1/3) = (1/3) * f a ∧ a = 1/27 := by
  sorry

end find_a_l3849_384969


namespace third_term_of_geometric_sequence_l3849_384927

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 4) :
  a 3 = 8 := by
sorry

end third_term_of_geometric_sequence_l3849_384927


namespace min_area_quadrilateral_on_parabola_l3849_384944

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4*x

/-- Point on a parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : par.eq x y

/-- Chord of a parabola -/
structure Chord (par : Parabola) where
  p1 : PointOnParabola par
  p2 : PointOnParabola par

/-- Theorem: Minimum area of quadrilateral ABCD -/
theorem min_area_quadrilateral_on_parabola (par : Parabola)
  (A B C D : PointOnParabola par) 
  (chord_AC chord_BD : Chord par)
  (perp : chord_AC.p1.x = A.x ∧ chord_AC.p1.y = A.y ∧ 
          chord_AC.p2.x = C.x ∧ chord_AC.p2.y = C.y ∧
          chord_BD.p1.x = B.x ∧ chord_BD.p1.y = B.y ∧
          chord_BD.p2.x = D.x ∧ chord_BD.p2.y = D.y ∧
          (chord_AC.p2.y - chord_AC.p1.y) * (chord_BD.p2.y - chord_BD.p1.y) = 
          -(chord_AC.p2.x - chord_AC.p1.x) * (chord_BD.p2.x - chord_BD.p1.x))
  (through_focus : ∃ t : ℝ, 
    chord_AC.p1.x + t * (chord_AC.p2.x - chord_AC.p1.x) = par.p / 2 ∧
    chord_AC.p1.y + t * (chord_AC.p2.y - chord_AC.p1.y) = 0 ∧
    chord_BD.p1.x + t * (chord_BD.p2.x - chord_BD.p1.x) = par.p / 2 ∧
    chord_BD.p1.y + t * (chord_BD.p2.y - chord_BD.p1.y) = 0) :
  ∃ area : ℝ, area ≥ 32 ∧ 
    area = (1/2) * Real.sqrt ((A.x - C.x)^2 + (A.y - C.y)^2) * 
                    Real.sqrt ((B.x - D.x)^2 + (B.y - D.y)^2) := by
  sorry

end min_area_quadrilateral_on_parabola_l3849_384944


namespace regression_line_equation_specific_regression_line_equation_l3849_384906

/-- The regression line equation given the y-intercept and a point it passes through -/
theorem regression_line_equation (a : ℝ) (x_center y_center : ℝ) :
  let b := (y_center - a) / x_center
  (∀ x y, y = b * x + a) → (y_center = b * x_center + a) :=
by
  sorry

/-- The specific regression line equation for the given problem -/
theorem specific_regression_line_equation :
  let a := 0.2
  let x_center := 4
  let y_center := 5
  let b := (y_center - a) / x_center
  (∀ x y, y = b * x + a) ∧ (y_center = b * x_center + a) ∧ (b = 1.2) :=
by
  sorry

end regression_line_equation_specific_regression_line_equation_l3849_384906


namespace roberts_trip_l3849_384960

/-- Proves that given the conditions of Robert's trip, the return trip takes 2.5 hours -/
theorem roberts_trip (distance : ℝ) (outbound_time : ℝ) (saved_time : ℝ) (avg_speed : ℝ) :
  distance = 180 →
  outbound_time = 3 →
  saved_time = 0.5 →
  avg_speed = 80 →
  (2 * distance) / (outbound_time + (outbound_time + saved_time - 2 * saved_time) - 2 * saved_time) = avg_speed →
  outbound_time + saved_time - 2 * saved_time = 2.5 := by
  sorry

end roberts_trip_l3849_384960


namespace largest_inscribed_circle_in_right_triangle_l3849_384959

/-- For a right-angled triangle with perimeter k, the radius r of the largest inscribed circle
    is given by r = k/2 * (3 - 2√2). -/
theorem largest_inscribed_circle_in_right_triangle (k : ℝ) (h : k > 0) :
  ∃ (r : ℝ), r = k / 2 * (3 - 2 * Real.sqrt 2) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a + b + c = k →   -- perimeter condition
  2 * (a * b) / (a + b + c) ≤ r :=
by sorry

end largest_inscribed_circle_in_right_triangle_l3849_384959


namespace consecutive_numbers_problem_l3849_384946

theorem consecutive_numbers_problem (x y z w : ℚ) : 
  x > y ∧ y > z ∧  -- x, y, z are consecutive and in descending order
  w > x ∧  -- w is greater than x
  w = (5/3) * x ∧  -- ratio of x to w is 3:5
  w^2 = x * z ∧  -- w^2 = xz
  2*x + 3*y + 3*z = 5*y + 11 ∧  -- given equation
  x - y = y - z  -- x, y, z are equally spaced
  → z = 3 := by sorry

end consecutive_numbers_problem_l3849_384946


namespace inequality_relations_l3849_384933

theorem inequality_relations (r p q : ℝ) 
  (hr : r > 0) (hp : p > 0) (hq : q > 0) (hpq : p^2 * r > q^2 * r) : 
  p > q ∧ |p| > |q| ∧ 1/p < 1/q := by sorry

end inequality_relations_l3849_384933


namespace sweets_distribution_l3849_384983

theorem sweets_distribution (total_sweets : ℕ) (num_children : ℕ) (remaining_fraction : ℚ) 
  (h1 : total_sweets = 288)
  (h2 : num_children = 48)
  (h3 : remaining_fraction = 1 / 3)
  : (total_sweets * (1 - remaining_fraction)) / num_children = 4 := by
  sorry

end sweets_distribution_l3849_384983


namespace g_of_3_equals_5_l3849_384907

-- Define the function g
def g (y : ℝ) : ℝ := 2 * (y - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end g_of_3_equals_5_l3849_384907


namespace find_first_number_l3849_384934

theorem find_first_number (x : ℝ) (y : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (y + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 128 := by
sorry

end find_first_number_l3849_384934


namespace floor_neg_seven_fourths_l3849_384970

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_neg_seven_fourths_l3849_384970


namespace odd_function_zero_value_l3849_384991

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero_value (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end odd_function_zero_value_l3849_384991


namespace journey_distance_l3849_384936

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 40 ∧ speed1 = 20 ∧ speed2 = 30 →
  ∃ (distance : ℝ), 
    distance / speed1 / 2 + distance / speed2 / 2 = total_time ∧ 
    distance = 960 := by
  sorry

end journey_distance_l3849_384936


namespace square_difference_plus_constant_l3849_384920

theorem square_difference_plus_constant : (262^2 - 258^2) + 150 = 2230 := by
  sorry

end square_difference_plus_constant_l3849_384920


namespace min_value_of_m_range_of_x_l3849_384971

-- Define the conditions
def conditions (a b m : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^2 + b^2 = 9/2 ∧ a + b ≤ m

-- Part I: Minimum value of m
theorem min_value_of_m (a b m : ℝ) (h : conditions a b m) :
  m ≥ 3 :=
sorry

-- Part II: Range of x
theorem range_of_x (x : ℝ) :
  (∀ a b m, conditions a b m → 2*|x-1| + |x| ≥ a + b) →
  x ≤ -1/3 ∨ x ≥ 5/3 :=
sorry

end min_value_of_m_range_of_x_l3849_384971
