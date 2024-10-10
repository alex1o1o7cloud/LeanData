import Mathlib

namespace tan_equality_with_period_l409_40960

theorem tan_equality_with_period (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = 30 := by
  sorry

end tan_equality_with_period_l409_40960


namespace line_slope_intercept_sum_l409_40921

/-- Given a line passing through points (-3,1) and (1,3), prove that the sum of its slope and y-intercept is 3. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x = -3 ∧ y = 1) ∨ (x = 1 ∧ y = 3)) → 
  m + b = 3 := by
  sorry

end line_slope_intercept_sum_l409_40921


namespace terrell_weight_lifting_l409_40974

/-- The number of times Terrell lifts the 25-pound weights -/
def original_lifts : ℕ := 10

/-- The weight of each 25-pound weight in pounds -/
def original_weight : ℕ := 25

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The weight of each 20-pound weight in pounds -/
def new_weight : ℕ := 20

/-- The total weight lifted with the original weights -/
def total_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of times Terrell must lift the new weights to lift the same total weight -/
def new_lifts : ℚ := total_weight / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 12.5 := by sorry

end terrell_weight_lifting_l409_40974


namespace veranda_area_l409_40908

/-- Given a rectangular room with length 17 m and width 12 m, surrounded by a veranda of width 2 m on all sides, the area of the veranda is 132 square meters. -/
theorem veranda_area (room_length : ℝ) (room_width : ℝ) (veranda_width : ℝ) :
  room_length = 17 →
  room_width = 12 →
  veranda_width = 2 →
  let total_length := room_length + 2 * veranda_width
  let total_width := room_width + 2 * veranda_width
  let total_area := total_length * total_width
  let room_area := room_length * room_width
  total_area - room_area = 132 :=
by sorry

end veranda_area_l409_40908


namespace three_divisions_not_imply_symmetry_l409_40901

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon for this problem
  mk :: 

/-- A division of a polygon is a way to split it into two equal parts. -/
structure Division (P : Polygon) where
  -- We don't need to define the full structure of a division for this problem
  mk ::

/-- A symmetry of a polygon is either a center of symmetry or an axis of symmetry. -/
inductive Symmetry (P : Polygon)
  | Center : Symmetry P
  | Axis : Symmetry P

/-- A polygon has three divisions if there exist three distinct ways to split it into two equal parts. -/
def has_three_divisions (P : Polygon) : Prop :=
  ∃ (d1 d2 d3 : Division P), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

/-- A polygon has a symmetry if it has either a center of symmetry or an axis of symmetry. -/
def has_symmetry (P : Polygon) : Prop :=
  ∃ (s : Symmetry P), true

/-- 
The existence of three ways to divide a polygon into two equal parts 
does not necessarily imply the existence of a center or axis of symmetry for that polygon.
-/
theorem three_divisions_not_imply_symmetry :
  ∃ (P : Polygon), has_three_divisions P ∧ ¬has_symmetry P :=
sorry

end three_divisions_not_imply_symmetry_l409_40901


namespace mel_weight_proof_l409_40900

/-- Mel's weight in pounds -/
def mels_weight : ℝ := 70

/-- Brenda's weight in pounds -/
def brendas_weight : ℝ := 220

/-- Relationship between Brenda's and Mel's weights -/
def weight_relationship (m : ℝ) : Prop := brendas_weight = 3 * m + 10

theorem mel_weight_proof : 
  weight_relationship mels_weight ∧ mels_weight = 70 := by
  sorry

end mel_weight_proof_l409_40900


namespace apple_picking_solution_l409_40990

/-- Represents the apple picking problem --/
def apple_picking_problem (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : Prop :=
  let first_day := (total : ℚ) * first_day_fraction
  let second_day := 2 * first_day
  let third_day := (total : ℚ) - remaining - first_day - second_day
  (third_day - first_day) = 20

/-- Theorem stating the solution to the apple picking problem --/
theorem apple_picking_solution :
  apple_picking_problem 200 (1/5) 20 := by
  sorry


end apple_picking_solution_l409_40990


namespace polygon_congruence_l409_40989

/-- A convex polygon in the plane -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry -- Convexity condition

/-- The side length between two consecutive vertices of a polygon -/
def sideLength (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- The angle at a vertex of a polygon -/
def angle (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- Two polygons are congruent if there exists a rigid motion that maps one to the other -/
def congruent (p q : ConvexPolygon n) : Prop := sorry

/-- Main theorem: Two convex n-gons with equal corresponding sides and n-3 equal corresponding angles are congruent -/
theorem polygon_congruence (n : ℕ) (p q : ConvexPolygon n) 
  (h_sides : ∀ i : Fin n, sideLength p i = sideLength q i)
  (h_angles : ∃ (s : Finset (Fin n)), s.card = n - 3 ∧ ∀ i ∈ s, angle p i = angle q i) :
  congruent p q :=
sorry

end polygon_congruence_l409_40989


namespace min_value_expression_l409_40943

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c = Real.sqrt 6) :
  ∃ (min_val : ℝ), min_val = 8 * Real.sqrt 2 - 4 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → y + z = Real.sqrt 6 →
    (x * z^2 + 2 * x) / (y * z) + 16 / (x + 2) ≥ min_val :=
by sorry

end min_value_expression_l409_40943


namespace expression_evaluation_l409_40946

theorem expression_evaluation (x : ℝ) (h : x = 1.25) :
  (3 * x^2 - 8 * x + 2) * (4 * x - 5) = 0 := by
  sorry

end expression_evaluation_l409_40946


namespace projection_scalar_multiple_l409_40957

def proj_w (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_scalar_multiple (v w : ℝ × ℝ) :
  proj_w v w = (4, 3) → proj_w (7 • v) w = (28, 21) := by
  sorry

end projection_scalar_multiple_l409_40957


namespace final_value_after_percentage_changes_l409_40919

theorem final_value_after_percentage_changes (initial_value : ℝ) 
  (increase_percent : ℝ) (decrease_percent : ℝ) : 
  initial_value = 1500 → 
  increase_percent = 20 → 
  decrease_percent = 40 → 
  let increased_value := initial_value * (1 + increase_percent / 100)
  let final_value := increased_value * (1 - decrease_percent / 100)
  final_value = 1080 := by
  sorry

end final_value_after_percentage_changes_l409_40919


namespace garden_area_difference_l409_40915

/-- Represents a rectangular garden with a pathway around it. -/
structure Garden where
  totalLength : ℕ
  totalWidth : ℕ
  pathwayWidth : ℕ

/-- Calculates the effective gardening area of a garden. -/
def effectiveArea (g : Garden) : ℕ :=
  (g.totalLength - 2 * g.pathwayWidth) * (g.totalWidth - 2 * g.pathwayWidth)

/-- Karl's garden dimensions -/
def karlGarden : Garden :=
  { totalLength := 30
  , totalWidth := 50
  , pathwayWidth := 2 }

/-- Makenna's garden dimensions -/
def makennaGarden : Garden :=
  { totalLength := 35
  , totalWidth := 55
  , pathwayWidth := 3 }

/-- Theorem stating the difference in effective gardening area -/
theorem garden_area_difference :
  effectiveArea makennaGarden - effectiveArea karlGarden = 225 :=
by sorry

end garden_area_difference_l409_40915


namespace senior_ticket_price_l409_40966

/-- Proves that the price of senior citizen tickets is $10 -/
theorem senior_ticket_price
  (total_tickets : ℕ)
  (regular_price : ℕ)
  (total_sales : ℕ)
  (regular_tickets : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_price = 15)
  (h3 : total_sales = 855)
  (h4 : regular_tickets = 41)
  : (total_sales - regular_tickets * regular_price) / (total_tickets - regular_tickets) = 10 := by
  sorry

end senior_ticket_price_l409_40966


namespace not_arithmetic_sequence_l409_40956

theorem not_arithmetic_sequence : ¬∃ (m n k : ℤ) (a d : ℝ), 
  m < n ∧ n < k ∧ 
  1 = a + (m - 1) * d ∧ 
  Real.sqrt 3 = a + (n - 1) * d ∧ 
  2 = a + (k - 1) * d :=
sorry

end not_arithmetic_sequence_l409_40956


namespace remainder_two_power_1000_mod_17_l409_40963

theorem remainder_two_power_1000_mod_17 : 2^1000 % 17 = 1 := by
  sorry

end remainder_two_power_1000_mod_17_l409_40963


namespace min_value_abc_l409_40911

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by sorry

end min_value_abc_l409_40911


namespace first_five_multiples_average_l409_40965

theorem first_five_multiples_average (n : ℝ) : 
  (n + 2*n + 3*n + 4*n + 5*n) / 5 = 27 → n = 9 := by
  sorry

end first_five_multiples_average_l409_40965


namespace vector_addition_l409_40906

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The sum of two vectors is equal to the vector from the start of the first to the end of the second. -/
theorem vector_addition (a b : V) :
  ∃ c : V, a + b = c ∧ ∃ (x y : V), x + a = y ∧ y + b = x + c :=
sorry

end vector_addition_l409_40906


namespace inequality_proof_l409_40932

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by sorry

end inequality_proof_l409_40932


namespace aston_comic_pages_l409_40905

/-- The number of pages in each comic -/
def pages_per_comic : ℕ := 25

/-- The number of untorn comics initially in the box -/
def initial_comics : ℕ := 5

/-- The total number of comics in the box after Aston put them back together -/
def final_comics : ℕ := 11

/-- The number of pages Aston found on the floor -/
def pages_found : ℕ := (final_comics - initial_comics) * pages_per_comic

theorem aston_comic_pages : pages_found = 150 := by
  sorry

end aston_comic_pages_l409_40905


namespace lcm_16_35_l409_40950

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end lcm_16_35_l409_40950


namespace amount_r_holds_l409_40918

theorem amount_r_holds (total : ℝ) (r_fraction : ℝ) (r_amount : ℝ) : 
  total = 7000 →
  r_fraction = 2/3 →
  r_amount = r_fraction * (total / (1 + r_fraction)) →
  r_amount = 2800 := by
sorry

end amount_r_holds_l409_40918


namespace max_students_distribution_l409_40903

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 3540) (h2 : pencils = 2860) :
  Nat.gcd pens pencils = 40 :=
by sorry

end max_students_distribution_l409_40903


namespace number_equals_scientific_notation_l409_40945

-- Define the number we want to represent in scientific notation
def number : ℕ := 11700000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.17 * (10 ^ 7)

-- Theorem stating that the number is equal to its scientific notation representation
theorem number_equals_scientific_notation : (number : ℝ) = scientific_notation := by
  sorry

end number_equals_scientific_notation_l409_40945


namespace missing_digit_divisible_by_nine_l409_40964

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def has_form_173x5 (n : ℕ) : Prop :=
  ∃ x : ℕ, x < 10 ∧ n = 17300 + 10 * x + 5

theorem missing_digit_divisible_by_nine :
  ∃! n : ℕ, is_five_digit n ∧ has_form_173x5 n ∧ n % 9 = 0 :=
sorry

end missing_digit_divisible_by_nine_l409_40964


namespace balls_in_box_l409_40961

theorem balls_in_box (initial_balls : ℕ) (balls_taken : ℕ) (balls_left : ℕ) : 
  initial_balls = 10 → balls_taken = 3 → balls_left = initial_balls - balls_taken → balls_left = 7 := by
  sorry

end balls_in_box_l409_40961


namespace g_value_l409_40951

/-- Definition of g(n) as the smallest possible number of integers left on the blackboard --/
def g (n : ℕ) : ℕ := sorry

/-- Theorem stating the value of g(n) for all n ≥ 2 --/
theorem g_value (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℕ, n = 2^k ∧ g n = 2) ∨ (¬∃ k : ℕ, n = 2^k) ∧ g n = 3 := by sorry

end g_value_l409_40951


namespace circle_condition_l409_40982

/-- The equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + a = 0

/-- Definition of a circle in 2D space -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_condition (a : ℝ) :
  (∀ x y, ∃ center radius, circle_equation x y a → is_circle center radius x y) ↔ a < 2 :=
sorry

end circle_condition_l409_40982


namespace unique_function_satisfying_equation_l409_40953

theorem unique_function_satisfying_equation :
  ∃! f : ℤ → ℝ, (∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) = 1) ∧
                 (∀ x : ℤ, f x = 1) := by
  sorry

end unique_function_satisfying_equation_l409_40953


namespace gcd_of_90_and_252_l409_40968

theorem gcd_of_90_and_252 : Nat.gcd 90 252 = 18 := by
  sorry

end gcd_of_90_and_252_l409_40968


namespace min_sum_inverse_squares_l409_40986

/-- Given two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0,
    where a and b are real numbers, ab ≠ 0, and the circles have exactly three common tangent lines,
    prove that the minimum value of 1/a^2 + 1/b^2 is 1. -/
theorem min_sum_inverse_squares (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h_tangent : ∃ (t1 t2 t3 : ℝ × ℝ), 
      t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
      (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∨ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
        ((x - t1.1)^2 + (y - t1.2)^2 = 0 ∨
         (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨
         (x - t3.1)^2 + (y - t3.2)^2 = 0))) :
  (∀ c d : ℝ, c ≠ 0 → d ≠ 0 → 
    (∃ (t1' t2' t3' : ℝ × ℝ), 
      t1' ≠ t2' ∧ t1' ≠ t3' ∧ t2' ≠ t3' ∧
      (∀ (x y : ℝ), (x^2 + y^2 + 2*c*x + c^2 - 4 = 0 ∨ x^2 + y^2 - 4*d*y - 1 + 4*d^2 = 0) →
        ((x - t1'.1)^2 + (y - t1'.2)^2 = 0 ∨
         (x - t2'.1)^2 + (y - t2'.2)^2 = 0 ∨
         (x - t3'.1)^2 + (y - t3'.2)^2 = 0))) →
    1 / c^2 + 1 / d^2 ≥ 1) ∧
  (1 / a^2 + 1 / b^2 = 1) := by
sorry

end min_sum_inverse_squares_l409_40986


namespace isosceles_smallest_hypotenuse_l409_40972

-- Define a triangle with sides a, b, c and angles α, β, γ
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Ensure angles sum to π
  angle_sum : α + β + γ = π
  -- Ensure sides are positive
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  -- Law of sines
  law_of_sines : a / Real.sin α = b / Real.sin β
                ∧ b / Real.sin β = c / Real.sin γ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem: Among all triangles with fixed perimeter and angle γ,
-- the isosceles triangle (α = β) has the smallest hypotenuse
theorem isosceles_smallest_hypotenuse 
  (t1 t2 : Triangle) 
  (same_perimeter : perimeter t1 = perimeter t2)
  (same_gamma : t1.γ = t2.γ)
  (t2_isosceles : t2.α = t2.β)
  : t2.c ≤ t1.c := by
  sorry

end isosceles_smallest_hypotenuse_l409_40972


namespace ali_age_difference_l409_40962

/-- Given the ages of Ali and Umar, and the relationship between Umar and Yusaf's ages,
    prove that Ali is 3 years older than Yusaf. -/
theorem ali_age_difference (ali_age umar_age : ℕ) (h1 : ali_age = 8) (h2 : umar_age = 10)
  (h3 : umar_age = 2 * (umar_age / 2)) : ali_age - (umar_age / 2) = 3 := by
  sorry

end ali_age_difference_l409_40962


namespace supplementary_angles_difference_l409_40948

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 7 / 2 →  -- The ratio of the measures is 7:2
  max a b - min a b = 100 :=  -- The positive difference is 100°
by sorry

end supplementary_angles_difference_l409_40948


namespace inequality_proof_l409_40933

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  3 * Real.rpow (1 / (a * b * c) + 6 * (a + b + c)) (1/3) ≤ Real.rpow 3 (1/3) / (a * b * c) := by
  sorry

end inequality_proof_l409_40933


namespace mary_flour_calculation_l409_40983

/-- The amount of flour Mary has already put in the cake -/
def flour_put_in : ℕ := sorry

/-- The total amount of flour required by the recipe -/
def total_flour_required : ℕ := 12

/-- The amount of flour still needed -/
def flour_still_needed : ℕ := 2

theorem mary_flour_calculation :
  flour_put_in = total_flour_required - flour_still_needed :=
sorry

end mary_flour_calculation_l409_40983


namespace relationship_between_A_B_C_l409_40916

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the relationships between A, B, and C
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary_and_sufficient (P Q : Prop) : Prop :=
  (P ↔ Q)

-- Theorem statement
theorem relationship_between_A_B_C
  (h1 : sufficient_not_necessary A B)
  (h2 : necessary_and_sufficient B C) :
  sufficient_not_necessary C A :=
sorry

end relationship_between_A_B_C_l409_40916


namespace arithmetic_sequence_length_l409_40939

/-- An arithmetic sequence with given start, end, and common difference -/
def arithmetic_sequence (start end_ diff : ℕ) : List ℕ :=
  let n := (end_ - start) / diff + 1
  List.range n |>.map (fun i => start + i * diff)

/-- The problem statement -/
theorem arithmetic_sequence_length :
  (arithmetic_sequence 20 150 5).length = 27 := by
  sorry

end arithmetic_sequence_length_l409_40939


namespace exists_good_submatrix_l409_40929

/-- Definition of a binary matrix -/
def BinaryMatrix (n : ℕ) := Matrix (Fin n) (Fin n) Bool

/-- Definition of a good matrix -/
def IsGoodMatrix {n : ℕ} (A : BinaryMatrix n) : Prop :=
  ∃ (x y : Bool), ∀ (i j : Fin n),
    (i < j → A i j = x) ∧
    (j < i → A i j = y)

/-- Main theorem -/
theorem exists_good_submatrix :
  ∃ (M : ℕ), ∀ (n : ℕ) (A : BinaryMatrix n),
    n > M →
    ∃ (m : ℕ) (indices : Fin m → Fin n),
      Function.Injective indices ∧
      IsGoodMatrix (Matrix.submatrix A indices indices) :=
by sorry

end exists_good_submatrix_l409_40929


namespace ellipse_eccentricity_l409_40958

/-- Ellipse C with foci F₁ and F₂, and point P on C -/
structure Ellipse :=
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (F₁ F₂ P : ℝ × ℝ)
  (h_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1)
  (h_perp : (P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
  (h_angle : Real.cos (30 * π / 180) = 
    ((P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2)) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)))

/-- The eccentricity of an ellipse is √3/3 -/
theorem ellipse_eccentricity (C : Ellipse) : 
  Real.sqrt ((C.F₂.1 - C.F₁.1)^2 + (C.F₂.2 - C.F₁.2)^2) / (2 * C.a) = Real.sqrt 3 / 3 := by
  sorry

end ellipse_eccentricity_l409_40958


namespace circumcircle_equation_l409_40907

-- Define the points and line
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of the triangle
def area_ABC : ℝ := 10

-- Define the possible equations of the circumcircle
def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 1/2 * x - 5 * y - 3/2 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 25/6 * x - 89/9 * y + 347/18 = 0

-- Theorem statement
theorem circumcircle_equation :
  ∃ (C : ℝ × ℝ), line_C C.1 C.2 ∧
  (∀ (x y : ℝ), circle_eq1 x y ∨ circle_eq2 x y) :=
sorry

end circumcircle_equation_l409_40907


namespace exponential_inequality_l409_40924

theorem exponential_inequality (a x y : ℝ) :
  (a > 1 ∧ x > y → a^x > a^y) ∧ (a < 1 ∧ x > y → a^x < a^y) := by
sorry

end exponential_inequality_l409_40924


namespace inequality_and_equality_condition_l409_40902

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≥ 0) :
  x^(3/2) + 6*x^(5/4) + 8*x^(3/4) ≥ 15*x ∧
  (x^(3/2) + 6*x^(5/4) + 8*x^(3/4) = 15*x ↔ x = 0 ∨ x = 1) := by
  sorry

end inequality_and_equality_condition_l409_40902


namespace macks_travel_problem_l409_40912

/-- Mack's travel problem -/
theorem macks_travel_problem (speed_to_office : ℝ) (total_time : ℝ) (time_to_office : ℝ) 
  (h1 : speed_to_office = 58)
  (h2 : total_time = 3)
  (h3 : time_to_office = 1.4) :
  (speed_to_office * time_to_office) / (total_time - time_to_office) = 50.75 := by
  sorry

end macks_travel_problem_l409_40912


namespace average_marks_of_failed_candidates_l409_40988

theorem average_marks_of_failed_candidates
  (total_candidates : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_candidates : ℕ)
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : passed_average = 39)
  (h4 : passed_candidates = 100) :
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := total_candidates * overall_average
  let passed_marks := passed_candidates * passed_average
  let failed_marks := total_marks - passed_marks
  failed_marks / failed_candidates = 15 :=
by sorry

end average_marks_of_failed_candidates_l409_40988


namespace geometry_biology_overlap_difference_l409_40937

theorem geometry_biology_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119) :
  let max_overlap := min geometry biology
  let min_overlap := geometry + biology - total
  max_overlap - min_overlap = 88 := by
sorry

end geometry_biology_overlap_difference_l409_40937


namespace man_speed_in_still_water_l409_40913

def downstream_distance : ℝ := 24
def upstream_distance : ℝ := 18
def time : ℝ := 3
def current_speed : ℝ := 2

def man_speed : ℝ := 6

theorem man_speed_in_still_water :
  (downstream_distance / time = man_speed + current_speed) ∧
  (upstream_distance / time = man_speed - current_speed) :=
by sorry

end man_speed_in_still_water_l409_40913


namespace sandy_initial_books_l409_40934

/-- The number of books Sandy had initially -/
def sandy_books : ℕ := 10

/-- The number of books Tim has -/
def tim_books : ℕ := 33

/-- The number of books Benny lost -/
def benny_lost : ℕ := 24

/-- The number of books Sandy and Tim have together after Benny lost some -/
def remaining_books : ℕ := 19

/-- Theorem stating that Sandy had 10 books initially -/
theorem sandy_initial_books : 
  sandy_books + tim_books = remaining_books + benny_lost := by sorry

end sandy_initial_books_l409_40934


namespace factorial_simplification_l409_40995

theorem factorial_simplification : (12 : ℕ).factorial / ((10 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 1320 / 13 := by
  sorry

end factorial_simplification_l409_40995


namespace stratified_sampling_male_count_l409_40971

theorem stratified_sampling_male_count 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 32) 
  (h2 : total_female = 24) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 := by
  sorry

#check stratified_sampling_male_count

end stratified_sampling_male_count_l409_40971


namespace number_greater_than_half_l409_40967

theorem number_greater_than_half : ∃ x : ℝ, x = 1/2 + 0.3 ∧ x = 0.8 := by
  sorry

end number_greater_than_half_l409_40967


namespace wrong_observation_value_l409_40927

theorem wrong_observation_value 
  (n : ℕ) 
  (initial_mean correct_value new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 32)
  (h3 : correct_value = 48)
  (h4 : new_mean = 32.5) :
  ∃ wrong_value : ℝ,
    (n : ℝ) * new_mean = (n : ℝ) * initial_mean - wrong_value + correct_value ∧
    wrong_value = 23 := by
sorry

end wrong_observation_value_l409_40927


namespace inequality_solution_set_l409_40909

theorem inequality_solution_set (x : ℝ) : 
  (abs (x - 1) + abs (x - 2) < 2) ↔ (1/2 < x ∧ x < 5/2) := by
  sorry

end inequality_solution_set_l409_40909


namespace vectors_parallel_opposite_direction_l409_40955

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -4)

theorem vectors_parallel_opposite_direction :
  ∃ k : ℝ, k < 0 ∧ b = (k • a.1, k • a.2) :=
sorry

end vectors_parallel_opposite_direction_l409_40955


namespace max_min_difference_z_l409_40944

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (sum_squares_condition : x^2 + y^2 + z^2 = 29) :
  ∃ (z_max z_min : ℝ),
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 29) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 29) → z' ≥ z_min) ∧
    z_max - z_min = 20/3 :=
by sorry

end max_min_difference_z_l409_40944


namespace solution_set_when_a_eq_2_unique_a_for_integer_solution_set_l409_40942

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Theorem for part 1
theorem solution_set_when_a_eq_2 :
  ∀ x : ℝ, f x 2 ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 := by sorry

-- Theorem for part 2
theorem unique_a_for_integer_solution_set :
  (∃! a : ℝ, ∀ x : ℤ, f (x : ℝ) a < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (∀ a : ℝ, (∀ x : ℤ, f (x : ℝ) a < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → a = 2) := by sorry

end solution_set_when_a_eq_2_unique_a_for_integer_solution_set_l409_40942


namespace ellipse_focal_length_l409_40926

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 = 2

/-- The focal length of an ellipse -/
def focal_length : ℝ := 4

/-- Theorem: The focal length of the ellipse defined by x^2/2 + y^2/4 = 2 is equal to 4 -/
theorem ellipse_focal_length :
  ∀ x y : ℝ, ellipse_equation x y → focal_length = 4 := by
  sorry

end ellipse_focal_length_l409_40926


namespace kennel_dogs_l409_40949

/-- Given a kennel with cats and dogs, prove the number of dogs. -/
theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 2 / 3 →  -- ratio of cats to dogs is 2:3
  cats = dogs - 6 →            -- 6 fewer cats than dogs
  dogs = 18 := by              -- prove that there are 18 dogs
sorry

end kennel_dogs_l409_40949


namespace fruit_bowl_oranges_l409_40969

theorem fruit_bowl_oranges :
  let bananas : ℕ := 4
  let apples : ℕ := 3 * bananas
  let pears : ℕ := 5
  let total_fruits : ℕ := 30
  let oranges : ℕ := total_fruits - (bananas + apples + pears)
  oranges = 9 :=
by sorry

end fruit_bowl_oranges_l409_40969


namespace solve_video_game_problem_l409_40970

def video_game_problem (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : Prop :=
  let remaining_players := total_lives / lives_per_player
  let players_quit := initial_players - remaining_players
  players_quit = 5

theorem solve_video_game_problem :
  video_game_problem 11 5 30 :=
by
  sorry

end solve_video_game_problem_l409_40970


namespace m_is_even_l409_40952

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem m_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  Even M := by sorry

end m_is_even_l409_40952


namespace line_ellipse_intersection_range_l409_40998

/-- The range of b values for which the line y = kx + b always has two common points with the ellipse x²/9 + y²/4 = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), 
  (∀ (b : ℝ), (∃! (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + b ∧ 
    y₂ = k * x₂ + b ∧ 
    x₁^2 / 9 + y₁^2 / 4 = 1 ∧ 
    x₂^2 / 9 + y₂^2 / 4 = 1)) ↔ 
  (-2 < b ∧ b < 2) :=
sorry

end line_ellipse_intersection_range_l409_40998


namespace comparison_of_b_and_c_l409_40996

theorem comparison_of_b_and_c (a b c : ℝ) 
  (h1 : 2*a^3 - b^3 + 2*c^3 - 6*a^2*b + 3*a*b^2 - 3*a*c^2 - 3*b*c^2 + 6*a*b*c = 0)
  (h2 : a < b) : 
  b < c ∧ c < 2*b - a := by
  sorry

end comparison_of_b_and_c_l409_40996


namespace function_periodicity_l409_40979

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the functional equation
def functionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- Define the existence of c
def existsC (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ f (c / 2) = 0

-- Define periodicity
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

-- Theorem statement
theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : functionalEquation f) 
  (h2 : existsC f) :
  ∃ T : ℝ, T > 0 ∧ isPeriodic f T :=
sorry

end function_periodicity_l409_40979


namespace best_of_three_win_probability_l409_40923

/-- The probability of winning a single set -/
def p : ℝ := 0.6

/-- The probability of winning a best-of-three match given the probability of winning each set -/
def win_probability (p : ℝ) : ℝ := p^2 + 3 * p^2 * (1 - p)

/-- Theorem: The probability of winning a best-of-three match when p = 0.6 is 0.648 -/
theorem best_of_three_win_probability :
  win_probability p = 0.648 := by sorry

end best_of_three_win_probability_l409_40923


namespace remainder_proof_l409_40959

theorem remainder_proof : ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := by
  sorry

end remainder_proof_l409_40959


namespace players_who_quit_video_game_problem_l409_40920

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem video_game_problem :
  players_who_quit 13 6 30 = 8 := by
  sorry

end players_who_quit_video_game_problem_l409_40920


namespace porridge_eaters_today_l409_40947

/-- Represents the number of children eating porridge daily -/
def daily_eaters : ℕ := 5

/-- Represents the number of children eating porridge every other day -/
def alternate_eaters : ℕ := 7

/-- Represents the number of children who ate porridge yesterday -/
def yesterday_eaters : ℕ := 9

/-- Calculates the number of children eating porridge today -/
def today_eaters : ℕ := daily_eaters + (alternate_eaters - (yesterday_eaters - daily_eaters))

/-- Theorem stating that the number of children eating porridge today is 8 -/
theorem porridge_eaters_today : today_eaters = 8 := by
  sorry

end porridge_eaters_today_l409_40947


namespace min_value_a_min_value_a_achievable_l409_40925

theorem min_value_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) → 
  a ≥ 4 :=
by sorry

theorem min_value_a_achievable : 
  ∃ a : ℝ, a = 4 ∧ (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) :=
by sorry

end min_value_a_min_value_a_achievable_l409_40925


namespace treadmill_theorem_l409_40938

def treadmill_problem (days : Nat) (distance_per_day : Real) 
  (speeds : List Real) (constant_speed : Real) : Prop :=
  days = 4 ∧
  distance_per_day = 3 ∧
  speeds = [6, 4, 3, 5] ∧
  constant_speed = 5 ∧
  let actual_time := (List.map (fun s => distance_per_day / s) speeds).sum
  let constant_time := (days * distance_per_day) / constant_speed
  (actual_time - constant_time) * 60 = 27

theorem treadmill_theorem : 
  ∃ (days : Nat) (distance_per_day : Real) (speeds : List Real) (constant_speed : Real),
  treadmill_problem days distance_per_day speeds constant_speed :=
sorry

end treadmill_theorem_l409_40938


namespace oldest_person_is_A_l409_40980

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person
  | D : Person

-- Define the age relation
def olderThan : Person → Person → Prop := sorry

-- Define the statements made by each person
def statementA : Prop := olderThan Person.B Person.D
def statementB : Prop := olderThan Person.C Person.A
def statementC : Prop := olderThan Person.D Person.C
def statementD : Prop := olderThan Person.B Person.C

-- Define a function to check if a statement is true
def isTrueStatement (p : Person) : Prop :=
  match p with
  | Person.A => statementA
  | Person.B => statementB
  | Person.C => statementC
  | Person.D => statementD

-- Theorem to prove
theorem oldest_person_is_A :
  (∀ (p q : Person), p ≠ q → olderThan p q ∨ olderThan q p) →
  (∀ (p q r : Person), olderThan p q → olderThan q r → olderThan p r) →
  (∃! (p : Person), isTrueStatement p) →
  (∀ (p : Person), isTrueStatement p → ∀ (q : Person), q ≠ p → olderThan p q) →
  (∀ (p : Person), olderThan Person.A p ∨ p = Person.A) :=
sorry

end oldest_person_is_A_l409_40980


namespace probability_non_defective_pencils_l409_40936

/-- The probability of selecting 3 non-defective pencils from a box of 10 pencils with 2 defective pencils -/
theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 10
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 7 / 15 := by
  sorry

end probability_non_defective_pencils_l409_40936


namespace gcd_problem_l409_40991

theorem gcd_problem (b : ℤ) (h : 2373 ∣ b) : 
  Nat.gcd (Int.natAbs (b^2 + 13*b + 40)) (Int.natAbs (b + 5)) = 5 := by
  sorry

end gcd_problem_l409_40991


namespace cone_surface_area_l409_40931

/-- The surface area of a cone given its slant height and angle between slant height and axis -/
theorem cone_surface_area (slant_height : ℝ) (angle : ℝ) : 
  slant_height = 20 →
  angle = 30 * π / 180 →
  ∃ (surface_area : ℝ), surface_area = 300 * π := by
sorry

end cone_surface_area_l409_40931


namespace plane_equation_l409_40917

/-- A plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  coprime : Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

def parallel (p1 p2 : Plane) : Prop :=
  p1.a * p2.b = p1.b * p2.a ∧ p1.a * p2.c = p1.c * p2.a ∧ p1.b * p2.c = p1.c * p2.b

def passes_through (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem plane_equation : 
  ∃ (plane : Plane), 
    plane.a = 3 ∧ 
    plane.b = -4 ∧ 
    plane.c = 1 ∧ 
    plane.d = 7 ∧ 
    passes_through plane ⟨2, 3, -1⟩ ∧ 
    parallel plane ⟨3, -4, 1, -5, by sorry, by sorry⟩ := by
  sorry

end plane_equation_l409_40917


namespace percentage_pies_with_forks_l409_40981

def total_pies : ℕ := 2000
def pies_not_with_forks : ℕ := 640

theorem percentage_pies_with_forks :
  (total_pies - pies_not_with_forks : ℚ) / total_pies * 100 = 68 := by
  sorry

end percentage_pies_with_forks_l409_40981


namespace article_sale_profit_loss_l409_40935

theorem article_sale_profit_loss (cost_price selling_price_profit selling_price_25_percent : ℕ)
  (h1 : cost_price = 1400)
  (h2 : selling_price_profit = 1520)
  (h3 : selling_price_25_percent = 1750)
  (h4 : selling_price_25_percent = cost_price + cost_price / 4) :
  ∃ (selling_price_loss : ℕ),
    selling_price_loss = 1280 ∧
    (selling_price_profit - cost_price) / cost_price =
    (cost_price - selling_price_loss) / cost_price :=
by sorry

end article_sale_profit_loss_l409_40935


namespace simplify_fraction_l409_40994

theorem simplify_fraction : (130 : ℚ) / 16900 * 65 = 1 / 2 := by
  sorry

end simplify_fraction_l409_40994


namespace solve_linear_system_l409_40999

theorem solve_linear_system (b : ℚ) : 
  (∃ x y : ℚ, x + b * y = 0 ∧ x + y = -1 ∧ x = 1) →
  b = 1/2 := by
sorry

end solve_linear_system_l409_40999


namespace brother_birth_year_l409_40985

/-- Given Karina's birth year, current age, and the fact that she is twice as old as her brother,
    prove her brother's birth year. -/
theorem brother_birth_year
  (karina_birth_year : ℕ)
  (karina_current_age : ℕ)
  (h_karina_birth : karina_birth_year = 1970)
  (h_karina_age : karina_current_age = 40)
  (h_twice_age : karina_current_age = 2 * (karina_current_age / 2)) :
  karina_birth_year + karina_current_age - (karina_current_age / 2) = 1990 := by
  sorry

end brother_birth_year_l409_40985


namespace age_problem_l409_40975

theorem age_problem (billy joe sarah : ℕ) 
  (h1 : billy = 3 * joe)
  (h2 : billy + joe = 48)
  (h3 : joe + sarah = 30) :
  billy = 36 ∧ joe = 12 ∧ sarah = 18 := by
sorry

end age_problem_l409_40975


namespace sum_reciprocal_n_n_plus_three_l409_40910

/-- The sum of the infinite series ∑(n=1 to ∞) 1/(n(n+3)) is equal to 11/18. -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' (n : ℕ), 1 / (n * (n + 3)) = 11 / 18 := by sorry

end sum_reciprocal_n_n_plus_three_l409_40910


namespace profit_calculation_l409_40922

def trees : ℕ := 30
def planks_per_tree : ℕ := 25
def planks_per_table : ℕ := 15
def selling_price : ℕ := 300
def labor_cost : ℕ := 3000

theorem profit_calculation :
  let total_planks := trees * planks_per_tree
  let tables_made := total_planks / planks_per_table
  let revenue := tables_made * selling_price
  revenue - labor_cost = 12000 := by
sorry

end profit_calculation_l409_40922


namespace intersection_of_A_and_B_l409_40954

def set_A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def set_B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l409_40954


namespace parallel_vectors_t_value_l409_40978

theorem parallel_vectors_t_value (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, 3) →
  b = (3, t) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  t = 9 := by
sorry

end parallel_vectors_t_value_l409_40978


namespace souvenir_purchasing_plans_l409_40973

def number_of_purchasing_plans (total_items : ℕ) (types : ℕ) (items_per_type : ℕ) : ℕ :=
  let f (x : ℕ → ℕ) := x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10
  let coefficient_of_x25 := (Nat.choose 24 3) - 4 * (Nat.choose 14 3) + 6 * (Nat.choose 4 3)
  coefficient_of_x25

theorem souvenir_purchasing_plans :
  number_of_purchasing_plans 25 4 10 = 592 :=
sorry

end souvenir_purchasing_plans_l409_40973


namespace triangle_ratio_bound_l409_40993

theorem triangle_ratio_bound (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≤ 1 :=
sorry

end triangle_ratio_bound_l409_40993


namespace sin_B_range_in_acute_triangle_l409_40940

theorem sin_B_range_in_acute_triangle (A B C : Real) (a b c : Real) (S : Real) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  S = (1 / 2) * b * c * Real.sin A →
  a^2 = 2 * S + (b - c)^2 →
  3 / 5 < Real.sin B ∧ Real.sin B < 1 := by
  sorry

end sin_B_range_in_acute_triangle_l409_40940


namespace largest_lcm_with_15_l409_40987

theorem largest_lcm_with_15 : 
  (Finset.image (fun n => Nat.lcm 15 n) {3, 5, 6, 9, 10, 15}).max = some 30 := by
  sorry

end largest_lcm_with_15_l409_40987


namespace horizontal_distance_is_three_l409_40976

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Define the points P and Q
def P : { x : ℝ // f x = 10 } := sorry
def Q : { x : ℝ // |f x| = 2 } := sorry

-- Theorem statement
theorem horizontal_distance_is_three :
  ∃ (xp xq : ℝ), 
    f xp = 10 ∧ 
    |f xq| = 2 ∧ 
    |xp - xq| = 3 :=
sorry

end horizontal_distance_is_three_l409_40976


namespace slope_of_line_l409_40941

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) = (-4/7) * (x - 0) :=
by sorry

end slope_of_line_l409_40941


namespace inequality_fraction_comparison_l409_40977

theorem inequality_fraction_comparison (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  a / b > b / a := by
  sorry

end inequality_fraction_comparison_l409_40977


namespace last_name_length_proof_l409_40997

/-- Given information about the lengths of last names, prove the length of another person's last name --/
theorem last_name_length_proof (samantha_length bobbie_length other_length : ℕ) : 
  samantha_length = 7 →
  bobbie_length = samantha_length + 3 →
  bobbie_length - 2 = 2 * other_length →
  other_length = 4 := by
  sorry

end last_name_length_proof_l409_40997


namespace no_solution_exists_l409_40914

theorem no_solution_exists : ¬∃ x : ℝ, 1000^2 + 1001^2 + 1002^2 + x^2 + 1004^2 = 6 := by
  sorry

end no_solution_exists_l409_40914


namespace distance_point_to_line_l409_40984

/-- The distance from the point (√2, -√2) to the line x + y = 1 is √2/2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)
  let line (x y : ℝ) : Prop := x + y = 1
  abs (point.1 + point.2 - 1) / Real.sqrt 2 = Real.sqrt 2 / 2 := by sorry

end distance_point_to_line_l409_40984


namespace geometric_subsequence_k4_l409_40992

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- A subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) where
  k : ℕ → ℕ
  q : ℝ
  h_geom : ∀ n : ℕ, as.a (k (n + 1)) = q * as.a (k n)
  h_k1 : k 1 ≠ 1
  h_k2 : k 2 ≠ 2
  h_k3 : k 3 ≠ 6

/-- The main theorem -/
theorem geometric_subsequence_k4 (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  gs.k 4 = 22 := by
  sorry

end geometric_subsequence_k4_l409_40992


namespace sin_135_degrees_l409_40928

theorem sin_135_degrees : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry

end sin_135_degrees_l409_40928


namespace reflection_symmetry_l409_40930

/-- Represents an L-like shape with two segments --/
structure LShape :=
  (top_segment : ℝ)
  (bottom_segment : ℝ)

/-- Reflects an L-shape over a horizontal line --/
def reflect (shape : LShape) : LShape :=
  { top_segment := shape.bottom_segment,
    bottom_segment := shape.top_segment }

/-- Checks if two L-shapes are equal --/
def is_equal (shape1 shape2 : LShape) : Prop :=
  shape1.top_segment = shape2.top_segment ∧ shape1.bottom_segment = shape2.bottom_segment

theorem reflection_symmetry (original : LShape) :
  original.top_segment > original.bottom_segment →
  is_equal (reflect original) { top_segment := original.bottom_segment, bottom_segment := original.top_segment } :=
by
  sorry

#check reflection_symmetry

end reflection_symmetry_l409_40930


namespace total_score_is_54_l409_40904

/-- The number of players on the basketball team -/
def num_players : ℕ := 8

/-- The points scored by each player -/
def player_scores : Fin num_players → ℕ
  | ⟨0, _⟩ => 7
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 2
  | ⟨3, _⟩ => 11
  | ⟨4, _⟩ => 6
  | ⟨5, _⟩ => 12
  | ⟨6, _⟩ => 1
  | ⟨7, _⟩ => 7

/-- The theorem stating that the sum of all player scores is 54 -/
theorem total_score_is_54 : (Finset.univ.sum player_scores) = 54 := by
  sorry

end total_score_is_54_l409_40904
