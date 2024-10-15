import Mathlib

namespace NUMINAMATH_CALUDE_bowling_ball_volume_l112_11298

/-- The volume of a sphere with cylindrical holes drilled into it -/
theorem bowling_ball_volume (d : ℝ) (r1 r2 r3 h1 h2 h3 : ℝ) : 
  d = 36 → r1 = 1 → r2 = 1 → r3 = 2 → h1 = 9 → h2 = 10 → h3 = 9 → 
  (4 / 3 * π * (d / 2)^3) - (π * r1^2 * h1) - (π * r2^2 * h2) - (π * r3^2 * h3) = 7721 * π := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l112_11298


namespace NUMINAMATH_CALUDE_area_circle_circumscribed_equilateral_triangle_l112_11216

/-- The area of a circle circumscribed about an equilateral triangle with side length 15 units is 75π square units. -/
theorem area_circle_circumscribed_equilateral_triangle :
  let s : ℝ := 15  -- Side length of the equilateral triangle
  let r : ℝ := s * Real.sqrt 3 / 3  -- Radius of the circumscribed circle
  let area : ℝ := π * r^2  -- Area of the circle
  area = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_area_circle_circumscribed_equilateral_triangle_l112_11216


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l112_11271

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 →  -- angles are complementary
  x = 4 * y →   -- ratio of angles is 4:1
  |x - y| = 54  -- absolute difference between angles is 54°
  := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l112_11271


namespace NUMINAMATH_CALUDE_james_ali_difference_l112_11291

def total_amount : ℕ := 250
def james_amount : ℕ := 145

theorem james_ali_difference :
  ∀ (ali_amount : ℕ),
  ali_amount + james_amount = total_amount →
  james_amount > ali_amount →
  james_amount - ali_amount = 40 :=
by sorry

end NUMINAMATH_CALUDE_james_ali_difference_l112_11291


namespace NUMINAMATH_CALUDE_brookes_science_problems_l112_11297

/-- Represents the number of problems and time for each subject in Brooke's homework --/
structure Homework where
  math_problems : ℕ
  social_studies_problems : ℕ
  science_problems : ℕ
  math_time_per_problem : ℚ
  social_studies_time_per_problem : ℚ
  science_time_per_problem : ℚ
  total_time : ℚ

/-- Calculates the total time spent on homework --/
def total_homework_time (hw : Homework) : ℚ :=
  hw.math_problems * hw.math_time_per_problem +
  hw.social_studies_problems * hw.social_studies_time_per_problem +
  hw.science_problems * hw.science_time_per_problem

/-- Theorem stating that Brooke has 10 science problems --/
theorem brookes_science_problems (hw : Homework)
  (h1 : hw.math_problems = 15)
  (h2 : hw.social_studies_problems = 6)
  (h3 : hw.math_time_per_problem = 2)
  (h4 : hw.social_studies_time_per_problem = 1/2)
  (h5 : hw.science_time_per_problem = 3/2)
  (h6 : hw.total_time = 48)
  (h7 : total_homework_time hw = hw.total_time) :
  hw.science_problems = 10 := by
  sorry


end NUMINAMATH_CALUDE_brookes_science_problems_l112_11297


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l112_11208

theorem polynomial_factor_coefficients :
  ∀ (a b : ℤ),
  (∃ (c d : ℤ), ∀ (x : ℚ),
    a * x^4 + b * x^3 + 32 * x^2 - 16 * x + 6 = (3 * x^2 - 2 * x + 1) * (c * x^2 + d * x + 6)) →
  a = 18 ∧ b = -24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l112_11208


namespace NUMINAMATH_CALUDE_solve_x_and_y_l112_11218

-- Define the universal set I
def I (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}

-- Define set A
def A : Set ℝ := {5}

-- Define the complement of A with respect to I
def complement_A (x : ℝ) (y : ℝ) : Set ℝ := {2, y}

-- Theorem statement
theorem solve_x_and_y (x : ℝ) (y : ℝ) :
  (5 ∈ I x) ∧ (complement_A x y = (I x) \ A) →
  ((x = -4 ∨ x = 2) ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_x_and_y_l112_11218


namespace NUMINAMATH_CALUDE_locker_count_l112_11212

/-- The cost of a single digit in dollars -/
def digit_cost : ℚ := 0.03

/-- The total cost of labeling all lockers in dollars -/
def total_cost : ℚ := 206.91

/-- Calculates the cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let one_digit := min n 9
  let two_digit := min (n - 9) 90
  let three_digit := min (n - 99) 900
  let four_digit := max (n - 999) 0
  digit_cost * (one_digit + 2 * two_digit + 3 * three_digit + 4 * four_digit)

/-- The theorem stating that 2001 lockers can be labeled with the given total cost -/
theorem locker_count : labeling_cost 2001 = total_cost := by
  sorry

end NUMINAMATH_CALUDE_locker_count_l112_11212


namespace NUMINAMATH_CALUDE_prime_implication_l112_11280

theorem prime_implication (p : ℕ) (hp : Prime p) (h8p2_1 : Prime (8 * p^2 + 1)) :
  Prime (8 * p^2 - p + 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_implication_l112_11280


namespace NUMINAMATH_CALUDE_initial_overs_played_l112_11247

/-- Proves the number of overs played initially in a cricket game -/
theorem initial_overs_played (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 8.333333333333334 →
  remaining_overs = 30 →
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * required_rate = target ∧
    initial_overs = 10 :=
by sorry

end NUMINAMATH_CALUDE_initial_overs_played_l112_11247


namespace NUMINAMATH_CALUDE_hypotenuse_length_l112_11238

/-- A right triangle with specific medians -/
structure RightTriangleWithMedians where
  /-- First leg of the triangle -/
  a : ℝ
  /-- Second leg of the triangle -/
  b : ℝ
  /-- First median (from vertex of acute angle) -/
  m₁ : ℝ
  /-- Second median (from vertex of acute angle) -/
  m₂ : ℝ
  /-- The first median is 6 -/
  h₁ : m₁ = 6
  /-- The second median is 3√13 -/
  h₂ : m₂ = 3 * Real.sqrt 13
  /-- Relationship between first leg and first median -/
  h₃ : m₁^2 = a^2 + (3*b/2)^2
  /-- Relationship between second leg and second median -/
  h₄ : m₂^2 = b^2 + (3*a/2)^2

/-- The theorem stating that the hypotenuse of the triangle is 3√23 -/
theorem hypotenuse_length (t : RightTriangleWithMedians) : 
  Real.sqrt (9 * (t.a^2 + t.b^2)) = 3 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l112_11238


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l112_11230

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Translates a point vertically -/
def translateY (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem -/
theorem circle_reflection_translation :
  let Q : Point := { x := 3, y := -4 }
  let Q' := translateY (reflectX Q) 5
  Q'.x = 3 ∧ Q'.y = 9 := by sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l112_11230


namespace NUMINAMATH_CALUDE_inverse_f_at_neg_1_l112_11256

-- Define f as a function with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition that f(2) = -1
axiom f_at_2 : f 2 = -1

-- State the theorem to be proved
theorem inverse_f_at_neg_1 : Function.invFun f (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_neg_1_l112_11256


namespace NUMINAMATH_CALUDE_simplify_polynomial_l112_11282

theorem simplify_polynomial (x : ℝ) : (3 * x^2 + 9 * x - 5) - (2 * x^2 + 3 * x - 10) = x^2 + 6 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l112_11282


namespace NUMINAMATH_CALUDE_ellipse_equation_l112_11203

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (c a b : ℝ) (h1 : c = 3) (h2 : a = 5) (h3 : b = 4) 
  (h4 : c / a = 3 / 5) (h5 : a^2 = b^2 + c^2) :
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 16 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l112_11203


namespace NUMINAMATH_CALUDE_percentage_sum_proof_l112_11281

theorem percentage_sum_proof : (0.08 * 24) + (0.10 * 40) = 5.92 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_proof_l112_11281


namespace NUMINAMATH_CALUDE_problem_solution_l112_11200

theorem problem_solution (x y : ℝ) : 
  y - Real.sqrt (x - 2022) = Real.sqrt (2022 - x) - 2023 →
  (x + y) ^ 2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l112_11200


namespace NUMINAMATH_CALUDE_sibling_ages_l112_11294

/-- A family with 4 siblings: Richard, David, Scott, and Jane. -/
structure Family :=
  (Richard David Scott Jane : ℕ)

/-- The conditions and question of the problem -/
theorem sibling_ages (f : Family) : 
  f.Richard = f.David + 6 →
  f.David = f.Scott + 8 →
  f.Jane = f.Richard - 5 →
  f.Richard + 8 = 2 * (f.Scott + 8) →
  f.Jane + 10 = (f.David + 10) / 2 + 4 →
  f.Scott + 12 + f.Jane + 12 = 60 →
  f.Richard - 3 + f.David - 3 + f.Scott - 3 + f.Jane - 3 = 43 := by
  sorry

#check sibling_ages

end NUMINAMATH_CALUDE_sibling_ages_l112_11294


namespace NUMINAMATH_CALUDE_eight_holes_when_unfolded_l112_11278

/-- Represents a fold on the triangular paper -/
structure Fold where
  vertex : ℕ
  midpoint : ℕ

/-- Represents the triangular paper with its folds and holes -/
structure TriangularPaper where
  folds : List Fold
  holes : ℕ

/-- Performs a fold on the triangular paper -/
def applyFold (paper : TriangularPaper) (fold : Fold) : TriangularPaper :=
  { folds := fold :: paper.folds, holes := paper.holes }

/-- Punches holes in the folded paper -/
def punchHoles (paper : TriangularPaper) (n : ℕ) : TriangularPaper :=
  { folds := paper.folds, holes := paper.holes + n }

/-- Unfolds the paper and calculates the total number of holes -/
def unfold (paper : TriangularPaper) : ℕ :=
  match paper.folds.length with
  | 0 => paper.holes
  | 1 => 2 * paper.holes
  | _ => 4 * paper.holes

/-- Theorem stating that folding an equilateral triangle twice and punching two holes results in eight holes when unfolded -/
theorem eight_holes_when_unfolded (initialPaper : TriangularPaper) : 
  let firstFold := Fold.mk 1 2
  let secondFold := Fold.mk 3 4
  let foldedPaper := applyFold (applyFold initialPaper firstFold) secondFold
  let punchedPaper := punchHoles foldedPaper 2
  unfold punchedPaper = 8 := by
  sorry


end NUMINAMATH_CALUDE_eight_holes_when_unfolded_l112_11278


namespace NUMINAMATH_CALUDE_email_difference_l112_11244

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- The difference between the number of emails Jack received in the morning and evening -/
theorem email_difference : morning_emails - evening_emails = 2 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l112_11244


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l112_11214

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 5}

-- Define set A
def A : Set ℝ := {x | -3 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- Define the result set
def result : Set ℝ := {x | -5 ≤ x ∧ x ≤ -3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = result := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l112_11214


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l112_11286

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 - 2*m + 1 = 0) → (n^2 - 2*n + 1 = 0) → 
  (m + n) / (m^2 - 2*m) = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l112_11286


namespace NUMINAMATH_CALUDE_percentage_problem_l112_11210

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 4000) = 90) → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l112_11210


namespace NUMINAMATH_CALUDE_interval_of_decrease_l112_11222

/-- Given a function f with derivative f'(x) = 2x - 4, 
    prove that the interval of decrease for f(x-1) is (-∞, 3) -/
theorem interval_of_decrease (f : ℝ → ℝ) (h : ∀ x, deriv f x = 2 * x - 4) :
  ∀ x, x < 3 ↔ deriv (fun y ↦ f (y - 1)) x < 0 := by
  sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l112_11222


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l112_11211

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 0, p x) ↔ ∀ x > 0, ¬ p x := by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 0, Real.log x > x - 1) ↔ (∀ x > 0, Real.log x ≤ x - 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l112_11211


namespace NUMINAMATH_CALUDE_pencil_distribution_l112_11279

/-- The number of ways to distribute n identical objects among k people,
    where each person must receive at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

theorem pencil_distribution :
  distribute 7 4 = 52 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l112_11279


namespace NUMINAMATH_CALUDE_paths_through_B_l112_11219

/-- The number of paths between two points on a grid -/
def grid_paths (right : ℕ) (down : ℕ) : ℕ := Nat.choose (right + down) down

/-- The theorem stating the number of 11-step paths from A to C passing through B -/
theorem paths_through_B : 
  let paths_A_to_B := grid_paths 4 2
  let paths_B_to_C := grid_paths 3 3
  paths_A_to_B * paths_B_to_C = 300 := by sorry

end NUMINAMATH_CALUDE_paths_through_B_l112_11219


namespace NUMINAMATH_CALUDE_fraction_simplification_and_division_l112_11259

theorem fraction_simplification_and_division (a x : ℝ) : 
  (a = -2 → (a^2 + a) / (a^2 - 3*a) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2/3) ∧
  ((x^2 - 1) / (x - 4) / ((x + 1) / (4 - x)) = 1 - x) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_division_l112_11259


namespace NUMINAMATH_CALUDE_restaurant_change_l112_11261

/-- Calculates the change received after a restaurant meal -/
theorem restaurant_change 
  (lee_money : ℕ) 
  (friend_money : ℕ) 
  (wings_cost : ℕ) 
  (salad_cost : ℕ) 
  (soda_cost : ℕ) 
  (soda_quantity : ℕ) 
  (tax : ℕ) 
  (h1 : lee_money = 10) 
  (h2 : friend_money = 8) 
  (h3 : wings_cost = 6) 
  (h4 : salad_cost = 4) 
  (h5 : soda_cost = 1) 
  (h6 : soda_quantity = 2) 
  (h7 : tax = 3) : 
  lee_money + friend_money - (wings_cost + salad_cost + soda_cost * soda_quantity + tax) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_change_l112_11261


namespace NUMINAMATH_CALUDE_parallelogram_line_theorem_l112_11237

/-- A parallelogram with vertices at (15,35), (15,95), (27,122), and (27,62) -/
structure Parallelogram :=
  (v1 : ℝ × ℝ) (v2 : ℝ × ℝ) (v3 : ℝ × ℝ) (v4 : ℝ × ℝ)

/-- A line that passes through the origin -/
structure Line :=
  (slope : ℚ)

/-- The line cuts the parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop := sorry

/-- m and n are relatively prime integers -/
def are_relatively_prime (m n : ℕ) : Prop := sorry

theorem parallelogram_line_theorem (p : Parallelogram) (l : Line) 
  (h1 : p.v1 = (15, 35)) (h2 : p.v2 = (15, 95)) (h3 : p.v3 = (27, 122)) (h4 : p.v4 = (27, 62))
  (h5 : cuts_into_congruent_polygons p l)
  (h6 : ∃ (m n : ℕ), l.slope = m / n ∧ are_relatively_prime m n) :
  ∃ (m n : ℕ), l.slope = m / n ∧ are_relatively_prime m n ∧ m + n = 71 := by sorry

end NUMINAMATH_CALUDE_parallelogram_line_theorem_l112_11237


namespace NUMINAMATH_CALUDE_line_segment_both_symmetric_l112_11284

-- Define the shapes
inductive Shape
| EquilateralTriangle
| IsoscelesTriangle
| Parallelogram
| LineSegment

-- Define symmetry properties
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.LineSegment => true
  | _ => false

def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.IsoscelesTriangle => true
  | Shape.LineSegment => true
  | _ => false

-- Theorem statement
theorem line_segment_both_symmetric :
  ∀ s : Shape, (isCentrallySymmetric s ∧ isAxiallySymmetric s) ↔ s = Shape.LineSegment :=
by sorry

end NUMINAMATH_CALUDE_line_segment_both_symmetric_l112_11284


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l112_11288

/-- 
Given an ellipse b^2 x^2 + a^2 y^2 = a^2 b^2 and a line y = px + q,
this theorem states the condition for the line to be tangent to the ellipse
and provides the coordinates of the tangency point.
-/
theorem line_tangent_to_ellipse 
  (a b p q : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hq : q ≠ 0) :
  (∀ x y : ℝ, b^2 * x^2 + a^2 * y^2 = a^2 * b^2 ∧ y = p * x + q) →
  (a^2 * p^2 + b^2 = q^2 ∧ 
   ∃ x y : ℝ, x = -a^2 * p / q ∧ y = b^2 / q ∧ 
   b^2 * x^2 + a^2 * y^2 = a^2 * b^2 ∧ y = p * x + q) :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l112_11288


namespace NUMINAMATH_CALUDE_number_added_to_55_l112_11225

theorem number_added_to_55 : ∃ x : ℤ, 55 + x = 88 ∧ x = 33 := by sorry

end NUMINAMATH_CALUDE_number_added_to_55_l112_11225


namespace NUMINAMATH_CALUDE_propositions_p_and_q_true_l112_11276

theorem propositions_p_and_q_true : (∃ x₀ : ℝ, x₀^2 < x₀) ∧ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_true_l112_11276


namespace NUMINAMATH_CALUDE_geometric_sequences_theorem_l112_11248

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences where
  a : ℝ
  q : ℝ
  r : ℝ
  a_pos : a > 0
  b1_minus_a1 : a * r - a = 1
  b2_minus_a2 : a * r * r - a * q = 2
  b3_minus_a3 : a * r^3 - a * q^2 = 3

/-- The general term of the sequence a_n -/
def a_n (gs : GeometricSequences) (n : ℕ) : ℝ := gs.a * gs.q^(n-1)

/-- The general term of the sequence b_n -/
def b_n (gs : GeometricSequences) (n : ℕ) : ℝ := gs.a * gs.r^(n-1)

theorem geometric_sequences_theorem (gs : GeometricSequences) :
  (gs.a = 1 → (∀ n : ℕ, a_n gs n = (2 + Real.sqrt 2)^(n-1) ∨ a_n gs n = (2 - Real.sqrt 2)^(n-1))) ∧
  ((∃! q : ℝ, ∀ n : ℕ, a_n gs n = gs.a * q^(n-1)) → gs.a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequences_theorem_l112_11248


namespace NUMINAMATH_CALUDE_characterize_f_l112_11260

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ a b c : ℕ, a ≥ 2 → b ≥ 2 → c ≥ 2 →
    (f^[a*b*c - a] (a*b*c)) + (f^[a*b*c - b] (a*b*c)) + (f^[a*b*c - c] (a*b*c)) = a + b + c

theorem characterize_f (f : ℕ → ℕ) (h : is_valid_f f) :
  ∀ n : ℕ, n ≥ 3 → f n = n - 1 :=
sorry

end NUMINAMATH_CALUDE_characterize_f_l112_11260


namespace NUMINAMATH_CALUDE_farmer_problem_l112_11224

theorem farmer_problem (total_cost : ℕ) (rabbit_cost chicken_cost : ℕ) 
  (h_total : total_cost = 1125)
  (h_rabbit : rabbit_cost = 30)
  (h_chicken : chicken_cost = 45) :
  ∃! (r c : ℕ), 
    r > 0 ∧ c > 0 ∧ 
    r * rabbit_cost + c * chicken_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_problem_l112_11224


namespace NUMINAMATH_CALUDE_fraction_transformation_l112_11272

theorem fraction_transformation (n : ℚ) : (4 + n) / (7 + n) = 2 / 3 → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_transformation_l112_11272


namespace NUMINAMATH_CALUDE_unit_digit_product_l112_11254

theorem unit_digit_product : ∃ n : ℕ, (3^68 * 6^59 * 7^71) % 10 = 8 ∧ n = (3^68 * 6^59 * 7^71) := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_product_l112_11254


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l112_11215

/-- Given an ellipse with equation 9x^2 + 16y^2 = 144, the distance between its foci is 2√7 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + 16 * y^2 = 144) → (∃ f₁ f₂ : ℝ × ℝ, 
    f₁ ≠ f₂ ∧ 
    (∀ p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 144 → 
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 8) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l112_11215


namespace NUMINAMATH_CALUDE_no_solution_condition_l112_11295

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (x - a) / (x - 1) - 3 / x ≠ 1) ↔ (a = 1 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l112_11295


namespace NUMINAMATH_CALUDE_find_number_l112_11283

theorem find_number : ∃ x : ℝ, x / 2 = 9 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l112_11283


namespace NUMINAMATH_CALUDE_line_with_acute_inclination_l112_11217

/-- Given a line passing through points A(2,1) and B(1,m) with an acute angle of inclination, 
    the value of m must be less than 1. -/
theorem line_with_acute_inclination (m : ℝ) : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (1, m)
  let slope : ℝ := (m - A.2) / (B.1 - A.1)
  (0 < slope) ∧ (slope < 1) → m < 1 := by sorry

end NUMINAMATH_CALUDE_line_with_acute_inclination_l112_11217


namespace NUMINAMATH_CALUDE_john_profit_l112_11229

def calculate_profit (woodburning_qty : ℕ) (woodburning_price : ℚ)
                     (metal_qty : ℕ) (metal_price : ℚ)
                     (painting_qty : ℕ) (painting_price : ℚ)
                     (glass_qty : ℕ) (glass_price : ℚ)
                     (wood_cost : ℚ) (metal_cost : ℚ)
                     (paint_cost : ℚ) (glass_cost : ℚ)
                     (woodburning_discount : ℚ) (glass_discount : ℚ)
                     (sales_tax : ℚ) : ℚ :=
  sorry

theorem john_profit :
  calculate_profit 20 15 15 25 10 40 5 30 100 150 120 90 (10/100) (15/100) (5/100) = 771.13 :=
sorry

end NUMINAMATH_CALUDE_john_profit_l112_11229


namespace NUMINAMATH_CALUDE_leftover_coins_value_l112_11209

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40
def sally_quarters : ℕ := 101
def sally_dimes : ℕ := 173
def ben_quarters : ℕ := 150
def ben_dimes : ℕ := 195
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coins_value :
  let total_quarters := sally_quarters + ben_quarters
  let total_dimes := sally_dimes + ben_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 3.55 := by sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l112_11209


namespace NUMINAMATH_CALUDE_opposite_solutions_imply_a_value_l112_11269

theorem opposite_solutions_imply_a_value (a x y : ℚ) : 
  (x - y = 3 * a + 1) → 
  (x + y = 9 - 5 * a) → 
  (x = -y) → 
  (a = 9 / 5) := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_imply_a_value_l112_11269


namespace NUMINAMATH_CALUDE_painted_cells_20210_1505_l112_11232

/-- The number of unique cells painted by two diagonals in a rectangle --/
def painted_cells (width height : ℕ) : ℕ :=
  let gcd := width.gcd height
  let subrect_width := width / gcd
  let subrect_height := height / gcd
  let cells_per_subrect := subrect_width + subrect_height - 1
  let total_cells := 2 * gcd * cells_per_subrect
  let overlap_cells := gcd
  total_cells - overlap_cells

/-- Theorem stating the number of painted cells in a 20210 × 1505 rectangle --/
theorem painted_cells_20210_1505 :
  painted_cells 20210 1505 = 42785 := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_20210_1505_l112_11232


namespace NUMINAMATH_CALUDE_blue_balls_count_l112_11243

theorem blue_balls_count (red green : ℕ) (p : ℚ) (blue : ℕ) : 
  red = 4 → 
  green = 2 → 
  p = 4/30 → 
  (red / (red + blue + green : ℚ)) * ((red - 1) / (red + blue + green - 1 : ℚ)) = p → 
  blue = 4 :=
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l112_11243


namespace NUMINAMATH_CALUDE_right_triangle_construction_l112_11273

/-- Given a length b (representing one leg) and a length c (representing the projection of the other leg onto the hypotenuse), a right triangle can be constructed. -/
theorem right_triangle_construction (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  ∃ (a x : ℝ), a > 0 ∧ x > 0 ∧ x + c = a ∧ b^2 = x * a :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_construction_l112_11273


namespace NUMINAMATH_CALUDE_prime_mod_8_not_sum_of_three_squares_l112_11289

theorem prime_mod_8_not_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬ ∃ (a b c : ℤ), (a ^ 2 + b ^ 2 + c ^ 2 : ℤ) = p := by
  sorry

end NUMINAMATH_CALUDE_prime_mod_8_not_sum_of_three_squares_l112_11289


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l112_11233

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → -2 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l112_11233


namespace NUMINAMATH_CALUDE_chess_tournament_games_l112_11255

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l112_11255


namespace NUMINAMATH_CALUDE_A_inter_B_eq_B_l112_11252

-- Define set A
def A : Set ℝ := {y | ∃ x, y = |x| - 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem A_inter_B_eq_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_B_l112_11252


namespace NUMINAMATH_CALUDE_complex_product_theorem_l112_11292

theorem complex_product_theorem (Q E D : ℂ) : 
  Q = 3 + 4*I ∧ E = 2*I ∧ D = 3 - 4*I → 2 * Q * E * D = 100 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l112_11292


namespace NUMINAMATH_CALUDE_circumscribed_parallelepiped_surface_area_l112_11202

/-- A right parallelepiped circumscribed by a sphere -/
structure CircumscribedParallelepiped where
  /-- The first base diagonal of the parallelepiped -/
  a : ℝ
  /-- The second base diagonal of the parallelepiped -/
  b : ℝ
  /-- The parallelepiped is circumscribed by a sphere -/
  is_circumscribed : True

/-- The surface area of a circumscribed parallelepiped -/
def surface_area (p : CircumscribedParallelepiped) : ℝ :=
  6 * p.a * p.b

/-- Theorem: The surface area of a right parallelepiped circumscribed by a sphere,
    with base diagonals a and b, is equal to 6ab -/
theorem circumscribed_parallelepiped_surface_area
  (p : CircumscribedParallelepiped) :
  surface_area p = 6 * p.a * p.b := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_parallelepiped_surface_area_l112_11202


namespace NUMINAMATH_CALUDE_student_council_committees_l112_11226

theorem student_council_committees (x : ℕ) : 
  (x.choose 3 = 20) → (x.choose 4 = 15) := by sorry

end NUMINAMATH_CALUDE_student_council_committees_l112_11226


namespace NUMINAMATH_CALUDE_money_distribution_exists_l112_11277

/-- Represents the money distribution problem with Ram, Gopal, Krishan, and Shekhar -/
def MoneyDistribution (x : ℚ) : Prop :=
  let ram_share := 7
  let gopal_share := 17
  let krishan_share := 17
  let shekhar_share := x
  let ram_money := 490
  let unit_value := ram_money / ram_share
  let gopal_shekhar_ratio := 2 / 1
  (gopal_share / shekhar_share = gopal_shekhar_ratio) ∧
  (shekhar_share * unit_value = 595)

/-- Theorem stating that there exists a valid money distribution satisfying all conditions -/
theorem money_distribution_exists : ∃ x, MoneyDistribution x := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_exists_l112_11277


namespace NUMINAMATH_CALUDE_james_nickels_l112_11220

/-- Represents the number of nickels in James' jar -/
def n : ℕ := sorry

/-- Represents the number of quarters in James' jar -/
def q : ℕ := sorry

/-- The total value in cents -/
def total_cents : ℕ := 685

/-- Theorem stating the number of nickels in James' jar -/
theorem james_nickels : 
  (5 * n + 25 * q = total_cents) ∧ 
  (n = q + 11) → 
  n = 32 := by sorry

end NUMINAMATH_CALUDE_james_nickels_l112_11220


namespace NUMINAMATH_CALUDE_boat_speed_calculation_l112_11293

/-- Given the downstream speed and upstream speed of a boat, 
    calculate the stream speed and the man's rowing speed. -/
theorem boat_speed_calculation (R S : ℝ) :
  ∃ (x y : ℝ), 
    (R = y + x) ∧ 
    (S = y - x) ∧ 
    (x = (R - S) / 2) ∧ 
    (y = (R + S) / 2) := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_calculation_l112_11293


namespace NUMINAMATH_CALUDE_max_product_constraint_l112_11204

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → a + 2 * b = 10 → ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + 2 * y = 10 → x * y ≤ m ∧ a * b = m :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l112_11204


namespace NUMINAMATH_CALUDE_errand_time_is_110_minutes_l112_11251

def driving_time_one_way : ℕ := 20
def parent_teacher_night_time : ℕ := 70

def total_errand_time : ℕ :=
  2 * driving_time_one_way + parent_teacher_night_time

theorem errand_time_is_110_minutes :
  total_errand_time = 110 := by
  sorry

end NUMINAMATH_CALUDE_errand_time_is_110_minutes_l112_11251


namespace NUMINAMATH_CALUDE_english_only_students_l112_11262

/-- Represents the number of students in each language class -/
structure LanguageClasses where
  english : ℕ
  french : ℕ
  spanish : ℕ

/-- The conditions of the problem -/
def language_class_conditions (c : LanguageClasses) : Prop :=
  c.english + c.french + c.spanish = 40 ∧
  c.english = 3 * c.french ∧
  c.english = 2 * c.spanish

/-- The theorem to prove -/
theorem english_only_students (c : LanguageClasses) 
  (h : language_class_conditions c) : 
  c.english - (c.french + c.spanish) = 30 := by
  sorry


end NUMINAMATH_CALUDE_english_only_students_l112_11262


namespace NUMINAMATH_CALUDE_equality_of_fractions_l112_11205

theorem equality_of_fractions (x y z l : ℝ) :
  (9 / (x + y + 1) = l / (x + z - 1)) ∧
  (l / (x + z - 1) = 13 / (z - y + 2)) →
  l = 22 := by
sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l112_11205


namespace NUMINAMATH_CALUDE_ternary_1021_is_34_l112_11263

def ternary_to_decimal (t : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 3 + d) 0 t.reverse

theorem ternary_1021_is_34 :
  ternary_to_decimal [1, 0, 2, 1] = 34 := by
  sorry

end NUMINAMATH_CALUDE_ternary_1021_is_34_l112_11263


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l112_11275

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Theorem: Given a hyperbola with specific asymptote and focus, determine its parameters -/
theorem hyperbola_parameters 
  (h : Hyperbola) 
  (h_asymptote : b / a = 2) 
  (h_focus : Real.sqrt (a^2 + b^2) = Real.sqrt 5) : 
  h.a = 1 ∧ h.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l112_11275


namespace NUMINAMATH_CALUDE_alice_sold_120_oranges_l112_11201

/-- The number of oranges Emily sold -/
def emily_oranges : ℕ := sorry

/-- The number of oranges Alice sold -/
def alice_oranges : ℕ := 2 * emily_oranges

/-- The total number of oranges sold -/
def total_oranges : ℕ := 180

/-- Theorem stating that Alice sold 120 oranges -/
theorem alice_sold_120_oranges : alice_oranges = 120 :=
  by
    sorry

/-- The condition that the total oranges sold is the sum of Alice's and Emily's oranges -/
axiom total_is_sum : total_oranges = alice_oranges + emily_oranges

end NUMINAMATH_CALUDE_alice_sold_120_oranges_l112_11201


namespace NUMINAMATH_CALUDE_average_sale_per_month_l112_11239

def sales : List ℝ := [2435, 2920, 2855, 3230, 2560, 1000]

theorem average_sale_per_month :
  (sales.sum / sales.length : ℝ) = 2500 := by sorry

end NUMINAMATH_CALUDE_average_sale_per_month_l112_11239


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l112_11285

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l112_11285


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l112_11241

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ (x y z : ℝ), y = m * x + z + 2 ∧ y = (3 * m - 2) * x + z + 5) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l112_11241


namespace NUMINAMATH_CALUDE_custom_ops_theorem_l112_11250

/-- Custom addition operation for natural numbers -/
def customAdd (a b : ℕ) : ℕ := a + b + 1

/-- Custom multiplication operation for natural numbers -/
def customMul (a b : ℕ) : ℕ := a * b - 1

/-- Theorem stating that (5 ⊕ 7) ⊕ (2 ⊗ 4) = 21 -/
theorem custom_ops_theorem : customAdd (customAdd 5 7) (customMul 2 4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_ops_theorem_l112_11250


namespace NUMINAMATH_CALUDE_tree_planting_campaign_l112_11207

theorem tree_planting_campaign (february_trees : ℕ) (planned_trees : ℕ) : 
  february_trees = planned_trees * 19 / 20 →
  (planned_trees * 11 / 10 : ℚ) = 
    (planned_trees * 11 / 10 : ℕ) ∧ planned_trees > 0 →
  ∃ (total_trees : ℕ), total_trees = planned_trees * 11 / 10 :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_campaign_l112_11207


namespace NUMINAMATH_CALUDE_max_a_inequality_l112_11265

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, Real.sqrt (2 * x) - a ≥ Real.sqrt (9 - 5 * x)) → 
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_max_a_inequality_l112_11265


namespace NUMINAMATH_CALUDE_ninas_toys_l112_11228

theorem ninas_toys (toy_price : ℕ) (card_packs : ℕ) (card_price : ℕ) (shirts : ℕ) (shirt_price : ℕ) (total_spent : ℕ) : 
  toy_price = 10 →
  card_packs = 2 →
  card_price = 5 →
  shirts = 5 →
  shirt_price = 6 →
  total_spent = 70 →
  ∃ (num_toys : ℕ), num_toys * toy_price + card_packs * card_price + shirts * shirt_price = total_spent ∧ num_toys = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ninas_toys_l112_11228


namespace NUMINAMATH_CALUDE_inequality_proof_l112_11249

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x^2*y + x*y^2 + 1 ≤ x^2*y^2 + x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l112_11249


namespace NUMINAMATH_CALUDE_constant_zero_arithmetic_not_geometric_l112_11227

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def constant_zero_sequence : ℕ → ℝ :=
  λ _ => 0

theorem constant_zero_arithmetic_not_geometric :
  is_arithmetic_sequence constant_zero_sequence ∧
  ¬ is_geometric_sequence constant_zero_sequence :=
by sorry

end NUMINAMATH_CALUDE_constant_zero_arithmetic_not_geometric_l112_11227


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l112_11296

theorem imaginary_part_of_complex (z : ℂ) (h : z = 1 - 2 * Complex.I) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l112_11296


namespace NUMINAMATH_CALUDE_divisibility_proof_l112_11242

theorem divisibility_proof :
  ∃ (n : ℕ), (425897 + n) % 456 = 0 ∧ n = 47 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l112_11242


namespace NUMINAMATH_CALUDE_curve_length_is_pi_l112_11270

/-- A closed convex curve in the plane -/
structure ClosedConvexCurve where
  -- Add necessary fields here
  -- This is just a placeholder definition

/-- The length of a curve -/
noncomputable def curve_length (c : ClosedConvexCurve) : ℝ :=
  sorry

/-- The length of the projection of a curve onto a line -/
noncomputable def projection_length (c : ClosedConvexCurve) (l : Line) : ℝ :=
  sorry

/-- A line in the plane -/
structure Line where
  -- Add necessary fields here
  -- This is just a placeholder definition

theorem curve_length_is_pi (c : ClosedConvexCurve) 
  (h : ∀ l : Line, projection_length c l = 1) : 
  curve_length c = π :=
sorry

end NUMINAMATH_CALUDE_curve_length_is_pi_l112_11270


namespace NUMINAMATH_CALUDE_combination_equations_l112_11235

def A (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem combination_equations :
  (∃! x : ℕ+, 3 * (A x.val 3) = 2 * (A (x.val + 1) 2) + 6 * (A x.val 2)) ∧
  (∃ x : ℕ+, x = 1 ∨ x = 2) ∧ (∀ x : ℕ+, A 8 x.val = A 8 (5 * x.val - 4) → x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_combination_equations_l112_11235


namespace NUMINAMATH_CALUDE_minimum_cost_for_boxes_l112_11206

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the problem parameters -/
structure ProblemParams where
  boxDims : BoxDimensions
  costPerBox : ℝ
  totalVolumeNeeded : ℝ

theorem minimum_cost_for_boxes (p : ProblemParams)
  (h1 : p.boxDims.length = 20)
  (h2 : p.boxDims.width = 20)
  (h3 : p.boxDims.height = 12)
  (h4 : p.costPerBox = 0.4)
  (h5 : p.totalVolumeNeeded = 2160000) :
  ⌈p.totalVolumeNeeded / boxVolume p.boxDims⌉ * p.costPerBox = 180 := by
  sorry

#check minimum_cost_for_boxes

end NUMINAMATH_CALUDE_minimum_cost_for_boxes_l112_11206


namespace NUMINAMATH_CALUDE_unique_scores_count_l112_11223

/-- Represents the number of baskets made by the player -/
def total_baskets : ℕ := 7

/-- Represents the possible point values for each basket -/
inductive BasketType
| two_point : BasketType
| three_point : BasketType

/-- Calculates the total score given a list of basket types -/
def calculate_score (baskets : List BasketType) : ℕ :=
  baskets.foldl (fun acc b => acc + match b with
    | BasketType.two_point => 2
    | BasketType.three_point => 3) 0

/-- Generates all possible combinations of basket types -/
def generate_combinations : List (List BasketType) :=
  sorry

/-- Theorem stating that the number of unique possible scores is 8 -/
theorem unique_scores_count :
  (generate_combinations.map calculate_score).toFinset.card = 8 := by sorry

end NUMINAMATH_CALUDE_unique_scores_count_l112_11223


namespace NUMINAMATH_CALUDE_m_range_theorem_l112_11221

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

-- State the theorem
theorem m_range_theorem : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l112_11221


namespace NUMINAMATH_CALUDE_circle_intersection_range_l112_11299

/-- The circle described by (x-a)^2 + (y-a)^2 = 4 always has two distinct points
    at distance 1 from the origin if and only if a is in the given range -/
theorem circle_intersection_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + (y₁ - a)^2 = 4 ∧ 
    (x₂ - a)^2 + (y₂ - a)^2 = 4 ∧
    x₁^2 + y₁^2 = 1 ∧
    x₂^2 + y₂^2 = 1 ∧
    (x₁, y₁) ≠ (x₂, y₂)) ↔ 
  (a ∈ Set.Ioo (-3 * Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) ∪ 
       Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_range_l112_11299


namespace NUMINAMATH_CALUDE_profit_equation_store_profit_equation_l112_11246

/-- Represents the profit equation for a store selling goods -/
theorem profit_equation (initial_price initial_cost initial_volume : ℕ) 
                        (price_increase : ℕ) (volume_decrease_rate : ℕ) 
                        (profit_increase : ℕ) : Prop :=
  let new_price := initial_price + price_increase
  let new_volume := initial_volume - volume_decrease_rate * price_increase
  let profit_per_unit := new_price - initial_cost
  profit_per_unit * new_volume = initial_volume * (initial_price - initial_cost) + profit_increase

/-- The specific profit equation for the given problem -/
theorem store_profit_equation (x : ℕ) : 
  profit_equation 36 20 200 x 5 1200 ↔ (x + 16) * (200 - 5 * x) = 1200 :=
sorry

end NUMINAMATH_CALUDE_profit_equation_store_profit_equation_l112_11246


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l112_11257

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_twelve_divisors : 
  ∀ n : ℕ, n > 0 → divisor_count n = 12 → n ≥ 96 :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l112_11257


namespace NUMINAMATH_CALUDE_tea_customers_l112_11234

/-- Proves that the number of tea customers is 8 given the conditions of the problem -/
theorem tea_customers (coffee_price : ℕ) (tea_price : ℕ) (coffee_customers : ℕ) (total_revenue : ℕ) :
  coffee_price = 5 →
  tea_price = 4 →
  coffee_customers = 7 →
  total_revenue = 67 →
  ∃ tea_customers : ℕ, 
    tea_customers = 8 ∧ 
    coffee_price * coffee_customers + tea_price * tea_customers = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_tea_customers_l112_11234


namespace NUMINAMATH_CALUDE_solution_replacement_concentration_l112_11287

theorem solution_replacement_concentration 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (replaced_fraction : ℝ) 
  (replacement_concentration : ℝ) : 
  initial_concentration = 45 ∧ 
  final_concentration = 35 ∧ 
  replaced_fraction = 0.5 → 
  replacement_concentration = 25 := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_concentration_l112_11287


namespace NUMINAMATH_CALUDE_bella_age_l112_11253

theorem bella_age (bella_age : ℕ) (brother_age : ℕ) : 
  brother_age = bella_age + 9 →
  bella_age + brother_age = 19 →
  bella_age = 5 := by
sorry

end NUMINAMATH_CALUDE_bella_age_l112_11253


namespace NUMINAMATH_CALUDE_steps_climbed_proof_l112_11245

/-- Calculates the total number of steps climbed given the number of steps and climbs for two ladders -/
def total_steps_climbed (full_ladder_steps : ℕ) (full_ladder_climbs : ℕ) 
                        (small_ladder_steps : ℕ) (small_ladder_climbs : ℕ) : ℕ :=
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

/-- Proves that the total number of steps climbed is 152 given the specific ladder configurations -/
theorem steps_climbed_proof :
  total_steps_climbed 11 10 6 7 = 152 := by
  sorry

end NUMINAMATH_CALUDE_steps_climbed_proof_l112_11245


namespace NUMINAMATH_CALUDE_exists_sequence_expectation_sum_neq_sum_expectation_l112_11267

/-- A sequence of random variables -/
def RandomSequence := ℕ → MeasurableSpace ℝ

/-- Expected value of a random variable -/
noncomputable def expectation (X : MeasurableSpace ℝ) : ℝ := sorry

/-- Infinite sum of random variables -/
noncomputable def infiniteSum (ξ : RandomSequence) : MeasurableSpace ℝ := sorry

/-- Theorem: There exists a sequence of random variables where the expectation of the sum
    is not equal to the sum of expectations -/
theorem exists_sequence_expectation_sum_neq_sum_expectation :
  ∃ ξ : RandomSequence,
    expectation (infiniteSum ξ) ≠ ∑' n, expectation (ξ n) := by sorry

end NUMINAMATH_CALUDE_exists_sequence_expectation_sum_neq_sum_expectation_l112_11267


namespace NUMINAMATH_CALUDE_probability_ratio_l112_11231

def num_balls : ℕ := 20
def num_bins : ℕ := 5

def distribution_A : List ℕ := [2, 4, 4, 3, 7]
def distribution_B : List ℕ := [3, 3, 4, 4, 4]

def probability_A : ℚ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.choose 2 1) * (Nat.factorial 20) / 
  ((Nat.factorial 2) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 7) * (Nat.choose (num_balls + num_bins - 1) (num_bins - 1)))

def probability_B : ℚ := (Nat.choose 5 2) * (Nat.choose 3 3) * (Nat.factorial 20) / 
  ((Nat.factorial 3) * (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.choose (num_balls + num_bins - 1) (num_bins - 1)))

theorem probability_ratio : probability_A / probability_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l112_11231


namespace NUMINAMATH_CALUDE_remainder_theorem_l112_11264

theorem remainder_theorem (n : ℤ) (h : n % 5 = 3) : (7 * n + 4) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l112_11264


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_five_sqrt_two_over_two_l112_11290

theorem sqrt_sum_equals_five_sqrt_two_over_two :
  Real.sqrt 8 + Real.sqrt (1/2) = (5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_five_sqrt_two_over_two_l112_11290


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l112_11236

theorem min_value_expression (x y : ℝ) (h1 : y > 0) (h2 : y = -1/x + 1) :
  2*x + 1/y ≥ 2*Real.sqrt 2 + 3 :=
by
  sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), y > 0 ∧ y = -1/x + 1 ∧ 2*x + 1/y = 2*Real.sqrt 2 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l112_11236


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonpositive_l112_11240

theorem quadratic_inequality_always_nonpositive :
  ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonpositive_l112_11240


namespace NUMINAMATH_CALUDE_acute_angles_subset_first_quadrant_l112_11274

-- Define the sets M, N, and P
def M : Set ℝ := {θ | 0 < θ ∧ θ < 90}
def N : Set ℝ := {θ | θ < 90}
def P : Set ℝ := {θ | ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90}

-- Theorem to prove
theorem acute_angles_subset_first_quadrant : M ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_subset_first_quadrant_l112_11274


namespace NUMINAMATH_CALUDE_pizza_price_proof_l112_11268

/-- The standard price of a pizza at Piazzanos Pizzeria -/
def standard_price : ℚ := 5

/-- The number of triple cheese pizzas purchased -/
def triple_cheese_count : ℕ := 10

/-- The number of meat lovers pizzas purchased -/
def meat_lovers_count : ℕ := 9

/-- The total cost of the purchase -/
def total_cost : ℚ := 55

theorem pizza_price_proof :
  (triple_cheese_count / 2 + meat_lovers_count * 2 / 3) * standard_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_pizza_price_proof_l112_11268


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l112_11258

theorem remaining_problems_to_grade 
  (problems_per_paper : ℕ) 
  (total_papers : ℕ) 
  (graded_papers : ℕ) 
  (h1 : problems_per_paper = 15)
  (h2 : total_papers = 45)
  (h3 : graded_papers = 18)
  : (total_papers - graded_papers) * problems_per_paper = 405 :=
by sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l112_11258


namespace NUMINAMATH_CALUDE_simplify_expression_l112_11213

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l112_11213


namespace NUMINAMATH_CALUDE_frog_jump_theorem_l112_11266

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Function to calculate the number of final frog positions -/
def numFrogPositions (ℓ : ℕ) : ℕ :=
  ((ℓ + 2) / 2) * ((ℓ + 4) / 2) * (((ℓ + 1) / 2) * ((ℓ + 3) / 2))^2 / 8

/-- Main theorem statement -/
theorem frog_jump_theorem (abc : Triangle) (ℓ : ℕ) (m n : Point) :
  (abc.a.x = 0 ∧ abc.a.y = 0) →  -- A at origin
  (abc.b.x = 1 ∧ abc.b.y = 0) →  -- B at (1,0)
  (abc.c.x = 1/2 ∧ abc.c.y = Real.sqrt 3 / 2) →  -- C at (1/2, √3/2)
  (m.x = ℓ ∧ m.y = 0) →  -- M on AB
  (n.x = ℓ/2 ∧ n.y = ℓ * Real.sqrt 3 / 2) →  -- N on AC
  ∃ (finalPositions : ℕ), finalPositions = numFrogPositions ℓ :=
by sorry

end NUMINAMATH_CALUDE_frog_jump_theorem_l112_11266
