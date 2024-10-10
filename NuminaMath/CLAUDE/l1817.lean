import Mathlib

namespace collinearity_condition_acute_angle_condition_l1817_181764

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Define acute angle
def acute_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) > 0

-- Theorem for collinearity
theorem collinearity_condition (m : ℝ) :
  collinear OA OB (OC m) ↔ m = 1/2 := by sorry

-- Theorem for acute angle
theorem acute_angle_condition (m : ℝ) :
  acute_angle OA OB (OC m) ↔ m ∈ Set.Ioo (-3/4) (1/2) ∪ Set.Ioi (1/2) := by sorry

end collinearity_condition_acute_angle_condition_l1817_181764


namespace expression_simplification_and_evaluation_l1817_181731

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 + Real.sqrt 2
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_and_evaluation_l1817_181731


namespace intersection_of_M_and_N_l1817_181777

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end intersection_of_M_and_N_l1817_181777


namespace min_value_theorem_l1817_181730

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 15) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 19 ∧
  ((x + 15) / Real.sqrt (x - 4) = 2 * Real.sqrt 19 ↔ x = 23) := by
sorry

end min_value_theorem_l1817_181730


namespace circle_ratio_l1817_181793

theorem circle_ratio (R r a b : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : 0 < a) (h4 : 0 < b) 
  (h5 : π * R^2 = (b/a) * (π * R^2 - π * r^2)) : 
  R / r = (b / (a - b))^(1/2) := by
  sorry

end circle_ratio_l1817_181793


namespace field_area_is_625_l1817_181701

/-- Represents a square field -/
structure SquareField where
  /-- The length of one side of the square field in kilometers -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- Calculates the perimeter of a square field -/
def perimeter (field : SquareField) : ℝ := 4 * field.side

/-- Calculates the area of a square field -/
def area (field : SquareField) : ℝ := field.side ^ 2

/-- The speed of the horse in km/h -/
def horse_speed : ℝ := 25

/-- The time taken by the horse to run around the field in hours -/
def lap_time : ℝ := 4

theorem field_area_is_625 (field : SquareField) 
  (h : perimeter field = horse_speed * lap_time) : 
  area field = 625 := by
  sorry

end field_area_is_625_l1817_181701


namespace lydia_almonds_l1817_181734

theorem lydia_almonds (lydia_almonds max_almonds : ℕ) : 
  lydia_almonds = max_almonds + 8 →
  max_almonds = lydia_almonds / 3 →
  lydia_almonds = 12 := by
sorry

end lydia_almonds_l1817_181734


namespace opposite_number_theorem_l1817_181760

theorem opposite_number_theorem (m : ℤ) : (m + 1 = -(-4)) → m = 3 := by
  sorry

end opposite_number_theorem_l1817_181760


namespace binomial_expansion_cube_problem_solution_l1817_181721

theorem binomial_expansion_cube (x : ℕ) :
  x^3 + 3*(x^2) + 3*x + 1 = (x + 1)^3 :=
by sorry

theorem problem_solution : 
  85^3 + 3*(85^2) + 3*85 + 1 = 636256 :=
by sorry

end binomial_expansion_cube_problem_solution_l1817_181721


namespace simplify_expression_l1817_181733

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 + 3 * z^2) = -1 - 8 * z^2 := by
  sorry

end simplify_expression_l1817_181733


namespace cubic_roots_problem_l1817_181783

theorem cubic_roots_problem (a b c : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : a + b + c = -1)
  (h4 : a * b + b * c + a * c = -4)
  (h5 : a * b * c = -2) : 
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 := by
  sorry

end cubic_roots_problem_l1817_181783


namespace milk_jug_problem_l1817_181780

theorem milk_jug_problem (x y : ℝ) : 
  x + y = 70 ∧ 
  0.875 * x = y + 0.125 * x → 
  x = 40 ∧ y = 30 := by
sorry

end milk_jug_problem_l1817_181780


namespace inscribed_triangle_theorem_l1817_181755

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The segments of one side divided by the point of tangency
  s₁ : ℝ
  s₂ : ℝ
  -- Conditions
  side_division : a = s₁ + s₂
  radius_positive : r > 0
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem stating the relationship between the sides and radius -/
theorem inscribed_triangle_theorem (t : InscribedTriangle) 
  (h₁ : t.s₁ = 10 ∧ t.s₂ = 14)
  (h₂ : t.r = 5)
  (h₃ : t.b = 30) :
  t.c = 36 := by
  sorry

end inscribed_triangle_theorem_l1817_181755


namespace quadratic_equation_roots_l1817_181761

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + k*y - 2 = 0 ∧ y = 1 ∧ k = 1) :=
by sorry

end quadratic_equation_roots_l1817_181761


namespace f_range_theorem_l1817_181794

-- Define the function f
def f (x y z : ℝ) := (z - x) * (z - y)

-- State the theorem
theorem f_range_theorem (x y z : ℝ) 
  (h1 : x + y + z = 1) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) 
  (h4 : z ≥ 0) :
  ∃ (w : ℝ), f x y z = w ∧ w ∈ Set.Icc (-1/8 : ℝ) 1 := by
  sorry

end f_range_theorem_l1817_181794


namespace area_ratio_quadrupled_triangle_l1817_181744

/-- Given a triangle whose dimensions are quadrupled to form a larger triangle,
    this theorem relates the area of the larger triangle to the area of the original triangle. -/
theorem area_ratio_quadrupled_triangle (A : ℝ) :
  (4 * 4 * A = 64) → (A = 4) := by
  sorry

end area_ratio_quadrupled_triangle_l1817_181744


namespace seats_filled_percentage_l1817_181740

/-- The percentage of filled seats in a public show -/
def percentage_filled (total_seats vacant_seats : ℕ) : ℚ :=
  (total_seats - vacant_seats : ℚ) / total_seats * 100

/-- Theorem stating that the percentage of filled seats is 62% -/
theorem seats_filled_percentage (total_seats vacant_seats : ℕ)
  (h1 : total_seats = 600)
  (h2 : vacant_seats = 228) :
  percentage_filled total_seats vacant_seats = 62 := by
  sorry

end seats_filled_percentage_l1817_181740


namespace angle_side_relationship_l1817_181739

-- Define a triangle with angles A, B, C and sides a, b, c
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Triangle inequality
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  -- Angle sum in a triangle is π
  angle_sum : A + B + C = π
  -- Side lengths satisfy triangle inequality
  side_ineq_a : a < b + c
  side_ineq_b : b < a + c
  side_ineq_c : c < a + b

theorem angle_side_relationship (t : Triangle) : t.A > t.B ↔ t.a > t.b := by
  sorry

end angle_side_relationship_l1817_181739


namespace exists_valid_statement_l1817_181773

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a statement type
structure Statement where
  content : String
  canBeMadeBy : PersonType → Prop
  truthValueKnown : Prop

-- Define the property of a valid statement
def validStatement (s : Statement) : Prop :=
  (s.canBeMadeBy PersonType.Normal) ∧
  (¬s.canBeMadeBy PersonType.Knight) ∧
  (¬s.canBeMadeBy PersonType.Liar) ∧
  (¬s.truthValueKnown)

-- Theorem: There exists a valid statement
theorem exists_valid_statement : ∃ s : Statement, validStatement s := by
  sorry

end exists_valid_statement_l1817_181773


namespace option_B_not_mapping_l1817_181798

-- Define the sets and mappings
def CartesianPlane : Type := ℝ × ℝ
def CircleOnPlane : Type := Unit -- Placeholder type for circles
def TriangleOnPlane : Type := Unit -- Placeholder type for triangles

-- Option A
def mappingA : CartesianPlane → CartesianPlane := id

-- Option B (not a mapping)
noncomputable def correspondenceB : CircleOnPlane → Set TriangleOnPlane := sorry

-- Option C
def mappingC : ℕ → Fin 2 := fun n => n % 2

-- Option D
def mappingD : Fin 3 → Fin 3 := fun n => n^2

-- Theorem stating that B is not a mapping while others are
theorem option_B_not_mapping :
  (∀ x : CartesianPlane, ∃! y : CartesianPlane, mappingA x = y) ∧
  (∃ c : CircleOnPlane, ¬∃! t : TriangleOnPlane, t ∈ correspondenceB c) ∧
  (∀ n : ℕ, ∃! m : Fin 2, mappingC n = m) ∧
  (∀ x : Fin 3, ∃! y : Fin 3, mappingD x = y) := by
  sorry

end option_B_not_mapping_l1817_181798


namespace quadratic_minimum_l1817_181747

/-- Given a quadratic function f(x) = x^2 + px + qx where p and q are positive constants,
    prove that the x-coordinate of its minimum value occurs at x = -(p+q)/2 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q*x
  ∃ (x_min : ℝ), x_min = -(p + q) / 2 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by
  sorry

end quadratic_minimum_l1817_181747


namespace inequality_proof_l1817_181759

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b / (a+b))^2 + (a*b / (a+b)) * (a*c / (a+c)) + (a*c / (a+c))^2) := by
  sorry

end inequality_proof_l1817_181759


namespace parallel_vectors_x_value_l1817_181790

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x + 1)
  parallel a b → x = 1 :=
by
  sorry

end parallel_vectors_x_value_l1817_181790


namespace product_eleven_sum_possibilities_l1817_181751

theorem product_eleven_sum_possibilities (a b c : ℤ) : 
  a * b * c = -11 → (a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13) := by
  sorry

end product_eleven_sum_possibilities_l1817_181751


namespace inequality_proof_l1817_181745

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) ∧
  ((1 + a / b) ^ n + (1 + b / a) ^ n = 2^(n + 1) ↔ a = b) := by
  sorry

end inequality_proof_l1817_181745


namespace third_coaster_speed_l1817_181758

/-- Theorem: Given 5 rollercoasters with specified speeds and average, prove the speed of the third coaster -/
theorem third_coaster_speed 
  (v1 v2 v3 v4 v5 : ℝ) 
  (h1 : v1 = 50)
  (h2 : v2 = 62)
  (h4 : v4 = 70)
  (h5 : v5 = 40)
  (h_avg : (v1 + v2 + v3 + v4 + v5) / 5 = 59) :
  v3 = 73 := by
sorry

end third_coaster_speed_l1817_181758


namespace circle_area_difference_l1817_181736

theorem circle_area_difference : 
  let r1 : ℝ := 15
  let d2 : ℝ := 14
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 176 * π := by sorry

end circle_area_difference_l1817_181736


namespace prob_red_then_black_our_deck_l1817_181750

/-- A customized deck of cards -/
structure CustomDeck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (deck : CustomDeck) : ℚ :=
  (deck.red_cards : ℚ) * (deck.black_cards : ℚ) / ((deck.total_cards : ℚ) * (deck.total_cards - 1 : ℚ))

/-- Our specific deck -/
def our_deck : CustomDeck :=
  { total_cards := 78
  , red_cards := 39
  , black_cards := 39 }

theorem prob_red_then_black_our_deck :
  prob_red_then_black our_deck = 507 / 2002 := by
  sorry

end prob_red_then_black_our_deck_l1817_181750


namespace matrix_problem_l1817_181711

-- Define 2x2 matrices A and B
variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- State the conditions
axiom cond1 : A * B = A ^ 2 * B ^ 2 - (A * B) ^ 2
axiom cond2 : Matrix.det B = 2

-- Theorem statement
theorem matrix_problem :
  Matrix.det A = 0 ∧ Matrix.det (A + 2 • B) - Matrix.det (B + 2 • A) = 6 := by
  sorry

end matrix_problem_l1817_181711


namespace max_value_abc_inverse_sum_cubed_l1817_181702

theorem max_value_abc_inverse_sum_cubed (a b c : ℝ) (h : a + b + c = 0) :
  abc * (1/a + 1/b + 1/c)^3 ≤ 27/8 :=
sorry

end max_value_abc_inverse_sum_cubed_l1817_181702


namespace shaded_area_calculation_l1817_181774

/-- The area of the shaded region in a grid with given dimensions and an unshaded triangle -/
theorem shaded_area_calculation (grid_width grid_height triangle_base triangle_height : ℝ) 
  (hw : grid_width = 15)
  (hh : grid_height = 5)
  (hb : triangle_base = grid_width)
  (ht : triangle_height = 3) :
  grid_width * grid_height - (1/2 * triangle_base * triangle_height) = 52.5 := by
  sorry

#check shaded_area_calculation

end shaded_area_calculation_l1817_181774


namespace simplify_product_of_square_roots_l1817_181742

theorem simplify_product_of_square_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) = 60 * x^2 * Real.sqrt 35 := by
  sorry

end simplify_product_of_square_roots_l1817_181742


namespace camping_trip_percentage_l1817_181769

theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (15 * total_students) / 100)
  (h2 : (75 * (students_more_than_100 * 100 / 25)) / 100 + students_more_than_100 = (60 * total_students) / 100) :
  (students_more_than_100 * 100 / 25) * 100 / total_students = 60 :=
sorry

end camping_trip_percentage_l1817_181769


namespace inequality_solution_l1817_181700

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -11/6 ∨ x > -4/3 :=
by sorry

end inequality_solution_l1817_181700


namespace correct_number_of_pretzels_l1817_181796

/-- The number of pretzels in Mille's snack packs. -/
def pretzels : ℕ := 64

/-- The number of kids in the class. -/
def kids : ℕ := 16

/-- The number of items in each baggie. -/
def items_per_baggie : ℕ := 22

/-- The number of suckers. -/
def suckers : ℕ := 32

/-- Theorem stating that the number of pretzels is correct given the conditions. -/
theorem correct_number_of_pretzels :
  pretzels * 5 + suckers = kids * items_per_baggie :=
by sorry

end correct_number_of_pretzels_l1817_181796


namespace smallest_sum_of_reciprocals_l1817_181719

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧ ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end smallest_sum_of_reciprocals_l1817_181719


namespace unique_solution_l1817_181770

theorem unique_solution (x y z : ℝ) : 
  x + 3 * y = 33 ∧ 
  y = 10 ∧ 
  2 * x - y + z = 15 → 
  x = 3 ∧ y = 10 ∧ z = 19 := by
sorry

end unique_solution_l1817_181770


namespace original_triangle_area_l1817_181797

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ (side : ℝ), new_area = (5 * side)^2 / 2 → original_area = side^2 / 2) →
  new_area = 200 →
  original_area = 8 :=
by sorry

end original_triangle_area_l1817_181797


namespace perpendicular_lines_sum_l1817_181748

-- Define the lines and point
def line1 (m : ℝ) (x y : ℝ) : Prop := 2 * x + m * y - 1 = 0
def line2 (n : ℝ) (x y : ℝ) : Prop := 3 * x - 2 * y + n = 0
def foot (p : ℝ) : ℝ × ℝ := (2, p)

-- State the theorem
theorem perpendicular_lines_sum (m n p : ℝ) : 
  (∀ x y, line1 m x y → line2 n x y → (x - 2) * (3 * x - 2 * y + n) + (y - p) * (2 * x + m * y - 1) = 0) →  -- perpendicularity condition
  line1 m 2 p →  -- foot satisfies line1
  line2 n 2 p →  -- foot satisfies line2
  m + n + p = -6 := by
sorry

end perpendicular_lines_sum_l1817_181748


namespace solid_volume_l1817_181706

/-- A solid with a square base and specific edge lengths -/
structure Solid where
  s : ℝ
  base_side_length : s > 0
  upper_edge_length : ℝ := 3 * s
  other_edge_length : ℝ := s

/-- The volume of the solid -/
def volume (solid : Solid) : ℝ := sorry

theorem solid_volume : 
  ∀ (solid : Solid), solid.s = 8 * Real.sqrt 2 → volume solid = 5760 := by
  sorry

end solid_volume_l1817_181706


namespace complex_square_root_l1817_181714

theorem complex_square_root (a b : ℕ+) (h : (↑a + ↑b * Complex.I) ^ 2 = 5 + 12 * Complex.I) :
  ↑a + ↑b * Complex.I = 3 + 2 * Complex.I := by
  sorry

end complex_square_root_l1817_181714


namespace line_through_points_l1817_181710

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def lies_on_line (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨2, -2⟩
  let p3 : Point := ⟨5, 7⟩
  let p4 : Point := ⟨4, 4⟩
  let p5 : Point := ⟨10, 22⟩
  let p6 : Point := ⟨-2, -12⟩
  let p7 : Point := ⟨1, -5⟩
  lies_on_line p3 p1 p2 ∧
  lies_on_line p4 p1 p2 ∧
  lies_on_line p5 p1 p2 ∧
  lies_on_line p7 p1 p2 ∧
  ¬ lies_on_line p6 p1 p2 :=
by
  sorry


end line_through_points_l1817_181710


namespace min_value_of_expression_l1817_181722

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x / Real.sqrt (1 - x)) + (y / Real.sqrt (1 - y)) ≥ Real.sqrt 2 := by
sorry

end min_value_of_expression_l1817_181722


namespace ac_plus_bd_value_l1817_181763

theorem ac_plus_bd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 10)
  (eq2 : a + b + d = -6)
  (eq3 : a + c + d = 0)
  (eq4 : b + c + d = 15) :
  a * c + b * d = -1171 / 9 := by
  sorry

end ac_plus_bd_value_l1817_181763


namespace gretchen_weekend_profit_l1817_181752

/-- Calculates Gretchen's profit from drawing caricatures over a weekend -/
def weekend_profit (
  full_body_price : ℕ)  -- Price of a full-body caricature
  (face_only_price : ℕ) -- Price of a face-only caricature
  (full_body_count : ℕ) -- Number of full-body caricatures drawn on Saturday
  (face_only_count : ℕ) -- Number of face-only caricatures drawn on Sunday
  (hourly_park_fee : ℕ) -- Hourly park fee
  (hours_per_day : ℕ)   -- Hours worked per day
  (art_supplies_cost : ℕ) -- Daily cost of art supplies
  : ℕ :=
  let total_revenue := full_body_price * full_body_count + face_only_price * face_only_count
  let total_park_fee := hourly_park_fee * hours_per_day * 2
  let total_supplies_cost := art_supplies_cost * 2
  let total_expenses := total_park_fee + total_supplies_cost
  total_revenue - total_expenses

/-- Theorem stating Gretchen's profit for the weekend -/
theorem gretchen_weekend_profit :
  weekend_profit 25 15 24 16 5 6 8 = 764 := by
  sorry

end gretchen_weekend_profit_l1817_181752


namespace complex_fraction_equality_l1817_181729

theorem complex_fraction_equality : (5 : ℂ) / (2 - I) = 2 + I := by sorry

end complex_fraction_equality_l1817_181729


namespace trains_meet_at_11am_l1817_181765

/-- The distance between stations A and B in kilometers -/
def distance_between_stations : ℝ := 155

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 20

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 25

/-- The time difference between the trains' departures in hours -/
def time_difference : ℝ := 1

/-- The meeting time of the trains after the second train's departure -/
def meeting_time : ℝ := 3

theorem trains_meet_at_11am :
  speed_train1 * (time_difference + meeting_time) +
  speed_train2 * meeting_time = distance_between_stations :=
sorry

end trains_meet_at_11am_l1817_181765


namespace circle_area_ratio_l1817_181785

/-- Given two circles X and Y where an arc of 90° on circle X has the same length as an arc of 60° on circle Y, 
    the ratio of the area of circle X to the area of circle Y is 9/4 -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) 
  (h : (π / 2) * X = (π / 3) * Y) : 
  (π * X^2) / (π * Y^2) = 9 / 4 := by
  sorry

end circle_area_ratio_l1817_181785


namespace sin_plus_sqrt3_cos_l1817_181737

theorem sin_plus_sqrt3_cos (x : ℝ) (h1 : x ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (x + π/12) = Real.sqrt 2 / 10) : 
  Real.sin x + Real.sqrt 3 * Real.cos x = 8/5 := by
  sorry

end sin_plus_sqrt3_cos_l1817_181737


namespace function_property_l1817_181787

noncomputable def f (x : ℝ) : ℝ := x * (1 - Real.log x)

theorem function_property (x₁ x₂ : ℝ) (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂) (h4 : f x₁ = f x₂) :
  x₁ + x₂ < Real.exp 1 := by
sorry

end function_property_l1817_181787


namespace lines_are_skew_l1817_181786

/-- Two lines in 3D space are skew if and only if b is not equal to -79/19 -/
theorem lines_are_skew (b : ℚ) : 
  (∀ (t u : ℚ), (2 : ℚ) + 3*t ≠ 3 + 7*u ∨ 1 + 4*t ≠ 5 + 3*u ∨ b + 5*t ≠ 2 + u) ↔ 
  b ≠ -79/19 := by
sorry

end lines_are_skew_l1817_181786


namespace sin_thirty_degrees_l1817_181749

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_thirty_degrees_l1817_181749


namespace number_of_cars_l1817_181741

theorem number_of_cars (total_distance : ℝ) (car_spacing : ℝ) (h1 : total_distance = 242) (h2 : car_spacing = 5.5) :
  ⌊total_distance / car_spacing⌋ + 1 = 45 := by
  sorry

end number_of_cars_l1817_181741


namespace smallest_among_four_l1817_181768

theorem smallest_among_four (a b c d : ℚ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end smallest_among_four_l1817_181768


namespace greg_dog_walking_earnings_l1817_181789

/-- Greg's dog walking business model and earnings calculation --/
theorem greg_dog_walking_earnings :
  let base_charge : ℕ := 20
  let per_minute_charge : ℕ := 1
  let one_dog_minutes : ℕ := 10
  let two_dogs_minutes : ℕ := 7
  let three_dogs_minutes : ℕ := 9
  let total_earnings := 
    (base_charge + per_minute_charge * one_dog_minutes) +
    2 * (base_charge + per_minute_charge * two_dogs_minutes) +
    3 * (base_charge + per_minute_charge * three_dogs_minutes)
  total_earnings = 171 := by sorry

end greg_dog_walking_earnings_l1817_181789


namespace dice_probability_l1817_181757

def num_dice : ℕ := 6
def num_sides : ℕ := 15
def num_low_sides : ℕ := 9
def num_high_sides : ℕ := 6

def prob_low : ℚ := num_low_sides / num_sides
def prob_high : ℚ := num_high_sides / num_sides

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dice_probability : 
  (choose num_dice (num_dice / 2)) * (prob_low ^ (num_dice / 2)) * (prob_high ^ (num_dice / 2)) = 4320 / 15625 := by
  sorry

end dice_probability_l1817_181757


namespace sqrt_77_consecutive_integers_product_l1817_181717

theorem sqrt_77_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 77 ∧ 
  Real.sqrt 77 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 72 := by
  sorry

end sqrt_77_consecutive_integers_product_l1817_181717


namespace factorial_ratio_l1817_181724

theorem factorial_ratio : Nat.factorial 15 / Nat.factorial 14 = 15 := by
  sorry

end factorial_ratio_l1817_181724


namespace circle_and_lines_theorem_l1817_181792

/-- Represents a circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- Represents a line y = k(x - 2) -/
structure Line where
  k : ℝ

/-- Checks if a circle satisfies the given conditions -/
def satisfiesConditions (c : Circle) : Prop :=
  c.r > 0 ∧
  2 * c.a + c.b = 0 ∧
  (2 - c.a)^2 + (-1 - c.b)^2 = c.r^2 ∧
  |c.a + c.b - 1| / Real.sqrt 2 = c.r

/-- Checks if a line divides the circle into arcs with length ratio 1:2 -/
def dividesCircle (c : Circle) (l : Line) : Prop :=
  ∃ (θ : ℝ), θ = Real.arccos ((1 - l.k * (c.a - 2) - c.b) / (c.r * Real.sqrt (1 + l.k^2))) ∧
              θ / (2 * Real.pi - θ) = 1 / 2

/-- The main theorem stating the properties of the circle and lines -/
theorem circle_and_lines_theorem (c : Circle) (l : Line) :
  satisfiesConditions c →
  dividesCircle c l →
  (c.a = 1 ∧ c.b = -2 ∧ c.r = Real.sqrt 2) ∧
  (l.k = 1 ∨ l.k = 7) := by
  sorry

end circle_and_lines_theorem_l1817_181792


namespace cube_surface_area_equal_volume_l1817_181754

/-- The surface area of a cube with volume equal to a 9x3x27 inch rectangular prism is 486 square inches. -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) : 
  l = 9 ∧ w = 3 ∧ h = 27 →
  cube_edge ^ 3 = l * w * h →
  6 * cube_edge ^ 2 = 486 := by
  sorry

end cube_surface_area_equal_volume_l1817_181754


namespace train_speed_with_stoppages_l1817_181756

/-- Given a train that travels at 400 km/h without stoppages and stops for 6 minutes per hour,
    its average speed with stoppages is 360 km/h. -/
theorem train_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 400
  let minutes_stopped_per_hour : ℝ := 6
  let minutes_per_hour : ℝ := 60
  let speed_with_stoppages : ℝ := speed_without_stoppages * (minutes_per_hour - minutes_stopped_per_hour) / minutes_per_hour
  speed_with_stoppages = 360 := by
  sorry

end train_speed_with_stoppages_l1817_181756


namespace solution_set_of_inequality_l1817_181738

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, 0 < x ∧ x < 1 ↔ f (Real.exp x) < 1 :=
by
  sorry

-- Define the properties of f
axiom f_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = (x - (Real.exp 1 - 1)) / x

axiom f_value_at_1 : f 1 = 1

axiom f_value_at_e : f (Real.exp 1) = 1

end solution_set_of_inequality_l1817_181738


namespace gcf_of_lcms_l1817_181779

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end gcf_of_lcms_l1817_181779


namespace circumscribed_circle_equation_l1817_181778

/-- The equation of a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- A triangle defined by three lines -/
structure Triangle where
  side1 : Line
  side2 : Line
  side3 : Line

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The circumscribed circle of a triangle -/
def circumscribedCircle (t : Triangle) : Circle := sorry

/-- Theorem: The circumscribed circle of the given triangle has the equation (x - 2)^2 + (y - 2)^2 = 25 -/
theorem circumscribed_circle_equation (t : Triangle) 
  (h1 : t.side1 = ⟨1, 1⟩) 
  (h2 : t.side2 = ⟨-1/2, -2⟩) 
  (h3 : t.side3 = ⟨3, -9⟩) : 
  circumscribedCircle t = ⟨2, 2, 5⟩ := by sorry

end circumscribed_circle_equation_l1817_181778


namespace probability_even_product_excluding_13_l1817_181709

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_integer (n : ℕ) : Prop :=
  4 ≤ n ∧ n ≤ 20 ∧ n ≠ 13

def count_valid_integers : ℕ := 16

def count_even_valid_integers : ℕ := 9

def count_odd_valid_integers : ℕ := 7

def total_combinations : ℕ := count_valid_integers.choose 2

def even_product_combinations : ℕ := 
  count_even_valid_integers.choose 2 + count_even_valid_integers * count_odd_valid_integers

theorem probability_even_product_excluding_13 :
  (even_product_combinations : ℚ) / total_combinations = 33 / 40 := by sorry

end probability_even_product_excluding_13_l1817_181709


namespace drug_price_reduction_l1817_181708

/-- Proves that given an initial price of 100 yuan and a final price of 81 yuan
    after two equal percentage reductions, the average percentage reduction each time is 10% -/
theorem drug_price_reduction (initial_price : ℝ) (final_price : ℝ) (reduction_percentage : ℝ) :
  initial_price = 100 →
  final_price = 81 →
  final_price = initial_price * (1 - reduction_percentage)^2 →
  reduction_percentage = 0.1 := by
sorry


end drug_price_reduction_l1817_181708


namespace fraction_simplification_l1817_181771

theorem fraction_simplification (b y : ℝ) (h : b^2 + y^3 ≠ 0) :
  (Real.sqrt (b^2 + y^3) - (y^3 - b^2) / Real.sqrt (b^2 + y^3)) / (b^2 + y^3) = 
  2 * b^2 / (b^2 + y^3)^(3/2) := by
  sorry

end fraction_simplification_l1817_181771


namespace cube_split_59_l1817_181781

/-- The number of odd terms in the split of m³ -/
def split_terms (m : ℕ) : ℕ := (m + 2) * (m - 1) / 2

/-- The nth odd number starting from 3 -/
def nth_odd (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_59 (m : ℕ) (h1 : m > 1) :
  (∃ k, k ≤ split_terms m ∧ nth_odd k = 59) → m = 8 := by
  sorry

end cube_split_59_l1817_181781


namespace unique_power_sum_l1817_181725

theorem unique_power_sum (k : ℕ) : (∃ (n t : ℕ), t ≥ 2 ∧ 3^k + 5^k = n^t) ↔ k = 1 := by
  sorry

end unique_power_sum_l1817_181725


namespace baseball_team_average_l1817_181718

theorem baseball_team_average (total_score : ℕ) (total_players : ℕ) (top_scorers : ℕ) (top_average : ℕ) (remaining_average : ℕ) : 
  total_score = 270 →
  total_players = 9 →
  top_scorers = 5 →
  top_average = 50 →
  remaining_average = 5 →
  top_scorers * top_average + (total_players - top_scorers) * remaining_average = total_score :=
by sorry

end baseball_team_average_l1817_181718


namespace mean_value_theorem_application_l1817_181767

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- State the theorem
theorem mean_value_theorem_application :
  ∃ c ∈ (Set.Ioo (-1) 2), 
    (deriv f c) = (f 2 - f (-1)) / (2 - (-1)) :=
by
  sorry

end mean_value_theorem_application_l1817_181767


namespace jennys_grade_is_95_l1817_181791

-- Define the grades as natural numbers
def jennys_grade : ℕ := sorry
def jasons_grade : ℕ := sorry
def bobs_grade : ℕ := sorry

-- State the conditions
axiom condition1 : jasons_grade = jennys_grade - 25
axiom condition2 : bobs_grade = jasons_grade / 2
axiom condition3 : bobs_grade = 35

-- Theorem to prove
theorem jennys_grade_is_95 : jennys_grade = 95 := by sorry

end jennys_grade_is_95_l1817_181791


namespace solar_panel_installation_l1817_181726

/-- The number of homes that can have solar panels fully installed given the total number of homes,
    panels required per home, and the shortage in supplied panels. -/
def homes_with_panels (total_homes : ℕ) (panels_per_home : ℕ) (panel_shortage : ℕ) : ℕ :=
  ((total_homes * panels_per_home - panel_shortage) / panels_per_home)

/-- Theorem stating that given 20 homes, each requiring 10 solar panels, and a supplier bringing
    50 panels less than required, the number of homes that can have their panels fully installed is 15. -/
theorem solar_panel_installation :
  homes_with_panels 20 10 50 = 15 := by
  sorry

#eval homes_with_panels 20 10 50

end solar_panel_installation_l1817_181726


namespace sodium_hydrogen_sulfate_effect_l1817_181712

-- Define the water ionization equilibrium
def water_ionization (temp : ℝ) : Prop :=
  temp = 25 → ∃ (K : ℝ), K > 0 ∧ ∀ (c_H2O c_H c_OH : ℝ),
    c_H * c_OH = K * c_H2O

-- Define the enthalpy change
def delta_H_positive : Prop := ∃ (ΔH : ℝ), ΔH > 0

-- Define the addition of sodium hydrogen sulfate
def add_NaHSO4 (c_H_initial c_H_final : ℝ) : Prop :=
  c_H_final > c_H_initial

-- Theorem statement
theorem sodium_hydrogen_sulfate_effect
  (h1 : water_ionization 25)
  (h2 : delta_H_positive)
  (h3 : ∃ (c_H_initial c_H_final : ℝ), add_NaHSO4 c_H_initial c_H_final) :
  ∃ (K : ℝ), K > 0 ∧
    (∀ (c_H2O c_H c_OH : ℝ), c_H * c_OH = K * c_H2O) ∧
    (∃ (c_H_initial c_H_final : ℝ), c_H_final > c_H_initial) :=
sorry

end sodium_hydrogen_sulfate_effect_l1817_181712


namespace stamp_collection_percentage_l1817_181704

theorem stamp_collection_percentage (total : ℕ) (chinese_percent : ℚ) (japanese_count : ℕ) : 
  total = 100 →
  chinese_percent = 35 / 100 →
  japanese_count = 45 →
  (total - (chinese_percent * total).floor - japanese_count) / total * 100 = 20 := by
sorry

end stamp_collection_percentage_l1817_181704


namespace intersection_condition_l1817_181782

theorem intersection_condition (m n : ℝ) : 
  let A : Set ℝ := {2, m / (2 * n)}
  let B : Set ℝ := {m, n}
  (A ∩ B : Set ℝ) = {1} → n = 1/2 := by
sorry

end intersection_condition_l1817_181782


namespace A_inverse_l1817_181753

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, 5;
    -2, 9]

-- Define the claimed inverse matrix A_inv
def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/46, -5/46;
    1/23, 2/23]

-- Theorem stating that A_inv is the inverse of A
theorem A_inverse : A⁻¹ = A_inv := by
  sorry

end A_inverse_l1817_181753


namespace sin_monotone_increasing_l1817_181762

open Real

theorem sin_monotone_increasing (t : ℝ) (h : 0 < t ∧ t < π / 6) :
  StrictMonoOn (fun x => sin (2 * x + π / 6)) (Set.Ioo (-t) t) := by
  sorry

end sin_monotone_increasing_l1817_181762


namespace money_sum_l1817_181715

/-- Given three people A, B, and C with a total amount of money, prove the sum of A and C's money. -/
theorem money_sum (total money_B_C money_C : ℕ) 
  (h1 : total = 900)
  (h2 : money_B_C = 750)
  (h3 : money_C = 250) :
  ∃ (money_A : ℕ), money_A + money_C = 400 :=
by sorry

end money_sum_l1817_181715


namespace dog_distance_l1817_181743

/-- The total distance run by a dog between two people walking towards each other -/
theorem dog_distance (total_distance : ℝ) (speed_A speed_B speed_dog : ℝ) : 
  total_distance = 100 ∧ 
  speed_A = 6 ∧ 
  speed_B = 4 ∧ 
  speed_dog = 10 → 
  (total_distance / (speed_A + speed_B)) * speed_dog = 100 :=
by sorry

end dog_distance_l1817_181743


namespace nonagon_configuration_count_l1817_181703

structure NonagonConfiguration where
  vertices : Fin 9 → Fin 11
  center : Fin 11
  midpoint : Fin 11
  all_different : ∀ i j, i ≠ j → 
    (vertices i ≠ vertices j) ∧ 
    (vertices i ≠ center) ∧ 
    (vertices i ≠ midpoint) ∧ 
    (center ≠ midpoint)
  equal_sums : ∀ i : Fin 9, 
    (vertices i : ℕ) + (midpoint : ℕ) + (center : ℕ) = 
    (vertices 0 : ℕ) + (midpoint : ℕ) + (center : ℕ)

def count_valid_configurations : ℕ := sorry

theorem nonagon_configuration_count :
  count_valid_configurations = 10321920 := by sorry

end nonagon_configuration_count_l1817_181703


namespace initial_persimmons_l1817_181720

/-- The number of persimmons eaten -/
def eaten : ℕ := 5

/-- The number of persimmons left -/
def left : ℕ := 12

/-- The initial number of persimmons -/
def initial : ℕ := eaten + left

theorem initial_persimmons : initial = 17 := by
  sorry

end initial_persimmons_l1817_181720


namespace area_of_EFGH_l1817_181727

/-- Parallelogram with vertices E, F, G, H in 2D space -/
structure Parallelogram where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  let base := |p.F.2 - p.E.2|
  let height := |p.G.1 - p.E.1|
  base * height

/-- The specific parallelogram EFGH from the problem -/
def EFGH : Parallelogram :=
  { E := (2, -3)
    F := (2, 2)
    G := (7, 9)
    H := (7, 2) }

theorem area_of_EFGH : parallelogramArea EFGH = 25 := by
  sorry

end area_of_EFGH_l1817_181727


namespace third_number_proof_l1817_181799

def digit_sum (n : ℕ) : ℕ := sorry

def has_same_remainder (a b c n : ℕ) : Prop :=
  ∃ r, a % n = r ∧ b % n = r ∧ c % n = r

theorem third_number_proof :
  ∃! x : ℕ,
    ∃ n : ℕ,
      has_same_remainder 1305 4665 x n ∧
      (∀ m : ℕ, has_same_remainder 1305 4665 x m → m ≤ n) ∧
      digit_sum n = 4 ∧
      x = 4705 :=
sorry

end third_number_proof_l1817_181799


namespace derivative_of_y_l1817_181795

noncomputable def y (x : ℝ) : ℝ := x + Real.cos x

theorem derivative_of_y (x : ℝ) : 
  deriv y x = 1 - Real.sin x := by sorry

end derivative_of_y_l1817_181795


namespace extended_quadrilateral_area_l1817_181713

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral sides
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Extended sides
  bb' : ℝ
  cc' : ℝ
  dd' : ℝ
  aa' : ℝ
  -- Area of original quadrilateral
  area : ℝ
  -- Conditions
  ab_eq : ab = 5
  bc_eq : bc = 8
  cd_eq : cd = 4
  da_eq : da = 7
  bb'_eq : bb' = 1.5 * ab
  cc'_eq : cc' = 1.5 * bc
  dd'_eq : dd' = 1.5 * cd
  aa'_eq : aa' = 1.5 * da
  area_eq : area = 20

/-- The area of the extended quadrilateral A'B'C'D' is 140 -/
theorem extended_quadrilateral_area (q : ExtendedQuadrilateral) :
  q.area + (q.bb' - q.ab) * q.ab / 2 + (q.cc' - q.bc) * q.bc / 2 +
  (q.dd' - q.cd) * q.cd / 2 + (q.aa' - q.da) * q.da / 2 = 140 := by
  sorry

end extended_quadrilateral_area_l1817_181713


namespace no_palindromes_with_two_fives_l1817_181784

def isPalindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 2000 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 ≠ (n / 10) % 10)

def hasTwoFives (n : ℕ) : Prop :=
  (n / 1000 = 5) ∨ ((n / 100) % 10 = 5) ∨ ((n / 10) % 10 = 5) ∨ (n % 10 = 5)

theorem no_palindromes_with_two_fives :
  ¬∃ n : ℕ, isPalindrome n ∧ hasTwoFives n :=
sorry

end no_palindromes_with_two_fives_l1817_181784


namespace part_one_part_two_l1817_181707

/-- A quadratic equation ax^2 + bx + c = 0 is a double root equation if one root is twice the other -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ 0 ∧ y = 2*x ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0

/-- The first part of the theorem -/
theorem part_one : is_double_root_equation 1 (-3) 2 := by sorry

/-- The second part of the theorem -/
theorem part_two :
  ∀ (a b : ℝ), is_double_root_equation a b (-6) →
  (∃ (x : ℝ), x = 2 ∧ a*x^2 + b*x - 6 = 0) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) := by sorry

end part_one_part_two_l1817_181707


namespace p_is_power_of_two_l1817_181772

def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2^k

theorem p_is_power_of_two (p : ℕ) (h1 : p > 2) (h2 : ∃! d : ℕ, Odd d ∧ (32 * p) % d = 0) :
  is_power_of_two p := by
sorry

end p_is_power_of_two_l1817_181772


namespace min_value_of_parallel_lines_l1817_181716

theorem min_value_of_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  ∀ x y : ℝ, 2 * a + 3 * b ≥ 25 :=
by sorry

end min_value_of_parallel_lines_l1817_181716


namespace unique_solution_system_l1817_181788

theorem unique_solution_system (x y z : ℝ) : 
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  1/x + 1/y + 1/z = 1/2008 →
  (x = 2008 ∧ y = 4016 ∧ z = -4016) ∨
  (x = 2008 ∧ y = -4016 ∧ z = 4016) ∨
  (x = 4016 ∧ y = 2008 ∧ z = -4016) ∨
  (x = 4016 ∧ y = -4016 ∧ z = 2008) ∨
  (x = -4016 ∧ y = 2008 ∧ z = 4016) ∨
  (x = -4016 ∧ y = 4016 ∧ z = 2008) :=
by sorry

end unique_solution_system_l1817_181788


namespace function_inequality_implies_a_range_l1817_181705

/-- An even function that is decreasing on the non-negative reals -/
def EvenDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

/-- The inequality condition from the problem -/
def InequalityCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, 0 < x → x ≤ Real.sqrt 2 → 
    f (-a * x + x^3 + 1) + f (a * x - x^3 - 1) ≥ 2 * f 1

theorem function_inequality_implies_a_range 
  (f : ℝ → ℝ) (a : ℝ) 
  (hf : EvenDecreasingFunction f) 
  (h_ineq : InequalityCondition f a) : 
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end function_inequality_implies_a_range_l1817_181705


namespace four_common_tangents_l1817_181776

-- Define the circle type
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define the function to count common tangents
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem four_common_tangents (c1 c2 : Circle) 
  (h1 : c1.radius = 3)
  (h2 : c2.radius = 5)
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 10) :
  countCommonTangents c1 c2 = 4 := by sorry

end four_common_tangents_l1817_181776


namespace product_not_divisible_by_prime_l1817_181775

theorem product_not_divisible_by_prime (p a b : ℕ) : 
  Prime p → a > 0 → b > 0 → a < p → b < p → ¬(p ∣ (a * b)) := by
  sorry

end product_not_divisible_by_prime_l1817_181775


namespace fish_market_problem_l1817_181728

theorem fish_market_problem (mackerel croaker tuna : ℕ) : 
  mackerel = 48 →
  mackerel * 11 = croaker * 6 →
  croaker * 8 = tuna →
  tuna = 704 := by
sorry

end fish_market_problem_l1817_181728


namespace rectangular_prism_volume_specific_prism_volume_l1817_181732

/-- A right rectangular prism with face areas a, b, and c has volume equal to the square root of their product. -/
theorem rectangular_prism_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ l w h : ℝ, l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = a ∧ w * h = b ∧ l * h = c ∧
  l * w * h = Real.sqrt (a * b * c) := by
  sorry

/-- The volume of a right rectangular prism with face areas 10, 14, and 35 square inches is 70 cubic inches. -/
theorem specific_prism_volume :
  ∃ l w h : ℝ, l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = 10 ∧ w * h = 14 ∧ l * h = 35 ∧
  l * w * h = 70 := by
  sorry

end rectangular_prism_volume_specific_prism_volume_l1817_181732


namespace xy_value_l1817_181723

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(6*y) = 81) :
  x * y = 644 / 9 := by
  sorry

end xy_value_l1817_181723


namespace rational_number_problems_l1817_181766

theorem rational_number_problems :
  (∀ (a b : ℚ), a * b = -2 ∧ a = 1/7 → b = -14) ∧
  (∀ (x y z : ℚ), x + y + z = -5 ∧ x = 1 ∧ y = -4 → z = -2) := by sorry

end rational_number_problems_l1817_181766


namespace halving_r_problem_l1817_181735

theorem halving_r_problem (r : ℝ) (n : ℝ) (a : ℝ) :
  a = (2 * r) ^ n →
  ((r / 2) ^ n = 0.125 * a) →
  n = 3 := by
sorry

end halving_r_problem_l1817_181735


namespace cones_from_twelve_cylinders_l1817_181746

/-- The number of cones that can be cast from a given number of cylinders -/
def cones_from_cylinders (num_cylinders : ℕ) : ℕ :=
  3 * num_cylinders

/-- The volume ratio between a cylinder and a cone with the same base and height -/
def cylinder_cone_volume_ratio : ℕ := 3

theorem cones_from_twelve_cylinders :
  cones_from_cylinders 12 = 36 :=
by sorry

end cones_from_twelve_cylinders_l1817_181746
