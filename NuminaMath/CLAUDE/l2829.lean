import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_l2829_282984

theorem sum_of_squares (x y : ℝ) (hx : x^2 = 8*x + y) (hy : y^2 = x + 8*y) (hxy : x ≠ y) :
  x^2 + y^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2829_282984


namespace NUMINAMATH_CALUDE_fraction_addition_l2829_282941

theorem fraction_addition : (5 / (8/13)) + (4/7) = 487/56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2829_282941


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2829_282977

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_10 : a 10 = 30)
  (h_20 : a 20 = 50) :
  ∃ b c : ℝ, ∀ n : ℕ, a n = b * n + c ∧ b = 2 ∧ c = 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2829_282977


namespace NUMINAMATH_CALUDE_quadratic_domain_range_implies_power_l2829_282937

/-- A quadratic function f(x) = x^2 - 4x + 4 + m with domain and range [2, n] -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 4 + m

/-- The theorem stating that if f has domain and range [2, n], then m^n = 8 -/
theorem quadratic_domain_range_implies_power (m n : ℝ) :
  (∀ x, x ∈ Set.Icc 2 n ↔ f m x ∈ Set.Icc 2 n) →
  m^n = 8 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_domain_range_implies_power_l2829_282937


namespace NUMINAMATH_CALUDE_triangle_shape_determination_l2829_282929

structure Triangle where
  -- Define a triangle structure
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the different sets of data
def ratioSideToAngleBisector (t : Triangle) : ℝ := sorry
def ratiosOfAngleBisectors (t : Triangle) : (ℝ × ℝ × ℝ) := sorry
def midpointsOfSides (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry
def twoSidesAndOppositeAngle (t : Triangle) : (ℝ × ℝ × ℝ) := sorry
def ratioOfTwoAngles (t : Triangle) : ℝ := sorry

-- Define what it means for a set of data to uniquely determine a triangle
def uniquelyDetermines (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → t1 = t2

theorem triangle_shape_determination :
  (¬ uniquelyDetermines ratioSideToAngleBisector) ∧
  (uniquelyDetermines ratiosOfAngleBisectors) ∧
  (¬ uniquelyDetermines midpointsOfSides) ∧
  (uniquelyDetermines twoSidesAndOppositeAngle) ∧
  (uniquelyDetermines ratioOfTwoAngles) := by sorry

end NUMINAMATH_CALUDE_triangle_shape_determination_l2829_282929


namespace NUMINAMATH_CALUDE_abs_negative_two_l2829_282914

theorem abs_negative_two : abs (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l2829_282914


namespace NUMINAMATH_CALUDE_inequality_product_sum_l2829_282947

theorem inequality_product_sum (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_inequality_product_sum_l2829_282947


namespace NUMINAMATH_CALUDE_prime_cube_difference_equation_l2829_282981

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem prime_cube_difference_equation :
  ∀ p q r : ℕ,
  is_prime p ∧ is_prime q ∧ is_prime r →
  p^3 - q^3 = 5*r →
  p = 7 ∧ q = 2 ∧ r = 67 :=
sorry

end NUMINAMATH_CALUDE_prime_cube_difference_equation_l2829_282981


namespace NUMINAMATH_CALUDE_students_enjoying_both_music_and_sports_l2829_282997

theorem students_enjoying_both_music_and_sports 
  (total : ℕ) (music : ℕ) (sports : ℕ) (neither : ℕ) : 
  total = 55 → music = 35 → sports = 45 → neither = 4 → 
  music + sports - (total - neither) = 29 := by
sorry

end NUMINAMATH_CALUDE_students_enjoying_both_music_and_sports_l2829_282997


namespace NUMINAMATH_CALUDE_hawks_score_l2829_282945

/-- The number of touchdowns scored by the Hawks -/
def num_touchdowns : ℕ := 3

/-- The number of points per touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total points scored by the Hawks -/
def total_points : ℕ := num_touchdowns * points_per_touchdown

theorem hawks_score :
  total_points = 21 :=
sorry

end NUMINAMATH_CALUDE_hawks_score_l2829_282945


namespace NUMINAMATH_CALUDE_equation_equivalence_l2829_282934

theorem equation_equivalence : ∀ x : ℝ, (x + 8) * (x - 1) = -5 ↔ x^2 + 7*x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2829_282934


namespace NUMINAMATH_CALUDE_two_sixty_billion_scientific_notation_l2829_282988

-- Define 260 billion
def two_hundred_sixty_billion : ℝ := 260000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.6 * (10 ^ 11)

-- Theorem stating that 260 billion is equal to its scientific notation
theorem two_sixty_billion_scientific_notation : 
  two_hundred_sixty_billion = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_two_sixty_billion_scientific_notation_l2829_282988


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2829_282993

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2829_282993


namespace NUMINAMATH_CALUDE_power_function_uniqueness_l2829_282951

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_uniqueness 
  (f : ℝ → ℝ) 
  (h1 : is_power_function f) 
  (h2 : f 27 = 3) : 
  ∀ x : ℝ, f x = x ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_power_function_uniqueness_l2829_282951


namespace NUMINAMATH_CALUDE_one_third_equals_six_l2829_282975

theorem one_third_equals_six (x : ℝ) : (1 / 3 : ℝ) * x = 6 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_third_equals_six_l2829_282975


namespace NUMINAMATH_CALUDE_hat_cost_l2829_282939

/-- The cost of each hat when a person has enough hats for 2 weeks and the total cost is $700 -/
theorem hat_cost (num_weeks : ℕ) (days_per_week : ℕ) (total_cost : ℕ) : 
  num_weeks = 2 → days_per_week = 7 → total_cost = 700 → 
  total_cost / (num_weeks * days_per_week) = 50 := by
  sorry

end NUMINAMATH_CALUDE_hat_cost_l2829_282939


namespace NUMINAMATH_CALUDE_isosceles_triangle_80_vertex_angle_l2829_282943

/-- An isosceles triangle with one angle of 80 degrees -/
structure IsoscelesTriangle80 where
  /-- The measure of the vertex angle in degrees -/
  vertex_angle : ℝ
  /-- The measure of one of the base angles in degrees -/
  base_angle : ℝ
  /-- The triangle is isosceles -/
  isosceles : base_angle = 180 - vertex_angle - base_angle
  /-- One angle is 80 degrees -/
  has_80_degree : vertex_angle = 80 ∨ base_angle = 80
  /-- The sum of angles is 180 degrees -/
  angle_sum : vertex_angle + 2 * base_angle = 180

/-- The vertex angle in an isosceles triangle with one 80-degree angle is either 80 or 20 degrees -/
theorem isosceles_triangle_80_vertex_angle (t : IsoscelesTriangle80) :
  t.vertex_angle = 80 ∨ t.vertex_angle = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_80_vertex_angle_l2829_282943


namespace NUMINAMATH_CALUDE_kiera_had_one_fruit_cup_l2829_282959

/-- Represents the breakfast items and their costs -/
structure Breakfast where
  muffin_cost : ℕ
  fruit_cup_cost : ℕ
  francis_muffins : ℕ
  francis_fruit_cups : ℕ
  kiera_muffins : ℕ
  total_cost : ℕ

/-- Calculates the number of fruit cups Kiera had -/
def kieras_fruit_cups (b : Breakfast) : ℕ :=
  (b.total_cost - (b.francis_muffins * b.muffin_cost + b.francis_fruit_cups * b.fruit_cup_cost + b.kiera_muffins * b.muffin_cost)) / b.fruit_cup_cost

/-- Theorem stating that Kiera had 1 fruit cup given the problem conditions -/
theorem kiera_had_one_fruit_cup (b : Breakfast) 
  (h1 : b.muffin_cost = 2)
  (h2 : b.fruit_cup_cost = 3)
  (h3 : b.francis_muffins = 2)
  (h4 : b.francis_fruit_cups = 2)
  (h5 : b.kiera_muffins = 2)
  (h6 : b.total_cost = 17) :
  kieras_fruit_cups b = 1 := by
  sorry

#eval kieras_fruit_cups { muffin_cost := 2, fruit_cup_cost := 3, francis_muffins := 2, francis_fruit_cups := 2, kiera_muffins := 2, total_cost := 17 }

end NUMINAMATH_CALUDE_kiera_had_one_fruit_cup_l2829_282959


namespace NUMINAMATH_CALUDE_projection_problem_l2829_282912

/-- Given that the projection of [2, 5] onto w is [2/5, -1/5],
    prove that the projection of [3, 2] onto w is [8/5, -4/5] -/
theorem projection_problem (w : ℝ × ℝ) :
  let v₁ : ℝ × ℝ := (2, 5)
  let v₂ : ℝ × ℝ := (3, 2)
  let proj₁ : ℝ × ℝ := (2/5, -1/5)
  (∃ (k : ℝ), w = k • proj₁) →
  (v₁ • w / (w • w)) • w = proj₁ →
  (v₂ • w / (w • w)) • w = (8/5, -4/5) := by
sorry

end NUMINAMATH_CALUDE_projection_problem_l2829_282912


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l2829_282989

theorem isosceles_triangle_proof (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : 2 * (Real.cos B) * (Real.sin A) = Real.sin C) : A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l2829_282989


namespace NUMINAMATH_CALUDE_distance_right_focus_to_line_l2829_282970

/-- The distance from the right focus of the hyperbola x²/4 - y²/5 = 1 to the line x + 2y - 8 = 0 is √5 -/
theorem distance_right_focus_to_line : ∃ (d : ℝ), d = Real.sqrt 5 ∧ 
  ∀ (x y : ℝ), 
    (x^2 / 4 - y^2 / 5 = 1) →  -- Hyperbola equation
    (x + 2*y - 8 = 0) →       -- Line equation
    d = Real.sqrt ((x - 3)^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_right_focus_to_line_l2829_282970


namespace NUMINAMATH_CALUDE_circle_properties_l2829_282900

-- Define the circle C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 6

-- Define the circle C in rectangular coordinates
def C_rect (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2

-- Theorem statement
theorem circle_properties :
  -- 1. Equivalence of polar and rectangular equations
  (∀ x y : ℝ, C_rect x y ↔ ∃ ρ θ : ℝ, C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  -- 2. Maximum value of x + y is 6
  (∀ x y : ℝ, C_rect x y → x + y ≤ 6) ∧
  -- 3. (3, 3) is on C and achieves the maximum
  C_rect 3 3 ∧ 3 + 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2829_282900


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2829_282910

/-- The volume of a cube with surface area 150 cm² is 125 cm³. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 150 → s^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2829_282910


namespace NUMINAMATH_CALUDE_student_B_visited_C_l2829_282930

structure Student :=
  (name : String)
  (visited : Finset String)

def University : Type := String

theorem student_B_visited_C (studentA studentB studentC : Student) 
  (univA univB univC : University) :
  studentA.name = "A" →
  studentB.name = "B" →
  studentC.name = "C" →
  univA = "A" →
  univB = "B" →
  univC = "C" →
  studentA.visited.card > studentB.visited.card →
  univA ∉ studentA.visited →
  univB ∉ studentB.visited →
  ∃ (u : University), u ∈ studentA.visited ∧ u ∈ studentB.visited ∧ u ∈ studentC.visited →
  univC ∈ studentB.visited :=
by sorry

end NUMINAMATH_CALUDE_student_B_visited_C_l2829_282930


namespace NUMINAMATH_CALUDE_curtis_family_children_l2829_282909

/-- Represents the Curtis family -/
structure CurtisFamily where
  mother_age : ℕ
  father_age : ℕ
  num_children : ℕ
  children_ages : Fin num_children → ℕ

/-- The average age of the family -/
def family_average_age (f : CurtisFamily) : ℚ :=
  (f.mother_age + f.father_age + (Finset.sum Finset.univ f.children_ages)) / (2 + f.num_children)

/-- The average age of the mother and children -/
def mother_children_average_age (f : CurtisFamily) : ℚ :=
  (f.mother_age + (Finset.sum Finset.univ f.children_ages)) / (1 + f.num_children)

/-- The theorem stating the number of children in the Curtis family -/
theorem curtis_family_children (f : CurtisFamily) 
  (h1 : family_average_age f = 25)
  (h2 : f.father_age = 50)
  (h3 : mother_children_average_age f = 20) : 
  f.num_children = 4 := by
  sorry


end NUMINAMATH_CALUDE_curtis_family_children_l2829_282909


namespace NUMINAMATH_CALUDE_quadratic_through_point_l2829_282932

/-- Prove that for a quadratic function y = ax² passing through the point (-1, 4), the value of a is 4. -/
theorem quadratic_through_point (a : ℝ) : (∀ x : ℝ, (a * x^2) = 4) ↔ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_point_l2829_282932


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2829_282917

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 1 > 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2829_282917


namespace NUMINAMATH_CALUDE_stream_speed_l2829_282962

/-- Given that a canoe rows upstream at 9 km/hr and downstream at 12 km/hr,
    the speed of the stream is 1.5 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 9)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2829_282962


namespace NUMINAMATH_CALUDE_smiles_cookies_leftover_l2829_282907

theorem smiles_cookies_leftover (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_smiles_cookies_leftover_l2829_282907


namespace NUMINAMATH_CALUDE_f_of_g_composition_l2829_282931

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x + 1

theorem f_of_g_composition : f (1 + g 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_composition_l2829_282931


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l2829_282960

def is_valid (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 24 = 2

theorem greatest_valid_integer : 
  is_valid 194 ∧ ∀ m : ℕ, is_valid m → m ≤ 194 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l2829_282960


namespace NUMINAMATH_CALUDE_square_of_binomial_l2829_282969

theorem square_of_binomial (b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, 9 * x^2 + 24 * x + b = (3 * x + c)^2) → b = 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2829_282969


namespace NUMINAMATH_CALUDE_hyperbola_focus_implies_m_l2829_282966

/-- The hyperbola equation -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 / m - x^2 / 9 = 1

/-- The focus of the hyperbola -/
def focus : ℝ × ℝ := (0, 5)

/-- Theorem: If F(0,5) is a focus of the hyperbola y^2/m - x^2/9 = 1, then m = 16 -/
theorem hyperbola_focus_implies_m (m : ℝ) :
  (∀ x y, hyperbola_equation x y m → (x - focus.1)^2 + (y - focus.2)^2 = (x + focus.1)^2 + (y - focus.2)^2) →
  m = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_implies_m_l2829_282966


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2829_282904

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2829_282904


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2829_282995

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 36 = 0) ↔ (m = 12 * Real.sqrt 3 ∨ m = -12 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2829_282995


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2829_282948

theorem simplify_radical_product (q : ℝ) (h : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (39 * q) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2829_282948


namespace NUMINAMATH_CALUDE_shape_is_cone_l2829_282915

/-- The shape described by ρ = c sin φ in spherical coordinates is a cone -/
theorem shape_is_cone (c : ℝ) (h : c > 0) :
  ∃ (cone : Set (ℝ × ℝ × ℝ)),
    ∀ (ρ θ φ : ℝ),
      (ρ, θ, φ) ∈ cone ↔ ρ = c * Real.sin φ ∧ ρ ≥ 0 ∧ θ ∈ Set.Icc 0 (2 * Real.pi) ∧ φ ∈ Set.Icc 0 Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shape_is_cone_l2829_282915


namespace NUMINAMATH_CALUDE_green_blue_difference_after_border_l2829_282999

/-- Represents a hexagonal figure with tiles --/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles needed for a single border layer of a hexagon --/
def single_border_tiles : ℕ := 6 * 3

/-- Calculates the number of tiles needed for a double border layer of a hexagon --/
def double_border_tiles : ℕ := single_border_tiles + 6 * 4

/-- Adds a double border of green tiles to a hexagonal figure --/
def add_double_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + double_border_tiles }

/-- The main theorem to prove --/
theorem green_blue_difference_after_border (initial_figure : HexagonalFigure)
    (h1 : initial_figure.blue_tiles = 20)
    (h2 : initial_figure.green_tiles = 10) :
    let new_figure := add_double_border initial_figure
    new_figure.green_tiles - new_figure.blue_tiles = 32 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_border_l2829_282999


namespace NUMINAMATH_CALUDE_circle_symmetry_l2829_282927

-- Define the symmetry property
def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

-- Define the equation of circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y = 0

-- Define the equation of circle C'
def circle_C' (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 10

-- Theorem statement
theorem circle_symmetry :
  (∀ x y : ℝ, circle_C x y → circle_C (symmetric_point x y).1 (symmetric_point x y).2) →
  (∀ x y : ℝ, circle_C' x y ↔ circle_C (symmetric_point x y).1 (symmetric_point x y).2) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2829_282927


namespace NUMINAMATH_CALUDE_inradius_plus_circumradius_le_height_l2829_282996

/-- An acute-angled triangle is a triangle where all angles are less than 90 degrees. -/
structure AcuteTriangle where
  /-- The greatest height of the triangle -/
  height : ℝ
  /-- The inradius of the triangle -/
  inradius : ℝ
  /-- The circumradius of the triangle -/
  circumradius : ℝ
  /-- All angles are less than 90 degrees -/
  acute : height > 0 ∧ inradius > 0 ∧ circumradius > 0

/-- For any acute-angled triangle, the sum of its inradius and circumradius
    is less than or equal to its greatest height. -/
theorem inradius_plus_circumradius_le_height (t : AcuteTriangle) :
  t.inradius + t.circumradius ≤ t.height := by
  sorry

end NUMINAMATH_CALUDE_inradius_plus_circumradius_le_height_l2829_282996


namespace NUMINAMATH_CALUDE_cab_delay_l2829_282971

/-- Proves that a cab with reduced speed arrives 15 minutes late -/
theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) : 
  usual_time = 75 → speed_ratio = 5/6 → 
  (usual_time / speed_ratio) - usual_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_cab_delay_l2829_282971


namespace NUMINAMATH_CALUDE_baseball_cost_l2829_282946

/-- The cost of a baseball given the cost of a football, total payment, and change received. -/
theorem baseball_cost (football_cost change_received total_payment : ℚ) 
  (h1 : football_cost = 9.14)
  (h2 : change_received = 4.05)
  (h3 : total_payment = 20) : 
  total_payment - change_received - football_cost = 6.81 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cost_l2829_282946


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2829_282918

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2829_282918


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2829_282926

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 12 = 54 → Nat.gcd n 12 = 8 → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2829_282926


namespace NUMINAMATH_CALUDE_function_upper_bound_l2829_282922

theorem function_upper_bound
  (a r : ℝ)
  (ha : a > 1)
  (hr : r > 1)
  (f : ℝ → ℝ)
  (hf_pos : ∀ x > 0, f x > 0)
  (hf_cond1 : ∀ x > 0, (f x)^2 ≤ a * x^r * f (x/a))
  (hf_cond2 : ∀ x > 0, x < 1/2^2000 → f x < 2^2000) :
  ∀ x > 0, f x ≤ x^r * a^(1-r) := by
sorry

end NUMINAMATH_CALUDE_function_upper_bound_l2829_282922


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_for_abs_x_leq_one_l2829_282987

theorem sufficient_but_not_necessary_condition_for_abs_x_leq_one :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x| ≤ 1) ∧
  ¬(∀ x : ℝ, |x| ≤ 1 → 0 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_for_abs_x_leq_one_l2829_282987


namespace NUMINAMATH_CALUDE_marbles_ratio_l2829_282924

/-- Proves that the ratio of marbles given to Savanna to Miriam's current marbles is 3:1 -/
theorem marbles_ratio (initial : ℕ) (current : ℕ) (brother : ℕ) (sister : ℕ) 
  (h1 : initial = 300)
  (h2 : current = 30)
  (h3 : brother = 60)
  (h4 : sister = 2 * brother) : 
  (initial - current - brother - sister) / current = 3 := by
  sorry

end NUMINAMATH_CALUDE_marbles_ratio_l2829_282924


namespace NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l2829_282964

theorem infinitely_many_primes_3_mod_4 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 4 = 3} := by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l2829_282964


namespace NUMINAMATH_CALUDE_min_value_expression_l2829_282913

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + b = a*b) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y → 1/(a-1) + 2/(b-2) ≤ 1/(x-1) + 2/(y-2) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2829_282913


namespace NUMINAMATH_CALUDE_sum_and_equality_implies_b_value_l2829_282980

theorem sum_and_equality_implies_b_value
  (a b c : ℝ)
  (sum_eq : a + b + c = 117)
  (equality : a + 8 = b - 10 ∧ b - 10 = 4 * c) :
  b = 550 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equality_implies_b_value_l2829_282980


namespace NUMINAMATH_CALUDE_minimum_value_of_f_l2829_282901

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∀ x : ℝ, x > 2 → f x ≥ f 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_l2829_282901


namespace NUMINAMATH_CALUDE_prob_one_of_each_specific_jar_l2829_282998

/-- Represents the number of marbles of each color in the jar -/
structure MarbleJar :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)

/-- Calculates the probability of drawing one red, one blue, and one yellow marble -/
def prob_one_of_each (jar : MarbleJar) : ℚ :=
  sorry

/-- The theorem statement -/
theorem prob_one_of_each_specific_jar :
  prob_one_of_each ⟨3, 8, 9⟩ = 18 / 95 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_of_each_specific_jar_l2829_282998


namespace NUMINAMATH_CALUDE_triangle_problem_l2829_282955

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_problem (ABC : Triangle) 
  (h1 : ABC.a * Real.sin ABC.A + ABC.c * Real.sin ABC.C = Real.sqrt 2 * ABC.a * Real.sin ABC.C + ABC.b * Real.sin ABC.B)
  (h2 : ABC.A = 5 * Real.pi / 12) :
  ABC.B = Real.pi / 4 ∧ ABC.a = 1 + Real.sqrt 3 ∧ ABC.c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2829_282955


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2829_282936

/-- 
Given a right circular cone with diameter 16 units and altitude 20 units, 
and an inscribed right circular cylinder with height equal to its diameter 
and coinciding axis with the cone, the radius of the cylinder is 40/9 units.
-/
theorem inscribed_cylinder_radius 
  (cone_diameter : ℝ) 
  (cone_altitude : ℝ) 
  (cylinder_radius : ℝ) :
  cone_diameter = 16 →
  cone_altitude = 20 →
  cylinder_radius * 2 = cylinder_radius * 2 →  -- Height equals diameter
  (cone_altitude - cylinder_radius * 2) / cylinder_radius = 5 / 2 →  -- Similar triangles ratio
  cylinder_radius = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2829_282936


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l2829_282968

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 5

-- Define the x-intercept
def x_intercept : ℝ := parabola 0

-- Define the y-intercepts
def y_intercepts : Set ℝ := {y | parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧
  x_intercept + b + c = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l2829_282968


namespace NUMINAMATH_CALUDE_min_value_expression_l2829_282933

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  8 * a^4 + 12 * b^4 + 40 * c^4 + 18 * d^4 + 9 / (4 * a * b * c * d) ≥ 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2829_282933


namespace NUMINAMATH_CALUDE_hidden_piece_area_l2829_282972

/-- Represents the surface areas of the 7 visible pieces of the wooden block -/
def visible_areas : List ℝ := [148, 46, 72, 28, 88, 126, 58]

/-- The total number of pieces the wooden block is cut into -/
def total_pieces : ℕ := 8

/-- Theorem: Given a wooden block cut into 8 pieces, where the surface areas of 7 pieces are known,
    and the sum of these areas is 566, the surface area of the 8th piece is 22. -/
theorem hidden_piece_area (h1 : visible_areas.length = total_pieces - 1)
                          (h2 : visible_areas.sum = 566) : 
  ∃ (hidden_area : ℝ), hidden_area = 22 ∧ 
    visible_areas.sum + hidden_area = (visible_areas.sum + hidden_area) / 2 * 2 := by
  sorry

end NUMINAMATH_CALUDE_hidden_piece_area_l2829_282972


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l2829_282957

/-- The number of cards in a standard deck of playing cards -/
def standard_deck_size : ℕ := 52

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 319

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := 7

/-- Theorem: The shopkeeper has 6 complete decks of playing cards -/
theorem shopkeeper_decks :
  (total_cards - additional_cards) / standard_deck_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l2829_282957


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l2829_282938

theorem opposite_of_negative_six : -((-6) : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l2829_282938


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l2829_282965

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l2829_282965


namespace NUMINAMATH_CALUDE_john_old_cards_l2829_282940

/-- The number of baseball cards John puts on each page of the binder -/
def cards_per_page : ℕ := 3

/-- The number of new cards John has -/
def new_cards : ℕ := 8

/-- The total number of pages John used in the binder -/
def total_pages : ℕ := 8

/-- The number of old cards John had -/
def old_cards : ℕ := total_pages * cards_per_page - new_cards

theorem john_old_cards : old_cards = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_old_cards_l2829_282940


namespace NUMINAMATH_CALUDE_sector_area_theorem_l2829_282923

/-- Given a circular sector with central angle θ and arc length l,
    prove that if θ = 2 and l = 2, then the area of the sector is 1. -/
theorem sector_area_theorem (θ l : Real) (h1 : θ = 2) (h2 : l = 2) :
  let r := l / θ
  (1 / 2) * r^2 * θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_theorem_l2829_282923


namespace NUMINAMATH_CALUDE_train_speed_l2829_282992

/-- Proves that the speed of a train is 90 km/hr, given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 225) (h2 : time = 9) :
  (length / 1000) / (time / 3600) = 90 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2829_282992


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2829_282920

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 15)
  (h2 : c + a = 18)
  (h3 : a + b = 21) :
  Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2829_282920


namespace NUMINAMATH_CALUDE_parabolic_triangle_area_l2829_282963

theorem parabolic_triangle_area (n : ℕ) : 
  ∃ (a b : ℤ) (m : ℕ), 
    Odd m ∧ 
    (a * (b^2 - a^2) : ℤ) = (2^n * m)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabolic_triangle_area_l2829_282963


namespace NUMINAMATH_CALUDE_david_rosy_age_difference_l2829_282903

/-- David and Rosy's ages problem -/
theorem david_rosy_age_difference :
  ∀ (david_age rosy_age : ℕ),
    rosy_age = 12 →
    david_age + 6 = 2 * (rosy_age + 6) →
    david_age - rosy_age = 18 :=
by sorry

end NUMINAMATH_CALUDE_david_rosy_age_difference_l2829_282903


namespace NUMINAMATH_CALUDE_classroom_contribution_prove_classroom_contribution_l2829_282954

/-- Proves that the amount contributed by each of the eight families is $10 --/
theorem classroom_contribution : ℝ → Prop :=
  fun x =>
    let goal : ℝ := 200
    let raised_from_two : ℝ := 2 * 20
    let raised_from_ten : ℝ := 10 * 5
    let raised_from_eight : ℝ := 8 * x
    let total_raised : ℝ := raised_from_two + raised_from_ten + raised_from_eight
    let remaining : ℝ := 30
    total_raised + remaining = goal → x = 10

/-- Proof of the classroom_contribution theorem --/
theorem prove_classroom_contribution : classroom_contribution 10 := by
  sorry

end NUMINAMATH_CALUDE_classroom_contribution_prove_classroom_contribution_l2829_282954


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2829_282961

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 2

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  (num_Ba : ℝ) * atomic_weight_Ba + 
  (num_O : ℝ) * atomic_weight_O + 
  (num_H : ℝ) * atomic_weight_H

theorem compound_molecular_weight : 
  molecular_weight = 171.35 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2829_282961


namespace NUMINAMATH_CALUDE_decreasing_direct_proportion_negative_k_l2829_282979

/-- A direct proportion function y = kx where y decreases as x increases -/
structure DecreasingDirectProportion where
  k : ℝ
  decreasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → k * x₁ > k * x₂

/-- Theorem: If y = kx is a decreasing direct proportion function, then k < 0 -/
theorem decreasing_direct_proportion_negative_k (f : DecreasingDirectProportion) : f.k < 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_direct_proportion_negative_k_l2829_282979


namespace NUMINAMATH_CALUDE_common_chord_circle_through_AB_center_on_line_smallest_circle_through_AB_l2829_282967

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the line y = -x
def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

-- Theorem for the common chord
theorem common_chord : ∀ x y : ℝ, C₁ x y ∧ C₂ x y → x - 2*y + 4 = 0 :=
sorry

-- Theorem for the circle passing through A and B with center on y = -x
theorem circle_through_AB_center_on_line : ∃ h k : ℝ, 
  line_y_eq_neg_x h k ∧ 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 10 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
sorry

-- Theorem for the circle with smallest area passing through A and B
theorem smallest_circle_through_AB : ∀ x y : ℝ,
  (x + 2)^2 + (y - 1)^2 = 5 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) :=
sorry

end NUMINAMATH_CALUDE_common_chord_circle_through_AB_center_on_line_smallest_circle_through_AB_l2829_282967


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_eq_one_l2829_282985

/-- Given two real numbers x and y satisfying specific equations, prove that cos(x + 2y) = 1 -/
theorem cos_x_plus_2y_eq_one (x y : ℝ) 
  (hx : x^3 + Real.cos x + x - 2 = 0)
  (hy : 8 * y^3 - 2 * (Real.cos y)^2 + 2 * y + 3 = 0) :
  Real.cos (x + 2 * y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_eq_one_l2829_282985


namespace NUMINAMATH_CALUDE_only_B_in_fourth_quadrant_l2829_282973

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (1, -1)
def point_C : ℝ × ℝ := (-2, 1)
def point_D : ℝ × ℝ := (-2, -1)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem only_B_in_fourth_quadrant :
  in_fourth_quadrant point_B ∧
  ¬in_fourth_quadrant point_A ∧
  ¬in_fourth_quadrant point_C ∧
  ¬in_fourth_quadrant point_D :=
by sorry

end NUMINAMATH_CALUDE_only_B_in_fourth_quadrant_l2829_282973


namespace NUMINAMATH_CALUDE_investment_interest_rate_exists_and_unique_l2829_282944

theorem investment_interest_rate_exists_and_unique :
  ∃! r : ℝ, 
    r > 0 ∧ 
    6000 * (1 + r)^10 = 24000 ∧ 
    6000 * (1 + r)^15 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_exists_and_unique_l2829_282944


namespace NUMINAMATH_CALUDE_square_circle_radius_l2829_282902

theorem square_circle_radius (square_perimeter : ℝ) (circle_radius : ℝ) : 
  square_perimeter = 28 →
  circle_radius = square_perimeter / 4 →
  circle_radius = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_radius_l2829_282902


namespace NUMINAMATH_CALUDE_distribute_5_4_l2829_282983

/-- The number of ways to distribute n distinct objects into k indistinguishable containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 4 indistinguishable containers,
    allowing empty containers, is 51. -/
theorem distribute_5_4 : distribute 5 4 = 51 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l2829_282983


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_three_equals_three_l2829_282978

theorem absolute_value_of_negative_three_equals_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_three_equals_three_l2829_282978


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l2829_282953

theorem not_p_sufficient_not_necessary_for_q :
  ∃ (x : ℝ), (x > 1 → 1 / x < 1) ∧ (1 / x < 1 → ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l2829_282953


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2829_282942

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 3 = 0) → (x₂^2 + 5*x₂ - 3 = 0) → (x₁ + x₂ = -5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2829_282942


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l2829_282974

theorem drum_capacity_ratio :
  ∀ (C_X C_Y : ℝ),
  C_X > 0 → C_Y > 0 →
  (1/2 * C_X) + (1/4 * C_Y) = 1/2 * C_Y →
  C_Y / C_X = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l2829_282974


namespace NUMINAMATH_CALUDE_workshop_problem_l2829_282921

theorem workshop_problem :
  ∃ (x y : ℕ),
    x ≥ 1 ∧ y ≥ 1 ∧
    6 + 11 * (x - 1) = 7 + 10 * (y - 1) ∧
    100 ≤ 6 + 11 * (x - 1) ∧
    6 + 11 * (x - 1) ≤ 200 ∧
    x = 12 ∧ y = 13 :=
by sorry

end NUMINAMATH_CALUDE_workshop_problem_l2829_282921


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2829_282990

theorem power_tower_mod_500 : 2^(2^(2^2)) ≡ 36 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2829_282990


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2829_282956

theorem complex_magnitude_equation (t : ℝ) : 
  t > 0 ∧ Complex.abs (-7 + t * Complex.I) = 15 → t = 4 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2829_282956


namespace NUMINAMATH_CALUDE_integral_root_iff_odd_l2829_282928

theorem integral_root_iff_odd (n : ℕ) :
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_integral_root_iff_odd_l2829_282928


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l2829_282952

theorem cos_angle_between_vectors (a b : ℝ × ℝ) :
  a = (-2, 1) →
  a + 2 • b = (2, 3) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l2829_282952


namespace NUMINAMATH_CALUDE_sin_160_equals_sin_20_l2829_282905

theorem sin_160_equals_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_160_equals_sin_20_l2829_282905


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l2829_282991

theorem rectangle_area_diagonal_relation :
  ∀ (length width diagonal : ℝ),
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 5 / 2 →
  length ^ 2 + width ^ 2 = diagonal ^ 2 →
  diagonal = 13 →
  ∃ (k : ℝ), length * width = k * diagonal ^ 2 ∧ k = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l2829_282991


namespace NUMINAMATH_CALUDE_ship_supplies_l2829_282994

/-- Calculates the remaining supplies on a ship given initial amount and usage rates --/
theorem ship_supplies (initial_supply : ℚ) (first_day_usage : ℚ) (next_days_usage : ℚ) :
  initial_supply = 400 ∧ 
  first_day_usage = 2/5 ∧ 
  next_days_usage = 3/5 →
  initial_supply * (1 - first_day_usage) * (1 - next_days_usage) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ship_supplies_l2829_282994


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l2829_282950

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (z_eq_3x : z = 3 * x)
  (ordered : x ≤ y ∧ y ≤ z)
  (max_triple : z ≤ 3 * x) : 
  ∃ (min_prod : ℝ), min_prod = 9 / 343 ∧ x * y * z ≥ min_prod :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l2829_282950


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2829_282906

/-- Represents the number of people in the club after k years -/
def club_size (k : ℕ) : ℕ :=
  match k with
  | 0 => 8
  | n + 1 => 2 * club_size n - 2

/-- Theorem stating that the club size after 4 years is 98 -/
theorem club_size_after_four_years :
  club_size 4 = 98 := by
  sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l2829_282906


namespace NUMINAMATH_CALUDE_parabola_theorem_l2829_282916

/-- Represents a parabola in the form y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The parabola passes through the point (2, 0) -/
def passes_through_A (p : Parabola) : Prop :=
  4 + 2 * p.b + p.c = 0

/-- The parabola passes through the point (0, 6) -/
def passes_through_B (p : Parabola) : Prop :=
  p.c = 6

/-- The parabola equation is y = x^2 - 5x + 6 -/
def is_correct_equation (p : Parabola) : Prop :=
  p.b = -5 ∧ p.c = 6

/-- The y-coordinate of the point (4, 0) on the parabola -/
def y_at_x_4 (p : Parabola) : ℝ :=
  16 - 5 * 4 + p.c

/-- The downward shift required for the parabola to pass through (4, 0) -/
def downward_shift (p : Parabola) : ℝ :=
  y_at_x_4 p

theorem parabola_theorem (p : Parabola) 
  (h1 : passes_through_A p) (h2 : passes_through_B p) : 
  is_correct_equation p ∧ downward_shift p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_theorem_l2829_282916


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l2829_282976

/-- Proves that John's candy store spending is $0.88 given his allowance and spending pattern -/
theorem johns_candy_store_spending (allowance : ℚ) : 
  allowance = 33/10 →
  let arcade_spending := 3/5 * allowance
  let remaining_after_arcade := allowance - arcade_spending
  let toy_store_spending := 1/3 * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 88/100 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l2829_282976


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l2829_282911

/-- Calculates the size of each serving of orange juice in ounces. -/
def serving_size (concentrate_cans : ℕ) (water_ratio : ℕ) (concentrate_oz : ℕ) (total_servings : ℕ) : ℚ :=
  let total_cans := concentrate_cans * (water_ratio + 1)
  let total_oz := total_cans * concentrate_oz
  (total_oz : ℚ) / total_servings

/-- Proves that the size of each serving is 6 ounces under the given conditions. -/
theorem orange_juice_serving_size :
  serving_size 34 3 12 272 = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l2829_282911


namespace NUMINAMATH_CALUDE_miran_has_fewest_paper_l2829_282949

def miran_paper : ℕ := 6
def junga_paper : ℕ := 13
def minsu_paper : ℕ := 10

theorem miran_has_fewest_paper :
  miran_paper ≤ junga_paper ∧ miran_paper ≤ minsu_paper :=
sorry

end NUMINAMATH_CALUDE_miran_has_fewest_paper_l2829_282949


namespace NUMINAMATH_CALUDE_annette_sara_weight_difference_l2829_282935

/-- Given the weights of combinations of people, prove that Annette weighs 8 pounds more than Sara. -/
theorem annette_sara_weight_difference 
  (annette caitlin sara bob : ℝ) 
  (h1 : annette + caitlin = 95)
  (h2 : caitlin + sara = 87)
  (h3 : annette + sara = 97)
  (h4 : caitlin + bob = 100)
  (h5 : annette + caitlin + bob = 155) :
  annette - sara = 8 := by
sorry

end NUMINAMATH_CALUDE_annette_sara_weight_difference_l2829_282935


namespace NUMINAMATH_CALUDE_pyramid_volume_change_l2829_282908

/-- Given a pyramid with rectangular base and volume 60 cubic feet, 
    prove that tripling its length, doubling its width, and increasing its height by 20% 
    results in a new volume of 432 cubic feet. -/
theorem pyramid_volume_change (V : ℝ) (l w h : ℝ) : 
  V = 60 → 
  V = (1/3) * l * w * h → 
  (1/3) * (3*l) * (2*w) * (1.2*h) = 432 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_change_l2829_282908


namespace NUMINAMATH_CALUDE_wolf_chase_deer_l2829_282958

theorem wolf_chase_deer (t : ℕ) : t ≤ 28 ↔ ∀ (x y : ℝ), x > 0 → y > 0 → x * y > 0.78 * x * y * (1 + t / 100) := by
  sorry

end NUMINAMATH_CALUDE_wolf_chase_deer_l2829_282958


namespace NUMINAMATH_CALUDE_vessel_width_calculation_l2829_282986

/-- Proves that given a cube with edge length 15 cm immersed in a rectangular vessel 
    with base length 20 cm, if the water level rises by 11.25 cm, 
    then the width of the vessel's base is 15 cm. -/
theorem vessel_width_calculation (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) :
  cube_edge = 15 →
  vessel_length = 20 →
  water_rise = 11.25 →
  (cube_edge ^ 3) = (vessel_length * (cube_edge ^ 3 / (vessel_length * water_rise))) * water_rise →
  cube_edge ^ 3 / (vessel_length * water_rise) = 15 := by
  sorry

#check vessel_width_calculation

end NUMINAMATH_CALUDE_vessel_width_calculation_l2829_282986


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2829_282982

/-- Represents a number with an integer part and a repeating decimal part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The given repeating decimal 5.341341341... -/
def givenNumber : RepeatingDecimal :=
  { integerPart := 5, repeatingPart := 341 }

/-- Theorem stating that 5.341341341... equals 5336/999 -/
theorem repeating_decimal_equals_fraction :
  toRational givenNumber = 5336 / 999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2829_282982


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l2829_282919

theorem smaller_circle_radius (r_large : ℝ) (r_small : ℝ) : 
  r_large = 4 →
  π * r_small^2 = (1/2) * π * r_large^2 →
  (π * r_small^2) + (π * r_large^2 - π * r_small^2) = 2 * (π * r_large^2 - π * r_small^2) →
  r_small = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l2829_282919


namespace NUMINAMATH_CALUDE_quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square_l2829_282925

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_bisecting_diagonals (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square :
  ∃ q : Quadrilateral, 
    has_perpendicular_diagonals q ∧ 
    has_bisecting_diagonals q ∧ 
    ¬ is_square q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square_l2829_282925
