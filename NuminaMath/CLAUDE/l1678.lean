import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l1678_167839

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (2 - x) = 1 / (2 - x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l1678_167839


namespace NUMINAMATH_CALUDE_code_decryption_probability_l1678_167801

theorem code_decryption_probability 
  (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_code_decryption_probability_l1678_167801


namespace NUMINAMATH_CALUDE_table_height_is_130_l1678_167884

/-- The height of the table in centimeters -/
def table_height : ℝ := 130

/-- The height of the bottle in centimeters -/
def bottle_height : ℝ := sorry

/-- The height of the can in centimeters -/
def can_height : ℝ := sorry

/-- The distance from the top of the can on the floor to the top of the bottle on the table is 150 cm -/
axiom can_floor_to_bottle_table : table_height + bottle_height = can_height + 150

/-- The distance from the top of the bottle on the floor to the top of the can on the table is 110 cm -/
axiom bottle_floor_to_can_table : table_height + can_height = bottle_height + 110

theorem table_height_is_130 : table_height = 130 := by sorry

end NUMINAMATH_CALUDE_table_height_is_130_l1678_167884


namespace NUMINAMATH_CALUDE_circle_sections_theorem_l1678_167858

-- Define the circle and its sections
def Circle (r : ℝ) := { x : ℝ × ℝ | x.1^2 + x.2^2 = r^2 }

structure Section (r : ℝ) where
  area : ℝ
  perimeter : ℝ

-- Define the theorem
theorem circle_sections_theorem (r : ℝ) (h : r > 0) :
  ∃ (s1 s2 s3 : Section r),
    -- Areas are equal and sum to the circle's area
    s1.area = s2.area ∧ s2.area = s3.area ∧
    s1.area + s2.area + s3.area = π * r^2 ∧
    -- Each section's area is r²π/3
    s1.area = (π * r^2) / 3 ∧
    -- Perimeters are equal to the circle's perimeter
    s1.perimeter = s2.perimeter ∧ s2.perimeter = s3.perimeter ∧
    s1.perimeter = 2 * π * r :=
by
  sorry


end NUMINAMATH_CALUDE_circle_sections_theorem_l1678_167858


namespace NUMINAMATH_CALUDE_special_function_18_48_l1678_167827

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ+ → ℕ+) : Prop :=
  (∀ x : ℕ+, f x x = x) ∧
  (∀ x y : ℕ+, f x y = f y x) ∧
  (∀ x y : ℕ+, (x + y) * (f x y) = x * (f x (x + y)))

/-- The main theorem -/
theorem special_function_18_48 (f : ℕ+ → ℕ+ → ℕ+) (h : special_function f) :
  f 18 48 = 48 := by
  sorry

end NUMINAMATH_CALUDE_special_function_18_48_l1678_167827


namespace NUMINAMATH_CALUDE_divisibility_and_expression_l1678_167822

theorem divisibility_and_expression (k : ℕ) : 
  30^k ∣ 929260 → 3^k - k^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_expression_l1678_167822


namespace NUMINAMATH_CALUDE_intersecting_lines_equality_l1678_167876

/-- Given two linear functions y = ax + b and y = cx + d that intersect at (1, 0),
    prove that a^3 + c^2 = d^2 - b^3 -/
theorem intersecting_lines_equality (a b c d : ℝ) 
  (h1 : a * 1 + b = 0)  -- y = ax + b passes through (1, 0)
  (h2 : c * 1 + d = 0)  -- y = cx + d passes through (1, 0)
  : a^3 + c^2 = d^2 - b^3 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_equality_l1678_167876


namespace NUMINAMATH_CALUDE_all_false_if_some_false_l1678_167898

-- Define the universe of quadrilaterals
variable (Q : Type)

-- Define property A
variable (A : Q → Prop)

-- Theorem statement
theorem all_false_if_some_false :
  (¬ ∃ x : Q, A x) → ¬ (∀ x : Q, A x) := by
  sorry

end NUMINAMATH_CALUDE_all_false_if_some_false_l1678_167898


namespace NUMINAMATH_CALUDE_james_cd_purchase_total_l1678_167882

def cd_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def total_price (prices : List ℝ) (discount_rate : ℝ) : ℝ :=
  (prices.map (λ p => cd_price p discount_rate)).sum

theorem james_cd_purchase_total :
  let prices : List ℝ := [10, 10, 15, 6, 18]
  let discount_rate : ℝ := 0.1
  total_price prices discount_rate = 53.10 := by
  sorry

end NUMINAMATH_CALUDE_james_cd_purchase_total_l1678_167882


namespace NUMINAMATH_CALUDE_dot_product_range_l1678_167807

-- Define the points O and A
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)

-- Define the set of points P on the right branch of the hyperbola
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2 = 1 ∧ p.1 > 0}

-- Define the dot product of OA and OP
def dot_product (p : ℝ × ℝ) : ℝ := p.1 + p.2

-- Theorem statement
theorem dot_product_range :
  ∀ p ∈ P, dot_product p > 0 ∧ ∀ M : ℝ, ∃ q ∈ P, dot_product q > M :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l1678_167807


namespace NUMINAMATH_CALUDE_max_min_on_interval_l1678_167887

/-- A function satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0) ∧
  (f 2 = -1)

/-- Theorem stating the existence and values of maximum and minimum on [-6,6] -/
theorem max_min_on_interval (f : ℝ → ℝ) (h : f_properties f) :
  (∃ max_val : ℝ, IsGreatest {y | ∃ x ∈ Set.Icc (-6) 6, f x = y} max_val ∧ max_val = 3) ∧
  (∃ min_val : ℝ, IsLeast {y | ∃ x ∈ Set.Icc (-6) 6, f x = y} min_val ∧ min_val = -3) :=
sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l1678_167887


namespace NUMINAMATH_CALUDE_division_problem_l1678_167875

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 52)
  (h2 : quotient = 16)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1678_167875


namespace NUMINAMATH_CALUDE_seventh_term_is_64_l1678_167846

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  first_third_product : a 1 * a 3 = 4
  ninth_term : a 9 = 256

/-- The 7th term of the geometric sequence is 64 -/
theorem seventh_term_is_64 (seq : GeometricSequence) : seq.a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_64_l1678_167846


namespace NUMINAMATH_CALUDE_angle_FHP_equals_angle_BAC_l1678_167844

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define that ABC is an acute triangle
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define that BC > CA
def BC_greater_than_CA (A B C : ℝ × ℝ) : Prop := sorry

-- Define the circumcenter O
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the orthocenter H
def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the foot F of the altitude from C to AB
def altitude_foot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the point P
def point_P (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_FHP_equals_angle_BAC
  (h1 : is_acute_triangle A B C)
  (h2 : BC_greater_than_CA A B C)
  (O : ℝ × ℝ) (hO : O = circumcenter A B C)
  (H : ℝ × ℝ) (hH : H = orthocenter A B C)
  (F : ℝ × ℝ) (hF : F = altitude_foot A B C)
  (P : ℝ × ℝ) (hP : P = point_P A B C) :
  angle (F - H) (P - H) = angle (B - A) (C - A) :=
sorry

end NUMINAMATH_CALUDE_angle_FHP_equals_angle_BAC_l1678_167844


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l1678_167867

/-- Represents the height reached by a monkey with a specific climbing pattern -/
def monkey_climb_height (climb_rate : ℕ) (slip_rate : ℕ) (total_time : ℕ) : ℕ :=
  let full_cycles := total_time / 2
  let remainder := total_time % 2
  full_cycles * (climb_rate - slip_rate) + remainder * climb_rate

/-- Theorem stating that given the specific climbing pattern and time, the monkey reaches 60 meters -/
theorem monkey_climb_theorem (climb_rate slip_rate total_time : ℕ) 
  (h_climb : climb_rate = 6)
  (h_slip : slip_rate = 3)
  (h_time : total_time = 37) :
  monkey_climb_height climb_rate slip_rate total_time = 60 := by
  sorry

#eval monkey_climb_height 6 3 37

end NUMINAMATH_CALUDE_monkey_climb_theorem_l1678_167867


namespace NUMINAMATH_CALUDE_area_ADC_approx_l1678_167890

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector AD
def angleBisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry
def hasAngleBisector (t : Triangle) : Prop := sorry
def sideAB (t : Triangle) : ℝ := sorry
def sideBC (t : Triangle) : ℝ := sorry
def sideAC (t : Triangle) : ℝ := sorry

-- Define the area calculation function
def areaADC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem area_ADC_approx (t : Triangle) 
  (h1 : isRightTriangle t)
  (h2 : hasAngleBisector t)
  (h3 : sideAB t = 80)
  (h4 : ∃ x, sideBC t = x ∧ sideAC t = 2*x - 10) :
  ∃ ε > 0, |areaADC t - 949| < ε :=
sorry

end NUMINAMATH_CALUDE_area_ADC_approx_l1678_167890


namespace NUMINAMATH_CALUDE_square_of_sum_equals_81_l1678_167810

theorem square_of_sum_equals_81 (x : ℝ) (h : Real.sqrt (x + 3) = 3) : 
  (x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_equals_81_l1678_167810


namespace NUMINAMATH_CALUDE_deck_size_l1678_167886

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/6 →
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l1678_167886


namespace NUMINAMATH_CALUDE_problem_statement_l1678_167819

theorem problem_statement (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) 
  (h4 : k > 0) : k = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1678_167819


namespace NUMINAMATH_CALUDE_linda_remaining_candies_l1678_167831

-- Define the initial number of candies Linda has
def initial_candies : ℝ := 34.0

-- Define the number of candies Linda gave away
def candies_given : ℝ := 28.0

-- Define the number of candies Linda has left
def remaining_candies : ℝ := initial_candies - candies_given

-- Theorem statement
theorem linda_remaining_candies :
  remaining_candies = 6.0 := by sorry

end NUMINAMATH_CALUDE_linda_remaining_candies_l1678_167831


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_specific_m_value_diameter_circle_equation_l1678_167821

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_intersection_theorem :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 5 :=
sorry

theorem specific_m_value :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_equation x1 y1 (8/5) ∧
  circle_equation x2 y2 (8/5) ∧
  line_equation x1 y1 ∧
  line_equation x2 y2 ∧
  perpendicular x1 y1 x2 y2 →
  (8/5 : ℝ) = 8/5 :=
sorry

theorem diameter_circle_equation :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_equation x1 y1 (8/5) ∧
  circle_equation x2 y2 (8/5) ∧
  line_equation x1 y1 ∧
  line_equation x2 y2 ∧
  perpendicular x1 y1 x2 y2 →
  ∀ x y : ℝ,
  x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔
  (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_specific_m_value_diameter_circle_equation_l1678_167821


namespace NUMINAMATH_CALUDE_ships_within_visibility_range_l1678_167877

/-- Two ships traveling on perpendicular courses -/
structure Ship where
  x : ℝ
  y : ℝ
  v : ℝ

/-- The problem setup -/
def ship_problem (v : ℝ) : Prop :=
  let ship1 : Ship := ⟨20, 0, v⟩
  let ship2 : Ship := ⟨0, 15, v⟩
  ∃ t : ℝ, t ≥ 0 ∧ 
    ((20 - v * t)^2 + (15 - v * t)^2) ≤ 4^2

/-- The main theorem -/
theorem ships_within_visibility_range (v : ℝ) (h : v > 0) : 
  ship_problem v :=
sorry

end NUMINAMATH_CALUDE_ships_within_visibility_range_l1678_167877


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l1678_167854

/-- A triangle with an inscribed circle --/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- The length of the first segment on one side --/
  s1 : ℝ
  /-- The length of the second segment on one side --/
  s2 : ℝ
  /-- The length of the second side --/
  a : ℝ
  /-- The length of the third side --/
  b : ℝ
  /-- Ensure all lengths are positive --/
  r_pos : r > 0
  s1_pos : s1 > 0
  s2_pos : s2 > 0
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem about a specific triangle with an inscribed circle --/
theorem inscribed_circle_triangle_sides (t : InscribedCircleTriangle)
  (h1 : t.r = 4)
  (h2 : t.s1 = 6)
  (h3 : t.s2 = 8) :
  t.a = 13 ∧ t.b = 15 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l1678_167854


namespace NUMINAMATH_CALUDE_third_number_value_l1678_167855

theorem third_number_value : ∃ x : ℝ, 3 + 33 + x + 3.33 = 369.63 ∧ x = 330.30 := by
  sorry

end NUMINAMATH_CALUDE_third_number_value_l1678_167855


namespace NUMINAMATH_CALUDE_fancy_shape_charge_proof_l1678_167812

/-- The cost to trim up a single boxwood -/
def trim_cost : ℚ := 5

/-- The total number of boxwoods -/
def total_boxwoods : ℕ := 30

/-- The number of boxwoods to be trimmed into fancy shapes -/
def fancy_boxwoods : ℕ := 4

/-- The total charge for the job -/
def total_charge : ℚ := 210

/-- The charge for trimming a boxwood into a fancy shape -/
def fancy_shape_charge : ℚ := 15

theorem fancy_shape_charge_proof :
  fancy_shape_charge * fancy_boxwoods + trim_cost * total_boxwoods = total_charge :=
sorry

end NUMINAMATH_CALUDE_fancy_shape_charge_proof_l1678_167812


namespace NUMINAMATH_CALUDE_solution_absolute_value_equation_l1678_167880

theorem solution_absolute_value_equation (x : ℝ) : 5 * x + 2 * |x| = 3 * x → x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_absolute_value_equation_l1678_167880


namespace NUMINAMATH_CALUDE_top_layer_blocks_l1678_167851

/-- Represents a four-layer pyramid with a specific block distribution -/
structure Pyramid :=
  (top : ℕ)  -- Number of blocks in the top layer

/-- The total number of blocks in the pyramid -/
def Pyramid.total (p : Pyramid) : ℕ :=
  p.top + 3 * p.top + 9 * p.top + 27 * p.top

theorem top_layer_blocks (p : Pyramid) :
  p.total = 40 → p.top = 1 := by
  sorry

#check top_layer_blocks

end NUMINAMATH_CALUDE_top_layer_blocks_l1678_167851


namespace NUMINAMATH_CALUDE_unique_divisibility_l1678_167885

theorem unique_divisibility (n : ℕ) (hn : n > 1) :
  ∃! A : ℕ, A < n^2 ∧ A > 0 ∧ (n ∣ (n^2 / A + 1)) ∧ A = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_l1678_167885


namespace NUMINAMATH_CALUDE_max_team_size_l1678_167802

/-- A function that represents a valid selection of team numbers -/
def ValidSelection (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, x ≤ 100 ∧
  ∀ y ∈ s, ∀ z ∈ s, x ≠ y + z ∧
  ∀ y ∈ s, x ≠ 2 * y

/-- The theorem stating the maximum size of a valid selection is 50 -/
theorem max_team_size :
  (∃ s : Finset ℕ, ValidSelection s ∧ s.card = 50) ∧
  ∀ s : Finset ℕ, ValidSelection s → s.card ≤ 50 := by sorry

end NUMINAMATH_CALUDE_max_team_size_l1678_167802


namespace NUMINAMATH_CALUDE_paperclip_theorem_l1678_167899

/-- The day of the week when Jasmine first has more than 500 paperclips -/
theorem paperclip_theorem : ∃ k : ℕ, k > 0 ∧ 
  (∀ j : ℕ, j < k → 5 * 3^j ≤ 500) ∧ 
  5 * 3^k > 500 ∧
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_theorem_l1678_167899


namespace NUMINAMATH_CALUDE_equation_solution_l1678_167823

theorem equation_solution : 
  ∀ x : ℝ, |2001*x - 2001| = 2001 ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1678_167823


namespace NUMINAMATH_CALUDE_point_on_line_l1678_167891

/-- Given that point A (3, a) lies on the line 2x + y - 7 = 0, prove that a = 1 -/
theorem point_on_line (a : ℝ) : 2 * 3 + a - 7 = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1678_167891


namespace NUMINAMATH_CALUDE_sum_of_first_5n_integers_l1678_167809

theorem sum_of_first_5n_integers (n : ℕ) : 
  (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210 → 
  (5 * n * (5 * n + 1)) / 2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_5n_integers_l1678_167809


namespace NUMINAMATH_CALUDE_percent_of_self_l1678_167815

theorem percent_of_self (y : ℝ) (h1 : y > 0) (h2 : y * (y / 100) = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_self_l1678_167815


namespace NUMINAMATH_CALUDE_max_points_difference_between_adjacent_teams_l1678_167892

/-- Represents a football league with the given properties -/
structure FootballLeague where
  num_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculates the maximum points a team can achieve in the league -/
def max_points (league : FootballLeague) : Nat :=
  (league.num_teams - 1) * 2 * league.points_for_win

/-- Calculates the minimum points a team can achieve in the league -/
def min_points (league : FootballLeague) : Nat :=
  (league.num_teams - 1) * 2 * league.points_for_draw

/-- Theorem stating the maximum points difference between adjacent teams -/
theorem max_points_difference_between_adjacent_teams 
  (league : FootballLeague) 
  (h1 : league.num_teams = 12)
  (h2 : league.points_for_win = 2)
  (h3 : league.points_for_draw = 1)
  (h4 : league.points_for_loss = 0) :
  max_points league - min_points league = 24 := by
  sorry


end NUMINAMATH_CALUDE_max_points_difference_between_adjacent_teams_l1678_167892


namespace NUMINAMATH_CALUDE_point_vector_relations_l1678_167804

/-- Given points A, B, C in ℝ², and points M, N such that CM = 3CA and CN = 2CB,
    prove that M and N have specific coordinates and MN has a specific value. -/
theorem point_vector_relations (A B C M N : ℝ × ℝ) :
  A = (-2, 4) →
  B = (3, -1) →
  C = (-3, -4) →
  M - C = 3 • (A - C) →
  N - C = 2 • (B - C) →
  M = (0, 20) ∧
  N = (9, 2) ∧
  M - N = (9, -18) := by
  sorry

end NUMINAMATH_CALUDE_point_vector_relations_l1678_167804


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1678_167811

theorem consecutive_integers_square_sum : 
  ∀ (a b c : ℕ), 
    a > 0 → 
    b = a + 1 → 
    c = b + 1 → 
    a * b * c = 6 * (a + b + c) → 
    a^2 + b^2 + c^2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1678_167811


namespace NUMINAMATH_CALUDE_fraction_addition_l1678_167824

theorem fraction_addition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 / x + 2 / y = (3 * y + 2 * x) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1678_167824


namespace NUMINAMATH_CALUDE_no_sequence_exists_l1678_167818

theorem no_sequence_exists : ¬ ∃ (a : Fin 7 → ℝ), 
  (∀ i, 0 ≤ a i) ∧ 
  (a 0 = 0) ∧ 
  (a 6 = 0) ∧ 
  (∀ i ∈ Finset.range 5, a (i + 2) + a i > Real.sqrt 3 * a (i + 1)) := by
sorry

end NUMINAMATH_CALUDE_no_sequence_exists_l1678_167818


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1678_167860

theorem solution_set_inequality (x : ℝ) : x^2 - 3*x > 0 ↔ x < 0 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1678_167860


namespace NUMINAMATH_CALUDE_problem_solution_l1678_167829

def f (x : ℝ) := |x - 3| - 2
def g (x : ℝ) := -|x + 1| + 4

theorem problem_solution :
  (∀ x, f x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 6) ∧
  (∀ x, f x - g x ≥ -2) ∧
  (∀ m, (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1678_167829


namespace NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1678_167863

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1678_167863


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_main_theorem_l1678_167861

/-- Theorem: Speed of a boat in still water
Given:
- The rate of current is 4 km/hr
- The boat travels downstream for 44 minutes
- The distance travelled downstream is 33.733333333333334 km
Prove: The speed of the boat in still water is 42.09090909090909 km/hr
-/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (travel_time_minutes : ℝ) 
  (distance_downstream : ℝ) : ℝ :=
  let travel_time_hours := travel_time_minutes / 60
  let downstream_speed := (distance_downstream / travel_time_hours) - current_speed
  downstream_speed

/-- Main theorem application -/
theorem main_theorem : 
  boat_speed_in_still_water 4 44 33.733333333333334 = 42.09090909090909 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_main_theorem_l1678_167861


namespace NUMINAMATH_CALUDE_economy_relationship_l1678_167835

/-- Given an economy with product X, price P, and total cost C, prove the relationship
    between these variables and calculate specific values. -/
theorem economy_relationship (k k' : ℝ) : 
  (∀ (X P : ℝ), X * P = k) →  -- X is inversely proportional to P
  (200 : ℝ) * 10 = k →        -- When P = 10, X = 200
  (∀ (C X : ℝ), C = k' * X) → -- C is directly proportional to X
  4000 = k' * 200 →           -- When X = 200, C = 4000
  (∃ (X C : ℝ), X * 50 = k ∧ C = k' * X ∧ X = 40 ∧ C = 800) := by
sorry

end NUMINAMATH_CALUDE_economy_relationship_l1678_167835


namespace NUMINAMATH_CALUDE_system_solution_sum_of_squares_l1678_167878

theorem system_solution_sum_of_squares (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 120) :
  x^2 + y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_system_solution_sum_of_squares_l1678_167878


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1678_167806

theorem decimal_to_fraction : (0.34 : ℚ) = 17 / 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1678_167806


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1678_167894

theorem sin_2alpha_value (α : Real) (h : Real.cos (α - Real.pi / 4) = Real.sqrt 3 / 3) :
  Real.sin (2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1678_167894


namespace NUMINAMATH_CALUDE_complex_root_ratio_l1678_167800

theorem complex_root_ratio (m n : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (1 + 2 * Complex.I : ℂ) ^ 2 + m * (1 + 2 * Complex.I : ℂ) + n = 0 →
  m / n = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_root_ratio_l1678_167800


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l1678_167825

theorem no_integer_solutions_for_equation : 
  ¬ ∃ (x y : ℤ), (2 : ℝ) ^ (2 * x) - (3 : ℝ) ^ (2 * y) = 35 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l1678_167825


namespace NUMINAMATH_CALUDE_expression_equals_two_fifths_l1678_167833

theorem expression_equals_two_fifths :
  (((3^1 : ℚ) - 6 + 4^2 - 3)⁻¹ * 4) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_fifths_l1678_167833


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1678_167893

theorem floor_abs_negative_real : ⌊|(-58.6 : ℝ)|⌋ = 58 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1678_167893


namespace NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_1530_l1678_167888

/-- Calculates the total interest earned on two certificates of deposit --/
theorem total_interest_calculation (total_investment : ℝ) (rate1 rate2 : ℝ) 
  (fraction_higher_rate : ℝ) : ℝ :=
  let amount_higher_rate := total_investment * fraction_higher_rate
  let amount_lower_rate := total_investment - amount_higher_rate
  let interest_higher_rate := amount_higher_rate * rate2
  let interest_lower_rate := amount_lower_rate * rate1
  interest_higher_rate + interest_lower_rate

/-- Proves that the total interest earned is $1,530 given the problem conditions --/
theorem total_interest_is_1530 : 
  total_interest_calculation 20000 0.06 0.09 0.55 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_1530_l1678_167888


namespace NUMINAMATH_CALUDE_circle_containment_l1678_167838

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- A circle contains another circle's center if the center is inside the circle -/
def contains_center (c1 c2 : Circle) : Prop :=
  is_inside c2.center c1

theorem circle_containment (circles : Fin 6 → Circle) (O : ℝ × ℝ)
  (h : ∀ i : Fin 6, is_inside O (circles i)) :
  ∃ i j : Fin 6, i ≠ j ∧ contains_center (circles i) (circles j) := by
  sorry

end NUMINAMATH_CALUDE_circle_containment_l1678_167838


namespace NUMINAMATH_CALUDE_infinite_solutions_l1678_167832

-- Define α as the positive root of x^2 - 1989x - 1 = 0
noncomputable def α : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

-- Define the equation we want to prove holds for infinitely many n
def equation (n : ℕ) : Prop :=
  ⌊α * n + 1989 * α * ⌊α * n⌋⌋ = 1989 * n + (1989^2 + 1) * ⌊α * n⌋

-- Theorem statement
theorem infinite_solutions :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → equation n :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1678_167832


namespace NUMINAMATH_CALUDE_smallest_integer_l1678_167853

theorem smallest_integer (m n x : ℕ) : 
  m = 72 →
  x > 0 →
  Nat.gcd m n = x + 8 →
  Nat.lcm m n = x * (x + 8) →
  n ≥ 8 ∧ (∃ (y : ℕ), y > 0 ∧ y + 8 ∣ 72 ∧ y < x → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_l1678_167853


namespace NUMINAMATH_CALUDE_partition_6_3_l1678_167848

/-- Represents a partition of n into at most k parts -/
def Partition (n : ℕ) (k : ℕ) := { p : List ℕ // p.length ≤ k ∧ p.sum = n }

/-- Counts the number of partitions of n into at most k indistinguishable parts -/
def countPartitions (n : ℕ) (k : ℕ) : ℕ := sorry

theorem partition_6_3 : countPartitions 6 3 = 6 := by sorry

end NUMINAMATH_CALUDE_partition_6_3_l1678_167848


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_parallel_lines_l1678_167820

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def perpendicular_plane_plane (p1 p2 : Plane3D) : Prop :=
  sorry

theorem planes_perpendicular_from_parallel_lines
  (m n : Line3D) (α β : Plane3D)
  (h1 : parallel m n)
  (h2 : contained_in m α)
  (h3 : perpendicular_line_plane n β) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_parallel_lines_l1678_167820


namespace NUMINAMATH_CALUDE_three_hundredth_non_square_l1678_167868

/-- The count of perfect squares less than or equal to a given number -/
def countSquaresUpTo (n : ℕ) : ℕ := (n.sqrt : ℕ)

/-- The nth term of the sequence of non-square positive integers -/
def nthNonSquare (n : ℕ) : ℕ :=
  n + countSquaresUpTo n

theorem three_hundredth_non_square : nthNonSquare 300 = 318 := by
  sorry

end NUMINAMATH_CALUDE_three_hundredth_non_square_l1678_167868


namespace NUMINAMATH_CALUDE_joshua_needs_32_cents_l1678_167896

/-- Calculates the additional cents needed to purchase a pen -/
def additional_cents_needed (pen_cost : ℕ) (initial_money : ℕ) (borrowed_cents : ℕ) : ℕ :=
  pen_cost * 100 - (initial_money * 100 + borrowed_cents)

/-- Proves that Joshua needs 32 more cents to purchase the pen -/
theorem joshua_needs_32_cents : 
  additional_cents_needed 6 5 68 = 32 := by
  sorry

#eval additional_cents_needed 6 5 68

end NUMINAMATH_CALUDE_joshua_needs_32_cents_l1678_167896


namespace NUMINAMATH_CALUDE_arctan_sum_of_roots_l1678_167842

theorem arctan_sum_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ * Real.sin (3 * π / 5) + Real.cos (3 * π / 5) = 0 →
  x₂^2 - x₂ * Real.sin (3 * π / 5) + Real.cos (3 * π / 5) = 0 →
  Real.arctan x₁ + Real.arctan x₂ = π / 5 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_of_roots_l1678_167842


namespace NUMINAMATH_CALUDE_hyperbola_property_l1678_167828

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a line passing through the left focus
def line_through_left_focus (x y : ℝ) : Prop := sorry

-- Define the left branch of the hyperbola
def left_branch (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define the intersection points
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- State the theorem
theorem hyperbola_property :
  hyperbola point_M.1 point_M.2 ∧
  hyperbola point_N.1 point_N.2 ∧
  left_branch point_M.1 point_M.2 ∧
  left_branch point_N.1 point_N.2 ∧
  line_through_left_focus point_M.1 point_M.2 ∧
  line_through_left_focus point_N.1 point_N.2
  →
  abs (dist point_M right_focus) + abs (dist point_N right_focus) - abs (dist point_M point_N) = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_property_l1678_167828


namespace NUMINAMATH_CALUDE_triangle_problem_l1678_167857

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (2 * Real.sqrt 3 / 3 * b * c * Real.sin A = b^2 + c^2 - a^2) →
  (c = 5) →
  (Real.cos B = 1 / 7) →
  (A = π / 3 ∧ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1678_167857


namespace NUMINAMATH_CALUDE_average_gas_mileage_l1678_167866

/-- Calculates the average gas mileage for a trip with electric and gas cars -/
theorem average_gas_mileage 
  (total_distance : ℝ) 
  (electric_distance : ℝ) 
  (rented_distance : ℝ) 
  (electric_efficiency : ℝ) 
  (rented_efficiency : ℝ) 
  (h1 : total_distance = 400)
  (h2 : electric_distance = 300)
  (h3 : rented_distance = 100)
  (h4 : electric_efficiency = 50)
  (h5 : rented_efficiency = 25)
  (h6 : total_distance = electric_distance + rented_distance) :
  (total_distance / (electric_distance / electric_efficiency + rented_distance / rented_efficiency)) = 40 := by
  sorry

#check average_gas_mileage

end NUMINAMATH_CALUDE_average_gas_mileage_l1678_167866


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l1678_167879

/-- A function satisfying the given inequality for all real numbers x < y < z -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x < y → y < z →
    f y - ((z - y) / (z - x) * f x + (y - x) / (z - x) * f z) ≤ f ((x + z) / 2) - (f x + f z) / 2

/-- The characterization of functions satisfying the inequality -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesInequality f ↔
    ∃ a b c : ℝ, a ≤ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l1678_167879


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1678_167836

theorem unique_solution_quadratic (m : ℚ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = m + 3 * x) ↔ m = 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1678_167836


namespace NUMINAMATH_CALUDE_number_problem_l1678_167805

theorem number_problem (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1678_167805


namespace NUMINAMATH_CALUDE_laptop_tote_weight_difference_l1678_167850

/-- Represents the weights of various items in pounds -/
structure Weights where
  karens_tote : ℝ
  kevins_empty_briefcase : ℝ
  kevins_full_briefcase : ℝ
  kevins_work_papers : ℝ
  kevins_laptop : ℝ

/-- Conditions of the problem -/
def problem_conditions (w : Weights) : Prop :=
  w.karens_tote = 8 ∧
  w.karens_tote = 2 * w.kevins_empty_briefcase ∧
  w.kevins_full_briefcase = 2 * w.karens_tote ∧
  w.kevins_work_papers = (w.kevins_full_briefcase - w.kevins_empty_briefcase) / 6 ∧
  w.kevins_laptop = w.kevins_full_briefcase - w.kevins_empty_briefcase - w.kevins_work_papers

/-- The theorem to be proved -/
theorem laptop_tote_weight_difference (w : Weights) 
  (h : problem_conditions w) : w.kevins_laptop - w.karens_tote = 2 := by
  sorry

end NUMINAMATH_CALUDE_laptop_tote_weight_difference_l1678_167850


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1678_167849

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 25) → 
  k = 40 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1678_167849


namespace NUMINAMATH_CALUDE_binary_11011011_to_base4_l1678_167852

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem binary_11011011_to_base4 :
  decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true]) =
  [3, 1, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_binary_11011011_to_base4_l1678_167852


namespace NUMINAMATH_CALUDE_kira_song_memory_space_l1678_167862

/-- Calculates the total memory space occupied by downloaded songs -/
def total_memory_space (morning_songs : ℕ) (afternoon_songs : ℕ) (night_songs : ℕ) (song_size : ℕ) : ℕ :=
  (morning_songs + afternoon_songs + night_songs) * song_size

/-- Proves that the total memory space occupied by Kira's downloaded songs is 140 MB -/
theorem kira_song_memory_space :
  total_memory_space 10 15 3 5 = 140 := by
sorry

end NUMINAMATH_CALUDE_kira_song_memory_space_l1678_167862


namespace NUMINAMATH_CALUDE_binomial_expansions_l1678_167881

theorem binomial_expansions (x a b : ℝ) : 
  ((x + 1) * (x + 2) = x^2 + 3*x + 2) ∧
  ((x + 1) * (x - 2) = x^2 - x - 2) ∧
  ((x - 1) * (x + 2) = x^2 + x - 2) ∧
  ((x - 1) * (x - 2) = x^2 - 3*x + 2) ∧
  ((x + a) * (x + b) = x^2 + (a + b)*x + a*b) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansions_l1678_167881


namespace NUMINAMATH_CALUDE_sphere_in_truncated_cone_l1678_167864

/-- 
Given a sphere perfectly fitted inside a truncated right circular cone, 
if the volume of the truncated cone is three times that of the sphere, 
then the ratio of the radius of the larger base to the radius of the smaller base 
of the truncated cone is (5 + √21) / 2.
-/
theorem sphere_in_truncated_cone (R r s : ℝ) 
  (h_fit : s^2 = R * r)  -- sphere fits perfectly inside the truncated cone
  (h_volume : (π / 3) * (R^2 + R*r + r^2) * (2*s + (2*s*r)/(R-r)) - 
              (π / 3) * r^2 * ((2*s*r)/(R-r)) = 
              4 * π * s^3) :  -- volume relation
  R / r = (5 + Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_in_truncated_cone_l1678_167864


namespace NUMINAMATH_CALUDE_highest_a_divisible_by_8_first_digit_is_three_l1678_167813

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem highest_a_divisible_by_8 :
  ∃ (a : ℕ), a ≤ 9 ∧
  is_divisible_by_8 (3 * 100000 + a * 1000 + 524) ∧
  (∀ (b : ℕ), b ≤ 9 → b > a →
    ¬is_divisible_by_8 (3 * 100000 + b * 1000 + 524)) ∧
  a = 8 :=
sorry

theorem first_digit_is_three :
  ∀ (a : ℕ), a ≤ 9 →
  (3 * 100000 + a * 1000 + 524) / 100000 = 3 :=
sorry

end NUMINAMATH_CALUDE_highest_a_divisible_by_8_first_digit_is_three_l1678_167813


namespace NUMINAMATH_CALUDE_triangle_values_theorem_l1678_167837

def triangle (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c * x * y

theorem triangle_values_theorem (a b c d : ℚ) :
  (∀ x : ℚ, triangle a b c x d = x) ∧
  (triangle a b c 1 2 = 3) ∧
  (triangle a b c 2 3 = 4) ∧
  (d ≠ 0) →
  a = 5 ∧ b = 0 ∧ c = -1 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_values_theorem_l1678_167837


namespace NUMINAMATH_CALUDE_ram_weight_increase_l1678_167883

theorem ram_weight_increase (ram_initial : ℝ) (shyam_initial : ℝ) 
  (h_ratio : ram_initial / shyam_initial = 4 / 5)
  (h_total_new : ram_initial * (1 + x / 100) + shyam_initial * 1.19 = 82.8)
  (h_total_increase : ram_initial * (1 + x / 100) + shyam_initial * 1.19 = (ram_initial + shyam_initial) * 1.15)
  : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ram_weight_increase_l1678_167883


namespace NUMINAMATH_CALUDE_f_min_value_inequality_proof_l1678_167814

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f
theorem f_min_value : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 3 := by sorry

-- Theorem for the inequality
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq : a + b + c = 3) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_proof_l1678_167814


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_integers_divisible_by_three_l1678_167870

theorem sum_of_three_consecutive_even_integers_divisible_by_three (n : ℤ) (h : Even n) :
  ∃ k : ℤ, n + (n + 2) + (n + 4) = 3 * k :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_integers_divisible_by_three_l1678_167870


namespace NUMINAMATH_CALUDE_parabola_max_q_y_l1678_167830

/-- Represents a parabola of the form y = -x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The y-coordinate of point Q where the parabola intersects x = -5 -/
def q_y_coord (p : Parabola) : ℝ :=
  25 - 5 * p.b + p.c

/-- Condition that the vertex of the parabola lies on the line y = 3x + 1 -/
def vertex_on_line (p : Parabola) : Prop :=
  (4 * p.c + p.b^2) / 4 = 3 * (p.b / 2) + 1

theorem parabola_max_q_y :
  ∃ (max_y : ℝ), max_y = -47/4 ∧
  ∀ (p : Parabola), vertex_on_line p →
  q_y_coord p ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_parabola_max_q_y_l1678_167830


namespace NUMINAMATH_CALUDE_exam_score_theorem_l1678_167871

/-- Proves that given the exam conditions, the number of correctly answered questions is 34 -/
theorem exam_score_theorem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) : 
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 110 →
  ∃ (correct_answers : ℕ),
    correct_answers = 34 ∧
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score :=
by sorry

end NUMINAMATH_CALUDE_exam_score_theorem_l1678_167871


namespace NUMINAMATH_CALUDE_blue_marble_probability_l1678_167889

/-- Represents the probability of selecting a blue marble from a bag with specific conditions. -/
theorem blue_marble_probability (total : ℕ) (yellow : ℕ) (h1 : total = 60) (h2 : yellow = 20) :
  let green := yellow / 2
  let remaining := total - yellow - green
  let blue := remaining / 2
  (blue : ℚ) / total = 1/4 := by sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l1678_167889


namespace NUMINAMATH_CALUDE_special_cone_volume_l1678_167859

/-- A cone with circumscribed and inscribed spheres sharing the same center -/
structure SpecialCone where
  /-- The radius of the circumscribed sphere -/
  r_circum : ℝ
  /-- The circumscribed and inscribed spheres have the same center -/
  spheres_same_center : Bool

/-- The volume of the special cone -/
noncomputable def volume (cone : SpecialCone) : ℝ := sorry

/-- Theorem: The volume of the special cone is 3π when the radius of the circumscribed sphere is 2 -/
theorem special_cone_volume (cone : SpecialCone) 
  (h1 : cone.r_circum = 2) 
  (h2 : cone.spheres_same_center = true) : 
  volume cone = 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_special_cone_volume_l1678_167859


namespace NUMINAMATH_CALUDE_min_wins_for_playoffs_l1678_167847

theorem min_wins_for_playoffs (total_games : ℕ) (win_points loss_points min_points : ℕ) :
  total_games = 22 →
  win_points = 2 →
  loss_points = 1 →
  min_points = 36 →
  (∃ (wins : ℕ), 
    wins ≤ total_games ∧ 
    wins * win_points + (total_games - wins) * loss_points ≥ min_points ∧
    ∀ (w : ℕ), w < wins → w * win_points + (total_games - w) * loss_points < min_points) →
  14 = (min_points - total_games * loss_points) / (win_points - loss_points) := by
sorry

end NUMINAMATH_CALUDE_min_wins_for_playoffs_l1678_167847


namespace NUMINAMATH_CALUDE_remainder_theorem_l1678_167895

theorem remainder_theorem (z : ℕ) (hz : z > 0) (hz_div : 4 ∣ z) :
  (z * (2 + 4 + z) + 3) % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1678_167895


namespace NUMINAMATH_CALUDE_overlapping_strips_area_l1678_167856

theorem overlapping_strips_area (left_length right_length total_length : ℝ)
  (left_only_area right_only_area : ℝ) :
  left_length = 9 →
  right_length = 7 →
  total_length = 16 →
  left_length + right_length = total_length →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ),
    overlap_area = 13.5 ∧
    (left_only_area + overlap_area) / (right_only_area + overlap_area) = left_length / right_length :=
by sorry

end NUMINAMATH_CALUDE_overlapping_strips_area_l1678_167856


namespace NUMINAMATH_CALUDE_circular_lake_diameter_l1678_167803

/-- The diameter of a circular lake with radius 7 meters is 14 meters. -/
theorem circular_lake_diameter (radius : ℝ) (h : radius = 7) : 2 * radius = 14 := by
  sorry

end NUMINAMATH_CALUDE_circular_lake_diameter_l1678_167803


namespace NUMINAMATH_CALUDE_r_fraction_of_total_l1678_167808

/-- Given a total amount of 6000 and r having 2400, prove that the fraction of the total amount that r has is 2/5 -/
theorem r_fraction_of_total (total : ℕ) (r_amount : ℕ) 
  (h_total : total = 6000) (h_r : r_amount = 2400) : 
  (r_amount : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_r_fraction_of_total_l1678_167808


namespace NUMINAMATH_CALUDE_power_five_mod_eighteen_l1678_167826

theorem power_five_mod_eighteen : 5^100 % 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_eighteen_l1678_167826


namespace NUMINAMATH_CALUDE_zero_points_product_bound_l1678_167874

noncomputable def f (a x : ℝ) : ℝ := |Real.log x / Real.log a| - (1/2)^x

theorem zero_points_product_bound (a x₁ x₂ : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) 
  (hx₁ : f a x₁ = 0) 
  (hx₂ : f a x₂ = 0) : 
  0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_points_product_bound_l1678_167874


namespace NUMINAMATH_CALUDE_square_of_negative_square_l1678_167843

theorem square_of_negative_square (x : ℝ) : (-x^2)^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_square_l1678_167843


namespace NUMINAMATH_CALUDE_duplicate_card_ratio_l1678_167897

theorem duplicate_card_ratio :
  ∀ (total_cards duplicate_cards traded_cards new_cards : ℕ),
    total_cards = 500 →
    traded_cards = duplicate_cards / 5 →
    new_cards = 25 →
    traded_cards = new_cards →
    (duplicate_cards : ℚ) / total_cards = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_duplicate_card_ratio_l1678_167897


namespace NUMINAMATH_CALUDE_angleBMeasureApprox_l1678_167834

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : ℝ  -- Measure of angle A in degrees
  B : ℝ  -- Measure of angle B in degrees
  C : ℝ  -- Measure of angle C in degrees
  isIsosceles : B = C
  angleRelation : C = 3 * A + 10
  angleSum : A + B + C = 180

/-- The measure of angle B in the isosceles triangle -/
def angleBMeasure (triangle : IsoscelesTriangle) : ℝ := triangle.B

/-- Theorem stating the measure of angle B -/
theorem angleBMeasureApprox (triangle : IsoscelesTriangle) : 
  ∃ ε > 0, |angleBMeasure triangle - 550/7| < ε :=
sorry

end NUMINAMATH_CALUDE_angleBMeasureApprox_l1678_167834


namespace NUMINAMATH_CALUDE_calculation_proofs_l1678_167816

theorem calculation_proofs :
  (40 + (1/6 - 2/3 + 3/4) * 12 = 43) ∧
  ((-1)^2021 + |(-9)| * (2/3) + (-3) / (1/5) = -10) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l1678_167816


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1678_167840

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4}
  A ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1678_167840


namespace NUMINAMATH_CALUDE_smallest_power_of_1512_l1678_167845

theorem smallest_power_of_1512 :
  ∃ (n : ℕ), 1512 * 49 = n^3 ∧
  ∀ (x : ℕ), x > 0 ∧ x < 49 → ¬∃ (m : ℕ), ∃ (k : ℕ), 1512 * x = m^k := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_of_1512_l1678_167845


namespace NUMINAMATH_CALUDE_distance_sum_theorem_l1678_167865

theorem distance_sum_theorem (x z w : ℝ) 
  (hx : x = -1)
  (hz : z = 3.7)
  (hw : w = 9.3) :
  |z - x| + |w - x| = 15 := by
sorry

end NUMINAMATH_CALUDE_distance_sum_theorem_l1678_167865


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l1678_167817

theorem integer_less_than_sqrt_23 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l1678_167817


namespace NUMINAMATH_CALUDE_divisor_problem_l1678_167872

theorem divisor_problem (d : ℕ) (z : ℤ) 
  (h1 : d > 0)
  (h2 : ∃ k : ℤ, (z + 3) / d = k)
  (h3 : ∃ m : ℤ, z = m * d + 6) :
  d = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1678_167872


namespace NUMINAMATH_CALUDE_onions_on_shelf_l1678_167869

def remaining_onions (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

theorem onions_on_shelf : remaining_onions 98 65 = 33 := by
  sorry

end NUMINAMATH_CALUDE_onions_on_shelf_l1678_167869


namespace NUMINAMATH_CALUDE_sum_of_cubes_equality_l1678_167873

theorem sum_of_cubes_equality (a b c : ℝ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : a^3 + b^3 + c^3 = 3*a*b*c) : 
  a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equality_l1678_167873


namespace NUMINAMATH_CALUDE_one_third_percent_of_200_plus_50_l1678_167841

/-- Calculates the result of taking a percentage of a number and adding a constant to it. -/
def percentageOfPlusConstant (percentage : ℚ) (number : ℚ) (constant : ℚ) : ℚ :=
  percentage / 100 * number + constant

/-- The main theorem stating that 1/3% of 200 plus 50 is approximately 50.6667 -/
theorem one_third_percent_of_200_plus_50 :
  ∃ (result : ℚ), abs (percentageOfPlusConstant (1/3) 200 50 - result) < 0.00005 ∧ result = 50.6667 := by
  sorry

#eval percentageOfPlusConstant (1/3) 200 50

end NUMINAMATH_CALUDE_one_third_percent_of_200_plus_50_l1678_167841
