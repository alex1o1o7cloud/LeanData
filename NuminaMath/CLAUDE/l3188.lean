import Mathlib

namespace min_value_reciprocal_sum_l3188_318860

/-- Given a line 2ax + by - 2 = 0 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 1/a + 1/b is 4 -/
theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b = 2 → (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 2 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 := by sorry

end min_value_reciprocal_sum_l3188_318860


namespace area_of_median_triangle_l3188_318848

/-- Given a triangle ABC with area S, the area of a triangle whose sides are equal to the medians of ABC is 3/4 * S -/
theorem area_of_median_triangle (A B C : ℝ × ℝ) (S : ℝ) : 
  let triangle_area := S
  let median_triangle_area := (3/4 : ℝ) * S
  triangle_area = S → median_triangle_area = (3/4 : ℝ) * triangle_area := by
sorry

end area_of_median_triangle_l3188_318848


namespace diagonal_FH_range_l3188_318857

/-- Represents a quadrilateral with integer side lengths and diagonals -/
structure Quadrilateral where
  EF : ℕ
  FG : ℕ
  GH : ℕ
  HE : ℕ
  EH : ℕ
  FH : ℕ

/-- The specific quadrilateral from the problem -/
def specificQuad : Quadrilateral where
  EF := 7
  FG := 13
  GH := 7
  HE := 20
  EH := 0  -- We don't know the exact value, but it's an integer
  FH := 0  -- This is what we're trying to prove

theorem diagonal_FH_range (q : Quadrilateral) (h : q = specificQuad) : 
  14 ≤ q.FH ∧ q.FH ≤ 19 := by
  sorry

#check diagonal_FH_range

end diagonal_FH_range_l3188_318857


namespace intersecting_lines_sum_l3188_318854

/-- Two lines intersect at a point -/
structure IntersectingLines where
  m : ℝ
  b : ℝ
  intersect_x : ℝ
  intersect_y : ℝ
  eq1 : intersect_y = m * intersect_x + 2
  eq2 : intersect_y = -2 * intersect_x + b

/-- Theorem: For two lines y = mx + 2 and y = -2x + b intersecting at (4, 12), b + m = 22.5 -/
theorem intersecting_lines_sum (lines : IntersectingLines)
    (h1 : lines.intersect_x = 4)
    (h2 : lines.intersect_y = 12) :
    lines.b + lines.m = 22.5 := by
  sorry

end intersecting_lines_sum_l3188_318854


namespace power_ratio_approximation_l3188_318855

theorem power_ratio_approximation :
  let ratio := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  ratio = 101 / 20 ∧ 
  ∀ n : ℤ, |ratio - 5| ≤ |ratio - n| := by
  sorry

end power_ratio_approximation_l3188_318855


namespace cos_right_angle_l3188_318803

theorem cos_right_angle (D E F : ℝ) (h1 : D = 90) (h2 : E = 9) (h3 : F = 40) : Real.cos D = 0 := by
  sorry

end cos_right_angle_l3188_318803


namespace sports_club_overlap_l3188_318802

theorem sports_club_overlap (N B T BT Neither : ℕ) : 
  N = 35 →
  B = 15 →
  T = 18 →
  Neither = 5 →
  B + T - BT = N - Neither →
  BT = 3 :=
by sorry

end sports_club_overlap_l3188_318802


namespace cubic_factorization_l3188_318826

theorem cubic_factorization (a : ℝ) : a^3 - 4*a^2 + 4*a = a*(a-2)^2 := by
  sorry

end cubic_factorization_l3188_318826


namespace three_disjoint_edges_exist_l3188_318827

/-- A graph with 6 vertices where each vertex has degree at least 3 -/
structure SixVertexGraph where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_symmetry : ∀ (u v : Fin 6), (u, v) ∈ edges → (v, u) ∈ edges
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 3

/-- A set of 3 disjoint edges that cover all vertices -/
def ThreeDisjointEdges (G : SixVertexGraph) : Prop :=
  ∃ (e₁ e₂ e₃ : Fin 6 × Fin 6),
    e₁ ∈ G.edges ∧ e₂ ∈ G.edges ∧ e₃ ∈ G.edges ∧
    e₁.1 ≠ e₁.2 ∧ e₂.1 ≠ e₂.2 ∧ e₃.1 ≠ e₃.2 ∧
    e₁.1 ≠ e₂.1 ∧ e₁.1 ≠ e₂.2 ∧ e₁.1 ≠ e₃.1 ∧ e₁.1 ≠ e₃.2 ∧
    e₁.2 ≠ e₂.1 ∧ e₁.2 ≠ e₂.2 ∧ e₁.2 ≠ e₃.1 ∧ e₁.2 ≠ e₃.2 ∧
    e₂.1 ≠ e₃.1 ∧ e₂.1 ≠ e₃.2 ∧ e₂.2 ≠ e₃.1 ∧ e₂.2 ≠ e₃.2

/-- Theorem: In a graph with 6 vertices where each vertex has degree at least 3,
    there exists a set of 3 disjoint edges that cover all vertices -/
theorem three_disjoint_edges_exist (G : SixVertexGraph) : ThreeDisjointEdges G :=
sorry

end three_disjoint_edges_exist_l3188_318827


namespace circle_triangle_area_ratio_l3188_318886

/-- For a right triangle circumscribed about a circle -/
theorem circle_triangle_area_ratio
  (h a b R : ℝ)
  (h_positive : h > 0)
  (R_positive : R > 0)
  (right_triangle : a^2 + b^2 = h^2)
  (circumradius : R = h / 2) :
  π * R^2 / (a * b / 2) = π * h / (4 * R) :=
by sorry

end circle_triangle_area_ratio_l3188_318886


namespace equation_condition_l3188_318879

theorem equation_condition (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  (11 * a + b) * (11 * a + c) = 121 * a * (a + 1) + 11 * b * c ↔ b + c = 11 :=
sorry

end equation_condition_l3188_318879


namespace exams_left_to_grade_l3188_318830

theorem exams_left_to_grade (total_exams : ℕ) (monday_percent : ℚ) (tuesday_percent : ℚ)
  (h1 : total_exams = 120)
  (h2 : monday_percent = 60 / 100)
  (h3 : tuesday_percent = 75 / 100) :
  total_exams - (monday_percent * total_exams).floor - (tuesday_percent * (total_exams - (monday_percent * total_exams).floor)).floor = 12 :=
by
  sorry

end exams_left_to_grade_l3188_318830


namespace complex_modulus_l3188_318887

theorem complex_modulus (a b : ℝ) (z : ℂ) :
  (a + Complex.I)^2 = b * Complex.I →
  z = a + b * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end complex_modulus_l3188_318887


namespace rectangle_area_problem_l3188_318805

theorem rectangle_area_problem (p q : ℝ) : 
  q = (2/5) * p →  -- point (p, q) is on the line y = 2/5 x
  p * q = 90 →     -- area of the rectangle is 90
  p = 15 :=        -- prove that p = 15
by sorry

end rectangle_area_problem_l3188_318805


namespace square_tiles_count_l3188_318890

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular
| 1 => 4  -- square
| 2 => 5  -- pentagonal
| _ => 0  -- unreachable

/-- Proves that given 30 tiles with 108 edges in total, there are 6 square tiles -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30) 
  (h_total_edges : total_edges = 108) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3 * t + 4 * s + 5 * p = total_edges ∧ 
    s = 6 :=
by
  sorry

#check square_tiles_count

end square_tiles_count_l3188_318890


namespace vector_sum_length_one_l3188_318851

theorem vector_sum_length_one (x : Real) :
  let a := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
  let b := (Real.cos (x / 2), -Real.sin (x / 2))
  (0 ≤ x) ∧ (x ≤ Real.pi) →
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1 →
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 := by
sorry

end vector_sum_length_one_l3188_318851


namespace item_value_proof_l3188_318828

def import_tax_rate : ℝ := 0.07
def tax_threshold : ℝ := 1000
def tax_paid : ℝ := 87.50

theorem item_value_proof (total_value : ℝ) : 
  total_value = 2250 := by
  sorry

end item_value_proof_l3188_318828


namespace family_brownie_consumption_percentage_l3188_318849

theorem family_brownie_consumption_percentage
  (total_brownies : ℕ)
  (children_consumption_percentage : ℚ)
  (lorraine_extra_consumption : ℕ)
  (leftover_brownies : ℕ)
  (h1 : total_brownies = 16)
  (h2 : children_consumption_percentage = 1/4)
  (h3 : lorraine_extra_consumption = 1)
  (h4 : leftover_brownies = 5) :
  let remaining_after_children := total_brownies - (children_consumption_percentage * total_brownies).num
  let family_consumption := remaining_after_children - leftover_brownies - lorraine_extra_consumption
  (family_consumption : ℚ) / remaining_after_children = 1/2 :=
sorry

end family_brownie_consumption_percentage_l3188_318849


namespace collinear_vectors_x_equals_three_l3188_318863

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a and b, prove that if they are collinear, then x = 3 -/
theorem collinear_vectors_x_equals_three (x : ℝ) :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 := by
  sorry


end collinear_vectors_x_equals_three_l3188_318863


namespace line_tangent_to_parabola_l3188_318835

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
   ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 1 := by sorry

end line_tangent_to_parabola_l3188_318835


namespace cows_ran_away_theorem_l3188_318853

/-- Represents the number of cows that ran away from a farm --/
def cows_that_ran_away (initial_cows : ℕ) (initial_days : ℕ) (days_passed : ℕ) (remaining_cows : ℕ) : ℕ :=
  initial_cows - remaining_cows

/-- Theorem stating the number of cows that ran away under given conditions --/
theorem cows_ran_away_theorem (initial_cows : ℕ) (initial_days : ℕ) (days_passed : ℕ) :
  initial_cows = 1000 →
  initial_days = 50 →
  days_passed = 10 →
  (initial_cows * initial_days - initial_cows * days_passed) = 
    (initial_cows - cows_that_ran_away initial_cows initial_days days_passed (initial_cows - 200)) * initial_days →
  cows_that_ran_away initial_cows initial_days days_passed (initial_cows - 200) = 200 :=
by
  sorry

#eval cows_that_ran_away 1000 50 10 800

end cows_ran_away_theorem_l3188_318853


namespace sunflower_plants_count_l3188_318804

/-- The number of corn plants -/
def corn_plants : ℕ := 81

/-- The number of tomato plants -/
def tomato_plants : ℕ := 63

/-- The maximum number of plants in one row -/
def max_plants_per_row : ℕ := 9

/-- The number of rows for corn plants -/
def corn_rows : ℕ := corn_plants / max_plants_per_row

/-- The number of rows for tomato plants -/
def tomato_rows : ℕ := tomato_plants / max_plants_per_row

/-- The number of rows for sunflower plants -/
def sunflower_rows : ℕ := max corn_rows tomato_rows

/-- The theorem stating the number of sunflower plants -/
theorem sunflower_plants_count : 
  ∃ (sunflower_plants : ℕ), 
    sunflower_plants = sunflower_rows * max_plants_per_row ∧ 
    sunflower_plants = 81 :=
by sorry

end sunflower_plants_count_l3188_318804


namespace traditionalist_fraction_l3188_318811

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) 
  (total_progressives : ℚ) :
  num_provinces = 4 →
  num_traditionalists_per_province = total_progressives / 12 →
  (num_provinces : ℚ) * num_traditionalists_per_province / 
    (total_progressives + (num_provinces : ℚ) * num_traditionalists_per_province) = 1/4 := by
  sorry

end traditionalist_fraction_l3188_318811


namespace expression_value_l3188_318844

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 2 * y = 1 := by
  sorry

end expression_value_l3188_318844


namespace min_value_a_plus_9b_l3188_318834

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_arith_seq : (1/a + 1/b) / 2 = 1/2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (1/x + 1/y) / 2 = 1/2 → x + 9*y ≥ 16 ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (1/x₀ + 1/y₀) / 2 = 1/2 ∧ x₀ + 9*y₀ = 16 :=
by sorry

#check min_value_a_plus_9b

end min_value_a_plus_9b_l3188_318834


namespace star_property_l3188_318876

def star (m n : ℝ) : ℝ := (3 * m - 2 * n)^2

theorem star_property (x y : ℝ) : star ((3 * x - 2 * y)^2) ((2 * y - 3 * x)^2) = (3 * x - 2 * y)^4 := by
  sorry

end star_property_l3188_318876


namespace survey_preference_theorem_l3188_318897

theorem survey_preference_theorem (total_students : ℕ) 
                                  (mac_preference : ℕ) 
                                  (no_preference : ℕ) 
                                  (h1 : total_students = 350)
                                  (h2 : mac_preference = 100)
                                  (h3 : no_preference = 140) : 
  total_students - mac_preference - (mac_preference / 5) - no_preference = 90 := by
  sorry

#check survey_preference_theorem

end survey_preference_theorem_l3188_318897


namespace vector_c_coordinates_l3188_318822

def a : Fin 3 → ℝ := ![0, 1, -1]
def b : Fin 3 → ℝ := ![1, 2, 3]
def c : Fin 3 → ℝ := λ i => 3 * a i - b i

theorem vector_c_coordinates :
  c = ![-1, 1, -6] := by sorry

end vector_c_coordinates_l3188_318822


namespace quadratic_roots_relation_l3188_318896

theorem quadratic_roots_relation (a b : ℝ) (r₁ r₂ : ℂ) : 
  (∀ x : ℂ, x^2 + a*x + b = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ x : ℂ, x^2 + b*x + a = 0 ↔ x = 3*r₁ ∨ x = 3*r₂) →
  a/b = -3 := by
sorry

end quadratic_roots_relation_l3188_318896


namespace expression_evaluation_l3188_318838

theorem expression_evaluation : (3^2 - 3 + 1) - (4^2 - 4 + 1) + (5^2 - 5 + 1) - (6^2 - 6 + 1) = -16 := by
  sorry

end expression_evaluation_l3188_318838


namespace order_of_numbers_l3188_318864

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_of_numbers : x < y ∧ y < z ∧ z < w := by sorry

end order_of_numbers_l3188_318864


namespace appended_digit_problem_l3188_318846

theorem appended_digit_problem (x y : ℕ) : 
  x > 0 → y < 10 → (10 * x + y) - x^2 = 8 * x → 
  ((x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8)) := by sorry

end appended_digit_problem_l3188_318846


namespace cubic_roots_sum_l3188_318882

theorem cubic_roots_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 8 = 0 →
  s^3 - 15*s^2 + 13*s - 8 = 0 →
  t^3 - 15*t^2 + 13*t - 8 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 199/9 := by
sorry

end cubic_roots_sum_l3188_318882


namespace chess_club_boys_count_l3188_318868

theorem chess_club_boys_count :
  ∀ (G B : ℕ),
  G + B = 30 →
  (2 * G) / 3 + (3 * B) / 4 = 18 →
  B = 24 :=
by
  sorry

end chess_club_boys_count_l3188_318868


namespace survey_results_l3188_318833

theorem survey_results (total : ℕ) (believe_percent : ℚ) (not_believe_percent : ℚ) 
  (h_total : total = 1240)
  (h_believe : believe_percent = 46/100)
  (h_not_believe : not_believe_percent = 31/100)
  (h_rounding : ∀ x : ℚ, 0 ≤ x → x < 1 → ⌊x * total⌋ + 1 = ⌈x * total⌉) :
  let min_believers := ⌈(believe_percent - 1/200) * total⌉
  let min_non_believers := ⌈(not_believe_percent - 1/200) * total⌉
  let max_refusals := total - min_believers - min_non_believers
  min_believers = 565 ∧ max_refusals = 296 := by
  sorry

#check survey_results

end survey_results_l3188_318833


namespace fourth_number_nth_row_l3188_318810

/-- The kth number in the triangular array -/
def triangular_array (k : ℕ) : ℕ := 2^(k - 1)

/-- The position of the 4th number from left to right in the nth row -/
def fourth_number_position (n : ℕ) : ℕ := n * (n - 1) / 2 + 4

theorem fourth_number_nth_row (n : ℕ) (h : n ≥ 4) :
  triangular_array (fourth_number_position n) = 2^((n^2 - n + 6) / 2) :=
sorry

end fourth_number_nth_row_l3188_318810


namespace square_perimeter_ratio_l3188_318806

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_area_ratio : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 := by
  sorry

end square_perimeter_ratio_l3188_318806


namespace student_multiplication_problem_l3188_318869

theorem student_multiplication_problem (x : ℝ) (y : ℝ) 
  (h1 : x = 129)
  (h2 : x * y - 148 = 110) : 
  y = 2 := by
  sorry

end student_multiplication_problem_l3188_318869


namespace range_x_when_m_is_one_range_m_for_not_p_sufficient_not_necessary_l3188_318875

-- Define propositions p and q
def p (x m : ℝ) : Prop := |2*x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3*x) / (x + 2) > 0

-- Theorem for part (I)
theorem range_x_when_m_is_one :
  ∃ a b : ℝ, a = -2 ∧ b = 0 ∧
  ∀ x : ℝ, a < x ∧ x ≤ b ↔ p x 1 ∧ q x :=
sorry

-- Theorem for part (II)
theorem range_m_for_not_p_sufficient_not_necessary :
  ∃ a b : ℝ, a = -3 ∧ b = -1/3 ∧
  ∀ m : ℝ, a ≤ m ∧ m ≤ b ↔
    (∀ x : ℝ, ¬(p x m) → q x) ∧
    ¬(∀ x : ℝ, q x → ¬(p x m)) :=
sorry

end range_x_when_m_is_one_range_m_for_not_p_sufficient_not_necessary_l3188_318875


namespace exponent_calculation_l3188_318819

theorem exponent_calculation : (8^5 / 8^2) * 4^4 = 2^17 := by
  sorry

end exponent_calculation_l3188_318819


namespace probability_d_divides_z_l3188_318836

def D : Finset Nat := Finset.filter (λ x => 100 % x = 0) (Finset.range 101)
def Z : Finset Nat := Finset.range 101

theorem probability_d_divides_z : 
  (Finset.sum D (λ d => (Finset.filter (λ z => z % d = 0) Z).card)) / (D.card * Z.card) = 217 / 900 := by
  sorry

end probability_d_divides_z_l3188_318836


namespace largest_x_sqrt_3x_eq_5x_l3188_318829

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) → 
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) ∧
  (Real.sqrt (3 * (3/25)) = 5 * (3/25)) := by
sorry

end largest_x_sqrt_3x_eq_5x_l3188_318829


namespace min_value_implies_a_l3188_318841

theorem min_value_implies_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = Real.sin x ^ 2 - 2 * a * Real.sin x + 1) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ 1/2) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1/2) →
  a = Real.sqrt 2 / 2 :=
by sorry

end min_value_implies_a_l3188_318841


namespace alice_acorn_price_l3188_318817

/-- The price Alice paid for each acorn -/
def alice_price_per_acorn (alice_acorns : ℕ) (alice_bob_price_ratio : ℕ) (bob_total_price : ℕ) : ℚ :=
  (alice_bob_price_ratio * bob_total_price : ℚ) / alice_acorns

/-- Proof that Alice paid $15 for each acorn -/
theorem alice_acorn_price :
  alice_price_per_acorn 3600 9 6000 = 15 := by
  sorry

#eval alice_price_per_acorn 3600 9 6000

end alice_acorn_price_l3188_318817


namespace monster_feeding_interval_l3188_318862

/-- Represents the monster's feeding pattern over 300 years -/
structure MonsterFeedingPattern where
  interval : ℕ  -- The interval at which the monster rises
  total_consumed : ℕ  -- Total number of people consumed over 300 years
  first_ship : ℕ  -- Number of people on the first ship

/-- Theorem stating the conditions and the conclusion about the monster's feeding interval -/
theorem monster_feeding_interval (m : MonsterFeedingPattern) : 
  m.total_consumed = 847 ∧ 
  m.first_ship = 121 ∧ 
  m.total_consumed = m.first_ship + 2 * m.first_ship + 4 * m.first_ship → 
  m.interval = 100 := by
  sorry

end monster_feeding_interval_l3188_318862


namespace system_solution_exists_l3188_318847

theorem system_solution_exists (m : ℝ) : 
  m ≠ 3 → ∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9 := by
  sorry

end system_solution_exists_l3188_318847


namespace tree_planting_theorem_l3188_318898

/-- The number of trees planted by Class 2-5 -/
def trees_2_5 : ℕ := 142

/-- The difference in trees planted between Class 2-5 and Class 2-3 -/
def difference : ℕ := 18

/-- The number of trees planted by Class 2-3 -/
def trees_2_3 : ℕ := trees_2_5 - difference

/-- The total number of trees planted by both classes -/
def total_trees : ℕ := trees_2_5 + trees_2_3

theorem tree_planting_theorem :
  trees_2_3 = 124 ∧ total_trees = 266 :=
by sorry

end tree_planting_theorem_l3188_318898


namespace harper_mineral_water_cost_l3188_318895

/-- The amount Harper spends on mineral water for 240 days -/
def mineral_water_cost (daily_consumption : ℚ) (bottles_per_case : ℕ) (case_cost : ℚ) (total_days : ℕ) : ℚ :=
  let cases_needed := (total_days : ℚ) * daily_consumption / bottles_per_case
  cases_needed.ceil * case_cost

/-- Theorem stating the cost of mineral water for Harper -/
theorem harper_mineral_water_cost :
  mineral_water_cost (1/2) 24 12 240 = 60 := by
  sorry

end harper_mineral_water_cost_l3188_318895


namespace lunch_break_is_60_minutes_l3188_318889

/-- Represents the painting rates and work done on each day -/
structure PaintingData where
  paula_rate : ℝ
  helpers_rate : ℝ
  day1_hours : ℝ
  day1_work : ℝ
  day2_hours : ℝ
  day2_work : ℝ
  day3_hours : ℝ
  day3_work : ℝ

/-- The lunch break duration in hours -/
def lunch_break : ℝ := 1

/-- Theorem stating that the lunch break is 60 minutes given the painting data -/
theorem lunch_break_is_60_minutes (data : PaintingData) : 
  (data.day1_hours - lunch_break) * (data.paula_rate + data.helpers_rate) = data.day1_work ∧
  (data.day2_hours - lunch_break) * data.helpers_rate = data.day2_work ∧
  (data.day3_hours - lunch_break) * data.paula_rate = data.day3_work →
  lunch_break * 60 = 60 := by
  sorry

#eval lunch_break * 60  -- Should output 60

end lunch_break_is_60_minutes_l3188_318889


namespace complex_number_equation_l3188_318813

theorem complex_number_equation (z : ℂ) : (z * Complex.I = Complex.I + z) → z = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end complex_number_equation_l3188_318813


namespace project_completion_time_l3188_318893

theorem project_completion_time (a b c : ℝ) 
  (h1 : a + b = 1/2)   -- A and B together complete in 2 days
  (h2 : b + c = 1/4)   -- B and C together complete in 4 days
  (h3 : c + a = 1/2.4) -- C and A together complete in 2.4 days
  : 1/a = 3 :=         -- A alone completes in 3 days
by
  sorry

end project_completion_time_l3188_318893


namespace stream_speed_l3188_318814

/-- Prove that the speed of a stream is 3.75 km/h given the boat's travel times and distances -/
theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 100)
  (h2 : downstream_time = 8)
  (h3 : upstream_distance = 75)
  (h4 : upstream_time = 15) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 3.75 := by
  sorry

end stream_speed_l3188_318814


namespace arithmetic_sequence_a8_l3188_318871

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 4) 
  (h_a4 : a 4 = 2) : 
  a 8 = -2 := by
sorry

end arithmetic_sequence_a8_l3188_318871


namespace greatest_fraction_l3188_318883

theorem greatest_fraction : 
  let f1 := (3 : ℚ) / 10
  let f2 := (4 : ℚ) / 7
  let f3 := (5 : ℚ) / 23
  let f4 := (2 : ℚ) / 3
  let f5 := (1 : ℚ) / 2
  f4 > f1 ∧ f4 > f2 ∧ f4 > f3 ∧ f4 > f5 := by sorry

end greatest_fraction_l3188_318883


namespace cone_properties_l3188_318861

/-- Properties of a cone with specific dimensions -/
theorem cone_properties (r h l : ℝ) : 
  r = 2 → -- base radius is 2
  π * l = 2 * π * r → -- lateral surface unfolds to a semicircle
  l^2 = r^2 + h^2 → -- Pythagorean theorem
  (l = 4 ∧ (1/3) * π * r^2 * h = (8 * Real.sqrt 3 / 3) * π) := by sorry

end cone_properties_l3188_318861


namespace least_positive_integer_divisible_by_four_primes_l3188_318865

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m)) ∧
  n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_primes_l3188_318865


namespace quadratic_properties_l3188_318867

def f (x : ℝ) := x^2 - 2*x - 1

theorem quadratic_properties :
  (∃ (x y : ℝ), (x, y) = (1, -2) ∧ ∀ t, f t ≥ f x) ∧
  (∃ (x₁ x₂ : ℝ), x₁ = 1 + Real.sqrt 2 ∧ 
                  x₂ = 1 - Real.sqrt 2 ∧ 
                  f x₁ = 0 ∧ 
                  f x₂ = 0 ∧
                  ∀ x, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_properties_l3188_318867


namespace k_range_given_one_integer_solution_l3188_318831

/-- The inequality system has only one integer solution -/
def has_one_integer_solution (k : ℝ) : Prop :=
  ∃! (x : ℤ), (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k+7)*x + 7*k < 0)

/-- The range of k -/
def k_range (k : ℝ) : Prop :=
  (k ≥ -5 ∧ k < 3) ∨ (k > 4 ∧ k ≤ 5)

/-- Theorem stating the range of k given the conditions -/
theorem k_range_given_one_integer_solution :
  ∀ k : ℝ, has_one_integer_solution k ↔ k_range k :=
sorry

end k_range_given_one_integer_solution_l3188_318831


namespace unicorn_tether_problem_l3188_318850

theorem unicorn_tether_problem (rope_length : ℝ) (tower_radius : ℝ) (unicorn_height : ℝ) 
  (rope_end_distance : ℝ) (p q r : ℕ) (h_rope_length : rope_length = 25)
  (h_tower_radius : tower_radius = 10) (h_unicorn_height : unicorn_height = 5)
  (h_rope_end_distance : rope_end_distance = 5) (h_r_prime : Nat.Prime r)
  (h_rope_tower_length : (p - Real.sqrt q) / r = 
    rope_length - Real.sqrt ((rope_end_distance + tower_radius)^2 + unicorn_height^2)) :
  p + q + r = 1128 := by
  sorry

end unicorn_tether_problem_l3188_318850


namespace soda_consumption_l3188_318823

theorem soda_consumption (carol_soda bob_soda : ℝ) 
  (h1 : carol_soda = 20)
  (h2 : bob_soda = carol_soda * 1.25)
  (h3 : carol_soda ≥ 0)
  (h4 : bob_soda ≥ 0) :
  ∃ (transfer : ℝ),
    0 ≤ transfer ∧
    transfer ≤ bob_soda * 0.2 ∧
    carol_soda * 0.8 + transfer = bob_soda * 0.8 - transfer ∧
    carol_soda * 0.8 + transfer + (bob_soda * 0.8 - transfer) = 36 :=
by sorry

end soda_consumption_l3188_318823


namespace count_prime_base_n_l3188_318899

/-- Represents the number 10001 in base n -/
def base_n (n : ℕ) : ℕ := n^4 + 1

/-- Counts the number of positive integers n ≥ 2 for which 10001_n is prime -/
theorem count_prime_base_n : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (base_n n) := by
  sorry

end count_prime_base_n_l3188_318899


namespace potato_bag_weight_l3188_318842

theorem potato_bag_weight (weight : ℝ) (fraction : ℝ) : 
  weight = 36 → weight / fraction = 36 → fraction = 1 := by sorry

end potato_bag_weight_l3188_318842


namespace baker_cakes_left_l3188_318891

theorem baker_cakes_left (total_cakes sold_cakes : ℕ) 
  (h1 : total_cakes = 54)
  (h2 : sold_cakes = 41) :
  total_cakes - sold_cakes = 13 := by
  sorry

end baker_cakes_left_l3188_318891


namespace sequence_a_general_term_l3188_318870

/-- Sequence a_n with sum S_n satisfying the given conditions -/
def sequence_a (n : ℕ) : ℚ := sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := sorry

/-- The main theorem to prove -/
theorem sequence_a_general_term :
  ∀ n : ℕ, n > 0 →
  (2 * S n - n * sequence_a n = n) ∧
  (sequence_a 2 = 3) →
  sequence_a n = 2 * n - 1 := by sorry

end sequence_a_general_term_l3188_318870


namespace philip_paintings_l3188_318866

/-- Calculates the total number of paintings Philip will have after a given number of days -/
def total_paintings (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + paintings_per_day * days

/-- Theorem: Philip will have 80 paintings after 30 days -/
theorem philip_paintings :
  total_paintings 2 20 30 = 80 := by
  sorry

end philip_paintings_l3188_318866


namespace total_pens_equals_sum_l3188_318843

/-- The number of pens given to friends -/
def pens_given : ℕ := 22

/-- The number of pens kept for herself -/
def pens_kept : ℕ := 34

/-- The total number of pens bought by her parents -/
def total_pens : ℕ := pens_given + pens_kept

/-- Theorem stating that the total number of pens is the sum of pens given and pens kept -/
theorem total_pens_equals_sum : total_pens = pens_given + pens_kept := by sorry

end total_pens_equals_sum_l3188_318843


namespace evaluate_expression_l3188_318808

theorem evaluate_expression (a : ℝ) (h : a = 2) : (5 * a^2 - 13 * a + 4) * (2 * a - 3) = -2 := by
  sorry

end evaluate_expression_l3188_318808


namespace brooke_jacks_eight_days_l3188_318877

/-- Represents the number of jumping jacks Sidney does on a given day -/
def sidney_jacks : Nat → Nat
  | 0 => 20  -- Monday
  | 1 => 36  -- Tuesday
  | n + 2 => sidney_jacks (n + 1) + (16 + 2 * n)  -- Following days

/-- The total number of jumping jacks Sidney does over 8 days -/
def sidney_total : Nat := (List.range 8).map sidney_jacks |>.sum

/-- The number of jumping jacks Brooke does is four times Sidney's -/
def brooke_total : Nat := 4 * sidney_total

theorem brooke_jacks_eight_days : brooke_total = 2880 := by
  sorry

end brooke_jacks_eight_days_l3188_318877


namespace sqrt_product_sqrt_main_theorem_l3188_318837

theorem sqrt_product_sqrt (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * Real.sqrt b) = Real.sqrt a * Real.sqrt (Real.sqrt b) :=
by sorry

theorem main_theorem : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by sorry

end sqrt_product_sqrt_main_theorem_l3188_318837


namespace bethany_riding_time_l3188_318878

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- Represents the number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Represents Bethany's riding schedule -/
structure RidingSchedule where
  monday : ℕ     -- minutes ridden on Monday
  wednesday : ℕ  -- minutes ridden on Wednesday
  friday : ℕ     -- minutes ridden on Friday
  tuesday : ℕ    -- minutes ridden on Tuesday
  thursday : ℕ   -- minutes ridden on Thursday
  saturday : ℕ   -- minutes ridden on Saturday

/-- Calculates the total minutes ridden in two weeks -/
def total_minutes (schedule : RidingSchedule) : ℕ :=
  2 * (schedule.monday + schedule.wednesday + schedule.friday + 
       schedule.tuesday + schedule.thursday + schedule.saturday)

/-- Theorem stating Bethany's riding time on Monday, Wednesday, and Friday -/
theorem bethany_riding_time (schedule : RidingSchedule) 
  (h1 : schedule.tuesday = 30)
  (h2 : schedule.thursday = 30)
  (h3 : schedule.saturday = 2 * minutes_in_hour)
  (h4 : total_minutes schedule = 12 * minutes_in_hour) :
  2 * (schedule.monday + schedule.wednesday + schedule.friday) = 6 * minutes_in_hour := by
  sorry

#check bethany_riding_time

end bethany_riding_time_l3188_318878


namespace jolene_raised_180_l3188_318821

/-- Represents Jolene's fundraising activities --/
structure JoleneFundraising where
  num_babysitting_families : ℕ
  babysitting_rate : ℕ
  num_cars_washed : ℕ
  car_wash_rate : ℕ

/-- Calculates the total amount Jolene raised --/
def total_raised (j : JoleneFundraising) : ℕ :=
  j.num_babysitting_families * j.babysitting_rate + j.num_cars_washed * j.car_wash_rate

/-- Theorem stating that Jolene raised $180 --/
theorem jolene_raised_180 :
  ∃ j : JoleneFundraising,
    j.num_babysitting_families = 4 ∧
    j.babysitting_rate = 30 ∧
    j.num_cars_washed = 5 ∧
    j.car_wash_rate = 12 ∧
    total_raised j = 180 :=
  sorry

end jolene_raised_180_l3188_318821


namespace cubic_root_reciprocal_squares_sum_l3188_318888

theorem cubic_root_reciprocal_squares_sum (p q : ℂ) (z₁ z₂ z₃ : ℂ) : 
  z₁^3 + p*z₁ + q = 0 → 
  z₂^3 + p*z₂ + q = 0 → 
  z₃^3 + p*z₃ + q = 0 → 
  z₁ ≠ z₂ → z₂ ≠ z₃ → z₃ ≠ z₁ →
  q ≠ 0 →
  1/z₁^2 + 1/z₂^2 + 1/z₃^2 = p^2 / q^2 := by
sorry

end cubic_root_reciprocal_squares_sum_l3188_318888


namespace arithmetic_sequence_sum_l3188_318809

/-- Given an arithmetic sequence {a_n} where a_3 and a_15 are the roots of x^2 - 6x + 8 = 0,
    the sum a_7 + a_8 + a_9 + a_10 + a_11 is equal to 15. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 3)^2 - 6 * (a 3) + 8 = 0 →  -- a_3 is a root
  (a 15)^2 - 6 * (a 15) + 8 = 0 →  -- a_15 is a root
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
sorry

end arithmetic_sequence_sum_l3188_318809


namespace rectangular_prism_surface_area_l3188_318856

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The surface area of a rectangular prism with dimensions 10, 6, and 5 is 280 -/
theorem rectangular_prism_surface_area :
  surface_area 10 6 5 = 280 := by
  sorry

end rectangular_prism_surface_area_l3188_318856


namespace henry_walk_distance_l3188_318801

/-- Represents a 2D point --/
structure Point where
  x : Float
  y : Float

/-- Calculates the distance between two points --/
def distance (p1 p2 : Point) : Float :=
  Float.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Converts meters to feet --/
def metersToFeet (meters : Float) : Float :=
  meters * 3.281

theorem henry_walk_distance : 
  let start := Point.mk 0 0
  let end_point := Point.mk 40 (-(metersToFeet 15 + 48))
  Float.abs (distance start end_point - 105.1) < 0.1 := by
  sorry


end henry_walk_distance_l3188_318801


namespace parallel_vectors_x_value_l3188_318839

/-- Given two parallel vectors a = (2, 5) and b = (x, -2), prove that x = -4/5 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 5]
  let b : Fin 2 → ℝ := ![x, -2]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  x = -4/5 := by
  sorry

end parallel_vectors_x_value_l3188_318839


namespace beatrice_gilbert_ratio_l3188_318815

/-- The number of crayons in each person's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ

/-- The conditions of the problem -/
def problem_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = boxes.gilbert ∧
  boxes.gilbert = 4 * boxes.judah ∧
  boxes.karen = 128 ∧
  boxes.judah = 8

/-- The theorem stating that Beatrice and Gilbert have the same number of crayons -/
theorem beatrice_gilbert_ratio (boxes : CrayonBoxes) 
  (h : problem_conditions boxes) : boxes.beatrice = boxes.gilbert := by
  sorry

#check beatrice_gilbert_ratio

end beatrice_gilbert_ratio_l3188_318815


namespace f_is_even_and_increasing_l3188_318894

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l3188_318894


namespace mean_temperature_l3188_318852

def temperatures : List ℝ := [82, 84, 86, 88, 90, 92, 84, 85]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 86.375 := by
  sorry

end mean_temperature_l3188_318852


namespace total_books_is_182_l3188_318873

/-- The number of books each person has -/
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45
def kim_books : ℕ := 14
def alex_books : ℕ := 48

/-- The total number of books -/
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books + kim_books + alex_books

/-- Theorem stating that the total number of books is 182 -/
theorem total_books_is_182 : total_books = 182 := by
  sorry

end total_books_is_182_l3188_318873


namespace pascal_triangle_51_row_5th_number_l3188_318832

theorem pascal_triangle_51_row_5th_number : 
  let n : ℕ := 51  -- number of elements in the row
  let k : ℕ := 4   -- index of the number we're looking for (0-based)
  Nat.choose (n - 1) k = 220500 := by
sorry

end pascal_triangle_51_row_5th_number_l3188_318832


namespace cookie_count_l3188_318807

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end cookie_count_l3188_318807


namespace cubic_minus_linear_factorization_l3188_318845

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l3188_318845


namespace worker_d_rate_l3188_318884

-- Define work rates for workers a, b, c, and d
variable (A B C D : ℚ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 15
def condition2 : Prop := A + B + C = 1 / 12
def condition3 : Prop := C + D = 1 / 20

-- Theorem statement
theorem worker_d_rate 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) 
  (h3 : condition3 C D) : 
  D = 1 / 30 := by sorry

end worker_d_rate_l3188_318884


namespace geometric_sequence_problem_l3188_318816

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  is_geometric a →
  (a 3)^2 + 7*(a 3) + 9 = 0 →
  (a 7)^2 + 7*(a 7) + 9 = 0 →
  (a 5 = 3 ∨ a 5 = -3) :=
by sorry

end geometric_sequence_problem_l3188_318816


namespace inequality_proof_l3188_318881

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sqrt ((b + c) / (2 * a + b + c)) + 
  Real.sqrt ((c + a) / (2 * b + c + a)) + 
  Real.sqrt ((a + b) / (2 * c + a + b)) ≤ 1 + 2 / Real.sqrt 3 := by
  sorry

end inequality_proof_l3188_318881


namespace vector_product_l3188_318885

/-- Given vectors a and b, if |a| = 2 and a ⊥ b, then mn = -6 -/
theorem vector_product (m n : ℝ) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (2, n)
  (a.1^2 + a.2^2 = 4) → -- |a| = 2
  (a.1 * b.1 + a.2 * b.2 = 0) → -- a ⊥ b
  m * n = -6 := by
sorry

end vector_product_l3188_318885


namespace negative_integer_solution_l3188_318880

theorem negative_integer_solution : ∃! (N : ℤ), N < 0 ∧ N + 2 * N^2 = 12 ∧ N = -3 := by
  sorry

end negative_integer_solution_l3188_318880


namespace vanaspati_percentage_l3188_318800

/-- Proves that the percentage of vanaspati in the original ghee mixture is 40% -/
theorem vanaspati_percentage
  (original_quantity : ℝ)
  (pure_ghee_percentage : ℝ)
  (added_pure_ghee : ℝ)
  (new_vanaspati_percentage : ℝ)
  (h1 : original_quantity = 10)
  (h2 : pure_ghee_percentage = 0.6)
  (h3 : added_pure_ghee = 10)
  (h4 : new_vanaspati_percentage = 0.2)
  (h5 : (1 - pure_ghee_percentage) * original_quantity = 
        new_vanaspati_percentage * (original_quantity + added_pure_ghee)) :
  (1 - pure_ghee_percentage) * 100 = 40 := by
sorry

end vanaspati_percentage_l3188_318800


namespace f_strictly_decreasing_on_interval_l3188_318818

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Theorem statement
theorem f_strictly_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 3 → f x > f y := by
  sorry

end f_strictly_decreasing_on_interval_l3188_318818


namespace probability_diamond_or_ace_in_two_draws_l3188_318824

/-- The probability of at least one of two cards being a diamond or an ace
    when drawn with replacement from a modified deck. -/
theorem probability_diamond_or_ace_in_two_draws :
  let total_cards : ℕ := 54
  let diamond_cards : ℕ := 13
  let ace_cards : ℕ := 4
  let diamond_or_ace_cards : ℕ := diamond_cards + ace_cards
  let prob_not_diamond_or_ace : ℚ := (total_cards - diamond_or_ace_cards) / total_cards
  let prob_at_least_one_diamond_or_ace : ℚ := 1 - prob_not_diamond_or_ace ^ 2
  prob_at_least_one_diamond_or_ace = 368 / 729 :=
by sorry

end probability_diamond_or_ace_in_two_draws_l3188_318824


namespace intersection_point_y_coordinate_l3188_318892

theorem intersection_point_y_coordinate : ∃ (x : ℝ), 
  0 < x ∧ x < π / 2 ∧ 
  2 + 3 * Real.cos (2 * x) = 3 * Real.sqrt 3 * Real.sin x ∧
  2 + 3 * Real.cos (2 * x) = 3 := by sorry

end intersection_point_y_coordinate_l3188_318892


namespace max_label_in_sample_l3188_318840

/-- Systematic sampling function that returns the maximum label in the sample -/
def systematic_sample_max (total : ℕ) (sample_size : ℕ) (first_item : ℕ) : ℕ :=
  let interval := total / sample_size
  let position := (first_item % interval) + 1
  (sample_size - (sample_size - position)) * interval + first_item

/-- Theorem stating the maximum label in the systematic sample -/
theorem max_label_in_sample :
  systematic_sample_max 80 5 10 = 74 := by
  sorry

#eval systematic_sample_max 80 5 10

end max_label_in_sample_l3188_318840


namespace expression_value_l3188_318858

theorem expression_value : (2^2 * 5) / (8 * 10) * (3 * 4 * 8) / (2 * 5 * 3) = 0.8 := by
  sorry

end expression_value_l3188_318858


namespace quadratic_equation_roots_l3188_318859

theorem quadratic_equation_roots : ∃ x : ℝ, (∀ y : ℝ, -y^2 + 2*y - 1 = 0 ↔ y = x) :=
sorry

end quadratic_equation_roots_l3188_318859


namespace divisors_of_572_divisors_of_572a3bc_case1_divisors_of_572a3bc_case2_l3188_318820

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_572 :
  count_divisors 572 = 12 :=
sorry

theorem divisors_of_572a3bc_case1 (a b c : ℕ) 
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c)
  (ha_gt : a > 20) (hb_gt : b > 20) (hc_gt : c > 20)
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  count_divisors (572 * a^3 * b * c) = 192 :=
sorry

theorem divisors_of_572a3bc_case2 :
  count_divisors (572 * 31^3 * 32 * 33) = 384 :=
sorry

end divisors_of_572_divisors_of_572a3bc_case1_divisors_of_572a3bc_case2_l3188_318820


namespace expression_equality_l3188_318872

theorem expression_equality (x : ℝ) (h1 : x^3 + 1 ≠ 0) (h2 : x^3 - 1 ≠ 0) : 
  ((x + 1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * 
  ((x - 1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 = 1 := by
  sorry

end expression_equality_l3188_318872


namespace no_real_roots_equation_implies_value_l3188_318874

theorem no_real_roots_equation_implies_value (a b : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (x / (x - 1) + (x - 1) / x ≠ (a + b * x) / (x^2 - x))) →
  8 * a + 4 * b - 5 = 11 := by
sorry

end no_real_roots_equation_implies_value_l3188_318874


namespace downstream_speed_l3188_318812

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℝ
  still_water : ℝ
  downstream : ℝ

/-- 
Given a rower's speed upstream and in still water, 
calculates and proves the rower's speed downstream
-/
theorem downstream_speed (r : RowerSpeed) 
  (h_upstream : r.upstream = 35)
  (h_still : r.still_water = 40) :
  r.downstream = 45 := by
  sorry

#check downstream_speed

end downstream_speed_l3188_318812


namespace ceiling_sqrt_200_l3188_318825

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end ceiling_sqrt_200_l3188_318825
