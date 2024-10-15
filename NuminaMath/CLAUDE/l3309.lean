import Mathlib

namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3309_330906

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon is a 12-sided polygon -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3309_330906


namespace NUMINAMATH_CALUDE_inscribed_cube_side_length_is_sqrt6_div_2_l3309_330938

/-- Represents a pyramid with a regular hexagonal base and equilateral triangle lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_is_equilateral : Bool

/-- Represents a cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  bottom_covers_base : Bool
  top_touches_midpoints : Bool

/-- Calculates the side length of an inscribed cube in a hexagonal pyramid -/
def inscribed_cube_side_length (cube : InscribedCube) : ℝ :=
  sorry

/-- Theorem stating that the side length of the inscribed cube is √6/2 -/
theorem inscribed_cube_side_length_is_sqrt6_div_2 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side_length = 2)
  (h2 : cube.pyramid.lateral_face_is_equilateral = true)
  (h3 : cube.bottom_covers_base = true)
  (h4 : cube.top_touches_midpoints = true) :
  inscribed_cube_side_length cube = Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_side_length_is_sqrt6_div_2_l3309_330938


namespace NUMINAMATH_CALUDE_defective_units_shipped_l3309_330944

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.09)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * 100) = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l3309_330944


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3309_330978

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, 2^n.val + 12^n.val + 2011^n.val = m^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3309_330978


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3309_330985

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3309_330985


namespace NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l3309_330994

/-- A hexagon with angles in arithmetic progression has one angle of 114 degrees. -/
theorem hexagon_arithmetic_progression_angle (a d : ℝ) : 
  (6 * a + 15 * d = 720) →  -- Sum of angles in hexagon
  (∃ k : ℕ, k < 6 ∧ a + k * d = 114) :=  -- One angle is 114 degrees
by sorry

end NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l3309_330994


namespace NUMINAMATH_CALUDE_window_width_is_24_inches_l3309_330910

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure Window where
  pane : GlassPane
  num_columns : ℕ
  num_rows : ℕ
  border_width : ℝ

/-- Calculates the total width of the window -/
def total_width (w : Window) : ℝ :=
  w.num_columns * w.pane.width + (w.num_columns + 1) * w.border_width

/-- Theorem stating that the total width of the window is 24 inches -/
theorem window_width_is_24_inches (w : Window) 
  (h1 : w.pane.height / w.pane.width = 3 / 4)
  (h2 : w.border_width = 3)
  (h3 : w.num_columns = 3)
  (h4 : w.num_rows = 2) :
  total_width w = 24 := by
  sorry


end NUMINAMATH_CALUDE_window_width_is_24_inches_l3309_330910


namespace NUMINAMATH_CALUDE_solution_value_l3309_330926

theorem solution_value (a : ℝ) : (1 + 1) * a = 2 * (2 * 1 - a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3309_330926


namespace NUMINAMATH_CALUDE_perpendicular_plane_line_condition_l3309_330912

-- Define the types for planes and lines
variable (Point : Type) (Vector : Type)
variable (Plane : Type) (Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define what it means for a line to be in a plane
variable (line_in_plane : Line → Plane → Prop)

theorem perpendicular_plane_line_condition 
  (α β : Plane) (m : Line) 
  (h_diff : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_plane_line_condition_l3309_330912


namespace NUMINAMATH_CALUDE_cube_volume_from_face_area_l3309_330969

theorem cube_volume_from_face_area (face_area : ℝ) (volume : ℝ) :
  face_area = 16 →
  volume = face_area ^ (3/2) →
  volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_area_l3309_330969


namespace NUMINAMATH_CALUDE_total_camp_attendance_l3309_330995

def lawrence_county_camp : ℕ := 34044
def lawrence_county_home : ℕ := 134867
def outside_county_camp : ℕ := 424944

theorem total_camp_attendance : 
  lawrence_county_camp + outside_county_camp = 459988 := by
  sorry

end NUMINAMATH_CALUDE_total_camp_attendance_l3309_330995


namespace NUMINAMATH_CALUDE_ratio_of_roots_quadratic_l3309_330936

theorem ratio_of_roots_quadratic (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ - 16 = 0 ∧ 
    x₂^2 + p*x₂ - 16 = 0 ∧ 
    x₁/x₂ = -4) → 
  p = 6 ∨ p = -6 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_roots_quadratic_l3309_330936


namespace NUMINAMATH_CALUDE_series_sum_after_removal_equals_neg_3026_l3309_330946

def series_sum (n : ℕ) : ℤ :=
  if n % 4 = 0 then
    (n - 3) - (n - 2) - (n - 1) + n
  else if n % 4 = 1 then
    n - (n + 1) - (n + 2)
  else
    0

def remove_multiples_of_10 (n : ℤ) : ℤ :=
  if n % 10 = 0 then 0 else n

def final_sum : ℤ :=
  (List.range 2015).foldl (λ acc i => acc + remove_multiples_of_10 (series_sum (i + 1))) 0

theorem series_sum_after_removal_equals_neg_3026 :
  final_sum = -3026 :=
sorry

end NUMINAMATH_CALUDE_series_sum_after_removal_equals_neg_3026_l3309_330946


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3309_330919

theorem quadratic_coefficient (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = (1 + Real.sqrt 3) / 2 ∧ 
                 x₂ = (1 - Real.sqrt 3) / 2 ∧ 
                 a * x₁^2 - x₁ - 1/2 = 0 ∧ 
                 a * x₂^2 - x₂ - 1/2 = 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3309_330919


namespace NUMINAMATH_CALUDE_problem_solution_l3309_330950

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 5/x + 1/x^2 = 40)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 11 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3309_330950


namespace NUMINAMATH_CALUDE_danes_daughters_flowers_l3309_330972

theorem danes_daughters_flowers (total_baskets : Nat) (flowers_per_basket : Nat) 
  (growth : Nat) (died : Nat) (num_daughters : Nat) :
  total_baskets = 5 →
  flowers_per_basket = 4 →
  growth = 20 →
  died = 10 →
  num_daughters = 2 →
  (total_baskets * flowers_per_basket + died - growth) / num_daughters = 5 := by
  sorry

end NUMINAMATH_CALUDE_danes_daughters_flowers_l3309_330972


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_g_l3309_330983

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - 2

-- Define the function g
def g (x : ℝ) : ℝ := |x + 3| - |2*x - 1| - 2

-- Theorem 1: Solution set of f(x) < |x-1|
theorem solution_set_f (x : ℝ) : f x < |x - 1| ↔ x < 0 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_g_l3309_330983


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3309_330939

theorem quadratic_equal_roots (k : ℝ) (A : ℝ) : 
  (∃ x : ℝ, A * x^2 + 6 * k * x + 2 = 0 ∧ 
   ∀ y : ℝ, A * y^2 + 6 * k * y + 2 = 0 → y = x) ∧ 
  k = 0.4444444444444444 → 
  A = 9 * k^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3309_330939


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3309_330924

/-- 
Given a right triangle with hypotenuse h, legs a and b, and an inscribed circle with radius r,
prove that the ratio of the area of the inscribed circle to the area of the triangle is πr / (h + r).
-/
theorem inscribed_circle_area_ratio (h a b r : ℝ) (h_positive : h > 0) (r_positive : r > 0) 
  (right_triangle : a^2 + b^2 = h^2) (inscribed_circle : r = (a + b - h) / 2) : 
  (π * r^2) / ((1/2) * a * b) = π * r / (h + r) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3309_330924


namespace NUMINAMATH_CALUDE_isosceles_triangle_l3309_330951

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.cos t.A * Real.cos t.B = 1 - Real.cos t.C) : 
  t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l3309_330951


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l3309_330977

theorem initial_markup_percentage
  (initial_price : ℝ)
  (price_increase : ℝ)
  (h1 : initial_price = 34)
  (h2 : price_increase = 6)
  (h3 : initial_price + price_increase = 2 * (initial_price - (initial_price + price_increase) / 2)) :
  (initial_price - (initial_price + price_increase) / 2) / ((initial_price + price_increase) / 2) = 0.7 :=
by sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l3309_330977


namespace NUMINAMATH_CALUDE_friend_team_assignment_l3309_330916

theorem friend_team_assignment (n : ℕ) (k : ℕ) : 
  n = 6 → k = 3 → k ^ n = 729 := by sorry

end NUMINAMATH_CALUDE_friend_team_assignment_l3309_330916


namespace NUMINAMATH_CALUDE_food_distribution_l3309_330933

/-- Given a total amount of food and a number of full boxes, calculates the amount of food per box. -/
def food_per_box (total_food : ℕ) (num_boxes : ℕ) : ℚ :=
  (total_food : ℚ) / (num_boxes : ℚ)

/-- Proves that given 777 kilograms of food and 388 full boxes, each box contains 2 kilograms of food. -/
theorem food_distribution (total_food : ℕ) (num_boxes : ℕ) 
  (h1 : total_food = 777) (h2 : num_boxes = 388) : 
  food_per_box total_food num_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_food_distribution_l3309_330933


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3309_330908

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3309_330908


namespace NUMINAMATH_CALUDE_expression_value_l3309_330934

theorem expression_value (y : ℝ) (some_variable : ℝ) 
  (h1 : some_variable / (2 * y) = 3 / 2)
  (h2 : (7 * some_variable + 5 * y) / (some_variable - 2 * y) = 26) :
  some_variable = 3 * y :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3309_330934


namespace NUMINAMATH_CALUDE_candy_distribution_l3309_330974

theorem candy_distribution (initial_candy : ℕ) (eaten : ℕ) (bowls : ℕ) (taken : ℕ) : 
  initial_candy = 100 →
  eaten = 8 →
  bowls = 4 →
  taken = 3 →
  (initial_candy - eaten) / bowls - taken = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3309_330974


namespace NUMINAMATH_CALUDE_company_production_l3309_330962

/-- The number of bottles a case can hold -/
def bottles_per_case : ℕ := 13

/-- The number of cases required for one-day production -/
def cases_per_day : ℕ := 5000

/-- The total number of bottles produced in one day -/
def bottles_per_day : ℕ := bottles_per_case * cases_per_day

/-- Theorem stating that the company produces 65,000 bottles per day -/
theorem company_production : bottles_per_day = 65000 := by
  sorry

end NUMINAMATH_CALUDE_company_production_l3309_330962


namespace NUMINAMATH_CALUDE_yellow_marbles_fraction_l3309_330953

theorem yellow_marbles_fraction (total : ℝ) (h : total > 0) :
  let initial_green := (2/3) * total
  let initial_yellow := total - initial_green
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = 3/5 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_fraction_l3309_330953


namespace NUMINAMATH_CALUDE_vector_equation_l3309_330928

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_equation : 4 • a - 3 • (a + b) = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equation_l3309_330928


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3309_330925

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_x2y2 (a b : ℕ) : ℕ := 
  binomial_coefficient a 2 * binomial_coefficient b 2

theorem expansion_coefficient : 
  coefficient_x2y2 3 4 = 18 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3309_330925


namespace NUMINAMATH_CALUDE_impossibleToAchieveTwoHundreds_l3309_330954

/-- Represents the score changes that can be applied to exam scores. -/
inductive ScoreChange
  | AddOneToAll
  | DecreaseOneIncreaseTwo

/-- Represents the scores for three exams. -/
structure ExamScores where
  russian : ℕ
  physics : ℕ
  mathematics : ℕ

/-- Applies a score change to the exam scores. -/
def applyScoreChange (scores : ExamScores) (change : ScoreChange) : ExamScores :=
  match change with
  | ScoreChange.AddOneToAll =>
      { russian := scores.russian + 1,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }
  | ScoreChange.DecreaseOneIncreaseTwo =>
      { russian := scores.russian - 3,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }

/-- Checks if at least two scores are equal to 100. -/
def atLeastTwoEqual100 (scores : ExamScores) : Prop :=
  (scores.russian = 100 ∧ scores.physics = 100) ∨
  (scores.russian = 100 ∧ scores.mathematics = 100) ∨
  (scores.physics = 100 ∧ scores.mathematics = 100)

/-- Theorem stating the impossibility of achieving at least two scores of 100. -/
theorem impossibleToAchieveTwoHundreds (initialScores : ExamScores)
  (hRussian : initialScores.russian = initialScores.physics - 5)
  (hPhysics : initialScores.physics = initialScores.mathematics - 9)
  (hMaxScore : ∀ scores : ExamScores, scores.russian ≤ 100 ∧ scores.physics ≤ 100 ∧ scores.mathematics ≤ 100) :
  ¬∃ (changes : List ScoreChange), atLeastTwoEqual100 (changes.foldl applyScoreChange initialScores) :=
sorry

end NUMINAMATH_CALUDE_impossibleToAchieveTwoHundreds_l3309_330954


namespace NUMINAMATH_CALUDE_X_related_Y_probability_l3309_330981

/-- The probability of k² being greater than or equal to 10.83 under the null hypothesis -/
def p_k_squared_ge_10_83 : ℝ := 0.001

/-- The null hypothesis states that variable X is unrelated to variable Y -/
def H₀ : Prop := sorry

/-- The probability that variable X is related to variable Y -/
def p_X_related_Y : ℝ := sorry

/-- Theorem stating the relationship between p_X_related_Y and p_k_squared_ge_10_83 -/
theorem X_related_Y_probability : 
  p_X_related_Y = 1 - p_k_squared_ge_10_83 := by sorry

end NUMINAMATH_CALUDE_X_related_Y_probability_l3309_330981


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3309_330988

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (a + 2) * x + 2 < 0}
  (a = 0 → solution_set = {x | x > 1}) ∧
  (0 < a ∧ a < 2 → solution_set = {x | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → solution_set = ∅) ∧
  (a > 2 → solution_set = {x | 2/a < x ∧ x < 1}) ∧
  (a < 0 → solution_set = {x | x < 2/a ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3309_330988


namespace NUMINAMATH_CALUDE_sum_of_roots_l3309_330900

theorem sum_of_roots (α β : ℝ) : 
  α^3 - 3*α^2 + 5*α - 4 = 0 → 
  β^3 - 3*β^2 + 5*β - 2 = 0 → 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3309_330900


namespace NUMINAMATH_CALUDE_triangular_pyramid_not_circular_top_view_l3309_330979

-- Define the types of solids
inductive Solid
  | Sphere
  | Cylinder
  | Cone
  | TriangularPyramid

-- Define a property for having a circular top view
def has_circular_top_view (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | Solid.Cylinder => True
  | Solid.Cone => True
  | Solid.TriangularPyramid => False

-- Theorem statement
theorem triangular_pyramid_not_circular_top_view :
  ∀ s : Solid, ¬(has_circular_top_view s) ↔ s = Solid.TriangularPyramid :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_not_circular_top_view_l3309_330979


namespace NUMINAMATH_CALUDE_expression_equality_l3309_330984

theorem expression_equality : 484 + 2 * 22 * 5 + 25 = 729 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3309_330984


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3309_330914

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3309_330914


namespace NUMINAMATH_CALUDE_negative_inequality_l3309_330920

theorem negative_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l3309_330920


namespace NUMINAMATH_CALUDE_pieces_per_box_l3309_330902

/-- Given information about Adam's chocolate candy boxes -/
structure ChocolateBoxes where
  totalBought : ℕ
  givenAway : ℕ
  piecesLeft : ℕ

/-- Theorem stating the number of pieces in each box -/
theorem pieces_per_box (boxes : ChocolateBoxes)
  (h1 : boxes.totalBought = 13)
  (h2 : boxes.givenAway = 7)
  (h3 : boxes.piecesLeft = 36) :
  boxes.piecesLeft / (boxes.totalBought - boxes.givenAway) = 6 := by
  sorry


end NUMINAMATH_CALUDE_pieces_per_box_l3309_330902


namespace NUMINAMATH_CALUDE_village_population_l3309_330930

theorem village_population (P : ℝ) : 
  (P > 0) →
  (0.8 * (0.9 * P) = 4500) →
  P = 6250 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3309_330930


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_diameter_l3309_330923

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The diameter of the circumscribed circle of a cyclic quadrilateral -/
def circumscribedCircleDiameter (q : CyclicQuadrilateral) : ℝ := sorry

/-- Theorem: The diameter of the circumscribed circle of a cyclic quadrilateral 
    with side lengths 25, 39, 52, and 60 is 65 -/
theorem cyclic_quadrilateral_diameter :
  ∀ (q : CyclicQuadrilateral), 
    q.a = 25 ∧ q.b = 39 ∧ q.c = 52 ∧ q.d = 60 →
    circumscribedCircleDiameter q = 65 := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_diameter_l3309_330923


namespace NUMINAMATH_CALUDE_percent_subtraction_problem_l3309_330905

theorem percent_subtraction_problem : ∃ x : ℝ, 0.12 * 160 - 0.38 * x = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_percent_subtraction_problem_l3309_330905


namespace NUMINAMATH_CALUDE_prob_even_and_divisible_by_three_on_two_dice_l3309_330911

/-- The probability of rolling an even number on a six-sided die -/
def prob_even_on_six_sided_die : ℚ := 1/2

/-- The probability of rolling a number divisible by three on a six-sided die -/
def prob_divisible_by_three_on_six_sided_die : ℚ := 1/3

/-- The probability of rolling an even number on one six-sided die
    and a number divisible by three on another six-sided die -/
theorem prob_even_and_divisible_by_three_on_two_dice :
  prob_even_on_six_sided_die * prob_divisible_by_three_on_six_sided_die = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_and_divisible_by_three_on_two_dice_l3309_330911


namespace NUMINAMATH_CALUDE_calculate_expression_l3309_330970

theorem calculate_expression : 
  3⁻¹ + (27 : ℝ) ^ (1/3) - (5 - Real.sqrt 5)^0 + |Real.sqrt 3 - 1/3| = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3309_330970


namespace NUMINAMATH_CALUDE_pentagon_x_coordinate_l3309_330961

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a horizontal line of symmetry -/
def hasHorizontalSymmetry (p : Pentagon) : Prop := sorry

theorem pentagon_x_coordinate :
  ∀ (p : Pentagon) (xc : ℝ),
    p.A = (0, 0) →
    p.B = (0, 6) →
    p.C = (xc, 12) →
    p.D = (6, 6) →
    p.E = (6, 0) →
    hasHorizontalSymmetry p →
    pentagonArea p = 60 →
    xc = 8 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_x_coordinate_l3309_330961


namespace NUMINAMATH_CALUDE_range_of_a_l3309_330958

theorem range_of_a (a : ℝ) : 
  (∀ x, 2*x^2 - x - 1 ≤ 0 → x^2 - (2*a-1)*x + a*(a-1) ≤ 0) ∧ 
  (∃ x, 2*x^2 - x - 1 ≤ 0 ∧ x^2 - (2*a-1)*x + a*(a-1) > 0) →
  1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3309_330958


namespace NUMINAMATH_CALUDE_fraction_inequality_l3309_330997

theorem fraction_inequality (x : ℝ) : 
  -3 ≤ x ∧ x ≤ 1 ∧ (3 * x + 8 ≥ 3 * (5 - 2 * x)) → 7/9 ≤ x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3309_330997


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3309_330940

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3309_330940


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3309_330915

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 5/4 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), (y = (4/3) * x ∨ y = -(4/3) * x) ↔ (y = (a/b) * x ∨ y = -(a/b) * x)) →
  (∀ (x y : ℝ), y^2/a^2 - x^2/b^2 = 1 → x = 0 → ∃ (c : ℝ), y^2 = a^2 + c^2) →
  e = (a^2 + b^2).sqrt / a := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3309_330915


namespace NUMINAMATH_CALUDE_triangle_roots_range_l3309_330966

theorem triangle_roots_range (m : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ - 2) * (x₁^2 - 4*x₁ + m) = 0 ∧
    (x₂ - 2) * (x₂^2 - 4*x₂ + m) = 0 ∧
    (x₃ - 2) * (x₃^2 - 4*x₃ + m) = 0 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ + x₂ > x₃ ∧ x₂ + x₃ > x₁ ∧ x₃ + x₁ > x₂) →
  3 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_roots_range_l3309_330966


namespace NUMINAMATH_CALUDE_inequality_proof_l3309_330929

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3309_330929


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3309_330945

theorem polynomial_coefficient_sum :
  ∀ (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ),
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3309_330945


namespace NUMINAMATH_CALUDE_expression_evaluation_l3309_330913

theorem expression_evaluation (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 13) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3309_330913


namespace NUMINAMATH_CALUDE_horner_method_v3_l3309_330971

def horner_polynomial (x : ℤ) : ℤ := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6

def horner_v3 (x : ℤ) : ℤ :=
  let v0 := 2
  let v1 := v0 * x + 6
  let v2 := v1 * x + 1
  v2 * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -36 ∧
  horner_polynomial (-4) = ((((horner_v3 (-4) * (-4) - 8) * (-4) + 25) * (-4)) + 10) :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3309_330971


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l3309_330943

theorem subtraction_of_fractions : (1 : ℚ) / 6 - (5 : ℚ) / 12 = (-1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l3309_330943


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l3309_330941

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards with one number and 1 card with a different number -/
def q : ℚ := ((distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number) : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating that q/p equals 225 -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l3309_330941


namespace NUMINAMATH_CALUDE_max_value_of_z_minus_i_l3309_330986

theorem max_value_of_z_minus_i (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - Complex.I) ≤ 2 ∧ ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_z_minus_i_l3309_330986


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l3309_330955

theorem max_value_x_plus_y (x y : ℝ) (h : x - Real.sqrt (x + 1) = Real.sqrt (y + 3) - y) :
  ∃ (M : ℝ), M = 4 ∧ x + y ≤ M ∧ ∀ (N : ℝ), (x + y ≤ N) → (M ≤ N) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l3309_330955


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l3309_330956

theorem subset_sum_divisible_by_2n (n : ℕ) (a : Fin n → ℕ) 
  (h1 : n ≥ 4)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, 0 < a i ∧ a i < 2*n) :
  ∃ (i j : Fin n), i < j ∧ (2*n) ∣ (a i + a j) :=
sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l3309_330956


namespace NUMINAMATH_CALUDE_min_value_theorem_l3309_330903

theorem min_value_theorem (C : ℝ) (x : ℝ) (h1 : C > 0) (h2 : x^3 - 1/x^3 = C) :
  C^2 + 9 ≥ 6 * C ∧ ∃ (C₀ : ℝ) (x₀ : ℝ), C₀ > 0 ∧ x₀^3 - 1/x₀^3 = C₀ ∧ C₀^2 + 9 = 6 * C₀ :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3309_330903


namespace NUMINAMATH_CALUDE_system_solution_sum_l3309_330975

theorem system_solution_sum (a b : ℝ) : 
  (a * 1 + b * 2 = 4 ∧ b * 1 - a * 2 = 7) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_sum_l3309_330975


namespace NUMINAMATH_CALUDE_scooter_initial_cost_l3309_330993

theorem scooter_initial_cost (P : ℝ) : 
  (P + 300) * 1.1 = 1320 → P = 900 := by sorry

end NUMINAMATH_CALUDE_scooter_initial_cost_l3309_330993


namespace NUMINAMATH_CALUDE_max_value_of_f_l3309_330973

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3309_330973


namespace NUMINAMATH_CALUDE_xy_value_l3309_330931

theorem xy_value (x y : ℝ) : y = Real.sqrt (x - 1/2) + Real.sqrt (1/2 - x) - 6 → x * y = -3 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3309_330931


namespace NUMINAMATH_CALUDE_sum_zero_in_2x2_square_l3309_330982

/-- Given a 2x2 square with numbers a, b, c, d that are pairwise distinct,
    with the sum of numbers in the first row equal to the sum of numbers in the second row,
    and the product of numbers in the first column equal to the product of numbers in the second column,
    prove that the sum of all four numbers is zero. -/
theorem sum_zero_in_2x2_square (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (row_sum : a + b = c + d)
  (col_prod : a * c = b * d) :
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_zero_in_2x2_square_l3309_330982


namespace NUMINAMATH_CALUDE_digit_property_l3309_330991

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def S (n : ℕ) : ℕ :=
  (digits n).sum

def P (n : ℕ) : ℕ :=
  (digits n).prod

theorem digit_property :
  ({ n : ℕ | n > 0 ∧ n = P n } = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  ({ n : ℕ | n > 0 ∧ n = S n + P n } = {19, 29, 39, 49, 59, 69, 79, 89, 99}) :=
by sorry

end NUMINAMATH_CALUDE_digit_property_l3309_330991


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3309_330990

/-- Given an arithmetic sequence {a_n} where a_2 = 3 and a_6 = 13, 
    prove that the common difference is 5/2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h1 : a 2 = 3) 
  (h2 : a 6 = 13) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) :
  ∃ d : ℚ, d = 5/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3309_330990


namespace NUMINAMATH_CALUDE_simplify_expression_l3309_330963

theorem simplify_expression (x : ℝ) : 3 * x^5 * (4 * x^3) = 12 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3309_330963


namespace NUMINAMATH_CALUDE_wheel_turns_l3309_330964

/-- A wheel makes 6 turns every 30 seconds. This theorem proves that it makes 1440 turns in 2 hours. -/
theorem wheel_turns (turns_per_30_sec : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → hours = 2 → turns_per_30_sec * 240 * hours = 1440 := by
  sorry

end NUMINAMATH_CALUDE_wheel_turns_l3309_330964


namespace NUMINAMATH_CALUDE_x_value_l3309_330989

def A (x : ℝ) : Set ℝ := {2, x, x^2 - 30}

theorem x_value (x : ℝ) (h : -5 ∈ A x) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3309_330989


namespace NUMINAMATH_CALUDE_expected_remaining_bullets_value_l3309_330932

/-- The probability of hitting the target -/
def p : ℝ := 0.6

/-- The total number of bullets -/
def n : ℕ := 4

/-- The expected number of remaining bullets -/
def expected_remaining_bullets : ℝ :=
  (n - 1) * p + (n - 2) * (1 - p) * p + (n - 3) * (1 - p)^2 * p + 0 * (1 - p)^3 * p

/-- Theorem stating the expected number of remaining bullets -/
theorem expected_remaining_bullets_value :
  expected_remaining_bullets = 2.376 := by sorry

end NUMINAMATH_CALUDE_expected_remaining_bullets_value_l3309_330932


namespace NUMINAMATH_CALUDE_magical_stack_with_151_fixed_l3309_330927

/-- A stack of cards is magical if at least one card from each pile retains its original position after restacking -/
def is_magical (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ 2*n ∧
  (a = 2*a - 1 ∨ b = 2*(b - n))

theorem magical_stack_with_151_fixed (n : ℕ) 
  (h_magical : is_magical n) 
  (h_151_fixed : 151 ≤ n ∧ 151 = 2*151 - 1) : 
  n = 226 ∧ 2*n = 452 := by sorry

end NUMINAMATH_CALUDE_magical_stack_with_151_fixed_l3309_330927


namespace NUMINAMATH_CALUDE_sundae_booth_packs_l3309_330917

/-- Calculates the number of packs needed for a given topping -/
def packs_needed (total_items : ℕ) (items_per_pack : ℕ) : ℕ :=
  (total_items + items_per_pack - 1) / items_per_pack

/-- Represents the sundae booth problem -/
theorem sundae_booth_packs (monday_sundaes tuesday_sundaes : ℕ)
  (monday_mms monday_gummy monday_marsh : ℕ)
  (tuesday_mms tuesday_gummy tuesday_marsh : ℕ)
  (mms_per_pack gummy_per_pack marsh_per_pack : ℕ)
  (h_monday : monday_sundaes = 40)
  (h_tuesday : tuesday_sundaes = 20)
  (h_monday_mms : monday_mms = 6)
  (h_monday_gummy : monday_gummy = 4)
  (h_monday_marsh : monday_marsh = 8)
  (h_tuesday_mms : tuesday_mms = 10)
  (h_tuesday_gummy : tuesday_gummy = 5)
  (h_tuesday_marsh : tuesday_marsh = 12)
  (h_mms_pack : mms_per_pack = 40)
  (h_gummy_pack : gummy_per_pack = 30)
  (h_marsh_pack : marsh_per_pack = 50) :
  (packs_needed (monday_sundaes * monday_mms + tuesday_sundaes * tuesday_mms) mms_per_pack = 11) ∧
  (packs_needed (monday_sundaes * monday_gummy + tuesday_sundaes * tuesday_gummy) gummy_per_pack = 9) ∧
  (packs_needed (monday_sundaes * monday_marsh + tuesday_sundaes * tuesday_marsh) marsh_per_pack = 12) :=
by sorry

end NUMINAMATH_CALUDE_sundae_booth_packs_l3309_330917


namespace NUMINAMATH_CALUDE_two_color_theorem_l3309_330909

/-- A line in a plane --/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- A region in a plane formed by intersecting lines --/
structure Region where
  -- We don't need to define the specifics of a region for this problem

/-- A color used for coloring regions --/
inductive Color
  | Red
  | Blue

/-- A configuration of lines in a plane --/
def Configuration := List Line

/-- A coloring of regions --/
def Coloring := Region → Color

/-- Check if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop :=
  sorry -- Definition of adjacency

/-- A valid coloring ensures no adjacent regions have the same color --/
def valid_coloring (c : Configuration) (coloring : Coloring) : Prop :=
  ∀ r1 r2 : Region, adjacent r1 r2 → coloring r1 ≠ coloring r2

/-- The main theorem: for any configuration of lines, there exists a valid coloring --/
theorem two_color_theorem (c : Configuration) : 
  ∃ coloring : Coloring, valid_coloring c coloring :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l3309_330909


namespace NUMINAMATH_CALUDE_safe_mushrooms_l3309_330907

/-- Given the following conditions about mushroom foraging:
  * The total number of mushrooms is 32
  * The number of poisonous mushrooms is twice the number of safe mushrooms
  * There are 5 uncertain mushrooms
  * The sum of safe, poisonous, and uncertain mushrooms equals the total
  Prove that the number of safe mushrooms is 9. -/
theorem safe_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) (uncertain : ℕ) 
  (h1 : total = 32)
  (h2 : poisonous = 2 * safe)
  (h3 : uncertain = 5)
  (h4 : safe + poisonous + uncertain = total) :
  safe = 9 := by sorry

end NUMINAMATH_CALUDE_safe_mushrooms_l3309_330907


namespace NUMINAMATH_CALUDE_max_value_w_l3309_330952

theorem max_value_w (p q : ℝ) 
  (h1 : 2 * p - q ≥ 0) 
  (h2 : 3 * q - 2 * p ≥ 0) 
  (h3 : 6 - 2 * q ≥ 0) : 
  Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q) ≤ 3 * Real.sqrt 2 ∧
  (Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q) = 3 * Real.sqrt 2 ↔ p = 2 ∧ q = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_w_l3309_330952


namespace NUMINAMATH_CALUDE_pond_length_l3309_330999

/-- Given a rectangular field and a square pond, prove the length of the pond. -/
theorem pond_length (field_length field_width pond_area_ratio : ℝ) 
  (h1 : field_length = 96)
  (h2 : field_width = 48)
  (h3 : field_length = 2 * field_width)
  (h4 : pond_area_ratio = 1 / 72) : 
  Real.sqrt (pond_area_ratio * field_length * field_width) = 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l3309_330999


namespace NUMINAMATH_CALUDE_factorial_sum_solution_l3309_330960

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_sum_solution :
  ∀ a b c d e f : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
    a > b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ e ≥ f →
    factorial a = factorial b + factorial c + factorial d + factorial e + factorial f →
    ((a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 5 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4 ∧ f = 4)) :=
by
  sorry

#check factorial_sum_solution

end NUMINAMATH_CALUDE_factorial_sum_solution_l3309_330960


namespace NUMINAMATH_CALUDE_train_meeting_time_l3309_330957

theorem train_meeting_time (distance : ℝ) (speed_diff : ℝ) (final_speed : ℝ) :
  distance = 450 →
  speed_diff = 6 →
  final_speed = 48 →
  (distance / (final_speed + (final_speed + speed_diff))) = 75 / 17 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_time_l3309_330957


namespace NUMINAMATH_CALUDE_photo_arrangements_l3309_330918

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def adjacent_arrangements (total_people : ℕ) (adjacent_pair : ℕ) : ℕ :=
  factorial (total_people - adjacent_pair + 1) * factorial adjacent_pair

theorem photo_arrangements :
  adjacent_arrangements 6 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3309_330918


namespace NUMINAMATH_CALUDE_tangent_slopes_sum_l3309_330996

/-- Parabola P with equation y = (x-3)^2 + 2 -/
def P : ℝ → ℝ := λ x ↦ (x - 3)^2 + 2

/-- Point Q -/
def Q : ℝ × ℝ := (15, 7)

/-- The sum of the slopes of the two tangent lines from Q to P is 48 -/
theorem tangent_slopes_sum : 
  ∃ (r s : ℝ), (∀ m : ℝ, (r < m ∧ m < s) ↔ 
    ∀ x : ℝ, P x ≠ (m * (x - Q.1) + Q.2)) ∧ r + s = 48 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slopes_sum_l3309_330996


namespace NUMINAMATH_CALUDE_sqrt_solution_l3309_330976

theorem sqrt_solution (x : ℝ) (h : x > 0) : 
  let y : ℝ → ℝ := λ x => Real.sqrt x
  2 * y x * (deriv y x) = 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_solution_l3309_330976


namespace NUMINAMATH_CALUDE_trinomial_fourth_power_l3309_330980

theorem trinomial_fourth_power (a b c : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_fourth_power_l3309_330980


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3309_330949

theorem triangle_angle_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 90) (h3 : b = 58) : c = 32 := by
  sorry

#check triangle_angle_sum

end NUMINAMATH_CALUDE_triangle_angle_sum_l3309_330949


namespace NUMINAMATH_CALUDE_train_speed_l3309_330901

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 32) : 
  (train_length + bridge_length) / crossing_time = 12.5 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3309_330901


namespace NUMINAMATH_CALUDE_original_number_proof_l3309_330965

theorem original_number_proof : 
  ∃! x : ℕ, 
    (∃ k : ℕ, x + 5 = 23 * k) ∧ 
    (∀ y : ℕ, y < 5 → ∀ m : ℕ, x + y ≠ 23 * m) :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3309_330965


namespace NUMINAMATH_CALUDE_eric_egg_collection_l3309_330922

/-- Represents the types of birds on Eric's farm -/
inductive BirdType
  | Chicken
  | Duck
  | Goose

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def num_birds (b : BirdType) : Nat :=
  match b with
  | BirdType.Chicken => 6
  | BirdType.Duck => 4
  | BirdType.Goose => 2

def normal_laying_rate (b : BirdType) : Nat :=
  match b with
  | BirdType.Chicken => 3
  | BirdType.Duck => 2
  | BirdType.Goose => 1

def is_sunday (d : Day) : Bool :=
  match d with
  | Day.Sunday => true
  | _ => false

def laying_rate (b : BirdType) (d : Day) : Nat :=
  if is_sunday d then
    max (normal_laying_rate b - 1) 0
  else
    normal_laying_rate b

def daily_eggs (d : Day) : Nat :=
  (num_birds BirdType.Chicken * laying_rate BirdType.Chicken d) +
  (num_birds BirdType.Duck * laying_rate BirdType.Duck d) +
  (num_birds BirdType.Goose * laying_rate BirdType.Goose d)

def weekly_eggs : Nat :=
  daily_eggs Day.Monday +
  daily_eggs Day.Tuesday +
  daily_eggs Day.Wednesday +
  daily_eggs Day.Thursday +
  daily_eggs Day.Friday +
  daily_eggs Day.Saturday +
  daily_eggs Day.Sunday

theorem eric_egg_collection : weekly_eggs = 184 := by
  sorry

end NUMINAMATH_CALUDE_eric_egg_collection_l3309_330922


namespace NUMINAMATH_CALUDE_sum_of_squares_l3309_330937

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 12)
  (eq2 : y^2 + 5*z = -15)
  (eq3 : z^2 + 7*x = -21) :
  x^2 + y^2 + z^2 = 83/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3309_330937


namespace NUMINAMATH_CALUDE_midnight_temperature_l3309_330904

/-- Given an initial temperature, a temperature rise, and a temperature drop,
    calculate the final temperature. -/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/-- Theorem stating that given the specific temperature changes in the problem,
    the final temperature is 2°C. -/
theorem midnight_temperature :
  final_temperature (-2) 12 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l3309_330904


namespace NUMINAMATH_CALUDE_book_arrangements_eq_103680_l3309_330968

/-- The number of ways to arrange 11 books (3 Arabic, 2 German, 4 Spanish, and 2 French) on a shelf,
    keeping the Arabic books together and the Spanish books together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 2
  let spanish_books : ℕ := 4
  let french_books : ℕ := 2
  let grouped_units : ℕ := 1 + 1 + german_books + french_books  -- Arabic and Spanish groups + individual German and French books
  (Nat.factorial grouped_units) * (Nat.factorial arabic_books) * (Nat.factorial spanish_books)

theorem book_arrangements_eq_103680 : book_arrangements = 103680 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_103680_l3309_330968


namespace NUMINAMATH_CALUDE_factorization_of_cyclic_expression_l3309_330998

theorem factorization_of_cyclic_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + a*c + b*c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cyclic_expression_l3309_330998


namespace NUMINAMATH_CALUDE_integral_convergence_l3309_330935

/-- The floor function, returning the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The integrand of our improper integral -/
noncomputable def f (x : ℝ) : ℝ :=
  (-1 : ℝ) ^ (floor (1 / x)) / x

/-- Statement of the convergence properties of our improper integral -/
theorem integral_convergence :
  ¬ (∃ (I : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (a : ℝ), 0 < a ∧ a < δ → |∫ x in a..1, |f x| - I| < ε) ∧
  (∃ (I : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (a : ℝ), 0 < a ∧ a < δ → |∫ x in a..1, f x - I| < ε) :=
by sorry

end NUMINAMATH_CALUDE_integral_convergence_l3309_330935


namespace NUMINAMATH_CALUDE_nina_payment_l3309_330967

theorem nina_payment (x y z w : ℕ) : 
  x + y + z + w = 27 →  -- Total number of coins
  y = 2 * z →           -- Number of 5 kopek coins is twice the number of 2 kopek coins
  z = 2 * x →           -- Number of 2 kopek coins is twice the number of 10 kopek coins
  7 < w →               -- Number of 3 kopek coins is more than 7
  w < 20 →              -- Number of 3 kopek coins is less than 20
  10 * x + 5 * y + 2 * z + 3 * w = 107 := by
sorry

end NUMINAMATH_CALUDE_nina_payment_l3309_330967


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3309_330948

theorem quadratic_inequality (x : ℝ) : x^2 - 10*x + 21 < 0 ↔ 3 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3309_330948


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3309_330942

theorem sum_with_radical_conjugate :
  let x : ℝ := 5 - Real.sqrt 500
  let y : ℝ := 5 + Real.sqrt 500
  x + y = 10 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3309_330942


namespace NUMINAMATH_CALUDE_stone_statue_cost_is_20_l3309_330947

/-- The cost of a stone statue -/
def stone_statue_cost : ℚ := 20

/-- The number of stone statues produced monthly -/
def stone_statues_per_month : ℕ := 10

/-- The number of wooden statues produced monthly -/
def wooden_statues_per_month : ℕ := 20

/-- The cost of a wooden statue -/
def wooden_statue_cost : ℚ := 5

/-- The tax rate as a decimal -/
def tax_rate : ℚ := 1/10

/-- The monthly earnings after taxes -/
def monthly_earnings_after_taxes : ℚ := 270

/-- Theorem stating that the cost of a stone statue is $20 -/
theorem stone_statue_cost_is_20 :
  stone_statue_cost * stone_statues_per_month +
  wooden_statue_cost * wooden_statues_per_month =
  monthly_earnings_after_taxes / (1 - tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_stone_statue_cost_is_20_l3309_330947


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3309_330959

theorem hyperbola_focal_length (b : ℝ) : 
  (b > 0) → 
  (∃ (x y : ℝ), x^2 - y^2/b^2 = 1) → 
  (∃ (c : ℝ), c = 2) → 
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3309_330959


namespace NUMINAMATH_CALUDE_equal_sums_exist_l3309_330921

/-- Represents a cell in the table -/
inductive Cell
  | Neg : Cell  -- Represents -1
  | Zero : Cell -- Represents 0
  | Pos : Cell  -- Represents 1

/-- Represents a (2n+1) × (2n+1) table -/
def Table (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → Cell

/-- Calculates the sum of a row or column -/
def sum_line (t : Table n) (is_row : Bool) (i : Fin (2*n+1)) : ℤ :=
  sorry

/-- The main theorem -/
theorem equal_sums_exist (n : ℕ) (t : Table n) :
  ∃ (i j : Fin (2*n+1)) (b₁ b₂ : Bool), 
    (i ≠ j ∨ b₁ ≠ b₂) ∧ sum_line t b₁ i = sum_line t b₂ j :=
sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l3309_330921


namespace NUMINAMATH_CALUDE_spring_sales_l3309_330987

/-- Represents the sales data for a fast food chain's hamburger sales across seasons --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total sales is the sum of sales from all seasons --/
def totalSales (s : SeasonalSales) : ℝ :=
  s.spring + s.summer + s.fall + s.winter

/-- Given the conditions of the problem --/
theorem spring_sales (s : SeasonalSales)
    (h1 : s.summer = 6)
    (h2 : s.fall = 4)
    (h3 : s.winter = 3)
    (h4 : s.winter = 0.2 * totalSales s) :
    s.spring = 2 := by
  sorry


end NUMINAMATH_CALUDE_spring_sales_l3309_330987


namespace NUMINAMATH_CALUDE_airport_visit_total_l3309_330992

theorem airport_visit_total (first_graders : ℕ) (second_graders_difference : ℕ) : 
  first_graders = 358 →
  second_graders_difference = 64 →
  first_graders + (first_graders - second_graders_difference) = 652 :=
by sorry

end NUMINAMATH_CALUDE_airport_visit_total_l3309_330992
