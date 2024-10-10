import Mathlib

namespace john_yasmin_children_ratio_l2148_214843

/-- The number of children John has -/
def john_children : ℕ := sorry

/-- The number of children Yasmin has -/
def yasmin_children : ℕ := 2

/-- The total number of grandchildren Gabriel has -/
def gabriel_grandchildren : ℕ := 6

/-- The ratio of John's children to Yasmin's children -/
def children_ratio : ℚ := john_children / yasmin_children

theorem john_yasmin_children_ratio :
  (john_children + yasmin_children = gabriel_grandchildren) →
  children_ratio = 2 := by
sorry

end john_yasmin_children_ratio_l2148_214843


namespace min_tiles_cover_rect_l2148_214836

/-- The side length of a square tile in inches -/
def tile_side : ℕ := 6

/-- The length of the rectangular region in feet -/
def rect_length : ℕ := 6

/-- The width of the rectangular region in feet -/
def rect_width : ℕ := 3

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The minimum number of tiles needed to cover the rectangular region -/
def min_tiles : ℕ := 72

theorem min_tiles_cover_rect : 
  (rect_length * inches_per_foot) * (rect_width * inches_per_foot) = 
  min_tiles * (tile_side * tile_side) :=
by sorry

end min_tiles_cover_rect_l2148_214836


namespace range_of_f_when_a_is_1_a_values_when_f_min_is_3_l2148_214869

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Theorem 1: Range of f when a = 1
theorem range_of_f_when_a_is_1 :
  ∀ y ∈ Set.Icc 0 9, ∃ x ∈ Set.Icc 0 2, f 1 x = y ∧
  ∀ x ∈ Set.Icc 0 2, 0 ≤ f 1 x ∧ f 1 x ≤ 9 :=
sorry

-- Theorem 2: Values of a when f has minimum value 3
theorem a_values_when_f_min_is_3 :
  (∃ x ∈ Set.Icc 0 2, f a x = 3 ∧ ∀ y ∈ Set.Icc 0 2, f a y ≥ 3) →
  (a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10) :=
sorry

end range_of_f_when_a_is_1_a_values_when_f_min_is_3_l2148_214869


namespace revenue_change_with_price_increase_and_quantity_decrease_l2148_214807

/-- Theorem: Effect on revenue when price increases and quantity decreases -/
theorem revenue_change_with_price_increase_and_quantity_decrease 
  (P Q : ℝ) (P_new Q_new R_new : ℝ) :
  P_new = P * (1 + 0.30) →
  Q_new = Q * (1 - 0.20) →
  R_new = P_new * Q_new →
  R_new = P * Q * 1.04 := by
sorry

end revenue_change_with_price_increase_and_quantity_decrease_l2148_214807


namespace line_circle_intersection_l2148_214828

theorem line_circle_intersection (a : ℝ) :
  ∃ (x y : ℝ), (a * x - y + 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end line_circle_intersection_l2148_214828


namespace binomial_10_choose_3_l2148_214826

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l2148_214826


namespace field_trip_seats_l2148_214852

theorem field_trip_seats (students : ℕ) (buses : ℕ) (seats_per_bus : ℕ) 
  (h1 : students = 28) 
  (h2 : buses = 4) 
  (h3 : students = buses * seats_per_bus) : 
  seats_per_bus = 7 := by
  sorry

end field_trip_seats_l2148_214852


namespace fraction_problem_l2148_214813

theorem fraction_problem (x : ℚ) :
  (x / (2 * x + 11) = 3 / 4) → x = -33 / 2 := by
  sorry

end fraction_problem_l2148_214813


namespace lottery_winning_numbers_l2148_214802

/-- Calculates the number of winning numbers on each lottery ticket -/
theorem lottery_winning_numbers
  (num_tickets : ℕ)
  (winning_number_value : ℕ)
  (total_amount_won : ℕ)
  (h1 : num_tickets = 3)
  (h2 : winning_number_value = 20)
  (h3 : total_amount_won = 300)
  (h4 : total_amount_won % winning_number_value = 0)
  (h5 : (total_amount_won / winning_number_value) % num_tickets = 0) :
  total_amount_won / winning_number_value / num_tickets = 5 :=
by sorry

end lottery_winning_numbers_l2148_214802


namespace train_crossing_tree_time_l2148_214820

/-- Given a train and a platform with specific properties, calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_pass_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : platform_length = 400)
  (h3 : time_pass_platform = 160) :
  (train_length / ((train_length + platform_length) / time_pass_platform)) = 120 := by
  sorry

#check train_crossing_tree_time

end train_crossing_tree_time_l2148_214820


namespace equilateral_triangle_exists_l2148_214891

-- Define a type for colors
inductive Color
| Black
| White

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  p1 : Point
  p2 : Point
  p3 : Point
  eq_sides : distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

-- Theorem statement
theorem equilateral_triangle_exists :
  ∃ (t : EquilateralTriangle),
    (distance t.p1 t.p2 = 1 ∨ distance t.p1 t.p2 = Real.sqrt 3) ∧
    (coloring t.p1 = coloring t.p2 ∧ coloring t.p2 = coloring t.p3) :=
by sorry

end equilateral_triangle_exists_l2148_214891


namespace cosine_tangent_equality_l2148_214847

theorem cosine_tangent_equality : 4 * Real.cos (10 * π / 180) - Real.tan (80 * π / 180) = -Real.sqrt 3 := by
  sorry

end cosine_tangent_equality_l2148_214847


namespace spherical_segment_volume_ratio_l2148_214814

theorem spherical_segment_volume_ratio (α : ℝ) :
  let R : ℝ := 1  -- Assume unit sphere for simplicity
  let V_sphere : ℝ := (4 / 3) * Real.pi * R^3
  let H : ℝ := 2 * R * Real.sin (α / 4)^2
  let V_seg : ℝ := Real.pi * H^2 * (R - H / 3)
  V_seg / V_sphere = Real.sin (α / 4)^4 * (2 + Real.cos (α / 2)) :=
by sorry

end spherical_segment_volume_ratio_l2148_214814


namespace chessboard_polygon_tasteful_tiling_l2148_214842

-- Define a chessboard polygon
def ChessboardPolygon : Type := sorry

-- Define a domino
def Domino : Type := sorry

-- Define a tiling
def Tiling (p : ChessboardPolygon) : Type := sorry

-- Define a tasteful tiling
def TastefulTiling (p : ChessboardPolygon) : Type := sorry

-- Define the property of being tileable by dominoes
def IsTileable (p : ChessboardPolygon) : Prop := sorry

-- Theorem statement
theorem chessboard_polygon_tasteful_tiling 
  (p : ChessboardPolygon) (h : IsTileable p) :
  (∃ t : TastefulTiling p, True) ∧ 
  (∀ t1 t2 : TastefulTiling p, t1 = t2) :=
sorry

end chessboard_polygon_tasteful_tiling_l2148_214842


namespace remainder_theorem_l2148_214878

theorem remainder_theorem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 3 * u * y + 2) % y = (v + 2) % y := by
  sorry

end remainder_theorem_l2148_214878


namespace bamboo_problem_l2148_214851

/-- 
Given a geometric sequence of 9 terms where the sum of the first 3 terms is 2 
and the sum of the last 3 terms is 128, the 5th term is equal to 32/7.
-/
theorem bamboo_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 1 + a 2 + a 3 = 2 →         -- sum of first 3 terms
  a 7 + a 8 + a 9 = 128 →       -- sum of last 3 terms
  a 5 = 32 / 7 := by
sorry

end bamboo_problem_l2148_214851


namespace lines_skew_iff_b_not_neg_6_4_l2148_214804

/-- Two lines in 3D space --/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew --/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (b : ℝ), l1.point.2.2 = b ∧ 
  ¬∃ (t u : ℝ), 
    (l1.point.1 + t * l1.direction.1 = l2.point.1 + u * l2.direction.1) ∧
    (l1.point.2.1 + t * l1.direction.2.1 = l2.point.2.1 + u * l2.direction.2.1) ∧
    (b + t * l1.direction.2.2 = l2.point.2.2 + u * l2.direction.2.2)

theorem lines_skew_iff_b_not_neg_6_4 :
  ∀ (b : ℝ), are_skew 
    (Line3D.mk (2, 3, b) (3, 4, 5)) 
    (Line3D.mk (5, 2, 1) (6, 3, 2))
  ↔ b ≠ -6.4 := by sorry

end lines_skew_iff_b_not_neg_6_4_l2148_214804


namespace willam_tax_is_960_l2148_214823

/-- Represents the farm tax scenario in Mr. Willam's village -/
structure FarmTax where
  -- Total taxable land in the village
  total_taxable_land : ℝ
  -- Tax rate per unit of taxable land
  tax_rate : ℝ
  -- Percentage of Mr. Willam's taxable land
  willam_land_percentage : ℝ

/-- Calculates Mr. Willam's tax payment -/
def willam_tax_payment (ft : FarmTax) : ℝ :=
  ft.total_taxable_land * ft.tax_rate * ft.willam_land_percentage

/-- Theorem stating that Mr. Willam's tax payment is $960 -/
theorem willam_tax_is_960 (ft : FarmTax) 
    (h1 : ft.tax_rate * ft.total_taxable_land = 3840)
    (h2 : ft.willam_land_percentage = 0.25) : 
  willam_tax_payment ft = 960 := by
  sorry


end willam_tax_is_960_l2148_214823


namespace simplify_polynomial_l2148_214835

theorem simplify_polynomial (x : ℝ) : 
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) = 
  8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 := by
  sorry

end simplify_polynomial_l2148_214835


namespace associates_hired_to_change_ratio_l2148_214865

/-- The number of additional associates hired to change the ratio -/
def additional_associates (initial_ratio_partners initial_ratio_associates new_ratio_partners new_ratio_associates current_partners : ℕ) : ℕ :=
  let initial_associates := (initial_ratio_associates * current_partners) / initial_ratio_partners
  let total_new_associates := (new_ratio_associates * current_partners) / new_ratio_partners
  total_new_associates - initial_associates

/-- Theorem stating that 50 additional associates were hired to change the ratio -/
theorem associates_hired_to_change_ratio :
  additional_associates 2 63 1 34 20 = 50 := by
  sorry

end associates_hired_to_change_ratio_l2148_214865


namespace ratio_problem_l2148_214896

theorem ratio_problem (c d : ℚ) : 
  (c / d = 5) → (c = 18 - 7 * d) → (d = 3 / 2) := by
sorry

end ratio_problem_l2148_214896


namespace quadratic_equation_one_solutions_l2148_214834

theorem quadratic_equation_one_solutions (x : ℝ) :
  x^2 - 6*x - 1 = 0 ↔ x = 3 + Real.sqrt 10 ∨ x = 3 - Real.sqrt 10 :=
sorry

end quadratic_equation_one_solutions_l2148_214834


namespace solution_set_l2148_214882

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom increasing_f : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0
axiom odd_shifted : ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

-- State the theorem
theorem solution_set (x : ℝ) : f (1 - x) > 0 ↔ x < 0 := by
  sorry

end solution_set_l2148_214882


namespace logic_propositions_l2148_214883

-- Define the propositions
def corresponding_angles_equal (l₁ l₂ : Line) : Prop := sorry
def lines_parallel (l₁ l₂ : Line) : Prop := sorry

-- Define the sine function and angle measure
def sin : ℝ → ℝ := sorry
def degree : ℝ → ℝ := sorry

-- Define the theorem
theorem logic_propositions :
  -- 1. Contrapositive
  (∀ l₁ l₂ : Line, (corresponding_angles_equal l₁ l₂ → lines_parallel l₁ l₂) ↔ 
    (¬lines_parallel l₁ l₂ → ¬corresponding_angles_equal l₁ l₂)) ∧
  -- 2. Necessary but not sufficient condition
  (∀ α : ℝ, sin α = 1/2 → degree α = 30) ∧
  (∃ β : ℝ, sin β = 1/2 ∧ degree β ≠ 30) ∧
  -- 3. Falsity of conjunction
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  -- 4. Negation of existence
  (¬(∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by sorry

end logic_propositions_l2148_214883


namespace average_of_remaining_checks_l2148_214816

def travelers_checks_problem (x y z : ℕ) : Prop :=
  x + y = 30 ∧ 
  50 * x + z * y = 1800 ∧ 
  x ≥ 24 ∧
  z > 0

theorem average_of_remaining_checks (x y z : ℕ) 
  (h : travelers_checks_problem x y z) : 
  (1800 - 50 * 24) / (30 - 24) = 100 :=
sorry

end average_of_remaining_checks_l2148_214816


namespace min_angle_B_in_special_triangle_l2148_214803

open Real

theorem min_angle_B_in_special_triangle (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  ∃ k : ℝ, tan A + k = (1 + sqrt 2) * tan B ∧ (1 + sqrt 2) * tan B + k = tan C →
  π / 4 ≤ B :=
by sorry

end min_angle_B_in_special_triangle_l2148_214803


namespace min_value_and_range_l2148_214867

theorem min_value_and_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b - a*b = 0 → x + 2*y ≤ a + 2*b) ∧ y > 1 := by
  sorry

end min_value_and_range_l2148_214867


namespace chess_group_players_l2148_214873

theorem chess_group_players (n : ℕ) : 
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < n * (n - 1) / 2) →
  (∀ (game : ℕ), game < n * (n - 1) / 2 → ∃! (i j : ℕ), i < n ∧ j < n ∧ i ≠ j) →
  n * (n - 1) / 2 = 105 →
  n = 15 := by
sorry

end chess_group_players_l2148_214873


namespace four_layer_grid_triangles_l2148_214840

/-- Calculates the total number of triangles in a triangular grid with a given number of layers. -/
def triangles_in_grid (layers : ℕ) : ℕ :=
  let small_triangles := (layers * (layers + 1)) / 2
  let medium_triangles := if layers ≥ 3 then (layers - 2) * (layers - 1) / 2 else 0
  let large_triangles := 1
  small_triangles + medium_triangles + large_triangles

/-- Theorem stating that a triangular grid with 4 layers contains 21 triangles. -/
theorem four_layer_grid_triangles :
  triangles_in_grid 4 = 21 :=
by sorry

end four_layer_grid_triangles_l2148_214840


namespace cos_75_deg_l2148_214825

/-- Proves that cos 75° = (√6 - √2) / 4 using cos 60° and cos 15° -/
theorem cos_75_deg (cos_60_deg : Real) (cos_15_deg : Real) :
  cos_60_deg = 1 / 2 →
  cos_15_deg = (Real.sqrt 6 + Real.sqrt 2) / 4 →
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_deg_l2148_214825


namespace triangle_area_l2148_214898

/-- Given a triangle ABC where sin A = 3/5 and the dot product of vectors AB and AC is 8,
    prove that the area of the triangle is 3. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let sinA : ℝ := 3/5
  let dotProduct : ℝ := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  dotProduct = 8 →
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
         Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * sinA = 3 := by
  sorry

end triangle_area_l2148_214898


namespace fraction_inequality_l2148_214874

theorem fraction_inequality (a b : ℝ) (h : b < a ∧ a < 0) : 1 / a < 1 / b := by
  sorry

end fraction_inequality_l2148_214874


namespace total_gold_spent_l2148_214832

-- Define the quantities and prices
def gary_grams : ℝ := 30
def gary_price : ℝ := 15
def anna_grams : ℝ := 50
def anna_price : ℝ := 20
def lisa_grams : ℝ := 40
def lisa_price : ℝ := 15
def john_grams : ℝ := 60
def john_price : ℝ := 18

-- Define conversion rates
def euro_to_dollar : ℝ := 1.1
def pound_to_dollar : ℝ := 1.3

-- Define the total spent function
def total_spent : ℝ := 
  gary_grams * gary_price + 
  anna_grams * anna_price + 
  lisa_grams * lisa_price * euro_to_dollar + 
  john_grams * john_price * pound_to_dollar

-- Theorem statement
theorem total_gold_spent : total_spent = 3514 := by
  sorry

end total_gold_spent_l2148_214832


namespace negation_of_universal_proposition_l2148_214885

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2148_214885


namespace smallest_norwegian_number_l2148_214881

/-- A number is Norwegian if it has three distinct positive divisors whose sum is equal to 2022. -/
def IsNorwegian (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧
    d₁ + d₂ + d₃ = 2022

/-- 1344 is the smallest Norwegian number. -/
theorem smallest_norwegian_number : 
  IsNorwegian 1344 ∧ ∀ m : ℕ, m < 1344 → ¬IsNorwegian m :=
by sorry

end smallest_norwegian_number_l2148_214881


namespace equation_root_condition_l2148_214887

theorem equation_root_condition (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ y : ℝ, y > 0 → |y| ≠ a * y + 1) → 
  a > -1 := by sorry

end equation_root_condition_l2148_214887


namespace least_11_heavy_three_digit_l2148_214880

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_11_heavy_three_digit : 
  (∀ n : ℕ, is_three_digit n → is_11_heavy n → 107 ≤ n) ∧ 
  is_three_digit 107 ∧ 
  is_11_heavy 107 :=
sorry

end least_11_heavy_three_digit_l2148_214880


namespace josh_marbles_count_l2148_214846

theorem josh_marbles_count (initial_marbles found_marbles : ℕ) 
  (h1 : initial_marbles = 21)
  (h2 : found_marbles = 7) :
  initial_marbles + found_marbles = 28 := by
  sorry

end josh_marbles_count_l2148_214846


namespace inverse_proportion_point_relation_l2148_214806

theorem inverse_proportion_point_relation :
  ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = 3 / (-5) →
  y₂ = 3 / (-3) →
  y₃ = 3 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end inverse_proportion_point_relation_l2148_214806


namespace large_lemonade_price_l2148_214810

/-- Represents the price and sales data for Tonya's lemonade stand --/
structure LemonadeStand where
  small_price : ℝ
  medium_price : ℝ
  large_price : ℝ
  total_sales : ℝ
  small_sales : ℝ
  medium_sales : ℝ
  large_cups_sold : ℕ

/-- Theorem stating that the price of a large cup of lemonade is $3 --/
theorem large_lemonade_price (stand : LemonadeStand)
  (h1 : stand.small_price = 1)
  (h2 : stand.medium_price = 2)
  (h3 : stand.total_sales = 50)
  (h4 : stand.small_sales = 11)
  (h5 : stand.medium_sales = 24)
  (h6 : stand.large_cups_sold = 5)
  (h7 : stand.total_sales = stand.small_sales + stand.medium_sales + stand.large_price * stand.large_cups_sold) :
  stand.large_price = 3 := by
  sorry


end large_lemonade_price_l2148_214810


namespace smallest_factorizable_b_l2148_214848

/-- A polynomial of degree 2 with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents a factorization of a quadratic polynomial into two linear factors -/
structure Factorization where
  p : ℤ
  q : ℤ

/-- Checks if a factorization is valid for a given quadratic polynomial -/
def isValidFactorization (poly : QuadraticPolynomial) (fac : Factorization) : Prop :=
  poly.a = 1 ∧ poly.b = fac.p + fac.q ∧ poly.c = fac.p * fac.q

/-- Theorem stating that 259 is the smallest positive integer b for which
    x^2 + bx + 2008 can be factored into a product of two polynomials
    with integer coefficients -/
theorem smallest_factorizable_b :
  ∀ b : ℤ, b > 0 →
  (∃ fac : Factorization, isValidFactorization ⟨1, b, 2008⟩ fac) →
  b ≥ 259 :=
sorry

end smallest_factorizable_b_l2148_214848


namespace quadrilateral_area_is_12_825_l2148_214899

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α
  pos : 0 < side

/-- Represents the configuration of three squares aligned on their bottom edges -/
structure SquareConfiguration (α : Type*) [LinearOrderedField α] where
  small : Square α
  medium : Square α
  large : Square α
  alignment : small.side + medium.side + large.side > 0

/-- Calculates the area of the quadrilateral formed in the square configuration -/
noncomputable def quadrilateralArea {α : Type*} [LinearOrderedField α] (config : SquareConfiguration α) : α :=
  sorry

/-- Theorem stating that the area of the quadrilateral in the given configuration is 12.825 -/
theorem quadrilateral_area_is_12_825 :
  let config : SquareConfiguration ℝ := {
    small := { side := 3, pos := by norm_num },
    medium := { side := 5, pos := by norm_num },
    large := { side := 7, pos := by norm_num },
    alignment := by norm_num
  }
  quadrilateralArea config = 12.825 := by
  sorry

end quadrilateral_area_is_12_825_l2148_214899


namespace expression_not_constant_l2148_214858

theorem expression_not_constant (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  ¬ ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 3 → x ≠ -2 → 
    (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - 
    (x^2 + 4 * x + 4) / ((x - 3) * (x + 2)) = c :=
by sorry

end expression_not_constant_l2148_214858


namespace geometric_arithmetic_progression_problem_l2148_214849

theorem geometric_arithmetic_progression_problem (a b c : ℝ) :
  (∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ 12 = a * q^2) ∧  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ 9 = a + 2 * d) →        -- Arithmetic progression condition
  ((a = -9 ∧ b = -6 ∧ c = 12) ∨ (a = 15 ∧ b = 12 ∧ c = 9)) :=
by sorry

end geometric_arithmetic_progression_problem_l2148_214849


namespace special_number_in_list_l2148_214875

theorem special_number_in_list (numbers : List ℝ) (n : ℝ) : 
  numbers.length = 21 ∧ 
  n ∈ numbers ∧
  n = 4 * ((numbers.sum - n) / 20) →
  n = (1 / 6) * numbers.sum :=
by sorry

end special_number_in_list_l2148_214875


namespace power_sum_l2148_214822

theorem power_sum (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(m+n) = 6 := by
  sorry

end power_sum_l2148_214822


namespace banana_arrangements_l2148_214860

def word_length : ℕ := 7

def identical_b_count : ℕ := 2

def distinct_letter_count : ℕ := 5

theorem banana_arrangements :
  (word_length.factorial / identical_b_count.factorial) = 2520 := by
  sorry

end banana_arrangements_l2148_214860


namespace philatelist_stamps_problem_l2148_214809

theorem philatelist_stamps_problem :
  ∃! x : ℕ,
    x % 3 = 1 ∧
    x % 5 = 3 ∧
    x % 7 = 5 ∧
    150 < x ∧
    x ≤ 300 ∧
    x = 208 := by
  sorry

end philatelist_stamps_problem_l2148_214809


namespace polynomial_relation_l2148_214812

theorem polynomial_relation (r : ℝ) : r^3 - 2*r + 1 = 0 → r^6 - 4*r^4 + 4*r^2 - 1 = 0 := by
  sorry

end polynomial_relation_l2148_214812


namespace second_integer_problem_l2148_214808

theorem second_integer_problem (x y : ℕ+) (hx : x = 3) (h : x * y + x = 33) : y = 10 := by
  sorry

end second_integer_problem_l2148_214808


namespace total_watermelon_slices_l2148_214805

-- Define the number of watermelons and slices for each person
def danny_watermelons : ℕ := 3
def danny_slices_per_melon : ℕ := 10

def sister_watermelons : ℕ := 1
def sister_slices_per_melon : ℕ := 15

def cousin_watermelons : ℕ := 2
def cousin_slices_per_melon : ℕ := 8

def aunt_watermelons : ℕ := 4
def aunt_slices_per_melon : ℕ := 12

def grandfather_watermelons : ℕ := 1
def grandfather_slices_per_melon : ℕ := 6

-- Define the total number of slices
def total_slices : ℕ := 
  danny_watermelons * danny_slices_per_melon +
  sister_watermelons * sister_slices_per_melon +
  cousin_watermelons * cousin_slices_per_melon +
  aunt_watermelons * aunt_slices_per_melon +
  grandfather_watermelons * grandfather_slices_per_melon

-- Theorem statement
theorem total_watermelon_slices : total_slices = 115 := by
  sorry

end total_watermelon_slices_l2148_214805


namespace inverse_proportion_in_first_third_quadrants_l2148_214837

/-- An inverse proportion function -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- A function whose graph lies in the first and third quadrants -/
def FirstThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

theorem inverse_proportion_in_first_third_quadrants
  (f : ℝ → ℝ) (h1 : InverseProportion f) (h2 : FirstThirdQuadrants f) :
  ∃ k : ℝ, k > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x :=
sorry

end inverse_proportion_in_first_third_quadrants_l2148_214837


namespace quadratic_expression_l2148_214864

/-- A quadratic function passing through the point (3, 10) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that f passes through (3, 10) -/
def passes_through (a b c : ℝ) : Prop := f a b c 3 = 10

theorem quadratic_expression (a b c : ℝ) (h : passes_through a b c) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end quadratic_expression_l2148_214864


namespace circle_area_ratio_l2148_214890

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) :
  (60 / 360 * (2 * Real.pi * C) = 40 / 360 * (2 * Real.pi * D)) →
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end circle_area_ratio_l2148_214890


namespace subset_condition_disjoint_condition_l2148_214868

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part (1)
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem for part (2)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end subset_condition_disjoint_condition_l2148_214868


namespace road_division_l2148_214872

theorem road_division (a b c : ℝ) : 
  a + b + c = 28 →
  a > 0 → b > 0 → c > 0 →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c / 2) - a / 2 = 16 →
  b = 4 :=
by sorry

end road_division_l2148_214872


namespace a_range_l2148_214839

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a*x^2 - x + a)

-- Define the theorem
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (a ≥ 1/2 ∧ a < 1) :=
sorry

end a_range_l2148_214839


namespace prob_same_suit_60_card_deck_l2148_214850

/-- A deck of cards with a specified number of ranks and suits. -/
structure Deck :=
  (num_ranks : ℕ)
  (num_suits : ℕ)

/-- The probability of drawing two cards of the same suit from a deck. -/
def prob_same_suit (d : Deck) : ℚ :=
  if d.num_ranks * d.num_suits = 0 then 0
  else (d.num_ranks - 1) / (d.num_ranks * d.num_suits - 1)

/-- Theorem stating the probability of drawing two cards of the same suit
    from a 60-card deck with 15 ranks and 4 suits. -/
theorem prob_same_suit_60_card_deck :
  prob_same_suit ⟨15, 4⟩ = 14 / 59 := by
  sorry

end prob_same_suit_60_card_deck_l2148_214850


namespace new_savings_amount_l2148_214893

def monthly_salary : ℕ := 6500
def initial_savings_rate : ℚ := 1/5
def expense_increase_rate : ℚ := 1/5

theorem new_savings_amount :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_expenses * expense_increase_rate
  let new_expenses := initial_expenses + expense_increase
  let new_savings := monthly_salary - new_expenses
  new_savings = 260 := by sorry

end new_savings_amount_l2148_214893


namespace cubic_roots_sum_l2148_214871

theorem cubic_roots_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  s^3 - 15*s^2 + 25*s - 10 = 0 →
  t^3 - 15*t^2 + 25*t - 10 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 175/11 := by
sorry

end cubic_roots_sum_l2148_214871


namespace quadratic_completion_l2148_214866

theorem quadratic_completion (b : ℝ) (p : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 1 = (x + p)^2 - 7/4) → 
  b = Real.sqrt 11 := by
sorry

end quadratic_completion_l2148_214866


namespace circle_center_sum_l2148_214892

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y + 13

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 13) / 2

theorem circle_center_sum :
  ∀ h k : ℝ, CircleCenter h k CircleEquation → h + k = 7 :=
by sorry

end circle_center_sum_l2148_214892


namespace rectangular_floor_length_l2148_214801

theorem rectangular_floor_length (floor_width : ℝ) (square_size : ℝ) (num_squares : ℕ) :
  floor_width = 6 →
  square_size = 2 →
  num_squares = 15 →
  floor_width * (num_squares * square_size^2 / floor_width) = 10 :=
by sorry

end rectangular_floor_length_l2148_214801


namespace simplify_expression_1_simplify_expression_2_l2148_214897

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  y * (x + y) + (x + y) * (x - y) = x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) (h1 : m ≠ -1) (h2 : m^2 + 2*m + 1 ≠ 0) :
  ((2*m + 1) / (m + 1) + m - 1) / ((m + 2) / (m^2 + 2*m + 1)) = m^2 + m := by sorry

end simplify_expression_1_simplify_expression_2_l2148_214897


namespace solution_set_l2148_214827

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := 2 * a - b + 3

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  otimes 0.5 x > -2 ∧ otimes (2 * x) 5 > 3 * x + 1

-- State the theorem
theorem solution_set :
  ∀ x : ℝ, inequality_system x ↔ 3 < x ∧ x < 6 := by sorry

end solution_set_l2148_214827


namespace intersection_of_P_and_Q_l2148_214876

def P : Set ℕ := {x : ℕ | x * (x - 3) ≤ 0}
def Q : Set ℕ := {x : ℕ | x ≥ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {2, 3} := by
  sorry

end intersection_of_P_and_Q_l2148_214876


namespace f_is_odd_ellipse_y_axis_iff_l2148_214817

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^2))

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Define the ellipse equation
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1

-- Theorem 2: Necessary and sufficient condition for ellipse with foci on y-axis
theorem ellipse_y_axis_iff (m n : ℝ) : 
  is_ellipse_y_axis m n ↔ m > n ∧ n > 0 := by sorry

end f_is_odd_ellipse_y_axis_iff_l2148_214817


namespace medium_supermarkets_sample_l2148_214859

/-- Represents the number of supermarkets to be sampled -/
def sample_size : ℕ := 200

/-- Represents the number of large supermarkets -/
def large_supermarkets : ℕ := 200

/-- Represents the number of medium supermarkets -/
def medium_supermarkets : ℕ := 400

/-- Represents the number of small supermarkets -/
def small_supermarkets : ℕ := 1400

/-- Represents the total number of supermarkets -/
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets

/-- Theorem stating that the number of medium supermarkets to be sampled is 40 -/
theorem medium_supermarkets_sample :
  (sample_size : ℚ) * medium_supermarkets / total_supermarkets = 40 := by
  sorry

end medium_supermarkets_sample_l2148_214859


namespace complex_sum_theorem_l2148_214819

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 6 →
  g = -2*a - c - e →
  (2*a + b*Complex.I) + (c + 2*d*Complex.I) + (e + f*Complex.I) + (g + 2*h*Complex.I) = 8*Complex.I →
  d + f + h = 3/2 := by
sorry

end complex_sum_theorem_l2148_214819


namespace green_team_score_l2148_214863

theorem green_team_score (other_team_score lead : ℕ) (h1 : other_team_score = 68) (h2 : lead = 29) :
  ∃ G : ℕ, other_team_score = G + lead ∧ G = 39 := by
  sorry

end green_team_score_l2148_214863


namespace quadratic_two_distinct_roots_l2148_214884

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 :=
by sorry

end quadratic_two_distinct_roots_l2148_214884


namespace line_equivalence_l2148_214833

/-- Definition of the line using dot product equation -/
def line_equation (x y : ℝ) : Prop :=
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0

/-- Slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

/-- The slope and y-intercept of the line -/
def slope_intercept_params : ℝ × ℝ := (2, -10)

theorem line_equivalence :
  ∀ (x y : ℝ),
    line_equation x y ↔ slope_intercept_form (slope_intercept_params.1) (slope_intercept_params.2) x y :=
by sorry

#check line_equivalence

end line_equivalence_l2148_214833


namespace triangle_area_implies_cd_one_l2148_214821

theorem triangle_area_implies_cd_one (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_line : ∀ x y, 2*c*x + 3*d*y = 12 → x ≥ 0 ∧ y ≥ 0)
  (h_area : (1/2) * (6/c) * (4/d) = 12) : c * d = 1 := by
sorry

end triangle_area_implies_cd_one_l2148_214821


namespace gino_bears_count_l2148_214844

theorem gino_bears_count (total : ℕ) (brown : ℕ) (white : ℕ) (black : ℕ) : 
  total = 66 → brown = 15 → white = 24 → total = brown + white + black → black = 27 := by
sorry

end gino_bears_count_l2148_214844


namespace intersection_of_A_and_B_l2148_214838

def A : Set ℝ := {x | 2 * x - 1 > 0}
def B : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1/2 < x ∧ x < 2} := by
sorry

end intersection_of_A_and_B_l2148_214838


namespace c_work_days_l2148_214889

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def work_rate_ABC : ℚ := 1 / 4

-- Define C's work rate as a function of x (days C needs to complete the work)
def work_rate_C (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem c_work_days :
  ∃ x : ℚ, x = 16 ∧ work_rate_A + work_rate_B + work_rate_C x = work_rate_ABC :=
sorry

end c_work_days_l2148_214889


namespace triangle_sequence_2009_position_l2148_214856

def triangle_sequence (n : ℕ) : ℕ := n

def row_of_term (n : ℕ) : ℕ :=
  (n.sqrt : ℕ) + 1

def position_in_row (n : ℕ) : ℕ :=
  n - (row_of_term n - 1)^2

theorem triangle_sequence_2009_position :
  row_of_term 2009 = 45 ∧ position_in_row 2009 = 73 := by
  sorry

end triangle_sequence_2009_position_l2148_214856


namespace f_shifted_l2148_214800

/-- Given a function f(x) = 3x - 5, prove that f(x - 4) = 3x - 17 for any real number x -/
theorem f_shifted (x : ℝ) : (fun x => 3 * x - 5) (x - 4) = 3 * x - 17 := by
  sorry

end f_shifted_l2148_214800


namespace clearance_sale_earnings_l2148_214888

/-- Calculates the total earnings from a clearance sale of winter jackets --/
theorem clearance_sale_earnings 
  (total_jackets : ℕ)
  (price_before_noon : ℚ)
  (price_after_noon : ℚ)
  (jackets_sold_after_noon : ℕ)
  (h1 : total_jackets = 214)
  (h2 : price_before_noon = 31.95)
  (h3 : price_after_noon = 18.95)
  (h4 : jackets_sold_after_noon = 133) :
  (total_jackets - jackets_sold_after_noon) * price_before_noon +
  jackets_sold_after_noon * price_after_noon = 5107.30 := by
  sorry


end clearance_sale_earnings_l2148_214888


namespace special_functions_identity_l2148_214841

/-- Non-constant, differentiable functions satisfying certain conditions -/
class SpecialFunctions (f g : ℝ → ℝ) where
  non_constant_f : ∃ x y, f x ≠ f y
  non_constant_g : ∃ x y, g x ≠ g y
  differentiable_f : Differentiable ℝ f
  differentiable_g : Differentiable ℝ g
  condition1 : ∀ x y, f (x + y) = f x * f y - g x * g y
  condition2 : ∀ x y, g (x + y) = f x * g y + g x * f y
  condition3 : deriv f 0 = 0

/-- Theorem stating that f(x)^2 + g(x)^2 = 1 for all x ∈ ℝ -/
theorem special_functions_identity {f g : ℝ → ℝ} [SpecialFunctions f g] :
  ∀ x, f x ^ 2 + g x ^ 2 = 1 := by
  sorry

end special_functions_identity_l2148_214841


namespace candidate_X_votes_and_result_l2148_214855

-- Define the number of votes for each candidate
def votes_Z : ℕ := 25000
def votes_Y : ℕ := (3 * votes_Z) / 5
def votes_X : ℕ := (3 * votes_Y) / 2

-- Define the winning threshold
def winning_threshold : ℕ := 30000

-- Theorem to prove
theorem candidate_X_votes_and_result : 
  votes_X = 22500 ∧ votes_X < winning_threshold :=
by sorry

end candidate_X_votes_and_result_l2148_214855


namespace sixth_term_value_l2148_214845

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end sixth_term_value_l2148_214845


namespace codecracker_combinations_l2148_214879

/-- The number of colors available in the CodeCracker game -/
def num_colors : ℕ := 8

/-- The number of slots in a CodeCracker code -/
def num_slots : ℕ := 4

/-- Theorem stating the total number of possible codes in CodeCracker -/
theorem codecracker_combinations : (num_colors ^ num_slots : ℕ) = 4096 := by
  sorry

end codecracker_combinations_l2148_214879


namespace ellipse_m_value_l2148_214811

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : a^2 - b^2 = c^2

/-- The given ellipse with a = 5 and left focus at (-4, 0). -/
def given_ellipse (m : ℝ) : Ellipse :=
  { a := 5
    b := m
    c := 4
    h_positive := by sorry
    h_relation := by sorry }

/-- Theorem stating that m = 3 for the given ellipse. -/
theorem ellipse_m_value :
  ∀ m > 0, (given_ellipse m).b = 3 := by sorry

end ellipse_m_value_l2148_214811


namespace power_of_two_equality_l2148_214824

theorem power_of_two_equality (n : ℕ) : 2^n = 2 * 16^2 * 64^3 → n = 27 := by
  sorry

end power_of_two_equality_l2148_214824


namespace cubic_polynomial_root_l2148_214870

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 5 (1/3) + 1 →
  x^3 - 3*x^2 + 3*x - 6 = 0 ∧ 
  (∃ (a b c : ℤ), x^3 - 3*x^2 + 3*x - 6 = x^3 + a*x^2 + b*x + c) := by
  sorry

end cubic_polynomial_root_l2148_214870


namespace shortest_side_is_eight_l2148_214877

/-- Represents a rectangular solid with sides in geometric progression -/
structure GeometricSolid where
  b : ℝ
  s : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq : volume = b^3 / s
  surface_area_eq : surface_area = 2 * (b^2 / s + b^2 * s + b^2)

/-- The shortest side length of a geometric solid with given properties is 8 -/
theorem shortest_side_is_eight (solid : GeometricSolid)
  (h_volume : solid.volume = 512)
  (h_surface_area : solid.surface_area = 384) :
  min (solid.b / solid.s) (min solid.b (solid.b * solid.s)) = 8 := by
  sorry

#check shortest_side_is_eight

end shortest_side_is_eight_l2148_214877


namespace arithmetic_sequence_length_l2148_214894

theorem arithmetic_sequence_length
  (a : ℤ)  -- First term
  (l : ℤ)  -- Last term
  (d : ℤ)  -- Common difference
  (h1 : a = -22)
  (h2 : l = 50)
  (h3 : d = 7)
  : (l - a) / d + 1 = 11 :=
by sorry

end arithmetic_sequence_length_l2148_214894


namespace nicoles_clothes_theorem_l2148_214815

/-- Calculates the total number of clothing pieces Nicole ends up with --/
def nicoles_total_clothes (nicole_start : ℕ) : ℕ :=
  let first_sister := nicole_start / 3
  let second_sister := nicole_start + 5
  let third_sister := 2 * first_sister
  let youngest_four_total := nicole_start + first_sister + second_sister + third_sister
  let oldest_sister := (youngest_four_total / 4 * 3 + (youngest_four_total % 4) / 2 + 1) / 2
  nicole_start + first_sister + second_sister + third_sister + oldest_sister

theorem nicoles_clothes_theorem :
  nicoles_total_clothes 15 = 69 := by
  sorry

end nicoles_clothes_theorem_l2148_214815


namespace snow_probability_l2148_214895

theorem snow_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/4) (h2 : p2 = 1/2) (h3 : p3 = 1/3) :
  1 - (1 - p1)^2 * (1 - p2)^3 * (1 - p3)^2 = 31/32 := by
  sorry

end snow_probability_l2148_214895


namespace quadratic_inequality_solution_set_l2148_214818

theorem quadratic_inequality_solution_set (m : ℝ) : 
  m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by sorry

end quadratic_inequality_solution_set_l2148_214818


namespace square_division_perimeter_l2148_214831

/-- Given a square with perimeter 160 units, when divided into two congruent rectangles
    horizontally and one of those rectangles is further divided into two congruent rectangles
    vertically, the perimeter of one of the smaller rectangles is 80 units. -/
theorem square_division_perimeter :
  ∀ (s : ℝ),
  s > 0 →
  4 * s = 160 →
  let horizontal_rectangle_width := s
  let horizontal_rectangle_height := s / 2
  let vertical_rectangle_width := s / 2
  let vertical_rectangle_height := s / 2
  2 * (vertical_rectangle_width + vertical_rectangle_height) = 80 :=
by
  sorry

#check square_division_perimeter

end square_division_perimeter_l2148_214831


namespace fraction_equality_l2148_214857

theorem fraction_equality : (36 + 12) / (6 - 3) = 16 := by
  sorry

end fraction_equality_l2148_214857


namespace sunday_no_arguments_l2148_214830

/-- Probability of a spouse arguing with their mother-in-law -/
def p_argue_with_mil : ℚ := 2/3

/-- Probability of siding with own mother in case of conflict -/
def p_side_with_mother : ℚ := 1/2

/-- Probability of no arguments between spouses on a Sunday -/
def p_no_arguments : ℚ := 4/9

theorem sunday_no_arguments : 
  p_no_arguments = 1 - (2 * p_argue_with_mil * p_side_with_mother - (p_argue_with_mil * p_side_with_mother)^2) := by
  sorry

end sunday_no_arguments_l2148_214830


namespace volunteer_arrangement_count_l2148_214886

/-- The number of volunteers --/
def n : ℕ := 6

/-- The number of exhibition areas --/
def m : ℕ := 4

/-- The number of areas that require one person --/
def single_person_areas : ℕ := 2

/-- The number of areas that require two people --/
def double_person_areas : ℕ := 2

/-- The number of specific volunteers that cannot be together --/
def restricted_volunteers : ℕ := 2

/-- The total number of arrangements without restrictions --/
def total_arrangements : ℕ := 180

/-- The number of arrangements where the restricted volunteers are together --/
def restricted_arrangements : ℕ := 24

theorem volunteer_arrangement_count :
  (n = 6) →
  (m = 4) →
  (single_person_areas = 2) →
  (double_person_areas = 2) →
  (restricted_volunteers = 2) →
  (total_arrangements = 180) →
  (restricted_arrangements = 24) →
  (total_arrangements - restricted_arrangements = 156) := by
  sorry

end volunteer_arrangement_count_l2148_214886


namespace chess_club_committee_probability_l2148_214854

def total_members : ℕ := 27
def boys : ℕ := 15
def girls : ℕ := 12
def committee_size : ℕ := 5

theorem chess_club_committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let all_boys_committees := Nat.choose boys committee_size
  let all_girls_committees := Nat.choose girls committee_size
  let favorable_committees := total_committees - (all_boys_committees + all_girls_committees)
  (favorable_committees : ℚ) / total_committees = 76935 / 80730 := by sorry

end chess_club_committee_probability_l2148_214854


namespace tea_price_calculation_l2148_214861

theorem tea_price_calculation (coffee_customers : ℕ) (tea_customers : ℕ) (coffee_price : ℚ) (total_revenue : ℚ) :
  coffee_customers = 7 →
  tea_customers = 8 →
  coffee_price = 5 →
  total_revenue = 67 →
  ∃ tea_price : ℚ, tea_price = 4 ∧ coffee_customers * coffee_price + tea_customers * tea_price = total_revenue :=
by sorry

end tea_price_calculation_l2148_214861


namespace typists_problem_l2148_214862

theorem typists_problem (initial_letters : ℕ) (initial_time : ℕ) (new_typists : ℕ) (new_letters : ℕ) (new_time : ℕ) :
  initial_letters = 48 →
  initial_time = 20 →
  new_typists = 30 →
  new_letters = 216 →
  new_time = 60 →
  ∃ x : ℕ, x > 0 ∧ (initial_letters / x : ℚ) * new_typists * (new_time / initial_time : ℚ) = new_letters :=
by
  sorry

end typists_problem_l2148_214862


namespace power_sum_equals_39_l2148_214853

theorem power_sum_equals_39 : 
  (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 3 + 2^1 + 2^2 + 2^3 + 2^4 = 39 := by
  sorry

end power_sum_equals_39_l2148_214853


namespace obtuse_triangle_one_obtuse_angle_equilateral_triangle_60_degree_angles_l2148_214829

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180

-- Define an obtuse triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i > 90

-- Define an equilateral triangle
def EquilateralTriangle (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∧ t.angles 1 = t.angles 2

theorem obtuse_triangle_one_obtuse_angle (t : Triangle) (h : ObtuseTriangle t) :
  ∃! i : Fin 3, t.angles i > 90 :=
sorry

theorem equilateral_triangle_60_degree_angles (t : Triangle) (h : EquilateralTriangle t) :
  ∀ i : Fin 3, t.angles i = 60 :=
sorry

end obtuse_triangle_one_obtuse_angle_equilateral_triangle_60_degree_angles_l2148_214829
