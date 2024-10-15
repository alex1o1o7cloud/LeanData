import Mathlib

namespace NUMINAMATH_CALUDE_m_range_theorem_l1518_151844

/-- Proposition p: There exists x ∈ ℝ, such that 2x² + (m-1)x + 1/2 ≤ 0 -/
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

/-- Proposition q: The curve C₁: x²/m² + y²/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def q (m : ℝ) : Prop := m ≠ 0 ∧ 2 * m + 8 > 0 ∧ m^2 > 2 * m + 8

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  (3 ≤ m ∧ m ≤ 4) ∨ (-2 ≤ m ∧ m ≤ -1) ∨ m ≤ -4

theorem m_range_theorem (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m := by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1518_151844


namespace NUMINAMATH_CALUDE_polynomial_at_most_one_zero_l1518_151812

theorem polynomial_at_most_one_zero (n : ℤ) :
  ∃! (r : ℝ), r^4 - 1994*r^3 + (1993 + n : ℝ)*r^2 - 11*r + (n : ℝ) = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_at_most_one_zero_l1518_151812


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1518_151839

theorem regular_polygon_sides (n : ℕ) : 
  n > 2 → 
  (180 * (n - 2) : ℝ) / n = 160 → 
  n = 18 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1518_151839


namespace NUMINAMATH_CALUDE_divide_powers_l1518_151879

theorem divide_powers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  2 * x^2 * y^3 / (x * y^2) = 2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_divide_powers_l1518_151879


namespace NUMINAMATH_CALUDE_mary_snake_count_l1518_151863

/-- The number of breeding balls -/
def num_breeding_balls : ℕ := 3

/-- The number of snakes in each breeding ball -/
def snakes_per_ball : ℕ := 8

/-- The number of additional pairs of snakes -/
def num_snake_pairs : ℕ := 6

/-- The total number of snakes Mary saw -/
def total_snakes : ℕ := num_breeding_balls * snakes_per_ball + 2 * num_snake_pairs

theorem mary_snake_count : total_snakes = 36 := by
  sorry

end NUMINAMATH_CALUDE_mary_snake_count_l1518_151863


namespace NUMINAMATH_CALUDE_workshop_novelists_l1518_151856

theorem workshop_novelists (total : ℕ) (ratio_novelists : ℕ) (ratio_poets : ℕ) 
  (h1 : total = 24)
  (h2 : ratio_novelists = 5)
  (h3 : ratio_poets = 3) :
  (total * ratio_novelists) / (ratio_novelists + ratio_poets) = 15 := by
  sorry

end NUMINAMATH_CALUDE_workshop_novelists_l1518_151856


namespace NUMINAMATH_CALUDE_least_triangle_area_l1518_151804

/-- The solutions of the equation (z+4)^10 = 32 form a convex regular decagon in the complex plane. -/
def is_solution (z : ℂ) : Prop := (z + 4) ^ 10 = 32

/-- The set of all solutions forms a convex regular decagon. -/
def solution_set : Set ℂ := {z | is_solution z}

/-- A point is a vertex of the decagon if it's a solution. -/
def is_vertex (z : ℂ) : Prop := z ∈ solution_set

/-- The area of a triangle formed by three vertices of the decagon. -/
def triangle_area (v1 v2 v3 : ℂ) : ℝ :=
  sorry -- Definition of the area calculation

/-- The theorem stating the least possible area of a triangle formed by three vertices. -/
theorem least_triangle_area :
  ∃ (v1 v2 v3 : ℂ), is_vertex v1 ∧ is_vertex v2 ∧ is_vertex v3 ∧
    (∀ (w1 w2 w3 : ℂ), is_vertex w1 → is_vertex w2 → is_vertex w3 →
      triangle_area v1 v2 v3 ≤ triangle_area w1 w2 w3) ∧
    triangle_area v1 v2 v3 = (2^(2/5) * (Real.sqrt 5 - 1)) / 8 :=
sorry

end NUMINAMATH_CALUDE_least_triangle_area_l1518_151804


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1518_151875

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + 1 / (5 * a * b * c * d) ≥ 4 * Real.sqrt 10 / 5 :=
by sorry

theorem min_value_achievable :
  ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + 1 / (5 * a * b * c * d) = 4 * Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1518_151875


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l1518_151855

/-- Given three rugs with a combined area of 200 square meters, prove that the area
    covered by exactly two layers of rug is 5 square meters when:
    1. The rugs cover a floor area of 138 square meters when overlapped.
    2. The area covered by exactly some layers of rug is 24 square meters.
    3. The area covered by three layers of rug is 19 square meters. -/
theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (some_layers_area : ℝ) (three_layers_area : ℝ)
    (h1 : total_area = 200)
    (h2 : covered_area = 138)
    (h3 : some_layers_area = 24)
    (h4 : three_layers_area = 19) :
    total_area - (covered_area + some_layers_area) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l1518_151855


namespace NUMINAMATH_CALUDE_soup_ingredients_weights_l1518_151864

/-- Represents the ingredients of the soup --/
structure SoupIngredients where
  fat : ℝ
  onion : ℝ
  potatoes : ℝ
  grain : ℝ
  water : ℝ

/-- The conditions of the soup recipe --/
def SoupConditions (s : SoupIngredients) : Prop :=
  s.water = s.grain + s.potatoes + s.onion + s.fat ∧
  s.grain = s.potatoes + s.onion + s.fat ∧
  s.potatoes = s.onion + s.fat ∧
  s.fat = s.onion / 2 ∧
  s.water + s.grain + s.potatoes + s.onion + s.fat = 12

/-- The theorem stating the correct weights of the ingredients --/
theorem soup_ingredients_weights :
  ∃ (s : SoupIngredients),
    SoupConditions s ∧
    s.fat = 0.5 ∧
    s.onion = 1 ∧
    s.potatoes = 1.5 ∧
    s.grain = 3 ∧
    s.water = 6 :=
  sorry

end NUMINAMATH_CALUDE_soup_ingredients_weights_l1518_151864


namespace NUMINAMATH_CALUDE_inequality_proof_l1518_151805

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + a^2) ≥ Real.sqrt 2 * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1518_151805


namespace NUMINAMATH_CALUDE_railroad_cars_theorem_l1518_151858

/-- Sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Minimum number of tries needed to determine if there is an equal number of both types of cars -/
def minTries (totalCars : ℕ) : ℕ := totalCars - sumBinaryDigits totalCars

theorem railroad_cars_theorem :
  let totalCars : ℕ := 2022
  minTries totalCars = 2014 := by sorry

end NUMINAMATH_CALUDE_railroad_cars_theorem_l1518_151858


namespace NUMINAMATH_CALUDE_monica_subjects_l1518_151882

theorem monica_subjects (monica marius millie : ℕ) 
  (h1 : millie = marius + 3)
  (h2 : marius = monica + 4)
  (h3 : monica + marius + millie = 41) : 
  monica = 10 :=
sorry

end NUMINAMATH_CALUDE_monica_subjects_l1518_151882


namespace NUMINAMATH_CALUDE_james_living_room_set_price_l1518_151828

/-- The final price James paid for the living room set after discount -/
theorem james_living_room_set_price (coach : ℝ) (sectional : ℝ) (other : ℝ) 
  (h1 : coach = 2500)
  (h2 : sectional = 3500)
  (h3 : other = 2000)
  (discount_rate : ℝ) 
  (h4 : discount_rate = 0.1) : 
  (coach + sectional + other) * (1 - discount_rate) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_james_living_room_set_price_l1518_151828


namespace NUMINAMATH_CALUDE_probability_sum_less_than_product_l1518_151868

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_number (n : ℕ) : Prop := is_even n ∧ 0 < n ∧ n ≤ 10

def valid_pair (a b : ℕ) : Prop := valid_number a ∧ valid_number b ∧ a + b < a * b

def total_pairs : ℕ := 25

def valid_pairs : ℕ := 16

theorem probability_sum_less_than_product :
  (valid_pairs : ℚ) / total_pairs = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_product_l1518_151868


namespace NUMINAMATH_CALUDE_non_negative_product_l1518_151846

theorem non_negative_product (a b c d e f g h : ℝ) :
  (max (ac + bd) (max (ae + bf) (max (ag + bh) (max (ce + df) (max (cg + dh) (eg + fh)))))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_product_l1518_151846


namespace NUMINAMATH_CALUDE_divisibility_by_17_l1518_151801

theorem divisibility_by_17 (x y : ℤ) : 
  (∃ k : ℤ, 2*x + 3*y = 17*k) → (∃ m : ℤ, 9*x + 5*y = 17*m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_17_l1518_151801


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1518_151841

theorem geometric_sequence_fourth_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
    a 1 = 6 →                            -- First term
    a 5 = 1458 →                         -- Last term
    a 4 = 162 :=                         -- Fourth term to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1518_151841


namespace NUMINAMATH_CALUDE_max_value_of_function_l1518_151842

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  ∃ (max_y : ℝ), max_y = 1/8 ∧ ∀ y, y = x * (1 - 2*x) → y ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1518_151842


namespace NUMINAMATH_CALUDE_ages_four_years_ago_l1518_151800

/-- Represents the ages of four people: Amar, Akbar, Anthony, and Alex -/
structure Ages :=
  (amar : ℕ)
  (akbar : ℕ)
  (anthony : ℕ)
  (alex : ℕ)

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.amar + ages.akbar + ages.anthony + ages.alex = 88 ∧
  (ages.amar - 4) + (ages.akbar - 4) + (ages.anthony - 4) = 66 ∧
  ages.amar = 2 * ages.alex ∧
  ages.akbar = ages.amar - 3

/-- The theorem to be proved -/
theorem ages_four_years_ago (ages : Ages) 
  (h : satisfies_conditions ages) : 
  (ages.amar - 4) + (ages.akbar - 4) + (ages.anthony - 4) + (ages.alex - 4) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ages_four_years_ago_l1518_151800


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_fifth_power_l1518_151867

theorem imaginary_part_of_one_plus_i_fifth_power (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 : ℂ) + i)^5 = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_fifth_power_l1518_151867


namespace NUMINAMATH_CALUDE_children_left_on_bus_l1518_151892

theorem children_left_on_bus (initial_children : Nat) (children_off : Nat) : 
  initial_children = 43 → children_off = 22 → initial_children - children_off = 21 := by
  sorry

end NUMINAMATH_CALUDE_children_left_on_bus_l1518_151892


namespace NUMINAMATH_CALUDE_simplify_expression_l1518_151899

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1518_151899


namespace NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1518_151806

/-- Represents the weight ratio problem of Jake and his sister -/
theorem jake_sister_weight_ratio :
  let jake_present_weight : ℕ := 108
  let total_weight : ℕ := 156
  let weight_loss : ℕ := 12
  let jake_new_weight : ℕ := jake_present_weight - weight_loss
  let sister_weight : ℕ := total_weight - jake_new_weight
  (jake_new_weight : ℚ) / sister_weight = 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1518_151806


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1518_151884

/-- The perimeter of a rhombus with diagonals of 12 inches and 16 inches is 40 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1518_151884


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l1518_151860

/-- Given that real numbers 4, m, 9 form a geometric sequence,
    prove that the eccentricity of the conic section x^2/m + y^2 = 1
    is either √30/6 or √7 -/
theorem conic_section_eccentricity (m : ℝ) 
  (h_geom_seq : (4 : ℝ) * 9 = m^2) :
  let e := if m > 0 
    then Real.sqrt (1 - m / 6) / Real.sqrt (m / 6)
    else Real.sqrt (1 + 6 / m) / 1
  e = Real.sqrt 30 / 6 ∨ e = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l1518_151860


namespace NUMINAMATH_CALUDE_min_value_of_function_l1518_151848

theorem min_value_of_function (x : ℝ) (h : x > 5/4) : 
  ∀ y : ℝ, y = 4*x + 1/(4*x - 5) → y ≥ 7 := by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1518_151848


namespace NUMINAMATH_CALUDE_difference_d_minus_b_l1518_151830

theorem difference_d_minus_b (a b c d : ℕ+) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_difference_d_minus_b_l1518_151830


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l1518_151849

theorem hyperbola_ellipse_shared_foci (m : ℝ) : 
  (∃ (c : ℝ), c > 0 ∧ 
    c^2 = 12 - 4 ∧ 
    c^2 = m + 1) → 
  m = 7 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l1518_151849


namespace NUMINAMATH_CALUDE_road_construction_progress_l1518_151825

theorem road_construction_progress (total_length : ℚ) 
  (h1 : total_length = 1/2) 
  (day1_progress : ℚ) (h2 : day1_progress = 1/10)
  (day2_progress : ℚ) (h3 : day2_progress = 1/5) :
  1 - day1_progress - day2_progress = 7/10 := by
sorry

end NUMINAMATH_CALUDE_road_construction_progress_l1518_151825


namespace NUMINAMATH_CALUDE_at_least_two_correct_coats_l1518_151861

theorem at_least_two_correct_coats (n : ℕ) (h : n = 5) : 
  (Finset.sum (Finset.range (n - 1)) (λ k => (n.choose (k + 2)) * ((n - k - 2).factorial))) = 31 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_correct_coats_l1518_151861


namespace NUMINAMATH_CALUDE_special_square_PT_l1518_151890

/-- A square with side length 2 and special points T and U -/
structure SpecialSquare where
  -- Point P is at (0, 0), Q at (2, 0), R at (2, 2), and S at (0, 2)
  T : ℝ × ℝ  -- Point on PQ
  U : ℝ × ℝ  -- Point on SQ
  h_T_on_PQ : T.1 ∈ Set.Icc 0 2 ∧ T.2 = 0
  h_U_on_SQ : U.1 = 2 ∧ U.2 ∈ Set.Icc 0 2
  h_PT_eq_QU : T.1 = 2 - U.2  -- PT = QU
  h_fold : (2 - T.1)^2 + T.1^2 = 8  -- Condition for PR and SR to coincide with RQ when folded

theorem special_square_PT (s : SpecialSquare) : s.T.1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_square_PT_l1518_151890


namespace NUMINAMATH_CALUDE_tan_sum_identity_l1518_151845

theorem tan_sum_identity : 
  (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 
  2 + Real.tan (23 * π / 180) * Real.tan (22 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l1518_151845


namespace NUMINAMATH_CALUDE_ellipse_parabola_properties_l1518_151803

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focal length c -/
structure Parabola where
  c : ℝ
  h_pos : 0 < c

/-- The configuration of the ellipse and parabola as described in the problem -/
structure Configuration where
  C₁ : Ellipse
  C₂ : Parabola
  h_focus : C₂.c = C₁.a - C₁.b  -- Right focus of C₁ coincides with focus of C₂
  h_center : C₁.a = 2 * C₂.c    -- Center of C₁ coincides with vertex of C₂
  h_chord_ratio : 3 * C₂.c = 2 * C₁.b^2 / C₁.a  -- |CD| = 4/3 |AB|
  h_vertex_sum : C₁.a + C₂.c = 6  -- Sum of distances from vertices to directrix is 12

/-- The main theorem stating the properties to be proved -/
theorem ellipse_parabola_properties (config : Configuration) :
  config.C₁.a = 4 ∧ 
  config.C₁.b^2 = 12 ∧ 
  config.C₂.c = 2 ∧
  (config.C₁.a - config.C₁.b) / config.C₁.a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_properties_l1518_151803


namespace NUMINAMATH_CALUDE_investment_ratio_l1518_151871

/-- 
Given two investors p and q who divide their profit in the ratio 4:5,
prove that if p invested 52000, then q invested 65000.
-/
theorem investment_ratio (p q : ℕ) (h1 : p = 52000) : 
  (p : ℚ) / q = 4 / 5 → q = 65000 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_l1518_151871


namespace NUMINAMATH_CALUDE_square_minus_four_l1518_151809

theorem square_minus_four (y : ℤ) (h : y^2 = 2209) : (y + 2) * (y - 2) = 2205 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_four_l1518_151809


namespace NUMINAMATH_CALUDE_pipe_filling_time_l1518_151807

theorem pipe_filling_time (t_b t_ab : ℝ) (h_b : t_b = 20) (h_ab : t_ab = 20/3) :
  let t_a := (t_b * t_ab) / (t_b - t_ab)
  t_a = 10 := by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l1518_151807


namespace NUMINAMATH_CALUDE_candy_bar_sales_l1518_151837

theorem candy_bar_sales (seth_sales : ℕ) (other_sales : ℕ) 
  (h1 : seth_sales = 3 * other_sales + 6) 
  (h2 : seth_sales = 78) : 
  other_sales = 24 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l1518_151837


namespace NUMINAMATH_CALUDE_joshuas_share_l1518_151817

theorem joshuas_share (total : ℕ) (joshua_share : ℕ) (justin_share : ℕ) : 
  total = 40 → 
  joshua_share = 3 * justin_share → 
  total = joshua_share + justin_share → 
  joshua_share = 30 := by
sorry

end NUMINAMATH_CALUDE_joshuas_share_l1518_151817


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1518_151854

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let A := ((x - y) * Real.sqrt (x^2 + y^2) + 
            (y - z) * Real.sqrt (y^2 + z^2) + 
            (z - x) * Real.sqrt (z^2 + x^2) + 
            Real.sqrt 2) / 
           ((x - y)^2 + (y - z)^2 + (z - x)^2 + 2)
  A ≤ 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1518_151854


namespace NUMINAMATH_CALUDE_max_perimeter_of_divided_isosceles_triangle_l1518_151818

/-- The maximum perimeter of a piece when an isosceles triangle is divided into four equal areas -/
theorem max_perimeter_of_divided_isosceles_triangle :
  let base : ℝ := 12
  let height : ℝ := 15
  let segment_length : ℝ := base / 4
  let perimeter (k : ℝ) : ℝ := segment_length + Real.sqrt (height^2 + k^2) + Real.sqrt (height^2 + (k + 1)^2)
  let max_perimeter : ℝ := perimeter 2
  max_perimeter = 3 + Real.sqrt 229 + Real.sqrt 234 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_divided_isosceles_triangle_l1518_151818


namespace NUMINAMATH_CALUDE_brandon_skittles_proof_l1518_151891

def brandon_initial_skittles (skittles_lost : ℕ) (final_skittles : ℕ) : ℕ :=
  final_skittles + skittles_lost

theorem brandon_skittles_proof :
  brandon_initial_skittles 9 87 = 96 :=
by sorry

end NUMINAMATH_CALUDE_brandon_skittles_proof_l1518_151891


namespace NUMINAMATH_CALUDE_product_in_base7_l1518_151876

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Converts a base 7 number to base 10 --/
def fromBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := sorry

theorem product_in_base7 : 
  multiplyBase7 (toBase7 231) (toBase7 452) = 613260 := by sorry

end NUMINAMATH_CALUDE_product_in_base7_l1518_151876


namespace NUMINAMATH_CALUDE_distribute_5_2_l1518_151816

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_5_2 : distribute 5 2 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_5_2_l1518_151816


namespace NUMINAMATH_CALUDE_dinner_cost_bret_dinner_cost_l1518_151888

theorem dinner_cost (people : ℕ) (main_meal_cost appetizer_cost : ℚ) 
  (appetizers : ℕ) (tip_percentage : ℚ) (rush_fee : ℚ) : ℚ :=
  let main_meals_total := people * main_meal_cost
  let appetizers_total := appetizers * appetizer_cost
  let subtotal := main_meals_total + appetizers_total
  let tip := tip_percentage * subtotal
  let total := subtotal + tip + rush_fee
  total

theorem bret_dinner_cost : 
  dinner_cost 4 12 6 2 (20/100) 5 = 77 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_bret_dinner_cost_l1518_151888


namespace NUMINAMATH_CALUDE_maria_score_l1518_151898

def test_score (total_questions : ℕ) (correct_points : ℕ) (incorrect_deduction : ℕ) (correct_answers : ℕ) : ℤ :=
  (correct_answers * correct_points : ℤ) - ((total_questions - correct_answers) * incorrect_deduction)

theorem maria_score :
  test_score 30 20 5 19 = 325 := by
  sorry

end NUMINAMATH_CALUDE_maria_score_l1518_151898


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1518_151866

/-- Given a quadratic expression x^2 - 20x + 100 that can be written in the form (x + b)^2 + c,
    prove that b + c = -10 -/
theorem quadratic_sum_of_constants (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 100 = (x + b)^2 + c) → b + c = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1518_151866


namespace NUMINAMATH_CALUDE_product_first_three_terms_l1518_151887

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_first_three_terms 
  (a : ℕ → ℕ) 
  (h1 : arithmetic_sequence a 2)
  (h2 : a 7 = 20) : 
  a 1 * a 2 * a 3 = 960 := by
sorry

end NUMINAMATH_CALUDE_product_first_three_terms_l1518_151887


namespace NUMINAMATH_CALUDE_union_of_sets_l1518_151834

theorem union_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∪ N = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1518_151834


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l1518_151820

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- A sphere touching three faces of a regular tetrahedron and three sides of its fourth face -/
structure InscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  touches_three_faces : True  -- Placeholder for the condition
  touches_three_sides_of_fourth_face : True  -- Placeholder for the condition

/-- The radius of the inscribed sphere is √6/8 -/
theorem inscribed_sphere_radius (t : RegularTetrahedron) (s : InscribedSphere t) :
  s.radius = Real.sqrt 6 / 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l1518_151820


namespace NUMINAMATH_CALUDE_points_collinear_iff_k_eq_one_l1518_151872

-- Define the vectors
def OA : ℝ × ℝ := (1, -3)
def OB : ℝ × ℝ := (2, -1)
def OC (k : ℝ) : ℝ × ℝ := (k + 1, k - 2)

-- Define collinearity condition
def areCollinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Theorem statement
theorem points_collinear_iff_k_eq_one :
  ∀ k : ℝ, areCollinear OA OB (OC k) ↔ k = 1 := by sorry

end NUMINAMATH_CALUDE_points_collinear_iff_k_eq_one_l1518_151872


namespace NUMINAMATH_CALUDE_connor_hourly_wage_l1518_151813

def sarah_daily_wage : ℝ := 288
def sarah_hours_worked : ℝ := 8
def sarah_connor_wage_ratio : ℝ := 6

theorem connor_hourly_wage :
  let sarah_hourly_wage := sarah_daily_wage / sarah_hours_worked
  sarah_hourly_wage / sarah_connor_wage_ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_connor_hourly_wage_l1518_151813


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_is_eight_l1518_151873

theorem sum_of_w_and_y_is_eight (W X Y Z : ℤ) : 
  W ∈ ({1, 2, 3, 5} : Set ℤ) →
  X ∈ ({1, 2, 3, 5} : Set ℤ) →
  Y ∈ ({1, 2, 3, 5} : Set ℤ) →
  Z ∈ ({1, 2, 3, 5} : Set ℤ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / (X : ℚ) - (Y : ℚ) / (Z : ℚ) = 1 →
  W + Y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_is_eight_l1518_151873


namespace NUMINAMATH_CALUDE_max_area_difference_line_l1518_151843

/-- The line that maximizes the area difference when passing through P(1,1) and dividing the circle (x-2)^2+y^2 ≤ 4 -/
theorem max_area_difference_line (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 2)^2 + b^2 ≤ 4 → 
    (x - y = 0 → 
      ∀ (m n : ℝ), (m + n - 2 = 0 ∨ y - 1 = 0 ∨ m + 3*n - 4 = 0) → 
        (abs ((a - 2)^2 + b^2 - ((a - x)^2 + (b - y)^2)) ≤ 
         abs ((a - 2)^2 + b^2 - ((a - m)^2 + (b - n)^2))))) :=
by sorry

end NUMINAMATH_CALUDE_max_area_difference_line_l1518_151843


namespace NUMINAMATH_CALUDE_triangle_inequality_implies_equilateral_l1518_151822

/-- A triangle with sides a, b, c, area S, and centroid distances x, y, z from the vertices. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The theorem stating that if a triangle satisfies the given inequality, it is equilateral. -/
theorem triangle_inequality_implies_equilateral (t : Triangle) :
  (t.x + t.y + t.z)^2 ≤ (t.a^2 + t.b^2 + t.c^2)/2 + 2*t.S*Real.sqrt 3 →
  t.a = t.b ∧ t.b = t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_implies_equilateral_l1518_151822


namespace NUMINAMATH_CALUDE_symmetric_points_y_coordinate_l1518_151815

/-- Given two points P and Q in a 2D Cartesian coordinate system that are symmetric about the origin,
    prove that the y-coordinate of Q is -3. -/
theorem symmetric_points_y_coordinate
  (P Q : ℝ × ℝ)  -- P and Q are points in 2D real space
  (h_P : P = (-3, 5))  -- Coordinates of P
  (h_Q : Q.1 = 3 ∧ Q.2 = m - 2)  -- x-coordinate of Q is 3, y-coordinate is m-2
  (h_sym : P.1 = -Q.1 ∧ P.2 = -Q.2)  -- P and Q are symmetric about the origin
  : m = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_y_coordinate_l1518_151815


namespace NUMINAMATH_CALUDE_triangle_solution_l1518_151835

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem setup
def triangle_problem (t : Triangle) (median_CM : Line) (altitude_BH : Line) : Prop :=
  t.A = (5, 1) ∧
  median_CM = ⟨2, -1, -5⟩ ∧
  altitude_BH = ⟨1, -2, -5⟩

-- Theorem statement
theorem triangle_solution (t : Triangle) (median_CM : Line) (altitude_BH : Line) 
  (h : triangle_problem t median_CM altitude_BH) :
  (∃ (line_AC : Line), line_AC = ⟨2, 1, -11⟩) ∧ 
  t.B = (-1, -3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_solution_l1518_151835


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l1518_151893

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_tangent_to_parallel_lines (x y : ℚ) :
  -- The circle is tangent to these two lines
  (6 * x - 5 * y = 40 ∨ 6 * x - 5 * y = -20) →
  -- The center lies on this line
  (3 * x + 2 * y = 0) →
  -- The point (20/27, -10/9) is the center of the circle
  x = 20/27 ∧ y = -10/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l1518_151893


namespace NUMINAMATH_CALUDE_berts_spending_l1518_151857

/-- Bert's spending problem -/
theorem berts_spending (n : ℝ) : 
  (1/2) * ((2/3) * n - 7) = 10.5 → n = 42 := by
  sorry

end NUMINAMATH_CALUDE_berts_spending_l1518_151857


namespace NUMINAMATH_CALUDE_tenth_meeting_position_l1518_151897

/-- Represents a robot on a circular track -/
structure Robot where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the system -/
structure State where
  robotA : Robot
  robotB : Robot
  position : ℝ  -- Position on the track (0 ≤ position < 8)
  meetings : ℕ

/-- Updates the state after a meeting -/
def updateState (s : State) : State :=
  { s with
    robotB := { s.robotB with direction := !s.robotB.direction }
    meetings := s.meetings + 1
  }

/-- Simulates the movement of robots until they meet 10 times -/
def simulate (initialState : State) : ℝ :=
  sorry

theorem tenth_meeting_position (initialA initialB : Robot) :
  let initialState : State :=
    { robotA := initialA
      robotB := initialB
      position := 0
      meetings := 0
    }
  simulate initialState = 0 :=
sorry

end NUMINAMATH_CALUDE_tenth_meeting_position_l1518_151897


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1518_151865

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243 →
  a 9^2 / a 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1518_151865


namespace NUMINAMATH_CALUDE_more_b_shoes_than_a_l1518_151877

/-- Given the conditions about shoe boxes, prove that there are 640 more pairs of (B) shoes than (A) shoes. -/
theorem more_b_shoes_than_a : 
  ∀ (pairs_per_box : ℕ) (num_a_boxes : ℕ) (num_b_boxes : ℕ),
  pairs_per_box = 20 →
  num_a_boxes = 8 →
  num_b_boxes = 5 * num_a_boxes →
  num_b_boxes * pairs_per_box - num_a_boxes * pairs_per_box = 640 :=
by
  sorry

#check more_b_shoes_than_a

end NUMINAMATH_CALUDE_more_b_shoes_than_a_l1518_151877


namespace NUMINAMATH_CALUDE_exam_score_problem_l1518_151810

theorem exam_score_problem (correct_score : ℕ) (wrong_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  correct_score = 4 →
  wrong_score = 1 →
  total_score = 160 →
  correct_answers = 44 →
  ∃ (total_questions : ℕ),
    total_questions = correct_answers + (total_score - correct_score * correct_answers) / wrong_score ∧
    total_questions = 60 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1518_151810


namespace NUMINAMATH_CALUDE_team_a_faster_by_three_hours_l1518_151874

/-- Proves that Team A finishes 3 hours faster than Team W in a 300-mile race -/
theorem team_a_faster_by_three_hours 
  (course_length : ℝ) 
  (speed_w : ℝ) 
  (speed_difference : ℝ) : 
  course_length = 300 → 
  speed_w = 20 → 
  speed_difference = 5 → 
  (course_length / speed_w) - (course_length / (speed_w + speed_difference)) = 3 := by
  sorry

#check team_a_faster_by_three_hours

end NUMINAMATH_CALUDE_team_a_faster_by_three_hours_l1518_151874


namespace NUMINAMATH_CALUDE_relationship_abc_l1518_151869

theorem relationship_abc (a b c : ℝ) : 
  a = 2 → b = Real.log 9 → c = 2 * Real.sin (9 * π / 5) → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1518_151869


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1518_151889

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 48) 
  (h3 : b * c = 72) : 
  a * b * c = 168 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1518_151889


namespace NUMINAMATH_CALUDE_inverse_exists_mod_prime_wilsons_theorem_l1518_151896

-- Define primality
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Part 1: Inverse exists for non-zero elements modulo prime
theorem inverse_exists_mod_prime (p k : ℕ) (hp : isPrime p) (hk : ¬(p ∣ k)) :
  ∃ l : ℕ, k * l ≡ 1 [ZMOD p] :=
sorry

-- Part 2: Wilson's theorem
theorem wilsons_theorem (n : ℕ) :
  isPrime n ↔ (Nat.factorial (n - 1)) ≡ -1 [ZMOD n] :=
sorry

end NUMINAMATH_CALUDE_inverse_exists_mod_prime_wilsons_theorem_l1518_151896


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_is_negative_six_l1518_151878

theorem smallest_integer_y (y : ℤ) : (10 + 3 * y ≤ -8) ↔ (y ≤ -6) :=
  sorry

theorem smallest_integer_is_negative_six : ∃ (y : ℤ), (10 + 3 * y ≤ -8) ∧ (∀ (z : ℤ), (10 + 3 * z ≤ -8) → z ≥ y) ∧ y = -6 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_is_negative_six_l1518_151878


namespace NUMINAMATH_CALUDE_expand_product_l1518_151853

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1518_151853


namespace NUMINAMATH_CALUDE_infinitely_many_triplets_sum_of_squares_l1518_151832

theorem infinitely_many_triplets_sum_of_squares :
  ∃ f : ℕ → ℤ, ∀ k : ℕ,
    (∃ a b : ℤ, f k = a^2 + b^2) ∧
    (∃ c d : ℤ, f k + 1 = c^2 + d^2) ∧
    (∃ e g : ℤ, f k + 2 = e^2 + g^2) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_triplets_sum_of_squares_l1518_151832


namespace NUMINAMATH_CALUDE_power_division_rule_l1518_151827

theorem power_division_rule (a : ℝ) : a^7 / a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1518_151827


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l1518_151895

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Function to get the last two nonzero digits of a number -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the last two nonzero digits of 70! are 44 -/
theorem last_two_nonzero_digits_70_factorial :
  lastTwoNonzeroDigits (factorial 70) = 44 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l1518_151895


namespace NUMINAMATH_CALUDE_expression_evaluation_l1518_151883

theorem expression_evaluation : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1518_151883


namespace NUMINAMATH_CALUDE_min_value_implies_a_solution_set_inequality_l1518_151821

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f (2*x + a) + f (2*x - a) ≥ 4) ∧
  (∃ x, f (2*x + a) + f (2*x - a) = 4) →
  a = 2 ∨ a = -2 :=
sorry

-- Theorem for part 2
theorem solution_set_inequality :
  {x : ℝ | f x > 1 - (1/2)*x} = {x : ℝ | x > -2 ∨ x < -10} :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_solution_set_inequality_l1518_151821


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l1518_151870

theorem cos_sin_sum_equals_half : 
  Real.cos (263 * π / 180) * Real.cos (203 * π / 180) + 
  Real.sin (83 * π / 180) * Real.sin (23 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l1518_151870


namespace NUMINAMATH_CALUDE_complex_modulus_l1518_151886

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1518_151886


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1518_151880

/-- A quadratic function f(x) = ax^2 + bx + 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- Theorem: If f(1) = 3 and f(2) = 6, then f(3) = 10 -/
theorem quadratic_function_value (a b : ℝ) :
  f a b 1 = 3 → f a b 2 = 6 → f a b 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1518_151880


namespace NUMINAMATH_CALUDE_problem_solution_l1518_151850

theorem problem_solution : (2021^2 - 2021) / 2021 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1518_151850


namespace NUMINAMATH_CALUDE_task_completion_time_l1518_151814

/-- The time taken to complete a task when two people work together, with one person stopping early. -/
def completionTime (john_rate : ℚ) (jane_rate : ℚ) (early_stop : ℕ) : ℚ :=
  let combined_rate := john_rate + jane_rate
  let x := (1 - john_rate * early_stop) / combined_rate
  x + early_stop

theorem task_completion_time :
  let john_rate : ℚ := 1 / 20
  let jane_rate : ℚ := 1 / 12
  let early_stop : ℕ := 4
  completionTime john_rate jane_rate early_stop = 10 := by
  sorry

#eval completionTime (1 / 20 : ℚ) (1 / 12 : ℚ) 4

end NUMINAMATH_CALUDE_task_completion_time_l1518_151814


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l1518_151831

def a : Fin 3 → ℝ := ![3, 5, 4]
def b : Fin 3 → ℝ := ![5, 9, 7]
def c₁ : Fin 3 → ℝ := fun i => -2 * a i + b i
def c₂ : Fin 3 → ℝ := fun i => 3 * a i - 2 * b i

theorem vectors_not_collinear : ¬ ∃ (k : ℝ), c₁ = fun i => k * c₂ i := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l1518_151831


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1518_151851

theorem baker_remaining_cakes (total_cakes friend_bought : ℕ) 
  (h1 : total_cakes = 155)
  (h2 : friend_bought = 140) :
  total_cakes - friend_bought = 15 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1518_151851


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1518_151826

theorem bicycle_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 225) :
  ∃ (cost_price_A : ℝ), 
    cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) = price_C ∧ 
    cost_price_A = 150 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l1518_151826


namespace NUMINAMATH_CALUDE_complete_square_sum_l1518_151847

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1518_151847


namespace NUMINAMATH_CALUDE_linear_system_integer_solution_l1518_151859

theorem linear_system_integer_solution (a b : ℤ) :
  ∃ (x y z t : ℤ), x + y + 2*z + 2*t = a ∧ 2*x - 2*y + z - t = b := by
sorry

end NUMINAMATH_CALUDE_linear_system_integer_solution_l1518_151859


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l1518_151824

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 5| ≤ 22 → y ≥ x) ↔ x = -5 := by
sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l1518_151824


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l1518_151881

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum :
  (interior_sum 4 = 6) →
  (interior_sum 5 = 14) →
  (∀ k ≥ 3, interior_sum k = 2^(k-1) - 2) →
  interior_sum 9 = 254 :=
by sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l1518_151881


namespace NUMINAMATH_CALUDE_max_fleas_l1518_151852

/-- Represents a flea's direction of movement --/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a position on the board --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a flea on the board --/
structure Flea :=
  (position : Position)
  (direction : Direction)

/-- Represents the board state --/
def BoardState := List Flea

/-- The size of the board --/
def boardSize : Nat := 10

/-- The duration of observation in minutes --/
def observationTime : Nat := 60

/-- Function to update a flea's position based on its direction --/
def updateFleaPosition (f : Flea) : Flea :=
  sorry

/-- Function to check if two fleas occupy the same position --/
def fleaCollision (f1 f2 : Flea) : Bool :=
  sorry

/-- Function to simulate one minute of flea movement --/
def simulateMinute (state : BoardState) : BoardState :=
  sorry

/-- Function to simulate the entire observation period --/
def simulateObservation (initialState : BoardState) : Bool :=
  sorry

/-- Theorem stating the maximum number of fleas --/
theorem max_fleas : 
  ∀ (initialState : BoardState),
    simulateObservation initialState → List.length initialState ≤ 40 :=
  sorry

end NUMINAMATH_CALUDE_max_fleas_l1518_151852


namespace NUMINAMATH_CALUDE_ma_xiaohu_speed_ma_xiaohu_speed_proof_l1518_151862

/-- Proves that Ma Xiaohu's speed is 80 meters per minute given the problem conditions -/
theorem ma_xiaohu_speed : ℝ → Prop :=
  fun (x : ℝ) ↦
    let total_distance : ℝ := 1800
    let catch_up_distance : ℝ := 200
    let father_delay : ℝ := 10
    let father_speed : ℝ := 2 * x
    let ma_distance : ℝ := total_distance - catch_up_distance
    let ma_time : ℝ := ma_distance / x
    let father_time : ℝ := ma_distance / father_speed
    ma_time - father_time = father_delay → x = 80

/-- Proof of the theorem -/
theorem ma_xiaohu_speed_proof : ma_xiaohu_speed 80 := by
  sorry

end NUMINAMATH_CALUDE_ma_xiaohu_speed_ma_xiaohu_speed_proof_l1518_151862


namespace NUMINAMATH_CALUDE_triangle_side_length_l1518_151885

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  c = 4 * Real.sqrt 2 →
  B = π / 4 →
  S = 2 →
  S = (1 / 2) * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1518_151885


namespace NUMINAMATH_CALUDE_student_tickets_sold_l1518_151840

theorem student_tickets_sold (adult_price student_price : ℚ)
  (total_tickets : ℕ) (total_amount : ℚ)
  (h1 : adult_price = 4)
  (h2 : student_price = 5/2)
  (h3 : total_tickets = 59)
  (h4 : total_amount = 445/2) :
  ∃ (student_tickets : ℕ),
    student_tickets = 9 ∧
    student_tickets ≤ total_tickets ∧
    ∃ (adult_tickets : ℕ),
      adult_tickets + student_tickets = total_tickets ∧
      adult_price * adult_tickets + student_price * student_tickets = total_amount :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l1518_151840


namespace NUMINAMATH_CALUDE_johann_oranges_l1518_151823

theorem johann_oranges (initial : ℕ) (eaten : ℕ) (returned : ℕ) (final : ℕ) : 
  initial = 60 →
  returned = 5 →
  final = 30 →
  (initial - eaten) / 2 + returned = final →
  eaten = 10 := by
sorry

end NUMINAMATH_CALUDE_johann_oranges_l1518_151823


namespace NUMINAMATH_CALUDE_janelle_blue_marble_bags_l1518_151894

/-- Calculates the number of bags of blue marbles Janelle bought -/
def blue_marble_bags (initial_green : ℕ) (marbles_per_bag : ℕ) (green_given : ℕ) (blue_given : ℕ) (total_remaining : ℕ) : ℕ :=
  ((total_remaining + green_given + blue_given) - initial_green) / marbles_per_bag

/-- Proves that Janelle bought 6 bags of blue marbles -/
theorem janelle_blue_marble_bags :
  blue_marble_bags 26 10 6 8 72 = 6 := by
  sorry

#eval blue_marble_bags 26 10 6 8 72

end NUMINAMATH_CALUDE_janelle_blue_marble_bags_l1518_151894


namespace NUMINAMATH_CALUDE_sine_of_angle_between_vectors_l1518_151833

/-- Given vectors a and b with an angle θ between them, 
    if a = (2, 1) and 3b + a = (5, 4), then sin θ = √10/10 -/
theorem sine_of_angle_between_vectors (a b : ℝ × ℝ) (θ : ℝ) :
  a = (2, 1) →
  3 • b + a = (5, 4) →
  Real.sin θ = (Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_between_vectors_l1518_151833


namespace NUMINAMATH_CALUDE_roses_in_vase_l1518_151811

/-- The number of roses in a vase after adding new roses -/
def total_roses (initial_roses new_roses : ℕ) : ℕ :=
  initial_roses + new_roses

/-- Theorem: There are 23 roses in the vase after Jessica adds her newly cut roses -/
theorem roses_in_vase : total_roses 7 16 = 23 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1518_151811


namespace NUMINAMATH_CALUDE_problem_sequence_sum_largest_fib_is_196418_l1518_151802

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The sequence in the problem -/
def problem_sequence : List ℤ :=
  [2, -3, -5, 8, 13, -21, -34, 55, 89, -144, -233, 377, 46368, -75025, -121393, 196418]

/-- The sum of the problem sequence -/
def sequence_sum : ℤ := problem_sequence.sum

/-- Theorem stating that the sum of the problem sequence equals 196418 -/
theorem problem_sequence_sum : sequence_sum = 196418 := by
  sorry

/-- The largest Fibonacci number in the sequence -/
def largest_fib : ℕ := 196418

/-- Theorem stating that the largest Fibonacci number in the sequence is 196418 -/
theorem largest_fib_is_196418 : fib 27 = largest_fib := by
  sorry

end NUMINAMATH_CALUDE_problem_sequence_sum_largest_fib_is_196418_l1518_151802


namespace NUMINAMATH_CALUDE_exists_non_prime_product_l1518_151819

/-- The k-th prime number -/
def nthPrime (k : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers plus 1 -/
def primeProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * nthPrime (i + 1)) 1 + 1

/-- Theorem stating that there exists a number n such that primeProduct n is not prime -/
theorem exists_non_prime_product : ∃ n : ℕ, ¬ Nat.Prime (primeProduct n) := by sorry

end NUMINAMATH_CALUDE_exists_non_prime_product_l1518_151819


namespace NUMINAMATH_CALUDE_total_rainfall_is_correct_l1518_151838

-- Define conversion factors
def inch_to_cm : ℝ := 2.54
def mm_to_cm : ℝ := 0.1

-- Define daily rainfall measurements
def monday_rain : ℝ := 0.12962962962962962
def tuesday_rain : ℝ := 3.5185185185185186
def wednesday_rain : ℝ := 0.09259259259259259
def thursday_rain : ℝ := 0.10222222222222223
def friday_rain : ℝ := 12.222222222222221
def saturday_rain : ℝ := 0.2222222222222222
def sunday_rain : ℝ := 0.17444444444444446

-- Define the units for each day's measurement
inductive RainUnit
| Centimeter
| Millimeter
| Inch

def monday_unit : RainUnit := RainUnit.Centimeter
def tuesday_unit : RainUnit := RainUnit.Millimeter
def wednesday_unit : RainUnit := RainUnit.Centimeter
def thursday_unit : RainUnit := RainUnit.Inch
def friday_unit : RainUnit := RainUnit.Millimeter
def saturday_unit : RainUnit := RainUnit.Centimeter
def sunday_unit : RainUnit := RainUnit.Inch

-- Function to convert a measurement to centimeters based on its unit
def to_cm (measurement : ℝ) (unit : RainUnit) : ℝ :=
  match unit with
  | RainUnit.Centimeter => measurement
  | RainUnit.Millimeter => measurement * mm_to_cm
  | RainUnit.Inch => measurement * inch_to_cm

-- Theorem statement
theorem total_rainfall_is_correct : 
  to_cm monday_rain monday_unit +
  to_cm tuesday_rain tuesday_unit +
  to_cm wednesday_rain wednesday_unit +
  to_cm thursday_rain thursday_unit +
  to_cm friday_rain friday_unit +
  to_cm saturday_rain saturday_unit +
  to_cm sunday_rain sunday_unit = 2.721212629851652 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_is_correct_l1518_151838


namespace NUMINAMATH_CALUDE_hammond_statues_weight_l1518_151829

/-- The weight of Hammond's marble statues problem -/
theorem hammond_statues_weight (original_weight : ℕ) (first_statue : ℕ) (third_statue : ℕ) (fourth_statue : ℕ) (discarded : ℕ) :
  original_weight = 80 ∧ 
  first_statue = 10 ∧ 
  third_statue = 15 ∧ 
  fourth_statue = 15 ∧ 
  discarded = 22 →
  ∃ (second_statue : ℕ), 
    second_statue = 18 ∧ 
    original_weight = first_statue + second_statue + third_statue + fourth_statue + discarded :=
by sorry

end NUMINAMATH_CALUDE_hammond_statues_weight_l1518_151829


namespace NUMINAMATH_CALUDE_annies_class_size_l1518_151836

theorem annies_class_size :
  ∀ (total_spent : ℚ) (candy_cost : ℚ) (candies_per_classmate : ℕ) (candies_left : ℕ),
    total_spent = 8 →
    candy_cost = 1/10 →
    candies_per_classmate = 2 →
    candies_left = 12 →
    (total_spent / candy_cost - candies_left) / candies_per_classmate = 34 := by
  sorry

end NUMINAMATH_CALUDE_annies_class_size_l1518_151836


namespace NUMINAMATH_CALUDE_motorcycle_trip_distance_l1518_151808

/-- Given a motorcycle trip from A to B to C with the following conditions:
  - The average speed for the entire trip is 25 miles per hour
  - The time from A to B is 3 times the time from B to C
  - The distance from B to C is half the distance from A to B
Prove that the distance from A to B is 100/3 miles -/
theorem motorcycle_trip_distance (average_speed : ℝ) (time_ratio : ℝ) (distance_ratio : ℝ) :
  average_speed = 25 →
  time_ratio = 3 →
  distance_ratio = 1/2 →
  ∃ (distance_AB : ℝ), distance_AB = 100/3 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_trip_distance_l1518_151808
