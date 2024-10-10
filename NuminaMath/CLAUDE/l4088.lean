import Mathlib

namespace min_segments_polyline_l4088_408830

/-- Represents a square grid divided into n^2 smaller squares -/
structure SquareGrid (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents a polyline that passes through the centers of all smaller squares -/
structure Polyline (n : ℕ) where
  grid : SquareGrid n
  segments : ℕ
  passes_all_centers : segments ≥ 1

/-- Theorem stating the minimum number of segments in the polyline -/
theorem min_segments_polyline (n : ℕ) (h : n > 0) :
  ∃ (p : Polyline n), ∀ (q : Polyline n), p.segments ≤ q.segments ∧ p.segments = 2 * n - 2 :=
sorry

end min_segments_polyline_l4088_408830


namespace book_distribution_count_correct_l4088_408828

/-- The number of ways to distribute 5 distinct books among 3 people,
    where one person receives 1 book and two people receive 2 books each. -/
def book_distribution_count : ℕ := 90

/-- Theorem stating that the number of book distribution methods is correct. -/
theorem book_distribution_count_correct :
  let n_books : ℕ := 5
  let n_people : ℕ := 3
  let books_per_person : List ℕ := [2, 2, 1]
  true → book_distribution_count = 90 :=
by
  sorry

end book_distribution_count_correct_l4088_408828


namespace sqrt_sum_simplification_l4088_408897

theorem sqrt_sum_simplification : 
  Real.sqrt 75 - 9 * Real.sqrt (1/3) + Real.sqrt 48 = 6 * Real.sqrt 3 := by
  sorry

end sqrt_sum_simplification_l4088_408897


namespace quadratic_root_implies_k_l4088_408858

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 3 = 0) ∧ (1^2 + k*1 - 3 = 0) → k = 2 := by
  sorry

end quadratic_root_implies_k_l4088_408858


namespace number_pairs_theorem_l4088_408802

theorem number_pairs_theorem (a b : ℝ) :
  a^2 + b^2 = 15 * (a + b) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b)) →
  (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
by sorry

end number_pairs_theorem_l4088_408802


namespace marble_distribution_l4088_408804

theorem marble_distribution (total_marbles : ℕ) (ratio_a ratio_b : ℕ) (given_marbles : ℕ) : 
  total_marbles = 36 →
  ratio_a = 4 →
  ratio_b = 5 →
  given_marbles = 2 →
  (ratio_b * (total_marbles / (ratio_a + ratio_b))) - given_marbles = 18 :=
by sorry

end marble_distribution_l4088_408804


namespace unequal_grandchildren_probability_l4088_408884

theorem unequal_grandchildren_probability (n : ℕ) (p_male : ℝ) (p_female : ℝ) : 
  n = 12 →
  p_male = 0.6 →
  p_female = 0.4 →
  p_male + p_female = 1 →
  let p_equal := (n.choose (n / 2)) * (p_male ^ (n / 2)) * (p_female ^ (n / 2))
  1 - p_equal = 0.823 := by
sorry

end unequal_grandchildren_probability_l4088_408884


namespace modular_inverse_of_5_mod_31_l4088_408885

theorem modular_inverse_of_5_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end modular_inverse_of_5_mod_31_l4088_408885


namespace abc_sum_l4088_408860

theorem abc_sum (a b c : ℕ+) 
  (eq1 : a * b + c = 55)
  (eq2 : b * c + a = 55)
  (eq3 : a * c + b = 55) :
  a + b + c = 40 := by
  sorry

end abc_sum_l4088_408860


namespace profit_for_two_yuan_reduction_selling_price_for_770_profit_no_price_for_880_profit_l4088_408871

/-- Represents the supermarket beverage pricing and sales model -/
structure BeverageModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the profit for a given price reduction -/
def profit (model : BeverageModel) (price_reduction : ℝ) : ℝ :=
  let new_price := model.initial_price - price_reduction
  let new_sales := model.initial_sales + model.price_sensitivity * price_reduction
  (new_price - model.cost_price) * new_sales

/-- Theorem: The profit with a 2 yuan price reduction is 800 yuan -/
theorem profit_for_two_yuan_reduction (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  profit model 2 = 800 := by sorry

/-- Theorem: To achieve a profit of 770 yuan, the selling price should be 55 yuan -/
theorem selling_price_for_770_profit (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  ∃ (price_reduction : ℝ), profit model price_reduction = 770 ∧ 
  model.initial_price - price_reduction = 55 := by sorry

/-- Theorem: There is no selling price that can achieve a profit of 880 yuan -/
theorem no_price_for_880_profit (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  ¬∃ (price_reduction : ℝ), profit model price_reduction = 880 := by sorry

end profit_for_two_yuan_reduction_selling_price_for_770_profit_no_price_for_880_profit_l4088_408871


namespace geometric_sequence_a4_l4088_408824

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + (2/3) * a 2 = 3 →
  a 4^2 = (1/9) * a 3 * a 7 →
  a 4 = 27 := by
sorry

end geometric_sequence_a4_l4088_408824


namespace equidistant_complex_function_l4088_408829

theorem equidistant_complex_function (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ z : ℂ, Complex.abs ((a + Complex.I * b) * z - z) = Complex.abs ((a + Complex.I * b) * z)) →
  Complex.abs (a + Complex.I * b) = 8 →
  b^2 = 255/4 := by
sorry

end equidistant_complex_function_l4088_408829


namespace class_age_difference_l4088_408851

theorem class_age_difference (n : ℕ) (T : ℕ) : 
  T = n * 40 →
  (T + 408) / (n + 12) = 36 →
  40 - (T + 408) / (n + 12) = 4 :=
by sorry

end class_age_difference_l4088_408851


namespace incenter_orthocenter_collinearity_l4088_408818

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

-- Define the necessary concepts
def isIncenter (I : Point) (t : Triangle) : Prop := sorry
def isOrthocenter (H : Point) (t : Triangle) : Prop := sorry
def isMidpoint (M : Point) (A B : Point) : Prop := sorry
def liesOn (P : Point) (A B : Point) : Prop := sorry
def intersectsAt (A B C D : Point) (K : Point) : Prop := sorry
def isCircumcenter (O : Point) (t : Triangle) : Prop := sorry
def areCollinear (A B C : Point) : Prop := sorry
def areaTriangle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem incenter_orthocenter_collinearity 
  (t : Triangle) (I H B₁ C₁ B₂ C₂ K A₁ : Point) : 
  isIncenter I t → 
  isOrthocenter H t → 
  isMidpoint B₁ t.A t.C → 
  isMidpoint C₁ t.A t.B → 
  liesOn B₂ t.A t.B → 
  liesOn B₂ B₁ I → 
  B₂ ≠ t.B → 
  liesOn C₂ t.A C₁ → 
  liesOn C₂ C₁ I → 
  intersectsAt B₂ C₂ t.B t.C K → 
  isCircumcenter A₁ ⟨t.B, H, t.C⟩ → 
  (areCollinear t.A I A₁ ↔ areaTriangle t.B K B₂ = areaTriangle t.C K C₂) :=
sorry

end incenter_orthocenter_collinearity_l4088_408818


namespace theresa_kayla_ratio_l4088_408898

/-- The number of chocolate bars Theresa bought -/
def theresa_chocolate : ℕ := 12

/-- The number of soda cans Theresa bought -/
def theresa_soda : ℕ := 18

/-- The total number of items Kayla bought -/
def kayla_total : ℕ := 15

/-- The ratio of items Theresa bought to items Kayla bought -/
def item_ratio : ℚ := (theresa_chocolate + theresa_soda : ℚ) / kayla_total

theorem theresa_kayla_ratio : item_ratio = 2 := by sorry

end theresa_kayla_ratio_l4088_408898


namespace converse_negation_equivalence_triangle_angles_arithmetic_sequence_inequality_system_not_equivalent_squared_inequality_implication_l4088_408801

-- 1. Converse and negation of a proposition
theorem converse_negation_equivalence (P Q : Prop) : 
  (P → Q) ↔ ¬Q → ¬P := by sorry

-- 2. Triangle angles forming arithmetic sequence
theorem triangle_angles_arithmetic_sequence (A B C : ℝ) :
  (A + B + C = 180) → (B = 60 ↔ 2 * B = A + C) := by sorry

-- 3. Inequality system counterexample
theorem inequality_system_not_equivalent :
  ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ ¬(x > 1 ∧ y > 2) := by sorry

-- 4. Squared inequality implication
theorem squared_inequality_implication (a b : ℝ) :
  (∀ m : ℝ, a * m^2 < b * m^2 → a < b) ∧
  ¬(∀ a b : ℝ, a < b → ∀ m : ℝ, a * m^2 < b * m^2) := by sorry

end converse_negation_equivalence_triangle_angles_arithmetic_sequence_inequality_system_not_equivalent_squared_inequality_implication_l4088_408801


namespace min_value_geometric_sequence_l4088_408822

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem min_value_geometric_sequence (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 2) :
  ∃ (min_value : ℝ), 
    (∀ a₁ a₃, a 1 = a₁ ∧ a 3 = a₃ → a₁ + 2 * a₃ ≥ min_value) ∧
    (∃ a₁ a₃, a 1 = a₁ ∧ a 3 = a₃ ∧ a₁ + 2 * a₃ = min_value) ∧
    min_value = 4 * Real.sqrt 2 :=
sorry

end min_value_geometric_sequence_l4088_408822


namespace cube_edge_length_l4088_408888

/-- Given the cost of paint, coverage per quart, and total cost to paint a cube,
    prove that the edge length of the cube is 10 feet. -/
theorem cube_edge_length
  (paint_cost_per_quart : ℝ)
  (coverage_per_quart : ℝ)
  (total_cost : ℝ)
  (h1 : paint_cost_per_quart = 3.2)
  (h2 : coverage_per_quart = 60)
  (h3 : total_cost = 32)
  : ∃ (edge_length : ℝ), edge_length = 10 ∧ 6 * edge_length^2 = total_cost / paint_cost_per_quart * coverage_per_quart :=
by sorry

end cube_edge_length_l4088_408888


namespace four_intersections_implies_a_range_l4088_408811

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs x + a - 1

-- State the theorem
theorem four_intersections_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) →
  1 < a ∧ a < 5/4 :=
by sorry

end four_intersections_implies_a_range_l4088_408811


namespace earnings_difference_l4088_408847

/-- Mateo's hourly rate in dollars -/
def mateo_hourly_rate : ℕ := 20

/-- Sydney's daily rate in dollars -/
def sydney_daily_rate : ℕ := 400

/-- Number of hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Mateo's total earnings for one week in dollars -/
def mateo_earnings : ℕ := mateo_hourly_rate * hours_per_week

/-- Sydney's total earnings for one week in dollars -/
def sydney_earnings : ℕ := sydney_daily_rate * days_per_week

theorem earnings_difference : mateo_earnings - sydney_earnings = 560 := by
  sorry

end earnings_difference_l4088_408847


namespace expression_evaluation_l4088_408889

theorem expression_evaluation : 
  let a : ℝ := 2 * Real.sin (π / 4) + (1 / 2)⁻¹
  ((a^2 - 4) / a) / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end expression_evaluation_l4088_408889


namespace weighted_cauchy_schwarz_l4088_408813

theorem weighted_cauchy_schwarz (p q x y : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q < 1) : 
  (p * x + q * y)^2 ≤ p * x^2 + q * y^2 := by
  sorry

end weighted_cauchy_schwarz_l4088_408813


namespace range_of_m_for_line_intersecting_semicircle_l4088_408877

/-- A line intersecting a semicircle at exactly two points -/
structure LineIntersectingSemicircle where
  m : ℝ
  intersects_twice : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ + y₁ = m ∧ y₁ = Real.sqrt (9 - x₁^2) ∧ y₁ ≥ 0 ∧
    x₂ + y₂ = m ∧ y₂ = Real.sqrt (9 - x₂^2) ∧ y₂ ≥ 0 ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- The range of m values for lines intersecting the semicircle at exactly two points -/
theorem range_of_m_for_line_intersecting_semicircle (l : LineIntersectingSemicircle) :
  l.m ≥ 3 ∧ l.m < 3 * Real.sqrt 2 :=
sorry

end range_of_m_for_line_intersecting_semicircle_l4088_408877


namespace rectangle_to_square_l4088_408899

/-- Given a rectangle with perimeter 50 cm, prove that decreasing its length by 4 cm
    and increasing its width by 3 cm results in a square with side 12 cm and equal area. -/
theorem rectangle_to_square (L W : ℝ) : 
  L > 0 ∧ W > 0 ∧                    -- Length and width are positive
  2 * L + 2 * W = 50 ∧               -- Perimeter of original rectangle is 50 cm
  L * W = (L - 4) * (W + 3) →        -- Area remains constant after transformation
  L = 16 ∧ W = 9 ∧                   -- Original rectangle dimensions
  L - 4 = 12 ∧ W + 3 = 12            -- New shape is a square with side 12 cm
  := by sorry

end rectangle_to_square_l4088_408899


namespace subset_proof_l4088_408833

def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem subset_proof : N ⊆ M := by
  sorry

end subset_proof_l4088_408833


namespace smallest_inverse_domain_l4088_408859

def g (x : ℝ) : ℝ := (x - 3)^2 - 1

theorem smallest_inverse_domain (d : ℝ) :
  (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end smallest_inverse_domain_l4088_408859


namespace cos_theta_value_l4088_408825

theorem cos_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi) : 
  Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end cos_theta_value_l4088_408825


namespace path_length_calculation_l4088_408890

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 500

/-- Represents the length of the path on the map in inches -/
def path_length_on_map : ℝ := 3.5

/-- Calculates the actual length of the path in feet -/
def actual_path_length : ℝ := map_scale * path_length_on_map

theorem path_length_calculation :
  actual_path_length = 1750 := by sorry

end path_length_calculation_l4088_408890


namespace fraction_zero_implies_x_equals_five_l4088_408850

theorem fraction_zero_implies_x_equals_five (x : ℝ) : 
  (x^2 - 25) / (x + 5) = 0 ∧ x + 5 ≠ 0 → x = 5 := by
  sorry

end fraction_zero_implies_x_equals_five_l4088_408850


namespace quadratic_polynomial_value_l4088_408866

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_polynomial_value (a b c : ℚ) :
  let p : ℚ → ℚ := QuadraticPolynomial a b c
  (∀ x : ℚ, (x - 1) * (x + 1) * (x - 8) ∣ p x ^ 3 - x) →
  p 13 = -3 := by
  sorry

end quadratic_polynomial_value_l4088_408866


namespace infinite_sum_not_diff_powers_l4088_408882

theorem infinite_sum_not_diff_powers (n : ℕ) (hn : n > 1) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧
    (∀ k ∈ S, ∃ a b : ℕ, k = a^n + b^n) ∧
    (∀ k ∈ S, ∀ c d : ℕ, k ≠ c^n - d^n) :=
sorry

end infinite_sum_not_diff_powers_l4088_408882


namespace seven_by_seven_checkerboard_shading_l4088_408841

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a checkerboard shading pattern on a grid -/
def checkerboard_shading (g : Grid) : ℕ :=
  (g.size * g.size) / 2

/-- Calculates the percentage of shaded squares in a grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (checkerboard_shading g : ℚ) / (g.size * g.size : ℚ) * 100

/-- Theorem: The percentage of shaded squares in a 7x7 checkerboard is 2400/49 -/
theorem seven_by_seven_checkerboard_shading :
  shaded_percentage { size := 7 } = 2400 / 49 := by
  sorry

end seven_by_seven_checkerboard_shading_l4088_408841


namespace strengthened_erdos_mordell_inequality_l4088_408894

theorem strengthened_erdos_mordell_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * area + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry

end strengthened_erdos_mordell_inequality_l4088_408894


namespace action_figure_shelves_l4088_408864

/-- Given a room with action figures and shelves, calculate the number of shelves. -/
theorem action_figure_shelves 
  (total_figures : ℕ) 
  (figures_per_shelf : ℕ) 
  (h1 : total_figures = 120) 
  (h2 : figures_per_shelf = 15) 
  (h3 : figures_per_shelf > 0) : 
  total_figures / figures_per_shelf = 8 := by
  sorry

end action_figure_shelves_l4088_408864


namespace lower_limit_of_a_l4088_408807

theorem lower_limit_of_a (a b : ℤ) (h1 : a < 15) (h2 : b > 6) (h3 : b < 21)
  (h4 : (a : ℝ) / 7 - (a : ℝ) / 20 = 1.55) : a ≥ 17 := by
  sorry

end lower_limit_of_a_l4088_408807


namespace quadratic_root_theorem_l4088_408872

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℝ) : Prop :=
  ∃ k : ℝ, b = k * a ∧ c = k * b

/-- Theorem: The root of a specific quadratic equation -/
theorem quadratic_root_theorem (p q r : ℝ) (h1 : isGeometricSequence p q r)
    (h2 : p ≤ q ∧ q ≤ r ∧ r ≤ 0) (h3 : ∃! x : ℝ, p * x^2 + q * x + r = 0) :
    ∃ x : ℝ, p * x^2 + q * x + r = 0 ∧ x = -1 := by
  sorry

end quadratic_root_theorem_l4088_408872


namespace find_M_l4088_408878

theorem find_M : ∃ M : ℕ+, (36 ^ 2 : ℕ) * (75 ^ 2) = (30 ^ 2) * (M.val ^ 2) ∧ M.val = 90 := by
  sorry

end find_M_l4088_408878


namespace store_purchase_price_l4088_408857

theorem store_purchase_price (tax_rate : Real) (discount : Real) (cody_payment : Real) : 
  tax_rate = 0.05 → discount = 8 → cody_payment = 17 → 
  ∃ (original_price : Real), 
    (original_price * (1 + tax_rate) - discount) / 2 = cody_payment ∧ 
    original_price = 40 := by
  sorry

end store_purchase_price_l4088_408857


namespace fraction_simplification_l4088_408895

theorem fraction_simplification :
  (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 - Real.sqrt 3)) = -(1 / 2) := by
  sorry

end fraction_simplification_l4088_408895


namespace olivias_paper_pieces_l4088_408876

/-- The number of paper pieces Olivia used -/
def pieces_used : ℕ := 56

/-- The number of paper pieces Olivia has left -/
def pieces_left : ℕ := 25

/-- The initial number of paper pieces Olivia had -/
def initial_pieces : ℕ := pieces_used + pieces_left

theorem olivias_paper_pieces : initial_pieces = 81 := by
  sorry

end olivias_paper_pieces_l4088_408876


namespace interval_and_sum_l4088_408892

theorem interval_and_sum : 
  ∃ (m M : ℝ), 
    (∀ x : ℝ, x > 0 ∧ 2 * |x^2 - 9| ≤ 9 * |x| ↔ m ≤ x ∧ x ≤ M) ∧
    m = 3/2 ∧ 
    M = 6 ∧
    10 * m + M = 21 := by
  sorry

end interval_and_sum_l4088_408892


namespace workshop_average_salary_l4088_408809

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (other_salary : ℕ)
  (h1 : total_workers = 14)
  (h2 : technicians = 7)
  (h3 : technician_salary = 12000)
  (h4 : other_salary = 6000) :
  (technicians * technician_salary + (total_workers - technicians) * other_salary) / total_workers = 9000 :=
by sorry

end workshop_average_salary_l4088_408809


namespace mckenna_work_hours_l4088_408826

-- Define the start and end times of Mckenna's work day
def start_time : ℕ := 8
def office_end_time : ℕ := 11
def conference_end_time : ℕ := 13
def work_end_time : ℕ := conference_end_time + 2

-- Define the duration of each part of Mckenna's work day
def office_duration : ℕ := office_end_time - start_time
def conference_duration : ℕ := conference_end_time - office_end_time
def after_conference_duration : ℕ := 2

-- Theorem to prove
theorem mckenna_work_hours :
  office_duration + conference_duration + after_conference_duration = 7 := by
  sorry


end mckenna_work_hours_l4088_408826


namespace cupcakes_remaining_l4088_408823

theorem cupcakes_remaining (total : ℕ) (given_away_fraction : ℚ) (eaten : ℕ) : 
  total = 60 → given_away_fraction = 4/5 → eaten = 3 →
  total * (1 - given_away_fraction) - eaten = 9 := by
sorry

end cupcakes_remaining_l4088_408823


namespace pentagon_area_l4088_408821

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a pentagon -/
structure Pentagon :=
  (F G H I J : Point)

/-- Calculates the area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Theorem: The area of the specific pentagon FGHIJ is 71√3/4 -/
theorem pentagon_area (p : Pentagon) :
  angle p.F p.G p.H = 120 * π / 180 →
  angle p.J p.F p.G = 120 * π / 180 →
  distance p.J p.F = 3 →
  distance p.F p.G = 3 →
  distance p.G p.H = 3 →
  distance p.H p.I = 5 →
  distance p.I p.J = 5 →
  area p = 71 * Real.sqrt 3 / 4 := by sorry

end pentagon_area_l4088_408821


namespace toy_factory_production_l4088_408867

/-- Represents the production constraints and goal for a toy factory --/
theorem toy_factory_production :
  ∃ (x y : ℕ),
    15 * x + 10 * y ≤ 450 ∧  -- Labor constraint
    20 * x + 5 * y ≤ 400 ∧   -- Raw material constraint
    80 * x + 45 * y = 2200   -- Total selling price
    := by sorry

end toy_factory_production_l4088_408867


namespace h_derivative_l4088_408815

/-- Given f = 5, g = 4g', and h(x) = (f + 2) / x, prove that h'(x) = 5/16 -/
theorem h_derivative (f g g' : ℝ) (h : ℝ → ℝ) :
  f = 5 →
  g = 4 * g' →
  (∀ x, h x = (f + 2) / x) →
  ∀ x, deriv h x = 5 / 16 :=
by
  sorry

end h_derivative_l4088_408815


namespace complex_cube_real_l4088_408835

theorem complex_cube_real (a b : ℝ) (hb : b ≠ 0) 
  (h : ∃ (r : ℝ), (Complex.mk a b)^3 = r) : b^2 = 3 * a^2 := by
  sorry

end complex_cube_real_l4088_408835


namespace valentines_given_away_l4088_408805

/-- Given Mrs. Franklin's initial and remaining Valentines, calculate how many she gave away. -/
theorem valentines_given_away
  (initial : ℝ)
  (remaining : ℝ)
  (h_initial : initial = 58.5)
  (h_remaining : remaining = 16.25) :
  initial - remaining = 42.25 := by
  sorry

end valentines_given_away_l4088_408805


namespace mean_median_difference_l4088_408880

/-- Represents the score distribution in the math competition -/
structure ScoreDistribution where
  score72 : Float
  score84 : Float
  score86 : Float
  score92 : Float
  score98 : Float
  sum_to_one : score72 + score84 + score86 + score92 + score98 = 1

/-- Calculates the median score given the score distribution -/
def median (d : ScoreDistribution) : Float :=
  86

/-- Calculates the mean score given the score distribution -/
def mean (d : ScoreDistribution) : Float :=
  72 * d.score72 + 84 * d.score84 + 86 * d.score86 + 92 * d.score92 + 98 * d.score98

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score72 = 0.15)
  (h2 : d.score84 = 0.30)
  (h3 : d.score86 = 0.25)
  (h4 : d.score92 = 0.10) :
  mean d - median d = 0.3 := by
  sorry

#check mean_median_difference

end mean_median_difference_l4088_408880


namespace combined_mean_l4088_408812

theorem combined_mean (set1_count : Nat) (set1_mean : ℝ) (set2_count : Nat) (set2_mean : ℝ) 
  (h1 : set1_count = 5)
  (h2 : set1_mean = 13)
  (h3 : set2_count = 6)
  (h4 : set2_mean = 24) :
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 19 :=
by
  sorry

end combined_mean_l4088_408812


namespace inverse_proportion_problem_inverse_proportion_comparison_l4088_408856

theorem inverse_proportion_problem (k : ℝ) (h_k : k ≠ 0) :
  (∀ x : ℝ, x > 0 → x ≤ 1 → k / x > 3 * x) ↔ k = 3 :=
by sorry

theorem inverse_proportion_comparison (m : ℝ) (h_m : m ≠ 0) :
  (∀ x : ℝ, x > 0 → x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) :=
by sorry

end inverse_proportion_problem_inverse_proportion_comparison_l4088_408856


namespace no_solution_iff_a_geq_five_l4088_408869

theorem no_solution_iff_a_geq_five (a : ℝ) :
  (∀ x : ℝ, ¬(x ≤ 5 ∧ x > a)) ↔ a ≥ 5 := by
  sorry

end no_solution_iff_a_geq_five_l4088_408869


namespace smallest_unbounded_population_l4088_408803

theorem smallest_unbounded_population : ∃ N : ℕ, N = 61 ∧ 
  (∀ m : ℕ, m < N → 2 * (m - 30) ≤ m) ∧ 
  (2 * (N - 30) > N) := by
  sorry

end smallest_unbounded_population_l4088_408803


namespace final_cost_is_33_08_l4088_408861

/-- The cost of a single deck in dollars -/
def deck_cost : ℚ := 7

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 0.1

/-- The sales tax rate as a decimal -/
def sales_tax_rate : ℚ := 0.05

/-- The number of decks Frank bought -/
def frank_decks : ℕ := 3

/-- The number of decks Frank's friend bought -/
def friend_decks : ℕ := 2

/-- The total cost before discount and tax -/
def total_cost : ℚ := deck_cost * (frank_decks + friend_decks)

/-- The discounted cost -/
def discounted_cost : ℚ := total_cost * (1 - discount_rate)

/-- The final cost including tax -/
def final_cost : ℚ := discounted_cost * (1 + sales_tax_rate)

/-- Theorem stating the final cost is $33.08 -/
theorem final_cost_is_33_08 : 
  ∃ (ε : ℚ), abs (final_cost - 33.08) < ε ∧ ε = 0.005 := by
  sorry

end final_cost_is_33_08_l4088_408861


namespace largest_prime_factor_l4088_408836

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^4 + 2 * 17^2 + 1 - 16^4) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (17^4 + 2 * 17^2 + 1 - 16^4) → q ≤ p) →
  (∃ p : ℕ, p = 17 ∧ Nat.Prime p ∧ p ∣ (17^4 + 2 * 17^2 + 1 - 16^4) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (17^4 + 2 * 17^2 + 1 - 16^4) → q ≤ p) :=
by sorry

end largest_prime_factor_l4088_408836


namespace original_ball_count_original_ball_count_is_960_l4088_408852

-- Define the initial ratio of red to white balls
def initial_ratio : Rat := 19 / 13

-- Define the ratio after adding red balls
def ratio_after_red : Rat := 5 / 3

-- Define the ratio after adding white balls
def ratio_after_white : Rat := 13 / 11

-- Define the difference in added balls
def added_difference : ℕ := 80

-- Theorem statement
theorem original_ball_count : ℕ :=
  let initial_red : ℕ := 57
  let initial_white : ℕ := 39
  let final_red : ℕ := 65
  let final_white : ℕ := 55
  let portion_size : ℕ := added_difference / (final_white - initial_white - (final_red - initial_red))
  (initial_red + initial_white) * portion_size

-- Proof
theorem original_ball_count_is_960 : original_ball_count = 960 := by
  sorry

end original_ball_count_original_ball_count_is_960_l4088_408852


namespace ellipse_properties_and_max_area_l4088_408893

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem ellipse_properties_and_max_area 
  (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h4 : eccentricity c a = Real.sqrt 3 / 2)
  (h5 : ellipse_equation a b c (b^2/a))
  (h6 : distance c (b^2/a) = Real.sqrt 13 / 2) :
  (∃ (x y : ℝ), ellipse_equation 2 1 x y) ∧
  (∃ (S : ℝ), S = 4 ∧ 
    ∀ (m : ℝ), abs m < Real.sqrt 2 → 
      2 * Real.sqrt (m^2 * (8 - 4 * m^2)) ≤ S) := by
sorry

end ellipse_properties_and_max_area_l4088_408893


namespace polygon_coloring_l4088_408848

/-- Given a regular 103-sided polygon with 79 red vertices and 24 blue vertices,
    A is the number of pairs of adjacent red vertices and
    B is the number of pairs of adjacent blue vertices. -/
theorem polygon_coloring (A B : ℕ) :
  (∀ i : ℕ, 0 ≤ i ∧ i ≤ 23 → (A = 55 + i ∧ B = i)) ∧
  (B = 14 →
    (Nat.choose 23 10 * Nat.choose 78 9) / 14 =
      (Nat.choose 23 9 * Nat.choose 78 9) / 10) :=
by sorry

end polygon_coloring_l4088_408848


namespace count_divisible_by_11_between_100_and_500_l4088_408840

def count_divisible (lower upper divisor : ℕ) : ℕ :=
  (upper / divisor - (lower - 1) / divisor)

theorem count_divisible_by_11_between_100_and_500 :
  count_divisible 100 500 11 = 36 := by
  sorry

end count_divisible_by_11_between_100_and_500_l4088_408840


namespace coefficient_x5_in_expansion_l4088_408832

theorem coefficient_x5_in_expansion : 
  let n : ℕ := 9
  let k : ℕ := 4
  let a : ℝ := 3 * Real.sqrt 2
  (Nat.choose n k) * a^k = 40824 := by sorry

end coefficient_x5_in_expansion_l4088_408832


namespace natalia_clip_sales_l4088_408819

/-- Natalia's clip sales problem -/
theorem natalia_clip_sales 
  (x : ℝ) -- number of clips sold to each friend in April
  (y : ℝ) -- number of clips sold in May
  (z : ℝ) -- total earnings in dollars
  (h1 : y = x / 2) -- y is half of x
  : (48 * x + y = 97 * x / 2) ∧ (z / (48 * x + y) = 2 * z / (97 * x)) := by
  sorry

end natalia_clip_sales_l4088_408819


namespace gcd_operation_result_l4088_408843

theorem gcd_operation_result : (Nat.gcd 7350 165 - 15) * 3 = 0 := by sorry

end gcd_operation_result_l4088_408843


namespace min_value_theorem_l4088_408883

theorem min_value_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 7) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2 ≥ 36 := by
  sorry

end min_value_theorem_l4088_408883


namespace ancient_chinese_gold_tax_l4088_408875

theorem ancient_chinese_gold_tax (x : ℚ) : 
  x > 0 ∧ 
  x/2 + x/2 * 1/3 + x/3 * 1/4 + x/4 * 1/5 + x/5 * 1/6 = 1 → 
  x/5 * 1/6 = 1/25 := by
  sorry

end ancient_chinese_gold_tax_l4088_408875


namespace smallest_positive_period_of_f_l4088_408808

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem smallest_positive_period_of_f :
  is_smallest_positive_period f Real.pi := by sorry

end smallest_positive_period_of_f_l4088_408808


namespace mediant_inequality_l4088_408868

theorem mediant_inequality (a b p q r s : ℕ) 
  (h1 : q * r - p * s = 1) 
  (h2 : (p : ℚ) / q < (a : ℚ) / b) 
  (h3 : (a : ℚ) / b < (r : ℚ) / s) : 
  b ≥ q + s := by
  sorry

end mediant_inequality_l4088_408868


namespace x_squared_minus_y_squared_l4088_408853

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
sorry

end x_squared_minus_y_squared_l4088_408853


namespace mapping_A_to_B_l4088_408891

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {0, 3, 8, 15, 24}

def f (x : ℕ) : ℕ := x^2 - 1

theorem mapping_A_to_B :
  ∀ x ∈ A, f x ∈ B :=
by sorry

end mapping_A_to_B_l4088_408891


namespace current_speed_l4088_408834

/-- Given a boat's upstream and downstream speeds, calculate the speed of the current. -/
theorem current_speed (upstream_time : ℝ) (downstream_time : ℝ) :
  upstream_time = 20 →
  downstream_time = 9 →
  let upstream_speed := 60 / upstream_time
  let downstream_speed := 60 / downstream_time
  abs ((downstream_speed - upstream_speed) / 2 - 1.835) < 0.001 := by
  sorry

end current_speed_l4088_408834


namespace ones_digit_of_14_power_power_of_4_cycle_exponent_even_ones_digit_14_power_14_7_power_7_l4088_408849

theorem ones_digit_of_14_power (n : ℕ) : (14^n) % 10 = (4^n) % 10 := by sorry

theorem power_of_4_cycle : ∀ n : ℕ, (4^n) % 10 = (4^(n % 2 + 1)) % 10 := by sorry

theorem exponent_even : (14 * (7^7)) % 2 = 0 := by sorry

theorem ones_digit_14_power_14_7_power_7 : (14^(14 * (7^7))) % 10 = 4 := by sorry

end ones_digit_of_14_power_power_of_4_cycle_exponent_even_ones_digit_14_power_14_7_power_7_l4088_408849


namespace rohan_salary_rohan_salary_proof_l4088_408874

/-- Rohan's monthly salary calculation --/
theorem rohan_salary (food_percent : ℝ) (rent_percent : ℝ) (entertainment_percent : ℝ) 
  (conveyance_percent : ℝ) (savings : ℝ) : ℝ :=
  let total_expenses_percent : ℝ := food_percent + rent_percent + entertainment_percent + conveyance_percent
  let savings_percent : ℝ := 1 - total_expenses_percent
  savings / savings_percent

/-- Proof of Rohan's monthly salary --/
theorem rohan_salary_proof :
  rohan_salary 0.4 0.2 0.1 0.1 2500 = 12500 := by
  sorry

end rohan_salary_rohan_salary_proof_l4088_408874


namespace opposite_of_negative_fraction_l4088_408844

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  -(-(1 : ℚ) / n) = 1 / n :=
by sorry

end opposite_of_negative_fraction_l4088_408844


namespace complex_subtraction_l4088_408810

/-- Given complex numbers c and d, prove that c - 3d = 2 + 6i -/
theorem complex_subtraction (c d : ℂ) (hc : c = 5 + 3*I) (hd : d = 1 - I) :
  c - 3*d = 2 + 6*I := by
  sorry

end complex_subtraction_l4088_408810


namespace solve_quadratic_equation_1_solve_quadratic_equation_2_l4088_408816

-- Problem 1
theorem solve_quadratic_equation_1 :
  ∀ x : ℝ, x^2 - 4*x = 5 ↔ x = 5 ∨ x = -1 :=
sorry

-- Problem 2
theorem solve_quadratic_equation_2 :
  ∀ x : ℝ, 2*x^2 - 3*x + 1 = 0 ↔ x = 1 ∨ x = 1/2 :=
sorry

end solve_quadratic_equation_1_solve_quadratic_equation_2_l4088_408816


namespace set_disjoint_iff_m_range_l4088_408839

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m < x ∧ x < m+1}

theorem set_disjoint_iff_m_range (m : ℝ) : 
  (∀ x ∈ A, x ∉ B m) ↔ m ∈ Set.Iic (-2) ∪ Set.Ici 1 :=
sorry

end set_disjoint_iff_m_range_l4088_408839


namespace max_value_inequality_l4088_408870

theorem max_value_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 := by
  sorry

end max_value_inequality_l4088_408870


namespace no_solutions_in_interval_l4088_408862

theorem no_solutions_in_interval (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (1 / Real.sin x + 1 / Real.cos x ≠ 4) :=
by sorry

end no_solutions_in_interval_l4088_408862


namespace magnitude_of_z_l4088_408814

theorem magnitude_of_z (z : ℂ) (h : z^2 = 24 - 32*I) : Complex.abs z = 2 * Real.sqrt 10 := by
  sorry

end magnitude_of_z_l4088_408814


namespace expression_evaluation_l4088_408855

theorem expression_evaluation :
  ∀ x y : ℝ,
  (abs x = 2) →
  (y = 1) →
  (x * y < 0) →
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end expression_evaluation_l4088_408855


namespace amber_work_hours_l4088_408820

theorem amber_work_hours :
  ∀ (amber armand ella : ℝ),
  armand = amber / 3 →
  ella = 2 * amber →
  amber + armand + ella = 40 →
  amber = 12 := by
sorry

end amber_work_hours_l4088_408820


namespace circle_equation_l4088_408817

theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-2, 3)
  let is_tangent_to_x_axis : Prop := ∃ (x_0 : ℝ), (x_0 + 2)^2 + 3^2 = (x + 2)^2 + (y - 3)^2
  is_tangent_to_x_axis →
  (x + 2)^2 + (y - 3)^2 = 9 :=
by
  sorry

end circle_equation_l4088_408817


namespace locus_of_equilateral_triangle_vertex_l4088_408863

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a rotation function
def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  sorry

-- Define a function to check if a point is on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

-- Define a function to check if a triangle is equilateral
def isEquilateral (a b c : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem locus_of_equilateral_triangle_vertex (C : Circle) (P : ℝ × ℝ) :
  let locusM := {M : ℝ × ℝ | ∃ K, onCircle K C ∧ isEquilateral P K M}
  let rotated_circle_1 := {p : ℝ × ℝ | ∃ q, onCircle q C ∧ p = rotate q P (π/3)}
  let rotated_circle_2 := {p : ℝ × ℝ | ∃ q, onCircle q C ∧ p = rotate q P (-π/3)}
  if P = C.center then
    locusM = {p : ℝ × ℝ | onCircle p C}
  else
    locusM = rotated_circle_1 ∪ rotated_circle_2 :=
by sorry


end locus_of_equilateral_triangle_vertex_l4088_408863


namespace super_ball_distance_l4088_408827

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let descentDistances := List.range (numBounces + 1) |>.map (fun i => initialHeight * bounceRatio^i)
  let ascentDistances := List.range numBounces |>.map (fun i => initialHeight * bounceRatio^(i + 1))
  (descentDistances.sum + ascentDistances.sum)

/-- Theorem: The total distance traveled by a ball dropped from 20 meters, 
    bouncing 5/8 of its previous height each time, and hitting the ground 4 times, 
    is 73.442078125 meters. -/
theorem super_ball_distance :
  totalDistance 20 (5/8) 4 = 73.442078125 := by
  sorry


end super_ball_distance_l4088_408827


namespace face_masks_per_box_l4088_408842

/-- Proves the number of face masks in each box given the problem conditions --/
theorem face_masks_per_box :
  ∀ (num_boxes : ℕ) (sell_price : ℚ) (total_cost : ℚ) (total_profit : ℚ),
    num_boxes = 3 →
    sell_price = 1/2 →
    total_cost = 15 →
    total_profit = 15 →
    ∃ (masks_per_box : ℕ),
      masks_per_box = 20 ∧
      (num_boxes * masks_per_box : ℚ) * sell_price - total_cost = total_profit :=
by
  sorry


end face_masks_per_box_l4088_408842


namespace total_earnings_is_18_56_l4088_408838

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 150

/-- Represents the percentage of white marbles -/
def white_percent : ℚ := 20 / 100

/-- Represents the percentage of black marbles -/
def black_percent : ℚ := 25 / 100

/-- Represents the percentage of blue marbles -/
def blue_percent : ℚ := 30 / 100

/-- Represents the percentage of green marbles -/
def green_percent : ℚ := 15 / 100

/-- Represents the percentage of red marbles -/
def red_percent : ℚ := 10 / 100

/-- Represents the price of a white marble in dollars -/
def white_price : ℚ := 5 / 100

/-- Represents the price of a black marble in dollars -/
def black_price : ℚ := 10 / 100

/-- Represents the price of a blue marble in dollars -/
def blue_price : ℚ := 15 / 100

/-- Represents the price of a green marble in dollars -/
def green_price : ℚ := 12 / 100

/-- Represents the price of a red marble in dollars -/
def red_price : ℚ := 25 / 100

/-- Theorem stating that the total earnings from selling all marbles is $18.56 -/
theorem total_earnings_is_18_56 : 
  (↑total_marbles * white_percent * white_price) +
  (↑total_marbles * black_percent * black_price) +
  (↑total_marbles * blue_percent * blue_price) +
  (↑total_marbles * green_percent * green_price) +
  (↑total_marbles * red_percent * red_price) = 1856 / 100 := by
  sorry

end total_earnings_is_18_56_l4088_408838


namespace valid_numbers_l4088_408837

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_valid_sequence (a bc de fg : ℕ) : Prop :=
  2 ∣ a ∧
  is_prime bc ∧
  5 ∣ de ∧
  3 ∣ fg ∧
  fg - de = de - bc ∧
  de - bc = bc - a

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a bc de fg : ℕ),
    is_valid_sequence a bc de fg ∧
    n = de * 100 + bc

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 2013 ∨ n = 4023 := by sorry

end valid_numbers_l4088_408837


namespace cubic_tangent_max_value_l4088_408873

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_tangent_max_value (a b m : ℝ) (hm : m ≠ 0) :
  (f a b m = 0) →                          -- f(x) is zero at x = m
  (f' a b m = 0) →                         -- f'(x) is zero at x = m
  (∀ x, f a b x ≤ (1/2)) →                 -- maximum value of f(x) is 1/2
  (∃ x, f a b x = (1/2)) →                 -- f(x) achieves the maximum value 1/2
  m = (3/2) := by sorry

end cubic_tangent_max_value_l4088_408873


namespace ratio_problem_l4088_408800

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 1 / 4) :
  d / e = 1 / 4 := by
sorry

end ratio_problem_l4088_408800


namespace robin_gum_packages_l4088_408886

theorem robin_gum_packages (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 15) (h2 : total_pieces = 135) :
  total_pieces / pieces_per_package = 9 := by
  sorry

end robin_gum_packages_l4088_408886


namespace house_rent_fraction_l4088_408845

theorem house_rent_fraction (salary : ℝ) 
  (food_fraction : ℝ) (conveyance_fraction : ℝ) (left_amount : ℝ) (food_conveyance_expense : ℝ)
  (h1 : food_fraction = 3/10)
  (h2 : conveyance_fraction = 1/8)
  (h3 : left_amount = 1400)
  (h4 : food_conveyance_expense = 3400)
  (h5 : food_fraction * salary + conveyance_fraction * salary = food_conveyance_expense)
  (h6 : salary - (food_fraction * salary + conveyance_fraction * salary + left_amount) = 
        salary * (1 - food_fraction - conveyance_fraction - left_amount / salary)) :
  1 - food_fraction - conveyance_fraction - left_amount / salary = 2/5 := by
sorry

end house_rent_fraction_l4088_408845


namespace total_spent_on_toys_l4088_408831

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 := by
  sorry

end total_spent_on_toys_l4088_408831


namespace projection_theorem_l4088_408896

def vector_projection (u v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot_product := (u 0 * v 0 + u 1 * v 1)
  let norm_squared := (v 0 * v 0 + v 1 * v 1)
  fun i => (dot_product / norm_squared) * v i

def linear_transformation (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  vector_projection v (fun i => if i = 0 then 2 else -3)

theorem projection_theorem :
  let v : Fin 2 → ℚ := fun i => if i = 0 then 3 else -1
  let result := linear_transformation v
  result 0 = 18/13 ∧ result 1 = -27/13 := by
  sorry

end projection_theorem_l4088_408896


namespace commute_time_difference_l4088_408881

/-- Given a set of 5 commuting times (a, b, 8, 9, 10) with an average of 9 and a variance of 2, prove that |a-b| = 4 -/
theorem commute_time_difference (a b : ℝ) 
  (h_mean : (a + b + 8 + 9 + 10) / 5 = 9)
  (h_variance : ((a - 9)^2 + (b - 9)^2 + (8 - 9)^2 + (9 - 9)^2 + (10 - 9)^2) / 5 = 2) :
  |a - b| = 4 := by
sorry

end commute_time_difference_l4088_408881


namespace line_equations_l4088_408879

/-- Given a line passing through (-b, c) that cuts a triangular region with area U from the second quadrant,
    this theorem states the equations of the inclined line and the horizontal line passing through its y-intercept. -/
theorem line_equations (b c U : ℝ) (h_b : b > 0) (h_c : c > 0) (h_U : U > 0) :
  ∃ (m k : ℝ),
    (∀ x y, y = m * x + k ↔ 2 * U * x - b^2 * y + 2 * U * b + c * b^2 = 0) ∧
    (k = 2 * U / b + c) := by
  sorry

end line_equations_l4088_408879


namespace ants_crushed_calculation_l4088_408865

/-- The number of ants crushed by a man's foot, given the original number of ants and the number of ants left alive -/
def antsCrushed (originalAnts : ℕ) (antsAlive : ℕ) : ℕ :=
  originalAnts - antsAlive

/-- Theorem stating that 60 ants were crushed when 102 ants were originally present and 42 ants remained alive -/
theorem ants_crushed_calculation :
  antsCrushed 102 42 = 60 := by
  sorry

end ants_crushed_calculation_l4088_408865


namespace weight_of_b_l4088_408806

/-- Given three weights a, b, and c, prove that b = 31 under certain conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →  -- average of a, b, and c is 45
  (a + b) / 2 = 40 →      -- average of a and b is 40
  (b + c) / 2 = 43 →      -- average of b and c is 43
  b = 31 := by
    sorry


end weight_of_b_l4088_408806


namespace equation_represents_hyperbola_l4088_408846

/-- The equation x^2 - 16y^2 - 10x + 4y + 36 = 0 represents a hyperbola. -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (A B : ℝ),
    A > 0 ∧ B > 0 ∧
    ∀ x y : ℝ,
      x^2 - 16*y^2 - 10*x + 4*y + 36 = 0 ↔
      ((x - h)^2 / A - (y - k)^2 / B = 1 ∨ (x - h)^2 / A - (y - k)^2 / B = -1) :=
by sorry

end equation_represents_hyperbola_l4088_408846


namespace right_triangle_from_trig_equality_l4088_408887

theorem right_triangle_from_trig_equality (α β : Real) (h : 0 < α ∧ 0 < β ∧ α + β < Real.pi) :
  (Real.cos α + Real.cos β = Real.sin α + Real.sin β) → ∃ γ : Real, α + β + γ = Real.pi ∧ γ = Real.pi / 2 :=
by
  sorry

end right_triangle_from_trig_equality_l4088_408887


namespace yellow_shirt_pairs_l4088_408854

/-- Given a math contest with blue and yellow shirted students, this theorem proves
    the number of pairs where both students wear yellow shirts. -/
theorem yellow_shirt_pairs
  (total_students : ℕ)
  (blue_students : ℕ)
  (yellow_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)
  (h1 : total_students = blue_students + yellow_students)
  (h2 : total_students = 150)
  (h3 : blue_students = 65)
  (h4 : yellow_students = 85)
  (h5 : total_pairs = 75)
  (h6 : blue_blue_pairs = 30) :
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 40 ∧
  yellow_yellow_pairs + blue_blue_pairs + (total_students - 2 * blue_blue_pairs - 2 * yellow_yellow_pairs) / 2 = total_pairs :=
by sorry

end yellow_shirt_pairs_l4088_408854
