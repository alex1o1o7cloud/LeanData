import Mathlib

namespace NUMINAMATH_CALUDE_line_intersection_area_ratio_l2048_204853

theorem line_intersection_area_ratio (c : ℝ) (h1 : 0 < c) (h2 : c < 6) : 
  let P : ℝ × ℝ := (0, c)
  let Q : ℝ × ℝ := (c, 0)
  let S : ℝ × ℝ := (6, c - 6)
  let area_QRS := (1/2) * (6 - c) * (c - 6)
  let area_QOP := (1/2) * c * c
  area_QRS / area_QOP = 4/25 → c = 30/7 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_area_ratio_l2048_204853


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l2048_204877

/-- The ratio of distances traveled in alternating months -/
theorem travel_distance_ratio :
  ∀ (regular_distance : ℝ) (total_distance : ℝ) (x : ℝ),
    regular_distance = 400 →
    total_distance = 14400 →
    12 * regular_distance + 12 * (x * regular_distance) = total_distance →
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_ratio_l2048_204877


namespace NUMINAMATH_CALUDE_running_yardage_difference_l2048_204888

def player_yardage (total_yards pass_yards : ℕ) : ℕ :=
  total_yards - pass_yards

theorem running_yardage_difference (
  player_a_total player_a_pass player_b_total player_b_pass : ℕ
) (h1 : player_a_total = 150)
  (h2 : player_a_pass = 60)
  (h3 : player_b_total = 180)
  (h4 : player_b_pass = 80) :
  (player_yardage player_a_total player_a_pass : ℤ) - 
  (player_yardage player_b_total player_b_pass : ℤ) = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_running_yardage_difference_l2048_204888


namespace NUMINAMATH_CALUDE_sin_35pi_over_6_l2048_204896

theorem sin_35pi_over_6 : Real.sin (35 * π / 6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_35pi_over_6_l2048_204896


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2048_204897

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f (x + y)) = f (y * f x) + x^2) →
  (∀ x : ℝ, f x = x ∨ f x = -x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2048_204897


namespace NUMINAMATH_CALUDE_next_number_with_property_l2048_204817

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_number_with_property :
  has_property 1818 ∧
  (∀ m, 1818 < m ∧ m < 1832 → ¬ has_property m) ∧
  has_property 1832 := by sorry

end NUMINAMATH_CALUDE_next_number_with_property_l2048_204817


namespace NUMINAMATH_CALUDE_range_of_m_l2048_204850

-- Define the set of real numbers between 1 and 2
def OpenInterval := {x : ℝ | 1 < x ∧ x < 2}

-- Define the inequality condition
def InequalityCondition (m : ℝ) : Prop :=
  ∀ x ∈ OpenInterval, x^2 + m*x + 2 ≥ 0

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (InequalityCondition m) ↔ m ≥ -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2048_204850


namespace NUMINAMATH_CALUDE_abs_neg_one_fifth_l2048_204810

theorem abs_neg_one_fifth : |(-1 : ℚ) / 5| = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_fifth_l2048_204810


namespace NUMINAMATH_CALUDE_unique_A_value_l2048_204844

-- Define the ♣ operation
def clubsuit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

-- Theorem statement
theorem unique_A_value : ∃! A : ℝ, clubsuit A 3 = 73 ∧ A = 50/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_A_value_l2048_204844


namespace NUMINAMATH_CALUDE_eighth_grade_girls_l2048_204849

theorem eighth_grade_girls (total_students : ℕ) (boys girls : ℕ) : 
  total_students = 68 →
  boys = 2 * girls - 16 →
  total_students = boys + girls →
  girls = 28 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_girls_l2048_204849


namespace NUMINAMATH_CALUDE_fraction_simplification_l2048_204805

theorem fraction_simplification (m : ℝ) (hm : m ≠ 0) (hm1 : m ≠ 1) (hm2 : m ≠ -1) :
  ((m - 1) / m) / ((m^2 - 1) / m^2) = m / (m + 1) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2048_204805


namespace NUMINAMATH_CALUDE_custard_pie_price_per_slice_l2048_204803

/-- The price per slice of custard pie given the number of pies, slices per pie, and total earnings -/
def price_per_slice (num_pies : ℕ) (slices_per_pie : ℕ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (num_pies * slices_per_pie)

/-- Theorem stating that the price per slice of custard pie is $3 under given conditions -/
theorem custard_pie_price_per_slice :
  let num_pies : ℕ := 6
  let slices_per_pie : ℕ := 10
  let total_earnings : ℚ := 180
  price_per_slice num_pies slices_per_pie total_earnings = 3 := by
  sorry

end NUMINAMATH_CALUDE_custard_pie_price_per_slice_l2048_204803


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l2048_204882

/-- Given the weights of four textbooks, prove that the difference between
    the sum of the middle two weights and the difference between the
    largest and smallest weights is 2.5 pounds. -/
theorem textbook_weight_difference
  (chemistry_weight geometry_weight calculus_weight biology_weight : ℝ)
  (h1 : chemistry_weight = 7.125)
  (h2 : geometry_weight = 0.625)
  (h3 : calculus_weight = 5.25)
  (h4 : biology_weight = 3.75)
  : (calculus_weight + biology_weight) - (chemistry_weight - geometry_weight) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l2048_204882


namespace NUMINAMATH_CALUDE_mayo_savings_l2048_204837

/-- Proves the savings when buying mayo in bulk -/
theorem mayo_savings (costco_price : ℝ) (store_price : ℝ) (gallon_oz : ℝ) (bottle_oz : ℝ) :
  costco_price = 8 →
  store_price = 3 →
  gallon_oz = 128 →
  bottle_oz = 16 →
  (gallon_oz / bottle_oz) * store_price - costco_price = 16 := by
sorry

end NUMINAMATH_CALUDE_mayo_savings_l2048_204837


namespace NUMINAMATH_CALUDE_system_solution_l2048_204806

/-- The system of equations:
    y^2 = (x+8)(x^2 + 2)
    y^2 - (8+4x)y + (16+16x-5x^2) = 0
    has solutions (0, ±4), (-2, ±6), (-5, ±9), and (19, ±99) -/
theorem system_solution :
  ∀ (x y : ℝ),
    (y^2 = (x+8)*(x^2 + 2) ∧
     y^2 - (8+4*x)*y + (16+16*x-5*x^2) = 0) ↔
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨
     (x = -2 ∧ (y = 6 ∨ y = -6)) ∨
     (x = -5 ∧ (y = 9 ∨ y = -9)) ∨
     (x = 19 ∧ (y = 99 ∨ y = -99))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2048_204806


namespace NUMINAMATH_CALUDE_fibSeriesSum_l2048_204811

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the series sum
noncomputable def fibSeries : ℝ := ∑' n : ℕ, (fib (2 * n + 1) : ℝ) / (5 : ℝ) ^ n

-- Theorem statement
theorem fibSeriesSum : fibSeries = 35 / 3 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l2048_204811


namespace NUMINAMATH_CALUDE_negation_of_existence_square_leq_one_negation_l2048_204874

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, x < 1 ∧ p x) ↔ (∀ x, x < 1 → ¬ p x) :=
by sorry

theorem square_leq_one_negation :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 ≤ 1) ↔ (∀ x : ℝ, x < 1 → x^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_leq_one_negation_l2048_204874


namespace NUMINAMATH_CALUDE_prob_select_AB_l2048_204858

/-- The number of employees -/
def total_employees : ℕ := 4

/-- The number of employees to be selected -/
def selected_employees : ℕ := 2

/-- The probability of selecting at least one of A and B -/
def prob_at_least_one_AB : ℚ := 5/6

/-- Theorem stating the probability of selecting at least one of A and B -/
theorem prob_select_AB : 
  1 - (Nat.choose (total_employees - 2) selected_employees : ℚ) / (Nat.choose total_employees selected_employees : ℚ) = prob_at_least_one_AB :=
sorry

end NUMINAMATH_CALUDE_prob_select_AB_l2048_204858


namespace NUMINAMATH_CALUDE_triangle_area_with_60_degree_angle_l2048_204827

/-- The area of a triangle with one angle of 60 degrees and adjacent sides of 15 cm and 12 cm is 45√3 cm² -/
theorem triangle_area_with_60_degree_angle (a b : ℝ) (h1 : a = 15) (h2 : b = 12) :
  (1/2) * a * b * Real.sqrt 3 = 45 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_60_degree_angle_l2048_204827


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2048_204883

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2048_204883


namespace NUMINAMATH_CALUDE_inequality_representation_l2048_204845

theorem inequality_representation (x y : ℝ) : 
  abs x + abs y ≤ Real.sqrt (2 * (x^2 + y^2)) ∧ 
  Real.sqrt (2 * (x^2 + y^2)) ≤ 2 * max (abs x) (abs y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_representation_l2048_204845


namespace NUMINAMATH_CALUDE_duck_profit_l2048_204824

/-- Calculates the profit from buying and selling ducks -/
theorem duck_profit
  (num_ducks : ℕ)
  (cost_per_duck : ℝ)
  (weight_per_duck : ℝ)
  (sell_price_per_pound : ℝ)
  (h1 : num_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : sell_price_per_pound = 5) :
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_weight * sell_price_per_pound
  total_revenue - total_cost = 300 :=
by sorry

end NUMINAMATH_CALUDE_duck_profit_l2048_204824


namespace NUMINAMATH_CALUDE_honey_market_optimization_l2048_204801

/-- Represents the honey market in Milnlandia -/
structure HoneyMarket where
  /-- Inverse demand function: P = 310 - 3Q -/
  demand : ℝ → ℝ
  /-- Production cost per jar in milns -/
  cost : ℝ
  /-- Tax per jar in milns -/
  tax : ℝ

/-- Profit function for the honey producer -/
def profit (market : HoneyMarket) (quantity : ℝ) : ℝ :=
  (market.demand quantity) * quantity - market.cost * quantity - market.tax * quantity

/-- Tax revenue function for the government -/
def taxRevenue (market : HoneyMarket) (quantity : ℝ) : ℝ :=
  market.tax * quantity

/-- The statement to be proved -/
theorem honey_market_optimization (market : HoneyMarket) 
    (h_demand : ∀ q, market.demand q = 310 - 3 * q)
    (h_cost : market.cost = 10) :
  (∃ q_max : ℝ, q_max = 50 ∧ 
    ∀ q, profit market q ≤ profit market q_max) ∧
  (∃ t_max : ℝ, t_max = 150 ∧
    ∀ t, market.tax = t → 
      taxRevenue { market with tax := t } 
        ((310 - t) / 6) ≤ 
      taxRevenue { market with tax := t_max } 
        ((310 - t_max) / 6)) := by
  sorry


end NUMINAMATH_CALUDE_honey_market_optimization_l2048_204801


namespace NUMINAMATH_CALUDE_greater_number_sum_and_difference_l2048_204872

theorem greater_number_sum_and_difference (x y : ℝ) : 
  x + y = 30 → x - y = 6 → x > y → x = 18 := by sorry

end NUMINAMATH_CALUDE_greater_number_sum_and_difference_l2048_204872


namespace NUMINAMATH_CALUDE_lucys_cookies_l2048_204800

/-- Lucy's grocery shopping problem -/
theorem lucys_cookies (total_packs cake_packs cookie_packs : ℕ) : 
  total_packs = 27 → cake_packs = 4 → total_packs = cookie_packs + cake_packs → cookie_packs = 23 := by
  sorry

end NUMINAMATH_CALUDE_lucys_cookies_l2048_204800


namespace NUMINAMATH_CALUDE_factor_sum_l2048_204870

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 2*X + 5) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 31 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2048_204870


namespace NUMINAMATH_CALUDE_ines_shopping_result_l2048_204894

/-- Represents the shopping scenario for Ines at the farmers' market -/
def shopping_scenario (initial_amount : ℚ) (peach_price peach_qty cherry_price cherry_qty
                       baguette_price baguette_qty strawberry_price strawberry_qty
                       salad_price salad_qty : ℚ) : ℚ :=
  let total_cost := peach_price * peach_qty + cherry_price * cherry_qty +
                    baguette_price * baguette_qty + strawberry_price * strawberry_qty +
                    salad_price * salad_qty
  let discount_rate := if total_cost > 10 then 0.1 else 0 +
                       if peach_qty > 0 && cherry_qty > 0 && baguette_qty > 0 &&
                          strawberry_qty > 0 && salad_qty > 0
                       then 0.05 else 0
  let discounted_total := total_cost * (1 - discount_rate)
  let with_tax := discounted_total * 1.05
  let final_total := with_tax * 1.02
  initial_amount - final_total

/-- Theorem stating that Ines will be short by $4.58 after her shopping trip -/
theorem ines_shopping_result :
  shopping_scenario 20 2 3 3.5 2 1.25 4 4 1 2.5 2 = -4.58 := by
  sorry

end NUMINAMATH_CALUDE_ines_shopping_result_l2048_204894


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l2048_204843

theorem book_arrangement_problem (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 4) :
  Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l2048_204843


namespace NUMINAMATH_CALUDE_scientific_notation_of_8790000_l2048_204899

theorem scientific_notation_of_8790000 :
  8790000 = 8.79 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8790000_l2048_204899


namespace NUMINAMATH_CALUDE_rebus_no_solution_l2048_204813

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
def FourDigitNumber := Fin 10000

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Converts a four-digit number to its decimal representation -/
def toDecimal (n : FourDigitNumber) : ℕ := n.val

/-- Converts a five-digit number to its decimal representation -/
def toDecimalFive (n : FiveDigitNumber) : ℕ := n.val

/-- Constructs a four-digit number from individual digits -/
def makeNumber (k u s y : Digit) : FourDigitNumber :=
  ⟨k.val * 1000 + u.val * 100 + s.val * 10 + y.val, by sorry⟩

/-- Constructs a five-digit number from individual digits -/
def makeNumberFive (u k s y u' s' : Digit) : FiveDigitNumber :=
  ⟨u.val * 10000 + k.val * 1000 + s.val * 100 + y.val * 10 + u'.val, by sorry⟩

/-- The main theorem stating that the rebus has no solution -/
theorem rebus_no_solution :
  ¬∃ (k u s y : Digit),
    k ≠ u ∧ k ≠ s ∧ k ≠ y ∧ u ≠ s ∧ u ≠ y ∧ s ≠ y ∧
    toDecimal (makeNumber k u s y) + toDecimal (makeNumber u k s y) =
    toDecimalFive (makeNumberFive u k s y u s) :=
by sorry

end NUMINAMATH_CALUDE_rebus_no_solution_l2048_204813


namespace NUMINAMATH_CALUDE_yellow_marbles_fraction_l2048_204879

theorem yellow_marbles_fraction (total : ℝ) (h : total > 0) :
  let initial_green := (2/3) * total
  let initial_yellow := total - initial_green
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = 3/5 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_fraction_l2048_204879


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l2048_204809

theorem number_exceeding_fraction : ∃ x : ℝ, x = (5 / 9) * x + 150 ∧ x = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l2048_204809


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2048_204826

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem possible_values_of_a :
  ∃ S : Set ℝ, S = {a : ℝ | A a ∩ B a = {2, 5}} ∧ S = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2048_204826


namespace NUMINAMATH_CALUDE_circle_center_l2048_204875

theorem circle_center (c : ℝ × ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    (∀ p : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 → 
      (3 * p.1 + 4 * p.2 = 24 ∨ 3 * p.1 + 4 * p.2 = -6))) ∧ 
  c.1 - 3 * c.2 = 0 → 
  c = (27/13, 9/13) := by
sorry


end NUMINAMATH_CALUDE_circle_center_l2048_204875


namespace NUMINAMATH_CALUDE_blue_sky_project_exhibition_l2048_204865

theorem blue_sky_project_exhibition (n : ℕ) (m : ℕ) :
  n = 6 →
  m = 6 →
  (Nat.choose n 2) * (5^(n - 2)) = (Nat.choose 6 2) * 5^4 :=
by sorry

end NUMINAMATH_CALUDE_blue_sky_project_exhibition_l2048_204865


namespace NUMINAMATH_CALUDE_open_box_volume_l2048_204864

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume (sheet_length sheet_width cut_size : ℝ) 
  (h1 : sheet_length = 100)
  (h2 : sheet_width = 50)
  (h3 : cut_size = 10) : 
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 24000 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l2048_204864


namespace NUMINAMATH_CALUDE_rotation_volume_of_specific_trapezoid_l2048_204842

/-- A trapezoid with given properties -/
structure Trapezoid where
  larger_base : ℝ
  smaller_base : ℝ
  adjacent_angle : ℝ

/-- The volume of the solid formed by rotating the trapezoid about its larger base -/
def rotation_volume (t : Trapezoid) : ℝ := sorry

/-- The theorem stating the volume of the rotated trapezoid -/
theorem rotation_volume_of_specific_trapezoid :
  let t : Trapezoid := {
    larger_base := 8,
    smaller_base := 2,
    adjacent_angle := Real.pi / 4  -- 45° in radians
  }
  rotation_volume t = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rotation_volume_of_specific_trapezoid_l2048_204842


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l2048_204895

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  biology : ℕ
  chemistry : ℕ

/-- Calculates the average of marks --/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.biology + m.chemistry : ℚ) / 5

theorem chemistry_marks_proof (m : Marks) 
  (h1 : m.english = 73)
  (h2 : m.mathematics = 69)
  (h3 : m.physics = 92)
  (h4 : m.biology = 82)
  (h5 : average m = 76) :
  m.chemistry = 64 := by
sorry


end NUMINAMATH_CALUDE_chemistry_marks_proof_l2048_204895


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l2048_204835

theorem magnitude_of_complex_number (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 1/w = s) : 
  Complex.abs w = 1 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l2048_204835


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2048_204863

theorem meaningful_expression_range (x : ℝ) : 
  (∃ (y : ℝ), y = (Real.sqrt (x + 4)) / (x - 2)) ↔ (x ≥ -4 ∧ x ≠ 2) := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2048_204863


namespace NUMINAMATH_CALUDE_pyramid_edges_l2048_204807

/-- Represents a pyramid with a polygonal base. -/
structure Pyramid where
  base_sides : ℕ
  deriving Repr

/-- The number of faces in a pyramid. -/
def num_faces (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of vertices in a pyramid. -/
def num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of edges in a pyramid. -/
def num_edges (p : Pyramid) : ℕ := p.base_sides + p.base_sides

/-- Theorem: A pyramid with 16 faces and vertices combined has 14 edges. -/
theorem pyramid_edges (p : Pyramid) : 
  num_faces p + num_vertices p = 16 → num_edges p = 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edges_l2048_204807


namespace NUMINAMATH_CALUDE_product_of_primes_with_conditions_l2048_204892

theorem product_of_primes_with_conditions :
  ∃ (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧
    (r - q = 2 * p) ∧
    (r * q + p^2 = 676) ∧
    (p * q * r = 2001) := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_with_conditions_l2048_204892


namespace NUMINAMATH_CALUDE_total_marbles_lost_l2048_204836

def initial_marbles : ℕ := 120

def marbles_lost_outside (total : ℕ) : ℕ :=
  total / 4

def marbles_given_away (remaining : ℕ) : ℕ :=
  remaining / 2

def marbles_lost_bag_tear : ℕ := 10

theorem total_marbles_lost : 
  let remaining_after_outside := initial_marbles - marbles_lost_outside initial_marbles
  let remaining_after_giving := remaining_after_outside - marbles_given_away remaining_after_outside
  let final_remaining := remaining_after_giving - marbles_lost_bag_tear
  initial_marbles - final_remaining = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_lost_l2048_204836


namespace NUMINAMATH_CALUDE_min_moves_for_chess_like_coloring_l2048_204802

/-- Represents a cell in the 5x5 grid -/
inductive Cell
| white
| black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if two cells are neighbors -/
def are_neighbors (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

/-- Represents a move (changing colors of two neighboring cells) -/
structure Move where
  cell1 : Fin 5 × Fin 5
  cell2 : Fin 5 × Fin 5
  are_neighbors : are_neighbors cell1 cell2

/-- Applies a move to a grid -/
def apply_move (g : Grid) (m : Move) : Grid :=
  sorry

/-- Checks if a grid has a chess-like coloring -/
def is_chess_like (g : Grid) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_for_chess_like_coloring :
  ∃ (moves : List Move),
    moves.length = 12 ∧
    (∀ g : Grid, (∀ i j, g i j = Cell.white) →
      is_chess_like (moves.foldl apply_move g)) ∧
    (∀ (moves' : List Move),
      moves'.length < 12 →
      ¬∃ g : Grid, (∀ i j, g i j = Cell.white) ∧
        is_chess_like (moves'.foldl apply_move g)) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_for_chess_like_coloring_l2048_204802


namespace NUMINAMATH_CALUDE_triangle_properties_l2048_204860

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about the properties of a triangle -/
theorem triangle_properties (t : Triangle) :
  (Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) →
  (2 * t.a^2 = t.b^2 + t.c^2) ∧
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2048_204860


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2048_204889

theorem bobby_candy_problem (initial_candy : ℕ) (chocolate : ℕ) (candy_chocolate_diff : ℕ) :
  initial_candy = 38 →
  chocolate = 16 →
  candy_chocolate_diff = 58 →
  (initial_candy + chocolate + candy_chocolate_diff) - initial_candy = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2048_204889


namespace NUMINAMATH_CALUDE_otimes_inequality_iff_a_range_l2048_204816

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem otimes_inequality_iff_a_range :
  ∀ a : ℝ, (∀ x : ℝ, otimes x (x + a) < 1) ↔ (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_iff_a_range_l2048_204816


namespace NUMINAMATH_CALUDE_binomial_20_10_l2048_204808

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 31824)
                       (h2 : Nat.choose 18 9 = 48620)
                       (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 172822 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l2048_204808


namespace NUMINAMATH_CALUDE_quadratic_radical_problem_l2048_204838

-- Define what it means for two quadratic radicals to be of the same type
def same_type (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (p₁ p₂ : ℕ), c₁ > 0 ∧ c₂ > 0 ∧ 
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧
  Real.sqrt x = c₁ * Real.sqrt (p₁ : ℝ) ∧
  Real.sqrt y = c₂ * Real.sqrt (p₂ : ℝ) ∧
  c₁ = c₂

-- State the theorem
theorem quadratic_radical_problem (a : ℝ) :
  same_type (3*a - 4) 8 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_problem_l2048_204838


namespace NUMINAMATH_CALUDE_max_value_w_l2048_204878

theorem max_value_w (p q : ℝ) 
  (h1 : 2 * p - q ≥ 0) 
  (h2 : 3 * q - 2 * p ≥ 0) 
  (h3 : 6 - 2 * q ≥ 0) : 
  Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q) ≤ 3 * Real.sqrt 2 ∧
  (Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q) = 3 * Real.sqrt 2 ↔ p = 2 ∧ q = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_w_l2048_204878


namespace NUMINAMATH_CALUDE_cube_diagonal_pairs_l2048_204855

/-- The number of diagonals on the faces of a cube -/
def num_diagonals : ℕ := 12

/-- The total number of pairs of diagonals -/
def total_pairs : ℕ := num_diagonals.choose 2

/-- The number of pairs of diagonals that do not form a 60° angle -/
def non_60_degree_pairs : ℕ := 18

/-- The number of pairs of diagonals that form a 60° angle -/
def pairs_60_degree : ℕ := total_pairs - non_60_degree_pairs

theorem cube_diagonal_pairs :
  pairs_60_degree = 48 := by sorry

end NUMINAMATH_CALUDE_cube_diagonal_pairs_l2048_204855


namespace NUMINAMATH_CALUDE_student_count_l2048_204828

theorem student_count : ∃ n : ℕ, n < 40 ∧ n % 7 = 3 ∧ n % 6 = 1 ∧ n = 31 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2048_204828


namespace NUMINAMATH_CALUDE_alices_spending_l2048_204822

theorem alices_spending (B : ℝ) : 
  ∃ (book magazine : ℝ),
    book = 0.25 * (B - magazine) ∧
    magazine = 0.1 * (B - book) ∧
    book + magazine = (4/13) * B :=
by sorry

end NUMINAMATH_CALUDE_alices_spending_l2048_204822


namespace NUMINAMATH_CALUDE_solve_for_q_l2048_204857

theorem solve_for_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p*q = 9) :
  q = (9 + 3*Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l2048_204857


namespace NUMINAMATH_CALUDE_like_terms_sum_l2048_204861

/-- Given that x^(n+1)y^3 and (1/3)x^3y^(m-1) are like terms, prove that m + n = 6 -/
theorem like_terms_sum (m n : ℤ) : 
  (∃ (x y : ℝ), x^(n+1) * y^3 = (1/3) * x^3 * y^(m-1)) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l2048_204861


namespace NUMINAMATH_CALUDE_calculate_expression_l2048_204832

theorem calculate_expression : (-8) * 3 / ((-2)^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2048_204832


namespace NUMINAMATH_CALUDE_count_nines_in_range_l2048_204880

/-- The number of occurrences of the digit 9 in all integers from 1 to 1000 (inclusive) -/
def count_nines : ℕ := sorry

/-- The range of integers we're considering -/
def range_start : ℕ := 1
def range_end : ℕ := 1000

theorem count_nines_in_range : count_nines = 300 := by sorry

end NUMINAMATH_CALUDE_count_nines_in_range_l2048_204880


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2048_204841

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) :
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2048_204841


namespace NUMINAMATH_CALUDE_carousel_horses_count_l2048_204851

theorem carousel_horses_count :
  let blue_horses : ℕ := 3
  let purple_horses : ℕ := 3 * blue_horses
  let green_horses : ℕ := 2 * purple_horses
  let gold_horses : ℕ := green_horses / 6
  blue_horses + purple_horses + green_horses + gold_horses = 33 :=
by sorry

end NUMINAMATH_CALUDE_carousel_horses_count_l2048_204851


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_non_negative_l2048_204856

theorem negation_of_absolute_value_non_negative :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_non_negative_l2048_204856


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l2048_204818

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 7| + 1

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | 8/3 ≤ x ∧ x ≤ 6} :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x - 2 * |x - 1| ≤ a) → a ≥ -4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l2048_204818


namespace NUMINAMATH_CALUDE_randy_blocks_theorem_l2048_204854

/-- The number of blocks Randy used for the tower -/
def blocks_used : ℕ := 19

/-- The number of blocks Randy has left -/
def blocks_left : ℕ := 59

/-- The initial number of blocks Randy had -/
def initial_blocks : ℕ := blocks_used + blocks_left

theorem randy_blocks_theorem : initial_blocks = 78 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_theorem_l2048_204854


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2048_204834

/-- Given that the least common multiple of x, 15, and 21 is 105, 
    the greatest possible value of x is 105. -/
theorem greatest_x_with_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 ∧ ∃ y : ℕ, y > 105 → Nat.lcm y (Nat.lcm 15 21) > 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2048_204834


namespace NUMINAMATH_CALUDE_strawberry_distribution_l2048_204825

theorem strawberry_distribution (num_girls : ℕ) (strawberries_per_girl : ℕ) 
  (h1 : num_girls = 8) (h2 : strawberries_per_girl = 6) :
  num_girls * strawberries_per_girl = 48 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_distribution_l2048_204825


namespace NUMINAMATH_CALUDE_divisibility_arithmetic_progression_l2048_204812

theorem divisibility_arithmetic_progression (K : ℕ) :
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_arithmetic_progression_l2048_204812


namespace NUMINAMATH_CALUDE_length_AB_on_parabola_l2048_204829

/-- Parabola type -/
structure Parabola where
  a : ℝ
  C : ℝ × ℝ → Prop
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.C point

/-- Tangent line to a parabola at a point -/
def tangent_line (p : Parabola) (pt : PointOnParabola p) : ℝ × ℝ → Prop := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Length of AB on parabola y² = 6x -/
theorem length_AB_on_parabola (p : Parabola) 
  (h_eq : p.C = fun (x, y) ↦ y^2 = 6*x) 
  (A B : PointOnParabola p) 
  (F : ℝ × ℝ) 
  (h_focus : F = p.focus)
  (h_collinear : ∃ (m : ℝ), A.point.1 = m * A.point.2 + F.1 ∧ 
                             B.point.1 = m * B.point.2 + F.1)
  (P : ℝ × ℝ)
  (h_tangent_intersect : (tangent_line p A) P ∧ (tangent_line p B) P)
  (h_PF_distance : distance P F = 2 * Real.sqrt 3) :
  distance A.point B.point = 8 := by sorry

end NUMINAMATH_CALUDE_length_AB_on_parabola_l2048_204829


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l2048_204823

theorem min_value_a_plus_b (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y) ∧ (m = -3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l2048_204823


namespace NUMINAMATH_CALUDE_hazel_lemonade_cups_l2048_204859

/-- The number of cups of lemonade Hazel sold to kids on bikes -/
def cups_sold_to_kids : ℕ := 18

/-- The number of cups of lemonade Hazel gave to her friends -/
def cups_given_to_friends : ℕ := cups_sold_to_kids / 2

/-- The number of cups of lemonade Hazel drank herself -/
def cups_drunk_by_hazel : ℕ := 1

/-- The total number of cups of lemonade Hazel made -/
def total_cups : ℕ := 56

theorem hazel_lemonade_cups : 
  2 * (cups_sold_to_kids + cups_given_to_friends + cups_drunk_by_hazel) = total_cups := by
  sorry

#check hazel_lemonade_cups

end NUMINAMATH_CALUDE_hazel_lemonade_cups_l2048_204859


namespace NUMINAMATH_CALUDE_isabel_paper_left_l2048_204886

/-- Given that Isabel bought 900 pieces of paper initially and used 156 pieces,
    prove that she has 744 pieces left. -/
theorem isabel_paper_left : 
  let initial_paper : ℕ := 900
  let used_paper : ℕ := 156
  initial_paper - used_paper = 744 := by sorry

end NUMINAMATH_CALUDE_isabel_paper_left_l2048_204886


namespace NUMINAMATH_CALUDE_four_more_laps_needed_l2048_204840

/-- Calculates the number of additional laps needed to reach a total distance -/
def additional_laps_needed (total_distance : ℕ) (track_length : ℕ) (laps_run_per_person : ℕ) (num_people : ℕ) : ℕ :=
  let total_laps_run := laps_run_per_person * num_people
  let distance_covered := total_laps_run * track_length
  let remaining_distance := total_distance - distance_covered
  remaining_distance / track_length

/-- Theorem: Given the problem conditions, 4 additional laps are needed -/
theorem four_more_laps_needed :
  additional_laps_needed 2400 150 6 2 = 4 := by
  sorry

#eval additional_laps_needed 2400 150 6 2

end NUMINAMATH_CALUDE_four_more_laps_needed_l2048_204840


namespace NUMINAMATH_CALUDE_haley_deleted_files_l2048_204831

/-- The number of files deleted from a flash drive -/
def files_deleted (initial_music : ℕ) (initial_video : ℕ) (files_left : ℕ) : ℕ :=
  initial_music + initial_video - files_left

/-- Proof that 11 files were deleted from Haley's flash drive -/
theorem haley_deleted_files : files_deleted 27 42 58 = 11 := by
  sorry

end NUMINAMATH_CALUDE_haley_deleted_files_l2048_204831


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2048_204871

/-- Definition of a quadratic equation in x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation x^2 + 3x - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 3*x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2048_204871


namespace NUMINAMATH_CALUDE_quadratic_polynomial_roots_l2048_204804

theorem quadratic_polynomial_roots (x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = 8) (h_product : x₁ * x₂ = 16) :
  x₁ * x₂ = 16 ∧ x₁ + x₂ = 8 ↔ x₁^2 - 8*x₁ + 16 = 0 ∧ x₂^2 - 8*x₂ + 16 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_roots_l2048_204804


namespace NUMINAMATH_CALUDE_exists_column_with_many_zeros_l2048_204876

/-- Represents a row in the grid -/
def Row := Fin 6 → Fin 2

/-- The grid -/
def Grid (n : ℕ) := Fin n → Row

/-- Condition: integers in each row are distinct -/
def distinct_rows (g : Grid n) : Prop :=
  ∀ i j, i ≠ j → g i ≠ g j

/-- Condition: for any two rows, their element-wise product exists as a row -/
def product_exists (g : Grid n) : Prop :=
  ∀ i j, ∃ k, ∀ m, g k m = (g i m * g j m : Fin 2)

/-- Count of 0s in a column -/
def zero_count (g : Grid n) (col : Fin 6) : ℕ :=
  (Finset.filter (λ i => g i col = 0) Finset.univ).card

/-- Main theorem -/
theorem exists_column_with_many_zeros (n : ℕ) (hn : n ≥ 2) (g : Grid n)
  (h_distinct : distinct_rows g) (h_product : product_exists g) :
  ∃ col, zero_count g col ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_column_with_many_zeros_l2048_204876


namespace NUMINAMATH_CALUDE_prob_A_win_match_is_correct_l2048_204868

/-- The probability of player A winning a single game -/
def prob_A_win : ℝ := 0.6

/-- The probability of player B winning a single game -/
def prob_B_win : ℝ := 0.4

/-- The probability of player A winning the match after winning the first game -/
def prob_A_win_match : ℝ := prob_A_win + prob_B_win * prob_A_win

/-- Theorem stating that the probability of A winning the match after winning the first game is 0.84 -/
theorem prob_A_win_match_is_correct : prob_A_win_match = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_win_match_is_correct_l2048_204868


namespace NUMINAMATH_CALUDE_juice_reduction_fraction_l2048_204869

/-- Proves that the fraction of the original volume that the juice was reduced to is 1/12 --/
theorem juice_reduction_fraction (original_volume : ℚ) (quart_to_cup : ℚ) (sugar_added : ℚ) (final_volume : ℚ) :
  original_volume = 6 →
  quart_to_cup = 4 →
  sugar_added = 1 →
  final_volume = 3 →
  (final_volume - sugar_added) / (original_volume * quart_to_cup) = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_juice_reduction_fraction_l2048_204869


namespace NUMINAMATH_CALUDE_ratio_of_areas_ratio_of_perimeters_l2048_204852

-- Define the side lengths of squares A and B
def side_length_A : ℝ := 48
def side_length_B : ℝ := 60

-- Define the areas of squares A and B
def area_A : ℝ := side_length_A ^ 2
def area_B : ℝ := side_length_B ^ 2

-- Define the perimeters of squares A and B
def perimeter_A : ℝ := 4 * side_length_A
def perimeter_B : ℝ := 4 * side_length_B

-- Theorem stating the ratio of areas
theorem ratio_of_areas :
  area_A / area_B = 16 / 25 := by sorry

-- Theorem stating the ratio of perimeters
theorem ratio_of_perimeters :
  perimeter_A / perimeter_B = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_of_areas_ratio_of_perimeters_l2048_204852


namespace NUMINAMATH_CALUDE_auditorium_seats_l2048_204893

/-- Represents the number of seats in a row of an auditorium -/
def seats (x : ℕ) : ℕ := 2 * x + 18

theorem auditorium_seats :
  (seats 1 = 20) ∧
  (seats 19 = 56) ∧
  (seats 26 = 70) :=
by sorry

end NUMINAMATH_CALUDE_auditorium_seats_l2048_204893


namespace NUMINAMATH_CALUDE_fifth_month_sales_l2048_204830

def sales_1 : ℕ := 5435
def sales_2 : ℕ := 5927
def sales_3 : ℕ := 5855
def sales_4 : ℕ := 6230
def sales_6 : ℕ := 3991
def average_sale : ℕ := 5500
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 5562 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l2048_204830


namespace NUMINAMATH_CALUDE_translation_down_3_units_l2048_204898

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 3 * x - 1

def vertical_translation (h : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ h x - d

theorem translation_down_3_units :
  vertical_translation f 3 = g := by sorry

end NUMINAMATH_CALUDE_translation_down_3_units_l2048_204898


namespace NUMINAMATH_CALUDE_election_total_votes_l2048_204881

/-- Represents an election with two candidates -/
structure Election where
  totalValidVotes : ℕ
  invalidVotes : ℕ
  losingCandidatePercentage : ℚ
  voteDifference : ℕ

/-- The total number of polled votes in the election -/
def totalPolledVotes (e : Election) : ℕ :=
  e.totalValidVotes + e.invalidVotes

/-- Theorem stating the total polled votes for the given election scenario -/
theorem election_total_votes (e : Election) 
  (h1 : e.losingCandidatePercentage = 1/5) 
  (h2 : e.voteDifference = 500) 
  (h3 : e.invalidVotes = 10) :
  totalPolledVotes e = 843 := by
  sorry

end NUMINAMATH_CALUDE_election_total_votes_l2048_204881


namespace NUMINAMATH_CALUDE_seating_theorem_l2048_204839

/-- The number of desks in a row -/
def num_desks : ℕ := 6

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- The minimum number of empty desks required between students -/
def min_gap : ℕ := 1

/-- The number of ways to seat students in desks with the given constraints -/
def seating_arrangements (n_desks n_students min_gap : ℕ) : ℕ :=
  sorry

theorem seating_theorem :
  seating_arrangements num_desks num_students min_gap = 9 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2048_204839


namespace NUMINAMATH_CALUDE_quadratic_roots_l2048_204890

theorem quadratic_roots (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 6*x + c = 0 ↔ x = (-3 + Real.sqrt c) ∨ x = (-3 - Real.sqrt c)) → 
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2048_204890


namespace NUMINAMATH_CALUDE_power_product_equality_l2048_204848

theorem power_product_equality : (-4 : ℝ)^2013 * (-0.25 : ℝ)^2014 = -0.25 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2048_204848


namespace NUMINAMATH_CALUDE_quadratic_term_elimination_l2048_204821

/-- The polynomial in question -/
def polynomial (x m : ℝ) : ℝ := 3*x^2 - 10 - 2*x - 4*x^2 + m*x^2

/-- The coefficient of x^2 in the polynomial -/
def x_squared_coefficient (m : ℝ) : ℝ := 3 - 4 + m

theorem quadratic_term_elimination :
  ∃ (m : ℝ), x_squared_coefficient m = 0 ∧ m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_term_elimination_l2048_204821


namespace NUMINAMATH_CALUDE_jenny_jill_game_percentage_l2048_204866

theorem jenny_jill_game_percentage :
  -- Define the number of games Jenny played against Mark
  ∀ (games_with_mark : ℕ),
  -- Define Mark's wins
  ∀ (mark_wins : ℕ),
  -- Define Jenny's total wins
  ∀ (jenny_total_wins : ℕ),
  -- Conditions
  games_with_mark = 10 →
  mark_wins = 1 →
  jenny_total_wins = 14 →
  -- Conclusion: Jill's win percentage is 75%
  (((2 * games_with_mark) - (jenny_total_wins - (games_with_mark - mark_wins))) / (2 * games_with_mark) : ℚ) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_jenny_jill_game_percentage_l2048_204866


namespace NUMINAMATH_CALUDE_total_amount_spent_l2048_204884

/-- Calculates the total amount spent on a meal given the base food price, sales tax rate, and tip rate. -/
theorem total_amount_spent
  (food_price : ℝ)
  (sales_tax_rate : ℝ)
  (tip_rate : ℝ)
  (h1 : food_price = 150)
  (h2 : sales_tax_rate = 0.1)
  (h3 : tip_rate = 0.2) :
  food_price * (1 + sales_tax_rate) * (1 + tip_rate) = 198 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_spent_l2048_204884


namespace NUMINAMATH_CALUDE_games_in_specific_league_l2048_204862

/-- The number of games played in a season for a league with a given number of teams and repetitions -/
def games_in_season (num_teams : ℕ) (repetitions : ℕ) : ℕ :=
  (num_teams * (num_teams - 1) / 2) * repetitions

/-- Theorem stating the number of games in a season for a specific league setup -/
theorem games_in_specific_league : games_in_season 14 5 = 455 := by
  sorry

end NUMINAMATH_CALUDE_games_in_specific_league_l2048_204862


namespace NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l2048_204820

-- Problem 1
theorem solution_set_inequality_1 (x : ℝ) :
  (2 - x) / (x + 4) ≤ 0 ↔ x ≥ 2 ∨ x < -4 := by sorry

-- Problem 2
theorem solution_set_inequality_2 (x a : ℝ) :
  x^2 - 3*a*x + 2*a^2 ≥ 0 ↔
    (a > 0 → (x ≥ 2*a ∨ x ≤ a)) ∧
    (a < 0 → (x ≥ a ∨ x ≤ 2*a)) ∧
    (a = 0 → True) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l2048_204820


namespace NUMINAMATH_CALUDE_calculation_proofs_l2048_204815

theorem calculation_proofs :
  (7 - (-1/2) + 3/2 = 9) ∧
  ((-1)^99 + (1-5)^2 * (3/8) = 5) ∧
  (-(2^3) * (5/8) / (-1/3) - 6 * (2/3 - 1/2) = 14) := by
sorry

end NUMINAMATH_CALUDE_calculation_proofs_l2048_204815


namespace NUMINAMATH_CALUDE_button_remainder_l2048_204885

theorem button_remainder (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 3)
  (h4 : n % 5 = 3) : 
  n % 12 = 7 := by sorry

end NUMINAMATH_CALUDE_button_remainder_l2048_204885


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2048_204846

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, |x - 1| < 2 → x + 1 ≥ 0) ∧ 
  (∃ x : ℝ, x + 1 ≥ 0 ∧ |x - 1| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2048_204846


namespace NUMINAMATH_CALUDE_circle_centered_at_parabola_focus_l2048_204814

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  (x - parabola_focus.1)^2 + (y - parabola_focus.2)^2 = circle_radius^2

theorem circle_centered_at_parabola_focus :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + y^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_centered_at_parabola_focus_l2048_204814


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l2048_204891

theorem cube_roots_of_unity :
  let z₁ : ℂ := 1
  let z₂ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  (z₁^3 = 1) ∧ (z₂^3 = 1) ∧ (z₃^3 = 1) ∧
  ∀ z : ℂ, z^3 = 1 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l2048_204891


namespace NUMINAMATH_CALUDE_expected_value_biased_die_l2048_204873

/-- A biased die with six faces and specified winning conditions -/
structure BiasedDie where
  /-- The probability of rolling each number is 1/6 -/
  prob : Fin 6 → ℚ
  prob_eq : ∀ i, prob i = 1/6
  /-- The winnings for each roll -/
  winnings : Fin 6 → ℚ
  /-- Rolling 1 or 2 wins $5 -/
  win_12 : winnings 0 = 5 ∧ winnings 1 = 5
  /-- Rolling 3 or 4 wins $0 -/
  win_34 : winnings 2 = 0 ∧ winnings 3 = 0
  /-- Rolling 5 or 6 loses $4 -/
  lose_56 : winnings 4 = -4 ∧ winnings 5 = -4

/-- The expected value of winnings after one roll of the biased die is 1/3 -/
theorem expected_value_biased_die (d : BiasedDie) : 
  (Finset.univ.sum fun i => d.prob i * d.winnings i) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_biased_die_l2048_204873


namespace NUMINAMATH_CALUDE_divisor_between_l2048_204847

theorem divisor_between (n a b : ℕ) (hn : n > 8) (ha : a > 0) (hb : b > 0) 
  (hab : a < b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (heq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end NUMINAMATH_CALUDE_divisor_between_l2048_204847


namespace NUMINAMATH_CALUDE_tickets_spent_on_beanie_l2048_204867

/-- Proves the number of tickets spent on a beanie given initial tickets, additional tickets won, and final ticket count. -/
theorem tickets_spent_on_beanie 
  (initial_tickets : ℕ) 
  (additional_tickets : ℕ) 
  (final_tickets : ℕ) 
  (h1 : initial_tickets = 49)
  (h2 : additional_tickets = 6)
  (h3 : final_tickets = 30)
  : initial_tickets - (initial_tickets - final_tickets + additional_tickets) = 25 := by
  sorry

end NUMINAMATH_CALUDE_tickets_spent_on_beanie_l2048_204867


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2048_204887

-- Define the angle α
def α : Real := sorry

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem tan_alpha_plus_pi_fourth (h : P.fst = -Real.tan α ∧ P.snd = Real.tan α * P.fst) :
  Real.tan (α + π/4) = -1/3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2048_204887


namespace NUMINAMATH_CALUDE_tangent_lines_to_C_value_of_m_l2048_204833

-- Define the curve C
def curve_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define point P
def point_P : ℝ × ℝ := (3, -1)

-- Define the intersecting line
def line_L (x y : ℝ) : Prop :=
  x + 2*y + 5 = 0

-- Part 1: Tangent lines
theorem tangent_lines_to_C :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, curve_C 1 x y → (x = 3 ∨ 5*x + 12*y - 3 = 0) → 
      ((x - 3)^2 + (y + 1)^2 = k^2)) ∧
    (∀ x y : ℝ, (x = 3 ∨ 5*x + 12*y - 3 = 0) → 
      ((x - 3)^2 + (y + 1)^2 ≤ k^2)) :=
sorry

-- Part 2: Value of m
theorem value_of_m :
  ∃! m : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ,
      curve_C m x1 y1 ∧ curve_C m x2 y2 ∧
      line_L x1 y1 ∧ line_L x2 y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 20) ∧
    m = -20 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_C_value_of_m_l2048_204833


namespace NUMINAMATH_CALUDE_sturgeon_count_l2048_204819

/-- Represents the number of fish caught of each type -/
structure FishCaught where
  pikes : ℕ
  sturgeons : ℕ
  herrings : ℕ

/-- The total number of fish caught is the sum of all types -/
def totalFish (f : FishCaught) : ℕ :=
  f.pikes + f.sturgeons + f.herrings

theorem sturgeon_count (f : FishCaught) 
  (h1 : f.pikes = 30)
  (h2 : f.herrings = 75)
  (h3 : totalFish f = 145) : 
  f.sturgeons = 40 := by
  sorry

end NUMINAMATH_CALUDE_sturgeon_count_l2048_204819
