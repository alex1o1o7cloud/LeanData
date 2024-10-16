import Mathlib

namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1386_138689

theorem diophantine_equation_solutions :
  ∀ (a b : ℕ), (2017 : ℕ) ^ a = b ^ 6 - 32 * b + 1 ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1386_138689


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1386_138670

theorem rectangle_perimeter (h w : ℝ) : 
  h > 0 ∧ w > 0 ∧              -- positive dimensions
  h * w = 40 ∧                 -- area of rectangle is 40
  w > 2 * h ∧                  -- width more than twice the height
  h * (w - h) = 24 →           -- area of parallelogram after folding
  2 * h + 2 * w = 28 :=        -- perimeter of original rectangle
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1386_138670


namespace NUMINAMATH_CALUDE_horse_division_l1386_138613

theorem horse_division (total_horses : ℕ) (eldest_share middle_share youngest_share : ℕ) : 
  total_horses = 7 →
  eldest_share = 4 →
  middle_share = 2 →
  youngest_share = 1 →
  eldest_share + middle_share + youngest_share = total_horses →
  eldest_share = (total_horses + 1) / 2 →
  middle_share = (total_horses + 1) / 4 →
  youngest_share = (total_horses + 1) / 8 :=
by sorry

end NUMINAMATH_CALUDE_horse_division_l1386_138613


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l1386_138626

/-- Represents the dimensions of a rectangular base pyramid roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  height : ℝ
  area : ℝ

/-- Conditions for the roof dimensions -/
def roof_conditions (r : RoofDimensions) : Prop :=
  r.length = 4 * r.width ∧
  r.area = 1024 ∧
  r.height = 50 ∧
  r.area = r.length * r.width

/-- Theorem stating the difference between length and width -/
theorem roof_dimension_difference (r : RoofDimensions) 
  (h : roof_conditions r) : r.length - r.width = 48 := by
  sorry

#check roof_dimension_difference

end NUMINAMATH_CALUDE_roof_dimension_difference_l1386_138626


namespace NUMINAMATH_CALUDE_optimal_mask_pricing_l1386_138632

/-- Represents the cost and pricing model for masks during an epidemic --/
structure MaskPricing where
  costA : ℝ  -- Cost of type A masks
  costB : ℝ  -- Cost of type B masks
  sellingPrice : ℝ  -- Selling price of type B masks
  profit : ℝ  -- Daily average total profit

/-- Conditions for the mask pricing problem --/
def MaskPricingConditions (m : MaskPricing) : Prop :=
  m.costB = 2 * m.costA - 10 ∧  -- Condition 1
  6000 / m.costA = 10000 / m.costB ∧  -- Condition 2
  m.profit = (m.sellingPrice - m.costB) * (100 - 5 * (m.sellingPrice - 60))  -- Conditions 3 and 4 combined

/-- Theorem stating the optimal solution for the mask pricing problem --/
theorem optimal_mask_pricing :
  ∃ m : MaskPricing,
    MaskPricingConditions m ∧
    m.costA = 30 ∧
    m.costB = 50 ∧
    m.sellingPrice = 65 ∧
    m.profit = 1125 ∧
    ∀ m' : MaskPricing, MaskPricingConditions m' → m'.profit ≤ m.profit :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_mask_pricing_l1386_138632


namespace NUMINAMATH_CALUDE_probability_second_quality_l1386_138603

theorem probability_second_quality (p : ℝ) : 
  (1 - p^2 = 0.91) → p = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_quality_l1386_138603


namespace NUMINAMATH_CALUDE_roots_of_quartic_equation_l1386_138633

theorem roots_of_quartic_equation :
  let f : ℝ → ℝ := λ x => 7 * x^4 - 44 * x^3 + 78 * x^2 - 44 * x + 7
  ∃ (a b c d : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
    a = 2 ∧ 
    b = 1/2 ∧ 
    c = (8 + Real.sqrt 15) / 7 ∧ 
    d = (8 - Real.sqrt 15) / 7 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quartic_equation_l1386_138633


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l1386_138608

def total_trees : ℕ := 15
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := total_trees - birch_trees

theorem birch_tree_arrangement_probability :
  let favorable_arrangements := (non_birch_trees + 1).choose birch_trees
  let total_arrangements := total_trees.choose birch_trees
  (favorable_arrangements : ℚ) / total_arrangements = 6 / 143 := by
sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l1386_138608


namespace NUMINAMATH_CALUDE_calculate_expression_l1386_138673

theorem calculate_expression : (-1 : ℝ)^200 - (-1/2 : ℝ)^0 + (3⁻¹ : ℝ) * 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1386_138673


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1386_138697

/-- Calculate the total amount owed after one year with simple interest -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) :
  principal = 35 →
  rate = 0.05 →
  time = 1 →
  principal + principal * rate * time = 36.75 := by
  sorry


end NUMINAMATH_CALUDE_simple_interest_calculation_l1386_138697


namespace NUMINAMATH_CALUDE_yellow_bead_cost_l1386_138694

/-- The cost of a box of yellow beads, given the following conditions:
  * Red beads cost $1.30 per box
  * 10 boxes of mixed beads cost $1.72 per box
  * 4 boxes of each color (red and yellow) are used to make the 10 mixed boxes
-/
theorem yellow_bead_cost (red_cost : ℝ) (mixed_cost : ℝ) (red_boxes : ℕ) (yellow_boxes : ℕ) :
  red_cost = 1.30 →
  mixed_cost = 1.72 →
  red_boxes = 4 →
  yellow_boxes = 4 →
  red_boxes * red_cost + yellow_boxes * (3 : ℝ) = 10 * mixed_cost :=
by sorry

end NUMINAMATH_CALUDE_yellow_bead_cost_l1386_138694


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l1386_138649

theorem quadratic_roots_reciprocal_sum (m n : ℝ) : 
  (m^2 + 4*m - 1 = 0) → (n^2 + 4*n - 1 = 0) → (1/m + 1/n = 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l1386_138649


namespace NUMINAMATH_CALUDE_call_center_team_b_fraction_l1386_138612

/-- The fraction of total calls processed by team B in a call center with two teams -/
theorem call_center_team_b_fraction (team_a team_b : ℕ) (calls_a calls_b : ℚ) :
  team_a = (5 : ℚ) / 8 * team_b →
  calls_a = (7 : ℚ) / 5 * calls_b →
  (team_b * calls_b) / (team_a * calls_a + team_b * calls_b) = (8 : ℚ) / 15 := by
sorry


end NUMINAMATH_CALUDE_call_center_team_b_fraction_l1386_138612


namespace NUMINAMATH_CALUDE_fairview_population_l1386_138620

/-- The number of cities in the District of Fairview -/
def num_cities : ℕ := 25

/-- The average population of cities in the District of Fairview -/
def avg_population : ℕ := 3800

/-- The total population of the District of Fairview -/
def total_population : ℕ := num_cities * avg_population

theorem fairview_population :
  total_population = 95000 := by
  sorry

end NUMINAMATH_CALUDE_fairview_population_l1386_138620


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1386_138687

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem fourth_term_of_geometric_sequence :
  let a₁ : ℝ := 6
  let a₈ : ℝ := 186624
  let r : ℝ := (a₈ / a₁) ^ (1 / 7)
  geometric_sequence a₁ r 4 = 1296 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1386_138687


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l1386_138610

theorem video_recorder_wholesale_cost
  (wholesale_cost : ℝ)
  (retail_price : ℝ)
  (employee_price : ℝ)
  (h1 : retail_price = wholesale_cost * 1.2)
  (h2 : employee_price = retail_price * 0.95)
  (h3 : employee_price = 228)
  : wholesale_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l1386_138610


namespace NUMINAMATH_CALUDE_area_of_triangle_def_is_nine_l1386_138617

/-- A triangle with vertices on the sides of a rectangle -/
structure TriangleInRectangle where
  /-- Width of the rectangle -/
  width : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- x-coordinate of vertex D -/
  dx : ℝ
  /-- y-coordinate of vertex D -/
  dy : ℝ
  /-- x-coordinate of vertex E -/
  ex : ℝ
  /-- y-coordinate of vertex E -/
  ey : ℝ
  /-- x-coordinate of vertex F -/
  fx : ℝ
  /-- y-coordinate of vertex F -/
  fy : ℝ
  /-- Ensure D is on the left side of the rectangle -/
  hd : dx = 0 ∧ 0 ≤ dy ∧ dy ≤ height
  /-- Ensure E is on the bottom side of the rectangle -/
  he : ey = 0 ∧ 0 ≤ ex ∧ ex ≤ width
  /-- Ensure F is on the top side of the rectangle -/
  hf : fy = height ∧ 0 ≤ fx ∧ fx ≤ width

/-- Calculate the area of the triangle DEF -/
def areaOfTriangleDEF (t : TriangleInRectangle) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle DEF is 9 square units -/
theorem area_of_triangle_def_is_nine (t : TriangleInRectangle) 
    (h_width : t.width = 6) 
    (h_height : t.height = 4)
    (h_d : t.dx = 0 ∧ t.dy = 2)
    (h_e : t.ex = 6 ∧ t.ey = 0)
    (h_f : t.fx = 3 ∧ t.fy = 4) : 
  areaOfTriangleDEF t = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_def_is_nine_l1386_138617


namespace NUMINAMATH_CALUDE_percentage_difference_l1386_138640

theorem percentage_difference : (0.60 * 50) - (0.45 * 30) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1386_138640


namespace NUMINAMATH_CALUDE_hermione_badges_l1386_138637

theorem hermione_badges (total luna celestia : ℕ) (h1 : total = 83) (h2 : luna = 17) (h3 : celestia = 52) :
  total - luna - celestia = 14 := by
  sorry

end NUMINAMATH_CALUDE_hermione_badges_l1386_138637


namespace NUMINAMATH_CALUDE_greatest_product_prime_factorization_sum_l1386_138627

/-- The greatest product of positive integers summing to 2014 -/
def A : ℕ := 3^670 * 2^2

/-- The sum of all positive integers that produce A -/
def sum_of_factors : ℕ := 2014

/-- Function to calculate the sum of bases and exponents in prime factorization -/
def sum_bases_and_exponents (n : ℕ) : ℕ := sorry

theorem greatest_product_prime_factorization_sum :
  sum_bases_and_exponents A = 677 :=
sorry

end NUMINAMATH_CALUDE_greatest_product_prime_factorization_sum_l1386_138627


namespace NUMINAMATH_CALUDE_angle_order_l1386_138666

-- Define the angles of inclination
variable (α₁ α₂ α₃ : Real)

-- Define the slopes of the lines
def m₁ : Real := 1
def m₂ : Real := -1
def m₃ : Real := -2

-- Define the relationship between angles and slopes
axiom tan_α₁ : Real.tan α₁ = m₁
axiom tan_α₂ : Real.tan α₂ = m₂
axiom tan_α₃ : Real.tan α₃ = m₃

-- Theorem to prove
theorem angle_order : α₁ < α₃ ∧ α₃ < α₂ := by
  sorry

end NUMINAMATH_CALUDE_angle_order_l1386_138666


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1386_138641

theorem sum_of_reciprocals_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_neg_hundred : a * b * c = -100) : 
  1/a + 1/b + 1/c > 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1386_138641


namespace NUMINAMATH_CALUDE_remainder_problem_l1386_138656

theorem remainder_problem (y : ℤ) : 
  y % 23 = 19 → y % 276 = 180 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1386_138656


namespace NUMINAMATH_CALUDE_total_apples_is_340_l1386_138618

/-- The number of apples Kylie picked -/
def kylie_apples : ℕ := 66

/-- The number of apples Kayla picked -/
def kayla_apples : ℕ := 274

/-- The relationship between Kayla's and Kylie's apples -/
axiom kayla_kylie_relation : kayla_apples = 4 * kylie_apples + 10

/-- The total number of apples picked by Kylie and Kayla -/
def total_apples : ℕ := kylie_apples + kayla_apples

/-- Theorem: The total number of apples picked by Kylie and Kayla is 340 -/
theorem total_apples_is_340 : total_apples = 340 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_340_l1386_138618


namespace NUMINAMATH_CALUDE_student_height_removal_l1386_138611

theorem student_height_removal (N : ℕ) (heights : Finset ℤ) : 
  heights.card = 3 * N + 1 → ∃ (subset : Finset ℤ), 
    subset ⊆ heights ∧ 
    subset.card = N + 1 ∧ 
    ∀ (x y : ℤ), x ∈ subset → y ∈ subset → x ≠ y → |x - y| ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_student_height_removal_l1386_138611


namespace NUMINAMATH_CALUDE_min_value_theorem_l1386_138683

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1386_138683


namespace NUMINAMATH_CALUDE_abc_fraction_value_l1386_138646

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 2)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 7) :
  a * b * c / (a * b + b * c + c * a) = 140 / 59 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l1386_138646


namespace NUMINAMATH_CALUDE_trey_bracelet_sales_l1386_138678

/-- The average number of bracelets Trey needs to sell each day -/
def average_bracelets_per_day (total_cost : ℕ) (num_days : ℕ) (bracelet_price : ℕ) : ℚ :=
  (total_cost : ℚ) / (num_days : ℚ)

/-- Theorem stating that Trey needs to sell 8 bracelets per day on average -/
theorem trey_bracelet_sales :
  average_bracelets_per_day 112 14 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trey_bracelet_sales_l1386_138678


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l1386_138628

theorem hippopotamus_crayons (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 62)
  (h2 : remaining_crayons = 10) :
  initial_crayons - remaining_crayons = 52 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_crayons_l1386_138628


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1386_138600

theorem geometric_sequence_formula (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a (k + 1) = 3 * a k) →  -- Geometric sequence with common ratio 3
  a 1 = 4 →                     -- First term is 4
  a n = 4 * 3^(n - 1) :=        -- General formula
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1386_138600


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l1386_138661

/-- Calculates the total profit of a partnership given the investments and one partner's share --/
def calculate_total_profit (invest_a invest_b invest_c c_share : ℕ) : ℕ :=
  let total_parts := invest_a + invest_b + invest_c
  let c_parts := invest_c
  let profit_per_part := c_share / c_parts
  profit_per_part * total_parts

/-- Theorem stating that given the specific investments and C's share, the total profit is 90000 --/
theorem partnership_profit_calculation :
  calculate_total_profit 30000 45000 50000 36000 = 90000 := by
  sorry

#eval calculate_total_profit 30000 45000 50000 36000

end NUMINAMATH_CALUDE_partnership_profit_calculation_l1386_138661


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l1386_138659

theorem least_possible_third_side_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a = 5 → b = 12 →
  (a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + b^2 = c^2) →
  c ≥ Real.sqrt 119 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l1386_138659


namespace NUMINAMATH_CALUDE_population_equality_l1386_138688

/-- The number of years it takes for two villages' populations to be equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  18

theorem population_equality (x_initial y_initial x_decrease y_increase : ℕ)
  (h1 : x_initial = 78000)
  (h2 : x_decrease = 1200)
  (h3 : y_initial = 42000)
  (h4 : y_increase = 800) :
  x_initial - x_decrease * (years_until_equal_population x_initial x_decrease y_initial y_increase) =
  y_initial + y_increase * (years_until_equal_population x_initial x_decrease y_initial y_increase) :=
by sorry

end NUMINAMATH_CALUDE_population_equality_l1386_138688


namespace NUMINAMATH_CALUDE_line_mb_less_than_neg_one_l1386_138606

/-- A line passing through two points (0, 3) and (2, -1) -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  point1 : m * 0 + b = 3
  point2 : m * 2 + b = -1

/-- Theorem stating that for a line passing through (0, 3) and (2, -1), mb < -1 -/
theorem line_mb_less_than_neg_one (l : Line) : l.m * l.b < -1 := by
  sorry


end NUMINAMATH_CALUDE_line_mb_less_than_neg_one_l1386_138606


namespace NUMINAMATH_CALUDE_lemonade_pitchers_sum_l1386_138634

theorem lemonade_pitchers_sum : 
  let first_intermission : ℚ := 0.25
  let second_intermission : ℚ := 0.4166666666666667
  let third_intermission : ℚ := 0.25
  first_intermission + second_intermission + third_intermission = 0.9166666666666667 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_sum_l1386_138634


namespace NUMINAMATH_CALUDE_total_subjects_is_six_l1386_138636

/-- 
Given a student's marks:
- The average mark in n subjects is 74
- The average mark in 5 subjects is 74
- The mark in the last subject is 74
Prove that the total number of subjects is 6
-/
theorem total_subjects_is_six (n : ℕ) (average_n : ℝ) (average_5 : ℝ) (last_subject : ℝ) :
  average_n = 74 →
  average_5 = 74 →
  last_subject = 74 →
  n * average_n = 5 * average_5 + last_subject →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_subjects_is_six_l1386_138636


namespace NUMINAMATH_CALUDE_average_weight_problem_l1386_138664

/-- Given the average weight of three people and some additional information,
    prove that the average weight of two of them is 43 kg. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- average weight of a, b, and c
  (a + b) / 2 = 40 →       -- average weight of a and b
  b = 31 →                 -- weight of b
  (b + c) / 2 = 43         -- average weight of b and c to be proved
  := by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1386_138664


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1386_138674

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpLine : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpLinePlane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpPlane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perpLine m n → 
  perpLinePlane m α → 
  perpLinePlane n β → 
  perpPlane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1386_138674


namespace NUMINAMATH_CALUDE_lunch_change_calculation_l1386_138638

/-- Calculates the change received when buying lunch items --/
theorem lunch_change_calculation (hamburger_cost onion_rings_cost smoothie_cost amount_paid : ℕ) :
  hamburger_cost = 4 →
  onion_rings_cost = 2 →
  smoothie_cost = 3 →
  amount_paid = 20 →
  amount_paid - (hamburger_cost + onion_rings_cost + smoothie_cost) = 11 := by
  sorry

end NUMINAMATH_CALUDE_lunch_change_calculation_l1386_138638


namespace NUMINAMATH_CALUDE_range_of_b_l1386_138667

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem range_of_b (b c : ℝ) :
  (∃ x₀ : ℝ, f (f b c x₀) b c = 0 ∧ f b c x₀ ≠ 0) →
  b < 0 ∨ b ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1386_138667


namespace NUMINAMATH_CALUDE_equation_solution_l1386_138657

theorem equation_solution : ∃! x : ℚ, x + 2/5 = 8/15 + 1/3 ∧ x = 7/15 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1386_138657


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_11_l1386_138686

theorem smallest_positive_integer_ending_in_9_divisible_by_11 :
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧
  ∀ (m : ℕ), m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_11_l1386_138686


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l1386_138653

-- Define the cones and their properties
structure Cone where
  radius : ℝ
  height : ℝ
  volume : ℝ

-- Define the marble
def marbleRadius : ℝ := 1

-- Define the cones
def narrowCone : Cone := { radius := 3, height := 0, volume := 0 }
def wideCone : Cone := { radius := 6, height := 0, volume := 0 }

-- State that both cones contain the same amount of liquid
axiom equal_volume : narrowCone.volume = wideCone.volume

-- Define the rise in liquid level after dropping the marble
def liquidRise (c : Cone) : ℝ := sorry

-- Theorem to prove
theorem liquid_rise_ratio :
  liquidRise narrowCone / liquidRise wideCone = 4 := by sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l1386_138653


namespace NUMINAMATH_CALUDE_rearrangement_inequality_applications_l1386_138671

theorem rearrangement_inequality_applications :
  -- Definition of rearrangement inequality
  (∀ (a b c d : ℝ), a ≥ b → c ≥ d → a * c + b * d ≥ a * d + b * c) →
  -- Application to a² + b² ≥ 2ab
  (∀ (a b : ℝ), a^2 + b^2 ≥ 2*a*b) ∧
  -- Application to tourniquet lemma
  (∀ (a b c : ℝ), a ≥ b ∧ b ≥ c → a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧
  -- Application to Exercise 1 (special case of tourniquet lemma)
  (∀ (a b : ℝ), a ≥ b ∧ b ≥ 1 → a^2 + b^2 + 1 ≥ a*b + b + a) ∧
  -- Application to Exercise 2 (special case of a² + b² ≥ 2ab)
  (∀ (a : ℝ), a > 0 → (a + 1/a)^2 ≥ 4) ∧
  -- Application to x³ + y³ ≥ x²y + xy²
  (∀ (x y : ℝ), x ≥ y ∧ x > 0 ∧ y > 0 → x^3 + y^3 ≥ x^2*y + x*y^2) :=
by sorry

end NUMINAMATH_CALUDE_rearrangement_inequality_applications_l1386_138671


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1386_138621

-- Define a regular polygon
def RegularPolygon (n : ℕ) := {p : ℕ | p ≥ 3}

-- Define a decagon
def Decagon := RegularPolygon 10

-- Define the number of interior intersection points of diagonals
def InteriorIntersectionPoints (p : RegularPolygon 10) : ℕ := sorry

-- Define the number of ways to choose 4 vertices from 10
def Choose4From10 : ℕ := Nat.choose 10 4

-- Theorem statement
theorem decagon_diagonal_intersections (d : Decagon) : 
  InteriorIntersectionPoints d = Choose4From10 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1386_138621


namespace NUMINAMATH_CALUDE_solutions_to_quadratic_equation_l1386_138690

theorem solutions_to_quadratic_equation :
  ∀ x : ℝ, 3 * x^2 = 27 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_solutions_to_quadratic_equation_l1386_138690


namespace NUMINAMATH_CALUDE_dividing_sum_theorem_l1386_138602

def is_valid_solution (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  a ∣ (b + c + d) ∧
  b ∣ (a + c + d) ∧
  c ∣ (a + b + d) ∧
  d ∣ (a + b + c)

def basis_solutions : List (ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 3, 6), (1, 2, 6, 9), (1, 3, 8, 12), (1, 4, 5, 10), (1, 6, 14, 21), (2, 3, 10, 15)]

theorem dividing_sum_theorem :
  ∀ a b c d : ℕ, is_valid_solution a b c d →
    ∃ k : ℕ, ∃ (x y z w : ℕ), (x, y, z, w) ∈ basis_solutions ∧
      a = k * x ∧ b = k * y ∧ c = k * z ∧ d = k * w :=
sorry

end NUMINAMATH_CALUDE_dividing_sum_theorem_l1386_138602


namespace NUMINAMATH_CALUDE_subtracted_value_l1386_138681

theorem subtracted_value (N : ℝ) (x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 4) / 10 = 5) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1386_138681


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1386_138660

theorem trigonometric_problem (x : ℝ) (h : 3 * Real.sin (x/2) - Real.cos (x/2) = 0) :
  Real.tan x = 3/4 ∧ (Real.cos (2*x)) / (Real.sqrt 2 * Real.cos (π/4 + x) * Real.sin x) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1386_138660


namespace NUMINAMATH_CALUDE_max_gcd_lcm_l1386_138692

theorem max_gcd_lcm (x y z : ℕ) 
  (h : Nat.gcd (Nat.lcm x y) z * Nat.lcm (Nat.gcd x y) z = 1400) : 
  Nat.gcd (Nat.lcm x y) z ≤ 10 ∧ 
  ∃ (a b c : ℕ), Nat.gcd (Nat.lcm a b) c = 10 ∧ 
                 Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 1400 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_l1386_138692


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_ten_l1386_138655

theorem sum_of_fractions_equals_ten : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (5 / 10 : ℚ) + 
  (6 / 10 : ℚ) + (7 / 10 : ℚ) + (8 / 10 : ℚ) + (9 / 10 : ℚ) + (55 / 10 : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_ten_l1386_138655


namespace NUMINAMATH_CALUDE_no_solution_exists_l1386_138635

theorem no_solution_exists : ¬∃ (x y : ℤ), (x + y = 2021 ∧ (10*x + y = 2221 ∨ x + 10*y = 2221)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1386_138635


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l1386_138639

theorem largest_number_with_given_hcf_and_lcm_factors (a b : ℕ+) 
  (h_hcf : Nat.gcd a b = 52)
  (h_lcm : Nat.lcm a b = 52 * 11 * 12) :
  max a b = 624 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l1386_138639


namespace NUMINAMATH_CALUDE_kangaroo_population_change_l1386_138631

theorem kangaroo_population_change 
  (G : ℝ) -- Initial number of grey kangaroos
  (R : ℝ) -- Initial number of red kangaroos
  (h1 : G > 0) -- Assumption: initial grey kangaroo population is positive
  (h2 : R > 0) -- Assumption: initial red kangaroo population is positive
  (h3 : 1.28 * G / (0.72 * R) = R / G) -- Ratio reversal condition
  : (2.24 * G) / ((7/3) * G) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_population_change_l1386_138631


namespace NUMINAMATH_CALUDE_addends_satisfy_conditions_l1386_138645

/-- Represents the correct sum of two addends -/
def correct_sum : Nat := 982

/-- Represents the incorrect sum when one addend is missing a 0 in the units place -/
def incorrect_sum : Nat := 577

/-- Represents the first addend -/
def addend1 : Nat := 450

/-- Represents the second addend -/
def addend2 : Nat := 532

/-- Theorem stating that the two addends satisfy the problem conditions -/
theorem addends_satisfy_conditions : 
  (addend1 + addend2 = correct_sum) ∧ 
  (addend1 + (addend2 - 50) = incorrect_sum) := by
  sorry

#check addends_satisfy_conditions

end NUMINAMATH_CALUDE_addends_satisfy_conditions_l1386_138645


namespace NUMINAMATH_CALUDE_algebraic_equality_l1386_138642

theorem algebraic_equality (a b c : ℝ) : a - b + c = a - (b - c) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l1386_138642


namespace NUMINAMATH_CALUDE_distance_2_neg5_abs_calculations_abs_equation_solutions_min_value_expression_l1386_138648

-- Define the distance function on the number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem 1: Distance between 2 and -5
theorem distance_2_neg5 : distance 2 (-5) = 7 := by sorry

-- Theorem 2: Absolute value calculations
theorem abs_calculations : 
  (|-4 + 6| = 2) ∧ (|-2 - 4| = 6) := by sorry

-- Theorem 3: Solutions to |x+2| = 4
theorem abs_equation_solutions :
  ∀ x : ℝ, |x + 2| = 4 ↔ (x = -6 ∨ x = 2) := by sorry

-- Theorem 4: Minimum value of |x+1| + |x-3|
theorem min_value_expression :
  ∃ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 3| ≥ m) ∧ 
  (∃ x : ℝ, |x + 1| + |x - 3| = m) ∧ 
  (m = 4) := by sorry

end NUMINAMATH_CALUDE_distance_2_neg5_abs_calculations_abs_equation_solutions_min_value_expression_l1386_138648


namespace NUMINAMATH_CALUDE_prime_composite_inequality_l1386_138680

theorem prime_composite_inequality (n : ℕ) : 
  (Nat.Prime (2 * n - 1) → 
    ∀ (a : Fin n → ℕ), Function.Injective a → 
      ∃ i j : Fin n, (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∧
  (¬Nat.Prime (2 * n - 1) → 
    ∃ (a : Fin n → ℕ), Function.Injective a ∧
      ∀ i j : Fin n, (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_inequality_l1386_138680


namespace NUMINAMATH_CALUDE_last_two_average_l1386_138614

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 63 →
  ((list.take 3).sum / 3 : ℝ) = 58 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_last_two_average_l1386_138614


namespace NUMINAMATH_CALUDE_min_value_xyz_l1386_138615

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 18 ∧ x^2 + 4*x*y + y^2 + 3*z^2 = 63 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l1386_138615


namespace NUMINAMATH_CALUDE_work_completion_time_l1386_138682

/-- Given workers A and B, where:
  * A can complete the entire work in 15 days
  * A works for 5 days and then leaves
  * B completes the remaining work in 6 days
  This theorem proves that B alone can complete the entire work in 9 days -/
theorem work_completion_time (a_total_days b_completion_days : ℕ) 
  (a_worked_days : ℕ) (h1 : a_total_days = 15) (h2 : a_worked_days = 5) 
  (h3 : b_completion_days = 6) : 
  (b_completion_days : ℚ) / ((a_total_days - a_worked_days : ℚ) / a_total_days) = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1386_138682


namespace NUMINAMATH_CALUDE_min_phi_value_l1386_138643

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * (x + φ))

theorem min_phi_value (φ : ℝ) : 
  (φ > 0) →
  (∀ x, g x φ = g ((2 * π / 3) - x) φ) →
  (∀ ψ, ψ > 0 → (∀ x, g x ψ = g ((2 * π / 3) - x) ψ) → φ ≤ ψ) →
  φ = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_min_phi_value_l1386_138643


namespace NUMINAMATH_CALUDE_combination_equality_implies_three_l1386_138685

theorem combination_equality_implies_three (x : ℕ) : 
  (Nat.choose 5 x = Nat.choose 5 (x - 1)) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_three_l1386_138685


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1386_138625

/-- The number of people that can be seated in one seat of the Ferris wheel -/
def people_per_seat : ℕ := 9

/-- The number of seats on the Ferris wheel -/
def number_of_seats : ℕ := 2

/-- The total number of people that can ride the Ferris wheel at the same time -/
def total_riders : ℕ := people_per_seat * number_of_seats

theorem ferris_wheel_capacity : total_riders = 18 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1386_138625


namespace NUMINAMATH_CALUDE_coffee_consumption_ratio_l1386_138684

/-- Represents the number of coffees John used to buy daily -/
def old_coffee_count : ℕ := 4

/-- Represents the original price of each coffee in dollars -/
def old_coffee_price : ℚ := 2

/-- Represents the percentage increase in coffee price -/
def price_increase_percent : ℚ := 50

/-- Represents the amount John saves daily compared to his old spending in dollars -/
def daily_savings : ℚ := 2

/-- Theorem stating that the ratio of John's current coffee consumption to his previous consumption is 1:2 -/
theorem coffee_consumption_ratio :
  ∃ (new_coffee_count : ℕ),
    new_coffee_count * (old_coffee_price * (1 + price_increase_percent / 100)) = 
      old_coffee_count * old_coffee_price - daily_savings ∧
    new_coffee_count * 2 = old_coffee_count := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_ratio_l1386_138684


namespace NUMINAMATH_CALUDE_existence_of_increasing_pair_l1386_138616

theorem existence_of_increasing_pair {α : Type*} [LinearOrder α] (a b : ℕ → α) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q :=
sorry

end NUMINAMATH_CALUDE_existence_of_increasing_pair_l1386_138616


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1386_138695

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ), 
  a = 30 → b = 40 → c^2 = a^2 + b^2 → c = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1386_138695


namespace NUMINAMATH_CALUDE_expression_simplification_l1386_138629

theorem expression_simplification (x : ℝ) (hx : x > 0) :
  (x - 1) / (x^(3/4) + x^(1/2)) * (x^(1/2) + x^(1/4)) / (x^(1/2) + 1) * x^(1/4) + 1 = x^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1386_138629


namespace NUMINAMATH_CALUDE_spelling_bee_contest_l1386_138654

theorem spelling_bee_contest (initial_students : ℕ) : 
  (initial_students : ℚ) * (1 - 0.66) * (1 - 3/4) = 30 →
  initial_students = 120 := by
sorry

end NUMINAMATH_CALUDE_spelling_bee_contest_l1386_138654


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1386_138658

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ a > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1386_138658


namespace NUMINAMATH_CALUDE_problem_solution_l1386_138665

theorem problem_solution (a b c : ℝ) (m n : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq1 : (a + b) * (a + c) = b * c + 2)
  (h_eq2 : (b + c) * (b + a) = c * a + 5)
  (h_eq3 : (c + a) * (c + b) = a * b + 9)
  (h_abc : a * b * c = m / n)
  (h_coprime : Nat.Coprime m n) : 
  100 * m + n = 4532 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1386_138665


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l1386_138601

def carrot_sticks_before_dinner (total : ℕ) (after_dinner : ℕ) : ℕ :=
  total - after_dinner

theorem james_carrot_sticks :
  carrot_sticks_before_dinner 37 15 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l1386_138601


namespace NUMINAMATH_CALUDE_painting_cost_is_147_l1386_138604

/-- Represents a side of the street with houses --/
structure StreetSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the cost of painting house numbers for a given street side --/
def calculate_side_cost (side : StreetSide) : ℕ := sorry

/-- Calculates the additional cost for numbers that are multiples of 10 --/
def calculate_multiples_of_10_cost (south : StreetSide) (north : StreetSide) : ℕ := sorry

/-- Main theorem: The total cost of painting all house numbers is $147 --/
theorem painting_cost_is_147 
  (south : StreetSide)
  (north : StreetSide)
  (h_south : south = { start := 5, diff := 7, count := 25 })
  (h_north : north = { start := 6, diff := 8, count := 25 }) :
  calculate_side_cost south + calculate_side_cost north + 
  calculate_multiples_of_10_cost south north = 147 := by sorry

end NUMINAMATH_CALUDE_painting_cost_is_147_l1386_138604


namespace NUMINAMATH_CALUDE_complex_power_difference_l1386_138644

theorem complex_power_difference (i : ℂ) : i^2 = -1 → i^123 - i^45 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1386_138644


namespace NUMINAMATH_CALUDE_household_electricity_most_suitable_l1386_138696

/-- Represents an investigation option --/
inductive InvestigationOption
  | ProductPopularity
  | TVViewershipRatings
  | AmmunitionExplosivePower
  | HouseholdElectricityConsumption

/-- Defines what makes an investigation suitable for a census method --/
def suitableForCensus (option : InvestigationOption) : Prop :=
  match option with
  | InvestigationOption.HouseholdElectricityConsumption => True
  | _ => False

/-- Theorem stating that investigating household electricity consumption is most suitable for census --/
theorem household_electricity_most_suitable :
    ∀ option : InvestigationOption,
      suitableForCensus option →
      option = InvestigationOption.HouseholdElectricityConsumption :=
by
  sorry

/-- Definition of a census method --/
def censusMethod (population : Type) (examine : population → Prop) : Prop :=
  ∀ subject : population, examine subject

#check household_electricity_most_suitable

end NUMINAMATH_CALUDE_household_electricity_most_suitable_l1386_138696


namespace NUMINAMATH_CALUDE_duck_selling_price_l1386_138605

/-- Calculates the selling price per pound of ducks -/
def selling_price_per_pound (num_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (total_profit : ℚ) : ℚ :=
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_cost + total_profit
  total_revenue / total_weight

/-- Proves that the selling price per pound is $5 given the problem conditions -/
theorem duck_selling_price :
  selling_price_per_pound 30 10 4 300 = 5 := by
  sorry

#eval selling_price_per_pound 30 10 4 300

end NUMINAMATH_CALUDE_duck_selling_price_l1386_138605


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1386_138676

theorem sum_of_roots_quadratic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let roots := {x : ℝ | f x = d}
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z, z ∈ roots → z = x ∨ z = y) →
  x + y = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x + 8
  let roots := {x : ℝ | f x = 15}
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z, z ∈ roots → z = x ∨ z = y) →
  x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1386_138676


namespace NUMINAMATH_CALUDE_m_range_theorem_l1386_138691

-- Define propositions p and q as functions of x and m
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)
def q (x : ℝ) : Prop := x^2 + 3*x - 4 < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -7 ∨ m ≥ 1

-- Theorem statement
theorem m_range_theorem :
  (∀ x m : ℝ, q x → p x m) ∧ 
  (∃ x m : ℝ, p x m ∧ ¬(q x)) →
  ∀ m : ℝ, m_range m ↔ ∃ x : ℝ, p x m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1386_138691


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1386_138630

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1386_138630


namespace NUMINAMATH_CALUDE_fifth_month_sale_l1386_138677

def sales_1 : ℕ := 5921
def sales_2 : ℕ := 5468
def sales_3 : ℕ := 5568
def sales_4 : ℕ := 6088
def sales_6 : ℕ := 5922
def average_sale : ℕ := 5900
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sales_5 : ℕ),
    sales_5 = average_sale * num_months - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6433 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l1386_138677


namespace NUMINAMATH_CALUDE_pell_equation_infinite_solutions_l1386_138650

theorem pell_equation_infinite_solutions (a : ℤ) (h_a : a > 1) 
  (u v : ℤ) (h_sol : u^2 - a * v^2 = -1) : 
  ∃ (S : Set (ℤ × ℤ)), Infinite S ∧ ∀ (p : ℤ × ℤ), p ∈ S → (p.1)^2 - a * (p.2)^2 = -1 :=
sorry

end NUMINAMATH_CALUDE_pell_equation_infinite_solutions_l1386_138650


namespace NUMINAMATH_CALUDE_triangle_side_length_l1386_138699

theorem triangle_side_length (A B C : ℝ) (h1 : Real.cos (3*A - B) + Real.sin (A + B) = 2) 
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π) (h5 : A + B + C = π) 
  (h6 : (4 : ℝ) = 4 * Real.sin A / Real.sin C) : 
  4 * Real.sin B / Real.sin C = 2 * Real.sqrt (2 - Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1386_138699


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1386_138679

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 11 * x - 20 = 0 :=
by
  -- The unique positive solution is x = 4/3
  use 4/3
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1386_138679


namespace NUMINAMATH_CALUDE_percentage_written_second_week_l1386_138651

/-- Proves that the percentage of remaining pages written in the second week is 30% --/
theorem percentage_written_second_week :
  ∀ (total_pages : ℕ) 
    (first_week_pages : ℕ) 
    (damaged_percentage : ℚ) 
    (final_empty_pages : ℕ),
  total_pages = 500 →
  first_week_pages = 150 →
  damaged_percentage = 20 / 100 →
  final_empty_pages = 196 →
  ∃ (second_week_percentage : ℚ),
    second_week_percentage = 30 / 100 ∧
    final_empty_pages = 
      (1 - damaged_percentage) * 
      (total_pages - first_week_pages - 
       (second_week_percentage * (total_pages - first_week_pages))) :=
by sorry

end NUMINAMATH_CALUDE_percentage_written_second_week_l1386_138651


namespace NUMINAMATH_CALUDE_complex_power_sum_l1386_138624

open Complex

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + (1 / z)^1500 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1386_138624


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1386_138609

/-- Given α = 60°, for a point P(a,b) on the terminal side of α where a > 0 and b > 0,
    the eccentricity of the hyperbola x²/a² - y²/b² = 1 is 2. -/
theorem hyperbola_eccentricity (α : Real) (a b : Real) :
  α = Real.pi / 3 →
  a > 0 →
  b > 0 →
  b / a = Real.sqrt 3 →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1386_138609


namespace NUMINAMATH_CALUDE_samuel_remaining_money_l1386_138698

theorem samuel_remaining_money (total : ℝ) (samuel_fraction : ℝ) (spent_fraction : ℝ) : 
  total = 240 →
  samuel_fraction = 3/8 →
  spent_fraction = 1/5 →
  samuel_fraction * total - spent_fraction * total = 42 := by
sorry

end NUMINAMATH_CALUDE_samuel_remaining_money_l1386_138698


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l1386_138663

/-- Given a rectangular room carpeted with multiple strips of carpet, calculate the breadth of the room. -/
theorem room_breadth_calculation 
  (room_length : ℝ)
  (carpet_width : ℝ)
  (carpet_cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : room_length = 15)
  (h2 : carpet_width = 0.75)
  (h3 : carpet_cost_per_meter = 0.30)
  (h4 : total_cost = 36) :
  (total_cost / carpet_cost_per_meter) / room_length * carpet_width = 6 :=
by sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l1386_138663


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1386_138607

/-- A quadratic function with vertex at (2,1) and opening upward -/
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

theorem quadratic_function_properties :
  (∀ x y z : ℝ, f ((x + y) / 2) ≤ (f x + f y) / 2) ∧  -- Convexity (implies upward opening)
  (∀ x : ℝ, f x ≥ f 2) ∧                              -- Minimum at x = 2
  f 2 = 1                                             -- Vertex y-coordinate is 1
:= by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1386_138607


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1386_138668

theorem complex_fraction_simplification :
  (5 + 6 * Complex.I) / (2 - 3 * Complex.I) = (-8 : ℚ) / 13 + (27 : ℚ) / 13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1386_138668


namespace NUMINAMATH_CALUDE_simplify_expression_l1386_138693

theorem simplify_expression (a : ℝ) : (36 * a ^ 9) ^ 4 * (63 * a ^ 9) ^ 4 = a ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1386_138693


namespace NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l1386_138662

/-- The number of coins Lucy flips -/
def n : ℕ := 12

/-- The probability of getting fewer heads than tails when flipping n coins -/
def prob_fewer_heads (n : ℕ) : ℚ :=
  793 / 2048

theorem prob_fewer_heads_12_coins : 
  prob_fewer_heads n = 793 / 2048 := by sorry

end NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l1386_138662


namespace NUMINAMATH_CALUDE_range_of_m_l1386_138623

-- Define the conditions
def p (x : ℝ) : Prop := (x - 2) * (x - 6) ≤ 32
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1386_138623


namespace NUMINAMATH_CALUDE_largest_multiple_under_500_l1386_138672

theorem largest_multiple_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_under_500_l1386_138672


namespace NUMINAMATH_CALUDE_erased_number_l1386_138675

/-- Represents a quadratic polynomial ax^2 + bx + c with roots m and n -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  m : ℤ
  n : ℤ

/-- Checks if the given QuadraticPolynomial satisfies Vieta's formulas -/
def satisfiesVieta (p : QuadraticPolynomial) : Prop :=
  p.c = p.a * p.m * p.n ∧ p.b = -p.a * (p.m + p.n)

/-- Checks if four of the five numbers in the QuadraticPolynomial are 2, 3, 4, -5 -/
def hasFourOf (p : QuadraticPolynomial) : Prop :=
  (p.a = 2 ∧ p.b = 4 ∧ p.m = 3 ∧ p.n = -5) ∨
  (p.a = 2 ∧ p.b = 4 ∧ p.m = 3 ∧ p.c = -5) ∨
  (p.a = 2 ∧ p.b = 4 ∧ p.n = -5 ∧ p.c = 3) ∨
  (p.a = 2 ∧ p.m = 3 ∧ p.n = -5 ∧ p.c = 4) ∨
  (p.b = 4 ∧ p.m = 3 ∧ p.n = -5 ∧ p.c = 2)

theorem erased_number (p : QuadraticPolynomial) :
  satisfiesVieta p → hasFourOf p → 
  p.a = -30 ∨ p.b = -30 ∨ p.c = -30 ∨ p.m = -30 ∨ p.n = -30 := by
  sorry


end NUMINAMATH_CALUDE_erased_number_l1386_138675


namespace NUMINAMATH_CALUDE_maze_paths_count_l1386_138622

/-- Represents a branching point in the maze -/
inductive BranchingPoint
  | Major
  | Minor

/-- Represents the structure of the maze -/
structure Maze where
  entrance : Unit
  exit : Unit
  initialChoices : Nat
  majorToMinor : Nat
  minorChoices : Nat

/-- Calculates the number of paths through the maze -/
def numberOfPaths (maze : Maze) : Nat :=
  maze.initialChoices * (maze.minorChoices ^ maze.majorToMinor)

/-- The specific maze from the problem -/
def problemMaze : Maze :=
  { entrance := ()
  , exit := ()
  , initialChoices := 2
  , majorToMinor := 3
  , minorChoices := 2
  }

theorem maze_paths_count :
  numberOfPaths problemMaze = 16 := by
  sorry

#eval numberOfPaths problemMaze

end NUMINAMATH_CALUDE_maze_paths_count_l1386_138622


namespace NUMINAMATH_CALUDE_older_brother_allowance_l1386_138669

theorem older_brother_allowance (younger_allowance older_allowance : ℕ) : 
  younger_allowance + older_allowance = 12000 →
  older_allowance = younger_allowance + 1000 →
  older_allowance = 6500 := by
sorry

end NUMINAMATH_CALUDE_older_brother_allowance_l1386_138669


namespace NUMINAMATH_CALUDE_square_starts_with_sequence_l1386_138647

theorem square_starts_with_sequence (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k :=
by sorry

end NUMINAMATH_CALUDE_square_starts_with_sequence_l1386_138647


namespace NUMINAMATH_CALUDE_frances_towel_weight_frances_towel_weight_is_240_ounces_l1386_138619

/-- Calculates the weight of Frances's towels in ounces given the conditions of the beach towel problem -/
theorem frances_towel_weight (mary_towel_count : ℕ) (total_weight_pounds : ℕ) : ℕ :=
  let frances_towel_count := mary_towel_count / 4
  let total_weight_ounces := total_weight_pounds * 16
  let mary_towel_weight_ounces := (total_weight_ounces / mary_towel_count) * mary_towel_count
  let frances_towel_weight_ounces := total_weight_ounces - mary_towel_weight_ounces
  frances_towel_weight_ounces

/-- Proves that Frances's towels weigh 240 ounces given the conditions of the beach towel problem -/
theorem frances_towel_weight_is_240_ounces : frances_towel_weight 24 60 = 240 := by
  sorry

end NUMINAMATH_CALUDE_frances_towel_weight_frances_towel_weight_is_240_ounces_l1386_138619


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l1386_138652

theorem perfect_square_pairs (x y : ℕ) :
  (∃ a : ℕ, x^2 + 8*y = a^2) ∧ (∃ b : ℕ, y^2 - 8*x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨
  ((x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l1386_138652
