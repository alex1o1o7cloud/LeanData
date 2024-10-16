import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_proof_l126_12629

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The given line 2x - 3y + 4 = 0 -/
def given_line : Line :=
  { a := 2, b := -3, c := 4 }

/-- The point (-1, 2) -/
def point : (ℝ × ℝ) :=
  (-1, 2)

/-- The equation of the line we want to prove -/
def target_line : Line :=
  { a := 2, b := -3, c := 8 }

theorem line_equation_proof :
  parallel target_line given_line ∧
  point_on_line point.1 point.2 target_line :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l126_12629


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l126_12666

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l126_12666


namespace NUMINAMATH_CALUDE_four_block_selection_count_l126_12678

def grid_size : ℕ := 6
def blocks_to_select : ℕ := 4

theorem four_block_selection_count :
  (Nat.choose grid_size blocks_to_select) *
  (Nat.choose grid_size blocks_to_select) *
  (Nat.factorial blocks_to_select) = 5400 := by
  sorry

end NUMINAMATH_CALUDE_four_block_selection_count_l126_12678


namespace NUMINAMATH_CALUDE_inequality_proof_l126_12686

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_sum : a + b + c + d = 1) : 
  a^2 / (1 + a) + b^2 / (1 + b) + c^2 / (1 + c) + d^2 / (1 + d) ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l126_12686


namespace NUMINAMATH_CALUDE_xyz_value_l126_12642

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l126_12642


namespace NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen_l126_12657

theorem remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen :
  (13^13 + 13) % 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen_l126_12657


namespace NUMINAMATH_CALUDE_max_notebooks_purchase_l126_12611

theorem max_notebooks_purchase (notebook_price : ℕ) (available_money : ℚ) : 
  notebook_price = 45 → available_money = 40.5 → 
  ∃ max_notebooks : ℕ, max_notebooks = 90 ∧ 
  (max_notebooks : ℚ) * (notebook_price : ℚ) / 100 ≤ available_money ∧
  ∀ n : ℕ, (n : ℚ) * (notebook_price : ℚ) / 100 ≤ available_money → n ≤ max_notebooks :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchase_l126_12611


namespace NUMINAMATH_CALUDE_train_length_l126_12694

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 80 * (5/18) → time = 9 → speed * time = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l126_12694


namespace NUMINAMATH_CALUDE_randi_has_more_nickels_l126_12637

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Ray's initial amount in cents -/
def ray_initial_amount : ℕ := 175

/-- Amount given to Peter in cents -/
def amount_to_peter : ℕ := 30

/-- Amount given to Randi in cents -/
def amount_to_randi : ℕ := 2 * amount_to_peter

/-- Number of nickels Randi receives -/
def randi_nickels : ℕ := amount_to_randi / nickel_value

/-- Number of nickels Peter receives -/
def peter_nickels : ℕ := amount_to_peter / nickel_value

theorem randi_has_more_nickels : randi_nickels - peter_nickels = 6 := by
  sorry

end NUMINAMATH_CALUDE_randi_has_more_nickels_l126_12637


namespace NUMINAMATH_CALUDE_square_difference_formula_l126_12647

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 2/15) : x^2 - y^2 = 16/225 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l126_12647


namespace NUMINAMATH_CALUDE_ellipse_foci_l126_12638

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 = 8

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- Theorem: The foci of the ellipse 2x^2 + y^2 = 8 are at (0, ±2) -/
theorem ellipse_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci_coordinates ↔ 
  (∃ (a b c : ℝ), 
    (∀ x y, ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (a > b) ∧
    (c^2 = a^2 - b^2) ∧
    (f = (0, c) ∨ f = (0, -c))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l126_12638


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l126_12685

def digits : Finset ℕ := {1, 2, 3, 5, 5, 8}

def is_valid_arrangement (n : ℕ) : Prop :=
  n ≥ 100000 ∧ n < 1000000 ∧ ∀ d, d ∈ digits.toList.map Nat.digitChar → d ∈ n.repr.data

def is_divisible_by_15 (n : ℕ) : Prop := n % 15 = 0

def total_arrangements : ℕ := 6 * 5 * 4 * 3 * 2 * 1

def favorable_arrangements : ℕ := 2 * (5 * 4 * 3 * 2 * 1)

theorem probability_divisible_by_15 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l126_12685


namespace NUMINAMATH_CALUDE_perfect_line_fit_l126_12699

/-- A sample point in a 2D plane -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : SamplePoint) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Calculate the sum of squared residuals -/
def sumSquaredResiduals (points : List SamplePoint) (l : Line) : ℝ :=
  (points.map (fun p => (p.y - (l.slope * p.x + l.intercept))^2)).sum

/-- Calculate the correlation coefficient -/
def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry  -- Actual calculation of correlation coefficient

/-- Theorem: If all sample points fall on a straight line, 
    then the sum of squared residuals is 0 and 
    the absolute value of the correlation coefficient is 1 -/
theorem perfect_line_fit (points : List SamplePoint) (l : Line) :
  (∀ p ∈ points, pointOnLine p l) →
  sumSquaredResiduals points l = 0 ∧ |correlationCoefficient points| = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_line_fit_l126_12699


namespace NUMINAMATH_CALUDE_intersection_point_determines_k_l126_12636

/-- Line with slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem intersection_point_determines_k 
  (m n : Line)
  (p : Point)
  (k : ℝ)
  (h1 : m.slope = 4)
  (h2 : m.intercept = 2)
  (h3 : n.slope = k)
  (h4 : n.intercept = 3)
  (h5 : p.x = 1)
  (h6 : p.y = 6)
  (h7 : p.on_line m)
  (h8 : p.on_line n)
  : k = 3 := by
  sorry

#check intersection_point_determines_k

end NUMINAMATH_CALUDE_intersection_point_determines_k_l126_12636


namespace NUMINAMATH_CALUDE_esperanza_salary_l126_12648

/-- Calculates the gross monthly salary given the specified expenses and savings. -/
def gross_monthly_salary (rent food_ratio mortgage_ratio savings tax_ratio : ℝ) : ℝ :=
  let food := food_ratio * rent
  let mortgage := mortgage_ratio * food
  let taxes := tax_ratio * savings
  rent + food + mortgage + savings + taxes

/-- Theorem stating the gross monthly salary under given conditions. -/
theorem esperanza_salary : 
  gross_monthly_salary 600 (3/5) 3 2000 (2/5) = 4840 := by
  sorry

end NUMINAMATH_CALUDE_esperanza_salary_l126_12648


namespace NUMINAMATH_CALUDE_solution_existence_l126_12692

/-- The set of real solutions (x, y) satisfying both equations -/
def SolutionSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 9 = 0 ∧ p.1^2 - 2*p.2 + 6 = 0}

/-- Theorem stating that real solutions exist if and only if y = -5 or y = 3 -/
theorem solution_existence : 
  ∃ (x : ℝ), (x, y) ∈ SolutionSet ↔ y = -5 ∨ y = 3 :=
sorry

end NUMINAMATH_CALUDE_solution_existence_l126_12692


namespace NUMINAMATH_CALUDE_triangle_side_length_l126_12633

theorem triangle_side_length (A B C a b c : ℝ) : 
  A + C = 2 * B → 
  a + c = 8 → 
  a * c = 15 → 
  b = Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l126_12633


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_l126_12627

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def VerticalAngles (α β : Real) : Prop := sorry

theorem vertical_angles_are_equal (α β : Real) :
  VerticalAngles α β → α = β := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_l126_12627


namespace NUMINAMATH_CALUDE_additional_surcharge_l126_12682

/-- Calculates the additional surcharge for a special project given the tax information --/
theorem additional_surcharge (tax_1995 tax_1996 : ℕ) (increase_rate : ℚ) : 
  tax_1995 = 1800 →
  increase_rate = 6 / 100 →
  tax_1996 = 2108 →
  (tax_1996 : ℚ) = (tax_1995 : ℚ) * (1 + increase_rate) + 200 := by
  sorry

end NUMINAMATH_CALUDE_additional_surcharge_l126_12682


namespace NUMINAMATH_CALUDE_correct_calculation_l126_12632

theorem correct_calculation (m n : ℝ) : 4*m + 2*n - (n - m) = 5*m + n := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l126_12632


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l126_12626

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let original_area := L * W
  let new_length := 1.2 * L
  let new_width := 1.2 * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 44 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l126_12626


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l126_12696

theorem smallest_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 5 = 1 ∧
  n % 7 = 2 ∧
  n % 9 = 3 ∧
  n % 11 = 4 ∧
  ∀ m : ℕ, m > 0 ∧ m % 5 = 1 ∧ m % 7 = 2 ∧ m % 9 = 3 ∧ m % 11 = 4 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l126_12696


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l126_12689

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth :
  12 / (1 / 6 : ℚ) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l126_12689


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l126_12610

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 4 * x^2 - 3 * x + 2 < 0) ↔ (∃ x : ℝ, 4 * x^2 - 3 * x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l126_12610


namespace NUMINAMATH_CALUDE_max_value_of_3a_plus_2_l126_12661

theorem max_value_of_3a_plus_2 (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) :
  3 * a + 2 ≤ (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_3a_plus_2_l126_12661


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l126_12606

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → ∀ a b : ℕ, a^2 - b^2 = 187 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 205 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l126_12606


namespace NUMINAMATH_CALUDE_problem_proof_l126_12659

theorem problem_proof (x : ℕ+) (y : ℚ) 
  (h1 : (x : ℚ) = 11 * y + 4)
  (h2 : (2 * x : ℚ) = 24 * y + 3) :
  13 * y - (x : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l126_12659


namespace NUMINAMATH_CALUDE_bamboo_break_height_l126_12604

theorem bamboo_break_height (total_height : ℝ) (fall_distance : ℝ) (break_height : ℝ) : 
  total_height = 9 → 
  fall_distance = 3 → 
  break_height^2 + fall_distance^2 = (total_height - break_height)^2 →
  break_height = 4 := by
sorry

end NUMINAMATH_CALUDE_bamboo_break_height_l126_12604


namespace NUMINAMATH_CALUDE_increasing_sequence_with_properties_l126_12662

theorem increasing_sequence_with_properties :
  ∃ (a : ℕ → ℕ) (C : ℝ), 
    (∀ n, a n < a (n + 1)) ∧ 
    (∀ m : ℕ+, ∃! (i j : ℕ), m = a j - a i) ∧
    (∀ k : ℕ+, (a k : ℝ) ≤ C * (k : ℝ)^3) :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_with_properties_l126_12662


namespace NUMINAMATH_CALUDE_robotic_octopus_dressing_orders_l126_12618

/-- Represents the number of legs on the robotic octopus -/
def num_legs : ℕ := 4

/-- Represents the number of tentacles on the robotic octopus -/
def num_tentacles : ℕ := 2

/-- Represents the number of items per leg (glove and boot) -/
def items_per_leg : ℕ := 2

/-- Represents the number of items per tentacle (bracelet) -/
def items_per_tentacle : ℕ := 1

/-- Calculates the total number of items to be worn -/
def total_items : ℕ := num_legs * items_per_leg + num_tentacles * items_per_tentacle

/-- Theorem stating the number of different dressing orders for the robotic octopus -/
theorem robotic_octopus_dressing_orders : 
  (Nat.factorial num_tentacles) * (2 ^ num_legs) * (Nat.factorial (num_legs * items_per_leg)) = 1286400 :=
sorry

end NUMINAMATH_CALUDE_robotic_octopus_dressing_orders_l126_12618


namespace NUMINAMATH_CALUDE_triangle_area_is_40_5_l126_12691

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A right triangle formed by a line intersecting the x and y axes -/
structure RightTriangle where
  line : Line

/-- Calculate the area of the right triangle -/
def area (triangle : RightTriangle) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem triangle_area_is_40_5 :
  let l : Line := { point1 := (-3, 6), point2 := (-6, 3) }
  let t : RightTriangle := { line := l }
  area t = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_40_5_l126_12691


namespace NUMINAMATH_CALUDE_at_least_one_negative_l126_12697

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  (a < 0) ∨ (b < 0) ∨ (c < 0) ∨ (d < 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l126_12697


namespace NUMINAMATH_CALUDE_original_savings_calculation_l126_12679

theorem original_savings_calculation (savings : ℝ) : 
  (4 / 5 : ℝ) * savings + 100 = savings → savings = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l126_12679


namespace NUMINAMATH_CALUDE_bus_fraction_proof_l126_12695

def total_distance : ℝ := 129.9999999999999
def train_fraction : ℚ := 3/5
def walk_distance : ℝ := 6.5

theorem bus_fraction_proof :
  let bus_distance := total_distance - train_fraction * total_distance - walk_distance
  bus_distance / total_distance = 7/20 := by sorry

end NUMINAMATH_CALUDE_bus_fraction_proof_l126_12695


namespace NUMINAMATH_CALUDE_pen_diary_cost_l126_12658

/-- Given that 6 pens and 5 diaries cost $6.10, and 3 pens and 4 diaries cost $4.60,
    prove that 12 pens and 8 diaries cost $10.16 -/
theorem pen_diary_cost : ∃ (pen_cost diary_cost : ℝ),
  (6 * pen_cost + 5 * diary_cost = 6.10) ∧
  (3 * pen_cost + 4 * diary_cost = 4.60) ∧
  (12 * pen_cost + 8 * diary_cost = 10.16) := by
  sorry


end NUMINAMATH_CALUDE_pen_diary_cost_l126_12658


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l126_12624

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

/-- Theorem: A batsman's average after 12 innings is 58, given the conditions -/
theorem batsman_average_after_12th_innings 
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 80 = b.average + 2)
  : newAverage b 80 = 58 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l126_12624


namespace NUMINAMATH_CALUDE_percentage_problem_l126_12664

theorem percentage_problem (P : ℝ) : 25 = (P / 100) * 25 + 21 → P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l126_12664


namespace NUMINAMATH_CALUDE_quadratic_max_value_l126_12654

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

-- Define the theorem
theorem quadratic_max_value (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ a → f x ≤ 15) ∧
  (∃ x, 1 ≤ x ∧ x ≤ a ∧ f x = 15) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l126_12654


namespace NUMINAMATH_CALUDE_fraction_calculation_l126_12602

theorem fraction_calculation : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l126_12602


namespace NUMINAMATH_CALUDE_book_selection_theorem_l126_12634

/-- The number of ways to select 5 books from 10 books with specific conditions -/
def select_books (n : ℕ) (k : ℕ) (adjacent_pairs : ℕ) (remaining : ℕ) : ℕ :=
  adjacent_pairs * Nat.choose remaining (k - 2)

/-- Theorem stating the number of ways to select 5 books from 10 books 
    where order doesn't matter and two of the selected books must be adjacent -/
theorem book_selection_theorem :
  select_books 10 5 9 8 = 504 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l126_12634


namespace NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_equation_l126_12668

theorem smallest_positive_solution_sqrt_equation :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt (3 * x) - (5 * x - 1)
  ∃! x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_equation_l126_12668


namespace NUMINAMATH_CALUDE_x_plus_y_values_l126_12622

theorem x_plus_y_values (x y : ℝ) 
  (hx : |x| = 3) 
  (hy : |y| = 2) 
  (hxy : |x - y| = y - x) : 
  x + y = -1 ∨ x + y = -5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l126_12622


namespace NUMINAMATH_CALUDE_monomial_sum_exponent_l126_12613

theorem monomial_sum_exponent (m n : ℕ) : 
  (∃ (a : ℚ), ∀ (x y : ℝ), -x^(m-2) * y^3 + 2/3 * x^n * y^(2*m-3*n) = a * x^(m-2) * y^3) → 
  m^(-n : ℤ) = (1/3 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_exponent_l126_12613


namespace NUMINAMATH_CALUDE_reflection_line_equation_l126_12643

/-- The line of reflection for a triangle given its original and reflected coordinates -/
def line_of_reflection (D E F D' E' F' : ℝ × ℝ) : ℝ → Prop :=
  fun x ↦ x = -7

/-- Theorem: The equation of the line of reflection for the given triangle and its image -/
theorem reflection_line_equation :
  let D := (-3, 2)
  let E := (1, 4)
  let F := (-5, -1)
  let D' := (-11, 2)
  let E' := (-9, 4)
  let F' := (-15, -1)
  line_of_reflection D E F D' E' F' = fun x ↦ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l126_12643


namespace NUMINAMATH_CALUDE_multiples_of_three_imply_F_equals_six_l126_12612

def first_number (D E : ℕ) : ℕ := 8000000 + D * 100000 + 70000 + 3000 + E * 10 + 2

def second_number (D E F : ℕ) : ℕ := 4000000 + 100000 + 70000 + D * 1000 + E * 100 + 60 + F

theorem multiples_of_three_imply_F_equals_six (D E : ℕ) 
  (h1 : D < 10) (h2 : E < 10) 
  (h3 : ∃ k : ℕ, first_number D E = 3 * k) 
  (h4 : ∃ m : ℕ, second_number D E 6 = 3 * m) : 
  ∃ F : ℕ, F = 6 ∧ F < 10 ∧ ∃ n : ℕ, second_number D E F = 3 * n :=
sorry

end NUMINAMATH_CALUDE_multiples_of_three_imply_F_equals_six_l126_12612


namespace NUMINAMATH_CALUDE_water_depth_relationship_l126_12676

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ
  horizontalWaterDepth : ℝ

/-- Calculates the water depth when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) : ℝ :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_relationship (tank : WaterTank) 
  (h : tank.height = 20 ∧ tank.baseDiameter = 6 ∧ tank.horizontalWaterDepth = 2) :
  abs (verticalWaterDepth tank - 7.0) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_relationship_l126_12676


namespace NUMINAMATH_CALUDE_upstream_speed_l126_12693

/-- The speed of a man rowing upstream, given his speed in still water and the speed of the stream -/
theorem upstream_speed (downstream_speed still_water_speed stream_speed : ℝ) :
  downstream_speed = still_water_speed + stream_speed →
  still_water_speed > 0 →
  stream_speed > 0 →
  still_water_speed > stream_speed →
  (still_water_speed - stream_speed : ℝ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_upstream_speed_l126_12693


namespace NUMINAMATH_CALUDE_num_expressions_correct_l126_12619

/-- The number of algebraically different expressions obtained by placing parentheses in a₁ / a₂ / ... / aₙ -/
def num_expressions (n : ℕ) : ℕ :=
  if n ≥ 2 then 2^(n-2) else 0

/-- Theorem stating that for n ≥ 2, the number of algebraically different expressions
    obtained by placing parentheses in a₁ / a₂ / ... / aₙ is equal to 2^(n-2) -/
theorem num_expressions_correct (n : ℕ) (h : n ≥ 2) :
  num_expressions n = 2^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_num_expressions_correct_l126_12619


namespace NUMINAMATH_CALUDE_prob_xavier_yvonne_not_zelda_l126_12680

-- Define the difficulty factors and probabilities
variable (a b c : ℝ)
variable (p_xavier : ℝ := (1/3)^a)
variable (p_yvonne : ℝ := (1/2)^b)
variable (p_zelda : ℝ := (5/8)^c)

-- Define the theorem
theorem prob_xavier_yvonne_not_zelda :
  p_xavier * p_yvonne * (1 - p_zelda) = (1/16) * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_prob_xavier_yvonne_not_zelda_l126_12680


namespace NUMINAMATH_CALUDE_unique_odd_divisors_pair_l126_12617

/-- A number has an odd number of divisors if and only if it is a perfect square -/
def has_odd_divisors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The theorem states that 576 is the only positive integer n such that
    both n and n + 100 have an odd number of divisors -/
theorem unique_odd_divisors_pair :
  ∀ n : ℕ, n > 0 ∧ has_odd_divisors n ∧ has_odd_divisors (n + 100) → n = 576 :=
sorry

end NUMINAMATH_CALUDE_unique_odd_divisors_pair_l126_12617


namespace NUMINAMATH_CALUDE_product_equals_zero_l126_12623

theorem product_equals_zero (x y N : ℝ) : 
  ((x + 4) * (y - 4) = N) → 
  (∀ a b : ℝ, a^2 + b^2 ≥ x^2 + y^2) → 
  (x^2 + y^2 = 16) → 
  N = 0 := by sorry

end NUMINAMATH_CALUDE_product_equals_zero_l126_12623


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l126_12683

theorem pure_imaginary_product (x : ℝ) : 
  (∃ y : ℝ, (x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + Complex.I) = Complex.I * y) ↔ 
  x = -3 ∨ x = -1 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l126_12683


namespace NUMINAMATH_CALUDE_sum_of_roots_greater_than_five_l126_12656

theorem sum_of_roots_greater_than_five (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) > 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_greater_than_five_l126_12656


namespace NUMINAMATH_CALUDE_salary_calculation_l126_12628

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := 1375

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.20

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.20

/-- Represents the new savings amount after expense increase in Rupees -/
def new_savings : ℝ := 220

theorem salary_calculation :
  monthly_salary * savings_rate * (1 - expense_increase_rate) = new_savings :=
by sorry

end NUMINAMATH_CALUDE_salary_calculation_l126_12628


namespace NUMINAMATH_CALUDE_lcm_36_132_l126_12663

theorem lcm_36_132 : Nat.lcm 36 132 = 396 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_132_l126_12663


namespace NUMINAMATH_CALUDE_probability_of_passing_is_correct_l126_12608

-- Define the number of shots in the test
def num_shots : ℕ := 3

-- Define the minimum number of successful shots required to pass
def min_successful_shots : ℕ := 2

-- Define the probability of making a single shot
def single_shot_probability : ℝ := 0.6

-- Define the function to calculate the probability of passing the test
def probability_of_passing : ℝ := sorry

-- Theorem stating that the probability of passing is 0.648
theorem probability_of_passing_is_correct : probability_of_passing = 0.648 := by sorry

end NUMINAMATH_CALUDE_probability_of_passing_is_correct_l126_12608


namespace NUMINAMATH_CALUDE_village_population_l126_12650

/-- If 40% of a population is 23040, then the total population is 57600. -/
theorem village_population (population : ℕ) : (40 : ℕ) * population / 100 = 23040 → population = 57600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l126_12650


namespace NUMINAMATH_CALUDE_whale_plankton_theorem_l126_12690

/-- Calculates the total amount of plankton consumed by a whale during a 5-hour feeding frenzy -/
def whale_plankton_consumption (x : ℕ) : ℕ :=
  let hour1 := x
  let hour2 := x + 3
  let hour3 := x + 6
  let hour4 := x + 9
  let hour5 := x + 12
  hour1 + hour2 + hour3 + hour4 + hour5

/-- Theorem stating the total plankton consumption given the problem conditions -/
theorem whale_plankton_theorem : 
  ∀ x : ℕ, (x + 6 = 93) → whale_plankton_consumption x = 465 :=
by
  sorry

#eval whale_plankton_consumption 87

end NUMINAMATH_CALUDE_whale_plankton_theorem_l126_12690


namespace NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l126_12630

/-- Counts the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The number of trailing zeros in 800! is 199 -/
theorem factorial_800_trailing_zeros : trailingZeros 800 = 199 := by
  sorry

end NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l126_12630


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l126_12644

theorem largest_prime_factor_of_3913 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3913 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l126_12644


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l126_12667

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l126_12667


namespace NUMINAMATH_CALUDE_pc_length_l126_12687

/-- A convex quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points of the quadrilateral
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Point P on diagonal AC
  P : ℝ × ℝ
  -- Convexity condition (simplified)
  convex : True
  -- CD perpendicular to AC
  cd_perp_ac : (C.1 - D.1) * (C.1 - A.1) + (C.2 - D.2) * (C.2 - A.2) = 0
  -- AB perpendicular to BD
  ab_perp_bd : (A.1 - B.1) * (B.1 - D.1) + (A.2 - B.2) * (B.2 - D.2) = 0
  -- CD length
  cd_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 72
  -- AB length
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 35
  -- P on AC
  p_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  -- BP perpendicular to AD
  bp_perp_ad : (B.1 - P.1) * (A.1 - D.1) + (B.2 - P.2) * (A.2 - D.2) = 0
  -- AP length
  ap_length : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 15

/-- The main theorem stating that PC = 72.5 in the special quadrilateral -/
theorem pc_length (q : SpecialQuadrilateral) : 
  Real.sqrt ((q.P.1 - q.C.1)^2 + (q.P.2 - q.C.2)^2) = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_pc_length_l126_12687


namespace NUMINAMATH_CALUDE_M_subset_N_l126_12673

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2*a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4*b + 5}

theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l126_12673


namespace NUMINAMATH_CALUDE_croissant_cost_calculation_l126_12635

/-- Calculates the cost of croissants for a committee luncheon --/
theorem croissant_cost_calculation 
  (people : ℕ) 
  (sandwiches_per_person : ℕ) 
  (croissants_per_dozen : ℕ) 
  (cost_per_dozen : ℚ) : 
  people = 24 → 
  sandwiches_per_person = 2 → 
  croissants_per_dozen = 12 → 
  cost_per_dozen = 8 → 
  (people * sandwiches_per_person / croissants_per_dozen : ℚ) * cost_per_dozen = 32 :=
by sorry

#check croissant_cost_calculation

end NUMINAMATH_CALUDE_croissant_cost_calculation_l126_12635


namespace NUMINAMATH_CALUDE_valid_pairs_count_l126_12600

/-- Represents the number of books in each category -/
def num_books_per_category : ℕ := 4

/-- Represents the total number of books -/
def total_books : ℕ := 3 * num_books_per_category

/-- Represents the number of novels -/
def num_novels : ℕ := 2 * num_books_per_category

/-- Calculates the number of ways to choose 2 books such that each pair includes at least one novel -/
def count_valid_pairs : ℕ :=
  let total_choices := num_novels * (total_books - num_books_per_category)
  let overcounted_pairs := num_novels * num_books_per_category
  (total_choices - overcounted_pairs) / 2

theorem valid_pairs_count : count_valid_pairs = 28 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l126_12600


namespace NUMINAMATH_CALUDE_point_C_representation_l126_12665

def point_A : ℝ := -2

def point_B : ℝ := point_A - 2

def distance_BC : ℝ := 5

theorem point_C_representation :
  ∃ (point_C : ℝ), (point_C = point_B - distance_BC ∨ point_C = point_B + distance_BC) ∧
                    (point_C = -9 ∨ point_C = 1) := by
  sorry

end NUMINAMATH_CALUDE_point_C_representation_l126_12665


namespace NUMINAMATH_CALUDE_max_product_of_functions_l126_12614

theorem max_product_of_functions (f h : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.Icc (-5) 3) 
  (hh : ∀ x, h x ∈ Set.Icc (-3) 4) : 
  (⨆ x, f x * h x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_functions_l126_12614


namespace NUMINAMATH_CALUDE_y_axis_symmetry_sum_l126_12645

/-- Given a point M(a, b+3, 2c+1) with y-axis symmetric point M'(-4, -2, 15), prove a+b+c = -9 -/
theorem y_axis_symmetry_sum (a b c : ℝ) : 
  (a = 4) ∧ (b + 3 = -2) ∧ (2 * c + 1 = 15) → a + b + c = -9 := by
  sorry

end NUMINAMATH_CALUDE_y_axis_symmetry_sum_l126_12645


namespace NUMINAMATH_CALUDE_earbuds_tickets_proof_l126_12640

/-- The number of tickets Connie spent on earbuds -/
def tickets_on_earbuds (total_tickets : ℕ) (tickets_on_koala : ℕ) (tickets_on_bracelets : ℕ) : ℕ :=
  total_tickets - tickets_on_koala - tickets_on_bracelets

theorem earbuds_tickets_proof :
  let total_tickets : ℕ := 50
  let tickets_on_koala : ℕ := total_tickets / 2
  let tickets_on_bracelets : ℕ := 15
  tickets_on_earbuds total_tickets tickets_on_koala tickets_on_bracelets = 10 := by
  sorry

#eval tickets_on_earbuds 50 25 15

end NUMINAMATH_CALUDE_earbuds_tickets_proof_l126_12640


namespace NUMINAMATH_CALUDE_knife_value_l126_12616

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) :
  (n * n = 20 * k + 10 + m) →
  (1 ≤ m) →
  (m ≤ 9) →
  (∃ b : ℕ, 10 - b = m + b) →
  (∃ b : ℕ, b = 2) :=
by sorry

end NUMINAMATH_CALUDE_knife_value_l126_12616


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l126_12684

theorem smallest_sum_B_plus_b :
  ∀ (B b : ℕ),
  0 ≤ B ∧ B < 5 →
  b > 6 →
  31 * B = 4 * b + 4 →
  ∀ (B' b' : ℕ),
  0 ≤ B' ∧ B' < 5 →
  b' > 6 →
  31 * B' = 4 * b' + 4 →
  B + b ≤ B' + b' :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l126_12684


namespace NUMINAMATH_CALUDE_hyperbola_equation_l126_12652

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    its eccentricity is 2, and the distance from the origin to line AB
    (where A(a, 0) and B(0, -b)) is 3/2, prove that the equation of the
    hyperbola is x²/3 - y²/9 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c / a = 2) →
  (∃ d : ℝ, d = 3/2 ∧ d = |(a * b)| / Real.sqrt (a^2 + b^2)) →
  (∀ x y : ℝ, x^2 / 3 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l126_12652


namespace NUMINAMATH_CALUDE_negation_of_exists_cube_positive_l126_12653

theorem negation_of_exists_cube_positive :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_cube_positive_l126_12653


namespace NUMINAMATH_CALUDE_nested_custom_op_equals_two_l126_12670

/-- Custom operation [a,b,c] defined as (a+b)/c where c ≠ 0 -/
def custom_op (a b c : ℚ) : ℚ := (a + b) / c

/-- Theorem stating that [[72,18,90],[4,2,6],[12,6,18]] = 2 -/
theorem nested_custom_op_equals_two :
  custom_op (custom_op 72 18 90) (custom_op 4 2 6) (custom_op 12 6 18) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_custom_op_equals_two_l126_12670


namespace NUMINAMATH_CALUDE_roots_sum_l126_12674

theorem roots_sum (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, (m₁ * a^2 - (3 * m₁ - 2) * a + 7 = 0) ∧ 
              (m₁ * b^2 - (3 * m₁ - 2) * b + 7 = 0) ∧ 
              (a / b + b / a = 3 / 2)) ∧
  (∃ a b : ℝ, (m₂ * a^2 - (3 * m₂ - 2) * a + 7 = 0) ∧ 
              (m₂ * b^2 - (3 * m₂ - 2) * b + 7 = 0) ∧ 
              (a / b + b / a = 3 / 2)) →
  m₁ + m₂ = 73 / 18 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_l126_12674


namespace NUMINAMATH_CALUDE_min_like_both_l126_12601

theorem min_like_both (total : ℕ) (like_mozart : ℕ) (like_beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : like_mozart = 102)
  (h_beethoven : like_beethoven = 85)
  : ∃ (like_both : ℕ), like_both ≥ 67 ∧ 
    (∀ (x : ℕ), x < like_both → 
      ∃ (only_mozart only_beethoven : ℕ),
        x + only_mozart + only_beethoven ≤ total ∧
        x + only_mozart ≤ like_mozart ∧
        x + only_beethoven ≤ like_beethoven) :=
by sorry

end NUMINAMATH_CALUDE_min_like_both_l126_12601


namespace NUMINAMATH_CALUDE_parabola_focus_l126_12677

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * y^2

-- Define the focus of a parabola
def focus (p q : ℝ) : Prop := ∀ x y : ℝ, parabola_equation x y → (x + 1)^2 + y^2 = 1

-- Theorem statement
theorem parabola_focus : focus (-1) 0 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l126_12677


namespace NUMINAMATH_CALUDE_log_inequality_l126_12646

/-- The function f as defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

/-- The theorem statement -/
theorem log_inequality (m : ℝ) 
  (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f m x) 
  (h2 : m > 0) : 
  Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l126_12646


namespace NUMINAMATH_CALUDE_strawberry_charity_donation_is_correct_l126_12688

/-- The amount of money donated to charity from strawberry jam sales -/
def strawberry_charity_donation : ℚ :=
let betty_strawberries : ℕ := 25
let matthew_strawberries : ℕ := betty_strawberries + 30
let natalie_strawberries : ℕ := matthew_strawberries / 3
let emily_strawberries : ℕ := natalie_strawberries / 2
let ethan_strawberries : ℕ := natalie_strawberries * 2
let total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries + emily_strawberries + ethan_strawberries
let strawberries_per_jar : ℕ := 12
let jars_made : ℕ := total_strawberries / strawberries_per_jar
let price_per_jar : ℚ := 6
let total_revenue : ℚ := (jars_made : ℚ) * price_per_jar
let donation_percentage : ℚ := 40 / 100
donation_percentage * total_revenue

theorem strawberry_charity_donation_is_correct :
  strawberry_charity_donation = 26.4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_charity_donation_is_correct_l126_12688


namespace NUMINAMATH_CALUDE_fourth_root_of_2560000_l126_12671

theorem fourth_root_of_2560000 : (2560000 : ℝ) ^ (1/4 : ℝ) = 40 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_2560000_l126_12671


namespace NUMINAMATH_CALUDE_bacteria_growth_rate_l126_12625

/-- Represents the growth rate of bacteria in a dish -/
def growth_rate (r : ℝ) : Prop :=
  ∀ (t : ℕ), t ≥ 0 → (1 / 16 : ℝ) * r^30 = r^26 ∧ r^30 = r^30

theorem bacteria_growth_rate :
  ∃ (r : ℝ), r > 0 ∧ growth_rate r ∧ r = 2 :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_rate_l126_12625


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l126_12672

theorem polynomial_divisibility (n : ℤ) : 
  (∀ x : ℝ, (3 * x^2 + 5 * x + n) = (x - 2) * (3 * x + 11)) ↔ n = -22 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l126_12672


namespace NUMINAMATH_CALUDE_x_over_y_value_l126_12681

theorem x_over_y_value (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) 
  (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_value_l126_12681


namespace NUMINAMATH_CALUDE_prob_sum_not_greater_than_4_prob_first_less_than_second_plus_2_l126_12660

-- Define the sample space for a single die throw
def Die : Type := Fin 6

-- Define the sample space for two dice throws
def TwoDice : Type := Die × Die

-- Define the probability measure on TwoDice
noncomputable def P : Set TwoDice → ℝ := sorry

-- Define the event where the sum of dice is not greater than 4
def SumNotGreaterThan4 : Set TwoDice :=
  {x | x.1.val + x.2.val ≤ 4}

-- Define the event where the first die is less than the second die plus 2
def FirstLessThanSecondPlus2 : Set TwoDice :=
  {x | x.1.val < x.2.val + 2}

-- Theorem 1: Probability that the sum of dice is not greater than 4 is 1/6
theorem prob_sum_not_greater_than_4 :
  P SumNotGreaterThan4 = 1/6 := by sorry

-- Theorem 2: Probability that the first die is less than the second die plus 2 is 13/18
theorem prob_first_less_than_second_plus_2 :
  P FirstLessThanSecondPlus2 = 13/18 := by sorry

end NUMINAMATH_CALUDE_prob_sum_not_greater_than_4_prob_first_less_than_second_plus_2_l126_12660


namespace NUMINAMATH_CALUDE_expand_and_simplify_l126_12631

theorem expand_and_simplify (x : ℝ) : (2*x - 1)^2 - x*(4*x - 1) = -3*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l126_12631


namespace NUMINAMATH_CALUDE_squared_sum_bound_l126_12620

theorem squared_sum_bound (a b : ℝ) (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 3*(a+b)*x₁ + 4*a*b = 0) →
  (3 * x₂^2 + 3*(a+b)*x₂ + 4*a*b = 0) →
  (x₁ * (x₁ + 1) + x₂ * (x₂ + 1) = (x₁ + 1) * (x₂ + 1)) →
  (a + b)^2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_squared_sum_bound_l126_12620


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l126_12607

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l126_12607


namespace NUMINAMATH_CALUDE_arithmetic_equality_l126_12603

theorem arithmetic_equality : 4 * 7 + 5 * 12 + 12 * 4 + 4 * 9 = 172 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l126_12603


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l126_12655

/-- The volume of a rectangular solid with side lengths 1 m, 20 cm, and 50 cm is 100000 cm³ -/
theorem rectangular_solid_volume : 
  let length_m : ℝ := 1
  let width_cm : ℝ := 20
  let height_cm : ℝ := 50
  let m_to_cm : ℝ := 100
  (length_m * m_to_cm * width_cm * height_cm) = 100000 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l126_12655


namespace NUMINAMATH_CALUDE_range_m_theorem_l126_12621

/-- The range of m for which p is a necessary condition for q -/
def range_m_p_necessary_for_q : Set ℝ :=
  {m : ℝ | ∀ x, (1 - m^2 ≤ x ∧ x ≤ 1 + m^2) → (-2 ≤ x ∧ x ≤ 10)}

/-- The range of m for which ¬p is a necessary but not sufficient condition for ¬q -/
def range_m_not_p_necessary_not_sufficient_for_not_q : Set ℝ :=
  {m : ℝ | (∀ x, (x < 1 - m^2 ∨ x > 1 + m^2) → (x < -2 ∨ x > 10)) ∧
           (∃ x, (x < -2 ∨ x > 10) ∧ (1 - m^2 ≤ x ∧ x ≤ 1 + m^2))}

theorem range_m_theorem :
  range_m_p_necessary_for_q = {m : ℝ | -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3} ∧
  range_m_not_p_necessary_not_sufficient_for_not_q = {m : ℝ | m ≤ -3 ∨ m ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_range_m_theorem_l126_12621


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l126_12609

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : S seq 5 = seq.a 8 + 5)
  (h2 : S seq 6 = seq.a 7 + seq.a 9 - 5) :
  seq.d = -55 / 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l126_12609


namespace NUMINAMATH_CALUDE_bill_bouquets_to_buy_l126_12675

/-- Represents the rose business scenario for Bill --/
structure RoseBusiness where
  buy_roses_per_bouquet : ℕ
  buy_price_per_bouquet : ℕ
  sell_roses_per_bouquet : ℕ
  sell_price_per_bouquet : ℕ

/-- Calculates the number of bouquets Bill needs to buy to earn a specific profit --/
def bouquets_to_buy (rb : RoseBusiness) (target_profit : ℕ) : ℕ :=
  let buy_bouquets := rb.sell_roses_per_bouquet
  let sell_bouquets := rb.buy_roses_per_bouquet
  let profit_per_operation := sell_bouquets * rb.sell_price_per_bouquet - buy_bouquets * rb.buy_price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * buy_bouquets

/-- Theorem stating that Bill needs to buy 125 bouquets to earn $1000 --/
theorem bill_bouquets_to_buy :
  let rb : RoseBusiness := {
    buy_roses_per_bouquet := 7,
    buy_price_per_bouquet := 20,
    sell_roses_per_bouquet := 5,
    sell_price_per_bouquet := 20
  }
  bouquets_to_buy rb 1000 = 125 := by sorry

end NUMINAMATH_CALUDE_bill_bouquets_to_buy_l126_12675


namespace NUMINAMATH_CALUDE_min_tests_required_l126_12698

/-- Represents a set of batteries -/
def Battery := Fin 8

/-- Represents a pair of batteries -/
def BatteryPair := Battery × Battery

/-- Represents the state of a battery -/
inductive BatteryState
| Good
| Bad

/-- The total number of batteries -/
def totalBatteries : Nat := 8

/-- The number of good batteries -/
def goodBatteries : Nat := 4

/-- The number of bad batteries -/
def badBatteries : Nat := 4

/-- A function that determines if a pair of batteries works -/
def works (pair : BatteryPair) (state : Battery → BatteryState) : Prop :=
  state pair.1 = BatteryState.Good ∧ state pair.2 = BatteryState.Good

/-- The main theorem stating the minimum number of tests required -/
theorem min_tests_required :
  ∀ (state : Battery → BatteryState),
  (∃ (good : Finset Battery), good.card = goodBatteries ∧ ∀ b ∈ good, state b = BatteryState.Good) →
  ∃ (tests : Finset BatteryPair),
    tests.card = 7 ∧
    (∀ (pair : BatteryPair), works pair state → pair ∈ tests) ∧
    ∀ (tests' : Finset BatteryPair),
      tests'.card < 7 →
      ∃ (pair : BatteryPair), works pair state ∧ pair ∉ tests' :=
sorry

end NUMINAMATH_CALUDE_min_tests_required_l126_12698


namespace NUMINAMATH_CALUDE_fifth_element_row_21_l126_12605

/-- Pascal's triangle element -/
def pascal_triangle_element (n : ℕ) (k : ℕ) : ℕ := Nat.choose n (k - 1)

/-- The fifth element in Row 21 of Pascal's triangle is 1995 -/
theorem fifth_element_row_21 : pascal_triangle_element 21 5 = 1995 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_21_l126_12605


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l126_12615

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l126_12615


namespace NUMINAMATH_CALUDE_sample_size_calculation_l126_12639

/-- Represents the sample size for each school level -/
structure SampleSize where
  elementary : ℕ
  middle : ℕ
  high : ℕ

/-- Calculates the total sample size -/
def totalSampleSize (s : SampleSize) : ℕ :=
  s.elementary + s.middle + s.high

/-- The ratio of elementary:middle:high school students -/
def schoolRatio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5

theorem sample_size_calculation (s : SampleSize) 
  (h_ratio : s.elementary * schoolRatio 1 = s.middle * schoolRatio 0 ∧ 
             s.middle * schoolRatio 2 = s.high * schoolRatio 1)
  (h_middle : s.middle = 150) : 
  totalSampleSize s = 500 := by
sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l126_12639


namespace NUMINAMATH_CALUDE_problem_statement_l126_12669

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) :
  b < 0 ∧ |b| > |a| := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l126_12669


namespace NUMINAMATH_CALUDE_julie_newspaper_count_l126_12641

theorem julie_newspaper_count :
  let boxes : ℕ := 2
  let packages_per_box : ℕ := 5
  let sheets_per_package : ℕ := 250
  let sheets_per_newspaper : ℕ := 25
  let total_sheets : ℕ := boxes * packages_per_box * sheets_per_package
  let newspapers : ℕ := total_sheets / sheets_per_newspaper
  newspapers = 100 := by sorry

end NUMINAMATH_CALUDE_julie_newspaper_count_l126_12641


namespace NUMINAMATH_CALUDE_inequality_proof_l126_12649

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l126_12649


namespace NUMINAMATH_CALUDE_fraction_to_decimal_equiv_l126_12651

theorem fraction_to_decimal_equiv : (5 : ℚ) / 8 = 0.625 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_equiv_l126_12651
