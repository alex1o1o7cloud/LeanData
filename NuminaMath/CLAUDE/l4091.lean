import Mathlib

namespace point_D_coordinates_l4091_409175

def P : ℝ × ℝ := (2, -2)
def Q : ℝ × ℝ := (6, 4)

def is_on_segment (D P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • P + t • Q

def twice_distance (D P Q : ℝ × ℝ) : Prop :=
  ‖D - P‖ = 2 * ‖D - Q‖

theorem point_D_coordinates :
  ∃ D : ℝ × ℝ, is_on_segment D P Q ∧ twice_distance D P Q ∧ D = (3, -0.5) := by
sorry

end point_D_coordinates_l4091_409175


namespace sine_graph_shift_l4091_409136

theorem sine_graph_shift (x : ℝ) :
  (3 * Real.sin (2 * (x + π / 8))) = (3 * Real.sin (2 * x + π / 4)) :=
by sorry

end sine_graph_shift_l4091_409136


namespace total_groups_created_l4091_409186

def group_size : ℕ := 6
def eggs : ℕ := 18
def bananas : ℕ := 72
def marbles : ℕ := 66

theorem total_groups_created : 
  (eggs / group_size + bananas / group_size + marbles / group_size) = 26 := by
  sorry

end total_groups_created_l4091_409186


namespace set_operations_l4091_409111

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2)}
def N : Set ℝ := {x | x < 1 ∨ x > 3}

-- State the theorem
theorem set_operations :
  (M ∪ N = {x | x < 1 ∨ x ≥ 2}) ∧
  (M ∩ (Nᶜ) = {x | 2 ≤ x ∧ x ≤ 3}) := by
  sorry

end set_operations_l4091_409111


namespace unique_solution_for_equation_l4091_409159

theorem unique_solution_for_equation : ∃! (x y : ℤ), (x + 2)^4 - x^4 = y^3 :=
by
  -- The proof would go here
  sorry

end unique_solution_for_equation_l4091_409159


namespace range_of_m_l4091_409142

-- Define the necessary condition p
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)

-- Define the condition q
def q (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- Theorem statement
theorem range_of_m :
  {m : ℝ | necessary_but_not_sufficient m} = {m | m < -4 ∨ m > 4} :=
sorry

end range_of_m_l4091_409142


namespace no_prime_square_product_l4091_409117

theorem no_prime_square_product (p q r : Nat) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  ¬∃ n : Nat, (p^2 + p) * (q^2 + q) * (r^2 + r) = n^2 := by
  sorry

end no_prime_square_product_l4091_409117


namespace red_balls_count_l4091_409144

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The probability of drawing a red ball -/
def red_probability : ℝ := 0.85

/-- The number of red balls in the bag -/
def red_balls : ℕ := 17

theorem red_balls_count :
  (red_balls : ℝ) / (red_balls + black_balls) = red_probability :=
by sorry

end red_balls_count_l4091_409144


namespace first_plane_speed_calculation_l4091_409100

/-- The speed of the first plane in kilometers per hour -/
def first_plane_speed : ℝ := 110

/-- The speed of the second plane in kilometers per hour -/
def second_plane_speed : ℝ := 90

/-- The time taken for the planes to be 800 km apart in hours -/
def time : ℝ := 4.84848484848

/-- The distance between the planes after the given time in kilometers -/
def distance : ℝ := 800

theorem first_plane_speed_calculation :
  (first_plane_speed + second_plane_speed) * time = distance := by
  sorry

end first_plane_speed_calculation_l4091_409100


namespace cone_base_radius_l4091_409161

/-- Given a right cone with slant height 27 cm and lateral surface forming
    a circular sector of 220° when unrolled, the radius of the base is 16.5 cm. -/
theorem cone_base_radius (s : ℝ) (θ : ℝ) (h1 : s = 27) (h2 : θ = 220 * π / 180) :
  let r := s * θ / (2 * π)
  r = 16.5 := by sorry

end cone_base_radius_l4091_409161


namespace expression_value_l4091_409115

theorem expression_value : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end expression_value_l4091_409115


namespace binomial_coefficient_20_19_l4091_409119

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l4091_409119


namespace quadratic_inequality_solution_l4091_409174

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 5*x + 6 > 0 ∧ x ≠ 3) ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3 := by sorry

end quadratic_inequality_solution_l4091_409174


namespace new_student_weight_l4091_409151

/-- Given a group of 15 students where replacing a 150 kg student with a new student
    decreases the average weight by 8 kg, the weight of the new student is 30 kg. -/
theorem new_student_weight (total_weight : ℝ) (new_weight : ℝ) : 
  (15 : ℝ) * (total_weight / 15 - (total_weight - 150 + new_weight) / 15) = 8 →
  new_weight = 30 := by
sorry

end new_student_weight_l4091_409151


namespace ellipse_and_line_properties_l4091_409122

/-- Represents an ellipse with given properties -/
structure Ellipse where
  e : ℝ  -- eccentricity
  ab_length : ℝ  -- length of AB

/-- Represents a line that intersects the ellipse -/
structure IntersectingLine where
  k : ℝ  -- slope of the line y = kx + 2

/-- Main theorem about the ellipse and intersecting line -/
theorem ellipse_and_line_properties
  (ell : Ellipse)
  (line : IntersectingLine)
  (h_e : ell.e = Real.sqrt 6 / 3)
  (h_ab : ell.ab_length = 2 * Real.sqrt 3 / 3) :
  (∃ (a b : ℝ), a^2 / 3 + b^2 = 1) ∧  -- Ellipse equation
  (∃ (x y : ℝ), x^2 / 3 + y^2 = 1 ∧ y = line.k * x + 2) ∧  -- Line intersects ellipse
  (∃ (c d : ℝ × ℝ),
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (-1 - c.1)^2 + (0 - c.2)^2 ∧  -- Circle condition
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (-1 - d.1)^2 + (0 - d.2)^2 ∧
    c.2 = line.k * c.1 + 2 ∧
    d.2 = line.k * d.1 + 2) →
  line.k = 7 / 6 := by
sorry

end ellipse_and_line_properties_l4091_409122


namespace expand_product_l4091_409132

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l4091_409132


namespace binary_1010_is_10_l4091_409153

/-- Converts a binary number represented as a list of bits (0s and 1s) to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

theorem binary_1010_is_10 : binary_to_decimal [0, 1, 0, 1] = 10 := by
  sorry

end binary_1010_is_10_l4091_409153


namespace speed_ratio_l4091_409134

def equidistant_points (vA vB : ℝ) : Prop :=
  ∃ (t : ℝ), t * vA = |(-800 + t * vB)|

theorem speed_ratio : ∃ (vA vB : ℝ),
  vA > 0 ∧ vB > 0 ∧
  equidistant_points vA vB ∧
  equidistant_points (3 * vA) (3 * vB) ∧
  vA / vB = 3 / 4 :=
sorry

end speed_ratio_l4091_409134


namespace reduce_to_single_letter_l4091_409157

/-- Represents a circular arrangement of letters A and B -/
def CircularArrangement := List Bool

/-- Represents the operations that can be performed on the arrangement -/
inductive Operation
  | replaceABA
  | replaceBAB

/-- Applies an operation to a circular arrangement -/
def applyOperation (arr : CircularArrangement) (op : Operation) : CircularArrangement :=
  sorry

/-- Checks if the arrangement consists of only one type of letter -/
def isSingleLetter (arr : CircularArrangement) : Bool :=
  sorry

/-- Theorem stating that any initial arrangement of 41 letters can be reduced to a single letter -/
theorem reduce_to_single_letter (initial : CircularArrangement) :
  initial.length = 41 → ∃ (final : CircularArrangement), isSingleLetter final ∧ 
  ∃ (ops : List Operation), final = ops.foldl applyOperation initial :=
  sorry

end reduce_to_single_letter_l4091_409157


namespace middle_number_proof_l4091_409194

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b = 12) (h4 : a + c = 17) (h5 : b + c = 19) : b = 7 := by
  sorry

end middle_number_proof_l4091_409194


namespace triple_composition_even_l4091_409183

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (fun x ↦ g (g (g x))) := by
  sorry

end triple_composition_even_l4091_409183


namespace arithmetic_sequence_ratio_l4091_409160

/-- For an arithmetic sequence with common difference d ≠ 0, 
    if a_3 is the geometric mean of a_2 and a_6, then a_6 / a_3 = 2 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 3 ^ 2 = a 2 * a 6 →
  a 6 / a 3 = 2 := by
  sorry


end arithmetic_sequence_ratio_l4091_409160


namespace tank_filling_l4091_409176

theorem tank_filling (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2 / 5 →
  ∃ new_buckets : ℕ, 
    new_buckets = ⌈(original_buckets : ℚ) / capacity_ratio⌉ ∧
    new_buckets = 63 :=
by sorry

end tank_filling_l4091_409176


namespace ratio_x_to_y_l4091_409188

theorem ratio_x_to_y (x y : ℚ) (h : (14*x - 5*y) / (17*x - 3*y) = 2/7) : x/y = 29/64 := by
  sorry

end ratio_x_to_y_l4091_409188


namespace largest_valid_number_l4091_409141

def is_valid (n : ℕ) : Prop :=
  n % 10 ≠ 0 ∧
  ∀ (a b : ℕ), a < 10 → b < 10 →
    ∃ (x y z : ℕ), n = x * 100 + a * 10 + b + y * 10 + z ∧
    n % (x * 10 + y + z) = 0

theorem largest_valid_number : 
  is_valid 9999 ∧ 
  ∀ m : ℕ, m > 9999 → ¬(is_valid m) :=
sorry

end largest_valid_number_l4091_409141


namespace slope_product_l4091_409178

/-- Given two lines with slopes m and n, where one line makes three times
    the angle with the horizontal as the other and has 3 times the slope,
    prove that mn = 9/4 -/
theorem slope_product (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m = 3 * n)
  (h4 : Real.arctan m = 3 * Real.arctan n) : m * n = 9 / 4 := by
  sorry

end slope_product_l4091_409178


namespace integer_root_values_l4091_409113

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 7 = 0) ↔ a ∈ ({-71, -27, -11, 9} : Set ℤ) := by
  sorry

end integer_root_values_l4091_409113


namespace grid_coloring_inequality_l4091_409152

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the grid -/
def Grid (n : ℕ) := Fin n → Fin n → Cell

/-- Counts the number of black cells adjacent to a vertex -/
def countAdjacentBlack (g : Grid n) (i j : Fin n) : ℕ := sorry

/-- Determines if a vertex is red -/
def isRed (g : Grid n) (i j : Fin n) : Bool :=
  Odd (countAdjacentBlack g i j)

/-- Counts the number of red vertices in the grid -/
def countRedVertices (g : Grid n) : ℕ := sorry

/-- Represents an operation to change colors in a rectangle -/
structure Operation (n : ℕ) where
  topLeft : Fin n × Fin n
  bottomRight : Fin n × Fin n

/-- Applies an operation to the grid -/
def applyOperation (g : Grid n) (op : Operation n) : Grid n := sorry

/-- Checks if the grid is entirely white -/
def isAllWhite (g : Grid n) : Bool := sorry

/-- The minimum number of operations to make the grid white -/
noncomputable def minOperations (g : Grid n) : ℕ := sorry

theorem grid_coloring_inequality (n : ℕ) (g : Grid n) :
  let Y := countRedVertices g
  let X := minOperations g
  Y / 4 ≤ X ∧ X ≤ Y / 2 := by sorry

end grid_coloring_inequality_l4091_409152


namespace number_equation_solution_l4091_409108

theorem number_equation_solution :
  ∃ x : ℝ, 0.5 * x = 0.1667 * x + 10 ∧ x = 30 := by sorry

end number_equation_solution_l4091_409108


namespace chord_length_concentric_circles_l4091_409154

theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  R^2 - r^2 = 20 →
  ∃ c : ℝ, c^2 / 4 + r^2 = R^2 ∧ c = 4 * Real.sqrt 5 := by
  sorry

end chord_length_concentric_circles_l4091_409154


namespace inscribed_triangle_area_l4091_409195

/-- An ellipse with semi-major axis 3 and semi-minor axis 2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 9) = 1}

/-- An equilateral triangle inscribed in the ellipse -/
structure InscribedTriangle where
  vertices : Fin 3 → ℝ × ℝ
  on_ellipse : ∀ i, vertices i ∈ Ellipse
  is_equilateral : ∀ i j, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)
  centroid_origin : (vertices 0 + vertices 1 + vertices 2) / 3 = (0, 0)

/-- The square of the area of the inscribed equilateral triangle -/
def square_area (t : InscribedTriangle) : ℝ := sorry

/-- The main theorem: The square of the area of the inscribed equilateral triangle is 507/16 -/
theorem inscribed_triangle_area (t : InscribedTriangle) : square_area t = 507/16 := by
  sorry

end inscribed_triangle_area_l4091_409195


namespace machine_comparison_l4091_409124

def machine_A : List ℕ := [0, 2, 1, 0, 3, 0, 2, 1, 2, 4]
def machine_B : List ℕ := [2, 1, 1, 2, 1, 0, 2, 1, 3, 2]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (λ x => ((x : ℚ) - avg) ^ 2)).sum / l.length

theorem machine_comparison :
  average machine_A = average machine_B ∧
  variance machine_B < variance machine_A :=
sorry

end machine_comparison_l4091_409124


namespace p_days_correct_l4091_409131

/-- The number of days it takes for q to do the work alone -/
def q_days : ℝ := 10

/-- The fraction of work left after p and q work together for 2 days -/
def work_left : ℝ := 0.7

/-- The number of days it takes for p to do the work alone -/
def p_days : ℝ := 20

/-- Theorem stating that p_days is correct given the conditions -/
theorem p_days_correct : 
  2 * (1 / p_days + 1 / q_days) = 1 - work_left := by
  sorry

end p_days_correct_l4091_409131


namespace inequality_equivalence_l4091_409168

theorem inequality_equivalence (p : ℝ) : 
  (∀ q : ℝ, q > 0 → p + q ≠ 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p ≤ 7.275) := by
sorry

end inequality_equivalence_l4091_409168


namespace bill_amount_calculation_l4091_409189

/-- The amount of a bill given its true discount and banker's discount -/
def bill_amount (true_discount : ℚ) (bankers_discount : ℚ) : ℚ :=
  true_discount + true_discount

/-- Theorem stating that given a true discount of 360 and a banker's discount of 424.8, 
    the amount of the bill is 720 -/
theorem bill_amount_calculation :
  bill_amount 360 424.8 = 720 := by
  sorry

end bill_amount_calculation_l4091_409189


namespace problem_1_problem_2_l4091_409146

theorem problem_1 : (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (Real.pi / 3) - |1 - Real.sqrt 3| = 3 := by
  sorry

theorem problem_2 : ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (2 / (x + 1) + 1 = x / (x - 1) ↔ x = 3) := by
  sorry

end problem_1_problem_2_l4091_409146


namespace least_five_digit_square_cube_l4091_409105

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) →
    (∃ x : ℕ, m = x^2) →
    (∃ y : ℕ, m = y^3) →
    n ≤ m) ∧                  -- least such number
  n = 15625 := by
sorry

end least_five_digit_square_cube_l4091_409105


namespace polynomial_expansion_theorem_l4091_409182

theorem polynomial_expansion_theorem (a k n : ℤ) : 
  (∀ x : ℚ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) → 
  a - n + k = 7 := by
sorry

end polynomial_expansion_theorem_l4091_409182


namespace flood_damage_in_pounds_l4091_409162

def flood_damage_rupees : ℝ := 45000000
def exchange_rate : ℝ := 75

theorem flood_damage_in_pounds : 
  flood_damage_rupees / exchange_rate = 600000 := by sorry

end flood_damage_in_pounds_l4091_409162


namespace quadratic_root_implies_m_value_l4091_409192

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (2 * (-1)^2 - 3 * m * (-1) + 1 = 0) → m = -1 := by
  sorry

end quadratic_root_implies_m_value_l4091_409192


namespace simple_interest_problem_l4091_409109

/-- Simple interest calculation -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4016.25 → 
  rate = 0.14 → 
  time = 5 → 
  principal = interest / (rate * time) → 
  principal = 5737.5 := by
  sorry

end simple_interest_problem_l4091_409109


namespace card_shop_problem_l4091_409184

/-- The total cost of cards bought from two boxes -/
def total_cost (cost1 cost2 : ℚ) (cards1 cards2 : ℕ) : ℚ :=
  cost1 * cards1 + cost2 * cards2

/-- Theorem: The total cost of 6 cards from each box is $18.00 -/
theorem card_shop_problem :
  total_cost (25/20) (35/20) 6 6 = 18 := by
  sorry

end card_shop_problem_l4091_409184


namespace museum_tickets_l4091_409199

/-- Calculates the maximum number of tickets that can be purchased given a regular price, 
    discount price, discount threshold, and budget. -/
def maxTickets (regularPrice discountPrice discountThreshold budget : ℕ) : ℕ :=
  let fullPriceTickets := min discountThreshold (budget / regularPrice)
  let remainingBudget := budget - fullPriceTickets * regularPrice
  let discountTickets := remainingBudget / discountPrice
  fullPriceTickets + discountTickets

/-- Theorem stating that given the specific conditions of the problem, 
    the maximum number of tickets that can be purchased is 15. -/
theorem museum_tickets : maxTickets 11 8 10 150 = 15 := by
  sorry

end museum_tickets_l4091_409199


namespace complex_fraction_calculation_l4091_409137

theorem complex_fraction_calculation :
  |-(7/2)| * (12/7) / (4/3) / (-3)^2 = 1/2 := by sorry

end complex_fraction_calculation_l4091_409137


namespace polynomial_factor_implies_coefficients_l4091_409143

/-- The polynomial with coefficients p and q -/
def polynomial (p q : ℚ) (x : ℚ) : ℚ :=
  p * x^4 + q * x^3 + 20 * x^2 - 10 * x + 15

/-- The factor of the polynomial -/
def factor (x : ℚ) : ℚ :=
  5 * x^2 - 3 * x + 3

theorem polynomial_factor_implies_coefficients (p q : ℚ) :
  (∃ (a b : ℚ), ∀ x, polynomial p q x = factor x * (a * x^2 + b * x + 5)) →
  p = 0 ∧ q = 25/3 := by
sorry

end polynomial_factor_implies_coefficients_l4091_409143


namespace magnitude_of_vector_sum_l4091_409133

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 1)

theorem magnitude_of_vector_sum : 
  ‖(2 • a.1 + b.1, 2 • a.2 + b.2)‖ = 5 * Real.sqrt 2 :=
by sorry

end magnitude_of_vector_sum_l4091_409133


namespace expenditure_representation_l4091_409129

def represent_income (amount : ℝ) : ℝ := amount

theorem expenditure_representation (amount : ℝ) :
  (represent_income amount = amount) →
  (∃ (f : ℝ → ℝ), f amount = -amount) :=
by sorry

end expenditure_representation_l4091_409129


namespace sum_of_digits_of_large_number_l4091_409190

/-- The sum of the digits of (10^40 - 46) is 369. -/
theorem sum_of_digits_of_large_number : 
  (let k := 10^40 - 46
   Finset.sum (Finset.range 41) (λ i => (k / 10^i) % 10)) = 369 := by
  sorry

end sum_of_digits_of_large_number_l4091_409190


namespace square_rotation_lateral_area_l4091_409181

/-- The lateral surface area of a cylinder formed by rotating a square around one of its sides -/
theorem square_rotation_lateral_area (side_length : ℝ) (h : side_length = 2) :
  2 * side_length * Real.pi = 8 * Real.pi :=
by sorry

end square_rotation_lateral_area_l4091_409181


namespace distance_comparisons_l4091_409104

-- Define the driving conditions
def joseph_speed1 : ℝ := 48
def joseph_time1 : ℝ := 2.5
def joseph_speed2 : ℝ := 60
def joseph_time2 : ℝ := 1.5

def kyle_speed1 : ℝ := 70
def kyle_time1 : ℝ := 2
def kyle_speed2 : ℝ := 63
def kyle_time2 : ℝ := 2.5

def emily_speed : ℝ := 65
def emily_time : ℝ := 3

-- Define the distances driven
def joseph_distance : ℝ := joseph_speed1 * joseph_time1 + joseph_speed2 * joseph_time2
def kyle_distance : ℝ := kyle_speed1 * kyle_time1 + kyle_speed2 * kyle_time2
def emily_distance : ℝ := emily_speed * emily_time

-- Theorem to prove the distance comparisons
theorem distance_comparisons :
  (joseph_distance = 210) ∧
  (kyle_distance = 297.5) ∧
  (emily_distance = 195) ∧
  (joseph_distance - kyle_distance = -87.5) ∧
  (emily_distance - joseph_distance = -15) ∧
  (emily_distance - kyle_distance = -102.5) :=
by sorry

end distance_comparisons_l4091_409104


namespace percentage_to_pass_l4091_409102

/-- Given a test with maximum marks and a student's performance, 
    calculate the percentage needed to pass the test. -/
theorem percentage_to_pass (max_marks student_marks shortfall : ℕ) :
  max_marks = 400 →
  student_marks = 80 →
  shortfall = 40 →
  (((student_marks + shortfall) : ℚ) / max_marks) * 100 = 30 := by
  sorry

end percentage_to_pass_l4091_409102


namespace percent_above_sixty_percent_l4091_409185

theorem percent_above_sixty_percent (P Q : ℝ) (h : P > Q) :
  (P - 0.6 * Q) / Q * 100 = (100 * P - 60 * Q) / Q := by
  sorry

end percent_above_sixty_percent_l4091_409185


namespace kamal_present_age_l4091_409101

/-- Represents the present age of Kamal -/
def kamal_age : ℕ := sorry

/-- Represents the present age of Kamal's son -/
def son_age : ℕ := sorry

/-- Kamal was 4 times as old as his son 8 years ago -/
axiom condition1 : kamal_age - 8 = 4 * (son_age - 8)

/-- After 8 years, Kamal will be twice as old as his son -/
axiom condition2 : kamal_age + 8 = 2 * (son_age + 8)

/-- Theorem stating that Kamal's present age is 40 years -/
theorem kamal_present_age : kamal_age = 40 := by sorry

end kamal_present_age_l4091_409101


namespace weight_replacement_l4091_409116

theorem weight_replacement (initial_count : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  average_increase = 2.5 →
  new_weight = 55 →
  ∃ (replaced_weight : ℝ),
    replaced_weight = new_weight - (initial_count * average_increase) ∧
    replaced_weight = 35 :=
by sorry

end weight_replacement_l4091_409116


namespace flea_can_reach_all_naturals_l4091_409110

def jump_length (k : ℕ) : ℕ := 2^k + 1

theorem flea_can_reach_all_naturals :
  ∀ n : ℕ, ∃ (jumps : List (ℕ × Bool)), 
    (jumps.foldl (λ acc (len, dir) => if dir then acc + len else acc - len) 0 : ℤ) = n ∧
    ∀ k, k < jumps.length → (jumps.get ⟨k, by sorry⟩).1 = jump_length k :=
by sorry

end flea_can_reach_all_naturals_l4091_409110


namespace plane_equivalence_l4091_409107

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
def parametric_plane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - t
    y := 1 - 2*s
    z := 4 - s + 3*t }

/-- Represents the Cartesian equation of a plane -/
def cartesian_plane (p : Point3D) : Prop :=
  6 * p.x + 5 * p.y + 2 * p.z - 25 = 0

/-- Theorem stating the equivalence of the parametric and Cartesian representations -/
theorem plane_equivalence :
  ∀ (p : Point3D), (∃ (s t : ℝ), p = parametric_plane s t) ↔ cartesian_plane p :=
sorry

end plane_equivalence_l4091_409107


namespace intersection_locus_is_ellipse_l4091_409135

theorem intersection_locus_is_ellipse :
  ∀ (x y u : ℝ),
  (2 * u * x - 3 * y - 4 * u = 0) →
  (x - 3 * u * y + 4 = 0) →
  (x^2 / 16 + y^2 / 4 = 1) :=
by sorry

end intersection_locus_is_ellipse_l4091_409135


namespace max_perimeter_of_third_rectangle_l4091_409147

-- Define the rectangles
structure Rectangle where
  width : ℕ
  height : ℕ

-- Define the problem setup
def rectangle1 : Rectangle := ⟨70, 110⟩
def rectangle2 : Rectangle := ⟨40, 80⟩

-- Function to calculate perimeter
def perimeter (r : Rectangle) : ℕ :=
  2 * (r.width + r.height)

-- Function to check if three rectangles can form a larger rectangle
def canFormLargerRectangle (r1 r2 r3 : Rectangle) : Prop :=
  (r1.width + r2.width = r3.width ∧ max r1.height r2.height = r3.height) ∨
  (r1.height + r2.height = r3.height ∧ max r1.width r2.width = r3.width) ∨
  (r1.width + r2.height = r3.width ∧ r1.height + r2.width = r3.height) ∨
  (r1.height + r2.width = r3.width ∧ r1.width + r2.height = r3.height)

-- Theorem statement
theorem max_perimeter_of_third_rectangle :
  ∃ (r3 : Rectangle), canFormLargerRectangle rectangle1 rectangle2 r3 ∧
    perimeter r3 = 300 ∧
    ∀ (r : Rectangle), canFormLargerRectangle rectangle1 rectangle2 r →
      perimeter r ≤ 300 := by
  sorry

end max_perimeter_of_third_rectangle_l4091_409147


namespace sophia_reading_progress_l4091_409130

theorem sophia_reading_progress (total_pages : ℕ) (pages_finished : ℚ) : 
  total_pages = 270 → pages_finished = 2/3 → 
  (pages_finished * total_pages : ℚ) - ((1 - pages_finished) * total_pages : ℚ) = 90 := by
  sorry


end sophia_reading_progress_l4091_409130


namespace distance_between_points_l4091_409166

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (-5, 3)
  let pointB : ℝ × ℝ := (6, 3)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = 11 := by
  sorry

end distance_between_points_l4091_409166


namespace tangent_ratio_problem_l4091_409164

theorem tangent_ratio_problem (α : ℝ) (h : Real.tan (π - α) = 1/3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1/2 := by
  sorry

end tangent_ratio_problem_l4091_409164


namespace speed_difference_proof_l4091_409173

/-- Prove that given a distance of 8 miles, if person A travels for 40 minutes
    and person B travels for 1 hour, the difference in their average speeds is 4 mph. -/
theorem speed_difference_proof (distance : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance = 8 →
  time_A = 40 / 60 →
  time_B = 1 →
  (distance / time_A) - (distance / time_B) = 4 := by
sorry

end speed_difference_proof_l4091_409173


namespace equal_numbers_exist_l4091_409140

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Condition for a quadratic polynomial -/
def IsQuadratic {α : Type*} [Field α] (f : QuadraticPolynomial α) :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem equal_numbers_exist (f : QuadraticPolynomial ℝ) (l t v : ℝ)
    (hf : IsQuadratic f)
    (hl : f l = t + v)
    (ht : f t = l + v)
    (hv : f v = l + t) :
    l = t ∨ l = v ∨ t = v := by
  sorry

end equal_numbers_exist_l4091_409140


namespace golden_section_division_l4091_409179

/-- Given a line segment AB of length a, prove that the point H that divides AB
    such that AH = a(√5 - 1)/2 makes AH the mean proportional between AB and HB. -/
theorem golden_section_division (a : ℝ) (h : a > 0) :
  let x := a * (Real.sqrt 5 - 1) / 2
  x * x = a * (a - x) :=
by sorry

end golden_section_division_l4091_409179


namespace solve_g_inequality_range_of_a_l4091_409196

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- Theorem for part (1)
theorem solve_g_inequality :
  ∀ x : ℝ, |g x| < 5 ↔ -2 < x ∧ x < 4 :=
sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end solve_g_inequality_range_of_a_l4091_409196


namespace fraction_of_fraction_l4091_409197

theorem fraction_of_fraction : 
  (2 / 5 : ℚ) * (1 / 3 : ℚ) / (3 / 4 : ℚ) = 8 / 45 := by sorry

end fraction_of_fraction_l4091_409197


namespace power_product_equality_l4091_409121

theorem power_product_equality : (-4 : ℝ)^2010 * (-0.25 : ℝ)^2011 = -0.25 := by sorry

end power_product_equality_l4091_409121


namespace age_problem_l4091_409163

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 := by
sorry

end age_problem_l4091_409163


namespace intersection_false_necessary_not_sufficient_for_union_false_l4091_409198

theorem intersection_false_necessary_not_sufficient_for_union_false (P Q : Prop) :
  (¬(P ∨ Q) → ¬(P ∧ Q)) ∧ (∃ (P Q : Prop), ¬(P ∧ Q) ∧ (P ∨ Q)) :=
by sorry

end intersection_false_necessary_not_sufficient_for_union_false_l4091_409198


namespace shopping_mall_escalator_problem_l4091_409148

/-- Represents the escalator and staircase system in the shopping mall -/
structure EscalatorSystem where
  escalator_speed : ℝ
  a_step_rate : ℝ
  b_step_rate : ℝ
  a_steps_up : ℕ
  b_steps_up : ℕ

/-- Represents the result of the problem -/
structure ProblemResult where
  exposed_steps : ℕ
  catchup_location : Bool  -- true if on staircase, false if on escalator
  steps_walked : ℕ

/-- The main theorem that proves the result of the problem -/
theorem shopping_mall_escalator_problem (sys : EscalatorSystem) 
  (h1 : sys.a_step_rate = 2 * sys.b_step_rate)
  (h2 : sys.a_steps_up = 24)
  (h3 : sys.b_steps_up = 16) :
  ∃ (result : ProblemResult), 
    result.exposed_steps = 48 ∧ 
    result.catchup_location = true ∧ 
    result.steps_walked = 176 :=
by
  sorry

end shopping_mall_escalator_problem_l4091_409148


namespace intersection_of_A_and_B_l4091_409156

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {0} := by
  sorry

end intersection_of_A_and_B_l4091_409156


namespace bird_families_to_asia_count_l4091_409155

/-- The number of bird families that flew away to Asia -/
def bird_families_to_asia (total_migrated : ℕ) (to_africa : ℕ) : ℕ :=
  total_migrated - to_africa

/-- Theorem stating that 80 bird families flew away to Asia -/
theorem bird_families_to_asia_count : 
  bird_families_to_asia 118 38 = 80 := by
  sorry

end bird_families_to_asia_count_l4091_409155


namespace leahs_coins_value_l4091_409139

/-- Represents the value of a coin in cents -/
inductive Coin
| Penny : Coin
| Nickel : Coin

/-- The value of a coin in cents -/
def coin_value : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5

/-- A collection of coins -/
structure CoinCollection :=
  (pennies : Nat)
  (nickels : Nat)

/-- The total number of coins in a collection -/
def total_coins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels

/-- The total value of coins in a collection in cents -/
def total_value (c : CoinCollection) : Nat :=
  c.pennies * coin_value Coin.Penny + c.nickels * coin_value Coin.Nickel

/-- The main theorem -/
theorem leahs_coins_value (c : CoinCollection) :
  total_coins c = 15 ∧
  c.pennies = c.nickels + 2 →
  total_value c = 44 := by
  sorry


end leahs_coins_value_l4091_409139


namespace monika_beans_purchase_l4091_409118

def mall_cost : ℚ := 250
def movie_cost : ℚ := 24
def num_movies : ℕ := 3
def bean_cost : ℚ := 1.25
def total_spent : ℚ := 347

theorem monika_beans_purchase :
  (total_spent - (mall_cost + movie_cost * num_movies)) / bean_cost = 20 := by
  sorry

end monika_beans_purchase_l4091_409118


namespace matrix_equation_proof_l4091_409169

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 1, 1]

theorem matrix_equation_proof :
  N^3 - 3 • (N^2) + 4 • N = !![6, 12; 3, 6] := by sorry

end matrix_equation_proof_l4091_409169


namespace garden_length_l4091_409150

/-- Proves that a rectangular garden with width 5 m and area 60 m² has a length of 12 m -/
theorem garden_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 5 → area = 60 → area = length * width → length = 12 := by
  sorry

end garden_length_l4091_409150


namespace bug_triangle_probability_sum_of_numerator_denominator_l4091_409158

/-- Probability of being at the starting vertex after n moves -/
def prob_at_start (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else
    let prev := prob_at_start (n - 1)
    let prev_prev := prob_at_start (n - 2)
    (prev_prev + 2 * prev) / 4

theorem bug_triangle_probability :
  prob_at_start 10 = 171 / 1024 :=
sorry

#eval Nat.gcd 171 1024  -- To verify that 171 and 1024 are coprime

theorem sum_of_numerator_denominator :
  171 + 1024 = 1195 :=
sorry

end bug_triangle_probability_sum_of_numerator_denominator_l4091_409158


namespace hotdogs_served_today_l4091_409170

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := 2

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := lunch_hotdogs + dinner_hotdogs

theorem hotdogs_served_today : total_hotdogs = 11 := by
  sorry

end hotdogs_served_today_l4091_409170


namespace maurice_age_proof_l4091_409106

/-- Ron's current age -/
def ron_current_age : ℕ := 43

/-- Maurice's current age -/
def maurice_current_age : ℕ := 7

/-- Theorem stating that Maurice's current age is 7 years -/
theorem maurice_age_proof :
  (ron_current_age + 5 = 4 * (maurice_current_age + 5)) →
  maurice_current_age = 7 :=
by
  sorry

end maurice_age_proof_l4091_409106


namespace smallest_number_divisible_l4091_409165

theorem smallest_number_divisible (n : ℕ) : n ≥ 1015 ∧ 
  (∀ m : ℕ, m < 1015 → ¬(12 ∣ (m - 7) ∧ 16 ∣ (m - 7) ∧ 18 ∣ (m - 7) ∧ 21 ∣ (m - 7) ∧ 28 ∣ (m - 7))) →
  (12 ∣ (n - 7) ∧ 16 ∣ (n - 7) ∧ 18 ∣ (n - 7) ∧ 21 ∣ (n - 7) ∧ 28 ∣ (n - 7)) :=
by sorry

end smallest_number_divisible_l4091_409165


namespace two_numbers_product_l4091_409138

theorem two_numbers_product (n : ℕ) (h : n = 34) : ∃ x y : ℕ, 
  x ∈ Finset.range (n + 1) ∧ 
  y ∈ Finset.range (n + 1) ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range (n + 1)) id) - x - y = 22 * (y - x) ∧
  x * y = 416 := by
sorry

end two_numbers_product_l4091_409138


namespace george_final_stickers_l4091_409180

/-- The number of stickers each person has --/
structure Stickers where
  bob : ℕ
  tom : ℕ
  dan : ℕ
  george : ℕ

/-- The conditions of the problem --/
def sticker_conditions (s : Stickers) : Prop :=
  s.dan = 2 * s.tom ∧
  s.tom = 3 * s.bob ∧
  s.george = 5 * s.dan ∧
  s.bob = 12

/-- The total number of stickers to be distributed --/
def extra_stickers : ℕ := 100

/-- The number of people --/
def num_people : ℕ := 4

/-- Theorem stating that George will have 505 stickers in total --/
theorem george_final_stickers (s : Stickers) 
  (h : sticker_conditions s) : 
  s.george + (s.bob + s.tom + s.dan + s.george + extra_stickers) / num_people = 505 := by
  sorry


end george_final_stickers_l4091_409180


namespace max_profit_rate_l4091_409127

def f (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1
  else if 21 ≤ x ∧ x ≤ 60 then x / 10
  else 0

def g (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1 / (x + 80)
  else if 21 ≤ x ∧ x ≤ 60 then (2 * x) / (x^2 - x + 1600)
  else 0

theorem max_profit_rate :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 → g x ≤ 2/79 ∧ g 40 = 2/79 :=
by sorry

end max_profit_rate_l4091_409127


namespace i_power_difference_l4091_409193

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the cyclic property of i with period 4
axiom i_cyclic (n : ℤ) : i^n = i^(n % 4)

-- Theorem to prove
theorem i_power_difference : i^37 - i^29 = 0 := by sorry

end i_power_difference_l4091_409193


namespace sufficient_not_necessary_condition_l4091_409149

/-- A complex number z is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 2) (m^2 - 3*m - 2)

/-- m = -1 is a sufficient but not necessary condition for z to be purely imaginary -/
theorem sufficient_not_necessary_condition :
  (isPurelyImaginary (z (-1))) ∧
  (∃ m : ℝ, m ≠ -1 ∧ isPurelyImaginary (z m)) := by sorry

end sufficient_not_necessary_condition_l4091_409149


namespace runners_meet_at_6000_seconds_l4091_409171

/-- The time at which three runners meet again on a circular track -/
def runners_meeting_time (track_length : ℝ) (speed1 speed2 speed3 : ℝ) : ℝ :=
  let t := 6000
  t

/-- Theorem stating that the runners meet after 6000 seconds -/
theorem runners_meet_at_6000_seconds (track_length : ℝ) (speed1 speed2 speed3 : ℝ)
  (h_track : track_length = 600)
  (h_speed1 : speed1 = 4.4)
  (h_speed2 : speed2 = 4.9)
  (h_speed3 : speed3 = 5.1) :
  runners_meeting_time track_length speed1 speed2 speed3 = 6000 := by
  sorry

#check runners_meet_at_6000_seconds

end runners_meet_at_6000_seconds_l4091_409171


namespace reece_climbs_l4091_409128

-- Define constants
def keaton_ladder_feet : ℕ := 30
def keaton_climbs : ℕ := 20
def ladder_difference_feet : ℕ := 4
def total_climbed_inches : ℕ := 11880

-- Define functions
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

def reece_ladder_feet : ℕ := keaton_ladder_feet - ladder_difference_feet

-- Theorem statement
theorem reece_climbs : 
  (feet_to_inches keaton_ladder_feet * keaton_climbs + 
   feet_to_inches reece_ladder_feet * 15 = total_climbed_inches) := by
sorry

end reece_climbs_l4091_409128


namespace max_d_is_four_l4091_409114

/-- A function that constructs a 6-digit number of the form 6d6,33e -/
def construct_number (d e : ℕ) : ℕ := 
  600000 + d * 10000 + 6 * 1000 + 300 + 30 + e

/-- Proposition: The maximum value of d is 4 -/
theorem max_d_is_four :
  ∃ (d e : ℕ),
    d ≤ 9 ∧ e ≤ 9 ∧
    (construct_number d e) % 33 = 0 ∧
    d + e = 4 ∧
    ∀ (d' e' : ℕ), d' ≤ 9 ∧ e' ≤ 9 ∧ 
      (construct_number d' e') % 33 = 0 ∧ 
      d' + e' = 4 → 
      d' ≤ d :=
by
  sorry

end max_d_is_four_l4091_409114


namespace pascal_contest_average_age_l4091_409126

/-- Represents an age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  h : months < 12

/-- Converts an Age to total months -/
def ageToMonths (a : Age) : ℕ := a.years * 12 + a.months

/-- Converts total months to an Age -/
def monthsToAge (m : ℕ) : Age :=
  { years := m / 12
  , months := m % 12
  , h := by sorry }

/-- The average age of three contestants in the Pascal Contest -/
theorem pascal_contest_average_age (a1 a2 a3 : Age)
  (h1 : a1 = { years := 14, months := 9, h := by sorry })
  (h2 : a2 = { years := 15, months := 1, h := by sorry })
  (h3 : a3 = { years := 14, months := 8, h := by sorry }) :
  monthsToAge ((ageToMonths a1 + ageToMonths a2 + ageToMonths a3) / 3) =
  { years := 14, months := 10, h := by sorry } :=
by sorry

end pascal_contest_average_age_l4091_409126


namespace problem_solution_l4091_409187

def M : Set ℝ := {y | ∃ x, y = 3^x}
def N : Set ℝ := {-1, 0, 1}

theorem problem_solution : (Set.univ \ M) ∩ N = {-1, 0} := by sorry

end problem_solution_l4091_409187


namespace final_price_calculation_l4091_409120

-- Define the original price
def original_price : ℝ := 120

-- Define the first discount rate
def first_discount_rate : ℝ := 0.20

-- Define the second discount rate
def second_discount_rate : ℝ := 0.15

-- Theorem to prove
theorem final_price_calculation :
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 81.60 := by
  sorry


end final_price_calculation_l4091_409120


namespace classroom_ratio_l4091_409191

theorem classroom_ratio :
  ∀ (num_boys num_girls : ℕ),
  num_boys > 0 →
  num_girls > 0 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 
    3 * ((num_girls : ℚ) / (num_boys + num_girls : ℚ)) / 5 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3/8 :=
by
  sorry

end classroom_ratio_l4091_409191


namespace identity_proof_l4091_409177

theorem identity_proof (a b m n x y : ℝ) :
  (a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2) = 
  (a*n*y - a*m*x - b*m*y + b*n*x)^2 + (a*m*y + a*n*x + b*m*x - b*n*y)^2 := by
  sorry

end identity_proof_l4091_409177


namespace rectangle_y_value_l4091_409145

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices := [(0, y), (10, y), (0, 4), (10, 4)]
  let area := 90
  let length := 10
  let height := y - 4
  (length * height = area) → y = 13 := by
sorry

end rectangle_y_value_l4091_409145


namespace triangle_side_difference_l4091_409167

theorem triangle_side_difference (x : ℕ) : 
  x > 0 → x + 8 > 10 → x + 10 > 8 → 8 + 10 > x → 
  (∃ (max min : ℕ), 
    (∀ y : ℕ, y > 0 → y + 8 > 10 → y + 10 > 8 → 8 + 10 > y → y ≤ max) ∧
    (∀ y : ℕ, y > 0 → y + 8 > 10 → y + 10 > 8 → 8 + 10 > y → y ≥ min) ∧
    max - min = 14) := by
  sorry

end triangle_side_difference_l4091_409167


namespace car_travel_time_l4091_409125

/-- Given a truck and car with specific conditions, prove the car's travel time --/
theorem car_travel_time (truck_distance : ℝ) (truck_time : ℝ) (speed_difference : ℝ) (distance_difference : ℝ) :
  truck_distance = 296 →
  truck_time = 8 →
  speed_difference = 18 →
  distance_difference = 6.5 →
  let truck_speed := truck_distance / truck_time
  let car_speed := truck_speed + speed_difference
  let car_distance := truck_distance + distance_difference
  car_distance / car_speed = 5.5 := by sorry

end car_travel_time_l4091_409125


namespace mean_books_read_l4091_409112

def readers_3 : ℕ := 4
def books_3 : ℕ := 3
def readers_5 : ℕ := 5
def books_5 : ℕ := 5
def readers_7 : ℕ := 2
def books_7 : ℕ := 7
def readers_10 : ℕ := 1
def books_10 : ℕ := 10

def total_readers : ℕ := readers_3 + readers_5 + readers_7 + readers_10
def total_books : ℕ := readers_3 * books_3 + readers_5 * books_5 + readers_7 * books_7 + readers_10 * books_10

theorem mean_books_read :
  (total_books : ℚ) / (total_readers : ℚ) = 61 / 12 :=
sorry

end mean_books_read_l4091_409112


namespace at_least_one_by_cellini_son_not_both_by_cellini_not_both_by_other_l4091_409172

-- Define the possible makers of the caskets
inductive Maker
| Cellini
| CelliniSon
| Other

-- Define the caskets
structure Casket where
  material : String
  inscription : String
  maker : Maker

-- Define the problem setup
def goldenCasket : Casket := {
  material := "golden"
  inscription := "The silver casket was made by Cellini."
  maker := Maker.Other -- Initial assumption, will be proved
}

def silverCasket : Casket := {
  material := "silver"
  inscription := "The golden casket was made by someone other than Cellini."
  maker := Maker.Other -- Initial assumption, will be proved
}

-- The main theorem to prove
theorem at_least_one_by_cellini_son (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  g.maker = Maker.CelliniSon ∨ s.maker = Maker.CelliniSon := by
  sorry

-- Additional helper theorems if needed
theorem not_both_by_cellini (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  ¬(g.maker = Maker.Cellini ∧ s.maker = Maker.Cellini) := by
  sorry

theorem not_both_by_other (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  ¬(g.maker = Maker.Other ∧ s.maker = Maker.Other) := by
  sorry

end at_least_one_by_cellini_son_not_both_by_cellini_not_both_by_other_l4091_409172


namespace maxwell_current_age_l4091_409103

/-- Maxwell's current age --/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age --/
def sister_age : ℕ := 2

/-- Years into the future when Maxwell will be twice his sister's age --/
def years_future : ℕ := 2

theorem maxwell_current_age :
  maxwell_age = 6 ∧
  sister_age = 2 ∧
  maxwell_age + years_future = 2 * (sister_age + years_future) :=
by sorry

end maxwell_current_age_l4091_409103


namespace decreasing_quadratic_implies_a_geq_two_l4091_409123

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 5

-- State the theorem
theorem decreasing_quadratic_implies_a_geq_two :
  ∀ a : ℝ, (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) → a ≥ 2 :=
by sorry

end decreasing_quadratic_implies_a_geq_two_l4091_409123
