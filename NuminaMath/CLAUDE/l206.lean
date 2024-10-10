import Mathlib

namespace max_y_value_l206_20637

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * Real.log (y / x) - y * Real.exp x + x * (x + 1) ≥ 0) : 
  y ≤ 1 / Real.exp 1 := by
sorry

end max_y_value_l206_20637


namespace num_triangles_in_dodecagon_l206_20623

/-- A regular dodecagon has 12 vertices -/
def regular_dodecagon_vertices : ℕ := 12

/-- The number of triangles formed by choosing 3 vertices from a regular dodecagon -/
def num_triangles : ℕ := Nat.choose regular_dodecagon_vertices 3

/-- Theorem: The number of triangles formed by choosing 3 vertices from a regular dodecagon is 220 -/
theorem num_triangles_in_dodecagon : num_triangles = 220 := by sorry

end num_triangles_in_dodecagon_l206_20623


namespace geometric_sequence_arithmetic_mean_l206_20614

/-- The arithmetic mean of the first three terms of a geometric sequence 
    with first term 4 and common ratio 3 is 52/3. -/
theorem geometric_sequence_arithmetic_mean : 
  let a : ℝ := 4  -- First term
  let r : ℝ := 3  -- Common ratio
  let term1 : ℝ := a
  let term2 : ℝ := a * r
  let term3 : ℝ := a * r^2
  (term1 + term2 + term3) / 3 = 52 / 3 := by
  sorry


end geometric_sequence_arithmetic_mean_l206_20614


namespace percentage_difference_l206_20627

theorem percentage_difference : (80 / 100 * 60) - (4 / 5 * 25) = 28 := by
  sorry

end percentage_difference_l206_20627


namespace fifth_element_row_21_l206_20669

/-- Pascal's triangle element -/
def pascal_triangle_element (n : ℕ) (k : ℕ) : ℕ := Nat.choose n (k - 1)

/-- The fifth element in Row 21 of Pascal's triangle is 1995 -/
theorem fifth_element_row_21 : pascal_triangle_element 21 5 = 1995 := by
  sorry

end fifth_element_row_21_l206_20669


namespace bus_passengers_l206_20671

theorem bus_passengers (initial_passengers : ℕ) : 
  initial_passengers + 16 - 22 + 5 = 49 → initial_passengers = 50 := by
  sorry

end bus_passengers_l206_20671


namespace laborer_wage_calculation_l206_20607

/-- The daily wage of a general laborer -/
def laborer_wage : ℕ :=
  -- Define the wage here
  sorry

/-- The number of people hired -/
def total_hired : ℕ := 31

/-- The total payroll in dollars -/
def total_payroll : ℕ := 3952

/-- The daily wage of a heavy operator -/
def operator_wage : ℕ := 129

/-- The number of laborers employed -/
def laborers_employed : ℕ := 1

theorem laborer_wage_calculation : 
  laborer_wage = 82 ∧
  total_hired * operator_wage - (total_hired - laborers_employed) * operator_wage + laborer_wage = total_payroll :=
by sorry

end laborer_wage_calculation_l206_20607


namespace arthur_actual_weight_l206_20646

/-- The weight shown on the scales when weighing King Arthur -/
def arthur_scale : ℕ := 19

/-- The weight shown on the scales when weighing the royal horse -/
def horse_scale : ℕ := 101

/-- The weight shown on the scales when weighing King Arthur and the horse together -/
def combined_scale : ℕ := 114

/-- The actual weight of King Arthur -/
def arthur_weight : ℕ := 13

/-- The consistent error of the scales -/
def scale_error : ℕ := 6

theorem arthur_actual_weight :
  arthur_weight + scale_error = arthur_scale ∧
  arthur_weight + (horse_scale - scale_error) + scale_error = combined_scale :=
sorry

end arthur_actual_weight_l206_20646


namespace simplify_expression_l206_20673

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  a / b + b / a - 1 / (a * b) = 1 := by
sorry

end simplify_expression_l206_20673


namespace valid_grid_has_twelve_red_cells_l206_20661

/-- Represents the color of a cell -/
inductive Color
| Red
| Blue

/-- Represents a 4x4 grid of colored cells -/
def Grid := Fin 4 → Fin 4 → Color

/-- Returns the list of neighboring cells for a given position -/
def neighbors (i j : Fin 4) : List (Fin 4 × Fin 4) :=
  sorry

/-- Counts the number of neighbors of a given color -/
def countNeighbors (g : Grid) (i j : Fin 4) (c : Color) : Nat :=
  sorry

/-- Checks if the grid satisfies the conditions for red cells -/
def validRedCells (g : Grid) : Prop :=
  ∀ i j, g i j = Color.Red →
    countNeighbors g i j Color.Red > countNeighbors g i j Color.Blue

/-- Checks if the grid satisfies the conditions for blue cells -/
def validBlueCells (g : Grid) : Prop :=
  ∀ i j, g i j = Color.Blue →
    countNeighbors g i j Color.Red = countNeighbors g i j Color.Blue

/-- Counts the total number of red cells in the grid -/
def countRedCells (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that a valid grid has exactly 12 red cells -/
theorem valid_grid_has_twelve_red_cells (g : Grid)
  (h_red : validRedCells g)
  (h_blue : validBlueCells g)
  (h_both_colors : ∃ i j, g i j = Color.Red ∧ ∃ i' j', g i' j' = Color.Blue) :
  countRedCells g = 12 :=
sorry

end valid_grid_has_twelve_red_cells_l206_20661


namespace bacteria_growth_rate_l206_20604

/-- Represents the growth rate of bacteria in a dish -/
def growth_rate (r : ℝ) : Prop :=
  ∀ (t : ℕ), t ≥ 0 → (1 / 16 : ℝ) * r^30 = r^26 ∧ r^30 = r^30

theorem bacteria_growth_rate :
  ∃ (r : ℝ), r > 0 ∧ growth_rate r ∧ r = 2 :=
sorry

end bacteria_growth_rate_l206_20604


namespace smallest_blocks_needed_l206_20648

/-- Represents the dimensions of a block -/
structure Block where
  height : ℕ
  length : ℕ

/-- Represents the dimensions of the wall -/
structure Wall where
  length : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for the wall -/
def blocksNeeded (wall : Wall) (block3 : Block) (block1 : Block) : ℕ :=
  let rowsCount := wall.height / block3.height
  let oddRowBlocks := wall.length / block3.length
  let evenRowBlocks := 2 + (wall.length - 2 * block1.length) / block3.length
  (rowsCount / 2) * oddRowBlocks + ((rowsCount + 1) / 2) * evenRowBlocks

/-- The theorem stating the smallest number of blocks needed -/
theorem smallest_blocks_needed (wall : Wall) (block3 : Block) (block1 : Block) :
  wall.length = 120 ∧ wall.height = 8 ∧
  block3.height = 1 ∧ block3.length = 3 ∧
  block1.height = 1 ∧ block1.length = 1 →
  blocksNeeded wall block3 block1 = 324 := by
  sorry

#eval blocksNeeded ⟨120, 8⟩ ⟨1, 3⟩ ⟨1, 1⟩

end smallest_blocks_needed_l206_20648


namespace trajectory_of_right_angle_vertex_l206_20650

/-- Given points M(-2,0) and N(2,0), prove that any point P(x,y) forming a right-angled triangle
    with MN as the hypotenuse satisfies the equation x^2 + y^2 = 4, where x ≠ ±2. -/
theorem trajectory_of_right_angle_vertex (x y : ℝ) :
  x ≠ -2 → x ≠ 2 →
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 →
  x^2 + y^2 = 4 :=
by sorry

end trajectory_of_right_angle_vertex_l206_20650


namespace variance_of_transformed_binomial_l206_20656

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- The variance of a linear transformation of a random variable -/
def varianceLinearTransform (a b : ℝ) (v : ℝ) : ℝ := a^2 * v

theorem variance_of_transformed_binomial :
  let ξ : BinomialDistribution := ⟨100, 0.3, by norm_num⟩
  varianceLinearTransform 3 (-5) (variance ξ) = 189 := by
  sorry

end variance_of_transformed_binomial_l206_20656


namespace total_leaves_eq_696_l206_20624

def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6
def cilantro_pots : ℕ := 7
def lavender_pots : ℕ := 4

def basil_leaves_per_plant : ℕ := 4
def rosemary_leaves_per_plant : ℕ := 18
def thyme_leaves_per_plant : ℕ := 30
def cilantro_leaves_per_plant : ℕ := 42
def lavender_leaves_per_plant : ℕ := 12

def total_leaves : ℕ := 
  basil_pots * basil_leaves_per_plant +
  rosemary_pots * rosemary_leaves_per_plant +
  thyme_pots * thyme_leaves_per_plant +
  cilantro_pots * cilantro_leaves_per_plant +
  lavender_pots * lavender_leaves_per_plant

theorem total_leaves_eq_696 : total_leaves = 696 := by
  sorry

end total_leaves_eq_696_l206_20624


namespace insurance_percentage_l206_20611

theorem insurance_percentage (salary tax_rate utility_rate remaining_amount : ℝ) 
  (h1 : salary = 2000)
  (h2 : tax_rate = 0.2)
  (h3 : utility_rate = 0.25)
  (h4 : remaining_amount = 1125)
  (h5 : ∃ insurance_rate : ℝ, 
    remaining_amount = salary * (1 - tax_rate - insurance_rate) * (1 - utility_rate)) :
  ∃ insurance_rate : ℝ, insurance_rate = 0.05 := by
sorry

end insurance_percentage_l206_20611


namespace probability_white_or_red_ball_l206_20609

theorem probability_white_or_red_ball (white black red : ℕ) 
  (h_white : white = 8)
  (h_black : black = 7)
  (h_red : red = 4) :
  (white + red : ℚ) / (white + black + red) = 12 / 19 :=
by sorry

end probability_white_or_red_ball_l206_20609


namespace complex_power_sum_l206_20663

theorem complex_power_sum (i : ℂ) (h : i^2 = -1) :
  i^14 + i^19 + i^24 + i^29 + 3*i^34 + 2*i^39 = -3 - 2*i := by
  sorry

end complex_power_sum_l206_20663


namespace probability_two_painted_faces_is_three_eighths_l206_20688

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  total_small_cubes : ℕ
  small_cubes_with_two_painted_faces : ℕ

/-- The probability of selecting a small cube with exactly two painted faces -/
def probability_two_painted_faces (c : CutCube) : ℚ :=
  c.small_cubes_with_two_painted_faces / c.total_small_cubes

/-- A cube cut into 64 smaller cubes -/
def cube_64 : CutCube :=
  { total_small_cubes := 64,
    small_cubes_with_two_painted_faces := 24 }

theorem probability_two_painted_faces_is_three_eighths :
  probability_two_painted_faces cube_64 = 3 / 8 := by
  sorry

end probability_two_painted_faces_is_three_eighths_l206_20688


namespace abc_sum_l206_20699

theorem abc_sum (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b = 2 * (a + b)) (hbc : b * c = 3 * (b + c)) (hca : c * a = 4 * (c + a)) :
  a + b + c = 1128 / 35 := by
  sorry

end abc_sum_l206_20699


namespace linear_function_max_value_l206_20670

theorem linear_function_max_value (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → m * x - 2 * m ≤ 6) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ m * x - 2 * m = 6) →
  m = -2 ∨ m = 6 := by
sorry

end linear_function_max_value_l206_20670


namespace friends_team_assignment_l206_20677

theorem friends_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 := by
  sorry

end friends_team_assignment_l206_20677


namespace circle_radius_determines_m_l206_20641

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle_equation (h k r m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 - 2*h*x - 2*k*y + (h^2 + k^2 - r^2 + m) = 0

theorem circle_radius_determines_m :
  ∀ m : ℝ, (∃ h k : ℝ, is_circle_equation h k 2 m) → m = 1 :=
sorry

end circle_radius_determines_m_l206_20641


namespace total_cases_after_three_days_l206_20683

-- Define the parameters
def initial_cases : ℕ := 2000
def increase_rate : ℚ := 20 / 100
def recovery_rate : ℚ := 2 / 100
def days : ℕ := 3

-- Function to calculate the cases for the next day
def next_day_cases (current_cases : ℚ) : ℚ :=
  current_cases + current_cases * increase_rate - current_cases * recovery_rate

-- Function to calculate cases after n days
def cases_after_days (n : ℕ) : ℚ :=
  match n with
  | 0 => initial_cases
  | n + 1 => next_day_cases (cases_after_days n)

-- Theorem statement
theorem total_cases_after_three_days :
  ⌊cases_after_days days⌋ = 3286 :=
sorry

end total_cases_after_three_days_l206_20683


namespace welcoming_and_planning_committees_l206_20662

theorem welcoming_and_planning_committees 
  (n : ℕ) -- Number of students
  (h1 : Nat.choose n 2 = 10) -- There are 10 ways to choose 2 from n
  : Nat.choose n 3 = 10 := by
  sorry

end welcoming_and_planning_committees_l206_20662


namespace square_area_from_perimeter_l206_20655

theorem square_area_from_perimeter (p : ℝ) (p_pos : p > 0) : 
  let perimeter := 12 * p
  let side_length := perimeter / 4
  let area := side_length ^ 2
  area = 9 * p ^ 2 := by
sorry

end square_area_from_perimeter_l206_20655


namespace arithmetic_equality_l206_20667

theorem arithmetic_equality : 4 * 7 + 5 * 12 + 12 * 4 + 4 * 9 = 172 := by
  sorry

end arithmetic_equality_l206_20667


namespace problem_solution_l206_20651

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : a * b + b * c + a * c = a * b * c)
  (h5 : a * b * c = d) : d = 36 := by
  sorry

end problem_solution_l206_20651


namespace white_ball_count_l206_20666

/-- Given a bag of 100 glass balls with red, black, and white colors,
    prove that if the frequency of drawing red balls is 15% and black balls is 40%,
    then the number of white balls is 45. -/
theorem white_ball_count (total : ℕ) (red_freq black_freq : ℚ) :
  total = 100 →
  red_freq = 15 / 100 →
  black_freq = 40 / 100 →
  ∃ (white_count : ℕ), white_count = 45 ∧ white_count = total * (1 - red_freq - black_freq) :=
sorry

end white_ball_count_l206_20666


namespace collinear_points_b_value_l206_20657

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end collinear_points_b_value_l206_20657


namespace hexagon_congruent_angles_l206_20617

/-- In a hexagon with three congruent angles and two pairs of supplementary angles,
    each of the congruent angles measures 120 degrees. -/
theorem hexagon_congruent_angles (F I G U R E : Real) : 
  F = I ∧ I = U ∧  -- Three angles are congruent
  G + E = 180 ∧    -- One pair of supplementary angles
  R + U = 180 ∧    -- Another pair of supplementary angles
  F + I + G + U + R + E = 720  -- Sum of angles in a hexagon
  → U = 120 := by sorry

end hexagon_congruent_angles_l206_20617


namespace total_supermarkets_l206_20640

def FGH_chain (us canada : ℕ) : Prop :=
  (us = 49) ∧ (us = canada + 14)

theorem total_supermarkets (us canada : ℕ) (h : FGH_chain us canada) : 
  us + canada = 84 :=
sorry

end total_supermarkets_l206_20640


namespace negation_of_implication_l206_20645

theorem negation_of_implication :
  ¬(x = 1 → x^2 = 1) ↔ (x = 1 → x^2 ≠ 1) :=
by sorry

end negation_of_implication_l206_20645


namespace num_triangles_eq_choose_l206_20676

/-- The number of triangles formed by n lines in general position on a plane -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3

/-- 
Theorem: The number of triangles formed by n lines in general position on a plane
is equal to (n choose 3).
-/
theorem num_triangles_eq_choose (n : ℕ) : 
  num_triangles n = Nat.choose n 3 := by
  sorry

end num_triangles_eq_choose_l206_20676


namespace quadratic_inequality_l206_20687

-- Define the quadratic polynomial
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, quadratic a b c x < 0) :
  b / a < c / a + 1 := by
  sorry

end quadratic_inequality_l206_20687


namespace bad_carrots_count_l206_20642

/-- The number of bad carrots in Vanessa's garden -/
def bad_carrots (vanessa_carrots mother_carrots good_carrots : ℕ) : ℕ :=
  vanessa_carrots + mother_carrots - good_carrots

theorem bad_carrots_count : bad_carrots 17 14 24 = 7 := by
  sorry

end bad_carrots_count_l206_20642


namespace expression_equivalence_l206_20615

theorem expression_equivalence :
  let original := -1/2 + Real.sqrt 3 / 2
  let a := -(1 + Real.sqrt 3) / 2
  let b := (Real.sqrt 3 - 1) / 2
  let c := -(1 - Real.sqrt 3) / 2
  let d := (-1 + Real.sqrt 3) / 2
  (a ≠ original) ∧ (b = original) ∧ (c = original) ∧ (d = original) := by
sorry

end expression_equivalence_l206_20615


namespace x_plus_p_equals_2p_plus_3_l206_20681

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2*p + 3 := by
  sorry

end x_plus_p_equals_2p_plus_3_l206_20681


namespace ones_digit_of_8_to_47_l206_20626

theorem ones_digit_of_8_to_47 : 8^47 % 10 = 2 := by sorry

end ones_digit_of_8_to_47_l206_20626


namespace final_amoeba_type_l206_20682

/-- Represents the type of a Martian amoeba -/
inductive AmoebaTy
  | A
  | B
  | C

/-- Represents the state of the amoeba population -/
structure AmoebaPop where
  a : Nat
  b : Nat
  c : Nat

/-- Merges two amoebas of different types into the third type -/
def merge (pop : AmoebaPop) : AmoebaPop :=
  sorry

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Prop :=
  n % 2 = 1

/-- The initial population of amoebas -/
def initialPop : AmoebaPop :=
  { a := 20, b := 21, c := 22 }

theorem final_amoeba_type (finalPop : AmoebaPop)
    (h : ∃ n : Nat, finalPop = (merge^[n] initialPop))
    (hTotal : finalPop.a + finalPop.b + finalPop.c = 1) :
    isOdd finalPop.b ∧ ¬isOdd finalPop.a ∧ ¬isOdd finalPop.c :=
  sorry

end final_amoeba_type_l206_20682


namespace arithmetic_computation_l206_20697

theorem arithmetic_computation : -(12 * 2) - (3 * 2) + (-18 / 3 * -4) = -6 := by
  sorry

end arithmetic_computation_l206_20697


namespace max_product_sum_2000_l206_20654

theorem max_product_sum_2000 : 
  ∃ (x : ℤ), ∀ (y : ℤ), y * (2000 - y) ≤ x * (2000 - x) ∧ x * (2000 - x) = 1000000 :=
by sorry

end max_product_sum_2000_l206_20654


namespace log_sum_equality_l206_20686

theorem log_sum_equality : 
  Real.log 8 / Real.log 2 + 3 * (Real.log 4 / Real.log 2) + 
  4 * (Real.log 16 / Real.log 4) + 2 * (Real.log 32 / Real.log 8) = 61 / 3 := by
  sorry

end log_sum_equality_l206_20686


namespace root_difference_ratio_l206_20606

theorem root_difference_ratio (a b : ℝ) : 
  a > b ∧ b > 0 ∧ 
  a^2 - 6*a + 4 = 0 ∧ 
  b^2 - 6*b + 4 = 0 → 
  (Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b) = Real.sqrt 5 / 5 := by
  sorry

end root_difference_ratio_l206_20606


namespace tan_alpha_minus_2beta_l206_20633

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2/5)
  (h2 : Real.tan β = 1/2) : 
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end tan_alpha_minus_2beta_l206_20633


namespace remaining_money_l206_20621

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := 5273
def rental_car_cost : ℕ := 1500

theorem remaining_money :
  octal_to_decimal john_savings - rental_car_cost = 1247 := by sorry

end remaining_money_l206_20621


namespace max_value_at_zero_l206_20698

/-- The function f(x) = x³ - 3x² + 1 reaches its maximum value at x = 0 -/
theorem max_value_at_zero (f : ℝ → ℝ) (h : f = λ x => x^3 - 3*x^2 + 1) :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ x₀ = 0 :=
sorry

end max_value_at_zero_l206_20698


namespace tens_digit_of_9_pow_2023_l206_20652

theorem tens_digit_of_9_pow_2023 : ∃ n : ℕ, 9^2023 ≡ 80 + n [ZMOD 100] ∧ 0 ≤ n ∧ n < 10 :=
sorry

end tens_digit_of_9_pow_2023_l206_20652


namespace unique_odd_divisors_pair_l206_20659

/-- A number has an odd number of divisors if and only if it is a perfect square -/
def has_odd_divisors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The theorem states that 576 is the only positive integer n such that
    both n and n + 100 have an odd number of divisors -/
theorem unique_odd_divisors_pair :
  ∀ n : ℕ, n > 0 ∧ has_odd_divisors n ∧ has_odd_divisors (n + 100) → n = 576 :=
sorry

end unique_odd_divisors_pair_l206_20659


namespace function_range_theorem_l206_20603

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * x - a

theorem function_range_theorem (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.sin x₀ ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc (Real.exp (-1) - 1) (Real.exp 1 + 1) :=
by sorry

end function_range_theorem_l206_20603


namespace burger_problem_l206_20602

theorem burger_problem (total_burgers : ℕ) (total_cost : ℚ) (single_cost : ℚ) (double_cost : ℚ) 
  (h1 : total_burgers = 50)
  (h2 : total_cost = 64.5)
  (h3 : single_cost = 1)
  (h4 : double_cost = 1.5) :
  ∃ (single_count double_count : ℕ),
    single_count + double_count = total_burgers ∧
    single_cost * single_count + double_cost * double_count = total_cost ∧
    double_count = 29 := by
  sorry

end burger_problem_l206_20602


namespace even_function_implies_a_eq_neg_one_l206_20613

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x + 1) * (x + a)

theorem even_function_implies_a_eq_neg_one :
  IsEven (f a) → a = -1 := by sorry

end even_function_implies_a_eq_neg_one_l206_20613


namespace a_10_value_a_satisfies_conditions_l206_20618

def sequence_a (n : ℕ+) : ℚ :=
  1 / (3 * n - 2)

theorem a_10_value :
  sequence_a 10 = 1 / 28 :=
by sorry

theorem a_satisfies_conditions :
  sequence_a 1 = 1 ∧
  ∀ n : ℕ+, 1 / sequence_a (n + 1) = 1 / sequence_a n + 3 :=
by sorry

end a_10_value_a_satisfies_conditions_l206_20618


namespace solution_when_a_eq_one_two_solutions_range_max_value_F_l206_20672

-- Define the functions
def f (a x : ℝ) := |x - a|
def g (a x : ℝ) := a * x
def F (a x : ℝ) := g a x * f a x

-- Theorem 1
theorem solution_when_a_eq_one :
  ∃ x : ℝ, f 1 x = g 1 x ∧ x = 1/2 := by sorry

-- Theorem 2
theorem two_solutions_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = g a x ∧ f a y = g a y) ↔ 
  (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) := by sorry

-- Theorem 3
theorem max_value_F :
  ∀ a : ℝ, a > 0 → 
  (∃ max : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 2 → F a x ≤ max) ∧
  (let max := if a < 5/3 then 4*a - 2*a^2
              else if a ≤ 2 then a^2 - a
              else if a < 4 then a^3/4
              else 2*a^2 - 4*a;
   ∀ x : ℝ, x ∈ Set.Icc 1 2 → F a x ≤ max) := by sorry

end solution_when_a_eq_one_two_solutions_range_max_value_F_l206_20672


namespace square_root_extraction_scheme_l206_20612

theorem square_root_extraction_scheme (n : Nat) (root : Nat) : 
  n = 418089 ∧ root = 647 → root * root = n := by
  sorry

end square_root_extraction_scheme_l206_20612


namespace decimal_expansion_3_11_l206_20675

theorem decimal_expansion_3_11 : 
  ∃ (n : ℕ) (a b : ℕ), 
    (3 : ℚ) / 11 = (a : ℚ) / (10^n - 1) ∧ 
    b = 10^n - 1 ∧ 
    n = 2 ∧
    a < b := by sorry

end decimal_expansion_3_11_l206_20675


namespace divisors_of_86400000_l206_20692

/-- The number of divisors of 86,400,000 -/
def num_divisors : ℕ := 264

/-- The sum of all divisors of 86,400,000 -/
def sum_divisors : ℕ := 319823280

/-- The prime factorization of 86,400,000 -/
def n : ℕ := 2^10 * 3^3 * 5^5

theorem divisors_of_86400000 :
  (∃ (d : Finset ℕ), d.card = num_divisors ∧ 
    (∀ x : ℕ, x ∈ d ↔ x ∣ n) ∧
    d.sum id = sum_divisors) :=
sorry

end divisors_of_86400000_l206_20692


namespace robotic_octopus_dressing_orders_l206_20660

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

end robotic_octopus_dressing_orders_l206_20660


namespace group_size_l206_20690

theorem group_size (B S B_intersect_S : ℕ) 
  (hB : B = 50)
  (hS : S = 70)
  (hIntersect : B_intersect_S = 20) :
  B + S - B_intersect_S = 100 := by
  sorry

end group_size_l206_20690


namespace popped_kernels_in_first_bag_l206_20653

/-- Represents the number of kernels in a bag -/
structure BagOfKernels where
  total : ℕ
  popped : ℕ

/-- Given information about three bags of popcorn kernels, proves that
    the number of popped kernels in the first bag is 61. -/
theorem popped_kernels_in_first_bag
  (bag1 : BagOfKernels)
  (bag2 : BagOfKernels)
  (bag3 : BagOfKernels)
  (h1 : bag1.total = 75)
  (h2 : bag2.total = 50 ∧ bag2.popped = 42)
  (h3 : bag3.total = 100 ∧ bag3.popped = 82)
  (h_avg : (bag1.popped + bag2.popped + bag3.popped) / (bag1.total + bag2.total + bag3.total) = 82 / 100) :
  bag1.popped = 61 := by
  sorry

#check popped_kernels_in_first_bag

end popped_kernels_in_first_bag_l206_20653


namespace completing_square_equivalence_l206_20619

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 2 = 0 ↔ (x - 3)^2 = 7 := by
  sorry

end completing_square_equivalence_l206_20619


namespace at_least_one_leq_neg_four_l206_20643

theorem at_least_one_leq_neg_four (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
sorry

end at_least_one_leq_neg_four_l206_20643


namespace cosine_equation_solution_l206_20695

theorem cosine_equation_solution (x : ℝ) : 
  (1 + Real.cos (3 * x) = 2 * Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = 2 * k * Real.pi ∨ x = Real.pi / 6 + k * Real.pi ∨ x = 5 * Real.pi / 6 + k * Real.pi) :=
by sorry

end cosine_equation_solution_l206_20695


namespace problem_statement_l206_20608

def additive_inverse (a b : ℚ) : Prop := a + b = 0

def multiplicative_inverse (b c : ℚ) : Prop := b * c = 1

def cubic_identity (m : ℚ) : Prop := m^3 = m

theorem problem_statement 
  (a b c m : ℚ) 
  (h1 : additive_inverse a b) 
  (h2 : multiplicative_inverse b c) 
  (h3 : cubic_identity m) :
  (∃ S : ℚ, 
    (2*a + 2*b) / (m + 2) + a*c = -1 ∧
    (a > 1 → m < 0 → 
      S = |2*a - 3*b| - 2*|b - m| - |b + 1/2| →
      4*(2*a - S) + 2*(2*a - S) - (2*a - S) = -25/2) ∧
    (m ≠ 0 → ∃ (max_val : ℚ), 
      (∀ (x : ℚ), |x + m| - |x - m| ≤ max_val) ∧
      (∃ (x : ℚ), |x + m| - |x - m| = max_val) ∧
      max_val = 2)) :=
by sorry

end problem_statement_l206_20608


namespace min_amount_spent_l206_20644

/-- Represents the price of a volleyball in yuan -/
def volleyball_price : ℝ := 80

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := 100

/-- Represents the total number of balls to be purchased -/
def total_balls : ℕ := 50

/-- Represents the minimum number of soccer balls to be purchased -/
def min_soccer_balls : ℕ := 25

/-- Theorem stating the minimum amount spent on purchasing the balls -/
theorem min_amount_spent :
  let x := min_soccer_balls
  let y := total_balls - x
  x * soccer_ball_price + y * volleyball_price = 4500 ∧
  x ≥ y ∧
  500 / soccer_ball_price = 400 / volleyball_price ∧
  soccer_ball_price = volleyball_price + 20 := by
  sorry


end min_amount_spent_l206_20644


namespace fraction_transformation_l206_20629

theorem fraction_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ) / (b + 3) = 2 * (a / b) → a = 2 ∧ b = 3 :=
sorry

end fraction_transformation_l206_20629


namespace power_of_power_l206_20616

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l206_20616


namespace salary_decrease_equivalence_l206_20647

-- Define the pay cuts
def first_cut : ℝ := 0.05
def second_cut : ℝ := 0.10
def third_cut : ℝ := 0.15

-- Define the function to calculate the equivalent single percentage decrease
def equivalent_decrease (c1 c2 c3 : ℝ) : ℝ :=
  (1 - (1 - c1) * (1 - c2) * (1 - c3)) * 100

-- State the theorem
theorem salary_decrease_equivalence :
  equivalent_decrease first_cut second_cut third_cut = 27.325 := by
  sorry

end salary_decrease_equivalence_l206_20647


namespace marble_problem_l206_20620

/-- The total number of marbles given the conditions of the problem -/
def total_marbles : ℕ := 36

/-- Mario's share of marbles before Manny gives away 2 marbles -/
def mario_marbles : ℕ := 16

/-- Manny's share of marbles before giving away 2 marbles -/
def manny_marbles : ℕ := 20

/-- The ratio of Mario's marbles to Manny's marbles -/
def marble_ratio : Rat := 4 / 5

theorem marble_problem :
  (mario_marbles : ℚ) / (manny_marbles : ℚ) = marble_ratio ∧
  manny_marbles - 2 = 18 ∧
  total_marbles = mario_marbles + manny_marbles :=
by sorry

end marble_problem_l206_20620


namespace sum_of_coefficients_l206_20635

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - x^3)^3 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 := by
  sorry

end sum_of_coefficients_l206_20635


namespace garrett_cat_count_l206_20630

/-- The number of cats Mrs. Sheridan has -/
def sheridan_cats : ℕ := 11

/-- The difference between Mrs. Garrett's and Mrs. Sheridan's cats -/
def cat_difference : ℕ := 13

/-- Mrs. Garrett's cats -/
def garrett_cats : ℕ := sheridan_cats + cat_difference

theorem garrett_cat_count : garrett_cats = 24 := by
  sorry

end garrett_cat_count_l206_20630


namespace valid_pairs_count_l206_20658

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

end valid_pairs_count_l206_20658


namespace min_value_of_f_l206_20634

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 + 3*y^2 + 8*x - 6*y + x*y + 22

/-- Theorem stating that the minimum value of f is 3 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 3 ∧ ∀ (x y : ℝ), f x y ≥ min :=
sorry

end min_value_of_f_l206_20634


namespace sum_of_digits_equality_l206_20638

def num1 : ℕ := (10^100 - 1) / 9
def num2 : ℕ := 4 * ((10^50 - 1) / 9)

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_equality :
  sumOfDigits (num1 * num2) = sumOfDigits (4 * (10^150 - 10^100 - 10^50 + 1) / 81) :=
by sorry

end sum_of_digits_equality_l206_20638


namespace reflection_line_equation_l206_20689

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

end reflection_line_equation_l206_20689


namespace rectangle_length_l206_20649

/-- Given a rectangle with width 4 inches and area 8 square inches, prove its length is 2 inches. -/
theorem rectangle_length (width : ℝ) (area : ℝ) (h1 : width = 4) (h2 : area = 8) :
  area / width = 2 := by
  sorry

end rectangle_length_l206_20649


namespace number_of_students_l206_20628

theorem number_of_students (N : ℕ) : 
  (N : ℚ) * 15 = 4 * 14 + 10 * 16 + 9 → N = 15 := by
  sorry

#check number_of_students

end number_of_students_l206_20628


namespace geometric_sequence_a10_l206_20691

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 6 = 2/3 → q = Real.sqrt 3 → a 10 = 6 := by
  sorry

end geometric_sequence_a10_l206_20691


namespace donny_piggy_bank_money_l206_20636

def initial_money (kite_price frisbee_price money_left : ℕ) : ℕ :=
  kite_price + frisbee_price + money_left

theorem donny_piggy_bank_money :
  initial_money 8 9 61 = 78 := by
  sorry

end donny_piggy_bank_money_l206_20636


namespace crossing_point_distance_less_than_one_l206_20639

/-- Represents a ladder in the ditch -/
structure Ladder :=
  (length : ℝ)
  (base_point : ℝ × ℝ)
  (top_point : ℝ × ℝ)

/-- Represents the ditch setup -/
structure DitchSetup :=
  (width : ℝ)
  (height : ℝ)
  (ladder1 : Ladder)
  (ladder2 : Ladder)

/-- The crossing point of two ladders -/
def crossing_point (l1 l2 : Ladder) : ℝ × ℝ := sorry

/-- Distance from a point to the left wall of the ditch -/
def distance_to_left_wall (p : ℝ × ℝ) : ℝ := p.1

/-- Main theorem: The crossing point is less than 1m from the left wall -/
theorem crossing_point_distance_less_than_one (setup : DitchSetup) :
  setup.ladder1.length = 3 →
  setup.ladder2.length = 2 →
  setup.ladder1.base_point.1 = 0 →
  setup.ladder2.base_point.1 = setup.width →
  setup.ladder1.top_point.2 = setup.height →
  setup.ladder2.top_point.2 = setup.height →
  distance_to_left_wall (crossing_point setup.ladder1 setup.ladder2) < 1 := by
  sorry

end crossing_point_distance_less_than_one_l206_20639


namespace max_value_expression_l206_20664

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (a + b) * (c + d) + (a + 1) * (d + 1) ≤ 112 :=
by sorry

end max_value_expression_l206_20664


namespace log_inequality_l206_20678

/-- The function f as defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

/-- The theorem statement -/
theorem log_inequality (m : ℝ) 
  (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f m x) 
  (h2 : m > 0) : 
  Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) := by
  sorry

end log_inequality_l206_20678


namespace smallest_integer_with_remainders_l206_20622

theorem smallest_integer_with_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 ∧
  n % 4 = 3 ∧
  n % 3 = 2 ∧
  n % 2 = 1 ∧
  (∀ m : ℕ, m > 0 →
    m % 10 = 9 →
    m % 9 = 8 →
    m % 8 = 7 →
    m % 7 = 6 →
    m % 6 = 5 →
    m % 5 = 4 →
    m % 4 = 3 →
    m % 3 = 2 →
    m % 2 = 1 →
    n ≤ m) ∧
  n = 2519 :=
by sorry

end smallest_integer_with_remainders_l206_20622


namespace weeks_to_save_l206_20696

def console_cost : ℕ := 282
def initial_savings : ℕ := 42
def weekly_allowance : ℕ := 24

theorem weeks_to_save : 
  (console_cost - initial_savings) / weekly_allowance = 10 :=
sorry

end weeks_to_save_l206_20696


namespace fraction_denominator_l206_20694

theorem fraction_denominator (n : ℕ) (d : ℕ) :
  n = 35 →
  (n : ℚ) / d = 2 / 10^20 →
  d = 175 * 10^20 := by
  sorry

end fraction_denominator_l206_20694


namespace shadow_length_sams_shadow_length_l206_20600

/-- Given a lamp post and a person walking towards it, this theorem calculates
    the length of the person's shadow at a new position. -/
theorem shadow_length (lamp_height : ℝ) (initial_distance : ℝ) (initial_shadow : ℝ) 
                      (new_distance : ℝ) : ℝ :=
  let person_height := lamp_height * initial_shadow / (initial_distance + initial_shadow)
  let new_shadow := person_height * new_distance / (lamp_height - person_height)
  new_shadow

/-- The main theorem that proves the specific shadow length for the given scenario. -/
theorem sams_shadow_length : 
  shadow_length 8 12 4 8 = 8/3 := by
  sorry

end shadow_length_sams_shadow_length_l206_20600


namespace right_triangle_shorter_leg_l206_20679

theorem right_triangle_shorter_leg (a b c m : ℝ) : 
  a > 0 → b > 0 → c > 0 → m > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle
  m = c / 2 →        -- Median to hypotenuse
  m = 15 →           -- Median length
  b = a + 9 →        -- One leg 9 units longer
  a = (-9 + Real.sqrt 1719) / 2 := by
sorry

end right_triangle_shorter_leg_l206_20679


namespace max_pieces_on_chessboard_l206_20693

/-- Represents a chessboard configuration -/
def ChessboardConfiguration := Fin 8 → Fin 8 → Bool

/-- Checks if a given position is on the board -/
def isOnBoard (row col : ℕ) : Prop := row < 8 ∧ col < 8

/-- Checks if a piece is placed at a given position -/
def hasPiece (config : ChessboardConfiguration) (row col : Fin 8) : Prop :=
  config row col = true

/-- Counts the number of pieces on a given diagonal -/
def piecesOnDiagonal (config : ChessboardConfiguration) (startRow startCol : Fin 8) (rowStep colStep : Int) : ℕ :=
  sorry

/-- Checks if the configuration is valid (no more than 3 pieces on any diagonal) -/
def isValidConfiguration (config : ChessboardConfiguration) : Prop :=
  ∀ (startRow startCol : Fin 8) (rowStep colStep : Int),
    piecesOnDiagonal config startRow startCol rowStep colStep ≤ 3

/-- Counts the total number of pieces on the board -/
def totalPieces (config : ChessboardConfiguration) : ℕ :=
  sorry

/-- The main theorem -/
theorem max_pieces_on_chessboard :
  ∃ (config : ChessboardConfiguration),
    isValidConfiguration config ∧
    totalPieces config = 38 ∧
    ∀ (otherConfig : ChessboardConfiguration),
      isValidConfiguration otherConfig →
      totalPieces otherConfig ≤ 38 :=
  sorry

end max_pieces_on_chessboard_l206_20693


namespace union_of_A_and_B_l206_20610

open Set

def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : A ∪ B = Ioc (-1) 3 := by sorry

end union_of_A_and_B_l206_20610


namespace triangle_cosine_b_l206_20680

theorem triangle_cosine_b (ω : ℝ) (A B C a b c : ℝ) :
  ω > 0 →
  (∀ x, 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) - 2 * Real.sin (ω * x / 2) ^ 2 =
        2 * Real.sin (2 * x / 3 + π / 6) - 1) →
  a < b →
  b < c →
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  2 * Real.sin (A + π / 2) - 1 = 11 / 13 →
  Real.cos B = (5 * Real.sqrt 3 + 12) / 26 := by
sorry

end triangle_cosine_b_l206_20680


namespace exponential_function_at_zero_l206_20632

theorem exponential_function_at_zero (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  (fun x : ℝ => a^x) 0 = 1 := by
  sorry

end exponential_function_at_zero_l206_20632


namespace consecutive_draw_probability_l206_20625

/-- The probability of drawing one red marble and then one blue marble consecutively from a bag of marbles. -/
theorem consecutive_draw_probability
  (red : ℕ) (blue : ℕ) (green : ℕ)
  (h_red : red = 5)
  (h_blue : blue = 4)
  (h_green : green = 6)
  : (red : ℚ) / (red + blue + green) * (blue : ℚ) / (red + blue + green - 1) = 2 / 21 := by
  sorry

end consecutive_draw_probability_l206_20625


namespace yanna_afternoon_biscuits_l206_20674

/-- The number of butter cookies Yanna baked in the afternoon -/
def afternoon_butter_cookies : ℕ := 10

/-- The difference between biscuits and butter cookies baked in the afternoon -/
def biscuit_cookie_difference : ℕ := 30

/-- The number of biscuits Yanna baked in the afternoon -/
def afternoon_biscuits : ℕ := afternoon_butter_cookies + biscuit_cookie_difference

theorem yanna_afternoon_biscuits : afternoon_biscuits = 40 := by
  sorry

end yanna_afternoon_biscuits_l206_20674


namespace pens_left_in_jar_l206_20601

/-- The number of pens left in a jar after removing some pens -/
def pens_left (initial_blue initial_black initial_red removed_blue removed_black : ℕ) : ℕ :=
  (initial_blue - removed_blue) + (initial_black - removed_black) + initial_red

/-- Theorem stating the number of pens left in the jar -/
theorem pens_left_in_jar : pens_left 9 21 6 4 7 = 25 := by
  sorry

end pens_left_in_jar_l206_20601


namespace hypotenuse_value_l206_20605

-- Define a right triangle with sides 3, 5, and x (hypotenuse)
def right_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x^2 = 3^2 + 5^2

-- Theorem statement
theorem hypotenuse_value :
  ∃ x : ℝ, right_triangle x ∧ x = Real.sqrt 34 :=
by sorry

end hypotenuse_value_l206_20605


namespace monotonic_function_property_l206_20685

/-- A monotonic function f: ℝ → ℝ satisfying f[f(x) - 3^x] = 4 for all x ∈ ℝ has f(2) = 10 -/
theorem monotonic_function_property (f : ℝ → ℝ) 
  (h_monotonic : Monotone f)
  (h_property : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f 2 = 10 := by sorry

end monotonic_function_property_l206_20685


namespace smallest_multiple_of_twelve_power_l206_20684

theorem smallest_multiple_of_twelve_power (k : ℕ) : 
  (3^k - k^3 = 1) → (∀ n : ℕ, n > 0 ∧ 12^k ∣ n → n ≥ 144) :=
by
  sorry

end smallest_multiple_of_twelve_power_l206_20684


namespace sara_movie_tickets_l206_20665

-- Define the constants
def ticket_cost : ℚ := 10.62
def rental_cost : ℚ := 1.59
def purchase_cost : ℚ := 13.95
def total_spent : ℚ := 36.78

-- Define the theorem
theorem sara_movie_tickets :
  ∃ (n : ℕ), n * ticket_cost + rental_cost + purchase_cost = total_spent ∧ n = 2 :=
sorry

end sara_movie_tickets_l206_20665


namespace cost_per_topping_is_two_l206_20631

/-- Represents the cost of a pizza order with toppings and tip --/
def pizza_order_cost (large_pizza_cost : ℝ) (num_pizzas : ℕ) (toppings_per_pizza : ℕ) 
  (tip_percentage : ℝ) (topping_cost : ℝ) : ℝ :=
  let base_cost := large_pizza_cost * num_pizzas
  let total_toppings := num_pizzas * toppings_per_pizza
  let toppings_cost := total_toppings * topping_cost
  let subtotal := base_cost + toppings_cost
  let tip := subtotal * tip_percentage
  subtotal + tip

/-- The cost per topping is $2 --/
theorem cost_per_topping_is_two :
  ∃ (topping_cost : ℝ),
    pizza_order_cost 14 2 3 0.25 topping_cost = 50 ∧ 
    topping_cost = 2 :=
by sorry

end cost_per_topping_is_two_l206_20631


namespace bamboo_break_height_l206_20668

theorem bamboo_break_height (total_height : ℝ) (fall_distance : ℝ) (break_height : ℝ) : 
  total_height = 9 → 
  fall_distance = 3 → 
  break_height^2 + fall_distance^2 = (total_height - break_height)^2 →
  break_height = 4 := by
sorry

end bamboo_break_height_l206_20668
