import Mathlib

namespace triangle_problem_l646_64669

/-- Given a triangle ABC with vertex A at (5,1), altitude CH from AB with equation x-2y-5=0,
    and median BM from AC with equation 2x-y-1=0, prove the coordinates of B and the equation
    of the perpendicular bisector of BC. -/
theorem triangle_problem (B : ℝ × ℝ) (perpBisectorBC : ℝ → ℝ → ℝ) : 
  let A : ℝ × ℝ := (5, 1)
  let altitude_CH (x y : ℝ) := x - 2*y - 5 = 0
  let median_BM (x y : ℝ) := 2*x - y - 1 = 0
  B = (3, 5) ∧ 
  (∀ x y, perpBisectorBC x y = 0 ↔ 21*x + 24*y + 43 = 0) := by
  sorry

end triangle_problem_l646_64669


namespace two_and_three_digit_number_product_l646_64657

theorem two_and_three_digit_number_product : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 
  100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 9 * x * y ∧
  x + y = 126 := by
sorry

end two_and_three_digit_number_product_l646_64657


namespace sticks_difference_l646_64602

theorem sticks_difference (picked_up left : ℕ) 
  (h1 : picked_up = 14)
  (h2 : left = 4) :
  picked_up - left = 10 := by
  sorry

end sticks_difference_l646_64602


namespace weighted_average_calculation_l646_64642

theorem weighted_average_calculation (math_score math_weight history_score history_weight third_weight target_average : ℚ)
  (h1 : math_score = 72 / 100)
  (h2 : math_weight = 50 / 100)
  (h3 : history_score = 84 / 100)
  (h4 : history_weight = 30 / 100)
  (h5 : third_weight = 20 / 100)
  (h6 : target_average = 75 / 100)
  (h7 : math_weight + history_weight + third_weight ≤ 1) :
  ∃ (third_score fourth_weight : ℚ),
    third_score = 69 / 100 ∧
    fourth_weight = 0 ∧
    math_weight + history_weight + third_weight + fourth_weight = 1 ∧
    math_score * math_weight + history_score * history_weight + third_score * third_weight = target_average :=
by sorry

end weighted_average_calculation_l646_64642


namespace afternoon_bundles_burned_eq_three_l646_64677

/-- Given the number of wood bundles burned in the morning, at the start of the day, and at the end of the day, 
    calculate the number of wood bundles burned in the afternoon. -/
def afternoon_bundles_burned (morning_burned start_of_day end_of_day : ℕ) : ℕ :=
  (start_of_day - end_of_day) - morning_burned

theorem afternoon_bundles_burned_eq_three : 
  afternoon_bundles_burned 4 10 3 = 3 := by
  sorry

end afternoon_bundles_burned_eq_three_l646_64677


namespace problem_1_problem_2_problem_3_problem_4_l646_64676

-- Problem 1
theorem problem_1 : 18 + (-12) + (-18) = -12 := by sorry

-- Problem 2
theorem problem_2 : (1 + 3 / 7) + (-(2 + 1 / 3)) + (2 + 4 / 7) + (-(1 + 2 / 3)) = 0 := by sorry

-- Problem 3
theorem problem_3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := by sorry

-- Problem 4
theorem problem_4 : -(1 ^ 2023) - ((-2) ^ 3) - ((-2) * (-3)) = 1 := by sorry

end problem_1_problem_2_problem_3_problem_4_l646_64676


namespace janet_snowball_percentage_l646_64687

/-- The number of snowballs Janet made -/
def janet_snowballs : ℕ := 50

/-- The number of snowballs Janet's brother made -/
def brother_snowballs : ℕ := 150

/-- The total number of snowballs made -/
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

/-- The percentage of snowballs Janet made -/
def janet_percentage : ℚ := (janet_snowballs : ℚ) / (total_snowballs : ℚ) * 100

theorem janet_snowball_percentage : janet_percentage = 25 := by
  sorry

end janet_snowball_percentage_l646_64687


namespace union_of_A_and_B_l646_64613

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 4} := by sorry

end union_of_A_and_B_l646_64613


namespace isabellas_hair_length_l646_64638

/-- Calculates the final length of Isabella's hair after a haircut -/
def hair_length_after_cut (initial_length cut_length : ℕ) : ℕ :=
  initial_length - cut_length

/-- Theorem stating that Isabella's hair length after the cut is 9 inches -/
theorem isabellas_hair_length :
  hair_length_after_cut 18 9 = 9 := by
  sorry

end isabellas_hair_length_l646_64638


namespace first_dog_consumption_l646_64653

/-- Represents the weekly food consumption of three dogs -/
structure DogFoodConsumption where
  first_dog : ℝ
  second_dog : ℝ
  third_dog : ℝ

/-- The total weekly food consumption of the three dogs -/
def total_consumption (d : DogFoodConsumption) : ℝ :=
  d.first_dog + d.second_dog + d.third_dog

theorem first_dog_consumption :
  ∃ (d : DogFoodConsumption),
    total_consumption d = 15 ∧
    d.second_dog = 2 * d.first_dog ∧
    d.third_dog = 6 ∧
    d.first_dog = 3 := by
  sorry

end first_dog_consumption_l646_64653


namespace rectangle_length_equals_two_l646_64615

theorem rectangle_length_equals_two (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 4 →
  rect_width = 8 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 2 := by
sorry

end rectangle_length_equals_two_l646_64615


namespace rohan_salary_l646_64681

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 5000

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 1000

theorem rohan_salary :
  monthly_salary * (1 - (food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage) / 100) = savings :=
by sorry

end rohan_salary_l646_64681


namespace adam_laundry_theorem_l646_64654

/-- The number of loads Adam has already washed -/
def washed_loads : ℕ := 8

/-- The number of loads Adam still needs to wash -/
def remaining_loads : ℕ := 6

/-- The total number of loads Adam has to wash -/
def total_loads : ℕ := washed_loads + remaining_loads

theorem adam_laundry_theorem : total_loads = 14 := by sorry

end adam_laundry_theorem_l646_64654


namespace solve_system_l646_64617

theorem solve_system (x y z : ℚ) 
  (eq1 : x - y - z = 8)
  (eq2 : x + y + z = 20)
  (eq3 : x - y + 2*z = 16) :
  z = 8/3 := by
sorry

end solve_system_l646_64617


namespace expression_greater_than_e_l646_64612

theorem expression_greater_than_e (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  Real.exp y - 8/x > Real.exp 1 :=
sorry

end expression_greater_than_e_l646_64612


namespace point_outside_circle_l646_64600

theorem point_outside_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0 → (a - x)^2 + (2 - y)^2 > 0) ↔ 
  (2 < a ∧ a < 9/4) :=
sorry

end point_outside_circle_l646_64600


namespace circle_area_equals_circumference_squared_l646_64641

theorem circle_area_equals_circumference_squared : 
  ∀ (r : ℝ), r > 0 → 2 * (π * r^2 / 2) = (2 * π * r)^2 / (4 * π) := by
  sorry

end circle_area_equals_circumference_squared_l646_64641


namespace total_comics_in_box_l646_64695

-- Define the problem parameters
def pages_per_comic : ℕ := 25
def found_pages : ℕ := 150
def untorn_comics : ℕ := 5

-- State the theorem
theorem total_comics_in_box : 
  (found_pages / pages_per_comic) + untorn_comics = 11 := by
  sorry

end total_comics_in_box_l646_64695


namespace min_sum_cube_relation_l646_64690

theorem min_sum_cube_relation (m n : ℕ+) (h : 108 * m = n ^ 3) :
  ∃ (m₀ n₀ : ℕ+), 108 * m₀ = n₀ ^ 3 ∧ m₀ + n₀ = 8 ∧ ∀ (m' n' : ℕ+), 108 * m' = n' ^ 3 → m' + n' ≥ 8 := by
  sorry

end min_sum_cube_relation_l646_64690


namespace inscribed_hexagon_area_l646_64662

/-- The area of a regular hexagon inscribed in a circle -/
theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 100 * Real.pi) :
  let r := (circle_area / Real.pi).sqrt
  let hexagon_area := 6 * (r^2 * Real.sqrt 3 / 4)
  hexagon_area = 150 * Real.sqrt 3 := by
  sorry

end inscribed_hexagon_area_l646_64662


namespace second_month_sale_l646_64618

def sales_data : List ℕ := [8435, 8855, 9230, 8562, 6991]
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem second_month_sale :
  let total_sale := average_sale * num_months
  let known_sales_sum := sales_data.sum
  let second_month_sale := total_sale - known_sales_sum
  second_month_sale = 8927 := by sorry

end second_month_sale_l646_64618


namespace intersection_of_M_and_N_l646_64689

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 5}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(4, -1)} := by
  sorry

end intersection_of_M_and_N_l646_64689


namespace length_XX₁_l646_64660

/-- Configuration of two right triangles with angle bisectors -/
structure TriangleConfig where
  -- Triangle DEF
  DE : ℝ
  DF : ℝ
  hDE : DE = 13
  hDF : DF = 5
  hDEF_right : DE^2 = DF^2 + EF^2
  
  -- D₁ is on EF such that ∠FDD₁ = ∠EDD₁
  D₁F : ℝ
  D₁E : ℝ
  hD₁_on_EF : D₁F + D₁E = EF
  hD₁_bisector : D₁F / D₁E = DF / EF
  
  -- Triangle XYZ
  XY : ℝ
  XZ : ℝ
  hXY : XY = D₁E
  hXZ : XZ = D₁F
  hXYZ_right : XY^2 = XZ^2 + YZ^2
  
  -- X₁ is on YZ such that ∠ZXX₁ = ∠YXX₁
  X₁Z : ℝ
  X₁Y : ℝ
  hX₁_on_YZ : X₁Z + X₁Y = YZ
  hX₁_bisector : X₁Z / X₁Y = XZ / XY

/-- The length of XX₁ in the given configuration is 20/17 -/
theorem length_XX₁ (config : TriangleConfig) : X₁Z = 20/17 := by
  sorry

end length_XX₁_l646_64660


namespace smallest_max_sum_l646_64621

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 4020) : 
  (∃ (a' b' c' d' e' f' : ℕ+), 
    a' + b' + c' + d' + e' + f' = 4020 ∧ 
    max (a' + b') (max (b' + c') (max (c' + d') (max (d' + e') (e' + f')))) = 805) ∧
  (∀ (a'' b'' c'' d'' e'' f'' : ℕ+),
    a'' + b'' + c'' + d'' + e'' + f'' = 4020 →
    max (a'' + b'') (max (b'' + c'') (max (c'' + d'') (max (d'' + e'') (e'' + f'')))) ≥ 805) :=
by sorry

end smallest_max_sum_l646_64621


namespace simplify_trig_expression_l646_64655

theorem simplify_trig_expression :
  let x : Real := 10 * π / 180  -- 10 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.cos (17 * x) ^ 2)) = Real.tan x :=
by sorry

end simplify_trig_expression_l646_64655


namespace kayla_waiting_time_l646_64671

/-- The number of years Kayla needs to wait before reaching the minimum driving age -/
def years_until_driving (minimum_age : ℕ) (kimiko_age : ℕ) : ℕ :=
  minimum_age - kimiko_age / 2

/-- Proof that Kayla needs to wait 5 years before she can start driving -/
theorem kayla_waiting_time :
  years_until_driving 18 26 = 5 := by
  sorry

end kayla_waiting_time_l646_64671


namespace complement_intersection_empty_l646_64601

def I : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_empty :
  (I \ M) ∩ (I \ N) = ∅ :=
by
  sorry

end complement_intersection_empty_l646_64601


namespace russian_in_top_three_l646_64675

structure ChessTournament where
  total_players : Nat
  russian_players : Nat
  foreign_players : Nat
  games_per_pair : Nat
  total_points : Nat
  russian_points : Nat
  foreign_points : Nat

def valid_tournament (t : ChessTournament) : Prop :=
  t.total_players = 11 ∧
  t.russian_players = 4 ∧
  t.foreign_players = 7 ∧
  t.games_per_pair = 2 ∧
  t.total_points = t.total_players * (t.total_players - 1) ∧
  t.russian_points = t.foreign_points ∧
  t.russian_points + t.foreign_points = t.total_points

theorem russian_in_top_three (t : ChessTournament) (h : valid_tournament t) :
  ∃ (top_three : Finset Nat) (russian : Nat),
    top_three.card = 3 ∧
    russian ∈ top_three ∧
    russian ≤ t.russian_players :=
  sorry

end russian_in_top_three_l646_64675


namespace square_sum_simplification_l646_64652

theorem square_sum_simplification (a : ℝ) : a^2 + 2*a^2 = 3*a^2 := by
  sorry

end square_sum_simplification_l646_64652


namespace sector_perimeter_l646_64668

/-- Given a sector with central angle 54° and radius 20 cm, its perimeter is (6π + 40) cm -/
theorem sector_perimeter (θ : Real) (r : Real) : 
  θ = 54 * Real.pi / 180 → r = 20 → 
  (θ * r) + 2 * r = 6 * Real.pi + 40 := by
  sorry

end sector_perimeter_l646_64668


namespace triangular_intersection_solids_l646_64679

-- Define the types for geometric solids and plane
inductive GeometricSolid
| Cone
| Cylinder
| Pyramid
| Cube

structure Plane

-- Define the intersection of a plane and a geometric solid
def Intersection (p : Plane) (s : GeometricSolid) : Set (ℝ × ℝ × ℝ) := sorry

-- Define what it means for an intersection to be triangular
def IsTriangularIntersection (i : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem triangular_intersection_solids 
  (p : Plane) (s : GeometricSolid) 
  (h : IsTriangularIntersection (Intersection p s)) : 
  s = GeometricSolid.Cone ∨ s = GeometricSolid.Pyramid ∨ s = GeometricSolid.Cube := by
  sorry


end triangular_intersection_solids_l646_64679


namespace window_installation_time_l646_64696

theorem window_installation_time (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) 
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_per_window = 4) : 
  (total_windows - installed_windows) * time_per_window = 36 := by
  sorry

end window_installation_time_l646_64696


namespace exists_solution_with_y_seven_l646_64626

theorem exists_solution_with_y_seven :
  ∃ (x y z t : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    x + y + z + t = 10 ∧
    y = 7 := by
  sorry

end exists_solution_with_y_seven_l646_64626


namespace two_numbers_sum_product_l646_64637

theorem two_numbers_sum_product : ∃ x y : ℝ, x + y = 20 ∧ x * y = 96 ∧ ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := by
  sorry

end two_numbers_sum_product_l646_64637


namespace line_slope_intercept_product_l646_64648

theorem line_slope_intercept_product (m b : ℚ) : 
  m = -3/4 → b = 3/2 → m * b < -1 := by sorry

end line_slope_intercept_product_l646_64648


namespace second_largest_is_eleven_l646_64646

def numbers : Finset ℕ := {10, 11, 12}

theorem second_largest_is_eleven :
  ∃ (a b : ℕ), a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧
  (∀ x ∈ numbers, x ≤ a) ∧
  (∃ y ∈ numbers, y > 11) ∧
  (∀ z ∈ numbers, z > 11 → z ≥ a) :=
sorry

end second_largest_is_eleven_l646_64646


namespace banana_groups_l646_64632

theorem banana_groups (total_bananas : ℕ) (group_size : ℕ) (h1 : total_bananas = 180) (h2 : group_size = 18) :
  total_bananas / group_size = 10 := by
  sorry

end banana_groups_l646_64632


namespace select_four_shoes_with_one_match_l646_64608

/-- The number of ways to select four shoes from four different pairs, such that exactly one pair matches. -/
def selectFourShoesWithOneMatch : ℕ := 48

/-- The number of different pairs of shoes. -/
def numPairs : ℕ := 4

/-- The number of shoes to be selected. -/
def shoesToSelect : ℕ := 4

theorem select_four_shoes_with_one_match :
  selectFourShoesWithOneMatch = 
    numPairs * (Nat.choose (numPairs - 1) 2) * 2^2 := by
  sorry

end select_four_shoes_with_one_match_l646_64608


namespace no_solution_exists_l646_64697

theorem no_solution_exists : ¬ ∃ (n m r : ℕ), 
  n ≥ 1 ∧ m ≥ 1 ∧ r ≥ 1 ∧ n^5 + 49^m = 1221^r := by
  sorry

end no_solution_exists_l646_64697


namespace simplest_fraction_of_decimal_l646_64645

theorem simplest_fraction_of_decimal (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 0.428125 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.428125 → a * d ≤ b * c →
  a = 137 ∧ b = 320 := by
sorry

end simplest_fraction_of_decimal_l646_64645


namespace four_digit_sum_with_reverse_l646_64682

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Returns the reversed digits of a four-digit number -/
def reverseDigits (x : FourDigitNumber) : FourDigitNumber :=
  sorry

/-- The sum of a number and its reverse -/
def sumWithReverse (x : FourDigitNumber) : ℕ :=
  x.val + (reverseDigits x).val

theorem four_digit_sum_with_reverse (x : FourDigitNumber) :
  x.val % 10 ≠ 0 →
  (sumWithReverse x) % 100 = 0 →
  sumWithReverse x = 11000 := by
  sorry

end four_digit_sum_with_reverse_l646_64682


namespace cubic_function_b_value_l646_64606

/-- A cubic function f(x) = x³ + bx² + cx + d passing through (-1, 0), (1, 0), and (0, 2) has b = -2 -/
theorem cubic_function_b_value (b c d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x + d
  (f (-1) = 0) → (f 1 = 0) → (f 0 = 2) → b = -2 := by
  sorry

end cubic_function_b_value_l646_64606


namespace quadratic_real_root_condition_l646_64691

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end quadratic_real_root_condition_l646_64691


namespace same_color_marble_probability_same_color_marble_probability_is_one_twentieth_l646_64627

/-- The probability that all 3 girls select the same colored marble from a bag with 3 white and 3 black marbles -/
theorem same_color_marble_probability : ℚ :=
  let total_marbles : ℕ := 6
  let white_marbles : ℕ := 3
  let black_marbles : ℕ := 3
  let girls : ℕ := 3
  let prob_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  let prob_all_black : ℚ := (black_marbles / total_marbles) * ((black_marbles - 1) / (total_marbles - 1)) * ((black_marbles - 2) / (total_marbles - 2))
  prob_all_white + prob_all_black

theorem same_color_marble_probability_is_one_twentieth : same_color_marble_probability = 1 / 20 := by
  sorry

end same_color_marble_probability_same_color_marble_probability_is_one_twentieth_l646_64627


namespace prob_more_heads_ten_coins_l646_64639

/-- The number of coins being flipped -/
def n : ℕ := 10

/-- The probability of getting heads on a single coin flip -/
def p : ℚ := 1/2

/-- The probability of getting more heads than tails when flipping n fair coins -/
def prob_more_heads (n : ℕ) (p : ℚ) : ℚ :=
  1/2 * (1 - (n.choose (n/2)) / (2^n))

theorem prob_more_heads_ten_coins :
  prob_more_heads n p = 193/512 := by
  sorry

#eval prob_more_heads n p

end prob_more_heads_ten_coins_l646_64639


namespace difference_of_squares_l646_64640

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end difference_of_squares_l646_64640


namespace minimum_total_tests_l646_64644

/-- Represents the test data for a student -/
structure StudentData where
  name : String
  numTests : ℕ
  avgScore : ℕ
  totalScore : ℕ

/-- The problem statement -/
theorem minimum_total_tests (k m r : StudentData) : 
  k.name = "Michael K" →
  m.name = "Michael M" →
  r.name = "Michael R" →
  k.avgScore = 90 →
  m.avgScore = 91 →
  r.avgScore = 92 →
  k.numTests > m.numTests →
  m.numTests > r.numTests →
  m.totalScore > r.totalScore →
  r.totalScore > k.totalScore →
  k.totalScore = k.numTests * k.avgScore →
  m.totalScore = m.numTests * m.avgScore →
  r.totalScore = r.numTests * r.avgScore →
  k.numTests + m.numTests + r.numTests ≥ 413 :=
by sorry

end minimum_total_tests_l646_64644


namespace point_b_value_l646_64664

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moving right on a number line -/
def moveRight (p : Point) (units : ℤ) : Point :=
  ⟨p.value + units⟩

theorem point_b_value (a b : Point) (h1 : a.value = -3) (h2 : b = moveRight a 4) :
  b.value = 1 := by
  sorry

end point_b_value_l646_64664


namespace envelope_addressing_problem_l646_64629

/-- A manufacturer's envelope addressing problem -/
theorem envelope_addressing_problem 
  (initial_machine : ℝ) 
  (first_added_machine : ℝ) 
  (combined_initial_and_first : ℝ) 
  (all_three_machines : ℝ) 
  (h1 : initial_machine = 600 / 10)
  (h2 : first_added_machine = 600 / 5)
  (h3 : combined_initial_and_first = 600 / 3)
  (h4 : all_three_machines = 600 / 1) :
  (600 / (all_three_machines - initial_machine - first_added_machine)) = 10 / 7 := by
  sorry

#check envelope_addressing_problem

end envelope_addressing_problem_l646_64629


namespace train_length_proof_l646_64619

/-- Given a train with constant speed that crosses two platforms of different lengths,
    prove that the length of the train is 110 meters. -/
theorem train_length_proof (speed : ℝ) (length : ℝ) :
  speed > 0 →
  speed * 15 = length + 160 →
  speed * 20 = length + 250 →
  length = 110 :=
by sorry

end train_length_proof_l646_64619


namespace photographer_arrangement_exists_l646_64670

-- Define a type for photographers
def Photographer := Fin 6

-- Define a type for positions in the plane
def Position := ℝ × ℝ

-- Define a function to check if a photographer is between two others
def isBetween (p₁ p₂ p₃ : Position) : Prop := sorry

-- Define a function to check if two photographers can see each other
def canSee (positions : Photographer → Position) (p₁ p₂ : Photographer) : Prop :=
  ∀ p₃, p₃ ≠ p₁ ∧ p₃ ≠ p₂ → ¬ isBetween (positions p₁) (positions p₃) (positions p₂)

-- State the theorem
theorem photographer_arrangement_exists :
  ∃ (positions : Photographer → Position),
    ∀ p, (∃! (s : Finset Photographer), s.card = 4 ∧ ∀ p' ∈ s, canSee positions p p') :=
sorry

end photographer_arrangement_exists_l646_64670


namespace sum_of_coefficients_equals_one_l646_64630

theorem sum_of_coefficients_equals_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + 2*x)^7 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + 
                      a₄*(1-x)^4 + a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by
  sorry

end sum_of_coefficients_equals_one_l646_64630


namespace binomial_18_choose_6_l646_64683

theorem binomial_18_choose_6 : Nat.choose 18 6 = 13260 := by
  sorry

end binomial_18_choose_6_l646_64683


namespace carrots_planted_per_hour_l646_64604

theorem carrots_planted_per_hour 
  (rows : ℕ) 
  (plants_per_row : ℕ) 
  (total_hours : ℕ) 
  (h1 : rows = 400) 
  (h2 : plants_per_row = 300) 
  (h3 : total_hours = 20) : 
  (rows * plants_per_row) / total_hours = 6000 := by
sorry

end carrots_planted_per_hour_l646_64604


namespace problems_per_page_l646_64667

theorem problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 110) 
  (h2 : finished_problems = 47) 
  (h3 : remaining_pages = 7) 
  (h4 : finished_problems < total_problems) : 
  (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end problems_per_page_l646_64667


namespace inverse_variation_problem_l646_64647

/-- Given that quantities a and b vary inversely, prove that b = 0.375 when a = 1600 -/
theorem inverse_variation_problem (a b : ℝ) (h1 : a * b = 800 * 0.5) 
  (h2 : (2 * 800) * (b / 2) = a * b + 200) : 
  (a = 1600) → (b = 0.375) := by
  sorry

end inverse_variation_problem_l646_64647


namespace greatest_q_minus_r_l646_64623

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  1043 = 23 * q + r ∧ 
  r < 23 ∧
  ∀ (q' r' : ℕ), 1043 = 23 * q' + r' ∧ r' < 23 → q' - r' ≤ q - r :=
by sorry

end greatest_q_minus_r_l646_64623


namespace polynomial_root_relation_l646_64636

/-- Given two polynomial equations:
    1. x^2 - ax + b = 0 with roots α and β
    2. x^2 - px + q = 0 with roots α^2 + β^2 and αβ
    Prove that p = a^2 - b -/
theorem polynomial_root_relation (a b p q α β : ℝ) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = α ∨ x = β) →
  (∀ x, x^2 - p*x + q = 0 ↔ x = α^2 + β^2 ∨ x = α*β) →
  p = a^2 - b := by
sorry

end polynomial_root_relation_l646_64636


namespace intersection_coordinate_sum_zero_l646_64616

/-- Two lines in the coordinate plane -/
structure Line where
  slope : ℝ
  intercept : ℝ
  is_x_intercept : Bool

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- Theorem: The sum of coordinates of the intersection point is 0 -/
theorem intersection_coordinate_sum_zero :
  let line_a : Line := ⟨-1, 2, true⟩
  let line_b : Line := ⟨5, -10, false⟩
  let (a, b) := intersection line_a line_b
  a + b = 0 := by sorry

end intersection_coordinate_sum_zero_l646_64616


namespace volume_circumscribed_sphere_folded_rectangle_l646_64699

/-- The volume of the circumscribed sphere of a tetrahedron formed by folding a rectangle --/
theorem volume_circumscribed_sphere_folded_rectangle (a b : ℝ) (ha : a = 4) (hb : b = 3) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = (125/6) * Real.pi := by sorry

end volume_circumscribed_sphere_folded_rectangle_l646_64699


namespace horner_method_v₃_l646_64643

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

def horner_v₃ (x v v₁ : ℝ) : ℝ :=
  let v₂ := v₁ * x - 4
  v₂ * x + 3

theorem horner_method_v₃ :
  let x : ℝ := 5
  let v : ℝ := 2
  let v₁ : ℝ := 5
  horner_v₃ x v v₁ = 108 := by sorry

end horner_method_v₃_l646_64643


namespace triangle_proof_l646_64661

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) (P : ℝ × ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = Real.sqrt 5 ∧
  c * Real.sin A = Real.sqrt 2 * Real.sin ((A + B) / 2) ∧
  Real.sqrt 5 = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
  Real.sqrt 5 = Real.sqrt ((1 - P.1)^2 + (0 - P.2)^2) ∧
  1 = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
  3 * π / 4 = Real.arccos ((P.1 * 1 + P.2 * 0) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt 5)) →
  C = π / 2 ∧
  Real.sqrt ((1 - P.1)^2 + (0 - P.2)^2) = Real.sqrt ((P.1 - 1)^2 + P.2^2) :=
by sorry

end triangle_proof_l646_64661


namespace adjacent_even_sum_l646_64656

theorem adjacent_even_sum (numbers : Vector ℕ 2019) : 
  ∃ i : Fin 2019, Even ((numbers.get i) + (numbers.get ((i + 1) % 2019))) :=
sorry

end adjacent_even_sum_l646_64656


namespace consecutive_four_product_plus_one_is_square_l646_64698

theorem consecutive_four_product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end consecutive_four_product_plus_one_is_square_l646_64698


namespace zigzag_angle_theorem_l646_64684

/-- In a zigzag inside a rectangle, if certain angles are given, prove that angle CDE (θ) equals 11 degrees. -/
theorem zigzag_angle_theorem (ACB FEG DCE DEC : ℝ) (h1 : ACB = 80) (h2 : FEG = 64) (h3 : DCE = 86) (h4 : DEC = 83) :
  180 - DCE - DEC = 11 :=
sorry

end zigzag_angle_theorem_l646_64684


namespace clock_hand_speed_ratio_l646_64603

/-- Represents the number of degrees in a full rotation of a clock face. -/
def clock_degrees : ℕ := 360

/-- Represents the number of minutes it takes for the minute hand to complete a full rotation. -/
def minute_hand_period : ℕ := 60

/-- Represents the number of hours it takes for the hour hand to complete a full rotation. -/
def hour_hand_period : ℕ := 12

/-- Theorem stating that the ratio of the speeds of the hour hand to the minute hand is 2:24. -/
theorem clock_hand_speed_ratio :
  (clock_degrees / (hour_hand_period * minute_hand_period) : ℚ) / 
  (clock_degrees / minute_hand_period : ℚ) = 2 / 24 := by
  sorry

end clock_hand_speed_ratio_l646_64603


namespace min_value_sum_products_l646_64649

theorem min_value_sum_products (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = a * b * c) (h2 : a + b + c = a^3) :
  ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = x * y * z ∧ x + y + z = x^3 →
  x * y + y * z + z * x ≥ 9 :=
by sorry

end min_value_sum_products_l646_64649


namespace pyramid_lateral_surface_area_l646_64614

/-- Given a cylinder with height H and base radius R, and a pyramid inside the cylinder
    with its height coinciding with the cylinder's slant height, and its base being an
    isosceles triangle ABC inscribed in the cylinder's base with ∠A = 120°,
    the lateral surface area of the pyramid is (R/4) * (4H + √(3R² + 12H²)). -/
theorem pyramid_lateral_surface_area 
  (H R : ℝ) 
  (H_pos : H > 0) 
  (R_pos : R > 0) : 
  ∃ (pyramid_area : ℝ), 
    pyramid_area = (R / 4) * (4 * H + Real.sqrt (3 * R^2 + 12 * H^2)) :=
by sorry

end pyramid_lateral_surface_area_l646_64614


namespace inequality_solution_range_l646_64673

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, abs (x + 2) - abs (x + 3) > m) → m < 1 := by
  sorry

end inequality_solution_range_l646_64673


namespace spending_problem_l646_64624

theorem spending_problem (initial_amount : ℚ) : 
  (initial_amount * (5/7) * (10/13) * (4/5) * (8/11) = 5400) → 
  initial_amount = 16890 := by
  sorry

end spending_problem_l646_64624


namespace regular_polygon_sides_l646_64625

theorem regular_polygon_sides (exterior_angle : ℝ) :
  exterior_angle = 40 →
  (∃ n : ℕ, n * exterior_angle = 360 ∧ n = 9) :=
by sorry

end regular_polygon_sides_l646_64625


namespace michaela_needs_20_oranges_l646_64658

/-- The number of oranges Michaela needs to eat until she gets full. -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to eat until she gets full. -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked. -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after both ate until full. -/
def remaining_oranges : ℕ := 30

/-- Proves that Michaela needs 20 oranges to get full given the conditions. -/
theorem michaela_needs_20_oranges : 
  michaela_oranges = 20 ∧ 
  cassandra_oranges = 2 * michaela_oranges ∧
  total_oranges = 90 ∧
  remaining_oranges = 30 ∧
  total_oranges = michaela_oranges + cassandra_oranges + remaining_oranges :=
by sorry

end michaela_needs_20_oranges_l646_64658


namespace union_of_M_and_N_l646_64663

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end union_of_M_and_N_l646_64663


namespace ceiling_floor_difference_l646_64665

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 :=
by sorry

end ceiling_floor_difference_l646_64665


namespace min_value_expression_l646_64610

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b + 3 = b) :
  (1 / a + 2 * b) ≥ 8 + 4 * Real.sqrt 3 :=
by sorry

end min_value_expression_l646_64610


namespace election_vote_difference_l646_64633

theorem election_vote_difference (total_votes : ℕ) (invalid_votes : ℕ) (losing_percentage : ℚ) :
  total_votes = 12600 →
  invalid_votes = 100 →
  losing_percentage = 30 / 100 →
  ∃ (valid_votes winning_votes losing_votes : ℕ),
    valid_votes = total_votes - invalid_votes ∧
    losing_votes = (losing_percentage * valid_votes).floor ∧
    winning_votes = valid_votes - losing_votes ∧
    winning_votes - losing_votes = 5000 := by
  sorry

#check election_vote_difference

end election_vote_difference_l646_64633


namespace min_value_on_interval_l646_64688

def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ 
  (∀ y ∈ Set.Icc 0 3, f y ≥ f x) ∧
  f x = -15 :=
sorry

end min_value_on_interval_l646_64688


namespace tan_is_periodic_l646_64631

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define the property of being periodic
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- State the theorem
theorem tan_is_periodic : is_periodic tan π := by
  sorry

end tan_is_periodic_l646_64631


namespace function_inequality_l646_64674

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 2) * (deriv^[2] f x) > 0) : 
  f 2 < f 0 ∧ f 0 < f (-3) := by
  sorry

end function_inequality_l646_64674


namespace integral_inequality_l646_64634

open Real MeasureTheory

theorem integral_inequality : 
  ∫ x in (1:ℝ)..2, (1 / x) < ∫ x in (1:ℝ)..2, x ∧ ∫ x in (1:ℝ)..2, x < ∫ x in (1:ℝ)..2, exp x := by
  sorry

end integral_inequality_l646_64634


namespace reciprocal_of_negative_fraction_l646_64609

theorem reciprocal_of_negative_fraction :
  ((-1 : ℚ) / 2023)⁻¹ = -2023 := by
  sorry

end reciprocal_of_negative_fraction_l646_64609


namespace data_set_range_is_67_l646_64650

-- Define a structure for our data set
structure DataSet where
  points : List ℝ
  min_value : ℝ
  max_value : ℝ
  h_min : min_value ∈ points
  h_max : max_value ∈ points
  h_lower_bound : ∀ x ∈ points, min_value ≤ x
  h_upper_bound : ∀ x ∈ points, x ≤ max_value

-- Define the range of a data set
def range (d : DataSet) : ℝ := d.max_value - d.min_value

-- Theorem statement
theorem data_set_range_is_67 (d : DataSet) 
  (h_min : d.min_value = 31)
  (h_max : d.max_value = 98) : 
  range d = 67 := by
  sorry

end data_set_range_is_67_l646_64650


namespace two_digit_number_sum_l646_64680

/-- Given a two-digit number n = 10a + b, where a and b are single digits,
    if the difference between n and its reverse is 7 times the sum of its digits,
    then the sum of n and its reverse is 99. -/
theorem two_digit_number_sum (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) (ha_pos : a > 0) :
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end two_digit_number_sum_l646_64680


namespace specific_trapezoid_area_l646_64635

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  -- Length of the top base
  AB : ℝ
  -- Distance from point D to point N on the bottom base
  DN : ℝ
  -- Radius of the inscribed circle
  r : ℝ

/-- The area of a trapezoid with an inscribed circle -/
def trapezoidArea (t : InscribedCircleTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 27 -/
theorem specific_trapezoid_area :
  ∀ (t : InscribedCircleTrapezoid),
    t.AB = 12 ∧ t.DN = 1 ∧ t.r = 2 →
    trapezoidArea t = 27 :=
  sorry

end specific_trapezoid_area_l646_64635


namespace gunther_typing_words_l646_64659

-- Define the typing speeds and durations
def first_phase_speed : ℕ := 160
def first_phase_duration : ℕ := 2 * 60
def second_phase_speed : ℕ := 200
def second_phase_duration : ℕ := 3 * 60
def third_phase_speed : ℕ := 140
def third_phase_duration : ℕ := 4 * 60

-- Define the interval duration (in minutes)
def interval_duration : ℕ := 3

-- Function to calculate words typed in a phase
def words_in_phase (speed : ℕ) (duration : ℕ) : ℕ :=
  (duration / interval_duration) * speed

-- Theorem statement
theorem gunther_typing_words :
  words_in_phase first_phase_speed first_phase_duration +
  words_in_phase second_phase_speed second_phase_duration +
  words_in_phase third_phase_speed third_phase_duration = 29600 := by
  sorry

end gunther_typing_words_l646_64659


namespace function_is_zero_l646_64678

def is_logarithmic_property (f : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m + f n

def is_non_decreasing (f : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, f (n + 1) ≥ f n

theorem function_is_zero
  (f : ℕ+ → ℝ)
  (h1 : is_logarithmic_property f)
  (h2 : is_non_decreasing f) :
  ∀ n : ℕ+, f n = 0 := by
  sorry

end function_is_zero_l646_64678


namespace arithmetic_sequence_formula_l646_64672

/-- Given an arithmetic sequence with first three terms a-1, a+1, 2a+3, 
    prove that its general formula is a_n = 2n - 3 -/
theorem arithmetic_sequence_formula 
  (a : ℝ) 
  (seq : ℕ → ℝ) 
  (h1 : seq 1 = a - 1) 
  (h2 : seq 2 = a + 1) 
  (h3 : seq 3 = 2*a + 3) 
  (h_arithmetic : ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)) :
  ∀ n : ℕ, seq n = 2*n - 3 :=
by sorry

end arithmetic_sequence_formula_l646_64672


namespace pet_food_price_l646_64605

/-- Given a manufacturer's suggested retail price and discount conditions, prove the price is $35 -/
theorem pet_food_price (M : ℝ) : 
  (M * (1 - 0.3) * (1 - 0.2) = 19.6) → M = 35 := by
  sorry

end pet_food_price_l646_64605


namespace library_books_total_l646_64692

/-- The total number of books obtained from the library -/
def total_books (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 54 initial books and 23 additional books, the total is 77 -/
theorem library_books_total : total_books 54 23 = 77 := by
  sorry

end library_books_total_l646_64692


namespace odd_function_value_at_negative_two_l646_64607

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_two
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x * (x - 1)) :
  f (-2) = -2 := by
  sorry

end odd_function_value_at_negative_two_l646_64607


namespace proportionality_coefficient_l646_64620

/-- Given variables x y z : ℝ and a constant k : ℕ+, prove that k = 3 -/
theorem proportionality_coefficient (x y z : ℝ) (k : ℕ+) : 
  (z - y = k * x) →  -- The difference of z and y is proportional to x
  (x - z = k * y) →  -- The difference of x and z is proportional to y
  (∃ (x' y' z' : ℝ), z' = (5/3) * (x' - y')) →  -- A certain value of z is 5/3 times the difference of x and y
  k = 3 := by
sorry

end proportionality_coefficient_l646_64620


namespace expression_evaluation_l646_64628

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) :
  -a - b^2 + a*b = -16 := by sorry

end expression_evaluation_l646_64628


namespace consecutive_even_product_l646_64651

theorem consecutive_even_product : 442 * 444 * 446 = 87526608 := by
  sorry

end consecutive_even_product_l646_64651


namespace line_equation_l646_64694

/-- A line passing through (1,1) with y-intercept 3 has equation 2x + y - 3 = 0 -/
theorem line_equation (x y : ℝ) : 
  (2 * 1 + 1 - 3 = 0) ∧ 
  (2 * 0 + 3 - 3 = 0) ∧ 
  (∀ x y, y = -2 * x + 3) → 
  2 * x + y - 3 = 0 := by sorry

end line_equation_l646_64694


namespace min_side_length_of_A_l646_64666

-- Define the squares
structure Square where
  sideLength : ℕ

-- Define the configuration
structure SquareConfiguration where
  A : Square
  B : Square
  C : Square
  D : Square
  vertexCondition : A.sideLength = B.sideLength + C.sideLength + D.sideLength
  areaCondition : A.sideLength^2 / 2 = B.sideLength^2 + C.sideLength^2 + D.sideLength^2

-- Theorem statement
theorem min_side_length_of_A (config : SquareConfiguration) :
  config.A.sideLength ≥ 3 :=
sorry

end min_side_length_of_A_l646_64666


namespace smallest_solution_l646_64685

-- Define the equation
def equation (x : ℝ) : Prop := x * (abs x) + 3 * x = 5 * x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | equation x}

-- State the theorem
theorem smallest_solution :
  ∃ (x : ℝ), x ∈ solution_set ∧ ∀ (y : ℝ), y ∈ solution_set → x ≤ y ∧ x = -1 - Real.sqrt 3 :=
sorry

end smallest_solution_l646_64685


namespace monotonically_decreasing_condition_l646_64622

def f (x : ℝ) := x^2 - 2*x + 3

theorem monotonically_decreasing_condition (m : ℝ) :
  (∀ x y, x < y ∧ y < m → f x > f y) ↔ m ≤ 1 := by sorry

end monotonically_decreasing_condition_l646_64622


namespace max_value_of_sum_of_roots_l646_64693

theorem max_value_of_sum_of_roots (a b c : ℝ) 
  (sum_eq_two : a + b + c = 2)
  (a_ge_neg_one : a ≥ -1)
  (b_ge_neg_two : b ≥ -2)
  (c_ge_neg_three : c ≥ -3) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 6 ∧
    ∀ (x : ℝ), x = Real.sqrt (4 * a + 2) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 14) → x ≤ max :=
by sorry

end max_value_of_sum_of_roots_l646_64693


namespace function_inequality_implies_a_range_l646_64611

open Real

theorem function_inequality_implies_a_range (a b : ℝ) :
  (∀ x ∈ Set.Ioo (Real.exp 1) ((Real.exp 1) ^ 2),
    ∀ b ≤ 0,
      a * log x - b * x^2 ≥ x) →
  a ≥ (Real.exp 1)^2 / 2 :=
by sorry

end function_inequality_implies_a_range_l646_64611


namespace inequality_proof_l646_64686

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2 / 3 ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) :=
by sorry

end inequality_proof_l646_64686
