import Mathlib

namespace unique_root_quadratic_l3724_372437

/-- The quadratic equation x^2 - 6mx + 9m has exactly one real root if and only if m = 1 (for positive m) -/
theorem unique_root_quadratic (m : ℝ) (h : m > 0) : 
  (∃! x : ℝ, x^2 - 6*m*x + 9*m = 0) ↔ m = 1 :=
by sorry

end unique_root_quadratic_l3724_372437


namespace exists_five_digit_number_with_digit_sum_31_divisible_by_31_l3724_372418

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_five_digit_number_with_digit_sum_31_divisible_by_31 :
  ∃ n : ℕ, is_five_digit n ∧ digit_sum n = 31 ∧ n % 31 = 0 := by
  sorry

end exists_five_digit_number_with_digit_sum_31_divisible_by_31_l3724_372418


namespace triangle_probability_l3724_372494

def stickLengths : List ℕ := [1, 2, 4, 6, 9, 10, 14, 15, 18]

def canFormTriangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def validTriangleCombinations : List (ℕ × ℕ × ℕ) := 
  [(4, 6, 9), (4, 9, 10), (4, 9, 14), (4, 10, 14), (4, 14, 15),
   (6, 9, 10), (6, 9, 14), (6, 10, 14), (6, 14, 15), (6, 9, 15), (6, 10, 15),
   (9, 10, 14), (9, 14, 15), (9, 10, 15),
   (10, 14, 15)]

def totalCombinations : ℕ := Nat.choose 9 3

theorem triangle_probability : 
  (validTriangleCombinations.length : ℚ) / totalCombinations = 4 / 21 := by
  sorry

end triangle_probability_l3724_372494


namespace quadratic_roots_reciprocal_sum_l3724_372477

/-- Given a quadratic equation mx^2 - (m+2)x + m/4 = 0 with two distinct real roots,
    if the sum of the reciprocals of the roots is 4m, then m = 2 -/
theorem quadratic_roots_reciprocal_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 - (m + 2) * x + m / 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  1 / x₁ + 1 / x₂ = 4 * m →
  m = 2 := by
sorry

end quadratic_roots_reciprocal_sum_l3724_372477


namespace number_relationship_l3724_372462

/-- Given two real numbers satisfying certain conditions, prove they are approximately equal to specific values. -/
theorem number_relationship (x y : ℝ) 
  (h1 : 0.25 * x = 1.3 * 0.35 * y) 
  (h2 : x - y = 155) : 
  ∃ (εx εy : ℝ), εx < 1 ∧ εy < 1 ∧ |x - 344| < εx ∧ |y - 189| < εy :=
sorry

end number_relationship_l3724_372462


namespace digit_proportion_theorem_l3724_372459

theorem digit_proportion_theorem :
  ∀ n : ℕ,
  (n / 2 : ℚ) + (n / 5 : ℚ) + (n / 5 : ℚ) + (n / 10 : ℚ) = n →
  (n / 2 : ℕ) + (n / 5 : ℕ) + (n / 5 : ℕ) + (n / 10 : ℕ) = n →
  n = 10 :=
by
  sorry

end digit_proportion_theorem_l3724_372459


namespace x_fourth_minus_reciprocal_l3724_372444

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 527 := by
  sorry

end x_fourth_minus_reciprocal_l3724_372444


namespace cubic_polynomial_satisfies_conditions_l3724_372485

theorem cubic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = -2/3 * x^3 + 3 * x^2 - 35/3 * x - 2) ∧
    q 0 = -2 ∧
    q 1 = -8 ∧
    q 3 = -18 ∧
    q 5 = -52 := by
  sorry

end cubic_polynomial_satisfies_conditions_l3724_372485


namespace trigonometric_expression_equality_l3724_372428

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.sin (-20/3 * Real.pi)) / Real.tan (11/3 * Real.pi) - 
  Real.cos (13/4 * Real.pi) * Real.tan (-35/4 * Real.pi) = 
  (Real.sqrt 2 + Real.sqrt 3) / 2 := by
  sorry

end trigonometric_expression_equality_l3724_372428


namespace family_age_sum_seven_years_ago_l3724_372451

/-- A family of 5 members -/
structure Family :=
  (age1 age2 age3 age4 age5 : ℕ)

/-- The sum of ages of the family members -/
def ageSum (f : Family) : ℕ := f.age1 + f.age2 + f.age3 + f.age4 + f.age5

/-- Theorem: Given a family of 5 whose ages sum to 80, with the two youngest being 6 and 8 years old,
    the sum of their ages 7 years ago was 45 -/
theorem family_age_sum_seven_years_ago (f : Family)
  (h1 : ageSum f = 80)
  (h2 : f.age4 = 8)
  (h3 : f.age5 = 6)
  (h4 : f.age1 ≥ 7 ∧ f.age2 ≥ 7 ∧ f.age3 ≥ 7) :
  (f.age1 - 7) + (f.age2 - 7) + (f.age3 - 7) + 1 = 45 :=
by sorry

end family_age_sum_seven_years_ago_l3724_372451


namespace money_left_after_transactions_l3724_372427

def initial_money : ℕ := 50 * 10 + 24 * 25 + 40 * 5 + 75

def candy_cost : ℕ := 6 * 85
def lollipop_cost : ℕ := 3 * 50
def chips_cost : ℕ := 4 * 95
def soda_cost : ℕ := 2 * 125

def total_cost : ℕ := candy_cost + lollipop_cost + chips_cost + soda_cost

theorem money_left_after_transactions : 
  initial_money - total_cost = 85 := by
sorry

end money_left_after_transactions_l3724_372427


namespace fraction_equality_l3724_372449

theorem fraction_equality (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  let C : ℝ := 19 / 5
  let D : ℝ := 17 / 5
  (D * x - 17) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5) := by
  sorry

end fraction_equality_l3724_372449


namespace area_PQRSTU_l3724_372406

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a polygon with 6 vertices -/
structure Hexagon :=
  (P Q R S T U : Point)

/-- The given hexagonal polygon PQRSTU -/
def PQRSTU : Hexagon := sorry

/-- Point V, the intersection of extended lines QT and PU -/
def V : Point := sorry

/-- Length of side PQ -/
def PQ_length : ℝ := 8

/-- Length of side QR -/
def QR_length : ℝ := 10

/-- Length of side UT -/
def UT_length : ℝ := 7

/-- Length of side TU -/
def TU_length : ℝ := 3

/-- Predicate stating that PQRV is a rectangle -/
def is_rectangle_PQRV (h : Hexagon) (v : Point) : Prop := sorry

/-- Predicate stating that VUT is a rectangle -/
def is_rectangle_VUT (h : Hexagon) (v : Point) : Prop := sorry

/-- Function to calculate the area of a polygon -/
def area (h : Hexagon) : ℝ := sorry

/-- Theorem stating that the area of PQRSTU is 65 square units -/
theorem area_PQRSTU :
  is_rectangle_PQRV PQRSTU V →
  is_rectangle_VUT PQRSTU V →
  area PQRSTU = 65 := by sorry

end area_PQRSTU_l3724_372406


namespace power_product_equals_4410000_l3724_372421

theorem power_product_equals_4410000 : 2^4 * 3^2 * 5^4 * 7^2 = 4410000 := by
  sorry

end power_product_equals_4410000_l3724_372421


namespace banana_orange_equivalence_l3724_372473

-- Define the worth of bananas in terms of oranges
def banana_orange_ratio : ℚ := 12 / (3/4 * 16)

-- Theorem statement
theorem banana_orange_equivalence : 
  banana_orange_ratio * (2/5 * 10 : ℚ) = 4 := by
  sorry

end banana_orange_equivalence_l3724_372473


namespace centers_form_square_l3724_372466

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Represents a square -/
structure Square where
  center : Point2D
  side_length : ℝ

/-- Function to construct squares on the sides of a parallelogram -/
def constructSquaresOnParallelogram (p : Parallelogram) : 
  (Square × Square × Square × Square) := sorry

/-- Function to get the centers of the squares -/
def getSquareCenters (squares : Square × Square × Square × Square) : 
  (Point2D × Point2D × Point2D × Point2D) := sorry

/-- Function to determine if a quadrilateral formed by four points is a square -/
def isSquare (p1 p2 p3 p4 : Point2D) : Prop := sorry

/-- Theorem stating that the quadrilateral formed by the centers of squares 
    drawn on the sides of a parallelogram is a square -/
theorem centers_form_square (p : Parallelogram) : 
  let squares := constructSquaresOnParallelogram p
  let (c1, c2, c3, c4) := getSquareCenters squares
  isSquare c1 c2 c3 c4 := by sorry

end centers_form_square_l3724_372466


namespace largest_side_of_special_triangle_l3724_372424

/-- Given a scalene triangle with sides x and y, and area Δ, satisfying the equation
    x + 2Δ/x = y + 2Δ/y, prove that when x = 60 and y = 63, the largest side is 87. -/
theorem largest_side_of_special_triangle (x y Δ : ℝ) 
  (hx : x = 60)
  (hy : y = 63)
  (h_eq : x + 2 * Δ / x = y + 2 * Δ / y)
  (h_scalene : x ≠ y)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_Δ : Δ > 0) :
  max x (max y (Real.sqrt (x^2 + y^2))) = 87 :=
sorry

end largest_side_of_special_triangle_l3724_372424


namespace radio_loss_percentage_l3724_372416

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℕ) : ℚ :=
  (cost_price - selling_price : ℚ) / cost_price * 100

theorem radio_loss_percentage :
  let cost_price := 1500
  let selling_price := 1260
  loss_percentage cost_price selling_price = 16 := by
sorry

end radio_loss_percentage_l3724_372416


namespace john_running_days_l3724_372408

/-- The number of days John ran before getting injured -/
def days_ran (daily_distance : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / daily_distance

theorem john_running_days :
  days_ran 1700 10200 = 6 :=
by sorry

end john_running_days_l3724_372408


namespace price_determined_by_digits_price_for_one_price_for_twelve_price_for_five_hundred_twelve_l3724_372490

/-- Calculates the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Calculates the price based on the number of digits -/
def price (quantity : ℕ) : ℕ := 1000 * num_digits quantity

/-- Theorem stating that the price is determined by the number of digits -/
theorem price_determined_by_digits (quantity : ℕ) :
  price quantity = 1000 * num_digits quantity :=
by sorry

/-- Theorem verifying the price for one unit -/
theorem price_for_one : price 1 = 1000 :=
by sorry

/-- Theorem verifying the price for twelve units -/
theorem price_for_twelve : price 12 = 2000 :=
by sorry

/-- Theorem verifying the price for five hundred twelve units -/
theorem price_for_five_hundred_twelve : price 512 = 3000 :=
by sorry

end price_determined_by_digits_price_for_one_price_for_twelve_price_for_five_hundred_twelve_l3724_372490


namespace wedding_decoration_cost_l3724_372400

/-- Calculates the total cost of decorations for a wedding reception --/
def total_decoration_cost (num_tables : ℕ) (tablecloth_cost : ℕ) (place_settings_per_table : ℕ) 
  (place_setting_cost : ℕ) (roses_per_centerpiece : ℕ) (rose_cost : ℕ) (lilies_per_centerpiece : ℕ) 
  (lily_cost : ℕ) : ℕ :=
  num_tables * (tablecloth_cost + place_settings_per_table * place_setting_cost + 
  roses_per_centerpiece * rose_cost + lilies_per_centerpiece * lily_cost)

/-- Theorem stating that the total decoration cost for the given parameters is 3500 --/
theorem wedding_decoration_cost : 
  total_decoration_cost 20 25 4 10 10 5 15 4 = 3500 := by
  sorry

#eval total_decoration_cost 20 25 4 10 10 5 15 4

end wedding_decoration_cost_l3724_372400


namespace max_visible_sum_l3724_372471

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Finset Nat)
  (face_count : faces.card = 6)
  (valid_faces : faces = {1, 2, 4, 8, 16, 32})

/-- Represents a stack of four cubes --/
def CubeStack := Fin 4 → Cube

/-- The sum of visible numbers in a cube stack --/
def visible_sum (stack : CubeStack) : Nat :=
  sorry

/-- Theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∀ stack : CubeStack, visible_sum stack ≤ 244 :=
sorry

end max_visible_sum_l3724_372471


namespace lomonosov_kvass_affordability_l3724_372469

theorem lomonosov_kvass_affordability 
  (x y : ℝ) 
  (initial_budget : x + y = 1) 
  (first_increase : 0.6 * x + 1.2 * y = 1) :
  1 ≥ 1.44 * y := by
  sorry

end lomonosov_kvass_affordability_l3724_372469


namespace coin_problem_l3724_372481

theorem coin_problem (initial_coins : ℚ) : 
  initial_coins > 0 →
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (3 / 4 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = 5 / 12 := by
sorry

end coin_problem_l3724_372481


namespace smallest_value_l3724_372483

theorem smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (1 / x^2 < x) ∧ 
  (1 / x^2 < x^2) ∧ 
  (1 / x^2 < 2*x^2) ∧ 
  (1 / x^2 < 3*x) ∧ 
  (1 / x^2 < Real.sqrt x) ∧ 
  (1 / x^2 < 1 / x) :=
by sorry

end smallest_value_l3724_372483


namespace investment_ratio_l3724_372480

theorem investment_ratio (a b c : ℝ) (profit total_profit : ℝ) : 
  a = 3 * b →                           -- A invests 3 times as much as B
  profit = 15000.000000000002 →         -- C's share
  total_profit = 55000 →                -- Total profit
  profit / total_profit = c / (a + b + c) → -- Profit distribution ratio
  a / c = 2                             -- Ratio of A's investment to C's investment
:= by sorry

end investment_ratio_l3724_372480


namespace triangle_shape_l3724_372487

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a * cos(A) = b * cos(B), then the triangle is either isosceles or right-angled. -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ A = B) ∨ A + B = Real.pi / 2 := by
  sorry

end triangle_shape_l3724_372487


namespace triangle_cosB_value_l3724_372488

theorem triangle_cosB_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  A = π / 4 →
  c * Real.sin B = Real.sqrt 3 * b * Real.cos C →
  Real.cos B = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry


end triangle_cosB_value_l3724_372488


namespace complex_sum_equals_seven_plus_three_i_l3724_372409

theorem complex_sum_equals_seven_plus_three_i :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -3
  let R : ℂ := -2*I
  let T : ℂ := 1 + 3*I
  B - Q + R + T = 7 + 3*I :=
by sorry

end complex_sum_equals_seven_plus_three_i_l3724_372409


namespace blanket_collection_l3724_372486

theorem blanket_collection (team_size : ℕ) (first_day_per_person : ℕ) (second_day_multiplier : ℕ) (third_day_fixed : ℕ) :
  team_size = 15 →
  first_day_per_person = 2 →
  second_day_multiplier = 3 →
  third_day_fixed = 22 →
  (team_size * first_day_per_person) + 
  (team_size * first_day_per_person * second_day_multiplier) + 
  third_day_fixed = 142 := by
sorry

end blanket_collection_l3724_372486


namespace symmetric_function_axis_l3724_372474

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := 1

-- State the theorem
theorem symmetric_function_axis (x : ℝ) : 
  f x = f (2 - x) → 
  f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
sorry

end symmetric_function_axis_l3724_372474


namespace intersection_y_intercept_l3724_372491

/-- Given two lines that intersect at a specific x-coordinate, 
    prove that the y-intercept of the first line has a specific value. -/
theorem intersection_y_intercept (k : ℝ) : 
  (∃ y : ℝ, -3 * (-6.8) + y = k ∧ 0.25 * (-6.8) + y = 10) → k = 32.1 := by
  sorry

end intersection_y_intercept_l3724_372491


namespace sawmill_equivalence_l3724_372482

/-- Represents the number of cuts needed to divide a log into smaller logs -/
def cuts_needed (original_length : ℕ) (target_length : ℕ) : ℕ :=
  original_length / target_length - 1

/-- Represents the total number of cuts that can be made in one day -/
def cuts_per_day (logs_per_day : ℕ) (original_length : ℕ) (target_length : ℕ) : ℕ :=
  logs_per_day * cuts_needed original_length target_length

/-- Represents the time (in days) needed to cut a given number of logs -/
def time_needed (num_logs : ℕ) (original_length : ℕ) (target_length : ℕ) (cuts_per_day : ℕ) : ℚ :=
  (num_logs * cuts_needed original_length target_length : ℚ) / cuts_per_day

theorem sawmill_equivalence :
  let nine_meter_logs_per_day : ℕ := 600
  let twelve_meter_logs : ℕ := 400
  let cuts_per_day := cuts_per_day nine_meter_logs_per_day 9 3
  time_needed twelve_meter_logs 12 3 cuts_per_day = 1 := by
  sorry

end sawmill_equivalence_l3724_372482


namespace average_goals_calculation_l3724_372460

theorem average_goals_calculation (layla_goals kristin_goals : ℕ) : 
  layla_goals = 104 →
  kristin_goals = layla_goals - 24 →
  (layla_goals + kristin_goals) / 2 = 92 := by
  sorry

end average_goals_calculation_l3724_372460


namespace right_triangle_area_l3724_372402

theorem right_triangle_area (a c : ℝ) (h1 : a = 15) (h2 : c = 17) : ∃ b : ℝ, 
  a^2 + b^2 = c^2 ∧ (1/2) * a * b = 60 :=
by sorry

end right_triangle_area_l3724_372402


namespace custom_op_neg_two_neg_one_l3724_372448

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem statement
theorem custom_op_neg_two_neg_one :
  customOp (-2) (-1) = 3 := by
  sorry

end custom_op_neg_two_neg_one_l3724_372448


namespace binomial_coefficient_20_19_l3724_372441

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by sorry

end binomial_coefficient_20_19_l3724_372441


namespace adam_apples_l3724_372426

theorem adam_apples (jackie_apples : ℕ) (adam_apples : ℕ) 
  (h1 : jackie_apples = 10) 
  (h2 : jackie_apples = adam_apples + 1) : 
  adam_apples = 9 := by
  sorry

end adam_apples_l3724_372426


namespace gcd_powers_of_two_minus_one_problem_4_l3724_372414

theorem gcd_powers_of_two_minus_one (a b : Nat) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := by sorry

theorem problem_4 : Nat.gcd (2^6 - 1) (2^9 - 1) = 7 := by sorry

end gcd_powers_of_two_minus_one_problem_4_l3724_372414


namespace fraction_equality_solution_l3724_372463

theorem fraction_equality_solution :
  ∃! y : ℚ, (2 + y) / (6 + y) = (3 + y) / (4 + y) :=
by
  -- The unique solution is y = -10/3
  use -10/3
  sorry

end fraction_equality_solution_l3724_372463


namespace cards_taken_away_l3724_372442

theorem cards_taken_away (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 67)
  (h2 : final_cards = 58) :
  initial_cards - final_cards = 9 := by
  sorry

end cards_taken_away_l3724_372442


namespace volume_of_parallelepiped_l3724_372403

/-- A rectangular parallelepiped with given diagonal and side face diagonals -/
structure RectParallelepiped where
  diag : ℝ
  side_diag1 : ℝ
  side_diag2 : ℝ
  volume : ℝ

/-- The volume of a rectangular parallelepiped with the given dimensions -/
def volume_calc (p : RectParallelepiped) : Prop :=
  p.diag = 13 ∧ p.side_diag1 = 4 * Real.sqrt 10 ∧ p.side_diag2 = 3 * Real.sqrt 17 → p.volume = 144

theorem volume_of_parallelepiped :
  ∀ p : RectParallelepiped, volume_calc p :=
sorry

end volume_of_parallelepiped_l3724_372403


namespace seven_prime_pairs_l3724_372457

/-- A function that returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of pairs of distinct primes p and q such that p^2 * q^2 < n -/
def countPrimePairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 7 pairs of distinct primes p and q such that p^2 * q^2 < 1000 -/
theorem seven_prime_pairs :
  countPrimePairs 1000 = 7 := by sorry

end seven_prime_pairs_l3724_372457


namespace complex_magnitude_problem_l3724_372455

open Complex

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 + I) = 1 - 2*I) : abs z = 1 := by
  sorry

end complex_magnitude_problem_l3724_372455


namespace absolute_value_non_negative_l3724_372468

theorem absolute_value_non_negative (x : ℝ) : 0 ≤ |x| := by
  sorry

end absolute_value_non_negative_l3724_372468


namespace covered_area_equals_transformed_square_l3724_372443

/-- A square in a 2D plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Transformation of a square by rotation and scaling -/
def transform_square (s : Square) (angle : ℝ) (scale : ℝ) : Square :=
  { center := s.center,
    side_length := s.side_length * scale }

/-- The set of all points covered by squares with one diagonal on the given square -/
def covered_area (s : Square) : Set (ℝ × ℝ) :=
  { p | ∃ (sq : Square), sq.center = s.center ∧ 
        (sq.side_length)^2 = 2 * (s.side_length)^2 ∧
        p ∈ { q | ∃ (x y : ℝ), 
              (x - sq.center.1)^2 + (y - sq.center.2)^2 ≤ (sq.side_length / 2)^2 } }

theorem covered_area_equals_transformed_square (s : Square) :
  covered_area s = { p | ∃ (x y : ℝ), 
                        (x - s.center.1)^2 + (y - s.center.2)^2 ≤ (s.side_length * Real.sqrt 2)^2 } := by
  sorry

end covered_area_equals_transformed_square_l3724_372443


namespace sum_xyz_equals_four_l3724_372493

theorem sum_xyz_equals_four (X Y Z : ℕ+) 
  (h_gcd : Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1)
  (h_eq : (X : ℝ) * (Real.log 3 / Real.log 100) + (Y : ℝ) * (Real.log 4 / Real.log 100) = Z) :
  X + Y + Z = 4 := by
  sorry

end sum_xyz_equals_four_l3724_372493


namespace snail_wins_l3724_372413

-- Define the race parameters
def race_distance : ℝ := 200

-- Define the animals' movements
structure Snail where
  speed : ℝ
  
structure Rabbit where
  initial_distance : ℝ
  speed : ℝ
  run_time1 : ℝ
  nap_time1 : ℝ
  run_time2 : ℝ
  nap_time2 : ℝ

-- Define the race conditions
def race_conditions (s : Snail) (r : Rabbit) : Prop :=
  s.speed > 0 ∧
  r.speed > 0 ∧
  r.initial_distance = 120 ∧
  r.run_time1 > 0 ∧
  r.nap_time1 > 0 ∧
  r.run_time2 > 0 ∧
  r.nap_time2 > 0 ∧
  r.initial_distance + r.speed * (r.run_time1 + r.run_time2) = race_distance

-- Theorem statement
theorem snail_wins (s : Snail) (r : Rabbit) 
  (h : race_conditions s r) : 
  s.speed * (r.run_time1 + r.nap_time1 + r.run_time2 + r.nap_time2) = race_distance :=
sorry

end snail_wins_l3724_372413


namespace prime_power_sum_l3724_372484

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 1260 →
  w + 2*x + 3*y + 4*z = 13 := by
  sorry

end prime_power_sum_l3724_372484


namespace totalPaintingCost_l3724_372411

/-- Calculates the sum of digits for a given range of an arithmetic sequence -/
def sumOfDigits (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

/-- Calculates the total cost to paint house numbers on one side of the street -/
def sideCost (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

/-- The total cost to paint all house numbers on the street -/
theorem totalPaintingCost : 
  let eastSideCost := sideCost 5 7 25
  let westSideCost := sideCost 6 8 25
  eastSideCost + westSideCost = 123 := by
  sorry

end totalPaintingCost_l3724_372411


namespace arccos_cos_three_l3724_372410

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 := by sorry

end arccos_cos_three_l3724_372410


namespace excess_meat_sales_l3724_372452

def meat_market_sales (thursday_sales : ℕ) (saturday_sales : ℕ) (original_plan : ℕ) : Prop :=
  let friday_sales := 2 * thursday_sales
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  total_sales - original_plan = 325

theorem excess_meat_sales : meat_market_sales 210 130 500 := by
  sorry

end excess_meat_sales_l3724_372452


namespace train_passing_time_l3724_372458

/-- Proves that a train of given length and speed takes approximately the calculated time to pass a pole -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (ε : ℝ) :
  train_length = 125 →
  train_speed_kmh = 60 →
  ε > 0 →
  ∃ (t : ℝ), t > 0 ∧ abs (t - 7.5) < ε ∧ t = train_length / (train_speed_kmh * 1000 / 3600) :=
by sorry

end train_passing_time_l3724_372458


namespace complex_equation_solution_l3724_372438

theorem complex_equation_solution (i : ℂ) (z : ℂ) :
  i * i = -1 →
  (2 - i) * z = i^3 →
  z = 1/5 - (2/5) * i :=
by
  sorry

end complex_equation_solution_l3724_372438


namespace average_beef_sold_is_260_l3724_372498

/-- The average amount of beef sold per day over three days -/
def average_beef_sold (thursday_sales : ℕ) (saturday_sales : ℕ) : ℚ :=
  (thursday_sales + 2 * thursday_sales + saturday_sales) / 3

/-- Proof that the average amount of beef sold per day is 260 pounds -/
theorem average_beef_sold_is_260 :
  average_beef_sold 210 150 = 260 := by
  sorry

end average_beef_sold_is_260_l3724_372498


namespace binomial_divisibility_l3724_372497

theorem binomial_divisibility (p n : ℕ) (hp : Prime p) (hn : p < n) 
  (hdiv : p ∣ (n + 1)) (hcoprime : Nat.gcd (n / p) (Nat.factorial (p - 1)) = 1) :
  p * (n / p)^2 ∣ (Nat.choose n p - n / p) := by
  sorry

end binomial_divisibility_l3724_372497


namespace rose_difference_l3724_372422

theorem rose_difference (santiago_roses garrett_roses : ℕ) 
  (h1 : santiago_roses = 58) 
  (h2 : garrett_roses = 24) : 
  santiago_roses - garrett_roses = 34 := by
sorry

end rose_difference_l3724_372422


namespace x₂_1994th_place_l3724_372417

-- Define the equation
def equation (x : ℝ) : Prop := x * Real.sqrt 8 + 1 / (x * Real.sqrt 8) = Real.sqrt 8

-- Define the two real solutions
axiom x₁ : ℝ
axiom x₂ : ℝ

-- Define that x₁ and x₂ satisfy the equation
axiom x₁_satisfies : equation x₁
axiom x₂_satisfies : equation x₂

-- Define the decimal place function (simplified for this problem)
def decimal_place (x : ℝ) (n : ℕ) : ℕ := sorry

-- Define that the 1994th decimal place of x₁ is 6
axiom x₁_1994th_place : decimal_place x₁ 1994 = 6

-- Theorem to prove
theorem x₂_1994th_place : decimal_place x₂ 1994 = 3 := by sorry

end x₂_1994th_place_l3724_372417


namespace stratified_sampling_survey_l3724_372436

theorem stratified_sampling_survey (total_counties : ℕ) (jiujiang_counties : ℕ) (jiujiang_samples : ℕ) : 
  total_counties = 20 → jiujiang_counties = 8 → jiujiang_samples = 2 →
  ∃ (total_samples : ℕ), total_samples = 5 := by
sorry

end stratified_sampling_survey_l3724_372436


namespace share_division_l3724_372433

theorem share_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 585)
  (h_equal : 4 * a = 6 * b ∧ 6 * b = 3 * c)
  (h_sum : a + b + c = total) :
  c = 260 := by
  sorry

end share_division_l3724_372433


namespace cubic_sum_inequality_quadratic_roots_bound_l3724_372405

theorem cubic_sum_inequality (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 := by
  sorry

theorem quadratic_roots_bound (a b : ℝ) (h : |a| + |b| < 1) :
  ∀ x, x^2 + a*x + b = 0 → |x| < 1 := by
  sorry

end cubic_sum_inequality_quadratic_roots_bound_l3724_372405


namespace canoe_rental_cost_l3724_372420

/-- Represents the daily rental cost and count of canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℝ
  kayak_cost : ℝ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def total_revenue (info : RentalInfo) : ℝ :=
  info.canoe_cost * info.canoe_count + info.kayak_cost * info.kayak_count

/-- Theorem stating that the daily rental cost of a canoe is $15 --/
theorem canoe_rental_cost (info : RentalInfo) :
  info.kayak_cost = 18 ∧
  info.canoe_count = (3 * info.kayak_count) / 2 ∧
  total_revenue info = 405 ∧
  info.canoe_count = info.kayak_count + 5 →
  info.canoe_cost = 15 := by
  sorry

end canoe_rental_cost_l3724_372420


namespace absolute_value_equality_l3724_372434

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x - 5| → x = 4 := by
  sorry

end absolute_value_equality_l3724_372434


namespace inequalities_hold_l3724_372472

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a) (h2 : y^2 < b) (h3 : z^2 < c) : 
  (x*y + y*z + z*x < a + b + c) ∧ 
  (x^4 + y^4 + z^4 < a^2 + b^2 + c^2) ∧ 
  (x^3*y^3*z^3 < a*b*c) := by
  sorry

end inequalities_hold_l3724_372472


namespace first_house_delivery_l3724_372465

/-- Calculates the number of bottles delivered to the first house -/
def bottles_delivered (total : ℕ) (cider : ℕ) (beer : ℕ) : ℕ :=
  let mixed := total - (cider + beer)
  (cider / 2) + (beer / 2) + (mixed / 2)

/-- Theorem stating that given the problem conditions, 90 bottles are delivered to the first house -/
theorem first_house_delivery :
  bottles_delivered 180 40 80 = 90 := by
  sorry

end first_house_delivery_l3724_372465


namespace water_left_after_experiment_l3724_372461

-- Define the initial amount of water
def initial_water : ℚ := 2

-- Define the amount of water used in the experiment
def water_used : ℚ := 7/6

-- Theorem to prove
theorem water_left_after_experiment :
  initial_water - water_used = 5/6 := by sorry

end water_left_after_experiment_l3724_372461


namespace triangle_area_l3724_372456

/-- Given a triangle ABC with the following properties:
  * sin(C/2) = √6/4
  * c = 2
  * sin B = 2 sin A
  Prove that the area of the triangle is √15/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) 
  (h_sin_half_C : Real.sin (C / 2) = Real.sqrt 6 / 4)
  (h_c : c = 2)
  (h_sin_B : Real.sin B = 2 * Real.sin A) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 / 4 := by
  sorry

end triangle_area_l3724_372456


namespace num_eulerian_circuits_city_graph_l3724_372464

/-- A graph representing the road network between cities. -/
structure RoadGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)
  sym : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ a, (a, a) ∉ edges

/-- The degree of a vertex in the graph. -/
def degree (G : RoadGraph) (v : Char) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- A graph is Eulerian if all vertices have even degree. -/
def is_eulerian (G : RoadGraph) : Prop :=
  ∀ v ∈ G.vertices, Even (degree G v)

/-- The number of Eulerian circuits in a graph. -/
def num_eulerian_circuits (G : RoadGraph) : ℕ :=
  sorry

/-- The specific road graph described in the problem. -/
def city_graph : RoadGraph :=
  { vertices := {'A', 'B', 'C', 'D', 'E'},
    edges := sorry,
    sym := sorry,
    no_self_loops := sorry }

/-- The main theorem stating the number of Eulerian circuits in the city graph. -/
theorem num_eulerian_circuits_city_graph :
  is_eulerian city_graph →
  degree city_graph 'A' = 6 →
  degree city_graph 'B' = 4 →
  degree city_graph 'C' = 4 →
  degree city_graph 'D' = 4 →
  degree city_graph 'E' = 2 →
  num_eulerian_circuits city_graph = 264 :=
sorry

end num_eulerian_circuits_city_graph_l3724_372464


namespace smallest_bdf_l3724_372404

theorem smallest_bdf (a b c d e f : ℕ+) : 
  (∃ A : ℚ, A = (a / b) * (c / d) * (e / f) ∧ 
   ((a + 1) / b) * (c / d) * (e / f) = A + 3 ∧
   (a / b) * ((c + 1) / d) * (e / f) = A + 4 ∧
   (a / b) * (c / d) * ((e + 1) / f) = A + 5) →
  (∃ m : ℕ+, b * d * f = m ∧ ∀ n : ℕ+, b * d * f ≤ n) →
  b * d * f = 60 :=
by sorry

end smallest_bdf_l3724_372404


namespace choose_four_from_multiset_l3724_372467

/-- Represents a multiset of letters -/
def LetterMultiset : Type := List Char

/-- The specific multiset of letters in our problem -/
def problemMultiset : LetterMultiset := ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

/-- Counts the number of ways to choose k elements from a multiset -/
def countChoices (ms : LetterMultiset) (k : Nat) : Nat :=
  sorry -- Implementation not required for the statement

/-- The main theorem stating that there are 175 ways to choose 4 letters from the given multiset -/
theorem choose_four_from_multiset :
  countChoices problemMultiset 4 = 175 := by
  sorry

end choose_four_from_multiset_l3724_372467


namespace self_common_tangents_l3724_372423

-- Define the concept of a self-common tangent
def has_self_common_tangent (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ x₂ y m b : ℝ), x₁ ≠ x₂ ∧ 
    f x₁ y ∧ f x₂ y ∧
    (∀ x y, f x y → y = m * x + b)

-- Define the four curves
def curve1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def curve2 (x y : ℝ) : Prop := y = x^2 - abs x
def curve3 (x y : ℝ) : Prop := y = 3 * Real.sin x + 4 * Real.cos x
def curve4 (x y : ℝ) : Prop := abs x + 1 = Real.sqrt (4 - y^2)

-- Theorem statement
theorem self_common_tangents :
  has_self_common_tangent curve2 ∧ 
  has_self_common_tangent curve3 ∧
  ¬has_self_common_tangent curve1 ∧
  ¬has_self_common_tangent curve4 :=
sorry

end self_common_tangents_l3724_372423


namespace students_in_both_clubs_count_l3724_372489

/-- Represents the number of students in both drama and art clubs -/
def students_in_both_clubs (total : ℕ) (drama : ℕ) (art : ℕ) (drama_or_art : ℕ) : ℕ :=
  drama + art - drama_or_art

/-- Theorem stating the number of students in both drama and art clubs -/
theorem students_in_both_clubs_count : 
  students_in_both_clubs 300 120 150 220 = 50 := by
  sorry

end students_in_both_clubs_count_l3724_372489


namespace equation_solutions_l3724_372496

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (1/4 + Real.sqrt 17 / 4) ∧ x₂ = (1/4 - Real.sqrt 17 / 4) ∧
    2 * x₁^2 - 2 = x₁ ∧ 2 * x₂^2 - 2 = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 2 ∧
    x₁ * (x₁ - 2) + x₁ - 2 = 0 ∧ x₂ * (x₂ - 2) + x₂ - 2 = 0) :=
by sorry

end equation_solutions_l3724_372496


namespace original_expression_proof_l3724_372415

theorem original_expression_proof (X a b c : ℤ) : 
  X + (a*b - 2*b*c + 3*a*c) = 2*b*c - 3*a*c + 2*a*b → 
  X = 4*b*c - 6*a*c + a*b := by
sorry

end original_expression_proof_l3724_372415


namespace problem_1_problem_2_problem_3_l3724_372432

-- Problem 1
theorem problem_1 : Real.sqrt 48 / Real.sqrt 3 * (1/4) = 1 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 12 - Real.sqrt 3 + Real.sqrt (1/3) = (4 * Real.sqrt 3) / 3 := by sorry

-- Problem 3
theorem problem_3 : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) + Real.sqrt 3 * (2 - Real.sqrt 3) = 2 * Real.sqrt 3 - 2 := by sorry

end problem_1_problem_2_problem_3_l3724_372432


namespace weekend_haircut_price_l3724_372412

theorem weekend_haircut_price (weekday_price : ℝ) (weekend_markup : ℝ) : 
  weekday_price = 18 → weekend_markup = 0.5 → weekday_price * (1 + weekend_markup) = 27 := by
  sorry

end weekend_haircut_price_l3724_372412


namespace quadratic_factorization_l3724_372478

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end quadratic_factorization_l3724_372478


namespace valid_lineups_count_l3724_372430

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 6

/-- The number of players who refuse to play together -/
def refusing_players : ℕ := 3

/-- Calculates the number of valid lineups -/
def valid_lineups : ℕ := 
  Nat.choose total_players lineup_size - Nat.choose (total_players - refusing_players) (lineup_size - refusing_players)

theorem valid_lineups_count : valid_lineups = 4785 := by sorry

end valid_lineups_count_l3724_372430


namespace group_collection_l3724_372447

/-- Calculates the total collection in rupees for a group where each member contributes as many paise as the number of members -/
def total_collection (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Proves that for a group of 68 members, the total collection is 46.24 rupees -/
theorem group_collection :
  total_collection 68 = 46.24 := by
  sorry

end group_collection_l3724_372447


namespace inverse_matrices_values_l3724_372407

def Matrix1 (a : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![a, 2],
    ![1, 4]]

def Matrix2 (b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![-2/7, 1/7],
    ![b, 3/14]]

theorem inverse_matrices_values (a b : ℚ) : 
  Matrix1 a * Matrix2 b = 1 → a = -3 ∧ b = 1/14 := by
  sorry

end inverse_matrices_values_l3724_372407


namespace ashton_pencils_l3724_372425

theorem ashton_pencils (initial_pencils_per_box : ℕ) : 
  (2 * initial_pencils_per_box) - 6 = 22 → initial_pencils_per_box = 14 :=
by sorry

end ashton_pencils_l3724_372425


namespace cubic_root_sum_inverse_squares_l3724_372429

theorem cubic_root_sum_inverse_squares (a b c : ℝ) : 
  a^3 - 8*a^2 + 6*a - 3 = 0 →
  b^3 - 8*b^2 + 6*b - 3 = 0 →
  c^3 - 8*c^2 + 6*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = -4/3 :=
by sorry

end cubic_root_sum_inverse_squares_l3724_372429


namespace annie_cookie_ratio_l3724_372499

-- Define the number of cookies eaten on each day
def monday_cookies : ℕ := 5
def tuesday_cookies : ℕ := 10  -- We know this from the solution, but it's not given in the problem
def wednesday_cookies : ℕ := (tuesday_cookies * 140) / 100

-- Define the total number of cookies eaten
def total_cookies : ℕ := 29

-- State the theorem
theorem annie_cookie_ratio :
  monday_cookies + tuesday_cookies + wednesday_cookies = total_cookies ∧
  wednesday_cookies = (tuesday_cookies * 140) / 100 ∧
  tuesday_cookies / monday_cookies = 2 := by
sorry

end annie_cookie_ratio_l3724_372499


namespace negation_of_universal_proposition_l3724_372492

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end negation_of_universal_proposition_l3724_372492


namespace ratio_problem_l3724_372445

theorem ratio_problem (x y : ℚ) (h : (3*x - 2*y) / (2*x + y) = 3/4) : x / y = 11/6 := by
  sorry

end ratio_problem_l3724_372445


namespace f_composition_eq_log_range_l3724_372440

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then (1/2) * x - 1/2 else Real.log x

theorem f_composition_eq_log_range (a : ℝ) :
  f (f a) = Real.log (f a) → a ∈ Set.Ici (Real.exp 1) :=
sorry

end f_composition_eq_log_range_l3724_372440


namespace curve_symmetry_l3724_372401

/-- A curve in the xy-plane -/
class Curve (f : ℝ → ℝ → Prop) : Prop

/-- Symmetry of a curve with respect to a line -/
def symmetricTo (f : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y ↔ f (y + 3) (x - 3)

/-- The line x - y - 3 = 0 -/
def symmetryLine (x y : ℝ) : ℝ := x - y - 3

/-- Theorem: If a curve f is symmetric with respect to the line x - y - 3 = 0,
    then its equation is f(y + 3, x - 3) = 0 -/
theorem curve_symmetry (f : ℝ → ℝ → Prop) [Curve f] 
    (h : symmetricTo f symmetryLine) :
  ∀ x y, f x y ↔ f (y + 3) (x - 3) := by
  sorry

end curve_symmetry_l3724_372401


namespace expression_simplification_l3724_372476

theorem expression_simplification (a b : ℚ) (h1 : a = -2) (h2 : b = 2/3) :
  3 * (2 * a^2 - 3 * a * b - 5 * a - 1) - 6 * (a^2 - a * b + 1) = 25 := by
  sorry

end expression_simplification_l3724_372476


namespace least_subtraction_for_divisibility_problem_solution_l3724_372479

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 5 ∧ (378461 - k) % 13 = 0 ∧ ∀ (m : ℕ), m < k → (378461 - m) % 13 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l3724_372479


namespace quadratic_equation_solution_l3724_372419

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4*x} = {0, 4} := by sorry

end quadratic_equation_solution_l3724_372419


namespace square_floor_theorem_l3724_372431

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor where
  side_length : ℕ

/-- The number of black tiles on the diagonals of a square floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem: If a square floor has 75 black tiles on its diagonals, 
    then the total number of tiles is 1444 -/
theorem square_floor_theorem :
  ∃ (floor : SquareFloor), black_tiles floor = 75 ∧ total_tiles floor = 1444 :=
by
  sorry

end square_floor_theorem_l3724_372431


namespace circle_equation_from_ellipse_and_hyperbola_l3724_372495

/-- Given an ellipse and a hyperbola, prove that a circle centered at the right focus of the ellipse
    and tangent to the asymptotes of the hyperbola has the equation x^2 + y^2 - 10x + 9 = 0 -/
theorem circle_equation_from_ellipse_and_hyperbola 
  (ellipse : ∀ x y : ℝ, x^2 / 169 + y^2 / 144 = 1 → Set ℝ × ℝ)
  (hyperbola : ∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 → Set ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (is_right_focus : circle_center = (5, 0))
  (is_tangent_to_asymptotes : ∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x) → 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 16)) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 + 9 = 0} ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = 16 :=
by sorry

end circle_equation_from_ellipse_and_hyperbola_l3724_372495


namespace water_percentage_in_fresh_mushrooms_l3724_372439

theorem water_percentage_in_fresh_mushrooms 
  (fresh_mass : ℝ) 
  (dried_mass : ℝ) 
  (dried_water_percentage : ℝ) 
  (h1 : fresh_mass = 22) 
  (h2 : dried_mass = 2.5) 
  (h3 : dried_water_percentage = 12) : 
  (fresh_mass - dried_mass * (1 - dried_water_percentage / 100)) / fresh_mass * 100 = 90 := by
sorry

end water_percentage_in_fresh_mushrooms_l3724_372439


namespace area_difference_l3724_372446

-- Define the perimeter of the square playground
def square_perimeter : ℝ := 36

-- Define the perimeter of the rectangular basketball court
def rect_perimeter : ℝ := 38

-- Define the width of the rectangular basketball court
def rect_width : ℝ := 15

-- Theorem statement
theorem area_difference :
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rect_length := (rect_perimeter - 2 * rect_width) / 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 21 := by sorry

end area_difference_l3724_372446


namespace unsold_bars_l3724_372435

/-- Proves the number of unsold chocolate bars given total bars, price per bar, and total revenue --/
theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_revenue : ℕ) : 
  total_bars = 13 → price_per_bar = 2 → total_revenue = 18 → 
  total_bars - (total_revenue / price_per_bar) = 4 := by
  sorry

end unsold_bars_l3724_372435


namespace music_library_space_per_hour_l3724_372454

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  totalSpace : ℕ

/-- Calculates the average disk space per hour of music in a library -/
def averageSpacePerHour (library : MusicLibrary) : ℕ :=
  let totalHours := library.days * 24
  (library.totalSpace + totalHours - 1) / totalHours

theorem music_library_space_per_hour :
  let library := MusicLibrary.mk 15 20000
  averageSpacePerHour library = 56 := by
  sorry

end music_library_space_per_hour_l3724_372454


namespace find_number_l3724_372470

theorem find_number : ∃ x : ℝ, 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + x + (0.5 : ℝ)^2 = 0.3000000000000001 := by
  sorry

end find_number_l3724_372470


namespace negative_two_two_two_two_mod_thirteen_l3724_372453

theorem negative_two_two_two_two_mod_thirteen : ∃! n : ℤ, 0 ≤ n ∧ n < 13 ∧ -2222 ≡ n [ZMOD 13] ∧ n = 12 := by
  sorry

end negative_two_two_two_two_mod_thirteen_l3724_372453


namespace child_ticket_cost_l3724_372450

/-- Proves that the cost of a child ticket is 1 dollar given the conditions of the problem -/
theorem child_ticket_cost
  (adult_ticket_cost : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (child_attendees : ℕ)
  (h1 : adult_ticket_cost = 8)
  (h2 : total_attendees = 22)
  (h3 : total_revenue = 50)
  (h4 : child_attendees = 18) :
  (total_revenue - (total_attendees - child_attendees) * adult_ticket_cost) / child_attendees = 1 :=
by sorry

end child_ticket_cost_l3724_372450


namespace expression_simplification_l3724_372475

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (3 * a / (a^2 - 4)) * (1 - 2 / a) - 4 / (a + 2) = 1 - Real.sqrt 2 := by
  sorry

end expression_simplification_l3724_372475
