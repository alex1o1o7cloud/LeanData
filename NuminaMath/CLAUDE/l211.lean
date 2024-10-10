import Mathlib

namespace symmetry_wrt_origin_l211_21146

/-- Given a point P(3, 2) in the Cartesian coordinate system, 
    its symmetrical point P' with respect to the origin has coordinates (-3, -2). -/
theorem symmetry_wrt_origin :
  let P : ℝ × ℝ := (3, 2)
  let P' : ℝ × ℝ := (-P.1, -P.2)
  P' = (-3, -2) := by sorry

end symmetry_wrt_origin_l211_21146


namespace vector_magnitude_proof_l211_21133

def a : Fin 2 → ℝ := ![0, 1]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_magnitude_proof : 
  ‖(2 • (a : Fin 2 → ℝ)) + (b : Fin 2 → ℝ)‖ = Real.sqrt 5 := by
  sorry

end vector_magnitude_proof_l211_21133


namespace burger_cost_l211_21185

theorem burger_cost (alice_burgers alice_sodas alice_total bill_burgers bill_sodas bill_total : ℕ)
  (h_alice : alice_burgers = 4 ∧ alice_sodas = 3 ∧ alice_total = 420)
  (h_bill : bill_burgers = 3 ∧ bill_sodas = 2 ∧ bill_total = 310) :
  ∃ (burger_cost soda_cost : ℕ),
    alice_burgers * burger_cost + alice_sodas * soda_cost = alice_total ∧
    bill_burgers * burger_cost + bill_sodas * soda_cost = bill_total ∧
    burger_cost = 90 := by
  sorry

end burger_cost_l211_21185


namespace negation_of_existence_quadratic_inequality_negation_l211_21198

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 - 2*x - 3 < 0) ↔ (∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) :=
by sorry

end negation_of_existence_quadratic_inequality_negation_l211_21198


namespace original_number_proof_l211_21122

theorem original_number_proof : ∃! n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (n / 1000 = 6) ∧
  (1000 * (n % 1000) + 6 = n - 1152) ∧
  n = 6538 := by
sorry

end original_number_proof_l211_21122


namespace average_weight_is_15_l211_21120

def regression_weight (age : ℕ) : ℝ := 2 * age + 7

def children_ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

theorem average_weight_is_15 :
  (children_ages.map regression_weight).sum / children_ages.length = 15 := by
  sorry

end average_weight_is_15_l211_21120


namespace equal_selection_probability_l211_21186

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents the population and sample characteristics -/
structure Population where
  total_items : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  fourth_grade : ℕ
  sample_size : ℕ

/-- Calculates the probability of an item being selected for a given sampling method -/
def selection_probability (pop : Population) (method : SamplingMethod) : ℚ :=
  pop.sample_size / pop.total_items

/-- The main theorem stating that all sampling methods have the same selection probability -/
theorem equal_selection_probability (pop : Population) 
  (h1 : pop.total_items = 160)
  (h2 : pop.first_grade = 48)
  (h3 : pop.second_grade = 64)
  (h4 : pop.third_grade = 32)
  (h5 : pop.fourth_grade = 16)
  (h6 : pop.sample_size = 20)
  (h7 : pop.total_items = pop.first_grade + pop.second_grade + pop.third_grade + pop.fourth_grade) :
  ∀ m : SamplingMethod, selection_probability pop m = 1/8 := by
  sorry

#check equal_selection_probability

end equal_selection_probability_l211_21186


namespace constant_intersection_point_range_l211_21195

/-- Given that when m ∈ ℝ, the function f(x) = m(x^2 - 1) + x - a has a constant 
    intersection point with the x-axis, then a ∈ ℝ when m = 0 and a ∈ [-1, 1] when m ≠ 0 -/
theorem constant_intersection_point_range (m a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, m * (x^2 - 1) + x - a = 0 → x = k) → 
  ((m = 0 → a ∈ Set.univ) ∧ (m ≠ 0 → a ∈ Set.Icc (-1) 1)) :=
sorry

end constant_intersection_point_range_l211_21195


namespace intersection_line_of_circles_l211_21180

/-- The line passing through the intersection points of two circles -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x - 2)^2 + (y + 3)^2 = 8^2 →
  (x + 5)^2 + (y - 7)^2 = 136 →
  x + y = 4.35 :=
by
  sorry


end intersection_line_of_circles_l211_21180


namespace triangle_to_pentagon_area_ratio_l211_21107

/-- Given a pentagon formed by placing an equilateral triangle atop a square,
    where the side length of the square equals the height of the triangle,
    prove that the ratio of the triangle's area to the pentagon's area is (3(√3 - 1))/6 -/
theorem triangle_to_pentagon_area_ratio :
  ∀ s : ℝ, s > 0 →
  let h := s * (Real.sqrt 3 / 2)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  let square_area := h^2
  let pentagon_area := triangle_area + square_area
  triangle_area / pentagon_area = (3 * (Real.sqrt 3 - 1)) / 6 := by
sorry

end triangle_to_pentagon_area_ratio_l211_21107


namespace root_sum_fraction_l211_21158

theorem root_sum_fraction (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 175/11 :=
by sorry

end root_sum_fraction_l211_21158


namespace max_value_of_s_l211_21123

theorem max_value_of_s (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (product_condition : p * q + p * r + p * s + q * r + q * s + r * s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_s_l211_21123


namespace octal_127_equals_binary_1010111_l211_21119

def octal_to_decimal (x : ℕ) : ℕ := 
  (x % 10) + 8 * ((x / 10) % 10) + 64 * (x / 100)

def decimal_to_binary (x : ℕ) : List ℕ :=
  if x = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux x []

theorem octal_127_equals_binary_1010111 : 
  decimal_to_binary (octal_to_decimal 127) = [1, 0, 1, 0, 1, 1, 1] := by
  sorry

#eval octal_to_decimal 127
#eval decimal_to_binary (octal_to_decimal 127)

end octal_127_equals_binary_1010111_l211_21119


namespace prob_three_common_books_l211_21104

/-- The number of books in Mr. Johnson's list -/
def total_books : ℕ := 12

/-- The number of books each student must choose -/
def books_to_choose : ℕ := 5

/-- The number of common books we're interested in -/
def common_books : ℕ := 3

/-- The probability of Alice and Bob selecting exactly 3 common books -/
def prob_common_books : ℚ := 55 / 209

theorem prob_three_common_books :
  (Nat.choose total_books common_books *
   Nat.choose (total_books - common_books) (books_to_choose - common_books) *
   Nat.choose (total_books - books_to_choose) (books_to_choose - common_books)) /
  (Nat.choose total_books books_to_choose)^2 = prob_common_books :=
sorry

end prob_three_common_books_l211_21104


namespace maggie_bouncy_balls_l211_21137

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 6

/-- The number of balls in each pack of red bouncy balls -/
def red_balls_per_pack : ℕ := 12

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 10

/-- The number of balls in each pack of yellow bouncy balls -/
def yellow_balls_per_pack : ℕ := 8

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := 4

/-- The number of balls in each pack of green bouncy balls -/
def green_balls_per_pack : ℕ := 15

/-- The number of packs of blue bouncy balls -/
def blue_packs : ℕ := 3

/-- The number of balls in each pack of blue bouncy balls -/
def blue_balls_per_pack : ℕ := 20

/-- The total number of bouncy balls Maggie bought -/
def total_bouncy_balls : ℕ := 
  red_packs * red_balls_per_pack + 
  yellow_packs * yellow_balls_per_pack + 
  green_packs * green_balls_per_pack + 
  blue_packs * blue_balls_per_pack

theorem maggie_bouncy_balls : total_bouncy_balls = 272 := by
  sorry

end maggie_bouncy_balls_l211_21137


namespace correct_quadratic_equation_l211_21110

-- Define the quadratic equation
def quadratic_equation (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the roots from the first student's mistake
def root1 : ℝ := 3
def root2 : ℝ := 7

-- Define the roots from the second student's mistake
def root3 : ℝ := 5
def root4 : ℝ := -1

-- Theorem statement
theorem correct_quadratic_equation :
  ∃ (b c : ℝ),
    (root1 + root2 = -b) ∧
    (root3 * root4 = c) ∧
    (∀ x, quadratic_equation b c x = x^2 - 10*x - 5) :=
sorry

end correct_quadratic_equation_l211_21110


namespace part_one_part_two_l211_21152

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Theorem for part (1)
theorem part_one (a b : ℝ) (ha : a ≠ 0) 
  (h_solution_set : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  a = -1 ∧ b = 4 := by sorry

-- Theorem for part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_f_1 : f a b 1 = 2) :
  (1 / a + 4 / b) ≥ 9 ∧ ∃ (a b : ℝ), 1 / a + 4 / b = 9 := by sorry

end part_one_part_two_l211_21152


namespace divides_two_pow_36_minus_1_l211_21176

theorem divides_two_pow_36_minus_1 : 
  ∃! (n : ℕ), 40 ≤ n ∧ n ≤ 50 ∧ (2^36 - 1) % n = 0 ∧ n = 49 := by
  sorry

end divides_two_pow_36_minus_1_l211_21176


namespace solve_equation1_solve_equation2_l211_21121

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 - 2*x - 8 = 0
def equation2 (x : ℝ) : Prop := x^2 - 2*x - 5 = 0

-- Theorem for the first equation
theorem solve_equation1 : 
  ∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = -2 ∧ equation1 x1 ∧ equation1 x2 ∧
  ∀ x : ℝ, equation1 x → x = x1 ∨ x = x2 :=
sorry

-- Theorem for the second equation
theorem solve_equation2 : 
  ∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 6 ∧ x2 = 1 - Real.sqrt 6 ∧ 
  equation2 x1 ∧ equation2 x2 ∧
  ∀ x : ℝ, equation2 x → x = x1 ∨ x = x2 :=
sorry

end solve_equation1_solve_equation2_l211_21121


namespace three_digit_subtraction_problem_l211_21139

theorem three_digit_subtraction_problem :
  ∀ h t u : ℕ,
  h ≤ 9 ∧ t ≤ 9 ∧ u ≤ 9 →  -- Ensure single-digit numbers
  u = h - 5 →
  (100 * h + 10 * t + u) - (100 * h + 10 * u + t) = 96 →
  h = 5 ∧ t = 9 ∧ u = 0 :=
by sorry

end three_digit_subtraction_problem_l211_21139


namespace common_solution_y_values_l211_21127

theorem common_solution_y_values : 
  ∀ x y : ℝ, 
  (x^2 + y^2 - 9 = 0 ∧ x^2 + 2*y - 7 = 0) ↔ 
  (y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3) :=
by sorry

end common_solution_y_values_l211_21127


namespace inequality_solution_l211_21141

theorem inequality_solution (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 5 * x + 5) ↔ 
  (x > 3 + Real.sqrt 10 ∧ x < (7 + Real.sqrt 65) / 2) :=
by sorry

end inequality_solution_l211_21141


namespace machines_count_l211_21187

theorem machines_count (x : ℝ) (N : ℕ) (R : ℝ) : 
  N * R = x / 3 →
  45 * R = 5 * x / 10 →
  N = 30 := by
  sorry

end machines_count_l211_21187


namespace integer_sqrt_divisibility_l211_21126

theorem integer_sqrt_divisibility (n : ℕ) (h1 : n ≥ 4) :
  (Int.floor (Real.sqrt n) + 1 ∣ n - 1) ∧
  (Int.floor (Real.sqrt n) - 1 ∣ n + 1) →
  n = 4 ∨ n = 7 ∨ n = 9 ∨ n = 13 ∨ n = 31 := by
  sorry

end integer_sqrt_divisibility_l211_21126


namespace race_track_inner_circumference_l211_21184

/-- Given a circular race track with an outer radius of 140.0563499208679 m and a width of 18 m, 
    the inner circumference is approximately 767.145882893066 m. -/
theorem race_track_inner_circumference :
  let outer_radius : ℝ := 140.0563499208679
  let track_width : ℝ := 18
  let inner_radius : ℝ := outer_radius - track_width
  let inner_circumference : ℝ := 2 * Real.pi * inner_radius
  ∃ ε > 0, abs (inner_circumference - 767.145882893066) < ε :=
by sorry

end race_track_inner_circumference_l211_21184


namespace jose_peanut_count_l211_21101

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := 133

-- Define the difference between Kenya's and Jose's peanuts
def peanut_difference : ℕ := 48

-- Define Jose's peanuts
def jose_peanuts : ℕ := kenya_peanuts - peanut_difference

-- Theorem statement
theorem jose_peanut_count : jose_peanuts = 85 := by sorry

end jose_peanut_count_l211_21101


namespace sum_has_five_digits_l211_21109

def is_nonzero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def number_to_digits (n : ℕ) : ℕ := 
  if n = 0 then 1 else (Nat.log 10 n).succ

theorem sum_has_five_digits (A B : ℕ) 
  (hA : is_nonzero_digit A) (hB : is_nonzero_digit B) : 
  number_to_digits (19876 + (10000 * A + 1000 * B + 320) + (200 * B + 1)) = 5 := by
  sorry

end sum_has_five_digits_l211_21109


namespace exists_positive_a_leq_inverse_l211_21149

theorem exists_positive_a_leq_inverse : ∃ a : ℝ, a > 0 ∧ a ≤ 1 / a := by
  sorry

end exists_positive_a_leq_inverse_l211_21149


namespace orvin_balloon_purchase_l211_21144

def regular_price : ℕ := 4
def initial_balloons : ℕ := 35
def discount_ratio : ℚ := 1/2

def max_balloons : ℕ := 42

theorem orvin_balloon_purchase :
  let total_money := initial_balloons * regular_price
  let discounted_set_cost := 2 * regular_price + discount_ratio * regular_price
  let num_sets := total_money / discounted_set_cost
  num_sets * 3 = max_balloons :=
by sorry

end orvin_balloon_purchase_l211_21144


namespace michael_has_270_eggs_l211_21182

/-- The number of eggs Michael has after buying and giving away crates -/
def michael_eggs (initial_crates : ℕ) (given_away : ℕ) (bought_later : ℕ) (eggs_per_crate : ℕ) : ℕ :=
  ((initial_crates - given_away) + bought_later) * eggs_per_crate

/-- Theorem stating that Michael has 270 eggs given the problem conditions -/
theorem michael_has_270_eggs :
  michael_eggs 6 2 5 30 = 270 := by
  sorry

end michael_has_270_eggs_l211_21182


namespace expression_equality_l211_21191

theorem expression_equality (x : ℝ) (h : 3 * x^2 - 6 * x + 4 = 7) : 
  x^2 - 2 * x + 2 = 3 := by
  sorry

end expression_equality_l211_21191


namespace revenue_maximized_at_064_l211_21196

/-- Revenue function for electricity pricing -/
def revenue (x : ℝ) : ℝ := (1 + 50 * (x - 0.8)^2) * (x - 0.5)

/-- The domain of the revenue function -/
def price_range (x : ℝ) : Prop := 0.5 < x ∧ x < 0.8

theorem revenue_maximized_at_064 :
  ∃ (x : ℝ), price_range x ∧
    (∀ (y : ℝ), price_range y → revenue y ≤ revenue x) ∧
    x = 0.64 :=
sorry

end revenue_maximized_at_064_l211_21196


namespace max_area_of_rectangle_with_constraints_l211_21128

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.x + r.y)

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.x * r.y

/-- The condition that one side is at least twice as long as the other -/
def oneSideAtLeastTwiceOther (r : Rectangle) : Prop := r.x ≥ 2 * r.y

theorem max_area_of_rectangle_with_constraints :
  ∃ (r : Rectangle),
    perimeter r = 60 ∧
    oneSideAtLeastTwiceOther r ∧
    area r = 200 ∧
    ∀ (s : Rectangle),
      perimeter s = 60 →
      oneSideAtLeastTwiceOther s →
      area s ≤ area r :=
by sorry

end max_area_of_rectangle_with_constraints_l211_21128


namespace cone_volume_over_pi_l211_21175

/-- The volume of a cone formed from a 300-degree sector of a circle with radius 18, divided by π, is equal to 225√11 -/
theorem cone_volume_over_pi (r : ℝ) (sector_angle : ℝ) :
  r = 18 →
  sector_angle = 300 →
  let base_radius := sector_angle / 360 * r
  let height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * height
  volume / π = 225 * Real.sqrt 11 := by
  sorry

end cone_volume_over_pi_l211_21175


namespace expression_bounds_l211_21174

theorem expression_bounds (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ∧ 
  4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ≤ 22 + 4*Real.sqrt 5 := by
  sorry

end expression_bounds_l211_21174


namespace no_prime_sum_10003_l211_21157

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 10003 := by
  sorry

end no_prime_sum_10003_l211_21157


namespace platform_length_is_605_l211_21197

/-- Calculates the length of a platform given train movement parameters. -/
def platformLength (
  platformPassTime : Real
) (manPassTime : Real)
  (manDistance : Real)
  (initialSpeed : Real)
  (acceleration : Real) : Real :=
  let trainLength := manPassTime * initialSpeed + 0.5 * acceleration * manPassTime ^ 2 - manDistance
  let platformPassDistance := platformPassTime * initialSpeed + 0.5 * acceleration * platformPassTime ^ 2
  platformPassDistance - trainLength

/-- The length of the platform is 605 meters. -/
theorem platform_length_is_605 :
  platformLength 40 20 5 15 0.5 = 605 := by
  sorry

end platform_length_is_605_l211_21197


namespace checker_in_center_l211_21142

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a checker placement on the board -/
structure Placement :=
  (board : Board)
  (positions : Finset (ℕ × ℕ))

/-- Defines symmetry with respect to both main diagonals -/
def is_symmetric (p : Placement) : Prop :=
  ∀ (i j : ℕ), (i, j) ∈ p.positions ↔
    (j, i) ∈ p.positions ∧
    (p.board.size - 1 - i, p.board.size - 1 - j) ∈ p.positions

/-- The central cell of the board -/
def central_cell (b : Board) : ℕ × ℕ :=
  (b.size / 2, b.size / 2)

/-- The main theorem -/
theorem checker_in_center (p : Placement)
  (h_size : p.board.size = 25)
  (h_count : p.positions.card = 25)
  (h_sym : is_symmetric p) :
  central_cell p.board ∈ p.positions :=
sorry

end checker_in_center_l211_21142


namespace min_value_of_expression_l211_21140

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + a*b - 3 = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + x*y - 3 = 0 → 4*x + y ≥ 6 :=
by sorry

end min_value_of_expression_l211_21140


namespace expand_and_simplify_solve_equation_l211_21153

-- Problem 1
theorem expand_and_simplify (x y : ℝ) : 
  (x + 3*y)^2 - (x + 3*y)*(x - 3*y) = 6*x*y + 18*y^2 := by sorry

-- Problem 2
theorem solve_equation : 
  ∃ (x : ℝ), x / (2*x - 1) = 2 - 3 / (1 - 2*x) ∧ x = -1/3 := by sorry

end expand_and_simplify_solve_equation_l211_21153


namespace solution_set_quadratic_inequality_l211_21151

theorem solution_set_quadratic_inequality (x : ℝ) :
  x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 := by sorry

end solution_set_quadratic_inequality_l211_21151


namespace amount_owed_after_one_year_l211_21118

/-- Calculates the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the total amount owed after one year --/
theorem amount_owed_after_one_year :
  let principal : ℝ := 54
  let rate : ℝ := 0.05
  let time : ℝ := 1
  total_amount_owed principal rate time = 56.70 := by
sorry

end amount_owed_after_one_year_l211_21118


namespace hyperbola_asymptote_l211_21113

/-- Given a hyperbola with equation x²/a² - y² = 1 and an asymptote √3x + y = 0,
    prove that a = √3/3 -/
theorem hyperbola_asymptote (a : ℝ) : 
  (∃ x y : ℝ, x^2/a^2 - y^2 = 1) ∧ 
  (∃ x y : ℝ, Real.sqrt 3 * x + y = 0) → 
  a = Real.sqrt 3 / 3 := by
  sorry

end hyperbola_asymptote_l211_21113


namespace two_over_x_values_l211_21106

theorem two_over_x_values (x : ℝ) (h : 1 - 9/x + 20/x^2 = 0) :
  2/x = 1/2 ∨ 2/x = 0.4 := by
  sorry

end two_over_x_values_l211_21106


namespace wrong_multiplication_correction_l211_21159

theorem wrong_multiplication_correction (x : ℝ) (h : x * 2.4 = 288) : (x / 2.4) / 5 = 10 := by
  sorry

end wrong_multiplication_correction_l211_21159


namespace intersection_perpendicular_l211_21135

/-- The line y = x - 2 intersects the parabola y^2 = 2x at points A and B. 
    This theorem proves that OA ⊥ OB, where O is the origin (0, 0). -/
theorem intersection_perpendicular (A B : ℝ × ℝ) : 
  (∃ x y : ℝ, A = (x, y) ∧ y = x - 2 ∧ y^2 = 2*x) →
  (∃ x y : ℝ, B = (x, y) ∧ y = x - 2 ∧ y^2 = 2*x) →
  A ≠ B →
  let O : ℝ × ℝ := (0, 0)
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0 :=
by sorry


end intersection_perpendicular_l211_21135


namespace extreme_values_of_f_max_min_on_interval_parallel_tangents_midpoint_l211_21125

/-- The function f(x) = x^3 - 12x + 12 --/
def f (x : ℝ) : ℝ := x^3 - 12*x + 12

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem extreme_values_of_f :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 2 ∧ 
   f x₁ = 28 ∧ f x₂ = -4 ∧
   ∀ x : ℝ, f x ≤ f x₁ ∧ f x₂ ≤ f x) :=
sorry

theorem max_min_on_interval :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → -4 ≤ f x) ∧
  (∃ x₁ x₂ : ℝ, -3 ≤ x₁ ∧ x₁ ≤ 4 ∧ -3 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ = 28 ∧ f x₂ = -4) :=
sorry

theorem parallel_tangents_midpoint :
  ∀ a b : ℝ, f' a = f' b →
  (a + b) / 2 = 0 ∧ (f a + f b) / 2 = 12 :=
sorry

end extreme_values_of_f_max_min_on_interval_parallel_tangents_midpoint_l211_21125


namespace line_intersects_parabola_once_l211_21103

-- Define the point A
def A : ℝ × ℝ := (1, 2)

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line L
def L (y : ℝ) : Prop := y = 2

-- Theorem statement
theorem line_intersects_parabola_once :
  L (A.2) ∧ 
  (∃! p : ℝ × ℝ, C p.1 p.2 ∧ L p.2) ∧
  (∀ y : ℝ, L y → ∃ x : ℝ, (x, y) = A ∨ C x y) :=
sorry

end line_intersects_parabola_once_l211_21103


namespace eve_hit_ten_l211_21164

-- Define the set of possible scores
def ScoreSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a type for players
inductive Player : Type
| Alex | Becca | Carli | Dan | Eve | Fiona

-- Define a function that returns a player's score
def player_score : Player → ℕ
| Player.Alex => 20
| Player.Becca => 5
| Player.Carli => 13
| Player.Dan => 15
| Player.Eve => 21
| Player.Fiona => 6

-- Define a function that returns a pair of scores for a player
def player_throws (p : Player) : ℕ × ℕ := sorry

-- State the theorem
theorem eve_hit_ten :
  ∀ (p : Player),
    (∀ (q : Player), p ≠ q → player_throws p ≠ player_throws q) ∧
    (∀ (p : Player), (player_throws p).1 ∈ ScoreSet ∧ (player_throws p).2 ∈ ScoreSet) ∧
    (∀ (p : Player), (player_throws p).1 + (player_throws p).2 = player_score p) →
    (player_throws Player.Eve).1 = 10 ∨ (player_throws Player.Eve).2 = 10 :=
by sorry

end eve_hit_ten_l211_21164


namespace arithmetic_geometric_ratio_l211_21183

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  ∃ r, r = (a 3) / (a 2) ∧ r = (a 6) / (a 3) →  -- geometric sequence condition
  (a 3) / (a 2) = 3 := by
sorry

end arithmetic_geometric_ratio_l211_21183


namespace division_makes_equation_true_l211_21170

theorem division_makes_equation_true : (6 / 3) + 4 - (2 - 1) = 5 := by
  sorry

end division_makes_equation_true_l211_21170


namespace bricklayers_theorem_l211_21163

/-- Represents the problem of two bricklayers building a wall --/
structure BricklayersProblem where
  total_bricks : ℕ
  time_first : ℕ
  time_second : ℕ
  joint_decrease : ℕ
  joint_time : ℕ

/-- The solution to the bricklayers problem --/
def solve_bricklayers_problem (p : BricklayersProblem) : Prop :=
  p.total_bricks = 288 ∧
  p.time_first = 8 ∧
  p.time_second = 12 ∧
  p.joint_decrease = 12 ∧
  p.joint_time = 6 ∧
  (p.total_bricks / p.time_first + p.total_bricks / p.time_second - p.joint_decrease) * p.joint_time = p.total_bricks

theorem bricklayers_theorem (p : BricklayersProblem) :
  solve_bricklayers_problem p :=
sorry

end bricklayers_theorem_l211_21163


namespace abs_eq_solution_l211_21143

theorem abs_eq_solution (x : ℝ) : |x + 1| = 2*x + 4 ↔ x = -5/3 := by
  sorry

end abs_eq_solution_l211_21143


namespace ball_cost_l211_21117

/-- Given that Kyoko paid $4.62 for 3 balls, prove that each ball costs $1.54. -/
theorem ball_cost (total_paid : ℝ) (num_balls : ℕ) (h1 : total_paid = 4.62) (h2 : num_balls = 3) :
  total_paid / num_balls = 1.54 := by
sorry

end ball_cost_l211_21117


namespace average_of_remaining_digits_l211_21145

theorem average_of_remaining_digits 
  (total_digits : Nat) 
  (subset_digits : Nat)
  (total_average : ℚ) 
  (subset_average : ℚ) :
  total_digits = 9 →
  subset_digits = 4 →
  total_average = 18 →
  subset_average = 8 →
  (total_digits * total_average - subset_digits * subset_average) / (total_digits - subset_digits) = 26 :=
by sorry

end average_of_remaining_digits_l211_21145


namespace find_divisor_l211_21167

theorem find_divisor (n : ℕ) (d : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 ∧ n % d = 8 → d = 50 := by
  sorry

end find_divisor_l211_21167


namespace count_k_eq_1006_l211_21199

/-- The number of positive integers k such that (k/2013)(a+b) = lcm(a,b) has a solution in positive integers (a,b) -/
def count_k : ℕ := sorry

/-- The equation (k/2013)(a+b) = lcm(a,b) has a solution in positive integers (a,b) -/
def has_solution (k : ℕ+) : Prop :=
  ∃ (a b : ℕ+), (k : ℚ) / 2013 * (a + b) = Nat.lcm a b

theorem count_k_eq_1006 : count_k = 1006 := by sorry

end count_k_eq_1006_l211_21199


namespace bus_departure_interval_l211_21172

/-- Represents the speed of individual B -/
def speed_B : ℝ := 1

/-- Represents the speed of individual A -/
def speed_A : ℝ := 3 * speed_B

/-- Represents the time interval (in minutes) at which buses overtake A -/
def overtake_time_A : ℝ := 10

/-- Represents the time interval (in minutes) at which buses overtake B -/
def overtake_time_B : ℝ := 6

/-- Represents the speed of the buses -/
def speed_bus : ℝ := speed_A + speed_B

theorem bus_departure_interval (t : ℝ) :
  (t = overtake_time_A ∧ speed_bus * t = speed_A * overtake_time_A + speed_bus * overtake_time_A) ∧
  (t = overtake_time_B ∧ speed_bus * t = speed_B * overtake_time_B + speed_bus * overtake_time_B) →
  t = 5 := by sorry

end bus_departure_interval_l211_21172


namespace pencils_per_row_l211_21108

/-- Theorem: Number of pencils in each row when 6 pencils are equally distributed into 2 rows -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (h1 : total_pencils = 6) (h2 : num_rows = 2) :
  total_pencils / num_rows = 3 := by
  sorry

end pencils_per_row_l211_21108


namespace identity_is_increasing_proportional_l211_21160

/-- A proportional function where y increases as x increases -/
def increasing_proportional_function (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x)

/-- The function f(x) = x is an increasing proportional function -/
theorem identity_is_increasing_proportional : increasing_proportional_function (λ x : ℝ => x) := by
  sorry


end identity_is_increasing_proportional_l211_21160


namespace fraction_simplification_l211_21181

theorem fraction_simplification :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 := by
  sorry

end fraction_simplification_l211_21181


namespace no_real_solutions_l211_21166

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 3*x + 7)^2 + 2 = -|2*x| := by
  sorry

end no_real_solutions_l211_21166


namespace quadratic_inequality_and_distance_comparison_l211_21155

theorem quadratic_inequality_and_distance_comparison :
  (∀ (k : ℝ), (∀ (x : ℝ), 2 * k * x^2 + k * x - 3/8 < 0) ↔ (k > -3 ∧ k ≤ 0)) ∧
  (∀ (a b : ℝ), a ≠ b → |(a^2 + b^2)/2 - ((a+b)/2)^2| > |a*b - ((a+b)/2)^2|) :=
by sorry

end quadratic_inequality_and_distance_comparison_l211_21155


namespace smallest_resolvable_debt_l211_21147

/-- The value of a pig in dollars -/
def pig_value : ℕ := 500

/-- The value of a goat in dollars -/
def goat_value : ℕ := 350

/-- The smallest positive debt that can be resolved -/
def smallest_debt : ℕ := 50

theorem smallest_resolvable_debt :
  smallest_debt = Nat.gcd pig_value goat_value ∧
  ∃ (p g : ℤ), smallest_debt = p * pig_value + g * goat_value :=
sorry

end smallest_resolvable_debt_l211_21147


namespace subset_with_fourth_power_product_l211_21148

theorem subset_with_fourth_power_product 
  (M : Finset ℕ+) 
  (distinct : M.card = 1985) 
  (prime_bound : ∀ n ∈ M, ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 23) :
  ∃ (a b c d : ℕ+), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (m : ℕ+), a * b * c * d = m ^ 4 :=
sorry

end subset_with_fourth_power_product_l211_21148


namespace penny_to_nickel_ratio_l211_21114

/-- Represents the number of coins of each type -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes + 25 * coins.quarters

/-- The main theorem stating the ratio of pennies to nickels -/
theorem penny_to_nickel_ratio (coins : CoinCounts) :
  coins.pennies = 120 ∧
  coins.nickels = 5 * coins.dimes ∧
  coins.quarters = 2 * coins.dimes ∧
  totalValue coins = 800 →
  coins.pennies / coins.nickels = 3 :=
by sorry

end penny_to_nickel_ratio_l211_21114


namespace smallest_sum_of_digits_plus_one_l211_21188

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest possible sum of digits of n+1 is 2, given that the sum of digits of n is 2017 -/
theorem smallest_sum_of_digits_plus_one (n : ℕ) (h : sum_of_digits n = 2017) :
  ∃ m : ℕ, sum_of_digits (n + 1) = 2 ∧ ∀ k : ℕ, sum_of_digits (k + 1) < 2 → sum_of_digits k ≠ 2017 :=
sorry

end smallest_sum_of_digits_plus_one_l211_21188


namespace largest_a_value_l211_21194

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_triangular (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2

def valid_phone_number (a b c d e f g h i j : ℕ) : Prop :=
  a > b ∧ b > c ∧
  d > e ∧ e > f ∧
  g > h ∧ h > i ∧ i > j ∧
  is_square d ∧ is_square e ∧ is_square f ∧
  is_triangular g ∧ is_triangular h ∧ is_triangular i ∧ is_triangular j ∧
  a + b + c = 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem largest_a_value :
  ∀ a b c d e f g h i j : ℕ,
  valid_phone_number a b c d e f g h i j →
  a ≤ 5 :=
by sorry

end largest_a_value_l211_21194


namespace altitude_length_right_triangle_l211_21134

/-- Given a right triangle where the angle bisector divides the hypotenuse into segments
    of lengths p and q, the length of the altitude to the hypotenuse (m) is:
    m = (pq(p+q)) / (p^2 + q^2) -/
theorem altitude_length_right_triangle (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let m := (p * q * (p + q)) / (p^2 + q^2)
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 = b^2 + c^2 ∧
    (b / p = c / q) ∧
    m = (b * c) / a :=
by
  sorry

end altitude_length_right_triangle_l211_21134


namespace cab_driver_average_income_l211_21116

/-- Calculates the average daily income of a cab driver over 5 days --/
theorem cab_driver_average_income :
  let day1_earnings := 250
  let day1_commission_rate := 0.1
  let day2_earnings := 400
  let day2_expense := 50
  let day3_earnings := 750
  let day3_commission_rate := 0.15
  let day4_earnings := 400
  let day4_expense := 40
  let day5_earnings := 500
  let day5_commission_rate := 0.2
  let total_days := 5
  let total_net_income := 
    (day1_earnings * (1 - day1_commission_rate)) +
    (day2_earnings - day2_expense) +
    (day3_earnings * (1 - day3_commission_rate)) +
    (day4_earnings - day4_expense) +
    (day5_earnings * (1 - day5_commission_rate))
  let average_daily_income := total_net_income / total_days
  average_daily_income = 394.50 := by
sorry


end cab_driver_average_income_l211_21116


namespace rowing_distance_l211_21131

/-- Calculates the total distance traveled by a man rowing in a river --/
theorem rowing_distance (v_man : ℝ) (v_river : ℝ) (total_time : ℝ) : 
  v_man = 7 →
  v_river = 1.2 →
  total_time = 1 →
  let d := (v_man^2 - v_river^2) * total_time / (2 * v_man)
  2 * d = 7 := by sorry

end rowing_distance_l211_21131


namespace root_values_l211_21150

theorem root_values (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 + c * k + d = 0)
  (h2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = I ∨ k = -I :=
sorry

end root_values_l211_21150


namespace largest_integer_less_than_85_remainder_2_mod_6_l211_21136

theorem largest_integer_less_than_85_remainder_2_mod_6 : 
  ∃ (n : ℤ), n < 85 ∧ n % 6 = 2 ∧ ∀ (m : ℤ), m < 85 ∧ m % 6 = 2 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_integer_less_than_85_remainder_2_mod_6_l211_21136


namespace lunks_needed_for_bananas_l211_21102

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks_rate : ℚ := 4 / 7

/-- Exchange rate between kunks and bananas -/
def kunks_to_bananas_rate : ℚ := 5 / 3

/-- Number of bananas to purchase -/
def bananas_to_buy : ℕ := 20

/-- Theorem stating the number of lunks needed to buy the specified number of bananas -/
theorem lunks_needed_for_bananas : 
  ⌈(bananas_to_buy : ℚ) / (kunks_to_bananas_rate * lunks_to_kunks_rate)⌉ = 21 := by
  sorry

end lunks_needed_for_bananas_l211_21102


namespace modulo_equivalence_unique_solution_l211_21138

theorem modulo_equivalence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 15827 [ZMOD 12] := by
  sorry

end modulo_equivalence_unique_solution_l211_21138


namespace quadratic_polynomial_proof_l211_21193

theorem quadratic_polynomial_proof (p q : ℝ) : 
  (∃ a b : ℝ, a + b + p + q = 2 ∧ a * b * p * q = 12 ∧ a + b = -p ∧ a * b = q) →
  p = 3 ∧ q = 2 := by
sorry

end quadratic_polynomial_proof_l211_21193


namespace power_difference_solutions_l211_21156

theorem power_difference_solutions :
  ∀ m n : ℕ+,
  (2^(m : ℕ) - 3^(n : ℕ) = 1 ∧ 3^(n : ℕ) - 2^(m : ℕ) = 1) ↔ 
  ((m = 2 ∧ n = 1) ∨ (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 2)) :=
by sorry

end power_difference_solutions_l211_21156


namespace price_reduction_effect_l211_21173

theorem price_reduction_effect (price_reduction : ℝ) (revenue_increase : ℝ) (sales_increase : ℝ) : 
  price_reduction = 30 →
  revenue_increase = 26 →
  (1 - price_reduction / 100) * (1 + sales_increase / 100) = 1 + revenue_increase / 100 →
  sales_increase = 80 := by
sorry

end price_reduction_effect_l211_21173


namespace is_center_of_ellipse_l211_21165

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * x * y + y^2 + 2 * x + 2 * y - 4 = 0

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (0, -1)

/-- Theorem stating that the given point is the center of the ellipse -/
theorem is_center_of_ellipse :
  ∀ (x y : ℝ), ellipse_equation x y →
  ellipse_center = (0, -1) := by sorry

end is_center_of_ellipse_l211_21165


namespace inequality_proof_l211_21124

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ e) (h6 : 0 ≤ f)
  (h7 : a + b ≤ e) (h8 : c + d ≤ f) : 
  Real.sqrt (a * c) + Real.sqrt (b * d) ≤ Real.sqrt (e * f) := by
  sorry

end inequality_proof_l211_21124


namespace inequality_problem_l211_21100

theorem inequality_problem (m : ℝ) (h : ∀ x : ℝ, |x - 2| + |x - 3| ≥ m) :
  (∃ k : ℝ, k = 1 ∧ (∀ m' : ℝ, (∀ x : ℝ, |x - 2| + |x - 3| ≥ m') → m' ≤ k)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/(2*b) + 1/(3*c) = 1 → a + 2*b + 3*c ≥ 9) :=
by sorry

end inequality_problem_l211_21100


namespace binomial_12_6_l211_21169

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_l211_21169


namespace unique_k_for_prime_roots_l211_21132

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the quadratic equation
def quadraticEquation (x k : ℤ) : Prop := x^2 - 99*x + k = 0

-- Define a function to check if both roots are prime
def bothRootsPrime (k : ℤ) : Prop :=
  ∃ p q : ℤ, 
    quadraticEquation p k ∧ 
    quadraticEquation q k ∧ 
    p ≠ q ∧ 
    isPrime p.natAbs ∧ 
    isPrime q.natAbs

-- Theorem statement
theorem unique_k_for_prime_roots : 
  ∃! k : ℤ, bothRootsPrime k :=
sorry

end unique_k_for_prime_roots_l211_21132


namespace parallelogram_area_18_16_l211_21105

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 16 cm is 288 square centimeters -/
theorem parallelogram_area_18_16 : parallelogram_area 18 16 = 288 := by
  sorry

end parallelogram_area_18_16_l211_21105


namespace prime_divides_repunit_iff_l211_21154

/-- A number of the form 111...1 (consisting entirely of the digit '1') -/
def repunit (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem stating that a prime number p is a divisor of some repunit if and only if p ≠ 2 and p ≠ 5 -/
theorem prime_divides_repunit_iff (p : ℕ) (hp : Prime p) :
  (∃ n : ℕ, p ∣ repunit n) ↔ p ≠ 2 ∧ p ≠ 5 := by
  sorry

end prime_divides_repunit_iff_l211_21154


namespace cost_of_five_basketballs_l211_21192

/-- The cost of buying multiple basketballs -/
def cost_of_basketballs (price_per_ball : ℝ) (num_balls : ℕ) : ℝ :=
  price_per_ball * num_balls

/-- Theorem: The cost of 5 basketballs is 5a yuan, given that one basketball costs a yuan -/
theorem cost_of_five_basketballs (a : ℝ) :
  cost_of_basketballs a 5 = 5 * a := by
  sorry

end cost_of_five_basketballs_l211_21192


namespace box_weight_sum_l211_21179

theorem box_weight_sum (a b c d : ℝ) 
  (h1 : a + b + c = 135)
  (h2 : a + b + d = 139)
  (h3 : a + c + d = 142)
  (h4 : b + c + d = 145)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a + b + c + d = 187 := by
  sorry

end box_weight_sum_l211_21179


namespace square_to_rectangle_area_increase_l211_21190

theorem square_to_rectangle_area_increase : 
  ∀ (a : ℝ), a > 0 →
  let original_area := a * a
  let new_length := a * 1.4
  let new_breadth := a * 1.3
  let new_area := new_length * new_breadth
  (new_area - original_area) / original_area = 0.82 := by
sorry

end square_to_rectangle_area_increase_l211_21190


namespace car_distance_problem_l211_21171

/-- Proves that Car X travels 245 miles from when Car Y starts until both stop -/
theorem car_distance_problem :
  let speed_x : ℝ := 35  -- speed of Car X in miles per hour
  let speed_y : ℝ := 41  -- speed of Car Y in miles per hour
  let head_start_time : ℝ := 72 / 60  -- head start time for Car X in hours
  let head_start_distance : ℝ := speed_x * head_start_time  -- distance Car X travels before Car Y starts
  let catch_up_time : ℝ := head_start_distance / (speed_y - speed_x)  -- time it takes for Car Y to catch up
  let distance_x : ℝ := speed_x * catch_up_time  -- distance Car X travels while Car Y is moving
  distance_x = 245 := by
  sorry

end car_distance_problem_l211_21171


namespace coefficient_x6_in_expansion_l211_21177

theorem coefficient_x6_in_expansion : 
  (Finset.range 5).sum (fun k => 
    (Nat.choose 4 k : ℝ) * (1 : ℝ)^(4 - k) * (3 : ℝ)^k * 
    if k = 2 then 1 else 0) = 54 := by sorry

end coefficient_x6_in_expansion_l211_21177


namespace perpendicular_vectors_x_value_l211_21129

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then x else 3
def b : Fin 2 → ℝ := λ i => if i = 0 then 3 else 1

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, dot_product (a x) b = 0 → x = -1 := by sorry

end perpendicular_vectors_x_value_l211_21129


namespace average_leaves_theorem_l211_21130

/-- The number of leaves that fell in the first hour -/
def leaves_first_hour : ℕ := 7

/-- The rate of leaves falling per hour for the second and third hour -/
def leaves_rate_later : ℕ := 4

/-- The total number of hours of observation -/
def total_hours : ℕ := 3

/-- The total number of leaves that fell during the observation period -/
def total_leaves : ℕ := leaves_first_hour + leaves_rate_later * (total_hours - 1)

/-- The average number of leaves falling per hour -/
def average_leaves_per_hour : ℚ := total_leaves / total_hours

theorem average_leaves_theorem : average_leaves_per_hour = 5 := by
  sorry

end average_leaves_theorem_l211_21130


namespace fraction_value_l211_21162

theorem fraction_value : (1 : ℚ) / (4 * 5) = 0.05 := by
  sorry

end fraction_value_l211_21162


namespace paper_towel_package_rolls_l211_21189

/-- Given a package of paper towels with the following properties:
  * The package price is $9
  * The individual roll price is $1
  * The savings per roll in the package is 25% compared to individual purchase
  Prove that the number of rolls in the package is 12 -/
theorem paper_towel_package_rolls : 
  ∀ (package_price individual_price : ℚ) (savings_percent : ℚ) (num_rolls : ℕ),
  package_price = 9 →
  individual_price = 1 →
  savings_percent = 25 / 100 →
  package_price = num_rolls * (individual_price * (1 - savings_percent)) →
  num_rolls = 12 := by
sorry

end paper_towel_package_rolls_l211_21189


namespace car_catch_up_time_l211_21168

/-- The time it takes for a car to catch up with a truck, given their speeds and the truck's head start -/
theorem car_catch_up_time (truck_speed car_speed : ℝ) (head_start : ℝ) : 
  truck_speed = 45 →
  car_speed = 60 →
  head_start = 1 →
  (car_speed * t - truck_speed * t = truck_speed * head_start) →
  t = 3 :=
by
  sorry

end car_catch_up_time_l211_21168


namespace smallest_period_scaled_function_l211_21115

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem smallest_period_scaled_function
  (f : ℝ → ℝ) (h : is_periodic f 10) :
  ∃ b : ℝ, b > 0 ∧ (∀ x, f ((x - b) / 2) = f (x / 2)) ∧
    ∀ b' : ℝ, 0 < b' → (∀ x, f ((x - b') / 2) = f (x / 2)) → b ≤ b' :=
sorry

end smallest_period_scaled_function_l211_21115


namespace quadratic_binomial_square_l211_21112

theorem quadratic_binomial_square (a b : ℝ) : 
  (∃ c d : ℝ, ∀ x : ℝ, 6 * x^2 + 18 * x + a = (c * x + d)^2) ∧
  (∃ c d : ℝ, ∀ x : ℝ, 3 * x^2 + b * x + 4 = (c * x + d)^2) →
  a = 13.5 ∧ b = 18 := by
  sorry

end quadratic_binomial_square_l211_21112


namespace dave_final_tickets_l211_21111

/-- Calculates the number of tickets Dave had left after a series of transactions at the arcade. -/
def dave_tickets_left (initial : ℕ) (won : ℕ) (spent1 : ℕ) (traded : ℕ) (spent2 : ℕ) : ℕ :=
  initial + won - spent1 + traded - spent2

/-- Proves that Dave had 57 tickets left at the end of his arcade visit. -/
theorem dave_final_tickets :
  dave_tickets_left 25 127 84 45 56 = 57 := by
  sorry

end dave_final_tickets_l211_21111


namespace magnitude_of_complex_square_l211_21161

theorem magnitude_of_complex_square : Complex.abs ((3 - 4*Complex.I)^2) = 25 := by
  sorry

end magnitude_of_complex_square_l211_21161


namespace geometric_sequence_seventh_term_l211_21178

/-- Given a geometric sequence {aₙ} with a₁ = 3 and common ratio q = √2, prove that a₇ = 24 -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ), 
    (a 1 = 3) →
    (∀ n : ℕ, a (n + 1) = a n * Real.sqrt 2) →
    (a 7 = 24) := by
  sorry

end geometric_sequence_seventh_term_l211_21178
