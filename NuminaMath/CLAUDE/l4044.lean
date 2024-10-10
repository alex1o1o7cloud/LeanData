import Mathlib

namespace arithmetic_calculations_l4044_404424

theorem arithmetic_calculations :
  ((-0.9) + 1.5 = 0.6) ∧
  (1/2 + (-2/3) = -1/6) ∧
  (1 + (-1/2) + 1/3 + (-1/6) = 2/3) ∧
  (3 + 1/4 + (-2 - 3/5) + 5 + 3/4 + (-8 - 2/5) = -2) := by
  sorry

end arithmetic_calculations_l4044_404424


namespace problem_1_problem_2_l4044_404417

-- Problem 1
theorem problem_1 : 
  (2 + Real.sqrt 3) ^ 0 + 3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 2| + (1/2)⁻¹ = 1 + 2 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (ha : a^2 - 4*a + 3 = 0) (hne : a*(a+3)*(a-3) ≠ 0) : 
  (a^2 - 9) / (a^2 - 3*a) / ((a^2 + 9) / a + 6) = 1/4 := by
  sorry

end problem_1_problem_2_l4044_404417


namespace sams_weight_l4044_404488

/-- Given the weights of Tyler, Sam, and Peter, prove Sam's weight -/
theorem sams_weight (tyler sam peter : ℕ) 
  (h1 : tyler = sam + 25)
  (h2 : peter * 2 = tyler)
  (h3 : peter = 65) : 
  sam = 105 := by sorry

end sams_weight_l4044_404488


namespace min_gennadys_correct_l4044_404477

/-- Represents the number of people with a specific name at the festival -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat
  gennadies : Nat

/-- Checks if the given name counts satisfy the festival conditions -/
def satisfiesConditions (counts : NameCount) : Prop :=
  counts.alexanders = 45 ∧
  counts.borises = 122 ∧
  counts.vasilies = 27 ∧
  counts.alexanders + counts.borises + counts.vasilies + counts.gennadies - 1 ≥ counts.borises

/-- The minimum number of Gennadys required for the festival -/
def minGennadys : Nat := 49

/-- Theorem stating that the minimum number of Gennadys is correct -/
theorem min_gennadys_correct :
  (∀ counts : NameCount, satisfiesConditions counts → counts.gennadies ≥ minGennadys) ∧
  (∃ counts : NameCount, satisfiesConditions counts ∧ counts.gennadies = minGennadys) := by
  sorry

#check min_gennadys_correct

end min_gennadys_correct_l4044_404477


namespace complex_expression_equality_l4044_404495

theorem complex_expression_equality : 
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(2/3) + (1.5)^2 + (Real.sqrt 2 * 43)^4 = 5/4 + 4 * 43^4 := by
  sorry

end complex_expression_equality_l4044_404495


namespace number_count_l4044_404483

theorem number_count (n : ℕ) (S : ℝ) : 
  S / n = 60 →                  -- average of all numbers is 60
  (58 * 6 : ℝ) = S / n * 6 →    -- average of first 6 numbers is 58
  (65 * 6 : ℝ) = S / n * 6 →    -- average of last 6 numbers is 65
  78 = S / n →                  -- 6th number is 78
  n = 11 := by
sorry

end number_count_l4044_404483


namespace polynomial_factorization_l4044_404453

theorem polynomial_factorization (y : ℝ) : 
  (20 * y^4 + 100 * y - 10) - (5 * y^3 - 15 * y + 10) = 5 * (4 * y^4 - y^3 + 23 * y - 4) := by
  sorry

end polynomial_factorization_l4044_404453


namespace x_eq_2_sufficient_not_necessary_l4044_404408

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- The statement that x = 2 is sufficient but not necessary for a ∥ b -/
theorem x_eq_2_sufficient_not_necessary (x : ℝ) :
  (∀ x, x = 2 → are_parallel (1, x - 1) (x + 1, 3)) ∧
  (∃ x, x ≠ 2 ∧ are_parallel (1, x - 1) (x + 1, 3)) := by
  sorry

end x_eq_2_sufficient_not_necessary_l4044_404408


namespace greatest_integer_radius_for_circle_l4044_404434

theorem greatest_integer_radius_for_circle (A : ℝ) (h : A < 75 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ 8 :=
sorry

end greatest_integer_radius_for_circle_l4044_404434


namespace equation_solution_l4044_404419

theorem equation_solution : ∃ x : ℝ, (3 / (x + 2) = 2 / x) ∧ x = 4 := by
  sorry

end equation_solution_l4044_404419


namespace lcm_from_product_and_hcf_l4044_404433

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 145862784)
  (h_hcf : Nat.gcd a b = 792) :
  Nat.lcm a b = 184256 := by
  sorry

end lcm_from_product_and_hcf_l4044_404433


namespace parabola_vertex_x_coordinate_l4044_404494

/-- Given a parabola y = ax^2 + bx + c passing through points (-2, 9), (4, 9), and (5, 13),
    the x-coordinate of its vertex is 1. -/
theorem parabola_vertex_x_coordinate
  (a b c : ℝ)
  (h1 : a * (-2)^2 + b * (-2) + c = 9)
  (h2 : a * 4^2 + b * 4 + c = 9)
  (h3 : a * 5^2 + b * 5 + c = 13) :
  (∃ y : ℝ, a * 1^2 + b * 1 + c = y ∧
    ∀ x : ℝ, a * x^2 + b * x + c ≤ y) :=
by sorry

end parabola_vertex_x_coordinate_l4044_404494


namespace five_power_sum_squares_l4044_404430

/-- A function that checks if a number is expressible as a sum of two squares -/
def is_sum_of_two_squares (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a^2 + b^2

/-- A function that checks if two numbers have the same parity -/
def same_parity (n m : ℕ) : Prop :=
  n % 2 = m % 2

theorem five_power_sum_squares (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  is_sum_of_two_squares (5^m + 5^n) ↔ same_parity n m :=
sorry

end five_power_sum_squares_l4044_404430


namespace factorial_sum_remainder_l4044_404403

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n ≥ 100) :
  sum_factorials n % 30 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 30 := by
  sorry

end factorial_sum_remainder_l4044_404403


namespace count_valid_integers_l4044_404468

/-- A function that returns true if a natural number is a four-digit positive integer -/
def isFourDigitPositive (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that returns true if a natural number is divisible by 25 -/
def isDivisibleBy25 (n : ℕ) : Prop :=
  n % 25 = 0

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- A function that returns true if the sum of digits of a natural number is divisible by 3 -/
def sumOfDigitsDivisibleBy3 (n : ℕ) : Prop :=
  (sumOfDigits n) % 3 = 0

/-- The count of positive four-digit integers divisible by 25 with sum of digits divisible by 3 -/
def countValidIntegers : ℕ :=
  sorry

/-- Theorem stating that the count of valid integers satisfies all conditions -/
theorem count_valid_integers :
  ∃ (n : ℕ), n = countValidIntegers ∧
  ∀ (m : ℕ), (isFourDigitPositive m ∧ isDivisibleBy25 m ∧ sumOfDigitsDivisibleBy3 m) →
  (m ∈ Finset.range n) :=
  sorry

end count_valid_integers_l4044_404468


namespace line_slope_from_y_intercept_l4044_404416

/-- Given a line with equation x + ay + 1 = 0 where a is a real number,
    and y-intercept -2, prove that the slope of the line is -2. -/
theorem line_slope_from_y_intercept (a : ℝ) :
  (∀ x y, x + a * y + 1 = 0 → (x = 0 → y = -2)) →
  ∃ m b, ∀ x y, y = m * x + b ∧ m = -2 :=
sorry

end line_slope_from_y_intercept_l4044_404416


namespace tangent_and_locus_l4044_404400

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (-1, -4)

-- Define point N
def point_N : ℝ × ℝ := (2, 0)

-- Define the tangent line equations
def tangent_line (x y : ℝ) : Prop := x = -1 ∨ 15*x - 8*y - 17 = 0

-- Define the locus of midpoint T
def locus_T (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1 ∧ 0 ≤ x ∧ x < 1/2

theorem tangent_and_locus :
  (∀ x y, circle_O x y → 
    (∃ x' y', tangent_line x' y' ∧ 
      (x' = point_M.1 ∧ y' = point_M.2))) ∧
  (∀ x y, locus_T x y ↔ 
    (∃ p q : ℝ × ℝ, 
      circle_O p.1 p.2 ∧ 
      circle_O q.1 q.2 ∧ 
      (q.2 - point_N.2) * (p.1 - point_N.1) = (q.1 - point_N.1) * (p.2 - point_N.2) ∧
      x = (p.1 + q.1) / 2 ∧ 
      y = (p.2 + q.2) / 2)) :=
by sorry

end tangent_and_locus_l4044_404400


namespace milk_volume_is_ten_l4044_404480

/-- The total volume of milk sold by Josephine -/
def total_milk_volume : ℝ :=
  3 * 2 + 2 * 0.75 + 5 * 0.5

/-- Theorem stating that the total volume of milk sold is 10 liters -/
theorem milk_volume_is_ten : total_milk_volume = 10 := by
  sorry

end milk_volume_is_ten_l4044_404480


namespace isosceles_triangle_perimeter_l4044_404428

-- Define an isosceles triangle with two known side lengths
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b ∨ a = 3 ∨ b = 3

-- Define the perimeter of the triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + 3

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) (h1 : t.a = 3 ∨ t.a = 4) (h2 : t.b = 3 ∨ t.b = 4) :
  perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end isosceles_triangle_perimeter_l4044_404428


namespace additional_chicken_wings_l4044_404499

theorem additional_chicken_wings 
  (num_friends : ℕ) 
  (pre_cooked_wings : ℕ) 
  (wings_per_friend : ℕ) : 
  num_friends = 4 → 
  pre_cooked_wings = 9 → 
  wings_per_friend = 4 → 
  num_friends * wings_per_friend - pre_cooked_wings = 7 := by
  sorry

end additional_chicken_wings_l4044_404499


namespace kendras_family_size_l4044_404432

/-- Proves the number of people in Kendra's family given the cookie baking scenario --/
theorem kendras_family_size 
  (cookies_per_batch : ℕ) 
  (num_batches : ℕ) 
  (chips_per_cookie : ℕ) 
  (chips_per_person : ℕ) 
  (h1 : cookies_per_batch = 12)
  (h2 : num_batches = 3)
  (h3 : chips_per_cookie = 2)
  (h4 : chips_per_person = 18)
  : (cookies_per_batch * num_batches * chips_per_cookie) / chips_per_person = 4 := by
  sorry

#eval (12 * 3 * 2) / 18  -- Should output 4

end kendras_family_size_l4044_404432


namespace units_digit_of_147_25_50_l4044_404435

-- Define a function to calculate the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to calculate the units digit of a power
def unitsDigitOfPower (base : ℕ) (exponent : ℕ) : ℕ :=
  unitsDigit ((unitsDigit base)^exponent)

-- Theorem to prove
theorem units_digit_of_147_25_50 :
  unitsDigitOfPower (unitsDigitOfPower 147 25) 50 = 9 := by
  sorry

end units_digit_of_147_25_50_l4044_404435


namespace x_intercept_of_perpendicular_lines_l4044_404436

/-- Given two lines l₁ and l₂ in the form of linear equations,
    prove that the x-intercept of l₁ is 2 when l₁ is perpendicular to l₂ -/
theorem x_intercept_of_perpendicular_lines
  (a : ℝ)
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ (a + 3) * x + y - 4 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x + (a - 1) * y + 4 = 0)
  (h_perp : (a + 3) * 1 + (a - 1) * 1 = 0) :
  ∃ x, l₁ x 0 ∧ x = 2 := by
sorry

end x_intercept_of_perpendicular_lines_l4044_404436


namespace find_n_l4044_404444

theorem find_n (n : ℕ) : 
  (Nat.lcm n 14 = 56) → (Nat.gcd n 14 = 12) → n = 48 := by
  sorry

end find_n_l4044_404444


namespace adams_purchase_cost_l4044_404413

/-- The cost of Adam's purchases of nuts and dried fruits -/
theorem adams_purchase_cost :
  let nuts_quantity : ℝ := 3
  let dried_fruits_quantity : ℝ := 2.5
  let nuts_price_per_kg : ℝ := 12
  let dried_fruits_price_per_kg : ℝ := 8
  let total_cost : ℝ := nuts_quantity * nuts_price_per_kg + dried_fruits_quantity * dried_fruits_price_per_kg
  total_cost = 56 := by
  sorry

end adams_purchase_cost_l4044_404413


namespace tomatoes_left_l4044_404492

theorem tomatoes_left (total : ℕ) (eaten_fraction : ℚ) (left : ℕ) : 
  total = 21 → 
  eaten_fraction = 1/3 →
  left = total - (total * eaten_fraction).floor →
  left = 14 := by
sorry

end tomatoes_left_l4044_404492


namespace adjacent_different_colors_l4044_404456

/-- Represents a square on the grid -/
structure Square where
  row : Fin 10
  col : Fin 10

/-- Represents the color of a piece -/
inductive Color
  | White
  | Black

/-- Represents the state of the grid at any point in the process -/
def GridState := Square → Option Color

/-- Represents a single step in the replacement process -/
structure ReplacementStep where
  removed : Square
  placed : Square

/-- The sequence of replacement steps -/
def ReplacementSequence := List ReplacementStep

/-- Two squares are adjacent if they share a common edge -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ (s1.col.val + 1 = s2.col.val ∨ s2.col.val + 1 = s1.col.val)) ∨
  (s1.col = s2.col ∧ (s1.row.val + 1 = s2.row.val ∨ s2.row.val + 1 = s1.row.val))

/-- The initial state of the grid with 91 white pieces -/
def initialState : GridState :=
  sorry

/-- The state of the grid after applying a sequence of replacement steps -/
def applyReplacements (initial : GridState) (steps : ReplacementSequence) : GridState :=
  sorry

/-- Theorem: There exists a point in the replacement process where two adjacent squares have different colored pieces -/
theorem adjacent_different_colors (steps : ReplacementSequence) :
  ∃ (partialSteps : ReplacementSequence) (s1 s2 : Square),
    partialSteps.length < steps.length ∧
    adjacent s1 s2 ∧
    let state := applyReplacements initialState partialSteps
    (state s1).isSome ∧ (state s2).isSome ∧ state s1 ≠ state s2 := by
  sorry

end adjacent_different_colors_l4044_404456


namespace last_two_digits_sum_l4044_404422

theorem last_two_digits_sum (n : ℕ) : (8^25 + 12^25) % 100 = 0 := by
  sorry

end last_two_digits_sum_l4044_404422


namespace line_parallel_to_plane_l4044_404427

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (a b : Line) (α : Plane) :
  parallel_line a b → 
  parallel_line_plane b α → 
  ¬ contained_in a α → 
  parallel_line_plane a α :=
sorry

end line_parallel_to_plane_l4044_404427


namespace figure_area_l4044_404467

theorem figure_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) 
  (h1 : rect1_width = 7 ∧ rect1_height = 7)
  (h2 : rect2_width = 3 ∧ rect2_height = 2)
  (h3 : rect3_width = 4 ∧ rect3_height = 4) :
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height = 71 := by
sorry

end figure_area_l4044_404467


namespace cistern_length_is_ten_l4044_404466

/-- Represents a cistern with given dimensions and water level --/
structure Cistern where
  length : ℝ
  width : ℝ
  waterDepth : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterDepth + 2 * c.width * c.waterDepth

/-- Theorem stating that a cistern with given dimensions has a length of 10 meters --/
theorem cistern_length_is_ten :
  ∃ (c : Cistern), c.width = 8 ∧ c.waterDepth = 1.5 ∧ wetSurfaceArea c = 134 → c.length = 10 := by
  sorry

end cistern_length_is_ten_l4044_404466


namespace max_balls_count_l4044_404420

/-- Represents the count of balls -/
def n : ℕ := 45

/-- The number of green balls in the first 45 -/
def initial_green : ℕ := 41

/-- The number of green balls in each subsequent batch of 10 -/
def subsequent_green : ℕ := 9

/-- The total number of balls in each subsequent batch -/
def batch_size : ℕ := 10

/-- The minimum percentage of green balls required -/
def min_green_percentage : ℚ := 92 / 100

theorem max_balls_count :
  ∀ m : ℕ, m > n →
    (initial_green : ℚ) / n < min_green_percentage ∨
    (initial_green + (m - n) / batch_size * subsequent_green : ℚ) / m < min_green_percentage :=
by sorry

end max_balls_count_l4044_404420


namespace ab_equals_six_l4044_404425

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l4044_404425


namespace aria_apple_purchase_l4044_404414

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Aria needs to eat an apple -/
def weeks : ℕ := 2

/-- The number of apples Aria should buy -/
def apples_to_buy : ℕ := days_per_week * weeks

theorem aria_apple_purchase : apples_to_buy = 14 := by
  sorry

end aria_apple_purchase_l4044_404414


namespace complex_expression_simplification_l4044_404461

theorem complex_expression_simplification (i : ℂ) (h : i^2 = -1) :
  i * (1 - i) - 1 = i := by
  sorry

end complex_expression_simplification_l4044_404461


namespace convention_handshakes_l4044_404401

theorem convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : 
  num_companies = 5 → 
  reps_per_company = 4 → 
  (num_companies * reps_per_company * (num_companies * reps_per_company - reps_per_company)) / 2 = 160 := by
sorry

end convention_handshakes_l4044_404401


namespace dave_initial_apps_l4044_404460

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 18

/-- The number of apps remaining after deletion -/
def remaining_apps : ℕ := 5

/-- Theorem stating that Dave initially had 23 apps -/
theorem dave_initial_apps : initial_apps = 23 := by
  sorry

end dave_initial_apps_l4044_404460


namespace original_banana_count_l4044_404462

/-- The number of bananas Willie and Charles originally had together -/
def total_bananas (willie_bananas : ℝ) (charles_bananas : ℝ) : ℝ :=
  willie_bananas + charles_bananas

/-- Theorem stating that Willie and Charles originally had 83.0 bananas together -/
theorem original_banana_count : total_bananas 48.0 35.0 = 83.0 := by
  sorry

end original_banana_count_l4044_404462


namespace not_divides_power_minus_one_l4044_404452

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) :
  ¬(n ∣ (2^n - 1)) := by
sorry

end not_divides_power_minus_one_l4044_404452


namespace custom_op_example_l4044_404421

/-- Custom binary operation ※ -/
def custom_op (a b : ℕ) : ℕ := a + 5 + b * 15

/-- Theorem stating that 105 ※ 5 = 185 -/
theorem custom_op_example : custom_op 105 5 = 185 := by
  sorry

end custom_op_example_l4044_404421


namespace product_of_specific_numbers_l4044_404484

theorem product_of_specific_numbers : 469160 * 9999 = 4690696840 := by
  sorry

end product_of_specific_numbers_l4044_404484


namespace exterior_angle_theorem_l4044_404451

theorem exterior_angle_theorem (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = 150 →      -- Exterior angle is 150°
  γ = 70 →           -- One remote interior angle is 70°
  β = 80 :=          -- The other remote interior angle is 80°
by sorry

end exterior_angle_theorem_l4044_404451


namespace sqrt_fraction_equality_l4044_404474

theorem sqrt_fraction_equality (x : ℝ) : 
  (1 < x ∧ x ≤ 3) ↔ Real.sqrt ((3 - x) / (x - 1)) = Real.sqrt (3 - x) / Real.sqrt (x - 1) :=
by sorry

end sqrt_fraction_equality_l4044_404474


namespace hallies_art_earnings_l4044_404482

/-- Calculates the total money Hallie makes from her art -/
def total_money (prize : ℕ) (num_paintings : ℕ) (price_per_painting : ℕ) : ℕ :=
  prize + num_paintings * price_per_painting

/-- Proves that Hallie's total earnings from her art is $300 -/
theorem hallies_art_earnings : total_money 150 3 50 = 300 := by
  sorry

end hallies_art_earnings_l4044_404482


namespace q_satisfies_conditions_l4044_404411

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (8/5) * x^2 - (18/5) * x - 1/5

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-1) = 5 ∧ q 2 = -1 ∧ q 4 = 11 := by
  sorry

#eval q (-1)
#eval q 2
#eval q 4

end q_satisfies_conditions_l4044_404411


namespace calculation_proofs_l4044_404473

theorem calculation_proofs :
  (7 - (-1/2) + 3/2 = 9) ∧
  ((-1)^99 + (1-5)^2 * (3/8) = 5) ∧
  (-2^3 * (5/8) / (-1/3) - 6 * (2/3 - 1/2) = 14) := by
sorry

end calculation_proofs_l4044_404473


namespace star_op_two_neg_four_l4044_404490

-- Define the * operation for rational numbers
def star_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Theorem statement
theorem star_op_two_neg_four : star_op 2 (-4) = 4 := by sorry

end star_op_two_neg_four_l4044_404490


namespace marys_final_book_count_marys_library_end_year_l4044_404478

/-- Calculates the final number of books in Mary's mystery book library after a year of changes. -/
theorem marys_final_book_count (initial : ℕ) (book_club : ℕ) (bookstore : ℕ) (yard_sales : ℕ) 
  (daughter : ℕ) (mother : ℕ) (donated : ℕ) (sold : ℕ) : ℕ :=
  initial + book_club + bookstore + yard_sales + daughter + mother - donated - sold

/-- Proves that Mary has 81 books at the end of the year given the specific conditions. -/
theorem marys_library_end_year : 
  marys_final_book_count 72 12 5 2 1 4 12 3 = 81 := by
  sorry

end marys_final_book_count_marys_library_end_year_l4044_404478


namespace students_per_group_l4044_404405

theorem students_per_group 
  (total_students : ℕ) 
  (unpicked_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 65) 
  (h2 : unpicked_students = 17) 
  (h3 : num_groups = 8) : 
  (total_students - unpicked_students) / num_groups = 6 := by
  sorry

end students_per_group_l4044_404405


namespace missing_number_proof_l4044_404496

theorem missing_number_proof (numbers : List ℕ) (missing : ℕ) : 
  numbers = [744, 745, 747, 748, 749, 753, 755, 755] →
  (numbers.sum + missing) / 9 = 750 →
  missing = 804 := by
  sorry

end missing_number_proof_l4044_404496


namespace town_population_division_l4044_404440

/-- Proves that in a town with a population of 480, if the population is divided into three equal parts, each part consists of 160 people. -/
theorem town_population_division (total_population : ℕ) (num_parts : ℕ) (part_size : ℕ) : 
  total_population = 480 → 
  num_parts = 3 → 
  total_population = num_parts * part_size → 
  part_size = 160 := by
  sorry

end town_population_division_l4044_404440


namespace angle_325_same_terminal_side_as_neg_35_l4044_404493

/-- 
Given an angle θ in degrees, this function returns true if θ has the same terminal side as -35°.
-/
def hasSameTerminalSideAs (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * 360 + (-35)

/-- 
This theorem states that 325° has the same terminal side as -35° and is between 0° and 360°.
-/
theorem angle_325_same_terminal_side_as_neg_35 :
  hasSameTerminalSideAs 325 ∧ 0 ≤ 325 ∧ 325 < 360 := by
  sorry

end angle_325_same_terminal_side_as_neg_35_l4044_404493


namespace rectangle_perimeter_bound_l4044_404402

/-- Given a unit square covered by m^2 rectangles, there exists a rectangle with perimeter at least 4/m -/
theorem rectangle_perimeter_bound (m : ℝ) (h_m : m > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b ≤ 1 / m^2 ∧ 2 * (a + b) ≥ 4 / m := by
  sorry


end rectangle_perimeter_bound_l4044_404402


namespace toys_between_l4044_404407

theorem toys_between (n : ℕ) (pos_a pos_b : ℕ) (h1 : n = 19) (h2 : pos_a = 9) (h3 : pos_b = 15) :
  pos_b - pos_a - 1 = 5 := by
  sorry

end toys_between_l4044_404407


namespace kamal_math_marks_l4044_404491

/-- Calculates Kamal's marks in Mathematics given his marks in other subjects and his average -/
theorem kamal_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) 
  (h1 : english = 76)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 74) :
  let total := average * 5
  let math := total - (english + physics + chemistry + biology)
  math = 60 := by sorry

end kamal_math_marks_l4044_404491


namespace unique_solution_logarithmic_equation_l4044_404442

theorem unique_solution_logarithmic_equation :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log (x^3 + (1/3) * y^3 + 1/9) = Real.log x + Real.log y := by
  sorry

end unique_solution_logarithmic_equation_l4044_404442


namespace average_of_numbers_l4044_404479

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 114391.81818181818 := by
  sorry

end average_of_numbers_l4044_404479


namespace emily_candy_distribution_l4044_404443

/-- Given that Emily has 34 pieces of candy and 5 friends, prove that she needs to remove 4 pieces
    to distribute the remaining candies equally among her friends. -/
theorem emily_candy_distribution (total_candy : Nat) (num_friends : Nat) 
    (h1 : total_candy = 34) (h2 : num_friends = 5) :
    ∃ (removed : Nat) (distributed : Nat),
      removed = 4 ∧
      distributed * num_friends = total_candy - removed ∧
      ∀ r, r < removed → ¬∃ d, d * num_friends = total_candy - r :=
by sorry

end emily_candy_distribution_l4044_404443


namespace order_of_expressions_l4044_404455

theorem order_of_expressions :
  let a := 2 + (1/5) * Real.log 2
  let b := 1 + 2^(1/5)
  let c := 2^(11/10)
  a < c ∧ c < b := by
  sorry

end order_of_expressions_l4044_404455


namespace combined_mpg_calculation_l4044_404429

/-- Calculates the combined miles per gallon for three cars given their individual efficiencies and a common distance traveled. -/
def combinedMPG (ray_mpg tom_mpg amy_mpg distance : ℚ) : ℚ :=
  let total_distance := 3 * distance
  let total_gas := distance / ray_mpg + distance / tom_mpg + distance / amy_mpg
  total_distance / total_gas

/-- Theorem stating that the combined MPG for the given conditions is 3600/114 -/
theorem combined_mpg_calculation :
  combinedMPG 50 20 40 120 = 3600 / 114 := by
  sorry

#eval combinedMPG 50 20 40 120

end combined_mpg_calculation_l4044_404429


namespace solve_movie_problem_l4044_404487

def movie_problem (rented_movie_cost bought_movie_cost total_spent : ℚ) : Prop :=
  let num_tickets : ℕ := 2
  let other_costs : ℚ := rented_movie_cost + bought_movie_cost
  let ticket_total_cost : ℚ := total_spent - other_costs
  let ticket_cost : ℚ := ticket_total_cost / num_tickets
  ticket_cost = 10.62

theorem solve_movie_problem :
  movie_problem 1.59 13.95 36.78 := by
  sorry

end solve_movie_problem_l4044_404487


namespace dice_probability_l4044_404441

def num_dice : ℕ := 5
def dice_sides : ℕ := 6

def prob_all_same : ℚ := 1 / (dice_sides ^ (num_dice - 1))

def prob_four_same : ℚ := 
  (num_dice * (1 / dice_sides ^ (num_dice - 2)) * ((dice_sides - 1) / dice_sides))

theorem dice_probability : 
  prob_all_same + prob_four_same = 13 / 648 :=
sorry

end dice_probability_l4044_404441


namespace fraction_simplification_l4044_404445

theorem fraction_simplification (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  (1 / y) / (1 / x) = 2 / 3 := by
  sorry

end fraction_simplification_l4044_404445


namespace marble_weight_l4044_404431

theorem marble_weight (marble_weight : ℚ) (car_weight : ℚ) : 
  9 * marble_weight = 5 * car_weight →
  4 * car_weight = 120 →
  marble_weight = 50 / 3 := by
  sorry

end marble_weight_l4044_404431


namespace simplify_fraction_sum_l4044_404476

theorem simplify_fraction_sum (a b c d : ℕ) : 
  a = 75 → b = 135 → 
  (∃ (k : ℕ), k * c = a ∧ k * d = b) → 
  (∀ (m : ℕ), m * c = a ∧ m * d = b → m ≤ k) →
  c + d = 14 := by
  sorry

end simplify_fraction_sum_l4044_404476


namespace multiplication_and_division_results_l4044_404415

theorem multiplication_and_division_results : 
  ((-2) * (-1/8) = 1/4) ∧ ((-5) / (6/5) = -25/6) := by sorry

end multiplication_and_division_results_l4044_404415


namespace rope_cutting_problem_l4044_404423

theorem rope_cutting_problem :
  let total_length_feet : ℝ := 6
  let number_of_pieces : ℕ := 10
  let inches_per_foot : ℝ := 12
  let piece_length_inches : ℝ := total_length_feet * inches_per_foot / number_of_pieces
  piece_length_inches = 7.2 := by sorry

end rope_cutting_problem_l4044_404423


namespace count_triples_sum_8_l4044_404449

/-- The number of ordered triples of natural numbers that sum to a given natural number. -/
def count_triples (n : ℕ) : ℕ := Nat.choose (n + 2) 2

/-- Theorem stating that the number of ordered triples (A, B, C) of natural numbers
    that satisfy A + B + C = 8 is equal to 21. -/
theorem count_triples_sum_8 : count_triples 8 = 21 := by
  sorry

end count_triples_sum_8_l4044_404449


namespace moving_circle_trajectory_l4044_404463

-- Define the circles M₁ and M₂
def M₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def M₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory of the center of the moving circle M
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1 ∧ x ≠ -2

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
    (∃ (R : ℝ), 
      (∀ (x₁ y₁ : ℝ), M₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + R)^2) ∧
      (∀ (x₂ y₂ : ℝ), M₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (3 - R)^2)) →
    trajectory x y :=
by sorry

end moving_circle_trajectory_l4044_404463


namespace lucas_100_mod_5_l4044_404485

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas (n + 1) + lucas n

theorem lucas_100_mod_5 : lucas 99 % 5 = 2 := by
  sorry

end lucas_100_mod_5_l4044_404485


namespace dogwood_trees_planted_tomorrow_l4044_404446

theorem dogwood_trees_planted_tomorrow 
  (initial_trees : ℕ) 
  (planted_today : ℕ) 
  (final_total : ℕ) :
  initial_trees = 7 →
  planted_today = 3 →
  final_total = 12 →
  final_total - (initial_trees + planted_today) = 2 :=
by sorry

end dogwood_trees_planted_tomorrow_l4044_404446


namespace chord_length_at_specific_angle_shortest_chord_equation_l4044_404437

-- Define the circle
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 8}

-- Define point P0
def P0 : ℝ × ℝ := (-1, 2)

-- Define a chord AB passing through P0
def chord (α : ℝ) : Set (ℝ × ℝ) := {p | p.2 - P0.2 = Real.tan α * (p.1 - P0.1)}

-- Define the length of a chord
def chordLength (α : ℝ) : ℝ := sorry

-- Theorem 1
theorem chord_length_at_specific_angle :
  chordLength (3 * Real.pi / 4) = Real.sqrt 30 := by sorry

-- Define the shortest chord
def shortestChord : Set (ℝ × ℝ) := sorry

-- Theorem 2
theorem shortest_chord_equation :
  shortestChord = {p | p.1 - 2 * p.2 + 5 = 0} := by sorry

end chord_length_at_specific_angle_shortest_chord_equation_l4044_404437


namespace sam_tuesday_letters_l4044_404469

-- Define the number of days
def num_days : ℕ := 2

-- Define the average number of letters per day
def average_letters : ℕ := 5

-- Define the number of letters written on Wednesday
def wednesday_letters : ℕ := 3

-- Define the function to calculate the number of letters written on Tuesday
def tuesday_letters : ℕ := num_days * average_letters - wednesday_letters

-- Theorem statement
theorem sam_tuesday_letters :
  tuesday_letters = 7 := by sorry

end sam_tuesday_letters_l4044_404469


namespace boys_camp_total_l4044_404418

theorem boys_camp_total (total : ℝ) 
  (school_a_percentage : total * 0.2 = total * (20 / 100))
  (science_percentage : (total * 0.2) * 0.3 = (total * 0.2) * (30 / 100))
  (non_science_count : (total * 0.2) * 0.7 = 49) : 
  total = 350 := by
  sorry

end boys_camp_total_l4044_404418


namespace matrix_equation_solution_l4044_404447

theorem matrix_equation_solution :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 2 • M = !![8, 16; 4, 8] := by
  sorry

end matrix_equation_solution_l4044_404447


namespace total_amount_proof_l4044_404439

/-- Proves that the total amount of money divided into two parts is Rs. 2600 -/
theorem total_amount_proof (total : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (income : ℝ) :
  part1 + part2 = total →
  part1 = 1600 →
  rate1 = 0.05 →
  rate2 = 0.06 →
  part1 * rate1 + part2 * rate2 = income →
  income = 140 →
  total = 2600 := by
  sorry

#check total_amount_proof

end total_amount_proof_l4044_404439


namespace x_seventh_x_n_plus_one_l4044_404489

variable (x : ℝ)

-- Define the conditions
axiom x_is_root : x^2 - x - 1 = 0
axiom x_squared : x^2 = x + 1
axiom x_cubed : x^3 = 2*x + 1
axiom x_fourth : x^4 = 3*x + 2
axiom x_fifth : x^5 = 5*x + 3
axiom x_sixth : x^6 = 8*x + 5

-- Define x^n = αx + β
variable (n : ℕ) (α β : ℝ)
axiom x_nth : x^n = α*x + β

-- Theorem statements
theorem x_seventh : x^7 = 13*x + 8 := by sorry

theorem x_n_plus_one : x^(n+1) = (α + β)*x + α := by sorry

end x_seventh_x_n_plus_one_l4044_404489


namespace geometric_sequence_theorem_l4044_404459

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

/-- Given conditions for the geometric sequence -/
def satisfies_conditions (seq : GeometricSequence) : Prop :=
  seq.a 3 + seq.a 6 = 36 ∧ seq.a 4 + seq.a 7 = 18

theorem geometric_sequence_theorem (seq : GeometricSequence) 
  (h : satisfies_conditions seq) : 
  ∃ n : ℕ, seq.a n = 1/2 ∧ n = 9 :=
sorry

end geometric_sequence_theorem_l4044_404459


namespace binomial_16_12_l4044_404406

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l4044_404406


namespace train_speed_calculation_l4044_404458

/-- Calculates the speed of trains given their length, crossing time, and direction --/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 →
  crossing_time = 12 →
  (2 * train_length) / crossing_time * 3.6 = 36 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l4044_404458


namespace equation_solution_l4044_404464

theorem equation_solution :
  let f (x : ℝ) := (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2
  ∀ x : ℝ, x ≠ 2 → (f x = 0 ↔ x = (5 + Real.sqrt 5) / 3 ∨ x = (5 - Real.sqrt 5) / 3) :=
by sorry

end equation_solution_l4044_404464


namespace total_cups_is_twenty_l4044_404498

/-- Represents the number of cups of tea drunk by each merchant -/
structure Merchants where
  sosipatra : ℕ
  olympiada : ℕ
  poliksena : ℕ

/-- Defines the conditions given in the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  m.sosipatra + m.olympiada = 11 ∧
  m.olympiada + m.poliksena = 15 ∧
  m.sosipatra + m.poliksena = 14

/-- Theorem stating that the total number of cups is 20 -/
theorem total_cups_is_twenty (m : Merchants) (h : satisfies_conditions m) :
  m.sosipatra + m.olympiada + m.poliksena = 20 := by
  sorry

end total_cups_is_twenty_l4044_404498


namespace sequence_divisibility_l4044_404486

theorem sequence_divisibility (n : ℤ) : 
  (∃ k : ℤ, 7 * n - 3 = 5 * k) ∧ 
  (∀ m : ℤ, 7 * n - 3 ≠ 3 * m) ↔ 
  ∃ t : ℕ, n = 5 * t - 1 ∧ ∀ m : ℕ, t ≠ 3 * m - 1 := by
  sorry

end sequence_divisibility_l4044_404486


namespace similar_squares_side_length_l4044_404471

theorem similar_squares_side_length (small_side : ℝ) (area_ratio : ℝ) :
  small_side = 4 →
  area_ratio = 9 →
  ∃ large_side : ℝ,
    large_side = small_side * Real.sqrt area_ratio ∧
    large_side = 12 := by
  sorry

end similar_squares_side_length_l4044_404471


namespace sixth_group_frequency_l4044_404450

/-- Given a sample of 40 data points divided into 6 groups, with the frequencies
    of the first four groups and the fifth group as specified, 
    the frequency of the sixth group is 0.2. -/
theorem sixth_group_frequency 
  (total_points : ℕ) 
  (group_count : ℕ)
  (freq_1 freq_2 freq_3 freq_4 freq_5 : ℚ) :
  total_points = 40 →
  group_count = 6 →
  freq_1 = 10 / 40 →
  freq_2 = 5 / 40 →
  freq_3 = 7 / 40 →
  freq_4 = 6 / 40 →
  freq_5 = 1 / 10 →
  ∃ freq_6 : ℚ, freq_6 = 1 - (freq_1 + freq_2 + freq_3 + freq_4 + freq_5) ∧ freq_6 = 1 / 5 :=
by sorry

end sixth_group_frequency_l4044_404450


namespace greatest_prime_factor_of_144_l4044_404404

theorem greatest_prime_factor_of_144 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 144 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 144 → q ≤ p :=
by sorry

end greatest_prime_factor_of_144_l4044_404404


namespace sum_of_ratios_l4044_404470

def is_multiplicative (f : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m + n) = f m * f n

theorem sum_of_ratios (f : ℕ → ℝ) (h_mult : is_multiplicative f) (h_f1 : f 1 = 2) :
  (Finset.range 2010).sum (λ i => f (i + 2) / f (i + 1)) = 4020 := by
  sorry

end sum_of_ratios_l4044_404470


namespace geometric_ratio_in_arithmetic_sequence_l4044_404409

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- State the theorem
theorem geometric_ratio_in_arithmetic_sequence
  (a₁ d : ℝ) (h : d ≠ 0) :
  let a := arithmetic_sequence a₁ d
  (a 2) * (a 6) = (a 3)^2 →
  (a 3) / (a 2) = 3 :=
by sorry

end geometric_ratio_in_arithmetic_sequence_l4044_404409


namespace exponent_rule_multiplication_l4044_404448

theorem exponent_rule_multiplication (a : ℝ) : a^4 * a^6 = a^10 := by sorry

end exponent_rule_multiplication_l4044_404448


namespace zacks_marbles_l4044_404454

theorem zacks_marbles : ∃ (M : ℕ), 
  (∃ (k : ℕ), M - 5 = 3 * k) ∧ 
  (M - (3 * 20) - 5 = 5) ∧ 
  M = 70 := by
  sorry

end zacks_marbles_l4044_404454


namespace ellipse_equation_and_product_constant_l4044_404481

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

-- Define the x-intercept of line QM
def x_intercept_QM (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₂ * y₁ - x₁ * y₂) / (y₁ + y₂)

-- Define the y-intercept of line QN
def y_intercept_QN (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ * y₂ + x₂ * y₁) / (x₁ - x₂)

-- Define the slope of line OR
def slope_OR (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₁ + y₂) / (x₁ + x₂)

theorem ellipse_equation_and_product_constant (a b : ℝ) 
  (h₁ : a > b) (h₂ : b > 0) (h₃ : eccentricity a b = 1/2) :
  (∃ (x y : ℝ), Ellipse 2 (Real.sqrt 3) (x, y)) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → y₁ ≠ y₂ → 
    Ellipse a b (x₁, y₁) → Ellipse a b (x₂, y₂) →
    (x_intercept_QM x₁ y₁ x₂ y₂) * (y_intercept_QN x₁ y₁ x₂ y₂) * (slope_OR x₁ y₁ x₂ y₂) = 0) := by
  sorry

end ellipse_equation_and_product_constant_l4044_404481


namespace john_bought_490_packs_l4044_404412

/-- The number of packs John buys for each student -/
def packsPerStudent : ℕ := 4

/-- The number of extra packs John purchases for supplies -/
def extraPacks : ℕ := 10

/-- The number of students in each class -/
def studentsPerClass : List ℕ := [24, 18, 30, 20, 28]

/-- The total number of packs John bought -/
def totalPacks : ℕ := 
  (studentsPerClass.map (· * packsPerStudent)).sum + extraPacks

theorem john_bought_490_packs : totalPacks = 490 := by
  sorry

end john_bought_490_packs_l4044_404412


namespace power_two_2014_mod_7_l4044_404426

theorem power_two_2014_mod_7 :
  ∃ (k : ℤ), 2^2014 = 7 * k + 9 := by sorry

end power_two_2014_mod_7_l4044_404426


namespace quadratic_vertex_form_l4044_404475

theorem quadratic_vertex_form (x : ℝ) : ∃ (a h k : ℝ), 
  3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end quadratic_vertex_form_l4044_404475


namespace fraction_sum_proof_l4044_404410

theorem fraction_sum_proof : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_sum_proof_l4044_404410


namespace rotation_implies_equilateral_l4044_404457

-- Define the triangle
variable (A₁ A₂ A₃ : ℝ × ℝ)

-- Define the rotation function
def rotate (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the sequence of rotations
def rotate_sequence (n : ℕ) (P₀ : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define equilateral triangle
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  sorry

theorem rotation_implies_equilateral 
  (P₀ : ℝ × ℝ) 
  (h : rotate_sequence 1986 P₀ = P₀) : 
  is_equilateral A₁ A₂ A₃ :=
sorry

end rotation_implies_equilateral_l4044_404457


namespace airplane_passengers_l4044_404438

theorem airplane_passengers (total : ℕ) (children : ℕ) (h1 : total = 80) (h2 : children = 20) :
  let adults := total - children
  let men := adults / 2
  men = 30 := by
  sorry

end airplane_passengers_l4044_404438


namespace tan_ratio_from_sin_sum_diff_l4044_404465

theorem tan_ratio_from_sin_sum_diff (p q : Real) 
  (h1 : Real.sin (p + q) = 0.6) 
  (h2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := by
  sorry

end tan_ratio_from_sin_sum_diff_l4044_404465


namespace belt_and_road_population_l4044_404472

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem belt_and_road_population : 
  toScientificNotation 4400000000 = ScientificNotation.mk 4.4 9 (by norm_num) :=
sorry

end belt_and_road_population_l4044_404472


namespace function_composition_identity_l4044_404497

/-- Given a function f(x) = (2ax - b) / (3cx + d) where b ≠ 0, d ≠ 0, abcd ≠ 0,
    and f(f(x)) = x for all x in the domain of f, there exist real numbers b and c
    such that 3a - 2d = -4.5c - 4b -/
theorem function_composition_identity (a b c d : ℝ) : 
  b ≠ 0 → d ≠ 0 → a * b * c * d ≠ 0 → 
  (∀ x, (2 * a * ((2 * a * x - b) / (3 * c * x + d)) - b) / 
        (3 * c * ((2 * a * x - b) / (3 * c * x + d)) + d) = x) →
  ∃ (b c : ℝ), 3 * a - 2 * d = -4.5 * c - 4 * b :=
by sorry

end function_composition_identity_l4044_404497
