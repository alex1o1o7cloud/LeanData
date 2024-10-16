import Mathlib

namespace NUMINAMATH_CALUDE_x_plus_y_value_l465_46564

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l465_46564


namespace NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l465_46547

theorem ellipse_max_y_coordinate :
  let ellipse := {(x, y) : ℝ × ℝ | (x^2 / 49) + ((y - 3)^2 / 25) = 1}
  ∃ (y_max : ℝ), y_max = 8 ∧ ∀ (x y : ℝ), (x, y) ∈ ellipse → y ≤ y_max :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l465_46547


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l465_46596

theorem scientific_notation_equality : 
  122254 = 1.22254 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l465_46596


namespace NUMINAMATH_CALUDE_find_k_l465_46597

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity of e₁ and e₂
variable (h_non_collinear : ∀ (r : ℝ), e₁ ≠ r • e₂)

-- Define the vectors AB, CD, and CB
variable (k : ℝ)
def AB := 2 • e₁ + k • e₂
def CD := 2 • e₁ - 1 • e₂
def CB := 1 • e₁ + 3 • e₂

-- Define collinearity of A, B, and D
def collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

-- State the theorem
theorem find_k : 
  collinear (AB e₁ e₂ k) (CD e₁ e₂ - CB e₁ e₂) → k = -8 :=
sorry

end NUMINAMATH_CALUDE_find_k_l465_46597


namespace NUMINAMATH_CALUDE_least_cans_proof_l465_46549

/-- The number of liters of Maaza -/
def maaza_liters : ℕ := 50

/-- The number of liters of Pepsi -/
def pepsi_liters : ℕ := 144

/-- The number of liters of Sprite -/
def sprite_liters : ℕ := 368

/-- The least number of cans required to pack all drinks -/
def least_cans : ℕ := 281

/-- Theorem stating that the least number of cans required is 281 -/
theorem least_cans_proof :
  ∃ (can_size : ℕ), can_size > 0 ∧
  maaza_liters % can_size = 0 ∧
  pepsi_liters % can_size = 0 ∧
  sprite_liters % can_size = 0 ∧
  least_cans = maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size ∧
  ∀ (other_size : ℕ), other_size > 0 →
    maaza_liters % other_size = 0 →
    pepsi_liters % other_size = 0 →
    sprite_liters % other_size = 0 →
    least_cans ≤ maaza_liters / other_size + pepsi_liters / other_size + sprite_liters / other_size :=
by
  sorry

end NUMINAMATH_CALUDE_least_cans_proof_l465_46549


namespace NUMINAMATH_CALUDE_gavin_green_shirts_l465_46539

/-- The number of green shirts Gavin has -/
def num_green_shirts (total_shirts blue_shirts : ℕ) : ℕ :=
  total_shirts - blue_shirts

/-- Theorem stating that Gavin has 17 green shirts -/
theorem gavin_green_shirts : 
  num_green_shirts 23 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gavin_green_shirts_l465_46539


namespace NUMINAMATH_CALUDE_floor_sqrt_15_plus_1_squared_l465_46560

theorem floor_sqrt_15_plus_1_squared : (⌊Real.sqrt 15⌋ + 1)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_15_plus_1_squared_l465_46560


namespace NUMINAMATH_CALUDE_area_of_rotated_squares_l465_46528

/-- Represents a square sheet of paper -/
structure Square :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping squares -/
structure OverlappingSquares :=
  (base : Square)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)

/-- Calculates the area of the 24-sided polygon formed by the overlapping squares -/
def area_of_polygon (config : OverlappingSquares) : ℝ :=
  sorry

theorem area_of_rotated_squares :
  let config := OverlappingSquares.mk (Square.mk 8) (20 * π / 180) (45 * π / 180)
  area_of_polygon config = 192 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rotated_squares_l465_46528


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l465_46548

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l465_46548


namespace NUMINAMATH_CALUDE_product_abcd_l465_46553

theorem product_abcd (a b c d : ℚ) : 
  (3 * a + 2 * b + 4 * c + 6 * d = 42) →
  (4 * d + 2 * c = b) →
  (4 * b - 2 * c = a) →
  (d + 2 = c) →
  (a * b * c * d = -(5 * 83 * 46 * 121) / (44 * 44 * 11 * 11)) := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l465_46553


namespace NUMINAMATH_CALUDE_square_fencing_cost_l465_46525

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The cost of fencing each side of the square in dollars -/
def cost_per_side : ℕ := 79

/-- The total cost of fencing a square -/
def total_cost : ℕ := square_sides * cost_per_side

theorem square_fencing_cost : total_cost = 316 := by
  sorry

end NUMINAMATH_CALUDE_square_fencing_cost_l465_46525


namespace NUMINAMATH_CALUDE_system_solution_approximation_l465_46518

/-- Proof that the solution to the system of equations 4x - 6y = -2 and 5x + 3y = 2.6
    is approximately (0.4571, 0.1048) -/
theorem system_solution_approximation :
  ∃ (x y : ℝ), 
    (4 * x - 6 * y = -2) ∧ 
    (5 * x + 3 * y = 2.6) ∧ 
    (abs (x - 0.4571) < 0.0001) ∧ 
    (abs (y - 0.1048) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_approximation_l465_46518


namespace NUMINAMATH_CALUDE_smallest_odd_five_primes_l465_46565

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_five_prime_factors (n : ℕ) : Prop := sorry

def smallest_odd_with_five_prime_factors : ℕ := 15015

theorem smallest_odd_five_primes :
  has_exactly_five_prime_factors smallest_odd_with_five_prime_factors ∧
  ∀ m : ℕ, m < smallest_odd_with_five_prime_factors →
    ¬(has_exactly_five_prime_factors m ∧ Odd m) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_five_primes_l465_46565


namespace NUMINAMATH_CALUDE_base6_154_to_decimal_l465_46507

/-- Converts a list of digits in base 6 to its decimal (base 10) representation -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem base6_154_to_decimal :
  base6ToDecimal [4, 5, 1] = 70 := by
  sorry

#eval base6ToDecimal [4, 5, 1]

end NUMINAMATH_CALUDE_base6_154_to_decimal_l465_46507


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l465_46588

theorem weight_loss_challenge (original_weight : ℝ) (x : ℝ) : 
  x > 0 →
  x < 100 →
  let final_weight := original_weight * (1 - x / 100 + 2 / 100)
  let measured_loss_percentage := 13.3
  final_weight = original_weight * (1 - measured_loss_percentage / 100) →
  x = 15.3 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l465_46588


namespace NUMINAMATH_CALUDE_line_arrangement_with_restriction_l465_46572

def number_of_students : ℕ := 5

def total_arrangements (n : ℕ) : ℕ := n.factorial

def restricted_arrangements (n : ℕ) : ℕ := 
  (n - 1).factorial * 2

theorem line_arrangement_with_restriction :
  total_arrangements number_of_students - restricted_arrangements number_of_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangement_with_restriction_l465_46572


namespace NUMINAMATH_CALUDE_max_xy_perpendicular_vectors_l465_46561

theorem max_xy_perpendicular_vectors (x y : ℝ) :
  let a : ℝ × ℝ := (1, x - 1)
  let b : ℝ × ℝ := (y, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  ∃ (m : ℝ), (∀ (x' y' : ℝ), 
    let a' : ℝ × ℝ := (1, x' - 1)
    let b' : ℝ × ℝ := (y', 2)
    (a'.1 * b'.1 + a'.2 * b'.2 = 0) → x' * y' ≤ m) ∧
  m = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_xy_perpendicular_vectors_l465_46561


namespace NUMINAMATH_CALUDE_square_sum_and_product_l465_46545

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 4) 
  (h2 : (x + y)^2 = 64) : 
  x^2 + y^2 = 34 ∧ x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l465_46545


namespace NUMINAMATH_CALUDE_second_price_reduction_l465_46570

theorem second_price_reduction (P : ℝ) (x : ℝ) (h1 : P > 0) :
  (P - 0.25 * P) * (1 - x / 100) = P * (1 - 0.7) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_price_reduction_l465_46570


namespace NUMINAMATH_CALUDE_absolute_value_multiplication_l465_46519

theorem absolute_value_multiplication : -2 * |(-3)| = -6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_multiplication_l465_46519


namespace NUMINAMATH_CALUDE_trapezoid_areas_l465_46505

/-- Represents a trapezoid with given dimensions and a parallel line through the intersection of diagonals -/
structure Trapezoid :=
  (ad : ℝ) -- Length of base AD
  (bc : ℝ) -- Length of base BC
  (ab : ℝ) -- Length of side AB
  (cd : ℝ) -- Length of side CD

/-- Calculates the areas of the two resulting trapezoids formed by a line parallel to the bases through the diagonal intersection point -/
def calculate_areas (t : Trapezoid) : ℝ × ℝ := sorry

/-- Theorem stating the areas of the resulting trapezoids for the given dimensions -/
theorem trapezoid_areas (t : Trapezoid) 
  (h1 : t.ad = 84) (h2 : t.bc = 42) (h3 : t.ab = 39) (h4 : t.cd = 45) : 
  calculate_areas t = (588, 1680) := by sorry

end NUMINAMATH_CALUDE_trapezoid_areas_l465_46505


namespace NUMINAMATH_CALUDE_dogsled_race_speed_difference_l465_46534

/-- Proves that the difference in average speeds between two teams is 5 mph
    given specific conditions of a dogsled race. -/
theorem dogsled_race_speed_difference
  (course_length : ℝ)
  (team_r_speed : ℝ)
  (time_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : team_r_speed = 20)
  (h3 : time_difference = 3)
  : ∃ (team_a_speed : ℝ),
    team_a_speed = course_length / (course_length / team_r_speed - time_difference) ∧
    team_a_speed - team_r_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_difference_l465_46534


namespace NUMINAMATH_CALUDE_calculate_savings_l465_46598

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
theorem calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income_ratio > 0 ∧ expenditure_ratio > 0 ∧ income = 18000 ∧ income_ratio = 5 ∧ expenditure_ratio = 4 →
  income - (income * expenditure_ratio / income_ratio) = 3600 := by
sorry

end NUMINAMATH_CALUDE_calculate_savings_l465_46598


namespace NUMINAMATH_CALUDE_inequality_theorem_l465_46552

/-- A function f: ℝ⁺ → ℝ⁺ such that f(x)/x is increasing on ℝ⁺ -/
def IncreasingRatioFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x < y → (f x) / x < (f y) / y

theorem inequality_theorem (f : ℝ → ℝ) (h : IncreasingRatioFunction f) 
    (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
  3 * ((f a + f b + f c) / (a + b + c)) + f a / a + f b / b + f c / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l465_46552


namespace NUMINAMATH_CALUDE_problem_solution_l465_46509

def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) ∧
  (∀ k : ℝ, (∃ x : ℝ, (f 2 x + f 2 (-x)) / 3 < |k|) ↔ k < -2/3 ∨ k > 2/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l465_46509


namespace NUMINAMATH_CALUDE_error_arrangement_probability_l465_46503

/-- The number of letters in the word "error" -/
def word_length : Nat := 5

/-- The number of 'r's in the word "error" -/
def num_r : Nat := 3

/-- The number of ways to arrange the letters in "error" -/
def total_arrangements : Nat := 20

/-- The probability of incorrectly arranging the letters in "error" -/
def incorrect_probability : Rat := 19 / 20

/-- Theorem stating that the probability of incorrectly arranging the letters in "error" is 19/20 -/
theorem error_arrangement_probability :
  incorrect_probability = 19 / 20 :=
by sorry

end NUMINAMATH_CALUDE_error_arrangement_probability_l465_46503


namespace NUMINAMATH_CALUDE_fraction_simplification_l465_46510

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) : (x^2 - y^2) / (x - y) = x + y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l465_46510


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l465_46558

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_a4 : a 4 = 4) :
  a 2 * a 6 = a 4 * a 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l465_46558


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l465_46513

theorem max_ratio_two_digit_integers (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 →  -- x and y are two-digit positive integers
  (x + y) / 2 = 55 →                     -- mean of x and y is 55
  ∃ (z : ℕ), x * y = z ^ 2 →             -- product xy is a square number
  ∀ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
    (a + b) / 2 = 55 ∧
    (∃ (w : ℕ), a * b = w ^ 2) →
    x / y ≥ a / b →
  x / y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l465_46513


namespace NUMINAMATH_CALUDE_square_root_difference_l465_46592

theorem square_root_difference : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_l465_46592


namespace NUMINAMATH_CALUDE_dime_nickel_difference_l465_46514

/-- Proves that given 70 cents total and 2 nickels, the number of dimes exceeds the number of nickels by 4 -/
theorem dime_nickel_difference :
  ∀ (total_cents : ℕ) (num_nickels : ℕ) (nickel_value : ℕ) (dime_value : ℕ),
    total_cents = 70 →
    num_nickels = 2 →
    nickel_value = 5 →
    dime_value = 10 →
    ∃ (num_dimes : ℕ),
      num_dimes * dime_value + num_nickels * nickel_value = total_cents ∧
      num_dimes = num_nickels + 4 := by
  sorry

end NUMINAMATH_CALUDE_dime_nickel_difference_l465_46514


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l465_46533

theorem cubic_roots_problem (a b c : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : a + b + c = -1)
  (h4 : a * b + b * c + a * c = -4)
  (h5 : a * b * c = -2) : 
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l465_46533


namespace NUMINAMATH_CALUDE_rectangle_count_l465_46582

theorem rectangle_count (h v : ℕ) (h_eq : h = 5) (v_eq : v = 5) :
  (Nat.choose h 2) * (Nat.choose v 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l465_46582


namespace NUMINAMATH_CALUDE_tape_length_for_circular_base_l465_46559

/-- The length of tape needed for a circular lamp base -/
theorem tape_length_for_circular_base :
  let area : ℝ := 176
  let π_approx : ℝ := 22 / 7
  let extra_tape : ℝ := 3
  let radius : ℝ := Real.sqrt (area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let total_length : ℝ := circumference + extra_tape
  ∃ ε > 0, abs (total_length - 50.058) < ε :=
by sorry

end NUMINAMATH_CALUDE_tape_length_for_circular_base_l465_46559


namespace NUMINAMATH_CALUDE_profit_calculation_l465_46581

theorem profit_calculation (x : ℝ) 
  (h1 : 20 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l465_46581


namespace NUMINAMATH_CALUDE_expression_evaluation_l465_46562

theorem expression_evaluation (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l465_46562


namespace NUMINAMATH_CALUDE_investment_proof_l465_46530

-- Define the interest rates
def interest_rate_x : ℚ := 23 / 100
def interest_rate_y : ℚ := 17 / 100

-- Define the investment in fund X
def investment_x : ℚ := 42000

-- Define the interest difference
def interest_difference : ℚ := 200

-- Define the total investment
def total_investment : ℚ := 100000

-- Theorem statement
theorem investment_proof :
  ∃ (investment_y : ℚ),
    investment_y * interest_rate_y = investment_x * interest_rate_x + interest_difference ∧
    investment_x + investment_y = total_investment :=
by
  sorry


end NUMINAMATH_CALUDE_investment_proof_l465_46530


namespace NUMINAMATH_CALUDE_benny_spent_85_dollars_l465_46571

def baseball_gear_total (glove_price baseball_price bat_price helmet_price gloves_price : ℕ) : ℕ :=
  glove_price + baseball_price + bat_price + helmet_price + gloves_price

theorem benny_spent_85_dollars : 
  baseball_gear_total 25 5 30 15 10 = 85 := by
  sorry

end NUMINAMATH_CALUDE_benny_spent_85_dollars_l465_46571


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l465_46527

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^(1/4 : ℝ)) 
  (h₂ : a₂ = 2^(1/8 : ℝ)) 
  (h₃ : a₃ = 2^(1/16 : ℝ)) 
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = 2^(-1/16 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l465_46527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l465_46569

/-- An arithmetic sequence with first term -2015 -/
def arithmetic_sequence (n : ℕ) : ℤ := -2015 + (n - 1) * d
  where d : ℤ := 2  -- We define d here, but it should be derived in the proof

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (2 * (-2015) + (n - 1) * d) / 2
  where d : ℤ := 2  -- We define d here, but it should be derived in the proof

/-- Main theorem -/
theorem arithmetic_sequence_sum :
  2 * S 6 - 3 * S 4 = 24 → S 2015 = -2015 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l465_46569


namespace NUMINAMATH_CALUDE_keith_and_jason_books_l465_46515

/-- The number of books Keith and Jason have together -/
def total_books (keith_books jason_books : ℕ) : ℕ :=
  keith_books + jason_books

/-- Theorem: Keith and Jason have 41 books together -/
theorem keith_and_jason_books :
  total_books 20 21 = 41 := by
  sorry

end NUMINAMATH_CALUDE_keith_and_jason_books_l465_46515


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l465_46587

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l465_46587


namespace NUMINAMATH_CALUDE_mixture_ratio_theorem_l465_46557

/-- Represents the components of the mixture -/
inductive Component
  | Milk
  | Water
  | Juice

/-- Calculates the amount of a component in the initial mixture -/
def initial_amount (c : Component) : ℚ :=
  match c with
  | Component.Milk => 60 * (5 / 8)
  | Component.Water => 60 * (2 / 8)
  | Component.Juice => 60 * (1 / 8)

/-- Calculates the amount of a component after adding water and juice -/
def final_amount (c : Component) : ℚ :=
  match c with
  | Component.Milk => initial_amount Component.Milk
  | Component.Water => initial_amount Component.Water + 15
  | Component.Juice => initial_amount Component.Juice + 5

/-- Represents the final ratio of the mixture components -/
def final_ratio : Fin 3 → ℕ
  | 0 => 15  -- Milk
  | 1 => 12  -- Water
  | 2 => 5   -- Juice
  | _ => 0   -- This case is unreachable, but needed for completeness

theorem mixture_ratio_theorem :
  ∃ (k : ℚ), k > 0 ∧
    (final_amount Component.Milk = k * final_ratio 0) ∧
    (final_amount Component.Water = k * final_ratio 1) ∧
    (final_amount Component.Juice = k * final_ratio 2) :=
sorry

end NUMINAMATH_CALUDE_mixture_ratio_theorem_l465_46557


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l465_46568

theorem max_value_sum_fractions (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_sum : a + b + c = 1) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l465_46568


namespace NUMINAMATH_CALUDE_shadow_length_of_shorter_cycle_l465_46524

/-- Given two similar right-angled triangles formed by cycles and their shadows,
    this theorem proves the length of the shadow for the shorter cycle. -/
theorem shadow_length_of_shorter_cycle
  (H1 : ℝ) (S1 : ℝ) (H2 : ℝ)
  (height1 : H1 = 2.5)
  (shadow1 : S1 = 5)
  (height2 : H2 = 2)
  (similar_triangles : H1 / S1 = H2 / S2)
  : S2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_of_shorter_cycle_l465_46524


namespace NUMINAMATH_CALUDE_evaluate_expression_l465_46502

theorem evaluate_expression : 7 - 5 * (6 - 2^3) * 3 = -23 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l465_46502


namespace NUMINAMATH_CALUDE_point_on_line_l465_46520

/-- Given a line passing through point M(0, 1) with slope -1,
    prove that any point P(3, m) on this line satisfies m = -2 -/
theorem point_on_line (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P.1 = 3 ∧ P.2 = m ∧ 
   (m - 1) / (3 - 0) = -1) → 
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l465_46520


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l465_46580

theorem remainder_of_large_number (N : ℕ) (d : ℕ) (h : N = 9876543210123456789 ∧ d = 252) :
  N % d = 27 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l465_46580


namespace NUMINAMATH_CALUDE_smallest_valid_arrangement_l465_46506

/-- Represents a circular table with chairs -/
structure CircularTable :=
  (num_chairs : ℕ)

/-- Checks if a seating arrangement is valid -/
def is_valid_arrangement (table : CircularTable) (seated : ℕ) : Prop :=
  seated > 0 ∧ seated ≤ table.num_chairs ∧ 
  ∀ (new_seat : ℕ), new_seat ≤ table.num_chairs → 
    ∃ (occupied_seat : ℕ), occupied_seat ≤ table.num_chairs ∧ 
      (new_seat = occupied_seat + 1 ∨ new_seat = occupied_seat - 1 ∨ 
       (occupied_seat = table.num_chairs ∧ new_seat = 1) ∨ 
       (occupied_seat = 1 ∧ new_seat = table.num_chairs))

/-- The theorem to be proved -/
theorem smallest_valid_arrangement (table : CircularTable) 
  (h : table.num_chairs = 100) : 
  (∃ (n : ℕ), is_valid_arrangement table n ∧ 
    ∀ (m : ℕ), m < n → ¬is_valid_arrangement table m) ∧
  (∃ (n : ℕ), is_valid_arrangement table n ∧ n = 20) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_arrangement_l465_46506


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l465_46521

theorem modulus_of_complex_quotient :
  let z : ℂ := (1 - Complex.I) / (3 + 4 * Complex.I)
  Complex.abs z = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l465_46521


namespace NUMINAMATH_CALUDE_min_value_and_existence_l465_46579

/-- The circle C defined by x^2 + y^2 = x + y where x, y > 0 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = p.1 + p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem min_value_and_existence : 
  (∀ p ∈ C, 1 / p.1 + 1 / p.2 ≥ 2) ∧ 
  (∃ p ∈ C, (p.1 + 1) * (p.2 + 1) = 4) := by
sorry

end NUMINAMATH_CALUDE_min_value_and_existence_l465_46579


namespace NUMINAMATH_CALUDE_infinite_lcm_greater_than_ck_l465_46589

theorem infinite_lcm_greater_than_ck 
  (a : ℕ → ℕ) 
  (c : ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ n, a n > 0) 
  (h_c : 0 < c ∧ c < 1.5) : 
  ∀ N, ∃ k > N, Nat.lcm (a k) (a (k + 1)) > ⌊c * k⌋ := by
  sorry

end NUMINAMATH_CALUDE_infinite_lcm_greater_than_ck_l465_46589


namespace NUMINAMATH_CALUDE_library_book_sorting_l465_46584

theorem library_book_sorting (total_removed : ℕ) (damaged : ℕ) (x : ℚ) 
  (h1 : total_removed = 69)
  (h2 : damaged = 11)
  (h3 : total_removed = damaged + (x * damaged - 8)) :
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_library_book_sorting_l465_46584


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l465_46556

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l465_46556


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l465_46578

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l465_46578


namespace NUMINAMATH_CALUDE_overlapping_part_length_l465_46516

/-- Given three wooden planks of equal length and a total fence length,
    calculate the length of one overlapping part. -/
theorem overlapping_part_length
  (plank_length : ℝ)
  (num_planks : ℕ)
  (fence_length : ℝ)
  (h1 : plank_length = 217)
  (h2 : num_planks = 3)
  (h3 : fence_length = 627)
  (h4 : num_planks > 1) :
  let overlap_length := (num_planks * plank_length - fence_length) / (num_planks - 1)
  overlap_length = 12 := by
sorry

end NUMINAMATH_CALUDE_overlapping_part_length_l465_46516


namespace NUMINAMATH_CALUDE_two_digit_perfect_square_divisible_by_five_l465_46531

theorem two_digit_perfect_square_divisible_by_five :
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ ∃ m : ℕ, n = m^2 ∧ n % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_perfect_square_divisible_by_five_l465_46531


namespace NUMINAMATH_CALUDE_goose_eggs_count_l465_46511

/-- The number of goose eggs laid at the pond -/
def total_eggs : ℕ := 1125

/-- The fraction of eggs that hatched -/
def hatched_fraction : ℚ := 1/3

/-- The fraction of hatched geese that survived the first month -/
def survived_month_fraction : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def not_survived_year_fraction : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_year : ℕ := 120

theorem goose_eggs_count :
  (↑survived_year : ℚ) = (↑total_eggs * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction)) ∧
  ∀ n : ℕ, n ≠ total_eggs → 
    (↑survived_year : ℚ) ≠ (↑n * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l465_46511


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l465_46541

theorem imaginary_part_of_complex_fraction :
  Complex.im (2 / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l465_46541


namespace NUMINAMATH_CALUDE_maria_remaining_towels_l465_46538

def green_towels : ℕ := 35
def white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34

theorem maria_remaining_towels :
  green_towels + white_towels - towels_given_to_mother = 22 :=
by sorry

end NUMINAMATH_CALUDE_maria_remaining_towels_l465_46538


namespace NUMINAMATH_CALUDE_emily_total_points_l465_46517

/-- Represents the points scored in each round of the trivia game -/
structure GameRounds where
  round1 : ℤ
  round2 : ℤ
  round3 : ℤ
  round4 : ℤ
  round5 : ℤ

/-- Calculates the total points at the end of the game -/
def totalPoints (game : GameRounds) : ℤ :=
  game.round1 + 2 * game.round2 + (-game.round3) + 2 * game.round4 + game.round5 / 3

/-- Emily's trivia game -/
def emilyGame : GameRounds :=
  { round1 := 16
  , round2 := 16  -- 32 points after doubling
  , round3 := 27  -- -27 points after reversing
  , round4 := 46  -- 92 points before halving
  , round5 := 4   -- 12 points after tripling
  }

/-- Theorem stating that Emily's total points at the end of the game is 117 -/
theorem emily_total_points : totalPoints emilyGame = 117 := by
  sorry


end NUMINAMATH_CALUDE_emily_total_points_l465_46517


namespace NUMINAMATH_CALUDE_rectangle_ratio_in_square_arrangement_l465_46508

/-- Represents the arrangement of rectangles around a square -/
structure SquareArrangement where
  s : ℝ  -- side length of the inner square
  x : ℝ  -- longer side of each rectangle
  y : ℝ  -- shorter side of each rectangle

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_ratio_in_square_arrangement
  (arr : SquareArrangement)
  (h1 : arr.s > 0)  -- inner square side length is positive
  (h2 : arr.s + 2 * arr.y = 3 * arr.s)  -- outer square side length relation
  (h3 : arr.x + arr.s = 3 * arr.s)  -- outer square side length relation in perpendicular direction
  : arr.x / arr.y = 2 := by
  sorry

#check rectangle_ratio_in_square_arrangement

end NUMINAMATH_CALUDE_rectangle_ratio_in_square_arrangement_l465_46508


namespace NUMINAMATH_CALUDE_min_cost_rectangular_container_l465_46566

/-- Represents the cost function for a rectangular container -/
def cost_function (a b : ℝ) : ℝ := 20 * a * b + 10 * 2 * (a + b)

/-- Theorem stating the minimum cost for the rectangular container -/
theorem min_cost_rectangular_container :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = 4 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y = 4 → cost_function a b ≤ cost_function x y) ∧
  cost_function a b = 160 :=
sorry

end NUMINAMATH_CALUDE_min_cost_rectangular_container_l465_46566


namespace NUMINAMATH_CALUDE_zero_in_interval_l465_46540

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x - 9

-- State the theorem
theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l465_46540


namespace NUMINAMATH_CALUDE_rectangle_area_l465_46554

theorem rectangle_area (w l : ℕ) : 
  (2 * (w + l) = 60) →  -- Perimeter is 60 units
  (l = w + 1) →         -- Length and width are consecutive integers
  (w * l = 210)         -- Area is 210 square units
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_l465_46554


namespace NUMINAMATH_CALUDE_old_cards_count_l465_46504

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 6

theorem old_cards_count : 
  (total_pages * cards_per_page) - new_cards = 10 := by
  sorry

end NUMINAMATH_CALUDE_old_cards_count_l465_46504


namespace NUMINAMATH_CALUDE_difference_of_ones_and_zeros_313_l465_46555

/-- The number of zeros in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- The number of ones in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

theorem difference_of_ones_and_zeros_313 : 
  count_ones 313 - count_zeros 313 = 3 := by sorry

end NUMINAMATH_CALUDE_difference_of_ones_and_zeros_313_l465_46555


namespace NUMINAMATH_CALUDE_sphere_surface_area_increase_l465_46537

theorem sphere_surface_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.1 * r
  let original_area := 4 * Real.pi * r^2
  let new_area := 4 * Real.pi * new_radius^2
  (new_area - original_area) / original_area = 0.21 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_increase_l465_46537


namespace NUMINAMATH_CALUDE_cube_diagonal_l465_46551

theorem cube_diagonal (s : ℝ) (h : s > 0) (eq : s^3 + 36*s = 12*s^2) : 
  Real.sqrt (3 * s^2) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_l465_46551


namespace NUMINAMATH_CALUDE_theta_range_l465_46591

theorem theta_range (θ : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l465_46591


namespace NUMINAMATH_CALUDE_complex_power_problem_l465_46550

theorem complex_power_problem : ((1 - Complex.I) / (1 + Complex.I)) ^ 10 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l465_46550


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l465_46574

theorem arithmetic_mean_difference (p q r : ℝ) : 
  (p + q) / 2 = 10 → (q + r) / 2 = 24 → r - p = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l465_46574


namespace NUMINAMATH_CALUDE_unique_solution_cubic_linear_l465_46542

/-- The system of equations y = x^3 and y = 4x + m has exactly one real solution if and only if m = -8 -/
theorem unique_solution_cubic_linear (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^3 = 4*p.1 + m ∧ p.2 = p.1^3) ↔ m = -8 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_linear_l465_46542


namespace NUMINAMATH_CALUDE_parameterization_validity_l465_46523

def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

def valid_parameterization (p₀ : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_equation (p₀.1 + t * v.1) (p₀.2 + t * v.2)

theorem parameterization_validity :
  valid_parameterization (0, 4) (1, -3) ∧
  valid_parameterization (-2, 10) (-2, 6) ∧
  valid_parameterization (-1, 7) (2, -6) ∧
  ¬ valid_parameterization (1, 1) (3, -1) ∧
  ¬ valid_parameterization (4, -8) (0.5, -1.5) :=
sorry

end NUMINAMATH_CALUDE_parameterization_validity_l465_46523


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l465_46536

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The ternary representation of the number -/
def ternary_121 : List Nat := [1, 2, 1]

theorem ternary_121_equals_16 :
  ternary_to_decimal ternary_121 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l465_46536


namespace NUMINAMATH_CALUDE_percent_decrease_long_distance_call_l465_46544

def original_cost : ℝ := 50
def new_cost : ℝ := 10

theorem percent_decrease_long_distance_call :
  (original_cost - new_cost) / original_cost * 100 = 80 := by sorry

end NUMINAMATH_CALUDE_percent_decrease_long_distance_call_l465_46544


namespace NUMINAMATH_CALUDE_isosceles_max_angle_diff_l465_46526

/-- An isosceles triangle has two equal angles -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)

/-- Given an isosceles triangle with one angle 50°, prove that the maximum difference between the other two angles is 30° -/
theorem isosceles_max_angle_diff (t : IsoscelesTriangle) (h : t.a = 50 ∨ t.b = 50 ∨ t.c = 50) :
  ∃ (x y : ℝ), ((x = t.a ∧ y = t.b) ∨ (x = t.b ∧ y = t.c) ∨ (x = t.a ∧ y = t.c)) ∧
  (∀ (x' y' : ℝ), ((x' = t.a ∧ y' = t.b) ∨ (x' = t.b ∧ y' = t.c) ∨ (x' = t.a ∧ y' = t.c)) →
  |x' - y'| ≤ |x - y|) ∧ |x - y| = 30 :=
sorry

end NUMINAMATH_CALUDE_isosceles_max_angle_diff_l465_46526


namespace NUMINAMATH_CALUDE_slower_train_speed_l465_46590

/-- Proves that the speed of the slower train is 36 kmph given the conditions of the problem -/
theorem slower_train_speed 
  (faster_speed : ℝ) 
  (faster_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : faster_speed = 72) 
  (h2 : faster_length = 180) 
  (h3 : crossing_time = 18) : 
  ∃ (slower_speed : ℝ), slower_speed = 36 ∧ 
    faster_length = (faster_speed - slower_speed) * (5/18) * crossing_time :=
sorry

end NUMINAMATH_CALUDE_slower_train_speed_l465_46590


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l465_46583

-- Problem 1
theorem problem_1 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + 2*y)^2 - (-2*x*y^2)^2 / (x*y^3) = x^2 + 4*y^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx1 : x ≠ 1) (hx3 : x ≠ 3) :
  (x - 1) / (x - 3) * (2 - x + 2 / (x - 1)) = -x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l465_46583


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l465_46595

-- Problem 1
theorem problem_1 (x : ℚ) : 
  16 * (6*x - 1) * (2*x - 1) * (3*x + 1) * (x - 1) + 25 = (24*x^2 - 16*x - 3)^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℚ) : 
  (6*x - 1) * (2*x - 1) * (3*x - 1) * (x - 1) + x^2 = (6*x^2 - 6*x + 1)^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℚ) : 
  (6*x - 1) * (4*x - 1) * (3*x - 1) * (x - 1) + 9*x^4 = (9*x^2 - 7*x + 1)^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l465_46595


namespace NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_e_l465_46529

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem instantaneous_rate_of_change_at_e :
  deriv f e = 0 := by sorry

end NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_e_l465_46529


namespace NUMINAMATH_CALUDE_train_speed_with_stoppages_l465_46593

/-- Given a train that travels at 400 km/h without stoppages and stops for 6 minutes per hour,
    its average speed with stoppages is 360 km/h. -/
theorem train_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 400
  let minutes_stopped_per_hour : ℝ := 6
  let minutes_per_hour : ℝ := 60
  let speed_with_stoppages : ℝ := speed_without_stoppages * (minutes_per_hour - minutes_stopped_per_hour) / minutes_per_hour
  speed_with_stoppages = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_with_stoppages_l465_46593


namespace NUMINAMATH_CALUDE_y_intercept_range_l465_46500

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line y = kx + 1
def line1 (k x y : ℝ) : Prop := y = k * x + 1

-- Define the line l passing through (-2, 0) and the midpoint of AB
def line_l (m b x y : ℝ) : Prop := y = m * x + b

-- Define the condition that k is in the valid range
def valid_k (k : ℝ) : Prop := 1 < k ∧ k < Real.sqrt 2

-- Define the range of b
def b_range (b : ℝ) : Prop := b < -2 - Real.sqrt 2 ∨ b > 2

-- Main theorem
theorem y_intercept_range (k m b : ℝ) : 
  valid_k k →
  (∃ x1 y1 x2 y2 : ℝ, 
    hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
    line1 k x1 y1 ∧ line1 k x2 y2 ∧
    x1 < 0 ∧ x2 < 0 ∧
    line_l m b (-2) 0 ∧
    line_l m b ((x1 + x2) / 2) ((y1 + y2) / 2)) →
  b_range b :=
sorry

end NUMINAMATH_CALUDE_y_intercept_range_l465_46500


namespace NUMINAMATH_CALUDE_alice_number_theorem_l465_46599

def smallest_prime_divisor (n : ℕ) : ℕ := sorry

def subtract_smallest_prime_divisor (n : ℕ) : ℕ := n - smallest_prime_divisor n

def iterate_subtraction (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterate_subtraction (subtract_smallest_prime_divisor n) k

theorem alice_number_theorem (n : ℕ) :
  n > 0 ∧ Nat.Prime (iterate_subtraction n 2022) →
  n = 4046 ∨ n = 4047 :=
sorry

end NUMINAMATH_CALUDE_alice_number_theorem_l465_46599


namespace NUMINAMATH_CALUDE_laticia_socks_count_l465_46501

/-- The number of pairs of socks Laticia knitted in the first week -/
def first_week : ℕ := 12

/-- The number of pairs of socks Laticia knitted in the second week -/
def second_week : ℕ := first_week + 4

/-- The number of pairs of socks Laticia knitted in the third week -/
def third_week : ℕ := (first_week + second_week) / 2

/-- The number of pairs of socks Laticia knitted in the fourth week -/
def fourth_week : ℕ := third_week - 3

/-- The total number of pairs of socks Laticia knitted over four weeks -/
def total_socks : ℕ := first_week + second_week + third_week + fourth_week

theorem laticia_socks_count : total_socks = 53 := by
  sorry

end NUMINAMATH_CALUDE_laticia_socks_count_l465_46501


namespace NUMINAMATH_CALUDE_third_coaster_speed_l465_46546

/-- Theorem: Given 5 rollercoasters with specified speeds and average, prove the speed of the third coaster -/
theorem third_coaster_speed 
  (v1 v2 v3 v4 v5 : ℝ) 
  (h1 : v1 = 50)
  (h2 : v2 = 62)
  (h4 : v4 = 70)
  (h5 : v5 = 40)
  (h_avg : (v1 + v2 + v3 + v4 + v5) / 5 = 59) :
  v3 = 73 := by
sorry

end NUMINAMATH_CALUDE_third_coaster_speed_l465_46546


namespace NUMINAMATH_CALUDE_dice_probability_l465_46594

def num_dice : ℕ := 6
def num_sides : ℕ := 15
def num_low_sides : ℕ := 9
def num_high_sides : ℕ := 6

def prob_low : ℚ := num_low_sides / num_sides
def prob_high : ℚ := num_high_sides / num_sides

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dice_probability : 
  (choose num_dice (num_dice / 2)) * (prob_low ^ (num_dice / 2)) * (prob_high ^ (num_dice / 2)) = 4320 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l465_46594


namespace NUMINAMATH_CALUDE_bill_sue_score_ratio_l465_46543

theorem bill_sue_score_ratio :
  ∀ (john_score sue_score : ℕ),
    45 = john_score + 20 →
    45 + john_score + sue_score = 160 →
    (45 : ℚ) / sue_score = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bill_sue_score_ratio_l465_46543


namespace NUMINAMATH_CALUDE_inequality_proof_l465_46576

theorem inequality_proof (a b c d : ℝ) (h1 : a * b > 0) (h2 : -c / a < -d / b) :
  b * c > a * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l465_46576


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_i_l465_46573

theorem imaginary_part_of_reciprocal_i : Complex.im (1 / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_i_l465_46573


namespace NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l465_46586

def f (y : ℝ) : ℝ := y^4 - y^3 + 2*y - 1

def g (y : ℝ) : ℝ := y^2 + y - 3

theorem product_of_g_at_roots_of_f :
  ∀ y₁ y₂ y₃ y₄ : ℝ,
  f y₁ = 0 → f y₂ = 0 → f y₃ = 0 → f y₄ = 0 →
  ∃ result : ℝ, g y₁ * g y₂ * g y₃ * g y₄ = result :=
by sorry

end NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l465_46586


namespace NUMINAMATH_CALUDE_angle_side_inequality_l465_46577

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define the angle and side length functions
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
def side_length (t : Triangle) (v : Fin 3) : ℝ := sorry

-- State the theorem
theorem angle_side_inequality (t : Triangle) :
  angle t 0 > angle t 1 → side_length t 0 > side_length t 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_side_inequality_l465_46577


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l465_46567

/-- Given a quadratic equation x^2 + 2100x + 4200, prove that when rewritten in the form (x+b)^2 + c, the value of c/b is -1034 -/
theorem quadratic_rewrite_ratio : 
  ∃ (b c : ℝ), (∀ x, x^2 + 2100*x + 4200 = (x + b)^2 + c) ∧ c/b = -1034 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l465_46567


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l465_46522

theorem arithmetic_mean_problem (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l465_46522


namespace NUMINAMATH_CALUDE_transportation_cost_optimization_l465_46575

/-- The transportation cost problem -/
theorem transportation_cost_optimization (a : ℝ) :
  let distance : ℝ := 300
  let fuel_cost_constant : ℝ := 1/2
  let other_costs : ℝ := 800
  let cost_function (x : ℝ) : ℝ := 150 * (x + 1600 / x)
  let optimal_speed : ℝ := if a ≥ 40 then 40 else a
  0 < a →
  (∀ x > 0, x ≤ a → cost_function optimal_speed ≤ cost_function x) :=
by sorry

end NUMINAMATH_CALUDE_transportation_cost_optimization_l465_46575


namespace NUMINAMATH_CALUDE_value_of_expression_l465_46563

theorem value_of_expression (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l465_46563


namespace NUMINAMATH_CALUDE_one_point_one_billion_scientific_notation_l465_46512

/-- Expresses 1.1 billion in scientific notation -/
theorem one_point_one_billion_scientific_notation :
  (1.1 * 10^9 : ℝ) = 1100000000 := by
  sorry

end NUMINAMATH_CALUDE_one_point_one_billion_scientific_notation_l465_46512


namespace NUMINAMATH_CALUDE_sqrt_sum_equal_l465_46585

theorem sqrt_sum_equal : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equal_l465_46585


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l465_46535

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → a ∈ Set.Icc (-2) 3 := by sorry


end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l465_46535


namespace NUMINAMATH_CALUDE_intersection_condition_l465_46532

theorem intersection_condition (m n : ℝ) : 
  let A : Set ℝ := {2, m / (2 * n)}
  let B : Set ℝ := {m, n}
  (A ∩ B : Set ℝ) = {1} → n = 1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_condition_l465_46532
