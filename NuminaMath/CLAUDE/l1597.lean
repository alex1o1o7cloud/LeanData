import Mathlib

namespace NUMINAMATH_CALUDE_games_mike_can_buy_l1597_159749

theorem games_mike_can_buy (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 69 → spent_amount = 24 → game_cost = 5 →
  (initial_amount - spent_amount) / game_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_games_mike_can_buy_l1597_159749


namespace NUMINAMATH_CALUDE_machine_purchase_price_l1597_159786

def machine_value (purchase_price : ℝ) (years : ℕ) : ℝ :=
  purchase_price * (1 - 0.3) ^ years

theorem machine_purchase_price : 
  ∃ (purchase_price : ℝ), 
    purchase_price > 0 ∧ 
    machine_value purchase_price 2 = 3200 ∧
    purchase_price = 8000 := by
  sorry

end NUMINAMATH_CALUDE_machine_purchase_price_l1597_159786


namespace NUMINAMATH_CALUDE_cost_of_four_books_l1597_159715

/-- Given that two identical books cost $36, prove that four of these books cost $72. -/
theorem cost_of_four_books (cost_of_two : ℝ) (h : cost_of_two = 36) : 
  2 * cost_of_two = 72 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_four_books_l1597_159715


namespace NUMINAMATH_CALUDE_balcony_seat_cost_l1597_159724

/-- Theorem: Cost of a balcony seat in a theater --/
theorem balcony_seat_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (orchestra_price : ℕ)
  (balcony_orchestra_diff : ℕ)
  (h1 : total_tickets = 355)
  (h2 : total_revenue = 3320)
  (h3 : orchestra_price = 12)
  (h4 : balcony_orchestra_diff = 115) :
  ∃ (balcony_price : ℕ),
    balcony_price = 8 ∧
    balcony_price * (total_tickets / 2 + balcony_orchestra_diff / 2) +
    orchestra_price * (total_tickets / 2 - balcony_orchestra_diff / 2) =
    total_revenue :=
by sorry

end NUMINAMATH_CALUDE_balcony_seat_cost_l1597_159724


namespace NUMINAMATH_CALUDE_odd_function_zero_at_origin_l1597_159733

-- Define the function f on the interval [-1, 1]
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x

-- State the theorem
theorem odd_function_zero_at_origin (h : isOdd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_at_origin_l1597_159733


namespace NUMINAMATH_CALUDE_complement_of_A_l1597_159723

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A (x : ℝ) : x ∈ (Set.compl A) ↔ x ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1597_159723


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l1597_159705

/-- The probability of selecting two red balls from a bag with given ball counts. -/
theorem probability_two_red_balls (red blue green : ℕ) (h : red = 5 ∧ blue = 6 ∧ green = 2) :
  let total := red + blue + green
  let choose_two (n : ℕ) := n * (n - 1) / 2
  (choose_two red : ℚ) / (choose_two total) = 5 / 39 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l1597_159705


namespace NUMINAMATH_CALUDE_quiz_smallest_n_l1597_159778

/-- The smallest possible value of n in the quiz problem -/
theorem quiz_smallest_n : ∃ (n : ℤ), n = 89 ∧ 
  ∀ (m : ℕ+) (n' : ℤ),
  (m : ℤ) * (n' + 2) - m * (m + 1) = 2009 →
  n ≤ n' :=
by sorry

end NUMINAMATH_CALUDE_quiz_smallest_n_l1597_159778


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l1597_159794

theorem new_ratio_after_addition (a b c : ℤ) : 
  (3 * a = b) → 
  (b = 15) → 
  (c = a + 10) → 
  (c = b) := by sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l1597_159794


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1597_159707

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 7 * a 13 = 2) →
  (a 7 + a 13 = 3) →
  a 2 * a 18 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1597_159707


namespace NUMINAMATH_CALUDE_lowest_common_multiple_10_to_30_l1597_159788

theorem lowest_common_multiple_10_to_30 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, 10 ≤ k ∧ k ≤ 30 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 10 ≤ k ∧ k ≤ 30 → k ∣ m) → n ≤ m) ∧
  n = 232792560 :=
by sorry

end NUMINAMATH_CALUDE_lowest_common_multiple_10_to_30_l1597_159788


namespace NUMINAMATH_CALUDE_common_point_polar_coords_l1597_159730

-- Define the circle O in polar coordinates
def circle_O (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

-- Theorem statement
theorem common_point_polar_coords :
  ∃ (ρ θ : ℝ), 
    circle_O ρ θ ∧ 
    line_l ρ θ ∧ 
    0 < θ ∧ 
    θ < Real.pi ∧ 
    ρ = 1 ∧ 
    θ = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_common_point_polar_coords_l1597_159730


namespace NUMINAMATH_CALUDE_carries_payment_is_correct_l1597_159779

/-- Calculates Carrie's payment for clothes given the quantities and prices of items, and her mom's contribution ratio --/
def carries_payment (shirt_qty : ℕ) (shirt_price : ℚ) 
                    (pants_qty : ℕ) (pants_price : ℚ)
                    (jacket_qty : ℕ) (jacket_price : ℚ)
                    (skirt_qty : ℕ) (skirt_price : ℚ)
                    (shoes_qty : ℕ) (shoes_price : ℚ)
                    (mom_ratio : ℚ) : ℚ :=
  let total_cost := shirt_qty * shirt_price + 
                    pants_qty * pants_price + 
                    jacket_qty * jacket_price + 
                    skirt_qty * skirt_price + 
                    shoes_qty * shoes_price
  total_cost - (mom_ratio * total_cost)

/-- Theorem: Carrie's payment for clothes is $228.67 --/
theorem carries_payment_is_correct : 
  carries_payment 8 12 4 25 4 75 3 30 2 50 (2/3) = 228.67 := by
  sorry

end NUMINAMATH_CALUDE_carries_payment_is_correct_l1597_159779


namespace NUMINAMATH_CALUDE_no_mem_is_veen_l1597_159753

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem En Veen : Set U) -- Subsets of U

-- State the theorem
theorem no_mem_is_veen 
  (h1 : Mem ⊆ En) -- All Mems are Ens
  (h2 : En ∩ Veen = ∅) -- No Ens are Veens
  : Mem ∩ Veen = ∅ := -- No Mem is a Veen
by
  sorry

end NUMINAMATH_CALUDE_no_mem_is_veen_l1597_159753


namespace NUMINAMATH_CALUDE_jame_gold_bars_l1597_159716

/-- The number of gold bars Jame has left after tax and divorce -/
def gold_bars_left (initial : ℕ) (tax_rate : ℚ) (divorce_loss : ℚ) : ℕ :=
  let after_tax := initial - (initial * tax_rate).floor
  (after_tax - (after_tax * divorce_loss).floor).toNat

/-- Theorem stating that Jame has 27 gold bars left after tax and divorce -/
theorem jame_gold_bars :
  gold_bars_left 60 (1/10) (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_jame_gold_bars_l1597_159716


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_smallestOneDigitPrimes_are_prime_smallestTwoDigitPrime_is_prime_l1597_159777

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define the two smallest one-digit primes
def smallestOneDigitPrimes : Fin 2 → ℕ
| 0 => 2
| 1 => 3

-- Define the smallest two-digit prime
def smallestTwoDigitPrime : ℕ := 11

-- Theorem statement
theorem product_of_smallest_primes : 
  (smallestOneDigitPrimes 0) * (smallestOneDigitPrimes 1) * smallestTwoDigitPrime = 66 :=
by
  sorry

-- Prove that the defined numbers are indeed prime
theorem smallestOneDigitPrimes_are_prime :
  ∀ i : Fin 2, isPrime (smallestOneDigitPrimes i) :=
by
  sorry

theorem smallestTwoDigitPrime_is_prime :
  isPrime smallestTwoDigitPrime :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_smallestOneDigitPrimes_are_prime_smallestTwoDigitPrime_is_prime_l1597_159777


namespace NUMINAMATH_CALUDE_polygon_exists_l1597_159780

-- Define the number of matches and their length
def num_matches : ℕ := 12
def match_length : ℝ := 2

-- Define the target area
def target_area : ℝ := 16

-- Define a polygon as a list of points
def Polygon := List (ℝ × ℝ)

-- Function to calculate the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Function to calculate the area of a polygon
def area (p : Polygon) : ℝ := sorry

-- Theorem stating the existence of a polygon satisfying the conditions
theorem polygon_exists : 
  ∃ (p : Polygon), 
    perimeter p = num_matches * match_length ∧ 
    area p = target_area :=
sorry

end NUMINAMATH_CALUDE_polygon_exists_l1597_159780


namespace NUMINAMATH_CALUDE_power_six_times_three_six_l1597_159736

theorem power_six_times_three_six : 6^6 * 3^6 = 34012224 := by
  sorry

end NUMINAMATH_CALUDE_power_six_times_three_six_l1597_159736


namespace NUMINAMATH_CALUDE_daughter_age_in_three_years_l1597_159769

/-- Given a mother's current age and the fact that she was twice her daughter's age 5 years ago,
    this function calculates the daughter's age in 3 years. -/
def daughters_future_age (mothers_current_age : ℕ) : ℕ :=
  let mothers_past_age := mothers_current_age - 5
  let daughters_past_age := mothers_past_age / 2
  let daughters_current_age := daughters_past_age + 5
  daughters_current_age + 3

/-- Theorem stating that given the problem conditions, the daughter will be 26 years old in 3 years. -/
theorem daughter_age_in_three_years :
  daughters_future_age 41 = 26 := by
  sorry

end NUMINAMATH_CALUDE_daughter_age_in_three_years_l1597_159769


namespace NUMINAMATH_CALUDE_round_robin_tournament_teams_l1597_159762

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 28 games, there are 8 teams -/
theorem round_robin_tournament_teams : ∃ (n : ℕ), n > 0 ∧ num_games n = 28 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_teams_l1597_159762


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l1597_159793

theorem restaurant_menu_fraction (total_vegan : ℕ) (total_menu : ℕ) (vegan_with_nuts : ℕ) :
  total_vegan = 6 →
  total_vegan = total_menu / 3 →
  vegan_with_nuts = 1 →
  (total_vegan - vegan_with_nuts : ℚ) / total_menu = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l1597_159793


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1597_159798

-- Define the set of x values that make the expression meaningful
def meaningful_x : Set ℝ := {x | x ≥ -5 ∧ x ≠ 0}

-- Theorem statement
theorem meaningful_expression_range : 
  {x : ℝ | (∃ y : ℝ, y = Real.sqrt (x + 5) / x) ∧ x ≠ 0} = meaningful_x := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1597_159798


namespace NUMINAMATH_CALUDE_two_discounts_price_l1597_159739

/-- The final price of a product after two consecutive 10% discounts -/
def final_price (a : ℝ) : ℝ := a * (1 - 0.1)^2

/-- Theorem stating that the final price after two 10% discounts is correct -/
theorem two_discounts_price (a : ℝ) :
  final_price a = a * (1 - 0.1)^2 := by
  sorry

end NUMINAMATH_CALUDE_two_discounts_price_l1597_159739


namespace NUMINAMATH_CALUDE_ball_problem_l1597_159751

/-- Given the conditions of the ball problem, prove that the number of red, yellow, and white balls is (45, 40, 75). -/
theorem ball_problem (red yellow white : ℕ) : 
  (red + yellow + white = 160) →
  (2 * red / 3 + 3 * yellow / 4 + 4 * white / 5 = 120) →
  (4 * red / 5 + 3 * yellow / 4 + 2 * white / 3 = 116) →
  (red = 45 ∧ yellow = 40 ∧ white = 75) := by
sorry

end NUMINAMATH_CALUDE_ball_problem_l1597_159751


namespace NUMINAMATH_CALUDE_distance_to_origin_l1597_159758

/-- The distance from point P(1, 2, 2) to the origin (0, 0, 0) is 3. -/
theorem distance_to_origin : Real.sqrt (1^2 + 2^2 + 2^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1597_159758


namespace NUMINAMATH_CALUDE_speed_ratio_l1597_159792

-- Define the speeds of A and B
def speed_A : ℝ := sorry
def speed_B : ℝ := sorry

-- Define the initial position of B
def initial_B : ℝ := -800

-- Define the equidistant condition after 1 minute
def equidistant_1 : Prop :=
  speed_A = |initial_B + speed_B|

-- Define the equidistant condition after 7 minutes
def equidistant_7 : Prop :=
  7 * speed_A = |initial_B + 7 * speed_B|

-- Theorem stating the ratio of speeds
theorem speed_ratio :
  equidistant_1 → equidistant_7 → speed_A / speed_B = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l1597_159792


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l1597_159771

theorem ellipse_min_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (3 * Real.sqrt 3)^2 / m^2 + 1 / n^2 = 1 → m + n ≥ 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l1597_159771


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1597_159790

/-- Given a polynomial division, prove that the remainder is 1 -/
theorem polynomial_division_remainder : 
  let P (z : ℝ) := 4 * z^3 - 5 * z^2 - 17 * z + 4
  let D (z : ℝ) := 4 * z + 6
  let Q (z : ℝ) := z^2 - 4 * z + 1/2
  ∃ (R : ℝ → ℝ), (∀ z, P z = D z * Q z + R z) ∧ (∀ z, R z = 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1597_159790


namespace NUMINAMATH_CALUDE_combined_average_age_l1597_159764

/-- Given two groups of people with their respective sizes and average ages,
    calculate the average age of all people combined. -/
theorem combined_average_age
  (size_a : ℕ) (avg_a : ℚ) (size_b : ℕ) (avg_b : ℚ)
  (h1 : size_a = 8)
  (h2 : avg_a = 45)
  (h3 : size_b = 6)
  (h4 : avg_b = 20) :
  (size_a : ℚ) * avg_a + (size_b : ℚ) * avg_b = 240 ∧
  (size_a : ℚ) + (size_b : ℚ) = 14 →
  (size_a : ℚ) * avg_a + (size_b : ℚ) * avg_b / ((size_a : ℚ) + (size_b : ℚ)) = 240 / 7 :=
by sorry

#check combined_average_age

end NUMINAMATH_CALUDE_combined_average_age_l1597_159764


namespace NUMINAMATH_CALUDE_number_satisfying_equations_l1597_159750

theorem number_satisfying_equations (x : ℝ) : 
  16 * x = 3408 ∧ 1.6 * x = 340.8 → x = 213 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equations_l1597_159750


namespace NUMINAMATH_CALUDE_reflection_result_l1597_159742

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The final position of point F after two reflections -/
def finalPosition (F : ℝ × ℝ) : ℝ × ℝ :=
  reflectOverXAxis (reflectOverYAxis F)

theorem reflection_result :
  finalPosition (-1, -1) = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_result_l1597_159742


namespace NUMINAMATH_CALUDE_total_food_consumption_l1597_159722

/-- Calculates the total daily food consumption for two armies with different rations -/
theorem total_food_consumption
  (food_per_soldier_side1 : ℕ)
  (food_difference : ℕ)
  (soldiers_side1 : ℕ)
  (soldier_difference : ℕ)
  (h1 : food_per_soldier_side1 = 10)
  (h2 : food_difference = 2)
  (h3 : soldiers_side1 = 4000)
  (h4 : soldier_difference = 500) :
  let food_per_soldier_side2 := food_per_soldier_side1 - food_difference
  let soldiers_side2 := soldiers_side1 - soldier_difference
  soldiers_side1 * food_per_soldier_side1 + soldiers_side2 * food_per_soldier_side2 = 68000 := by
sorry


end NUMINAMATH_CALUDE_total_food_consumption_l1597_159722


namespace NUMINAMATH_CALUDE_cubic_root_sum_fourth_power_l1597_159717

theorem cubic_root_sum_fourth_power (p q r : ℝ) : 
  (p^3 - p^2 + 2*p - 3 = 0) → 
  (q^3 - q^2 + 2*q - 3 = 0) → 
  (r^3 - r^2 + 2*r - 3 = 0) → 
  p^4 + q^4 + r^4 = 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_fourth_power_l1597_159717


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1597_159706

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a^2 + b^2 + c^2 = 1800 →  -- sum of squares of all sides
  c = 30 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1597_159706


namespace NUMINAMATH_CALUDE_solution_sets_equality_l1597_159797

-- Define the parameter a
def a : ℝ := 1

-- Define the solution set of ax - 1 > 0
def solution_set_1 : Set ℝ := {x | x > 1}

-- Define the solution set of (ax-1)(x+2) ≥ 0
def solution_set_2 : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

-- State the theorem
theorem solution_sets_equality (h : solution_set_1 = {x | x > 1}) : 
  solution_set_2 = {x | x ≤ -2 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l1597_159797


namespace NUMINAMATH_CALUDE_icosikaipentagon_diagonals_from_vertex_l1597_159760

/-- The number of diagonals from a single vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- An icosikaipentagon is a polygon with 25 sides -/
def icosikaipentagon_sides : ℕ := 25

theorem icosikaipentagon_diagonals_from_vertex : 
  diagonals_from_vertex icosikaipentagon_sides = 22 := by
  sorry

end NUMINAMATH_CALUDE_icosikaipentagon_diagonals_from_vertex_l1597_159760


namespace NUMINAMATH_CALUDE_triangle_angle_C_l1597_159752

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  (a = Real.sqrt 6) →
  (b = 2) →
  (B = π / 4) → -- 45° in radians
  (Real.tan A * Real.tan C > 1) →
  -- Conclusion
  C = 5 * π / 12 -- 75° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l1597_159752


namespace NUMINAMATH_CALUDE_multiply_polynomials_l1597_159776

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 8*x^2 + 16) * (x^2 - 4) = x^4 + 8*x^2 + 12 := by
sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l1597_159776


namespace NUMINAMATH_CALUDE_min_value_expression_l1597_159755

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 16) :
  x^2 + 8*x*y + 16*y^2 + 4*z^2 ≥ 48 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 16 ∧ x₀^2 + 8*x₀*y₀ + 16*y₀^2 + 4*z₀^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1597_159755


namespace NUMINAMATH_CALUDE_exactly_one_valid_number_l1597_159796

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- 3-digit whole number
  (n / 100 + (n / 10) % 10 + n % 10 = 28) ∧  -- digit-sum is 28
  (n % 10 < 7) ∧  -- units digit is less than 7
  (n % 2 = 0)  -- units digit is an even number

theorem exactly_one_valid_number : 
  ∃! n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_exactly_one_valid_number_l1597_159796


namespace NUMINAMATH_CALUDE_abs_minus_three_minus_three_eq_zero_l1597_159729

theorem abs_minus_three_minus_three_eq_zero : |(-3 : ℤ)| - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_abs_minus_three_minus_three_eq_zero_l1597_159729


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_fraction_l1597_159732

theorem inhabitable_earth_surface_fraction :
  let total_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  (land_fraction * inhabitable_land_fraction : ℚ) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_fraction_l1597_159732


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1597_159718

theorem quadratic_equation_roots (p : ℝ) (x₁ x₂ : ℝ) : 
  p > 0 → 
  x₁^2 + p*x₁ + 1 = 0 → 
  x₂^2 + p*x₂ + 1 = 0 → 
  |x₁^2 - x₂^2| = p → 
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1597_159718


namespace NUMINAMATH_CALUDE_total_pizza_slices_l1597_159748

theorem total_pizza_slices :
  let num_pizzas : ℕ := 21
  let slices_per_pizza : ℕ := 8
  num_pizzas * slices_per_pizza = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l1597_159748


namespace NUMINAMATH_CALUDE_fraction_problem_l1597_159781

theorem fraction_problem (n : ℝ) (h : (1/3) * (1/4) * n = 15) : 
  ∃ f : ℝ, f * n = 54 ∧ f = 3/10 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1597_159781


namespace NUMINAMATH_CALUDE_pigeonhole_trees_leaves_l1597_159768

theorem pigeonhole_trees_leaves (n : ℕ) (L : ℕ → ℕ) 
  (h1 : ∀ i, i < n → L i < n) : 
  ∃ i j, i < n ∧ j < n ∧ i ≠ j ∧ L i = L j :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_trees_leaves_l1597_159768


namespace NUMINAMATH_CALUDE_water_leak_proof_l1597_159700

/-- A linear function representing the total water amount over time -/
def water_function (k b : ℝ) (t : ℝ) : ℝ := k * t + b

theorem water_leak_proof (k b : ℝ) :
  water_function k b 1 = 7 →
  water_function k b 2 = 12 →
  (k = 5 ∧ b = 2) ∧
  water_function k b 20 = 102 ∧
  ((water_function k b 1440 * 30) / 1500 : ℝ) = 144 :=
by sorry


end NUMINAMATH_CALUDE_water_leak_proof_l1597_159700


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1597_159775

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {5,6,7}

theorem shaded_area_theorem : U \ (M ∪ N) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1597_159775


namespace NUMINAMATH_CALUDE_right_triangle_median_property_l1597_159744

theorem right_triangle_median_property (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_median : (c/2)^2 = a*b) : c/2 = (c/2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_property_l1597_159744


namespace NUMINAMATH_CALUDE_intersection_dot_product_l1597_159785

/-- An ellipse with equation x²/25 + y²/16 = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- A hyperbola with equation x²/4 - y²/5 = 1 -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- The common foci of the ellipse and hyperbola -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point P that lies on both the ellipse and the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Vector from P to F₁ -/
def PF₁ : ℝ × ℝ := (F₁.1 - P.1, F₁.2 - P.2)

/-- Vector from P to F₂ -/
def PF₂ : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem intersection_dot_product :
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2 →
  dot_product PF₁ PF₂ = 11 := by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l1597_159785


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1597_159745

theorem greatest_sum_consecutive_integers (n : ℤ) : 
  (n * (n + 1) < 360) → (∀ m : ℤ, m > n → m * (m + 1) ≥ 360) → n + (n + 1) = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1597_159745


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1597_159719

theorem binomial_square_condition (a : ℝ) : 
  (∃ (p q : ℝ), ∀ x, 4*x^2 + 16*x + a = (p*x + q)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1597_159719


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l1597_159737

/-- The amount of oil leaked before repairs, in liters -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked during repairs, in liters -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked, in liters -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem oil_leak_calculation :
  total_oil_leaked = 11687 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l1597_159737


namespace NUMINAMATH_CALUDE_new_person_weight_l1597_159747

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 4 →
  replaced_weight = 70 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 110 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1597_159747


namespace NUMINAMATH_CALUDE_binomial_product_equals_6720_l1597_159746

theorem binomial_product_equals_6720 : Nat.choose 10 3 * Nat.choose 8 3 = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_equals_6720_l1597_159746


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1597_159709

def is_root (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

theorem isosceles_triangle_perimeter : 
  ∀ (leg : ℝ), 
  is_root leg → 
  leg > 0 → 
  leg + leg > 4 → 
  leg + leg + 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1597_159709


namespace NUMINAMATH_CALUDE_computer_cost_l1597_159757

theorem computer_cost (total_budget : ℕ) (tv_cost : ℕ) (fridge_computer_diff : ℕ) 
  (h1 : total_budget = 1600)
  (h2 : tv_cost = 600)
  (h3 : fridge_computer_diff = 500) : 
  ∃ (computer_cost : ℕ), 
    computer_cost + tv_cost + (computer_cost + fridge_computer_diff) = total_budget ∧ 
    computer_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_computer_cost_l1597_159757


namespace NUMINAMATH_CALUDE_pi_approximation_accuracy_l1597_159738

-- Define the approximation of π
def pi_approx : ℚ := 3.14

-- Define the true value of π (we'll use a rational approximation for simplicity)
def pi_true : ℚ := 355 / 113

-- Define the accuracy of the approximation
def accuracy : ℚ := 0.01

-- Theorem statement
theorem pi_approximation_accuracy :
  |pi_approx - pi_true| < accuracy :=
sorry

end NUMINAMATH_CALUDE_pi_approximation_accuracy_l1597_159738


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1597_159774

/-- Calculate simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proof that the simple interest for the given conditions is 4016.25 -/
theorem simple_interest_calculation :
  let principal : ℚ := 44625
  let rate : ℚ := 1
  let time : ℚ := 9
  simpleInterest principal rate time = 4016.25 := by
  sorry

#eval simpleInterest 44625 1 9

end NUMINAMATH_CALUDE_simple_interest_calculation_l1597_159774


namespace NUMINAMATH_CALUDE_f_composition_value_l1597_159754

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) else Real.tan x

theorem f_composition_value : f (f (3 * Real.pi / 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1597_159754


namespace NUMINAMATH_CALUDE_algebraic_expression_theorem_l1597_159743

-- Define the algebraic expression
def algebraic_expression (a b x : ℝ) : ℝ :=
  (a*x - 3) * (2*x + 4) - x^2 - b

-- Define the condition for no x^2 term
def no_x_squared_term (a : ℝ) : Prop :=
  2*a - 1 = 0

-- Define the condition for no constant term
def no_constant_term (b : ℝ) : Prop :=
  -12 - b = 0

-- Define the final expression to be calculated
def final_expression (a b : ℝ) : ℝ :=
  (2*a + b)^2 - (2 - 2*b)*(2 + 2*b) - 3*a*(a - b)

-- Theorem statement
theorem algebraic_expression_theorem (a b : ℝ) :
  no_x_squared_term a ∧ no_constant_term b →
  a = 1/2 ∧ b = -12 ∧ final_expression a b = 678 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_theorem_l1597_159743


namespace NUMINAMATH_CALUDE_total_cds_count_l1597_159728

/-- The number of CDs Dawn has -/
def dawn_cds : ℕ := 10

/-- The number of CDs Kristine has -/
def kristine_cds : ℕ := dawn_cds + 7

/-- The number of CDs Mark has -/
def mark_cds : ℕ := 2 * kristine_cds

/-- The total number of CDs owned by Dawn, Kristine, and Mark -/
def total_cds : ℕ := dawn_cds + kristine_cds + mark_cds

theorem total_cds_count : total_cds = 61 := by
  sorry

end NUMINAMATH_CALUDE_total_cds_count_l1597_159728


namespace NUMINAMATH_CALUDE_f_property_l1597_159763

/-- Represents a number with k digits, all being 1 -/
def rep_ones (k : ℕ) : ℕ :=
  (10^k - 1) / 9

/-- The function f(x) = 9x^2 + 2x -/
def f (x : ℕ) : ℕ :=
  9 * x^2 + 2 * x

/-- Theorem stating the property of f for numbers with all digits being 1 -/
theorem f_property (k : ℕ) :
  f (rep_ones k) = rep_ones (2 * k) :=
sorry

end NUMINAMATH_CALUDE_f_property_l1597_159763


namespace NUMINAMATH_CALUDE_library_shelves_l1597_159782

theorem library_shelves (books : ℕ) (additional_books : ℕ) (shelves : ℕ) : 
  books = 4305 →
  additional_books = 11 →
  (books + additional_books) % shelves = 0 →
  shelves = 11 :=
by sorry

end NUMINAMATH_CALUDE_library_shelves_l1597_159782


namespace NUMINAMATH_CALUDE_distance_between_signs_l1597_159756

theorem distance_between_signs (total_distance : ℕ) (distance_to_first_sign : ℕ) (distance_after_second_sign : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_to_first_sign - distance_after_second_sign = 375 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_signs_l1597_159756


namespace NUMINAMATH_CALUDE_count_3033_arrangements_l1597_159711

/-- The set of digits in the number 3033 -/
def digits : Finset Nat := {0, 3}

/-- A function that counts the number of four-digit numbers that can be formed from the given digits -/
def countFourDigitNumbers (d : Finset Nat) : Nat :=
  (d.filter (· ≠ 0)).card * d.card * d.card * d.card

/-- Theorem stating that the number of different four-digit numbers formed from 3033 is 1 -/
theorem count_3033_arrangements : countFourDigitNumbers digits = 1 := by
  sorry

end NUMINAMATH_CALUDE_count_3033_arrangements_l1597_159711


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l1597_159784

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 150) 
  (h2 : books_per_shelf = 15) : 
  num_shelves * books_per_shelf = 2250 := by
sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l1597_159784


namespace NUMINAMATH_CALUDE_evaluate_expression_l1597_159721

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1597_159721


namespace NUMINAMATH_CALUDE_unique_number_in_intersection_l1597_159734

theorem unique_number_in_intersection : ∃! x : ℝ, 3 < x ∧ x < 8 ∧ 6 < x ∧ x < 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_in_intersection_l1597_159734


namespace NUMINAMATH_CALUDE_candy_distribution_l1597_159767

theorem candy_distribution (total : Nat) (friends : Nat) (to_remove : Nat) : 
  total = 47 → friends = 5 → to_remove = 2 → 
  to_remove = (total % friends) ∧ 
  ∀ k : Nat, k < to_remove → (total - k) % friends ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1597_159767


namespace NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_l1597_159713

theorem smallest_integer_solution (x : ℤ) : 
  (7 - 3 * x > 22) ∧ (x < 5) → x ≥ -6 :=
by
  sorry

theorem smallest_integer_solution_exists : 
  ∃ x : ℤ, (7 - 3 * x > 22) ∧ (x < 5) ∧ (x = -6) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_l1597_159713


namespace NUMINAMATH_CALUDE_restaurant_bill_average_cost_l1597_159740

theorem restaurant_bill_average_cost
  (total_bill : ℝ)
  (gratuity_rate : ℝ)
  (num_people : ℕ)
  (h1 : total_bill = 720)
  (h2 : gratuity_rate = 0.2)
  (h3 : num_people = 6) :
  (total_bill / (1 + gratuity_rate)) / num_people = 100 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_average_cost_l1597_159740


namespace NUMINAMATH_CALUDE_existence_of_prime_and_cube_root_l1597_159761

theorem existence_of_prime_and_cube_root (n : ℕ+) :
  ∃ (p : ℕ) (m : ℤ), 
    Nat.Prime p ∧ 
    p % 6 = 5 ∧ 
    ¬(p ∣ n.val) ∧ 
    n.val % p = (m ^ 3) % p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_cube_root_l1597_159761


namespace NUMINAMATH_CALUDE_circles_intersection_theorem_l1597_159727

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the given conditions
def O₁ : Point := sorry
def O₂ : Point := sorry
def A : Point := sorry
def B : Point := sorry
def P : Point := sorry
def Q : Point := sorry

def circle₁ : Circle := ⟨O₁, sorry⟩
def circle₂ : Circle := ⟨O₂, sorry⟩

-- Define the necessary predicates
def intersect (c₁ c₂ : Circle) (p : Point) : Prop := sorry
def on_circle (c : Circle) (p : Point) : Prop := sorry
def on_segment (p₁ p₂ p : Point) : Prop := sorry

-- State the theorem
theorem circles_intersection_theorem :
  intersect circle₁ circle₂ A ∧
  intersect circle₁ circle₂ B ∧
  on_circle circle₁ Q ∧
  on_circle circle₂ P ∧
  (∃ (c : Circle), on_circle c O₁ ∧ on_circle c A ∧ on_circle c O₂ ∧ on_circle c P ∧ on_circle c Q) →
  on_segment O₁ Q B ∧ on_segment O₂ P B :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_theorem_l1597_159727


namespace NUMINAMATH_CALUDE_arrangement_count_l1597_159710

theorem arrangement_count : ℕ := by
  -- Define the total number of people
  let total_people : ℕ := 7

  -- Define the number of boys and girls
  let num_boys : ℕ := 5
  let num_girls : ℕ := 2

  -- Define that boy A must be in the middle
  let boy_A_position : ℕ := (total_people + 1) / 2

  -- Define that the girls must be adjacent
  let girls_adjacent : Prop := true

  -- The number of ways to arrange them
  let arrangement_ways : ℕ := 192

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1597_159710


namespace NUMINAMATH_CALUDE_ariel_current_age_l1597_159704

/-- Calculates Ariel's current age based on given information -/
theorem ariel_current_age :
  let birth_year : ℕ := 1992
  let fencing_start_year : ℕ := 2006
  let years_fencing : ℕ := 16
  let current_year : ℕ := fencing_start_year + years_fencing
  let current_age : ℕ := current_year - birth_year
  current_age = 30 := by sorry

end NUMINAMATH_CALUDE_ariel_current_age_l1597_159704


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1597_159783

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_element : ℕ
  h_pop_size : population_size > 0
  h_sample_size : sample_size > 0
  h_sample_size_le_pop : sample_size ≤ population_size
  h_first_element : first_element > 0 ∧ first_element ≤ population_size

/-- The interval between elements in a systematic sample -/
def SystematicSample.interval (s : SystematicSample) : ℕ :=
  s.population_size / s.sample_size

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_element + k * s.interval ∧ n ≤ s.population_size

/-- The theorem to be proved -/
theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop_size : s.population_size = 36)
  (h_sample_size : s.sample_size = 4)
  (h_contains_5 : s.contains 5)
  (h_contains_23 : s.contains 23)
  (h_contains_32 : s.contains 32) :
  s.contains 14 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1597_159783


namespace NUMINAMATH_CALUDE_divisibility_puzzle_l1597_159712

theorem divisibility_puzzle :
  ∃ N : ℕ, (N % 2 = 0) ∧ (N % 4 = 0) ∧ (N % 12 = 0) ∧ (N % 24 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_puzzle_l1597_159712


namespace NUMINAMATH_CALUDE_trajectory_equation_of_M_l1597_159725

theorem trajectory_equation_of_M (x y : ℝ) (h : y ≠ 0) :
  let P : ℝ × ℝ := (x, 3/2 * y)
  (P.1^2 + P.2^2 = 1) →
  (x^2 + (9 * y^2) / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_of_M_l1597_159725


namespace NUMINAMATH_CALUDE_abs_diff_inequality_l1597_159731

theorem abs_diff_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < (5/2) := by sorry

end NUMINAMATH_CALUDE_abs_diff_inequality_l1597_159731


namespace NUMINAMATH_CALUDE_max_area_region_T_l1597_159702

/-- A configuration of four circles tangent to a line -/
structure CircleConfiguration where
  radii : Fin 4 → ℝ
  tangent_point : ℝ × ℝ
  line : Set (ℝ × ℝ)

/-- The region T formed by points inside exactly one circle -/
def region_T (config : CircleConfiguration) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating the maximum area of region T -/
theorem max_area_region_T :
  ∃ (config : CircleConfiguration),
    (config.radii 0 = 2) ∧
    (config.radii 1 = 4) ∧
    (config.radii 2 = 6) ∧
    (config.radii 3 = 8) ∧
    (∀ (other_config : CircleConfiguration),
      (other_config.radii 0 = 2) →
      (other_config.radii 1 = 4) →
      (other_config.radii 2 = 6) →
      (other_config.radii 3 = 8) →
      area (region_T config) ≥ area (region_T other_config)) ∧
    area (region_T config) = 84 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_max_area_region_T_l1597_159702


namespace NUMINAMATH_CALUDE_integer_root_iff_a_value_l1597_159799

def polynomial (a x : ℤ) : ℤ := x^3 + 3*x^2 + a*x - 7

theorem integer_root_iff_a_value (a : ℤ) : 
  (∃ x : ℤ, polynomial a x = 0) ↔ (a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3) := by sorry

end NUMINAMATH_CALUDE_integer_root_iff_a_value_l1597_159799


namespace NUMINAMATH_CALUDE_point_on_same_side_l1597_159701

/-- A point (x, y) is on the same side of the line 2x - y + 1 = 0 as (1, 2) if both points satisfy 2x - y + 1 > 0 -/
def same_side (x y : ℝ) : Prop :=
  2*x - y + 1 > 0 ∧ 2*1 - 2 + 1 > 0

/-- The point (1, 0) is on the same side of the line 2x - y + 1 = 0 as the point (1, 2) -/
theorem point_on_same_side : same_side 1 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_same_side_l1597_159701


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1597_159765

/-- Given a point Q(a-1, a+2) that lies on the x-axis, prove that its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ Q : ℝ × ℝ, Q.1 = a - 1 ∧ Q.2 = a + 2 ∧ Q.2 = 0) → 
  (∃ Q : ℝ × ℝ, Q = (-3, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1597_159765


namespace NUMINAMATH_CALUDE_marys_marbles_l1597_159708

/-- Given that Mary has 9.0 yellow marbles initially and gives 3.0 yellow marbles to Joan,
    prove that Mary will have 6.0 yellow marbles left. -/
theorem marys_marbles (initial : ℝ) (given : ℝ) (left : ℝ) 
    (h1 : initial = 9.0) 
    (h2 : given = 3.0) 
    (h3 : left = initial - given) : 
  left = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_marys_marbles_l1597_159708


namespace NUMINAMATH_CALUDE_difference_of_squares_l1597_159773

theorem difference_of_squares (a : ℝ) : a^2 - 1 = (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1597_159773


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1597_159703

/-- AMC 12 scoring system and problem parameters -/
structure AMC12Params where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Rat
  incorrect_penalty : Rat
  unanswered_points : Rat
  target_score : Rat

/-- Calculate the score based on the number of correct answers -/
def calculate_score (params : AMC12Params) (correct : Nat) : Rat :=
  let incorrect := params.attempted_problems - correct
  let unanswered := params.total_problems - params.attempted_problems
  correct * params.correct_points + 
  incorrect * (-params.incorrect_penalty) + 
  unanswered * params.unanswered_points

/-- The main theorem to prove -/
theorem min_correct_answers_for_target_score 
  (params : AMC12Params)
  (h_total : params.total_problems = 25)
  (h_attempted : params.attempted_problems = 15)
  (h_correct_points : params.correct_points = 7.5)
  (h_incorrect_penalty : params.incorrect_penalty = 2)
  (h_unanswered_points : params.unanswered_points = 2)
  (h_target_score : params.target_score = 120) :
  (∀ k < 14, calculate_score params k < params.target_score) ∧ 
  calculate_score params 14 ≥ params.target_score := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1597_159703


namespace NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l1597_159726

theorem two_digit_number_divisible_by_55 (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → 
  (10 * a + b) % 55 = 0 → 
  (∀ (x y : ℕ), x ≤ 9 → y ≤ 9 → (10 * x + y) % 55 = 0 → x * y ≤ b * a) →
  b * a ≤ 15 →
  10 * a + b = 55 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l1597_159726


namespace NUMINAMATH_CALUDE_inequality_solution_l1597_159789

theorem inequality_solution (x : ℝ) :
  3 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9 * x + 1 →
  x > (5 + Real.sqrt 29) / 2 ∧ x < 11 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1597_159789


namespace NUMINAMATH_CALUDE_smallest_ef_minus_de_l1597_159766

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.de + t.fd > t.ef ∧ t.ef + t.fd > t.de

/-- Theorem: The smallest possible value of EF - DE is 1 for a triangle DEF 
    with integer side lengths, perimeter 2010, and DE < EF ≤ FD -/
theorem smallest_ef_minus_de :
  ∀ t : Triangle,
    t.de + t.ef + t.fd = 2010 →
    t.de < t.ef →
    t.ef ≤ t.fd →
    isValidTriangle t →
    (∀ t' : Triangle,
      t'.de + t'.ef + t'.fd = 2010 →
      t'.de < t'.ef →
      t'.ef ≤ t'.fd →
      isValidTriangle t' →
      t'.ef - t'.de ≥ t.ef - t.de) →
    t.ef - t.de = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_ef_minus_de_l1597_159766


namespace NUMINAMATH_CALUDE_gravel_path_cost_l1597_159787

def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter_paise : ℝ := 80

theorem gravel_path_cost :
  let larger_length := plot_length + 2 * path_width
  let larger_width := plot_width + 2 * path_width
  let larger_area := larger_length * larger_width
  let plot_area := plot_length * plot_width
  let path_area := larger_area - plot_area
  let cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
  path_area * cost_per_sq_meter_rupees = 720 :=
by sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l1597_159787


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1597_159791

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = 16)
  (h_a3 : a 3 = 1) :
  a 9 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1597_159791


namespace NUMINAMATH_CALUDE_multiples_of_ten_l1597_159772

theorem multiples_of_ten (n : ℕ) : 
  100 + (n - 1) * 10 = 10000 ↔ n = 991 :=
by sorry

#check multiples_of_ten

end NUMINAMATH_CALUDE_multiples_of_ten_l1597_159772


namespace NUMINAMATH_CALUDE_gcd_problem_l1597_159720

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1187 * k) :
  Nat.gcd (Int.natAbs (2 * b^2 + 31 * b + 67)) (Int.natAbs (b + 15)) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l1597_159720


namespace NUMINAMATH_CALUDE_block_run_difference_l1597_159770

/-- The difference in distance run around a square block between outer and inner paths -/
theorem block_run_difference (block_side : ℝ) (street_width : ℝ) : 
  block_side = 500 → street_width = 30 → 
  (4 * (block_side + street_width / 2) * π / 2) = 1030 * π := by sorry

end NUMINAMATH_CALUDE_block_run_difference_l1597_159770


namespace NUMINAMATH_CALUDE_complex_expression_equals_one_l1597_159795

theorem complex_expression_equals_one : 
  Real.sqrt 6 / Real.sqrt 2 + |1 - Real.sqrt 3| - Real.sqrt 12 + (1/2)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_one_l1597_159795


namespace NUMINAMATH_CALUDE_probability_heart_then_diamond_l1597_159741

/-- The probability of drawing a heart first and a diamond second from a standard deck of cards -/
theorem probability_heart_then_diamond (total_cards : ℕ) (suits : ℕ) (cards_per_suit : ℕ) 
  (h_total : total_cards = 52)
  (h_suits : suits = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_deck : total_cards = suits * cards_per_suit) :
  (cards_per_suit : ℚ) / total_cards * cards_per_suit / (total_cards - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_diamond_l1597_159741


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l1597_159759

theorem sqrt_50_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l1597_159759


namespace NUMINAMATH_CALUDE_sum_of_47_and_negative_27_l1597_159735

theorem sum_of_47_and_negative_27 : 47 + (-27) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_47_and_negative_27_l1597_159735


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l1597_159714

/-- A quadratic function f(x) = x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- The property of f being decreasing on (-∞, 2] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The main theorem: if f is decreasing on (-∞, 2], then a ≤ -4 -/
theorem decreasing_quadratic_condition (a : ℝ) :
  is_decreasing_on_interval a → a ≤ -4 := by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l1597_159714
