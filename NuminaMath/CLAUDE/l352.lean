import Mathlib

namespace NUMINAMATH_CALUDE_production_exceeds_target_l352_35237

/-- The initial production in 2014 -/
def initial_production : ℕ := 40000

/-- The annual increase rate -/
def increase_rate : ℚ := 1/5

/-- The target production to exceed -/
def target_production : ℕ := 120000

/-- The logarithm of 2 -/
def log_2 : ℚ := 3010/10000

/-- The logarithm of 3 -/
def log_3 : ℚ := 4771/10000

/-- The number of years after 2014 when production exceeds the target -/
def years_to_exceed_target : ℕ := 7

theorem production_exceeds_target :
  years_to_exceed_target = 
    (Nat.ceil (log_3 / (increase_rate * log_2))) :=
by sorry

end NUMINAMATH_CALUDE_production_exceeds_target_l352_35237


namespace NUMINAMATH_CALUDE_matchstick_houses_l352_35212

theorem matchstick_houses (initial_matchsticks : ℕ) (matchsticks_per_house : ℕ) : 
  initial_matchsticks = 600 → 
  matchsticks_per_house = 10 → 
  (initial_matchsticks / 2) / matchsticks_per_house = 30 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_houses_l352_35212


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l352_35236

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a < 100 ∧ b < 100 ∧ 
  is_perfect_square (a + b) ∧ is_perfect_square (a * b)

def solution_set : List (ℕ × ℕ) :=
  [(2, 2), (5, 20), (8, 8), (10, 90), (18, 18), (20, 80), (9, 16), 
   (32, 32), (50, 50), (72, 72), (2, 98), (98, 98), (36, 64)]

theorem perfect_square_pairs :
  ∀ a b : ℕ, valid_pair a b ↔ (a, b) ∈ solution_set ∨ (b, a) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l352_35236


namespace NUMINAMATH_CALUDE_set_intersection_complement_l352_35297

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem set_intersection_complement :
  A ∩ (Set.univ \ B) = {x | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l352_35297


namespace NUMINAMATH_CALUDE_mischief_meet_handshakes_l352_35255

/-- Calculates the number of handshakes in a group where everyone shakes hands with everyone else -/
def handshakes_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the Regional Mischief Meet -/
structure MischiefMeet where
  num_gremlins : ℕ
  num_imps : ℕ
  num_cooperative_imps : ℕ

/-- Calculates the total number of handshakes at the Regional Mischief Meet -/
def total_handshakes (meet : MischiefMeet) : ℕ :=
  handshakes_in_group meet.num_gremlins +
  handshakes_in_group meet.num_cooperative_imps +
  meet.num_gremlins * meet.num_imps

theorem mischief_meet_handshakes :
  let meet : MischiefMeet := {
    num_gremlins := 30,
    num_imps := 20,
    num_cooperative_imps := 10
  }
  total_handshakes meet = 1080 := by sorry

end NUMINAMATH_CALUDE_mischief_meet_handshakes_l352_35255


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l352_35278

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the origin -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, -c)

theorem symmetric_line_correct (a b c : ℝ) :
  let (a', b', c') := symmetric_line a b c
  ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (a' * (-x) + b' * (-y) + c' = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l352_35278


namespace NUMINAMATH_CALUDE_infinite_product_equals_nine_l352_35268

def infinite_product : ℕ → ℝ
  | 0 => 3^(1/2)
  | n + 1 => infinite_product n * (3^(n+1))^(1 / 2^(n+1))

theorem infinite_product_equals_nine :
  ∃ (limit : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |infinite_product n - limit| < ε) ∧ limit = 9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_product_equals_nine_l352_35268


namespace NUMINAMATH_CALUDE_club_membership_after_four_years_l352_35263

/-- Represents the number of members in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 3 * club_members n - 16

/-- The club membership problem -/
theorem club_membership_after_four_years :
  club_members 4 = 980 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_after_four_years_l352_35263


namespace NUMINAMATH_CALUDE_a_33_mod_42_l352_35264

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a_33 divided by 42 has a remainder of 20 -/
theorem a_33_mod_42 : a 33 % 42 = 20 := by sorry

end NUMINAMATH_CALUDE_a_33_mod_42_l352_35264


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l352_35284

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 48 →
  triangle_height = 48 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
  triangle_base = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l352_35284


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l352_35259

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → ∃ (a b : ℕ), a^2 - b^2 = 221 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 229 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l352_35259


namespace NUMINAMATH_CALUDE_chess_tournament_games_l352_35223

/-- The number of games played in a round-robin tournament -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 8 players, where each player plays every other player once,
    the total number of games played is 28. -/
theorem chess_tournament_games :
  gamesPlayed 8 = 28 := by
  sorry

#eval gamesPlayed 8  -- This should output 28

end NUMINAMATH_CALUDE_chess_tournament_games_l352_35223


namespace NUMINAMATH_CALUDE_regression_variable_nature_l352_35254

/-- A variable in regression analysis -/
inductive RegressionVariable
  | Independent
  | Dependent

/-- The nature of a variable -/
inductive VariableNature
  | Deterministic
  | Random

/-- Determines the nature of a regression variable -/
def variableNature (v : RegressionVariable) : VariableNature :=
  match v with
  | RegressionVariable.Independent => VariableNature.Deterministic
  | RegressionVariable.Dependent => VariableNature.Random

theorem regression_variable_nature :
  (variableNature RegressionVariable.Independent = VariableNature.Deterministic) ∧
  (variableNature RegressionVariable.Dependent = VariableNature.Random) := by
  sorry

end NUMINAMATH_CALUDE_regression_variable_nature_l352_35254


namespace NUMINAMATH_CALUDE_rationalized_sum_l352_35211

theorem rationalized_sum (A B C D E F : ℤ) : 
  (∃ (k : ℚ), k * (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F = 1 / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5)) →
  F > 0 →
  (∀ (d : ℤ), d ∣ A ∧ d ∣ B ∧ d ∣ C ∧ d ∣ D ∧ d ∣ F → d = 1 ∨ d = -1) →
  A + B + C + D + E + F = 52 := by
sorry

end NUMINAMATH_CALUDE_rationalized_sum_l352_35211


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1000_l352_35299

theorem smallest_n_divisible_by_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬(1000 ∣ (m+1)*(m+2)*(m+3)*(m+4))) ∧ 
  (1000 ∣ (n+1)*(n+2)*(n+3)*(n+4)) ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1000_l352_35299


namespace NUMINAMATH_CALUDE_julie_reading_problem_l352_35272

/-- The number of pages in Julie's book -/
def total_pages : ℕ := 120

/-- The number of pages Julie read yesterday -/
def pages_yesterday : ℕ := 12

/-- The number of pages Julie read today -/
def pages_today : ℕ := 2 * pages_yesterday

/-- The number of pages remaining after Julie read yesterday and today -/
def remaining_pages : ℕ := total_pages - (pages_yesterday + pages_today)

theorem julie_reading_problem :
  (pages_yesterday = 12) ∧
  (total_pages = 120) ∧
  (pages_today = 2 * pages_yesterday) ∧
  (remaining_pages / 2 = 42) :=
by sorry

end NUMINAMATH_CALUDE_julie_reading_problem_l352_35272


namespace NUMINAMATH_CALUDE_max_difference_is_61_l352_35291

def digits : List Nat := [2, 4, 5, 8]

def two_digit_number (d1 d2 : Nat) : Nat := 10 * d1 + d2

def valid_two_digit_number (n : Nat) : Prop :=
  ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = two_digit_number d1 d2

theorem max_difference_is_61 :
  ∃ a b, valid_two_digit_number a ∧ valid_two_digit_number b ∧
    (∀ x y, valid_two_digit_number x → valid_two_digit_number y →
      x - y ≤ a - b) ∧
    a - b = 61 := by sorry

end NUMINAMATH_CALUDE_max_difference_is_61_l352_35291


namespace NUMINAMATH_CALUDE_pizza_order_l352_35260

theorem pizza_order (slices_per_pizza : ℕ) (total_slices : ℕ) (num_people : ℕ)
  (h1 : slices_per_pizza = 4)
  (h2 : total_slices = 68)
  (h3 : num_people = 25) :
  total_slices / slices_per_pizza = 17 := by
sorry

end NUMINAMATH_CALUDE_pizza_order_l352_35260


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l352_35270

def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) ∧ y > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ b ∈ B, d ∣ b) ∧ (∀ m : ℕ, m > 0 → (∀ b ∈ B, m ∣ b) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l352_35270


namespace NUMINAMATH_CALUDE_smallest_n_cookies_l352_35229

theorem smallest_n_cookies : ∃ (n : ℕ), n > 0 ∧ 16 ∣ (25 * n - 3) ∧ ∀ (m : ℕ), m > 0 ∧ 16 ∣ (25 * m - 3) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_cookies_l352_35229


namespace NUMINAMATH_CALUDE_saeyoung_money_conversion_l352_35281

/-- The exchange rate from yuan to yen -/
def exchange_rate : ℝ := 17.25

/-- The value of Saeyoung's 1000 yuan bill -/
def bill_value : ℝ := 1000

/-- The value of Saeyoung's 10 yuan coin -/
def coin_value : ℝ := 10

/-- The total value of Saeyoung's Chinese money in yen -/
def total_yen : ℝ := (bill_value + coin_value) * exchange_rate

theorem saeyoung_money_conversion :
  total_yen = 17422.5 := by sorry

end NUMINAMATH_CALUDE_saeyoung_money_conversion_l352_35281


namespace NUMINAMATH_CALUDE_bus_passengers_l352_35265

/-- 
Given a bus that starts with 64 students and loses one-third of its 
passengers at each stop, prove that after four stops, 1024/81 students remain.
-/
theorem bus_passengers (initial_students : ℕ) (stops : ℕ) : 
  initial_students = 64 → 
  stops = 4 → 
  (initial_students : ℚ) * (2/3)^stops = 1024/81 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_l352_35265


namespace NUMINAMATH_CALUDE_triangle_properties_l352_35249

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = t.b * Real.sin t.A)
  (h2 : t.b = 3 * Real.sqrt 2)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a + t.b + t.c = 6 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l352_35249


namespace NUMINAMATH_CALUDE_smallest_s_plus_d_l352_35219

theorem smallest_s_plus_d : ∀ s d : ℕ+,
  (1 : ℚ) / s + (1 : ℚ) / (2 * s) + (1 : ℚ) / (3 * s) = (1 : ℚ) / (d^2 - 2*d) →
  ∀ s' d' : ℕ+,
  (1 : ℚ) / s' + (1 : ℚ) / (2 * s') + (1 : ℚ) / (3 * s') = (1 : ℚ) / (d'^2 - 2*d') →
  (s + d : ℕ) ≤ (s' + d' : ℕ) →
  (s + d : ℕ) = 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_s_plus_d_l352_35219


namespace NUMINAMATH_CALUDE_avery_chicken_count_l352_35279

theorem avery_chicken_count :
  ∀ (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) (filled_cartons : ℕ),
    eggs_per_chicken = 6 →
    eggs_per_carton = 12 →
    filled_cartons = 10 →
    filled_cartons * eggs_per_carton / eggs_per_chicken = 20 :=
by sorry

end NUMINAMATH_CALUDE_avery_chicken_count_l352_35279


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_on_hyperbola_l352_35286

/-- Given a hyperbola and a point M, prove that a line passing through M and intersecting the hyperbola at two points with M as their midpoint has a specific equation. -/
theorem line_equation_through_midpoint_on_hyperbola (x y : ℝ → ℝ) (A B M : ℝ × ℝ) :
  (∀ t : ℝ, (x t)^2 - (y t)^2 / 2 = 1) →  -- Hyperbola equation
  M = (2, 1) →  -- Coordinates of point M
  (∃ t₁ t₂ : ℝ, A = (x t₁, y t₁) ∧ B = (x t₂, y t₂)) →  -- A and B are on the hyperbola
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  ∃ k b : ℝ, k = 4 ∧ b = -7 ∧ ∀ x y : ℝ, y = k * x + b ↔ 4 * x - y - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_on_hyperbola_l352_35286


namespace NUMINAMATH_CALUDE_exists_a_with_full_domain_and_range_l352_35269

/-- Given a real number a, f is a function from ℝ to ℝ defined as f(x) = ax^2 + x + 1 -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + x + 1

/-- Theorem stating that there exists a real number a such that f(a) has domain and range ℝ -/
theorem exists_a_with_full_domain_and_range :
  ∃ a : ℝ, Function.Surjective (f a) ∧ Function.Injective (f a) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_with_full_domain_and_range_l352_35269


namespace NUMINAMATH_CALUDE_tiles_needed_is_100_l352_35256

/-- Calculates the number of tiles needed to cover a rectangular room with a central pillar -/
def calculate_tiles (room_length room_width pillar_side border_tile_side central_tile_side : ℕ) : ℕ :=
  let border_tiles := 2 * room_width
  let central_area := room_length * (room_width - 2) - pillar_side^2
  let central_tiles := (central_area + central_tile_side^2 - 1) / central_tile_side^2
  border_tiles + central_tiles

/-- The total number of tiles needed for the specific room configuration is 100 -/
theorem tiles_needed_is_100 : calculate_tiles 30 20 2 1 3 = 100 := by sorry

end NUMINAMATH_CALUDE_tiles_needed_is_100_l352_35256


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l352_35227

theorem polynomial_division_degree (f d q r : Polynomial ℝ) :
  (Polynomial.degree f = 15) →
  (f = d * q + r) →
  (Polynomial.degree q = 8) →
  (r = 5 * X^4 + 3 * X^2 - 2 * X + 7) →
  (Polynomial.degree d = 7) := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l352_35227


namespace NUMINAMATH_CALUDE_sally_cards_total_l352_35296

/-- The number of cards Sally has now is equal to the sum of her initial cards,
    the cards Dan gave her, and the cards she bought. -/
theorem sally_cards_total
  (initial : ℕ)  -- Sally's initial number of cards
  (from_dan : ℕ) -- Number of cards Dan gave Sally
  (bought : ℕ)   -- Number of cards Sally bought
  (h1 : initial = 27)
  (h2 : from_dan = 41)
  (h3 : bought = 20) :
  initial + from_dan + bought = 88 :=
by sorry

end NUMINAMATH_CALUDE_sally_cards_total_l352_35296


namespace NUMINAMATH_CALUDE_expand_product_l352_35246

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l352_35246


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l352_35257

theorem polynomial_divisibility (a b : ℝ) : 
  (∀ (X : ℝ), (X - 1)^2 ∣ (a * X^4 + b * X^3 + 1)) ↔ 
  (a = 3 ∧ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l352_35257


namespace NUMINAMATH_CALUDE_quadratic_minimum_l352_35232

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, x^2 - 4*x + 3 ≥ -1) ∧ (∃ x, x^2 - 4*x + 3 = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l352_35232


namespace NUMINAMATH_CALUDE_prime_power_divisors_l352_35261

theorem prime_power_divisors (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) :
  (∀ d : ℕ, d ∣ p^4 * q^x ↔ d ∈ Finset.range 51) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisors_l352_35261


namespace NUMINAMATH_CALUDE_complex_equation_solution_l352_35250

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 1) : 
  z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l352_35250


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l352_35214

theorem divisibility_implies_equality (a b : ℕ) 
  (h : (a^2 + a*b + 1) % (b^2 + b*a + 1) = 0) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l352_35214


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l352_35266

theorem divisible_by_thirteen (n : ℕ) : ∃ k : ℤ, (7^(2*n) + 10^(n+1) + 2 * 10^n) = 13 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l352_35266


namespace NUMINAMATH_CALUDE_circumscribing_circle_diameter_l352_35290

/-- The diameter of a circle circumscribing 8 tangent circles -/
theorem circumscribing_circle_diameter (r : ℝ) (h : r = 5) : 
  let n : ℕ := 8
  let small_circle_radius := r
  let large_circle_diameter := 2 * r * (3 + Real.sqrt 3)
  large_circle_diameter = 10 * (3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circumscribing_circle_diameter_l352_35290


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l352_35245

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l352_35245


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l352_35200

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun i => 2 * i + 1) = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l352_35200


namespace NUMINAMATH_CALUDE_vet_recommendation_difference_l352_35201

/-- Given a total number of vets and percentages recommending two different brands of dog food,
    prove that the difference in the number of vets recommending each brand is as expected. -/
theorem vet_recommendation_difference
  (total_vets : ℕ)
  (puppy_kibble_percent : ℚ)
  (yummy_dog_kibble_percent : ℚ)
  (h_total : total_vets = 1000)
  (h_puppy : puppy_kibble_percent = 1/5)
  (h_yummy : yummy_dog_kibble_percent = 3/10) :
  (total_vets : ℚ) * yummy_dog_kibble_percent - (total_vets : ℚ) * puppy_kibble_percent = 100 :=
by sorry


end NUMINAMATH_CALUDE_vet_recommendation_difference_l352_35201


namespace NUMINAMATH_CALUDE_homes_cleaned_l352_35210

-- Define the given conditions
def earnings_per_home : ℕ := 46
def total_earnings : ℕ := 276

-- Define the theorem to prove
theorem homes_cleaned (earnings_per_home : ℕ) (total_earnings : ℕ) : 
  total_earnings / earnings_per_home = 6 :=
sorry

end NUMINAMATH_CALUDE_homes_cleaned_l352_35210


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l352_35298

/-- Arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_seq (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Geometric sequence with first term 1 and common ratio 2 -/
def geometric_seq (n : ℕ) : ℕ := 2^(n - 1)

/-- The sum of specific terms in the arithmetic sequence -/
theorem sum_of_specific_terms :
  arithmetic_seq (geometric_seq 2) + 
  arithmetic_seq (geometric_seq 3) + 
  arithmetic_seq (geometric_seq 4) = 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l352_35298


namespace NUMINAMATH_CALUDE_ratio_difference_theorem_l352_35285

theorem ratio_difference_theorem (x : ℝ) (h1 : x > 0) :
  (2 * x) / (3 * x) = 2 / 3 ∧
  (2 * x + 4) / (3 * x + 4) = 5 / 7 →
  3 * x - 2 * x = 8 :=
by sorry

end NUMINAMATH_CALUDE_ratio_difference_theorem_l352_35285


namespace NUMINAMATH_CALUDE_trajectory_is_straight_line_l352_35226

/-- The trajectory of a point P(x, y) equidistant from M(-2, 0) and the line x = -2 is a straight line y = 0 -/
theorem trajectory_is_straight_line :
  ∀ (x y : ℝ), 
    (|x + 2| = Real.sqrt ((x + 2)^2 + y^2)) → 
    y = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_straight_line_l352_35226


namespace NUMINAMATH_CALUDE_cubic_root_sum_l352_35209

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 12*a^2 + 27*a - 18 = 0 →
  b^3 - 12*b^2 + 27*b - 18 = 0 →
  c^3 - 12*c^2 + 27*c - 18 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^3 + 1/b^3 + 1/c^3 = 13/24 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l352_35209


namespace NUMINAMATH_CALUDE_monotonicity_range_and_minimum_value_l352_35217

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := exp x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * exp (a * x - 1) - 2 * a * x + f a x

theorem monotonicity_range_and_minimum_value 
  (h1 : ∀ x, x > 0)
  (h2 : ∀ a, a < 0) :
  (∃ S : Set ℝ, S = { a | ∀ x ∈ (Set.Ioo 0 (log 3)), 
    (Monotone (f a) ↔ Monotone (F a)) ∧ S = Set.Iic (-3)}) ∧
  (∀ a ∈ Set.Iic (-1 / (exp 2)), 
    IsMinOn (g a) (Set.Ioi 0) 0) := by sorry

end NUMINAMATH_CALUDE_monotonicity_range_and_minimum_value_l352_35217


namespace NUMINAMATH_CALUDE_modulus_of_one_minus_i_l352_35244

theorem modulus_of_one_minus_i :
  let z : ℂ := 1 - Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_minus_i_l352_35244


namespace NUMINAMATH_CALUDE_overtime_calculation_l352_35274

/-- A worker's pay calculation --/
def worker_pay (regular_hours : ℕ) (overtime_hours : ℕ) : ℕ :=
  let regular_rate : ℕ := 3
  let overtime_rate : ℕ := 2 * regular_rate
  let max_regular_hours : ℕ := 40
  min regular_hours max_regular_hours * regular_rate +
  overtime_hours * overtime_rate

theorem overtime_calculation :
  ∃ (overtime_hours : ℕ), worker_pay 40 overtime_hours = 198 ∧ overtime_hours = 13 := by
  sorry

end NUMINAMATH_CALUDE_overtime_calculation_l352_35274


namespace NUMINAMATH_CALUDE_tile_ratio_after_modification_l352_35240

/-- Represents a square tile pattern -/
structure TilePattern where
  side : Nat
  black_tiles : Nat
  white_tiles : Nat

/-- Extends the tile pattern with a double border and replaces middle row and column -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 4,
    black_tiles := p.black_tiles + (p.side + 1) + (p.side + 2)^2 - p.side^2,
    white_tiles := p.white_tiles + (p.side + 4)^2 - (p.side + 2)^2 }

/-- The main theorem to prove -/
theorem tile_ratio_after_modification (p : TilePattern) 
  (h1 : p.side = 7)
  (h2 : p.black_tiles = 18) 
  (h3 : p.white_tiles = 39) : 
  let extended := extend_pattern p
  (extended.black_tiles : Rat) / extended.white_tiles = 63 / 79 := by
  sorry

end NUMINAMATH_CALUDE_tile_ratio_after_modification_l352_35240


namespace NUMINAMATH_CALUDE_fishing_problem_l352_35277

/-- The number of fish caught by Ollie -/
def ollie_fish : ℕ := 5

/-- The number of fish caught by Angus relative to Patrick -/
def angus_more_than_patrick : ℕ := 4

/-- The number of fish Ollie caught fewer than Angus -/
def ollie_fewer_than_angus : ℕ := 7

/-- The number of fish caught by Patrick -/
def patrick_fish : ℕ := 8

theorem fishing_problem :
  ollie_fish + ollie_fewer_than_angus - angus_more_than_patrick = patrick_fish := by
  sorry

end NUMINAMATH_CALUDE_fishing_problem_l352_35277


namespace NUMINAMATH_CALUDE_chess_tournament_properties_l352_35273

/-- A chess tournament between Earthlings and aliens -/
structure ChessTournament where
  t : ℕ  -- number of Earthlings
  a : ℕ  -- number of aliens

/-- Total number of matches in the tournament -/
def totalMatches (ct : ChessTournament) : ℕ :=
  (ct.t + ct.a) * (ct.t + ct.a - 1) / 2

/-- Total points of Earthlings -/
def earthlingPoints (ct : ChessTournament) : ℕ :=
  ct.t * (ct.t - 1) / 2 + ct.a * (ct.a - 1) / 2

/-- Theorem stating the properties of the chess tournament -/
theorem chess_tournament_properties (ct : ChessTournament) :
  (totalMatches ct = (ct.t + ct.a) * (ct.t + ct.a - 1) / 2) ∧
  (earthlingPoints ct = ct.t * (ct.t - 1) / 2 + ct.a * (ct.a - 1) / 2) ∧
  (∃ n : ℕ, ct.t + ct.a = n * n) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_properties_l352_35273


namespace NUMINAMATH_CALUDE_container_evaporation_l352_35204

theorem container_evaporation (initial_content : ℝ) : 
  initial_content = 1 →
  let remaining_after_day1 := initial_content - (2/3 * initial_content)
  let remaining_after_day2 := remaining_after_day1 - (1/4 * remaining_after_day1)
  remaining_after_day2 = 1/4 * initial_content := by sorry

end NUMINAMATH_CALUDE_container_evaporation_l352_35204


namespace NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l352_35243

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ := sorry

/-- Theorem stating the area of the region -/
theorem area_between_circles_and_x_axis :
  let c1 : Circle := { center := (3, 5), radius := 5 }
  let c2 : Circle := { center := (15, 5), radius := 3 }
  areaRegion c1 c2 = 60 - 17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l352_35243


namespace NUMINAMATH_CALUDE_complex_product_equals_2401_l352_35252

theorem complex_product_equals_2401 :
  let x : ℂ := Complex.exp (2 * Real.pi * I / 9)
  (3 * x + x^2) * (3 * x^2 + x^4) * (3 * x^3 + x^6) * (3 * x^4 + x^8) *
  (3 * x^5 + x^10) * (3 * x^6 + x^12) * (3 * x^7 + x^14) = 2401 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_2401_l352_35252


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l352_35238

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a2 : a 2 = 1/2) 
  (h_a5 : a 5 = 4) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l352_35238


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l352_35251

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l352_35251


namespace NUMINAMATH_CALUDE_ratio_simplification_l352_35208

theorem ratio_simplification (a b c d : ℚ) (m n : ℕ) :
  (a : ℚ) / (b : ℚ) = (c : ℚ) / (d : ℚ) →
  (m : ℚ) / (n : ℚ) = ((250 : ℚ) * 1000) / ((2 : ℚ) / 5 * 1000000) →
  (1.25 : ℚ) / (5 / 8 : ℚ) = (2 : ℚ) / (1 : ℚ) ∧
  (m : ℚ) / (n : ℚ) = (5 : ℚ) / (8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ratio_simplification_l352_35208


namespace NUMINAMATH_CALUDE_deane_gas_cost_l352_35242

/-- Calculates the total cost of gas for Mr. Deane --/
def total_gas_cost (rollback : ℝ) (current_price : ℝ) (liters_today : ℝ) (liters_friday : ℝ) : ℝ :=
  let price_friday := current_price - rollback
  let cost_today := current_price * liters_today
  let cost_friday := price_friday * liters_friday
  cost_today + cost_friday

/-- Proves that Mr. Deane's total gas cost is $39 --/
theorem deane_gas_cost :
  let rollback : ℝ := 0.4
  let current_price : ℝ := 1.4
  let liters_today : ℝ := 10
  let liters_friday : ℝ := 25
  total_gas_cost rollback current_price liters_today liters_friday = 39 := by
  sorry

#eval total_gas_cost 0.4 1.4 10 25

end NUMINAMATH_CALUDE_deane_gas_cost_l352_35242


namespace NUMINAMATH_CALUDE_currency_notes_existence_l352_35288

theorem currency_notes_existence : 
  ∃ (x y z : ℕ), x + 5*y + 10*z = 480 ∧ x + y + z = 90 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_existence_l352_35288


namespace NUMINAMATH_CALUDE_undefined_fraction_l352_35294

theorem undefined_fraction (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b - 2) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 := by
sorry

end NUMINAMATH_CALUDE_undefined_fraction_l352_35294


namespace NUMINAMATH_CALUDE_x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x_l352_35262

theorem x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x :
  (∃ x : ℝ, x < -1 → abs x > x) ∧ 
  (∃ x : ℝ, abs x > x ∧ x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x_l352_35262


namespace NUMINAMATH_CALUDE_exam_girls_count_l352_35228

/-- Proves that the number of girls is 1800 given the exam conditions -/
theorem exam_girls_count :
  ∀ (boys girls : ℕ),
  boys + girls = 2000 →
  (34 * boys + 32 * girls : ℚ) = 331 * 20 →
  girls = 1800 := by
sorry

end NUMINAMATH_CALUDE_exam_girls_count_l352_35228


namespace NUMINAMATH_CALUDE_fake_to_total_purse_ratio_l352_35235

/-- Given a collection of purses and handbags, prove that the ratio of fake purses to total purses is 1:2 -/
theorem fake_to_total_purse_ratio (total_purses total_handbags : ℕ) 
  (authentic_items : ℕ) (h1 : total_purses = 26) (h2 : total_handbags = 24) 
  (h3 : authentic_items = 31) : 
  (total_purses - authentic_items + total_handbags - total_handbags / 4) / total_purses = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fake_to_total_purse_ratio_l352_35235


namespace NUMINAMATH_CALUDE_permanent_non_technicians_percentage_l352_35292

structure Factory where
  total_workers : ℕ
  technicians : ℕ
  non_technicians : ℕ
  permanent_technicians : ℕ
  temporary_workers : ℕ

def Factory.valid (f : Factory) : Prop :=
  f.technicians + f.non_technicians = f.total_workers ∧
  f.technicians = f.non_technicians ∧
  f.permanent_technicians = f.technicians / 2 ∧
  f.temporary_workers = f.total_workers / 2

theorem permanent_non_technicians_percentage (f : Factory) 
  (h : f.valid) : 
  (f.non_technicians - (f.temporary_workers - f.permanent_technicians)) / f.non_technicians = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_permanent_non_technicians_percentage_l352_35292


namespace NUMINAMATH_CALUDE_missing_number_equation_l352_35267

theorem missing_number_equation (x : ℤ) : 10010 - x * 3 * 2 = 9938 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l352_35267


namespace NUMINAMATH_CALUDE_tire_circumference_l352_35225

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (revolutions_per_minute : ℝ) (car_speed_kmh : ℝ) :
  revolutions_per_minute = 400 →
  car_speed_kmh = 48 →
  (car_speed_kmh * 1000 / 60) / revolutions_per_minute = 2 :=
by sorry

end NUMINAMATH_CALUDE_tire_circumference_l352_35225


namespace NUMINAMATH_CALUDE_calculate_expression_l352_35220

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l352_35220


namespace NUMINAMATH_CALUDE_remainder_problem_l352_35206

theorem remainder_problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 1816 = k * x + 6) : 
  ∃ l : ℕ, 1442 = l * x + 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l352_35206


namespace NUMINAMATH_CALUDE_fifteen_foot_string_wicks_l352_35253

/-- Calculates the total number of wicks that can be cut from a string of given length,
    where the wicks are of two different lengths and there are an equal number of each. -/
def total_wicks (total_length_feet : ℕ) (wick_length_1 : ℕ) (wick_length_2 : ℕ) : ℕ :=
  let total_length_inches := total_length_feet * 12
  let pair_length := wick_length_1 + wick_length_2
  let num_pairs := total_length_inches / pair_length
  2 * num_pairs

/-- Theorem stating that a 15-foot string cut into equal numbers of 6-inch and 12-inch wicks
    results in a total of 20 wicks. -/
theorem fifteen_foot_string_wicks :
  total_wicks 15 6 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_foot_string_wicks_l352_35253


namespace NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l352_35231

theorem greatest_difference_of_units_digit (x : ℕ) : 
  (x < 10) →
  (637 * 10 + x) % 3 = 0 →
  ∃ y z, y < 10 ∧ z < 10 ∧ 
         (637 * 10 + y) % 3 = 0 ∧ 
         (637 * 10 + z) % 3 = 0 ∧ 
         y - z ≤ 6 ∧
         ∀ w, w < 10 → (637 * 10 + w) % 3 = 0 → y - w ≤ 6 ∧ w - z ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l352_35231


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l352_35248

theorem rational_inequality_solution (x : ℝ) : 
  (x + 2) / (x^2 + 3*x + 10) ≥ 0 ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l352_35248


namespace NUMINAMATH_CALUDE_pencil_count_l352_35218

/-- The total number of pencils after multiplication and addition -/
def total_pencils (initial : ℕ) (factor : ℕ) (additional : ℕ) : ℕ :=
  initial * factor + additional

/-- Theorem stating that the total number of pencils is 153 -/
theorem pencil_count : total_pencils 27 4 45 = 153 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l352_35218


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l352_35222

/-- Represents a number in base 6 as XX₆ -/
def base6 (x : ℕ) : ℕ := 6 * x + x

/-- Represents a number in base 8 as YY₈ -/
def base8 (y : ℕ) : ℕ := 8 * y + y

/-- Checks if a digit is valid in base 6 -/
def validBase6Digit (x : ℕ) : Prop := x ≤ 5

/-- Checks if a digit is valid in base 8 -/
def validBase8Digit (y : ℕ) : Prop := y ≤ 7

theorem smallest_dual_base_representation :
  ∃ (x y : ℕ), validBase6Digit x ∧ validBase8Digit y ∧
    base6 x = base8 y ∧
    base6 x = 63 ∧
    (∀ (x' y' : ℕ), validBase6Digit x' → validBase8Digit y' →
      base6 x' = base8 y' → base6 x' ≥ 63) :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l352_35222


namespace NUMINAMATH_CALUDE_unique_consecutive_odd_primes_l352_35276

theorem unique_consecutive_odd_primes :
  ∀ p q r : ℕ,
  Prime p ∧ Prime q ∧ Prime r →
  p < q ∧ q < r →
  Odd p ∧ Odd q ∧ Odd r →
  q = p + 2 ∧ r = q + 2 →
  p = 3 ∧ q = 5 ∧ r = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_consecutive_odd_primes_l352_35276


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l352_35234

/-- The probability that the enemy plane is hit given A's and B's hit probabilities -/
theorem enemy_plane_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.6) (h_B : p_B = 0.4) :
  1 - (1 - p_A) * (1 - p_B) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l352_35234


namespace NUMINAMATH_CALUDE_stratified_sampling_eleventh_grade_l352_35295

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def student_ratio : Fin 3 → ℕ
  | 0 => 4  -- 10th grade
  | 1 => 3  -- 11th grade
  | 2 => 3  -- 12th grade

/-- Total number of parts in the ratio -/
def total_ratio : ℕ := (student_ratio 0) + (student_ratio 1) + (student_ratio 2)

/-- Total sample size -/
def sample_size : ℕ := 50

/-- Calculates the number of students drawn from the 11th grade -/
def eleventh_grade_sample : ℕ := 
  (student_ratio 1 * sample_size) / total_ratio

theorem stratified_sampling_eleventh_grade :
  eleventh_grade_sample = 15 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_eleventh_grade_l352_35295


namespace NUMINAMATH_CALUDE_markers_multiple_of_four_l352_35213

-- Define the types of items
structure Items where
  coloring_books : ℕ
  markers : ℕ
  crayons : ℕ

-- Define the function to calculate the maximum number of baskets
def max_baskets (items : Items) : ℕ :=
  min (min (items.coloring_books) (items.markers)) (items.crayons)

-- Theorem statement
theorem markers_multiple_of_four (items : Items) 
  (h1 : items.coloring_books = 12)
  (h2 : items.crayons = 36)
  (h3 : max_baskets items = 4) :
  ∃ k : ℕ, items.markers = 4 * k :=
sorry

end NUMINAMATH_CALUDE_markers_multiple_of_four_l352_35213


namespace NUMINAMATH_CALUDE_cricketer_wickets_after_match_l352_35230

/-- Represents a cricketer's bowling statistics -/
structure CricketerStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : CricketerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: A cricketer with given stats takes 5 wickets for 26 runs, decreasing average by 0.4 -/
theorem cricketer_wickets_after_match 
  (stats : CricketerStats)
  (h1 : stats.average = 12.4)
  (h2 : newAverage stats 5 26 = stats.average - 0.4) :
  stats.wickets + 5 = 90 := by
sorry

end NUMINAMATH_CALUDE_cricketer_wickets_after_match_l352_35230


namespace NUMINAMATH_CALUDE_log_lower_bound_l352_35271

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ :=
  (Nat.factors n).toFinset.card

/-- For any positive integer n, log(n) ≥ k * log(2), where k is the number of distinct prime factors of n -/
theorem log_lower_bound (n : ℕ+) :
  Real.log n ≥ (num_distinct_prime_factors n : ℝ) * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_log_lower_bound_l352_35271


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l352_35289

theorem circle_equation_through_points : 
  let equation (x y : ℝ) := x^2 + y^2 - 4*x - 6*y
  ∀ (x y : ℝ), 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) → 
    equation x y = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l352_35289


namespace NUMINAMATH_CALUDE_theater_eye_colors_l352_35224

theorem theater_eye_colors (total : ℕ) (blue brown black green : ℕ) : 
  total = 100 →
  blue = 19 →
  brown = total / 2 →
  black = total / 4 →
  green = total - (blue + brown + black) →
  green = 6 := by
sorry

end NUMINAMATH_CALUDE_theater_eye_colors_l352_35224


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_constant_l352_35205

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Defines a point on a parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

/-- Defines the perpendicular bisector of a line segment -/
def PerpendicularBisector (A B M : ℝ × ℝ) : Prop :=
  -- Definition of perpendicular bisector passing through M
  sorry

/-- Main theorem -/
theorem midpoint_x_coordinate_constant
  (p : Parabola)
  (A B : ℝ × ℝ)
  (hA : PointOnParabola p A.1 A.2)
  (hB : PointOnParabola p B.1 B.2)
  (hAB : A.2 - B.2 ≠ 0)  -- AB not perpendicular to x-axis
  (hM : PerpendicularBisector A B (4, 0)) :
  (A.1 + B.1) / 2 = 2 :=
sorry

/-- Setup for the specific problem -/
def problem_setup : Parabola :=
  { equation := fun x y => y^2 = 4*x
  , focus := (1, 0) }

#check midpoint_x_coordinate_constant problem_setup

end NUMINAMATH_CALUDE_midpoint_x_coordinate_constant_l352_35205


namespace NUMINAMATH_CALUDE_snatch_percentage_increase_l352_35233

/-- Calculates the percentage increase in Snatch weight given initial weights and new total -/
theorem snatch_percentage_increase
  (initial_clean_jerk : ℝ)
  (initial_snatch : ℝ)
  (new_total : ℝ)
  (h1 : initial_clean_jerk = 80)
  (h2 : initial_snatch = 50)
  (h3 : new_total = 250)
  (h4 : 2 * initial_clean_jerk + new_snatch = new_total)
  : (new_snatch - initial_snatch) / initial_snatch * 100 = 80 :=
by
  sorry

#check snatch_percentage_increase

end NUMINAMATH_CALUDE_snatch_percentage_increase_l352_35233


namespace NUMINAMATH_CALUDE_parallelogram_side_length_comparison_l352_35280

/-- Represents a parallelogram in 2D space -/
structure Parallelogram :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Checks if parallelogram inner is inside parallelogram outer -/
def is_inside (inner outer : Parallelogram) : Prop := sorry

/-- Checks if the vertices of inner are on the edges of outer -/
def vertices_on_edges (inner outer : Parallelogram) : Prop := sorry

/-- Checks if the sides of para1 are parallel to the sides of para2 -/
def sides_parallel (para1 para2 : Parallelogram) : Prop := sorry

/-- Computes the length of a side of a parallelogram -/
def side_length (p : Parallelogram) (side : Fin 4) : ℝ := sorry

theorem parallelogram_side_length_comparison 
  (P1 P2 P3 : Parallelogram) 
  (h1 : is_inside P3 P2)
  (h2 : is_inside P2 P1)
  (h3 : vertices_on_edges P3 P2)
  (h4 : vertices_on_edges P2 P1)
  (h5 : sides_parallel P3 P1) :
  ∃ (side : Fin 4), side_length P3 side ≥ (side_length P1 side) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_comparison_l352_35280


namespace NUMINAMATH_CALUDE_quadratic_function_range_l352_35258

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem quadratic_function_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, ∃ y ∈ Set.Icc (-4 : ℝ) 12, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-4 : ℝ) 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l352_35258


namespace NUMINAMATH_CALUDE_product_evaluation_l352_35216

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l352_35216


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l352_35202

/-- Given a circle with center (2,3) and one endpoint of a diameter at (-1,-1),
    the other endpoint of the diameter is at (5,7). -/
theorem circle_diameter_endpoint (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  O = (2, 3) → A = (-1, -1) → 
  (O.1 - A.1 = B.1 - O.1 ∧ O.2 - A.2 = B.2 - O.2) → 
  B = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l352_35202


namespace NUMINAMATH_CALUDE_complex_sum_inverse_real_iff_unit_magnitude_l352_35215

theorem complex_sum_inverse_real_iff_unit_magnitude 
  (a b : ℝ) (hb : b ≠ 0) : 
  let z : ℂ := Complex.mk a b
  (z + z⁻¹).im = 0 ↔ Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_inverse_real_iff_unit_magnitude_l352_35215


namespace NUMINAMATH_CALUDE_rhombus_area_l352_35207

/-- Rhombus in a plane rectangular coordinate system -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a rhombus given its vertices -/
def area (r : Rhombus) : ℝ := sorry

/-- Theorem: Area of rhombus ABCD with given conditions -/
theorem rhombus_area : 
  ∀ (r : Rhombus), 
    r.A = (-4, 0) →
    r.B = (0, -3) →
    (∃ (x y : ℝ), r.C = (x, 0) ∧ r.D = (0, y)) →  -- vertices on axes
    area r = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l352_35207


namespace NUMINAMATH_CALUDE_inscribed_rhombus_triangle_sides_l352_35221

/-- A triangle with an inscribed rhombus -/
structure InscribedRhombusTriangle where
  -- Side lengths of the triangle
  BC : ℝ
  AB : ℝ
  AC : ℝ
  -- Length of rhombus side
  m : ℝ
  -- Segments of BC
  p : ℝ
  q : ℝ
  -- Conditions
  rhombus_inscribed : m > 0
  positive_segments : p > 0 ∧ q > 0
  k_on_bc : BC = p + q

/-- Theorem: The sides of the triangle with an inscribed rhombus -/
theorem inscribed_rhombus_triangle_sides 
  (t : InscribedRhombusTriangle) : 
  t.BC = t.p + t.q ∧ 
  t.AB = t.m * (t.p + t.q) / t.q ∧ 
  t.AC = t.m * (t.p + t.q) / t.p :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_triangle_sides_l352_35221


namespace NUMINAMATH_CALUDE_sin_2alpha_over_cos_2beta_l352_35241

theorem sin_2alpha_over_cos_2beta (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (α - β) = 3) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_over_cos_2beta_l352_35241


namespace NUMINAMATH_CALUDE_cos_225_degrees_l352_35293

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l352_35293


namespace NUMINAMATH_CALUDE_visitors_calculation_l352_35283

/-- The number of visitors to Buckingham Palace on a specific day, given the total visitors over 85 days and the visitors on the previous day. -/
def visitors_on_day (total_visitors : ℕ) (previous_day_visitors : ℕ) : ℕ :=
  total_visitors - previous_day_visitors

/-- Theorem stating that the number of visitors on a specific day is equal to
    the total visitors over 85 days minus the visitors on the previous day. -/
theorem visitors_calculation (total_visitors previous_day_visitors : ℕ) 
    (h1 : total_visitors = 829)
    (h2 : previous_day_visitors = 45) :
  visitors_on_day total_visitors previous_day_visitors = 784 := by
  sorry

#eval visitors_on_day 829 45

end NUMINAMATH_CALUDE_visitors_calculation_l352_35283


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l352_35282

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_factorial_eight_ten :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l352_35282


namespace NUMINAMATH_CALUDE_plane_equation_proof_l352_35247

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of a plane in 3D space -/
structure PlaneEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given points A, B, and C, proves that the equation x + 2y + 4z - 5 = 0
    represents the plane passing through point A and perpendicular to vector BC -/
theorem plane_equation_proof 
  (A : Point3D) 
  (B : Point3D) 
  (C : Point3D) 
  (h1 : A.x = -7 ∧ A.y = 0 ∧ A.z = 3)
  (h2 : B.x = 1 ∧ B.y = -5 ∧ B.z = -4)
  (h3 : C.x = 2 ∧ C.y = -3 ∧ C.z = 0) :
  let BC : Vector3D := ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩
  let plane : PlaneEquation := ⟨1, 2, 4, -5⟩
  (plane.a * (A.x - x) + plane.b * (A.y - y) + plane.c * (A.z - z) = 0) ∧
  (plane.a * BC.x + plane.b * BC.y + plane.c * BC.z = 0) :=
by sorry


end NUMINAMATH_CALUDE_plane_equation_proof_l352_35247


namespace NUMINAMATH_CALUDE_valid_colorings_3x10_l352_35287

/-- Represents the number of ways to color a 3 × 2n grid with black and white,
    such that no five squares in an 'X' configuration are all the same color. -/
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 8
  | n+2 => 7 * a n + 4 * a (n-1)

/-- The number of valid colorings for a 3 × 10 grid -/
def N : ℕ := (a 5)^2

/-- Theorem stating that the number of valid colorings for a 3 × 10 grid
    is equal to 25636^2 -/
theorem valid_colorings_3x10 : N = 25636^2 := by
  sorry

end NUMINAMATH_CALUDE_valid_colorings_3x10_l352_35287


namespace NUMINAMATH_CALUDE_final_sign_is_minus_l352_35239

/-- Represents the two possible signs on the board -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the board -/
structure Board :=
  (plusCount : Nat)
  (minusCount : Nat)

/-- Applies the transformation rule to two signs -/
def transform (s1 s2 : Sign) : Sign :=
  match s1, s2 with
  | Sign.Plus, Sign.Plus => Sign.Plus
  | Sign.Minus, Sign.Minus => Sign.Plus
  | _, _ => Sign.Minus

/-- Theorem stating that the final sign will be minus -/
theorem final_sign_is_minus 
  (initial : Board)
  (h_initial_plus : initial.plusCount = 2004)
  (h_initial_minus : initial.minusCount = 2005) :
  ∃ (final : Board), final.plusCount + final.minusCount = 1 ∧ final.minusCount = 1 := by
  sorry


end NUMINAMATH_CALUDE_final_sign_is_minus_l352_35239


namespace NUMINAMATH_CALUDE_acid_solution_mixture_l352_35275

/-- Proves that adding 40 ounces of pure water and 200/9 ounces of 10% acid solution
    to 40 ounces of 25% acid solution results in a 15% acid solution. -/
theorem acid_solution_mixture : 
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.25
  let water_added : ℝ := 40
  let dilute_solution_added : ℝ := 200 / 9
  let dilute_concentration : ℝ := 0.1
  let final_concentration : ℝ := 0.15
  let final_volume : ℝ := initial_volume + water_added + dilute_solution_added
  let final_acid_amount : ℝ := initial_volume * initial_concentration + 
                                dilute_solution_added * dilute_concentration
  final_acid_amount / final_volume = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_acid_solution_mixture_l352_35275


namespace NUMINAMATH_CALUDE_min_sum_and_inequality_l352_35203

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- State the theorem
theorem min_sum_and_inequality (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x, f x a b ≥ 4) : 
  a + b ≥ 4 ∧ (a + b = 4 → 1/a + 4/b ≥ 9/4) := by
  sorry


end NUMINAMATH_CALUDE_min_sum_and_inequality_l352_35203
