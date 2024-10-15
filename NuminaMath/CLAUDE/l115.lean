import Mathlib

namespace NUMINAMATH_CALUDE_fence_length_l115_11503

/-- Given a straight wire fence with 12 equally spaced posts, where the distance between
    the third and the sixth post is 3.3 m, the total length of the fence is 12.1 meters. -/
theorem fence_length (num_posts : ℕ) (distance_3_to_6 : ℝ) :
  num_posts = 12 →
  distance_3_to_6 = 3.3 →
  (num_posts - 1 : ℝ) * (distance_3_to_6 / 3) = 12.1 := by
  sorry

end NUMINAMATH_CALUDE_fence_length_l115_11503


namespace NUMINAMATH_CALUDE_spade_problem_l115_11564

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_problem : spade 5 (spade 3 9) = 1 := by sorry

end NUMINAMATH_CALUDE_spade_problem_l115_11564


namespace NUMINAMATH_CALUDE_broken_bulbs_in_foyer_l115_11545

/-- The number of light bulbs in the kitchen -/
def kitchen_bulbs : ℕ := 35

/-- The fraction of broken light bulbs in the kitchen -/
def kitchen_broken_fraction : ℚ := 3 / 5

/-- The fraction of broken light bulbs in the foyer -/
def foyer_broken_fraction : ℚ := 1 / 3

/-- The number of light bulbs not broken in both the foyer and kitchen -/
def total_not_broken : ℕ := 34

/-- The number of broken light bulbs in the foyer -/
def foyer_broken : ℕ := 10

theorem broken_bulbs_in_foyer :
  foyer_broken = 10 := by sorry

end NUMINAMATH_CALUDE_broken_bulbs_in_foyer_l115_11545


namespace NUMINAMATH_CALUDE_original_numbers_proof_l115_11562

theorem original_numbers_proof : ∃ (a b c : ℕ), 
  a + b = 39 ∧ 
  b + c = 96 ∧ 
  a = 21 ∧ 
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_original_numbers_proof_l115_11562


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l115_11541

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_3402 : 
  largest_perfect_square_factor 3402 = 81 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l115_11541


namespace NUMINAMATH_CALUDE_phi_11_0_decomposition_l115_11553

/-- The Φ₁₁⁰ series -/
def phi_11_0 : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1
| 3 => 2
| 4 => 3
| 5 => 5
| 6 => 8
| 7 => 2
| 8 => 10
| 9 => 1
| n + 10 => phi_11_0 n

/-- The decomposed series -/
def c (n : ℕ) : ℚ := 3 * 8^n + 8 * 4^n

/-- Predicate to check if a sequence is an 11-arithmetic Fibonacci series -/
def is_11_arithmetic_fibonacci (f : ℕ → ℚ) : Prop :=
  ∀ n, f (n + 11) = f (n + 10) + f (n + 9)

/-- Predicate to check if a sequence is a geometric progression -/
def is_geometric_progression (f : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n, f (n + 1) = r * f n

theorem phi_11_0_decomposition :
  (∃ f g : ℕ → ℚ, 
    (∀ n, c n = f n + g n) ∧
    is_11_arithmetic_fibonacci f ∧
    is_11_arithmetic_fibonacci g ∧
    is_geometric_progression f ∧
    is_geometric_progression g) ∧
  (∀ n, (phi_11_0 n : ℚ) = c n) :=
sorry

end NUMINAMATH_CALUDE_phi_11_0_decomposition_l115_11553


namespace NUMINAMATH_CALUDE_trisection_intersection_l115_11595

theorem trisection_intersection (f : ℝ → ℝ) (A B C E : ℝ × ℝ) :
  f = (λ x => Real.exp x) →
  A = (0, 1) →
  B = (3, Real.exp 3) →
  C.1 = 1 →
  C.2 = 2/3 * A.2 + 1/3 * B.2 →
  E.2 = C.2 →
  f E.1 = E.2 →
  E.1 = Real.log ((2 + Real.exp 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_trisection_intersection_l115_11595


namespace NUMINAMATH_CALUDE_grocery_store_bottles_l115_11502

theorem grocery_store_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 28) (h2 : diet_soda = 2) : 
  regular_soda + diet_soda = 30 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_bottles_l115_11502


namespace NUMINAMATH_CALUDE_same_color_probability_l115_11540

def total_marbles : ℕ := 9
def marbles_per_color : ℕ := 3
def num_draws : ℕ := 3

theorem same_color_probability :
  let prob_same_color := (marbles_per_color / total_marbles) ^ num_draws * 3
  prob_same_color = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l115_11540


namespace NUMINAMATH_CALUDE_parabola_range_l115_11561

theorem parabola_range (a b m : ℝ) : 
  (∃ x y : ℝ, y = -x^2 + 2*a*x + b ∧ y = x^2) →
  (m*a - (a^2 + b) - 2*m + 1 = 0) →
  (m ≥ 5/2 ∨ m ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_range_l115_11561


namespace NUMINAMATH_CALUDE_salt_solution_volume_l115_11505

/-- Given a salt solution where 25 cubic centimeters contain 0.375 grams of salt,
    the volume of solution containing 15 grams of salt is 1000 cubic centimeters. -/
theorem salt_solution_volume (volume : ℝ) (salt_mass : ℝ) 
    (h1 : volume > 0)
    (h2 : salt_mass > 0)
    (h3 : 25 / volume = 0.375 / salt_mass) : 
  volume * (15 / salt_mass) = 1000 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l115_11505


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l115_11569

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The actual counts of balls in the box -/
def initialCounts : BallCounts :=
  { red := 35, green := 27, yellow := 22, blue := 18, white := 15, black := 12 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : Nat := 20

/-- Theorem stating the minimum number of balls to draw to guarantee the target count -/
theorem min_balls_to_draw (counts : BallCounts) (target : Nat) :
  counts = initialCounts → target = targetCount →
  (∃ (n : Nat), n = 103 ∧
    (∀ (m : Nat), m < n →
      ¬∃ (color : Nat), color ≥ target ∧
        (color ≤ counts.red ∨ color ≤ counts.green ∨ color ≤ counts.yellow ∨
         color ≤ counts.blue ∨ color ≤ counts.white ∨ color ≤ counts.black)) ∧
    (∃ (color : Nat), color ≥ target ∧
      (color ≤ counts.red ∨ color ≤ counts.green ∨ color ≤ counts.yellow ∨
       color ≤ counts.blue ∨ color ≤ counts.white ∨ color ≤ counts.black))) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_l115_11569


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_four_l115_11557

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (((2 * x + 2) / (x^2 - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1))) = x - 1 :=
by sorry

theorem evaluate_at_four : 
  (((2 * 4 + 2) / (4^2 - 1) + 1) / ((4 + 1) / (4^2 - 2*4 + 1))) = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_four_l115_11557


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l115_11530

theorem solution_satisfies_system :
  let x : ℚ := 130/161
  let y : ℚ := 76/23
  let z : ℚ := 3
  (7 * x - 3 * y + 2 * z = 4) ∧
  (4 * y - x - 5 * z = -3) ∧
  (3 * x + 2 * y - z = 7) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l115_11530


namespace NUMINAMATH_CALUDE_john_profit_l115_11550

/-- Calculates the profit from buying and selling ducks -/
def duck_profit (num_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (selling_price_per_pound : ℚ) : ℚ :=
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_weight * selling_price_per_pound
  total_revenue - total_cost

/-- Proves that John's profit is $300 -/
theorem john_profit :
  duck_profit 30 10 4 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_john_profit_l115_11550


namespace NUMINAMATH_CALUDE_problem_statement_l115_11510

theorem problem_statement : |1 - Real.sqrt 3| - Real.sqrt 3 * (Real.sqrt 3 + 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l115_11510


namespace NUMINAMATH_CALUDE_multiplication_value_l115_11552

theorem multiplication_value : 
  let original_number : ℝ := 6.5
  let divisor : ℝ := 6
  let result : ℝ := 13
  let multiplication_factor : ℝ := 12
  (original_number / divisor) * multiplication_factor = result := by
sorry

end NUMINAMATH_CALUDE_multiplication_value_l115_11552


namespace NUMINAMATH_CALUDE_ellie_bike_oil_needed_l115_11581

/-- The amount of oil needed to fix a bicycle --/
def oil_needed (oil_per_wheel : ℕ) (oil_for_rest : ℕ) (num_wheels : ℕ) : ℕ :=
  oil_per_wheel * num_wheels + oil_for_rest

/-- Theorem: The total amount of oil needed to fix Ellie's bike is 25ml --/
theorem ellie_bike_oil_needed :
  oil_needed 10 5 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ellie_bike_oil_needed_l115_11581


namespace NUMINAMATH_CALUDE_root_bound_average_l115_11549

theorem root_bound_average (A B C D : ℝ) 
  (h1 : ∀ x : ℂ, x^2 + A*x + B = 0 → Complex.abs x < 1)
  (h2 : ∀ x : ℂ, x^2 + C*x + D = 0 → Complex.abs x < 1) :
  ∀ x : ℂ, x^2 + ((A+C)/2)*x + ((B+D)/2) = 0 → Complex.abs x < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_root_bound_average_l115_11549


namespace NUMINAMATH_CALUDE_evaluate_expression_l115_11512

theorem evaluate_expression (a : ℝ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l115_11512


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l115_11559

theorem tangent_line_to_circle (x y : ℝ) : 
  -- The line is perpendicular to y = x + 1
  (∀ x₁ y₁ x₂ y₂ : ℝ, y₁ = x₁ + 1 → y₂ = x₂ + 1 → (y₂ - y₁) * (x + y - Real.sqrt 2 - y₁) = -(x₂ - x₁)) →
  -- The line is tangent to the circle x^2 + y^2 = 1
  ((x^2 + y^2 = 1 ∧ x + y - Real.sqrt 2 = 0) → 
    ∀ a b : ℝ, a^2 + b^2 = 1 → (a + b - Real.sqrt 2) * (a + b - Real.sqrt 2) ≥ 0) →
  -- The tangent point is in the first quadrant
  (x > 0 ∧ y > 0) →
  -- The equation of the line is x + y - √2 = 0
  x + y - Real.sqrt 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l115_11559


namespace NUMINAMATH_CALUDE_triangle_center_distance_l115_11579

/-- Given a triangle with circumradius R, inradius r, and distance d between
    the circumcenter and incenter, prove that d^2 = R^2 - 2Rr. -/
theorem triangle_center_distance (R r d : ℝ) (hR : R > 0) (hr : r > 0) (hd : d > 0) :
  d^2 = R^2 - 2*R*r := by
  sorry

end NUMINAMATH_CALUDE_triangle_center_distance_l115_11579


namespace NUMINAMATH_CALUDE_fourth_year_area_l115_11514

def initial_area : ℝ := 10000
def annual_increase : ℝ := 0.2

def area_after_n_years (n : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ n

theorem fourth_year_area :
  area_after_n_years 3 = 17280 :=
by sorry

end NUMINAMATH_CALUDE_fourth_year_area_l115_11514


namespace NUMINAMATH_CALUDE_mary_has_fifty_cards_l115_11551

/-- The number of Pokemon cards Mary has after receiving new cards from Sam -/
def marys_final_cards (initial_cards torn_cards new_cards : ℕ) : ℕ :=
  initial_cards - torn_cards + new_cards

/-- Theorem stating that Mary has 50 Pokemon cards after the given scenario -/
theorem mary_has_fifty_cards :
  marys_final_cards 33 6 23 = 50 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_fifty_cards_l115_11551


namespace NUMINAMATH_CALUDE_infinitely_many_sqrt_eight_eight_eight_l115_11547

theorem infinitely_many_sqrt_eight_eight_eight (k : ℕ) : 
  (9 * k - 1 + 0.888 : ℝ) < Real.sqrt (81 * k^2 - 2 * k) ∧ 
  Real.sqrt (81 * k^2 - 2 * k) < (9 * k - 1 + 0.889 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_sqrt_eight_eight_eight_l115_11547


namespace NUMINAMATH_CALUDE_max_parts_formula_max_parts_special_cases_l115_11537

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of parts a plane can be divided into by n lines is (n^2 + n + 2) / 2 -/
theorem max_parts_formula (n : ℕ) : max_parts n = (n^2 + n + 2) / 2 := by
  sorry

/-- Corollary: Special cases for n = 1, 2, 3, and 4 -/
theorem max_parts_special_cases :
  (max_parts 1 = 2) ∧
  (max_parts 2 = 4) ∧
  (max_parts 3 = 7) ∧
  (max_parts 4 = 11) := by
  sorry

end NUMINAMATH_CALUDE_max_parts_formula_max_parts_special_cases_l115_11537


namespace NUMINAMATH_CALUDE_binomial_not_divisible_by_prime_l115_11591

theorem binomial_not_divisible_by_prime (p : ℕ) (n : ℕ) : 
  Prime p → 
  (∀ m : ℕ, m ≤ n → ¬(p ∣ Nat.choose n m)) ↔ 
  ∃ k s : ℕ, n = s * p^k - 1 ∧ 1 ≤ s ∧ s ≤ p :=
by sorry

end NUMINAMATH_CALUDE_binomial_not_divisible_by_prime_l115_11591


namespace NUMINAMATH_CALUDE_james_remaining_money_l115_11558

def weekly_allowance : ℕ := 10
def saving_weeks : ℕ := 4
def video_game_fraction : ℚ := 1/2
def book_fraction : ℚ := 1/4

theorem james_remaining_money :
  let total_savings := weekly_allowance * saving_weeks
  let after_video_game := total_savings * (1 - video_game_fraction)
  let book_cost := after_video_game * book_fraction
  let remaining := after_video_game - book_cost
  remaining = 15 := by sorry

end NUMINAMATH_CALUDE_james_remaining_money_l115_11558


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l115_11571

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_regular_hexagon : ℝ := 120

/-- Theorem: The measure of each interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle :
  interior_angle_regular_hexagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l115_11571


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l115_11598

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l115_11598


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l115_11594

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  1 / a + 1 / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l115_11594


namespace NUMINAMATH_CALUDE_wilson_theorem_plus_one_l115_11529

theorem wilson_theorem_plus_one (p : Nat) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  (p - 1).factorial + 1 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_wilson_theorem_plus_one_l115_11529


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l115_11517

theorem sum_of_roots_zero (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  p = -q → 
  p + q = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l115_11517


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l115_11596

theorem arctan_equation_solution (x : ℝ) : 
  Real.arctan (1 / x^2) + Real.arctan (1 / x^4) = π / 4 ↔ 
  x = Real.sqrt ((1 + Real.sqrt 5) / 2) ∨ x = -Real.sqrt ((1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l115_11596


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l115_11518

theorem largest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 3 * b →
  5 * a = 3 * c →
  d = 5 * a / 2 →
  d = 480 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l115_11518


namespace NUMINAMATH_CALUDE_four_genuine_probability_l115_11506

/-- The number of genuine coins -/
def genuine_coins : ℕ := 12

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- The total number of coins -/
def total_coins : ℕ := genuine_coins + counterfeit_coins

/-- The probability of selecting 4 genuine coins when drawing two pairs randomly without replacement -/
def prob_four_genuine : ℚ := 33 / 91

theorem four_genuine_probability :
  (genuine_coins.choose 2 * (genuine_coins - 2).choose 2) / (total_coins.choose 2 * (total_coins - 2).choose 2) = prob_four_genuine := by
  sorry

end NUMINAMATH_CALUDE_four_genuine_probability_l115_11506


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l115_11531

def n : ℕ := 2^31 * 3^19 * 5^7

-- Function to count divisors of a number given its prime factorization
def count_divisors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (λ acc (_, exp) => acc * (exp + 1)) 1

-- Function to count divisors less than n
def count_divisors_less_than_n (total_divisors : ℕ) : ℕ :=
  (total_divisors - 1) / 2

theorem divisors_of_n_squared_less_than_n_not_dividing_n :
  let n_squared_factorization : List (ℕ × ℕ) := [(2, 62), (3, 38), (5, 14)]
  let n_factorization : List (ℕ × ℕ) := [(2, 31), (3, 19), (5, 7)]
  let total_divisors_n_squared := count_divisors n_squared_factorization
  let divisors_less_than_n := count_divisors_less_than_n total_divisors_n_squared
  let divisors_of_n := count_divisors n_factorization
  divisors_less_than_n - divisors_of_n = 13307 :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l115_11531


namespace NUMINAMATH_CALUDE_unique_solution_mn_l115_11568

theorem unique_solution_mn : ∃! (m n : ℕ+), 18 * m * n = 63 - 9 * m - 3 * n ∧ m = 7 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l115_11568


namespace NUMINAMATH_CALUDE_distinct_feeding_sequences_l115_11587

def number_of_pairs : ℕ := 5

def feeding_sequence (n : ℕ) : ℕ := 
  match n with
  | 0 => 1  -- The first animal (male lion) is fixed
  | 1 => number_of_pairs  -- First choice of female
  | k => if k % 2 = 0 then number_of_pairs - k / 2 else number_of_pairs - (k - 1) / 2

theorem distinct_feeding_sequences :
  (List.range (2 * number_of_pairs)).foldl (fun acc i => acc * feeding_sequence i) 1 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_distinct_feeding_sequences_l115_11587


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l115_11583

/-- The minimum number of buses needed to transport students for a field trip. -/
def min_buses (total_students : ℕ) (bus_capacity : ℕ) (min_buses : ℕ) : ℕ :=
  max (min_buses) (((total_students + bus_capacity - 1) / bus_capacity) : ℕ)

/-- Theorem stating the minimum number of buses needed for the given conditions. -/
theorem min_buses_for_field_trip :
  min_buses 500 45 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l115_11583


namespace NUMINAMATH_CALUDE_max_value_of_f_l115_11589

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 6 * x

-- Define the interval
def I : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 9/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l115_11589


namespace NUMINAMATH_CALUDE_rectangle_area_measurement_error_l115_11576

theorem rectangle_area_measurement_error (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let true_area := L * W
  let measured_length := 1.20 * L
  let measured_width := 0.90 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - true_area
  let error_percent := (error / true_area) * 100
  error_percent = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_measurement_error_l115_11576


namespace NUMINAMATH_CALUDE_complex_division_proof_l115_11534

theorem complex_division_proof : ∀ (i : ℂ), i^2 = -1 → (1 : ℂ) / (1 + i) = (1 - i) / 2 := by sorry

end NUMINAMATH_CALUDE_complex_division_proof_l115_11534


namespace NUMINAMATH_CALUDE_road_travel_cost_l115_11555

/-- The cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost (lawn_length lawn_width road_width : ℕ) (cost_per_sqm : ℚ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 2 → 
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 2600 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_cost_l115_11555


namespace NUMINAMATH_CALUDE_bakery_storage_theorem_l115_11575

def bakery_storage_problem (sugar : ℕ) (flour : ℕ) (baking_soda : ℕ) (added_baking_soda : ℕ) : Prop :=
  sugar = 2400 ∧
  sugar = flour ∧
  10 * baking_soda = flour ∧
  added_baking_soda = 60 ∧
  8 * (baking_soda + added_baking_soda) = flour

theorem bakery_storage_theorem :
  ∃ (sugar flour baking_soda added_baking_soda : ℕ),
    bakery_storage_problem sugar flour baking_soda added_baking_soda :=
by
  sorry

end NUMINAMATH_CALUDE_bakery_storage_theorem_l115_11575


namespace NUMINAMATH_CALUDE_expansion_and_factorization_l115_11509

theorem expansion_and_factorization :
  (∀ y : ℝ, (y - 1) * (y + 5) = y^2 + 4*y - 5) ∧
  (∀ x y : ℝ, -x^2 + 4*x*y - 4*y^2 = -(x - 2*y)^2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_and_factorization_l115_11509


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l115_11578

/-- The common factor of a polynomial 4x(m-n) + 2y(m-n)^2 is 2(m-n) -/
theorem common_factor_of_polynomial (x y m n : ℤ) :
  ∃ (k : ℤ), (4*x*(m-n) + 2*y*(m-n)^2) = 2*(m-n) * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l115_11578


namespace NUMINAMATH_CALUDE_hot_chocolate_consumption_l115_11521

/-- The number of cups of hot chocolate John drinks in 5 hours -/
def cups_in_five_hours : ℕ := 15

/-- The time interval between each cup of hot chocolate in minutes -/
def interval : ℕ := 20

/-- The total time in minutes -/
def total_time : ℕ := 5 * 60

theorem hot_chocolate_consumption :
  cups_in_five_hours = total_time / interval :=
by sorry

end NUMINAMATH_CALUDE_hot_chocolate_consumption_l115_11521


namespace NUMINAMATH_CALUDE_log_base_condition_l115_11526

theorem log_base_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ici 2 → |Real.log x / Real.log a| > 1) → 
  (a < 2 ∧ a ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_log_base_condition_l115_11526


namespace NUMINAMATH_CALUDE_angle_between_vectors_l115_11511

/-- Given vectors a, b, and c in a real inner product space,
    if their norms are equal and nonzero, and a + b = √3 * c,
    then the angle between a and c is π/6. -/
theorem angle_between_vectors (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b c : V) (h1 : ‖a‖ = ‖b‖) (h2 : ‖b‖ = ‖c‖) (h3 : ‖a‖ ≠ 0)
  (h4 : a + b = Real.sqrt 3 • c) :
  Real.arccos (inner a c / (‖a‖ * ‖c‖)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l115_11511


namespace NUMINAMATH_CALUDE_exists_special_subset_l115_11592

/-- Given a set of 40 elements and a function that maps each 19-element subset to a unique element (common friend), 
    there exists a 20-element subset M₀ such that for all a ∈ M₀, the common friend of M₀ \ {a} is not a. -/
theorem exists_special_subset (I : Finset Nat) (f : Finset Nat → Nat) : 
  I.card = 40 → 
  (∀ A : Finset Nat, A ⊆ I → A.card = 19 → f A ∈ I) →
  (∀ A : Finset Nat, A ⊆ I → A.card = 19 → f A ∉ A) →
  ∃ M₀ : Finset Nat, M₀ ⊆ I ∧ M₀.card = 20 ∧ 
    ∀ a ∈ M₀, f (M₀ \ {a}) ≠ a := by
  sorry

end NUMINAMATH_CALUDE_exists_special_subset_l115_11592


namespace NUMINAMATH_CALUDE_smoothie_ingredients_total_l115_11504

theorem smoothie_ingredients_total (strawberries yogurt orange_juice : ℚ) 
  (h1 : strawberries = 0.2)
  (h2 : yogurt = 0.1)
  (h3 : orange_juice = 0.2) :
  strawberries + yogurt + orange_juice = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_ingredients_total_l115_11504


namespace NUMINAMATH_CALUDE_melanie_missed_games_l115_11522

/-- The number of football games Melanie missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Melanie missed 4 games given the conditions -/
theorem melanie_missed_games :
  let total_games : ℕ := 7
  let attended_games : ℕ := 3
  games_missed total_games attended_games = 4 := by
  sorry


end NUMINAMATH_CALUDE_melanie_missed_games_l115_11522


namespace NUMINAMATH_CALUDE_matt_current_age_l115_11519

def james_age_3_years_ago : ℕ := 27
def years_since_james_age : ℕ := 3
def years_until_matt_double : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_since_james_age

def james_future_age : ℕ := james_current_age + years_until_matt_double

def matt_future_age : ℕ := 2 * james_future_age

theorem matt_current_age : matt_future_age - years_until_matt_double = 65 := by
  sorry

end NUMINAMATH_CALUDE_matt_current_age_l115_11519


namespace NUMINAMATH_CALUDE_operations_are_finite_l115_11597

/-- Represents a (2n+1)-gon with integers assigned to its vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℤ
  sum_positive : 0 < (Finset.univ.sum vertices)

/-- Represents an operation on three consecutive vertices -/
def operation (p : Polygon n) (i : Fin (2*n+1)) : Polygon n :=
  sorry

/-- Predicate to check if an operation is valid (i.e., y < 0) -/
def is_valid_operation (p : Polygon n) (i : Fin (2*n+1)) : Prop :=
  sorry

/-- A sequence of operations -/
def operation_sequence (p : Polygon n) : List (Fin (2*n+1)) → Polygon n
  | [] => p
  | (i :: is) => operation_sequence (operation p i) is

/-- Theorem stating that any sequence of valid operations is finite -/
theorem operations_are_finite (n : ℕ) (p : Polygon n) :
  ∃ (N : ℕ), ∀ (seq : List (Fin (2*n+1))),
    (∀ i ∈ seq, is_valid_operation p i) →
    seq.length ≤ N :=
  sorry

end NUMINAMATH_CALUDE_operations_are_finite_l115_11597


namespace NUMINAMATH_CALUDE_tunnel_construction_equation_l115_11585

/-- Represents the tunnel construction scenario -/
def tunnel_construction (x : ℝ) : Prop :=
  let total_length : ℝ := 1280
  let increased_speed : ℝ := 1.4 * x
  let weeks_saved : ℝ := 2
  (total_length - x) / x = (total_length - x) / increased_speed + weeks_saved

theorem tunnel_construction_equation :
  ∀ x : ℝ, x > 0 → tunnel_construction x :=
by
  sorry

end NUMINAMATH_CALUDE_tunnel_construction_equation_l115_11585


namespace NUMINAMATH_CALUDE_prob_white_then_red_is_four_fifteenths_l115_11535

/-- Represents the number of red marbles in the bag -/
def red_marbles : ℕ := 4

/-- Represents the number of white marbles in the bag -/
def white_marbles : ℕ := 6

/-- Represents the total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles

/-- The probability of drawing a white marble first and a red marble second -/
def prob_white_then_red : ℚ :=
  (white_marbles : ℚ) / total_marbles * red_marbles / (total_marbles - 1)

theorem prob_white_then_red_is_four_fifteenths :
  prob_white_then_red = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_then_red_is_four_fifteenths_l115_11535


namespace NUMINAMATH_CALUDE_unique_number_with_digit_product_l115_11513

/-- Given a natural number n, multiply_digits n returns the product of n and all its digits. -/
def multiply_digits (n : ℕ) : ℕ := sorry

/-- Given a natural number n, digits n returns the list of digits of n. -/
def digits (n : ℕ) : List ℕ := sorry

theorem unique_number_with_digit_product : ∃! n : ℕ, multiply_digits n = 1995 ∧ digits n = [5, 7] := by sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_product_l115_11513


namespace NUMINAMATH_CALUDE_different_color_probability_l115_11520

def total_balls : ℕ := 5
def blue_balls : ℕ := 3
def yellow_balls : ℕ := 2

theorem different_color_probability :
  let total_outcomes := Nat.choose total_balls 2
  let favorable_outcomes := blue_balls * yellow_balls
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l115_11520


namespace NUMINAMATH_CALUDE_roundness_of_1728_l115_11590

/-- Roundness of a number is defined as the sum of the exponents in its prime factorization -/
def roundness (n : Nat) : Nat :=
  sorry

/-- 1728 can be expressed as 2^6 * 3^3 -/
axiom factorization_1728 : 1728 = 2^6 * 3^3

theorem roundness_of_1728 : roundness 1728 = 9 := by
  sorry

end NUMINAMATH_CALUDE_roundness_of_1728_l115_11590


namespace NUMINAMATH_CALUDE_sum_of_extrema_l115_11536

/-- A function f(x) = 2x³ - ax² + 1 with exactly one zero in (0, +∞) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * x^3 - a * x^2 + 1

/-- The property that f has exactly one zero in (0, +∞) -/
def has_one_zero (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ f a x = 0

/-- The theorem stating that if f has one zero in (0, +∞), then the sum of its max and min on [-1, 1] is -3 -/
theorem sum_of_extrema (a : ℝ) (h : has_one_zero a) :
  (⨆ x ∈ Set.Icc (-1) 1, f a x) + (⨅ x ∈ Set.Icc (-1) 1, f a x) = -3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l115_11536


namespace NUMINAMATH_CALUDE_f_composition_equals_result_l115_11554

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

theorem f_composition_equals_result : 
  f (f (f (f (1 + 2*I)))) = (23882205 - 24212218*I)^3 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_result_l115_11554


namespace NUMINAMATH_CALUDE_john_toy_store_spending_l115_11599

def weekly_allowance : ℚ := 9/4  -- $2.25 as a rational number

def arcade_fraction : ℚ := 3/5

def candy_store_spending : ℚ := 3/5  -- $0.60 as a rational number

theorem john_toy_store_spending :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_spending := remaining_after_arcade - candy_store_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by sorry

end NUMINAMATH_CALUDE_john_toy_store_spending_l115_11599


namespace NUMINAMATH_CALUDE_unique_solution_l115_11527

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem unique_solution :
  ∀ x y : ℕ+, (3 ^ x.val + x.val ^ 4 = factorial y.val + 2019) ↔ (x = 6 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l115_11527


namespace NUMINAMATH_CALUDE_f_range_l115_11565

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -2 ≤ y ∧ y ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l115_11565


namespace NUMINAMATH_CALUDE_always_negative_quadratic_function_l115_11523

/-- The function f(x) = kx^2 - kx - 1 is always negative if and only if -4 < k ≤ 0 -/
theorem always_negative_quadratic_function (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 := by sorry

end NUMINAMATH_CALUDE_always_negative_quadratic_function_l115_11523


namespace NUMINAMATH_CALUDE_equation_equivalent_to_circles_l115_11548

def equation (x y : ℝ) : Prop :=
  x^4 - 16*x^2 + 2*x^2*y^2 - 16*y^2 + y^4 = 4*x^3 + 4*x*y^2 - 64*x

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16

def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

theorem equation_equivalent_to_circles :
  ∀ x y : ℝ, equation x y ↔ (circle1 x y ∨ circle2 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_circles_l115_11548


namespace NUMINAMATH_CALUDE_negation_square_nonnegative_l115_11582

theorem negation_square_nonnegative (x : ℝ) : 
  ¬(x ≥ 0 → x^2 > 0) ↔ (x < 0 → x^2 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_square_nonnegative_l115_11582


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l115_11560

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (1 - a * Complex.I) * (1 + Complex.I) →
  z.im = -3 →
  Complex.abs z = Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l115_11560


namespace NUMINAMATH_CALUDE_stratified_sample_second_year_l115_11543

theorem stratified_sample_second_year (total_students : ℕ) (second_year_students : ℕ) (sample_size : ℕ) : 
  total_students = 1000 →
  second_year_students = 320 →
  sample_size = 200 →
  (second_year_students * sample_size) / total_students = 64 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_second_year_l115_11543


namespace NUMINAMATH_CALUDE_oprah_car_collection_l115_11539

/-- The number of cars in Oprah's collection -/
def total_cars : ℕ := 3500

/-- The average number of cars Oprah gives away per year -/
def cars_given_per_year : ℕ := 50

/-- The number of years it takes to reduce the collection -/
def years_to_reduce : ℕ := 60

/-- The number of cars left after giving away -/
def cars_left : ℕ := 500

theorem oprah_car_collection :
  total_cars = cars_left + cars_given_per_year * years_to_reduce :=
by sorry

end NUMINAMATH_CALUDE_oprah_car_collection_l115_11539


namespace NUMINAMATH_CALUDE_three_solutions_imply_b_neg_c_zero_l115_11586

theorem three_solutions_imply_b_neg_c_zero
  (f : ℝ → ℝ)
  (b c : ℝ)
  (h1 : ∀ x, f x = x^2 + b * |x| + c)
  (h2 : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0)
  (h3 : ∀ (w v : ℝ), f w = 0 ∧ f v = 0 ∧ w ≠ v → w = x ∨ w = y ∨ w = z ∨ v = x ∨ v = y ∨ v = z) :
  b < 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_three_solutions_imply_b_neg_c_zero_l115_11586


namespace NUMINAMATH_CALUDE_freight_train_speed_proof_l115_11516

-- Define the total distance between points A and B
def total_distance : ℝ := 460

-- Define the time it takes for the trains to meet
def meeting_time : ℝ := 2

-- Define the speed of the passenger train
def passenger_train_speed : ℝ := 120

-- Define the speed of the freight train (to be proven)
def freight_train_speed : ℝ := 110

-- Theorem statement
theorem freight_train_speed_proof :
  total_distance = (passenger_train_speed + freight_train_speed) * meeting_time :=
by sorry

end NUMINAMATH_CALUDE_freight_train_speed_proof_l115_11516


namespace NUMINAMATH_CALUDE_student_a_score_l115_11574

def final_score (total_questions : ℕ) (correct_answers : ℕ) : ℤ :=
  correct_answers - 2 * (total_questions - correct_answers)

theorem student_a_score :
  final_score 100 93 = 79 := by
  sorry

end NUMINAMATH_CALUDE_student_a_score_l115_11574


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l115_11515

theorem quadratic_is_square_of_binomial (r : ℝ) (hr : r ≠ 0) :
  ∃ (p q : ℝ), ∀ x, r^2 * x^2 - 20 * x + 100 / r^2 = (p * x + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l115_11515


namespace NUMINAMATH_CALUDE_mushroom_distribution_l115_11501

theorem mushroom_distribution (morning_mushrooms afternoon_mushrooms : ℕ) 
  (rabbit_count : ℕ) (h1 : morning_mushrooms = 94) (h2 : afternoon_mushrooms = 85) 
  (h3 : rabbit_count = 8) :
  let total_mushrooms := morning_mushrooms + afternoon_mushrooms
  (total_mushrooms / rabbit_count = 22) ∧ (total_mushrooms % rabbit_count = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_mushroom_distribution_l115_11501


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l115_11533

theorem fourth_root_equation_solution :
  ∃! x : ℚ, (62 - 3*x)^(1/4) + (38 + 3*x)^(1/4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l115_11533


namespace NUMINAMATH_CALUDE_inequality_of_three_nonnegative_reals_l115_11542

theorem inequality_of_three_nonnegative_reals (a b c : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  |c * a - a * b| + |a * b - b * c| + |b * c - c * a| ≤ 
  |b^2 - c^2| + |c^2 - a^2| + |a^2 - b^2| := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_three_nonnegative_reals_l115_11542


namespace NUMINAMATH_CALUDE_continued_fraction_value_l115_11507

theorem continued_fraction_value : 
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) → x = (3 + Real.sqrt 39) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l115_11507


namespace NUMINAMATH_CALUDE_solve_star_equation_l115_11528

-- Define the * operation
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- State the theorem
theorem solve_star_equation : 
  ∃! x : ℝ, star (x - 4) 1 = 0 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solve_star_equation_l115_11528


namespace NUMINAMATH_CALUDE_johns_cloth_cost_l115_11544

/-- The total cost of cloth for John, given the length and price per metre -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem stating that John's total cost for cloth is $444 -/
theorem johns_cloth_cost : 
  total_cost 9.25 48 = 444 := by
  sorry

end NUMINAMATH_CALUDE_johns_cloth_cost_l115_11544


namespace NUMINAMATH_CALUDE_garage_cars_count_l115_11508

theorem garage_cars_count (total_wheels : ℕ) (total_bicycles : ℕ) 
  (bicycle_wheels : ℕ) (car_wheels : ℕ) :
  total_wheels = 82 →
  total_bicycles = 9 →
  bicycle_wheels = 2 →
  car_wheels = 4 →
  ∃ (total_cars : ℕ), 
    total_wheels = (total_bicycles * bicycle_wheels) + (total_cars * car_wheels) ∧
    total_cars = 16 := by
  sorry

end NUMINAMATH_CALUDE_garage_cars_count_l115_11508


namespace NUMINAMATH_CALUDE_probability_specific_case_l115_11538

/-- The probability of drawing one green, one white, and one blue ball simultaneously -/
def probability_three_colors (green white blue : ℕ) : ℚ :=
  let total := green + white + blue
  let favorable := green * white * blue
  let total_combinations := (total * (total - 1) * (total - 2)) / 6
  (favorable : ℚ) / total_combinations

/-- Theorem stating the probability of drawing one green, one white, and one blue ball -/
theorem probability_specific_case : 
  probability_three_colors 12 10 8 = 24 / 101 := by
  sorry


end NUMINAMATH_CALUDE_probability_specific_case_l115_11538


namespace NUMINAMATH_CALUDE_spanish_not_german_students_l115_11573

theorem spanish_not_german_students (total : ℕ) (both : ℕ) (spanish : ℕ) (german : ℕ) : 
  total = 30 →
  both = 2 →
  spanish = 3 * german →
  spanish + german - both = total →
  spanish - both = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_spanish_not_german_students_l115_11573


namespace NUMINAMATH_CALUDE_vector_calculation_l115_11588

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, -2)

theorem vector_calculation : 
  (2 : ℝ) • vector_a - vector_b = (5, 8) := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l115_11588


namespace NUMINAMATH_CALUDE_line_perp_para_implies_planes_perp_l115_11525

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_para_implies_planes_perp 
  (a : Line) (α β : Plane) :
  perp a α → para a β → plane_perp α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_para_implies_planes_perp_l115_11525


namespace NUMINAMATH_CALUDE_task_probability_l115_11570

theorem task_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 5/8) 
  (h2 : p2 = 3/5) 
  (h3 : p3 = 7/10) 
  (h4 : p4 = 9/12) : 
  p1 * (1 - p2) * (1 - p3) * p4 = 9/160 := by
  sorry

end NUMINAMATH_CALUDE_task_probability_l115_11570


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l115_11524

/-- The number of distinct red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of distinct white balls in the bag -/
def num_white_balls : ℕ := 6

/-- The score for drawing a red ball -/
def red_score : ℕ := 2

/-- The score for drawing a white ball -/
def white_score : ℕ := 1

/-- The number of ways to draw 4 balls such that the number of red balls is not less than the number of white balls -/
def ways_to_draw_4_balls : ℕ := 115

/-- The number of ways to draw 5 balls such that the total score is at least 7 points -/
def ways_to_draw_5_balls_score_7_plus : ℕ := 186

/-- The number of ways to arrange 5 drawn balls (with a score of 8 points) such that only two red balls are adjacent -/
def ways_to_arrange_5_balls_score_8 : ℕ := 4320

theorem ball_drawing_theorem : 
  ways_to_draw_4_balls = 115 ∧ 
  ways_to_draw_5_balls_score_7_plus = 186 ∧ 
  ways_to_arrange_5_balls_score_8 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l115_11524


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l115_11593

/-- Two vectors a and b in R² -/
def a : ℝ × ℝ := (3, 1)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, -3)

/-- The dot product of two vectors in R² -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If a and b are perpendicular, then x = 1 -/
theorem perpendicular_vectors_x_equals_one :
  (∃ x : ℝ, dot_product a (b x) = 0) → 
  (∃ x : ℝ, b x = (1, -3)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l115_11593


namespace NUMINAMATH_CALUDE_equation_solution_l115_11556

theorem equation_solution : ∃ y : ℝ, (3 * y + 7 * y = 282 - 8 * (y - 3)) ∧ y = 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l115_11556


namespace NUMINAMATH_CALUDE_number_of_divisors_720_l115_11584

theorem number_of_divisors_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_720_l115_11584


namespace NUMINAMATH_CALUDE_angle_C_measure_l115_11567

def triangle_ABC (A B C : ℝ) := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

theorem angle_C_measure (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_AC : c = Real.sqrt 6)
  (h_BC : b = 2)
  (h_angle_B : B = Real.pi / 3) :
  C = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l115_11567


namespace NUMINAMATH_CALUDE_tiger_catch_deer_distance_l115_11572

/-- The distance a tiger needs to run to catch a deer given their speeds and initial separation -/
theorem tiger_catch_deer_distance 
  (tiger_leaps_behind : ℕ)
  (tiger_leaps_per_minute : ℕ)
  (deer_leaps_per_minute : ℕ)
  (tiger_meters_per_leap : ℕ)
  (deer_meters_per_leap : ℕ)
  (h1 : tiger_leaps_behind = 50)
  (h2 : tiger_leaps_per_minute = 5)
  (h3 : deer_leaps_per_minute = 4)
  (h4 : tiger_meters_per_leap = 8)
  (h5 : deer_meters_per_leap = 5) :
  (tiger_leaps_behind * tiger_meters_per_leap * tiger_leaps_per_minute) /
  (tiger_leaps_per_minute * tiger_meters_per_leap - deer_leaps_per_minute * deer_meters_per_leap) = 800 :=
by sorry

end NUMINAMATH_CALUDE_tiger_catch_deer_distance_l115_11572


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l115_11546

theorem complex_number_real_condition (a : ℝ) : 
  (2 * Complex.I - a / (1 - Complex.I)).im = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l115_11546


namespace NUMINAMATH_CALUDE_units_digit_difference_l115_11563

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_difference (p : ℕ) 
  (h1 : p % 2 = 0) 
  (h2 : units_digit p > 0) 
  (h3 : units_digit (p + 2) = 8) : 
  units_digit (p^3) - units_digit (p^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_difference_l115_11563


namespace NUMINAMATH_CALUDE_angle_range_given_monotonic_function_l115_11577

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 2√2|b| ≠ 0 and f(x) = 2x³ + 3|a|x² + 6(a · b)x + 7 
    monotonically increasing on ℝ, prove that the angle θ between 
    a and b satisfies 0 ≤ θ ≤ π/4 -/
theorem angle_range_given_monotonic_function 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 2 * Real.sqrt 2 * ‖b‖) 
  (h2 : ‖b‖ ≠ 0) 
  (h3 : Monotone (fun x : ℝ => 2 * x^3 + 3 * ‖a‖ * x^2 + 6 * inner a b * x + 7)) :
  let θ := Real.arccos (inner a b / (‖a‖ * ‖b‖))
  0 ≤ θ ∧ θ ≤ π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_given_monotonic_function_l115_11577


namespace NUMINAMATH_CALUDE_tims_coins_value_l115_11500

/-- Represents the number of coins Tim has -/
def total_coins : ℕ := 18

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the number of dimes Tim has -/
def num_dimes : ℕ := 8

/-- Represents the number of quarters Tim has -/
def num_quarters : ℕ := 10

/-- Theorem stating the total value of Tim's coins -/
theorem tims_coins_value :
  (num_dimes * dime_value + num_quarters * quarter_value = 330) ∧
  (num_dimes + num_quarters = total_coins) ∧
  (num_dimes + 2 = num_quarters) :=
sorry

end NUMINAMATH_CALUDE_tims_coins_value_l115_11500


namespace NUMINAMATH_CALUDE_tamikas_speed_l115_11566

/-- Tamika's driving problem -/
theorem tamikas_speed (tamika_time logan_time logan_speed extra_distance : ℝ) 
  (h1 : tamika_time = 8)
  (h2 : logan_time = 5)
  (h3 : logan_speed = 55)
  (h4 : extra_distance = 85)
  : (logan_time * logan_speed + extra_distance) / tamika_time = 45 := by
  sorry

#check tamikas_speed

end NUMINAMATH_CALUDE_tamikas_speed_l115_11566


namespace NUMINAMATH_CALUDE_souvenir_sales_problem_l115_11532

/-- Souvenir sales problem -/
theorem souvenir_sales_problem
  (cost_price : ℕ)
  (initial_price : ℕ)
  (initial_sales : ℕ)
  (price_change : ℕ → ℤ)
  (sales_change : ℕ → ℤ)
  (h1 : cost_price = 40)
  (h2 : initial_price = 44)
  (h3 : initial_sales = 300)
  (h4 : ∀ x : ℕ, price_change x = x)
  (h5 : ∀ x : ℕ, sales_change x = -10 * x)
  (h6 : ∀ x : ℕ, initial_price + price_change x ≥ 44)
  (h7 : ∀ x : ℕ, initial_price + price_change x ≤ 60) :
  (∃ x : ℕ, (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) = 2640 ∧
             initial_price + price_change x = 52) ∧
  (∃ x : ℕ, ∀ y : ℕ, 
    (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) ≥
    (initial_price + price_change y - cost_price) * (initial_sales + sales_change y) ∧
    initial_price + price_change x = 57) ∧
  (∃ max_profit : ℕ, 
    (∃ x : ℕ, (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) = max_profit) ∧
    (∀ y : ℕ, (initial_price + price_change y - cost_price) * (initial_sales + sales_change y) ≤ max_profit) ∧
    max_profit = 2890) :=
by sorry

end NUMINAMATH_CALUDE_souvenir_sales_problem_l115_11532


namespace NUMINAMATH_CALUDE_expected_value_is_negative_one_fifth_l115_11580

/-- A die with two faces: Star and Moon -/
inductive DieFace
| star
| moon

/-- The probability of getting a Star face -/
def probStar : ℚ := 2/5

/-- The probability of getting a Moon face -/
def probMoon : ℚ := 3/5

/-- The winnings for Star face -/
def winStar : ℚ := 4

/-- The losses for Moon face -/
def lossMoon : ℚ := -3

/-- The expected value of one roll of the die -/
def expectedValue : ℚ := probStar * winStar + probMoon * lossMoon

theorem expected_value_is_negative_one_fifth :
  expectedValue = -1/5 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_negative_one_fifth_l115_11580
