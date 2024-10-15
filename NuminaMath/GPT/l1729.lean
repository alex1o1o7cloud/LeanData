import Mathlib

namespace NUMINAMATH_GPT_flyers_left_to_hand_out_l1729_172976

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_flyers_left_to_hand_out_l1729_172976


namespace NUMINAMATH_GPT_distance_between_points_l1729_172915

open Real -- opening real number namespace

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

theorem distance_between_points :
  let A := polar_to_cartesian 2 (π / 3)
  let B := polar_to_cartesian 2 (2 * π / 3)
  dist A B = 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1729_172915


namespace NUMINAMATH_GPT_find_wheel_diameter_l1729_172943

noncomputable def wheel_diameter (revolutions distance : ℝ) (π_approx : ℝ) : ℝ := 
  distance / (π_approx * revolutions)

theorem find_wheel_diameter : wheel_diameter 47.04276615104641 4136 3.14159 = 27.99 :=
by
  sorry

end NUMINAMATH_GPT_find_wheel_diameter_l1729_172943


namespace NUMINAMATH_GPT_green_peaches_more_than_red_l1729_172945

theorem green_peaches_more_than_red :
  let red_peaches := 5
  let green_peaches := 11
  (green_peaches - red_peaches) = 6 := by
  sorry

end NUMINAMATH_GPT_green_peaches_more_than_red_l1729_172945


namespace NUMINAMATH_GPT_ferry_speeds_l1729_172944

theorem ferry_speeds (v_P v_Q : ℝ) 
  (h1: v_P = v_Q - 1) 
  (h2: 3 * v_P * 3 = v_Q * (3 + 5))
  : v_P = 8 := 
sorry

end NUMINAMATH_GPT_ferry_speeds_l1729_172944


namespace NUMINAMATH_GPT_like_terms_exponent_l1729_172948

theorem like_terms_exponent (a : ℝ) : (2 * a = a + 3) → a = 3 := 
by
  intros h
  -- Proof here
  sorry

end NUMINAMATH_GPT_like_terms_exponent_l1729_172948


namespace NUMINAMATH_GPT_points_lie_on_parabola_l1729_172955

theorem points_lie_on_parabola (u : ℝ) :
  ∃ (x y : ℝ), x = 3^u - 4 ∧ y = 9^u - 7 * 3^u - 2 ∧ y = x^2 + x - 14 :=
by
  sorry

end NUMINAMATH_GPT_points_lie_on_parabola_l1729_172955


namespace NUMINAMATH_GPT_odd_number_expression_l1729_172993

theorem odd_number_expression (o n : ℤ) (ho : o % 2 = 1) : (o^2 + n * o + 1) % 2 = 1 ↔ n % 2 = 1 := by
  sorry

end NUMINAMATH_GPT_odd_number_expression_l1729_172993


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1729_172995

theorem line_passes_through_fixed_point (m : ℝ) :
  (m-1) * 9 + (2*m-1) * (-4) = m - 5 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1729_172995


namespace NUMINAMATH_GPT_daphne_necklaces_l1729_172922

/--
Given:
1. Total cost of necklaces and earrings is $240,000.
2. Necklaces are equal in price.
3. Earrings were three times as expensive as any one necklace.
4. Cost of a single necklace is $40,000.

Prove:
Princess Daphne bought 3 necklaces.
-/
theorem daphne_necklaces (total_cost : ℤ) (price_necklace : ℤ) (price_earrings : ℤ) (n : ℤ)
  (h1 : total_cost = 240000)
  (h2 : price_necklace = 40000)
  (h3 : price_earrings = 3 * price_necklace)
  (h4 : total_cost = n * price_necklace + price_earrings) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_daphne_necklaces_l1729_172922


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l1729_172960

theorem polynomial_coeff_sum (a0 a1 a2 a3 : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a3 * x^3 + a2 * x^2 + a1 * x + a0) →
  a0 + a1 + a2 + a3 = 27 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l1729_172960


namespace NUMINAMATH_GPT_find_number_l1729_172936

theorem find_number : ∃ x : ℝ, (x / 5 + 7 = x / 4 - 7) ∧ x = 280 :=
by
  -- Here, we state the existence of a real number x
  -- such that the given condition holds and x = 280.
  sorry

end NUMINAMATH_GPT_find_number_l1729_172936


namespace NUMINAMATH_GPT_units_digit_13_times_41_l1729_172910

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_13_times_41 :
  units_digit (13 * 41) = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_13_times_41_l1729_172910


namespace NUMINAMATH_GPT_total_shaded_area_l1729_172985

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  1 * S ^ 2 + 8 * (T ^ 2) = 13.5 := by
  sorry

end NUMINAMATH_GPT_total_shaded_area_l1729_172985


namespace NUMINAMATH_GPT_min_value_binom_l1729_172906

theorem min_value_binom
  (a b : ℕ → ℕ)
  (n : ℕ) (hn : 0 < n)
  (h1 : ∀ n, a n = 2^n)
  (h2 : ∀ n, b n = 4^n) :
  ∃ n, 2^n + (1 / 2^n) = 5 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_binom_l1729_172906


namespace NUMINAMATH_GPT_fraction_simplification_l1729_172953

theorem fraction_simplification : (8 : ℝ) / (4 * 25) = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1729_172953


namespace NUMINAMATH_GPT_solve_problem_l1729_172981

theorem solve_problem
    (x y z : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : x^2 + x * y + y^2 = 2)
    (h5 : y^2 + y * z + z^2 = 5)
    (h6 : z^2 + z * x + x^2 = 3) :
    x * y + y * z + z * x = 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1729_172981


namespace NUMINAMATH_GPT_least_positive_integer_x_l1729_172923

theorem least_positive_integer_x (x : ℕ) (h1 : x + 3721 ≡ 1547 [MOD 12]) (h2 : x % 2 = 0) : x = 2 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_x_l1729_172923


namespace NUMINAMATH_GPT_drink_price_half_promotion_l1729_172957

theorem drink_price_half_promotion (P : ℝ) (h : P + (1/2) * P = 13.5) : P = 9 := 
by
  sorry

end NUMINAMATH_GPT_drink_price_half_promotion_l1729_172957


namespace NUMINAMATH_GPT_min_value_expression_l1729_172913

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1729_172913


namespace NUMINAMATH_GPT_max_result_of_operation_l1729_172916

theorem max_result_of_operation : ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 → 3 * (300 - m) ≤ 870) ∧ 3 * (300 - n) = 870 :=
by
  sorry

end NUMINAMATH_GPT_max_result_of_operation_l1729_172916


namespace NUMINAMATH_GPT_find_c_l1729_172941

theorem find_c (c : ℝ) (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ 3 * x^2 + 12 * x - 27 = 0)
                      (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ 4 * x^2 - 12 * x + 5 = 0) :
                      c = -8.5 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1729_172941


namespace NUMINAMATH_GPT_certain_number_l1729_172956

theorem certain_number (x : ℝ) (h : (2.28 * x) / 6 = 480.7) : x = 1265.0 := 
by 
  sorry

end NUMINAMATH_GPT_certain_number_l1729_172956


namespace NUMINAMATH_GPT_nancy_target_amount_l1729_172907

theorem nancy_target_amount {rate : ℝ} {hours : ℝ} (h1 : rate = 28 / 4) (h2 : hours = 10) : 28 / 4 * 10 = 70 :=
by
  sorry

end NUMINAMATH_GPT_nancy_target_amount_l1729_172907


namespace NUMINAMATH_GPT_geometric_progression_theorem_l1729_172974

theorem geometric_progression_theorem 
  (a b c d : ℝ) (q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2) 
  (h3 : d = a * q^3) 
  : (a - d)^2 = (a - c)^2 + (b - c)^2 + (b - d)^2 := 
by sorry

end NUMINAMATH_GPT_geometric_progression_theorem_l1729_172974


namespace NUMINAMATH_GPT_roots_ratio_quadratic_l1729_172946

theorem roots_ratio_quadratic (p : ℤ) (h : (∃ x1 x2 : ℤ, x1*x2 = -16 ∧ x1 + x2 = -p ∧ x2 = -4 * x1)) :
  p = 6 ∨ p = -6 :=
sorry

end NUMINAMATH_GPT_roots_ratio_quadratic_l1729_172946


namespace NUMINAMATH_GPT_rhombus_diagonals_ratio_l1729_172986

theorem rhombus_diagonals_ratio (a b d1 d2 : ℝ) 
  (h1: a > 0) (h2: b > 0)
  (h3: d1 = 2 * (a / Real.cos θ))
  (h4: d2 = 2 * (b / Real.cos θ)) :
  d1 / d2 = a / b := 
sorry

end NUMINAMATH_GPT_rhombus_diagonals_ratio_l1729_172986


namespace NUMINAMATH_GPT_line_through_A_and_B_l1729_172954

variables (x y x₁ y₁ x₂ y₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * x₁ - 4 * y₁ - 2 = 0
def condition2 : Prop := 3 * x₂ - 4 * y₂ - 2 = 0

-- Proof that the line passing through A(x₁, y₁) and B(x₂, y₂) is 3x - 4y - 2 = 0
theorem line_through_A_and_B (h1 : condition1 x₁ y₁) (h2 : condition2 x₂ y₂) :
    ∀ (x y : ℝ), (∃ k : ℝ, x = x₁ + k * (x₂ - x₁) ∧ y = y₁ + k * (y₂ - y₁)) → 3 * x - 4 * y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_line_through_A_and_B_l1729_172954


namespace NUMINAMATH_GPT_Claire_takes_6_photos_l1729_172961

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end NUMINAMATH_GPT_Claire_takes_6_photos_l1729_172961


namespace NUMINAMATH_GPT_binom_21_13_l1729_172965

theorem binom_21_13 : (Nat.choose 21 13) = 203490 :=
by
  have h1 : (Nat.choose 20 13) = 77520 := by sorry
  have h2 : (Nat.choose 20 12) = 125970 := by sorry
  have pascal : (Nat.choose 21 13) = (Nat.choose 20 13) + (Nat.choose 20 12) :=
    by rw [Nat.choose_succ_succ, h1, h2]
  exact pascal

end NUMINAMATH_GPT_binom_21_13_l1729_172965


namespace NUMINAMATH_GPT_sample_variance_is_two_l1729_172947

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : (1 / 5) * ((-1 - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sample_variance_is_two_l1729_172947


namespace NUMINAMATH_GPT_brian_shoes_l1729_172951

theorem brian_shoes (J E B : ℕ) (h1 : J = E / 2) (h2 : E = 3 * B) (h3 : J + E + B = 121) : B = 22 :=
sorry

end NUMINAMATH_GPT_brian_shoes_l1729_172951


namespace NUMINAMATH_GPT_pool_water_volume_after_evaporation_l1729_172968

theorem pool_water_volume_after_evaporation :
  let initial_volume := 300
  let evaporation_first_15_days := 1 -- in gallons per day
  let evaporation_next_15_days := 2 -- in gallons per day
  initial_volume - (15 * evaporation_first_15_days + 15 * evaporation_next_15_days) = 255 :=
by
  sorry

end NUMINAMATH_GPT_pool_water_volume_after_evaporation_l1729_172968


namespace NUMINAMATH_GPT_bottles_per_case_correct_l1729_172903

-- Define the conditions given in the problem
def daily_bottle_production : ℕ := 120000
def number_of_cases_needed : ℕ := 10000

-- Define the expected answer
def bottles_per_case : ℕ := 12

-- The statement we need to prove
theorem bottles_per_case_correct :
  daily_bottle_production / number_of_cases_needed = bottles_per_case :=
by
  -- Leap of logic: actually solving this for correctness is here considered a leap
  sorry

end NUMINAMATH_GPT_bottles_per_case_correct_l1729_172903


namespace NUMINAMATH_GPT_total_items_in_jar_l1729_172939

/--
A jar contains 3409.0 pieces of candy and 145.0 secret eggs with a prize.
We aim to prove that the total number of items in the jar is 3554.0.
-/
theorem total_items_in_jar :
  let number_of_pieces_of_candy := 3409.0
  let number_of_secret_eggs := 145.0
  number_of_pieces_of_candy + number_of_secret_eggs = 3554.0 :=
by
  sorry

end NUMINAMATH_GPT_total_items_in_jar_l1729_172939


namespace NUMINAMATH_GPT_jessica_found_seashells_l1729_172984

-- Define the given conditions
def mary_seashells : ℕ := 18
def total_seashells : ℕ := 59

-- Define the goal for the number of seashells Jessica found
def jessica_seashells (mary_seashells total_seashells : ℕ) : ℕ := total_seashells - mary_seashells

-- The theorem stating Jessica found 41 seashells
theorem jessica_found_seashells : jessica_seashells mary_seashells total_seashells = 41 := by
  -- We assume the conditions and skip the proof
  sorry

end NUMINAMATH_GPT_jessica_found_seashells_l1729_172984


namespace NUMINAMATH_GPT_five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l1729_172971

variable (p q : ℕ)
variable (hp : p % 2 = 1)  -- p is odd
variable (hq : q % 2 = 1)  -- q is odd

theorem five_p_squared_plus_two_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (5 * p^2 + 2 * q^2) % 2 = 1 := 
sorry

theorem p_squared_plus_pq_plus_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (p^2 + p * q + q^2) % 2 = 1 := 
sorry

end NUMINAMATH_GPT_five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l1729_172971


namespace NUMINAMATH_GPT_largest_x_not_defined_l1729_172964

theorem largest_x_not_defined : 
  (∀ x, (6 * x ^ 2 - 17 * x + 5 = 0) → x ≤ 2.5) ∧
  (∃ x, (6 * x ^ 2 - 17 * x + 5 = 0) ∧ x = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_largest_x_not_defined_l1729_172964


namespace NUMINAMATH_GPT_bookstore_discount_l1729_172978

theorem bookstore_discount (P MP price_paid : ℝ) (h1 : MP = 0.80 * P) (h2 : price_paid = 0.60 * MP) :
  price_paid / P = 0.48 :=
by
  sorry

end NUMINAMATH_GPT_bookstore_discount_l1729_172978


namespace NUMINAMATH_GPT_total_marbles_l1729_172991

namespace MarbleBag

def numBlue : ℕ := 5
def numRed : ℕ := 9
def probRedOrWhite : ℚ := 5 / 6

theorem total_marbles (total_mar : ℕ) (numWhite : ℕ) (h1 : probRedOrWhite = (numRed + numWhite) / total_mar)
                      (h2 : total_mar = numBlue + numRed + numWhite) :
  total_mar = 30 :=
by
  sorry

end MarbleBag

end NUMINAMATH_GPT_total_marbles_l1729_172991


namespace NUMINAMATH_GPT_abc_area_l1729_172917

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

theorem abc_area :
  let smaller_side := 7
  let longer_side := 2 * smaller_side
  let length := 3 * longer_side -- since there are 3 identical rectangles placed side by side
  let width := smaller_side
  rectangle_area length width = 294 :=
by
  sorry

end NUMINAMATH_GPT_abc_area_l1729_172917


namespace NUMINAMATH_GPT_tenth_pair_in_twentieth_row_l1729_172914

def nth_pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if h : n > 0 ∧ k > 0 ∧ n >= k then (k, n + 1 - k)
  else (0, 0) -- define (0,0) as a default for invalid inputs

theorem tenth_pair_in_twentieth_row : nth_pair_in_row 20 10 = (10, 11) :=
by sorry

end NUMINAMATH_GPT_tenth_pair_in_twentieth_row_l1729_172914


namespace NUMINAMATH_GPT_square_root_area_ratio_l1729_172988

theorem square_root_area_ratio 
  (side_C : ℝ) (side_D : ℝ)
  (hC : side_C = 45) 
  (hD : side_D = 60) : 
  Real.sqrt ((side_C^2) / (side_D^2)) = 3 / 4 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_square_root_area_ratio_l1729_172988


namespace NUMINAMATH_GPT_line_equation_l1729_172973

theorem line_equation 
    (passes_through_intersection : ∃ (P : ℝ × ℝ), P ∈ { (x, y) | 11 * x + 3 * y - 7 = 0 } ∧ P ∈ { (x, y) | 12 * x + y - 19 = 0 })
    (equidistant_from_A_and_B : ∃ (P : ℝ × ℝ), dist P (3, -2) = dist P (-1, 6)) :
    ∃ (a b c : ℝ), (a = 7 ∧ b = 1 ∧ c = -9) ∨ (a = 2 ∧ b = 1 ∧ c = 1) ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
sorry

end NUMINAMATH_GPT_line_equation_l1729_172973


namespace NUMINAMATH_GPT_closest_square_to_350_l1729_172929

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end NUMINAMATH_GPT_closest_square_to_350_l1729_172929


namespace NUMINAMATH_GPT_number_of_pupils_l1729_172950

-- Define the number of total people
def total_people : ℕ := 803

-- Define the number of parents
def parents : ℕ := 105

-- We need to prove the number of pupils is 698
theorem number_of_pupils : (total_people - parents) = 698 := 
by
  -- Skip the proof steps
  sorry

end NUMINAMATH_GPT_number_of_pupils_l1729_172950


namespace NUMINAMATH_GPT_solution_pairs_l1729_172959

theorem solution_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x ^ 2 + y ^ 2 - 5 * x * y + 5 = 0 ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2) := by
  sorry

end NUMINAMATH_GPT_solution_pairs_l1729_172959


namespace NUMINAMATH_GPT_intersection_point_on_square_diagonal_l1729_172912

theorem intersection_point_on_square_diagonal (a b c : ℝ) (h : c = (a + b) / 2) :
  (b / 2) = (-a / 2) + c :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_on_square_diagonal_l1729_172912


namespace NUMINAMATH_GPT_units_digit_of_product_of_odds_between_10_and_50_l1729_172987

def product_of_odds_units_digit : ℕ :=
  let odds := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
  let product := odds.foldl (· * ·) 1
  product % 10

theorem units_digit_of_product_of_odds_between_10_and_50 : product_of_odds_units_digit = 5 :=
  sorry

end NUMINAMATH_GPT_units_digit_of_product_of_odds_between_10_and_50_l1729_172987


namespace NUMINAMATH_GPT_find_original_selling_price_l1729_172938

variable (x : ℝ) (discount_rate : ℝ) (final_price : ℝ)

def original_selling_price_exists (x : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  (x * (1 - discount_rate) = final_price) → (x = 700)

theorem find_original_selling_price
  (discount_rate : ℝ := 0.20)
  (final_price : ℝ := 560) :
  ∃ x : ℝ, original_selling_price_exists x discount_rate final_price :=
by
  use 700
  sorry

end NUMINAMATH_GPT_find_original_selling_price_l1729_172938


namespace NUMINAMATH_GPT_total_area_needed_l1729_172969

-- Definitions based on conditions
def oak_trees_first_half := 100
def pine_trees_first_half := 100
def oak_trees_second_half := 150
def pine_trees_second_half := 150
def oak_tree_planting_ratio := 4
def pine_tree_planting_ratio := 2
def oak_tree_space := 4
def pine_tree_space := 2

-- Total area needed for tree planting during the entire year
theorem total_area_needed : (oak_trees_first_half * oak_tree_planting_ratio * oak_tree_space) + ((pine_trees_first_half + pine_trees_second_half) * pine_tree_planting_ratio * pine_tree_space) = 2600 :=
by
  sorry

end NUMINAMATH_GPT_total_area_needed_l1729_172969


namespace NUMINAMATH_GPT_beta_still_water_speed_l1729_172905

-- Definitions that are used in the conditions
def alpha_speed_still_water : ℝ := 56 
def beta_speed_still_water : ℝ := 52  
def water_current_speed : ℝ := 4

-- The main theorem statement 
theorem beta_still_water_speed : β_speed_still_water = 61 := 
  sorry -- the proof goes here

end NUMINAMATH_GPT_beta_still_water_speed_l1729_172905


namespace NUMINAMATH_GPT_angle_sum_l1729_172918

theorem angle_sum (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 3 / 4)
  (sin_β : Real.sin β = 3 / 5) :
  α + 3 * β = 5 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_angle_sum_l1729_172918


namespace NUMINAMATH_GPT_perpendicular_vectors_l1729_172952

def vector (α : Type) := (α × α)
def dot_product {α : Type} [Add α] [Mul α] (a b : vector α) : α :=
  a.1 * b.1 + a.2 * b.2

theorem perpendicular_vectors
    (a : vector ℝ) (b : vector ℝ)
    (h : dot_product a b = 0)
    (ha : a = (2, 4))
    (hb : b = (-1, n)) : 
    n = 1 / 2 := 
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l1729_172952


namespace NUMINAMATH_GPT_g_neg_one_l1729_172996

variables {F : Type*} [Field F]

def odd_function (f : F → F) := ∀ x, f (-x) = -f x

variables (f : F → F) (g : F → F)

-- Given conditions
lemma given_conditions :
  (∀ x, f (-x) + (-x)^2 = -(f x + x^2)) ∧
  f 1 = 1 ∧
  (∀ x, g x = f x + 2) :=
sorry

-- Prove that g(-1) = -1
theorem g_neg_one :
  g (-1) = -1 :=
sorry

end NUMINAMATH_GPT_g_neg_one_l1729_172996


namespace NUMINAMATH_GPT_jerry_initial_candy_l1729_172926

theorem jerry_initial_candy
  (total_bags : ℕ)
  (chocolate_hearts_bags : ℕ)
  (chocolate_kisses_bags : ℕ)
  (nonchocolate_pieces : ℕ)
  (remaining_bags : ℕ)
  (pieces_per_bag : ℕ)
  (initial_candy : ℕ)
  (h_total_bags : total_bags = 9)
  (h_chocolate_hearts_bags : chocolate_hearts_bags = 2)
  (h_chocolate_kisses_bags : chocolate_kisses_bags = 3)
  (h_nonchocolate_pieces : nonchocolate_pieces = 28)
  (h_remaining_bags : remaining_bags = total_bags - chocolate_hearts_bags - chocolate_kisses_bags)
  (h_pieces_per_bag : pieces_per_bag = nonchocolate_pieces / remaining_bags)
  (h_initial_candy : initial_candy = total_bags * pieces_per_bag) :
  initial_candy = 63 := by
  sorry

end NUMINAMATH_GPT_jerry_initial_candy_l1729_172926


namespace NUMINAMATH_GPT_triangle_construction_condition_l1729_172998

variable (varrho_a varrho_b m_c : ℝ)

theorem triangle_construction_condition :
  (∃ (triangle : Type) (ABC : triangle)
    (r_a : triangle → ℝ)
    (r_b : triangle → ℝ)
    (h_from_C : triangle → ℝ),
      r_a ABC = varrho_a ∧
      r_b ABC = varrho_b ∧
      h_from_C ABC = m_c)
  ↔ 
  (1 / m_c = 1 / 2 * (1 / varrho_a + 1 / varrho_b)) :=
sorry

end NUMINAMATH_GPT_triangle_construction_condition_l1729_172998


namespace NUMINAMATH_GPT_line_MN_parallel_to_y_axis_l1729_172962

-- Definition of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector between two points
def vector_between (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y,
    z := Q.z - P.z }

-- Given points M and N
def M : Point := { x := 3, y := -2, z := 1 }
def N : Point := { x := 3, y := 2, z := 1 }

-- The vector \overrightarrow{MN}
def vec_MN : Point := vector_between M N

-- Theorem: The vector between points M and N is parallel to the y-axis
theorem line_MN_parallel_to_y_axis : vec_MN = {x := 0, y := 4, z := 0} := by
  sorry

end NUMINAMATH_GPT_line_MN_parallel_to_y_axis_l1729_172962


namespace NUMINAMATH_GPT_inscribed_circle_radius_A_B_D_l1729_172990

theorem inscribed_circle_radius_A_B_D (AB CD: ℝ) (angleA acuteAngleD: Prop)
  (M N: Type) (MN: ℝ) (area_trapezoid: ℝ)
  (radius: ℝ) : 
  AB = 2 ∧ CD = 3 ∧ angleA ∧ acuteAngleD ∧ MN = 4 ∧ area_trapezoid = (26 * Real.sqrt 2) / 3 
  → radius = (16 * Real.sqrt 2) / (15 + Real.sqrt 129) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_A_B_D_l1729_172990


namespace NUMINAMATH_GPT_find_a_l1729_172997

theorem find_a (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 - x + a^2 - 2*a - 2 = 0 ∧ x = 1) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1729_172997


namespace NUMINAMATH_GPT_common_divisors_count_l1729_172908

-- Given conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8000

-- Prime factorizations from conditions
def fact_num1 : List ℕ := [2^3, 3^1, 5^1, 7^2]
def fact_num2 : List ℕ := [2^6, 5^3]

-- Computing gcd based on factorizations
def gcd : ℕ := 2^3 * 5^1

-- The goal is to prove the number of divisors of gcd is 8
theorem common_divisors_count : 
  ∃ d, d = (3+1)*(1+1) ∧ d = 8 := 
by
  sorry

end NUMINAMATH_GPT_common_divisors_count_l1729_172908


namespace NUMINAMATH_GPT_difference_pencils_l1729_172932

theorem difference_pencils (x : ℕ) (h1 : 162 = x * n_g) (h2 : 216 = x * n_f) : n_f - n_g = 3 :=
by
  sorry

end NUMINAMATH_GPT_difference_pencils_l1729_172932


namespace NUMINAMATH_GPT_fraction_add_eq_l1729_172972

theorem fraction_add_eq (n : ℤ) :
  (3 + n) = 4 * ((4 + n) - 5) → n = 1 := sorry

end NUMINAMATH_GPT_fraction_add_eq_l1729_172972


namespace NUMINAMATH_GPT_find_dividend_l1729_172949

theorem find_dividend
  (R : ℕ)
  (Q : ℕ)
  (D : ℕ)
  (hR : R = 6)
  (hD_eq_5Q : D = 5 * Q)
  (hD_eq_3R_plus_2 : D = 3 * R + 2) :
  D * Q + R = 86 :=
by
  sorry

end NUMINAMATH_GPT_find_dividend_l1729_172949


namespace NUMINAMATH_GPT_solve_problem_l1729_172982
open Complex

noncomputable def problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  (a ≠ -1) ∧ (b ≠ -1) ∧ (c ≠ -1) ∧ (d ≠ -1) ∧ (ω ^ 4 = 1) ∧ (ω ≠ 1) ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω ^ 2)
  
theorem solve_problem {a b c d : ℝ} {ω : ℂ} (h : problem a b c d ω) : 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2) :=
sorry

end NUMINAMATH_GPT_solve_problem_l1729_172982


namespace NUMINAMATH_GPT_range_of_a_l1729_172958

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → (a ≤ 1 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1729_172958


namespace NUMINAMATH_GPT_grandfather_age_l1729_172924

variables (M G y z : ℕ)

-- Conditions
def condition1 : Prop := G = 6 * M
def condition2 : Prop := G + y = 5 * (M + y)
def condition3 : Prop := G + y + z = 4 * (M + y + z)

-- Theorem to prove Grandfather's current age is 72
theorem grandfather_age : 
  condition1 M G → 
  condition2 M G y → 
  condition3 M G y z → 
  G = 72 :=
by
  intros h1 h2 h3
  unfold condition1 at h1
  unfold condition2 at h2
  unfold condition3 at h3
  sorry

end NUMINAMATH_GPT_grandfather_age_l1729_172924


namespace NUMINAMATH_GPT_commute_solution_l1729_172977

noncomputable def commute_problem : Prop :=
  let t : ℝ := 1                -- 1 hour from 7:00 AM to 8:00 AM
  let late_minutes : ℝ := 5 / 60  -- 5 minutes = 5/60 hours
  let early_minutes : ℝ := 4 / 60 -- 4 minutes = 4/60 hours
  let speed1 : ℝ := 30          -- 30 mph
  let speed2 : ℝ := 70          -- 70 mph
  let d1 : ℝ := speed1 * (t + late_minutes)
  let d2 : ℝ := speed2 * (t - early_minutes)

  ∃ (speed : ℝ), d1 = d2 ∧ speed = d1 / t ∧ speed = 32.5

theorem commute_solution : commute_problem :=
by sorry

end NUMINAMATH_GPT_commute_solution_l1729_172977


namespace NUMINAMATH_GPT_solve_system_l1729_172992

theorem solve_system :
  ∃ (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1729_172992


namespace NUMINAMATH_GPT_ticket_price_increase_one_day_later_l1729_172980

noncomputable def ticket_price : ℝ := 1050
noncomputable def days_before_departure : ℕ := 14
noncomputable def daily_increase_rate : ℝ := 0.05

theorem ticket_price_increase_one_day_later :
  ∀ (price : ℝ) (days : ℕ) (rate : ℝ), price = ticket_price → days = days_before_departure → rate = daily_increase_rate →
  price * rate = 52.50 :=
by
  intros price days rate hprice hdays hrate
  rw [hprice, hrate]
  exact sorry

end NUMINAMATH_GPT_ticket_price_increase_one_day_later_l1729_172980


namespace NUMINAMATH_GPT_steven_peaches_l1729_172942

theorem steven_peaches (jake_peaches : ℕ) (steven_peaches : ℕ) (h1 : jake_peaches = 3) (h2 : jake_peaches + 10 = steven_peaches) : steven_peaches = 13 :=
by
  sorry

end NUMINAMATH_GPT_steven_peaches_l1729_172942


namespace NUMINAMATH_GPT_find_z_value_l1729_172902

theorem find_z_value (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1/w = 15) (h2 : w^2 + 1/z = 3) : z = 44/3 := 
by 
  sorry

end NUMINAMATH_GPT_find_z_value_l1729_172902


namespace NUMINAMATH_GPT_tenth_term_geometric_sequence_l1729_172933

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5
  let r : ℚ := 3 / 4
  let a_n (n : ℕ) : ℚ := a * r ^ (n - 1)
  a_n 10 = 98415 / 262144 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_geometric_sequence_l1729_172933


namespace NUMINAMATH_GPT_focus_of_parabola_l1729_172911

def parabola (x : ℝ) : ℝ := (x - 3) ^ 2

theorem focus_of_parabola :
  ∃ f : ℝ × ℝ, f = (3, 1 / 4) ∧
  ∀ x : ℝ, parabola x = (x - 3)^2 :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_l1729_172911


namespace NUMINAMATH_GPT_unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l1729_172983

theorem unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2 : ∃! (x : ℤ), x - 9 / (x - 2) = 5 - 9 / (x - 2) := 
by
  sorry

end NUMINAMATH_GPT_unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l1729_172983


namespace NUMINAMATH_GPT_solution_of_inequality_l1729_172970

theorem solution_of_inequality (a : ℝ) :
  (a = 0 → ∀ x : ℝ, ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1) ∧
  (a < 0 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1 ∨ x < 1/a)) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (a > 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ∀ x : ℝ, ¬(ax^2 - (a + 1) * x + 1 < 0)) :=
by
  sorry

end NUMINAMATH_GPT_solution_of_inequality_l1729_172970


namespace NUMINAMATH_GPT_apples_used_l1729_172909

theorem apples_used (x : ℕ) 
  (initial_apples : ℕ := 23) 
  (bought_apples : ℕ := 6) 
  (final_apples : ℕ := 9) 
  (h : (initial_apples - x) + bought_apples = final_apples) : 
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_apples_used_l1729_172909


namespace NUMINAMATH_GPT_valves_fill_pool_l1729_172975

theorem valves_fill_pool
  (a b c d : ℝ)
  (h1 : 1 / a + 1 / b + 1 / c = 1 / 12)
  (h2 : 1 / b + 1 / c + 1 / d = 1 / 15)
  (h3 : 1 / a + 1 / d = 1 / 20) :
  1 / a + 1 / b + 1 / c + 1 / d = 1 / 10 := 
sorry

end NUMINAMATH_GPT_valves_fill_pool_l1729_172975


namespace NUMINAMATH_GPT_sum_x_coordinates_common_points_l1729_172979

theorem sum_x_coordinates_common_points (x y : ℤ) (h1 : y ≡ 3 * x + 5 [ZMOD 13]) (h2 : y ≡ 9 * x + 1 [ZMOD 13]) : x ≡ 5 [ZMOD 13] :=
sorry

end NUMINAMATH_GPT_sum_x_coordinates_common_points_l1729_172979


namespace NUMINAMATH_GPT_distance_formula_proof_l1729_172900

open Real

noncomputable def distance_between_points_on_curve
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  ℝ :=
  |c - a| * sqrt (1 + m^2 * (c + a)^2)

theorem distance_formula_proof
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  distance_between_points_on_curve a b c d m k h1 h2 = |c - a| * sqrt (1 + m^2 * (c + a)^2) :=
by
  sorry

end NUMINAMATH_GPT_distance_formula_proof_l1729_172900


namespace NUMINAMATH_GPT_lines_intersect_and_sum_l1729_172937

theorem lines_intersect_and_sum (a b : ℝ) :
  (∃ x y : ℝ, x = (1 / 3) * y + a ∧ y = (1 / 3) * x + b ∧ x = 3 ∧ y = 3) →
  a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_and_sum_l1729_172937


namespace NUMINAMATH_GPT_evaluate_expression_l1729_172904

theorem evaluate_expression : (900^2 / (153^2 - 147^2)) = 450 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1729_172904


namespace NUMINAMATH_GPT_shoe_price_calculation_l1729_172919

theorem shoe_price_calculation :
  let initialPrice : ℕ := 50
  let increasedPrice : ℕ := 60  -- initialPrice * 1.2
  let discountAmount : ℕ := 9    -- increasedPrice * 0.15
  increasedPrice - discountAmount = 51 := 
by
  sorry

end NUMINAMATH_GPT_shoe_price_calculation_l1729_172919


namespace NUMINAMATH_GPT_min_segments_to_erase_l1729_172989

noncomputable def nodes (m n : ℕ) : ℕ := (m - 2) * (n - 2)

noncomputable def segments_to_erase (m n : ℕ) : ℕ := (nodes m n + 1) / 2

theorem min_segments_to_erase (m n : ℕ) (hm : m = 11) (hn : n = 11) :
  segments_to_erase m n = 41 := by
  sorry

end NUMINAMATH_GPT_min_segments_to_erase_l1729_172989


namespace NUMINAMATH_GPT_metallic_sheet_dimension_l1729_172931

theorem metallic_sheet_dimension
  (length_cut : ℕ) (other_dim : ℕ) (volume : ℕ) (x : ℕ)
  (length_cut_eq : length_cut = 8)
  (other_dim_eq : other_dim = 36)
  (volume_eq : volume = 4800)
  (volume_formula : volume = (x - 2 * length_cut) * (other_dim - 2 * length_cut) * length_cut) :
  x = 46 :=
by
  sorry

end NUMINAMATH_GPT_metallic_sheet_dimension_l1729_172931


namespace NUMINAMATH_GPT_Noemi_blackjack_loss_l1729_172940

-- Define the conditions
def start_amount : ℕ := 1700
def end_amount : ℕ := 800
def roulette_loss : ℕ := 400

-- Define the total loss calculation
def total_loss : ℕ := start_amount - end_amount

-- Main theorem statement
theorem Noemi_blackjack_loss :
  ∃ (blackjack_loss : ℕ), blackjack_loss = total_loss - roulette_loss := 
by
  -- Start by calculating the total_loss
  let total_loss_eq := start_amount - end_amount
  -- The blackjack loss should be 900 - 400, which we claim to be 500
  use total_loss_eq - roulette_loss
  sorry

end NUMINAMATH_GPT_Noemi_blackjack_loss_l1729_172940


namespace NUMINAMATH_GPT_quadratic_polynomial_half_coefficient_l1729_172999

theorem quadratic_polynomial_half_coefficient :
  ∃ b c : ℚ, ∀ x : ℤ, ∃ k : ℤ, (1/2 : ℚ) * (x^2 : ℚ) + b * (x : ℚ) + c = (k : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_half_coefficient_l1729_172999


namespace NUMINAMATH_GPT_min_checkout_counters_l1729_172967

variable (n : ℕ)
variable (x y : ℝ)

-- Conditions based on problem statement
axiom cond1 : 40 * y = 20 * x + n
axiom cond2 : 36 * y = 12 * x + n

theorem min_checkout_counters (m : ℕ) (h : 6 * m * y > 6 * x + n) : m ≥ 6 :=
  sorry

end NUMINAMATH_GPT_min_checkout_counters_l1729_172967


namespace NUMINAMATH_GPT_water_charge_rel_water_usage_from_charge_l1729_172927

-- Define the conditions and functional relationship
theorem water_charge_rel (x : ℝ) (hx : x > 5) : y = 3.5 * x - 7.5 :=
  sorry

-- Prove the specific case where the charge y is 17 yuan
theorem water_usage_from_charge (h : 17 = 3.5 * x - 7.5) :
  x = 7 :=
  sorry

end NUMINAMATH_GPT_water_charge_rel_water_usage_from_charge_l1729_172927


namespace NUMINAMATH_GPT_fermat_little_theorem_l1729_172994

theorem fermat_little_theorem (N p : ℕ) (hp : Nat.Prime p) (hNp : ¬ p ∣ N) : p ∣ (N ^ (p - 1) - 1) := 
sorry

end NUMINAMATH_GPT_fermat_little_theorem_l1729_172994


namespace NUMINAMATH_GPT_range_of_a_l1729_172925

noncomputable def setA : Set ℝ := {x | 3 + 2 * x - x^2 >= 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) : (setA ∩ setB a).Nonempty → a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1729_172925


namespace NUMINAMATH_GPT_smallest_6_digit_div_by_111_l1729_172920

theorem smallest_6_digit_div_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 := by
  sorry

end NUMINAMATH_GPT_smallest_6_digit_div_by_111_l1729_172920


namespace NUMINAMATH_GPT_find_center_and_tangent_slope_l1729_172921

theorem find_center_and_tangent_slope :
  let C := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 8 = 0 }
  let center := (3, 0)
  let k := - (Real.sqrt 2 / 4)
  (∃ c ∈ C, c = center) ∧
  (∃ q ∈ C, q.2 < 0 ∧ q.2 = k * q.1 ∧
             |3 * k| / Real.sqrt (k ^ 2 + 1) = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_center_and_tangent_slope_l1729_172921


namespace NUMINAMATH_GPT_minimize_sum_m_n_l1729_172930

-- Definitions of the given conditions
def last_three_digits_equal (a b : ℕ) : Prop :=
  (a % 1000) = (b % 1000)

-- The main statement to prove
theorem minimize_sum_m_n (m n : ℕ) (h1 : n > m) (h2 : 1 ≤ m) 
  (h3 : last_three_digits_equal (1978^n) (1978^m)) : m + n = 106 :=
sorry

end NUMINAMATH_GPT_minimize_sum_m_n_l1729_172930


namespace NUMINAMATH_GPT_molecular_weight_acetic_acid_l1729_172934

-- Define atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of each atom in acetic acid
def num_C : ℕ := 2
def num_H : ℕ := 4
def num_O : ℕ := 2

-- Define the molecular formula of acetic acid
def molecular_weight_CH3COOH : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_O * atomic_weight_O

-- State the proposition
theorem molecular_weight_acetic_acid :
  molecular_weight_CH3COOH = 60.052 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_acetic_acid_l1729_172934


namespace NUMINAMATH_GPT_algae_free_day_22_l1729_172935

def algae_coverage (day : ℕ) : ℝ :=
if day = 25 then 1 else 2 ^ (25 - day)

theorem algae_free_day_22 :
  1 - algae_coverage 22 = 0.875 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_algae_free_day_22_l1729_172935


namespace NUMINAMATH_GPT_sum_diff_9114_l1729_172966

def sum_odd_ints (n : ℕ) := (n + 1) / 2 * (1 + n)
def sum_even_ints (n : ℕ) := n / 2 * (2 + n)

theorem sum_diff_9114 : 
  let m := sum_odd_ints 215
  let t := sum_even_ints 100
  m - t = 9114 :=
by
  sorry

end NUMINAMATH_GPT_sum_diff_9114_l1729_172966


namespace NUMINAMATH_GPT_expression_defined_if_x_not_3_l1729_172928

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_expression_defined_if_x_not_3_l1729_172928


namespace NUMINAMATH_GPT_rhombus_diagonals_l1729_172963

theorem rhombus_diagonals (x y : ℝ) 
  (h1 : x * y = 234)
  (h2 : x + y = 31) :
  (x = 18 ∧ y = 13) ∨ (x = 13 ∧ y = 18) := by
sorry

end NUMINAMATH_GPT_rhombus_diagonals_l1729_172963


namespace NUMINAMATH_GPT_consecutive_integers_sum_l1729_172901

theorem consecutive_integers_sum (a b : ℤ) (sqrt_33 : ℝ) (h1 : a < sqrt_33) (h2 : sqrt_33 < b) (h3 : b = a + 1) (h4 : sqrt_33 = Real.sqrt 33) : a + b = 11 :=
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l1729_172901
