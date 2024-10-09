import Mathlib

namespace range_of_a_l852_85245

theorem range_of_a (a : ℚ) (h_pos : 0 < a) (h_int_count : ∀ n : ℕ, 2 * n + 1 = 2007 -> ∃ k : ℤ, -a < ↑k ∧ ↑k < a) : 1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l852_85245


namespace markup_rate_correct_l852_85206

noncomputable def selling_price : ℝ := 10.00
noncomputable def profit_percentage : ℝ := 0.20
noncomputable def expenses_percentage : ℝ := 0.15
noncomputable def cost (S : ℝ) : ℝ := S - (profit_percentage * S + expenses_percentage * S)
noncomputable def markup_rate (S C : ℝ) : ℝ := (S - C) / C * 100

theorem markup_rate_correct :
  markup_rate selling_price (cost selling_price) = 53.85 := 
by
  sorry

end markup_rate_correct_l852_85206


namespace part1_part2_l852_85297

theorem part1 (a : ℝ) (h : 48 * a^2 = 75) (ha : a > 0) : a = 5 / 4 :=
sorry

theorem part2 (θ : ℝ) 
  (h₁ : 10 * (Real.sin θ) ^ 2 = 5) 
  (h₀ : 0 < θ ∧ θ < Real.pi / 2) 
  : θ = Real.pi / 4 :=
sorry

end part1_part2_l852_85297


namespace count_triangles_in_hexagonal_grid_l852_85268

-- Define the number of smallest triangles in the figure.
def small_triangles : ℕ := 10

-- Define the number of medium triangles in the figure, composed of 4 small triangles each.
def medium_triangles : ℕ := 6

-- Define the number of large triangles in the figure, composed of 9 small triangles each.
def large_triangles : ℕ := 3

-- Define the number of extra-large triangle composed of 16 small triangles.
def extra_large_triangle : ℕ := 1

-- Define the total number of triangles in the figure.
def total_triangles : ℕ := small_triangles + medium_triangles + large_triangles + extra_large_triangle

-- The theorem we want to prove: the total number of triangles is 20.
theorem count_triangles_in_hexagonal_grid : total_triangles = 20 := by
  -- Placeholder for the proof.
  sorry

end count_triangles_in_hexagonal_grid_l852_85268


namespace intersection_result_l852_85258

open Set

namespace ProofProblem

def A : Set ℝ := {x | |x| ≤ 4}
def B : Set ℝ := {x | 4 ≤ x ∧ x < 5}

theorem intersection_result : A ∩ B = {4} :=
  sorry

end ProofProblem

end intersection_result_l852_85258


namespace average_age_of_troupe_l852_85256

theorem average_age_of_troupe
  (number_females : ℕ) (number_males : ℕ) 
  (average_age_females : ℕ) (average_age_males : ℕ)
  (total_people : ℕ) (total_age : ℕ)
  (h1 : number_females = 12) 
  (h2 : number_males = 18) 
  (h3 : average_age_females = 25) 
  (h4 : average_age_males = 30)
  (h5 : total_people = 30)
  (h6 : total_age = (25 * 12 + 30 * 18)) :
  total_age / total_people = 28 :=
by
  -- Proof goes here
  sorry

end average_age_of_troupe_l852_85256


namespace kelly_carrot_weight_l852_85203

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l852_85203


namespace number_of_yellow_marbles_l852_85212

theorem number_of_yellow_marbles (total_marbles blue_marbles red_marbles green_marbles yellow_marbles : ℕ)
    (h_total : total_marbles = 164) 
    (h_blue : blue_marbles = total_marbles / 2)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27) :
    yellow_marbles = total_marbles - (blue_marbles + red_marbles + green_marbles) →
    yellow_marbles = 14 := by
  sorry

end number_of_yellow_marbles_l852_85212


namespace time_to_fill_tank_with_leak_l852_85281

theorem time_to_fill_tank_with_leak (A L : ℚ) (hA : A = 1/6) (hL : L = 1/24) :
  (1 / (A - L)) = 8 := 
by 
  sorry

end time_to_fill_tank_with_leak_l852_85281


namespace possible_sets_l852_85202

theorem possible_sets 
  (A B C : Set ℕ) 
  (U : Set ℕ := {a, b, c, d, e, f}) 
  (H1 : A ∪ B ∪ C = U) 
  (H2 : A ∩ B = {a, b, c, d}) 
  (H3 : c ∈ A ∩ B ∩ C) : 
  ∃ (n : ℕ), n = 200 :=
sorry

end possible_sets_l852_85202


namespace min_value_sin_cos_l852_85216

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l852_85216


namespace shaded_area_calculation_l852_85228

-- Define the grid and the side length conditions
def grid_size : ℕ := 5 * 4
def side_length : ℕ := 1
def total_squares : ℕ := 5 * 4

-- Define the area of one small square
def area_of_square (side: ℕ) : ℕ := side * side

-- Define the shaded region in terms of number of small squares fully or partially occupied
def shaded_squares : ℕ := 11

-- By analyzing the grid based on given conditions, prove that the area of the shaded region is 11
theorem shaded_area_calculation : (shaded_squares * side_length * side_length) = 11 := sorry

end shaded_area_calculation_l852_85228


namespace min_value_of_m_l852_85232

theorem min_value_of_m : (2 ∈ {x | ∃ (m : ℤ), x * (x - m) < 0}) → ∃ (m : ℤ), m = 3 :=
by
  sorry

end min_value_of_m_l852_85232


namespace find_m_l852_85287

theorem find_m (m x : ℝ) (h : (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) (hx : x = 0) : m = -2 :=
by sorry

end find_m_l852_85287


namespace intersection_of_A_and_B_l852_85219

-- Definitions of sets A and B
def set_A : Set ℝ := { x | x^2 - x - 6 < 0 }
def set_B : Set ℝ := { x | (x + 4) * (x - 2) > 0 }

-- Theorem statement for the intersection of A and B
theorem intersection_of_A_and_B : set_A ∩ set_B = { x | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_of_A_and_B_l852_85219


namespace sin_double_angle_l852_85215

theorem sin_double_angle (h1 : Real.pi / 2 < β)
    (h2 : β < α)
    (h3 : α < 3 * Real.pi / 4)
    (h4 : Real.cos (α - β) = 12 / 13)
    (h5 : Real.sin (α + β) = -3 / 5) :
    Real.sin (2 * α) = -56 / 65 := 
by
  sorry

end sin_double_angle_l852_85215


namespace arithmetic_geometric_mean_inequality_l852_85229

variable {a b : ℝ}

noncomputable def A (a b : ℝ) := (a + b) / 2
noncomputable def B (a b : ℝ) := Real.sqrt (a * b)

theorem arithmetic_geometric_mean_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ b) : A a b > B a b := 
by
  sorry

end arithmetic_geometric_mean_inequality_l852_85229


namespace quadratic_has_two_real_distinct_roots_and_find_m_l852_85240

theorem quadratic_has_two_real_distinct_roots_and_find_m 
  (m : ℝ) :
  (x : ℝ) → 
  (h1 : x^2 - (2 * m - 2) * x + (m^2 - 2 * m) = 0) →
  (x1 x2 : ℝ) →
  (h2 : x1^2 + x2^2 = 10) →
  (x1 + x2 = 2 * m - 2) →
  (x1 * x2 = m^2 - 2 * m) →
  (x1 ≠ x2) ∧ (m = -1 ∨ m = 3) :=
by sorry

end quadratic_has_two_real_distinct_roots_and_find_m_l852_85240


namespace water_pouring_problem_l852_85289

theorem water_pouring_problem : ∃ n : ℕ, n = 3 ∧
  (1 / (2 * n - 1) = 1 / 5) :=
by
  sorry

end water_pouring_problem_l852_85289


namespace sin_diff_l852_85251

variable (θ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5)

theorem sin_diff
  (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5) :
  Real.sin (θ - π / 4) = Real.sqrt 10 / 10 :=
sorry

end sin_diff_l852_85251


namespace inverse_proportion_quadrants_l852_85247

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0))) ↔ k > 0 := by
  sorry

end inverse_proportion_quadrants_l852_85247


namespace nth_wise_number_1990_l852_85220

/--
A natural number that can be expressed as the difference of squares 
of two other natural numbers is called a "wise number".
-/
def is_wise_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 - y^2 = n

/--
The 1990th "wise number" is 2659.
-/
theorem nth_wise_number_1990 : ∃ n : ℕ, is_wise_number n ∧ n = 2659 :=
  sorry

end nth_wise_number_1990_l852_85220


namespace intersection_point_exists_l852_85243

def line_param_eq (x y z : ℝ) (t : ℝ) := x = 5 + t ∧ y = 3 - t ∧ z = 2
def plane_eq (x y z : ℝ) := 3 * x + y - 5 * z - 12 = 0

theorem intersection_point_exists : 
  ∃ t : ℝ, ∃ x y z : ℝ, line_param_eq x y z t ∧ plane_eq x y z ∧ x = 7 ∧ y = 1 ∧ z = 2 :=
by {
  -- Skipping the proof
  sorry
}

end intersection_point_exists_l852_85243


namespace incorrect_comparison_l852_85241

theorem incorrect_comparison :
  ¬ (- (2 / 3) < - (4 / 5)) :=
by
  sorry

end incorrect_comparison_l852_85241


namespace Jack_sent_correct_number_of_BestBuy_cards_l852_85261

def price_BestBuy_gift_card : ℕ := 500
def price_Walmart_gift_card : ℕ := 200
def initial_BestBuy_gift_cards : ℕ := 6
def initial_Walmart_gift_cards : ℕ := 9

def total_price_of_initial_gift_cards : ℕ :=
  (initial_BestBuy_gift_cards * price_BestBuy_gift_card) +
  (initial_Walmart_gift_cards * price_Walmart_gift_card)

def price_of_Walmart_sent : ℕ := 2 * price_Walmart_gift_card
def value_of_gift_cards_remaining : ℕ := 3900

def prove_sent_BestBuy_worth : Prop :=
  total_price_of_initial_gift_cards - value_of_gift_cards_remaining - price_of_Walmart_sent = 1 * price_BestBuy_gift_card

theorem Jack_sent_correct_number_of_BestBuy_cards :
  prove_sent_BestBuy_worth :=
by
  sorry

end Jack_sent_correct_number_of_BestBuy_cards_l852_85261


namespace positive_A_satisfies_eq_l852_85292

theorem positive_A_satisfies_eq :
  ∃ (A : ℝ), A > 0 ∧ A^2 + 49 = 194 → A = Real.sqrt 145 :=
by
  sorry

end positive_A_satisfies_eq_l852_85292


namespace orange_count_in_bin_l852_85242

-- Definitions of the conditions
def initial_oranges : Nat := 5
def oranges_thrown_away : Nat := 2
def new_oranges_added : Nat := 28

-- The statement of the proof problem
theorem orange_count_in_bin : initial_oranges - oranges_thrown_away + new_oranges_added = 31 :=
by
  sorry

end orange_count_in_bin_l852_85242


namespace fraction_multiplication_l852_85218

theorem fraction_multiplication :
  ((2 / 5) * (5 / 7) * (7 / 3) * (3 / 8) = 1 / 4) :=
sorry

end fraction_multiplication_l852_85218


namespace part_I_part_II_l852_85235

open Real  -- Specify that we are working with real numbers

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- The first theorem: Prove the result for a = 1
theorem part_I (x : ℝ) : f x 1 + x > 0 ↔ (x > -3 ∧ x < 1 ∨ x > 3) :=
by
  sorry

-- The second theorem: Prove the range of a such that f(x) ≤ 3 for all x
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 3) ↔ (-5 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l852_85235


namespace value_of_a_m_minus_2n_l852_85279

variable (a : ℝ) (m n : ℝ)

theorem value_of_a_m_minus_2n (h1 : a^m = 8) (h2 : a^n = 4) : a^(m - 2 * n) = 1 / 2 :=
by
  sorry

end value_of_a_m_minus_2n_l852_85279


namespace min_squared_distance_l852_85280

theorem min_squared_distance : 
  ∀ (x y : ℝ), (x - y = 1) → (∃ (a b : ℝ), 
  ((a - 2) ^ 2 + (b - 2) ^ 2 <= (x - 2) ^ 2 + (y - 2) ^ 2) ∧ ((a - 2) ^ 2 + (b - 2) ^ 2 = 1 / 2)) := 
by
  sorry

end min_squared_distance_l852_85280


namespace exists_divisible_by_3_l852_85200

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end exists_divisible_by_3_l852_85200


namespace min_product_ab_l852_85227

theorem min_product_ab (a b : ℝ) (h : 20 * a * b = 13 * a + 14 * b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a * b = 1.82 :=
sorry

end min_product_ab_l852_85227


namespace fruit_seller_profit_percentage_l852_85284

/-- Suppose a fruit seller sells mangoes at the rate of Rs. 12 per kg and incurs a loss of 15%. 
    The mangoes should have been sold at Rs. 14.823529411764707 per kg to make a specific profit percentage. 
    This statement proves that the profit percentage is 5%. 
-/
theorem fruit_seller_profit_percentage :
  ∃ P : ℝ, 
    (∀ (CP SP : ℝ), 
        SP = 14.823529411764707 ∧ CP = 12 / 0.85 → 
        SP = CP * (1 + P / 100)) → 
    P = 5 := 
sorry

end fruit_seller_profit_percentage_l852_85284


namespace product_discount_rate_l852_85255

theorem product_discount_rate (cost_price marked_price : ℝ) (desired_profit_rate : ℝ) :
  cost_price = 200 → marked_price = 300 → desired_profit_rate = 0.2 →
  (∃ discount_rate : ℝ, discount_rate = 0.8 ∧ marked_price * discount_rate = cost_price * (1 + desired_profit_rate)) :=
by
  intros
  sorry

end product_discount_rate_l852_85255


namespace sum_of_products_circle_l852_85267

theorem sum_of_products_circle 
  (a b c d : ℤ) 
  (h : a + b + c + d = 0) : 
  -((a * (b + d)) + (b * (a + c)) + (c * (b + d)) + (d * (a + c))) = 2 * (a + c) ^ 2 :=
sorry

end sum_of_products_circle_l852_85267


namespace range_of_a_l852_85276

noncomputable
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (A a ∪ B = B) ↔ a < -4 ∨ a > 5 :=
sorry

end range_of_a_l852_85276


namespace kindergarteners_line_up_probability_l852_85213

theorem kindergarteners_line_up_probability :
  let total_line_up := Nat.choose 20 9
  let first_scenario := Nat.choose 14 9
  let second_scenario_single := Nat.choose 13 8
  let second_scenario := 6 * second_scenario_single
  let valid_arrangements := first_scenario + second_scenario
  valid_arrangements / total_line_up = 9724 / 167960 := by
  sorry

end kindergarteners_line_up_probability_l852_85213


namespace sum_squares_of_solutions_eq_l852_85294

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l852_85294


namespace negation_of_universal_proposition_l852_85296

theorem negation_of_universal_proposition :
  ¬ (∀ (m : ℝ), ∃ (x : ℝ), x^2 + x + m = 0) ↔ ∃ (m : ℝ), ¬ ∃ (x : ℝ), x^2 + x + m = 0 :=
by sorry

end negation_of_universal_proposition_l852_85296


namespace smallest_lcm_value_theorem_l852_85274

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l852_85274


namespace power_sum_roots_l852_85254

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end power_sum_roots_l852_85254


namespace find_x_given_y_l852_85286

variable (x y : ℝ)

theorem find_x_given_y :
  (0 < x) → (0 < y) → 
  (∃ k : ℝ, (3 * x^2 * y = k)) → 
  (y = 18 → x = 3) → 
  (y = 2400) → 
  x = 9 * Real.sqrt 6 / 85 :=
by
  -- Proof goes here
  sorry

end find_x_given_y_l852_85286


namespace area_identity_tg_cos_l852_85290

variable (a b c α β γ : Real)
variable (s t : Real) (area_of_triangle : Real)

-- Assume t is the area of the triangle and s is the semiperimeter
axiom area_of_triangle_eq_heron :
  t = Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Assume trigonometric identities for tangents and cosines of half-angles
axiom tg_half_angle_α : Real.tan (α / 2) = Real.sqrt ((s - b) * (s - c) / (s * (s - a)))
axiom tg_half_angle_β : Real.tan (β / 2) = Real.sqrt ((s - c) * (s - a) / (s * (s - b)))
axiom tg_half_angle_γ : Real.tan (γ / 2) = Real.sqrt ((s - a) * (s - b) / (s * (s - c)))

axiom cos_half_angle_α : Real.cos (α / 2) = Real.sqrt (s * (s - a) / (b * c))
axiom cos_half_angle_β : Real.cos (β / 2) = Real.sqrt (s * (s - b) / (c * a))
axiom cos_half_angle_γ : Real.cos (γ / 2) = Real.sqrt (s * (s - c) / (a * b))

theorem area_identity_tg_cos :
  t = s^2 * Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) ∧
  t = (a * b * c / s) * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) :=
by
  sorry

end area_identity_tg_cos_l852_85290


namespace wavelength_scientific_notation_l852_85278

theorem wavelength_scientific_notation :
  (0.000000193 : Float) = 1.93 * (10 : Float) ^ (-7) :=
sorry

end wavelength_scientific_notation_l852_85278


namespace pizza_problem_l852_85269

noncomputable def pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) : ℕ := 
  let slices_with_both := total_slices - (pepperoni_slices + mushroom_slices - total_slices)
  slices_with_both

theorem pizza_problem 
  (total_slices pepperoni_slices mushroom_slices : ℕ)
  (h_total: total_slices = 16)
  (h_pepperoni: pepperoni_slices = 8)
  (h_mushrooms: mushroom_slices = 12)
  (h_at_least_one: pepperoni_slices + mushroom_slices - total_slices ≥ 0)
  (h_no_three_toppings: total_slices = pepperoni_slices + mushroom_slices - 
   (total_slices - (pepperoni_slices + mushroom_slices - total_slices))) : 
  pizza_slices total_slices pepperoni_slices mushroom_slices = 4 :=
by 
  rw [h_total, h_pepperoni, h_mushrooms]
  sorry

end pizza_problem_l852_85269


namespace smallest_k_value_eq_sqrt475_div_12_l852_85270

theorem smallest_k_value_eq_sqrt475_div_12 :
  ∀ (k : ℝ), (dist (⟨5 * Real.sqrt 3, k - 2⟩ : ℝ × ℝ) ⟨0, 0⟩ = 5 * k) →
  k = (1 + Real.sqrt 475) / 12 := 
by
  intro k
  sorry

end smallest_k_value_eq_sqrt475_div_12_l852_85270


namespace bezdikov_population_l852_85211

variable (W M : ℕ) -- original number of women and men
variable (W_current M_current : ℕ) -- current number of women and men

theorem bezdikov_population (h1 : W = M + 30)
                          (h2 : W_current = W / 4)
                          (h3 : M_current = M - 196)
                          (h4 : W_current = M_current + 10) : W_current + M_current = 134 :=
by
  sorry

end bezdikov_population_l852_85211


namespace regina_earnings_l852_85248

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end regina_earnings_l852_85248


namespace savings_percentage_is_correct_l852_85260

-- Definitions for given conditions
def jacket_original_price : ℕ := 100
def shirt_original_price : ℕ := 50
def shoes_original_price : ℕ := 60

def jacket_discount : ℝ := 0.30
def shirt_discount : ℝ := 0.40
def shoes_discount : ℝ := 0.25

-- Definitions for savings
def jacket_savings : ℝ := jacket_original_price * jacket_discount
def shirt_savings : ℝ := shirt_original_price * shirt_discount
def shoes_savings : ℝ := shoes_original_price * shoes_discount

-- Definition for total savings and total original cost
def total_savings : ℝ := jacket_savings + shirt_savings + shoes_savings
def total_original_cost : ℕ := jacket_original_price + shirt_original_price + shoes_original_price

-- The theorems to be proven
theorem savings_percentage_is_correct : (total_savings / total_original_cost * 100) = 30.95 := by
  sorry

end savings_percentage_is_correct_l852_85260


namespace katie_total_marbles_l852_85208

def pink_marbles := 13
def orange_marbles := pink_marbles - 9
def purple_marbles := 4 * orange_marbles
def blue_marbles := 2 * purple_marbles
def total_marbles := pink_marbles + orange_marbles + purple_marbles + blue_marbles

theorem katie_total_marbles : total_marbles = 65 := 
by
  -- The proof is omitted here.
  sorry

end katie_total_marbles_l852_85208


namespace correct_operation_l852_85237

variable (a b : ℝ)

theorem correct_operation : (-a^2 * b + 2 * a^2 * b = a^2 * b) :=
by sorry

end correct_operation_l852_85237


namespace oliver_gave_janet_l852_85271

def initial_candy : ℕ := 78
def remaining_candy : ℕ := 68

theorem oliver_gave_janet : initial_candy - remaining_candy = 10 :=
by
  sorry

end oliver_gave_janet_l852_85271


namespace least_number_to_addition_l852_85253

-- Given conditions
def n : ℤ := 2496

-- The least number to be added to n to make it divisible by 5
def least_number_to_add (n : ℤ) : ℤ :=
  if (n % 5 = 0) then 0 else (5 - (n % 5))

-- Prove that adding 4 to 2496 makes it divisible by 5
theorem least_number_to_addition : (least_number_to_add n) = 4 :=
  by
    sorry

end least_number_to_addition_l852_85253


namespace question_correctness_l852_85265

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l852_85265


namespace find_single_digit_number_l852_85239

theorem find_single_digit_number (n : ℕ) : 
  (5 < n ∧ n < 9 ∧ n > 7) ↔ n = 8 :=
by
  sorry

end find_single_digit_number_l852_85239


namespace express_A_using_roster_method_l852_85277

def A := {x : ℕ | ∃ (n : ℕ), 8 / (2 - x) = n }

theorem express_A_using_roster_method :
  A = {0, 1} :=
sorry

end express_A_using_roster_method_l852_85277


namespace train_length_in_terms_of_james_cycle_l852_85217

/-- Define the mathematical entities involved: L (train length), J (James's cycle length), T (train length per cycle) -/
theorem train_length_in_terms_of_james_cycle 
  (L J T : ℝ) 
  (h1 : 130 * J = L + 130 * T) 
  (h2 : 26 * J = L - 26 * T) 
    : L = 58 * J := 
by 
  sorry

end train_length_in_terms_of_james_cycle_l852_85217


namespace rectangle_properties_l852_85230

noncomputable def diagonal (x1 y1 x2 y2 : ℕ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def area (length width : ℕ) : ℕ :=
  length * width

theorem rectangle_properties :
  diagonal 1 1 9 7 = 10 ∧ area (9 - 1) (7 - 1) = 48 := by
  sorry

end rectangle_properties_l852_85230


namespace range_of_a_l852_85224

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  is_increasing_on (λ x => x^2 + a * x + 1 / x) (Set.Ioi (1 / 2)) ↔ 3 ≤ a := 
by
  sorry

end range_of_a_l852_85224


namespace black_eyes_ratio_l852_85285

-- Define the number of people in the theater
def total_people : ℕ := 100

-- Define the number of people with blue eyes
def blue_eyes : ℕ := 19

-- Define the number of people with brown eyes
def brown_eyes : ℕ := 50

-- Define the number of people with green eyes
def green_eyes : ℕ := 6

-- Define the number of people with black eyes
def black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)

-- Prove that the ratio of the number of people with black eyes to the total number of people is 1:4
theorem black_eyes_ratio :
  black_eyes * 4 = total_people := by
  sorry

end black_eyes_ratio_l852_85285


namespace intersection_of_AB_CD_l852_85223

def point (α : Type*) := (α × α × α)

def A : point ℚ := (5, -8, 9)
def B : point ℚ := (15, -18, 14)
def C : point ℚ := (1, 4, -7)
def D : point ℚ := (3, -4, 11)

def parametric_AB (t : ℚ) : point ℚ :=
  (5 + 10 * t, -8 - 10 * t, 9 + 5 * t)

def parametric_CD (s : ℚ) : point ℚ :=
  (1 + 2 * s, 4 - 8 * s, -7 + 18 * s)

def intersection_point (pi : point ℚ) :=
  ∃ t s : ℚ, parametric_AB t = pi ∧ parametric_CD s = pi

theorem intersection_of_AB_CD : intersection_point (76/15, -118/15, 170/15) :=
  sorry

end intersection_of_AB_CD_l852_85223


namespace abs_neg_two_eq_two_l852_85221

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end abs_neg_two_eq_two_l852_85221


namespace num_of_terms_in_arith_seq_l852_85210

-- Definitions of the conditions
def a : Int := -5 -- Start of the arithmetic sequence
def l : Int := 85 -- End of the arithmetic sequence
def d : Nat := 5  -- Common difference

-- The theorem that needs to be proved
theorem num_of_terms_in_arith_seq : (l - a) / d + 1 = 19 := sorry

end num_of_terms_in_arith_seq_l852_85210


namespace inequality_sqrt_sum_l852_85238

theorem inequality_sqrt_sum (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2 - a * b) + Real.sqrt (b^2 + c^2 - b * c)) ≥ Real.sqrt (a^2 + c^2 + a * c) :=
sorry

end inequality_sqrt_sum_l852_85238


namespace find_x_l852_85288

theorem find_x (x : ℝ) (h : (1 / Real.log x / Real.log 5 + 1 / Real.log x / Real.log 7 + 1 / Real.log x / Real.log 11) = 1) : x = 385 := 
sorry

end find_x_l852_85288


namespace no_such_ab_exists_l852_85244

theorem no_such_ab_exists : ¬ ∃ (a b : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → (a * x + b)^2 - Real.cos x * (a * x + b) < (1 / 4) * (Real.sin x)^2 :=
by
  sorry

end no_such_ab_exists_l852_85244


namespace isosceles_triangle_perimeter_l852_85263

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : b < a + a) : a + a + b = 22 := by
  sorry

end isosceles_triangle_perimeter_l852_85263


namespace div_by_3_iff_n_form_l852_85201

theorem div_by_3_iff_n_form (n : ℕ) : (3 ∣ (n * 2^n + 1)) ↔ (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 2) :=
by
  sorry

end div_by_3_iff_n_form_l852_85201


namespace compare_abc_l852_85246

noncomputable def a := Real.sqrt 0.3
noncomputable def b := Real.sqrt 0.4
noncomputable def c := Real.log 0.6 / Real.log 3

theorem compare_abc : c < a ∧ a < b :=
by
  -- Proof goes here
  sorry

end compare_abc_l852_85246


namespace units_digit_k_squared_plus_2_k_l852_85257

noncomputable def k : ℕ := 2017^2 + 2^2017

theorem units_digit_k_squared_plus_2_k : (k^2 + 2^k) % 10 = 3 := 
  sorry

end units_digit_k_squared_plus_2_k_l852_85257


namespace bettys_herb_garden_l852_85204

theorem bettys_herb_garden :
  ∀ (basil oregano thyme rosemary total : ℕ),
    oregano = 2 * basil + 2 →
    thyme = 3 * basil - 3 →
    rosemary = (basil + thyme) / 2 →
    basil = 5 →
    total = basil + oregano + thyme + rosemary →
    total ≤ 50 →
    total = 37 :=
by
  intros basil oregano thyme rosemary total h_oregano h_thyme h_rosemary h_basil h_total h_le_total
  sorry

end bettys_herb_garden_l852_85204


namespace present_population_l852_85282

theorem present_population (P : ℝ) (h : 1.04 * P = 1289.6) : P = 1240 :=
by
  sorry

end present_population_l852_85282


namespace geometric_sequence_eighth_term_l852_85233

theorem geometric_sequence_eighth_term (a r : ℝ) (h₀ : a = 27) (h₁ : r = 1/3) :
  a * r^7 = 1/81 :=
by
  rw [h₀, h₁]
  sorry

end geometric_sequence_eighth_term_l852_85233


namespace perpendicular_tangent_inequality_l852_85275

variable {A B C : Type} 

-- Definitions according to conditions in part a)
def isAcuteAngledTriangle (a b c : Type) : Prop :=
  -- A triangle being acute-angled in Euclidean geometry
  sorry

def triangleArea (a b c : Type) : ℝ :=
  -- Definition of the area of a triangle
  sorry

def perpendicularLengthToLine (point line : Type) : ℝ :=
  -- Length of the perpendicular from a point to a line
  sorry

def tangentOfAngleA (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle A in the triangle
  sorry

def tangentOfAngleB (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle B in the triangle
  sorry

def tangentOfAngleC (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle C in the triangle
  sorry

theorem perpendicular_tangent_inequality (a b c line : Type) 
  (ht : isAcuteAngledTriangle a b c)
  (u := perpendicularLengthToLine a line)
  (v := perpendicularLengthToLine b line)
  (w := perpendicularLengthToLine c line):
  u^2 * tangentOfAngleA a b c + v^2 * tangentOfAngleB a b c + w^2 * tangentOfAngleC a b c ≥ 
  2 * triangleArea a b c :=
sorry

end perpendicular_tangent_inequality_l852_85275


namespace intersection_points_relation_l852_85226

-- Suppressing noncomputable theory to focus on the structure
-- of the Lean statement rather than computability aspects.

noncomputable def intersection_points (k : ℕ) : ℕ :=
sorry -- This represents the function f(k)

axiom no_parallel (k : ℕ) : Prop
axiom no_three_intersect (k : ℕ) : Prop

theorem intersection_points_relation (k : ℕ) (h1 : no_parallel k) (h2 : no_three_intersect k) :
  intersection_points (k + 1) = intersection_points k + k :=
sorry

end intersection_points_relation_l852_85226


namespace value_at_points_zero_l852_85249

def odd_function (v : ℝ → ℝ) := ∀ x : ℝ, v (-x) = -v x

theorem value_at_points_zero (v : ℝ → ℝ)
  (hv : odd_function v) :
  v (-2.1) + v (-1.2) + v (1.2) + v (2.1) = 0 :=
by {
  sorry
}

end value_at_points_zero_l852_85249


namespace find_base_numerica_l852_85236

theorem find_base_numerica (r : ℕ) (h_gadget_cost : 5*r^2 + 3*r = 530) (h_payment : r^3 + r^2 = 1100) (h_change : 4*r^2 + 6*r = 460) :
  r = 9 :=
sorry

end find_base_numerica_l852_85236


namespace quadratic_equation_solution_l852_85207

theorem quadratic_equation_solution (m : ℝ) (h : m ≠ 1) : 
  (m^2 - 3 * m + 2 = 0) → m = 2 :=
by
  sorry

end quadratic_equation_solution_l852_85207


namespace greatest_common_divisor_is_40_l852_85299

def distance_to_boston : ℕ := 840
def distance_to_atlanta : ℕ := 440

theorem greatest_common_divisor_is_40 :
  Nat.gcd distance_to_boston distance_to_atlanta = 40 :=
by
  -- The theorem statement as described is correct
  -- Proof is omitted as per instructions
  sorry

end greatest_common_divisor_is_40_l852_85299


namespace abs_sum_div_diff_sqrt_7_5_l852_85266

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end abs_sum_div_diff_sqrt_7_5_l852_85266


namespace percentage_needed_to_pass_l852_85225

def MikeScore : ℕ := 212
def Shortfall : ℕ := 19
def MaxMarks : ℕ := 770

theorem percentage_needed_to_pass :
  (231.0 / (770.0 : ℝ)) * 100 = 30 := by
  -- placeholder for proof
  sorry

end percentage_needed_to_pass_l852_85225


namespace range_of_m_l852_85231

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end range_of_m_l852_85231


namespace g_neither_even_nor_odd_l852_85283

noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem g_neither_even_nor_odd :
  (∀ x, g (-x) = g x → false) ∧ (∀ x, g (-x) = -g x → false) :=
by
  unfold g
  sorry

end g_neither_even_nor_odd_l852_85283


namespace polynomial_divisible_by_3_l852_85252

/--
Given q and p are integers where q is divisible by 3 and p+1 is divisible by 3,
prove that the polynomial Q(x) = x^3 - x + (p+1)x + q is divisible by 3 for any integer x.
-/
theorem polynomial_divisible_by_3 (q p : ℤ) (hq : 3 ∣ q) (hp1 : 3 ∣ (p + 1)) :
  ∀ x : ℤ, 3 ∣ (x^3 - x + (p+1) * x + q) :=
by {
  sorry
}

end polynomial_divisible_by_3_l852_85252


namespace simplify_expression_l852_85259

theorem simplify_expression (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := 
  sorry

end simplify_expression_l852_85259


namespace polygon_area_is_400_l852_85298

-- Definition of the points and polygon
def Point := (ℝ × ℝ)
def Polygon := List Point

def points : List Point := [(0, 0), (20, 0), (20, 20), (0, 20), (10, 0), (20, 10), (10, 20), (0, 10)]

def polygon : Polygon := [(0,0), (10,0), (20,10), (20,20), (10,20), (0,10), (0,0)]

-- Function to calculate the area of the polygon
noncomputable def polygon_area (p : Polygon) : ℝ := 
  -- Assume we have the necessary function to calculate the area of a polygon given a list of vertices
  sorry

-- Theorem statement: The area of the given polygon is 400
theorem polygon_area_is_400 : polygon_area polygon = 400 := sorry

end polygon_area_is_400_l852_85298


namespace line_equation_l852_85222

theorem line_equation {a b c : ℝ} (x : ℝ) (y : ℝ)
  (point : ∃ p: ℝ × ℝ, p = (-1, 0))
  (perpendicular : ∀ k: ℝ, k = 1 → 
    ∀ m: ℝ, m = -1 → 
      ∀ b1: ℝ, b1 = 0 → 
        ∀ x1: ℝ, x1 = -1 →
          ∀ y1: ℝ, y1 = 0 →
            ∀ l: ℝ, l = b1 + k * (x1 - (-1)) + m * (y1 - 0) → 
              x - y + 1 = 0) :
  x - y + 1 = 0 :=
sorry

end line_equation_l852_85222


namespace milan_billed_minutes_l852_85273

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l852_85273


namespace aqua_park_earnings_l852_85209

def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def meal_fee : ℕ := 10
def souvenir_fee : ℕ := 8

def group1_admission_count : ℕ := 10
def group1_tour_count : ℕ := 10
def group1_meal_count : ℕ := 10
def group1_souvenir_count : ℕ := 10
def group1_discount : ℚ := 0.10

def group2_admission_count : ℕ := 15
def group2_meal_count : ℕ := 15
def group2_meal_discount : ℚ := 0.05

def group3_admission_count : ℕ := 8
def group3_tour_count : ℕ := 8
def group3_souvenir_count : ℕ := 8

-- total cost for group 1 before discount
def group1_total_before_discount : ℕ := 
  (group1_admission_count * admission_fee) +
  (group1_tour_count * tour_fee) +
  (group1_meal_count * meal_fee) +
  (group1_souvenir_count * souvenir_fee)

-- group 1 total cost after discount
def group1_total_after_discount : ℚ :=
  group1_total_before_discount * (1 - group1_discount)

-- total cost for group 2 before discount
def group2_admission_total_before_discount : ℕ := 
  group2_admission_count * admission_fee
def group2_meal_total_before_discount : ℕ := 
  group2_meal_count * meal_fee

-- group 2 total cost after discount
def group2_meal_total_after_discount : ℚ :=
  group2_meal_total_before_discount * (1 - group2_meal_discount)
def group2_total_after_discount : ℚ :=
  group2_admission_total_before_discount + group2_meal_total_after_discount

-- total cost for group 3 before discount
def group3_total_before_discount : ℕ := 
  (group3_admission_count * admission_fee) +
  (group3_tour_count * tour_fee) +
  (group3_souvenir_count * souvenir_fee)

-- group 3 total cost after discount (no discount applied)
def group3_total_after_discount : ℕ := group3_total_before_discount

-- total earnings from all groups
def total_earnings : ℚ :=
  group1_total_after_discount +
  group2_total_after_discount +
  group3_total_after_discount

theorem aqua_park_earnings : total_earnings = 854.50 := by
  sorry

end aqua_park_earnings_l852_85209


namespace value_of_k_l852_85262

theorem value_of_k (x y k : ℝ) (h1 : 3 * x + 2 * y = k + 1) (h2 : 2 * x + 3 * y = k) (h3 : x + y = 2) :
  k = 9 / 2 :=
by
  sorry

end value_of_k_l852_85262


namespace average_of_numbers_l852_85250

theorem average_of_numbers (x : ℝ) (h : (2 + x + 12) / 3 = 8) : x = 10 :=
by sorry

end average_of_numbers_l852_85250


namespace increased_cost_is_4_percent_l852_85272

-- Initial declarations
variables (initial_cost : ℕ) (price_change_eggs price_change_apples percentage_increase : ℕ)

-- Cost definitions based on initial conditions
def initial_cost_eggs := 100
def initial_cost_apples := 100

-- Price adjustments
def new_cost_eggs := initial_cost_eggs - (initial_cost_eggs * 2 / 100)
def new_cost_apples := initial_cost_apples + (initial_cost_apples * 10 / 100)

-- New combined cost
def new_combined_cost := new_cost_eggs + new_cost_apples

-- Old combined cost
def old_combined_cost := initial_cost_eggs + initial_cost_apples

-- Increase in cost
def increase_in_cost := new_combined_cost - old_combined_cost

-- Percentage increase
def calculated_percentage_increase := (increase_in_cost * 100) / old_combined_cost

-- The proof statement
theorem increased_cost_is_4_percent :
  initial_cost = 100 →
  price_change_eggs = 2 →
  price_change_apples = 10 →
  percentage_increase = 4 →
  calculated_percentage_increase = percentage_increase :=
sorry

end increased_cost_is_4_percent_l852_85272


namespace max_positive_integer_value_l852_85264

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n: ℕ, ∃ q: ℝ, a (n + 1) = a n * q

theorem max_positive_integer_value
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 = 4)
  (h4 : a 1 + a 2 + a 3 = 14) : 
  ∃ n, n ≤ 4 ∧ a n * a (n+1) * a (n+2) > 1 / 9 :=
sorry

end max_positive_integer_value_l852_85264


namespace mayo_bottles_count_l852_85291

theorem mayo_bottles_count
  (ketchup_ratio mayo_ratio : ℕ) 
  (ratio_multiplier ketchup_bottles : ℕ)
  (h_ratio_eq : 3 = ketchup_ratio)
  (h_mayo_ratio_eq : 2 = mayo_ratio)
  (h_ketchup_bottles_eq : 6 = ketchup_bottles)
  (h_ratio_condition : ketchup_bottles * mayo_ratio = ketchup_ratio * ratio_multiplier) :
  ratio_multiplier = 4 := 
by 
  sorry

end mayo_bottles_count_l852_85291


namespace beads_per_bracelet_l852_85205

-- Definitions for the conditions
def Nancy_metal_beads : ℕ := 40
def Nancy_pearl_beads : ℕ := Nancy_metal_beads + 20
def Rose_crystal_beads : ℕ := 20
def Rose_stone_beads : ℕ := Rose_crystal_beads * 2
def total_beads : ℕ := Nancy_metal_beads + Nancy_pearl_beads + Rose_crystal_beads + Rose_stone_beads
def bracelets : ℕ := 20

-- Statement to prove
theorem beads_per_bracelet :
  total_beads / bracelets = 8 :=
by
  -- skip the proof
  sorry

end beads_per_bracelet_l852_85205


namespace difference_between_mean_and_median_l852_85214

def percent_students := {p : ℝ // 0 ≤ p ∧ p ≤ 1}

def students_scores_distribution (p60 p75 p85 p95 : percent_students) : Prop :=
  p60.val + p75.val + p85.val + p95.val = 1 ∧
  p60.val = 0.15 ∧
  p75.val = 0.20 ∧
  p85.val = 0.40 ∧
  p95.val = 0.25

noncomputable def weighted_mean (p60 p75 p85 p95 : percent_students) : ℝ :=
  60 * p60.val + 75 * p75.val + 85 * p85.val + 95 * p95.val

noncomputable def median_score (p60 p75 p85 p95 : percent_students) : ℝ :=
  if p60.val + p75.val < 0.5 then 85 else if p60.val + p75.val < 0.9 then 95 else 60

theorem difference_between_mean_and_median :
  ∀ (p60 p75 p85 p95 : percent_students),
    students_scores_distribution p60 p75 p85 p95 →
    abs (median_score p60 p75 p85 p95 - weighted_mean p60 p75 p85 p95) = 3.25 :=
by
  intro p60 p75 p85 p95
  intro h
  sorry

end difference_between_mean_and_median_l852_85214


namespace Tom_water_intake_daily_l852_85293

theorem Tom_water_intake_daily (cans_per_day : ℕ) (oz_per_can : ℕ) (fluid_per_week : ℕ) (days_per_week : ℕ)
  (h1 : cans_per_day = 5) 
  (h2 : oz_per_can = 12) 
  (h3 : fluid_per_week = 868) 
  (h4 : days_per_week = 7) : 
  ((fluid_per_week - (cans_per_day * oz_per_can * days_per_week)) / days_per_week) = 64 := 
sorry

end Tom_water_intake_daily_l852_85293


namespace similar_triangles_x_value_l852_85234

-- Define the conditions of the problem
variables (x : ℝ) (h₁ : 10 / x = 8 / 5)

-- State the theorem/proof problem
theorem similar_triangles_x_value : x = 6.25 :=
by
  -- Proof goes here
  sorry

end similar_triangles_x_value_l852_85234


namespace problem_statement_l852_85295

noncomputable def smallest_integer_exceeding := 
  let x : ℝ := (Real.sqrt 3 + Real.sqrt 2) ^ 8
  Int.ceil x

theorem problem_statement : smallest_integer_exceeding = 5360 :=
by 
  -- The proof is omitted
  sorry

end problem_statement_l852_85295
