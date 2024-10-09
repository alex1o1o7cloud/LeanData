import Mathlib

namespace triangle_obtuse_l1022_102232

def is_obtuse_triangle (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

theorem triangle_obtuse (A B C : ℝ) (h1 : A > 3 * B) (h2 : C < 2 * B) (h3 : A + B + C = 180) : is_obtuse_triangle A B C :=
by sorry

end triangle_obtuse_l1022_102232


namespace ajay_total_gain_l1022_102207

noncomputable def ajay_gain : ℝ :=
  let cost1 := 15 * 14.50
  let cost2 := 10 * 13
  let total_cost := cost1 + cost2
  let total_weight := 15 + 10
  let selling_price := total_weight * 15
  selling_price - total_cost

theorem ajay_total_gain :
  ajay_gain = 27.50 := by
  sorry

end ajay_total_gain_l1022_102207


namespace ava_legs_count_l1022_102260

-- Conditions:
-- There are a total of 9 animals in the farm.
-- There are only chickens and buffalos in the farm.
-- There are 5 chickens in the farm.

def total_animals : Nat := 9
def num_chickens : Nat := 5
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

-- Proof statement: Ava counted 26 legs.
theorem ava_legs_count (num_buffalos : Nat) 
  (H1 : total_animals = num_chickens + num_buffalos) : 
  num_chickens * legs_per_chicken + num_buffalos * legs_per_buffalo = 26 :=
by 
  have H2 : num_buffalos = total_animals - num_chickens := by sorry
  sorry

end ava_legs_count_l1022_102260


namespace samantha_tenth_finger_l1022_102229

def g (x : ℕ) : ℕ :=
  match x with
  | 2 => 2
  | _ => 0  -- Assume a simple piecewise definition for the sake of the example.

theorem samantha_tenth_finger : g (2) = 2 :=
by  sorry

end samantha_tenth_finger_l1022_102229


namespace calculate_expression_l1022_102219

theorem calculate_expression : 
  |(-3)| - 2 * Real.tan (Real.pi / 4) + (-1:ℤ)^(2023) - (Real.sqrt 3 - Real.pi)^(0:ℤ) = -1 :=
  by
  sorry

end calculate_expression_l1022_102219


namespace smallest_integer_cube_ends_in_528_l1022_102217

theorem smallest_integer_cube_ends_in_528 :
  ∃ (n : ℕ), (n^3 % 1000 = 528 ∧ ∀ m : ℕ, (m^3 % 1000 = 528) → m ≥ n) ∧ n = 428 :=
by
  sorry

end smallest_integer_cube_ends_in_528_l1022_102217


namespace diameter_of_larger_sphere_l1022_102248

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (hr : r = 9)
    (h1 : 3 * (4/3) * π * r^3 = (4/3) * π * ((2 * a * b^(1/3)) / 2)^3) 
    (h2 : ¬∃ c : ℕ, c^3 = b) : a + b = 21 :=
sorry

end diameter_of_larger_sphere_l1022_102248


namespace count_valid_a_values_l1022_102298

def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def valid_a_values (a : ℕ) : Prop :=
1 ≤ a ∧ a ≤ 100 ∧ is_perfect_square (16 * a + 9)

theorem count_valid_a_values :
  ∃ N : ℕ, N = Nat.card {a : ℕ | valid_a_values a} := sorry

end count_valid_a_values_l1022_102298


namespace opposite_neg_three_over_two_l1022_102209

-- Define the concept of the opposite number
def opposite (x : ℚ) : ℚ := -x

-- State the problem: The opposite number of -3/2 is 3/2
theorem opposite_neg_three_over_two :
  opposite (- (3 / 2 : ℚ)) = (3 / 2 : ℚ) := 
  sorry

end opposite_neg_three_over_two_l1022_102209


namespace compute_fraction_l1022_102200

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem compute_fraction : (f (g (f 1))) / (g (f (g 1))) = 6801 / 281 := 
by 
  sorry

end compute_fraction_l1022_102200


namespace range_of_m_l1022_102284

theorem range_of_m (x m : ℝ) (h₁ : x^2 - 3 * x + 2 > 0) (h₂ : ¬(x^2 - 3 * x + 2 > 0) → x < m) : 2 < m :=
by
  sorry

end range_of_m_l1022_102284


namespace daniel_initial_noodles_l1022_102278

variable (give : ℕ)
variable (left : ℕ)
variable (initial : ℕ)

theorem daniel_initial_noodles (h1 : give = 12) (h2 : left = 54) (h3 : initial = left + give) : initial = 66 := by
  sorry

end daniel_initial_noodles_l1022_102278


namespace gcd_13924_32451_eq_one_l1022_102240

-- Define the two given integers.
def x : ℕ := 13924
def y : ℕ := 32451

-- State and prove that the greatest common divisor of x and y is 1.
theorem gcd_13924_32451_eq_one : Nat.gcd x y = 1 := by
  sorry

end gcd_13924_32451_eq_one_l1022_102240


namespace greatest_integer_jo_thinking_of_l1022_102285

theorem greatest_integer_jo_thinking_of :
  ∃ n : ℕ, n < 150 ∧ (∃ k : ℕ, n = 9 * k - 1) ∧ (∃ m : ℕ, n = 5 * m - 2) ∧ n = 143 :=
by
  sorry

end greatest_integer_jo_thinking_of_l1022_102285


namespace ratio_sum_eq_l1022_102236

variable {x y z : ℝ}

-- Conditions: 3x, 4y, 5z form a geometric sequence
def geom_sequence (x y z : ℝ) : Prop :=
  (∃ r : ℝ, 4 * y = 3 * x * r ∧ 5 * z = 4 * y * r)

-- Conditions: 1/x, 1/y, 1/z form an arithmetic sequence
def arith_sequence (x y z : ℝ) : Prop :=
  2 * x * z = y * z + x * y

-- Conclude: x/z + z/x = 34/15
theorem ratio_sum_eq (h1 : geom_sequence x y z) (h2 : arith_sequence x y z) : 
  (x / z + z / x) = (34 / 15) :=
sorry

end ratio_sum_eq_l1022_102236


namespace sum_binomials_l1022_102216

-- Defining binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem sum_binomials : binom 12 4 + binom 10 3 = 615 :=
by
  -- Here we state the problem, and the proof will be left as 'sorry'.
  sorry

end sum_binomials_l1022_102216


namespace find_January_salary_l1022_102253

-- Definitions and conditions
variables (J F M A May : ℝ)
def avg_Jan_to_Apr : Prop := (J + F + M + A) / 4 = 8000
def avg_Feb_to_May : Prop := (F + M + A + May) / 4 = 8300
def May_salary : Prop := May = 6500

-- Theorem statement
theorem find_January_salary (h1 : avg_Jan_to_Apr J F M A) 
                            (h2 : avg_Feb_to_May F M A May) 
                            (h3 : May_salary May) : 
                            J = 5300 :=
sorry

end find_January_salary_l1022_102253


namespace find_x_cube_plus_reciprocal_cube_l1022_102247

variable {x : ℝ}

theorem find_x_cube_plus_reciprocal_cube (hx : x + 1/x = 10) : x^3 + 1/x^3 = 970 :=
sorry

end find_x_cube_plus_reciprocal_cube_l1022_102247


namespace min_expression_value_l1022_102215

theorem min_expression_value (x y z : ℝ) (xyz_eq : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n : ℝ, (∀ x y z : ℝ, x * y * z = 1 → 0 < x → 0 < y → 0 < z → 2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ n)
    ∧ n = 72 :=
sorry

end min_expression_value_l1022_102215


namespace no_m_for_P_eq_S_m_le_3_for_P_implies_S_l1022_102208

namespace ProofProblem

def P (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def S (m x : ℝ) : Prop := |x - 1| ≤ m

theorem no_m_for_P_eq_S : ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S m x := sorry

theorem m_le_3_for_P_implies_S : ∀ (m : ℝ), (m ≤ 3) → (∀ x, S m x → P x) := sorry

end ProofProblem

end no_m_for_P_eq_S_m_le_3_for_P_implies_S_l1022_102208


namespace probability_same_color_is_one_third_l1022_102287

-- Define a type for colors
inductive Color 
| red 
| white 
| blue 

open Color

-- Define the function to calculate the probability of the same color selection
def sameColorProbability : ℚ :=
  let total_outcomes := 3 * 3
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

-- Theorem stating that the probability is 1/3
theorem probability_same_color_is_one_third : sameColorProbability = 1 / 3 :=
by
  -- Steps of proof will be provided here
  sorry

end probability_same_color_is_one_third_l1022_102287


namespace count_divisible_by_8_l1022_102241

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l1022_102241


namespace volume_of_prism_l1022_102210

noncomputable def volume_of_triangular_prism
  (area_lateral_face : ℝ)
  (distance_cc1_to_lateral_face : ℝ) : ℝ :=
  area_lateral_face * distance_cc1_to_lateral_face

theorem volume_of_prism (area_lateral_face : ℝ) 
    (distance_cc1_to_lateral_face : ℝ)
    (h_area : area_lateral_face = 4)
    (h_distance : distance_cc1_to_lateral_face = 2):
  volume_of_triangular_prism area_lateral_face distance_cc1_to_lateral_face = 4 := by
  sorry

end volume_of_prism_l1022_102210


namespace real_roots_approx_correct_to_4_decimal_places_l1022_102292

noncomputable def f (x : ℝ) : ℝ := x^4 - (2 * 10^10 + 1) * x^2 - x + 10^20 + 10^10 - 1

theorem real_roots_approx_correct_to_4_decimal_places :
  ∃ x1 x2 : ℝ, 
  abs (x1 - 99999.9997) ≤ 0.0001 ∧ 
  abs (x2 - 100000.0003) ≤ 0.0001 ∧ 
  f x1 = 0 ∧ 
  f x2 = 0 :=
sorry

end real_roots_approx_correct_to_4_decimal_places_l1022_102292


namespace terminal_zeros_75_480_l1022_102251

theorem terminal_zeros_75_480 :
  let x := 75
  let y := 480
  let fact_x := 5^2 * 3
  let fact_y := 2^5 * 3 * 5
  let product := fact_x * fact_y
  let num_zeros := min (3) (5)
  num_zeros = 3 :=
by
  sorry

end terminal_zeros_75_480_l1022_102251


namespace number_of_girls_in_colins_class_l1022_102291

variables (g b : ℕ)

theorem number_of_girls_in_colins_class
  (h1 : g / b = 3 / 4)
  (h2 : g + b = 35)
  (h3 : b > 15) :
  g = 15 :=
sorry

end number_of_girls_in_colins_class_l1022_102291


namespace num_classes_received_basketballs_l1022_102261

theorem num_classes_received_basketballs (total_basketballs left_basketballs : ℕ) 
  (h : total_basketballs = 54) (h_left : left_basketballs = 5) : 
  (total_basketballs - left_basketballs) / 7 = 7 :=
by
  sorry

end num_classes_received_basketballs_l1022_102261


namespace max_f_value_l1022_102212

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ := (S_n n : ℝ) / ((n + 32) * S_n (n + 1))

theorem max_f_value : ∃ n : ℕ, f n = 1 / 50 := by
  sorry

end max_f_value_l1022_102212


namespace total_population_milburg_l1022_102249

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end total_population_milburg_l1022_102249


namespace range_of_f_l1022_102227

noncomputable def f (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 1)

theorem range_of_f : Set.Icc (-1/4 : ℝ) (1/4) = Set.range f :=
by
  sorry

end range_of_f_l1022_102227


namespace oliver_shelves_needed_l1022_102275

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end oliver_shelves_needed_l1022_102275


namespace t_is_perfect_square_l1022_102235

variable (n : ℕ) (hpos : 0 < n)
variable (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2))

theorem t_is_perfect_square (n : ℕ) (hpos : 0 < n) (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2)) : 
  ∃ k : ℕ, t = k * k := 
sorry

end t_is_perfect_square_l1022_102235


namespace interval_of_x_l1022_102295

theorem interval_of_x (x : ℝ) (h : x = ((-x)^2 / x) + 3) : 3 < x ∧ x ≤ 6 :=
by
  sorry

end interval_of_x_l1022_102295


namespace sum_of_digits_of_greatest_prime_divisor_of_4095_l1022_102242

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def greatest_prime_divisor_of_4095 : ℕ := 13

theorem sum_of_digits_of_greatest_prime_divisor_of_4095 :
  sum_of_digits greatest_prime_divisor_of_4095 = 4 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_4095_l1022_102242


namespace remainder_abc_mod9_l1022_102272

open Nat

-- Define the conditions for the problem
variables (a b c : ℕ)

-- Assume conditions: a, b, c are non-negative and less than 9, and the given congruences
theorem remainder_abc_mod9 (h1 : a < 9) (h2 : b < 9) (h3 : c < 9)
  (h4 : (a + 3 * b + 2 * c) % 9 = 3)
  (h5 : (2 * a + 2 * b + 3 * c) % 9 = 6)
  (h6 : (3 * a + b + 2 * c) % 9 = 1) :
  (a * b * c) % 9 = 4 :=
sorry

end remainder_abc_mod9_l1022_102272


namespace max_value_of_sample_l1022_102204

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end max_value_of_sample_l1022_102204


namespace ab_value_l1022_102223

theorem ab_value 
  (a b : ℝ) 
  (hx : 2 = b + 1) 
  (hy : a = -3) : 
  a * b = -3 :=
by
  sorry

end ab_value_l1022_102223


namespace kiera_total_envelopes_l1022_102267

-- Define the number of blue envelopes
def blue_envelopes : ℕ := 14

-- Define the number of yellow envelopes as 6 fewer than the number of blue envelopes
def yellow_envelopes : ℕ := blue_envelopes - 6

-- Define the number of green envelopes as 3 times the number of yellow envelopes
def green_envelopes : ℕ := 3 * yellow_envelopes

-- The total number of envelopes is the sum of blue, yellow, and green envelopes
def total_envelopes : ℕ := blue_envelopes + yellow_envelopes + green_envelopes

-- Prove that the total number of envelopes is 46
theorem kiera_total_envelopes : total_envelopes = 46 := by
  sorry

end kiera_total_envelopes_l1022_102267


namespace division_problem_l1022_102294

theorem division_problem (x y n : ℕ) 
  (h1 : x = n * y + 4) 
  (h2 : 2 * x = 14 * y + 1) 
  (h3 : 5 * y - x = 3) : n = 4 := 
sorry

end division_problem_l1022_102294


namespace least_subtraction_divisible_l1022_102211

theorem least_subtraction_divisible (n : ℕ) (h : n = 3830) (lcm_val : ℕ) (hlcm : lcm_val = Nat.lcm (Nat.lcm 3 7) 11) 
(largest_multiple : ℕ) (h_largest : largest_multiple = (n / lcm_val) * lcm_val) :
  ∃ x : ℕ, x = n - largest_multiple ∧ x = 134 := 
by
  sorry

end least_subtraction_divisible_l1022_102211


namespace proof_theorem_l1022_102264

noncomputable def proof_problem (y1 y2 y3 y4 y5 : ℝ) :=
  y1 + 8*y2 + 27*y3 + 64*y4 + 125*y5 = 7 ∧
  8*y1 + 27*y2 + 64*y3 + 125*y4 + 216*y5 = 100 ∧
  27*y1 + 64*y2 + 125*y3 + 216*y4 + 343*y5 = 1000 →
  64*y1 + 125*y2 + 216*y3 + 343*y4 + 512*y5 = -5999

theorem proof_theorem : ∀ (y1 y2 y3 y4 y5 : ℝ), proof_problem y1 y2 y3 y4 y5 :=
  by intros y1 y2 y3 y4 y5
     unfold proof_problem
     intro h
     sorry

end proof_theorem_l1022_102264


namespace part_a_impossibility_l1022_102297

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l1022_102297


namespace lava_lamp_probability_l1022_102203

/-- Ryan has 4 red lava lamps and 2 blue lava lamps; 
    he arranges them in a row on a shelf randomly, and then randomly turns 3 of them on. 
    Prove that the probability that the leftmost lamp is blue and off, 
    and the rightmost lamp is red and on is 2/25. -/
theorem lava_lamp_probability : 
  let total_arrangements := (Nat.choose 6 2) 
  let total_on := (Nat.choose 6 3)
  let favorable_arrangements := (Nat.choose 4 1)
  let favorable_on := (Nat.choose 4 2)
  let favorable_outcomes := 4 * 6
  let probability := (favorable_outcomes : ℚ) / (total_arrangements * total_on : ℚ)
  probability = 2 / 25 := 
by
  sorry

end lava_lamp_probability_l1022_102203


namespace percentage_of_loss_is_15_percent_l1022_102276

/-- 
Given:
  SP₁ = 168 -- Selling price when gaining 20%
  Gain = 20% 
  SP₂ = 119 -- Selling price when calculating loss

Prove:
  The percentage of loss when the article is sold for Rs. 119 is 15%
--/

noncomputable def percentage_loss (CP SP₂: ℝ) : ℝ :=
  ((CP - SP₂) / CP) * 100

theorem percentage_of_loss_is_15_percent (CP SP₂ SP₁: ℝ) (Gain: ℝ):
  CP = 140 ∧ SP₁ = 168 ∧ SP₂ = 119 ∧ Gain = 20 → percentage_loss CP SP₂ = 15 :=
by
  intro h
  sorry

end percentage_of_loss_is_15_percent_l1022_102276


namespace four_consecutive_integers_product_plus_one_is_square_l1022_102245

theorem four_consecutive_integers_product_plus_one_is_square (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n^2 + n - 1)^2 := by
  sorry

end four_consecutive_integers_product_plus_one_is_square_l1022_102245


namespace mark_paid_more_than_anne_by_three_dollars_l1022_102213

theorem mark_paid_more_than_anne_by_three_dollars :
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  mark_total - anne_total = 3 :=
by
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  sorry

end mark_paid_more_than_anne_by_three_dollars_l1022_102213


namespace integer_divisibility_l1022_102221

open Nat

theorem integer_divisibility {a b : ℕ} :
  (2 * b^2 + 1) ∣ (a^3 + 1) ↔ a = 2 * b^2 + 1 := sorry

end integer_divisibility_l1022_102221


namespace solve_quadratic1_solve_quadratic2_l1022_102257

theorem solve_quadratic1 (x : ℝ) :
  x^2 + 10 * x + 16 = 0 ↔ (x = -2 ∨ x = -8) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  x * (x + 4) = 8 * x + 12 ↔ (x = -2 ∨ x = 6) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l1022_102257


namespace zach_babysitting_hours_l1022_102296

theorem zach_babysitting_hours :
  ∀ (bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed : ℕ),
    bike_cost = 100 →
    weekly_allowance = 5 →
    mowing_pay = 10 →
    babysitting_rate = 7 →
    saved_amount = 65 →
    needed_additional_amount = 6 →
    saved_amount + weekly_allowance + mowing_pay + hours_needed * babysitting_rate = bike_cost - needed_additional_amount →
    hours_needed = 2 :=
by
  intros bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zach_babysitting_hours_l1022_102296


namespace intersection_of_A_and_B_l1022_102262

def A : Set ℤ := {-1, 0, 3, 5}
def B : Set ℤ := {x | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := 
by 
  sorry

end intersection_of_A_and_B_l1022_102262


namespace probability_yellow_face_l1022_102205

-- Define the total number of faces and the number of yellow faces on the die
def total_faces := 12
def yellow_faces := 4

-- Define the probability calculation
def probability_of_yellow := yellow_faces / total_faces

-- State the theorem
theorem probability_yellow_face : probability_of_yellow = 1 / 3 := by
  sorry

end probability_yellow_face_l1022_102205


namespace pairs_satisfying_int_l1022_102254

theorem pairs_satisfying_int (a b : ℕ) :
  ∃ n : ℕ, a = 2 * n^2 + 1 ∧ b = n ↔ (2 * a * b^2 + 1) ∣ (a^3 + 1) := by
  sorry

end pairs_satisfying_int_l1022_102254


namespace students_prefer_windows_l1022_102231

theorem students_prefer_windows (total_students students_prefer_mac equally_prefer_both no_preference : ℕ) 
  (h₁ : total_students = 210)
  (h₂ : students_prefer_mac = 60)
  (h₃ : equally_prefer_both = 20)
  (h₄ : no_preference = 90) :
  total_students - students_prefer_mac - equally_prefer_both - no_preference = 40 := 
  by
    -- Proof goes here
    sorry

end students_prefer_windows_l1022_102231


namespace pow_calculation_l1022_102238

-- We assume a is a non-zero real number or just a variable
variable (a : ℝ)

theorem pow_calculation : (2 * a^2)^3 = 8 * a^6 := 
by
  sorry

end pow_calculation_l1022_102238


namespace cricket_team_members_l1022_102268

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℚ) (wk_keeper_age : ℚ) 
  (avg_whole_team : ℚ) (avg_remaining_players : ℚ)
  (h1 : captain_age = 25)
  (h2 : wk_keeper_age = 28)
  (h3 : avg_whole_team = 22)
  (h4 : avg_remaining_players = 21)
  (h5 : 22 * n = 25 + 28 + 21 * (n - 2)) :
  n = 11 :=
by sorry

end cricket_team_members_l1022_102268


namespace perfect_square_expression_l1022_102277

theorem perfect_square_expression (n : ℕ) (h : 7 ≤ n) : ∃ k : ℤ, (n + 2) ^ 2 = k ^ 2 :=
by 
  sorry

end perfect_square_expression_l1022_102277


namespace triangle_inequality_l1022_102226

theorem triangle_inequality (S : Finset (ℕ × ℕ)) (m n : ℕ) (hS : S.card = m)
  (h_ab : ∀ (a b : ℕ), (a, b) ∈ S → (1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ a ≠ b)) :
  ∃ (t : Finset (ℕ × ℕ × ℕ)),
    (t.card ≥ (4 * m / (3 * n)) * (m - (n^2) / 4)) ∧
    ∀ (a b c : ℕ), (a, b, c) ∈ t → (a, b) ∈ S ∧ (b, c) ∈ S ∧ (c, a) ∈ S := by
  sorry

end triangle_inequality_l1022_102226


namespace divisibility_of_product_l1022_102202

theorem divisibility_of_product (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a ∣ b^3) (h2 : b ∣ c^3) (h3 : c ∣ a^3) : abc ∣ (a + b + c) ^ 13 := by
  sorry

end divisibility_of_product_l1022_102202


namespace apple_cost_l1022_102280

theorem apple_cost (l q : ℕ)
  (h1 : 30 * l + 6 * q = 366)
  (h2 : 15 * l = 150)
  (h3 : 30 * l + (333 - 30 * l) / q * q = 333) :
  30 + (333 - 30 * l) / q = 33 := 
sorry

end apple_cost_l1022_102280


namespace ions_electron_shell_structure_l1022_102289

theorem ions_electron_shell_structure
  (a b n m : ℤ) 
  (same_electron_shell_structure : a + n = b - m) :
  a + m = b - n :=
by
  sorry

end ions_electron_shell_structure_l1022_102289


namespace problem_proof_l1022_102230

theorem problem_proof :
  (3 ∣ 18) ∧
  (17 ∣ 187 ∧ ¬ (17 ∣ 52)) ∧
  ¬ ((24 ∣ 72) ∧ (24 ∣ 67)) ∧
  ¬ (13 ∣ 26 ∧ ¬ (13 ∣ 52)) ∧
  (8 ∣ 160) :=
by 
  sorry

end problem_proof_l1022_102230


namespace eggs_used_afternoon_l1022_102266

theorem eggs_used_afternoon (eggs_pumpkin eggs_apple eggs_cherry eggs_total : ℕ)
  (h_pumpkin : eggs_pumpkin = 816)
  (h_apple : eggs_apple = 384)
  (h_cherry : eggs_cherry = 120)
  (h_total : eggs_total = 1820) :
  eggs_total - (eggs_pumpkin + eggs_apple + eggs_cherry) = 500 :=
by
  sorry

end eggs_used_afternoon_l1022_102266


namespace petya_friends_l1022_102271

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l1022_102271


namespace find_AB_value_l1022_102282

theorem find_AB_value :
  ∃ A B : ℕ, (A + B = 5 ∧ (A - B) % 11 = 5 % 11) ∧
           990 * 991 * 992 * 993 = 966428 * 100000 + A * 9100 + B * 40 :=
sorry

end find_AB_value_l1022_102282


namespace coin_flip_sequences_l1022_102269

theorem coin_flip_sequences : 
  let flips := 10
  let choices := 2
  let total_sequences := choices ^ flips
  total_sequences = 1024 :=
by
  sorry

end coin_flip_sequences_l1022_102269


namespace find_two_numbers_l1022_102286

theorem find_two_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 5) (harmonic_mean : 2 * a * b / (a + b) = 5 / 3) :
  (a = (15 + Real.sqrt 145) / 4 ∧ b = (15 - Real.sqrt 145) / 4) ∨
  (a = (15 - Real.sqrt 145) / 4 ∧ b = (15 + Real.sqrt 145) / 4) :=
by
  sorry

end find_two_numbers_l1022_102286


namespace first_six_divisors_l1022_102273

theorem first_six_divisors (a b : ℤ) (h : 5 * b = 14 - 3 * a) : 
  ∃ n, n = 5 ∧ ∀ k ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), (3 * b + 18) % k = 0 ↔ k ∈ ({1, 2, 3, 5, 6} : Finset ℕ) :=
by
  sorry

end first_six_divisors_l1022_102273


namespace molecular_weight_2N_5O_l1022_102244

def molecular_weight (num_N num_O : ℕ) (atomic_weight_N atomic_weight_O : ℝ) : ℝ :=
  (num_N * atomic_weight_N) + (num_O * atomic_weight_O)

theorem molecular_weight_2N_5O :
  molecular_weight 2 5 14.01 16.00 = 108.02 :=
by
  -- proof goes here
  sorry

end molecular_weight_2N_5O_l1022_102244


namespace gain_percentage_is_30_l1022_102222

-- Definitions based on the conditions
def selling_price : ℕ := 195
def gain : ℕ := 45
def cost_price : ℕ := selling_price - gain
def gain_percentage : ℕ := (gain * 100) / cost_price

-- The statement to prove the gain percentage
theorem gain_percentage_is_30 : gain_percentage = 30 := 
by 
  -- Allow usage of fictive sorry for incomplete proof
  sorry

end gain_percentage_is_30_l1022_102222


namespace product_of_five_consecutive_not_square_l1022_102255

theorem product_of_five_consecutive_not_square (n : ℤ) :
  ¬ ∃ k : ℤ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2) :=
by
  sorry

end product_of_five_consecutive_not_square_l1022_102255


namespace vertex_of_parabola_l1022_102234

theorem vertex_of_parabola (c d : ℝ) (h : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  ∃ v : ℝ × ℝ, v = (1, 25) :=
sorry

end vertex_of_parabola_l1022_102234


namespace exists_divisible_by_2021_l1022_102243

def concatenated_number (n m : ℕ) : ℕ := 
  -- This function should concatenate the digits from n to m inclusively
  sorry

theorem exists_divisible_by_2021 : ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concatenated_number n m :=
by
  sorry

end exists_divisible_by_2021_l1022_102243


namespace sum_of_ten_distinct_numbers_lt_75_l1022_102263

theorem sum_of_ten_distinct_numbers_lt_75 :
  ∃ (S : Finset ℕ), S.card = 10 ∧
  (∃ (S_div_5 : Finset ℕ), S_div_5 ⊆ S ∧ S_div_5.card = 3 ∧ ∀ x ∈ S_div_5, 5 ∣ x) ∧
  (∃ (S_div_4 : Finset ℕ), S_div_4 ⊆ S ∧ S_div_4.card = 4 ∧ ∀ x ∈ S_div_4, 4 ∣ x) ∧
  S.sum id < 75 :=
by { 
  sorry 
}

end sum_of_ten_distinct_numbers_lt_75_l1022_102263


namespace find_number_l1022_102228

theorem find_number (n : ℕ) (h1 : n % 20 = 1) (h2 : n / 20 = 9) : n = 181 := 
by {
  -- proof not required
  sorry
}

end find_number_l1022_102228


namespace balloons_remaining_each_friend_l1022_102265

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end balloons_remaining_each_friend_l1022_102265


namespace sum_of_interior_angles_n_plus_2_l1022_102237

-- Define the sum of the interior angles formula for a convex polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the degree measure of the sum of the interior angles of a convex polygon with n sides being 1800
def sum_of_n_sides_is_1800 (n : ℕ) : Prop := sum_of_interior_angles n = 1800

-- Translate the proof problem as a theorem statement in Lean
theorem sum_of_interior_angles_n_plus_2 (n : ℕ) (h: sum_of_n_sides_is_1800 n) : 
  sum_of_interior_angles (n + 2) = 2160 :=
sorry

end sum_of_interior_angles_n_plus_2_l1022_102237


namespace boat_stream_speed_l1022_102299

theorem boat_stream_speed :
  ∀ (v : ℝ), (∀ (downstream_speed boat_speed : ℝ), boat_speed = 22 ∧ downstream_speed = 54/2 ∧ downstream_speed = boat_speed + v) -> v = 5 :=
by
  sorry

end boat_stream_speed_l1022_102299


namespace inequality_triangle_areas_l1022_102250

theorem inequality_triangle_areas (a b c α β γ : ℝ) (hα : α = 2 * Real.sqrt (b * c)) (hβ : β = 2 * Real.sqrt (a * c)) (hγ : γ = 2 * Real.sqrt (a * b)) : 
  a / α + b / β + c / γ ≥ 3 / 2 := 
by
  sorry

end inequality_triangle_areas_l1022_102250


namespace david_started_with_15_samsung_phones_l1022_102225

-- Definitions
def SamsungPhonesAtEnd : ℕ := 10 -- S_e
def IPhonesAtEnd : ℕ := 5 -- I_e
def SamsungPhonesThrownOut : ℕ := 2 -- S_d
def IPhonesThrownOut : ℕ := 1 -- I_d
def TotalPhonesSold : ℕ := 4 -- C

-- Number of iPhones sold
def IPhonesSold : ℕ := IPhonesThrownOut

-- Assume: The remaining phones sold are Samsung phones
def SamsungPhonesSold : ℕ := TotalPhonesSold - IPhonesSold

-- Calculate the number of Samsung phones David started the day with
def SamsungPhonesAtStart : ℕ := SamsungPhonesAtEnd + SamsungPhonesThrownOut + SamsungPhonesSold

-- Statement
theorem david_started_with_15_samsung_phones : SamsungPhonesAtStart = 15 := by
  sorry

end david_started_with_15_samsung_phones_l1022_102225


namespace eccentricity_of_ellipse_l1022_102279

theorem eccentricity_of_ellipse {a b c e : ℝ} 
  (h1 : b^2 = 3) 
  (h2 : c = 1 / 4)
  (h3 : a^2 = b^2 + c^2)
  (h4 : a = 7 / 4) 
  : e = c / a → e = 1 / 7 :=
by 
  intros
  sorry

end eccentricity_of_ellipse_l1022_102279


namespace meaningful_domain_of_function_l1022_102259

theorem meaningful_domain_of_function : ∀ x : ℝ, (∃ y : ℝ, y = 3 / Real.sqrt (x - 2)) → x > 2 :=
by
  intros x h
  sorry

end meaningful_domain_of_function_l1022_102259


namespace negation_of_proposition_l1022_102252

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by sorry

end negation_of_proposition_l1022_102252


namespace landscaping_charges_l1022_102218

theorem landscaping_charges
    (x : ℕ)
    (h : 63 * x + 9 * 11 + 10 * 9 = 567) :
  x = 6 :=
by
  sorry

end landscaping_charges_l1022_102218


namespace inequality_holds_for_all_x_l1022_102288

theorem inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ (-2 < k ∧ k < 6) :=
by
  sorry

end inequality_holds_for_all_x_l1022_102288


namespace find_a_l1022_102214

noncomputable def f (a x : ℝ) : ℝ := a^x - 4 * a + 3

theorem find_a (H : ∃ (a : ℝ), ∃ (x y : ℝ), f a x = y ∧ f y x = a ∧ x = 2 ∧ y = -1): ∃ a : ℝ, a = 2 :=
by
  obtain ⟨a, x, y, hx, hy, hx2, hy1⟩ := H
  --skipped proof
  sorry

end find_a_l1022_102214


namespace flip_ratio_l1022_102233

theorem flip_ratio (jen_triple_flips tyler_double_flips : ℕ)
  (hjen : jen_triple_flips = 16)
  (htyler : tyler_double_flips = 12)
  : 2 * tyler_double_flips / 3 * jen_triple_flips = 1 / 2 := 
by
  rw [hjen, htyler]
  norm_num
  sorry

end flip_ratio_l1022_102233


namespace sufficient_not_necessary_condition_l1022_102290

theorem sufficient_not_necessary_condition
  (x : ℝ) : 
  x^2 - 4*x - 5 > 0 → (x > 5 ∨ x < -1) ∧ (x > 5 → x^2 - 4*x - 5 > 0) ∧ ¬(x^2 - 4*x - 5 > 0 → x > 5) := 
sorry

end sufficient_not_necessary_condition_l1022_102290


namespace determine_range_of_a_l1022_102206

noncomputable def f (x a : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

noncomputable def g (x a : ℝ) : ℝ := f x a - 2*x

theorem determine_range_of_a (a : ℝ) :
  (∀ x, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) →
  (-1 ≤ a ∧ a < 2) :=
by
  intro h
  sorry

end determine_range_of_a_l1022_102206


namespace pamphlet_cost_l1022_102256

theorem pamphlet_cost (p : ℝ) 
  (h1 : 9 * p < 10)
  (h2 : 10 * p > 11) : p = 1.11 :=
sorry

end pamphlet_cost_l1022_102256


namespace no_three_digit_number_exists_l1022_102283

theorem no_three_digit_number_exists (a b c : ℕ) (h₁ : 0 ≤ a ∧ a < 10) (h₂ : 0 ≤ b ∧ b < 10) (h₃ : 0 ≤ c ∧ c < 10) (h₄ : a ≠ 0) :
  ¬ ∃ k : ℕ, k^2 = 99 * (a - c) :=
by
  sorry

end no_three_digit_number_exists_l1022_102283


namespace min_possible_value_of_coefficient_x_l1022_102281

theorem min_possible_value_of_coefficient_x 
  (c d : ℤ) 
  (h1 : c * d = 15) 
  (h2 : ∃ (C : ℤ), C = c + d) 
  (h3 : c ≠ d ∧ c ≠ 34 ∧ d ≠ 34) :
  (∃ (C : ℤ), C = c + d ∧ C = 34) :=
sorry

end min_possible_value_of_coefficient_x_l1022_102281


namespace smallest_number_of_students_l1022_102258

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l1022_102258


namespace value_of_expression_l1022_102270

theorem value_of_expression (x : ℕ) (h : x = 2) : x + x * x^x = 10 := by
  rw [h] -- Substituting x = 2
  sorry

end value_of_expression_l1022_102270


namespace value_of_r_squared_plus_s_squared_l1022_102201

theorem value_of_r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 24) (h2 : r + s = 10) :
  r^2 + s^2 = 52 :=
sorry

end value_of_r_squared_plus_s_squared_l1022_102201


namespace right_handed_total_l1022_102224

theorem right_handed_total
  (total_players : ℕ)
  (throwers : ℕ)
  (left_handed_non_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (non_throwers : ℕ)
  (right_handed_non_throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = non_throwers / 3 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  right_handed_throwers = throwers →
  right_handed_throwers + right_handed_non_throwers = 56 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end right_handed_total_l1022_102224


namespace problem1_problem2a_problem2b_l1022_102274

noncomputable def x : ℝ := Real.sqrt 6 - Real.sqrt 2
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem1 : x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5) = 1 - 2 * Real.sqrt 3 := 
by
  sorry

theorem problem2a : a - b = 2 * Real.sqrt 2 := 
by 
  sorry

theorem problem2b : a^2 - 2 * a * b + b^2 = 8 := 
by 
  sorry

end problem1_problem2a_problem2b_l1022_102274


namespace moles_of_Cu_CN_2_is_1_l1022_102239

def moles_of_HCN : Nat := 2
def moles_of_CuSO4 : Nat := 1
def moles_of_Cu_CN_2_formed (hcn : Nat) (cuso4 : Nat) : Nat :=
  if hcn = 2 ∧ cuso4 = 1 then 1 else 0

theorem moles_of_Cu_CN_2_is_1 : moles_of_Cu_CN_2_formed moles_of_HCN moles_of_CuSO4 = 1 :=
by
  sorry

end moles_of_Cu_CN_2_is_1_l1022_102239


namespace inequality_sol_set_a_eq_2_inequality_sol_set_general_l1022_102246

theorem inequality_sol_set_a_eq_2 :
  ∀ x : ℝ, (x^2 - x + 2 - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem inequality_sol_set_general (a : ℝ) :
  (∀ x : ℝ, (x^2 - x + a - a^2 ≤ 0) ↔
    (if a < 1/2 then a ≤ x ∧ x ≤ 1 - a
    else if a > 1/2 then 1 - a ≤ x ∧ x ≤ a
    else x = 1/2)) :=
by sorry

end inequality_sol_set_a_eq_2_inequality_sol_set_general_l1022_102246


namespace solution_set_of_inequality_l1022_102293

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l1022_102293


namespace percentage_of_additional_money_is_10_l1022_102220

-- Define the conditions
def months := 11
def payment_per_month := 15
def total_borrowed := 150

-- Define the function to calculate the total amount paid
def total_paid (months payment_per_month : ℕ) : ℕ :=
  months * payment_per_month

-- Define the function to calculate the additional amount paid
def additional_paid (total_paid total_borrowed : ℕ) : ℕ :=
  total_paid - total_borrowed

-- Define the function to calculate the percentage of the additional amount
def percentage_additional (additional_paid total_borrowed : ℕ) : ℕ :=
  (additional_paid * 100) / total_borrowed

-- State the theorem to prove the percentage of the additional money is 10%
theorem percentage_of_additional_money_is_10 :
  percentage_additional (additional_paid (total_paid months payment_per_month) total_borrowed) total_borrowed = 10 :=
by
  sorry

end percentage_of_additional_money_is_10_l1022_102220
