import Mathlib

namespace sum_n_k_of_binomial_coefficient_ratio_l623_62380

theorem sum_n_k_of_binomial_coefficient_ratio :
  ∃ (n k : ℕ), (n = (7 * k + 5) / 2) ∧ (2 * (n - k) = 5 * (k + 1)) ∧ 
    ((k % 2 = 1) ∧ (n + k = 7 ∨ n + k = 16)) ∧ (23 = 7 + 16) :=
by
  sorry

end sum_n_k_of_binomial_coefficient_ratio_l623_62380


namespace total_chocolates_distributed_l623_62357

theorem total_chocolates_distributed 
  (boys girls : ℕ)
  (chocolates_per_boy chocolates_per_girl : ℕ)
  (h_boys : boys = 60)
  (h_girls : girls = 60)
  (h_chocolates_per_boy : chocolates_per_boy = 2)
  (h_chocolates_per_girl : chocolates_per_girl = 3) : 
  boys * chocolates_per_boy + girls * chocolates_per_girl = 300 :=
by {
  sorry
}

end total_chocolates_distributed_l623_62357


namespace relationship_between_abc_l623_62341

theorem relationship_between_abc 
  (a b c : ℝ) 
  (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) 
  (ha : Real.exp a = 9 * a * Real.log 11)
  (hb : Real.exp b = 10 * b * Real.log 10)
  (hc : Real.exp c = 11 * c * Real.log 9) : 
  a < b ∧ b < c :=
sorry

end relationship_between_abc_l623_62341


namespace arithmetic_geometric_inequality_l623_62330

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + n * d

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ := b1 * r^n

theorem arithmetic_geometric_inequality
  (a1 b1 : ℝ) (d r : ℝ) (n : ℕ)
  (h_pos : 0 < a1) 
  (ha1_eq_b1 : a1 = b1) 
  (h_eq_2np1 : arithmetic_sequence a1 d (2*n+1) = geometric_sequence b1 r (2*n+1)) :
  arithmetic_sequence a1 d (n+1) ≥ geometric_sequence b1 r (n+1) :=
sorry

end arithmetic_geometric_inequality_l623_62330


namespace fraction_zero_iff_l623_62318

theorem fraction_zero_iff (x : ℝ) (h₁ : (x - 1) / (2 * x - 4) = 0) (h₂ : 2 * x - 4 ≠ 0) : x = 1 := sorry

end fraction_zero_iff_l623_62318


namespace find_y_l623_62337

theorem find_y (y : ℕ) (h : (2 * y) / 5 = 10) : y = 25 :=
sorry

end find_y_l623_62337


namespace triangle_area_base_10_height_10_l623_62305

theorem triangle_area_base_10_height_10 :
  let base := 10
  let height := 10
  (base * height) / 2 = 50 := by
  sorry

end triangle_area_base_10_height_10_l623_62305


namespace smallest_a_l623_62379

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_a (a : ℕ) (h1 : 5880 = 2^3 * 3^1 * 5^1 * 7^2)
                    (h2 : ∀ b : ℕ, b < a → ¬ is_perfect_square (5880 * b))
                    : a = 15 :=
by
  sorry

end smallest_a_l623_62379


namespace scientific_notation_of_74850000_l623_62383

theorem scientific_notation_of_74850000 : 74850000 = 7.485 * 10^7 :=
  by
  sorry

end scientific_notation_of_74850000_l623_62383


namespace monotonic_increasing_iff_monotonic_decreasing_on_interval_l623_62370

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 - a * x - 1

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ a ≤ 0 :=
by 
  sorry

theorem monotonic_decreasing_on_interval (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → ∀ y : ℝ, -1 < y ∧ y < 1 → x < y → f y a < f x a) ↔ 3 ≤ a :=
by 
  sorry

end monotonic_increasing_iff_monotonic_decreasing_on_interval_l623_62370


namespace find_a_and_max_value_l623_62311

noncomputable def f (x a : ℝ) := 2 * x^3 - 6 * x^2 + a

theorem find_a_and_max_value :
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≥ 0) ∧ (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≤ 3)) :=
by
  sorry

end find_a_and_max_value_l623_62311


namespace problem1_problem2_problem3_l623_62391

noncomputable def f (b a : ℝ) (x : ℝ) := (b - 2^x) / (2^x + a) 

-- (1) Prove values of a and b
theorem problem1 (a b : ℝ) : 
  (f b a 0 = 0) ∧ (f b a (-1) = -f b a 1) → (a = 1 ∧ b = 1) :=
sorry

-- (2) Prove f is decreasing function
theorem problem2 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f b a x₁ - f b a x₂ > 0 :=
sorry

-- (3) Find range of k such that inequality always holds
theorem problem3 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) (k : ℝ) : 
  (∀ t : ℝ, f b a (t^2 - 2*t) + f b a (2*t^2 - k) < 0) → k < -(1/3) :=
sorry

end problem1_problem2_problem3_l623_62391


namespace remainder_of_83_div_9_l623_62322

theorem remainder_of_83_div_9 : ∃ r : ℕ, 83 = 9 * 9 + r ∧ r = 2 :=
by {
  sorry
}

end remainder_of_83_div_9_l623_62322


namespace magician_earnings_at_least_l623_62396

def magician_starting_decks := 15
def magician_remaining_decks := 3
def decks_sold := magician_starting_decks - magician_remaining_decks
def standard_price_per_deck := 3
def discount := 1
def discounted_price_per_deck := standard_price_per_deck - discount
def min_earnings := decks_sold * discounted_price_per_deck

theorem magician_earnings_at_least :
  min_earnings ≥ 24 :=
by sorry

end magician_earnings_at_least_l623_62396


namespace smallest_n_for_sum_is_24_l623_62366

theorem smallest_n_for_sum_is_24 :
  ∃ (n : ℕ), (0 < n) ∧ 
    (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
    ∀ (m : ℕ), ((0 < m) ∧ 
                (∃ (k' : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = k') → n ≤ m) := sorry

end smallest_n_for_sum_is_24_l623_62366


namespace jane_crayon_count_l623_62332

def billy_crayons : ℝ := 62.0
def total_crayons : ℝ := 114
def jane_crayons : ℝ := total_crayons - billy_crayons

theorem jane_crayon_count : jane_crayons = 52 := by
  unfold jane_crayons
  show total_crayons - billy_crayons = 52
  sorry

end jane_crayon_count_l623_62332


namespace arithmetic_geometric_value_l623_62326

-- Definitions and annotations
variables {a1 a2 b1 b2 : ℝ}
variable {d : ℝ} -- common difference for the arithmetic sequence
variable {q : ℝ} -- common ratio for the geometric sequence

-- Assuming input values for the initial elements of the sequences
axiom h1 : -9 = -9
axiom h2 : -9 + 3 * d = -1
axiom h3 : b1 = -9 * q
axiom h4 : b2 = -9 * q^2

-- The desired equality to prove
theorem arithmetic_geometric_value :
  b2 * (a2 - a1) = -8 :=
sorry

end arithmetic_geometric_value_l623_62326


namespace find_n_l623_62313

theorem find_n (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) = 14) : n = 2 :=
sorry

end find_n_l623_62313


namespace colten_chickens_l623_62350

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l623_62350


namespace minimum_students_in_class_l623_62310

def min_number_of_students (b g : ℕ) : ℕ :=
  b + g

theorem minimum_students_in_class
  (b g : ℕ)
  (h1 : b = 2 * g / 3)
  (h2 : ∃ k : ℕ, g = 3 * k)
  (h3 : ∃ k : ℕ, 1 / 2 < (2 / 3) * g / b) :
  min_number_of_students b g = 5 :=
sorry

end minimum_students_in_class_l623_62310


namespace evaluate_expression_l623_62397

theorem evaluate_expression (x : Int) (h : x = -2023) : abs (abs (abs x - x) + abs x) + x = 4046 :=
by
  rw [h]
  sorry

end evaluate_expression_l623_62397


namespace find_x_l623_62382

-- conditions
variable (k : ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- proportional relationship
def proportional_relationship (k x y z : ℝ) : Prop := 
  x = (k * y^2) / z

-- initial conditions
def initial_conditions (k : ℝ) : Prop := 
  proportional_relationship k 6 1 3

-- prove x = 24 when y = 2 and z = 3 under given conditions
theorem find_x (k : ℝ) (h : initial_conditions k) : 
  proportional_relationship k 24 2 3 :=
sorry

end find_x_l623_62382


namespace impossible_to_color_25_cells_l623_62301

theorem impossible_to_color_25_cells :
  ¬ ∃ (n : ℕ) (n_k : ℕ → ℕ), n = 25 ∧ (∀ k, k > 0 → k < 5 → (k % 2 = 1 → ∃ c : ℕ, n_k c = k)) :=
by
  sorry

end impossible_to_color_25_cells_l623_62301


namespace line_eq_l623_62342

-- Conditions
def circle_eq (x y : ℝ) (a : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 5 - a

def midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  2*xm = x1 + x2 ∧ 2*ym = y1 + y2

-- Theorem statement
theorem line_eq (a : ℝ) (h : a < 3) :
  circle_eq 0 1 a →
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

end line_eq_l623_62342


namespace buildings_subset_count_l623_62324

theorem buildings_subset_count :
  let buildings := Finset.range (16 + 1) \ {0}
  ∃ S ⊆ buildings, ∀ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S → ∃ k, (b - a = 2 * k + 1) ∨ (a - b = 2 * k + 1) ∧ Finset.card S = 510 :=
sorry

end buildings_subset_count_l623_62324


namespace total_cards_l623_62369

theorem total_cards (Brenda_card Janet_card Mara_card Michelle_card : ℕ)
  (h1 : Janet_card = Brenda_card + 9)
  (h2 : Mara_card = 7 * Janet_card / 4)
  (h3 : Michelle_card = 4 * Mara_card / 5)
  (h4 : Mara_card = 210 - 60) :
  Janet_card + Brenda_card + Mara_card + Michelle_card = 432 :=
by
  sorry

end total_cards_l623_62369


namespace complete_the_square_l623_62307

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end complete_the_square_l623_62307


namespace problem_l623_62327

theorem problem (x : ℝ) (h : 15 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 17 := 
by sorry

end problem_l623_62327


namespace minimum_value_fraction_l623_62358

theorem minimum_value_fraction (m n : ℝ) (h_line : 2 * m * 2 + n * 2 - 4 = 0) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (m + n / 2 = 1) -> ∃ (m n : ℝ), (m > 0 ∧ n > 0) ∧ (3 + 2 * Real.sqrt 2 ≤ (1 / m + 4 / n)) :=
by
  sorry

end minimum_value_fraction_l623_62358


namespace student_weekly_allowance_l623_62394

theorem student_weekly_allowance (A : ℝ) 
  (h1 : ∃ spent_arcade, spent_arcade = (3 / 5) * A)
  (h2 : ∃ spent_toy, spent_toy = (1 / 3) * ((2 / 5) * A))
  (h3 : ∃ spent_candy, spent_candy = 0.60)
  (h4 : ∃ remaining_after_toy, remaining_after_toy = ((6 / 15) * A - (2 / 15) * A))
  (h5 : remaining_after_toy = 0.60) : 
  A = 2.25 := by
  sorry

end student_weekly_allowance_l623_62394


namespace foreman_can_establish_corr_foreman_cannot_with_less_l623_62352

-- Define the given conditions:
def num_rooms (n : ℕ) := 2^n
def num_checks (n : ℕ) := 2 * n

-- Part (a)
theorem foreman_can_establish_corr (n : ℕ) : 
  ∃ (c : ℕ), c = num_checks n ∧ (c ≥ 2 * n) :=
by
  sorry

-- Part (b)
theorem foreman_cannot_with_less (n : ℕ) : 
  ¬ (∃ (c : ℕ), c = 2 * n - 1 ∧ (c < 2 * n)) :=
by
  sorry

end foreman_can_establish_corr_foreman_cannot_with_less_l623_62352


namespace bill_toys_l623_62300

variable (B H : ℕ)

theorem bill_toys (h1 : H = B / 2 + 9) (h2 : B + H = 99) : B = 60 := by
  sorry

end bill_toys_l623_62300


namespace find_m_l623_62338

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A based on the condition in the problem
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5 * x + m = 0}

-- Define the complement of A in the universal set U
def complementA (m : ℕ) : Set ℕ := U \ A m

-- Given condition that the complement of A in U is {2, 3}
def complementA_condition : Set ℕ := {2, 3}

-- The proof problem statement: Prove that m = 4 given the conditions
theorem find_m (m : ℕ) (h : complementA m = complementA_condition) : m = 4 :=
sorry

end find_m_l623_62338


namespace am_gm_inequality_l623_62390

theorem am_gm_inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := 
by
  sorry

end am_gm_inequality_l623_62390


namespace C_pow_50_l623_62371

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_pow_50 :
  C ^ 50 = !![-299, -100; 800, 249] := by
  sorry

end C_pow_50_l623_62371


namespace suki_bags_l623_62339

theorem suki_bags (bag_weight_suki : ℕ) (bag_weight_jimmy : ℕ) (containers : ℕ) 
  (container_weight : ℕ) (num_bags_jimmy : ℝ) (num_containers : ℕ)
  (h1 : bag_weight_suki = 22) 
  (h2 : bag_weight_jimmy = 18) 
  (h3 : container_weight = 8) 
  (h4 : num_bags_jimmy = 4.5)
  (h5 : num_containers = 28) : 
  6 = ⌊(num_containers * container_weight - num_bags_jimmy * bag_weight_jimmy) / bag_weight_suki⌋ :=
by
  sorry

end suki_bags_l623_62339


namespace base_6_to_10_conversion_l623_62364

theorem base_6_to_10_conversion : 
  ∀ (n : ℕ), n = 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0 → n = 1295 :=
by
  intro n h
  sorry

end base_6_to_10_conversion_l623_62364


namespace complement_union_l623_62355

open Finset

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 4}
def N : Finset ℕ := {2, 4}

theorem complement_union :
  U \ (M ∪ N) = {1, 3} := by
  sorry

end complement_union_l623_62355


namespace sum_of_remainders_l623_62388

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  sorry

end sum_of_remainders_l623_62388


namespace average_temperature_l623_62378

theorem average_temperature (T_tue T_wed T_thu : ℝ) 
  (h1 : (42 + T_tue + T_wed + T_thu) / 4 = 48)
  (T_fri : ℝ := 34) :
  ((T_tue + T_wed + T_thu + T_fri) / 4 = 46) :=
by
  sorry

end average_temperature_l623_62378


namespace derek_initial_lunch_cost_l623_62308

-- Definitions based on conditions
def derek_initial_money : ℕ := 40
def derek_dad_lunch_cost : ℕ := 11
def derek_more_lunch_cost : ℕ := 5
def dave_initial_money : ℕ := 50
def dave_mom_lunch_cost : ℕ := 7
def dave_difference : ℕ := 33

-- Variable X to represent Derek's initial lunch cost
variable (X : ℕ)

-- Definitions based on conditions
def derek_total_spending (X : ℕ) := X + derek_dad_lunch_cost + derek_more_lunch_cost
def derek_remaining_money (X : ℕ) := derek_initial_money - derek_total_spending X
def dave_remaining_money := dave_initial_money - dave_mom_lunch_cost

-- The main theorem to prove Derek spent $14 initially
theorem derek_initial_lunch_cost (h : dave_remaining_money = derek_remaining_money X + dave_difference) : X = 14 := by
  sorry

end derek_initial_lunch_cost_l623_62308


namespace cost_50_jasmines_discounted_l623_62317

variable (cost_per_8_jasmines : ℝ) (num_jasmines : ℕ) (discount : ℝ)
variable (proportional : Prop) (c_50_jasmines : ℝ)

-- Given the cost of a bouquet with 8 jasmines
def cost_of_8_jasmines : ℝ := 24

-- Given the price is directly proportional to the number of jasmines
def price_proportional := ∀ (n : ℕ), num_jasmines = 8 → proportional

-- Given the bouquet with 50 jasmines
def num_jasmines_50 : ℕ := 50

-- Applying a 10% discount
def ten_percent_discount : ℝ := 0.9

-- Prove the cost of the bouquet with 50 jasmines after a 10% discount
theorem cost_50_jasmines_discounted :
  proportional ∧ (c_50_jasmines = (cost_of_8_jasmines / 8) * num_jasmines_50) →
  (c_50_jasmines * ten_percent_discount) = 135 :=
by
  sorry

end cost_50_jasmines_discounted_l623_62317


namespace equal_distribution_l623_62389

theorem equal_distribution (k : ℤ) : ∃ n : ℤ, n = 81 + 95 * k ∧ ∃ b : ℤ, (19 + 6 * n) = 95 * b :=
by
  -- to be proved
  sorry

end equal_distribution_l623_62389


namespace inequality_nonneg_ab_l623_62384

theorem inequality_nonneg_ab (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 :=
by
  sorry

end inequality_nonneg_ab_l623_62384


namespace five_ab4_is_perfect_square_l623_62325

theorem five_ab4_is_perfect_square (a b : ℕ) (h : 5000 ≤ 5000 + 100 * a + 10 * b + 4 ∧ 5000 + 100 * a + 10 * b + 4 ≤ 5999) :
    ∃ n, n^2 = 5000 + 100 * a + 10 * b + 4 → a + b = 9 :=
by
  sorry

end five_ab4_is_perfect_square_l623_62325


namespace other_number_is_twelve_l623_62368

variable (x certain_number : ℕ)
variable (h1: certain_number = 60)
variable (h2: certain_number = 5 * x)

theorem other_number_is_twelve :
  x = 12 :=
by
  sorry

end other_number_is_twelve_l623_62368


namespace min_value_x_plus_2y_l623_62386

theorem min_value_x_plus_2y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 * y - x * y = 0) : x + 2 * y = 8 := 
by
  sorry

end min_value_x_plus_2y_l623_62386


namespace prime_quadratic_root_range_l623_62392

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end prime_quadratic_root_range_l623_62392


namespace positive_integer_conditions_l623_62315

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) : 
  (∃ k : ℕ, k > 0 ∧ 4 * p + 28 = k * (3 * p - 7)) ↔ (p = 6 ∨ p = 28) :=
by
  sorry

end positive_integer_conditions_l623_62315


namespace linear_elimination_l623_62309

theorem linear_elimination (a b : ℤ) (x y : ℤ) :
  (a = 2) ∧ (b = -5) → 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = 0) → 
  (10 * x - 4 * y + -10 * x - 15 * y = 8 + -45) :=
by
  sorry

end linear_elimination_l623_62309


namespace total_length_proof_l623_62304

def length_of_first_tape : ℝ := 25
def overlap : ℝ := 3
def number_of_tapes : ℝ := 64

def total_tape_length : ℝ :=
  let effective_length_per_subsequent_tape := length_of_first_tape - overlap
  let length_of_remaining_tapes := effective_length_per_subsequent_tape * (number_of_tapes - 1)
  length_of_first_tape + length_of_remaining_tapes

theorem total_length_proof : total_tape_length = 1411 := by
  sorry

end total_length_proof_l623_62304


namespace test_scores_order_l623_62345

def kaleana_score : ℕ := 75

variable (M Q S : ℕ)

-- Assuming conditions from the problem
axiom h1 : Q = kaleana_score
axiom h2 : M < max Q S
axiom h3 : S > min Q M
axiom h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S

-- Theorem statement
theorem test_scores_order (M Q S : ℕ) (h1 : Q = kaleana_score) (h2 : M < max Q S) (h3 : S > min Q M) (h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S) :
  M < Q ∧ Q < S :=
sorry

end test_scores_order_l623_62345


namespace k_value_and_set_exists_l623_62333

theorem k_value_and_set_exists
  (x1 x2 x3 x4 : ℚ)
  (h1 : (x1 + x2) / (x3 + x4) = -1)
  (h2 : (x1 + x3) / (x2 + x4) = -1)
  (h3 : (x1 + x4) / (x2 + x3) = -1)
  (hne : x1 ≠ x2 ∨ x1 ≠ x3 ∨ x1 ≠ x4 ∨ x2 ≠ x3 ∨ x2 ≠ x4 ∨ x3 ≠ x4) :
  ∃ (A B C : ℚ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ x1 = A ∧ x2 = B ∧ x3 = C ∧ x4 = -A - B - C := 
sorry

end k_value_and_set_exists_l623_62333


namespace change_from_15_dollars_l623_62359

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end change_from_15_dollars_l623_62359


namespace Xiaoli_estimate_is_larger_l623_62346

variables {x y x' y' : ℝ}

theorem Xiaoli_estimate_is_larger (h1 : x > y) (h2 : y > 0) (h3 : x' = 1.01 * x) (h4 : y' = 0.99 * y) : x' - y' > x - y :=
by sorry

end Xiaoli_estimate_is_larger_l623_62346


namespace hypotenuse_length_l623_62302

variable (a b c : ℝ)

-- Given conditions
theorem hypotenuse_length (h1 : b = 3 * a) 
                          (h2 : a^2 + b^2 + c^2 = 500) 
                          (h3 : c^2 = a^2 + b^2) : 
                          c = 5 * Real.sqrt 10 := 
by 
  sorry

end hypotenuse_length_l623_62302


namespace linear_function_graph_not_in_second_quadrant_l623_62314

open Real

theorem linear_function_graph_not_in_second_quadrant 
  (k b : ℝ) (h1 : k > 0) (h2 : b < 0) :
  ¬ ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b := 
sorry

end linear_function_graph_not_in_second_quadrant_l623_62314


namespace min_rectilinear_distance_to_parabola_l623_62361

theorem min_rectilinear_distance_to_parabola :
  ∃ t : ℝ, ∀ t', (|t' + 1| + t'^2) ≥ (|t + 1| + t^2) ∧ (|t + 1| + t^2) = 3 / 4 := sorry

end min_rectilinear_distance_to_parabola_l623_62361


namespace geometric_seq_a7_l623_62363

theorem geometric_seq_a7 (a : ℕ → ℝ) (r : ℝ) (h1 : a 3 = 16) (h2 : a 5 = 4) (h_geom : ∀ n, a (n + 1) = a n * r) : a 7 = 1 := by
  sorry

end geometric_seq_a7_l623_62363


namespace jason_additional_manager_months_l623_62376

def additional_manager_months (bartender_years manager_years total_exp_months : ℕ) : ℕ :=
  let bartender_months := bartender_years * 12
  let manager_months := manager_years * 12
  total_exp_months - (bartender_months + manager_months)

theorem jason_additional_manager_months : 
  additional_manager_months 9 3 150 = 6 := 
by 
  sorry

end jason_additional_manager_months_l623_62376


namespace complementary_angles_ratio_l623_62372

theorem complementary_angles_ratio (x : ℝ) (hx : 5 * x = 90) : abs (4 * x - x) = 54 :=
by
  have h₁ : x = 18 := by 
    linarith [hx]
  rw [h₁]
  norm_num

end complementary_angles_ratio_l623_62372


namespace denomination_other_currency_notes_l623_62365

noncomputable def denomination_proof : Prop :=
  ∃ D x y : ℕ, 
  (x + y = 85) ∧
  (100 * x + D * y = 5000) ∧
  (D * y = 3500) ∧
  (D = 50)

theorem denomination_other_currency_notes :
  denomination_proof :=
sorry

end denomination_other_currency_notes_l623_62365


namespace meals_distinct_pairs_l623_62398

theorem meals_distinct_pairs :
  let entrees := 4
  let drinks := 3
  let desserts := 3
  let total_meals := entrees * drinks * desserts
  total_meals * (total_meals - 1) = 1260 :=
by 
  sorry

end meals_distinct_pairs_l623_62398


namespace compare_decimal_to_fraction_l623_62373

theorem compare_decimal_to_fraction : (0.650 - (1 / 8) = 0.525) :=
by
  /- We need to prove that 0.650 - 1/8 = 0.525 -/
  sorry

end compare_decimal_to_fraction_l623_62373


namespace tan_of_angle_l623_62354

open Real

-- Given conditions in the problem
variables {α : ℝ}

-- Define the given conditions
def sinα_condition (α : ℝ) : Prop := sin α = 3 / 5
def α_in_quadrant_2 (α : ℝ) : Prop := π / 2 < α ∧ α < π

-- Define the Lean statement
theorem tan_of_angle {α : ℝ} (h1 : sinα_condition α) (h2 : α_in_quadrant_2 α) :
  tan α = -3 / 4 :=
sorry

end tan_of_angle_l623_62354


namespace price_of_first_variety_l623_62340

theorem price_of_first_variety
  (p2 : ℝ) (p3 : ℝ) (r : ℝ) (w : ℝ)
  (h1 : p2 = 135)
  (h2 : p3 = 177.5)
  (h3 : r = 154)
  (h4 : w = 4) :
  ∃ p1 : ℝ, 1 * p1 + 1 * p2 + 2 * p3 = w * r ∧ p1 = 126 :=
by {
  sorry
}

end price_of_first_variety_l623_62340


namespace expectation_of_transformed_binomial_l623_62320

def binomial_expectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

def linear_property_of_expectation (a b : ℚ) (E_ξ : ℚ) : ℚ :=
  a * E_ξ + b

theorem expectation_of_transformed_binomial (ξ : ℚ) :
  ξ = binomial_expectation 5 (2/5) →
  linear_property_of_expectation 5 2 ξ = 12 :=
by
  intros h
  rw [h]
  unfold linear_property_of_expectation binomial_expectation
  sorry

end expectation_of_transformed_binomial_l623_62320


namespace absents_probability_is_correct_l623_62329

-- Conditions
def probability_absent := 1 / 10
def probability_present := 9 / 10

-- Calculation of combined probability
def combined_probability : ℚ :=
  3 * (probability_absent * probability_absent * probability_present)

-- Conversion to percentage
def percentage_probability : ℚ :=
  combined_probability * 100

-- Theorem statement
theorem absents_probability_is_correct :
  percentage_probability = 2.7 := 
sorry

end absents_probability_is_correct_l623_62329


namespace baseball_game_earnings_l623_62393

theorem baseball_game_earnings
  (S : ℝ) (W : ℝ)
  (h1 : S = 2662.50)
  (h2 : W + S = 5182.50) :
  S - W = 142.50 :=
by
  sorry

end baseball_game_earnings_l623_62393


namespace sum_of_largest_and_smallest_four_digit_numbers_is_11990_l623_62395

theorem sum_of_largest_and_smallest_four_digit_numbers_is_11990 (A B C D : ℕ) 
    (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D)
    (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
    (h_eq : 1001 * A + 110 * B + 110 * C + 1001 * D = 11990) :
    (min (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 1999) ∧
    (max (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 9991) :=
by
  sorry

end sum_of_largest_and_smallest_four_digit_numbers_is_11990_l623_62395


namespace largest_y_coordinate_l623_62344

theorem largest_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y = 2 := 
by 
  -- Proof will be provided here
  sorry

end largest_y_coordinate_l623_62344


namespace least_number_to_add_to_4499_is_1_l623_62387

theorem least_number_to_add_to_4499_is_1 (x : ℕ) : (4499 + x) % 9 = 0 → x = 1 := sorry

end least_number_to_add_to_4499_is_1_l623_62387


namespace roots_transformation_l623_62343

-- Given polynomial
def poly1 (x : ℝ) : ℝ := x^3 - 3*x^2 + 8

-- Polynomial with roots 3*r1, 3*r2, 3*r3
def poly2 (x : ℝ) : ℝ := x^3 - 9*x^2 + 216

-- Theorem stating the equivalence
theorem roots_transformation (r1 r2 r3 : ℝ) 
  (h : ∀ x, poly1 x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x, poly2 x = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
sorry

end roots_transformation_l623_62343


namespace find_cos_squared_y_l623_62323

noncomputable def α : ℝ := Real.arccos (-3 / 7)

def arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

def transformed_arithmetic_progression (a b c : ℝ) : Prop :=
  14 / Real.cos b = 1 / Real.cos a + 1 / Real.cos c

theorem find_cos_squared_y (x y z : ℝ)
  (h1 : arithmetic_progression x y z)
  (h2 : transformed_arithmetic_progression x y z)
  (hα : 2 * α = z - x) : Real.cos y ^ 2 = 10 / 13 :=
by
  sorry

end find_cos_squared_y_l623_62323


namespace jake_snake_length_l623_62367

theorem jake_snake_length (j p : ℕ) (h1 : j = p + 12) (h2 : j + p = 70) : j = 41 := by
  sorry

end jake_snake_length_l623_62367


namespace find_common_ratio_limit_SN_over_TN_l623_62374

noncomputable def S (q : ℚ) (n : ℕ) : ℚ := (1 - q^n) / (1 - q)
noncomputable def T (q : ℚ) (n : ℕ) : ℚ := (1 - q^(2 * n)) / (1 - q^2)

theorem find_common_ratio
  (S3 : S q 3 = 3)
  (S6 : S q 6 = -21) :
  q = -2 :=
sorry

theorem limit_SN_over_TN
  (q_pos : 0 < q)
  (Tn_def : ∀ n, T q n = 1) :
  (q > 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 0| < ε) ∧
  (0 < q ∧ q < 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - (1 + q)| < ε) ∧
  (q = 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 1| < ε) :=
sorry

end find_common_ratio_limit_SN_over_TN_l623_62374


namespace part1_part2_l623_62375

-- Define A and B according to given expressions
def A (a b : ℚ) : ℚ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) : ℚ := -a^2 + a * b - 1

-- Prove the first statement
theorem part1 (a b : ℚ) : 4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 :=
by sorry

-- Prove the second statement
theorem part2 (F : ℚ) (b : ℚ) : (∀ a, A a b + 2 * B a b = F) → b = 2 / 5 :=
by sorry

end part1_part2_l623_62375


namespace common_difference_of_arithmetic_sequence_l623_62399

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ) -- define the arithmetic sequence
  (h_arith : ∀ n : ℕ, a n = a 0 + n * 4) -- condition of arithmetic sequence
  (h_a5 : a 4 = 8) -- given a_5 = 8
  (h_a9 : a 8 = 24) -- given a_9 = 24
  : 4 = 4 := -- statement to be proven
by
  sorry

end common_difference_of_arithmetic_sequence_l623_62399


namespace value_of_a_l623_62328

theorem value_of_a (M : Set ℝ) (N : Set ℝ) (a : ℝ) 
  (hM : M = {-1, 0, 1, 2}) (hN : N = {x | x^2 - a * x < 0}) 
  (hIntersect : M ∩ N = {1, 2}) : 
  a = 3 := 
sorry

end value_of_a_l623_62328


namespace alice_unanswered_questions_l623_62351

theorem alice_unanswered_questions 
    (c w u : ℕ)
    (h1 : 6 * c - 2 * w + 3 * u = 120)
    (h2 : 3 * c - w = 70)
    (h3 : c + w + u = 40) :
    u = 10 :=
sorry

end alice_unanswered_questions_l623_62351


namespace prob_even_sum_is_one_third_l623_62335

def is_even_sum_first_last (d1 d2 d3 d4 : Nat) : Prop :=
  (d1 + d4) % 2 = 0

def num_unique_arrangements : Nat := 12

def num_favorable_arrangements : Nat := 4

def prob_even_sum_first_last : Rat :=
  num_favorable_arrangements / num_unique_arrangements

theorem prob_even_sum_is_one_third :
  prob_even_sum_first_last = 1 / 3 := 
  sorry

end prob_even_sum_is_one_third_l623_62335


namespace part1_part2_l623_62356

-- Definition of sets A, B, and Proposition p for Part 1
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a = 0}
def p (a : ℝ) : Prop := ∀ x ∈ B a, x ∈ A

-- Part 1: Prove the range of a
theorem part1 (a : ℝ) : (p a) → 0 < a ∧ a ≤ 1 :=
  by sorry

-- Definition of sets A and C for Part 2
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 3 > 0}
def necessary_condition (m : ℝ) : Prop := ∀ x ∈ A, x ∈ C m

-- Part 2: Prove the range of m
theorem part2 (m : ℝ) : necessary_condition m → m ≤ 7 / 2 :=
  by sorry

end part1_part2_l623_62356


namespace probability_of_all_girls_chosen_is_1_over_11_l623_62303

-- Defining parameters and conditions
def total_members : ℕ := 12
def boys : ℕ := 6
def girls : ℕ := 6
def chosen_members : ℕ := 3

-- Number of combinations to choose 3 members from 12
def total_combinations : ℕ := Nat.choose total_members chosen_members

-- Number of combinations to choose 3 girls from 6
def girl_combinations : ℕ := Nat.choose girls chosen_members

-- Probability is defined as the ratio of these combinations
def probability_all_girls_chosen : ℚ := girl_combinations / total_combinations

-- Proof Statement
theorem probability_of_all_girls_chosen_is_1_over_11 : probability_all_girls_chosen = 1 / 11 := by
  sorry -- Proof to be completed

end probability_of_all_girls_chosen_is_1_over_11_l623_62303


namespace subtraction_to_nearest_thousandth_l623_62348

theorem subtraction_to_nearest_thousandth : 
  (456.789 : ℝ) - (234.567 : ℝ) = 222.222 :=
by
  sorry

end subtraction_to_nearest_thousandth_l623_62348


namespace maximum_diagonal_intersections_l623_62362

theorem maximum_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
  ∃ k, k = (n * (n - 1) * (n - 2) * (n - 3)) / 24 :=
by sorry

end maximum_diagonal_intersections_l623_62362


namespace base5_product_is_correct_l623_62381

-- Definitions for the problem context
def base5_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 528 then [4, 1, 0, 0, 3] else []

-- Theorem to prove the base-5 multiplication result
theorem base5_product_is_correct :
  base10_to_base5 (base5_to_base10 2 3 1 * base5_to_base10 1 3 0) = [4, 1, 0, 0, 3] :=
by
  sorry

end base5_product_is_correct_l623_62381


namespace grayson_time_per_answer_l623_62360

variable (totalQuestions : ℕ) (unansweredQuestions : ℕ) (totalTimeHours : ℕ)

def timePerAnswer (totalQuestions : ℕ) (unansweredQuestions : ℕ) (totalTimeHours : ℕ) : ℕ :=
  let answeredQuestions := totalQuestions - unansweredQuestions
  let totalTimeMinutes := totalTimeHours * 60
  totalTimeMinutes / answeredQuestions

theorem grayson_time_per_answer :
  totalQuestions = 100 →
  unansweredQuestions = 40 →
  totalTimeHours = 2 →
  timePerAnswer totalQuestions unansweredQuestions totalTimeHours = 2 :=
by
  intros hTotal hUnanswered hTime
  rw [hTotal, hUnanswered, hTime]
  sorry

end grayson_time_per_answer_l623_62360


namespace solve_for_x_l623_62319

theorem solve_for_x (x : ℝ) (h : (5 * x - 3) / (6 * x - 6) = (4 / 3)) : x = 5 / 3 :=
sorry

end solve_for_x_l623_62319


namespace carlos_improved_lap_time_l623_62349

-- Define the initial condition using a function to denote time per lap initially
def initial_lap_time : ℕ := (45 * 60) / 15

-- Define the later condition using a function to denote time per lap later on
def current_lap_time : ℕ := (42 * 60) / 18

-- Define the proof that calculates the improvement in seconds
theorem carlos_improved_lap_time : initial_lap_time - current_lap_time = 40 := by
  sorry

end carlos_improved_lap_time_l623_62349


namespace fraction_identity_l623_62377

theorem fraction_identity (a b : ℝ) (h : a / b = 3 / 4) : a / (a + b) = 3 / 7 := 
by
  sorry

end fraction_identity_l623_62377


namespace trig_identity_one_trig_identity_two_l623_62316

theorem trig_identity_one :
  2 * (Real.cos (45 * Real.pi / 180)) - (3 / 2) * (Real.tan (30 * Real.pi / 180)) * (Real.cos (30 * Real.pi / 180)) + (Real.sin (60 * Real.pi / 180))^2 = Real.sqrt 2 :=
sorry

theorem trig_identity_two :
  (Real.sin (30 * Real.pi / 180))⁻¹ * (Real.sin (60 * Real.pi / 180) - Real.cos (45 * Real.pi / 180)) - Real.sqrt ((1 - Real.tan (60 * Real.pi / 180))^2) = 1 - Real.sqrt 2 :=
sorry

end trig_identity_one_trig_identity_two_l623_62316


namespace find_f_2013_l623_62353

noncomputable def f : ℝ → ℝ := sorry
axiom functional_eq : ∀ (m n : ℝ), f (m + n^2) = f m + 2 * (f n)^2
axiom f_1_ne_0 : f 1 ≠ 0

theorem find_f_2013 : f 2013 = 4024 * (f 1)^2 + f 1 :=
sorry

end find_f_2013_l623_62353


namespace quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l623_62306

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l623_62306


namespace g_is_even_l623_62321

noncomputable def g (x : ℝ) : ℝ := 4^(x^2 - 3) - 2 * |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l623_62321


namespace milo_skateboarding_speed_l623_62385

theorem milo_skateboarding_speed (cory_speed milo_skateboarding_speed : ℝ) 
  (h1 : cory_speed = 12) 
  (h2 : cory_speed = 2 * milo_skateboarding_speed) : 
  milo_skateboarding_speed = 6 :=
by sorry

end milo_skateboarding_speed_l623_62385


namespace mabel_counts_sharks_l623_62334

theorem mabel_counts_sharks 
    (fish_day1 : ℕ) 
    (fish_day2 : ℕ) 
    (shark_percentage : ℚ) 
    (total_fish : ℕ) 
    (total_sharks : ℕ) 
    (h1 : fish_day1 = 15) 
    (h2 : fish_day2 = 3 * fish_day1) 
    (h3 : shark_percentage = 0.25) 
    (h4 : total_fish = fish_day1 + fish_day2) 
    (h5 : total_sharks = total_fish * shark_percentage) : 
    total_sharks = 15 := 
by {
  sorry
}

end mabel_counts_sharks_l623_62334


namespace algebraic_expression_evaluation_l623_62336

theorem algebraic_expression_evaluation (x y : ℝ) (h : 2 * x - y + 1 = 3) : 4 * x - 2 * y + 5 = 9 := 
by
  sorry

end algebraic_expression_evaluation_l623_62336


namespace total_selling_price_correct_l623_62331

def original_price : ℝ := 100
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.08

theorem total_selling_price_correct :
  let discount := original_price * discount_percent
  let sale_price := original_price - discount
  let tax := sale_price * tax_percent
  let total_selling_price := sale_price + tax
  total_selling_price = 75.6 := by
sorry

end total_selling_price_correct_l623_62331


namespace systematic_sampling_starts_with_srs_l623_62312

-- Define the concept of systematic sampling
def systematically_sampled (initial_sampled: Bool) : Bool :=
  initial_sampled

-- Initial sample is determined by simple random sampling
def simple_random_sampling : Bool :=
  True

-- We need to prove that systematic sampling uses simple random sampling at the start
theorem systematic_sampling_starts_with_srs : systematically_sampled simple_random_sampling = True :=
by 
  sorry

end systematic_sampling_starts_with_srs_l623_62312


namespace solve_system_l623_62347

def system_of_equations : Prop :=
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ x + 2 * y = -2 ∧ x = 2 ∧ y = -2

theorem solve_system : system_of_equations := by
  sorry

end solve_system_l623_62347
