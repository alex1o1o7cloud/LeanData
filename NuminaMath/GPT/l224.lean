import Mathlib

namespace problem_solution_l224_224414

variables {p q r : ℝ}

theorem problem_solution (h1 : (p + q) * (q + r) * (r + p) / (p * q * r) = 24)
  (h2 : (p - 2 * q) * (q - 2 * r) * (r - 2 * p) / (p * q * r) = 10) :
  ∃ m n : ℕ, (m.gcd n = 1 ∧ (p/q + q/r + r/p = m/n) ∧ m + n = 39) :=
sorry

end problem_solution_l224_224414


namespace elena_pens_l224_224548

theorem elena_pens (X Y : ℕ) (h1 : X + Y = 12) (h2 : 4*X + 22*Y = 420) : X = 9 := by
  sorry

end elena_pens_l224_224548


namespace train_car_passengers_l224_224465

theorem train_car_passengers (x : ℕ) (h : 60 * x = 732 + 228) : x = 16 :=
by
  sorry

end train_car_passengers_l224_224465


namespace inverse_tangent_line_l224_224169

theorem inverse_tangent_line
  (f : ℝ → ℝ)
  (hf₁ : ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x) 
  (hf₂ : ∀ x, deriv f x ≠ 0)
  (h_tangent : ∀ x₀, (2 * x₀ - f x₀ + 3) = 0) :
  ∀ x₀, (x₀ - 2 * f x₀ - 3) = 0 :=
by
  sorry

end inverse_tangent_line_l224_224169


namespace combined_weight_l224_224922

-- Define the conditions
variables (Ron_weight Roger_weight Rodney_weight : ℕ)

-- Define the conditions as Lean propositions
def conditions : Prop :=
  Rodney_weight = 2 * Roger_weight ∧ 
  Roger_weight = 4 * Ron_weight - 7 ∧ 
  Rodney_weight = 146

-- Define the proof goal
def proof_goal : Prop :=
  Rodney_weight + Roger_weight + Ron_weight = 239

theorem combined_weight (Ron_weight Roger_weight Rodney_weight : ℕ) (h : conditions Ron_weight Roger_weight Rodney_weight) : 
  proof_goal Ron_weight Roger_weight Rodney_weight :=
sorry

end combined_weight_l224_224922


namespace rate_of_markup_l224_224641

theorem rate_of_markup (S : ℝ) (hS : S = 8)
  (profit_percent : ℝ) (h_profit_percent : profit_percent = 0.20)
  (expense_percent : ℝ) (h_expense_percent : expense_percent = 0.10) :
  (S - (S * (1 - profit_percent - expense_percent))) / (S * (1 - profit_percent - expense_percent)) * 100 = 42.857 :=
by
  sorry

end rate_of_markup_l224_224641


namespace smallest_base10_integer_exists_l224_224343

theorem smallest_base10_integer_exists : ∃ (n a b : ℕ), a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1 ∧ n = 10 := by
  sorry

end smallest_base10_integer_exists_l224_224343


namespace evaluate_power_l224_224353

theorem evaluate_power (n : ℕ) (h : 3^(2 * n) = 81) : 9^(n + 1) = 729 :=
by sorry

end evaluate_power_l224_224353


namespace exists_triangle_with_prime_angles_l224_224157

-- Definition of prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Definition of being an angle of a triangle
def is_valid_angle (α : ℕ) : Prop := α > 0 ∧ α < 180

-- Main statement
theorem exists_triangle_with_prime_angles :
  ∃ (α β γ : ℕ), is_prime α ∧ is_prime β ∧ is_prime γ ∧ is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧ α + β + γ = 180 :=
by
  sorry

end exists_triangle_with_prime_angles_l224_224157


namespace bottles_per_person_l224_224750

theorem bottles_per_person
  (boxes : ℕ)
  (bottles_per_box : ℕ)
  (bottles_eaten : ℕ)
  (people : ℕ)
  (total_bottles : ℕ := boxes * bottles_per_box)
  (remaining_bottles : ℕ := total_bottles - bottles_eaten)
  (bottles_per_person : ℕ := remaining_bottles / people) :
  boxes = 7 → bottles_per_box = 9 → bottles_eaten = 7 → people = 8 → bottles_per_person = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end bottles_per_person_l224_224750


namespace initial_rope_length_l224_224723

theorem initial_rope_length : 
  ∀ (π : ℝ), 
  ∀ (additional_area : ℝ) (new_rope_length : ℝ), 
  additional_area = 933.4285714285714 →
  new_rope_length = 21 →
  ∃ (initial_rope_length : ℝ), 
  additional_area = π * (new_rope_length^2 - initial_rope_length^2) ∧
  initial_rope_length = 12 :=
by
  sorry

end initial_rope_length_l224_224723


namespace reciprocal_of_neg_2023_l224_224066

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l224_224066


namespace point_on_line_l224_224911

theorem point_on_line (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 4 = 2 * (n + k) + 5) : k = 2 := by
  sorry

end point_on_line_l224_224911


namespace rectangle_y_value_l224_224935

theorem rectangle_y_value 
  (y : ℝ)
  (A : (0, 0) = E ∧ (0, 5) = F ∧ (y, 5) = G ∧ (y, 0) = H)
  (area : 5 * y = 35)
  (y_pos : y > 0) :
  y = 7 :=
sorry

end rectangle_y_value_l224_224935


namespace find_percentage_l224_224675

variable (P x : ℝ)

theorem find_percentage (h1 : x = 10)
    (h2 : (P / 100) * x = 0.05 * 500 - 20) : P = 50 := by
  sorry

end find_percentage_l224_224675


namespace gcd_polynomial_l224_224318

theorem gcd_polynomial (a : ℕ) (h : 270 ∣ a) : Nat.gcd (5 * a^3 + 3 * a^2 + 5 * a + 45) a = 45 :=
sorry

end gcd_polynomial_l224_224318


namespace age_in_1930_l224_224111

/-- A person's age at the time of their death (y) was one 31st of their birth year,
and we want to prove the person's age in 1930 (x). -/
theorem age_in_1930 (x y : ℕ) (h : 31 * y + x = 1930) (hx : 0 < x) (hxy : x < y) :
  x = 39 :=
sorry

end age_in_1930_l224_224111


namespace least_positive_integer_fac_6370_factorial_l224_224553

theorem least_positive_integer_fac_6370_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, (6370 ∣ m.factorial) → m ≥ n) ∧ n = 14 :=
by
  sorry

end least_positive_integer_fac_6370_factorial_l224_224553


namespace prism_height_l224_224300

theorem prism_height (a h : ℝ) 
  (base_side : a = 10) 
  (total_edge_length : 3 * a + 3 * a + 3 * h = 84) : 
  h = 8 :=
by sorry

end prism_height_l224_224300


namespace greatest_divisor_same_remainder_l224_224440

theorem greatest_divisor_same_remainder (a b c : ℕ) (d1 d2 d3 : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113)
(hd1 : d1 = b - a) (hd2 : d2 = c - b) (hd3 : d3 = c - a) :
  Nat.gcd (Nat.gcd d1 d2) d3 = 6 :=
by
  -- some computation here which we are skipping
  sorry

end greatest_divisor_same_remainder_l224_224440


namespace coeff_abs_sum_eq_729_l224_224054

-- Given polynomial (2x - 1)^6 expansion
theorem coeff_abs_sum_eq_729 (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (2 * x - 1) ^ 6 = a_6 * x ^ 6 + a_5 * x ^ 5 + a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + a_0 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end coeff_abs_sum_eq_729_l224_224054


namespace solve_fractional_eq_l224_224442

theorem solve_fractional_eq (x : ℝ) (h₀ : x ≠ 2) (h₁ : x ≠ -2) :
  (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) → (x = 3 / 2) :=
by sorry

end solve_fractional_eq_l224_224442


namespace ellipse_iff_k_range_l224_224406

theorem ellipse_iff_k_range (k : ℝ) :
  (∃ x y, (x ^ 2 / (1 - k)) + (y ^ 2 / (1 + k)) = 1) ↔ (-1 < k ∧ k < 1 ∧ k ≠ 0) :=
by
  sorry

end ellipse_iff_k_range_l224_224406


namespace opposite_of_neg_quarter_l224_224244

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end opposite_of_neg_quarter_l224_224244


namespace general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l224_224459

-- Define the conditions
axiom condition1 (n : ℕ) (h : 2 ≤ n) : ∀ (a : ℕ → ℕ), a 1 = 1 → a n = n / (n-1) * a (n-1)
axiom condition2 (n : ℕ) : ∀ (S : ℕ → ℕ), 2 * S n = n^2 + n
axiom condition3 (n : ℕ) : ∀ (a : ℕ → ℕ), a 1 = 1 → a 3 = 3 → (a n + a (n+2)) = 2 * a (n+1)

-- Proof statements
theorem general_formula_condition1 : ∀ (n : ℕ) (a : ℕ → ℕ) (h : 2 ≤ n), (a 1 = 1) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition2 : ∀ (n : ℕ) (S a : ℕ → ℕ), (2 * S n = n^2 + n) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition3 : ∀ (n : ℕ) (a : ℕ → ℕ), (a 1 = 1) → (a 3 = 3) → (∀ n, a n + a (n+2) = 2 * a (n+1)) → (∀ n, a n = n) :=
by sorry

theorem sum_Tn : ∀ (b : ℕ → ℕ) (T : ℕ → ℝ), (b 1 = 2) → (b 2 + b 3 = 12) → (∀ n, T n = 2 * (1 - 1 / (n + 1))) :=
by sorry

end general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l224_224459


namespace range_of_m_l224_224842

-- Definitions of the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x + 1| > m
def q (m : ℝ) : Prop := ∀ x > 2, 2 * x - 2 * m > 0

-- The main theorem statement
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l224_224842


namespace probability_red_or_white_is_7_over_10_l224_224936

/-
A bag consists of 20 marbles, of which 6 are blue, 9 are red, and the remainder are white.
If Lisa is to select a marble from the bag at random, prove that the probability that the
marble will be red or white is 7/10.
-/
def num_marbles : ℕ := 20
def num_blue : ℕ := 6
def num_red : ℕ := 9
def num_white : ℕ := num_marbles - (num_blue + num_red)

def probability_red_or_white : ℚ :=
  (num_red + num_white) / num_marbles

theorem probability_red_or_white_is_7_over_10 :
  probability_red_or_white = 7 / 10 := 
sorry

end probability_red_or_white_is_7_over_10_l224_224936


namespace collinear_condition_l224_224752

theorem collinear_condition {a b c d : ℝ} (h₁ : a < b) (h₂ : c < d) (h₃ : a < d) (h₄ : c < b) :
  (a / d) + (c / b) = 1 := 
sorry

end collinear_condition_l224_224752


namespace hannah_payment_l224_224965

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end hannah_payment_l224_224965


namespace show_R_r_eq_l224_224270

variables {a b c R r : ℝ}

-- Conditions
def sides_of_triangle (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

def circumradius (R a b c : ℝ) (Δ : ℝ) : Prop :=
R = a * b * c / (4 * Δ)

def inradius (r Δ : ℝ) (s : ℝ) : Prop :=
r = Δ / s

theorem show_R_r_eq (a b c : ℝ) (R r : ℝ) (Δ : ℝ) (s : ℝ) (h_sides : sides_of_triangle a b c)
  (h_circumradius : circumradius R a b c Δ)
  (h_inradius : inradius r Δ s)
  (h_semiperimeter : s = (a + b + c) / 2) :
  R * r = a * b * c / (2 * (a + b + c)) :=
sorry

end show_R_r_eq_l224_224270


namespace find_sum_of_variables_l224_224208

variables (a b c d : ℤ)

theorem find_sum_of_variables
    (h1 : a - b + c = 7)
    (h2 : b - c + d = 8)
    (h3 : c - d + a = 4)
    (h4 : d - a + b = 3)
    (h5 : a + b + c - d = 10) :
    a + b + c + d = 16 := 
sorry

end find_sum_of_variables_l224_224208


namespace probability_is_one_twelfth_l224_224326

def probability_red_gt4_green_odd_blue_lt4 : ℚ :=
  let total_outcomes := 6 * 6 * 6
  let successful_outcomes := 2 * 3 * 3
  successful_outcomes / total_outcomes

theorem probability_is_one_twelfth :
  probability_red_gt4_green_odd_blue_lt4 = 1 / 12 :=
by
  -- proof here
  sorry

end probability_is_one_twelfth_l224_224326


namespace square_floor_tiling_total_number_of_tiles_l224_224098

theorem square_floor_tiling (s : ℕ) (h : (2 * s - 1 : ℝ) / (s ^ 2 : ℝ) = 0.41) : s = 4 :=
by
  sorry

theorem total_number_of_tiles : 4^2 = 16 := 
by
  norm_num

end square_floor_tiling_total_number_of_tiles_l224_224098


namespace integer_root_count_l224_224887

theorem integer_root_count (b : ℝ) :
  (∃ r s : ℤ, r + s = b ∧ r * s = 8 * b) ↔
  b = -9 ∨ b = 0 ∨ b = 9 :=
sorry

end integer_root_count_l224_224887


namespace smallest_c_for_inverse_l224_224907

noncomputable def g (x : ℝ) : ℝ := (x + 3)^2 - 6

theorem smallest_c_for_inverse : 
  ∃ (c : ℝ), (∀ x1 x2, x1 ≥ c → x2 ≥ c → g x1 = g x2 → x1 = x2) ∧ 
            (∀ c', c' < c → ∃ x1 x2, x1 ≥ c' → x2 ≥ c' → g x1 = g x2 ∧ x1 ≠ x2) ∧ 
            c = -3 :=
by 
  sorry

end smallest_c_for_inverse_l224_224907


namespace catch_two_salmon_l224_224050

def totalTroutWeight : ℕ := 8
def numBass : ℕ := 6
def weightPerBass : ℕ := 2
def totalBassWeight : ℕ := numBass * weightPerBass
def campers : ℕ := 22
def weightPerCamper : ℕ := 2
def totalFishWeightRequired : ℕ := campers * weightPerCamper
def totalTroutAndBassWeight : ℕ := totalTroutWeight + totalBassWeight
def additionalFishWeightRequired : ℕ := totalFishWeightRequired - totalTroutAndBassWeight
def weightPerSalmon : ℕ := 12
def numSalmon : ℕ := additionalFishWeightRequired / weightPerSalmon

theorem catch_two_salmon : numSalmon = 2 := by
  sorry

end catch_two_salmon_l224_224050


namespace bottle_caps_found_l224_224104

theorem bottle_caps_found
  (caps_current : ℕ) 
  (caps_earlier : ℕ) 
  (h_current : caps_current = 32) 
  (h_earlier : caps_earlier = 25) :
  caps_current - caps_earlier = 7 :=
by 
  sorry

end bottle_caps_found_l224_224104


namespace math_problem_l224_224011

theorem math_problem :
  let result := 83 - 29
  let final_sum := result + 58
  let rounded := if final_sum % 10 < 5 then final_sum - final_sum % 10 else final_sum + (10 - final_sum % 10)
  rounded = 110 := by
  sorry

end math_problem_l224_224011


namespace parallelogram_height_l224_224305

theorem parallelogram_height (A b h : ℝ) (hA : A = 288) (hb : b = 18) : h = 16 :=
by
  sorry

end parallelogram_height_l224_224305


namespace cube_difference_l224_224408

theorem cube_difference (n : ℕ) (h: 0 < n) : (n + 1)^3 - n^3 = 3 * n^2 + 3 * n + 1 := 
sorry

end cube_difference_l224_224408


namespace evaluate_expression_l224_224731

-- Define the mathematical expressions using Lean's constructs
def expr1 : ℕ := 201 * 5 + 1220 - 2 * 3 * 5 * 7

-- State the theorem we aim to prove
theorem evaluate_expression : expr1 = 2015 := by
  sorry

end evaluate_expression_l224_224731


namespace bd_le_q2_l224_224544

theorem bd_le_q2 (a b c d p q : ℝ) (h1 : a * b + c * d = 2 * p * q) (h2 : a * c ≥ p^2 ∧ p^2 > 0) : b * d ≤ q^2 :=
sorry

end bd_le_q2_l224_224544


namespace pentomino_symmetry_count_l224_224655

def is_pentomino (shape : Type) : Prop :=
  -- Define the property of being a pentomino as composed of five squares edge to edge
  sorry

def has_reflectional_symmetry (shape : Type) : Prop :=
  -- Define the property of having at least one line of reflectional symmetry
  sorry

def has_rotational_symmetry_of_order_2 (shape : Type) : Prop :=
  -- Define the property of having rotational symmetry of order 2 (180 degrees rotation results in the same shape)
  sorry

noncomputable def count_valid_pentominoes : Nat :=
  -- Assume that we have a list of 18 pentominoes
  -- Count the number of pentominoes that meet both criteria
  sorry

theorem pentomino_symmetry_count :
  count_valid_pentominoes = 4 :=
sorry

end pentomino_symmetry_count_l224_224655


namespace smaller_group_men_l224_224467

theorem smaller_group_men (M : ℕ) (h1 : 36 * 25 = M * 90) : M = 10 :=
by
  -- Here we would provide the proof. Unfortunately, proving this in Lean 4 requires knowledge of algebra.
  sorry

end smaller_group_men_l224_224467


namespace relationship_of_y1_y2_l224_224719

theorem relationship_of_y1_y2 (y1 y2 : ℝ) : 
  (∃ y1 y2, (y1 = 2 / -2) ∧ (y2 = 2 / -1)) → (y1 > y2) :=
by
  sorry

end relationship_of_y1_y2_l224_224719


namespace tamara_is_68_inch_l224_224927

-- Defining the conditions
variables (K T : ℕ)

-- Condition 1: Tamara's height in terms of Kim's height
def tamara_height := T = 3 * K - 4

-- Condition 2: Combined height of Tamara and Kim
def combined_height := T + K = 92

-- Statement to prove: Tamara's height is 68 inches
theorem tamara_is_68_inch (h1 : tamara_height T K) (h2 : combined_height T K) : T = 68 :=
by
  sorry

end tamara_is_68_inch_l224_224927


namespace max_n_factoring_polynomial_l224_224395

theorem max_n_factoring_polynomial :
  ∃ n A B : ℤ, (3 * n + A = 217) ∧ (A * B = 72) ∧ (3 * B + A = n) :=
sorry

end max_n_factoring_polynomial_l224_224395


namespace income_of_sixth_member_l224_224900

def income_member1 : ℝ := 11000
def income_member2 : ℝ := 15000
def income_member3 : ℝ := 10000
def income_member4 : ℝ := 9000
def income_member5 : ℝ := 13000
def number_of_members : ℕ := 6
def average_income : ℝ := 12000
def total_income_of_five_members := income_member1 + income_member2 + income_member3 + income_member4 + income_member5

theorem income_of_sixth_member :
  6 * average_income - total_income_of_five_members = 14000 := by
  sorry

end income_of_sixth_member_l224_224900


namespace unique_point_intersection_l224_224019

theorem unique_point_intersection (k : ℝ) :
  (∃ x y, y = k * x + 2 ∧ y ^ 2 = 8 * x) → 
  ((k = 0) ∨ (k = 1)) :=
by {
  sorry
}

end unique_point_intersection_l224_224019


namespace arithmetic_sequence_sum_l224_224899

theorem arithmetic_sequence_sum :
  3 * (75 + 77 + 79 + 81 + 83) = 1185 := by
  sorry

end arithmetic_sequence_sum_l224_224899


namespace fraction_distinctly_marked_l224_224818

theorem fraction_distinctly_marked 
  (area_large_rectangle : ℕ)
  (fraction_shaded : ℚ)
  (fraction_further_marked : ℚ)
  (h_area_large_rectangle : area_large_rectangle = 15 * 24)
  (h_fraction_shaded : fraction_shaded = 1/3)
  (h_fraction_further_marked : fraction_further_marked = 1/2) :
  (fraction_further_marked * fraction_shaded = 1/6) :=
by
  sorry

end fraction_distinctly_marked_l224_224818


namespace line_through_point_with_equal_intercepts_l224_224372

theorem line_through_point_with_equal_intercepts (x y : ℝ) :
  (∃ b : ℝ, 3 * x + y = 0) ∨ (∃ b : ℝ, x - y + 4 = 0) ∨ (∃ b : ℝ, x + y - 2 = 0) :=
  sorry

end line_through_point_with_equal_intercepts_l224_224372


namespace pascal_fifth_number_l224_224798

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end pascal_fifth_number_l224_224798


namespace magnitude_of_b_l224_224393

variables (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ)

-- Defining the conditions
def vector_a_magnitude : Prop := ‖a‖ = 1
def vector_angle_condition : Prop := θ = Real.pi / 3
def linear_combination_magnitude : Prop := ‖2 • a - b‖ = 2 * Real.sqrt 3
def b_magnitude : Prop := ‖b‖ = 4

-- The statement we want to prove
theorem magnitude_of_b (h1 : vector_a_magnitude a) (h2 : vector_angle_condition θ) (h3 : linear_combination_magnitude a b) : b_magnitude b :=
sorry

end magnitude_of_b_l224_224393


namespace new_room_correct_size_l224_224438

-- Definitions of conditions
def current_bedroom := 309 -- sq ft
def current_bathroom := 150 -- sq ft
def current_space := current_bedroom + current_bathroom
def new_room_size := 2 * current_space

-- Proving the new room size
theorem new_room_correct_size : new_room_size = 918 := by
  sorry

end new_room_correct_size_l224_224438


namespace sample_mean_and_variance_l224_224539

def sample : List ℕ := [10, 12, 9, 14, 13]
def n : ℕ := 5

-- Definition of sample mean
noncomputable def sampleMean : ℝ := (sample.sum / n)

-- Definition of sample variance using population formula
noncomputable def sampleVariance : ℝ := (sample.map (λ x_i => (x_i - sampleMean)^2)).sum / n

theorem sample_mean_and_variance :
  sampleMean = 11.6 ∧ sampleVariance = 3.44 := by
  sorry

end sample_mean_and_variance_l224_224539


namespace find_kids_l224_224524

theorem find_kids (A K : ℕ) (h1 : A + K = 12) (h2 : 3 * A = 15) : K = 7 :=
sorry

end find_kids_l224_224524


namespace square_side_increase_l224_224777

theorem square_side_increase (p : ℝ) (h : (1 + p / 100)^2 = 1.69) : p = 30 :=
by {
  sorry
}

end square_side_increase_l224_224777


namespace smallest_diff_l224_224498

noncomputable def triangleSides : ℕ → ℕ → ℕ → Prop := λ AB BC AC =>
  AB < BC ∧ BC ≤ AC ∧ AB + BC + AC = 2007

theorem smallest_diff (AB BC AC : ℕ) (h : triangleSides AB BC AC) : BC - AB = 1 :=
  sorry

end smallest_diff_l224_224498


namespace like_terms_exp_l224_224302

theorem like_terms_exp (a b : ℝ) (m n x : ℝ)
  (h₁ : 2 * a ^ x * b ^ (n + 1) = -3 * a * b ^ (2 * m))
  (h₂ : x = 1) (h₃ : n + 1 = 2 * m) : 
  (2 * m - n) ^ x = 1 := 
by
  sorry

end like_terms_exp_l224_224302


namespace ratio_of_radii_of_touching_circles_l224_224063

theorem ratio_of_radii_of_touching_circles
  (r R : ℝ) (A B C D : ℝ) (h1 : A + B + C = D)
  (h2 : 3 * A = 7 * B)
  (h3 : 7 * B = 2 * C)
  (h4 : R = D / 2)
  (h5 : B = R - 3 * A)
  (h6 : C = R - 2 * A)
  (h7 : r = 4 * A)
  (h8 : R = 6 * A) :
  R / r = 3 / 2 := by
  sorry

end ratio_of_radii_of_touching_circles_l224_224063


namespace equal_sets_l224_224571

def M : Set ℝ := {x | x^2 + 16 = 0}
def N : Set ℝ := {x | x^2 + 6 = 0}

theorem equal_sets : M = N := by
  sorry

end equal_sets_l224_224571


namespace minimum_value_amgm_l224_224839

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end minimum_value_amgm_l224_224839


namespace find_9a_value_l224_224089

theorem find_9a_value (a : ℚ) 
  (h : (4 - a) / (5 - a) = (4 / 5) ^ 2) : 9 * a = 20 :=
by
  sorry

end find_9a_value_l224_224089


namespace goshawk_eurasian_reserve_hawks_l224_224264

variable (H P : ℝ)

theorem goshawk_eurasian_reserve_hawks :
  P = 100 ∧
  (35 / 100) * P = P - (H + (40 / 100) * (P - H) + (25 / 100) * (40 / 100) * (P - H))
    → H = 25 :=
by sorry

end goshawk_eurasian_reserve_hawks_l224_224264


namespace arithmetic_sequence_a8_l224_224132

theorem arithmetic_sequence_a8 (a_1 : ℕ) (S_5 : ℕ) (h_a1 : a_1 = 1) (h_S5 : S_5 = 35) : 
    ∃ a_8 : ℕ, a_8 = 22 :=
by
  sorry

end arithmetic_sequence_a8_l224_224132


namespace parallelogram_sides_l224_224214

theorem parallelogram_sides (a b : ℝ)
  (h1 : 2 * (a + b) = 32)
  (h2 : b - a = 8) :
  a = 4 ∧ b = 12 :=
by
  -- Proof is to be provided
  sorry

end parallelogram_sides_l224_224214


namespace problem_part1_problem_part2_l224_224231

theorem problem_part1 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 * y - x * y^2 = 4 * Real.sqrt 2 := 
  sorry

theorem problem_part2 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 - x * y + y^2 = 33 := 
  sorry

end problem_part1_problem_part2_l224_224231


namespace ellipse_equation_l224_224722

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end ellipse_equation_l224_224722


namespace train_B_time_to_destination_l224_224780

theorem train_B_time_to_destination (speed_A : ℕ) (time_A : ℕ) (speed_B : ℕ) (dA : ℕ) :
  speed_A = 100 ∧ time_A = 9 ∧ speed_B = 150 ∧ dA = speed_A * time_A →
  dA / speed_B = 6 := 
by
  sorry

end train_B_time_to_destination_l224_224780


namespace number_of_roses_two_days_ago_l224_224458

-- Define the conditions
variables (R : ℕ) 
-- Condition 1: Variable R is the number of roses planted two days ago.
-- Condition 2: The number of roses planted yesterday is R + 20.
-- Condition 3: The number of roses planted today is 2R.
-- Condition 4: The total number of roses planted over three days is 220.
axiom condition_1 : 0 ≤ R
axiom condition_2 : (R + (R + 20) + (2 * R)) = 220

-- Proof goal: Prove that R = 50 
theorem number_of_roses_two_days_ago : R = 50 :=
by sorry

end number_of_roses_two_days_ago_l224_224458


namespace cubic_inequality_l224_224671

theorem cubic_inequality 
  (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end cubic_inequality_l224_224671


namespace work_problem_l224_224001

theorem work_problem (A B : ℝ) (hA : A = 1/4) (hB : B = 1/12) :
  (2 * (A + B) + 4 * B = 1) :=
by
  -- Work rate of A and B together
  -- Work done in 2 days by both
  -- Remaining work and time taken by B alone
  -- Final Result
  sorry

end work_problem_l224_224001


namespace choir_average_age_l224_224745

theorem choir_average_age
  (avg_females_age : ℕ)
  (num_females : ℕ)
  (avg_males_age : ℕ)
  (num_males : ℕ)
  (females_avg_condition : avg_females_age = 28)
  (females_num_condition : num_females = 8)
  (males_avg_condition : avg_males_age = 32)
  (males_num_condition : num_males = 17) :
  ((avg_females_age * num_females + avg_males_age * num_males) / (num_females + num_males) = 768 / 25) :=
by
  sorry

end choir_average_age_l224_224745


namespace trapezoid_circumscribed_radius_l224_224892

theorem trapezoid_circumscribed_radius 
  (a b : ℝ) 
  (height : ℝ)
  (ratio_ab : a / b = 5 / 12)
  (height_eq_midsegment : height = 17) :
  ∃ r : ℝ, r = 13 :=
by
  -- Assuming conditions directly as given
  have h1 : a / b = 5 / 12 := ratio_ab
  have h2 : height = 17 := height_eq_midsegment
  -- The rest of the proof goes here
  sorry

end trapezoid_circumscribed_radius_l224_224892


namespace range_of_a_l224_224673

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → deriv (f a) x < 0) ∧
  (∀ x, 6 < x → deriv (f a) x > 0) →
  5 ≤ a ∧ a ≤ 7 :=
sorry

end range_of_a_l224_224673


namespace sum_345_consecutive_sequences_l224_224283

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end sum_345_consecutive_sequences_l224_224283


namespace number_of_divisors_of_36_l224_224010

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l224_224010


namespace emberly_total_miles_l224_224999

noncomputable def totalMilesWalkedInMarch : ℕ :=
  let daysInMarch := 31
  let daysNotWalked := 4
  let milesPerDay := 4
  (daysInMarch - daysNotWalked) * milesPerDay

theorem emberly_total_miles : totalMilesWalkedInMarch = 108 :=
by
  sorry

end emberly_total_miles_l224_224999


namespace unique_9_tuple_satisfying_condition_l224_224886

theorem unique_9_tuple_satisfying_condition :
  ∃! (a : Fin 9 → ℕ), 
    (∀ i j k : Fin 9, i < j ∧ j < k →
      ∃ l : Fin 9, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100) :=
sorry

end unique_9_tuple_satisfying_condition_l224_224886


namespace abc_equivalence_l224_224229

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end abc_equivalence_l224_224229


namespace power_equality_l224_224301

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l224_224301


namespace min_y_squared_l224_224030

noncomputable def isosceles_trapezoid_bases (EF GH : ℝ) := EF = 102 ∧ GH = 26

noncomputable def trapezoid_sides (EG FH y : ℝ) := EG = y ∧ FH = y

noncomputable def tangent_circle (center_on_EF tangent_to_EG_FH : Prop) := 
  ∃ P : ℝ × ℝ, true -- center P exists somewhere and lies on EF

theorem min_y_squared (EF GH EG FH y : ℝ) (center_on_EF tangent_to_EG_FH : Prop) 
  (h1 : isosceles_trapezoid_bases EF GH)
  (h2 : trapezoid_sides EG FH y)
  (h3 : tangent_circle center_on_EF tangent_to_EG_FH) : 
  ∃ n : ℝ, n^2 = 1938 :=
sorry

end min_y_squared_l224_224030


namespace orthogonal_vectors_y_value_l224_224855

theorem orthogonal_vectors_y_value (y : ℝ) :
  (3 : ℝ) * (-1) + (4 : ℝ) * y = 0 → y = 3 / 4 :=
by
  sorry

end orthogonal_vectors_y_value_l224_224855


namespace number_of_sodas_bought_l224_224476

-- Definitions based on conditions
def cost_sandwich : ℝ := 1.49
def cost_two_sandwiches : ℝ := 2 * cost_sandwich
def cost_soda : ℝ := 0.87
def total_cost : ℝ := 6.46

-- We need to prove that the number of sodas bought is 4 given these conditions
theorem number_of_sodas_bought : (total_cost - cost_two_sandwiches) / cost_soda = 4 := by
  sorry

end number_of_sodas_bought_l224_224476


namespace Wendy_age_l224_224137

theorem Wendy_age
  (years_as_accountant : ℕ)
  (years_as_manager : ℕ)
  (percent_accounting_related : ℝ)
  (total_accounting_related : ℕ)
  (total_lifespan : ℝ) :
  years_as_accountant = 25 →
  years_as_manager = 15 →
  percent_accounting_related = 0.50 →
  total_accounting_related = years_as_accountant + years_as_manager →
  (total_accounting_related : ℝ) = percent_accounting_related * total_lifespan →
  total_lifespan = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Wendy_age_l224_224137


namespace find_k_l224_224289

variable (m n k : ℝ)

def line (x y : ℝ) : Prop := x = 2 * y + 3
def point1_on_line : Prop := line m n
def point2_on_line : Prop := line (m + 2) (n + k)

theorem find_k (h1 : point1_on_line m n) (h2 : point2_on_line m n k) : k = 0 :=
by
  sorry

end find_k_l224_224289


namespace abs_neg_seventeen_l224_224880

theorem abs_neg_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end abs_neg_seventeen_l224_224880


namespace triangle_congruence_example_l224_224133

variable {A B C : Type}
variable (A' B' C' : Type)

def triangle (A B C : Type) : Prop := true

def congruent (t1 t2 : Prop) : Prop := true

variable (P : ℕ)

def perimeter (t : Prop) (p : ℕ) : Prop := true

def length (a b : Type) (l : ℕ) : Prop := true

theorem triangle_congruence_example :
  ∀ (A B C A' B' C' : Type) (h_cong : congruent (triangle A B C) (triangle A' B' C'))
    (h_perimeter : perimeter (triangle A B C) 20)
    (h_AB : length A B 8)
    (h_BC : length B C 5),
    length A C 7 :=
by sorry

end triangle_congruence_example_l224_224133


namespace work_b_alone_l224_224520

theorem work_b_alone (a b : ℕ) (h1 : 2 * b = a) (h2 : a + b = 3) (h3 : (a + b) * 11 = 33) : 33 = 33 :=
by
  -- sorry is used here because we are skipping the actual proof
  sorry

end work_b_alone_l224_224520


namespace triangle_d_not_right_l224_224303

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_d_not_right :
  ¬is_right_triangle 7 8 13 :=
by sorry

end triangle_d_not_right_l224_224303


namespace amount_tom_should_pay_l224_224314

theorem amount_tom_should_pay (original_price : ℝ) (multiplier : ℝ) 
  (h1 : original_price = 3) (h2 : multiplier = 3) : 
  original_price * multiplier = 9 :=
sorry

end amount_tom_should_pay_l224_224314


namespace number_of_subsets_l224_224230

theorem number_of_subsets (P : Finset ℤ) (h : P = {-1, 0, 1}) : P.powerset.card = 8 := 
by
  rw [h]
  sorry

end number_of_subsets_l224_224230


namespace x4_value_l224_224572

/-- Define x_n sequence based on given initial value and construction rules -/
def x_n (n : ℕ) : ℕ :=
  if n = 1 then 27
  else if n = 2 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1
  else if n = 3 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1 -- Need to generalize for actual sequence definition
  else if n = 4 then 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  else 0 -- placeholder for general case (not needed here)

/-- Prove that x_4 = 23 given x_1=27 and the sequence construction criteria --/
theorem x4_value : x_n 4 = 23 :=
by
  -- Proof not required, hence sorry is used
  sorry

end x4_value_l224_224572


namespace find_m_perpendicular_l224_224660

-- Define the two vectors
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1, 2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Theorem stating the mathematically equivalent proof problem
theorem find_m_perpendicular (m : ℝ) (h : dot_product (a m) b = 0) : m = 2 :=
by sorry

end find_m_perpendicular_l224_224660


namespace multiplication_results_l224_224096

theorem multiplication_results
  (h1 : 25 * 4 = 100) :
  25 * 8 = 200 ∧ 25 * 12 = 300 ∧ 250 * 40 = 10000 ∧ 25 * 24 = 600 :=
by
  sorry

end multiplication_results_l224_224096


namespace rods_in_one_mile_l224_224611

theorem rods_in_one_mile (mile_to_furlong : ℕ) (furlong_to_rod : ℕ) (mile_eq : 1 = 8 * mile_to_furlong) (furlong_eq: 1 = 50 * furlong_to_rod) : 
  (1 * 8 * 50 = 400) :=
by
  sorry

end rods_in_one_mile_l224_224611


namespace total_students_accommodated_l224_224371

structure BusConfig where
  columns : ℕ
  rows : ℕ
  broken_seats : ℕ

structure SplitBusConfig where
  columns : ℕ
  left_rows : ℕ
  right_rows : ℕ
  broken_seats : ℕ

structure ComplexBusConfig where
  columns : ℕ
  rows : ℕ
  special_rows_broken_seats : ℕ

def bus1 : BusConfig := { columns := 4, rows := 10, broken_seats := 2 }
def bus2 : BusConfig := { columns := 5, rows := 8, broken_seats := 4 }
def bus3 : BusConfig := { columns := 3, rows := 12, broken_seats := 3 }
def bus4 : SplitBusConfig := { columns := 4, left_rows := 6, right_rows := 8, broken_seats := 1 }
def bus5 : SplitBusConfig := { columns := 6, left_rows := 8, right_rows := 10, broken_seats := 5 }
def bus6 : ComplexBusConfig := { columns := 5, rows := 10, special_rows_broken_seats := 4 }

theorem total_students_accommodated :
  let seats_bus1 := (bus1.columns * bus1.rows) - bus1.broken_seats;
  let seats_bus2 := (bus2.columns * bus2.rows) - bus2.broken_seats;
  let seats_bus3 := (bus3.columns * bus3.rows) - bus3.broken_seats;
  let seats_bus4 := (bus4.columns * bus4.left_rows) + (bus4.columns * bus4.right_rows) - bus4.broken_seats;
  let seats_bus5 := (bus5.columns * bus5.left_rows) + (bus5.columns * bus5.right_rows) - bus5.broken_seats;
  let seats_bus6 := (bus6.columns * bus6.rows) - bus6.special_rows_broken_seats;
  seats_bus1 + seats_bus2 + seats_bus3 + seats_bus4 + seats_bus5 + seats_bus6 = 311 :=
sorry

end total_students_accommodated_l224_224371


namespace estimated_white_balls_is_correct_l224_224875

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of trials
def trials : ℕ := 100

-- Define the number of times a red ball is drawn
def red_draws : ℕ := 80

-- Define the function to estimate the number of red balls based on the frequency
def estimated_red_balls (total_balls : ℕ) (red_draws : ℕ) (trials : ℕ) : ℕ :=
  total_balls * red_draws / trials

-- Define the function to estimate the number of white balls
def estimated_white_balls (total_balls : ℕ) (estimated_red_balls : ℕ) : ℕ :=
  total_balls - estimated_red_balls

-- State the theorem to prove the estimated number of white balls
theorem estimated_white_balls_is_correct : 
  estimated_white_balls total_balls (estimated_red_balls total_balls red_draws trials) = 2 :=
by
  sorry

end estimated_white_balls_is_correct_l224_224875


namespace extreme_points_inequality_l224_224304

noncomputable def f (a x : ℝ) : ℝ := a * x - (a / x) - 2 * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := (a * x^2 - 2 * x + a) / x^2

theorem extreme_points_inequality (a x1 x2 : ℝ) (h1 : a > 0) (h2 : 1 < x1) (h3 : x1 < Real.exp 1)
  (h4 : f a x1 = 0) (h5 : f a x2 = 0) (h6 : x1 ≠ x2) : |f a x1 - f a x2| < 1 :=
by
  sorry

end extreme_points_inequality_l224_224304


namespace ice_cream_vendor_l224_224335

theorem ice_cream_vendor (M : ℕ) (h3 : 50 - (3 / 5) * 50 = 20) (h4 : (2 / 3) * M = 2 * M / 3) 
  (h5 : (50 - 30) + M - (2 * M / 3) = 38) :
  M = 12 :=
by
  sorry

end ice_cream_vendor_l224_224335


namespace product_of_five_consecutive_divisible_by_30_l224_224161

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l224_224161


namespace problems_per_page_l224_224367

theorem problems_per_page (pages_math pages_reading total_problems x : ℕ) (h1 : pages_math = 2) (h2 : pages_reading = 4) (h3 : total_problems = 30) : 
  (pages_math + pages_reading) * x = total_problems → x = 5 := by
  sorry

end problems_per_page_l224_224367


namespace select_1996_sets_l224_224710

theorem select_1996_sets (k : ℕ) (sets : Finset (Finset ℕ)) (h : k > 1993006) (h_sets : sets.card = k) :
  ∃ (selected_sets : Finset (Finset ℕ)), selected_sets.card = 1996 ∧
  ∀ (x y z : Finset ℕ), x ∈ selected_sets → y ∈ selected_sets → z ∈ selected_sets → z = x ∪ y → false :=
sorry

end select_1996_sets_l224_224710


namespace rebecca_income_percentage_l224_224516

-- Define Rebecca's initial income
def rebecca_initial_income : ℤ := 15000
-- Define Jimmy's income
def jimmy_income : ℤ := 18000
-- Define the increase in Rebecca's income
def rebecca_income_increase : ℤ := 7000

-- Define the new income for Rebecca after increase
def rebecca_new_income : ℤ := rebecca_initial_income + rebecca_income_increase
-- Define the new combined income
def new_combined_income : ℤ := rebecca_new_income + jimmy_income

-- State the theorem to prove that Rebecca's new income is 55% of the new combined income
theorem rebecca_income_percentage : 
  (rebecca_new_income * 100) / new_combined_income = 55 :=
sorry

end rebecca_income_percentage_l224_224516


namespace john_pays_in_30_day_month_l224_224728

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end john_pays_in_30_day_month_l224_224728


namespace simplify_expression_l224_224341

theorem simplify_expression: 3 * Real.sqrt 48 - 6 * Real.sqrt (1 / 3) + (Real.sqrt 3 - 1) ^ 2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end simplify_expression_l224_224341


namespace soccer_team_points_l224_224997

theorem soccer_team_points
  (x y : ℕ)
  (h1 : x + y = 8)
  (h2 : 3 * x - y = 12) : 
  (x + y = 8 ∧ 3 * x - y = 12) :=
by
  exact ⟨h1, h2⟩

end soccer_team_points_l224_224997


namespace company_fund_initial_amount_l224_224172

theorem company_fund_initial_amount
  (n : ℕ) -- number of employees
  (initial_bonus_per_employee : ℕ := 60)
  (shortfall : ℕ := 10)
  (revised_bonus_per_employee : ℕ := 50)
  (fund_remaining : ℕ := 150)
  (initial_fund : ℕ := initial_bonus_per_employee * n - shortfall) -- condition that the fund was $10 short when planning the initial bonus
  (revised_fund : ℕ := revised_bonus_per_employee * n + fund_remaining) -- condition after distributing the $50 bonuses

  (eqn : initial_fund = revised_fund) -- equating initial and revised budget calculations
  
  : initial_fund = 950 := 
sorry

end company_fund_initial_amount_l224_224172


namespace unique_fraction_difference_l224_224964

theorem unique_fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  (1 / x) - (1 / y) = (y - x) / (x * y) :=
by sorry

end unique_fraction_difference_l224_224964


namespace females_in_group_l224_224080

theorem females_in_group (n F M : ℕ) (Index_F Index_M : ℝ) 
  (h1 : n = 25) 
  (h2 : Index_F = (n - F) / n)
  (h3 : Index_M = (n - M) / n) 
  (h4 : Index_F - Index_M = 0.36) :
  F = 8 := 
by
  sorry

end females_in_group_l224_224080


namespace pages_left_to_write_l224_224319

theorem pages_left_to_write : 
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  remaining_pages = 315 :=
by
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  show remaining_pages = 315
  sorry

end pages_left_to_write_l224_224319


namespace range_of_a_l224_224445

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (|x - 1| - |x - 3|) > a) → a < 2 :=
by
  sorry

end range_of_a_l224_224445


namespace probability_rain_at_most_3_days_in_july_l224_224869

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end probability_rain_at_most_3_days_in_july_l224_224869


namespace simplify_expression_l224_224588

variable (x : ℝ) (hx : x ≠ 0)

theorem simplify_expression : 
  ( (x + 3)^2 + (x + 3) * (x - 3) ) / (2 * x) = x + 3 := by
  sorry

end simplify_expression_l224_224588


namespace each_child_plays_equally_l224_224521

theorem each_child_plays_equally (total_time : ℕ) (num_children : ℕ)
  (play_group_size : ℕ) (play_time : ℕ) :
  num_children = 6 ∧ play_group_size = 3 ∧ total_time = 120 ∧ play_time = (total_time * play_group_size) / num_children →
  play_time = 60 :=
by
  intros h
  sorry

end each_child_plays_equally_l224_224521


namespace total_cost_correct_l224_224767

-- Definitions of the constants based on given problem conditions
def cost_burger : ℕ := 5
def cost_pack_of_fries : ℕ := 2
def num_packs_of_fries : ℕ := 2
def cost_salad : ℕ := 3 * cost_pack_of_fries

-- The total cost calculation based on the conditions
def total_cost : ℕ := cost_burger + num_packs_of_fries * cost_pack_of_fries + cost_salad

-- The statement to prove that the total cost Benjamin paid is $15
theorem total_cost_correct : total_cost = 15 := by
  -- This is where the proof would go, but we're omitting it for now.
  sorry

end total_cost_correct_l224_224767


namespace plane_through_point_and_line_l224_224392

noncomputable def point_on_plane (A B C D : ℤ) (x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

def line_eq_1 (x y : ℤ) : Prop :=
  3 * x + 4 * y - 20 = 0

def line_eq_2 (y z : ℤ) : Prop :=
  -3 * y + 2 * z + 18 = 0

theorem plane_through_point_and_line 
  (A B C D : ℤ)
  (h_point : point_on_plane A B C D 1 9 (-8))
  (h_line1 : ∀ x y, line_eq_1 x y → point_on_plane A B C D x y 0)
  (h_line2 : ∀ y z, line_eq_2 y z → point_on_plane A B C D 0 y z)
  (h_gcd : Int.gcd (Int.gcd (Int.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1) 
  (h_pos : A > 0) :
  A = 75 ∧ B = -29 ∧ C = 86 ∧ D = 274 :=
sorry

end plane_through_point_and_line_l224_224392


namespace sum_p_q_l224_224277

-- Define the cubic polynomial q(x)
def cubic_q (q : ℚ) (x : ℚ) := q * x * (x - 1) * (x + 1)

-- Define the linear polynomial p(x)
def linear_p (p : ℚ) (x : ℚ) := p * x

-- Prove the result for p(x) + q(x)
theorem sum_p_q : 
  (∀ p q : ℚ, linear_p p 4 = 4 → cubic_q q 3 = 3 → (∀ x : ℚ, linear_p p x + cubic_q q x = (1 / 24) * x^3 + (23 / 24) * x)) :=
by
  intros p q hp hq x
  sorry

end sum_p_q_l224_224277


namespace angle_between_sides_of_triangle_l224_224127

noncomputable def right_triangle_side_lengths1 : Nat × Nat × Nat := (15, 36, 39)
noncomputable def right_triangle_side_lengths2 : Nat × Nat × Nat := (40, 42, 58)

-- Assuming both triangles are right triangles
def is_right_triangle (a b c : Nat) : Prop := a^2 + b^2 = c^2

theorem angle_between_sides_of_triangle
  (h1 : is_right_triangle 15 36 39)
  (h2 : is_right_triangle 40 42 58) : 
  ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end angle_between_sides_of_triangle_l224_224127


namespace calculate_expression_l224_224143

theorem calculate_expression : (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := 
by 
  sorry

end calculate_expression_l224_224143


namespace power_of_power_example_l224_224434

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l224_224434


namespace solve_for_y_l224_224013


theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
    Matrix.det ![
        ![y + b, y, y],
        ![y, y + b, y],
        ![y, y, y + b]] = 0 → y = -b := by
  sorry

end solve_for_y_l224_224013


namespace pi_is_irrational_l224_224345

theorem pi_is_irrational (π : ℝ) (h : π = Real.pi) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ π = a / b :=
by
  sorry

end pi_is_irrational_l224_224345


namespace find_m_l224_224360

def f (x m : ℝ) : ℝ := x ^ 2 - 3 * x + m
def g (x m : ℝ) : ℝ := 2 * x ^ 2 - 6 * x + 5 * m

theorem find_m (m : ℝ) (h : 3 * f 3 m = 2 * g 3 m) : m = 0 :=
by sorry

end find_m_l224_224360


namespace rowing_distance_l224_224490

def man_rowing_speed_still_water : ℝ := 10
def stream_speed : ℝ := 8
def rowing_time_downstream : ℝ := 5
def effective_speed_downstream : ℝ := man_rowing_speed_still_water + stream_speed

theorem rowing_distance :
  effective_speed_downstream * rowing_time_downstream = 90 := 
by 
  sorry

end rowing_distance_l224_224490


namespace determine_numbers_l224_224269

theorem determine_numbers (A B n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 9) (h2 : A = 10 * B + n) (h3 : A + B = 2022) : 
  A = 1839 ∧ B = 183 :=
by
  -- proof will be filled in here
  sorry

end determine_numbers_l224_224269


namespace partners_count_l224_224833

theorem partners_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 :=
sorry

end partners_count_l224_224833


namespace distance_sum_l224_224124

theorem distance_sum (a : ℝ) (x y : ℝ) 
  (AB CD : ℝ) (A B C D P Q M N : ℝ)
  (h_AB : AB = 4) (h_CD : CD = 8) 
  (h_M_AB : M = (A + B) / 2) (h_N_CD : N = (C + D) / 2)
  (h_P_AB : P ∈ [A, B]) (h_Q_CD : Q ∈ [C, D])
  (h_x : x = dist P M) (h_y : y = dist Q N)
  (h_y_eq_2x : y = 2 * x) (h_x_eq_a : x = a) :
  x + y = 3 * a := 
by
  sorry

end distance_sum_l224_224124


namespace sin_double_angle_l224_224405

open Real 

theorem sin_double_angle (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4) 
  (h4 : cos (α - β) = 12 / 13) 
  (h5 : sin (α + β) = -3 / 5) : 
  sin (2 * α) = -56 / 65 := 
by 
  sorry

end sin_double_angle_l224_224405


namespace find_reciprocal_sum_l224_224400

theorem find_reciprocal_sum
  (m n : ℕ)
  (h_sum : m + n = 72)
  (h_hcf : Nat.gcd m n = 6)
  (h_lcm : Nat.lcm m n = 210) :
  (1 / (m : ℚ)) + (1 / (n : ℚ)) = 6 / 105 :=
by
  sorry

end find_reciprocal_sum_l224_224400


namespace minimize_quadratic_l224_224603

theorem minimize_quadratic : ∃ x : ℝ, x = -4 ∧ ∀ y : ℝ, x^2 + 8*x + 7 ≤ y^2 + 8*y + 7 :=
by 
  use -4
  sorry

end minimize_quadratic_l224_224603


namespace intersection_A_B_l224_224680

def A : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}
def B : Set ℝ := {x | x * (x + 1) ≥ 0}

theorem intersection_A_B :
  (A ∩ B) = {x | (0 ≤ x ∧ x ≤ 1) ∨ x = -1} :=
  sorry

end intersection_A_B_l224_224680


namespace find_tangent_point_l224_224569

theorem find_tangent_point (x : ℝ) (y : ℝ) (h_curve : y = x^2) (h_slope : 2 * x = 1) : 
    (x, y) = (1/2, 1/4) :=
sorry

end find_tangent_point_l224_224569


namespace ken_gets_back_16_dollars_l224_224844

-- Given constants and conditions
def price_per_pound_steak : ℕ := 7
def pounds_of_steak : ℕ := 2
def price_carton_eggs : ℕ := 3
def price_gallon_milk : ℕ := 4
def price_pack_bagels : ℕ := 6
def bill_20_dollar : ℕ := 20
def bill_10_dollar : ℕ := 10
def bill_5_dollar_count : ℕ := 2
def coin_1_dollar_count : ℕ := 3

-- Calculate total cost of items
def total_cost_items : ℕ :=
  (pounds_of_steak * price_per_pound_steak) +
  price_carton_eggs +
  price_gallon_milk +
  price_pack_bagels

-- Calculate total amount paid
def total_amount_paid : ℕ :=
  bill_20_dollar +
  bill_10_dollar +
  (bill_5_dollar_count * 5) +
  (coin_1_dollar_count * 1)

-- Theorem statement to be proved
theorem ken_gets_back_16_dollars :
  total_amount_paid - total_cost_items = 16 := by
  sorry

end ken_gets_back_16_dollars_l224_224844


namespace negation_prop_l224_224897

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end negation_prop_l224_224897


namespace smallest_k_l224_224433

def u (n : ℕ) : ℕ := n^4 + 3 * n^2 + 2

def delta (k : ℕ) (u : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => u
  | k+1 => fun n => delta k u (n+1) - delta k u n

theorem smallest_k (n : ℕ) : ∃ k, (forall m, delta k u m = 0) ∧ 
                            (forall j, (∀ m, delta j u m = 0) → j ≥ k) := sorry

end smallest_k_l224_224433


namespace perpendicular_MP_MQ_l224_224185

variable (k m : ℝ)

def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

def line (x y : ℝ) := y = k*x + m

def fixed_point_exists (k m : ℝ) : Prop :=
  let P := (-(4 * k) / m, 3 / m)
  let Q := (4, 4 * k + m)
  ∃ (M : ℝ), (M = 1 ∧ ((P.1 - M) * (Q.1 - M) + P.2 * Q.2 = 0))

theorem perpendicular_MP_MQ : fixed_point_exists k m := sorry

end perpendicular_MP_MQ_l224_224185


namespace Tim_total_money_l224_224547

theorem Tim_total_money :
  let nickels_amount := 3 * 0.05
  let dimes_amount_shoes := 13 * 0.10
  let shining_shoes := nickels_amount + dimes_amount_shoes
  let dimes_amount_tip_jar := 7 * 0.10
  let half_dollars_amount := 9 * 0.50
  let tip_jar := dimes_amount_tip_jar + half_dollars_amount
  let total := shining_shoes + tip_jar
  total = 6.65 :=
by
  sorry

end Tim_total_money_l224_224547


namespace hannah_stocking_stuffers_l224_224829

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l224_224829


namespace sum_of_integer_n_l224_224316

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l224_224316


namespace determine_amount_of_substance_l224_224545

noncomputable def amount_of_substance 
  (A : ℝ) (R : ℝ) (delta_T : ℝ) : ℝ :=
  (2 * A) / (R * delta_T)

theorem determine_amount_of_substance 
  (A : ℝ := 831) 
  (R : ℝ := 8.31) 
  (delta_T : ℝ := 100) 
  (nu : ℝ := amount_of_substance A R delta_T) :
  nu = 2 := by
  -- Conditions rewritten as definitions
  -- Definition: A = 831 J
  -- Definition: R = 8.31 J/(mol * K)
  -- Definition: delta_T = 100 K
  -- The correct answer to be proven: nu = 2 mol
  sorry

end determine_amount_of_substance_l224_224545


namespace gcd_of_repeated_three_digit_l224_224245

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end gcd_of_repeated_three_digit_l224_224245


namespace price_of_silver_l224_224691

theorem price_of_silver
  (side : ℕ) (side_eq : side = 3)
  (weight_per_cubic_inch : ℕ) (weight_per_cubic_inch_eq : weight_per_cubic_inch = 6)
  (selling_price : ℝ) (selling_price_eq : selling_price = 4455)
  (markup_percentage : ℝ) (markup_percentage_eq : markup_percentage = 1.10)
  : 4050 / 162 = 25 :=
by
  -- Given conditions are side_eq, weight_per_cubic_inch_eq, selling_price_eq, and markup_percentage_eq
  -- The statement requiring proof, i.e., price per ounce calculation, is provided.
  sorry

end price_of_silver_l224_224691


namespace xy_sum_one_l224_224979

theorem xy_sum_one (x y : ℝ) (h : x > 0) (k : y > 0) (hx : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : x + y = 1 :=
sorry

end xy_sum_one_l224_224979


namespace savings_by_december_l224_224570

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l224_224570


namespace inequality_proof_l224_224241

theorem inequality_proof (a b c : ℝ) (ha1 : 0 ≤ a) (ha2 : a ≤ 1) (hb1 : 0 ≤ b) (hb2 : b ≤ 1) (hc1 : 0 ≤ c) (hc2 : c ≤ 1) :
  (a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by
  sorry

end inequality_proof_l224_224241


namespace a3_value_l224_224967

-- Define the geometric sequence
def geom_seq (r : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

-- Given conditions
variables (a : ℕ → ℝ) (r : ℝ)
axiom h_geom : geom_seq r a
axiom h_a1 : a 1 = 1
axiom h_a5 : a 5 = 4

-- Goal to prove
theorem a3_value : a 3 = 2 ∨ a 3 = -2 := by
  sorry

end a3_value_l224_224967


namespace cost_of_apple_is_two_l224_224170

-- Define the costs and quantities
def cost_of_apple (A : ℝ) : Prop :=
  let total_cost := 12 * A + 4 * 1 + 4 * 3
  let total_pieces := 12 + 4 + 4
  let average_cost := 2
  total_cost = total_pieces * average_cost

theorem cost_of_apple_is_two : cost_of_apple 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cost_of_apple_is_two_l224_224170


namespace turtle_reaches_waterhole_28_minutes_after_meeting_l224_224561

theorem turtle_reaches_waterhole_28_minutes_after_meeting (x : ℝ) (distance_lion1 : ℝ := 5 * x) 
  (speed_lion2 : ℝ := 1.5 * x) (distance_turtle : ℝ := 30) (speed_turtle : ℝ := 1/30) : 
  ∃ t_meeting : ℝ, t_meeting = 2 ∧ (distance_turtle - speed_turtle * t_meeting) / speed_turtle = 28 :=
by 
  sorry

end turtle_reaches_waterhole_28_minutes_after_meeting_l224_224561


namespace bus_speed_excluding_stoppages_l224_224102

theorem bus_speed_excluding_stoppages (v : ℝ) 
  (speed_including_stoppages : ℝ := 45) 
  (stoppage_time : ℝ := 1/6) 
  (h : v * (1 - stoppage_time) = speed_including_stoppages) : 
  v = 54 := 
by 
  sorry

end bus_speed_excluding_stoppages_l224_224102


namespace base7_to_base10_l224_224776

theorem base7_to_base10 : 6 * 7^3 + 4 * 7^2 + 2 * 7^1 + 3 * 7^0 = 2271 := by
  sorry

end base7_to_base10_l224_224776


namespace num_positive_solutions_eq_32_l224_224475

theorem num_positive_solutions_eq_32 : 
  ∃ n : ℕ, (∀ x y : ℕ, 4 * x + 7 * y = 888 → x > 0 ∧ y > 0) ∧ n = 32 :=
sorry

end num_positive_solutions_eq_32_l224_224475


namespace sequence_inequality_l224_224693

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + m) ≤ a n + a m)
  (h2 : ∀ n : ℕ, 0 ≤ a n) (n m : ℕ) (hnm : n ≥ m) : 
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end sequence_inequality_l224_224693


namespace minimum_value_of_expression_l224_224020

theorem minimum_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 := 
sorry

end minimum_value_of_expression_l224_224020


namespace caterpillar_length_difference_l224_224249

-- Define the lengths of the caterpillars
def green_caterpillar_length : ℝ := 3
def orange_caterpillar_length : ℝ := 1.17

-- State the theorem we need to prove
theorem caterpillar_length_difference :
  green_caterpillar_length - orange_caterpillar_length = 1.83 :=
by
  sorry

end caterpillar_length_difference_l224_224249


namespace triangle_altitude_l224_224972

variable (Area : ℝ) (base : ℝ) (altitude : ℝ)

theorem triangle_altitude (hArea : Area = 1250) (hbase : base = 50) :
  2 * Area / base = altitude :=
by
  sorry

end triangle_altitude_l224_224972


namespace simplify_expr1_simplify_expr2_l224_224920

-- Defining the necessary variables as real numbers for the proof
variables (x y : ℝ)

-- Prove the first expression simplification
theorem simplify_expr1 : 
  (x + 2 * y) * (x - 2 * y) - x * (x + 3 * y) = -4 * y^2 - 3 * x * y :=
  sorry

-- Prove the second expression simplification
theorem simplify_expr2 : 
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4 * x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
  sorry

end simplify_expr1_simplify_expr2_l224_224920


namespace manu_wins_probability_l224_224163

def prob_manu_wins : ℚ :=
  let a := (1/2) ^ 5
  let r := (1/2) ^ 4
  a / (1 - r)

theorem manu_wins_probability : prob_manu_wins = 1 / 30 :=
  by
  -- here we would have the proof steps
  sorry

end manu_wins_probability_l224_224163


namespace find_y_l224_224747

-- Let s be the result of tripling both the base and exponent of c^d
-- Given the condition s = c^d * y^d, we need to prove y = 27c^2

variable (c d y : ℝ)
variable (h_d : d > 0)
variable (h : (3 * c)^(3 * d) = c^d * y^d)

theorem find_y (h_d : d > 0) (h : (3 * c)^(3 * d) = c^d * y^d) : y = 27 * c ^ 2 :=
by sorry

end find_y_l224_224747


namespace train_crossing_time_l224_224531

noncomputable def length_first_train : ℝ := 200  -- meters
noncomputable def speed_first_train_kmph : ℝ := 72  -- km/h
noncomputable def speed_first_train : ℝ := speed_first_train_kmph * (1000 / 3600)  -- m/s

noncomputable def length_second_train : ℝ := 300  -- meters
noncomputable def speed_second_train_kmph : ℝ := 36  -- km/h
noncomputable def speed_second_train : ℝ := speed_second_train_kmph * (1000 / 3600)  -- m/s

noncomputable def relative_speed : ℝ := speed_first_train - speed_second_train -- m/s
noncomputable def total_length : ℝ := length_first_train + length_second_train  -- meters
noncomputable def time_to_cross : ℝ := total_length / relative_speed  -- seconds

theorem train_crossing_time :
  time_to_cross = 50 := by
  sorry

end train_crossing_time_l224_224531


namespace union_sets_l224_224773

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

theorem union_sets : A ∪ B = {x | x ≥ 1} :=
  by
    sorry

end union_sets_l224_224773


namespace intersection_of_sets_l224_224782

noncomputable def setA : Set ℝ := {x | 1 / (x - 1) ≤ 1}
def setB : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_sets : setA ∩ setB = {-1, 0, 2} := 
by
  sorry

end intersection_of_sets_l224_224782


namespace hcf_of_210_and_671_l224_224692

theorem hcf_of_210_and_671 :
  let lcm := 2310
  let a := 210
  let b := 671
  gcd a b = 61 :=
by
  let lcm := 2310
  let a := 210
  let b := 671
  let hcf := gcd a b
  have rel : lcm * hcf = a * b := by sorry
  have hcf_eq : hcf = 61 := by sorry
  exact hcf_eq

end hcf_of_210_and_671_l224_224692


namespace common_chord_eqn_l224_224546

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 12 * x - 2 * y - 13 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 12 * x + 16 * y - 25 = 0

-- Define the proposition stating the common chord equation
theorem common_chord_eqn : ∀ x y : ℝ, C1 x y ∧ C2 x y → 4 * x + 3 * y - 2 = 0 :=
by
  sorry

end common_chord_eqn_l224_224546


namespace B_oxen_count_l224_224044

/- 
  A puts 10 oxen for 7 months.
  B puts some oxen for 5 months.
  C puts 15 oxen for 3 months.
  The rent of the pasture is Rs. 175.
  C should pay Rs. 45 as his share of rent.
  We need to prove that B put 12 oxen for grazing.
-/

def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

def A_ox_months := oxen_months 10 7
def C_ox_months := oxen_months 15 3

def total_rent : ℕ := 175
def C_rent_share : ℕ := 45

theorem B_oxen_count (x : ℕ) : 
  (C_rent_share : ℝ) / total_rent = (C_ox_months : ℝ) / (A_ox_months + 5 * x + C_ox_months) →
  x = 12 := 
by
  sorry

end B_oxen_count_l224_224044


namespace cubic_inches_needed_l224_224537

/-- The dimensions of each box are 20 inches by 20 inches by 12 inches. -/
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

/-- The cost of each box is $0.40. -/
def box_cost : ℝ := 0.40

/-- The minimum spending required by the university on boxes is $200. -/
def min_spending : ℝ := 200

/-- Given the above conditions, the total cubic inches needed to package the collection is 2,400,000 cubic inches. -/
theorem cubic_inches_needed :
  (min_spending / box_cost) * (box_length * box_width * box_height) = 2400000 := by
  sorry

end cubic_inches_needed_l224_224537


namespace three_digit_number_108_l224_224471

theorem three_digit_number_108 (a b c : ℕ) (ha : a ≠ 0) (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10) (h₃: 100*a + 10*b + c = 12*(a + b + c)) : 
  100*a + 10*b + c = 108 := 
by 
  sorry

end three_digit_number_108_l224_224471


namespace adoption_days_l224_224663

def initial_puppies : ℕ := 15
def additional_puppies : ℕ := 62
def adoption_rate : ℕ := 7

def total_puppies : ℕ := initial_puppies + additional_puppies

theorem adoption_days :
  total_puppies / adoption_rate = 11 :=
by
  sorry

end adoption_days_l224_224663


namespace wholesome_bakery_loaves_on_wednesday_l224_224029

theorem wholesome_bakery_loaves_on_wednesday :
  ∀ (L_wed L_thu L_fri L_sat L_sun L_mon : ℕ),
    L_thu = 7 →
    L_fri = 10 →
    L_sat = 14 →
    L_sun = 19 →
    L_mon = 25 →
    L_thu - L_wed = 2 →
    L_wed = 5 :=
by intros L_wed L_thu L_fri L_sat L_sun L_mon;
   intros H_thu H_fri H_sat H_sun H_mon H_diff;
   sorry

end wholesome_bakery_loaves_on_wednesday_l224_224029


namespace sum_of_powers_sequence_l224_224251

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sum_of_powers_sequence_l224_224251


namespace determine_linear_relation_l224_224193

-- Define the set of options
inductive PlotType
| Scatter
| StemAndLeaf
| FrequencyHistogram
| FrequencyLineChart

-- Define the question and state the expected correct answer
def correctPlotTypeForLinearRelation : PlotType :=
  PlotType.Scatter

-- Prove that the correct method for determining linear relation in a set of data is a Scatter plot
theorem determine_linear_relation :
  correctPlotTypeForLinearRelation = PlotType.Scatter :=
by
  sorry

end determine_linear_relation_l224_224193


namespace find_angle_A_l224_224014
open Real

theorem find_angle_A
  (a b : ℝ)
  (A B : ℝ)
  (h1 : b = 2 * a)
  (h2 : B = A + 60) :
  A = 30 :=
by 
  sorry

end find_angle_A_l224_224014


namespace seating_arrangement_l224_224672

theorem seating_arrangement : 
  ∃ x y z : ℕ, 
  7 * x + 8 * y + 9 * z = 65 ∧ z = 1 ∧ x + y + z = r :=
sorry

end seating_arrangement_l224_224672


namespace arrange_letters_l224_224488

-- Definitions based on conditions
def total_letters := 6
def identical_bs := 2 -- Number of B's that are identical
def distinct_as := 3  -- Number of A's that are distinct
def distinct_ns := 1  -- Number of N's that are distinct

-- Now formulate the proof statement
theorem arrange_letters :
    (Nat.factorial total_letters) / (Nat.factorial identical_bs) = 360 :=
by
  sorry

end arrange_letters_l224_224488


namespace mr_green_expects_expected_potatoes_yield_l224_224482

theorem mr_green_expects_expected_potatoes_yield :
  ∀ (length_steps width_steps: ℕ) (step_length yield_per_sqft: ℝ),
  length_steps = 18 →
  width_steps = 25 →
  step_length = 2.5 →
  yield_per_sqft = 0.75 →
  (length_steps * step_length) * (width_steps * step_length) * yield_per_sqft = 2109.375 :=
by
  intros length_steps width_steps step_length yield_per_sqft
  intros h_length_steps h_width_steps h_step_length h_yield_per_sqft
  rw [h_length_steps, h_width_steps, h_step_length, h_yield_per_sqft]
  sorry

end mr_green_expects_expected_potatoes_yield_l224_224482


namespace top_card_yellow_second_card_not_yellow_l224_224062

-- Definitions based on conditions
def total_cards : Nat := 65

def yellow_cards : Nat := 13

def non_yellow_cards : Nat := total_cards - yellow_cards

-- Total combinations of choosing two cards
def total_combinations : Nat := total_cards * (total_cards - 1)

-- Numerator for desired probability 
def desired_combinations : Nat := yellow_cards * non_yellow_cards

-- Target probability
def desired_probability : Rat := Rat.ofInt (desired_combinations) / Rat.ofInt (total_combinations)

-- Mathematical proof statement
theorem top_card_yellow_second_card_not_yellow :
  desired_probability = Rat.ofInt 169 / Rat.ofInt 1040 :=
by
  sorry

end top_card_yellow_second_card_not_yellow_l224_224062


namespace cat_finishes_food_on_tuesday_second_week_l224_224785

def initial_cans : ℚ := 8
def extra_treat : ℚ := 1 / 6
def morning_diet : ℚ := 1 / 4
def evening_diet : ℚ := 1 / 5

def daily_consumption (morning_diet evening_diet : ℚ) : ℚ :=
  morning_diet + evening_diet

def first_day_consumption (daily_consumption extra_treat : ℚ) : ℚ :=
  daily_consumption + extra_treat

theorem cat_finishes_food_on_tuesday_second_week 
  (initial_cans extra_treat morning_diet evening_diet : ℚ)
  (h1 : initial_cans = 8)
  (h2 : extra_treat = 1 / 6)
  (h3 : morning_diet = 1 / 4)
  (h4 : evening_diet = 1 / 5) :
  -- The computation must be performed here or defined previously
  -- The proof of this theorem is the task, the result is postulated as a theorem
  final_day = "Tuesday (second week)" :=
sorry

end cat_finishes_food_on_tuesday_second_week_l224_224785


namespace find_ten_x_l224_224613

theorem find_ten_x (x : ℝ) 
  (h : 4^(2*x) + 2^(-x) + 1 = (129 + 8 * Real.sqrt 2) * (4^x + 2^(- x) - 2^x)) : 
  10 * x = 35 := 
sorry

end find_ten_x_l224_224613


namespace line_through_points_eq_l224_224751

theorem line_through_points_eq
  (x1 y1 x2 y2 : ℝ)
  (h1 : 2 * x1 + 3 * y1 = 4)
  (h2 : 2 * x2 + 3 * y2 = 4) :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) ↔ (2 * x + 3 * y = 4)) :=
by
  sorry

end line_through_points_eq_l224_224751


namespace subset_A_B_l224_224076

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l224_224076


namespace coordinates_of_point_in_fourth_quadrant_l224_224840

-- Define the conditions as separate hypotheses
def point_in_fourth_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- State the main theorem
theorem coordinates_of_point_in_fourth_quadrant
  (x y : ℝ) (h1 : point_in_fourth_quadrant x y) (h2 : |x| = 3) (h3 : |y| = 5) :
  (x = 3) ∧ (y = -5) :=
by
  sorry

end coordinates_of_point_in_fourth_quadrant_l224_224840


namespace trigonometric_identity_l224_224247

-- The main statement to prove
theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 :=
by
  sorry

end trigonometric_identity_l224_224247


namespace four_real_solutions_l224_224175

-- Definitions used in the problem
def P (x : ℝ) : Prop := (6 * x) / (x^2 + 2 * x + 5) + (4 * x) / (x^2 - 4 * x + 5) = -2 / 3

-- Statement of the problem
theorem four_real_solutions : ∃ (x1 x2 x3 x4 : ℝ), P x1 ∧ P x2 ∧ P x3 ∧ P x4 ∧ 
  ∀ x, P x → (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :=
sorry

end four_real_solutions_l224_224175


namespace total_players_on_team_l224_224186

theorem total_players_on_team (M W : ℕ) (h1 : W = M + 2) (h2 : (M : ℝ) / W = 0.7777777777777778) : M + W = 16 :=
by 
  sorry

end total_players_on_team_l224_224186


namespace assign_questions_to_students_l224_224867

theorem assign_questions_to_students:
  ∃ (assignment : Fin 20 → Fin 20), 
  (∀ s : Fin 20, ∃ q1 q2 : Fin 20, (assignment s = q1 ∨ assignment s = q2) ∧ q1 ≠ q2 ∧ ∀ q : Fin 20, ∃ s1 s2 : Fin 20, (assignment s1 = q ∧ assignment s2 = q) ∧ s1 ≠ s2) :=
by
  sorry

end assign_questions_to_students_l224_224867


namespace denominator_of_fraction_l224_224381

theorem denominator_of_fraction (n : ℕ) (h1 : n = 20) (h2 : num = 35) (dec_value : ℝ) (h3 : dec_value = 2 / 10^n) : denom = 175 * 10^20 :=
by
  sorry

end denominator_of_fraction_l224_224381


namespace sonya_fell_times_l224_224065

theorem sonya_fell_times (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) :
  steven_falls = 3 →
  stephanie_falls = steven_falls + 13 →
  sonya_falls = 6 →
  sonya_falls = (stephanie_falls / 2) - 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at *
  sorry

end sonya_fell_times_l224_224065


namespace nested_sqrt_eq_two_l224_224218

theorem nested_sqrt_eq_two (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 :=
sorry

end nested_sqrt_eq_two_l224_224218


namespace candy_distribution_l224_224336

theorem candy_distribution (candy_total friends : ℕ) (candies : List ℕ) :
  candy_total = 47 ∧ friends = 5 ∧ List.length candies = friends ∧
  (∀ n ∈ candies, n = 9) → (47 % 5 = 2) :=
by
  sorry

end candy_distribution_l224_224336


namespace solve_a_l224_224034

theorem solve_a (a x : ℤ) (h₀ : x = 5) (h₁ : a * x - 8 = 20 + a) : a = 7 :=
by
  sorry

end solve_a_l224_224034


namespace negation_of_proposition_l224_224369

theorem negation_of_proposition :
  (¬ (∃ x_0 : ℝ, x_0 ≤ 0 ∧ x_0^2 ≥ 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end negation_of_proposition_l224_224369


namespace has_zero_when_a_gt_0_l224_224582

noncomputable def f (x a : ℝ) : ℝ :=
  x * Real.log (x - 1) - a

theorem has_zero_when_a_gt_0 (a : ℝ) (h : a > 0) :
  ∃ x0 : ℝ, f x0 a = 0 ∧ 2 < x0 :=
sorry

end has_zero_when_a_gt_0_l224_224582


namespace pythagorean_numbers_b_l224_224530

-- Define Pythagorean numbers and conditions
variable (a b c m : ℕ)
variable (h1 : a = 1/2 * m^2 - 1/2)
variable (h2 : c = 1/2 * m^2 + 1/2)
variable (h3 : m > 1 ∧ ¬ even m)

theorem pythagorean_numbers_b (h4 : c^2 = a^2 + b^2) : b = m :=
sorry

end pythagorean_numbers_b_l224_224530


namespace value_of_x_is_two_l224_224850

theorem value_of_x_is_two (x : ℝ) (h : x + x^3 = 10) : x = 2 :=
sorry

end value_of_x_is_two_l224_224850


namespace find_side_b_in_triangle_l224_224523

theorem find_side_b_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180) 
  (h3 : a + c = 8) 
  (h4 : a * c = 15) 
  (h5 : (b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos (B * Real.pi / 180))) : 
  b = Real.sqrt 19 := 
  by sorry

end find_side_b_in_triangle_l224_224523


namespace range_of_m_l224_224394

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def is_monotonically_decreasing_in_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_m (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : is_monotonically_decreasing_in_domain f (-2) 2) :
  ∀ m : ℝ, (f (1 - m) + f (1 - m^2) < 0) → -2 < m ∧ m < 1 :=
sorry

end range_of_m_l224_224394


namespace darla_total_payment_l224_224155

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end darla_total_payment_l224_224155


namespace total_amount_l224_224827

theorem total_amount (a b c : ℕ) (h1 : a * 5 = b * 3) (h2 : c * 5 = b * 9) (h3 : b = 50) :
  a + b + c = 170 := by
  sorry

end total_amount_l224_224827


namespace sum_series_l224_224550

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end sum_series_l224_224550


namespace fran_speed_l224_224957

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l224_224957


namespace divisibility_problem_l224_224235

theorem divisibility_problem :
  (2^62 + 1) % (2^31 + 2^16 + 1) = 0 := 
sorry

end divisibility_problem_l224_224235


namespace composite_sum_of_ab_l224_224295

theorem composite_sum_of_ab (a b : ℕ) (h : 31 * a = 54 * b) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ a + b = k * l :=
sorry

end composite_sum_of_ab_l224_224295


namespace coeff_b_l224_224770

noncomputable def g (a b c d e : ℝ) (x : ℝ) :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem coeff_b (a b c d e : ℝ):
  -- The function g(x) has roots at x = -1, 0, 1, 2
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  -- The function passes through the point (0, 3)
  (g a b c d e 0 = 3) →
  -- Assuming a = 1
  (a = 1) →
  -- Prove that b = -2
  b = -2 :=
by
  intros _ _ _ _ _ a_eq_1
  -- Proof omitted
  sorry

end coeff_b_l224_224770


namespace gray_area_l224_224834

def center_C : (ℝ × ℝ) := (6, 5)
def center_D : (ℝ × ℝ) := (14, 5)
def radius_C : ℝ := 3
def radius_D : ℝ := 3

theorem gray_area :
  let area_rectangle := 8 * 5
  let area_sector_C := (1 / 2) * π * radius_C^2
  let area_sector_D := (1 / 2) * π * radius_D^2
  area_rectangle - (area_sector_C + area_sector_D) = 40 - 9 * π :=
by
  sorry

end gray_area_l224_224834


namespace square_side_length_l224_224485

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end square_side_length_l224_224485


namespace set_equality_example_l224_224858

theorem set_equality_example : {x : ℕ | 2 * x + 3 ≥ 3 * x} = {0, 1, 2, 3} := by
  sorry

end set_equality_example_l224_224858


namespace problem_statement_l224_224495

theorem problem_statement : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end problem_statement_l224_224495


namespace geometric_sequence_common_ratio_l224_224043

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q)
  (h0 : a 1 = 2) (h1 : a 4 = 1 / 4) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l224_224043


namespace find_x_pow_3a_minus_b_l224_224418

variable (x : ℝ) (a b : ℝ)
theorem find_x_pow_3a_minus_b (h1 : x^a = 2) (h2 : x^b = 9) : x^(3 * a - b) = 8 / 9 :=
  sorry

end find_x_pow_3a_minus_b_l224_224418


namespace perpendicular_lines_implies_perpendicular_plane_l224_224103

theorem perpendicular_lines_implies_perpendicular_plane
  (triangle_sides : Line → Prop)
  (circle_diameters : Line → Prop)
  (perpendicular : Line → Line → Prop)
  (is_perpendicular_to_plane : Line → Prop) :
  (∀ l₁ l₂, triangle_sides l₁ → triangle_sides l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) ∧
  (∀ l₁ l₂, circle_diameters l₁ → circle_diameters l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) :=
  sorry

end perpendicular_lines_implies_perpendicular_plane_l224_224103


namespace remainder_div_1442_l224_224213

theorem remainder_div_1442 (x k l r : ℤ) (h1 : 1816 = k * x + 6) (h2 : 1442 = l * x + r) (h3 : x = Int.gcd 1810 374) : r = 0 := by
  sorry

end remainder_div_1442_l224_224213


namespace hotel_towels_l224_224614

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l224_224614


namespace maria_made_144_cookies_l224_224425

def cookies (C : ℕ) : Prop :=
  (2 * 1 / 4 * C = 72)

theorem maria_made_144_cookies: ∃ (C : ℕ), cookies C ∧ C = 144 :=
by
  existsi 144
  unfold cookies
  sorry

end maria_made_144_cookies_l224_224425


namespace soda_cans_ratio_l224_224923

theorem soda_cans_ratio
  (initial_cans : ℕ := 22)
  (cans_taken : ℕ := 6)
  (final_cans : ℕ := 24)
  (x : ℚ := 1 / 2)
  (cans_left : ℕ := 16)
  (cans_bought : ℕ := 16 * 1 / 2) :
  (cans_bought / cans_left : ℚ) = 1 / 2 :=
sorry

end soda_cans_ratio_l224_224923


namespace range_of_m_l224_224366

-- Definition of the propositions and conditions
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 3
def prop (m : ℝ) : Prop := (¬(p m ∧ q m) ∧ (p m ∨ q m))

-- The proof statement showing the range of m
theorem range_of_m (m : ℝ) : prop m ↔ (1 ≤ m ∧ m ≤ 2) ∨ (m > 3) :=
by
  sorry

end range_of_m_l224_224366


namespace solve_for_y_l224_224492

theorem solve_for_y (y : ℝ) (h : y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) : y = -4 :=
by {
  sorry
}

end solve_for_y_l224_224492


namespace alcohol_solution_contradiction_l224_224468

theorem alcohol_solution_contradiction (initial_volume : ℕ) (added_water : ℕ) 
                                        (final_volume : ℕ) (final_concentration : ℕ) 
                                        (initial_concentration : ℕ) : 
                                        initial_volume = 75 → added_water = 50 → 
                                        final_volume = initial_volume + added_water → 
                                        final_concentration = 45 → 
                                        ¬ (initial_concentration * initial_volume = final_concentration * final_volume) :=
by 
  intro h_initial_volume h_added_water h_final_volume h_final_concentration
  sorry

end alcohol_solution_contradiction_l224_224468


namespace largest_number_of_cakes_without_ingredients_l224_224538

theorem largest_number_of_cakes_without_ingredients :
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  ∃ (max_no_ingredients : ℕ), max_no_ingredients = 24 :=
by
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  existsi (60 - max 20 (max 30 (max 36 6))) -- max value should be used to reflect maximum coverage content
  sorry -- Proof to be completed

end largest_number_of_cakes_without_ingredients_l224_224538


namespace escalator_rate_is_15_l224_224718

noncomputable def rate_escalator_moves (escalator_length : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_length / time) - person_speed

theorem escalator_rate_is_15 :
  rate_escalator_moves 200 5 10 = 15 := by
  sorry

end escalator_rate_is_15_l224_224718


namespace prism_surface_area_l224_224453

-- Define the base of the prism as an isosceles trapezoid ABCD
structure Trapezoid :=
(AB CD : ℝ)
(BC : ℝ)
(AD : ℝ)

-- Define the properties of the prism
structure Prism :=
(base : Trapezoid)
(diagonal_cross_section_area : ℝ)

-- Define the specific isosceles trapezoid from the problem
def myTrapezoid : Trapezoid :=
{ AB := 13, CD := 13, BC := 11, AD := 21 }

-- Define the specific prism from the problem with the given conditions
noncomputable def myPrism : Prism :=
{ base := myTrapezoid, diagonal_cross_section_area := 180 }

-- Define the total surface area as a function
noncomputable def total_surface_area (p : Prism) : ℝ :=
2 * (1 / 2 * (p.base.AD + p.base.BC) * (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2))) +
(p.base.AB + p.base.BC + p.base.CD + p.base.AD) * (p.diagonal_cross_section_area / (Real.sqrt ((1 / 2 * (p.base.AD + p.base.BC)) ^ 2 + (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2)) ^ 2)))

-- The proof problem in Lean
theorem prism_surface_area :
  total_surface_area myPrism = 906 :=
sorry

end prism_surface_area_l224_224453


namespace total_cost_at_discount_l224_224764

-- Definitions for conditions
def original_price_notebook : ℕ := 15
def original_price_planner : ℕ := 10
def discount_rate : ℕ := 20
def number_of_notebooks : ℕ := 4
def number_of_planners : ℕ := 8

-- Theorem statement for the proof
theorem total_cost_at_discount :
  let discounted_price_notebook := original_price_notebook - (original_price_notebook * discount_rate / 100)
  let discounted_price_planner := original_price_planner - (original_price_planner * discount_rate / 100)
  let total_cost := (number_of_notebooks * discounted_price_notebook) + (number_of_planners * discounted_price_planner)
  total_cost = 112 :=
by
  sorry

end total_cost_at_discount_l224_224764


namespace total_lives_l224_224953

-- Defining the number of lives for each animal according to the given conditions:
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7
def elephant_lives : ℕ := 2 * cat_lives - 5
def fish_lives : ℕ := if (dog_lives + mouse_lives) < (elephant_lives / 2) then (dog_lives + mouse_lives) else elephant_lives / 2

-- The main statement we need to prove:
theorem total_lives :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 :=
by
  sorry

end total_lives_l224_224953


namespace min_f_value_inequality_solution_l224_224037

theorem min_f_value (x : ℝ) : |x+7| + |x-1| ≥ 8 := by
  sorry

theorem inequality_solution (x : ℝ) (m : ℝ) (h : m = 8) : |x-3| - 2*x ≤ 2*m - 12 ↔ x ≥ -1/3 := by
  sorry

end min_f_value_inequality_solution_l224_224037


namespace tetrahedron_edges_sum_of_squares_l224_224112

-- Given conditions
variables {a b c d e f x y z : ℝ}

-- Mathematical statement
theorem tetrahedron_edges_sum_of_squares :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (x^2 + y^2 + z^2) :=
sorry

end tetrahedron_edges_sum_of_squares_l224_224112


namespace binary_101_is_5_l224_224891

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l224_224891


namespace find_percentage_l224_224991

noncomputable def percentage (X : ℝ) : ℝ := (377.8020134228188 * 100 * 5.96) / 1265

theorem find_percentage : percentage 178 = 178 := by
  -- Conditions
  let P : ℝ := 178
  let A : ℝ := 1265
  let divisor : ℝ := 5.96
  let result : ℝ := 377.8020134228188

  -- Define the percentage calculation
  let X := (result * 100 * divisor) / A

  -- Verify the calculation matches
  have h : X = P := by sorry

  trivial

end find_percentage_l224_224991


namespace problem_statement_l224_224347

theorem problem_statement (x : ℕ) (h : 423 - x = 421) : (x * 423) + 421 = 1267 := by
  sorry

end problem_statement_l224_224347


namespace largest_possible_value_l224_224420

-- Definitions for the conditions
def lower_x_bound := -4
def upper_x_bound := -2
def lower_y_bound := 2
def upper_y_bound := 4

-- The proposition to prove
theorem largest_possible_value (x y : ℝ) 
    (h1 : lower_x_bound ≤ x) (h2 : x ≤ upper_x_bound)
    (h3 : lower_y_bound ≤ y) (h4 : y ≤ upper_y_bound) :
    ∃ v, v = (x + y) / x ∧ ∀ (w : ℝ), w = (x + y) / x → w ≤ 1/2 :=
by
  sorry

end largest_possible_value_l224_224420


namespace rectangle_shorter_side_l224_224670

theorem rectangle_shorter_side
  (x : ℝ)
  (a b d : ℝ)
  (h₁ : a = 3 * x)
  (h₂ : b = 4 * x)
  (h₃ : d = 9) :
  a = 5.4 := 
by
  sorry

end rectangle_shorter_side_l224_224670


namespace inequality_does_not_hold_l224_224735

noncomputable def f : ℝ → ℝ := sorry -- define f satisfying the conditions from a)

theorem inequality_does_not_hold :
  (∀ x, f (-x) = f x) ∧ -- f is even
  (∀ x, f x = f (x + 2)) ∧ -- f is periodic with period 2
  (∀ x, 3 ≤ x ∧ x ≤ 4 → f x = 2^x) → -- f(x) = 2^x when x is in [3, 4]
  ¬ (f (Real.sin 3) < f (Real.cos 3)) := by
  -- skipped proof
  sorry

end inequality_does_not_hold_l224_224735


namespace pure_gala_trees_l224_224974

variables (T F G : ℕ)

theorem pure_gala_trees :
  (0.1 * T : ℝ) + F = 238 ∧ F = (3 / 4) * ↑T → G = T - F → G = 70 :=
by
  intro h
  sorry

end pure_gala_trees_l224_224974


namespace range_of_k_l224_224668

theorem range_of_k (k : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ Set.Icc (-1 : ℝ) 3 →
    ∃ (x0 : ℝ), x0 ∈ Set.Icc (-1 : ℝ) 3 ∧ (2 * x1^2 + x1 - k) ≤ (x0^3 - 3 * x0)) →
  k ≥ 3 :=
by
  -- This is the place for the proof. 'sorry' is used to indicate that the proof is omitted.
  sorry

end range_of_k_l224_224668


namespace bob_total_distance_l224_224384

theorem bob_total_distance:
  let time1 := 1.5
  let speed1 := 60
  let time2 := 2
  let speed2 := 45
  (time1 * speed1) + (time2 * speed2) = 180 := 
  by
  sorry

end bob_total_distance_l224_224384


namespace min_coach_handshakes_l224_224079

-- Definitions based on the problem conditions
def total_gymnasts : ℕ := 26
def total_handshakes : ℕ := 325

/- 
  The main theorem stating that the fewest number of handshakes 
  the coaches could have participated in is 0.
-/
theorem min_coach_handshakes (n : ℕ) (h : 0 ≤ n ∧ n * (n - 1) / 2 = total_handshakes) : 
  n = total_gymnasts → (total_handshakes - n * (n - 1) / 2) = 0 :=
by 
  intros h_n_eq_26
  sorry

end min_coach_handshakes_l224_224079


namespace find_m_l224_224320

-- Define the vectors a and b
def veca (m : ℝ) : ℝ × ℝ := (m, 4)
def vecb (m : ℝ) : ℝ × ℝ := (m + 4, 1)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition that the dot product of a and b is zero
def are_perpendicular (m : ℝ) : Prop :=
  dot_product (veca m) (vecb m) = 0

-- The goal is to prove that if a and b are perpendicular, then m = -2
theorem find_m (m : ℝ) (h : are_perpendicular m) : m = -2 :=
by {
  -- Proof will be filled here
  sorry
}

end find_m_l224_224320


namespace range_of_m_l224_224809

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x < y → y < 0 → f y < f x) (h_cond : ∀ m : ℝ, f (1 - m) < f (m - 3)) : ∀ m, 1 < m ∧ m < 2 :=
by
  intros m
  sorry

end range_of_m_l224_224809


namespace vector_sum_is_zero_l224_224060

variables {V : Type*} [AddCommGroup V]

variables (AB CF BC FA : V)

-- Condition: Vectors form a closed polygon
def vectors_form_closed_polygon (AB CF BC FA : V) : Prop :=
  AB + BC + CF + FA = 0

theorem vector_sum_is_zero
  (h : vectors_form_closed_polygon AB CF BC FA) :
  AB + BC + CF + FA = 0 :=
  h

end vector_sum_is_zero_l224_224060


namespace increasing_function_odd_function_l224_224287

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem increasing_function (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
sorry

theorem odd_function (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) ↔ a = 1 :=
sorry

end increasing_function_odd_function_l224_224287


namespace RobertAteNine_l224_224210

-- Define the number of chocolates Nickel ate
def chocolatesNickelAte : ℕ := 2

-- Define the additional chocolates Robert ate compared to Nickel
def additionalChocolates : ℕ := 7

-- Define the total chocolates Robert ate
def chocolatesRobertAte : ℕ := chocolatesNickelAte + additionalChocolates

-- State the theorem we want to prove
theorem RobertAteNine : chocolatesRobertAte = 9 := by
  -- Skip the proof
  sorry

end RobertAteNine_l224_224210


namespace juan_distance_l224_224286

def time : ℝ := 80.0
def speed : ℝ := 10.0
def distance (t : ℝ) (s : ℝ) : ℝ := t * s

theorem juan_distance : distance time speed = 800.0 := by
  sorry

end juan_distance_l224_224286


namespace union_of_A_B_l224_224778

open Set

def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem union_of_A_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 2} :=
by sorry

end union_of_A_B_l224_224778


namespace oak_trees_remaining_l224_224863

theorem oak_trees_remaining (initial_trees cut_down_trees remaining_trees : ℕ)
  (h1 : initial_trees = 9)
  (h2 : cut_down_trees = 2)
  (h3 : remaining_trees = initial_trees - cut_down_trees) :
  remaining_trees = 7 :=
by 
  sorry

end oak_trees_remaining_l224_224863


namespace probability_useful_parts_l224_224746

noncomputable def probability_three_parts_useful (pipe_length : ℝ) (min_length : ℝ) : ℝ :=
  let total_area := (pipe_length * pipe_length) / 2
  let feasible_area := ((pipe_length - min_length) * (pipe_length - min_length)) / 2
  feasible_area / total_area

theorem probability_useful_parts :
  probability_three_parts_useful 300 75 = 1 / 16 :=
by
  sorry

end probability_useful_parts_l224_224746


namespace candidates_appeared_equal_l224_224379

theorem candidates_appeared_equal 
  (A_candidates B_candidates : ℕ)
  (A_selected B_selected : ℕ)
  (h1 : 6 * A_candidates = A_selected * 100)
  (h2 : 7 * B_candidates = B_selected * 100)
  (h3 : B_selected = A_selected + 83)
  (h4 : A_candidates = B_candidates):
  A_candidates = 8300 :=
by
  sorry

end candidates_appeared_equal_l224_224379


namespace my_age_is_five_times_son_age_l224_224916

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end my_age_is_five_times_son_age_l224_224916


namespace discount_problem_l224_224437

theorem discount_problem (x : ℝ) (h : 560 * (1 - x / 100) * 0.70 = 313.6) : x = 20 := 
by
  sorry

end discount_problem_l224_224437


namespace probability_divisible_by_5_l224_224351

def spinner_nums : List ℕ := [1, 2, 3, 5]

def total_outcomes (spins : ℕ) : ℕ :=
  List.length spinner_nums ^ spins

def count_divisible_by_5 (spins : ℕ) : ℕ :=
  let units_digit := 1
  let rest_combinations := (List.length spinner_nums) ^ (spins - units_digit)
  rest_combinations

theorem probability_divisible_by_5 : 
  let spins := 3 
  let successful_cases := count_divisible_by_5 spins
  let all_cases := total_outcomes spins
  successful_cases / all_cases = 1 / 4 :=
by
  sorry

end probability_divisible_by_5_l224_224351


namespace exists_1990_gon_with_conditions_l224_224263

/-- A polygon structure with side lengths and properties to check equality of interior angles and side lengths -/
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℕ)
  (angles_equal : Prop)

/-- Given conditions -/
def condition_1 (P : Polygon 1990) : Prop := P.angles_equal
def condition_2 (P : Polygon 1990) : Prop :=
  ∃ (σ : Fin 1990 → Fin 1990), ∀ i, P.sides i = (σ i + 1)^2

/-- The main theorem to be proven -/
theorem exists_1990_gon_with_conditions :
  ∃ P : Polygon 1990, condition_1 P ∧ condition_2 P :=
sorry

end exists_1990_gon_with_conditions_l224_224263


namespace max_f_l224_224398

open Real

noncomputable def f (x y z : ℝ) := (1 - y * z + z) * (1 - z * x + x) * (1 - x * y + y)

theorem max_f (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  f x y z ≤ 1 ∧ (x = 1 ∧ y = 1 ∧ z = 1 → f x y z = 1) := sorry

end max_f_l224_224398


namespace only_set_C_is_pythagorean_triple_l224_224871

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem only_set_C_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 15 20 25 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 1 3 5 :=
by {
  -- Proof goes here
  sorry
}

end only_set_C_is_pythagorean_triple_l224_224871


namespace blue_water_bottles_initial_count_l224_224898

theorem blue_water_bottles_initial_count
    (red : ℕ) (black : ℕ) (taken_out : ℕ) (left : ℕ) (initial_blue : ℕ) :
    red = 2 →
    black = 3 →
    taken_out = 5 →
    left = 4 →
    initial_blue + red + black = taken_out + left →
    initial_blue = 4 := by
  intros
  sorry

end blue_water_bottles_initial_count_l224_224898


namespace sum_of_decimals_is_fraction_l224_224707

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l224_224707


namespace solution_set_abs_inequality_l224_224854

theorem solution_set_abs_inequality : {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l224_224854


namespace total_points_other_7_members_is_15_l224_224039

variable (x y : ℕ)
variable (h1 : y ≤ 21)
variable (h2 : y = x * 7 / 15 - 18)
variable (h3 : (1 / 3) * x + (1 / 5) * x + 18 + y = x)

theorem total_points_other_7_members_is_15 (h : x * 7 % 15 = 0) : y = 15 :=
by
  sorry

end total_points_other_7_members_is_15_l224_224039


namespace initial_pieces_l224_224203

-- Definitions of the conditions
def pieces_eaten : ℕ := 7
def pieces_given : ℕ := 21
def pieces_now : ℕ := 37

-- The proposition to prove
theorem initial_pieces (C : ℕ) (h : C - pieces_eaten + pieces_given = pieces_now) : C = 23 :=
by
  -- Proof would go here
  sorry

end initial_pieces_l224_224203


namespace negative_large_base_zero_exponent_l224_224258

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end negative_large_base_zero_exponent_l224_224258


namespace miles_driven_on_tuesday_l224_224234

-- Define the conditions given in the problem
theorem miles_driven_on_tuesday (T : ℕ) (h_avg : (12 + T + 21) / 3 = 17) :
  T = 18 :=
by
  -- We state what we want to prove, but we leave the proof with sorry
  sorry

end miles_driven_on_tuesday_l224_224234


namespace households_using_neither_brands_l224_224188

def total_households : Nat := 240
def only_brand_A_households : Nat := 60
def both_brands_households : Nat := 25
def ratio_B_to_both : Nat := 3
def only_brand_B_households : Nat := ratio_B_to_both * both_brands_households
def either_brand_households : Nat := only_brand_A_households + only_brand_B_households + both_brands_households
def neither_brand_households : Nat := total_households - either_brand_households

theorem households_using_neither_brands :
  neither_brand_households = 80 :=
by
  -- Proof can be filled out here
  sorry

end households_using_neither_brands_l224_224188


namespace ratio_is_7_to_10_l224_224826

-- Given conditions in the problem translated to Lean definitions
def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 10 * leopards
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := 670
def other_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + alligators
def cheetahs : ℕ := total_animals - other_animals

-- The ratio of cheetahs to snakes to be proven
def ratio_cheetahs_to_snakes (cheetahs snakes : ℕ) : ℚ := cheetahs / snakes

theorem ratio_is_7_to_10 : ratio_cheetahs_to_snakes cheetahs snakes = 7 / 10 :=
by
  sorry

end ratio_is_7_to_10_l224_224826


namespace average_student_headcount_l224_224354

theorem average_student_headcount (h1 : ℕ := 10900) (h2 : ℕ := 10500) (h3 : ℕ := 10700) (h4 : ℕ := 11300) : 
  (h1 + h2 + h3 + h4) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l224_224354


namespace geometric_progression_general_term_l224_224975

noncomputable def a_n (n : ℕ) : ℝ := 2^(n-1)

theorem geometric_progression_general_term :
  (∀ n : ℕ, n ≥ 1 → a_n n > 0) ∧
  a_n 1 = 1 ∧
  a_n 2 + a_n 3 = 6 →
  ∀ n, a_n n = 2^(n-1) :=
by
  intros h
  sorry

end geometric_progression_general_term_l224_224975


namespace B_squared_B_sixth_l224_224374

noncomputable def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![0, 3], ![2, -1]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  1

theorem B_squared :
  B * B = 3 * B - I := by
  sorry

theorem B_sixth :
  B^6 = 84 * B - 44 * I := by
  sorry

end B_squared_B_sixth_l224_224374


namespace water_level_decrease_l224_224291

theorem water_level_decrease (increase_notation : ℝ) (h : increase_notation = 2) :
  -increase_notation = -2 :=
by
  sorry

end water_level_decrease_l224_224291


namespace angle_between_bisectors_of_trihedral_angle_l224_224280

noncomputable def angle_between_bisectors_trihedral (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) : ℝ :=
  60

theorem angle_between_bisectors_of_trihedral_angle (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) :
  angle_between_bisectors_trihedral α β γ hα hβ hγ = 60 := 
sorry

end angle_between_bisectors_of_trihedral_angle_l224_224280


namespace parallel_lines_slope_l224_224808

theorem parallel_lines_slope (a : ℝ) :
  (∃ (a : ℝ), ∀ x y, (3 * y - a = 9 * x + 1) ∧ (y - 2 = (2 * a - 3) * x)) → a = 3 :=
by
  sorry

end parallel_lines_slope_l224_224808


namespace pats_stick_covered_l224_224787

/-
Assumptions:
1. Pat's stick is 30 inches long.
2. Jane's stick is 22 inches long.
3. Jane’s stick is two feet (24 inches) shorter than Sarah’s stick.
4. The portion of Pat's stick not covered in dirt is half as long as Sarah’s stick.

Prove that the length of Pat's stick covered in dirt is 7 inches.
-/

theorem pats_stick_covered  (pat_stick_len : ℕ) (jane_stick_len : ℕ) (jane_sarah_diff : ℕ) (pat_not_covered_by_dirt : ℕ) :
  pat_stick_len = 30 → jane_stick_len = 22 → jane_sarah_diff = 24 → pat_not_covered_by_dirt * 2 = jane_stick_len + jane_sarah_diff → 
    (pat_stick_len - pat_not_covered_by_dirt) = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end pats_stick_covered_l224_224787


namespace binomial_coeff_sum_l224_224350

theorem binomial_coeff_sum {a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ}
  (h : (1 - x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| = 128 :=
by
  sorry

end binomial_coeff_sum_l224_224350


namespace diminish_value_l224_224634

theorem diminish_value (a b : ℕ) (h1 : a = 1015) (h2 : b = 12) (h3 : b = 16) (h4 : b = 18) (h5 : b = 21) (h6 : b = 28) :
  ∃ k, a - k = lcm (lcm (lcm b b) (lcm b b)) (lcm b b) ∧ k = 7 :=
sorry

end diminish_value_l224_224634


namespace minimum_cost_for_28_apples_l224_224543

/--
Conditions:
  - apples can be bought at a rate of 4 for 15 cents,
  - apples can be bought at a rate of 7 for 30 cents,
  - you need to buy exactly 28 apples.
Prove that the minimum total cost to buy exactly 28 apples is 120 cents.
-/
theorem minimum_cost_for_28_apples : 
  let cost_4_for_15 := 15
  let cost_7_for_30 := 30
  let apples_needed := 28
  ∃ (n m : ℕ), n * 4 + m * 7 = apples_needed ∧ n * cost_4_for_15 + m * cost_7_for_30 = 120 := sorry

end minimum_cost_for_28_apples_l224_224543


namespace remaining_volume_is_21_l224_224866

-- Definitions of edge lengths and volumes
def edge_length_original : ℕ := 3
def edge_length_small : ℕ := 1
def volume (a : ℕ) : ℕ := a ^ 3

-- Volumes of the original cube and the small cubes
def volume_original : ℕ := volume edge_length_original
def volume_small : ℕ := volume edge_length_small
def number_of_faces : ℕ := 6
def total_volume_cut : ℕ := number_of_faces * volume_small

-- Volume of the remaining part
def volume_remaining : ℕ := volume_original - total_volume_cut

-- Proof statement
theorem remaining_volume_is_21 : volume_remaining = 21 := by
  sorry

end remaining_volume_is_21_l224_224866


namespace shortest_chord_l224_224966

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop := 2 * m * x - y - 8 * m - 3 = 0
noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 6)^2 = 25

theorem shortest_chord (m : ℝ) :
  (∃ x y, line_eq m x y ∧ circle_eq x y) →
  m = 1 / 6 :=
by sorry

end shortest_chord_l224_224966


namespace width_of_wall_l224_224662

def volume_of_brick (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_wall (length width height : ℝ) : ℝ :=
  length * width * height

theorem width_of_wall
  (l_b w_b h_b : ℝ) (n : ℝ) (L H : ℝ)
  (volume_brick := volume_of_brick l_b w_b h_b)
  (total_volume_bricks := n * volume_brick) :
  volume_of_wall L (total_volume_bricks / (L * H)) H = total_volume_bricks :=
by
  sorry

end width_of_wall_l224_224662


namespace solve_equation_l224_224262

theorem solve_equation (x : ℝ) : 2 * x - 1 = 3 * x + 3 → x = -4 :=
by
  intro h
  sorry

end solve_equation_l224_224262


namespace max_squares_fitting_l224_224177

theorem max_squares_fitting (L S : ℕ) (hL : L = 8) (hS : S = 2) : (L / S) * (L / S) = 16 := by
  -- Proof goes here
  sorry

end max_squares_fitting_l224_224177


namespace circle_radii_l224_224484

noncomputable def smaller_circle_radius (r : ℝ) :=
  r = 4

noncomputable def larger_circle_radius (r : ℝ) :=
  r = 9

theorem circle_radii (r : ℝ) (h1 : ∀ (r: ℝ), (r + 5) - r = 5) (h2 : ∀ (r: ℝ), 2.4 * r = 2.4 * r):
  smaller_circle_radius r → larger_circle_radius (r + 5) :=
by
  sorry

end circle_radii_l224_224484


namespace triangle_perimeter_ABF_l224_224448

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 21) = 1

-- Define the line
def line (x : ℝ) : Prop := x = -2

-- Define the foci of the ellipse
def right_focus : ℝ := 2
def left_focus : ℝ := -2

-- Points A and B are on the ellipse and line
def point_A (x y : ℝ) : Prop := ellipse x y ∧ line x
def point_B (x y : ℝ) : Prop := ellipse x y ∧ line x

-- Point F is the right focus of the ellipse
def point_F (x y : ℝ) : Prop := x = right_focus ∧ y = 0

-- Perimeter of the triangle ABF
def perimeter (A B F : ℝ × ℝ) : ℝ :=
  sorry -- Calculation of the perimeter of triangle ABF

-- Theorem statement that perimeter is 20
theorem triangle_perimeter_ABF 
  (A B F : ℝ × ℝ) 
  (hA : point_A (A.fst) (A.snd)) 
  (hB : point_B (B.fst) (B.snd))
  (hF : point_F (F.fst) (F.snd)) :
  perimeter A B F = 20 :=
sorry

end triangle_perimeter_ABF_l224_224448


namespace bus_interval_l224_224995

theorem bus_interval (num_departures : ℕ) (total_duration : ℕ) (interval : ℕ)
  (h1 : num_departures = 11)
  (h2 : total_duration = 60)
  (h3 : interval = total_duration / (num_departures - 1)) :
  interval = 6 :=
by
  sorry

end bus_interval_l224_224995


namespace seq_common_max_l224_224150

theorem seq_common_max : ∃ a, a ≤ 250 ∧ 1 ≤ a ∧ a % 8 = 1 ∧ a % 9 = 4 ∧ ∀ b, b ≤ 250 ∧ 1 ≤ b ∧ b % 8 = 1 ∧ b % 9 = 4 → b ≤ a :=
by 
  sorry

end seq_common_max_l224_224150


namespace kite_ratio_equality_l224_224888

-- Definitions for points, lines, and conditions in the geometric setup
variables {Point : Type*} [MetricSpace Point]

-- Assuming A, B, C, D, P, E, F, G, H, I, J are points
variable (A B C D P E F G H I J : Point)

-- Conditions based on the problem
variables (AB_eq_AD : dist A B = dist A D)
          (BC_eq_CD : dist B C = dist C D)
          (on_BD : P ∈ line B D)
          (line_PE_inter_AD : E ∈ line P E ∧ E ∈ line A D)
          (line_PF_inter_BC : F ∈ line P F ∧ F ∈ line B C)
          (line_PG_inter_AB : G ∈ line P G ∧ G ∈ line A B)
          (line_PH_inter_CD : H ∈ line P H ∧ H ∈ line C D)
          (GF_inter_BD_at_I : I ∈ line G F ∧ I ∈ line B D)
          (EH_inter_BD_at_J : J ∈ line E H ∧ J ∈ line B D)

-- The statement to prove
theorem kite_ratio_equality :
  dist P I / dist P B = dist P J / dist P D := sorry

end kite_ratio_equality_l224_224888


namespace route_down_distance_l224_224562

-- Definitions
def rate_up : ℝ := 7
def time_up : ℝ := 2
def distance_up : ℝ := rate_up * time_up
def rate_down : ℝ := 1.5 * rate_up
def time_down : ℝ := time_up
def distance_down : ℝ := rate_down * time_down

-- Theorem
theorem route_down_distance : distance_down = 21 := by
  sorry

end route_down_distance_l224_224562


namespace proof_problem_l224_224021

open Real

-- Definitions
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Conditions
def eccentricity (c a : ℝ) : Prop :=
  c / a = (sqrt 2) / 2

def min_distance_to_focus (a c : ℝ) : Prop :=
  a - c = sqrt 2 - 1

-- Proof problem statement
theorem proof_problem (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (b_lt_a : b < a)
  (ecc : eccentricity c a) (min_dist : min_distance_to_focus a c)
  (x y k m : ℝ) (line_condition : y = k * x + m) :
  ellipse_equation x y a b → ellipse_equation x y (sqrt 2) 1 ∧
  (parabola_equation x y → (y = sqrt 2 / 2 * x + sqrt 2 ∨ y = -sqrt 2 / 2 * x - sqrt 2)) :=
sorry

end proof_problem_l224_224021


namespace no_club_member_is_fraternity_member_l224_224236

variable (Student : Type) (isHonest : Student → Prop) 
                       (isFraternityMember : Student → Prop) 
                       (isClubMember : Student → Prop)

axiom some_students_not_honest : ∃ x : Student, ¬ isHonest x
axiom all_frats_honest : ∀ y : Student, isFraternityMember y → isHonest y
axiom no_clubs_honest : ∀ z : Student, isClubMember z → ¬ isHonest z

theorem no_club_member_is_fraternity_member : ∀ w : Student, isClubMember w → ¬ isFraternityMember w :=
by sorry

end no_club_member_is_fraternity_member_l224_224236


namespace john_bought_two_shirts_l224_224421

/-- The number of shirts John bought, given the conditions:
1. The first shirt costs $6 more than the second shirt.
2. The first shirt costs $15.
3. The total cost of the shirts is $24,
is equal to 2. -/
theorem john_bought_two_shirts
  (S : ℝ) 
  (first_shirt_cost : ℝ := 15)
  (second_shirt_cost : ℝ := S)
  (cost_difference : first_shirt_cost = second_shirt_cost + 6)
  (total_cost : first_shirt_cost + second_shirt_cost = 24)
  : 2 = 2 :=
by
  sorry

end john_bought_two_shirts_l224_224421


namespace Im_abcd_eq_zero_l224_224162

noncomputable def normalized (z : ℂ) : ℂ := z / Complex.abs z

theorem Im_abcd_eq_zero (a b c d : ℂ)
  (h1 : ∃ α : ℝ, ∃ w : ℂ, w = Complex.cos α + Complex.sin α * Complex.I ∧ (normalized b = w * normalized a) ∧ (normalized d = w * normalized c)) :
  Complex.im (a * b * c * d) = 0 :=
by
  sorry

end Im_abcd_eq_zero_l224_224162


namespace reciprocal_sum_l224_224086

theorem reciprocal_sum (a b c d : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 3) (h4 : d = 4) : 
  (a / b + c / d)⁻¹ = (20 : ℚ) / 23 := 
by
  sorry

end reciprocal_sum_l224_224086


namespace molly_age_condition_l224_224908

-- Definitions
def S : ℕ := 38 - 6
def M : ℕ := 24

-- The proof problem
theorem molly_age_condition :
  (S / M = 4 / 3) → (S = 32) → (M = 24) :=
by
  intro h_ratio h_S
  sorry

end molly_age_condition_l224_224908


namespace distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l224_224729

-- We will assume the depth of the well as a constant
def well_depth : ℝ := 4.0

-- Climb and slide distances as per each climb
def first_climb : ℝ := 1.2
def first_slide : ℝ := 0.4
def second_climb : ℝ := 1.4
def second_slide : ℝ := 0.5
def third_climb : ℝ := 1.1
def third_slide : ℝ := 0.3
def fourth_climb : ℝ := 1.2
def fourth_slide : ℝ := 0.2

noncomputable def net_gain_four_climbs : ℝ :=
  (first_climb - first_slide) + (second_climb - second_slide) +
  (third_climb - third_slide) + (fourth_climb - fourth_slide)

noncomputable def distance_from_top_after_four : ℝ := 
  well_depth - net_gain_four_climbs

noncomputable def total_distance_covered_four_climbs : ℝ :=
  first_climb + first_slide + second_climb + second_slide +
  third_climb + third_slide + fourth_climb + fourth_slide

noncomputable def can_climb_out_fifth_climb : Bool :=
  well_depth < (net_gain_four_climbs + first_climb)

-- Now we state the theorems we need to prove

theorem distance_from_top_correct :
  distance_from_top_after_four = 0.5 := by
  sorry

theorem total_distance_covered_correct :
  total_distance_covered_four_climbs = 6.3 := by
  sorry

theorem fifth_climb_success :
  can_climb_out_fifth_climb = true := by
  sorry

end distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l224_224729


namespace infinitely_many_arithmetic_progression_triples_l224_224950

theorem infinitely_many_arithmetic_progression_triples :
  ∃ (u v: ℤ) (a b c: ℤ), 
  (∀ n: ℤ, (a = 2 * u) ∧ 
    (b = 2 * u + v) ∧
    (c = 2 * u + 2 * v) ∧ 
    (u > 0) ∧
    (v > 0) ∧
    ∃ k m n: ℤ, 
    (a * b + 1 = k * k) ∧ 
    (b * c + 1 = m * m) ∧ 
    (c * a + 1 = n * n)) :=
sorry

end infinitely_many_arithmetic_progression_triples_l224_224950


namespace math_problem_l224_224463

theorem math_problem :
  let initial := 180
  let thirty_five_percent := 0.35 * initial
  let one_third_less := thirty_five_percent - (thirty_five_percent / 3)
  let remaining := initial - one_third_less
  let three_fifths_remaining := (3 / 5) * remaining
  (three_fifths_remaining ^ 2) = 6857.84 :=
by
  sorry

end math_problem_l224_224463


namespace no_such_triples_l224_224651

theorem no_such_triples : ¬ ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Prime ((a-2)*(b-2)*(c-2)+12) ∧ 
  ((a-2)*(b-2)*(c-2)+12) ∣ (a^2 + b^2 + c^2 + a*b*c - 2017) := 
by sorry

end no_such_triples_l224_224651


namespace student_number_in_eighth_group_l224_224784

-- Definitions corresponding to each condition
def students : ℕ := 50
def group_size : ℕ := 5
def third_group_student_number : ℕ := 12
def kth_group_number (k : ℕ) (n : ℕ) : ℕ := n + (k - 3) * group_size

-- Main statement to prove
theorem student_number_in_eighth_group :
  kth_group_number 8 third_group_student_number = 37 :=
  by
  sorry

end student_number_in_eighth_group_l224_224784


namespace max_area_perpendicular_l224_224119

theorem max_area_perpendicular (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : 
  ∃ θ_max, θ_max = Real.pi / 2 ∧ (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  (0 < Real.sin θ → (1 / 2) * a * b * Real.sin θ ≤ (1 / 2) * a * b * 1)) :=
sorry

end max_area_perpendicular_l224_224119


namespace total_surface_area_is_correct_l224_224278

-- Define the problem constants and structure
def num_cubes := 20
def edge_length := 1
def bottom_layer := 9
def middle_layer := 8
def top_layer := 3
def total_painted_area : ℕ := 55

-- Define a function to calculate the exposed surface area
noncomputable def calc_exposed_area (num_bottom : ℕ) (num_middle : ℕ) (num_top : ℕ) (edge : ℕ) : ℕ := 
    let bottom_exposed := num_bottom * (edge * edge)
    let middle_corners_exposed := 4 * 3 * edge
    let middle_edges_exposed := (num_middle - 4) * (2 * edge)
    let top_exposed := num_top * (5 * edge)
    bottom_exposed + middle_corners_exposed + middle_edges_exposed + top_exposed

-- Statement to prove the total painted area
theorem total_surface_area_is_correct : calc_exposed_area bottom_layer middle_layer top_layer edge_length = total_painted_area :=
by
  -- The proof itself is omitted, focus is on the statement.
  sorry

end total_surface_area_is_correct_l224_224278


namespace exponentiation_equation_l224_224427

theorem exponentiation_equation : 4^2011 * (-0.25)^2010 - 1 = 3 := 
by { sorry }

end exponentiation_equation_l224_224427


namespace equal_distances_l224_224097

def Point := ℝ × ℝ × ℝ

def dist (p1 p2 : Point) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2

def A : Point := (-8, 0, 0)
def B : Point := (0, 4, 0)
def C : Point := (0, 0, -6)
def D : Point := (0, 0, 0)
def P : Point := (-4, 2, -3)

theorem equal_distances : dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D :=
by
  sorry

end equal_distances_l224_224097


namespace oyster_crab_ratio_l224_224417

theorem oyster_crab_ratio
  (O1 C1 : ℕ)
  (h1 : O1 = 50)
  (h2 : C1 = 72)
  (h3 : ∃ C2 : ℕ, C2 = (2 * C1) / 3)
  (h4 : ∃ O2 : ℕ, O1 + C1 + O2 + C2 = 195) :
  ∃ ratio : ℚ, ratio = O2 / O1 ∧ ratio = (1 : ℚ) / 2 := 
by 
  sorry

end oyster_crab_ratio_l224_224417


namespace terms_before_five_l224_224142

theorem terms_before_five (a₁ : ℤ) (d : ℤ) (n : ℤ) :
  a₁ = 75 → d = -5 → (a₁ + (n - 1) * d = 5) → n - 1 = 14 :=
by
  intros h1 h2 h3
  sorry

end terms_before_five_l224_224142


namespace train_speed_is_correct_l224_224228

/-- Define the length of the train and the time taken to cross the telegraph post. --/
def train_length : ℕ := 240
def crossing_time : ℕ := 16

/-- Define speed calculation based on train length and crossing time. --/
def train_speed : ℕ := train_length / crossing_time

/-- Prove that the computed speed of the train is 15 meters per second. --/
theorem train_speed_is_correct : train_speed = 15 := sorry

end train_speed_is_correct_l224_224228


namespace largest_square_plot_size_l224_224224

def field_side_length := 50
def available_fence_length := 4000

theorem largest_square_plot_size :
  ∃ (s : ℝ), (0 < s) ∧ (s ≤ field_side_length) ∧ 
  (100 * (field_side_length - s) = available_fence_length) →
  s = 10 :=
by
  sorry

end largest_square_plot_size_l224_224224


namespace probability_calc_l224_224508

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let pairs_count := 169
  let valid_pairs_count := 17
  1 - (valid_pairs_count / pairs_count : ℚ)

theorem probability_calc :
  probability_no_distinct_positive_real_roots = 152 / 169 := by sorry

end probability_calc_l224_224508


namespace total_eggs_collected_by_all_four_l224_224340

def benjamin_eggs := 6
def carla_eggs := 3 * benjamin_eggs
def trisha_eggs := benjamin_eggs - 4
def david_eggs := 2 * trisha_eggs

theorem total_eggs_collected_by_all_four :
  benjamin_eggs + carla_eggs + trisha_eggs + david_eggs = 30 := by
  sorry

end total_eggs_collected_by_all_four_l224_224340


namespace part1_part2_l224_224407

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l224_224407


namespace no_constant_term_in_expansion_l224_224686

theorem no_constant_term_in_expansion : 
  ∀ (x : ℂ), ¬ ∃ (k : ℕ), ∃ (c : ℂ), c * x ^ (k / 3 - 2 * (12 - k)) = 0 :=
by sorry

end no_constant_term_in_expansion_l224_224686


namespace standard_equation_of_ellipse_midpoint_of_chord_l224_224117

variables (a b c : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (A B : ℝ × ℝ)

axiom conditions :
  a > b ∧ b > 0 ∧
  (c / a = (Real.sqrt 6) / 3) ∧
  a = Real.sqrt 3 ∧
  a^2 = b^2 + c^2 ∧
  (A = (-1, 0)) ∧ (B = (x2, y2)) ∧
  A ≠ B ∧
  (∃ l : ℝ -> ℝ, l (-1) = 0 ∧ ∀ x, l x = x + 1) ∧
  (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = -3 / 2)

theorem standard_equation_of_ellipse :
  ∃ (e : ℝ), e = 1 ∧ (x1 / 3) + y1 = 1 := sorry

theorem midpoint_of_chord :
  ∃ (m : ℝ × ℝ), m = (-(3 / 4), 1 / 4) := sorry

end standard_equation_of_ellipse_midpoint_of_chord_l224_224117


namespace volume_ratio_surface_area_ratio_l224_224156

theorem volume_ratio_surface_area_ratio (V1 V2 S1 S2 : ℝ) (h : V1 / V2 = 8 / 27) :
  S1 / S2 = 4 / 9 :=
by
  sorry

end volume_ratio_surface_area_ratio_l224_224156


namespace find_k_l224_224401

theorem find_k 
  (x y k : ℚ) 
  (h1 : y = 4 * x - 1) 
  (h2 : y = -1 / 3 * x + 11) 
  (h3 : y = 2 * x + k) : 
  k = 59 / 13 :=
sorry

end find_k_l224_224401


namespace diamonds_in_F10_l224_224743

def diamonds_in_figure (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (Nat.add (Nat.mul (n - 1) n) 0) / 2

theorem diamonds_in_F10 : diamonds_in_figure 10 = 136 :=
by
  sorry

end diamonds_in_F10_l224_224743


namespace race_problem_l224_224557

theorem race_problem 
    (d : ℕ) (a1 : ℕ) (a2 : ℕ) 
    (h1 : d = 60)
    (h2 : a1 = 10)
    (h3 : a2 = 20) 
    (const_speed : ∀ (x y z : ℕ), x * y = z → y ≠ 0 → x = z / y) :
  (d - d * (d - a1) / (d - a2) = 12) := 
by {
  sorry
}

end race_problem_l224_224557


namespace initial_money_l224_224166

-- Definitions based on conditions in the problem
def money_left_after_purchase : ℕ := 3
def cost_of_candy_bar : ℕ := 1

-- Theorem statement to prove the initial amount of money
theorem initial_money (initial_amount : ℕ) :
  initial_amount - cost_of_candy_bar = money_left_after_purchase → initial_amount = 4 :=
sorry

end initial_money_l224_224166


namespace tim_initial_balls_correct_l224_224646

-- Defining the initial number of balls Robert had
def robert_initial_balls : ℕ := 25

-- Defining the final number of balls Robert had
def robert_final_balls : ℕ := 45

-- Defining the number of balls Tim had initially
def tim_initial_balls := 40

-- Now, we state the proof problem:
theorem tim_initial_balls_correct :
  robert_initial_balls + (tim_initial_balls / 2) = robert_final_balls :=
by
  -- This is the part where you typically write the proof.
  -- However, we put sorry here because the task does not require the proof itself.
  sorry

end tim_initial_balls_correct_l224_224646


namespace ram_total_distance_l224_224744

noncomputable def total_distance 
  (speed1 speed2 time1 total_time : ℝ) 
  (h_speed1 : speed1 = 20) 
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8) 
  : ℝ := 
  speed1 * time1 + speed2 * (total_time - time1)

theorem ram_total_distance
  (speed1 speed2 time1 total_time : ℝ)
  (h_speed1 : speed1 = 20)
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8)
  : total_distance speed1 speed2 time1 total_time h_speed1 h_speed2 h_time1 h_total_time = 400 :=
  sorry

end ram_total_distance_l224_224744


namespace find_x_sq_add_y_sq_l224_224813

theorem find_x_sq_add_y_sq (x y : ℝ) (h1 : (x + y) ^ 2 = 36) (h2 : x * y = 10) : x ^ 2 + y ^ 2 = 16 :=
by
  sorry

end find_x_sq_add_y_sq_l224_224813


namespace pyramid_partition_volumes_l224_224933

noncomputable def pyramid_partition_ratios (S A B C D P Q V1 V2 : ℝ) : Prop :=
  let P := ((S + B) / 2 : ℝ)
  let Q := ((S + D) / 2 : ℝ)
  (V1 < V2) → 
  (V2 / V1 = 5)

theorem pyramid_partition_volumes
  (S A B C D P Q : ℝ)
  (V1 V2 : ℝ)
  (hP : P = (S + B) / 2)
  (hQ : Q = (S + D) / 2)
  (hV1 : V1 < V2)
  : V2 / V1 = 5 := 
sorry

end pyramid_partition_volumes_l224_224933


namespace roots_triple_relation_l224_224725

theorem roots_triple_relation (p q r α β : ℝ) (h1 : α + β = -q / p) (h2 : α * β = r / p) (h3 : β = 3 * α) :
  3 * q ^ 2 = 16 * p * r :=
sorry

end roots_triple_relation_l224_224725


namespace area_of_ring_between_concentric_circles_l224_224077

theorem area_of_ring_between_concentric_circles :
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  area_ring = 95 * Real.pi :=
by
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  show area_ring = 95 * Real.pi
  sorry

end area_of_ring_between_concentric_circles_l224_224077


namespace probability_blue_or_purple_l224_224621

def total_jelly_beans : ℕ := 35
def blue_jelly_beans : ℕ := 7
def purple_jelly_beans : ℕ := 10

theorem probability_blue_or_purple : (blue_jelly_beans + purple_jelly_beans: ℚ) / total_jelly_beans = 17 / 35 := 
by sorry

end probability_blue_or_purple_l224_224621


namespace range_of_m_l224_224114

theorem range_of_m (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ 7 < m ∧ m < 24 :=
sorry

end range_of_m_l224_224114


namespace length_of_AB_l224_224890

-- Definitions based on given conditions:
variables (AB BC CD DE AE AC : ℕ)
variables (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21)

-- The theorem stating the length of AB given the conditions.
theorem length_of_AB (AB BC CD DE AE AC : ℕ)
  (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21) : AB = 5 := by
  sorry

end length_of_AB_l224_224890


namespace find_m_l224_224061

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end find_m_l224_224061


namespace sufficient_but_not_necessary_l224_224285

theorem sufficient_but_not_necessary (a b : ℝ) (hp : a > 1 ∧ b > 1) (hq : a + b > 2 ∧ a * b > 1) : 
  (a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧ ¬(a + b > 2 ∧ a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l224_224285


namespace remainder_of_power_mod_l224_224328

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l224_224328


namespace handshake_problem_l224_224260

theorem handshake_problem (n : ℕ) (hn : n = 11) (H : n * (n - 1) / 2 = 55) : 10 = n - 1 :=
by
  sorry

end handshake_problem_l224_224260


namespace fish_count_when_james_discovers_l224_224555

def fish_in_aquarium (initial_fish : ℕ) (bobbit_worm_eats : ℕ) (predatory_fish_eats : ℕ)
  (reproduction_rate : ℕ × ℕ) (days_1 : ℕ) (added_fish: ℕ) (days_2 : ℕ) : ℕ :=
  let predation_rate := bobbit_worm_eats + predatory_fish_eats
  let total_eaten_in_14_days := predation_rate * days_1
  let reproduction_events_in_14_days := days_1 / reproduction_rate.snd
  let fish_born_in_14_days := reproduction_events_in_14_days * reproduction_rate.fst
  let fish_after_14_days := initial_fish - total_eaten_in_14_days + fish_born_in_14_days
  let fish_after_14_days_non_negative := max fish_after_14_days 0
  let fish_after_addition := fish_after_14_days_non_negative + added_fish
  let total_eaten_in_7_days := predation_rate * days_2
  let reproduction_events_in_7_days := days_2 / reproduction_rate.snd
  let fish_born_in_7_days := reproduction_events_in_7_days * reproduction_rate.fst
  let fish_after_7_days := fish_after_addition - total_eaten_in_7_days + fish_born_in_7_days
  max fish_after_7_days 0

theorem fish_count_when_james_discovers :
  fish_in_aquarium 60 2 3 (2, 3) 14 8 7 = 4 :=
sorry

end fish_count_when_james_discovers_l224_224555


namespace quartic_two_real_roots_l224_224047

theorem quartic_two_real_roots
  (a b c d e : ℝ)
  (h : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + e - d = 0)
  (ha : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^4 + b * x1^3 + c * x1^2 + d * x1 + e = 0) ∧ (a * x2^4 + b * x2^3 + c * x2^2 + d * x2 + e = 0) := 
  sorry

end quartic_two_real_roots_l224_224047


namespace total_population_l224_224917

-- Define the conditions
variables (T G Td Lb : ℝ)

-- Given conditions and the result
def conditions : Prop :=
  G = 1 / 2 * T ∧
  Td = 0.60 * G ∧
  Lb = 16000 ∧
  T = Td + G + Lb

-- Problem statement: Prove that the total population T is 80000
theorem total_population (h : conditions T G Td Lb) : T = 80000 :=
by
  sorry

end total_population_l224_224917


namespace range_of_m_l224_224183

-- Define the polynomial p(x)
def p (x : ℝ) (m : ℝ) := x^2 + 2*x - m

-- Given conditions: p(1) is false and p(2) is true
theorem range_of_m (m : ℝ) : 
  (p 1 m ≤ 0) ∧ (p 2 m > 0) → (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l224_224183


namespace lucille_total_revenue_l224_224876

theorem lucille_total_revenue (salary_ratio stock_ratio : ℕ) (salary_amount : ℝ) (h_ratio : salary_ratio / stock_ratio = 4 / 11) (h_salary : salary_amount = 800) : 
  ∃ total_revenue : ℝ, total_revenue = 3000 :=
by
  sorry

end lucille_total_revenue_l224_224876


namespace train_pass_jogger_time_l224_224331

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 60
noncomputable def initial_distance_m : ℝ := 350
noncomputable def train_length_m : ℝ := 250

noncomputable def relative_speed_m_per_s : ℝ := 
  ((train_speed_km_per_hr - jogger_speed_km_per_hr) * 1000) / 3600

noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m

noncomputable def time_to_pass_s : ℝ := total_distance_m / relative_speed_m_per_s

theorem train_pass_jogger_time :
  abs (time_to_pass_s - 42.35) < 0.01 :=
by 
  sorry

end train_pass_jogger_time_l224_224331


namespace casper_entry_exit_ways_correct_l224_224195

-- Define the total number of windows
def num_windows : Nat := 8

-- Define the number of ways Casper can enter and exit through different windows
def casper_entry_exit_ways (num_windows : Nat) : Nat :=
  num_windows * (num_windows - 1)

-- Create a theorem to state the problem and its solution
theorem casper_entry_exit_ways_correct : casper_entry_exit_ways num_windows = 56 := by
  sorry

end casper_entry_exit_ways_correct_l224_224195


namespace infinitely_many_c_exist_l224_224766

theorem infinitely_many_c_exist :
  ∃ c: ℕ, ∃ x y z: ℕ, (x^2 - c) * (y^2 - c) = z^2 - c ∧ (x^2 + c) * (y^2 - c) = z^2 - c :=
by
  sorry

end infinitely_many_c_exist_l224_224766


namespace range_of_a_l224_224885

noncomputable def p (a : ℝ) := ∀ x : ℝ, x^2 + a ≥ 0
noncomputable def q (a : ℝ) := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≥ 0) := by
  sorry

end range_of_a_l224_224885


namespace midpoint_ellipse_trajectory_l224_224361

theorem midpoint_ellipse_trajectory (x y x0 y0 x1 y1 x2 y2 : ℝ) :
  (x0 / 12) + (y0 / 8) = 1 →
  (x1^2 / 24) + (y1^2 / 16) = 1 →
  (x2^2 / 24) + (y2^2 / 16) = 1 →
  x = (x1 + x2) / 2 →
  y = (y1 + y2) / 2 →
  ∃ x y, ((x - 1)^2 / (5 / 2)) + ((y - 1)^2 / (5 / 3)) = 1 :=
by
  sorry

end midpoint_ellipse_trajectory_l224_224361


namespace miles_per_gallon_l224_224233

theorem miles_per_gallon (miles gallons : ℝ) (h : miles = 100 ∧ gallons = 5) : miles / gallons = 20 := by
  cases h with
  | intro miles_eq gallons_eq =>
    rw [miles_eq, gallons_eq]
    norm_num

end miles_per_gallon_l224_224233


namespace range_of_t_l224_224620

noncomputable def f : ℝ → ℝ := sorry

axiom f_symmetric (x : ℝ) : f (x - 3) = f (-x - 3)
axiom f_ln_definition (x : ℝ) (h : x ≤ -3) : f x = Real.log (-x)

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f (Real.sin x - t) > f (3 * Real.sin x - 1)) ↔ (t < -1 ∨ t > 9) := sorry

end range_of_t_l224_224620


namespace drug_price_reduction_eq_l224_224963

variable (x : ℝ)
variable (initial_price : ℝ := 144)
variable (final_price : ℝ := 81)

theorem drug_price_reduction_eq :
  initial_price * (1 - x)^2 = final_price :=
by
  sorry

end drug_price_reduction_eq_l224_224963


namespace find_speed_of_car_y_l224_224713

noncomputable def average_speed_of_car_y (sₓ : ℝ) (delay : ℝ) (d_afterₓ_started : ℝ) : ℝ :=
  let tₓ_before := delay
  let dₓ_before := sₓ * tₓ_before
  let total_dₓ := dₓ_before + d_afterₓ_started
  let tₓ_after := d_afterₓ_started / sₓ
  let total_time_y := tₓ_after
  d_afterₓ_started / total_time_y

theorem find_speed_of_car_y (h₁ : ∀ t, t = 1.2) (h₂ : ∀ sₓ, sₓ = 35) (h₃ : ∀ d_afterₓ_started, d_afterₓ_started = 42) : 
  average_speed_of_car_y 35 1.2 42 = 35 := by
  unfold average_speed_of_car_y
  simp
  sorry

end find_speed_of_car_y_l224_224713


namespace range_of_a_l224_224178

noncomputable def condition_p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
noncomputable def condition_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬(∀ x, condition_p x)) → (¬(∀ x, condition_q x a)) → 
  (∀ x, condition_p x ↔ condition_q x a) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l224_224178


namespace prevent_four_digit_number_l224_224196

theorem prevent_four_digit_number (N : ℕ) (n : ℕ) :
  n = 123 + 102 * N ∧ ∀ x : ℕ, (3 + 2 * x) % 10 < 1000 → x < 1000 := 
sorry

end prevent_four_digit_number_l224_224196


namespace function_increasing_l224_224617

variable {α : Type*} [LinearOrderedField α]

def is_increasing (f : α → α) : Prop :=
  ∀ x y : α, x < y → f x < f y

theorem function_increasing (f : α → α) (h : ∀ x1 x2 : α, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1) :
  is_increasing f :=
by
  sorry

end function_increasing_l224_224617


namespace orchard_problem_l224_224024

theorem orchard_problem (number_of_peach_trees number_of_apple_trees : ℕ) 
  (h1 : number_of_apple_trees = number_of_peach_trees + 1700)
  (h2 : number_of_apple_trees = 3 * number_of_peach_trees + 200) :
  number_of_peach_trees = 750 ∧ number_of_apple_trees = 2450 :=
by
  sorry

end orchard_problem_l224_224024


namespace sahil_selling_price_l224_224232

-- Definitions based on the conditions
def purchase_price : ℕ := 10000
def repair_costs : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

def total_cost : ℕ := purchase_price + repair_costs + transportation_charges
def profit : ℕ := (profit_percentage * total_cost) / 100
def selling_price : ℕ := total_cost + profit

-- The theorem we need to prove
theorem sahil_selling_price : selling_price = 24000 :=
by
  sorry

end sahil_selling_price_l224_224232


namespace find_k_values_l224_224452

open Set

def A : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def B (k : ℝ) : Set ℝ := {x | x^2 - (k + 1) * x + k = 0}

theorem find_k_values (k : ℝ) : (A ∩ B k = B k) ↔ k ∈ ({1, -3} : Set ℝ) := by
  sorry

end find_k_values_l224_224452


namespace de_morgan_union_de_morgan_inter_l224_224447

open Set

variable {α : Type*} (A B : Set α)

theorem de_morgan_union : ∀ (A B : Set α), 
  compl (A ∪ B) = compl A ∩ compl B := 
by 
  intro A B
  sorry

theorem de_morgan_inter : ∀ (A B : Set α), 
  compl (A ∩ B) = compl A ∪ compl B := 
by 
  intro A B
  sorry

end de_morgan_union_de_morgan_inter_l224_224447


namespace probability_of_desired_roll_l224_224294

-- Definitions of six-sided dice rolls and probability results
def is_greater_than_four (n : ℕ) : Prop := n > 4
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

-- Definitions of probabilities based on dice outcomes
def prob_greater_than_four : ℚ := 2 / 6
def prob_prime : ℚ := 3 / 6

-- Definition of joint probability for independent events
def joint_prob : ℚ := prob_greater_than_four * prob_prime

-- Theorem to prove
theorem probability_of_desired_roll : joint_prob = 1 / 6 := 
by
  sorry

end probability_of_desired_roll_l224_224294


namespace mean_score_juniors_is_103_l224_224040

noncomputable def mean_score_juniors : Prop :=
  ∃ (students juniors non_juniors m_j m_nj : ℝ),
  students = 160 ∧
  (students * 82) = (juniors * m_j + non_juniors * m_nj) ∧
  juniors = 0.4 * non_juniors ∧
  m_j = 1.4 * m_nj ∧
  m_j = 103

theorem mean_score_juniors_is_103 : mean_score_juniors :=
by
  sorry

end mean_score_juniors_is_103_l224_224040


namespace total_snowballs_l224_224992

theorem total_snowballs (Lc : ℕ) (Ch : ℕ) (Pt : ℕ)
  (h1 : Ch = Lc + 31)
  (h2 : Lc = 19)
  (h3 : Pt = 47) : 
  Ch + Lc + Pt = 116 := by
  sorry

end total_snowballs_l224_224992


namespace base_n_representation_l224_224932

theorem base_n_representation (n : ℕ) (b : ℕ) (h₀ : 8 < n) (h₁ : ∃ b, (n : ℤ)^2 - (n+8) * (n : ℤ) + b = 0) : 
  b = 8 * n :=
by
  sorry

end base_n_representation_l224_224932


namespace simplify_expression1_simplify_expression2_l224_224955

theorem simplify_expression1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 :=
by
  sorry

theorem simplify_expression2 (a : ℝ) : 
  (5*a^2 + 2*a - 1) - 4*(3 - 8*a + 2*a^2) = -3*a^2 + 34*a - 13 :=
by
  sorry

end simplify_expression1_simplify_expression2_l224_224955


namespace circle_radius_squared_l224_224856

-- Let r be the radius of the circle.
-- Let AB and CD be chords of the circle with lengths 10 and 7 respectively.
-- Let the extensions of AB and CD intersect at a point P outside the circle.
-- Let ∠APD be 60 degrees.
-- Let BP be 8.

theorem circle_radius_squared
  (r : ℝ)       -- radius of the circle
  (AB : ℝ)     -- length of chord AB
  (CD : ℝ)     -- length of chord CD
  (APD : ℝ)    -- angle APD
  (BP : ℝ)     -- length of segment BP
  (hAB : AB = 10)
  (hCD : CD = 7)
  (hAPD : APD = 60)
  (hBP : BP = 8)
  : r^2 = 73 := 
  sorry

end circle_radius_squared_l224_224856


namespace problem_equiv_l224_224279

theorem problem_equiv :
  ((2001 * 2021 + 100) * (1991 * 2031 + 400)) / (2011^4) = 1 :=
by
  sorry

end problem_equiv_l224_224279


namespace total_packs_equiv_117_l224_224071

theorem total_packs_equiv_117 
  (nancy_cards : ℕ)
  (melanie_cards : ℕ)
  (mary_cards : ℕ)
  (alyssa_cards : ℕ)
  (nancy_pack : ℝ)
  (melanie_pack : ℝ)
  (mary_pack : ℝ)
  (alyssa_pack : ℝ)
  (H_nancy : nancy_cards = 540)
  (H_melanie : melanie_cards = 620)
  (H_mary : mary_cards = 480)
  (H_alyssa : alyssa_cards = 720)
  (H_nancy_pack : nancy_pack = 18.5)
  (H_melanie_pack : melanie_pack = 22.5)
  (H_mary_pack : mary_pack = 15.3)
  (H_alyssa_pack : alyssa_pack = 24) :
  (⌊nancy_cards / nancy_pack⌋₊ + ⌊melanie_cards / melanie_pack⌋₊ + ⌊mary_cards / mary_pack⌋₊ + ⌊alyssa_cards / alyssa_pack⌋₊) = 117 :=
by
  sorry

end total_packs_equiv_117_l224_224071


namespace triangle_side_length_l224_224624

theorem triangle_side_length (B C : Real) (b c : Real) 
  (h1 : c * Real.cos B = 12) 
  (h2 : b * Real.sin C = 5) 
  (h3 : b * Real.sin B = 5) : 
  c = 13 := 
sorry

end triangle_side_length_l224_224624


namespace option_b_results_in_2x_cubed_l224_224240

variable (x : ℝ)

theorem option_b_results_in_2x_cubed : |x^3| + x^3 = 2 * x^3 := 
sorry

end option_b_results_in_2x_cubed_l224_224240


namespace solve_for_q_l224_224904

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l224_224904


namespace smallest_base_to_express_100_with_three_digits_l224_224952

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end smallest_base_to_express_100_with_three_digits_l224_224952


namespace simplify_expression_l224_224563

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- Define the expression and the simplified expression
def original_expr := -a^2 * (-2 * a * b) + 3 * a * (a^2 * b - 1)
def simplified_expr := 5 * a^3 * b - 3 * a

-- Statement that the original expression is equal to the simplified expression
theorem simplify_expression : original_expr a b = simplified_expr a b :=
by
  sorry

end simplify_expression_l224_224563


namespace Isabelle_ticket_cost_l224_224181

theorem Isabelle_ticket_cost :
  (∀ (week_salary : ℕ) (weeks_worked : ℕ) (brother_ticket_cost : ℕ) (brothers_saved : ℕ) (Isabelle_saved : ℕ),
  week_salary = 3 ∧ weeks_worked = 10 ∧ brother_ticket_cost = 10 ∧ brothers_saved = 5 ∧ Isabelle_saved = 5 →
  Isabelle_saved + (week_salary * weeks_worked) - ((brother_ticket_cost * 2) - brothers_saved) = 15) :=
by
  sorry

end Isabelle_ticket_cost_l224_224181


namespace cricket_innings_l224_224088

theorem cricket_innings (n : ℕ) (h1 : (32 * n + 137) / (n + 1) = 37) : n = 20 :=
sorry

end cricket_innings_l224_224088


namespace find_A_students_l224_224323

variables (Alan Beth Carlos Diana : Prop)
variable (num_As : ℕ)

def Alan_implies_Beth := Alan → Beth
def Beth_implies_no_Carlos_A := Beth → ¬Carlos
def Carlos_implies_Diana := Carlos → Diana
def Beth_implies_Diana := Beth → Diana

theorem find_A_students 
  (h1 : Alan_implies_Beth Alan Beth)
  (h2 : Beth_implies_no_Carlos_A Beth Carlos)
  (h3 : Carlos_implies_Diana Carlos Diana)
  (h4 : Beth_implies_Diana Beth Diana)
  (h_cond : num_As = 2) :
  (Alan ∧ Beth) ∨ (Beth ∧ Diana) ∨ (Carlos ∧ Diana) :=
by sorry

end find_A_students_l224_224323


namespace contrapositive_of_given_condition_l224_224881

-- Definitions
variable (P Q : Prop)

-- Given condition: If Jane answered all questions correctly, she will get a prize
axiom h : P → Q

-- Statement to be proven: If Jane did not get a prize, she answered at least one question incorrectly
theorem contrapositive_of_given_condition : ¬ Q → ¬ P := by
  sorry

end contrapositive_of_given_condition_l224_224881


namespace paint_more_expensive_than_wallpaper_l224_224209

variable (x y z : ℝ)
variable (h : 4 * x + 4 * y = 7 * x + 2 * y + z)

theorem paint_more_expensive_than_wallpaper : y > x :=
by
  sorry

end paint_more_expensive_than_wallpaper_l224_224209


namespace find_polynomial_l224_224961

def polynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

theorem find_polynomial
  (a b c : ℚ)
  (h1 : polynomial a b c (-3) = 0)
  (h2 : polynomial a b c 6 = 0)
  (h3 : polynomial a b c 2 = -24) :
  a = 6/5 ∧ b = -18/5 ∧ c = -108/5 :=
by 
  sorry

end find_polynomial_l224_224961


namespace monotonicity_of_f_l224_224243

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (x + 1)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → 0 < a → f a x1 < f a x2) ∧
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → a < 0 → f a x1 > f a x2) :=
by {
  sorry
}

end monotonicity_of_f_l224_224243


namespace total_amount_due_is_correct_l224_224921

-- Define the initial conditions
def initial_amount : ℝ := 350
def first_year_interest_rate : ℝ := 0.03
def second_and_third_years_interest_rate : ℝ := 0.05

-- Define the total amount calculation after three years.
def total_amount_after_three_years (P : ℝ) (r1 : ℝ) (r2 : ℝ) : ℝ :=
  let first_year_amount := P * (1 + r1)
  let second_year_amount := first_year_amount * (1 + r2)
  let third_year_amount := second_year_amount * (1 + r2)
  third_year_amount

theorem total_amount_due_is_correct : 
  total_amount_after_three_years initial_amount first_year_interest_rate second_and_third_years_interest_rate = 397.45 :=
by
  sorry

end total_amount_due_is_correct_l224_224921


namespace rahim_books_bought_l224_224120

theorem rahim_books_bought (x : ℕ) 
  (first_shop_cost second_shop_cost total_books : ℕ)
  (avg_price total_spent : ℕ)
  (h1 : first_shop_cost = 1500)
  (h2 : second_shop_cost = 340)
  (h3 : total_books = x + 60)
  (h4 : avg_price = 16)
  (h5 : total_spent = first_shop_cost + second_shop_cost)
  (h6 : avg_price = total_spent / total_books) :
  x = 55 :=
by
  sorry

end rahim_books_bought_l224_224120


namespace smallest_y_value_l224_224666

theorem smallest_y_value :
  ∃ y : ℝ, (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) ∧ ∀ z : ℝ, (3 * z ^ 2 + 33 * z - 90 = z * (z + 16)) → y ≤ z :=
sorry

end smallest_y_value_l224_224666


namespace expand_product_l224_224958

theorem expand_product :
  (3 * x + 4) * (x - 2) * (x + 6) = 3 * x^3 + 16 * x^2 - 20 * x - 48 :=
by
  sorry

end expand_product_l224_224958


namespace length_of_AC_l224_224307

theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) : AC = 30.1 :=
sorry

end length_of_AC_l224_224307


namespace third_side_correct_length_longest_side_feasibility_l224_224632

-- Definitions for part (a)
def adjacent_side_length : ℕ := 40
def total_fencing_length : ℕ := 140

-- Define third side given the conditions
def third_side_length : ℕ :=
  total_fencing_length - (2 * adjacent_side_length)

-- Problem (a)
theorem third_side_correct_length (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  third_side_length = 60 :=
sorry

-- Definitions for part (b)
def longest_side_possible1 : ℕ := 85
def longest_side_possible2 : ℕ := 65

-- Problem (b)
theorem longest_side_feasibility (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  ¬ (longest_side_possible1 = 85 ∧ longest_side_possible2 = 65) :=
sorry

end third_side_correct_length_longest_side_feasibility_l224_224632


namespace ratio_of_beef_to_pork_l224_224514

/-- 
James buys 20 pounds of beef. 
James buys an unknown amount of pork. 
James uses 1.5 pounds of meat to make each meal. 
Each meal sells for $20. 
James made $400 from selling meals.
The ratio of the amount of beef to the amount of pork James bought is 2:1.
-/
theorem ratio_of_beef_to_pork (beef pork : ℝ) (meal_weight : ℝ) (meal_price : ℝ) (total_revenue : ℝ)
  (h_beef : beef = 20)
  (h_meal_weight : meal_weight = 1.5)
  (h_meal_price : meal_price = 20)
  (h_total_revenue : total_revenue = 400) :
  (beef / pork) = 2 :=
by
  sorry

end ratio_of_beef_to_pork_l224_224514


namespace virginia_taught_fewer_years_l224_224059

variable (V A : ℕ)

theorem virginia_taught_fewer_years (h1 : V + A + 40 = 93) (h2 : V = A + 9) : 40 - V = 9 := by
  sorry

end virginia_taught_fewer_years_l224_224059


namespace isosceles_triangle_circumscribed_radius_and_height_l224_224507

/-
Conditions:
- The isosceles triangle has two equal sides of 20 inches.
- The base of the triangle is 24 inches.

Prove:
1. The radius of the circumscribed circle is 5 inches.
2. The height of the triangle is 16 inches.
-/

theorem isosceles_triangle_circumscribed_radius_and_height 
  (h_eq_sides : ∀ A B C : Type, ∀ (AB AC : ℝ), ∀ (BC : ℝ), AB = 20 → AC = 20 → BC = 24) 
  (R : ℝ) (h : ℝ) : 
  R = 5 ∧ h = 16 := 
sorry

end isosceles_triangle_circumscribed_radius_and_height_l224_224507


namespace perimeter_of_triangle_ABC_l224_224698

noncomputable def triangle_perimeter (r1 r2 r3 : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let x3 := r3 * Real.cos θ3
  let y3 := r3 * Real.sin θ3
  let d12 := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d23 := Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)
  let d31 := Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)
  d12 + d23 + d31

--prove

theorem perimeter_of_triangle_ABC (θ1 θ2 θ3: ℝ)
  (h1: θ1 - θ2 = Real.pi / 3)
  (h2: θ2 - θ3 = Real.pi / 3) :
  triangle_perimeter 4 5 7 θ1 θ2 θ3 = sorry := 
sorry

end perimeter_of_triangle_ABC_l224_224698


namespace find_f1_l224_224478

theorem find_f1 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + (-x) ^ 2 = -(f x + x ^ 2))
  (h2 : ∀ x, f (-x) + 2 ^ (-x) = f x + 2 ^ x) :
  f 1 = -7 / 4 := by
sorry

end find_f1_l224_224478


namespace simplify_evaluate_l224_224526

theorem simplify_evaluate (x y : ℝ) (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2 * y) - (x + y)^2) / y = 1 :=
by
  sorry

end simplify_evaluate_l224_224526


namespace find_a_l224_224558

noncomputable def M (a : ℤ) : Set ℤ := {a, 0}
noncomputable def N : Set ℤ := { x : ℤ | 2 * x^2 - 3 * x < 0 }

theorem find_a (a : ℤ) (h : (M a ∩ N).Nonempty) : a = 1 := sorry

end find_a_l224_224558


namespace dog_treats_cost_l224_224941

theorem dog_treats_cost
  (treats_per_day : ℕ)
  (cost_per_treat : ℚ)
  (days_in_month : ℕ)
  (H1 : treats_per_day = 2)
  (H2 : cost_per_treat = 0.1)
  (H3 : days_in_month = 30) :
  treats_per_day * days_in_month * cost_per_treat = 6 :=
by sorry

end dog_treats_cost_l224_224941


namespace same_side_of_line_l224_224045

theorem same_side_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) > 0 ↔ a < -7 ∨ a > 24 :=
by
  sorry

end same_side_of_line_l224_224045


namespace correct_order_shopping_process_l224_224811

/-- Definition of each step --/
def step1 : String := "The buyer logs into the Taobao website to select products."
def step2 : String := "The buyer selects the product, clicks the buy button, and pays through Alipay."
def step3 : String := "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company."
def step4 : String := "The buyer receives the goods, inspects them for any issues, and confirms receipt online."
def step5 : String := "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."

/-- The correct sequence of steps --/
def correct_sequence : List String := [
  "The buyer logs into the Taobao website to select products.",
  "The buyer selects the product, clicks the buy button, and pays through Alipay.",
  "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company.",
  "The buyer receives the goods, inspects them for any issues, and confirms receipt online.",
  "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."
]

theorem correct_order_shopping_process :
  [step1, step2, step3, step4, step5] = correct_sequence :=
by
  sorry

end correct_order_shopping_process_l224_224811


namespace almonds_weight_l224_224267

def nuts_mixture (almonds_ratio walnuts_ratio total_weight : ℚ) : ℚ :=
  let total_parts := almonds_ratio + walnuts_ratio
  let weight_per_part := total_weight / total_parts
  let weight_almonds := weight_per_part * almonds_ratio
  weight_almonds

theorem almonds_weight (total_weight : ℚ) (h1 : total_weight = 140) : nuts_mixture 5 1 total_weight = 116.67 :=
by
  sorry

end almonds_weight_l224_224267


namespace symmetric_point_origin_l224_224284

def Point := (ℝ × ℝ × ℝ)

def symmetric_point (P : Point) (O : Point) : Point :=
  let (x, y, z) := P
  let (ox, oy, oz) := O
  (2 * ox - x, 2 * oy - y, 2 * oz - z)

theorem symmetric_point_origin :
  symmetric_point (1, 3, 5) (0, 0, 0) = (-1, -3, -5) :=
by sorry

end symmetric_point_origin_l224_224284


namespace natural_numbers_satisfy_equation_l224_224349

theorem natural_numbers_satisfy_equation:
  ∀ (n k : ℕ), (k^5 + 5 * n^4 = 81 * k) ↔ (n = 2 ∧ k = 1) :=
by
  sorry

end natural_numbers_satisfy_equation_l224_224349


namespace solve_inequality_system_l224_224168

theorem solve_inequality_system (x : ℝ) :
  (x + 1 < 4 ∧ 1 - 3 * x ≥ -5) ↔ (x ≤ 2) :=
by
  sorry

end solve_inequality_system_l224_224168


namespace arithmetic_statement_not_basic_l224_224167

-- Define the basic algorithmic statements as a set
def basic_algorithmic_statements : Set String := 
  {"Input statement", "Output statement", "Assignment statement", "Conditional statement", "Loop statement"}

-- Define the arithmetic statement
def arithmetic_statement : String := "Arithmetic statement"

-- Prove that arithmetic statement is not a basic algorithmic statement
theorem arithmetic_statement_not_basic :
  arithmetic_statement ∉ basic_algorithmic_statements :=
sorry

end arithmetic_statement_not_basic_l224_224167


namespace other_root_is_seven_thirds_l224_224948

theorem other_root_is_seven_thirds {m : ℝ} (h : ∃ r : ℝ, 3 * r * r + m * r - 7 = 0 ∧ r = -1) : 
  ∃ r' : ℝ, r' ≠ -1 ∧ 3 * r' * r' + m * r' - 7 = 0 ∧ r' = 7 / 3 :=
by
  sorry

end other_root_is_seven_thirds_l224_224948


namespace product_of_two_numbers_l224_224849

theorem product_of_two_numbers (a b : ℕ) (H1 : Nat.gcd a b = 20) (H2 : Nat.lcm a b = 128) : a * b = 2560 :=
by
  sorry

end product_of_two_numbers_l224_224849


namespace original_cost_of_each_bag_l224_224542

theorem original_cost_of_each_bag (C : ℕ) (hC : C % 13 = 0) (h4 : (85 * C) % 400 = 0) : C / 5 = 208 := by
  sorry

end original_cost_of_each_bag_l224_224542


namespace evaluate_f_difference_l224_224893

def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

theorem evaluate_f_difference : f 5 - f (-5) = 800 := by
  sorry

end evaluate_f_difference_l224_224893


namespace num_integer_distance_pairs_5x5_grid_l224_224128

-- Define the problem conditions
def grid_size : ℕ := 5

-- Define a function to calculate the number of pairs of vertices with integer distances
noncomputable def count_integer_distance_pairs (n : ℕ) : ℕ := sorry

-- The theorem to prove
theorem num_integer_distance_pairs_5x5_grid : count_integer_distance_pairs grid_size = 108 :=
by
  sorry

end num_integer_distance_pairs_5x5_grid_l224_224128


namespace number_of_women_l224_224450

variable (W : ℕ) (x : ℝ)

-- Conditions
def daily_wage_men_and_women (W : ℕ) (x : ℝ) : Prop :=
  24 * 350 + W * x = 11600

def half_men_and_37_women (W : ℕ) (x : ℝ) : Prop :=
  12 * 350 + 37 * x = 24 * 350 + W * x

def daily_wage_man := (350 : ℝ)

-- Proposition to prove
theorem number_of_women (W : ℕ) (x : ℝ) (h1 : daily_wage_men_and_women W x)
  (h2 : half_men_and_37_women W x) : W = 16 := 
  by
  sorry

end number_of_women_l224_224450


namespace tom_age_ratio_l224_224643

-- Definitions of given conditions
variables (T N : ℕ) -- Tom's age (T) and number of years ago (N)

-- Tom's age is T years
-- The sum of the ages of Tom's three children is also T
-- N years ago, Tom's age was twice the sum of his children's ages then

theorem tom_age_ratio (h1 : T - N = 2 * (T - 3 * N)) : T / N = 5 :=
sorry

end tom_age_ratio_l224_224643


namespace cyclic_sum_inequality_l224_224033

noncomputable def cyclic_sum (f : ℝ → ℝ → ℝ) (x y z : ℝ) : ℝ :=
  f x y + f y z + f z x

theorem cyclic_sum_inequality
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x = a + (1 / b) - 1) 
  (hy : y = b + (1 / c) - 1) 
  (hz : z = c + (1 / a) - 1)
  (hpx : x > 0) (hpy : y > 0) (hpz : z > 0) :
  cyclic_sum (fun x y => (x * y) / (Real.sqrt (x * y) + 2)) x y z ≥ 1 :=
sorry

end cyclic_sum_inequality_l224_224033


namespace number_of_tickets_bought_l224_224129

noncomputable def ticketCost : ℕ := 5
noncomputable def popcornCost : ℕ := (80 * ticketCost) / 100
noncomputable def sodaCost : ℕ := (50 * popcornCost) / 100
noncomputable def totalSpent : ℕ := 36
noncomputable def numberOfPopcorns : ℕ := 2 
noncomputable def numberOfSodas : ℕ := 4

theorem number_of_tickets_bought : 
  (totalSpent - (numberOfPopcorns * popcornCost + numberOfSodas * sodaCost)) = 4 * ticketCost :=
by
  sorry

end number_of_tickets_bought_l224_224129


namespace length_of_DC_l224_224602

theorem length_of_DC (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30) (h2 : angle_ADB = pi / 2) (h3 : sin_A = 3 / 5) (h4 : sin_C = 1 / 4) :
  ∃ DC : ℝ, DC = 18 * Real.sqrt 15 :=
by
  sorry

end length_of_DC_l224_224602


namespace scientific_notation_10200000_l224_224069

theorem scientific_notation_10200000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 10.2 * 10^7 = a * 10^n := 
sorry

end scientific_notation_10200000_l224_224069


namespace problem_statement_l224_224774

variable (p q r s : ℝ) (ω : ℂ)

theorem problem_statement (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1) 
  (hω : ω ^ 4 = 1) (hω_ne : ω ≠ 1)
  (h_eq : (1 / (p + ω) + 1 / (q + ω) + 1 / (r + ω) + 1 / (s + ω)) = 3 / ω^2) :
  1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1) + 1 / (s + 1) = 3 := 
by sorry

end problem_statement_l224_224774


namespace infinitely_many_solutions_eq_l224_224499

theorem infinitely_many_solutions_eq {a b : ℝ} 
  (H : ∀ x : ℝ, a * (a - x) - b * (b - x) = 0) : a = b :=
sorry

end infinitely_many_solutions_eq_l224_224499


namespace arrangements_ABC_together_l224_224446

noncomputable def permutation_count_ABC_together (n : Nat) (unit_size : Nat) (remaining : Nat) : Nat :=
  (Nat.factorial unit_size) * (Nat.factorial (remaining + 1))

theorem arrangements_ABC_together : permutation_count_ABC_together 6 3 3 = 144 :=
by
  sorry

end arrangements_ABC_together_l224_224446


namespace training_weeks_l224_224070

variable (adoption_fee training_per_week cert_cost insurance_coverage out_of_pocket : ℕ)
variable (x : ℕ)

def adoption_fee_value : ℕ := 150
def training_per_week_cost : ℕ := 250
def certification_cost_value : ℕ := 3000
def insurance_coverage_percentage : ℕ := 90
def total_out_of_pocket : ℕ := 3450

theorem training_weeks :
  adoption_fee = adoption_fee_value →
  training_per_week = training_per_week_cost →
  cert_cost = certification_cost_value →
  insurance_coverage = insurance_coverage_percentage →
  out_of_pocket = total_out_of_pocket →
  (out_of_pocket = adoption_fee + training_per_week * x + (cert_cost * (100 - insurance_coverage)) / 100) →
  x = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end training_weeks_l224_224070


namespace sequence_a_1998_value_l224_224504

theorem sequence_a_1998_value :
  (∃ (a : ℕ → ℕ),
    (∀ n : ℕ, 0 <= a n) ∧
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ k : ℕ, ∃ i j t : ℕ, k = a i + 2 * a j + 4 * a t) ∧
    a 1998 = 1227096648) := sorry

end sequence_a_1998_value_l224_224504


namespace fuel_calculation_l224_224622

def total_fuel_needed (empty_fuel_per_mile people_fuel_per_mile bag_fuel_per_mile num_passengers num_crew bags_per_person miles : ℕ) : ℕ :=
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let total_fuel_per_mile := empty_fuel_per_mile + people_fuel_per_mile * total_people + bag_fuel_per_mile * total_bags
  total_fuel_per_mile * miles

theorem fuel_calculation :
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 :=
by
  sorry

end fuel_calculation_l224_224622


namespace new_mean_rent_is_880_l224_224661

theorem new_mean_rent_is_880
  (num_friends : ℕ)
  (initial_average_rent : ℝ)
  (increase_percentage : ℝ)
  (original_rent_increased : ℝ)
  (new_mean_rent : ℝ) :
  num_friends = 4 →
  initial_average_rent = 800 →
  increase_percentage = 20 →
  original_rent_increased = 1600 →
  new_mean_rent = 880 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_is_880_l224_224661


namespace cost_per_person_l224_224067

def total_cost : ℕ := 30000  -- Cost in million dollars
def num_people : ℕ := 300    -- Number of people in million

theorem cost_per_person : total_cost / num_people = 100 :=
by
  sorry

end cost_per_person_l224_224067


namespace solution_set_l224_224255

open BigOperators

noncomputable def f (x : ℝ) := 2016^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2016 - 2016^(-x)

theorem solution_set (x : ℝ) (h1 : ∀ x, f (-x) = -f (x)) (h2 : ∀ x1 x2, x1 < x2 → f (x1) < f (x2)) :
  x > -1 / 4 ↔ f (3 * x + 1) + f (x) > 0 := 
by
  sorry

end solution_set_l224_224255


namespace sam_morning_run_distance_l224_224623

variable (x : ℝ) -- The distance of Sam's morning run in miles

theorem sam_morning_run_distance (h1 : ∀ y, y = 2 * x) (h2 : 12 = 12) (h3 : x + 2 * x + 12 = 18) : x = 2 :=
by sorry

end sam_morning_run_distance_l224_224623


namespace range_of_a_l224_224604

noncomputable def f (x a : ℝ) : ℝ := 
  x * (a - 1 / Real.exp x)

noncomputable def gx (x : ℝ) : ℝ :=
  (1 + x) / Real.exp x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 a = 0 ∧ f x2 a = 0) →
  a < 2 / Real.exp 1 :=
by
  sorry

end range_of_a_l224_224604


namespace shooting_competition_hits_l224_224364

noncomputable def a1 : ℝ := 1
noncomputable def d : ℝ := 0.5
noncomputable def S_n (n : ℝ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

theorem shooting_competition_hits (n : ℝ) (h : S_n n = 7) : 25 - n = 21 :=
by
  -- sequence of proof steps
  sorry

end shooting_competition_hits_l224_224364


namespace initial_apples_correct_l224_224559

-- Define the conditions
def apples_handout : Nat := 5
def pies_made : Nat := 9
def apples_per_pie : Nat := 5

-- Calculate the number of apples used for pies
def apples_for_pies := pies_made * apples_per_pie

-- Define the total number of apples initially
def apples_initial := apples_for_pies + apples_handout

-- State the theorem to prove
theorem initial_apples_correct : apples_initial = 50 :=
by
  sorry

end initial_apples_correct_l224_224559


namespace tan_nine_pi_over_three_l224_224669

theorem tan_nine_pi_over_three : Real.tan (9 * Real.pi / 3) = 0 := by
  sorry

end tan_nine_pi_over_three_l224_224669


namespace find_sum_zero_l224_224441

open Complex

noncomputable def complex_numbers_satisfy (a1 a2 a3 : ℂ) : Prop :=
  a1^2 + a2^2 + a3^2 = 0 ∧
  a1^3 + a2^3 + a3^3 = 0 ∧
  a1^4 + a2^4 + a3^4 = 0

theorem find_sum_zero (a1 a2 a3 : ℂ) (h : complex_numbers_satisfy a1 a2 a3) :
  a1 + a2 + a3 = 0 :=
by {
  sorry
}

end find_sum_zero_l224_224441


namespace product_of_solutions_of_abs_equation_l224_224200

theorem product_of_solutions_of_abs_equation : 
  (∃ x1 x2 : ℝ, |5 * x1| + 2 = 47 ∧ |5 * x2| + 2 = 47 ∧ x1 ≠ x2 ∧ x1 * x2 = -81) :=
sorry

end product_of_solutions_of_abs_equation_l224_224200


namespace refrigerator_profit_l224_224565

theorem refrigerator_profit 
  (marked_price : ℝ) 
  (cost_price : ℝ) 
  (profit_margin : ℝ ) 
  (discount1 : ℝ) 
  (profit1 : ℝ)
  (discount2 : ℝ):
  profit_margin = 0.1 → 
  profit1 = 200 → 
  cost_price = 2000 → 
  discount1 = 0.8 → 
  discount2 = 0.85 → 
  discount1 * marked_price - cost_price = profit1 → 

  (discount2 * marked_price - cost_price) = 337.5 := 
by 
  intros; 
  let marked_price := 2750; 
  sorry

end refrigerator_profit_l224_224565


namespace polar_to_cartesian_coordinates_l224_224273

theorem polar_to_cartesian_coordinates (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = 5 * Real.pi / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (-Real.sqrt 3, 1) :=
by
  sorry

end polar_to_cartesian_coordinates_l224_224273


namespace system_of_equations_l224_224717

theorem system_of_equations (x y : ℝ) 
  (h1 : 2019 * x + 2020 * y = 2018) 
  (h2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 :=
by sorry

end system_of_equations_l224_224717


namespace find_x_l224_224315

-- Define the given conditions
def constant_ratio (k : ℚ) : Prop :=
  ∀ (x y : ℚ), (3 * x - 4) / (y + 15) = k

def initial_condition (k : ℚ) : Prop :=
  (3 * 5 - 4) / (4 + 15) = k

def new_condition (k : ℚ) (x : ℚ) : Prop :=
  (3 * x - 4) / 30 = k

-- Prove that x = 406/57 given the conditions
theorem find_x (k : ℚ) (x : ℚ) :
  constant_ratio k →
  initial_condition k →
  new_condition k x →
  x = 406 / 57 :=
  sorry

end find_x_l224_224315


namespace compute_expression_l224_224493

theorem compute_expression (a b c : ℝ) (h : a^3 - 6 * a^2 + 11 * a - 6 = 0 ∧ b^3 - 6 * b^2 + 11 * b - 6 = 0 ∧ c^3 - 6 * c^2 + 11 * c - 6 = 0) :
  (ab / c + bc / a + ca / b) = 49 / 6 := 
  by
  sorry -- Placeholder for the proof

end compute_expression_l224_224493


namespace additional_time_due_to_leak_l224_224139

theorem additional_time_due_to_leak (fill_time_no_leak: ℝ) (leak_empty_time: ℝ) (fill_rate_no_leak: fill_time_no_leak ≠ 0):
  (fill_time_no_leak = 3) → 
  (leak_empty_time = 12) → 
  (1 / fill_time_no_leak - 1 / leak_empty_time ≠ 0) → 
  ((1 / fill_time_no_leak - 1 / leak_empty_time) / (1 / (1 / fill_time_no_leak - 1 / leak_empty_time)) - fill_time_no_leak = 1) := 
by
  intro h_fill h_leak h_effective_rate
  sorry

end additional_time_due_to_leak_l224_224139


namespace third_quadrant_to_first_third_fourth_l224_224969

theorem third_quadrant_to_first_third_fourth (k : ℤ) (α : ℝ) 
  (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2) : 
  ∃ n : ℤ, (2 * k / 3 % 2) * Real.pi + Real.pi / 3 < α / 3 ∧ α / 3 < (2 * k / 3 % 2) * Real.pi + Real.pi / 2 ∨
            (2 * (3 * n + 1) % 2) * Real.pi + Real.pi < α / 3 ∧ α / 3 < (2 * (3 * n + 1) % 2) * Real.pi + 7 * Real.pi / 6 ∨
            (2 * (3 * n + 2) % 2) * Real.pi + 5 * Real.pi / 3 < α / 3 ∧ α / 3 < (2 * (3 * n + 2) % 2) * Real.pi + 11 * Real.pi / 6 :=
sorry

end third_quadrant_to_first_third_fourth_l224_224969


namespace rabbit_calories_l224_224015

theorem rabbit_calories (C : ℕ) :
  (6 * 300 = 2 * C + 200) → C = 800 :=
by
  intro h
  sorry

end rabbit_calories_l224_224015


namespace james_weekly_hours_l224_224443

def james_meditation_total : ℕ :=
  let weekly_minutes := (30 * 2 * 6) + (30 * 2 * 2) -- 1 hour/day for 6 days + 2 hours on Sunday
  weekly_minutes / 60

def james_yoga_total : ℕ :=
  let weekly_minutes := (45 * 2) -- 45 minutes on Monday and Friday
  weekly_minutes / 60

def james_bikeride_total : ℕ :=
  let weekly_minutes := 90
  weekly_minutes / 60

def james_dance_total : ℕ :=
  2 -- 2 hours on Saturday

def james_total_activity_hours : ℕ :=
  james_meditation_total + james_yoga_total + james_bikeride_total + james_dance_total

theorem james_weekly_hours : james_total_activity_hours = 13 := by
  sorry

end james_weekly_hours_l224_224443


namespace matrix_eigenvalue_problem_l224_224046

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l224_224046


namespace avg_goals_l224_224293

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end avg_goals_l224_224293


namespace find_special_two_digit_integer_l224_224145

theorem find_special_two_digit_integer (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : (n + 3) % 3 = 0)
  (h3 : (n + 4) % 4 = 0)
  (h4 : (n + 5) % 5 = 0) :
  n = 60 := by
  sorry

end find_special_two_digit_integer_l224_224145


namespace polygon_sides_l224_224094

theorem polygon_sides {n k : ℕ} (h1 : k = n * (n - 3) / 2) (h2 : k = 3 * n / 2) : n = 6 :=
by
  sorry

end polygon_sides_l224_224094


namespace first_term_geometric_series_l224_224165

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l224_224165


namespace triangle_BD_length_l224_224346

theorem triangle_BD_length 
  (A B C D : Type) 
  (hAC : AC = 8) 
  (hBC : BC = 8) 
  (hAD : AD = 6) 
  (hCD : CD = 5) : BD = 6 :=
  sorry

end triangle_BD_length_l224_224346


namespace transformed_inequality_l224_224629

theorem transformed_inequality (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by
  sorry

end transformed_inequality_l224_224629


namespace exists_distinct_abc_sum_l224_224151

theorem exists_distinct_abc_sum (n : ℕ) (h : n ≥ 1) (X : Finset ℤ)
  (h_card : X.card = n + 2)
  (h_abs : ∀ x ∈ X, abs x ≤ n) :
  ∃ (a b c : ℤ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

end exists_distinct_abc_sum_l224_224151


namespace m_value_l224_224708

theorem m_value (m : ℝ) (h : (243:ℝ) ^ (1/3) = 3 ^ m) : m = 5 / 3 :=
sorry

end m_value_l224_224708


namespace find_x_for_fraction_equality_l224_224575

theorem find_x_for_fraction_equality (x : ℝ) : 
  (4 + 2 * x) / (7 + x) = (2 + x) / (3 + x) ↔ (x = -2 ∨ x = 1) := by
  sorry

end find_x_for_fraction_equality_l224_224575


namespace probability_each_person_selected_l224_224589

-- Define the number of initial participants
def initial_participants := 2007

-- Define the number of participants to exclude
def exclude_participants := 7

-- Define the final number of participants remaining after exclusion
def remaining_participants := initial_participants - exclude_participants

-- Define the number of participants to select
def select_participants := 50

-- Define the probability of each participant being selected
def selection_probability : ℚ :=
  select_participants * remaining_participants / (initial_participants * remaining_participants)

theorem probability_each_person_selected :
  selection_probability = (50 / 2007 : ℚ) :=
sorry

end probability_each_person_selected_l224_224589


namespace value_of_x_that_makes_sqrt_undefined_l224_224658

theorem value_of_x_that_makes_sqrt_undefined (x : ℕ) (hpos : 0 < x) : (x = 1) ∨ (x = 2) ↔ (x - 3 < 0) := by
  sorry

end value_of_x_that_makes_sqrt_undefined_l224_224658


namespace tan_theta_equation_l224_224802

theorem tan_theta_equation (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) :
  Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  sorry

end tan_theta_equation_l224_224802


namespace max_xy_l224_224190

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end max_xy_l224_224190


namespace older_brother_catches_up_l224_224577

-- Define the initial conditions and required functions
def younger_brother_steps_before_chase : ℕ := 10
def time_per_3_steps_older := 1  -- in seconds
def time_per_4_steps_younger := 1  -- in seconds 
def dist_older_in_5_steps : ℕ := 7  -- 7d_younger / 5
def dist_younger_in_7_steps : ℕ := 5
def speed_older : ℕ := 3 * dist_older_in_5_steps / 5  -- steps/second 
def speed_younger : ℕ := 4 * dist_younger_in_7_steps / 7  -- steps/second

theorem older_brother_catches_up : ∃ n : ℕ, n = 150 :=
by sorry  -- final theorem statement with proof omitted

end older_brother_catches_up_l224_224577


namespace larger_number_is_1641_l224_224338

theorem larger_number_is_1641 (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 6 * S + 15) : L = 1641 :=
by
  sorry

end larger_number_is_1641_l224_224338


namespace part1_part2_l224_224519

noncomputable def f (x : ℝ) := |x - 3| + |x - 4|

theorem part1 (a : ℝ) (h : ∃ x : ℝ, f x < a) : a > 1 :=
sorry

theorem part2 (x : ℝ) : f x ≥ 7 + 7 * x - x ^ 2 ↔ x ≤ 0 ∨ 7 ≤ x :=
sorry

end part1_part2_l224_224519


namespace find_q_l224_224823

noncomputable def p (q : ℝ) : ℝ := 16 / (3 * q)

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 3/2) (h4 : p * q = 16/3) : q = 24 / 6 + 19.6 / 6 :=
by
  sorry

end find_q_l224_224823


namespace rectangle_area_invariant_l224_224265

theorem rectangle_area_invariant
    (x y : ℕ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 3) * (y + 2)) :
    x * y = 15 :=
by sorry

end rectangle_area_invariant_l224_224265


namespace evaporation_period_length_l224_224250

def initial_water_amount : ℝ := 10
def daily_evaporation_rate : ℝ := 0.0008
def percentage_evaporated : ℝ := 0.004  -- 0.4% expressed as a decimal

theorem evaporation_period_length :
  (percentage_evaporated * initial_water_amount) / daily_evaporation_rate = 50 := by
  sorry

end evaporation_period_length_l224_224250


namespace fraction_meaningful_condition_l224_224225

-- Define a variable x
variable (x : ℝ)

-- State the condition that makes the fraction meaningful
def fraction_meaningful (x : ℝ) : Prop := (x - 2) ≠ 0

-- State the theorem we want to prove
theorem fraction_meaningful_condition : fraction_meaningful x ↔ x ≠ 2 := sorry

end fraction_meaningful_condition_l224_224225


namespace sin_70_equals_1_minus_2a_squared_l224_224365

variable (a : ℝ)

theorem sin_70_equals_1_minus_2a_squared (h : Real.sin (10 * Real.pi / 180) = a) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * a^2 := 
sorry

end sin_70_equals_1_minus_2a_squared_l224_224365


namespace c_minus_a_equals_90_l224_224915

variable (a b c : ℝ)

def average_a_b (a b : ℝ) : Prop := (a + b) / 2 = 45
def average_b_c (b c : ℝ) : Prop := (b + c) / 2 = 90

theorem c_minus_a_equals_90
  (h1 : average_a_b a b)
  (h2 : average_b_c b c) :
  c - a = 90 :=
  sorry

end c_minus_a_equals_90_l224_224915


namespace units_digit_base7_product_l224_224988

theorem units_digit_base7_product (a b : ℕ) (ha : a = 354) (hb : b = 78) : (a * b) % 7 = 4 := by
  sorry

end units_digit_base7_product_l224_224988


namespace cat_food_more_than_dog_food_l224_224212

theorem cat_food_more_than_dog_food :
  let cat_food_packs := 6
  let cans_per_cat_pack := 9
  let dog_food_packs := 2
  let cans_per_dog_pack := 3
  let total_cat_food_cans := cat_food_packs * cans_per_cat_pack
  let total_dog_food_cans := dog_food_packs * cans_per_dog_pack
  total_cat_food_cans - total_dog_food_cans = 48 :=
by
  sorry

end cat_food_more_than_dog_food_l224_224212


namespace football_game_attendance_l224_224171

theorem football_game_attendance :
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  wednesday - monday = 50 :=
by
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  show wednesday - monday = 50
  sorry

end football_game_attendance_l224_224171


namespace product_of_undefined_x_l224_224337

-- Define the quadratic equation condition
def quad_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The main theorem to prove the product of all x such that the expression is undefined
theorem product_of_undefined_x :
  (∃ x₁ x₂ : ℝ, quad_eq 1 4 3 x₁ ∧ quad_eq 1 4 3 x₂ ∧ x₁ * x₂ = 3) :=
by
  sorry

end product_of_undefined_x_l224_224337


namespace cannot_factor_polynomial_l224_224607

theorem cannot_factor_polynomial (a b c d : ℤ) :
  ¬(x^4 + 3 * x^3 + 6 * x^2 + 9 * x + 12 = (x^2 + a * x + b) * (x^2 + c * x + d)) := 
by {
  sorry
}

end cannot_factor_polynomial_l224_224607


namespace min_value_of_b_plus_3_div_a_l224_224644

theorem min_value_of_b_plus_3_div_a (a : ℝ) (b : ℝ) :
  0 < a →
  (∀ x, 0 < x → (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) →
  b + 3 / a = 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_b_plus_3_div_a_l224_224644


namespace polynomial_coeff_sum_l224_224313

variable (d : ℤ)
variable (h : d ≠ 0)

theorem polynomial_coeff_sum : 
  (∃ a b c e : ℤ, (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e ∧ a + b + c + e = 42) :=
by
  sorry

end polynomial_coeff_sum_l224_224313


namespace arithmetic_expression_evaluation_l224_224541

theorem arithmetic_expression_evaluation :
  4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end arithmetic_expression_evaluation_l224_224541


namespace train_speed_l224_224960

theorem train_speed (distance_AB : ℕ) (start_time_A : ℕ) (start_time_B : ℕ) (meet_time : ℕ) (speed_B : ℕ) (time_travel_A : ℕ) (time_travel_B : ℕ)
  (total_distance : ℕ) (distance_B_covered : ℕ) (speed_A : ℕ)
  (h1 : distance_AB = 330)
  (h2 : start_time_A = 8)
  (h3 : start_time_B = 9)
  (h4 : meet_time = 11)
  (h5 : speed_B = 75)
  (h6 : time_travel_A = meet_time - start_time_A)
  (h7 : time_travel_B = meet_time - start_time_B)
  (h8 : distance_B_covered = time_travel_B * speed_B)
  (h9 : total_distance = distance_AB)
  (h10 : total_distance = time_travel_A * speed_A + distance_B_covered):
  speed_A = 60 := 
by
  sorry

end train_speed_l224_224960


namespace number_of_fence_panels_is_10_l224_224223

def metal_rods_per_sheet := 10
def metal_rods_per_beam := 4
def sheets_per_panel := 3
def beams_per_panel := 2
def total_metal_rods := 380

theorem number_of_fence_panels_is_10 :
  (total_metal_rods = 380) →
  (metal_rods_per_sheet = 10) →
  (metal_rods_per_beam = 4) →
  (sheets_per_panel = 3) →
  (beams_per_panel = 2) →
  380 / (3 * 10 + 2 * 4) = 10 := 
by 
  sorry

end number_of_fence_panels_is_10_l224_224223


namespace first_positive_term_is_7_l224_224720

-- Define the conditions and the sequence
def a1 : ℚ := -1
def d : ℚ := 1 / 5

-- Define the general term of the sequence
def a_n (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Define the proposition that the 7th term is the first positive term
theorem first_positive_term_is_7 :
  ∀ n : ℕ, (0 < a_n n) → (7 <= n) :=
by
  intro n h
  sorry

end first_positive_term_is_7_l224_224720


namespace percentage_of_girls_after_changes_l224_224110

theorem percentage_of_girls_after_changes :
  let boys_classA := 15
  let girls_classA := 20
  let boys_classB := 25
  let girls_classB := 35
  let boys_transferAtoB := 3
  let girls_transferAtoB := 2
  let boys_joiningA := 4
  let girls_joiningA := 6

  let boys_classA_after := boys_classA - boys_transferAtoB + boys_joiningA
  let girls_classA_after := girls_classA - girls_transferAtoB + girls_joiningA
  let boys_classB_after := boys_classB + boys_transferAtoB
  let girls_classB_after := girls_classB + girls_transferAtoB

  let total_students := boys_classA_after + girls_classA_after + boys_classB_after + girls_classB_after
  let total_girls := girls_classA_after + girls_classB_after 

  (total_girls / total_students : ℝ) * 100 = 58.095 := by
  sorry

end percentage_of_girls_after_changes_l224_224110


namespace parabolas_intersect_on_circle_l224_224924

theorem parabolas_intersect_on_circle :
  let parabola1 (x y : ℝ) := y = (x - 2)^2
  let parabola2 (x y : ℝ) := x + 6 = (y + 1)^2
  ∃ (cx cy r : ℝ), ∀ (x y : ℝ), (parabola1 x y ∧ parabola2 x y) → (x - cx)^2 + (y - cy)^2 = r^2 ∧ r^2 = 33/2 :=
by
  sorry

end parabolas_intersect_on_circle_l224_224924


namespace average_age_of_community_l224_224742

theorem average_age_of_community 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_age_women : ℝ := 30)
  (avg_age_men : ℝ := 35)
  (total_women_age : ℝ := avg_age_women * hwomen)
  (total_men_age : ℝ := avg_age_men * hmen)
  (total_population : ℕ := hwomen + hmen)
  (total_age : ℝ := total_women_age + total_men_age) : 
  total_age / total_population = 32 + 1 / 12 :=
by
  sorry

end average_age_of_community_l224_224742


namespace sweaters_to_wash_l224_224807

theorem sweaters_to_wash (pieces_per_load : ℕ) (total_loads : ℕ) (shirts_to_wash : ℕ) 
  (h1 : pieces_per_load = 5) (h2 : total_loads = 9) (h3 : shirts_to_wash = 43) : ℕ :=
  if total_loads * pieces_per_load - shirts_to_wash = 2 then 2 else 0

end sweaters_to_wash_l224_224807


namespace three_digit_number_l224_224081

-- Define the variables involved.
variables (a b c n : ℕ)

-- Condition 1: c = 3a
def condition1 (a c : ℕ) : Prop := c = 3 * a

-- Condition 2: n is three-digit number constructed from a, b, and c.
def is_three_digit (a b c n : ℕ) : Prop := n = 100 * a + 10 * b + c

-- Condition 3: n leaves a remainder of 4 when divided by 5.
def condition2 (n : ℕ) : Prop := n % 5 = 4

-- Condition 4: n leaves a remainder of 3 when divided by 11.
def condition3 (n : ℕ) : Prop := n % 11 = 3

-- Define the main theorem
theorem three_digit_number (a b c n : ℕ) 
(h1: condition1 a c) 
(h2: is_three_digit a b c n) 
(h3: condition2 n) 
(h4: condition3 n) : 
n = 359 := 
sorry

end three_digit_number_l224_224081


namespace inverse_proposition_l224_224986

theorem inverse_proposition (a b : ℝ) (h1 : a < 1) (h2 : b < 1) : a + b ≠ 2 :=
by sorry

end inverse_proposition_l224_224986


namespace inequality_holds_l224_224796

theorem inequality_holds (x : ℝ) (hx : 0 < x ∧ x < 4) :
  ∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y) :=
by
  intros y hy_gt_zero
  sorry

end inequality_holds_l224_224796


namespace problem1_problem2_l224_224276

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end problem1_problem2_l224_224276


namespace mall_incur_1_percent_loss_l224_224552

theorem mall_incur_1_percent_loss
  (a b x : ℝ)
  (ha : x = a * 1.1)
  (hb : x = b * 0.9) :
  (2 * x - (a + b)) / (a + b) = -0.01 :=
sorry

end mall_incur_1_percent_loss_l224_224552


namespace window_width_l224_224092

theorem window_width (length area : ℝ) (h_length : length = 6) (h_area : area = 60) :
  area / length = 10 :=
by
  sorry

end window_width_l224_224092


namespace m_range_for_circle_l224_224882

def is_circle (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m - 3) * x + 2 * y + 5 = 0

theorem m_range_for_circle (m : ℝ) :
  (∀ x y : ℝ, is_circle x y m) → ((m > 5) ∨ (m < 1)) :=
by 
  sorry -- Proof not required

end m_range_for_circle_l224_224882


namespace isosceles_trapezoid_l224_224825

-- Define a type for geometric points
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define structures for geometric properties
structure Trapezoid :=
  (A B C D M N : Point)
  (is_midpoint_M : 2 * M.x = A.x + B.x ∧ 2 * M.y = A.y + B.y)
  (is_midpoint_N : 2 * N.x = C.x + D.x ∧ 2 * N.y = C.y + D.y)
  (AB_parallel_CD : (B.y - A.y) * (D.x - C.x) = (B.x - A.x) * (D.y - C.y)) -- AB || CD
  (MN_perpendicular_AB_CD : (N.y - M.y) * (B.y - A.y) + (N.x - M.x) * (B.x - A.x) = 0 ∧
                            (N.y - M.y) * (D.y - C.y) + (N.x - M.x) * (D.x - C.x) = 0) -- MN ⊥ AB && MN ⊥ CD

-- The isosceles condition
def is_isosceles (T : Trapezoid) : Prop :=
  ((T.A.x - T.D.x) ^ 2 + (T.A.y - T.D.y) ^ 2) = ((T.B.x - T.C.x) ^ 2 + (T.B.y - T.C.y) ^ 2)

-- The theorem statement
theorem isosceles_trapezoid (T : Trapezoid) : is_isosceles T :=
by
  sorry

end isosceles_trapezoid_l224_224825


namespace area_comparison_perimeter_comparison_l224_224814

-- Define side length of square and transformation to sides of the rectangle
variable (a : ℝ)

-- Conditions: side lengths of the rectangle relative to the square
def long_side : ℝ := 1.11 * a
def short_side : ℝ := 0.9 * a

-- Area calculations and comparison
def square_area : ℝ := a^2
def rectangle_area : ℝ := long_side a * short_side a

theorem area_comparison : (rectangle_area a / square_area a) = 0.999 := by
  sorry

-- Perimeter calculations and comparison
def square_perimeter : ℝ := 4 * a
def rectangle_perimeter : ℝ := 2 * (long_side a + short_side a)

theorem perimeter_comparison : (rectangle_perimeter a / square_perimeter a) = 1.005 := by
  sorry

end area_comparison_perimeter_comparison_l224_224814


namespace sum_infinite_series_l224_224702

theorem sum_infinite_series : 
  ∑' n : ℕ, (3 * (n + 1) - 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3)) = 73 / 12 := 
by sorry

end sum_infinite_series_l224_224702


namespace floor_expression_bounds_l224_224636

theorem floor_expression_bounds (x : ℝ) (h : ⌊x * ⌊x / 2⌋⌋ = 12) : 
  4.9 ≤ x ∧ x < 5.1 :=
sorry

end floor_expression_bounds_l224_224636


namespace proof_correctness_l224_224460

-- Define the new operation
def new_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Definitions for the conclusions
def conclusion_1 : Prop := new_op 1 (-2) = -8
def conclusion_2 : Prop := ∀ a b : ℝ, new_op a b = new_op b a
def conclusion_3 : Prop := ∀ a b : ℝ, new_op a b = 0 → a = 0
def conclusion_4 : Prop := ∀ a b : ℝ, a + b = 0 → (new_op a a + new_op b b = 8 * a^2)

-- Specify the correct conclusions
def correct_conclusions : Prop := conclusion_1 ∧ conclusion_2 ∧ ¬conclusion_3 ∧ conclusion_4

-- State the theorem
theorem proof_correctness : correct_conclusions := by
  sorry

end proof_correctness_l224_224460


namespace find_reggie_long_shots_l224_224580

-- Define the constants used in the problem
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shooting results
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := sorry -- we need to find this

-- Define Reggie's brother's shooting results
def brother_long_shots : ℕ := 4

-- Given conditions
def reggie_total_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
def brother_total_points := brother_long_shots * long_shot_points

def reggie_lost_by_2_points := reggie_total_points + 2 = brother_total_points

-- The theorem we need to prove
theorem find_reggie_long_shots : reggie_long_shots = 1 :=
by
  sorry

end find_reggie_long_shots_l224_224580


namespace find_number_l224_224699

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l224_224699


namespace tens_digit_of_even_not_divisible_by_10_l224_224439

theorem tens_digit_of_even_not_divisible_by_10 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) :
  (N ^ 20) % 100 / 10 % 10 = 7 :=
sorry

end tens_digit_of_even_not_divisible_by_10_l224_224439


namespace same_terminal_side_l224_224626

open Real

theorem same_terminal_side (k : ℤ) : (∃ k : ℤ, k * 360 - 315 = 9 / 4 * 180) :=
by
  sorry

end same_terminal_side_l224_224626


namespace cousin_typing_time_l224_224125

theorem cousin_typing_time (speed_ratio : ℕ) (my_time_hours : ℕ) (minutes_per_hour : ℕ) (my_time_minutes : ℕ) :
  speed_ratio = 4 →
  my_time_hours = 3 →
  minutes_per_hour = 60 →
  my_time_minutes = my_time_hours * minutes_per_hour →
  ∃ (cousin_time : ℕ), cousin_time = my_time_minutes / speed_ratio := by
  sorry

end cousin_typing_time_l224_224125


namespace x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l224_224684

theorem x_is_sufficient_but_not_necessary_for_x_squared_eq_one : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
by
  sorry

end x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l224_224684


namespace a_5_eq_16_S_8_eq_255_l224_224466

open Nat

-- Definitions from the conditions
def a : ℕ → ℕ
| 0     => 1
| (n+1) => 2 * a n

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Proof problem statements
theorem a_5_eq_16 : a 4 = 16 := sorry

theorem S_8_eq_255 : S 8 = 255 := sorry

end a_5_eq_16_S_8_eq_255_l224_224466


namespace plastering_cost_correct_l224_224140

noncomputable def tank_length : ℝ := 25
noncomputable def tank_width : ℝ := 12
noncomputable def tank_depth : ℝ := 6
noncomputable def cost_per_sqm_paise : ℝ := 75
noncomputable def cost_per_sqm_rupees : ℝ := cost_per_sqm_paise / 100

noncomputable def total_cost_plastering : ℝ :=
  let long_wall_area := 2 * (tank_length * tank_depth)
  let short_wall_area := 2 * (tank_width * tank_depth)
  let bottom_area := tank_length * tank_width
  let total_area := long_wall_area + short_wall_area + bottom_area
  total_area * cost_per_sqm_rupees

theorem plastering_cost_correct : total_cost_plastering = 558 := by
  sorry

end plastering_cost_correct_l224_224140


namespace distribute_7_balls_into_4_boxes_l224_224761

-- Define the problem conditions
def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  if balls < boxes then 0 else Nat.choose (balls - 1) (boxes - 1)

-- Prove the specific case
theorem distribute_7_balls_into_4_boxes : number_of_ways_to_distribute_balls 7 4 = 20 :=
by
  -- Definition and proof to be filled
  sorry

end distribute_7_balls_into_4_boxes_l224_224761


namespace cos_half_pi_plus_alpha_l224_224598

theorem cos_half_pi_plus_alpha (α : ℝ) (h : Real.sin (π - α) = 1 / 3) : Real.cos (π / 2 + α) = - (1 / 3) :=
by
  sorry

end cos_half_pi_plus_alpha_l224_224598


namespace simplify_expression_l224_224759

variable (y : ℤ)

theorem simplify_expression : 5 * y + 7 * y - 3 * y = 9 * y := by
  sorry

end simplify_expression_l224_224759


namespace spheres_in_base_l224_224333

theorem spheres_in_base (n : ℕ) (T_n : ℕ) (total_spheres : ℕ) :
  (total_spheres = 165) →
  (total_spheres = (1 / 6 : ℚ) * ↑n * ↑(n + 1) * ↑(n + 2)) →
  (T_n = n * (n + 1) / 2) →
  n = 9 →
  T_n = 45 :=
by
  intros _ _ _ _
  sorry

end spheres_in_base_l224_224333


namespace peggy_buys_three_folders_l224_224642

theorem peggy_buys_three_folders 
  (red_sheets : ℕ) (green_sheets : ℕ) (blue_sheets : ℕ)
  (red_stickers_per_sheet : ℕ) (green_stickers_per_sheet : ℕ) (blue_stickers_per_sheet : ℕ)
  (total_stickers : ℕ) :
  red_sheets = 10 →
  green_sheets = 10 →
  blue_sheets = 10 →
  red_stickers_per_sheet = 3 →
  green_stickers_per_sheet = 2 →
  blue_stickers_per_sheet = 1 →
  total_stickers = 60 →
  1 + 1 + 1 = 3 :=
by 
  intros _ _ _ _ _ _ _
  sorry

end peggy_buys_three_folders_l224_224642


namespace angle_C_is_3pi_over_4_l224_224741

theorem angle_C_is_3pi_over_4 (A B C : ℝ) (a b c : ℝ) (h_tri : 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (h_eq : b * Real.cos C + c * Real.sin B = 0) : C = 3 * π / 4 :=
by
  sorry

end angle_C_is_3pi_over_4_l224_224741


namespace nth_inequality_l224_224625

theorem nth_inequality (n : ℕ) (x : ℝ) (h : x > 0) : x + n^n / x^n ≥ n + 1 := 
sorry

end nth_inequality_l224_224625


namespace employee_selection_l224_224056

theorem employee_selection
  (total_employees : ℕ)
  (under_35 : ℕ)
  (between_35_and_49 : ℕ)
  (over_50 : ℕ)
  (selected_employees : ℕ) :
  total_employees = 500 →
  under_35 = 125 →
  between_35_and_49 = 280 →
  over_50 = 95 →
  selected_employees = 100 →
  (under_35 * selected_employees / total_employees = 25) ∧
  (between_35_and_49 * selected_employees / total_employees = 56) ∧
  (over_50 * selected_employees / total_employees = 19) := by
  intros h1 h2 h3 h4 h5
  sorry

end employee_selection_l224_224056


namespace sequence_a10_l224_224788

theorem sequence_a10 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n+1) - a n = 1 / (4 * ↑n^2 - 1)) :
  a 10 = 28 / 19 :=
by
  sorry

end sequence_a10_l224_224788


namespace problem1_solution_problem2_solution_l224_224105

-- Problem 1: 
theorem problem1_solution (x : ℝ) (h : 4 * x^2 = 9) : x = 3 / 2 ∨ x = - (3 / 2) := 
by sorry

-- Problem 2: 
theorem problem2_solution (x : ℝ) (h : (1 - 2 * x)^3 = 8) : x = - 1 / 2 := 
by sorry

end problem1_solution_problem2_solution_l224_224105


namespace find_s_l224_224416

theorem find_s 
  (a b c x s z : ℕ)
  (h1 : a + b = x)
  (h2 : x + c = s)
  (h3 : s + a = z)
  (h4 : b + c + z = 16) : 
  s = 8 := 
sorry

end find_s_l224_224416


namespace digit_is_two_l224_224984

theorem digit_is_two (d : ℕ) (h : d < 10) : (∃ k : ℤ, d - 2 = 11 * k) ↔ d = 2 := 
by sorry

end digit_is_two_l224_224984


namespace divisor_of_polynomial_l224_224697

theorem divisor_of_polynomial (a : ℤ) (h : ∀ x : ℤ, (x^2 - x + a) ∣ (x^13 + x + 180)) : a = 1 :=
sorry

end divisor_of_polynomial_l224_224697


namespace equivalent_expression_l224_224310

theorem equivalent_expression (m n : ℕ) (P Q : ℕ) (hP : P = 3^m) (hQ : Q = 5^n) :
  15^(m + n) = P * Q :=
by
  sorry

end equivalent_expression_l224_224310


namespace circular_board_area_l224_224640

theorem circular_board_area (C : ℝ) (R T : ℝ) (h1 : R = 62.8) (h2 : T = 10) (h3 : C = R / T) (h4 : C = 2 * Real.pi) : 
  ∀ r A : ℝ, (r = C / (2 * Real.pi)) → (A = Real.pi * r^2)  → A = Real.pi :=
by
  intro r A
  intro hr hA
  sorry

end circular_board_area_l224_224640


namespace triangle_find_C_angle_triangle_find_perimeter_l224_224650

variable (A B C a b c : ℝ)

theorem triangle_find_C_angle
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c) :
  C = π / 3 :=
sorry

theorem triangle_find_perimeter
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h2 : c = Real.sqrt 7)
  (h3 : a * b = 6) :
  a + b + c = 5 + Real.sqrt 7 :=
sorry

end triangle_find_C_angle_triangle_find_perimeter_l224_224650


namespace three_consecutive_arithmetic_l224_224601

def seq (n : ℕ) : ℝ := 
  if n % 2 = 1 then (n : ℝ)
  else 2 * 3^(n / 2 - 1)

theorem three_consecutive_arithmetic (m : ℕ) (h_m : seq m + seq (m+2) = 2 * seq (m+1)) : m = 1 :=
  sorry

end three_consecutive_arithmetic_l224_224601


namespace haley_marble_distribution_l224_224517

theorem haley_marble_distribution (total_marbles : ℕ) (num_boys : ℕ) (h1 : total_marbles = 20) (h2 : num_boys = 2) : (total_marbles / num_boys) = 10 := 
by 
  sorry

end haley_marble_distribution_l224_224517


namespace motorist_travel_distance_l224_224387

def total_distance_traveled (time_first_half time_second_half speed_first_half speed_second_half : ℕ) : ℕ :=
  (speed_first_half * time_first_half) + (speed_second_half * time_second_half)

theorem motorist_travel_distance :
  total_distance_traveled 3 3 60 48 = 324 :=
by sorry

end motorist_travel_distance_l224_224387


namespace baseball_team_groups_l224_224154

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) (h_new : new_players = 48) (h_return : returning_players = 6) (h_per_group : players_per_group = 6) : (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end baseball_team_groups_l224_224154


namespace total_trips_correct_l224_224902

-- Define Timothy's movie trips in 2009
def timothy_2009_trips : ℕ := 24

-- Define Timothy's movie trips in 2010
def timothy_2010_trips : ℕ := timothy_2009_trips + 7

-- Define Theresa's movie trips in 2009
def theresa_2009_trips : ℕ := timothy_2009_trips / 2

-- Define Theresa's movie trips in 2010
def theresa_2010_trips : ℕ := timothy_2010_trips * 2

-- Define the total number of trips for Timothy and Theresa in 2009 and 2010
def total_trips : ℕ := (timothy_2009_trips + timothy_2010_trips) + (theresa_2009_trips + theresa_2010_trips)

-- Prove the total number of trips is 129
theorem total_trips_correct : total_trips = 129 :=
by
  sorry

end total_trips_correct_l224_224902


namespace big_al_ate_40_bananas_on_june_7_l224_224896

-- Given conditions
def bananas_eaten_on_day (initial_bananas : ℕ) (day : ℕ) : ℕ :=
  initial_bananas + 4 * (day - 1)

def total_bananas_eaten (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 1 +
  bananas_eaten_on_day initial_bananas 2 +
  bananas_eaten_on_day initial_bananas 3 +
  bananas_eaten_on_day initial_bananas 4 +
  bananas_eaten_on_day initial_bananas 5 +
  bananas_eaten_on_day initial_bananas 6 +
  bananas_eaten_on_day initial_bananas 7

noncomputable def final_bananas_on_june_7 (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 7

-- Theorem to be proved
theorem big_al_ate_40_bananas_on_june_7 :
  ∃ initial_bananas, total_bananas_eaten initial_bananas = 196 ∧ final_bananas_on_june_7 initial_bananas = 40 :=
sorry

end big_al_ate_40_bananas_on_june_7_l224_224896


namespace minimum_value_l224_224584

open Real

theorem minimum_value (x : ℝ) (hx : x > 2) : 
  ∃ y ≥ 4 * Real.sqrt 2, ∀ z, (z = (x + 6) / (Real.sqrt (x - 2)) → y ≤ z) := 
sorry

end minimum_value_l224_224584


namespace min_distance_from_C_to_circle_l224_224712

theorem min_distance_from_C_to_circle
  (R : ℝ) (AC : ℝ) (CB : ℝ) (C M : ℝ)
  (hR : R = 6) (hAC : AC = 4) (hCB : CB = 5)
  (hCM_eq : C = 12 - M) :
  C * M = 20 → (M < 6) → M = 2 := 
sorry

end min_distance_from_C_to_circle_l224_224712


namespace geometric_sequence_third_fourth_terms_l224_224716

theorem geometric_sequence_third_fourth_terms
  (a : ℕ → ℝ)
  (r : ℝ)
  (ha : ∀ n, a (n + 1) = r * a n)
  (hS2 : a 0 + a 1 = 3 * a 1) :
  (a 2 + a 3) / (a 0 + a 1) = 1 / 4 :=
by
  -- proof to be filled in
  sorry

end geometric_sequence_third_fourth_terms_l224_224716


namespace greatest_value_of_b_l224_224906

theorem greatest_value_of_b (b : ℝ) : -b^2 + 8 * b - 15 ≥ 0 → b ≤ 5 := sorry

end greatest_value_of_b_l224_224906


namespace find_b_perpendicular_l224_224479

theorem find_b_perpendicular
  (b : ℝ)
  (line1 : ∀ x y : ℝ, 2 * x - 3 * y + 5 = 0)
  (line2 : ∀ x y : ℝ, b * x - 3 * y + 1 = 0)
  (perpendicular : (2 / 3) * (b / 3) = -1)
  : b = -9/2 :=
sorry

end find_b_perpendicular_l224_224479


namespace seashells_given_l224_224413

theorem seashells_given (initial left given : ℕ) (h1 : initial = 8) (h2 : left = 2) (h3 : given = initial - left) : given = 6 := by
  sorry

end seashells_given_l224_224413


namespace range_of_b_l224_224272

theorem range_of_b (M : Set (ℝ × ℝ)) (N : ℝ → ℝ → Set (ℝ × ℝ)) :
  (∀ m : ℝ, (∃ x y : ℝ, (x, y) ∈ M ∧ (x, y) ∈ (N m b))) ↔ b ∈ Set.Icc (- Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by
  sorry

end range_of_b_l224_224272


namespace donation_calculation_l224_224222

/-- Patricia's initial hair length -/
def initial_length : ℕ := 14

/-- Patricia's hair growth -/
def growth_length : ℕ := 21

/-- Desired remaining hair length after donation -/
def remaining_length : ℕ := 12

/-- Calculate the donation length -/
def donation_length (L G R : ℕ) : ℕ := (L + G) - R

-- Theorem stating the donation length required for Patricia to achieve her goal.
theorem donation_calculation : donation_length initial_length growth_length remaining_length = 23 :=
by
  -- Proof omitted
  sorry

end donation_calculation_l224_224222


namespace factor_expression_l224_224993

theorem factor_expression (x : ℝ) : 
  5 * x * (x - 2) + 9 * (x - 2) - 4 * (x - 2) = 5 * (x - 2) * (x + 1) :=
by
  -- proof goes here
  sorry

end factor_expression_l224_224993


namespace patio_perimeter_is_100_feet_l224_224540

theorem patio_perimeter_is_100_feet
  (rectangle : Prop)
  (length : ℝ)
  (width : ℝ)
  (length_eq_40 : length = 40)
  (length_eq_4_times_width : length = 4 * width) :
  2 * length + 2 * width = 100 := 
by
  sorry

end patio_perimeter_is_100_feet_l224_224540


namespace books_needed_to_buy_clarinet_l224_224989

def cost_of_clarinet : ℕ := 90
def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def halfway_loss : ℕ := (cost_of_clarinet - initial_savings) / 2

theorem books_needed_to_buy_clarinet 
    (cost_of_clarinet initial_savings price_per_book halfway_loss : ℕ)
    (initial_savings_lost : halfway_loss = (cost_of_clarinet - initial_savings) / 2) : 
    ((cost_of_clarinet - initial_savings + halfway_loss) / price_per_book) = 24 := 
sorry

end books_needed_to_buy_clarinet_l224_224989


namespace probability_different_colors_l224_224794

def total_chips : ℕ := 12

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def prob_diff_color (x y : ℕ) : ℚ :=
(x / total_chips) * (y / total_chips) + (y / total_chips) * (x / total_chips)

theorem probability_different_colors :
  prob_diff_color blue_chips yellow_chips +
  prob_diff_color blue_chips red_chips +
  prob_diff_color yellow_chips red_chips = 47 / 72 := by
sorry

end probability_different_colors_l224_224794


namespace exists_root_f_between_0_and_1_l224_224049

noncomputable def f (x : ℝ) : ℝ := 4 - 4 * x - Real.exp x

theorem exists_root_f_between_0_and_1 :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
sorry

end exists_root_f_between_0_and_1_l224_224049


namespace percentage_less_than_l224_224281

theorem percentage_less_than (x y z : Real) (h1 : x = 1.20 * y) (h2 : x = 0.84 * z) : 
  ((z - y) / z) * 100 = 30 := 
sorry

end percentage_less_than_l224_224281


namespace paper_folding_ratio_l224_224031

theorem paper_folding_ratio :
  ∃ (side length small_perim large_perim : ℕ), 
    side_length = 6 ∧ 
    small_perim = 2 * (3 + 3) ∧ 
    large_perim = 2 * (6 + 3) ∧ 
    small_perim / large_perim = 2 / 3 :=
by sorry

end paper_folding_ratio_l224_224031


namespace total_pizzas_served_l224_224566

-- Define the conditions
def pizzas_lunch : Nat := 9
def pizzas_dinner : Nat := 6

-- Define the theorem to prove
theorem total_pizzas_served : pizzas_lunch + pizzas_dinner = 15 := by
  sorry

end total_pizzas_served_l224_224566


namespace magnitude_z1_pure_imaginary_l224_224464

open Complex

theorem magnitude_z1_pure_imaginary 
  (a : ℝ)
  (z1 : ℂ := a + 2 * I)
  (z2 : ℂ := 3 - 4 * I)
  (h : (z1 / z2).re = 0) :
  Complex.abs z1 = 10 / 3 := 
sorry

end magnitude_z1_pure_imaginary_l224_224464


namespace find_linear_in_two_variables_l224_224901

def is_linear_in_two_variables (eq : String) : Bool :=
  eq = "x=y+1"

theorem find_linear_in_two_variables :
  (is_linear_in_two_variables "4xy=2" = false) ∧
  (is_linear_in_two_variables "1-x=7" = false) ∧
  (is_linear_in_two_variables "x^2+2y=-2" = false) ∧
  (is_linear_in_two_variables "x=y+1" = true) :=
by
  sorry

end find_linear_in_two_variables_l224_224901


namespace number_of_multiples_of_15_between_35_and_200_l224_224131

theorem number_of_multiples_of_15_between_35_and_200 : ∃ n : ℕ, n = 11 ∧ ∃ k : ℕ, k ≤ 200 ∧ k ≥ 35 ∧ (∃ m : ℕ, m < n ∧ 45 + m * 15 = k) :=
by
  sorry

end number_of_multiples_of_15_between_35_and_200_l224_224131


namespace largest_K_inequality_l224_224905

noncomputable def largest_K : ℝ := 18

theorem largest_K_inequality (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
(h_cond : a * b + b * c + c * a = a * b * c) :
( (a^a * (b^2 + c^2)) / ((a^a - 1)^2) + (b^b * (c^2 + a^2)) / ((b^b - 1)^2) + (c^c * (a^2 + b^2)) / ((c^c - 1)^2) )
≥ largest_K * ((a + b + c) / (a * b * c - 1)) ^ 2 :=
sorry

end largest_K_inequality_l224_224905


namespace relationship_of_variables_l224_224534

theorem relationship_of_variables
  (a b c d : ℚ)
  (h : (a + b) / (b + c) = (c + d) / (d + a)) :
  a = c ∨ a + b + c + d = 0 :=
by sorry

end relationship_of_variables_l224_224534


namespace parabola_above_line_l224_224981

variable {a b c : ℝ}

theorem parabola_above_line
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (H : (b - c) ^ 2 - 4 * a * c < 0) :
  (b + c) ^ 2 - 4 * c * (a + b) < 0 := 
sorry

end parabola_above_line_l224_224981


namespace third_recipe_soy_sauce_l224_224106

theorem third_recipe_soy_sauce :
  let bottle_ounces := 16
  let cup_ounces := 8
  let first_recipe_cups := 2
  let second_recipe_cups := 1
  let total_bottles := 3
  (total_bottles * bottle_ounces) / cup_ounces - (first_recipe_cups + second_recipe_cups) = 3 :=
by
  sorry

end third_recipe_soy_sauce_l224_224106


namespace largest_x_l224_224091

theorem largest_x (x : ℝ) : 
  (∃ x, (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → 
  (x ≤ 1) := sorry

end largest_x_l224_224091


namespace brady_work_hours_l224_224518

theorem brady_work_hours (A : ℕ) :
    (A * 30 + 5 * 30 + 8 * 30 = 3 * 190) → 
    A = 6 :=
by sorry

end brady_work_hours_l224_224518


namespace restaurant_A2_probability_l224_224831

noncomputable def prob_A2 (P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 : ℝ) : ℝ :=
  P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1

theorem restaurant_A2_probability :
  let P_A1 := 0.4
  let P_B1 := 0.6
  let P_A2_given_A1 := 0.6
  let P_A2_given_B1 := 0.5
  prob_A2 P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 = 0.54 :=
by
  sorry

end restaurant_A2_probability_l224_224831


namespace mass_percentage_H_in_NH4I_is_correct_l224_224612

noncomputable def molar_mass_NH4I : ℝ := 1 * 14.01 + 4 * 1.01 + 1 * 126.90

noncomputable def mass_H_in_NH4I : ℝ := 4 * 1.01

noncomputable def mass_percentage_H_in_NH4I : ℝ := (mass_H_in_NH4I / molar_mass_NH4I) * 100

theorem mass_percentage_H_in_NH4I_is_correct :
  abs (mass_percentage_H_in_NH4I - 2.79) < 0.01 := by
  sorry

end mass_percentage_H_in_NH4I_is_correct_l224_224612


namespace max_weight_of_flock_l224_224141

def MaxWeight (A E Af: ℕ): ℕ := A * 5 + E * 10 + Af * 15

theorem max_weight_of_flock :
  ∀ (A E Af: ℕ),
    A = 2 * E →
    Af = 3 * A →
    A + E + Af = 120 →
    MaxWeight A E Af = 1415 :=
by
  sorry

end max_weight_of_flock_l224_224141


namespace stock_percent_change_l224_224415

-- define initial value of stock
def initial_stock_value (x : ℝ) := x

-- define value after first day's decrease
def value_after_day_one (x : ℝ) := 0.85 * x

-- define value after second day's increase
def value_after_day_two (x : ℝ) := 1.25 * value_after_day_one x

-- Theorem stating the overall percent change is 6.25%
theorem stock_percent_change (x : ℝ) (h : x > 0) :
  ((value_after_day_two x - initial_stock_value x) / initial_stock_value x) * 100 = 6.25 := by sorry

end stock_percent_change_l224_224415


namespace find_pairs_gcd_lcm_l224_224533

theorem find_pairs_gcd_lcm : 
  { (a, b) : ℕ × ℕ | Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 } = {(24, 360), (72, 120)} := 
by
  sorry

end find_pairs_gcd_lcm_l224_224533


namespace coffee_maker_capacity_l224_224585

theorem coffee_maker_capacity (x : ℝ) (h : 0.36 * x = 45) : x = 125 :=
sorry

end coffee_maker_capacity_l224_224585


namespace triangle_inequality_l224_224690

theorem triangle_inequality
  (R r p : ℝ) (a b c : ℝ)
  (h1 : a * b + b * c + c * a = r^2 + p^2 + 4 * R * r)
  (h2 : 16 * R * r - 5 * r^2 ≤ p^2)
  (h3 : p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2):
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := 
  by
    sorry

end triangle_inequality_l224_224690


namespace percentage_heavier_l224_224496

variables (J M : ℝ)

theorem percentage_heavier (hM : M ≠ 0) : 
  100 * ((J + 3) - M) / M = 100 * ((J + 3) - M) / M := 
sorry

end percentage_heavier_l224_224496


namespace acme_vowel_soup_sequences_l224_224837

-- Define the vowels and their frequencies
def vowels : List (Char × ℕ) := [('A', 6), ('E', 6), ('I', 6), ('O', 4), ('U', 4)]

-- Noncomputable definition to calculate the total number of sequences
noncomputable def number_of_sequences : ℕ :=
  let single_vowel_choices := 6 + 6 + 6 + 4 + 4
  single_vowel_choices^5

-- Theorem stating the number of five-letter sequences
theorem acme_vowel_soup_sequences : number_of_sequences = 11881376 := by
  sorry

end acme_vowel_soup_sequences_l224_224837


namespace student_ratio_l224_224841

theorem student_ratio (total_students below_eight eight_years above_eight : ℕ) 
  (h1 : below_eight = total_students * 20 / 100) 
  (h2 : eight_years = 72) 
  (h3 : total_students = 150) 
  (h4 : total_students = below_eight + eight_years + above_eight) :
  (above_eight / eight_years) = 2 / 3 :=
by
  sorry

end student_ratio_l224_224841


namespace men_in_group_initial_l224_224026

variable (M : ℕ)  -- Initial number of men in the group
variable (A : ℕ)  -- Initial average age of the group

theorem men_in_group_initial : (2 * 50 - (18 + 22) = 60) → ((M + 6) = 60 / 6) → (M = 10) :=
by
  sorry

end men_in_group_initial_l224_224026


namespace cuboid_dimensions_l224_224174

-- Define the problem conditions and the goal
theorem cuboid_dimensions (x y v : ℕ) :
  (v * (x * y - 1) = 602) ∧ (x * (v * y - 1) = 605) →
  v = x + 3 →
  x = 11 ∧ y = 4 ∧ v = 14 :=
by
  sorry

end cuboid_dimensions_l224_224174


namespace tim_points_l224_224148

theorem tim_points (J T K : ℝ) (h1 : T = J + 20) (h2 : T = K / 2) (h3 : J + T + K = 100) : T = 30 := 
by 
  sorry

end tim_points_l224_224148


namespace gcd_90_252_eq_18_l224_224819

theorem gcd_90_252_eq_18 : Nat.gcd 90 252 = 18 := 
sorry

end gcd_90_252_eq_18_l224_224819


namespace proof_P_otimes_Q_l224_224424

-- Define the sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def Q : Set ℝ := { x | 1 < x }

-- Define the operation ⊗ between sets
def otimes (P Q : Set ℝ) : Set ℝ := { x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q }

-- Prove that P ⊗ Q = [0,1] ∪ (2, +∞)
theorem proof_P_otimes_Q :
  otimes P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (2 < x)} :=
by
 sorry

end proof_P_otimes_Q_l224_224424


namespace interest_rate_is_10_percent_l224_224583

theorem interest_rate_is_10_percent (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) 
  (hP : P = 9999.99999999988) 
  (ht : t = 1) 
  (hd : d = 25)
  : P * (1 + r / 2)^(2 * t) - P - (P * r * t) = d → r = 0.1 :=
by
  intros h
  rw [hP, ht, hd] at h
  sorry

end interest_rate_is_10_percent_l224_224583


namespace average_increase_l224_224207

def scores : List ℕ := [92, 85, 90, 95]

def initial_average (s : List ℕ) : ℚ := (s.take 3).sum / 3

def new_average (s : List ℕ) : ℚ := s.sum / s.length

theorem average_increase :
  initial_average scores + 1.5 = new_average scores := 
by
  sorry

end average_increase_l224_224207


namespace cone_lateral_surface_area_ratio_l224_224790

/-- Let a be the side length of the equilateral triangle front view of a cone.
    The base area of the cone is (π * (a / 2)^2).
    The lateral surface area of the cone is (π * (a / 2) * a).
    We want to show that the ratio of the lateral surface area to the base area is 2.
 -/
theorem cone_lateral_surface_area_ratio 
  (a : ℝ) 
  (base_area : ℝ := π * (a / 2)^2) 
  (lateral_surface_area : ℝ := π * (a / 2) * a) 
  : lateral_surface_area / base_area = 2 :=
by
  sorry

end cone_lateral_surface_area_ratio_l224_224790


namespace Z_divisible_by_10001_l224_224859

def is_eight_digit_integer (Z : Nat) : Prop :=
  (10^7 ≤ Z) ∧ (Z < 10^8)

def first_four_equal_last_four (Z : Nat) : Prop :=
  ∃ (a b c d : Nat), a ≠ 0 ∧ (Z = 1001 * (1000 * a + 100 * b + 10 * c + d))

theorem Z_divisible_by_10001 (Z : Nat) (h1 : is_eight_digit_integer Z) (h2 : first_four_equal_last_four Z) : 
  10001 ∣ Z :=
sorry

end Z_divisible_by_10001_l224_224859


namespace remainder_when_sum_divided_l224_224576

theorem remainder_when_sum_divided (p q : ℕ) (m n : ℕ) (hp : p = 80 * m + 75) (hq : q = 120 * n + 115) :
  (p + q) % 40 = 30 := 
by sorry

end remainder_when_sum_divided_l224_224576


namespace integer_solutions_count_correct_1992_integer_solutions_count_correct_1993_integer_solutions_count_correct_1994_l224_224377

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
  if n = 1992 then 90
  else if n = 1993 then 6
  else if n = 1994 then 6
  else 0

theorem integer_solutions_count_correct_1992 :
  count_integer_solutions 1992 = 90 :=
by
  sorry

theorem integer_solutions_count_correct_1993 :
  count_integer_solutions 1993 = 6 :=
by
  sorry

theorem integer_solutions_count_correct_1994 :
  count_integer_solutions 1994 = 6 :=
by
  sorry

example :
  count_integer_solutions 1992 = 90 ∧
  count_integer_solutions 1993 = 6 ∧
  count_integer_solutions 1994 = 6 :=
by
  exact ⟨integer_solutions_count_correct_1992, integer_solutions_count_correct_1993, integer_solutions_count_correct_1994⟩

end integer_solutions_count_correct_1992_integer_solutions_count_correct_1993_integer_solutions_count_correct_1994_l224_224377


namespace intersection_S_T_l224_224058

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l224_224058


namespace find_n_l224_224271

theorem find_n (n : ℕ) : (1/5)^35 * (1/4)^18 = 1/(n*(10)^35) → n = 2 :=
by
  sorry

end find_n_l224_224271


namespace sin_double_angle_l224_224797

-- Define the conditions and the goal
theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
sorry

end sin_double_angle_l224_224797


namespace janessa_gives_dexter_cards_l224_224852

def initial_cards : Nat := 4
def father_cards : Nat := 13
def ordered_cards : Nat := 36
def bad_cards : Nat := 4
def kept_cards : Nat := 20

theorem janessa_gives_dexter_cards :
  initial_cards + father_cards + ordered_cards - bad_cards - kept_cards = 29 := 
by
  sorry

end janessa_gives_dexter_cards_l224_224852


namespace tan_alpha_plus_beta_mul_tan_alpha_l224_224180

theorem tan_alpha_plus_beta_mul_tan_alpha (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := 
by
  sorry

end tan_alpha_plus_beta_mul_tan_alpha_l224_224180


namespace shaded_area_percentage_correct_l224_224373

-- Define a square and the conditions provided
def square (side_length : ℕ) : ℕ := side_length ^ 2

-- Define conditions
def EFGH_side_length : ℕ := 6
def total_area : ℕ := square EFGH_side_length

def shaded_area_1 : ℕ := square 2
def shaded_area_2 : ℕ := square 4 - square 3
def shaded_area_3 : ℕ := square 6 - square 5

def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

def shaded_percentage : ℚ := total_shaded_area / total_area * 100

-- Statement of the theorem to prove
theorem shaded_area_percentage_correct :
  shaded_percentage = 61.11 := by sorry

end shaded_area_percentage_correct_l224_224373


namespace hyperbola_representation_iff_l224_224356

theorem hyperbola_representation_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (2 + m) - (y^2) / (m + 1) = 1) ↔ (m > -1 ∨ m < -2) :=
by
  sorry

end hyperbola_representation_iff_l224_224356


namespace exists_xi_l224_224592

variable (f : ℝ → ℝ)
variable (hf_diff : ∀ x, DifferentiableAt ℝ f x)
variable (hf_twice_diff : ∀ x, DifferentiableAt ℝ (deriv f) x)
variable (hf₀ : f 0 = 2)
variable (hf_prime₀ : deriv f 0 = -2)
variable (hf₁ : f 1 = 1)

theorem exists_xi (h0 : f 0 = 2) (h1 : deriv f 0 = -2) (h2 : f 1 = 1) :
  ∃ ξ ∈ Set.Ioo 0 1, f ξ * deriv f ξ + deriv (deriv f) ξ = 0 :=
sorry

end exists_xi_l224_224592


namespace fifteen_percent_of_x_equals_sixty_l224_224645

theorem fifteen_percent_of_x_equals_sixty (x : ℝ) (h : 0.15 * x = 60) : x = 400 :=
by
  sorry

end fifteen_percent_of_x_equals_sixty_l224_224645


namespace arrangement_of_letters_l224_224515

-- Define the set of letters with subscripts
def letters : Finset String := {"B", "A₁", "B₁", "A₂", "B₂", "A₃"}

-- Define the number of ways to arrange 6 distinct letters
theorem arrangement_of_letters : letters.card.factorial = 720 := 
by {
  sorry
}

end arrangement_of_letters_l224_224515


namespace distance_between_skew_lines_l224_224095

-- Definitions for the geometric configuration
def AB : ℝ := 4
def AA1 : ℝ := 4
def AD : ℝ := 3

-- Theorem statement to prove the distance between skew lines A1D and B1D1
theorem distance_between_skew_lines:
  ∃ d : ℝ, d = (6 * Real.sqrt 34) / 17 :=
sorry

end distance_between_skew_lines_l224_224095


namespace other_x_intercept_l224_224256

-- Definition of the two foci
def f1 : ℝ × ℝ := (0, 2)
def f2 : ℝ × ℝ := (3, 0)

-- One x-intercept is given as
def intercept1 : ℝ × ℝ := (0, 0)

-- We need to prove the other x-intercept is (15/4, 0)
theorem other_x_intercept : ∃ x : ℝ, (x, 0) = (15/4, 0) ∧
  (dist (x, 0) f1 + dist (x, 0) f2 = dist intercept1 f1 + dist intercept1 f2) :=
by
  sorry

end other_x_intercept_l224_224256


namespace sets_equal_sufficient_condition_l224_224816

variable (a : ℝ)

-- Define sets A and B
def A (x : ℝ) : Prop := 0 < a * x + 1 ∧ a * x + 1 ≤ 5
def B (x : ℝ) : Prop := -1/2 < x ∧ x ≤ 2

-- Statement for Part 1: Sets A and B can be equal if and only if a = 2
theorem sets_equal (h : ∀ x, A a x ↔ B x) : a = 2 :=
sorry

-- Statement for Part 2: Proposition p ⇒ q holds if and only if a > 2 or a < -8
theorem sufficient_condition (h : ∀ x, A a x → B x) (h_neq : ∃ x, B x ∧ ¬A a x) : a > 2 ∨ a < -8 :=
sorry

end sets_equal_sufficient_condition_l224_224816


namespace isosceles_trapezoid_AB_length_l224_224512

theorem isosceles_trapezoid_AB_length (BC AD : ℝ) (r : ℝ) (a : ℝ) (h_isosceles : BC = a) (h_ratio : AD = 3 * a) (h_area : 4 * a * r = Real.sqrt 3 / 2) (h_radius : r = a * Real.sqrt 3 / 2) :
  2 * a = 1 :=
by
 sorry

end isosceles_trapezoid_AB_length_l224_224512


namespace base_conversion_positive_b_l224_224704

theorem base_conversion_positive_b :
  (∃ (b : ℝ), 3 * 5^1 + 2 * 5^0 = 17 ∧ 1 * b^2 + 2 * b^1 + 0 * b^0 = 17 ∧ b = -1 + 3 * Real.sqrt 2) :=
by
  sorry

end base_conversion_positive_b_l224_224704


namespace aaron_earnings_l224_224828

def time_worked_monday := 75 -- in minutes
def time_worked_tuesday := 50 -- in minutes
def time_worked_wednesday := 145 -- in minutes
def time_worked_friday := 30 -- in minutes
def hourly_rate := 3 -- dollars per hour

def total_minutes_worked := 
  time_worked_monday + time_worked_tuesday + 
  time_worked_wednesday + time_worked_friday

def total_hours_worked := total_minutes_worked / 60

def total_earnings := total_hours_worked * hourly_rate

theorem aaron_earnings :
  total_earnings = 15 := by
  sorry

end aaron_earnings_l224_224828


namespace largest_root_is_1011_l224_224590

theorem largest_root_is_1011 (a b c d x : ℝ) 
  (h1 : a + d = 2022) 
  (h2 : b + c = 2022) 
  (h3 : a ≠ c) 
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) : 
  x = 1011 := 
sorry

end largest_root_is_1011_l224_224590


namespace tangent_line_equation_l224_224506

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P.2 = P.1^2)
  (h_perpendicular : ∃ k : ℝ, k * -1/2 = -1) : 
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end tangent_line_equation_l224_224506


namespace admission_counts_l224_224474

-- Define the total number of ways to admit students under given conditions.
def ways_of_admission : Nat := 1518

-- Statement of the problem: given conditions, prove the result
theorem admission_counts (n_colleges : Nat) (n_students : Nat) (admitted_two_colleges : Bool) : 
  n_colleges = 23 → 
  n_students = 3 → 
  admitted_two_colleges = true →
  ways_of_admission = 1518 :=
by
  intros
  sorry

end admission_counts_l224_224474


namespace angle_opposite_c_exceeds_l224_224404

theorem angle_opposite_c_exceeds (a b : ℝ) (c : ℝ) (C : ℝ) (h_a : a = 2) (h_b : b = 2) (h_c : c >= 4) : 
  C >= 120 := 
sorry

end angle_opposite_c_exceeds_l224_224404


namespace sides_of_second_polygon_l224_224749

theorem sides_of_second_polygon (s : ℝ) (n : ℕ) 
  (perimeter1_is_perimeter2 : 38 * (2 * s) = n * s) : 
  n = 76 := by
  sorry

end sides_of_second_polygon_l224_224749


namespace average_calculation_l224_224007

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 4 1) (average_two 3 2) 5 = 59 / 18 :=
by
  sorry

end average_calculation_l224_224007


namespace no_function_f_satisfies_condition_l224_224355

theorem no_function_f_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + y^2 :=
by
  sorry

end no_function_f_satisfies_condition_l224_224355


namespace box_volume_l224_224786

variable (l w h : ℝ)
variable (lw_eq : l * w = 30)
variable (wh_eq : w * h = 40)
variable (lh_eq : l * h = 12)

theorem box_volume : l * w * h = 120 := by
  sorry

end box_volume_l224_224786


namespace unique_solution_j_l224_224836

theorem unique_solution_j (j : ℝ) : (∀ x : ℝ, (2 * x + 7) * (x - 5) = -43 + j * x) → (j = 5 ∨ j = -11) :=
by
  sorry

end unique_solution_j_l224_224836


namespace train_speed_l224_224657

theorem train_speed (length time : ℝ) (h_length : length = 120) (h_time : time = 11.999040076793857) :
  (length / time) * 3.6 = 36.003 :=
by
  sorry

end train_speed_l224_224657


namespace power_computation_l224_224385

theorem power_computation : (12 ^ (12 / 2)) = 2985984 := by
  sorry

end power_computation_l224_224385


namespace rectangle_perimeter_l224_224756

theorem rectangle_perimeter (a b c width : ℕ) (area : ℕ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) 
  (h5 : area = (a * b) / 2) 
  (h6 : width = 5) 
  (h7 : area = width * ((area * 2) / (a * b)))
  : 2 * (width + (area / width)) = 22 := 
by 
  sorry

end rectangle_perimeter_l224_224756


namespace average_employees_per_week_l224_224817

theorem average_employees_per_week (x : ℝ)
  (h1 : ∀ (x : ℝ), ∃ y : ℝ, y = x + 200)
  (h2 : ∀ (x : ℝ), ∃ z : ℝ, z = x + 150)
  (h3 : ∀ (x : ℝ), ∃ w : ℝ, w = 2 * (x + 150))
  (h4 : ∀ (w : ℝ), w = 400) :
  (250 + 50 + 200 + 400) / 4 = 225 :=
by 
  sorry

end average_employees_per_week_l224_224817


namespace xiaoming_mirrored_time_l224_224339

-- Define the condition: actual time is 7:10 AM.
def actual_time : (ℕ × ℕ) := (7, 10)

-- Define a function to compute the mirrored time given an actual time.
def mirror_time (h m : ℕ) : (ℕ × ℕ) :=
  let mirrored_minute := if m = 0 then 0 else 60 - m
  let mirrored_hour := if m = 0 then if h = 12 then 12 else (12 - h) % 12
                        else if h = 12 then 11 else (11 - h) % 12
  (mirrored_hour, mirrored_minute)

-- Our goal is to verify that the mirrored time of 7:10 is 4:50.
theorem xiaoming_mirrored_time : mirror_time 7 10 = (4, 50) :=
by
  -- Proof will verify that mirror_time (7, 10) evaluates to (4, 50).
  sorry

end xiaoming_mirrored_time_l224_224339


namespace strawberries_left_l224_224783

theorem strawberries_left (initial : ℝ) (eaten : ℝ) (remaining : ℝ) : initial = 78.0 → eaten = 42.0 → remaining = 36.0 → initial - eaten = remaining :=
by
  sorry

end strawberries_left_l224_224783


namespace tenth_term_geometric_sequence_l224_224928

theorem tenth_term_geometric_sequence :
  let a := (8 : ℚ)
  let r := (-2 / 3 : ℚ)
  a * r^9 = -4096 / 19683 :=
by
  sorry

end tenth_term_geometric_sequence_l224_224928


namespace marbles_leftover_l224_224199

theorem marbles_leftover (r p j : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) (hj : j % 8 = 2) : (r + p + j) % 8 = 6 := 
sorry

end marbles_leftover_l224_224199


namespace certain_event_C_union_D_l224_224282

variable {Ω : Type} -- Omega, the sample space
variable {P : Set Ω → Prop} -- P as the probability function predicates the events

-- Definitions of the events
variable {A B C D : Set Ω}

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ x, x ∈ A → x ∉ B
def complementary (A C : Set Ω) : Prop := ∀ x, x ∈ C ↔ x ∉ A

-- Given conditions
axiom A_and_B_mutually_exclusive : mutually_exclusive A B
axiom C_is_complementary_to_A : complementary A C
axiom D_is_complementary_to_B : complementary B D

-- Theorem statement
theorem certain_event_C_union_D : ∀ x, x ∈ C ∪ D := by
  sorry

end certain_event_C_union_D_l224_224282


namespace hou_yi_score_l224_224689

theorem hou_yi_score (a b c : ℕ) (h1 : 2 * b + c = 29) (h2 : 2 * a + c = 43) : a + b + c = 36 := 
by 
  sorry

end hou_yi_score_l224_224689


namespace division_addition_l224_224469

theorem division_addition (n : ℕ) (h : 32 - 16 = n * 4) : n / 4 + 16 = 17 :=
by 
  sorry

end division_addition_l224_224469


namespace complement_of_union_l224_224593

open Set

namespace Proof

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of the union of sets A and B with respect to U
theorem complement_of_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 3}) (hB : B = {3, 5}) : 
  U \ (A ∪ B) = {0, 2, 4} :=
by {
  sorry
}

end Proof

end complement_of_union_l224_224593


namespace smallest_possible_N_l224_224153

theorem smallest_possible_N (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) 
  (hr : r > 0) (hs : s > 0) (ht : t > 0) (h_sum : p + q + r + s + t = 4020) :
  ∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1005 :=
sorry

end smallest_possible_N_l224_224153


namespace min_value_expression_min_value_is_7_l224_224942

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  6 * x + 1 / (x^6) ≥ 7 :=
sorry

theorem min_value_is_7 : 
  6 * 1 + 1 / (1^6) = 7 :=
by norm_num

end min_value_expression_min_value_is_7_l224_224942


namespace speed_of_second_person_l224_224246

-- Definitions based on the conditions
def speed_person1 := 70 -- km/hr
def distance_AB := 600 -- km

def time_traveled := 4 -- hours (from 10 am to 2 pm)

-- The goal is to prove that the speed of the second person is 80 km/hr
theorem speed_of_second_person :
  (distance_AB - speed_person1 * time_traveled) / time_traveled = 80 := 
by 
  sorry

end speed_of_second_person_l224_224246


namespace find_all_good_sets_l224_224573

def is_good_set (A : Finset ℕ) : Prop :=
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) ∧
  (∀ (b c : ℕ), b ∈ A → c ∈ A → b ≠ c → ∃ (a : ℕ), a ∈ A ∧ a ≠ b ∧ a ≠ c ∧ (b * c) % a = 0)

theorem find_all_good_sets : ∀ (A : Finset ℕ), is_good_set A ↔ 
  (A = {a, b, a * b} ∧ Nat.gcd a b = 1) ∨ 
  ∃ (p q r : ℕ), Nat.gcd p q = 1 ∧ Nat.gcd q r = 1 ∧ Nat.gcd r p = 1 ∧ A = {p * q, q * r, r * p} :=
by
  sorry

end find_all_good_sets_l224_224573


namespace tangent_line_y_intercept_at_P_1_12_is_9_l224_224857

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

noncomputable def tangent_slope_at (x : ℝ) : ℝ := 3 * x^2

noncomputable def tangent_line_y_intercept : ℝ :=
  let P : ℝ × ℝ := (1, curve 1)
  let slope := tangent_slope_at 1
  P.snd - slope * P.fst

theorem tangent_line_y_intercept_at_P_1_12_is_9 :
  tangent_line_y_intercept = 9 :=
sorry

end tangent_line_y_intercept_at_P_1_12_is_9_l224_224857


namespace determine_OP_l224_224581

theorem determine_OP 
  (a b c d k : ℝ)
  (h1 : k * b ≤ c) 
  (h2 : (A : ℝ) = a)
  (h3 : (B : ℝ) = k * b)
  (h4 : (C : ℝ) = c)
  (h5 : (D : ℝ) = k * d)
  (AP_PD : ∀ (P : ℝ), (a - P) / (P - k * d) = k * (k * b - P) / (P - c))
  :
  ∃ P : ℝ, P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
sorry

end determine_OP_l224_224581


namespace monthly_incomes_l224_224946

theorem monthly_incomes (a b c d e : ℕ) : 
  a + b = 8100 ∧ 
  b + c = 10500 ∧ 
  a + c = 8400 ∧
  (a + b + d) / 3 = 4800 ∧
  (c + d + e) / 3 = 6000 ∧
  (b + a + e) / 3 = 4500 → 
  (a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400) :=
by sorry

end monthly_incomes_l224_224946


namespace domain_g_eq_l224_224939

noncomputable def domain_f : Set ℝ := {x | -8 ≤ x ∧ x ≤ 4}

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

theorem domain_g_eq (f : ℝ → ℝ) (h : ∀ x, x ∈ domain_f → f x ∈ domain_f) :
  {x | x ∈ [-2, 4]} = {x | -2 ≤ x ∧ x ≤ 4} :=
by {
  sorry
}

end domain_g_eq_l224_224939


namespace general_term_correct_S_maximum_value_l224_224470

noncomputable def general_term (n : ℕ) : ℤ :=
  if n = 1 then -1 + 24 else (-n^2 + 24 * n) - (-(n - 1)^2 + 24 * (n - 1))

noncomputable def S (n : ℕ) : ℤ :=
  -n^2 + 24 * n

theorem general_term_correct (n : ℕ) (h : 1 ≤ n) : general_term n = -2 * n + 25 := by
  sorry

theorem S_maximum_value : ∃ n : ℕ, S n = 144 ∧ ∀ m : ℕ, S m ≤ 144 := by
  existsi 12
  sorry

end general_term_correct_S_maximum_value_l224_224470


namespace extreme_point_inequality_l224_224201

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x - a / x - 2 * Real.log x

theorem extreme_point_inequality (x₁ x₂ a : ℝ) (h1 : x₁ < x₂) (h2 : f x₁ a = 0) (h3 : f x₂ a = 0) 
(h_a_range : 0 < a) (h_a_lt_1 : a < 1) :
  f x₂ a < x₂ - 1 :=
sorry

end extreme_point_inequality_l224_224201


namespace misread_weight_l224_224873

theorem misread_weight (n : ℕ) (average_incorrect : ℚ) (average_correct : ℚ) (corrected_weight : ℚ) (incorrect_total correct_total diff : ℚ)
  (h1 : n = 20)
  (h2 : average_incorrect = 58.4)
  (h3 : average_correct = 59)
  (h4 : corrected_weight = 68)
  (h5 : incorrect_total = n * average_incorrect)
  (h6 : correct_total = n * average_correct)
  (h7 : diff = correct_total - incorrect_total)
  (h8 : diff = corrected_weight - x) : x = 56 := 
sorry

end misread_weight_l224_224873


namespace patio_length_l224_224740

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := 
by 
  sorry

end patio_length_l224_224740


namespace largest_n_is_253_l224_224594

-- Define the triangle property for a set
def triangle_property (s : Set ℕ) : Prop :=
∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → c < a + b

-- Define the problem statement
def largest_possible_n (n : ℕ) : Prop :=
∀ (s : Finset ℕ), (∀ (x : ℕ), x ∈ s → 4 ≤ x ∧ x ≤ n) → (s.card = 10 → triangle_property s)

-- The given proof problem
theorem largest_n_is_253 : largest_possible_n 253 :=
by
  sorry

end largest_n_is_253_l224_224594


namespace snowfall_total_l224_224909

theorem snowfall_total (snowfall_wed snowfall_thu snowfall_fri : ℝ)
  (h_wed : snowfall_wed = 0.33)
  (h_thu : snowfall_thu = 0.33)
  (h_fri : snowfall_fri = 0.22) :
  snowfall_wed + snowfall_thu + snowfall_fri = 0.88 :=
by
  rw [h_wed, h_thu, h_fri]
  norm_num

end snowfall_total_l224_224909


namespace inequality_proof_l224_224947

theorem inequality_proof (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2) (h2 : a2 ≥ a3) (h3 : a3 > 0) 
  (h4 : b1 ≥ b2) (h5 : b2 ≥ b3) (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) : 
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := sorry

end inequality_proof_l224_224947


namespace range_m_distinct_roots_l224_224073

theorem range_m_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (4^x₁ - m * 2^(x₁+1) + 2 - m = 0) ∧ (4^x₂ - m * 2^(x₂+1) + 2 - m = 0)) ↔ 1 < m ∧ m < 2 :=
by
  sorry

end range_m_distinct_roots_l224_224073


namespace change_after_buying_tickets_l224_224779

def cost_per_ticket := 8
def number_of_tickets := 2
def total_money := 25

theorem change_after_buying_tickets :
  total_money - number_of_tickets * cost_per_ticket = 9 := by
  sorry

end change_after_buying_tickets_l224_224779


namespace area_of_triangle_AMN_is_correct_l224_224868

noncomputable def area_triangle_AMN : ℝ :=
  let A := (120 + 56 * Real.sqrt 3) / 3
  let M := (12 + 20 * Real.sqrt 3) / 3
  let N := 4 * Real.sqrt 3 + 20
  (A * N) / 2

theorem area_of_triangle_AMN_is_correct :
  area_triangle_AMN = (224 * Real.sqrt 3 + 240) / 3 := sorry

end area_of_triangle_AMN_is_correct_l224_224868


namespace product_of_three_numbers_l224_224358

theorem product_of_three_numbers (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x = 4 * (y + z)) 
  (h3 : y = 7 * z) :
  x * y * z = 28 := 
by 
  sorry

end product_of_three_numbers_l224_224358


namespace find_b_l224_224647

-- Define the number 1234567 in base 36
def numBase36 : ℤ := 1 * 36^6 + 2 * 36^5 + 3 * 36^4 + 4 * 36^3 + 5 * 36^2 + 6 * 36^1 + 7 * 36^0

-- Prove that for b being an integer such that 0 ≤ b ≤ 10,
-- and given (numBase36 - b) is a multiple of 17, b must be 0
theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 10) (h3 : (numBase36 - b) % 17 = 0) : b = 0 :=
by
  sorry

end find_b_l224_224647


namespace unknown_number_is_10_l224_224587

def operation_e (x y : ℕ) : ℕ := 2 * x * y

theorem unknown_number_is_10 (n : ℕ) (h : operation_e 8 (operation_e n 5) = 640) : n = 10 :=
by
  sorry

end unknown_number_is_10_l224_224587


namespace mitya_age_l224_224822

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end mitya_age_l224_224822


namespace frood_least_throw_points_more_than_eat_l224_224930

theorem frood_least_throw_points_more_than_eat (n : ℕ) : n^2 > 12 * n ↔ n ≥ 13 :=
sorry

end frood_least_throw_points_more_than_eat_l224_224930


namespace solve_for_m_l224_224036

theorem solve_for_m (n : ℝ) (m : ℝ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 1 / 2 := 
sorry

end solve_for_m_l224_224036


namespace infinite_primes_divide_f_l224_224934

def non_constant_function (f : ℕ → ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ f a ≠ f b

def divisibility_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a ≠ b → (a - b) ∣ (f a - f b)

theorem infinite_primes_divide_f (f : ℕ → ℕ) 
  (h_non_const : non_constant_function f)
  (h_div : divisibility_condition f) :
  ∃ᶠ p in Filter.atTop, ∃ c : ℕ, p ∣ f c := sorry

end infinite_primes_divide_f_l224_224934


namespace area_enclosed_curves_l224_224176

theorem area_enclosed_curves (a : ℝ) (h1 : (1 + 1/a)^5 = 1024) :
  ∫ x in (0 : ℝ)..1, (x^(1/3) - x^2) = 5/12 :=
sorry

end area_enclosed_curves_l224_224176


namespace platform_length_l224_224074

theorem platform_length
  (train_length : ℕ)
  (time_pole : ℕ)
  (time_platform : ℕ)
  (h_train_length : train_length = 300)
  (h_time_pole : time_pole = 18)
  (h_time_platform : time_platform = 39) :
  ∃ (platform_length : ℕ), platform_length = 350 :=
by
  sorry

end platform_length_l224_224074


namespace max_lessons_l224_224982

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end max_lessons_l224_224982


namespace factor_x8_minus_81_l224_224676

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := 
by 
  sorry

end factor_x8_minus_81_l224_224676


namespace square_difference_l224_224983

theorem square_difference (x : ℤ) (h : x^2 = 1444) : (x + 1) * (x - 1) = 1443 := 
by 
  sorry

end square_difference_l224_224983


namespace alice_arrives_earlier_l224_224309

/-
Alice and Bob are heading to a park that is 2 miles away from their home. 
They leave home at the same time. 
Alice cycles to the park at a speed of 12 miles per hour, 
while Bob jogs there at a speed of 6 miles per hour. 
Prove that Alice arrives 10 minutes earlier at the park than Bob.
-/

theorem alice_arrives_earlier 
  (d : ℕ) (a_speed : ℕ) (b_speed : ℕ) (arrival_difference_minutes : ℕ) 
  (h1 : d = 2) 
  (h2 : a_speed = 12) 
  (h3 : b_speed = 6) 
  (h4 : arrival_difference_minutes = 10) 
  : (d / a_speed * 60) + arrival_difference_minutes = d / b_speed * 60 :=
by
  sorry

end alice_arrives_earlier_l224_224309


namespace probability_bus_there_when_mark_arrives_l224_224211

noncomputable def isProbabilityBusThereWhenMarkArrives : Prop :=
  let busArrival : ℝ := 60 -- The bus can arrive from time 0 to 60 minutes (2:00 PM to 3:00 PM)
  let busWait : ℝ := 30 -- The bus waits for 30 minutes
  let markArrival : ℝ := 90 -- Mark can arrive from time 30 to 90 minutes (2:30 PM to 3:30 PM)
  let overlapArea : ℝ := 1350 -- Total shaded area where bus arrival overlaps with Mark's arrival
  let totalArea : ℝ := busArrival * (markArrival - 30)
  let probability := overlapArea / totalArea
  probability = 1 / 4

theorem probability_bus_there_when_mark_arrives : isProbabilityBusThereWhenMarkArrives :=
by
  sorry

end probability_bus_there_when_mark_arrives_l224_224211


namespace Marcy_120_votes_l224_224118

-- Definitions based on conditions
def votes (name : String) : ℕ := sorry -- placeholder definition

-- Conditions
def Joey_votes := votes "Joey" = 8
def Jill_votes := votes "Jill" = votes "Joey" + 4
def Barry_votes := votes "Barry" = 2 * (votes "Joey" + votes "Jill")
def Marcy_votes := votes "Marcy" = 3 * votes "Barry"
def Tim_votes := votes "Tim" = votes "Marcy" / 2
def Sam_votes := votes "Sam" = votes "Tim" + 10

-- Theorem to prove
theorem Marcy_120_votes : Joey_votes → Jill_votes → Barry_votes → Marcy_votes → Tim_votes → Sam_votes → votes "Marcy" = 120 := by
  intros
  -- Skipping the proof
  sorry

end Marcy_120_votes_l224_224118


namespace smallest_of_three_consecutive_odd_numbers_l224_224126

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) (h : x + (x + 2) + (x + 4) = 69) : x = 21 :=
sorry

end smallest_of_three_consecutive_odd_numbers_l224_224126


namespace prime_dates_in_2008_l224_224864

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_prime_date (month day : ℕ) : Prop := is_prime month ∧ is_prime day

noncomputable def prime_dates_2008 : ℕ :=
  let prime_months := [2, 3, 5, 7, 11]
  let prime_days_31 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let prime_days_30 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_days_29 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  prime_months.foldl (λ acc month => 
    acc + match month with
      | 2 => List.length prime_days_29
      | 3 | 5 | 7 => List.length prime_days_31
      | 11 => List.length prime_days_30
      | _ => 0
    ) 0

theorem prime_dates_in_2008 : 
  prime_dates_2008 = 53 :=
  sorry

end prime_dates_in_2008_l224_224864


namespace width_of_plot_is_60_l224_224121

-- Defining the conditions
def length_of_plot := 90
def distance_between_poles := 5
def number_of_poles := 60

-- The theorem statement
theorem width_of_plot_is_60 :
  ∃ width : ℕ, 2 * (length_of_plot + width) = number_of_poles * distance_between_poles ∧ width = 60 :=
sorry

end width_of_plot_is_60_l224_224121


namespace max_xy_l224_224426

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l224_224426


namespace area_shaded_region_is_correct_l224_224755

noncomputable def radius_of_larger_circle : ℝ := 8
noncomputable def radius_of_smaller_circle := radius_of_larger_circle / 2

-- Define areas
noncomputable def area_of_larger_circle := Real.pi * radius_of_larger_circle ^ 2
noncomputable def area_of_smaller_circle := Real.pi * radius_of_smaller_circle ^ 2
noncomputable def total_area_of_smaller_circles := 2 * area_of_smaller_circle
noncomputable def area_of_shaded_region := area_of_larger_circle - total_area_of_smaller_circles

-- Prove that the area of the shaded region is 32π
theorem area_shaded_region_is_correct : area_of_shaded_region = 32 * Real.pi := by
  sorry

end area_shaded_region_is_correct_l224_224755


namespace arithmetic_sequence_term_13_l224_224648

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_term_13 (h_arith : arithmetic_sequence a d)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := 
by 
  sorry

end arithmetic_sequence_term_13_l224_224648


namespace bottles_needed_exceed_initial_l224_224682

-- Define the initial conditions and their relationships
def initial_bottles : ℕ := 4 * 12 -- four dozen bottles

def bottles_first_break (players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  players * bottles_per_player

def bottles_second_break (total_players : ℕ) (bottles_per_player : ℕ) (exhausted_players : ℕ) (extra_bottles : ℕ) : ℕ :=
  total_players * bottles_per_player + exhausted_players * extra_bottles

def bottles_third_break (remaining_players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  remaining_players * bottles_per_player

-- Prove that the bottles needed exceed the initial amount by 4
theorem bottles_needed_exceed_initial : 
  bottles_first_break 11 2 + bottles_second_break 14 1 4 1 + bottles_third_break 12 1 = initial_bottles + 4 :=
by
  -- Proof will be completed here
  sorry

end bottles_needed_exceed_initial_l224_224682


namespace miquels_theorem_l224_224483

-- Define a triangle ABC with points D, E, F on sides BC, CA, and AB respectively
variables {A B C D E F : Type}

-- Assume we have a function that checks for collinearity of points
def is_on_side (X Y Z: Type) : Bool := sorry

-- Assume a function that returns the circumcircle of a triangle formed by given points
def circumcircle (X Y Z: Type) : Type := sorry 

-- Define the function that checks the intersection of circumcircles
def have_common_point (circ1 circ2 circ3: Type) : Bool := sorry

-- The theorem statement
theorem miquels_theorem (A B C D E F : Type) 
  (hD: is_on_side D B C) 
  (hE: is_on_side E C A) 
  (hF: is_on_side F A B) : 
  have_common_point (circumcircle A E F) (circumcircle B D F) (circumcircle C D E) :=
sorry

end miquels_theorem_l224_224483


namespace sqrt_3_between_inequalities_l224_224342

theorem sqrt_3_between_inequalities (n : ℕ) (h1 : 1 + (3 : ℝ) / (n + 1) < Real.sqrt 3) (h2 : Real.sqrt 3 < 1 + (3 : ℝ) / n) : n = 4 := 
sorry

end sqrt_3_between_inequalities_l224_224342


namespace sum_of_coefficients_l224_224053

theorem sum_of_coefficients :
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ,
  (∀ x : ℤ, (2 - x)^7 = a₀ + a₁ * (1 + x)^2 + a₂ * (1 + x)^3 + a₃ * (1 + x)^4 + a₄ * (1 + x)^5 + a₅ * (1 + x)^6 + a₆ * (1 + x)^7 + a₇ * (1 + x)^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 129 := by sorry

end sum_of_coefficients_l224_224053


namespace min_m_min_expression_l224_224838

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part (Ⅰ)
theorem min_m (m : ℝ) (h : ∃ x₀ : ℝ, f x₀ ≤ m) : m ≥ 2 := sorry

-- Part (Ⅱ)
theorem min_expression (a b : ℝ) (h1 : 3 * a + b = 2) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / (2 * a) + 1 / (a + b)) ≥ 2 := sorry

end min_m_min_expression_l224_224838


namespace satisfies_differential_eqn_l224_224429

noncomputable def y (x : ℝ) : ℝ := 5 * Real.exp (-2 * x) + (1 / 3) * Real.exp x

theorem satisfies_differential_eqn : ∀ x : ℝ, (deriv y x) + 2 * y x = Real.exp x :=
by
  -- The proof is to be provided
  sorry

end satisfies_differential_eqn_l224_224429


namespace symmetric_points_sum_l224_224532

variable {p q : ℤ}

theorem symmetric_points_sum (h1 : p = -6) (h2 : q = 2) : p + q = -4 := by
  sorry

end symmetric_points_sum_l224_224532


namespace polygon_sides_l224_224149

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end polygon_sides_l224_224149


namespace school_boys_count_l224_224865

theorem school_boys_count (B G : ℕ) (h1 : B + G = 1150) (h2 : G = (B / 1150) * 100) : B = 1058 := 
by 
  sorry

end school_boys_count_l224_224865


namespace equation_solution_l224_224861

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l224_224861


namespace set_problems_l224_224025

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_problems :
  (A ∩ B = ({4} : Set ℤ)) ∧
  (A ∪ B = ({1, 2, 4, 5, 6, 7, 8, 9, 10} : Set ℤ)) ∧
  (U \ (A ∪ B) = ({3} : Set ℤ)) ∧
  ((U \ A) ∩ (U \ B) = ({3} : Set ℤ)) :=
by
  sorry

end set_problems_l224_224025


namespace go_stones_problem_l224_224977

theorem go_stones_problem
  (x : ℕ) 
  (h1 : x / 7 + 40 = 555 / 5) 
  (black_stones : ℕ) 
  (h2 : black_stones = 55) :
  (x - black_stones = 442) :=
sorry

end go_stones_problem_l224_224977


namespace find_legs_of_triangle_l224_224147

theorem find_legs_of_triangle (a b : ℝ) (h : a / b = 3 / 4) (h_sum : a^2 + b^2 = 70^2) : 
  (a = 42) ∧ (b = 56) :=
sorry

end find_legs_of_triangle_l224_224147


namespace eggs_leftover_l224_224055

theorem eggs_leftover (d e f : ℕ) (total_eggs_per_carton : ℕ) 
  (h_d : d = 53) (h_e : e = 65) (h_f : f = 26) (h_carton : total_eggs_per_carton = 15) : (d + e + f) % total_eggs_per_carton = 9 :=
by {
  sorry
}

end eggs_leftover_l224_224055


namespace johns_average_speed_l224_224903

def continuous_driving_duration (start_time end_time : ℝ) (distance : ℝ) : Prop :=
start_time = 10.5 ∧ end_time = 14.75 ∧ distance = 190

theorem johns_average_speed
  (start_time end_time : ℝ) 
  (distance : ℝ)
  (h : continuous_driving_duration start_time end_time distance) :
  (distance / (end_time - start_time) = 44.7) :=
by
  sorry

end johns_average_speed_l224_224903


namespace second_storm_duration_l224_224454

theorem second_storm_duration
  (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : 30 * x + 15 * y = 975) :
  y = 25 := 
sorry

end second_storm_duration_l224_224454


namespace ratio_of_triangle_areas_bcx_acx_l224_224884

theorem ratio_of_triangle_areas_bcx_acx
  (BC AC : ℕ) (hBC : BC = 36) (hAC : AC = 45)
  (is_angle_bisector_CX : ∀ BX AX : ℕ, BX / AX = BC / AC) :
  (∃ BX AX : ℕ, BX / AX = 4 / 5) :=
by
  have h_ratio := is_angle_bisector_CX 36 45
  rw [hBC, hAC] at h_ratio
  exact ⟨4, 5, h_ratio⟩

end ratio_of_triangle_areas_bcx_acx_l224_224884


namespace reyn_pieces_l224_224549

-- Define the conditions
variables (total_pieces : ℕ) (pieces_each : ℕ) (pieces_left : ℕ)
variables (R : ℕ) (Rhys : ℕ) (Rory : ℕ)

-- Initial Conditions
def mrs_young_conditions :=
  total_pieces = 300 ∧
  pieces_each = total_pieces / 3 ∧
  Rhys = 2 * R ∧
  Rory = 3 * R ∧
  6 * R + pieces_left = total_pieces ∧
  pieces_left = 150

-- The statement of our proof goal
theorem reyn_pieces (h : mrs_young_conditions total_pieces pieces_each pieces_left R Rhys Rory) : R = 25 :=
sorry

end reyn_pieces_l224_224549


namespace average_throws_to_lasso_l224_224275

theorem average_throws_to_lasso (p : ℝ) (h₁ : 1 - (1 - p)^3 = 0.875) : (1 / p) = 2 :=
by
  sorry

end average_throws_to_lasso_l224_224275


namespace sin_2pi_minus_alpha_l224_224862

theorem sin_2pi_minus_alpha (α : ℝ) (h₁ : Real.cos (α + Real.pi) = Real.sqrt 3 / 2) (h₂ : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
    Real.sin (2 * Real.pi - α) = -1 / 2 := 
sorry

end sin_2pi_minus_alpha_l224_224862


namespace ratio_docking_to_license_l224_224380

noncomputable def Mitch_savings : ℕ := 20000
noncomputable def boat_cost_per_foot : ℕ := 1500
noncomputable def license_and_registration_fees : ℕ := 500
noncomputable def max_boat_length : ℕ := 12

theorem ratio_docking_to_license :
  let remaining_amount := Mitch_savings - license_and_registration_fees
  let cost_of_longest_boat := boat_cost_per_foot * max_boat_length
  let docking_fees := remaining_amount - cost_of_longest_boat
  docking_fees / license_and_registration_fees = 3 :=
by
  sorry

end ratio_docking_to_license_l224_224380


namespace largest_multiple_of_9_less_than_100_l224_224727

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l224_224727


namespace find_center_of_circle_l224_224308

noncomputable def center_of_circle (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 + 4 * y = 16

theorem find_center_of_circle (x y : ℝ) (h : center_of_circle x y) : (x, y) = (4, -2) :=
by 
  sorry

end find_center_of_circle_l224_224308


namespace necessary_but_not_sufficient_condition_l224_224370

def isEllipse (a b : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x y = 1

theorem necessary_but_not_sufficient_condition (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  isEllipse a b (λ x y => a * x^2 + b * y^2) → ¬(∃ x y : ℝ, a * x^2 + b * y^2 = 1) :=
sorry

end necessary_but_not_sufficient_condition_l224_224370


namespace shipping_cost_per_unit_l224_224721

-- Define the conditions
def cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def num_components : ℝ := 150
def lowest_selling_price : ℝ := 196.67

-- Define the revenue and total cost
def total_cost (S : ℝ) : ℝ := (cost_per_component * num_components) + fixed_monthly_cost + (num_components * S)
def total_revenue : ℝ := lowest_selling_price * num_components

-- Define the proposition to be proved
theorem shipping_cost_per_unit (S : ℝ) :
  total_cost S ≤ total_revenue → S ≤ 6.67 :=
by sorry

end shipping_cost_per_unit_l224_224721


namespace translated_graph_symmetric_l224_224306

noncomputable def f (x : ℝ) : ℝ := sorry

theorem translated_graph_symmetric (f : ℝ → ℝ)
  (h_translate : ∀ x, f (x - 1) = e^x)
  (h_symmetric : ∀ x, f x = f (-x)) :
  ∀ x, f x = e^(-x - 1) :=
by
  sorry

end translated_graph_symmetric_l224_224306


namespace expenditure_on_house_rent_l224_224038

variable (X : ℝ) -- Let X be Bhanu's total income in rupees

-- Condition 1: Bhanu spends 300 rupees on petrol, which is 30% of his income
def condition_on_petrol : Prop := 0.30 * X = 300

-- Definition of remaining income
def remaining_income : ℝ := X - 300

-- Definition of house rent expenditure: 10% of remaining income
def house_rent : ℝ := 0.10 * remaining_income X

-- Theorem: If the condition on petrol holds, then the house rent expenditure is 70 rupees
theorem expenditure_on_house_rent (h : condition_on_petrol X) : house_rent X = 70 :=
  sorry

end expenditure_on_house_rent_l224_224038


namespace number_with_29_proper_divisors_is_720_l224_224028

theorem number_with_29_proper_divisors_is_720
  (n : ℕ) (h1 : n < 1000)
  (h2 : ∀ d, 1 < d ∧ d < n -> ∃ m, n = d * m):
  n = 720 := by
  sorry

end number_with_29_proper_divisors_is_720_l224_224028


namespace problem1_problem2_l224_224451

-- Problem 1: Prove that 3 * sqrt(20) - sqrt(45) + sqrt(1 / 5) = (16 * sqrt(5)) / 5
theorem problem1 : 3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1 / 5) = (16 * Real.sqrt 5) / 5 := 
sorry

-- Problem 2: Prove that (sqrt(6) - 2 * sqrt(3))^2 - (2 * sqrt(5) + sqrt(2)) * (2 * sqrt(5) - sqrt(2)) = -12 * sqrt(2)
theorem problem2 : (Real.sqrt 6 - 2 * Real.sqrt 3) ^ 2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2 := 
sorry

end problem1_problem2_l224_224451


namespace lily_profit_is_correct_l224_224638

-- Define the conditions
def first_ticket_price : ℕ := 1
def price_increment : ℕ := 1
def number_of_tickets : ℕ := 5
def prize_amount : ℕ := 11

-- Define the sum of arithmetic series formula
def total_amount_collected (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Calculate the total amount collected
def total : ℕ := total_amount_collected number_of_tickets first_ticket_price price_increment

-- Define the profit calculation
def profit : ℕ := total - prize_amount

-- The statement we need to prove
theorem lily_profit_is_correct : profit = 4 := by
  sorry

end lily_profit_is_correct_l224_224638


namespace hypotenuse_intersection_incircle_diameter_l224_224851

/-- Let \( a \) and \( b \) be the legs of a right triangle with hypotenuse \( c \). 
    Let two circles be centered at the endpoints of the hypotenuse, with radii \( a \) and \( b \). 
    Prove that the segment of the hypotenuse that lies in the intersection of the two circles is equal in length to the diameter of the incircle of the triangle. -/
theorem hypotenuse_intersection_incircle_diameter (a b : ℝ) :
    let c := Real.sqrt (a^2 + b^2)
    let x := a + b - c
    let r := (a + b - c) / 2
    x = 2 * r :=
by
  let c := Real.sqrt (a^2 + b^2)
  let x := a + b - c
  let r := (a + b - c) / 2
  show x = 2 * r
  sorry

end hypotenuse_intersection_incircle_diameter_l224_224851


namespace symmetric_point_l224_224736

theorem symmetric_point : ∃ (x0 y0 : ℝ), 
  (x0 = -6 ∧ y0 = -3) ∧ 
  (∃ (m1 m2 : ℝ), 
    m1 = -1 ∧ 
    m2 = (y0 - 2) / (x0 + 1) ∧ 
    m1 * m2 = -1) ∧ 
  (∃ (x_mid y_mid : ℝ), 
    x_mid = (x0 - 1) / 2 ∧ 
    y_mid = (y0 + 2) / 2 ∧ 
    x_mid + y_mid + 4 = 0) := 
sorry

end symmetric_point_l224_224736


namespace rowing_distance_l224_224248
-- Lean 4 Statement

theorem rowing_distance (v_m v_t D : ℝ) 
  (h1 : D = v_m + v_t)
  (h2 : 30 = 10 * (v_m - v_t))
  (h3 : 30 = 6 * (v_m + v_t)) :
  D = 5 :=
by sorry

end rowing_distance_l224_224248


namespace product_of_integers_l224_224217

theorem product_of_integers (A B C D : ℚ)
  (h1 : A + B + C + D = 100)
  (h2 : A + 5 = B - 5)
  (h3 : A + 5 = 2 * C)
  (h4 : A + 5 = D / 2) :
  A * B * C * D = 1517000000 / 6561 := by
  sorry

end product_of_integers_l224_224217


namespace Dave_earning_l224_224610

def action_games := 3
def adventure_games := 2
def role_playing_games := 3

def price_action := 6
def price_adventure := 5
def price_role_playing := 7

def earning_from_action_games := action_games * price_action
def earning_from_adventure_games := adventure_games * price_adventure
def earning_from_role_playing_games := role_playing_games * price_role_playing

def total_earning := earning_from_action_games + earning_from_adventure_games + earning_from_role_playing_games

theorem Dave_earning : total_earning = 49 := by
  show total_earning = 49
  sorry

end Dave_earning_l224_224610


namespace kendra_fish_count_l224_224567

variable (K : ℕ) -- Number of fish Kendra caught
variable (Ken_fish : ℕ) -- Number of fish Ken brought home

-- Conditions
axiom twice_as_many : Ken_fish = 2 * K - 3
axiom total_fish : K + Ken_fish = 87

-- The theorem we need to prove
theorem kendra_fish_count : K = 30 :=
by
  -- Lean proof goes here
  sorry

end kendra_fish_count_l224_224567


namespace relationship_a_b_c_d_l224_224444

theorem relationship_a_b_c_d 
  (a b c d : ℤ)
  (h : (a + b + 1) * (d + a + 2) = (c + d + 1) * (b + c + 2)) : 
  a + b + c + d = -2 := 
sorry

end relationship_a_b_c_d_l224_224444


namespace bags_total_weight_l224_224695

noncomputable def total_weight_of_bags (x y z : ℕ) : ℕ := x + y + z

theorem bags_total_weight (x y z : ℕ) (h1 : x + y = 90) (h2 : y + z = 100) (h3 : z + x = 110) :
  total_weight_of_bags x y z = 150 :=
by
  sorry

end bags_total_weight_l224_224695


namespace original_deck_size_l224_224101

/-- 
Aubrey adds 2 additional cards to a deck and then splits the deck evenly among herself and 
two other players, each player having 18 cards. 
We want to prove that the original number of cards in the deck was 52. 
-/
theorem original_deck_size :
  ∃ (n : ℕ), (n + 2) / 3 = 18 ∧ n = 52 :=
by
  sorry

end original_deck_size_l224_224101


namespace prime_pairs_solution_l224_224330

theorem prime_pairs_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  7 * p * q^2 + p = q^3 + 43 * p^3 + 1 ↔ (p = 2 ∧ q = 7) :=
by
  sorry

end prime_pairs_solution_l224_224330


namespace pizza_slices_left_over_l224_224733

def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8
def small_pizzas_purchased : ℕ := 3
def large_pizzas_purchased : ℕ := 2
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

def total_pizza_slices : ℕ := (small_pizzas_purchased * small_pizza_slices) + (large_pizzas_purchased * large_pizza_slices)
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

theorem pizza_slices_left_over : total_pizza_slices - total_slices_eaten = 10 :=
by sorry

end pizza_slices_left_over_l224_224733


namespace abs_eq_imp_b_eq_2_l224_224597

theorem abs_eq_imp_b_eq_2 (b : ℝ) (h : |1 - b| = |3 - b|) : b = 2 := 
sorry

end abs_eq_imp_b_eq_2_l224_224597


namespace percentage_of_second_solution_correct_l224_224419

noncomputable def percentage_of_alcohol_in_second_solution : ℝ :=
  let total_liters := 80
  let percentage_final_solution := 0.49
  let volume_first_solution := 24
  let percentage_first_solution := 0.4
  let volume_second_solution := 56
  let total_alcohol_in_final_solution := total_liters * percentage_final_solution
  let total_alcohol_first_solution := volume_first_solution * percentage_first_solution
  let x := (total_alcohol_in_final_solution - total_alcohol_first_solution) / volume_second_solution
  x

theorem percentage_of_second_solution_correct : 
  percentage_of_alcohol_in_second_solution = 0.5285714286 := by sorry

end percentage_of_second_solution_correct_l224_224419


namespace same_school_probability_l224_224843

theorem same_school_probability :
  let total_teachers : ℕ := 6
  let teachers_from_school_A : ℕ := 3
  let teachers_from_school_B : ℕ := 3
  let ways_to_choose_2_from_6 : ℕ := Nat.choose total_teachers 2
  let ways_to_choose_2_from_A := Nat.choose teachers_from_school_A 2
  let ways_to_choose_2_from_B := Nat.choose teachers_from_school_B 2
  let same_school_ways : ℕ := ways_to_choose_2_from_A + ways_to_choose_2_from_B
  let probability := (same_school_ways : ℚ) / ways_to_choose_2_from_6 
  probability = (2 : ℚ) / (5 : ℚ) := by sorry

end same_school_probability_l224_224843


namespace market_value_of_stock_l224_224012

theorem market_value_of_stock 
  (yield : ℝ) 
  (dividend_percentage : ℝ) 
  (face_value : ℝ) 
  (market_value : ℝ) 
  (h1 : yield = 0.10) 
  (h2 : dividend_percentage = 0.07) 
  (h3 : face_value = 100) 
  (h4 : market_value = (dividend_percentage * face_value) / yield) :
  market_value = 70 := by
  sorry

end market_value_of_stock_l224_224012


namespace rate_at_which_bowls_were_bought_l224_224215

theorem rate_at_which_bowls_were_bought 
    (total_bowls : ℕ) (sold_bowls : ℕ) (price_per_sold_bowl : ℝ) (remaining_bowls : ℕ) (percentage_gain : ℝ) 
    (total_bowls_eq : total_bowls = 115) 
    (sold_bowls_eq : sold_bowls = 104) 
    (price_per_sold_bowl_eq : price_per_sold_bowl = 20) 
    (remaining_bowls_eq : remaining_bowls = 11) 
    (percentage_gain_eq : percentage_gain = 0.4830917874396135) 
  : ∃ (R : ℝ), R = 18 :=
  sorry

end rate_at_which_bowls_were_bought_l224_224215


namespace percentage_increase_decrease_exceeds_original_l224_224805

open Real

theorem percentage_increase_decrease_exceeds_original (p q M : ℝ) (hp : 0 < p) (hq1 : 0 < q) (hq2 : q < 100) (hM : 0 < M) :
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > (100 * q) / (100 - q)) :=
by
  sorry

end percentage_increase_decrease_exceeds_original_l224_224805


namespace largest_four_digit_by_two_moves_l224_224325

def moves (n : Nat) (d1 d2 d3 d4 : Nat) : Prop :=
  ∃ x y : ℕ, d1 = x → d2 = y → n = 1405 → (x ≤ 2 ∧ y ≤ 2)

theorem largest_four_digit_by_two_moves :
  ∃ n : ℕ, moves 1405 1 4 0 5 ∧ n = 7705 :=
by
  sorry

end largest_four_digit_by_two_moves_l224_224325


namespace sqrt_mult_minus_two_l224_224064

theorem sqrt_mult_minus_two (x y : ℝ) (hx : x = Real.sqrt 3) (hy : y = Real.sqrt 6) : 
  2 < x * y - 2 ∧ x * y - 2 < 3 := by
  sorry

end sqrt_mult_minus_two_l224_224064


namespace barbi_weight_loss_duration_l224_224134

theorem barbi_weight_loss_duration :
  (∃ x : ℝ, 
    (∃ l_barbi l_luca : ℝ, 
      l_barbi = 1.5 * x ∧ 
      l_luca = 99 ∧ 
      l_luca = l_barbi + 81) ∧
    x = 12) :=
by
  sorry

end barbi_weight_loss_duration_l224_224134


namespace inequality_solution_empty_l224_224146

theorem inequality_solution_empty {a x: ℝ} : 
  (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0 → 
  (-2 < a) ∧ (a < 6 / 5) :=
sorry

end inequality_solution_empty_l224_224146


namespace girls_in_class4_1_l224_224389

theorem girls_in_class4_1 (total_students grade: ℕ)
    (total_girls: ℕ)
    (students_class4_1: ℕ)
    (boys_class4_2: ℕ)
    (h1: total_students = 72)
    (h2: total_girls = 35)
    (h3: students_class4_1 = 36)
    (h4: boys_class4_2 = 19) :
    (total_girls - (total_students - students_class4_1 - boys_class4_2) = 18) :=
by
    sorry

end girls_in_class4_1_l224_224389


namespace cost_to_fill_pool_l224_224596

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l224_224596


namespace arithmetic_mean_eqn_l224_224951

theorem arithmetic_mean_eqn : 
  (3/5 + 6/7) / 2 = 51/70 :=
  by sorry

end arithmetic_mean_eqn_l224_224951


namespace trays_from_second_table_l224_224894

def trays_per_trip : ℕ := 4
def trips : ℕ := 9
def trays_from_first_table : ℕ := 20

theorem trays_from_second_table :
  trays_per_trip * trips - trays_from_first_table = 16 :=
by
  sorry

end trays_from_second_table_l224_224894


namespace distance_traveled_on_foot_l224_224226

theorem distance_traveled_on_foot (x y : ℝ) (h1 : x + y = 80) (h2 : x / 8 + y / 16 = 7) : x = 32 :=
by
  sorry

end distance_traveled_on_foot_l224_224226


namespace number_of_rods_in_one_mile_l224_224144

theorem number_of_rods_in_one_mile (miles_to_furlongs : 1 = 10 * 1)
  (furlongs_to_rods : 1 = 50 * 1) : 1 = 500 * 1 :=
by {
  sorry
}

end number_of_rods_in_one_mile_l224_224144


namespace range_of_a_l224_224824

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a * x^2 - x + 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ -1) ↔ (a ≥ 1/12) :=
by
  sorry

end range_of_a_l224_224824


namespace possible_values_of_A_l224_224179

theorem possible_values_of_A :
  ∃ (A : ℕ), (A ≤ 4 ∧ A < 10) ∧ A = 5 :=
sorry

end possible_values_of_A_l224_224179


namespace mirror_area_proof_l224_224005

-- Definitions of conditions
def outer_width := 100
def outer_height := 70
def frame_width := 15
def mirror_width := outer_width - 2 * frame_width -- 100 - 2 * 15 = 70
def mirror_height := outer_height - 2 * frame_width -- 70 - 2 * 15 = 40

-- Statement of the proof problem
theorem mirror_area_proof : 
  (mirror_width * mirror_height) = 2800 := 
by
  sorry

end mirror_area_proof_l224_224005


namespace digit_A_divisibility_l224_224257

theorem digit_A_divisibility :
  ∃ (A : ℕ), (0 ≤ A ∧ A < 10) ∧ (∃ k_5 : ℕ, 353809 * 10 + A = 5 * k_5) ∧ 
  (∃ k_7 : ℕ, 353809 * 10 + A = 7 * k_7) ∧ (∃ k_11 : ℕ, 353809 * 10 + A = 11 * k_11) 
  ∧ A = 0 :=
by 
  sorry

end digit_A_divisibility_l224_224257


namespace product_mnp_l224_224357

theorem product_mnp (a x y z c : ℕ) (m n p : ℕ) :
  (a ^ 8 * x * y * z - a ^ 7 * y * z - a ^ 6 * x * z = a ^ 5 * (c ^ 5 - 1) ∧
   (a ^ m * x * z - a ^ n) * (a ^ p * y * z - a ^ 3) = a ^ 5 * c ^ 5) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  sorry

end product_mnp_l224_224357


namespace sales_volume_relation_maximize_profit_l224_224919

-- Definition of the conditions given in the problem
def cost_price : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def sales_decrease_rate : ℝ := 20

-- Lean statement for part 1
theorem sales_volume_relation (x : ℝ) : 
  (45 ≤ x) →
  (y = 700 - 20 * (x - 45)) → 
  y = -20 * x + 1600 := sorry

-- Lean statement for part 2
theorem maximize_profit (x : ℝ) :
  (45 ≤ x) →
  (P = (x - 40) * (-20 * x + 1600)) →
  ∃ max_x max_P, max_x = 60 ∧ max_P = 8000 := sorry

end sales_volume_relation_maximize_profit_l224_224919


namespace algebraic_expression_value_l224_224334

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := by
  sorry

end algebraic_expression_value_l224_224334


namespace find_a_value_l224_224631

namespace Proof

-- Define the context and variables
variables (a b c : ℝ)
variables (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variables (h2 : a * 15 * 2 = 4)

-- State the theorem we want to prove
theorem find_a_value: a = 6 :=
by
  sorry

end Proof

end find_a_value_l224_224631


namespace wrapping_paper_area_l224_224042

theorem wrapping_paper_area (a : ℝ) (h : ℝ) : h = a ∧ 1 ≥ 0 → 4 * a^2 = 4 * a^2 :=
by sorry

end wrapping_paper_area_l224_224042


namespace kombucha_bottles_l224_224926

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end kombucha_bottles_l224_224926


namespace y_paisa_for_each_rupee_x_l224_224737

theorem y_paisa_for_each_rupee_x (p : ℕ) (x : ℕ) (y_share total_amount : ℕ) 
  (h₁ : y_share = 2700) 
  (h₂ : total_amount = 10500) 
  (p_condition : (130 + p) * x = total_amount) 
  (y_condition : p * x = y_share) : 
  p = 45 := 
by
  sorry

end y_paisa_for_each_rupee_x_l224_224737


namespace maximize_ab2c3_l224_224918

def positive_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def sum_constant (a b c A : ℝ) : Prop :=
  a + b + c = A

noncomputable def maximize_expression (a b c : ℝ) : ℝ :=
  a * b^2 * c^3

theorem maximize_ab2c3 (a b c A : ℝ) (h1 : positive_numbers a b c)
  (h2 : sum_constant a b c A) : 
  maximize_expression a b c ≤ maximize_expression (A / 6) (A / 3) (A / 2) :=
sorry

end maximize_ab2c3_l224_224918


namespace necessary_but_not_sufficient_condition_l224_224390

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l224_224390


namespace sample_size_calculation_l224_224082

theorem sample_size_calculation :
  let workshop_A := 120
  let workshop_B := 80
  let workshop_C := 60
  let sample_from_C := 3
  let sampling_fraction := sample_from_C / workshop_C
  let sample_A := workshop_A * sampling_fraction
  let sample_B := workshop_B * sampling_fraction
  let sample_C := workshop_C * sampling_fraction
  let n := sample_A + sample_B + sample_C
  n = 13 :=
by
  sorry

end sample_size_calculation_l224_224082


namespace original_cost_price_l224_224254

theorem original_cost_price (S P C : ℝ) (h1 : S = 260) (h2 : S = 1.20 * C) : C = 216.67 := sorry

end original_cost_price_l224_224254


namespace right_triangle_acute_angle_l224_224068

theorem right_triangle_acute_angle (A B : ℝ) (h₁ : A + B = 90) (h₂ : A = 40) : B = 50 :=
by
  sorry

end right_triangle_acute_angle_l224_224068


namespace four_squares_cover_larger_square_l224_224219

structure Square :=
  (side : ℝ) (h_positive : side > 0)

theorem four_squares_cover_larger_square (large small : Square) 
  (h_side_relation: large.side = 2 * small.side) : 
  large.side^2 = 4 * small.side^2 :=
by
  sorry

end four_squares_cover_larger_square_l224_224219


namespace certain_number_l224_224579

theorem certain_number (a b : ℕ) (n : ℕ) 
  (h1: a % n = 0) (h2: b % n = 0) 
  (h3: b = a + 9 * n)
  (h4: b = a + 126) : n = 14 :=
by
  sorry

end certain_number_l224_224579


namespace sum_abs_values_of_factors_l224_224238

theorem sum_abs_values_of_factors (a w c d : ℤ)
  (h1 : 6 * (x : ℤ)^2 + x - 12 = (a * x + w) * (c * x + d)) :
  abs a + abs w + abs c + abs d = 22 :=
sorry

end sum_abs_values_of_factors_l224_224238


namespace find_xyz_l224_224976

theorem find_xyz (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 45) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15) (h3 : x + y + z = 5) : x * y * z = 10 :=
by
  sorry

end find_xyz_l224_224976


namespace part1_part2_l224_224578

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2

theorem part1 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → f x a > 3*a*x) → a < 2*Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) :
  ∀ x : ℝ,
    ((a = 0) → x > 2) ∧
    ((a > 0) → (x < -1/a ∨ x > 2)) ∧
    ((-1/2 < a ∧ a < 0) → (2 < x ∧ x < -1/a)) ∧
    ((a = -1/2) → false) ∧
    ((a < -1/2) → (-1/a < x ∧ x < 2)) :=
sorry

end part1_part2_l224_224578


namespace quadratic_b_value_l224_224075

theorem quadratic_b_value {b m : ℝ} (h : ∀ x, x^2 + b * x + 44 = (x + m)^2 + 8) : b = 12 :=
by
  -- hint for proving: expand (x+m)^2 + 8 and equate it with x^2 + bx + 44 to solve for b 
  sorry

end quadratic_b_value_l224_224075


namespace eq_op_op_op_92_l224_224937

noncomputable def opN (N : ℝ) : ℝ := 0.75 * N + 2

theorem eq_op_op_op_92 : opN (opN (opN 92)) = 43.4375 :=
by
  sorry

end eq_op_op_op_92_l224_224937


namespace range_of_2x_plus_y_range_of_c_l224_224009

open Real

def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

theorem range_of_2x_plus_y (x y : ℝ) (h : point_on_circle x y) : 
  1 - sqrt 2 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + sqrt 2 :=
sorry

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, point_on_circle x y → x + y + c > 0) → c ≥ -1 :=
sorry

end range_of_2x_plus_y_range_of_c_l224_224009


namespace red_balls_removal_condition_l224_224503

theorem red_balls_removal_condition (total_balls : ℕ) (initial_red_balls : ℕ) (r : ℕ) : 
  total_balls = 600 → 
  initial_red_balls = 420 → 
  60 * (total_balls - r) = 100 * (initial_red_balls - r) → 
  r = 150 :=
by
  sorry

end red_balls_removal_condition_l224_224503


namespace production_rate_l224_224274

theorem production_rate (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * x * x = x) → (y * y * z) / x^2 = y^2 * z / x^2 :=
by
  intro h
  sorry

end production_rate_l224_224274


namespace rate_per_square_meter_l224_224522

theorem rate_per_square_meter 
  (L : ℝ) (W : ℝ) (C : ℝ)
  (hL : L = 5.5) 
  (hW : W = 3.75)
  (hC : C = 20625)
  : C / (L * W) = 1000 :=
by
  sorry

end rate_per_square_meter_l224_224522


namespace sum_of_first_three_terms_l224_224528

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end sum_of_first_three_terms_l224_224528


namespace jose_julia_completion_time_l224_224513

variable (J N L : ℝ)

theorem jose_julia_completion_time :
  J + N + L = 1/4 ∧
  J * (1/3) = 1/18 ∧
  N = 1/9 ∧
  L * (1/3) = 1/18 →
  1/J = 6 ∧ 1/L = 6 := sorry

end jose_julia_completion_time_l224_224513


namespace fractions_equiv_x_zero_l224_224436

theorem fractions_equiv_x_zero (x b : ℝ) (h : x + 3 * b ≠ 0) : 
  (x + 2 * b) / (x + 3 * b) = 2 / 3 ↔ x = 0 :=
by sorry

end fractions_equiv_x_zero_l224_224436


namespace main_theorem_l224_224108

/-- A good integer is an integer whose absolute value is not a perfect square. -/
def good (n : ℤ) : Prop := ∀ k : ℤ, k^2 ≠ |n|

/-- Integer m can be represented as a sum of three distinct good integers u, v, w whose product is the square of an odd integer. -/
def special_representation (m : ℤ) : Prop :=
  ∃ u v w : ℤ,
    good u ∧ good v ∧ good w ∧
    (u ≠ v ∧ u ≠ w ∧ v ≠ w) ∧
    (∃ k : ℤ, (u * v * w = k^2 ∧ k % 2 = 1)) ∧
    (m = u + v + w)

/-- All integers m having the property that they can be represented in infinitely many ways as a sum of three distinct good integers whose product is the square of an odd integer are those which are congruent to 3 modulo 4. -/
theorem main_theorem (m : ℤ) : special_representation m ↔ m % 4 = 3 := sorry

end main_theorem_l224_224108


namespace simplify_and_evaluate_l224_224115

theorem simplify_and_evaluate :
  ∀ (x y : ℝ), x = -1/2 → y = 3 → 3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l224_224115


namespace hypotenuse_length_l224_224324

variables (a b c : ℝ)

-- Definitions from conditions
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def sum_of_squares_is_2000 (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 2000

def perimeter_is_60 (a b c : ℝ) : Prop :=
  a + b + c = 60

theorem hypotenuse_length (a b c : ℝ)
  (h1 : right_angled_triangle a b c)
  (h2 : sum_of_squares_is_2000 a b c)
  (h3 : perimeter_is_60 a b c) :
  c = 10 * Real.sqrt 10 :=
sorry

end hypotenuse_length_l224_224324


namespace angles_arithmetic_sequence_sides_l224_224296

theorem angles_arithmetic_sequence_sides (A B C a b c : ℝ)
  (h_angle_ABC : A + B + C = 180)
  (h_arithmetic_sequence : 2 * B = A + C)
  (h_cos_B : A * A + c * c - b * b = 2 * a * c)
  (angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < 180 ∧ B < 180 ∧ C < 180) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end angles_arithmetic_sequence_sides_l224_224296


namespace correct_operation_l224_224847

variable (a : ℕ)

theorem correct_operation :
  (3 * a + 2 * a ≠ 5 * a^2) ∧
  (3 * a - 2 * a ≠ 1) ∧
  a^2 * a^3 = a^5 ∧
  (a / a^2 ≠ a) :=
by
  sorry

end correct_operation_l224_224847


namespace max_xy_value_l224_224018

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l224_224018


namespace john_needs_more_money_l224_224087

def total_needed : ℝ := 2.50
def current_amount : ℝ := 0.75
def remaining_amount : ℝ := 1.75

theorem john_needs_more_money : total_needed - current_amount = remaining_amount :=
by
  sorry

end john_needs_more_money_l224_224087


namespace no_integer_solution_l224_224004

theorem no_integer_solution (x : ℤ) : ¬ (x + 12 > 15 ∧ -3 * x > -9) :=
by {
  sorry
}

end no_integer_solution_l224_224004


namespace first_number_value_l224_224198

theorem first_number_value (A B LCM HCF : ℕ) (h_lcm : LCM = 2310) (h_hcf : HCF = 30) (h_b : B = 210) (h_mul : A * B = LCM * HCF) : A = 330 := 
by
  -- Use sorry to skip the proof
  sorry

end first_number_value_l224_224198


namespace length_of_top_side_l224_224187

def height_of_trapezoid : ℝ := 8
def area_of_trapezoid : ℝ := 72
def top_side_is_shorter (b : ℝ) : Prop := ∃ t : ℝ, t = b - 6

theorem length_of_top_side (b t : ℝ) (h_height : height_of_trapezoid = 8)
  (h_area : area_of_trapezoid = 72) 
  (h_top_side : top_side_is_shorter b)
  (h_area_formula : (1/2) * (b + t) * 8 = 72) : t = 6 := 
by 
  sorry

end length_of_top_side_l224_224187


namespace slower_speed_is_35_l224_224973

-- Define the given conditions
def distance : ℝ := 70 -- distance is 70 km
def speed_on_time : ℝ := 40 -- on-time average speed is 40 km/hr
def delay : ℝ := 0.25 -- delay is 15 minutes or 0.25 hours

-- This is the statement we need to prove
theorem slower_speed_is_35 :
  ∃ slower_speed : ℝ, 
    slower_speed = distance / (distance / speed_on_time + delay) ∧ slower_speed = 35 :=
by
  sorry

end slower_speed_is_35_l224_224973


namespace perpendicular_k_value_exists_l224_224730

open Real EuclideanSpace

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (3, 2)

theorem perpendicular_k_value_exists : ∃ k : ℝ, (vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) ∧ k = 5 / 4 := by
  sorry

end perpendicular_k_value_exists_l224_224730


namespace hats_count_l224_224312

theorem hats_count (T M W : ℕ) (hT : T = 1800)
  (hM : M = (2 * T) / 3) (hW : W = T - M) 
  (hats_men : ℕ) (hats_women : ℕ) (m_hats_condition : hats_men = 15 * M / 100)
  (w_hats_condition : hats_women = 25 * W / 100) :
  hats_men + hats_women = 330 :=
by sorry

end hats_count_l224_224312


namespace correct_insights_l224_224259

def insight1 := ∀ connections : Type, (∃ journey : connections → Prop, ∀ (x : connections), ¬journey x)
def insight2 := ∀ connections : Type, (∃ (beneficial : connections → Prop), ∀ (x : connections), beneficial x → True)
def insight3 := ∀ connections : Type, (∃ (accidental : connections → Prop), ∀ (x : connections), accidental x → False)
def insight4 := ∀ connections : Type, (∃ (conditional : connections → Prop), ∀ (x : connections), conditional x → True)

theorem correct_insights : ¬ insight1 ∧ insight2 ∧ ¬ insight3 ∧ insight4 :=
by sorry

end correct_insights_l224_224259


namespace no_rectangle_from_six_different_squares_l224_224292

theorem no_rectangle_from_six_different_squares (a1 a2 a3 a4 a5 a6 : ℝ) (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) :
  ¬ (∃ (L W : ℝ), a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = L * W) :=
sorry

end no_rectangle_from_six_different_squares_l224_224292


namespace train_length_l224_224688

theorem train_length (L : ℝ) :
  (20 * (L + 160) = 15 * (L + 250)) -> L = 110 :=
by
  intro h
  sorry

end train_length_l224_224688


namespace gcd_7429_12345_l224_224397

theorem gcd_7429_12345 : Int.gcd 7429 12345 = 1 := 
by 
  sorry

end gcd_7429_12345_l224_224397


namespace athenas_min_wins_l224_224016

theorem athenas_min_wins (total_games : ℕ) (games_played : ℕ) (wins_so_far : ℕ) (losses_so_far : ℕ) 
                          (win_percentage_threshold : ℝ) (remaining_games : ℕ) (additional_wins_needed : ℕ) :
  total_games = 44 ∧ games_played = wins_so_far + losses_so_far ∧ wins_so_far = 20 ∧ losses_so_far = 15 ∧ 
  win_percentage_threshold = 0.6 ∧ remaining_games = total_games - games_played ∧ additional_wins_needed = 27 - wins_so_far → 
  additional_wins_needed = 7 :=
by
  sorry

end athenas_min_wins_l224_224016


namespace range_of_f_l224_224925

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ y, (∃ x, x ∈ Set.Icc (-1:ℝ) 1 ∧ f x = y) ↔ y ∈ Set.Icc 0 (Real.pi^4 / 8) :=
sorry

end range_of_f_l224_224925


namespace sally_turnip_count_l224_224472

theorem sally_turnip_count (total_turnips : ℕ) (mary_turnips : ℕ) (sally_turnips : ℕ) 
  (h1: total_turnips = 242) 
  (h2: mary_turnips = 129) 
  (h3: total_turnips = mary_turnips + sally_turnips) : 
  sally_turnips = 113 := 
by 
  sorry

end sally_turnip_count_l224_224472


namespace inequality_holds_for_all_real_numbers_l224_224609

theorem inequality_holds_for_all_real_numbers (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (k ∈ Set.Icc (-3 : ℝ) 0) := 
sorry

end inequality_holds_for_all_real_numbers_l224_224609


namespace coefficient_of_x_in_expansion_l224_224615

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end coefficient_of_x_in_expansion_l224_224615


namespace B_can_finish_alone_in_27_5_days_l224_224711

-- Definitions of work rates
variable (B A C : Type)

-- Conditions translations
def efficiency_of_A (x : ℝ) : Prop := ∀ (work_rate_A work_rate_B : ℝ), work_rate_A = 1 / (2 * x) ∧ work_rate_B = 1 / x
def efficiency_of_C (x : ℝ) : Prop := ∀ (work_rate_C work_rate_B : ℝ), work_rate_C = 1 / (3 * x) ∧ work_rate_B = 1 / x
def combined_work_rate (x : ℝ) : Prop := (1 / (2 * x) + 1 / x + 1 / (3 * x)) = 1 / 15

-- Proof statement
theorem B_can_finish_alone_in_27_5_days :
  ∃ (x : ℝ), efficiency_of_A x ∧ efficiency_of_C x ∧ combined_work_rate x ∧ x = 27.5 :=
sorry

end B_can_finish_alone_in_27_5_days_l224_224711


namespace hyperbola_asymptotes_l224_224084

theorem hyperbola_asymptotes (a : ℝ) (x y : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  (∃ M : ℝ × ℝ, M.1 ^ 2 / a ^ 2 - M.2 ^ 2 = 1 ∧ M.2 ^ 2 = 8 * M.1 ∧ abs (dist M F) = 5) →
  (F.1 = 2 ∧ F.2 = 0) →
  (a = 3 / 5) → 
  (∀ x y : ℝ, (5 * x + 3 * y = 0) ∨ (5 * x - 3 * y = 0)) :=
by
  sorry

end hyperbola_asymptotes_l224_224084


namespace sum_reciprocal_eq_eleven_eighteen_l224_224399

noncomputable def sum_reciprocal (n : ℕ) : ℝ := ∑' (n : ℕ), 1 / (n * (n + 3))

theorem sum_reciprocal_eq_eleven_eighteen :
  sum_reciprocal = 11 / 18 :=
by
  sorry

end sum_reciprocal_eq_eleven_eighteen_l224_224399


namespace find_a7_l224_224810

theorem find_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ)
  (h : x^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 +
            a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 +
            a_7 * (x + 1)^7 + a_8 * (x + 1)^8) : 
  a_7 = -8 := 
sorry

end find_a7_l224_224810


namespace partners_in_firm_l224_224637

theorem partners_in_firm (P A : ℕ) (h1 : P * 63 = 2 * A) (h2 : P * 34 = 1 * (A + 45)) : P = 18 :=
by
  sorry

end partners_in_firm_l224_224637


namespace merchant_cost_price_l224_224473

theorem merchant_cost_price (x : ℝ) (h₁ : x + (x^2 / 100) = 39) : x = 30 :=
sorry

end merchant_cost_price_l224_224473


namespace g_at_pi_over_3_l224_224487

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

noncomputable def g (x : ℝ) (ω φ : ℝ) : ℝ := 3 * Real.sin (ω * x + φ) - 1

theorem g_at_pi_over_3 (ω φ : ℝ) :
  (∀ x : ℝ, f (π / 3 + x) ω φ = f (π / 3 - x) ω φ) →
  g (π / 3) ω φ = -1 :=
by sorry

end g_at_pi_over_3_l224_224487


namespace quadratic_has_two_distinct_real_roots_l224_224667

theorem quadratic_has_two_distinct_real_roots :
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 :=
by
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  show discriminant > 0
  sorry

end quadratic_has_two_distinct_real_roots_l224_224667


namespace estimated_red_balls_l224_224321

-- Definitions based on conditions
def total_balls : ℕ := 15
def black_ball_frequency : ℝ := 0.6
def red_ball_frequency : ℝ := 1 - black_ball_frequency

-- Theorem stating the proof problem
theorem estimated_red_balls :
  (total_balls : ℝ) * red_ball_frequency = 6 := by
  sorry

end estimated_red_balls_l224_224321


namespace water_flow_total_l224_224461

theorem water_flow_total
  (R1 R2 R3 : ℕ)
  (h1 : R2 = 36)
  (h2 : R2 = (3 / 2) * R1)
  (h3 : R3 = (5 / 4) * R2)
  : R1 + R2 + R3 = 105 :=
sorry

end water_flow_total_l224_224461


namespace find_number_l224_224123

theorem find_number (N : ℕ) (h : N / 16 = 16 * 8) : N = 2048 :=
sorry

end find_number_l224_224123


namespace odd_multiple_of_9_implies_multiple_of_3_l224_224734

-- Define an odd number that is a multiple of 9
def odd_multiple_of_nine (m : ℤ) : Prop := 9 * m % 2 = 1

-- Define multiples of 3 and 9
def multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k
def multiple_of_nine (n : ℤ) : Prop := ∃ k : ℤ, n = 9 * k

-- The main statement
theorem odd_multiple_of_9_implies_multiple_of_3 (n : ℤ) 
  (h1 : ∀ n, multiple_of_nine n → multiple_of_three n)
  (h2 : odd_multiple_of_nine n ∧ multiple_of_nine n) : 
  multiple_of_three n :=
by sorry

end odd_multiple_of_9_implies_multiple_of_3_l224_224734


namespace find_y_value_l224_224574

noncomputable def y_value (y : ℝ) : Prop :=
  let side1_sq_area := 9 * y^2
  let side2_sq_area := 36 * y^2
  let triangle_area := 9 * y^2
  (side1_sq_area + side2_sq_area + triangle_area = 1000)

theorem find_y_value (y : ℝ) : y_value y → y = 10 * Real.sqrt 3 / 3 :=
sorry

end find_y_value_l224_224574


namespace roof_ratio_l224_224197

theorem roof_ratio (L W : ℕ) (h1 : L * W = 768) (h2 : L - W = 32) : L / W = 3 := 
sorry

end roof_ratio_l224_224197


namespace quotient_of_numbers_l224_224700

noncomputable def larger_number : ℕ := 22
noncomputable def smaller_number : ℕ := 8

theorem quotient_of_numbers : (larger_number.toFloat / smaller_number.toFloat) = 2.75 := by
  sorry

end quotient_of_numbers_l224_224700


namespace max_not_sum_S_l224_224889

def S : Set ℕ := {n | ∃ k : ℕ, n = 10^k + 1000}

theorem max_not_sum_S : ∀ x : ℕ, (∀ y ∈ S, ∃ m : ℕ, x ≠ m * y) ↔ x = 34999 := by
  sorry

end max_not_sum_S_l224_224889


namespace powers_of_i_cyclic_l224_224317

theorem powers_of_i_cyclic {i : ℂ} (h_i_squared : i^2 = -1) :
  i^(66) + i^(103) = -1 - i :=
by {
  -- Providing the proof steps as sorry.
  -- This is a placeholder for the actual proof.
  sorry
}

end powers_of_i_cyclic_l224_224317


namespace octagon_perimeter_l224_224422

theorem octagon_perimeter (n : ℕ) (side_length : ℝ) (h1 : n = 8) (h2 : side_length = 2) : 
  n * side_length = 16 :=
by
  sorry

end octagon_perimeter_l224_224422


namespace first_discount_percentage_l224_224639

theorem first_discount_percentage (x : ℕ) :
  let original_price := 175
  let discounted_price := original_price * (100 - x) / 100
  let final_price := discounted_price * 95 / 100
  final_price = 133 → x = 20 :=
by
  sorry

end first_discount_percentage_l224_224639


namespace no_positive_solution_for_special_k_l224_224329
open Nat

theorem no_positive_solution_for_special_k (p : ℕ) (hp : p.Prime) (hmod : p % 4 = 3) :
    ¬ ∃ n m k : ℕ, (n > 0) ∧ (m > 0) ∧ (k = p^2) ∧ (n^2 + m^2 = k * (m^4 + n)) :=
sorry

end no_positive_solution_for_special_k_l224_224329


namespace bus_full_problem_l224_224152

theorem bus_full_problem
      (cap : ℕ := 80)
      (first_pickup_ratio : ℚ := 3/5)
      (second_pickup_exit : ℕ := 15)
      (waiting_people : ℕ := 50) :
      waiting_people - (cap - (first_pickup_ratio * cap - second_pickup_exit)) = 3 := by
  sorry

end bus_full_problem_l224_224152


namespace exists_pretty_hexagon_max_area_pretty_hexagon_l224_224402

-- Define the condition of a "pretty" hexagon
structure PrettyHexagon (L ℓ h : ℝ) : Prop :=
  (diag1 : (L + ℓ)^2 + h^2 = 1)
  (diag2 : (L + ℓ)^2 + h^2 = 1)
  (diag3 : (L + ℓ)^2 + h^2 = 1)
  (diag4 : (L + ℓ)^2 + h^2 = 1)
  (L_pos : L > 0) (L_lt_1 : L < 1)
  (ℓ_pos : ℓ > 0) (ℓ_lt_1 : ℓ < 1)
  (h_pos : h > 0) (h_lt_1 : h < 1)

-- Area of the hexagon given L, ℓ, and h
def hexagon_area (L ℓ h : ℝ) := 2 * (L + ℓ) * h

-- Question (a): Existence of a pretty hexagon with a given area
theorem exists_pretty_hexagon (k : ℝ) (hk : 0 < k ∧ k < 1) : 
  ∃ L ℓ h : ℝ, PrettyHexagon L ℓ h ∧ hexagon_area L ℓ h = k :=
sorry

-- Question (b): Maximum area of any pretty hexagon is at most 1
theorem max_area_pretty_hexagon : 
  ∀ L ℓ h : ℝ, PrettyHexagon L ℓ h → hexagon_area L ℓ h ≤ 1 :=
sorry

end exists_pretty_hexagon_max_area_pretty_hexagon_l224_224402


namespace shelves_used_l224_224322

def initial_books : Nat := 87
def sold_books : Nat := 33
def books_per_shelf : Nat := 6

theorem shelves_used :
  (initial_books - sold_books) / books_per_shelf = 9 := by
  sorry

end shelves_used_l224_224322


namespace cube_root_rational_l224_224970

theorem cube_root_rational (a b : ℚ) (r : ℚ) (h1 : ∃ x : ℚ, x^3 = a) (h2 : ∃ y : ℚ, y^3 = b) (h3 : ∃ x y : ℚ, x + y = r ∧ x^3 = a ∧ y^3 = b) :
  (∃ x : ℚ, x^3 = a) ∧ (∃ y : ℚ, y^3 = b) :=
sorry

end cube_root_rational_l224_224970


namespace bacteria_after_10_hours_l224_224138

def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

theorem bacteria_after_10_hours : bacteria_count 10 = 1024 := by
  sorry

end bacteria_after_10_hours_l224_224138


namespace not_all_prime_distinct_l224_224449

theorem not_all_prime_distinct (a1 a2 a3 : ℕ) (h1 : a1 ≠ a2) (h2 : a2 ≠ a3) (h3 : a1 ≠ a3)
  (h4 : 0 < a1) (h5 : 0 < a2) (h6 : 0 < a3)
  (h7 : a1 ∣ (a2 + a3 + a2 * a3)) (h8 : a2 ∣ (a3 + a1 + a3 * a1)) (h9 : a3 ∣ (a1 + a2 + a1 * a2)) :
  ¬ (Nat.Prime a1 ∧ Nat.Prime a2 ∧ Nat.Prime a3) :=
by
  sorry

end not_all_prime_distinct_l224_224449


namespace number_of_schools_l224_224738

theorem number_of_schools (cost_per_school : ℝ) (population : ℝ) (savings_per_day_per_person : ℝ) (days_in_year : ℕ) :
  cost_per_school = 5 * 10^5 →
  population = 1.3 * 10^9 →
  savings_per_day_per_person = 0.01 →
  days_in_year = 365 →
  (population * savings_per_day_per_person * days_in_year) / cost_per_school = 9.49 * 10^3 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_schools_l224_224738


namespace a_minus_b_perfect_square_l224_224848

theorem a_minus_b_perfect_square (a b : ℕ) (h : 2 * a^2 + a = 3 * b^2 + b) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, a - b = k^2 :=
by sorry

end a_minus_b_perfect_square_l224_224848


namespace age_in_1988_equals_sum_of_digits_l224_224600

def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

def age_in_1988 (birth_year : ℕ) : ℕ := 1988 - birth_year

def sum_of_digits (x y : ℕ) : ℕ := 1 + 9 + x + y

theorem age_in_1988_equals_sum_of_digits (x y : ℕ) (h0 : 0 ≤ x) (h1 : x ≤ 9) (h2 : 0 ≤ y) (h3 : y ≤ 9) 
  (h4 : age_in_1988 (birth_year x y) = sum_of_digits x y) :
  age_in_1988 (birth_year x y) = 22 :=
by {
  sorry
}

end age_in_1988_equals_sum_of_digits_l224_224600


namespace eggs_per_basket_l224_224536

theorem eggs_per_basket (n : ℕ) (total_eggs_red total_eggs_orange min_eggs_per_basket : ℕ) (h_red : total_eggs_red = 20) (h_orange : total_eggs_orange = 30) (h_min : min_eggs_per_basket = 5) (h_div_red : total_eggs_red % n = 0) (h_div_orange : total_eggs_orange % n = 0) (h_at_least : n ≥ min_eggs_per_basket) : n = 5 :=
sorry

end eggs_per_basket_l224_224536


namespace trail_length_is_48_meters_l224_224182

noncomputable def length_of_trail (d: ℝ) : Prop :=
  let normal_speed := 8 -- normal speed in m/s
  let mud_speed := normal_speed / 4 -- speed in mud in m/s

  let time_mud := (1 / 3 * d) / mud_speed -- time through the mud in seconds
  let time_normal := (2 / 3 * d) / normal_speed -- time through the normal trail in seconds

  let total_time := 12 -- total time in seconds

  total_time = time_mud + time_normal

theorem trail_length_is_48_meters : ∃ d: ℝ, length_of_trail d ∧ d = 48 :=
sorry

end trail_length_is_48_meters_l224_224182


namespace class_funds_l224_224099

theorem class_funds (total_contribution : ℕ) (students : ℕ) (contribution_per_student : ℕ) (remaining_amount : ℕ) 
    (h1 : total_contribution = 90) 
    (h2 : students = 19) 
    (h3 : contribution_per_student = 4) 
    (h4 : remaining_amount = total_contribution - (students * contribution_per_student)) : 
    remaining_amount = 14 :=
sorry

end class_funds_l224_224099


namespace set_intersection_complement_l224_224726

-- Definitions of the sets A and B
def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

-- Statement of the problem for Lean 4
theorem set_intersection_complement :
  A ∩ (Set.compl B) = {1, 5, 7} := 
sorry

end set_intersection_complement_l224_224726


namespace negation_proof_l224_224944

theorem negation_proof : 
  (¬(∀ x : ℝ, x < 2^x) ↔ ∃ x : ℝ, x ≥ 2^x) :=
by
  sorry

end negation_proof_l224_224944


namespace time_saved_correct_l224_224945

-- Define the conditions as constants
def section1_problems : Nat := 20
def section2_problems : Nat := 15

def time_with_calc_sec1 : Nat := 3
def time_without_calc_sec1 : Nat := 8

def time_with_calc_sec2 : Nat := 5
def time_without_calc_sec2 : Nat := 10

-- Calculate the total times
def total_time_with_calc : Nat :=
  (section1_problems * time_with_calc_sec1) +
  (section2_problems * time_with_calc_sec2)

def total_time_without_calc : Nat :=
  (section1_problems * time_without_calc_sec1) +
  (section2_problems * time_without_calc_sec2)

-- The time saved using a calculator
def time_saved : Nat :=
  total_time_without_calc - total_time_with_calc

-- State the proof problem
theorem time_saved_correct :
  time_saved = 175 := by
  sorry

end time_saved_correct_l224_224945


namespace trajectory_sum_of_distances_to_axes_l224_224959

theorem trajectory_sum_of_distances_to_axes (x y : ℝ) (h : |x| + |y| = 6) :
  |x| + |y| = 6 := 
by 
  sorry

end trajectory_sum_of_distances_to_axes_l224_224959


namespace dig_site_date_l224_224987

theorem dig_site_date (S F T Fourth : ℤ) 
  (h₁ : F = S - 352)
  (h₂ : T = F + 3700)
  (h₃ : Fourth = 2 * T)
  (h₄ : Fourth = 8400) : S = 852 := 
by 
  sorry

end dig_site_date_l224_224987


namespace nectar_water_percentage_l224_224628

-- Definitions as per conditions
def nectar_weight : ℝ := 1.2
def honey_weight : ℝ := 1
def honey_water_ratio : ℝ := 0.4

-- Final statement to prove
theorem nectar_water_percentage : (honey_weight * honey_water_ratio + (nectar_weight - honey_weight)) / nectar_weight = 0.5 := by
  sorry

end nectar_water_percentage_l224_224628


namespace strictly_increasing_interval_l224_224792

def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem strictly_increasing_interval : { x : ℝ | -1 < x ∧ x < 1 } = { x : ℝ | -3 * (x + 1) * (x - 1) > 0 } :=
sorry

end strictly_increasing_interval_l224_224792


namespace at_least_502_friendly_numbers_l224_224008

def friendly (a : ℤ) : Prop :=
  ∃ (m n : ℤ), m > 0 ∧ n > 0 ∧ (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem at_least_502_friendly_numbers :
  ∃ S : Finset ℤ, (∀ a ∈ S, friendly a) ∧ 502 ≤ S.card ∧ ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2012 :=
by
  sorry

end at_least_502_friendly_numbers_l224_224008


namespace degree_of_g_l224_224202

open Polynomial

theorem degree_of_g (f g : Polynomial ℂ) (h1 : f = -3 * X^5 + 4 * X^4 - X^2 + C 2) (h2 : degree (f + g) = 2) : degree g = 5 :=
sorry

end degree_of_g_l224_224202


namespace range_of_m_eq_l224_224205

theorem range_of_m_eq (m: ℝ) (x: ℝ) :
  (m+1 = 0 ∧ 4 > 0) ∨ 
  ((m + 1 > 0) ∧ ((m^2 - 2 * m - 3)^2 - 4 * (m + 1) * (-m + 3) < 0)) ↔ 
  (m ∈ Set.Icc (-1 : ℝ) 1 ∪ Set.Ico (1 : ℝ) 3) := 
sorry

end range_of_m_eq_l224_224205


namespace rectangle_to_square_l224_224980

theorem rectangle_to_square (length width : ℕ) (h1 : 2 * (length + width) = 40) (h2 : length - 8 = width + 2) :
  width + 2 = 7 :=
by {
  -- Proof goes here
  sorry
}

end rectangle_to_square_l224_224980


namespace fraction_of_phone_numbers_l224_224423

-- Define the total number of valid 7-digit phone numbers
def totalValidPhoneNumbers : Nat := 7 * 10^6

-- Define the number of valid phone numbers that begin with 3 and end with 5
def validPhoneNumbersBeginWith3EndWith5 : Nat := 10^5

-- Prove the fraction of phone numbers that begin with 3 and end with 5 is 1/70
theorem fraction_of_phone_numbers (h : validPhoneNumbersBeginWith3EndWith5 = 10^5) 
(h2 : totalValidPhoneNumbers = 7 * 10^6) : 
validPhoneNumbersBeginWith3EndWith5 / totalValidPhoneNumbers = 1 / 70 := 
sorry

end fraction_of_phone_numbers_l224_224423


namespace pythagorean_triples_l224_224431

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triples :
  is_pythagorean_triple 3 4 5 ∧ is_pythagorean_triple 6 8 10 :=
by
  sorry

end pythagorean_triples_l224_224431


namespace evaluate_expression_l224_224772

noncomputable def ln (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression : 
  2017 ^ ln (ln 2017) - (ln 2017) ^ ln 2017 = 0 :=
by
  sorry

end evaluate_expression_l224_224772


namespace OilBillJanuary_l224_224052

theorem OilBillJanuary (J F : ℝ) (h1 : F / J = 5 / 4) (h2 : (F + 30) / J = 3 / 2) : J = 120 := by
  sorry

end OilBillJanuary_l224_224052


namespace sum_of_a_b_c_d_e_l224_224895

theorem sum_of_a_b_c_d_e (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) 
  (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) : a + b + c + d + e = 33 := by
  sorry

end sum_of_a_b_c_d_e_l224_224895


namespace z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l224_224753

def is_real (z : ℂ) := z.im = 0
def is_complex (z : ℂ) := z.im ≠ 0
def is_pure_imaginary (z : ℂ) := z.re = 0 ∧ z.im ≠ 0

def z (m : ℝ) : ℂ := ⟨m - 3, m^2 - 2 * m - 15⟩

theorem z_is_real_iff (m : ℝ) : is_real (z m) ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_is_complex_iff (m : ℝ) : is_complex (z m) ↔ m ≠ -3 ∧ m ≠ 5 :=
by sorry

theorem z_is_pure_imaginary_iff (m : ℝ) : is_pure_imaginary (z m) ↔ m = 3 :=
by sorry

end z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l224_224753


namespace final_result_is_110_l224_224938

theorem final_result_is_110 (x : ℕ) (h1 : x = 155) : (x * 2 - 200) = 110 :=
by
  -- placeholder for the solution proof
  sorry

end final_result_is_110_l224_224938


namespace find_n_l224_224288

theorem find_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28) : n = 27 :=
sorry

end find_n_l224_224288


namespace max_ab_condition_l224_224714

theorem max_ab_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 4 = 0)
  (line_check : ∀ x y : ℝ, (x = 1 ∧ y = -2) → 2*a*x - b*y - 2 = 0) : ab ≤ 1/4 :=
by
  sorry

end max_ab_condition_l224_224714


namespace maria_needs_more_cartons_l224_224006

theorem maria_needs_more_cartons
  (total_needed : ℕ)
  (strawberries : ℕ)
  (blueberries : ℕ)
  (already_has : ℕ)
  (more_needed : ℕ)
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : blueberries = 8)
  (h4 : already_has = strawberries + blueberries)
  (h5 : more_needed = total_needed - already_has) :
  more_needed = 9 :=
by sorry

end maria_needs_more_cartons_l224_224006


namespace julia_height_is_172_7_cm_l224_224789

def julia_height_in_cm (height_in_inches : ℝ) (conversion_factor : ℝ) : ℝ :=
  height_in_inches * conversion_factor

theorem julia_height_is_172_7_cm :
  julia_height_in_cm 68 2.54 = 172.7 :=
by
  sorry

end julia_height_is_172_7_cm_l224_224789


namespace total_jury_duty_days_l224_224022

-- Conditions
def jury_selection_days : ℕ := 2
def trial_multiplier : ℕ := 4
def evidence_review_hours : ℕ := 2
def lunch_hours : ℕ := 1
def trial_session_hours : ℕ := 6
def hours_per_day : ℕ := evidence_review_hours + lunch_hours + trial_session_hours
def deliberation_hours_per_day : ℕ := 14 - 2

def deliberation_first_defendant_days : ℕ := 6
def deliberation_second_defendant_days : ℕ := 4
def deliberation_third_defendant_days : ℕ := 5

def deliberation_first_defendant_total_hours : ℕ := deliberation_first_defendant_days * deliberation_hours_per_day
def deliberation_second_defendant_total_hours : ℕ := deliberation_second_defendant_days * deliberation_hours_per_day
def deliberation_third_defendant_total_hours : ℕ := deliberation_third_defendant_days * deliberation_hours_per_day

def deliberation_days_conversion (total_hours: ℕ) : ℕ := (total_hours + deliberation_hours_per_day - 1) / deliberation_hours_per_day

-- Total days spent
def total_days_spent : ℕ :=
  let trial_days := jury_selection_days * trial_multiplier
  let deliberation_days := deliberation_days_conversion deliberation_first_defendant_total_hours + deliberation_days_conversion deliberation_second_defendant_total_hours + deliberation_days_conversion deliberation_third_defendant_total_hours
  jury_selection_days + trial_days + deliberation_days

#eval total_days_spent -- Expected: 25

theorem total_jury_duty_days : total_days_spent = 25 := by
  sorry

end total_jury_duty_days_l224_224022


namespace graph_single_point_l224_224962

theorem graph_single_point (c : ℝ) : 
  (∃ x y : ℝ, ∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + 16 * x' - 6 * y' + c = 0 → (x' = x ∧ y' = y)) → c = 7 := 
by
  sorry

end graph_single_point_l224_224962


namespace kerosene_cost_l224_224072

/-- A dozen eggs cost as much as a pound of rice, a half-liter of kerosene costs as much as 8 eggs,
and each pound of rice costs $0.33. Prove that a liter of kerosene costs 44 cents. -/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost := half_liter_kerosene_cost * 2
  liter_kerosene_cost * 100 = 44 := 
by
  sorry

end kerosene_cost_l224_224072


namespace original_weight_of_beef_l224_224754

theorem original_weight_of_beef (w_after : ℝ) (loss_percentage : ℝ) (w_before : ℝ) : 
  (w_after = 550) → (loss_percentage = 0.35) → (w_after = 550) → (w_before = 846.15) :=
by
  intros
  sorry

end original_weight_of_beef_l224_224754


namespace tom_paid_correct_amount_l224_224268

-- Define the conditions given in the problem
def kg_apples : ℕ := 8
def rate_apples : ℕ := 70
def kg_mangoes : ℕ := 9
def rate_mangoes : ℕ := 45

-- Define the cost calculations
def cost_apples : ℕ := kg_apples * rate_apples
def cost_mangoes : ℕ := kg_mangoes * rate_mangoes
def total_amount : ℕ := cost_apples + cost_mangoes

-- The proof problem statement
theorem tom_paid_correct_amount : total_amount = 965 :=
by
  -- The proof steps are omitted and replaced with sorry
  sorry

end tom_paid_correct_amount_l224_224268


namespace arithmetic_sequence_ratio_l224_224914

theorem arithmetic_sequence_ratio (x y a₁ a₂ a₃ b₁ b₂ b₃ b₄ : ℝ) (h₁ : x ≠ y)
    (h₂ : a₁ = x + d) (h₃ : a₂ = x + 2 * d) (h₄ : a₃ = x + 3 * d) (h₅ : y = x + 4 * d)
    (h₆ : b₁ = x - d') (h₇ : b₂ = x + d') (h₈ : b₃ = x + 2 * d') (h₉ : y = x + 3 * d') (h₁₀ : b₄ = x + 4 * d') :
    (b₄ - b₃) / (a₂ - a₁) = 8 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l224_224914


namespace initial_percentage_filled_l224_224382

theorem initial_percentage_filled {P : ℝ} 
  (h1 : 45 + (P / 100) * 100 = (3 / 4) * 100) : 
  P = 30 := by
  sorry

end initial_percentage_filled_l224_224382


namespace joe_left_pocket_initial_l224_224505

-- Definitions from conditions
def total_money : ℕ := 200
def initial_left_pocket (L : ℕ) : ℕ := L
def initial_right_pocket (R : ℕ) : ℕ := R
def transfer_one_fourth (L : ℕ) : ℕ := L - L / 4
def add_to_right (R : ℕ) (L : ℕ) : ℕ := R + L / 4
def transfer_20 (L : ℕ) : ℕ := transfer_one_fourth L - 20
def add_20_to_right (R : ℕ) (L : ℕ) : ℕ := add_to_right R L + 20

-- Statement to prove
theorem joe_left_pocket_initial (L R : ℕ) (h₁ : L + R = total_money) 
  (h₂ : transfer_20 L = add_20_to_right R L) : 
  initial_left_pocket L = 160 :=
by
  sorry

end joe_left_pocket_initial_l224_224505


namespace initial_discount_l224_224931

theorem initial_discount (P D : ℝ) 
  (h1 : P - 71.4 = 5.25)
  (h2 : P * (1 - D) * 1.25 = 71.4) : 
  D = 0.255 :=
by {
  sorry
}

end initial_discount_l224_224931


namespace rectangle_width_is_16_l224_224311

-- Definitions based on the conditions
def length : ℝ := 24
def ratio := 6 / 5
def perimeter := 80

-- The proposition to prove
theorem rectangle_width_is_16 (W : ℝ) (h1 : length = 24) (h2 : length = ratio * W) (h3 : 2 * length + 2 * W = perimeter) :
  W = 16 :=
by
  sorry

end rectangle_width_is_16_l224_224311


namespace monica_total_savings_l224_224968

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l224_224968


namespace g_at_neg2_eq_8_l224_224525

-- Define the functions f and g
def f (x : ℤ) : ℤ := 4 * x - 6
def g (y : ℤ) : ℤ := 3 * (y + 6/4)^2 + 4 * (y + 6/4) + 1

-- Statement of the math proof problem:
theorem g_at_neg2_eq_8 : g (-2) = 8 := 
by 
  sorry

end g_at_neg2_eq_8_l224_224525


namespace min_points_to_guarantee_victory_l224_224348

noncomputable def points_distribution (pos : ℕ) : ℕ :=
  match pos with
  | 1 => 7
  | 2 => 4
  | 3 => 2
  | _ => 0

def max_points_per_race : ℕ := 7
def num_races : ℕ := 3

theorem min_points_to_guarantee_victory : ∃ min_points, min_points = 18 ∧ 
  (∀ other_points, other_points < 18) := 
by {
  sorry
}

end min_points_to_guarantee_victory_l224_224348


namespace hamburger_per_meatball_l224_224812

theorem hamburger_per_meatball (family_members : ℕ) (total_hamburger : ℕ) (antonio_meatballs : ℕ) 
    (hmembers : family_members = 8)
    (hhamburger : total_hamburger = 4)
    (hantonio : antonio_meatballs = 4) : 
    (total_hamburger : ℝ) / (family_members * antonio_meatballs) = 0.125 := 
by
  sorry

end hamburger_per_meatball_l224_224812


namespace karen_tests_graded_l224_224502

theorem karen_tests_graded (n : ℕ) (T : ℕ) 
  (avg_score_70 : T = 70 * n)
  (combined_score_290 : T + 290 = 85 * (n + 2)) : 
  n = 8 := 
sorry

end karen_tests_graded_l224_224502


namespace average_rate_decrease_price_reduction_l224_224940

-- Define the initial and final factory prices
def initial_price : ℝ := 200
def final_price : ℝ := 162

-- Define the function representing the average rate of decrease
def average_rate_of_decrease (x : ℝ) : Prop :=
  initial_price * (1 - x) * (1 - x) = final_price

-- Theorem stating the average rate of decrease (proving x = 0.1)
theorem average_rate_decrease : ∃ x : ℝ, average_rate_of_decrease x ∧ x = 0.1 :=
by
  use 0.1
  sorry

-- Define the selling price without reduction, sold without reduction, increase in pieces sold, and profit
def selling_price : ℝ := 200
def sold_without_reduction : ℕ := 20
def increase_pcs_per_5yuan_reduction : ℕ := 10
def profit : ℝ := 1150

-- Define the function representing the price reduction determination
def price_reduction_correct (m : ℝ) : Prop :=
  (38 - m) * (sold_without_reduction + 2 * m / 5) = profit

-- Theorem stating the price reduction (proving m = 15)
theorem price_reduction : ∃ m : ℝ, price_reduction_correct m ∧ m = 15 :=
by
  use 15
  sorry

end average_rate_decrease_price_reduction_l224_224940


namespace caterer_cheapest_option_l224_224192

theorem caterer_cheapest_option :
  ∃ x : ℕ, x ≥ 42 ∧ (∀ y : ℕ, y ≥ x → (20 * y < 120 + 18 * y) ∧ (20 * y < 250 + 14 * y)) := 
by
  sorry

end caterer_cheapest_option_l224_224192


namespace sum_of_reciprocals_l224_224135

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) : 
  (1 / x) + (1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l224_224135


namespace train_cross_first_platform_in_15_seconds_l224_224763

noncomputable def length_of_train : ℝ := 100
noncomputable def length_of_second_platform : ℝ := 500
noncomputable def time_to_cross_second_platform : ℝ := 20
noncomputable def length_of_first_platform : ℝ := 350
noncomputable def speed_of_train := (length_of_train + length_of_second_platform) / time_to_cross_second_platform
noncomputable def time_to_cross_first_platform := (length_of_train + length_of_first_platform) / speed_of_train

theorem train_cross_first_platform_in_15_seconds : time_to_cross_first_platform = 15 := by
  sorry

end train_cross_first_platform_in_15_seconds_l224_224763


namespace largest_digit_divisible_by_6_l224_224410

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N = 8 ∧ (45670 + N) % 6 = 0 :=
sorry

end largest_digit_divisible_by_6_l224_224410


namespace remainder_of_37_div_8_is_5_l224_224821

theorem remainder_of_37_div_8_is_5 : ∃ A B : ℤ, 37 = 8 * A + B ∧ 0 ≤ B ∧ B < 8 ∧ B = 5 := 
by
  sorry

end remainder_of_37_div_8_is_5_l224_224821


namespace total_cost_price_of_items_l224_224327

/-- 
  Definition of the selling prices of the items A, B, and C.
  Definition of the profit percentages of the items A, B, and C.
  The statement is the total cost price calculation.
-/
def ItemA_SP : ℝ := 800
def ItemA_Profit : ℝ := 0.25
def ItemB_SP : ℝ := 1200
def ItemB_Profit : ℝ := 0.20
def ItemC_SP : ℝ := 1500
def ItemC_Profit : ℝ := 0.30

theorem total_cost_price_of_items :
  let CP_A := ItemA_SP / (1 + ItemA_Profit)
  let CP_B := ItemB_SP / (1 + ItemB_Profit)
  let CP_C := ItemC_SP / (1 + ItemC_Profit)
  CP_A + CP_B + CP_C = 2793.85 :=
by
  sorry

end total_cost_price_of_items_l224_224327


namespace integer_remainder_18_l224_224685

theorem integer_remainder_18 (n : ℤ) (h : n ∈ ({14, 15, 16, 17, 18} : Set ℤ)) : n % 7 = 4 :=
by
  sorry

end integer_remainder_18_l224_224685


namespace find_number_of_rabbits_l224_224853

def total_heads (R P : ℕ) : ℕ := R + P
def total_legs (R P : ℕ) : ℕ := 4 * R + 2 * P

theorem find_number_of_rabbits (R P : ℕ)
  (h1 : total_heads R P = 60)
  (h2 : total_legs R P = 192) :
  R = 36 := by
  sorry

end find_number_of_rabbits_l224_224853


namespace greatest_possible_sum_of_digits_l224_224649

theorem greatest_possible_sum_of_digits 
  (n : ℕ) (a b d : ℕ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_d : d ≠ 0)
  (h1 : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (d * ((10 ^ (3 * n1) - 1) / 9) - b * ((10 ^ n1 - 1) / 9) = a^3 * ((10^n1 - 1) / 9)^3) 
                      ∧ (d * ((10 ^ (3 * n2) - 1) / 9) - b * ((10 ^ n2 - 1) / 9) = a^3 * ((10^n2 - 1) / 9)^3)) : 
  a + b + d = 12 := 
sorry

end greatest_possible_sum_of_digits_l224_224649


namespace pacific_ocean_area_rounded_l224_224760

def pacific_ocean_area : ℕ := 19996800

def ten_thousand : ℕ := 10000

noncomputable def pacific_ocean_area_in_ten_thousands (area : ℕ) : ℕ :=
  (area / ten_thousand + if (area % ten_thousand) >= (ten_thousand / 2) then 1 else 0)

theorem pacific_ocean_area_rounded :
  pacific_ocean_area_in_ten_thousands pacific_ocean_area = 2000 :=
by
  sorry

end pacific_ocean_area_rounded_l224_224760


namespace part1a_part1b_part2_part3a_part3b_l224_224694

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 2

-- Prove f(2) = 1/3
theorem part1a : f 2 = 1 / 3 := 
by sorry

-- Prove g(2) = 6
theorem part1b : g 2 = 6 :=
by sorry

-- Prove f[g(2)] = 1/7 
theorem part2 : f (g 2) = 1 / 7 :=
by sorry

-- Prove f[g(x)] = 1/(x^2 + 3) 
theorem part3a : ∀ x : ℝ, f (g x) = 1 / (x^2 + 3) :=
by sorry

-- Prove g[f(x)] = 1/((1 + x)^2) + 2 
theorem part3b : ∀ x : ℝ, g (f x) = 1 / (1 + x)^2 + 2 :=
by sorry

end part1a_part1b_part2_part3a_part3b_l224_224694


namespace base3_to_base5_conversion_l224_224378

-- Define the conversion from base 3 to decimal
def base3_to_decimal (n : ℕ) : ℕ :=
  n % 10 * 1 + (n / 10 % 10) * 3 + (n / 100 % 10) * 9 + (n / 1000 % 10) * 27 + (n / 10000 % 10) * 81

-- Define the conversion from decimal to base 5
def decimal_to_base5 (n : ℕ) : ℕ :=
  n % 5 + (n / 5 % 5) * 10 + (n / 25 % 5) * 100

-- The initial number in base 3
def initial_number_base3 : ℕ := 10121

-- The final number in base 5
def final_number_base5 : ℕ := 342

-- The theorem that states the conversion result
theorem base3_to_base5_conversion :
  decimal_to_base5 (base3_to_decimal initial_number_base3) = final_number_base5 :=
by
  sorry

end base3_to_base5_conversion_l224_224378


namespace total_books_in_week_l224_224820

def books_read (n : ℕ) : ℕ :=
  if n = 0 then 2 -- day 1 (indexed by 0)
  else if n = 1 then 2 -- day 2
  else 2 + n -- starting from day 3 (indexed by 2)

-- Summing the books read from day 1 to day 7 (indexed from 0 to 6)
theorem total_books_in_week : (List.sum (List.map books_read [0, 1, 2, 3, 4, 5, 6])) = 29 := by
  sorry

end total_books_in_week_l224_224820


namespace range_of_x_l224_224344

theorem range_of_x (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
by {
  sorry
}

end range_of_x_l224_224344


namespace largest_integer_same_cost_l224_224605

def cost_base_10 (n : ℕ) : ℕ :=
  (n.digits 10).sum

def cost_base_2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem largest_integer_same_cost : ∃ n < 1000, 
  cost_base_10 n = cost_base_2 n ∧
  ∀ m < 1000, cost_base_10 m = cost_base_2 m → n ≥ m :=
sorry

end largest_integer_same_cost_l224_224605


namespace neighbors_receive_mangoes_l224_224791

-- Definitions of the conditions
def harvested_mangoes : ℕ := 560
def sold_mangoes : ℕ := harvested_mangoes / 2
def given_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Calculation of mangoes left
def mangoes_left : ℕ := harvested_mangoes - sold_mangoes - given_to_family

-- The statement we want to prove
theorem neighbors_receive_mangoes : mangoes_left / num_neighbors = 19 := by
  sorry

end neighbors_receive_mangoes_l224_224791


namespace perp_vec_m_l224_224432

theorem perp_vec_m (m : ℝ) : (1 : ℝ) * (-1 : ℝ) + 2 * m = 0 → m = 1 / 2 :=
by 
  intro h
  -- Translate the given condition directly
  sorry

end perp_vec_m_l224_224432


namespace given_expression_simplifies_to_l224_224216

-- Given conditions: a ≠ ±1, a ≠ 0, b ≠ -1, b ≠ 0
variable (a b : ℝ)
variable (ha1 : a ≠ 1)
variable (ha2 : a ≠ -1)
variable (ha3 : a ≠ 0)
variable (hb1 : b ≠ 0)
variable (hb2 : b ≠ -1)

theorem given_expression_simplifies_to (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : b ≠ -1) :
    (a * b^(2/3) - b^(2/3) - a + 1) / ((1 - a^(1/3)) * ((a^(1/3) + 1)^2 - a^(1/3)) * (b^(1/3) + 1))
  + (a * b)^(1/3) * (1/a^(1/3) + 1/b^(1/3)) = 1 + a^(1/3) := by
  sorry

end given_expression_simplifies_to_l224_224216


namespace exponentiation_problem_l224_224654

theorem exponentiation_problem : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by
  sorry

end exponentiation_problem_l224_224654


namespace sum_of_fractions_l224_224606

theorem sum_of_fractions :
  (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 :=
by
  sorry

end sum_of_fractions_l224_224606


namespace age_difference_l224_224403

variable (P M Mo N : ℚ)

-- Given conditions as per problem statement
axiom ratio_P_M : (P / M) = 3 / 5
axiom ratio_M_Mo : (M / Mo) = 3 / 4
axiom ratio_Mo_N : (Mo / N) = 5 / 7
axiom sum_ages : P + M + Mo + N = 228

-- Statement to prove
theorem age_difference (ratio_P_M : (P / M) = 3 / 5)
                        (ratio_M_Mo : (M / Mo) = 3 / 4)
                        (ratio_Mo_N : (Mo / N) = 5 / 7)
                        (sum_ages : P + M + Mo + N = 228) :
  N - P = 69.5 := 
sorry

end age_difference_l224_224403


namespace number_of_birds_is_122_l224_224391

-- Defining the variables
variables (b m i : ℕ)

-- Define the conditions as part of an axiom
axiom heads_count : b + m + i = 300
axiom legs_count : 2 * b + 4 * m + 6 * i = 1112

-- We aim to prove the number of birds is 122
theorem number_of_birds_is_122 (h1 : b + m + i = 300) (h2 : 2 * b + 4 * m + 6 * i = 1112) : b = 122 := by
  sorry

end number_of_birds_is_122_l224_224391


namespace car_round_trip_time_l224_224000

theorem car_round_trip_time
  (d_AB : ℝ) (v_AB_downhill : ℝ) (v_BA_uphill : ℝ)
  (h_d_AB : d_AB = 75.6)
  (h_v_AB_downhill : v_AB_downhill = 33.6)
  (h_v_BA_uphill : v_BA_uphill = 25.2) :
  d_AB / v_AB_downhill + d_AB / v_BA_uphill = 5.25 := by
  sorry

end car_round_trip_time_l224_224000


namespace total_trip_hours_l224_224412

-- Define the given conditions
def speed1 := 50 -- Speed in mph for the first 4 hours
def time1 := 4 -- First 4 hours
def distance1 := speed1 * time1 -- Distance covered in the first 4 hours

def speed2 := 80 -- Speed in mph for additional hours
def average_speed := 65 -- Average speed for the entire trip

-- Define the proof problem
theorem total_trip_hours (T : ℕ) (A : ℕ) :
  distance1 + (speed2 * A) = average_speed * T ∧ T = time1 + A → T = 8 :=
by
  sorry

end total_trip_hours_l224_224412


namespace walk_usual_time_l224_224635

theorem walk_usual_time (T : ℝ) (S : ℝ) (h1 : (5 / 4 : ℝ) = (T + 10) / T) : T = 40 :=
sorry

end walk_usual_time_l224_224635


namespace find_angle_D_l224_224116

-- Define the given angles and conditions
def angleA := 30
def angleB (D : ℝ) := 2 * D
def angleC (D : ℝ) := D + 40
def sum_of_angles (A B C D : ℝ) := A + B + C + D = 360

theorem find_angle_D (D : ℝ) (hA : angleA = 30) (hB : angleB D = 2 * D) (hC : angleC D = D + 40) (hSum : sum_of_angles angleA (angleB D) (angleC D) D):
  D = 72.5 :=
by
  -- Proof is omitted
  sorry

end find_angle_D_l224_224116


namespace visiting_plans_count_l224_224035

-- Let's define the exhibitions
inductive Exhibition
| OperaCultureExhibition
| MingDynastyImperialCellarPorcelainExhibition
| AncientGreenLandscapePaintingExhibition
| ZhaoMengfuCalligraphyAndPaintingExhibition

open Exhibition

-- The condition is that the student must visit at least one painting exhibition in the morning and another in the afternoon
-- Proof that the number of different visiting plans is 10.
theorem visiting_plans_count :
  let exhibitions := [OperaCultureExhibition, MingDynastyImperialCellarPorcelainExhibition, AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  let painting_exhibitions := [AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  ∃ visits : List (Exhibition × Exhibition), (∀ (m a : Exhibition), (m ∈ painting_exhibitions ∨ a ∈ painting_exhibitions)) → visits.length = 10 :=
sorry

end visiting_plans_count_l224_224035


namespace trajectory_moving_point_hyperbola_l224_224494

theorem trajectory_moving_point_hyperbola {n m : ℝ} (h_neg_n : n < 0) :
    (∃ y < 0, (y^2 = 16) ∧ (m^2 = (n^2 / 4 - 4))) ↔ ( ∃ (y : ℝ), (y^2 / 16) - (m^2 / 4) = 1 ∧ y < 0 ) := 
sorry

end trajectory_moving_point_hyperbola_l224_224494


namespace center_of_circle_l224_224762

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

-- Define what it means to be the center of the circle, which is (h, k)
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 1

-- The statement that we need to prove
theorem center_of_circle : is_center 1 3 :=
sorry

end center_of_circle_l224_224762


namespace triathlon_bike_speed_l224_224830

theorem triathlon_bike_speed :
  ∀ (t_total t_swim t_run t_bike : ℚ) (d_swim d_run d_bike : ℚ)
    (v_swim v_run r_bike : ℚ),
  t_total = 3 →
  d_swim = 1 / 2 →
  v_swim = 1 →
  d_run = 4 →
  v_run = 5 →
  d_bike = 10 →
  t_swim = d_swim / v_swim →
  t_run = d_run / v_run →
  t_bike = t_total - (t_swim + t_run) →
  r_bike = d_bike / t_bike →
  r_bike = 100 / 17 :=
by
  intros t_total t_swim t_run t_bike d_swim d_run d_bike v_swim v_run r_bike
         h_total h_d_swim h_v_swim h_d_run h_v_run h_d_bike h_t_swim h_t_run h_t_bike h_r_bike
  sorry

end triathlon_bike_speed_l224_224830


namespace area_of_square_A_l224_224910

noncomputable def square_areas (a b : ℕ) : Prop :=
  (b ^ 2 = 81) ∧ (a = b + 4)

theorem area_of_square_A : ∃ a b : ℕ, square_areas a b → a ^ 2 = 169 :=
by
  sorry

end area_of_square_A_l224_224910


namespace function_passes_through_fixed_point_l224_224375

noncomputable def given_function (a : ℝ) (x : ℝ) : ℝ :=
  a^(x - 1) + 7

theorem function_passes_through_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  given_function a 1 = 8 :=
by
  sorry

end function_passes_through_fixed_point_l224_224375


namespace triangle_bisector_ratio_l224_224870

theorem triangle_bisector_ratio (AB BC CA : ℝ) (h_AB_pos : 0 < AB) (h_BC_pos : 0 < BC) (h_CA_pos : 0 < CA)
  (AA1_bisector : True) (BB1_bisector : True) (O_intersection : True) : 
  AA1 / OA1 = 3 :=
by
  sorry

end triangle_bisector_ratio_l224_224870


namespace no_such_quadratic_exists_l224_224032

theorem no_such_quadratic_exists : ¬ ∃ (b c : ℝ), 
  (∀ x : ℝ, 6 * x ≤ 3 * x^2 + 3 ∧ 3 * x^2 + 3 ≤ x^2 + b * x + c) ∧
  (x^2 + b * x + c = 1) :=
by
  sorry

end no_such_quadratic_exists_l224_224032


namespace van_helsing_removed_percentage_l224_224678

theorem van_helsing_removed_percentage :
  ∀ (V W : ℕ), 
  (5 * V / 2 + 10 * 8 = 105) →
  (W = 4 * V) →
  8 / W * 100 = 20 := 
by
  sorry

end van_helsing_removed_percentage_l224_224678


namespace permutation_sum_l224_224002

theorem permutation_sum (n : ℕ) (h1 : n + 3 ≤ 2 * n) (h2 : n + 1 ≤ 4) (h3 : n > 0) :
  Nat.factorial (2 * n) / Nat.factorial (2 * n - (n + 3)) + Nat.factorial 4 / Nat.factorial (4 - (n + 1)) = 744 :=
by
  sorry

end permutation_sum_l224_224002


namespace polygon_sides_l224_224803

open Real

theorem polygon_sides (n : ℕ) : 
  (∀ (angle : ℝ), angle = 40 → n * angle = 360) → n = 9 := by
  intro h
  have h₁ := h 40 rfl
  sorry

end polygon_sides_l224_224803


namespace binary_to_decimal_l224_224586

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l224_224586


namespace density_of_second_part_l224_224376

theorem density_of_second_part (ρ₁ : ℝ) (V₁ V : ℝ) (m₁ m : ℝ) (h₁ : ρ₁ = 2700) (h₂ : V₁ = 0.25 * V) (h₃ : m₁ = 0.4 * m) :
  (0.6 * m) / (0.75 * V) = 2160 :=
by
  --- Proof omitted
  sorry

end density_of_second_part_l224_224376


namespace boat_speed_in_still_water_l224_224665

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 6) (h2 : B - S = 4) : B = 5 := by
  sorry

end boat_speed_in_still_water_l224_224665


namespace julias_preferred_number_l224_224687

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem julias_preferred_number : ∃ n : ℕ, n > 100 ∧ n < 200 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 104 :=
by
  sorry

end julias_preferred_number_l224_224687


namespace expected_number_of_digits_is_1_55_l224_224083

/-- Brent rolls a fair icosahedral die with numbers 1 through 20 on its faces -/
noncomputable def expectedNumberOfDigits : ℚ :=
  let P_one_digit := 9 / 20
  let P_two_digit := 11 / 20
  (P_one_digit * 1) + (P_two_digit * 2)

/-- The expected number of digits Brent will roll is 1.55 -/
theorem expected_number_of_digits_is_1_55 : expectedNumberOfDigits = 1.55 := by
  sorry

end expected_number_of_digits_is_1_55_l224_224083


namespace class_size_l224_224599

theorem class_size (n : ℕ) (h1 : 85 - 33 + 90 - 40 = 102) (h2 : (102 : ℚ) / n = 1.5): n = 68 :=
by
  sorry

end class_size_l224_224599


namespace polynomial_solution_l224_224383

noncomputable def roots (a b c : ℤ) : Set ℝ :=
  { x : ℝ | a * x ^ 2 + b * x + c = 0 }

theorem polynomial_solution :
  let x1 := (1 + Real.sqrt 13) / 2
  let x2 := (1 - Real.sqrt 13) / 2
  x1 ∈ roots 1 (-1) (-3) → x2 ∈ roots 1 (-1) (-3) →
  ((x1^5 - 20) * (3*x2^4 - 2*x2 - 35) = -1063) :=
by
  sorry

end polynomial_solution_l224_224383


namespace ratio_of_smaller_to_bigger_l224_224793

theorem ratio_of_smaller_to_bigger (S B : ℕ) (h_bigger: B = 104) (h_sum: S + B = 143) :
  S / B = 39 / 104 := sorry

end ratio_of_smaller_to_bigger_l224_224793


namespace digits_are_different_probability_l224_224985

noncomputable def prob_diff_digits : ℚ :=
  let total := 999 - 100 + 1
  let same_digits := 9
  1 - (same_digits / total)

theorem digits_are_different_probability :
  prob_diff_digits = 99 / 100 :=
by
  sorry

end digits_are_different_probability_l224_224985


namespace total_cows_l224_224509

theorem total_cows (Matthews Aaron Tyron Marovich : ℕ) 
  (h1 : Matthews = 60)
  (h2 : Aaron = 4 * Matthews)
  (h3 : Tyron = Matthews - 20)
  (h4 : Aaron + Matthews + Tyron = Marovich + 30) :
  Aaron + Matthews + Tyron + Marovich = 650 :=
by
  sorry

end total_cows_l224_224509


namespace henry_games_given_l224_224701

theorem henry_games_given (G : ℕ) (henry_initial : ℕ) (neil_initial : ℕ) (henry_now : ℕ) (neil_now : ℕ) :
  henry_initial = 58 →
  neil_initial = 7 →
  henry_now = henry_initial - G →
  neil_now = neil_initial + G →
  henry_now = 4 * neil_now →
  G = 6 :=
by
  intros h_initial n_initial h_now n_now eq_henry
  sorry

end henry_games_given_l224_224701


namespace integrate_diff_eq_l224_224457

noncomputable def particular_solution (x y : ℝ) : Prop :=
  (y^2 - x^2) / 2 + Real.exp y - Real.log ((x + Real.sqrt (1 + x^2)) / (2 + Real.sqrt 5)) = Real.exp 1 - 3 / 2

theorem integrate_diff_eq (x y : ℝ) :
  (∀ x y : ℝ, y' = (x * Real.sqrt (1 + x^2) + 1) / (Real.sqrt (1 + x^2) * (y + Real.exp y))) → 
  (∃ x0 y0 : ℝ, x0 = 2 ∧ y0 = 1) → 
  particular_solution x y :=
sorry

end integrate_diff_eq_l224_224457


namespace add_fractions_l224_224608

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = 5 / 8 :=
by
  sorry

end add_fractions_l224_224608


namespace smallest_non_representable_number_l224_224057

theorem smallest_non_representable_number :
  ∀ n : ℕ, (∀ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d) → n < 11) ∧
           (∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d)) :=
sorry

end smallest_non_representable_number_l224_224057


namespace sum_of_roots_of_quadratic_l224_224709

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, (x - 1) * (x + 4) = 18 -> (∃ a b c : ℝ, a = 1 ∧ b = 3 ∧ c = -22 ∧ ((a * x^2 + b * x + c = 0) ∧ (-b / a = -3))) :=
by
  sorry

end sum_of_roots_of_quadratic_l224_224709


namespace solution1_solution2_solution3_solution4_solution5_l224_224771

noncomputable def problem1 : ℤ :=
  -3 + 8 - 15 - 6

theorem solution1 : problem1 = -16 := by
  sorry

noncomputable def problem2 : ℚ :=
  -35 / -7 * (-1 / 7)

theorem solution2 : problem2 = -(5 / 7) := by
  sorry

noncomputable def problem3 : ℤ :=
  -2^2 - |2 - 5| / -3

theorem solution3 : problem3 = -3 := by
  sorry

noncomputable def problem4 : ℚ :=
  (1 / 2 + 5 / 6 - 7 / 12) * -24 

theorem solution4 : problem4 = -18 := by
  sorry

noncomputable def problem5 : ℚ :=
  (-99 - 6 / 11) * 22

theorem solution5 : problem5 = -2190 := by
  sorry

end solution1_solution2_solution3_solution4_solution5_l224_224771


namespace maximize_profit_l224_224674

-- Conditions
def price_bound (p : ℝ) := p ≤ 22
def books_sold (p : ℝ) := 110 - 4 * p
def profit (p : ℝ) := (p - 2) * books_sold p

-- The main theorem statement
theorem maximize_profit : ∃ p : ℝ, price_bound p ∧ profit p = profit 15 :=
sorry

end maximize_profit_l224_224674


namespace days_to_finish_together_l224_224832

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end days_to_finish_together_l224_224832


namespace cube_surface_area_726_l224_224130

noncomputable def cubeSurfaceArea (volume : ℝ) : ℝ :=
  let side := volume^(1 / 3)
  6 * (side ^ 2)

theorem cube_surface_area_726 (h : cubeSurfaceArea 1331 = 726) : cubeSurfaceArea 1331 = 726 :=
by
  sorry

end cube_surface_area_726_l224_224130


namespace calc_result_l224_224835

theorem calc_result : (377 / 13 / 29 * 1 / 4 / 2) = 0.125 := 
by sorry

end calc_result_l224_224835


namespace minimum_value_of_expression_l224_224769

noncomputable def min_value_expression (x y : ℝ) : ℝ := 
  (x + 1)^2 / (x + 2) + 3 / (x + 2) + y^2 / (y + 1)

theorem minimum_value_of_expression :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → x + y = 2 → min_value_expression x y = 14 / 5 :=
by
  sorry

end minimum_value_of_expression_l224_224769


namespace scientific_notation_43300000_l224_224715

theorem scientific_notation_43300000 : 43300000 = 4.33 * 10^7 :=
by
  sorry

end scientific_notation_43300000_l224_224715


namespace remainder_19_pow_19_plus_19_mod_20_l224_224481

theorem remainder_19_pow_19_plus_19_mod_20 : (19^19 + 19) % 20 = 18 := 
by
  sorry

end remainder_19_pow_19_plus_19_mod_20_l224_224481


namespace value_of_3W5_l224_224462

-- Define the operation W
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- State the theorem to prove
theorem value_of_3W5 : W 3 5 = 23 := by
    sorry

end value_of_3W5_l224_224462


namespace A_n_plus_B_n_eq_2n_cubed_l224_224874

-- Definition of A_n given the grouping of positive integers
def A_n (n : ℕ) : ℕ :=
  let sum_first_n_squared := n * n * (n * n + 1) / 2
  let sum_first_n_minus_1_squared := (n - 1) * (n - 1) * ((n - 1) * (n - 1) + 1) / 2
  sum_first_n_squared - sum_first_n_minus_1_squared

-- Definition of B_n given the array of cubes of natural numbers
def B_n (n : ℕ) : ℕ := n * n * n - (n - 1) * (n - 1) * (n - 1)

-- The theorem to prove that A_n + B_n = 2n^3
theorem A_n_plus_B_n_eq_2n_cubed (n : ℕ) : A_n n + B_n n = 2 * n^3 := by
  sorry

end A_n_plus_B_n_eq_2n_cubed_l224_224874


namespace servant_leaves_after_nine_months_l224_224883

-- Definitions based on conditions
def yearly_salary : ℕ := 90 + 90
def monthly_salary : ℕ := yearly_salary / 12
def amount_received : ℕ := 45 + 90

-- The theorem to prove
theorem servant_leaves_after_nine_months :
    amount_received / monthly_salary = 9 :=
by
  -- Using the provided conditions, we establish the equality we need.
  sorry

end servant_leaves_after_nine_months_l224_224883


namespace probability_both_A_and_B_selected_l224_224703

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l224_224703


namespace find_m_value_l224_224706

theorem find_m_value : 
  ∃ (m : ℝ), 
  (∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 1 ∧ (x - y + m = 0)) → m = -3 :=
by
  sorry

end find_m_value_l224_224706


namespace beef_cubes_per_slab_l224_224971

-- Define the conditions as variables
variables (kabob_sticks : ℕ) (cubes_per_stick : ℕ) (cost_per_slab : ℕ) (total_cost : ℕ) (total_kabob_sticks : ℕ)

-- Assume the conditions from step a)
theorem beef_cubes_per_slab 
  (h1 : cubes_per_stick = 4) 
  (h2 : cost_per_slab = 25) 
  (h3 : total_cost = 50) 
  (h4 : total_kabob_sticks = 40)
  : total_cost / cost_per_slab * (total_kabob_sticks * cubes_per_stick) / (total_cost / cost_per_slab) = 80 := 
by {
  -- the proof goes here
  sorry
}

end beef_cubes_per_slab_l224_224971


namespace ellipse_foci_y_axis_l224_224122

theorem ellipse_foci_y_axis (k : ℝ) :
  (∃ a b : ℝ, a = 15 - k ∧ b = k - 9 ∧ a > 0 ∧ b > 0) ↔ (12 < k ∧ k < 15) :=
by
  sorry

end ellipse_foci_y_axis_l224_224122


namespace find_abs_product_abc_l224_224184

theorem find_abs_product_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h : a + 1 / b = b + 1 / c ∧ b + 1 / c = c + 1 / a) : |a * b * c| = 1 :=
sorry

end find_abs_product_abc_l224_224184


namespace find_A_l224_224191

-- Define the condition as an axiom
axiom A : ℝ
axiom condition : A + 10 = 15 

-- Prove that given the condition, A must be 5
theorem find_A : A = 5 := 
by {
  sorry
}

end find_A_l224_224191


namespace equation_infinitely_many_solutions_iff_b_eq_neg9_l224_224775

theorem equation_infinitely_many_solutions_iff_b_eq_neg9 (b : ℤ) :
  (∀ x : ℤ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  sorry

end equation_infinitely_many_solutions_iff_b_eq_neg9_l224_224775


namespace solve_trig_eq_l224_224683

-- Define the equation
def equation (x : ℝ) : Prop := 3 * Real.sin x = 1 + Real.cos (2 * x)

-- Define the solution set
def solution_set (x : ℝ) : Prop := ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)

-- The proof problem statement
theorem solve_trig_eq {x : ℝ} : equation x ↔ solution_set x := sorry

end solve_trig_eq_l224_224683


namespace sum_of_smallest_ns_l224_224480

theorem sum_of_smallest_ns : ∀ n1 n2 : ℕ, (n1 ≡ 1 [MOD 4] ∧ n1 ≡ 2 [MOD 7]) ∧ (n2 ≡ 1 [MOD 4] ∧ n2 ≡ 2 [MOD 7]) ∧ n1 < n2 →
  n1 = 9 ∧ n2 = 37 → (n1 + n2 = 46) :=
by
  sorry

end sum_of_smallest_ns_l224_224480


namespace oak_trees_cut_down_l224_224996

-- Define the conditions
def initial_oak_trees : ℕ := 9
def final_oak_trees : ℕ := 7

-- Prove that the number of oak trees cut down is 2
theorem oak_trees_cut_down : (initial_oak_trees - final_oak_trees) = 2 :=
by
  -- Proof is omitted
  sorry

end oak_trees_cut_down_l224_224996


namespace log_expression_evaluation_l224_224332

theorem log_expression_evaluation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^4)) * (Real.log (x^7) / Real.log (y^3)) =
  (1 / 4) * (Real.log x / Real.log y) := 
by
  sorry

end log_expression_evaluation_l224_224332


namespace sport_formulation_water_l224_224017

theorem sport_formulation_water
  (f c w : ℕ)  -- flavoring, corn syrup, and water respectively in standard formulation
  (f_s c_s w_s : ℕ)  -- flavoring, corn syrup, and water respectively in sport formulation
  (corn_syrup_sport : ℤ) -- amount of corn syrup in sport formulation in ounces
  (h_std_ratio : f = 1 ∧ c = 12 ∧ w = 30) -- given standard formulation ratios
  (h_sport_fc_ratio : f_s * 4 = c_s) -- sport formulation flavoring to corn syrup ratio
  (h_sport_fw_ratio : f_s * 60 = w_s) -- sport formulation flavoring to water ratio
  (h_corn_syrup_sport : c_s = corn_syrup_sport) -- amount of corn syrup in sport formulation
  : w_s = 30 := 
by 
  sorry

end sport_formulation_water_l224_224017


namespace ratio_of_fallen_cakes_is_one_half_l224_224990

noncomputable def ratio_fallen_to_total (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ) :=
  fallen_cakes / total_cakes

theorem ratio_of_fallen_cakes_is_one_half :
  ∀ (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ),
    total_cakes = 12 →
    pick_up = fallen_cakes / 2 →
    pick_up = destroyed_cakes →
    destroyed_cakes = 3 →
    ratio_fallen_to_total total_cakes fallen_cakes pick_up destroyed_cakes = 1 / 2 :=
by
  intros total_cakes fallen_cakes pick_up destroyed_cakes h1 h2 h3 h4
  rw [h1, h4, ratio_fallen_to_total]
  -- proof goes here
  sorry

end ratio_of_fallen_cakes_is_one_half_l224_224990


namespace total_kids_got_in_equals_148_l224_224724

def total_kids : ℕ := 120 + 90 + 50

def denied_riverside : ℕ := (20 * 120) / 100
def denied_west_side : ℕ := (70 * 90) / 100
def denied_mountaintop : ℕ := 50 / 2

def got_in_riverside : ℕ := 120 - denied_riverside
def got_in_west_side : ℕ := 90 - denied_west_side
def got_in_mountaintop : ℕ := 50 - denied_mountaintop

def total_got_in : ℕ := got_in_riverside + got_in_west_side + got_in_mountaintop

theorem total_kids_got_in_equals_148 :
  total_got_in = 148 := 
by
  unfold total_got_in
  unfold got_in_riverside got_in_west_side got_in_mountaintop
  unfold denied_riverside denied_west_side denied_mountaintop
  sorry

end total_kids_got_in_equals_148_l224_224724


namespace proof_of_diagonals_and_angles_l224_224352

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def sum_of_internal_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem proof_of_diagonals_and_angles :
  let p_diagonals := number_of_diagonals 5
  let o_diagonals := number_of_diagonals 8
  let total_diagonals := p_diagonals + o_diagonals
  let p_internal_angles := sum_of_internal_angles 5
  let o_internal_angles := sum_of_internal_angles 8
  let total_internal_angles := p_internal_angles + o_internal_angles
  total_diagonals = 25 ∧ total_internal_angles = 1620 :=
by
  sorry

end proof_of_diagonals_and_angles_l224_224352


namespace no_boys_love_cards_l224_224799

def boys_love_marbles := 13
def total_marbles := 26
def marbles_per_boy := 2

theorem no_boys_love_cards (boys_love_marbles total_marbles marbles_per_boy : ℕ)
  (h1 : boys_love_marbles * marbles_per_boy = total_marbles) : 
  ∃ no_boys_love_cards : ℕ, no_boys_love_cards = 0 :=
by
  sorry

end no_boys_love_cards_l224_224799


namespace clea_escalator_time_standing_l224_224362

noncomputable def escalator_time (c : ℕ) : ℝ :=
  let s := (7 * c) / 5
  let d := 72 * c
  let t := d / s
  t

theorem clea_escalator_time_standing (c : ℕ) (h1 : 72 * c = 72 * c) (h2 : 30 * (c + (7 * c) / 5) = 72 * c): escalator_time c = 51 :=
by
  sorry

end clea_escalator_time_standing_l224_224362


namespace factors_of_180_multiple_of_15_count_l224_224564

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l224_224564


namespace solve_problem_l224_224551

theorem solve_problem : 
  ∃ p q : ℝ, 
    (p ≠ q) ∧ 
    ((∀ x : ℝ, (x = p ∨ x = q) ↔ (x-4)*(x+4) = 24*x - 96)) ∧ 
    (p > q) ∧ 
    (p - q = 16) :=
by
  sorry

end solve_problem_l224_224551


namespace find_matrix_A_l224_224529

-- Let A be a 2x2 matrix such that 
def A (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

theorem find_matrix_A :
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ,
  (A.mulVec ![4, 1] = ![8, 14]) ∧ (A.mulVec ![2, -3] = ![-2, 11]) ∧
  A = ![![2, 1/2], ![-1, -13/3]] :=
by
  sorry

end find_matrix_A_l224_224529


namespace find_distance_to_place_l224_224136

noncomputable def distance_to_place (speed_boat : ℝ) (speed_stream : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := speed_boat + speed_stream
  let upstream_speed := speed_boat - speed_stream
  let distance := (total_time * (downstream_speed * upstream_speed)) / (downstream_speed + upstream_speed)
  distance

theorem find_distance_to_place :
  distance_to_place 16 2 937.1428571428571 = 7392.92 :=
by
  sorry

end find_distance_to_place_l224_224136


namespace circle_in_fourth_quadrant_l224_224652

theorem circle_in_fourth_quadrant (a : ℝ) :
  (∃ (x y: ℝ), x^2 + y^2 - 2 * a * x + 4 * a * y + 6 * a^2 - a = 0 ∧ (a > 0) ∧ (-2 * y < 0)) → (0 < a ∧ a < 1) :=
by
  sorry

end circle_in_fourth_quadrant_l224_224652


namespace bob_speed_lt_40_l224_224913

theorem bob_speed_lt_40 (v_b v_a : ℝ) (h1 : v_a > 45) (h2 : 180 / v_a < 180 / v_b - 0.5) :
  v_b < 40 :=
by
  -- Variables and constants
  let distance := 180
  let min_speed_alice := 45
  -- Conditions
  have h_distance := distance
  have h_min_speed_alice := min_speed_alice
  have h_time_alice := (distance : ℝ) / v_a
  have h_time_bob := (distance : ℝ) / v_b
  -- Given conditions inequalities
  have ineq := h2
  have alice_min_speed := h1
  -- Now apply these facts and derived inequalities to prove bob_speed_lt_40
  sorry

end bob_speed_lt_40_l224_224913


namespace ratio_of_periods_l224_224100

variable (I_B T_B : ℝ)
variable (I_A T_A : ℝ)
variable (Profit_A Profit_B TotalProfit : ℝ)
variable (k : ℝ)

-- Define the conditions
axiom h1 : I_A = 3 * I_B
axiom h2 : T_A = k * T_B
axiom h3 : Profit_B = 4500
axiom h4 : TotalProfit = 31500
axiom h5 : Profit_A = TotalProfit - Profit_B

-- The profit shares are proportional to the product of investment and time period
axiom h6 : Profit_A = I_A * T_A
axiom h7 : Profit_B = I_B * T_B

theorem ratio_of_periods : T_A / T_B = 2 := by
  sorry

end ratio_of_periods_l224_224100


namespace arithmetic_seq_sum_equidistant_l224_224554

variable (a : ℕ → ℤ)

theorem arithmetic_seq_sum_equidistant :
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) → a 4 = 12 → a 1 + a 7 = 24 :=
by
  intros h_seq h_a4
  sorry

end arithmetic_seq_sum_equidistant_l224_224554


namespace min_beans_l224_224299

theorem min_beans (r b : ℕ) (H1 : r ≥ 3 + 2 * b) (H2 : r ≤ 3 * b) : b ≥ 3 := 
sorry

end min_beans_l224_224299


namespace relatively_prime_powers_of_two_l224_224386

theorem relatively_prime_powers_of_two (a : ℤ) (h₁ : a % 2 = 1) (n m : ℕ) (h₂ : n ≠ m) :
  Int.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 :=
by
  sorry

end relatively_prime_powers_of_two_l224_224386


namespace smallest_m_n_l224_224409

noncomputable def g (m n : ℕ) (x : ℝ) : ℝ := Real.arccos (Real.log (↑n * x) / Real.log (↑m))

theorem smallest_m_n (m n : ℕ) (h1 : 1 < m) (h2 : ∀ x : ℝ, -1 ≤ Real.log (↑n * x) / Real.log (↑m) ∧
                      Real.log (↑n * x) / Real.log (↑m) ≤ 1 ∧
                      (forall a b : ℝ,  a ≤ x ∧ x ≤ b -> b - a = 1 / 1007)) :
  m + n = 1026 :=
sorry

end smallest_m_n_l224_224409


namespace tank_full_capacity_l224_224619

-- Define the conditions
def gas_tank_initially_full_fraction : ℚ := 4 / 5
def gas_tank_after_usage_fraction : ℚ := 1 / 3
def used_gallons : ℚ := 18

-- Define the statement that translates to "How many gallons does this tank hold when it is full?"
theorem tank_full_capacity (x : ℚ) : 
  gas_tank_initially_full_fraction * x - gas_tank_after_usage_fraction * x = used_gallons → 
  x = 270 / 7 :=
sorry

end tank_full_capacity_l224_224619


namespace tangent_line_parabola_l224_224806

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  sorry

end tangent_line_parabola_l224_224806


namespace flowchart_basic_elements_includes_loop_l224_224677

theorem flowchart_basic_elements_includes_loop 
  (sequence_structure : Prop)
  (condition_structure : Prop)
  (loop_structure : Prop)
  : ∃ element : ℕ, element = 2 := 
by
  -- Assume 0 is A: Judgment
  -- Assume 1 is B: Directed line
  -- Assume 2 is C: Loop
  -- Assume 3 is D: Start
  sorry

end flowchart_basic_elements_includes_loop_l224_224677


namespace number_of_five_digit_numbers_with_one_odd_digit_l224_224027

def odd_digits : List ℕ := [1, 3, 5, 7, 9]
def even_digits : List ℕ := [0, 2, 4, 6, 8]

def five_digit_numbers_with_one_odd_digit : ℕ :=
  let num_1st_position := odd_digits.length * even_digits.length ^ 4
  let num_other_positions := 4 * odd_digits.length * (even_digits.length - 1) * (even_digits.length ^ 3)
  num_1st_position + num_other_positions

theorem number_of_five_digit_numbers_with_one_odd_digit :
  five_digit_numbers_with_one_odd_digit = 10625 :=
by
  sorry

end number_of_five_digit_numbers_with_one_odd_digit_l224_224027


namespace bill_money_left_l224_224051

def bill_remaining_money (merchantA_qty : Int) (merchantA_rate : Int) 
                        (merchantB_qty : Int) (merchantB_rate : Int)
                        (fine : Int) (merchantC_qty : Int) (merchantC_rate : Int) 
                        (protection_costs : Int) (passerby_qty : Int) 
                        (passerby_rate : Int) : Int :=
let incomeA := merchantA_qty * merchantA_rate
let incomeB := merchantB_qty * merchantB_rate
let incomeC := merchantC_qty * merchantC_rate
let incomeD := passerby_qty * passerby_rate
let total_income := incomeA + incomeB + incomeC + incomeD
let total_expenses := fine + protection_costs
total_income - total_expenses

theorem bill_money_left 
    (merchantA_qty : Int := 8) 
    (merchantA_rate : Int := 9) 
    (merchantB_qty : Int := 15) 
    (merchantB_rate : Int := 11) 
    (fine : Int := 80)
    (merchantC_qty : Int := 25) 
    (merchantC_rate : Int := 8) 
    (protection_costs : Int := 30) 
    (passerby_qty : Int := 12) 
    (passerby_rate : Int := 7) : 
    bill_remaining_money merchantA_qty merchantA_rate 
                         merchantB_qty merchantB_rate 
                         fine merchantC_qty merchantC_rate 
                         protection_costs passerby_qty 
                         passerby_rate = 411 := by 
  sorry

end bill_money_left_l224_224051


namespace cost_price_of_product_is_100_l224_224653

theorem cost_price_of_product_is_100 
  (x : ℝ) 
  (h : x * 1.2 * 0.9 - x = 8) : 
  x = 100 := 
sorry

end cost_price_of_product_is_100_l224_224653


namespace Jack_gave_Mike_six_notebooks_l224_224041

theorem Jack_gave_Mike_six_notebooks :
  ∀ (Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike : ℕ),
  Gerald_notebooks = 8 →
  Jack_notebooks_left = 10 →
  notebooks_given_to_Paula = 5 →
  total_notebooks_initial = Gerald_notebooks + 13 →
  jack_notebooks_after_Paula = total_notebooks_initial - notebooks_given_to_Paula →
  notebooks_given_to_Mike = jack_notebooks_after_Paula - Jack_notebooks_left →
  notebooks_given_to_Mike = 6 :=
by
  intros Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike
  intros Gerald_notebooks_eq Jack_notebooks_left_eq notebooks_given_to_Paula_eq total_notebooks_initial_eq jack_notebooks_after_Paula_eq notebooks_given_to_Mike_eq
  sorry

end Jack_gave_Mike_six_notebooks_l224_224041


namespace team_total_games_123_l224_224801

theorem team_total_games_123 {G : ℕ} 
  (h1 : (55 / 100) * 35 + (90 / 100) * (G - 35) = (80 / 100) * G) : 
  G = 123 :=
sorry

end team_total_games_123_l224_224801


namespace animals_consuming_hay_l224_224800

-- Define the rate of consumption for each animal
def rate_goat : ℚ := 1 / 6 -- goat consumes 1 cartload per 6 weeks
def rate_sheep : ℚ := 1 / 8 -- sheep consumes 1 cartload per 8 weeks
def rate_cow : ℚ := 1 / 3 -- cow consumes 1 cartload per 3 weeks

-- Define the number of animals
def num_goats : ℚ := 5
def num_sheep : ℚ := 3
def num_cows : ℚ := 2

-- Define the total rate of consumption
def total_rate : ℚ := (num_goats * rate_goat) + (num_sheep * rate_sheep) + (num_cows * rate_cow)

-- Define the total amount of hay to be consumed
def total_hay : ℚ := 30

-- Define the time required to consume the total hay at the calculated rate
def time_required : ℚ := total_hay / total_rate

-- Theorem stating the time required to consume 30 cartloads of hay is 16 weeks.
theorem animals_consuming_hay : time_required = 16 := by
  sorry

end animals_consuming_hay_l224_224800


namespace simplify_expr_l224_224489

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l224_224489


namespace seashells_left_l224_224253

-- Definitions based on conditions
def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

-- Theorem stating the proof problem
theorem seashells_left (initial_seashells seashells_given_away : ℕ) : initial_seashells - seashells_given_away = 17 := 
    by
        sorry

end seashells_left_l224_224253


namespace abs_diff_of_roots_eq_one_l224_224998

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end abs_diff_of_roots_eq_one_l224_224998


namespace cardProblem_l224_224616

structure InitialState where
  jimmy_cards : ℕ
  bob_cards : ℕ
  sarah_cards : ℕ

structure UpdatedState where
  jimmy_cards_final : ℕ
  sarah_cards_final : ℕ
  sarahs_friends_cards : ℕ

def cardProblemSolved (init : InitialState) (final : UpdatedState) : Prop :=
  let bob_initial := init.bob_cards + 6
  let bob_to_sarah := bob_initial / 3
  let bob_final := bob_initial - bob_to_sarah
  let sarah_initial := init.sarah_cards + bob_to_sarah
  let sarah_friends := sarah_initial / 3
  let sarah_final := sarah_initial - 3 * sarah_friends
  let mary_cards := 2 * 6
  let jimmy_final := init.jimmy_cards - 6 - mary_cards
  let sarah_to_tim := 0 -- since Sarah can't give fractional cards
  (final.jimmy_cards_final = jimmy_final) ∧ 
  (final.sarah_cards_final = sarah_final - sarah_to_tim) ∧ 
  (final.sarahs_friends_cards = sarah_friends)

theorem cardProblem : 
  cardProblemSolved 
    { jimmy_cards := 68, bob_cards := 5, sarah_cards := 7 }
    { jimmy_cards_final := 50, sarah_cards_final := 1, sarahs_friends_cards := 3 } :=
by 
  sorry

end cardProblem_l224_224616


namespace cistern_empty_time_without_tap_l224_224591

noncomputable def leak_rate (L : ℕ) : Prop :=
  let tap_rate := 4
  let cistern_volume := 480
  let empty_time_with_tap := 24
  let empty_rate_net := cistern_volume / empty_time_with_tap
  L - tap_rate = empty_rate_net

theorem cistern_empty_time_without_tap (L : ℕ) (h : leak_rate L) :
  480 / L = 20 := by
  -- placeholder for the proof
  sorry

end cistern_empty_time_without_tap_l224_224591


namespace intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l224_224696

-- Definitions for U, A, B
def U := { x : ℤ | 0 < x ∧ x <= 10 }
def A : Set ℤ := { 1, 2, 4, 5, 9 }
def B : Set ℤ := { 4, 6, 7, 8, 10 }

-- 1. Prove A ∩ B = {4}
theorem intersection_eq : A ∩ B = {4} := by
  sorry

-- 2. Prove A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}
theorem union_eq : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

-- 3. Prove complement_U (A ∪ B) = {3}
def complement_U (s : Set ℤ) : Set ℤ := { x ∈ U | ¬ (x ∈ s) }
theorem complement_union_eq : complement_U (A ∪ B) = {3} := by
  sorry

-- 4. Prove (complement_U A) ∩ (complement_U B) = {3}
theorem intersection_complements_eq : (complement_U A) ∩ (complement_U B) = {3} := by
  sorry

end intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l224_224696


namespace total_vehicle_wheels_in_parking_lot_l224_224501

def vehicles_wheels := (1 * 4) + (1 * 4) + (8 * 4) + (4 * 2) + (3 * 6) + (2 * 4) + (1 * 8) + (2 * 3)

theorem total_vehicle_wheels_in_parking_lot : vehicles_wheels = 88 :=
by {
    sorry
}

end total_vehicle_wheels_in_parking_lot_l224_224501


namespace trail_length_l224_224664

variables (a b c d e : ℕ)

theorem trail_length : 
  a + b + c = 45 ∧
  b + d = 36 ∧
  c + d + e = 60 ∧
  a + d = 32 → 
  a + b + c + d + e = 69 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end trail_length_l224_224664


namespace susan_average_speed_l224_224048

noncomputable def average_speed_trip (d1 d2 : ℝ) (v1 v2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let time1 := d1 / v1
  let time2 := d2 / v2
  let total_time := time1 + time2
  total_distance / total_time

theorem susan_average_speed :
  average_speed_trip 60 30 30 60 = 36 := 
by
  -- The proof can be filled in here
  sorry

end susan_average_speed_l224_224048


namespace circular_pipes_equivalence_l224_224003

/-- Determine how many circular pipes with an inside diameter 
of 2 inches are required to carry the same amount of water as 
one circular pipe with an inside diameter of 8 inches. -/
theorem circular_pipes_equivalence 
  (d_small d_large : ℝ)
  (h1 : d_small = 2)
  (h2 : d_large = 8) :
  (d_large / 2) ^ 2 / (d_small / 2) ^ 2 = 16 :=
by
  sorry

end circular_pipes_equivalence_l224_224003


namespace abc_divisibility_l224_224949

theorem abc_divisibility (a b c : ℕ) (h1 : a^2 * b ∣ a^3 + b^3 + c^3) (h2 : b^2 * c ∣ a^3 + b^3 + c^3) (h3 : c^2 * a ∣ a^3 + b^3 + c^3) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end abc_divisibility_l224_224949


namespace gamma_received_eight_donuts_l224_224994

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end gamma_received_eight_donuts_l224_224994


namespace find_stream_speed_l224_224491

theorem find_stream_speed (b s : ℝ) 
  (h1 : b + s = 10) 
  (h2 : b - s = 8) : s = 1 :=
by
  sorry

end find_stream_speed_l224_224491


namespace f_neg_2008_value_l224_224239

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem f_neg_2008_value (h : f a b 2008 = 10) : f a b (-2008) = -12 := by
  sorry

end f_neg_2008_value_l224_224239


namespace number_of_bass_caught_l224_224560

/-
Statement:
Given:
1. An eight-pound trout.
2. Two twelve-pound salmon.
3. They need to feed 22 campers with two pounds of fish each.
Prove that the number of two-pound bass caught is 6.
-/

theorem number_of_bass_caught
  (weight_trout : ℕ := 8)
  (weight_salmon : ℕ := 12)
  (num_salmon : ℕ := 2)
  (num_campers : ℕ := 22)
  (required_per_camper : ℕ := 2)
  (weight_bass : ℕ := 2) :
  (num_campers * required_per_camper - (weight_trout + num_salmon * weight_salmon)) / weight_bass = 6 :=
by
  sorry  -- Proof to be completed

end number_of_bass_caught_l224_224560


namespace local_extrema_l224_224359

-- Defining the function y = 1 + 3x - x^3
def y (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

-- Statement of the problem to be proved
theorem local_extrema :
  (∃ x : ℝ, x = -1 ∧ y x = -1 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 1) < δ → y z ≥ y (-1)) ∧
  (∃ x : ℝ, x = 1 ∧ y x = 3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z - 1) < δ → y z ≤ y 1) :=
by sorry

end local_extrema_l224_224359


namespace only_one_true_l224_224085

-- Definitions based on conditions
def line := Type
def plane := Type
def parallel (m n : line) : Prop := sorry
def perpendicular (m n : line) : Prop := sorry
def subset (m : line) (alpha : plane) : Prop := sorry

-- Propositions derived from conditions
def prop1 (m n : line) (alpha : plane) : Prop := parallel m alpha ∧ parallel n alpha → ¬ parallel m n
def prop2 (m n : line) (alpha : plane) : Prop := perpendicular m alpha ∧ perpendicular n alpha → parallel m n
def prop3 (m n : line) (alpha beta : plane) : Prop := parallel alpha beta ∧ subset m alpha ∧ subset n beta → parallel m n
def prop4 (m n : line) (alpha beta : plane) : Prop := perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular m alpha → perpendicular n beta

-- Theorem statement that only one proposition is true
theorem only_one_true (m n : line) (alpha beta : plane) :
  (prop1 m n alpha = false) ∧
  (prop2 m n alpha = true) ∧
  (prop3 m n alpha beta = false) ∧
  (prop4 m n alpha beta = false) :=
by sorry

end only_one_true_l224_224085


namespace faye_earned_total_l224_224435

-- Definitions of the necklace sales
def bead_necklaces := 3
def bead_price := 7
def gemstone_necklaces := 7
def gemstone_price := 10
def pearl_necklaces := 2
def pearl_price := 12
def crystal_necklaces := 5
def crystal_price := 15

-- Total amount calculation
def total_amount := 
  bead_necklaces * bead_price + 
  gemstone_necklaces * gemstone_price + 
  pearl_necklaces * pearl_price + 
  crystal_necklaces * crystal_price

-- Proving the total amount equals $190
theorem faye_earned_total : total_amount = 190 := by
  sorry

end faye_earned_total_l224_224435


namespace parents_present_l224_224618

theorem parents_present (pupils teachers total_people parents : ℕ)
  (h_pupils : pupils = 724)
  (h_teachers : teachers = 744)
  (h_total_people : total_people = 1541) :
  parents = total_people - (pupils + teachers) :=
sorry

end parents_present_l224_224618


namespace complement_M_l224_224497

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}
def C (s : Set ℝ) : Set ℝ := sᶜ -- complement of a set

theorem complement_M :
  C M = {x : ℝ | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l224_224497


namespace circular_garden_area_l224_224242

open Real

theorem circular_garden_area (r : ℝ) (h₁ : r = 8)
      (h₂ : 2 * π * r = (1 / 4) * π * r ^ 2) :
  π * r ^ 2 = 64 * π :=
by
  -- The proof will go here
  sorry

end circular_garden_area_l224_224242


namespace eq_op_l224_224659

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 + 2 * x - y

-- State the theorem to be proven
theorem eq_op (k : ℝ) : op k (op k k) = k := sorry

end eq_op_l224_224659


namespace arithmetic_sequence_sum_l224_224535

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end arithmetic_sequence_sum_l224_224535


namespace circle_tangent_line_l224_224732

theorem circle_tangent_line {m : ℝ} : 
  (3 * (0 : ℝ) - 4 * (1 : ℝ) - 6 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 2 * y + m = 0) → 
  m = -3 := by
  sorry

end circle_tangent_line_l224_224732


namespace speed_of_other_train_l224_224500

theorem speed_of_other_train :
  ∀ (d : ℕ) (v1 v2 : ℕ), d = 120 → v1 = 30 → 
    ∀ (d_remaining : ℕ), d_remaining = 70 → 
    v1 + v2 = d_remaining → 
    v2 = 40 :=
by
  intros d v1 v2 h_d h_v1 d_remaining h_d_remaining h_rel_speed
  sorry

end speed_of_other_train_l224_224500


namespace totalPeoplePresent_is_630_l224_224748

def totalParents : ℕ := 105
def totalPupils : ℕ := 698

def groupA_fraction : ℚ := 30 / 100
def groupB_fraction : ℚ := 25 / 100
def groupC_fraction : ℚ := 20 / 100
def groupD_fraction : ℚ := 15 / 100
def groupE_fraction : ℚ := 10 / 100

def groupA_attendance : ℚ := 90 / 100
def groupB_attendance : ℚ := 80 / 100
def groupC_attendance : ℚ := 70 / 100
def groupD_attendance : ℚ := 60 / 100
def groupE_attendance : ℚ := 50 / 100

def junior_fraction : ℚ := 30 / 100
def intermediate_fraction : ℚ := 35 / 100
def senior_fraction : ℚ := 20 / 100
def advanced_fraction : ℚ := 15 / 100

def junior_attendance : ℚ := 85 / 100
def intermediate_attendance : ℚ := 80 / 100
def senior_attendance : ℚ := 75 / 100
def advanced_attendance : ℚ := 70 / 100

noncomputable def totalPeoplePresent : ℚ := 
  totalParents * groupA_fraction * groupA_attendance +
  totalParents * groupB_fraction * groupB_attendance +
  totalParents * groupC_fraction * groupC_attendance +
  totalParents * groupD_fraction * groupD_attendance +
  totalParents * groupE_fraction * groupE_attendance +
  totalPupils * junior_fraction * junior_attendance +
  totalPupils * intermediate_fraction * intermediate_attendance +
  totalPupils * senior_fraction * senior_attendance +
  totalPupils * advanced_fraction * advanced_attendance

theorem totalPeoplePresent_is_630 : totalPeoplePresent.floor = 630 := 
by 
  sorry -- no proof required as per the instructions

end totalPeoplePresent_is_630_l224_224748


namespace original_number_of_men_l224_224633

/--A group of men decided to complete a work in 6 days. 
 However, 4 of them became absent, and the remaining men finished the work in 12 days. 
 Given these conditions, we need to prove that the original number of men was 8. --/
theorem original_number_of_men 
  (x : ℕ) -- original number of men
  (h1 : x * 6 = (x - 4) * 12) -- total work remains the same
  : x = 8 := 
sorry

end original_number_of_men_l224_224633


namespace max_sum_a_b_l224_224090

theorem max_sum_a_b (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := 
by sorry

end max_sum_a_b_l224_224090


namespace julies_balls_after_1729_steps_l224_224298

-- Define the process described
def increment_base_8 (n : ℕ) : List ℕ := 
by
  if n = 0 then
    exact [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 8) (n % 8 :: acc)
    exact loop n []

-- Define the total number of balls after 'steps' steps
def julies_total_balls (steps : ℕ) : ℕ :=
by 
  exact (increment_base_8 steps).sum

theorem julies_balls_after_1729_steps : julies_total_balls 1729 = 7 :=
by
  sorry

end julies_balls_after_1729_steps_l224_224298


namespace arithmetic_seq_sum_l224_224428

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n+1) - a n = a (n+2) - a (n+1))
  (h2 : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l224_224428


namespace sum_of_fourth_powers_l224_224411

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 :=
by sorry

end sum_of_fourth_powers_l224_224411


namespace infinite_solutions_no_solutions_l224_224237

-- Define the geometric sequence with first term a1 = 1 and common ratio q
def a1 : ℝ := 1
def a2 (q : ℝ) : ℝ := a1 * q
def a3 (q : ℝ) : ℝ := a1 * q^2
def a4 (q : ℝ) : ℝ := a1 * q^3

-- Define the system of linear equations
def system_of_eqns (x y q : ℝ) : Prop :=
  a1 * x + a3 q * y = 3 ∧ a2 q * x + a4 q * y = -2

-- Conditions for infinitely many solutions
theorem infinite_solutions (q x y : ℝ) :
  q = -2 / 3 → ∃ x y, system_of_eqns x y q :=
by
  sorry

-- Conditions for no solutions
theorem no_solutions (q : ℝ) :
  q ≠ -2 / 3 → ¬∃ x y, system_of_eqns x y q :=
by
  sorry

end infinite_solutions_no_solutions_l224_224237


namespace center_of_square_l224_224705

theorem center_of_square (O : ℝ × ℝ) (A B C D : ℝ × ℝ) 
  (hAB : dist A B = 1) 
  (hA : A = (0, 0)) 
  (hB : B = (1, 0)) 
  (hC : C = (1, 1)) 
  (hD : D = (0, 1)) 
  (h_sum_squares : (dist O A)^2 + (dist O B)^2 + (dist O C)^2 + (dist O D)^2 = 2): 
  O = (1/2, 1/2) :=
by sorry

end center_of_square_l224_224705


namespace rectangle_area_l224_224679

theorem rectangle_area (length : ℝ) (width : ℝ) (area : ℝ) 
  (h1 : length = 24) 
  (h2 : width = 0.875 * length) 
  (h3 : area = length * width) : 
  area = 504 := 
by
  sorry

end rectangle_area_l224_224679


namespace giant_lollipop_calories_l224_224758

-- Definitions based on the conditions
def sugar_per_chocolate_bar := 10
def chocolate_bars_bought := 14
def sugar_in_giant_lollipop := 37
def total_sugar := 177
def calories_per_gram_of_sugar := 4

-- Prove that the number of calories in the giant lollipop is 148 given the conditions
theorem giant_lollipop_calories : (sugar_in_giant_lollipop * calories_per_gram_of_sugar) = 148 := by
  sorry

end giant_lollipop_calories_l224_224758


namespace fraction_of_canvas_painted_blue_l224_224556

noncomputable def square_canvas_blue_fraction : ℚ :=
  sorry

theorem fraction_of_canvas_painted_blue :
  square_canvas_blue_fraction = 3 / 8 :=
  sorry

end fraction_of_canvas_painted_blue_l224_224556


namespace CarmenBrushLengthInCentimeters_l224_224765

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_l224_224765


namespace middle_schoolers_count_l224_224510

theorem middle_schoolers_count (total_students : ℕ) (fraction_girls : ℚ) 
  (primary_girls_fraction : ℚ) (primary_boys_fraction : ℚ) 
  (num_girls : ℕ) (num_boys: ℕ) (primary_grade_girls : ℕ) 
  (primary_grade_boys : ℕ) :
  total_students = 800 →
  fraction_girls = 5 / 8 →
  primary_girls_fraction = 7 / 10 →
  primary_boys_fraction = 2 / 5 →
  num_girls = fraction_girls * total_students →
  num_boys = total_students - num_girls →
  primary_grade_girls = primary_girls_fraction * num_girls →
  primary_grade_boys = primary_boys_fraction * num_boys →
  total_students - (primary_grade_girls + primary_grade_boys) = 330 :=
by
  intros
  sorry

end middle_schoolers_count_l224_224510


namespace round_table_arrangement_l224_224456

theorem round_table_arrangement :
  ∀ (n : ℕ), n = 10 → (∃ factorial_value : ℕ, factorial_value = Nat.factorial (n - 1) ∧ factorial_value = 362880) := by
  sorry

end round_table_arrangement_l224_224456


namespace fewer_mpg_in_city_l224_224261

def city_miles : ℕ := 336
def highway_miles : ℕ := 462
def city_mpg : ℕ := 24

def tank_size : ℕ := city_miles / city_mpg
def highway_mpg : ℕ := highway_miles / tank_size

theorem fewer_mpg_in_city : highway_mpg - city_mpg = 9 :=
by
  sorry

end fewer_mpg_in_city_l224_224261


namespace complementary_angles_decrease_86_percent_l224_224266

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l224_224266


namespace value_of_m_l224_224781

theorem value_of_m (m : ℝ) :
  let A := {2, 3}
  let B := {x : ℝ | m * x - 6 = 0}
  (B ⊆ A) → (m = 0 ∨ m = 2 ∨ m = 3) :=
by
  intros A B h
  sorry

end value_of_m_l224_224781


namespace tangent_parallel_and_point_P_l224_224846

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_parallel_and_point_P (P : ℝ × ℝ) (hP1 : P = (1, f 1)) (hP2 : P = (-1, f (-1))) :
  (f 1 = 3 ∧ f (-1) = 3) ∧ (deriv f 1 = 2 ∧ deriv f (-1) = 2) :=
by
  sorry

end tangent_parallel_and_point_P_l224_224846


namespace length_of_AB_l224_224795

-- Define the parabola and the line passing through the focus F
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line (x y : ℝ) : Prop := y = x - 1

theorem length_of_AB : 
  (∃ F : ℝ × ℝ, F = (1, 0) ∧ line F.1 F.2) →
  (∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)) :=
by
  sorry

end length_of_AB_l224_224795


namespace find_x1_l224_224363

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3)
  (h3 : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4 / 5 :=
sorry

end find_x1_l224_224363


namespace problem_solution_l224_224595

theorem problem_solution (k m : ℕ) (h1 : 30^k ∣ 929260) (h2 : 20^m ∣ 929260) : (3^k - k^3) + (2^m - m^3) = 2 := 
by sorry

end problem_solution_l224_224595


namespace min_value_of_expression_l224_224568

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) : 
  x + 2 * y ≥ 9 + 4 * Real.sqrt 2 := 
sorry

end min_value_of_expression_l224_224568


namespace percentage_increase_l224_224113

theorem percentage_increase (D1 D2 : ℕ) (total_days : ℕ) (H1 : D1 = 4) (H2 : total_days = 9) (H3 : D1 + D2 = total_days) : 
  (D2 - D1) / D1 * 100 = 25 := 
sorry

end percentage_increase_l224_224113


namespace correct_operation_l224_224511

theorem correct_operation (a : ℝ) :
  (2 * a^2) * a = 2 * a^3 :=
by sorry

end correct_operation_l224_224511


namespace min_value_l224_224368

variable (d : ℕ) (a_n S_n : ℕ → ℕ)
variable (a1 : ℕ) (H1 : d ≠ 0)
variable (H2 : a1 = 1)
variable (H3 : (a_n 3)^2 = a1 * (a_n 13))
variable (H4 : a_n n = a1 + (n - 1) * d)
variable (H5 : S_n n = (n * (a1 + a_n n)) / 2)

theorem min_value (n : ℕ) (Hn : 1 ≤ n) : 
  ∃ n, ∀ m, 1 ≤ m → (2 * S_n n + 16) / (a_n n + 3) ≥ (2 * S_n m + 16) / (a_n m + 3) ∧ (2 * S_n n + 16) / (a_n n + 3) = 4 :=
sorry

end min_value_l224_224368


namespace perfect_cubes_not_divisible_by_10_l224_224757

-- Definitions based on conditions
def is_divisible_by_10 (n : ℕ) : Prop := 10 ∣ n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k ^ 3
def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

-- Main statement
theorem perfect_cubes_not_divisible_by_10 (x : ℕ) :
  is_perfect_cube x ∧ ¬ is_divisible_by_10 x ∧ is_perfect_cube (erase_last_three_digits x) →
  x = 1331 ∨ x = 1728 :=
by
  sorry

end perfect_cubes_not_divisible_by_10_l224_224757


namespace guinea_pig_food_ratio_l224_224388

-- Definitions of amounts of food consumed by each guinea pig
def first_guinea_pig_food : ℕ := 2
variable (x : ℕ)
def second_guinea_pig_food : ℕ := x
def third_guinea_pig_food : ℕ := x + 3

-- Total food requirement condition
def total_food_required := first_guinea_pig_food + second_guinea_pig_food x + third_guinea_pig_food x = 13

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- The goal is to prove this ratio given the conditions
theorem guinea_pig_food_ratio (h : total_food_required x) : ratio (second_guinea_pig_food x) first_guinea_pig_food = 2 := by
  sorry

end guinea_pig_food_ratio_l224_224388


namespace crackers_eaten_by_Daniel_and_Elsie_l224_224078

theorem crackers_eaten_by_Daniel_and_Elsie :
  ∀ (initial_crackers remaining_crackers eaten_by_Ally eaten_by_Bob eaten_by_Clair: ℝ),
    initial_crackers = 27.5 →
    remaining_crackers = 10.5 →
    eaten_by_Ally = 3.5 →
    eaten_by_Bob = 4.0 →
    eaten_by_Clair = 5.5 →
    initial_crackers - remaining_crackers = (eaten_by_Ally + eaten_by_Bob + eaten_by_Clair) + (4 : ℝ) :=
by sorry

end crackers_eaten_by_Daniel_and_Elsie_l224_224078


namespace no_monochromatic_10_term_progression_l224_224877

def can_color_without_monochromatic_progression (n k : ℕ) (c : Fin n → Fin k) : Prop :=
  ∀ (a d : ℕ), (a < n) → (a + (9 * d) < n) → (∀ i : ℕ, i < 10 → c ⟨a + (i * d), sorry⟩ = c ⟨a, sorry⟩) → 
    (∃ j i : ℕ, j < 10 ∧ i < 10 ∧ c ⟨a + (i * d), sorry⟩ ≠ c ⟨a + (j * d), sorry⟩)

theorem no_monochromatic_10_term_progression :
  ∃ c : Fin 2008 → Fin 4, can_color_without_monochromatic_progression 2008 4 c :=
sorry

end no_monochromatic_10_term_progression_l224_224877


namespace zoo_animals_left_l224_224164

noncomputable def totalAnimalsLeft (x : ℕ) : ℕ := 
  let initialFoxes := 2 * x
  let initialRabbits := 3 * x
  let foxesAfterMove := initialFoxes - 10
  let rabbitsAfterMove := initialRabbits / 2
  foxesAfterMove + rabbitsAfterMove

theorem zoo_animals_left (x : ℕ) (h : 20 * x - 100 = 39 * x / 2) : totalAnimalsLeft x = 690 := by
  sorry

end zoo_animals_left_l224_224164


namespace value_of_a_l224_224220

theorem value_of_a (a b : ℚ) (h₁ : b = 3 * a) (h₂ : b = 12 - 5 * a) : a = 3 / 2 :=
by
  sorry

end value_of_a_l224_224220


namespace sum_of_vertical_asymptotes_l224_224194

noncomputable def sum_of_roots (a b c : ℝ) (h_discriminant : b^2 - 4*a*c ≠ 0) : ℝ :=
-(b/a)

theorem sum_of_vertical_asymptotes :
  let f := (6 * (x^2) - 8) / (4 * (x^2) + 7*x + 3)
  ∃ c d, c ≠ d ∧ (4*c^2 + 7*c + 3 = 0) ∧ (4*d^2 + 7*d + 3 = 0)
  ∧ c + d = -7 / 4 :=
by
  sorry

end sum_of_vertical_asymptotes_l224_224194


namespace hats_needed_to_pay_51_l224_224956

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_amount : ℕ := 51
def num_shirts : ℕ := 3
def num_jeans : ℕ := 2

theorem hats_needed_to_pay_51 :
  ∃ (n : ℕ), total_amount = num_shirts * shirt_cost + num_jeans * jeans_cost + n * hat_cost ∧ n = 4 :=
by
  sorry

end hats_needed_to_pay_51_l224_224956


namespace circle_center_sum_l224_224815

/-- Given the equation of a circle, prove that the sum of the x and y coordinates of the center is -1. -/
theorem circle_center_sum (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by 
  sorry

end circle_center_sum_l224_224815


namespace cone_shorter_height_ratio_l224_224477

theorem cone_shorter_height_ratio 
  (circumference : ℝ) (original_height : ℝ) (volume_shorter_cone : ℝ) 
  (shorter_height : ℝ) (radius : ℝ) :
  circumference = 24 * Real.pi ∧ 
  original_height = 40 ∧ 
  volume_shorter_cone = 432 * Real.pi ∧ 
  2 * Real.pi * radius = circumference ∧ 
  volume_shorter_cone = (1 / 3) * Real.pi * radius^2 * shorter_height
  → shorter_height / original_height = 9 / 40 :=
by
  sorry

end cone_shorter_height_ratio_l224_224477


namespace average_marks_math_chem_l224_224204

theorem average_marks_math_chem (M P C : ℝ) (h1 : M + P = 60) (h2 : C = P + 20) : 
  (M + C) / 2 = 40 := 
by
  sorry

end average_marks_math_chem_l224_224204


namespace jessie_weight_before_jogging_l224_224107

theorem jessie_weight_before_jogging (current_weight lost_weight : ℕ) 
(hc : current_weight = 67)
(hl : lost_weight = 7) : 
current_weight + lost_weight = 74 := 
by
  -- Here we skip the proof part
  sorry

end jessie_weight_before_jogging_l224_224107


namespace original_average_age_older_l224_224627

theorem original_average_age_older : 
  ∀ (n : ℕ) (T : ℕ), (T = n * 40) →
  (T + 408) / (n + 12) = 36 →
  40 - 36 = 4 :=
by
  intros n T hT hNewAvg
  sorry

end original_average_age_older_l224_224627


namespace average_of_all_results_is_24_l224_224221

-- Definitions translated from conditions
def average_1 := 20
def average_2 := 30
def n1 := 30
def n2 := 20
def total_sum_1 := n1 * average_1
def total_sum_2 := n2 * average_2

-- Lean 4 statement
theorem average_of_all_results_is_24
  (h1 : total_sum_1 = n1 * average_1)
  (h2 : total_sum_2 = n2 * average_2) :
  ((total_sum_1 + total_sum_2) / (n1 + n2) = 24) :=
by
  sorry

end average_of_all_results_is_24_l224_224221


namespace sequence_evaluation_l224_224739

noncomputable def a : ℕ → ℤ → ℤ
| 0, x => 1
| 1, x => x^2 + x + 1
| (n + 2), x => (x^n + 1) * a (n + 1) x - a n x 

theorem sequence_evaluation : a 2010 1 = 4021 := by
  sorry

end sequence_evaluation_l224_224739


namespace find_a3_l224_224396

theorem find_a3 (a0 a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, x^4 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4) →
  a3 = -8 :=
by
  sorry

end find_a3_l224_224396


namespace quadratic_intersection_l224_224173

def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_intersection:
  ∃ a b c : ℝ, 
  quadratic a b c (-3) = 16 ∧ 
  quadratic a b c 0 = -5 ∧ 
  quadratic a b c 3 = -8 ∧ 
  quadratic a b c (-1) = 0 :=
sorry

end quadratic_intersection_l224_224173


namespace third_studio_students_l224_224159

theorem third_studio_students 
  (total_students : ℕ)
  (first_studio : ℕ)
  (second_studio : ℕ) 
  (third_studio : ℕ) 
  (h1 : total_students = 376) 
  (h2 : first_studio = 110) 
  (h3 : second_studio = 135) 
  (h4 : total_students = first_studio + second_studio + third_studio) :
  third_studio = 131 := 
sorry

end third_studio_students_l224_224159


namespace neil_initial_games_l224_224954

theorem neil_initial_games (N : ℕ) 
  (H₀ : ℕ) (H₀_eq : H₀ = 58)
  (H₁ : ℕ) (H₁_eq : H₁ = H₀ - 6)
  (H₁_condition : H₁ = 4 * (N + 6)) : N = 7 :=
by {
  -- Substituting the given values and simplifying to show the final equation
  sorry
}

end neil_initial_games_l224_224954


namespace gcd_poly_correct_l224_224978

-- Define the conditions
def is_even_multiple_of (x k : ℕ) : Prop :=
  ∃ (n : ℕ), x = k * 2 * n

variable (b : ℕ)

-- Given condition
axiom even_multiple_7768 : is_even_multiple_of b 7768

-- Define the polynomials
def poly1 (b : ℕ) := 4 * b * b + 37 * b + 72
def poly2 (b : ℕ) := 3 * b + 8

-- Proof statement
theorem gcd_poly_correct : gcd (poly1 b) (poly2 b) = 8 :=
  sorry

end gcd_poly_correct_l224_224978


namespace pure_alcohol_addition_problem_l224_224527

-- Define the initial conditions
def initial_volume := 6
def initial_concentration := 0.30
def final_concentration := 0.50

-- Define the amount of pure alcohol to be added
def x := 2.4

-- Proof problem statement
theorem pure_alcohol_addition_problem (initial_volume initial_concentration final_concentration x : ℝ) :
  initial_volume * initial_concentration + x = final_concentration * (initial_volume + x) :=
by
  -- Initial condition values definition
  let initial_volume := 6
  let initial_concentration := 0.30
  let final_concentration := 0.50
  let x := 2.4
  -- Skip the proof
  sorry

end pure_alcohol_addition_problem_l224_224527


namespace minimize_m_at_l224_224879

noncomputable def m (x y : ℝ) : ℝ := 4 * x ^ 2 - 12 * x * y + 10 * y ^ 2 + 4 * y + 9

theorem minimize_m_at (x y : ℝ) : m x y = 5 ↔ (x = -3 ∧ y = -2) := 
sorry

end minimize_m_at_l224_224879


namespace fibonacci_series_sum_l224_224912

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)

noncomputable def sum_fibonacci_fraction : ℚ :=
  ∑' (n : ℕ), (fibonacci n : ℚ) / (5^n : ℚ)

theorem fibonacci_series_sum : sum_fibonacci_fraction = 5 / 19 := by
  sorry

end fibonacci_series_sum_l224_224912


namespace calc_total_push_ups_correct_l224_224630

-- Definitions based on conditions
def sets : ℕ := 9
def push_ups_per_set : ℕ := 12
def reduced_push_ups : ℕ := 8

-- Calculate total push-ups considering the reduction in the ninth set
def total_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (reduced_push_ups : ℕ) : ℕ :=
  (sets - 1) * push_ups_per_set + (push_ups_per_set - reduced_push_ups)

-- Theorem statement
theorem calc_total_push_ups_correct :
  total_push_ups sets push_ups_per_set reduced_push_ups = 100 :=
by
  sorry

end calc_total_push_ups_correct_l224_224630


namespace number_of_girls_l224_224109

theorem number_of_girls (total_children boys : ℕ) (h1 : total_children = 60) (h2 : boys = 16) : total_children - boys = 44 := by
  sorry

end number_of_girls_l224_224109


namespace remainder_67pow67_add_67_div_68_l224_224943

-- Lean statement starting with the question and conditions translated to Lean

theorem remainder_67pow67_add_67_div_68 : 
  (67 ^ 67 + 67) % 68 = 66 := 
by
  -- Condition: 67 ≡ -1 mod 68
  have h : 67 % 68 = -1 % 68 := by norm_num
  sorry

end remainder_67pow67_add_67_div_68_l224_224943


namespace inequality_system_integer_solutions_l224_224227

theorem inequality_system_integer_solutions :
  { x : ℤ | 5 * x + 1 > 3 * (x - 1) ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {-1, 0, 1, 2} := by
  sorry

end inequality_system_integer_solutions_l224_224227


namespace symmetric_intersection_range_l224_224290

theorem symmetric_intersection_range (k m p : ℝ)
  (intersection_symmetric : ∀ (x y : ℝ), 
    (x = k*y - 1 ∧ (x^2 + y^2 + k*x + m*y + 2*p = 0)) → 
    (y = x)) 
  : p < -3/2 := 
sorry

end symmetric_intersection_range_l224_224290


namespace solve_for_diamond_l224_224860

theorem solve_for_diamond (d : ℕ) (h1 : d * 9 + 6 = d * 10 + 3) (h2 : d < 10) : d = 3 :=
by
  sorry

end solve_for_diamond_l224_224860


namespace height_of_building_l224_224486

-- Define the conditions as hypotheses
def height_of_flagstaff : ℝ := 17.5
def shadow_length_of_flagstaff : ℝ := 40.25
def shadow_length_of_building : ℝ := 28.75

-- Define the height ratio based on similar triangles
theorem height_of_building :
  (height_of_flagstaff / shadow_length_of_flagstaff = 12.47 / shadow_length_of_building) :=
by
  sorry

end height_of_building_l224_224486


namespace find_a_l224_224206

-- Define the main inequality condition
def inequality_condition (a x : ℝ) : Prop := |x^2 + a * x + 4 * a| ≤ 3

-- Define the condition that there is exactly one solution to the inequality
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (inequality_condition a x) ∧ (∀ y : ℝ, x ≠ y → ¬(inequality_condition a y))

-- The theorem that states the specific values of a
theorem find_a (a : ℝ) : has_exactly_one_solution a ↔ a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13 := 
by
  sorry

end find_a_l224_224206


namespace number_of_children_l224_224158

theorem number_of_children (total_crayons children_crayons children : ℕ) 
  (h1 : children_crayons = 3) 
  (h2 : total_crayons = 18) 
  (h3 : total_crayons = children_crayons * children) : 
  children = 6 := 
by 
  sorry

end number_of_children_l224_224158


namespace f_zero_is_two_l224_224093

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x1 x2 x3 x4 x5 : ℝ) : 
  f (x1 + x2 + x3 + x4 + x5) = f x1 + f x2 + f x3 + f x4 + f x5 - 8

theorem f_zero_is_two : f 0 = 2 := 
by
  sorry

end f_zero_is_two_l224_224093


namespace savings_correct_l224_224297

-- Define the conditions
def in_store_price : ℝ := 320
def discount_rate : ℝ := 0.05
def monthly_payment : ℝ := 62
def monthly_payments : ℕ := 5
def shipping_handling : ℝ := 10

-- Prove that the savings from buying in-store is 16 dollars.
theorem savings_correct : 
  (monthly_payments * monthly_payment + shipping_handling) - (in_store_price * (1 - discount_rate)) = 16 := 
by
  sorry

end savings_correct_l224_224297


namespace wool_production_equivalence_l224_224430

variable (x y z w v : ℕ)

def wool_per_sheep_of_breed_A_per_day : ℚ :=
  (y:ℚ) / ((x:ℚ) * (z:ℚ))

def wool_per_sheep_of_breed_B_per_day : ℚ :=
  2 * wool_per_sheep_of_breed_A_per_day x y z

def total_wool_produced_by_breed_B (x y z w v: ℕ) : ℚ :=
  (w:ℚ) * wool_per_sheep_of_breed_B_per_day x y z * (v:ℚ)

theorem wool_production_equivalence :
  total_wool_produced_by_breed_B x y z w v = 2 * (y:ℚ) * (w:ℚ) * (v:ℚ) / ((x:ℚ) * (z:ℚ)) := by
  sorry

end wool_production_equivalence_l224_224430


namespace mike_notebooks_total_l224_224023

theorem mike_notebooks_total
  (red_notebooks : ℕ)
  (green_notebooks : ℕ)
  (blue_notebooks_cost : ℕ)
  (total_cost : ℕ)
  (red_cost : ℕ)
  (green_cost : ℕ)
  (blue_cost : ℕ)
  (h1 : red_notebooks = 3)
  (h2 : red_cost = 4)
  (h3 : green_notebooks = 2)
  (h4 : green_cost = 2)
  (h5 : total_cost = 37)
  (h6 : blue_cost = 3)
  (h7 : total_cost = red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks_cost) :
  (red_notebooks + green_notebooks + blue_notebooks_cost / blue_cost = 12) :=
by {
  sorry
}

end mike_notebooks_total_l224_224023


namespace greatest_of_consecutive_integers_with_sum_39_l224_224768

theorem greatest_of_consecutive_integers_with_sum_39 :
  ∃ x : ℤ, x + (x + 1) + (x + 2) = 39 ∧ max (max x (x + 1)) (x + 2) = 14 :=
by
  sorry

end greatest_of_consecutive_integers_with_sum_39_l224_224768


namespace fish_size_difference_l224_224681

variables {S J W : ℝ}

theorem fish_size_difference (h1 : S = J + 21.52) (h2 : J = W - 12.64) : S - W = 8.88 :=
sorry

end fish_size_difference_l224_224681


namespace circumcenter_distance_two_l224_224656

noncomputable def distance_between_circumcenter (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1)
  : ℝ :=
dist ( ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 ) ) ( ( (B.1 + C.1) / 2, (B.2 + C.2) / 2 )) 

theorem circumcenter_distance_two (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1) 
  : distance_between_circumcenter A B C M hAB hBC hAC hM_on_AC hCM = 2 :=
sorry

end circumcenter_distance_two_l224_224656


namespace smallest_number_l224_224455

def binary_101010 : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0
def base5_111 : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0
def octal_32 : ℕ := 3 * 8^1 + 2 * 8^0
def base6_54 : ℕ := 5 * 6^1 + 4 * 6^0

theorem smallest_number : octal_32 < binary_101010 ∧ octal_32 < base5_111 ∧ octal_32 < base6_54 :=
by
  sorry

end smallest_number_l224_224455


namespace principal_amount_l224_224189

theorem principal_amount (A2 A3 : ℝ) (interest : ℝ) (principal : ℝ) (h1 : A2 = 3450) 
  (h2 : A3 = 3655) (h_interest : interest = A3 - A2) (h_principal : principal = A2 - interest) : 
  principal = 3245 :=
by
  sorry

end principal_amount_l224_224189


namespace sum_of_roots_of_quadratic_l224_224160

theorem sum_of_roots_of_quadratic (x1 x2 : ℝ) (h : x1 * x2 + -(x1 + x2) * 6 + 5 = 0) : x1 + x2 = 6 :=
by
-- Vieta's formulas for the sum of the roots of a quadratic equation state that x1 + x2 = -b / a.
sorry

end sum_of_roots_of_quadratic_l224_224160


namespace general_term_formula_not_arithmetic_sequence_l224_224872

noncomputable def geometric_sequence (n : ℕ) : ℕ := 2^n

theorem general_term_formula :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    (∃ (q : ℕ),
      ∀ n, a n = 2^n) :=
by
  sorry

theorem not_arithmetic_sequence :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    ¬(∃ m n p : ℕ, m < n ∧ n < p ∧ (2 * a n = a m + a p)) :=
by
  sorry

end general_term_formula_not_arithmetic_sequence_l224_224872


namespace pencil_length_eq_eight_l224_224804

theorem pencil_length_eq_eight (L : ℝ) 
  (h1 : (1/8) * L + (1/2) * ((7/8) * L) + (7/2) = L) : 
  L = 8 :=
by
  sorry

end pencil_length_eq_eight_l224_224804


namespace nine_fifths_sum_l224_224845

open Real

theorem nine_fifths_sum (a b: ℝ) (ha: a > 0) (hb: b > 0)
    (h1: a * (sqrt a) + b * (sqrt b) = 183) 
    (h2: a * (sqrt b) + b * (sqrt a) = 182) : 
    9 / 5 * (a + b) = 657 := 
by 
    sorry

end nine_fifths_sum_l224_224845


namespace ferry_journey_time_difference_l224_224929

/-
  Problem statement:
  Prove that the journey of ferry Q is 1 hour longer than the journey of ferry P,
  given the following conditions:
  1. Ferry P travels for 3 hours at 6 kilometers per hour.
  2. Ferry Q takes a route that is two times longer than ferry P.
  3. Ferry P is slower than ferry Q by 3 kilometers per hour.
-/

theorem ferry_journey_time_difference :
  let speed_P := 6
  let time_P := 3
  let distance_P := speed_P * time_P
  let distance_Q := 2 * distance_P
  let speed_diff := 3
  let speed_Q := speed_P + speed_diff
  let time_Q := distance_Q / speed_Q
  time_Q - time_P = 1 :=
by
  sorry

end ferry_journey_time_difference_l224_224929


namespace truth_of_q_l224_224878

variable {p q : Prop}

theorem truth_of_q (hnp : ¬ p) (hpq : p ∨ q) : q :=
  by
  sorry

end truth_of_q_l224_224878


namespace total_number_of_baseball_cards_l224_224252

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end total_number_of_baseball_cards_l224_224252
