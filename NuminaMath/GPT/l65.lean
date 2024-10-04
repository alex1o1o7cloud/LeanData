import Mathlib

namespace smallest_number_condition_l65_65269

theorem smallest_number_condition :
  ∃ n : ℕ, (n + 1) % 12 = 0 ∧
           (n + 1) % 18 = 0 ∧
           (n + 1) % 24 = 0 ∧
           (n + 1) % 32 = 0 ∧
           (n + 1) % 40 = 0 ∧
           n = 2879 :=
sorry

end smallest_number_condition_l65_65269


namespace bobby_roaming_area_l65_65905

noncomputable def accessible_area (radius : ℝ) : ℝ :=
  (3 / 4) * π * radius ^ 2

theorem bobby_roaming_area (radius fence_w length_w fence_h length_h gyro_w gyro_h distance : ℝ)
  (h1 : radius = 5)
  (h2 : fence_w = 4)
  (h3 : fence_h = 6)
  (h4 : gyro_w = 1)
  (h5 : gyro_h = 1)
  (h6 : distance = 3)
  (h7 : accessible_area radius = 75 / 4 * π) :
  accessible_area radius = 75 / 4 * π :=
by
  sorry

end bobby_roaming_area_l65_65905


namespace smallest_base_power_l65_65254

theorem smallest_base_power (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h_log_eq : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ Real.log y / Real.log 3 = Real.log z / Real.log 5) :
  z ^ (1 / 5) < x ^ (1 / 2) ∧ z ^ (1 / 5) < y ^ (1 / 3) :=
by
  -- required proof here
  sorry

end smallest_base_power_l65_65254


namespace eccentricity_of_hyperbola_l65_65806

noncomputable def find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) : ℝ :=
Real.sqrt (1 + (b / a)^2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) :
  find_eccentricity a b h1 h2 h3 = 3 / 2 := by
  sorry

end eccentricity_of_hyperbola_l65_65806


namespace adjusted_smallest_part_proof_l65_65048

theorem adjusted_smallest_part_proof : 
  ∀ (x : ℝ), 14 * x = 100 → x + 12 = 19 + 1 / 7 := 
by
  sorry

end adjusted_smallest_part_proof_l65_65048


namespace Marian_credit_card_balance_l65_65669

theorem Marian_credit_card_balance :
  let initial_balance := 126.00 in
  let groceries := 60.00 in
  let gas := groceries / 2 in
  let returned := 45.00 in
  initial_balance + groceries + gas - returned = 171.00 :=
by
  let initial_balance := 126.00
  let groceries := 60.00
  let gas := groceries / 2
  let returned := 45.00
  calc
    126.00 + 60.00 + 30.00 - 45.00 = 216.00 - 45.00 : by congr
    ... = 171.00 : by norm_num

#suppressAllProofSteps

end Marian_credit_card_balance_l65_65669


namespace initial_fee_correct_l65_65121

-- Define the relevant values
def initialFee := 2.25
def chargePerSegment := 0.4
def totalDistance := 3.6
def totalCharge := 5.85
noncomputable def segments := (totalDistance * (5 / 2))
noncomputable def costForDistance := segments * chargePerSegment

-- Define the theorem
theorem initial_fee_correct :
  totalCharge = initialFee + costForDistance :=
by
  -- Proof is omitted.
  sorry

end initial_fee_correct_l65_65121


namespace geometric_sequence_ratio_l65_65845

theorem geometric_sequence_ratio
  (a1 r : ℝ) (h_r : r ≠ 1)
  (h : (1 - r^6) / (1 - r^3) = 1 / 2) :
  (1 - r^9) / (1 - r^3) = 3 / 4 :=
  sorry

end geometric_sequence_ratio_l65_65845


namespace find_distance_AC_l65_65162

noncomputable def distance_AC : ℝ :=
  let speed := 25  -- km per hour
  let angleA := 30  -- degrees
  let angleB := 135 -- degrees
  let distanceBC := 25 -- km
  (distanceBC * Real.sin (angleB * Real.pi / 180)) / (Real.sin (angleA * Real.pi / 180))

theorem find_distance_AC :
  distance_AC = 25 * Real.sqrt 2 :=
by
  sorry

end find_distance_AC_l65_65162


namespace remaining_payment_l65_65847

theorem remaining_payment (part_payment total_cost : ℝ) (percent_payment : ℝ) 
  (h1 : part_payment = 650) 
  (h2 : percent_payment = 15 / 100) 
  (h3 : part_payment = percent_payment * total_cost) : 
  total_cost - part_payment = 3683.33 := 
by 
  sorry

end remaining_payment_l65_65847


namespace equal_number_of_boys_and_girls_l65_65153

theorem equal_number_of_boys_and_girls
    (num_classrooms : ℕ) (girls : ℕ) (total_per_classroom : ℕ)
    (equal_boys_and_girls : ∀ (c : ℕ), c ≤ num_classrooms → (girls + boys) = total_per_classroom):
    num_classrooms = 4 → girls = 44 → total_per_classroom = 25 → boys = 44 :=
by
  sorry

end equal_number_of_boys_and_girls_l65_65153


namespace find_equation_of_line_l_l65_65630

-- Define the conditions
def point_P : ℝ × ℝ := (2, 3)

noncomputable def angle_of_inclination : ℝ := 2 * Real.pi / 3

def intercept_condition (a b : ℝ) : Prop := a + b = 0

-- The proof statement
theorem find_equation_of_line_l :
  ∃ (k : ℝ), k = Real.tan angle_of_inclination ∧
  ∃ (C : ℝ), ∀ (x y : ℝ), (y - 3 = k * (x - 2)) ∧ C = (3 + 2 * (Real.sqrt 3)) ∨ 
             (intercept_condition (x / point_P.1) (y / point_P.2) ∧ C = 1) ∨ 
             -- The standard forms of the line equation
             ((Real.sqrt 3 * x + y - C = 0) ∨ (x - y + 1 = 0)) :=
sorry

end find_equation_of_line_l_l65_65630


namespace expand_polynomial_l65_65916

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l65_65916


namespace bill_experience_now_l65_65599

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l65_65599


namespace quadratic_inequality_solution_set_l65_65643

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) : 
  ∀ x : ℝ, (x^2 - (m + 1/m) * x + 1 < 0) ↔ m < x ∧ x < 1/m :=
by
  sorry

end quadratic_inequality_solution_set_l65_65643


namespace solve_for_x_l65_65280

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 = 6 - x) : x = 3 :=
by
  sorry

end solve_for_x_l65_65280


namespace find_numbers_l65_65078

theorem find_numbers (x : ℚ) (a : ℚ) (b : ℚ) (h₁ : a = 8 * x) (h₂ : b = x^2 - 1) :
  (a * b + a = (2 * x)^3) ∧ (a * b + b = (2 * x - 1)^3) → 
  x = 14 / 13 ∧ a = 112 / 13 ∧ b = 27 / 169 :=
by
  intros h
  sorry

end find_numbers_l65_65078


namespace additional_rows_added_l65_65190

theorem additional_rows_added
  (initial_tiles : ℕ) (initial_rows : ℕ) (initial_columns : ℕ) (new_columns : ℕ) (new_rows : ℕ)
  (h1 : initial_tiles = 48)
  (h2 : initial_rows = 6)
  (h3 : initial_columns = initial_tiles / initial_rows)
  (h4 : new_columns = initial_columns - 2)
  (h5 : new_rows = initial_tiles / new_columns) :
  new_rows - initial_rows = 2 := by sorry

end additional_rows_added_l65_65190


namespace halfway_fraction_l65_65994

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65994


namespace sector_area_sexagesimal_l65_65983

theorem sector_area_sexagesimal (r : ℝ) (n : ℝ) (α_sex : ℝ) (π : ℝ) (two_pi : ℝ):
  r = 4 →
  n = 6000 →
  α_sex = 625 →
  two_pi = 2 * π →
  (1/2 * (α_sex / n * two_pi) * r^2) = (5 * π) / 3 :=
by
  intros
  sorry

end sector_area_sexagesimal_l65_65983


namespace reciprocals_not_arithmetic_sequence_l65_65627

theorem reciprocals_not_arithmetic_sequence 
  (a b c : ℝ) (h : 2 * b = a + c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_neq : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ¬ (1 / a + 1 / c = 2 / b) :=
by
  sorry

end reciprocals_not_arithmetic_sequence_l65_65627


namespace marikas_father_twice_her_age_l65_65681

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l65_65681


namespace solve_system_of_equations_l65_65964

variable (x y z : ℝ)

theorem solve_system_of_equations {x y z : ℝ} :
  (x * (y + z) * (x + y + z) = 1170) ∧
  (y * (z + x) * (x + y + z) = 1008) ∧
  (z * (x + y) * (x + y + z) = 1458) →
  (x = 5) ∧ (y = 4) ∧ (z = 9) :=
begin
  sorry
end

end solve_system_of_equations_l65_65964


namespace range_of_x_coordinate_l65_65357

def is_on_line (A : ℝ × ℝ) : Prop := A.1 + A.2 = 6

def is_on_circle (C : ℝ × ℝ) : Prop := (C.1 - 1)^2 + (C.2 - 1)^2 = 4

def angle_BAC_is_60_degrees (A B C : ℝ × ℝ) : Prop :=
  -- This definition is simplified as an explanation. Angle computation in Lean might be more intricate.
  sorry 

theorem range_of_x_coordinate (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (hA_on_line : is_on_line A)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (h_angle_BAC : angle_BAC_is_60_degrees A B C) :
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
sorry

end range_of_x_coordinate_l65_65357


namespace simplify_fraction_l65_65812

theorem simplify_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a / b)^b := 
by sorry

end simplify_fraction_l65_65812


namespace ordered_pairs_satisfying_condition_l65_65097

theorem ordered_pairs_satisfying_condition : 
  ∃! (pairs : Finset (ℕ × ℕ)),
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 144) ∧ 
    pairs.card = 4 := sorry

end ordered_pairs_satisfying_condition_l65_65097


namespace evaluate_expression_l65_65140

-- Definitions for a and b
def a : Int := 1
def b : Int := -1

theorem evaluate_expression : 
  5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b) + 1 = -17 := by
  -- Simplification steps skipped
  sorry

end evaluate_expression_l65_65140


namespace time_b_is_54_l65_65102

-- Define the time A takes to complete the work
def time_a := 27

-- Define the time B takes to complete the work as twice the time A takes
def time_b := 2 * time_a

-- Prove that B takes 54 days to complete the work
theorem time_b_is_54 : time_b = 54 :=
by
  sorry

end time_b_is_54_l65_65102


namespace determine_f_l65_65887

theorem determine_f (d e f : ℝ) 
  (h_eq : ∀ y : ℝ, (-3) = d * y^2 + e * y + f)
  (h_vertex : ∀ k : ℝ, -1 = d * (3 - k)^2 + e * (3 - k) + f) :
  f = -5 / 2 :=
sorry

end determine_f_l65_65887


namespace length_of_DE_l65_65003

theorem length_of_DE (base : ℝ) (area_ratio : ℝ) (height_ratio : ℝ) :
  base = 18 → area_ratio = 0.09 → height_ratio = 0.3 → DE = 2 :=
by
  sorry

end length_of_DE_l65_65003


namespace percentage_loss_l65_65577

variable (CP SP : ℝ)
variable (HCP : CP = 1600)
variable (HSP : SP = 1408)

theorem percentage_loss (HCP : CP = 1600) (HSP : SP = 1408) : 
  (CP - SP) / CP * 100 = 12 := by
sorry

end percentage_loss_l65_65577


namespace probability_AC_lt_15_l65_65901

open MeasureTheory Probability

-- Conditions definitions
def pointA : ℝ × ℝ := (-12, 0)
def pointB : ℝ × ℝ := (0, 0)
def radiusB : ℝ := 8
def radiusA : ℝ := 15
def alpha : Set ℝ := Set.Ioo 0 (Real.pi / 2)

-- The probability that the distance AC is less than 15 cm.
theorem probability_AC_lt_15 :
  -- Given alpha ∈ (0, π/2)
  sorry -- placeholder for event definition
  
  -- The statement to be proved
  (P(AC < 15)) = 1 / 3 :=
sorry

end probability_AC_lt_15_l65_65901


namespace alpha_when_beta_neg4_l65_65142

theorem alpha_when_beta_neg4 :
  (∀ (α β : ℝ), (β ≠ 0) → α = 5 → β = 2 → α * β^2 = α * 4) →
   ∃ (α : ℝ), α = 5 → ∃ β, β = -4 → α = 5 / 4 :=
  by
    intros h
    use 5 / 4
    sorry

end alpha_when_beta_neg4_l65_65142


namespace Ricciana_run_distance_l65_65942

def Ricciana_jump : ℕ := 4

def Margarita_run : ℕ := 18

def Margarita_jump (Ricciana_jump : ℕ) : ℕ := 2 * Ricciana_jump - 1

def Margarita_total_distance (Margarita_run Margarita_jump : ℕ) : ℕ := Margarita_run + Margarita_jump

def Ricciana_total_distance (Ricciana_run Ricciana_jump : ℕ) : ℕ := Ricciana_run + Ricciana_jump

theorem Ricciana_run_distance (R : ℕ) 
  (Ricciana_total : ℕ := R + Ricciana_jump) 
  (Margarita_total : ℕ := Margarita_run + Margarita_jump Ricciana_jump) 
  (h : Margarita_total = Ricciana_total + 1) : 
  R = 20 :=
by
  sorry

end Ricciana_run_distance_l65_65942


namespace set_C_is_pythagorean_triple_l65_65900

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem set_C_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
sorry

end set_C_is_pythagorean_triple_l65_65900


namespace non_chocolate_candy_count_l65_65517

theorem non_chocolate_candy_count (total_candy : ℕ) (total_bags : ℕ) 
  (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) (each_bag_pieces : ℕ) 
  (non_chocolate_bags : ℕ) : 
  total_candy = 63 ∧ 
  total_bags = 9 ∧ 
  chocolate_hearts_bags = 2 ∧ 
  chocolate_kisses_bags = 3 ∧ 
  total_candy / total_bags = each_bag_pieces ∧ 
  total_bags - (chocolate_hearts_bags + chocolate_kisses_bags) = non_chocolate_bags ∧ 
  non_chocolate_bags * each_bag_pieces = 28 :=
by
  -- use "sorry" to skip the proof
  sorry

end non_chocolate_candy_count_l65_65517


namespace used_mystery_books_l65_65194

theorem used_mystery_books (total_books used_adventure_books new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13.0)
  (h3 : new_crime_books = 15.0) :
  total_books - (used_adventure_books + new_crime_books) = 17.0 := by
  sorry

end used_mystery_books_l65_65194


namespace three_dice_probability_even_l65_65160

/-- A die is represented by numbers from 1 to 6. -/
def die := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

/-- Define an event where three dice are thrown, and we check if their sum is even. -/
def three_dice_sum_even (d1 d2 d3 : die) : Prop :=
  (d1.val + d2.val + d3.val) % 2 = 0

/-- Define the probability that a single die shows an odd number. -/
def prob_odd := 1 / 2

/-- Define the probability that a single die shows an even number. -/
def prob_even := 1 / 2

/-- Define the total probability for the sum of three dice to be even. -/
def prob_sum_even : ℚ :=
  prob_even ^ 3 + (3 * prob_odd ^ 2 * prob_even)

theorem three_dice_probability_even :
  prob_sum_even = 1 / 2 :=
by
  sorry

end three_dice_probability_even_l65_65160


namespace sequence_product_mod_4_l65_65457

theorem sequence_product_mod_4 :
  let seq := list.range 10 |>.map (fun n => 3 + 10 * n)
  (seq.product) % 4 = 1 := sorry

end sequence_product_mod_4_l65_65457


namespace transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l65_65536

-- Define the polynomial transformation
def transform_polynomial (x : ℝ) : ℝ := x^2 + 8 * x - 1

-- Transformation problem
theorem transform_polynomial_eq (x m n : ℝ) :
  (x + 4)^2 - 17 = transform_polynomial x := 
sorry

-- Define the polynomial for correction
def factor_polynomial (x : ℝ) : ℝ := x^2 - 3 * x - 40

-- Factoring correction problem
theorem correct_factorization (x : ℝ) :
  factor_polynomial x = (x + 5) * (x - 8) := 
sorry

-- Define the polynomial for the positivity proof
def positive_polynomial (x y : ℝ) : ℝ := x^2 + y^2 - 2 * x - 4 * y + 16

-- Positive polynomial proof
theorem positive_polynomial_gt_zero (x y : ℝ) :
  positive_polynomial x y > 0 := 
sorry

end transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l65_65536


namespace ferris_wheel_time_10_seconds_l65_65441

noncomputable def time_to_reach_height (R : ℝ) (T : ℝ) (h : ℝ) : ℝ :=
  let ω := 2 * Real.pi / T
  let t := (Real.arcsin (h / R - 1)) / ω
  t

theorem ferris_wheel_time_10_seconds :
  time_to_reach_height 30 120 15 = 10 :=
by
  sorry

end ferris_wheel_time_10_seconds_l65_65441


namespace masha_happy_max_l65_65389

/-- Masha has 2021 weights, all with unique masses. She places weights one at a 
time on a two-pan balance scale without removing previously placed weights. 
Every time the scale balances, Masha feels happy. Prove that the maximum number 
of times she can find the scales in perfect balance is 673. -/
theorem masha_happy_max (weights : Finset ℕ) (h_unique : weights.card = 2021) : 
  ∃ max_happy_times : ℕ, max_happy_times = 673 := 
sorry

end masha_happy_max_l65_65389


namespace marble_probability_l65_65751

theorem marble_probability :
  let total_ways := (Nat.choose 6 4)
  let favorable_ways := 
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1)
  let probability := (favorable_ways : ℚ) / total_ways
  probability = 4 / 5 := by
  sorry

end marble_probability_l65_65751


namespace chocolate_game_winner_l65_65558

-- Definitions of conditions for the problem
def chocolate_bar (m n : ℕ) := m * n

-- Theorem statement with conditions and conclusion
theorem chocolate_game_winner (m n : ℕ) (h1 : chocolate_bar m n = 48) : 
  ( ∃ first_player_wins : true, true) :=
by sorry

end chocolate_game_winner_l65_65558


namespace sin_cos_sum_inequality_l65_65654

theorem sin_cos_sum_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := 
sorry

end sin_cos_sum_inequality_l65_65654


namespace integral_2x_minus_1_eq_6_l65_65789

noncomputable def definite_integral_example : ℝ :=
  ∫ x in (0:ℝ)..(3:ℝ), (2 * x - 1)

theorem integral_2x_minus_1_eq_6 : definite_integral_example = 6 :=
by
  sorry

end integral_2x_minus_1_eq_6_l65_65789


namespace min_c_value_l65_65913

theorem min_c_value (c : ℝ) : (-c^2 + 9 * c - 14 >= 0) → (c >= 2) :=
by {
  sorry
}

end min_c_value_l65_65913


namespace convex_quadrilaterals_from_12_points_l65_65556

theorem convex_quadrilaterals_from_12_points : 
  ∃ (s : Finset (Fin 12)), s.card = 495 :=
by 
  let points := Finset.univ : Finset (Fin 12)
  have h1 : Finset.card points = 12 := Finset.card_fin 12
  let quadrilaterals := points.powersetLen 4
  have h2 : Finset.card quadrilaterals = 495
    := by sorry -- proof goes here
  exact ⟨quadrilaterals, h2⟩

end convex_quadrilaterals_from_12_points_l65_65556


namespace simplify_trig_expression_trig_identity_l65_65438

-- Defining the necessary functions
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

-- First problem
theorem simplify_trig_expression (α : ℝ) :
  (sin (2 * Real.pi - α) * sin (Real.pi + α) * cos (-Real.pi - α)) / (sin (3 * Real.pi - α) * cos (Real.pi - α)) = sin α :=
sorry

-- Second problem
theorem trig_identity (x : ℝ) (hx : cos x ≠ 0) (hx' : 1 - sin x ≠ 0) :
  (cos x / (1 - sin x)) = ((1 + sin x) / cos x) :=
sorry

end simplify_trig_expression_trig_identity_l65_65438


namespace ones_digit_of_largest_power_of_three_dividing_27_factorial_l65_65343

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  let k := (27 / 3) + (27 / 9) + (27 / 27)
  let x := 3^k
  (x % 10) = 3 := by
  sorry

end ones_digit_of_largest_power_of_three_dividing_27_factorial_l65_65343


namespace difference_in_overlap_l65_65434

variable (total_students : ℕ) (geometry_students : ℕ) (biology_students : ℕ)

theorem difference_in_overlap
  (h1 : total_students = 232)
  (h2 : geometry_students = 144)
  (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students;
  let min_overlap := geometry_students + biology_students - total_students;
  max_overlap - min_overlap = 88 :=
by 
  sorry

end difference_in_overlap_l65_65434


namespace b_range_l65_65823

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - 6*b*x + 3*b

theorem b_range (b : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ is_local_min (f x b)) → 0 < b ∧ b < 1/2 := by
  sorry

end b_range_l65_65823


namespace polynomial_real_root_l65_65337

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a * x^3 - x^2 + a^2 * x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end polynomial_real_root_l65_65337


namespace soccer_team_total_games_l65_65769

variable (total_games : ℕ)
variable (won_games : ℕ)

-- Given conditions
def team_won_percentage (p : ℝ) := p = 0.60
def team_won_games (w : ℕ) := w = 78

-- The proof goal
theorem soccer_team_total_games 
    (h1 : team_won_percentage 0.60)
    (h2 : team_won_games 78) :
    total_games = 130 :=
sorry

end soccer_team_total_games_l65_65769


namespace cost_price_of_ball_l65_65435

theorem cost_price_of_ball (x : ℕ) (h : 13 * x = 720 + 5 * x) : x = 90 :=
by sorry

end cost_price_of_ball_l65_65435


namespace meaningful_expression_l65_65105

theorem meaningful_expression (x : ℝ) : 
    (x + 2 > 0 ∧ x - 1 ≠ 0) ↔ (x > -2 ∧ x ≠ 1) :=
by
  sorry

end meaningful_expression_l65_65105


namespace dividend_rate_l65_65762

theorem dividend_rate (face_value market_value expected_interest interest_rate : ℝ)
  (h1 : face_value = 52)
  (h2 : expected_interest = 0.12)
  (h3 : market_value = 39)
  : ((expected_interest * market_value) / face_value) * 100 = 9 := by
  sorry

end dividend_rate_l65_65762


namespace max_people_transition_l65_65216

theorem max_people_transition (a : ℕ) (b : ℕ) (c : ℕ) 
  (hA : a = 850 * 6 / 100) (hB : b = 1500 * 42 / 1000) (hC : c = 4536 / 72) :
  max a (max b c) = 63 := 
sorry

end max_people_transition_l65_65216


namespace skyscraper_anniversary_l65_65108

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l65_65108


namespace non_chocolate_candy_count_l65_65516

theorem non_chocolate_candy_count (total_candy : ℕ) (total_bags : ℕ) 
  (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) (each_bag_pieces : ℕ) 
  (non_chocolate_bags : ℕ) : 
  total_candy = 63 ∧ 
  total_bags = 9 ∧ 
  chocolate_hearts_bags = 2 ∧ 
  chocolate_kisses_bags = 3 ∧ 
  total_candy / total_bags = each_bag_pieces ∧ 
  total_bags - (chocolate_hearts_bags + chocolate_kisses_bags) = non_chocolate_bags ∧ 
  non_chocolate_bags * each_bag_pieces = 28 :=
by
  -- use "sorry" to skip the proof
  sorry

end non_chocolate_candy_count_l65_65516


namespace area_of_quadrilateral_EFGH_l65_65852

-- Define the properties of rectangle ABCD and the areas
def rectangle (A B C D : Type) := 
  ∃ (area : ℝ), area = 48

-- Define the positions of the points E, G, F, H
def points_positions (A D C B E G F H : Type) :=
  ∃ (one_third : ℝ) (two_thirds : ℝ), one_third = 1/3 ∧ two_thirds = 2/3

-- Define the area calculation for quadrilateral EFGH
def area_EFGH (area_ABCD : ℝ) (one_third : ℝ) : ℝ :=
  (one_third * one_third) * area_ABCD

-- The proof statement that area of EFGH is 5 1/3 square meters
theorem area_of_quadrilateral_EFGH 
  (A B C D E F G H : Type)
  (area_ABCD : ℝ)
  (one_third : ℝ) :
  rectangle A B C D →
  points_positions A D C B E G F H →
  area_ABCD = 48 →
  one_third = 1/3 →
  area_EFGH area_ABCD one_third = 16/3 :=
by
  intros h1 h2 h3 h4
  have h5 : area_EFGH area_ABCD one_third = 16/3 :=
  sorry
  exact h5

end area_of_quadrilateral_EFGH_l65_65852


namespace no_such_f_exists_l65_65397

theorem no_such_f_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x : ℝ), f (f x) = x^2 - 2 := by
  sorry

end no_such_f_exists_l65_65397


namespace find_PB_l65_65663

noncomputable def PA : ℝ := 5
noncomputable def PT (AB : ℝ) : ℝ := 2 * (AB - PA) + 1
noncomputable def PB (AB : ℝ) : ℝ := PA + AB

theorem find_PB (AB : ℝ) (AB_condition : AB = PB AB - PA) :
  PB AB = (81 + Real.sqrt 5117) / 8 :=
by
  sorry

end find_PB_l65_65663


namespace distance_from_Q_to_AD_l65_65409

-- Define the square $ABCD$ with side length 6
def square_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 6) ∧ B = (6, 6) ∧ C = (6, 0) ∧ D = (0, 0)

-- Define point $N$ as the midpoint of $\overline{CD}$
def midpoint_CD (C D N : ℝ × ℝ) : Prop :=
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the intersection condition of the circles centered at $N$ and $A$
def intersect_circles (N A Q D : ℝ × ℝ) : Prop :=
  (Q = D ∨ (∃ r₁ r₂, (Q.1 - N.1)^2 + Q.2^2 = r₁ ∧ Q.1^2 + (Q.2 - A.2)^2 = r₂))

-- Prove the distance from $Q$ to $\overline{AD}$ equals 12/5
theorem distance_from_Q_to_AD (A B C D N Q : ℝ × ℝ)
  (h_square : square_ABCD A B C D)
  (h_midpoint : midpoint_CD C D N)
  (h_intersect : intersect_circles N A Q D) :
  Q.2 = 12 / 5 :=
sorry

end distance_from_Q_to_AD_l65_65409


namespace will_money_left_l65_65284

def initial_money : ℝ := 74
def sweater_cost : ℝ := 9
def tshirt_cost : ℝ := 11
def shoes_cost : ℝ := 30
def hat_cost : ℝ := 5
def socks_cost : ℝ := 4
def refund_percentage : ℝ := 0.85
def discount_percentage : ℝ := 0.1
def tax_percentage : ℝ := 0.05

-- Total cost before returns and discounts
def total_cost_before : ℝ := 
  sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost

-- Refund for shoes
def shoes_refund : ℝ := refund_percentage * shoes_cost

-- New total cost after refund
def total_cost_after_refund : ℝ := total_cost_before - shoes_refund

-- Total cost of remaining items (excluding shoes)
def remaining_items_cost : ℝ := total_cost_before - shoes_cost

-- Discount on remaining items
def discount : ℝ := discount_percentage * remaining_items_cost

-- New total cost after discount
def total_cost_after_discount : ℝ := total_cost_after_refund - discount

-- Sales tax on the final purchase amount
def sales_tax : ℝ := tax_percentage * total_cost_after_discount

-- Final purchase amount with tax
def final_purchase_amount : ℝ := total_cost_after_discount + sales_tax

-- Money left after the final purchase
def money_left : ℝ := initial_money - final_purchase_amount

theorem will_money_left : money_left = 41.87 := by 
  sorry

end will_money_left_l65_65284


namespace product_of_last_two_digits_l65_65373

theorem product_of_last_two_digits (n : ℤ) (A B : ℤ) :
  (n % 8 = 0) ∧ (A + B = 15) ∧ (n % 10 = B) ∧ (n / 10 % 10 = A) →
  A * B = 54 :=
by
-- Add proof here
sorry

end product_of_last_two_digits_l65_65373


namespace pet_preferences_l65_65580

/-- A store has several types of pets: 20 puppies, 10 kittens, 8 hamsters, and 5 birds.
Alice, Bob, Charlie, and David each want a different kind of pet, with the following preferences:
- Alice does not want a bird.
- Bob does not want a hamster.
- Charlie does not want a kitten.
- David does not want a puppy.
Prove that the number of ways they can choose different types of pets satisfying
their preferences is 791440. -/
theorem pet_preferences :
  let P := 20    -- Number of puppies
  let K := 10    -- Number of kittens
  let H := 8     -- Number of hamsters
  let B := 5     -- Number of birds
  let Alice_options := P + K + H -- Alice does not want a bird
  let Bob_options := P + K + B   -- Bob does not want a hamster
  let Charlie_options := P + H + B -- Charlie does not want a kitten
  let David_options := K + H + B   -- David does not want a puppy
  let Alice_pick := Alice_options
  let Bob_pick := Bob_options - 1
  let Charlie_pick := Charlie_options - 2
  let David_pick := David_options - 3
  Alice_pick * Bob_pick * Charlie_pick * David_pick = 791440 :=
by
  sorry

end pet_preferences_l65_65580


namespace depth_of_second_hole_l65_65567

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let total_man_hours1 := workers1 * hours1
  let rate_of_work := depth1 / total_man_hours1
  let workers2 := 45 + 45
  let hours2 := 6
  let total_man_hours2 := workers2 * hours2
  let depth2 := rate_of_work * total_man_hours2
  depth2 = 45 := by
    sorry

end depth_of_second_hole_l65_65567


namespace solve_for_y_l65_65784

theorem solve_for_y (y : ℝ) (h : (y * (y^5)^(1/4))^(1/3) = 4) : y = 2^(8/3) :=
by {
  sorry
}

end solve_for_y_l65_65784


namespace f_neg_def_l65_65606

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def f_pos_def (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x = x * (1 - x)

theorem f_neg_def (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f_pos_def f) :
  ∀ x : ℝ, x < 0 → f x = x * (1 + x) :=
by
  sorry

end f_neg_def_l65_65606


namespace king_total_payment_l65_65756

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l65_65756


namespace carol_blocks_l65_65460

theorem carol_blocks (x : ℕ) (h : x - 25 = 17) : x = 42 :=
sorry

end carol_blocks_l65_65460


namespace only_statement_4_is_correct_l65_65745

-- Defining conditions for input/output statement correctness
def INPUT_statement_is_correct (s : String) : Prop :=
  s = "INPUT x=, 2"

def PRINT_statement_is_correct (s : String) : Prop :=
  s = "PRINT 20, 4"

-- List of statements
def statement_1 := "INPUT a; b; c"
def statement_2 := "PRINT a=1"
def statement_3 := "INPUT x=2"
def statement_4 := "PRINT 20, 4"

-- Predicate for correctness of statements
def statement_is_correct (s : String) : Prop :=
  (s = statement_4) ∧
  ¬(s = statement_1 ∨ s = statement_2 ∨ s = statement_3)

-- Theorem to prove that only statement 4 is correct
theorem only_statement_4_is_correct :
  ∀ s : String, (statement_is_correct s) ↔ (s = statement_4) :=
by
  intros s
  sorry

end only_statement_4_is_correct_l65_65745


namespace find_uv_l65_65075

open Real

def vec1 : ℝ × ℝ := (3, -2)
def vec2 : ℝ × ℝ := (-1, 2)
def vec3 : ℝ × ℝ := (1, -1)
def vec4 : ℝ × ℝ := (4, -7)
def vec5 : ℝ × ℝ := (-3, 5)

theorem find_uv (u v : ℝ) :
  vec1 + ⟨4 * u, -7 * u⟩ = vec2 + ⟨-3 * v, 5 * v⟩ + vec3 →
  u = 3 / 4 ∧ v = -9 / 4 :=
by
  sorry

end find_uv_l65_65075


namespace credit_card_balance_l65_65670

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end credit_card_balance_l65_65670


namespace amount_returned_l65_65246

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l65_65246


namespace solve_eq1_solve_eq2_l65_65539

noncomputable def eq1 (x : ℝ) : Prop := x - 2 = 4 * (x - 2)^2
noncomputable def eq2 (x : ℝ) : Prop := x * (2 * x + 1) = 8 * x - 3

theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 2 ∨ x = 9 / 4 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 1 / 2 ∨ x = 3 :=
by
  sorry

end solve_eq1_solve_eq2_l65_65539


namespace cost_to_paint_cube_l65_65004

theorem cost_to_paint_cube :
  let cost_per_kg := 50
  let coverage_per_kg := 20
  let side_length := 20
  let surface_area := 6 * (side_length * side_length)
  let amount_of_paint := surface_area / coverage_per_kg
  let total_cost := amount_of_paint * cost_per_kg
  total_cost = 6000 :=
by
  sorry

end cost_to_paint_cube_l65_65004


namespace subtract_two_percent_is_multiplying_l65_65170

theorem subtract_two_percent_is_multiplying (a : ℝ) : (a - 0.02 * a) = 0.98 * a := by
  sorry

end subtract_two_percent_is_multiplying_l65_65170


namespace annual_rent_per_sqft_l65_65708

theorem annual_rent_per_sqft
  (length width monthly_rent : ℕ)
  (H_length : length = 10)
  (H_width : width = 8)
  (H_monthly_rent : monthly_rent = 2400) :
  (12 * monthly_rent) / (length * width) = 360 := by
  sorry

end annual_rent_per_sqft_l65_65708


namespace product_of_symmetric_complex_numbers_l65_65256

def z1 : ℂ := 1 + 2 * Complex.I

def z2 : ℂ := -1 + 2 * Complex.I

theorem product_of_symmetric_complex_numbers :
  z1 * z2 = -5 :=
by 
  sorry

end product_of_symmetric_complex_numbers_l65_65256


namespace chromium_percentage_in_new_alloy_l65_65511

noncomputable def percentage_chromium_new_alloy (w1 w2 p1 p2 : ℝ) : ℝ :=
  ((p1 * w1 + p2 * w2) / (w1 + w2)) * 100

theorem chromium_percentage_in_new_alloy :
  percentage_chromium_new_alloy 15 35 0.12 0.10 = 10.6 :=
by
  sorry

end chromium_percentage_in_new_alloy_l65_65511


namespace major_axis_length_l65_65297

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (hr : r = 2) 
  (h_minor : minor_axis = 2 * r)
  (h_major : major_axis = 1.25 * minor_axis) :
  major_axis = 5 :=
by
  sorry

end major_axis_length_l65_65297


namespace mehki_age_l65_65390

variable (Mehki Jordyn Zrinka : ℕ)

axiom h1 : Mehki = Jordyn + 10
axiom h2 : Jordyn = 2 * Zrinka
axiom h3 : Zrinka = 6

theorem mehki_age : Mehki = 22 := by
  -- sorry to skip the proof
  sorry

end mehki_age_l65_65390


namespace arithmetic_seq_a9_l65_65513

theorem arithmetic_seq_a9 (a : ℕ → ℤ) (h1 : a 3 = 3) (h2 : a 6 = 24) : a 9 = 45 :=
by
  -- Proof goes here
  sorry

end arithmetic_seq_a9_l65_65513


namespace part_one_part_two_l65_65084

namespace ProofProblem

def setA (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def setB := {x : ℝ | 0 < x ∧ x < 1}

theorem part_one (a : ℝ) (h : a = 1/2) : 
  setA a ∩ setB = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

theorem part_two (a : ℝ) (h_subset : setB ⊆ setA a) : 
  0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end ProofProblem

end part_one_part_two_l65_65084


namespace ben_mms_count_l65_65056

theorem ben_mms_count (S M : ℕ) (hS : S = 50) (h_diff : S = M + 30) : M = 20 := by
  sorry

end ben_mms_count_l65_65056


namespace democrats_and_republicans_seating_l65_65750

theorem democrats_and_republicans_seating : 
  let n := 6
  let factorial := Nat.factorial
  let arrangements := (factorial n) * (factorial n)
  let circular_table := 1
  arrangements * circular_table = 518400 :=
by 
  sorry

end democrats_and_republicans_seating_l65_65750


namespace graph_of_equation_is_line_and_hyperbola_l65_65782

theorem graph_of_equation_is_line_and_hyperbola :
  ∀ (x y : ℝ), ((x^2 - 1) * (x + y) = y^2 * (x + y)) ↔ (y = -x) ∨ ((x + y) * (x - y) = 1) := by
  intro x y
  sorry

end graph_of_equation_is_line_and_hyperbola_l65_65782


namespace solve_inequality_l65_65204

def p (x : ℝ) : ℝ := x^2 - 5*x + 3

theorem solve_inequality (x : ℝ) : 
  abs (p x) < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
sorry

end solve_inequality_l65_65204


namespace probability_Q_within_two_units_l65_65300

noncomputable def probability_within_two_units_of_origin (s : set (ℝ × ℝ)) (circle_center : ℝ × ℝ) (radius : ℝ) : ℝ :=
  let area_square := 6 * 6 in
  let area_circle := π * radius^2 in
  area_circle / area_square

theorem probability_Q_within_two_units 
  (Q : set (ℝ × ℝ)) 
  (center_origin : (0, 0) = ⟨0, 0⟩)
  (radius_two : ∃ (circle_center : ℝ × ℝ), circle_center = (0, 0) ∧ radius = 2)
  (square_with_vertices : Q = {p : ℝ × ℝ | -3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3}) :
  probability_within_two_units_of_origin Q (0, 0) 2 = π / 9 :=
by
  sorry

end probability_Q_within_two_units_l65_65300


namespace time_to_cover_length_l65_65034

-- Definitions from conditions
def escalator_speed : Real := 15 -- ft/sec
def escalator_length : Real := 180 -- feet
def person_speed : Real := 3 -- ft/sec

-- Combined speed definition
def combined_speed : Real := escalator_speed + person_speed

-- Lean theorem statement proving the time taken
theorem time_to_cover_length : escalator_length / combined_speed = 10 := by
  sorry

end time_to_cover_length_l65_65034


namespace area_of_rectangular_region_l65_65892

-- Mathematical Conditions
variables (a b c d : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

-- Lean 4 Statement of the proof problem
theorem area_of_rectangular_region :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c :=
by sorry

end area_of_rectangular_region_l65_65892


namespace total_earnings_correct_l65_65846

-- Define the weekly earnings and the duration of the harvest.
def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

-- Theorems to state the problem requiring a proof.
theorem total_earnings_correct : (weekly_earnings * harvest_duration = 1216) := 
by
  sorry -- Proof is not required.

end total_earnings_correct_l65_65846


namespace leak_out_time_l65_65444

theorem leak_out_time (T_A T_full : ℝ) (h1 : T_A = 16) (h2 : T_full = 80) :
  ∃ T_B : ℝ, (1 / T_A - 1 / T_B = 1 / T_full) ∧ T_B = 80 :=
by {
  sorry
}

end leak_out_time_l65_65444


namespace middle_number_l65_65018

theorem middle_number (x y z : ℤ) 
  (h1 : x + y = 21)
  (h2 : x + z = 25)
  (h3 : y + z = 28)
  (h4 : x < y)
  (h5 : y < z) : 
  y = 12 :=
sorry

end middle_number_l65_65018


namespace tan_of_acute_angle_l65_65103

open Real

theorem tan_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 2 * sin (α - 15 * π / 180) - 1 = 0) : tan α = 1 :=
by
  sorry

end tan_of_acute_angle_l65_65103


namespace max_n_base_10_l65_65412

theorem max_n_base_10:
  ∃ (A B C n: ℕ), (A < 5 ∧ B < 5 ∧ C < 5) ∧
                 (n = 25 * A + 5 * B + C) ∧ (n = 81 * C + 9 * B + A) ∧ 
                 (∀ (A' B' C' n': ℕ), 
                 (A' < 5 ∧ B' < 5 ∧ C' < 5) ∧ (n' = 25 * A' + 5 * B' + C') ∧ 
                 (n' = 81 * C' + 9 * B' + A') → n' ≤ n) →
  n = 111 :=
by {
    sorry
}

end max_n_base_10_l65_65412


namespace integer_roots_p_l65_65826

theorem integer_roots_p (p x1 x2 : ℤ) (h1 : x1 * x2 = p + 4) (h2 : x1 + x2 = -p) : p = 8 ∨ p = -4 := 
sorry

end integer_roots_p_l65_65826


namespace equal_sum_sequence_a18_l65_65973

def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_sequence_a18 (a : ℕ → ℕ) (h : equal_sum_sequence a 5) (h1 : a 1 = 2) : a 18 = 3 :=
  sorry

end equal_sum_sequence_a18_l65_65973


namespace deadlift_weight_loss_is_200_l65_65660

def initial_squat : ℕ := 700
def initial_bench : ℕ := 400
def initial_deadlift : ℕ := 800
def lost_squat_percent : ℕ := 30
def new_total : ℕ := 1490

theorem deadlift_weight_loss_is_200 : initial_deadlift - (new_total - ((initial_squat * (100 - lost_squat_percent)) / 100 + initial_bench)) = 200 :=
by
  sorry

end deadlift_weight_loss_is_200_l65_65660


namespace tangent_circle_distance_relation_l65_65041

theorem tangent_circle_distance_relation
  (A P Q : Point)
  (u v w : ℝ)
  (h1 : Tangent (CircleThrough [P, Q]) (LineThrough P Q))
  (h2 : dist (Foot A (LineThrough P Q)) A = w)
  (h3 : dist P (LineThrough P Q) = u)
  (h4 : dist Q (LineThrough P Q) = v) :
  (u * v) / w^2 = Real.sin (AngleBetween (RayFrom A P) (RayFrom A Q) / 2)^2 := 
sorry

end tangent_circle_distance_relation_l65_65041


namespace anne_distance_l65_65934

theorem anne_distance (speed time : ℕ) (h_speed : speed = 2) (h_time : time = 3) : 
  (speed * time) = 6 := by
  sorry

end anne_distance_l65_65934


namespace find_a_plus_b_l65_65371

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0) (h : x = a + Real.sqrt b) 
  (hx : x^2 + 3 * x + ↑(3) / x + 1 / x^2 = 30) : 
  a + b = 5 := 
sorry

end find_a_plus_b_l65_65371


namespace Francie_remaining_money_l65_65347

theorem Francie_remaining_money :
  let weekly_allowance_8_weeks : ℕ := 5 * 8
  let weekly_allowance_6_weeks : ℕ := 6 * 6
  let cash_gift : ℕ := 20
  let initial_total_savings := weekly_allowance_8_weeks + weekly_allowance_6_weeks + cash_gift

  let investment_amount : ℕ := 10
  let expected_return_investment_1 : ℚ := 0.05 * 10
  let expected_return_investment_2 : ℚ := (0.5 * 0.10 * 10) + (0.5 * 0.02 * 10)
  let best_investment_return := max expected_return_investment_1 expected_return_investment_2
  let final_savings_after_investment : ℚ := initial_total_savings - investment_amount + best_investment_return

  let amount_for_clothes : ℚ := final_savings_after_investment / 2
  let remaining_after_clothes := final_savings_after_investment - amount_for_clothes
  let cost_of_video_game : ℕ := 35
  
  remaining_after_clothes.sub cost_of_video_game = 8.30 :=
by
  intros
  sorry

end Francie_remaining_money_l65_65347


namespace faucet_leakage_volume_l65_65329

def leakage_rate : ℝ := 0.1
def time_seconds : ℝ := 14400
def expected_volume : ℝ := 1.4 * 10^3

theorem faucet_leakage_volume : 
  leakage_rate * time_seconds = expected_volume := 
by
  -- proof
  sorry

end faucet_leakage_volume_l65_65329


namespace additional_regular_gift_bags_needed_l65_65459

-- Defining the conditions given in the question
def confirmed_guests : ℕ := 50
def additional_guests_70pc : ℕ := 30
def additional_guests_40pc : ℕ := 15
def probability_70pc : ℚ := 0.7
def probability_40pc : ℚ := 0.4
def extravagant_bags_prepared : ℕ := 10
def special_bags_prepared : ℕ := 25
def regular_bags_prepared : ℕ := 20

-- Defining the expected number of additional guests based on probabilities
def expected_guests_70pc : ℚ := additional_guests_70pc * probability_70pc
def expected_guests_40pc : ℚ := additional_guests_40pc * probability_40pc

-- Defining the total expected guests including confirmed guests and expected additional guests
def total_expected_guests : ℚ := confirmed_guests + expected_guests_70pc + expected_guests_40pc

-- Defining the problem statement in Lean, proving the additional regular gift bags needed
theorem additional_regular_gift_bags_needed : 
  total_expected_guests = 77 → regular_bags_prepared = 20 → 22 = 22 :=
by
  sorry

end additional_regular_gift_bags_needed_l65_65459


namespace length_GH_l65_65239

theorem length_GH (AB BC : ℝ) (hAB : AB = 10) (hBC : BC = 5) (DG DH GH : ℝ)
  (hDG : DG = DH) (hArea_DGH : 1 / 2 * DG * DH = 1 / 5 * (AB * BC)) :
  GH = 2 * Real.sqrt 10 :=
by
  sorry

end length_GH_l65_65239


namespace nat_n_divisibility_cond_l65_65917

theorem nat_n_divisibility_cond (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end nat_n_divisibility_cond_l65_65917


namespace total_candy_eaten_by_bobby_l65_65055

-- Definitions based on the problem conditions
def candy_eaten_by_bobby_round1 : ℕ := 28
def candy_eaten_by_bobby_round2 : ℕ := 42
def chocolate_eaten_by_bobby : ℕ := 63

-- Define the statement to prove
theorem total_candy_eaten_by_bobby : 
  candy_eaten_by_bobby_round1 + candy_eaten_by_bobby_round2 + chocolate_eaten_by_bobby = 133 :=
  by
  -- Skipping the proof itself
  sorry

end total_candy_eaten_by_bobby_l65_65055


namespace part_I_part_II_l65_65799

open Real

noncomputable def alpha : ℝ := sorry

def OA := (sin alpha, 1)
def OB := (cos alpha, 0)
def OC := (- sin alpha, 2)

def P : ℝ × ℝ := (2 * cos alpha - sin alpha, 1)

-- Condition for collinearity of O, P, and C
def collinear (O P C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (P.1 = k * O.1) ∧ (P.2 = k * O.2) ∧ (C.1 = k * O.1) ∧ (C.2 = k * O.2)

theorem part_I (hcollinear : collinear (0, 0) P OC) : tan alpha = 4 / 3 := by
  sorry

theorem part_II (h_tan_alpha : tan alpha = 4 / 3) : 
  (sin (2 * alpha) + sin alpha) / (2 * cos (2 * alpha) + 2 * sin alpha^2 + cos alpha) + sin (2 * alpha) = 172 / 75 := 
by
  sorry

end part_I_part_II_l65_65799


namespace minimum_moves_to_determine_polynomial_l65_65692

-- Define quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define conditions as per the given problem
variables {f g : ℝ → ℝ}
def is_quadratic (p : ℝ → ℝ) := ∃ a b c : ℝ, ∀ x : ℝ, p x = quadratic a b c x

axiom f_is_quadratic : is_quadratic f
axiom g_is_quadratic : is_quadratic g

-- Define the main problem statement
theorem minimum_moves_to_determine_polynomial (n : ℕ) :
  (∀ (t : ℕ → ℝ), (∀ m ≤ n, (f (t m) = g (t m)) ∨ (f (t m) ≠ g (t m))) →
  (∃ a b c: ℝ, ∀ x: ℝ, f x = quadratic a b c x ∨ g x = quadratic a b c x)) ↔ n = 8 :=
sorry -- Proof is omitted

end minimum_moves_to_determine_polynomial_l65_65692


namespace positive_difference_l65_65550

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 14) : y - x = 9.714 :=
sorry

end positive_difference_l65_65550


namespace common_chord_diameter_circle_eqn_l65_65093
noncomputable theory

-- Mathematical conditions and the given circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * a * x = 0
def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * b * y = 0

-- The derived equation of the circle whose diameter is the common chord
def desired_circle_eqn (a b x y : ℝ) : Prop := 
  (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0

-- Prove the two circles and their conditions result in the desired circle equation
theorem common_chord_diameter_circle_eqn (a b : ℝ) (hb : b ≠ 0) : 
  ∃ x y : ℝ, circle1 a x y ∧ circle2 b x y → desired_circle_eqn a b x y :=
by
  intro x y h
  sorry


end common_chord_diameter_circle_eqn_l65_65093


namespace crayons_per_box_l65_65328

theorem crayons_per_box (total_crayons : ℕ) (total_boxes : ℕ)
  (h1 : total_crayons = 321)
  (h2 : total_boxes = 45) :
  (total_crayons / total_boxes) = 7 :=
by
  sorry

end crayons_per_box_l65_65328


namespace average_speed_of_planes_l65_65158

-- Definitions for the conditions
def num_passengers_plane1 : ℕ := 50
def num_passengers_plane2 : ℕ := 60
def num_passengers_plane3 : ℕ := 40
def base_speed : ℕ := 600
def speed_reduction_per_passenger : ℕ := 2

-- Calculate speeds of each plane according to given conditions
def speed_plane1 := base_speed - num_passengers_plane1 * speed_reduction_per_passenger
def speed_plane2 := base_speed - num_passengers_plane2 * speed_reduction_per_passenger
def speed_plane3 := base_speed - num_passengers_plane3 * speed_reduction_per_passenger

-- Calculate the total speed and average speed
def total_speed := speed_plane1 + speed_plane2 + speed_plane3
def average_speed := total_speed / 3

-- The theorem to prove the average speed is 500 MPH
theorem average_speed_of_planes : average_speed = 500 := by
  sorry

end average_speed_of_planes_l65_65158


namespace arithmetic_sequence_problem_l65_65804

variable (a_2 a_4 a_3 : ℤ)

theorem arithmetic_sequence_problem (h : a_2 + a_4 = 16) : a_3 = 8 :=
by
  -- The proof is not needed as per the instructions
  sorry

end arithmetic_sequence_problem_l65_65804


namespace triangle_is_isosceles_l65_65828

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop := ∃ (s : ℝ), a = s ∧ b = s

theorem triangle_is_isosceles 
  (A B C a b c : ℝ) 
  (h_sides_angles : a = c ∧ b = c) 
  (h_cos_eq : a * Real.cos B = b * Real.cos A) : 
  is_isosceles_triangle A B C a b c := 
by 
  sorry

end triangle_is_isosceles_l65_65828


namespace variance_red_ball_draws_l65_65967

-- Suppose there are two red balls and one black ball in a bag.
-- Drawing with replacement is performed three times.
-- Let X be the number of times a red ball is drawn in these three attempts.
-- Each ball has an equal probability of being drawn, and each draw is independent of the others.
-- Prove that the variance D(X) is 2/3.

noncomputable def variance_binom_three_trials : ℚ :=
  let p := 2 / 3 in
  let n := 3 in
  n * p * (1 - p)

theorem variance_red_ball_draws :
  variance_binom_three_trials = 2 / 3 :=
by
  -- Placeholder for the proof
  sorry

end variance_red_ball_draws_l65_65967


namespace am_hm_inequality_l65_65951

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : ℝ :=
  (a + b + c) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem am_hm_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  smallest_possible_value a b c d h1 h2 h3 h4 ≥ 9 / 2 :=
by
  sorry

end am_hm_inequality_l65_65951


namespace walk_to_cafe_and_back_time_l65_65815

theorem walk_to_cafe_and_back_time 
  (t_p : ℝ) (d_p : ℝ) (half_dp : ℝ) (pace : ℝ)
  (h1 : t_p = 30) 
  (h2 : d_p = 3) 
  (h3 : half_dp = d_p / 2) 
  (h4 : pace = t_p / d_p) :
  2 * half_dp * pace = 30 :=
by 
  sorry

end walk_to_cafe_and_back_time_l65_65815


namespace opposite_of_neg3_l65_65722

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l65_65722


namespace shirts_and_pants_neither_plaid_nor_purple_l65_65703

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end shirts_and_pants_neither_plaid_nor_purple_l65_65703


namespace probability_of_one_in_pascal_rows_l65_65327

theorem probability_of_one_in_pascal_rows (n : ℕ) (h : n = 20) : 
  let total_elements := (n * (n + 1)) / 2,
      ones := 1 + 2 * (n - 1) in
  (ones / total_elements : ℚ) = 39 / 210 :=
by
  sorry

end probability_of_one_in_pascal_rows_l65_65327


namespace orchard_trees_l65_65649

theorem orchard_trees (x p : ℕ) (h : x + p = 480) (h2 : p = 3 * x) : x = 120 ∧ p = 360 :=
by
  sorry

end orchard_trees_l65_65649


namespace min_value_of_f_at_sqrt2_l65_65607

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x)))

theorem min_value_of_f_at_sqrt2 :
  f (Real.sqrt 2) = (11 * Real.sqrt 2) / 6 :=
sorry

end min_value_of_f_at_sqrt2_l65_65607


namespace part1_solution_set_part2_range_of_a_l65_65930

open Real

-- For part (1)
theorem part1_solution_set (x a : ℝ) (h : a = 3) : |2 * x - a| + a ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3 := 
by {
  sorry
}

-- For part (2)
theorem part2_range_of_a (f g : ℝ → ℝ) (hf : ∀ x, f x = |2 * x - a| + a) (hg : ∀ x, g x = |2 * x - 3|) :
  (∀ x, f x + g x ≥ 5) ↔ a ≥ 11 / 3 :=
by {
  sorry
}

end part1_solution_set_part2_range_of_a_l65_65930


namespace roots_sum_of_quadratic_l65_65112

theorem roots_sum_of_quadratic:
  (∃ a b : ℝ, (a ≠ b) ∧ (a * b = 5) ∧ (a + b = 8)) →
  (a + b = 8) :=
by
  sorry

end roots_sum_of_quadratic_l65_65112


namespace cone_lateral_surface_area_l65_65929

theorem cone_lateral_surface_area (l d : ℝ) (h_l : l = 5) (h_d : d = 8) : 
  (π * (d / 2) * l) = 20 * π :=
by
  sorry

end cone_lateral_surface_area_l65_65929


namespace contrapositive_true_l65_65628

theorem contrapositive_true (x : ℝ) : (x^2 - 2*x - 8 ≤ 0 → x ≥ -3) :=
by
  -- Proof omitted
  sorry

end contrapositive_true_l65_65628


namespace transformation_sequence_terminates_l65_65045

-- Define the basic elements involved in the problem.
noncomputable def point := ℝ³

def is_no_four_coplanar (points: finset point) : Prop := sorry -- Exact definition omitted

def partition (s: finset point) : Type :=
  Σ (A B : finset point), (∀ x y, x ∈ A → y ∈ B → x ≠ y)

structure AB_tree (A B : finset point) :=
  (edges : finset (point × point))
  (no_closed_polyline : sorry) -- Exact definition omitted

def transformation (A B : finset point) (t : AB_tree A B) : Prop :=
  ∃ (a1 a2 : point) (b1 b2 : point), a1 ∈ A ∧ a2 ∈ A ∧ b1 ∈ B ∧ b2 ∈ B ∧
  (a1, b1) ∈ t.edges ∧ (a2, b2) ∈ t.edges ∧
  (|a1.to_real - b1.to_real| + |a2.to_real - b2.to_real| > |a1.to_real - b2.to_real| + |a2.to_real - b1.to_real|)

theorem transformation_sequence_terminates {n : ℕ} (points: finset point) (h : is_no_four_coplanar points) 
  (A B : finset point) (p : partition points) (t : AB_tree A B):
  ∀ sequence : ℕ → AB_tree A B, 
  (∀ k, transformation A B (sequence k) → sequence (k+1) = some_transformation (sequence k)) → 
  ∃ K, sequence K = sequence (K + 1) := 
sorry

end transformation_sequence_terminates_l65_65045


namespace negation_of_all_have_trap_consumption_l65_65811

-- Definitions for the conditions
def domestic_mobile_phone : Type := sorry

def has_trap_consumption (phone : domestic_mobile_phone) : Prop := sorry

def all_have_trap_consumption : Prop := ∀ phone : domestic_mobile_phone, has_trap_consumption phone

-- Statement of the problem
theorem negation_of_all_have_trap_consumption :
  ¬ all_have_trap_consumption ↔ ∃ phone : domestic_mobile_phone, ¬ has_trap_consumption phone :=
sorry

end negation_of_all_have_trap_consumption_l65_65811


namespace sqrt_equation_has_solution_l65_65071

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l65_65071


namespace represents_not_much_different_l65_65977

def not_much_different_from (x : ℝ) (c : ℝ) : Prop := x - c ≤ 0

theorem represents_not_much_different {x : ℝ} :
  (not_much_different_from x 2023) = (x - 2023 ≤ 0) :=
by
  sorry

end represents_not_much_different_l65_65977


namespace allowance_fraction_l65_65364

theorem allowance_fraction (a : ℚ) (arcade_fraction : ℚ) (total_allowance : ℚ) (final_spending : ℚ) (remaining_after_arcade : ℚ) (toy_store_spending : ℚ) 
  (h1 : total_allowance = 225 / 100)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : a = (arcade_fraction * total_allowance))
  (h4 : remaining_after_arcade = total_allowance - a)
  (h5 : final_spending = 60 / 100)
  (h6 : toy_store_spending = remaining_after_arcade - final_spending) :
  toy_store_spending / remaining_after_arcade = 1 / 3 :=
by
  sorry

end allowance_fraction_l65_65364


namespace minute_hand_gains_per_hour_l65_65574

theorem minute_hand_gains_per_hour (total_gain : ℕ) (total_hours : ℕ) (gain_by_6pm : total_gain = 63) (hours_from_9_to_6 : total_hours = 9) : (total_gain / total_hours) = 7 :=
by
  -- The proof is not required as per instruction.
  sorry

end minute_hand_gains_per_hour_l65_65574


namespace largest_possible_n_base10_l65_65410

theorem largest_possible_n_base10 :
  ∃ (n A B C : ℕ),
    n = 25 * A + 5 * B + C ∧ 
    n = 81 * C + 9 * B + A ∧ 
    A < 5 ∧ B < 5 ∧ C < 5 ∧ 
    n = 69 :=
by {
  sorry
}

end largest_possible_n_base10_l65_65410


namespace opposite_of_neg3_l65_65720

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l65_65720


namespace necessary_and_sufficient_condition_l65_65665

-- Variables and conditions
variables (a : ℕ) (A B : ℝ)
variable (positive_a : 0 < a)

-- System of equations
def system_has_positive_integer_solutions (x y z : ℕ) : Prop :=
  (x^2 + y^2 + z^2 = (13 * a)^2) ∧ 
  (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
    (1 / 4) * (2 * A + B) * (13 * a)^4)

-- Statement of the theorem
theorem necessary_and_sufficient_condition:
  (∃ (x y z : ℕ), system_has_positive_integer_solutions a A B x y z) ↔ B = 2 * A :=
sorry

end necessary_and_sufficient_condition_l65_65665


namespace polygon_sides_l65_65504

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1080) : n = 8 :=
sorry

end polygon_sides_l65_65504


namespace relatively_prime_ratios_l65_65087

theorem relatively_prime_ratios (r s : ℕ) (h_coprime: Nat.gcd r s = 1) 
  (h_cond: (r : ℝ) / s = 2 * (Real.sqrt 2 + Real.sqrt 10) / (5 * Real.sqrt (3 + Real.sqrt 5))) :
  r = 4 ∧ s = 5 :=
by
  sorry

end relatively_prime_ratios_l65_65087


namespace union_A_B_l65_65350

open Set

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : A ∪ B = {x | x < 2} := 
by sorry

end union_A_B_l65_65350


namespace halfway_fraction_l65_65996

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65996


namespace probability_within_two_units_l65_65299

-- Conditions
def is_in_square (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -3 ∧ Q.1 ≤ 3 ∧ Q.2 ≥ -3 ∧ Q.2 ≤ 3

def is_within_two_units (Q : ℝ × ℝ) : Prop :=
  Q.1 * Q.1 + Q.2 * Q.2 ≤ 4

-- Problem Statement
theorem probability_within_two_units :
  (measure_theory.measure_of {Q : ℝ × ℝ | is_within_two_units Q} / measure_theory.measure_of {Q : ℝ × ℝ | is_in_square Q} = π / 9) := by
  sorry

end probability_within_two_units_l65_65299


namespace functional_equation_solution_l65_65335

theorem functional_equation_solution {f : ℚ → ℚ} :
  (∀ x y z t : ℚ, x < y ∧ y < z ∧ z < t ∧ (y - x) = (z - y) ∧ (z - y) = (t - z) →
    f x + f t = f y + f z) → 
  ∃ c b : ℚ, ∀ q : ℚ, f q = c * q + b := 
by
  sorry

end functional_equation_solution_l65_65335


namespace average_age_l65_65970

def proportion (x y z : ℕ) : Prop :=  y / x = 3 ∧ z / x = 4

theorem average_age (A B C : ℕ) 
    (h1 : proportion 2 6 8)
    (h2 : A = 15)
    (h3 : B = 45)
    (h4 : C = 60) :
    (A + B + C) / 3 = 40 := 
    by
    sorry

end average_age_l65_65970


namespace cos_double_angle_l65_65483

theorem cos_double_angle (α : ℝ) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : Real.cos α = 1 / 3 :=
sorry

end cos_double_angle_l65_65483


namespace rationalize_denominator_l65_65260

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l65_65260


namespace ellipse_AB_distance_l65_65332

noncomputable def ellipse_distance : ℝ :=
  let a := 4
  let b := 2
  let A := (7, -2)
  let B := (3, 0)
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem ellipse_AB_distance :
  (4 * (x - 3)^2 + 16 * (y + 2)^2 = 64) →
  (A = (7, -2)) →
  (B = (3, 0)) →
  (ellipse_distance = 2 * real.sqrt 5) :=
by
  intros h1 h2 h3
  sorry

end ellipse_AB_distance_l65_65332


namespace a_formula_T_bounds_l65_65363

noncomputable theory
open_locale classical big_operators

-- Definitions of sequences
def a : ℕ → ℝ
| 1 := 2
| (n + 1) := 2^(n + 1)

def b (n : ℕ) : ℝ :=
1 / (Real.log 2 (a n) * Real.log 2 (a (n + 2)))

def T (n : ℕ) : ℝ :=
∑ i in Finset.range (n + 1), b i

-- Theorem statements
theorem a_formula (n : ℕ) (h : 0 < n) : a n = 2^n := 
sorry

theorem T_bounds (n : ℕ) (h : 0 < n) : 1 / 3 ≤ T n ∧ T n < 3 / 4 :=
sorry

end a_formula_T_bounds_l65_65363


namespace range_of_k_l65_65152

theorem range_of_k (k : ℝ) (h : -3 < k ∧ k ≤ 0) : ∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0 :=
sorry

end range_of_k_l65_65152


namespace students_not_pass_l65_65395

theorem students_not_pass (total_students : ℕ) (percentage_passed : ℕ) (students_passed : ℕ) (students_not_passed : ℕ) :
  total_students = 804 →
  percentage_passed = 75 →
  students_passed = total_students * percentage_passed / 100 →
  students_not_passed = total_students - students_passed →
  students_not_passed = 201 :=
by
  intros h1 h2 h3 h4
  sorry

end students_not_pass_l65_65395


namespace f_2015_l65_65666

def f : ℤ → ℤ := sorry

axiom f1 : f 1 = 1
axiom f2 : f 2 = 0
axiom functional_eq (x y : ℤ) : f (x + y) = f x * f (1 - y) + f (1 - x) * f y

theorem f_2015 : f 2015 = 1 ∨ f 2015 = -1 :=
sorry

end f_2015_l65_65666


namespace probability_A_given_B_l65_65829

namespace ProbabilityProof

def total_parts : ℕ := 100
def A_parts_produced : ℕ := 0
def A_parts_qualified : ℕ := 35
def B_parts_produced : ℕ := 60
def B_parts_qualified : ℕ := 50

def event_A (x : ℕ) : Prop := x ≤ B_parts_qualified + A_parts_qualified
def event_B (x : ℕ) : Prop := x ≤ A_parts_produced

-- Formalizing the probability condition P(A | B) = 7/8, logically this should be revised with practical events.
theorem probability_A_given_B : (event_B x → event_A x) := sorry

end ProbabilityProof

end probability_A_given_B_l65_65829


namespace count_distinct_rat_k_l65_65211

theorem count_distinct_rat_k : 
  (∃ N : ℕ, N = 108 ∧ ∀ k : ℚ, abs k < 300 → (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0) →
  (∃! k, abs k < 300 ∧ (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0))) :=
sorry

end count_distinct_rat_k_l65_65211


namespace negative_exp_eq_l65_65453

theorem negative_exp_eq :
  (-2 : ℤ)^3 = (-2 : ℤ)^3 := by
  sorry

end negative_exp_eq_l65_65453


namespace terminal_side_alpha_minus_beta_nonneg_x_axis_l65_65088

theorem terminal_side_alpha_minus_beta_nonneg_x_axis
  (α β : ℝ) (k : ℤ) (h : α = k * 360 + β) : 
  (∃ m : ℤ, α - β = m * 360) := 
sorry

end terminal_side_alpha_minus_beta_nonneg_x_axis_l65_65088


namespace volume_of_pyramid_l65_65215

noncomputable def volume_of_regular_triangular_pyramid (h R : ℝ) : ℝ :=
  (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4

theorem volume_of_pyramid (h R : ℝ) : volume_of_regular_triangular_pyramid h R = (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4 :=
  by sorry

end volume_of_pyramid_l65_65215


namespace find_C_l65_65515

-- Define the sum of interior angles of a triangle
def sum_of_triangle_angles := 180

-- Define the total angles sum in a closed figure formed by multiple triangles
def total_internal_angles := 1080

-- Define the value to prove
def C := total_internal_angles - sum_of_triangle_angles

theorem find_C:
  C = 900 := by
  sorry

end find_C_l65_65515


namespace mean_of_S_permutations_no_consecutive_l65_65950

def permutations (n : ℕ) : Finset (Perm (Fin n)) :=
  Finset.univ

-- Question 1 translation
theorem mean_of_S: 
  let S := permutations 8
  let S_val (σ : Perm (Fin 8)) := 
    σ 0 * σ 1 + σ 2 * σ 3 + σ 4 * σ 5 + σ 6 * σ 7
  (Finset.sum S (λ σ, S_val σ):ℝ) / S.card = 81 :=
sorry

-- Question 2 translation
theorem permutations_no_consecutive:
  let S := permutations 8
  let condition (σ : Perm (Fin 8)) : Prop :=
    ∀ k : Fin 7, σ (k + 1) ≠ k + 1
  (S.filter condition).card = 41787 :=
sorry

end mean_of_S_permutations_no_consecutive_l65_65950


namespace cleaner_steps_l65_65753

theorem cleaner_steps (a b c : ℕ) (h1 : a < 10 ∧ b < 10 ∧ c < 10) (h2 : 100 * a + 10 * b + c > 100 * c + 10 * b + a) (h3 : 100 * a + 10 * b + c + 100 * c + 10 * b + a = 746) :
  (100 * a + 10 * b + c) * 2 = 944 ∨ (100 * a + 10 * b + c) * 2 = 1142 :=
by
  sorry

end cleaner_steps_l65_65753


namespace jenny_collects_20_cans_l65_65948

theorem jenny_collects_20_cans (b c : ℕ) (h1 : 6 * b + 2 * c = 100) (h2 : 10 * b + 3 * c = 160) : c = 20 := 
by sorry

end jenny_collects_20_cans_l65_65948


namespace y_intercept_of_line_l65_65746

theorem y_intercept_of_line (m x y b : ℝ) (h1 : m = 4) (h2 : x = 50) (h3 : y = 300) (h4 : y = m * x + b) : b = 100 := by
  sorry

end y_intercept_of_line_l65_65746


namespace sin_theta_minus_cos_theta_l65_65368

theorem sin_theta_minus_cos_theta (θ : ℝ) (b : ℝ) (hθ_acute : 0 < θ ∧ θ < π / 2) (h_cos2θ : Real.cos (2 * θ) = b) :
  ∃ x, x = Real.sin θ - Real.cos θ ∧ (x = Real.sqrt b ∨ x = -Real.sqrt b) := 
by
  sorry

end sin_theta_minus_cos_theta_l65_65368


namespace area_between_curves_eq_nine_l65_65778

def f (x : ℝ) := 2 * x - x^2 + 3
def g (x : ℝ) := x^2 - 4 * x + 3

theorem area_between_curves_eq_nine :
  ∫ x in (0 : ℝ)..(3 : ℝ), (f x - g x) = 9 := by
  sorry

end area_between_curves_eq_nine_l65_65778


namespace train_speed_l65_65192

theorem train_speed
  (train_length : ℝ) (platform_length : ℝ) (time_seconds : ℝ)
  (h_train_length : train_length = 450)
  (h_platform_length : platform_length = 300.06)
  (h_time : time_seconds = 25) :
  (train_length + platform_length) / time_seconds * 3.6 = 108.01 :=
by
  -- skipping the proof with sorry
  sorry

end train_speed_l65_65192


namespace student_marks_l65_65587

variable (x : ℕ)
variable (passing_marks : ℕ)
variable (max_marks : ℕ := 400)
variable (fail_by : ℕ := 14)

theorem student_marks :
  (passing_marks = 36 * max_marks / 100) →
  (x + fail_by = passing_marks) →
  x = 130 :=
by sorry

end student_marks_l65_65587


namespace project_total_hours_l65_65532

def pat_time (k : ℕ) : ℕ := 2 * k
def mark_time (k : ℕ) : ℕ := k + 120

theorem project_total_hours (k : ℕ) (H1 : 3 * 2 * k = k + 120) :
  k + pat_time k + mark_time k = 216 :=
by
  sorry

end project_total_hours_l65_65532


namespace math_problem_proof_l65_65497

-- Define the conditions for the function f(x)
variables {a b c : ℝ}
variables (ha : a ≠ 0) (h1 : (b/a) > 0) (h2 : (-2 * c/a) > 0) (h3 : (b^2 + 8 * a * c) > 0)

-- Define the statements to be proved based on the conditions
theorem math_problem_proof :
    (a ≠ 0) →
    (b/a > 0) →
    (-2 * c/a > 0) →
    (b^2 + 8*a*c > 0) →
    (ab : (a*b) > 0) ∧    -- B
    ((b^2 + 8*a*c) > 0) ∧ -- C
    (ac : a*c < 0)        -- D
 := by
    intros ha h1 h2 h3
    sorry

end math_problem_proof_l65_65497


namespace day_before_yesterday_l65_65822

theorem day_before_yesterday (day_after_tomorrow_is_monday : String) : String :=
by
  have tomorrow := "Sunday"
  have today := "Saturday"
  exact today

end day_before_yesterday_l65_65822


namespace train_trip_length_l65_65050

theorem train_trip_length (x D : ℝ) (h1 : D > 0) (h2 : x > 0) 
(h3 : 2 + 3 * (D - 2 * x) / (2 * x) + 1 = (x + 240) / x + 1 + 3 * (D - 2 * x - 120) / (2 * x) - 0.5) 
(h4 : 3 + 3 * (D - 2 * x) / (2 * x) = 7) :
  D = 640 :=
by
  sorry

end train_trip_length_l65_65050


namespace max_f_exists_max_f_l65_65707

open Set

variable {α : Type*} [LinearOrderedField α] 

noncomputable def f (x : α) : α := 5 * real.sqrt (x - 1) + real.sqrt (10 - 2 * x)

theorem max_f : ∀ x ∈ Icc (1 : α) 5, f x ≤ 6 * real.sqrt 3 :=
begin
  sorry
end

theorem exists_max_f : ∃ x ∈ Icc (1 : α) 5, f x = 6 * real.sqrt 3 :=
begin
  use (127 / 27 : α),
  split,
  { split; linarith },
  { sorry }
end

end max_f_exists_max_f_l65_65707


namespace Tonya_initial_stamps_l65_65659

theorem Tonya_initial_stamps :
  ∀ (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (jimmy_matchbooks : ℕ) (tonya_remaining_stamps : ℕ),
  stamps_per_match = 12 →
  matches_per_matchbook = 24 →
  jimmy_matchbooks = 5 →
  tonya_remaining_stamps = 3 →
  tonya_remaining_stamps + (jimmy_matchbooks * matches_per_matchbook) / stamps_per_match = 13 := 
by
  intros stamps_per_match matches_per_matchbook jimmy_matchbooks tonya_remaining_stamps
  sorry

end Tonya_initial_stamps_l65_65659


namespace common_chord_eq_l65_65543

theorem common_chord_eq (x y : ℝ) :
  x^2 + y^2 + 2*x = 0 →
  x^2 + y^2 - 4*y = 0 →
  x + 2*y = 0 :=
by
  intros h1 h2
  sorry

end common_chord_eq_l65_65543


namespace sin_double_angle_l65_65482

variable {θ : Real}

theorem sin_double_angle (h : cos θ + sin θ = 7 / 5) : sin (2 * θ) = 24 / 25 :=
by
  sorry

end sin_double_angle_l65_65482


namespace lottery_ticket_might_win_l65_65859

theorem lottery_ticket_might_win (p_win : ℝ) (h : p_win = 0.01) : 
  (∃ (n : ℕ), n = 1 ∧ 0 < p_win ∧ p_win < 1) :=
by 
  sorry

end lottery_ticket_might_win_l65_65859


namespace negation_of_proposition_l65_65709

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l65_65709


namespace angle_AOC_is_45_or_15_l65_65619

theorem angle_AOC_is_45_or_15 (A O B C : Type) (α β γ : ℝ) 
  (h1 : α = 30) (h2 : β = 15) : γ = 45 ∨ γ = 15 :=
sorry

end angle_AOC_is_45_or_15_l65_65619


namespace valid_outfit_choices_l65_65640

def shirts := 6
def pants := 6
def hats := 12
def patterned_hats := 6

theorem valid_outfit_choices : 
  (shirts * pants * hats) - shirts - (patterned_hats * shirts * (pants - 1)) = 246 := by
  sorry

end valid_outfit_choices_l65_65640


namespace quadratic_root_ratio_l65_65981

theorem quadratic_root_ratio {m p q : ℝ} (h₁ : m ≠ 0) (h₂ : p ≠ 0) (h₃ : q ≠ 0)
  (h₄ : ∀ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) →
    (∃ t₁ t₂ : ℝ, t₁ = 3 * s₁ ∧ t₂ = 3 * s₂ ∧ (t₁ + t₂ = -m ∧ t₁ * t₂ = p))) :
  p / q = 27 :=
by
  sorry

end quadratic_root_ratio_l65_65981


namespace cos_identity_15_30_degrees_l65_65057

theorem cos_identity_15_30_degrees (a b : ℝ) (h : b = 2 * a^2 - 1) : 2 * a^2 - b = 1 :=
by
  sorry

end cos_identity_15_30_degrees_l65_65057


namespace jeremy_uncle_money_l65_65835

def total_cost (num_jerseys : Nat) (cost_per_jersey : Nat) (basketball_cost : Nat) (shorts_cost : Nat) : Nat :=
  (num_jerseys * cost_per_jersey) + basketball_cost + shorts_cost

def total_money_given (total_cost : Nat) (money_left : Nat) : Nat :=
  total_cost + money_left

theorem jeremy_uncle_money :
  total_money_given (total_cost 5 2 18 8) 14 = 50 :=
by
  sorry

end jeremy_uncle_money_l65_65835


namespace star_proof_l65_65781

def star (a b : ℕ) : ℕ := 3 + b ^ a

theorem star_proof : star (star 2 1) 4 = 259 :=
by
  sorry

end star_proof_l65_65781


namespace children_boys_count_l65_65872

theorem children_boys_count (girls : ℕ) (total_children : ℕ) (boys : ℕ) 
  (h₁ : girls = 35) (h₂ : total_children = 62) : boys = 27 :=
by
  sorry

end children_boys_count_l65_65872


namespace pentagon_ABEDF_area_l65_65904

theorem pentagon_ABEDF_area (BD_diagonal : ∀ (ABCD : Nat) (BD : Nat),
                            ABCD = BD^2 / 2 → BD = 20) 
                            (BDFE_is_rectangle : ∀ (BDFE : Nat), BDFE = 2 * BD) 
                            : ∃ (area : Nat), area = 300 :=
by
  -- Placeholder for the actual proof
  sorry

end pentagon_ABEDF_area_l65_65904


namespace probability_two_units_of_origin_l65_65305

def square_vertices (x_min x_max y_min y_max : ℝ) :=
  { p : ℝ × ℝ // x_min ≤ p.1 ∧ p.1 ≤ x_max ∧ y_min ≤ p.2 ∧ p.2 ≤ y_max }

def within_radius (r : ℝ) (origin : ℝ × ℝ) (p : ℝ × ℝ) :=
  (p.1 - origin.1)^2 + (p.2 - origin.2)^2 ≤ r^2

noncomputable def probability_within_radius (x_min x_max y_min y_max r : ℝ) : ℝ :=
  let square_area := (x_max - x_min) * (y_max - y_min)
  let circle_area := r^2 * Real.pi
  circle_area / square_area

theorem probability_two_units_of_origin :
  probability_within_radius (-3) 3 (-3) 3 2 = Real.pi / 9 :=
by
  sorry

end probability_two_units_of_origin_l65_65305


namespace reaction_produces_nh3_l65_65341

-- Define the Chemical Equation as a structure
structure Reaction where
  reagent1 : ℕ -- moles of NH4NO3
  reagent2 : ℕ -- moles of NaOH
  product  : ℕ -- moles of NH3

-- Given conditions
def reaction := Reaction.mk 2 2 2

-- Theorem stating that given 2 moles of NH4NO3 and 2 moles of NaOH,
-- the number of moles of NH3 formed is 2 moles.
theorem reaction_produces_nh3 (r : Reaction) (h1 : r.reagent1 = 2)
  (h2 : r.reagent2 = 2) : r.product = 2 := by
  sorry

end reaction_produces_nh3_l65_65341


namespace youtube_likes_l65_65885

theorem youtube_likes (L D : ℕ) 
  (h1 : D = (1 / 2 : ℝ) * L + 100)
  (h2 : D + 1000 = 2600) : 
  L = 3000 := 
by
  sorry

end youtube_likes_l65_65885


namespace arithmetic_sum_S8_proof_l65_65356

-- Definitions of variables and constants
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def a1_condition : a 1 = -40 := sorry
def a6_a10_condition : a 6 + a 10 = -10 := sorry

-- Theorem to prove
theorem arithmetic_sum_S8_proof (a : ℕ → ℝ) (S : ℕ → ℝ)
  (a1 : a 1 = -40)
  (a6a10 : a 6 + a 10 = -10)
  : S 8 = -180 := 
sorry

end arithmetic_sum_S8_proof_l65_65356


namespace length_of_the_train_is_120_l65_65898

noncomputable def train_length (time: ℝ) (speed_km_hr: ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time

theorem length_of_the_train_is_120 :
  train_length 3.569962336897346 121 = 120 := by
  sorry

end length_of_the_train_is_120_l65_65898


namespace spent_on_puzzle_l65_65613

-- Defining all given conditions
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def final_amount : ℕ := 1

-- Define the total money before spending on the puzzle
def total_before_puzzle := initial_money + saved_money - spent_on_comic

-- Prove that the amount spent on the puzzle is $18
theorem spent_on_puzzle : (total_before_puzzle - final_amount) = 18 := 
by {
  sorry
}

end spent_on_puzzle_l65_65613


namespace neg_mul_neg_pos_mul_neg_neg_l65_65200

theorem neg_mul_neg_pos (a b : Int) (ha : a < 0) (hb : b < 0) : a * b > 0 :=
sorry

theorem mul_neg_neg : (-1) * (-3) = 3 := 
by
  have h1 : -1 < 0 := by norm_num
  have h2 : -3 < 0 := by norm_num
  have h_pos := neg_mul_neg_pos (-1) (-3) h1 h2
  linarith

end neg_mul_neg_pos_mul_neg_neg_l65_65200


namespace trig_identity_l65_65471

open Real

theorem trig_identity (theta : ℝ) (h : tan theta = 2) : 
  (sin (π / 2 + theta) - cos (π - theta)) / (sin (π / 2 - theta) - sin (π - theta)) = -2 :=
by
  sorry

end trig_identity_l65_65471


namespace distance_between_A_and_B_l65_65049

noncomputable def time_from_A_to_B (D : ℝ) : ℝ := D / 200

noncomputable def time_from_B_to_A (D : ℝ) : ℝ := time_from_A_to_B D + 3

def condition (D : ℝ) : Prop := 
  D = 100 * (time_from_B_to_A D)

theorem distance_between_A_and_B :
  ∃ D : ℝ, condition D ∧ D = 600 :=
by
  sorry

end distance_between_A_and_B_l65_65049


namespace arithmetic_expression_l65_65331

theorem arithmetic_expression : (-9) + 18 + 2 + (-1) = 10 :=
by 
  sorry

end arithmetic_expression_l65_65331


namespace percent_of_75_of_125_l65_65178

theorem percent_of_75_of_125 : (75 / 125) * 100 = 60 := by
  sorry

end percent_of_75_of_125_l65_65178


namespace average_six_conseq_ints_l65_65261

theorem average_six_conseq_ints (c d : ℝ) (h₁ : d = c + 2.5) :
  (d - 2 + d - 1 + d + d + 1 + d + 2 + d + 3) / 6 = c + 3 :=
by
  sorry

end average_six_conseq_ints_l65_65261


namespace part1_geometric_sequence_part2_sum_of_terms_l65_65012

/- Part 1 -/
theorem part1_geometric_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h₀ : a 1 = 3) 
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2 * a n) 
  (h₂ : ∀ n, 2 ^ b n = a n + 1) :
  ∃ r, ∀ n, b (n + 1) = r * b n ∧ r = 2 :=
by 
  use 2 
  sorry

/- Part 2 -/
theorem part2_sum_of_terms (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₀ : ∀ n, b n = 2 ^ n)
  (h₁ : ∀ n, c n = n / b n + 1) :
  ∀ n, T n = n + 2 - (n + 2) / 2 ^ n :=
by
  sorry

end part1_geometric_sequence_part2_sum_of_terms_l65_65012


namespace find_x_l65_65345

def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

theorem find_x (x : ℝ) (h : otimes (x + 1) (x - 2) = 5) : x = 0 ∨ x = 4 := 
by
  sorry

end find_x_l65_65345


namespace find_eccentricity_l65_65351

-- Define the hyperbola structure
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : 0 < a)
  (b_pos : 0 < b)

-- Define the point P and focus F₁ F₂ relationship
structure PointsRelation (C : Hyperbola) where
  P : ℝ × ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  (distance_condition : dist P F1 = 3 * dist P F2)
  (dot_product_condition : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = C.a^2)

noncomputable def eccentricity (C : Hyperbola) (rel : PointsRelation C) : ℝ :=
  Real.sqrt (1 + (C.b ^ 2) / (C.a ^ 2))

theorem find_eccentricity (C : Hyperbola) (rel : PointsRelation C) : eccentricity C rel = Real.sqrt 2 := by
  sorry

end find_eccentricity_l65_65351


namespace book_cost_l65_65704

theorem book_cost (x : ℕ) (hx1 : 10 * x ≤ 1100) (hx2 : 11 * x > 1200) : x = 110 := 
by
  sorry

end book_cost_l65_65704


namespace solve_x_l65_65346

theorem solve_x (x : ℝ) :
  (5 + 2 * x) / (7 + 3 * x) = (4 + 3 * x) / (9 + 4 * x) ↔
  x = (-5 + Real.sqrt 93) / 2 ∨ x = (-5 - Real.sqrt 93) / 2 :=
by
  sorry

end solve_x_l65_65346


namespace potatoes_cost_l65_65945

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end potatoes_cost_l65_65945


namespace extra_apples_l65_65420

-- Defining the given conditions
def redApples : Nat := 60
def greenApples : Nat := 34
def studentsWantFruit : Nat := 7

-- Defining the theorem to prove the number of extra apples
theorem extra_apples : redApples + greenApples - studentsWantFruit = 87 := by
  sorry

end extra_apples_l65_65420


namespace speed_of_man_upstream_l65_65890

-- Conditions stated as definitions 
def V_m : ℝ := 33 -- Speed of the man in still water
def V_downstream : ℝ := 40 -- Speed of the man rowing downstream

-- Required proof problem
theorem speed_of_man_upstream : V_m - (V_downstream - V_m) = 26 := 
by
  -- the following sorry is a placeholder for the actual proof
  sorry

end speed_of_man_upstream_l65_65890


namespace sin_cos_eq_one_l65_65616

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h1 : x < 2 * Real.pi) :
  sin x + cos x = 1 → x = 0 ∨ x = Real.pi / 2 :=
by
  sorry

end sin_cos_eq_one_l65_65616


namespace number_of_towers_l65_65186

theorem number_of_towers (
  red_cubes : ℕ
  blue_cubes : ℕ
  green_cubes : ℕ
  total_cubes : ℕ
  tower_height : ℕ
) : 
  red_cubes = 3 → 
  blue_cubes = 2 → 
  green_cubes = 5 → 
  total_cubes = 10 → 
  tower_height = 7 → 
  (nat.choose total_cubes tower_height * nat.perms tower_height) / (nat.factorial red_cubes * nat.factorial blue_cubes * nat.factorial (tower_height - (red_cubes + blue_cubes))) = 210 :=
by
  intros,
  sorry

end number_of_towers_l65_65186


namespace sin_double_angle_l65_65481

variable {θ : ℝ}

theorem sin_double_angle (h : cos θ + sin θ = 7/5) : sin (2 * θ) = 24/25 :=
by
  sorry

end sin_double_angle_l65_65481


namespace alcohol_percentage_l65_65179

theorem alcohol_percentage (x : ℝ)
  (h1 : 8 * x / 100 + 2 * 12 / 100 = 22.4 * 10 / 100) : x = 25 :=
by
  -- skip the proof
  sorry

end alcohol_percentage_l65_65179


namespace webinar_end_time_correct_l65_65772

-- Define start time and duration as given conditions
def startTime : Nat := 3*60 + 15  -- 3:15 p.m. in minutes after noon
def duration : Nat := 350         -- duration of the webinar in minutes

-- Define the expected end time in minutes after noon (9:05 p.m. is 9*60 + 5 => 545 minutes after noon)
def endTimeExpected : Nat := 9*60 + 5

-- Statement to prove that the calculated end time matches the expected end time
theorem webinar_end_time_correct : startTime + duration = endTimeExpected :=
by
  sorry

end webinar_end_time_correct_l65_65772


namespace Bryce_raisins_l65_65479

theorem Bryce_raisins (B C : ℚ) (h1 : B = C + 10) (h2 : C = B / 4) : B = 40 / 3 :=
by
 -- The proof goes here, but we skip it for now
 sorry

end Bryce_raisins_l65_65479


namespace number_of_students_in_range_l65_65610

-- Define the basic variables and conditions
variable (a b : ℝ) -- Heights of the rectangles in the histogram

-- Define the total number of surveyed students
def total_students : ℝ := 1500

-- Define the width of each histogram group
def group_width : ℝ := 5

-- State the theorem with the conditions and the expected result
theorem number_of_students_in_range (a b : ℝ) :
    5 * (a + b) * total_students = 7500 * (a + b) :=
by
  -- Proof will be added here
  sorry

end number_of_students_in_range_l65_65610


namespace vertex_of_parabola_l65_65705

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = (x - 6)^2 + 3 ↔ (x = 6 ∧ y = 3)) :=
sorry

end vertex_of_parabola_l65_65705


namespace fish_worth_bags_of_rice_l65_65509

variable (f l a r : ℝ)

theorem fish_worth_bags_of_rice
    (h1 : 5 * f = 3 * l)
    (h2 : l = 6 * a)
    (h3 : 2 * a = r) :
    1 / f = 9 / (5 * r) :=
by
  sorry

end fish_worth_bags_of_rice_l65_65509


namespace cost_of_potatoes_l65_65946

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end cost_of_potatoes_l65_65946


namespace determine_m_value_l65_65928

theorem determine_m_value (m : ℤ) (A : Set ℤ) : 
  A = {1, m + 2, m^2 + 4} → 5 ∈ A → m = 3 ∨ m = 1 := 
by
  sorry

end determine_m_value_l65_65928


namespace geometric_sequence_a8_value_l65_65117

variable {a : ℕ → ℕ}

-- Assuming a is a geometric sequence, provide the condition a_3 * a_9 = 4 * a_4
def geometric_sequence_condition (a : ℕ → ℕ) :=
  (a 3) * (a 9) = 4 * (a 4)

-- Prove that a_8 = 4 under the given condition
theorem geometric_sequence_a8_value (a : ℕ → ℕ) (h : geometric_sequence_condition a) : a 8 = 4 :=
  sorry

end geometric_sequence_a8_value_l65_65117


namespace domain_of_f_l65_65007

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (Real.log x / Real.log 2 - 1))

theorem domain_of_f :
  {x : ℝ | x > 2} = {x : ℝ | x > 0 ∧ Real.log x / Real.log 2 - 1 > 0} := 
by
  sorry

end domain_of_f_l65_65007


namespace inequality_proof_l65_65879

theorem inequality_proof (a b c : ℝ) (h : a * c^2 > b * c^2) (hc2 : c^2 > 0) : a > b :=
sorry

end inequality_proof_l65_65879


namespace wrapping_paper_cost_l65_65096
noncomputable def cost_per_roll (shirt_boxes XL_boxes: ℕ) (cost_total: ℝ) : ℝ :=
  let rolls_for_shirts := shirt_boxes / 5
  let rolls_for_xls := XL_boxes / 3
  let total_rolls := rolls_for_shirts + rolls_for_xls
  cost_total / total_rolls

theorem wrapping_paper_cost : cost_per_roll 20 12 32 = 4 :=
by
  sorry

end wrapping_paper_cost_l65_65096


namespace find_larger_number_l65_65974

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1355) (h2 : L = 6 * S + 15) : L = 1623 :=
sorry

end find_larger_number_l65_65974


namespace lindsey_squat_weight_l65_65129

-- Define the conditions
def num_bands : ℕ := 2
def resistance_per_band : ℤ := 5
def dumbbell_weight : ℤ := 10

-- Define the weight Lindsay will squat
def total_weight : ℤ := num_bands * resistance_per_band + dumbbell_weight

-- State the theorem
theorem lindsey_squat_weight : total_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l65_65129


namespace math_problem_l65_65270

theorem math_problem
  (z : ℝ)
  (hz : z = 80)
  (y : ℝ)
  (hy : y = (1/4) * z)
  (x : ℝ)
  (hx : x = (1/3) * y)
  (w : ℝ)
  (hw : w = x + y + z) :
  x = 20 / 3 ∧ w = 320 / 3 :=
by
  sorry

end math_problem_l65_65270


namespace largest_by_changing_first_digit_l65_65282

def value_with_digit_changed (d : Nat) : Float :=
  match d with
  | 1 => 0.86123
  | 2 => 0.78123
  | 3 => 0.76823
  | 4 => 0.76183
  | 5 => 0.76128
  | _ => 0.76123 -- default case

theorem largest_by_changing_first_digit :
  ∀ d : Nat, d ∈ [1, 2, 3, 4, 5] → value_with_digit_changed 1 ≥ value_with_digit_changed d :=
by
  intro d hd_list
  sorry

end largest_by_changing_first_digit_l65_65282


namespace max_transition_channel_BC_lowest_cost_per_transition_highest_profit_from_channel_C_l65_65217

theorem max_transition_channel_BC (hB: (1500: ℝ) * 0.042 = 63) (hC1: (4536: ℝ) / 72 = 63):
  max 63 63 = (63: ℝ) :=
by {
  simp [*, max_def];
}

theorem lowest_cost_per_transition (hA: (3417: ℝ) / 51 = 67):
  (67: ℝ) ≤ 78 :=
by {
  linarith,
}

theorem highest_profit_from_channel_C (hC_sales: (63: ℝ) * 0.05 = 3.15) (rounded_sales_C: ⌊3.15⌋ = 3) 
  (sale_revenue_C: 3 * 2500 = 7500) (total_cost_C: 4536):
  7500 - 4536 = (2964: ℝ) :=
by {
  norm_num,
}

end max_transition_channel_BC_lowest_cost_per_transition_highest_profit_from_channel_C_l65_65217


namespace balls_distribution_l65_65958

def ways_to_distribute_balls : Nat := 
  (Nat.choose 7 4) + (Nat.choose 7 3) + (Nat.choose 7 2)

theorem balls_distribution :
  ways_to_distribute_balls = 91 :=
by
  -- Proof goes here
  sorry

end balls_distribution_l65_65958


namespace largest_domain_of_f_l65_65148

theorem largest_domain_of_f (f : ℝ → ℝ) (dom : ℝ → Prop) :
  (∀ x : ℝ, dom x → dom (1 / x)) →
  (∀ x : ℝ, dom x → (f x + f (1 / x) = x)) →
  (∀ x : ℝ, dom x ↔ x = 1 ∨ x = -1) :=
by
  intro h1 h2
  sorry

end largest_domain_of_f_l65_65148


namespace sum_of_ages_l65_65424

-- Definition of the ages based on the intervals and the youngest child's age.
def youngest_age : ℕ := 6
def second_youngest_age : ℕ := youngest_age + 2
def middle_age : ℕ := youngest_age + 4
def second_oldest_age : ℕ := youngest_age + 6
def oldest_age : ℕ := youngest_age + 8

-- The theorem stating the total sum of the ages of the children, given the conditions.
theorem sum_of_ages :
  youngest_age + second_youngest_age + middle_age + second_oldest_age + oldest_age = 50 :=
by sorry

end sum_of_ages_l65_65424


namespace cyc_inequality_l65_65857

theorem cyc_inequality (x y z : ℝ) (hx : 0 < x ∧ x < 2) (hy : 0 < y ∧ y < 2) (hz : 0 < z ∧ z < 2) 
  (hxyz : x^2 + y^2 + z^2 = 3) : 
  3 / 2 < (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) ∧ 
  (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) < 3 := 
by
  sorry

end cyc_inequality_l65_65857


namespace positive_difference_solutions_l65_65740

theorem positive_difference_solutions : 
  ∀ (r : ℝ), r ≠ -3 → 
  (∃ r1 r2 : ℝ, (r^2 - 6*r - 20) / (r + 3) = 3*r + 10 → r1 ≠ r2 ∧ 
  |r1 - r2| = 20) :=
by
  sorry

end positive_difference_solutions_l65_65740


namespace calculate_length_of_train_l65_65193

noncomputable def length_of_train (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := (relative_speed_kmh : ℝ) * 1000 / 3600
  relative_speed_ms * time_seconds

theorem calculate_length_of_train :
  length_of_train 50 5 7.2 = 110 := by
  -- This is where the actual proof would go, but it's omitted for now as per instructions.
  sorry

end calculate_length_of_train_l65_65193


namespace divide_milk_into_equal_parts_l65_65870

def initial_state : (ℕ × ℕ × ℕ) := (8, 0, 0)

def is_equal_split (state : ℕ × ℕ × ℕ) : Prop :=
  state.1 = 4 ∧ state.2 = 4

theorem divide_milk_into_equal_parts : 
  ∃ (state_steps : Fin 25 → ℕ × ℕ × ℕ),
  initial_state = state_steps 0 ∧
  is_equal_split (state_steps 24) :=
sorry

end divide_milk_into_equal_parts_l65_65870


namespace polynomial_div_simplify_l65_65565

theorem polynomial_div_simplify (x : ℝ) (hx : x ≠ 0) :
  (6 * x ^ 4 - 4 * x ^ 3 + 2 * x ^ 2) / (2 * x ^ 2) = 3 * x ^ 2 - 2 * x + 1 :=
by sorry

end polynomial_div_simplify_l65_65565


namespace probability_correct_l65_65348

-- Definitions and conditions
def G : List Char := ['A', 'B', 'C', 'D']

-- Number of favorable arrangements where A is adjacent to B and C
def favorable_arrangements : ℕ := 4  -- ABCD, BCDA, DABC, and CDAB

-- Total possible arrangements of 4 people
def total_arrangements : ℕ := 24  -- 4!

-- Probability calculation
def probability_A_adjacent_B_C : ℚ := favorable_arrangements / total_arrangements

-- Prove that this probability equals 1/6
theorem probability_correct : probability_A_adjacent_B_C = 1 / 6 := by
  sorry

end probability_correct_l65_65348


namespace watermelons_last_6_weeks_l65_65120

variable (initial_watermelons : ℕ) (eaten_per_week : ℕ) (given_away_per_week : ℕ)

def watermelons_last_weeks (initial_watermelons : ℕ) (weekly_usage : ℕ) : ℕ :=
initial_watermelons / weekly_usage

theorem watermelons_last_6_weeks :
  initial_watermelons = 30 ∧ eaten_per_week = 3 ∧ given_away_per_week = 2 →
  watermelons_last_weeks initial_watermelons (eaten_per_week + given_away_per_week) = 6 := 
by
  intro h
  cases h with h_initial he 
  cases he with  h_eaten h_given
  have weekly_usage := h_eaten + h_given
  have weeks := watermelons_last_weeks h_initial weekly_usage
  sorry

end watermelons_last_6_weeks_l65_65120


namespace find_a7_l65_65222

-- Definitions based on given conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k : ℕ, a (n + k) = a n + k * (a 1 - a 0)

-- Given condition in Lean statement
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 11 = 22

-- Proof problem
theorem find_a7 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : a 7 = 11 := 
  sorry

end find_a7_l65_65222


namespace cost_of_potatoes_l65_65947

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end cost_of_potatoes_l65_65947


namespace probability_one_in_first_20_rows_l65_65322

theorem probability_one_in_first_20_rows :
  let total_elements := 210
  let number_of_ones := 39
  (number_of_ones / total_elements : ℚ) = 13 / 70 :=
by
  sorry

end probability_one_in_first_20_rows_l65_65322


namespace find_v_3_l65_65384

def u (x : ℤ) : ℤ := 4 * x - 9

def v (z : ℤ) : ℤ := z^2 + 4 * z - 1

theorem find_v_3 : v 3 = 20 := by
  sorry

end find_v_3_l65_65384


namespace angle_B_is_60_l65_65827

theorem angle_B_is_60 (A B C : ℝ) (h_seq : 2 * B = A + C) (h_sum : A + B + C = 180) : B = 60 := 
by 
  sorry

end angle_B_is_60_l65_65827


namespace area_of_triangle_is_27_over_5_l65_65467

def area_of_triangle_bounded_by_y_axis_and_lines : ℚ :=
  let y_intercept_1 := -2
  let y_intercept_2 := 4
  let base := y_intercept_2 - y_intercept_1
  let x_intersection : ℚ := 9 / 5   -- Calculated using the system of equations
  1 / 2 * base * x_intersection

theorem area_of_triangle_is_27_over_5 :
  area_of_triangle_bounded_by_y_axis_and_lines = 27 / 5 := by
  sorry

end area_of_triangle_is_27_over_5_l65_65467


namespace elementary_schools_in_Lansing_l65_65250

theorem elementary_schools_in_Lansing (total_students : ℕ) (students_per_school : ℕ) (h1 : total_students = 6175) (h2 : students_per_school = 247) : total_students / students_per_school = 25 := 
by sorry

end elementary_schools_in_Lansing_l65_65250


namespace invertible_matrixA_matrixA_inverse_is_correct_l65_65794

open Matrix

def matrixA : Matrix (Fin 2) (Fin 2) ℝ :=
  matrixOf ![![4, 7], ![2, 6]]

def matrixA_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  matrixOf ![![0.6, -0.7], ![-0.2, 0.4]]

theorem invertible_matrixA : invertible matrixA :=
  by 
  have h_det : det matrixA ≠ 0 := by
    simp [matrixA, det, Finset.sum, Matrix.fin_two_sum, algebra_map, fintype.univ]
  use matrixA_inv, sorry

theorem matrixA_inverse_is_correct:
  matrix.mul matrixA matrixA_inv = 1 ∧ matrix.mul matrixA_inv matrixA = 1 := 
  by
  sorry

end invertible_matrixA_matrixA_inverse_is_correct_l65_65794


namespace books_returned_wednesday_correct_l65_65987

def initial_books : Nat := 250
def books_taken_out_Tuesday : Nat := 120
def books_taken_out_Thursday : Nat := 15
def books_remaining_after_Thursday : Nat := 150

def books_after_tuesday := initial_books - books_taken_out_Tuesday
def books_before_thursday := books_remaining_after_Thursday + books_taken_out_Thursday
def books_returned_wednesday := books_before_thursday - books_after_tuesday

theorem books_returned_wednesday_correct : books_returned_wednesday = 35 := by
  sorry

end books_returned_wednesday_correct_l65_65987


namespace sum_of_first_15_terms_l65_65940

-- Given an arithmetic sequence {a_n} such that a_4 + a_6 + a_8 + a_10 + a_12 = 40
-- we need to prove that the sum of the first 15 terms is 120

theorem sum_of_first_15_terms 
  (a_4 a_6 a_8 a_10 a_12 : ℤ)
  (h1 : a_4 + a_6 + a_8 + a_10 + a_12 = 40)
  (a1 d : ℤ)
  (h2 : a_4 = a1 + 3*d)
  (h3 : a_6 = a1 + 5*d)
  (h4 : a_8 = a1 + 7*d)
  (h5 : a_10 = a1 + 9*d)
  (h6 : a_12 = a1 + 11*d) :
  (15 * (a1 + 7*d) = 120) :=
by
  sorry

end sum_of_first_15_terms_l65_65940


namespace jim_travel_distance_l65_65818

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end jim_travel_distance_l65_65818


namespace find_a_l65_65802

-- Assuming the existence of functions and variables as per conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Defining the given conditions
axiom cond1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x - 5
axiom cond2 : f a = 6

-- Now stating the proof goal
theorem find_a : a = 7 / 4 := by
  sorry

end find_a_l65_65802


namespace M_value_l65_65100

noncomputable def x : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)

noncomputable def y : ℝ := Real.sqrt (4 - 2 * Real.sqrt 3)

noncomputable def M : ℝ := x - y

theorem M_value :
  M = (5 / 2) * Real.sqrt 2 - Real.sqrt 3 + (3 / 2) :=
sorry

end M_value_l65_65100


namespace increasing_exponential_function_l65_65265

theorem increasing_exponential_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → (a ^ x) < (a ^ y)) → (1 < a) :=
by
  sorry

end increasing_exponential_function_l65_65265


namespace convex_quadrilaterals_from_12_points_l65_65554

theorem convex_quadrilaterals_from_12_points : 
  ∀ (points : Finset ℕ), points.card = 12 → 
  (∃ n : ℕ, n = Multichoose 12 4 ∧ n = 495) :=
by
  sorry

end convex_quadrilaterals_from_12_points_l65_65554


namespace leila_money_left_l65_65840

theorem leila_money_left (initial_money spent_on_sweater spent_on_jewelry total_spent left_money : ℕ) 
  (h1 : initial_money = 160) 
  (h2 : spent_on_sweater = 40) 
  (h3 : spent_on_jewelry = 100) 
  (h4 : total_spent = spent_on_sweater + spent_on_jewelry) 
  (h5 : total_spent = 140) : 
  initial_money - total_spent = 20 := by
  sorry

end leila_money_left_l65_65840


namespace bacon_cost_l65_65971

namespace PancakeBreakfast

def cost_of_stack_pancakes : ℝ := 4.0
def stacks_sold : ℕ := 60
def slices_bacon_sold : ℕ := 90
def total_revenue : ℝ := 420.0

theorem bacon_cost (B : ℝ) 
  (h1 : stacks_sold * cost_of_stack_pancakes + slices_bacon_sold * B = total_revenue) : 
  B = 2 :=
  by {
    sorry
  }

end PancakeBreakfast

end bacon_cost_l65_65971


namespace safer_four_engine_airplane_l65_65455

theorem safer_four_engine_airplane (P : ℝ) (hP : 0 < P ∧ P < 1):
  (∃ p : ℝ, p = 1 - P ∧ (p^4 + 4 * p^3 * (1 - p) + 6 * p^2 * (1 - p)^2 > p^2 + 2 * p * (1 - p) ↔ P > 2 / 3)) :=
sorry

end safer_four_engine_airplane_l65_65455


namespace inequality_abc_l65_65694

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^3) / (a^2 + a * b + b^2) + (b^3) / (b^2 + b * c + c^2) + (c^3) / (c^2 + c * a + a^2) ≥ (a + b + c) / 3 := 
by
    sorry

end inequality_abc_l65_65694


namespace certain_event_positive_integers_sum_l65_65317

theorem certain_event_positive_integers_sum :
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b > 1 :=
by
  intros a b ha hb
  sorry

end certain_event_positive_integers_sum_l65_65317


namespace bill_experience_l65_65595

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l65_65595


namespace sequence_increasing_range_l65_65502

theorem sequence_increasing_range (a : ℝ) (h : ∀ n : ℕ, (n - a) ^ 2 < (n + 1 - a) ^ 2) :
  a < 3 / 2 :=
by
  sorry

end sequence_increasing_range_l65_65502


namespace nested_sqrt_solution_l65_65069

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l65_65069


namespace vehicle_distance_traveled_l65_65559

theorem vehicle_distance_traveled 
  (perimeter_back : ℕ) (perimeter_front : ℕ) (revolution_difference : ℕ)
  (R : ℕ)
  (h1 : perimeter_back = 9)
  (h2 : perimeter_front = 7)
  (h3 : revolution_difference = 10)
  (h4 : (R * perimeter_back) = ((R + revolution_difference) * perimeter_front)) :
  (R * perimeter_back) = 315 :=
by
  -- Prove that the distance traveled by the vehicle is 315 feet
  -- given the conditions and the hypothesis.
  sorry

end vehicle_distance_traveled_l65_65559


namespace triangle_tangent_ratio_l65_65927

variable {A B C a b c : ℝ}

theorem triangle_tangent_ratio 
  (h : a * Real.cos B - b * Real.cos A = (3 / 5) * c)
  : Real.tan A / Real.tan B = 4 :=
sorry

end triangle_tangent_ratio_l65_65927


namespace range_of_a_l65_65360

open Classical

noncomputable def parabola_line_common_point_range (a : ℝ) : Prop :=
  ∃ (k : ℝ), ∃ (x : ℝ), ∃ (y : ℝ), 
  (y = a * x ^ 2) ∧ ((y + 2 = k * (x - 1)) ∨ (y + 2 = - (1 / k) * (x - 1)))

theorem range_of_a (a : ℝ) : 
  (∃ k : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
    y = a * x ^ 2 ∧ (y + 2 = k * (x - 1) ∨ y + 2 = - (1 / k) * (x - 1))) ↔ 
  0 < a ∧ a <= 1 / 8 :=
sorry

end range_of_a_l65_65360


namespace polynomial_sum_of_squares_l65_65138

theorem polynomial_sum_of_squares (P : Polynomial ℝ) 
  (hP : ∀ x : ℝ, 0 ≤ P.eval x) : 
  ∃ (f g : Polynomial ℝ), P = f * f + g * g := 
sorry

end polynomial_sum_of_squares_l65_65138


namespace percent_half_dollars_correct_l65_65285

open Rat

noncomputable def value_nickels (n : ℕ) : ℚ := n * 5
noncomputable def value_half_dollars (h : ℕ) : ℚ := h * 50
noncomputable def total_value (n h : ℕ) : ℚ := value_nickels n + value_half_dollars h
noncomputable def percent_half_dollars (n h : ℕ) : ℚ :=
  value_half_dollars h / total_value n h * 100

theorem percent_half_dollars_correct :
  percent_half_dollars 80 40 = 83.33 := by
  sorry

end percent_half_dollars_correct_l65_65285


namespace problem_1_problem_2a_problem_2b_l65_65228

noncomputable def v_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def v_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (v_a x).1 * (v_b).1 + (v_a x).2 * (v_b).2

theorem problem_1 (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) : 
  (v_a x).1 * (v_b).2 = (v_a x).2 * (v_b).1 → x = (5 * Real.pi / 6) :=
by
  sorry

theorem problem_2a : 
  ∃ x ∈ Set.Icc 0 Real.pi, f x = 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≤ 3 :=
by
  sorry

theorem problem_2b :
  ∃ x ∈ Set.Icc 0 Real.pi, f x = -2 * Real.sqrt 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≥ -2 * Real.sqrt 3 :=
by
  sorry

end problem_1_problem_2a_problem_2b_l65_65228


namespace tan_of_tan_squared_2025_deg_l65_65202

noncomputable def tan_squared (x : ℝ) : ℝ := (Real.tan x) ^ 2

theorem tan_of_tan_squared_2025_deg : 
  Real.tan (tan_squared (2025 * Real.pi / 180)) = Real.tan (Real.pi / 180) :=
by
  sorry

end tan_of_tan_squared_2025_deg_l65_65202


namespace average_weight_of_whole_class_l65_65289

theorem average_weight_of_whole_class :
  let students_A := 26
  let students_B := 34
  let avg_weight_A := 50
  let avg_weight_B := 30
  let total_weight_A := avg_weight_A * students_A
  let total_weight_B := avg_weight_B * students_B
  let total_weight_class := total_weight_A + total_weight_B
  let total_students_class := students_A + students_B
  let avg_weight_class := total_weight_class / total_students_class
  avg_weight_class = 38.67 :=
by {
  sorry -- Proof is not required as per instructions
}

end average_weight_of_whole_class_l65_65289


namespace most_marbles_l65_65272

def total_marbles := 24
def red_marble_fraction := 1 / 4
def red_marbles := red_marble_fraction * total_marbles
def blue_marbles := red_marbles + 6
def yellow_marbles := total_marbles - red_marbles - blue_marbles

theorem most_marbles : blue_marbles > red_marbles ∧ blue_marbles > yellow_marbles :=
by
  sorry

end most_marbles_l65_65272


namespace solve_fractional_equation_l65_65014

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x + 1) / 5 - x / 10 = 2 → x = 6 :=
by
  intros x h
  sorry

end solve_fractional_equation_l65_65014


namespace no_five_consecutive_terms_divisible_by_2005_l65_65809

noncomputable def a (n : ℕ) : ℤ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_terms_divisible_by_2005 : ¬ ∃ n : ℕ, (a n % 2005 = 0) ∧ (a (n+1) % 2005 = 0) ∧ (a (n+2) % 2005 = 0) ∧ (a (n+3) % 2005 = 0) ∧ (a (n+4) % 2005 = 0) := sorry

end no_five_consecutive_terms_divisible_by_2005_l65_65809


namespace curve_statements_incorrect_l65_65487

theorem curve_statements_incorrect (t : ℝ) :
  (1 < t ∧ t < 3 → ¬ ∀ x y : ℝ, (x^2 / (3 - t) + y^2 / (t - 1) = 1 → x^2 + y^2 ≠ 1)) ∧
  ((3 - t) * (t - 1) < 0 → ¬ t < 1) :=
by
  sorry

end curve_statements_incorrect_l65_65487


namespace triangle_third_side_l65_65807

open Nat

theorem triangle_third_side (a b c : ℝ) (h1 : a = 4) (h2 : b = 9) (h3 : c > 0) :
  (5 < c ∧ c < 13) ↔ c = 6 :=
by
  sorry

end triangle_third_side_l65_65807


namespace pears_sales_l65_65895

variable (x : ℝ)
variable (morning_sales : ℝ := x)
variable (afternoon_sales : ℝ := 2 * x)
variable (evening_sales : ℝ := 3 * afternoon_sales)
variable (total_sales : ℝ := morning_sales + afternoon_sales + evening_sales)

theorem pears_sales :
  (total_sales = 510) →
  (afternoon_sales = 113.34) :=
by
  sorry

end pears_sales_l65_65895


namespace fraction_value_l65_65218

theorem fraction_value
  (a b c d : ℚ)
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b ≠ 0)
  (h4 : d ≠ 0)
  (h5 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 :=
sorry

end fraction_value_l65_65218


namespace jacob_river_water_collection_l65_65656

/-- Definitions: 
1. Capacity of the tank in milliliters
2. Daily water collected from the rain in milliliters
3. Number of days to fill the tank
4. To be proved: Daily water collected from the river in milliliters
-/
def tank_capacity_ml : Int := 50000
def daily_rain_ml : Int := 800
def days_to_fill : Int := 20
def daily_river_ml : Int := 1700

/-- Prove that the amount of water Jacob collects from the river every day equals 1700 milliliters.
-/
theorem jacob_river_water_collection (total_water: Int) 
  (rain_water: Int) (days: Int) (correct_river_water: Int) : 
  total_water = tank_capacity_ml → 
  rain_water = daily_rain_ml → 
  days = days_to_fill → 
  correct_river_water = daily_river_ml → 
  (total_water - rain_water * days) / days = correct_river_water := 
by 
  intros; 
  sorry

end jacob_river_water_collection_l65_65656


namespace geom_seq_a4_l65_65830

theorem geom_seq_a4 (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h2 : a 3 = 9)
  (h3 : a 5 = 1) :
  a 4 = 3 ∨ a 4 = -3 :=
by {
  sorry
}

end geom_seq_a4_l65_65830


namespace f_g_of_4_eq_18_sqrt_21_div_7_l65_65253

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x - 3

theorem f_g_of_4_eq_18_sqrt_21_div_7 : f (g 4) = (18 * Real.sqrt 21) / 7 := by
  sorry

end f_g_of_4_eq_18_sqrt_21_div_7_l65_65253


namespace valid_integer_pairs_l65_65791

theorem valid_integer_pairs :
  ∀ a b : ℕ, 1 ≤ a → 1 ≤ b → a ^ (b ^ 2) = b ^ a → (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end valid_integer_pairs_l65_65791


namespace water_fall_amount_l65_65243

theorem water_fall_amount (M_before J_before M_after J_after n : ℕ) 
  (h1 : M_before = 48) 
  (h2 : M_before = J_before + 32)
  (h3 : M_after = M_before + n) 
  (h4 : J_after = J_before + n)
  (h5 : M_after = 2 * J_after) : 
  n = 16 :=
by 
  -- proof omitted
  sorry

end water_fall_amount_l65_65243


namespace alcohol_percentage_new_mixture_l65_65036

/--
Given:
1. The initial mixture has 15 liters.
2. The mixture contains 20% alcohol.
3. 5 liters of water is added to the mixture.

Prove:
The percentage of alcohol in the new mixture is 15%.
-/
theorem alcohol_percentage_new_mixture :
  let initial_mixture_volume := 15 -- in liters
  let initial_alcohol_percentage := 20 / 100
  let initial_alcohol_volume := initial_alcohol_percentage * initial_mixture_volume
  let added_water_volume := 5 -- in liters
  let new_total_volume := initial_mixture_volume + added_water_volume
  let new_alcohol_percentage := (initial_alcohol_volume / new_total_volume) * 100
  new_alcohol_percentage = 15 := 
by
  -- Proof steps go here
  sorry

end alcohol_percentage_new_mixture_l65_65036


namespace meaningful_fraction_condition_l65_65737

theorem meaningful_fraction_condition (x : ℝ) : x - 2 ≠ 0 ↔ x ≠ 2 := 
by 
  sorry

end meaningful_fraction_condition_l65_65737


namespace probability_of_one_in_pascal_rows_l65_65326

theorem probability_of_one_in_pascal_rows (n : ℕ) (h : n = 20) : 
  let total_elements := (n * (n + 1)) / 2,
      ones := 1 + 2 * (n - 1) in
  (ones / total_elements : ℚ) = 39 / 210 :=
by
  sorry

end probability_of_one_in_pascal_rows_l65_65326


namespace km_to_m_is_750_l65_65800

-- Define 1 kilometer equals 5 hectometers
def km_to_hm := 5

-- Define 1 hectometer equals 10 dekameters
def hm_to_dam := 10

-- Define 1 dekameter equals 15 meters
def dam_to_m := 15

-- Theorem stating that the number of meters in one kilometer is 750
theorem km_to_m_is_750 : 1 * km_to_hm * hm_to_dam * dam_to_m = 750 :=
by 
  -- Proof goes here
  sorry

end km_to_m_is_750_l65_65800


namespace range_of_m_l65_65911

noncomputable def f (x m a : ℝ) : ℝ := Real.exp (x + 1) - m * a
noncomputable def g (x a : ℝ) : ℝ := a * Real.exp x - x

theorem range_of_m (h : ∃ a : ℝ, ∀ x : ℝ, f x m a ≤ g x a) : m ≥ -1 / Real.exp 1 :=
by
  sorry

end range_of_m_l65_65911


namespace remainder_sand_amount_l65_65271

def total_sand : ℝ := 2548726
def bag_capacity : ℝ := 85741.2
def full_bags : ℝ := 29
def not_full_bag_sand : ℝ := 62231.2

theorem remainder_sand_amount :
  total_sand - (full_bags * bag_capacity) = not_full_bag_sand :=
by
  sorry

end remainder_sand_amount_l65_65271


namespace probability_of_both_even_l65_65566

def totalBalls : ℕ := 17
def evenBalls : ℕ := 8
def firstDrawProb : ℚ := evenBalls / totalBalls
def secondDrawEvenBalls : ℕ := evenBalls - 1
def totalRemainingBalls : ℕ := totalBalls - 1
def secondDrawProb : ℚ := secondDrawEvenBalls / totalRemainingBalls

theorem probability_of_both_even : firstDrawProb * secondDrawProb = 7 / 34 :=
by
  -- proof to be filled
  sorry

end probability_of_both_even_l65_65566


namespace diane_honey_harvest_l65_65785

theorem diane_honey_harvest (last_year : ℕ) (increase : ℕ) (this_year : ℕ) :
  last_year = 2479 → increase = 6085 → this_year = last_year + increase → this_year = 8564 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end diane_honey_harvest_l65_65785


namespace sum_of_angles_l65_65291

namespace BridgeProblem

def is_isosceles (A B C : Type) (AB AC : ℝ) : Prop := AB = AC

def angle_bac (A B C : Type) : ℝ := 15

def angle_edf (D E F : Type) : ℝ := 45

theorem sum_of_angles (A B C D E F : Type) 
  (h_isosceles_ABC : is_isosceles A B C 1 1)
  (h_isosceles_DEF : is_isosceles D E F 1 1)
  (h_angle_BAC : angle_bac A B C = 15)
  (h_angle_EDF : angle_edf D E F = 45) :
  true := 
by 
  sorry

end BridgeProblem

end sum_of_angles_l65_65291


namespace probability_two_acceptable_cans_l65_65182

variable {total_cans : ℕ} {unacceptable : ℕ}

-- Conditions
def total_cans : ℕ := 6
def unacceptable : ℕ := 2
def acceptable : ℕ := total_cans - unacceptable
def total_ways_to_choose_two := Nat.choose total_cans 2
def acceptable_ways_to_choose_two := Nat.choose acceptable 2

-- Statement to prove the probability
theorem probability_two_acceptable_cans : 
  (acceptable_ways_to_choose_two : ℚ) / (total_ways_to_choose_two : ℚ) = 2 / 5 := 
by
  sorry

end probability_two_acceptable_cans_l65_65182


namespace negation_example_l65_65864

theorem negation_example :
  ¬ (∀ n : ℕ, (n^2 + n) % 2 = 0) ↔ ∃ n : ℕ, (n^2 + n) % 2 ≠ 0 :=
by
  sorry

end negation_example_l65_65864


namespace years_until_5_years_before_anniversary_l65_65106

-- Definitions
def years_built_ago := 100
def upcoming_anniversary := 200
def years_before_anniversary := 5

-- Theorem statement
theorem years_until_5_years_before_anniversary :
  let years_until_anniversary := upcoming_anniversary - years_built_ago in
  let future_years := years_until_anniversary - years_before_anniversary in
  future_years = 95 :=
by
  sorry

end years_until_5_years_before_anniversary_l65_65106


namespace correct_random_variable_l65_65886

-- Define the given conditions
def total_white_balls := 5
def total_red_balls := 3
def total_balls := total_white_balls + total_red_balls
def balls_drawn := 3

-- Define the random variable
noncomputable def is_random_variable_correct (option : ℕ) :=
  option = 2

-- The theorem to be proved
theorem correct_random_variable: is_random_variable_correct 2 :=
by
  sorry

end correct_random_variable_l65_65886


namespace math_problem_proof_l65_65496

-- Define the conditions for the function f(x)
variables {a b c : ℝ}
variables (ha : a ≠ 0) (h1 : (b/a) > 0) (h2 : (-2 * c/a) > 0) (h3 : (b^2 + 8 * a * c) > 0)

-- Define the statements to be proved based on the conditions
theorem math_problem_proof :
    (a ≠ 0) →
    (b/a > 0) →
    (-2 * c/a > 0) →
    (b^2 + 8*a*c > 0) →
    (ab : (a*b) > 0) ∧    -- B
    ((b^2 + 8*a*c) > 0) ∧ -- C
    (ac : a*c < 0)        -- D
 := by
    intros ha h1 h2 h3
    sorry

end math_problem_proof_l65_65496


namespace find_percentage_l65_65293

theorem find_percentage (P : ℝ) : 
  (P / 100) * 700 = 210 ↔ P = 30 := by
  sorry

end find_percentage_l65_65293


namespace polygon_sides_l65_65505

  theorem polygon_sides (S : ℤ) (n : ℤ) (h : S = 1080) : 180 * (n - 2) = S → n = 8 :=
  by
    intro h_sum
    rw h at h_sum
    sorry
  
end polygon_sides_l65_65505


namespace solve_swim_problem_l65_65189

/-- A man swims downstream 36 km and upstream some distance taking 3 hours each time. 
The speed of the man in still water is 9 km/h. -/
def swim_problem : Prop :=
  ∃ (v : ℝ) (d : ℝ),
    (9 + v) * 3 = 36 ∧ -- effective downstream speed and distance condition
    (9 - v) * 3 = d ∧ -- effective upstream speed and distance relation
    d = 18            -- required distance upstream is 18 km

theorem solve_swim_problem : swim_problem :=
  sorry

end solve_swim_problem_l65_65189


namespace social_media_usage_in_week_l65_65614

def days_in_week : ℕ := 7
def daily_phone_usage : ℕ := 16
def daily_social_media_usage : ℕ := daily_phone_usage / 2

theorem social_media_usage_in_week :
  daily_social_media_usage * days_in_week = 56 :=
by
  sorry

end social_media_usage_in_week_l65_65614


namespace sqrt_expression_simplification_l65_65906

theorem sqrt_expression_simplification :
  (Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt (3) - 1| + Real.sqrt 3) = -13 / 4 + 2 * Real.sqrt 3 :=
by
  have h1 : Real.sqrt (1 / 16) = 1 / 4 := sorry
  have h2 : Real.sqrt (25 / 4) = 5 / 2 := sorry
  have h3 : |Real.sqrt 3 - 1| = Real.sqrt 3 - 1 := sorry
  linarith [h1, h2, h3]

end sqrt_expression_simplification_l65_65906


namespace part_I_solution_part_II_solution_l65_65477

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem part_I_solution (x : ℝ) :
  (|x + 1| - |x - 1| >= x) ↔ (x <= -2 ∨ (0 <= x ∧ x <= 2)) :=
by
  sorry

theorem part_II_solution (m : ℝ) :
  (∀ (x a : ℝ), (0 < m ∧ m < 1 ∧ (a <= -3 ∨ 3 <= a)) → (f x a m >= 2)) ↔ (m = 1/3) :=
by
  sorry

end part_I_solution_part_II_solution_l65_65477


namespace solve_for_x_l65_65406

theorem solve_for_x (x : ℝ) : 9 * x^2 - 4 = 0 → (x = 2/3 ∨ x = -2/3) :=
by
  sorry

end solve_for_x_l65_65406


namespace find_b_l65_65355

theorem find_b (b : ℝ) : (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ y = 2 * x + b) → b = 0 := by
  sorry

end find_b_l65_65355


namespace prove_temperature_on_Thursday_l65_65592

def temperature_on_Thursday 
  (temps : List ℝ)   -- List of temperatures for 6 days.
  (avg : ℝ)          -- Average temperature for the week.
  (sum_six_days : ℝ) -- Sum of temperature readings for 6 days.
  (days : ℕ := 7)    -- Number of days in the week.
  (missing_day : ℕ := 1)  -- One missing day (Thursday).
  (thurs_temp : ℝ := 99.8) -- Temperature on Thursday to be proved.
: Prop := (avg * days) - sum_six_days = thurs_temp

theorem prove_temperature_on_Thursday 
  : temperature_on_Thursday [99.1, 98.2, 98.7, 99.3, 99, 98.9] 99 593.2 :=
by
  sorry

end prove_temperature_on_Thursday_l65_65592


namespace students_not_pass_l65_65394

theorem students_not_pass (total_students : ℕ) (percentage_passed : ℕ) (students_passed : ℕ) (students_not_passed : ℕ) :
  total_students = 804 →
  percentage_passed = 75 →
  students_passed = total_students * percentage_passed / 100 →
  students_not_passed = total_students - students_passed →
  students_not_passed = 201 :=
by
  intros h1 h2 h3 h4
  sorry

end students_not_pass_l65_65394


namespace find_coordinates_of_A_l65_65650

-- Definition of points in Cartesian coordinate system
structure Point where
  x : Int
  y : Int

-- Definitions for translations
def translate_left (A : Point) (distance : Int) : Point :=
  { x := A.x - distance, y := A.y }

def translate_up (A : Point) (distance : Int) : Point :=
  { x := A.x, y := A.y + distance }

-- Given conditions
variables A : Point
variables h1 : ∃ d, translate_left A d = { x := 1, y := 2 }
variables h2 : ∃ d, translate_up A d = { x := 3, y := 4 }

-- Proof statement
theorem find_coordinates_of_A : A = { x := 3, y := 2 } :=
sorry

end find_coordinates_of_A_l65_65650


namespace king_paid_after_tip_l65_65761

theorem king_paid_after_tip:
  (crown_cost tip_percentage total_cost : ℝ)
  (h_crown_cost : crown_cost = 20000)
  (h_tip_percentage : tip_percentage = 0.1) :
  total_cost = crown_cost + (crown_cost * tip_percentage) :=
by
  have h_tip := h_crown_cost.symm ▸ h_tip_percentage.symm ▸ 20000 * 0.1
  have h_total := h_crown_cost.symm ▸ (h_tip.symm ▸ 2000)
  rw [h_crown_cost, h_tip, h_total]
  exact rfl

end king_paid_after_tip_l65_65761


namespace max_newsstands_six_corridors_l65_65653

def number_of_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem max_newsstands_six_corridors : number_of_intersections 6 = 15 := 
by sorry

end max_newsstands_six_corridors_l65_65653


namespace vector_AB_equality_l65_65378

variable {V : Type*} [AddCommGroup V]

variables (a b : V)

theorem vector_AB_equality (BC CA : V) (hBC : BC = a) (hCA : CA = b) :
  CA - BC = b - a :=
by {
  sorry
}

end vector_AB_equality_l65_65378


namespace king_paid_after_tip_l65_65758

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l65_65758


namespace skyscraper_anniversary_l65_65110

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l65_65110


namespace highest_score_l65_65171

variables (H L : ℕ) (average_46 : ℕ := 61) (innings_46 : ℕ := 46) 
                (difference : ℕ := 150) (average_44 : ℕ := 58) (innings_44 : ℕ := 44)

theorem highest_score:
  (H - L = difference) →
  (average_46 * innings_46 = average_44 * innings_44 + H + L) →
  H = 202 :=
by
  intros h_diff total_runs_eq
  sorry

end highest_score_l65_65171


namespace infinite_nested_sqrt_l65_65065

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l65_65065


namespace perpendicular_lines_m_value_l65_65374

theorem perpendicular_lines_m_value (m : ℝ) (l1_perp_l2 : (m ≠ 0) → (m * (-1 / m^2)) = -1) : m = 0 ∨ m = 1 :=
sorry

end perpendicular_lines_m_value_l65_65374


namespace average_cookies_per_package_l65_65237

def cookies_per_package : List ℕ := [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]

theorem average_cookies_per_package :
  (cookies_per_package.sum : ℚ) / cookies_per_package.length = 13.5 := by
  sorry

end average_cookies_per_package_l65_65237


namespace regression_lines_have_common_point_l65_65552

theorem regression_lines_have_common_point
  (n m : ℕ)
  (h₁ : n = 10)
  (h₂ : m = 15)
  (s t : ℝ)
  (data_A data_B : Fin n → Fin n → ℝ)
  (avg_x_A avg_x_B : ℝ)
  (avg_y_A avg_y_B : ℝ)
  (regression_line_A regression_line_B : ℝ → ℝ)
  (h₃ : avg_x_A = s)
  (h₄ : avg_x_B = s)
  (h₅ : avg_y_A = t)
  (h₆ : avg_y_B = t)
  (h₇ : ∀ x, regression_line_A x = a*x + b)
  (h₈ : ∀ x, regression_line_B x = c*x + d)
  : regression_line_A s = t ∧ regression_line_B s = t :=
by
  sorry

end regression_lines_have_common_point_l65_65552


namespace time_for_A_and_C_l65_65172

variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := A + B = 1 / 8
def condition2 : Prop := B + C = 1 / 12
def condition3 : Prop := A + B + C = 1 / 6

theorem time_for_A_and_C (h1 : condition1 A B)
                        (h2 : condition2 B C)
                        (h3 : condition3 A B C) :
  1 / (A + C) = 8 :=
sorry

end time_for_A_and_C_l65_65172


namespace marikas_father_age_twice_in_2036_l65_65687

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l65_65687


namespace water_drank_is_gallons_l65_65897

noncomputable def total_water_drunk (traveler_weight: ℝ) (traveler_percent: ℝ) (camel_weight: ℝ) (camel_percent: ℝ) 
(pounds_to_ounces: ℝ) (ounces_to_gallon: ℝ) : ℝ :=
(traveler_weight * traveler_percent / 100 * pounds_to_ounces + camel_weight * camel_percent / 100 * pounds_to_ounces) / ounces_to_gallon

theorem water_drank_is_gallons :
  total_water_drunk 160 0.5 1200 2 16 128 = 3.1 :=
by
  unfold total_water_drunk
  norm_num
  sorry

end water_drank_is_gallons_l65_65897


namespace square_side_length_l65_65240

-- Define the given dimensions and total length
def rectangle_width : ℕ := 2
def total_length : ℕ := 7

-- Define the unknown side length of the square
variable (Y : ℕ)

-- State the problem and provide the conclusion
theorem square_side_length : Y + rectangle_width = total_length -> Y = 5 :=
by 
  sorry

end square_side_length_l65_65240


namespace repeating_decimal_denominator_l65_65006

theorem repeating_decimal_denominator (S : ℚ) (h : S = 0.27) : ∃ d : ℤ, (S = d / 11) :=
by
  sorry

end repeating_decimal_denominator_l65_65006


namespace rebecca_soda_bottles_left_l65_65400

theorem rebecca_soda_bottles_left:
  (let half_bottles_per_day := 1 / 2
       total_bottles_bought := 3 * 6
       days_per_week := 7
       weeks := 4
       total_half_bottles_consumed := weeks * days_per_week
       total_full_bottles_consumed := total_half_bottles_consumed / 2
       bottles_left := total_bottles_bought - total_full_bottles_consumed in
   bottles_left = 4) :=
by
  sorry

end rebecca_soda_bottles_left_l65_65400


namespace intersection_complement_l65_65530

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5 }
def C : Set ℝ := {x | x ≤ 2 ∨ x ≥ 5 }

theorem intersection_complement :
  (A ∩ C) = {1, 2, 5, 6} :=
by sorry

end intersection_complement_l65_65530


namespace Konstantin_mother_returns_amount_l65_65244

theorem Konstantin_mother_returns_amount
  (deposit_usd : ℝ)
  (exchange_rate : ℝ)
  (equivalent_rubles : ℝ)
  (h_deposit_usd : deposit_usd = 10000)
  (h_exchange_rate : exchange_rate = 58.15)
  (h_equivalent_rubles : equivalent_rubles = deposit_usd * exchange_rate) :
  equivalent_rubles = 581500 :=
by {
  rw [h_deposit_usd, h_exchange_rate] at h_equivalent_rubles,
  exact h_equivalent_rubles
}

end Konstantin_mother_returns_amount_l65_65244


namespace share_of_a_l65_65437

theorem share_of_a 
  (A B C : ℝ)
  (h1 : A = (2/3) * (B + C))
  (h2 : B = (2/3) * (A + C))
  (h3 : A + B + C = 200) :
  A = 60 :=
by {
  sorry
}

end share_of_a_l65_65437


namespace average_speed_of_planes_l65_65159

def planePassengers (n1 n2 n3 : ℕ) := (n1, n2, n3)
def emptyPlaneSpeed : ℕ := 600
def speedReductionPerPassenger : ℕ := 2
def planeSpeed (s0 r n : ℕ) : ℕ := s0 - r * n
def averageSpeed (speeds : List ℕ) : ℕ := (List.sum speeds) / speeds.length

theorem average_speed_of_planes :
  let (n1, n2, n3) := planePassengers 50 60 40 in
  let s0 := emptyPlaneSpeed in
  let r := speedReductionPerPassenger in
  let speed1 := planeSpeed s0 r n1 in
  let speed2 := planeSpeed s0 r n2 in
  let speed3 := planeSpeed s0 r n3 in
  averageSpeed [speed1, speed2, speed3] = 500 := by
  sorry

end average_speed_of_planes_l65_65159


namespace arithmetic_sequence_general_formula_l65_65476

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom a2 : a 2 = 6
axiom S5 : S 5 = 40

-- Prove the general formulas
theorem arithmetic_sequence_general_formula (n : ℕ)
  (h1 : ∃ d a1, ∀ n, a n = a1 + (n - 1) * d)
  (h2 : ∃ d a1, ∀ n, S n = n * ((2 * a1) + (n - 1) * d) / 2) :
  (a n = 2 * n + 2) ∧ (S n = n * (n + 3)) := by
  sorry

end arithmetic_sequence_general_formula_l65_65476


namespace distance_from_origin_to_line_l65_65923

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

-- definition of the perpendicular property of chords
def perpendicular (O A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

theorem distance_from_origin_to_line
  (xA yA xB yB : ℝ)
  (hA : ellipse xA yA)
  (hB : ellipse xB yB)
  (h_perpendicular : perpendicular (0, 0) (xA, yA) (xB, yB))
  : ∃ d : ℝ, d = (Real.sqrt 6) / 3 :=
sorry

end distance_from_origin_to_line_l65_65923


namespace king_paid_after_tip_l65_65759

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l65_65759


namespace cost_of_paving_l65_65862

-- declaring the definitions and the problem statement
def length_of_room := 5.5
def width_of_room := 4
def rate_per_sq_meter := 700

theorem cost_of_paving (length : ℝ) (width : ℝ) (rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → (length * width * rate) = 15400 :=
by
  intros h_length h_width h_rate
  rw [h_length, h_width, h_rate]
  sorry

end cost_of_paving_l65_65862


namespace smallest_possible_value_of_d_l65_65537

noncomputable def smallest_value_of_d : ℝ :=
  2 + Real.sqrt 2

theorem smallest_possible_value_of_d (c d : ℝ) (h1 : 2 < c) (h2 : c < d)
    (triangle_condition1 : ¬ (2 + c > d ∧ 2 + d > c ∧ c + d > 2))
    (triangle_condition2 : ¬ ( (2 / d) + (2 / c) > 2)) : d = smallest_value_of_d :=
  sorry

end smallest_possible_value_of_d_l65_65537


namespace square_root_then_square_l65_65741

theorem square_root_then_square (x : ℕ) (hx : x = 49) : (Nat.sqrt x) ^ 2 = 49 := by
  sorry

end square_root_then_square_l65_65741


namespace employees_use_public_transportation_l65_65445

theorem employees_use_public_transportation 
  (total_employees : ℕ)
  (percentage_drive : ℕ)
  (half_of_non_drivers_take_transport : ℕ)
  (h1 : total_employees = 100)
  (h2 : percentage_drive = 60)
  (h3 : half_of_non_drivers_take_transport = 1 / 2) 
  : (total_employees - percentage_drive * total_employees / 100) / 2 = 20 := 
  by
  sorry

end employees_use_public_transportation_l65_65445


namespace degrees_to_minutes_l65_65176

theorem degrees_to_minutes (d : ℚ) (fractional_part : ℚ) (whole_part : ℤ) :
  1 ≤ d ∧ d = fractional_part + whole_part ∧ fractional_part = 0.45 ∧ whole_part = 1 →
  (whole_part + fractional_part) * 60 = 1 * 60 + 27 :=
by { sorry }

end degrees_to_minutes_l65_65176


namespace find_n_l65_65281

theorem find_n (n k : ℕ) (a b : ℝ) (h_pos : k > 0) (h_n : n ≥ 2) (h_ab_neq : a ≠ 0 ∧ b ≠ 0) (h_a : a = (k + 1) * b) : n = 2 * k + 2 :=
by sorry

end find_n_l65_65281


namespace perimeter_of_region_l65_65581

noncomputable def side_length : ℝ := 2 / Real.pi

noncomputable def semicircle_perimeter : ℝ := 2

theorem perimeter_of_region (s : ℝ) (p : ℝ) (h1 : s = 2 / Real.pi) (h2 : p = 2) :
  4 * (p / 2) = 4 :=
by
  sorry

end perimeter_of_region_l65_65581


namespace housewife_spending_l65_65894

theorem housewife_spending
    (R : ℝ) (P : ℝ) (M : ℝ)
    (h1 : R = 25)
    (h2 : R = 0.85 * P)
    (h3 : M / R - M / P = 3) :
  M = 450 :=
by
  sorry

end housewife_spending_l65_65894


namespace find_Tom_favorite_numbers_l65_65990

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_multiple_of (n k : ℕ) : Prop :=
  n % k = 0

def Tom_favorite_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧
  is_multiple_of n 13 ∧
  ¬ is_multiple_of n 3 ∧
  is_multiple_of (sum_of_digits n) 4

theorem find_Tom_favorite_numbers :
  ∃ n : ℕ, Tom_favorite_number n ∧ (n = 130 ∨ n = 143) :=
by
  sorry

end find_Tom_favorite_numbers_l65_65990


namespace opposite_of_neg_3_is_3_l65_65723

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l65_65723


namespace ratio_area_of_circle_to_triangle_l65_65545

theorem ratio_area_of_circle_to_triangle
  (h r b : ℝ)
  (h_triangle : ∃ a, a = b + r ∧ a^2 + b^2 = h^2) :
  (∃ A s : ℝ, s = b + (r + h) / 2 ∧ A = r * s ∧ (∃ circle_area triangle_area : ℝ, circle_area = π * r^2 ∧ triangle_area = 2 * A ∧ circle_area / triangle_area = 2 * π * r / (2 * b + r + h))) :=
by
  sorry

end ratio_area_of_circle_to_triangle_l65_65545


namespace servings_made_l65_65989

noncomputable def chickpeas_per_can := 16 -- ounces in one can
noncomputable def ounces_per_serving := 6 -- ounces needed per serving
noncomputable def total_cans := 8 -- total cans Thomas buys

theorem servings_made : (total_cans * chickpeas_per_can) / ounces_per_serving = 21 :=
by
  sorry

end servings_made_l65_65989


namespace runners_meet_after_3000_seconds_l65_65736

theorem runners_meet_after_3000_seconds :
  let v1 := 45
      v2 := 49
      v3 := 51
      track_length := 600
  ∃ t, t = 3000 ∧
    (∃ k, 0.4 * t = 600 * k) ∧
    (∃ m, 0.2 * t = 600 * m) ∧
    (∃ n, 0.6 * t = 600 * n) :=
by {
  let t := 3000;
  use t;
  split;
  { exact rfl, },
  { split; use 2; norm_num,
    split; use 1; norm_num,
    use 5/3; norm_num, },
}

end runners_meet_after_3000_seconds_l65_65736


namespace hole_digging_problem_l65_65393

theorem hole_digging_problem
  (total_distance : ℕ)
  (original_interval : ℕ)
  (new_interval : ℕ)
  (original_holes : ℕ)
  (new_holes : ℕ)
  (lcm_interval : ℕ)
  (common_holes : ℕ)
  (new_holes_to_be_dug : ℕ)
  (original_holes_discarded : ℕ)
  (h1 : total_distance = 3000)
  (h2 : original_interval = 50)
  (h3 : new_interval = 60)
  (h4 : original_holes = total_distance / original_interval + 1)
  (h5 : new_holes = total_distance / new_interval + 1)
  (h6 : lcm_interval = Nat.lcm original_interval new_interval)
  (h7 : common_holes = total_distance / lcm_interval + 1)
  (h8 : new_holes_to_be_dug = new_holes - common_holes)
  (h9 : original_holes_discarded = original_holes - common_holes) :
  new_holes_to_be_dug = 40 ∧ original_holes_discarded = 50 :=
sorry

end hole_digging_problem_l65_65393


namespace arithmetic_seq_sum_l65_65941

theorem arithmetic_seq_sum (a : ℕ → ℤ) (h_arith_seq : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) (h_a5 : a 5 = 15) : a 2 + a 4 + a 6 + a 8 = 60 := 
by
  sorry

end arithmetic_seq_sum_l65_65941


namespace solution_set_ineq_l65_65013

theorem solution_set_ineq (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end solution_set_ineq_l65_65013


namespace lines_intersection_l65_65147

theorem lines_intersection :
  ∃ (x y : ℝ), 
    (x - y = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end lines_intersection_l65_65147


namespace gina_order_rose_cups_l65_65468

theorem gina_order_rose_cups 
  (rose_cups_per_hour : ℕ) 
  (lily_cups_per_hour : ℕ) 
  (total_lily_cups_order : ℕ) 
  (total_pay : ℕ) 
  (pay_per_hour : ℕ) 
  (total_hours_worked : ℕ) 
  (hours_spent_with_lilies : ℕ)
  (hours_spent_with_roses : ℕ) 
  (rose_cups_order : ℕ) :
  rose_cups_per_hour = 6 →
  lily_cups_per_hour = 7 →
  total_lily_cups_order = 14 →
  total_pay = 90 →
  pay_per_hour = 30 →
  total_hours_worked = total_pay / pay_per_hour →
  hours_spent_with_lilies = total_lily_cups_order / lily_cups_per_hour →
  hours_spent_with_roses = total_hours_worked - hours_spent_with_lilies →
  rose_cups_order = rose_cups_per_hour * hours_spent_with_roses →
  rose_cups_order = 6 := 
by
  sorry

end gina_order_rose_cups_l65_65468


namespace max_reflections_l65_65308

theorem max_reflections (n : ℕ) (angle_CDA : ℝ) (h_angle : angle_CDA = 12) : n ≤ 7 ↔ 12 * n ≤ 90 := by
    sorry

end max_reflections_l65_65308


namespace tan_nine_pi_over_three_l65_65072

theorem tan_nine_pi_over_three : Real.tan (9 * Real.pi / 3) = 0 := by
  sorry

end tan_nine_pi_over_three_l65_65072


namespace garden_perimeter_l65_65766

-- We are given:
variables (a b : ℝ)
variables (h1 : b = 3 * a)
variables (h2 : a^2 + b^2 = 34^2)
variables (h3 : a * b = 240)

-- We must prove:
theorem garden_perimeter (h4 : a^2 + 9 * a^2 = 1156) (h5 : 10 * a^2 = 1156) (h6 : a^2 = 115.6) 
  (h7 : 3 * a^2 = 240) (h8 : a^2 = 80) :
  2 * (a + b) = 72 := 
by
  sorry

end garden_perimeter_l65_65766


namespace construct_segment_AB_l65_65083

-- Define the two points A and B and assume the distance between them is greater than 1 meter
variables {A B : Point} (dist_AB_gt_1m : Distance A B > 1)

-- Define the ruler length as 10 cm
def ruler_length : ℝ := 0.1

theorem construct_segment_AB 
  (h : dist_AB_gt_1m) 
  (ruler : ℝ := ruler_length) : ∃ (AB : Segment), Distance A B = AB.length ∧ AB.length > 1 :=
sorry

end construct_segment_AB_l65_65083


namespace we_the_people_cows_l65_65025

theorem we_the_people_cows (W : ℕ) (h1 : ∃ H : ℕ, H = 3 * W + 2) (h2 : W + 3 * W + 2 = 70) : W = 17 :=
sorry

end we_the_people_cows_l65_65025


namespace fraction_of_height_of_head_l65_65531

theorem fraction_of_height_of_head (h_leg: ℝ) (h_total: ℝ) (h_rest: ℝ) (h_head: ℝ):
  h_leg = 1 / 3 ∧ h_total = 60 ∧ h_rest = 25 ∧ h_head = h_total - (h_leg * h_total + h_rest) 
  → h_head / h_total = 1 / 4 :=
by sorry

end fraction_of_height_of_head_l65_65531


namespace prime_between_30_and_40_has_remainder_7_l65_65765

theorem prime_between_30_and_40_has_remainder_7 (p : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_interval : 30 < p ∧ p < 40) 
  (h_mod : p % 9 = 7) : 
  p = 34 := 
sorry

end prime_between_30_and_40_has_remainder_7_l65_65765


namespace find_multiple_of_q_l65_65236

theorem find_multiple_of_q
  (q : ℕ)
  (x : ℕ := 55 + 2 * q)
  (y : ℕ)
  (m : ℕ)
  (h1 : y = m * q + 41)
  (h2 : x = y)
  (h3 : q = 7) : m = 4 :=
by
  sorry

end find_multiple_of_q_l65_65236


namespace smallest_n_square_average_l65_65165

theorem smallest_n_square_average (n : ℕ) (h : n > 1)
  (S : ℕ := (n * (n + 1) * (2 * n + 1)) / 6)
  (avg : ℕ := S / n) :
  (∃ k : ℕ, avg = k^2) → n = 337 := by
  sorry

end smallest_n_square_average_l65_65165


namespace probability_of_point_within_two_units_l65_65303

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let area_of_circle := 4 * Real.pi
  let area_of_square := 36
  area_of_circle / area_of_square

theorem probability_of_point_within_two_units :
  probability_within_two_units_of_origin = Real.pi / 9 := 
by
  -- The proof steps are omitted as per the requirements
  sorry

end probability_of_point_within_two_units_l65_65303


namespace function_max_min_l65_65495

theorem function_max_min (a b c : ℝ) (h : a ≠ 0) (h1 : ∃ xₘ xₘₐ : ℝ, (0 < xₘ ∧ xₘ < xₘₐ ∧ xₘₐ < ∞) ∧ 
  (∀ x ∈ set.Ioo 0 ∞, dite (f' x = 0) (λ _, differentiable_at ℝ (f' x)) (λ _, true))) :
  (ab : ℝ > 0) ∧ (b^2 + 8ac > 0) ∧ (ac < 0) :=
by
  -- Define the function
  let f := λ x : ℝ, a * log x + b / x + c / x^2
  have h_f_domain : ∀ x, x ∈ set.Ioi (0 : ℝ) → differentiable_at ℝ (f x),
    from sorry
  have h_f_deriv : ∀ x, x ∈ set.Ioi (0 : ℝ) → deriv (f x) = a / x - b / x^2 - 2 * c / x^3,
    from sorry
  have h_f_critical : ∀ x, deriv (f x) = 0 → ∃ xₘ xₘₐ, (xₘ * xₘₐ) > 0 ∧ fourier.coefficients xₘ + xₘₐ > 0,
    from sorry
  show  (ab : ℝ > 0) ∧ (b^2 + 8ac > 0) ∧ (ac < 0),
    from sorry

end function_max_min_l65_65495


namespace focus_of_parabola_l65_65076

theorem focus_of_parabola (p : ℝ) :
  (∃ p, x ^ 2 = 4 * p * y ∧ x ^ 2 = 4 * 1 * y) → (0, p) = (0, 1) :=
by
  sorry

end focus_of_parabola_l65_65076


namespace sum_of_coefficients_l65_65009

theorem sum_of_coefficients :
  ∃ (A B C D E F G H J K : ℤ),
  (∀ x y : ℤ, 125 * x ^ 8 - 2401 * y ^ 8 = (A * x + B * y) * (C * x ^ 4 + D * x * y + E * y ^ 4) * (F * x + G * y) * (H * x ^ 4 + J * x * y + K * y ^ 4))
  ∧ A + B + C + D + E + F + G + H + J + K = 102 := 
sorry

end sum_of_coefficients_l65_65009


namespace symbols_invariance_l65_65691

def final_symbol_invariant (symbols : List Char) : Prop :=
  ∀ (erase : List Char → List Char), 
  (∀ (l : List Char), 
    (erase l = List.cons '+' (List.tail (List.tail l)) ∨ 
    erase l = List.cons '-' (List.tail (List.tail l))) → 
    erase (erase l) = List.cons '+' (List.tail (List.tail (erase l))) ∨ 
    erase (erase l) = List.cons '-' (List.tail (List.tail (erase l)))) →
  (symbols = []) ∨ (symbols = ['+']) ∨ (symbols = ['-'])

theorem symbols_invariance (symbols : List Char) (h : final_symbol_invariant symbols) : 
  ∃ (s : Char), s = '+' ∨ s = '-' :=
  sorry

end symbols_invariance_l65_65691


namespace cos_beta_value_l65_65128

variable (α β : ℝ)
variable (h₁ : 0 < α ∧ α < π)
variable (h₂ : 0 < β ∧ β < π)
variable (h₃ : Real.sin (α + β) = 5 / 13)
variable (h₄ : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_value : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_value_l65_65128


namespace yoongi_number_division_l65_65881

theorem yoongi_number_division (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end yoongi_number_division_l65_65881


namespace louie_pie_share_l65_65777

theorem louie_pie_share :
  let leftover := (6 : ℝ) / 7
  let people := 3
  leftover / people = (2 : ℝ) / 7 := 
by
  sorry

end louie_pie_share_l65_65777


namespace knight_moves_equal_n_seven_l65_65652

def knight_moves (n : ℕ) : ℕ := sorry -- Function to calculate the minimum number of moves for a knight.

theorem knight_moves_equal_n_seven :
  ∀ {n : ℕ}, n = 7 →
    knight_moves n = knight_moves n := by
  -- Conditions: Position on standard checkerboard 
  -- and the knight moves described above.
  sorry

end knight_moves_equal_n_seven_l65_65652


namespace simplify_expression_l65_65404

theorem simplify_expression :
  15 * (18 / 5) * (-42 / 45) = -50.4 :=
by
  sorry

end simplify_expression_l65_65404


namespace center_of_circle_l65_65706

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (3, 8)) (h2 : (x2, y2) = (11, -4)) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 2) := by
  sorry

end center_of_circle_l65_65706


namespace skyscraper_anniversary_l65_65111

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l65_65111


namespace marikas_father_twice_her_age_l65_65679

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l65_65679


namespace simple_interest_initial_amount_l65_65589

theorem simple_interest_initial_amount :
  ∃ P : ℝ, (P + P * 0.04 * 5 = 900) ∧ P = 750 :=
by
  sorry

end simple_interest_initial_amount_l65_65589


namespace four_letter_product_eq_l65_65611

open Nat

def letter_value (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26 | _ => 0

theorem four_letter_product_eq (list1 : List Char) (list2 : List Char) :
  (list1 = ['T', 'U', 'W', 'Y']) →
  (list1.foldr (fun c acc => acc * (letter_value c)) 1 = list2.foldr (fun c acc => acc * (letter_value c)) 1) →
  (list2 = ['N', 'O', 'W', 'Y']) :=
by
  intros h h_eq
  -- Proof steps here
  sorry

end four_letter_product_eq_l65_65611


namespace average_grade_of_male_students_l65_65146

theorem average_grade_of_male_students (M : ℝ) (H1 : (90 : ℝ) = (8 + 32 : ℝ) / 40) 
(H2 : (92 : ℝ) = 32 / 40) :
  M = 82 := 
sorry

end average_grade_of_male_students_l65_65146


namespace possible_values_of_k_l65_65500

-- Definition of the proposition
def proposition (k : ℝ) : Prop :=
  ∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0

-- The main statement to prove in Lean 4
theorem possible_values_of_k (k : ℝ) : ¬ proposition k ↔ (k = 1 ∨ (1 < k ∧ k < 7)) :=
by 
  sorry

end possible_values_of_k_l65_65500


namespace cups_added_l65_65040

/--
A bowl was half full of water. Some cups of water were then added to the bowl, filling the bowl to 70% of its capacity. There are now 14 cups of water in the bowl.
Prove that the number of cups of water added to the bowl is 4.
-/
theorem cups_added (C : ℚ) (h1 : C / 2 + 0.2 * C = 14) : 
  14 - C / 2 = 4 :=
by
  sorry

end cups_added_l65_65040


namespace acid_volume_16_liters_l65_65032

theorem acid_volume_16_liters (V A_0 B_0 A_1 B_1 : ℝ) 
  (h_initial_ratio : 4 * B_0 = A_0)
  (h_initial_volume : A_0 + B_0 = V)
  (h_remove_mixture : 10 * A_0 / V = A_1)
  (h_remove_mixture_base : 10 * B_0 / V = B_1)
  (h_new_A : A_1 = A_0 - 8)
  (h_new_B : B_1 = B_0 - 2 + 10)
  (h_new_ratio : 2 * B_1 = 3 * A_1) :
  A_0 = 16 :=
by {
  -- Here we will have the proof steps, which are omitted.
  sorry
}

end acid_volume_16_liters_l65_65032


namespace parabola_symmetric_points_l65_65114

theorem parabola_symmetric_points (a : ℝ) (h : 0 < a) :
  (∃ (P Q : ℝ × ℝ), (P ≠ Q) ∧ ((P.fst + P.snd = 0) ∧ (Q.fst + Q.snd = 0)) ∧
    (P.snd = a * P.fst ^ 2 - 1) ∧ (Q.snd = a * Q.fst ^ 2 - 1)) ↔ (3 / 4 < a) := 
sorry

end parabola_symmetric_points_l65_65114


namespace unbalanced_primitive_integer_squares_infinite_l65_65662

theorem unbalanced_primitive_integer_squares_infinite :
  ∃ (B D : ℤ × ℤ × ℤ), 
  (gcd B.1.1 (gcd B.1.2 B.2) = 1 ∧ gcd D.1.1 (gcd D.1.2 D.2) = 1) ∧
  (abs B.1.1 + abs B.1.2 + abs B.2 ≠ abs D.1.1 + abs D.1.2 + abs D.2) ∧
  ∀ t : ℤ, B.1.1 ^ 2 + B.1.2 ^ 2 + B.2 ^ 2 = t ^ 2 ∧ D.1.1 ^ 2 + D.1.2 ^ 2 + D.2 ^ 2 = t ^ 2 ∧
  B.1.1 * D.1.1 + B.1.2 * D.1.2 + B.2 * D.2 = 0 ∧
  (∀ c : ℤ, c ≠ 0 → B ≠ c • D) :=
sorry

end unbalanced_primitive_integer_squares_infinite_l65_65662


namespace third_angle_of_triangle_l65_65863

theorem third_angle_of_triangle (a b : ℝ) (h₁ : a = 25) (h₂ : b = 70) : 180 - a - b = 85 := 
by
  sorry

end third_angle_of_triangle_l65_65863


namespace cost_of_each_book_l65_65387

theorem cost_of_each_book 
  (B : ℝ)
  (num_books_plant : ℕ)
  (num_books_fish : ℕ)
  (num_magazines : ℕ)
  (cost_magazine : ℝ)
  (total_spent : ℝ) :
  num_books_plant = 9 →
  num_books_fish = 1 →
  num_magazines = 10 →
  cost_magazine = 2 →
  total_spent = 170 →
  10 * B + 10 * cost_magazine = total_spent →
  B = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cost_of_each_book_l65_65387


namespace triangle_angle_sum_l65_65241

theorem triangle_angle_sum (x : ℝ) :
  let a := 40
  let b := 60
  let sum_of_angles := 180
  a + b + x = sum_of_angles → x = 80 :=
by
  intros
  sorry

end triangle_angle_sum_l65_65241


namespace men_required_l65_65232

variable (m w : ℝ) -- Work done by one man and one woman in one day respectively
variable (x : ℝ) -- Number of men

-- Conditions from the problem
def condition1 (m w : ℝ) (x : ℝ) : Prop :=
  x * m = 12 * w

def condition2 (m w : ℝ) : Prop :=
  (6 * m + 11 * w) * 12 = 1

-- Proving that the number of men required to do the work in 20 days is x
theorem men_required (m w : ℝ) (x : ℝ) (h1 : condition1 m w x) (h2 : condition2 m w) : 
  (∃ x, condition1 m w x ∧ condition2 m w) := 
sorry

end men_required_l65_65232


namespace exists_x_for_bounded_positive_measure_set_l65_65851

open MeasureTheory

theorem exists_x_for_bounded_positive_measure_set (E : Set ℝ)
  (hE : MeasurableSet E)
  (hE_bounded : Bounded E)
  (hE_positive_measure : 0 < volume E) :
  ∀ u < (1 : ℝ) / 2,
    ∃ x : ℝ, ∀ ε > 0, ∃ δ > 0, δ < ε →
      (volume ((Ioo (x - δ) (x + δ)) ∩ E) ≥ u * δ ∧ 
      volume ((Ioo (x - δ) (x + δ)) ∩ (univ \ E)) ≥ u * δ) :=
by
  sorry

end exists_x_for_bounded_positive_measure_set_l65_65851


namespace year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l65_65880

-- Define what it means to be a leap year based on the given conditions.
def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ (y % 4 = 0 ∧ y % 100 ≠ 0)

-- Define the specific years we are examining.
def year_1800 := 1800
def year_1992 := 1992
def year_1994 := 1994
def year_2040 := 2040

-- Assertions about whether each year is a leap year or a common year
theorem year_1800_is_common : ¬ is_leap_year year_1800 :=
  by sorry

theorem year_1992_is_leap : is_leap_year year_1992 :=
  by sorry

theorem year_1994_is_common : ¬ is_leap_year year_1994 :=
  by sorry

theorem year_2040_is_leap : is_leap_year year_2040 :=
  by sorry

end year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l65_65880


namespace unique_k_value_l65_65603
noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m ∣ n → m = n

theorem unique_k_value :
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 74 ∧ p * q = 213) ∧
  ∀ (p₁ q₁ k₁ p₂ q₂ k₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ p₁ + q₁ = 74 ∧ p₁ * q₁ = k₁ ∧
    is_prime p₂ ∧ is_prime q₂ ∧ p₂ + q₂ = 74 ∧ p₂ * q₂ = k₂ →
    k₁ = k₂ :=
by
  sorry

end unique_k_value_l65_65603


namespace halfway_fraction_l65_65997

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65997


namespace average_percentage_of_15_students_l65_65440

open Real

theorem average_percentage_of_15_students :
  ∀ (x : ℝ),
  (15 + 10 = 25) →
  (10 * 90 = 900) →
  (25 * 84 = 2100) →
  (15 * x + 900 = 2100) →
  x = 80 :=
by
  intro x h_sum h_10_avg h_25_avg h_total
  sorry

end average_percentage_of_15_students_l65_65440


namespace find_point_A_l65_65651

-- Definitions of the conditions
def point_A_left_translated_to_B (A B : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, A.1 - l = B.1 ∧ A.2 = B.2

def point_A_upward_translated_to_C (A C : ℝ × ℝ) : Prop :=
  ∃ u : ℝ, A.1 = C.1 ∧ A.2 + u = C.2

-- Given points B and C
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 4)

-- The statement to prove the coordinates of point A
theorem find_point_A (A : ℝ × ℝ) : 
  point_A_left_translated_to_B A B ∧ point_A_upward_translated_to_C A C → A = (3, 2) :=
by 
  sorry

end find_point_A_l65_65651


namespace money_left_over_l65_65735

theorem money_left_over 
  (num_books : ℕ) 
  (price_per_book : ℝ) 
  (num_records : ℕ) 
  (price_per_record : ℝ) 
  (total_books : num_books = 200) 
  (book_price : price_per_book = 1.5) 
  (total_records : num_records = 75) 
  (record_price : price_per_record = 3) :
  (num_books * price_per_book - num_records * price_per_record) = 75 :=
by 
  -- calculation
  sorry

end money_left_over_l65_65735


namespace probability_not_monday_l65_65979

theorem probability_not_monday (P_monday : ℚ) (h : P_monday = 1/7) : P_monday ≠ 1 → ∃ P_not_monday : ℚ, P_not_monday = 6/7 :=
by
  sorry

end probability_not_monday_l65_65979


namespace calculation_result_l65_65907

theorem calculation_result : 
  2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := 
by 
  sorry

end calculation_result_l65_65907


namespace payment_relationship_l65_65727

noncomputable def payment_amount (x : ℕ) (price_per_book : ℕ) (discount_percent : ℕ) : ℕ :=
  if x > 20 then ((x - 20) * (price_per_book * (100 - discount_percent) / 100) + 20 * price_per_book) else x * price_per_book

theorem payment_relationship (x : ℕ) (h : x > 20) : payment_amount x 25 20 = 20 * x + 100 := 
by
  sorry

end payment_relationship_l65_65727


namespace find_parabola_coeffs_l65_65262

def parabola_vertex_form (a b c : ℝ) : Prop :=
  ∃ k:ℝ, k = c - b^2 / (4*a) ∧ k = 3

def parabola_through_point (a b c : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, x = 0 ∧ y = 1 ∧  y = a * x^2 + b * x + c

theorem find_parabola_coeffs :
  ∃ a b c : ℝ, parabola_vertex_form a b c ∧ parabola_through_point a b c ∧
  a = -1/2 ∧ b = 2 ∧ c = 1 :=
by
  sorry

end find_parabola_coeffs_l65_65262


namespace probability_of_Q_l65_65307

noncomputable def probability_Q_within_two_units_of_origin : ℚ :=
  let side_length_square := 6
  let area_square := side_length_square ^ 2
  let radius_circle := 2
  let area_circle := π * radius_circle ^ 2
  area_circle / area_square

theorem probability_of_Q :
  probability_Q_within_two_units_of_origin = π / 9 :=
by
  -- The proof would go here
  sorry

end probability_of_Q_l65_65307


namespace circle_radius_l65_65588

/-- Let a circle have a maximum distance of 11 cm and a minimum distance of 5 cm from a point P.
Prove that the radius of the circle can be either 3 cm or 8 cm. -/
theorem circle_radius (max_dist min_dist : ℕ) (h_max : max_dist = 11) (h_min : min_dist = 5) :
  (∃ r : ℕ, r = 3 ∨ r = 8) :=
by
  sorry

end circle_radius_l65_65588


namespace middle_number_of_pairs_l65_65425

theorem middle_number_of_pairs (x y z : ℕ) (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 21) : y = 9 := 
by
  sorry

end middle_number_of_pairs_l65_65425


namespace circle_radius_l65_65002

theorem circle_radius (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) (h2 : A = Real.pi * r ^ 2) : r = 6 :=
sorry

end circle_radius_l65_65002


namespace rate_of_current_l65_65984

theorem rate_of_current (c : ℝ) (h1 : 7.5 = (20 + c) * 0.3) : c = 5 :=
by
  sorry

end rate_of_current_l65_65984


namespace constant_speed_total_distance_l65_65866

def travel_time : ℝ := 5.5
def distance_per_hour : ℝ := 100
def speed := distance_per_hour

theorem constant_speed : ∀ t : ℝ, (1 ≤ t) ∧ (t ≤ travel_time) → speed = distance_per_hour := 
by sorry

theorem total_distance : speed * travel_time = 550 :=
by sorry

end constant_speed_total_distance_l65_65866


namespace conditions_for_local_extrema_l65_65491

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * log x + b / x + c / (x^2)

theorem conditions_for_local_extrema
  (a b c : ℝ) (ha : a ≠ 0) (D : ℝ → ℝ) (hD : ∀ x, D x = deriv (f a b c) x) :
  (∀ x > 0, D x = (a * x^2 - b * x - 2 * c) / x^3) →
  (∃ x y > 0, D x = 0 ∧ D y = 0 ∧ x ≠ y) ↔
    (a * b > 0 ∧ a * c < 0 ∧ b^2 + 8 * a * c > 0) :=
sorry

end conditions_for_local_extrema_l65_65491


namespace jim_travel_distance_l65_65821

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end jim_travel_distance_l65_65821


namespace volume_of_cuboid_l65_65233

theorem volume_of_cuboid (l w h : ℝ) (hlw: l * w = 120) (hwh: w * h = 72) (hhl: h * l = 60) : l * w * h = 720 :=
  sorry

end volume_of_cuboid_l65_65233


namespace race_min_distance_l65_65982

noncomputable def min_distance : ℝ :=
  let A : ℝ × ℝ := (0, 300)
  let B : ℝ × ℝ := (1200, 500)
  let wall_length : ℝ := 1200
  let B' : ℝ × ℝ := (1200, -500)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B'

theorem race_min_distance :
  min_distance = 1442 := sorry

end race_min_distance_l65_65982


namespace river_current_speed_l65_65311

theorem river_current_speed :
  ∀ (D v A_speed B_speed time_interval : ℝ),
    D = 200 →
    A_speed = 36 →
    B_speed = 64 →
    time_interval = 4 →
    3 * D = (A_speed + v) * 2 * (1 + time_interval / ((A_speed + v) + (B_speed - v))) * 200 :=
sorry

end river_current_speed_l65_65311


namespace opposite_of_neg3_l65_65721

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l65_65721


namespace muffin_is_twice_as_expensive_as_banana_l65_65001

variable (m b : ℚ)
variable (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12)
variable (h2 : 3 * m + 5 * b = S)

theorem muffin_is_twice_as_expensive_as_banana (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12) : m = 2 * b :=
by
  sorry

end muffin_is_twice_as_expensive_as_banana_l65_65001


namespace tan_beta_minus_2alpha_l65_65367

open Real

-- Given definitions
def condition1 (α : ℝ) : Prop :=
  (sin α * cos α) / (1 - cos (2 * α)) = 1 / 4

def condition2 (α β : ℝ) : Prop :=
  tan (α - β) = 2

-- Proof problem statement
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : condition1 α) (h2 : condition2 α β) :
  tan (β - 2 * α) = 4 / 3 :=
sorry

end tan_beta_minus_2alpha_l65_65367


namespace system_solutions_l65_65792

theorem system_solutions : 
  ∃ (x y z t : ℝ), 
    (x * y - t^2 = 9) ∧ 
    (x^2 + y^2 + z^2 = 18) ∧ 
    ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ 
     (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by {
  sorry
}

end system_solutions_l65_65792


namespace length_of_crate_l65_65443

theorem length_of_crate (h crate_dim : ℕ) (radius : ℕ) (h_radius : radius = 8) 
  (h_dims : crate_dim = 18) (h_fit : 2 * radius = 16)
  : h = 18 := 
sorry

end length_of_crate_l65_65443


namespace father_twice_marika_age_in_2036_l65_65684

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l65_65684


namespace arithmetic_sequence_sum_l65_65529

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1))
  (h2 : S 3 = 6)
  (h3 : a 3 = 3) :
  S 2023 / 2023 = 1012 := by
  sorry

end arithmetic_sequence_sum_l65_65529


namespace max_cars_with_ac_but_not_rs_l65_65831

namespace CarProblem

variables (total_cars : ℕ) 
          (cars_without_ac : ℕ)
          (cars_with_rs : ℕ)
          (cars_with_ac : ℕ := total_cars - cars_without_ac)
          (cars_with_ac_and_rs : ℕ)
          (cars_with_ac_but_not_rs : ℕ := cars_with_ac - cars_with_ac_and_rs)

theorem max_cars_with_ac_but_not_rs 
        (h1 : total_cars = 100)
        (h2 : cars_without_ac = 37)
        (h3 : cars_with_rs ≥ 51)
        (h4 : cars_with_ac_and_rs = min cars_with_rs cars_with_ac) :
        cars_with_ac_but_not_rs = 12 := by
    sorry

end CarProblem

end max_cars_with_ac_but_not_rs_l65_65831


namespace neither_plaid_nor_purple_l65_65700

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end neither_plaid_nor_purple_l65_65700


namespace volleyball_team_selection_l65_65259

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (n.choose k)

theorem volleyball_team_selection : 
  let quadruplets := ["Bella", "Bianca", "Becca", "Brooke"];
  let total_players := 16;
  let starters := 7;
  let num_quadruplets := quadruplets.length;
  ∃ ways : ℕ, 
    ways = binom num_quadruplets 3 * binom (total_players - num_quadruplets) (starters - 3) 
    ∧ ways = 1980 :=
by
  sorry

end volleyball_team_selection_l65_65259


namespace rearrangement_impossible_l65_65655

-- Define the primary problem conditions and goal
theorem rearrangement_impossible :
  ¬ ∃ (f : Fin 100 → Fin 51), 
    (∀ k : Fin 51, ∃ i j : Fin 100, 
      f i = k ∧ f j = k ∧ (i < j ∧ j.val - i.val = k.val + 1)) :=
sorry

end rearrangement_impossible_l65_65655


namespace parabola_directrix_l65_65860

theorem parabola_directrix (x y : ℝ) (h : x^2 = 2 * y) : y = -1 / 2 := 
  sorry

end parabola_directrix_l65_65860


namespace probability_two_units_of_origin_l65_65304

def square_vertices (x_min x_max y_min y_max : ℝ) :=
  { p : ℝ × ℝ // x_min ≤ p.1 ∧ p.1 ≤ x_max ∧ y_min ≤ p.2 ∧ p.2 ≤ y_max }

def within_radius (r : ℝ) (origin : ℝ × ℝ) (p : ℝ × ℝ) :=
  (p.1 - origin.1)^2 + (p.2 - origin.2)^2 ≤ r^2

noncomputable def probability_within_radius (x_min x_max y_min y_max r : ℝ) : ℝ :=
  let square_area := (x_max - x_min) * (y_max - y_min)
  let circle_area := r^2 * Real.pi
  circle_area / square_area

theorem probability_two_units_of_origin :
  probability_within_radius (-3) 3 (-3) 3 2 = Real.pi / 9 :=
by
  sorry

end probability_two_units_of_origin_l65_65304


namespace inverse_variation_l65_65966

theorem inverse_variation (a b k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) : (∃ a, b = 4 → a = 1 / 8) :=
by
  sorry

end inverse_variation_l65_65966


namespace find_inverse_modulo_l65_65790

theorem find_inverse_modulo :
  113 * 113 ≡ 1 [MOD 114] :=
by
  sorry

end find_inverse_modulo_l65_65790


namespace opposite_of_neg_3_l65_65718

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l65_65718


namespace game_winning_strategy_l65_65051

theorem game_winning_strategy (n : ℕ) (h : n ≥ 3) :
  (∃ k : ℕ, n = 3 * k + 2) → (∃ k : ℕ, n = 3 * k + 2 ∨ ∀ k : ℕ, n ≠ 3 * k + 2) :=
by
  sorry

end game_winning_strategy_l65_65051


namespace root_triple_condition_l65_65622

theorem root_triple_condition (a b c α β : ℝ)
  (h_eq : a * α^2 + b * α + c = 0)
  (h_β_eq : β = 3 * α)
  (h_vieta_sum : α + β = -b / a)
  (h_vieta_product : α * β = c / a) :
  3 * b^2 = 16 * a * c :=
by
  sorry

end root_triple_condition_l65_65622


namespace shirts_and_pants_neither_plaid_nor_purple_l65_65702

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end shirts_and_pants_neither_plaid_nor_purple_l65_65702


namespace ratio_of_numbers_l65_65732

theorem ratio_of_numbers (A B D M : ℕ) 
  (h1 : A + B + D = M)
  (h2 : Nat.gcd A B = D)
  (h3 : Nat.lcm A B = M)
  (h4 : A ≥ B) : A / B = 3 / 2 :=
by
  sorry

end ratio_of_numbers_l65_65732


namespace largest_n_is_253_l65_65191

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

end largest_n_is_253_l65_65191


namespace tan_alpha_l65_65081

theorem tan_alpha (α : ℝ) (hα1 : α > π / 2) (hα2 : α < π) (h_sin : Real.sin α = 4 / 5) : Real.tan α = - (4 / 3) :=
by 
  sorry

end tan_alpha_l65_65081


namespace f_99_eq_1_l65_65352

-- Define an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- The conditions to be satisfied by the function f
variables (f : ℝ → ℝ)
variable (h_even : is_even_function f)
variable (h_f1 : f 1 = 1)
variable (h_period : ∀ x, f (x + 4) = f x)

-- Prove that f(99) = 1
theorem f_99_eq_1 : f 99 = 1 :=
by
  sorry

end f_99_eq_1_l65_65352


namespace rectangle_area_pairs_l65_65839

theorem rectangle_area_pairs :
  { p : ℕ × ℕ | p.1 * p.2 = 12 ∧ p.1 > 0 ∧ p.2 > 0 } = { (1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1) } :=
by {
  sorry
}

end rectangle_area_pairs_l65_65839


namespace prove_k_eq_5_l65_65257

variable (a b k : ℕ)

theorem prove_k_eq_5 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (a^2 - 1 - b^2) / (a * b - 1) = k) : k = 5 :=
sorry

end prove_k_eq_5_l65_65257


namespace proposition_a_proposition_b_proposition_c_proposition_d_l65_65817

variable (a b c : ℝ)

-- Proposition A: If ac^2 > bc^2, then a > b
theorem proposition_a (h : a * c^2 > b * c^2) : a > b := sorry

-- Proposition B: If a > b, then ac^2 > bc^2
theorem proposition_b (h : a > b) : ¬ (a * c^2 > b * c^2) := sorry

-- Proposition C: If a > b, then 1/a < 1/b
theorem proposition_c (h : a > b) : ¬ (1/a < 1/b) := sorry

-- Proposition D: If a > b > 0, then a^2 > ab > b^2
theorem proposition_d (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 := sorry

end proposition_a_proposition_b_proposition_c_proposition_d_l65_65817


namespace calculate_integral_cos8_l65_65198

noncomputable def integral_cos8 : ℝ :=
  ∫ x in (Real.pi / 2)..(2 * Real.pi), 2^8 * (Real.cos x)^8

theorem calculate_integral_cos8 :
  integral_cos8 = 219 * Real.pi :=
by
  sorry

end calculate_integral_cos8_l65_65198


namespace sophia_age_in_three_years_l65_65154

def current_age_jeremy : Nat := 40
def current_age_sebastian : Nat := current_age_jeremy + 4

def sum_ages_in_three_years (age_jeremy age_sebastian age_sophia : Nat) : Nat :=
  (age_jeremy + 3) + (age_sebastian + 3) + (age_sophia + 3)

theorem sophia_age_in_three_years (age_sophia : Nat) 
  (h1 : sum_ages_in_three_years current_age_jeremy current_age_sebastian age_sophia = 150) :
  age_sophia + 3 = 60 := by
  sorry

end sophia_age_in_three_years_l65_65154


namespace remaining_digits_product_l65_65157

theorem remaining_digits_product (a b c : ℕ)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (b + c) % 10 = a % 10)
  (h3 : (c + a) % 10 = b % 10) :
  ((a * b * c) % 1000 = 0 ∨
   (a * b * c) % 1000 = 250 ∨
   (a * b * c) % 1000 = 500 ∨
   (a * b * c) % 1000 = 750) :=
sorry

end remaining_digits_product_l65_65157


namespace cos_double_angle_identity_l65_65470

variable (α : Real)

theorem cos_double_angle_identity (h : Real.sin (Real.pi / 6 + α) = 1/3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7/9 :=
by
  sorry

end cos_double_angle_identity_l65_65470


namespace january_revenue_fraction_l65_65185

theorem january_revenue_fraction (N D J : ℚ) 
  (h1 : N = (3 / 5) * D)
  (h2 : D = (20 / 7) * (N + J) / 2) :
  J / N = 1 / 6 :=
sorry

end january_revenue_fraction_l65_65185


namespace halfway_fraction_l65_65998

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65998


namespace probability_of_green_ball_is_correct_l65_65333

-- Defining the conditions
def prob_container_selected : ℚ := 1 / 3
def prob_green_ball_in_A : ℚ := 7 / 10
def prob_green_ball_in_B : ℚ := 5 / 10
def prob_green_ball_in_C : ℚ := 5 / 10

-- Defining each case's probability of drawing a green ball
def prob_A_and_green : ℚ := prob_container_selected * prob_green_ball_in_A
def prob_B_and_green : ℚ := prob_container_selected * prob_green_ball_in_B
def prob_C_and_green : ℚ := prob_container_selected * prob_green_ball_in_C

-- The overall probability that a green ball is selected
noncomputable def total_prob_green : ℚ := prob_A_and_green + prob_B_and_green + prob_C_and_green

-- The theorem to be proved
theorem probability_of_green_ball_is_correct : total_prob_green = 17 / 30 := 
by
  sorry

end probability_of_green_ball_is_correct_l65_65333


namespace dark_squares_exceed_light_squares_by_one_l65_65568

theorem dark_squares_exceed_light_squares_by_one :
  let dark_squares := 25
  let light_squares := 24
  dark_squares - light_squares = 1 :=
by
  sorry

end dark_squares_exceed_light_squares_by_one_l65_65568


namespace max_value_of_f_l65_65340

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

theorem max_value_of_f (x : ℝ) (h : 0 ≤ x) : ∃ M, M = 12 ∧ ∀ y, 0 ≤ y → f y ≤ M :=
sorry

end max_value_of_f_l65_65340


namespace polynomial_never_33_l65_65136

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by
  sorry

end polynomial_never_33_l65_65136


namespace function_max_min_l65_65489

theorem function_max_min (a b c : ℝ) (h_a : a ≠ 0) (h_sum_pos : a * b > 0) (h_discriminant_pos : b^2 + 8 * a * c > 0) (h_product_neg : a * c < 0) : 
  (∀ x > 0, ∃ x1 x2 > 0, x1 + x2 = b / a ∧ x1 * x2 = -2 * c / a) := 
sorry

end function_max_min_l65_65489


namespace abcdeq_five_l65_65073

theorem abcdeq_five (a b c d : ℝ) 
    (h1 : a + b + c + d = 20) 
    (h2 : ab + ac + ad + bc + bd + cd = 150) : 
    a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5 := 
  by
  sorry

end abcdeq_five_l65_65073


namespace frames_per_page_l65_65381

theorem frames_per_page (total_frames : ℕ) (total_pages : ℝ) (h1 : total_frames = 1573) (h2 : total_pages = 11.0) : total_frames / total_pages = 143 := by
  sorry

end frames_per_page_l65_65381


namespace skateboarder_speed_l65_65856

-- Defining the conditions
def distance_feet : ℝ := 476.67
def time_seconds : ℝ := 25
def feet_per_mile : ℝ := 5280
def seconds_per_hour : ℝ := 3600

-- Defining the expected speed in miles per hour
def expected_speed_mph : ℝ := 13.01

-- The problem statement: Prove that the skateboarder's speed is 13.01 mph given the conditions
theorem skateboarder_speed : (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour) = expected_speed_mph := by
  sorry

end skateboarder_speed_l65_65856


namespace probability_is_1_over_90_l65_65836

/-- Probability Calculation -/
noncomputable def probability_of_COLD :=
  (1 / (Nat.choose 5 3)) * (2 / 3) * (1 / (Nat.choose 4 2))

theorem probability_is_1_over_90 :
  probability_of_COLD = (1 / 90) :=
by
  sorry

end probability_is_1_over_90_l65_65836


namespace value_of_a_l65_65101

theorem value_of_a (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) (h3 : a > b) (h4 : a - b = 8) : a = 10 := 
by 
sorry

end value_of_a_l65_65101


namespace quadratic_expression_value_l65_65629

variable (x y : ℝ)

theorem quadratic_expression_value (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 100 := 
by 
  sorry

end quadratic_expression_value_l65_65629


namespace profit_percent_calculation_l65_65645

variable (SP : ℝ) (CP : ℝ) (Profit : ℝ) (ProfitPercent : ℝ)
variable (h1 : CP = 0.75 * SP)
variable (h2 : Profit = SP - CP)
variable (h3 : ProfitPercent = (Profit / CP) * 100)

theorem profit_percent_calculation : ProfitPercent = 33.33 := 
sorry

end profit_percent_calculation_l65_65645


namespace first_present_cost_is_18_l65_65124

-- Conditions as definitions
variables (x : ℕ)

-- Given conditions
def first_present_cost := x
def second_present_cost := x + 7
def third_present_cost := x - 11
def total_cost := first_present_cost x + second_present_cost x + third_present_cost x

-- Statement of the problem
theorem first_present_cost_is_18 (h : total_cost x = 50) : x = 18 :=
by {
  sorry  -- Proof omitted
}

end first_present_cost_is_18_l65_65124


namespace min_days_to_plant_trees_l65_65582

theorem min_days_to_plant_trees (n : ℕ) (h : 2 ≤ n) :
  (2 ^ (n + 1) - 2 ≥ 1000) ↔ (n ≥ 9) :=
by sorry

end min_days_to_plant_trees_l65_65582


namespace problem1_problem2_l65_65749

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 - 3 * x

theorem problem1 (a : ℝ) : (∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → a ≤ 0 :=
sorry

theorem problem2 (a : ℝ) (h : a = 6) :
  x = 3 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → f x 6 ≤ -6 ∧ f x 6 ≥ -18) :=
sorry

end problem1_problem2_l65_65749


namespace no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l65_65436

-- Definition of a natural number being divisible by another
def divisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

-- Definition of the sum of the digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

-- Statement of the problem
theorem no_nat_number_divisible_by_1998_has_digit_sum_lt_27 :
  ¬ ∃ n : ℕ, divisible n 1998 ∧ sum_of_digits n < 27 :=
by 
  sorry

end no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l65_65436


namespace condition_on_a_b_l65_65953

theorem condition_on_a_b (a b : ℝ) (h : a^2 * b^2 + 5 > 2 * a * b - a^2 - 4 * a) : ab ≠ 1 ∨ a ≠ -2 :=
by
  sorry

end condition_on_a_b_l65_65953


namespace floor_add_self_eq_14_5_iff_r_eq_7_5_l65_65074

theorem floor_add_self_eq_14_5_iff_r_eq_7_5 (r : ℝ) : 
  (⌊r⌋ + r = 14.5) ↔ r = 7.5 :=
by
  sorry

end floor_add_self_eq_14_5_iff_r_eq_7_5_l65_65074


namespace extrema_of_function_l65_65466

noncomputable def f (x : ℝ) := x / 8 + 2 / x

theorem extrema_of_function : 
  ∀ x ∈ Set.Ioo (-5 : ℝ) (10),
  (x ≠ 0) →
  (f (-4) = -1 ∧ f 4 = 1) ∧
  (∀ x ∈ Set.Ioc (-5) 0, f x ≤ -1) ∧
  (∀ x ∈ Set.Ioo 0 10, f x ≥ 1) := by
  sorry

end extrema_of_function_l65_65466


namespace no_14_non_square_rectangles_l65_65770

theorem no_14_non_square_rectangles (side_len : ℕ) 
    (h_side_len : side_len = 9) 
    (num_rectangles : ℕ) 
    (h_num_rectangles : num_rectangles = 14) 
    (min_side_len : ℕ → ℕ → Prop) 
    (h_min_side_len : ∀ l w, min_side_len l w → l ≥ 2 ∧ w ≥ 2) : 
    ¬ (∀ l w, min_side_len l w → l ≠ w) :=
by {
    sorry
}

end no_14_non_square_rectangles_l65_65770


namespace bus_arrives_on_time_exactly_4_times_out_of_5_l65_65865

noncomputable theory

def bus_on_time_probability (p : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem bus_arrives_on_time_exactly_4_times_out_of_5 :
  bus_on_time_probability 0.9 4 5 = 0.328 := by
  sorry

end bus_arrives_on_time_exactly_4_times_out_of_5_l65_65865


namespace otimes_identity_l65_65912

-- Define the operation ⊗
def otimes (k l : ℝ) : ℝ := k^2 - l^2

-- The goal is to show k ⊗ (k ⊗ k) = k^2 for any real number k
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 :=
by sorry

end otimes_identity_l65_65912


namespace ratio_of_Katie_to_Cole_l65_65168

variable (K C : ℕ)

theorem ratio_of_Katie_to_Cole (h1 : 3 * K = 84) (h2 : C = 7) : K / C = 4 :=
by
  sorry

end ratio_of_Katie_to_Cole_l65_65168


namespace price_of_pants_l65_65251

theorem price_of_pants (P : ℝ) (h1 : 4 * 33 = 132) (h2 : 2 * P + 132 = 240) : P = 54 :=
sorry

end price_of_pants_l65_65251


namespace TamekaBoxesRelation_l65_65540

theorem TamekaBoxesRelation 
  (S : ℤ)
  (h1 : 40 + S + S / 2 = 145) :
  S - 40 = 30 :=
by
  sorry

end TamekaBoxesRelation_l65_65540


namespace train_stops_time_l65_65287

theorem train_stops_time 
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 20 := 
by
  sorry

end train_stops_time_l65_65287


namespace smallest_x_for_cubic_l65_65731

theorem smallest_x_for_cubic (x N : ℕ) (h1 : 1260 * x = N^3) : x = 7350 :=
sorry

end smallest_x_for_cubic_l65_65731


namespace division_problem_l65_65465

theorem division_problem : 250 / (15 + 13 * 3 - 4) = 5 := by
  sorry

end division_problem_l65_65465


namespace marika_father_age_twice_l65_65675

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l65_65675


namespace Danny_more_wrappers_than_bottle_caps_l65_65058

theorem Danny_more_wrappers_than_bottle_caps
  (initial_wrappers : ℕ)
  (initial_bottle_caps : ℕ)
  (found_wrappers : ℕ)
  (found_bottle_caps : ℕ) :
  initial_wrappers = 67 →
  initial_bottle_caps = 35 →
  found_wrappers = 18 →
  found_bottle_caps = 15 →
  (initial_wrappers + found_wrappers) - (initial_bottle_caps + found_bottle_caps) = 35 :=
by
  intros h1 h2 h3 h4
  sorry

end Danny_more_wrappers_than_bottle_caps_l65_65058


namespace solve_diamond_eq_l65_65922

noncomputable def diamond_op (a b : ℝ) := a / b

theorem solve_diamond_eq (x : ℝ) (h : x ≠ 0) : diamond_op 2023 (diamond_op 7 x) = 150 ↔ x = 1050 / 2023 := by
  sorry

end solve_diamond_eq_l65_65922


namespace max_y_value_l65_65000

theorem max_y_value (x y : Int) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
by sorry

end max_y_value_l65_65000


namespace james_earnings_per_subscriber_is_9_l65_65379

/-
Problem:
James streams on Twitch. He had 150 subscribers and then someone gifted 50 subscribers. If he gets a certain amount per month per subscriber and now makes $1800 a month, how much does he make per subscriber?
-/

def initial_subscribers : ℕ := 150
def gifted_subscribers : ℕ := 50
def total_subscribers := initial_subscribers + gifted_subscribers
def total_earnings : ℤ := 1800

def earnings_per_subscriber := total_earnings / total_subscribers

/-
Theorem: James makes $9 per month for each subscriber.
-/
theorem james_earnings_per_subscriber_is_9 : earnings_per_subscriber = 9 := by
  -- to be filled in with proof steps
  sorry

end james_earnings_per_subscriber_is_9_l65_65379


namespace halfway_fraction_l65_65995

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65995


namespace johns_speed_final_push_l65_65837

-- Definitions for the given conditions
def john_behind_steve : ℝ := 14
def steve_speed : ℝ := 3.7
def john_ahead_steve : ℝ := 2
def john_final_push_time : ℝ := 32

-- Proving the statement
theorem johns_speed_final_push : 
  (∃ (v : ℝ), v * john_final_push_time = steve_speed * john_final_push_time + john_behind_steve + john_ahead_steve) -> 
  ∃ (v : ℝ), v = 4.2 :=
by
  sorry

end johns_speed_final_push_l65_65837


namespace log_10_7_eqn_l65_65231

variables (p q : ℝ)
noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_10_7_eqn (h1 : log_base 4 5 = p) (h2 : log_base 5 7 = q) : 
  log_base 10 7 = (2 * p * q) / (2 * p + 1) :=
by 
  sorry

end log_10_7_eqn_l65_65231


namespace k_values_l65_65937

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def find_k (k : ℝ) : Prop :=
  (vector_dot (2, 3) (1, k) = 0) ∨
  (vector_dot (2, 3) (-1, k - 3) = 0) ∨
  (vector_dot (1, k) (-1, k - 3) = 0)

theorem k_values :
  ∃ k : ℝ, find_k k ∧ 
  (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13 ) / 2) :=
by
  sorry

end k_values_l65_65937


namespace sum_of_squares_of_roots_l65_65210

theorem sum_of_squares_of_roots :
  ∀ r1 r2 : ℝ, (r1 + r2 = 14) ∧ (r1 * r2 = 8) → (r1^2 + r2^2 = 180) := by
  sorry

end sum_of_squares_of_roots_l65_65210


namespace constantin_mother_deposit_return_l65_65248

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l65_65248


namespace soccer_balls_percentage_holes_l65_65258

variable (x : ℕ)

theorem soccer_balls_percentage_holes 
    (h1 : ∃ x, 0 ≤ x ∧ x ≤ 100)
    (h2 : 48 = 80 * (100 - x) / 100) : 
  x = 40 := sorry

end soccer_balls_percentage_holes_l65_65258


namespace total_cost_of_two_books_l65_65814

theorem total_cost_of_two_books (C1 C2 total_cost: ℝ) :
  C1 = 262.5 →
  0.85 * C1 = 1.19 * C2 →
  total_cost = C1 + C2 →
  total_cost = 450 :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_two_books_l65_65814


namespace inequality_proof_l65_65366

theorem inequality_proof (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) (h5 : a * d = b * c) :
  (a - d) ^ 2 ≥ 4 * d + 8 := 
sorry

end inequality_proof_l65_65366


namespace smallest_number_divisible_by_set_l65_65617

theorem smallest_number_divisible_by_set : ∃ x : ℕ, (∀ d ∈ [12, 24, 36, 48, 56, 72, 84], (x - 24) % d = 0) ∧ x = 1032 := 
by {
  sorry
}

end smallest_number_divisible_by_set_l65_65617


namespace opposite_of_neg3_l65_65719

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l65_65719


namespace sqrt_of_square_eq_seven_l65_65933

theorem sqrt_of_square_eq_seven (x : ℝ) (h : x^2 = 7) : x = Real.sqrt 7 ∨ x = -Real.sqrt 7 :=
sorry

end sqrt_of_square_eq_seven_l65_65933


namespace halfway_fraction_l65_65999

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65999


namespace starting_lineups_possible_l65_65909

open Nat

theorem starting_lineups_possible (total_players : ℕ) (all_stars : ℕ) (lineup_size : ℕ) 
  (fixed_in_lineup : ℕ) (choose_size : ℕ) 
  (h_fixed : fixed_in_lineup = all_stars)
  (h_remaining : total_players - fixed_in_lineup = choose_size)
  (h_lineup : lineup_size = all_stars + choose_size) :
  (Nat.choose choose_size 3 = 220) :=
by
  sorry

end starting_lineups_possible_l65_65909


namespace roy_total_pens_l65_65696

def number_of_pens (blue black red green purple : ℕ) : ℕ :=
  blue + black + red + green + purple

theorem roy_total_pens (blue black red green purple : ℕ)
  (h1 : blue = 8)
  (h2 : black = 4 * blue)
  (h3 : red = blue + black - 5)
  (h4 : green = red / 2)
  (h5 : purple = blue + green - 3) :
  number_of_pens blue black red green purple = 114 := by
  sorry

end roy_total_pens_l65_65696


namespace min_value_of_expression_l65_65385

noncomputable def min_expression := 4 * (Real.rpow 5 (1/4) - 1)^2

theorem min_value_of_expression (a b c : ℝ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = min_expression :=
sorry

end min_value_of_expression_l65_65385


namespace single_point_graph_d_l65_65414

theorem single_point_graph_d (d : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + d = 0 ↔ x = -1 ∧ y = 6) → d = 39 :=
by 
  sorry

end single_point_graph_d_l65_65414


namespace infinite_nested_sqrt_l65_65064

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l65_65064


namespace oranges_for_juice_l65_65415

-- Define conditions
def total_oranges : ℝ := 7 -- in million tons
def export_percentage : ℝ := 0.25
def juice_percentage : ℝ := 0.60

-- Define the mathematical problem
theorem oranges_for_juice : 
  (total_oranges * (1 - export_percentage) * juice_percentage) = 3.2 :=
by
  sorry

end oranges_for_juice_l65_65415


namespace div_by_5_factor_l65_65354

theorem div_by_5_factor {x y z : ℤ} (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * 5 * (y - z) * (z - x) * (x - y) :=
sorry

end div_by_5_factor_l65_65354


namespace total_chrome_parts_l65_65612

theorem total_chrome_parts (a b : ℕ) 
  (h1 : a + b = 21) 
  (h2 : 3 * a + 2 * b = 50) : 2 * a + 4 * b = 68 := 
sorry

end total_chrome_parts_l65_65612


namespace ordered_pair_exists_l65_65868

theorem ordered_pair_exists :
  ∃ p q : ℝ, 
  (3 + 8 * p = 2 - 3 * q) ∧ (-4 - 6 * p = -3 + 4 * q) ∧ (p = -1/14) ∧ (q = -1/7) :=
by
  sorry

end ordered_pair_exists_l65_65868


namespace bill_experience_l65_65598

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l65_65598


namespace inequality_correctness_l65_65219

variable (a b : ℝ)
variable (h1 : a < b) (h2 : b < 0)

theorem inequality_correctness : a^2 > ab ∧ ab > b^2 := by
  sorry

end inequality_correctness_l65_65219


namespace min_dot_product_on_hyperbola_l65_65954

theorem min_dot_product_on_hyperbola (x1 y1 x2 y2 : ℝ) 
  (hA : x1^2 - y1^2 = 2) 
  (hB : x2^2 - y2^2 = 2)
  (h_x1 : x1 > 0) 
  (h_x2 : x2 > 0) : 
  x1 * x2 + y1 * y2 ≥ 2 :=
sorry

end min_dot_product_on_hyperbola_l65_65954


namespace probability_of_selecting_one_is_correct_l65_65325

-- Define the number of elements in the first 20 rows of Pascal's triangle
def totalElementsInPascalFirst20Rows : ℕ := 210

-- Define the number of ones in the first 20 rows of Pascal's triangle
def totalOnesInPascalFirst20Rows : ℕ := 39

-- The probability as a rational number
def probabilityOfSelectingOne : ℚ := totalOnesInPascalFirst20Rows / totalElementsInPascalFirst20Rows

theorem probability_of_selecting_one_is_correct :
  probabilityOfSelectingOne = 13 / 70 :=
by
  -- Proof is omitted
  sorry

end probability_of_selecting_one_is_correct_l65_65325


namespace coeff_x3_in_expansion_l65_65972

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion :
  (2 : ℚ)^(4 - 2) * binomial_coeff 4 2 = 24 := by 
  sorry

end coeff_x3_in_expansion_l65_65972


namespace length_PX_l65_65833

theorem length_PX (CX DP PW PX : ℕ) (hCX : CX = 60) (hDP : DP = 20) (hPW : PW = 40)
  (parallel_CD_WX : true)  -- We use a boolean to denote the parallel condition for simplicity
  (h1 : DP + PW = CX)  -- The sum of the segments from point C through P to point X
  (h2 : DP * 2 = PX)  -- The ratio condition derived from the similarity of triangles
  : PX = 40 := 
by
  -- using the given conditions and h2 to solve for PX
  sorry

end length_PX_l65_65833


namespace total_asphalt_used_1520_tons_l65_65893

noncomputable def asphalt_used (L W : ℕ) (asphalt_per_100m2 : ℕ) : ℕ :=
  (L * W / 100) * asphalt_per_100m2

theorem total_asphalt_used_1520_tons :
  asphalt_used 800 50 3800 = 1520000 := by
  sorry

end total_asphalt_used_1520_tons_l65_65893


namespace maximize_profit_l65_65754

noncomputable def profit (x a : ℝ) : ℝ :=
  19 - 24 / (x + 2) - (3 / 2) * x

theorem maximize_profit (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ 
  (if a ≥ 2 then x = 2 else x = a) :=
by
  sorry

end maximize_profit_l65_65754


namespace sum_of_7_and_2_terms_l65_65349

open Nat

variable {α : Type*} [Field α]

-- Definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d
  
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∀ m n k : ℕ, m < n → n < k → a n * a n = a m * a k
  
def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a n)) / 2

-- Given Conditions
variable (a : ℕ → α) 
variable (d : α)

-- Checked. Arithmetic sequence with non-zero common difference
axiom h1 : is_arithmetic_sequence a d

-- Known values provided in the problem statement
axiom h2 : a 1 = 6

-- Terms forming a geometric sequence
axiom h3 : is_geometric_sequence a

-- The goal is to find the sum of the first 7 terms and the first 2 terms
theorem sum_of_7_and_2_terms : sum_first_n_terms a 7 + sum_first_n_terms a 2 = 80 := 
by {
  -- Proof will be here
  sorry
}

end sum_of_7_and_2_terms_l65_65349


namespace percentage_markup_l65_65978

theorem percentage_markup (SP CP : ℕ) (h1 : SP = 8340) (h2 : CP = 6672) :
  ((SP - CP) / CP * 100) = 25 :=
by
  -- Before proving, we state our assumptions
  sorry

end percentage_markup_l65_65978


namespace fraction_unseated_l65_65019

theorem fraction_unseated :
  ∀ (tables seats_per_table seats_taken : ℕ),
  tables = 15 →
  seats_per_table = 10 →
  seats_taken = 135 →
  ((tables * seats_per_table - seats_taken : ℕ) / (tables * seats_per_table : ℕ) : ℚ) = 1 / 10 :=
by
  intros tables seats_per_table seats_taken h_tables h_seats_per_table h_seats_taken
  sorry

end fraction_unseated_l65_65019


namespace heather_payment_per_weed_l65_65813

noncomputable def seconds_in_hour : ℕ := 60 * 60

noncomputable def weeds_per_hour (seconds_per_weed : ℕ) : ℕ :=
  seconds_in_hour / seconds_per_weed

noncomputable def payment_per_weed (hourly_pay : ℕ) (weeds_per_hour : ℕ) : ℚ :=
  hourly_pay / weeds_per_hour

theorem heather_payment_per_weed (seconds_per_weed : ℕ) (hourly_pay : ℕ) :
  seconds_per_weed = 18 ∧ hourly_pay = 10 → payment_per_weed hourly_pay (weeds_per_hour seconds_per_weed) = 0.05 :=
by
  sorry

end heather_payment_per_weed_l65_65813


namespace albert_earnings_l65_65586

theorem albert_earnings (E P : ℝ) 
  (h1 : E * 1.20 = 660) 
  (h2 : E * (1 + P) = 693) : 
  P = 0.26 :=
sorry

end albert_earnings_l65_65586


namespace fan_rotation_is_not_translation_l65_65878

def phenomenon := Type

def is_translation (p : phenomenon) : Prop := sorry

axiom elevator_translation : phenomenon
axiom drawer_translation : phenomenon
axiom fan_rotation : phenomenon
axiom car_translation : phenomenon

axiom elevator_is_translation : is_translation elevator_translation
axiom drawer_is_translation : is_translation drawer_translation
axiom car_is_translation : is_translation car_translation

theorem fan_rotation_is_not_translation : ¬ is_translation fan_rotation := sorry

end fan_rotation_is_not_translation_l65_65878


namespace solve_inequality_l65_65141

theorem solve_inequality : 
  {x : ℝ | -3 * x^2 + 9 * x + 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
by {
  sorry
}

end solve_inequality_l65_65141


namespace domain_fraction_function_l65_65631

theorem domain_fraction_function (f : ℝ → ℝ):
  (∀ x : ℝ, -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 0) →
  (∀ x : ℝ, x ≠ 0 → -2 ≤ x ∧ x < 0) →
  (∀ x, (2^x - 1) ≠ 0) →
  true := sorry

end domain_fraction_function_l65_65631


namespace nickels_count_l65_65882

theorem nickels_count (N Q : ℕ) 
  (h_eq : N = Q) 
  (h_total_value : 5 * N + 25 * Q = 1200) :
  N = 40 := 
by 
  sorry

end nickels_count_l65_65882


namespace quadratic_has_real_roots_find_pos_m_l65_65362

-- Proof problem 1:
theorem quadratic_has_real_roots (m : ℝ) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x^2 - 4 * m * x + 3 * m^2 = 0 :=
by
  sorry

-- Proof problem 2:
theorem find_pos_m (m x1 x2 : ℝ) (hm : x1 > x2) (h_diff : x1 - x2 = 2)
  (h_roots : ∀ m, (x^2 - 4*m*x + 3*m^2 = 0)) : m = 1 :=
by
  sorry

end quadratic_has_real_roots_find_pos_m_l65_65362


namespace sufficient_condition_implies_true_l65_65644

variable {p q : Prop}

theorem sufficient_condition_implies_true (h : p → q) : (p → q) = true :=
by
  sorry

end sufficient_condition_implies_true_l65_65644


namespace domain_of_f_l65_65207

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | x^2 - 4 >= 0 ∧ x^2 - 4 ≠ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end domain_of_f_l65_65207


namespace plant_lamp_arrangement_count_l65_65399

theorem plant_lamp_arrangement_count :
  let basil_plants := 2
  let aloe_plants := 2
  let white_lamps := 3
  let red_lamps := 3
  (∀ plant, plant = basil_plants ∨ plant = aloe_plants)
  ∧ (∀ lamp, lamp = white_lamps ∨ lamp = red_lamps)
  → (∀ plant, ∃ lamp, plant → lamp)
  → ∃ count, count = 50 := 
by
  sorry

end plant_lamp_arrangement_count_l65_65399


namespace rectangles_in_cube_l65_65931

/-- Number of rectangles that can be formed by the vertices of a cube is 12. -/
theorem rectangles_in_cube : 
  ∃ (n : ℕ), (n = 12) := by
  -- The cube has vertices, and squares are a subset of rectangles.
  -- We need to count rectangles including squares among vertices of the cube.
  sorry

end rectangles_in_cube_l65_65931


namespace sandwich_is_not_condiments_l65_65896

theorem sandwich_is_not_condiments (sandwich_weight condiments_weight : ℕ)
  (h1 : sandwich_weight = 150)
  (h2 : condiments_weight = 45) :
  (sandwich_weight - condiments_weight) / sandwich_weight * 100 = 70 := 
by sorry

end sandwich_is_not_condiments_l65_65896


namespace problem1_problem2_l65_65797

def box (n : ℕ) : ℕ := (10^n - 1) / 9

theorem problem1 (m : ℕ) :
  let b := box (3^m)
  b % (3^m) = 0 ∧ b % (3^(m+1)) ≠ 0 :=
  sorry

theorem problem2 (n : ℕ) :
  (n % 27 = 0) ↔ (box n % 27 = 0) :=
  sorry

end problem1_problem2_l65_65797


namespace ones_digit_of_largest_power_of_3_dividing_27_factorial_is_3_l65_65342

theorem ones_digit_of_largest_power_of_3_dividing_27_factorial_is_3 :
  let n : ℕ := 27,
      k := ∏ i in finset.range (n + 1), i + 1,   -- 27!
      largest_power_of_3 := 9 + 3 + 1,          -- highest power of 3 dividing 27!
      power := 3 ^ largest_power_of_3,
      ones_digit := power % 10
  in ones_digit = 3 :=
by
  let n : ℕ := 27,
      k := ∏ i in finset.range (n + 1), i + 1,   -- definition of 27!
      largest_power_of_3 := 9 + 3 + 1,          -- calculation of largest power of 3 dividing 27!
      power := 3 ^ largest_power_of_3,          -- calculation of 3 to the power of largest_power_of_3
      ones_digit := power % 10                  
  in
  -- prove that ones_digit is 3 (skipped here, replace with proper proof)
  sorry

end ones_digit_of_largest_power_of_3_dividing_27_factorial_is_3_l65_65342


namespace calculate_expression_l65_65605

theorem calculate_expression :
  36 + (150 / 15) + (12 ^ 2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end calculate_expression_l65_65605


namespace beth_speed_l65_65564

noncomputable def beth_average_speed (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ) : ℚ :=
  let jerry_time_hours := jerry_time_minutes / 60
  let jerry_distance := jerry_speed * jerry_time_hours
  let beth_distance := jerry_distance + beth_extra_miles
  let beth_time_hours := (jerry_time_minutes + beth_extra_time_minutes) / 60
  beth_distance / beth_time_hours

theorem beth_speed {beth_avg_speed : ℚ}
  (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ)
  (h_jerry_speed : jerry_speed = 40)
  (h_jerry_time : jerry_time_minutes = 30)
  (h_beth_extra_miles : beth_extra_miles = 5)
  (h_beth_extra_time : beth_extra_time_minutes = 20) :
  beth_average_speed jerry_speed jerry_time_minutes beth_extra_miles beth_extra_time_minutes = 30 := 
by 
  -- Leaving out the proof steps
  sorry

end beth_speed_l65_65564


namespace sum_of_real_roots_eq_five_pi_l65_65618

theorem sum_of_real_roots_eq_five_pi :
  ∀ x : ℝ, 0 < x ∧ x < 2 * Real.pi → 3 * (Real.tan x)^2 + 8 * Real.tan x + 3 = 0 → x = 5 * Real.pi :=
begin
  sorry
end

end sum_of_real_roots_eq_five_pi_l65_65618


namespace marikas_father_twice_her_age_l65_65680

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l65_65680


namespace product_of_roots_l65_65932

noncomputable def quadratic_equation (x : ℝ) : Prop :=
  (x + 4) * (x - 5) = 22

theorem product_of_roots :
  ∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → (x1 * x2 = -42) := 
by
  sorry

end product_of_roots_l65_65932


namespace principal_amount_simple_interest_l65_65503

theorem principal_amount_simple_interest 
    (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
    (hR : R = 4)
    (hT : T = 5)
    (hSI : SI = P - 2080)
    (hInterestFormula : SI = (P * R * T) / 100) :
    P = 2600 := 
by
  sorry

end principal_amount_simple_interest_l65_65503


namespace remaining_pictures_l65_65739

theorem remaining_pictures (k m : ℕ) (d1 := 9 * k + 4) (d2 := 9 * m + 6) :
  (d1 * d2) % 9 = 6 → 9 - (d1 * d2 % 9) = 3 :=
by
  intro h
  sorry

end remaining_pictures_l65_65739


namespace xyz_not_divisible_by_3_l65_65125

theorem xyz_not_divisible_by_3 (x y z : ℕ) (h1 : x % 2 = 1) (h2 : y % 2 = 1) (h3 : z % 2 = 1) 
  (h4 : Nat.gcd (Nat.gcd x y) z = 1) (h5 : (x^2 + y^2 + z^2) % (x + y + z) = 0) : 
  (x + y + z - 2) % 3 ≠ 0 :=
by
  sorry

end xyz_not_divisible_by_3_l65_65125


namespace sum_of_first_five_terms_l65_65730

theorem sum_of_first_five_terms 
  (a₂ a₃ a₄ : ℤ)
  (h1 : a₂ = 4)
  (h2 : a₃ = 7)
  (h3 : a₄ = 10) :
  ∃ a1 a5, a1 + a₂ + a₃ + a₄ + a5 = 35 :=
by
  sorry

end sum_of_first_five_terms_l65_65730


namespace jasmine_percentage_l65_65454

namespace ProofExample

variables (original_volume : ℝ) (initial_percent_jasmine : ℝ) (added_jasmine : ℝ) (added_water : ℝ)
variables (initial_jasmine : ℝ := initial_percent_jasmine * original_volume / 100)
variables (total_jasmine : ℝ := initial_jasmine + added_jasmine)
variables (total_volume : ℝ := original_volume + added_jasmine + added_water)
variables (final_percent_jasmine : ℝ := (total_jasmine / total_volume) * 100)

theorem jasmine_percentage 
  (h1 : original_volume = 80)
  (h2 : initial_percent_jasmine = 10)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 12)
  : final_percent_jasmine = 16 := 
sorry

end ProofExample

end jasmine_percentage_l65_65454


namespace recreation_percentage_l65_65520

variable (W : ℝ) 

def recreation_last_week (W : ℝ) : ℝ := 0.10 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def recreation_this_week (W : ℝ) : ℝ := 0.40 * (wages_this_week W)

theorem recreation_percentage : 
  (recreation_this_week W) / (recreation_last_week W) * 100 = 360 :=
by sorry

end recreation_percentage_l65_65520


namespace max_marks_l65_65959

variable (M : ℝ)

theorem max_marks (h1 : 0.35 * M = 175) : M = 500 := by
  -- Proof goes here
  sorry

end max_marks_l65_65959


namespace coprime_odd_sum_of_floors_l65_65255

theorem coprime_odd_sum_of_floors (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h_coprime : Nat.gcd p q = 1) : 
  (List.sum (List.map (λ i => Nat.floor ((i • q : ℚ) / p)) ((List.range (p / 2 + 1)).tail)) +
   List.sum (List.map (λ i => Nat.floor ((i • p : ℚ) / q)) ((List.range (q / 2 + 1)).tail))) =
  (p - 1) * (q - 1) / 4 :=
by
  sorry

end coprime_odd_sum_of_floors_l65_65255


namespace hindi_speaking_children_l65_65116

-- Condition Definitions
def total_children : ℕ := 90
def percent_only_english : ℝ := 0.25
def percent_only_hindi : ℝ := 0.15
def percent_only_spanish : ℝ := 0.10
def percent_english_hindi : ℝ := 0.20
def percent_english_spanish : ℝ := 0.15
def percent_hindi_spanish : ℝ := 0.10
def percent_all_three : ℝ := 0.05

-- Question translated to a Lean statement
theorem hindi_speaking_children :
  (percent_only_hindi + percent_english_hindi + percent_hindi_spanish + percent_all_three) * total_children = 45 :=
by
  sorry

end hindi_speaking_children_l65_65116


namespace tan_y_l65_65115

theorem tan_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hy : 0 < y ∧ y < π / 2)
  (hsiny : Real.sin y = 2 * a * b / (a^2 + b^2)) :
  Real.tan y = 2 * a * b / (a^2 - b^2) :=
sorry

end tan_y_l65_65115


namespace billy_reads_60_pages_per_hour_l65_65602

theorem billy_reads_60_pages_per_hour
  (free_time_per_day : ℕ)
  (days : ℕ)
  (video_games_time_percentage : ℝ)
  (books : ℕ)
  (pages_per_book : ℕ)
  (remaining_time_percentage : ℝ)
  (total_free_time := free_time_per_day * days)
  (time_playing_video_games := video_games_time_percentage * total_free_time)
  (time_reading := remaining_time_percentage * total_free_time)
  (total_pages := books * pages_per_book)
  (pages_per_hour := total_pages / time_reading) :
  free_time_per_day = 8 →
  days = 2 →
  video_games_time_percentage = 0.75 →
  remaining_time_percentage = 0.25 →
  books = 3 →
  pages_per_book = 80 →
  pages_per_hour = 60 :=
by
  intros
  sorry

end billy_reads_60_pages_per_hour_l65_65602


namespace isosceles_triangle_perimeter_l65_65626

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : a = 3 ∨ a = 6)
  (h3 : b = 3 ∨ b = 6) (h4 : c = 3 ∨ c = 6) (h5 : a + b + c = 15) : a + b + c = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l65_65626


namespace dot_product_a_b_l65_65227

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (4, -3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Statement of the theorem to prove
theorem dot_product_a_b : dot_product vector_a vector_b = -1 := 
by sorry

end dot_product_a_b_l65_65227


namespace years_until_5_years_before_anniversary_l65_65107

-- Definitions
def years_built_ago := 100
def upcoming_anniversary := 200
def years_before_anniversary := 5

-- Theorem statement
theorem years_until_5_years_before_anniversary :
  let years_until_anniversary := upcoming_anniversary - years_built_ago in
  let future_years := years_until_anniversary - years_before_anniversary in
  future_years = 95 :=
by
  sorry

end years_until_5_years_before_anniversary_l65_65107


namespace exists_star_of_david_arrangement_l65_65132

noncomputable def star_of_david_arrangement : Prop :=
  ∃ (positions : Fin 7 → Fin 7 × Fin 5) (rows : Fin 5 → Set (Fin 7)) (in_row : Fin 7 ↔ Set (Fin 7)),

  -- positions map each bush (indexed by Fin 7) to a coordinate (x, y in Fin 7 × Fin 5)
  (∀ i : Fin 7, positions i ∈ (Fin 7) × (Fin 5)) ∧
  
  -- rows is a function that maps each row index (Fin 5) to a Set of bushes (indexed by Fin 7)
  (∀ j : Fin 5, rows j ⊆ set.univ ∧ set.card (rows j) = 3) ∧
  
  -- each position appears in exactly three rows
  (∀ i : Fin 7, set.card (in_row i) = 3) ∧
  
  -- each row has exactly three bushes
  (∀ j : Fin 5, set.card (rows j) = 3) ∧

  -- in_row function aligns with rows
  (∀ i : Fin 7, ∀ j : (Fin 5), i ∈ rows j ↔ in_row i = rows j)

-- Here, we state the existence of such an arrangement:
theorem exists_star_of_david_arrangement : star_of_david_arrangement :=
sorry

end exists_star_of_david_arrangement_l65_65132


namespace problem_correctness_l65_65926

variable (f : ℝ → ℝ)
variable (h₀ : ∀ x, f x > 0)
variable (h₁ : ∀ a b, f a * f b = f (a + b))

theorem problem_correctness :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1 / 3)) :=
by 
  -- Using the hypotheses provided
  sorry

end problem_correctness_l65_65926


namespace change_received_after_discounts_and_taxes_l65_65728

theorem change_received_after_discounts_and_taxes :
  let price_wooden_toy : ℝ := 20
  let price_hat : ℝ := 10
  let tax_rate : ℝ := 0.08
  let discount_wooden_toys : ℝ := 0.15
  let discount_hats : ℝ := 0.10
  let quantity_wooden_toys : ℝ := 3
  let quantity_hats : ℝ := 4
  let amount_paid : ℝ := 200
  let cost_wooden_toys := quantity_wooden_toys * price_wooden_toy
  let discounted_cost_wooden_toys := cost_wooden_toys - (discount_wooden_toys * cost_wooden_toys)
  let cost_hats := quantity_hats * price_hat
  let discounted_cost_hats := cost_hats - (discount_hats * cost_hats)
  let total_cost_before_tax := discounted_cost_wooden_toys + discounted_cost_hats
  let tax := tax_rate * total_cost_before_tax
  let total_cost_after_tax := total_cost_before_tax + tax
  let change_received := amount_paid - total_cost_after_tax
  change_received = 106.04 := by
  -- All the conditions and intermediary steps are defined above, from problem to solution.
  sorry

end change_received_after_discounts_and_taxes_l65_65728


namespace number_of_gigs_played_l65_65571

-- Definitions based on given conditions
def earnings_per_member : ℕ := 20
def number_of_members : ℕ := 4
def total_earnings : ℕ := 400

-- Proof statement in Lean 4
theorem number_of_gigs_played : (total_earnings / (earnings_per_member * number_of_members)) = 5 :=
by
  sorry

end number_of_gigs_played_l65_65571


namespace marikas_father_age_twice_in_2036_l65_65688

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l65_65688


namespace marikas_father_twice_her_age_l65_65682

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l65_65682


namespace functions_increase_faster_l65_65562

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- Restate the problem in Lean
theorem functions_increase_faster :
  (∀ (x : ℝ), deriv y₁ x = 100) ∧
  (∀ (x : ℝ), deriv y₂ x = 100) ∧
  (∀ (x : ℝ), deriv y₃ x = 99) ∧
  (100 > 99) :=
by
  sorry

end functions_increase_faster_l65_65562


namespace func_has_extrema_l65_65492

theorem func_has_extrema (a b c : ℝ) (h_a_nonzero : a ≠ 0) (h_discriminant_positive : b^2 + 8 * a * c > 0) 
    (h_pos_sum_roots : b / a > 0) (h_pos_product_roots : -2 * c / a > 0) : 
    (a * b > 0) ∧ (a * c < 0) :=
by 
  -- Proof skipped.
  sorry

end func_has_extrema_l65_65492


namespace probability_within_two_units_l65_65298

-- Conditions
def is_in_square (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -3 ∧ Q.1 ≤ 3 ∧ Q.2 ≥ -3 ∧ Q.2 ≤ 3

def is_within_two_units (Q : ℝ × ℝ) : Prop :=
  Q.1 * Q.1 + Q.2 * Q.2 ≤ 4

-- Problem Statement
theorem probability_within_two_units :
  (measure_theory.measure_of {Q : ℝ × ℝ | is_within_two_units Q} / measure_theory.measure_of {Q : ℝ × ℝ | is_in_square Q} = π / 9) := by
  sorry

end probability_within_two_units_l65_65298


namespace meaningful_expr_x_value_l65_65234

theorem meaningful_expr_x_value (x : ℝ) :
  (10 - x >= 0) ∧ (x ≠ 4) ↔ (x = 8) :=
begin
  -- proof omitted
  sorry
end

end meaningful_expr_x_value_l65_65234


namespace roots_transformation_l65_65525

-- Given polynomial
def poly1 (x : ℝ) : ℝ := x^3 - 3*x^2 + 8

-- Polynomial with roots 3*r1, 3*r2, 3*r3
def poly2 (x : ℝ) : ℝ := x^3 - 9*x^2 + 216

-- Theorem stating the equivalence
theorem roots_transformation (r1 r2 r3 : ℝ) 
  (h : ∀ x, poly1 x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x, poly2 x = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
sorry

end roots_transformation_l65_65525


namespace length_of_major_axis_l65_65638

theorem length_of_major_axis (x y : ℝ) (h : (x^2 / 25) + (y^2 / 16) = 1) : 10 = 10 :=
by
  sorry

end length_of_major_axis_l65_65638


namespace fixed_point_of_transformed_logarithmic_function_l65_65674

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_a (a : ℝ) (x : ℝ) : ℝ := 1 + log_a a (x - 1)

theorem fixed_point_of_transformed_logarithmic_function
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1) : f_a a 2 = 1 :=
by
  -- Prove the theorem using given conditions
  sorry

end fixed_point_of_transformed_logarithmic_function_l65_65674


namespace event_eq_conds_l65_65020

-- Definitions based on conditions
def Die := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }
def sum_points (d1 d2 : Die) : ℕ := d1.val + d2.val

def event_xi_eq_4 (d1 d2 : Die) : Prop := 
  sum_points d1 d2 = 4

def condition_a (d1 d2 : Die) : Prop := 
  d1.val = 2 ∧ d2.val = 2

def condition_b (d1 d2 : Die) : Prop := 
  (d1.val = 3 ∧ d2.val = 1) ∨ (d1.val = 1 ∧ d2.val = 3)

def event_condition (d1 d2 : Die) : Prop :=
  condition_a d1 d2 ∨ condition_b d1 d2

-- The main Lean statement
theorem event_eq_conds (d1 d2 : Die) : 
  event_xi_eq_4 d1 d2 ↔ event_condition d1 d2 := 
by
  sorry

end event_eq_conds_l65_65020


namespace af_cd_ratio_l65_65485

theorem af_cd_ratio (a b c d e f : ℝ) 
  (h1 : a * b * c = 130) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 750) 
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 2 / 3 := 
by
  sorry

end af_cd_ratio_l65_65485


namespace triangle_perimeter_l65_65149

/-- The lengths of two sides of a triangle are 3 and 5 respectively. The third side is a root of the equation x^2 - 7x + 12 = 0. Find the perimeter of the triangle. -/
theorem triangle_perimeter :
  let side1 := 3
  let side2 := 5
  let third_side1 := 3
  let third_side2 := 4
  (third_side1 * third_side1 - 7 * third_side1 + 12 = 0) ∧
  (third_side2 * third_side2 - 7 * third_side2 + 12 = 0) →
  (side1 + side2 + third_side1 = 11 ∨ side1 + side2 + third_side2 = 12) :=
by
  sorry

end triangle_perimeter_l65_65149


namespace compound_interest_amount_l65_65547

theorem compound_interest_amount 
  (P_si : ℝ := 3225) 
  (R_si : ℝ := 8) 
  (T_si : ℝ := 5) 
  (R_ci : ℝ := 15) 
  (T_ci : ℝ := 2) 
  (SI : ℝ := P_si * R_si * T_si / 100) 
  (CI : ℝ := 2 * SI) 
  (CI_formula : ℝ := P_ci * ((1 + R_ci / 100)^T_ci - 1))
  (P_ci := 516 / 0.3225) :
  P_ci = 1600 := 
by
  sorry

end compound_interest_amount_l65_65547


namespace photo_gallery_total_l65_65668

theorem photo_gallery_total (initial_photos: ℕ) (first_day_photos: ℕ) (second_day_photos: ℕ)
  (h_initial: initial_photos = 400) 
  (h_first_day: first_day_photos = initial_photos / 2)
  (h_second_day: second_day_photos = first_day_photos + 120) : 
  initial_photos + first_day_photos + second_day_photos = 920 :=
by
  sorry

end photo_gallery_total_l65_65668


namespace solve_equation_l65_65609

theorem solve_equation (x : ℝ) (h_eq : 1 / (x - 2) = 3 / (x - 5)) : 
  x = 1 / 2 :=
  sorry

end solve_equation_l65_65609


namespace x_minus_y_eq_14_l65_65695

theorem x_minus_y_eq_14 (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 100) : x - y = 14 :=
sorry

end x_minus_y_eq_14_l65_65695


namespace probability_of_point_within_two_units_l65_65302

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let area_of_circle := 4 * Real.pi
  let area_of_square := 36
  area_of_circle / area_of_square

theorem probability_of_point_within_two_units :
  probability_within_two_units_of_origin = Real.pi / 9 := 
by
  -- The proof steps are omitted as per the requirements
  sorry

end probability_of_point_within_two_units_l65_65302


namespace calc_a8_l65_65475

variable {a : ℕ+ → ℕ}

-- Conditions
axiom recur_relation : ∀ (p q : ℕ+), a (p + q) = a p * a q
axiom initial_condition : a 2 = 2

-- Proof statement
theorem calc_a8 : a 8 = 16 := by
  sorry

end calc_a8_l65_65475


namespace train_cross_time_approx_l65_65452
noncomputable def time_to_cross_bridge
  (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  ((train_length + bridge_length) / (speed_kmh * 1000 / 3600))

theorem train_cross_time_approx (train_length bridge_length speed_kmh : ℝ)
  (h_train_length : train_length = 250)
  (h_bridge_length : bridge_length = 300)
  (h_speed_kmh : speed_kmh = 44) :
  abs (time_to_cross_bridge train_length bridge_length speed_kmh - 45) < 1 :=
by
  sorry

end train_cross_time_approx_l65_65452


namespace car_travel_distance_l65_65573

theorem car_travel_distance (distance : ℝ) 
  (speed1 : ℝ := 80) 
  (speed2 : ℝ := 76.59574468085106) 
  (time_difference : ℝ := 2 / 3600) : 
  (distance / speed2 = distance / speed1 + time_difference) → 
  distance = 0.998177 :=
by
  -- assuming the above equation holds, we need to conclude the distance
  sorry

end car_travel_distance_l65_65573


namespace halfway_fraction_l65_65993

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65993


namespace unique_digit_solution_l65_65433

-- Define the constraints as Lean predicates.
def sum_top_less_7 (A B C D E : ℕ) := A + B = (C + D + E) / 7
def sum_left_less_5 (A B C D E : ℕ) := A + C = (B + D + E) / 5

-- The main theorem statement asserting there is a unique solution.
theorem unique_digit_solution :
  ∃! (A B C D E : ℕ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 
  sum_top_less_7 A B C D E ∧ sum_left_less_5 A B C D E ∧
  (A, B, C, D, E) = (1, 2, 3, 4, 6) := sorry

end unique_digit_solution_l65_65433


namespace infinite_radical_solution_l65_65063

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l65_65063


namespace length_of_arc_l65_65089

theorem length_of_arc (S : ℝ) (α : ℝ) (hS : S = 4) (hα : α = 2) : 
  ∃ l : ℝ, l = 4 :=
by
  sorry

end length_of_arc_l65_65089


namespace no_common_points_l65_65464

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def curve2 (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

theorem no_common_points :
  ¬ ∃ (x y : ℝ), curve1 x y ∧ curve2 x y :=
by sorry

end no_common_points_l65_65464


namespace area_S3_l65_65914

theorem area_S3 {s1 s2 s3 : ℝ} (h1 : s1^2 = 25)
  (h2 : s2 = s1 / Real.sqrt 2)
  (h3 : s3 = s2 / Real.sqrt 2)
  : s3^2 = 6.25 :=
by
  sorry

end area_S3_l65_65914


namespace f_periodic_analytic_expression_f_distinct_real_roots_l65_65952

noncomputable def f (x : ℝ) (k : ℤ) : ℝ := (x - 2 * k)^2

def I_k (k : ℤ) : Set ℝ := { x | 2 * k - 1 < x ∧ x ≤ 2 * k + 1 }

def M_k (k : ℕ) : Set ℝ := { a | 0 < a ∧ a ≤ 1 / (2 * ↑k + 1) }

theorem f_periodic (x : ℝ) (k : ℤ) : f x k = f (x - 2 * k) 0 := by
  sorry

theorem analytic_expression_f (x : ℝ) (k : ℤ) (hx : x ∈ I_k k) : f x k = (x - 2 * k)^2 := by
  sorry

theorem distinct_real_roots (k : ℕ) (a : ℝ) (h : a ∈ M_k k) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ I_k k ∧ x2 ∈ I_k k ∧ f x1 k = a * x1 ∧ f x2 k = a * x2 := by
  sorry

end f_periodic_analytic_expression_f_distinct_real_roots_l65_65952


namespace fixed_point_line_l65_65824

theorem fixed_point_line (k : ℝ) :
  ∃ A : ℝ × ℝ, (3 + k) * A.1 - 2 * A.2 + 1 - k = 0 ∧ (A = (1, 2)) :=
by
  let A : ℝ × ℝ := (1, 2)
  use A
  sorry

end fixed_point_line_l65_65824


namespace gcd_4557_1953_5115_l65_65010

theorem gcd_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 :=
by
  -- We use 'sorry' to skip the proof part as per the instructions.
  sorry

end gcd_4557_1953_5115_l65_65010


namespace victoria_more_scoops_l65_65391

theorem victoria_more_scoops (Oli_scoops : ℕ) (Victoria_scoops : ℕ) 
  (hOli : Oli_scoops = 4) (hVictoria : Victoria_scoops = 2 * Oli_scoops) : 
  (Victoria_scoops - Oli_scoops) = 4 :=
by
  sorry

end victoria_more_scoops_l65_65391


namespace find_wrong_number_read_l65_65416

theorem find_wrong_number_read (avg_initial avg_correct num_total wrong_num : ℕ) 
    (h1 : avg_initial = 15)
    (h2 : avg_correct = 16)
    (h3 : num_total = 10)
    (h4 : wrong_num = 36) 
    : wrong_num - (avg_correct * num_total - avg_initial * num_total) = 26 := 
by
  -- This is where the proof would go.
  sorry

end find_wrong_number_read_l65_65416


namespace bill_experience_l65_65597

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l65_65597


namespace Randy_blocks_used_l65_65139

theorem Randy_blocks_used (blocks_tower : ℕ) (blocks_house : ℕ) (total_blocks_used : ℕ) :
  blocks_tower = 27 → blocks_house = 53 → total_blocks_used = (blocks_tower + blocks_house) → total_blocks_used = 80 :=
by
  sorry

end Randy_blocks_used_l65_65139


namespace pond_depth_range_l65_65275

theorem pond_depth_range (d : ℝ) (adam_false : d < 10) (ben_false : d > 8) (carla_false : d ≠ 7) : 
    8 < d ∧ d < 10 :=
by
  sorry

end pond_depth_range_l65_65275


namespace river_depth_in_mid_may_l65_65238

variable (D : ℕ)
variable (h1 : D + 10 - 5 + 8 = 45)

theorem river_depth_in_mid_may (h1 : D + 13 = 45) : D = 32 := by
  sorry

end river_depth_in_mid_may_l65_65238


namespace range_of_a_l65_65095

noncomputable
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (A a ∪ B = B) ↔ a < -4 ∨ a > 5 :=
sorry

end range_of_a_l65_65095


namespace opposite_of_neg_3_is_3_l65_65724

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l65_65724


namespace ball_drawing_ways_l65_65156

theorem ball_drawing_ways :
    ∃ (r w y : ℕ), 
      0 ≤ r ∧ r ≤ 2 ∧
      0 ≤ w ∧ w ≤ 3 ∧
      0 ≤ y ∧ y ≤ 5 ∧
      r + w + y = 5 ∧
      10 ≤ 5 * r + 2 * w + y ∧ 
      5 * r + 2 * w + y ≤ 15 := 
sorry

end ball_drawing_ways_l65_65156


namespace product_of_intersection_coords_l65_65783

open Real

-- Define the two circles
def circle1 (x y: ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 21 = 0
def circle2 (x y: ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 52 = 0

-- Prove that the product of the coordinates of intersection points equals 189
theorem product_of_intersection_coords :
  (∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧ circle1 x2 y2 ∧ circle2 x2 y2 ∧ x1 * y1 * x2 * y2 = 189) :=
by
  sorry

end product_of_intersection_coords_l65_65783


namespace no_valid_rectangles_l65_65309

theorem no_valid_rectangles 
  (a b x y : ℝ) (h_ab_lt : a < b) (h_xa_lt : x < a) (h_ya_lt : y < a) 
  (h_perimeter : 2 * (x + y) = (2 * (a + b)) / 3) 
  (h_area : x * y = (a * b) / 3) : false := 
sorry

end no_valid_rectangles_l65_65309


namespace balls_boxes_exactly_3_non_matching_l65_65199

/-- The number of ways to place 10 balls labeled 1 to 10 into 10 boxes 
such that each box contains one ball, and exactly 3 balls are placed in boxes 
with non-matching labels, is 360. -/
theorem balls_boxes_exactly_3_non_matching : 
  ∃ (perm : Equiv.Perm (Fin 10)), (∃ (non_matching : Finset (Fin 10)), 
    non_matching.card = 3 ∧ 
    ∀ x ∈ non_matching, perm x ≠ x) → 
    (∃ (matching : Finset (Fin 10)), 
    matching.card = 7 ∧ 
    ∀ x ∈ matching, perm x = x) :=
sorry

end balls_boxes_exactly_3_non_matching_l65_65199


namespace bob_total_investment_l65_65195

variable (x : ℝ) -- the amount invested at 14%

noncomputable def total_investment_amount : ℝ :=
  let interest18 := 7000 * 0.18
  let interest14 := x * 0.14
  let total_interest := 3360
  let total_investment := 7000 + x
  total_investment

theorem bob_total_investment (h : 7000 * 0.18 + x * 0.14 = 3360) :
  total_investment_amount x = 22000 := by
  sorry

end bob_total_investment_l65_65195


namespace range_of_function_x_l65_65729

theorem range_of_function_x (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := sorry

end range_of_function_x_l65_65729


namespace point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l65_65925

def is_on_line (n : ℕ) (a_n : ℕ) : Prop := a_n = 2 * n + 1

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n m, a n - a m = d * (n - m)

theorem point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence (a : ℕ → ℕ) :
  (∀ n, is_on_line n (a n)) → is_arithmetic_sequence a ∧ 
  ¬ (is_arithmetic_sequence a → ∀ n, is_on_line n (a n)) :=
sorry

end point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l65_65925


namespace function_max_min_l65_65494

theorem function_max_min (a b c : ℝ) (h : a ≠ 0) (h1 : ∃ xₘ xₘₐ : ℝ, (0 < xₘ ∧ xₘ < xₘₐ ∧ xₘₐ < ∞) ∧ 
  (∀ x ∈ set.Ioo 0 ∞, dite (f' x = 0) (λ _, differentiable_at ℝ (f' x)) (λ _, true))) :
  (ab : ℝ > 0) ∧ (b^2 + 8ac > 0) ∧ (ac < 0) :=
by
  -- Define the function
  let f := λ x : ℝ, a * log x + b / x + c / x^2
  have h_f_domain : ∀ x, x ∈ set.Ioi (0 : ℝ) → differentiable_at ℝ (f x),
    from sorry
  have h_f_deriv : ∀ x, x ∈ set.Ioi (0 : ℝ) → deriv (f x) = a / x - b / x^2 - 2 * c / x^3,
    from sorry
  have h_f_critical : ∀ x, deriv (f x) = 0 → ∃ xₘ xₘₐ, (xₘ * xₘₐ) > 0 ∧ fourier.coefficients xₘ + xₘₐ > 0,
    from sorry
  show  (ab : ℝ > 0) ∧ (b^2 + 8ac > 0) ∧ (ac < 0),
    from sorry

end function_max_min_l65_65494


namespace problem_solution_l65_65278

theorem problem_solution :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 :=
by sorry

end problem_solution_l65_65278


namespace probability_of_selecting_one_is_correct_l65_65324

-- Define the number of elements in the first 20 rows of Pascal's triangle
def totalElementsInPascalFirst20Rows : ℕ := 210

-- Define the number of ones in the first 20 rows of Pascal's triangle
def totalOnesInPascalFirst20Rows : ℕ := 39

-- The probability as a rational number
def probabilityOfSelectingOne : ℚ := totalOnesInPascalFirst20Rows / totalElementsInPascalFirst20Rows

theorem probability_of_selecting_one_is_correct :
  probabilityOfSelectingOne = 13 / 70 :=
by
  -- Proof is omitted
  sorry

end probability_of_selecting_one_is_correct_l65_65324


namespace find_three_digit_number_l65_65033

theorem find_three_digit_number (A B C : ℕ) (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) : 100 * A + 10 * B + C = 253 :=
by {
  sorry
}

end find_three_digit_number_l65_65033


namespace candy_not_chocolate_l65_65518

theorem candy_not_chocolate (candy_total : ℕ) (bags : ℕ) (choc_heart_bags : ℕ) (choc_kiss_bags : ℕ) : 
  candy_total = 63 ∧ bags = 9 ∧ choc_heart_bags = 2 ∧ choc_kiss_bags = 3 → 
  (candy_total - (choc_heart_bags * (candy_total / bags) + choc_kiss_bags * (candy_total / bags))) = 28 :=
by
  intros h
  sorry

end candy_not_chocolate_l65_65518


namespace younger_brother_age_l65_65017

variable (x y : ℕ)

-- Conditions
axiom sum_of_ages : x + y = 46
axiom younger_is_third_plus_ten : y = x / 3 + 10

theorem younger_brother_age : y = 19 := 
by
  sorry

end younger_brother_age_l65_65017


namespace div_by_10_l65_65963

theorem div_by_10 (n : ℕ) (hn : 10 ∣ (3^n + 1)) : 10 ∣ (3^(n+4) + 1) :=
by
  sorry

end div_by_10_l65_65963


namespace pascal_triangle_prob_1_l65_65318

theorem pascal_triangle_prob_1 : 
  let total_elements := (20 * 21) / 2,
      num_ones := 19 * 2 + 1
  in (num_ones / total_elements = 39 / 210) := by
  sorry

end pascal_triangle_prob_1_l65_65318


namespace average_of_4_8_N_l65_65150

-- Define the condition for N
variable (N : ℝ) (cond : 7 < N ∧ N < 15)

-- State the theorem to prove
theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) :
  (frac12 + N) / 3 = 7 ∨ (12 + N) / 3 = 9 :=
sorry

end average_of_4_8_N_l65_65150


namespace speed_of_stream_correct_l65_65422

noncomputable def speed_of_stream : ℝ :=
  let boat_speed := 9
  let distance := 210
  let total_distance := 2 * distance
  let total_time := 84
  let x := Real.sqrt 39
  x

theorem speed_of_stream_correct :
  ∀ (boat_speed distance total_time : ℝ), 
    boat_speed = 9 ∧ distance = 210 ∧ total_time = 84 →
    ∃ (x : ℝ), 
      x = speed_of_stream ∧ 
      (210 / (boat_speed + x) + 210 / (boat_speed - x) = 84) := 
by
  sorry

end speed_of_stream_correct_l65_65422


namespace algebraic_expression_evaluation_l65_65921

theorem algebraic_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ( ( (a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) ) = 1 :=
by sorry

end algebraic_expression_evaluation_l65_65921


namespace arithmetic_sequence_common_difference_l65_65015

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : S 5 = 35)
  (h3 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l65_65015


namespace solution_k_system_eq_l65_65113

theorem solution_k_system_eq (x y k : ℝ) 
  (h1 : x + y = 5 * k) 
  (h2 : x - y = k) 
  (h3 : 2 * x + 3 * y = 24) : k = 2 :=
by
  sorry

end solution_k_system_eq_l65_65113


namespace min_value_of_c_l65_65528

variable {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0)
variable (hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
variable (semi_focal_dist : c = Real.sqrt (a^2 + b^2))
variable (distance_condition : ∀ (d : ℝ), d = a * b / c = 1 / 3 * c + 1)

theorem min_value_of_c : c = 6 := 
sorry

end min_value_of_c_l65_65528


namespace perpendicular_line_through_circle_center_l65_65338

theorem perpendicular_line_through_circle_center :
  ∀ (x y : ℝ), (x^2 + (y-1)^2 = 4) → (3*x + 2*y + 1 = 0) → (2*x - 3*y + 3 = 0) :=
by
  intros x y h_circle h_line
  sorry

end perpendicular_line_through_circle_center_l65_65338


namespace capital_payment_l65_65183

theorem capital_payment (m : ℕ) (hm : m ≥ 3) : 
  ∃ d : ℕ, d = (1000 * (3^m - 2^(m-1))) / (3^m - 2^m) 
  ∧ (∃ a : ℕ, a = 4000 ∧ a = ((3/2)^(m-1) * (3000 - 3 * d) + 2 * d)) := 
by
  sorry

end capital_payment_l65_65183


namespace bill_experience_now_l65_65600

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l65_65600


namespace ann_age_l65_65775

theorem ann_age {a b y : ℕ} (h1 : a + b = 44) (h2 : y = a - b) (h3 : b = a / 2 + 2 * (a - b)) : a = 24 :=
by
  sorry

end ann_age_l65_65775


namespace filling_rate_in_cubic_meters_per_hour_l65_65551

def barrels_per_minute_filling_rate : ℝ := 3
def liters_per_barrel : ℝ := 159
def liters_per_cubic_meter : ℝ := 1000
def minutes_per_hour : ℝ := 60

theorem filling_rate_in_cubic_meters_per_hour :
  (barrels_per_minute_filling_rate * liters_per_barrel / liters_per_cubic_meter * minutes_per_hour) = 28.62 :=
sorry

end filling_rate_in_cubic_meters_per_hour_l65_65551


namespace q_at_2_equals_9_l65_65591

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

-- Define the function q(x)
noncomputable def q (x : ℝ) : ℝ :=
sgn (3 * x - 1) * |3 * x - 1| ^ (1/2) +
3 * sgn (3 * x - 1) * |3 * x - 1| ^ (1/3) +
|3 * x - 1| ^ (1/4)

-- The theorem stating that q(2) equals 9
theorem q_at_2_equals_9 : q 2 = 9 :=
by sorry

end q_at_2_equals_9_l65_65591


namespace find_initial_principal_amount_l65_65796

noncomputable def compound_interest (initial_principal : ℝ) : ℝ :=
  let year1 := initial_principal * 1.09
  let year2 := (year1 + 500) * 1.10
  let year3 := (year2 - 300) * 1.08
  let year4 := year3 * 1.08
  let year5 := year4 * 1.09
  year5

theorem find_initial_principal_amount :
  ∃ (P : ℝ), (|compound_interest P - 1120| < 0.01) :=
sorry

end find_initial_principal_amount_l65_65796


namespace probability_red_white_red_l65_65442

-- Definitions and assumptions
def total_marbles := 10
def red_marbles := 4
def white_marbles := 6

def P_first_red : ℚ := red_marbles / total_marbles
def P_second_white_given_first_red : ℚ := white_marbles / (total_marbles - 1)
def P_third_red_given_first_red_and_second_white : ℚ := (red_marbles - 1) / (total_marbles - 2)

-- The target probability hypothesized
theorem probability_red_white_red :
  P_first_red * P_second_white_given_first_red * P_third_red_given_first_red_and_second_white = 1 / 10 :=
by
  sorry

end probability_red_white_red_l65_65442


namespace nth_equation_l65_65134

theorem nth_equation (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := 
by 
  sorry

end nth_equation_l65_65134


namespace volume_not_occupied_by_cones_l65_65022

/-- Two cones with given dimensions are enclosed in a cylinder, and we want to find the volume 
    in the cylinder not occupied by the cones. -/
theorem volume_not_occupied_by_cones : 
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  volume_cylinder - total_volume_cones = (2600 / 3) * π :=
by
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  sorry

end volume_not_occupied_by_cones_l65_65022


namespace min_value_xyz_l65_65798

open Real

theorem min_value_xyz
  (x y z : ℝ)
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : 5 * x + 16 * y + 33 * z ≥ 136) :
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 :=
sorry

end min_value_xyz_l65_65798


namespace remainder_and_division_l65_65871

theorem remainder_and_division (n : ℕ) (h1 : n = 1680) (h2 : n % 9 = 0) : 
  1680 % 1677 = 3 :=
by {
  sorry
}

end remainder_and_division_l65_65871


namespace ratio_of_times_l65_65572

-- Given conditions as definitions
def distance : ℕ := 630 -- distance in km
def previous_time : ℕ := 6 -- time in hours
def new_speed : ℕ := 70 -- speed in km/h

-- Calculation of times
def previous_speed : ℕ := distance / previous_time

def new_time : ℕ := distance / new_speed

-- Main theorem statement
theorem ratio_of_times :
  (new_time : ℚ) / (previous_time : ℚ) = 3 / 2 :=
  sorry

end ratio_of_times_l65_65572


namespace beta_gt_half_alpha_l65_65533

theorem beta_gt_half_alpha (alpha beta : ℝ) (h1 : Real.sin beta = (3/4) * Real.sin alpha) (h2 : 0 < alpha ∧ alpha ≤ 90) : beta > alpha / 2 :=
by
  sorry

end beta_gt_half_alpha_l65_65533


namespace min_intersection_l65_65621

open Finset

-- Definition of subset count function
def n (S : Finset ℕ) : ℕ :=
  2 ^ S.card

theorem min_intersection {A B C : Finset ℕ} (hA : A.card = 100) (hB : B.card = 100) 
  (h_subsets : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≥ 97 := by
  sorry

end min_intersection_l65_65621


namespace problem_solution_l65_65843

theorem problem_solution
  (n m k l : ℕ)
  (h1 : n ≠ 1)
  (h2 : 0 < n)
  (h3 : 0 < m)
  (h4 : 0 < k)
  (h5 : 0 < l)
  (h6 : n^k + m * n^l + 1 ∣ n^(k + l) - 1) :
  (m = 1 ∧ l = 2 * k) ∨ (l ∣ k ∧ m = (n^(k - l) - 1) / (n^l - 1)) :=
by
  sorry

end problem_solution_l65_65843


namespace min_calls_correct_l65_65143

-- Define a function that calculates the minimum number of calls given n people
def min_calls (n : ℕ) : ℕ :=
  2 * n - 2

-- Theorem to prove that min_calls(n) given the conditions is equal to 2n - 2
theorem min_calls_correct (n : ℕ) (h : n ≥ 2) : min_calls n = 2 * n - 2 :=
by
  sorry

end min_calls_correct_l65_65143


namespace lowest_possible_price_l65_65449

theorem lowest_possible_price
  (MSRP : ℕ) (max_initial_discount_percent : ℕ) (platinum_discount_percent : ℕ)
  (h1 : MSRP = 35) (h2 : max_initial_discount_percent = 40) (h3 : platinum_discount_percent = 30) :
  let initial_discount := max_initial_discount_percent * MSRP / 100
  let price_after_initial_discount := MSRP - initial_discount
  let platinum_discount := platinum_discount_percent * price_after_initial_discount / 100
  let lowest_price := price_after_initial_discount - platinum_discount
  lowest_price = 147 / 10 :=
by
  sorry

end lowest_possible_price_l65_65449


namespace binary_sum_correct_l65_65585

-- Definitions of the binary numbers
def bin1 : ℕ := 0b1011
def bin2 : ℕ := 0b101
def bin3 : ℕ := 0b11001
def bin4 : ℕ := 0b1110
def bin5 : ℕ := 0b100101

-- The statement to prove
theorem binary_sum_correct : bin1 + bin2 + bin3 + bin4 + bin5 = 0b1111010 := by
  sorry

end binary_sum_correct_l65_65585


namespace trader_cloth_sold_l65_65314

variable (x : ℕ)
variable (profit_per_meter total_profit : ℕ)

theorem trader_cloth_sold (h_profit_per_meter : profit_per_meter = 55)
  (h_total_profit : total_profit = 2200) :
  55 * x = 2200 → x = 40 :=
by 
  sorry

end trader_cloth_sold_l65_65314


namespace area_of_pentagon_l65_65903

-- Condition that BD is the diagonal of a square ABCD with length 20 cm
axiom BD_diagonal_of_square : ∀ (s : ℝ), s * Real.sqrt 2 = 20 → s = 10 * Real.sqrt 2

-- Definition of the sides and areas involved
def side_length_of_square (d : ℝ) : ℝ := d / Real.sqrt 2
def area_of_square (s : ℝ) : ℝ := s * s
def area_of_triangle (s : ℝ) : ℝ := (1 / 2) * s * s

-- The main problem statement to be proven in Lean
theorem area_of_pentagon (d : ℝ) (h : d = 20) : 
  area_of_square (side_length_of_square d) + area_of_triangle (side_length_of_square d) = 300 := by
  sorry

end area_of_pentagon_l65_65903


namespace max_sum_at_n_is_6_l65_65221

-- Assuming an arithmetic sequence a_n where a_1 = 4 and d = -5/7
def arithmetic_seq (n : ℕ) : ℚ := (33 / 7) - (5 / 7) * n

-- Sum of the first n terms (S_n) of the arithmetic sequence {a_n}
def sum_arithmetic_seq (n : ℕ) : ℚ := (n / 2) * (2 * (arithmetic_seq 1) + (n - 1) * (-5 / 7))

theorem max_sum_at_n_is_6 
  (a_1 : ℚ) (d : ℚ) (h1 : a_1 = 4) (h2 : d = -5/7) :
  ∀ n : ℕ, sum_arithmetic_seq n ≤ sum_arithmetic_seq 6 :=
by
  sorry

end max_sum_at_n_is_6_l65_65221


namespace sum_of_numbers_l65_65910

theorem sum_of_numbers : 148 + 35 + 17 + 13 + 9 = 222 := 
by
  sorry

end sum_of_numbers_l65_65910


namespace father_twice_marika_age_in_2036_l65_65683

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l65_65683


namespace triangle_area_is_18_l65_65874

noncomputable def area_triangle : ℝ :=
  let vertices : List (ℝ × ℝ) := [(1, 2), (7, 6), (1, 8)]
  let base := (8 - 2) -- Length between (1, 2) and (1, 8)
  let height := (7 - 1) -- Perpendicular distance from (7, 6) to x = 1
  (1 / 2) * base * height

theorem triangle_area_is_18 : area_triangle = 18 := by
  sorry

end triangle_area_is_18_l65_65874


namespace original_combined_price_l65_65620

theorem original_combined_price (C S : ℝ)
  (hC_new : (C + 0.25 * C) = 12.5)
  (hS_new : (S + 0.50 * S) = 13.5) :
  (C + S) = 19 := by
  -- sorry makes sure to skip the proof
  sorry

end original_combined_price_l65_65620


namespace solution_l65_65641

def problem (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (x + a) * (x - 3) = x^2 + 2 * x - b

theorem solution (a b : ℝ) (h : problem a b) : a - b = -10 :=
  sorry

end solution_l65_65641


namespace minimum_value_of_n_l65_65446

open Int

theorem minimum_value_of_n (n d : ℕ) (h1 : n > 0) (h2 : d > 0) (h3 : d % n = 0)
    (h4 : 10 * n - 20 = 90) : n = 11 :=
by
  sorry

end minimum_value_of_n_l65_65446


namespace trains_pass_time_l65_65035

def length_train1 : ℕ := 200
def length_train2 : ℕ := 280

def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * 1000 / 3600

def relative_speed_mps : ℚ :=
  kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

def total_length : ℕ :=
  length_train1 + length_train2

def time_to_pass_trains : ℚ :=
  total_length / relative_speed_mps

theorem trains_pass_time :
  time_to_pass_trains = 24 := by
  sorry

end trains_pass_time_l65_65035


namespace sixty_fifth_term_is_sixteen_l65_65915

def apply_rule (n : ℕ) : ℕ :=
  if n <= 12 then
    7 * n
  else if n % 2 = 0 then
    n - 7
  else
    n / 3

def sequence_term (a_0 : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate apply_rule n a_0

theorem sixty_fifth_term_is_sixteen : sequence_term 65 64 = 16 := by
  sorry

end sixty_fifth_term_is_sixteen_l65_65915


namespace gcd_lcm_sum_ge_sum_l65_65960

theorem gcd_lcm_sum_ge_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  Nat.gcd a b + Nat.lcm a b ≥ a + b := 
sorry

end gcd_lcm_sum_ge_sum_l65_65960


namespace Sandra_brought_20_pairs_l65_65131

-- Definitions for given conditions
variable (S : ℕ) -- S for Sandra's pairs of socks
variable (C : ℕ) -- C for Lisa's cousin's pairs of socks

-- Conditions translated into Lean definitions
def initial_pairs : ℕ := 12
def mom_pairs : ℕ := 3 * initial_pairs + 8 -- Lisa's mom brought 8 more than three times the number of pairs Lisa started with
def cousin_pairs (S : ℕ) : ℕ := S / 5       -- Lisa's cousin brought one-fifth the number of pairs that Sandra bought
def total_pairs (S : ℕ) : ℕ := initial_pairs + S + cousin_pairs S + mom_pairs -- Total pairs of socks Lisa ended up with

-- The theorem to prove
theorem Sandra_brought_20_pairs (h : total_pairs S = 80) : S = 20 :=
by
  sorry

end Sandra_brought_20_pairs_l65_65131


namespace king_total_payment_l65_65757

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l65_65757


namespace men_absent_l65_65296

theorem men_absent (original_men absent_men remaining_men : ℕ) (total_work : ℕ) 
  (h1 : original_men = 15) (h2 : total_work = original_men * 40) (h3 : 60 * remaining_men = total_work) : 
  remaining_men = original_men - absent_men → absent_men = 5 := 
by
  sorry

end men_absent_l65_65296


namespace percentage_of_sikh_boys_l65_65648

-- Define the conditions
def total_boys : ℕ := 850
def percentage_muslim_boys : ℝ := 0.46
def percentage_hindu_boys : ℝ := 0.28
def boys_other_communities : ℕ := 136

-- Theorem to prove the percentage of Sikh boys is 10%
theorem percentage_of_sikh_boys : 
  (((total_boys - 
      (percentage_muslim_boys * total_boys + 
       percentage_hindu_boys * total_boys + 
       boys_other_communities))
    / total_boys) * 100 = 10) :=
by
  -- sorry prevents the need to provide proof here
  sorry

end percentage_of_sikh_boys_l65_65648


namespace polynomial_identity_l65_65098

theorem polynomial_identity (a b c d e f : ℤ)
  (h_eq : ∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
  sorry

end polynomial_identity_l65_65098


namespace evaluate_expression_l65_65206

theorem evaluate_expression :
  - (18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l65_65206


namespace train_speed_l65_65315

/-- 
Theorem: Given the length of the train L = 1200 meters and the time T = 30 seconds, the speed of the train S is 40 meters per second.
-/
theorem train_speed (L : ℕ) (T : ℕ) (hL : L = 1200) (hT : T = 30) : L / T = 40 := by
  sorry

end train_speed_l65_65315


namespace find_angle_C_l65_65803

noncomputable def area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

-- Given values
def A : ℝ := 3 - Real.sqrt 3
def B_obtuse : Prop := Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2
def AB : ℝ := 2
def BC : ℝ := Real.sqrt 3 - 1
noncomputable def AC : ℝ := Real.sqrt 6

-- Main theorem 
theorem find_angle_C :
  area AB AC (2 * Real.pi / 3) = A / 2 →
  Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 → 
  ∠C = 45 :=
by
  intro area_eq sinC_eq
  sorry

end find_angle_C_l65_65803


namespace valid_x_for_expression_l65_65235

theorem valid_x_for_expression :
  (∃ x : ℝ, x = 8 ∧ (10 - x ≥ 0) ∧ (x - 4 ≠ 0)) ↔ (∃ x : ℝ, x = 8) :=
by
  sorry

end valid_x_for_expression_l65_65235


namespace distance_of_intersection_points_l65_65637

def C1 (x y : ℝ) : Prop := x - y + 4 = 0
def C2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

theorem distance_of_intersection_points {A B : ℝ × ℝ} (hA1 : C1 A.fst A.snd) (hA2 : C2 A.fst A.snd)
  (hB1 : C1 B.fst B.snd) (hB2 : C2 B.fst B.snd) : dist A B = Real.sqrt 2 := by
  sorry

end distance_of_intersection_points_l65_65637


namespace king_paid_after_tip_l65_65760

theorem king_paid_after_tip:
  (crown_cost tip_percentage total_cost : ℝ)
  (h_crown_cost : crown_cost = 20000)
  (h_tip_percentage : tip_percentage = 0.1) :
  total_cost = crown_cost + (crown_cost * tip_percentage) :=
by
  have h_tip := h_crown_cost.symm ▸ h_tip_percentage.symm ▸ 20000 * 0.1
  have h_total := h_crown_cost.symm ▸ (h_tip.symm ▸ 2000)
  rw [h_crown_cost, h_tip, h_total]
  exact rfl

end king_paid_after_tip_l65_65760


namespace roots_real_and_equal_l65_65608

theorem roots_real_and_equal :
  ∀ x : ℝ,
  (x^2 - 4 * x * Real.sqrt 5 + 20 = 0) →
  (Real.sqrt ((-4 * Real.sqrt 5)^2 - 4 * 1 * 20) = 0) →
  (∃ r : ℝ, x = r ∧ x = r) :=
by
  intro x h_eq h_discriminant
  sorry

end roots_real_and_equal_l65_65608


namespace complement_U_A_l65_65092

open Set

def U : Set ℝ := {x | -3 < x ∧ x < 3}
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem complement_U_A : 
  (U \ A) = {x | -3 < x ∧ x ≤ -2} ∪ {x | 1 < x ∧ x < 3} :=
by
  sorry

end complement_U_A_l65_65092


namespace first_discount_percentage_l65_65838

/-
  Prove that under the given conditions:
  1. The price before the first discount is $33.78.
  2. The final price after the first and second discounts is $19.
  3. The second discount is 25%.
-/
theorem first_discount_percentage (x : ℝ) :
  (33.78 * (1 - x / 100) * (1 - 25 / 100) = 19) →
  x = 25 :=
by
  -- Proof steps (to be filled)
  sorry

end first_discount_percentage_l65_65838


namespace watermelon_count_l65_65850

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ)
  (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) : total_seeds / seeds_per_watermelon = 4 :=
by
  sorry

end watermelon_count_l65_65850


namespace adjustments_to_equal_boys_and_girls_l65_65853

theorem adjustments_to_equal_boys_and_girls (n : ℕ) :
  let initial_boys := 40
  let initial_girls := 0
  let boys_after_n := initial_boys - 3 * n
  let girls_after_n := initial_girls + 2 * n
  boys_after_n = girls_after_n → n = 8 :=
by
  sorry

end adjustments_to_equal_boys_and_girls_l65_65853


namespace point_D_not_in_region_l65_65764

-- Define the condition that checks if a point is not in the region defined by 3x + 2y < 6
def point_not_in_region (x y : ℝ) : Prop :=
  ¬ (3 * x + 2 * y < 6)

-- Define the points
def A := (0, 0)
def B := (1, 1)
def C := (0, 2)
def D := (2, 0)

-- The proof problem as a Lean statement
theorem point_D_not_in_region : point_not_in_region (2:ℝ) (0:ℝ) :=
by
  show point_not_in_region 2 0
  sorry

end point_D_not_in_region_l65_65764


namespace find_k_for_xy_solution_l65_65918

theorem find_k_for_xy_solution :
  ∀ (k : ℕ), (∃ (x y : ℕ), x * (x + k) = y * (y + 1))
  → k = 1 ∨ k ≥ 4 :=
by
  intros k h
  sorry -- proof goes here

end find_k_for_xy_solution_l65_65918


namespace number_of_pieces_of_paper_used_l65_65658

theorem number_of_pieces_of_paper_used
  (P : ℕ)
  (h1 : 1 / 5 > 0)
  (h2 : 2 / 5 > 0)
  (h3 : 1 < (P : ℝ) * (1 / 5) + 2 / 5 ∧ (P : ℝ) * (1 / 5) + 2 / 5 ≤ 2) : 
  P = 8 :=
sorry

end number_of_pieces_of_paper_used_l65_65658


namespace common_chord_circle_eq_l65_65094

theorem common_chord_circle_eq {a b : ℝ} (hb : b ≠ 0) :
  ∃ x y : ℝ, 
    (x^2 + y^2 - 2 * a * x = 0) ∧ 
    (x^2 + y^2 - 2 * b * y = 0) ∧ 
    (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0 :=
by sorry

end common_chord_circle_eq_l65_65094


namespace halfway_fraction_l65_65992

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l65_65992


namespace angle_FBG_l65_65353

noncomputable def midpoint (A B : Point) : Point := (A + B) / 2

theorem angle_FBG {A B C D E F G : Point}
  (square_ABCD : square A B C D)
  (midpoint_E : E = midpoint B C)
  (perpendicular_BF : ∃ P : Line, perpendicular P (Line.mk A E) ∧ contains (Line.mk B F) P)
  (perpendicular_DG : ∃ Q : Line, perpendicular Q (Line.mk A E) ∧ contains (Line.mk D G) Q) :
  angle F B G = 45 :=
  sorry

notation "square" := square unit.square
notation "Line.mk" := mk
notation "midpoint" := midpoint
notation "angle" := angle

end angle_FBG_l65_65353


namespace probability_one_in_first_20_rows_l65_65323

theorem probability_one_in_first_20_rows :
  let total_elements := 210
  let number_of_ones := 39
  (number_of_ones / total_elements : ℚ) = 13 / 70 :=
by
  sorry

end probability_one_in_first_20_rows_l65_65323


namespace parallel_lines_equal_slopes_l65_65632

theorem parallel_lines_equal_slopes (a : ℝ) :
  (∀ x y, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = -7 + a) →
  a = 3 := sorry

end parallel_lines_equal_slopes_l65_65632


namespace find_even_odd_functions_l65_65801

variable {X : Type} [AddGroup X]

def even_function (f : X → X) : Prop :=
∀ x, f (-x) = f x

def odd_function (f : X → X) : Prop :=
∀ x, f (-x) = -f x

theorem find_even_odd_functions
  (f g : X → X)
  (h_even : even_function f)
  (h_odd : odd_function g)
  (h_eq : ∀ x, f x + g x = 0) :
  (∀ x, f x = 0) ∧ (∀ x, g x = 0) :=
sorry

end find_even_odd_functions_l65_65801


namespace find_matrix_N_l65_65209

theorem find_matrix_N (N : Matrix (Fin 4) (Fin 4) ℤ)
  (hi : N.mulVec ![1, 0, 0, 0] = ![3, 4, -9, 1])
  (hj : N.mulVec ![0, 1, 0, 0] = ![-1, 6, -3, 2])
  (hk : N.mulVec ![0, 0, 1, 0] = ![8, -2, 5, 0])
  (hl : N.mulVec ![0, 0, 0, 1] = ![1, 0, 7, -1]) :
  N = ![![3, -1, 8, 1],
         ![4, 6, -2, 0],
         ![-9, -3, 5, 7],
         ![1, 2, 0, -1]] := by
  sorry

end find_matrix_N_l65_65209


namespace no_rational_roots_of_odd_coefficient_quadratic_l65_65534

theorem no_rational_roots_of_odd_coefficient_quadratic 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, r * r * a + r * b + c = 0 :=
by
  sorry

end no_rational_roots_of_odd_coefficient_quadratic_l65_65534


namespace largest_possible_median_l65_65085

theorem largest_possible_median (l : List ℕ) (h1 : l.length = 10) 
  (h2 : ∀ x ∈ l, 0 < x) (exists6l : ∃ l1 : List ℕ, l1 = [3, 4, 5, 7, 8, 9]) :
  ∃ median_val : ℝ, median_val = 8.5 := 
sorry

end largest_possible_median_l65_65085


namespace find_a_l65_65635

variable (a b c : ℤ)

theorem find_a (h1 : a + b = 2) (h2 : b + c = 0) (h3 : |c| = 1) : a = 3 ∨ a = 1 := 
sorry

end find_a_l65_65635


namespace greatest_combination_bathrooms_stock_l65_65133

theorem greatest_combination_bathrooms_stock 
  (toilet_paper : ℕ) 
  (soap : ℕ) 
  (towels : ℕ) 
  (shower_gels : ℕ) 
  (h_tp : toilet_paper = 36)
  (h_soap : soap = 18)
  (h_towels : towels = 24)
  (h_shower_gels : shower_gels = 12) : 
  Nat.gcd (Nat.gcd (Nat.gcd toilet_paper soap) towels) shower_gels = 6 :=
by
  sorry

end greatest_combination_bathrooms_stock_l65_65133


namespace convex_quadrilaterals_l65_65555

open Nat

theorem convex_quadrilaterals (n : ℕ) (h : n = 12) : 
  (choose n 4) = 495 :=
by
  rw h
  norm_num
  sorry

end convex_quadrilaterals_l65_65555


namespace pencils_lost_l65_65230

theorem pencils_lost (bought_pencils remaining_pencils lost_pencils : ℕ)
                     (h1 : bought_pencils = 16)
                     (h2 : remaining_pencils = 8)
                     (h3 : lost_pencils = bought_pencils - remaining_pencils) :
                     lost_pencils = 8 :=
by {
  sorry
}

end pencils_lost_l65_65230


namespace a_n_minus_1_has_n_distinct_prime_divisors_l65_65024

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 0     := 5
| (n+1) := (a n)^2

-- Theorem statement
theorem a_n_minus_1_has_n_distinct_prime_divisors (n : ℕ) (h : n ≥ 1) : (a n) - 1 ≥ n :=
by sorry

end a_n_minus_1_has_n_distinct_prime_divisors_l65_65024


namespace cost_price_of_radio_l65_65450

theorem cost_price_of_radio (C : ℝ) 
  (overhead_expenses : ℝ := 20) 
  (selling_price : ℝ := 300) 
  (profit_percent : ℝ := 22.448979591836732) :
  C = 228.57 :=
by
  sorry

end cost_price_of_radio_l65_65450


namespace greatest_M_inequality_l65_65793

theorem greatest_M_inequality :
  ∀ x y z : ℝ, x^4 + y^4 + z^4 + x * y * z * (x + y + z) ≥ (2/3) * (x * y + y * z + z * x)^2 :=
by
  sorry

end greatest_M_inequality_l65_65793


namespace sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l65_65213

theorem sin_30_eq_one_half : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

theorem cos_11pi_over_4_eq_neg_sqrt2_over_2 : Real.cos (11 * Real.pi / 4) = - Real.sqrt 2 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

end sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l65_65213


namespace gcd_of_powers_of_two_l65_65163

noncomputable def m := 2^2048 - 1
noncomputable def n := 2^2035 - 1

theorem gcd_of_powers_of_two : Int.gcd m n = 8191 := by
  sorry

end gcd_of_powers_of_two_l65_65163


namespace find_number_l65_65936

theorem find_number (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 := sorry

end find_number_l65_65936


namespace even_of_square_even_l65_65472

theorem even_of_square_even (a : Int) (h1 : ∃ n : Int, a = 2 * n) (h2 : Even (a ^ 2)) : Even a := 
sorry

end even_of_square_even_l65_65472


namespace earrings_ratio_l65_65054

theorem earrings_ratio :
  ∃ (M R : ℕ), 10 = M / 4 ∧ 10 + M + R = 70 ∧ M / R = 2 := by
  sorry

end earrings_ratio_l65_65054


namespace percentage_of_total_l65_65177

theorem percentage_of_total (total part : ℕ) (h₁ : total = 100) (h₂ : part = 30):
  (part / total) * 100 = 30 := by
  sorry

end percentage_of_total_l65_65177


namespace parallel_lines_slope_eq_l65_65224

theorem parallel_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 ↔ 6 * x + m * y + 11 = 0) → m = 8 :=
by
  sorry

end parallel_lines_slope_eq_l65_65224


namespace roots_sum_one_imp_b_eq_neg_a_l65_65463

theorem roots_sum_one_imp_b_eq_neg_a (a b c : ℝ) (h : a ≠ 0) 
  (hr : ∀ (r s : ℝ), r + s = 1 → (r * s = c / a) → a * (r^2 + (b/a) * r + c/a) = 0) : b = -a :=
sorry

end roots_sum_one_imp_b_eq_neg_a_l65_65463


namespace exist_infinitely_many_coprime_pairs_l65_65137

theorem exist_infinitely_many_coprime_pairs (a b : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : Nat.gcd a b = 1) : 
  ∃ (a b : ℕ), (a + b).mod (a^b + b^a) = 0 :=
sorry

end exist_infinitely_many_coprime_pairs_l65_65137


namespace fx_expression_cos_theta_value_l65_65469

open Real

noncomputable def vector_a (ω x : ℝ) : ℝ × ℝ := (1 + cos (ω * x), -1)
noncomputable def vector_b (ω x : ℝ) : ℝ × ℝ := (sqrt 3, sin (ω * x))
noncomputable def f (ω x : ℝ) : ℝ := (1 + cos (ω * x)) * sqrt 3 - sin (ω * x)

theorem fx_expression (ω : ℝ) (hω : ω > 0) (h_period : ∀ x, f ω (x + 2 * pi) = f ω x) :
  ∀ x, f ω x = sqrt 3 - 2 * sin (x - pi / 3) :=
sorry

theorem cos_theta_value (θ : ℝ) (hθ : θ ∈ Ioo 0 (pi / 2))
  (hfθ : f 1 θ = sqrt 3 + 6 / 5) : cos θ = (3 * sqrt 3 + 4) / 10 :=
sorry

end fx_expression_cos_theta_value_l65_65469


namespace find_loss_percentage_l65_65046

theorem find_loss_percentage (W : ℝ) (profit_percentage : ℝ) (remaining_percentage : ℝ)
  (overall_loss : ℝ) (stock_worth : ℝ) (L : ℝ) :
  W = 12499.99 →
  profit_percentage = 0.20 →
  remaining_percentage = 0.80 →
  overall_loss = -500 →
  0.04 * W - (L / 100) * (remaining_percentage * W) = overall_loss →
  L = 10 :=
by
  intro hW hprofit_percentage hremaining_percentage hoverall_loss heq
  -- We'll provide the proof here
  sorry

end find_loss_percentage_l65_65046


namespace man_work_alone_in_5_days_l65_65043

theorem man_work_alone_in_5_days (d : ℕ) (h1 : ∀ m : ℕ, (1 / (m : ℝ)) + 1 / 20 = 1 / 4):
  d = 5 := by
  sorry

end man_work_alone_in_5_days_l65_65043


namespace tic_tac_toe_lines_l65_65365

theorem tic_tac_toe_lines (n : ℕ) (h_pos : 0 < n) : 
  ∃ lines : ℕ, lines = (5^n - 3^n) / 2 :=
sorry

end tic_tac_toe_lines_l65_65365


namespace first_prize_prob_correct_second_prize_prob_correct_any_prize_prob_correct_l65_65451

namespace lottery_problem

noncomputable def prob_of_winning_first_prize : ℝ := 
  (choose 4 2 * choose 5 2 : ℝ) / (choose 10 2 * choose 10 2)

noncomputable def prob_of_winning_second_prize : ℝ :=
  ((choose 4 2 * choose 5 1 * choose 5 1) + (choose 4 1 * choose 6 1 * choose 5 2) : ℝ) / (choose 10 2 * choose 10 2)

noncomputable def prob_of_winning_any_prize : ℝ :=
  (1 - (choose 6 2 * choose 5 2 : ℝ) / (choose 10 2 * choose 10 2))

theorem first_prize_prob_correct : 
  prob_of_winning_first_prize = 4 / 135 := sorry

theorem second_prize_prob_correct : 
  prob_of_winning_second_prize = 26 / 135 := sorry

theorem any_prize_prob_correct : 
  prob_of_winning_any_prize = 75 / 81 := sorry

end lottery_problem

end first_prize_prob_correct_second_prize_prob_correct_any_prize_prob_correct_l65_65451


namespace total_players_count_l65_65988

theorem total_players_count (M W : ℕ) (h1 : W = M + 4) (h2 : (M : ℚ) / W = 5 / 9) : M + W = 14 :=
sorry

end total_players_count_l65_65988


namespace even_sum_probability_l65_65873

theorem even_sum_probability :
  let wheel1 := (2/6, 3/6, 1/6)   -- (probability of even, odd, zero) for the first wheel
  let wheel2 := (2/4, 2/4)        -- (probability of even, odd) for the second wheel
  let both_even := (1/3) * (1/2)  -- probability of both numbers being even
  let both_odd := (1/2) * (1/2)   -- probability of both numbers being odd
  let zero_and_even := (1/6) * (1/2)  -- probability of one number being zero and the other even
  let total_probability := both_even + both_odd + zero_and_even
  total_probability = 1/2 := by sorry

end even_sum_probability_l65_65873


namespace days_to_complete_work_l65_65181

theorem days_to_complete_work {D : ℝ} (h1 : D > 0)
  (h2 : (1 / D) + (2 / D) = 0.3) :
  D = 10 :=
sorry

end days_to_complete_work_l65_65181


namespace arithmetic_sequence_properties_l65_65086

-- Definitions and conditions
def S (n : ℕ) : ℤ := -2 * n^2 + 15 * n

-- Statement of the problem as a theorem
theorem arithmetic_sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = 17 - 4 * (n + 1)) ∧
  (∃ n : ℕ, S n = 28 ∧ ∀ m : ℕ, S m ≤ S n) :=
by {sorry}

end arithmetic_sequence_properties_l65_65086


namespace arctan_arcsin_arccos_sum_l65_65330

theorem arctan_arcsin_arccos_sum :
  (Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1 / 2) + Real.arccos 1 = 0) :=
by
  sorry

end arctan_arcsin_arccos_sum_l65_65330


namespace max_value_of_expression_l65_65664

theorem max_value_of_expression 
  (a b c : ℝ)
  (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  6 * a + 3 * b + 10 * c ≤ 3.2 :=
sorry

end max_value_of_expression_l65_65664


namespace ratio_bananas_apples_is_3_to_1_l65_65273

def ratio_of_bananas_to_apples (oranges apples bananas peaches total_fruit : ℕ) : ℚ :=
if oranges = 6 ∧ apples = oranges - 2 ∧ peaches = bananas / 2 ∧ total_fruit = 28
   ∧ 6 + apples + bananas + peaches = total_fruit then
    bananas / apples
else 0

theorem ratio_bananas_apples_is_3_to_1 : ratio_of_bananas_to_apples 6 4 12 6 28 = 3 := by
sorry

end ratio_bananas_apples_is_3_to_1_l65_65273


namespace total_cost_is_734_l65_65956

-- Define the cost of each ice cream flavor
def cost_vanilla : ℕ := 99
def cost_chocolate : ℕ := 129
def cost_strawberry : ℕ := 149

-- Define the amount of each flavor Mrs. Hilt buys
def num_vanilla : ℕ := 2
def num_chocolate : ℕ := 3
def num_strawberry : ℕ := 1

-- Calculate the total cost in cents
def total_cost : ℕ :=
  (num_vanilla * cost_vanilla) +
  (num_chocolate * cost_chocolate) +
  (num_strawberry * cost_strawberry)

-- Statement of the proof problem
theorem total_cost_is_734 : total_cost = 734 :=
by
  sorry

end total_cost_is_734_l65_65956


namespace second_integer_is_ninety_point_five_l65_65016

theorem second_integer_is_ninety_point_five
  (n : ℝ)
  (first_integer fourth_integer : ℝ)
  (h1 : first_integer = n - 2)
  (h2 : fourth_integer = n + 1)
  (h_sum : first_integer + fourth_integer = 180) :
  n = 90.5 :=
by
  -- sorry to skip the proof
  sorry

end second_integer_is_ninety_point_five_l65_65016


namespace smallest_integer_greater_than_20_l65_65832

noncomputable def smallest_integer_greater_than_A : ℕ :=
  let a (n : ℕ) := 4 * n - 3
  let A := Real.sqrt (a 1580) - 1 / 4
  Nat.ceil A

theorem smallest_integer_greater_than_20 :
  smallest_integer_greater_than_A = 20 :=
sorry

end smallest_integer_greater_than_20_l65_65832


namespace opposite_of_neg_three_l65_65713

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l65_65713


namespace opposite_of_neg_3_is_3_l65_65725

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l65_65725


namespace find_a_l65_65808

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : (min (log_a a 2) (log_a a 4)) * (max (log_a a 2) (log_a a 4)) = 2) : 
  a = (1 / 2) ∨ a = 2 :=
sorry

end find_a_l65_65808


namespace area_of_inscribed_rectangle_l65_65042

theorem area_of_inscribed_rectangle (r l w : ℝ) (h1 : r = 8) (h2 : l / w = 3) (h3 : w = 2 * r) : l * w = 768 :=
by
  sorry

end area_of_inscribed_rectangle_l65_65042


namespace more_ducks_than_four_times_chickens_l65_65119

def number_of_chickens (C : ℕ) : Prop :=
  185 = 150 + C

def number_of_ducks (C : ℕ) (MoreDucks : ℕ) : Prop :=
  150 = 4 * C + MoreDucks

theorem more_ducks_than_four_times_chickens (C MoreDucks : ℕ) (h1 : number_of_chickens C) (h2 : number_of_ducks C MoreDucks) : MoreDucks = 10 := by
  sorry

end more_ducks_than_four_times_chickens_l65_65119


namespace customer_savings_l65_65563

variables (P : ℝ) (reducedPrice negotiatedPrice savings : ℝ)

-- Conditions:
def initialReduction : reducedPrice = 0.95 * P := by sorry
def finalNegotiation : negotiatedPrice = 0.90 * reducedPrice := by sorry
def savingsCalculation : savings = P - negotiatedPrice := by sorry

-- Proof problem:
theorem customer_savings : savings = 0.145 * P :=
by {
  sorry
}

end customer_savings_l65_65563


namespace six_cube_2d_faces_count_l65_65768

open BigOperators

theorem six_cube_2d_faces_count :
    let vertices := 64
    let edges_1d := 192
    let edges_2d := 240
    let small_cubes := 46656
    let faces_per_plane := 36
    let planes_count := 15 * 7^4
    faces_per_plane * planes_count = 1296150 := by
  sorry

end six_cube_2d_faces_count_l65_65768


namespace angle_comparison_l65_65636

theorem angle_comparison :
  let A := 60.4
  let B := 60.24
  let C := 60.24
  A > B ∧ B = C :=
by
  sorry

end angle_comparison_l65_65636


namespace function_has_extremes_l65_65499

variable (a b c : ℝ)

theorem function_has_extremes
  (h₀ : a ≠ 0)
  (h₁ : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧
    ∀ x : ℝ, f (a, b, c) x ≤ f (a, b, c) x₁ ∧
    f (a, b, c) x ≤ f (a, b, c) x₂) :
  (ab > 0) ∧ (b² + 8ac > 0) ∧ (ac < 0) := sorry

def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + b / x + c / x^2

end function_has_extremes_l65_65499


namespace projection_inequality_l65_65883

theorem projection_inequality
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  c ≥ (a + b) / Real.sqrt 2 :=
by
  sorry

end projection_inequality_l65_65883


namespace total_price_eq_2500_l65_65155

theorem total_price_eq_2500 (C P : ℕ)
  (hC : C = 2000)
  (hE : C + 500 + P = 6 * P)
  : C + P = 2500 := 
by
  sorry

end total_price_eq_2500_l65_65155


namespace calculate_product_l65_65197

theorem calculate_product : 6^6 * 3^6 = 34012224 := by
  sorry

end calculate_product_l65_65197


namespace jessica_withdrew_200_l65_65949

noncomputable def initial_balance (final_balance : ℝ) : ℝ :=
  (final_balance * 25 / 18)

noncomputable def withdrawn_amount (initial_balance : ℝ) : ℝ :=
  (initial_balance * 2 / 5)

theorem jessica_withdrew_200 :
  ∀ (final_balance : ℝ), final_balance = 360 → withdrawn_amount (initial_balance final_balance) = 200 :=
by
  intros final_balance h
  rw [h]
  unfold initial_balance withdrawn_amount
  sorry

end jessica_withdrew_200_l65_65949


namespace find_x_l65_65226

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Lean statement asserting that if a is parallel to b for some x, then x = 2
theorem find_x (x : ℝ) (h : parallel a (b x)) : x = 2 := 
by sorry

end find_x_l65_65226


namespace largest_possible_n_base10_l65_65411

theorem largest_possible_n_base10 :
  ∃ (n A B C : ℕ),
    n = 25 * A + 5 * B + C ∧ 
    n = 81 * C + 9 * B + A ∧ 
    A < 5 ∧ B < 5 ∧ C < 5 ∧ 
    n = 69 :=
by {
  sorry
}

end largest_possible_n_base10_l65_65411


namespace opposite_of_neg_3_is_3_l65_65726

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l65_65726


namespace eighth_term_is_79_l65_65145

variable (a d : ℤ)

def fourth_term_condition : Prop := a + 3 * d = 23
def sixth_term_condition : Prop := a + 5 * d = 51

theorem eighth_term_is_79 (h₁ : fourth_term_condition a d) (h₂ : sixth_term_condition a d) : a + 7 * d = 79 :=
sorry

end eighth_term_is_79_l65_65145


namespace sammy_pickles_l65_65961

theorem sammy_pickles 
  (T S R : ℕ) 
  (h1 : T = 2 * S) 
  (h2 : R = 8 * T / 10) 
  (h3 : R = 24) : 
  S = 15 :=
by
  sorry

end sammy_pickles_l65_65961


namespace simple_interest_rate_l65_65884

theorem simple_interest_rate (P R T A : ℝ) (h_double: A = 2 * P) (h_si: A = P + P * R * T / 100) (h_T: T = 5) : R = 20 :=
by
  have h1: A = 2 * P := h_double
  have h2: A = P + P * R * T / 100 := h_si
  have h3: T = 5 := h_T
  sorry

end simple_interest_rate_l65_65884


namespace marika_father_age_twice_l65_65676

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l65_65676


namespace profit_share_difference_l65_65316

theorem profit_share_difference
    (P_A P_B P_C P_D : ℕ) (R_A R_B R_C R_D parts_A parts_B parts_C parts_D : ℕ) (profit_B : ℕ)
    (h1 : P_A = 8000) (h2 : P_B = 10000) (h3 : P_C = 12000) (h4 : P_D = 15000)
    (h5 : R_A = 3) (h6 : R_B = 5) (h7 : R_C = 6) (h8 : R_D = 7)
    (h9: profit_B = 2000) :
    profit_B / R_B = 400 ∧ P_C * R_C / R_B - P_A * R_A / R_B = 1200 :=
by
  sorry

end profit_share_difference_l65_65316


namespace distance_to_line_l65_65763

theorem distance_to_line (a : ℝ) (d : ℝ)
  (h1 : d = 6)
  (h2 : |3 * a + 6| / 5 = d) :
  a = 8 ∨ a = -12 :=
by
  sorry

end distance_to_line_l65_65763


namespace sum_of_f_values_l65_65223

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_of_f_values 
  (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_periodic : ∀ x, f (2 - x) = f x)
  (hf_neg_one : f (-1) = 1) :
  f 1 + f 2 + f 3 + f 4 + (502 * (f 1 + f 2 + f 3 + f 4)) = -1 := 
sorry

end sum_of_f_values_l65_65223


namespace neither_plaid_nor_purple_l65_65701

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end neither_plaid_nor_purple_l65_65701


namespace jim_travel_distance_l65_65820

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end jim_travel_distance_l65_65820


namespace mortgage_loan_l65_65277

theorem mortgage_loan (D : ℝ) (hD : D = 2000000) : 
  ∃ C : ℝ, (C = D + 0.75 * C) ∧ (0.75 * C = 6000000) :=
by
  -- (Optional) Set up the problem with condition D = 2,000,000
  use 8000000  -- From the solution steps, we found C = 8000000
  split
  · -- Show that the equation C = D + 0.75 * C is satisfied
    rw [hD]
    linarith
  · -- Show the mortgage loan amount is 6,000,000
    linarith

end mortgage_loan_l65_65277


namespace concentration_after_removal_l65_65188

/-- 
Given:
1. A container has 27 liters of 40% acidic liquid.
2. 9 liters of water is removed from this container.

Prove that the concentration of the acidic liquid in the container after removal is 60%.
-/
theorem concentration_after_removal :
  let initial_volume := 27
  let initial_concentration := 0.4
  let water_removed := 9
  let pure_acid := initial_concentration * initial_volume
  let new_volume := initial_volume - water_removed
  let final_concentration := (pure_acid / new_volume) * 100
  final_concentration = 60 :=
by {
  sorry
}

end concentration_after_removal_l65_65188


namespace reciprocal_sqrt5_minus_2_l65_65268

theorem reciprocal_sqrt5_minus_2 : 1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := 
by
  sorry

end reciprocal_sqrt5_minus_2_l65_65268


namespace father_twice_marika_age_in_2036_l65_65686

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l65_65686


namespace roller_coaster_costs_4_l65_65242

-- Definitions from conditions
def tickets_initial: ℕ := 5                     -- Jeanne initially has 5 tickets
def tickets_to_buy: ℕ := 8                      -- Jeanne needs to buy 8 more tickets
def total_tickets_needed: ℕ := tickets_initial + tickets_to_buy -- Total tickets needed
def tickets_ferris_wheel: ℕ := 5                -- Ferris wheel costs 5 tickets
def tickets_total_after_ferris_wheel: ℕ := total_tickets_needed - tickets_ferris_wheel -- Remaining tickets after Ferris wheel

-- Definition to be proved (question = answer)
def cost_roller_coaster_bumper_cars: ℕ := tickets_total_after_ferris_wheel / 2 -- Each of roller coaster and bumper cars cost

-- The theorem that corresponds to the solution
theorem roller_coaster_costs_4 :
  cost_roller_coaster_bumper_cars = 4 :=
by
  sorry

end roller_coaster_costs_4_l65_65242


namespace egg_count_l65_65538

theorem egg_count (E : ℕ) (son_daughter_eaten : ℕ) (rhea_husband_eaten : ℕ) (total_eaten : ℕ) (total_eggs : ℕ) (uneaten : ℕ) (trays : ℕ) 
  (H1 : son_daughter_eaten = 2 * 2 * 7)
  (H2 : rhea_husband_eaten = 4 * 2 * 7)
  (H3 : total_eaten = son_daughter_eaten + rhea_husband_eaten)
  (H4 : uneaten = 6)
  (H5 : total_eggs = total_eaten + uneaten)
  (H6 : trays = 2)
  (H7 : total_eggs = E * trays) : 
  E = 45 :=
by
  sorry

end egg_count_l65_65538


namespace aunt_may_morning_milk_l65_65590

-- Defining the known quantities as variables
def evening_milk : ℕ := 380
def sold_milk : ℕ := 612
def leftover_milk : ℕ := 15
def milk_left : ℕ := 148

-- Main statement to be proven
theorem aunt_may_morning_milk (M : ℕ) :
  M + evening_milk + leftover_milk - sold_milk = milk_left → M = 365 := 
by {
  -- Skipping the proof
  sorry
}

end aunt_may_morning_milk_l65_65590


namespace rebecca_soda_left_l65_65401

-- Definitions of the conditions
def total_bottles_purchased : ℕ := 3 * 6
def days_in_four_weeks : ℕ := 4 * 7
def total_half_bottles_drinks : ℕ := days_in_four_weeks
def total_whole_bottles_drinks : ℕ := total_half_bottles_drinks / 2

-- The final statement we aim to prove
theorem rebecca_soda_left : 
  total_bottles_purchased - total_whole_bottles_drinks = 4 := 
by
  -- proof is not required as per the guidelines
  sorry

end rebecca_soda_left_l65_65401


namespace floss_leftover_l65_65787

noncomputable def leftover_floss
    (students : ℕ)
    (floss_per_student : ℚ)
    (floss_per_packet : ℚ) :
    ℚ :=
  let total_needed := students * floss_per_student
  let packets_needed := (total_needed / floss_per_packet).ceil
  let total_floss := packets_needed * floss_per_packet
  total_floss - total_needed

theorem floss_leftover {students : ℕ} {floss_per_student floss_per_packet : ℚ}
    (h_students : students = 20)
    (h_floss_per_student : floss_per_student = 3 / 2)
    (h_floss_per_packet : floss_per_packet = 35) :
    leftover_floss students floss_per_student floss_per_packet = 5 :=
by
  rw [h_students, h_floss_per_student, h_floss_per_packet]
  simp only [leftover_floss]
  norm_num
  sorry

end floss_leftover_l65_65787


namespace gcd_210_162_l65_65023

-- Define the numbers
def a := 210
def b := 162

-- The proposition we need to prove: The GCD of 210 and 162 is 6
theorem gcd_210_162 : Nat.gcd a b = 6 :=
by
  sorry

end gcd_210_162_l65_65023


namespace eq_a2b2_of_given_condition_l65_65369

theorem eq_a2b2_of_given_condition (a b : ℝ) (h : a^4 + b^4 = a^2 - 2 * a^2 * b^2 + b^2 + 6) : a^2 + b^2 = 3 :=
sorry

end eq_a2b2_of_given_condition_l65_65369


namespace smallest_n_divisibility_l65_65560

theorem smallest_n_divisibility :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (m > 0) ∧ (72 ∣ m^2) ∧ (1728 ∣ m^3) → (n ≤ m)) ∧
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) :=
by
  sorry

end smallest_n_divisibility_l65_65560


namespace certain_amount_added_l65_65579

theorem certain_amount_added {x y : ℕ} 
    (h₁ : x = 15) 
    (h₂ : 3 * (2 * x + y) = 105) : y = 5 :=
by
  sorry

end certain_amount_added_l65_65579


namespace solve_exponential_eq_l65_65816

theorem solve_exponential_eq (x : ℝ) : 
  ((5 - 2 * x)^(x + 1) = 1) ↔ (x = -1 ∨ x = 2 ∨ x = 3) := by
  sorry

end solve_exponential_eq_l65_65816


namespace solve_equation_l65_65407

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → 
  x = -4 ∨ x = -2 :=
by 
  sorry

end solve_equation_l65_65407


namespace possible_values_of_quadratic_expression_l65_65419

theorem possible_values_of_quadratic_expression (x : ℝ) (h : 2 < x ∧ x < 3) : 
  20 < x^2 + 5 * x + 6 ∧ x^2 + 5 * x + 6 < 30 :=
by
  sorry

end possible_values_of_quadratic_expression_l65_65419


namespace probability_heads_greater_tails_l65_65576

noncomputable def probability_more_heads_than_tails : ℚ :=
  (4.choose 3) * (1/2)^4 + (4.choose 4) * (1/2)^4

theorem probability_heads_greater_tails : 
  probability_more_heads_than_tails = 5 / 16 :=
by
  sorry

end probability_heads_greater_tails_l65_65576


namespace ellen_smoothies_total_cups_l65_65205

structure SmoothieIngredients where
  strawberries : ℝ
  yogurt       : ℝ
  orange_juice : ℝ
  honey        : ℝ
  chia_seeds   : ℝ
  spinach      : ℝ

def ounces_to_cups (ounces : ℝ) : ℝ := ounces * 0.125
def tablespoons_to_cups (tablespoons : ℝ) : ℝ := tablespoons * 0.0625

noncomputable def total_cups (ing : SmoothieIngredients) : ℝ :=
  ing.strawberries +
  ing.yogurt +
  ing.orange_juice +
  ounces_to_cups (ing.honey) +
  tablespoons_to_cups (ing.chia_seeds) +
  ing.spinach

theorem ellen_smoothies_total_cups :
  total_cups {
    strawberries := 0.2,
    yogurt := 0.1,
    orange_juice := 0.2,
    honey := 1.0,
    chia_seeds := 2.0,
    spinach := 0.5
  } = 1.25 := by
  sorry

end ellen_smoothies_total_cups_l65_65205


namespace find_a1_over_d_l65_65969

variable {a : ℕ → ℝ} (d : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1_over_d 
  (d_ne_zero : d ≠ 0) 
  (seq : arithmetic_sequence a d) 
  (h : a 2021 = a 20 + a 21) : 
  a 1 / d = 1981 :=
by 
  sorry

end find_a1_over_d_l65_65969


namespace number_of_welders_left_l65_65037

-- Define the constants and variables
def welders_total : ℕ := 36
def days_to_complete : ℕ := 5
def rate : ℝ := 1  -- Assume the rate per welder is 1 for simplicity
def total_work : ℝ := welders_total * days_to_complete * rate

def days_after_first : ℕ := 6
def work_done_in_first_day : ℝ := welders_total * 1 * rate
def remaining_work : ℝ := total_work - work_done_in_first_day

-- Define the theorem to solve for the number of welders x that started to work on another project
theorem number_of_welders_left (x : ℕ) : (welders_total - x) * days_after_first * rate = remaining_work → x = 12 := by
  intros h
  sorry

end number_of_welders_left_l65_65037


namespace initial_tomato_count_l65_65313

variable (T : ℝ)
variable (H1 : T - (1 / 4 * T + 20 + 40) = 15)

theorem initial_tomato_count : T = 100 :=
by
  sorry

end initial_tomato_count_l65_65313


namespace nested_sqrt_eq_five_l65_65067

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l65_65067


namespace fraction_of_students_getting_F_l65_65647

theorem fraction_of_students_getting_F
  (students_A students_B students_C students_D passing_fraction : ℚ) 
  (hA : students_A = 1/4)
  (hB : students_B = 1/2)
  (hC : students_C = 1/8)
  (hD : students_D = 1/12)
  (hPassing : passing_fraction = 0.875) :
  (1 - (students_A + students_B + students_C + students_D)) = 1/24 :=
by
  sorry

end fraction_of_students_getting_F_l65_65647


namespace evaluate_f_l65_65386

def f (n : ℕ) : ℕ :=
  if n < 4 then n^2 - 1 else 3*n - 2

theorem evaluate_f (h : f (f (f 2)) = 22) : f (f (f 2)) = 22 :=
by
  -- we state the final result directly
  sorry

end evaluate_f_l65_65386


namespace problem1_problem2_problem3_l65_65201

-- 1. Prove that (3ab³)² = 9a²b⁶
theorem problem1 (a b : ℝ) : (3 * a * b^3)^2 = 9 * a^2 * b^6 :=
by sorry

-- 2. Prove that x ⋅ x³ + x² ⋅ x² = 2x⁴
theorem problem2 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 :=
by sorry

-- 3. Prove that (12x⁴ - 6x³) ÷ 3x² = 4x² - 2x
theorem problem3 (x : ℝ) : (12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x :=
by sorry

end problem1_problem2_problem3_l65_65201


namespace gcd_m_n_l65_65127

-- Define the numbers m and n
def m : ℕ := 555555555
def n : ℕ := 1111111111

-- State the problem: Prove that gcd(m, n) = 1
theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Proof goes here
  sorry

end gcd_m_n_l65_65127


namespace neg_one_exponent_difference_l65_65748

theorem neg_one_exponent_difference : (-1 : ℤ) ^ 2004 - (-1 : ℤ) ^ 2003 = 2 := by
  sorry

end neg_one_exponent_difference_l65_65748


namespace bill_experience_now_l65_65601

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l65_65601


namespace cake_volume_icing_area_sum_l65_65575

-- Define the conditions based on the problem description
def cube_edge_length : ℕ := 4
def volume_of_piece := 16
def icing_area := 12

-- Define the statements to be proven
theorem cake_volume_icing_area_sum : 
  volume_of_piece + icing_area = 28 := 
sorry

end cake_volume_icing_area_sum_l65_65575


namespace candy_bar_cost_l65_65462

theorem candy_bar_cost {initial_money left_money cost_bar : ℕ} 
                        (h_initial : initial_money = 4)
                        (h_left : left_money = 3)
                        (h_cost : cost_bar = initial_money - left_money) :
                        cost_bar = 1 :=
by 
  sorry -- Proof is not required as per the instructions

end candy_bar_cost_l65_65462


namespace marians_new_balance_l65_65671

theorem marians_new_balance :
  ∀ (initial_balance grocery_cost return_amount : ℝ),
    initial_balance = 126 →
    grocery_cost = 60 →
    return_amount = 45 →
    let gas_cost := grocery_cost / 2 in
    let new_balance_before_returns := initial_balance + grocery_cost + gas_cost in
    new_balance_before_returns - return_amount = 171 :=
begin
  intros initial_balance grocery_cost return_amount h_init h_grocery h_return,
  let gas_cost := grocery_cost / 2,
  let new_balance_before_returns := initial_balance + grocery_cost + gas_cost,
  have h_gas : gas_cost = 30,
  { 
    rw h_grocery, 
    norm_num },
  have h_new_balance : new_balance_before_returns = 216,
  {
    rw [h_init, h_grocery, h_gas],
    norm_num },
  rw [h_new_balance, h_return],
  norm_num,
end

end marians_new_balance_l65_65671


namespace at_least_one_tails_up_l65_65938

-- Define propositions p and q
variable (p q : Prop)

-- The theorem statement
theorem at_least_one_tails_up : (¬p ∨ ¬q) ↔ ¬(p ∧ q) := by
  sorry

end at_least_one_tails_up_l65_65938


namespace right_triangle_perimeter_l65_65976

theorem right_triangle_perimeter (a b : ℕ) (h : a^2 + b^2 = 100) (r : ℕ := 1) :
  (a + b + 10) = 24 :=
sorry

end right_triangle_perimeter_l65_65976


namespace max_value_of_x_plus_2y_l65_65667

theorem max_value_of_x_plus_2y {x y : ℝ} (h : |x| + |y| ≤ 1) : (x + 2 * y) ≤ 2 :=
sorry

end max_value_of_x_plus_2y_l65_65667


namespace sum_of_xy_l65_65091

theorem sum_of_xy (x y : ℝ) (h1 : x + 3 * y = 12) (h2 : 3 * x + y = 8) : x + y = 5 := 
by
  sorry

end sum_of_xy_l65_65091


namespace lumber_price_increase_l65_65448

noncomputable def percentage_increase_in_lumber_cost : ℝ :=
  let original_cost_lumber := 450
  let cost_nails := 30
  let cost_fabric := 80
  let original_total_cost := original_cost_lumber + cost_nails + cost_fabric
  let increase_in_total_cost := 97
  let new_total_cost := original_total_cost + increase_in_total_cost
  let unchanged_cost := cost_nails + cost_fabric
  let new_cost_lumber := new_total_cost - unchanged_cost
  let increase_lumber_cost := new_cost_lumber - original_cost_lumber
  (increase_lumber_cost / original_cost_lumber) * 100

theorem lumber_price_increase :
  percentage_increase_in_lumber_cost = 21.56 := by
  sorry

end lumber_price_increase_l65_65448


namespace actual_distance_traveled_l65_65935

variable (t : ℝ) -- let t be the actual time in hours
variable (d : ℝ) -- let d be the actual distance traveled at 12 km/hr

-- Conditions
def condition1 := 20 * t = 12 * t + 30
def condition2 := d = 12 * t

-- The target we want to prove
theorem actual_distance_traveled (t : ℝ) (d : ℝ) (h1 : condition1 t) (h2 : condition2 t d) : 
  d = 45 := by
  sorry

end actual_distance_traveled_l65_65935


namespace nine_by_nine_chessboard_dark_light_excess_l65_65180

theorem nine_by_nine_chessboard_dark_light_excess :
  let board_size := 9
  let odd_row_dark := 5
  let odd_row_light := 4
  let even_row_dark := 4
  let even_row_light := 5
  let num_odd_rows := (board_size + 1) / 2
  let num_even_rows := board_size / 2
  let total_dark_squares := (odd_row_dark * num_odd_rows) + (even_row_dark * num_even_rows)
  let total_light_squares := (odd_row_light * num_odd_rows) + (even_row_light * num_even_rows)
  total_dark_squares - total_light_squares = 1 :=
by {
  sorry
}

end nine_by_nine_chessboard_dark_light_excess_l65_65180


namespace terminal_side_quadrant_l65_65480

theorem terminal_side_quadrant (k : ℤ) : 
  ∃ quadrant, quadrant = 1 ∨ quadrant = 3 ∧
  ∀ (α : ℝ), α = k * 180 + 45 → 
  (quadrant = 1 ∧ (∃ n : ℕ, k = 2 * n)) ∨ (quadrant = 3 ∧ (∃ n : ℕ, k = 2 * n + 1)) :=
by
  sorry

end terminal_side_quadrant_l65_65480


namespace min_socks_to_guarantee_10_pairs_l65_65447

/--
Given a drawer containing 100 red socks, 80 green socks, 60 blue socks, and 40 black socks, 
and socks are selected one at a time without seeing their color. 
The minimum number of socks that must be selected to guarantee at least 10 pairs is 23.
-/
theorem min_socks_to_guarantee_10_pairs 
  (red_socks green_socks blue_socks black_socks : ℕ) 
  (total_pairs : ℕ)
  (h_red : red_socks = 100)
  (h_green : green_socks = 80)
  (h_blue : blue_socks = 60)
  (h_black : black_socks = 40)
  (h_total_pairs : total_pairs = 10) :
  ∃ (n : ℕ), n = 23 := 
sorry

end min_socks_to_guarantee_10_pairs_l65_65447


namespace smallest_n_mod_l65_65432

theorem smallest_n_mod : ∃ n : ℕ, 5 * n ≡ 2024 [MOD 26] ∧ n > 0 ∧ ∀ m : ℕ, (5 * m ≡ 2024 [MOD 26] ∧ m > 0) → n ≤ m :=
  sorry

end smallest_n_mod_l65_65432


namespace lindsey_squat_weight_l65_65130

theorem lindsey_squat_weight :
  let num_bands := 2 in
  let resistance_per_band := 5 in
  let total_resistance := num_bands * resistance_per_band in
  let dumbbell_weight := 10 in
  total_resistance + dumbbell_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l65_65130


namespace laptop_cost_l65_65144

theorem laptop_cost
  (C : ℝ) (down_payment := 0.2 * C + 20) (installments_paid := 65 * 4) (balance_after_4_months := 520)
  (h : C - (down_payment + installments_paid) = balance_after_4_months) :
  C = 1000 :=
by
  sorry

end laptop_cost_l65_65144


namespace conditions_for_local_extrema_l65_65490

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * log x + b / x + c / (x^2)

theorem conditions_for_local_extrema
  (a b c : ℝ) (ha : a ≠ 0) (D : ℝ → ℝ) (hD : ∀ x, D x = deriv (f a b c) x) :
  (∀ x > 0, D x = (a * x^2 - b * x - 2 * c) / x^3) →
  (∃ x y > 0, D x = 0 ∧ D y = 0 ∧ x ≠ y) ↔
    (a * b > 0 ∧ a * c < 0 ∧ b^2 + 8 * a * c > 0) :=
sorry

end conditions_for_local_extrema_l65_65490


namespace boat_crossing_time_l65_65423

theorem boat_crossing_time :
  ∀ (width_of_river speed_of_current speed_of_boat : ℝ),
  width_of_river = 1.5 →
  speed_of_current = 8 →
  speed_of_boat = 10 →
  (width_of_river / (Real.sqrt (speed_of_boat ^ 2 - speed_of_current ^ 2)) * 60) = 15 :=
by
  intros width_of_river speed_of_current speed_of_boat h1 h2 h3
  sorry

end boat_crossing_time_l65_65423


namespace find_y_l65_65370

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) : y = x :=
sorry

end find_y_l65_65370


namespace find_cost_price_of_radio_l65_65583

def cost_price_of_radio
  (profit_percent: ℝ) (overhead_expenses: ℝ) (selling_price: ℝ) (C: ℝ) : Prop :=
  profit_percent = ((selling_price - (C + overhead_expenses)) / C) * 100

theorem find_cost_price_of_radio :
  cost_price_of_radio 21.457489878542503 15 300 234.65 :=
by
  sorry

end find_cost_price_of_radio_l65_65583


namespace negation_of_existence_statement_l65_65361

theorem negation_of_existence_statement :
  (¬ ∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_existence_statement_l65_65361


namespace opposite_of_neg_three_l65_65714

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l65_65714


namespace quadratic_decomposition_l65_65980

theorem quadratic_decomposition (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) → a + b + c = 228 :=
sorry

end quadratic_decomposition_l65_65980


namespace div_eq_210_over_79_l65_65779

def a_at_b (a b : ℕ) : ℤ := a^2 * b - a * (b^2)
def a_hash_b (a b : ℕ) : ℤ := a^2 + b^2 - a * b

theorem div_eq_210_over_79 : (a_at_b 10 3) / (a_hash_b 10 3) = 210 / 79 :=
by
  -- This is a placeholder and needs to be filled with the actual proof.
  sorry

end div_eq_210_over_79_l65_65779


namespace binary_11011011_to_base4_is_3123_l65_65461

def binary_to_base4 (b : Nat) : Nat :=
  -- Function to convert binary number to base 4
  -- This will skip implementation details
  sorry

theorem binary_11011011_to_base4_is_3123 :
  binary_to_base4 0b11011011 = 0x3123 := 
sorry

end binary_11011011_to_base4_is_3123_l65_65461


namespace domain_of_log_base_half_l65_65008

noncomputable def domain_log_base_half : Set ℝ := { x : ℝ | x > 5 }

theorem domain_of_log_base_half :
  (∀ x : ℝ, x > 5 ↔ x - 5 > 0) →
  (domain_log_base_half = { x : ℝ | x - 5 > 0 }) :=
by
  sorry

end domain_of_log_base_half_l65_65008


namespace smallest_base_l65_65027

theorem smallest_base (k b : ℕ) (h_k : k = 6) : 64 ^ k > b ^ 16 ↔ b < 5 :=
by
  have h1 : 64 ^ k = 2 ^ (6 * k) := by sorry
  have h2 : 2 ^ (6 * k) > b ^ 16 := by sorry
  exact sorry

end smallest_base_l65_65027


namespace second_to_last_digit_of_n_squared_plus_2n_l65_65267
open Nat

theorem second_to_last_digit_of_n_squared_plus_2n (n : ℕ) (h : (n^2 + 2 * n) % 10 = 4) : ((n^2 + 2 * n) / 10) % 10 = 2 :=
  sorry

end second_to_last_digit_of_n_squared_plus_2n_l65_65267


namespace quadratic_polynomial_AT_BT_l65_65044

theorem quadratic_polynomial_AT_BT (p s : ℝ) :
  ∃ (AT BT : ℝ), (AT + BT = p + 3) ∧ (AT * BT = s^2) ∧ (∀ (x : ℝ), (x^2 - (p+3) * x + s^2) = (x - AT) * (x - BT)) := 
sorry

end quadratic_polynomial_AT_BT_l65_65044


namespace show_R_r_eq_l65_65403

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

end show_R_r_eq_l65_65403


namespace amount_returned_l65_65247

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l65_65247


namespace sum_n_k_l65_65264

theorem sum_n_k (n k : ℕ) (h₁ : (x+1)^n = 2 * x^k + 3 * x^(k+1) + 4 * x^(k+2)) (h₂ : 3 * k + 3 = 2 * n - 2 * k)
  (h₃ : 4 * k + 8 = 3 * n - 3 * k - 3) : n + k = 47 := 
sorry

end sum_n_k_l65_65264


namespace jim_travel_distance_l65_65819

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end jim_travel_distance_l65_65819


namespace sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l65_65458

-- Problem (1)
theorem sector_angle_given_circumference_and_area :
  (∀ (r l : ℝ), 2 * r + l = 10 ∧ (1 / 2) * l * r = 4 → l / r = (1 / 2)) := by
  sorry

-- Problem (2)
theorem max_sector_area_given_circumference :
  (∀ (r l : ℝ), 2 * r + l = 40 → (r = 10 ∧ l = 20 ∧ (1 / 2) * l * r = 100 ∧ l / r = 2)) := by
  sorry

end sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l65_65458


namespace function_max_min_l65_65488

theorem function_max_min (a b c : ℝ) (h_a : a ≠ 0) (h_sum_pos : a * b > 0) (h_discriminant_pos : b^2 + 8 * a * c > 0) (h_product_neg : a * c < 0) : 
  (∀ x > 0, ∃ x1 x2 > 0, x1 + x2 = b / a ∧ x1 * x2 = -2 * c / a) := 
sorry

end function_max_min_l65_65488


namespace triangle_inequality_l65_65473

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b - c) * (a - b + c) * (-a + b + c) ≤ a * b * c := 
sorry

end triangle_inequality_l65_65473


namespace max_n_base_10_l65_65413

theorem max_n_base_10:
  ∃ (A B C n: ℕ), (A < 5 ∧ B < 5 ∧ C < 5) ∧
                 (n = 25 * A + 5 * B + C) ∧ (n = 81 * C + 9 * B + A) ∧ 
                 (∀ (A' B' C' n': ℕ), 
                 (A' < 5 ∧ B' < 5 ∧ C' < 5) ∧ (n' = 25 * A' + 5 * B' + C') ∧ 
                 (n' = 81 * C' + 9 * B' + A') → n' ≤ n) →
  n = 111 :=
by {
    sorry
}

end max_n_base_10_l65_65413


namespace simplify_expression_l65_65855

theorem simplify_expression (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) * (x - 2) + (x - 2) * (2 * x^2 - 3 * x + 9) - (4 * x - 7) * (x - 2) * (x - 3) 
  = x^3 + x^2 + 12 * x - 36 := 
by
  sorry

end simplify_expression_l65_65855


namespace function_has_extremes_l65_65498

variable (a b c : ℝ)

theorem function_has_extremes
  (h₀ : a ≠ 0)
  (h₁ : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧
    ∀ x : ℝ, f (a, b, c) x ≤ f (a, b, c) x₁ ∧
    f (a, b, c) x ≤ f (a, b, c) x₂) :
  (ab > 0) ∧ (b² + 8ac > 0) ∧ (ac < 0) := sorry

def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + b / x + c / x^2

end function_has_extremes_l65_65498


namespace surface_area_ratio_volume_ratio_l65_65776

-- Given conditions
def tetrahedron_surface_area (S : ℝ) : ℝ := 4 * S
def tetrahedron_volume (V : ℝ) : ℝ := 27 * V
def polyhedron_G_surface_area (S : ℝ) : ℝ := 28 * S
def polyhedron_G_volume (V : ℝ) : ℝ := 23 * V

-- Statements to prove
theorem surface_area_ratio (S : ℝ) (h1 : S > 0) :
  tetrahedron_surface_area S / polyhedron_G_surface_area S = 9 / 7 := by
  simp [tetrahedron_surface_area, polyhedron_G_surface_area]
  sorry

theorem volume_ratio (V : ℝ) (h1 : V > 0) :
  tetrahedron_volume V / polyhedron_G_volume V = 27 / 23 := by
  simp [tetrahedron_volume, polyhedron_G_volume]
  sorry

end surface_area_ratio_volume_ratio_l65_65776


namespace smallest_d_for_inverse_l65_65524

noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse : ∃ d : ℝ, (∀ x1 x2, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = 3 := 
sorry

end smallest_d_for_inverse_l65_65524


namespace find_c_minus_a_l65_65486

theorem find_c_minus_a (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 :=
sorry

end find_c_minus_a_l65_65486


namespace number_of_arrangements_l65_65697

theorem number_of_arrangements (n : ℕ) (h : n = 7) :
  ∃ (arrangements : ℕ), arrangements = 20 :=
by
  sorry

end number_of_arrangements_l65_65697


namespace monotonic_intervals_slope_tangent_line_inequality_condition_l65_65358

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a + 2) * x^2 + 2 * a * x
noncomputable def g (a x : ℝ) : ℝ := (1/2) * (a - 5) * x^2

theorem monotonic_intervals (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  ((∀ x, x < 2 → deriv (f a) x > 0) ∧ (∀ x, x > a → deriv (f a) x > 0)) ∧
  (∀ x, 2 < x ∧ x < a → deriv (f a) x < 0) :=
sorry

theorem slope_tangent_line (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  (∀ x_0 y_0 k, y_0 = f a x_0 ∧ k = deriv (f a) x_0 ∧ k ≥ -(25/4) →
    4 ≤ a ∧ a ≤ 7) :=
sorry

theorem inequality_condition (a : ℝ) (h : a ≥ 4) :
  (∀ x_1 x_2, 3 ≤ x_1 ∧ x_1 < x_2 ∧ x_2 ≤ 4 →
    abs (f a x_1 - f a x_2) > abs (g a x_1 - g a x_2)) →
  (14/3 ≤ a ∧ a ≤ 6) :=
sorry

end monotonic_intervals_slope_tangent_line_inequality_condition_l65_65358


namespace min_5a2_plus_6a3_l65_65126

theorem min_5a2_plus_6a3 (a_1 a_2 a_3 : ℝ) (r : ℝ)
  (h1 : a_1 = 2)
  (h2 : a_2 = a_1 * r)
  (h3 : a_3 = a_1 * r^2) :
  5 * a_2 + 6 * a_3 ≥ -25 / 12 :=
by
  sorry

end min_5a2_plus_6a3_l65_65126


namespace Jed_older_than_Matt_l65_65380

-- Definitions of ages and conditions
def Jed_current_age : ℕ := sorry
def Matt_current_age : ℕ := sorry
axiom condition1 : Jed_current_age + 10 = 25
axiom condition2 : Jed_current_age + Matt_current_age = 20

-- Proof statement
theorem Jed_older_than_Matt : Jed_current_age - Matt_current_age = 10 :=
by
  sorry

end Jed_older_than_Matt_l65_65380


namespace original_number_l65_65891

theorem original_number (x : ℝ) (h : 1.47 * x = 1214.33) : x = 826.14 :=
sorry

end original_number_l65_65891


namespace candy_not_chocolate_l65_65519

theorem candy_not_chocolate (candy_total : ℕ) (bags : ℕ) (choc_heart_bags : ℕ) (choc_kiss_bags : ℕ) : 
  candy_total = 63 ∧ bags = 9 ∧ choc_heart_bags = 2 ∧ choc_kiss_bags = 3 → 
  (candy_total - (choc_heart_bags * (candy_total / bags) + choc_kiss_bags * (candy_total / bags))) = 28 :=
by
  intros h
  sorry

end candy_not_chocolate_l65_65519


namespace sin_inequality_l65_65396

open Real

theorem sin_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (haq : a < π/4) (hb : 0 < b) (hbq : b < π/4) (hn : 0 < n) :
  (sin a)^n + (sin b)^n / (sin a + sin b)^n ≥ (sin (2 * a))^n + (sin (2 * b))^n / (sin (2 * a) + sin (2* b))^n :=
sorry

end sin_inequality_l65_65396


namespace greater_number_is_64_l65_65867

theorem greater_number_is_64
  (x y : ℕ)
  (h1 : x * y = 2048)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) :
  x = 64 :=
by
  -- proof to be filled in
  sorry

end greater_number_is_64_l65_65867


namespace geometric_sequence_common_ratio_range_l65_65514

theorem geometric_sequence_common_ratio_range (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 < 0) 
  (h2 : ∀ n : ℕ, 0 < n → a n < a (n + 1))
  (hq : ∀ n : ℕ, a (n + 1) = a n * q) :
  0 < q ∧ q < 1 :=
sorry

end geometric_sequence_common_ratio_range_l65_65514


namespace binomial_divisibility_l65_65841

-- Define the function α_n (the number of 1's in the binary representation of n)
def α (n : ℕ) : ℕ := (((Nat.bits n).filter (λ b => b = true)).length)

-- Define the core theorem
theorem binomial_divisibility (n r : ℕ) (hn : 0 < n) (hr : 0 < r) :
  2^(2*n - α n) ∣ (Finset.range (2*n + 1)).sum (λ k : ℕ, Nat.choose (2*n) (n + k - n) * (k - n)^(2*r)) :=
by
  sorry

end binomial_divisibility_l65_65841


namespace chairlift_halfway_l65_65135

theorem chairlift_halfway (total_chairs current_chair halfway_chair : ℕ) 
  (h_total_chairs : total_chairs = 96)
  (h_current_chair : current_chair = 66) : halfway_chair = 18 :=
sorry

end chairlift_halfway_l65_65135


namespace dogs_not_eating_any_foods_l65_65939

theorem dogs_not_eating_any_foods :
  let total_dogs := 80
  let dogs_like_watermelon := 18
  let dogs_like_salmon := 58
  let dogs_like_both_salmon_watermelon := 7
  let dogs_like_chicken := 16
  let dogs_like_both_chicken_salmon := 6
  let dogs_like_both_chicken_watermelon := 4
  let dogs_like_all_three := 3
  let dogs_like_any_food := dogs_like_watermelon + dogs_like_salmon + dogs_like_chicken - 
                            dogs_like_both_salmon_watermelon - dogs_like_both_chicken_salmon - 
                            dogs_like_both_chicken_watermelon + dogs_like_all_three
  total_dogs - dogs_like_any_food = 2 := by
  sorry

end dogs_not_eating_any_foods_l65_65939


namespace carl_speed_l65_65943

theorem carl_speed 
  (time : ℝ) (distance : ℝ) 
  (h_time : time = 5) 
  (h_distance : distance = 10) 
  : (distance / time) = 2 :=
by
  rw [h_time, h_distance]
  sorry

end carl_speed_l65_65943


namespace annie_ride_miles_l65_65673

noncomputable def annie_ride_distance : ℕ := 14

theorem annie_ride_miles
  (mike_base_rate : ℝ := 2.5)
  (mike_per_mile_rate : ℝ := 0.25)
  (mike_miles : ℕ := 34)
  (annie_base_rate : ℝ := 2.5)
  (annie_bridge_toll : ℝ := 5.0)
  (annie_per_mile_rate : ℝ := 0.25)
  (annie_miles : ℕ := annie_ride_distance)
  (mike_cost : ℝ := mike_base_rate + mike_per_mile_rate * mike_miles)
  (annie_cost : ℝ := annie_base_rate + annie_bridge_toll + annie_per_mile_rate * annie_miles) :
  mike_cost = annie_cost → annie_miles = 14 := 
by
  sorry

end annie_ride_miles_l65_65673


namespace compute_fraction_l65_65252

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem compute_fraction : (f (g (f 1))) / (g (f (g 1))) = 6801 / 281 := 
by 
  sorry

end compute_fraction_l65_65252


namespace raghu_investment_l65_65991

-- Define the conditions as Lean definitions
def invest_raghu : Real := sorry
def invest_trishul := 0.90 * invest_raghu
def invest_vishal := 1.10 * invest_trishul
def invest_chandni := 1.15 * invest_vishal
def total_investment := invest_raghu + invest_trishul + invest_vishal + invest_chandni

-- State the proof problem
theorem raghu_investment (h : total_investment = 10700) : invest_raghu = 2656.25 :=
by
  sorry

end raghu_investment_l65_65991


namespace original_number_is_1212_or_2121_l65_65266

theorem original_number_is_1212_or_2121 (x y z t : ℕ) (h₁ : t ≠ 0)
  (h₂ : 1000 * x + 100 * y + 10 * z + t + 1000 * t + 100 * x + 10 * y + z = 3333) : 
  (1000 * x + 100 * y + 10 * z + t = 1212) ∨ (1000 * x + 100 * y + 10 * z + t = 2121) :=
sorry

end original_number_is_1212_or_2121_l65_65266


namespace skyscraper_anniversary_l65_65109

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l65_65109


namespace heat_of_neutralization_combination_l65_65118

-- Define instruments
inductive Instrument
| Balance
| MeasuringCylinder
| Beaker
| Burette
| Thermometer
| TestTube
| AlcoholLamp

def correct_combination : List Instrument :=
  [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer]

theorem heat_of_neutralization_combination :
  correct_combination = [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer] :=
sorry

end heat_of_neutralization_combination_l65_65118


namespace pictures_left_l65_65030

def initial_zoo_pics : ℕ := 49
def initial_museum_pics : ℕ := 8
def deleted_pics : ℕ := 38

theorem pictures_left (total_pics : ℕ) :
  total_pics = initial_zoo_pics + initial_museum_pics →
  total_pics - deleted_pics = 19 :=
by
  intro h1
  rw [h1]
  sorry

end pictures_left_l65_65030


namespace graph_pairwise_connected_by_green_or_third_point_l65_65755

open Classical

universe u

variable {V : Type u} [Fintype V]

def is_colored_graph (G : SimpleGraph V) :=
  ∀ (u v : V), u ≠ v → G.adj u v

def only_two_pts_not_connected_by_red_path (G : SimpleGraph V) (red : V → V → Prop) (A B : V) :=
  ¬ (∃ (p : List V), p ≠ [] ∧ p.headI = A ∧ p.getLast sorry = B ∧ ∀ i ∈ p.zip p.tail, red i.1 i.2)

theorem graph_pairwise_connected_by_green_or_third_point (G : SimpleGraph V) 
    (red green : V → V → Prop) (A B : V)
    (h_colored: is_colored_graph G)
    (h_edges_colored: ∀ (u v : V), u ≠ v → red u v ∨ green u v)
    (h_not_red_path: only_two_pts_not_connected_by_red_path G red A B) :
    ∀ (X Y : V), X ≠ Y → green X Y ∨ (∃ (Z : V), green X Z ∧ green Y Z) :=
by
  sorry

end graph_pairwise_connected_by_green_or_third_point_l65_65755


namespace negation_universal_proposition_l65_65710

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, |x| + x^4 ≥ 0) ↔ ∃ x₀ : ℝ, |x₀| + x₀^4 < 0 :=
by
  sorry

end negation_universal_proposition_l65_65710


namespace expansion_abs_coeff_sum_l65_65080

theorem expansion_abs_coeff_sum :
  ∀ (a a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - x)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 32 :=
by
  sorry

end expansion_abs_coeff_sum_l65_65080


namespace mortgage_loan_amount_l65_65276

theorem mortgage_loan_amount (C : ℝ) (hC : C = 8000000) : 0.75 * C = 6000000 :=
by
  sorry

end mortgage_loan_amount_l65_65276


namespace area_of_path_correct_l65_65310

noncomputable def area_of_path (length_field : ℝ) (width_field : ℝ) (path_width : ℝ) : ℝ :=
  let length_total := length_field + 2 * path_width
  let width_total := width_field + 2 * path_width
  let area_total := length_total * width_total
  let area_field := length_field * width_field
  area_total - area_field

theorem area_of_path_correct :
  area_of_path 75 55 3.5 = 959 := 
by
  sorry

end area_of_path_correct_l65_65310


namespace marika_father_age_twice_l65_65678

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l65_65678


namespace bill_experience_l65_65594

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l65_65594


namespace owners_riding_to_total_ratio_l65_65292

theorem owners_riding_to_total_ratio (R W : ℕ) (h1 : 4 * R + 6 * W = 90) (h2 : R + W = 18) : R / (R + W) = 1 / 2 :=
by
  sorry

end owners_riding_to_total_ratio_l65_65292


namespace rope_segment_length_l65_65767

theorem rope_segment_length (L : ℕ) (half_fold_times : ℕ) (dm_to_cm : ℕ → ℕ) 
  (hL : L = 8) (h_half_fold_times : half_fold_times = 2) (h_dm_to_cm : dm_to_cm 1 = 10)
  : dm_to_cm (L / 2 ^ half_fold_times) = 20 := 
by 
  sorry

end rope_segment_length_l65_65767


namespace average_of_roots_l65_65624

theorem average_of_roots (a b c : ℚ) (h : a ≠ 0) (h_eq : 3 * a = 9 ∧ 4 * a = 36 / 9 ∧ -8 * a = -24) : 
  average_of_roots (Poly.of_coeffs![3, 4, -8]) = -2/3 :=
sorry

end average_of_roots_l65_65624


namespace find_quotient_l65_65975

-- Define the problem variables and conditions
def larger_number : ℕ := 1620
def smaller_number : ℕ := larger_number - 1365
def remainder : ℕ := 15

-- Define the proof problem
theorem find_quotient :
  larger_number = smaller_number * 6 + remainder :=
sorry

end find_quotient_l65_65975


namespace cricket_team_members_count_l65_65869

theorem cricket_team_members_count 
(captain_age : ℕ) (wk_keeper_age : ℕ) (whole_team_avg_age : ℕ)
(remaining_players_avg_age : ℕ) (n : ℕ) 
(h1 : captain_age = 28)
(h2 : wk_keeper_age = captain_age + 3)
(h3 : whole_team_avg_age = 25)
(h4 : remaining_players_avg_age = 24)
(h5 : (n * whole_team_avg_age - (captain_age + wk_keeper_age)) / (n - 2) = remaining_players_avg_age) :
n = 11 := 
sorry

end cricket_team_members_count_l65_65869


namespace min_stamps_for_target_value_l65_65388

theorem min_stamps_for_target_value :
  ∃ (c f : ℕ), 5 * c + 7 * f = 50 ∧ ∀ (c' f' : ℕ), 5 * c' + 7 * f' = 50 → c + f ≤ c' + f' → c + f = 8 :=
by
  sorry

end min_stamps_for_target_value_l65_65388


namespace circle_properties_l65_65090

noncomputable def circle_eq (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0
noncomputable def line_eq (x y : ℝ) := x + 2*y - 4 = 0
noncomputable def perpendicular (x1 y1 x2 y2 : ℝ) := 
  (x1 * x2 + y1 * y2 = 0)

theorem circle_properties (m : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, circle_eq x y m) →
  (∀ x, line_eq x (y1 + y2)) →
  perpendicular (4 - 2*y1) y1 (4 - 2*y2) y2 →
  m = 8 / 5 ∧ 
  (∀ x y, (x^2 + y^2 - (8 / 5) * x - (16 / 5) * y = 0) ↔ 
           (x - (4 - 2*(16/5))) * (x - (4 - 2*(16/5))) + (y - (16/5)) * (y - (16/5)) = 5 - (8/5)) :=
sorry

end circle_properties_l65_65090


namespace problem_discussion_organization_l65_65053

theorem problem_discussion_organization 
    (students : Fin 20 → Finset (Fin 20))
    (problems : Fin 20 → Finset (Fin 20))
    (h1 : ∀ s, (students s).card = 2)
    (h2 : ∀ p, (problems p).card = 2)
    (h3 : ∀ s p, s ∈ problems p ↔ p ∈ students s) : 
    ∃ (discussion : Fin 20 → Fin 20), 
        (∀ s, discussion s ∈ students s) ∧ 
        (Finset.univ.image discussion).card = 20 :=
by
  -- proof goes here
  sorry

end problem_discussion_organization_l65_65053


namespace grain_spilled_l65_65312

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled : (original_grain - remaining_grain) = 49952 :=
by
  -- Proof goes here
  sorry

end grain_spilled_l65_65312


namespace value_of_inverse_product_l65_65225

theorem value_of_inverse_product (x y : ℝ) (h1 : x * y > 0) (h2 : 1/x + 1/y = 15) (h3 : (x + y) / 5 = 0.6) :
  1 / (x * y) = 5 :=
by 
  sorry

end value_of_inverse_product_l65_65225


namespace range_of_x_for_sqrt_l65_65834

-- Define the condition under which the expression inside the square root is non-negative.
def sqrt_condition (x : ℝ) : Prop :=
  x - 7 ≥ 0

-- Main theorem to prove the range of values for x
theorem range_of_x_for_sqrt (x : ℝ) : sqrt_condition x ↔ x ≥ 7 :=
by
  -- Proof steps go here (omitted as per instructions)
  sorry

end range_of_x_for_sqrt_l65_65834


namespace pascal_triangle_prob_1_l65_65319

theorem pascal_triangle_prob_1 : 
  let total_elements := (20 * 21) / 2,
      num_ones := 19 * 2 + 1
  in (num_ones / total_elements = 39 / 210) := by
  sorry

end pascal_triangle_prob_1_l65_65319


namespace total_animals_received_l65_65011

-- Define the conditions
def cats : ℕ := 40
def additionalCats : ℕ := 20
def dogs : ℕ := cats - additionalCats

-- Prove the total number of animals received
theorem total_animals_received : (cats + dogs) = 60 := by
  -- The proof itself is not required in this task
  sorry

end total_animals_received_l65_65011


namespace max_ratio_lemma_l65_65220

theorem max_ratio_lemma (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn : ∀ n, S n = (n + 1) / 2 * a n)
  (hSn_minus_one : ∀ n, S (n - 1) = n / 2 * a (n - 1)) :
  ∀ n > 1, (a n / a (n - 1) ≤ 2) ∧ (a 2 / a 1 = 2) := sorry

end max_ratio_lemma_l65_65220


namespace total_cost_proof_l65_65417

noncomputable def cost_of_4kg_mangos_3kg_rice_5kg_flour (M R F : ℝ) : ℝ :=
  4 * M + 3 * R + 5 * F

theorem total_cost_proof
  (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 22) :
  cost_of_4kg_mangos_3kg_rice_5kg_flour M R F = 941.6 :=
  sorry

end total_cost_proof_l65_65417


namespace marika_father_age_twice_l65_65677

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l65_65677


namespace brenda_more_than_jeff_l65_65038

def emma_amount : ℕ := 8
def daya_amount : ℕ := emma_amount + (emma_amount * 25 / 100)
def jeff_amount : ℕ := (2 / 5) * daya_amount
def brenda_amount : ℕ := 8

theorem brenda_more_than_jeff :
  brenda_amount - jeff_amount = 4 :=
sorry

end brenda_more_than_jeff_l65_65038


namespace calculate_first_year_sample_l65_65771

noncomputable def stratified_sampling : ℕ :=
  let total_sample_size := 300
  let first_grade_ratio := 4
  let second_grade_ratio := 5
  let third_grade_ratio := 5
  let fourth_grade_ratio := 6
  let total_ratio := first_grade_ratio + second_grade_ratio + third_grade_ratio + fourth_grade_ratio
  let first_grade_proportion := first_grade_ratio / total_ratio
  300 * first_grade_proportion

theorem calculate_first_year_sample :
  stratified_sampling = 60 :=
by sorry

end calculate_first_year_sample_l65_65771


namespace potatoes_cost_l65_65944

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end potatoes_cost_l65_65944


namespace problem1_problem2_problem3_problem4_l65_65604

theorem problem1 : 9 - 5 - (-4) + 2 = 10 := by
  sorry

theorem problem2 : (- (3 / 4) + 7 / 12 - 5 / 9) / (-(1 / 36)) = 26 := by
  sorry

theorem problem3 : -2^4 - ((-5) + 1 / 2) * (4 / 11) + (-2)^3 / (abs (-3^2 + 1)) = -15 := by
  sorry

theorem problem4 : (100 - 1 / 72) * (-36) = -(3600) + (1 / 2) := by
  sorry

end problem1_problem2_problem3_problem4_l65_65604


namespace price_sugar_salt_l65_65151

/-- The price of two kilograms of sugar and five kilograms of salt is $5.50. If a kilogram of sugar 
    costs $1.50, then how much is the price of three kilograms of sugar and some kilograms of salt, 
    if the total price is $5? -/
theorem price_sugar_salt 
  (price_sugar_per_kg : ℝ)
  (price_total_2kg_sugar_5kg_salt : ℝ)
  (total_price : ℝ) :
  price_sugar_per_kg = 1.50 →
  price_total_2kg_sugar_5kg_salt = 5.50 →
  total_price = 5 →
  2 * price_sugar_per_kg + 5 * (price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5 = 5.50 →
  3 * price_sugar_per_kg + (total_price - 3 * price_sugar_per_kg) / ((price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5) = 1 →
  true :=
by
  sorry

end price_sugar_salt_l65_65151


namespace batteries_difference_is_correct_l65_65553

-- Define the number of batteries used in each item
def flashlights_batteries : ℝ := 3.5
def toys_batteries : ℝ := 15.75
def remote_controllers_batteries : ℝ := 7.25
def wall_clock_batteries : ℝ := 4.8
def wireless_mouse_batteries : ℝ := 3.4

-- Define the combined total of batteries used in the other items
def combined_total : ℝ := flashlights_batteries + remote_controllers_batteries + wall_clock_batteries + wireless_mouse_batteries

-- Define the difference between the total number of batteries used in toys and the combined total of other items
def batteries_difference : ℝ := toys_batteries - combined_total

theorem batteries_difference_is_correct : batteries_difference = -3.2 :=
by
  sorry

end batteries_difference_is_correct_l65_65553


namespace inequality_of_positive_numbers_l65_65527

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_of_positive_numbers_l65_65527


namespace no_nonzero_real_solutions_l65_65920

theorem no_nonzero_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ¬ (2 / x + 3 / y = 1 / (x + y)) :=
by sorry

end no_nonzero_real_solutions_l65_65920


namespace isosceles_triangle_perimeter_l65_65542

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Define the roots based on factorization of the given equation
def root1 := 2
def root2 := 4

-- Define the perimeter of the isosceles triangle given the roots
def triangle_perimeter := root2 + root2 + root1

-- Prove that the perimeter of the isosceles triangle is 10
theorem isosceles_triangle_perimeter : triangle_perimeter = 10 :=
by
  -- We need to verify the solution without providing the steps explicitly
  sorry

end isosceles_triangle_perimeter_l65_65542


namespace unknown_number_is_105_l65_65164

theorem unknown_number_is_105 :
  ∃ x : ℝ, x^2 + 94^2 = 19872 ∧ x = 105 :=
by
  sorry

end unknown_number_is_105_l65_65164


namespace undefined_integer_count_l65_65079

noncomputable def expression (x : ℤ) : ℚ := (x^2 - 16) / ((x^2 - x - 6) * (x - 4))

theorem undefined_integer_count : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x^2 - x - 6) * (x - 4) = 0) ∧ S.card = 3 :=
  sorry

end undefined_integer_count_l65_65079


namespace Toms_dog_age_in_6_years_l65_65021

-- Let's define the conditions
variables (B D : ℕ)
axiom h1 : B = 4 * D
axiom h2 : B + 6 = 30

-- Now we state the theorem
theorem Toms_dog_age_in_6_years :
  D + 6 = 12 :=
by
  sorry

end Toms_dog_age_in_6_years_l65_65021


namespace intersection_of_M_and_N_l65_65810

-- Define sets M and N
def M : Set ℕ := {0, 2, 3, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

-- State the problem as a theorem
theorem intersection_of_M_and_N : (M ∩ N) = {0, 4} :=
by
    sorry

end intersection_of_M_and_N_l65_65810


namespace find_y_l65_65742

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end find_y_l65_65742


namespace triangle_area_arithmetic_sequence_l65_65902

theorem triangle_area_arithmetic_sequence :
  ∃ (S_1 S_2 S_3 S_4 S_5 : ℝ) (d : ℝ),
  S_1 + S_2 + S_3 + S_4 + S_5 = 420 ∧
  S_2 = S_1 + d ∧
  S_3 = S_1 + 2 * d ∧
  S_4 = S_1 + 3 * d ∧
  S_5 = S_1 + 4 * d ∧
  S_5 = 112 :=
by
  sorry

end triangle_area_arithmetic_sequence_l65_65902


namespace total_students_l65_65169

theorem total_students (T : ℕ) (h1 : (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T) : 
  T = 400 :=
sorry

end total_students_l65_65169


namespace nested_sqrt_solution_l65_65068

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l65_65068


namespace larry_win_probability_correct_l65_65661

/-- Define the probabilities of knocking off the bottle for both players in the first four turns. -/
structure GameProb (turns : ℕ) :=
  (larry_prob : ℚ)
  (julius_prob : ℚ)

/-- Define the probabilities of knocking off the bottle for both players from the fifth turn onwards. -/
def subsequent_turns_prob : ℚ := 1 / 2
/-- Initial probabilities for the first four turns -/
def initial_prob : GameProb 4 := { larry_prob := 2 / 3, julius_prob := 1 / 3 }
/-- The probability that Larry wins the game -/
def larry_wins (prob : GameProb 4) (subsequent_prob : ℚ) : ℚ :=
  -- Calculation logic goes here resulting in the final probability
  379 / 648

theorem larry_win_probability_correct :
  larry_wins initial_prob subsequent_turns_prob = 379 / 648 :=
sorry

end larry_win_probability_correct_l65_65661


namespace price_of_one_shirt_l65_65274

variable (P : ℝ)

-- Conditions
def cost_two_shirts := 1.5 * P
def cost_three_shirts := 1.9 * P 
def full_price_three_shirts := 3 * P
def savings := full_price_three_shirts - cost_three_shirts

-- Correct answer
theorem price_of_one_shirt (hs : savings = 11) : P = 10 :=
by
  sorry

end price_of_one_shirt_l65_65274


namespace prime_root_condition_l65_65334

theorem prime_root_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℤ, x ≠ y ∧ (x^2 + 2 * p * x - 240 * p = 0) ∧ (y^2 + 2 * p * y - 240 * p = 0) ∧ x*y = -240*p) → p = 5 :=
by sorry

end prime_root_condition_l65_65334


namespace real_solutions_count_l65_65795

theorem real_solutions_count : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), (2 : ℝ) ^ (3 * x ^ 2 - 8 * x + 4) = 1 → x = 2 ∨ x = 2 / 3 :=
by
  sorry

end real_solutions_count_l65_65795


namespace num_five_letter_words_correct_l65_65402

-- Define the number of letters in the alphabet
def num_letters : ℕ := 26

-- Define the number of vowels
def num_vowels : ℕ := 5

-- Define a function that calculates the number of valid five-letter words
def num_five_letter_words : ℕ :=
  num_letters * num_vowels * num_letters * num_letters

-- The theorem statement we need to prove
theorem num_five_letter_words_correct : num_five_letter_words = 87700 :=
by
  -- The proof is omitted; it should equate the calculated value to 87700
  sorry

end num_five_letter_words_correct_l65_65402


namespace largest_divisor_of_m_l65_65288

-- Definitions
def positive_integer (m : ℕ) : Prop := m > 0
def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement
theorem largest_divisor_of_m (m : ℕ) (h1 : positive_integer m) (h2 : divisible_by (m^2) 54) : ∃ k : ℕ, k = 9 ∧ k ∣ m := 
sorry

end largest_divisor_of_m_l65_65288


namespace calculate_fraction_l65_65028

theorem calculate_fraction : (2002 - 1999)^2 / 169 = 9 / 169 :=
by
  sorry

end calculate_fraction_l65_65028


namespace y_is_triangular_l65_65965

theorem y_is_triangular (k : ℕ) (hk : k > 0) : 
  ∃ n : ℕ, y = (n * (n + 1)) / 2 :=
by
  let y := (9^k - 1) / 8
  sorry

end y_is_triangular_l65_65965


namespace incorrect_inequality_transformation_l65_65642

theorem incorrect_inequality_transformation 
    (a b : ℝ) 
    (h : a > b) 
    : ¬(1 - a > 1 - b) := 
by {
  sorry 
}

end incorrect_inequality_transformation_l65_65642


namespace log_increasing_on_interval_l65_65339

theorem log_increasing_on_interval :
  ∀ x : ℝ, x < 1 → (0.2 : ℝ)^(x^2 - 3*x + 2) > 1 :=
by
  sorry

end log_increasing_on_interval_l65_65339


namespace car_speed_second_hour_l65_65549

theorem car_speed_second_hour (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : (s1 + s2) / 2 = 35) : s2 = 60 := by
  sorry

end car_speed_second_hour_l65_65549


namespace mat_radius_increase_l65_65187

theorem mat_radius_increase (C1 C2 : ℝ) (h1 : C1 = 40) (h2 : C2 = 50) :
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  (r2 - r1) = 5 / Real.pi := by
  sorry

end mat_radius_increase_l65_65187


namespace lowest_students_l65_65286

theorem lowest_students (n : ℕ) (h₁ : ∃ k₁ : ℕ, n = 15 * k₁) (h₂ : ∃ k₂ : ℕ, n = 24 * k₂) : n = Nat.lcm 15 24 := by sorry

end lowest_students_l65_65286


namespace utility_bills_l65_65848

-- Definitions for the conditions
def four_hundred := 4 * 100
def five_fifty := 5 * 50
def seven_twenty := 7 * 20
def eight_ten := 8 * 10
def total := four_hundred + five_fifty + seven_twenty + eight_ten

-- Lean statement for the proof problem
theorem utility_bills : total = 870 :=
by
  -- inserting skip proof placeholder
  sorry

end utility_bills_l65_65848


namespace text_messages_ratio_l65_65657

theorem text_messages_ratio :
  ∀ (T : ℕ),
    (220 + T + 3 * 50 = 5 * 96) →
    T = 110 →
    (T : ℚ) / 220 = 1 / 2 :=
by
  intros T h1 h2
  rw [←Rat.div_self (by norm_cast; linarith [220] : 220 ≠ 0 : ℚ)]
  have h3 : (110 : ℚ) = T :=
    by exact_mod_cast h2
  rw [←h3, Rat.div_eq_div_iff] -- some rewriting with rational equality
  norm_cast
  sorry

end text_messages_ratio_l65_65657


namespace tan_theta_point_l65_65825

open Real

theorem tan_theta_point :
  ∀ θ : ℝ,
  ∃ (x y : ℝ), x = -sqrt 3 / 2 ∧ y = 1 / 2 ∧ (tan θ) = y / x → (tan θ) = -sqrt 3 / 3 :=
by
  sorry

end tan_theta_point_l65_65825


namespace sample_size_of_survey_l65_65849

theorem sample_size_of_survey (total_students : ℕ) (analyzed_students : ℕ)
  (h1 : total_students = 4000) (h2 : analyzed_students = 500) :
  analyzed_students = 500 :=
by
  sorry

end sample_size_of_survey_l65_65849


namespace nested_sqrt_eq_five_l65_65066

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l65_65066


namespace quadratic_roots_and_expression_value_l65_65203

theorem quadratic_roots_and_expression_value :
  let a := 3 + Real.sqrt 21
  let b := 3 - Real.sqrt 21
  (a ≥ b) →
  (∃ x : ℝ, x^2 - 6 * x + 11 = 23) →
  3 * a + 2 * b = 15 + Real.sqrt 21 :=
by
  intros a b h1 h2
  sorry

end quadratic_roots_and_expression_value_l65_65203


namespace triangle_c_and_area_l65_65507

theorem triangle_c_and_area
  (a b : ℝ) (C : ℝ)
  (h_a : a = 1)
  (h_b : b = 2)
  (h_C : C = Real.pi / 3) :
  ∃ (c S : ℝ), c = Real.sqrt 3 ∧ S = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_c_and_area_l65_65507


namespace constantin_mother_deposit_return_l65_65249

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l65_65249


namespace profit_percentage_is_25_l65_65005

-- Definitions of the variables involved
variables (C S : ℝ)
variables (x : ℕ)

-- Condition given in the problem
def condition1 : Prop := 20 * C = x * S
def condition2 : Prop := x = 16

-- The profit percentage we're aiming to prove
def profit_percentage : ℝ := ((S - C) / C) * 100

-- The theorem to prove
theorem profit_percentage_is_25 (h1 : condition1) (h2 : condition2) :
  profit_percentage C S = 25 :=
sorry

end profit_percentage_is_25_l65_65005


namespace simplify_expression_l65_65854

variable (x y : ℝ)

theorem simplify_expression : (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y :=
by
  sorry

end simplify_expression_l65_65854


namespace students_distribute_l65_65184

theorem students_distribute (x y : ℕ) (h₁ : x + y = 4200)
        (h₂ : x * 108 / 100 + y * 111 / 100 = 4620) :
    x = 1400 ∧ y = 2800 :=
by
  sorry

end students_distribute_l65_65184


namespace probability_of_one_in_pascals_triangle_l65_65321

theorem probability_of_one_in_pascals_triangle :
  let total_elements := Nat.sum (List.range 21 |>.map (λ n => n + 1))
  let ones_count := 2 * 19 + 1
  let p := (ones_count : ℚ) / total_elements
  p = (13 / 70 : ℚ) :=
by
  let total_elements := Nat.sum (List.range 21 |>.map (λ n => n + 1))
  let ones_count := 2 * 19 + 1
  let p := (ones_count : ℚ) / total_elements
  have h : p = (13 / 70 : ℚ) := sorry
  exact h

end probability_of_one_in_pascals_triangle_l65_65321


namespace two_digit_factors_count_l65_65229

-- Definition of the expression 10^8 - 1
def expr : ℕ := 10^8 - 1

-- Factorization of 10^8 - 1
def factored_expr : List ℕ := [73, 137, 101, 11, 3^2]

-- Define the condition for being a two-digit factor
def is_two_digit (n : ℕ) : Bool := n > 9 ∧ n < 100

-- Count the number of positive two-digit factors in the factorization of 10^8 - 1
def num_two_digit_factors : ℕ := List.length (factored_expr.filter is_two_digit)

-- The theorem stating our proof problem
theorem two_digit_factors_count : num_two_digit_factors = 2 := by
  sorry

end two_digit_factors_count_l65_65229


namespace yellow_percentage_l65_65047

theorem yellow_percentage (s w : ℝ) 
  (h_cross : w * w + 4 * w * (s - 2 * w) = 0.49 * s * s) : 
  (w / s) ^ 2 = 0.2514 :=
by
  sorry

end yellow_percentage_l65_65047


namespace percentage_second_division_l65_65376

theorem percentage_second_division (total_students : ℕ) 
                                  (first_division_percentage : ℝ) 
                                  (just_passed : ℕ) 
                                  (all_students_passed : total_students = 300) 
                                  (percentage_first_division : first_division_percentage = 26) 
                                  (students_just_passed : just_passed = 60) : 
  (26 / 100 * 300 + (total_students - (26 / 100 * 300 + 60)) + 60) = 300 → 
  ((total_students - (26 / 100 * 300 + 60)) / total_students * 100) = 54 := 
by 
  sorry

end percentage_second_division_l65_65376


namespace sum_ge_sqrtab_and_sqrt_avg_squares_l65_65526

theorem sum_ge_sqrtab_and_sqrt_avg_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ sqrt (a * b) + sqrt ((a^2 + b^2) / 2) := 
sorry

end sum_ge_sqrtab_and_sqrt_avg_squares_l65_65526


namespace infinite_radical_solution_l65_65062

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l65_65062


namespace radio_range_l65_65428

-- Define constants for speeds and time
def speed_team_1 : ℝ := 20
def speed_team_2 : ℝ := 30
def time : ℝ := 2.5

-- Define the distances each team travels
def distance_team_1 := speed_team_1 * time
def distance_team_2 := speed_team_2 * time

-- Define the total distance which is the range of the radios
def total_distance := distance_team_1 + distance_team_2

-- Prove that the total distance when they lose radio contact is 125 miles
theorem radio_range : total_distance = 125 := by
  sorry

end radio_range_l65_65428


namespace probability_prime_sum_l65_65569

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def possible_outcomes : ℕ := 48

def prime_sums : Finset ℕ := {2, 3, 5, 7, 11, 13}

def prime_count : ℕ := 19

theorem probability_prime_sum :
  ((prime_count : ℚ) / possible_outcomes) = 19 / 48 := 
by
  sorry

end probability_prime_sum_l65_65569


namespace minimum_production_volume_to_avoid_loss_l65_65733

open Real

-- Define the cost function
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * (x ^ 2)

-- Define the revenue function
def revenue (x : ℕ) : ℝ := 25 * x

-- Condition: 0 < x < 240 and x ∈ ℕ (naturals greater than 0)
theorem minimum_production_volume_to_avoid_loss (x : ℕ) (hx1 : 0 < x) (hx2 : x < 240) (hx3 : x ∈ (Set.Ioi 0)) :
  revenue x ≥ cost x ↔ x ≥ 150 :=
by
  sorry

end minimum_production_volume_to_avoid_loss_l65_65733


namespace opposite_of_neg_3_l65_65717

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l65_65717


namespace second_third_parts_length_l65_65698

variable (total_length : ℝ) (first_part : ℝ) (last_part : ℝ)
variable (second_third_part_length : ℝ)

def is_equal_length (x y : ℝ) := x = y

theorem second_third_parts_length :
  total_length = 74.5 ∧ first_part = 15.5 ∧ last_part = 16 → 
  is_equal_length (second_third_part_length) 21.5 :=
by
  intros h
  let remaining_distance := total_length - first_part - last_part
  let second_third_part_length := remaining_distance / 2
  sorry

end second_third_parts_length_l65_65698


namespace largest_whole_number_l65_65208

theorem largest_whole_number (x : ℕ) : 8 * x < 120 → x ≤ 14 :=
by
  intro h
  -- prove the main statement here
  sorry

end largest_whole_number_l65_65208


namespace linear_function_solution_l65_65375

theorem linear_function_solution (k : ℝ) (h₁ : k ≠ 0) (h₂ : 0 = k * (-2) + 3) :
  ∃ x : ℝ, k * (x - 5) + 3 = 0 ∧ x = 3 :=
by
  sorry

end linear_function_solution_l65_65375


namespace minimum_value_problem_l65_65474

theorem minimum_value_problem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1 / 2) : 
  ∃ m : ℝ, m = 10 ∧ ∀ z, z = (2 / (1 - x) + 1 / (1 - y)) → z ≥ m :=
by
  sorry

end minimum_value_problem_l65_65474


namespace marie_profit_l65_65672

-- Define constants and conditions
def loaves_baked : ℕ := 60
def morning_price : ℝ := 3.00
def discount : ℝ := 0.25
def afternoon_price : ℝ := morning_price * (1 - discount)
def cost_per_loaf : ℝ := 1.00
def donated_loaves : ℕ := 5

-- Define the number of loaves sold and revenue
def morning_loaves : ℕ := loaves_baked / 3
def morning_revenue : ℝ := morning_loaves * morning_price

def remaining_after_morning : ℕ := loaves_baked - morning_loaves
def afternoon_loaves : ℕ := remaining_after_morning / 2
def afternoon_revenue : ℝ := afternoon_loaves * afternoon_price

def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_loaves
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

-- Define the total revenue and cost
def total_revenue : ℝ := morning_revenue + afternoon_revenue
def total_cost : ℝ := loaves_baked * cost_per_loaf

-- Define the profit
def profit : ℝ := total_revenue - total_cost

-- State the proof problem
theorem marie_profit : profit = 45 := by
  sorry

end marie_profit_l65_65672


namespace solve_for_x_l65_65344

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem solve_for_x (x : ℝ) (h : determinant_2x2 (x+1) (x+2) (x-3) (x-1) = 2023) :
  x = 2018 :=
by {
  sorry
}

end solve_for_x_l65_65344


namespace bread_left_l65_65957

def initial_bread : ℕ := 1000
def bomi_ate : ℕ := 350
def yejun_ate : ℕ := 500

theorem bread_left : initial_bread - (bomi_ate + yejun_ate) = 150 :=
by
  sorry

end bread_left_l65_65957


namespace range_of_m_l65_65099

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → x^2 - 2 * x - 3 > 0) → (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l65_65099


namespace opposite_of_neg_3_l65_65716

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l65_65716


namespace min_sum_of_factors_l65_65418

theorem min_sum_of_factors (x y z : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x * y * z = 3920) : x + y + z = 70 :=
sorry

end min_sum_of_factors_l65_65418


namespace find_a5_l65_65924

-- Sequence definition
def a : ℕ → ℤ
| 0     => 1
| (n+1) => 2 * a n + 3

-- Theorem to prove
theorem find_a5 : a 4 = 61 := sorry

end find_a5_l65_65924


namespace emery_family_trip_l65_65060

theorem emery_family_trip 
  (first_part_distance : ℕ) (first_part_time : ℕ) (total_time : ℕ) (speed : ℕ) (second_part_time : ℕ) :
  first_part_distance = 100 ∧ first_part_time = 1 ∧ total_time = 4 ∧ speed = 100 ∧ second_part_time = 3 →
  second_part_time * speed = 300 :=
by 
  sorry

end emery_family_trip_l65_65060


namespace find_A_l65_65876

-- Given a three-digit number AB2 such that AB2 - 41 = 591
def valid_number (A B : ℕ) : Prop :=
  (A * 100) + (B * 10) + 2 - 41 = 591

-- We aim to prove that A = 6 given B = 2
theorem find_A (A : ℕ) (B : ℕ) (hB : B = 2) : A = 6 :=
  by
  have h : valid_number A B := by sorry
  sorry

end find_A_l65_65876


namespace find_number_of_two_dollar_pairs_l65_65738

noncomputable def pairs_of_two_dollars (x y z : ℕ) : Prop :=
  x + y + z = 15 ∧ 2 * x + 4 * y + 5 * z = 38 ∧ x >= 1 ∧ y >= 1 ∧ z >= 1

theorem find_number_of_two_dollar_pairs (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : 2 * x + 4 * y + 5 * z = 38) 
  (hx : x >= 1) 
  (hy : y >= 1) 
  (hz : z >= 1) :
  pairs_of_two_dollars x y z → x = 12 :=
by
  intros
  sorry

end find_number_of_two_dollar_pairs_l65_65738


namespace father_twice_marika_age_in_2036_l65_65685

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l65_65685


namespace opposite_of_neg_three_l65_65711

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l65_65711


namespace func_has_extrema_l65_65493

theorem func_has_extrema (a b c : ℝ) (h_a_nonzero : a ≠ 0) (h_discriminant_positive : b^2 + 8 * a * c > 0) 
    (h_pos_sum_roots : b / a > 0) (h_pos_product_roots : -2 * c / a > 0) : 
    (a * b > 0) ∧ (a * c < 0) :=
by 
  -- Proof skipped.
  sorry

end func_has_extrema_l65_65493


namespace max_marks_l65_65693

theorem max_marks (marks_obtained failed_by : ℝ) (passing_percentage : ℝ) (M : ℝ) : 
  marks_obtained = 180 ∧ failed_by = 40 ∧ passing_percentage = 0.45 ∧ (marks_obtained + failed_by = passing_percentage * M) → M = 489 :=
by 
  sorry

end max_marks_l65_65693


namespace find_f2_l65_65359

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 3) : f a b 2 = -1 :=
by
  sorry

end find_f2_l65_65359


namespace rachel_colored_pictures_l65_65398

theorem rachel_colored_pictures :
  ∃ b1 b2 : ℕ, b1 = 23 ∧ b2 = 32 ∧ ∃ remaining: ℕ, remaining = 11 ∧ (b1 + b2) - remaining = 44 :=
by
  sorry

end rachel_colored_pictures_l65_65398


namespace military_unit_soldiers_l65_65578

theorem military_unit_soldiers:
  ∃ (x N : ℕ), 
      (N = x * (x + 5)) ∧
      (N = 5 * (x + 845)) ∧
      N = 4550 :=
by
  sorry

end military_unit_soldiers_l65_65578


namespace find_missing_exponent_l65_65290

theorem find_missing_exponent (b e₁ e₂ e₃ e₄ : ℝ) (h1 : e₁ = 5.6) (h2 : e₂ = 10.3) (h3 : e₃ = 13.33744) (h4 : e₄ = 2.56256) :
  (b ^ e₁ * b ^ e₂) / b ^ e₄ = b ^ e₃ :=
by
  have h5 : e₁ + e₂ = 15.9 := sorry
  have h6 : 15.9 - e₄ = 13.33744 := sorry
  exact sorry

end find_missing_exponent_l65_65290


namespace find_quadruples_l65_65336

def is_prime (n : ℕ) := ∀ m, m ∣ n → m = 1 ∨ m = n

 theorem find_quadruples (p q a b : ℕ) (hp : is_prime p) (hq : is_prime q) (ha : 1 < a)
  : (p^a = 1 + 5 * q^b ↔ ((p = 2 ∧ q = 3 ∧ a = 4 ∧ b = 1) ∨ (p = 3 ∧ q = 2 ∧ a = 4 ∧ b = 4))) :=
by {
  sorry
}

end find_quadruples_l65_65336


namespace yura_catches_up_l65_65031

theorem yura_catches_up (a : ℕ) (x : ℕ) (h1 : 2 * a * x = a * (x + 5)) : x = 5 :=
by
  sorry

end yura_catches_up_l65_65031


namespace frustum_volume_l65_65773

noncomputable def volume_of_frustum (V₁ V₂ : ℝ) : ℝ :=
  V₁ - V₂

theorem frustum_volume : 
  let base_edge_original := 15
  let height_original := 10
  let base_edge_smaller := 9
  let height_smaller := 6
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let V_original := (1 / 3 : ℝ) * base_area_original * height_original
  let V_smaller := (1 / 3 : ℝ) * base_area_smaller * height_smaller
  volume_of_frustum V_original V_smaller = 588 := 
by
  sorry

end frustum_volume_l65_65773


namespace trigonometric_identity_l65_65212

theorem trigonometric_identity :
  4 * Real.cos (10 * (Real.pi / 180)) - Real.tan (80 * (Real.pi / 180)) = -Real.sqrt 3 := 
by 
  sorry

end trigonometric_identity_l65_65212


namespace GCF_36_54_81_l65_65430

def GCF (a b : ℕ) : ℕ := nat.gcd a b

theorem GCF_36_54_81 : GCF (GCF 36 54) 81 = 9 := by
  sorry

end GCF_36_54_81_l65_65430


namespace speed_of_faster_train_l65_65161

noncomputable def speed_of_slower_train_kmph := 36
def time_to_cross_seconds := 12
def length_of_faster_train_meters := 120

-- Speed of train V_f in kmph 
theorem speed_of_faster_train 
  (relative_speed_mps : ℝ := length_of_faster_train_meters / time_to_cross_seconds)
  (speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * (1000 / 3600))
  (speed_of_faster_train_mps : ℝ := relative_speed_mps + speed_of_slower_train_mps)
  (speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * (3600 / 1000) )
  : speed_of_faster_train_kmph = 72 := 
sorry

end speed_of_faster_train_l65_65161


namespace probability_Q_within_two_units_l65_65301

noncomputable def probability_within_two_units_of_origin (s : set (ℝ × ℝ)) (circle_center : ℝ × ℝ) (radius : ℝ) : ℝ :=
  let area_square := 6 * 6 in
  let area_circle := π * radius^2 in
  area_circle / area_square

theorem probability_Q_within_two_units 
  (Q : set (ℝ × ℝ)) 
  (center_origin : (0, 0) = ⟨0, 0⟩)
  (radius_two : ∃ (circle_center : ℝ × ℝ), circle_center = (0, 0) ∧ radius = 2)
  (square_with_vertices : Q = {p : ℝ × ℝ | -3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3}) :
  probability_within_two_units_of_origin Q (0, 0) 2 = π / 9 :=
by
  sorry

end probability_Q_within_two_units_l65_65301


namespace simplify_polynomial_l65_65405

theorem simplify_polynomial (x : ℝ) : 
  (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 = 32 * x ^ 5 := 
by 
  sorry

end simplify_polynomial_l65_65405


namespace solution_set_of_inequality_l65_65548

theorem solution_set_of_inequality (x : ℝ) (h : |x - 1| < 1) : 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l65_65548


namespace no_two_right_angles_in_triangle_l65_65535

theorem no_two_right_angles_in_triangle 
  (α β γ : ℝ)
  (h1 : α + β + γ = 180) :
  ¬ (α = 90 ∧ β = 90) :=
by
  sorry

end no_two_right_angles_in_triangle_l65_65535


namespace necessary_condition_of_equilateral_triangle_l65_65377

variable {A B C: ℝ}
variable {a b c: ℝ}

theorem necessary_condition_of_equilateral_triangle
  (h1 : B + C = 2 * A)
  (h2 : b + c = 2 * a)
  : (A = B ∧ B = C ∧ a = b ∧ b = c) ↔ (B + C = 2 * A ∧ b + c = 2 * a) := 
by
  sorry

end necessary_condition_of_equilateral_triangle_l65_65377


namespace expression_value_l65_65214

theorem expression_value (a b c : ℝ) (h : a + b + c = 0) : (a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b)) = 3 := 
by 
  sorry

end expression_value_l65_65214


namespace least_possible_sum_l65_65104

theorem least_possible_sum (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 21 * (q + 1)) : p + q = 5 :=
sorry

end least_possible_sum_l65_65104


namespace parabola_x_intercepts_l65_65639

theorem parabola_x_intercepts :
  ∃! (x : ℝ), ∃ (y : ℝ), y = 0 ∧ x = -2 * y^2 + y + 1 :=
sorry

end parabola_x_intercepts_l65_65639


namespace floating_time_l65_65752

theorem floating_time (boat_with_current: ℝ) (boat_against_current: ℝ) (distance: ℝ) (time: ℝ) : 
boat_with_current = 28 ∧ boat_against_current = 24 ∧ distance = 20 ∧ 
time = distance / ((boat_with_current - boat_against_current) / 2) → 
time = 10 := by
  sorry

end floating_time_l65_65752


namespace intensity_of_replacement_paint_l65_65699

theorem intensity_of_replacement_paint (f : ℚ) (I_new : ℚ) (I_orig : ℚ) (I_repl : ℚ) :
  f = 2/3 → I_new = 40 → I_orig = 60 → I_repl = (40 - 1/3 * 60) * (3/2) := by
  sorry

end intensity_of_replacement_paint_l65_65699


namespace twice_shorter_vs_longer_l65_65039

-- Definitions and conditions
def total_length : ℝ := 20
def shorter_length : ℝ := 8
def longer_length : ℝ := total_length - shorter_length

-- Statement to prove
theorem twice_shorter_vs_longer :
  2 * shorter_length - longer_length = 4 :=
by
  sorry

end twice_shorter_vs_longer_l65_65039


namespace sector_area_l65_65541

theorem sector_area (n : ℝ) (r : ℝ) (h₁ : n = 120) (h₂ : r = 4) : 
  (n * Real.pi * r^2 / 360) = (16 * Real.pi / 3) :=
by 
  sorry

end sector_area_l65_65541


namespace greatest_line_segment_length_l65_65294

theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ d : ℝ, d = 2 * r ∧ d = 8 :=
by
  sorry

end greatest_line_segment_length_l65_65294


namespace bill_experience_l65_65596

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l65_65596


namespace evaluate_expression_l65_65788

theorem evaluate_expression (a b : ℕ) (ha : a = 7) (hb : b = 5) : 3 * (a^3 + b^3) / (a^2 - a * b + b^2) = 36 :=
by
  rw [ha, hb]
  sorry

end evaluate_expression_l65_65788


namespace employee_overtime_hours_l65_65774

theorem employee_overtime_hours (gross_pay : ℝ) (rate_regular : ℝ) (regular_hours : ℕ) (rate_overtime : ℝ) :
  gross_pay = 622 → rate_regular = 11.25 → regular_hours = 40 → rate_overtime = 16 →
  ∃ (overtime_hours : ℕ), overtime_hours = 10 :=
by
  sorry

end employee_overtime_hours_l65_65774


namespace molecular_weight_is_correct_l65_65875

structure Compound :=
  (H C N Br O : ℕ)

structure AtomicWeights :=
  (H C N Br O : ℝ)

noncomputable def molecularWeight (compound : Compound) (weights : AtomicWeights) : ℝ :=
  compound.H * weights.H +
  compound.C * weights.C +
  compound.N * weights.N +
  compound.Br * weights.Br +
  compound.O * weights.O

def givenCompound : Compound :=
  { H := 2, C := 2, N := 1, Br := 1, O := 4 }

def givenWeights : AtomicWeights :=
  { H := 1.008, C := 12.011, N := 14.007, Br := 79.904, O := 15.999 }

theorem molecular_weight_is_correct : molecularWeight givenCompound givenWeights = 183.945 := by
  sorry

end molecular_weight_is_correct_l65_65875


namespace no_real_solutions_l65_65561

theorem no_real_solutions (x : ℝ) : ¬ (3 * x^2 + 5 = |4 * x + 2| - 3) :=
by
  sorry

end no_real_solutions_l65_65561


namespace necessary_and_sufficient_condition_l65_65985

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x → x + (1 / x) > a) ↔ a < 2 :=
sorry

end necessary_and_sufficient_condition_l65_65985


namespace max_candies_per_student_l65_65584

theorem max_candies_per_student (n_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) (max_candies : ℕ) :
  n_students = 50 ∧
  mean_candies = 7 ∧
  min_candies = 1 ∧
  max_candies = 20 →
  ∃ m : ℕ, m ≤ max_candies :=
by
  intro h
  use 20
  sorry

end max_candies_per_student_l65_65584


namespace linear_function_through_origin_l65_65544

theorem linear_function_through_origin (k : ℝ) (h : ∃ x y : ℝ, (x = 0 ∧ y = 0) ∧ y = (k - 2) * x + (k^2 - 4)) : k = -2 :=
by
  sorry

end linear_function_through_origin_l65_65544


namespace bill_experience_l65_65593

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l65_65593


namespace students_in_classroom_l65_65295

theorem students_in_classroom :
  ∃ n : ℕ, (n < 50) ∧ (n % 6 = 5) ∧ (n % 3 = 2) ∧ 
  (n = 5 ∨ n = 11 ∨ n = 17 ∨ n = 23 ∨ n = 29 ∨ n = 35 ∨ n = 41 ∨ n = 47) :=
by
  sorry

end students_in_classroom_l65_65295


namespace probability_prime_sum_correct_l65_65570

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_ways_to_form_sum (s : ℕ) : ℕ :=
  ∑ i in finset.Icc 1 6, (if 1 ≤ s - i ∧ s - i ≤ 8 then 1 else 0)

def number_of_ways_to_get_prime_sum : ℕ :=
  ∑ p in finset.filter is_prime (finset.Icc 2 14), count_ways_to_form_sum p

def total_outcomes : ℕ := 6 * 8

def probability_prime_sum : ℚ :=
  number_of_ways_to_get_prime_sum / total_outcomes

-- Theorem that needs proving
theorem probability_prime_sum_correct :
  probability_prime_sum = 11 / 24 :=
sorry

end probability_prime_sum_correct_l65_65570


namespace leftover_floss_l65_65786

/-
Conditions:
1. There are 20 students in his class.
2. Each student needs 1.5 yards of floss.
3. Each packet of floss contains 35 yards.
4. He buys the least amount necessary.
-/

def students : ℕ := 20
def floss_needed_per_student : ℝ := 1.5
def total_floss_needed : ℝ := students * floss_needed_per_student
def floss_per_packet : ℝ := 35

theorem leftover_floss : floss_per_packet - total_floss_needed = 5 :=
by
  -- Assuming these values from the conditions
  have students_val : 20 = students := rfl
  have floss_needed_val : 1.5 = floss_needed_per_student := rfl
  have total_needed_val : total_floss_needed = 30 := by 
    simp [students, floss_needed_per_student, total_floss_needed]
  have floss_per_packet_val : 35 = floss_per_packet := rfl
  
  -- Calculation to get the leftover floss
  calc
    floss_per_packet - total_floss_needed 
        = 35 - 30 : by rw [total_needed_val]
    ... = 5 : by norm_num

end leftover_floss_l65_65786


namespace infinite_coprime_terms_l65_65523

theorem infinite_coprime_terms (a b m : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_m_pos : 0 < m) (h_coprime : Nat.gcd a b = 1) :
  ∃^∞ k, Nat.gcd (a + k * b) m = 1 :=
sorry

end infinite_coprime_terms_l65_65523


namespace fraction_subtraction_simplest_form_l65_65456

theorem fraction_subtraction_simplest_form :
  (8 / 24 - 5 / 40 = 5 / 24) :=
by
  sorry

end fraction_subtraction_simplest_form_l65_65456


namespace probability_of_Q_l65_65306

noncomputable def probability_Q_within_two_units_of_origin : ℚ :=
  let side_length_square := 6
  let area_square := side_length_square ^ 2
  let radius_circle := 2
  let area_circle := π * radius_circle ^ 2
  area_circle / area_square

theorem probability_of_Q :
  probability_Q_within_two_units_of_origin = π / 9 :=
by
  -- The proof would go here
  sorry

end probability_of_Q_l65_65306


namespace largest_of_A_B_C_l65_65029

noncomputable def A : ℝ := (2010 / 2009) + (2010 / 2011)
noncomputable def B : ℝ := (2010 / 2011) + (2012 / 2011)
noncomputable def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_of_A_B_C : B > A ∧ B > C := by
  sorry

end largest_of_A_B_C_l65_65029


namespace probability_of_one_in_pascals_triangle_l65_65320

theorem probability_of_one_in_pascals_triangle :
  let total_elements := Nat.sum (List.range 21 |>.map (λ n => n + 1))
  let ones_count := 2 * 19 + 1
  let p := (ones_count : ℚ) / total_elements
  p = (13 / 70 : ℚ) :=
by
  let total_elements := Nat.sum (List.range 21 |>.map (λ n => n + 1))
  let ones_count := 2 * 19 + 1
  let p := (ones_count : ℚ) / total_elements
  have h : p = (13 / 70 : ℚ) := sorry
  exact h

end probability_of_one_in_pascals_triangle_l65_65320


namespace oil_bill_additional_amount_l65_65546

variables (F JanuaryBill : ℝ) (x : ℝ)

-- Given conditions
def condition1 : Prop := F / JanuaryBill = 5 / 4
def condition2 : Prop := (F + x) / JanuaryBill = 3 / 2
def JanuaryBillVal : Prop := JanuaryBill = 180

-- The theorem to prove
theorem oil_bill_additional_amount
  (h1 : condition1 F JanuaryBill)
  (h2 : condition2 F JanuaryBill x)
  (h3 : JanuaryBillVal JanuaryBill) :
  x = 45 := 
  sorry

end oil_bill_additional_amount_l65_65546


namespace max_d_minus_r_l65_65501

theorem max_d_minus_r (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) : 
  d - r = 35 :=
sorry

end max_d_minus_r_l65_65501


namespace x_plus_y_possible_values_l65_65484

theorem x_plus_y_possible_values (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < 20) (h4 : y < 20) (h5 : x + y + x * y = 99) : 
  x + y = 23 ∨ x + y = 18 :=
by
  sorry

end x_plus_y_possible_values_l65_65484


namespace students_brought_apples_l65_65508

theorem students_brought_apples (A B C D : ℕ) (h1 : B = 8) (h2 : C = 10) (h3 : D = 5) (h4 : A - D + B - D = C) : A = 12 :=
by {
  sorry
}

end students_brought_apples_l65_65508


namespace payment_n_amount_l65_65173

def payment_m_n (m n : ℝ) : Prop :=
  m + n = 550 ∧ m = 1.2 * n

theorem payment_n_amount : ∃ n : ℝ, ∀ m : ℝ, payment_m_n m n → n = 250 :=
by
  sorry

end payment_n_amount_l65_65173


namespace point_not_on_line_l65_65522

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : ¬ ((2023, 0) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) :=
by
  -- proof is omitted
  sorry

end point_not_on_line_l65_65522


namespace marikas_father_age_twice_in_2036_l65_65689

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l65_65689


namespace gcf_36_54_81_l65_65431

theorem gcf_36_54_81 : Nat.gcd (Nat.gcd 36 54) 81 = 9 :=
by
  -- The theorem states that the greatest common factor of 36, 54, and 81 is 9.
  sorry

end gcf_36_54_81_l65_65431


namespace quadratic_eq_equal_roots_l65_65625

theorem quadratic_eq_equal_roots (m x : ℝ) (h : (x^2 - m * x + m - 1 = 0) ∧ ((x - 1)^2 = 0)) : 
    m = 2 ∧ ((x = 1 ∧ x = 1)) :=
by
  sorry

end quadratic_eq_equal_roots_l65_65625


namespace rightmost_four_digits_of_7_pow_2045_l65_65026

theorem rightmost_four_digits_of_7_pow_2045 : (7^2045 % 10000) = 6807 :=
by
  sorry

end rightmost_four_digits_of_7_pow_2045_l65_65026


namespace kat_boxing_trainings_per_week_l65_65382

noncomputable def strength_training_hours_per_week : ℕ := 3
noncomputable def boxing_training_hours (x : ℕ) : ℚ := 1.5 * x
noncomputable def total_training_hours : ℕ := 9

theorem kat_boxing_trainings_per_week (x : ℕ) (h : total_training_hours = strength_training_hours_per_week + boxing_training_hours x) : x = 4 :=
by
  sorry

end kat_boxing_trainings_per_week_l65_65382


namespace range_x_sub_cos_y_l65_65174

theorem range_x_sub_cos_y (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) : 
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 :=
sorry

end range_x_sub_cos_y_l65_65174


namespace opposite_number_subtraction_l65_65283

variable (a b : ℝ)

theorem opposite_number_subtraction : -(a - b) = b - a := 
sorry

end opposite_number_subtraction_l65_65283


namespace sin_cos_eq_one_sol_set_l65_65615

-- Define the interval
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := Real.sin x + Real.cos x = 1

-- Theorem statement: prove that the solution set is {0, π/2}
theorem sin_cos_eq_one_sol_set :
  ∀ (x : ℝ), in_interval x → satisfies_eq x ↔ x = 0 ∨ x = Real.pi / 2 := by
  sorry

end sin_cos_eq_one_sol_set_l65_65615


namespace real_roots_approx_correct_to_4_decimal_places_l65_65059

noncomputable def f (x : ℝ) : ℝ := x^4 - (2 * 10^10 + 1) * x^2 - x + 10^20 + 10^10 - 1

theorem real_roots_approx_correct_to_4_decimal_places :
  ∃ x1 x2 : ℝ, 
  abs (x1 - 99999.9997) ≤ 0.0001 ∧ 
  abs (x2 - 100000.0003) ≤ 0.0001 ∧ 
  f x1 = 0 ∧ 
  f x2 = 0 :=
sorry

end real_roots_approx_correct_to_4_decimal_places_l65_65059


namespace horner_v4_at_2_l65_65429

def horner (a : List Int) (x : Int) : Int :=
  a.foldr (fun ai acc => ai + x * acc) 0

noncomputable def poly_coeffs : List Int := [1, -12, 60, -160, 240, -192, 64]

theorem horner_v4_at_2 : horner poly_coeffs 2 = 80 := by
  sorry

end horner_v4_at_2_l65_65429


namespace value_of_star_l65_65623

theorem value_of_star (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : (a + b) % 4 = 0) : a^2 + 2*a*b + b^2 = 64 :=
by
  sorry

end value_of_star_l65_65623


namespace dogsled_race_time_difference_l65_65427

theorem dogsled_race_time_difference :
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  T_W - T_A = 3 :=
by
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  sorry

end dogsled_race_time_difference_l65_65427


namespace mail_distribution_l65_65986

-- Define the number of houses
def num_houses : ℕ := 10

-- Define the pieces of junk mail per house
def mail_per_house : ℕ := 35

-- Define total pieces of junk mail delivered
def total_pieces_of_junk_mail : ℕ := num_houses * mail_per_house

-- Main theorem statement
theorem mail_distribution : total_pieces_of_junk_mail = 350 := by
  sorry

end mail_distribution_l65_65986


namespace find_Sum_4n_l65_65082

variable {a : ℕ → ℕ} -- Define a sequence a_n

-- Define our conditions about the sums Sn and S3n
axiom Sum_n : ℕ → ℕ 
axiom Sum_3n : ℕ → ℕ 
axiom Sum_4n : ℕ → ℕ 

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a n + a 0)) / 2

axiom h1 : is_arithmetic_sequence a
axiom h2 : Sum_n 1 = 2
axiom h3 : Sum_3n 3 = 12

theorem find_Sum_4n : Sum_4n 4 = 20 :=
sorry

end find_Sum_4n_l65_65082


namespace marikas_father_age_twice_in_2036_l65_65690

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l65_65690


namespace Z_4_3_eq_neg11_l65_65780

def Z (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2

theorem Z_4_3_eq_neg11 : Z 4 3 = -11 := 
by
  sorry

end Z_4_3_eq_neg11_l65_65780


namespace rotational_homothety_commutes_l65_65842

-- Definitions for our conditions
variable (H1 H2 : Point → Point)

-- Definition of rotational homothety. 
-- You would define it based on your bespoke library/formalization.
axiom is_rot_homothety : ∀ (H : Point → Point), Prop

-- Main theorem statement
theorem rotational_homothety_commutes (H1 H2 : Point → Point) (A : Point) 
    (h1_rot : is_rot_homothety H1) (h2_rot : is_rot_homothety H2) : 
    (H1 ∘ H2 = H2 ∘ H1) ↔ (H1 (H2 A) = H2 (H1 A)) :=
sorry

end rotational_homothety_commutes_l65_65842


namespace average_of_roots_l65_65955

theorem average_of_roots (a b: ℝ) (h : a ≠ 0) (hr : ∃ x1 x2: ℝ, a * x1 ^ 2 - 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 - 3 * a * x2 + b = 0 ∧ x1 ≠ x2):
  (∃ r1 r2: ℝ, a * r1 ^ 2 - 3 * a * r1 + b = 0 ∧ a * r2 ^ 2 - 3 * a * r2 + b = 0 ∧ r1 ≠ r2) →
  ((r1 + r2) / 2 = 3 / 2) :=
by
  sorry

end average_of_roots_l65_65955


namespace point_in_second_quadrant_l65_65512

theorem point_in_second_quadrant {x y : ℝ} (hx : x < 0) (hy : y > 0) : 
  ∃ q, q = 2 :=
by
  sorry

end point_in_second_quadrant_l65_65512


namespace total_rainfall_l65_65196

theorem total_rainfall :
  let monday := 0.12962962962962962
  let tuesday := 0.35185185185185186
  let wednesday := 0.09259259259259259
  let thursday := 0.25925925925925924
  let friday := 0.48148148148148145
  let saturday := 0.2222222222222222
  let sunday := 0.4444444444444444
  (monday + tuesday + wednesday + thursday + friday + saturday + sunday) = 1.9814814814814815 :=
by
  -- proof to be filled here
  sorry

end total_rainfall_l65_65196


namespace option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l65_65744

section

variable (π : Real) (x : Real)

-- Definition of a fraction in this context
def is_fraction (num denom : Real) : Prop := denom ≠ 0

-- Proving each given option is a fraction
theorem option_a_is_fraction : is_fraction 1 π := 
sorry

theorem option_b_is_fraction : is_fraction x 3 :=
sorry

theorem option_c_is_fraction : is_fraction 2 5 :=
sorry

theorem option_d_is_fraction : is_fraction 1 (x - 1) :=
sorry

end

end option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l65_65744


namespace sqrt_equation_has_solution_l65_65070

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l65_65070


namespace eccentricity_range_l65_65861

-- Definitions and conditions
variable (a b c e : ℝ) (A B: ℝ × ℝ)
variable (d1 d2 : ℝ)

variable (a_pos : a > 2)
variable (b_pos : b > 0)
variable (c_pos : c > 0)
variable (c_eq : c = Real.sqrt (a ^ 2 + b ^ 2))
variable (A_def : A = (a, 0))
variable (B_def : B = (0, b))
variable (d1_def : d1 = abs (b * 2 + a * 0 - a * b ) / Real.sqrt (a^2 + b^2))
variable (d2_def : d2 = abs (b * (-2) + a * 0 - a * b) / Real.sqrt (a^2 + b^2))
variable (d_ineq : d1 + d2 ≥ (4 / 5) * c)
variable (eccentricity : e = c / a)

-- Theorem statement
theorem eccentricity_range : (Real.sqrt 5 / 2 ≤ e) ∧ (e ≤ Real.sqrt 5) :=
by sorry

end eccentricity_range_l65_65861


namespace box_length_l65_65889

theorem box_length :
  ∃ (length : ℝ), 
  let box_height := 8
  let box_width := 10
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let num_blocks := 40
  let box_volume := box_height * box_width * length
  let block_volume := block_height * block_width * block_length
  num_blocks * block_volume = box_volume ∧ length = 12 := by
  sorry

end box_length_l65_65889


namespace sandy_total_money_l65_65962

-- Definitions based on conditions
def X_initial (X : ℝ) : Prop := 
  X - 0.30 * X = 210

def watch_cost : ℝ := 50

-- Question translated into a proof goal
theorem sandy_total_money (X : ℝ) (h : X_initial X) : 
  X + watch_cost = 350 := by
  sorry

end sandy_total_money_l65_65962


namespace A_elements_l65_65478

open Set -- Open the Set namespace for easy access to set operations

def A : Set ℕ := {x | ∃ (n : ℕ), 12 = n * (6 - x)}

theorem A_elements : A = {0, 2, 3, 4, 5} :=
by
  -- proof steps here
  sorry

end A_elements_l65_65478


namespace remaining_distance_l65_65263

theorem remaining_distance (total_depth distance_traveled remaining_distance : ℕ) (h_total_depth : total_depth = 1218) 
  (h_distance_traveled : distance_traveled = 849) : remaining_distance = total_depth - distance_traveled := 
by
  sorry

end remaining_distance_l65_65263


namespace combined_instruments_l65_65908

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_horns := charlie_horns / 2
def carli_harps := 0

def charlie_total := charlie_flutes + charlie_horns + charlie_harps
def carli_total := carli_flutes + carli_horns + carli_harps
def combined_total := charlie_total + carli_total

theorem combined_instruments :
  combined_total = 7 :=
by
  sorry

end combined_instruments_l65_65908


namespace solve_equation1_solve_equation2_l65_65408

theorem solve_equation1 (x : ℝ) : 3 * (x - 1)^3 = 24 ↔ x = 3 := by
  sorry

theorem solve_equation2 (x : ℝ) : (x - 3)^2 = 64 ↔ x = 11 ∨ x = -5 := by
  sorry

end solve_equation1_solve_equation2_l65_65408


namespace total_amount_shared_l65_65506

theorem total_amount_shared (z : ℝ) (hz : z = 150) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 555 :=
by
  sorry

end total_amount_shared_l65_65506


namespace find_n_in_range_l65_65421

theorem find_n_in_range :
  ∃ n : ℕ, n > 1 ∧ 
           n % 3 = 2 ∧ 
           n % 5 = 2 ∧ 
           n % 7 = 2 ∧ 
           101 ≤ n ∧ n ≤ 134 :=
by sorry

end find_n_in_range_l65_65421


namespace opposite_of_neg_three_l65_65712

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l65_65712


namespace seeking_the_cause_from_the_result_means_sufficient_condition_l65_65426

-- Define the necessary entities for the conditions
inductive Condition
| Necessary
| Sufficient
| NecessaryAndSufficient
| NecessaryOrSufficient

-- Define the statement of the proof problem
theorem seeking_the_cause_from_the_result_means_sufficient_condition :
  (seeking_the_cause_from_the_result : Condition) = Condition.Sufficient :=
sorry

end seeking_the_cause_from_the_result_means_sufficient_condition_l65_65426


namespace sara_spent_correct_amount_on_movies_l65_65392

def cost_ticket : ℝ := 10.62
def num_tickets : ℕ := 2
def cost_rented_movie : ℝ := 1.59
def cost_purchased_movie : ℝ := 13.95

def total_amount_spent : ℝ :=
  num_tickets * cost_ticket + cost_rented_movie + cost_purchased_movie

theorem sara_spent_correct_amount_on_movies :
  total_amount_spent = 36.78 :=
sorry

end sara_spent_correct_amount_on_movies_l65_65392


namespace problem_part_1_problem_part_2_l65_65805

def f (x m : ℝ) := 2 * x^2 + (2 - m) * x - m
def g (x m : ℝ) := x^2 - x + 2 * m

theorem problem_part_1 (x : ℝ) : f x 1 > 0 ↔ (x > 1/2 ∨ x < -1) :=
by sorry

theorem problem_part_2 {m x : ℝ} (hm : 0 < m) : f x m ≤ g x m ↔ (-3 ≤ x ∧ x ≤ m) :=
by sorry

end problem_part_1_problem_part_2_l65_65805


namespace karen_drive_l65_65123

theorem karen_drive (a b c x : ℕ) (h1 : a ≥ 1) (h2 : a + b + c ≤ 9) (h3 : 33 * (c - a) = 25 * x) :
  a^2 + b^2 + c^2 = 75 :=
sorry

end karen_drive_l65_65123


namespace slope_of_line_l65_65633

theorem slope_of_line (x1 x2 y1 y2 : ℝ) (h1 : 1 = (x1 + x2) / 2) (h2 : 1 = (y1 + y2) / 2) 
                      (h3 : (x1^2 / 36) + (y1^2 / 9) = 1) (h4 : (x2^2 / 36) + (y2^2 / 9) = 1) :
  (y2 - y1) / (x2 - x1) = -1 / 4 :=
by
  sorry

end slope_of_line_l65_65633


namespace prob_return_to_freezer_l65_65052

-- Define the probabilities of picking two pops of each flavor
def probability_same_flavor (total: ℕ) (pop1: ℕ) (pop2: ℕ) : ℚ :=
  (pop1 * pop2) / (total * (total - 1))

-- Definitions according to the problem conditions
def cherry_pops : ℕ := 4
def orange_pops : ℕ := 3
def lemon_lime_pops : ℕ := 4
def total_pops : ℕ := cherry_pops + orange_pops + lemon_lime_pops

-- Calculate the probability of picking two ice pops of the same flavor
def prob_cherry : ℚ := probability_same_flavor total_pops cherry_pops (cherry_pops - 1)
def prob_orange : ℚ := probability_same_flavor total_pops orange_pops (orange_pops - 1)
def prob_lemon_lime : ℚ := probability_same_flavor total_pops lemon_lime_pops (lemon_lime_pops - 1)

def prob_same_flavor : ℚ := prob_cherry + prob_orange + prob_lemon_lime
def prob_diff_flavor : ℚ := 1 - prob_same_flavor

-- Theorem stating the probability of needing to return to the freezer
theorem prob_return_to_freezer : prob_diff_flavor = 8 / 11 := by
  sorry

end prob_return_to_freezer_l65_65052


namespace B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l65_65634

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x < 1 ∨ x > 2}
def setB (m : ℝ) : Set ℝ := 
  if m = 0 then {x : ℝ | x > 1} 
  else if m < 0 then {x : ℝ | x > 1 ∨ x < (2/m)}
  else if 0 < m ∧ m < 2 then {x : ℝ | 1 < x ∧ x < (2/m)}
  else if m = 2 then ∅
  else {x : ℝ | (2/m) < x ∧ x < 1}

-- Complement of set A
def complementA : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

-- Proposition: if B subset of complement of A
theorem B_subset_complementA (m : ℝ) : setB m ⊆ complementA ↔ 1 ≤ m ∧ m ≤ 2 := by
  sorry

-- Similarly, we can define the other two propositions
theorem A_intersection_B_nonempty (m : ℝ) : (setA ∩ setB m).Nonempty ↔ m < 1 ∨ m > 2 := by
  sorry

theorem A_union_B_eq_A (m : ℝ) : setA ∪ setB m = setA ↔ m ≥ 2 := by
  sorry

end B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l65_65634


namespace ticket_is_five_times_soda_l65_65747

variable (p_i p_r : ℝ)

theorem ticket_is_five_times_soda
  (h1 : 6 * p_i + 20 * p_r = 50)
  (h2 : 6 * p_r = p_i + p_r) : p_i = 5 * p_r :=
sorry

end ticket_is_five_times_soda_l65_65747


namespace joan_original_seashells_l65_65122

theorem joan_original_seashells (a b total: ℕ) (h1 : a = 63) (h2 : b = 16) (h3: total = a + b) : total = 79 :=
by
  rw [h1, h2] at h3
  exact h3

end joan_original_seashells_l65_65122


namespace weight_of_mixture_is_correct_l65_65734

def weight_of_mixture (weight_a_per_l : ℕ) (weight_b_per_l : ℕ) 
                      (total_volume : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : ℚ :=
  let volume_a := (ratio_a : ℚ) / (ratio_a + ratio_b) * total_volume
  let volume_b := (ratio_b : ℚ) / (ratio_a + ratio_b) * total_volume
  let weight_a := volume_a * weight_a_per_l
  let weight_b := volume_b * weight_b_per_l
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_of_mixture 800 850 3 3 2 = 2.46 :=
by
  sorry

end weight_of_mixture_is_correct_l65_65734


namespace opposite_of_neg_3_l65_65715

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l65_65715


namespace total_surface_area_with_holes_l65_65899

def cube_edge_length : ℝ := 5
def hole_side_length : ℝ := 2

/-- Calculate the total surface area of a modified cube with given edge length and holes -/
theorem total_surface_area_with_holes 
  (l : ℝ) (h : ℝ)
  (hl_pos : l > 0) (hh_pos : h > 0) (hh_lt_hl : h < l) : 
  (6 * l^2 - 6 * h^2 + 6 * 4 * h^2) = 222 :=
by sorry

end total_surface_area_with_holes_l65_65899


namespace not_perfect_square_7p_3p_4_l65_65372

theorem not_perfect_square_7p_3p_4 (p : ℕ) (hp : Nat.Prime p) : ¬∃ a : ℕ, a^2 = 7 * p + 3^p - 4 := 
by
  sorry

end not_perfect_square_7p_3p_4_l65_65372


namespace ratio_of_new_circumference_to_increase_in_area_l65_65877

theorem ratio_of_new_circumference_to_increase_in_area
  (r k : ℝ) (h_k : 0 < k) :
  (2 * π * (r + k)) / (π * (2 * r * k + k ^ 2)) = 2 * (r + k) / (2 * r * k + k ^ 2) :=
by
  sorry

end ratio_of_new_circumference_to_increase_in_area_l65_65877


namespace find_y_l65_65743

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end find_y_l65_65743


namespace solution_set_l65_65919

def inequality_solution (x : ℝ) : Prop :=
  4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9

theorem solution_set :
  { x : ℝ | inequality_solution x } = { x : ℝ | (63 / 26 : ℝ) < x ∧ x ≤ (28 / 11 : ℝ) } :=
by
  sorry

end solution_set_l65_65919


namespace jim_juice_amount_l65_65858

def susan_juice : ℚ := 3 / 8
def jim_fraction : ℚ := 5 / 6

theorem jim_juice_amount : jim_fraction * susan_juice = 5 / 16 := by
  sorry

end jim_juice_amount_l65_65858


namespace compute_f_1_g_3_l65_65521

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x + 2

theorem compute_f_1_g_3 : f (1 + g 3) = 7 := 
by
  -- Proof goes here
  sorry

end compute_f_1_g_3_l65_65521


namespace Konstantin_mother_returns_amount_l65_65245

theorem Konstantin_mother_returns_amount
  (deposit_usd : ℝ)
  (exchange_rate : ℝ)
  (equivalent_rubles : ℝ)
  (h_deposit_usd : deposit_usd = 10000)
  (h_exchange_rate : exchange_rate = 58.15)
  (h_equivalent_rubles : equivalent_rubles = deposit_usd * exchange_rate) :
  equivalent_rubles = 581500 :=
by {
  rw [h_deposit_usd, h_exchange_rate] at h_equivalent_rubles,
  exact h_equivalent_rubles
}

end Konstantin_mother_returns_amount_l65_65245


namespace inscribed_circle_diameter_of_right_triangle_l65_65175

theorem inscribed_circle_diameter_of_right_triangle (a b : ℕ) (hc : a = 8) (hb : b = 15) :
  2 * (60 / (a + b + Int.sqrt (a ^ 2 + b ^ 2))) = 6 :=
by
  sorry

end inscribed_circle_diameter_of_right_triangle_l65_65175


namespace probability_defective_second_given_first_l65_65166

theorem probability_defective_second_given_first :
  let total_products := 20
  let defective_products := 4
  let genuine_products := 16
  let prob_A := defective_products / total_products
  let prob_AB := (defective_products * (defective_products - 1)) / (total_products * (total_products - 1))
  let prob_B_given_A := prob_AB / prob_A
  prob_B_given_A = 3 / 19 :=
by
  sorry

end probability_defective_second_given_first_l65_65166


namespace car_speed_l65_65439

theorem car_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 495) (h_time : time = 5) : 
  distance / time = 99 :=
by
  rw [h_distance, h_time]
  norm_num

end car_speed_l65_65439


namespace problem_l65_65383

theorem problem (a k : ℕ) (h_a_pos : 0 < a) (h_a_k_pos : 0 < k) (h_div : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : k ≥ a :=
sorry

end problem_l65_65383


namespace divisor_is_679_l65_65077

noncomputable def x : ℕ := 8
noncomputable def y : ℕ := 9
noncomputable def z : ℝ := 549.7025036818851
noncomputable def p : ℕ := x^3
noncomputable def q : ℕ := y^3
noncomputable def r : ℕ := p * q

theorem divisor_is_679 (k : ℝ) (h : r / k = z) : k = 679 := by
  sorry

end divisor_is_679_l65_65077


namespace boat_travel_distance_downstream_l65_65888

-- Define the given conditions
def speed_boat_still : ℝ := 22
def speed_stream : ℝ := 5
def time_downstream : ℝ := 5

-- Define the effective speed and the computed distance
def effective_speed_downstream : ℝ := speed_boat_still + speed_stream
def distance_traveled_downstream : ℝ := effective_speed_downstream * time_downstream

-- State the proof problem that distance_traveled_downstream is 135 km
theorem boat_travel_distance_downstream :
  distance_traveled_downstream = 135 :=
by
  -- The proof will go here
  sorry

end boat_travel_distance_downstream_l65_65888


namespace intersection_P_Q_l65_65844

def P (k : ℤ) (α : ℝ) : Prop := 2 * k * Real.pi ≤ α ∧ α ≤ (2 * k + 1) * Real.pi
def Q (α : ℝ) : Prop := -4 ≤ α ∧ α ≤ 4

theorem intersection_P_Q :
  (∃ k : ℤ, P k α) ∧ Q α ↔ (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi) :=
by
  sorry

end intersection_P_Q_l65_65844


namespace isosceles_trapezoid_side_length_l65_65968

theorem isosceles_trapezoid_side_length (A b1 b2 : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 48) (hb1 : b1 = 9) (hb2 : b2 = 15) 
  (h_area : A = 1 / 2 * (b1 + b2) * h) 
  (h_h : h = 4)
  (h_s : s^2 = h^2 + ((b2 - b1) / 2)^2) :
  s = 5 :=
by sorry

end isosceles_trapezoid_side_length_l65_65968


namespace angle_B_l65_65646

theorem angle_B (A B C a b c : ℝ) (h : 2 * b * (Real.cos A) = 2 * c - Real.sqrt 3 * a) :
  B = Real.pi / 6 :=
sorry

end angle_B_l65_65646


namespace time_taken_by_A_l65_65510

theorem time_taken_by_A (t : ℚ) (h1 : 3 * (t + 1 / 2) = 4 * t) : t = 3 / 2 ∧ (t + 1 / 2) = 2 := 
  by
  intros
  sorry

end time_taken_by_A_l65_65510


namespace ben_less_than_jack_l65_65061

def jack_amount := 26
def total_amount := 50
def eric_ben_difference := 10

theorem ben_less_than_jack (E B J : ℕ) (h1 : E = B - eric_ben_difference) (h2 : J = jack_amount) (h3 : E + B + J = total_amount) :
  J - B = 9 :=
by sorry

end ben_less_than_jack_l65_65061


namespace find_2023rd_letter_in_sequence_l65_65279

def repeating_sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'F', 'E', 'D', 'C', 'B', 'A']

def nth_in_repeating_sequence (n : ℕ) : Char :=
  repeating_sequence.get! (n % 13)

theorem find_2023rd_letter_in_sequence :
  nth_in_repeating_sequence 2023 = 'H' :=
by
  sorry

end find_2023rd_letter_in_sequence_l65_65279


namespace number_of_convex_quadrilaterals_l65_65557

theorem number_of_convex_quadrilaterals (n : ℕ := 12) : (nat.choose n 4) = 495 :=
by
  have h1 : nat.choose 12 4 = 495 := by sorry
  exact h1

end number_of_convex_quadrilaterals_l65_65557


namespace magic_coin_l65_65167

theorem magic_coin (m n : ℕ) (h_m_prime: Nat.gcd m n = 1)
  (h_prob : (m : ℚ) / n = 1 / 158760): m + n = 158761 := by
  sorry

end magic_coin_l65_65167
