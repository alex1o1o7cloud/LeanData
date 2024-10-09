import Mathlib

namespace items_priced_at_9_yuan_l865_86520

theorem items_priced_at_9_yuan (equal_number_items : ℕ)
  (total_cost : ℕ)
  (price_8_yuan : ℕ)
  (price_9_yuan : ℕ)
  (price_8_yuan_count : ℕ)
  (price_9_yuan_count : ℕ) :
  equal_number_items * 2 = price_8_yuan_count + price_9_yuan_count ∧
  (price_8_yuan_count * price_8_yuan + price_9_yuan_count * price_9_yuan = total_cost) ∧
  (price_8_yuan = 8) ∧
  (price_9_yuan = 9) ∧
  (total_cost = 172) →
  price_9_yuan_count = 12 :=
by
  sorry

end items_priced_at_9_yuan_l865_86520


namespace exists_k_for_blocks_of_2022_l865_86570

theorem exists_k_for_blocks_of_2022 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (0 < k) ∧ (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ j, 
  k^i / 10^j % 10^4 = 2022)) :=
sorry

end exists_k_for_blocks_of_2022_l865_86570


namespace max_value_of_quadratic_l865_86504

theorem max_value_of_quadratic :
  ∃ x_max : ℝ, x_max = 1.5 ∧
  ∀ x : ℝ, -3 * x^2 + 9 * x + 24 ≤ -3 * (1.5)^2 + 9 * 1.5 + 24 := by
  sorry

end max_value_of_quadratic_l865_86504


namespace greatest_p_meets_conditions_l865_86533

-- Define a four-digit number and its reversal being divisible by 63 and another condition of divisibility
def is_divisible_by (n m : ℕ) : Prop :=
  m % n = 0

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ a d => a * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def p := 9507

-- The main theorem we aim to prove.
theorem greatest_p_meets_conditions (p q : ℕ) 
  (h1 : is_four_digit p) 
  (h2 : is_four_digit q) 
  (h3 : reverse_digits p = q) 
  (h4 : is_divisible_by 63 p) 
  (h5 : is_divisible_by 63 q) 
  (h6 : is_divisible_by 9 p) : 
  p = 9507 :=
sorry

end greatest_p_meets_conditions_l865_86533


namespace max_value_fraction_l865_86501

theorem max_value_fraction (x y : ℝ) (hx : 1 / 3 ≤ x ∧ x ≤ 3 / 5) (hy : 1 / 4 ≤ y ∧ y ≤ 1 / 2) :
  (∃ x y, (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (xy / (x^2 + y^2) = 6 / 13)) :=
by
  sorry

end max_value_fraction_l865_86501


namespace chris_remaining_money_l865_86579

variable (video_game_cost : ℝ)
variable (discount_rate : ℝ)
variable (candy_cost : ℝ)
variable (tax_rate : ℝ)
variable (shipping_fee : ℝ)
variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)

noncomputable def remaining_money (video_game_cost discount_rate candy_cost tax_rate shipping_fee hourly_rate hours_worked : ℝ) : ℝ :=
  let discount := discount_rate * video_game_cost
  let discounted_price := video_game_cost - discount
  let total_video_game_cost := discounted_price + shipping_fee
  let video_tax := tax_rate * total_video_game_cost
  let candy_tax := tax_rate * candy_cost
  let total_cost := (total_video_game_cost + video_tax) + (candy_cost + candy_tax)
  let earnings := hourly_rate * hours_worked
  earnings - total_cost

theorem chris_remaining_money : remaining_money 60 0.15 5 0.10 3 8 9 = 7.1 :=
by
  sorry

end chris_remaining_money_l865_86579


namespace sum_of_natural_numbers_l865_86539

theorem sum_of_natural_numbers (n : ℕ) (h : n * (n + 1) = 812) : n = 28 := by
  sorry

end sum_of_natural_numbers_l865_86539


namespace sin_cos_inequality_l865_86588

open Real

theorem sin_cos_inequality 
  (x : ℝ) (hx : 0 < x ∧ x < π / 2) 
  (m n : ℕ) (hmn : n > m)
  : 2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) :=
sorry

end sin_cos_inequality_l865_86588


namespace imo_1988_problem_29_l865_86553

variable (d r : ℕ)
variable (h1 : d > 1)
variable (h2 : 1059 % d = r)
variable (h3 : 1417 % d = r)
variable (h4 : 2312 % d = r)

theorem imo_1988_problem_29 :
  d - r = 15 := by sorry

end imo_1988_problem_29_l865_86553


namespace probability_white_ball_from_first_urn_correct_l865_86556

noncomputable def probability_white_ball_from_first_urn : ℝ :=
  let p_H1 : ℝ := 0.5
  let p_H2 : ℝ := 0.5
  let p_A_given_H1 : ℝ := 0.7
  let p_A_given_H2 : ℝ := 0.6
  let p_A : ℝ := p_H1 * p_A_given_H1 + p_H2 * p_A_given_H2
  p_H1 * p_A_given_H1 / p_A

theorem probability_white_ball_from_first_urn_correct :
  probability_white_ball_from_first_urn = 0.538 :=
sorry

end probability_white_ball_from_first_urn_correct_l865_86556


namespace interest_rate_l865_86509

theorem interest_rate (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) (diff : ℝ) 
    (hP : P = 1500)
    (ht : t = 2)
    (hdiff : diff = 15)
    (hCI : CI = P * (1 + r / 100)^t - P)
    (hSI : SI = P * r * t / 100)
    (hCI_SI_diff : CI - SI = diff) :
    r = 1 := 
by
  sorry -- proof goes here


end interest_rate_l865_86509


namespace sector_area_l865_86586

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360) * Real.pi * r^2 = (35 * Real.pi) / 3 :=
by
  -- Using the provided conditions to simplify the expression
  rw [h_r, h_θ]
  -- Simplify and solve the expression
  sorry

end sector_area_l865_86586


namespace obtuse_triangle_side_range_l865_86598

theorem obtuse_triangle_side_range (a : ℝ) (h1 : 0 < a)
  (h2 : a + (a + 1) > a + 2)
  (h3 : (a + 1) + (a + 2) > a)
  (h4 : (a + 2) + a > a + 1)
  (h5 : (a + 2)^2 > a^2 + (a + 1)^2) : 1 < a ∧ a < 3 :=
by
  -- proof omitted
  sorry

end obtuse_triangle_side_range_l865_86598


namespace sugar_water_inequality_acute_triangle_inequality_l865_86571

-- Part 1: Proving the inequality \(\frac{a}{b} < \frac{a+m}{b+m}\)
theorem sugar_water_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) :=
by
  sorry

-- Part 2: Proving the inequality in an acute triangle \(\triangle ABC\)
theorem acute_triangle_inequality (A B C : ℝ) (hA : A < B + C) (hB : B < C + A) (hC : C < A + B) : 
  (A / (B + C)) + (B / (C + A)) + (C / (A + B)) < 2 :=
by
  sorry

end sugar_water_inequality_acute_triangle_inequality_l865_86571


namespace combined_height_of_trees_is_correct_l865_86542

noncomputable def original_height_of_trees 
  (h1_current : ℝ) (h1_growth_rate : ℝ)
  (h2_current : ℝ) (h2_growth_rate : ℝ)
  (h3_current : ℝ) (h3_growth_rate : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  let h1 := h1_current / (1 + h1_growth_rate)
  let h2 := h2_current / (1 + h2_growth_rate)
  let h3 := h3_current / (1 + h3_growth_rate)
  (h1 + h2 + h3) / conversion_rate

theorem combined_height_of_trees_is_correct :
  original_height_of_trees 240 0.70 300 0.50 180 0.60 12 = 37.81 :=
by
  sorry

end combined_height_of_trees_is_correct_l865_86542


namespace total_gulbis_is_correct_l865_86562

-- Definitions based on given conditions
def num_dureums : ℕ := 156
def num_gulbis_in_one_dureum : ℕ := 20

-- Definition of total gulbis calculated
def total_gulbis : ℕ := num_dureums * num_gulbis_in_one_dureum

-- Statement to prove
theorem total_gulbis_is_correct : total_gulbis = 3120 := by
  -- The actual proof would go here
  sorry

end total_gulbis_is_correct_l865_86562


namespace area_of_original_triangle_l865_86581

variable (H : ℝ) (H' : ℝ := 0.65 * H) 
variable (A' : ℝ := 14.365)
variable (k : ℝ := 0.65) 
variable (A : ℝ)

theorem area_of_original_triangle (h₁ : H' = k * H) (h₂ : A' = 14.365) (h₃ : k = 0.65) : A = 34 := by
  sorry

end area_of_original_triangle_l865_86581


namespace max_sequence_length_l865_86597

theorem max_sequence_length (a : ℕ → ℝ) (n : ℕ)
  (H1 : ∀ k : ℕ, k + 4 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4)) < 0)
  (H2 : ∀ k : ℕ, k + 8 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8)) > 0) : 
  n ≤ 12 :=
sorry

end max_sequence_length_l865_86597


namespace smallest_number_l865_86534

theorem smallest_number (x : ℕ) (h1 : 2 * x = third) (h2 : 4 * x = second) (h3 : 7 * x = fourth) (h4 : (x + second + third + fourth) / 4 = 77) :
  x = 22 :=
by sorry

end smallest_number_l865_86534


namespace smallest_number_of_eggs_over_150_l865_86511

theorem smallest_number_of_eggs_over_150 
  (d : ℕ) 
  (h1: 12 * d - 3 > 150) 
  (h2: ∀ k < d, 12 * k - 3 ≤ 150) :
  12 * d - 3 = 153 :=
by
  sorry

end smallest_number_of_eggs_over_150_l865_86511


namespace maurice_age_l865_86530

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l865_86530


namespace veranda_area_l865_86518

/-- The width of the veranda on all sides of the room. -/
def width_of_veranda : ℝ := 2

/-- The length of the room. -/
def length_of_room : ℝ := 21

/-- The width of the room. -/
def width_of_room : ℝ := 12

/-- The area of the veranda given the conditions. -/
theorem veranda_area (length_of_room width_of_room width_of_veranda : ℝ) :
  (length_of_room + 2 * width_of_veranda) * (width_of_room + 2 * width_of_veranda) - length_of_room * width_of_room = 148 :=
by
  sorry

end veranda_area_l865_86518


namespace find_C_when_F_10_l865_86506

theorem find_C_when_F_10 : (∃ C : ℚ, ∀ F : ℚ, F = 10 → F = (9 / 5 : ℚ) * C + 32 → C = -110 / 9) :=
by
  sorry

end find_C_when_F_10_l865_86506


namespace moscow_probability_higher_l865_86551

def total_combinations : ℕ := 64 * 63

def invalid_combinations_ural : ℕ := 8 * 7 + 8 * 7

def valid_combinations_moscow : ℕ := total_combinations

def valid_combinations_ural : ℕ := total_combinations - invalid_combinations_ural

def probability_moscow : ℚ := valid_combinations_moscow / total_combinations

def probability_ural : ℚ := valid_combinations_ural / total_combinations

theorem moscow_probability_higher :
  probability_moscow > probability_ural :=
by
  unfold probability_moscow probability_ural
  unfold valid_combinations_moscow valid_combinations_ural invalid_combinations_ural total_combinations
  sorry

end moscow_probability_higher_l865_86551


namespace lesser_of_two_numbers_l865_86557

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l865_86557


namespace line_intersects_parabola_once_l865_86591

theorem line_intersects_parabola_once (k : ℝ) :
  (x = k)
  ∧ (x = -3 * y^2 - 4 * y + 7)
  ∧ (3 * y^2 + 4 * y + (k - 7)) = 0
  ∧ ((4)^2 - 4 * 3 * (k - 7) = 0)
  → k = 25 / 3 := 
by
  sorry

end line_intersects_parabola_once_l865_86591


namespace original_number_is_14_l865_86573

theorem original_number_is_14 (x : ℝ) (h : (2 * x + 2) / 3 = 10) : x = 14 := by
  sorry

end original_number_is_14_l865_86573


namespace xanthia_hot_dogs_l865_86582

theorem xanthia_hot_dogs (a b : ℕ) (h₁ : a = 5) (h₂ : b = 7) :
  ∃ n m : ℕ, n * a = m * b ∧ n = 7 := by 
sorry

end xanthia_hot_dogs_l865_86582


namespace necessary_but_not_sufficient_condition_l865_86532

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (h : ¬p) : p ∨ q ↔ true :=
by
  sorry

end necessary_but_not_sufficient_condition_l865_86532


namespace quadratic_sequence_l865_86596

theorem quadratic_sequence (a x₁ b x₂ c : ℝ)
  (h₁ : a + b = 2 * x₁)
  (h₂ : x₁ + x₂ = 2 * b)
  (h₃ : a + c = 2 * b)
  (h₄ : x₁ + x₂ = -6 / a)
  (h₅ : x₁ * x₂ = c / a) :
  b = -2 * a ∧ c = -5 * a :=
by
  sorry

end quadratic_sequence_l865_86596


namespace constant_term_in_first_equation_l865_86510

/-- Given the system of equations:
  1. 5x + y = C
  2. x + 3y = 1
  3. 3x + 2y = 10
  Prove that the constant term C is 19.
-/
theorem constant_term_in_first_equation
  (x y C : ℝ)
  (h1 : 5 * x + y = C)
  (h2 : x + 3 * y = 1)
  (h3 : 3 * x + 2 * y = 10) :
  C = 19 :=
by
  sorry

end constant_term_in_first_equation_l865_86510


namespace smallest_hot_dog_packages_l865_86559

theorem smallest_hot_dog_packages (d : ℕ) (b : ℕ) (hd : d = 10) (hb : b = 15) :
  ∃ n : ℕ, n * d = m * b ∧ n = 3 :=
by
  sorry

end smallest_hot_dog_packages_l865_86559


namespace min_cubes_needed_l865_86526

def minimum_cubes_for_views (front_view side_view : ℕ) : ℕ :=
  4

theorem min_cubes_needed (front_view_cond side_view_cond : ℕ) :
  front_view_cond = 2 ∧ side_view_cond = 3 → minimum_cubes_for_views front_view_cond side_view_cond = 4 :=
by
  intro h
  cases h
  -- Proving the condition based on provided views
  sorry

end min_cubes_needed_l865_86526


namespace hvac_cost_per_vent_l865_86549

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end hvac_cost_per_vent_l865_86549


namespace unique_reversible_six_digit_number_exists_l865_86550

theorem unique_reversible_six_digit_number_exists :
  ∃! (N : ℤ), 100000 ≤ N ∧ N < 1000000 ∧
  ∃ (f e d c b a : ℤ), 
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧ 
  9 * N = 100000 * f + 10000 * e + 1000 * d + 100 * c + 10 * b + a := 
sorry

end unique_reversible_six_digit_number_exists_l865_86550


namespace volume_ratio_remainder_520_l865_86578

noncomputable def simplex_ratio_mod : Nat :=
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000

theorem volume_ratio_remainder_520 :
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000 = 520 :=
by 
  sorry

end volume_ratio_remainder_520_l865_86578


namespace angelas_insects_l865_86569

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end angelas_insects_l865_86569


namespace pencil_pen_eraser_cost_l865_86589

-- Define the problem conditions and question
theorem pencil_pen_eraser_cost 
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 4.10)
  (h2 : 2 * p + 3 * q = 3.70) :
  p + q + 0.85 = 2.41 :=
sorry

end pencil_pen_eraser_cost_l865_86589


namespace minimum_workers_needed_l865_86500

noncomputable def units_per_first_worker : Nat := 48
noncomputable def units_per_second_worker : Nat := 32
noncomputable def units_per_third_worker : Nat := 28

def minimum_workers_first_process : Nat := 14
def minimum_workers_second_process : Nat := 21
def minimum_workers_third_process : Nat := 24

def lcm_3_nat (a b c : Nat) : Nat :=
  Nat.lcm (Nat.lcm a b) c

theorem minimum_workers_needed (a b c : Nat) (w1 w2 w3 : Nat)
  (h1 : a = 48) (h2 : b = 32) (h3 : c = 28)
  (hw1 : w1 = minimum_workers_first_process )
  (hw2 : w2 = minimum_workers_second_process )
  (hw3 : w3 = minimum_workers_third_process ) :
  lcm_3_nat a b c / a = w1 ∧ lcm_3_nat a b c / b = w2 ∧ lcm_3_nat a b c / c = w3 :=
by
  sorry

end minimum_workers_needed_l865_86500


namespace additional_cost_per_kg_l865_86548

theorem additional_cost_per_kg (l m : ℝ) 
  (h1 : 168 = 30 * l + 3 * m) 
  (h2 : 186 = 30 * l + 6 * m) 
  (h3 : 20 * l = 100) : 
  m = 6 := 
by
  sorry

end additional_cost_per_kg_l865_86548


namespace initial_pigs_l865_86574

theorem initial_pigs (x : ℕ) (h1 : x + 22 = 86) : x = 64 :=
by
  sorry

end initial_pigs_l865_86574


namespace score_order_l865_86552

-- Definitions that come from the problem conditions
variables (M Q S K : ℝ)
variables (hQK : Q = K) (hMK : M > K) (hSK : S < K)

-- The theorem to prove
theorem score_order (hQK : Q = K) (hMK : M > K) (hSK : S < K) : S < Q ∧ Q < M :=
by {
  sorry
}

end score_order_l865_86552


namespace convert_base_7_to_base_10_l865_86577

theorem convert_base_7_to_base_10 (n : ℕ) (h : n = 6 * 7^2 + 5 * 7^1 + 3 * 7^0) : n = 332 := by
  sorry

end convert_base_7_to_base_10_l865_86577


namespace sequence_sixth_term_l865_86521

theorem sequence_sixth_term :
  ∃ (a : ℕ → ℕ),
    a 1 = 3 ∧
    a 5 = 43 ∧
    (∀ n, a (n + 1) = (1/4) * (a n + a (n + 2))) →
    a 6 = 129 :=
sorry

end sequence_sixth_term_l865_86521


namespace sine_triangle_inequality_l865_86594

theorem sine_triangle_inequality 
  {a b c : ℝ} (h_triangle : a + b + c ≤ 2 * Real.pi) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (ha_lt_pi : a < Real.pi) (hb_lt_pi : b < Real.pi) (hc_lt_pi : c < Real.pi) :
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sine_triangle_inequality_l865_86594


namespace largest_negative_integer_is_neg_one_l865_86572

def is_negative_integer (n : Int) : Prop := n < 0

def is_largest_negative_integer (n : Int) : Prop := 
  is_negative_integer n ∧ ∀ m : Int, is_negative_integer m → m ≤ n

theorem largest_negative_integer_is_neg_one : 
  is_largest_negative_integer (-1) := by
  sorry

end largest_negative_integer_is_neg_one_l865_86572


namespace evaluate_expression_l865_86538

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a^a - a*(a-2)^a)^a = 1358954496 :=
by
  rw [h]  -- Substitute a with 4
  sorry

end evaluate_expression_l865_86538


namespace prove_problem_statement_l865_86560

noncomputable def problem_statement : Prop :=
  let E := (0, 0)
  let F := (2, 4)
  let G := (6, 2)
  let H := (7, 0)
  let line_through_E x y := y = -2 * x + 14
  let intersection_x := 37 / 8
  let intersection_y := 19 / 4
  let intersection_point := (intersection_x, intersection_y)
  let u := 37
  let v := 8
  let w := 19
  let z := 4
  u + v + w + z = 68

theorem prove_problem_statement : problem_statement :=
  sorry

end prove_problem_statement_l865_86560


namespace seats_in_hall_l865_86508

theorem seats_in_hall (S : ℝ) (h1 : 0.50 * S = 300) : S = 600 :=
by
  sorry

end seats_in_hall_l865_86508


namespace count_two_digit_even_congruent_to_1_mod_4_l865_86563

theorem count_two_digit_even_congruent_to_1_mod_4 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n % 4 = 1 ∧ 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0) ∧ S.card = 23 := 
sorry

end count_two_digit_even_congruent_to_1_mod_4_l865_86563


namespace digits_partition_impossible_l865_86524

theorem digits_partition_impossible : 
  ¬ ∃ (A B : Finset ℕ), 
    A.card = 4 ∧ B.card = 4 ∧ A ∪ B = {1, 2, 3, 4, 5, 7, 8, 9} ∧ A ∩ B = ∅ ∧ 
    A.sum id = B.sum id := 
by
  sorry

end digits_partition_impossible_l865_86524


namespace area_diff_circle_square_l865_86555

theorem area_diff_circle_square (s r : ℝ) (A_square A_circle : ℝ) (d : ℝ) (pi : ℝ) 
  (h1 : d = 8) -- diagonal of the square
  (h2 : d = 2 * r) -- diameter of the circle is 8, so radius is 4
  (h3 : s^2 + s^2 = d^2) -- Pythagorean Theorem for the square
  (h4 : A_square = s^2) -- area of the square
  (h5 : A_circle = pi * r^2) -- area of the circle
  (h6 : pi = 3.14159) -- approximation for π
  : abs (A_circle - A_square) - 18.3 < 0.1 := sorry

end area_diff_circle_square_l865_86555


namespace triangles_area_possibilities_unique_l865_86519

noncomputable def triangle_area_possibilities : ℕ :=
  -- Define lengths of segments on the first line
  let AB := 1
  let BC := 2
  let CD := 3
  -- Sum to get total lengths
  let AC := AB + BC -- 3
  let AD := AB + BC + CD -- 6
  -- Define length of the segment on the second line
  let EF := 2
  -- GH is a segment not parallel to the first two lines
  let GH := 1
  -- The number of unique possible triangle areas
  4

theorem triangles_area_possibilities_unique :
  triangle_area_possibilities = 4 := 
sorry

end triangles_area_possibilities_unique_l865_86519


namespace smallest_a_undefined_inverse_l865_86580

theorem smallest_a_undefined_inverse (a : ℕ) (ha : a = 2) :
  (∀ (a : ℕ), 0 < a → ((Nat.gcd a 40 > 1) ∧ (Nat.gcd a 90 > 1)) ↔ a = 2) :=
by
  sorry

end smallest_a_undefined_inverse_l865_86580


namespace sum_of_possible_values_l865_86541

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 24) : 
  ∃ x1 x2 : ℝ, (x1 + 3) * (x1 - 4) = 24 ∧ (x2 + 3) * (x2 - 4) = 24 ∧ x1 + x2 = 1 := 
by
  sorry

end sum_of_possible_values_l865_86541


namespace ball_draw_probability_red_is_one_ninth_l865_86525

theorem ball_draw_probability_red_is_one_ninth :
  let A_red := 4
  let A_white := 2
  let B_red := 1
  let B_white := 5
  let P_red_A := A_red / (A_red + A_white)
  let P_red_B := B_red / (B_red + B_white)
  P_red_A * P_red_B = 1 / 9 := by
    -- Proof here
    sorry

end ball_draw_probability_red_is_one_ninth_l865_86525


namespace percent_calculation_l865_86554

-- Given conditions
def part : ℝ := 120.5
def whole : ℝ := 80.75

-- Theorem statement
theorem percent_calculation : (part / whole) * 100 = 149.26 := 
sorry

end percent_calculation_l865_86554


namespace count_C_sets_l865_86517

-- Definitions of sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2}

-- The predicate that a set C satisfies B ∪ C = A
def satisfies_condition (C : Set ℕ) : Prop := B ∪ C = A

-- The claim that there are exactly 4 such sets C
theorem count_C_sets : 
  ∃ (C1 C2 C3 C4 : Set ℕ), 
    (satisfies_condition C1 ∧ satisfies_condition C2 ∧ satisfies_condition C3 ∧ satisfies_condition C4) 
    ∧ 
    (∀ C', satisfies_condition C' → C' = C1 ∨ C' = C2 ∨ C' = C3 ∨ C' = C4)
    ∧ 
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C1 ≠ C4 ∧ C2 ≠ C3 ∧ C2 ≠ C4 ∧ C3 ≠ C4) := 
sorry

end count_C_sets_l865_86517


namespace updated_mean_l865_86567

theorem updated_mean (n : ℕ) (observation_mean decrement : ℕ) 
  (h1 : n = 50) (h2 : observation_mean = 200) (h3 : decrement = 15) : 
  ((observation_mean * n - decrement * n) / n = 185) :=
by
  sorry

end updated_mean_l865_86567


namespace quadratic_roots_ratio_l865_86512

theorem quadratic_roots_ratio (p x1 x2 : ℝ) (h_eq : x1^2 + p * x1 - 16 = 0) (h_ratio : x1 / x2 = -4) :
  p = 6 ∨ p = -6 :=
by {
  sorry
}

end quadratic_roots_ratio_l865_86512


namespace f_increasing_f_odd_zero_l865_86584

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Prove that f(x) is always an increasing function for any real a.
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

-- 2. Determine the value of a such that f(-x) + f(x) = 0 always holds.
theorem f_odd_zero (a : ℝ) : (∀ x : ℝ, f a (-x) + f a x = 0) → a = 1 :=
by
  sorry

end f_increasing_f_odd_zero_l865_86584


namespace population_percentage_l865_86535

-- Definitions based on the given conditions
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Conditions from the problem statement
def part_population : ℕ := 23040
def total_population : ℕ := 25600

-- The theorem stating that the percentage is 90
theorem population_percentage : percentage part_population total_population = 90 :=
  by
    -- Proof steps would go here, we only need to state the theorem
    sorry

end population_percentage_l865_86535


namespace isosceles_triangle_height_l865_86529

theorem isosceles_triangle_height (l w h : ℝ) 
  (h1 : l * w = (1 / 2) * w * h) : h = 2 * l :=
by
  sorry

end isosceles_triangle_height_l865_86529


namespace inverse_matrix_correct_l865_86590

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1, 2, 3],
    ![0, -1, 2],
    ![3, 0, 7]
  ]

def A_inv_correct : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![-1/2, -1, 1/2],
    ![3/7, -1/7, -1/7],
    ![3/14, 3/7, -1/14]
  ]

theorem inverse_matrix_correct : A⁻¹ = A_inv_correct := by
  sorry

end inverse_matrix_correct_l865_86590


namespace brown_house_number_l865_86587

-- Defining the problem conditions
def sum_arithmetic_series (k : ℕ) := k * (k + 1) / 2

theorem brown_house_number (t n : ℕ) (h1 : 20 < t) (h2 : t < 500)
    (h3 : sum_arithmetic_series n = sum_arithmetic_series t / 2) : n = 84 := by
  sorry

end brown_house_number_l865_86587


namespace car_robot_collections_l865_86583

variable (t m b s j : ℕ)

axiom tom_has_15 : t = 15
axiom michael_robots : m = 3 * t - 5
axiom bob_robots : b = 8 * (t + m)
axiom sarah_robots : s = b / 2 - 7
axiom jane_robots : j = (s - t) / 3

theorem car_robot_collections :
  t = 15 ∧
  m = 40 ∧
  b = 440 ∧
  s = 213 ∧
  j = 66 :=
  by
    sorry

end car_robot_collections_l865_86583


namespace no_real_root_for_3_in_g_l865_86544

noncomputable def g (x c : ℝ) : ℝ := x^2 + 3 * x + c

theorem no_real_root_for_3_in_g (c : ℝ) :
  (21 - 4 * c) < 0 ↔ c > 21 / 4 := by
sorry

end no_real_root_for_3_in_g_l865_86544


namespace min_value_x2_y2_z2_l865_86503

theorem min_value_x2_y2_z2 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + 2 * y + 3 * z = 2) : 
  x^2 + y^2 + z^2 ≥ 2 / 7 :=
sorry

end min_value_x2_y2_z2_l865_86503


namespace find_y1_l865_86545

theorem find_y1
  (y1 y2 y3 : ℝ)
  (h1 : 0 ≤ y3)
  (h2 : y3 ≤ y2)
  (h3 : y2 ≤ y1)
  (h4 : y1 ≤ 1)
  (h5 : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = 3 / 4 :=
sorry

end find_y1_l865_86545


namespace cricket_run_rate_l865_86593

theorem cricket_run_rate
  (run_rate_first_10_overs : ℝ)
  (overs_first_10_overs : ℕ)
  (target_runs : ℕ)
  (remaining_overs : ℕ)
  (run_rate_required : ℝ) :
  run_rate_first_10_overs = 3.2 →
  overs_first_10_overs = 10 →
  target_runs = 242 →
  remaining_overs = 40 →
  run_rate_required = 5.25 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) = 210 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) / remaining_overs = run_rate_required :=
by
  sorry

end cricket_run_rate_l865_86593


namespace polynomial_solution_l865_86513
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end polynomial_solution_l865_86513


namespace emily_sixth_quiz_score_l865_86561

-- Define the scores Emily has received
def scores : List ℕ := [92, 96, 87, 89, 100]

-- Define the number of quizzes
def num_quizzes : ℕ := 6

-- Define the desired average score
def desired_average : ℕ := 94

-- The theorem to prove the score Emily needs on her sixth quiz to achieve the desired average
theorem emily_sixth_quiz_score : ∃ (x : ℕ), List.sum scores + x = desired_average * num_quizzes := by
  sorry

end emily_sixth_quiz_score_l865_86561


namespace not_age_of_child_l865_86516

noncomputable def sum_from_1_to_n (n : ℕ) := n * (n + 1) / 2

theorem not_age_of_child (N : ℕ) (S : Finset ℕ) (a b : ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11} ∧
  N = 1100 * a + 11 * b ∧
  a ≠ b ∧
  N ≥ 1000 ∧ N < 10000 ∧
  ((S.sum id) = N) ∧
  (∀ age ∈ S, N % age = 0) →
  10 ∉ S := 
by
  sorry

end not_age_of_child_l865_86516


namespace sufficient_but_not_necessary_condition_l865_86540

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 2 ∧ y = -1) → (x + y - 1 = 0) ∧ ¬(∀ x y, x + y - 1 = 0 → (x = 2 ∧ y = -1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l865_86540


namespace ratio_pea_patch_to_radish_patch_l865_86537

-- Definitions
def sixth_of_pea_patch : ℝ := 5
def whole_radish_patch : ℝ := 15

-- Theorem to prove
theorem ratio_pea_patch_to_radish_patch :
  (6 * sixth_of_pea_patch) / whole_radish_patch = 2 :=
by 
  -- skip the actual proof since it's not required
  sorry

end ratio_pea_patch_to_radish_patch_l865_86537


namespace sin_sum_diff_l865_86585

theorem sin_sum_diff (α β : ℝ) 
  (hα : Real.sin α = 1/3) 
  (hβ : Real.sin β = 1/2) : 
  Real.sin (α + β) * Real.sin (α - β) = -5/36 := 
sorry

end sin_sum_diff_l865_86585


namespace sequence_length_div_by_four_l865_86564

theorem sequence_length_div_by_four (a : ℕ) (h0 : a = 11664) (H : ∀ n, a = (4 ^ n) * b → b ≠ 0 ∧ n ≤ 3) : 
  ∃ n, n + 1 = 4 :=
by
  sorry

end sequence_length_div_by_four_l865_86564


namespace value_of_x_l865_86599

theorem value_of_x (x : ℝ) (h₁ : x > 0) (h₂ : x^3 = 19683) : x = 27 :=
sorry

end value_of_x_l865_86599


namespace percentage_of_x_l865_86523

variable {x y : ℝ}
variable {P : ℝ}

theorem percentage_of_x (h1 : (P / 100) * x = (20 / 100) * y) (h2 : x / y = 2) : P = 10 := by
  sorry

end percentage_of_x_l865_86523


namespace pencils_initial_count_l865_86595

theorem pencils_initial_count (pencils_given : ℕ) (pencils_left : ℕ) (initial_pencils : ℕ) :
  pencils_given = 31 → pencils_left = 111 → initial_pencils = pencils_given + pencils_left → initial_pencils = 142 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_initial_count_l865_86595


namespace shaded_region_area_l865_86528

variables (a b : ℕ) 
variable (A : Type) 

def AD := 5
def CD := 2
def semi_major_axis := 6
def semi_minor_axis := 4

noncomputable def area_ellipse := Real.pi * semi_major_axis * semi_minor_axis
noncomputable def area_rectangle := AD * CD
noncomputable def area_shaded_region := area_ellipse - area_rectangle

theorem shaded_region_area : area_shaded_region = 24 * Real.pi - 10 :=
by {
  sorry
}

end shaded_region_area_l865_86528


namespace problem1_problem2_l865_86566

-- First Problem
theorem problem1 : 
  Real.cos (Real.pi / 3) + Real.sin (Real.pi / 4) - Real.tan (Real.pi / 4) = (-1 + Real.sqrt 2) / 2 :=
by
  sorry

-- Second Problem
theorem problem2 : 
  6 * (Real.tan (Real.pi / 6))^2 - Real.sqrt 3 * Real.sin (Real.pi / 3) - 2 * Real.cos (Real.pi / 4) = 1 / 2 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l865_86566


namespace find_asymptote_slope_l865_86565

theorem find_asymptote_slope (x y : ℝ) (h : (y^2) / 9 - (x^2) / 4 = 1) : y = 3 / 2 * x :=
sorry

end find_asymptote_slope_l865_86565


namespace sum_of_interior_angles_l865_86547

theorem sum_of_interior_angles (n : ℕ) (h₁ : 180 * (n - 2) = 2340) : 
  180 * ((n - 3) - 2) = 1800 := by
  -- Here, we'll solve the theorem using Lean's capabilities.
  sorry

end sum_of_interior_angles_l865_86547


namespace fraction_cows_sold_is_one_fourth_l865_86527

def num_cows : ℕ := 184
def num_dogs (C : ℕ) : ℕ := C / 2
def remaining_animals : ℕ := 161
def fraction_dogs_sold : ℚ := 3 / 4
def fraction_cows_sold (C remaining_cows : ℕ) : ℚ := (C - remaining_cows) / C

theorem fraction_cows_sold_is_one_fourth :
  ∀ (C remaining_dogs remaining_cows: ℕ),
    C = 184 →
    remaining_animals = 161 →
    remaining_dogs = (1 - fraction_dogs_sold) * num_dogs C →
    remaining_cows = remaining_animals - remaining_dogs →
    fraction_cows_sold C remaining_cows = 1 / 4 :=
by sorry

end fraction_cows_sold_is_one_fourth_l865_86527


namespace charlie_acorns_l865_86522

theorem charlie_acorns (x y : ℕ) (hc hs : ℕ)
  (h5 : x = 5 * hc)
  (h7 : y = 7 * hs)
  (total : x + y = 145)
  (holes : hs = hc - 3) :
  x = 70 :=
by
  sorry

end charlie_acorns_l865_86522


namespace monotonic_criteria_l865_86558

noncomputable def monotonic_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 4 → 
  (-2 * x₁^2 + m * x₁ + 1) ≤ (-2 * x₂^2 + m * x₂ + 1)

theorem monotonic_criteria (m : ℝ) : 
  (m ≤ -4 ∨ m ≥ 16) ↔ monotonic_interval m := 
sorry

end monotonic_criteria_l865_86558


namespace cubic_roots_solution_sum_l865_86514

theorem cubic_roots_solution_sum (u v w : ℝ) (h1 : (u - 2) * (u - 3) * (u - 4) = 1 / 2)
                                     (h2 : (v - 2) * (v - 3) * (v - 4) = 1 / 2)
                                     (h3 : (w - 2) * (w - 3) * (w - 4) = 1 / 2)
                                     (distinct_roots : u ≠ v ∧ v ≠ w ∧ u ≠ w) :
  u^3 + v^3 + w^3 = -42 :=
sorry

end cubic_roots_solution_sum_l865_86514


namespace percentage_increase_l865_86543

variable {α : Type} [LinearOrderedField α]

theorem percentage_increase (x y : α) (h : x = 0.5 * y) : y = x + x :=
by
  -- The steps of the proof are omitted and 'sorry' is used to skip actual proof.
  sorry

end percentage_increase_l865_86543


namespace candy_cost_l865_86575

theorem candy_cost
    (grape_candies : ℕ)
    (cherry_candies : ℕ)
    (apple_candies : ℕ)
    (total_cost : ℝ)
    (total_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 24)
    (h2 : grape_candies = 3 * cherry_candies)
    (h3 : apple_candies = 2 * grape_candies)
    (h4 : total_cost = 200)
    (h5 : total_candies = cherry_candies + grape_candies + apple_candies)
    (h6 : cost_per_candy = total_cost / total_candies) :
    cost_per_candy = 2.50 :=
by
    sorry

end candy_cost_l865_86575


namespace triangle_is_right_angle_l865_86576

theorem triangle_is_right_angle (A B C : ℝ) : 
  (A / B = 2 / 3) ∧ (A / C = 2 / 5) ∧ (A + B + C = 180) →
  (A = 36) ∧ (B = 54) ∧ (C = 90) :=
by 
  intro h
  sorry

end triangle_is_right_angle_l865_86576


namespace susan_backward_spaces_l865_86531

variable (spaces_to_win total_spaces : ℕ)
variables (first_turn second_turn_forward second_turn_back third_turn : ℕ)

theorem susan_backward_spaces :
  ∀ (total_spaces first_turn second_turn_forward second_turn_back third_turn win_left : ℕ),
  total_spaces = 48 →
  first_turn = 8 →
  second_turn_forward = 2 →
  third_turn = 6 →
  win_left = 37 →
  first_turn + second_turn_forward + third_turn - second_turn_back + win_left = total_spaces →
  second_turn_back = 6 :=
by
  intros total_spaces first_turn second_turn_forward second_turn_back third_turn win_left
  intros h_total h_first h_second_forward h_third h_win h_eq
  rw [h_total, h_first, h_second_forward, h_third, h_win] at h_eq
  sorry

end susan_backward_spaces_l865_86531


namespace sum_geometric_sequence_l865_86592

theorem sum_geometric_sequence {a : ℕ → ℝ} (ha : ∃ q, ∀ n, a n = 3 * q ^ n)
  (h1 : a 1 = 3) (h2 : a 1 + a 2 + a 3 = 9) :
  a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72 :=
sorry

end sum_geometric_sequence_l865_86592


namespace ratio_of_apple_to_orange_cost_l865_86536

-- Define the costs of fruits based on the given conditions.
def cost_per_kg_oranges : ℝ := 12
def cost_per_kg_apples : ℝ := 2

-- The theorem to prove.
theorem ratio_of_apple_to_orange_cost : cost_per_kg_apples / cost_per_kg_oranges = 1 / 6 :=
by
  sorry

end ratio_of_apple_to_orange_cost_l865_86536


namespace find_a_l865_86507

variable (a : ℤ) -- We assume a is an integer for simplicity

def point_on_x_axis (P : Nat × ℤ) : Prop :=
  P.snd = 0

theorem find_a (h : point_on_x_axis (4, 2 * a + 6)) : a = -3 :=
by
  sorry

end find_a_l865_86507


namespace simplify_expression_l865_86546

theorem simplify_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := 
by 
  sorry 

end simplify_expression_l865_86546


namespace denominator_of_speed_l865_86568

theorem denominator_of_speed (h : 0.8 = 8 / d * 3600 / 1000) : d = 36 := 
by
  sorry

end denominator_of_speed_l865_86568


namespace tina_made_more_140_dollars_l865_86505

def candy_bars_cost : ℕ := 2
def marvin_candy_bars : ℕ := 35
def tina_candy_bars : ℕ := 3 * marvin_candy_bars
def marvin_money : ℕ := marvin_candy_bars * candy_bars_cost
def tina_money : ℕ := tina_candy_bars * candy_bars_cost
def tina_extra_money : ℕ := tina_money - marvin_money

theorem tina_made_more_140_dollars :
  tina_extra_money = 140 := by
  sorry

end tina_made_more_140_dollars_l865_86505


namespace positive_integer_solutions_l865_86515

theorem positive_integer_solutions
  (m n k : ℕ)
  (hm : 0 < m) (hn : 0 < n) (hk : 0 < k) :
  3 * m + 4 * n = 5 * k ↔ (m = 1 ∧ n = 2 ∧ k = 2) := 
by
  sorry

end positive_integer_solutions_l865_86515


namespace num_points_satisfying_inequalities_l865_86502

theorem num_points_satisfying_inequalities :
  ∃ (n : ℕ), n = 2551 ∧
  ∀ (x y : ℤ), (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100) → 
              ∃ (p : ℕ), p = n := 
by
  sorry

end num_points_satisfying_inequalities_l865_86502
