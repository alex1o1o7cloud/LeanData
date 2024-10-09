import Mathlib

namespace verna_sherry_total_weight_l694_69464

theorem verna_sherry_total_weight (haley verna sherry : ℕ)
  (h1 : verna = haley + 17)
  (h2 : verna = sherry / 2)
  (h3 : haley = 103) :
  verna + sherry = 360 :=
by
  sorry

end verna_sherry_total_weight_l694_69464


namespace line_intersects_circle_l694_69429

theorem line_intersects_circle
    (r : ℝ) (d : ℝ)
    (hr : r = 6) (hd : d = 5) : d < r :=
by
    rw [hr, hd]
    exact by norm_num

end line_intersects_circle_l694_69429


namespace smallest_positive_q_with_property_l694_69444

theorem smallest_positive_q_with_property :
  ∃ q : ℕ, (
    q > 0 ∧
    ∀ m : ℕ, (1 ≤ m ∧ m ≤ 1006) →
    ∃ n : ℤ, 
      (m * q : ℤ) / 1007 < n ∧
      (m + 1) * q / 1008 > n) ∧
   q = 2015 := 
sorry

end smallest_positive_q_with_property_l694_69444


namespace cubical_storage_unit_blocks_l694_69436

theorem cubical_storage_unit_blocks :
  let side_length := 8
  let thickness := 1
  let total_volume := side_length ^ 3
  let interior_side_length := side_length - 2 * thickness
  let interior_volume := interior_side_length ^ 3
  let blocks_required := total_volume - interior_volume
  blocks_required = 296 := by
    sorry

end cubical_storage_unit_blocks_l694_69436


namespace calculate_fg3_l694_69440

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l694_69440


namespace mat_weavers_equiv_l694_69484

theorem mat_weavers_equiv {x : ℕ} 
  (h1 : 4 * 1 = 4) 
  (h2 : 16 * (64 / 16) = 64) 
  (h3 : 1 = 64 / (16 * x)) : x = 4 :=
by
  sorry

end mat_weavers_equiv_l694_69484


namespace find_range_of_a_l694_69477

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem find_range_of_a (p q a : ℝ) (h : 0 < a) (hpq : p < q) :
  (∀ x : ℝ, 0 < x → x ∈ Set.Icc p q → f a x ≤ 0) → 
  (0 < a ∧ a < 1 / Real.exp 1) :=
by
  sorry

end find_range_of_a_l694_69477


namespace ball_returns_to_Ben_after_three_throws_l694_69457

def circle_throw (n : ℕ) (skip : ℕ) (start : ℕ) : ℕ :=
  (start + skip) % n

theorem ball_returns_to_Ben_after_three_throws :
  ∀ (n : ℕ) (skip : ℕ) (start : ℕ),
  n = 15 → skip = 5 → start = 1 →
  (circle_throw n skip (circle_throw n skip (circle_throw n skip start))) = start :=
by
  intros n skip start hn hskip hstart
  sorry

end ball_returns_to_Ben_after_three_throws_l694_69457


namespace total_rainfall_cm_l694_69452

theorem total_rainfall_cm :
  let monday := 0.12962962962962962
  let tuesday := 3.5185185185185186 * 0.1
  let wednesday := 0.09259259259259259
  let thursday := 0.10222222222222223 * 2.54
  let friday := 12.222222222222221 * 0.1
  let saturday := 0.2222222222222222
  let sunday := 0.17444444444444446 * 2.54
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 2.721212629851652 :=
by
  sorry

end total_rainfall_cm_l694_69452


namespace magicStack_cardCount_l694_69495

-- Define the conditions and question based on a)
def isMagicStack (n : ℕ) : Prop :=
  let totalCards := 2 * n
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range totalCards) ∧
    (∀ x ∈ A, x < n) ∧ (∀ x ∈ B, x ≥ n) ∧
    (∀ i ∈ A, i % 2 = 1) ∧ (∀ j ∈ B, j % 2 = 0) ∧
    (151 ∈ A) ∧
    ∃ (newStack : Finset ℕ), (newStack = A ∪ B) ∧
    (∀ k ∈ newStack, k ∈ A ∨ k ∈ B) ∧
    (151 = 151)

-- The theorem that states the number of cards, when card 151 retains its position, is 452.
theorem magicStack_cardCount :
  isMagicStack 226 → 2 * 226 = 452 :=
by
  sorry

end magicStack_cardCount_l694_69495


namespace certain_number_l694_69448

theorem certain_number (a n b : ℕ) (h1 : a = 30) (h2 : a * n = b^2) (h3 : ∀ m : ℕ, (m * n = b^2 → a ≤ m)) :
  n = 30 :=
by
  sorry

end certain_number_l694_69448


namespace number_before_star_is_five_l694_69486

theorem number_before_star_is_five (n : ℕ) (h1 : n % 72 = 0) (h2 : n % 10 = 0) (h3 : ∃ k, n = 400 + 10 * k) : (n / 10) % 10 = 5 :=
sorry

end number_before_star_is_five_l694_69486


namespace fraction_addition_l694_69435

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l694_69435


namespace milkshakes_more_than_ice_cream_cones_l694_69426

def ice_cream_cones_sold : ℕ := 67
def milkshakes_sold : ℕ := 82

theorem milkshakes_more_than_ice_cream_cones : milkshakes_sold - ice_cream_cones_sold = 15 := by
  sorry

end milkshakes_more_than_ice_cream_cones_l694_69426


namespace haley_initial_shirts_l694_69475

-- Defining the conditions
def returned_shirts := 6
def endup_shirts := 5

-- The theorem statement
theorem haley_initial_shirts : returned_shirts + endup_shirts = 11 := by 
  sorry

end haley_initial_shirts_l694_69475


namespace scientific_notation_of_16907_l694_69466

theorem scientific_notation_of_16907 :
  16907 = 1.6907 * 10^4 :=
sorry

end scientific_notation_of_16907_l694_69466


namespace probability_after_50_bell_rings_l694_69471

noncomputable def game_probability : ℝ :=
  let p_keep_money := (1 : ℝ) / 4
  let p_give_money := (3 : ℝ) / 4
  let p_same_distribution := p_keep_money^3 + 2 * p_give_money^3
  p_same_distribution^50

theorem probability_after_50_bell_rings : abs (game_probability - 0.002) < 0.01 :=
by
  sorry

end probability_after_50_bell_rings_l694_69471


namespace smallest_pos_integer_for_frac_reducible_l694_69408

theorem smallest_pos_integer_for_frac_reducible :
  ∃ n : ℕ, n > 0 ∧ ∃ d > 1, d ∣ (n - 17) ∧ d ∣ (6 * n + 8) ∧ n = 127 :=
by
  sorry

end smallest_pos_integer_for_frac_reducible_l694_69408


namespace determinant_value_l694_69476

-- Given definitions and conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c
def special_determinant (m : ℤ) : ℤ := determinant (m^2) (m-3) (1-2*m) (m-2)

-- The proof problem
theorem determinant_value (m : ℤ) (h : m^2 - 2 * m - 3 = 0) : special_determinant m = 9 := sorry

end determinant_value_l694_69476


namespace find_number_satisfying_equation_l694_69439

theorem find_number_satisfying_equation :
  ∃ x : ℝ, (196 * x^3) / 568 = 43.13380281690141 ∧ x = 5 :=
by
  sorry

end find_number_satisfying_equation_l694_69439


namespace tg_arccos_le_cos_arctg_l694_69411

theorem tg_arccos_le_cos_arctg (x : ℝ) (h₀ : -1 ≤ x ∧ x ≤ 1) :
  (Real.tan (Real.arccos x) ≤ Real.cos (Real.arctan x)) → 
  (x ∈ Set.Icc (-1:ℝ) 0 ∨ x ∈ Set.Icc (Real.sqrt ((Real.sqrt 5 - 1) / 2)) 1) :=
by
  sorry

end tg_arccos_le_cos_arctg_l694_69411


namespace product_of_first_four_consecutive_primes_l694_69497

theorem product_of_first_four_consecutive_primes : 
  (2 * 3 * 5 * 7) = 210 :=
by
  sorry

end product_of_first_four_consecutive_primes_l694_69497


namespace value_to_be_subtracted_l694_69494

theorem value_to_be_subtracted (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 24) / 10 = 3) : x = 5 := by
  sorry

end value_to_be_subtracted_l694_69494


namespace not_both_zero_l694_69468

theorem not_both_zero (x y : ℝ) (h : x^2 + y^2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
by {
  sorry
}

end not_both_zero_l694_69468


namespace gcd_143_117_l694_69465

theorem gcd_143_117 : Nat.gcd 143 117 = 13 :=
by
  have h1 : 143 = 11 * 13 := by rfl
  have h2 : 117 = 9 * 13 := by rfl
  sorry

end gcd_143_117_l694_69465


namespace T_5_3_l694_69419

def T (x y : ℕ) : ℕ := 4 * x + 5 * y + x * y

theorem T_5_3 : T 5 3 = 50 :=
by
  sorry

end T_5_3_l694_69419


namespace jack_jill_meeting_distance_l694_69437

-- Definitions for Jack's and Jill's initial conditions
def jack_speed_uphill := 12 -- km/hr
def jack_speed_downhill := 15 -- km/hr
def jill_speed_uphill := 14 -- km/hr
def jill_speed_downhill := 18 -- km/hr

def head_start := 0.2 -- hours
def total_distance := 12 -- km
def turn_point_distance := 7 -- km
def return_distance := 5 -- km

-- Statement of the problem to prove the distance from the turning point where they meet
theorem jack_jill_meeting_distance :
  let jack_time_to_turn := (turn_point_distance : ℚ) / jack_speed_uphill
  let jill_time_to_turn := (turn_point_distance : ℚ) / jill_speed_uphill
  let x_meet := (8.95 : ℚ) / 29
  7 - (14 * ((x_meet - 0.2) / 1)) = (772 / 145 : ℚ) := 
sorry

end jack_jill_meeting_distance_l694_69437


namespace tangent_slope_at_one_l694_69432

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_slope_at_one : deriv f 1 = 2 * Real.exp 1 := sorry

end tangent_slope_at_one_l694_69432


namespace pumpkins_eaten_l694_69488

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end pumpkins_eaten_l694_69488


namespace problem_1_problem_2_l694_69409

theorem problem_1 (a b c: ℝ) (h1: a > 0) (h2: b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

theorem problem_2 (a b c: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_1_problem_2_l694_69409


namespace find_number_l694_69418

theorem find_number (x : ℝ) (h : 0.3 * x + 0.1 * 0.5 = 0.29) : x = 0.8 :=
by
  sorry

end find_number_l694_69418


namespace sum_of_squares_first_20_l694_69491

-- Define the sum of squares function
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Specific problem instance
theorem sum_of_squares_first_20 : sum_of_squares 20 = 5740 :=
  by
  -- Proof skipping placeholder
  sorry

end sum_of_squares_first_20_l694_69491


namespace exists_unique_pair_l694_69490

theorem exists_unique_pair (X : Set ℤ) :
  (∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n) :=
sorry

end exists_unique_pair_l694_69490


namespace largest_divisor_of_m_l694_69449

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 36 ∣ m :=
sorry

end largest_divisor_of_m_l694_69449


namespace problem1_problem2_l694_69400

-- Problem 1: Prove the range of k for any real number x
theorem problem1 (k : ℝ) (x : ℝ) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  1 ≤ k ∧ k < 13 :=
sorry

-- Problem 2: Prove the range of k for any x in the interval (0, 1]
theorem problem2 (k : ℝ) (x : ℝ) (hx : 0 < x) (hx1 : x ≤ 1) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  k > -1/2 :=
sorry

end problem1_problem2_l694_69400


namespace parallel_lines_slope_equality_l694_69451

theorem parallel_lines_slope_equality (m : ℝ) : (∀ x y : ℝ, 3 * x + y - 3 = 0) ∧ (∀ x y : ℝ, 6 * x + m * y + 1 = 0) → m = 2 :=
by 
  sorry

end parallel_lines_slope_equality_l694_69451


namespace sqrt_multiplication_division_l694_69487

theorem sqrt_multiplication_division :
  Real.sqrt 27 * Real.sqrt (8 / 3) / Real.sqrt (1 / 2) = 18 :=
by
  sorry

end sqrt_multiplication_division_l694_69487


namespace range_of_a_l694_69461

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem range_of_a {a : ℝ} (h : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  a > 1/7 ∧ a < 1/3 :=
sorry

end range_of_a_l694_69461


namespace simplify_expression_l694_69469

variable (x y z : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hne : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := 
by 
  sorry

end simplify_expression_l694_69469


namespace apples_on_tree_now_l694_69406

-- Definitions based on conditions
def initial_apples : ℕ := 11
def apples_picked : ℕ := 7
def new_apples : ℕ := 2

-- Theorem statement proving the final number of apples on the tree
theorem apples_on_tree_now : initial_apples - apples_picked + new_apples = 6 := 
by 
  sorry

end apples_on_tree_now_l694_69406


namespace taxi_ride_cost_l694_69496

theorem taxi_ride_cost :
  let base_fare : ℝ := 2.00
  let cost_per_mile_first_3 : ℝ := 0.30
  let cost_per_mile_additional : ℝ := 0.40
  let total_distance : ℕ := 8
  let first_3_miles_cost : ℝ := base_fare + 3 * cost_per_mile_first_3
  let additional_miles_cost : ℝ := (total_distance - 3) * cost_per_mile_additional
  let total_cost : ℝ := first_3_miles_cost + additional_miles_cost
  total_cost = 4.90 :=
by
  sorry

end taxi_ride_cost_l694_69496


namespace largest_three_digit_integer_l694_69413

theorem largest_three_digit_integer (n : ℕ) :
  75 * n ≡ 300 [MOD 450] →
  n < 1000 →
  ∃ m : ℕ, n = m ∧ (∀ k : ℕ, 75 * k ≡ 300 [MOD 450] ∧ k < 1000 → k ≤ n) := by
  sorry

end largest_three_digit_integer_l694_69413


namespace ice_cream_scoops_l694_69478

theorem ice_cream_scoops (total_money : ℝ) (spent_on_restaurant : ℝ) (remaining_money : ℝ) 
  (cost_per_scoop_after_discount : ℝ) (remaining_each : ℝ) 
  (initial_savings : ℝ) (service_charge_percent : ℝ) (restaurant_percent : ℝ) 
  (ice_cream_discount_percent : ℝ) (money_each : ℝ) :
  total_money = 400 ∧
  spent_on_restaurant = 320 ∧
  remaining_money = 80 ∧
  cost_per_scoop_after_discount = 5 ∧
  remaining_each = 8 ∧
  initial_savings = 200 ∧
  service_charge_percent = 0.20 ∧
  restaurant_percent = 0.80 ∧
  ice_cream_discount_percent = 0.10 ∧
  money_each = 5 → 
  ∃ (scoops_per_person : ℕ), scoops_per_person = 5 :=
by
  sorry

end ice_cream_scoops_l694_69478


namespace average_fish_per_person_l694_69454

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l694_69454


namespace none_of_these_true_l694_69424

def op_star (a b : ℕ) := b ^ a -- Define the binary operation

theorem none_of_these_true :
  ¬ (∀ a b : ℕ, 0 < a ∧ 0 < b → op_star a b = op_star b a) ∧
  ¬ (∀ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c → op_star a (op_star b c) = op_star (op_star a b) c) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → (op_star a b) ^ n = op_star n (op_star a b)) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → op_star a (b ^ n) = op_star n (op_star b a)) :=
sorry

end none_of_these_true_l694_69424


namespace tan_alpha_second_quadrant_l694_69416

theorem tan_alpha_second_quadrant (α : ℝ) 
(h_cos : Real.cos α = -4/5) 
(h_quadrant : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 :=
by
  sorry

end tan_alpha_second_quadrant_l694_69416


namespace quotient_when_divided_by_44_l694_69425

theorem quotient_when_divided_by_44 :
  ∃ N Q : ℕ, (N % 44 = 0) ∧ (N % 39 = 15) ∧ (N / 44 = Q) ∧ (Q = 3) :=
by {
  sorry
}

end quotient_when_divided_by_44_l694_69425


namespace sin_minus_cos_third_quadrant_l694_69423

theorem sin_minus_cos_third_quadrant (α : ℝ) (h_tan : Real.tan α = 2) (h_quadrant : π < α ∧ α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 := 
by 
  sorry

end sin_minus_cos_third_quadrant_l694_69423


namespace find_f_107_5_l694_69456

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x, f x = f (-x)
axiom func_eq : ∀ x, f (x + 3) = - (1 / f x)
axiom cond_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x

theorem find_f_107_5 : f 107.5 = 1 / 10 := by {
  sorry
}

end find_f_107_5_l694_69456


namespace eccentricity_of_ellipse_l694_69462

theorem eccentricity_of_ellipse :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 4 :=
by
  sorry

end eccentricity_of_ellipse_l694_69462


namespace gcd_pow_sub_l694_69441

theorem gcd_pow_sub (m n : ℕ) (h₁ : m = 2 ^ 2000 - 1) (h₂ : n = 2 ^ 1990 - 1) :
  Nat.gcd m n = 1023 :=
sorry

end gcd_pow_sub_l694_69441


namespace pq_false_l694_69407

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x > 3 ↔ x^2 > 9
def q (a b : ℝ) : Prop := a^2 > b^2 ↔ a > b

-- Theorem to prove that p ∨ q is false given the conditions
theorem pq_false (x a b : ℝ) (hp : ¬ p x) (hq : ¬ q a b) : ¬ (p x ∨ q a b) :=
by
  sorry

end pq_false_l694_69407


namespace abs_inequality_interval_notation_l694_69474

variable (x : ℝ)

theorem abs_inequality_interval_notation :
  {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end abs_inequality_interval_notation_l694_69474


namespace situps_ratio_l694_69499

theorem situps_ratio (ken_situps : ℕ) (nathan_situps : ℕ) (bob_situps : ℕ) :
  ken_situps = 20 →
  nathan_situps = 2 * ken_situps →
  bob_situps = ken_situps + 10 →
  (bob_situps : ℚ) / (ken_situps + nathan_situps : ℚ) = 1 / 2 :=
by
  sorry

end situps_ratio_l694_69499


namespace is_positive_integer_iff_l694_69434

theorem is_positive_integer_iff (p : ℕ) : 
  (p > 0 → ∃ k : ℕ, (4 * p + 17 = k * (3 * p - 7))) ↔ (3 ≤ p ∧ p ≤ 40) := 
sorry

end is_positive_integer_iff_l694_69434


namespace earrings_cost_l694_69442

theorem earrings_cost (initial_savings necklace_cost remaining_savings : ℕ) 
  (h_initial : initial_savings = 80) 
  (h_necklace : necklace_cost = 48) 
  (h_remaining : remaining_savings = 9) : 
  initial_savings - remaining_savings - necklace_cost = 23 := 
by {
  -- insert proof steps here -- 
  sorry
}

end earrings_cost_l694_69442


namespace sum_of_all_possible_values_of_M_l694_69455

-- Given conditions
-- M * (M - 8) = -7
-- We need to prove that the sum of all possible values of M is 8

theorem sum_of_all_possible_values_of_M : 
  ∃ M1 M2 : ℝ, (M1 * (M1 - 8) = -7) ∧ (M2 * (M2 - 8) = -7) ∧ (M1 + M2 = 8) :=
by
  sorry

end sum_of_all_possible_values_of_M_l694_69455


namespace gas_cycle_work_done_l694_69460

noncomputable def p0 : ℝ := 10^5
noncomputable def V0 : ℝ := 1

theorem gas_cycle_work_done :
  (3 * Real.pi * p0 * V0 = 942) :=
by
  have h1 : p0 = 10^5 := by rfl
  have h2 : V0 = 1 := by rfl
  sorry

end gas_cycle_work_done_l694_69460


namespace cistern_filling_time_l694_69427

theorem cistern_filling_time :
  let rate_P := (1 : ℚ) / 12
  let rate_Q := (1 : ℚ) / 15
  let combined_rate := rate_P + rate_Q
  let time_combined := 6
  let filled_after_combined := combined_rate * time_combined
  let remaining_after_combined := 1 - filled_after_combined
  let time_Q := remaining_after_combined / rate_Q
  time_Q = 1.5 := sorry

end cistern_filling_time_l694_69427


namespace determine_a_l694_69404

theorem determine_a : ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by
  sorry

end determine_a_l694_69404


namespace find_x_y_l694_69414

theorem find_x_y (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : y ≥ x) (h4 : x + y ≤ 20) 
  (h5 : ¬(∃ s, (x * y = s) → x + y = s ∧ ∃ a b : ℕ, a * b = s ∧ a ≠ x ∧ b ≠ y))
  (h6 : ∃ s_t, (x + y = s_t) → x * y = s_t):
  x = 2 ∧ y = 11 :=
by {
  sorry
}

end find_x_y_l694_69414


namespace variance_daily_reading_time_l694_69450

theorem variance_daily_reading_time :
  let mean10 := 2.7
  let var10 := 1
  let num10 := 800

  let mean11 := 3.1
  let var11 := 2
  let num11 := 600

  let mean12 := 3.3
  let var12 := 3
  let num12 := 600

  let num_total := num10 + num11 + num12

  let total_mean := (2.7 * 800 + 3.1 * 600 + 3.3 * 600) / 2000

  let var_total := (800 / 2000) * (1 + (2.7 - total_mean)^2) +
                   (600 / 2000) * (2 + (3.1 - total_mean)^2) +
                   (600 / 2000) * (3 + (3.3 - total_mean)^2)

  var_total = 1.966 :=
by
  sorry

end variance_daily_reading_time_l694_69450


namespace inverse_sum_l694_69412

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by
  -- proof steps will go here
  sorry

end inverse_sum_l694_69412


namespace prasanna_speed_l694_69422

variable (v_L : ℝ) (d t : ℝ)

theorem prasanna_speed (hLaxmiSpeed : v_L = 18) (htime : t = 1) (hdistance : d = 45) : 
  ∃ v_P : ℝ, v_P = 27 :=
  sorry

end prasanna_speed_l694_69422


namespace ratio_of_rectangle_to_triangle_l694_69463

variable (L W : ℝ)

theorem ratio_of_rectangle_to_triangle (hL : L > 0) (hW : W > 0) : 
    L * W / (1/2 * L * W) = 2 := 
by
  sorry

end ratio_of_rectangle_to_triangle_l694_69463


namespace lcm_first_ten_numbers_l694_69405

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l694_69405


namespace derivative_of_volume_is_surface_area_l694_69431

noncomputable def V_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem derivative_of_volume_is_surface_area (R : ℝ) (h : 0 < R) : 
  (deriv V_sphere R) = 4 * Real.pi * R^2 :=
by sorry

end derivative_of_volume_is_surface_area_l694_69431


namespace find_z_when_x_is_1_l694_69473

-- We start by defining the conditions
variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
variable (h_inv : ∃ k₁ : ℝ, ∀ x, x^2 * y = k₁)
variable (h_dir : ∃ k₂ : ℝ, ∀ y, y / z = k₂)
variable (h_y : y = 8) (h_z : z = 32) (h_x4 : x = 4)

-- Now we need to define the problem statement: 
-- proving that z = 512 when x = 1
theorem find_z_when_x_is_1 (h_x1 : x = 1) : z = 512 :=
  sorry

end find_z_when_x_is_1_l694_69473


namespace same_terminal_side_l694_69430

theorem same_terminal_side (k : ℤ) : 
  {α | ∃ k : ℤ, α = k * 360 + (-263 : ℤ)} = 
  {α | ∃ k : ℤ, α = k * 360 - 263} := 
by sorry

end same_terminal_side_l694_69430


namespace find_x_l694_69489

theorem find_x (x : ℚ) (h : (3 * x + 4) / 5 = 15) : x = 71 / 3 :=
by
  sorry

end find_x_l694_69489


namespace solve_fraction_eq_l694_69433

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) → x = 10 :=
by
  sorry

end solve_fraction_eq_l694_69433


namespace number_of_zeros_at_end_of_factorial_30_l694_69445

-- Lean statement for the equivalence proof problem
def count_factors_of (p n : Nat) : Nat :=
  n / p + n / (p * p) + n / (p * p * p) + n / (p * p * p * p) + n / (p * p * p * p * p)

def zeros_at_end_of_factorial (n : Nat) : Nat :=
  count_factors_of 5 n

theorem number_of_zeros_at_end_of_factorial_30 : zeros_at_end_of_factorial 30 = 7 :=
by 
  sorry

end number_of_zeros_at_end_of_factorial_30_l694_69445


namespace max_imaginary_part_of_root_l694_69420

theorem max_imaginary_part_of_root (z : ℂ) (h : z^6 - z^4 + z^2 - 1 = 0) (hne : z^2 ≠ 1) : 
  ∃ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 ∧ Complex.im z = Real.sin θ ∧ θ = 90 := 
sorry

end max_imaginary_part_of_root_l694_69420


namespace similar_triangle_perimeter_l694_69410

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (T : Triangle) : Prop :=
  T.a = T.b ∨ T.a = T.c ∨ T.b = T.c

def similar_triangles (T1 T2 : Triangle) : Prop :=
  T1.a / T2.a = T1.b / T2.b ∧ T1.b / T2.b = T1.c / T2.c ∧ T1.a / T2.a = T1.c / T2.c

noncomputable def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

theorem similar_triangle_perimeter
  (T1 T2 : Triangle)
  (T1_isosceles : is_isosceles T1)
  (T1_sides : T1.a = 7 ∧ T1.b = 7 ∧ T1.c = 12)
  (T2_similar : similar_triangles T1 T2)
  (T2_longest_side : T2.c = 30) :
  perimeter T2 = 65 :=
by
  sorry

end similar_triangle_perimeter_l694_69410


namespace M_intersection_N_equals_M_l694_69417

variable (x a : ℝ)

def M : Set ℝ := { y | ∃ x, y = x^2 + 1 }
def N : Set ℝ := { y | ∃ a, y = 2 * a^2 - 4 * a + 1 }

theorem M_intersection_N_equals_M : M ∩ N = M := by
  sorry

end M_intersection_N_equals_M_l694_69417


namespace steps_climbed_l694_69453

-- Definitions
def flights : ℕ := 9
def feet_per_flight : ℕ := 10
def inches_per_step : ℕ := 18

-- Proving the number of steps John climbs up
theorem steps_climbed : 
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  steps = 60 := 
by
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  sorry

end steps_climbed_l694_69453


namespace jade_savings_per_month_l694_69492

def jade_monthly_income : ℝ := 1600
def jade_living_expense_rate : ℝ := 0.75
def jade_insurance_rate : ℝ := 0.2

theorem jade_savings_per_month : 
  jade_monthly_income * (1 - jade_living_expense_rate - jade_insurance_rate) = 80 := by
  sorry

end jade_savings_per_month_l694_69492


namespace solve_for_y_l694_69415

theorem solve_for_y (y : ℝ) (h : (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) : 
  y = (9 / 7) :=
by
  sorry

end solve_for_y_l694_69415


namespace red_shells_correct_l694_69485

-- Define the conditions
def total_shells : Nat := 291
def green_shells : Nat := 49
def non_red_green_shells : Nat := 166

-- Define the number of red shells as per the given conditions
def red_shells : Nat :=
  total_shells - green_shells - non_red_green_shells

-- State the theorem
theorem red_shells_correct : red_shells = 76 :=
by
  sorry

end red_shells_correct_l694_69485


namespace union_M_N_l694_69470

open Set Classical

noncomputable def M : Set ℝ := {x | x^2 = x}
noncomputable def N : Set ℝ := {x | Real.log x ≤ 0}

theorem union_M_N : M ∪ N = Icc 0 1 := by
  sorry

end union_M_N_l694_69470


namespace paul_number_proof_l694_69447

theorem paul_number_proof (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a - b = 7) :
  (10 * a + b = 81) ∨ (10 * a + b = 92) :=
  sorry

end paul_number_proof_l694_69447


namespace find_teaspoons_of_salt_l694_69458

def sodium_in_salt (S : ℕ) : ℕ := 50 * S
def sodium_in_parmesan (P : ℕ) : ℕ := 25 * P

-- Initial total sodium amount with 8 ounces of parmesan
def initial_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 8

-- Reduced sodium after removing 4 ounces of parmesan
def reduced_sodium (S : ℕ) : ℕ := initial_total_sodium S * 2 / 3

-- Reduced sodium with 4 fewer ounces of parmesan cheese
def new_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 4

theorem find_teaspoons_of_salt : ∃ (S : ℕ), reduced_sodium S = new_total_sodium S ∧ S = 2 :=
by
  sorry

end find_teaspoons_of_salt_l694_69458


namespace find_b_value_l694_69446

variable (a p q b : ℝ)
variable (h1 : p * 0 + q * (3 * a) + b * 1 = 1)
variable (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
variable (h3 : p * 0 + q * (3 * a) + b * 0 = 1)

theorem find_b_value : b = 0 :=
by
  sorry

end find_b_value_l694_69446


namespace find_cost_of_pencil_and_pen_l694_69480

variable (p q r : ℝ)

-- Definitions based on conditions
def condition1 := 3 * p + 2 * q + r = 4.20
def condition2 := p + 3 * q + 2 * r = 4.75
def condition3 := 2 * r = 3.00

-- The theorem to prove
theorem find_cost_of_pencil_and_pen (p q r : ℝ) (h1 : condition1 p q r) (h2 : condition2 p q r) (h3 : condition3 r) :
  p + q = 1.12 :=
by
  sorry

end find_cost_of_pencil_and_pen_l694_69480


namespace truck_boxes_per_trip_l694_69401

theorem truck_boxes_per_trip (total_boxes trips : ℕ) (h1 : total_boxes = 871) (h2 : trips = 218) : total_boxes / trips = 4 := by
  sorry

end truck_boxes_per_trip_l694_69401


namespace illuminated_area_correct_l694_69493

noncomputable def cube_illuminated_area (a ρ : ℝ) (h₁ : a = 1 / Real.sqrt 2) (h₂ : ρ = Real.sqrt (2 - Real.sqrt 3)) : ℝ :=
  (Real.sqrt 3 - 3 / 2) * (Real.pi + 3)

theorem illuminated_area_correct :
  cube_illuminated_area (1 / Real.sqrt 2) (Real.sqrt (2 - Real.sqrt 3)) (by norm_num) (by norm_num) = (Real.sqrt 3 - 3 / 2) * (Real.pi + 3) :=
sorry

end illuminated_area_correct_l694_69493


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l694_69443

section JointPurchases

/-- Given that joint purchases allow significant cost savings, reduced overhead costs,
improved quality assessment, and community trust, prove that joint purchases 
are popular in many countries despite the risks. -/
theorem joint_purchases_popular
    (cost_savings : Prop)
    (reduced_overhead_costs : Prop)
    (improved_quality_assessment : Prop)
    (community_trust : Prop)
    : Prop :=
    cost_savings ∧ reduced_overhead_costs ∧ improved_quality_assessment ∧ community_trust

/-- Given that high transaction costs, organizational difficulties,
convenience of proximity to stores, and potential disputes are challenges for neighbors,
prove that joint purchases of groceries and household goods are unpopular among neighbors. -/
theorem joint_purchases_unpopular_among_neighbors
    (high_transaction_costs : Prop)
    (organizational_difficulties : Prop)
    (convenience_proximity : Prop)
    (potential_disputes : Prop)
    : Prop :=
    high_transaction_costs ∧ organizational_difficulties ∧ convenience_proximity ∧ potential_disputes

end JointPurchases

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l694_69443


namespace sufficient_but_not_necessary_condition_for_negativity_l694_69479

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b*x + c

theorem sufficient_but_not_necessary_condition_for_negativity (b c : ℝ) :
  (c < 0 → ∃ x : ℝ, f b c x < 0) ∧ (∃ b c : ℝ, ∃ x : ℝ, c ≥ 0 ∧ f b c x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_negativity_l694_69479


namespace sufficient_not_necessary_condition_l694_69402

theorem sufficient_not_necessary_condition (x : ℝ) : (|x - 1/2| < 1/2) → (x^3 < 1) ∧ ¬(x^3 < 1) → (|x - 1/2| < 1/2) :=
sorry

end sufficient_not_necessary_condition_l694_69402


namespace jimmy_income_l694_69467

theorem jimmy_income (r_income : ℕ) (r_increase : ℕ) (combined_percent : ℚ) (j_income : ℕ) : 
  r_income = 15000 → 
  r_increase = 7000 → 
  combined_percent = 0.55 → 
  (combined_percent * (r_income + r_increase + j_income) = r_income + r_increase) → 
  j_income = 18000 := 
by
  intros h1 h2 h3 h4
  sorry

end jimmy_income_l694_69467


namespace convert_to_dms_l694_69483

-- Define the conversion factors
def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- The main proof statement
theorem convert_to_dms (d : ℝ) :
  d = 24.29 →
  (24, 17, 24) = (24, degrees_to_minutes (0.29), minutes_to_seconds 0.4) :=
by
  sorry

end convert_to_dms_l694_69483


namespace savings_of_person_l694_69498

theorem savings_of_person (income expenditure : ℕ) (h_ratio : 3 * expenditure = 2 * income) (h_income : income = 21000) :
  income - expenditure = 7000 :=
by
  sorry

end savings_of_person_l694_69498


namespace unique_sum_of_squares_power_of_two_l694_69481

theorem unique_sum_of_squares_power_of_two (n : ℕ) :
  ∃! (a b : ℕ), 2^n = a^2 + b^2 := 
sorry

end unique_sum_of_squares_power_of_two_l694_69481


namespace min_value_of_expr_l694_69403

theorem min_value_of_expr (x : ℝ) (h : x > 2) : ∃ y, (y = x + 4 / (x - 2)) ∧ y ≥ 6 :=
by
  sorry

end min_value_of_expr_l694_69403


namespace math_proof_problem_l694_69438

noncomputable def expr : ℚ :=
  ((5 / 8 * (3 / 7) + 1 / 4 * (2 / 6)) - (2 / 3 * (1 / 4) - 1 / 5 * (4 / 9))) * 
  ((7 / 9 * (2 / 5) * (1 / 2) * 5040 + 1 / 3 * (3 / 8) * (9 / 11) * 4230))

theorem math_proof_problem : expr = 336 := 
  by
  sorry

end math_proof_problem_l694_69438


namespace min_focal_length_of_hyperbola_l694_69459

theorem min_focal_length_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_area : a * b = 8) :
  ∃ c ≥ 4, 2 * c = 8 :=
by sorry

end min_focal_length_of_hyperbola_l694_69459


namespace people_left_is_10_l694_69472

def initial_people : ℕ := 12
def people_joined : ℕ := 15
def final_people : ℕ := 17
def people_left := initial_people - final_people + people_joined

theorem people_left_is_10 : people_left = 10 :=
by sorry

end people_left_is_10_l694_69472


namespace uncovered_area_frame_l694_69428

def length_frame : ℕ := 40
def width_frame : ℕ := 32
def length_photo : ℕ := 32
def width_photo : ℕ := 28

def area_frame (l_f w_f : ℕ) : ℕ := l_f * w_f
def area_photo (l_p w_p : ℕ) : ℕ := l_p * w_p

theorem uncovered_area_frame :
  area_frame length_frame width_frame - area_photo length_photo width_photo = 384 :=
by
  sorry

end uncovered_area_frame_l694_69428


namespace unique_plants_in_all_beds_l694_69482

theorem unique_plants_in_all_beds:
  let A := 600
  let B := 500
  let C := 400
  let D := 300
  let AB := 80
  let AC := 70
  let ABD := 40
  let BC := 0
  let AD := 0
  let BD := 0
  let CD := 0
  let ABC := 0
  let ACD := 0
  let BCD := 0
  let ABCD := 0
  A + B + C + D - AB - AC - BC - AD - BD - CD + ABC + ABD + ACD + BCD - ABCD = 1690 :=
by
  sorry

end unique_plants_in_all_beds_l694_69482


namespace minimum_y_value_l694_69421

noncomputable def minimum_y (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x - 15) + abs (x - a - 15)

theorem minimum_y_value (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  minimum_y x a = 15 :=
by
  sorry

end minimum_y_value_l694_69421
