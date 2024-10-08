import Mathlib

namespace min_sides_regular_polygon_l94_94404

/-- A regular polygon can accurately be placed back in its original position 
    when rotated by 50°.  Prove that the minimum number of sides the polygon 
    should have is 36. -/

theorem min_sides_regular_polygon (n : ℕ) (h : ∃ k : ℕ, 50 * k = 360 / n) : n = 36 :=
  sorry

end min_sides_regular_polygon_l94_94404


namespace remainder_range_l94_94321

theorem remainder_range (x y z a b c d e : ℕ)
(h1 : x % 211 = a) (h2 : y % 211 = b) (h3 : z % 211 = c)
(h4 : x % 251 = c) (h5 : y % 251 = d) (h6 : z % 251 = e)
(h7 : a < 211) (h8 : b < 211) (h9 : c < 211)
(h10 : c < 251) (h11 : d < 251) (h12 : e < 251) :
0 ≤ (2 * x - y + 3 * z + 47) % (211 * 251) ∧
(2 * x - y + 3 * z + 47) % (211 * 251) < (211 * 251) :=
by
  sorry

end remainder_range_l94_94321


namespace roots_of_quadratic_discriminant_positive_l94_94362

theorem roots_of_quadratic_discriminant_positive {a b c : ℝ} (h : b^2 - 4 * a * c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by {
  sorry
}

end roots_of_quadratic_discriminant_positive_l94_94362


namespace common_difference_l94_94087

noncomputable def a : ℕ := 3
noncomputable def an : ℕ := 28
noncomputable def Sn : ℕ := 186

theorem common_difference (d : ℚ) (n : ℕ) (h1 : an = a + (n-1) * d) (h2 : Sn = n * (a + an) / 2) : d = 25 / 11 :=
sorry

end common_difference_l94_94087


namespace base_any_number_l94_94730

open Nat

theorem base_any_number (n k : ℕ) (h1 : k ≥ 0) (h2 : (30 ^ k) ∣ 929260) (h3 : n ^ k - k ^ 3 = 1) : true :=
by
  sorry

end base_any_number_l94_94730


namespace find_tricksters_l94_94580

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l94_94580


namespace book_cost_l94_94415

theorem book_cost (C_1 C_2 : ℝ)
  (h1 : C_1 + C_2 = 420)
  (h2 : C_1 * 0.85 = C_2 * 1.19) :
  C_1 = 245 :=
by
  -- We skip the proof here using sorry.
  sorry

end book_cost_l94_94415


namespace difference_of_squares_l94_94701

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l94_94701


namespace sum_of_three_numbers_l94_94729

theorem sum_of_three_numbers : ∃ (a b c : ℝ), a ≤ b ∧ b ≤ c ∧ b = 8 ∧ 
  (a + b + c) / 3 = a + 8 ∧ (a + b + c) / 3 = c - 20 ∧ a + b + c = 60 :=
sorry

end sum_of_three_numbers_l94_94729


namespace probability_correct_l94_94289

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l94_94289


namespace sum_of_roots_abs_gt_six_l94_94446

theorem sum_of_roots_abs_gt_six {p r1 r2 : ℝ} (h1 : r1 + r2 = -p) (h2 : r1 * r2 = 9) (h3 : r1 ≠ r2) (h4 : p^2 > 36) : |r1 + r2| > 6 :=
sorry

end sum_of_roots_abs_gt_six_l94_94446


namespace parallel_planes_of_perpendicular_lines_l94_94477

-- Definitions of planes and lines
variable (Plane Line : Type)
variable (α β γ : Plane)
variable (m n : Line)

-- Relations between planes and lines
variable (perpendicular : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Conditions for the proof
variable (m_perp_α : perpendicular α m)
variable (n_perp_β : perpendicular β n)
variable (m_par_n : line_parallel m n)

-- Statement of the theorem
theorem parallel_planes_of_perpendicular_lines :
  parallel α β :=
sorry

end parallel_planes_of_perpendicular_lines_l94_94477


namespace area_when_other_side_shortened_l94_94561

def original_width := 5
def original_length := 8
def target_area := 24
def shortened_amount := 2

theorem area_when_other_side_shortened :
  (original_width - shortened_amount) * original_length = target_area →
  original_width * (original_length - shortened_amount) = 30 :=
by
  intros h
  sorry

end area_when_other_side_shortened_l94_94561


namespace find_larger_page_l94_94111

theorem find_larger_page {x y : ℕ} (h1 : y = x + 1) (h2 : x + y = 125) : y = 63 :=
by
  sorry

end find_larger_page_l94_94111


namespace book_cost_price_l94_94273

theorem book_cost_price 
  (M : ℝ) (hM : M = 64.54) 
  (h1 : ∃ L : ℝ, 0.92 * L = M ∧ L = 1.25 * 56.12) :
  ∃ C : ℝ, C = 56.12 :=
by
  sorry

end book_cost_price_l94_94273


namespace michael_total_time_l94_94253

def time_for_200_meters (distance speed : ℕ) : ℚ :=
  distance / speed

def total_time_per_lap : ℚ :=
  (time_for_200_meters 200 6) + (time_for_200_meters 200 3)

def total_time_8_laps : ℚ :=
  8 * total_time_per_lap

theorem michael_total_time : total_time_8_laps = 800 :=
by
  -- The proof would go here
  sorry

end michael_total_time_l94_94253


namespace trigonometric_identity_l94_94983

theorem trigonometric_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (70 * Real.pi / 180) +
   Real.sin (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180)) = 1 / 4 :=
by sorry

end trigonometric_identity_l94_94983


namespace computation_problems_count_l94_94582

theorem computation_problems_count
    (C W : ℕ)
    (h1 : 3 * C + 5 * W = 110)
    (h2 : C + W = 30) :
    C = 20 :=
by
  sorry

end computation_problems_count_l94_94582


namespace average_of_last_20_students_l94_94926

theorem average_of_last_20_students 
  (total_students : ℕ) (first_group_size : ℕ) (second_group_size : ℕ) 
  (total_average : ℕ) (first_group_average : ℕ) (second_group_average : ℕ) 
  (total_students_eq : total_students = 50) 
  (first_group_size_eq : first_group_size = 30)
  (second_group_size_eq : second_group_size = 20)
  (total_average_eq : total_average = 92) 
  (first_group_average_eq : first_group_average = 90) :
  second_group_average = 95 :=
by
  sorry

end average_of_last_20_students_l94_94926


namespace pine_cone_weight_on_roof_l94_94551

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l94_94551


namespace part1_l94_94526

def p (m x : ℝ) := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) := (x + 2)^2 < 1

theorem part1 (x : ℝ) (m : ℝ) (hm : m = -2) : p m x ∧ q x ↔ -3 < x ∧ x ≤ -2 :=
by
  unfold p q
  sorry

end part1_l94_94526


namespace three_digit_diff_l94_94466

theorem three_digit_diff (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) :
  ∃ d : ℕ, d = a - b ∧ (d < 10 ∨ (10 ≤ d ∧ d < 100) ∨ (100 ≤ d ∧ d < 1000)) :=
sorry

end three_digit_diff_l94_94466


namespace smallest_period_cos_l94_94154

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem smallest_period_cos (x : ℝ) : 
  smallest_positive_period (λ x => 2 * (Real.cos x)^2 + 1) Real.pi := 
by 
  sorry

end smallest_period_cos_l94_94154


namespace finger_cycle_2004th_l94_94052

def finger_sequence : List String :=
  ["Little finger", "Ring finger", "Middle finger", "Index finger", "Thumb", "Index finger", "Middle finger", "Ring finger"]

theorem finger_cycle_2004th : 
  finger_sequence.get! ((2004 - 1) % finger_sequence.length) = "Index finger" :=
by
  -- The proof is not required, so we use sorry
  sorry

end finger_cycle_2004th_l94_94052


namespace sandwiches_ordered_l94_94672

-- Definitions of the given conditions
def sandwichCost : ℕ := 5
def payment : ℕ := 20
def change : ℕ := 5

-- Statement to prove how many sandwiches Jack ordered
theorem sandwiches_ordered : (payment - change) / sandwichCost = 3 := by
  -- Sorry to skip the proof
  sorry

end sandwiches_ordered_l94_94672


namespace unique_arrangement_l94_94858

def valid_grid (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  (∀ i : Fin 4, (∃ j1 j2 j3 : Fin 4,
    j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3 ∧
    arrangement i j1 = 'A' ∧
    arrangement i j2 = 'B' ∧
    arrangement i j3 = 'C')) ∧
  (∀ j : Fin 4, (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 j = 'A' ∧
    arrangement i2 j = 'B' ∧
    arrangement i3 j = 'C')) ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 i1 = 'A' ∧
    arrangement i2 i2 = 'B' ∧
    arrangement i3 i3 = 'C') ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 (Fin.mk (3 - i1.val) sorry) = 'A' ∧
    arrangement i2 (Fin.mk (3 - i2.val) sorry) = 'B' ∧
    arrangement i3 (Fin.mk (3 - i3.val) sorry) = 'C')

def fixed_upper_left (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  arrangement 0 0 = 'A'

theorem unique_arrangement : ∃! arrangement : Matrix (Fin 4) (Fin 4) Char,
  valid_grid arrangement ∧ fixed_upper_left arrangement :=
sorry

end unique_arrangement_l94_94858


namespace negation_of_universal_statement_l94_94885

open Real

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 :=
by
  sorry

end negation_of_universal_statement_l94_94885


namespace initial_machines_l94_94813

theorem initial_machines (r : ℝ) (x : ℕ) (h1 : x * 42 * r = 7 * 36 * r) : x = 6 :=
by
  sorry

end initial_machines_l94_94813


namespace find_geometric_ratio_l94_94159

-- Definitions for the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def geometric_sequence (a1 a3 a4 : ℝ) (q : ℝ) : Prop :=
  a3 * a3 = a1 * a4 ∧ a3 = a1 * q ∧ a4 = a3 * q

-- Definition for the proof statement
theorem find_geometric_ratio (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hnz : ∀ n, a n ≠ 0)
  (hq : ∃ (q : ℝ), geometric_sequence (a 0) (a 2) (a 3) q) :
  ∃ q, q = 1 ∨ q = 1 / 2 := sorry

end find_geometric_ratio_l94_94159


namespace total_receipts_correct_l94_94100

def cost_adult_ticket : ℝ := 5.50
def cost_children_ticket : ℝ := 2.50
def number_of_adults : ℕ := 152
def number_of_children : ℕ := number_of_adults / 2

def receipts_from_adults : ℝ := number_of_adults * cost_adult_ticket
def receipts_from_children : ℝ := number_of_children * cost_children_ticket
def total_receipts : ℝ := receipts_from_adults + receipts_from_children

theorem total_receipts_correct : total_receipts = 1026 := 
by
  -- Proof omitted, proof needed to validate theorem statement.
  sorry

end total_receipts_correct_l94_94100


namespace number_of_sets_given_to_sister_l94_94759

-- Defining the total number of cards, sets given to his brother and friend, total cards given away,
-- number of cards per set, and expected answer for sets given to his sister.
def total_cards := 365
def sets_given_to_brother := 8
def sets_given_to_friend := 2
def total_cards_given_away := 195
def cards_per_set := 13
def sets_given_to_sister := 5

theorem number_of_sets_given_to_sister :
  sets_given_to_brother * cards_per_set + 
  sets_given_to_friend * cards_per_set + 
  sets_given_to_sister * cards_per_set = total_cards_given_away :=
by
  -- It skips the proof but ensures the statement is set up correctly.
  sorry

end number_of_sets_given_to_sister_l94_94759


namespace find_m_l94_94458

theorem find_m (x m : ℝ) :
  (2 * x + m) * (x - 3) = 2 * x^2 - 3 * m ∧ 
  (∀ c : ℝ, c * x = 0 → c = 0) → 
  m = 6 :=
by sorry

end find_m_l94_94458


namespace final_concentration_after_procedure_l94_94597

open Real

def initial_salt_concentration : ℝ := 0.16
def final_salt_concentration : ℝ := 0.107

def volume_ratio_large : ℝ := 10
def volume_ratio_medium : ℝ := 4
def volume_ratio_small : ℝ := 3

def overflow_due_to_small_ball : ℝ := 0.1

theorem final_concentration_after_procedure :
  (initial_salt_concentration * (overflow_due_to_small_ball)) * volume_ratio_small / (volume_ratio_large + volume_ratio_medium + volume_ratio_small) =
  final_salt_concentration :=
sorry

end final_concentration_after_procedure_l94_94597


namespace expression_evaluation_l94_94075

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := 
by sorry

end expression_evaluation_l94_94075


namespace eggs_distributed_equally_l94_94133

-- Define the total number of eggs
def total_eggs : ℕ := 8484

-- Define the number of baskets
def baskets : ℕ := 303

-- Define the expected number of eggs per basket
def eggs_per_basket : ℕ := 28

-- State the theorem
theorem eggs_distributed_equally :
  total_eggs / baskets = eggs_per_basket := sorry

end eggs_distributed_equally_l94_94133


namespace min_value_of_quadratic_expression_l94_94783

theorem min_value_of_quadratic_expression (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (u : ℝ), (2 * x^2 + 3 * y^2 + z^2 = u) ∧ u = 6 / 11 :=
sorry

end min_value_of_quadratic_expression_l94_94783


namespace least_sum_of_exponents_l94_94968

theorem least_sum_of_exponents {n : ℕ} (h : n = 520) (h_exp : ∃ (a b : ℕ), 2^a + 2^b = n ∧ a ≠ b ∧ a = 9 ∧ b = 3) : 
    (∃ (s : ℕ), s = 9 + 3) :=
by
  sorry

end least_sum_of_exponents_l94_94968


namespace least_subtracted_to_divisible_by_10_l94_94884

theorem least_subtracted_to_divisible_by_10 (n : ℕ) (k : ℕ) (h : n = 724946) (div_cond : (n - k) % 10 = 0) : k = 6 :=
by
  sorry

end least_subtracted_to_divisible_by_10_l94_94884


namespace g_of_900_eq_34_l94_94832

theorem g_of_900_eq_34 (g : ℕ+ → ℝ) 
  (h_mul : ∀ x y : ℕ+, g (x * y) = g x + g y)
  (h_30 : g 30 = 17)
  (h_60 : g 60 = 21) :
  g 900 = 34 :=
sorry

end g_of_900_eq_34_l94_94832


namespace option_not_equal_to_three_halves_l94_94807

theorem option_not_equal_to_three_halves (d : ℚ) (h1 : d = 3/2) 
    (hA : 9/6 = 3/2) 
    (hB : 1 + 1/2 = 3/2) 
    (hC : 1 + 2/4 = 3/2)
    (hE : 1 + 6/12 = 3/2) :
  1 + 2/3 ≠ 3/2 :=
by
  sorry

end option_not_equal_to_three_halves_l94_94807


namespace university_minimum_spend_l94_94840

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def box_cost : ℝ := 1.20
def total_volume : ℝ := 3.06 * (10^6)

def box_volume : ℕ := box_length * box_width * box_height

noncomputable def number_of_boxes : ℕ := Nat.ceil (total_volume / box_volume)
noncomputable def total_cost : ℝ := number_of_boxes * box_cost

theorem university_minimum_spend : total_cost = 612 := by
  sorry

end university_minimum_spend_l94_94840


namespace max_S_value_l94_94138

noncomputable def maximize_S (a b c : ℝ) : ℝ :=
  (a^2 - a * b + b^2) * (b^2 - b * c + c^2) * (c^2 - c * a + a^2)

theorem max_S_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 3 →
  maximize_S a b c ≤ 12 :=
by sorry

end max_S_value_l94_94138


namespace problem_l94_94542

theorem problem
    (a b c d : ℕ)
    (h1 : a = b + 7)
    (h2 : b = c + 15)
    (h3 : c = d + 25)
    (h4 : d = 90) :
  a = 137 := by
  sorry

end problem_l94_94542


namespace val_need_33_stamps_l94_94612

def valerie_needs_total_stamps 
    (thank_you_cards : ℕ) 
    (bills_water : ℕ) 
    (bills_electric : ℕ) 
    (bills_internet : ℕ) 
    (rebate_addition : ℕ) 
    (rebate_stamps : ℕ) 
    (job_apps_multiplier : ℕ) 
    (job_app_stamps : ℕ) 
    (total_stamps : ℕ) : Prop :=
    thank_you_cards = 3 ∧
    bills_water = 1 ∧
    bills_electric = 2 ∧
    bills_internet = 3 ∧
    rebate_addition = 3 ∧
    rebate_stamps = 2 ∧
    job_apps_multiplier = 2 ∧
    job_app_stamps = 1 ∧
    total_stamps = 33

theorem val_need_33_stamps : 
  valerie_needs_total_stamps 3 1 2 3 3 2 2 1 33 :=
by 
  -- proof skipped
  sorry

end val_need_33_stamps_l94_94612


namespace not_even_or_odd_l94_94977

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem not_even_or_odd : ¬(∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

end not_even_or_odd_l94_94977


namespace find_other_number_l94_94330

theorem find_other_number (m n : ℕ) (H1 : n = 26) 
  (H2 : Nat.lcm n m = 52) (H3 : Nat.gcd n m = 8) : m = 16 := by
  sorry

end find_other_number_l94_94330


namespace midpoint_sum_of_coordinates_l94_94347

theorem midpoint_sum_of_coordinates : 
  let p1 := (8, 10)
  let p2 := (-4, -10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 2 :=
by
  sorry

end midpoint_sum_of_coordinates_l94_94347


namespace evaluate_expression_at_4_l94_94743

theorem evaluate_expression_at_4 :
  ∀ x : ℝ, x = 4 → (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  intro x
  intro hx
  sorry

end evaluate_expression_at_4_l94_94743


namespace percentage_decrease_l94_94702

-- Define conditions and variables
def original_selling_price : ℝ := 659.9999999999994
def profit_rate1 : ℝ := 0.10
def increase_in_selling_price : ℝ := 42
def profit_rate2 : ℝ := 0.30

-- Define the actual proof problem
theorem percentage_decrease (C C_prime : ℝ) 
    (h1 : 1.10 * C = original_selling_price) 
    (h2 : 1.30 * C_prime = original_selling_price + increase_in_selling_price) : 
    ((C - C_prime) / C) * 100 = 10 := 
sorry

end percentage_decrease_l94_94702


namespace candies_per_friend_l94_94092

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l94_94092


namespace room_width_l94_94738

theorem room_width (length : ℝ) (cost : ℝ) (rate : ℝ) (h_length : length = 5.5)
                    (h_cost : cost = 16500) (h_rate : rate = 800) : 
                    (cost / rate / length = 3.75) :=
by 
  sorry

end room_width_l94_94738


namespace reciprocal_sum_greater_l94_94216

theorem reciprocal_sum_greater (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
    (1 / a + 1 / b) > 1 / (a + b) :=
sorry

end reciprocal_sum_greater_l94_94216


namespace cone_volume_l94_94848

noncomputable def volume_of_cone_from_lateral_surface (radius_semicircle : ℝ) 
  (circumference_base : ℝ := 2 * radius_semicircle * Real.pi) 
  (radius_base : ℝ := circumference_base / (2 * Real.pi)) 
  (height_cone : ℝ := Real.sqrt ((radius_semicircle:ℝ) ^ 2 - (radius_base:ℝ) ^ 2)) : ℝ := 
  (1 / 3) * Real.pi * (radius_base ^ 2) * height_cone

theorem cone_volume (h_semicircle : 2 = 2) : volume_of_cone_from_lateral_surface 2 = (Real.sqrt 3) / 3 * Real.pi := 
by
  -- Importing Real.sqrt and Real.pi to bring them into scope
  sorry

end cone_volume_l94_94848


namespace value_of_x_l94_94998

theorem value_of_x (x : ℝ) (h : x = 88 + 0.25 * 88) : x = 110 :=
sorry

end value_of_x_l94_94998


namespace length_of_segments_equal_d_l94_94205

noncomputable def d_eq (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) : ℝ :=
  if h_eq : AB = 550 ∧ BC = 580 ∧ AC = 620 then 342 else 0

theorem length_of_segments_equal_d (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) :
  d_eq AB BC AC h = 342 :=
by
  sorry

end length_of_segments_equal_d_l94_94205


namespace calculate_expression_l94_94755

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 25) * b^0 + (1 / (25 * b))^0 - 81^(-1 / 4 : ℝ) - (-27)^(-1 / 3 : ℝ) = 26 / 25 :=
by sorry

end calculate_expression_l94_94755


namespace max_ab_min_3x_4y_max_f_l94_94396

-- Proof Problem 1
theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : ab <= 1/16 :=
  sorry

-- Proof Problem 2
theorem min_3x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y >= 5 :=
  sorry

-- Proof Problem 3
theorem max_f (x : ℝ) (h1 : x < 5/4) : 4 * x - 2 + 1 / (4 * x - 5) <= 1 :=
  sorry

end max_ab_min_3x_4y_max_f_l94_94396


namespace product_equals_16896_l94_94139

theorem product_equals_16896 (A B C D : ℕ) (h1 : A + B + C + D = 70)
  (h2 : A = 3 * C + 1) (h3 : B = 3 * C + 5) (h4 : C = C) (h5 : D = 3 * C^2) :
  A * B * C * D = 16896 :=
by
  sorry

end product_equals_16896_l94_94139


namespace one_fourth_way_from_x1_to_x2_l94_94143

-- Definitions of the points
def x1 : ℚ := 1 / 5
def x2 : ℚ := 4 / 5

-- Problem statement: Prove that one fourth of the way from x1 to x2 is 7/20
theorem one_fourth_way_from_x1_to_x2 : (3 * x1 + 1 * x2) / 4 = 7 / 20 := by
  sorry

end one_fourth_way_from_x1_to_x2_l94_94143


namespace area_of_region_l94_94344

-- The problem definition
def condition_1 (z : ℂ) : Prop := 
  0 < z.re / 20 ∧ z.re / 20 < 1 ∧
  0 < z.im / 20 ∧ z.im / 20 < 1 ∧
  0 < (20 / z).re ∧ (20 / z).re < 1 ∧
  0 < (20 / z).im ∧ (20 / z).im < 1

-- The proof statement
theorem area_of_region {z : ℂ} (h : condition_1 z) : 
  ∃ s : ℝ, s = 300 - 50 * Real.pi := sorry

end area_of_region_l94_94344


namespace min_ab_value_l94_94728

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 4 / b = Real.sqrt (a * b)) :
  a * b = 4 :=
  sorry

end min_ab_value_l94_94728


namespace total_amount_spent_l94_94158

variables (D B : ℝ)

-- Conditions
def condition1 : Prop := B = 1.5 * D
def condition2 : Prop := D = B - 15

-- Question: Prove that the total amount they spent together is 75.00
theorem total_amount_spent (h1 : condition1 D B) (h2 : condition2 D B) : B + D = 75 :=
sorry

end total_amount_spent_l94_94158


namespace fg_neg_one_eq_neg_eight_l94_94276

def f (x : ℤ) : ℤ := x - 4
def g (x : ℤ) : ℤ := x^2 + 2*x - 3

theorem fg_neg_one_eq_neg_eight : f (g (-1)) = -8 := by
  sorry

end fg_neg_one_eq_neg_eight_l94_94276


namespace wrong_mark_is_43_l94_94301

theorem wrong_mark_is_43
  (correct_mark : ℕ)
  (wrong_mark : ℕ)
  (num_students : ℕ)
  (avg_increase : ℕ)
  (h_correct : correct_mark = 63)
  (h_num_students : num_students = 40)
  (h_avg_increase : avg_increase = 40 / 2) 
  (h_wrong_avg : (num_students - 1) * (correct_mark + avg_increase) / num_students = (num_students - 1) * (wrong_mark + avg_increase + correct_mark) / num_students) :
  wrong_mark = 43 :=
sorry

end wrong_mark_is_43_l94_94301


namespace carly_cooks_in_72_minutes_l94_94438

def total_time_to_cook_burgers (total_guests : ℕ) (cook_time_per_side : ℕ) (burgers_per_grill : ℕ) : ℕ :=
  let guests_who_want_two_burgers := total_guests / 2
  let guests_who_want_one_burger := total_guests - guests_who_want_two_burgers
  let total_burgers := (guests_who_want_two_burgers * 2) + guests_who_want_one_burger
  let total_batches := (total_burgers + burgers_per_grill - 1) / burgers_per_grill  -- ceil division for total batches
  total_batches * (2 * cook_time_per_side)  -- total time

theorem carly_cooks_in_72_minutes : 
  total_time_to_cook_burgers 30 4 5 = 72 :=
by 
  sorry

end carly_cooks_in_72_minutes_l94_94438


namespace cube_surface_divisible_into_12_squares_l94_94022

theorem cube_surface_divisible_into_12_squares (a : ℝ) :
  (∃ b : ℝ, b = a / Real.sqrt 2 ∧
  ∀ cube_surface_area: ℝ, cube_surface_area = 6 * a^2 →
  ∀ smaller_square_area: ℝ, smaller_square_area = b^2 →
  12 * smaller_square_area = cube_surface_area) :=
sorry

end cube_surface_divisible_into_12_squares_l94_94022


namespace sin_cos_value_l94_94947

variable (α : ℝ) (a b : ℝ × ℝ)
def vectors_parallel : Prop := b = (Real.sin α, Real.cos α) ∧
a = (4, 3) ∧ (∃ k : ℝ, a = (k * (Real.sin α), k * (Real.cos α)))

theorem sin_cos_value (h : vectors_parallel α a b) : ((Real.sin α) * (Real.cos α)) = 12 / 25 :=
by
  sorry

end sin_cos_value_l94_94947


namespace digit_agreement_l94_94209

theorem digit_agreement (N : ℕ) (abcd : ℕ) (h1 : N % 10000 = abcd) (h2 : N ^ 2 % 10000 = abcd) (h3 : ∃ a b c d, abcd = a * 1000 + b * 100 + c * 10 + d ∧ a ≠ 0) : abcd / 10 = 937 := sorry

end digit_agreement_l94_94209


namespace joan_gave_sam_43_seashells_l94_94259

def joan_original_seashells : ℕ := 70
def joan_seashells_left : ℕ := 27
def seashells_given_to_sam : ℕ := 43

theorem joan_gave_sam_43_seashells :
  joan_original_seashells - joan_seashells_left = seashells_given_to_sam :=
by
  sorry

end joan_gave_sam_43_seashells_l94_94259


namespace a_2016_value_l94_94918

theorem a_2016_value (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6) 
  (rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 2016 = -3 :=
sorry

end a_2016_value_l94_94918


namespace P_union_Q_eq_Q_l94_94144

noncomputable def P : Set ℝ := {x : ℝ | x > 1}
noncomputable def Q : Set ℝ := {x : ℝ | x^2 - x > 0}

theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end P_union_Q_eq_Q_l94_94144


namespace correct_option_l94_94577

-- Definitions based on the conditions of the problem
def exprA (a : ℝ) : Prop := 7 * a + a = 7 * a^2
def exprB (x y : ℝ) : Prop := 3 * x^2 * y - 2 * x^2 * y = x^2 * y
def exprC (y : ℝ) : Prop := 5 * y - 3 * y = 2
def exprD (a b : ℝ) : Prop := 3 * a + 2 * b = 5 * a * b

-- Proof problem statement verifying the correctness of the given expressions
theorem correct_option (x y : ℝ) : exprB x y :=
by
  -- (No proof is required, the statement is sufficient)
  sorry

end correct_option_l94_94577


namespace vector_parallel_x_value_l94_94090

theorem vector_parallel_x_value :
  ∀ (x : ℝ), let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  (∃ k : ℝ, b = (k * 3, k * 1)) → x = -9 :=
by
  intro x
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  intro h
  sorry

end vector_parallel_x_value_l94_94090


namespace regular_dinosaur_weight_l94_94706

namespace DinosaurWeight

-- Given Conditions
def Barney_weight (x : ℝ) : ℝ := 5 * x + 1500
def combined_weight (x : ℝ) : ℝ := Barney_weight x + 5 * x

-- Target Proof
theorem regular_dinosaur_weight :
  (∃ x : ℝ, combined_weight x = 9500) -> 
  ∃ x : ℝ, x = 800 :=
by {
  sorry
}

end DinosaurWeight

end regular_dinosaur_weight_l94_94706


namespace no_arith_prog_of_sines_l94_94636

theorem no_arith_prog_of_sines (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : x₂ ≠ x₃) (h₃ : x₁ ≠ x₃)
    (hx : 0 < x₁ ∧ x₁ < (Real.pi / 2))
    (hy : 0 < x₂ ∧ x₂ < (Real.pi / 2))
    (hz : 0 < x₃ ∧ x₃ < (Real.pi / 2))
    (h : 2 * Real.sin x₂ = Real.sin x₁ + Real.sin x₃) :
    ¬ (x₁ + x₃ = 2 * x₂) :=
sorry

end no_arith_prog_of_sines_l94_94636


namespace probability_odd_product_l94_94128

-- Given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the proof problem
theorem probability_odd_product (h: choices = 15 ∧ odd_choices = 3) :
  (odd_choices : ℚ) / choices = 1 / 5 :=
by sorry

end probability_odd_product_l94_94128


namespace mapping_problem_l94_94866

open Set

noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f₂ (x : ℝ) : ℝ := 1 / x
def f₃ (x : ℝ) : ℝ := x^2 - 2
def f₄ (x : ℝ) : ℝ := x^2

def A₁ : Set ℝ := {1, 4, 9}
def B₁ : Set ℝ := {-3, -2, -1, 1, 2, 3}
def A₂ : Set ℝ := univ
def B₂ : Set ℝ := univ
def A₃ : Set ℝ := univ
def B₃ : Set ℝ := univ
def A₄ : Set ℝ := {-1, 0, 1}
def B₄ : Set ℝ := {-1, 0, 1}

theorem mapping_problem : 
  ¬ (∀ x ∈ A₁, f₁ x ∈ B₁) ∧
  ¬ (∀ x ∈ A₂, x ≠ 0 → f₂ x ∈ B₂) ∧
  (∀ x ∈ A₃, f₃ x ∈ B₃) ∧
  (∀ x ∈ A₄, f₄ x ∈ B₄) :=
by
  sorry

end mapping_problem_l94_94866


namespace grid_game_winner_l94_94640

theorem grid_game_winner {m n : ℕ} :
  (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") = (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") := by
  sorry

end grid_game_winner_l94_94640


namespace inequality_solution_l94_94400

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l94_94400


namespace initial_population_correct_l94_94439

-- Definitions based on conditions
def initial_population (P : ℝ) := P
def population_after_bombardment (P : ℝ) := 0.9 * P
def population_after_fear (P : ℝ) := 0.8 * (population_after_bombardment P)
def final_population := 3240

-- Theorem statement
theorem initial_population_correct (P : ℝ) (h : population_after_fear P = final_population) :
  initial_population P = 4500 :=
sorry

end initial_population_correct_l94_94439


namespace pears_morning_sales_l94_94946

theorem pears_morning_sales (morning afternoon : ℕ) 
  (h1 : afternoon = 2 * morning)
  (h2 : morning + afternoon = 360) : 
  morning = 120 := 
sorry

end pears_morning_sales_l94_94946


namespace distance_to_x_axis_l94_94747

theorem distance_to_x_axis (x y : ℝ) :
  (x^2 / 9 - y^2 / 16 = 1) →
  (x^2 + y^2 = 25) →
  abs y = 16 / 5 :=
by
  -- Conditions: x^2 / 9 - y^2 / 16 = 1, x^2 + y^2 = 25
  -- Conclusion: abs y = 16 / 5 
  intro h1 h2
  sorry

end distance_to_x_axis_l94_94747


namespace circle_diameter_of_circumscribed_square_l94_94459

theorem circle_diameter_of_circumscribed_square (r : ℝ) (s : ℝ) (h1 : s = 2 * r) (h2 : 4 * s = π * r^2) : 2 * r = 16 / π := by
  sorry

end circle_diameter_of_circumscribed_square_l94_94459


namespace parameterization_theorem_l94_94155

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end parameterization_theorem_l94_94155


namespace samantha_birth_year_l94_94843

theorem samantha_birth_year (first_kangaroo_year birth_year kangaroo_freq : ℕ)
  (h_first_kangaroo: first_kangaroo_year = 1991)
  (h_kangaroo_freq: kangaroo_freq = 1)
  (h_samantha_age: ∃ y, y = (first_kangaroo_year + 9 * kangaroo_freq) ∧ 2000 - 14 = y) :
  birth_year = 1986 :=
by sorry

end samantha_birth_year_l94_94843


namespace rows_of_seats_l94_94634

theorem rows_of_seats (r : ℕ) (h : r * 4 = 80) : r = 20 :=
sorry

end rows_of_seats_l94_94634


namespace weight_of_six_moles_BaF2_l94_94631

variable (atomic_weight_Ba : ℝ := 137.33) -- Atomic weight of Barium in g/mol
variable (atomic_weight_F : ℝ := 19.00) -- Atomic weight of Fluorine in g/mol
variable (moles_BaF2 : ℝ := 6) -- Number of moles of BaF2

theorem weight_of_six_moles_BaF2 :
  moles_BaF2 * (atomic_weight_Ba + 2 * atomic_weight_F) = 1051.98 :=
by sorry

end weight_of_six_moles_BaF2_l94_94631


namespace bamboo_capacity_l94_94067

theorem bamboo_capacity :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 d : ℚ),
    a_1 + a_2 + a_3 = 4 ∧
    a_6 + a_7 + a_8 + a_9 = 3 ∧
    a_2 = a_1 + d ∧
    a_3 = a_1 + 2*d ∧
    a_4 = a_1 + 3*d ∧
    a_5 = a_1 + 4*d ∧
    a_7 = a_1 + 5*d ∧
    a_8 = a_1 + 6*d ∧
    a_9 = a_1 + 7*d ∧
    a_4 = 1 + 8/66 ∧
    a_5 = 1 + 1/66 :=
sorry

end bamboo_capacity_l94_94067


namespace nonagon_perimeter_is_28_l94_94059

-- Definitions based on problem conditions
def numSides : Nat := 9
def lengthSides1 : Nat := 3
def lengthSides2 : Nat := 4
def numSidesOfLength1 : Nat := 8
def numSidesOfLength2 : Nat := 1

-- Theorem statement proving that the perimeter is 28 units
theorem nonagon_perimeter_is_28 : 
  numSides = numSidesOfLength1 + numSidesOfLength2 →
  8 * lengthSides1 + 1 * lengthSides2 = 28 :=
by
  intros
  sorry

end nonagon_perimeter_is_28_l94_94059


namespace problem_statement_l94_94212

theorem problem_statement (x y : ℝ) (h₁ : 2.5 * x = 0.75 * y) (h₂ : x = 20) : y = 200 / 3 := by
  sorry

end problem_statement_l94_94212


namespace muscovy_more_than_cayuga_l94_94080

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end muscovy_more_than_cayuga_l94_94080


namespace games_per_season_l94_94384

-- Define the problem parameters
def total_goals : ℕ := 1244
def louie_last_match_goals : ℕ := 4
def louie_previous_goals : ℕ := 40
def louie_season_total_goals := louie_last_match_goals + louie_previous_goals
def brother_goals_per_game := 2 * louie_last_match_goals
def seasons : ℕ := 3

-- Prove the number of games in each season
theorem games_per_season : ∃ G : ℕ, louie_season_total_goals + (seasons * brother_goals_per_game * G) = total_goals ∧ G = 50 := 
by {
  sorry
}

end games_per_season_l94_94384


namespace apples_in_each_basket_l94_94397

theorem apples_in_each_basket (total_apples : ℕ) (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : baskets = 19) 
  (h3 : apples_per_basket = total_apples / baskets) : 
  apples_per_basket = 26 :=
by 
  rw [h1, h2] at h3
  exact h3

end apples_in_each_basket_l94_94397


namespace number_of_unique_combinations_l94_94113

-- Define the inputs and the expected output.
def n := 8
def r := 3
def expected_combinations := 56

-- We state our theorem indicating that the combination of 8 toppings chosen 3 at a time
-- equals 56.
theorem number_of_unique_combinations :
  (Nat.choose n r = expected_combinations) :=
by
  sorry

end number_of_unique_combinations_l94_94113


namespace Kelly_needs_to_give_away_l94_94208

variable (n k : Nat)

theorem Kelly_needs_to_give_away (h_n : n = 20) (h_k : k = 12) : n - k = 8 := 
by
  sorry

end Kelly_needs_to_give_away_l94_94208


namespace crystal_discount_is_50_percent_l94_94905

noncomputable def discount_percentage_original_prices_and_revenue
  (original_price_cupcake : ℝ)
  (original_price_cookie : ℝ)
  (total_cupcakes_sold : ℕ)
  (total_cookies_sold : ℕ)
  (total_revenue : ℝ)
  (percentage_discount : ℝ) :
  Prop :=
  total_cupcakes_sold * (original_price_cupcake * (1 - percentage_discount / 100)) +
  total_cookies_sold * (original_price_cookie * (1 - percentage_discount / 100)) = total_revenue

theorem crystal_discount_is_50_percent :
  discount_percentage_original_prices_and_revenue 3 2 16 8 32 50 :=
by sorry

end crystal_discount_is_50_percent_l94_94905


namespace geometric_sequence_product_l94_94394

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 * a 11 = 8) :
  a 2 * a 8 = 4 :=
sorry

end geometric_sequence_product_l94_94394


namespace max_sum_of_squares_eq_7_l94_94815

theorem max_sum_of_squares_eq_7 :
  ∃ (x y : ℤ), (x^2 + y^2 = 25 ∧ x + y = 7) ∧
  (∀ x' y' : ℤ, (x'^2 + y'^2 = 25 → x' + y' ≤ 7)) := by
sorry

end max_sum_of_squares_eq_7_l94_94815


namespace sum_of_squares_l94_94179

theorem sum_of_squares (a b c : ℝ)
  (h1 : a + b + c = 19)
  (h2 : a * b + b * c + c * a = 131) :
  a^2 + b^2 + c^2 = 99 :=
by
  sorry

end sum_of_squares_l94_94179


namespace find_other_endpoint_l94_94928

theorem find_other_endpoint (x y : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 3 ∧ x1 = -1 ∧ y1 = 7 ∧ x2 = x ∧ y2 = y) → (x = 5 ∧ y = -1) :=
by
  sorry

end find_other_endpoint_l94_94928


namespace not_square_l94_94908

open Int

theorem not_square (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ k : ℤ, (a^2 : ℤ) + ⌈(4 * a^2 : ℤ) / b⌉ = k^2 :=
by
  sorry

end not_square_l94_94908


namespace problem_solution_l94_94714

open Nat

def sum_odd (n : ℕ) : ℕ :=
  n ^ 2

def sum_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem problem_solution : 
  sum_odd 1010 - sum_even 1009 = 1010 :=
by
  -- Here the proof would go
  sorry

end problem_solution_l94_94714


namespace corn_growth_first_week_l94_94568

theorem corn_growth_first_week (x : ℝ) (h1 : x + 2*x + 8*x = 22) : x = 2 :=
by
  sorry

end corn_growth_first_week_l94_94568


namespace part1_part2_l94_94888

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem part1 : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 := by
  sorry

theorem part2 (m : ℝ) : 1 ≤ m →
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f x ≤ f m) ∧ 
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f 1 ≤ f x) →
  f m - f 1 = 1 / 2 →
  m = 2 := by
  sorry

end part1_part2_l94_94888


namespace angle_equivalence_modulo_l94_94306

-- Defining the given angles
def theta1 : ℤ := -510
def theta2 : ℤ := 210

-- Proving that the angles are equivalent modulo 360
theorem angle_equivalence_modulo : theta1 % 360 = theta2 % 360 :=
by sorry

end angle_equivalence_modulo_l94_94306


namespace problem1_problem2_problem3_l94_94705

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x - 4) = -f x)
variable (h_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ 2 → x ≤ y → y ≤ 2 → f x ≤ f y)

-- Problem statements
theorem problem1 : f 2012 = 0 := sorry

theorem problem2 : ∀ x, f (4 - x) = -f (4 + x) := sorry

theorem problem3 : f (-25) < f 80 ∧ f 80 < f 11 := sorry

end problem1_problem2_problem3_l94_94705


namespace petya_pencils_l94_94923

theorem petya_pencils (x : ℕ) (promotion : x + 12 = 61) :
  x = 49 :=
by
  sorry

end petya_pencils_l94_94923


namespace participants_with_exactly_five_problems_l94_94223

theorem participants_with_exactly_five_problems (n : ℕ) 
  (p : Fin 6 → Fin 6 → ℕ)
  (h1 : ∀ i j : Fin 6, i ≠ j → p i j > 2 * n / 5)
  (h2 : ¬ ∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → p i j = n)
  : ∃ k1 k2 : Fin n, k1 ≠ k2 ∧ (∀ i : Fin 6, (p i k1 = 5) ∧ (p i k2 = 5)) :=
sorry

end participants_with_exactly_five_problems_l94_94223


namespace square_area_inscribed_triangle_l94_94106

-- Definitions from the conditions of the problem
variable (EG : ℝ) (hF : ℝ)

-- Since EG = 12 inches and the altitude from F to EG is 7 inches
theorem square_area_inscribed_triangle 
(EG_eq : EG = 12) 
(hF_eq : hF = 7) :
  ∃ (AB : ℝ), AB ^ 2 = 36 :=
by 
  sorry

end square_area_inscribed_triangle_l94_94106


namespace ratio_height_radius_l94_94723

variable (V r h : ℝ)

theorem ratio_height_radius (h_eq_2r : h = 2 * r) (volume_eq : π * r^2 * h = V) : h / r = 2 :=
by
  sorry

end ratio_height_radius_l94_94723


namespace sequence_filling_l94_94823

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end sequence_filling_l94_94823


namespace range_of_x_satisfying_inequality_l94_94586

theorem range_of_x_satisfying_inequality (x : ℝ) : x^2 < |x| ↔ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1) :=
by
  sorry

end range_of_x_satisfying_inequality_l94_94586


namespace ab_sum_eq_2_l94_94221

theorem ab_sum_eq_2 (a b : ℝ) (M : Set ℝ) (N : Set ℝ) (f : ℝ → ℝ) 
  (hM : M = {b / a, 1})
  (hN : N = {a, 0})
  (hf : ∀ x ∈ M, f x ∈ N)
  (f_def : ∀ x, f x = 2 * x) :
  a + b = 2 :=
by
  -- proof goes here.
  sorry

end ab_sum_eq_2_l94_94221


namespace length_of_goods_train_l94_94066

theorem length_of_goods_train 
  (speed_kmph : ℝ) (platform_length : ℝ) (time_sec : ℝ) (train_length : ℝ) 
  (h1 : speed_kmph = 72)
  (h2 : platform_length = 270) 
  (h3 : time_sec = 26) 
  (h4 : train_length = (speed_kmph * 1000 / 3600 * time_sec) - platform_length)
  : train_length = 250 := 
  by
    sorry

end length_of_goods_train_l94_94066


namespace red_balls_count_l94_94791

theorem red_balls_count (r y b : ℕ) (total_balls : ℕ := 15) (prob_neither_red : ℚ := 2/7) :
    y + b = total_balls - r → (15 - r) * (14 - r) = 60 → r = 5 :=
by
  intros h1 h2
  sorry

end red_balls_count_l94_94791


namespace only_pair_2_2_satisfies_l94_94457

theorem only_pair_2_2_satisfies :
  ∀ a b : ℕ, (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by sorry

end only_pair_2_2_satisfies_l94_94457


namespace seulgi_second_round_score_l94_94572

theorem seulgi_second_round_score
    (h_score1 : Nat) (h_score2 : Nat)
    (hj_score1 : Nat) (hj_score2 : Nat)
    (s_score1 : Nat) (required_second_score : Nat) :
    h_score1 = 23 →
    h_score2 = 28 →
    hj_score1 = 32 →
    hj_score2 = 17 →
    s_score1 = 27 →
    required_second_score = 25 →
    s_score1 + required_second_score > h_score1 + h_score2 ∧ 
    s_score1 + required_second_score > hj_score1 + hj_score2 :=
by
  intros
  sorry

end seulgi_second_round_score_l94_94572


namespace change_given_l94_94931

theorem change_given (pants_cost : ℕ) (shirt_cost : ℕ) (tie_cost : ℕ) (total_paid : ℕ) (total_cost : ℕ) (change : ℕ) :
  pants_cost = 140 ∧ shirt_cost = 43 ∧ tie_cost = 15 ∧ total_paid = 200 ∧ total_cost = (pants_cost + shirt_cost + tie_cost) ∧ change = (total_paid - total_cost) → change = 2 :=
by
  sorry

end change_given_l94_94931


namespace evaluate_expression_l94_94460

theorem evaluate_expression :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = (3^1002) / (2^1000) :=
by sorry

end evaluate_expression_l94_94460


namespace min_value_of_expression_l94_94811

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) :=
sorry

end min_value_of_expression_l94_94811


namespace sock_pairs_l94_94084

def total_ways (n_white n_brown n_blue n_red : ℕ) : ℕ :=
  n_blue * n_white + n_blue * n_brown + n_blue * n_red

theorem sock_pairs (n_white n_brown n_blue n_red : ℕ) (h_white : n_white = 5) (h_brown : n_brown = 4) (h_blue : n_blue = 2) (h_red : n_red = 1) :
  total_ways n_white n_brown n_blue n_red = 20 := by
  -- insert the proof steps here
  sorry

end sock_pairs_l94_94084


namespace minimum_value_l94_94357

def f (x a : ℝ) : ℝ := x^3 - a*x^2 - a^2*x
def f_prime (x a : ℝ) : ℝ := 3*x^2 - 2*a*x - a^2

theorem minimum_value (a : ℝ) (hf_prime : f_prime 1 a = 0) (ha : a = -3) : ∃ x : ℝ, f x a = -5 := 
sorry

end minimum_value_l94_94357


namespace incorrect_intersection_point_l94_94869

def linear_function (x : ℝ) : ℝ := -2 * x + 4

theorem incorrect_intersection_point : ¬(linear_function 0 = 4) :=
by {
  /- Proof can be filled here later -/
  sorry
}

end incorrect_intersection_point_l94_94869


namespace purchasing_methods_l94_94620

theorem purchasing_methods :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 7 ∧
    ∀ (x y : ℕ), (x, y) ∈ s ↔ 60 * x + 70 * y ≤ 500 ∧ 3 ≤ x ∧ 2 ≤ y :=
sorry

end purchasing_methods_l94_94620


namespace intersection_eq_l94_94507

def A : Set ℝ := { x | abs x ≤ 2 }
def B : Set ℝ := { x | 3 * x - 2 ≥ 1 }

theorem intersection_eq :
  A ∩ B = { x | 1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_eq_l94_94507


namespace proof_problem_l94_94806

variables {R : Type*} [CommRing R]

-- f is a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Variable definitions for the conditions
variables (h_odd : is_odd f)
(h_f1 : f 1 = 1)
(h_period : ∀ x, f (x + 6) = f x + f 3)

-- The proof problem statement
theorem proof_problem : f 2015 + f 2016 = -1 :=
by
  sorry

end proof_problem_l94_94806


namespace boat_stream_speed_l94_94006

theorem boat_stream_speed (v : ℝ) (h : (60 / (15 - v)) - (60 / (15 + v)) = 2) : v = 3.5 := 
by 
  sorry
 
end boat_stream_speed_l94_94006


namespace find_k_circle_radius_l94_94652

theorem find_k_circle_radius (k : ℝ) :
  (∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) → ((x + 4)^2 + (y + 2)^2 = 7^2)) → k = 29 :=
sorry

end find_k_circle_radius_l94_94652


namespace carson_pumps_needed_l94_94538

theorem carson_pumps_needed 
  (full_tire_capacity : ℕ) (flat_tires_count : ℕ) 
  (full_percentage_tire_1 : ℚ) (full_percentage_tire_2 : ℚ)
  (air_per_pump : ℕ) : 
  flat_tires_count = 2 →
  full_tire_capacity = 500 →
  full_percentage_tire_1 = 0.40 →
  full_percentage_tire_2 = 0.70 →
  air_per_pump = 50 →
  let needed_air_flat_tires := flat_tires_count * full_tire_capacity
  let needed_air_tire_1 := (1 - full_percentage_tire_1) * full_tire_capacity
  let needed_air_tire_2 := (1 - full_percentage_tire_2) * full_tire_capacity
  let total_needed_air := needed_air_flat_tires + needed_air_tire_1 + needed_air_tire_2
  let pumps_needed := total_needed_air / air_per_pump
  pumps_needed = 29 := 
by
  intros
  sorry

end carson_pumps_needed_l94_94538


namespace cubic_polynomial_roots_value_l94_94371

theorem cubic_polynomial_roots_value
  (a b c d : ℝ) 
  (h_cond : a ≠ 0 ∧ d ≠ 0)
  (h_equiv : (a * (1/2)^3 + b * (1/2)^2 + c * (1/2) + d) + (a * (-1/2)^3 + b * (-1/2)^2 + c * (-1/2) + d) = 1000 * d)
  (h_roots : ∃ (x1 x2 x3 : ℝ), a * x1^3 + b * x1^2 + c * x1 + d = 0 ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
  : (∃ (x1 x2 x3 : ℝ), (1 / (x1 * x2) + 1 / (x2 * x3) + 1 / (x1 * x3) = 1996)) :=
by
  sorry

end cubic_polynomial_roots_value_l94_94371


namespace sequence_general_formula_l94_94942

theorem sequence_general_formula :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (∀ n : ℕ, n > 0 → a n - a (n + 1) = 2 * a n * a (n + 1) / (n * (n + 1))) →
  ∀ n : ℕ, n > 0 → a n = n / (3 * n - 2) :=
by
  intros a h1 h_rec n hn
  sorry

end sequence_general_formula_l94_94942


namespace problem_proof_l94_94964

variables {m n : ℝ}

-- Line definitions
def l1 (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l2 (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Conditions
def intersects_at (m n : ℝ) : Prop :=
  l1 m n m (-1) ∧ l2 m m (-1)

def parallel (m n : ℝ) : Prop :=
  (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)

def perpendicular (m n : ℝ) : Prop :=
  m = 0 ∧ n = 8

theorem problem_proof :
  intersects_at m n → (m = 1 ∧ n = 7) ∧
  parallel m n → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2) ∧
  perpendicular m n → (m = 0 ∧ n = 8) :=
by
  sorry

end problem_proof_l94_94964


namespace jack_estimate_larger_l94_94069

variable {x y a b : ℝ}

theorem jack_estimate_larger (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (ha : 0 < a) (hb : 0 < b) : 
  (x + a) - (y - b) > x - y :=
by
  sorry

end jack_estimate_larger_l94_94069


namespace parabola_intersects_line_exactly_once_l94_94269

theorem parabola_intersects_line_exactly_once (p q : ℚ) : 
  (∀ x : ℝ, 2 * (x - p) ^ 2 = x - 4 ↔ p = 31 / 8) ∧ 
  (∀ x : ℝ, 2 * x ^ 2 - q = x - 4 ↔ q = 31 / 8) := 
by 
  sorry

end parabola_intersects_line_exactly_once_l94_94269


namespace clock_angle_at_330_l94_94116

/--
At 3:00, the hour hand is at 90 degrees from the 12 o'clock position.
The minute hand at 3:30 is at 180 degrees from the 12 o'clock position.
The hour hand at 3:30 has moved an additional 15 degrees (0.5 degrees per minute).
Prove that the smaller angle formed by the hour and minute hands of a clock at 3:30 is 75.0 degrees.
-/
theorem clock_angle_at_330 : 
  let hour_pos_at_3 := 90
  let min_pos_at_330 := 180
  let hour_additional := 15
  (min_pos_at_330 - (hour_pos_at_3 + hour_additional) = 75)
  :=
  by
  sorry

end clock_angle_at_330_l94_94116


namespace determine_sum_l94_94988

theorem determine_sum (P R : ℝ) (h : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 78) : 
  P = 2600 :=
sorry

end determine_sum_l94_94988


namespace third_side_length_not_12_l94_94945

theorem third_side_length_not_12 (x : ℕ) (h1 : x % 2 = 0) (h2 : 5 < x) (h3 : x < 11) : x ≠ 12 := 
sorry

end third_side_length_not_12_l94_94945


namespace mask_production_l94_94369

theorem mask_production (M : ℕ) (h : 16 * M = 48000) : M = 3000 :=
by
  sorry

end mask_production_l94_94369


namespace problem1_problem2_l94_94694

variable (a b : ℝ)

-- (1) Prove a + b = 2 given the conditions
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x : ℝ, abs (x - a) + abs (x + b) ≥ 2) : a + b = 2 :=
sorry

-- (2) Prove it is not possible for both a^2 + a > 2 and b^2 + b > 2 to hold simultaneously
theorem problem2 (h1: a + b = 2) (h2 : a^2 + a > 2) (h3 : b^2 + b > 2) : False :=
sorry

end problem1_problem2_l94_94694


namespace rectangle_diagonal_length_proof_parallel_l94_94588

-- Definition of a rectangle whose sides are parallel to the coordinate axes
structure RectangleParallel :=
  (a b : ℕ)
  (area_eq : a * b = 2018)
  (diagonal_length : ℕ)

-- Prove that the length of the diagonal of the given rectangle is sqrt(1018085)
def rectangle_diagonal_length_parallel : RectangleParallel → Prop :=
  fun r => r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)

theorem rectangle_diagonal_length_proof_parallel (r : RectangleParallel)
  (h1 : r.a * r.b = 2018)
  (h2 : r.a ≠ r.b)
  (h3 : r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)) :
  r.diagonal_length = Int.sqrt 1018085 := 
  sorry

end rectangle_diagonal_length_proof_parallel_l94_94588


namespace problem1_problem2_l94_94724

-- Problem 1
theorem problem1 : 3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem2 : abs (Real.sqrt 3 - Real.sqrt 2) + abs (Real.sqrt 3 - 2) + Real.sqrt 4 = 4 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l94_94724


namespace expression_value_l94_94191

open Real

theorem expression_value :
  3 + sqrt 3 + 1 / (3 + sqrt 3) + 1 / (sqrt 3 - 3) = 3 + 2 * sqrt 3 / 3 := 
sorry

end expression_value_l94_94191


namespace minimum_sum_of_squares_l94_94292

theorem minimum_sum_of_squares (α p q : ℝ) 
  (h1: p + q = α - 2) (h2: p * q = - (α + 1)) :
  p^2 + q^2 ≥ 5 :=
by
  sorry

end minimum_sum_of_squares_l94_94292


namespace range_of_f_l94_94480

noncomputable def f (x y z : ℝ) := ((x * y + y * z + z * x) * (x + y + z)) / ((x + y) * (y + z) * (z + x))

theorem range_of_f :
  ∃ x y z : ℝ, (0 < x ∧ 0 < y ∧ 0 < z) ∧ f x y z = r ↔ 1 ≤ r ∧ r ≤ 9 / 8 :=
sorry

end range_of_f_l94_94480


namespace problem_l94_94148

theorem problem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 := 
sorry

end problem_l94_94148


namespace set_cannot_be_divided_l94_94543

theorem set_cannot_be_divided
  (p : ℕ) (prime_p : Nat.Prime p) (p_eq_3_mod_4 : p % 4 = 3)
  (S : Finset ℕ) (hS : S.card = p - 1) :
  ¬∃ A B : Finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
by {
  sorry
}

end set_cannot_be_divided_l94_94543


namespace carA_catches_up_with_carB_at_150_km_l94_94372

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l94_94372


namespace total_weight_of_courtney_marble_collection_l94_94965

def marble_weight_first_jar : ℝ := 80 * 0.35
def marble_weight_second_jar : ℝ := 160 * 0.45
def marble_weight_third_jar : ℝ := 20 * 0.25

/-- The total weight of Courtney's marble collection -/
theorem total_weight_of_courtney_marble_collection :
    marble_weight_first_jar + marble_weight_second_jar + marble_weight_third_jar = 105 := by
  sorry

end total_weight_of_courtney_marble_collection_l94_94965


namespace sqrt_11_custom_op_l94_94670

noncomputable def sqrt := Real.sqrt

def custom_op (x y : Real) := (x + y) ^ 2 - (x - y) ^ 2

theorem sqrt_11_custom_op : custom_op (sqrt 11) (sqrt 11) = 44 :=
by
  sorry

end sqrt_11_custom_op_l94_94670


namespace smallest_n_l94_94788

theorem smallest_n (n : ℕ) (h : 23 * n ≡ 789 [MOD 11]) : n = 9 :=
sorry

end smallest_n_l94_94788


namespace initial_action_figures_l94_94035

theorem initial_action_figures (x : ℕ) (h : x + 2 - 7 = 10) : x = 15 :=
by
  sorry

end initial_action_figures_l94_94035


namespace find_angle_B_l94_94589

theorem find_angle_B (A B C : ℝ) (a b c : ℝ)
  (hAngleA : A = 120) (ha : a = 2) (hb : b = 2 * Real.sqrt 3 / 3) : B = 30 :=
sorry

end find_angle_B_l94_94589


namespace integer_solution_unique_l94_94972

theorem integer_solution_unique (x y : ℝ) (h : -1 < (y - x) / (x + y) ∧ (y - x) / (x + y) < 2) (hyx : ∃ n : ℤ, y = n * x) : y = x :=
by
  sorry

end integer_solution_unique_l94_94972


namespace c_share_of_rent_l94_94266

/-- 
Given the conditions:
- a puts 10 oxen for 7 months,
- b puts 12 oxen for 5 months,
- c puts 15 oxen for 3 months,
- The rent of the pasture is Rs. 210,
Prove that C should pay Rs. 54 as his share of rent.
-/
noncomputable def total_rent : ℝ := 210
noncomputable def oxen_months_a : ℝ := 10 * 7
noncomputable def oxen_months_b : ℝ := 12 * 5
noncomputable def oxen_months_c : ℝ := 15 * 3
noncomputable def total_oxen_months : ℝ := oxen_months_a + oxen_months_b + oxen_months_c

theorem c_share_of_rent : (total_rent / total_oxen_months) * oxen_months_c = 54 :=
by
  sorry

end c_share_of_rent_l94_94266


namespace width_of_boxes_l94_94327

theorem width_of_boxes
  (total_volume : ℝ)
  (total_payment : ℝ)
  (cost_per_box : ℝ)
  (h1 : total_volume = 1.08 * 10^6)
  (h2 : total_payment = 120)
  (h3 : cost_per_box = 0.2) :
  (∃ w : ℝ, w = (total_volume / (total_payment / cost_per_box))^(1/3)) :=
by {
  sorry
}

end width_of_boxes_l94_94327


namespace problem_l94_94948

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l94_94948


namespace tyler_saltwater_aquariums_l94_94072

def num_animals_per_aquarium : ℕ := 39
def total_saltwater_animals : ℕ := 2184

theorem tyler_saltwater_aquariums : 
  total_saltwater_animals / num_animals_per_aquarium = 56 := 
by
  sorry

end tyler_saltwater_aquariums_l94_94072


namespace x_y_sum_l94_94468

theorem x_y_sum (x y : ℝ) (h1 : |x| - 2 * x + y = 1) (h2 : x - |y| + y = 8) :
  x + y = 17 ∨ x + y = 1 :=
by
  sorry

end x_y_sum_l94_94468


namespace a_range_l94_94011

variables {x a : ℝ}

def p (x : ℝ) : Prop := (4 * x - 3) ^ 2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (h : ∀ x, ¬p x → ¬q x a ∧ (∃ x, q x a ∧ ¬p x)) :
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end a_range_l94_94011


namespace multiplication_of_powers_of_10_l94_94969

theorem multiplication_of_powers_of_10 : (10 : ℝ) ^ 65 * (10 : ℝ) ^ 64 = (10 : ℝ) ^ 129 := by
  sorry

end multiplication_of_powers_of_10_l94_94969


namespace smallest_number_divisible_1_through_12_and_15_l94_94326

theorem smallest_number_divisible_1_through_12_and_15 :
  ∃ n, (∀ i, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ∧ 15 ∣ n ∧ n = 27720 :=
by {
  sorry
}

end smallest_number_divisible_1_through_12_and_15_l94_94326


namespace simplify_f_value_f_second_quadrant_l94_94403

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (3 * Real.pi / 2 - α)) / 
  (Real.cos (Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) : 
  f α = Real.cos α := 
sorry

theorem value_f_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (hcosα : Real.cos (π / 2 + α) = -1 / 3) :
  f α = - (2 * Real.sqrt 2) / 3 := 
sorry

end simplify_f_value_f_second_quadrant_l94_94403


namespace same_grades_percentage_l94_94409

theorem same_grades_percentage (total_students same_grades_A same_grades_B same_grades_C same_grades_D : ℕ) 
  (total_eq : total_students = 50) 
  (same_A : same_grades_A = 3) 
  (same_B : same_grades_B = 6) 
  (same_C : same_grades_C = 7) 
  (same_D : same_grades_D = 2) : 
  (same_grades_A + same_grades_B + same_grades_C + same_grades_D) * 100 / total_students = 36 := 
by
  sorry

end same_grades_percentage_l94_94409


namespace evaluate_at_5_l94_94763

def f(x: ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 27 * x^3 - 20 * x^2 - 72 * x + 40

theorem evaluate_at_5 : f 5 = 2515 :=
by
  sorry

end evaluate_at_5_l94_94763


namespace scaled_system_solution_l94_94834

theorem scaled_system_solution (a1 b1 c1 a2 b2 c2 x y : ℝ) 
  (h1 : a1 * 8 + b1 * 3 = c1) 
  (h2 : a2 * 8 + b2 * 3 = c2) : 
  4 * a1 * 10 + 3 * b1 * 5 = 5 * c1 ∧ 4 * a2 * 10 + 3 * b2 * 5 = 5 * c2 := 
by 
  sorry

end scaled_system_solution_l94_94834


namespace initial_population_of_first_village_l94_94962

theorem initial_population_of_first_village (P : ℕ) :
  (P - 1200 * 18) = (42000 + 800 * 18) → P = 78000 :=
by
  sorry

end initial_population_of_first_village_l94_94962


namespace calculate_train_length_l94_94074

noncomputable def train_length (speed_kmph : ℕ) (time_secs : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * time_secs
  total_distance - bridge_length_m

theorem calculate_train_length :
  train_length 60 14.998800095992321 140 = 110 :=
by
  sorry

end calculate_train_length_l94_94074


namespace simplify_fractions_l94_94675

theorem simplify_fractions : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end simplify_fractions_l94_94675


namespace find_x_y_sum_l94_94646

def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem find_x_y_sum (n x y : ℕ) (hn : n = 450) (hx : x > 0) (hy : y > 0)
  (hxsq : is_perfect_square (n * x))
  (hycube : is_perfect_cube (n * y)) :
  x + y = 62 :=
  sorry

end find_x_y_sum_l94_94646


namespace price_after_9_years_decreases_continuously_l94_94814

theorem price_after_9_years_decreases_continuously (price_current : ℝ) (price_after_9_years : ℝ) :
  (∀ k : ℕ, k % 3 = 0 → price_current = 8100 → price_after_9_years = 2400) :=
sorry

end price_after_9_years_decreases_continuously_l94_94814


namespace c_over_e_l94_94417

theorem c_over_e (a b c d e : ℝ) (h1 : 1 * 2 * 3 * a + 1 * 2 * 4 * a + 1 * 3 * 4 * a + 2 * 3 * 4 * a = -d)
  (h2 : 1 * 2 * 3 * 4 = e / a)
  (h3 : 1 * 2 * a + 1 * 3 * a + 1 * 4 * a + 2 * 3 * a + 2 * 4 * a + 3 * 4 * a = c) :
  c / e = 35 / 24 :=
by
  sorry

end c_over_e_l94_94417


namespace hike_took_one_hour_l94_94912

-- Define the constants and conditions
def initial_cups : ℕ := 6
def remaining_cups : ℕ := 1
def leak_rate : ℕ := 1 -- cups per hour
def drank_last_mile : ℚ := 1
def drank_first_3_miles_per_mile : ℚ := 2/3
def first_3_miles : ℕ := 3

-- Define the hike duration we want to prove
def hike_duration := 1

-- The total water drank
def total_drank := drank_last_mile + drank_first_3_miles_per_mile * first_3_miles

-- Prove the hike took 1 hour
theorem hike_took_one_hour :
  ∃ hours : ℕ, (initial_cups - remaining_cups = hours * leak_rate + total_drank) ∧ (hours = hike_duration) :=
by
  sorry

end hike_took_one_hour_l94_94912


namespace sum_of_polynomials_l94_94820

theorem sum_of_polynomials (d : ℕ) :
  let expr1 := 15 * d + 17 + 16 * d ^ 2
  let expr2 := 3 * d + 2
  let sum_expr := expr1 + expr2
  let a := 16
  let b := 18
  let c := 19
  sum_expr = a * d ^ 2 + b * d + c ∧ a + b + c = 53 := by
    sorry

end sum_of_polynomials_l94_94820


namespace solve_abs_inequality_l94_94697

theorem solve_abs_inequality (x : ℝ) (h : x ≠ 1) : 
  abs ((3 * x - 2) / (x - 1)) > 3 ↔ (5 / 6 < x ∧ x < 1) ∨ (x > 1) := 
by 
  sorry

end solve_abs_inequality_l94_94697


namespace george_initial_candy_l94_94959

theorem george_initial_candy (number_of_bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : number_of_bags = 8) (h2 : pieces_per_bag = 81) : 
  number_of_bags * pieces_per_bag = 648 := 
by 
  sorry

end george_initial_candy_l94_94959


namespace question_1_question_2_l94_94558

variable (m x : ℝ)
def f (x : ℝ) := |x + m|

theorem question_1 (h : f 1 + f (-2) ≥ 5) : 
  m ≤ -2 ∨ m ≥ 3 := sorry

theorem question_2 (hx : x ≠ 0) : 
  f (1 / x) + f (-x) ≥ 2 := sorry

end question_1_question_2_l94_94558


namespace songs_in_each_album_l94_94710

variable (X : ℕ)

theorem songs_in_each_album (h : 6 * X + 2 * X = 72) : X = 9 :=
by sorry

end songs_in_each_album_l94_94710


namespace problem_statement_l94_94893

theorem problem_statement (x y : ℤ) (h1 : x = 8) (h2 : y = 3) :
  (x - 2 * y) * (x + 2 * y) = 28 :=
by
  sorry

end problem_statement_l94_94893


namespace sum_of_digits_x_squared_l94_94098

theorem sum_of_digits_x_squared {r x p q : ℕ} (h_r : r ≤ 400) 
  (h_x_form : x = p * r^3 + p * r^2 + q * r + q) 
  (h_pq_condition : 7 * q = 17 * p) 
  (h_x2_form : ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + d * r^3 + c * r^2 + b * r + a ∧ d = 0) :
  p + p + q + q = 400 := 
sorry

end sum_of_digits_x_squared_l94_94098


namespace complete_consoles_production_rate_l94_94236

-- Define the production rates of each chip
def production_rate_A := 467
def production_rate_B := 413
def production_rate_C := 532
def production_rate_D := 356
def production_rate_E := 494

-- Define the maximum number of consoles that can be produced per day
def max_complete_consoles (A B C D E : ℕ) := min (min (min (min A B) C) D) E

-- Statement
theorem complete_consoles_production_rate :
  max_complete_consoles production_rate_A production_rate_B production_rate_C production_rate_D production_rate_E = 356 :=
by
  sorry

end complete_consoles_production_rate_l94_94236


namespace smallest_factor_of_36_sum_4_l94_94803

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end smallest_factor_of_36_sum_4_l94_94803


namespace squares_of_natural_numbers_l94_94031

theorem squares_of_natural_numbers (x y z : ℕ) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) : ∃ a b c : ℕ, x = a^2 ∧ y = b^2 ∧ z = c^2 := 
by
  sorry

end squares_of_natural_numbers_l94_94031


namespace continuous_function_solution_l94_94286

theorem continuous_function_solution (f : ℝ → ℝ) (a : ℝ) (h_continuous : Continuous f) (h_pos : 0 < a)
    (h_equation : ∀ x, f x = a^x * f (x / 2)) :
    ∃ C : ℝ, ∀ x, f x = C * a^(2 * x) := 
sorry

end continuous_function_solution_l94_94286


namespace Tim_younger_than_Jenny_l94_94219

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end Tim_younger_than_Jenny_l94_94219


namespace find_a1_l94_94226

theorem find_a1 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h2 : a 2 = 2)
  : a 1 = 1 / 2 :=
sorry

end find_a1_l94_94226


namespace is_minimum_value_l94_94531

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem is_minimum_value (h : ∀ x > 0, f x ≥ 0) : ∃ (a : ℝ) (h : a > 0), f a = 0 :=
by {
  sorry
}

end is_minimum_value_l94_94531


namespace Sarah_is_26_l94_94445

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end Sarah_is_26_l94_94445


namespace min_value_reciprocal_sum_l94_94585

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hmean : (a + b) / 2 = 1 / 2) : 
  ∃ c, c = (1 / a + 1 / b) ∧ c ≥ 4 := 
sorry

end min_value_reciprocal_sum_l94_94585


namespace parallelogram_height_l94_94042

theorem parallelogram_height (b A : ℝ) (h : ℝ) (h_base : b = 28) (h_area : A = 896) : h = A / b := by
  simp [h_base, h_area]
  norm_num
  sorry

end parallelogram_height_l94_94042


namespace painters_work_days_l94_94470

theorem painters_work_days 
  (six_painters_days : ℝ) (number_six_painters : ℝ) (total_work_units : ℝ)
  (number_four_painters : ℝ) 
  (h1 : number_six_painters = 6)
  (h2 : six_painters_days = 1.4)
  (h3 : total_work_units = number_six_painters * six_painters_days) 
  (h4 : number_four_painters = 4) :
  2 + 1 / 10 = total_work_units / number_four_painters :=
by
  rw [h3, h1, h2, h4]
  sorry

end painters_work_days_l94_94470


namespace inverse_sum_l94_94012

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_sum :
  (∃ x₁, g x₁ = -2 ∧ x₁ ≠ 5) ∨ (∃ x₂, g x₂ = 0 ∧ x₂ = 3) ∨ (∃ x₃, g x₃ = 4 ∧ x₃ = -1) → 
  g⁻¹ (-2) + g⁻¹ (0) + g⁻¹ (4) = 6 :=
by
  sorry

end inverse_sum_l94_94012


namespace jasmine_milk_gallons_l94_94598

theorem jasmine_milk_gallons (G : ℝ) 
  (coffee_cost_per_pound : ℝ) (milk_cost_per_gallon : ℝ) (total_cost : ℝ)
  (coffee_pounds : ℝ) :
  coffee_cost_per_pound = 2.50 →
  milk_cost_per_gallon = 3.50 →
  total_cost = 17 →
  coffee_pounds = 4 →
  total_cost - coffee_pounds * coffee_cost_per_pound = G * milk_cost_per_gallon →
  G = 2 :=
by
  intros
  sorry

end jasmine_milk_gallons_l94_94598


namespace max_lateral_surface_area_l94_94510

theorem max_lateral_surface_area : ∀ (x y : ℝ), 6 * x + 3 * y = 12 → (3 * x * y) ≤ 6 :=
by
  intros x y h
  have xy_le_2 : x * y ≤ 2 :=
    by
      sorry
  have max_area_6 : 3 * x * y ≤ 6 :=
    by
      sorry
  exact max_area_6

end max_lateral_surface_area_l94_94510


namespace hcf_of_two_numbers_900_l94_94268

theorem hcf_of_two_numbers_900 (A B H : ℕ) (h_lcm : lcm A B = H * 11 * 15) (h_A : A = 900) : gcd A B = 165 :=
by
  sorry

end hcf_of_two_numbers_900_l94_94268


namespace Brady_average_hours_l94_94224

-- Definitions based on conditions
def hours_per_day_April : ℕ := 6
def hours_per_day_June : ℕ := 5
def hours_per_day_September : ℕ := 8
def days_in_April : ℕ := 30
def days_in_June : ℕ := 30
def days_in_September : ℕ := 30

-- Definition to prove
def average_hours_per_month : ℕ := 190

-- Theorem statement
theorem Brady_average_hours :
  (hours_per_day_April * days_in_April + hours_per_day_June * days_in_June + hours_per_day_September * days_in_September) / 3 = average_hours_per_month :=
sorry

end Brady_average_hours_l94_94224


namespace sonic_leads_by_19_2_meters_l94_94661

theorem sonic_leads_by_19_2_meters (v_S v_D : ℝ)
  (h1 : ∀ t, t = 200 / v_S → 200 = v_S * t)
  (h2 : ∀ t, t = 184 / v_D → 184 = v_D * t)
  (h3 : v_S / v_D = 200 / 184)
  :  240 / v_S - (200 / v_S / (200 / 184) * 240) = 19.2 := by
  sorry

end sonic_leads_by_19_2_meters_l94_94661


namespace average_star_rating_is_four_l94_94316

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l94_94316


namespace part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l94_94686

noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def W (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

theorem part_1_relationship (x : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 40) :
  W x = -10 * x^2 + 500 * x - 4000 := by
  sorry

theorem part_2_solution (x : ℝ) (h₀ : W x = 1250) :
  x = 15 ∨ x = 35 := by
  sorry

theorem part_2_preferred (x : ℝ) (h₀ : W x = 1250) (h₁ : y 15 ≥ y 35) :
  x = 15 := by
  sorry

theorem part_3_max_W (x : ℝ) (h₀ : 28 ≤ x) (h₁ : x ≤ 35) :
  W x ≤ 2160 := by
  sorry

theorem part_3_max_at_28 :
  W 28 = 2160 := by
  sorry

end part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l94_94686


namespace rectangular_field_area_l94_94192

theorem rectangular_field_area (L B : ℝ) (h1 : B = 0.6 * L) (h2 : 2 * L + 2 * B = 800) : L * B = 37500 :=
by
  -- Proof will go here
  sorry

end rectangular_field_area_l94_94192


namespace total_clothes_washed_l94_94026

theorem total_clothes_washed (cally_white_shirts : ℕ) (cally_colored_shirts : ℕ) (cally_shorts : ℕ) (cally_pants : ℕ) 
                             (danny_white_shirts : ℕ) (danny_colored_shirts : ℕ) (danny_shorts : ℕ) (danny_pants : ℕ) 
                             (total_clothes : ℕ)
                             (hcally : cally_white_shirts = 10 ∧ cally_colored_shirts = 5 ∧ cally_shorts = 7 ∧ cally_pants = 6)
                             (hdanny : danny_white_shirts = 6 ∧ danny_colored_shirts = 8 ∧ danny_shorts = 10 ∧ danny_pants = 6)
                             (htotal : total_clothes = 58) : 
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + 
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants = total_clothes := 
by {
  sorry
}

end total_clothes_washed_l94_94026


namespace integer_pairs_satisfying_equation_l94_94995

theorem integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1)^3 + (p.2)^3 - 3*(p.1)^2 + 6*(p.2)^2 + 3*(p.1) + 12*(p.2) + 6 = 0}
  = {(1, -1), (2, -2)} := 
sorry

end integer_pairs_satisfying_equation_l94_94995


namespace difference_increased_decreased_l94_94727

theorem difference_increased_decreased (x : ℝ) (hx : x = 80) : 
  ((x * 1.125) - (x * 0.75)) = 30 := by
  have h1 : x * 1.125 = 90 := by rw [hx]; norm_num
  have h2 : x * 0.75 = 60 := by rw [hx]; norm_num
  rw [h1, h2]
  norm_num
  done

end difference_increased_decreased_l94_94727


namespace gcd_max_1001_l94_94782

theorem gcd_max_1001 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1001) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 143 := 
sorry

end gcd_max_1001_l94_94782


namespace clock_tick_intervals_l94_94005

theorem clock_tick_intervals (intervals_6: ℕ) (intervals_12: ℕ) (total_time_12: ℕ) (interval_time: ℕ):
  intervals_6 = 5 →
  intervals_12 = 11 →
  total_time_12 = 88 →
  interval_time = total_time_12 / intervals_12 →
  intervals_6 * interval_time = 40 :=
by
  intros h1 h2 h3 h4
  -- will continue proof here
  sorry

end clock_tick_intervals_l94_94005


namespace min_value_inverse_sum_l94_94160

theorem min_value_inverse_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) :
  (1 / x + 1 / y + 1 / z) ≥ 9 :=
  sorry

end min_value_inverse_sum_l94_94160


namespace gcd_40_120_45_l94_94663

theorem gcd_40_120_45 : Nat.gcd (Nat.gcd 40 120) 45 = 5 :=
by
  sorry

end gcd_40_120_45_l94_94663


namespace solution_to_fraction_problem_l94_94222

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end solution_to_fraction_problem_l94_94222


namespace total_combinations_meals_l94_94287

-- Define the total number of menu items
def menu_items : ℕ := 12

-- Define the function for computing the number of combinations of meals ordered by three people
def combinations_of_meals (n : ℕ) : ℕ := n * n * n

-- Theorem stating the total number of different combinations of meals is 1728
theorem total_combinations_meals : combinations_of_meals menu_items = 1728 :=
by
  -- Placeholder for actual proof
  sorry

end total_combinations_meals_l94_94287


namespace triangular_pyramid_height_l94_94514

noncomputable def pyramid_height (a b c h : ℝ) : Prop :=
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2

theorem triangular_pyramid_height {a b c h : ℝ} (h_gt_0 : h > 0) (a_gt_0 : a > 0) (b_gt_0 : b > 0) (c_gt_0 : c > 0) :
  pyramid_height a b c h := by
  sorry

end triangular_pyramid_height_l94_94514


namespace max_product_xyz_l94_94453

theorem max_product_xyz : ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 12 ∧ z ≤ 3 * x ∧ ∀ (a b c : ℕ), a + b + c = 12 → c ≤ 3 * a → 0 < a ∧ 0 < b ∧ 0 < c → a * b * c ≤ 48 :=
by
  sorry

end max_product_xyz_l94_94453


namespace range_of_m_l94_94623

-- Definitions
def is_circle_eqn (d e f : ℝ) : Prop :=
  d^2 + e^2 - 4 * f > 0

-- Main statement 
theorem range_of_m (m : ℝ) : 
  is_circle_eqn (-2) (-4) m → m < 5 :=
by
  intro h
  sorry

end range_of_m_l94_94623


namespace sphere_views_identical_l94_94528

-- Define the geometric shape as a type
inductive GeometricShape
| sphere
| cube
| other (name : String)

-- Define a function to get the view of a sphere
def view (s : GeometricShape) (direction : String) : String :=
  match s with
  | GeometricShape.sphere => "circle"
  | GeometricShape.cube => "square"
  | GeometricShape.other _ => "unknown"

-- The theorem to prove that a sphere has identical front, top, and side views
theorem sphere_views_identical :
  ∀ (direction1 direction2 : String), view GeometricShape.sphere direction1 = view GeometricShape.sphere direction2 :=
by
  intros direction1 direction2
  sorry

end sphere_views_identical_l94_94528


namespace range_of_a_l94_94549

theorem range_of_a (x y z a : ℝ) 
    (h1 : x > 0) 
    (h2 : y > 0) 
    (h3 : z > 0) 
    (h4 : x + y + z = 1) 
    (h5 : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) : 
    0 < a ∧ a ≤ 7 / 27 := 
  sorry

end range_of_a_l94_94549


namespace problem_I3_1_l94_94776

theorem problem_I3_1 (w x y z : ℝ) (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) (h3 : w > 0) : 
  w = 4 :=
by
  sorry

end problem_I3_1_l94_94776


namespace inradius_of_triangle_l94_94954

theorem inradius_of_triangle (A p s r : ℝ) 
  (h1 : A = (1/2) * p) 
  (h2 : p = 2 * s) 
  (h3 : A = r * s) : 
  r = 1 :=
by
  sorry

end inradius_of_triangle_l94_94954


namespace find_k_in_geometric_sequence_l94_94368

theorem find_k_in_geometric_sequence (c k : ℝ) (h1_nonzero : c ≠ 0)
  (S : ℕ → ℝ) (a : ℕ → ℝ) (h2 : ∀ n, a (n + 1) = c * a n)
  (h3 : ∀ n, S n = 3^n + k)
  (h4 : a 1 = 3 + k)
  (h5 : a 2 = S 2 - S 1)
  (h6 : a 3 = S 3 - S 2) : k = -1 := by
  sorry

end find_k_in_geometric_sequence_l94_94368


namespace problem_statement_l94_94506

-- Problem statement in Lean 4
theorem problem_statement (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b :=
by 
  sorry

end problem_statement_l94_94506


namespace carlos_picks_24_integers_l94_94025

def is_divisor (n m : ℕ) : Prop := m % n = 0

theorem carlos_picks_24_integers :
  ∃ (s : Finset ℕ), s.card = 24 ∧ ∀ n ∈ s, is_divisor n 4500 ∧ 1 ≤ n ∧ n ≤ 4500 ∧ n % 3 = 0 :=
by
  sorry

end carlos_picks_24_integers_l94_94025


namespace sum_of_solutions_eq_zero_l94_94671

theorem sum_of_solutions_eq_zero (x : ℝ) (h : 6 * x / 30 = 7 / x) :
  (∃ x₁ x₂ : ℝ, x₁^2 = 35 ∧ x₂^2 = 35 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l94_94671


namespace employed_females_part_time_percentage_l94_94038

theorem employed_females_part_time_percentage (P : ℕ) (hP1 : 0 < P)
  (h1 : ∀ x : ℕ, x = P * 6 / 10) -- 60% of P are employed
  (h2 : ∀ e : ℕ, e = P * 6 / 10) -- e is the number of employed individuals
  (h3 : ∀ f : ℕ, f = e * 4 / 10) -- 40% of employed are females
  (h4 : ∀ pt : ℕ, pt = f * 6 / 10) -- 60% of employed females are part-time
  (h5 : ∀ m : ℕ, m = P * 48 / 100) -- 48% of P are employed males
  (h6 : e = f + m) -- Employed individuals are either males or females
  : f * 6 / f * 10 = 60 := sorry

end employed_females_part_time_percentage_l94_94038


namespace sin_identity_l94_94529

theorem sin_identity (α : ℝ) (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 :=
sorry

end sin_identity_l94_94529


namespace largest_prime_divisor_l94_94961

-- Let n be a positive integer
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

-- Define that n equals the sum of the squares of its four smallest positive divisors
def is_sum_of_squares_of_smallest_divisors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a = 2 ∧ b = 5 ∧ c = 10 ∧ n = 1 + a^2 + b^2 + c^2

-- Prove that the largest prime divisor of n is 13
theorem largest_prime_divisor (n : ℕ) (h1 : is_positive_integer n) (h2 : is_sum_of_squares_of_smallest_divisors n) :
  ∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q ∧ q ∣ n → q ≤ p ∧ p = 13 :=
by
  sorry

end largest_prime_divisor_l94_94961


namespace common_ratio_of_geometric_sequence_l94_94576

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 2)
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  (∃ a1 : ℝ, ∃ q : ℝ,
    (∀ n, a n = a1 * q ^ (n - 1)) ∧ 
    q = 2) := 
by 
  sorry

end common_ratio_of_geometric_sequence_l94_94576


namespace mean_of_three_added_numbers_l94_94864

theorem mean_of_three_added_numbers (x y z : ℝ) :
  (∀ (s : ℝ), (s / 7 = 75) → (s + x + y + z) / 10 = 90) → (x + y + z) / 3 = 125 :=
by
  intro h
  sorry

end mean_of_three_added_numbers_l94_94864


namespace abes_present_age_l94_94094

theorem abes_present_age :
  ∃ A : ℕ, A + (A - 7) = 27 ∧ A = 17 :=
by
  sorry

end abes_present_age_l94_94094


namespace H_perimeter_is_44_l94_94876

-- Defining the dimensions of the rectangles
def vertical_rectangle_length : ℕ := 6
def vertical_rectangle_width : ℕ := 3
def horizontal_rectangle_length : ℕ := 6
def horizontal_rectangle_width : ℕ := 2

-- Defining the perimeter calculations, excluding overlapping parts
def vertical_rectangle_perimeter : ℕ := 2 * vertical_rectangle_length + 2 * vertical_rectangle_width
def horizontal_rectangle_perimeter : ℕ := 2 * horizontal_rectangle_length + 2 * horizontal_rectangle_width

-- Non-overlapping combined perimeter calculation for the 'H'
def H_perimeter : ℕ := 2 * vertical_rectangle_perimeter + horizontal_rectangle_perimeter - 2 * (2 * horizontal_rectangle_width)

-- Main theorem statement
theorem H_perimeter_is_44 : H_perimeter = 44 := by
  -- Provide a proof here
  sorry

end H_perimeter_is_44_l94_94876


namespace determine_A_l94_94649

open Real

theorem determine_A (A B C : ℝ)
  (h_decomposition : ∀ x, x ≠ 4 ∧ x ≠ -2 -> (x + 2) / (x^3 - 9 * x^2 + 14 * x + 24) = A / (x - 4) + B / (x - 3) + C / (x + 2)^2)
  (h_factorization : ∀ x, (x^3 - 9 * x^2 + 14 * x + 24) = (x - 4) * (x - 3) * (x + 2)^2) :
  A = 1 / 6 := 
sorry

end determine_A_l94_94649


namespace div_problem_l94_94847

variables (A B C : ℝ)

theorem div_problem (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : B = 93 :=
by {
  sorry
}

end div_problem_l94_94847


namespace Jessica_paid_1000_for_rent_each_month_last_year_l94_94183

/--
Jessica paid $200 for food each month last year.
Jessica paid $100 for car insurance each month last year.
This year her rent goes up by 30%.
This year food costs increase by 50%.
This year the cost of her car insurance triples.
Jessica pays $7200 more for her expenses over the whole year compared to last year.
-/
theorem Jessica_paid_1000_for_rent_each_month_last_year
  (R : ℝ) -- monthly rent last year
  (h1 : 12 * (0.30 * R + 100 + 200) = 7200) :
  R = 1000 :=
sorry

end Jessica_paid_1000_for_rent_each_month_last_year_l94_94183


namespace max_odd_integers_chosen_l94_94903

theorem max_odd_integers_chosen (a b c d e f : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h_prod_even : a * b * c * d * e * f % 2 = 0) : 
  (∀ n : ℕ, n = 5 → ∃ a b c d e, (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1) ∧ f % 2 = 0) :=
sorry

end max_odd_integers_chosen_l94_94903


namespace single_cakes_needed_l94_94432

theorem single_cakes_needed :
  ∀ (layer_cake_frosting single_cake_frosting cupcakes_frosting brownies_frosting : ℝ)
  (layer_cakes cupcakes brownies total_frosting : ℕ)
  (single_cakes_needed : ℝ),
  layer_cake_frosting = 1 →
  single_cake_frosting = 0.5 →
  cupcakes_frosting = 0.5 →
  brownies_frosting = 0.5 →
  layer_cakes = 3 →
  cupcakes = 6 →
  brownies = 18 →
  total_frosting = 21 →
  single_cakes_needed = (total_frosting - (layer_cakes * layer_cake_frosting + cupcakes * cupcakes_frosting + brownies * brownies_frosting)) / single_cake_frosting →
  single_cakes_needed = 12 :=
by
  intros
  sorry

end single_cakes_needed_l94_94432


namespace problem_3_at_7_hash_4_l94_94021

def oper_at (a b : ℕ) : ℚ := (a * b) / (a + b)
def oper_hash (c d : ℚ) : ℚ := c + d

theorem problem_3_at_7_hash_4 :
  oper_hash (oper_at 3 7) 4 = 61 / 10 := by
  sorry

end problem_3_at_7_hash_4_l94_94021


namespace simplify_expression_l94_94981

noncomputable def expression : ℝ :=
  (4 * (Real.sqrt 3 + Real.sqrt 7)) / (5 * Real.sqrt (3 + (1 / 2)))

theorem simplify_expression : expression = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end simplify_expression_l94_94981


namespace geom_prog_terms_exist_l94_94665

theorem geom_prog_terms_exist (b3 b6 : ℝ) (h1 : b3 = -1) (h2 : b6 = 27 / 8) :
  ∃ (b1 q : ℝ), b1 = -4 / 9 ∧ q = -3 / 2 :=
by
  sorry

end geom_prog_terms_exist_l94_94665


namespace probability_log3_N_integer_l94_94519
noncomputable def probability_log3_integer : ℚ :=
  let count := 2
  let total := 900
  count / total

theorem probability_log3_N_integer :
  probability_log3_integer = 1 / 450 :=
sorry

end probability_log3_N_integer_l94_94519


namespace river_depth_mid_June_l94_94919

theorem river_depth_mid_June (D : ℝ) : 
    (∀ (mid_May mid_June mid_July : ℝ),
    mid_May = 5 →
    mid_June = mid_May + D →
    mid_July = 3 * mid_June →
    mid_July = 45) →
    D = 10 :=
by
    sorry

end river_depth_mid_June_l94_94919


namespace trapezoid_bd_length_l94_94440

theorem trapezoid_bd_length
  (AB CD AC BD : ℝ)
  (tanC tanB : ℝ)
  (h1 : AB = 24)
  (h2 : CD = 15)
  (h3 : AC = 30)
  (h4 : tanC = 2)
  (h5 : tanB = 1.25)
  (h6 : AC ^ 2 = AB ^ 2 + (CD - AB) ^ 2) :
  BD = 9 * Real.sqrt 11 := by
  sorry

end trapezoid_bd_length_l94_94440


namespace find_loan_amount_l94_94196

-- Define the conditions
def rate_of_interest : ℝ := 0.06
def time_period : ℝ := 6
def interest_paid : ℝ := 432

-- Define the simple interest formula
def simple_interest (P r t : ℝ) : ℝ := P * r * t

-- State the theorem to prove the loan amount
theorem find_loan_amount (P : ℝ) (h1 : rate_of_interest = 0.06) (h2 : time_period = 6) (h3 : interest_paid = 432) (h4 : simple_interest P rate_of_interest time_period = interest_paid) : P = 1200 :=
by
  -- Here should be the proof, but it's omitted for now
  sorry

end find_loan_amount_l94_94196


namespace prob_A_and_B_succeed_prob_vaccine_A_successful_l94_94101

-- Define the probabilities of success for Company A, Company B, and Company C
def P_A := (2 : ℚ) / 3
def P_B := (1 : ℚ) / 2
def P_C := (3 : ℚ) / 5

-- Define the theorem statements

-- Theorem for the probability that both Company A and Company B succeed
theorem prob_A_and_B_succeed : P_A * P_B = 1 / 3 := by
  sorry

-- Theorem for the probability that vaccine A is successfully developed
theorem prob_vaccine_A_successful : 1 - ((1 - P_A) * (1 - P_B)) = 5 / 6 := by
  sorry

end prob_A_and_B_succeed_prob_vaccine_A_successful_l94_94101


namespace is_quadratic_equation_l94_94737

open Real

-- Define the candidate equations as statements in Lean 4
def equation_A (x : ℝ) : Prop := 3 * x^2 = 1 - 1 / (3 * x)
def equation_B (x m : ℝ) : Prop := (m - 2) * x^2 - m * x + 3 = 0
def equation_C (x : ℝ) : Prop := (x^2 - 3) * (x - 1) = 0
def equation_D (x : ℝ) : Prop := x^2 = 2

-- Prove that among the given equations, equation_D is the only quadratic equation
theorem is_quadratic_equation (x : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_A x = (a * x^2 + b * x + c = 0)) ∨
  (∃ m a b c : ℝ, a ≠ 0 ∧ equation_B x m = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_C x = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_D x = (a * x^2 + b * x + c = 0)) := by
  sorry

end is_quadratic_equation_l94_94737


namespace pentagon_area_l94_94511

theorem pentagon_area {a b c d e : ℕ} (split: ℕ) (non_parallel1 non_parallel2 parallel1 parallel2 : ℕ)
  (h1 : a = 16) (h2 : b = 25) (h3 : c = 30) (h4 : d = 26) (h5 : e = 25)
  (split_condition : a + b + c + d + e = 5 * split)
  (np_condition1: non_parallel1 = c) (np_condition2: non_parallel2 = a)
  (p_condition1: parallel1 = d) (p_condition2: parallel2 = e)
  (area_triangle: 1 / 2 * b * a = 200)
  (area_trapezoid: 1 / 2 * (parallel1 + parallel2) * non_parallel1 = 765) :
  a + b + c + d + e = 965 := by
  sorry

end pentagon_area_l94_94511


namespace sarah_amount_l94_94516

theorem sarah_amount:
  ∀ (X : ℕ), (X + (X + 50) = 300) → X = 125 := by
  sorry

end sarah_amount_l94_94516


namespace Kates_hair_length_l94_94282

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l94_94282


namespace value_of_expression_l94_94503

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l94_94503


namespace combined_supply_duration_l94_94307

variable (third_of_pill_per_third_day : ℕ → Prop)
variable (alternate_days : ℕ → ℕ → Prop)
variable (supply : ℕ)
variable (days_in_month : ℕ)

-- Conditions:
def one_third_per_third_day (p: ℕ) (d: ℕ) : Prop := 
  third_of_pill_per_third_day d ∧ alternate_days d (d + 3)
def total_supply (s: ℕ) := s = 60
def duration_per_pill (d: ℕ) := d = 9
def month_days (m: ℕ) := m = 30

-- Proof Problem Statement:
theorem combined_supply_duration :
  ∀ (s t: ℕ), total_supply s ∧ duration_per_pill t ∧ month_days 30 → 
  (s * t / 30) = 18 :=
by
  intros s t h
  sorry

end combined_supply_duration_l94_94307


namespace team_size_is_nine_l94_94881

noncomputable def number_of_workers (n x y : ℕ) : ℕ :=
  if 7 * n = (n - 2) * x ∧ 7 * n = (n - 6) * y then n else 0

theorem team_size_is_nine (x y : ℕ) :
  number_of_workers 9 x y = 9 :=
by
  sorry

end team_size_is_nine_l94_94881


namespace max_floor_l94_94365

theorem max_floor (x : ℝ) (h : ⌊(x + 4) / 10⌋ = 5) : ⌊(6 * x) / 5⌋ = 67 :=
  sorry

end max_floor_l94_94365


namespace trajectory_point_M_l94_94560

theorem trajectory_point_M (x y : ℝ) : 
  (∃ (m n : ℝ), x^2 + y^2 = 9 ∧ (m = x) ∧ (n = 3 * y)) → 
  (x^2 / 9 + y^2 = 1) :=
by
  sorry

end trajectory_point_M_l94_94560


namespace average_marks_passed_l94_94951

noncomputable def total_candidates := 120
noncomputable def total_average_marks := 35
noncomputable def passed_candidates := 100
noncomputable def failed_candidates := total_candidates - passed_candidates
noncomputable def average_marks_failed := 15
noncomputable def total_marks := total_average_marks * total_candidates
noncomputable def total_marks_failed := average_marks_failed * failed_candidates

theorem average_marks_passed :
  ∃ P, P * passed_candidates + total_marks_failed = total_marks ∧ P = 39 := by
  sorry

end average_marks_passed_l94_94951


namespace inequality_problems_l94_94637

theorem inequality_problems
  (m n l : ℝ)
  (h1 : m > n)
  (h2 : n > l) :
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) :=
by
  sorry

end inequality_problems_l94_94637


namespace sum_of_reciprocals_l94_94150

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) : 
  (1 / x + 1 / y = 1 / 2) :=
by 
  sorry

end sum_of_reciprocals_l94_94150


namespace find_fractions_l94_94555

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end find_fractions_l94_94555


namespace complement_union_correct_l94_94653

open Set

variable (U : Set Int)
variable (A B : Set Int)

theorem complement_union_correct (hU : U = {-2, -1, 0, 1, 2}) (hA : A = {1, 2}) (hB : B = {-2, 1, 2}) :
  A ∪ (U \ B) = {-1, 0, 1, 2} := by
  rw [hU, hA, hB]
  simp
  sorry

end complement_union_correct_l94_94653


namespace no_even_is_prime_equiv_l94_94162

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end no_even_is_prime_equiv_l94_94162


namespace value_of_m_l94_94278

theorem value_of_m : ∃ (m : ℕ), (3 * 4 * 5 * m = Nat.factorial 8) ∧ m = 672 := by
  sorry

end value_of_m_l94_94278


namespace find_f2_l94_94165

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (f x) = (x ^ 2 - x) / 2 * f x + 2 - x

theorem find_f2 : f 2 = 2 :=
by
  sorry

end find_f2_l94_94165


namespace speed_of_A_is_24_speed_of_A_is_18_l94_94735

-- Definitions for part 1
def speed_of_B (x : ℝ) := x
def speed_of_A_1 (x : ℝ) := 1.2 * x
def distance_AB := 30 -- kilometers
def distance_B_rides_first := 2 -- kilometers
def time_A_catches_up := 0.5 -- hours

theorem speed_of_A_is_24 (x : ℝ) (h1 : 0.6 * x = 2 + 0.5 * x) : speed_of_A_1 x = 24 := by
  sorry

-- Definitions for part 2
def speed_of_A_2 (y : ℝ) := 1.2 * y
def time_B_rides_first := 1/3 -- hours
def time_difference := 1/3 -- hours

theorem speed_of_A_is_18 (y : ℝ) (h2 : (30 / y) - (30 / (1.2 * y)) = 1/3) : speed_of_A_2 y = 18 := by
  sorry

end speed_of_A_is_24_speed_of_A_is_18_l94_94735


namespace train_speed_correct_l94_94251

-- Definitions based on the conditions in a)
def train_length_meters : ℝ := 160
def time_seconds : ℝ := 4

-- Correct answer identified in b)
def expected_speed_kmh : ℝ := 144

-- Proof statement verifying that speed computed from the conditions equals the expected speed
theorem train_speed_correct :
  train_length_meters / 1000 / (time_seconds / 3600) = expected_speed_kmh :=
by
  sorry

end train_speed_correct_l94_94251


namespace quadratic_solution_l94_94592

theorem quadratic_solution :
  (∀ x : ℝ, 3 * x^2 - 13 * x + 5 = 0 → 
           x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6) 
  := by
  sorry

end quadratic_solution_l94_94592


namespace schlaf_flachs_divisible_by_271_l94_94709

theorem schlaf_flachs_divisible_by_271 
(S C F H L A : ℕ) 
(hS : S ≠ 0) 
(hF : F ≠ 0) 
(hS_digit : S < 10)
(hC_digit : C < 10)
(hF_digit : F < 10)
(hH_digit : H < 10)
(hL_digit : L < 10)
(hA_digit : A < 10) :
  (100000 * S + 10000 * C + 1000 * H + 100 * L + 10 * A + F - 
   (100000 * F + 10000 * L + 1000 * A + 100 * C + 10 * H + S)) % 271 = 0 ↔ 
  C = L ∧ H = A := 
sorry

end schlaf_flachs_divisible_by_271_l94_94709


namespace birdhouse_flight_distance_l94_94001

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end birdhouse_flight_distance_l94_94001


namespace initial_apples_l94_94156

-- Define the number of initial fruits
def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def fruits_given : ℕ := 40
def fruits_left : ℕ := 15

-- Define the equation for the initial number of fruits
def initial_total_fruits (A : ℕ) : Prop :=
  initial_plums + initial_guavas + A = fruits_left + fruits_given

-- Define the proof problem to find the number of apples
theorem initial_apples : ∃ A : ℕ, initial_total_fruits A ∧ A = 21 :=
  by
    sorry

end initial_apples_l94_94156


namespace value_of_m_l94_94040

theorem value_of_m (m x : ℝ) (h : x = 3) (h_eq : 3 * m - 2 * x = 6) : m = 4 := by
  -- Given x = 3
  subst h
  -- Now we have to show m = 4
  sorry

end value_of_m_l94_94040


namespace bacteria_doubling_time_l94_94651

noncomputable def doubling_time_population 
    (initial final : ℝ) 
    (time : ℝ) 
    (growth_factor : ℕ) : ℝ :=
    time / (Real.log growth_factor / Real.log 2)

theorem bacteria_doubling_time :
  doubling_time_population 1000 500000 26.897352853986263 500 = 0.903 :=
by
  sorry

end bacteria_doubling_time_l94_94651


namespace increasing_function_solve_inequality_find_range_l94_94343

noncomputable def f : ℝ → ℝ := sorry
def a1 := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x
def a2 := f 1 = 1
def a3 := ∀ m n : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ -1 ≤ n ∧ n ≤ 1 ∧ m + n ≠ 0 → (f m + f n) / (m + n) > 0

-- Statement for question (1)
theorem increasing_function : 
  (∀ x y : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x < f y) :=
by 
  apply sorry

-- Statement for question (2)
theorem solve_inequality (x : ℝ) :
  (f (x^2 - 1) + f (3 - 3*x) < 0 ↔ 1 < x ∧ x ≤ 4/3) :=
by 
  apply sorry

-- Statement for question (3)
theorem find_range (t : ℝ) :
  (∀ x a : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ a ∧ a ≤ 1 → f x ≤ t^2 - 2*a*t + 1) 
  ↔ (2 ≤ t ∨ t ≤ -2 ∨ t = 0) :=
by 
  apply sorry

end increasing_function_solve_inequality_find_range_l94_94343


namespace regular_price_adult_ticket_l94_94836

theorem regular_price_adult_ticket : 
  ∀ (concessions_cost_children cost_adult1 cost_adult2 cost_adult3 cost_adult4 cost_adult5
       ticket_cost_child cost_discount1 cost_discount2 cost_discount3 total_cost : ℝ),
  (concessions_cost_children = 3) → 
  (cost_adult1 = 5) → 
  (cost_adult2 = 6) → 
  (cost_adult3 = 7) → 
  (cost_adult4 = 4) → 
  (cost_adult5 = 9) → 
  (ticket_cost_child = 7) → 
  (cost_discount1 = 3) → 
  (cost_discount2 = 2) → 
  (cost_discount3 = 1) → 
  (total_cost = 139) → 
  (∀ A : ℝ, total_cost = 
    (2 * concessions_cost_children + cost_adult1 + cost_adult2 + cost_adult3 + cost_adult4 + cost_adult5) + 
    (2 * ticket_cost_child + (2 * A + (A - cost_discount1) + (A - cost_discount2) + (A - cost_discount3))) → 
    5 * A - 6 = 88 →
    A = 18.80) :=
by
  intros
  sorry

end regular_price_adult_ticket_l94_94836


namespace radio_show_length_l94_94664

theorem radio_show_length :
  let s3 := 10
  let s2 := s3 + 5
  let s4 := s2 / 2
  let s5 := 2 * s4
  let s1 := 2 * (s2 + s3 + s4 + s5)
  s1 + s2 + s3 + s4 + s5 = 142.5 :=
by
  sorry

end radio_show_length_l94_94664


namespace janice_work_days_l94_94674

variable (dailyEarnings : Nat)
variable (overtimeEarnings : Nat)
variable (numOvertimeShifts : Nat)
variable (totalEarnings : Nat)

theorem janice_work_days
    (h1 : dailyEarnings = 30)
    (h2 : overtimeEarnings = 15)
    (h3 : numOvertimeShifts = 3)
    (h4 : totalEarnings = 195)
    : let overtimeTotal := numOvertimeShifts * overtimeEarnings
      let regularEarnings := totalEarnings - overtimeTotal
      let workDays := regularEarnings / dailyEarnings
      workDays = 5 :=
by
  sorry

end janice_work_days_l94_94674


namespace carriage_and_people_l94_94680

variable {x y : ℕ}

theorem carriage_and_people :
  (3 * (x - 2) = y) ∧ (2 * x + 9 = y) :=
sorry

end carriage_and_people_l94_94680


namespace abc_eq_l94_94896

theorem abc_eq (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a * a - 1) ^ 2) : a = b :=
sorry

end abc_eq_l94_94896


namespace min_value_of_alpha_beta_l94_94911

theorem min_value_of_alpha_beta 
  (k : ℝ)
  (h_k : k ≤ -4 ∨ k ≥ 5)
  (α β : ℝ)
  (h_αβ : α^2 - 2 * k * α + (k + 20) = 0 ∧ β^2 - 2 * k * β + (k + 20) = 0) :
  (α + 1) ^ 2 + (β + 1) ^ 2 = 18 → k = -4 :=
sorry

end min_value_of_alpha_beta_l94_94911


namespace find_n_times_s_l94_94024

noncomputable def g (x : ℝ) : ℝ :=
  if x = 1 then 2011
  else if x = 2 then (1 / 2 + 2010)
  else 0 /- For purposes of the problem -/

theorem find_n_times_s :
  (∀ x y : ℝ, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x + 1 / y + 2010)) →
  ∃ n s : ℝ, n = 1 ∧ s = (4021 / 2) ∧ n * s = 4021 / 2 :=
by
  sorry

end find_n_times_s_l94_94024


namespace percentage_of_students_passed_l94_94324

def total_students : ℕ := 740
def failed_students : ℕ := 481
def passed_students : ℕ := total_students - failed_students
def pass_percentage : ℚ := (passed_students / total_students) * 100

theorem percentage_of_students_passed : pass_percentage = 35 := by
  sorry

end percentage_of_students_passed_l94_94324


namespace chairs_to_remove_l94_94650

/-- A conference hall is setting up seating for a lecture with specific conditions.
    Given the total number of chairs, chairs per row, and participants expected to attend,
    prove the number of chairs to be removed to have complete rows with the least number of empty seats. -/
theorem chairs_to_remove
  (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  total_chairs - (chairs_per_row * ((expected_participants + chairs_per_row - 1) / chairs_per_row)) = 75 :=
by
  sorry

end chairs_to_remove_l94_94650


namespace arctan_sum_eq_pi_div_two_l94_94829

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
sorry

end arctan_sum_eq_pi_div_two_l94_94829


namespace remaining_budget_after_purchases_l94_94126

theorem remaining_budget_after_purchases :
  let budget := 80
  let fried_chicken_cost := 12
  let beef_cost_per_pound := 3
  let beef_quantity := 4.5
  let soup_cost_per_can := 2
  let soup_quantity := 3
  let milk_original_price := 4
  let milk_discount := 0.10
  let beef_cost := beef_quantity * beef_cost_per_pound
  let paid_soup_quantity := soup_quantity / 2
  let milk_discounted_price := milk_original_price * (1 - milk_discount)
  let total_cost := fried_chicken_cost + beef_cost + (paid_soup_quantity * soup_cost_per_can) + milk_discounted_price
  let remaining_budget := budget - total_cost
  remaining_budget = 47.90 :=
by
  sorry

end remaining_budget_after_purchases_l94_94126


namespace XT_value_l94_94535

noncomputable def AB := 15
noncomputable def BC := 20
noncomputable def height_P := 30
noncomputable def volume_ratio := 9

theorem XT_value 
  (AB BC height_P : ℕ)
  (volume_ratio : ℕ)
  (h1 : AB = 15)
  (h2 : BC = 20)
  (h3 : height_P = 30)
  (h4 : volume_ratio = 9) : 
  ∃ (m n : ℕ), m + n = 97 ∧ m.gcd n = 1 :=
by sorry

end XT_value_l94_94535


namespace find_p_l94_94320

theorem find_p (p : ℕ) (h : 81^6 = 3^p) : p = 24 :=
sorry

end find_p_l94_94320


namespace janet_saving_l94_94839

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end janet_saving_l94_94839


namespace valid_plantings_count_l94_94725

-- Define the grid structure
structure Grid3x3 :=
  (sections : Fin 9 → String)

noncomputable def crops := ["corn", "wheat", "soybeans", "potatoes", "oats"]

-- Define the adjacency relationships and restrictions as predicates
def adjacent (i j : Fin 9) : Prop :=
  (i = j + 1 ∧ j % 3 ≠ 2) ∨ (i = j - 1 ∧ i % 3 ≠ 2) ∨ (i = j + 3) ∨ (i = j - 3)

def valid_crop_planting (g : Grid3x3) : Prop :=
  ∀ i j, adjacent i j →
    (¬(g.sections i = "corn" ∧ g.sections j = "wheat") ∧ 
    ¬(g.sections i = "wheat" ∧ g.sections j = "corn") ∧
    ¬(g.sections i = "soybeans" ∧ g.sections j = "potatoes") ∧
    ¬(g.sections i = "potatoes" ∧ g.sections j = "soybeans") ∧
    ¬(g.sections i = "oats" ∧ g.sections j = "potatoes") ∧ 
    ¬(g.sections i = "potatoes" ∧ g.sections j = "oats"))

noncomputable def count_valid_plantings : Nat :=
  -- Placeholder for the actual count computing function
  sorry

theorem valid_plantings_count : count_valid_plantings = 5 :=
  sorry

end valid_plantings_count_l94_94725


namespace mark_first_vaccine_wait_time_l94_94252

-- Define the variables and conditions
variable (x : ℕ)
variable (total_wait_time : ℕ)
variable (second_appointment_wait : ℕ)
variable (effectiveness_wait : ℕ)

-- Given conditions
axiom h1 : second_appointment_wait = 20
axiom h2 : effectiveness_wait = 14
axiom h3 : total_wait_time = 38

-- The statement to be proven
theorem mark_first_vaccine_wait_time
  (h4 : x + second_appointment_wait + effectiveness_wait = total_wait_time) :
  x = 4 := by
  sorry

end mark_first_vaccine_wait_time_l94_94252


namespace robin_bobin_can_meet_prescription_l94_94638

def large_gr_pill : ℝ := 11
def medium_gr_pill : ℝ := -1.1
def small_gr_pill : ℝ := -0.11
def prescribed_gr : ℝ := 20.13

theorem robin_bobin_can_meet_prescription :
  ∃ (large : ℕ) (medium : ℕ) (small : ℕ), large ≥ 1 ∧ medium ≥ 1 ∧ small ≥ 1 ∧
  large_gr_pill * large + medium_gr_pill * medium + small_gr_pill * small = prescribed_gr :=
sorry

end robin_bobin_can_meet_prescription_l94_94638


namespace daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l94_94431

-- Definitions based on conditions
def purchase_price : ℝ := 30
def max_selling_price : ℝ := 55
def linear_relationship (x : ℝ) : ℝ := -2 * x + 140
def profit (x : ℝ) : ℝ := (x - purchase_price) * linear_relationship x

-- Part 1: Daily profit when selling price is 35 yuan
theorem daily_profit_35 : profit 35 = 350 :=
  sorry

-- Part 2: Selling price for a daily profit of 600 yuan
theorem selling_price_for_600_profit (x : ℝ) (h1 : 30 ≤ x) (h2 : x ≤ 55) : profit x = 600 → x = 40 :=
  sorry

-- Part 3: Possibility of daily profit of 900 yuan
theorem no_900_profit_possible (h1 : ∀ x, 30 ≤ x ∧ x ≤ 55 → profit x ≠ 900) : ¬ ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ profit x = 900 :=
  sorry

end daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l94_94431


namespace num_lineups_l94_94395

-- Define the given conditions
def num_players : ℕ := 12
def num_lineman : ℕ := 4
def num_qb_among_lineman : ℕ := 2
def num_running_backs : ℕ := 3

-- State the problem and the result as a theorem
theorem num_lineups : 
  (num_lineman * (num_qb_among_lineman) * (num_running_backs) * (num_players - num_lineman - num_qb_among_lineman - num_running_backs + 3) = 216) := 
by
  -- The proof will go here
  sorry

end num_lineups_l94_94395


namespace find_divisor_l94_94442

theorem find_divisor (d : ℕ) : ((23 = (d * 7) + 2) → d = 3) :=
by
  sorry

end find_divisor_l94_94442


namespace add_neg_two_and_three_l94_94073

theorem add_neg_two_and_three : -2 + 3 = 1 :=
by
  sorry

end add_neg_two_and_three_l94_94073


namespace sum_eq_two_l94_94668

theorem sum_eq_two (x y : ℝ) (hx : x^3 - 3 * x^2 + 5 * x = 1) (hy : y^3 - 3 * y^2 + 5 * y = 5) : x + y = 2 := 
sorry

end sum_eq_two_l94_94668


namespace intersection_M_N_l94_94182

def M : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l94_94182


namespace find_m_value_l94_94678

open Real

-- Define the vectors a and b as specified in the problem
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (3, -2)

-- Define the sum of vectors a and b
def vec_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the dot product of the vector sum with vector b to be zero as the given condition
def dot_product (m : ℝ) : ℝ := (vec_sum m).1 * vec_b.1 + (vec_sum m).2 * vec_b.2

-- The theorem to prove that given the defined conditions, m equals 8
theorem find_m_value (m : ℝ) (h : dot_product m = 0) : m = 8 := by
  sorry

end find_m_value_l94_94678


namespace problem_I_inequality_solution_problem_II_condition_on_b_l94_94056

-- Define the function f(x).
def f (x : ℝ) : ℝ := |x - 2|

-- Problem (I): Proving the solution set to the given inequality.
theorem problem_I_inequality_solution (x : ℝ) : 
  f x + f (x + 1) ≥ 5 ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

-- Problem (II): Proving the condition on |b|.
theorem problem_II_condition_on_b (a b : ℝ) (ha : |a| > 1) (h : f (a * b) > |a| * f (b / a)) :
  |b| > 2 :=
sorry

end problem_I_inequality_solution_problem_II_condition_on_b_l94_94056


namespace average_of_rest_l94_94348

theorem average_of_rest 
  (total_students : ℕ)
  (marks_5_students : ℕ)
  (marks_3_students : ℕ)
  (marks_others : ℕ)
  (average_class : ℚ)
  (remaining_students : ℕ)
  (expected_average : ℚ) 
  (h1 : total_students = 27) 
  (h2 : marks_5_students = 5 * 95) 
  (h3 : marks_3_students = 3 * 0) 
  (h4 : average_class = 49.25925925925926) 
  (h5 : remaining_students = 27 - 5 - 3) 
  (h6 : (marks_5_students + marks_3_students + marks_others) = total_students * average_class)
  : marks_others / remaining_students = expected_average :=
sorry

end average_of_rest_l94_94348


namespace tank_cost_minimization_l94_94452

def volume := 4800
def depth := 3
def cost_per_sqm_bottom := 150
def cost_per_sqm_walls := 120

theorem tank_cost_minimization (x : ℝ) 
  (S1 : ℝ := volume / depth)
  (S2 : ℝ := 6 * (x + (S1 / x)))
  (cost := cost_per_sqm_bottom * S1 + cost_per_sqm_walls * S2) :
  (x = 40) → cost = 297600 :=
sorry

end tank_cost_minimization_l94_94452


namespace relationship_even_increasing_l94_94422

-- Even function definition
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- Monotonically increasing function definition on interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

variable {f : ℝ → ℝ}

-- The proof problem statement
theorem relationship_even_increasing (h_even : even_function f) (h_increasing : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
by
  sorry

end relationship_even_increasing_l94_94422


namespace rectangle_area_l94_94401

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end rectangle_area_l94_94401


namespace num_games_round_robin_l94_94639

-- There are 10 classes in the second grade, each class forms one team.
def num_teams := 10

-- A round-robin format means each team plays against every other team once.
def num_games (n : Nat) := n * (n - 1) / 2

-- Proving the total number of games played with num_teams equals to 45
theorem num_games_round_robin : num_games num_teams = 45 := by
  sorry

end num_games_round_robin_l94_94639


namespace solve_for_a_and_b_l94_94293

theorem solve_for_a_and_b (a b : ℤ) :
  (∀ x : ℤ, (x + a) * (x - 2) = x^2 + b * x - 6) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_for_a_and_b_l94_94293


namespace find_x_l94_94381

theorem find_x (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := 
sorry

end find_x_l94_94381


namespace box_width_l94_94393

theorem box_width (rate : ℝ) (time : ℝ) (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) : 
  rate = 4 ∧ time = 21 ∧ length = 7 ∧ depth = 2 ∧ volume = rate * time ∧ volume = length * width * depth → width = 6 :=
by
  sorry

end box_width_l94_94393


namespace can_capacity_l94_94520

/-- Given a can with a mixture of milk and water in the ratio 4:3, and adding 10 liters of milk
results in the can being full and changes the ratio to 5:2, prove that the capacity of the can is 30 liters. -/
theorem can_capacity (x : ℚ)
  (h1 : 4 * x + 3 * x + 10 = 30)
  (h2 : (4 * x + 10) / (3 * x) = 5 / 2) :
  4 * x + 3 * x + 10 = 30 := 
by sorry

end can_capacity_l94_94520


namespace simplify_and_evaluate_l94_94713

variable (x y : ℝ)

theorem simplify_and_evaluate (h : x / y = 3) : 
  (1 + y^2 / (x^2 - y^2)) * (x - y) / x = 3 / 4 :=
by
  sorry

end simplify_and_evaluate_l94_94713


namespace find_other_solution_l94_94081

theorem find_other_solution (x₁ : ℚ) (x₂ : ℚ) 
  (h₁ : x₁ = 3 / 4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) 
  (eq : 72 * x₂^2 + 39 * x₂ - 18 = 0 ∧ x₂ ≠ x₁) : 
  x₂ = -31 / 6 := 
sorry

end find_other_solution_l94_94081


namespace z_is_greater_by_50_percent_of_w_l94_94760

variable (w q y z : ℝ)

def w_is_60_percent_q : Prop := w = 0.60 * q
def q_is_60_percent_y : Prop := q = 0.60 * y
def z_is_54_percent_y : Prop := z = 0.54 * y

theorem z_is_greater_by_50_percent_of_w (h1 : w_is_60_percent_q w q) 
                                        (h2 : q_is_60_percent_y q y) 
                                        (h3 : z_is_54_percent_y z y) : 
  ((z - w) / w) * 100 = 50 :=
sorry

end z_is_greater_by_50_percent_of_w_l94_94760


namespace max_non_overlapping_areas_l94_94993

theorem max_non_overlapping_areas (n : ℕ) : 
  ∃ (max_areas : ℕ), max_areas = 3 * n := by
  sorry

end max_non_overlapping_areas_l94_94993


namespace water_to_add_l94_94174

theorem water_to_add (x : ℚ) (alcohol water : ℚ) (ratio : ℚ) :
  alcohol = 4 → water = 4 →
  (3 : ℚ) / (3 + 5) = (3 : ℚ) / 8 →
  (5 : ℚ) / (3 + 5) = (5 : ℚ) / 8 →
  ratio = 5 / 8 →
  (4 + x) / (8 + x) = ratio →
  x = 8 / 3 :=
by
  intros
  sorry

end water_to_add_l94_94174


namespace difference_between_numbers_l94_94487

theorem difference_between_numbers (x y : ℕ) (h : x - y = 9) :
  (10 * x + y) - (10 * y + x) = 81 :=
by
  sorry

end difference_between_numbers_l94_94487


namespace students_total_l94_94463

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l94_94463


namespace value_of_expression_l94_94045

variables (x y z : ℝ)

axiom eq1 : 3 * x - 4 * y - 2 * z = 0
axiom eq2 : 2 * x + 6 * y - 21 * z = 0
axiom z_ne_zero : z ≠ 0

theorem value_of_expression : (x^2 + 4 * x * y) / (y^2 + z^2) = 7 :=
sorry

end value_of_expression_l94_94045


namespace number_of_cities_sampled_from_group_B_l94_94548

variable (N_total : ℕ) (N_A : ℕ) (N_B : ℕ) (N_C : ℕ) (S : ℕ)

theorem number_of_cities_sampled_from_group_B :
    N_total = 48 → 
    N_A = 10 → 
    N_B = 18 → 
    N_C = 20 → 
    S = 16 → 
    (N_B * S) / N_total = 6 :=
by
  sorry

end number_of_cities_sampled_from_group_B_l94_94548


namespace cookie_total_l94_94994

-- Definitions of the conditions
def rows_large := 5
def rows_medium := 4
def rows_small := 6
def cookies_per_row_large := 6
def cookies_per_row_medium := 7
def cookies_per_row_small := 8
def number_of_trays := 4
def extra_row_large_first_tray := 1
def total_large_cookies := rows_large * cookies_per_row_large * number_of_trays + extra_row_large_first_tray * cookies_per_row_large
def total_medium_cookies := rows_medium * cookies_per_row_medium * number_of_trays
def total_small_cookies := rows_small * cookies_per_row_small * number_of_trays

-- Theorem to prove the total number of cookies is 430
theorem cookie_total : 
  total_large_cookies + total_medium_cookies + total_small_cookies = 430 :=
by
  -- Proof is omitted
  sorry

end cookie_total_l94_94994


namespace hyperbola_range_m_l94_94436

theorem hyperbola_range_m (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (|m| - 1)) - (y^2 / (m - 2)) = 1) ↔ (m < -1) ∨ (m > 2) := 
by
  sorry

end hyperbola_range_m_l94_94436


namespace reduced_expression_none_of_these_l94_94978

theorem reduced_expression_none_of_these (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : b ≠ a^2) (h4 : ab ≠ a^3) :
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 1 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (b^2 + b) / (b - a^2) ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 0 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (a^2 + b) / (a^2 - b) :=
by
  sorry

end reduced_expression_none_of_these_l94_94978


namespace reservoir_original_content_l94_94078

noncomputable def original_content (T O : ℝ) : Prop :=
  (80 / 100) * T = O + 120 ∧
  O = (50 / 100) * T

theorem reservoir_original_content (T : ℝ) (h1 : (80 / 100) * T = (50 / 100) * T + 120) : 
  (50 / 100) * T = 200 :=
by
  sorry

end reservoir_original_content_l94_94078


namespace prove_a_minus_b_plus_c_eq_3_l94_94740

variable {a b c m n : ℝ}

theorem prove_a_minus_b_plus_c_eq_3 
    (h : ∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) :
    a - b + c = 3 :=
sorry

end prove_a_minus_b_plus_c_eq_3_l94_94740


namespace average_correct_l94_94430

theorem average_correct :
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1252140 + 2345) / 10 = 125831.9 := 
sorry

end average_correct_l94_94430


namespace distance_traveled_l94_94429

theorem distance_traveled :
  ∫ t in (3:ℝ)..(5:ℝ), (2 * t + 3 : ℝ) = 22 :=
by
  sorry

end distance_traveled_l94_94429


namespace total_number_of_boys_l94_94203

-- Define the circular arrangement and the opposite positions
variable (n : ℕ)

theorem total_number_of_boys (h : (40 ≠ 10 ∧ (40 - 10) * 2 = n - 2)) : n = 62 := 
sorry

end total_number_of_boys_l94_94203


namespace total_number_of_notes_l94_94909

theorem total_number_of_notes (x : ℕ) (h₁ : 37 * 50 + x * 500 = 10350) : 37 + x = 54 :=
by
  -- We state that the total value of 37 Rs. 50 notes plus x Rs. 500 notes equals Rs. 10350.
  -- According to this information, we prove that the total number of notes is 54.
  sorry

end total_number_of_notes_l94_94909


namespace sum_of_positive_factors_of_90_eq_234_l94_94662

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end sum_of_positive_factors_of_90_eq_234_l94_94662


namespace min_value_inequality_l94_94060

open Real

theorem min_value_inequality (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 47 :=
sorry

end min_value_inequality_l94_94060


namespace num_multiples_6_not_12_lt_300_l94_94008

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l94_94008


namespace number_of_elements_in_M_l94_94901

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end number_of_elements_in_M_l94_94901


namespace students_calculation_l94_94894

def number_of_stars : ℝ := 3.0
def students_per_star : ℝ := 41.33333333
def total_students : ℝ := 124

theorem students_calculation : number_of_stars * students_per_star = total_students := 
by
  sorry

end students_calculation_l94_94894


namespace inequality_transitive_l94_94602

theorem inequality_transitive (a b c : ℝ) : a * c^2 > b * c^2 → a > b :=
sorry

end inequality_transitive_l94_94602


namespace compute_value_l94_94127

theorem compute_value : 302^2 - 298^2 = 2400 :=
by
  sorry

end compute_value_l94_94127


namespace fruit_salad_cherries_l94_94789

variable (b r g c : ℕ)

theorem fruit_salad_cherries :
  (b + r + g + c = 350) ∧
  (r = 3 * b) ∧
  (g = 4 * c) ∧
  (c = 5 * r) →
  c = 66 :=
by
  sorry

end fruit_salad_cherries_l94_94789


namespace circle_radius_l94_94562

theorem circle_radius (r M N : ℝ) (hM : M = π * r^2) (hN : N = 2 * π * r) (hRatio : M / N = 20) : r = 40 := 
by
  sorry

end circle_radius_l94_94562


namespace bus_driver_compensation_l94_94096

theorem bus_driver_compensation : 
  let regular_rate := 16
  let regular_hours := 40
  let total_hours_worked := 57
  let overtime_rate := regular_rate + (0.75 * regular_rate)
  let regular_pay := regular_hours * regular_rate
  let overtime_hours_worked := total_hours_worked - regular_hours
  let overtime_pay := overtime_hours_worked * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1116 :=
by
  sorry

end bus_driver_compensation_l94_94096


namespace room_area_ratio_l94_94852

theorem room_area_ratio (total_squares overlapping_squares : ℕ) 
  (h_total : total_squares = 16) 
  (h_overlap : overlapping_squares = 4) : 
  total_squares / overlapping_squares = 4 := 
by 
  sorry

end room_area_ratio_l94_94852


namespace number_of_clothes_hangers_l94_94749

noncomputable def total_money : ℝ := 60
noncomputable def spent_on_tissues : ℝ := 34.8
noncomputable def price_per_hanger : ℝ := 1.6

theorem number_of_clothes_hangers : 
  let remaining_money := total_money - spent_on_tissues
  let hangers := remaining_money / price_per_hanger
  Int.floor hangers = 15 := 
by
  sorry

end number_of_clothes_hangers_l94_94749


namespace find_ab_pairs_l94_94095

open Set

-- Definitions
def f (a b x : ℝ) : ℝ := a * x + b

-- Main theorem
theorem find_ab_pairs (a b : ℝ) :
  (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → 
    f a b x * f a b y + f a b (x + y - x * y) ≤ 0) ↔ 
  (-1 ≤ b ∧ b ≤ 0 ∧ -(b + 1) ≤ a ∧ a ≤ -b) :=
by sorry

end find_ab_pairs_l94_94095


namespace skew_lines_angle_range_l94_94370

theorem skew_lines_angle_range (θ : ℝ) (h_skew : θ > 0 ∧ θ ≤ 90) :
  0 < θ ∧ θ ≤ 90 :=
sorry

end skew_lines_angle_range_l94_94370


namespace items_per_charge_is_five_l94_94481

-- Define the number of dog treats, chew toys, rawhide bones, and credit cards as constants.
def num_dog_treats := 8
def num_chew_toys := 2
def num_rawhide_bones := 10
def num_credit_cards := 4

-- Define the total number of items.
def total_items := num_dog_treats + num_chew_toys + num_rawhide_bones

-- Prove that the number of items per credit card charge is 5.
theorem items_per_charge_is_five :
  (total_items / num_credit_cards) = 5 :=
by
  -- Proof goes here (we use sorry to skip the actual proof)
  sorry

end items_per_charge_is_five_l94_94481


namespace veggies_minus_fruits_l94_94698

-- Definitions of quantities as given in the conditions
def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

-- Problem Statement
theorem veggies_minus_fruits : (cucumbers + tomatoes) - (apples + bananas) = 8 :=
by 
  -- insert proof here
  sorry

end veggies_minus_fruits_l94_94698


namespace smallest_base_l94_94521

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end smallest_base_l94_94521


namespace find_angle_C_find_area_l94_94107

open Real

-- Definition of the problem conditions and questions

-- Condition: Given a triangle and the trigonometric relationship
variables {A B C : ℝ} {a b c : ℝ}

-- Condition 1: Trigonometric identity provided in the problem
axiom trig_identity : (sqrt 3) * c / (cos C) = a / (cos (3 * π / 2 + A))

-- First part of the problem
theorem find_angle_C (h1 : sqrt 3 * c / cos C = a / cos (3 * π / 2 + A)) : C = π / 6 :=
sorry

-- Second part of the problem
noncomputable def area_of_triangle (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

variables {c' b' : ℝ}
-- Given conditions for the second question 
axiom condition_c_a : c' / a = 2
axiom condition_b : b' = 4 * sqrt 3

-- Definitions to align with the given problem
def c_from_a (a : ℝ) : ℝ := 2 * a

-- The final theorem for the second part
theorem find_area (hC : C = π / 6) (hc : c_from_a a = c') (hb : b' = 4 * sqrt 3) :
  area_of_triangle a b' C = 2 * sqrt 15 - 2 * sqrt 3 :=
sorry

end find_angle_C_find_area_l94_94107


namespace percentage_increase_l94_94916

variables (J T P : ℝ)

def income_conditions (J T P : ℝ) : Prop :=
  (T = 0.5 * J) ∧ (P = 0.8 * J)

theorem percentage_increase (J T P : ℝ) (h : income_conditions J T P) :
  ((P / T) - 1) * 100 = 60 :=
by
  sorry

end percentage_increase_l94_94916


namespace cost_of_double_burger_l94_94656

-- Definitions based on conditions
def total_cost : ℝ := 64.50
def total_burgers : ℕ := 50
def single_burger_cost : ℝ := 1.00
def double_burgers : ℕ := 29

-- Proof goal
theorem cost_of_double_burger : (total_cost - single_burger_cost * (total_burgers - double_burgers)) / double_burgers = 1.50 :=
by
  sorry

end cost_of_double_burger_l94_94656


namespace initial_orchid_bushes_l94_94017

def final_orchid_bushes : ℕ := 35
def orchid_bushes_to_be_planted : ℕ := 13

theorem initial_orchid_bushes :
  final_orchid_bushes - orchid_bushes_to_be_planted = 22 :=
by
  sorry

end initial_orchid_bushes_l94_94017


namespace no_prime_quadruple_l94_94161

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_quadruple 
    (a b c d : ℕ)
    (ha_prime : is_prime a) 
    (hb_prime : is_prime b)
    (hc_prime : is_prime c)
    (hd_prime : is_prime d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    (1 / a + 1 / d ≠ 1 / b + 1 / c) := 
by 
  sorry

end no_prime_quadruple_l94_94161


namespace initial_weights_of_apples_l94_94934

variables {A B : ℕ}

theorem initial_weights_of_apples (h₁ : A + B = 75) (h₂ : A - 5 = (B + 5) + 7) :
  A = 46 ∧ B = 29 :=
by
  sorry

end initial_weights_of_apples_l94_94934


namespace monotonicity_of_g_l94_94317

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb a (|x + 1|)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.logb a (- (3 / 2) * x^2 + a * x)

theorem monotonicity_of_g (a : ℝ) (h : 0 < a ∧ a ≠ 1) (h0 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x a < 0) :
  ∀ x : ℝ, 0 < x ∧ x ≤ a / 3 → (g x a) < (g (x + ε) a) := 
sorry


end monotonicity_of_g_l94_94317


namespace quadratic_factorization_l94_94660

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 20 * x + 96 = (x - a) * (x - b)) (h2 : a > b) : 2 * b - a = 4 :=
sorry

end quadratic_factorization_l94_94660


namespace total_coins_received_l94_94288

theorem total_coins_received (coins_first_day coins_second_day : ℕ) 
  (h_first_day : coins_first_day = 22) 
  (h_second_day : coins_second_day = 12) : 
  coins_first_day + coins_second_day = 34 := 
by 
  sorry

end total_coins_received_l94_94288


namespace quadratic_real_roots_range_l94_94115

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (m-1)*x^2 + x + 1 = 0) → (m ≤ 5/4 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_range_l94_94115


namespace number_of_integers_x_l94_94853

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

def valid_range_x (x : ℝ) : Prop :=
  13 < x ∧ x < 43

def conditions_for_acute_triangle (x : ℝ) : Prop :=
  (x > 28 ∧ x^2 < 1009) ∨ (x ≤ 28 ∧ x > 23.64)

theorem number_of_integers_x (count : ℤ) :
  (∃ (x : ℤ), valid_range_x x ∧ is_triangle 15 28 x ∧ is_acute_triangle 15 28 x ∧ conditions_for_acute_triangle x) →
  count = 8 :=
sorry

end number_of_integers_x_l94_94853


namespace julia_age_correct_l94_94552

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end julia_age_correct_l94_94552


namespace sum_ages_of_brothers_l94_94009

theorem sum_ages_of_brothers (x : ℝ) (ages : List ℝ) 
  (h1 : ages = [x, x + 1.5, x + 3, x + 4.5, x + 6, x + 7.5, x + 9])
  (h2 : x + 9 = 4 * x) : 
    List.sum ages = 52.5 := 
  sorry

end sum_ages_of_brothers_l94_94009


namespace find_y_l94_94147

theorem find_y (x k m y : ℤ) 
  (h1 : x = 82 * k + 5) 
  (h2 : x + y = 41 * m + 12) : 
  y = 7 := 
sorry

end find_y_l94_94147


namespace sum_of_decimals_l94_94421

theorem sum_of_decimals :
  0.3 + 0.04 + 0.005 + 0.0006 + 0.00007 = (34567 / 100000 : ℚ) :=
by
  -- The proof details would go here
  sorry

end sum_of_decimals_l94_94421


namespace lost_card_number_l94_94093

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l94_94093


namespace camp_problem_l94_94000

variable (x : ℕ) -- number of girls
variable (y : ℕ) -- number of boys
variable (total_children : ℕ) -- total number of children
variable (girls_cannot_swim : ℕ) -- number of girls who cannot swim
variable (boys_cannot_swim : ℕ) -- number of boys who cannot swim
variable (children_can_swim : ℕ) -- total number of children who can swim
variable (children_cannot_swim : ℕ) -- total number of children who cannot swim
variable (o_six_girls : ℕ) -- one-sixth of the total number of girls
variable (o_eight_boys : ℕ) -- one-eighth of the total number of boys

theorem camp_problem 
    (hc1 : total_children = 50)
    (hc2 : girls_cannot_swim = x / 6)
    (hc3 : boys_cannot_swim = y / 8)
    (hc4 : children_can_swim = 43)
    (hc5 : children_cannot_swim = total_children - children_can_swim)
    (h_total : x + y = total_children)
    (h_swim : children_cannot_swim = girls_cannot_swim + boys_cannot_swim) :
    x = 18 :=
  by
    have hc6 : children_cannot_swim = 7 := by sorry -- from hc4 and hc5
    have h_eq : x / 6 + (50 - x) / 8 = 7 := by sorry -- from hc2, hc3, hc6
    -- solving for x
    sorry

end camp_problem_l94_94000


namespace new_person_age_l94_94578

theorem new_person_age (T : ℕ) (A : ℕ) (n : ℕ) 
  (avg_age : ℕ) (new_avg_age : ℕ) 
  (h1 : avg_age = T / n) 
  (h2 : T = 14 * n)
  (h3 : n = 17) 
  (h4 : new_avg_age = 15) 
  (h5 : new_avg_age = (T + A) / (n + 1)) 
  : A = 32 := 
by 
  sorry

end new_person_age_l94_94578


namespace min_throws_for_repeated_sum_l94_94861

theorem min_throws_for_repeated_sum (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 16) : 
  ∃ m, m = 16 ∧ (∀ (k : ℕ), k < 16 → ∃ i < 16, ∃ j < 16, i ≠ j ∧ i + j = k) :=
by
  sorry

end min_throws_for_repeated_sum_l94_94861


namespace greatest_possible_median_l94_94992

theorem greatest_possible_median (k m r s t : ℕ) (h_avg : (k + m + r + s + t) / 5 = 10) (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) (h_t : t = 20) : r = 8 :=
by
  sorry

end greatest_possible_median_l94_94992


namespace opposite_of_a_is_2_l94_94425

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end opposite_of_a_is_2_l94_94425


namespace ratio_suspension_to_fingers_toes_l94_94262

-- Definition of conditions
def suspension_days_per_instance : Nat := 3
def bullying_instances : Nat := 20
def fingers_and_toes : Nat := 20

-- Theorem statement
theorem ratio_suspension_to_fingers_toes :
  (suspension_days_per_instance * bullying_instances) / fingers_and_toes = 3 :=
by
  sorry

end ratio_suspension_to_fingers_toes_l94_94262


namespace couple_tickets_sold_l94_94390

theorem couple_tickets_sold (S C : ℕ) :
  20 * S + 35 * C = 2280 ∧ S + 2 * C = 128 -> C = 56 :=
by
  intro h
  sorry

end couple_tickets_sold_l94_94390


namespace exponent_properties_l94_94349

variables (a : ℝ) (m n : ℕ)
-- Conditions
axiom h1 : a^m = 3
axiom h2 : a^n = 2

-- Goal
theorem exponent_properties :
  a^(m + n) = 6 :=
by
  sorry

end exponent_properties_l94_94349


namespace proof_a6_bounds_l94_94355

theorem proof_a6_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 :=
by
  sorry

end proof_a6_bounds_l94_94355


namespace correct_propositions_are_123_l94_94613

theorem correct_propositions_are_123
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x-1) = -f x → f x = f (x-2))
  (h2 : ∀ x, f (1 - x) = f (x - 1) → f (1 - x) = -f x)
  (h3 : ∀ x, f (x) = -f (-x)) :
  (∀ x, f (x-1) = -f x → ∃ c, c * (f (1-1)) = -f x) ∧
  (∀ x, f (1 - x) = f (x - 1) → ∀ x, f x = f (-x)) ∧
  (∀ x, f (x-1) = -f x → ∀ x, f (x - 2) = f x) :=
sorry

end correct_propositions_are_123_l94_94613


namespace range_of_m_l94_94967

theorem range_of_m (m : ℝ) (h₁ : ∀ x : ℝ, -x^2 + 7*x + 8 ≥ 0 → x^2 - 7*x - 8 ≤ 0)
  (h₂ : ∀ x : ℝ, x^2 - 2*x + 1 - 4*m^2 ≤ 0 → 1 - 2*m ≤ x ∧ x ≤ 1 + 2*m)
  (not_p_sufficient_for_not_q : ∀ x : ℝ, ¬(-x^2 + 7*x + 8 ≥ 0) → ¬(x^2 - 2*x + 1 - 4*m^2 ≤ 0))
  (suff_non_necess : ∀ x : ℝ, (x^2 - 2*x + 1 - 4*m^2 ≤ 0) → ¬(x^2 - 7*x - 8 ≤ 0))
  : 0 < m ∧ m ≤ 1 := sorry

end range_of_m_l94_94967


namespace maximize_takehome_pay_l94_94616

noncomputable def tax_initial (income : ℝ) : ℝ :=
  if income ≤ 20000 then 0.10 * income else 2000 + 0.05 * ((income - 20000) / 10000) * income

noncomputable def tax_beyond (income : ℝ) : ℝ :=
  (income - 20000) * ((0.005 * ((income - 20000) / 10000)) * income)

noncomputable def tax_total (income : ℝ) : ℝ :=
  if income ≤ 20000 then tax_initial income else tax_initial 20000 + tax_beyond income

noncomputable def takehome_pay_function (income : ℝ) : ℝ :=
  income - tax_total income

theorem maximize_takehome_pay : ∃ x, takehome_pay_function x = takehome_pay_function 30000 := 
sorry

end maximize_takehome_pay_l94_94616


namespace simplify_sqrt_expression_l94_94913

theorem simplify_sqrt_expression :
  2 * Real.sqrt 12 - Real.sqrt 27 - (Real.sqrt 3 * Real.sqrt (1 / 9)) = (2 * Real.sqrt 3) / 3 := 
by
  sorry

end simplify_sqrt_expression_l94_94913


namespace room_length_l94_94479

-- Defining conditions
def room_height : ℝ := 5
def room_width : ℝ := 7
def door_height : ℝ := 3
def door_width : ℝ := 1
def num_doors : ℝ := 2
def window1_height : ℝ := 1.5
def window1_width : ℝ := 2
def window2_height : ℝ := 1.5
def window2_width : ℝ := 1
def num_window2 : ℝ := 2
def paint_cost_per_sq_m : ℝ := 3
def total_paint_cost : ℝ := 474

-- Defining the problem as a statement to prove x (room length) is 10 meters
theorem room_length {x : ℝ} 
  (H1 : total_paint_cost = paint_cost_per_sq_m * ((2 * (x * room_height) + 2 * (room_width * room_height)) - (num_doors * (door_height * door_width) + (window1_height * window1_width) + num_window2 * (window2_height * window2_width)))) 
  : x = 10 :=
by 
  sorry

end room_length_l94_94479


namespace Molly_age_now_l94_94752

/- Definitions -/
def Sandy_curr_age : ℕ := 60
def Molly_curr_age (S : ℕ) : ℕ := 3 * S / 4
def Sandy_age_in_6_years (S : ℕ) : ℕ := S + 6

/- Theorem to prove -/
theorem Molly_age_now 
  (ratio_condition : ∀ S M : ℕ, S / M = 4 / 3 → M = 3 * S / 4)
  (age_condition : Sandy_age_in_6_years Sandy_curr_age = 66) : 
  Molly_curr_age Sandy_curr_age = 45 :=
by
  sorry

end Molly_age_now_l94_94752


namespace landlord_packages_l94_94792

def label_packages_required (start1 end1 start2 end2 start3 end3 : ℕ) : ℕ :=
  let digit_count := 1
  let hundreds_first := (end1 - start1 + 1)
  let hundreds_second := (end2 - start2 + 1)
  let hundreds_third := (end3 - start3 + 1)
  let total_hundreds := hundreds_first + hundreds_second + hundreds_third
  
  let tens_first := ((end1 - start1 + 1) / 10) 
  let tens_second := ((end2 - start2 + 1) / 10) 
  let tens_third := ((end3 - start3 + 1) / 10)
  let total_tens := tens_first + tens_second + tens_third

  let units_per_floor := 5
  let total_units := units_per_floor * 3
  
  let total_ones := total_hundreds + total_tens + total_units
  
  let packages_required := total_ones

  packages_required

theorem landlord_packages : label_packages_required 100 150 200 250 300 350 = 198 := 
  by sorry

end landlord_packages_l94_94792


namespace distance_from_diagonal_intersection_to_base_l94_94424

theorem distance_from_diagonal_intersection_to_base (AD BC AB R : ℝ) (O : ℝ → Prop) (M N Q : ℝ) :
  (AD + BC + 2 * AB = 8) ∧
  (AD + BC) = 4 ∧
  (R = 1 / 2) ∧
  (2 = R * (AD + BC) / 2) ∧
  (BC = AD + 2 * AB) ∧
  (∀ x, x * (2 - x) = (1 / 2) ^ 2)  →
  (Q = (2 - Real.sqrt 3) / 4) :=
by
  intros
  sorry

end distance_from_diagonal_intersection_to_base_l94_94424


namespace negation_of_p_l94_94086

def p : Prop := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x ≤ Real.sin x :=
by sorry

end negation_of_p_l94_94086


namespace white_surface_area_fraction_l94_94070

theorem white_surface_area_fraction
    (total_cubes : ℕ)
    (white_cubes : ℕ)
    (red_cubes : ℕ)
    (edge_length : ℕ)
    (white_exposed_area : ℕ)
    (total_surface_area : ℕ)
    (fraction : ℚ)
    (h1 : total_cubes = 64)
    (h2 : white_cubes = 14)
    (h3 : red_cubes = 50)
    (h4 : edge_length = 4)
    (h5 : white_exposed_area = 6)
    (h6 : total_surface_area = 96)
    (h7 : fraction = 1 / 16)
    (h8 : white_cubes + red_cubes = total_cubes)
    (h9 : 6 * (edge_length * edge_length) = total_surface_area)
    (h10 : white_exposed_area / total_surface_area = fraction) :
    fraction = 1 / 16 := by
    sorry

end white_surface_area_fraction_l94_94070


namespace necessary_and_sufficient_condition_for_f_to_be_odd_l94_94717

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f (a b x : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem necessary_and_sufficient_condition_for_f_to_be_odd (a b : ℝ) :
  is_odd_function (f a b) ↔ sorry :=
by
  -- This is where the proof would go.
  sorry

end necessary_and_sufficient_condition_for_f_to_be_odd_l94_94717


namespace apples_after_operations_l94_94062

-- Define the initial conditions
def initial_apples : ℕ := 38
def used_apples : ℕ := 20
def bought_apples : ℕ := 28

-- State the theorem we want to prove
theorem apples_after_operations : initial_apples - used_apples + bought_apples = 46 :=
by
  sorry

end apples_after_operations_l94_94062


namespace least_common_multiple_prime_numbers_l94_94915

theorem least_common_multiple_prime_numbers (x y : ℕ) (hx_prime : Prime x) (hy_prime : Prime y)
  (hxy : y < x) (h_eq : 2 * x + y = 12) : Nat.lcm x y = 10 :=
by
  sorry

end least_common_multiple_prime_numbers_l94_94915


namespace classroom_has_total_books_l94_94408

-- Definitions for the conditions
def num_children : Nat := 10
def books_per_child : Nat := 7
def additional_books : Nat := 8

-- Total number of books the children have
def total_books_from_children : Nat := num_children * books_per_child

-- The expected total number of books in the classroom
def total_books : Nat := total_books_from_children + additional_books

-- The main theorem to be proven
theorem classroom_has_total_books : total_books = 78 :=
by
  sorry

end classroom_has_total_books_l94_94408


namespace find_value_of_expression_l94_94796

-- Conditions as provided
axiom given_condition : ∃ (x : ℕ), 3^x + 3^x + 3^x + 3^x = 2187

-- Proof statement
theorem find_value_of_expression : (exists (x : ℕ), (3^x + 3^x + 3^x + 3^x = 2187) ∧ ((x + 2) * (x - 2) = 21)) :=
sorry

end find_value_of_expression_l94_94796


namespace quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l94_94667

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Problem 1: Prove that the quadratic function passes through the origin for m = 1 or m = -2
theorem quadratic_passes_through_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -2) ∧ quadratic m 0 = 0 := by
  sorry

-- Problem 2: Prove that the quadratic function is symmetric about the y-axis for m = 0
theorem quadratic_symmetric_about_y_axis :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, quadratic m x = quadratic m (-x) := by
  sorry

end quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l94_94667


namespace number_of_ordered_triples_l94_94924

theorem number_of_ordered_triples (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 3969) (h4 : a * c = 3969^2) :
    ∃ n : ℕ, n = 12 := sorry

end number_of_ordered_triples_l94_94924


namespace total_bars_is_7_l94_94865

variable (x : ℕ)

-- Each chocolate bar costs $3
def cost_per_bar := 3

-- Olivia sold all but 4 bars
def bars_sold (total_bars : ℕ) := total_bars - 4

-- Olivia made $9
def amount_made (total_bars : ℕ) := cost_per_bar * bars_sold total_bars

-- Given conditions
def condition1 (total_bars : ℕ) := amount_made total_bars = 9

-- Proof that the total number of bars is 7
theorem total_bars_is_7 : condition1 x -> x = 7 := by
  sorry

end total_bars_is_7_l94_94865


namespace complex_product_l94_94414

theorem complex_product : (3 + 4 * I) * (-2 - 3 * I) = -18 - 17 * I :=
by
  sorry

end complex_product_l94_94414


namespace total_revenue_correct_l94_94044

def price_per_book : ℝ := 25
def revenue_monday : ℝ := 60 * ((price_per_book * 0.9) * 1.05)
def revenue_tuesday : ℝ := 10 * (price_per_book * 1.03)
def revenue_wednesday : ℝ := 20 * ((price_per_book * 0.95) * 1.02)
def revenue_thursday : ℝ := 44 * ((price_per_book * 0.85) * 1.04)
def revenue_friday : ℝ := 66 * (price_per_book * 0.8)

def total_revenue : ℝ :=
  revenue_monday + revenue_tuesday + revenue_wednesday +
  revenue_thursday + revenue_friday

theorem total_revenue_correct :
  total_revenue = 4452.4 :=
by
  rw [total_revenue, revenue_monday, revenue_tuesday, revenue_wednesday, 
      revenue_thursday, revenue_friday]
  -- Verification steps would continue by calculating each term.
  sorry

end total_revenue_correct_l94_94044


namespace train_speeds_l94_94469

theorem train_speeds (v t : ℕ) (h1 : t = 1)
  (h2 : v + v * t = 90)
  (h3 : 90 * t = 90) :
  v = 45 := by
  sorry

end train_speeds_l94_94469


namespace boss_contribution_l94_94886

variable (boss_contrib : ℕ) (todd_contrib : ℕ) (employees_contrib : ℕ)
variable (cost : ℕ) (n_employees : ℕ) (emp_payment : ℕ)
variable (total_payment : ℕ)

-- Conditions
def birthday_gift_conditions :=
  cost = 100 ∧
  todd_contrib = 2 * boss_contrib ∧
  employees_contrib = n_employees * emp_payment ∧
  n_employees = 5 ∧
  emp_payment = 11 ∧
  total_payment = boss_contrib + todd_contrib + employees_contrib

-- The proof goal
theorem boss_contribution
  (h : birthday_gift_conditions boss_contrib todd_contrib employees_contrib cost n_employees emp_payment total_payment) :
  boss_contrib = 15 :=
by
  sorry

end boss_contribution_l94_94886


namespace quartic_to_quadratic_l94_94818

-- Defining the statement of the problem
theorem quartic_to_quadratic (a b c x : ℝ) (y : ℝ) :
  a * x^4 + b * x^3 + c * x^2 + b * x + a = 0 →
  y = x + 1 / x →
  ∃ y1 y2, (a * y^2 + b * y + (c - 2 * a) = 0) ∧
           (x^2 - y1 * x + 1 = 0 ∨ x^2 - y2 * x + 1 = 0) :=
by
  sorry

end quartic_to_quadratic_l94_94818


namespace complex_quadrant_l94_94345

open Complex

noncomputable def z : ℂ := (2 * I) / (1 - I)

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 2 * I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l94_94345


namespace sin_240_eq_neg_sqrt3_div_2_l94_94938

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l94_94938


namespace winning_percentage_votes_l94_94172

theorem winning_percentage_votes (P : ℝ) (votes_total : ℝ) (majority_votes : ℝ) (winning_votes : ℝ) : 
  votes_total = 4500 → majority_votes = 900 → 
  winning_votes = (P / 100) * votes_total → 
  majority_votes = winning_votes - ((100 - P) / 100) * votes_total → P = 60 := 
by
  intros h_total h_majority h_winning_votes h_majority_eq
  sorry

end winning_percentage_votes_l94_94172


namespace find_range_of_k_l94_94615

noncomputable def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k * x

theorem find_range_of_k :
  (∀ x : ℝ, 0 < x → 0 ≤ f x k) → (-1 ≤ k) :=
by
  sorry

end find_range_of_k_l94_94615


namespace sequence_conjecture_l94_94490

theorem sequence_conjecture (a : ℕ → ℝ) (h₁ : a 1 = 7)
  (h₂ : ∀ n, a (n + 1) = 7 * a n / (a n + 7)) :
  ∀ n, a n = 7 / n :=
by
  sorry

end sequence_conjecture_l94_94490


namespace physical_fitness_test_l94_94108

theorem physical_fitness_test (x : ℝ) (hx : x > 0) :
  (1000 / x - 1000 / (1.25 * x) = 30) :=
sorry

end physical_fitness_test_l94_94108


namespace handshakes_at_convention_l94_94376

theorem handshakes_at_convention (num_gremlins : ℕ) (num_imps : ℕ) 
  (H_gremlins_shake : num_gremlins = 25) (H_imps_shake_gremlins : num_imps = 20) : 
  let handshakes_among_gremlins := num_gremlins * (num_gremlins - 1) / 2
  let handshakes_between_imps_and_gremlins := num_imps * num_gremlins
  let total_handshakes := handshakes_among_gremlins + handshakes_between_imps_and_gremlins
  total_handshakes = 800 := 
by 
  sorry

end handshakes_at_convention_l94_94376


namespace lily_ducks_l94_94103

variable (D G : ℕ)
variable (Rayden_ducks : ℕ := 3 * D)
variable (Rayden_geese : ℕ := 4 * G)
variable (Lily_geese : ℕ := 10) -- Given G = 10
variable (Rayden_extra : ℕ := 70) -- Given Rayden has 70 more ducks and geese

theorem lily_ducks (h : 3 * D + 4 * Lily_geese = D + Lily_geese + Rayden_extra) : D = 20 :=
by sorry

end lily_ducks_l94_94103


namespace april_earnings_l94_94932

def price_per_rose := 7
def price_per_lily := 5
def initial_roses := 9
def initial_lilies := 6
def remaining_roses := 4
def remaining_lilies := 2

def total_roses_sold := initial_roses - remaining_roses
def total_lilies_sold := initial_lilies - remaining_lilies

def total_earnings := (total_roses_sold * price_per_rose) + (total_lilies_sold * price_per_lily)

theorem april_earnings : total_earnings = 55 := by
  sorry

end april_earnings_l94_94932


namespace find_f_2012_l94_94784

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1 / 4
axiom f_condition2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem find_f_2012 : f 2012 = -1 / 4 := 
sorry

end find_f_2012_l94_94784


namespace joshua_finishes_after_malcolm_l94_94465

def time_difference_between_runners
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (malcolm_finish_time : ℕ := malcolm_speed * race_length)
  (joshua_finish_time : ℕ := joshua_speed * race_length) : ℕ :=
joshua_finish_time - malcolm_finish_time

theorem joshua_finishes_after_malcolm
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (h_race_length : race_length = 12)
  (h_malcolm_speed : malcolm_speed = 7)
  (h_joshua_speed : joshua_speed = 9) : time_difference_between_runners race_length malcolm_speed joshua_speed = 24 :=
by 
  subst h_race_length
  subst h_malcolm_speed
  subst h_joshua_speed
  rfl

#print joshua_finishes_after_malcolm

end joshua_finishes_after_malcolm_l94_94465


namespace ratio_of_fusilli_to_penne_l94_94121

def number_of_students := 800
def preferred_pasta_types := ["penne", "tortellini", "fusilli", "spaghetti"]
def students_prefer_fusilli := 320
def students_prefer_penne := 160

theorem ratio_of_fusilli_to_penne : (students_prefer_fusilli / students_prefer_penne) = 2 := by
  -- Here we would provide the proof, but since it's a statement, we use sorry
  sorry

end ratio_of_fusilli_to_penne_l94_94121


namespace max_value_quadratic_function_l94_94499

open Real

theorem max_value_quadratic_function (r : ℝ) (x₀ y₀ : ℝ) (P_tangent : (2 / x₀) * x - y₀ = 0) 
  (circle_tangent : (x₀ - 3) * (x - 3) + y₀ * y = r^2) :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 1 / 2 * x * (3 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 9 / 8) :=
by
  sorry

end max_value_quadratic_function_l94_94499


namespace Enid_made_8_sweaters_l94_94010

def scarves : ℕ := 10
def sweaters_Aaron : ℕ := 5
def wool_per_scarf : ℕ := 3
def wool_per_sweater : ℕ := 4
def total_wool_used : ℕ := 82
def Enid_sweaters : ℕ := 8

theorem Enid_made_8_sweaters
  (scarves : ℕ)
  (sweaters_Aaron : ℕ)
  (wool_per_scarf : ℕ)
  (wool_per_sweater : ℕ)
  (total_wool_used : ℕ)
  (Enid_sweaters : ℕ)
  : Enid_sweaters = 8 :=
by
  sorry

end Enid_made_8_sweaters_l94_94010


namespace cake_eating_contest_l94_94366

-- Define the fractions representing the amounts of cake eaten by the two students.
def first_student : ℚ := 7 / 8
def second_student : ℚ := 5 / 6

-- The statement of our proof problem
theorem cake_eating_contest : first_student - second_student = 1 / 24 := by
  sorry

end cake_eating_contest_l94_94366


namespace soup_options_l94_94825

-- Define the given conditions
variables (lettuce_types tomato_types olive_types total_options : ℕ)
variable (S : ℕ)

-- State the conditions
theorem soup_options :
  lettuce_types = 2 →
  tomato_types = 3 →
  olive_types = 4 →
  total_options = 48 →
  (lettuce_types * tomato_types * olive_types * S = total_options) →
  S = 2 :=
by
  sorry

end soup_options_l94_94825


namespace exists_k_lt_ak_by_2001_fac_l94_94518

theorem exists_k_lt_ak_by_2001_fac (a : ℕ → ℝ) (H0 : a 0 = 1)
(Hn : ∀ n : ℕ, n > 0 → a n = a (⌊(7 * n / 9)⌋₊) + a (⌊(n / 9)⌋₊)) :
  ∃ k : ℕ, k > 0 ∧ a k < k / ↑(Nat.factorial 2001) := by
  sorry

end exists_k_lt_ak_by_2001_fac_l94_94518


namespace max_sum_is_2017_l94_94872

theorem max_sum_is_2017 (a b c : ℕ) 
  (h1 : a + b = 1014) 
  (h2 : c - b = 497) 
  (h3 : a > b) : 
  (a + b + c) ≤ 2017 := sorry

end max_sum_is_2017_l94_94872


namespace continuous_stripe_encircling_tetrahedron_probability_l94_94721

noncomputable def tetrahedron_continuous_stripe_probability : ℚ :=
  let total_combinations := 3^4
  let favorable_combinations := 2 
  favorable_combinations / total_combinations

theorem continuous_stripe_encircling_tetrahedron_probability :
  tetrahedron_continuous_stripe_probability = 2 / 81 :=
by
  -- the proof would be here
  sorry

end continuous_stripe_encircling_tetrahedron_probability_l94_94721


namespace find_s_for_g3_eq_0_l94_94778

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem find_s_for_g3_eq_0 : (g 3 s = 0) ↔ (s = -573) :=
by
  sorry

end find_s_for_g3_eq_0_l94_94778


namespace value_of_p_l94_94798

theorem value_of_p (m n p : ℝ) (h₁ : m = 8 * n + 5) (h₂ : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by {
  sorry
}

end value_of_p_l94_94798


namespace increasing_interval_of_f_l94_94685

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 15 * x ^ 2 + 36 * x - 24

theorem increasing_interval_of_f : (∀ x : ℝ, x = 2 → deriv f x = 0) → ∀ x : ℝ, 3 < x → 0 < deriv f x :=
by
  intro h x hx
  -- We know that the function has an extreme value at x = 2
  have : deriv f 2 = 0 := h 2 rfl
  -- Require to prove the function is increasing in interval (3, +∞)
  sorry

end increasing_interval_of_f_l94_94685


namespace natasha_can_achieve_plan_l94_94842

noncomputable def count_ways : Nat :=
  let num_1x1 := 4
  let num_1x2 := 24
  let target := 2021
  6517

theorem natasha_can_achieve_plan (num_1x1 num_1x2 target : Nat) (h1 : num_1x1 = 4) (h2 : num_1x2 = 24) (h3 : target = 2021) :
  count_ways = 6517 :=
by
  sorry

end natasha_can_achieve_plan_l94_94842


namespace game_spinner_probability_l94_94076

theorem game_spinner_probability (P_A P_B P_D P_C : ℚ) (h₁ : P_A = 1/4) (h₂ : P_B = 1/3) (h₃ : P_D = 1/6) (h₄ : P_A + P_B + P_C + P_D = 1) :
  P_C = 1/4 :=
by
  sorry

end game_spinner_probability_l94_94076


namespace base_seven_representation_l94_94708

theorem base_seven_representation 
  (k : ℕ) 
  (h1 : 4 ≤ k) 
  (h2 : k < 8) 
  (h3 : 500 / k^3 < k) 
  (h4 : 500 ≥ k^3) 
  : ∃ n m o p : ℕ, (500 = n * k^3 + m * k^2 + o * k + p) ∧ (p % 2 = 1) ∧ (n ≠ 0 ) :=
sorry

end base_seven_representation_l94_94708


namespace number_divisors_product_l94_94904

theorem number_divisors_product :
  ∃ N : ℕ, (∃ a b : ℕ, N = 3^a * 5^b ∧ (N^((a+1)*(b+1) / 2)) = 3^30 * 5^40) ∧ N = 3^3 * 5^4 :=
sorry

end number_divisors_product_l94_94904


namespace minimum_positive_temperature_announcement_l94_94333

-- Problem conditions translated into Lean
def num_interactions (x : ℕ) : ℕ := x * (x - 1)
def total_interactions := 132
def total_positive := 78
def total_negative := 54
def positive_temperature_count (x y : ℕ) : ℕ := y * (y - 1)
def negative_temperature_count (x y : ℕ) : ℕ := (x - y) * (x - 1 - y)
def minimum_positive_temperature (x y : ℕ) := 
  x = 12 → 
  total_interactions = total_positive + total_negative →
  total_positive + total_negative = num_interactions x →
  total_positive = positive_temperature_count x y →
  sorry -- proof goes here

theorem minimum_positive_temperature_announcement : ∃ y, 
  minimum_positive_temperature 12 y ∧ y = 3 :=
by {
  sorry -- proof goes here
}

end minimum_positive_temperature_announcement_l94_94333


namespace defective_rate_worker_y_l94_94809

theorem defective_rate_worker_y (d_x d_y : ℝ) (f_y : ℝ) (total_defective_rate : ℝ) :
  d_x = 0.005 → f_y = 0.8 → total_defective_rate = 0.0074 → 
  (0.2 * d_x + f_y * d_y = total_defective_rate) → d_y = 0.008 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end defective_rate_worker_y_l94_94809


namespace minimum_k_condition_l94_94867

def is_acute_triangle (a b c : ℕ) : Prop :=
  a * a + b * b > c * c

def any_subset_with_three_numbers_construct_acute_triangle (s : Finset ℕ) : Prop :=
  ∀ t : Finset ℕ, t.card = 3 → 
    (∃ a b c : ℕ, a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ 
      is_acute_triangle a b c ∨
      is_acute_triangle a c b ∨
      is_acute_triangle b c a)

theorem minimum_k_condition (k : ℕ) :
  (∀ s : Finset ℕ, s.card = k → any_subset_with_three_numbers_construct_acute_triangle s) ↔ (k = 29) :=
  sorry

end minimum_k_condition_l94_94867


namespace rachel_math_homework_l94_94407

/-- Rachel had to complete some pages of math homework. 
Given:
- 4 more pages of math homework than reading homework
- 3 pages of reading homework
Prove that Rachel had to complete 7 pages of math homework.
--/
theorem rachel_math_homework
  (r : ℕ) (h_r : r = 3)
  (m : ℕ) (h_m : m = r + 4) :
  m = 7 := by
  sorry

end rachel_math_homework_l94_94407


namespace books_sum_l94_94949

theorem books_sum (darryl_books lamont_books loris_books danielle_books : ℕ) 
  (h1 : darryl_books = 20)
  (h2 : lamont_books = 2 * darryl_books)
  (h3 : lamont_books = loris_books + 3)
  (h4 : danielle_books = lamont_books + darryl_books + 10) : 
  darryl_books + lamont_books + loris_books + danielle_books = 167 := 
by
  sorry

end books_sum_l94_94949


namespace work_problem_correct_l94_94773

noncomputable def work_problem : Prop :=
  let A := 1 / 36
  let C := 1 / 6
  let total_rate := 1 / 4
  ∃ B : ℝ, (A + B + C = total_rate) ∧ (B = 1 / 18)

-- Create the theorem statement which says if the conditions are met,
-- then the rate of b must be 1/18 and the number of days b alone takes to
-- finish the work is 18.
theorem work_problem_correct (A C total_rate B : ℝ) (h1 : A = 1 / 36) (h2 : C = 1 / 6) (h3 : total_rate = 1 / 4) 
(h4 : A + B + C = total_rate) : B = 1 / 18 ∧ (1 / B = 18) :=
  by
  sorry

end work_problem_correct_l94_94773


namespace max_value_of_a_l94_94157

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - a * x + 1

theorem max_value_of_a :
  ∃ (a : ℝ), (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) → |f a x| ≤ 1) ∧ a = 8 := by
  sorry

end max_value_of_a_l94_94157


namespace final_price_correct_l94_94462

def cost_price : ℝ := 20
def profit_percentage : ℝ := 0.30
def sale_discount_percentage : ℝ := 0.50
def local_tax_percentage : ℝ := 0.10
def packaging_fee : ℝ := 2

def selling_price_before_discount : ℝ := cost_price * (1 + profit_percentage)
def sale_discount : ℝ := sale_discount_percentage * selling_price_before_discount
def price_after_discount : ℝ := selling_price_before_discount - sale_discount
def tax : ℝ := local_tax_percentage * price_after_discount
def price_with_tax : ℝ := price_after_discount + tax
def final_price : ℝ := price_with_tax + packaging_fee

theorem final_price_correct : final_price = 16.30 :=
by
  sorry

end final_price_correct_l94_94462


namespace lucas_total_assignments_l94_94002

theorem lucas_total_assignments : 
  ∃ (total_assignments : ℕ), 
  (∀ (points : ℕ), 
    (points ≤ 10 → total_assignments = points * 1) ∧
    (10 < points ∧ points ≤ 20 → total_assignments = 10 * 1 + (points - 10) * 2) ∧
    (20 < points ∧ points ≤ 30 → total_assignments = 10 * 1 + 10 * 2 + (points - 20) * 3)
  ) ∧
  total_assignments = 60 :=
by
  sorry

end lucas_total_assignments_l94_94002


namespace zero_is_multiple_of_every_integer_l94_94274

theorem zero_is_multiple_of_every_integer (x : ℤ) : ∃ n : ℤ, 0 = n * x := by
  use 0
  exact (zero_mul x).symm

end zero_is_multiple_of_every_integer_l94_94274


namespace largest_n_l94_94195

theorem largest_n {x y z n : ℕ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (n:ℤ)^2 = (x:ℤ)^2 + (y:ℤ)^2 + (z:ℤ)^2 + 2*(x:ℤ)*(y:ℤ) + 2*(y:ℤ)*(z:ℤ) + 2*(z:ℤ)*(x:ℤ) + 6*(x:ℤ) + 6*(y:ℤ) + 6*(z:ℤ) - 12
  → n = 13 :=
sorry

end largest_n_l94_94195


namespace polygon_sides_l94_94950

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l94_94950


namespace find_t_l94_94110

theorem find_t (t : ℝ) : 
  (∃ a b : ℝ, a^2 = t^2 ∧ b^2 = 5 * t ∧ (a - b = 2 * Real.sqrt 6 ∨ b - a = 2 * Real.sqrt 6)) → 
  (t = 2 ∨ t = 3 ∨ t = 6) := 
by
  sorry

end find_t_l94_94110


namespace sequence_term_and_k_value_l94_94501

/-- Given a sequence {a_n} whose sum of the first n terms is S_n = n^2 - 9n,
    prove the sequence term a_n = 2n - 10, and if 5 < a_k < 8, then k = 8. -/
theorem sequence_term_and_k_value (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 9 * n) :
  (∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) →
  (∀ n, a n = 2 * n - 10) ∧ (∀ k, 5 < a k ∧ a k < 8 → k = 8) :=
by {
  -- Given S_n = n^2 - 9n, we need to show a_n = 2n - 10 and verify when 5 < a_k < 8, then k = 8
  sorry
}

end sequence_term_and_k_value_l94_94501


namespace lucy_apples_per_week_l94_94699

-- Define the conditions
def chandler_apples_per_week := 23
def total_apples_per_month := 168
def weeks_per_month := 4
def chandler_apples_per_month := chandler_apples_per_week * weeks_per_month
def lucy_apples_per_month := total_apples_per_month - chandler_apples_per_month

-- Define the proof problem statement
theorem lucy_apples_per_week :
  lucy_apples_per_month / weeks_per_month = 19 :=
  by sorry

end lucy_apples_per_week_l94_94699


namespace child_l94_94434

noncomputable def child's_ticket_cost : ℕ :=
  let adult_ticket_price := 7
  let total_tickets := 900
  let total_revenue := 5100
  let childs_tickets_sold := 400
  let adult_tickets_sold := total_tickets - childs_tickets_sold
  let total_adult_revenue := adult_tickets_sold * adult_ticket_price
  let total_child_revenue := total_revenue - total_adult_revenue
  let child's_ticket_price := total_child_revenue / childs_tickets_sold
  child's_ticket_price

theorem child's_ticket_cost_is_4 : child's_ticket_cost = 4 :=
by
  have adult_ticket_price := 7
  have total_tickets := 900
  have total_revenue := 5100
  have childs_tickets_sold := 400
  have adult_tickets_sold := total_tickets - childs_tickets_sold
  have total_adult_revenue := adult_tickets_sold * adult_ticket_price
  have total_child_revenue := total_revenue - total_adult_revenue
  have child's_ticket_price := total_child_revenue / childs_tickets_sold
  show child's_ticket_cost = 4
  sorry

end child_l94_94434


namespace division_of_decimals_l94_94420

theorem division_of_decimals :
  (0.1 / 0.001 = 100) ∧ (1 / 0.01 = 100) := by
  sorry

end division_of_decimals_l94_94420


namespace cost_of_cd_l94_94769

theorem cost_of_cd 
  (cost_film : ℕ) (cost_book : ℕ) (total_spent : ℕ) (num_cds : ℕ) (total_cost_films : ℕ)
  (total_cost_books : ℕ) (cost_cd : ℕ) : 
  cost_film = 5 → cost_book = 4 → total_spent = 79 →
  total_cost_films = 9 * cost_film → total_cost_books = 4 * cost_book →
  total_spent = total_cost_films + total_cost_books + num_cds * cost_cd →
  num_cds = 6 →
  cost_cd = 3 := 
by {
  -- proof would go here
  sorry
}

end cost_of_cd_l94_94769


namespace trigonometric_comparison_l94_94213

open Real

theorem trigonometric_comparison :
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  a < b ∧ b < c := 
by
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  sorry

end trigonometric_comparison_l94_94213


namespace square_root_condition_l94_94083

-- Define the condition
def meaningful_square_root (x : ℝ) : Prop :=
  x - 5 ≥ 0

-- Define the theorem that x must be greater than or equal to 5 for the square root to be meaningful
theorem square_root_condition (x : ℝ) : meaningful_square_root x ↔ x ≥ 5 := by
  sorry

end square_root_condition_l94_94083


namespace faucets_fill_time_l94_94433

theorem faucets_fill_time (fill_time_4faucets_200gallons_12min : 4 * 12 * faucet_rate = 200) 
    (fill_time_m_50gallons_seconds : ∃ (rate: ℚ), 8 * t_to_seconds * rate = 50) : 
    8 * t_to_seconds / 33.33 = 90 :=
by sorry


end faucets_fill_time_l94_94433


namespace find_a_l94_94350

noncomputable def binomial_expansion_term_coefficient
  (n : ℕ) (r : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  (2^(n-r)) * ((-a)^r) * (Nat.choose n r) * (x^(n - 2*r))

theorem find_a 
  (a : ℝ)
  (h : binomial_expansion_term_coefficient 7 5 a 1 = 84) 
  : a = -1 :=
sorry

end find_a_l94_94350


namespace largest_angle_of_pentagon_l94_94141

-- Define the angles of the pentagon and the conditions on them.
def is_angle_of_pentagon (A B C D E : ℝ) :=
  A = 108 ∧ B = 72 ∧ C = D ∧ E = 3 * C ∧
  A + B + C + D + E = 540

-- Prove the largest angle in the pentagon is 216
theorem largest_angle_of_pentagon (A B C D E : ℝ) (h : is_angle_of_pentagon A B C D E) :
  max (max (max (max A B) C) D) E = 216 :=
by
  sorry

end largest_angle_of_pentagon_l94_94141


namespace amateur_definition_l94_94341
-- Import necessary libraries

-- Define the meaning of "amateur" and state that it is "amateurish" or "non-professional"
def meaning_of_amateur : String :=
  "amateurish or non-professional"

-- The main statement asserting that the meaning of "amateur" is indeed "amateurish" or "non-professional"
theorem amateur_definition : meaning_of_amateur = "amateurish or non-professional" :=
by
  -- The proof is trivial and assumed to be correct
  sorry

end amateur_definition_l94_94341


namespace inequality_holds_for_all_x_l94_94300

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end inequality_holds_for_all_x_l94_94300


namespace lcm_prime_factors_l94_94285

-- Conditions
def n1 := 48
def n2 := 180
def n3 := 250

-- The equivalent proof problem
theorem lcm_prime_factors (l : ℕ) (h1: l = Nat.lcm n1 (Nat.lcm n2 n3)) :
  l = 18000 ∧ (∀ a : ℕ, a ∣ l ↔ a ∣ 2^4 * 3^2 * 5^3) :=
by
  sorry

end lcm_prime_factors_l94_94285


namespace ellen_legos_final_count_l94_94643

-- Definitions based on conditions
def initial_legos : ℕ := 380
def lost_legos_first_week : ℕ := 57
def additional_legos_second_week (remaining_legos : ℕ) : ℕ := 32
def borrowed_legos_third_week (total_legos : ℕ) : ℕ := 88

-- Computed values based on conditions
def legos_after_first_week (initial : ℕ) (lost : ℕ) : ℕ := initial - lost
def legos_after_second_week (remaining : ℕ) (additional : ℕ) : ℕ := remaining + additional
def legos_after_third_week (total : ℕ) (borrowed : ℕ) : ℕ := total - borrowed

-- Proof statement
theorem ellen_legos_final_count : 
  legos_after_third_week 
    (legos_after_second_week 
      (legos_after_first_week initial_legos lost_legos_first_week)
      (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))
    (borrowed_legos_third_week (legos_after_second_week 
                                  (legos_after_first_week initial_legos lost_legos_first_week)
                                  (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))) 
  = 267 :=
by 
  sorry

end ellen_legos_final_count_l94_94643


namespace max_constant_term_l94_94517

theorem max_constant_term (c : ℝ) : 
  (∀ x : ℝ, (x^2 - 6 * x + c = 0 → (x^2 - 6 * x + c ≥ 0))) → c ≤ 9 :=
by sorry

end max_constant_term_l94_94517


namespace necessary_but_not_sufficient_l94_94754

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l94_94754


namespace findLastNames_l94_94207

noncomputable def peachProblem : Prop :=
  ∃ (a b c d : ℕ),
    2 * a + 3 * b + 4 * c + 5 * d = 32 ∧
    a + b + c + d = 10 ∧
    (a = 3 ∧ b = 4 ∧ c = 1 ∧ d = 2)

theorem findLastNames :
  peachProblem :=
sorry

end findLastNames_l94_94207


namespace valid_starting_lineups_correct_l94_94614

-- Define the parameters from the problem
def volleyball_team : Finset ℕ := Finset.range 18
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the main computation: total lineups excluding those where all quadruplets are chosen
noncomputable def valid_starting_lineups : ℕ :=
  (volleyball_team.card.choose 7) - ((volleyball_team \ quadruplets).card.choose 3)

-- The theorem states that the number of valid starting lineups is 31460
theorem valid_starting_lineups_correct : valid_starting_lineups = 31460 := by
  sorry

end valid_starting_lineups_correct_l94_94614


namespace quadratic_radicals_x_le_10_l94_94423

theorem quadratic_radicals_x_le_10 (a x : ℝ) (h1 : 3 * a - 8 = 17 - 2 * a) (h2 : 4 * a - 2 * x ≥ 0) : x ≤ 10 :=
by
  sorry

end quadratic_radicals_x_le_10_l94_94423


namespace tricycles_count_l94_94817

-- Define the conditions
variable (b t s : ℕ)

def total_children := b + t + s = 10
def total_wheels := 2 * b + 3 * t + 2 * s = 29

-- Provide the theorem to prove
theorem tricycles_count (h1 : total_children b t s) (h2 : total_wheels b t s) : t = 9 := 
by
  sorry

end tricycles_count_l94_94817


namespace expression_evaluation_l94_94857

theorem expression_evaluation : |(-7: ℤ)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end expression_evaluation_l94_94857


namespace prime_of_the_form_4x4_plus_1_l94_94550

theorem prime_of_the_form_4x4_plus_1 (x : ℤ) (p : ℤ) (h : 4 * x ^ 4 + 1 = p) (hp : Prime p) : p = 5 :=
sorry

end prime_of_the_form_4x4_plus_1_l94_94550


namespace heesu_received_most_sweets_l94_94312

theorem heesu_received_most_sweets
  (total_sweets : ℕ)
  (minsus_sweets : ℕ)
  (jaeyoungs_sweets : ℕ)
  (heesus_sweets : ℕ)
  (h_total : total_sweets = 30)
  (h_minsu : minsus_sweets = 12)
  (h_jaeyoung : jaeyoungs_sweets = 3)
  (h_heesu : heesus_sweets = 15) :
  heesus_sweets = max minsus_sweets (max jaeyoungs_sweets heesus_sweets) :=
by sorry

end heesu_received_most_sweets_l94_94312


namespace basketball_club_boys_l94_94315

theorem basketball_club_boys (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : B + (1 / 3) * G = 18) : B = 12 := 
by
  sorry

end basketball_club_boys_l94_94315


namespace trader_profit_percentage_l94_94878

theorem trader_profit_percentage (P : ℝ) (h₀ : 0 ≤ P) : 
  let discount := 0.40
  let increase := 0.80
  let purchase_price := P * (1 - discount)
  let selling_price := purchase_price * (1 + increase)
  let profit := selling_price - P
  (profit / P) * 100 = 8 := 
by
  sorry

end trader_profit_percentage_l94_94878


namespace san_antonio_to_austin_buses_passed_l94_94722

def departure_schedule (departure_time_A_to_S departure_time_S_to_A travel_time : ℕ) : Prop :=
  ∀ t, (t < travel_time) →
       (∃ n, t = (departure_time_A_to_S + n * 60)) ∨
       (∃ m, t = (departure_time_S_to_A + m * 60)) →
       t < travel_time

theorem san_antonio_to_austin_buses_passed :
  let departure_time_A_to_S := 30  -- Austin to San Antonio buses leave every hour on the half-hour (e.g., 00:30, 1:30, ...)
  let departure_time_S_to_A := 0   -- San Antonio to Austin buses leave every hour on the hour (e.g., 00:00, 1:00, ...)
  let travel_time := 6 * 60        -- The trip takes 6 hours, or 360 minutes
  departure_schedule departure_time_A_to_S departure_time_S_to_A travel_time →
  ∃ count, count = 12 := 
by
  sorry

end san_antonio_to_austin_buses_passed_l94_94722


namespace rectangle_perimeter_l94_94464

theorem rectangle_perimeter :
  ∃ (a b : ℤ), a ≠ b ∧ a * b = 2 * (2 * a + 2 * b) ∧ 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l94_94464


namespace num_people_price_item_equation_l94_94644

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l94_94644


namespace xy_value_is_one_l94_94048

open Complex

theorem xy_value_is_one (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : x * y = 1 :=
by
  sorry

end xy_value_is_one_l94_94048


namespace simplest_square_root_l94_94879

theorem simplest_square_root (a b c d : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 3) (h3 : c = (Real.sqrt 2) / 2) (h4 : d = Real.sqrt 10) :
  d = Real.sqrt 10 ∧ (a ≠ Real.sqrt 10) ∧ (b ≠ Real.sqrt 10) ∧ (c ≠ Real.sqrt 10) := 
by 
  sorry

end simplest_square_root_l94_94879


namespace calculate_p_p1_neg1_p_neg5_neg2_l94_94443

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then
    x + y
  else if x < 0 ∧ y < 0 then
    x - 2 * y
  else
    3 * x + y

theorem calculate_p_p1_neg1_p_neg5_neg2 :
  p (p 1 (-1)) (p (-5) (-2)) = 5 :=
by
  sorry

end calculate_p_p1_neg1_p_neg5_neg2_l94_94443


namespace points_per_member_l94_94496

theorem points_per_member
  (total_members : ℕ)
  (members_didnt_show : ℕ)
  (total_points : ℕ)
  (H1 : total_members = 14)
  (H2 : members_didnt_show = 7)
  (H3 : total_points = 35) :
  total_points / (total_members - members_didnt_show) = 5 :=
by
  sorry

end points_per_member_l94_94496


namespace range_of_a_for_extreme_points_l94_94574

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

theorem range_of_a_for_extreme_points :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    ∀ a : ℝ, 0 < a ∧ a < (1 / 2) →
    (Real.exp x₁ * (x₁ + 1 - 2 * a * Real.exp x₁) = 0) ∧ 
    (Real.exp x₂ * (x₂ + 1 - 2 * a * Real.exp x₂) = 0)) ↔ 
  ∀ a : ℝ, 0 < a ∧ a < (1 / 2) :=
sorry

end range_of_a_for_extreme_points_l94_94574


namespace Craig_bench_press_percentage_l94_94973

theorem Craig_bench_press_percentage {Dave_weight : ℕ} (h1 : Dave_weight = 175) (h2 : ∀ w : ℕ, Dave_bench_press = 3 * Dave_weight) 
(Craig_bench_press Mark_bench_press : ℕ) (h3 : Mark_bench_press = 55) (h4 : Mark_bench_press = Craig_bench_press - 50) : 
(Craig_bench_press / (3 * Dave_weight) * 100) = 20 := by
  sorry

end Craig_bench_press_percentage_l94_94973


namespace customers_in_each_car_l94_94990

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end customers_in_each_car_l94_94990


namespace Jean_calls_thursday_l94_94495

theorem Jean_calls_thursday :
  ∃ (thursday_calls : ℕ), thursday_calls = 61 ∧ 
  (∃ (mon tue wed fri : ℕ),
    mon = 35 ∧ 
    tue = 46 ∧ 
    wed = 27 ∧ 
    fri = 31 ∧ 
    (mon + tue + wed + thursday_calls + fri = 40 * 5)) :=
sorry

end Jean_calls_thursday_l94_94495


namespace f_is_odd_function_f_is_increasing_f_max_min_in_interval_l94_94556

variable {f : ℝ → ℝ}

-- The conditions:
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom positive_for_positive : ∀ x : ℝ, x > 0 → f x > 0
axiom f_one_is_two : f 1 = 2

-- The proof tasks:
theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := 
sorry

theorem f_max_min_in_interval : 
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≤ 6) ∧ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -6) :=
sorry

end f_is_odd_function_f_is_increasing_f_max_min_in_interval_l94_94556


namespace original_cost_price_l94_94037

theorem original_cost_price (C : ℝ) 
  (h1 : 0.87 * C > 0) 
  (h2 : 1.2 * (0.87 * C) = 54000) : 
  C = 51724.14 :=
by
  sorry

end original_cost_price_l94_94037


namespace actual_revenue_percent_of_projected_l94_94684

noncomputable def projected_revenue (R : ℝ) : ℝ := 1.2 * R
noncomputable def actual_revenue (R : ℝ) : ℝ := 0.75 * R

theorem actual_revenue_percent_of_projected (R : ℝ) :
  (actual_revenue R / projected_revenue R) * 100 = 62.5 :=
  sorry

end actual_revenue_percent_of_projected_l94_94684


namespace cuboid_edge_length_l94_94522

theorem cuboid_edge_length
  (x : ℝ)
  (h_surface_area : 2 * (4 * x + 24 + 6 * x) = 148) :
  x = 5 :=
by
  sorry

end cuboid_edge_length_l94_94522


namespace quadratic_root_m_l94_94475

theorem quadratic_root_m (m : ℝ) : (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → (m = 1 ∨ m = -1) :=
by 
  sorry

end quadratic_root_m_l94_94475


namespace people_not_in_any_club_l94_94765

def num_people_company := 120
def num_people_club_A := 25
def num_people_club_B := 34
def num_people_club_C := 21
def num_people_club_D := 16
def num_people_club_E := 10
def overlap_C_D := 8
def overlap_D_E := 4

theorem people_not_in_any_club :
  num_people_company - 
  (num_people_club_A + num_people_club_B + 
  (num_people_club_C + (num_people_club_D - overlap_C_D) + (num_people_club_E - overlap_D_E))) = 26 :=
by
  unfold num_people_company num_people_club_A num_people_club_B num_people_club_C num_people_club_D num_people_club_E overlap_C_D overlap_D_E
  sorry

end people_not_in_any_club_l94_94765


namespace lcm_15_48_eq_240_l94_94563

def is_least_common_multiple (n a b : Nat) : Prop :=
  n % a = 0 ∧ n % b = 0 ∧ ∀ m, (m % a = 0 ∧ m % b = 0) → n ≤ m

theorem lcm_15_48_eq_240 : is_least_common_multiple 240 15 48 :=
by
  sorry

end lcm_15_48_eq_240_l94_94563


namespace coins_with_specific_probabilities_impossible_l94_94153

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l94_94153


namespace find_complex_number_l94_94933

def i := Complex.I
def z := -Complex.I - 1
def complex_equation (z : ℂ) := i * z = 1 - i

theorem find_complex_number : complex_equation z :=
by
  -- skip the proof here
  sorry

end find_complex_number_l94_94933


namespace odd_square_mod_eight_l94_94410

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l94_94410


namespace vector_subtraction_l94_94892

def a : ℝ × ℝ := (5, 3)
def b : ℝ × ℝ := (1, -2)
def scalar : ℝ := 2

theorem vector_subtraction :
  a.1 - scalar * b.1 = 3 ∧ a.2 - scalar * b.2 = 7 :=
by {
  -- here goes the proof
  sorry
}

end vector_subtraction_l94_94892


namespace repeating_decimals_subtraction_l94_94618

def x : Rat := 1 / 3
def y : Rat := 2 / 99

theorem repeating_decimals_subtraction :
  x - y = 31 / 99 :=
sorry

end repeating_decimals_subtraction_l94_94618


namespace at_least_two_squares_same_size_l94_94494

theorem at_least_two_squares_same_size (S : ℝ) : 
  ∃ a b : ℝ, a = b ∧ 
  (∀ i : ℕ, i < 10 → 
   ∀ j : ℕ, j < 10 → 
   (∃ k : ℕ, k < 9 ∧ 
    ((∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ x ≠ y → 
          (i = x ∧ j = y)) → 
        ((S / 10) = (a * k)) ∨ ((S / 10) = (b * k))))) := sorry

end at_least_two_squares_same_size_l94_94494


namespace problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l94_94109

-- Problem 1: Prove the remainder of the Euclidean division of \(9^{100}\) by 8 is 1.
theorem problem1_remainder_of_9_power_100_mod_8 :
  (9 ^ 100) % 8 = 1 :=
by
sorry

-- Problem 2: Prove the last digit of \(2012^{2012}\) is 6.
theorem problem2_last_digit_of_2012_power_2012 :
  (2012 ^ 2012) % 10 = 6 :=
by
sorry

end problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l94_94109


namespace overtime_percentage_increase_l94_94808

-- Define the conditions.
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def total_compensation : ℝ := 1116
def total_hours_worked : ℕ := 57
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Define the question and the answer as a proof problem.
theorem overtime_percentage_increase :
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  overtime_rate > regular_rate →
  ((overtime_rate - regular_rate) / regular_rate) * 100 = 75 := 
by
  sorry

end overtime_percentage_increase_l94_94808


namespace no_solution_for_p_eq_7_l94_94071

theorem no_solution_for_p_eq_7 : ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ( (x-3)/(x-4) = (x-7)/(x-8) ) → false := by
  intro x h1 h2 h
  sorry

end no_solution_for_p_eq_7_l94_94071


namespace solution_to_system_l94_94476

theorem solution_to_system (x y a b : ℝ) 
  (h1 : x = 1) (h2 : y = 2) 
  (h3 : a * x + b * y = 4) 
  (h4 : b * x - a * y = 7) : 
  a + b = 1 :=
by
  sorry

end solution_to_system_l94_94476


namespace least_number_to_add_l94_94014

theorem least_number_to_add (n : ℕ) :
  (exists n, 1202 + n % 4 = 0 ∧ (∀ m, (1202 + m) % 4 = 0 → n ≤ m)) → n = 2 :=
by
  sorry

end least_number_to_add_l94_94014


namespace border_area_is_correct_l94_94385

def framed_area (height width border: ℝ) : ℝ :=
  (height + 2 * border) * (width + 2 * border)

def photograph_area (height width: ℝ) : ℝ :=
  height * width

theorem border_area_is_correct (h w b : ℝ) (h6 : h = 6) (w8 : w = 8) (b3 : b = 3) :
  (framed_area h w b - photograph_area h w) = 120 := by
  sorry

end border_area_is_correct_l94_94385


namespace trig_problem_l94_94936

variable (α : ℝ)

theorem trig_problem
  (h1 : Real.sin (Real.pi + α) = -1 / 3) :
  Real.cos (α - 3 * Real.pi / 2) = -1 / 3 ∧
  (Real.sin (Real.pi / 2 + α) = 2 * Real.sqrt 2 / 3 ∨ Real.sin (Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3) ∧
  (Real.tan (5 * Real.pi - α) = -Real.sqrt 2 / 4 ∨ Real.tan (5 * Real.pi - α) = Real.sqrt 2 / 4) :=
sorry

end trig_problem_l94_94936


namespace chris_birthday_after_45_days_l94_94064

theorem chris_birthday_after_45_days (k : ℕ) (h : k = 45) (tuesday : ℕ) (h_tuesday : tuesday = 2) : 
  (tuesday + k) % 7 = 5 := 
sorry

end chris_birthday_after_45_days_l94_94064


namespace range_of_a_l94_94258

noncomputable def f (a : ℝ) (x : ℝ) := x * Real.log x - a * x^2

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l94_94258


namespace negation_of_diagonals_equal_l94_94329

-- Define a rectangle type and a function for the diagonals being equal
structure Rectangle :=
  (a b c d : ℝ) -- Assuming rectangle sides

-- Assume a function that checks if the diagonals of a given rectangle are equal
def diagonals_are_equal (r : Rectangle) : Prop :=
  sorry -- The actual function definition is omitted for this context

-- The proof problem
theorem negation_of_diagonals_equal :
  ¬ (∀ r : Rectangle, diagonals_are_equal r) ↔ (∃ r : Rectangle, ¬ diagonals_are_equal r) :=
by
  sorry

end negation_of_diagonals_equal_l94_94329


namespace problem_solution_l94_94016

theorem problem_solution (m : ℤ) (x : ℤ) (h : 4 * x + 2 * m = 14) : x = 2 → m = 3 :=
by sorry

end problem_solution_l94_94016


namespace quadratic_equation_with_means_l94_94135

theorem quadratic_equation_with_means (α β : ℝ) 
  (h_am : (α + β) / 2 = 8) 
  (h_gm : Real.sqrt (α * β) = 15) : 
  (Polynomial.X^2 - Polynomial.C (α + β) * Polynomial.X + Polynomial.C (α * β) = 0) := 
by
  have h1 : α + β = 16 := by linarith
  have h2 : α * β = 225 := by sorry
  rw [h1, h2]
  sorry

end quadratic_equation_with_means_l94_94135


namespace zero_point_interval_l94_94695

noncomputable def f (x : ℝ) := 6 / x - x ^ 2

theorem zero_point_interval : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l94_94695


namespace coordinates_of_point_with_respect_to_origin_l94_94534

theorem coordinates_of_point_with_respect_to_origin (P : ℝ × ℝ) (h : P = (-2, 4)) : P = (-2, 4) := 
by 
  exact h

end coordinates_of_point_with_respect_to_origin_l94_94534


namespace correct_result_l94_94943

-- Definitions to capture the problem conditions:
def cond1 (a b : ℤ) : Prop := 5 * a^2 * b - 2 * a^2 * b = 3 * a^2 * b
def cond2 (x : ℤ) : Prop := x^6 / x^2 = x^4
def cond3 (a b : ℤ) : Prop := (a - b)^2 = a^2 - b^2

-- Proof statement to verify the correct answer
theorem correct_result (x : ℤ) : (2 * x^2)^3 = 8 * x^6 :=
  by sorry

-- Note that cond1, cond2, and cond3 are intended to capture the erroneous conditions mentioned for completeness.

end correct_result_l94_94943


namespace inequality_system_correctness_l94_94608

theorem inequality_system_correctness :
  (∀ (x a b : ℝ), 
    (x - a ≥ 1) ∧ (x - b < 2) →
    ((∀ x, -1 ≤ x ∧ x < 3 → (a = -2 ∧ b = 1)) ∧
     (a = b → (a + 1 ≤ x ∧ x < a + 2)) ∧
     (¬(∃ x, a + 1 ≤ x ∧ x < b + 2) → a > b + 1) ∧
     ((∃ n : ℤ, n < 0 ∧ n ≥ -6 - a ∧ n ≥ -5) → -7 < a ∧ a ≤ -6))) :=
sorry

end inequality_system_correctness_l94_94608


namespace translate_parabola_l94_94313

theorem translate_parabola (x y : ℝ) :
  (y = 2 * x^2 + 3) →
  (∃ x y, y = 2 * (x - 3)^2 + 5) :=
sorry

end translate_parabola_l94_94313


namespace am_hm_inequality_l94_94405

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : ℝ :=
  (a + b + c) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem am_hm_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  smallest_possible_value a b c d h1 h2 h3 h4 ≥ 9 / 2 :=
by
  sorry

end am_hm_inequality_l94_94405


namespace problem_a51_l94_94379

-- Definitions of given conditions
variable {a : ℕ → ℤ}
variable (h1 : ∀ n : ℕ, a (n + 2) - 2 * a (n + 1) + a n = 16)
variable (h2 : a 63 = 10)
variable (h3 : a 89 = 10)

-- Proof problem statement
theorem problem_a51 :
  a 51 = 3658 :=
by
  sorry

end problem_a51_l94_94379


namespace algebraic_expression_correct_l94_94478

-- Definition of the problem
def algebraic_expression (x : ℝ) : ℝ :=
  2 * x + 3

-- Theorem statement
theorem algebraic_expression_correct (x : ℝ) :
  algebraic_expression x = 2 * x + 3 :=
by
  sorry

end algebraic_expression_correct_l94_94478


namespace negation_proposition_l94_94863

theorem negation_proposition :
  (∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proposition_l94_94863


namespace square_area_720_l94_94484

noncomputable def length_squared {α : Type*} [EuclideanDomain α] (a b : α) := a * a + b * b

theorem square_area_720
  (side x : ℝ)
  (h1 : BE = 20) (h2 : EF = 20) (h3 : FD = 20)
  (h4 : AE = 2 * ED) (h5 : BF = 2 * FC)
  : x * x = 720 :=
by
  let AE := 2/3 * side
  let ED := 1/3 * side
  let BF := 2/3 * side
  let FC := 1/3 * side
  have h6 : length_squared BF EF = BE * BE := sorry
  have h7 : x * x = 720 := sorry
  exact h7

end square_area_720_l94_94484


namespace slope_range_l94_94609

theorem slope_range (k : ℝ) : 
  (∃ (x : ℝ), ∀ (y : ℝ), y = k * (x - 1) + 1) ∧ (0 < 1 - k ∧ 1 - k < 2) → (-1 < k ∧ k < 1) :=
by
  sorry

end slope_range_l94_94609


namespace Bill_donut_combinations_correct_l94_94270

/-- Bill has to purchase exactly six donuts from a shop with four kinds of donuts, ensuring he gets at least one of each kind. -/
def Bill_donut_combinations : ℕ :=
  let k := 4  -- number of kinds of donuts
  let n := 6  -- total number of donuts Bill needs to buy
  let m := 2  -- remaining donuts after buying one of each kind
  let same_kind := k          -- ways to choose 2 donuts of the same kind
  let different_kind := (k * (k - 1)) / 2  -- ways to choose 2 donuts of different kinds
  same_kind + different_kind

theorem Bill_donut_combinations_correct : Bill_donut_combinations = 10 :=
  by
    sorry  -- Proof is omitted; we assert this statement is true

end Bill_donut_combinations_correct_l94_94270


namespace spring_problem_l94_94388

theorem spring_problem (x y : ℝ) : 
  (∀ x, y = 0.5 * x + 12) →
  (0.5 * 3 + 12 = 13.5) ∧
  (y = 0.5 * x + 12) ∧
  (0.5 * 5.5 + 12 = 14.75) ∧
  (20 = 0.5 * 16 + 12) :=
by 
  sorry

end spring_problem_l94_94388


namespace option_A_option_B_option_C_option_D_l94_94910

theorem option_A : (-4:ℤ)^2 ≠ -(4:ℤ)^2 := sorry
theorem option_B : (-2:ℤ)^3 = -2^3 := sorry
theorem option_C : (-1:ℤ)^2020 ≠ (-1:ℤ)^2021 := sorry
theorem option_D : ((2:ℚ)/(3:ℚ))^3 = ((2:ℚ)/(3:ℚ))^3 := sorry

end option_A_option_B_option_C_option_D_l94_94910


namespace hotel_ticket_ratio_l94_94688

theorem hotel_ticket_ratio (initial_amount : ℕ) (remaining_amount : ℕ) (ticket_cost : ℕ) (hotel_cost : ℕ) :
  initial_amount = 760 →
  remaining_amount = 310 →
  ticket_cost = 300 →
  initial_amount - remaining_amount - ticket_cost = hotel_cost →
  (hotel_cost : ℚ) / (ticket_cost : ℚ) = 1 / 2 :=
by
  intros h_initial h_remaining h_ticket h_hotel
  sorry

end hotel_ticket_ratio_l94_94688


namespace number_of_people_entered_l94_94604

-- Define the total number of placards
def total_placards : ℕ := 5682

-- Define the number of placards each person takes
def placards_per_person : ℕ := 2

-- The Lean theorem to prove the number of people who entered the stadium
theorem number_of_people_entered : total_placards / placards_per_person = 2841 :=
by
  -- Proof will be inserted here
  sorry

end number_of_people_entered_l94_94604


namespace dilation_image_l94_94515

theorem dilation_image 
  (z z₀ : ℂ) (k : ℝ) 
  (hz : z = -2 + i) 
  (hz₀ : z₀ = 1 - 3 * I) 
  (hk : k = 3) : 
  (k * (z - z₀) + z₀) = (-8 + 9 * I) := 
by 
  rw [hz, hz₀, hk]
  -- Sorry means here we didn't write the complete proof, we assume it is correct.
  sorry

end dilation_image_l94_94515


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l94_94117

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_13 : 
  ∃ p : ℕ, (p ∣ 16385 ∧ Nat.Prime p ∧ (∀ q : ℕ, q ∣ 16385 → Nat.Prime q → q ≤ p)) ∧ (Nat.digits 10 p).sum = 13 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l94_94117


namespace length_in_scientific_notation_l94_94063

theorem length_in_scientific_notation : (161000 : ℝ) = 1.61 * 10^5 := 
by 
  -- Placeholder proof
  sorry

end length_in_scientific_notation_l94_94063


namespace remainder_of_x_squared_div_20_l94_94015

theorem remainder_of_x_squared_div_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 4 * x ≡ 12 [ZMOD 20]) :
  (x * x) % 20 = 4 :=
sorry

end remainder_of_x_squared_div_20_l94_94015


namespace monotone_f_find_m_l94_94897

noncomputable def f (x : ℝ) : ℝ := (2 * x - 2) / (x + 2)

theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by
  sorry

theorem find_m (m : ℝ) : 
  (∃ m, (f m - f 1 = 1/2)) ↔ m = 2 :=
by
  sorry

end monotone_f_find_m_l94_94897


namespace find_b_l94_94785

variables (U : Set ℝ) (A : Set ℝ) (b : ℝ)

theorem find_b (hU : U = Set.univ)
               (hA : A = {x | 1 ≤ x ∧ x < b})
               (hComplA : U \ A = {x | x < 1 ∨ x ≥ 2}) :
  b = 2 :=
sorry

end find_b_l94_94785


namespace quadrilateral_area_correct_l94_94917

-- Definitions of given conditions
structure Quadrilateral :=
(W X Y Z : Type)
(WX XY YZ YW : ℝ)
(angle_WXY : ℝ)
(area : ℝ)

-- Quadrilateral satisfies given conditions
def quadrilateral_WXYZ : Quadrilateral :=
{ W := ℝ,
  X := ℝ,
  Y := ℝ,
  Z := ℝ,
  WX := 9,
  XY := 5,
  YZ := 12,
  YW := 15,
  angle_WXY := 90,
  area := 76.5 }

-- The theorem stating the area of quadrilateral WXYZ is 76.5
theorem quadrilateral_area_correct : quadrilateral_WXYZ.area = 76.5 :=
sorry

end quadrilateral_area_correct_l94_94917


namespace rice_weight_per_container_l94_94169

-- Given total weight of rice in pounds
def totalWeightPounds : ℚ := 25 / 2

-- Conversion factor from pounds to ounces
def poundsToOunces : ℚ := 16

-- Number of containers
def numberOfContainers : ℕ := 4

-- Total weight in ounces
def totalWeightOunces : ℚ := totalWeightPounds * poundsToOunces

-- Weight per container in ounces
def weightPerContainer : ℚ := totalWeightOunces / numberOfContainers

theorem rice_weight_per_container :
  weightPerContainer = 50 := 
sorry

end rice_weight_per_container_l94_94169


namespace find_expression_l94_94525

theorem find_expression (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 3 * x + 2) : 
  ∀ x : ℤ, f x = 3 * x - 1 :=
sorry

end find_expression_l94_94525


namespace count_triangles_in_3x3_grid_l94_94232

/--
In a 3x3 grid of dots, the number of triangles formed by connecting the dots is 20.
-/
def triangles_in_3x3_grid : Prop :=
  let num_rows := 3
  let num_cols := 3
  let total_triangles := 20
  ∃ (n : ℕ), n = total_triangles ∧ n = 20

theorem count_triangles_in_3x3_grid : triangles_in_3x3_grid :=
by {
  -- Insert the proof here
  sorry
}

end count_triangles_in_3x3_grid_l94_94232


namespace specific_value_of_n_l94_94003

theorem specific_value_of_n (n : ℕ) 
  (A_n : ℕ → ℕ)
  (C_n : ℕ → ℕ → ℕ)
  (h1 : A_n n ^ 2 = C_n n (n-3)) :
  n = 8 :=
sorry

end specific_value_of_n_l94_94003


namespace max_books_borrowed_l94_94047

theorem max_books_borrowed (students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) (more_books : ℕ)
  (h_students : students = 30)
  (h_no_books : no_books = 5)
  (h_one_book : one_book = 12)
  (h_two_books : two_books = 8)
  (h_more_books : more_books = students - no_books - one_book - two_books)
  (avg_books : ℕ)
  (h_avg_books : avg_books = 2) :
  ∃ max_books : ℕ, max_books = 20 := 
by 
  sorry

end max_books_borrowed_l94_94047


namespace factor_expression_l94_94906

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) :=
by sorry

end factor_expression_l94_94906


namespace find_y_value_l94_94231

theorem find_y_value :
  ∀ (y : ℝ), (dist (1, 3) (7, y) = 13) ∧ (y > 0) → y = 3 + Real.sqrt 133 :=
by
  sorry

end find_y_value_l94_94231


namespace region_to_the_upper_left_of_line_l94_94180

variable (x y : ℝ)

def line_eqn := 3 * x - 2 * y - 6 = 0

def region := 3 * x - 2 * y - 6 < 0

theorem region_to_the_upper_left_of_line :
  ∃ rect_upper_left, (rect_upper_left = region) := 
sorry

end region_to_the_upper_left_of_line_l94_94180


namespace volume_ratio_of_smaller_snowball_l94_94244

theorem volume_ratio_of_smaller_snowball (r : ℝ) (k : ℝ) :
  let V₀ := (4/3) * π * r^3
  let S := 4 * π * r^2
  let V_large := (4/3) * π * (2 * r)^3
  let V_large_half := V_large / 2
  let new_r := (V_large_half / ((4/3) * π))^(1/3)
  let reduction := 2*r - new_r
  let remaining_r := r - reduction
  let remaining_V := (4/3) * π * remaining_r^3
  let volume_ratio := remaining_V / V₀ 
  volume_ratio = 1/5 :=
by
  -- Proof goes here
  sorry

end volume_ratio_of_smaller_snowball_l94_94244


namespace units_digit_6_l94_94764

theorem units_digit_6 (p : ℤ) (hp : 0 < p % 10) (h1 : (p^3 % 10) = (p^2 % 10)) (h2 : (p + 2) % 10 = 8) : p % 10 = 6 :=
by
  sorry

end units_digit_6_l94_94764


namespace eugene_initial_pencils_l94_94502

theorem eugene_initial_pencils (e given left : ℕ) (h1 : given = 6) (h2 : left = 45) (h3 : e = given + left) : e = 51 := by
  sorry

end eugene_initial_pencils_l94_94502


namespace sum_of_squares_ge_sum_of_products_l94_94043

theorem sum_of_squares_ge_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end sum_of_squares_ge_sum_of_products_l94_94043


namespace smallest_constant_N_l94_94328

theorem smallest_constant_N (a : ℝ) (ha : a > 0) : 
  let b := a
  let c := a
  (a = b ∧ b = c) → (a^2 + b^2 + c^2) / (a + b + c) > (0 : ℝ) := 
by
  -- Assuming the proof steps are written here
  sorry

end smallest_constant_N_l94_94328


namespace range_of_a_l94_94233

noncomputable def p (a : ℝ) : Prop :=
∀ (x : ℝ), x > -1 → (x^2) / (x + 1) ≥ a

noncomputable def q (a : ℝ) : Prop :=
∃ (x : ℝ), (a*x^2 - a*x + 1 = 0)

theorem range_of_a (a : ℝ) :
  ¬ p a ∧ ¬ q a ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
by sorry

end range_of_a_l94_94233


namespace sum_of_values_for_one_solution_l94_94297

noncomputable def sum_of_a_values (a1 a2 : ℝ) : ℝ :=
  a1 + a2

theorem sum_of_values_for_one_solution :
  ∃ a1 a2 : ℝ, 
  (∀ x : ℝ, 4 * x^2 + (a1 + 8) * x + 9 = 0 ∨ 4 * x^2 + (a2 + 8) * x + 9 = 0) ∧
  ((a1 + 8)^2 - 144 = 0) ∧ ((a2 + 8)^2 - 144 = 0) ∧
  sum_of_a_values a1 a2 = -16 :=
by
  sorry

end sum_of_values_for_one_solution_l94_94297


namespace greatest_four_digit_divisible_by_conditions_l94_94794

-- Definitions based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

-- Problem statement: Finding the greatest 4-digit number divisible by 15, 25, 40, and 75
theorem greatest_four_digit_divisible_by_conditions :
  ∃ n, is_four_digit n ∧ is_divisible_by n 15 ∧ is_divisible_by n 25 ∧ is_divisible_by n 40 ∧ is_divisible_by n 75 ∧ n = 9600 :=
  sorry

end greatest_four_digit_divisible_by_conditions_l94_94794


namespace rabbit_count_l94_94816

-- Define the conditions
def original_rabbits : ℕ := 8
def new_rabbits_born : ℕ := 5

-- Define the total rabbits based on the conditions
def total_rabbits : ℕ := original_rabbits + new_rabbits_born

-- The statement to prove that the total number of rabbits is 13
theorem rabbit_count : total_rabbits = 13 :=
by
  -- Proof not needed, hence using sorry
  sorry

end rabbit_count_l94_94816


namespace perimeter_F_is_18_l94_94970

-- Define the dimensions of the rectangles.
def vertical_rectangle : ℤ × ℤ := (3, 5)
def horizontal_rectangle : ℤ × ℤ := (1, 5)

-- Define the perimeter calculation for a single rectangle.
def perimeter (width_height : ℤ × ℤ) : ℤ :=
  2 * width_height.1 + 2 * width_height.2

-- The overlapping width and height.
def overlap_width : ℤ := 5
def overlap_height : ℤ := 1

-- Perimeter of the letter F.
def perimeter_F : ℤ :=
  perimeter vertical_rectangle + perimeter horizontal_rectangle - 2 * overlap_width

-- Statement to prove.
theorem perimeter_F_is_18 : perimeter_F = 18 := by sorry

end perimeter_F_is_18_l94_94970


namespace total_volume_of_all_cubes_l94_94359

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (count : ℕ) (side_length : ℕ) : ℕ := count * (cube_volume side_length)

theorem total_volume_of_all_cubes :
  total_volume 4 3 + total_volume 3 4 = 300 :=
by
  sorry

end total_volume_of_all_cubes_l94_94359


namespace theta_value_l94_94739

theorem theta_value (Theta : ℕ) (h_digit : Θ < 10) (h_eq : 252 / Θ = 30 + 2 * Θ) : Θ = 6 := 
by
  sorry

end theta_value_l94_94739


namespace solve_for_x_l94_94682

-- We define that the condition and what we need to prove.
theorem solve_for_x (x : ℝ) : (x + 7) / (x - 4) = (x - 3) / (x + 6) → x = -3 / 2 :=
by sorry

end solve_for_x_l94_94682


namespace age_difference_of_declans_sons_l94_94019

theorem age_difference_of_declans_sons 
  (current_age_elder_son : ℕ) 
  (future_age_younger_son : ℕ) 
  (years_until_future : ℕ) 
  (current_age_elder_son_eq : current_age_elder_son = 40) 
  (future_age_younger_son_eq : future_age_younger_son = 60) 
  (years_until_future_eq : years_until_future = 30) :
  (current_age_elder_son - (future_age_younger_son - years_until_future)) = 10 := by
  sorry

end age_difference_of_declans_sons_l94_94019


namespace relationship_among_a_b_and_ab_l94_94787

noncomputable def a : ℝ := Real.log 0.4 / Real.log 0.2
noncomputable def b : ℝ := 1 - (1 / (Real.log 4 / Real.log 10))

theorem relationship_among_a_b_and_ab : a * b < a + b ∧ a + b < 0 := by
  sorry

end relationship_among_a_b_and_ab_l94_94787


namespace total_balloons_is_18_l94_94272

-- Define the number of balloons each person has
def Fred_balloons : Nat := 5
def Sam_balloons : Nat := 6
def Mary_balloons : Nat := 7

-- Define the total number of balloons
def total_balloons : Nat := Fred_balloons + Sam_balloons + Mary_balloons

-- The theorem statement to prove
theorem total_balloons_is_18 : total_balloons = 18 := sorry

end total_balloons_is_18_l94_94272


namespace fence_pole_count_l94_94795

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l94_94795


namespace inflated_cost_per_person_l94_94296

def estimated_cost : ℝ := 30e9
def people_sharing : ℝ := 200e6
def inflation_rate : ℝ := 0.05

theorem inflated_cost_per_person :
  (estimated_cost * (1 + inflation_rate)) / people_sharing = 157.5 := by
  sorry

end inflated_cost_per_person_l94_94296


namespace last_rope_length_l94_94594

def totalRopeLength : ℝ := 35
def rope1 : ℝ := 8
def rope2 : ℝ := 20
def rope3a : ℝ := 2
def rope3b : ℝ := 2
def rope3c : ℝ := 2
def knotLoss : ℝ := 1.2
def numKnots : ℝ := 4

theorem last_rope_length : 
  (35 + (4 * 1.2)) = (8 + 20 + 2 + 2 + 2 + x) → (x = 5.8) :=
sorry

end last_rope_length_l94_94594


namespace product_not_ending_in_1_l94_94883

theorem product_not_ending_in_1 : ∃ a b : ℕ, 111111 = a * b ∧ (a % 10 ≠ 1) ∧ (b % 10 ≠ 1) := 
sorry

end product_not_ending_in_1_l94_94883


namespace factor_theorem_l94_94235

noncomputable def polynomial_to_factor : Prop :=
  ∀ x : ℝ, x^4 - 4 * x^2 + 4 = (x^2 - 2)^2

theorem factor_theorem : polynomial_to_factor :=
by
  sorry

end factor_theorem_l94_94235


namespace factors_are_divisors_l94_94091

theorem factors_are_divisors (a b c d : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : d = 5) : 
  a ∣ 30 ∧ b ∣ 30 ∧ c ∣ 30 ∧ d ∣ 30 :=
by
  sorry

end factors_are_divisors_l94_94091


namespace daniel_original_noodles_l94_94230

-- Define the total number of noodles Daniel had originally
def original_noodles : ℕ := 81

-- Define the remaining noodles after giving 1/3 to William
def remaining_noodles (n : ℕ) : ℕ := (2 * n) / 3

-- State the theorem
theorem daniel_original_noodles (n : ℕ) (h : remaining_noodles n = 54) : n = original_noodles := by sorry

end daniel_original_noodles_l94_94230


namespace isosceles_obtuse_triangle_smallest_angle_l94_94800

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β γ : ℝ), α = 1.8 * 90 ∧ β = γ ∧ α + β + γ = 180 → β = 9 :=
by
  intros α β γ h
  sorry

end isosceles_obtuse_triangle_smallest_angle_l94_94800


namespace cylinder_radius_original_l94_94610

theorem cylinder_radius_original (r : ℝ) (h : ℝ) (h_given : h = 4) 
    (V_increase_radius : π * (r + 4) ^ 2 * h = π * r ^ 2 * (h + 4)) : 
    r = 12 := 
  by
    sorry

end cylinder_radius_original_l94_94610


namespace area_of_remaining_shape_l94_94963

/-- Define the initial 6x6 square grid with each cell of size 1 cm. -/
def initial_square_area : ℝ := 6 * 6

/-- Define the area of the combined dark gray triangles forming a 1x3 rectangle. -/
def dark_gray_area : ℝ := 1 * 3

/-- Define the area of the combined light gray triangles forming a 2x3 rectangle. -/
def light_gray_area : ℝ := 2 * 3

/-- Define the total area of the gray triangles cut out. -/
def total_gray_area : ℝ := dark_gray_area + light_gray_area

/-- Calculate the area of the remaining figure after cutting out the gray triangles. -/
def remaining_area : ℝ := initial_square_area - total_gray_area

/-- Proof that the area of the remaining shape is 27 square centimeters. -/
theorem area_of_remaining_shape : remaining_area = 27 := by
  sorry

end area_of_remaining_shape_l94_94963


namespace mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l94_94309

-- Definition of Mr. Wang's movements
def movements : List Int := [6, -3, 10, -8, 12, -7, -10]

-- Definitions of given conditions
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.3

theorem mr_wang_returned_to_1st_floor :
  (List.sum movements = 0) :=
by
  sorry

theorem mr_wang_electricity_consumption :
  (List.sum (movements.map Int.natAbs) * floor_height * electricity_per_meter = 50.4) :=
by
  sorry

end mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l94_94309


namespace original_price_of_table_l94_94360

noncomputable def original_price (sale_price : ℝ) (discount_rate : ℝ) : ℝ :=
  sale_price / (1 - discount_rate)

theorem original_price_of_table
  (d : ℝ) (p' : ℝ) (h_d : d = 0.10) (h_p' : p' = 450) :
  original_price p' d = 500 := by
  rw [h_d, h_p']
  -- Calculating the original price
  show original_price 450 0.10 = 500
  sorry

end original_price_of_table_l94_94360


namespace find_c_plus_1_over_b_l94_94956

theorem find_c_plus_1_over_b (a b c : ℝ) (h1: a * b * c = 1) 
    (h2: a + 1 / c = 7) (h3: b + 1 / a = 12) : c + 1 / b = 21 / 83 := 
by 
    sorry

end find_c_plus_1_over_b_l94_94956


namespace solve_for_n_l94_94619

theorem solve_for_n (n : ℚ) (h : (1 / (n + 2)) + (2 / (n + 2)) + (n / (n + 2)) = 3) : n = -3/2 := 
by
  sorry

end solve_for_n_l94_94619


namespace largest_possible_green_cards_l94_94573

-- Definitions of conditions
variables (g y t : ℕ)

-- Defining the total number of cards t
def total_cards := g + y

-- Condition on maximum number of cards
def max_total_cards := total_cards g y ≤ 2209

-- Probability condition for drawing 3 same-color cards
def probability_condition := 
  g * (g - 1) * (g - 2) + y * (y - 1) * (y - 2) 
  = (1 : ℚ) / 3 * t * (t - 1) * (t - 2)

-- Proving the largest possible number of green cards
theorem largest_possible_green_cards
  (h1 : total_cards g y = t)
  (h2 : max_total_cards g y)
  (h3 : probability_condition g y t) :
  g ≤ 1092 :=
sorry

end largest_possible_green_cards_l94_94573


namespace positive_diff_between_two_numbers_l94_94413

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end positive_diff_between_two_numbers_l94_94413


namespace area_ratio_proof_l94_94579

variables (BE CE DE AE : ℝ)
variables (S_alpha S_beta S_gamma S_delta : ℝ)
variables (x : ℝ)

-- Definitions for the given conditions
def BE_val := 80
def CE_val := 60
def DE_val := 40
def AE_val := 30

-- Expressing the ratios
def S_alpha_ratio := 2
def S_beta_ratio := 2

-- Assuming areas in terms of x
def S_alpha_val := 2 * x
def S_beta_val := 2 * x
def S_delta_val := x
def S_gamma_val := 2 * x

-- Problem statement
theorem area_ratio_proof
  (BE := BE_val)
  (CE := CE_val)
  (DE := DE_val)
  (AE := AE_val)
  (S_alpha := S_alpha_val)
  (S_beta := S_beta_val)
  (S_gamma := S_gamma_val)
  (S_delta := S_delta_val) :
  (S_gamma + S_delta) / (S_alpha + S_beta) = 5 / 4 :=
by
  sorry

end area_ratio_proof_l94_94579


namespace smallest_b_base_45b_perfect_square_l94_94976

theorem smallest_b_base_45b_perfect_square : ∃ b : ℕ, b > 3 ∧ (∃ n : ℕ, n^2 = 4 * b + 5) ∧ ∀ b' : ℕ, b' > 3 ∧ (∃ n' : ℕ, n'^2 = 4 * b' + 5) → b ≤ b' := 
sorry

end smallest_b_base_45b_perfect_square_l94_94976


namespace power_of_negative_fraction_l94_94263

theorem power_of_negative_fraction :
  (- (1/3))^2 = 1/9 := 
by 
  sorry

end power_of_negative_fraction_l94_94263


namespace solve_arithmetic_sequence_problem_l94_94104

noncomputable def arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) : Prop :=
  (∀ n, a n = a 0 + n * (a 1 - a 0)) ∧  -- Condition: sequence is arithmetic
  (∀ n, S n = (n * (a 0 + a (n - 1))) / 2) ∧  -- Condition: sum of first n terms
  (m > 1) ∧  -- Condition: m > 1
  (a (m - 1) + a (m + 1) - a m ^ 2 = 0) ∧  -- Given condition
  (S (2 * m - 1) = 38)  -- Given that sum of first 2m-1 terms equals 38

-- The statement we need to prove
theorem solve_arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) :
  arithmetic_sequence_problem a S m → m = 10 :=
by
  sorry  -- Proof to be completed

end solve_arithmetic_sequence_problem_l94_94104


namespace avg_of_8_numbers_l94_94279

theorem avg_of_8_numbers
  (n : ℕ)
  (h₁ : n = 8)
  (sum_first_half : ℝ)
  (h₂ : sum_first_half = 158.4)
  (avg_second_half : ℝ)
  (h₃ : avg_second_half = 46.6) :
  ((sum_first_half + avg_second_half * (n / 2)) / n) = 43.1 :=
by
  sorry

end avg_of_8_numbers_l94_94279


namespace jericho_money_left_l94_94512

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end jericho_money_left_l94_94512


namespace find_factor_l94_94600

theorem find_factor (x : ℕ) (f : ℕ) (h1 : x = 9)
  (h2 : (2 * x + 6) * f = 72) : f = 3 := by
  sorry

end find_factor_l94_94600


namespace monotonicity_intervals_number_of_zeros_l94_94920

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

theorem monotonicity_intervals (k : ℝ) :
  (k ≤ 0 → (∀ x, x < 0 → f k x < 0) ∧ (∀ x, x ≥ 0 → f k x > 0)) ∧
  (0 < k ∧ k < 1 → 
    (∀ x, x < Real.log k → f k x < 0) ∧ (∀ x, x ≥ Real.log k ∧ x < 0 → f k x > 0) ∧ 
    (∀ x, x > 0 → f k x > 0)) ∧
  (k = 1 → ∀ x, f k x > 0) ∧
  (k > 1 → 
    (∀ x, x < 0 → f k x < 0) ∧ 
    (∀ x, x ≥ 0 ∧ x < Real.log k → f k x > 0) ∧ 
    (∀ x, x > Real.log k → f k x > 0)) :=
sorry

theorem number_of_zeros (k : ℝ) (h_nonpos : k ≤ 0) :
  (k < 0 → (∃ a b : ℝ, a < 0 ∧ b > 0 ∧ f k a = 0 ∧ f k b = 0)) ∧
  (k = 0 → f k 1 = 0 ∧ (∀ x, x ≠ 1 → f k x ≠ 0)) :=
sorry

end monotonicity_intervals_number_of_zeros_l94_94920


namespace chocolate_cost_proof_l94_94391

/-- The initial amount of money Dan has. -/
def initial_amount : ℕ := 7

/-- The cost of the candy bar. -/
def candy_bar_cost : ℕ := 2

/-- The remaining amount of money Dan has after the purchases. -/
def remaining_amount : ℕ := 2

/-- The cost of the chocolate. -/
def chocolate_cost : ℕ := initial_amount - candy_bar_cost - remaining_amount

/-- Expected cost of the chocolate. -/
def expected_chocolate_cost : ℕ := 3

/-- Prove that the cost of the chocolate equals the expected cost. -/
theorem chocolate_cost_proof : chocolate_cost = expected_chocolate_cost :=
by
  sorry

end chocolate_cost_proof_l94_94391


namespace smallest_n_for_congruence_l94_94068

theorem smallest_n_for_congruence :
  ∃ n : ℕ, 827 * n % 36 = 1369 * n % 36 ∧ n > 0 ∧ (∀ m : ℕ, 827 * m % 36 = 1369 * m % 36 ∧ m > 0 → m ≥ 18) :=
by sorry

end smallest_n_for_congruence_l94_94068


namespace sqrt_range_l94_94323

theorem sqrt_range (x : ℝ) (hx : 0 ≤ x - 1) : 1 ≤ x :=
by sorry

end sqrt_range_l94_94323


namespace min_x2_y2_l94_94681

theorem min_x2_y2 (x y : ℝ) (h : x * y - x - y = 1) : x^2 + y^2 ≥ 6 - 4 * Real.sqrt 2 :=
by
  sorry

end min_x2_y2_l94_94681


namespace quadratic_monotonic_range_l94_94356

theorem quadratic_monotonic_range {a : ℝ} :
  (∀ x1 x2 : ℝ, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → (x1^2 - 2*a*x1 + 1) ≤ (x2^2 - 2*a*x2 + 1) ∨ (x1^2 - 2*a*x1 + 1) ≥ (x2^2 - 2*a*x2 + 1)) → (a ≤ 2 ∨ a ≥ 3) := 
sorry

end quadratic_monotonic_range_l94_94356


namespace min_value_of_function_l94_94940

noncomputable def y (θ : ℝ) : ℝ := (2 - Real.sin θ) / (1 - Real.cos θ)

theorem min_value_of_function : ∃ θ : ℝ, y θ = 3 / 4 :=
sorry

end min_value_of_function_l94_94940


namespace max_area_of_triangle_ABC_l94_94449

noncomputable def max_triangle_area (a b c : ℝ) (A B C : ℝ) := 
  1 / 2 * b * c * Real.sin A

theorem max_area_of_triangle_ABC :
  ∀ (a b c A B C : ℝ)
  (ha : a = 2)
  (hTrig : a = Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A))
  (hCondition: 3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0),
  max_triangle_area a b c A B C ≤ 2 := 
by
  intros a b c A B C ha hTrig hCondition
  sorry

end max_area_of_triangle_ABC_l94_94449


namespace winner_beats_by_16_secons_l94_94053

-- Definitions of the times for mathematician and physicist
variables (x y : ℕ)

-- Conditions based on the given problem
def condition1 := 2 * y - x = 24
def condition2 := 2 * x - y = 72

-- The statement to prove
theorem winner_beats_by_16_secons (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x - 2 * y = 16 := 
sorry

end winner_beats_by_16_secons_l94_94053


namespace distance_to_city_hall_l94_94984

variable (d : ℝ) (t : ℝ)

-- Conditions
def condition1 : Prop := d = 45 * (t + 1.5)
def condition2 : Prop := d - 45 = 65 * (t - 1.25)
def condition3 : Prop := t > 0

theorem distance_to_city_hall
  (h1 : condition1 d t)
  (h2 : condition2 d t)
  (h3 : condition3 t)
  : d = 300 := by
  sorry

end distance_to_city_hall_l94_94984


namespace find_h_l94_94999

theorem find_h (h j k : ℤ) (y_intercept1 : 3 * h ^ 2 + j = 2013) 
  (y_intercept2 : 2 * h ^ 2 + k = 2014)
  (x_intercepts1 : ∃ (y : ℤ), j = -3 * y ^ 2)
  (x_intercepts2 : ∃ (x : ℤ), k = -2 * x ^ 2) :
  h = 36 :=
by sorry

end find_h_l94_94999


namespace combined_time_alligators_walked_l94_94034

-- Define the conditions
def original_time : ℕ := 4
def return_time := original_time + 2 * Int.sqrt original_time

-- State the theorem to be proven
theorem combined_time_alligators_walked : original_time + return_time = 12 := by
  sorry

end combined_time_alligators_walked_l94_94034


namespace monotonic_increasing_range_l94_94375

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, (3*x^2 + 2*x - a) ≥ 0) ↔ (a ≤ -1/3) :=
by
  sorry

end monotonic_increasing_range_l94_94375


namespace secretaries_ratio_l94_94418

theorem secretaries_ratio (A B C : ℝ) (hA: A = 75) (h_total: A + B + C = 120) : B + C = 45 :=
by {
  -- sorry: We define this part to be explored by the theorem prover
  sorry
}

end secretaries_ratio_l94_94418


namespace max_value_of_f_l94_94745

noncomputable def f (x : Real) := 2 * (Real.sin x) ^ 2 - (Real.tan x) ^ 2

theorem max_value_of_f : 
  ∃ (x : Real), f x = 3 - 2 * Real.sqrt 2 := 
sorry

end max_value_of_f_l94_94745


namespace stanleyRanMore_l94_94291

def distanceStanleyRan : ℝ := 0.4
def distanceStanleyWalked : ℝ := 0.2

theorem stanleyRanMore : distanceStanleyRan - distanceStanleyWalked = 0.2 := by
  sorry

end stanleyRanMore_l94_94291


namespace runner_overtake_time_l94_94851

theorem runner_overtake_time
  (L : ℝ)
  (v1 v2 v3 : ℝ)
  (h1 : v1 = v2 + L / 6)
  (h2 : v1 = v3 + L / 10) :
  L / (v3 - v2) = 15 := by
  sorry

end runner_overtake_time_l94_94851


namespace fishAddedIs15_l94_94824

-- Define the number of fish Jason starts with
def initialNumberOfFish : ℕ := 6

-- Define the fish counts on each day
def fishOnDay2 := 2 * initialNumberOfFish
def fishOnDay3 := 2 * fishOnDay2 - (1 / 3 : ℚ) * (2 * fishOnDay2)
def fishOnDay4 := 2 * fishOnDay3
def fishOnDay5 := 2 * fishOnDay4 - (1 / 4 : ℚ) * (2 * fishOnDay4)
def fishOnDay6 := 2 * fishOnDay5
def fishOnDay7 := 2 * fishOnDay6

-- Define the total fish on the seventh day after adding some fish
def totalFishOnDay7 := 207

-- Define the number of fish Jason added on the seventh day
def fishAddedOnDay7 := totalFishOnDay7 - fishOnDay7

-- Prove that the number of fish Jason added on the seventh day is 15
theorem fishAddedIs15 : fishAddedOnDay7 = 15 := sorry

end fishAddedIs15_l94_94824


namespace exists_xyz_t_l94_94055

theorem exists_xyz_t (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0) (h5 : x + y + z + t = 15) : ∃ y, y = 12 :=
by
  sorry

end exists_xyz_t_l94_94055


namespace arithmetic_sequence_common_difference_l94_94197

theorem arithmetic_sequence_common_difference {a : ℕ → ℝ} (h₁ : a 1 = 2) (h₂ : a 2 + a 4 = a 6) : ∃ d : ℝ, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l94_94197


namespace percentage_failed_both_l94_94627

theorem percentage_failed_both 
    (p_h p_e p_p p_pe : ℝ)
    (h_p_h : p_h = 32)
    (h_p_e : p_e = 56)
    (h_p_p : p_p = 24)
    : p_pe = 12 := by 
    sorry

end percentage_failed_both_l94_94627


namespace cos_triple_angle_l94_94537

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l94_94537


namespace fibonacci_eighth_term_l94_94850

theorem fibonacci_eighth_term
  (F : ℕ → ℕ)
  (h1 : F 9 = 34)
  (h2 : F 10 = 55)
  (fib : ∀ n, F (n + 2) = F (n + 1) + F n) :
  F 8 = 21 :=
by
  sorry

end fibonacci_eighth_term_l94_94850


namespace arithmetic_seq_common_diff_l94_94498

theorem arithmetic_seq_common_diff (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 0 + a 2 = 10) 
  (h2 : a 3 + a 5 = 4)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = -1 := 
  sorry

end arithmetic_seq_common_diff_l94_94498


namespace solution_set_inequality_l94_94980

theorem solution_set_inequality (x : ℝ) : (x^2 - 2*x - 8 ≥ 0) ↔ (x ≤ -2 ∨ x ≥ 4) := 
sorry

end solution_set_inequality_l94_94980


namespace find_required_water_amount_l94_94606

-- Definitions based on the conditions
def sanitizer_volume : ℝ := 12
def initial_alcohol_concentration : ℝ := 0.60
def desired_alcohol_concentration : ℝ := 0.40

-- Statement of the proof problem
theorem find_required_water_amount : 
  ∃ (x : ℝ), x = 6 ∧ sanitizer_volume * initial_alcohol_concentration = desired_alcohol_concentration * (sanitizer_volume + x) :=
sorry

end find_required_water_amount_l94_94606


namespace initial_puppies_l94_94331

-- Define the conditions
variable (a : ℕ) (t : ℕ) (p_added : ℕ) (p_total_adopted : ℕ)

-- State the theorem with the conditions and the target proof
theorem initial_puppies
  (h₁ : a = 3) 
  (h₂ : t = 2)
  (h₃ : p_added = 3)
  (h₄ : p_total_adopted = a * t) :
  (p_total_adopted - p_added) = 3 :=
sorry

end initial_puppies_l94_94331


namespace proof_problem_l94_94099

variable (x y : ℝ)

theorem proof_problem 
  (h1 : 0.30 * x = 0.40 * 150 + 90)
  (h2 : 0.20 * x = 0.50 * 180 - 60)
  (h3 : y = 0.75 * x)
  (h4 : y^2 > x + 100) :
  x = 150 ∧ y = 112.5 :=
by
  sorry

end proof_problem_l94_94099


namespace remainder_div_180_l94_94137

theorem remainder_div_180 {j : ℕ} (h1 : 0 < j) (h2 : 120 % (j^2) = 12) : 180 % j = 0 :=
by
  sorry

end remainder_div_180_l94_94137


namespace bus_costs_unique_min_buses_cost_A_l94_94761

-- Defining the main conditions
def condition1 (x y : ℕ) : Prop := x + 2 * y = 300
def condition2 (x y : ℕ) : Prop := 2 * x + y = 270

-- Part 1: Proving individual bus costs
theorem bus_costs_unique (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 110 := 
by 
  sorry

-- Part 2: Minimum buses of type A and total cost constraint
def total_buses := 10
def total_cost (x y a : ℕ) : Prop := 
  x * a + y * (total_buses - a) ≤ 1000

theorem min_buses_cost_A (x y : ℕ) (hx : x = 80) (hy : y = 110) :
  ∃ a cost, total_cost x y a ∧ a >= 4 ∧ cost = x * 4 + y * (total_buses - 4) ∧ cost = 980 :=
by
  sorry

end bus_costs_unique_min_buses_cost_A_l94_94761


namespace inscribed_circle_radius_eq_l94_94982

noncomputable def inscribedCircleRadius :=
  let AB := 6
  let AC := 7
  let BC := 8
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r := K / s
  r

theorem inscribed_circle_radius_eq :
  inscribedCircleRadius = Real.sqrt 413.4375 / 10.5 := by
  sorry

end inscribed_circle_radius_eq_l94_94982


namespace opposite_of_a_is_2022_l94_94120

theorem opposite_of_a_is_2022 (a : Int) (h : -a = -2022) : a = 2022 := by
  sorry

end opposite_of_a_is_2022_l94_94120


namespace age_ratio_in_future_l94_94284

variables (t j x : ℕ)

theorem age_ratio_in_future:
  (t - 4 = 5 * (j - 4)) → 
  (t - 10 = 6 * (j - 10)) →
  (t + x = 3 * (j + x)) →
  x = 26 := 
by {
  sorry
}

end age_ratio_in_future_l94_94284


namespace cricket_jumps_to_100m_l94_94868

theorem cricket_jumps_to_100m (x y : ℕ) (h : 9 * x + 8 * y = 100) : x + y = 12 :=
sorry

end cricket_jumps_to_100m_l94_94868


namespace no_pieces_left_impossible_l94_94669

/-- Starting with 100 pieces and 1 pile, and given the ability to either:
1. Remove one piece from a pile of at least 3 pieces and divide the remaining pile into two non-empty piles,
2. Eliminate a pile containing a single piece,
prove that it is impossible to reach a situation with no pieces left. -/
theorem no_pieces_left_impossible :
  ∀ (p t : ℕ), p = 100 → t = 1 →
  (∀ (p' t' : ℕ),
    (p' = p - 1 ∧ t' = t + 1 ∧ 3 ≤ p) ∨
    (p' = p - 1 ∧ t' = t - 1 ∧ ∃ k, k = 1 ∧ t ≠ 0) →
    false) :=
by
  intros
  sorry

end no_pieces_left_impossible_l94_94669


namespace xiao_wang_exam_grades_l94_94856

theorem xiao_wang_exam_grades 
  (x y : ℕ) 
  (h1 : (x * y + 98) / (x + 1) = y + 1)
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  x + 2 = 10 ∧ y - 1 = 88 := 
by
  sorry

end xiao_wang_exam_grades_l94_94856


namespace number_of_groups_of_three_books_l94_94152

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l94_94152


namespace sufficient_condition_abs_sum_gt_one_l94_94591

theorem sufficient_condition_abs_sum_gt_one (x y : ℝ) (h : y ≤ -2) : |x| + |y| > 1 :=
  sorry

end sufficient_condition_abs_sum_gt_one_l94_94591


namespace percentage_increase_weekends_l94_94451

def weekday_price : ℝ := 18
def weekend_price : ℝ := 27

theorem percentage_increase_weekends : 
  (weekend_price - weekday_price) / weekday_price * 100 = 50 := by
  sorry

end percentage_increase_weekends_l94_94451


namespace smallest_value_of_x_l94_94028

theorem smallest_value_of_x :
  ∀ x : ℚ, ( ( (5 * x - 20) / (4 * x - 5) ) ^ 3
           + ( (5 * x - 20) / (4 * x - 5) ) ^ 2
           - ( (5 * x - 20) / (4 * x - 5) )
           - 15 = 0 ) → x = 10 / 3 :=
by
  sorry

end smallest_value_of_x_l94_94028


namespace find_number_l94_94726

-- Define the number x that satisfies the given condition
theorem find_number (x : ℤ) (h : x + 12 - 27 = 24) : x = 39 :=
by {
  -- This is where the proof steps will go, but we'll use sorry to indicate it's incomplete
  sorry
}

end find_number_l94_94726


namespace find_base_l94_94310

def distinct_three_digit_numbers (b : ℕ) : ℕ :=
    (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)

theorem find_base (b : ℕ) (h : distinct_three_digit_numbers b = 144) : b = 9 :=
by 
  sorry

end find_base_l94_94310


namespace minimum_value_of_expr_l94_94186

noncomputable def expr (x y : ℝ) : ℝ := 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4

theorem minimum_value_of_expr : ∃ x y : ℝ, expr x y = -1 ∧ ∀ (a b : ℝ), expr a b ≥ -1 := 
by
  sorry

end minimum_value_of_expr_l94_94186


namespace natural_number_base_conversion_l94_94703

theorem natural_number_base_conversion (n : ℕ) (h7 : n = 4 * 7 + 1) (h9 : n = 3 * 9 + 2) : 
  n = 3 * 8 + 5 := 
by 
  sorry

end natural_number_base_conversion_l94_94703


namespace binomial_expansion_l94_94085

theorem binomial_expansion (a b : ℕ) (h_a : a = 34) (h_b : b = 5) :
  a^2 + 2*a*b + b^2 = 1521 :=
by
  rw [h_a, h_b]
  sorry

end binomial_expansion_l94_94085


namespace circle_tangent_l94_94500

theorem circle_tangent {m : ℝ} (h : ∃ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 25 - m ∧ x^2 + y^2 = 1) :
  m = 9 :=
sorry

end circle_tangent_l94_94500


namespace painter_completes_at_9pm_l94_94772

noncomputable def mural_completion_time (start_time : Nat) (fraction_completed_time : Nat)
    (fraction_completed : ℚ) : Nat :=
  let fraction_per_hour := fraction_completed / fraction_completed_time
  start_time + Nat.ceil (1 / fraction_per_hour)

theorem painter_completes_at_9pm :
  mural_completion_time 9 3 (1/4) = 21 := by
  sorry

end painter_completes_at_9pm_l94_94772


namespace soccer_team_points_l94_94957

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end soccer_team_points_l94_94957


namespace binary_rep_of_21_l94_94050

theorem binary_rep_of_21 : 
  (Nat.digits 2 21) = [1, 0, 1, 0, 1] := 
by 
  sorry

end binary_rep_of_21_l94_94050


namespace ratio_of_socks_l94_94188

variable (b : ℕ)            -- the number of pairs of blue socks
variable (x : ℝ)            -- the price of blue socks per pair

def original_cost : ℝ := 5 * 3 * x + b * x
def interchanged_cost : ℝ := b * 3 * x + 5 * x

theorem ratio_of_socks :
  (5 : ℝ) / b = 5 / 14 :=
by
  sorry

end ratio_of_socks_l94_94188


namespace proof_l94_94488

variable (p : ℕ) (ε : ℤ)
variable (RR NN NR RN : ℕ)

-- Conditions
axiom h1 : ∀ n ≤ p - 2, 
  (n % 2 = 0 ∧ (n + 1) % 2 = 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 ≠ 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 = 0 ) ∨ 
  (n % 2 = 0 ∧ (n + 1) % 2 ≠ 0) 

axiom h2 :  RR + NN - RN - NR = 1

axiom h3 : ε = (-1) ^ ((p - 1) / 2)

axiom h4 : RR + RN = (p - 2 - ε) / 2

axiom h5 : RR + NR = (p - 1) / 2 - 1

axiom h6 : NR + NN = (p - 2 + ε) / 2

axiom h7 : RN + NN = (p - 1) / 2  

-- To prove
theorem proof : 
  RR = (p / 4) - (ε + 4) / 4 ∧ 
  RN = (p / 4) - (ε) / 4 ∧ 
  NN = (p / 4) + (ε - 2) / 4 ∧ 
  NR = (p / 4) + (ε - 2) / 4 := 
sorry

end proof_l94_94488


namespace chuck_team_score_proof_chuck_team_score_l94_94952

-- Define the conditions
def yellow_team_score : ℕ := 55
def lead : ℕ := 17

-- State the main proposition
theorem chuck_team_score (yellow_team_score : ℕ) (lead : ℕ) : ℕ :=
yellow_team_score + lead

-- Formulate the final proof goal
theorem proof_chuck_team_score : chuck_team_score yellow_team_score lead = 72 :=
by {
  -- This is the place where the proof should go
  sorry
}

end chuck_team_score_proof_chuck_team_score_l94_94952


namespace michael_water_left_l94_94855

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end michael_water_left_l94_94855


namespace difference_of_two_numbers_l94_94567

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 15) (h2 : x^2 - y^2 = 150) : x - y = 10 :=
by
  sorry

end difference_of_two_numbers_l94_94567


namespace max_min_y_l94_94225

def g (t : ℝ) : ℝ := 80 - 2 * t

def f (t : ℝ) : ℝ := 20 - |t - 10|

def y (t : ℝ) : ℝ := g t * f t

theorem max_min_y (t : ℝ) (h : 0 ≤ t ∧ t ≤ 20) :
  (y t = 1200 → t = 10) ∧ (y t = 400 → t = 20) :=
by
  sorry

end max_min_y_l94_94225


namespace intersection_A_B_l94_94322

-- Definitions of the sets A and B
def set_A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 7 }

-- Theorem statement to prove the intersection
theorem intersection_A_B : set_A ∩ set_B = { x | 3 ≤ x ∧ x < 7 } := by
  sorry

end intersection_A_B_l94_94322


namespace probability_one_painted_face_and_none_painted_l94_94482

-- Define the total number of smaller unit cubes
def total_cubes : ℕ := 125

-- Define the number of cubes with exactly one painted face
def one_painted_face : ℕ := 25

-- Define the number of cubes with no painted faces
def no_painted_faces : ℕ := 125 - 25 - 12

-- Define the total number of ways to select two cubes uniformly at random
def total_pairs : ℕ := (total_cubes * (total_cubes - 1)) / 2

-- Define the number of successful outcomes
def successful_outcomes : ℕ := one_painted_face * no_painted_faces

-- Define the sought probability
def desired_probability : ℚ := (successful_outcomes : ℚ) / (total_pairs : ℚ)

-- Lean statement to prove the probability
theorem probability_one_painted_face_and_none_painted :
  desired_probability = 44 / 155 :=
by
  sorry

end probability_one_painted_face_and_none_painted_l94_94482


namespace age_problem_l94_94027

variables (K M A B : ℕ)

theorem age_problem
  (h1 : K + 7 = 3 * M)
  (h2 : M = 5)
  (h3 : A + B = 2 * M + 4)
  (h4 : A = B - 3)
  (h5 : K + B = M + 9) :
  K = 8 ∧ M = 5 ∧ B = 6 ∧ A = 3 :=
sorry

end age_problem_l94_94027


namespace project_completion_days_l94_94741

-- A's work rate per day
def A_work_rate : ℚ := 1 / 20

-- B's work rate per day
def B_work_rate : ℚ := 1 / 30

-- Combined work rate per day
def combined_work_rate : ℚ := A_work_rate + B_work_rate

-- Work done by B alone in the last 5 days
def B_alone_work : ℚ := 5 * B_work_rate

-- Let variable x represent the number of days A and B work together
def x (x_days : ℚ) := x_days / combined_work_rate + B_alone_work = 1

theorem project_completion_days (x_days : ℚ) (total_days : ℚ) :
  A_work_rate = 1 / 20 → B_work_rate = 1 / 30 → combined_work_rate = 1 / 12 → x_days / 12 + 1 / 6 = 1 → x_days = 10 → total_days = x_days + 5 → total_days = 15 :=
by
  intros _ _ _ _ _ _
  sorry

end project_completion_days_l94_94741


namespace correct_answer_l94_94170

noncomputable def original_number (y : ℝ) :=
  (y - 14) / 2 = 50

theorem correct_answer (y : ℝ) (h : original_number y) :
  (y - 5) / 7 = 15 :=
by
  sorry

end correct_answer_l94_94170


namespace manny_original_marbles_l94_94835

/-- 
Let total marbles be 120, and the marbles are divided between Mario, Manny, and Mike in the ratio 4:5:6. 
Let x be the number of marbles Manny is left with after giving some marbles to his brother.
Prove that Manny originally had 40 marbles. 
-/
theorem manny_original_marbles (total_marbles : ℕ) (ratio_mario ratio_manny ratio_mike : ℕ)
    (present_marbles : ℕ) (total_parts : ℕ)
    (h_marbles : total_marbles = 120) 
    (h_ratio : ratio_mario = 4 ∧ ratio_manny = 5 ∧ ratio_mike = 6) 
    (h_total_parts : total_parts = ratio_mario + ratio_manny + ratio_mike)
    (h_manny_parts : total_marbles/total_parts * ratio_manny = 40) : 
  present_marbles = 40 := 
sorry

end manny_original_marbles_l94_94835


namespace f_2002_eq_0_l94_94979

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_2_eq_0 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem f_2002_eq_0 : f 2002 = 0 :=
by
  sorry

end f_2002_eq_0_l94_94979


namespace find_k_shelf_life_at_11_22_l94_94720

noncomputable def food_shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

-- Given conditions
def condition1 : food_shelf_life k b 0 = 192 := by sorry
def condition2 : food_shelf_life k b 33 = 24 := by sorry

-- Prove that k = - (Real.log 2) / 11
theorem find_k (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) : 
  k = - (Real.log 2) / 11 :=
by sorry

-- Use the found value of k to determine the shelf life at 11°C and 22°C
theorem shelf_life_at_11_22 (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) 
  (hk : k = - (Real.log 2) / 11) : 
  food_shelf_life k b 11 = 96 ∧ food_shelf_life k b 22 = 48 :=
by sorry

end find_k_shelf_life_at_11_22_l94_94720


namespace solve_log_eq_l94_94054

theorem solve_log_eq (x : ℝ) (h : 0 < x) :
  (1 / (Real.sqrt (Real.logb 5 (5 * x)) + Real.sqrt (Real.logb 5 x)) + Real.sqrt (Real.logb 5 x) = 2) ↔ x = 125 := 
  sorry

end solve_log_eq_l94_94054


namespace jake_has_3_peaches_l94_94217

-- Define the number of peaches Steven has.
def steven_peaches : ℕ := 13

-- Define the number of peaches Jake has based on the condition.
def jake_peaches (P_S : ℕ) : ℕ := P_S - 10

-- The theorem that states Jake has 3 peaches.
theorem jake_has_3_peaches : jake_peaches steven_peaches = 3 := sorry

end jake_has_3_peaches_l94_94217


namespace calculate_total_cost_l94_94762

theorem calculate_total_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let num_sandwiches := 6
  let num_sodas := 5
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 39 := by
  sorry

end calculate_total_cost_l94_94762


namespace base4_more_digits_than_base9_l94_94734

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l94_94734


namespace frac_y_over_x_plus_y_eq_one_third_l94_94339

theorem frac_y_over_x_plus_y_eq_one_third (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end frac_y_over_x_plus_y_eq_one_third_l94_94339


namespace number_of_houses_with_neither_feature_l94_94899

variable (T G P B : ℕ)

theorem number_of_houses_with_neither_feature 
  (hT : T = 90)
  (hG : G = 50)
  (hP : P = 40)
  (hB : B = 35) : 
  T - (G + P - B) = 35 := 
    by
      sorry

end number_of_houses_with_neither_feature_l94_94899


namespace stock_price_end_of_third_year_l94_94707

def stock_price_after_years (initial_price : ℝ) (year1_increase : ℝ) (year2_decrease : ℝ) (year3_increase : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_increase)
  let price_after_year2 := price_after_year1 * (1 - year2_decrease)
  let price_after_year3 := price_after_year2 * (1 + year3_increase)
  price_after_year3

theorem stock_price_end_of_third_year :
  stock_price_after_years 120 0.80 0.30 0.50 = 226.8 := 
by
  sorry

end stock_price_end_of_third_year_l94_94707


namespace points_on_curve_l94_94715

theorem points_on_curve (x y : ℝ) :
  (∃ p : ℝ, y = p^2 + (2 * p - 1) * x + 2 * x^2) ↔ y ≥ x^2 - x :=
by
  sorry

end points_on_curve_l94_94715


namespace find_tangent_point_and_slope_l94_94801

theorem find_tangent_point_and_slope :
  ∃ m n : ℝ, (m = 1 ∧ n = Real.exp 1 ∧ 
    (∀ x y : ℝ, y - n = (Real.exp m) * (x - m) → x = 0 ∧ y = 0) ∧ 
    (Real.exp m = Real.exp 1)) :=
sorry

end find_tangent_point_and_slope_l94_94801


namespace correct_negation_l94_94704

-- Define a triangle with angles A, B, and C
variables (α β γ : ℝ)

-- Define properties of the angles
def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180
def is_right_angle (angle : ℝ) : Prop := angle = 90
def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

-- Original statement to be negated
def original_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ is_right_angle γ → is_acute_angle α ∧ is_acute_angle β

-- Negation of the original statement
def negated_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ ¬ is_right_angle γ → ¬ (is_acute_angle α ∧ is_acute_angle β)

-- Proof statement: prove that the negated statement is the correct negation
theorem correct_negation (α β γ : ℝ) :
  negated_statement α β γ = ¬ original_statement α β γ :=
sorry

end correct_negation_l94_94704


namespace triangle_side_range_l94_94833

theorem triangle_side_range (x : ℝ) (h1 : x > 0) (h2 : x + (x + 1) + (x + 2) ≤ 12) :
  1 < x ∧ x ≤ 3 :=
by
  sorry

end triangle_side_range_l94_94833


namespace jason_two_weeks_eggs_l94_94167

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end jason_two_weeks_eggs_l94_94167


namespace tree_ratio_l94_94202

theorem tree_ratio (native_trees : ℕ) (total_planted : ℕ) (M : ℕ) 
  (h1 : native_trees = 30) 
  (h2 : total_planted = 80) 
  (h3 : total_planted = M + M / 3) :
  (native_trees + M) / native_trees = 3 :=
sorry

end tree_ratio_l94_94202


namespace percentage_students_below_8_years_l94_94227

theorem percentage_students_below_8_years :
  ∀ (n8 : ℕ) (n_gt8 : ℕ) (n_total : ℕ),
  n8 = 24 →
  n_gt8 = 2 * n8 / 3 →
  n_total = 50 →
  (n_total - (n8 + n_gt8)) * 100 / n_total = 20 :=
by
  intros n8 n_gt8 n_total h1 h2 h3
  sorry

end percentage_students_below_8_years_l94_94227


namespace find_possible_values_of_a_l94_94849

noncomputable def P : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_possible_values_of_a (a : ℝ) (h : Q a ⊆ P) :
  a = 0 ∨ a = -1/2 ∨ a = 1/3 := by
  sorry

end find_possible_values_of_a_l94_94849


namespace business_total_profit_l94_94779

noncomputable def total_profit (spending_ratio income_ratio total_income : ℕ) : ℕ :=
  let total_parts := spending_ratio + income_ratio
  let one_part_value := total_income / income_ratio
  let spending := spending_ratio * one_part_value
  total_income - spending

theorem business_total_profit :
  total_profit 5 9 108000 = 48000 :=
by
  -- We omit the proof steps, as instructed.
  sorry

end business_total_profit_l94_94779


namespace crossnumber_unique_solution_l94_94065

-- Definition of two-digit numbers
def two_digit_numbers (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Definition of prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of square
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The given conditions reformulated
def crossnumber_problem : Prop :=
  ∃ (one_across one_down two_down three_across : ℕ),
    two_digit_numbers one_across ∧ is_prime one_across ∧
    two_digit_numbers one_down ∧ is_square one_down ∧
    two_digit_numbers two_down ∧ is_square two_down ∧
    two_digit_numbers three_across ∧ is_square three_across ∧
    one_across = 83 ∧ one_down = 81 ∧ two_down = 16 ∧ three_across = 16

theorem crossnumber_unique_solution : crossnumber_problem :=
by
  sorry

end crossnumber_unique_solution_l94_94065


namespace houses_with_neither_l94_94290

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end houses_with_neither_l94_94290


namespace labor_productivity_increase_l94_94387

noncomputable def regression_equation (x : ℝ) : ℝ := 50 + 60 * x

theorem labor_productivity_increase (Δx : ℝ) (hx : Δx = 1) :
  regression_equation (x + Δx) - regression_equation x = 60 :=
by
  sorry

end labor_productivity_increase_l94_94387


namespace rearrange_marked_squares_l94_94570

theorem rearrange_marked_squares (n k : ℕ) (h : n > 1) (h' : k ≤ n + 1) :
  ∃ (f g : Fin n → Fin n), true := sorry

end rearrange_marked_squares_l94_94570


namespace non_prime_in_sequence_l94_94374

theorem non_prime_in_sequence : ∃ n : ℕ, ¬Prime (41 + n * (n - 1)) :=
by {
  use 41,
  sorry
}

end non_prime_in_sequence_l94_94374


namespace intersection_of_A_and_B_l94_94089

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l94_94089


namespace non_mobile_payment_probability_40_60_l94_94971

variable (total_customers : ℕ)
variable (num_non_mobile_40_50 : ℕ)
variable (num_non_mobile_50_60 : ℕ)

theorem non_mobile_payment_probability_40_60 
  (h_total_customers: total_customers = 100)
  (h_num_non_mobile_40_50: num_non_mobile_40_50 = 9)
  (h_num_non_mobile_50_60: num_non_mobile_50_60 = 5) : 
  (num_non_mobile_40_50 + num_non_mobile_50_60 : ℚ) / total_customers = 7 / 50 :=
by
  -- Placeholder for the actual proof
  sorry

end non_mobile_payment_probability_40_60_l94_94971


namespace solve_for_y_l94_94647

theorem solve_for_y (y : ℝ) : 5 * y - 100 = 125 ↔ y = 45 := by
  sorry

end solve_for_y_l94_94647


namespace initial_amount_of_money_l94_94571

-- Define the conditions
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20
def num_friends : ℕ := 2
def amount_left : ℝ := 114.85

-- Define the calculated amount given to friends
def total_given_to_friends : ℝ := given_to_each_friend * num_friends

-- State the theorem to prove the initial amount of money
theorem initial_amount_of_money :
  spent_on_sweets + total_given_to_friends + amount_left = 200.50 :=
by 
  -- proof goes here
  sorry

end initial_amount_of_money_l94_94571


namespace marian_returned_amount_l94_94635

theorem marian_returned_amount
  (B : ℕ) (G : ℕ) (H : ℕ) (N : ℕ)
  (hB : B = 126) (hG : G = 60) (hH : H = G / 2) (hN : N = 171) :
  (B + G + H - N) = 45 := 
by
  sorry

end marian_returned_amount_l94_94635


namespace total_tickets_needed_l94_94541

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end total_tickets_needed_l94_94541


namespace equation_solutions_equiv_l94_94799

theorem equation_solutions_equiv (p : ℕ) (hp : p.Prime) :
  (∃ x s : ℤ, x^2 - x + 3 - p * s = 0) ↔ 
  (∃ y t : ℤ, y^2 - y + 25 - p * t = 0) :=
by { sorry }

end equation_solutions_equiv_l94_94799


namespace multiple_of_9_digit_l94_94079

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end multiple_of_9_digit_l94_94079


namespace circumferences_ratio_l94_94505

theorem circumferences_ratio (r1 r2 : ℝ) (h : (π * r1 ^ 2) / (π * r2 ^ 2) = 49 / 64) : r1 / r2 = 7 / 8 :=
sorry

end circumferences_ratio_l94_94505


namespace intersecting_lines_implies_a_eq_c_l94_94308

theorem intersecting_lines_implies_a_eq_c
  (k b a c : ℝ)
  (h_kb : k ≠ b)
  (exists_point : ∃ (x y : ℝ), (y = k * x + k) ∧ (y = b * x + b) ∧ (y = a * x + c)) :
  a = c := 
sorry

end intersecting_lines_implies_a_eq_c_l94_94308


namespace a_81_eq_640_l94_94200

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 0 -- auxiliary value since sequence begins from n=1
else if n = 1 then 1
else (2 * n - 1) ^ 2 - (2 * n - 3) ^ 2

theorem a_81_eq_640 : sequence_a 81 = 640 :=
by
  sorry

end a_81_eq_640_l94_94200


namespace bread_weight_eq_anton_weight_l94_94077

-- Definitions of variables
variables (A B F X : ℝ)

-- Given conditions
axiom cond1 : X + F = A + B
axiom cond2 : B + X = A + F

-- Theorem to prove
theorem bread_weight_eq_anton_weight : X = A :=
by
  sorry

end bread_weight_eq_anton_weight_l94_94077


namespace parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l94_94953

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 2 * x

theorem parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2 :
  parabola_equation 1 (-Real.sqrt 2) :=
by
  sorry

end parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l94_94953


namespace math_olympiad_proof_l94_94255

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l94_94255


namespace calculate_expression_l94_94986

theorem calculate_expression : (632^2 - 568^2 + 100) = 76900 :=
by sorry

end calculate_expression_l94_94986


namespace find_min_value_omega_l94_94860

noncomputable def min_value_ω (ω : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 2 * Real.sin (ω * x)) → ω > 0 →
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -2) →
  ω = 3 / 2

-- The statement to be proved:
theorem find_min_value_omega : ∃ ω : ℝ, min_value_ω ω :=
by
  use 3 / 2
  sorry

end find_min_value_omega_l94_94860


namespace quadratic_solution_l94_94617

theorem quadratic_solution (x : ℝ) : 2 * x * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 3 / 2) := by
  sorry

end quadratic_solution_l94_94617


namespace number_of_whole_numbers_between_sqrt2_and_3e_is_7_l94_94112

noncomputable def number_of_whole_numbers_between_sqrt2_and_3e : ℕ :=
  let sqrt2 : ℝ := Real.sqrt 2
  let e : ℝ := Real.exp 1
  let small_int := Nat.ceil sqrt2 -- This is 2
  let large_int := Nat.floor (3 * e) -- This is 8
  large_int - small_int + 1 -- The number of integers between small_int and large_int (inclusive)

theorem number_of_whole_numbers_between_sqrt2_and_3e_is_7 :
  number_of_whole_numbers_between_sqrt2_and_3e = 7 := by
  sorry

end number_of_whole_numbers_between_sqrt2_and_3e_is_7_l94_94112


namespace bruce_money_left_l94_94564

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l94_94564


namespace domain_of_sqrt_ln_l94_94340

def domain_function (x : ℝ) : Prop := x - 1 ≥ 0 ∧ 2 - x > 0

theorem domain_of_sqrt_ln (x : ℝ) : domain_function x ↔ 1 ≤ x ∧ x < 2 := by
  sorry

end domain_of_sqrt_ln_l94_94340


namespace necessary_but_not_sufficient_l94_94467

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f a x - f a 1 ≥ 0) ↔ (a ≤ -2) :=
sorry

end necessary_but_not_sufficient_l94_94467


namespace find_fourth_speed_l94_94819

theorem find_fourth_speed 
  (avg_speed : ℝ)
  (speed1 speed2 speed3 fourth_speed : ℝ)
  (h_avg_speed : avg_speed = 11.52)
  (h_speed1 : speed1 = 6.0)
  (h_speed2 : speed2 = 12.0)
  (h_speed3 : speed3 = 18.0)
  (expected_avg_speed_eq : avg_speed = 4 / ((1 / speed1) + (1 / speed2) + (1 / speed3) + (1 / fourth_speed))) :
  fourth_speed = 2.095 :=
by 
  sorry

end find_fourth_speed_l94_94819


namespace inequality_solution_l94_94922

noncomputable def solution_set : Set ℝ :=
  {x | -4 < x ∧ x < (17 - Real.sqrt 201) / 4} ∪ {x | (17 + Real.sqrt 201) / 4 < x ∧ x < 2 / 3}

theorem inequality_solution (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 2 / 3) :
  (2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2) ↔ x ∈ solution_set := by
  sorry

end inequality_solution_l94_94922


namespace good_numbers_identification_l94_94569

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), 
    (∀ k : Fin n, ∃ m : ℕ, k.val + a k = m * m)

theorem good_numbers_identification : 
  { n : ℕ | ¬is_good_number n } = {1, 2, 4, 6, 7, 9, 11} :=
  sorry

end good_numbers_identification_l94_94569


namespace x_sq_sub_y_sq_l94_94539

theorem x_sq_sub_y_sq (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  sorry

end x_sq_sub_y_sq_l94_94539


namespace remainder_x150_l94_94958

theorem remainder_x150 (x : ℝ) : 
  ∃ r : ℝ, ∃ q : ℝ, x^150 = q * (x - 1)^3 + 11175*x^2 - 22200*x + 11026 := 
by
  sorry

end remainder_x150_l94_94958


namespace isosceles_triangle_base_length_l94_94975

theorem isosceles_triangle_base_length (x : ℝ) (h1 : 2 * x + 2 * x + x = 20) : x = 4 :=
sorry

end isosceles_triangle_base_length_l94_94975


namespace chris_is_14_l94_94821

-- Definitions from the given conditions
variables (a b c : ℕ)
variables (h1 : (a + b + c) / 3 = 10)
variables (h2 : c - 4 = a)
variables (h3 : b + 5 = (3 * (a + 5)) / 4)

theorem chris_is_14 (h1 : (a + b + c) / 3 = 10) (h2 : c - 4 = a) (h3 : b + 5 = (3 * (a + 5)) / 4) : c = 14 := 
sorry

end chris_is_14_l94_94821


namespace parabola_equation_standard_form_l94_94177

theorem parabola_equation_standard_form (p : ℝ) (x y : ℝ)
    (h₁ : y^2 = 2 * p * x)
    (h₂ : y = -4)
    (h₃ : x = -2) : y^2 = -8 * x := by
  sorry

end parabola_equation_standard_form_l94_94177


namespace find_integers_l94_94173

theorem find_integers 
  (A k : ℕ) 
  (h_sum : A + A * k + A * k^2 = 93) 
  (h_product : A * (A * k) * (A * k^2) = 3375) : 
  (A, A * k, A * k^2) = (3, 15, 75) := 
by 
  sorry

end find_integers_l94_94173


namespace divisibility_equiv_l94_94489

theorem divisibility_equiv (n : ℕ) : (7 ∣ 3^n + n^3) ↔ (7 ∣ 3^n * n^3 + 1) :=
by sorry

end divisibility_equiv_l94_94489


namespace shaded_region_area_and_circle_centers_l94_94280

theorem shaded_region_area_and_circle_centers :
  ∃ (R : ℝ) (center_big center_small1 center_small2 : ℝ × ℝ),
    R = 10 ∧ 
    center_small1 = (4, 0) ∧
    center_small2 = (10, 0) ∧
    center_big = (7, 0) ∧
    (π * R^2) - (π * 4^2 + π * 6^2) = 48 * π :=
by 
  sorry

end shaded_region_area_and_circle_centers_l94_94280


namespace molly_swam_28_meters_on_sunday_l94_94119

def meters_swam_on_saturday : ℕ := 45
def total_meters_swum : ℕ := 73
def meters_swam_on_sunday := total_meters_swum - meters_swam_on_saturday

theorem molly_swam_28_meters_on_sunday : meters_swam_on_sunday = 28 :=
by
  -- sorry to skip the proof
  sorry

end molly_swam_28_meters_on_sunday_l94_94119


namespace solve_x_squared_plus_15_eq_y_squared_l94_94190

theorem solve_x_squared_plus_15_eq_y_squared (x y : ℤ) : x^2 + 15 = y^2 → x = 7 ∨ x = -7 ∨ x = 1 ∨ x = -1 := by
  sorry

end solve_x_squared_plus_15_eq_y_squared_l94_94190


namespace rate_of_current_is_8_5_l94_94004

-- Define the constants for the problem
def downstream_speed : ℝ := 24
def upstream_speed : ℝ := 7
def rate_still_water : ℝ := 15.5

-- Define the rate of the current calculation
def rate_of_current : ℝ := downstream_speed - rate_still_water

-- Define the rate of the current proof statement
theorem rate_of_current_is_8_5 :
  rate_of_current = 8.5 :=
by
  -- This skip the actual proof
  sorry

end rate_of_current_is_8_5_l94_94004


namespace decimal_to_base7_l94_94198

theorem decimal_to_base7 :
    ∃ k₀ k₁ k₂ k₃ k₄, 1987 = k₀ * 7^4 + k₁ * 7^3 + k₂ * 7^2 + k₃ * 7^1 + k₄ * 7^0 ∧
    k₀ = 0 ∧
    k₁ = 5 ∧
    k₂ = 3 ∧
    k₃ = 5 ∧
    k₄ = 6 :=
by
  sorry

end decimal_to_base7_l94_94198


namespace sum_of_digits_of_smallest_divisible_is_6_l94_94854

noncomputable def smallest_divisible (n : ℕ) : ℕ :=
Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_smallest_divisible_is_6 : sum_of_digits (smallest_divisible 7) = 6 := 
by
  simp [smallest_divisible, sum_of_digits]
  sorry

end sum_of_digits_of_smallest_divisible_is_6_l94_94854


namespace union_is_real_l94_94030

-- Definitions of sets A and B
def setA : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def setB : Set ℝ := {x | x > -1}

-- Theorem to prove
theorem union_is_real :
  setA ∪ setB = Set.univ :=
by
  sorry

end union_is_real_l94_94030


namespace parabola_rotation_180_equivalent_l94_94593

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 - 2

-- Define the expected rotated parabola equation
def rotated_parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- Prove that the rotated parabola is correctly transformed
theorem parabola_rotation_180_equivalent :
  ∀ x, rotated_parabola x = -2 * (x - 3)^2 - 2 := 
by
  intro x
  unfold rotated_parabola
  sorry

end parabola_rotation_180_equivalent_l94_94593


namespace sum_of_digits_l94_94264

theorem sum_of_digits (P Q R : ℕ) (hP : P < 10) (hQ : Q < 10) (hR : R < 10)
 (h_sum : P * 1000 + Q * 100 + Q * 10 + R = 2009) : P + Q + R = 10 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l94_94264


namespace radius_of_circle_l94_94838

theorem radius_of_circle :
  ∀ (r : ℝ), (π * r^2 = 2.5 * 2 * π * r) → r = 5 :=
by sorry

end radius_of_circle_l94_94838


namespace sufficiency_condition_l94_94378

-- Definitions of p and q
def p (a b : ℝ) : Prop := a > |b|
def q (a b : ℝ) : Prop := a^2 > b^2

-- Main theorem statement
theorem sufficiency_condition (a b : ℝ) : (p a b → q a b) ∧ (¬(q a b → p a b)) := 
by
  sorry

end sufficiency_condition_l94_94378


namespace efficiency_of_worker_p_more_than_q_l94_94352

noncomputable def worker_p_rate : ℚ := 1 / 22
noncomputable def combined_rate : ℚ := 1 / 12

theorem efficiency_of_worker_p_more_than_q
  (W_p : ℚ) (W_q : ℚ)
  (h1 : W_p = worker_p_rate)
  (h2 : W_p + W_q = combined_rate) : (W_p / W_q) = 6 / 5 :=
by
  sorry

end efficiency_of_worker_p_more_than_q_l94_94352


namespace chromosome_stability_due_to_meiosis_and_fertilization_l94_94532

-- Definitions for conditions
def chrom_replicate_distribute_evenly : Prop := true
def central_cell_membrane_invagination : Prop := true
def mitosis : Prop := true
def meiosis_and_fertilization : Prop := true

-- Main theorem statement to be proved
theorem chromosome_stability_due_to_meiosis_and_fertilization :
  meiosis_and_fertilization :=
sorry

end chromosome_stability_due_to_meiosis_and_fertilization_l94_94532


namespace light_bulb_arrangement_l94_94363

theorem light_bulb_arrangement :
  let B := 6
  let R := 7
  let W := 9
  let total_arrangements := Nat.choose (B + R) B * Nat.choose (B + R + 1) W
  total_arrangements = 3435432 :=
by
  sorry

end light_bulb_arrangement_l94_94363


namespace lumberjack_trees_l94_94554

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end lumberjack_trees_l94_94554


namespace system_of_equations_solution_l94_94142

theorem system_of_equations_solution (x y : ℝ) (h1 : |y - x| - (|x| / x) + 1 = 0) (h2 : |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0) (hx : x ≠ 0) :
  (0 < x ∧ x ≤ 0.5 ∧ y = x) :=
by
  sorry

end system_of_equations_solution_l94_94142


namespace total_blue_balloons_l94_94305

def Joan_balloons : Nat := 9
def Sally_balloons : Nat := 5
def Jessica_balloons : Nat := 2

theorem total_blue_balloons : Joan_balloons + Sally_balloons + Jessica_balloons = 16 :=
by
  sorry

end total_blue_balloons_l94_94305


namespace minimum_value_inequality_l94_94281

theorem minimum_value_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 = 1) :
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3 / 2) :=
sorry

end minimum_value_inequality_l94_94281


namespace find_initial_crayons_l94_94676

namespace CrayonProblem

variable (gave : ℕ) (lost : ℕ) (additional_lost : ℕ) 

def correct_answer (gave lost additional_lost : ℕ) :=
  gave + lost = gave + (gave + additional_lost) ∧ gave + lost = 502

theorem find_initial_crayons
  (gave := 90)
  (lost := 412)
  (additional_lost := 322)
  : correct_answer gave lost additional_lost :=
by 
  sorry

end CrayonProblem

end find_initial_crayons_l94_94676


namespace triangle_third_side_l94_94332

open Nat

theorem triangle_third_side (a b c : ℝ) (h1 : a = 4) (h2 : b = 9) (h3 : c > 0) :
  (5 < c ∧ c < 13) ↔ c = 6 :=
by
  sorry

end triangle_third_side_l94_94332


namespace find_root_of_quadratic_equation_l94_94220

theorem find_root_of_quadratic_equation
  (a b c : ℝ)
  (h1 : 3 * a * (2 * b - 3 * c) ≠ 0)
  (h2 : 2 * b * (3 * c - 2 * a) ≠ 0)
  (h3 : 5 * c * (2 * a - 3 * b) ≠ 0)
  (r : ℝ)
  (h_roots : (r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) ∨ (r = (-2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) * 2)) :
  r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c)) :=
by
  sorry

end find_root_of_quadratic_equation_l94_94220


namespace jade_transactions_l94_94859

theorem jade_transactions (mabel anthony cal jade : ℕ) 
    (h1 : mabel = 90) 
    (h2 : anthony = mabel + (10 * mabel / 100)) 
    (h3 : cal = 2 * anthony / 3) 
    (h4 : jade = cal + 18) : 
    jade = 84 := by 
  -- Start with given conditions
  rw [h1] at h2 
  have h2a : anthony = 99 := by norm_num; exact h2 
  rw [h2a] at h3 
  have h3a : cal = 66 := by norm_num; exact h3 
  rw [h3a] at h4 
  norm_num at h4 
  exact h4

end jade_transactions_l94_94859


namespace number_of_tires_l94_94645

theorem number_of_tires (n : ℕ)
  (repair_cost : ℕ → ℝ)
  (sales_tax : ℕ → ℝ)
  (total_cost : ℝ) :
  (∀ t, repair_cost t = 7) →
  (∀ t, sales_tax t = 0.5) →
  (total_cost = n * (repair_cost 0 + sales_tax 0)) →
  total_cost = 30 →
  n = 4 :=
by 
  sorry

end number_of_tires_l94_94645


namespace arithmetic_sequence_ratio_l94_94416

noncomputable def sum_first_n_terms (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_ratio (d : ℚ) (h : d ≠ 0) :
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  (7 * S₅) / (5 * S₇) = 10 / 11 :=
by 
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  sorry

end arithmetic_sequence_ratio_l94_94416


namespace intersection_lines_l94_94679

theorem intersection_lines (x y : ℝ) :
  (2 * x - y - 10 = 0) ∧ (3 * x + 4 * y - 4 = 0) → (x = 4) ∧ (y = -2) :=
by
  -- The proof is provided here
  sorry

end intersection_lines_l94_94679


namespace quadratic_roots_ratio_l94_94746

noncomputable def value_of_m (m : ℚ) : Prop :=
  ∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ (r / s = 3) ∧ (r + s = -9) ∧ (r * s = m)

theorem quadratic_roots_ratio (m : ℚ) (h : value_of_m m) : m = 243 / 16 :=
by
  sorry

end quadratic_roots_ratio_l94_94746


namespace find_leak_rate_l94_94210

-- Conditions in Lean 4
def pool_capacity : ℝ := 60
def hose_rate : ℝ := 1.6
def fill_time : ℝ := 40

-- Define the leak rate calculation
def leak_rate (L : ℝ) : Prop :=
  pool_capacity = (hose_rate - L) * fill_time

-- The main theorem we want to prove
theorem find_leak_rate : ∃ L, leak_rate L ∧ L = 0.1 := by
  sorry

end find_leak_rate_l94_94210


namespace no_maximum_y_coordinate_for_hyperbola_l94_94641

theorem no_maximum_y_coordinate_for_hyperbola :
  ∀ y : ℝ, ∃ x : ℝ, y = 3 + (3 / 5) * x :=
by
  sorry

end no_maximum_y_coordinate_for_hyperbola_l94_94641


namespace shirt_price_percentage_l94_94247

variable (original_price : ℝ) (final_price : ℝ)

def calculate_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_new_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_final_price (p : ℝ) : ℝ := 0.85 * p

theorem shirt_price_percentage :
  (original_price = 60) →
  (final_price = calculate_final_price (calculate_new_sale_price (calculate_sale_price original_price))) →
  (final_price / original_price) * 100 = 54.4 :=
by
  intros h₁ h₂
  sorry

end shirt_price_percentage_l94_94247


namespace range_of_a_min_value_reciprocals_l94_94870

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ a) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem min_value_reciprocals (m n a : ℝ) (h : m + 2 * n = a) (ha : a = 2) : (1/m + 1/n) ≥ (3/2 + Real.sqrt 2) := by
  sorry

end range_of_a_min_value_reciprocals_l94_94870


namespace x1_mul_x2_l94_94775

open Real

theorem x1_mul_x2 (x1 x2 : ℝ) (h1 : x1 + x2 = 2 * sqrt 1703) (h2 : abs (x1 - x2) = 90) : x1 * x2 = -322 := by
  sorry

end x1_mul_x2_l94_94775


namespace nonneg_int_solutions_eq_l94_94020

theorem nonneg_int_solutions_eq (a b : ℕ) : a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by {
  sorry -- Proof omitted
}

end nonneg_int_solutions_eq_l94_94020


namespace sweets_distribution_l94_94658

theorem sweets_distribution (S : ℕ) (N : ℕ) (h1 : N - 70 > 0) (h2 : S = N * 24) (h3 : S = (N - 70) * 38) : N = 190 :=
by
  sorry

end sweets_distribution_l94_94658


namespace geometric_sequence_common_ratio_l94_94211

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 5 = 16)
  (h_pos : ∀ n : ℕ, 0 < a n) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l94_94211


namespace distance_between_city_centers_l94_94768

theorem distance_between_city_centers (d_map : ℝ) (scale : ℝ) (d_real : ℝ) (h1 : d_map = 112) (h2 : scale = 10) (h3 : d_real = d_map * scale) : d_real = 1120 := by
  sorry

end distance_between_city_centers_l94_94768


namespace cube_painted_probability_l94_94193

theorem cube_painted_probability :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 76
  let total_ways := Nat.choose total_cubes 2
  let favorable_ways := cubes_with_3_faces * cubes_with_no_faces
  let probability := (favorable_ways : ℚ) / total_ways
  probability = (2 : ℚ) / 205 :=
by
  sorry

end cube_painted_probability_l94_94193


namespace insurance_plan_percentage_l94_94426

theorem insurance_plan_percentage
(MSRP : ℝ) (I : ℝ) (total_cost : ℝ) (state_tax_rate : ℝ)
(hMSRP : MSRP = 30)
(htotal_cost : total_cost = 54)
(hstate_tax_rate : state_tax_rate = 0.5)
(h_total_cost_eq : MSRP + I + state_tax_rate * (MSRP + I) = total_cost) :
(I / MSRP) * 100 = 20 :=
by
  -- You can leave the proof as sorry, as it's not needed for the problem
  sorry

end insurance_plan_percentage_l94_94426


namespace age_problem_l94_94544

theorem age_problem 
  (A : ℕ) 
  (x : ℕ) 
  (h1 : 3 * (A + x) - 3 * (A - 3) = A) 
  (h2 : A = 18) : 
  x = 3 := 
by 
  sorry

end age_problem_l94_94544


namespace right_triangle_third_side_square_l94_94955

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) 
  (h₁ : a = 3) (h₂ : b = 4) (h₃ : a^2 + b^2 = c^2) :
  c^2 = 25 ∨ a^2 + c^2 = b^2 ∨ a^2 + b^2 = 7 :=
by
  sorry

end right_triangle_third_side_square_l94_94955


namespace average_speed_l94_94246

theorem average_speed (d1 d2 d3 v1 v2 v3 total_distance total_time avg_speed : ℝ)
    (h1 : d1 = 40) (h2 : d2 = 20) (h3 : d3 = 10) 
    (h4 : v1 = 8) (h5 : v2 = 40) (h6 : v3 = 20) 
    (h7 : total_distance = d1 + d2 + d3)
    (h8 : total_time = d1 / v1 + d2 / v2 + d3 / v3) 
    (h9 : avg_speed = total_distance / total_time) : avg_speed = 11.67 :=
by 
  sorry

end average_speed_l94_94246


namespace no_unique_y_exists_l94_94185

theorem no_unique_y_exists (x y : ℕ) (k m : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + 7) % y = 12) :
  ¬ ∃! y, (∃ k m : ℤ, x = 82 * k + 5 ∧ (x + 7) = y * m + 12) :=
by
  sorry

end no_unique_y_exists_l94_94185


namespace percentage_of_female_officers_on_duty_l94_94241

theorem percentage_of_female_officers_on_duty :
  ∀ (total_on_duty : ℕ) (half_on_duty : ℕ) (total_female_officers : ℕ), 
  total_on_duty = 204 → half_on_duty = total_on_duty / 2 → total_female_officers = 600 → 
  ((half_on_duty: ℚ) / total_female_officers) * 100 = 17 :=
by
  intro total_on_duty half_on_duty total_female_officers
  intros h1 h2 h3
  sorry

end percentage_of_female_officers_on_duty_l94_94241


namespace Marcia_wardrobe_cost_l94_94132

-- Definitions from the problem
def skirt_price : ℝ := 20
def blouse_price : ℝ := 15
def pant_price : ℝ := 30

def num_skirts : ℕ := 3
def num_blouses : ℕ := 5
def num_pants : ℕ := 2

-- The main theorem statement
theorem Marcia_wardrobe_cost :
  (num_skirts * skirt_price) + (num_blouses * blouse_price) + (pant_price + (pant_price / 2)) = 180 :=
by
  sorry

end Marcia_wardrobe_cost_l94_94132


namespace minimum_time_to_finish_food_l94_94914

-- Define the constants involved in the problem
def carrots_total : ℕ := 1000
def muffins_total : ℕ := 1000
def amy_carrots_rate : ℝ := 40 -- carrots per minute
def amy_muffins_rate : ℝ := 70 -- muffins per minute
def ben_carrots_rate : ℝ := 60 -- carrots per minute
def ben_muffins_rate : ℝ := 30 -- muffins per minute

-- Proof statement
theorem minimum_time_to_finish_food : 
  ∃ T : ℝ, 
  (∀ c : ℝ, c = 5 → 
  (∀ T_1 : ℝ, T_1 = (carrots_total / (amy_carrots_rate + ben_carrots_rate)) → 
  (∀ T_2 : ℝ, T_2 = ((muffins_total + (amy_muffins_rate * c)) / (amy_muffins_rate + ben_muffins_rate)) +
  (muffins_total / ben_muffins_rate) - T_1 - c →
  T = T_1 + T_2) ∧
  T = 23.5 )) :=
sorry

end minimum_time_to_finish_food_l94_94914


namespace complement_union_covers_until_1_l94_94605

open Set

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3*x - 4 ≤ 0}
noncomputable def complement_R_S := {x : ℝ | x ≤ -2}
noncomputable def union := complement_R_S ∪ T

theorem complement_union_covers_until_1 : union = {x : ℝ | x ≤ 1} := by
  sorry

end complement_union_covers_until_1_l94_94605


namespace smallest_term_l94_94334

theorem smallest_term (a1 d : ℕ) (h_a1 : a1 = 7) (h_d : d = 7) :
  ∃ n : ℕ, (a1 + (n - 1) * d) > 150 ∧ (a1 + (n - 1) * d) % 5 = 0 ∧
  (∀ m : ℕ, (a1 + (m - 1) * d) > 150 ∧ (a1 + (m - 1) * d) % 5 = 0 → (a1 + (m - 1) * d) ≥ (a1 + (n - 1) * d)) → a1 + (n - 1) * d = 175 :=
by
  -- We need to prove given the conditions.
  sorry

end smallest_term_l94_94334


namespace question1_question2_l94_94018

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem question1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ 1 := sorry

theorem question2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  x1 * Real.log x1 - x1 * Real.log x2 > x1 - x2 := sorry

end question1_question2_l94_94018


namespace verify_incorrect_option_l94_94189

variable (a : ℕ → ℝ) -- The sequence a_n
variable (S : ℕ → ℝ) -- The sum of the first n terms S_n

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℝ) : Prop := S 5 < S 6

def condition_2 (S : ℕ → ℝ) : Prop := S 6 = S 7 ∧ S 7 > S 8

theorem verify_incorrect_option (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond1 : condition_1 S)
  (h_cond2 : condition_2 S) :
  S 9 ≤ S 5 :=
sorry

end verify_incorrect_option_l94_94189


namespace cost_price_of_toy_l94_94691

theorem cost_price_of_toy 
  (cost_price : ℝ)
  (SP : ℝ := 120000)
  (num_toys : ℕ := 40)
  (profit_per_toy : ℝ := 500)
  (gain_per_toy : ℝ := cost_price + profit_per_toy)
  (total_gain : ℝ := 8 * cost_price + profit_per_toy * num_toys)
  (total_cost_price : ℝ := num_toys * cost_price)
  (SP_eq_cost_plus_gain : SP = total_cost_price + total_gain) :
  cost_price = 2083.33 :=
by
  sorry

end cost_price_of_toy_l94_94691


namespace students_not_picked_l94_94692

/-- There are 36 students trying out for the school's trivia teams. 
If some of them didn't get picked and the rest were put into 3 groups with 9 students in each group,
prove that the number of students who didn't get picked is 9. -/

theorem students_not_picked (total_students groups students_per_group picked_students not_picked_students : ℕ)
    (h1 : total_students = 36)
    (h2 : groups = 3)
    (h3 : students_per_group = 9)
    (h4 : picked_students = groups * students_per_group)
    (h5 : not_picked_students = total_students - picked_students) :
    not_picked_students = 9 :=
by
  sorry

end students_not_picked_l94_94692


namespace students_in_cars_l94_94486

theorem students_in_cars (total_students : ℕ := 396) (buses : ℕ := 7) (students_per_bus : ℕ := 56) :
  total_students - (buses * students_per_bus) = 4 := by
  sorry

end students_in_cars_l94_94486


namespace heather_initial_oranges_l94_94987

theorem heather_initial_oranges (given_oranges: ℝ) (total_oranges: ℝ) (initial_oranges: ℝ) 
    (h1: given_oranges = 35.0) 
    (h2: total_oranges = 95) : 
    initial_oranges = 60 :=
by
  sorry

end heather_initial_oranges_l94_94987


namespace fraction_of_apples_consumed_l94_94797

theorem fraction_of_apples_consumed (f : ℚ) 
  (bella_eats_per_day : ℚ := 6) 
  (days_per_week : ℕ := 7) 
  (grace_remaining_apples : ℚ := 504) 
  (weeks_passed : ℕ := 6) 
  (total_apples_picked : ℚ := 42 / f) :
  (total_apples_picked - (bella_eats_per_day * days_per_week * weeks_passed) = grace_remaining_apples) 
  → f = 1 / 18 :=
by
  intro h
  sorry

end fraction_of_apples_consumed_l94_94797


namespace probability_of_at_least_40_cents_l94_94786

-- Definitions for each type of coin and their individual values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- The total value needed for a successful outcome
def minimum_success_value := 40

-- Total number of possible outcomes from flipping 5 coins independently
def total_outcomes := 2^5

-- Count the successful outcomes that result in at least 40 cents
-- This is a placeholder for the actual successful counting method
noncomputable def successful_outcomes := 18

-- Calculate the probability of successful outcomes
noncomputable def probability := (successful_outcomes : ℚ) / total_outcomes

-- Proof statement to show the probability is 9/16
theorem probability_of_at_least_40_cents : probability = 9 / 16 := 
by
  sorry

end probability_of_at_least_40_cents_l94_94786


namespace machine_production_l94_94696

theorem machine_production
  (rate_per_minute : ℕ)
  (machines_total : ℕ)
  (production_minute : ℕ)
  (machines_sub : ℕ)
  (time_minutes : ℕ)
  (total_production : ℕ) :
  machines_total * rate_per_minute = production_minute →
  rate_per_minute = production_minute / machines_total →
  machines_sub * rate_per_minute = total_production / time_minutes →
  time_minutes * total_production / time_minutes = 900 :=
by
  sorry

end machine_production_l94_94696


namespace factorize_expression_l94_94777

theorem factorize_expression (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := 
by 
  sorry

end factorize_expression_l94_94777


namespace rainfall_ratio_l94_94260

theorem rainfall_ratio (r_wed tuesday_rate : ℝ)
    (h_monday : 7 * 1 = 7)
    (h_tuesday : 4 * 2 = 8)
    (h_total : 7 + 8 + 2 * r_wed = 23)
    (h_wed_eq: r_wed = 8 / 2)
    (h_tuesday_rate: tuesday_rate = 2) 
    : r_wed / tuesday_rate = 2 :=
by
  sorry

end rainfall_ratio_l94_94260


namespace number_of_hens_l94_94566

variables (H C : ℕ)

def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 144

theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 24 :=
sorry

end number_of_hens_l94_94566


namespace minimize_distance_l94_94895

noncomputable def f (x : ℝ) := x^2 - 2 * x
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, f x)
def Q : ℝ × ℝ := (4, -1)

theorem minimize_distance : ∃ (x : ℝ), dist (P x) Q = Real.sqrt 5 := by
  sorry

end minimize_distance_l94_94895


namespace negation_of_exists_l94_94129

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + x + 1 < 0) : ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l94_94129


namespace parabola_intersects_once_compare_y_values_l94_94243

noncomputable def parabola (x : ℝ) (m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_intersects_once (m : ℝ) : 
  ∃ x, parabola x m = 0 ↔ m = -2 := 
by 
  sorry

theorem compare_y_values (x1 x2 m : ℝ) (h1 : x1 > x2) (h2 : x2 > 2) : 
  parabola x1 m < parabola x2 m :=
by 
  sorry

end parabola_intersects_once_compare_y_values_l94_94243


namespace sunglasses_cap_probability_l94_94508

theorem sunglasses_cap_probability
  (sunglasses_count : ℕ) (caps_count : ℕ)
  (P_cap_and_sunglasses_given_cap : ℚ)
  (H1 : sunglasses_count = 60)
  (H2 : caps_count = 40)
  (H3 : P_cap_and_sunglasses_given_cap = 2/5) :
  (∃ (x : ℚ), x = (16 : ℚ) / 60 ∧ x = 4 / 15) := sorry

end sunglasses_cap_probability_l94_94508


namespace classrooms_student_hamster_difference_l94_94427

-- Define the problem conditions
def students_per_classroom := 22
def hamsters_per_classroom := 3
def number_of_classrooms := 5

-- Define the problem statement
theorem classrooms_student_hamster_difference :
  (students_per_classroom * number_of_classrooms) - 
  (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end classrooms_student_hamster_difference_l94_94427


namespace mixed_oil_rate_is_correct_l94_94184

def rate_of_mixed_oil (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℕ :=
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2)

theorem mixed_oil_rate_is_correct :
  rate_of_mixed_oil 10 50 5 68 = 56 := by
  sorry

end mixed_oil_rate_is_correct_l94_94184


namespace eventually_periodic_of_rational_cubic_l94_94712

noncomputable def is_rational_sequence (P : ℚ → ℚ) (q : ℕ → ℚ) :=
  ∀ n : ℕ, q (n + 1) = P (q n)

theorem eventually_periodic_of_rational_cubic (P : ℚ → ℚ) (q : ℕ → ℚ) (hP : ∃ a b c d : ℚ, ∀ x : ℚ, P x = a * x^3 + b * x^2 + c * x + d) (hq : is_rational_sequence P q) : 
  ∃ k ≥ 1, ∀ n ≥ 1, q (n + k) = q n := 
sorry

end eventually_periodic_of_rational_cubic_l94_94712


namespace integer_solutions_exist_l94_94810

theorem integer_solutions_exist (x y : ℤ) : 
  12 * x^2 + 7 * y^2 = 4620 ↔ 
  (x = 7 ∧ y = 24) ∨ 
  (x = -7 ∧ y = 24) ∨
  (x = 7 ∧ y = -24) ∨
  (x = -7 ∧ y = -24) ∨
  (x = 14 ∧ y = 18) ∨
  (x = -14 ∧ y = 18) ∨
  (x = 14 ∧ y = -18) ∨
  (x = -14 ∧ y = -18) :=
sorry

end integer_solutions_exist_l94_94810


namespace Tim_sleep_hours_l94_94474

theorem Tim_sleep_hours (x : ℕ) : 
  (x + x + 10 + 10 = 32) → x = 6 :=
by
  intro h
  sorry

end Tim_sleep_hours_l94_94474


namespace rationalize_denominator_l94_94828

theorem rationalize_denominator : (14 / Real.sqrt 14) = Real.sqrt 14 := by
  sorry

end rationalize_denominator_l94_94828


namespace calculate_paving_cost_l94_94295

theorem calculate_paving_cost
  (length : ℝ) (width : ℝ) (rate_per_sq_meter : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_rate : rate_per_sq_meter = 1200) :
  (length * width * rate_per_sq_meter = 24750) :=
by
  sorry

end calculate_paving_cost_l94_94295


namespace num_students_basketball_l94_94533

-- Definitions for conditions
def num_students_cricket : ℕ := 8
def num_students_both : ℕ := 5
def num_students_either : ℕ := 10

-- statement to be proven
theorem num_students_basketball : ∃ B : ℕ, B = 7 ∧ (num_students_either = B + num_students_cricket - num_students_both) := sorry

end num_students_basketball_l94_94533


namespace min_value_expression_l94_94716

theorem min_value_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2) : 
    (∃ c : ℝ,  c = 4 ∧ ∀ a b : ℝ, (0 < a → 0 < b → x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2 →  (3 - 2 * b)^2 / (2 * a) ≥ c)) :=
by
  sorry

end min_value_expression_l94_94716


namespace percentage_of_water_in_first_liquid_l94_94547

theorem percentage_of_water_in_first_liquid (x : ℝ) 
  (h1 : 0 < x ∧ x ≤ 1)
  (h2 : 0.35 = 0.35)
  (h3 : 10 = 10)
  (h4 : 4 = 4)
  (h5 : 0.24285714285714285 = 0.24285714285714285) :
  ((10 * x + 4 * 0.35) / (10 + 4) = 0.24285714285714285) → (x = 0.2) :=
sorry

end percentage_of_water_in_first_liquid_l94_94547


namespace jack_sugar_remaining_l94_94991

-- Define the initial amount of sugar and all daily transactions
def jack_initial_sugar : ℝ := 65
def jack_use_day1 : ℝ := 18.5
def alex_borrow_day1 : ℝ := 5.3
def jack_buy_day2 : ℝ := 30.2
def jack_use_day2 : ℝ := 12.7
def emma_give_day2 : ℝ := 4.75
def jack_buy_day3 : ℝ := 20.5
def jack_use_day3 : ℝ := 8.25
def alex_return_day3 : ℝ := 2.8
def alex_borrow_day3 : ℝ := 1.2
def jack_use_day4 : ℝ := 9.5
def olivia_give_day4 : ℝ := 6.35
def jack_use_day5 : ℝ := 10.75
def emma_borrow_day5 : ℝ := 3.1
def alex_return_day5 : ℝ := 3

-- Calculate the remaining sugar each day
def jack_sugar_day1 : ℝ := jack_initial_sugar - jack_use_day1 - alex_borrow_day1
def jack_sugar_day2 : ℝ := jack_sugar_day1 + jack_buy_day2 - jack_use_day2 + emma_give_day2
def jack_sugar_day3 : ℝ := jack_sugar_day2 + jack_buy_day3 - jack_use_day3 + alex_return_day3 - alex_borrow_day3
def jack_sugar_day4 : ℝ := jack_sugar_day3 - jack_use_day4 + olivia_give_day4
def jack_sugar_day5 : ℝ := jack_sugar_day4 - jack_use_day5 - emma_borrow_day5 + alex_return_day5

-- Final proof statement: Jack ends up with 63.3 pounds of sugar
theorem jack_sugar_remaining : jack_sugar_day5 = 63.3 := 
by sorry

end jack_sugar_remaining_l94_94991


namespace measure_of_angle_B_scalene_triangle_l94_94862

theorem measure_of_angle_B_scalene_triangle (A B C : ℝ) (hA_gt_0 : A > 0) (hB_gt_0 : B > 0) (hC_gt_0 : C > 0) 
(h_angles_sum : A + B + C = 180) (hB_eq_2A : B = 2 * A) (hC_eq_3A : C = 3 * A) : B = 60 :=
by
  sorry

end measure_of_angle_B_scalene_triangle_l94_94862


namespace solve_for_x_l94_94168

theorem solve_for_x (x : ℝ) (h : (3 * x - 15) / 4 = (x + 9) / 5) : x = 10 :=
by {
  sorry
}

end solve_for_x_l94_94168


namespace converse_x_gt_y_then_x_gt_abs_y_is_true_l94_94383

theorem converse_x_gt_y_then_x_gt_abs_y_is_true :
  (∀ x y : ℝ, (x > y) → (x > |y|)) → (∀ x y : ℝ, (x > |y|) → (x > y)) :=
by
  sorry

end converse_x_gt_y_then_x_gt_abs_y_is_true_l94_94383


namespace a4_b4_c4_double_square_l94_94122

theorem a4_b4_c4_double_square (a b c : ℤ) (h : a = b + c) : 
  a^4 + b^4 + c^4 = 2 * ((a^2 - b * c)^2) :=
by {
  sorry -- proof is not provided as per instructions
}

end a4_b4_c4_double_square_l94_94122


namespace manager_salary_calculation_l94_94655

theorem manager_salary_calculation :
  let percent_marketers := 0.60
  let salary_marketers := 50000
  let percent_engineers := 0.20
  let salary_engineers := 80000
  let percent_sales_reps := 0.10
  let salary_sales_reps := 70000
  let percent_managers := 0.10
  let total_average_salary := 75000
  let total_contribution := percent_marketers * salary_marketers + percent_engineers * salary_engineers + percent_sales_reps * salary_sales_reps
  let managers_total_contribution := total_average_salary - total_contribution
  let manager_salary := managers_total_contribution / percent_managers
  manager_salary = 220000 :=
by
  sorry

end manager_salary_calculation_l94_94655


namespace evaluate_expression_l94_94234

theorem evaluate_expression (a b c : ℚ) (h1 : c = b - 8) (h2 : b = a + 3) (h3 : a = 2) 
  (h4 : a + 1 ≠ 0) (h5 : b - 3 ≠ 0) (h6 : c + 5 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 7) / (c + 5) = 20 / 3 := by
  sorry

end evaluate_expression_l94_94234


namespace ratio_to_percentage_l94_94450

theorem ratio_to_percentage (x y : ℚ) (h : (2/3 * x) / (4/5 * y) = 5 / 6) : (5 / 6 : ℚ) * 100 = 83.33 :=
by
  sorry

end ratio_to_percentage_l94_94450


namespace opposite_of_one_sixth_l94_94731

theorem opposite_of_one_sixth : (-(1 / 6) : ℚ) = -1 / 6 := 
by
  sorry

end opposite_of_one_sixth_l94_94731


namespace number_of_integer_pairs_l94_94687

theorem number_of_integer_pairs (n : ℕ) : 
  ∃ (count : ℕ), count = 2 * n^2 + 2 * n + 1 ∧ 
  ∀ x y : ℤ, abs x + abs y ≤ n ↔
  count = 2 * n^2 + 2 * n + 1 :=
by
  sorry

end number_of_integer_pairs_l94_94687


namespace oil_truck_radius_l94_94406

/-- 
A full stationary oil tank that is a right circular cylinder has a radius of 100 feet 
and a height of 25 feet. Oil is pumped from the stationary tank to an oil truck that 
has a tank that is a right circular cylinder. The oil level dropped 0.025 feet in the stationary tank. 
The oil truck's tank has a height of 10 feet. The radius of the oil truck's tank is 5 feet. 
--/
theorem oil_truck_radius (r_stationary : ℝ) (h_stationary : ℝ) (h_truck : ℝ) 
  (Δh : ℝ) (r_truck : ℝ) 
  (h_stationary_pos : 0 < h_stationary) (h_truck_pos : 0 < h_truck) (r_stationary_pos : 0 < r_stationary) :
  r_stationary = 100 → h_stationary = 25 → Δh = 0.025 → h_truck = 10 → r_truck = 5 → 
  π * (r_stationary ^ 2) * Δh = π * (r_truck ^ 2) * h_truck :=
by 
  -- Use the conditions and perform algebra to show the equality.
  sorry

end oil_truck_radius_l94_94406


namespace michael_pets_kangaroos_l94_94718

theorem michael_pets_kangaroos :
  let total_pets := 24
  let fraction_dogs := 1 / 8
  let fraction_not_cows := 3 / 4
  let fraction_not_cats := 2 / 3
  let num_dogs := fraction_dogs * total_pets
  let num_cows := (1 - fraction_not_cows) * total_pets
  let num_cats := (1 - fraction_not_cats) * total_pets
  let num_kangaroos := total_pets - num_dogs - num_cows - num_cats
  num_kangaroos = 7 :=
by
  sorry

end michael_pets_kangaroos_l94_94718


namespace find_number_of_adults_l94_94402

variable (A : ℕ) -- Variable representing the number of adults.
def C : ℕ := 5  -- Number of children.

def meal_cost : ℕ := 3  -- Cost per meal in dollars.
def total_cost (A : ℕ) : ℕ := (A + C) * meal_cost  -- Total cost formula.

theorem find_number_of_adults 
  (h1 : meal_cost = 3)
  (h2 : total_cost A = 21)
  (h3 : C = 5) :
  A = 2 :=
sorry

end find_number_of_adults_l94_94402


namespace fraction_power_equiv_l94_94648

theorem fraction_power_equiv : (75000^4) / (25000^4) = 81 := by
  sorry

end fraction_power_equiv_l94_94648


namespace possible_values_a_l94_94629

noncomputable def setA (a : ℝ) : Set ℝ := { x | a * x + 2 = 0 }
def setB : Set ℝ := {-1, 2}

theorem possible_values_a :
  ∀ a : ℝ, setA a ⊆ setB ↔ a = -1 ∨ a = 0 ∨ a = 2 :=
by
  intro a
  sorry

end possible_values_a_l94_94629


namespace comic_story_books_proportion_l94_94599

theorem comic_story_books_proportion (x : ℕ) :
  let initial_comic_books := 140
  let initial_story_books := 100
  let borrowed_books_per_day := 4
  let comic_books_after_x_days := initial_comic_books - borrowed_books_per_day * x
  let story_books_after_x_days := initial_story_books - borrowed_books_per_day * x
  (comic_books_after_x_days = 3 * story_books_after_x_days) -> x = 20 :=
by
  sorry

end comic_story_books_proportion_l94_94599


namespace two_people_same_birthday_l94_94311

noncomputable def population : ℕ := 6000000000

noncomputable def max_age_seconds : ℕ := 150 * 366 * 24 * 60 * 60

theorem two_people_same_birthday :
  ∃ (a b : ℕ) (ha : a < population) (hb : b < population) (hab : a ≠ b),
  (∃ (t : ℕ) (ht_a : t < max_age_seconds) (ht_b : t < max_age_seconds), true) :=
by
  sorry

end two_people_same_birthday_l94_94311


namespace triangle_inequality_harmonic_mean_l94_94238

theorem triangle_inequality_harmonic_mean (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ DP DQ : ℝ, DP + DQ ≤ (2 * a * b) / (a + b) :=
by
  sorry

end triangle_inequality_harmonic_mean_l94_94238


namespace fraction_of_total_amount_l94_94654

-- Conditions
variable (p q r : ℕ)
variable (total_amount amount_r : ℕ)
variable (total_amount_eq : total_amount = 6000)
variable (amount_r_eq : amount_r = 2400)

-- Mathematical statement
theorem fraction_of_total_amount :
  amount_r / total_amount = 2 / 5 :=
by
  -- Sorry to skip the proof, as instructed
  sorry

end fraction_of_total_amount_l94_94654


namespace probability_hare_claims_not_hare_then_not_rabbit_l94_94351

noncomputable def probability_hare_given_claims : ℚ := (27 / 59)

theorem probability_hare_claims_not_hare_then_not_rabbit
  (population : ℚ) (hares : ℚ) (rabbits : ℚ)
  (belief_hare_not_hare : ℚ) (belief_hare_not_rabbit : ℚ)
  (belief_rabbit_not_hare : ℚ) (belief_rabbit_not_rabbit : ℚ) :
  population = 1 ∧ hares = 1/2 ∧ rabbits = 1/2 ∧
  belief_hare_not_hare = 1/4 ∧ belief_hare_not_rabbit = 3/4 ∧
  belief_rabbit_not_hare = 2/3 ∧ belief_rabbit_not_rabbit = 1/3 →
  (27 / 59) = probability_hare_given_claims :=
sorry

end probability_hare_claims_not_hare_then_not_rabbit_l94_94351


namespace exists_distinct_nonzero_ints_for_poly_factorization_l94_94124

theorem exists_distinct_nonzero_ints_for_poly_factorization :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ P Q : Polynomial ℤ, (P * Q = Polynomial.X * (Polynomial.X - Polynomial.C a) * 
   (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) + 1) ∧ 
   P.leadingCoeff = 1 ∧ Q.leadingCoeff = 1) :=
by
  sorry

end exists_distinct_nonzero_ints_for_poly_factorization_l94_94124


namespace greatest_num_consecutive_integers_sum_eq_36_l94_94802

theorem greatest_num_consecutive_integers_sum_eq_36 :
    ∃ a : ℤ, ∃ N : ℕ, N > 0 ∧ (N = 9) ∧ (N * (2 * a + N - 1) = 72) :=
sorry

end greatest_num_consecutive_integers_sum_eq_36_l94_94802


namespace difference_between_two_numbers_l94_94114

theorem difference_between_two_numbers :
  ∃ a b : ℕ, 
    a + 5 * b = 23405 ∧ 
    (∃ b' : ℕ, b = 10 * b' + 5 ∧ b' = 5 * a) ∧ 
    5 * b - a = 21600 :=
by {
  sorry
}

end difference_between_two_numbers_l94_94114


namespace problem_statement_l94_94254

theorem problem_statement : ∃ n : ℤ, 0 < n ∧ (1 / 3 + 1 / 4 + 1 / 8 + 1 / n : ℚ).den = 1 ∧ ¬ n > 96 := 
by 
  sorry

end problem_statement_l94_94254


namespace trigonometric_identity_1_l94_94242

theorem trigonometric_identity_1 :
  ( (Real.sqrt 3 * Real.sin (-1200 * Real.pi / 180)) / (Real.tan (11 * Real.pi / 3)) 
  - Real.cos (585 * Real.pi / 180) * Real.tan (-37 * Real.pi / 4) = (Real.sqrt 3 / 2) - (Real.sqrt 2 / 2) ) :=
by
  sorry

end trigonometric_identity_1_l94_94242


namespace price_per_kg_of_fruits_l94_94524

theorem price_per_kg_of_fruits (mangoes apples oranges : ℕ) (total_amount : ℕ)
  (h1 : mangoes = 400)
  (h2 : apples = 2 * mangoes)
  (h3 : oranges = mangoes + 200)
  (h4 : total_amount = 90000) :
  (total_amount / (mangoes + apples + oranges) = 50) :=
by
  sorry

end price_per_kg_of_fruits_l94_94524


namespace simplify_expr1_simplify_expr2_l94_94633

-- Define the variables a and b
variables (a b : ℝ)

-- First problem: simplify 2a^2 - 3a^3 + 5a + 2a^3 - a^2 to a^2 - a^3 + 5a
theorem simplify_expr1 : 2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a :=
  by sorry

-- Second problem: simplify (2 / 3) (2 * a - b) + 2 (b - 2 * a) - 3 (2 * a - b) - (4 / 3) (b - 2 * a) to -6 * a + 3 * b
theorem simplify_expr2 : 
  (2 / 3) * (2 * a - b) + 2 * (b - 2 * a) - 3 * (2 * a - b) - (4 / 3) * (b - 2 * a) = -6 * a + 3 * b :=
  by sorry

end simplify_expr1_simplify_expr2_l94_94633


namespace gunny_bag_capacity_in_tons_l94_94770

def ton_to_pounds := 2200
def pound_to_ounces := 16
def packets := 1760
def packet_weight_pounds := 16
def packet_weight_ounces := 4

theorem gunny_bag_capacity_in_tons :
  ((packets * (packet_weight_pounds + (packet_weight_ounces / pound_to_ounces))) / ton_to_pounds) = 13 :=
sorry

end gunny_bag_capacity_in_tons_l94_94770


namespace triangle_angle_sum_l94_94513

theorem triangle_angle_sum (x : ℝ) :
    let angle1 : ℝ := 40
    let angle2 : ℝ := 4 * x
    let angle3 : ℝ := 3 * x
    angle1 + angle2 + angle3 = 180 -> x = 20 := 
sorry

end triangle_angle_sum_l94_94513


namespace kanul_initial_amount_l94_94125

-- Definition based on the problem conditions
def spent_on_raw_materials : ℝ := 3000
def spent_on_machinery : ℝ := 2000
def spent_on_labor : ℝ := 1000
def percent_spent : ℝ := 0.15

-- Definition of the total amount initially had by Kanul
def total_amount_initial (X : ℝ) : Prop :=
  spent_on_raw_materials + spent_on_machinery + percent_spent * X + spent_on_labor = X

-- Theorem stating the conclusion based on the given conditions
theorem kanul_initial_amount : ∃ X : ℝ, total_amount_initial X ∧ X = 7058.82 :=
by {
  sorry
}

end kanul_initial_amount_l94_94125


namespace avg_ABC_l94_94444

variables (A B C : Set ℕ) -- Sets of people
variables (a b c : ℕ) -- Numbers of people in sets A, B, and C respectively
variables (sum_A sum_B sum_C : ℕ) -- Sums of the ages of people in sets A, B, and C respectively

-- Given conditions
axiom avg_A : sum_A / a = 30
axiom avg_B : sum_B / b = 20
axiom avg_C : sum_C / c = 45

axiom avg_AB : (sum_A + sum_B) / (a + b) = 25
axiom avg_AC : (sum_A + sum_C) / (a + c) = 40
axiom avg_BC : (sum_B + sum_C) / (b + c) = 32

theorem avg_ABC : (sum_A + sum_B + sum_C) / (a + b + c) = 35 :=
by
  sorry

end avg_ABC_l94_94444


namespace ron_chocolate_bar_cost_l94_94781

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l94_94781


namespace triangle_angle_A_l94_94013

theorem triangle_angle_A (a c : ℝ) (C A : ℝ) 
  (h1 : a = 4 * Real.sqrt 3)
  (h2 : c = 12)
  (h3 : C = Real.pi / 3)
  (h4 : a < c) :
  A = Real.pi / 6 :=
sorry

end triangle_angle_A_l94_94013


namespace sqrt8_same_type_as_sqrt2_l94_94607

def same_type_sqrt_2 (x : Real) : Prop := ∃ k : Real, k * Real.sqrt 2 = x

theorem sqrt8_same_type_as_sqrt2 : same_type_sqrt_2 (Real.sqrt 8) :=
  sorry

end sqrt8_same_type_as_sqrt2_l94_94607


namespace find_smallest_c_l94_94029

/-- Let a₀, a₁, ... and b₀, b₁, ... be geometric sequences with common ratios rₐ and r_b, 
respectively, such that ∑ i=0 ∞ aᵢ = ∑ i=0 ∞ bᵢ = 1 and 
(∑ i=0 ∞ aᵢ²)(∑ i=0 ∞ bᵢ²) = ∑ i=0 ∞ aᵢbᵢ. Prove that a₀ < 4/3 -/
theorem find_smallest_c (r_a r_b : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∑' n, a n = 1)
  (h2 : ∑' n, b n = 1)
  (h3 : (∑' n, (a n)^2) * (∑' n, (b n)^2) = ∑' n, (a n) * (b n)) :
  a 0 < 4 / 3 := by
  sorry

end find_smallest_c_l94_94029


namespace grove_town_fall_expenditure_l94_94837

-- Define the expenditures at the end of August and November
def expenditure_end_of_august : ℝ := 3.0
def expenditure_end_of_november : ℝ := 5.5

-- Define the spending during fall months (September, October, November)
def spending_during_fall_months : ℝ := 2.5

-- Statement to be proved
theorem grove_town_fall_expenditure :
  expenditure_end_of_november - expenditure_end_of_august = spending_during_fall_months :=
by
  sorry

end grove_town_fall_expenditure_l94_94837


namespace ned_long_sleeve_shirts_l94_94261

-- Define the conditions
def total_shirts_washed_before_school : ℕ := 29
def short_sleeve_shirts : ℕ := 9
def unwashed_shirts : ℕ := 1

-- Define the proof problem
theorem ned_long_sleeve_shirts (total_shirts_washed_before_school short_sleeve_shirts unwashed_shirts: ℕ) : 
(total_shirts_washed_before_school - unwashed_shirts - short_sleeve_shirts) = 19 :=
by
  -- It is given: 29 total shirts - 1 unwashed shirt = 28 washed shirts
  -- Out of the 28 washed shirts, 9 are short sleeve shirts
  -- Therefore, Ned washed 28 - 9 = 19 long sleeve shirts
  sorry

end ned_long_sleeve_shirts_l94_94261


namespace stack_height_difference_l94_94575

theorem stack_height_difference :
  ∃ S : ℕ,
    (7 + S + (S - 6) + (S + 4) + 2 * S = 55) ∧ (S - 7 = 3) := 
by 
  sorry

end stack_height_difference_l94_94575


namespace find_expression_l94_94822

theorem find_expression (x : ℝ) (h : (1 / Real.cos (2022 * x)) + Real.tan (2022 * x) = 1 / 2022) :
  (1 / Real.cos (2022 * x)) - Real.tan (2022 * x) = 2022 :=
by
  sorry

end find_expression_l94_94822


namespace range_of_a_l94_94471

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 →
    (x + 3 + 2 * Real.sin θ * Real.cos θ) ^ 2 +
    (x + a * Real.sin θ + a * Real.cos θ) ^ 2 ≥ 1 / 8) ↔
  (a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6) :=
by
  sorry

end range_of_a_l94_94471


namespace rectangle_width_l94_94364

theorem rectangle_width (L W : ℝ) (h1 : 2 * (L + W) = 16) (h2 : W = L + 2) : W = 5 :=
by
  sorry

end rectangle_width_l94_94364


namespace sample_average_l94_94338

theorem sample_average (x : ℝ) 
  (h1 : (1 + 3 + 2 + 5 + x) / 5 = 3) : x = 4 := 
by 
  sorry

end sample_average_l94_94338


namespace min_g_l94_94275

noncomputable def f (a m x : ℝ) := m + Real.log x / Real.log a -- definition of f(x) = m + logₐ(x)

-- Given conditions
variables (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
variables (m : ℝ)
axiom h_f8 : f a m 8 = 2
axiom h_f1 : f a m 1 = -1

-- Derived expressions
noncomputable def g (x : ℝ) := 2 * f a m x - f a m (x - 1)

-- Theorem statement
theorem min_g : ∃ (x : ℝ), x > 1 ∧ g a m x = 1 ∧ ∀ x' > 1, g a m x' ≥ 1 :=
sorry

end min_g_l94_94275


namespace MrKozelGarden_l94_94318

theorem MrKozelGarden :
  ∀ (x y : ℕ), 
  (y = 3 * x + 1) ∧ (y = 4 * (x - 1)) → (x = 5 ∧ y = 16) := 
by
  intros x y h
  sorry

end MrKozelGarden_l94_94318


namespace remaining_water_at_end_of_hike_l94_94131

-- Define conditions
def initial_water : ℝ := 9
def hike_length : ℝ := 7
def hike_duration : ℝ := 2
def leak_rate : ℝ := 1
def drink_rate_6_miles : ℝ := 0.6666666666666666
def drink_last_mile : ℝ := 2

-- Define the question and answer
def remaining_water (initial: ℝ) (duration: ℝ) (leak: ℝ) (drink6: ℝ) (drink_last: ℝ) : ℝ :=
  initial - ((drink6 * 6) + drink_last + (leak * duration))

-- Theorem stating the proof problem 
theorem remaining_water_at_end_of_hike :
  remaining_water initial_water hike_duration leak_rate drink_rate_6_miles drink_last_mile = 1 :=
by
  sorry

end remaining_water_at_end_of_hike_l94_94131


namespace rabbit_speed_final_result_l94_94447

def rabbit_speed : ℕ := 45

def double_speed (speed : ℕ) : ℕ := speed * 2

def add_four (n : ℕ) : ℕ := n + 4

def final_operation : ℕ := double_speed (add_four (double_speed rabbit_speed))

theorem rabbit_speed_final_result : final_operation = 188 := 
by
  sorry

end rabbit_speed_final_result_l94_94447


namespace smallest_fraction_numerator_l94_94240

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end smallest_fraction_numerator_l94_94240


namespace repeating_decimals_product_l94_94036

-- Definitions to represent the conditions
def repeating_decimal_03_as_frac : ℚ := 1 / 33
def repeating_decimal_36_as_frac : ℚ := 4 / 11

-- The statement to be proven
theorem repeating_decimals_product : (repeating_decimal_03_as_frac * repeating_decimal_36_as_frac) = (4 / 363) :=
by {
  sorry
}

end repeating_decimals_product_l94_94036


namespace area_of_shaded_region_l94_94303

-- Definitions of given conditions
def octagon_side_length : ℝ := 5
def arc_radius : ℝ := 4

-- Theorem statement
theorem area_of_shaded_region : 
  let octagon_area := 50
  let sectors_area := 16 * Real.pi
  octagon_area - sectors_area = 50 - 16 * Real.pi :=
by
  sorry

end area_of_shaded_region_l94_94303


namespace find_smaller_number_l94_94846

theorem find_smaller_number (a b : ℕ) (h_ratio : 11 * a = 7 * b) (h_diff : b = a + 16) : a = 28 :=
by
  sorry

end find_smaller_number_l94_94846


namespace p_sufficient_but_not_necessary_for_q_l94_94175

def condition_p (x : ℝ) : Prop := abs (x - 1) < 2
def condition_q (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

theorem p_sufficient_but_not_necessary_for_q : 
  (∀ x, condition_p x → condition_q x) ∧ 
  ¬ (∀ x, condition_q x → condition_p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l94_94175


namespace calculate_liquids_l94_94583

def water_ratio := 60 -- mL of water for every 400 mL of flour
def milk_ratio := 80 -- mL of milk for every 400 mL of flour
def flour_ratio := 400 -- mL of flour in one portion

def flour_quantity := 1200 -- mL of flour available

def number_of_portions := flour_quantity / flour_ratio

def total_water := number_of_portions * water_ratio
def total_milk := number_of_portions * milk_ratio

theorem calculate_liquids :
  total_water = 180 ∧ total_milk = 240 :=
by
  -- Proof will be filled in here. Skipping with sorry for now.
  sorry

end calculate_liquids_l94_94583


namespace problem_statement_l94_94565

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Nontrivial α]

def is_monotone_increasing (f : α → α) (s : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem problem_statement (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : ∀ x, f (-x) = -f (x + 4))
  (h2 : is_monotone_increasing f {x | x > 2})
  (hx1 : x1 < 2) (hx2 : 2 < x2) (h_sum : x1 + x2 < 4) :
  f (x1) + f (x2) < 0 :=
sorry

end problem_statement_l94_94565


namespace sequence_geometric_and_formula_l94_94929

theorem sequence_geometric_and_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  (∀ n, a n + 1 = 2 ^ n) ∧ (a n = 2 ^ n - 1) :=
sorry

end sequence_geometric_and_formula_l94_94929


namespace sqrt_prod_plus_one_equals_341_l94_94666

noncomputable def sqrt_prod_plus_one : ℕ :=
  Nat.sqrt ((20 * 19 * 18 * 17) + 1)

theorem sqrt_prod_plus_one_equals_341 :
  sqrt_prod_plus_one = 341 := 
by
  sorry

end sqrt_prod_plus_one_equals_341_l94_94666


namespace anne_distance_diff_l94_94683

def track_length := 300
def min_distance := 100

-- Define distances functions as described
def distance_AB (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Beth over time 
def distance_AC (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Carmen over time 

theorem anne_distance_diff (Anne_speed Beth_speed Carmen_speed : ℝ) 
  (hneA : Anne_speed ≠ Beth_speed)
  (hneC : Anne_speed ≠ Carmen_speed) :
  ∃ α ≥ 0, min_distance ≤ distance_AB α ∧ min_distance ≤ distance_AC α :=
sorry

end anne_distance_diff_l94_94683


namespace rain_probability_l94_94887

-- Define the probability of rain on any given day, number of trials, and specific number of successful outcomes.
def prob_rain_each_day : ℚ := 1/5
def num_days : ℕ := 10
def num_rainy_days : ℕ := 3

-- Define the binomial probability mass function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Statement to prove
theorem rain_probability : binomial_prob num_days num_rainy_days prob_rain_each_day = 1966080 / 9765625 :=
by
  sorry

end rain_probability_l94_94887


namespace max_value_of_2a_plus_b_l94_94237

variable (a b : ℝ)

def cond1 := 4 * a + 3 * b ≤ 10
def cond2 := 3 * a + 5 * b ≤ 11

theorem max_value_of_2a_plus_b : 
  cond1 a b → 
  cond2 a b → 
  2 * a + b ≤ 48 / 11 := 
by 
  sorry

end max_value_of_2a_plus_b_l94_94237


namespace remainder_2_power_404_l94_94325

theorem remainder_2_power_404 (y : ℕ) (h_y : y = 2^101) :
  (2^404 + 404) % (2^203 + 2^101 + 1) = 403 := by
sorry

end remainder_2_power_404_l94_94325


namespace sum_D_E_correct_sum_of_all_possible_values_of_D_E_l94_94930

theorem sum_D_E_correct :
  ∀ (D E : ℕ), (D < 10) → (E < 10) →
  (∃ k : ℕ, (10^8 * D + 4650000 + 1000 * E + 32) = 7 * k) →
  D + E = 1 ∨ D + E = 8 ∨ D + E = 15 :=
by sorry

theorem sum_of_all_possible_values_of_D_E :
  (1 + 8 + 15) = 24 :=
by norm_num

end sum_D_E_correct_sum_of_all_possible_values_of_D_E_l94_94930


namespace neither_sufficient_nor_necessary_l94_94882

variable {a b : ℝ}

theorem neither_sufficient_nor_necessary (hab_ne_zero : a * b ≠ 0) :
  ¬ (a * b > 1 → a > (1 / b)) ∧ ¬ (a > (1 / b) → a * b > 1) :=
sorry

end neither_sufficient_nor_necessary_l94_94882


namespace length_rest_of_body_l94_94164

theorem length_rest_of_body (height legs head arms rest_of_body : ℝ) 
  (hlegs : legs = (1/3) * height)
  (hhead : head = (1/4) * height)
  (harms : arms = (1/5) * height)
  (htotal : height = 180)
  (hr: rest_of_body = height - (legs + head + arms)) : 
  rest_of_body = 39 :=
by
  -- proof is not required
  sorry

end length_rest_of_body_l94_94164


namespace limiting_reactant_and_products_l94_94314

def balanced_reaction 
  (al_moles : ℕ) (h2so4_moles : ℕ) 
  (al2_so4_3_moles : ℕ) (h2_moles : ℕ) : Prop :=
  2 * al_moles >= 0 ∧ 3 * h2so4_moles >= 0 ∧ 
  al_moles = 2 ∧ h2so4_moles = 3 ∧ 
  al2_so4_3_moles = 1 ∧ h2_moles = 3 ∧ 
  (2 : ℕ) * al_moles + (3 : ℕ) * h2so4_moles = 2 * 2 + 3 * 3

theorem limiting_reactant_and_products :
  balanced_reaction 2 3 1 3 :=
by {
  -- Here we would provide the proof based on the conditions and balances provided in the problem statement.
  sorry
}

end limiting_reactant_and_products_l94_94314


namespace coin_flips_137_l94_94805

-- Definitions and conditions
def steph_transformation_heads (x : ℤ) : ℤ := 2 * x - 1
def steph_transformation_tails (x : ℤ) : ℤ := (x + 1) / 2
def jeff_transformation_heads (y : ℤ) : ℤ := y + 8
def jeff_transformation_tails (y : ℤ) : ℤ := y - 3

-- The problem statement
theorem coin_flips_137
  (a b : ℤ)
  (h₁ : a - b = 7)
  (h₂ : 8 * a - 3 * b = 381)
  (steph_initial jeff_initial : ℤ)
  (h₃ : steph_initial = 4)
  (h₄ : jeff_initial = 4) : a + b = 137 := 
by
  sorry

end coin_flips_137_l94_94805


namespace determinant_value_l94_94509

theorem determinant_value (t₁ t₂ : ℤ)
    (h₁ : t₁ = 2 * 3 + 3 * 5)
    (h₂ : t₂ = 5) :
    Matrix.det ![
      ![1, -1, t₁],
      ![0, 1, -1],
      ![-1, t₂, -6]
    ] = 14 := by
  rw [h₁, h₂]
  -- Actual proof would go here
  sorry

end determinant_value_l94_94509


namespace negation_proof_l94_94102

theorem negation_proof :
  (∃ x₀ : ℝ, x₀ < 2) → ¬ (∀ x : ℝ, x < 2) :=
by
  sorry

end negation_proof_l94_94102


namespace total_crayons_is_12_l94_94382

-- Definitions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Goal to prove
theorem total_crayons_is_12 : initial_crayons + added_crayons = 12 :=
by
  sorry

end total_crayons_is_12_l94_94382


namespace remainder_when_6n_divided_by_4_l94_94793

theorem remainder_when_6n_divided_by_4 (n : ℤ) (h : n % 4 = 1) : 6 * n % 4 = 2 := by
  sorry

end remainder_when_6n_divided_by_4_l94_94793


namespace sum_of_five_consecutive_squares_not_perfect_square_l94_94659

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ (k : ℤ), k^2 = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 := 
by
  sorry

end sum_of_five_consecutive_squares_not_perfect_square_l94_94659


namespace quadratic_vertex_properties_l94_94267

theorem quadratic_vertex_properties (a : ℝ) (x1 x2 y1 y2 : ℝ) (h_ax : a ≠ 0) (h_sum : x1 + x2 = 2) (h_order : x1 < x2) (h_value : y1 > y2) :
  a < -2 / 5 :=
sorry

end quadratic_vertex_properties_l94_94267


namespace fair_eight_sided_die_probability_l94_94277

def prob_at_least_seven_at_least_four_times (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem fair_eight_sided_die_probability : prob_at_least_seven_at_least_four_times 5 4 (1 / 4) + (1 / 4) ^ 5 = 1 / 64 :=
by
  sorry

end fair_eight_sided_die_probability_l94_94277


namespace stocks_higher_price_l94_94711

theorem stocks_higher_price
  (total_stocks : ℕ)
  (percent_increase : ℝ)
  (H L : ℝ)
  (H_eq : H = 1.35 * L)
  (sum_eq : H + L = 4200)
  (percent_increase_eq : percent_increase = 0.35)
  (total_stocks_eq : ↑total_stocks = 4200) :
  total_stocks = 2412 :=
by 
  sorry

end stocks_higher_price_l94_94711


namespace sample_size_l94_94767

theorem sample_size (k n : ℕ) (h_ratio : 3 * n / (3 + 4 + 7) = 9) : n = 42 :=
by
  sorry

end sample_size_l94_94767


namespace pallets_of_paper_cups_l94_94624

theorem pallets_of_paper_cups (total_pallets paper_towels tissues paper_plates : ℕ) 
  (H1 : total_pallets = 20) 
  (H2 : paper_towels = total_pallets / 2)
  (H3 : tissues = total_pallets / 4)
  (H4 : paper_plates = total_pallets / 5) : 
  total_pallets - paper_towels - tissues - paper_plates = 1 := 
  by
    sorry

end pallets_of_paper_cups_l94_94624


namespace airplane_speeds_l94_94134

theorem airplane_speeds (v : ℝ) 
  (h1 : 2.5 * v + 2.5 * 250 = 1625) : 
  v = 400 := 
sorry

end airplane_speeds_l94_94134


namespace find_constants_l94_94874

theorem find_constants (a b c d : ℚ) :
  (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) =
  18 * x^6 - 2 * x^5 + 16 * x^4 - (28 / 3) * x^3 + (8 / 3) * x^2 - 4 * x + 2 →
  a = 3 ∧ b = -1 / 3 ∧ c = 14 / 9 :=
by
  sorry

end find_constants_l94_94874


namespace intersection_with_complement_N_l94_94361

open Set Real

def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def N : Set ℝ := {x | 0 < x ∧ x < 2}
def complement_N : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

theorem intersection_with_complement_N : M ∩ complement_N = Ico 2 3 :=
by {
  sorry
}

end intersection_with_complement_N_l94_94361


namespace john_has_500_dollars_l94_94178

-- Define the initial amount and the condition
def initial_amount : ℝ := 1600
def condition (spent : ℝ) : Prop := (1600 - spent) = (spent - 600)

-- The final amount of money John still has
def final_amount (spent : ℝ) : ℝ := initial_amount - spent

-- The main theorem statement
theorem john_has_500_dollars : ∃ (spent : ℝ), condition spent ∧ final_amount spent = 500 :=
by
  sorry

end john_has_500_dollars_l94_94178


namespace casey_pumping_time_l94_94921

structure PlantRow :=
  (rows : ℕ) (plants_per_row : ℕ) (water_per_plant : ℚ)

structure Animal :=
  (count : ℕ) (water_per_animal : ℚ)

def morning_pump_rate := 3 -- gallons per minute
def afternoon_pump_rate := 5 -- gallons per minute

def corn := PlantRow.mk 4 15 0.5
def pumpkin := PlantRow.mk 3 10 0.8
def pigs := Animal.mk 10 4
def ducks := Animal.mk 20 0.25
def cows := Animal.mk 5 8

def total_water_needed_for_plants (corn pumpkin : PlantRow) : ℚ :=
  (corn.rows * corn.plants_per_row * corn.water_per_plant) +
  (pumpkin.rows * pumpkin.plants_per_row * pumpkin.water_per_plant)

def total_water_needed_for_animals (pigs ducks cows : Animal) : ℚ :=
  (pigs.count * pigs.water_per_animal) +
  (ducks.count * ducks.water_per_animal) +
  (cows.count * cows.water_per_animal)

def time_to_pump (total_water pump_rate : ℚ) : ℚ :=
  total_water / pump_rate

theorem casey_pumping_time :
  let total_water_plants := total_water_needed_for_plants corn pumpkin
  let total_water_animals := total_water_needed_for_animals pigs ducks cows
  let time_morning := time_to_pump total_water_plants morning_pump_rate
  let time_afternoon := time_to_pump total_water_animals afternoon_pump_rate
  time_morning + time_afternoon = 35 := by
sorry

end casey_pumping_time_l94_94921


namespace greatest_perimeter_isosceles_triangle_l94_94049

theorem greatest_perimeter_isosceles_triangle :
  let base := 12
  let height := 15
  let segments := 6
  let max_perimeter := 32.97
  -- Assuming division such that each of the 6 pieces is of equal area,
  -- the greatest perimeter among these pieces to the nearest hundredth is:
  (∀ (base height segments : ℝ), base = 12 ∧ height = 15 ∧ segments = 6 → 
   max_perimeter = 32.97) :=
by
  sorry

end greatest_perimeter_isosceles_triangle_l94_94049


namespace largest_angle_in_hexagon_l94_94877

-- Defining the conditions
variables (A B x y : ℝ)
variables (C D E F : ℝ)
variable (sum_of_angles_in_hexagon : ℝ) 

-- Given conditions
def condition1 : A = 100 := by sorry
def condition2 : B = 120 := by sorry
def condition3 : C = x := by sorry
def condition4 : D = x := by sorry
def condition5 : E = (2 * x + y) / 3 + 30 := by sorry
def condition6 : 100 + 120 + C + D + E + F = 720 := by sorry

-- Statement to prove
theorem largest_angle_in_hexagon :
  ∃ (largest_angle : ℝ), largest_angle = max A (max B (max C (max D (max E F)))) ∧ largest_angle = 147.5 := sorry

end largest_angle_in_hexagon_l94_94877


namespace average_pastries_per_day_l94_94194

-- Conditions
def pastries_on_monday := 2

def pastries_on_day (n : ℕ) : ℕ :=
  pastries_on_monday + n

def total_pastries_in_week : ℕ :=
  List.sum (List.map pastries_on_day (List.range 7))

def number_of_days_in_week : ℕ := 7

-- Theorem to prove
theorem average_pastries_per_day : (total_pastries_in_week / number_of_days_in_week) = 5 :=
by
  sorry

end average_pastries_per_day_l94_94194


namespace unique_prime_solution_l94_94925

-- Define the variables and properties
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the proof goal
theorem unique_prime_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hp_pos : 0 < p) (hq_pos : 0 < q) :
  p^2 - q^3 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end unique_prime_solution_l94_94925


namespace area_of_triangle_BP_Q_is_24_l94_94123

open Real

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem area_of_triangle_BP_Q_is_24
  (A B C P H Q : ℝ × ℝ)
  (h_triangle_ABC_right : C.1 = 0 ∧ C.2 = 0 ∧ B.2 = 0 ∧ A.2 ≠ 0)
  (h_BC_diameter : distance B C = 26)
  (h_tangent_AP : distance P B = distance P C ∧ P ≠ C)
  (h_PH_perpendicular_BC : P.1 = H.1 ∧ H.2 = 0)
  (h_PH_intersects_AB_at_Q : H.1 = Q.1 ∧ Q.2 ≠ 0)
  (h_BH_CH_ratio : 4 * distance B H = 9 * distance C H)
  : triangle_area B P Q = 24 :=
sorry

end area_of_triangle_BP_Q_is_24_l94_94123


namespace concurrent_segments_unique_solution_l94_94201

theorem concurrent_segments_unique_solution (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  4^c - 1 = (2^a - 1) * (2^b - 1) ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
by
  sorry

end concurrent_segments_unique_solution_l94_94201


namespace son_present_age_l94_94700

-- Definitions
variables (S M : ℕ)
-- Conditions
def age_diff : Prop := M = S + 22
def future_age_condition : Prop := M + 2 = 2 * (S + 2)

-- Theorem statement with proof placeholder
theorem son_present_age (H1 : age_diff S M) (H2 : future_age_condition S M) : S = 20 :=
by sorry

end son_present_age_l94_94700


namespace find_amount_of_alcohol_l94_94411

theorem find_amount_of_alcohol (A W : ℝ) (h₁ : A / W = 4 / 3) (h₂ : A / (W + 7) = 4 / 5) : A = 14 := 
sorry

end find_amount_of_alcohol_l94_94411


namespace tea_consumption_eq1_tea_consumption_eq2_l94_94377

theorem tea_consumption_eq1 (k : ℝ) (w_sunday t_sunday w_wednesday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : w_wednesday = 4) : 
  t_wednesday = 6 := 
  by sorry

theorem tea_consumption_eq2 (k : ℝ) (w_sunday t_sunday t_thursday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : t_thursday = 2) : 
  w_thursday = 12 := 
  by sorry

end tea_consumption_eq1_tea_consumption_eq2_l94_94377


namespace banknotes_combination_l94_94199

theorem banknotes_combination (a b c d : ℕ) (h : a + b + c + d = 10) (h_val : 2000 * a + 1000 * b + 500 * c + 200 * d = 5000) :
  (a = 0 ∧ b = 0 ∧ c = 10 ∧ d = 0) ∨ 
  (a = 1 ∧ b = 0 ∧ c = 4 ∧ d = 5) ∨ 
  (a = 0 ∧ b = 3 ∧ c = 2 ∧ d = 5) :=
by
  sorry

end banknotes_combination_l94_94199


namespace order_of_scores_l94_94890

variables (E L T N : ℝ)

-- Conditions
axiom Lana_condition_1 : L ≠ T
axiom Lana_condition_2 : L ≠ N
axiom Lana_condition_3 : T ≠ N

axiom Tom_condition : ∃ L' E', L' ≠ T ∧ E' > L' ∧ E' ≠ T ∧ E' ≠ L' 

axiom Nina_condition : N < E

-- Proof statement
theorem order_of_scores :
  N < L ∧ L < T :=
sorry

end order_of_scores_l94_94890


namespace spend_on_laundry_detergent_l94_94146

def budget : ℕ := 60
def price_shower_gel : ℕ := 4
def num_shower_gels : ℕ := 4
def price_toothpaste : ℕ := 3
def remaining_budget : ℕ := 30

theorem spend_on_laundry_detergent : 
  (budget - remaining_budget) = (num_shower_gels * price_shower_gel + price_toothpaste) + 11 := 
by
  sorry

end spend_on_laundry_detergent_l94_94146


namespace min_value_of_inverse_sum_l94_94632

theorem min_value_of_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : 
  ∃ c : ℝ, c = 4 + 2 * Real.sqrt 3 ∧ ∀x : ℝ, (x = (1 / a + 1 / b)) → x ≥ c :=
by
  sorry

end min_value_of_inverse_sum_l94_94632


namespace cost_price_of_article_l94_94419

theorem cost_price_of_article (x : ℝ) :
  (86 - x = x - 42) → x = 64 :=
by
  intro h
  sorry

end cost_price_of_article_l94_94419


namespace number_of_milkshakes_l94_94492

-- Define the amounts and costs
def initial_money : ℕ := 132
def remaining_money : ℕ := 70
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 5
def hamburgers_bought : ℕ := 8

-- Defining the money spent calculations
def hamburgers_spent : ℕ := hamburgers_bought * hamburger_cost
def total_spent : ℕ := initial_money - remaining_money
def milkshake_spent : ℕ := total_spent - hamburgers_spent

-- The final theorem to prove
theorem number_of_milkshakes : (milkshake_spent / milkshake_cost) = 6 :=
by
  sorry

end number_of_milkshakes_l94_94492


namespace first_valve_fill_time_l94_94898

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end first_valve_fill_time_l94_94898


namespace sequence_sum_l94_94473

variable (P Q R S T U V : ℤ)
variable (hR : R = 7)
variable (h1 : P + Q + R = 36)
variable (h2 : Q + R + S = 36)
variable (h3 : R + S + T = 36)
variable (h4 : S + T + U = 36)
variable (h5 : T + U + V = 36)

theorem sequence_sum (P Q R S T U V : ℤ)
  (hR : R = 7)
  (h1 : P + Q + R = 36)
  (h2 : Q + R + S = 36)
  (h3 : R + S + T = 36)
  (h4 : S + T + U = 36)
  (h5 : T + U + V = 36) :
  P + V = 29 := 
sorry

end sequence_sum_l94_94473


namespace problem_expression_value_l94_94229

variable (m n p q : ℝ)
variable (h1 : m + n = 0) (h2 : m / n = -1)
variable (h3 : p * q = 1) (h4 : m ≠ n)

theorem problem_expression_value : 
  (m + n) / m + 2 * p * q - m / n = 3 :=
by sorry

end problem_expression_value_l94_94229


namespace diet_soda_count_l94_94007

theorem diet_soda_count (D : ℕ) (h1 : 81 = D + 21) : D = 60 := by
  sorry

end diet_soda_count_l94_94007


namespace sum_of_six_selected_primes_is_even_l94_94941

noncomputable def prob_sum_even_when_selecting_six_primes : ℚ := 
  let first_twenty_primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
  let num_ways_to_choose_6_without_even_sum := Nat.choose 19 6
  let total_num_ways_to_choose_6 := Nat.choose 20 6
  num_ways_to_choose_6_without_even_sum / total_num_ways_to_choose_6

theorem sum_of_six_selected_primes_is_even : 
  prob_sum_even_when_selecting_six_primes = 354 / 505 := 
sorry

end sum_of_six_selected_primes_is_even_l94_94941


namespace eliminate_xy_l94_94051

variable {R : Type*} [Ring R]

theorem eliminate_xy
  (x y a b c : R)
  (h1 : a = x + y)
  (h2 : b = x^3 + y^3)
  (h3 : c = x^5 + y^5) :
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) :=
sorry

end eliminate_xy_l94_94051


namespace loan_balance_formula_l94_94105

variable (c V : ℝ) (t n : ℝ)

theorem loan_balance_formula :
  V = c / (1 + t)^(3 * n) →
  n = (Real.log (c / V)) / (3 * Real.log (1 + t)) :=
by sorry

end loan_balance_formula_l94_94105


namespace chris_money_left_over_l94_94581

-- Define the constants based on the conditions given in the problem.
def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def earnings_per_hour : ℕ := 8
def hours_worked : ℕ := 9

-- Define the intermediary results based on the problem's conditions.
def total_cost : ℕ := video_game_cost + candy_cost
def total_earnings : ℕ := earnings_per_hour * hours_worked

-- Define the final result to be proven.
def total_leftover : ℕ := total_earnings - total_cost

-- State the proof problem as a Lean theorem.
theorem chris_money_left_over : total_leftover = 7 := by
  sorry

end chris_money_left_over_l94_94581


namespace find_coordinates_l94_94214

def point_in_fourth_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 < 0
def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.2| = d
def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.1| = d

theorem find_coordinates :
  ∃ P : ℝ × ℝ, point_in_fourth_quadrant P ∧ distance_to_x_axis P 2 ∧ distance_to_y_axis P 5 ∧ P = (5, -2) :=
by
  sorry

end find_coordinates_l94_94214


namespace total_carrots_computation_l94_94448

-- Definitions
def initial_carrots : ℕ := 19
def thrown_out_carrots : ℕ := 4
def next_day_carrots : ℕ := 46

def total_carrots (c1 c2 t : ℕ) : ℕ := (c1 - t) + c2

-- The statement to prove
theorem total_carrots_computation :
  total_carrots initial_carrots next_day_carrots thrown_out_carrots = 61 :=
by sorry

end total_carrots_computation_l94_94448


namespace people_got_off_train_l94_94249

theorem people_got_off_train (initial_people : ℕ) (people_left : ℕ) (people_got_off : ℕ) 
  (h1 : initial_people = 48) 
  (h2 : people_left = 31) 
  : people_got_off = 17 := by
  sorry

end people_got_off_train_l94_94249


namespace complete_work_together_in_days_l94_94491

/-
p is 60% more efficient than q.
p can complete the work in 26 days.
Prove that p and q together will complete the work in approximately 18.57 days.
-/

noncomputable def work_together_days (p_efficiency q_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  let p_work_rate := 1 / p_days
  let q_work_rate := q_efficiency / p_efficiency * p_work_rate
  let combined_work_rate := p_work_rate + q_work_rate
  1 / combined_work_rate

theorem complete_work_together_in_days :
  ∀ (p_efficiency q_efficiency p_days : ℝ),
  p_efficiency = 1 ∧ q_efficiency = 0.4 ∧ p_days = 26 →
  abs (work_together_days p_efficiency q_efficiency p_days - 18.57) < 0.01 := by
  intros p_efficiency q_efficiency p_days
  rintro ⟨heff_p, heff_q, hdays_p⟩
  simp [heff_p, heff_q, hdays_p, work_together_days]
  sorry

end complete_work_together_in_days_l94_94491


namespace resting_time_is_thirty_l94_94960

-- Defining the conditions as Lean 4 definitions
def speed := 10 -- miles per hour
def time_first_part := 30 -- minutes
def distance_second_part := 15 -- miles
def distance_third_part := 20 -- miles
def total_time := 270 -- minutes

-- Function to convert hours to minutes
def hours_to_minutes (h : ℕ) : ℕ := h * 60

-- Problem statement in Lean 4: Proving the resting time is 30 minutes
theorem resting_time_is_thirty :
  let distance_first := speed * (time_first_part / 60)
  let time_second_part := (distance_second_part / speed) * 60
  let time_third_part := (distance_third_part / speed) * 60
  let times_sum := time_first_part + time_second_part + time_third_part
  total_time = times_sum + 30 := 
  sorry

end resting_time_is_thirty_l94_94960


namespace least_n_for_multiple_of_8_l94_94380

def is_positive_integer (n : ℕ) : Prop := n > 0

def is_multiple_of_8 (k : ℕ) : Prop := ∃ m : ℕ, k = 8 * m

theorem least_n_for_multiple_of_8 :
  ∀ n : ℕ, (is_positive_integer n → is_multiple_of_8 (Nat.factorial n)) → n ≥ 6 :=
by
  sorry

end least_n_for_multiple_of_8_l94_94380


namespace molecular_weight_C8H10N4O6_eq_258_22_l94_94751

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def number_C : ℕ := 8
def number_H : ℕ := 10
def number_N : ℕ := 4
def number_O : ℕ := 6

def molecular_weight : ℝ :=
    (number_C * atomic_weight_C) +
    (number_H * atomic_weight_H) +
    (number_N * atomic_weight_N) +
    (number_O * atomic_weight_O)

theorem molecular_weight_C8H10N4O6_eq_258_22 :
  molecular_weight = 258.22 :=
  by
    sorry

end molecular_weight_C8H10N4O6_eq_258_22_l94_94751


namespace prove_x_eq_one_l94_94435

variables (x y : ℕ)

theorem prove_x_eq_one 
  (hx : x > 0) 
  (hy : y > 0) 
  (hdiv : ∀ n : ℕ, n > 0 → (2^n * y + 1) ∣ (x^2^n - 1)) : 
  x = 1 :=
sorry

end prove_x_eq_one_l94_94435


namespace value_of_expression_l94_94454

open Real

theorem value_of_expression (α : ℝ) (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + sin (2 * α)) = 10 / 3 :=
by
  sorry

end value_of_expression_l94_94454


namespace beautiful_fold_probability_l94_94944

noncomputable def probability_beautiful_fold (a : ℝ) : ℝ := 1 / 2

theorem beautiful_fold_probability 
  (A B C D F : ℝ × ℝ) 
  (ABCD_square : (A.1 = 0) ∧ (A.2 = 0) ∧ 
                 (B.1 = a) ∧ (B.2 = 0) ∧ 
                 (C.1 = a) ∧ (C.2 = a) ∧ 
                 (D.1 = 0) ∧ (D.2 = a))
  (F_in_square : 0 ≤ F.1 ∧ F.1 ≤ a ∧ 0 ≤ F.2 ∧ F.2 ≤ a):
  probability_beautiful_fold a = 1 / 2 :=
sorry

end beautiful_fold_probability_l94_94944


namespace unpaintedRegionArea_l94_94557

def boardWidth1 : ℝ := 5
def boardWidth2 : ℝ := 7
def angle : ℝ := 45

theorem unpaintedRegionArea
  (bw1 bw2 angle : ℝ)
  (h1 : bw1 = boardWidth1)
  (h2 : bw2 = boardWidth2)
  (h3 : angle = 45) :
  let base := bw2 * Real.sqrt 2
  let height := bw1
  let area := base * height
  area = 35 * Real.sqrt 2 :=
by
  sorry

end unpaintedRegionArea_l94_94557


namespace job_completion_in_time_l94_94412

theorem job_completion_in_time (t_total t_1 w_1 : ℕ) (work_done : ℚ) (h : (t_total = 30) ∧ (t_1 = 6) ∧ (w_1 = 8) ∧ (work_done = 1/3)) :
  ∃ w : ℕ, w = 4 ∧ (t_total - t_1) * w_1 / t_1 * (1 / work_done) / w = 3 :=
by
  sorry

end job_completion_in_time_l94_94412


namespace cube_edge_length_proof_l94_94622

-- Define the edge length of the cube
def edge_length_of_cube := 15

-- Define the volume of the cube
def volume_of_cube (a : ℕ) := a^3

-- Define the volume of the displaced water
def volume_of_displaced_water := 20 * 15 * 11.25

-- The theorem to prove
theorem cube_edge_length_proof : ∃ a : ℕ, volume_of_cube a = 3375 ∧ a = edge_length_of_cube := 
by {
  sorry
}

end cube_edge_length_proof_l94_94622


namespace sin_double_angle_l94_94151

theorem sin_double_angle (θ : ℝ) 
    (h : Real.sin (Real.pi / 4 + θ) = 1 / 3) : 
    Real.sin (2 * θ) = -7 * Real.sqrt 2 / 9 :=
by
  sorry

end sin_double_angle_l94_94151


namespace second_cannibal_wins_l94_94966

/-- Define a data structure for the position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  deriving Inhabited, DecidableEq

/-- Check if two positions are adjacent in a legal move (vertical or horizontal) -/
def isAdjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1))

/-- Define the initial positions of the cannibals -/
def initialPositionFirstCannibal : Position := ⟨1, 1⟩
def initialPositionSecondCannibal : Position := ⟨8, 8⟩

/-- Define a move function for a cannibal (a valid move should keep it on the board) -/
def move (p : Position) (direction : String) : Position :=
  match direction with
  | "up"     => if p.y < 8 then ⟨p.x, p.y + 1⟩ else p
  | "down"   => if p.y > 1 then ⟨p.x, p.y - 1⟩ else p
  | "left"   => if p.x > 1 then ⟨p.x - 1, p.y⟩ else p
  | "right"  => if p.x < 8 then ⟨p.x + 1, p.y⟩ else p
  | _        => p

/-- Predicate determining if a cannibal can eat the other by moving to its position -/
def canEat (p1 p2 : Position) : Bool :=
  p1 = p2

/-- 
  Prove that the second cannibal will eat the first cannibal with the correct strategy. 
  We formalize the fact that with correct play, starting from the initial positions, 
  the second cannibal (initially at ⟨8, 8⟩) can always force a win.
-/
theorem second_cannibal_wins :
  ∀ (p1 p2 : Position), 
  p1 = initialPositionFirstCannibal →
  p2 = initialPositionSecondCannibal →
  (∃ strategy : (Position → String), ∀ positionFirstCannibal : Position, canEat (move p2 (strategy p2)) positionFirstCannibal) :=
by
  sorry

end second_cannibal_wins_l94_94966


namespace range_of_a_l94_94354

noncomputable def proposition_p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1
noncomputable def proposition_q (x : ℝ) (a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ (∃ x, ¬ proposition_p x) → ¬ (∃ x, ¬ proposition_q x a)) →
  (¬ (¬ (∃ x, ¬ proposition_p x) ∧ ¬ (¬ (∃ x, ¬ proposition_q x a)))) →
  (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  intro h₁ h₂
  sorry

end range_of_a_l94_94354


namespace inequality_proof_l94_94353

noncomputable def inequality (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : Prop :=
  (a * b) / (c - 1) + (b * c) / (a - 1) + (c * a) / (b - 1) >= 12

theorem inequality_proof (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : inequality a b c ha hb hc :=
by
  sorry

end inequality_proof_l94_94353


namespace sum_of_fractions_l94_94228

theorem sum_of_fractions :
  (1 / 3) + (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-9 / 20) = -9 / 20 := 
by
  sorry

end sum_of_fractions_l94_94228


namespace solve_quadratic_1_solve_quadratic_2_l94_94299

theorem solve_quadratic_1 (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by sorry

theorem solve_quadratic_2 (x : ℝ) : x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l94_94299


namespace find_x_l94_94530

-- Let's define the constants and the condition
def a : ℝ := 2.12
def b : ℝ := 0.345
def c : ℝ := 2.4690000000000003

-- We need to prove that there exists a number x such that
def x : ℝ := 0.0040000000000003

-- Formal statement
theorem find_x : a + b + x = c :=
by
  -- Proof skipped
  sorry
 
end find_x_l94_94530


namespace arithmetic_sequence_a12_l94_94130

def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) (a1 d : ℤ) (h : arithmetic_sequence a a1 d) :
  a 11 = 23 :=
by
  -- condtions
  let a1_val := 1
  let d_val := 2
  have ha1 : a1 = a1_val := sorry
  have hd : d = d_val := sorry
  
  -- proof
  rw [ha1, hd] at h
  
  sorry

end arithmetic_sequence_a12_l94_94130


namespace colleague_typing_time_l94_94889

theorem colleague_typing_time (T : ℝ) : 
  (∀ me_time : ℝ, (me_time = 180) →
  (∀ my_speed my_colleague_speed : ℝ, (my_speed = T / me_time) →
  (my_colleague_speed = 4 * my_speed) →
  (T / my_colleague_speed = 45))) :=
  sorry

end colleague_typing_time_l94_94889


namespace dealer_car_ratio_calculation_l94_94830

theorem dealer_car_ratio_calculation (X Y : ℝ) 
  (cond1 : 1.4 * X = 1.54 * (X + Y) - 1.6 * Y) :
  let a := 3
  let b := 7
  ((X / Y) = (3 / 7) ∧ (11 * a + 13 * b = 124)) :=
by
  sorry

end dealer_car_ratio_calculation_l94_94830


namespace city_A_fare_higher_than_city_B_l94_94265

def fare_in_city_A (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

def fare_in_city_B (x : ℝ) : ℝ :=
  8 + 2.5 * (x - 3)

theorem city_A_fare_higher_than_city_B (x : ℝ) (h : x > 3) :
  fare_in_city_A x > fare_in_city_B x → 3 < x ∧ x < 7 :=
by
  sorry

end city_A_fare_higher_than_city_B_l94_94265


namespace solve_for_A_plus_B_l94_94033

-- Definition of the problem conditions
def T := 7 -- The common total sum for rows and columns

-- Summing the rows and columns in the partially filled table
variable (A B : ℕ)
def table_condition :=
  4 + 1 + 2 = T ∧
  2 + A + B = T ∧
  4 + 2 + B = T ∧
  1 + A + B = T

-- Statement to prove
theorem solve_for_A_plus_B (A B : ℕ) (h : table_condition A B) : A + B = 5 :=
by
  sorry

end solve_for_A_plus_B_l94_94033


namespace print_time_325_pages_l94_94455

theorem print_time_325_pages (pages : ℕ) (rate : ℕ) (delay_pages : ℕ) (delay_time : ℕ)
  (h_pages : pages = 325) (h_rate : rate = 25) (h_delay_pages : delay_pages = 100) (h_delay_time : delay_time = 1) :
  let print_time := pages / rate
  let delays := pages / delay_pages
  let total_time := print_time + delays * delay_time
  total_time = 16 :=
by
  sorry

end print_time_325_pages_l94_94455


namespace rectangle_area_change_l94_94461

theorem rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  A' = 0.92 * A :=
by
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  show A' = 0.92 * A
  sorry

end rectangle_area_change_l94_94461


namespace problem1_l94_94058

theorem problem1 (a b : ℝ) : 
  ((-2 * a) ^ 3 * (- (a * b^2)) ^ 3 - 4 * a * b^2 * (2 * a^5 * b^4 + (1 / 2) * a * b^3 - 5)) / (-2 * a * b) = a * b^4 - 10 * b :=
sorry

end problem1_l94_94058


namespace brick_length_l94_94373

theorem brick_length 
  (width : ℝ) (height : ℝ) (num_bricks : ℕ)
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_vol : ℝ) :
  width = 10 →
  height = 7.5 →
  num_bricks = 27000 →
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_vol = width * height * (20:ℝ) →
  wall_length * wall_width * wall_height * 1000000 = num_bricks * brick_vol :=
by
  intros
  sorry

end brick_length_l94_94373


namespace right_triangle_sides_l94_94145

theorem right_triangle_sides (a b c : ℝ)
    (h1 : a + b + c = 30)
    (h2 : a^2 + b^2 = c^2)
    (h3 : ∃ r, a = (5 * r) / 2 ∧ a + b = 5 * r ∧ ∀ x y, x / y = 2 / 3) :
  a = 5 ∧ b = 12 ∧ c = 13 :=
sorry

end right_triangle_sides_l94_94145


namespace find_C_l94_94766

theorem find_C (A B C : ℕ) (h1 : 3 * A - A = 10) (h2 : B + A = 12) (h3 : C - B = 6) : C = 13 :=
by
  sorry

end find_C_l94_94766


namespace ratio_of_numbers_l94_94097

theorem ratio_of_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hsum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_numbers_l94_94097


namespace sector_area_l94_94386

theorem sector_area (radius area : ℝ) (θ : ℝ) (h1 : 2 * radius + θ * radius = 16) (h2 : θ = 2) : area = 16 :=
  sorry

end sector_area_l94_94386


namespace division_of_cubics_l94_94757

theorem division_of_cubics (a b c : ℕ) (h_a : a = 7) (h_b : b = 6) (h_c : c = 1) :
  (a^3 + b^3) / (a^2 - a * b + b^2 + c) = 559 / 44 :=
by
  rw [h_a, h_b, h_c]
  -- After these substitutions, the problem is reduced to proving
  -- (7^3 + 6^3) / (7^2 - 7 * 6 + 6^2 + 1) = 559 / 44
  sorry

end division_of_cubics_l94_94757


namespace debby_deletion_l94_94335

theorem debby_deletion :
  ∀ (zoo_pics museum_pics remaining_pics deleted_pics : ℕ),
    zoo_pics = 24 →
    museum_pics = 12 →
    remaining_pics = 22 →
    deleted_pics = zoo_pics + museum_pics - remaining_pics →
    deleted_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics deleted_pics h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end debby_deletion_l94_94335


namespace tea_sales_revenue_l94_94553

theorem tea_sales_revenue (x : ℝ) (price_last_year price_this_year : ℝ) (yield_last_year yield_this_year : ℝ) (revenue_last_year revenue_this_year : ℝ) :
  price_this_year = 10 * price_last_year →
  yield_this_year = 198.6 →
  yield_last_year = 198.6 + 87.4 →
  revenue_this_year = 198.6 * price_this_year →
  revenue_last_year = yield_last_year * price_last_year →
  revenue_this_year = revenue_last_year + 8500 →
  revenue_this_year = 9930 := 
by
  sorry

end tea_sales_revenue_l94_94553


namespace bounded_sequence_l94_94831

theorem bounded_sequence (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 2)
  (h_rec : ∀ n : ℕ, a (n + 2) = (a (n + 1) + a n) / Nat.gcd (a n) (a (n + 1))) :
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M := 
sorry

end bounded_sequence_l94_94831


namespace harvest_rate_l94_94673

def days := 3
def total_sacks := 24
def sacks_per_day := total_sacks / days

theorem harvest_rate :
  sacks_per_day = 8 :=
by
  sorry

end harvest_rate_l94_94673


namespace rectangle_perimeter_l94_94997
-- Refined definitions and setup
variables (AB BC AE BE CF : ℝ)
-- Conditions provided in the problem
def conditions := AB = 2 * BC ∧ AE = 10 ∧ BE = 26 ∧ CF = 5
-- Perimeter calculation based on the conditions
def perimeter (AB BC : ℝ) : ℝ := 2 * (AB + BC)
-- Main theorem stating the conditions and required result
theorem rectangle_perimeter {m n : ℕ} (h: conditions AB BC AE BE CF) :
  m + n = 105 ∧ Int.gcd m n = 1 ∧ perimeter AB BC = m / n := sorry

end rectangle_perimeter_l94_94997


namespace possible_value_of_n_l94_94546

open Nat

def coefficient_is_rational (n r : ℕ) : Prop :=
  (n - r) % 2 = 0 ∧ r % 3 = 0

theorem possible_value_of_n :
  ∃ n : ℕ, n > 0 ∧ (∀ r : ℕ, r ≤ n → coefficient_is_rational n r) ↔ n = 9 :=
sorry

end possible_value_of_n_l94_94546


namespace probability_answered_within_first_four_rings_l94_94937

theorem probability_answered_within_first_four_rings 
  (P1 P2 P3 P4 : ℝ) (h1 : P1 = 0.1) (h2 : P2 = 0.3) (h3 : P3 = 0.4) (h4 : P4 = 0.1) :
  (1 - ((1 - P1) * (1 - P2) * (1 - P3) * (1 - P4))) = 0.9 := 
sorry

end probability_answered_within_first_four_rings_l94_94937


namespace calc_2002_sq_minus_2001_mul_2003_l94_94804

theorem calc_2002_sq_minus_2001_mul_2003 : 2002 ^ 2 - 2001 * 2003 = 1 := 
by
  sorry

end calc_2002_sq_minus_2001_mul_2003_l94_94804


namespace problem1_simplified_problem2_simplified_l94_94485

-- Definition and statement for the first problem
def problem1_expression (x y : ℝ) : ℝ := 
  -3 * x * y - 3 * x^2 + 4 * x * y + 2 * x^2

theorem problem1_simplified (x y : ℝ) : 
  problem1_expression x y = x * y - x^2 := 
by
  sorry

-- Definition and statement for the second problem
def problem2_expression (a b : ℝ) : ℝ := 
  3 * (a^2 - 2 * a * b) - 5 * (a^2 + 4 * a * b)

theorem problem2_simplified (a b : ℝ) : 
  problem2_expression a b = -2 * a^2 - 26 * a * b :=
by
  sorry

end problem1_simplified_problem2_simplified_l94_94485


namespace sacks_after_6_days_l94_94733

theorem sacks_after_6_days (sacks_per_day : ℕ) (days : ℕ) 
  (h1 : sacks_per_day = 83) (h2 : days = 6) : 
  sacks_per_day * days = 498 :=
by
  sorry

end sacks_after_6_days_l94_94733


namespace factorized_expression_l94_94780

variable {a b c : ℝ}

theorem factorized_expression :
  ( ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / 
    ((a - b)^3 + (b - c)^3 + (c - a)^3) ) 
  = (a + b) * (a + c) * (b + c) := 
  sorry

end factorized_expression_l94_94780


namespace woman_year_of_birth_l94_94248

def year_of_birth (x : ℕ) : ℕ := x^2 - x

theorem woman_year_of_birth : ∃ (x : ℕ), 1850 ≤ year_of_birth x ∧ year_of_birth x < 1900 ∧ year_of_birth x = 1892 :=
by
  sorry

end woman_year_of_birth_l94_94248


namespace vector_coordinates_l94_94250

theorem vector_coordinates (A B : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (-1, 2)) :
  B - A = (-1, 1) :=
sorry

end vector_coordinates_l94_94250


namespace smallest_n_property_l94_94753

theorem smallest_n_property (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
 (hxy : x ∣ y^3) (hyz : y ∣ z^3) (hzx : z ∣ x^3) : 
  x * y * z ∣ (x + y + z) ^ 13 := 
by sorry

end smallest_n_property_l94_94753


namespace no_possible_arrangement_of_balloons_l94_94974

/-- 
  There are 10 balloons hanging in a row: blue and green. This statement proves that it is impossible 
  to arrange 10 balloons such that between every two blue balloons, there is an even number of 
  balloons and between every two green balloons, there is an odd number of balloons.
--/

theorem no_possible_arrangement_of_balloons :
  ¬ (∃ (color : Fin 10 → Bool), 
    (∀ i j, i < j ∧ color i = color j ∧ color i = tt → (j - i - 1) % 2 = 0) ∧
    (∀ i j, i < j ∧ color i = color j ∧ color i = ff → (j - i - 1) % 2 = 1)) :=
by
  sorry

end no_possible_arrangement_of_balloons_l94_94974


namespace calculate_total_weight_l94_94748

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l94_94748


namespace intersection_a_eq_1_parallel_lines_value_of_a_l94_94736

-- Define lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - a + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Part 1: Prove intersection point for a = 1
theorem intersection_a_eq_1 :
  line1 1 (-4) 3 ∧ line2 1 (-4) 3 :=
by sorry

-- Part 2: Prove value of a for which lines are parallel
theorem parallel_lines_value_of_a :
  ∃ a : ℝ, ∀ x y : ℝ, line1 a x y ∧ line2 a x y →
  (2 * a^2 - a - 3 = 0 ∧ a ≠ -1 ∧ a = 3/2) :=
by sorry

end intersection_a_eq_1_parallel_lines_value_of_a_l94_94736


namespace problem_not_true_equation_l94_94873

theorem problem_not_true_equation
  (a b : ℝ)
  (h : a / b = 2 / 3) : a / b ≠ (a + 2) / (b + 2) := 
sorry

end problem_not_true_equation_l94_94873


namespace factorize_expression_l94_94584

theorem factorize_expression (a b : ℝ) : b^2 - ab + a - b = (b - 1) * (b - a) :=
by
  sorry

end factorize_expression_l94_94584


namespace find_angle_A_l94_94337

theorem find_angle_A (A B C a b c : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : B = (A + C) / 2)
  (h3 : 2 * b ^ 2 = 3 * a * c) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l94_94337


namespace range_of_a_l94_94719

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l94_94719


namespace roberto_starting_salary_l94_94298

-- Given conditions as Lean definitions
def current_salary : ℝ := 134400
def previous_salary (S : ℝ) : ℝ := 1.40 * S

-- The proof problem statement
theorem roberto_starting_salary (S : ℝ) 
    (h1 : current_salary = 1.20 * previous_salary S) : 
    S = 80000 :=
by
  -- We will insert the proof here
  sorry

end roberto_starting_salary_l94_94298


namespace floor_e_eq_two_l94_94061

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l94_94061


namespace ellipse_intersection_l94_94628

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_intersection (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 5))
    (h2 : f2 = (4, 0))
    (origin_intersection : distance (0, 0) f1 + distance (0, 0) f2 = 5) :
    ∃ x : ℝ, (distance (x, 0) f1 + distance (x, 0) f2 = 5 ∧ x > 0 ∧ x ≠ 0 → x = 28 / 9) :=
by 
  sorry

end ellipse_intersection_l94_94628


namespace spherical_coordinate_conversion_l94_94204

theorem spherical_coordinate_conversion (ρ θ φ : ℝ) 
  (h_ρ : ρ > 0) 
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h_φ : 0 ≤ φ): 
  (ρ, θ, φ - 2 * Real.pi * ⌊φ / (2 * Real.pi)⌋) = (5, 3 * Real.pi / 4, Real.pi / 4) :=
  by 
  sorry

end spherical_coordinate_conversion_l94_94204


namespace find_BP_l94_94626

-- Define points
variables {A B C D P : Type}  

-- Define lengths
variables (AP PC BP DP BD : ℝ)

-- Provided conditions
axiom h1 : AP = 10
axiom h2 : PC = 2
axiom h3 : BD = 9

-- Assume intersect and lengths relations setup
axiom intersect : BP < DP
axiom power_of_point : AP * PC = BP * DP

-- Target statement
theorem find_BP (h1 : AP = 10) (h2 : PC = 2) (h3 : BD = 9)
  (intersect : BP < DP) (power_of_point : AP * PC = BP * DP) : BP = 4 :=
  sorry

end find_BP_l94_94626


namespace hammers_in_comparison_group_l94_94826

theorem hammers_in_comparison_group (H W x : ℝ) (h1 : 2 * H + 2 * W = 1 / 3 * (x * H + 5 * W)) (h2 : W = 2 * H) :
  x = 8 :=
sorry

end hammers_in_comparison_group_l94_94826


namespace eq_zero_or_one_if_square_eq_self_l94_94256

theorem eq_zero_or_one_if_square_eq_self (a : ℝ) (h : a^2 = a) : a = 0 ∨ a = 1 :=
sorry

end eq_zero_or_one_if_square_eq_self_l94_94256


namespace total_men_wages_l94_94023

-- Define our variables and parameters
variable (M W B : ℝ)
variable (W_women : ℝ)

-- Conditions from the problem:
-- 1. 12M = WW (where WW is W_women)
-- 2. WW = 20B
-- 3. 12M + WW + 20B = 450
axiom eq_12M_WW : 12 * M = W_women
axiom eq_WW_20B : W_women = 20 * B
axiom eq_total_earnings : 12 * M + W_women + 20 * B = 450

-- Prove total wages of the men is Rs. 150
theorem total_men_wages : 12 * M = 150 := by
  sorry

end total_men_wages_l94_94023


namespace maurice_needs_7_letters_l94_94456
noncomputable def prob_no_job (n : ℕ) : ℝ := (4 / 5) ^ n

theorem maurice_needs_7_letters :
  ∃ n : ℕ, (prob_no_job n) ≤ 1 / 4 ∧ n = 7 :=
by
  sorry

end maurice_needs_7_letters_l94_94456


namespace cylinder_base_ratio_l94_94342

variable (O : Point) -- origin
variable (a b c : ℝ) -- fixed point
variable (p q : ℝ) -- center of circular base
variable (α β : ℝ) -- intersection points with axis

-- Let O be the origin
-- Let (a, b, c) be the fixed point through which the cylinder passes
-- The cylinder's axis is parallel to the z-axis and the center of its base is (p, q)
-- The cylinder intersects the x-axis at (α, 0, 0) and the y-axis at (0, β, 0)
-- Let α = 2p and β = 2q

theorem cylinder_base_ratio : 
  α = 2 * p ∧ β = 2 * q → (a / p + b / q = 4) := by
  sorry

end cylinder_base_ratio_l94_94342


namespace find_x_value_l94_94057

theorem find_x_value (x : ℚ) (h : 5 * (x - 10) = 3 * (3 - 3 * x) + 9) : x = 34 / 7 := by
  sorry

end find_x_value_l94_94057


namespace cubic_foot_to_cubic_inches_l94_94907

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l94_94907


namespace repeating_decimal_as_fraction_l94_94996

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end repeating_decimal_as_fraction_l94_94996


namespace incorrect_median_l94_94827

/-- 
Given:
- A stem-and-leaf plot representation.
- Player B's scores are mainly between 30 and 40 points.
- Player B has 13 scores.
Prove:
The judgment "The median score of player B is 28" is incorrect.
-/
theorem incorrect_median (scores : List ℕ) (H_len : scores.length = 13) (H_range : ∀ x ∈ scores, 30 ≤ x ∧ x ≤ 40) 
  (H_median : ∃ median, median = scores.nthLe 6 sorry ∧ median = 28) : False := 
sorry

end incorrect_median_l94_94827


namespace average_salary_without_manager_l94_94136

theorem average_salary_without_manager (A : ℝ) (H : 15 * A + 4200 = 16 * (A + 150)) : A = 1800 :=
by {
  sorry
}

end average_salary_without_manager_l94_94136


namespace pants_cost_is_250_l94_94176

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end pants_cost_is_250_l94_94176


namespace sequence_formula_l94_94082

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = -2 ^ (n - 1) := 
by 
  sorry

end sequence_formula_l94_94082


namespace find_y_l94_94625

def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

theorem find_y (y : ℝ) (h : star 4 y = 80) : y = 16.8 :=
by
  sorry

end find_y_l94_94625


namespace max_volume_tank_l94_94880

theorem max_volume_tank (a b h : ℝ) (ha : a ≤ 1.5) (hb : b ≤ 1.5) (hh : h = 1.5) :
  a * b * h ≤ 3.375 :=
by {
  sorry
}

end max_volume_tank_l94_94880


namespace min_value_of_expression_l94_94657

theorem min_value_of_expression (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 10) : 
  ∃ B, B = x^2 + y^2 + z^2 + x^2 * y ∧ B ≥ 4 :=
by
  sorry

end min_value_of_expression_l94_94657


namespace max_value_of_quadratic_l94_94493

def quadratic_func (x : ℝ) : ℝ := -3 * (x - 2) ^ 2 - 3

theorem max_value_of_quadratic : 
  ∃ x : ℝ, quadratic_func x = -3 :=
by
  sorry

end max_value_of_quadratic_l94_94493


namespace reggie_books_l94_94171

/-- 
Reggie's father gave him $48. Reggie bought some books, each of which cost $2, 
and now he has $38 left. How many books did Reggie buy?
-/
theorem reggie_books (initial_amount spent_amount remaining_amount book_cost books_bought : ℤ)
  (h_initial : initial_amount = 48)
  (h_remaining : remaining_amount = 38)
  (h_book_cost : book_cost = 2)
  (h_spent : spent_amount = initial_amount - remaining_amount)
  (h_books_bought : books_bought = spent_amount / book_cost) :
  books_bought = 5 :=
by
  sorry

end reggie_books_l94_94171


namespace gross_profit_value_l94_94935

theorem gross_profit_value (C GP : ℝ) (h1 : GP = 1.6 * C) (h2 : 91 = C + GP) : GP = 56 :=
by
  sorry

end gross_profit_value_l94_94935


namespace Adeline_hourly_wage_l94_94283

theorem Adeline_hourly_wage
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hours_per_day = 9) 
  (h2 : days_per_week = 5) 
  (h3 : weeks = 7) 
  (h4 : total_earnings = 3780) :
  total_earnings = 12 * (hours_per_day * days_per_week * weeks) :=
by
  sorry

end Adeline_hourly_wage_l94_94283


namespace fg_of_one_eq_onehundredandfive_l94_94437

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2)^3

theorem fg_of_one_eq_onehundredandfive : f (g 1) = 105 :=
by
  -- proof would go here
  sorry

end fg_of_one_eq_onehundredandfive_l94_94437


namespace unique_integer_solution_l94_94523

def is_point_in_circle (x y cx cy radius : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ radius^2

theorem unique_integer_solution : ∃! (x : ℤ), is_point_in_circle (2 * x) (-x) 4 6 8 := by
  sorry

end unique_integer_solution_l94_94523


namespace unit_square_divisible_l94_94559

theorem unit_square_divisible (n : ℕ) (h: n ≥ 6) : ∃ squares : ℕ, squares = n :=
by
  sorry

end unit_square_divisible_l94_94559


namespace g100_value_l94_94596

-- Define the function g and its properties
def g (x : ℝ) : ℝ := sorry

theorem g100_value 
  (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g (x / y) + x - y) : 
  g 100 = 99 / 2 := 
sorry

end g100_value_l94_94596


namespace remainder_of_x13_plus_1_by_x_minus_1_l94_94689

-- Define the polynomial f(x) = x^13 + 1
def f (x : ℕ) : ℕ := x ^ 13 + 1

-- State the theorem using the Polynomial Remainder Theorem
theorem remainder_of_x13_plus_1_by_x_minus_1 : f 1 = 2 := by
  -- Skip the proof
  sorry

end remainder_of_x13_plus_1_by_x_minus_1_l94_94689


namespace triangle_side_and_altitude_sum_l94_94774

theorem triangle_side_and_altitude_sum 
(x y : ℕ) (h1 : x < 75) (h2 : y < 28)
(h3 : x * 60 = 75 * 28) (h4 : 100 * y = 75 * 28) : 
x + y = 56 := 
sorry

end triangle_side_and_altitude_sum_l94_94774


namespace xyz_squared_sum_l94_94302

theorem xyz_squared_sum (x y z : ℝ) 
  (h1 : x^2 + 4 * y^2 + 16 * z^2 = 48)
  (h2 : x * y + 4 * y * z + 2 * z * x = 24) :
  x^2 + y^2 + z^2 = 21 :=
sorry

end xyz_squared_sum_l94_94302


namespace find_number_l94_94041

theorem find_number (x : ℝ) (h : 0.36 * x = 129.6) : x = 360 :=
by sorry

end find_number_l94_94041


namespace find_smallest_y_l94_94891

noncomputable def x : ℕ := 5 * 15 * 35

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 4 = n

theorem find_smallest_y : ∃ y : ℕ, y > 0 ∧ is_perfect_fourth_power (x * y) ∧ y = 46485 := by
  sorry

end find_smallest_y_l94_94891


namespace ice_cream_flavors_l94_94812

theorem ice_cream_flavors (F : ℕ) (h1 : F / 4 + F / 2 + 25 = F) : F = 100 :=
by
  sorry

end ice_cream_flavors_l94_94812


namespace inequality_range_l94_94985

theorem inequality_range (y : ℝ) (b : ℝ) (hb : 0 < b) : (|y-5| + 2 * |y-2| > b) ↔ (b < 3) := 
sorry

end inequality_range_l94_94985


namespace green_beads_in_pattern_l94_94504

noncomputable def G : ℕ := 3
def P : ℕ := 5
def R (G : ℕ) : ℕ := 2 * G
def total_beads (G : ℕ) (P : ℕ) (R : ℕ) : ℕ := 3 * (G + P + R) + 10 * 5 * (G + P + R)

theorem green_beads_in_pattern :
  total_beads 3 5 (R 3) = 742 :=
by
  sorry

end green_beads_in_pattern_l94_94504


namespace volume_ratio_sum_is_26_l94_94392

noncomputable def volume_of_dodecahedron (s : ℝ) : ℝ :=
  (15 + 7 * Real.sqrt 5) * s ^ 3 / 4

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

noncomputable def volume_ratio_sum (s : ℝ) : ℝ :=
  let ratio := (volume_of_dodecahedron s) / (volume_of_cube s)
  let numerator := 15 + 7 * Real.sqrt 5
  let denominator := 4
  numerator + denominator

theorem volume_ratio_sum_is_26 (s : ℝ) : volume_ratio_sum s = 26 := by
  sorry

end volume_ratio_sum_is_26_l94_94392


namespace exists_colored_triangle_l94_94187

structure Point := (x : ℝ) (y : ℝ)
inductive Color
| red
| blue

def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)
  
def same_color_triangle_exists (S : Finset Point) (color : Point → Color) : Prop :=
  ∃ (A B C : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧
                    (color A = color B ∧ color B = color C) ∧
                    ¬ collinear A B C ∧
                    (∃ (X Y Z : Point), 
                      ((X ∈ S ∧ color X ≠ color A ∧ (X ≠ A ∧ X ≠ B ∧ X ≠ C)) ∧ 
                       (Y ∈ S ∧ color Y ≠ color A ∧ (Y ≠ A ∧ Y ≠ B ∧ Y ≠ C)) ∧
                       (Z ∈ S ∧ color Z ≠ color A ∧ (Z ≠ A ∧ Z ≠ B ∧ Z ≠ C)) → 
                       False))

theorem exists_colored_triangle 
  (S : Finset Point) (h1 : 5 ≤ S.card) (color : Point → Color) 
  (h2 : ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → (color A = color B ∧ color B = color C) → ¬ collinear A B C) 
  : same_color_triangle_exists S color :=
sorry

end exists_colored_triangle_l94_94187


namespace decreasing_linear_function_l94_94603

theorem decreasing_linear_function (k : ℝ) : 
  (∀ x1 x2 : ℝ, x1 < x2 → (k - 3) * x1 + 2 > (k - 3) * x2 + 2) → k < 3 := 
by 
  sorry

end decreasing_linear_function_l94_94603


namespace tan_alpha_value_l94_94750

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 3 / 5) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : Real.tan α = -3 / 4 := 
sorry

end tan_alpha_value_l94_94750


namespace sequence_a_n_eq_T_n_formula_C_n_formula_l94_94902

noncomputable def sequence_S (n : ℕ) : ℕ := n * (2 * n - 1)

def arithmetic_seq (n : ℕ) : ℚ := 2 * n - 1

def a_n (n : ℕ) : ℤ := 4 * n - 3

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * n + 1)

def c_n (n : ℕ) : ℚ := 3^(n - 1)

def C_n (n : ℕ) : ℚ := (3^n - 1) / 2

theorem sequence_a_n_eq (n : ℕ) : a_n n = 4 * n - 3 := by sorry

theorem T_n_formula (n : ℕ) : T_n n = (n : ℚ) / (4 * n + 1) := by sorry

theorem C_n_formula (n : ℕ) : C_n n = (3^n - 1) / 2 := by sorry

end sequence_a_n_eq_T_n_formula_C_n_formula_l94_94902


namespace find_m_l94_94595

-- Given conditions
variable (U : Set ℕ) (A : Set ℕ) (m : ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 })
variable (hCUA : U \ A = {1, 4})

-- Prove that m = 6
theorem find_m (U A : Set ℕ) (m : ℕ) 
               (hU : U = {1, 2, 3, 4}) 
               (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 }) 
               (hCUA : U \ A = {1, 4}) : 
  m = 6 := 
sorry

end find_m_l94_94595


namespace standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l94_94693

theorem standing_in_a_row (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 → 
  ∃ (ways : ℕ), ways = 120 :=
by
  sorry

theorem standing_in_a_row_AB_adj_CD_not_adj (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 24 :=
by
  sorry

theorem assign_to_classes (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 150 :=
by
  sorry

end standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l94_94693


namespace shaded_to_largest_ratio_l94_94841

theorem shaded_to_largest_ratio :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let r4 := 4
  let area (r : ℝ) := π * r^2
  let largest_circle_area := area r4
  let innermost_shaded_area := area r1
  let outermost_shaded_area := area r3 - area r2
  let shaded_area := innermost_shaded_area + outermost_shaded_area
  shaded_area / largest_circle_area = 3 / 8 :=
by
  sorry

end shaded_to_largest_ratio_l94_94841


namespace girl_speed_l94_94294

theorem girl_speed (distance time : ℝ) (h₁ : distance = 128) (h₂ : time = 32) : distance / time = 4 := 
by 
  rw [h₁, h₂]
  norm_num

end girl_speed_l94_94294


namespace min_value_of_expression_l94_94239

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 3) : 
  ∃ k : ℝ, k = 4 + 2 * Real.sqrt 3 ∧ ∀ z, (z = (1 / (x - 1) + 3 / (y - 1))) → z ≥ k :=
sorry

end min_value_of_expression_l94_94239


namespace surface_area_parallelepiped_l94_94181

theorem surface_area_parallelepiped (a b : ℝ) :
  ∃ S : ℝ, (S = 3 * a * b) :=
sorry

end surface_area_parallelepiped_l94_94181


namespace solution_l94_94756

noncomputable def F (a b c : ℝ) := a * (b ^ 3) + c

theorem solution (a : ℝ) (h : F a 2 3 = F a 3 10) : a = -7 / 19 := sorry

end solution_l94_94756


namespace sunzi_wood_problem_l94_94271

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l94_94271


namespace g_possible_values_l94_94590

noncomputable def g (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((x - 1) / (x + 1)) + Real.arctan (1 / x)

theorem g_possible_values (x : ℝ) (hx₁ : x ≠ 0) (hx₂ : x ≠ -1) (hx₃ : x ≠ 1) :
  g x = (Real.pi / 4) ∨ g x = (5 * Real.pi / 4) :=
sorry

end g_possible_values_l94_94590


namespace point_above_line_l94_94587

theorem point_above_line (t : ℝ) : (∃ y : ℝ, y = (2 : ℝ)/3) → (t > (2 : ℝ)/3) :=
  by
  intro h
  sorry

end point_above_line_l94_94587


namespace right_handed_players_total_l94_94900

theorem right_handed_players_total
    (total_players : ℕ)
    (throwers : ℕ)
    (left_handed : ℕ)
    (right_handed : ℕ) :
    total_players = 150 →
    throwers = 60 →
    left_handed = (total_players - throwers) / 2 →
    right_handed = (total_players - throwers) / 2 →
    total_players - throwers = 2 * left_handed →
    left_handed + right_handed + throwers = total_players →
    ∀ throwers : ℕ, throwers = 60 →
    right_handed + throwers = 105 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end right_handed_players_total_l94_94900


namespace value_of_5_T_3_l94_94989

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end value_of_5_T_3_l94_94989


namespace root_condition_l94_94677

noncomputable def f (x t : ℝ) := x^2 + t * x - t

theorem root_condition {t : ℝ} : (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) := 
  sorry

end root_condition_l94_94677


namespace expand_and_simplify_l94_94032

-- Define the two polynomials P and Q.
def P (x : ℝ) := 5 * x + 3
def Q (x : ℝ) := 2 * x^2 - x + 4

-- State the theorem we want to prove.
theorem expand_and_simplify (x : ℝ) : (P x * Q x) = 10 * x^3 + x^2 + 17 * x + 12 := 
by
  sorry

end expand_and_simplify_l94_94032


namespace ratio_calc_l94_94732

theorem ratio_calc :
  (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484) /
  ((8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)) = -423 := 
by
  sorry

end ratio_calc_l94_94732


namespace apples_in_box_at_first_l94_94358

noncomputable def initial_apples (X : ℕ) : Prop :=
  (X / 2 - 25 = 6)

theorem apples_in_box_at_first (X : ℕ) : initial_apples X ↔ X = 62 :=
by
  sorry

end apples_in_box_at_first_l94_94358


namespace value_of_a_squared_plus_b_squared_l94_94149

theorem value_of_a_squared_plus_b_squared (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) :
  a^2 + b^2 = 68 :=
sorry

end value_of_a_squared_plus_b_squared_l94_94149


namespace find_b_from_quadratic_l94_94398

theorem find_b_from_quadratic (b n : ℤ)
  (h1 : b > 0)
  (h2 : (x : ℤ) → (x + n)^2 - 6 = x^2 + b * x + 19) :
  b = 10 :=
sorry

end find_b_from_quadratic_l94_94398


namespace Sheila_attendance_probability_l94_94630

-- Definitions as per given conditions
def P_rain := 0.5
def P_sunny := 0.3
def P_cloudy := 0.2
def P_Sheila_goes_given_rain := 0.3
def P_Sheila_goes_given_sunny := 0.7
def P_Sheila_goes_given_cloudy := 0.5

-- Define the probability calculation
def P_Sheila_attends := 
  (P_rain * P_Sheila_goes_given_rain) + 
  (P_sunny * P_Sheila_goes_given_sunny) + 
  (P_cloudy * P_Sheila_goes_given_cloudy)

-- Final theorem statement
theorem Sheila_attendance_probability : P_Sheila_attends = 0.46 := by
  sorry

end Sheila_attendance_probability_l94_94630


namespace neg_div_neg_eq_pos_division_of_negatives_example_l94_94871

theorem neg_div_neg_eq_pos (a b : Int) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  -- You can complete the proof here
  sorry

theorem division_of_negatives_example : (-81 : Int) / (-9) = 9 :=
  neg_div_neg_eq_pos 81 9 (by decide)

end neg_div_neg_eq_pos_division_of_negatives_example_l94_94871


namespace shaded_area_eight_l94_94601

-- Definitions based on given conditions
def arcAQB (r : ℝ) : Prop := r = 2
def arcBRC (r : ℝ) : Prop := r = 2
def midpointQ (r : ℝ) : Prop := arcAQB r
def midpointR (r : ℝ) : Prop := arcBRC r
def midpointS (r : ℝ) : Prop := arcAQB r ∧ arcBRC r ∧ (arcAQB r ∨ arcBRC r)
def arcQRS (r : ℝ) : Prop := r = 2 ∧ midpointS r

-- The theorem to prove
theorem shaded_area_eight (r : ℝ) : arcAQB r ∧ arcBRC r ∧ arcQRS r → area_shaded_region = 8 := by
  sorry

end shaded_area_eight_l94_94601


namespace find_other_root_l94_94790

theorem find_other_root (x y : ℚ) (h : 48 * x^2 - 77 * x + 21 = 0) (hx : x = 3 / 4) : y = 7 / 12 → 48 * y^2 - 77 * y + 21 = 0 := by
  sorry

end find_other_root_l94_94790


namespace surface_area_increase_l94_94497

theorem surface_area_increase :
  let l := 4
  let w := 3
  let h := 2
  let side_cube := 1
  let original_surface := 2 * (l * w + l * h + w * h)
  let additional_surface := 6 * side_cube * side_cube
  let new_surface := original_surface + additional_surface
  new_surface = original_surface + 6 :=
by
  sorry

end surface_area_increase_l94_94497


namespace range_of_a_l94_94642

variable (a : ℝ)
def p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (hpq_or : p a ∨ q a) (hpq_and_false : ¬ (p a ∧ q a)) : 
    a ∈ Set.Iio 0 ∪ Set.Ioo (1/4) 4 :=
by
  sorry

end range_of_a_l94_94642


namespace units_digit_base7_of_multiplied_numbers_l94_94118

-- Define the numbers in base 10
def num1 : ℕ := 325
def num2 : ℕ := 67

-- Define the modulus used for base 7
def base : ℕ := 7

-- Function to determine the units digit of the base-7 representation
def units_digit_base7 (n : ℕ) : ℕ := n % base

-- Prove that units_digit_base7 (num1 * num2) = 5
theorem units_digit_base7_of_multiplied_numbers :
  units_digit_base7 (num1 * num2) = 5 :=
by
  sorry

end units_digit_base7_of_multiplied_numbers_l94_94118


namespace evaluate_gg2_l94_94428

noncomputable def g (x : ℚ) : ℚ := 1 / (x^2) + (x^2) / (1 + x^2)

theorem evaluate_gg2 : g (g 2) = 530881 / 370881 :=
by
  sorry

end evaluate_gg2_l94_94428


namespace percentage_increase_l94_94621

theorem percentage_increase (G P : ℝ) (h1 : G = 15 + (P / 100) * 15) 
                            (h2 : 15 + 2 * G = 51) : P = 20 :=
by 
  sorry

end percentage_increase_l94_94621


namespace train_speed_l94_94215

theorem train_speed (x : ℝ) (v : ℝ) 
  (h1 : (x / 50) + (2 * x / v) = 3 * x / 25) : v = 20 :=
by
  sorry

end train_speed_l94_94215


namespace birth_year_1957_l94_94140

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem birth_year_1957 (y x : ℕ) (h : y = 2023) (h1 : sum_of_digits x = y - x) : x = 1957 :=
by
  sorry

end birth_year_1957_l94_94140


namespace geometric_series_sum_l94_94472

theorem geometric_series_sum :
  let a := 1 / 4
  let r := - (1 / 4)
  ∃ S : ℚ, S = (a * (1 - r^6)) / (1 - r) ∧ S = 4095 / 81920 :=
by
  let a := 1 / 4
  let r := - (1 / 4)
  exists (a * (1 - r^6)) / (1 - r)
  sorry

end geometric_series_sum_l94_94472


namespace schedule_courses_l94_94218

-- Define the number of courses and periods
def num_courses : Nat := 4
def num_periods : Nat := 8

-- Define the total number of ways to schedule courses without restrictions
def unrestricted_schedules : Nat := Nat.choose num_periods num_courses * Nat.factorial num_courses

-- Define the number of invalid schedules using PIE (approximate value given in problem)
def invalid_schedules : Nat := 1008 + 180 + 120

-- Define the number of valid schedules
def valid_schedules : Nat := unrestricted_schedules - invalid_schedules

theorem schedule_courses : valid_schedules = 372 := sorry

end schedule_courses_l94_94218


namespace find_pairs_l94_94483

theorem find_pairs (p n : ℕ) (hp : Nat.Prime p) (h1 : n ≤ 2 * p) (h2 : n^(p-1) ∣ (p-1)^n + 1) : 
    (p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ (n = 1) :=
by
  sorry

end find_pairs_l94_94483


namespace y1_lt_y2_of_linear_function_l94_94771

theorem y1_lt_y2_of_linear_function (y1 y2 : ℝ) (h1 : y1 = 2 * (-3) + 1) (h2 : y2 = 2 * 2 + 1) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_function_l94_94771


namespace solve_system_of_equations_l94_94046

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 := 
sorry

end solve_system_of_equations_l94_94046


namespace proof_x_y_3_l94_94545

noncomputable def prime (n : ℤ) : Prop := 2 <= n ∧ ∀ m : ℤ, 1 ≤ m → m < n → n % m ≠ 0

theorem proof_x_y_3 (x y : ℝ) (p q r : ℤ) (h1 : x - y = p) (hp : prime p) 
  (h2 : x^2 - y^2 = q) (hq : prime q)
  (h3 : x^3 - y^3 = r) (hr : prime r) : p = 3 :=
sorry

end proof_x_y_3_l94_94545


namespace pancake_fundraiser_l94_94319

-- Define the constants and conditions
def cost_per_stack_of_pancakes : ℕ := 4
def cost_per_slice_of_bacon : ℕ := 2
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90
def total_raised : ℕ := 420

-- Define a theorem that states what we want to prove
theorem pancake_fundraiser : 
  (stacks_sold * cost_per_stack_of_pancakes + slices_sold * cost_per_slice_of_bacon) = total_raised :=
by
  sorry -- We place a sorry here to skip the proof, as instructed.

end pancake_fundraiser_l94_94319


namespace yoongi_flowers_left_l94_94389

theorem yoongi_flowers_left (initial_flowers given_to_eunji given_to_yuna : ℕ) 
  (h_initial : initial_flowers = 28) 
  (h_eunji : given_to_eunji = 7) 
  (h_yuna : given_to_yuna = 9) : 
  initial_flowers - (given_to_eunji + given_to_yuna) = 12 := 
by 
  sorry

end yoongi_flowers_left_l94_94389


namespace hours_in_one_year_l94_94927

/-- Given that there are 24 hours in a day and 365 days in a year,
    prove that there are 8760 hours in one year. -/
theorem hours_in_one_year (hours_per_day : ℕ) (days_per_year : ℕ) (hours_value : ℕ := 8760) : hours_per_day = 24 → days_per_year = 365 → hours_per_day * days_per_year = hours_value :=
by
  intros
  sorry

end hours_in_one_year_l94_94927


namespace minimize_sum_AP_BP_l94_94336

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (1, 0)
def center : point := (3, 4)
def radius : ℝ := 2

def on_circle (P : point) : Prop := (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

def AP_squared (P : point) : ℝ := (P.1 - A.1)^2 + (P.2 - A.2)^2
def BP_squared (P : point) : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
def sum_AP_BP_squared (P : point) : ℝ := AP_squared P + BP_squared P

theorem minimize_sum_AP_BP :
  ∀ P : point, on_circle P → sum_AP_BP_squared P = AP_squared (9/5, 12/5) + BP_squared (9/5, 12/5) → 
  P = (9/5, 12/5) :=
sorry

end minimize_sum_AP_BP_l94_94336


namespace standard_deviation_is_two_l94_94304

def weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum / l.length)

noncomputable def variance (l : List ℝ) : ℝ :=
  ((l.map (λ x => (x - mean l)^2)).sum / l.length)

noncomputable def standard_deviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_two : standard_deviation weights = 2 := 
by
  sorry

end standard_deviation_is_two_l94_94304


namespace simplify_fraction_l94_94039

theorem simplify_fraction (a b c d k : ℕ) (h₁ : a = 123) (h₂ : b = 9999) (h₃ : k = 41)
                           (h₄ : c = a / 3) (h₅ : d = b / 3)
                           (h₆ : c = k) (h₇ : d = 3333) :
  (a * k) / b = (k^2) / d :=
by
  sorry

end simplify_fraction_l94_94039


namespace max_gcd_dn_l94_94166

def a (n : ℕ) := 101 + n^2

def d (n : ℕ) := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_dn : ∃ n : ℕ, ∀ m : ℕ, d m ≤ 3 := sorry

end max_gcd_dn_l94_94166


namespace solve_quadratic_inequality_l94_94346

theorem solve_quadratic_inequality :
  ∀ x : ℝ, ((x - 1) * (x - 3) < 0) ↔ (1 < x ∧ x < 3) :=
by
  intro x
  sorry

end solve_quadratic_inequality_l94_94346


namespace part1_proof_l94_94939

variable (α β t x1 x2 : ℝ)

-- Conditions
def quadratic_roots := 2 * α ^ 2 - t * α - 2 = 0 ∧ 2 * β ^ 2 - t * β - 2 = 0
def roots_relation := α + β = t / 2 ∧ α * β = -1
def points_in_interval := α < β ∧ α ≤ x1 ∧ x1 ≤ β ∧ α ≤ x2 ∧ x2 ≤ β ∧ x1 ≠ x2

-- Proof of Part 1
theorem part1_proof (h1 : quadratic_roots α β t) (h2 : roots_relation α β t)
                    (h3 : points_in_interval α β x1 x2) : 
                    4 * x1 * x2 - t * (x1 + x2) - 4 < 0 := 
sorry

end part1_proof_l94_94939


namespace spider_eyes_solution_l94_94744

def spider_eyes_problem: Prop :=
  ∃ (x : ℕ), (3 * x) + (50 * 2) = 124 ∧ x = 8

theorem spider_eyes_solution : spider_eyes_problem :=
  sorry

end spider_eyes_solution_l94_94744


namespace number_of_rows_l94_94845

theorem number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 30) (h2 : pencils_per_row = 5) : total_pencils / pencils_per_row = 6 :=
by
  sorry

end number_of_rows_l94_94845


namespace weight_of_A_l94_94088

variable (A B C D E : ℕ)

axiom cond1 : A + B + C = 180
axiom cond2 : A + B + C + D = 260
axiom cond3 : E = D + 3
axiom cond4 : B + C + D + E = 256

theorem weight_of_A : A = 87 :=
by
  sorry

end weight_of_A_l94_94088


namespace permutation_of_digits_l94_94875

-- Definition of factorial
def fact : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * fact n

-- Given conditions
def n := 8
def n1 := 3
def n2 := 2
def n3 := 1
def n4 := 2

-- Statement
theorem permutation_of_digits :
  fact n / (fact n1 * fact n2 * fact n3 * fact n4) = 1680 :=
by
  sorry

end permutation_of_digits_l94_94875


namespace length_of_square_side_l94_94690

theorem length_of_square_side 
  (r : ℝ) 
  (A : ℝ) 
  (h : A = 42.06195997410015) 
  (side_length : ℝ := 2 * r)
  (area_of_square : ℝ := side_length ^ 2)
  (segment_area : ℝ := 4 * (π * r * r / 4))
  (enclosed_area: ℝ := area_of_square - segment_area)
  (h2 : enclosed_area = A) :
  side_length = 14 :=
by sorry

end length_of_square_side_l94_94690


namespace minimize_S_n_l94_94399

noncomputable def S_n (n : ℕ) : ℝ := 2 * (n : ℝ) ^ 2 - 30 * (n : ℝ)

theorem minimize_S_n :
  ∃ n : ℕ, S_n n = 2 * (7 : ℝ) ^ 2 - 30 * (7 : ℝ) ∨ S_n n = 2 * (8 : ℝ) ^ 2 - 30 * (8 : ℝ) := by
  sorry

end minimize_S_n_l94_94399


namespace present_age_of_son_l94_94540

variable (S M : ℝ)

-- Conditions
def condition1 : Prop := M = S + 35
def condition2 : Prop := M + 5 = 3 * (S + 5)

-- Proof Problem
theorem present_age_of_son
  (h1 : condition1 S M)
  (h2 : condition2 S M) :
  S = 12.5 :=
sorry

end present_age_of_son_l94_94540


namespace factorize_a3_minus_4ab2_l94_94527

theorem factorize_a3_minus_4ab2 (a b : ℝ) : a^3 - 4 * a * b^2 = a * (a + 2 * b) * (a - 2 * b) :=
by
  -- Proof is omitted; write 'sorry' as a placeholder
  sorry

end factorize_a3_minus_4ab2_l94_94527


namespace range_of_a_l94_94611

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x^2
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem range_of_a (a : ℝ) (h : ∀ s t : ℝ, (1/2 ≤ s ∧ s ≤ 2) → (1/2 ≤ t ∧ t ≤ 2) → f a s ≥ g t) : a ≥ 1 :=
sorry

end range_of_a_l94_94611


namespace probability_sum_8_9_10_l94_94758

/-- The faces of the first die -/
def first_die := [2, 2, 3, 3, 5, 5]

/-- The faces of the second die -/
def second_die := [1, 3, 4, 5, 6, 7]

/-- Predicate that checks if the sum of two numbers is either 8, 9, or 10 -/
def valid_sum (a b : ℕ) : Prop := a + b = 8 ∨ a + b = 9 ∨ a + b = 10

/-- Calculate the probability of a sum being 8, 9, or 10 according to the given dice setup -/
def calc_probability : ℚ := 
  let valid_pairs := [(2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (5, 3), (5, 4), (5, 5)] 
  (valid_pairs.length : ℚ) / (first_die.length * second_die.length : ℚ)

theorem probability_sum_8_9_10 : calc_probability = 4 / 9 :=
by
  sorry

end probability_sum_8_9_10_l94_94758


namespace f_specification_l94_94206

open Function

def f : ℕ → ℕ := sorry -- define function f here

axiom f_involution (n : ℕ) : f (f n) = n

axiom f_functional_property (n : ℕ) : f (f n + 1) = if n % 2 = 0 then n - 1 else n + 3

axiom f_bijective : Bijective f

axiom f_not_two (n : ℕ) : f (f n + 1) ≠ 2

axiom f_one_eq_two : f 1 = 2

theorem f_specification (n : ℕ) : 
  f n = if n % 2 = 1 then n + 1 else n - 1 :=
sorry

end f_specification_l94_94206


namespace solution_of_system_of_equations_l94_94742

-- Define the conditions of the problem.
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 6) ∧ (x = 2 * y)

-- Define the correct answer as a set.
def solution_set : Set (ℝ × ℝ) :=
  { (4, 2) }

-- State the proof problem.
theorem solution_of_system_of_equations : 
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ system_of_equations x y} = solution_set :=
  sorry

end solution_of_system_of_equations_l94_94742


namespace arithmetic_sequence_ninth_term_l94_94245

theorem arithmetic_sequence_ninth_term (a d : ℤ) 
    (h5 : a + 4 * d = 23) (h7 : a + 6 * d = 37) : 
    a + 8 * d = 51 := 
by 
  sorry

end arithmetic_sequence_ninth_term_l94_94245


namespace raisin_cost_fraction_l94_94844

theorem raisin_cost_fraction
  (R : ℚ) -- cost of a pound of raisins in dollars
  (cost_of_nuts : ℚ)
  (total_cost_raisins : ℚ)
  (total_cost_nuts : ℚ) :
  cost_of_nuts = 3 * R →
  total_cost_raisins = 5 * R →
  total_cost_nuts = 4 * cost_of_nuts →
  (total_cost_raisins / (total_cost_raisins + total_cost_nuts)) = 5 / 17 :=
by
  sorry

end raisin_cost_fraction_l94_94844


namespace absolute_inequality_solution_l94_94441

theorem absolute_inequality_solution (x : ℝ) (hx : x > 0) :
  |5 - 2 * x| ≤ 8 ↔ 0 ≤ x ∧ x ≤ 6.5 :=
by sorry

end absolute_inequality_solution_l94_94441


namespace student_percentage_l94_94257

theorem student_percentage (s1 s3 overall : ℕ) (percentage_second_subject : ℕ) :
  s1 = 60 →
  s3 = 85 →
  overall = 75 →
  (s1 + percentage_second_subject + s3) / 3 = overall →
  percentage_second_subject = 80 := by
  intros h1 h2 h3 h4
  sorry

end student_percentage_l94_94257


namespace arithmetic_sequence_odd_function_always_positive_l94_94536

theorem arithmetic_sequence_odd_function_always_positive
    (f : ℝ → ℝ) (a : ℕ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_monotone_geq_0 : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
    (h_arith_seq : ∀ n, a (n + 1) = a n + (a 2 - a 1))
    (h_a3_neg : a 3 < 0) :
    f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 := by
    sorry

end arithmetic_sequence_odd_function_always_positive_l94_94536


namespace tom_age_is_19_l94_94163

-- Define the ages of Carla, Tom, Dave, and Emily
variable (C : ℕ) -- Carla's age

-- Conditions
def tom_age := 2 * C - 1
def dave_age := C + 3
def emily_age := C / 2

-- Sum of their ages equating to 48
def total_age := C + tom_age C + dave_age C + emily_age C

-- Theorem to be proven
theorem tom_age_is_19 (h : total_age C = 48) : tom_age C = 19 := 
by {
  sorry
}

end tom_age_is_19_l94_94163


namespace complex_number_solution_l94_94367

variable (z : ℂ)
variable (i : ℂ)

theorem complex_number_solution (h : (1 - i)^2 / z = 1 + i) (hi : i^2 = -1) : z = -1 - i :=
sorry

end complex_number_solution_l94_94367
