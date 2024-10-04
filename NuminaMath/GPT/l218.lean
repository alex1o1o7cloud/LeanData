import Mathlib

namespace royal_children_count_l218_218364

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l218_218364


namespace train_length_l218_218397

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmh = 60 → time_s = 15 → (60 * 1000 / 3600) * 15 = length_m → length_m = 250 :=
by { intros, sorry }

end train_length_l218_218397


namespace power_equality_l218_218840

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218840


namespace division_result_l218_218017

theorem division_result : 180 / 6 / 3 / 2 = 5 := by
  sorry

end division_result_l218_218017


namespace problem_statement_l218_218045

noncomputable def a (n : ℕ) := n^2

theorem problem_statement (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n > 0) :
  x + a n / x ^ n ≥ n + 1 :=
sorry

end problem_statement_l218_218045


namespace square_root_condition_l218_218446

-- Define the condition
def meaningful_square_root (x : ℝ) : Prop :=
  x - 5 ≥ 0

-- Define the theorem that x must be greater than or equal to 5 for the square root to be meaningful
theorem square_root_condition (x : ℝ) : meaningful_square_root x ↔ x ≥ 5 := by
  sorry

end square_root_condition_l218_218446


namespace cells_count_after_9_days_l218_218731

theorem cells_count_after_9_days :
  let a := 5
  let r := 3
  let n := 3
  a * r^(n-1) = 45 :=
by
  let a := 5
  let r := 3
  let n := 3
  sorry

end cells_count_after_9_days_l218_218731


namespace product_of_equal_numbers_l218_218134

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l218_218134


namespace maximum_m_l218_218437

open Finset
open Set

variable (n : ℕ) (hn : n > 1)

noncomputable def S_m (m : ℕ) := {1..(m*n)}

def conditions (m : ℕ) (S : Set (Finset ℕ)) := 
  (|S| = 2 * n) ∧ 
  (∀ s ∈ S, s.card = m) ∧
  (∀ (s1 ∈ S) (s2 ∈ S), s1 ≠ s2 → (s1 ∩ s2).card ≤ 1) ∧
  (∀ x ∈ S_m m, (filter (λ s, x ∈ s) S).card = 2)

theorem maximum_m (m : ℕ) (S : Set (Finset ℕ)) (hS : conditions n m S) : m = n := 
sorry

end maximum_m_l218_218437


namespace number_of_true_statements_l218_218434

theorem number_of_true_statements 
  (a b c : ℝ) 
  (Hc : c ≠ 0) : 
  ((a > b → a * c^2 > b * c^2) ∧ (a * c^2 ≤ b * c^2 → a ≤ b)) ∧ 
  ¬((a * c^2 > b * c^2 → a > b) ∨ (a ≤ b → a * c^2 ≤ b * c^2)) :=
by
  sorry

end number_of_true_statements_l218_218434


namespace gcd_84_108_132_156_l218_218718

theorem gcd_84_108_132_156 : Nat.gcd (Nat.gcd 84 108) (Nat.gcd 132 156) = 12 := 
by
  sorry

end gcd_84_108_132_156_l218_218718


namespace royal_children_count_l218_218343

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l218_218343


namespace time_for_B_alone_to_paint_l218_218380

noncomputable def rate_A := 1 / 4
noncomputable def rate_BC := 1 / 3
noncomputable def rate_AC := 1 / 2
noncomputable def rate_DB := 1 / 6

theorem time_for_B_alone_to_paint :
  (1 / (rate_BC - (rate_AC - rate_A))) = 12 := by
  sorry

end time_for_B_alone_to_paint_l218_218380


namespace find_distance_between_intersection_points_l218_218967

-- Definitions of ellipse and parabola
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 36 = 1
def parabola (x y : ℝ) : Prop := x = (y^2 / (8 * real.sqrt 5)) - 2 * real.sqrt 5

-- Statement to prove the distance between the intersection points of ellipse and parabola
theorem find_distance_between_intersection_points : 
  ∃ (points : list (ℝ × ℝ)), 
    (∀ p, p ∈ points → ellipse p.1 p.2 ∧ parabola p.1 p.2) ∧ 
    ( ∃ y1 y2, (ellipse 0 y1 ∧ ellipse 0 y2 ∧ parabola 0 y1 ∧ parabola 0 y2 ∧ 
     points = [(0, y1), (0, y2)] ) → 
    distance (0, y1) (0, y2) = some_distance) := 
sorry

end find_distance_between_intersection_points_l218_218967


namespace combined_weight_of_Alexa_and_Katerina_l218_218011

variable (total_weight: ℝ) (alexas_weight: ℝ) (michaels_weight: ℝ)

theorem combined_weight_of_Alexa_and_Katerina
  (h1: total_weight = 154)
  (h2: alexas_weight = 46)
  (h3: michaels_weight = 62) :
  total_weight - michaels_weight = 92 :=
by 
  sorry

end combined_weight_of_Alexa_and_Katerina_l218_218011


namespace soccer_school_admission_prob_l218_218555

theorem soccer_school_admission_prob :
  let p_ac := 0.5 in  -- Probability of passing an assistant coach interview
  let p_hc := 0.3 in  -- Probability of passing the head coach's final review
  let p_both_ac := p_ac * p_ac in  -- Probability of passing both assistant coach interviews
  let p_one_ac_then_hc := 2 * (p_ac * (1 - p_ac) * p_hc) in  -- Probability of passing one assistant coach and head coach
  p_both_ac + p_one_ac_then_hc = 0.4 :=

by
  simp [p_ac, p_hc, p_both_ac, p_one_ac_then_hc],
  sorry

end soccer_school_admission_prob_l218_218555


namespace airplane_cost_correct_l218_218010

-- Define the conditions
def initial_amount : ℝ := 5.00
def change_received : ℝ := 0.72

-- Define the cost calculation
def airplane_cost (initial : ℝ) (change : ℝ) : ℝ := initial - change

-- Prove that the airplane cost is $4.28 given the conditions
theorem airplane_cost_correct : airplane_cost initial_amount change_received = 4.28 :=
by
  -- The actual proof goes here
  sorry

end airplane_cost_correct_l218_218010


namespace megatek_manufacturing_percentage_l218_218697

theorem megatek_manufacturing_percentage (angle_manufacturing : ℝ) (full_circle : ℝ) 
  (h1 : angle_manufacturing = 162) (h2 : full_circle = 360) :
  (angle_manufacturing / full_circle) * 100 = 45 :=
by
  sorry

end megatek_manufacturing_percentage_l218_218697


namespace quadratic_b_value_l218_218992

theorem quadratic_b_value (b m : ℝ) (h_b_pos : 0 < b) (h_quad_form : ∀ x, x^2 + b * x + 108 = (x + m)^2 - 4)
  (h_m_pos_sqrt : m = 4 * Real.sqrt 7 ∨ m = -4 * Real.sqrt 7) : b = 8 * Real.sqrt 7 :=
by
  sorry

end quadratic_b_value_l218_218992


namespace vanya_correct_answers_l218_218113

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218113


namespace max_value_expression_l218_218286

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (max_val : ℝ), max_val = 4 - 2 * Real.sqrt 2 ∧ 
  (∀ a b : ℝ, a * b > 0 → (a / (a + b) + 2 * b / (a + 2 * b)) ≤ max_val) := 
sorry

end max_value_expression_l218_218286


namespace LimingFatherAge_l218_218009

theorem LimingFatherAge
  (age month day : ℕ)
  (age_condition : 18 ≤ age ∧ age ≤ 70)
  (product_condition : age * month * day = 2975)
  (valid_month : 1 ≤ month ∧ month ≤ 12)
  (valid_day : 1 ≤ day ∧ day ≤ 31)
  : age = 35 := sorry

end LimingFatherAge_l218_218009


namespace shirts_per_minute_l218_218014

/--
An industrial machine made 8 shirts today and worked for 4 minutes today. 
Prove that the machine can make 2 shirts per minute.
-/
theorem shirts_per_minute (shirts_today : ℕ) (minutes_today : ℕ)
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  (shirts_today / minutes_today) = 2 :=
by sorry

end shirts_per_minute_l218_218014


namespace vasya_days_without_purchases_l218_218191

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218191


namespace samuel_teacups_left_l218_218479

-- Define the initial conditions
def total_boxes := 60
def pans_boxes := 12
def decoration_fraction := 1 / 4
def decoration_trade := 3
def trade_gain := 1
def teacups_per_box := 6 * 4 * 2
def broken_per_pickup := 4

-- Calculate the number of boxes initially containing teacups
def remaining_boxes := total_boxes - pans_boxes
def decoration_boxes := decoration_fraction * remaining_boxes
def initial_teacup_boxes := remaining_boxes - decoration_boxes

-- Adjust the number of teacup boxes after the trade
def teacup_boxes := initial_teacup_boxes + trade_gain

-- Calculate total number of teacups and the number of teacups broken
def total_teacups := teacup_boxes * teacups_per_box
def total_broken := teacup_boxes * broken_per_pickup

-- Calculate the number of teacups left
def teacups_left := total_teacups - total_broken

-- State the theorem
theorem samuel_teacups_left : teacups_left = 1628 := by
  sorry

end samuel_teacups_left_l218_218479


namespace revenue_decrease_percent_l218_218498

theorem revenue_decrease_percent (T C : ℝ) (hT_pos : T > 0) (hC_pos : C > 0) :
  let new_T := 0.75 * T
  let new_C := 1.10 * C
  let original_revenue := T * C
  let new_revenue := new_T * new_C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 17.5 := 
by {
  sorry
}

end revenue_decrease_percent_l218_218498


namespace vanya_correct_answers_l218_218106

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l218_218106


namespace prism_volume_l218_218150

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end prism_volume_l218_218150


namespace julia_total_spend_l218_218453

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end julia_total_spend_l218_218453


namespace Vanya_correct_answers_l218_218120

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l218_218120


namespace power_addition_l218_218801

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218801


namespace geometric_sequence_a2_a6_l218_218294

variable (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
variable (a_geom_seq : ∀ n, a n = a1 * r^(n-1))
variable (h_a4 : a 4 = 4)

theorem geometric_sequence_a2_a6 : a 2 * a 6 = 16 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a2_a6_l218_218294


namespace power_addition_l218_218805

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218805


namespace probability_one_common_number_approx_l218_218063

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l218_218063


namespace no_solutions_triples_l218_218276

theorem no_solutions_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a! + b^3 ≠ 18 + c^3 :=
by
  sorry

end no_solutions_triples_l218_218276


namespace nearly_tricky_7_tiny_count_l218_218551

-- Define a tricky polynomial
def is_tricky (P : Polynomial ℤ) : Prop :=
  Polynomial.eval 4 P = 0

-- Define a k-tiny polynomial
def is_k_tiny (k : ℤ) (P : Polynomial ℤ) : Prop :=
  P.degree ≤ 7 ∧ ∀ i, abs (Polynomial.coeff P i) ≤ k

-- Define a 1-tiny polynomial
def is_1_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 1 P

-- Define a nearly tricky polynomial as the sum of a tricky polynomial and a 1-tiny polynomial
def is_nearly_tricky (P : Polynomial ℤ) : Prop :=
  ∃ Q T : Polynomial ℤ, is_tricky Q ∧ is_1_tiny T ∧ P = Q + T

-- Define a 7-tiny polynomial
def is_7_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 7 P

-- Count the number of nearly tricky 7-tiny polynomials
def count_nearly_tricky_7_tiny : ℕ :=
  -- Simplification: hypothetical function counting the number of polynomials
  sorry

-- The main theorem statement
theorem nearly_tricky_7_tiny_count :
  count_nearly_tricky_7_tiny = 64912347 :=
sorry

end nearly_tricky_7_tiny_count_l218_218551


namespace total_transport_cost_l218_218126

def cost_per_kg : ℝ := 25000
def mass_sensor_g : ℝ := 350
def mass_communication_g : ℝ := 150

theorem total_transport_cost : 
  (cost_per_kg * (mass_sensor_g / 1000) + cost_per_kg * (mass_communication_g / 1000)) = 12500 :=
by
  sorry

end total_transport_cost_l218_218126


namespace binom_two_formula_l218_218910

def binom (n k : ℕ) : ℕ :=
  n.choose k

-- Formalizing the conditions
variable (n : ℕ)
variable (h : n ≥ 2)

-- Stating the problem mathematically in Lean
theorem binom_two_formula :
  binom n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_formula_l218_218910


namespace smallest_m_l218_218326

theorem smallest_m (m : ℤ) :
  (∀ x : ℝ, (3 * x * (m * x - 5) - x^2 + 8) = 0) → (257 - 96 * m < 0) → (m = 3) :=
sorry

end smallest_m_l218_218326


namespace power_of_three_l218_218779

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218779


namespace julias_total_spending_l218_218455

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end julias_total_spending_l218_218455


namespace rotate_cd_to_cd_l218_218524

def rotate180 (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

theorem rotate_cd_to_cd' :
  let C := (-1, 2)
  let C' := (1, -2)
  let D := (3, 2)
  let D' := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' :=
by
  sorry

end rotate_cd_to_cd_l218_218524


namespace power_of_three_l218_218817

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218817


namespace arithmetic_sequence_sum_l218_218406

-- Definitions for the conditions
def a := 70
def d := 3
def n := 10
def l := 97

-- Sum of the arithmetic series
def S := (n / 2) * (a + l)

-- Final calculation
theorem arithmetic_sequence_sum :
  3 * (70 + 73 + 76 + 79 + 82 + 85 + 88 + 91 + 94 + 97) = 2505 :=
by
  -- Lean will calculate these interactively when proving.
  sorry

end arithmetic_sequence_sum_l218_218406


namespace carol_mike_equal_savings_weeks_l218_218941

theorem carol_mike_equal_savings_weeks :
  ∃ x : ℕ, (60 + 9 * x = 90 + 3 * x) ↔ x = 5 := 
by
  sorry

end carol_mike_equal_savings_weeks_l218_218941


namespace royal_family_children_l218_218351

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l218_218351


namespace floor_sqrt_50_l218_218586

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l218_218586


namespace no_snow_probability_l218_218495

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l218_218495


namespace find_pairs_l218_218859

def sequence_a : Nat → Int
| 0 => 0
| 1 => 0
| n+2 => 2 * sequence_a (n+1) - sequence_a n + 2

def sequence_b : Nat → Int
| 0 => 8
| 1 => 8
| n+2 => 2 * sequence_b (n+1) - sequence_b n

theorem find_pairs :
  (sequence_a 1992 = 31872 ∧ sequence_b 1992 = 31880) ∨
  (sequence_a 1992 = -31872 ∧ sequence_b 1992 = -31864) :=
sorry

end find_pairs_l218_218859


namespace exact_two_solutions_l218_218602

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l218_218602


namespace customer_paid_l218_218708

theorem customer_paid (cost_price : ℕ) (markup_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 6672 → markup_percent = 25 → selling_price = cost_price + (markup_percent * cost_price / 100) → selling_price = 8340 :=
by
  intros h_cost_price h_markup_percent h_selling_price
  rw [h_cost_price, h_markup_percent] at h_selling_price
  exact h_selling_price

end customer_paid_l218_218708


namespace probability_of_selecting_one_painted_face_and_one_unpainted_face_l218_218575

noncomputable def probability_of_specific_selection :
  ℕ → ℕ → ℕ → ℚ
| total_cubes, painted_face_cubes, unpainted_face_cubes =>
  let total_pairs := (total_cubes * (total_cubes - 1)) / 2
  let success_pairs := painted_face_cubes * unpainted_face_cubes
  success_pairs / total_pairs

theorem probability_of_selecting_one_painted_face_and_one_unpainted_face :
  probability_of_specific_selection 36 13 17 = 221 / 630 :=
by
  sorry

end probability_of_selecting_one_painted_face_and_one_unpainted_face_l218_218575


namespace vasya_days_without_purchase_l218_218198

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218198


namespace percentage_of_rotten_bananas_l218_218390

theorem percentage_of_rotten_bananas :
  ∀ (total_oranges total_bananas : ℕ) 
    (percent_rotten_oranges : ℝ) 
    (percent_good_fruits : ℝ), 
  total_oranges = 600 → total_bananas = 400 → 
  percent_rotten_oranges = 0.15 → percent_good_fruits = 0.89 → 
  (100 - (((percent_good_fruits * (total_oranges + total_bananas)) - 
  ((1 - percent_rotten_oranges) * total_oranges)) / total_bananas) * 100) = 5 := 
by
  intros total_oranges total_bananas percent_rotten_oranges percent_good_fruits 
  intro ho hb hro hpf 
  sorry

end percentage_of_rotten_bananas_l218_218390


namespace weekly_goal_cans_l218_218526

theorem weekly_goal_cans : (20 +  (20 * 1.5) + (20 * 2) + (20 * 2.5) + (20 * 3)) = 200 := by
  sorry

end weekly_goal_cans_l218_218526


namespace Dave_guitar_strings_replacement_l218_218410

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end Dave_guitar_strings_replacement_l218_218410


namespace angle_sum_straight_line_l218_218279

  theorem angle_sum_straight_line (x : ℝ) (h : 90 + x + 20 = 180) : x = 70 :=
  by
    sorry
  
end angle_sum_straight_line_l218_218279


namespace simplify_fraction_l218_218308

-- Define the numerator and denominator
def numerator := 5^4 + 5^2
def denominator := 5^3 - 5

-- Define the simplified fraction
def simplified_fraction := 65 / 12

-- The proof problem statement
theorem simplify_fraction : (numerator / denominator) = simplified_fraction := 
by 
   -- Proof will go here
   sorry

end simplify_fraction_l218_218308


namespace max_lessons_l218_218889

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end max_lessons_l218_218889


namespace pigeons_problem_l218_218321

theorem pigeons_problem
  (x y : ℕ)
  (h1 : 6 * y + 3 = x)
  (h2 : 8 * y = x + 5) : x = 27 := 
sorry

end pigeons_problem_l218_218321


namespace number_of_children_l218_218370

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l218_218370


namespace evaluate_expression_at_values_l218_218482

theorem evaluate_expression_at_values (x y : ℤ) (h₁ : x = 1) (h₂ : y = -2) :
  (-2 * x ^ 2 + 2 * x - y) = 2 :=
by
  subst h₁
  subst h₂
  sorry

end evaluate_expression_at_values_l218_218482


namespace power_calculation_l218_218831

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218831


namespace triangle_cot_tan_identity_l218_218268

theorem triangle_cot_tan_identity 
  (a b c : ℝ) 
  (h : a^2 + b^2 = 2018 * c^2)
  (A B C : ℝ) 
  (triangle_ABC : ∀ (a b c : ℝ), a + b + c = π) 
  (cot_A : ℝ := Real.cos A / Real.sin A) 
  (cot_B : ℝ := Real.cos B / Real.sin B) 
  (tan_C : ℝ := Real.sin C / Real.cos C) :
  (cot_A + cot_B) * tan_C = -2 / 2017 :=
by sorry

end triangle_cot_tan_identity_l218_218268


namespace divisibility_problem_l218_218464

theorem divisibility_problem (q : ℕ) (hq : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬((q + 2)^(q - 3) + 1) % (q - 4) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % q = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 6) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 3) = 0 := sorry

end divisibility_problem_l218_218464


namespace photo_album_pages_l218_218556

noncomputable def P1 := 0
noncomputable def P2 := 10
noncomputable def remaining_pages := 20

theorem photo_album_pages (photos total_pages photos_per_page_set1 photos_per_page_set2 photos_per_page_remaining : ℕ) 
  (h1 : photos = 100)
  (h2 : total_pages = 30)
  (h3 : photos_per_page_set1 = 3)
  (h4 : photos_per_page_set2 = 4)
  (h5 : photos_per_page_remaining = 3) : 
  P1 = 0 ∧ P2 = 10 ∧ remaining_pages = 20 :=
by
  sorry

end photo_album_pages_l218_218556


namespace power_equality_l218_218842

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218842


namespace valve_solution_l218_218245

noncomputable def valve_problem : Prop :=
  ∀ (x y z : ℝ),
  (1 / (x + y + z) = 2) →
  (1 / (x + z) = 4) →
  (1 / (y + z) = 3) →
  (1 / (x + y) = 2.4)

theorem valve_solution : valve_problem :=
by
  -- proof omitted
  intros x y z h1 h2 h3
  sorry

end valve_solution_l218_218245


namespace power_calculation_l218_218835

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218835


namespace car_value_reduction_l218_218665

/-- Jocelyn bought a car 3 years ago at $4000. 
If the car's value has reduced by 30%, calculate the current value of the car. 
Prove that it is equal to $2800. -/
theorem car_value_reduction (initial_value : ℝ) (reduction_percentage : ℝ) (current_value : ℝ) 
  (h_initial : initial_value = 4000)
  (h_reduction : reduction_percentage = 30)
  (h_current : current_value = initial_value - (reduction_percentage / 100) * initial_value) :
  current_value = 2800 :=
by
  -- Formal proof goes here
  sorry

end car_value_reduction_l218_218665


namespace power_calculation_l218_218836

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218836


namespace school_seat_payment_l218_218931

def seat_cost (num_rows : ℕ) (seats_per_row : ℕ) (cost_per_seat : ℕ) (discount : ℕ → ℕ → ℕ) : ℕ :=
  let total_seats := num_rows * seats_per_row
  let total_cost := total_seats * cost_per_seat
  let groups_of_ten := total_seats / 10
  let total_discount := groups_of_ten * discount 10 cost_per_seat
  total_cost - total_discount

-- Define the discount function as 10% of the cost of a group of 10 seats
def discount (group_size : ℕ) (cost_per_seat : ℕ) : ℕ := (group_size * cost_per_seat) / 10

theorem school_seat_payment :
  seat_cost 5 8 30 discount = 1080 :=
sorry

end school_seat_payment_l218_218931


namespace unique_function_eq_id_l218_218306

theorem unique_function_eq_id (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f x = x^2 * f (1 / x)) →
  (∀ x y : ℝ, f (x + y) = f x + f y) →
  (f 1 = 1) →
  (∀ x : ℝ, f x = x) :=
by
  intro h1 h2 h3
  sorry

end unique_function_eq_id_l218_218306


namespace max_log_sum_value_l218_218261

noncomputable def max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) : ℝ :=
  Real.log x + Real.log y

theorem max_log_sum_value : ∀ (x y : ℝ), x > 0 → y > 0 → x + 4 * y = 40 → max_log_sum x y sorry sorry sorry = 2 :=
by
  intro x y h1 h2 h3
  sorry

end max_log_sum_value_l218_218261


namespace exponent_power_identity_l218_218789

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218789


namespace royal_family_children_l218_218352

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l218_218352


namespace roots_of_equation_l218_218611

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l218_218611


namespace gcd_problem_l218_218461

-- Define the conditions
def a (d : ℕ) : ℕ := d - 3
def b (d : ℕ) : ℕ := d - 2
def c (d : ℕ) : ℕ := d - 1

-- Define the number formed by digits in the specific form
def abcd (d : ℕ) : ℕ := 1000 * a d + 100 * b d + 10 * c d + d
def dcba (d : ℕ) : ℕ := 1000 * d + 100 * c d + 10 * b d + a d

-- Summing the two numbers
def num_sum (d : ℕ) : ℕ := abcd d + dcba d

-- The GCD of all num_sum(d) where d ranges from 3 to 9
def gcd_of_nums : ℕ := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (num_sum 3) (num_sum 4)) (num_sum 5)) (num_sum 6)) (Nat.gcd (num_sum 7) (Nat.gcd (num_sum 8) (num_sum 9)))

theorem gcd_problem : gcd_of_nums = 1111 := sorry

end gcd_problem_l218_218461


namespace train_cross_post_time_proof_l218_218936

noncomputable def train_cross_post_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length_m / speed_ms

theorem train_cross_post_time_proof : train_cross_post_time 40 190.0152 = 17.1 := by
  sorry

end train_cross_post_time_proof_l218_218936


namespace power_calculation_l218_218837

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218837


namespace royal_family_children_l218_218359

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l218_218359


namespace rope_cut_probability_l218_218003

theorem rope_cut_probability (L : ℝ) (cut_position : ℝ) (P : ℝ) :
  L = 4 → (∀ cut_position, 0 ≤ cut_position ∧ cut_position ≤ L →
  (cut_position ≥ 1.5 ∧ (L - cut_position) ≥ 1.5)) → P = 1 / 4 :=
by
  intros hL hcut
  sorry

end rope_cut_probability_l218_218003


namespace dolls_in_dollhouses_l218_218998

theorem dolls_in_dollhouses :
  let total_ways := Nat.choose 7 2 * 6 * Nat.factorial 5 in
  total_ways = 15120 := by
  sorry

end dolls_in_dollhouses_l218_218998


namespace find_t_plus_a3_l218_218317

noncomputable def geometric_sequence_sum (n : ℕ) (t : ℤ) : ℤ :=
  3 ^ n + t

noncomputable def a_1 (t : ℤ) : ℤ :=
  geometric_sequence_sum 1 t

noncomputable def a_2 (t : ℤ) : ℤ :=
  geometric_sequence_sum 2 t - geometric_sequence_sum 1 t

noncomputable def a_3 (t : ℤ) : ℤ :=
  geometric_sequence_sum 3 t - geometric_sequence_sum 2 t

theorem find_t_plus_a3 (t : ℤ) : t + a_3 t = 17 :=
sorry

end find_t_plus_a3_l218_218317


namespace ratio_of_volumes_l218_218154

def cone_radius_X := 10
def cone_height_X := 15
def cone_radius_Y := 15
def cone_height_Y := 10

noncomputable def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h

noncomputable def volume_X := volume_cone cone_radius_X cone_height_X
noncomputable def volume_Y := volume_cone cone_radius_Y cone_height_Y

theorem ratio_of_volumes : volume_X / volume_Y = 2 / 3 := sorry

end ratio_of_volumes_l218_218154


namespace vector_perpendicular_solve_x_l218_218765

theorem vector_perpendicular_solve_x
  (x : ℝ)
  (a : ℝ × ℝ := (4, 8))
  (b : ℝ × ℝ := (x, 4))
  (h : 4 * x + 8 * 4 = 0) :
  x = -8 :=
sorry

end vector_perpendicular_solve_x_l218_218765


namespace children_count_l218_218372

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l218_218372


namespace B_needs_days_l218_218545

theorem B_needs_days (A_rate B_rate Combined_rate : ℝ) (x : ℝ) (W : ℝ) (h1: A_rate = W / 140)
(h2: B_rate = W / (3 * x)) (h3 : Combined_rate = 60 * W) (h4 : Combined_rate = A_rate + B_rate) :
 x = 140 / 25197 :=
by
  sorry

end B_needs_days_l218_218545


namespace power_calculation_l218_218828

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218828


namespace royal_children_l218_218341

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l218_218341


namespace distinct_solution_condition_l218_218610

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l218_218610


namespace travel_distance_bus_l218_218530

theorem travel_distance_bus (D P T B : ℝ) 
    (hD : D = 1800)
    (hP : P = D / 3)
    (hT : T = (2 / 3) * B)
    (h_total : P + T + B = D) :
    B = 720 := 
by
    sorry

end travel_distance_bus_l218_218530


namespace find_alpha_l218_218309

variable (α β k : ℝ)

axiom h1 : α * β = k
axiom h2 : α = -4
axiom h3 : β = -8
axiom k_val : k = 32
axiom β_val : β = 12

theorem find_alpha (h1 : α * β = k) (h2 : α = -4) (h3 : β = -8) (k_val : k = 32) (β_val : β = 12) :
  α = 8 / 3 :=
sorry

end find_alpha_l218_218309


namespace royal_family_children_l218_218355

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l218_218355


namespace power_of_three_l218_218787

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218787


namespace orthogonal_projection_l218_218459

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_squared := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_squared * u.1, dot_uv / norm_u_squared * u.2)

theorem orthogonal_projection
  (a b : ℝ × ℝ)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj a (4, -4) = (-4/5, -8/5)) :
  proj b (4, -4) = (24/5, -12/5) :=
sorry

end orthogonal_projection_l218_218459


namespace fraction_division_l218_218722

theorem fraction_division :
  (3 / 7) / (2 / 5) = (15 / 14) :=
by
  sorry

end fraction_division_l218_218722


namespace relationship_among_y_values_l218_218476

theorem relationship_among_y_values (c y1 y2 y3 : ℝ) :
  (-1)^2 - 2 * (-1) + c = y1 →
  (3)^2 - 2 * 3 + c = y2 →
  (5)^2 - 2 * 5 + c = y3 →
  y1 = y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  sorry

end relationship_among_y_values_l218_218476


namespace simpl_eval_l218_218124

variable (a b : ℚ)

theorem simpl_eval (h_a : a = 1/2) (h_b : b = -1/3) :
    5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (- a * b ^ 2 + 3 * a ^ 2 * b) = -11 / 36 := by
  sorry

end simpl_eval_l218_218124


namespace bicentric_quad_lemma_l218_218964

-- Define the properties and radii of the bicentric quadrilateral
variables (KLMN : Type) (r ρ h : ℝ)

-- Assuming quadrilateral KLMN is bicentric with given radii
def is_bicentric (KLMN : Type) := true

-- State the theorem we wish to prove
theorem bicentric_quad_lemma (br : is_bicentric KLMN) : 
  (1 / (ρ + h) ^ 2) + (1 / (ρ - h) ^ 2) = (1 / r ^ 2) :=
sorry

end bicentric_quad_lemma_l218_218964


namespace total_money_together_l218_218565

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l218_218565


namespace positive_integer_triplets_l218_218751

theorem positive_integer_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_lcm : a + b + c = Nat.lcm a (Nat.lcm b c)) :
  (∃ k, k ≥ 1 ∧ a = k ∧ b = 2 * k ∧ c = 3 * k) :=
sorry

end positive_integer_triplets_l218_218751


namespace total_games_played_l218_218417

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end total_games_played_l218_218417


namespace product_of_equal_numbers_l218_218133

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l218_218133


namespace q_evaluation_at_3_point_5_l218_218939

def q (x : ℝ) : ℝ :=
  |x - 3|^(1/3) + 2*|x - 3|^(1/5) + |x - 3|^(1/7)

theorem q_evaluation_at_3_point_5 : q 3.5 = 3 :=
by
  sorry

end q_evaluation_at_3_point_5_l218_218939


namespace power_of_three_l218_218809

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218809


namespace intersection_with_x_axis_l218_218757

theorem intersection_with_x_axis (t : ℝ) (x y : ℝ) 
  (h1 : x = -2 + 5 * t) 
  (h2 : y = 1 - 2 * t) 
  (h3 : y = 0) : x = 1 / 2 := 
by 
  sorry

end intersection_with_x_axis_l218_218757


namespace sum_of_coordinates_D_l218_218305

theorem sum_of_coordinates_D
    (C N D : ℝ × ℝ) 
    (hC : C = (10, 5))
    (hN : N = (4, 9))
    (h_midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
    C.1 + D.1 + (C.2 + D.2) = 22 :=
  by sorry

end sum_of_coordinates_D_l218_218305


namespace find_c_l218_218142

theorem find_c (c q : ℤ) (h : ∃ (a b : ℤ), (3*x^3 + c*x + 9 = (x^2 + q*x + 1) * (a*x + b))) : c = -24 :=
sorry

end find_c_l218_218142


namespace total_amount_is_200_l218_218567

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l218_218567


namespace Carla_total_counts_l218_218421

def Monday_counts := (60 * 2) + (120 * 2) + (10 * 2)
def Tuesday_counts := (60 * 3) + (120 * 2) + (10 * 1)
def Wednesday_counts := (80 * 4) + (24 * 5)
def Thursday_counts := (60 * 1) + (80 * 2) + (120 * 3) + (10 * 4) + (24 * 5)
def Friday_counts := (60 * 1) + (120 * 2) + (80 * 2) + (10 * 3) + (24 * 3)

def total_counts := Monday_counts + Tuesday_counts + Wednesday_counts + Thursday_counts + Friday_counts

theorem Carla_total_counts : total_counts = 2552 :=
by 
  sorry

end Carla_total_counts_l218_218421


namespace apple_price_l218_218402

theorem apple_price :
  ∀ (l q : ℝ), 
    (10 * l = 3.62) →
    (30 * l + 3 * q = 11.67) →
    (30 * l + 6 * q = 12.48) :=
by
  intros l q h₁ h₂
  -- The proof would go here with the steps, but for now we use sorry.
  sorry

end apple_price_l218_218402


namespace largest_subset_size_l218_218298

theorem largest_subset_size (T : Finset ℕ) (h : ∀ x ∈ T, ∀ y ∈ T, x ≠ y → (x - y) % 2021 ≠ 5 ∧ (x - y) % 2021 ≠ 8) :
  T.card ≤ 918 := sorry

end largest_subset_size_l218_218298


namespace min_value_of_polynomial_l218_218269

theorem min_value_of_polynomial :
  ∃ x : ℝ, ∀ y, y = (x - 16) * (x - 14) * (x + 14) * (x + 16) → y ≥ -900 :=
by
  sorry

end min_value_of_polynomial_l218_218269


namespace inequality_addition_l218_218460

-- Definitions and Conditions
variables (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c > d)

-- Theorem statement: Prove that a + c > b + d
theorem inequality_addition (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := 
sorry

end inequality_addition_l218_218460


namespace unique_lcm_gcd_pairs_l218_218952

open Nat

theorem unique_lcm_gcd_pairs :
  { (a, b) : ℕ × ℕ | lcm a b = gcd a b + 19 } = 
  { (1, 20), (20, 1), (4, 5), (5, 4), (19, 38), (38, 19) } :=
by
  sorry

end unique_lcm_gcd_pairs_l218_218952


namespace instantaneous_rate_of_change_at_x1_l218_218002

open Real

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - x^2 + 8

theorem instantaneous_rate_of_change_at_x1 : deriv f 1 = -1 := by
  sorry

end instantaneous_rate_of_change_at_x1_l218_218002


namespace exact_two_solutions_l218_218600

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l218_218600


namespace combined_area_percentage_l218_218651

theorem combined_area_percentage (D_S : ℝ) (D_R : ℝ) (D_T : ℝ) (A_S A_R A_T : ℝ)
  (h1 : D_R = 0.20 * D_S)
  (h2 : D_T = 0.40 * D_R)
  (h3 : A_R = Real.pi * (D_R / 2) ^ 2)
  (h4 : A_T = Real.pi * (D_T / 2) ^ 2)
  (h5 : A_S = Real.pi * (D_S / 2) ^ 2) :
  ((A_R + A_T) / A_S) * 100 = 4.64 := by
  sorry

end combined_area_percentage_l218_218651


namespace cash_price_eq_8000_l218_218685

noncomputable def cash_price (d m s : ℕ) : ℕ :=
  d + 30 * m - s

theorem cash_price_eq_8000 :
  cash_price 3000 300 4000 = 8000 :=
by
  -- Proof omitted.
  sorry

end cash_price_eq_8000_l218_218685


namespace freddy_talk_time_dad_l218_218038

-- Conditions
def localRate : ℝ := 0.05
def internationalRate : ℝ := 0.25
def talkTimeBrother : ℕ := 31
def totalCost : ℝ := 10.0

-- Goal: Prove the duration of Freddy's local call to his dad is 45 minutes
theorem freddy_talk_time_dad : 
  ∃ (talkTimeDad : ℕ), 
    talkTimeDad = 45 ∧
    totalCost = (talkTimeBrother : ℝ) * internationalRate + (talkTimeDad : ℝ) * localRate := 
by
  sorry

end freddy_talk_time_dad_l218_218038


namespace Katie_homework_problems_l218_218993

theorem Katie_homework_problems :
  let finished_problems := 5
  let remaining_problems := 4
  let total_problems := finished_problems + remaining_problems
  total_problems = 9 :=
by
  sorry

end Katie_homework_problems_l218_218993


namespace hyperbola_eccentricity_l218_218709

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h_right_focus : ∀ x y, x = c ∧ y = 0)
    (h_circle : ∀ x y, (x - c)^2 + y^2 = 4 * a^2)
    (h_tangent : ∀ x y, x = c ∧ y = 0 → (x^2 + y^2 = a^2 + b^2))
    : ∃ e : ℝ, e = sqrt 5 := by sorry

end hyperbola_eccentricity_l218_218709


namespace product_div_by_six_l218_218278

theorem product_div_by_six (A B C : ℤ) (h1 : A^2 + B^2 = C^2) 
  (h2 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 4 * k + 2) 
  (h3 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 3 * k + 2) : 
  6 ∣ (A * B) :=
sorry

end product_div_by_six_l218_218278


namespace bucket_problem_l218_218712

theorem bucket_problem 
  (C : ℝ) -- original capacity of the bucket
  (N : ℕ) -- number of buckets required to fill the tank with the original bucket size
  (h : N * C = 25 * (2/5) * C) : 
  N = 10 :=
by
  sorry

end bucket_problem_l218_218712


namespace solve_equation_l218_218916

theorem solve_equation :
  let lhs := ((4 - 3.5 * (15/7 - 6/5)) / 0.16)
  let rhs := ((23/7 - (3/14) / (1/6)) / (3467/84 - 2449/60))
  lhs / 1 = rhs :=
by
  sorry

end solve_equation_l218_218916


namespace complex_inv_condition_l218_218277

theorem complex_inv_condition (i : ℂ) (h : i^2 = -1) : (i - 2 * i⁻¹)⁻¹ = -i / 3 :=
by
  sorry

end complex_inv_condition_l218_218277


namespace problem_statement_l218_218644

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l218_218644


namespace tickets_needed_l218_218715

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end tickets_needed_l218_218715


namespace first_bag_weight_l218_218246

def weight_of_first_bag (initial_weight : ℕ) (second_bag : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight - second_bag - initial_weight

theorem first_bag_weight : weight_of_first_bag 15 10 40 = 15 :=
by
  unfold weight_of_first_bag
  sorry

end first_bag_weight_l218_218246


namespace vasya_no_purchase_days_l218_218173

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218173


namespace perimeter_inequality_l218_218927

-- Define the problem parameters
variables {R S : ℝ}  -- radius and area of the inscribed polygon
variables (P : ℝ)    -- perimeter of the convex polygon formed by chosen points

-- Define the various conditions
def circle_with_polygon (r : ℝ) := r > 0 -- Circle with positive radius
def polygon_with_area (s : ℝ) := s > 0 -- Polygon with positive area

-- Main theorem to be proven
theorem perimeter_inequality (hR : circle_with_polygon R) (hS : polygon_with_area S) :
  P ≥ (2 * S / R) :=
sorry

end perimeter_inequality_l218_218927


namespace eq_sum_of_factorial_fractions_l218_218034

theorem eq_sum_of_factorial_fractions (b2 b3 b5 b6 b7 b8 : ℤ)
  (h2 : 0 ≤ b2 ∧ b2 < 2)
  (h3 : 0 ≤ b3 ∧ b3 < 3)
  (h5 : 0 ≤ b5 ∧ b5 < 5)
  (h6 : 0 ≤ b6 ∧ b6 < 6)
  (h7 : 0 ≤ b7 ∧ b7 < 7)
  (h8 : 0 ≤ b8 ∧ b8 < 8)
  (h_eq : (3 / 8 : ℚ) = (b2 / (2 * 1) + b3 / (3 * 2 * 1) + b5 / (5 * 4 * 3 * 2 * 1) +
                          b6 / (6 * 5 * 4 * 3 * 2 * 1) + b7 / (7 * 6 * 5 * 4 * 3 * 2 * 1) +
                          b8 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) : ℚ)) :
  b2 + b3 + b5 + b6 + b7 + b8 = 12 :=
by
  sorry

end eq_sum_of_factorial_fractions_l218_218034


namespace children_count_l218_218375

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l218_218375


namespace probability_of_one_common_l218_218076

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l218_218076


namespace parallel_vectors_l218_218973

open Real

theorem parallel_vectors (k : ℝ) 
  (a : ℝ × ℝ := (k-1, 1)) 
  (b : ℝ × ℝ := (k+3, k)) 
  (h : a.1 * b.2 = a.2 * b.1) : 
  k = 3 ∨ k = -1 :=
by
  sorry

end parallel_vectors_l218_218973


namespace royal_family_children_l218_218354

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l218_218354


namespace total_pupils_count_l218_218084

theorem total_pupils_count (girls boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) : girls + boys = 926 :=
by 
  sorry

end total_pupils_count_l218_218084


namespace Vanya_correct_answers_l218_218107

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l218_218107


namespace value_x_plus_2y_plus_3z_l218_218044

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end value_x_plus_2y_plus_3z_l218_218044


namespace five_digit_palindromes_count_l218_218414

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end five_digit_palindromes_count_l218_218414


namespace point_P_location_l218_218985

theorem point_P_location (a b : ℝ) : (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → a^2 + b^2 > 1 :=
by sorry

end point_P_location_l218_218985


namespace solution_80_percent_needs_12_ounces_l218_218162

theorem solution_80_percent_needs_12_ounces:
  ∀ (x y: ℝ), (x + y = 40) → (0.30 * x + 0.80 * y = 0.45 * 40) → (y = 12) :=
by
  intros x y h1 h2
  sorry

end solution_80_percent_needs_12_ounces_l218_218162


namespace vasya_did_not_buy_anything_days_l218_218220

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218220


namespace weight_order_l218_218295

variable {P Q R S T : ℕ}

theorem weight_order
    (h1 : Q + S = 1200)
    (h2 : R + T = 2100)
    (h3 : Q + T = 800)
    (h4 : Q + R = 900)
    (h5 : P + T = 700)
    (hP : P < 1000)
    (hQ : Q < 1000)
    (hR : R < 1000)
    (hS : S < 1000)
    (hT : T < 1000) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
sorry

end weight_order_l218_218295


namespace fraction_of_boys_among_attendees_l218_218475

def boys : ℕ := sorry
def girls : ℕ := boys
def teachers : ℕ := boys / 2

def boys_attending : ℕ := (4 * boys) / 5
def girls_attending : ℕ := girls / 2
def teachers_attending : ℕ := teachers / 10

theorem fraction_of_boys_among_attendees :
  (boys_attending : ℚ) / (boys_attending + girls_attending + teachers_attending) = 16 / 27 := sorry

end fraction_of_boys_among_attendees_l218_218475


namespace probability_one_common_number_approx_l218_218062

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l218_218062


namespace train_length_is_correct_l218_218395

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l218_218395


namespace ratio_of_sum_of_terms_l218_218966

variable {α : Type*}
variable [Field α]

def geometric_sequence (a : ℕ → α) := ∃ r, ∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) := S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem ratio_of_sum_of_terms (a : ℕ → α) (S : ℕ → α)
  (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h : S 8 / S 4 = 4) :
  S 12 / S 4 = 13 :=
by
  sorry

end ratio_of_sum_of_terms_l218_218966


namespace total_fat_l218_218383

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end total_fat_l218_218383


namespace tenth_pirate_receives_exactly_1296_coins_l218_218735

noncomputable def pirate_coins (n : ℕ) : ℕ :=
  if n = 0 then 0
  else Nat.factorial 9 / 11^9 * 11^(10 - n)

theorem tenth_pirate_receives_exactly_1296_coins :
  pirate_coins 10 = 1296 :=
sorry

end tenth_pirate_receives_exactly_1296_coins_l218_218735


namespace vasya_did_not_buy_anything_days_l218_218217

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218217


namespace ratio_to_percentage_l218_218677

theorem ratio_to_percentage (x y : ℚ) (h : (2/3 * x) / (4/5 * y) = 5 / 6) : (5 / 6 : ℚ) * 100 = 83.33 :=
by
  sorry

end ratio_to_percentage_l218_218677


namespace sequence_sum_l218_218947

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end sequence_sum_l218_218947


namespace one_minus_repeating_eight_l218_218033

-- Define the repeating decimal
def repeating_eight : Real := 0.8888888888 -- repeating of 8

-- Define the repeating decimal as a fraction
def repeating_eight_as_fraction : Real := 8 / 9

-- The proof statement
theorem one_minus_repeating_eight : 1 - repeating_eight = 1 / 9 := by
  -- Since proof is not required, we use sorry
  sorry

end one_minus_repeating_eight_l218_218033


namespace monthly_average_growth_rate_price_reduction_for_profit_l218_218381

-- Part 1: Monthly average growth rate of sales volume
theorem monthly_average_growth_rate (x : ℝ) : 
  256 * (1 + x) ^ 2 = 400 ↔ x = 0.25 :=
by
  sorry

-- Part 2: Price reduction to achieve profit of $4250
theorem price_reduction_for_profit (m : ℝ) : 
  (40 - m - 25) * (400 + 5 * m) = 4250 ↔ m = 5 :=
by
  sorry

end monthly_average_growth_rate_price_reduction_for_profit_l218_218381


namespace board_game_cost_l218_218089

theorem board_game_cost
  (v h : ℝ)
  (h1 : 3 * v = h + 490)
  (h2 : 5 * v = 2 * h + 540) :
  h = 830 := by
  sorry

end board_game_cost_l218_218089


namespace number_of_tiles_l218_218739

theorem number_of_tiles (w l : ℕ) (h1 : 2 * w + 2 * l - 4 = (w * l - (2 * w + 2 * l - 4)))
  (h2 : w > 0) (h3 : l > 0) : w * l = 48 ∨ w * l = 60 :=
by
  sorry

end number_of_tiles_l218_218739


namespace sixth_smallest_number_l218_218426

def valid_digits : Finset ℕ := {0, 4, 6, 7, 8}

def is_valid_number (n : ℕ) : Prop :=
let digits := n.digits 10 in digits.to_finset ⊆ valid_digits ∧ digits.length = 5 ∧ 0 ∉ digits.reverse.tail

def sixth_smallest : ℕ := 40876

theorem sixth_smallest_number : ∃ n, is_valid_number n ∧ (finset.sort (λ x y, x < y) {n | is_valid_number n}).nth 5 = some sixth_smallest :=
sorry

end sixth_smallest_number_l218_218426


namespace white_marbles_multiple_of_8_l218_218407

-- Definitions based on conditions
def blue_marbles : ℕ := 16
def num_groups : ℕ := 8

-- Stating the problem
theorem white_marbles_multiple_of_8 (white_marbles : ℕ) :
  (blue_marbles + white_marbles) % num_groups = 0 → white_marbles % num_groups = 0 :=
by
  sorry

end white_marbles_multiple_of_8_l218_218407


namespace complex_number_identity_l218_218620

theorem complex_number_identity (a b : ℝ) (i : ℂ) (h : (a + i) * (1 + i) = b * i) : a + b * i = 1 + 2 * i := 
by
  sorry

end complex_number_identity_l218_218620


namespace total_money_together_l218_218564

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l218_218564


namespace Vanya_correct_answers_l218_218122

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l218_218122


namespace has_exactly_two_solutions_iff_l218_218604

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l218_218604


namespace t_minus_d_l218_218908

-- Define amounts paid by Tom, Dorothy, and Sammy
def tom_paid : ℕ := 140
def dorothy_paid : ℕ := 90
def sammy_paid : ℕ := 220

-- Define the total amount and required equal share
def total_paid : ℕ := tom_paid + dorothy_paid + sammy_paid
def equal_share : ℕ := total_paid / 3

-- Define the amounts t and d where Tom and Dorothy balance the costs by paying Sammy
def t : ℤ := equal_share - tom_paid -- Amount Tom gave to Sammy
def d : ℤ := equal_share - dorothy_paid -- Amount Dorothy gave to Sammy

-- Prove that t - d = -50
theorem t_minus_d : t - d = -50 := by
  sorry

end t_minus_d_l218_218908


namespace degrees_for_salaries_l218_218733

def transportation_percent : ℕ := 15
def research_development_percent : ℕ := 9
def utilities_percent : ℕ := 5
def equipment_percent : ℕ := 4
def supplies_percent : ℕ := 2
def total_percent : ℕ := 100
def total_degrees : ℕ := 360

theorem degrees_for_salaries :
  total_degrees * (total_percent - (transportation_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent)) / total_percent = 234 := 
by
  sorry

end degrees_for_salaries_l218_218733


namespace total_games_played_l218_218420

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end total_games_played_l218_218420


namespace knights_and_liars_l218_218472

-- Define the conditions: 
variables (K L : ℕ) 

-- Total number of council members is 101
def total_members : Prop := K + L = 101

-- Inequality conditions
def knight_inequality : Prop := L > (K + L - 1) / 2
def liar_inequality : Prop := K <= (K + L - 1) / 2

-- The theorem we need to prove
theorem knights_and_liars (K L : ℕ) (h1 : total_members K L) (h2 : knight_inequality K L) (h3 : liar_inequality K L) : K = 50 ∧ L = 51 :=
by {
  sorry
}

end knights_and_liars_l218_218472


namespace range_of_m_l218_218675

open Set Real

theorem range_of_m (M N : Set ℝ) (m : ℝ) :
    (M = {x | x ≤ m}) →
    (N = {y | ∃ x : ℝ, y = 2^(-x)}) →
    (M ∩ N ≠ ∅) → m > 0 := by
  intros hM hN hMN
  sorry

end range_of_m_l218_218675


namespace quadratic_solution_l218_218443

theorem quadratic_solution (a b : ℚ) (h : a * 1^2 + b * 1 + 1 = 0) : 3 - a - b = 4 := 
by
  sorry

end quadratic_solution_l218_218443


namespace expansion_correct_l218_218950

-- Define the polynomials
def poly1 (z : ℤ) : ℤ := 3 * z^2 + 4 * z - 5
def poly2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2

-- Define the expected expanded polynomial
def expanded_poly (z : ℤ) : ℤ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- The theorem that proves the equivalence of the expanded form
theorem expansion_correct (z : ℤ) : (poly1 z) * (poly2 z) = expanded_poly z := by
  sorry

end expansion_correct_l218_218950


namespace circle_packing_line_equation_l218_218881

theorem circle_packing_line_equation
  (d : ℝ) (n1 n2 n3 : ℕ) (slope : ℝ)
  (l_intersects_tangencies : ℝ → ℝ → Prop)
  (l_divides_R : Prop)
  (gcd_condition : ℕ → ℕ → ℕ → ℕ)
  (a b c : ℕ)
  (a_pos : 0 < a) (b_neg : b < 0) (c_pos : 0 < c)
  (gcd_abc : gcd_condition a b c = 1)
  (correct_equation_format : Prop) :
  n1 = 4 ∧ n2 = 4 ∧ n3 = 2 →
  d = 2 →
  slope = 5 →
  l_divides_R →
  l_intersects_tangencies 1 1 →
  l_intersects_tangencies 4 6 → 
  correct_equation_format → 
  a^2 + b^2 + c^2 = 42 :=
by sorry

end circle_packing_line_equation_l218_218881


namespace negation_of_implication_l218_218047

theorem negation_of_implication {r p q : Prop} :
  ¬ (r → (p ∨ q)) ↔ (¬ r → (¬ p ∧ ¬ q)) :=
by sorry

end negation_of_implication_l218_218047


namespace vasya_days_without_purchase_l218_218202

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218202


namespace identify_a_b_l218_218242

theorem identify_a_b (a b : ℝ) (h : ∀ x y : ℝ, (⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋)) : 
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) :=
sorry

end identify_a_b_l218_218242


namespace power_equality_l218_218843

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218843


namespace power_addition_l218_218802

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218802


namespace mixed_number_division_l218_218940

theorem mixed_number_division :
  (4 + 2 / 3 + 5 + 1 / 4) / (3 + 1 / 2 - 2 + 3 / 5) = 11 + 1 / 54 :=
by
  sorry

end mixed_number_division_l218_218940


namespace max_k_value_l218_218848

theorem max_k_value (x y : ℝ) (k : ℝ) (hx : 0 < x) (hy : 0 < y)
(h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) := sorry

end max_k_value_l218_218848


namespace plane_equation_through_point_and_parallel_l218_218425

theorem plane_equation_through_point_and_parallel (P : ℝ × ℝ × ℝ) (D : ℝ)
  (normal_vector : ℝ × ℝ × ℝ) (A B C : ℝ)
  (h1 : normal_vector = (2, -1, 3))
  (h2 : P = (2, 3, -1))
  (h3 : A = 2) (h4 : B = -1) (h5 : C = 3)
  (hD : A * 2 + B * 3 + C * -1 + D = 0) :
  A * x + B * y + C * z + D = 0 :=
by
  sorry

end plane_equation_through_point_and_parallel_l218_218425


namespace problem1_problem2_l218_218302

open Real

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 3| - |2 * x - a|

-- Problem (1)
theorem problem1 {a : ℝ} (h : ∃ x, f x a ≤ -5) : a ≤ -8 ∨ a ≥ 2 :=
sorry

-- Problem (2)
theorem problem2 {a : ℝ} (h : ∀ x, f (x - 1/2) a + f (-x - 1/2) a = 0) : a = 1 :=
sorry

end problem1_problem2_l218_218302


namespace scientific_notation_of_great_wall_l218_218904

theorem scientific_notation_of_great_wall : 
  ∀ n : ℕ, (6700010 : ℝ) = 6.7 * 10^6 :=
by
  sorry

end scientific_notation_of_great_wall_l218_218904


namespace pair_with_gcf_20_l218_218523

theorem pair_with_gcf_20 (a b : ℕ) (h1 : a = 20) (h2 : b = 40) : Nat.gcd a b = 20 := by
  rw [h1, h2]
  sorry

end pair_with_gcf_20_l218_218523


namespace intersection_points_l218_218250

def curve (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (x y : ℝ) : Prop := y = x + 1

theorem intersection_points :
  {p : ℝ × ℝ | curve p.1 p.2 ∧ line p.1 p.2} = {(-1, 0), (0, 1)} :=
by 
  sorry

end intersection_points_l218_218250


namespace train_speed_l218_218935

theorem train_speed
  (distance: ℝ) (time_in_minutes : ℝ) (time_in_hours : ℝ) (speed: ℝ)
  (h1 : distance = 20)
  (h2 : time_in_minutes = 10)
  (h3 : time_in_hours = time_in_minutes / 60)
  (h4 : speed = distance / time_in_hours)
  : speed = 120 := 
by
  sorry

end train_speed_l218_218935


namespace vanya_correct_answers_l218_218115

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218115


namespace prime_divides_sequence_term_l218_218456

theorem prime_divides_sequence_term (k : ℕ) (h_prime : Nat.Prime k) (h_ne_two : k ≠ 2) (h_ne_five : k ≠ 5) :
  ∃ n ≤ k, k ∣ (Nat.ofDigits 10 (List.replicate n 1)) :=
by
  sorry

end prime_divides_sequence_term_l218_218456


namespace evening_temperature_l218_218658

-- Define the given conditions
def t_noon : ℤ := 1
def d : ℤ := 3

-- The main theorem stating that the evening temperature is -2℃
theorem evening_temperature : t_noon - d = -2 := by
  sorry

end evening_temperature_l218_218658


namespace roots_of_equation_l218_218613

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l218_218613


namespace five_digit_palindromes_count_l218_218412

theorem five_digit_palindromes_count : 
  let a_values := {a : ℕ | 1 ≤ a ∧ a ≤ 9}
  let b_values := {b : ℕ | 0 ≤ b ∧ b ≤ 9}
  let c_values := {c : ℕ | 0 ≤ c ∧ c ≤ 9}
  a_values.card * b_values.card * c_values.card = 900 := 
by 
  -- a has 9 possible values
  have a_card : a_values.card = 9 := sorry
  -- b has 10 possible values
  have b_card : b_values.card = 10 := sorry
  -- c has 10 possible values
  have c_card : c_values.card = 10 := sorry
  -- solve the multiplication
  sorry

end five_digit_palindromes_count_l218_218412


namespace find_last_number_2_l218_218699

theorem find_last_number_2 (A B C D : ℤ) 
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) : 
  D = 2 := 
sorry

end find_last_number_2_l218_218699


namespace power_calculation_l218_218834

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218834


namespace prove_bounds_l218_218876

variable (a b : ℝ)

-- Conditions
def condition1 : Prop := 6 * a - b = 45
def condition2 : Prop := 4 * a + b > 60

-- Proof problem statement
theorem prove_bounds (h1 : condition1 a b) (h2 : condition2 a b) : a > 10.5 ∧ b > 18 :=
sorry

end prove_bounds_l218_218876


namespace quadratic_matches_sin_values_l218_218929

noncomputable def quadratic_function (x : ℝ) : ℝ := - (4 / (Real.pi ^ 2)) * (x ^ 2) + (4 / Real.pi) * x

theorem quadratic_matches_sin_values :
  (quadratic_function 0 = Real.sin 0) ∧
  (quadratic_function (Real.pi / 2) = Real.sin (Real.pi / 2)) ∧
  (quadratic_function Real.pi = Real.sin Real.pi) :=
by
  sorry

end quadratic_matches_sin_values_l218_218929


namespace vanya_correct_answers_l218_218104

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l218_218104


namespace seating_arrangements_l218_218582

/-
Given:
1. There are 8 students.
2. Four different classes: (1), (2), (3), and (4).
3. Each class has 2 students.
4. There are 2 cars, Car A and Car B, each with a capacity for 4 students.
5. The two students from Class (1) (twin sisters) must ride in the same car.

Prove:
The total number of ways to seat the students such that exactly 2 students from the same class are in Car A is 24.
-/

theorem seating_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 24 :=
sorry

end seating_arrangements_l218_218582


namespace students_wearing_other_colors_l218_218535

-- Definitions according to the problem conditions
def total_students : ℕ := 900
def percentage_blue : ℕ := 44
def percentage_red : ℕ := 28
def percentage_green : ℕ := 10

-- Goal: Prove the number of students who wear other colors
theorem students_wearing_other_colors :
  (total_students * (100 - (percentage_blue + percentage_red + percentage_green))) / 100 = 162 :=
by
  -- Skipping the proof steps with sorry
  sorry

end students_wearing_other_colors_l218_218535


namespace probability_of_one_common_l218_218077

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l218_218077


namespace inequality_must_hold_l218_218646

variable (a b c : ℝ)

theorem inequality_must_hold (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := 
sorry

end inequality_must_hold_l218_218646


namespace fraction_of_coins_l218_218480

theorem fraction_of_coins (total_states joining_states : ℕ) 
  (h₁ : total_states = 32) 
  (h₂ : joining_states = 7) :  
  (joining_states:ℚ) / total_states = 7 / 32 :=
by 
  -- We skip the proof using sorry
  sorry

end fraction_of_coins_l218_218480


namespace vasya_days_without_purchases_l218_218192

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218192


namespace price_of_one_table_l218_218164

variable (C T : ℝ)

def cond1 := 2 * C + T = 0.6 * (C + 2 * T)
def cond2 := C + T = 60
def solution := T = 52.5

theorem price_of_one_table (h1 : cond1 C T) (h2 : cond2 C T) : solution T :=
by
  sorry

end price_of_one_table_l218_218164


namespace percent_relation_l218_218059

variable (x y z : ℝ)

theorem percent_relation (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : x = 0.78 * z :=
by sorry

end percent_relation_l218_218059


namespace no_rational_roots_l218_218755

theorem no_rational_roots : ¬ ∃ x : ℚ, 5 * x^3 - 4 * x^2 - 8 * x + 3 = 0 :=
by
  sorry

end no_rational_roots_l218_218755


namespace max_sin_x_value_l218_218670

theorem max_sin_x_value (x y z : ℝ) (h1 : Real.sin x = Real.cos y) (h2 : Real.sin y = Real.cos z) (h3 : Real.sin z = Real.cos x) : Real.sin x ≤ Real.sqrt 2 / 2 :=
by
  sorry

end max_sin_x_value_l218_218670


namespace find_line_eq_l218_218042

-- Definitions for the conditions
def passes_through_M (l : ℝ × ℝ) : Prop :=
  l = (1, 2)

def segment_intercepted_length (l : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ,
    ∀ p : ℝ × ℝ, l p → ((4 * p.1 + 3 * p.2 + 1 = 0 ∨ 4 * p.1 + 3 * p.2 + 6 = 0) ∧ (A = p ∨ B = p)) ∧
    dist A B = Real.sqrt 2

-- Predicates for the lines to be proven
def line_eq1 (p : ℝ × ℝ) : Prop :=
  p.1 + 7 * p.2 = 15

def line_eq2 (p : ℝ × ℝ) : Prop :=
  7 * p.1 - p.2 = 5

-- The proof problem statement
theorem find_line_eq (l : ℝ × ℝ → Prop) :
  passes_through_M (1, 2) →
  segment_intercepted_length l →
  (∀ p, l p → line_eq1 p) ∨ (∀ p, l p → line_eq2 p) :=
by
  sorry

end find_line_eq_l218_218042


namespace fg_difference_l218_218438

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference : f (g 3) - g (f 3) = 59 :=
by
  sorry

end fg_difference_l218_218438


namespace power_of_3_l218_218818

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218818


namespace cost_of_birthday_gift_l218_218471

theorem cost_of_birthday_gift 
  (boss_contrib : ℕ)
  (todd_contrib : ℕ)
  (employee_contrib : ℕ)
  (num_employees : ℕ)
  (h1 : boss_contrib = 15)
  (h2 : todd_contrib = 2 * boss_contrib)
  (h3 : employee_contrib = 11)
  (h4 : num_employees = 5) :
  boss_contrib + todd_contrib + num_employees * employee_contrib = 100 := by
  sorry

end cost_of_birthday_gift_l218_218471


namespace vasya_days_without_purchase_l218_218204

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218204


namespace f_le_2x_f_not_le_1_9x_l218_218465

-- Define the function f and conditions
def f : ℝ → ℝ := sorry

axiom non_neg_f : ∀ x, 0 ≤ x → 0 ≤ f x
axiom f_at_1 : f 1 = 1
axiom f_additivity : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- Proof for part (1): f(x) ≤ 2x for all x in [0, 1]
theorem f_le_2x : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x := 
by
  sorry

-- Part (2): The inequality f(x) ≤ 1.9x does not hold for all x
theorem f_not_le_1_9x : ¬ (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 1.9 * x) := 
by
  sorry

end f_le_2x_f_not_le_1_9x_l218_218465


namespace range_of_f_on_interval_l218_218580

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem range_of_f_on_interval :
  Set.Icc (-1 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = y} :=
by
  sorry

end range_of_f_on_interval_l218_218580


namespace fish_in_pond_l218_218080

-- Conditions
variable (N : ℕ)
variable (h₁ : 80 * 80 = 2 * N)

-- Theorem to prove 
theorem fish_in_pond (h₁ : 80 * 80 = 2 * N) : N = 3200 := 
by 
  sorry

end fish_in_pond_l218_218080


namespace vanya_correct_answers_l218_218103

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l218_218103


namespace remainder_of_98_mul_102_mod_9_l218_218723

theorem remainder_of_98_mul_102_mod_9 : (98 * 102) % 9 = 6 := 
by 
  -- Introducing the variables and arithmetic
  let x := 98 * 102 
  have h1 : x = 9996 := 
    by norm_num
  have h2 : x % 9 = 6 := 
    by norm_num
  -- Result
  exact h2

end remainder_of_98_mul_102_mod_9_l218_218723


namespace solve_for_x_l218_218539

theorem solve_for_x : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by
  use -5
  sorry

end solve_for_x_l218_218539


namespace measure_of_smaller_angle_l218_218511

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l218_218511


namespace tan_neg_two_simplifies_l218_218637

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l218_218637


namespace systematic_sample_seat_number_l218_218021

theorem systematic_sample_seat_number (total_students sample_size interval : ℕ) (seat1 seat2 seat3 : ℕ) 
  (H_total_students : total_students = 56)
  (H_sample_size : sample_size = 4)
  (H_interval : interval = total_students / sample_size)
  (H_seat1 : seat1 = 3)
  (H_seat2 : seat2 = 31)
  (H_seat3 : seat3 = 45) :
  ∃ seat4 : ℕ, seat4 = 17 :=
by 
  sorry

end systematic_sample_seat_number_l218_218021


namespace product_of_equal_numbers_l218_218129

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l218_218129


namespace num_valid_pairs_l218_218671

/-- 
Let S(n) denote the sum of the digits of a natural number n.
Define the predicate to check if the pair (m, n) satisfies the given conditions.
-/
def S (n : ℕ) : ℕ := (toString n).foldl (fun acc ch => acc + ch.toNat - '0'.toNat) 0

def valid_pair (m n : ℕ) : Prop :=
  m < 100 ∧ n < 100 ∧ m > n ∧ m + S n = n + 2 * S m

/-- 
Theorem: There are exactly 99 pairs (m, n) that satisfy the given conditions.
-/
theorem num_valid_pairs : ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 99 ∧
  ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
sorry

end num_valid_pairs_l218_218671


namespace smallest_value_arithmetic_geometric_seq_l218_218316

theorem smallest_value_arithmetic_geometric_seq :
  ∃ (E F G H : ℕ), (E < F) ∧ (F < G) ∧ (F * 4 = G * 7) ∧ (E + G = 2 * F) ∧ (F * F * 49 = G * G * 16) ∧ (E + F + G + H = 97) := 
sorry

end smallest_value_arithmetic_geometric_seq_l218_218316


namespace find_base_l218_218851
-- Import the necessary library

-- Define the conditions and the result
theorem find_base (x y b : ℕ) (h1 : x - y = 9) (h2 : x = 9) (h3 : b^x * 4^y = 19683) : b = 3 :=
by
  sorry

end find_base_l218_218851


namespace power_of_3_l218_218820

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218820


namespace number_of_flower_sets_l218_218990

theorem number_of_flower_sets (total_flowers : ℕ) (flowers_per_set : ℕ) (sets : ℕ) 
  (h1 : total_flowers = 270) 
  (h2 : flowers_per_set = 90) 
  (h3 : sets = total_flowers / flowers_per_set) : 
  sets = 3 := 
by 
  sorry

end number_of_flower_sets_l218_218990


namespace royal_children_count_l218_218345

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l218_218345


namespace part_1_part_2_l218_218267

variables (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : a = b ∧ b = c ∧ A + B + C = 180 ∧ A = 90 ∨ B = 90 ∨ C = 90)
variable (sin_condition : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C)

theorem part_1 (h : a = b) : Real.cos C = 7 / 8 :=
by { sorry }

theorem part_2 (h₁ : B = 90) (h₂ : a = Real.sqrt 2) : b = 2 :=
by { sorry }

end part_1_part_2_l218_218267


namespace trajectory_equation_l218_218914

theorem trajectory_equation (x y a : ℝ) (h : x^2 + y^2 = a^2) :
  (x - y)^2 + 2*x*y = a^2 :=
by
  sorry

end trajectory_equation_l218_218914


namespace vertical_asymptote_c_values_l218_218764

theorem vertical_asymptote_c_values (c : ℝ) :
  (∃ x : ℝ, (x^2 - x - 6) = 0 ∧ (x^2 - 2*x + c) ≠ 0 ∧ ∀ y : ℝ, ((y ≠ x) → (x ≠ 3) ∧ (x ≠ -2)))
  → (c = -3 ∨ c = -8) :=
by sorry

end vertical_asymptote_c_values_l218_218764


namespace ryan_final_tokens_l218_218307

-- Conditions
def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 2 / 3
def candy_crush_fraction : ℚ := 1 / 2
def skiball_tokens : ℕ := 7
def friend_borrowed_tokens : ℕ := 5
def friend_returned_tokens : ℕ := 8
def laser_tag_tokens : ℕ := 3
def parents_purchase_factor : ℕ := 10

-- Final Answer
theorem ryan_final_tokens : initial_tokens - 24  - 6 - skiball_tokens + friend_returned_tokens + (parents_purchase_factor * skiball_tokens) - laser_tag_tokens = 75 :=
by sorry

end ryan_final_tokens_l218_218307


namespace kittens_per_bunny_l218_218678

-- Conditions
def total_initial_bunnies : ℕ := 30
def fraction_given_to_friend : ℚ := 2 / 5
def total_bunnies_after_birth : ℕ := 54

-- Determine the number of kittens each bunny gave birth to
theorem kittens_per_bunny (initial_bunnies given_fraction total_bunnies_after : ℕ) 
  (h1 : initial_bunnies = total_initial_bunnies)
  (h2 : given_fraction = fraction_given_to_friend)
  (h3 : total_bunnies_after = total_bunnies_after_birth) :
  (total_bunnies_after - (total_initial_bunnies - (total_initial_bunnies * fraction_given_to_friend))) / 
    (total_initial_bunnies * (1 - fraction_given_to_friend)) = 2 :=
by
  sorry

end kittens_per_bunny_l218_218678


namespace length_of_fence_l218_218503

theorem length_of_fence (side_length : ℕ) (h : side_length = 28) : 4 * side_length = 112 :=
by
  sorry

end length_of_fence_l218_218503


namespace probability_not_snow_l218_218491

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l218_218491


namespace laser_beam_total_distance_l218_218548

theorem laser_beam_total_distance :
  let A := (3, 5)
  let D := (7, 5)
  let D'' := (-7, -5)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  distance A D'' = 10 * Real.sqrt 2 :=
by
  -- definitions and conditions are captured
  sorry -- the proof goes here, no proof is required as per instructions

end laser_beam_total_distance_l218_218548


namespace range_of_a_when_min_f_ge_neg_a_l218_218770

noncomputable def f (a x : ℝ) := a * Real.log x + 2 * x

theorem range_of_a_when_min_f_ge_neg_a (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x > 0, f a x ≥ -a) :
  -2 ≤ a ∧ a < 0 :=
sorry

end range_of_a_when_min_f_ge_neg_a_l218_218770


namespace functions_unique_l218_218598

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem functions_unique (f g: ℝ → ℝ) :
  (∀ x : ℝ, x < 0 → (f (g x) = x / (x * f x - 2)) ∧ (g (f x) = x / (x * g x - 2))) →
  (∀ x : ℝ, 0 < x → (f x = 3 / x ∧ g x = 3 / x)) :=
by
  sorry

end functions_unique_l218_218598


namespace census_survey_is_suitable_l218_218525

def suitable_for_census (s: String) : Prop :=
  s = "Understand the vision condition of students in a class"

theorem census_survey_is_suitable :
  suitable_for_census "Understand the vision condition of students in a class" :=
by
  sorry

end census_survey_is_suitable_l218_218525


namespace vasya_days_without_purchase_l218_218188

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218188


namespace max_valid_n_eq_3210_l218_218760

-- Define the digit sum function S
def digit_sum (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- The condition S(3n) = 3S(n) and all digits of n are distinct
def valid_n (n : ℕ) : Prop :=
  digit_sum (3 * n) = 3 * digit_sum n ∧ (Nat.digits 10 n).Nodup

-- Prove that the maximum value of such n is 3210
theorem max_valid_n_eq_3210 : ∃ n : ℕ, valid_n n ∧ n = 3210 :=
by
  existsi 3210
  sorry

end max_valid_n_eq_3210_l218_218760


namespace triangle_inequality_11_side_l218_218711

def valid_triangle_count : ℕ :=
  let pairs : Finset (ℕ × ℕ) := (Finset.Icc 1 10).product (Finset.Icc 1 10)
  pairs.filter (λ (ab : ℕ × ℕ), ab.1 + ab.2 > 11 ∧ ab.1 ≤ ab.2).card

theorem triangle_inequality_11_side :
  valid_triangle_count = 36 :=
by
  sorry

end triangle_inequality_11_side_l218_218711


namespace cubic_coefficient_relationship_l218_218965

theorem cubic_coefficient_relationship (a b c p q r : ℝ)
    (h1 : ∀ s1 s2 s3: ℝ, s1 + s2 + s3 = -a ∧ s1 * s2 + s2 * s3 + s3 * s1 = b ∧ s1 * s2 * s3 = -c)
    (h2 : ∀ s1 s2 s3: ℝ, s1^2 + s2^2 + s3^2 = -p ∧ s1^2 * s2^2 + s2^2 * s3^2 + s3^2 * s1^2 = q ∧ s1^2 * s2^2 * s3^2 = r) :
    p = a^2 - 2 * b ∧ q = b^2 + 2 * a * c ∧ r = c^2 :=
by
  sorry

end cubic_coefficient_relationship_l218_218965


namespace C_investment_value_is_correct_l218_218400

noncomputable def C_investment_contribution 
  (A_investment B_investment total_profit A_profit_share : ℝ) : ℝ :=
  let C_investment := 
    (A_profit_share * (A_investment + B_investment) - A_investment * total_profit) / 
    (total_profit - A_profit_share)
  C_investment

theorem C_investment_value_is_correct : 
  C_investment_contribution 6300 4200 13600 4080 = 10500 := 
by
  unfold C_investment_contribution
  norm_num
  sorry

end C_investment_value_is_correct_l218_218400


namespace intersection_M_N_l218_218094

-- Definitions of sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The statement to prove
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := 
by 
  sorry

end intersection_M_N_l218_218094


namespace power_of_three_l218_218786

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218786


namespace intersection_of_M_and_N_is_0_and_2_l218_218971

open Set

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N_is_0_and_2 : M ∩ N = {0, 2} :=
by
  sorry

end intersection_of_M_and_N_is_0_and_2_l218_218971


namespace quiz_answer_key_count_l218_218292

theorem quiz_answer_key_count :
  let true_false_possibilities := 6  -- Combinations for 3 T/F questions where not all are same
  let multiple_choice_possibilities := 4^3  -- 4 choices for each of 3 multiple-choice questions
  true_false_possibilities * multiple_choice_possibilities = 384 := by
  sorry

end quiz_answer_key_count_l218_218292


namespace solve_A_solve_area_l218_218768

noncomputable def angle_A (A : ℝ) : Prop :=
  2 * (Real.cos (A / 2))^2 + Real.cos A = 0

noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 → b + c = 4 → A = 2 * Real.pi / 3 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3

theorem solve_A (A : ℝ) : angle_A A → A = 2 * Real.pi / 3 :=
sorry

theorem solve_area (a b c A S : ℝ) : 
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  A = 2 * Real.pi / 3 →
  area_triangle a b c A →
  S = Real.sqrt 3 :=
sorry

end solve_A_solve_area_l218_218768


namespace distribute_tasks_l218_218726

theorem distribute_tasks (tasks boys : ℕ) (H_tasks : tasks = 6) (H_boys : boys = 3) : 
  (∑ s in (Finset.powerset (Finset.univ : Finset (Fin tasks))), ite (0 < s.card ∧ s.card < boys) (λ _, 0)) 
  = 540 :=
by
  simp only [H_tasks, H_boys]
  sorry

end distribute_tasks_l218_218726


namespace trigonometric_expression_evaluation_l218_218641

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l218_218641


namespace problem_solution_l218_218864

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x ^ Real.sqrt 3 + y ^ Real.sqrt 3 = 189 :=
sorry

end problem_solution_l218_218864


namespace train_length_l218_218396

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmh = 60 → time_s = 15 → (60 * 1000 / 3600) * 15 = length_m → length_m = 250 :=
by { intros, sorry }

end train_length_l218_218396


namespace inequality_for_a_l218_218667

noncomputable def f (x : ℝ) : ℝ :=
  2^x + (Real.log x) / (Real.log 2)

theorem inequality_for_a (n : ℕ) (a : ℝ) (h₁ : 2 < n) (h₂ : 0 < a) (h₃ : 2^a + Real.log a / Real.log 2 = n^2) :
  2 * Real.log n / Real.log 2 > a ∧ a > 2 * Real.log n / Real.log 2 - 1 / n :=
by
  sorry

end inequality_for_a_l218_218667


namespace royal_children_l218_218337

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l218_218337


namespace complementary_angle_ratio_l218_218506

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l218_218506


namespace cost_of_horse_l218_218532

theorem cost_of_horse (H C : ℝ) 
  (h1 : 4 * H + 9 * C = 13400)
  (h2 : 0.4 * H + 1.8 * C = 1880) :
  H = 2000 :=
by
  sorry

end cost_of_horse_l218_218532


namespace school_cases_of_water_l218_218732

theorem school_cases_of_water (bottles_per_case bottles_used_first_game bottles_left_after_second_game bottles_used_second_game : ℕ)
  (h1 : bottles_per_case = 20)
  (h2 : bottles_used_first_game = 70)
  (h3 : bottles_left_after_second_game = 20)
  (h4 : bottles_used_second_game = 110) :
  let total_bottles_used := bottles_used_first_game + bottles_used_second_game
  let total_bottles_initial := total_bottles_used + bottles_left_after_second_game
  let number_of_cases := total_bottles_initial / bottles_per_case
  number_of_cases = 10 :=
by
  -- The proof goes here
  sorry

end school_cases_of_water_l218_218732


namespace total_expenditure_is_3500_l218_218549

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thurs : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300
def cost_earphone : ℕ := 620
def cost_pen : ℕ := 30
def cost_notebook : ℕ := 50

def expenditure_fri : ℕ := cost_earphone + cost_pen + cost_notebook
def total_expenditure : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thurs + expenditure_fri + expenditure_sat + expenditure_sun

theorem total_expenditure_is_3500 : total_expenditure = 3500 := by
  sorry

end total_expenditure_is_3500_l218_218549


namespace total_participants_l218_218125

-- Define the number of indoor and outdoor participants
variables (x y : ℕ)

-- First condition: number of outdoor participants is 480 more than indoor participants
def condition1 : Prop := y = x + 480

-- Second condition: moving 50 participants results in outdoor participants being 5 times the indoor participants
def condition2 : Prop := y + 50 = 5 * (x - 50)

-- Theorem statement: the total number of participants is 870
theorem total_participants (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 870 :=
sorry

end total_participants_l218_218125


namespace exponent_power_identity_l218_218792

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218792


namespace quadratic_real_solutions_l218_218656

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end quadratic_real_solutions_l218_218656


namespace inequality_am_gm_l218_218538

variable {u v : ℝ}

theorem inequality_am_gm (hu : 0 < u) (hv : 0 < v) : u ^ 3 + v ^ 3 ≥ u ^ 2 * v + v ^ 2 * u := by
  sorry

end inequality_am_gm_l218_218538


namespace subset_A_B_l218_218676

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem subset_A_B : A ⊆ B := sorry

end subset_A_B_l218_218676


namespace vasya_no_purchase_days_l218_218170

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218170


namespace inequality_proof_l218_218266

theorem inequality_proof (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 2) : 
  (1 / a + 1 / b) ≥ 2 :=
sorry

end inequality_proof_l218_218266


namespace fewer_free_throws_l218_218700

noncomputable def Deshawn_free_throws : ℕ := 12
noncomputable def Kayla_free_throws : ℕ := Deshawn_free_throws + (Deshawn_free_throws / 2)
noncomputable def Annieka_free_throws : ℕ := 14

theorem fewer_free_throws :
  Annieka_free_throws = Kayla_free_throws - 4 :=
by
  sorry

end fewer_free_throws_l218_218700


namespace power_equality_l218_218844

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218844


namespace find_num_cows_l218_218083

variable (num_cows num_pigs : ℕ)

theorem find_num_cows (h1 : 4 * num_cows + 24 + 4 * num_pigs = 20 + 2 * (num_cows + 6 + num_pigs)) 
                      (h2 : 6 = 6) 
                      (h3 : ∀x, 2 * x = x + x) 
                      (h4 : ∀x, 4 * x = 2 * 2 * x) 
                      (h5 : ∀x, 4 * x = 4 * x) : 
                      num_cows = 6 := 
by {
  sorry
}

end find_num_cows_l218_218083


namespace total_people_surveyed_l218_218085

theorem total_people_surveyed (x y : ℝ) (h1 : 0.536 * x = 30) (h2 : 0.794 * y = x) : y = 71 :=
by
  sorry

end total_people_surveyed_l218_218085


namespace ab_leq_one_l218_218701

theorem ab_leq_one (a b : ℝ) (h : (a + b) * (a + b + a + b) = 9) : a * b ≤ 1 := 
  sorry

end ab_leq_one_l218_218701


namespace vasya_days_without_purchase_l218_218187

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218187


namespace power_equality_l218_218846

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218846


namespace tan_neg_two_simplifies_l218_218638

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l218_218638


namespace base_k_number_eq_binary_l218_218281

theorem base_k_number_eq_binary (k : ℕ) (h : k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_number_eq_binary_l218_218281


namespace expand_expression_l218_218030

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l218_218030


namespace income_is_20000_l218_218489

-- Definitions from conditions
def income (x : ℕ) : ℕ := 4 * x
def expenditure (x : ℕ) : ℕ := 3 * x
def savings : ℕ := 5000

-- Theorem to prove the income
theorem income_is_20000 (x : ℕ) (h : income x - expenditure x = savings) : income x = 20000 :=
by
  sorry

end income_is_20000_l218_218489


namespace darry_total_steps_l218_218576

def largest_ladder_steps : ℕ := 20
def largest_ladder_times : ℕ := 12

def medium_ladder_steps : ℕ := 15
def medium_ladder_times : ℕ := 8

def smaller_ladder_steps : ℕ := 10
def smaller_ladder_times : ℕ := 10

def smallest_ladder_steps : ℕ := 5
def smallest_ladder_times : ℕ := 15

theorem darry_total_steps :
  (largest_ladder_steps * largest_ladder_times)
  + (medium_ladder_steps * medium_ladder_times)
  + (smaller_ladder_steps * smaller_ladder_times)
  + (smallest_ladder_steps * smallest_ladder_times)
  = 535 := by
  sorry

end darry_total_steps_l218_218576


namespace vasya_days_without_purchase_l218_218185

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218185


namespace power_of_3_l218_218822

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218822


namespace binom_n_n_minus_2_l218_218720

noncomputable def factorial : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def binom (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem binom_n_n_minus_2 (n : ℕ) (h : 0 < n) : 
  binom n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binom_n_n_minus_2_l218_218720


namespace find_a_2018_l218_218767

noncomputable def a : ℕ → ℕ
| n => if n > 0 then 2 * n else sorry

theorem find_a_2018 (a : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 ∧ n > 0 → a m + a n = a (m + n)) 
  (h1 : a 1 = 2) : a 2018 = 4036 := by
  sorry

end find_a_2018_l218_218767


namespace abs_z1_purely_imaginary_l218_218264

noncomputable def z1 (a : ℝ) : Complex := ⟨a, 2⟩
def z2 : Complex := ⟨2, -1⟩

theorem abs_z1_purely_imaginary (a : ℝ) (ha : 2 * a - 2 = 0) : Complex.abs (z1 a) = Real.sqrt 5 :=
by
  sorry

end abs_z1_purely_imaginary_l218_218264


namespace part1_l218_218627

def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - a^2 - 2*a < 0}
def setB (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x - 2*a ∧ x ≤ 2}

theorem part1 (a : ℝ) (h : a = 3) : setA 3 ∪ setB 3 = Set.Ioo (-6) 5 :=
by
  sorry

end part1_l218_218627


namespace problem_statement_l218_218917

noncomputable def nonnegative_reals : Type := {x : ℝ // 0 ≤ x}

theorem problem_statement (x : nonnegative_reals) :
  x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) ≥ 15*x.1 ∧
  (x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) = 15*x.1 ↔ (x.1 = 0 ∨ x.1 = 1)) :=
by
  sorry

end problem_statement_l218_218917


namespace joe_paint_usage_l218_218090

theorem joe_paint_usage :
  let total_paint := 360
  let paint_first_week := total_paint * (1 / 4)
  let remaining_paint_after_first_week := total_paint - paint_first_week
  let paint_second_week := remaining_paint_after_first_week * (1 / 7)
  paint_first_week + paint_second_week = 128.57 :=
by
  sorry

end joe_paint_usage_l218_218090


namespace max_value_of_function_l218_218892

theorem max_value_of_function : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y == (2*x^2 - 2*x + 3) / (x^2 - x + 1)) → y ≤ 10/3) ∧
  (∃ x : ℝ, (2*x^2 - 2*x + 3) / (x^2 - x + 1) = 10/3) := 
sorry

end max_value_of_function_l218_218892


namespace bookstore_price_change_l218_218926

theorem bookstore_price_change (P : ℝ) (x : ℝ) (h : P > 0) : 
  (P * (1 + x / 100) * (1 - x / 100)) = 0.75 * P → x = 50 :=
by
  sorry

end bookstore_price_change_l218_218926


namespace proof_problem_l218_218040

structure Plane := (name : String)
structure Line := (name : String)

def parallel_planes (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def parallel_lines (m n : Line) : Prop := sorry

theorem proof_problem (m : Line) (α β : Plane) :
  parallel_planes α β → in_plane m α → parallel_lines m (Line.mk β.name) :=
sorry

end proof_problem_l218_218040


namespace evaluate_expression_l218_218026

theorem evaluate_expression : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 :=
by sorry

end evaluate_expression_l218_218026


namespace simplify_expression_solve_inequality_system_l218_218379

-- Problem 1
theorem simplify_expression (m n : ℝ) (h1 : 3 * m - 2 * n ≠ 0) (h2 : 3 * m + 2 * n ≠ 0) (h3 : 9 * m ^ 2 - 4 * n ^ 2 ≠ 0) :
  ((1 / (3 * m - 2 * n) - 1 / (3 * m + 2 * n)) / (m * n / (9 * m ^ 2 - 4 * n ^ 2))) = (4 / m) :=
sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) (h1 : 3 * x + 10 > 5 * x - 2 * (5 - x)) (h2 : (x + 3) / 5 > 1 - x) :
  1 / 3 < x ∧ x < 5 :=
sorry

end simplify_expression_solve_inequality_system_l218_218379


namespace distinct_solution_condition_l218_218609

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l218_218609


namespace number_of_children_l218_218368

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l218_218368


namespace complementary_angle_ratio_l218_218505

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l218_218505


namespace angle_E_degree_l218_218883

-- Given conditions
variables {E F G H : ℝ} -- degrees of the angles in quadrilateral EFGH

-- Condition 1: The angles satisfy a specific ratio
axiom angle_ratio : E = 3 * F ∧ E = 2 * G ∧ E = 6 * H

-- Condition 2: The sum of the angles in the quadrilateral is 360 degrees
axiom angle_sum : E + (E / 3) + (E / 2) + (E / 6) = 360

-- Prove the degree measure of angle E is 180 degrees
theorem angle_E_degree : E = 180 :=
by
  sorry

end angle_E_degree_l218_218883


namespace last_score_entered_is_75_l218_218680

theorem last_score_entered_is_75 (scores : List ℕ) (h : scores = [62, 75, 83, 90]) :
  ∃ last_score, last_score ∈ scores ∧ 
    (∀ (num list : List ℕ), list ≠ [] → list.length ≤ scores.length → 
    ¬ list.sum % list.length ≠ 0) → 
  last_score = 75 :=
by
  sorry

end last_score_entered_is_75_l218_218680


namespace octagon_perimeter_l218_218253

/-- 
  Represents the side length of the regular octagon
-/
def side_length : ℕ := 12

/-- 
  Represents the number of sides of a regular octagon
-/
def number_of_sides : ℕ := 8

/-- 
  Defines the perimeter of the regular octagon
-/
def perimeter (side_length : ℕ) (number_of_sides : ℕ) : ℕ :=
  side_length * number_of_sides

/-- 
  Proof statement: asserting that the perimeter of a regular octagon
  with a side length of 12 meters is 96 meters
-/
theorem octagon_perimeter :
  perimeter side_length number_of_sides = 96 :=
  sorry

end octagon_perimeter_l218_218253


namespace proof_x_plus_y_sum_l218_218039

noncomputable def x_and_y_sum (x y : ℝ) : Prop := 31.25 / x = 100 / 9.6 ∧ 13.75 / x = y / 9.6

theorem proof_x_plus_y_sum (x y : ℝ) (h : x_and_y_sum x y) : x + y = 47 :=
sorry

end proof_x_plus_y_sum_l218_218039


namespace power_of_3_l218_218824

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218824


namespace complex_number_quadrant_l218_218049

theorem complex_number_quadrant :
  let z := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z < 0 ∧ Complex.im z > 0 :=
by
  sorry

end complex_number_quadrant_l218_218049


namespace proof_C_D_values_l218_218139

-- Given the conditions
def denominator_factorization (x : ℝ) : Prop :=
  3 * x ^ 2 - x - 14 = (3 * x + 7) * (x - 2)

def fraction_equality (x : ℝ) (C D : ℝ) : Prop :=
  (3 * x ^ 2 + 7 * x - 20) / (3 * x ^ 2 - x - 14) =
  C / (x - 2) + D / (3 * x + 7)

-- The values to be proven
def values_C_D : Prop :=
  ∃ C D : ℝ, C = -14 / 13 ∧ D = 81 / 13 ∧ ∀ x : ℝ, (denominator_factorization x → fraction_equality x C D)

theorem proof_C_D_values : values_C_D :=
sorry

end proof_C_D_values_l218_218139


namespace grandma_Olga_grandchildren_l218_218974

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end grandma_Olga_grandchildren_l218_218974


namespace platform_length_l218_218541

theorem platform_length
  (L_train : ℕ) (T_platform : ℕ) (T_pole : ℕ) (P : ℕ)
  (h1 : L_train = 300)
  (h2 : T_platform = 39)
  (h3 : T_pole = 10)
  (h4 : L_train / T_pole * T_platform = L_train + P) :
  P = 870 := 
sorry

end platform_length_l218_218541


namespace royal_children_l218_218338

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l218_218338


namespace equal_number_of_coins_l218_218736

theorem equal_number_of_coins (x : ℕ) (hx : 1 * x + 5 * x + 10 * x + 25 * x + 100 * x = 305) : x = 2 :=
sorry

end equal_number_of_coins_l218_218736


namespace power_of_three_l218_218811

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218811


namespace proj_b_l218_218458

open Matrix Real

-- Definition of orthogonality
def orthogonal (a b : Vector ℝ 2) : Prop :=
  dot_product a b = 0

-- Projections
def proj (u v : Vector ℝ 2) : Vector ℝ 2 :=
  (dot_product u v / dot_product u u) • u

theorem proj_b (a b v : Vector ℝ 2) 
    (h_orthog : orthogonal a b)
    (h_proj_a : proj a v = ⟨[-4/5, -8/5]⟩) :
    proj b v = ⟨[24/5, -12/5]⟩ := 
by 
  sorry

end proj_b_l218_218458


namespace f_cos_eq_l218_218441

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given condition
axiom f_sin_eq : f (Real.sin x) = 3 - Real.cos (2 * x)

-- The statement we want to prove
theorem f_cos_eq : f (Real.cos x) = 3 + Real.cos (2 * x) := 
by
  sorry

end f_cos_eq_l218_218441


namespace vasya_did_not_buy_anything_days_l218_218214

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218214


namespace range_of_a_l218_218963

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (2 * x + 2) - abs (2 * x - 2) ≤ a) ↔ 4 ≤ a :=
sorry

end range_of_a_l218_218963


namespace trigonometric_expression_evaluation_l218_218640

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l218_218640


namespace alice_age_30_l218_218713

variable (A T : ℕ)

def tom_younger_alice (A T : ℕ) := T = A - 15
def ten_years_ago (A T : ℕ) := A - 10 = 4 * (T - 10)

theorem alice_age_30 (A T : ℕ) (h1 : tom_younger_alice A T) (h2 : ten_years_ago A T) : A = 30 := 
by sorry

end alice_age_30_l218_218713


namespace dave_guitar_strings_l218_218409

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end dave_guitar_strings_l218_218409


namespace intersection_eq_inter_l218_218972

noncomputable def M : Set ℝ := { x | x^2 < 4 }
noncomputable def N : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
noncomputable def inter : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_eq_inter : M ∩ N = inter :=
by sorry

end intersection_eq_inter_l218_218972


namespace power_addition_l218_218798

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218798


namespace power_of_three_l218_218784

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218784


namespace solve_system_I_solve_system_II_l218_218691

theorem solve_system_I (x y : ℝ) (h1 : y = x + 3) (h2 : x - 2 * y + 12 = 0) : x = 6 ∧ y = 9 :=
by
  sorry

theorem solve_system_II (x y : ℝ) (h1 : 4 * (x - y - 1) = 3 * (1 - y) - 2) (h2 : x / 2 + y / 3 = 2) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_system_I_solve_system_II_l218_218691


namespace gcd_polynomials_l218_218942

def P (n : ℤ) : ℤ := n^3 - 6 * n^2 + 11 * n - 6
def Q (n : ℤ) : ℤ := n^2 - 4 * n + 4

theorem gcd_polynomials (n : ℤ) (h : n ≥ 3) : Int.gcd (P n) (Q n) = n - 2 :=
by
  sorry

end gcd_polynomials_l218_218942


namespace gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l218_218433

def gervais_distance_miles_per_day : Real := 315
def gervais_days : Real := 3
def gervais_km_per_mile : Real := 1.60934

def henri_total_miles : Real := 1250
def madeleine_distance_miles_per_day : Real := 100
def madeleine_days : Real := 5

def gervais_total_km := gervais_distance_miles_per_day * gervais_days * gervais_km_per_mile
def henri_total_km := henri_total_miles * gervais_km_per_mile
def madeleine_total_km := madeleine_distance_miles_per_day * madeleine_days * gervais_km_per_mile

def combined_total_km := gervais_total_km + henri_total_km + madeleine_total_km

theorem gervais_km_correct : gervais_total_km = 1520.82405 := sorry
theorem henri_km_correct : henri_total_km = 2011.675 := sorry
theorem madeleine_km_correct : madeleine_total_km = 804.67 := sorry
theorem total_km_correct : combined_total_km = 4337.16905 := sorry
theorem henri_drove_farthest : henri_total_km = 2011.675 := sorry

end gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l218_218433


namespace find_y_l218_218442

theorem find_y (x k m y : ℤ) 
  (h1 : x = 82 * k + 5) 
  (h2 : x + y = 41 * m + 12) : 
  y = 7 := 
sorry

end find_y_l218_218442


namespace royal_family_children_l218_218349

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l218_218349


namespace maria_total_distance_l218_218581

-- Definitions
def total_distance (D : ℝ) : Prop :=
  let d1 := D/2   -- Distance traveled before first stop
  let r1 := D - d1 -- Distance remaining after first stop
  let d2 := r1/4  -- Distance traveled before second stop
  let r2 := r1 - d2 -- Distance remaining after second stop
  let d3 := r2/3  -- Distance traveled before third stop
  let r3 := r2 - d3 -- Distance remaining after third stop
  r3 = 270 -- Remaining distance after third stop equals 270 miles

-- Theorem statement
theorem maria_total_distance : ∃ D : ℝ, total_distance D ∧ D = 1080 :=
sorry

end maria_total_distance_l218_218581


namespace value_of_m_l218_218983

theorem value_of_m (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end value_of_m_l218_218983


namespace diagonal_cannot_be_good_l218_218291

def is_good (table : ℕ → ℕ → ℕ) (i j : ℕ) :=
  ∀ x y, (x = i ∨ y = j) → ∀ x' y', (x' = i ∨ y' = j) → (x ≠ x' ∨ y ≠ y') → table x y ≠ table x' y'

theorem diagonal_cannot_be_good :
  ∀ (table : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ table i j ∧ table i j ≤ 25) →
  ¬ ∀ k, (is_good table k k) :=
by
  sorry

end diagonal_cannot_be_good_l218_218291


namespace notebook_cost_3_dollars_l218_218235

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l218_218235


namespace power_calculation_l218_218833

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218833


namespace Austin_work_hours_on_Wednesdays_l218_218559

variable {W : ℕ}

theorem Austin_work_hours_on_Wednesdays
  (h1 : 5 * 2 + 5 * W + 5 * 3 = 25 + 5 * W)
  (h2 : 6 * (25 + 5 * W) = 180)
  : W = 1 := by
  sorry

end Austin_work_hours_on_Wednesdays_l218_218559


namespace polynomial_division_quotient_l218_218036

theorem polynomial_division_quotient :
  ∀ (x : ℝ), (x^5 - 21*x^3 + 8*x^2 - 17*x + 12) / (x - 3) = (x^4 + 3*x^3 - 12*x^2 - 28*x - 101) :=
by
  sorry

end polynomial_division_quotient_l218_218036


namespace four_P_plus_five_square_of_nat_l218_218681

theorem four_P_plus_five_square_of_nat 
  (a b : ℕ)
  (P : ℕ)
  (hP : P = (Nat.lcm a b) / (a + 1) + (Nat.lcm a b) / (b + 1))
  (h_prime : Nat.Prime P) : 
  ∃ n : ℕ, 4 * P + 5 = (2 * n + 1) ^ 2 :=
by
  sorry

end four_P_plus_five_square_of_nat_l218_218681


namespace probability_one_common_number_approx_l218_218060

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l218_218060


namespace arithmetic_geometric_sequence_l218_218271

theorem arithmetic_geometric_sequence :
  ∀ (a₁ a₂ b₂ : ℝ),
    -- Conditions for arithmetic sequence: -1, a₁, a₂, 8
    2 * a₁ = -1 + a₂ ∧
    2 * a₂ = a₁ + 8 →
    -- Conditions for geometric sequence: -1, b₁, b₂, b₃, -4
    (∃ (b₁ b₃ : ℝ), b₁^2 = b₂ ∧ b₁ != 0 ∧ -4 * b₁^4 = b₂ → -1 * b₁ = b₃) →
    -- Goal: Calculate and prove the value
    (a₁ * a₂ / b₂) = -5 :=
by {
  sorry
}

end arithmetic_geometric_sequence_l218_218271


namespace lottery_probability_exactly_one_common_l218_218070

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218070


namespace minimum_glue_drops_to_prevent_37_gram_subset_l218_218020

def stones : List ℕ := List.range' 1 36  -- List of stones with masses from 1 to 36 grams

def glue_drop_combination_invalid (stones : List ℕ) : Prop :=
  ¬ (∃ (subset : List ℕ), subset.sum = 37 ∧ (∀ s ∈ subset, s ∈ stones))

def min_glue_drops (stones : List ℕ) : ℕ := 
  9 -- as per the solution

theorem minimum_glue_drops_to_prevent_37_gram_subset :
  ∀ (s : List ℕ), s = stones → glue_drop_combination_invalid s → min_glue_drops s = 9 :=
by intros; sorry

end minimum_glue_drops_to_prevent_37_gram_subset_l218_218020


namespace complex_division_l218_218628

-- Define the complex numbers in Lean
def i : ℂ := Complex.I

-- Claim to be proved
theorem complex_division :
  (1 + i) / (3 - i) = (1 + 2 * i) / 5 :=
by
  sorry

end complex_division_l218_218628


namespace quadratic_real_solutions_l218_218655

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end quadratic_real_solutions_l218_218655


namespace royal_children_l218_218339

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l218_218339


namespace no_integers_abc_for_polynomial_divisible_by_9_l218_218872

theorem no_integers_abc_for_polynomial_divisible_by_9 :
  ¬ ∃ (a b c : ℤ), ∀ x : ℤ, 9 ∣ (x + a) * (x + b) * (x + c) - x ^ 3 - 1 :=
by
  sorry

end no_integers_abc_for_polynomial_divisible_by_9_l218_218872


namespace sqrt_a_plus_sqrt_b_eq_3_l218_218850

theorem sqrt_a_plus_sqrt_b_eq_3 (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) : Real.sqrt a + Real.sqrt b = 3 :=
sorry

end sqrt_a_plus_sqrt_b_eq_3_l218_218850


namespace tan_value_sin_cos_ratio_sin_squared_expression_l218_218619

theorem tan_value (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  Real.tan α = -1 / 3 :=
sorry

theorem sin_cos_ratio (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2 :=
sorry

theorem sin_squared_expression (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5 :=
sorry

end tan_value_sin_cos_ratio_sin_squared_expression_l218_218619


namespace range_of_a_for_monotonic_function_l218_218653

theorem range_of_a_for_monotonic_function (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 0 ≤ (1 / x) + a) → a ≥ -1 / 2 := 
by
  sorry

end range_of_a_for_monotonic_function_l218_218653


namespace gain_percent_l218_218529

theorem gain_percent (CP SP : ℝ) (hCP : CP = 110) (hSP : SP = 125) : 
  (SP - CP) / CP * 100 = 13.64 := by
  sorry

end gain_percent_l218_218529


namespace vasya_purchase_l218_218210

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218210


namespace find_negative_number_l218_218521

theorem find_negative_number : ∃ x ∈ ({} : set ℝ), x < 0 ∧ (x = -5) :=
by
  use -5
  split
  { 
    trivial 
  }
  {
    simp
  }

end find_negative_number_l218_218521


namespace MonotonicallyIncreasingInterval_l218_218769

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * sin x - cos x

noncomputable def g (x : ℝ) : ℝ := - Real.sqrt 2 * cos (3 * x)

lemma SymmetricCenter (a x : ℝ) (h : f (π / 4) a = 0) : a = 1 :=
by
  sorry

theorem MonotonicallyIncreasingInterval (k : ℤ) : 
    ∃ a : ℝ, f (π/4) a = 0 ∧ 
    ∀ x, (f x a = Real.sqrt 2 * sin (x - π / 4)) ∧ 
    g(x) = -Real.sqrt 2 * cos (3 * x) ∧ 
    (∃ x1 x2 : ℝ, (g x).increasing_on (set.Icc (2 * k * π / 3) (2 * k * π / 3 + π / 3))) :=
by
  sorry

end MonotonicallyIncreasingInterval_l218_218769


namespace find_abc_l218_218759

noncomputable def a_b_c_exist : Prop :=
  ∃ (a b c : ℝ), 
    (a + b + c = 21/4) ∧ 
    (1/a + 1/b + 1/c = 21/4) ∧ 
    (a * b * c = 1) ∧ 
    (a < b) ∧ (b < c) ∧ 
    (a = 1/4) ∧ (b = 1) ∧ (c = 4)

theorem find_abc : a_b_c_exist :=
sorry

end find_abc_l218_218759


namespace sum_of_cube_angles_l218_218141

theorem sum_of_cube_angles (W X Y Z : Point) (cube : Cube)
  (angle_WXY angle_XYZ angle_YZW angle_ZWX : ℝ)
  (h₁ : angle_WXY = 90)
  (h₂ : angle_XYZ = 90)
  (h₃ : angle_YZW = 90)
  (h₄ : angle_ZWX = 60) :
  angle_WXY + angle_XYZ + angle_YZW + angle_ZWX = 330 := by
  sorry

end sum_of_cube_angles_l218_218141


namespace sum_of_repeating_decimals_l218_218031

-- Definitions of the repeating decimals as fractions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 99
def z : ℚ := 3 / 999

-- Theorem stating the sum of these fractions is equal to the expected result
theorem sum_of_repeating_decimals : x + y + z = 164 / 1221 := 
  sorry

end sum_of_repeating_decimals_l218_218031


namespace defective_pens_count_l218_218854

theorem defective_pens_count (total_pens : ℕ) (prob_not_defective : ℚ) (D : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : prob_not_defective = 0.5357142857142857) : 
  D = 2 := 
by
  sorry

end defective_pens_count_l218_218854


namespace vanya_correct_answers_l218_218102

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l218_218102


namespace sum_of_consecutive_pages_l218_218543

theorem sum_of_consecutive_pages (n : ℕ) 
  (h : n * (n + 1) = 20412) : n + (n + 1) + (n + 2) = 429 := by
  sorry

end sum_of_consecutive_pages_l218_218543


namespace product_of_equal_numbers_l218_218128

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l218_218128


namespace length_of_ad_l218_218989

theorem length_of_ad (AB CD AD BC : ℝ) 
  (h1 : AB = 10) 
  (h2 : CD = 2 * AB) 
  (h3 : AD = BC) 
  (h4 : AB + BC + CD + AD = 42) : AD = 6 :=
by
  -- proof omitted
  sorry

end length_of_ad_l218_218989


namespace Vanya_correct_answers_l218_218110

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l218_218110


namespace power_equality_l218_218838

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218838


namespace floor_sqrt_fifty_l218_218585

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l218_218585


namespace f_four_times_even_l218_218673

variable (f : ℝ → ℝ) (x : ℝ)

-- Definition stating f is an odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem f_four_times_even (h : is_odd f) : is_even (f (f (f (f x)))) :=
by sorry

end f_four_times_even_l218_218673


namespace lottery_probability_exactly_one_common_l218_218074

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218074


namespace find_hypotenuse_of_right_angle_triangle_l218_218953

theorem find_hypotenuse_of_right_angle_triangle
  (PR : ℝ) (angle_QPR : ℝ)
  (h1 : PR = 16)
  (h2 : angle_QPR = Real.pi / 4) :
  ∃ PQ : ℝ, PQ = 16 * Real.sqrt 2 :=
by
  sorry

end find_hypotenuse_of_right_angle_triangle_l218_218953


namespace math_problem_l218_218001

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, a = b * b

theorem math_problem (a m : ℕ) (h1: m = 2992) (h2: a = m^2 + m^2 * (m+1)^2 + (m+1)^2) : is_perfect_square a :=
  sorry

end math_problem_l218_218001


namespace train_length_is_correct_l218_218394

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l218_218394


namespace total_earnings_l218_218574

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end total_earnings_l218_218574


namespace inscribed_sphere_radius_base_height_l218_218005

noncomputable def radius_of_inscribed_sphere (r base_radius height : ℝ) := 
  r = (30 / (Real.sqrt 5 + 1)) * (Real.sqrt 5 - 1) 

theorem inscribed_sphere_radius_base_height (r : ℝ) (b d : ℝ) (base_radius height : ℝ) 
  (h_base: base_radius = 15) (h_height: height = 30) 
  (h_radius: radius_of_inscribed_sphere r base_radius height) 
  (h_expr: r = b * (Real.sqrt d) - b) : 
  b + d = 12.5 :=
sorry

end inscribed_sphere_radius_base_height_l218_218005


namespace students_like_neither_l218_218086

theorem students_like_neither (N_Total N_Chinese N_Math N_Both N_Neither : ℕ)
  (h_total: N_Total = 62)
  (h_chinese: N_Chinese = 37)
  (h_math: N_Math = 49)
  (h_both: N_Both = 30)
  (h_neither: N_Neither = N_Total - (N_Chinese - N_Both) - (N_Math - N_Both) - N_Both) : 
  N_Neither = 6 :=
by 
  rw [h_total, h_chinese, h_math, h_both] at h_neither
  exact h_neither.trans (by norm_num)


end students_like_neither_l218_218086


namespace similarity_transformation_l218_218995

theorem similarity_transformation (C C' : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : C = (4, 1))
  (h3 : C' = (r * 4, r * 1)) : (C' = (12, 3) ∨ C' = (-12, -3)) := by
  sorry

end similarity_transformation_l218_218995


namespace equal_sums_arithmetic_sequences_l218_218863

-- Define the arithmetic sequences and their sums
def s₁ (n : ℕ) : ℕ := n * (5 * n + 13) / 2
def s₂ (n : ℕ) : ℕ := n * (3 * n + 37) / 2

-- State the theorem: for given n != 0, prove s₁ n = s₂ n implies n = 12
theorem equal_sums_arithmetic_sequences (n : ℕ) (h : n ≠ 0) : 
  s₁ n = s₂ n → n = 12 :=
by
  sorry

end equal_sums_arithmetic_sequences_l218_218863


namespace consecutive_even_integers_sum_l218_218894

theorem consecutive_even_integers_sum (n : ℕ) (h : n % 2 = 0) (h_pro : n * (n + 2) * (n + 4) = 3360) :
  n + (n + 2) + (n + 4) = 48 :=
by sorry

end consecutive_even_integers_sum_l218_218894


namespace vasya_days_without_purchase_l218_218199

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218199


namespace hat_cost_l218_218987

noncomputable def cost_of_hat (H : ℕ) : Prop :=
  let cost_shirts := 3 * 5
  let cost_jeans := 2 * 10
  let cost_hats := 4 * H
  let total_cost := 51
  cost_shirts + cost_jeans + cost_hats = total_cost

theorem hat_cost : ∃ H : ℕ, cost_of_hat H ∧ H = 4 :=
by 
  sorry

end hat_cost_l218_218987


namespace distinct_solution_condition_l218_218607

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l218_218607


namespace shirley_cases_l218_218123

-- Given conditions
def T : ℕ := 54  -- boxes of Trefoils sold
def S : ℕ := 36  -- boxes of Samoas sold
def M : ℕ := 48  -- boxes of Thin Mints sold
def t_per_case : ℕ := 4  -- boxes of Trefoils per case
def s_per_case : ℕ := 3  -- boxes of Samoas per case
def m_per_case : ℕ := 5  -- boxes of Thin Mints per case

-- Amount of boxes delivered per case should meet the required demand
theorem shirley_cases : ∃ (n_cases : ℕ), 
  n_cases * t_per_case ≥ T ∧ 
  n_cases * s_per_case ≥ S ∧ 
  n_cases * m_per_case ≥ M :=
by
  use 14
  sorry

end shirley_cases_l218_218123


namespace pages_removed_iff_original_pages_l218_218315

def booklet_sum (n r : ℕ) : ℕ :=
  (n * (2 * n + 1)) - (4 * r - 1)

theorem pages_removed_iff_original_pages (n r : ℕ) :
  booklet_sum n r = 963 ↔ (2 * n = 44 ∧ (2 * r - 1, 2 * r) = (13, 14)) :=
sorry

end pages_removed_iff_original_pages_l218_218315


namespace cuboid_length_l218_218251

theorem cuboid_length (SA w h : ℕ) (h_SA : SA = 700) (h_w : w = 14) (h_h : h = 7) 
  (h_surface_area : SA = 2 * l * w + 2 * l * h + 2 * w * h) : l = 12 :=
by
  intros
  sorry

end cuboid_length_l218_218251


namespace michael_meets_truck_once_l218_218304

def michael_speed := 5  -- feet per second
def pail_distance := 150  -- feet
def truck_speed := 15  -- feet per second
def truck_stop_time := 20  -- seconds

def initial_michael_position (t : ℕ) : ℕ := t * michael_speed
def initial_truck_position (t : ℕ) : ℕ := pail_distance + t * truck_speed - (t / (truck_speed * truck_stop_time))

def distance (t : ℕ) : ℕ := initial_truck_position t - initial_michael_position t

theorem michael_meets_truck_once :
  ∃ t, (distance t = 0) :=  
sorry

end michael_meets_truck_once_l218_218304


namespace ages_of_children_l218_218734

theorem ages_of_children : ∃ (a1 a2 a3 a4 : ℕ),
  a1 + a2 + a3 + a4 = 33 ∧
  (a1 - 3) + (a2 - 3) + (a3 - 3) + (a4 - 3) = 22 ∧
  (a1 - 7) + (a2 - 7) + (a3 - 7) + (a4 - 7) = 11 ∧
  (a1 - 13) + (a2 - 13) + (a3 - 13) + (a4 - 13) = 1 ∧
  a1 = 14 ∧ a2 = 11 ∧ a3 = 6 ∧ a4 = 2 :=
by
  sorry

end ages_of_children_l218_218734


namespace minimum_value_C2_minus_D2_l218_218866

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 11))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 4)) + (Real.sqrt (z + 9))

theorem minimum_value_C2_minus_D2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (C x y z)^2 - (D x y z)^2 ≥ 36 := by
  sorry

end minimum_value_C2_minus_D2_l218_218866


namespace complementary_angle_ratio_l218_218504

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l218_218504


namespace angles_arith_prog_triangle_l218_218882

noncomputable def a : ℕ := 8
noncomputable def b : ℕ := 37
noncomputable def c : ℕ := 0

theorem angles_arith_prog_triangle (y : ℝ) (h1 : y = 8 ∨ y * y = 37) :
  a + b + c = 45 := by
  -- skipping the detailed proof steps
  sorry

end angles_arith_prog_triangle_l218_218882


namespace simplify_and_evaluate_expression_l218_218875

variable (x y : ℝ)

theorem simplify_and_evaluate_expression
  (hx : x = 2)
  (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 :=
by
  sorry

end simplify_and_evaluate_expression_l218_218875


namespace negation_example_l218_218313

theorem negation_example :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := sorry

end negation_example_l218_218313


namespace total_distance_is_10_miles_l218_218404

noncomputable def total_distance_back_to_town : ℕ :=
  let distance1 := 3
  let distance2 := 3
  let distance3 := 4
  distance1 + distance2 + distance3

theorem total_distance_is_10_miles :
  total_distance_back_to_town = 10 :=
by
  sorry

end total_distance_is_10_miles_l218_218404


namespace box_height_l218_218922

theorem box_height (h : ℝ) :
  ∃ (h : ℝ), 
  let large_sphere_radius := 3
  let small_sphere_radius := 1.5
  let box_width := 6
  h = 12 := 
sorry

end box_height_l218_218922


namespace triangle_area_l218_218323

theorem triangle_area : 
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  0.5 * base * height = 24.0 :=
by
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  sorry

end triangle_area_l218_218323


namespace floor_sqrt_50_l218_218590

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l218_218590


namespace part2_proof_l218_218273

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log x) - Real.exp 1 * x

theorem part2_proof (x : ℝ) (h : 0 < x) :
  x * f x - Real.exp x + 2 * Real.exp 1 * x ≤ 0 := 
sorry

end part2_proof_l218_218273


namespace ratio_65_13_l218_218517

theorem ratio_65_13 : 65 / 13 = 5 := 
by
  sorry

end ratio_65_13_l218_218517


namespace probability_exactly_half_red_balls_l218_218752

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_half_red_balls :
  binomial_probability 8 4 (1/2) = 35/128 :=
by
  sorry

end probability_exactly_half_red_balls_l218_218752


namespace find_digits_l218_218578

-- Define real repeating decimal as a rational number
def repeating_decimal_ab (a b : ℕ) : ℚ := (10 * a + b) / 99
def repeating_decimal_abc (a b c : ℕ) : ℚ := (100 * a + 10 * b + c) / 999

-- Define the condition that their sum is 12/13 and that a, b, and c are distinct
noncomputable def condition (a b c : ℕ) : Prop :=
  (repeating_decimal_ab a b + repeating_decimal_abc a b c = 12 / 13) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Define the theorem that proves the digits
theorem find_digits : ∃ a b c : ℕ, condition a b c ∧ a = 4 ∧ b = 6 ∧ c = 3 :=
by
  existsi (4 : ℕ)
  existsi (6 : ℕ)
  existsi (3 : ℕ)
  have h1 : 4 ≠ 6 ∧ 6 ≠ 3 ∧ 4 ≠ 3 := by simp
  have h2 : repeating_decimal_ab 4 6 + repeating_decimal_abc 4 6 3 = 12 / 13 := by sorry
  exact ⟨⟨h2, h1⟩, rfl, rfl, rfl⟩

end find_digits_l218_218578


namespace power_of_3_l218_218823

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218823


namespace combined_weight_proof_l218_218912

-- Definitions of atomic weights
def weight_C : ℝ := 12.01
def weight_H : ℝ := 1.01
def weight_O : ℝ := 16.00
def weight_S : ℝ := 32.07

-- Definitions of molar masses of compounds
def molar_mass_C6H8O7 : ℝ := (6 * weight_C) + (8 * weight_H) + (7 * weight_O)
def molar_mass_H2SO4 : ℝ := (2 * weight_H) + weight_S + (4 * weight_O)

-- Definitions of number of moles
def moles_C6H8O7 : ℝ := 8
def moles_H2SO4 : ℝ := 4

-- Combined weight
def combined_weight : ℝ := (moles_C6H8O7 * molar_mass_C6H8O7) + (moles_H2SO4 * molar_mass_H2SO4)

theorem combined_weight_proof : combined_weight = 1929.48 :=
by
  -- calculations as explained in the problem
  let wC6H8O7 := moles_C6H8O7 * molar_mass_C6H8O7
  let wH2SO4 := moles_H2SO4 * molar_mass_H2SO4
  have h1 : wC6H8O7 = 8 * 192.14 := by sorry
  have h2 : wH2SO4 = 4 * 98.09 := by sorry
  have h3 : combined_weight = wC6H8O7 + wH2SO4 := by simp [combined_weight, wC6H8O7, wH2SO4]
  rw [h3, h1, h2]
  simp
  sorry -- finish the proof as necessary

end combined_weight_proof_l218_218912


namespace exists_special_number_l218_218468

theorem exists_special_number :
  ∃ N : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ 149 → k ∣ N) ∨ (k + 1 ∣ N) = false) :=
sorry

end exists_special_number_l218_218468


namespace quadratic_real_solutions_l218_218657

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end quadratic_real_solutions_l218_218657


namespace vasya_no_purchase_days_l218_218167

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218167


namespace seeds_per_flowerbed_l218_218474

theorem seeds_per_flowerbed (total_seeds flowerbeds : ℕ) (h1 : total_seeds = 32) (h2 : flowerbeds = 8) :
  total_seeds / flowerbeds = 4 :=
by {
  sorry
}

end seeds_per_flowerbed_l218_218474


namespace taeyeon_height_proof_l218_218696

noncomputable def seonghee_height : ℝ := 134.5
noncomputable def taeyeon_height : ℝ := seonghee_height * 1.06

theorem taeyeon_height_proof : taeyeon_height = 142.57 := 
by
  sorry

end taeyeon_height_proof_l218_218696


namespace complex_div_eq_i_l218_218622

noncomputable def i := Complex.I

theorem complex_div_eq_i : (1 + i) / (1 - i) = i := 
by
  sorry

end complex_div_eq_i_l218_218622


namespace royal_children_count_l218_218347

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l218_218347


namespace royal_children_count_l218_218365

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l218_218365


namespace power_of_three_l218_218782

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218782


namespace sequence_value_l218_218948

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end sequence_value_l218_218948


namespace power_equality_l218_218841

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218841


namespace mean_exterior_angles_l218_218284

theorem mean_exterior_angles (a b c : ℝ) (ha : a = 45) (hb : b = 75) (hc : c = 60) :
  (180 - a + 180 - b + 180 - c) / 3 = 120 :=
by 
  sorry

end mean_exterior_angles_l218_218284


namespace roots_of_equation_l218_218614

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l218_218614


namespace tracy_total_books_collected_l218_218153

variable (weekly_books_first_week : ℕ)
variable (multiplier : ℕ)
variable (weeks_next_period : ℕ)

-- Conditions
def first_week_books := 9
def second_period_books_per_week := first_week_books * 10
def books_next_five_weeks := second_period_books_per_week * 5

-- Theorem
theorem tracy_total_books_collected : 
  (first_week_books + books_next_five_weeks) = 459 := 
by 
  sorry

end tracy_total_books_collected_l218_218153


namespace notebook_cost_correct_l218_218239

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l218_218239


namespace vasya_purchase_l218_218213

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218213


namespace find_original_price_l218_218483

-- Define the conditions provided in the problem
def original_price (P : ℝ) : Prop :=
  let first_discount := 0.90 * P
  let second_discount := 0.85 * first_discount
  let taxed_price := 1.08 * second_discount
  taxed_price = 450

-- State and prove the main theorem
theorem find_original_price (P : ℝ) (h : original_price P) : P = 544.59 :=
  sorry

end find_original_price_l218_218483


namespace power_of_three_l218_218778

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218778


namespace prism_diagonal_correct_l218_218719

open Real

noncomputable def prism_diagonal_1 := 2 * sqrt 6
noncomputable def prism_diagonal_2 := sqrt 66

theorem prism_diagonal_correct (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  (prism_diagonal_1 = 2 * sqrt 6 ∧ prism_diagonal_2 = sqrt 66) :=
by
  sorry

end prism_diagonal_correct_l218_218719


namespace sum_of_coefficients_is_225_l218_218660

theorem sum_of_coefficients_is_225 :
  let C4 := 1
  let C41 := 4
  let C42 := 6
  let C43 := 4
  (C4 + C41 + C42 + C43)^2 = 225 :=
by
  sorry

end sum_of_coefficients_is_225_l218_218660


namespace train_length_correct_l218_218398

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l218_218398


namespace cyclist_speed_ratio_l218_218909

-- conditions: 
variables (T₁ T₂ o₁ o₂ : ℝ)
axiom h1 : o₁ + T₁ = o₂ + T₂
axiom h2 : T₁ = 2 * o₂
axiom h3 : T₂ = 4 * o₁

-- Proof statement to show that the second cyclist rides 1.5 times faster:
theorem cyclist_speed_ratio : T₁ / T₂ = 1.5 :=
by
  sorry

end cyclist_speed_ratio_l218_218909


namespace largest_divisor_for_odd_n_l218_218156

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem largest_divisor_for_odd_n (n : ℤ) (h : is_odd n ∧ n > 0) : 
  15 ∣ (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) := 
by 
  sorry

end largest_divisor_for_odd_n_l218_218156


namespace evaluate_expression_l218_218750

theorem evaluate_expression :
  4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 :=
by
  sorry

end evaluate_expression_l218_218750


namespace distance_inequality_l218_218860

theorem distance_inequality 
  (A B C D : Point)
  (dist : Point → Point → ℝ)
  (h_dist_pos : ∀ P Q : Point, dist P Q ≥ 0)
  (AC BD AD BC AB CD : ℝ)
  (hAC : AC = dist A C)
  (hBD : BD = dist B D)
  (hAD : AD = dist A D)
  (hBC : BC = dist B C)
  (hAB : AB = dist A B)
  (hCD : CD = dist C D) :
  AC^2 + BD^2 + AD^2 + BC^2 ≥ AB^2 + CD^2 := 
by
  sorry

end distance_inequality_l218_218860


namespace problem_statement_l218_218643

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l218_218643


namespace max_lessons_l218_218890

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end max_lessons_l218_218890


namespace scientific_calculator_ratio_l218_218737

theorem scientific_calculator_ratio (total : ℕ) (basic_cost : ℕ) (change : ℕ) (sci_ratio : ℕ → ℕ) (graph_ratio : ℕ → ℕ) : 
  total = 100 →
  basic_cost = 8 →
  sci_ratio basic_cost = 8 * x →
  graph_ratio (sci_ratio basic_cost) = 3 * sci_ratio basic_cost →
  change = 28 →
  8 + (8 * x) + (24 * x) = 72 →
  x = 2 :=
by
  sorry

end scientific_calculator_ratio_l218_218737


namespace number_of_children_l218_218366

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l218_218366


namespace Dave_guitar_strings_replacement_l218_218411

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end Dave_guitar_strings_replacement_l218_218411


namespace sum_of_squares_nonnegative_l218_218871

theorem sum_of_squares_nonnegative (x y z : ℝ) : x^2 + y^2 + z^2 - x * y - x * z - y * z ≥ 0 :=
  sorry

end sum_of_squares_nonnegative_l218_218871


namespace proof_problem_l218_218457

noncomputable def M : Set ℝ := { x | x ≥ 2 }
noncomputable def a : ℝ := Real.pi

theorem proof_problem : a ∈ M ∧ {a} ⊂ M :=
by
  sorry

end proof_problem_l218_218457


namespace find_sum_of_a_and_b_l218_218771

theorem find_sum_of_a_and_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (abs (x^2 - 2 * a * x + b) = 8) → (x = a ∨ x = a + 4 ∨ x = a - 4))
  (h2 : a^2 + (a - 4)^2 = (a + 4)^2) :
  a + b = 264 :=
by
  sorry

end find_sum_of_a_and_b_l218_218771


namespace area_difference_l218_218552

theorem area_difference (d : ℝ) (r : ℝ) (ratio : ℝ) (h1 : d = 10) (h2 : ratio = 2) (h3 : r = 5) :
  (π * r^2 - ((d^2 / (ratio^2 + 1)).sqrt * (2 * d^2 / (ratio^2 + 1)).sqrt)) = 38.5 :=
by
  sorry

end area_difference_l218_218552


namespace power_addition_l218_218800

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218800


namespace determine_a_l218_218138

theorem determine_a
  (a b : ℝ)
  (P1 P2 : ℝ × ℝ)
  (direction_vector : ℝ × ℝ)
  (h1 : P1 = (-3, 4))
  (h2 : P2 = (4, -1))
  (h3 : direction_vector = (4 - (-3), -1 - 4))
  (h4 : b = a / 2)
  (h5 : direction_vector = (7, -5)) :
  a = -10 :=
sorry

end determine_a_l218_218138


namespace inequality_and_equality_equality_condition_l218_218635

theorem inequality_and_equality (a b : ℕ) (ha : a > 1) (hb : b > 2) : a^b + 1 ≥ b * (a + 1) :=
by sorry

theorem equality_condition (a b : ℕ) : a = 2 ∧ b = 3 → a^b + 1 = b * (a + 1) :=
by
  intro h
  cases h
  sorry

end inequality_and_equality_equality_condition_l218_218635


namespace power_equality_l218_218845

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218845


namespace coefficient_third_term_expansion_l218_218035

theorem coefficient_third_term_expansion :
  ∀ x : ℝ, coefficient (expansion (1 - x) (expansion (1 + 2 * x) 5) 2) = 30 :=
by 
  sorry

end coefficient_third_term_expansion_l218_218035


namespace royal_family_children_l218_218357

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l218_218357


namespace vasya_days_l218_218175

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218175


namespace mutually_exclusive_not_complementary_l218_218024

-- Define the people
inductive Person
| A 
| B 
| C

open Person

-- Define the colors
inductive Color
| Red
| Yellow
| Blue

open Color

-- Event A: Person A gets the Red card
def event_a (assignment: Person → Color) : Prop := assignment A = Red

-- Event B: Person B gets the Red card
def event_b (assignment: Person → Color) : Prop := assignment B = Red

-- Definition of mutually exclusive events
def mutually_exclusive (P Q: Prop): Prop := P → ¬Q

-- Definition of complementary events
def complementary (P Q: Prop): Prop := P ↔ ¬Q

theorem mutually_exclusive_not_complementary :
  ∀ (assignment: Person → Color),
  mutually_exclusive (event_a assignment) (event_b assignment) ∧ ¬complementary (event_a assignment) (event_b assignment) :=
by
  sorry

end mutually_exclusive_not_complementary_l218_218024


namespace sodium_bicarbonate_moles_combined_l218_218252

theorem sodium_bicarbonate_moles_combined (HCl NaCl NaHCO3 : ℝ) (reaction : HCl + NaHCO3 = NaCl) 
  (HCl_eq_one : HCl = 1) (NaCl_eq_one : NaCl = 1) : 
  NaHCO3 = 1 := 
by 
  -- Placeholder for the proof
  sorry

end sodium_bicarbonate_moles_combined_l218_218252


namespace least_number_subtracted_divisible_by_5_l218_218157

def subtract_least_number (n : ℕ) (m : ℕ) : ℕ :=
  n % m

theorem least_number_subtracted_divisible_by_5 : subtract_least_number 9671 5 = 1 :=
by
  sorry

end least_number_subtracted_divisible_by_5_l218_218157


namespace power_of_3_l218_218826

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218826


namespace collinear_points_l218_218945

theorem collinear_points (k : ℝ) :
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  slope p1 p2 = slope p1 p3 → k = -1 :=
by 
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  sorry

end collinear_points_l218_218945


namespace expand_expression_l218_218027

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l218_218027


namespace vasya_days_l218_218181

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218181


namespace count_valid_subsets_l218_218996

theorem count_valid_subsets : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ⊆ {1, 2, 3, 4, 5} ∧ 
    (∀ a ∈ A, 6 - a ∈ A)) ∧ 
    S.card = 7 := 
sorry

end count_valid_subsets_l218_218996


namespace solve_for_x_l218_218687

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end solve_for_x_l218_218687


namespace square_area_twice_triangle_perimeter_l218_218742

noncomputable def perimeter_of_triangle (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area_of_square (side_length : ℕ) : ℕ :=
  side_length * side_length

theorem square_area_twice_triangle_perimeter (a b c : ℕ) (h1 : perimeter_of_triangle a b c = 22) (h2 : a = 5) (h3 : b = 7) (h4 : c = 10) : area_of_square (side_length_of_square (2 * perimeter_of_triangle a b c)) = 121 :=
by
  sorry

end square_area_twice_triangle_perimeter_l218_218742


namespace hundredth_number_is_201_l218_218664

-- Mathematical definition of the sequence
def counting_sequence (n : ℕ) : ℕ :=
  3 + (n - 1) * 2

-- Statement to prove
theorem hundredth_number_is_201 : counting_sequence 100 = 201 :=
by
  sorry

end hundredth_number_is_201_l218_218664


namespace floor_sqrt_50_l218_218588

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l218_218588


namespace vasya_purchase_l218_218211

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218211


namespace exact_two_solutions_l218_218599

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l218_218599


namespace min_a_value_l218_218984

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_a_value_l218_218984


namespace power_of_three_l218_218814

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218814


namespace find_m_l218_218886

noncomputable def m_value (m : ℝ) := 
  ((m ^ 2) - m - 1, (m ^ 2) - 2 * m - 1)

theorem find_m (m : ℝ) (h1 : (m ^ 2) - m - 1 = 1) (h2 : (m ^ 2) - 2 * m - 1 < 0) : 
  m = 2 :=
by sorry

end find_m_l218_218886


namespace flag_count_l218_218925

-- Definitions of colors as a datatype
inductive Color
| red : Color
| white : Color
| blue : Color
| green : Color
| yellow : Color

open Color

-- Total number of distinct flags possible
theorem flag_count : 
  (∃ m : Color, 
   (∃ t : Color, 
    (t ≠ m ∧ 
     ∃ b : Color, 
     (b ≠ m ∧ b ≠ red ∧ b ≠ blue)))) ∧ 
  (5 * 4 * 2 = 40) := 
  sorry

end flag_count_l218_218925


namespace five_digit_palindromes_count_l218_218413

theorem five_digit_palindromes_count :
  (∃ (A B C : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  ∃ (count : ℕ), count = 900 :=
by {
  intro h,
  use 900,
  sorry        -- Proof is omitted
}

end five_digit_palindromes_count_l218_218413


namespace royal_children_count_l218_218344

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l218_218344


namespace cone_surface_area_ratio_l218_218136

noncomputable def sector_angle := 135
noncomputable def sector_area (B : ℝ) := B
noncomputable def cone (A : ℝ) (B : ℝ) := A

theorem cone_surface_area_ratio (A B : ℝ) (h_sector_angle: sector_angle = 135) (h_sector_area: sector_area B = B) (h_cone_formed: cone A B = A) :
  A / B = 11 / 8 :=
by
  sorry

end cone_surface_area_ratio_l218_218136


namespace total_fat_served_l218_218385

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end total_fat_served_l218_218385


namespace vasya_days_without_purchase_l218_218205

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218205


namespace power_addition_l218_218807

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218807


namespace max_ab_min_4a2_b2_l218_218626

-- Define the positive numbers a and b and the condition 2a + b = 1
variables {a b : ℝ}
variable (h : a > 0 ∧ b > 0 ∧ 2 * a + b = 1)

-- Prove that the maximum value of ab is 1/8
theorem max_ab (h : a > 0 ∧ b > 0 ∧ 2 * a + b = 1) :
  ∃ a b, h → ab ≤ 1 / 8 :=
begin
  sorry  -- Actual proof goes here
end

-- Prove that the minimum value of 4a^2 + b^2 is 1/2
theorem min_4a2_b2 (h : a > 0 ∧ b > 0 ∧ 2 * a + b = 1) :
  ∃ a b, h → 4 * a^2 + b^2 ≥ 1 / 2 :=
begin
  sorry  -- Actual proof goes here
end

end max_ab_min_4a2_b2_l218_218626


namespace exponent_power_identity_l218_218793

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218793


namespace shared_divisors_count_l218_218577

-- Definitions and conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8820

-- Proof problem statement
theorem shared_divisors_count (a b : ℕ) (ha : a = num1) (hb : b = num2) :
  (Nat.divisors (Nat.gcd a b)).card = 24 := by
  subst_vars
  sorry

end shared_divisors_count_l218_218577


namespace spadesuit_eval_l218_218254

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 2 3) = 0 :=
by
  sorry

end spadesuit_eval_l218_218254


namespace trigonometric_expression_evaluation_l218_218642

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l218_218642


namespace field_trip_buses_l218_218692

-- Definitions of conditions
def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def seats_per_bus : ℕ := 72

-- Total calculations
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_people : ℕ := total_students + total_chaperones
def buses_needed : ℕ := (total_people + seats_per_bus - 1) / seats_per_bus

theorem field_trip_buses : buses_needed = 6 := by
  unfold buses_needed
  unfold total_people total_students total_chaperones chaperones_per_grade
  norm_num
  sorry

end field_trip_buses_l218_218692


namespace range_of_a_for_negative_root_l218_218763

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x + 1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_negative_root_l218_218763


namespace cube_volume_increase_l218_218282

variable (a : ℝ) (h : a ≥ 0)

theorem cube_volume_increase :
  ((2 * a) ^ 3) = 8 * (a ^ 3) :=
by sorry

end cube_volume_increase_l218_218282


namespace royal_family_children_l218_218348

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l218_218348


namespace polynomial_inequality_solution_l218_218429

theorem polynomial_inequality_solution :
  { x : ℝ | x * (x - 5) * (x - 10)^2 > 0 } = { x : ℝ | 0 < x ∧ x < 5 ∨ 10 < x } :=
by
  sorry

end polynomial_inequality_solution_l218_218429


namespace find_integers_l218_218424

theorem find_integers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) : 
  (a, b, c, d) = (1, 2, 3, 4) ∨ (a, b, c, d) = (1, 2, 4, 3) ∨ (a, b, c, d) = (1, 3, 2, 4) ∨ (a, b, c, d) = (1, 3, 4, 2) ∨ (a, b, c, d) = (1, 4, 2, 3) ∨ (a, b, c, d) = (1, 4, 3, 2) ∨ (a, b, c, d) = (2, 1, 3, 4) ∨ (a, b, c, d) = (2, 1, 4, 3) ∨ (a, b, c, d) = (2, 3, 1, 4) ∨ (a, b, c, d) = (2, 3, 4, 1) ∨ (a, b, c, d) = (2, 4, 1, 3) ∨ (a, b, c, d) = (2, 4, 3, 1) ∨ (a, b, c, d) = (3, 1, 2, 4) ∨ (a, b, c, d) = (3, 1, 4, 2) ∨ (a, b, c, d) = (3, 2, 1, 4) ∨ (a, b, c, d) = (3, 2, 4, 1) ∨ (a, b, c, d) = (3, 4, 1, 2) ∨ (a, b, c, d) = (3, 4, 2, 1) ∨ (a, b, c, d) = (4, 1, 2, 3) ∨ (a, b, c, d) = (4, 1, 3, 2) ∨ (a, b, c, d) = (4, 2, 1, 3) ∨ (a, b, c, d) = (4, 2, 3, 1) ∨ (a, b, c, d) = (4, 3, 1, 2) ∨ (a, b, c, d) = (4, 3, 2, 1) :=
sorry

end find_integers_l218_218424


namespace sum_cubes_div_product_eq_three_l218_218091

-- Given that x, y, z are non-zero real numbers and x + y + z = 3,
-- we need to prove that the possible value of (x^3 + y^3 + z^3) / xyz is 3.

theorem sum_cubes_div_product_eq_three 
  (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hxyz_sum : x + y + z = 3) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end sum_cubes_div_product_eq_three_l218_218091


namespace train_length_250_05_l218_218393

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  speed_m_s * time_s

theorem train_length_250_05 : length_of_train 60 15 = 250.05 :=
by
  -- Definitions from the problem
  let speed_km_hr := 60
  let time_s := 15
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let distance := speed_m_s * time_s
  -- The proven assertion
  show distance = 250.05
  sorry

end train_length_250_05_l218_218393


namespace vasya_did_not_buy_anything_days_l218_218218

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218218


namespace exists_distinct_abc_sum_l218_218466

theorem exists_distinct_abc_sum (n : ℕ) (h : n ≥ 1) (X : Finset ℤ)
  (h_card : X.card = n + 2)
  (h_abs : ∀ x ∈ X, abs x ≤ n) :
  ∃ (a b c : ℤ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

end exists_distinct_abc_sum_l218_218466


namespace royal_children_l218_218340

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l218_218340


namespace gdp_scientific_notation_l218_218288

noncomputable def gdp_nanning_2007 : ℝ := 1060 * 10^8

theorem gdp_scientific_notation :
  gdp_nanning_2007 = 1.06 * 10^11 :=
by sorry

end gdp_scientific_notation_l218_218288


namespace remainder_problem_l218_218467

theorem remainder_problem {x y z : ℤ} (h1 : x % 102 = 56) (h2 : y % 154 = 79) (h3 : z % 297 = 183) :
  x % 19 = 18 ∧ y % 22 = 13 ∧ z % 33 = 18 :=
by
  sorry

end remainder_problem_l218_218467


namespace problem_solution_l218_218647

theorem problem_solution
  (a d : ℝ)
  (h : (∀ x : ℝ, (x - 3) * (x + a) = x^2 + d * x - 18)) :
  d = 3 := 
sorry

end problem_solution_l218_218647


namespace has_exactly_two_solutions_iff_l218_218606

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l218_218606


namespace vasya_made_a_mistake_l218_218569

theorem vasya_made_a_mistake :
  ∀ x : ℝ, x^4 - 3*x^3 - 2*x^2 - 4*x + 1 = 0 → ¬ x < 0 :=
by sorry

end vasya_made_a_mistake_l218_218569


namespace imo_1989_q6_l218_218674

-- Define the odd integer m greater than 2
def isOdd (m : ℕ) := ∃ k : ℤ, m = 2 * k + 1

-- Define the condition for divisibility
def smallest_n (m : ℕ) (k : ℕ) (p : ℕ) : ℕ :=
  if k ≤ 1989 then 2 ^ (1989 - k) else 1

theorem imo_1989_q6 
  (m : ℕ) (h_m_gt2 : m > 2) (h_m_odd : isOdd m) (k : ℕ) (p : ℕ) (h_m_form : m = 2^k * p - 1) (h_p_odd : isOdd p) (h_k_gt1 : k > 1) :
  ∃ n : ℕ, (2^1989 ∣ m^n - 1) ∧ n = smallest_n m k p :=
by
  sorry

end imo_1989_q6_l218_218674


namespace find_negative_number_l218_218522

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end find_negative_number_l218_218522


namespace restore_price_by_percentage_l218_218333

theorem restore_price_by_percentage 
  (p : ℝ) -- original price
  (h₀ : p > 0) -- condition that price is positive
  (r₁ : ℝ := 0.25) -- reduction of 25%
  (r₁_applied : ℝ := p * (1 - r₁)) -- first reduction
  (r₂ : ℝ := 0.20) -- additional reduction of 20%
  (r₂_applied : ℝ := r₁_applied * (1 - r₂)) -- second reduction
  (final_price : ℝ := r₂_applied) -- final price after two reductions
  (increase_needed : ℝ := p - final_price) -- amount to increase to restore the price
  (percent_increase : ℝ := (increase_needed / final_price) * 100) -- percentage increase needed
  : abs (percent_increase - 66.67) < 0.01 := -- proof that percentage increase is approximately 66.67%
sorry

end restore_price_by_percentage_l218_218333


namespace ratio_of_legs_of_triangles_l218_218255

theorem ratio_of_legs_of_triangles (s a b : ℝ) (h1 : 0 < s)
  (h2 : a = s / 2)
  (h3 : b = (s * Real.sqrt 7) / 2) :
  b / a = Real.sqrt 7 := by
  sorry

end ratio_of_legs_of_triangles_l218_218255


namespace part1_part2_l218_218051

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem part1 (x : ℝ) (h : f x 2 ≥ 2) : x ≤ 1/2 ∨ x ≥ 2.5 := by
  sorry

theorem part2 (a : ℝ) (h_even : ∀ x : ℝ, f (-x) a = f x a) : a = -1 := by
  sorry

end part1_part2_l218_218051


namespace ned_did_not_wash_10_items_l218_218470

theorem ned_did_not_wash_10_items :
  let short_sleeve_shirts := 9
  let long_sleeve_shirts := 21
  let pairs_of_pants := 15
  let jackets := 8
  let total_items := short_sleeve_shirts + long_sleeve_shirts + pairs_of_pants + jackets
  let washed_items := 43
  let not_washed_Items := total_items - washed_items
  not_washed_Items = 10 := by
sorry

end ned_did_not_wash_10_items_l218_218470


namespace germination_relative_frequency_l218_218725

theorem germination_relative_frequency {n m : ℕ} (h₁ : n = 1000) (h₂ : m = 1000 - 90) : 
  (m : ℝ) / (n : ℝ) = 0.91 := by
  sorry

end germination_relative_frequency_l218_218725


namespace IncorrectStatement_l218_218519

-- Definitions of the events
def EventA (planeShot : ℕ → Prop) : Prop := planeShot 1 ∧ planeShot 2
def EventB (planeShot : ℕ → Prop) : Prop := ¬planeShot 1 ∧ ¬planeShot 2
def EventC (planeShot : ℕ → Prop) : Prop := (planeShot 1 ∧ ¬planeShot 2) ∨ (¬planeShot 1 ∧ planeShot 2)
def EventD (planeShot : ℕ → Prop) : Prop := planeShot 1 ∨ planeShot 2

-- Theorem statement to be proved (negation of the incorrect statement)
theorem IncorrectStatement (planeShot : ℕ → Prop) :
  ¬((EventA planeShot ∨ EventC planeShot) = (EventB planeShot ∨ EventD planeShot)) :=
by
  sorry

end IncorrectStatement_l218_218519


namespace find_certain_number_l218_218724

theorem find_certain_number (d q r : ℕ) (HD : d = 37) (HQ : q = 23) (HR : r = 16) :
    ∃ n : ℕ, n = d * q + r ∧ n = 867 := by
  sorry

end find_certain_number_l218_218724


namespace filling_material_heavier_than_sand_l218_218907

noncomputable def percentage_increase (full_sandbag_weight : ℝ) (partial_fill_percent : ℝ) (full_material_weight : ℝ) : ℝ :=
  let sand_weight := (partial_fill_percent / 100) * full_sandbag_weight
  let material_weight := full_material_weight
  let weight_increase := material_weight - sand_weight
  (weight_increase / sand_weight) * 100

theorem filling_material_heavier_than_sand :
  let full_sandbag_weight := 250
  let partial_fill_percent := 80
  let full_material_weight := 280
  percentage_increase full_sandbag_weight partial_fill_percent full_material_weight = 40 :=
by
  sorry

end filling_material_heavier_than_sand_l218_218907


namespace part_1_part_2_l218_218303

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def B (m : ℝ) : Set ℝ := { x | x^2 - (2*m + 1)*x + 2*m < 0 }

theorem part_1 (m : ℝ) (h : m < 1/2) : 
  B m = { x | 2*m < x ∧ x < 1 } := 
sorry

theorem part_2 (m : ℝ) : 
  (A ∪ B m = A) ↔ -1/2 ≤ m ∧ m ≤ 1 := 
sorry

end part_1_part_2_l218_218303


namespace basketball_problem_l218_218013

theorem basketball_problem :
  ∃ x y : ℕ, (3 + x + y = 14) ∧ (3 * 3 + 2 * x + y = 28) ∧ (x = 8) ∧ (y = 3) :=
by
  sorry

end basketball_problem_l218_218013


namespace functional_equation_solution_l218_218249

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, 
      y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)
  ) → (∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x) :=
by { sorry }

end functional_equation_solution_l218_218249


namespace simplify_expression_l218_218019

variable (a b : ℝ)

theorem simplify_expression : (a + b) * (3 * a - b) - b * (a - b) = 3 * a ^ 2 + a * b :=
by
  sorry

end simplify_expression_l218_218019


namespace vasya_purchase_l218_218209

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218209


namespace circle_radius_l218_218956

theorem circle_radius : ∀ (x y : ℝ), x^2 + 10*x + y^2 - 8*y + 25 = 0 → False := sorry

end circle_radius_l218_218956


namespace second_year_students_sampled_l218_218152

def total_students (f s t : ℕ) : ℕ := f + s + t

def proportion_second_year (s total_stu : ℕ) : ℚ := s / total_stu

def sampled_second_year_students (p : ℚ) (n : ℕ) : ℚ := p * n

theorem second_year_students_sampled
  (f s t : ℕ) (n : ℕ)
  (h1 : f = 600)
  (h2 : s = 780)
  (h3 : t = 720)
  (h4 : n = 35) :
  sampled_second_year_students (proportion_second_year s (total_students f s t)) n = 13 := 
sorry

end second_year_students_sampled_l218_218152


namespace intersection_of_sets_l218_218053

noncomputable def setA : Set ℝ := {x | 1 / (x - 1) ≤ 1}
def setB : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_sets : setA ∩ setB = {-1, 0, 2} := 
by
  sorry

end intersection_of_sets_l218_218053


namespace tickets_needed_l218_218714

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end tickets_needed_l218_218714


namespace necessary_but_not_sufficient_l218_218312

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  ((a ≥ 4 ∨ a ≤ 0) ↔ (∃ x : ℝ, f a x = 0)) ∧ ¬((a ≥ 4 ∨ a ≤ 0) → (∃ x : ℝ, f a x = 0)) :=
sorry

end necessary_but_not_sufficient_l218_218312


namespace quadratic_equation_roots_l218_218260

theorem quadratic_equation_roots (a b k k1 k2 : ℚ)
  (h_roots : ∀ x : ℚ, k * (x^2 - x) + x + 2 = 0)
  (h_ab_condition : (a / b) + (b / a) = 3 / 7)
  (h_k_values : ∀ x : ℚ, 7 * x^2 - 20 * x - 21 = 0)
  (h_k1k2 : k1 + k2 = 20 / 7)
  (h_k1k2_prod : k1 * k2 = -21 / 7) :
  (k1 / k2) + (k2 / k1) = -104 / 21 :=
sorry

end quadratic_equation_roots_l218_218260


namespace expression_f_range_a_l218_218093

noncomputable def f (x : ℝ) : ℝ :=
if h : -1 ≤ x ∧ x ≤ 1 then x^3
else if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
else (x-4)^3

theorem expression_f (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) :
  f x =
    if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
    else (x-4)^3 :=
by sorry

theorem range_a (a : ℝ) : 
  (∃ x, f x > a) ↔ a < 1 :=
by sorry

end expression_f_range_a_l218_218093


namespace find_stream_speed_l218_218542

-- Define the problem based on the provided conditions
theorem find_stream_speed (b s : ℝ) (h1 : b + s = 250 / 7) (h2 : b - s = 150 / 21) : s = 14.28 :=
by
  sorry

end find_stream_speed_l218_218542


namespace soccer_team_wins_l218_218004

theorem soccer_team_wins 
  (total_matches : ℕ)
  (total_points : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ)
  (points_per_loss : ℕ)
  (losses : ℕ)
  (H1 : total_matches = 10)
  (H2 : total_points = 17)
  (H3 : points_per_win = 3)
  (H4 : points_per_draw = 1)
  (H5 : points_per_loss = 0)
  (H6 : losses = 3) : 
  ∃ (wins : ℕ), wins = 5 := 
by
  sorry

end soccer_team_wins_l218_218004


namespace simplify_polynomial_l218_218686

def poly1 (x : ℝ) : ℝ := 5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7
def poly2 (x : ℝ) : ℝ := 7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3
def expected (x : ℝ) : ℝ := 12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3

theorem simplify_polynomial (x : ℝ) : poly1 x + poly2 x = expected x :=
  by sorry

end simplify_polynomial_l218_218686


namespace ratio_of_area_to_breadth_is_15_l218_218310

-- Definitions for our problem
def breadth := 5
def length := 15 -- since l - b = 10 and b = 5

-- Given conditions
axiom area_is_ktimes_breadth (k : ℝ) : length * breadth = k * breadth
axiom length_breadth_difference : length - breadth = 10

-- The proof statement
theorem ratio_of_area_to_breadth_is_15 : (length * breadth) / breadth = 15 := by
  sorry

end ratio_of_area_to_breadth_is_15_l218_218310


namespace complement_intersection_l218_218632

open Finset

-- Definitions of sets
def I : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {1, 3, 5}
def B : Finset ℕ := {2, 3, 6}
def C (S : Finset ℕ) : Finset ℕ := I \ S

theorem complement_intersection :
  (C A ∩ B) = {2, 6} := by
  sorry

end complement_intersection_l218_218632


namespace royal_children_count_l218_218362

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l218_218362


namespace probability_stopping_after_three_draws_l218_218501

def draws : List (List ℕ) := [
  [2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0], [0, 0, 1],
  [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [0, 3, 1], [3, 2, 0], [1, 2, 2], [1, 0, 3], [2, 3, 3]
]

def favorable_sequences (seqs : List (List ℕ)) : List (List ℕ) :=
  seqs.filter (λ seq => 0 ∈ seq ∧ 1 ∈ seq)

def probability_of_drawing_zhong_hua (seqs : List (List ℕ)) : ℚ :=
  (favorable_sequences seqs).length / seqs.length

theorem probability_stopping_after_three_draws :
  probability_of_drawing_zhong_hua draws = 5 / 18 := by
sorry

end probability_stopping_after_three_draws_l218_218501


namespace events_complementary_l218_218229

open Set

theorem events_complementary : 
  let A := {1, 3, 5}
  let B := {1, 2, 3}
  let C := {4, 5, 6}
  let S := {1, 2, 3, 4, 5, 6}
  (B ∩ C = ∅) ∧ (B ∪ C = S) :=
by
  let A := {1, 3, 5}
  let B := {1, 2, 3}
  let C := {4, 5, 6}
  let S := {1, 2, 3, 4, 5, 6}
  have h₁ : B ∩ C = ∅ := by sorry
  have h₂ : B ∪ C = S := by sorry
  exact ⟨h₁, h₂⟩
  sorry

end events_complementary_l218_218229


namespace carl_weight_l218_218746

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end carl_weight_l218_218746


namespace min_value_of_a_l218_218773

noncomputable def f (x a : ℝ) : ℝ :=
  Real.exp x * (x + (3 / x) - 3) - (a / x)

noncomputable def g (x : ℝ) : ℝ :=
  (x^2 - 3 * x + 3) * Real.exp x

theorem min_value_of_a (a : ℝ) :
  (∃ x > 0, f x a ≤ 0) → a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l218_218773


namespace correct_result_l218_218161

theorem correct_result (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end correct_result_l218_218161


namespace largest_divisor_same_remainder_l218_218515

theorem largest_divisor_same_remainder 
  (d : ℕ) (r : ℕ)
  (a b c : ℕ) 
  (h13511 : 13511 = a * d + r) 
  (h13903 : 13903 = b * d + r)
  (h14589 : 14589 = c * d + r) :
  d = 98 :=
by 
  sorry

end largest_divisor_same_remainder_l218_218515


namespace lottery_probability_exactly_one_common_l218_218072

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218072


namespace power_calculation_l218_218829

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218829


namespace product_not_divisible_by_201_l218_218318

theorem product_not_divisible_by_201 (a b : ℕ) (h₁ : a + b = 201) : ¬ (201 ∣ a * b) := sorry

end product_not_divisible_by_201_l218_218318


namespace tangent_line_at_pi_one_l218_218702

noncomputable def function (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1
noncomputable def tangent_line (x : ℝ) (y : ℝ) : ℝ := x * Real.exp Real.pi + y - 1 - Real.pi * Real.exp Real.pi

theorem tangent_line_at_pi_one :
  tangent_line x y = 0 ↔ y = function x → x = Real.pi ∧ y = 1 :=
by
  sorry

end tangent_line_at_pi_one_l218_218702


namespace single_reduction_equivalent_l218_218919

theorem single_reduction_equivalent (P : ℝ) (P_pos : 0 < P) : 
  (P - (P - 0.30 * P)) / P = 0.70 := 
by
  -- Let's denote the original price by P, 
  -- apply first 25% and then 60% reduction 
  -- and show that it's equivalent to a single 70% reduction
  sorry

end single_reduction_equivalent_l218_218919


namespace vasya_days_without_purchase_l218_218203

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218203


namespace find_ratio_l218_218776

-- Given conditions
variable (x y a b : ℝ)
variable (h1 : 2 * x - y = a)
variable (h2 : 4 * y - 8 * x = b)
variable (h3 : b ≠ 0)

theorem find_ratio (a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 4 := by
  sorry

end find_ratio_l218_218776


namespace royal_family_children_l218_218358

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l218_218358


namespace royal_family_children_l218_218353

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l218_218353


namespace continuous_stripe_probability_l218_218244

open ProbabilityTheory

noncomputable def total_stripe_combinations : ℕ := 4 ^ 6

noncomputable def favorable_stripe_outcomes : ℕ := 3 * 4

theorem continuous_stripe_probability :
  (favorable_stripe_outcomes : ℚ) / (total_stripe_combinations : ℚ) = 3 / 1024 := by
  sorry

end continuous_stripe_probability_l218_218244


namespace largest_number_among_list_l218_218727

theorem largest_number_among_list :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  sorry

end largest_number_among_list_l218_218727


namespace jack_round_trip_speed_l218_218857

noncomputable def jack_average_speed (d1 d2 : ℕ) (t1 t2 : ℕ) : ℕ :=
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let total_time_hours := total_time / 60
  total_distance / total_time_hours

theorem jack_round_trip_speed : jack_average_speed 3 3 45 15 = 6 := by
  -- Import necessary library
  sorry

end jack_round_trip_speed_l218_218857


namespace output_value_is_16_l218_218328

def f (x : ℤ) : ℤ :=
  if x < 0 then (x + 1) * (x + 1) else (x - 1) * (x - 1)

theorem output_value_is_16 : f 5 = 16 := by
  sorry

end output_value_is_16_l218_218328


namespace power_of_three_l218_218816

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218816


namespace lottery_probability_exactly_one_common_l218_218069

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218069


namespace field_trip_buses_needed_l218_218693

theorem field_trip_buses_needed
    (fifth_graders : ℕ) (sixth_graders : ℕ) (seventh_graders : ℕ)
    (teachers_per_grade : ℕ) (parents_per_grade : ℕ)
    (grades : ℕ) (seats_per_bus : ℕ)
    (H_fg : fifth_graders = 109)
    (H_sg : sixth_graders = 115)
    (H_sg2 : seventh_graders = 118)
    (H_tpg : teachers_per_grade = 4)
    (H_ppg : parents_per_grade = 2)
    (H_gr : grades = 3)
    (H_spb : seats_per_bus = 72) :
    let students := fifth_graders + sixth_graders + seventh_graders,
        adults := grades * (teachers_per_grade + parents_per_grade),
        total_people := students + adults in
    total_people / seats_per_bus = 5 := by
    sorry

end field_trip_buses_needed_l218_218693


namespace sum_of_roots_is_zero_l218_218439

-- Definitions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem Statement
theorem sum_of_roots_is_zero (f : ℝ → ℝ) (h_even : is_even f) (h_intersects : ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) : 
  x1 + x2 + x3 + x4 = 0 :=
by 
  sorry -- Proof can be provided here

end sum_of_roots_is_zero_l218_218439


namespace product_of_equal_numbers_l218_218127

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l218_218127


namespace find_range_for_two_real_solutions_l218_218052

noncomputable def f (k x : ℝ) := k * x
noncomputable def g (x : ℝ) := (Real.log x) / x

noncomputable def h (x : ℝ) := (Real.log x) / (x^2)

theorem find_range_for_two_real_solutions :
  (∃ k : ℝ, ∀ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → (f k x = g x ↔ k ∈ Set.Icc (1 / Real.exp 2) (1 / (2 * Real.exp 1)))) :=
sorry

end find_range_for_two_real_solutions_l218_218052


namespace probability_two_females_one_male_l218_218999

theorem probability_two_females_one_male
  (total_contestants : ℕ)
  (female_contestants : ℕ)
  (male_contestants : ℕ)
  (choose_count : ℕ)
  (total_combinations : ℕ)
  (female_combinations : ℕ)
  (male_combinations : ℕ)
  (favorable_outcomes : ℕ)
  (probability : ℚ)
  (h1 : total_contestants = 8)
  (h2 : female_contestants = 5)
  (h3 : male_contestants = 3)
  (h4 : choose_count = 3)
  (h5 : total_combinations = Nat.choose total_contestants choose_count)
  (h6 : female_combinations = Nat.choose female_contestants 2)
  (h7 : male_combinations = Nat.choose male_contestants 1)
  (h8 : favorable_outcomes = female_combinations * male_combinations)
  (h9 : probability = favorable_outcomes / total_combinations) :
  probability = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l218_218999


namespace find_y_given_conditions_l218_218710

theorem find_y_given_conditions (k : ℝ) (h1 : ∀ (x y : ℝ), xy = k) (h2 : ∀ (x y : ℝ), x + y = 30) (h3 : ∀ (x y : ℝ), x - y = 10) :
    ∀ x y, x = 8 → y = 25 :=
by
  sorry

end find_y_given_conditions_l218_218710


namespace distance_between_bars_l218_218335

theorem distance_between_bars (d V v : ℝ) 
  (h1 : x = 2 * d - 200)
  (h2 : d = P * V)
  (h3 : d - 200 = P * v)
  (h4 : V = (d - 200) / 4)
  (h5 : v = d / 9)
  (h6 : P = 4 * d / (d - 200))
  (h7 : P * (d - 200) = 8)
  (h8 : P * d = 18) :
  x = 1000 := by
  sorry

end distance_between_bars_l218_218335


namespace dress_cost_l218_218977

theorem dress_cost (x : ℝ) 
  (h1 : 30 * x = 10 + x) 
  (h2 : 3 * ((10 + x) / 30) = x) : 
  x = 10 / 9 :=
by
  sorry

end dress_cost_l218_218977


namespace power_of_3_l218_218821

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218821


namespace vasya_did_not_buy_anything_days_l218_218215

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218215


namespace decreasing_interval_l218_218579

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 2

theorem decreasing_interval : ∀ x : ℝ, (-2 < x ∧ x < 0) → (deriv f x < 0) := 
by
  sorry

end decreasing_interval_l218_218579


namespace smaller_angle_measure_l218_218507

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l218_218507


namespace money_together_l218_218562

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l218_218562


namespace vasya_no_purchase_days_l218_218171

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218171


namespace second_polygon_sides_l218_218322

theorem second_polygon_sides (s : ℝ) (n : ℝ) (h1 : 50 * 3 * s = n * s) : n = 150 := 
by
  sorry

end second_polygon_sides_l218_218322


namespace has_exactly_two_solutions_iff_l218_218603

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l218_218603


namespace probability_of_one_common_l218_218079

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l218_218079


namespace Merrill_marbles_Vivian_marbles_l218_218868

variable (M E S V : ℕ)

-- Conditions
axiom Merrill_twice_Elliot : M = 2 * E
axiom Merrill_Elliot_five_fewer_Selma : M + E = S - 5
axiom Selma_fifty_marbles : S = 50
axiom Vivian_35_percent_more_Elliot : V = (135 * E) / 100 -- since Lean works better with integers, use 135/100 instead of 1.35
axiom Vivian_Elliot_difference_greater_five : V - E > 5

-- Questions
theorem Merrill_marbles (M E S : ℕ) (h1: M = 2 * E) (h2: M + E = S - 5) (h3: S = 50) : M = 30 := by
  sorry

theorem Vivian_marbles (V E : ℕ) (h1: V = (135 * E) / 100) (h2: V - E > 5) (h3: E = 15) : V = 21 := by
  sorry

end Merrill_marbles_Vivian_marbles_l218_218868


namespace starting_player_ensures_non_trivial_solution_l218_218147

theorem starting_player_ensures_non_trivial_solution :
  ∀ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℚ), 
    ∃ (x y z : ℚ), 
    ((a1 * x + b1 * y + c1 * z = 0) ∧ 
     (a2 * x + b2 * y + c2 * z = 0) ∧ 
     (a3 * x + b3 * y + c3 * z = 0)) 
    ∧ ((a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2) = 0) ∧ 
         (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by
  intros a1 b1 c1 a2 b2 c2 a3 b3 c3
  sorry

end starting_player_ensures_non_trivial_solution_l218_218147


namespace functional_eq_app_only_solutions_l218_218753

noncomputable def f : Real → Real := sorry

theorem functional_eq_app (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2 :=
sorry

theorem only_solutions (f : ℝ → ℝ) (hf : ∀ n : ℕ, ∀ x : Fin n → ℝ, (∀ i, 0 ≤ x i) → f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2) :
  f = (fun x => 0) ∨ f = (fun x => x) :=
sorry

end functional_eq_app_only_solutions_l218_218753


namespace f_of_5_eq_1_l218_218462

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_5_eq_1
    (h1 : ∀ x : ℝ, f (-x) = -f x)
    (h2 : ∀ x : ℝ, f (-x) + f (x + 3) = 0)
    (h3 : f (-1) = 1) :
    f 5 = 1 :=
sorry

end f_of_5_eq_1_l218_218462


namespace arithmetic_sequence_num_terms_l218_218497

theorem arithmetic_sequence_num_terms (a_1 d S_n n : ℕ) 
  (h1 : a_1 = 4) (h2 : d = 3) (h3 : S_n = 650)
  (h4 : S_n = (n / 2) * (2 * a_1 + (n - 1) * d)) : n = 20 := by
  sorry

end arithmetic_sequence_num_terms_l218_218497


namespace train_speed_l218_218743

/-- Proof that calculates the speed of a train given the times to pass a man and a platform,
and the length of the platform, and shows it equals 54.00432 km/hr. -/
theorem train_speed (L V : ℝ) 
  (platform_length : ℝ := 360.0288)
  (time_to_pass_man : ℝ := 20)
  (time_to_pass_platform : ℝ := 44)
  (equation1 : L = V * time_to_pass_man)
  (equation2 : L + platform_length = V * time_to_pass_platform) :
  V = 15.0012 → V * 3.6 = 54.00432 :=
by sorry

end train_speed_l218_218743


namespace proof_problem_l218_218980

noncomputable def problem (a b c d : ℝ) : Prop :=
(a + b + c = 3) ∧ 
(a + b + d = -1) ∧ 
(a + c + d = 8) ∧ 
(b + c + d = 0) ∧ 
(a * b + c * d = -127 / 9)

theorem proof_problem (a b c d : ℝ) : 
  (a + b + c = 3) → 
  (a + b + d = -1) →
  (a + c + d = 8) → 
  (b + c + d = 0) → 
  (a * b + c * d = -127 / 9) :=
by 
  intro h1 h2 h3 h4
  -- Proof is omitted, "sorry" indicates it is to be filled in
  admit

end proof_problem_l218_218980


namespace max_lessons_l218_218888

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end max_lessons_l218_218888


namespace min_value_of_function_l218_218300

noncomputable def f (x y : ℝ) : ℝ := x^2 / (x + 2) + y^2 / (y + 1)

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  f x y ≥ 1 / 4 :=
sorry

end min_value_of_function_l218_218300


namespace vasya_days_without_purchase_l218_218189

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218189


namespace vasya_no_purchase_days_l218_218166

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218166


namespace distinct_digit_sum_equation_l218_218234

theorem distinct_digit_sum_equation :
  ∃ (F O R T Y S I X : ℕ), 
    F ≠ O ∧ F ≠ R ∧ F ≠ T ∧ F ≠ Y ∧ F ≠ S ∧ F ≠ I ∧ F ≠ X ∧ 
    O ≠ R ∧ O ≠ T ∧ O ≠ Y ∧ O ≠ S ∧ O ≠ I ∧ O ≠ X ∧ 
    R ≠ T ∧ R ≠ Y ∧ R ≠ S ∧ R ≠ I ∧ R ≠ X ∧ 
    T ≠ Y ∧ T ≠ S ∧ T ≠ I ∧ T ≠ X ∧ 
    Y ≠ S ∧ Y ≠ I ∧ Y ≠ X ∧ 
    S ≠ I ∧ S ≠ X ∧ 
    I ≠ X ∧ 
    FORTY = 10000 * F + 1000 * O + 100 * R + 10 * T + Y ∧ 
    TEN = 100 * T + 10 * E + N ∧ 
    SIXTY = 10000 * S + 1000 * I + 100 * X + 10 * T + Y ∧ 
    FORTY + TEN + TEN = SIXTY ∧ 
    SIXTY = 31486 :=
sorry

end distinct_digit_sum_equation_l218_218234


namespace original_cube_volume_l218_218097

theorem original_cube_volume 
  (a : ℕ) 
  (h : 3 * a * (a - a / 2) * a - a^3 = 2 * a^2) : 
  a = 4 → a^3 = 64 := 
by
  sorry

end original_cube_volume_l218_218097


namespace royal_children_count_l218_218342

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l218_218342


namespace subset_div_chain_l218_218994

theorem subset_div_chain (m n : ℕ) (h_m : m > 0) (h_n : n > 0) (S : Finset ℕ) (hS : S.card = (2^m - 1) * n + 1) (hS_subset : S ⊆ Finset.range (2^(m) * n + 1)) :
  ∃ (a : Fin (m+1) → ℕ), (∀ i, a i ∈ S) ∧ (∀ k : ℕ, k < m → a k ∣ a (k + 1)) :=
sorry

end subset_div_chain_l218_218994


namespace parents_give_per_year_l218_218469

def Mikail_age (x : ℕ) : Prop :=
  x = 3 * (x - 3)

noncomputable def money_per_year (total_money : ℕ) (age : ℕ) : ℕ :=
  total_money / age

theorem parents_give_per_year 
  (x : ℕ) (hx : Mikail_age x) : 
  money_per_year 45 x = 5 :=
sorry

end parents_give_per_year_l218_218469


namespace vasya_days_l218_218176

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218176


namespace proof_problem_l218_218901

noncomputable def aₙ (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d
noncomputable def Sₙ (a₁ d : ℝ) (n : ℕ) := n * a₁ + (n * (n - 1) / 2) * d

def given_conditions (Sₙ : ℕ → ℝ) : Prop :=
  Sₙ 10 = 0 ∧ Sₙ 15 = 25

theorem proof_problem (a₁ d : ℝ) (Sₙ : ℕ → ℝ)
  (h₁ : Sₙ 10 = 0) (h₂ : Sₙ 15 = 25) :
  (aₙ a₁ d 5 = -1/3) ∧
  (∀ n, Sₙ n = (1 / 3) * (n ^ 2 - 10 * n) → n = 5) ∧
  (∀ n, n * Sₙ n = (n ^ 3 / 3) - (10 * n ^ 2 / 3) → min (n * Sₙ n) = -49) ∧
  (¬ ∃ n, (Sₙ n / n) > 0) :=
sorry

end proof_problem_l218_218901


namespace children_count_l218_218374

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l218_218374


namespace minimum_perimeter_area_l218_218046

-- Define the focus point F of the parabola and point A
def F : ℝ × ℝ := (1, 0)  -- Focus for the parabola y² = 4x is (1, 0)
def A : ℝ × ℝ := (5, 4)

-- Parabola definition as a set of points (x, y) such that y² = 4x
def is_on_parabola (B : ℝ × ℝ) : Prop := B.2 * B.2 = 4 * B.1

-- The area of triangle ABF
def triangle_area (A B F : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 - B.1) * (A.2 - F.2) - (A.1 - F.1) * (A.2 - B.2))

-- Statement: The area of ∆ABF is 2 when the perimeter of ∆ABF is minimum
theorem minimum_perimeter_area (B : ℝ × ℝ) (hB : is_on_parabola B) 
  (hA_B_perimeter_min : ∀ (C : ℝ × ℝ), is_on_parabola C → 
                        (dist A C + dist C F ≥ dist A B + dist B F)) : 
  triangle_area A B F = 2 := 
sorry

end minimum_perimeter_area_l218_218046


namespace flowchart_output_is_minus_nine_l218_218703

-- Given initial state and conditions
def initialState : ℤ := 0

-- Hypothetical function representing the sequence of operations in the flowchart
-- (hiding the exact operations since they are speculative)
noncomputable def flowchartOperations (S : ℤ) : ℤ := S - 9  -- Assuming this operation represents the described flowchart

-- The proof problem
theorem flowchart_output_is_minus_nine : flowchartOperations initialState = -9 :=
by
  sorry

end flowchart_output_is_minus_nine_l218_218703


namespace sequence_general_term_l218_218774

open Nat

/-- Define the sequence recursively -/
def a : ℕ → ℤ
| 0     => -1
| (n+1) => 3 * a n - 1

/-- The general term of the sequence is given by - (3^n - 1) / 2 -/
theorem sequence_general_term (n : ℕ) : a n = - (3^n - 1) / 2 := 
by
  sorry

end sequence_general_term_l218_218774


namespace find_extrema_of_A_l218_218962

theorem find_extrema_of_A (x y : ℝ) (h : x^2 + y^2 = 4) : 2 ≤ x^2 + x * y + y^2 ∧ x^2 + x * y + y^2 ≤ 6 :=
by 
  sorry

end find_extrema_of_A_l218_218962


namespace solve_for_x_l218_218689

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_l218_218689


namespace P_is_sufficient_but_not_necessary_for_Q_l218_218618

def P (x : ℝ) : Prop := (2 * x - 3)^2 < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_is_sufficient_but_not_necessary_for_Q : 
  (∀ x, P x → Q x) ∧ (∃ x, Q x ∧ ¬ P x) :=
by
  sorry

end P_is_sufficient_but_not_necessary_for_Q_l218_218618


namespace notebook_cost_correct_l218_218238

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l218_218238


namespace royal_children_count_l218_218360

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l218_218360


namespace factorize_expr_l218_218423

theorem factorize_expr (a : ℝ) : a^2 - 8 * a = a * (a - 8) :=
sorry

end factorize_expr_l218_218423


namespace arithmetic_sequence_term_count_l218_218943

theorem arithmetic_sequence_term_count (a d l n : ℕ) (h1 : a = 11) (h2 : d = 4) (h3 : l = 107) :
  l = a + (n - 1) * d → n = 25 := by
  sorry

end arithmetic_sequence_term_count_l218_218943


namespace toys_lost_l218_218231

theorem toys_lost (initial_toys found_in_closet total_after_finding : ℕ) 
  (h1 : initial_toys = 40) 
  (h2 : found_in_closet = 9) 
  (h3 : total_after_finding = 43) : 
  initial_toys - (total_after_finding - found_in_closet) = 9 :=
by 
  sorry

end toys_lost_l218_218231


namespace Keiko_speed_l218_218296

theorem Keiko_speed (a b s : ℝ) (h1 : 8 = 8) 
  (h2 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : 
  s = π / 3 :=
by
  sorry

end Keiko_speed_l218_218296


namespace solve_for_x_l218_218688

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end solve_for_x_l218_218688


namespace vanya_correct_answers_l218_218099

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l218_218099


namespace vasya_did_not_buy_anything_days_l218_218216

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218216


namespace triangle_side_length_l218_218447

theorem triangle_side_length (A : ℝ) (AC BC AB : ℝ) 
  (hA : A = 60)
  (hAC : AC = 4)
  (hBC : BC = 2 * Real.sqrt 3) :
  AB = 2 :=
sorry

end triangle_side_length_l218_218447


namespace find_three_power_l218_218256

theorem find_three_power (m n : ℕ) (h₁: 3^m = 4) (h₂: 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end find_three_power_l218_218256


namespace sum_of_three_terms_divisible_by_3_l218_218263

theorem sum_of_three_terms_divisible_by_3 (a : Fin 5 → ℤ) :
  ∃ (i j k : Fin 5), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (a i + a j + a k) % 3 = 0 :=
by
  sorry

end sum_of_three_terms_divisible_by_3_l218_218263


namespace vasya_did_not_buy_anything_days_l218_218221

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218221


namespace prob1_prob2_prob3_l218_218095

-- Define the sequences for rows ①, ②, and ③
def seq1 (n : ℕ) : ℤ := (-2) ^ n
def seq2 (m : ℕ) : ℤ := (-2) ^ (m - 1)
def seq3 (m : ℕ) : ℤ := (-2) ^ (m - 1) - 1

-- Prove the $n^{th}$ number in row ①
theorem prob1 (n : ℕ) : seq1 n = (-2) ^ n :=
by sorry

-- Prove the relationship between $m^{th}$ numbers in row ② and row ③
theorem prob2 (m : ℕ) : seq3 m = seq2 m - 1 :=
by sorry

-- Prove the value of $x + y + z$ where $x$, $y$, and $z$ are the $2019^{th}$ numbers in rows ①, ②, and ③, respectively
theorem prob3 : seq1 2019 + seq2 2019 + seq3 2019 = -1 :=
by sorry

end prob1_prob2_prob3_l218_218095


namespace vanya_correct_answers_l218_218101

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l218_218101


namespace polynomial_equality_l218_218297

noncomputable def monic_poly (n : ℕ) (f : Polynomial ℂ) : Prop :=
  f.leadingCoeff = 1 ∧ f.natDegree = n

theorem polynomial_equality (n : ℕ) (f g : Polynomial ℂ)
  (a_i b_i c_i : Fin n → ℂ)
  (hf : monic_poly n f)
  (hg : monic_poly n g)
  (h_eq : f - g = ∏ i, (Polynomial.C (a_i i) * Polynomial.X + Polynomial.C (b_i i) * Polynomial.Y + Polynomial.C (c_i i))) :
  ∃ a b c : ℂ, f = (Polynomial.X + Polynomial.C a) ^ n + Polynomial.C c ∧ g = (Polynomial.Y + Polynomial.C b) ^ n + Polynomial.C c :=
sorry

end polynomial_equality_l218_218297


namespace log_base_250_2662sqrt10_l218_218224

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variables (a b : ℝ)
variables (h1 : log_base 50 55 = a) (h2 : log_base 55 20 = b)

theorem log_base_250_2662sqrt10 : log_base 250 (2662 * Real.sqrt 10) = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) :=
by
  sorry

end log_base_250_2662sqrt10_l218_218224


namespace total_fat_served_l218_218386

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end total_fat_served_l218_218386


namespace abs_eq_case_l218_218415

theorem abs_eq_case (x : ℝ) (h : |x - 3| = |x + 2|) : x = 1/2 :=
by
  sorry

end abs_eq_case_l218_218415


namespace sqrt_floor_eq_seven_l218_218593

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l218_218593


namespace rainfall_march_correct_l218_218143

def rainfall_march : ℝ :=
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  total_expected - total_april_to_july

theorem rainfall_march_correct (march_rainfall : ℝ) :
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  march_rainfall = total_expected - total_april_to_july :=
by
  sorry

end rainfall_march_correct_l218_218143


namespace lottery_probability_exactly_one_common_l218_218073

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218073


namespace vasya_days_without_purchases_l218_218197

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218197


namespace roots_of_equation_l218_218612

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l218_218612


namespace find_original_price_l218_218856

def initial_price (P : ℝ) : Prop :=
  let first_discount := P * 0.76
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 1.10
  final_price = 532

theorem find_original_price : ∃ P : ℝ, initial_price P :=
sorry

end find_original_price_l218_218856


namespace steps_in_staircase_l218_218906

theorem steps_in_staircase :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n = 19 :=
by
  sorry

end steps_in_staircase_l218_218906


namespace wilson_total_cost_l218_218527

noncomputable def total_cost_wilson_pays : ℝ :=
let hamburger_price : ℝ := 5
let cola_price : ℝ := 2
let fries_price : ℝ := 3
let sundae_price : ℝ := 4
let nugget_price : ℝ := 1.5
let salad_price : ℝ := 6.25
let hamburger_count : ℕ := 2
let cola_count : ℕ := 3
let nugget_count : ℕ := 4

let total_before_discounts := (hamburger_count * hamburger_price) +
                              (cola_count * cola_price) +
                              fries_price +
                              sundae_price +
                              (nugget_count * nugget_price) +
                              salad_price

let free_nugget_discount := 1 * nugget_price
let total_after_promotion := total_before_discounts - free_nugget_discount
let coupon_discount := 4
let total_after_coupon := total_after_promotion - coupon_discount
let loyalty_discount := 0.10 * total_after_coupon
let total_after_loyalty := total_after_coupon - loyalty_discount

total_after_loyalty

theorem wilson_total_cost : total_cost_wilson_pays = 26.77 := 
by
  sorry

end wilson_total_cost_l218_218527


namespace lottery_probability_exactly_one_common_l218_218075

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218075


namespace binom_n_n_sub_2_l218_218721

theorem binom_n_n_sub_2 (n : ℕ) (h : n > 0) : (Nat.choose n (n - 2)) = (n * (n - 1)) / 2 := by
  sorry

end binom_n_n_sub_2_l218_218721


namespace lines_parallel_if_perpendicular_to_same_plane_l218_218299

variable {Plane Line : Type}
variable {α β γ : Plane}
variable {m n : Line}

-- Define perpendicularity and parallelism as axioms for simplicity
axiom perp (L : Line) (P : Plane) : Prop
axiom parallel (L1 L2 : Line) : Prop

-- Assume conditions for the theorem
variables (h1 : perp m α) (h2 : perp n α)

-- The theorem proving the required relationship
theorem lines_parallel_if_perpendicular_to_same_plane : parallel m n := 
by
  sorry

end lines_parallel_if_perpendicular_to_same_plane_l218_218299


namespace game_cost_l218_218160

theorem game_cost
    (initial_amount : ℕ)
    (cost_per_toy : ℕ)
    (num_toys : ℕ)
    (remaining_amount := initial_amount - cost_per_toy * num_toys)
    (cost_of_game := initial_amount - remaining_amount)
    (h1 : initial_amount = 57)
    (h2 : cost_per_toy = 6)
    (h3 : num_toys = 5) :
  cost_of_game = 27 :=
by
  sorry

end game_cost_l218_218160


namespace ellipse_foci_distance_l218_218427

theorem ellipse_foci_distance 
  (a b : ℝ) 
  (h_a : a = 8) 
  (h_b : b = 3) : 
  2 * (Real.sqrt (a^2 - b^2)) = 2 * Real.sqrt 55 := 
by
  rw [h_a, h_b]
  sorry

end ellipse_foci_distance_l218_218427


namespace notebook_cost_3_dollars_l218_218237

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l218_218237


namespace find_a1_in_arithmetic_sequence_l218_218862

noncomputable def arithmetic_sequence_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem find_a1_in_arithmetic_sequence :
  ∀ (a₁ d : ℤ), d = -2 →
  (arithmetic_sequence_sum a₁ d 11 = arithmetic_sequence_sum a₁ d 10) →
  a₁ = 20 :=
by
  intro a₁ d hd hs
  sorry

end find_a1_in_arithmetic_sequence_l218_218862


namespace probability_exactly_one_common_number_l218_218065

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l218_218065


namespace vasya_days_without_purchases_l218_218195

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218195


namespace right_triangle_isosceles_l218_218740

-- Define the conditions for a right-angled triangle inscribed in a circle
variables (a b : ℝ)

-- Conditions provided in the problem
def right_triangle_inscribed (a b : ℝ) : Prop :=
  ∃ h : a ≠ 0 ∧ b ≠ 0, 2 * (a^2 + b^2) = (a + 2*b)^2 + b^2 ∧ 2 * (a^2 + b^2) = (2 * a + b)^2 + a^2

-- The theorem to prove based on the conditions
theorem right_triangle_isosceles (a b : ℝ) (h : right_triangle_inscribed a b) : a = b :=
by 
  sorry

end right_triangle_isosceles_l218_218740


namespace power_of_three_l218_218781

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218781


namespace vasya_no_purchase_days_l218_218172

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218172


namespace student_ages_inconsistent_l218_218081

theorem student_ages_inconsistent :
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  total_age_all_students < total_age_group1 + total_age_group2 + total_age_group3 :=
by {
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  have h₁ : total_age_all_students = 24 * 18 := rfl
  have h₂ : total_age_group1 = 6 * 16 := rfl
  have h₃ : total_age_group2 = 10 * 20 := rfl
  have h₄ : total_age_group3 = 7 * 22 := rfl
  have h₅ : 432 = 24 * 18 := by norm_num
  have h₆ : 96 = 6 * 16 := by norm_num
  have h₇ : 200 = 10 * 20 := by norm_num
  have h₈ : 154 = 7 * 22 := by norm_num
  have h₉ : 432 < 96 + 200 + 154 := by norm_num
  exact h₉
}

end student_ages_inconsistent_l218_218081


namespace product_of_equal_numbers_l218_218135

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l218_218135


namespace arithmetic_sequence_a1_a6_l218_218450

theorem arithmetic_sequence_a1_a6
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) : a 1 * a 6 = 14 :=
sorry

end arithmetic_sequence_a1_a6_l218_218450


namespace Ramya_reads_total_124_pages_l218_218096

theorem Ramya_reads_total_124_pages :
  let total_pages : ℕ := 300
  let pages_read_monday := (1/5 : ℚ) * total_pages
  let pages_remaining := total_pages - pages_read_monday
  let pages_read_tuesday := (4/15 : ℚ) * pages_remaining
  pages_read_monday + pages_read_tuesday = 124 := 
by
  sorry

end Ramya_reads_total_124_pages_l218_218096


namespace RobertAteNine_l218_218874

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

end RobertAteNine_l218_218874


namespace find_a_l218_218057

theorem find_a : 
  (∃ a : ℝ, (binom 5 2) * a^3 * (-1)^2 = 80) → a = 2 :=
by
  intro h,
  cases h with a ha,
  have h1 : (binom 5 2) = 10 := by sorry,  -- This should be replaced with an appropriate library theorem about binomial coefficients.
  have h2 : (-1)^2 = 1 := by norm_num,
  rw [h1, h2] at ha,
  rw ← mul_assoc at ha,
  have h3 : 10 * a^3 = 80 := ha,
  norm_num at h3,
  rw ← eq_div_iff at h3,
  norm_num at h3,
  have h4 : a^3 = 8 := h3,
  have h5 : a = real.cbrt 8 := by sorry,
  have h6 : real.cbrt 8 = 2 := by norm_num,
  rw h6 at h5,
  exact h5


end find_a_l218_218057


namespace vanya_correct_answers_l218_218114

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218114


namespace parabola_line_unique_eq_l218_218631

noncomputable def parabola_line_equation : Prop :=
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 4 * A.1) ∧ (B.2^2 = 4 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) ∧ ((A.2 + B.2) / 2 = 2) ∧
    ∀ x y, (y - 2 = 1 * (x - 2)) → (x - y = 0)

theorem parabola_line_unique_eq : parabola_line_equation :=
  sorry

end parabola_line_unique_eq_l218_218631


namespace albert_complete_laps_l218_218747

theorem albert_complete_laps (D L : ℝ) (I : ℕ) (hD : D = 256.5) (hL : L = 9.7) (hI : I = 6) :
  ⌊(D - I * L) / L⌋ = 20 :=
by
  sorry

end albert_complete_laps_l218_218747


namespace vasya_days_without_purchase_l218_218182

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218182


namespace sqrt_difference_square_l218_218621

theorem sqrt_difference_square (a b : ℝ) (h₁ : a = Real.sqrt 3 + Real.sqrt 2) (h₂ : b = Real.sqrt 3 - Real.sqrt 2) : a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end sqrt_difference_square_l218_218621


namespace power_of_three_l218_218785

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218785


namespace sequence_formula_l218_218436

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n + 1) :
  (a 1 = 3) ∧ (∀ n, n ≥ 2 → a n = 2 * n) :=
by
  sorry

end sequence_formula_l218_218436


namespace exponent_power_identity_l218_218795

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218795


namespace power_of_3_l218_218827

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218827


namespace exist_consecutive_days_20_games_l218_218226

theorem exist_consecutive_days_20_games 
  (a : ℕ → ℕ)
  (h_daily : ∀ n, a (n + 1) - a n ≥ 1)
  (h_weekly : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ a j - a i = 20 := by 
  sorry

end exist_consecutive_days_20_games_l218_218226


namespace intersection_in_fourth_quadrant_l218_218852

theorem intersection_in_fourth_quadrant (m : ℝ) :
  let x := (3 * m + 2) / 4
  let y := (-m - 2) / 8
  (x > 0) ∧ (y < 0) ↔ (m > -2 / 3) :=
by
  sorry

end intersection_in_fourth_quadrant_l218_218852


namespace no_solution_for_inequality_system_l218_218484

theorem no_solution_for_inequality_system (x : ℝ) : 
  ¬ ((2 * x + 3 ≥ x + 11) ∧ (((2 * x + 5) / 3 - 1) < (2 - x))) :=
by
  sorry

end no_solution_for_inequality_system_l218_218484


namespace distinct_solution_condition_l218_218608

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l218_218608


namespace basketball_game_score_difference_l218_218087

theorem basketball_game_score_difference :
  let blueFreeThrows := 18
  let blueTwoPointers := 25
  let blueThreePointers := 6
  let redFreeThrows := 15
  let redTwoPointers := 22
  let redThreePointers := 5
  let blueScore := blueFreeThrows * 1 + blueTwoPointers * 2 + blueThreePointers * 3
  let redScore := redFreeThrows * 1 + redTwoPointers * 2 + redThreePointers * 3
  blueScore - redScore = 12 := by
  sorry

end basketball_game_score_difference_l218_218087


namespace chess_group_players_l218_218502

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
by
  sorry

end chess_group_players_l218_218502


namespace badminton_tournament_l218_218289

theorem badminton_tournament (n x : ℕ) (h1 : 2 * n > 0) (h2 : 3 * n > 0) (h3 : (5 * n) * (5 * n - 1) = 14 * x) : n = 3 :=
by
  -- Placeholder for the proof
  sorry

end badminton_tournament_l218_218289


namespace michael_hours_worked_l218_218332

def michael_hourly_rate := 7
def michael_overtime_rate := 2 * michael_hourly_rate
def work_hours := 40
def total_earnings := 320

theorem michael_hours_worked :
  (total_earnings = michael_hourly_rate * work_hours + michael_overtime_rate * (42 - work_hours)) :=
sorry

end michael_hours_worked_l218_218332


namespace prism_volume_l218_218148

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end prism_volume_l218_218148


namespace always_positive_sum_reciprocal_inequality_l218_218921

-- Problem 1
theorem always_positive (x : ℝ) : x^6 - x^3 + x^2 - x + 1 > 0 :=
sorry

-- Problem 2
theorem sum_reciprocal_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  1/a + 1/b + 1/c ≥ 9 :=
sorry

end always_positive_sum_reciprocal_inequality_l218_218921


namespace inequality_l218_218301

variable {a b c : ℝ}

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := 
by 
  sorry

end inequality_l218_218301


namespace find_some_number_l218_218649

-- The conditions of the problem
variables (x y : ℝ)
axiom cond1 : 2 * x + y = 7
axiom cond2 : x + 2 * y = 5

-- The "some number" we want to prove exists
def some_number := 3

-- Statement of the problem: the value of 2xy / some_number should equal 2
theorem find_some_number (x y : ℝ) (cond1 : 2 * x + y = 7) (cond2 : x + 2 * y = 5) :
  2 * x * y / some_number = 2 :=
sorry

end find_some_number_l218_218649


namespace probability_exactly_one_common_number_l218_218064

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l218_218064


namespace spinner_probability_divisible_by_8_l218_218932

-- Define the spinner sections
def sections : Set ℕ := {1, 2, 3, 4}

-- Check if a number is divisible by 8
def divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

-- Define the event of interest where the number is divisible by 8
def event (hundreds tens units : ℕ) : Prop :=
  divisible_by_8 (100 * hundreds + 10 * tens + units)

-- The main theorem stating the probability is 1/8
theorem spinner_probability_divisible_by_8 :
  (∑ h in sections, ∑ t in sections, ∑ u in sections, if event h t u then 1 else 0) / (sections.card ^ 3) = 1 / 8 := by
  sorry

end spinner_probability_divisible_by_8_l218_218932


namespace number_of_children_l218_218367

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l218_218367


namespace problem_l218_218056

noncomputable def cubeRoot (x : ℝ) : ℝ :=
  x ^ (1 / 3)

theorem problem (t : ℝ) (h : t = 1 / (1 - cubeRoot 2)) :
  t = (1 + cubeRoot 2) * (1 + cubeRoot 4) :=
by
  sorry

end problem_l218_218056


namespace calculate_expression_l218_218018

theorem calculate_expression :
  ((1 / 3 : ℝ) ^ (-2 : ℝ)) + Real.tan (Real.pi / 4) - Real.sqrt ((-10 : ℝ) ^ 2) = 0 := by
  sorry

end calculate_expression_l218_218018


namespace vanya_correct_answers_l218_218117

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218117


namespace num_marked_cells_at_least_num_cells_in_one_square_l218_218222

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end num_marked_cells_at_least_num_cells_in_one_square_l218_218222


namespace number_of_persons_l218_218533

theorem number_of_persons (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
sorry

end number_of_persons_l218_218533


namespace apple_slices_per_group_l218_218583

-- defining the conditions
variables (a g : ℕ)

-- 1. Equal number of apple slices and grapes in groups
def equal_group (a g : ℕ) : Prop := a = g

-- 2. Grapes packed in groups of 9
def grapes_groups_of_9 (g : ℕ) : Prop := ∃ k : ℕ, g = 9 * k

-- 3. Smallest number of grapes is 18
def smallest_grapes (g : ℕ) : Prop := g = 18

-- theorem stating that the number of apple slices per group is 9
theorem apple_slices_per_group : equal_group a g ∧ grapes_groups_of_9 g ∧ smallest_grapes g → a = 9 := by
  sorry

end apple_slices_per_group_l218_218583


namespace max_n_for_factoring_l218_218954

theorem max_n_for_factoring (n : ℤ) :
  (∃ A B : ℤ, (5 * B + A = n) ∧ (A * B = 90)) → n = 451 :=
by
  sorry

end max_n_for_factoring_l218_218954


namespace point_on_y_axis_l218_218285

theorem point_on_y_axis (m n : ℝ) (h : (m, n).1 = 0) : m = 0 :=
by
  sorry

end point_on_y_axis_l218_218285


namespace vasya_days_without_purchases_l218_218193

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218193


namespace expression_evaluation_l218_218920

def eval_expression : Int := 
  let a := -2 ^ 3
  let b := abs (2 - 3)
  let c := -2 * (-1) ^ 2023
  a + b + c

theorem expression_evaluation :
  eval_expression = -5 :=
by
  sorry

end expression_evaluation_l218_218920


namespace evaluate_g_l218_218463

def g (a b c d : ℤ) : ℚ := (d * (c + 2 * a)) / (c + b)

theorem evaluate_g : g 4 (-1) (-8) 2 = 0 := 
by 
  sorry

end evaluate_g_l218_218463


namespace simplify_division_l218_218233

theorem simplify_division (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := 
by sorry

end simplify_division_l218_218233


namespace circumscribed_sphere_radius_l218_218293

theorem circumscribed_sphere_radius (a b c : ℝ) : 
  R = (1/2) * Real.sqrt (a^2 + b^2 + c^2) := sorry

end circumscribed_sphere_radius_l218_218293


namespace find_discount_percentage_l218_218228

noncomputable def discount_percentage (P B S : ℝ) (H1 : B = P * (1 - D / 100)) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : ℝ :=
D

theorem find_discount_percentage (P B S : ℝ) (H1 : B = P * (1 - (60 / 100))) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : 
  discount_percentage P B S H1 H2 H3 = 60 := sorry

end find_discount_percentage_l218_218228


namespace probability_exactly_one_common_number_l218_218067

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l218_218067


namespace speed_of_car_in_second_hour_l218_218900

noncomputable def speed_in_first_hour : ℝ := 90
noncomputable def average_speed : ℝ := 82.5
noncomputable def total_time : ℝ := 2

theorem speed_of_car_in_second_hour : 
  ∃ (speed_in_second_hour : ℝ), 
  (speed_in_first_hour + speed_in_second_hour) / total_time = average_speed ∧ 
  speed_in_first_hour = 90 ∧ 
  average_speed = 82.5 → 
  speed_in_second_hour = 75 :=
by 
  sorry

end speed_of_car_in_second_hour_l218_218900


namespace evaluate_expression_l218_218247

theorem evaluate_expression :
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  (x^3 * y^2 * z^2 * w) = (1 / 48 : ℚ) :=
by
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  sorry

end evaluate_expression_l218_218247


namespace length_percentage_increase_l218_218884

/--
Given that the area of a rectangle is 460 square meters and the breadth is 20 meters,
prove that the percentage increase in length compared to the breadth is 15%.
-/
theorem length_percentage_increase (A : ℝ) (b : ℝ) (l : ℝ) (hA : A = 460) (hb : b = 20) (hl : l = A / b) :
  ((l - b) / b) * 100 = 15 :=
by
  sorry

end length_percentage_increase_l218_218884


namespace relationship_x_a_b_l218_218880

theorem relationship_x_a_b (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) : 
  x^2 > a * b ∧ a * b > a^2 :=
by
  sorry

end relationship_x_a_b_l218_218880


namespace one_div_lt_one_div_of_gt_l218_218137

theorem one_div_lt_one_div_of_gt {a b : ℝ} (hab : a > b) (hb0 : b > 0) : (1 / a) < (1 / b) :=
sorry

end one_div_lt_one_div_of_gt_l218_218137


namespace min_value_proof_l218_218669

noncomputable def min_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 3 * y = 1) : ℝ :=
  if hx : x = 0 ∨ y = 0 then 
    0 -- this case will not occur due to the h₁ and h₂ constraints
  else
    let a := (1 / x) + (1 / y)
    in 5 + 3 * Real.sqrt 3

theorem min_value_proof (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 3 * y = 1) :
  min_value x y h₁ h₂ h₃ = 5 + 3 * Real.sqrt 3 :=
by
  sorry

end min_value_proof_l218_218669


namespace wine_count_l218_218558

theorem wine_count (S B total W : ℕ) (hS : S = 22) (hB : B = 17) (htotal : S - B + W = total) (htotal_val : total = 31) : W = 26 :=
by
  sorry

end wine_count_l218_218558


namespace min_period_and_max_value_l218_218050

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

theorem min_period_and_max_value :
  (∀ x, f (x + π) = f x) ∧ (∀ x, f x ≤ 4) ∧ (∃ x, f x = 4) :=
by
  sorry

end min_period_and_max_value_l218_218050


namespace curlers_total_l218_218025

theorem curlers_total (P B G : ℕ) (h1 : 4 * P = P + B + G) (h2 : B = 2 * P) (h3 : G = 4) : 
  4 * P = 16 := 
by sorry

end curlers_total_l218_218025


namespace directrix_of_parabola_l218_218756

theorem directrix_of_parabola (a b c : ℝ) (h_eqn : ∀ x, b = -4 * x^2 + c) : 
  b = 5 → c = 0 → (∃ y, y = 81 / 16) :=
by
  sorry

end directrix_of_parabola_l218_218756


namespace total_people_in_cars_by_end_of_race_l218_218500

-- Define the initial conditions and question
def initial_num_cars : ℕ := 20
def initial_num_passengers_per_car : ℕ := 2
def initial_num_drivers_per_car : ℕ := 1
def extra_passengers_per_car : ℕ := 1

-- Define the number of people per car initially
def initial_people_per_car : ℕ := initial_num_passengers_per_car + initial_num_drivers_per_car

-- Define the number of people per car after gaining extra passenger
def final_people_per_car : ℕ := initial_people_per_car + extra_passengers_per_car

-- The statement to be proven
theorem total_people_in_cars_by_end_of_race : initial_num_cars * final_people_per_car = 80 := by
  -- Prove the theorem
  sorry

end total_people_in_cars_by_end_of_race_l218_218500


namespace children_count_l218_218377

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l218_218377


namespace inequality_l218_218865

theorem inequality (a b c d e p q : ℝ) 
  (h0 : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (h1 : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * ((1 / a) + (1 / b) + (1 / c) + (1 / d) + (1 / e)) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
by
  sorry

end inequality_l218_218865


namespace tangent_lines_to_circle_passing_through_point_l218_218416

theorem tangent_lines_to_circle_passing_through_point :
  ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 1 → ((x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = -1)) :=
by
  sorry

end tangent_lines_to_circle_passing_through_point_l218_218416


namespace probability_not_snow_l218_218492

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l218_218492


namespace GrandmaOlga_grandchildren_l218_218975

theorem GrandmaOlga_grandchildren :
  (∃ d : ℕ, d = 3 ∧ ∀ i : Fin d, 6 ∈ ℕ) ∧
  (∃ s : ℕ, s = 3 ∧ ∀ j : Fin s, 5 ∈ ℕ) →
  18 + 15 = 33 :=
by
  intros h
  cases' h with h_d h_s
  cases' h_d with d_vals num_d
  cases' d_vals with d_eq d_cond
  cases' h_s with s_vals num_s
  cases' s_vals with s_eq s_cond
  sorry

end GrandmaOlga_grandchildren_l218_218975


namespace total_students_count_l218_218648

theorem total_students_count (n1 n2 n: ℕ) (avg1 avg2 avg_tot: ℝ)
  (h1: n1 = 15) (h2: avg1 = 70) (h3: n2 = 10) (h4: avg2 = 90) (h5: avg_tot = 78)
  (h6: (n1 * avg1 + n2 * avg2) / (n1 + n2) = avg_tot) :
  n = 25 :=
by
  sorry

end total_students_count_l218_218648


namespace Vanya_correct_answers_l218_218109

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l218_218109


namespace total_games_played_l218_218419

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end total_games_played_l218_218419


namespace range_of_a_l218_218772

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l218_218772


namespace lockers_number_l218_218557

theorem lockers_number (total_cost : ℝ) (cost_per_digit : ℝ) (total_lockers : ℕ) 
  (locker_numbered_from_one : ∀ n : ℕ, n >= 1) :
  total_cost = 248.43 → cost_per_digit = 0.03 → total_lockers = 2347 :=
by
  intros h_total_cost h_cost_per_digit
  sorry

end lockers_number_l218_218557


namespace power_of_three_l218_218813

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218813


namespace max_lessons_l218_218887

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end max_lessons_l218_218887


namespace condition_neither_sufficient_nor_necessary_l218_218223

variable (a b : ℝ)

theorem condition_neither_sufficient_nor_necessary 
    (h1 : ∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2))
    (h2 : ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b)) :
  ¬((a > b) ↔ (a^2 > b^2)) :=
sorry

end condition_neither_sufficient_nor_necessary_l218_218223


namespace hyperbola_asymptotes_iff_l218_218870

def hyperbola_asymptotes_orthogonal (a b c d e f : ℝ) : Prop :=
  a + c = 0

theorem hyperbola_asymptotes_iff (a b c d e f : ℝ) :
  (∃ x y : ℝ, a * x^2 + 2 * b * x * y + c * y^2 + d * x + e * y + f = 0) →
  hyperbola_asymptotes_orthogonal a b c d e f ↔ a + c = 0 :=
by sorry

end hyperbola_asymptotes_iff_l218_218870


namespace volume_of_given_solid_l218_218391

noncomputable def volume_of_solid (s : ℝ) (h : ℝ) : ℝ :=
  (h / 3) * (s^2 + (s * (3 / 2))^2 + (s * (3 / 2)) * s)

theorem volume_of_given_solid : volume_of_solid 8 10 = 3040 / 3 :=
by
  sorry

end volume_of_given_solid_l218_218391


namespace max_sum_marks_l218_218597

theorem max_sum_marks (a b c : ℕ) (h1 : a + b + c = 2019) (h2 : a ≤ c + 2) : 
  2 * a + b ≤ 2021 :=
by {
  -- We'll skip the proof but formulate the statement following conditions strictly.
  sorry
}

end max_sum_marks_l218_218597


namespace find_a_if_line_passes_through_center_l218_218445

-- Define the given circle equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the given line equation
def line_eqn (x y a : ℝ) : Prop := 3*x + y + a = 0

-- The coordinates of the center of the circle
def center_of_circle : (ℝ × ℝ) := (-1, 2)

-- Prove that a = 1 if the line passes through the center of the circle
theorem find_a_if_line_passes_through_center (a : ℝ) :
  line_eqn (-1) 2 a → a = 1 :=
by
  sorry

end find_a_if_line_passes_through_center_l218_218445


namespace exact_two_solutions_l218_218601

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l218_218601


namespace find_k_unique_solution_l218_218944

theorem find_k_unique_solution (k : ℝ) (h: k ≠ 0) : (∀ x : ℝ, (x + 3) / (k * x - 2) = x → k = -3/4) :=
sorry

end find_k_unique_solution_l218_218944


namespace C_days_to_finish_l218_218528

theorem C_days_to_finish (A B C : ℝ) 
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  -- Given equations
  have h1 : A + B = 1 / 15 := sorry
  have h2 : A + B + C = 1 / 11 := sorry
  -- Calculate C
  let C := 1 / 11 - 1 / 15
  -- Calculate days taken by C
  let days := 1 / C
  -- Prove the days equal to 41.25
  have days_eq : 41.25 = 165 / 4 := sorry
  exact sorry

end C_days_to_finish_l218_218528


namespace smaller_angle_measure_l218_218509

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l218_218509


namespace vasya_days_l218_218174

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218174


namespace order_of_6_with_respect_to_f_is_undefined_l218_218694

noncomputable def f (x : ℕ) : ℕ := x ^ 2 % 13

def order_of_6_undefined : Prop :=
  ∀ m : ℕ, m > 0 → f^[m] 6 ≠ 6

theorem order_of_6_with_respect_to_f_is_undefined : order_of_6_undefined :=
by
  sorry

end order_of_6_with_respect_to_f_is_undefined_l218_218694


namespace vasya_days_l218_218177

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218177


namespace max_value_of_expression_l218_218615

noncomputable def max_value (x : ℝ) : ℝ :=
  x * (1 + x) * (3 - x)

theorem max_value_of_expression :
  ∃ x : ℝ, 0 < x ∧ max_value x = (70 + 26 * Real.sqrt 13) / 27 :=
sorry

end max_value_of_expression_l218_218615


namespace exponent_power_identity_l218_218791

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218791


namespace solve_for_x_l218_218690

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_l218_218690


namespace rectangle_perimeter_l218_218554

theorem rectangle_perimeter (s : ℕ) (h : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end rectangle_perimeter_l218_218554


namespace hyperbola_focal_length_l218_218488

noncomputable def a : ℝ := Real.sqrt 10
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt (a ^ 2 + b ^ 2)
noncomputable def focal_length : ℝ := 2 * c

theorem hyperbola_focal_length :
  focal_length = 4 * Real.sqrt 3 := by
  sorry

end hyperbola_focal_length_l218_218488


namespace tan_neg_two_simplifies_l218_218639

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l218_218639


namespace number_is_4_less_than_opposite_l218_218378

-- Define the number and its opposite relationship
def opposite_relation (x : ℤ) : Prop := x = -x + (-4)

-- Theorem stating that the given number is 4 less than its opposite
theorem number_is_4_less_than_opposite (x : ℤ) : opposite_relation x :=
sorry

end number_is_4_less_than_opposite_l218_218378


namespace vasya_purchase_l218_218207

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218207


namespace length_of_real_axis_of_hyperbola_l218_218705

theorem length_of_real_axis_of_hyperbola :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 -> ∃ a : ℝ, 2 * a = 4 :=
by
intro x y h
sorry

end length_of_real_axis_of_hyperbola_l218_218705


namespace rational_iff_geometric_progression_l218_218481

theorem rational_iff_geometric_progression :
  (∃ x a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + a)*(x + c) = (x + b)^2) ↔
  (∃ x : ℚ, ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + (a : ℚ))*(x + (c : ℚ)) = (x + (b : ℚ))^2) :=
sorry

end rational_iff_geometric_progression_l218_218481


namespace min_value_quadratic_l218_218324

theorem min_value_quadratic (x : ℝ) : 
  ∃ m, m = 3 * x^2 - 18 * x + 2048 ∧ ∀ x, 3 * x^2 - 18 * x + 2048 ≥ 2021 :=
by sorry

end min_value_quadratic_l218_218324


namespace complement_U_P_l218_218766

theorem complement_U_P :
  let U := {y : ℝ | y ≠ 0 }
  let P := {y : ℝ | 0 < y ∧ y < 1/2}
  let complement_U_P := {y : ℝ | y ∈ U ∧ y ∉ P}
  (complement_U_P = {y : ℝ | y < 0} ∪ {y : ℝ | y > 1/2}) :=
by
  sorry

end complement_U_P_l218_218766


namespace vincent_total_laundry_loads_l218_218514

theorem vincent_total_laundry_loads :
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  total_loads = 26 :=
by {
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  show total_loads = 26
  sorry
}

end vincent_total_laundry_loads_l218_218514


namespace vasya_purchase_l218_218206

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218206


namespace project_B_days_l218_218544

theorem project_B_days (B : ℕ) : 
  (1 / 20 + 1 / B) * 10 + (1 / B) * 5 = 1 -> B = 30 :=
by
  sorry

end project_B_days_l218_218544


namespace cos120_sin_neg45_equals_l218_218905

noncomputable def cos120_plus_sin_neg45 : ℝ :=
  Real.cos (120 * Real.pi / 180) + Real.sin (-45 * Real.pi / 180)

theorem cos120_sin_neg45_equals : cos120_plus_sin_neg45 = - (1 + Real.sqrt 2) / 2 :=
by
  sorry

end cos120_sin_neg45_equals_l218_218905


namespace range_of_smallest_side_l218_218048

theorem range_of_smallest_side 
  (c : ℝ) -- the perimeter of the triangle
  (a : ℝ) (b : ℝ) (A : ℝ)  -- three sides of the triangle
  (ha : 0 < a) 
  (hb : b = 2 * a) 
  (hc : a + b + A = c)
  (htriangle : a + b > A ∧ a + A > b ∧ b + A > a) 
  : 
  ∃ (l u : ℝ), l = c / 6 ∧ u = c / 4 ∧ l < a ∧ a < u 
:= sorry

end range_of_smallest_side_l218_218048


namespace power_of_three_l218_218815

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218815


namespace arithmetic_calculation_l218_218016

theorem arithmetic_calculation : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 :=
by
  sorry

end arithmetic_calculation_l218_218016


namespace min_value_frac_inv_sum_l218_218668

theorem min_value_frac_inv_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + 3 * y = 1) : 
  ∃ (minimum_value : ℝ), minimum_value = 4 + 2 * Real.sqrt 3 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 1 → (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3) := 
sorry

end min_value_frac_inv_sum_l218_218668


namespace number_of_action_figures_removed_l218_218991

-- Definitions for conditions
def initial : ℕ := 15
def added : ℕ := 2
def current : ℕ := 10

-- The proof statement
theorem number_of_action_figures_removed (initial added current : ℕ) : 
  (initial + added - current) = 7 := by
  sorry

end number_of_action_figures_removed_l218_218991


namespace sale_price_tea_correct_l218_218933

noncomputable def sale_price_of_mixed_tea (weight1 weight2 price1 price2 profit_percentage : ℝ) : ℝ :=
let total_cost := weight1 * price1 + weight2 * price2
let total_weight := weight1 + weight2
let cost_price_per_kg := total_cost / total_weight
let profit_per_kg := profit_percentage * cost_price_per_kg
let sale_price_per_kg := cost_price_per_kg + profit_per_kg
sale_price_per_kg

theorem sale_price_tea_correct :
  sale_price_of_mixed_tea 80 20 15 20 0.20 = 19.2 :=
  by
  sorry

end sale_price_tea_correct_l218_218933


namespace Vanya_correct_answers_l218_218121

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l218_218121


namespace find_integer_solutions_l218_218951

theorem find_integer_solutions :
  {p : ℤ × ℤ | 2 * p.1^3 + p.1 * p.2 = 7} = {(-7, -99), (-1, -9), (1, 5), (7, -97)} :=
by
  -- Proof not required
  sorry

end find_integer_solutions_l218_218951


namespace largest_of_options_l218_218330

theorem largest_of_options :
  max (2 + 0 + 1 + 3) (max (2 * 0 + 1 + 3) (max (2 + 0 * 1 + 3) (max (2 + 0 + 1 * 3) (2 * 0 * 1 * 3)))) = 2 + 0 + 1 + 3 := by sorry

end largest_of_options_l218_218330


namespace vasya_days_without_purchases_l218_218196

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218196


namespace total_amount_is_200_l218_218566

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l218_218566


namespace vasya_days_without_purchase_l218_218201

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218201


namespace lottery_probability_exactly_one_common_l218_218068

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218068


namespace stops_away_pinedale_mall_from_yahya_house_l218_218698

-- Definitions based on problem conditions
def bus_speed_kmh : ℕ := 60
def stop_interval_minutes : ℕ := 5
def distance_to_mall_km : ℕ := 40

-- Definition of how many stops away is Pinedale mall from Yahya's house
def stops_to_mall : ℕ := distance_to_mall_km / (bus_speed_kmh / 60 * stop_interval_minutes)

-- Lean statement to prove the given conditions lead to the correct number of stops
theorem stops_away_pinedale_mall_from_yahya_house :
  stops_to_mall = 8 :=
by 
  -- This is a placeholder for the proof. 
  -- Actual proof steps would convert units and calculate as described in the problem.
  sorry

end stops_away_pinedale_mall_from_yahya_house_l218_218698


namespace money_together_l218_218561

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l218_218561


namespace shakes_indeterminable_l218_218230

theorem shakes_indeterminable (B S C x : ℝ) (h1 : 3 * B + 7 * S + C = 120) (h2 : 4 * B + x * S + C = 164.50) : ¬ (∃ B S C, ∀ x, 4 * B + x * S + C = 164.50) → false := 
by 
  sorry

end shakes_indeterminable_l218_218230


namespace temperature_problem_l218_218749

theorem temperature_problem (N : ℤ) (P : ℤ) (D : ℤ) (D_3_pm : ℤ) (P_3_pm : ℤ) :
  D = P + N →
  D_3_pm = D - 8 →
  P_3_pm = P + 9 →
  |D_3_pm - P_3_pm| = 1 →
  (N = 18 ∨ N = 16) →
  18 * 16 = 288 :=
by
  sorry

end temperature_problem_l218_218749


namespace power_addition_l218_218804

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218804


namespace valid_permutations_count_l218_218777

/-- 
Given five elements consisting of the numbers 1, 2, 3, and the symbols "+" and "-", 
we want to count the number of permutations such that no two numbers are adjacent.
-/
def count_valid_permutations : Nat := 
  let number_permutations := Nat.factorial 3 -- 3! permutations of 1, 2, 3
  let symbol_insertions := Nat.factorial 2  -- 2! permutations of "+" and "-"
  number_permutations * symbol_insertions

theorem valid_permutations_count : count_valid_permutations = 12 := by
  sorry

end valid_permutations_count_l218_218777


namespace vanya_correct_answers_l218_218118

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218118


namespace train_probability_correct_l218_218934

noncomputable def train_prob (a_train b_train a_john b_john wait : ℝ) : ℝ :=
  let total_time_frame := (b_train - a_train) * (b_john - a_john)
  let triangle_area := (1 / 2) * wait * wait
  let rectangle_area := wait * wait
  let total_overlap_area := triangle_area + rectangle_area
  total_overlap_area / total_time_frame

theorem train_probability_correct :
  train_prob 120 240 150 210 30 = 3 / 16 :=
by
  sorry

end train_probability_correct_l218_218934


namespace go_game_prob_l218_218659

theorem go_game_prob :
  ∀ (pA pB : ℝ),
    (pA = 0.6) →
    (pB = 0.4) →
    ((pA ^ 2) + (pB ^ 2) = 0.52) :=
by
  intros pA pB hA hB
  rw [hA, hB]
  sorry

end go_game_prob_l218_218659


namespace pencils_left_l218_218401

theorem pencils_left (anna_pencils : ℕ) (harry_pencils : ℕ)
  (h_anna : anna_pencils = 50) (h_harry : harry_pencils = 2 * anna_pencils)
  (lost_pencils : ℕ) (h_lost : lost_pencils = 19) :
  harry_pencils - lost_pencils = 81 :=
by
  sorry

end pencils_left_l218_218401


namespace floor_sqrt_50_l218_218589

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l218_218589


namespace find_point_on_y_axis_l218_218265

/-- 
Given points A (1, 2, 3) and B (2, -1, 4), and a point P on the y-axis 
such that the distances |PA| and |PB| are equal, 
prove that the coordinates of point P are (0, -7/6, 0).
 -/
theorem find_point_on_y_axis
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, 3))
  (hB : B = (2, -1, 4))
  (P : ℝ × ℝ × ℝ)
  (hP : ∃ y : ℝ, P = (0, y, 0)) :
  dist A P = dist B P → P = (0, -7/6, 0) :=
by
  sorry

end find_point_on_y_axis_l218_218265


namespace children_count_l218_218376

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l218_218376


namespace zog_words_count_l218_218473

-- Defining the number of letters in the Zoggian alphabet
def num_letters : ℕ := 6

-- Function to calculate the number of words with n letters
def words_with_n_letters (n : ℕ) : ℕ := num_letters ^ n

-- Definition to calculate the total number of words with at most 4 letters
def total_words : ℕ :=
  (words_with_n_letters 1) +
  (words_with_n_letters 2) +
  (words_with_n_letters 3) +
  (words_with_n_letters 4)

-- Theorem statement
theorem zog_words_count : total_words = 1554 := by
  sorry

end zog_words_count_l218_218473


namespace isosceles_trapezoid_inscribed_circle_ratio_l218_218748

noncomputable def ratio_perimeter_inscribed_circle (x : ℝ) : ℝ := 
  (50 * x) / (10 * Real.pi * x)

theorem isosceles_trapezoid_inscribed_circle_ratio 
  (x : ℝ)
  (h1 : x > 0)
  (r : ℝ) 
  (OK OP : ℝ) 
  (h2 : OK = 3 * x) 
  (h3 : OP = 5 * x) : 
  ratio_perimeter_inscribed_circle x = 5 / Real.pi :=
by
  sorry

end isosceles_trapezoid_inscribed_circle_ratio_l218_218748


namespace apples_total_l218_218958

-- Definitions as per conditions
def apples_on_tree : Nat := 5
def initial_apples_on_ground : Nat := 8
def apples_eaten_by_dog : Nat := 3

-- Calculate apples left on the ground
def apples_left_on_ground : Nat := initial_apples_on_ground - apples_eaten_by_dog

-- Calculate total apples left
def total_apples_left : Nat := apples_on_tree + apples_left_on_ground

theorem apples_total : total_apples_left = 10 := by
  -- the proof will go here
  sorry

end apples_total_l218_218958


namespace problem_statement_l218_218636

theorem problem_statement (x : ℝ) (h : 2 * x^2 + 1 = 17) : 4 * x^2 + 1 = 33 :=
by sorry

end problem_statement_l218_218636


namespace possible_values_of_b_l218_218477

theorem possible_values_of_b (b : ℝ) : (¬ ∃ x : ℝ, x^2 + b * x + 1 ≤ 0) → -2 < b ∧ b < 2 :=
by
  intro h
  sorry

end possible_values_of_b_l218_218477


namespace man_work_rate_l218_218388

theorem man_work_rate (W : ℝ) (M S : ℝ)
  (h1 : (M + S) * 3 = W)
  (h2 : S * 5.25 = W) :
  M * 7 = W :=
by 
-- The proof steps will be filled in here.
sorry

end man_work_rate_l218_218388


namespace power_addition_l218_218799

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218799


namespace lottery_probability_exactly_one_common_l218_218071

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l218_218071


namespace reynald_soccer_balls_l218_218684

theorem reynald_soccer_balls (total_balls basketballs_more soccer tennis baseball more_baseballs volleyballs : ℕ) 
(h_total_balls: total_balls = 145) 
(h_basketballs_more: basketballs_more = 5)
(h_tennis: tennis = 2 * soccer)
(h_more_baseballs: more_baseballs = 10)
(h_volleyballs: volleyballs = 30) 
(sum_eq: soccer + (soccer + basketballs_more) + tennis + (soccer + more_baseballs) + volleyballs = total_balls) : soccer = 20 := 
by
  sorry

end reynald_soccer_balls_l218_218684


namespace difference_between_wins_and_losses_l218_218382

noncomputable def number_of_wins (n m : ℕ) : Prop :=
  0 ≤ n ∧ 0 ≤ m ∧ n + m ≤ 42 ∧ n + (42 - n - m) / 2 = 30 / 1

theorem difference_between_wins_and_losses (n m : ℕ) (h : number_of_wins n m) : n - m = 18 :=
sorry

end difference_between_wins_and_losses_l218_218382


namespace shaded_area_ratio_l218_218518

-- Definitions based on conditions
def large_square_area : ℕ := 16
def shaded_components : ℕ := 4
def component_fraction : ℚ := 1 / 2
def shaded_square_area : ℚ := shaded_components * component_fraction
def large_square_area_q : ℚ := large_square_area

-- Goal statement
theorem shaded_area_ratio : (shaded_square_area / large_square_area_q) = (1 / 8) :=
by sorry

end shaded_area_ratio_l218_218518


namespace julia_total_spend_l218_218452

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end julia_total_spend_l218_218452


namespace algebraic_expression_value_l218_218257

theorem algebraic_expression_value (a b c : ℝ) (h : (∀ x : ℝ, (x - 1) * (x + 2) = a * x^2 + b * x + c)) :
  4 * a - 2 * b + c = 0 :=
sorry

end algebraic_expression_value_l218_218257


namespace steve_keeps_total_money_excluding_advance_l218_218879

-- Definitions of the conditions
def totalCopies : ℕ := 1000000
def advanceCopies : ℕ := 100000
def pricePerCopy : ℕ := 2
def agentCommissionRate : ℚ := 0.1

-- Question and final proof
theorem steve_keeps_total_money_excluding_advance :
  let totalEarnings := totalCopies * pricePerCopy
  let agentCommission := agentCommissionRate * totalEarnings
  let moneyKept := totalEarnings - agentCommission
  moneyKept = 1800000 := by
  -- Proof goes here, but we skip it for now
  sorry

end steve_keeps_total_money_excluding_advance_l218_218879


namespace T_n_formula_l218_218969

-- Define the given sequence sum S_n
def S (n : ℕ) : ℚ := (n^2 : ℚ) / 2 + (3 * n : ℚ) / 2

-- Define the general term a_n for the sequence {a_n}
def a (n : ℕ) : ℚ := if n = 1 then 2 else n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 2) - a n + 1 / (a (n + 2) * a n)

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℚ := 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3))

-- Prove the equality of T_n with the given expression
theorem T_n_formula (n : ℕ) : T n = 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := sorry

end T_n_formula_l218_218969


namespace expand_expression_l218_218028

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l218_218028


namespace tangent_and_normal_lines_l218_218729

noncomputable def x (t : ℝ) := 2 * Real.exp t
noncomputable def y (t : ℝ) := Real.exp (-t)

theorem tangent_and_normal_lines (t0 : ℝ) (x0 y0 : ℝ) (m_tangent m_normal : ℝ)
  (hx0 : x0 = x t0)
  (hy0 : y0 = y t0)
  (hm_tangent : m_tangent = -(1 / 2))
  (hm_normal : m_normal = 2) :
  (∀ x y : ℝ, y = m_tangent * x + 2) ∧ (∀ x y : ℝ, y = m_normal * x - 3) :=
by
  sorry

end tangent_and_normal_lines_l218_218729


namespace vanya_correct_answers_l218_218112

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218112


namespace construct_angle_from_19_l218_218262

theorem construct_angle_from_19 (θ : ℝ) (h : θ = 19) : ∃ n : ℕ, (n * θ) % 360 = 75 :=
by
  -- Placeholder for the proof
  sorry

end construct_angle_from_19_l218_218262


namespace intersecting_lines_l218_218283

variable (a b m : ℝ)

-- Conditions
def condition1 : Prop := 8 = -m + a
def condition2 : Prop := 8 = m + b

-- Statement to prove
theorem intersecting_lines : condition1 a m  → condition2 b m  → a + b = 16 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l218_218283


namespace number_of_children_l218_218371

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l218_218371


namespace quadratic_real_solutions_l218_218654

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end quadratic_real_solutions_l218_218654


namespace geometric_sequence_common_ratio_l218_218988

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a (n + 1) > a n) (h2 : a 2 = 2) (h3 : a 4 - a 3 = 4) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l218_218988


namespace range_of_theta_l218_218015

-- Definitions
def regular_hexagon := { A B C D E F : Type }

-- Conditions
def mid_point (A B : Type) := sorry
def point_on_side (X Y : Type) := sorry

-- The range condition for θ
theorem range_of_theta (A B C D E F P Q : Type) (hex : regular_hexagon)
  (hP : mid_point A B)
  (hQ : point_on_side B C)
  (hθ : ∃ θ : ℝ, θ = real.angle B P Q) :
  arcsin (3 * sqrt 3 / sqrt 127 : ℝ) < θ ∧ θ < arcsin (3 * sqrt 3 / sqrt 91 : ℝ) :=
sorry

end range_of_theta_l218_218015


namespace tank_missing_water_l218_218923

def max_capacity := 350000
def loss1_rate := 32000
def loss1_duration := 5
def loss2_rate := 10000
def loss2_duration := 10
def fill_rate := 40000
def fill_duration := 3

theorem tank_missing_water 
  (max_capacity = 350000)
  (loss1_rate = 32000)
  (loss1_duration = 5)
  (loss2_rate = 10000)
  (loss2_duration = 10)
  (fill_rate = 40000)
  (fill_duration = 3) : 
  (max_capacity - 
   ((loss1_rate * loss1_duration) + (loss2_rate * loss2_duration) - (fill_rate * fill_duration))) = 140000 :=
  by 
  unfold max_capacity loss1_rate loss1_duration loss2_rate loss2_duration fill_rate fill_duration
  sorry

end tank_missing_water_l218_218923


namespace prism_volume_l218_218151

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end prism_volume_l218_218151


namespace car_value_decrease_l218_218666

theorem car_value_decrease (original_price : ℝ) (decrease_percent : ℝ) (current_value : ℝ) :
  original_price = 4000 → decrease_percent = 0.30 → current_value = original_price * (1 - decrease_percent) → current_value = 2800 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end car_value_decrease_l218_218666


namespace P_div_by_Q_iff_l218_218531

def P (x : ℂ) (n : ℕ) : ℂ := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) : ℂ := x^4 + x^3 + x^2 + x + 1

theorem P_div_by_Q_iff (n : ℕ) : (Q x ∣ P x n) ↔ ¬(5 ∣ n) := sorry

end P_div_by_Q_iff_l218_218531


namespace train_length_250_05_l218_218392

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  speed_m_s * time_s

theorem train_length_250_05 : length_of_train 60 15 = 250.05 :=
by
  -- Definitions from the problem
  let speed_km_hr := 60
  let time_s := 15
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let distance := speed_m_s * time_s
  -- The proven assertion
  show distance = 250.05
  sorry

end train_length_250_05_l218_218392


namespace vasya_purchase_l218_218208

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218208


namespace div_by_64_l218_218682

theorem div_by_64 (n : ℕ) (h : n ≥ 1) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end div_by_64_l218_218682


namespace new_average_production_l218_218428

theorem new_average_production (n : ℕ) (daily_avg : ℕ) (today_prod : ℕ) (new_avg : ℕ) 
  (h1 : daily_avg = 50) 
  (h2 : today_prod = 95) 
  (h3 : n = 8) 
  (h4 : new_avg = (daily_avg * n + today_prod) / (n + 1)) : 
  new_avg = 55 := 
sorry

end new_average_production_l218_218428


namespace scientist_born_on_saturday_l218_218485

noncomputable def day_of_week := List String

noncomputable def calculate_day := 
  let days_in_regular_years := 113
  let days_in_leap_years := 2 * 37
  let total_days_back := days_in_regular_years + days_in_leap_years
  total_days_back % 7

theorem scientist_born_on_saturday :
  let anniversary_day := 4  -- 0=Sunday, 1=Monday, ..., 4=Thursday
  calculate_day = 5 → 
  let birth_day := (anniversary_day + 7 - calculate_day) % 7 
  birth_day = 6 := sorry

end scientist_born_on_saturday_l218_218485


namespace count_three_digit_congruent_to_5_mod_7_l218_218633

theorem count_three_digit_congruent_to_5_mod_7 : 
  (100 ≤ 7 * k + 5 ∧ 7 * k + 5 ≤ 999) → ∃ n : ℕ, n = 129 := sorry

end count_three_digit_congruent_to_5_mod_7_l218_218633


namespace div_eq_frac_l218_218405

theorem div_eq_frac : 250 / (5 + 12 * 3^2) = 250 / 113 :=
by
  sorry

end div_eq_frac_l218_218405


namespace first_term_value_l218_218903

noncomputable def find_first_term (a r : ℝ) := a / (1 - r) = 27 ∧ a^2 / (1 - r^2) = 108

theorem first_term_value :
  ∃ (a r : ℝ), find_first_term a r ∧ a = 216 / 31 :=
by
  sorry

end first_term_value_l218_218903


namespace expand_expression_l218_218029

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l218_218029


namespace cryptarithm_no_solution_proof_l218_218878

def cryptarithm_no_solution : Prop :=
  ∀ (D O N K A L E V G R : ℕ),
    D ≠ O ∧ D ≠ N ∧ D ≠ K ∧ D ≠ A ∧ D ≠ L ∧ D ≠ E ∧ D ≠ V ∧ D ≠ G ∧ D ≠ R ∧
    O ≠ N ∧ O ≠ K ∧ O ≠ A ∧ O ≠ L ∧ O ≠ E ∧ O ≠ V ∧ O ≠ G ∧ O ≠ R ∧
    N ≠ K ∧ N ≠ A ∧ N ≠ L ∧ N ≠ E ∧ N ≠ V ∧ N ≠ G ∧ N ≠ R ∧
    K ≠ A ∧ K ≠ L ∧ K ≠ E ∧ K ≠ V ∧ K ≠ G ∧ K ≠ R ∧
    A ≠ L ∧ A ≠ E ∧ A ≠ V ∧ A ≠ G ∧ A ≠ R ∧
    L ≠ E ∧ L ≠ V ∧ L ≠ G ∧ L ≠ R ∧
    E ≠ V ∧ E ≠ G ∧ E ≠ R ∧
    V ≠ G ∧ V ≠ R ∧
    G ≠ R ∧
    (D * 100 + O * 10 + N) + (O * 100 + K * 10 + A) +
    (L * 1000 + E * 100 + N * 10 + A) + (V * 10000 + O * 1000 + L * 100 + G * 10 + A) =
    A * 100000 + N * 10000 + G * 1000 + A * 100 + R * 10 + A →
    false

theorem cryptarithm_no_solution_proof : cryptarithm_no_solution :=
by sorry

end cryptarithm_no_solution_proof_l218_218878


namespace floor_sqrt_50_l218_218591

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l218_218591


namespace average_scissors_correct_l218_218661

-- Definitions for the initial number of scissors in each drawer
def initial_scissors_first_drawer : ℕ := 39
def initial_scissors_second_drawer : ℕ := 27
def initial_scissors_third_drawer : ℕ := 45

-- Definitions for the new scissors added by Dan
def added_scissors_first_drawer : ℕ := 13
def added_scissors_second_drawer : ℕ := 7
def added_scissors_third_drawer : ℕ := 10

-- Calculate the final number of scissors after Dan's addition
def final_scissors_first_drawer : ℕ := initial_scissors_first_drawer + added_scissors_first_drawer
def final_scissors_second_drawer : ℕ := initial_scissors_second_drawer + added_scissors_second_drawer
def final_scissors_third_drawer : ℕ := initial_scissors_third_drawer + added_scissors_third_drawer

-- Statement to prove the average number of scissors in all three drawers
theorem average_scissors_correct :
  (final_scissors_first_drawer + final_scissors_second_drawer + final_scissors_third_drawer) / 3 = 47 := by
  sorry

end average_scissors_correct_l218_218661


namespace power_of_three_l218_218810

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218810


namespace sequence_value_l218_218949

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end sequence_value_l218_218949


namespace pyramid_volume_pyramid_surface_area_l218_218486

noncomputable def volume_of_pyramid (l : ℝ) := (l^3 * Real.sqrt 2) / 12

noncomputable def surface_area_of_pyramid (l : ℝ) := (l^2 * (2 + Real.sqrt 2)) / 2

theorem pyramid_volume (l : ℝ) :
  volume_of_pyramid l = (l^3 * Real.sqrt 2) / 12 :=
sorry

theorem pyramid_surface_area (l : ℝ) :
  surface_area_of_pyramid l = (l^2 * (2 + Real.sqrt 2)) / 2 :=
sorry

end pyramid_volume_pyramid_surface_area_l218_218486


namespace apples_left_l218_218959

theorem apples_left (apples_on_tree apples_on_ground apples_eaten : ℕ)
    (h1 : apples_on_tree = 5)
    (h2 : apples_on_ground = 8)
    (h3 : apples_eaten = 3) :
    apples_on_tree + apples_on_ground - apples_eaten = 10 :=
by
    rw [h1, h2, h3] -- rewrite using the conditions
    sorry -- proof goes here

end apples_left_l218_218959


namespace probability_of_B_winning_is_correct_l218_218159

noncomputable def prob_A_wins : ℝ := 0.2
noncomputable def prob_draw : ℝ := 0.5
noncomputable def prob_B_wins : ℝ := 1 - (prob_A_wins + prob_draw)

theorem probability_of_B_winning_is_correct : prob_B_wins = 0.3 := by
  sorry

end probability_of_B_winning_is_correct_l218_218159


namespace floor_sqrt_50_l218_218594

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l218_218594


namespace slowest_pipe_time_l218_218098

noncomputable def fill_tank_rate (R : ℝ) : Prop :=
  let rate1 := 6 * R
  let rate3 := 2 * R
  let combined_rate := 9 * R
  combined_rate = 1 / 30

theorem slowest_pipe_time (R : ℝ) (h : fill_tank_rate R) : 1 / R = 270 :=
by
  have h1 := h
  sorry

end slowest_pipe_time_l218_218098


namespace dave_guitar_strings_l218_218408

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end dave_guitar_strings_l218_218408


namespace measure_of_smaller_angle_l218_218512

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l218_218512


namespace vasya_days_without_purchase_l218_218183

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218183


namespace total_pairs_purchased_l218_218432

-- Define the conditions as hypotheses
def foxPrice : ℝ := 15
def ponyPrice : ℝ := 18
def totalSaved : ℝ := 8.91
def foxPairs : ℕ := 3
def ponyPairs : ℕ := 2
def sumDiscountRates : ℝ := 0.22
def ponyDiscountRate : ℝ := 0.10999999999999996

-- Prove that the total number of pairs of jeans purchased is 5
theorem total_pairs_purchased : foxPairs + ponyPairs = 5 := by
  sorry

end total_pairs_purchased_l218_218432


namespace benny_turnips_l218_218867

-- Definitions and conditions
def melanie_turnips : ℕ := 139
def total_turnips : ℕ := 252

-- Question to prove
theorem benny_turnips : ∃ b : ℕ, b = total_turnips - melanie_turnips ∧ b = 113 :=
by {
    sorry
}

end benny_turnips_l218_218867


namespace exponent_power_identity_l218_218788

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218788


namespace marks_in_english_l218_218022

theorem marks_in_english :
  let m := 35             -- Marks in Mathematics
  let p := 52             -- Marks in Physics
  let c := 47             -- Marks in Chemistry
  let b := 55             -- Marks in Biology
  let n := 5              -- Number of subjects
  let avg := 46.8         -- Average marks
  let total_marks := avg * n
  total_marks - (m + p + c + b) = 45 := sorry

end marks_in_english_l218_218022


namespace power_calculation_l218_218832

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218832


namespace total_money_together_l218_218563

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l218_218563


namespace power_equality_l218_218847

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218847


namespace children_count_l218_218373

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l218_218373


namespace sin_cos_pi_over_12_l218_218596

theorem sin_cos_pi_over_12 :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
sorry

end sin_cos_pi_over_12_l218_218596


namespace volume_remaining_cube_l218_218227

theorem volume_remaining_cube (a : ℝ) (original_volume vertex_cube_volume : ℝ) (number_of_vertices : ℕ) :
  original_volume = a^3 → 
  vertex_cube_volume = 1 → 
  number_of_vertices = 8 → 
  a = 3 →
  original_volume - (number_of_vertices * vertex_cube_volume) = 19 := 
by
  sorry

end volume_remaining_cube_l218_218227


namespace Vanya_correct_answers_l218_218108

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l218_218108


namespace cost_of_replaced_tomatoes_l218_218634

def original_order : ℝ := 25
def delivery_tip : ℝ := 8
def new_total : ℝ := 35
def original_tomatoes : ℝ := 0.99
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00

def increase_in_lettuce := new_lettuce - original_lettuce
def increase_in_celery := new_celery - original_celery
def total_increase_except_tomatoes := increase_in_lettuce + increase_in_celery
def original_total_with_delivery := original_order + delivery_tip
def total_increase := new_total - original_total_with_delivery
def increase_due_to_tomatoes := total_increase - total_increase_except_tomatoes
def replaced_tomatoes := original_tomatoes + increase_due_to_tomatoes

theorem cost_of_replaced_tomatoes : replaced_tomatoes = 2.20 := by
  sorry

end cost_of_replaced_tomatoes_l218_218634


namespace num_men_in_second_group_l218_218055

def total_work_hours_week (men: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  men * hours_per_day * days_per_week

def earnings_per_man_hour (total_earnings: ℕ) (total_work_hours: ℕ) : ℚ :=
  total_earnings / total_work_hours

def required_man_hours (total_earnings: ℕ) (earnings_per_hour: ℚ) : ℚ :=
  total_earnings / earnings_per_hour

def number_of_men (total_man_hours: ℚ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℚ :=
  total_man_hours / (hours_per_day * days_per_week)

theorem num_men_in_second_group :
  let hours_per_day_1 := 10
  let hours_per_day_2 := 6
  let days_per_week := 7
  let men_1 := 4
  let earnings_1 := 1000
  let earnings_2 := 1350
  let work_hours_1 := total_work_hours_week men_1 hours_per_day_1 days_per_week
  let rate_1 := earnings_per_man_hour earnings_1 work_hours_1
  let work_hours_2 := required_man_hours earnings_2 rate_1
  number_of_men work_hours_2 hours_per_day_2 days_per_week = 9 := by
  sorry

end num_men_in_second_group_l218_218055


namespace cube_volume_ratio_l218_218320

theorem cube_volume_ratio (a b : ℝ) (h : (a^2 / b^2) = 9 / 25) :
  (b^3 / a^3) = 125 / 27 :=
by
  sorry

end cube_volume_ratio_l218_218320


namespace total_fat_l218_218384

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end total_fat_l218_218384


namespace cos_identity_l218_218979

theorem cos_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_identity_l218_218979


namespace vasya_no_purchase_days_l218_218168

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218168


namespace power_of_3_l218_218819

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218819


namespace alice_favorite_number_l218_218937

-- Define the conditions for Alice's favorite number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

-- Define the problem statement
theorem alice_favorite_number :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
           n % 13 = 0 ∧
           n % 3 ≠ 0 ∧
           sum_of_digits n % 4 = 0 ∧
           n = 130 :=
by
  sorry

end alice_favorite_number_l218_218937


namespace find_b_fixed_point_extremum_l218_218981

theorem find_b_fixed_point_extremum (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, f x = x ^ 3 + b * x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ (∀ x : ℝ, deriv f x₀ = 3 * x₀ ^ 2 + b) ∧ deriv f x₀ = 0) →
  b = -3 :=
by
  sorry

end find_b_fixed_point_extremum_l218_218981


namespace area_lt_perimeter_probability_l218_218000

theorem area_lt_perimeter_probability :
  (∃ s : ℕ, s ≥ 2 ∧ s ≤ 12 ∧ s * (s - 4) < 0) ∧ (1/36 + 1/18 = 1/12) :=
begin
  -- defines the probability space of rolling a pair of 6-sided dice
  let diceProb := pmf.of_finset {(i, j) | i ∈ finset.range 1 6 ∧ j ∈ finset.range 1 6} sorry,
  -- defines the event that the side length of the square (sum of dice) is such that the area < perimeter
  let event := {s | s ∈ finset.range 2 4},
  -- calculates the probability of the event
  have h : ∑ s in event, diceProb (λ (p : ℕ × ℕ), p.1 + p.2 = s) = 1 / 12, {
    sorry
  },
  -- proves the final equality of probabilities
  exact h,
end

end area_lt_perimeter_probability_l218_218000


namespace prime_large_factor_l218_218762

theorem prime_large_factor (p : ℕ) (hp : Nat.Prime p) (hp_ge_3 : p ≥ 3) (x : ℕ) (hx_large : ∃ N, x ≥ N) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ (p + 3) / 2 ∧ (∃ q : ℕ, Nat.Prime q ∧ q > p ∧ q ∣ (x + i)) := by
  sorry

end prime_large_factor_l218_218762


namespace power_of_3_l218_218825

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l218_218825


namespace vasya_did_not_buy_anything_days_l218_218219

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l218_218219


namespace triangle_side_AC_l218_218287

theorem triangle_side_AC 
  (AB BC : ℝ)
  (angle_C : ℝ)
  (h1 : AB = Real.sqrt 13)
  (h2 : BC = 3)
  (h3 : angle_C = Real.pi / 3) :
  ∃ AC : ℝ, AC = 4 :=
by 
  sorry

end triangle_side_AC_l218_218287


namespace total_cost_for_new_seats_l218_218930

-- Define the conditions
def seat_cost : ℕ := 30
def rows : ℕ := 5
def seats_per_row : ℕ := 8
def discount_percentage : ℕ := 10

-- Define the proof statement based on the conditions
theorem total_cost_for_new_seats:
  let total_seats := rows * seats_per_row in
  let cost_per_ten_seats := 10 * seat_cost in
  let discount_per_ten_seats := cost_per_ten_seats * discount_percentage / 100 in
  let discounted_cost_per_ten_seats := cost_per_ten_seats - discount_per_ten_seats in
  let num_sets_of_ten := total_seats / 10 in
  let total_cost := num_sets_of_ten * discounted_cost_per_ten_seats in
  total_cost = 1080 :=
by
  sorry

end total_cost_for_new_seats_l218_218930


namespace exponent_power_identity_l218_218794

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218794


namespace least_number_l218_218955

theorem least_number (n p q r s : ℕ) : 
  (n + p) % 24 = 0 ∧ 
  (n + q) % 32 = 0 ∧ 
  (n + r) % 36 = 0 ∧
  (n + s) % 54 = 0 →
  n = 863 :=
sorry

end least_number_l218_218955


namespace exponent_power_identity_l218_218790

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218790


namespace no_solution_set_1_2_4_l218_218961

theorem no_solution_set_1_2_4 
  (f : ℝ → ℝ) 
  (hf : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c)
  (t : ℝ) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (|x1 - t|) = 0 ∧ f (|x2 - t|) = 0 ∧ f (|x3 - t|) = 0 ∧ (x1 = 1 ∧ x2 = 2 ∧ x3 = 4) := 
sorry

end no_solution_set_1_2_4_l218_218961


namespace polynomial_division_l218_218430

noncomputable def poly1 : Polynomial ℤ := Polynomial.X ^ 13 - Polynomial.X + 100
noncomputable def poly2 : Polynomial ℤ := Polynomial.X ^ 2 + Polynomial.X + 2

theorem polynomial_division : ∃ q : Polynomial ℤ, poly1 = poly2 * q :=
by 
  sorry

end polynomial_division_l218_218430


namespace polynomial_min_value_l218_218707

noncomputable def poly (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

theorem polynomial_min_value : 
  ∃ x y : ℝ, poly x y = -18 :=
by
  sorry

end polynomial_min_value_l218_218707


namespace negative_number_among_options_l218_218520

theorem negative_number_among_options :
  let A := |(-2 : ℤ)|
      B := real.sqrt 3
      C := (0 : ℤ)
      D := (-5 : ℤ)
  in D = -5 := 
by 
  sorry

end negative_number_among_options_l218_218520


namespace cookies_per_child_l218_218448

theorem cookies_per_child 
  (total_cookies : ℕ) 
  (children : ℕ) 
  (x : ℚ) 
  (adults_fraction : total_cookies * x = total_cookies / 4) 
  (remaining_cookies : total_cookies - total_cookies * x = 180) 
  (correct_fraction : x = 1 / 4) 
  (correct_children : children = 6) :
  (total_cookies - total_cookies * x) / children = 30 := by
  sorry

end cookies_per_child_l218_218448


namespace right_triangle_third_side_square_l218_218625

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) 
  (h₁ : a = 3) (h₂ : b = 4) (h₃ : a^2 + b^2 = c^2) :
  c^2 = 25 ∨ a^2 + c^2 = b^2 ∨ a^2 + b^2 = 7 :=
by
  sorry

end right_triangle_third_side_square_l218_218625


namespace money_together_l218_218560

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l218_218560


namespace total_fish_at_wedding_l218_218487

def num_tables : ℕ := 32
def fish_per_table_except_one : ℕ := 2
def fish_on_special_table : ℕ := 3
def number_of_special_tables : ℕ := 1
def number_of_regular_tables : ℕ := num_tables - number_of_special_tables

theorem total_fish_at_wedding : 
  (number_of_regular_tables * fish_per_table_except_one) + (number_of_special_tables * fish_on_special_table) = 65 :=
by
  sorry

end total_fish_at_wedding_l218_218487


namespace typing_speed_ratio_l218_218915

theorem typing_speed_ratio (T t : ℝ) (h1 : T + t = 12) (h2 : T + 1.25 * t = 14) : t / T = 2 :=
by
  sorry

end typing_speed_ratio_l218_218915


namespace total_earnings_l218_218572

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end total_earnings_l218_218572


namespace range_of_a_l218_218440

theorem range_of_a (M N : Set ℝ) (a : ℝ) 
(hM : M = {x : ℝ | x < 2}) 
(hN : N = {x : ℝ | x < a}) 
(hSubset : M ⊆ N) : 
  2 ≤ a := 
sorry

end range_of_a_l218_218440


namespace multiple_proof_l218_218232

noncomputable def K := 185  -- Given KJ's stamps
noncomputable def AJ := 370  -- Given AJ's stamps
noncomputable def total_stamps := 930  -- Given total amount

-- Using the conditions to find C
noncomputable def stamps_of_three := AJ + K  -- Total stamps of KJ and AJ
noncomputable def C := total_stamps - stamps_of_three

-- Stating the equivalence we need to prove
theorem multiple_proof : ∃ M: ℕ, M * K + 5 = C := by
  -- The solution proof here if required
  existsi 2
  sorry  -- proof to be completed

end multiple_proof_l218_218232


namespace freeze_time_l218_218662

theorem freeze_time :
  ∀ (minutes_per_smoothie total_minutes num_smoothies freeze_time: ℕ),
    minutes_per_smoothie = 3 →
    total_minutes = 55 →
    num_smoothies = 5 →
    freeze_time = total_minutes - (num_smoothies * minutes_per_smoothie) →
    freeze_time = 40 :=
by
  intros minutes_per_smoothie total_minutes num_smoothies freeze_time
  intros H1 H2 H3 H4
  subst H1
  subst H2
  subst H3
  subst H4
  sorry

end freeze_time_l218_218662


namespace order_of_6_proof_l218_218695

noncomputable def f (x : ℕ) := x^2 % 13

def order_of_6 : ℕ := 36

theorem order_of_6_proof : (∃ n, n > 0 ∧ f^[n] 6 = 6 ∧ (∀ m < n, m > 0 → f^[m] 6 ≠ 6)) ∧ (order_of_6 = 36) :=
begin
  use 36,
  split,
  { split,
    { norm_num, },
    { split,
      { norm_num, },
      { intros m hm1 hm2,
        sorry, } } },
  norm_num,
end

end order_of_6_proof_l218_218695


namespace notebook_cost_correct_l218_218240

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l218_218240


namespace vanya_correct_answers_l218_218116

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218116


namespace exponent_power_identity_l218_218796

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218796


namespace probability_exactly_one_common_number_l218_218066

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l218_218066


namespace angle_subtraction_correct_polynomial_simplification_correct_l218_218570

noncomputable def angleSubtraction : Prop :=
  let a1 := 34 * 60 + 26 -- Convert 34°26' to total minutes
  let a2 := 25 * 60 + 33 -- Convert 25°33' to total minutes
  let diff := a1 - a2 -- Subtract in minutes
  let degrees := diff / 60 -- Convert back to degrees
  let minutes := diff % 60 -- Remainder in minutes
  degrees = 8 ∧ minutes = 53 -- Expected result in degrees and minutes

noncomputable def polynomialSimplification (m : Int) : Prop :=
  let expr := 5 * m^2 - (m^2 - 6 * m) - 2 * (-m + 3 * m^2)
  expr = -2 * m^2 + 8 * m -- Simplified form

-- Statements needing proof
theorem angle_subtraction_correct : angleSubtraction := by
  sorry

theorem polynomial_simplification_correct (m : Int) : polynomialSimplification m := by
  sorry

end angle_subtraction_correct_polynomial_simplification_correct_l218_218570


namespace floor_sqrt_50_l218_218587

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l218_218587


namespace sequence_sum_l218_218946

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end sequence_sum_l218_218946


namespace max_value_a4_a6_l218_218145

theorem max_value_a4_a6 (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6) :
  ∃ m, ∀ (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6), a 4 * a 6 ≤ m :=
sorry

end max_value_a4_a6_l218_218145


namespace power_addition_l218_218803

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218803


namespace value_x_plus_2y_plus_3z_l218_218043

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end value_x_plus_2y_plus_3z_l218_218043


namespace frog_eyes_in_pond_l218_218012

-- Definitions based on conditions
def num_frogs : ℕ := 6
def eyes_per_frog : ℕ := 2

-- The property to be proved
theorem frog_eyes_in_pond : num_frogs * eyes_per_frog = 12 :=
by
  sorry

end frog_eyes_in_pond_l218_218012


namespace g_10_44_l218_218241

def g (x y : ℕ) : ℕ := sorry

axiom g_cond1 (x : ℕ) : g x x = x ^ 2
axiom g_cond2 (x y : ℕ) : g x y = g y x
axiom g_cond3 (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_10_44 : g 10 44 = 440 := sorry

end g_10_44_l218_218241


namespace seating_arrangement_l218_218006

def numWaysCableCars (adults children cars capacity : ℕ) : ℕ := 
  sorry 

theorem seating_arrangement :
  numWaysCableCars 4 2 3 3 = 348 :=
by {
  sorry
}

end seating_arrangement_l218_218006


namespace solve_abs_eq_l218_218877

theorem solve_abs_eq (x : ℝ) : (|x + 4| = 3 - x) → (x = -1/2) := by
  intro h
  sorry

end solve_abs_eq_l218_218877


namespace vasya_days_without_purchases_l218_218190

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218190


namespace simplify_radical_product_l218_218911

theorem simplify_radical_product : 
  (32^(1/5)) * (8^(1/3)) * (4^(1/2)) = 8 := 
by
  sorry

end simplify_radical_product_l218_218911


namespace find_N_l218_218280

theorem find_N (N x : ℝ) (h1 : N / (1 + 4 / x) = 1) (h2 : x = 0.5) : N = 9 := 
by 
  sorry

end find_N_l218_218280


namespace vasya_days_l218_218179

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218179


namespace square_area_parabola_inscribed_l218_218553

theorem square_area_parabola_inscribed (s : ℝ) (x y : ℝ) :
  (y = x^2 - 6 * x + 8) ∧
  (s = -2 + 2 * Real.sqrt 5) ∧
  (x = 3 - s / 2 ∨ x = 3 + s / 2) →
  s ^ 2 = 24 - 8 * Real.sqrt 5 :=
by
  sorry

end square_area_parabola_inscribed_l218_218553


namespace measure_of_smaller_angle_l218_218510

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l218_218510


namespace prism_volume_l218_218149

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end prism_volume_l218_218149


namespace oldest_person_Jane_babysat_age_l218_218663

def Jane_current_age : ℕ := 32
def Jane_stop_babysitting_age : ℕ := 22 -- 32 - 10
def max_child_age_when_Jane_babysat : ℕ := Jane_stop_babysitting_age / 2  -- 22 / 2
def years_since_Jane_stopped : ℕ := Jane_current_age - Jane_stop_babysitting_age -- 32 - 22

theorem oldest_person_Jane_babysat_age :
  max_child_age_when_Jane_babysat + years_since_Jane_stopped = 21 :=
by
  sorry

end oldest_person_Jane_babysat_age_l218_218663


namespace bottles_drunk_l218_218679

theorem bottles_drunk (initial_bottles remaining_bottles : ℕ)
  (h₀ : initial_bottles = 17) (h₁ : remaining_bottles = 14) :
  initial_bottles - remaining_bottles = 3 :=
sorry

end bottles_drunk_l218_218679


namespace power_of_three_l218_218783

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218783


namespace factor_quadratic_polynomial_l218_218248

theorem factor_quadratic_polynomial :
  (∀ x : ℝ, x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5)) :=
by
  sorry

end factor_quadratic_polynomial_l218_218248


namespace proof_problem_l218_218730

theorem proof_problem (s t: ℤ) (h : 514 - s = 600 - t) : s < t ∧ t - s = 86 :=
by
  sorry

end proof_problem_l218_218730


namespace royal_children_count_l218_218363

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l218_218363


namespace sum_of_intercepts_of_line_l218_218902

theorem sum_of_intercepts_of_line (x y : ℝ) (hx : 2 * x - 3 * y + 6 = 0) :
  2 + (-3) = -1 :=
sorry

end sum_of_intercepts_of_line_l218_218902


namespace vasya_days_without_purchase_l218_218184

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218184


namespace lowest_fraction_job_done_in_1_hour_l218_218918

theorem lowest_fraction_job_done_in_1_hour (hA : 4 > 0) (hB : 5 > 0) (hC : 8 > 0) :
  let rateA := (1 : ℚ) / 4
  let rateB := (1 : ℚ) / 5
  let rateC := (1 : ℚ) / 8
  let combineTwoSlowest := rateB + rateC
  combineTwoSlowest = 13 / 40 :=
by
  let rateA := (1 : ℚ) / 4
  let rateB := (1 : ℚ) / 5
  let rateC := (1 : ℚ) / 8
  let combineTwoSlowest := rateB + rateC
  have : combineTwoSlowest = (1 / 5) + (1 / 8) := rfl
  have : combineTwoSlowest = (8 / 40) + (5 / 40) := by rw [this, div_eq_mul_one_div, div_eq_mul_one_div]; ratea
  have : combineTwoSlowest = 13 / 40 := by norm_num [this]
  exact this

end lowest_fraction_job_done_in_1_hour_l218_218918


namespace fewer_white_chairs_than_green_blue_l218_218082

-- Definitions of the conditions
def blue_chairs : ℕ := 10
def green_chairs : ℕ := 3 * blue_chairs
def total_chairs : ℕ := 67
def green_blue_chairs : ℕ := green_chairs + blue_chairs
def white_chairs : ℕ := total_chairs - green_blue_chairs

-- Statement of the theorem
theorem fewer_white_chairs_than_green_blue : green_blue_chairs - white_chairs = 13 :=
by
  -- This is where the proof would go, but we're omitting it as per instruction
  sorry

end fewer_white_chairs_than_green_blue_l218_218082


namespace common_elements_count_287_l218_218861

def set_U (n : ℕ) : Set ℕ := {k | ∃ i, ↑k = 5 * i ∧ i ≤ n}
def set_V (n : ℕ) : Set ℕ := {k | ∃ i, ↑k = 7 * i ∧ i ≤ n}
def lcm (a b : ℕ) := (a * b) / Nat.gcd a b

theorem common_elements_count_287 :
  let U := (set_U 2010)
  let V := (set_V 2010)
  (U ∩ V).to_finset.card = 287 := by
  sorry

end common_elements_count_287_l218_218861


namespace total_point_value_of_test_l218_218163

theorem total_point_value_of_test (total_questions : ℕ) (five_point_questions : ℕ) 
  (ten_point_questions : ℕ) (points_5 : ℕ) (points_10 : ℕ) 
  (h1 : total_questions = 30) (h2 : five_point_questions = 20) 
  (h3 : ten_point_questions = total_questions - five_point_questions) 
  (h4 : points_5 = 5) (h5 : points_10 = 10) : 
  five_point_questions * points_5 + ten_point_questions * points_10 = 200 :=
by
  sorry

end total_point_value_of_test_l218_218163


namespace power_of_x_is_one_l218_218146

-- The problem setup, defining the existence of distinct primes and conditions on exponents
theorem power_of_x_is_one (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (a b c : ℕ) (h_divisors : (a + 1) * (b + 1) * (c + 1) = 12) :
  a = 1 :=
sorry

end power_of_x_is_one_l218_218146


namespace vasya_purchase_l218_218212

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l218_218212


namespace find_a_if_odd_function_l218_218982

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + Real.sqrt (a + x^2))

theorem find_a_if_odd_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - f x a) → a = 1 :=
by
  sorry

end find_a_if_odd_function_l218_218982


namespace vasya_days_without_purchases_l218_218194

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l218_218194


namespace runway_show_time_correct_l218_218311

def runwayShowTime (bathing_suit_sets evening_wear_sets formal_wear_sets models trip_time_in_minutes : ℕ) : ℕ :=
  let trips_per_model := bathing_suit_sets + evening_wear_sets + formal_wear_sets
  let total_trips := models * trips_per_model
  total_trips * trip_time_in_minutes

theorem runway_show_time_correct :
  runwayShowTime 3 4 2 10 3 = 270 :=
by
  sorry

end runway_show_time_correct_l218_218311


namespace inequality_holds_for_all_x_l218_218431

variable (p : ℝ)
variable (x : ℝ)

theorem inequality_holds_for_all_x (h : -3 < p ∧ p < 6) : 
  -9 < (3*x^2 + p*x - 6) / (x^2 - x + 1) ∧ (3*x^2 + p*x - 6) / (x^2 - x + 1) < 6 := by
  sorry

end inequality_holds_for_all_x_l218_218431


namespace determine_m_ratio_l218_218744

def ratio_of_C_to_A_investment (x : ℕ) (m : ℕ) (total_gain : ℕ) (a_share : ℕ) : Prop :=
  total_gain = 18000 ∧ a_share = 6000 ∧
  (12 * x / (12 * x + 4 * m * x) = 1 / 3)

theorem determine_m_ratio (x : ℕ) (m : ℕ) (h : ratio_of_C_to_A_investment x m 18000 6000) :
  m = 6 :=
by
  sorry

end determine_m_ratio_l218_218744


namespace circle_equation_l218_218623

/-- Given a circle passing through points P(4, -2) and Q(-1, 3), and with the length of the segment 
intercepted by the circle on the y-axis as 4, prove that the standard equation of the circle
is (x-1)^2 + y^2 = 13 or (x-5)^2 + (y-4)^2 = 37 -/
theorem circle_equation {P Q : ℝ × ℝ} {a b k : ℝ} :
  P = (4, -2) ∧ Q = (-1, 3) ∧ k = 4 →
  (∃ (r : ℝ), (∀ y : ℝ, (b - y)^2 = r^2) ∧
    ((a - 1)^2 + b^2 = 13 ∨ (a - 5)^2 + (b - 4)^2 = 37)
  ) :=
by
  sorry

end circle_equation_l218_218623


namespace missing_water_calculation_l218_218924

def max_capacity : ℝ := 350000
def loss_rate1 : ℝ := 32000
def time1 : ℝ := 5
def loss_rate2 : ℝ := 10000
def time2 : ℝ := 10
def fill_rate : ℝ := 40000
def fill_time : ℝ := 3

theorem missing_water_calculation :
  350000 - ((350000 - (32000 * 5 + 10000 * 10)) + 40000 * 3) = 140000 :=
by
  sorry

end missing_water_calculation_l218_218924


namespace equal_numbers_product_l218_218131

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l218_218131


namespace ice_cream_flavors_l218_218978

theorem ice_cream_flavors (n k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n + k - 1).choose (k - 1) = 84 :=
by
  have h3 : n = 6 := h1
  have h4 : k = 4 := h2
  rw [h3, h4]
  sorry

end ice_cream_flavors_l218_218978


namespace differential_savings_is_4830_l218_218534

-- Defining the conditions
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.28
def annual_income : ℝ := 34500

-- Defining the calculation of tax before and after the tax rate change
def tax_before : ℝ := annual_income * initial_tax_rate
def tax_after : ℝ := annual_income * new_tax_rate

-- Defining the differential savings
def differential_savings : ℝ := tax_before - tax_after

-- Statement asserting that the differential savings is $4830
theorem differential_savings_is_4830 : differential_savings = 4830 := by sorry

end differential_savings_is_4830_l218_218534


namespace sum_of_A_and_B_l218_218158

theorem sum_of_A_and_B (A B : ℕ) (h1 : (1 / 6 : ℚ) * (1 / 3) = 1 / (A * 3))
                       (h2 : (1 / 6 : ℚ) * (1 / 3) = 1 / B) : A + B = 24 :=
by
  sorry

end sum_of_A_and_B_l218_218158


namespace complex_quadrant_l218_218451

theorem complex_quadrant (x y: ℝ) (h : x = 1 ∧ y = 2) : x > 0 ∧ y > 0 :=
by
  sorry

end complex_quadrant_l218_218451


namespace find_total_amount_l218_218331

theorem find_total_amount (x : ℝ) (h₁ : 1.5 * x = 40) : x + 1.5 * x + 0.5 * x = 80.01 :=
by
  sorry

end find_total_amount_l218_218331


namespace find_expression_l218_218960

theorem find_expression (a b : ℝ) (h₁ : a - b = 5) (h₂ : a * b = 2) :
  a^2 - a * b + b^2 = 27 := 
by
  sorry

end find_expression_l218_218960


namespace ratio_of_areas_of_triangles_l218_218088

-- Define the given conditions
variables {X Y Z T : Type}
variable (distance_XY : ℝ)
variable (distance_XZ : ℝ)
variable (distance_YZ : ℝ)
variable (is_angle_bisector : Prop)

-- Define the correct answer as a goal
theorem ratio_of_areas_of_triangles (h1 : distance_XY = 15)
    (h2 : distance_XZ = 25)
    (h3 : distance_YZ = 34)
    (h4 : is_angle_bisector) : 
    -- Ratio of the areas of triangle XYT to triangle XZT
    ∃ (ratio : ℝ), ratio = 3 / 5 :=
by
  -- This is where the proof would go, omitted with "sorry"
  sorry

end ratio_of_areas_of_triangles_l218_218088


namespace ellipse_equation_range_of_M_x_coordinate_l218_218272

-- Proof 1: Proving the equation of the ellipse
theorem ellipse_equation {a b : ℝ} (h_ab : a > b) (h_b0 : b > 0) (e : ℝ)
  (h_e : e = (Real.sqrt 3) / 3) (vertex : ℝ × ℝ) (h_vertex : vertex = (Real.sqrt 3, 0)) :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧ vertex = (Real.sqrt 3, 0) ∧ (∀ (x y : ℝ), (x^2) / 3 + (y^2) / 2 = 1)) :=
sorry

-- Proof 2: Proving the range of x-coordinate of point M
theorem range_of_M_x_coordinate (k : ℝ) (h_k : k ≠ 0) :
  (∃ M_x : ℝ, by sorry) :=
sorry


end ellipse_equation_range_of_M_x_coordinate_l218_218272


namespace no_pos_int_sol_l218_218873

theorem no_pos_int_sol (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ¬ ∃ (k : ℕ), (15 * a + b) * (a + 15 * b) = 3^k := 
sorry

end no_pos_int_sol_l218_218873


namespace find_k_l218_218706

theorem find_k 
    (x y k : ℝ)
    (h1 : 1.5 * x + y = 20)
    (h2 : -4 * x + y = k)
    (hx : x = -6) :
    k = 53 :=
by
  sorry

end find_k_l218_218706


namespace total_earnings_l218_218573

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end total_earnings_l218_218573


namespace range_of_a_l218_218629

-- Define the function f(x) and its condition
def f (x a : ℝ) : ℝ := x^2 + (a + 2) * x + (a - 1)

-- Given condition: f(-1, a) = -2
def condition (a : ℝ) : Prop := f (-1) a = -2

-- Requirement for the domain of g(x) = ln(f(x) + 3) being ℝ
def domain_requirement (a : ℝ) : Prop := ∀ x : ℝ, f x a + 3 > 0

-- Main theorem to prove the range of a
theorem range_of_a : {a : ℝ // condition a ∧ domain_requirement a} = {a : ℝ // -2 < a ∧ a < 2} :=
by sorry

end range_of_a_l218_218629


namespace total_coins_l218_218938

theorem total_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end total_coins_l218_218938


namespace negation_of_universal_proposition_l218_218893

def P (x : ℝ) : Prop := x^3 + 2 * x ≥ 0

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 0 ≤ x → P x) ↔ (∃ x : ℝ, 0 ≤ x ∧ ¬ P x) :=
by
  sorry

end negation_of_universal_proposition_l218_218893


namespace royal_children_count_l218_218361

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l218_218361


namespace toothpick_sequence_l218_218885

theorem toothpick_sequence (a d n : ℕ) (h1 : a = 6) (h2 : d = 4) (h3 : n = 150) : a + (n - 1) * d = 602 := by
  sorry

end toothpick_sequence_l218_218885


namespace masha_doll_arrangements_l218_218997

theorem masha_doll_arrangements : 
  let dolls := Fin 7
  let dollhouses := Fin 6 in
  (∃ a b : dolls, a ≠ b) ∧ 
  (∃ h : dollhouses, ∀ i ≠ h, ∃! d : dolls, d ≠ a ∧ d ≠ b) ∧ 
  ∃ h : dollhouses, ∃ a b : dolls, a ≠ b ∧ 
  (∀ (d1 d2 : dolls), d1 ≠ a ∧ d1 ≠ b ∧ d2 ≠ a ∧ d2 ≠ b → d1 ≠ d2) →
  21 * 6 * 120 = 15120 :=
by
  sorry

end masha_doll_arrangements_l218_218997


namespace train_length_l218_218007

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℚ) : 
  speed_kmh = 120 → 
  time_s = 25 → 
  length_m = 833.25 → 
  (speed_kmh * 1000 / 3600) * time_s = length_m :=
by
  intros
  sorry

end train_length_l218_218007


namespace problem_statement_l218_218645

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l218_218645


namespace vanya_correct_answers_l218_218111

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l218_218111


namespace train_length_correct_l218_218399

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l218_218399


namespace sum_floor_ceil_eq_seven_l218_218144

theorem sum_floor_ceil_eq_seven (x : ℝ) 
  (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 := 
sorry

end sum_floor_ceil_eq_seven_l218_218144


namespace no_such_P_exists_l218_218243

theorem no_such_P_exists (P : Polynomial ℤ) (r : ℕ) (r_ge_3 : r ≥ 3) (a : Fin r → ℤ)
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (P_cycle : ∀ i, P.eval (a i) = a ⟨(i + 1) % r, sorry⟩)
  : False :=
sorry

end no_such_P_exists_l218_218243


namespace one_minus_repeating_eight_l218_218032

-- Given the condition
def b : ℚ := 8 / 9

-- The proof problem statement
theorem one_minus_repeating_eight : 1 - b = 1 / 9 := 
by
  sorry  -- proof to be provided

end one_minus_repeating_eight_l218_218032


namespace height_of_pole_l218_218738

-- Definitions for the conditions
def ascends_first_minute := 2
def slips_second_minute := 1
def net_ascent_per_two_minutes := ascends_first_minute - slips_second_minute
def total_minutes := 17
def pairs_of_minutes := (total_minutes - 1) / 2  -- because the 17th minute is separate
def net_ascent_first_16_minutes := pairs_of_minutes * net_ascent_per_two_minutes

-- The final ascent in the 17th minute
def ascent_final_minute := 2

-- Total ascent
def total_ascent := net_ascent_first_16_minutes + ascent_final_minute

-- Statement to prove the height of the pole
theorem height_of_pole : total_ascent = 10 :=
by
  sorry

end height_of_pole_l218_218738


namespace vasya_days_without_purchase_l218_218186

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l218_218186


namespace most_likely_outcomes_l218_218957

noncomputable def probability_boy_or_girl : ℚ := 1 / 2

noncomputable def probability_all_boys (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def probability_all_girls (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_3_girls_2_boys : ℚ := binom 5 3 * probability_boy_or_girl^5

noncomputable def probability_3_boys_2_girls : ℚ := binom 5 2 * probability_boy_or_girl^5

theorem most_likely_outcomes :
  probability_3_girls_2_boys = 5/16 ∧
  probability_3_boys_2_girls = 5/16 ∧
  probability_all_boys 5 = 1/32 ∧
  probability_all_girls 5 = 1/32 ∧
  (5/16 > 1/32) :=
by
  sorry

end most_likely_outcomes_l218_218957


namespace square_side_length_increase_l218_218891

variables {a x : ℝ}

theorem square_side_length_increase 
  (h : (a * (1 + x / 100) * 1.8)^2 = (1 + 159.20000000000002 / 100) * (a^2 + (a * (1 + x / 100))^2)) : 
  x = 100 :=
by sorry

end square_side_length_increase_l218_218891


namespace total_amount_is_200_l218_218568

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l218_218568


namespace area_ratio_l218_218728

noncomputable def pentagon_area (R s : ℝ) := (5 / 2) * R * s * Real.sin (Real.pi * 2 / 5)
noncomputable def triangle_area (s : ℝ) := (s^2) / 4

theorem area_ratio (R s : ℝ) (h : R = s / (2 * Real.sin (Real.pi / 5))) :
  (pentagon_area R s) / (triangle_area s) = 5 * (Real.sin ((2 * Real.pi) / 5) / Real.sin (Real.pi / 5)) :=
by
  sorry

end area_ratio_l218_218728


namespace normal_price_of_article_l218_218325

theorem normal_price_of_article 
  (P : ℝ) 
  (h : (P * 0.88 * 0.78 * 0.85) * 1.06 = 144) : 
  P = 144 / (0.88 * 0.78 * 0.85 * 1.06) :=
sorry

end normal_price_of_article_l218_218325


namespace carl_weight_l218_218745

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end carl_weight_l218_218745


namespace power_of_three_l218_218808

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218808


namespace remainder_of_polynomial_division_is_88_l218_218758

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem remainder_of_polynomial_division_is_88 :
  p 2 = 88 :=
by
  sorry

end remainder_of_polynomial_division_is_88_l218_218758


namespace royal_children_l218_218336

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l218_218336


namespace no_snow_probability_l218_218496

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l218_218496


namespace vasya_no_purchase_days_l218_218169

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l218_218169


namespace total_area_of_hexagon_is_693_l218_218387

-- Conditions
def hexagon_side1_length := 3
def hexagon_side2_length := 2
def angle_between_length3_sides := 120
def all_internal_triangles_are_equilateral := true
def number_of_triangles := 6

-- Define the problem statement
theorem total_area_of_hexagon_is_693 
  (a1 : hexagon_side1_length = 3)
  (a2 : hexagon_side2_length = 2)
  (a3 : angle_between_length3_sides = 120)
  (a4 : all_internal_triangles_are_equilateral = true)
  (a5 : number_of_triangles = 6) :
  total_area_of_hexagon = 693 :=
by
  sorry

end total_area_of_hexagon_is_693_l218_218387


namespace geometric_sequence_product_bound_l218_218041

theorem geometric_sequence_product_bound {a1 a2 a3 m q : ℝ} (h_sum : a1 + a2 + a3 = 3 * m) (h_m_pos : 0 < m) (h_q_pos : 0 < q) (h_geom : a1 = a2 / q ∧ a3 = a2 * q) : 
  0 < a1 * a2 * a3 ∧ a1 * a2 * a3 ≤ m^3 := 
sorry

end geometric_sequence_product_bound_l218_218041


namespace rhombus_perimeter_l218_218716

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) 
  (h3 : ∀ {x y : ℝ}, (x = d1 / 2 ∧ y = d2 / 2) → (x^2 + y^2 = (d1 / 2)^2 + (d2 / 2)^2)) : 
  4 * (Real.sqrt ((d1/2)^2 + (d2/2)^2)) = 156 :=
by 
  rw [h1, h2]
  simp
  sorry

end rhombus_perimeter_l218_218716


namespace angle_measure_l218_218516

-- Define the angle in degrees
def angle (x : ℝ) : Prop :=
  180 - x = 3 * (90 - x)

-- Desired proof statement
theorem angle_measure :
  ∀ (x : ℝ), angle x → x = 45 := by
  intros x h
  sorry

end angle_measure_l218_218516


namespace correct_statements_l218_218259

theorem correct_statements (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ b, a = 1 - 2 * b → a^2 + b^2 ≥ 1/5) ∧
  (∀ a b, a + 2 * b = 1 → ab ≤ 1/8) ∧
  (∀ a b, a + 2 * b = 1 → 3 + 2 * Real.sqrt 2 ≤ (1 / a + 1 / b)) :=
by
  sorry

end correct_statements_l218_218259


namespace total_games_played_l218_218418

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end total_games_played_l218_218418


namespace fraction_meaningful_range_l218_218897

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_range_l218_218897


namespace min_value_2a_plus_b_value_of_t_l218_218258

theorem min_value_2a_plus_b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) :
  2 * a + b = 4 :=
sorry

theorem value_of_t (a b t : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) (h₄ : 4^a = t) (h₅ : 3^b = t) :
  t = 6 :=
sorry

end min_value_2a_plus_b_value_of_t_l218_218258


namespace negation_of_neither_even_l218_218540

variable (a b : Nat)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

theorem negation_of_neither_even 
  (H : ¬ (¬ is_even a ∧ ¬ is_even b)) : is_even a ∨ is_even b :=
sorry

end negation_of_neither_even_l218_218540


namespace select_rows_and_columns_l218_218435

theorem select_rows_and_columns (n : Nat) (pieces : Fin (2 * n) × Fin (2 * n) → Bool) :
  (∃ rows cols : Finset (Fin (2 * n)),
    rows.card = n ∧ cols.card = n ∧
    (∀ r c, r ∈ rows → c ∈ cols → pieces (r, c))) :=
sorry

end select_rows_and_columns_l218_218435


namespace gcd_153_119_l218_218140

theorem gcd_153_119 : Nat.gcd 153 119 = 17 :=
by
  sorry

end gcd_153_119_l218_218140


namespace minimum_value_a_l218_218968

noncomputable def f (a b x : ℝ) := a * Real.log x - (1 / 2) * x^2 + b * x

theorem minimum_value_a (h : ∀ b x : ℝ, x > 0 → f a b x > 0) : a ≥ -Real.exp 3 := 
sorry

end minimum_value_a_l218_218968


namespace max_floor_l218_218761

theorem max_floor (x : ℝ) (h : ⌊(x + 4) / 10⌋ = 5) : ⌊(6 * x) / 5⌋ = 67 :=
  sorry

end max_floor_l218_218761


namespace heloise_gives_dogs_to_janet_l218_218976

theorem heloise_gives_dogs_to_janet :
  ∃ d c : ℕ, d * 17 = c * 10 ∧ d + c = 189 ∧ d - 60 = 10 :=
by
  sorry

end heloise_gives_dogs_to_janet_l218_218976


namespace parabola_sum_l218_218550

def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - b * x + c

def f (a b c : ℝ) (x : ℝ) : ℝ := a * (x + 7) ^ 2 - b * (x + 7) + c

def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 3) ^ 2 - b * (x - 3) + c

def fg (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum (a b c x : ℝ) : fg a b c x = 2 * a * x ^ 2 + (8 * a - 2 * b) * x + (58 * a - 4 * b + 2 * c) := by
  sorry

end parabola_sum_l218_218550


namespace geometric_sequence_b_value_l218_218499

theorem geometric_sequence_b_value (b : ℝ) (r : ℝ) (h1 : 210 * r = b) (h2 : b * r = 35 / 36) (hb : b > 0) : 
  b = Real.sqrt (7350 / 36) :=
by
  sorry

end geometric_sequence_b_value_l218_218499


namespace gcd_of_36_between_70_and_85_is_81_l218_218704

theorem gcd_of_36_between_70_and_85_is_81 {n : ℕ} (h1 : n ≥ 70) (h2 : n ≤ 85) (h3 : Nat.gcd 36 n = 9) : n = 81 :=
by
  -- proof
  sorry

end gcd_of_36_between_70_and_85_is_81_l218_218704


namespace ratio_elephants_to_others_l218_218319

theorem ratio_elephants_to_others (L P E : ℕ) (h1 : L = 2 * P) (h2 : L = 200) (h3 : L + P + E = 450) :
  E / (L + P) = 1 / 2 :=
by
  sorry

end ratio_elephants_to_others_l218_218319


namespace probability_one_common_number_approx_l218_218061

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l218_218061


namespace sequence_formula_l218_218775

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 33) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 :=
by
  sorry

end sequence_formula_l218_218775


namespace width_to_length_ratio_l218_218855

variable (w : ℕ)

def length := 10
def perimeter := 36

theorem width_to_length_ratio
  (h_perimeter : 2 * w + 2 * length = perimeter) :
  w / length = 4 / 5 :=
by
  -- Skipping proof steps, putting sorry
  sorry

end width_to_length_ratio_l218_218855


namespace sum_of_roots_of_qubic_polynomial_l218_218546

noncomputable def Q (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_qubic_polynomial (a b c d : ℝ) 
  (h₁ : ∀ x : ℝ, Q a b c d (x^4 + x) ≥ Q a b c d (x^3 + 1))
  (h₂ : Q a b c d 1 = 0) : 
  -b / a = 3 / 2 :=
sorry

end sum_of_roots_of_qubic_polynomial_l218_218546


namespace phase_and_initial_phase_theorem_l218_218490

open Real

noncomputable def phase_and_initial_phase (x : ℝ) : ℝ := 3 * sin (-x + π / 6)

theorem phase_and_initial_phase_theorem :
  ∃ φ : ℝ, ∃ ψ : ℝ,
    ∀ x : ℝ, phase_and_initial_phase x = 3 * sin (x + φ) ∧
    (φ = 5 * π / 6) ∧ (ψ = φ) :=
sorry

end phase_and_initial_phase_theorem_l218_218490


namespace no_real_roots_iff_no_positive_discriminant_l218_218058

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem no_real_roots_iff_no_positive_discriminant (m : ℝ) 
  (h : discriminant m (-2*(m+2)) (m+5) < 0) : 
  (discriminant (m-5) (-2*(m+2)) m < 0 ∨ discriminant (m-5) (-2*(m+2)) m > 0 ∨ m - 5 = 0) :=
by 
  sorry

end no_real_roots_iff_no_positive_discriminant_l218_218058


namespace royal_family_children_l218_218350

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l218_218350


namespace find_p_l218_218650

theorem find_p (p : ℕ) (h : 81^6 = 3^p) : p = 24 :=
sorry

end find_p_l218_218650


namespace number_of_children_l218_218369

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l218_218369


namespace frustum_volume_fraction_l218_218741

theorem frustum_volume_fraction {V_original V_frustum : ℚ} 
(base_edge : ℚ) (height : ℚ) 
(h1 : base_edge = 24) (h2 : height = 18) 
(h3 : V_original = (1 / 3) * (base_edge ^ 2) * height)
(smaller_base_edge : ℚ) (smaller_height : ℚ) 
(h4 : smaller_height = (1 / 3) * height) (h5 : smaller_base_edge = base_edge / 3) 
(V_smaller : ℚ) (h6 : V_smaller = (1 / 3) * (smaller_base_edge ^ 2) * smaller_height)
(h7 : V_frustum = V_original - V_smaller) :
V_frustum / V_original = 13 / 27 :=
sorry

end frustum_volume_fraction_l218_218741


namespace specific_natural_numbers_expr_l218_218754

theorem specific_natural_numbers_expr (a b c : ℕ) 
  (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1) : 
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ (n = (a + b) / c + (b + c) / a + (c + a) / b) :=
by sorry

end specific_natural_numbers_expr_l218_218754


namespace james_total_toys_l218_218858

-- Definition for the number of toy cars
def numToyCars : ℕ := 20

-- Definition for the number of toy soldiers
def numToySoldiers : ℕ := 2 * numToyCars

-- The total number of toys is the sum of toy cars and toy soldiers
def totalToys : ℕ := numToyCars + numToySoldiers

-- Statement to prove: James buys a total of 60 toys
theorem james_total_toys : totalToys = 60 := by
  -- Insert proof here
  sorry

end james_total_toys_l218_218858


namespace sqrt_floor_eq_seven_l218_218592

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l218_218592


namespace power_of_three_l218_218812

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l218_218812


namespace royal_children_count_l218_218346

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l218_218346


namespace new_sign_cost_l218_218008

theorem new_sign_cost 
  (p_s : ℕ) (p_c : ℕ) (n : ℕ) (h_ps : p_s = 30) (h_pc : p_c = 26) (h_n : n = 10) : 
  (p_s - p_c) * n / 2 = 20 := 
by 
  sorry

end new_sign_cost_l218_218008


namespace hcf_of_two_numbers_is_18_l218_218334

theorem hcf_of_two_numbers_is_18
  (product : ℕ)
  (lcm : ℕ)
  (hcf : ℕ) :
  product = 571536 ∧ lcm = 31096 → hcf = 18 := 
by sorry

end hcf_of_two_numbers_is_18_l218_218334


namespace equal_numbers_product_l218_218130

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l218_218130


namespace ramanujan_number_l218_218449

open Complex

theorem ramanujan_number (r h : ℂ) (h_eq : h = 3 + 4 * I )
  (product_eq : r * h = 24 - 10 * I) : 
  r = (112 / 25) - (126 / 25) * I :=
by 
  sorry

end ramanujan_number_l218_218449


namespace floor_sqrt_fifty_l218_218584

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l218_218584


namespace greatest_product_of_two_even_integers_whose_sum_is_300_l218_218155

theorem greatest_product_of_two_even_integers_whose_sum_is_300 :
  ∃ (x y : ℕ), (2 ∣ x) ∧ (2 ∣ y) ∧ (x + y = 300) ∧ (x * y = 22500) :=
by
  sorry

end greatest_product_of_two_even_integers_whose_sum_is_300_l218_218155


namespace vanya_correct_answers_l218_218105

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l218_218105


namespace mass_fraction_K2SO4_l218_218023

theorem mass_fraction_K2SO4 :
  (2.61 * 100 / 160) = 1.63 :=
by
  -- Proof details are not required as per instructions
  sorry

end mass_fraction_K2SO4_l218_218023


namespace max_minus_min_depends_on_a_not_b_l218_218444

def quadratic_function (a b x : ℝ) : ℝ := x^2 + a * x + b

theorem max_minus_min_depends_on_a_not_b (a b : ℝ) :
  let f := quadratic_function a b
  let M := max (f 0) (f 1)
  let m := min (f 0) (f 1)
  M - m == |a| :=
sorry

end max_minus_min_depends_on_a_not_b_l218_218444


namespace number_of_teams_l218_218290

theorem number_of_teams (n : ℕ) (h1 : ∀ k, k = 10) (h2 : n * 10 * (n - 1) / 2 = 1900) : n = 20 :=
by
  sorry

end number_of_teams_l218_218290


namespace no_snow_probability_l218_218494

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l218_218494


namespace power_calculation_l218_218830

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l218_218830


namespace floor_sqrt_50_l218_218595

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l218_218595


namespace possible_values_of_m_l218_218054

theorem possible_values_of_m (a b : ℤ) (h1 : a * b = -14) :
  ∃ m : ℤ, m = a + b ∧ (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
by
  sorry

end possible_values_of_m_l218_218054


namespace even_quadruple_composition_l218_218672

variable {α : Type*} [AddGroup α]

-- Definition of an odd function
def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

theorem even_quadruple_composition {f : α → α} 
  (hf_odd : is_odd_function f) : 
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
by
  sorry

end even_quadruple_composition_l218_218672


namespace notebook_cost_3_dollars_l218_218236

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l218_218236


namespace intersection_eq_l218_218275

-- Given conditions
def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | x > 1 }

-- Statement of the problem to be proved
theorem intersection_eq : M ∩ N = { x | 1 < x ∧ x < 3 } :=
sorry

end intersection_eq_l218_218275


namespace probability_ge_first_second_l218_218547

noncomputable def probability_ge_rolls : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_ge_first_second :
  probability_ge_rolls = 9 / 16 :=
by
  sorry

end probability_ge_first_second_l218_218547


namespace total_earnings_l218_218571

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end total_earnings_l218_218571


namespace hyperbola_eccentricity_l218_218849

-- Define the hyperbola and the condition of the asymptote passing through (2,1)
def hyperbola (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧
               (a ≠ 0 ∧ b ≠ 0) ∧
               (x, y) = (2, 1)

-- Define the eccentricity of the hyperbola
def eccentricity (a b e : ℝ) : Prop :=
  a^2 + b^2 = (b * e)^2

theorem hyperbola_eccentricity (a b e : ℝ) 
  (hx : hyperbola a b)
  (ha : a = 2 * b)
  (ggt: (a^2 = 4 * b^2)) :
  eccentricity a b e → e = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_eccentricity_l218_218849


namespace warriors_wins_count_l218_218314

variable {wins : ℕ → ℕ}
variable (raptors hawks warriors spurs lakers : ℕ)

def conditions (wins : ℕ → ℕ) (raptors hawks warriors spurs lakers : ℕ) : Prop :=
  wins raptors > wins hawks ∧
  wins warriors > wins spurs ∧ wins warriors < wins lakers ∧
  wins spurs > 25

theorem warriors_wins_count
  (wins : ℕ → ℕ)
  (raptors hawks warriors spurs lakers : ℕ)
  (h : conditions wins raptors hawks warriors spurs lakers) :
  wins warriors = 37 := sorry

end warriors_wins_count_l218_218314


namespace probability_of_one_common_l218_218078

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l218_218078


namespace meaningful_fraction_l218_218895

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end meaningful_fraction_l218_218895


namespace value_of_expression_l218_218913

theorem value_of_expression : (3023 - 2990) ^ 2 / 121 = 9 := by
  sorry

end value_of_expression_l218_218913


namespace locus_of_C_l218_218624

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)

theorem locus_of_C :
  ∀ (C : ℝ × ℝ), (C.2 = (b / a) * C.1 ∧ (a * b / Real.sqrt (a ^ 2 + b ^ 2) ≤ C.1) ∧ (C.1 ≤ a)) :=
sorry

end locus_of_C_l218_218624


namespace canoes_vs_kayaks_l218_218165

theorem canoes_vs_kayaks (C K : ℕ) (h1 : 9 * C + 12 * K = 432) (h2 : C = 4 * K / 3) : C - K = 6 :=
sorry

end canoes_vs_kayaks_l218_218165


namespace edge_of_new_cube_l218_218537

theorem edge_of_new_cube (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  ∃ d : ℝ, d^3 = a^3 + b^3 + c^3 ∧ d = 12 :=
by
  sorry

end edge_of_new_cube_l218_218537


namespace royal_family_children_l218_218356

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l218_218356


namespace positive_difference_perimeters_l218_218513

theorem positive_difference_perimeters :
  let w1 := 3
  let h1 := 2
  let w2 := 6
  let h2 := 1
  let P1 := 2 * (w1 + h1)
  let P2 := 2 * (w2 + h2)
  P2 - P1 = 4 := by
  sorry

end positive_difference_perimeters_l218_218513


namespace tangents_secant_intersect_l218_218403

variable {A B C O1 P Q R : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (AB AC : Set (MetricSpace A)) (t : Tangent AB) (s : Tangent AC)

variable (BC : line ( Set A))
variable (APQ : secant A P Q) 

theorem tangents_secant_intersect { AR AP AQ : ℝ } :
  2 / AR = 1 / AP + 1 / AQ :=
by
  sorry

end tangents_secant_intersect_l218_218403


namespace max_value_of_a_l218_218630

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  exp (2 * x) - exp (-2 * x) - 4 * x - a * exp x + a * exp (-x) + 2 * a * x

theorem max_value_of_a : ∀ x > 0, f a x ≥ 0 ↔ a ≤ 8 :=
begin
  sorry
end

end max_value_of_a_l218_218630


namespace vanya_correct_answers_l218_218100

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l218_218100


namespace meaningful_fraction_l218_218896

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end meaningful_fraction_l218_218896


namespace arrow_in_48th_position_l218_218652

def arrow_sequence := ["→", "↔", "↓", "→", "↕"]

theorem arrow_in_48th_position :
  arrow_sequence[48 % arrow_sequence.length] = "↓" :=
by
  sorry

end arrow_in_48th_position_l218_218652


namespace probability_not_snow_l218_218493

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l218_218493


namespace total_votes_l218_218536

theorem total_votes (T F A : ℝ)
  (h1 : F = A + 68)
  (h2 : A = 0.40 * T)
  (h3 : T = F + A) :
  T = 340 :=
by sorry

end total_votes_l218_218536


namespace Vanya_correct_answers_l218_218119

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l218_218119


namespace ones_digit_of_first_in_sequence_l218_218037

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
  
def in_arithmetic_sequence (a d : ℕ) (n : ℕ) : Prop :=
  ∃ k, a = k * d + n

theorem ones_digit_of_first_in_sequence {p q r s t : ℕ}
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (ht : is_prime t)
  (hseq : in_arithmetic_sequence p 10 q ∧ 
          in_arithmetic_sequence q 10 r ∧
          in_arithmetic_sequence r 10 s ∧
          in_arithmetic_sequence s 10 t)
  (hincr : p < q ∧ q < r ∧ r < s ∧ s < t)
  (hstart : p > 5) :
  p % 10 = 1 := sorry

end ones_digit_of_first_in_sequence_l218_218037


namespace vasya_days_l218_218178

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218178


namespace rachel_removed_bottle_caps_l218_218683

def original_bottle_caps : ℕ := 87
def remaining_bottle_caps : ℕ := 40

theorem rachel_removed_bottle_caps :
  original_bottle_caps - remaining_bottle_caps = 47 := by
  sorry

end rachel_removed_bottle_caps_l218_218683


namespace vasya_days_l218_218180

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l218_218180


namespace determine_m_l218_218986

theorem determine_m (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) → m = 2 :=
sorry

end determine_m_l218_218986


namespace fraction_meaningful_range_l218_218898

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_range_l218_218898


namespace power_of_three_l218_218780

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l218_218780


namespace set_union_proof_l218_218274

theorem set_union_proof (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {1, 2^a})
  (hB : B = {a, b}) 
  (h_inter : A ∩ B = {1/4}) :
  A ∪ B = {-2, 1, 1/4} := 
by 
  sorry

end set_union_proof_l218_218274


namespace smaller_angle_measure_l218_218508

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l218_218508


namespace Hiram_age_l218_218329

theorem Hiram_age (H A : ℕ) (h₁ : H + 12 = 2 * A - 4) (h₂ : A = 28) : H = 40 :=
by
  sorry

end Hiram_age_l218_218329


namespace cost_of_600_candies_l218_218225

-- Definitions based on conditions
def costOfBox : ℕ := 6       -- The cost of one box of 25 candies in dollars
def boxSize   : ℕ := 25      -- The number of candies in one box
def cost (n : ℕ) : ℕ := (n / boxSize) * costOfBox -- The cost function for n candies

-- Theorem to be proven
theorem cost_of_600_candies : cost 600 = 144 :=
by sorry

end cost_of_600_candies_l218_218225


namespace conference_center_distance_l218_218478

theorem conference_center_distance
  (d : ℝ)  -- total distance to the conference center
  (t : ℝ)  -- total on-time duration
  (h1 : d = 40 * (t + 1.5))  -- condition from initial speed and late time
  (h2 : d - 40 = 60 * (t - 1.75))  -- condition from increased speed and early arrival
  : d = 310 := 
sorry

end conference_center_distance_l218_218478


namespace triangle_perimeter_l218_218853

theorem triangle_perimeter (x : ℕ) :
  (x = 6 ∨ x = 3) →
  ∃ (a b c : ℕ), (a = x ∧ (b = x ∨ c = x)) ∧ 
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by
  intro h
  sorry

end triangle_perimeter_l218_218853


namespace sum_of_medians_powers_l218_218327

noncomputable def median_length_squared (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / 4

noncomputable def sum_of_fourth_powers_of_medians (a b c : ℝ) : ℝ :=
  let mAD := (median_length_squared a b c)^2
  let mBE := (median_length_squared b c a)^2
  let mCF := (median_length_squared c a b)^2
  mAD^2 + mBE^2 + mCF^2

theorem sum_of_medians_powers :
  sum_of_fourth_powers_of_medians 13 14 15 = 7644.25 :=
by
  sorry

end sum_of_medians_powers_l218_218327


namespace dice_product_divisible_by_8_l218_218717

noncomputable def probability_divisible_by_8 : ℚ := 15 / 16

theorem dice_product_divisible_by_8 :
  ∀ (dices : Fin 8 → Fin 6), 
    (∃ (p : ℕ), (∀ i, dices i.1.succ ∈ {1, 3, 5}) → ¬ (8 ∣ p) → p = dices.prod (λ j, (j + 1 : ℕ))) → 
      (1 - probability_divisible_by_8 = 1 / 16) :=
by
  sorry

end dice_product_divisible_by_8_l218_218717


namespace exponent_power_identity_l218_218797

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l218_218797


namespace has_exactly_two_solutions_iff_l218_218605

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l218_218605


namespace seating_arrangement_l218_218422

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 6 * y = 57) : x = 1 :=
sorry

end seating_arrangement_l218_218422


namespace julias_total_spending_l218_218454

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end julias_total_spending_l218_218454


namespace proof_problem_l218_218270

variable {α : Type*} [LinearOrderedField α]

theorem proof_problem 
  (a b x y : α) 
  (h0 : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y)
  (h1 : a + b + x + y < 2)
  (h2 : a + b^2 = x + y^2)
  (h3 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := 
by
  sorry

end proof_problem_l218_218270


namespace smallest_reducible_fraction_l218_218616

theorem smallest_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 8)) ∧ n = 144 := by
  sorry

end smallest_reducible_fraction_l218_218616


namespace express_b_c_range_a_not_monotonic_l218_218970

noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp (-x)
noncomputable def f' (a b c x : ℝ) : ℝ := 
    (a * x^2 + b * x + c) * (-Real.exp (-x)) + (2 * a * x + b) * Real.exp (-x)

theorem express_b_c (a : ℝ) : 
    (∃ b c : ℝ, f a b c 0 = 2 * a ∧ f' a b c 0 = Real.pi / 4) → 
    (∃ b c : ℝ, b = 1 + 2 * a ∧ c = 2 * a) := 
sorry

noncomputable def g (a x : ℝ) : ℝ := -a * x^2 - x + 1

theorem range_a_not_monotonic (a : ℝ) : 
    (¬ (∀ x y : ℝ, x ∈ Set.Ici (1 / 2) → y ∈ Set.Ici (1 / 2) → x < y → g a x ≤ g a y)) → 
    (-1 / 4 < a ∧ a < 2) := 
sorry

end express_b_c_range_a_not_monotonic_l218_218970


namespace production_today_l218_218617

theorem production_today (n : ℕ) (P T : ℕ) 
  (h1 : n = 4) 
  (h2 : (P + T) / (n + 1) = 58) 
  (h3 : P = n * 50) : 
  T = 90 := 
by
  sorry

end production_today_l218_218617


namespace power_addition_l218_218806

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l218_218806


namespace area_of_field_l218_218389

theorem area_of_field (L W A : ℝ) (hL : L = 20) (hP : L + 2 * W = 25) : A = 50 :=
by
  sorry

end area_of_field_l218_218389


namespace class_size_l218_218928

theorem class_size (n : ℕ) (h1 : 85 - 33 + 90 - 40 = 102) (h2 : (102 : ℚ) / n = 1.5): n = 68 :=
by
  sorry

end class_size_l218_218928


namespace vasya_days_without_purchase_l218_218200

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l218_218200


namespace power_equality_l218_218839

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l218_218839


namespace cos_phi_expression_l218_218092

theorem cos_phi_expression (a b c : ℝ) (φ R : ℝ)
  (habc : a > 0 ∧ b > 0 ∧ c > 0)
  (angles : 2 * φ + 3 * φ + 4 * φ = π)
  (law_of_sines : a / Real.sin (2 * φ) = 2 * R ∧ b / Real.sin (3 * φ) = 2 * R ∧ c / Real.sin (4 * φ) = 2 * R) :
  Real.cos φ = (a + c) / (2 * b) := 
by 
  sorry

end cos_phi_expression_l218_218092


namespace equal_savings_l218_218899

theorem equal_savings (A B AE BE AS BS : ℕ) 
  (hA : A = 2000)
  (hA_B : 5 * B = 4 * A)
  (hAE_BE : 3 * BE = 2 * AE)
  (hSavings : AS = A - AE ∧ BS = B - BE ∧ AS = BS) :
  AS = 800 ∧ BS = 800 :=
by
  -- Placeholders for definitions and calculations
  sorry

end equal_savings_l218_218899


namespace car_passing_problem_l218_218869

noncomputable def maxCarsPerHourDividedBy10 : ℕ :=
  let unit_length (n : ℕ) := 5 * (n + 1)
  let cars_passed_in_one_hour (n : ℕ) := 10000 * n / unit_length n
  Nat.div (2000) (10)

theorem car_passing_problem : maxCarsPerHourDividedBy10 = 200 :=
  by
  sorry

end car_passing_problem_l218_218869


namespace equal_numbers_product_l218_218132

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l218_218132
