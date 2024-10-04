import Mathlib

namespace aram_fraction_of_fine_l285_285865

theorem aram_fraction_of_fine (F : ℝ) (H1 : Joe_paid = (1/4)*F + 3)
  (H2 : Peter_paid = (1/3)*F - 3)
  (H3 : Aram_paid = (1/2)*F - 4)
  (H4 : Joe_paid + Peter_paid + Aram_paid = F) : 
  Aram_paid / F = 5 / 12 := 
sorry

end aram_fraction_of_fine_l285_285865


namespace abc_value_l285_285896

-- Define constants for the problem
variable (a b c k : ℕ)

-- Assumptions based on the given conditions
axiom h1 : a - b = 3
axiom h2 : a^2 + b^2 = 29
axiom h3 : a^2 + b^2 + c^2 = k
axiom pos_k : k > 0
axiom pos_a : a > 0

-- The goal is to prove that abc = 10
theorem abc_value : a * b * c = 10 :=
by
  sorry

end abc_value_l285_285896


namespace find_common_ratio_l285_285477

noncomputable def geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 5 - a 1 = 15) ∧ (a 4 - a 2 = 6) → (q = 1/2 ∨ q = 2)

-- We declare this as a theorem statement
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) : geometric_sequence_common_ratio a q :=
sorry

end find_common_ratio_l285_285477


namespace find_divisor_l285_285050

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) 
  (h1 : dividend = 62976) 
  (h2 : quotient = 123) 
  (h3 : dividend = divisor * quotient) 
  : divisor = 512 := 
by
  sorry

end find_divisor_l285_285050


namespace problem_1_problem_2_l285_285298

def set_A := { y : ℝ | 2 < y ∧ y < 3 }
def set_B := { x : ℝ | x > 1 ∨ x < -1 }

theorem problem_1 : { x : ℝ | x ∈ set_A ∧ x ∈ set_B } = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

def set_C := { x : ℝ | x ∈ set_B ∧ ¬(x ∈ set_A) }

theorem problem_2 : set_C = { x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3 } :=
by
  sorry

end problem_1_problem_2_l285_285298


namespace scientific_notation_of_8200000_l285_285016

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l285_285016


namespace prove_range_of_m_prove_m_value_l285_285614

def quadratic_roots (m : ℝ) (x1 x2 : ℝ) : Prop := 
  x1 * x1 - (2 * m - 3) * x1 + m * m = 0 ∧ 
  x2 * x2 - (2 * m - 3) * x2 + m * m = 0

def range_of_m (m : ℝ) : Prop := 
  m <= 3/4

def condition_on_m (m : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = -(x1 * x2)

theorem prove_range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots m x1 x2) → range_of_m m :=
sorry

theorem prove_m_value (m : ℝ) (x1 x2 : ℝ) :
  quadratic_roots m x1 x2 → condition_on_m m x1 x2 → m = -3 :=
sorry

end prove_range_of_m_prove_m_value_l285_285614


namespace parallelogram_base_l285_285885

theorem parallelogram_base
  (Area Height Base : ℕ)
  (h_area : Area = 120)
  (h_height : Height = 10)
  (h_area_eq : Area = Base * Height) :
  Base = 12 :=
by
  /- 
    We assume the conditions:
    1. Area = 120
    2. Height = 10
    3. Area = Base * Height 
    Then, we need to prove that Base = 12.
  -/
  sorry

end parallelogram_base_l285_285885


namespace circles_intersect_at_two_points_l285_285315

theorem circles_intersect_at_two_points : 
  let C1 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}
  let C2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 36}
  ∃ pts : Finset (ℝ × ℝ), pts.card = 2 ∧ ∀ p ∈ pts, p ∈ C1 ∧ p ∈ C2 := 
sorry

end circles_intersect_at_two_points_l285_285315


namespace initial_blocks_l285_285212

-- Definitions of the given conditions
def blocks_eaten : ℕ := 29
def blocks_remaining : ℕ := 26

-- The statement we need to prove
theorem initial_blocks : blocks_eaten + blocks_remaining = 55 :=
by
  -- Proof is not required as per instructions
  sorry

end initial_blocks_l285_285212


namespace unattainable_y_l285_285742

theorem unattainable_y (x : ℝ) (hx : x ≠ -2 / 3) : ¬ (∃ x, y = (x - 3) / (3 * x + 2) ∧ y = 1 / 3) := by
  sorry

end unattainable_y_l285_285742


namespace fraction_simplified_l285_285943

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end fraction_simplified_l285_285943


namespace find_divisor_l285_285936

theorem find_divisor (q r : ℤ) : ∃ d : ℤ, 151 = d * q + r ∧ q = 11 ∧ r = -4 → d = 14 :=
by
  intros
  sorry

end find_divisor_l285_285936


namespace sum_of_coefficients_at_1_l285_285052

def P (x : ℝ) := 2 * (4 * x^8 - 3 * x^5 + 9)
def Q (x : ℝ) := 9 * (x^6 + 2 * x^3 - 8)
def R (x : ℝ) := P x + Q x

theorem sum_of_coefficients_at_1 : R 1 = -25 := by
  sorry

end sum_of_coefficients_at_1_l285_285052


namespace number_of_cows_l285_285325

variable (x y z : ℕ)

theorem number_of_cows (h1 : 4 * x + 2 * y + 2 * z = 24 + 2 * (x + y + z)) (h2 : z = y / 2) : x = 12 := 
sorry

end number_of_cows_l285_285325


namespace initial_maintenance_time_l285_285124

theorem initial_maintenance_time (x : ℝ) 
  (h1 : (1 + (1 / 3)) * x = 60) : 
  x = 45 :=
by
  sorry

end initial_maintenance_time_l285_285124


namespace history_only_students_l285_285192

theorem history_only_students 
  (total_students : ℕ)
  (history_students stats_students physics_students chem_students : ℕ) 
  (hist_stats hist_phys hist_chem stats_phys stats_chem phys_chem all_four : ℕ) 
  (h1 : total_students = 500)
  (h2 : history_students = 150)
  (h3 : stats_students = 130)
  (h4 : physics_students = 120)
  (h5 : chem_students = 100)
  (h6 : hist_stats = 60)
  (h7 : hist_phys = 50)
  (h8 : hist_chem = 40)
  (h9 : stats_phys = 35)
  (h10 : stats_chem = 30)
  (h11 : phys_chem = 25)
  (h12 : all_four = 20) : 
  (history_students - hist_stats - hist_phys - hist_chem + all_four) = 20 := 
by 
  sorry

end history_only_students_l285_285192


namespace original_price_of_saree_is_400_l285_285117

-- Define the original price of the saree
variable (P : ℝ)

-- Define the sale price after successive discounts
def sale_price (P : ℝ) : ℝ := 0.80 * P * 0.95

-- We want to prove that the original price P is 400 given that the sale price is 304
theorem original_price_of_saree_is_400 (h : sale_price P = 304) : P = 400 :=
sorry

end original_price_of_saree_is_400_l285_285117


namespace number_of_possible_triples_l285_285710

-- Given conditions
variables (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)

-- Revenue equation
def revenue_equation : Prop := 10 * x + 5 * y + z = 120

-- Proving the solution
theorem number_of_possible_triples (h : revenue_equation x y z) : 
  ∃ (n : ℕ), n = 121 :=
by
  sorry

end number_of_possible_triples_l285_285710


namespace unique_solution_l285_285809

theorem unique_solution :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + c * a) + 18) →
    (a = 1 ∧ b = 2 ∧ c = 3) :=
by
  intros a b c ha hb hc h
  have h1 : a = 1 := sorry
  have h2 : b = 2 := sorry
  have h3 : c = 3 := sorry
  exact ⟨h1, h2, h3⟩

end unique_solution_l285_285809


namespace flu_infection_equation_l285_285430

theorem flu_infection_equation (x : ℝ) :
  (1 + x)^2 = 144 :=
sorry

end flu_infection_equation_l285_285430


namespace sum_of_roots_l285_285084

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end sum_of_roots_l285_285084


namespace find_k_l285_285472

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l285_285472


namespace lecture_room_configuration_l285_285145

theorem lecture_room_configuration (m n : ℕ) (boys_per_row girls_per_column unoccupied_chairs : ℕ) :
    boys_per_row = 6 →
    girls_per_column = 8 →
    unoccupied_chairs = 15 →
    (m * n = boys_per_row * m + girls_per_column * n + unoccupied_chairs) →
    (m = 71 ∧ n = 7) ∨
    (m = 29 ∧ n = 9) ∨
    (m = 17 ∧ n = 13) ∨
    (m = 15 ∧ n = 15) ∨
    (m = 11 ∧ n = 27) ∨
    (m = 9 ∧ n = 69) :=
by
  intros h1 h2 h3 h4
  sorry

end lecture_room_configuration_l285_285145


namespace opposite_of_neg3_l285_285381

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l285_285381


namespace recurring_decimal_to_fraction_l285_285398

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l285_285398


namespace tank_salt_solution_l285_285841

theorem tank_salt_solution (x : ℝ) (hx1 : 0.20 * x / (3 / 4 * x + 30) = 1 / 3) : x = 200 :=
by sorry

end tank_salt_solution_l285_285841


namespace amount_paid_for_grapes_l285_285272

-- Definitions based on the conditions
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := 2.23

-- The statement to be proved
theorem amount_paid_for_grapes : total_spent + refund_for_cherries = 12.08 := 
by 
  -- Here the specific mathematical proof would go, but is replaced by sorry as instructed
  sorry

end amount_paid_for_grapes_l285_285272


namespace triangle_cosine_l285_285333

theorem triangle_cosine (LM : ℝ) (cos_N : ℝ) (LN : ℝ) (h1 : LM = 20) (h2 : cos_N = 3/5) :
  LM / LN = cos_N → LN = 100 / 3 :=
by
  intro h3
  sorry

end triangle_cosine_l285_285333


namespace fraction_equiv_l285_285399

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l285_285399


namespace M_gt_N_l285_285639

-- Define M and N
def M (x y : ℝ) : ℝ := x^2 + y^2 + 1
def N (x y : ℝ) : ℝ := 2 * (x + y - 1)

-- State the theorem to prove M > N given the conditions
theorem M_gt_N (x y : ℝ) : M x y > N x y := by
  sorry

end M_gt_N_l285_285639


namespace Q_has_negative_and_potentially_positive_roots_l285_285456

def Q (x : ℝ) : ℝ := x^7 - 4 * x^6 + 2 * x^5 - 9 * x^3 + 2 * x + 16

theorem Q_has_negative_and_potentially_positive_roots :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧ (∃ y : ℝ, y > 0 ∧ Q y = 0 ∨ ∀ z : ℝ, Q z > 0) :=
by
  sorry

end Q_has_negative_and_potentially_positive_roots_l285_285456


namespace probability_of_four_ones_l285_285553

noncomputable def probability_exactly_four_ones : ℚ :=
  (Nat.choose 12 4 * (1/6)^4 * (5/6)^8)

theorem probability_of_four_ones :
  abs (probability_exactly_four_ones.toReal - 0.114) < 0.001 :=
by
  sorry

end probability_of_four_ones_l285_285553


namespace number_of_zeros_of_f_l285_285115

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * Real.exp x - 2
else abs (2 - Real.log x / Real.log 2)

theorem number_of_zeros_of_f :
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = Real.log (1 / 2) ∧ x₂ = 4 :=
by
  sorry

end number_of_zeros_of_f_l285_285115


namespace max_license_plates_l285_285421

noncomputable def max_distinct_plates (m n : ℕ) : ℕ :=
  m ^ (n - 1)

theorem max_license_plates :
  max_distinct_plates 10 6 = 100000 := by
  sorry

end max_license_plates_l285_285421


namespace min_expression_l285_285740

theorem min_expression : ∀ x y : ℝ, ∃ x, 4 * x^2 + 4 * x * (Real.sin y) - (Real.cos y)^2 = -1 := by
  sorry

end min_expression_l285_285740


namespace age_of_b_l285_285136

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 72) : b = 28 :=
by
  sorry

end age_of_b_l285_285136


namespace curve_c1_polar_eqn_curve_c2_rect_eqn_line_segment_AB_length_l285_285655

theorem curve_c1_polar_eqn (x y : ℝ) (α : ℝ) (h₁ : x = 1 + Mathlib.cos α) (h₂ : y = Mathlib.sin α) :
  ∃ θ ρ, x = ρ * Mathlib.cos θ ∧ y = ρ * Mathlib.sin θ ∧ ρ = 2 * Mathlib.cos θ :=
by sorry

theorem curve_c2_rect_eqn (ρ θ : ℝ) (h : ρ = -2 * Mathlib.sin θ) :
  ∃ (x y : ℝ), x = ρ * Mathlib.cos θ ∧ y = ρ * Mathlib.sin θ ∧ x^2 + (y + 1)^2 = 1 :=
by sorry

theorem line_segment_AB_length (h₁ : ρ_1 = 2 * Mathlib.cos (-Math.pi / 3)) (h₂ : ρ_2 = -2 * Mathlib.sin (-Math.pi / 3))
  (h_line : ∀ x y, Mathlib.sqrt 3 * x + y = 0):
  ∃ (A B : ℝ), |(ρ_1 - ρ_2)| = Mathlib.sqrt 3 - 1 :=
by sorry

end curve_c1_polar_eqn_curve_c2_rect_eqn_line_segment_AB_length_l285_285655


namespace supplement_of_complement_of_30_degrees_l285_285240

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α
def α : ℝ := 30

theorem supplement_of_complement_of_30_degrees : supplement (complement α) = 120 := 
by
  sorry

end supplement_of_complement_of_30_degrees_l285_285240


namespace lottery_prob_l285_285587

open Finset

/-- Definition of combinations C(n, k) -/
def combinations (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem lottery_prob :
  let total_tickets := 10
  let winning_tickets := 3
  let people := 5
  let total_ways := combinations total_tickets people
  let non_winning_tickets := total_tickets - winning_tickets
  let non_winning_ways := combinations non_winning_tickets people
  let prob := 1 - (non_winning_ways / total_ways : ℚ)
  prob = 11 / 12 :=
by
  sorry

end lottery_prob_l285_285587


namespace speed_boat_in_still_water_l285_285961

-- Define the conditions
def speed_of_current := 20
def speed_upstream := 30

-- Define the effective speed given conditions
def effective_speed (speed_in_still_water : ℕ) := speed_in_still_water - speed_of_current

-- Theorem stating the problem
theorem speed_boat_in_still_water : 
  ∃ (speed_in_still_water : ℕ), effective_speed speed_in_still_water = speed_upstream ∧ speed_in_still_water = 50 := 
by 
  -- Proof to be filled in
  sorry

end speed_boat_in_still_water_l285_285961


namespace aarons_brothers_number_l285_285987

-- We are defining the conditions as functions

def number_of_aarons_sisters := 4
def bennetts_brothers := 6
def bennetts_cousins := 3
def twice_aarons_brothers_minus_two (Ba : ℕ) := 2 * Ba - 2
def bennetts_cousins_one_more_than_aarons_sisters (As : ℕ) := As + 1

-- We need to prove that Aaron's number of brothers Ba is 4 under these conditions

theorem aarons_brothers_number : ∃ (Ba : ℕ), 
  bennetts_brothers = twice_aarons_brothers_minus_two Ba ∧ 
  bennetts_cousins = bennetts_cousins_one_more_than_aarons_sisters number_of_aarons_sisters ∧ 
  Ba = 4 :=
by {
  sorry
}

end aarons_brothers_number_l285_285987


namespace perpendicular_vectors_l285_285625

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l285_285625


namespace white_tiles_count_l285_285338

theorem white_tiles_count (total_tiles yellow_tiles purple_tiles : ℕ)
    (hy : yellow_tiles = 3)
    (hb : ∃ blue_tiles, blue_tiles = yellow_tiles + 1)
    (hp : purple_tiles = 6)
    (ht : total_tiles = 20) : 
    ∃ white_tiles, white_tiles = 7 :=
by
  obtain ⟨blue_tiles, hb_eq⟩ := hb
  let non_white_tiles := yellow_tiles + blue_tiles + purple_tiles
  have hnwt : non_white_tiles = 3 + (3 + 1) + 6,
  {
    rw [hy, hp, hb_eq],
    ring,
  }
  have hwt : total_tiles - non_white_tiles = 7,
  {
    rw ht,
    rw hnwt,
    norm_num,
  }
  use total_tiles - non_white_tiles,
  exact hwt,

end white_tiles_count_l285_285338


namespace cost_price_l285_285996

theorem cost_price (SP : ℝ) (profit_percent : ℝ) (C : ℝ) 
  (h1 : SP = 400) 
  (h2 : profit_percent = 25) 
  (h3 : SP = C + (profit_percent / 100) * C) : 
  C = 320 := 
by
  sorry

end cost_price_l285_285996


namespace quadratic_eq_unique_k_l285_285476

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l285_285476


namespace find_f_8_l285_285049

def f (n : ℕ) : ℕ := n^2 - 3 * n + 20

theorem find_f_8 : f 8 = 60 := 
by 
sorry

end find_f_8_l285_285049


namespace jordyn_total_cost_l285_285323

-- Definitions for conditions
def price_cherries : ℝ := 5
def price_olives : ℝ := 7
def number_of_bags : ℕ := 50
def discount_rate : ℝ := 0.10 

-- Define the discounted price function
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate the total cost for Jordyn
def total_cost (price_cherries price_olives : ℝ) (number_of_bags : ℕ) (discount_rate : ℝ) : ℝ :=
  (number_of_bags * discounted_price price_cherries discount_rate) + 
  (number_of_bags * discounted_price price_olives discount_rate)

-- Prove the final cost
theorem jordyn_total_cost : total_cost price_cherries price_olives number_of_bags discount_rate = 540 := by
  sorry

end jordyn_total_cost_l285_285323


namespace find_decreased_amount_l285_285253

variables (x y : ℝ)

axiom h1 : 0.20 * x - y = 6
axiom h2 : x = 50.0

theorem find_decreased_amount : y = 4 :=
by
  sorry

end find_decreased_amount_l285_285253


namespace flowers_in_each_basket_l285_285283

theorem flowers_in_each_basket
  (plants_per_daughter : ℕ)
  (num_daughters : ℕ)
  (grown_flowers : ℕ)
  (died_flowers : ℕ)
  (num_baskets : ℕ)
  (h1 : plants_per_daughter = 5)
  (h2 : num_daughters = 2)
  (h3 : grown_flowers = 20)
  (h4 : died_flowers = 10)
  (h5 : num_baskets = 5) :
  (plants_per_daughter * num_daughters + grown_flowers - died_flowers) / num_baskets = 4 :=
by
  sorry

end flowers_in_each_basket_l285_285283


namespace base_of_parallelogram_l285_285291

theorem base_of_parallelogram (A h b : ℝ) (hA : A = 960) (hh : h = 16) :
  A = h * b → b = 60 :=
by
  sorry

end base_of_parallelogram_l285_285291


namespace quadratic_eq_unique_k_l285_285475

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l285_285475


namespace second_machine_equation_l285_285143

-- Let p1_rate and p2_rate be the rates of printing for machine 1 and 2 respectively.
-- Let x be the unknown time for the second machine to print 500 envelopes.

theorem second_machine_equation (x : ℝ) :
    (500 / 8) + (500 / x) = (500 / 2) :=
  sorry

end second_machine_equation_l285_285143


namespace find_amplitude_l285_285720

-- Conditions
variables (a b c d : ℝ)

theorem find_amplitude
  (h1 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h2 : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) :
  a = 4 :=
by 
  sorry

end find_amplitude_l285_285720


namespace customer_outreach_time_l285_285449

variable (x : ℝ)

theorem customer_outreach_time
  (h1 : 8 = x + x / 2 + 2) :
  x = 4 :=
by sorry

end customer_outreach_time_l285_285449


namespace flowers_per_basket_l285_285281

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end flowers_per_basket_l285_285281


namespace minimum_solutions_in_interval_l285_285026

open Function Real

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define what it means for a function to be periodic
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- Main theorem statement
theorem minimum_solutions_in_interval :
  ∀ (f : ℝ → ℝ),
  is_even f → is_periodic f 3 → f 2 = 0 →
  (∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ x1 < 6 ∧ f x1 = 0 ∧
                     0 < x2 ∧ x2 < 6 ∧ f x2 = 0 ∧
                     0 < x3 ∧ x3 < 6 ∧ f x3 = 0 ∧
                     0 < x4 ∧ x4 < 6 ∧ f x4 = 0 ∧
                     x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :=
by
  sorry

end minimum_solutions_in_interval_l285_285026


namespace find_first_purchase_find_max_profit_purchase_plan_l285_285598

-- Defining the parameters for the problem
structure KeychainParams where
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  total_purchase_cost_first : ℕ
  total_keychains_first : ℕ
  total_purchase_cost_second : ℕ
  total_keychains_second : ℕ
  purchase_cap_second : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ

-- Define the initial setup
def params : KeychainParams := {
  purchase_price_A := 30,
  purchase_price_B := 25,
  total_purchase_cost_first := 850,
  total_keychains_first := 30,
  total_purchase_cost_second := 2200,
  total_keychains_second := 80,
  purchase_cap_second := 2200,
  selling_price_A := 45,
  selling_price_B := 37
}

-- Part 1: Prove the number of keychains purchased for each type
theorem find_first_purchase (x y : ℕ)
  (h₁ : x + y = params.total_keychains_first)
  (h₂ : params.purchase_price_A * x + params.purchase_price_B * y = params.total_purchase_cost_first) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2: Prove the purchase plan that maximizes the sales profit
theorem find_max_profit_purchase_plan (m : ℕ)
  (h₃ : m + (params.total_keychains_second - m) = params.total_keychains_second)
  (h₄ : params.purchase_price_A * m + params.purchase_price_B * (params.total_keychains_second - m) ≤ params.purchase_cap_second) :
  m = 40 ∧ (params.selling_price_A - params.purchase_price_A) * m + (params.selling_price_B - params.purchase_price_B) * (params.total_keychains_second - m) = 1080 :=
sorry

end find_first_purchase_find_max_profit_purchase_plan_l285_285598


namespace max_value_of_expression_l285_285324

open Classical
open Real

theorem max_value_of_expression (a b : ℝ) (c : ℝ) (h1 : a^2 + b^2 = c^2 + ab) (h2 : c = 1) :
  ∃ x : ℝ, x = (1 / 2) * b + a ∧ x = (sqrt 21) / 3 := 
sorry

end max_value_of_expression_l285_285324


namespace number_of_younger_employees_correct_l285_285852

noncomputable def total_employees : ℕ := 200
noncomputable def younger_employees : ℕ := 120
noncomputable def sample_size : ℕ := 25

def number_of_younger_employees_to_be_drawn (total younger sample : ℕ) : ℕ :=
  sample * younger / total

theorem number_of_younger_employees_correct :
  number_of_younger_employees_to_be_drawn total_employees younger_employees sample_size = 15 := by
  sorry

end number_of_younger_employees_correct_l285_285852


namespace total_number_of_members_l285_285193

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end total_number_of_members_l285_285193


namespace girl_scouts_short_amount_l285_285536

-- Definitions based on conditions
def amount_earned : ℝ := 30
def pool_entry_cost_per_person : ℝ := 2.50
def num_people : ℕ := 10
def transportation_fee_per_person : ℝ := 1.25
def snack_cost_per_person : ℝ := 3.00

-- Calculate individual costs
def total_pool_entry_cost : ℝ := pool_entry_cost_per_person * num_people
def total_transportation_fee : ℝ := transportation_fee_per_person * num_people
def total_snack_cost : ℝ := snack_cost_per_person * num_people

-- Calculate total expenses
def total_expenses : ℝ := total_pool_entry_cost + total_transportation_fee + total_snack_cost

-- The amount left after expenses
def amount_left : ℝ := amount_earned - total_expenses

-- Proof problem statement
theorem girl_scouts_short_amount : amount_left = -37.50 := by
  sorry

end girl_scouts_short_amount_l285_285536


namespace smallest_number_l285_285588

theorem smallest_number (S : set ℤ) (h : S = {0, -3, 1, -1}) : ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 :=
by
  sorry

end smallest_number_l285_285588


namespace correct_regression_equation_l285_285767

-- Problem Statement
def negatively_correlated (x y : ℝ) : Prop := sorry -- Define negative correlation for x, y
def sample_mean_x : ℝ := 3
def sample_mean_y : ℝ := 3.5
def regression_equation (b0 b1 : ℝ) (x : ℝ) : ℝ := b0 + b1 * x

theorem correct_regression_equation 
    (H_neg_corr : negatively_correlated x y) :
    regression_equation 9.5 (-2) sample_mean_x = sample_mean_y :=
by
    -- The proof will go here, skipping with sorry
    sorry

end correct_regression_equation_l285_285767


namespace area_increase_by_nine_l285_285229

theorem area_increase_by_nine (a : ℝ) :
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := extended_side_length^2;
  extended_area / original_area = 9 :=
by
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := (extended_side_length)^2;
  sorry

end area_increase_by_nine_l285_285229


namespace solve_for_x_l285_285215

theorem solve_for_x (x : ℝ) (h : 3 * x - 7 = 2 * x + 5) : x = 12 :=
sorry

end solve_for_x_l285_285215


namespace percentage_increase_bears_l285_285915

-- Define the initial conditions
variables (B H : ℝ) -- B: bears per week without an assistant, H: hours per week without an assistant

-- Define the rate without assistant
def rate_without_assistant : ℝ := B / H

-- Define the working hours with an assistant
def hours_with_assistant : ℝ := 0.9 * H

-- Define the rate with an assistant (100% increase)
def rate_with_assistant : ℝ := 2 * rate_without_assistant

-- Define the number of bears per week with an assistant
def bears_with_assistant : ℝ := rate_with_assistant * hours_with_assistant

-- Prove the percentage increase in the number of bears made per week
theorem percentage_increase_bears (hB : B > 0) (hH : H > 0) :
  ((bears_with_assistant B H - B) / B) * 100 = 80 :=
by
  unfold bears_with_assistant rate_with_assistant hours_with_assistant rate_without_assistant
  simp
  sorry

end percentage_increase_bears_l285_285915


namespace solve_quadratic_eq_l285_285814

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end solve_quadratic_eq_l285_285814


namespace enrique_shredder_Y_feeds_l285_285162

theorem enrique_shredder_Y_feeds :
  let typeB_contracts := 350
  let pages_per_TypeB := 10
  let shredderY_capacity := 8
  let total_pages_TypeB := typeB_contracts * pages_per_TypeB
  let feeds_ShredderY := (total_pages_TypeB + shredderY_capacity - 1) / shredderY_capacity
  feeds_ShredderY = 438 := sorry

end enrique_shredder_Y_feeds_l285_285162


namespace maximum_value_inequality_l285_285732

theorem maximum_value_inequality (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 :=
sorry

end maximum_value_inequality_l285_285732


namespace axis_of_symmetry_translated_graph_l285_285319

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

theorem axis_of_symmetry_translated_graph (k : ℤ) :
 ∃ x : ℝ, 2 * x + π / 6 = k * π + π / 2 :=
sorry

end axis_of_symmetry_translated_graph_l285_285319


namespace total_money_given_to_children_l285_285147

theorem total_money_given_to_children (B : ℕ) (x : ℕ) (total : ℕ) 
  (h1 : B = 300) 
  (h2 : x = B / 3) 
  (h3 : total = (2 * x) + (3 * x) + (4 * x)) : 
  total = 900 := 
by 
  sorry

end total_money_given_to_children_l285_285147


namespace sum_x_coordinates_mod11_l285_285023

theorem sum_x_coordinates_mod11 :
    ∀ x y : ℕ, (y ≡ 3 * x + 1 [MOD 11]) → (y ≡ 7 * x + 5 [MOD 11]) → (0 ≤ x ∧ x < 11) → x = 10 := 
by
  intros x y h1 h2 bounds
  have : 3 * x + 1 ≡ 7 * x + 5 [MOD 11] := by rw [Nat.ModEq.symm h1, h2]
  have : 4 * x + 4 ≡ 0 [MOD 11] := by linarith 
  have : 4 * (x + 1) ≡ 0 [MOD 11] := by linarith
  have : x + 1 ≡ 0 [MOD 11] := sorry -- because 4 and 11 are coprime
  have : x ≡ 10 [MOD 11] := by linarith
  exact sorry -- constraints on x show it must be 10

end sum_x_coordinates_mod11_l285_285023


namespace complex_number_a_eq_1_l285_285481

theorem complex_number_a_eq_1 
  (a : ℝ) 
  (h : ∃ b : ℝ, (a - b * I) / (1 + I) = 0 + b * I) : 
  a = 1 := 
sorry

end complex_number_a_eq_1_l285_285481


namespace opposite_of_neg3_is_3_l285_285371

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l285_285371


namespace vanessa_scored_27_points_l285_285831

variable (P : ℕ) (number_of_players : ℕ) (average_points_per_player : ℚ) (vanessa_points : ℕ)

axiom team_total_points : P = 48
axiom other_players : number_of_players = 6
axiom average_points_per_other_player : average_points_per_player = 3.5

theorem vanessa_scored_27_points 
  (h1 : P = 48)
  (h2 : number_of_players = 6)
  (h3 : average_points_per_player = 3.5)
: vanessa_points = 27 :=
sorry

end vanessa_scored_27_points_l285_285831


namespace opposite_of_neg3_l285_285379

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l285_285379


namespace find_x_l285_285601

theorem find_x (x : ℝ) : 0.003 + 0.158 + x = 2.911 → x = 2.750 :=
by
  sorry

end find_x_l285_285601


namespace rationalize_denominator_l285_285491

theorem rationalize_denominator (t : ℝ) (h : t = 1 / (1 - Real.sqrt (Real.sqrt 2))) : 
  t = -(1 + Real.sqrt (Real.sqrt 2)) * (1 + Real.sqrt 2) :=
by
  sorry

end rationalize_denominator_l285_285491


namespace volume_of_one_slice_l285_285583

theorem volume_of_one_slice
  (circumference : ℝ)
  (c : circumference = 18 * Real.pi):
  ∃ V, V = 162 * Real.pi :=
by sorry

end volume_of_one_slice_l285_285583


namespace sqrt_ceil_eq_one_range_of_x_l285_285556

/-- Given $[m]$ represents the largest integer not greater than $m$, prove $[\sqrt{2}] = 1$. -/
theorem sqrt_ceil_eq_one (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) :
  floor (Real.sqrt 2) = 1 :=
sorry

/-- Given $[m]$ represents the largest integer not greater than $m$ and $[3 + \sqrt{x}] = 6$, 
  prove $9 \leq x < 16$. -/
theorem range_of_x (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) 
  (x : ℝ) (h : floor (3 + Real.sqrt x) = 6) :
  9 ≤ x ∧ x < 16 :=
sorry

end sqrt_ceil_eq_one_range_of_x_l285_285556


namespace robert_turns_30_after_2_years_l285_285098

variable (P R : ℕ) -- P for Patrick's age, R for Robert's age
variable (h1 : P = 14) -- Patrick is 14 years old now
variable (h2 : P * 2 = R) -- Patrick is half the age of Robert

theorem robert_turns_30_after_2_years : R + 2 = 30 :=
by
  -- Here should be the proof, but for now we skip it with sorry
  sorry

end robert_turns_30_after_2_years_l285_285098


namespace part1_part2_part3_l285_285929

noncomputable def a (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 3/2 else 
if n = 3 then 5/4 else 
sorry

noncomputable def S (n : ℕ) : ℝ := sorry

axiom recurrence {n : ℕ} (h : n ≥ 2) : 4 * S (n + 2) + 5 * S n = 8 * S (n + 1) + S (n - 1)

-- Part 1
theorem part1 : a 4 = 7 / 8 :=
sorry

-- Part 2
theorem part2 : ∃ (r : ℝ) (b : ℕ → ℝ), (r = 1/2) ∧ (∀ n ≥ 1, a (n + 1) - r * a n = b n) :=
sorry

-- Part 3
theorem part3 : ∀ n, a n = (2 * n - 1) / 2^(n - 1) :=
sorry

end part1_part2_part3_l285_285929


namespace tangent_line_at_origin_l285_285755

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

theorem tangent_line_at_origin (a : ℝ) (h : ∀ x, (3 * x^2 + 2 * a * x + (a - 2)) = 3 * (-x)^2 + 2 * a * (-x) + (a - 2)) :
  tangent_at (f a) (0 : ℝ) = -2 * x := 
sorry

end tangent_line_at_origin_l285_285755


namespace find_possible_values_l285_285345

noncomputable def complex_values (x y : ℂ) : Prop :=
  (x^2 + y^2) / (x + y) = 4 ∧ (x^4 + y^4) / (x^3 + y^3) = 2

theorem find_possible_values (x y : ℂ) (h : complex_values x y) :
  ∃ z : ℂ, z = (x^6 + y^6) / (x^5 + y^5) ∧ (z = 10 + 2 * Real.sqrt 17 ∨ z = 10 - 2 * Real.sqrt 17) :=
sorry

end find_possible_values_l285_285345


namespace trigonometric_identity_proof_l285_285762

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_proof_l285_285762


namespace common_denominator_first_set_common_denominator_second_set_l285_285886

theorem common_denominator_first_set (x y : ℕ) (h₁ : y ≠ 0) : Nat.lcm (3 * y) (2 * y^2) = 6 * y^2 :=
by sorry

theorem common_denominator_second_set (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : Nat.lcm (a^2 * b) (3 * a * b^2) = 3 * a^2 * b^2 :=
by sorry

end common_denominator_first_set_common_denominator_second_set_l285_285886


namespace calc1_calc2_calc3_calc4_calc5_calc6_l285_285939

theorem calc1 : 320 + 16 * 27 = 752 :=
by
  -- Proof goes here
  sorry

theorem calc2 : 1500 - 125 * 8 = 500 :=
by
  -- Proof goes here
  sorry

theorem calc3 : 22 * 22 - 84 = 400 :=
by
  -- Proof goes here
  sorry

theorem calc4 : 25 * 8 * 9 = 1800 :=
by
  -- Proof goes here
  sorry

theorem calc5 : (25 + 38) * 15 = 945 :=
by
  -- Proof goes here
  sorry

theorem calc6 : (62 + 12) * 38 = 2812 :=
by
  -- Proof goes here
  sorry

end calc1_calc2_calc3_calc4_calc5_calc6_l285_285939


namespace find_real_root_a_l285_285922

theorem find_real_root_a (a b c : ℂ) (ha : a.im = 0) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 3) : a = 1 :=
sorry

end find_real_root_a_l285_285922


namespace circumscribed_circle_radius_l285_285765

variables (A B C : ℝ) (a b c : ℝ) (R : ℝ) (area : ℝ)

-- Given conditions
def sides_ratio := a / b = 7 / 5 ∧ b / c = 5 / 3
def triangle_area := area = 45 * Real.sqrt 3
def sides := (a, b, c)
def angles := (A, B, C)

-- Prove radius
theorem circumscribed_circle_radius 
  (h_ratio : sides_ratio a b c)
  (h_area : triangle_area area) :
  R = 14 :=
sorry

end circumscribed_circle_radius_l285_285765


namespace min_x2_y2_z2_given_condition_l285_285665

theorem min_x2_y2_z2_given_condition (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ (c : ℝ), c = 3 ∧ (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3 * x * y * z = 8 → x^2 + y^2 + z^2 ≥ c) := 
sorry

end min_x2_y2_z2_given_condition_l285_285665


namespace ceil_sqrt_sum_l285_285458

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 5⌉ + ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 500⌉ + ⌈Real.sqrt 1000⌉ = 66 :=
by
  sorry

end ceil_sqrt_sum_l285_285458


namespace find_ticket_price_l285_285968

theorem find_ticket_price
  (P : ℝ) -- The original price of each ticket
  (h1 : 10 * 0.6 * P + 20 * 0.85 * P + 26 * P = 980) :
  P = 20 :=
sorry

end find_ticket_price_l285_285968


namespace max_students_l285_285443

open BigOperators

def seats_in_row (i : ℕ) : ℕ := 8 + 2 * i

def max_students_in_row (i : ℕ) : ℕ := 4 + i

def total_max_students : ℕ := ∑ i in Finset.range 15, max_students_in_row (i + 1)

theorem max_students (condition1 : true) : total_max_students = 180 :=
by
  sorry

end max_students_l285_285443


namespace type_B_machine_time_l285_285712

theorem type_B_machine_time :
  (2 * (1 / 5) + 3 * (1 / B) = 5 / 6) → B = 90 / 13 :=
by 
  intro h
  sorry

end type_B_machine_time_l285_285712


namespace find_s_l285_285511

noncomputable def polynomial : Polynomial ℝ :=
  polynomial.monomial 4 1 + polynomial.monomial 3 p + polynomial.monomial 2 q + polynomial.monomial 1 r + polynomial.monomial 0 s

theorem find_s (p q r s : ℝ) (h_roots : ∃ m1 m2 m3 m4 : ℝ, polynomial = (Polynomial.C 1) * (X + m1) * (X + m2) * (X + m3) * (X + m4)) :
  p + q + r + s = 8091 → s = 8064 :=
by
  sorry

end find_s_l285_285511


namespace simplify_product_of_fractions_l285_285101

theorem simplify_product_of_fractions :
  (252 / 21) * (7 / 168) * (12 / 4) = 3 / 2 :=
by
  sorry

end simplify_product_of_fractions_l285_285101


namespace brain_can_always_open_door_l285_285097

noncomputable def can_open_door (a b c n m k : ℕ) : Prop :=
∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3

theorem brain_can_always_open_door :
  ∀ (a b c n m k : ℕ), 
  ∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3 :=
by sorry

end brain_can_always_open_door_l285_285097


namespace interval_strictly_increasing_l285_285597

noncomputable def u : ℝ → ℝ := λ x, 2 * x ^ 2 - 3 * x + 1

noncomputable def f : ℝ → ℝ := λ x, (1/3) ^ u x

theorem interval_strictly_increasing :
  ∀ (x y : ℝ), x < y →
  f x < f y :=
begin
  assume x y hxy,
  change (1/3) ^ u x < (1/3) ^ u y,
  apply (order_iso.pow_lt (by norm_num : 1/3 > 0) (by norm_num : 1/3 < 1)).le_iff_le.mp,
  rw u,
  apply poly_increasing_on_interval_subset
    (λ z, 2 * z ^ 2 - 3 * z + 1) hxy
        (λ a b, by norm_num),
  exact @polynomial.strict_mono_on_of_div_ltx_0 ℝ _ _ (-∞, 3/4)
    (polynomial.monic _
      (λ a, @polynomial.monic_poly_map ℝ _ _ _ _ _ _ _ 0))
    (by norm_num)
end

end interval_strictly_increasing_l285_285597


namespace max_value_f_l285_285470

def f (a x y : ℝ) : ℝ := a * x + y

theorem max_value_f (a : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : |x| + |y| ≤ 1) :
    f a x y ≤ 1 :=
by
  sorry

end max_value_f_l285_285470


namespace div_of_powers_l285_285005

-- Definitions of conditions:
def is_power_of_3 (x : ℕ) := x = 3 ^ 3

-- The conditions for the problem:
variable (a b c : ℕ)
variable (h1 : is_power_of_3 27)
variable (h2 : a = 3)
variable (h3 : b = 12)
variable (h4 : c = 6)

-- The proof statement:
theorem div_of_powers : 3 ^ 12 / 27 ^ 2 = 729 :=
by
  have h₁ : 27 = 3 ^ 3 := h1
  have h₂ : 27 ^ 2 = (3 ^ 3) ^ 2 := by rw [h₁]
  have h₃ : (3 ^ 3) ^ 2 = 3 ^ 6 := by rw [← pow_mul]
  have h₄ : 27 ^ 2 = 3 ^ 6 := by rw [h₂, h₃]
  have h₅ : 3 ^ 12 / 3 ^ 6 = 3 ^ (12 - 6) := by rw [div_eq_mul_inv, ← pow_sub]
  show 3 ^ 6 = 729, from by norm_num
  sorry

end div_of_powers_l285_285005


namespace discriminant_quadratic_eq_l285_285834

-- Given the quadratic equation 5x^2 + 3x - 8
def quadratic_eq (x : ℝ) : ℝ := 5 * x^2 + 3 * x - 8

-- Define the discriminant function for the quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement to prove
theorem discriminant_quadratic_eq :
  discriminant 5 3 (-8) = 169 := by
sorry

end discriminant_quadratic_eq_l285_285834


namespace probability_product_even_or_prime_l285_285750

open Classical
open ProbabilityTheory

noncomputable def probability_even_or_prime : ℚ :=
  let outcomes := (Fin 8) × (Fin 8)
  let events := outcomes.filter (λ pair => 
    let prod := (pair.1.succ * pair.2.succ)
    (prod % 2 = 0) ∨ (Nat.Prime prod))
  events.count / outcomes.count

theorem probability_product_even_or_prime :
  probability_even_or_prime = 7 / 8 :=
by
  sorry

end probability_product_even_or_prime_l285_285750


namespace correct_answer_l285_285641

theorem correct_answer (a b c : ℤ) 
  (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by 
  sorry

end correct_answer_l285_285641


namespace mary_total_nickels_l285_285806

-- Definitions for the conditions
def initial_nickels := 7
def dad_nickels := 5
def mom_nickels := 3 * dad_nickels
def chore_nickels := 2

-- The proof problem statement
theorem mary_total_nickels : 
  initial_nickels + dad_nickels + mom_nickels + chore_nickels = 29 := 
by
  sorry

end mary_total_nickels_l285_285806


namespace cost_of_50_lavenders_l285_285191

noncomputable def cost_of_bouquet (lavenders : ℕ) : ℚ :=
  (25 / 15) * lavenders

theorem cost_of_50_lavenders :
  cost_of_bouquet 50 = 250 / 3 :=
sorry

end cost_of_50_lavenders_l285_285191


namespace car_and_bus_speeds_l285_285028

-- Definitions of given conditions
def car_speed : ℕ := 44
def bus_speed : ℕ := 52

-- Definition of total distance after 4 hours
def total_distance (car_speed bus_speed : ℕ) := 4 * car_speed + 4 * bus_speed

-- Definition of fact that cars started from the same point and traveled in opposite directions
def cars_from_same_point (car_speed bus_speed : ℕ) := car_speed + bus_speed

theorem car_and_bus_speeds :
  total_distance car_speed (car_speed + 8) = 384 :=
by
  -- Proof constructed based on the conditions given
  sorry

end car_and_bus_speeds_l285_285028


namespace vote_majority_is_160_l285_285327

-- Define the total number of votes polled
def total_votes : ℕ := 400

-- Define the percentage of votes polled by the winning candidate
def winning_percentage : ℝ := 0.70

-- Define the percentage of votes polled by the losing candidate
def losing_percentage : ℝ := 0.30

-- Define the number of votes gained by the winning candidate
def winning_votes := winning_percentage * total_votes

-- Define the number of votes gained by the losing candidate
def losing_votes := losing_percentage * total_votes

-- Define the vote majority
def vote_majority := winning_votes - losing_votes

-- Prove that the vote majority is 160 votes
theorem vote_majority_is_160 : vote_majority = 160 :=
sorry

end vote_majority_is_160_l285_285327


namespace flowers_count_l285_285348

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end flowers_count_l285_285348


namespace largest_divisor_of_consecutive_odd_integers_l285_285731

theorem largest_divisor_of_consecutive_odd_integers :
  ∀ (x : ℤ), (∃ (d : ℤ) (m : ℤ), d = 48 ∧ (x * (x + 2) * (x + 4) * (x + 6)) = d * m) :=
by 
-- We assert that for any integer x, 48 always divides the product of
-- four consecutive odd integers starting from x
sorry

end largest_divisor_of_consecutive_odd_integers_l285_285731


namespace smallest_positive_angle_terminal_side_eq_l285_285822

theorem smallest_positive_angle_terminal_side_eq (n : ℤ) :
  (0 ≤ n % 360 ∧ n % 360 < 360) → (∃ k : ℤ, n = -2015 + k * 360 ) → n % 360 = 145 :=
by
  sorry

end smallest_positive_angle_terminal_side_eq_l285_285822


namespace equal_elements_l285_285391

theorem equal_elements (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h_perm : ∃ (σ : Equiv.Perm (Fin 2011)), ∀ i, x' i = x (σ i))
  (h_eq : ∀ i : Fin 2011, x i + x ((i + 1) % 2011) = 2 * x' i) :
  ∀ i j : Fin 2011, x i = x j :=
by
  sorry

end equal_elements_l285_285391


namespace leap_day_2040_is_friday_l285_285342

def leap_day_day_of_week (start_year : ℕ) (start_day : ℕ) (end_year : ℕ) : ℕ :=
  let num_years := end_year - start_year
  let num_leap_years := (num_years + 4) / 4 -- number of leap years including start and end year
  let total_days := 365 * (num_years - num_leap_years) + 366 * num_leap_years
  let day_of_week := (total_days % 7 + start_day) % 7
  day_of_week

theorem leap_day_2040_is_friday :
  leap_day_day_of_week 2008 5 2040 = 5 := 
  sorry

end leap_day_2040_is_friday_l285_285342


namespace inequality_among_three_vars_l285_285664

theorem inequality_among_three_vars 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x + y + z ≥ 3) : 
  (
    1 / (x + y + z ^ 2) + 
    1 / (y + z + x ^ 2) + 
    1 / (z + x + y ^ 2) 
  ) ≤ 1 := 
  sorry

end inequality_among_three_vars_l285_285664


namespace triangle_at_most_one_obtuse_angle_l285_285134

theorem triangle_at_most_one_obtuse_angle :
  (∀ (α β γ : ℝ), α + β + γ = 180 → α ≤ 90 ∨ β ≤ 90 ∨ γ ≤ 90) ↔
  ¬ (∃ (α β γ : ℝ), α + β + γ = 180 ∧ α > 90 ∧ β > 90) :=
by
  sorry

end triangle_at_most_one_obtuse_angle_l285_285134


namespace arithmetic_seq_perfect_sixth_power_l285_285855

theorem arithmetic_seq_perfect_sixth_power 
  (a h : ℤ)
  (seq : ∀ n : ℕ, ℤ)
  (h_seq : ∀ n, seq n = a + n * h)
  (h1 : ∃ s₁ x, seq s₁ = x^2)
  (h2 : ∃ s₂ y, seq s₂ = y^3) :
  ∃ k s, seq s = k^6 := 
sorry

end arithmetic_seq_perfect_sixth_power_l285_285855


namespace probability_of_victory_l285_285782

theorem probability_of_victory (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.6) (independent : true) :
  p_A * p_B = 0.18 :=
by
  -- placeholder for proof
  sorry

end probability_of_victory_l285_285782


namespace exact_time_is_3_07_27_l285_285502

theorem exact_time_is_3_07_27 (t : ℝ) (H1 : t > 0) (H2 : t < 60) 
(H3 : 6 * (t + 8) = 89 + 0.5 * t) : t = 7 + 27/60 :=
by
  sorry

end exact_time_is_3_07_27_l285_285502


namespace greatest_num_consecutive_integers_l285_285008

theorem greatest_num_consecutive_integers (N a : ℤ) (h : (N * (2*a + N - 1) = 210)) :
  ∃ N, N = 210 :=
sorry

end greatest_num_consecutive_integers_l285_285008


namespace find_a_l285_285615

variable {a : ℝ}

def p (a : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁ * x₁ + 2 * a * x₁ + 1 = 0 ∧ x₂ * x₂ + 2 * a * x₂ + 1 = 0

def q (a : ℝ) := ∀ x : ℝ, a * x * x - a * x + 1 > 0 

theorem find_a (a : ℝ) : (p a ∨ q a) ∧ ¬ q a → a ≤ -1 :=
sorry

end find_a_l285_285615


namespace problem_statement_l285_285611

theorem problem_statement (a b : ℝ) (h0 : 0 < b) (h1 : b < 1/2) (h2 : 1/2 < a) (h3 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (ab < a^2) ∧ (a - 1/b < b - 1/a) :=
by 
  sorry

end problem_statement_l285_285611


namespace handshakes_exchanged_l285_285525

-- Let n be the number of couples
noncomputable def num_couples := 7

-- Total number of people at the gathering
noncomputable def total_people := num_couples * 2

-- Number of people each person shakes hands with
noncomputable def handshakes_per_person := total_people - 2

-- Total number of unique handshakes
noncomputable def total_handshakes := total_people * handshakes_per_person / 2

theorem handshakes_exchanged :
  total_handshakes = 77 :=
by
  sorry

end handshakes_exchanged_l285_285525


namespace right_triangle_medians_l285_285501

theorem right_triangle_medians
    (a b c d m : ℝ)
    (h1 : ∀(a b c d : ℝ), 2 * (c/d) = 3)
    (h2 : m = 4 * 3 ∨ m = (3/4)) :
    ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ (m₁ = 12 ∨ m₁ = 3/4) ∧ (m₂ = 12 ∨ m₂ = 3/4) :=
by 
  sorry

end right_triangle_medians_l285_285501


namespace goldfish_below_surface_l285_285693

theorem goldfish_below_surface (Toby_counts_at_surface : ℕ) (percentage_at_surface : ℝ) (total_goldfish : ℕ) (below_surface : ℕ) :
    (Toby_counts_at_surface = 15 ∧ percentage_at_surface = 0.25 ∧ Toby_counts_at_surface = percentage_at_surface * total_goldfish ∧ below_surface = total_goldfish - Toby_counts_at_surface) →
    below_surface = 45 :=
by
  sorry

end goldfish_below_surface_l285_285693


namespace solution_set_of_inequality_l285_285823

theorem solution_set_of_inequality (x : ℝ) : (x^2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := 
by 
  sorry

end solution_set_of_inequality_l285_285823


namespace math_problem_modulo_l285_285276

theorem math_problem_modulo :
    (245 * 15 - 20 * 8 + 5) % 17 = 1 := 
by
  sorry

end math_problem_modulo_l285_285276


namespace distance_apart_l285_285163

def race_total_distance : ℕ := 1000
def distance_Arianna_ran : ℕ := 184

theorem distance_apart :
  race_total_distance - distance_Arianna_ran = 816 :=
by
  sorry

end distance_apart_l285_285163


namespace FI_squared_l285_285194

-- Definitions for the given conditions
-- Note: Further geometric setup and formalization might be necessary to carry 
-- out the complete proof in Lean, but the setup will follow these basic definitions.

-- Let ABCD be a square
def ABCD_square (A B C D : ℝ × ℝ) : Prop :=
  -- conditions for ABCD being a square (to be properly defined based on coordinates and properties)
  sorry

-- Triangle AEH is an equilateral triangle with side length sqrt(3)
def equilateral_AEH (A E H : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A E = s ∧ dist E H = s ∧ dist H A = s 

-- Points E and H lie on AB and DA respectively
-- Points F and G lie on BC and CD respectively
-- Points I and J lie on EH with FI ⊥ EH and GJ ⊥ EH
-- Areas of triangles and quadrilaterals
def geometric_conditions (A B C D E F G H I J : ℝ × ℝ) : Prop :=
  sorry

-- Final statement to prove
theorem FI_squared (A B C D E F G H I J : ℝ × ℝ) (s : ℝ) 
  (h_square: ABCD_square A B C D) 
  (h_equilateral: equilateral_AEH A E H (Real.sqrt 3))
  (h_geo: geometric_conditions A B C D E F G H I J) :
  dist F I ^ 2 = 4 / 3 :=
sorry

end FI_squared_l285_285194


namespace compute_b_l285_285493

theorem compute_b (x y b : ℚ) (h1 : 5 * x - 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hy : y = 3) :
  b = 13 / 2 :=
sorry

end compute_b_l285_285493


namespace units_digit_17_pow_2024_l285_285972

theorem units_digit_17_pow_2024 : (17 ^ 2024) % 10 = 1 := 
by
  sorry

end units_digit_17_pow_2024_l285_285972


namespace swimmers_meet_l285_285004

def time_to_meet (pool_length speed1 speed2 time: ℕ) : ℕ :=
  (time * (speed1 + speed2)) / pool_length

theorem swimmers_meet
  (pool_length : ℕ)
  (speed1 : ℕ)
  (speed2 : ℕ)
  (total_time : ℕ) :
  total_time = 12 * 60 →
  pool_length = 90 →
  speed1 = 3 →
  speed2 = 2 →
  time_to_meet pool_length speed1 speed2 total_time = 20 := by
  sorry

end swimmers_meet_l285_285004


namespace find_original_price_l285_285367

theorem find_original_price (x y : ℝ) 
  (h1 : 60 * x + 75 * y = 2700)
  (h2 : 60 * 0.85 * x + 75 * 0.90 * y = 2370) : 
  x = 20 ∧ y = 20 :=
sorry

end find_original_price_l285_285367


namespace beds_with_fewer_beds_l285_285445

theorem beds_with_fewer_beds:
  ∀ (total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x : ℕ),
    total_rooms = 13 →
    rooms_with_fewer_beds = 8 →
    rooms_with_three_beds = total_rooms - rooms_with_fewer_beds →
    total_beds = 31 →
    8 * x + 3 * (total_rooms - rooms_with_fewer_beds) = total_beds →
    x = 2 :=
by
  intros total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x
  intros ht_rooms hrwb hrwtb htb h_eq
  sorry

end beds_with_fewer_beds_l285_285445


namespace determine_c_l285_285064

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem determine_c (a b : ℝ) (m c : ℝ) 
  (h1 : ∀ x, 0 ≤ x → f x a b = x^2 + a * x + b)
  (h2 : ∃ m : ℝ, ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end determine_c_l285_285064


namespace number_of_zeros_of_f_l285_285766

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017 ^ x + Real.log x / Real.log 2017 else -2017 ^ (-x) - Real.log (-x) / Real.log 2017

theorem number_of_zeros_of_f :
  (∃ x : ℝ, f x = 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) ∧ (∃ x : ℝ, x < 0 ∧ f x = 0) :=
sorry

end number_of_zeros_of_f_l285_285766


namespace cost_rose_bush_l285_285673

-- Define the constants
def total_roses := 6
def friend_roses := 2
def total_aloes := 2
def cost_aloe := 100
def total_spent_self := 500

-- Prove the cost of each rose bush
theorem cost_rose_bush : (total_spent_self - total_aloes * cost_aloe) / (total_roses - friend_roses) = 75 :=
by
  sorry

end cost_rose_bush_l285_285673


namespace exists_segment_with_points_l285_285920

theorem exists_segment_with_points (S : Finset ℕ) (n : ℕ) (hS : S.card = 6 * n)
  (hB : ∃ B : Finset ℕ, B ⊆ S ∧ B.card = 4 * n) (hG : ∃ G : Finset ℕ, G ⊆ S ∧ G.card = 2 * n) :
  ∃ t : Finset ℕ, t ⊆ S ∧ t.card = 3 * n ∧ (∃ B' : Finset ℕ, B' ⊆ t ∧ B'.card = 2 * n) ∧ (∃ G' : Finset ℕ, G' ⊆ t ∧ G'.card = n) :=
  sorry

end exists_segment_with_points_l285_285920


namespace download_speeds_l285_285108

theorem download_speeds (x : ℕ) (s4 : ℕ := 4) (s5 : ℕ := 60) :
  (600 / x - 600 / (15 * x) = 140) → (x = s4 ∧ 15 * x = s5) := by
  sorry

end download_speeds_l285_285108


namespace line_passes_fixed_point_l285_285161

open Real

theorem line_passes_fixed_point
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)
  (M N : ℝ × ℝ)
  (hM : M.1^2 / a^2 + M.2^2 / b^2 = 1)
  (hN : N.1^2 / a^2 + N.2^2 / b^2 = 1)
  (hMAhNA : (M.1 + a) * (N.1 + a) + M.2 * N.2 = 0):
  ∃ (P : ℝ × ℝ), P = (a * (b^2 - a^2) / (a^2 + b^2), 0) ∧ (N.2 - M.2) * (P.1 - M.1) = (P.2 - M.2) * (N.1 - M.1) :=
sorry

end line_passes_fixed_point_l285_285161


namespace net_effect_on_sale_l285_285566

variable (P S : ℝ) (orig_revenue : ℝ := P * S) (new_revenue : ℝ := 0.7 * P * 1.8 * S)

theorem net_effect_on_sale : new_revenue = orig_revenue * 1.26 := by
  sorry

end net_effect_on_sale_l285_285566


namespace determine_a_l285_285606

theorem determine_a (a : ℝ) 
  (h1 : (a - 1) * (0:ℝ)^2 + 0 + a^2 - 1 = 0)
  (h2 : a - 1 ≠ 0) : 
  a = -1 := 
sorry

end determine_a_l285_285606


namespace boxes_of_orange_crayons_l285_285234

theorem boxes_of_orange_crayons
  (n_orange_boxes : ℕ)
  (orange_crayons_per_box : ℕ := 8)
  (blue_boxes : ℕ := 7) (blue_crayons_per_box : ℕ := 5)
  (red_boxes : ℕ := 1) (red_crayons_per_box : ℕ := 11)
  (total_crayons : ℕ := 94)
  (h_total_crayons : (n_orange_boxes * orange_crayons_per_box) + (blue_boxes * blue_crayons_per_box) + (red_boxes * red_crayons_per_box) = total_crayons):
  n_orange_boxes = 6 := 
by sorry

end boxes_of_orange_crayons_l285_285234


namespace glued_cubes_surface_area_l285_285694

theorem glued_cubes_surface_area (L l : ℝ) (h1 : L = 2) (h2 : l = L / 2) : 
  6 * L^2 + 4 * l^2 = 28 :=
by
  sorry

end glued_cubes_surface_area_l285_285694


namespace fraction_simplified_l285_285944

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end fraction_simplified_l285_285944


namespace compare_powers_l285_285300

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end compare_powers_l285_285300


namespace largest_four_digit_perfect_square_l285_285133

theorem largest_four_digit_perfect_square :
  ∃ (n : ℕ), n = 9261 ∧ (∃ k : ℕ, k * k = n) ∧ ∀ (m : ℕ), m < 10000 → (∃ x, x * x = m) → m ≤ n := 
by 
  sorry

end largest_four_digit_perfect_square_l285_285133


namespace induction_base_case_not_necessarily_one_l285_285838

theorem induction_base_case_not_necessarily_one :
  (∀ (P : ℕ → Prop) (n₀ : ℕ), (P n₀) → (∀ n, n ≥ n₀ → P n → P (n + 1)) → ∀ n, n ≥ n₀ → P n) ↔
  (∃ n₀ : ℕ, n₀ ≠ 1) :=
sorry

end induction_base_case_not_necessarily_one_l285_285838


namespace total_worth_of_produce_is_630_l285_285609

def bundles_of_asparagus : ℕ := 60
def price_per_bundle_asparagus : ℝ := 3.00

def boxes_of_grapes : ℕ := 40
def price_per_box_grapes : ℝ := 2.50

def num_apples : ℕ := 700
def price_per_apple : ℝ := 0.50

def total_worth : ℝ :=
  bundles_of_asparagus * price_per_bundle_asparagus +
  boxes_of_grapes * price_per_box_grapes +
  num_apples * price_per_apple

theorem total_worth_of_produce_is_630 : 
  total_worth = 630 := by
  sorry

end total_worth_of_produce_is_630_l285_285609


namespace ordered_triples_unique_solution_l285_285072

theorem ordered_triples_unique_solution :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ a + b + c = 2 :=
sorry

end ordered_triples_unique_solution_l285_285072


namespace molecular_weight_correct_l285_285698

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms
def num_N : ℕ := 2
def num_O : ℕ := 3

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 76.02

-- The theorem to prove
theorem molecular_weight_correct :
  (num_N * atomic_weight_N + num_O * atomic_weight_O) = expected_molecular_weight := 
by
  sorry

end molecular_weight_correct_l285_285698


namespace Lin_trip_time_l285_285205

theorem Lin_trip_time
  (v : ℕ) -- speed on the mountain road in miles per minute
  (h1 : 80 = d_highway) -- Lin travels 80 miles on the highway
  (h2 : 20 = d_mountain) -- Lin travels 20 miles on the mountain road
  (h3 : v_highway = 2 * v) -- Lin drives twice as fast on the highway
  (h4 : 40 = 20 / v) -- Lin spent 40 minutes driving on the mountain road
  : 40 + 80 = 120 :=
by
  -- proof steps would go here
  sorry

end Lin_trip_time_l285_285205


namespace find_exponent_l285_285904

theorem find_exponent (n : ℕ) (some_number : ℕ) (h1 : n = 27) 
  (h2 : 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) = 4 ^ some_number) :
  some_number = 28 :=
by 
  sorry

end find_exponent_l285_285904


namespace four_times_num_mod_nine_l285_285429

theorem four_times_num_mod_nine (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 :=
sorry

end four_times_num_mod_nine_l285_285429


namespace volume_of_prism_in_cubic_feet_l285_285071

theorem volume_of_prism_in_cubic_feet:
  let length_yd := 1
  let width_yd := 2
  let height_yd := 3
  let yard_to_feet := 3
  let length_ft := length_yd * yard_to_feet
  let width_ft := width_yd * yard_to_feet
  let height_ft := height_yd * yard_to_feet
  let volume := length_ft * width_ft * height_ft
  volume = 162 := by
  sorry

end volume_of_prism_in_cubic_feet_l285_285071


namespace problem_solution_l285_285890

theorem problem_solution :
  let m := 9
  let n := 20
  let lhs := (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)
  let rhs := 9 / 20
  lhs = rhs → 10 * m + n = 110 :=
by sorry

end problem_solution_l285_285890


namespace f_iterated_l285_285744

noncomputable def f (z : ℂ) : ℂ :=
  if (∃ r : ℝ, z = r) then -z^2 else z^2

theorem f_iterated (z : ℂ) (h : z = 2 + 1 * complex.i) : 
  f (f (f (f z))) = 164833 + 354192 * complex.i :=
by 
  subst h
  sorry

end f_iterated_l285_285744


namespace expression_value_l285_285836

theorem expression_value (a b : ℕ) (h1 : a = 36) (h2 : b = 9) : (a + b)^2 - (b^2 + a^2) = 648 :=
by {
  rw [h1, h2],
  norm_num
}

end expression_value_l285_285836


namespace bricks_required_l285_285261

theorem bricks_required (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
  (brick_length_cm : ℕ) (brick_width_cm : ℕ)
  (h1 : courtyard_length_m = 30) (h2 : courtyard_width_m = 16)
  (h3 : brick_length_cm = 20) (h4 : brick_width_cm = 10) :
  (3000 * 1600) / (20 * 10) = 24000 :=
by sorry

end bricks_required_l285_285261


namespace find_r_l285_285884

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 20.7) : r = 10.7 := 
by 
  sorry 

end find_r_l285_285884


namespace pump_out_time_l285_285218

theorem pump_out_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (rate : ℝ)
  (H_length : length = 50)
  (H_width : width = 30)
  (H_depth : depth = 1.8)
  (H_rate : rate = 2.5) : 
  (length * width * depth) / rate / 60 = 18 :=
by
  sorry

end pump_out_time_l285_285218


namespace third_person_profit_share_l285_285295

noncomputable def investment_first : ℤ := 9000
noncomputable def investment_second : ℤ := investment_first + 2000
noncomputable def investment_third : ℤ := investment_second - 3000
noncomputable def investment_fourth : ℤ := 2 * investment_third
noncomputable def investment_fifth : ℤ := investment_fourth + 4000
noncomputable def total_investment : ℤ := investment_first + investment_second + investment_third + investment_fourth + investment_fifth

noncomputable def total_profit : ℤ := 25000
noncomputable def third_person_share : ℚ := (investment_third : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem third_person_profit_share :
  third_person_share = 3076.92 := sorry

end third_person_profit_share_l285_285295


namespace inequality_proof_l285_285895

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end inequality_proof_l285_285895


namespace compute_fraction_l285_285595

theorem compute_fraction :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) =
  1 / 4 :=
by
  -- Proof will go here
  sorry

end compute_fraction_l285_285595


namespace james_coursework_materials_expense_l285_285657

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end james_coursework_materials_expense_l285_285657


namespace logarithmic_relationship_l285_285057

theorem logarithmic_relationship
  (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (h5 : m = Real.log c / Real.log a)
  (h6 : n = Real.log c / Real.log b)
  (h7 : r = a ^ c) :
  n < m ∧ m < r :=
sorry

end logarithmic_relationship_l285_285057


namespace min_solution_l285_285468

theorem min_solution :
  ∀ (x : ℝ), (min (1 / (1 - x)) (2 / (1 - x)) = 2 / (x - 1) - 3) → x = 7 / 3 := 
by
  sorry

end min_solution_l285_285468


namespace arith_seq_S13_value_l285_285177

variable {α : Type*} [LinearOrderedField α]

-- Definitions related to an arithmetic sequence
structure ArithSeq (α : Type*) :=
  (a : ℕ → α) -- the sequence itself
  (sum_first_n_terms : ℕ → α) -- sum of the first n terms

def is_arith_seq (seq : ArithSeq α) :=
  ∀ (n : ℕ), seq.a (n + 1) - seq.a n = seq.a 2 - seq.a 1

-- Our conditions
noncomputable def a5 (seq : ArithSeq α) := seq.a 5
noncomputable def a7 (seq : ArithSeq α) := seq.a 7
noncomputable def a9 (seq : ArithSeq α) := seq.a 9
noncomputable def S13 (seq : ArithSeq α) := seq.sum_first_n_terms 13

-- Problem statement
theorem arith_seq_S13_value (seq : ArithSeq α) 
  (h_arith_seq : is_arith_seq seq)
  (h_condition : 2 * (a5 seq) + 3 * (a7 seq) + 2 * (a9 seq) = 14) : 
  S13 seq = 26 := 
  sorry

end arith_seq_S13_value_l285_285177


namespace joe_paint_problem_l285_285087

theorem joe_paint_problem (f : ℝ) (h₁ : 360 * f + (1 / 6) * (360 - 360 * f) = 135) : f = 1 / 4 := 
by
  sorry

end joe_paint_problem_l285_285087


namespace ellipse_focus_value_k_l285_285111

theorem ellipse_focus_value_k 
  (k : ℝ)
  (h : ∀ x y, 5 * x^2 + k * y^2 = 5 → abs y ≠ 2 → ∀ c : ℝ, c^2 = 4 → k = 1) :
  ∀ k : ℝ, (5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5) ∧ (5 * (0:ℝ)^2 + k * (-(2:ℝ))^2 = 5) → k = 1 := by
  sorry

end ellipse_focus_value_k_l285_285111


namespace third_vertex_coordinates_l285_285829

theorem third_vertex_coordinates (x : ℝ) (h : 6 * |x| = 96) : x = 16 ∨ x = -16 :=
by
  sorry

end third_vertex_coordinates_l285_285829


namespace Clea_ride_time_l285_285503

theorem Clea_ride_time
  (c s d t : ℝ)
  (h1 : d = 80 * c)
  (h2 : d = 30 * (c + s))
  (h3 : s = 5 / 3 * c)
  (h4 : t = d / s) :
  t = 48 := by sorry

end Clea_ride_time_l285_285503


namespace cups_needed_correct_l285_285148

-- Define the conditions
def servings : ℝ := 18.0
def cups_per_serving : ℝ := 2.0

-- Define the total cups needed calculation
def total_cups (servings : ℝ) (cups_per_serving : ℝ) : ℝ :=
  servings * cups_per_serving

-- State the proof problem
theorem cups_needed_correct :
  total_cups servings cups_per_serving = 36.0 :=
by
  sorry

end cups_needed_correct_l285_285148


namespace abs_value_solutions_l285_285224

theorem abs_value_solutions (x : ℝ) : abs x = 6.5 ↔ x = 6.5 ∨ x = -6.5 :=
by
  sorry

end abs_value_solutions_l285_285224


namespace intersection_A_B_l285_285508

open Set

def A : Set ℝ := Icc 1 2

def B : Set ℤ := {x : ℤ | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B :
  (A ∩ (coe '' B) : Set ℝ) = {1, 2} :=
sorry

end intersection_A_B_l285_285508


namespace solution_form_l285_285251

noncomputable def required_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) ≤ (x * f y + y * f x) / 2

theorem solution_form (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 < x → 0 < f x) : required_function f → ∃ a : ℝ, 0 < a ∧ ∀ x : ℝ, 0 < x → f x = a * x :=
by
  intros
  sorry

end solution_form_l285_285251


namespace twentieth_fisherman_catch_l285_285000

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l285_285000


namespace perpendicular_vectors_l285_285624

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l285_285624


namespace prime_implies_power_of_two_l285_285949

-- Conditions:
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Problem:
theorem prime_implies_power_of_two (n : ℕ) (h : is_prime (2^n + 1)) : ∃ k : ℕ, n = 2^k := sorry

end prime_implies_power_of_two_l285_285949


namespace danny_initial_caps_l285_285872

-- Define the conditions
variables (lostCaps : ℕ) (currentCaps : ℕ)
-- Assume given conditions
axiom lost_caps_condition : lostCaps = 66
axiom current_caps_condition : currentCaps = 25

-- Define the total number of bottle caps Danny had at first
def originalCaps (lostCaps currentCaps : ℕ) : ℕ := lostCaps + currentCaps

-- State the theorem to prove the number of bottle caps Danny originally had is 91
theorem danny_initial_caps : originalCaps lostCaps currentCaps = 91 :=
by
  -- Insert the proof here when available
  sorry

end danny_initial_caps_l285_285872


namespace percentage_error_divide_instead_of_multiply_l285_285250

theorem percentage_error_divide_instead_of_multiply (x : ℝ) : 
  let correct_result := 5 * x 
  let incorrect_result := x / 10 
  let error := correct_result - incorrect_result 
  let percentage_error := (error / correct_result) * 100 
  percentage_error = 98 :=
by
  sorry

end percentage_error_divide_instead_of_multiply_l285_285250


namespace square_of_binomial_l285_285640

theorem square_of_binomial (a : ℝ) : 16 * x^2 + 32 * x + a = (4 * x + 4)^2 :=
by
  sorry

end square_of_binomial_l285_285640


namespace gear_ratio_l285_285783

variable (a b c : ℕ) (ωG ωH ωI : ℚ)

theorem gear_ratio :
  (a * ωG = b * ωH) ∧ (b * ωH = c * ωI) ∧ (a * ωG = c * ωI) →
  ωG / ωH = bc / ac ∧ ωH / ωI = ac / ab ∧ ωG / ωI = bc / ab :=
by
  sorry

end gear_ratio_l285_285783


namespace range_k_l285_285774

theorem range_k (k : ℝ) :
  (∀ x : ℝ, (3/8 - k*x - 2*k*x^2) ≥ 0) ↔ (-3 ≤ k ∧ k ≤ 0) :=
sorry

end range_k_l285_285774


namespace solve_special_sine_system_l285_285705

noncomputable def special_sine_conditions1 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  let z := -(Real.pi / 2) + 2 * Real.pi * k
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = (-1)^n * Real.pi / 6 + Real.pi * n ∧
  z = -Real.pi / 2 + 2 * Real.pi * k

noncomputable def special_sine_conditions2 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := -Real.pi / 2 + 2 * Real.pi * k
  let z := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = -Real.pi / 2 + 2 * Real.pi * k ∧
  z = (-1)^n * Real.pi / 6 + Real.pi * n

theorem solve_special_sine_system (m n k : ℤ) :
  special_sine_conditions1 m n k ∨ special_sine_conditions2 m n k :=
sorry

end solve_special_sine_system_l285_285705


namespace find_m_l285_285631

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l285_285631


namespace combined_room_size_l285_285546

theorem combined_room_size (M J S : ℝ) 
  (h1 : M + J + S = 800) 
  (h2 : J = M + 100) 
  (h3 : S = M - 50) : 
  J + S = 550 := 
by
  sorry

end combined_room_size_l285_285546


namespace scientific_notation_of_8200000_l285_285014

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l285_285014


namespace compute_expression_l285_285725

theorem compute_expression :
  23 ^ 12 / 23 ^ 5 + 5 = 148035894 :=
  sorry

end compute_expression_l285_285725


namespace value_added_to_each_number_is_11_l285_285918

-- Given definitions and conditions
def initial_average : ℝ := 40
def number_count : ℕ := 15
def new_average : ℝ := 51

-- Mathematically equivalent proof statement
theorem value_added_to_each_number_is_11 (x : ℝ) 
  (h1 : number_count * initial_average = 600)
  (h2 : (600 + number_count * x) / number_count = new_average) : 
  x = 11 := 
by 
  sorry

end value_added_to_each_number_is_11_l285_285918


namespace number_of_triangles_l285_285891

theorem number_of_triangles (m : ℕ) (h : m > 0) :
  ∃ n : ℕ, n = (m * (m + 1)) / 2 :=
by sorry

end number_of_triangles_l285_285891


namespace remainder_of_n_plus_4500_l285_285186

theorem remainder_of_n_plus_4500 (n : ℕ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := 
by
  sorry

end remainder_of_n_plus_4500_l285_285186


namespace root_range_m_l285_285908

theorem root_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * m * x + 4 = 0 → (x > 1 ∧ ∃ y : ℝ, y < 1 ∧ y^2 - 2 * m * y + 4 = 0)
  ∨ (x < 1 ∧ ∃ y : ℝ, y > 1 ∧ y^2 - 2 * m * y + 4 = 0))
  → m > 5 / 2 := 
sorry

end root_range_m_l285_285908


namespace quadratic_function_properties_l285_285305

theorem quadratic_function_properties :
  ∃ a : ℝ, ∃ f : ℝ → ℝ,
    (∀ x : ℝ, f x = a * (x + 1) ^ 2 - 2) ∧
    (f 1 = 10) ∧
    (f (-1) = -2) ∧
    (∀ x : ℝ, x > -1 → f x ≥ f (-1))
:=
by
  sorry

end quadratic_function_properties_l285_285305


namespace problem_statement_l285_285463

theorem problem_statement (a n : ℕ) (h1 : 1 ≤ a) (h2 : n = 1) : ∃ m : ℤ, ((a + 1)^n - a^n) = m * n := by
  sorry

end problem_statement_l285_285463


namespace find_interest_rate_l285_285366

theorem find_interest_rate
  (P : ℝ) (CI : ℝ) (T : ℝ) (n : ℕ)
  (comp_int_formula : CI = P * ((1 + (r / (n : ℝ))) ^ (n * T)) - P) :
  r = 0.099 :=
by
  have h : CI = 788.13 := sorry
  have hP : P = 5000 := sorry
  have hT : T = 1.5 := sorry
  have hn : (n : ℝ) = 2 := sorry
  sorry

end find_interest_rate_l285_285366


namespace flowers_count_l285_285347

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end flowers_count_l285_285347


namespace triangle_def_ef_value_l285_285789

theorem triangle_def_ef_value (E F D : ℝ) (DE DF EF : ℝ) (h1 : E = 45)
  (h2 : DE = 100) (h3 : DF = 100 * Real.sqrt 2) :
  EF = Real.sqrt (30000 + 5000*(Real.sqrt 6 - Real.sqrt 2)) := 
sorry 

end triangle_def_ef_value_l285_285789


namespace train_more_passengers_l285_285440

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end train_more_passengers_l285_285440


namespace sum_of_three_consecutive_integers_divisible_by_3_l285_285537

theorem sum_of_three_consecutive_integers_divisible_by_3 (a : ℤ) :
  ∃ k : ℤ, k = 3 ∧ (a - 1 + a + (a + 1)) % k = 0 :=
by
  use 3
  sorry

end sum_of_three_consecutive_integers_divisible_by_3_l285_285537


namespace find_fraction_l285_285407

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end find_fraction_l285_285407


namespace cubic_polynomial_k_l285_285663

noncomputable def h (x : ℝ) : ℝ := x^3 - x - 2

theorem cubic_polynomial_k (k : ℝ → ℝ)
  (hk : ∃ (B : ℝ), ∀ (x : ℝ), k x = B * (x - (root1 ^ 2)) * (x - (root2 ^ 2)) * (x - (root3 ^ 2)))
  (hroots : h (root1) = 0 ∧ h (root2) = 0 ∧ h (root3) = 0)
  (h_values : k 0 = 2) :
  k (-8) = -20 :=
sorry

end cubic_polynomial_k_l285_285663


namespace total_cost_of_repair_l285_285507

noncomputable def cost_of_repair (tire_cost: ℝ) (num_tires: ℕ) (tax: ℝ) (city_fee: ℝ) (discount: ℝ) : ℝ :=
  let total_cost := (tire_cost * num_tires : ℝ)
  let total_tax := (tax * num_tires : ℝ)
  let total_city_fee := (city_fee * num_tires : ℝ)
  (total_cost + total_tax + total_city_fee - discount)

def car_A_tire_cost : ℝ := 7
def car_A_num_tires : ℕ := 3
def car_A_tax : ℝ := 0.5
def car_A_city_fee : ℝ := 2.5
def car_A_discount : ℝ := (car_A_tire_cost * car_A_num_tires) * 0.05

def car_B_tire_cost : ℝ := 8.5
def car_B_num_tires : ℕ := 2
def car_B_tax : ℝ := 0 -- no sales tax
def car_B_city_fee : ℝ := 2.5
def car_B_discount : ℝ := 0 -- expired coupon

theorem total_cost_of_repair : 
  cost_of_repair car_A_tire_cost car_A_num_tires car_A_tax car_A_city_fee car_A_discount + 
  cost_of_repair car_B_tire_cost car_B_num_tires car_B_tax car_B_city_fee car_B_discount = 50.95 :=
by
  sorry

end total_cost_of_repair_l285_285507


namespace probability_of_purple_is_one_fifth_l285_285106

-- Definitions related to the problem
def total_faces : ℕ := 10
def purple_faces : ℕ := 2
def probability_purple := (purple_faces : ℚ) / (total_faces : ℚ)

theorem probability_of_purple_is_one_fifth : probability_purple = 1 / 5 := 
by
  -- Converting the numbers to rationals explicitly ensures division is defined.
  change (2 : ℚ) / (10 : ℚ) = 1 / 5
  norm_num
  -- sorry (if finishing the proof manually isn't desired)

end probability_of_purple_is_one_fifth_l285_285106


namespace eval_p_positive_int_l285_285599

theorem eval_p_positive_int (p : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ (4 * p + 20) = n * (3 * p - 6)) ↔ p = 3 ∨ p = 4 ∨ p = 15 ∨ p = 28 := 
by sorry

end eval_p_positive_int_l285_285599


namespace james_coursework_materials_expense_l285_285656

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end james_coursework_materials_expense_l285_285656


namespace old_clock_slow_by_12_minutes_l285_285538

theorem old_clock_slow_by_12_minutes (overlap_interval: ℕ) (standard_day_minutes: ℕ)
  (h1: overlap_interval = 66) (h2: standard_day_minutes = 24 * 60):
  standard_day_minutes - 24 * 60 / 66 * 66 = 12 :=
by
  sorry

end old_clock_slow_by_12_minutes_l285_285538


namespace number_of_n_values_l285_285296

-- Definition of sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

-- The main statement to prove
theorem number_of_n_values : 
  ∃ M, M = 8 ∧ ∀ n : ℕ, (n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2010) → M = 8 :=
by
  sorry

end number_of_n_values_l285_285296


namespace Susan_total_peaches_l285_285107

-- Define the number of peaches in the knapsack
def peaches_in_knapsack : ℕ := 12

-- Define the condition that the number of peaches in the knapsack is half the number of peaches in each cloth bag
def peaches_per_cloth_bag (x : ℕ) : Prop := peaches_in_knapsack * 2 = x

-- Define the total number of peaches Susan bought
def total_peaches (x : ℕ) : ℕ := x + 2 * x

-- Theorem statement: Prove that the total number of peaches Susan bought is 60
theorem Susan_total_peaches (x : ℕ) (h : peaches_per_cloth_bag x) : total_peaches peaches_in_knapsack = 60 := by
  sorry

end Susan_total_peaches_l285_285107


namespace negative_three_degrees_below_zero_l285_285331

-- Definitions based on conditions
def positive_temperature (t : ℤ) : Prop := t > 0
def negative_temperature (t : ℤ) : Prop := t < 0
def above_zero (t : ℤ) : Prop := positive_temperature t
def below_zero (t : ℤ) : Prop := negative_temperature t

-- Example given in conditions
def ten_degrees_above_zero := above_zero 10

-- Lean 4 statement for the proof
theorem negative_three_degrees_below_zero : below_zero (-3) :=
by
  sorry

end negative_three_degrees_below_zero_l285_285331


namespace total_pencils_correct_l285_285034

def initial_pencils : ℕ := 245
def added_pencils : ℕ := 758
def total_pencils : ℕ := initial_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 1003 := 
by
  sorry

end total_pencils_correct_l285_285034


namespace solve_basketball_court_dimensions_l285_285286

theorem solve_basketball_court_dimensions 
  (A B C D E F : ℕ) 
  (h1 : A - B = C) 
  (h2 : D = 2 * (A + B)) 
  (h3 : E = A * B) 
  (h4 : F = 3) : 
  A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 := 
by 
  sorry

end solve_basketball_court_dimensions_l285_285286


namespace time_to_fill_tank_with_two_pipes_simultaneously_l285_285845

def PipeA : ℝ := 30
def PipeB : ℝ := 45

theorem time_to_fill_tank_with_two_pipes_simultaneously :
  let A := 1 / PipeA
  let B := 1 / PipeB
  let combined_rate := A + B
  let time_to_fill_tank := 1 / combined_rate
  time_to_fill_tank = 18 := 
by
  sorry

end time_to_fill_tank_with_two_pipes_simultaneously_l285_285845


namespace calc1_calc2_l285_285722

theorem calc1 : (1 * -11 + 8 + (-14) = -17) := by
  sorry

theorem calc2 : (13 - (-12) + (-21) = 4) := by
  sorry

end calc1_calc2_l285_285722


namespace inequalities_consistent_l285_285099

theorem inequalities_consistent (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1) ^ 2) (h3 : y * (y - 1) ≤ x ^ 2) : true := 
by 
  sorry

end inequalities_consistent_l285_285099


namespace triangle_formation_inequalities_l285_285677

theorem triangle_formation_inequalities (a b c d : ℝ)
  (h_abc_pos : 0 < a)
  (h_bcd_pos : 0 < b)
  (h_cde_pos : 0 < c)
  (h_def_pos : 0 < d)
  (tri_ineq_1 : a + b + c > d)
  (tri_ineq_2 : b + c + d > a)
  (tri_ineq_3 : a + d > b + c) :
  (a < (b + c + d) / 2) ∧ (b + c < a + d) ∧ (¬ (c + d < b / 2)) :=
by 
  sorry

end triangle_formation_inequalities_l285_285677


namespace nice_sequence_max_length_l285_285200

theorem nice_sequence_max_length (n : ℕ) (hn : 1 ≤ n) :
  ∃ (a : ℕ → ℝ), (a 0 + a 1 = -1 / n) ∧
  (∀ k ≥ 1, (a (k - 1) + a k) * (a k + a (k + 1)) = a (k - 1) + a (k + 1)) ∧
  ∀ b, b = n → ¬∃ (a' : ℕ → ℝ), (a' 0 + a' 1 = -1 / n) ∧
  (∀ k ≥ 1, (a' (k - 1) + a' k) * (a' k + a' (k + 1)) = a' (k - 1) + a' (k + 1)) ∧
  ∃ m, m = b + 1 := sorry

end nice_sequence_max_length_l285_285200


namespace find_m_l285_285630

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l285_285630


namespace scientific_notation_correctness_l285_285011

theorem scientific_notation_correctness : ∃ x : ℝ, x = 8.2 ∧ (8200000 : ℝ) = x * 10^6 :=
by
  use 8.2
  split
  · rfl
  · sorry

end scientific_notation_correctness_l285_285011


namespace rectangle_horizontal_length_l285_285352

variable (squareside rectheight : ℕ)

-- Condition: side of the square is 80 cm, vertical side length of the rectangle is 100 cm
def square_side_length := 80
def rect_vertical_length := 100

-- Question: Calculate the horizontal length of the rectangle
theorem rectangle_horizontal_length :
  (4 * square_side_length) = (2 * rect_vertical_length + 2 * rect_horizontal_length) -> rect_horizontal_length = 60 := by
  sorry

end rectangle_horizontal_length_l285_285352


namespace current_at_time_l285_285228

noncomputable def I (t : ℝ) : ℝ := 5 * (Real.sin (100 * Real.pi * t + Real.pi / 3))

theorem current_at_time (t : ℝ) (h : t = 1 / 200) : I t = 5 / 2 := by
  sorry

end current_at_time_l285_285228


namespace factor_expression_l285_285594

theorem factor_expression:
  ∀ (x : ℝ), (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by
  sorry

end factor_expression_l285_285594


namespace inscribed_sphere_radius_l285_285048

theorem inscribed_sphere_radius (a α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (ρ : ℝ), ρ = a * (1 - Real.cos α) / (2 * Real.sqrt (1 + Real.cos α) * (1 + Real.sqrt (- Real.cos α))) :=
  sorry

end inscribed_sphere_radius_l285_285048


namespace binom_1450_2_eq_1050205_l285_285277

def binom_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_1450_2_eq_1050205 : binom_coefficient 1450 2 = 1050205 :=
by {
  sorry
}

end binom_1450_2_eq_1050205_l285_285277


namespace trihedral_sphere_radius_l285_285326

noncomputable def sphere_radius 
  (α r : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  : ℝ :=
r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3)

theorem trihedral_sphere_radius 
  (α r R : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  (hR : R = sphere_radius α r hα) 
  : R = r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3) :=
by
  sorry

end trihedral_sphere_radius_l285_285326


namespace bus_minibus_seats_l285_285118

theorem bus_minibus_seats (x y : ℕ) 
    (h1 : x = y + 20) 
    (h2 : 5 * x + 5 * y = 300) : 
    x = 40 ∧ y = 20 := 
by
  sorry

end bus_minibus_seats_l285_285118


namespace reduction_for_1750_yuan_max_daily_profit_not_1900_l285_285365

def average_shirts_per_day : ℕ := 40 
def profit_per_shirt_initial : ℕ := 40 
def price_reduction_increase_shirts (reduction : ℝ) : ℝ := reduction * 2 
def daily_profit (reduction : ℝ) : ℝ := (profit_per_shirt_initial - reduction) * (average_shirts_per_day + price_reduction_increase_shirts reduction)

-- Part 1: Proving the reduction that results in 1750 yuan profit
theorem reduction_for_1750_yuan : ∃ x : ℝ, daily_profit x = 1750 ∧ x = 15 := 
by {
  sorry
}

-- Part 2: Proving that the maximum cannot reach 1900 yuan
theorem max_daily_profit_not_1900 : ∀ x : ℝ, daily_profit x ≤ 1800 ∧ (∀ y : ℝ, y ≥ daily_profit x → y < 1900) :=
by {
  sorry
}

end reduction_for_1750_yuan_max_daily_profit_not_1900_l285_285365


namespace find_a61_l285_285759

def seq (a : ℕ → ℕ) : Prop :=
  (∀ n, a (2 * n + 1) = a n + a (n + 1)) ∧
  (∀ n, a (2 * n) = a n) ∧
  a 1 = 1

theorem find_a61 (a : ℕ → ℕ) (h : seq a) : a 61 = 9 :=
by
  sorry

end find_a61_l285_285759


namespace total_tickets_sold_l285_285585

theorem total_tickets_sold (x y : ℕ) (h1 : 12 * x + 8 * y = 3320) (h2 : y = x + 240) : 
  x + y = 380 :=
by -- proof
  sorry

end total_tickets_sold_l285_285585


namespace scientific_notation_of_8200000_l285_285015

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l285_285015


namespace sequence_arithmetic_l285_285758

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * n^2 - 2 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (∀ n, a n - a (n - 1) = 4) :=
by
  intros hS ha
  sorry

end sequence_arithmetic_l285_285758


namespace boris_possible_amount_l285_285995

theorem boris_possible_amount (k : ℕ) : ∃ k : ℕ, 1 + 74 * k = 823 :=
by
  use 11
  sorry

end boris_possible_amount_l285_285995


namespace count_true_propositions_l285_285151

theorem count_true_propositions :
  let prop1 := false  -- Proposition ① is false
  let prop2 := true   -- Proposition ② is true
  let prop3 := true   -- Proposition ③ is true
  let prop4 := false  -- Proposition ④ is false
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) +
  (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 :=
by
  -- The theorem is expected to be proven here
  sorry

end count_true_propositions_l285_285151


namespace investment_rate_l285_285447

theorem investment_rate (P_total P_7000 P_15000 I_total : ℝ)
  (h_investment : P_total = 22000)
  (h_investment_7000 : P_7000 = 7000)
  (h_investment_15000 : P_15000 = P_total - P_7000)
  (R_7000 : ℝ)
  (h_rate_7000 : R_7000 = 0.18)
  (I_7000 : ℝ)
  (h_interest_7000 : I_7000 = P_7000 * R_7000)
  (h_total_interest : I_total = 3360) :
  ∃ (R_15000 : ℝ), (I_total - I_7000) = P_15000 * R_15000 ∧ R_15000 = 0.14 := 
by
  sorry

end investment_rate_l285_285447


namespace opposite_of_neg_three_l285_285389

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l285_285389


namespace quadratic_roots_prime_distinct_l285_285494

theorem quadratic_roots_prime_distinct (a α β m : ℕ) (h1: α ≠ β) (h2: Nat.Prime α) (h3: Nat.Prime β) (h4: α + β = m / a) (h5: α * β = 1996 / a) :
    a = 2 := by
  sorry

end quadratic_roots_prime_distinct_l285_285494


namespace find_k_l285_285069

theorem find_k (a b : ℤ × ℤ) (k : ℤ) 
  (h₁ : a = (2, 1)) 
  (h₂ : a.1 + b.1 = 1 ∧ a.2 + b.2 = k)
  (h₃ : a.1 * b.1 + a.2 * b.2 = 0) : k = 3 :=
sorry

end find_k_l285_285069


namespace expression_simplified_l285_285047

noncomputable def expression : ℚ := 1 + 3 / (4 + 5 / 6)

theorem expression_simplified : expression = 47 / 29 :=
by
  sorry

end expression_simplified_l285_285047


namespace proof_problem_l285_285422

-- Definition for the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def probability (b : ℕ) : ℚ :=
  (binom (40 - b) 2 + binom (b - 1) 2 : ℚ) / 1225

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def minimum_b (b : ℕ) : Prop :=
  b = 11 ∧ probability 11 = 857 / 1225 ∧ is_coprime 857 1225 ∧ 857 + 1225 = 2082

-- Statement to prove
theorem proof_problem : ∃ b, minimum_b b := 
by
  -- Lean statement goes here
  sorry

end proof_problem_l285_285422


namespace f_minimum_at_l285_285307

noncomputable def f (x : ℝ) : ℝ := x * 2^x

theorem f_minimum_at : ∀ x : ℝ, x = -Real.log 2 → (∀ y : ℝ, f y ≥ f x) :=
by
  sorry

end f_minimum_at_l285_285307


namespace simplify_expression_l285_285950

theorem simplify_expression (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + ((a + 1) / (1 - a)))) = (1 + a) / 2 := 
by
  sorry

end simplify_expression_l285_285950


namespace probability_task1_on_time_and_task2_not_on_time_l285_285139

theorem probability_task1_on_time_and_task2_not_on_time:
  let P_T1 := (2:ℚ) / 3
  let P_T2 := (3:ℚ) / 5
  let P_not_T2 := 1 - P_T2
  let P_T1_and_not_T2 := P_T1 * P_not_T2
  P_T1_and_not_T2 = (4:ℚ) / 15 := by
sorry

end probability_task1_on_time_and_task2_not_on_time_l285_285139


namespace mean_points_scored_l285_285717

def Mrs_Williams_points : ℝ := 50
def Mr_Adams_points : ℝ := 57
def Mrs_Browns_points : ℝ := 49
def Mrs_Daniels_points : ℝ := 57

def total_points : ℝ := Mrs_Williams_points + Mr_Adams_points + Mrs_Browns_points + Mrs_Daniels_points
def number_of_classes : ℝ := 4

theorem mean_points_scored :
  (total_points / number_of_classes) = 53.25 :=
by
  sorry

end mean_points_scored_l285_285717


namespace problem_statement_l285_285462

theorem problem_statement (a n : ℕ) (h1 : 1 ≤ a) (h2 : n = 1) : ∃ m : ℤ, ((a + 1)^n - a^n) = m * n := by
  sorry

end problem_statement_l285_285462


namespace tetrahedron_volume_minimum_l285_285679

theorem tetrahedron_volume_minimum (h1 h2 h3 : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (h3_pos : 0 < h3) :
  ∃ V : ℝ, V ≥ (1/3) * (h1 * h2 * h3) :=
sorry

end tetrahedron_volume_minimum_l285_285679


namespace lilly_can_buy_flowers_l285_285349

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end lilly_can_buy_flowers_l285_285349


namespace total_chess_games_l285_285231

theorem total_chess_games (n : ℕ) (h_n : n = 9) : (nat.choose n 2) = 36 :=
by
  rw h_n
  sorry

end total_chess_games_l285_285231


namespace dog_food_cans_l285_285271

theorem dog_food_cans 
  (packages_cat_food : ℕ)
  (cans_per_package_cat_food : ℕ)
  (packages_dog_food : ℕ)
  (additional_cans_cat_food : ℕ)
  (total_cans_cat_food : ℕ)
  (total_cans_dog_food : ℕ)
  (num_cans_dog_food_package : ℕ) :
  packages_cat_food = 9 →
  cans_per_package_cat_food = 10 →
  packages_dog_food = 7 →
  additional_cans_cat_food = 55 →
  total_cans_cat_food = packages_cat_food * cans_per_package_cat_food →
  total_cans_dog_food = packages_dog_food * num_cans_dog_food_package →
  total_cans_cat_food = total_cans_dog_food + additional_cans_cat_food →
  num_cans_dog_food_package = 5 :=
by
  sorry

end dog_food_cans_l285_285271


namespace binom_18_6_mul_smallest_prime_gt_10_eq_80080_l285_285871

theorem binom_18_6_mul_smallest_prime_gt_10_eq_80080 :
  (Nat.choose 18 6) * 11 = 80080 := sorry

end binom_18_6_mul_smallest_prime_gt_10_eq_80080_l285_285871


namespace cubic_sum_identity_l285_285953

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = -6) (h3 : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 :=
by
  sorry

end cubic_sum_identity_l285_285953


namespace part_a_l285_285847

theorem part_a (x y : ℝ) : x^2 - 2*y^2 = -((x + 2*y)^2 - 2*(x + y)^2) :=
sorry

end part_a_l285_285847


namespace product_of_x_y_l285_285082

-- Assume the given conditions
variables (EF GH FG HE : ℝ)
variables (x y : ℝ)
variable (EFGH : Type)

-- Conditions given
axiom h1 : EF = 58
axiom h2 : GH = 3 * x + 1
axiom h3 : FG = 2 * y^2
axiom h4 : HE = 36
-- It is given that EFGH forms a parallelogram
axiom h5 : EF = GH
axiom h6 : FG = HE

-- The product of x and y is determined by the conditions
theorem product_of_x_y : x * y = 57 * Real.sqrt 2 :=
by
  sorry

end product_of_x_y_l285_285082


namespace number_of_female_workers_l285_285911

theorem number_of_female_workers (M F : ℕ) (M_no F_no : ℝ) 
  (hM : M = 112)
  (h1 : M_no = 0.40 * M)
  (h2 : F_no = 0.25 * F)
  (h3 : M_no / (M_no + F_no) = 0.30)
  (h4 : F_no / (M_no + F_no) = 0.70)
  : F = 420 := 
by 
  sorry

end number_of_female_workers_l285_285911


namespace rectangle_area_l285_285320

theorem rectangle_area (a b : ℝ) (h : 2 * a^2 - 11 * a + 5 = 0) (hb : 2 * b^2 - 11 * b + 5 = 0) : a * b = 5 / 2 :=
sorry

end rectangle_area_l285_285320


namespace john_allowance_spent_l285_285600

theorem john_allowance_spent (B t d : ℝ) (h1 : t = 0.25 * (B - d)) (h2 : d = 0.10 * (B - t)) :
  (t + d) / B = 0.31 := by
  sorry

end john_allowance_spent_l285_285600


namespace transformed_roots_l285_285480

theorem transformed_roots 
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * (-1)^2 + b * (-1) + c = 0)
  (h₃ : a * 2^2 + b * 2 + c = 0) :
  (a * 0^2 + b * 0 + c = 0) ∧ (a * 3^2 + b * 3 + c = 0) :=
by 
  sorry

end transformed_roots_l285_285480


namespace probability_all_quit_from_same_tribe_l285_285544

def num_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def num_ways_to_choose_3_from_20 : ℕ := num_ways_to_choose 20 3
def num_ways_to_choose_3_from_10 : ℕ := num_ways_to_choose 10 3

theorem probability_all_quit_from_same_tribe :
  (num_ways_to_choose_3_from_10 * 2).toRat / num_ways_to_choose_3_from_20.toRat = 4 / 19 := 
  by
  sorry

end probability_all_quit_from_same_tribe_l285_285544


namespace Phil_earns_per_hour_l285_285093

-- Definitions based on the conditions in the problem
def Mike_hourly_rate : ℝ := 14
def Phil_hourly_rate : ℝ := Mike_hourly_rate - (0.5 * Mike_hourly_rate)

-- Mathematical assertion to prove
theorem Phil_earns_per_hour : Phil_hourly_rate = 7 :=
by 
  sorry

end Phil_earns_per_hour_l285_285093


namespace pigeons_count_l285_285141

theorem pigeons_count :
  let initial_pigeons := 1
  let additional_pigeons := 1
  (initial_pigeons + additional_pigeons) = 2 :=
by
  sorry

end pigeons_count_l285_285141


namespace minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l285_285172

-- Define the conditions once to reuse them for both proof statements.
variables {a b : ℝ} (ha: a > 0) (hb: b > 0) (h: (1/a) + (1/b) = Real.sqrt (a * b))

-- Problem (I)
theorem minimum_a3_b3 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

-- Problem (II)
theorem no_exist_a_b_2a_3b_eq_6 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  ¬ ∃ (a b : ℝ), 2 * a + 3 * b = 6 :=
sorry

end minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l285_285172


namespace solve_expression_l285_285923

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem solve_expression : f (g 3) - g (f 3) = -5 := by
  sorry

end solve_expression_l285_285923


namespace points_on_parabola_l285_285746

theorem points_on_parabola (t : ℝ) : 
  ∃ a b c : ℝ, ∀ (x y: ℝ), (x, y) = (Real.cos t ^ 2, Real.sin (2 * t)) → y^2 = 4 * x - 4 * x^2 := 
by
  sorry

end points_on_parabola_l285_285746


namespace hyperbola_center_l285_285738

theorem hyperbola_center (x y : ℝ) :
  (∃ h k, h = 2 ∧ k = -1 ∧ 
    (∀ x y, (3 * y + 3)^2 / 7^2 - (4 * x - 8)^2 / 6^2 = 1 ↔ 
      (y - (-1))^2 / ((7 / 3)^2) - (x - 2)^2 / ((3 / 2)^2) = 1)) :=
by sorry

end hyperbola_center_l285_285738


namespace fraction_of_menu_items_i_can_eat_l285_285716

def total_dishes (vegan_dishes non_vegan_dishes : ℕ) : ℕ := vegan_dishes + non_vegan_dishes

def vegan_dishes_without_soy (vegan_dishes vegan_with_soy : ℕ) : ℕ := vegan_dishes - vegan_with_soy

theorem fraction_of_menu_items_i_can_eat (vegan_dishes non_vegan_dishes vegan_with_soy : ℕ)
  (h_vegan_dishes : vegan_dishes = 6)
  (h_menu_total : vegan_dishes = (total_dishes vegan_dishes non_vegan_dishes) / 3)
  (h_vegan_with_soy : vegan_with_soy = 4)
  : (vegan_dishes_without_soy vegan_dishes vegan_with_soy) / (total_dishes vegan_dishes non_vegan_dishes) = 1 / 9 :=
by
  sorry

end fraction_of_menu_items_i_can_eat_l285_285716


namespace area_of_picture_l285_285265

theorem area_of_picture
  (paper_width : ℝ)
  (paper_height : ℝ)
  (left_margin : ℝ)
  (right_margin : ℝ)
  (top_margin_cm : ℝ)
  (bottom_margin_cm : ℝ)
  (cm_per_inch : ℝ)
  (converted_top_margin : ℝ := top_margin_cm * (1 / cm_per_inch))
  (converted_bottom_margin : ℝ := bottom_margin_cm * (1 / cm_per_inch))
  (picture_width : ℝ := paper_width - left_margin - right_margin)
  (picture_height : ℝ := paper_height - converted_top_margin - converted_bottom_margin)
  (area : ℝ := picture_width * picture_height)
  (h1 : paper_width = 8.5)
  (h2 : paper_height = 10)
  (h3 : left_margin = 1.5)
  (h4 : right_margin = 1.5)
  (h5 : top_margin_cm = 2)
  (h6 : bottom_margin_cm = 2.5)
  (h7 : cm_per_inch = 2.54)
  : area = 45.255925 :=
by sorry

end area_of_picture_l285_285265


namespace new_cost_percentage_l285_285454

variables (t c a x : ℝ) (n : ℕ)

def original_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * c * (a * x) ^ n

def new_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * (2 * c) * ((2 * a) * x) ^ (n + 2)

theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  new_cost t c a x n = 2^(n+1) * original_cost t c a x n * x^2 :=
by
  sorry

end new_cost_percentage_l285_285454


namespace p_pow_four_minus_one_divisible_by_ten_l285_285512

theorem p_pow_four_minus_one_divisible_by_ten
  (p : Nat) (prime_p : Nat.Prime p) (h₁ : p ≠ 2) (h₂ : p ≠ 5) : 
  10 ∣ (p^4 - 1) := 
by
  sorry

end p_pow_four_minus_one_divisible_by_ten_l285_285512


namespace most_likely_composition_l285_285827

def event_a : Prop := (1 / 3) * (1 / 3) * 2 = (2 / 9)
def event_d : Prop := 2 * (1 / 3 * 1 / 3) = (2 / 9)

theorem most_likely_composition :
  event_a ∧ event_d :=
by sorry

end most_likely_composition_l285_285827


namespace sum_first_four_terms_of_arithmetic_sequence_l285_285222

theorem sum_first_four_terms_of_arithmetic_sequence (a₈ a₉ a₁₀ : ℤ) (d : ℤ) (a₁ a₂ a₃ a₄ : ℤ) : 
  (a₈ = 21) →
  (a₉ = 17) →
  (a₁₀ = 13) →
  (d = a₉ - a₈) →
  (a₁ = a₈ - 7 * d) →
  (a₂ = a₁ + d) →
  (a₃ = a₂ + d) →
  (a₄ = a₃ + d) →
  a₁ + a₂ + a₃ + a₄ = 172 :=
by 
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
  sorry

end sum_first_four_terms_of_arithmetic_sequence_l285_285222


namespace total_homework_time_l285_285604

variable (num_math_problems num_social_studies_problems num_science_problems : ℕ)
variable (time_per_math_problem time_per_social_studies_problem time_per_science_problem : ℝ)

/-- Prove that the total time taken by Brooke to answer all his homework problems is 48 minutes -/
theorem total_homework_time :
  num_math_problems = 15 →
  num_social_studies_problems = 6 →
  num_science_problems = 10 →
  time_per_math_problem = 2 →
  time_per_social_studies_problem = 0.5 →
  time_per_science_problem = 1.5 →
  (num_math_problems * time_per_math_problem + num_social_studies_problems * time_per_social_studies_problem + num_science_problems * time_per_science_problem) = 48 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_homework_time_l285_285604


namespace savings_per_egg_l285_285116

def price_per_organic_egg : ℕ := 50 
def cost_of_tray : ℕ := 1200 -- in cents
def number_of_eggs_in_tray : ℕ := 30

theorem savings_per_egg : 
  price_per_organic_egg - (cost_of_tray / number_of_eggs_in_tray) = 10 := 
by
  sorry

end savings_per_egg_l285_285116


namespace faulty_balance_inequality_l285_285547

variable (m n a b G : ℝ)

theorem faulty_balance_inequality
  (h1 : m * a = n * G)
  (h2 : n * b = m * G) :
  (a + b) / 2 > G :=
sorry

end faulty_balance_inequality_l285_285547


namespace sum_of_solutions_eq_0_l285_285293

-- Define the conditions
def y : ℝ := 6
def main_eq (x : ℝ) : Prop := x^2 + y^2 = 145

-- State the theorem
theorem sum_of_solutions_eq_0 : 
  let x1 := Real.sqrt 109
  let x2 := -Real.sqrt 109
  x1 + x2 = 0 :=
by {
  sorry
}

end sum_of_solutions_eq_0_l285_285293


namespace intersect_at_0_intersect_at_180_intersect_at_90_l285_285395

-- Define radii R and r, and the distance c
variables {R r c : ℝ}

-- Formalize the conditions and corresponding angles
theorem intersect_at_0 (h : c = R - r) : True := 
sorry

theorem intersect_at_180 (h : c = R + r) : True := 
sorry

theorem intersect_at_90 (h : c = Real.sqrt (R^2 + r^2)) : True := 
sorry

end intersect_at_0_intersect_at_180_intersect_at_90_l285_285395


namespace find_difference_of_a_b_l285_285201

noncomputable def a_b_are_relative_prime_and_positive (a b : ℕ) (hab_prime : Nat.gcd a b = 1) (ha_pos : a > 0) (hb_pos : b > 0) (h_gt : a > b) : Prop :=
  a ^ 3 - b ^ 3 = (131 / 5) * (a - b) ^ 3

theorem find_difference_of_a_b (a b : ℕ) 
  (hab_prime : Nat.gcd a b = 1) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (h_gt : a > b) 
  (h_eq : (a ^ 3 - b ^ 3 : ℚ) / (a - b) ^ 3 = 131 / 5) : 
  a - b = 7 :=
  sorry

end find_difference_of_a_b_l285_285201


namespace find_first_number_l285_285579

def is_lcm (a b l : ℕ) : Prop := l = Nat.lcm a b

theorem find_first_number :
  ∃ (a b : ℕ), (5 * b) = a ∧ (4 * b) = b ∧ is_lcm a b 80 ∧ a = 20 :=
by
  sorry

end find_first_number_l285_285579


namespace collinear_points_min_value_l285_285509

open Real

/-- Let \(\overrightarrow{e_{1}}\) and \(\overrightarrow{e_{2}}\) be two non-collinear vectors in a plane,
    \(\overrightarrow{AB} = (a-1) \overrightarrow{e_{1}} + \overrightarrow{e_{2}}\),
    \(\overrightarrow{AC} = b \overrightarrow{e_{1}} - 2 \overrightarrow{e_{2}}\),
    with \(a > 0\) and \(b > 0\). 
    If points \(A\), \(B\), and \(C\) are collinear, then the minimum value of \(\frac{1}{a} + \frac{2}{b}\) is \(4\). -/
theorem collinear_points_min_value 
  (e1 e2 : ℝ) 
  (H_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0))
  (a b : ℝ) 
  (H_a_pos : a > 0) 
  (H_b_pos : b > 0)
  (H_collinear : ∃ x : ℝ, (a - 1) * e1 + e2 = x * (b * e1 - 2 * e2)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + (1/2) * b = 1 ∧ (∀ a b : ℝ, (1/a) + (2/b) ≥ 4) :=
sorry

end collinear_points_min_value_l285_285509


namespace remainder_problem_l285_285560

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 15 = 11) (hy : y % 15 = 13) (hz : z % 15 = 14) : 
  (y + z - x) % 15 = 1 := 
by 
  sorry

end remainder_problem_l285_285560


namespace cylindrical_to_rectangular_l285_285727

theorem cylindrical_to_rectangular (r θ z : ℝ) (h1 : r = 6) (h2 : θ = π / 3) (h3 : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 2) := 
by 
  rw [h1, h2, h3]
  sorry

end cylindrical_to_rectangular_l285_285727


namespace order_of_values_l285_285510

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem order_of_values (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  -- Proof would go here
  sorry

end order_of_values_l285_285510


namespace combined_time_third_attempt_l285_285088

noncomputable def first_lock_initial : ℕ := 5
noncomputable def second_lock_initial : ℕ := 3 * first_lock_initial - 3
noncomputable def combined_initial : ℕ := 5 * second_lock_initial

noncomputable def first_lock_second_attempt : ℝ := first_lock_initial - 0.1 * first_lock_initial
noncomputable def first_lock_third_attempt : ℝ := first_lock_second_attempt - 0.1 * first_lock_second_attempt

noncomputable def second_lock_second_attempt : ℝ := second_lock_initial - 0.15 * second_lock_initial
noncomputable def second_lock_third_attempt : ℝ := second_lock_second_attempt - 0.15 * second_lock_second_attempt

noncomputable def combined_third_attempt : ℝ := 5 * second_lock_third_attempt

theorem combined_time_third_attempt : combined_third_attempt = 43.35 :=
by
  sorry

end combined_time_third_attempt_l285_285088


namespace sum_of_sequences_l285_285450

def sequence1 := [2, 14, 26, 38, 50]
def sequence2 := [12, 24, 36, 48, 60]
def sequence3 := [5, 15, 25, 35, 45]

theorem sum_of_sequences :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := 
by 
  sorry

end sum_of_sequences_l285_285450


namespace solve_for_x_l285_285417

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end solve_for_x_l285_285417


namespace tangent_identity_problem_l285_285754

theorem tangent_identity_problem 
    (α β : ℝ) 
    (h1 : Real.tan (α + β) = 1) 
    (h2 : Real.tan (α - π / 3) = 1 / 3) 
    : Real.tan (β + π / 3) = 1 / 2 := 
sorry

end tangent_identity_problem_l285_285754


namespace greatest_positive_integer_difference_l285_285563

-- Define the conditions
def condition_x (x : ℝ) : Prop := 4 < x ∧ x < 6
def condition_y (y : ℝ) : Prop := 6 < y ∧ y < 10

-- Define the problem statement
theorem greatest_positive_integer_difference (x y : ℕ) (hx : condition_x x) (hy : condition_y y) : y - x = 4 :=
sorry

end greatest_positive_integer_difference_l285_285563


namespace range_of_b_l285_285179

theorem range_of_b (M : Set (ℝ × ℝ)) (N : ℝ → ℝ → Set (ℝ × ℝ)) :
  (∀ m : ℝ, (∃ x y : ℝ, (x, y) ∈ M ∧ (x, y) ∈ (N m b))) ↔ b ∈ Set.Icc (- Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by
  sorry

end range_of_b_l285_285179


namespace cube_surface_area_l285_285817

noncomputable def total_surface_area_of_cube (Q : ℝ) : ℝ :=
  8 * Q * Real.sqrt 3 / 3

theorem cube_surface_area (Q : ℝ) (h : Q > 0) :
  total_surface_area_of_cube Q = 8 * Q * Real.sqrt 3 / 3 :=
sorry

end cube_surface_area_l285_285817


namespace apples_in_market_l285_285120

theorem apples_in_market (A O : ℕ) 
    (h1 : A = O + 27) 
    (h2 : A + O = 301) : 
    A = 164 :=
by
  sorry

end apples_in_market_l285_285120


namespace original_coins_count_l285_285017

-- Define the initial amount of coins and the fractions taken out each day
def initial_coins := ℕ
def day1_fraction := 1 / 9
def day2_fraction := 1 / 8
def day3_fraction := 1 / 7
def day4_fraction := 1 / 6
def day5_fraction := 1 / 5
def day6_fraction := 1 / 4
def day7_fraction := 1 / 3
def day8_fraction := 1 / 2

-- Remaining coins after each extraction
def coins_after_day1 (initial_coins : ℕ) := initial_coins - (initial_coins * day1_fraction)
def coins_after_day2 (coins_after_day1 : ℕ) := coins_after_day1 - (coins_after_day1 * day2_fraction)
def coins_after_day3 (coins_after_day2 : ℕ) := coins_after_day2 - (coins_after_day2 * day3_fraction)
def coins_after_day4 (coins_after_day3 : ℕ) := coins_after_day3 - (coins_after_day3 * day4_fraction)
def coins_after_day5 (coins_after_day4 : ℕ) := coins_after_day4 - (coins_after_day4 * day5_fraction)
def coins_after_day6 (coins_after_day5 : ℕ) := coins_after_day5 - (coins_after_day5 * day6_fraction)
def coins_after_day7 (coins_after_day6 : ℕ) := coins_after_day6 - (coins_after_day6 * day7_fraction)
def coins_after_day8 (coins_after_day7 : ℕ) := coins_after_day7 - (coins_after_day7 * day8_fraction)

-- Main theorem stating the initial coins count equals to 45 coins after applying the sequential fractions
theorem original_coins_count : 
  ∃ initial_coins : ℕ, 
    coins_after_day8 (coins_after_day7 
      (coins_after_day6 
        (coins_after_day5 
          (coins_after_day4 
            (coins_after_day3 
              (coins_after_day2 
                (coins_after_day1 initial_coins)
              )
            )
          )
        )
      )
    ) = 5 → initial_coins = 45 
:= sorry

end original_coins_count_l285_285017


namespace solve_years_later_twice_age_l285_285029

-- Define the variables and the given conditions
def man_age (S: ℕ) := S + 25
def years_later_twice_age (S M: ℕ) (Y: ℕ) := (M + Y = 2 * (S + Y))

-- Given conditions
def present_age_son := 23
def present_age_man := man_age present_age_son

theorem solve_years_later_twice_age :
  ∃ Y, years_later_twice_age present_age_son present_age_man Y ∧ Y = 2 := by
  sorry

end solve_years_later_twice_age_l285_285029


namespace depth_of_sand_l285_285432

theorem depth_of_sand (h : ℝ) (fraction_above_sand : ℝ) :
  h = 9000 → fraction_above_sand = 1/9 → depth = 342 :=
by
  -- height of the pyramid
  let height := 9000
  -- ratio of submerged height to the total height
  let ratio := (8 / 9)^(1 / 3)
  -- height of the submerged part
  let submerged_height := height * ratio
  -- depth of the sand
  let depth := height - submerged_height
  sorry

end depth_of_sand_l285_285432


namespace crayons_initially_l285_285207

theorem crayons_initially (crayons_left crayons_lost : ℕ) (h_left : crayons_left = 134) (h_lost : crayons_lost = 345) :
  crayons_left + crayons_lost = 479 :=
by
  sorry

end crayons_initially_l285_285207


namespace dimes_left_l285_285357

-- Definitions based on the conditions
def Initial_dimes : ℕ := 8
def Sister_borrowed : ℕ := 4
def Friend_borrowed : ℕ := 2

-- The proof problem statement (without the proof)
theorem dimes_left (Initial_dimes Sister_borrowed Friend_borrowed : ℕ) : 
  Initial_dimes = 8 → Sister_borrowed = 4 → Friend_borrowed = 2 →
  Initial_dimes - (Sister_borrowed + Friend_borrowed) = 2 :=
by
  intros
  sorry

end dimes_left_l285_285357


namespace colleen_pencils_l285_285506

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (extra_cost : ℕ) (colleen_paid : ℕ)
  (H1 : joy_pencils = 30)
  (H2 : pencil_cost = 4)
  (H3 : extra_cost = 80)
  (H4 : colleen_paid = (joy_pencils * pencil_cost) + extra_cost) :
  colleen_paid / pencil_cost = 50 := 
by 
  -- Hints, if necessary
sorry

end colleen_pencils_l285_285506


namespace no_positive_integer_solution_exists_l285_285165

theorem no_positive_integer_solution_exists :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * x^2 + 2 * x + 2 = y^2 :=
by
  -- The proof steps will go here.
  sorry

end no_positive_integer_solution_exists_l285_285165


namespace gcd_sequence_terms_l285_285678

theorem gcd_sequence_terms (d m : ℕ) (hd : d > 1) (hm : m > 0) :
    ∃ k l : ℕ, k ≠ l ∧ gcd (2 ^ (2 ^ k) + d) (2 ^ (2 ^ l) + d) > m := 
sorry

end gcd_sequence_terms_l285_285678


namespace trains_crossing_time_l285_285239

theorem trains_crossing_time
  (L1 : ℕ) (L2 : ℕ) (T1 : ℕ) (T2 : ℕ)
  (H1 : L1 = 150) (H2 : L2 = 180)
  (H3 : T1 = 10) (H4 : T2 = 15) :
  (L1 + L2) / ((L1 / T1) + (L2 / T2)) = 330 / 27 := sorry

end trains_crossing_time_l285_285239


namespace eval_f_four_times_l285_285745

noncomputable def f (z : Complex) : Complex := 
if z.im ≠ 0 then z * z else -(z * z)

theorem eval_f_four_times : 
  f (f (f (f (Complex.mk 2 1)))) = Complex.mk 164833 354192 := 
by 
  sorry

end eval_f_four_times_l285_285745


namespace economical_shower_heads_l285_285423

theorem economical_shower_heads (x T : ℕ) (x_pos : 0 < x)
    (students : ℕ := 100)
    (preheat_time_per_shower : ℕ := 3)
    (shower_time_per_group : ℕ := 12) :
  (T = preheat_time_per_shower * x + shower_time_per_group * (students / x)) →
  (students * preheat_time_per_shower + shower_time_per_group * students / x = T) →
  x = 20 := by
  sorry

end economical_shower_heads_l285_285423


namespace percentage_decrease_l285_285146

theorem percentage_decrease (x : ℝ) (h : x > 0) : ∃ p : ℝ, p = 0.20 ∧ ((1.25 * x) * (1 - p) = x) :=
by
  sorry

end percentage_decrease_l285_285146


namespace opposite_of_neg_three_l285_285378

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l285_285378


namespace sum_evaluation_l285_285288

noncomputable def T : ℝ := ∑' k : ℕ, (2*k+1) / 5^(k+1)

theorem sum_evaluation : T = 5 / 16 := sorry

end sum_evaluation_l285_285288


namespace compare_a_b_c_compare_explicitly_defined_a_b_c_l285_285302

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end compare_a_b_c_compare_explicitly_defined_a_b_c_l285_285302


namespace f_at_one_f_increasing_f_range_for_ineq_l285_285056

-- Define the function f with its properties
noncomputable def f : ℝ → ℝ := sorry

-- Properties of f
axiom f_domain : ∀ x, 0 < x → f x ≠ 0 
axiom f_property_additive : ∀ x y, f (x * y) = f x + f y
axiom f_property_positive : ∀ x, (1 < x) → (0 < f x)
axiom f_property_fract : f (1/3) = -1

-- Proofs to be completed
theorem f_at_one : f 1 = 0 :=
sorry

theorem f_increasing : ∀ (x₁ x₂ : ℝ), (0 < x₁) → (0 < x₂) → (x₁ < x₂) → (f x₁ < f x₂) :=
sorry

theorem f_range_for_ineq : {x : ℝ | 2 < x ∧ x ≤ 9/4} = {x : ℝ | f x - f (x - 2) ≥ 2} :=
sorry

end f_at_one_f_increasing_f_range_for_ineq_l285_285056


namespace skee_ball_tickets_l285_285410

-- Represent the given conditions as Lean definitions
def whack_a_mole_tickets : ℕ := 33
def candy_cost_per_piece : ℕ := 6
def candies_bought : ℕ := 7
def total_candy_tickets : ℕ := candies_bought * candy_cost_per_piece

-- Goal: Prove the number of tickets won playing 'skee ball'
theorem skee_ball_tickets (h : 42 = total_candy_tickets): whack_a_mole_tickets + 9 = total_candy_tickets :=
by {
  sorry
}

end skee_ball_tickets_l285_285410


namespace opposite_of_neg_three_l285_285375

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l285_285375


namespace arithmetic_sequence_condition_l285_285902

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) ↔ (a + d = b + c) :=
sorry

end arithmetic_sequence_condition_l285_285902


namespace subset_iff_l285_285188

open Set

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_iff (a : ℝ) : A ⊆ B a ↔ 2 ≤ a :=
by sorry

end subset_iff_l285_285188


namespace roses_per_flat_l285_285681

-- Conditions
def flats_petunias := 4
def petunias_per_flat := 8
def flats_roses := 3
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer_needed := 314

-- Derived definitions
def total_petunias := flats_petunias * petunias_per_flat
def fertilizer_for_petunias := total_petunias * fertilizer_per_petunia
def fertilizer_for_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap
def total_fertilizer_needed_roses := total_fertilizer_needed - (fertilizer_for_petunias + fertilizer_for_venus_flytraps)

-- Proof statement
theorem roses_per_flat :
  ∃ R : ℕ, flats_roses * R * fertilizer_per_rose = total_fertilizer_needed_roses ∧ R = 6 :=
by
  -- Proof goes here
  sorry

end roses_per_flat_l285_285681


namespace expand_expression_l285_285880

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end expand_expression_l285_285880


namespace range_of_m_l285_285479

variable (m : ℝ)
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 := sorry

end range_of_m_l285_285479


namespace min_value_of_reciprocal_sum_l285_285894

variable (a b : ℝ)
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (condition : 2 * a + b = 1)

theorem min_value_of_reciprocal_sum : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 :=
by
  -- Proof is skipped
  sorry

end min_value_of_reciprocal_sum_l285_285894


namespace sum_of_odd_integers_21_to_51_l285_285010

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_odd_integers_21_to_51 : sum_arithmetic_seq 21 2 51 = 576 := by
  sorry

end sum_of_odd_integers_21_to_51_l285_285010


namespace total_chocolate_bars_l285_285426

theorem total_chocolate_bars (n_small_boxes : ℕ) (bars_per_box : ℕ) (total_bars : ℕ) :
  n_small_boxes = 16 → bars_per_box = 25 → total_bars = 16 * 25 → total_bars = 400 :=
by
  intros
  sorry

end total_chocolate_bars_l285_285426


namespace combined_resistance_parallel_l285_285328

theorem combined_resistance_parallel (R1 R2 : ℝ) (r : ℝ) 
  (hR1 : R1 = 8) (hR2 : R2 = 9) (h_parallel : (1 / r) = (1 / R1) + (1 / R2)) : 
  r = 72 / 17 :=
by
  sorry

end combined_resistance_parallel_l285_285328


namespace determine_k_l285_285747

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 :=
by
  sorry

end determine_k_l285_285747


namespace bryan_total_earnings_l285_285867

-- Declare the data given in the problem:
def num_emeralds : ℕ := 3
def num_rubies : ℕ := 2
def num_sapphires : ℕ := 3

def price_emerald : ℝ := 1785
def price_ruby : ℝ := 2650
def price_sapphire : ℝ := 2300

-- Calculate the total earnings from each type of stone:
def total_emeralds : ℝ := num_emeralds * price_emerald
def total_rubies : ℝ := num_rubies * price_ruby
def total_sapphires : ℝ := num_sapphires * price_sapphire

-- Calculate the overall total earnings:
def total_earnings : ℝ := total_emeralds + total_rubies + total_sapphires

-- Prove that Bryan got 17555 dollars in total:
theorem bryan_total_earnings : total_earnings = 17555 := by
  simp [total_earnings, total_emeralds, total_rubies, total_sapphires, num_emeralds, num_rubies, num_sapphires, price_emerald, price_ruby, price_sapphire]
  sorry

end bryan_total_earnings_l285_285867


namespace flight_duration_NY_to_CT_l285_285038

theorem flight_duration_NY_to_CT :
  let departure_London_to_NY : Nat := 6 -- time in ET on Monday
  let arrival_NY_later_hours : Nat := 18 -- hours after departure
  let arrival_NY : Nat := (departure_London_to_NY + arrival_NY_later_hours) % 24 -- time in ET on Tuesday
  let arrival_CapeTown : Nat := 10 -- time in ET on Tuesday
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24 -- duration calculation
  duration_flight_NY_to_CT = 10 :=
by
  let departure_London_to_NY := 6
  let arrival_NY_later_hours := 18
  let arrival_NY := (departure_London_to_NY + arrival_NY_later_hours) % 24
  let arrival_CapeTown := 10
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24
  show duration_flight_NY_to_CT = 10
  sorry

end flight_duration_NY_to_CT_l285_285038


namespace jugs_needed_to_provide_water_for_students_l285_285982

def jug_capacity : ℕ := 40
def students : ℕ := 200
def cups_per_student : ℕ := 10

def total_cups_needed := students * cups_per_student

theorem jugs_needed_to_provide_water_for_students :
  total_cups_needed / jug_capacity = 50 :=
by
  -- Proof goes here
  sorry

end jugs_needed_to_provide_water_for_students_l285_285982


namespace find_r_of_tangential_cones_l285_285966

theorem find_r_of_tangential_cones (r : ℝ) : 
  (∃ (r1 r2 r3 R : ℝ), r1 = 2 * r ∧ r2 = 3 * r ∧ r3 = 10 * r ∧ R = 15 ∧
  -- Additional conditions to ensure the three cones touch and share a slant height
  -- with the truncated cone of radius R
  true) → r = 29 :=
by
  intro h
  sorry

end find_r_of_tangential_cones_l285_285966


namespace logan_television_hours_l285_285202

-- Definitions
def minutes_in_an_hour : ℕ := 60
def logan_minutes_watched : ℕ := 300
def logan_hours_watched : ℕ := logan_minutes_watched / minutes_in_an_hour

-- Theorem statement
theorem logan_television_hours : logan_hours_watched = 5 := by
  sorry

end logan_television_hours_l285_285202


namespace radio_show_play_song_duration_l285_285434

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end radio_show_play_song_duration_l285_285434


namespace displacement_correct_l285_285297

-- Define the initial conditions of the problem
def init_north := 50
def init_east := 70
def init_south := 20
def init_west := 30

-- Define the net movements
def net_north := init_north - init_south
def net_east := init_east - init_west

-- Define the straight-line distance using the Pythagorean theorem
def displacement_AC := (net_north ^ 2 + net_east ^ 2).sqrt

theorem displacement_correct : displacement_AC = 50 := 
by sorry

end displacement_correct_l285_285297


namespace probability_of_four_ones_approx_l285_285550

noncomputable def probability_of_four_ones_in_twelve_dice : ℚ :=
  (nat.choose 12 4 : ℚ) * (1 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ 8

theorem probability_of_four_ones_approx :
  probability_of_four_ones_in_twelve_dice ≈ 0.089 :=
sorry

end probability_of_four_ones_approx_l285_285550


namespace complement_A_union_B_l285_285796

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Lean statement to prove the complement of A ∪ B with respect to U
theorem complement_A_union_B : U \ (A ∪ B) = {7, 8} :=
by
sorry

end complement_A_union_B_l285_285796


namespace expected_sixes_correct_l285_285970

-- Define probabilities for rolling individual numbers on a die
def P (n : ℕ) (k : ℕ) : ℚ := if k = n then 1 / 6 else 0

-- Expected value calculation for two dice
noncomputable def expected_sixes_two_dice_with_resets : ℚ :=
(0 * (13/18)) + (1 * (2/9)) + (2 * (1/36))

-- Main theorem to prove
theorem expected_sixes_correct :
  expected_sixes_two_dice_with_resets = 5 / 18 :=
by
  -- The actual proof steps go here; added sorry to skip the proof.
  sorry

end expected_sixes_correct_l285_285970


namespace ways_to_divide_week_l285_285442

def week_seconds : ℕ := 604800

theorem ways_to_divide_week (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : week_seconds = n * m) :
  (∃ (pairs : ℕ), pairs = 336) :=
sorry

end ways_to_divide_week_l285_285442


namespace no_infinite_prime_sequence_l285_285045

theorem no_infinite_prime_sequence (p : ℕ) (h_prime : Nat.Prime p) :
  ¬(∃ (p_seq : ℕ → ℕ), (∀ n, Nat.Prime (p_seq n)) ∧ (∀ n, p_seq (n + 1) = 2 * p_seq n + 1)) :=
by
  sorry

end no_infinite_prime_sequence_l285_285045


namespace pow_ge_double_plus_one_l285_285521

theorem pow_ge_double_plus_one (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2 * (n + 1) :=
sorry

end pow_ge_double_plus_one_l285_285521


namespace sum_m_n_l285_285130

-- Declare the namespaces and definitions for the problem
namespace DelegateProblem

-- Condition: total number of delegates
def total_delegates : Nat := 12

-- Condition: number of delegates from each country
def delegates_per_country : Nat := 4

-- Computation of m and n such that their sum is 452
-- This follows from the problem statement and the solution provided
def m : Nat := 221
def n : Nat := 231

-- Theorem statement in Lean for proving m + n = 452
theorem sum_m_n : m + n = 452 := by
  -- Algebraic proof omitted
  sorry

end DelegateProblem

end sum_m_n_l285_285130


namespace nearest_integer_to_expansion_l285_285284

theorem nearest_integer_to_expansion : 
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  abs (a^4 - 1090) < 1 :=
by
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  sorry

end nearest_integer_to_expansion_l285_285284


namespace total_hamburgers_menu_l285_285181

def meat_patties_choices := 4
def condiment_combinations := 2 ^ 9

theorem total_hamburgers_menu :
  meat_patties_choices * condiment_combinations = 2048 :=
by
  sorry

end total_hamburgers_menu_l285_285181


namespace probability_heads_exactly_8_in_10_l285_285833

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end probability_heads_exactly_8_in_10_l285_285833


namespace complex_number_on_line_l285_285077

theorem complex_number_on_line (a : ℝ) (h : (3 : ℝ) = (a - 1) + 2) : a = 2 :=
by
  sorry

end complex_number_on_line_l285_285077


namespace cable_cost_l285_285993

theorem cable_cost (num_ew_streets : ℕ) (length_ew_street : ℕ) 
                   (num_ns_streets : ℕ) (length_ns_street : ℕ) 
                   (cable_per_mile : ℕ) (cost_per_mile : ℕ) :
  num_ew_streets = 18 →
  length_ew_street = 2 →
  num_ns_streets = 10 →
  length_ns_street = 4 →
  cable_per_mile = 5 →
  cost_per_mile = 2000 →
  (num_ew_streets * length_ew_street + num_ns_streets * length_ns_street) * cable_per_mile * cost_per_mile = 760000 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  simp
  sorry

end cable_cost_l285_285993


namespace solve_quadratic_eq_l285_285813

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end solve_quadratic_eq_l285_285813


namespace samia_walked_distance_l285_285211

theorem samia_walked_distance :
  ∀ (total_distance cycling_speed walking_speed total_time : ℝ), 
  total_distance = 18 → 
  cycling_speed = 20 → 
  walking_speed = 4 → 
  total_time = 1 + 10 / 60 → 
  2 / 3 * total_distance / cycling_speed + 1 / 3 * total_distance / walking_speed = total_time → 
  1 / 3 * total_distance = 6 := 
by
  intros total_distance cycling_speed walking_speed total_time h1 h2 h3 h4 h5
  sorry

end samia_walked_distance_l285_285211


namespace tan_alpha_minus_pi_over_4_l285_285769

open Real

theorem tan_alpha_minus_pi_over_4
  (α : ℝ)
  (a b : ℝ × ℝ)
  (h1 : a = (cos α, -2))
  (h2 : b = (sin α, 1))
  (h3 : ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) :
  tan (α - π / 4) = -3 := 
sorry

end tan_alpha_minus_pi_over_4_l285_285769


namespace find_a_minus_b_l285_285771

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end find_a_minus_b_l285_285771


namespace boat_equation_l285_285329

-- Define the conditions given in the problem
def total_boats : ℕ := 8
def large_boat_capacity : ℕ := 6
def small_boat_capacity : ℕ := 4
def total_students : ℕ := 38

-- Define the theorem to be proven
theorem boat_equation (x : ℕ) (h0 : x ≤ total_boats) : 
  large_boat_capacity * (total_boats - x) + small_boat_capacity * x = total_students := by
  sorry

end boat_equation_l285_285329


namespace find_x_l285_285414

def hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

def hash_of_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_p p x + x

def triple_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_of_hash_p p x + x

theorem find_x (p x : ℤ) (h : triple_hash_p p x = -4) (hp : p = 18) : x = -21 :=
by
  sorry

end find_x_l285_285414


namespace bridge_length_is_115_meters_l285_285986

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_km_per_hr : ℝ) (time_to_pass : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * (1000 / 3600)
  let total_distance := speed_m_per_s * time_to_pass
  total_distance - length_of_train

theorem bridge_length_is_115_meters :
  length_of_bridge 300 35 42.68571428571429 = 115 :=
by
  -- Here the proof has to show the steps for converting speed and calculating distances
  sorry

end bridge_length_is_115_meters_l285_285986


namespace ratio_B_to_A_l285_285861

theorem ratio_B_to_A (A B C : ℝ) 
  (hA : A = 1 / 21) 
  (hC : C = 2 * B) 
  (h_sum : A + B + C = 1 / 3) : 
  B / A = 2 := 
by 
  /- Proof goes here, but it's omitted as per instructions -/
  sorry

end ratio_B_to_A_l285_285861


namespace compare_powers_l285_285299

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end compare_powers_l285_285299


namespace product_divisible_by_10_l285_285715

noncomputable def probability_divisible_by_10 (n : ℕ) (h : n > 1) : ℝ :=
  1 - (8^n + 5^n - 4^n) / 9^n

theorem product_divisible_by_10 (n : ℕ) (h : n > 1) :
  probability_divisible_by_10 n h = 1 - (8^n + 5^n - 4^n)/(9^n) :=
by
  sorry

end product_divisible_by_10_l285_285715


namespace min_value_proof_l285_285060

noncomputable def min_value (a b : ℝ) : ℝ := (1 : ℝ)/a + (1 : ℝ)/b

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) :
  min_value a b = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_proof_l285_285060


namespace find_S20_l285_285757

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry

axiom a_nonzero (n : ℕ) : a_seq n ≠ 0
axiom a1_eq : a_seq 1 = 1
axiom Sn_eq (n : ℕ) : S n = (a_seq n * a_seq (n + 1)) / 2

theorem find_S20 : S 20 = 210 := sorry

end find_S20_l285_285757


namespace total_cost_of_cable_l285_285994

-- Defining the conditions as constants
def east_west_streets := 18
def east_west_length := 2
def north_south_streets := 10
def north_south_length := 4
def cable_per_mile_street := 5
def cost_per_mile_cable := 2000

-- The theorem contains the problem statement and asserts the answer
theorem total_cost_of_cable :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) * cable_per_mile_street * cost_per_mile_cable = 760000 := 
  sorry

end total_cost_of_cable_l285_285994


namespace esther_biking_speed_l285_285786

theorem esther_biking_speed (d x : ℝ)
  (h_bike_speed : x > 0)
  (h_average_speed : 5 = 2 * d / (d / x + d / 3)) :
  x = 15 :=
by
  sorry

end esther_biking_speed_l285_285786


namespace even_combinations_result_in_486_l285_285258

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l285_285258


namespace simplify_expr_l285_285153

-- Define the expression
def expr (a : ℝ) := 4 * a ^ 2 * (3 * a - 1)

-- State the theorem
theorem simplify_expr (a : ℝ) : expr a = 12 * a ^ 3 - 4 * a ^ 2 := 
by 
  sorry

end simplify_expr_l285_285153


namespace sequence_sum_l285_285066

theorem sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = (n + 1) * (n + 1) - 1)
  (ha : ∀ n : ℕ, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 :=
by
  sorry

end sequence_sum_l285_285066


namespace folded_paper_area_ratio_l285_285441

theorem folded_paper_area_ratio (s : ℝ) (h : s > 0) :
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  (folded_area / A) = 7 / 4 :=
by
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  show (folded_area / A) = 7 / 4
  sorry

end folded_paper_area_ratio_l285_285441


namespace find_sale_month_4_l285_285262

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end find_sale_month_4_l285_285262


namespace distance_between_D_and_E_l285_285158

theorem distance_between_D_and_E 
  (A B C D E P : Type)
  (d_AB : ℕ) (d_BC : ℕ) (d_AC : ℕ) (d_PC : ℕ) 
  (AD_parallel_BC : Prop) (AB_parallel_CE : Prop) 
  (distance_DE : ℕ) :
  d_AB = 15 →
  d_BC = 18 → 
  d_AC = 21 → 
  d_PC = 7 → 
  AD_parallel_BC →
  AB_parallel_CE →
  distance_DE = 15 :=
by
  sorry

end distance_between_D_and_E_l285_285158


namespace symmetric_axis_parabola_l285_285122

theorem symmetric_axis_parabola (h k : ℝ) (x : ℝ) :
  (∀ x, y = (x - h)^2 + k) → h = 2 → (x = 2) :=
by
  sorry

end symmetric_axis_parabola_l285_285122


namespace sum_first_39_natural_numbers_l285_285975

theorem sum_first_39_natural_numbers :
  (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_first_39_natural_numbers_l285_285975


namespace opposite_of_neg3_is_3_l285_285374

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l285_285374


namespace gallons_of_gas_l285_285791

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end gallons_of_gas_l285_285791


namespace remove_terms_for_desired_sum_l285_285035

theorem remove_terms_for_desired_sum :
  let series_sum := (1/3) + (1/5) + (1/7) + (1/9) + (1/11) + (1/13)
  series_sum - (1/11 + 1/13) = 11/20 :=
by
  sorry

end remove_terms_for_desired_sum_l285_285035


namespace simplify_fraction_l285_285810

theorem simplify_fraction (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 :=
by
  sorry

end simplify_fraction_l285_285810


namespace quadratic_eq_solutions_l285_285119

open Real

theorem quadratic_eq_solutions (x : ℝ) :
  (2 * x + 1) ^ 2 = (2 * x + 1) * (x - 1) ↔ x = -1 / 2 ∨ x = -2 :=
by sorry

end quadratic_eq_solutions_l285_285119


namespace marsh_ducks_l285_285125

theorem marsh_ducks (D : ℕ) (h1 : 58 = D + 21) : D = 37 := 
by {
  sorry
}

end marsh_ducks_l285_285125


namespace fifth_term_is_2_11_over_60_l285_285684

noncomputable def fifth_term_geo_prog (a₁ a₂ a₃ : ℝ) (r : ℝ) : ℝ :=
  a₃ * r^2

theorem fifth_term_is_2_11_over_60
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4))
  (h₂ : a₂ = 2^(1/5))
  (h₃ : a₃ = 2^(1/6))
  (r : ℝ)
  (common_ratio : r = a₂ / a₁) :
  fifth_term_geo_prog a₁ a₂ a₃ r = 2^(11/60) :=
by
  sorry

end fifth_term_is_2_11_over_60_l285_285684


namespace solve_2019_gon_l285_285781

noncomputable def problem_2019_gon (x : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, (x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) + x (i+5) + x (i+6) + x (i+7) + x (i+8) = 300))
  ∧ (x 18 = 19)
  ∧ (x 19 = 20)

theorem solve_2019_gon :
  ∀ x : ℕ → ℕ,
  problem_2019_gon x →
  x 2018 = 61 :=
by sorry

end solve_2019_gon_l285_285781


namespace weeks_per_mouse_correct_l285_285751

def years_in_decade : ℕ := 10
def weeks_per_year : ℕ := 52
def total_mice : ℕ := 130

def total_weeks_in_decade : ℕ := years_in_decade * weeks_per_year
def weeks_per_mouse : ℕ := total_weeks_in_decade / total_mice

theorem weeks_per_mouse_correct : weeks_per_mouse = 4 := 
sorry

end weeks_per_mouse_correct_l285_285751


namespace exists_m_such_that_m_plus_one_pow_zero_eq_one_l285_285881

theorem exists_m_such_that_m_plus_one_pow_zero_eq_one : 
  ∃ m : ℤ, (m + 1)^0 = 1 ∧ m ≠ -1 :=
by
  sorry

end exists_m_such_that_m_plus_one_pow_zero_eq_one_l285_285881


namespace inequality_a_b_c_l285_285091

theorem inequality_a_b_c 
  (a b c : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
by 
  sorry

end inequality_a_b_c_l285_285091


namespace number_of_pupils_l285_285411

theorem number_of_pupils (n : ℕ) : (83 - 63) / n = 1 / 2 → n = 40 :=
by
  intro h
  -- This is where the proof would go.
  sorry

end number_of_pupils_l285_285411


namespace minimal_functions_l285_285602

open Int

theorem minimal_functions (f : ℤ → ℤ) (c : ℤ) :
  (∀ x, f (x + 2017) = f x) ∧
  (∀ x y, (f (f x + f y + 1) - f (f x + f y)) % 2017 = c) →
  (c = 1 ∨ c = 2016 ∨ c = 1008 ∨ c = 1009) :=
by
  sorry

end minimal_functions_l285_285602


namespace total_number_of_people_l285_285274

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end total_number_of_people_l285_285274


namespace find_abc_l285_285123

theorem find_abc : ∃ (a b c : ℝ), a + b + c = 1 ∧ 4 * a + 2 * b + c = 5 ∧ 9 * a + 3 * b + c = 13 ∧ a - b + c = 5 := by
  sorry

end find_abc_l285_285123


namespace elvis_squares_count_l285_285736

theorem elvis_squares_count :
  ∀ (total : ℕ) (Elvis_squares Ralph_squares squares_used_by_Ralph matchsticks_left : ℕ)
  (uses_by_Elvis_per_square uses_by_Ralph_per_square : ℕ),
  total = 50 →
  uses_by_Elvis_per_square = 4 →
  uses_by_Ralph_per_square = 8 →
  Ralph_squares = 3 →
  matchsticks_left = 6 →
  squares_used_by_Ralph = Ralph_squares * uses_by_Ralph_per_square →
  total = (Elvis_squares * uses_by_Elvis_per_square) + squares_used_by_Ralph + matchsticks_left →
  Elvis_squares = 5 :=
by
  sorry

end elvis_squares_count_l285_285736


namespace average_speed_trip_l285_285285

theorem average_speed_trip :
  let distance_1 := 65
  let distance_2 := 45
  let distance_3 := 55
  let distance_4 := 70
  let distance_5 := 60
  let total_time := 5
  let total_distance := distance_1 + distance_2 + distance_3 + distance_4 + distance_5
  let average_speed := total_distance / total_time
  average_speed = 59 :=
by
  sorry

end average_speed_trip_l285_285285


namespace quadrilateral_sides_equal_l285_285945

theorem quadrilateral_sides_equal (a b c d : ℕ) (h1 : a ∣ b + c + d) (h2 : b ∣ a + c + d) (h3 : c ∣ a + b + d) (h4 : d ∣ a + b + c) : a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equal_l285_285945


namespace train_more_passengers_l285_285439

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end train_more_passengers_l285_285439


namespace knights_probability_l285_285237

theorem knights_probability (total_knights : ℕ) (chosen_knights : ℕ) (P_num : ℚ) (P_den : ℚ)
  (h1 : total_knights = 30) 
  (h2 : chosen_knights = 4)
  (h3 : P_num = 541)
  (h4 : P_den = 609) :
  P_num + P_den = 1150 := 
by {
  -- Total ways to choose 4 knights from 30
  have H1 : Nat.choose total_knights chosen_knights = 27405,
  sorry,
  
  -- Ways to choose 4 knights with no neighbors
  have H2 : Nat.choose (total_knights - 3 * chosen_knights) chosen_knights = 3060,
  sorry,

  -- Calculate the probability P as a fraction
  have H3 : P_num = 541,
  sorry,

  have H4 : P_den = 609,
  sorry,

  -- Prove that the sum of P_num and P_den is 1150
  sorry
}

end knights_probability_l285_285237


namespace coursework_materials_spending_l285_285659

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end coursework_materials_spending_l285_285659


namespace travis_total_cost_l285_285969

namespace TravelCost

def cost_first_leg : ℝ := 1500
def discount_first_leg : ℝ := 0.25
def fees_first_leg : ℝ := 100

def cost_second_leg : ℝ := 800
def discount_second_leg : ℝ := 0.20
def fees_second_leg : ℝ := 75

def cost_third_leg : ℝ := 1200
def discount_third_leg : ℝ := 0.35
def fees_third_leg : ℝ := 120

def discounted_cost (cost : ℝ) (discount : ℝ) : ℝ :=
  cost - (cost * discount)

def total_leg_cost (cost : ℝ) (discount : ℝ) (fees : ℝ) : ℝ :=
  (discounted_cost cost discount) + fees

def total_journey_cost : ℝ :=
  total_leg_cost cost_first_leg discount_first_leg fees_first_leg + 
  total_leg_cost cost_second_leg discount_second_leg fees_second_leg + 
  total_leg_cost cost_third_leg discount_third_leg fees_third_leg

theorem travis_total_cost : total_journey_cost = 2840 := by
  sorry

end TravelCost

end travis_total_cost_l285_285969


namespace initial_children_count_l285_285393

theorem initial_children_count (passed retake : ℝ) (h_passed : passed = 105.0) (h_retake : retake = 593) : 
    passed + retake = 698 := 
by
  sorry

end initial_children_count_l285_285393


namespace intersection_A_B_l285_285514

def A : Set ℝ := { x | abs x ≤ 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l285_285514


namespace min_value_of_quadratic_l285_285875

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

theorem min_value_of_quadratic : ∃ (x : ℝ), f x = 6 :=
by sorry

end min_value_of_quadratic_l285_285875


namespace youseff_blocks_l285_285567

theorem youseff_blocks (x : ℕ) (h1 : x = 1 * x) (h2 : (20 / 60 : ℚ) * x = x / 3) (h3 : x = x / 3 + 8) : x = 12 := by
  have : x = x := rfl  -- trivial step to include the equality
  sorry

end youseff_blocks_l285_285567


namespace petya_can_reconstruct_numbers_l285_285675

theorem petya_can_reconstruct_numbers (n : ℕ) (h : n % 2 = 1) :
  ∀ (numbers_at_vertices : Fin n → ℕ) (number_at_center : ℕ) (triplets : Fin n → Tuple),
  Petya_can_reconstruct numbers_at_vertices number_at_center triplets :=
sorry

end petya_can_reconstruct_numbers_l285_285675


namespace car_highway_mileage_l285_285254

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end car_highway_mileage_l285_285254


namespace round_trip_time_l285_285094

noncomputable def time_to_complete_trip (speed_without_load speed_with_load distance rest_stops_in_minutes : ℝ) : ℝ :=
  let rest_stops_in_hours := rest_stops_in_minutes / 60
  let half_rest_time := 2 * rest_stops_in_hours
  let total_rest_time := 2 * half_rest_time
  let travel_time_with_load := distance / speed_with_load
  let travel_time_without_load := distance / speed_without_load
  travel_time_with_load + travel_time_without_load + total_rest_time

theorem round_trip_time :
  time_to_complete_trip 13 11 143 30 = 26 :=
sorry

end round_trip_time_l285_285094


namespace ratio_part_to_whole_number_l285_285937

theorem ratio_part_to_whole_number (P N : ℚ) 
  (h1 : (1 / 4) * (1 / 3) * P = 25) 
  (h2 : 0.40 * N = 300) : P / N = 2 / 5 :=
by
  sorry

end ratio_part_to_whole_number_l285_285937


namespace train_speed_l285_285702

def train_length : ℕ := 180
def crossing_time : ℕ := 12

theorem train_speed :
  train_length / crossing_time = 15 := sorry

end train_speed_l285_285702


namespace train_vs_airplane_passenger_capacity_l285_285438

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end train_vs_airplane_passenger_capacity_l285_285438


namespace probability_divisor_of_12_l285_285425

noncomputable def prob_divisor_of_12_rolling_d8 : ℚ :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  favorable_outcomes.to_finset.card / total_outcomes.to_finset.card

theorem probability_divisor_of_12 (h_fair: True) (h_8_sided: True) (h_range: set.Icc 1 8 = {1, 2, 3, 4, 5, 6, 7, 8}) : 
  prob_divisor_of_12_rolling_d8 = 5/8 := 
  sorry

end probability_divisor_of_12_l285_285425


namespace smallest_integer_l285_285734

/-- The smallest integer m such that m > 1 and m has a remainder of 1 when divided by any of 5, 7, and 3 is 106. -/
theorem smallest_integer (m : ℕ) : m > 1 ∧ m % 5 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ↔ m = 106 :=
by
    sorry

end smallest_integer_l285_285734


namespace find_side_b_in_triangle_l285_285190

noncomputable def triangle_side_b (a A : ℝ) (cosB : ℝ) : ℝ :=
  let sinB := Real.sqrt (1 - cosB^2)
  let sinA := Real.sin A
  (a * sinB) / sinA

theorem find_side_b_in_triangle :
  triangle_side_b 5 (Real.pi / 4) (3 / 5) = 4 * Real.sqrt 2 :=
by
  sorry

end find_side_b_in_triangle_l285_285190


namespace find_b_l285_285568

-- Define the conditions of the equations
def condition_1 (x y a : ℝ) : Prop := x * Real.cos a + y * Real.sin a + 3 ≤ 0
def condition_2 (x y b : ℝ) : Prop := x^2 + y^2 + 8 * x - 4 * y - b^2 + 6 * b + 11 = 0

-- Define the proof problem
theorem find_b (b : ℝ) :
  (∀ a x y, condition_1 x y a → condition_2 x y b) →
  b ∈ Set.Iic (-2 * Real.sqrt 5) ∪ Set.Ici (6 + 2 * Real.sqrt 5) :=
by
  sorry

end find_b_l285_285568


namespace find_larger_integer_l285_285565

variable (x : ℤ) (smaller larger : ℤ)
variable (ratio_1_to_4 : smaller = 1 * x ∧ larger = 4 * x)
variable (condition : smaller + 12 = larger)

theorem find_larger_integer : larger = 16 :=
by
  sorry

end find_larger_integer_l285_285565


namespace train_vs_airplane_passenger_capacity_l285_285437

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end train_vs_airplane_passenger_capacity_l285_285437


namespace existence_of_unusual_100_digit_numbers_l285_285851

theorem existence_of_unusual_100_digit_numbers :
  ∃ (n₁ n₂ : ℕ), 
  (n₁ = 10^100 - 1) ∧ (n₂ = 5 * 10^99 - 1) ∧ 
  (∀ x : ℕ, x = n₁ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) ∧
  (∀ x : ℕ, x = n₂ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) := 
sorry

end existence_of_unusual_100_digit_numbers_l285_285851


namespace max_quarters_is_13_l285_285358

noncomputable def number_of_quarters (total_value : ℝ) (quarters nickels dimes : ℝ) : Prop :=
  total_value = 4.55 ∧
  quarters = nickels ∧
  dimes = quarters / 2 ∧
  (0.25 * quarters + 0.05 * nickels + 0.05 * quarters / 2 = 4.55)

theorem max_quarters_is_13 : ∃ q : ℝ, number_of_quarters 4.55 q q (q / 2) ∧ q = 13 :=
by
  sorry

end max_quarters_is_13_l285_285358


namespace area_of_octagon_in_square_l285_285267

theorem area_of_octagon_in_square (perimeter : ℝ) (side_length : ℝ) (area_square : ℝ)
  (segment_length : ℝ) (area_triangle : ℝ) (total_area_triangles : ℝ) :
  perimeter = 144 →
  side_length = perimeter / 4 →
  segment_length = side_length / 3 →
  area_triangle = (segment_length * segment_length) / 2 →
  total_area_triangles = 4 * area_triangle →
  area_square = side_length * side_length →
  (area_square - total_area_triangles) = 1008 :=
by
  sorry

end area_of_octagon_in_square_l285_285267


namespace geometric_mean_of_4_and_9_l285_285535

theorem geometric_mean_of_4_and_9 :
  ∃ b : ℝ, (4 * 9 = b^2) ∧ (b = 6 ∨ b = -6) :=
by
  sorry

end geometric_mean_of_4_and_9_l285_285535


namespace bread_last_days_is_3_l285_285232

-- Define conditions
def num_members : ℕ := 4
def slices_breakfast : ℕ := 3
def slices_snacks : ℕ := 2
def slices_loaf : ℕ := 12
def num_loaves : ℕ := 5

-- Define the problem statement
def bread_last_days : ℕ :=
  (num_loaves * slices_loaf) / (num_members * (slices_breakfast + slices_snacks))

-- State the theorem to be proved
theorem bread_last_days_is_3 : bread_last_days = 3 :=
  sorry

end bread_last_days_is_3_l285_285232


namespace general_term_arithmetic_seq_sum_first_n_terms_geometric_seq_l285_285569

-- Problem 1: Arithmetic sequence
variable {a : ℕ → ℤ}
variable (a₂_eq_0 : a 2 = 0)
variable (a₆_plus_a₈_eq_neg10 : a 6 + a 8 = -10)

theorem general_term_arithmetic_seq :
  ∀ n, a n = 2 - n :=
by
  sorry

-- Problem 2: Geometric sequence
variable {b : ℕ → ℤ}
variable (b1_eq_3 : b 1 = 3)
variable (b2_eq_9 : b 2 = 9)

theorem sum_first_n_terms_geometric_seq :
  ∀ n, (finset.range (n + 1)).sum b = (3 ^ (n + 1) - 3) / 2 :=
by
  sorry

end general_term_arithmetic_seq_sum_first_n_terms_geometric_seq_l285_285569


namespace solve_system_of_equations_l285_285216

theorem solve_system_of_equations (x y z t : ℝ) :
  xy - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18 ↔ (x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∨ t = 0) :=
sorry

end solve_system_of_equations_l285_285216


namespace students_catching_up_on_homework_l285_285785

def total_students : ℕ := 24
def silent_reading_students : ℕ := total_students / 2
def board_games_students : ℕ := total_students / 3

theorem students_catching_up_on_homework : 
  total_students - (silent_reading_students + board_games_students) = 4 := by
  sorry

end students_catching_up_on_homework_l285_285785


namespace polynomial_root_expression_l285_285053

theorem polynomial_root_expression (a b : ℂ) 
  (h₁ : a + b = 5) (h₂ : a * b = 6) : 
  a^4 + a^5 * b^3 + a^3 * b^5 + b^4 = 2905 := by
  sorry

end polynomial_root_expression_l285_285053


namespace wool_production_equivalence_l285_285952

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

end wool_production_equivalence_l285_285952


namespace find_number_l285_285419

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end find_number_l285_285419


namespace spring_work_l285_285701

theorem spring_work :
  ∀ (k : ℝ), (1 = k * 0.01) →
    (∫ x in (0 : ℝ) .. 0.06, k * x) = 0.18 :=
by
  intro k
  intro hk
  have := integral_const_mul
  sorry

end spring_work_l285_285701


namespace max_period_initial_phase_function_l285_285819

theorem max_period_initial_phase_function 
  (A ω ϕ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : A = 1/2) 
  (h2 : ω = 6) 
  (h3 : ϕ = π/4) 
  (h4 : ∀ x, f x = A * Real.sin (ω * x + ϕ)) : 
  ∀ x, f x = (1/2) * Real.sin (6 * x + (π/4)) :=
by
  sorry

end max_period_initial_phase_function_l285_285819


namespace pencils_left_l285_285948

-- Define the initial quantities
def MondayPencils := 35
def TuesdayPencils := 42
def WednesdayPencils := 3 * TuesdayPencils
def WednesdayLoss := 20
def ThursdayPencils := WednesdayPencils / 2
def FridayPencils := 2 * MondayPencils
def WeekendLoss := 50

-- Define the total number of pencils Sarah has at the end of each day
def TotalMonday := MondayPencils
def TotalTuesday := TotalMonday + TuesdayPencils
def TotalWednesday := TotalTuesday + WednesdayPencils - WednesdayLoss
def TotalThursday := TotalWednesday + ThursdayPencils
def TotalFriday := TotalThursday + FridayPencils
def TotalWeekend := TotalFriday - WeekendLoss

-- The proof statement
theorem pencils_left : TotalWeekend = 266 :=
by
  sorry

end pencils_left_l285_285948


namespace p_is_sufficient_but_not_necessary_for_q_l285_285478

-- Definitions and conditions
def p (x : ℝ) : Prop := (x = 1)
def q (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0)

-- Theorem statement
theorem p_is_sufficient_but_not_necessary_for_q : ∀ x : ℝ, (p x → q x) ∧ (¬ (q x → p x)) :=
by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l285_285478


namespace yield_and_fertilization_correlated_l285_285413

-- Define the variables and conditions
def yield_of_crops : Type := sorry
def fertilization : Type := sorry

-- State the condition
def yield_depends_on_fertilization (Y : yield_of_crops) (F : fertilization) : Prop :=
  -- The yield of crops depends entirely on fertilization
  sorry

-- State the theorem with the given condition and the conclusion
theorem yield_and_fertilization_correlated {Y : yield_of_crops} {F : fertilization} :
  yield_depends_on_fertilization Y F → sorry := 
  -- There is a correlation between the yield of crops and fertilization
  sorry

end yield_and_fertilization_correlated_l285_285413


namespace opposite_of_neg3_l285_285383

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l285_285383


namespace tangent_parallel_l285_285897

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

theorem tangent_parallel (a : ℝ) (H : ∀ x1 : ℝ, ∃ x2 : ℝ, (a - 2 * Real.sin x1) = (-Real.exp x2 - 1)) :
  a < -3 := by
  sorry

end tangent_parallel_l285_285897


namespace Jessie_weight_l285_285195

theorem Jessie_weight (c l w : ℝ) (hc : c = 27) (hl : l = 101) : c + l = w ↔ w = 128 := by
  sorry

end Jessie_weight_l285_285195


namespace relationship_between_p_and_q_l285_285068

variable (x y : ℝ)

def p := x * y ≥ 0
def q := |x + y| = |x| + |y|

theorem relationship_between_p_and_q : (p x y ↔ q x y) :=
sorry

end relationship_between_p_and_q_l285_285068


namespace part1_solution_set_part2_value_of_t_l285_285802

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

-- Statement for the equivalent proof problem
theorem part1_solution_set (x : ℝ) : 
  (t = 3 → f 3 x > 0 ↔ (x < 1) ∨ (x > 3)) :=
by
  sorry

theorem part2_value_of_t (t : ℝ) :
  (∀ x : ℝ, f t x ≥ 0) → t = 1 :=
by
  sorry

end part1_solution_set_part2_value_of_t_l285_285802


namespace alcohol_percentage_in_original_solution_l285_285981

theorem alcohol_percentage_in_original_solution
  (P : ℚ)
  (alcohol_in_new_mixture : ℚ)
  (original_solution_volume : ℚ)
  (added_water_volume : ℚ)
  (new_mixture_volume : ℚ)
  (percentage_in_new_mixture : ℚ) :
  original_solution_volume = 11 →
  added_water_volume = 3 →
  new_mixture_volume = original_solution_volume + added_water_volume →
  percentage_in_new_mixture = 33 →
  alcohol_in_new_mixture = (percentage_in_new_mixture / 100) * new_mixture_volume →
  (P / 100) * original_solution_volume = alcohol_in_new_mixture →
  P = 42 :=
by
  sorry

end alcohol_percentage_in_original_solution_l285_285981


namespace savings_after_expense_increase_l285_285577

-- Define the conditions
def monthly_salary : ℝ := 6500
def initial_savings_percentage : ℝ := 0.20
def increase_expenses_percentage : ℝ := 0.20

-- Define the statement we want to prove
theorem savings_after_expense_increase :
  (monthly_salary - (monthly_salary - (initial_savings_percentage * monthly_salary) + (increase_expenses_percentage * (monthly_salary - (initial_savings_percentage * monthly_salary))))) = 260 :=
sorry

end savings_after_expense_increase_l285_285577


namespace num_valid_five_digit_numbers_l285_285680

-- Conditions
def S1 : Finset ℕ := {1, 3, 5}
def S2 : Finset ℕ := {2, 4, 6, 8}

-- Question: Number of valid five-digit numbers
theorem num_valid_five_digit_numbers :
  let num_ways := (S2.card.choose 3) * (S1.card.choose 2) * 3 * ((Finset.range 5).card.factorial / (Finset.range 1).card.factorial) in
  num_ways = 864 :=
by sorry

end num_valid_five_digit_numbers_l285_285680


namespace graveling_cost_correct_l285_285840

-- Define the dimensions of the rectangular lawn
def lawn_length : ℕ := 80 -- in meters
def lawn_breadth : ℕ := 50 -- in meters

-- Define the width of each road
def road_width : ℕ := 10 -- in meters

-- Define the cost per square meter for graveling the roads
def cost_per_sq_m : ℕ := 3 -- in Rs. per sq meter

-- Define the area of the road parallel to the length of the lawn
def area_road_parallel_length : ℕ := lawn_length * road_width

-- Define the effective length of the road parallel to the breadth of the lawn
def effective_road_parallel_breadth_length : ℕ := lawn_breadth - road_width

-- Define the area of the road parallel to the breadth of the lawn
def area_road_parallel_breadth : ℕ := effective_road_parallel_breadth_length * road_width

-- Define the total area to be graveled
def total_area_to_be_graveled : ℕ := area_road_parallel_length + area_road_parallel_breadth

-- Define the total cost of graveling
def total_graveling_cost : ℕ := total_area_to_be_graveled * cost_per_sq_m

-- Theorem: The total cost of graveling the two roads is Rs. 3600
theorem graveling_cost_correct : total_graveling_cost = 3600 := 
by
  unfold total_graveling_cost total_area_to_be_graveled area_road_parallel_length area_road_parallel_breadth effective_road_parallel_breadth_length lawn_length lawn_breadth road_width cost_per_sq_m
  exact rfl

end graveling_cost_correct_l285_285840


namespace intersection_A_B_l285_285516

def A : Set ℝ := {y | ∃ x : ℝ, y = x ^ (1 / 3)}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_B :
  A ∩ B = {x | x > 1} :=
sorry

end intersection_A_B_l285_285516


namespace rachel_older_than_leah_l285_285356

theorem rachel_older_than_leah (rachel_age leah_age : ℕ) (h1 : rachel_age = 19) (h2 : rachel_age + leah_age = 34) :
  rachel_age - leah_age = 4 :=
by sorry

end rachel_older_than_leah_l285_285356


namespace larger_number_is_1629_l285_285221

theorem larger_number_is_1629 (x y : ℕ) (h1 : y - x = 1360) (h2 : y = 6 * x + 15) : y = 1629 := 
by 
  sorry

end larger_number_is_1629_l285_285221


namespace inverse_of_problem_matrix_is_zero_matrix_l285_285739

def det (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

def zero_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 0], ![0, 0]]

noncomputable def problem_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -6], ![-2, 3]]

theorem inverse_of_problem_matrix_is_zero_matrix :
  det problem_matrix = 0 → problem_matrix⁻¹ = zero_matrix :=
by
  intro h
  -- Proof steps will be written here
  sorry

end inverse_of_problem_matrix_is_zero_matrix_l285_285739


namespace apples_in_market_l285_285121

-- Define variables for the number of apples and oranges
variables (A O : ℕ)

-- Given conditions
def condition1 : Prop := A = O + 27
def condition2 : Prop := A + O = 301

-- Theorem statement
theorem apples_in_market (h1 : condition1) (h2 : condition2) : A = 164 :=
by sorry

end apples_in_market_l285_285121


namespace phone_numbers_divisible_by_13_l285_285770

theorem phone_numbers_divisible_by_13 :
  ∃ (x y z : ℕ), (x < 10) ∧ (y < 10) ∧ (z < 10) ∧ (100 * x + 10 * y + z) % 13 = 0 ∧ (2 * y = x + z) :=
  sorry

end phone_numbers_divisible_by_13_l285_285770


namespace opposite_of_neg3_l285_285384

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l285_285384


namespace circle_tangent_proof_l285_285076

noncomputable def circle_tangent_range : Set ℝ :=
  { k : ℝ | k > 0 ∧ ((3 - 2 * k)^2 + (1 - k)^2 > k) }

theorem circle_tangent_proof :
  ∀ k > 0, ((3 - 2 * k)^2 + (1 - k)^2 > k) ↔ (k ∈ (Set.Ioo 0 1 ∪ Set.Ioi 2)) :=
by
  sorry

end circle_tangent_proof_l285_285076


namespace perpendicular_vectors_l285_285626

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l285_285626


namespace john_caffeine_consumption_l285_285660

noncomputable def caffeine_consumed : ℝ :=
let drink1_ounces : ℝ := 12
let drink1_caffeine : ℝ := 250
let drink2_ratio : ℝ := 3
let drink2_ounces : ℝ := 2

-- Calculate caffeine per ounce in the first drink
let caffeine1_per_ounce : ℝ := drink1_caffeine / drink1_ounces

-- Calculate caffeine per ounce in the second drink
let caffeine2_per_ounce : ℝ := caffeine1_per_ounce * drink2_ratio

-- Calculate total caffeine in the second drink
let drink2_caffeine : ℝ := caffeine2_per_ounce * drink2_ounces

-- Total caffeine from both drinks
let total_drinks_caffeine : ℝ := drink1_caffeine + drink2_caffeine

-- Caffeine in the pill is as much as the total from both drinks
let pill_caffeine : ℝ := total_drinks_caffeine

-- Total caffeine consumed
(drink1_caffeine + drink2_caffeine) + pill_caffeine

theorem john_caffeine_consumption :
  caffeine_consumed = 749.96 := by
    -- Proof is omitted
    sorry

end john_caffeine_consumption_l285_285660


namespace total_silver_dollars_l285_285935

-- Definitions based on conditions
def chiu_silver_dollars : ℕ := 56
def phung_silver_dollars : ℕ := chiu_silver_dollars + 16
def ha_silver_dollars : ℕ := phung_silver_dollars + 5

-- Theorem statement
theorem total_silver_dollars : chiu_silver_dollars + phung_silver_dollars + ha_silver_dollars = 205 :=
by
  -- We use "sorry" to fill in the proof part as instructed
  sorry

end total_silver_dollars_l285_285935


namespace climbing_difference_l285_285335

theorem climbing_difference (rate_matt rate_jason time : ℕ) (h_rate_matt : rate_matt = 6) (h_rate_jason : rate_jason = 12) (h_time : time = 7) : 
  rate_jason * time - rate_matt * time = 42 :=
by
  sorry

end climbing_difference_l285_285335


namespace number_of_even_results_l285_285259

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l285_285259


namespace remainder_pow_700_eq_one_l285_285243

theorem remainder_pow_700_eq_one (number : ℤ) (h : number ^ 700 % 100 = 1) : number ^ 700 % 100 = 1 :=
  by
  exact h

end remainder_pow_700_eq_one_l285_285243


namespace dalmatians_with_right_ear_spots_l285_285938

def TotalDalmatians := 101
def LeftOnlySpots := 29
def RightOnlySpots := 17
def NoEarSpots := 22

theorem dalmatians_with_right_ear_spots : 
  (TotalDalmatians - LeftOnlySpots - NoEarSpots) = 50 :=
by
  -- Proof goes here, but for now, we use sorry
  sorry

end dalmatians_with_right_ear_spots_l285_285938


namespace max_value_k_l285_285613

theorem max_value_k (x y : ℝ) (k : ℝ) (h₁ : x^2 + y^2 = 1) (h₂ : ∀ x y, x^2 + y^2 = 1 → x + y - k ≥ 0) : 
  k ≤ -Real.sqrt 2 :=
sorry

end max_value_k_l285_285613


namespace probability_second_third_different_colors_l285_285392

def probability_different_colors (blue_chips : ℕ) (red_chips : ℕ) (yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + red_chips + yellow_chips
  let prob_diff :=
    ((blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips)) +
    ((red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips)) +
    ((yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips))
  prob_diff

theorem probability_second_third_different_colors :
  probability_different_colors 7 6 5 = 107 / 162 :=
by
  sorry

end probability_second_third_different_colors_l285_285392


namespace remainder_x14_minus_1_div_x_plus_1_l285_285558

-- Define the polynomial f(x) = x^14 - 1
def f (x : ℝ) := x^14 - 1

-- Statement to prove that the remainder when f(x) is divided by x + 1 is 0
theorem remainder_x14_minus_1_div_x_plus_1 : f (-1) = 0 :=
by
  -- This is where the proof would go, but for now, we will just use sorry
  sorry

end remainder_x14_minus_1_div_x_plus_1_l285_285558


namespace find_interest_rate_l285_285412

theorem find_interest_rate (P r : ℝ) 
  (h1 : 460 = P * (1 + 3 * r)) 
  (h2 : 560 = P * (1 + 8 * r)) : 
  r = 0.05 :=
by
  sorry

end find_interest_rate_l285_285412


namespace greatest_common_divisor_is_one_l285_285241

-- Define the expressions for a and b
def a : ℕ := 114^2 + 226^2 + 338^2
def b : ℕ := 113^2 + 225^2 + 339^2

-- Now state that the gcd of a and b is 1
theorem greatest_common_divisor_is_one : Nat.gcd a b = 1 := sorry

end greatest_common_divisor_is_one_l285_285241


namespace indeterminate_4wheelers_l285_285703

-- Define conditions and the main theorem to state that the number of 4-wheelers cannot be uniquely determined.
theorem indeterminate_4wheelers (x y : ℕ) (h : 2 * x + 4 * y = 58) : ∃ k : ℤ, y = ((29 : ℤ) - k - x) / 2 :=
by
  sorry

end indeterminate_4wheelers_l285_285703


namespace base7_to_base10_and_frac_l285_285105

theorem base7_to_base10_and_frac (c d e : ℕ) 
  (h1 : (761 : ℕ) = 7^2 * 7 + 6 * 7^1 + 1 * 7^0)
  (h2 : (10 * 10 * c + 10 * d + e) = 386)
  (h3 : c = 3)
  (h4 : d = 8)
  (h5 : e = 6) :
  (d * e) / 15 = 48 / 15 := 
sorry

end base7_to_base10_and_frac_l285_285105


namespace solve_for_x_l285_285416

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end solve_for_x_l285_285416


namespace problem_statement_l285_285461

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end problem_statement_l285_285461


namespace infinite_solutions_to_congruence_l285_285355

theorem infinite_solutions_to_congruence :
  ∃ᶠ n in atTop, 3^((n-2)^(n-1)-1) ≡ 1 [MOD 17 * n^2] :=
by
  sorry

end infinite_solutions_to_congruence_l285_285355


namespace correct_average_l285_285955

theorem correct_average (incorrect_avg : ℝ) (num_values : ℕ) (misread_value actual_value : ℝ) 
  (h1 : incorrect_avg = 16) 
  (h2 : num_values = 10)
  (h3 : misread_value = 26)
  (h4 : actual_value = 46) : 
  (incorrect_avg * num_values + (actual_value - misread_value)) / num_values = 18 := 
by
  sorry

end correct_average_l285_285955


namespace jordyn_total_cost_l285_285322

theorem jordyn_total_cost (
  price_cherries : ℕ := 5,
  price_olives : ℕ := 7,
  discount : ℕ := 10,
  quantity : ℕ := 50
) : (50 * (price_cherries - (price_cherries * discount / 100)) + 50 * (price_olives - (price_olives * discount / 100))) = 540 :=
by
  sorry

end jordyn_total_cost_l285_285322


namespace squares_expression_l285_285979

theorem squares_expression (a : ℕ) : 
  a^2 + 5*a + 7 = (a+3) * (a+2)^2 + (a+2) * 1^2 := 
by
  sorry

end squares_expression_l285_285979


namespace find_number_l285_285954

theorem find_number :
  ∃ x : ℝ, (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 ∧ x = 20 :=
by
  sorry

end find_number_l285_285954


namespace find_x_minus_y_l285_285930

theorem find_x_minus_y {x y z : ℤ} (h1 : x - (y + z) = 5) (h2 : x - y + z = -1) : x - y = 2 :=
by
  sorry

end find_x_minus_y_l285_285930


namespace evaluate_f_g_f_l285_285197

-- Define f(x)
def f (x : ℝ) : ℝ := 4 * x + 4

-- Define g(x)
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem we're proving
theorem evaluate_f_g_f : f (g (f 3)) = 1360 := by
  sorry

end evaluate_f_g_f_l285_285197


namespace find_m_l285_285628

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l285_285628


namespace Marty_painting_combinations_l285_285672

theorem Marty_painting_combinations :
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  (parts_of_room * colors * methods) = 30 := 
by
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  show (parts_of_room * colors * methods) = 30
  sorry

end Marty_painting_combinations_l285_285672


namespace conclusion_2_conclusion_3_conclusion_4_l285_285808

variable (b : ℝ)

def f (x : ℝ) : ℝ := x^2 - |b| * x - 3

theorem conclusion_2 (h_min : ∃ x, f b x = -3) : b = 0 :=
  sorry

theorem conclusion_3 (h_b : b = -2) (x : ℝ) (hx : -2 < x ∧ x < 2) :
    -4 ≤ f b x ∧ f b x ≤ -3 :=
  sorry

theorem conclusion_4 (hb_ne : b ≠ 0) (m : ℝ) (h_roots : ∃ x1 x2, f b x1 = m ∧ f b x2 = m ∧ x1 ≠ x2) :
    m > -3 ∨ b^2 = -4 * m - 12 :=
  sorry

end conclusion_2_conclusion_3_conclusion_4_l285_285808


namespace smallest_number_is_minus_three_l285_285589

theorem smallest_number_is_minus_three :
  ∀ (a b c d : ℤ), (a = 0) → (b = -3) → (c = 1) → (d = -1) → b < d ∧ d < a ∧ a < c → b = -3 :=
by
  intros a b c d ha hb hc hd h
  exact hb

end smallest_number_is_minus_three_l285_285589


namespace scientific_notation_correctness_l285_285012

theorem scientific_notation_correctness : ∃ x : ℝ, x = 8.2 ∧ (8200000 : ℝ) = x * 10^6 :=
by
  use 8.2
  split
  · rfl
  · sorry

end scientific_notation_correctness_l285_285012


namespace even_combinations_after_six_operations_l285_285255

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l285_285255


namespace bread_last_days_l285_285233

theorem bread_last_days (num_members : ℕ) (breakfast_slices : ℕ) (snack_slices : ℕ) (slices_per_loaf : ℕ) (num_loaves : ℕ) :
  num_members = 4 →
  breakfast_slices = 3 →
  snack_slices = 2 →
  slices_per_loaf = 12 →
  num_loaves = 5 →
  (num_loaves * slices_per_loaf) / (num_members * (breakfast_slices + snack_slices)) = 3 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end bread_last_days_l285_285233


namespace minimum_planks_required_l285_285692

theorem minimum_planks_required (colors : Finset ℕ) (planks : List ℕ) :
  colors.card = 100 ∧
  ∀ i j, i ∈ colors → j ∈ colors → i ≠ j →
  ∃ k₁ k₂, k₁ < k₂ ∧ planks.get? k₁ = some i ∧ planks.get? k₂ = some j
  → planks.length = 199 := 
sorry

end minimum_planks_required_l285_285692


namespace remainder_of_7_pow_51_mod_8_l285_285835

theorem remainder_of_7_pow_51_mod_8 : (7^51 % 8) = 7 := sorry

end remainder_of_7_pow_51_mod_8_l285_285835


namespace evaluate_powers_of_i_mod_4_l285_285046

theorem evaluate_powers_of_i_mod_4 :
  (Complex.I ^ 48 + Complex.I ^ 96 + Complex.I ^ 144) = 3 := by
  sorry

end evaluate_powers_of_i_mod_4_l285_285046


namespace parker_bed_time_l285_285519

def sleep_duration : Time := Time.mk 7 0 0

def wake_up_time : Time := Time.mk 9 0 0

def bed_time (wake_up : Time) (sleep_duration : Time) : Time :=
  wake_up - sleep_duration

theorem parker_bed_time : bed_time wake_up_time sleep_duration = Time.mk 2 0 0 :=
sorry

end parker_bed_time_l285_285519


namespace ana_bonita_age_difference_l285_285863

theorem ana_bonita_age_difference (A B n : ℕ) 
  (h1 : A = B + n)
  (h2 : A - 1 = 7 * (B - 1))
  (h3 : A = B^3) : 
  n = 6 :=
sorry

end ana_bonita_age_difference_l285_285863


namespace radio_show_songs_duration_l285_285436

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_songs_duration_l285_285436


namespace quadratic_eq_one_solution_has_ordered_pair_l285_285541

theorem quadratic_eq_one_solution_has_ordered_pair (a c : ℝ) 
  (h1 : a * c = 25) 
  (h2 : a + c = 17) 
  (h3 : a > c) : 
  (a, c) = (15.375, 1.625) :=
sorry

end quadratic_eq_one_solution_has_ordered_pair_l285_285541


namespace exists_same_color_points_at_unit_distance_l285_285709

theorem exists_same_color_points_at_unit_distance
  (color : ℝ × ℝ → ℕ)
  (coloring : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_at_unit_distance_l285_285709


namespace max_books_borrowed_l285_285022

theorem max_books_borrowed (total_students books_per_student : ℕ) (students_with_no_books: ℕ) (students_with_one_book students_with_two_books: ℕ) (rest_at_least_three_books students : ℕ) :
  total_students = 20 →
  books_per_student = 2 →
  students_with_no_books = 2 →
  students_with_one_book = 8 →
  students_with_two_books = 3 →
  rest_at_least_three_books = total_students - (students_with_no_books + students_with_one_book + students_with_two_books) →
  (students_with_no_books * 0 + students_with_one_book * 1 + students_with_two_books * 2 + students * books_per_student = total_students * books_per_student) →
  (students * 3 + some_student_max = 26) →
  some_student_max ≥ 8 :=
by
  introv h1 h2 h3 h4 h5 h6 h7
  sorry

end max_books_borrowed_l285_285022


namespace random_event_proof_l285_285989

def statement_A := "Strong youth leads to a strong country"
def statement_B := "Scooping the moon in the water"
def statement_C := "Waiting by the stump for a hare"
def statement_D := "Green waters and lush mountains are mountains of gold and silver"

def is_random_event (statement : String) : Prop :=
statement = statement_C

theorem random_event_proof : is_random_event statement_C :=
by
  -- Based on the analysis in the problem, Statement C is determined to be random.
  sorry

end random_event_proof_l285_285989


namespace evaluate_expression_l285_285879

variable (y : ℕ)

theorem evaluate_expression (h : y = 3) : 
    (y^(1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) / y^(2 + 4 + 6 + 8 + 10 + 12)) = 3^58 :=
by
  -- Proof will be done here
  sorry

end evaluate_expression_l285_285879


namespace gallons_of_gas_l285_285792

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end gallons_of_gas_l285_285792


namespace factor_quadratic_l285_285737

theorem factor_quadratic : ∀ (x : ℝ), 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 :=
by
  intro x
  sorry

end factor_quadratic_l285_285737


namespace coursework_materials_spending_l285_285658

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end coursework_materials_spending_l285_285658


namespace opposite_of_neg3_l285_285380

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l285_285380


namespace final_grade_calculation_l285_285860

theorem final_grade_calculation
  (exam_score homework_score class_participation_score : ℝ)
  (exam_weight homework_weight participation_weight : ℝ)
  (h_exam_score : exam_score = 90)
  (h_homework_score : homework_score = 85)
  (h_class_participation_score : class_participation_score = 80)
  (h_exam_weight : exam_weight = 3)
  (h_homework_weight : homework_weight = 2)
  (h_participation_weight : participation_weight = 5) :
  (exam_score * exam_weight + homework_score * homework_weight + class_participation_score * participation_weight) /
  (exam_weight + homework_weight + participation_weight) = 84 :=
by
  -- The proof would go here
  sorry

end final_grade_calculation_l285_285860


namespace hands_coincide_again_l285_285113

-- Define the angular speeds of minute and hour hands
def speed_minute_hand : ℝ := 6
def speed_hour_hand : ℝ := 0.5

-- Define the initial condition: coincidence at midnight
def initial_time : ℝ := 0

-- Define the function that calculates the angle of the minute hand at time t
def angle_minute_hand (t : ℝ) : ℝ := speed_minute_hand * t

-- Define the function that calculates the angle of the hour hand at time t
def angle_hour_hand (t : ℝ) : ℝ := speed_hour_hand * t

-- Define the time at which the hands coincide again after midnight
noncomputable def coincidence_time : ℝ := 720 / 11

-- The proof problem statement: The hands coincide again at coincidence_time minutes
theorem hands_coincide_again : 
  angle_minute_hand coincidence_time = angle_hour_hand coincidence_time + 360 :=
sorry

end hands_coincide_again_l285_285113


namespace part1_part2_l285_285620

noncomputable def f : ℝ → ℝ := sorry

variable (x y : ℝ)
variable (hx0 : 0 < x)
variable (hy0 : 0 < y)
variable (hx12 : x < 1 → f x > 0)
variable (hf_half : f (1 / 2) = 1)
variable (hf_mul : f (x * y) = f x + f y)

theorem part1 : (∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) := sorry

theorem part2 : (∀ x, 3 < x → x < 4 → f (x - 3) > f (1 / x) - 2) := sorry

end part1_part2_l285_285620


namespace solve_for_x_l285_285700

theorem solve_for_x :
  ∃ x : ℝ, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 :=
by
  sorry

end solve_for_x_l285_285700


namespace probability_exactly_four_1s_l285_285552

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_four_1s_in_12_dice : ℝ :=
  let n := 12
  let k := 4
  let p := 1 / 6
  let q := 5 / 6
  (binomial_coefficient n k : ℝ) * p^k * q^(n - k)

theorem probability_exactly_four_1s : probability_of_four_1s_in_12_dice ≈ 0.089 :=
  by
  sorry

end probability_exactly_four_1s_l285_285552


namespace biased_coin_probability_l285_285572

noncomputable def probability_exactly_two_heads_in_four_tosses (p : ℚ) : ℚ :=
  (nat.choose 4 2) * p^2 * (1 - p)^(4 - 2)

theorem biased_coin_probability (p : ℚ) (v : ℚ) (h_p : p = 3/5) : v = 216/625 :=
  by
  have h_v : probability_exactly_two_heads_in_four_tosses p = v := sorry
  rw [h_p] at h_v
  sorry

end biased_coin_probability_l285_285572


namespace finding_f_of_neg_half_l285_285178

def f (x : ℝ) : ℝ := sorry

theorem finding_f_of_neg_half : f (-1/2) = Real.pi / 3 :=
by
  -- Given function definition condition: f (cos x) = x / 2 for 0 ≤ x ≤ π
  -- f should be defined on ℝ -> ℝ such that this condition holds;
  -- Applying this condition should verify our theorem.
  sorry

end finding_f_of_neg_half_l285_285178


namespace track_length_l285_285448

theorem track_length (y : ℝ) 
  (H1 : ∀ b s : ℝ, b + s = y ∧ b = y / 2 - 120 ∧ s = 120)
  (H2 : ∀ b s : ℝ, b + s = y + 180 ∧ b = y / 2 + 60 ∧ s = y / 2 - 60) :
  y = 600 :=
by 
  sorry

end track_length_l285_285448


namespace n_multiple_of_40_and_infinite_solutions_l285_285209

theorem n_multiple_of_40_and_infinite_solutions 
  (n : ℤ)
  (h1 : ∃ k₁ : ℤ, 2 * n + 1 = k₁^2)
  (h2 : ∃ k₂ : ℤ, 3 * n + 1 = k₂^2)
  : ∃ (m : ℤ), n = 40 * m ∧ ∃ (seq : ℕ → ℤ), 
    (∀ i : ℕ, ∃ k₁ k₂ : ℤ, (2 * (seq i) + 1 = k₁^2) ∧ (3 * (seq i) + 1 = k₂^2) ∧ 
     (i ≠ 0 → seq i ≠ seq (i - 1))) :=
by sorry

end n_multiple_of_40_and_infinite_solutions_l285_285209


namespace relationship_between_M_and_N_l285_285772
   
   variable (x : ℝ)
   def M := 2*x^2 - 12*x + 15
   def N := x^2 - 8*x + 11
   
   theorem relationship_between_M_and_N : M x ≥ N x :=
   by
     sorry
   
end relationship_between_M_and_N_l285_285772


namespace dice_probability_exactly_four_ones_l285_285551

noncomputable def dice_probability : ℚ := 
  (Nat.choose 12 4) * (1/6)^4 * (5/6)^8

theorem dice_probability_exactly_four_ones : (dice_probability : ℚ) ≈ 0.089 :=
  by sorry -- Skip the proof. 

#eval (dice_probability : ℚ)

end dice_probability_exactly_four_ones_l285_285551


namespace complement_intersection_l285_285310

section SetTheory

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 3}) (hB : B = {2, 3, 4}) : 
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end SetTheory

end complement_intersection_l285_285310


namespace circumscribed_circle_center_location_l285_285497

structure Trapezoid where
  is_isosceles : Bool
  angle_base : ℝ
  angle_between_diagonals : ℝ

theorem circumscribed_circle_center_location (T : Trapezoid)
  (h1 : T.is_isosceles = true)
  (h2 : T.angle_base = 50)
  (h3 : T.angle_between_diagonals = 40) :
  ∃ loc : String, loc = "Outside" := by
  sorry

end circumscribed_circle_center_location_l285_285497


namespace response_rate_increase_l285_285020

theorem response_rate_increase :
  let original_customers := 70
  let original_responses := 7
  let redesigned_customers := 63
  let redesigned_responses := 9
  let original_response_rate := (original_responses : ℝ) / original_customers
  let redesigned_response_rate := (redesigned_responses : ℝ) / redesigned_customers
  let percentage_increase := ((redesigned_response_rate - original_response_rate) / original_response_rate) * 100
  abs (percentage_increase - 42.86) < 0.01 :=
by
  sorry

end response_rate_increase_l285_285020


namespace calculate_s_at_2_l285_285090

-- Given definitions
def t (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 1
def s (p : ℝ) : ℝ := p^3 - 4 * p^2 + p + 6

-- The target statement
theorem calculate_s_at_2 : s 2 = ((5 + Real.sqrt 33) / 4)^3 - 4 * ((5 + Real.sqrt 33) / 4)^2 + ((5 + Real.sqrt 33) / 4) + 6 := 
by 
  sorry

end calculate_s_at_2_l285_285090


namespace total_amount_spent_l285_285934

-- Definitions for problem conditions
def mall_spent_before_discount : ℝ := 250
def clothes_discount_percent : ℝ := 0.15
def mall_tax_percent : ℝ := 0.08

def movie_ticket_price : ℝ := 24
def num_movies : ℝ := 3
def ticket_discount_percent : ℝ := 0.10
def movie_tax_percent : ℝ := 0.05

def beans_price : ℝ := 1.25
def num_beans : ℝ := 20
def cucumber_price : ℝ := 2.50
def num_cucumbers : ℝ := 5
def tomato_price : ℝ := 5.00
def num_tomatoes : ℝ := 3
def pineapple_price : ℝ := 6.50
def num_pineapples : ℝ := 2
def market_tax_percent : ℝ := 0.07

-- Proof statement
theorem total_amount_spent :
  let mall_spent_after_discount := mall_spent_before_discount * (1 - clothes_discount_percent)
  let mall_tax := mall_spent_after_discount * mall_tax_percent
  let total_mall_spent := mall_spent_after_discount + mall_tax

  let total_ticket_cost_before_discount := num_movies * movie_ticket_price
  let ticket_cost_after_discount := total_ticket_cost_before_discount * (1 - ticket_discount_percent)
  let movie_tax := ticket_cost_after_discount * movie_tax_percent
  let total_movie_spent := ticket_cost_after_discount + movie_tax

  let total_beans_cost := num_beans * beans_price
  let total_cucumbers_cost := num_cucumbers * cucumber_price
  let total_tomatoes_cost := num_tomatoes * tomato_price
  let total_pineapples_cost := num_pineapples * pineapple_price
  let total_market_spent_before_tax := total_beans_cost + total_cucumbers_cost + total_tomatoes_cost + total_pineapples_cost
  let market_tax := total_market_spent_before_tax * market_tax_percent
  let total_market_spent := total_market_spent_before_tax + market_tax
  
  let total_spent := total_mall_spent + total_movie_spent + total_market_spent
  total_spent = 367.63 :=
by
  sorry

end total_amount_spent_l285_285934


namespace number_of_girls_attending_picnic_l285_285182

variables (g b : ℕ)

def hms_conditions : Prop :=
  g + b = 1500 ∧ (3 / 4 : ℝ) * g + (3 / 5 : ℝ) * b = 975

theorem number_of_girls_attending_picnic (h : hms_conditions g b) : (3 / 4 : ℝ) * g = 375 :=
sorry

end number_of_girls_attending_picnic_l285_285182


namespace eccentricity_is_square_root_six_divided_by_three_l285_285427

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_is_square_root_six_divided_by_three
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (A B : ℝ × ℝ) (C : ℝ)
  (hA : A = (-a, 0))
  (hB : ∃ x1 y1, B = (x1, y1) ∧ ∃ t, C = (0, t) ∧ 
    (x1 / a)^2 + (y1 / b)^2 = 1 ∧ y1 = (1/3) * (x1 + a))
  (hABC : |(C - A) - (C - B)| = 0 ∧ angle C A B = π/2) :
  eccentricity a b = (Real.sqrt 6) / 3 :=
sorry

end eccentricity_is_square_root_six_divided_by_three_l285_285427


namespace number_of_true_propositions_l285_285825

def inverse_proposition (x y : ℝ) : Prop :=
  ¬(x + y = 0 → (x ≠ -y))

def contrapositive_proposition (a b : ℝ) : Prop :=
  (a^2 ≤ b^2) → (a ≤ b)

def negation_proposition (x : ℝ) : Prop :=
  (x ≤ -3) → ¬(x^2 + x - 6 > 0)

theorem number_of_true_propositions : 
  (∃ (x y : ℝ), inverse_proposition x y) ∧
  (∃ (a b : ℝ), contrapositive_proposition a b) ∧
  ¬(∃ (x : ℝ), negation_proposition x) → 
  2 = 2 :=
by
  sorry

end number_of_true_propositions_l285_285825


namespace sum_of_reciprocals_of_roots_l285_285661

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (h : ∀ x : ℝ, (x^3 - x - 6 = 0) → (x = p ∨ x = q ∨ x = r)) :
  1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12 :=
sorry

end sum_of_reciprocals_of_roots_l285_285661


namespace meeting_point_ratio_l285_285586

theorem meeting_point_ratio (v1 v2 : ℝ) (TA TB : ℝ)
  (h1 : TA = 45 * v2)
  (h2 : TB = 20 * v1)
  (h3 : (TA / v1) - (TB / v2) = 11) :
  TA / TB = 9 / 5 :=
by sorry

end meeting_point_ratio_l285_285586


namespace probability_exactly_four_ones_is_090_l285_285554
open Float (approxEq)

def dice_probability_exactly_four_ones : Float :=
  let n := 12
  let k := 4
  let p_one := (1 / 6 : Float)
  let p_not_one := (5 / 6 : Float)
  let combination := ((n.factorial) / (k.factorial * (n - k).factorial) : Float)
  let probability := combination * (p_one ^ k) * (p_not_one ^ (n - k))
  probability

theorem probability_exactly_four_ones_is_090 : dice_probability_exactly_four_ones ≈ 0.090 :=
  sorry

end probability_exactly_four_ones_is_090_l285_285554


namespace find_m_for_q_find_m_for_pq_l285_285312

variable (m : ℝ)

-- Statement q: The equation represents a hyperbola if and only if m > 3
def q (m : ℝ) : Prop := m > 3

-- Statement p: The inequality holds if and only if m >= 1
def p (m : ℝ) : Prop := m ≥ 1

-- 1. If statement q is true, find the range of values for m.
theorem find_m_for_q (h : q m) : m > 3 := by
  exact h

-- 2. If (p ∨ q) is true and (p ∧ q) is false, find the range of values for m.
theorem find_m_for_pq (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end find_m_for_q_find_m_for_pq_l285_285312


namespace number_of_tangent_and_parallel_lines_l285_285576

theorem number_of_tangent_and_parallel_lines (p : ℝ × ℝ) (a : ℝ) (h : p = (2, 4)) (hp_on_parabola : (p.1)^2 = 8 * p.2) :
  ∃ l1 l2 : (ℝ × ℝ) → Prop, 
    (l1 (2, 4) ∧ l2 (2, 4)) ∧ 
    (∀ l, (l = l1 ∨ l = l2) ↔ (∃ q, q ≠ p ∧ q ∈ {p' | (p'.1)^2 = 8 * p'.2})) ∧ 
    (∀ p' ∈ {p' | (p'.1)^2 = 8 * p'.2}, (l1 p' ∨ l2 p') → False) :=
sorry

end number_of_tangent_and_parallel_lines_l285_285576


namespace find_2a_2b_2c_2d_l285_285217

open Int

theorem find_2a_2b_2c_2d (a b c d : ℤ) 
  (h1 : a - b + c = 7) 
  (h2 : b - c + d = 8) 
  (h3 : c - d + a = 4) 
  (h4 : d - a + b = 1) : 
  2*a + 2*b + 2*c + 2*d = 20 := 
sorry

end find_2a_2b_2c_2d_l285_285217


namespace find_g_of_nine_l285_285369

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_nine (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = x) : g 9 = 2 :=
by
  sorry

end find_g_of_nine_l285_285369


namespace smallest_set_handshakes_l285_285850

-- Define the number of people
def num_people : Nat := 36

-- Define a type for people
inductive Person : Type
| a : Fin num_people → Person

-- Define the handshake relationship
def handshake (p1 p2 : Person) : Prop :=
  match p1, p2 with
  | Person.a i, Person.a j => i.val = (j.val + 1) % num_people ∨ j.val = (i.val + 1) % num_people

-- Define the problem statement
theorem smallest_set_handshakes :
  ∃ s : Finset Person, (∀ p : Person, p ∈ s ∨ ∃ q ∈ s, handshake p q) ∧ s.card = 18 :=
sorry

end smallest_set_handshakes_l285_285850


namespace periodic_function_with_period_sqrt2_l285_285927

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Definition of symmetry about x = sqrt(2)/2
def is_symmetric_about_line (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

-- Main theorem to prove
theorem periodic_function_with_period_sqrt2 (f : ℝ → ℝ) :
  is_even_function f → is_symmetric_about_line f (Real.sqrt 2 / 2) → ∃ T, T = Real.sqrt 2 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

end periodic_function_with_period_sqrt2_l285_285927


namespace journey_speed_first_half_l285_285853

theorem journey_speed_first_half (total_distance : ℕ) (total_time : ℕ) (second_half_distance : ℕ) (second_half_speed : ℕ)
  (distance_first_half_eq_half_total : second_half_distance = total_distance / 2)
  (time_for_journey_eq : total_time = 20)
  (journey_distance_eq : total_distance = 240)
  (second_half_speed_eq : second_half_speed = 15) :
  let v := second_half_distance / (total_time - (second_half_distance / second_half_speed))
  v = 10 := 
by
  sorry

end journey_speed_first_half_l285_285853


namespace overall_gain_percent_l285_285523

theorem overall_gain_percent {initial_cost first_repair second_repair third_repair sell_price : ℝ} 
  (h1 : initial_cost = 800) 
  (h2 : first_repair = 150) 
  (h3 : second_repair = 75) 
  (h4 : third_repair = 225) 
  (h5 : sell_price = 1600) :
  (sell_price - (initial_cost + first_repair + second_repair + third_repair)) / 
  (initial_cost + first_repair + second_repair + third_repair) * 100 = 28 := 
by 
  sorry

end overall_gain_percent_l285_285523


namespace minimum_area_of_sap_circle_l285_285958

noncomputable def function_y (x : ℝ) : ℝ := 1 / (|x| - 1)

def y_axis_intersection : ℝ := function_y 0

def symmetric_point : ℝ × ℝ := (0, -y_axis_intersection)

def distance (x : ℝ) : ℝ :=
  real.sqrt (x^2 + (function_y x - y_axis_intersection)^2)

def minimum_distance : ℝ :=
  real.sqrt 3

def radius : ℝ :=
  minimum_distance

theorem minimum_area_of_sap_circle : real.pi * radius^2 = 3 * real.pi :=
by
  sorry

end minimum_area_of_sap_circle_l285_285958


namespace solve_puzzle_l285_285003

theorem solve_puzzle (x1 x2 x3 x4 x5 x6 x7 x8 : ℕ) : 
  (8 + x1 + x2 = 20) →
  (x1 + x2 + x3 = 20) →
  (x2 + x3 + x4 = 20) →
  (x3 + x4 + x5 = 20) →
  (x4 + x5 + 5 = 20) →
  (x5 + 5 + x6 = 20) →
  (5 + x6 + x7 = 20) →
  (x6 + x7 + x8 = 20) →
  (x1 = 7 ∧ x2 = 5 ∧ x3 = 8 ∧ x4 = 7 ∧ x5 = 5 ∧ x6 = 8 ∧ x7 = 7 ∧ x8 = 5) :=
by {
  sorry
}

end solve_puzzle_l285_285003


namespace fraction_of_male_birds_l285_285844

theorem fraction_of_male_birds (T : ℕ) (h_cond1 : T ≠ 0) :
  let robins := (2 / 5) * T
  let bluejays := T - robins
  let male_robins := (2 / 3) * robins
  let male_bluejays := (1 / 3) * bluejays
  (male_robins + male_bluejays) / T = 7 / 15 :=
by 
  sorry

end fraction_of_male_birds_l285_285844


namespace probability_A_inter_B_l285_285067

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 5
def set_B (x : ℝ) : Prop := (x-2)/(3-x) > 0

def A_inter_B (x : ℝ) : Prop := set_A x ∧ set_B x

theorem probability_A_inter_B :
  let length_A := 5 - (-1)
  let length_A_inter_B := 3 - 2 
  length_A > 0 ∧ length_A_inter_B > 0 →
  length_A_inter_B / length_A = 1 / 6 :=
by
  intro h
  sorry

end probability_A_inter_B_l285_285067


namespace evaluate_g_at_2_l285_285344

def g (x : ℝ) : ℝ := x^3 + x^2 - 1

theorem evaluate_g_at_2 : g 2 = 11 := by
  sorry

end evaluate_g_at_2_l285_285344


namespace floor_factorial_expression_l285_285156

-- Mathematical definitions (conditions)
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Mathematical proof problem (statement)
theorem floor_factorial_expression :
  Int.floor ((factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)) = 2006 :=
sorry

end floor_factorial_expression_l285_285156


namespace largest_n_S_n_positive_l285_285316

-- We define the arithmetic sequence a_n.
def arith_seq (a_n : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Definitions for the conditions provided.
def first_term_positive (a_n : ℕ → ℝ) : Prop := 
  a_n 1 > 0

def term_sum_positive (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 + a_n 2017 > 0

def term_product_negative (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 * a_n 2017 < 0

-- Sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a_n 1 + a_n n) / 2

-- Statement we want to prove in Lean 4.
theorem largest_n_S_n_positive (a_n : ℕ → ℝ) 
  (h_seq : arith_seq a_n) 
  (h1 : first_term_positive a_n) 
  (h2 : term_sum_positive a_n) 
  (h3 : term_product_negative a_n) : 
  ∀ n : ℕ, sum_first_n_terms a_n n > 0 → n ≤ 4032 := 
sorry

end largest_n_S_n_positive_l285_285316


namespace percentage_increase_correct_l285_285913

-- Defining initial conditions
variables (B H : ℝ) -- bears per week without assistant and hours per week without assistant

-- Defining the rate of making bears per hour without assistant
def rate_without_assistant := B / H

-- Defining the rate with an assistant (100% increase in output per hour)
def rate_with_assistant := 2 * rate_without_assistant

-- Defining the number of hours worked per week with an assistant (10% fewer hours)
def hours_with_assistant := 0.9 * H

-- Calculating the number of bears made per week with an assistant
def bears_with_assistant := rate_with_assistant * hours_with_assistant

-- Calculating the percentage increase in the number of bears made per week when Jane works with an assistant
def percentage_increase : ℝ := ((bears_with_assistant / B) - 1) * 100

-- The theorem to prove
theorem percentage_increase_correct : percentage_increase B H = 80 :=
  by sorry

end percentage_increase_correct_l285_285913


namespace opposite_of_neg3_is_3_l285_285373

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l285_285373


namespace problem_statement_l285_285176

theorem problem_statement (x y : ℝ) (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 18) :
  17 * x ^ 2 + 24 * x * y + 17 * y ^ 2 = 532 :=
by
  sorry

end problem_statement_l285_285176


namespace road_unrepaired_is_42_percent_statement_is_false_l285_285027

def road_length : ℝ := 1
def phase1_completion : ℝ := 0.40
def phase2_remaining_factor : ℝ := 0.30

def remaining_road (road : ℝ) (phase1 : ℝ) (phase2_factor : ℝ) : ℝ :=
  road - phase1 - (road - phase1) * phase2_factor

theorem road_unrepaired_is_42_percent (road_length : ℝ) (phase1_completion : ℝ) (phase2_remaining_factor : ℝ) :
  remaining_road road_length phase1_completion phase2_remaining_factor = 0.42 :=
by
  sorry

theorem statement_is_false : ¬(remaining_road road_length phase1_completion phase2_remaining_factor = 0.30) :=
by
  sorry

end road_unrepaired_is_42_percent_statement_is_false_l285_285027


namespace emily_total_points_l285_285878

def score_round_1 : ℤ := 16
def score_round_2 : ℤ := 33
def score_round_3 : ℤ := -25
def score_round_4 : ℤ := 46
def score_round_5 : ℤ := 12
def score_round_6 : ℤ := 30 - (2 * score_round_5 / 3)

def total_score : ℤ :=
  score_round_1 + score_round_2 + score_round_3 + score_round_4 + score_round_5 + score_round_6

theorem emily_total_points : total_score = 104 := by
  sorry

end emily_total_points_l285_285878


namespace xy_over_y_plus_x_l285_285490

theorem xy_over_y_plus_x {x y z : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : 1/x + 1/y = 1/z) : z = xy/(y+x) :=
sorry

end xy_over_y_plus_x_l285_285490


namespace max_friends_in_compartment_l285_285912

/-- Definition of compartment and properties -/
def Compartment (P : Type) [Fintype P] (m : ℕ) (h_m : 3 ≤ m) :=
  ∀ (A : P) (S : Finset P), 
    A ∉ S → S.card = m - 1 → 
    ∃! C : P, C ∈ S ∧ ∀ (B : P), B ∈ S → (isFriend B C ∧ isFriend C B)
    where
      isFriend : P → P → Prop
      isFriend_symm : symmetric isFriend
      isFriend_irrefl : irreflexive isFriend

/-- Proving the maximum number of friends a person can have in the compartment(P), given the conditions -/
theorem max_friends_in_compartment (P : Type) [Fintype P] (m : ℕ) (h_m : 3 ≤ m) (cpt : Compartment P m h_m):
  ∀ (A : P), (∃ (k : ℕ), ∀ B : P, B ≠ A → isFriend A B → k <= m) := 
sorry

end max_friends_in_compartment_l285_285912


namespace number_of_integer_solutions_l285_285166

theorem number_of_integer_solutions (h : ∀ n : ℤ, (2020 - n) ^ 2 / (2020 - n ^ 2) ≥ 0) :
  ∃! (m : ℤ), m = 90 := 
sorry

end number_of_integer_solutions_l285_285166


namespace monotonic_increasing_interval_l285_285685

noncomputable def f (a : ℝ) (h : 0 < a ∧ a < 1) (x : ℝ) := a ^ (-x^2 + 3 * x + 2)

theorem monotonic_increasing_interval (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ x1 x2 : ℝ, (3 / 2 < x1 ∧ x1 < x2) → f a h x1 < f a h x2 :=
sorry

end monotonic_increasing_interval_l285_285685


namespace system_of_equations_solution_l285_285824

theorem system_of_equations_solution (x y : ℝ) (h1 : 4 * x + 3 * y = 11) (h2 : 4 * x - 3 * y = 5) :
  x = 2 ∧ y = 1 :=
by {
  sorry
}

end system_of_equations_solution_l285_285824


namespace compute_expression_l285_285999

theorem compute_expression :
  6 * (2 / 3)^4 - 1 / 6 = 55 / 54 :=
by
  sorry

end compute_expression_l285_285999


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l285_285967

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ p2 = p1 + 1 ∧ is_prime p3 ∧ p3 = p2 + 1

def sum_divisible_by_5 (p1 p2 p3 : ℕ) : Prop :=
  (p1 + p2 + p3) % 5 = 0

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (p1 p2 p3 : ℕ), consecutive_primes p1 p2 p3 ∧ sum_divisible_by_5 p1 p2 p3 ∧ p1 + p2 + p3 = 10 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l285_285967


namespace Phil_quarters_l285_285940

theorem Phil_quarters (initial_amount : ℝ)
  (pizza : ℝ) (soda : ℝ) (jeans : ℝ) (book : ℝ) (gum : ℝ) (ticket : ℝ)
  (quarter_value : ℝ) (spent := pizza + soda + jeans + book + gum + ticket)
  (remaining := initial_amount - spent)
  (quarters := remaining / quarter_value) :
  initial_amount = 40 ∧ pizza = 2.75 ∧ soda = 1.50 ∧ jeans = 11.50 ∧
  book = 6.25 ∧ gum = 1.75 ∧ ticket = 8.50 ∧ quarter_value = 0.25 →
  quarters = 31 :=
by
  intros
  sorry

end Phil_quarters_l285_285940


namespace quadratic_eq_unique_k_l285_285474

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l285_285474


namespace rectangle_area_perimeter_max_l285_285856

-- Define the problem conditions
variables {A P : ℝ}

-- Main statement: prove that the maximum value of A / P^2 for a rectangle results in m+n = 17
theorem rectangle_area_perimeter_max (h1 : A = l * w) (h2 : P = 2 * (l + w)) :
  let m := 1
  let n := 16
  m + n = 17 :=
sorry

end rectangle_area_perimeter_max_l285_285856


namespace auditorium_rows_l285_285965

theorem auditorium_rows (x : ℕ) (hx : (320 / x + 4) * (x + 1) = 420) : x = 20 :=
by
  sorry

end auditorium_rows_l285_285965


namespace quadratic_j_value_l285_285540

theorem quadratic_j_value (a b c : ℝ) (h : a * (0 : ℝ)^2 + b * (0 : ℝ) + c = 5 * ((0 : ℝ) - 3)^2 + 15) :
  ∃ m j n, 4 * a * (0 : ℝ)^2 + 4 * b * (0 : ℝ) + 4 * c = m * ((0 : ℝ) - j)^2 + n ∧ j = 3 :=
by
  sorry

end quadratic_j_value_l285_285540


namespace correct_formula_l285_285596

def table : List (ℕ × ℕ) :=
    [(1, 3), (2, 8), (3, 15), (4, 24), (5, 35)]

theorem correct_formula : ∀ x y, (x, y) ∈ table → y = x^2 + 4 * x + 3 :=
by
  intros x y H
  sorry

end correct_formula_l285_285596


namespace range_of_a_l285_285486

def A (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0
def B (x a : ℝ) : Prop := x < a + 1

theorem range_of_a (a : ℝ) : (∃ x : ℝ, A x ∧ B x a) ↔ a > 0 := by
  sorry

end range_of_a_l285_285486


namespace count_distinct_integer_sums_of_special_fractions_l285_285723

theorem count_distinct_integer_sums_of_special_fractions : 
  let special_fractions := {frac : ℚ | ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a + b = 18 ∧ frac = a / b} in
  let special_sum := {x + y | x y : ℚ, x ∈ special_fractions ∧ y ∈ special_fractions} in
  fintype.card {n : ℤ | ∃ x ∈ special_sum, x = n} = 6 :=
by
  sorry

end count_distinct_integer_sums_of_special_fractions_l285_285723


namespace reflect_point_l285_285533

def point_reflect_across_line (m : ℝ) :=
  (6 - m, m + 1)

theorem reflect_point (m : ℝ) :
  point_reflect_across_line m = (6 - m, m + 1) :=
  sorry

end reflect_point_l285_285533


namespace distance_between_trees_l285_285652

theorem distance_between_trees (num_trees : ℕ) (total_length : ℕ) (num_spaces : ℕ) (distance_per_space : ℕ) 
  (h_num_trees : num_trees = 11) (h_total_length : total_length = 180)
  (h_num_spaces : num_spaces = num_trees - 1) (h_distance_per_space : distance_per_space = total_length / num_spaces) :
  distance_per_space = 18 := 
  by 
    sorry

end distance_between_trees_l285_285652


namespace possible_values_of_m_l285_285062

def F1 := (-3, 0)
def F2 := (3, 0)
def possible_vals := [2, -1, 4, -3, 1/2]

noncomputable def is_valid_m (m : ℝ) : Prop :=
  abs (2 * m - 1) < 6 ∧ m ≠ 1/2

theorem possible_values_of_m : {m ∈ possible_vals | is_valid_m m} = {2, -1} := by
  sorry

end possible_values_of_m_l285_285062


namespace find_triangle_side1_l285_285821

def triangle_side1 (Perimeter Side2 Side3 Side1 : ℕ) : Prop :=
  Perimeter = Side1 + Side2 + Side3

theorem find_triangle_side1 :
  ∀ (Perimeter Side2 Side3 Side1 : ℕ), 
    (Perimeter = 160) → (Side2 = 50) → (Side3 = 70) → triangle_side1 Perimeter Side2 Side3 Side1 → Side1 = 40 :=
by
  intros Perimeter Side2 Side3 Side1 h1 h2 h3 h4
  sorry

end find_triangle_side1_l285_285821


namespace problem1_problem2_problem3_problem4_l285_285998

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2023 + (π - 3.14) ^ 0 - ((-1 / 2 : ℚ) ^ (-2 : ℤ)) = -4 := by
  sorry

-- Problem 2
theorem problem2 (x : ℚ) : 
  ((1 / 4 * x^4 + 2 * x^3 - 4 * x^2) / (-(2 * x))^2) = (1 / 16 * x^2 + 1 / 2 * x - 1) := by
  sorry

-- Problem 3
theorem problem3 (x y : ℚ) : 
  (2 * x + y + 1) * (2 * x + y - 1) = 4 * x^2 + 4 * x * y + y^2 - 1 := by
  sorry

-- Problem 4
theorem problem4 (x : ℚ) : 
  (2 * x + 3) * (2 * x - 3) - (2 * x - 1)^2 = 4 * x - 10 := by
  sorry

end problem1_problem2_problem3_problem4_l285_285998


namespace expected_value_fair_8_sided_die_l285_285574

theorem expected_value_fair_8_sided_die :
  (∑ n in Finset.range 8, (n + 1)^3) / 8 = 162 := by
  -- We will provide the proof here (not needed for this task)
  sorry

end expected_value_fair_8_sided_die_l285_285574


namespace oliver_gave_janet_l285_285743

def initial_candy : ℕ := 78
def remaining_candy : ℕ := 68

theorem oliver_gave_janet : initial_candy - remaining_candy = 10 :=
by
  sorry

end oliver_gave_janet_l285_285743


namespace even_number_combinations_l285_285257

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l285_285257


namespace conic_section_is_ellipse_l285_285873

theorem conic_section_is_ellipse :
  (∃ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0) ∧
  ∀ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0 →
    ((x - 2)^2 / (20 / 3) + (y - 2)^2 / 20 = 1) :=
sorry

end conic_section_is_ellipse_l285_285873


namespace find_m_l285_285629

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l285_285629


namespace logarithm_identity_l285_285360

theorem logarithm_identity :
  1 / (Real.log 3 / Real.log 8 + 1) + 
  1 / (Real.log 2 / Real.log 12 + 1) + 
  1 / (Real.log 4 / Real.log 9 + 1) = 3 := 
by
  sorry

end logarithm_identity_l285_285360


namespace largest_adjacent_to_1_number_of_good_cells_l285_285518

def table_width := 51
def table_height := 3
def total_cells := 153

-- Conditions
def condition_1_present (n : ℕ) : Prop := n ∈ Finset.range (total_cells + 1)
def condition_2_bottom_left : Prop := (1 = 1)
def condition_3_adjacent (a b : ℕ) : Prop := 
  (a = b + 1) ∨ 
  (a + 1 = b) ∧ 
  (condition_1_present a) ∧ 
  (condition_1_present b)

-- Part (a): Largest number adjacent to cell containing 1 is 152.
theorem largest_adjacent_to_1 : ∃ b, b = 152 ∧ condition_3_adjacent 1 b :=
by sorry

-- Part (b): Number of good cells that can contain the number 153 is 76.
theorem number_of_good_cells : ∃ count, count = 76 ∧ 
  ∀ (i : ℕ) (j: ℕ), (i, j) ∈ (Finset.range table_height).product (Finset.range table_width) →
  condition_1_present 153 ∧
  (i = table_height - 1 ∨ j = 0 ∨ j = table_width - 1 ∨ j ∈ (Finset.range (table_width - 2)).erase 1) →
  (condition_3_adjacent (i*table_width + j) 153) :=
by sorry

end largest_adjacent_to_1_number_of_good_cells_l285_285518


namespace number_of_BA3_in_sample_l285_285573

-- Definitions for the conditions
def strains_BA1 : Nat := 60
def strains_BA2 : Nat := 20
def strains_BA3 : Nat := 40
def total_sample_size : Nat := 30

def total_strains : Nat := strains_BA1 + strains_BA2 + strains_BA3

-- Theorem statement translating to the equivalent proof problem
theorem number_of_BA3_in_sample :
  total_sample_size * strains_BA3 / total_strains = 10 :=
by
  sorry

end number_of_BA3_in_sample_l285_285573


namespace compute_m_n_sum_l285_285025

theorem compute_m_n_sum :
  let AB := 10
  let BC := 15
  let height := 30
  let volume_ratio := 9
  let smaller_base_AB := AB / 3
  let smaller_base_BC := BC / 3
  let diagonal_AC := Real.sqrt (AB^2 + BC^2)
  let smaller_diagonal_A'C' := Real.sqrt ((smaller_base_AB)^2 + (smaller_base_BC)^2)
  let y_length := 145 / 9   -- derived from geometric considerations
  let YU := 20 + y_length
  let m := 325
  let n := 9
  YU = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 334 :=
  by
  sorry

end compute_m_n_sum_l285_285025


namespace weight_of_person_replaced_l285_285654

theorem weight_of_person_replaced (W : ℝ) (old_avg_weight : ℝ) (new_avg_weight : ℝ)
  (h_avg_increase : new_avg_weight = old_avg_weight + 1.5) (new_person_weight : ℝ) :
  ∃ (person_replaced_weight : ℝ), new_person_weight = 77 ∧ old_avg_weight = W / 8 ∧
  new_avg_weight = (W - person_replaced_weight + 77) / 8 ∧ person_replaced_weight = 65 := by
    sorry

end weight_of_person_replaced_l285_285654


namespace problem_statement_l285_285460

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end problem_statement_l285_285460


namespace smallest_positive_integer_x_for_cube_l285_285009

theorem smallest_positive_integer_x_for_cube (x : ℕ) (h1 : 1512 = 2^3 * 3^3 * 7) (h2 : ∀ n : ℕ, n > 0 → ∃ k : ℕ, 1512 * n = k^3) : x = 49 :=
sorry

end smallest_positive_integer_x_for_cube_l285_285009


namespace initially_calculated_average_l285_285219

theorem initially_calculated_average :
  ∀ (S : ℕ), (S / 10 = 18) →
  ((S - 46 + 26) / 10 = 16) :=
by
  sorry

end initially_calculated_average_l285_285219


namespace inverse_r_l285_285662

def p (x: ℝ) : ℝ := 4 * x + 5
def q (x: ℝ) : ℝ := 3 * x - 4
def r (x: ℝ) : ℝ := p (q x)

theorem inverse_r (x : ℝ) : r⁻¹ x = (x + 11) / 12 :=
sorry

end inverse_r_l285_285662


namespace restore_numbers_possible_l285_285674

theorem restore_numbers_possible (n : ℕ) (h : nat.odd n) : 
  (∀ (A : fin n → ℕ) (S : ℕ) 
    (triangles : fin n → (ℕ × ℕ × ℕ)),
      ∃ (vertices : fin n → ℕ), 
        ∃ (center : ℕ), 
          (forall i, triangles i = (vertices i, vertices (i.succ % n), center))) :=
by
  sorry

end restore_numbers_possible_l285_285674


namespace cars_to_sell_l285_285582

theorem cars_to_sell (clients : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) (total_clients : ℕ) (h1 : selections_per_client = 2) 
  (h2 : selections_per_car = 3) (h3 : total_clients = 24) : (total_clients * selections_per_client / selections_per_car = 16) :=
by
  sorry

end cars_to_sell_l285_285582


namespace opposite_of_neg3_l285_285386

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l285_285386


namespace minimum_fruits_l285_285236

open Nat

theorem minimum_fruits (n : ℕ) :
    (n % 3 = 2) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 6 = 5) →
    n = 59 := by
  sorry

end minimum_fruits_l285_285236


namespace inequality_proof_l285_285978

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y + y * z + z * x = 1) :
  3 - Real.sqrt 3 + (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ (x + y + z)^2 :=
by
  sorry

end inequality_proof_l285_285978


namespace system_inconsistent_l285_285103

-- Define the coefficient matrix and the augmented matrices.
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -2, 3], ![2, 3, -1], ![3, 1, 2]]

def B1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, -2, 3], ![7, 3, -1], ![10, 1, 2]]

-- Calculate the determinants.
noncomputable def delta : ℤ := A.det
noncomputable def delta1 : ℤ := B1.det

-- The main theorem statement: the system is inconsistent if Δ = 0 and Δ1 ≠ 0.
theorem system_inconsistent (h₁ : delta = 0) (h₂ : delta1 ≠ 0) : False :=
sorry

end system_inconsistent_l285_285103


namespace intersection_of_A_and_B_l285_285515

def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : setA ∩ setB = {2} :=
by
  sorry

end intersection_of_A_and_B_l285_285515


namespace perpendicular_vectors_l285_285627

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l285_285627


namespace radio_show_songs_duration_l285_285435

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_songs_duration_l285_285435


namespace mandy_toys_count_l285_285590

theorem mandy_toys_count (M A Am P : ℕ) 
    (h1 : A = 3 * M) 
    (h2 : A = Am - 2) 
    (h3 : A = P / 2) 
    (h4 : M + A + Am + P = 278) : 
    M = 21 := 
by
  sorry

end mandy_toys_count_l285_285590


namespace find_m_l285_285485

theorem find_m (m : ℝ) (A : Set ℝ) (B : Set ℝ) (hA : A = { -1, 2, 2 * m - 1 }) (hB : B = { 2, m^2 }) (hSubset : B ⊆ A) : m = 1 := by
  sorry
 
end find_m_l285_285485


namespace value_of_x0_l285_285642

noncomputable def f (x : ℝ) : ℝ := x^3

theorem value_of_x0 (x0 : ℝ) (h1 : f x0 = x0^3) (h2 : deriv f x0 = 3) :
  x0 = 1 ∨ x0 = -1 :=
by
  sorry

end value_of_x0_l285_285642


namespace sequence_unique_integers_l285_285794

theorem sequence_unique_integers (a : ℕ → ℤ) 
  (H_inf_pos : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n > N) 
  (H_inf_neg : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n < N)
  (H_diff_remainders : ∀ n : ℕ, n > 0 → ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) → (1 ≤ j ∧ j ≤ n) → i ≠ j → (a i % ↑n) ≠ (a j % ↑n)) :
  ∀ m : ℤ, ∃! n : ℕ, a n = m := sorry

end sequence_unique_integers_l285_285794


namespace quadratic_roots_are_correct_l285_285811

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end quadratic_roots_are_correct_l285_285811


namespace min_students_wearing_both_glasses_and_watches_l285_285779

theorem min_students_wearing_both_glasses_and_watches
  (n : ℕ)
  (H_glasses : n * 3 / 5 = 18)
  (H_watches : n * 5 / 6 = 25)
  (H_neither : n * 1 / 10 = 3):
  ∃ (x : ℕ), x = 16 := 
by
  sorry

end min_students_wearing_both_glasses_and_watches_l285_285779


namespace rectangle_length_l285_285959

theorem rectangle_length
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 100) :
  l = 20 :=
sorry

end rectangle_length_l285_285959


namespace find_m_range_l285_285482

-- Definitions for the conditions and the required proof
def condition_alpha (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m + 7
def condition_beta (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

-- Proof problem translated to Lean 4 statement
theorem find_m_range (m : ℝ) :
  (∀ x, condition_beta x → condition_alpha m x) → (-2 ≤ m ∧ m ≤ 0) :=
by sorry

end find_m_range_l285_285482


namespace average_fuel_efficiency_round_trip_l285_285263

noncomputable def average_fuel_efficiency (d1 d2 mpg1 mpg2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let fuel_used := (d1 / mpg1) + (d2 / mpg2)
  total_distance / fuel_used

theorem average_fuel_efficiency_round_trip :
  average_fuel_efficiency 180 180 36 24 = 28.8 :=
by 
  sorry

end average_fuel_efficiency_round_trip_l285_285263


namespace delegates_seating_probability_delegates_seating_sum_mn_l285_285129

noncomputable def delegate_probability: ℚ :=
  let total_arrangements := 12 * 11 * 10 * 9 * 7 * 5
  let unwanted_arrangements := 1260 - 144 + 24
  let valid_arrangements := total_arrangements - unwanted_arrangements
  valid_arrangements / total_arrangements

theorem delegates_seating_probability : 
  delegate_probability = 21 / 22 := 
  sorry

theorem delegates_seating_sum_mn : 
  let m := 21
  let n := 22
  m + n = 43 :=
  by
    simp
    rfl

end delegates_seating_probability_delegates_seating_sum_mn_l285_285129


namespace solve_system_solve_equation_l285_285102

-- 1. System of Equations
theorem solve_system :
  ∀ (x y : ℝ), (x + 2 * y = 9) ∧ (3 * x - 2 * y = 3) → (x = 3) ∧ (y = 3) :=
by sorry

-- 2. Single Equation
theorem solve_equation :
  ∀ (x : ℝ), (2 - x) / (x - 3) + 3 = 2 / (3 - x) → x = 5 / 2 :=
by sorry

end solve_system_solve_equation_l285_285102


namespace find_c_l285_285900

variables {x b c : ℝ}

theorem find_c (H : (x + 3) * (x + b) = x^2 + c * x + 12) (hb : b = 4) : c = 7 :=
by sorry

end find_c_l285_285900


namespace daniel_wins_probability_l285_285455

theorem daniel_wins_probability :
  let pd := 0.6
  let ps := 0.4
  ∃ (p : ℚ), p = 9 / 13 :=
by
  sorry

end daniel_wins_probability_l285_285455


namespace eval_polynomial_at_2_l285_285651

theorem eval_polynomial_at_2 : 
  ∃ a b c d : ℝ, (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 18) :=
by
  sorry

end eval_polynomial_at_2_l285_285651


namespace opposite_of_neg3_is_3_l285_285372

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l285_285372


namespace solve_for_x_l285_285527

theorem solve_for_x (x : ℤ) (h : 20 * 14 + x = 20 + 14 * x) : x = 20 := 
by 
  sorry

end solve_for_x_l285_285527


namespace yaya_bike_walk_l285_285974

theorem yaya_bike_walk (x y : ℝ) : 
  (x + y = 1.5 ∧ 15 * x + 5 * y = 20) ↔ (x + y = 1.5 ∧ 15 * x + 5 * y = 20) :=
by 
  sorry

end yaya_bike_walk_l285_285974


namespace neg_p_iff_neg_q_l285_285175

theorem neg_p_iff_neg_q (a : ℝ) : (¬ (a < 0)) ↔ (¬ (a^2 > a)) :=
by 
    sorry

end neg_p_iff_neg_q_l285_285175


namespace value_of_expression_l285_285405

theorem value_of_expression :
  (3150 - 3030)^2 / 144 = 100 :=
by {
  -- This imported module allows us to use basic mathematical functions and properties
  sorry -- We use sorry to skip the actual proof
}

end value_of_expression_l285_285405


namespace recurring_decimal_to_fraction_l285_285396

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l285_285396


namespace first_discount_percentage_l285_285990

theorem first_discount_percentage (normal_price sale_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) :
  normal_price = 149.99999999999997 →
  sale_price = 108 →
  second_discount = 0.20 →
  (1 - second_discount) * (1 - first_discount) * normal_price = sale_price →
  first_discount = 0.10 :=
by
  intros
  sorry

end first_discount_percentage_l285_285990


namespace clock_angle_5_30_l285_285971

theorem clock_angle_5_30 (h_degree : ℕ → ℝ) (m_degree : ℕ → ℝ) (hours_pos : ℕ → ℝ) :
  (h_degree 12 = 360) →
  (m_degree 60 = 360) →
  (hours_pos 5 + h_degree 1 - (m_degree 30 / 2) = 165) →
  (m_degree 30 = 180) →
  ∃ θ : ℝ, θ = abs (m_degree 30 - (hours_pos 5 + h_degree 1 - (m_degree 30 / 2))) ∧ θ = 15 :=
by
  sorry

end clock_angle_5_30_l285_285971


namespace difference_of_digits_l285_285220

theorem difference_of_digits (X Y : ℕ) (h1 : 10 * X + Y < 100) 
  (h2 : 72 = (10 * X + Y) - (10 * Y + X)) : (X - Y) = 8 :=
sorry

end difference_of_digits_l285_285220


namespace trig_expression_value_l285_285617

open Real

theorem trig_expression_value (θ : ℝ)
  (h1 : cos (π - θ) > 0)
  (h2 : cos (π / 2 + θ) * (1 - 2 * cos (θ / 2) ^ 2) < 0) :
  (sin θ / |sin θ|) + (|cos θ| / cos θ) + (tan θ / |tan θ|) = -1 :=
by
  sorry

end trig_expression_value_l285_285617


namespace average_production_n_days_l285_285054

theorem average_production_n_days (n : ℕ) (P : ℕ) 
  (hP : P = 80 * n)
  (h_new_avg : (P + 220) / (n + 1) = 95) : 
  n = 8 := 
by
  sorry -- Proof of the theorem

end average_production_n_days_l285_285054


namespace calculate_expression_l285_285721

theorem calculate_expression : 200 * 39.96 * 3.996 * 500 = (3996)^2 :=
by
  sorry

end calculate_expression_l285_285721


namespace eggs_left_over_l285_285728

def david_eggs : ℕ := 44
def elizabeth_eggs : ℕ := 52
def fatima_eggs : ℕ := 23
def carton_size : ℕ := 12

theorem eggs_left_over : 
  (david_eggs + elizabeth_eggs + fatima_eggs) % carton_size = 11 :=
by sorry

end eggs_left_over_l285_285728


namespace probability_15th_roll_last_is_approximately_l285_285905

noncomputable def probability_15th_roll_last : ℝ :=
  (7 / 8) ^ 13 * (1 / 8)

theorem probability_15th_roll_last_is_approximately :
  abs (probability_15th_roll_last - 0.022) < 0.001 :=
by sorry

end probability_15th_roll_last_is_approximately_l285_285905


namespace distance_from_dormitory_to_city_l285_285787

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h : (1/5) * D + (2/3) * D + 4 = D) : 
  D = 30 :=
sorry

end distance_from_dormitory_to_city_l285_285787


namespace sin_360_eq_0_l285_285155

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end sin_360_eq_0_l285_285155


namespace larry_expression_correct_l285_285670

theorem larry_expression_correct (a b c d : ℤ) (e : ℤ) :
  (a = 1) → (b = 2) → (c = 3) → (d = 4) →
  (a - b - c - d + e = -2 - e) → (e = 3) :=
by
  intros ha hb hc hd heq
  rw [ha, hb, hc, hd] at heq
  linarith

end larry_expression_correct_l285_285670


namespace solution_statement_l285_285711

-- Define the set of courses
inductive Course
| Physics | Chemistry | Literature | History | Philosophy | Psychology

open Course

-- Define the condition that a valid program must include Physics and at least one of Chemistry or Literature
def valid_program (program : Finset Course) : Prop :=
  Course.Physics ∈ program ∧
  (Course.Chemistry ∈ program ∨ Course.Literature ∈ program)

-- Define the problem statement
def problem_statement : Prop :=
  ∃ programs : Finset (Finset Course),
    programs.card = 9 ∧ ∀ program ∈ programs, program.card = 5 ∧ valid_program program

theorem solution_statement : problem_statement := sorry

end solution_statement_l285_285711


namespace total_difference_is_18_l285_285079

-- Define variables for Mike, Joe, and Anna's bills
variables (m j a : ℝ)

-- Define the conditions given in the problem
def MikeTipped := (0.15 * m = 3)
def JoeTipped := (0.25 * j = 3)
def AnnaTipped := (0.10 * a = 3)

-- Prove the total amount of money that was different between the highest and lowest bill is 18
theorem total_difference_is_18 (MikeTipped : 0.15 * m = 3) (JoeTipped : 0.25 * j = 3) (AnnaTipped : 0.10 * a = 3) :
  |a - j| = 18 := 
sorry

end total_difference_is_18_l285_285079


namespace new_person_weight_l285_285531

noncomputable def weight_of_new_person (W : ℝ) : ℝ :=
  W + 61 - 25

theorem new_person_weight {W : ℝ} : 
  ((W + 61 - 25) / 12 = W / 12 + 3) → 
  weight_of_new_person W = 61 :=
by
  intro h
  sorry

end new_person_weight_l285_285531


namespace fish_weight_l285_285741

theorem fish_weight (θ H T : ℝ) (h1 : θ = 4) (h2 : H = θ + 0.5 * T) (h3 : T = H + θ) : H + T + θ = 32 :=
by
  sorry

end fish_weight_l285_285741


namespace exists_subset_sum_2n_l285_285343

theorem exists_subset_sum_2n (n : ℕ) (h : n > 3) (s : Finset ℕ)
  (hs : ∀ x ∈ s, x < 2 * n) (hs_card : s.card = 2 * n)
  (hs_sum : s.sum id = 4 * n) :
  ∃ t ⊆ s, t.sum id = 2 * n :=
by sorry

end exists_subset_sum_2n_l285_285343


namespace larry_substitution_l285_285669

theorem larry_substitution (a b c d e : ℤ)
  (ha : a = 1)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 4)
  (h_ignored : a - b - c - d + e = a - (b - (c - (d + e)))) :
  e = 3 :=
by
  sorry

end larry_substitution_l285_285669


namespace hannah_trip_time_ratio_l285_285070

theorem hannah_trip_time_ratio 
  (u : ℝ) -- Speed on the first trip in miles per hour.
  (u_pos : u > 0) -- Speed should be positive.
  (t1 t2 : ℝ) -- Time taken for the first and second trip respectively.
  (h_t1 : t1 = 30 / u) -- Time for the first trip.
  (h_t2 : t2 = 150 / (4 * u)) -- Time for the second trip.
  : t2 / t1 = 1.25 := by
  sorry

end hannah_trip_time_ratio_l285_285070


namespace find_cos_A_l285_285646

noncomputable def cos_A_of_third_quadrant : Real :=
-3 / 5

theorem find_cos_A (A : Real) (h1 : A ∈ Set.Icc (π) (3 * π / 2)) 
  (h2 : Real.sin A = 4 / 5) : Real.cos A = -3 / 5 := 
sorry

end find_cos_A_l285_285646


namespace domain_log_sin_sqrt_l285_285292

theorem domain_log_sin_sqrt (x : ℝ) : 
  (2 < x ∧ x < (5 * Real.pi) / 3) ↔ 
  (∃ k : ℤ, (Real.pi / 3) + (4 * k * Real.pi) < x ∧ x < (5 * Real.pi / 3) + (4 * k * Real.pi) ∧ 2 < x) :=
by
  sorry

end domain_log_sin_sqrt_l285_285292


namespace contrapositive_example_l285_285570

theorem contrapositive_example (x : ℝ) :
  (¬ (x = 3 ∧ x = 4)) → (x^2 - 7 * x + 12 ≠ 0) →
  (x^2 - 7 * x + 12 = 0) → (x = 3 ∨ x = 4) :=
by
  intros h h1 h2
  sorry  -- proof is not required

end contrapositive_example_l285_285570


namespace weight_of_2019_is_correct_l285_285638

-- Declare the conditions as definitions to be used in Lean 4
def stick_weight : Real := 0.5
def digit_to_sticks (n : Nat) : Nat :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- other digits aren't considered in this problem

-- Calculate the total weight of the number 2019
def weight_of_2019 : Real :=
  (digit_to_sticks 2 + digit_to_sticks 0 + digit_to_sticks 1 + digit_to_sticks 9) * stick_weight

-- Statement to prove the weight of the number 2019
theorem weight_of_2019_is_correct : weight_of_2019 = 9.5 := by
  sorry

end weight_of_2019_is_correct_l285_285638


namespace johnson_potatoes_left_l285_285505

theorem johnson_potatoes_left :
  ∀ (initial gina tom anne remaining : Nat),
  initial = 300 →
  gina = 69 →
  tom = 2 * gina →
  anne = tom / 3 →
  remaining = initial - (gina + tom + anne) →
  remaining = 47 := by
sorry

end johnson_potatoes_left_l285_285505


namespace max_value_of_expression_l285_285199

theorem max_value_of_expression (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 3) :
  (xy / (x + y + 1) + xz / (x + z + 1) + yz / (y + z + 1)) ≤ 1 :=
sorry

end max_value_of_expression_l285_285199


namespace red_paint_cans_needed_l285_285354

-- Definitions for the problem
def ratio_red_white : ℚ := 3 / 2
def total_cans : ℕ := 30

-- Theorem statement to prove the number of cans of red paint
theorem red_paint_cans_needed : total_cans * (3 / 5) = 18 := by 
  sorry

end red_paint_cans_needed_l285_285354


namespace function_odd_and_decreasing_l285_285308

noncomputable def f (a x : ℝ) : ℝ := (1 / a) ^ x - a ^ x

theorem function_odd_and_decreasing (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -f a x) ∧ (∀ x y, x < y → f a x > f a y) :=
by
  sorry

end function_odd_and_decreasing_l285_285308


namespace simplify_complex_expr_l285_285214

theorem simplify_complex_expr : ∀ i : ℂ, i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 - i) = 14 :=
by 
  intro i 
  intro h
  sorry

end simplify_complex_expr_l285_285214


namespace find_d_minus_c_l285_285225

noncomputable def point_transformed (c d : ℝ) : Prop :=
  let Q := (c, d)
  let R := (2 * 2 - c, 2 * 3 - d)  -- Rotating Q by 180º about (2, 3)
  let S := (d, c)                -- Reflecting Q about the line y = x
  (S.1, S.2) = (2, -1)           -- Result is (2, -1)

theorem find_d_minus_c (c d : ℝ) (h : point_transformed c d) : d - c = -1 :=
by {
  sorry
}

end find_d_minus_c_l285_285225


namespace quadratic_roots_are_correct_l285_285812

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end quadratic_roots_are_correct_l285_285812


namespace windows_ways_l285_285144

theorem windows_ways (n : ℕ) (h : n = 8) : (n * (n - 1)) = 56 :=
by
  sorry

end windows_ways_l285_285144


namespace find_a_circle_line_intersection_l285_285306

theorem find_a_circle_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0)
  (h2 : ∀ x y : ℝ, x + 2 * y + 1 = 0) :
  a = 3 := 
sorry

end find_a_circle_line_intersection_l285_285306


namespace simple_interest_l285_285150

/-- Given:
    - Principal (P) = Rs. 80325
    - Rate (R) = 1% per annum
    - Time (T) = 5 years
    Prove that the total simple interest earned (SI) is Rs. 4016.25.
-/
theorem simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
  (hP : P = 80325)
  (hR : R = 1)
  (hT : T = 5)
  (hSI : SI = P * R * T / 100) :
  SI = 4016.25 :=
by
  sorry

end simple_interest_l285_285150


namespace parabola_vertex_and_point_l285_285706

/-- The vertex form of the parabola is at (7, -6) and passes through the point (1,0).
    Verify that the equation parameters a, b, c satisfy a + b + c = -43 / 6. -/
theorem parabola_vertex_and_point (a b c : ℚ)
  (h_eq : ∀ y, (a * y^2 + b * y + c) = a * (y + 6)^2 + 7)
  (h_vertex : ∃ x y, x = a * y^2 + b * y + c ∧ y = -6 ∧ x = 7)
  (h_point : ∃ x y, x = a * y^2 + b * y + c ∧ x = 1 ∧ y = 0) :
  a + b + c = -43 / 6 :=
by
  sorry

end parabola_vertex_and_point_l285_285706


namespace solve_inequality_l285_285735

theorem solve_inequality (x : ℝ) : 3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end solve_inequality_l285_285735


namespace find_m_of_perpendicular_vectors_l285_285635

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l285_285635


namespace molecular_weight_of_N2O5_is_correct_l285_285042

-- Definitions for atomic weights
def atomic_weight_N : ℚ := 14.01
def atomic_weight_O : ℚ := 16.00

-- Define the molecular weight calculation for N2O5
def molecular_weight_N2O5 : ℚ := (2 * atomic_weight_N) + (5 * atomic_weight_O)

-- The theorem to prove
theorem molecular_weight_of_N2O5_is_correct : molecular_weight_N2O5 = 108.02 := by
  -- Proof here
  sorry

end molecular_weight_of_N2O5_is_correct_l285_285042


namespace total_snowfall_yardley_l285_285495

theorem total_snowfall_yardley (a b c d : ℝ) (ha : a = 0.12) (hb : b = 0.24) (hc : c = 0.5) (hd : d = 0.36) :
  a + b + c + d = 1.22 :=
by
  sorry

end total_snowfall_yardley_l285_285495


namespace average_chore_time_l285_285081

theorem average_chore_time 
  (times : List ℕ := [4, 3, 2, 1, 0])
  (counts : List ℕ := [2, 4, 2, 1, 1]) 
  (total_students : ℕ := 10)
  (total_time : ℕ := List.sum (List.zipWith (λ t c => t * c) times counts)) :
  (total_time : ℚ) / total_students = 2.5 := by
  sorry

end average_chore_time_l285_285081


namespace trajectory_proof_l285_285499

noncomputable def trajectory_eqn (x y : ℝ) : Prop :=
  (y + Real.sqrt 2) * (y - Real.sqrt 2) / (x * x) = -2

theorem trajectory_proof :
  ∀ (x y : ℝ), x ≠ 0 → trajectory_eqn x y → (y*y / 2 + x*x = 1) :=
by
  intros x y hx htrajectory
  sorry

end trajectory_proof_l285_285499


namespace problem_l285_285370

theorem problem (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : (14 * y - 2)^2 = 258 := by
  sorry

end problem_l285_285370


namespace shelby_gold_stars_today_l285_285213

-- Define the number of gold stars Shelby earned yesterday
def gold_stars_yesterday := 4

-- Define the total number of gold stars Shelby earned
def total_gold_stars := 7

-- Define the number of gold stars Shelby earned today
def gold_stars_today := total_gold_stars - gold_stars_yesterday

-- The theorem to prove
theorem shelby_gold_stars_today : gold_stars_today = 3 :=
by 
  -- The proof will go here.
  sorry

end shelby_gold_stars_today_l285_285213


namespace time_to_pass_trolley_l285_285249

/--
Conditions:
- Length of the train = 110 m
- Speed of the train = 60 km/hr
- Speed of the trolley = 12 km/hr

Prove that the time it takes for the train to pass the trolley completely is 5.5 seconds.
-/
theorem time_to_pass_trolley :
  ∀ (train_length : ℝ) (train_speed_kmh : ℝ) (trolley_speed_kmh : ℝ),
    train_length = 110 →
    train_speed_kmh = 60 →
    trolley_speed_kmh = 12 →
  train_length / ((train_speed_kmh + trolley_speed_kmh) * (1000 / 3600)) = 5.5 :=
by
  intros
  sorry

end time_to_pass_trolley_l285_285249


namespace fraction_never_simplifiable_l285_285942

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_never_simplifiable_l285_285942


namespace nails_sum_is_correct_l285_285275

-- Define the fractions for sizes 2d, 3d, 5d, and 8d
def fraction_2d : ℚ := 1 / 6
def fraction_3d : ℚ := 2 / 15
def fraction_5d : ℚ := 1 / 10
def fraction_8d : ℚ := 1 / 8

-- Define the expected answer
def expected_fraction : ℚ := 21 / 40

-- The theorem to prove
theorem nails_sum_is_correct : fraction_2d + fraction_3d + fraction_5d + fraction_8d = expected_fraction :=
by
  -- The proof is not required as per the instructions
  sorry

end nails_sum_is_correct_l285_285275


namespace perpendicular_vectors_implies_m_value_l285_285632

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l285_285632


namespace white_tiles_count_l285_285340

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l285_285340


namespace journey_distance_last_day_l285_285784

theorem journey_distance_last_day (S₆ : ℕ) (q : ℝ) (n : ℕ) (a₁ : ℝ) : 
  S₆ = 378 ∧ q = 1 / 2 ∧ n = 6 ∧ S₆ = a₁ * (1 - q^n) / (1 - q)
  → a₁ * q^(n - 1) = 6 :=
by
  intro h
  sorry

end journey_distance_last_day_l285_285784


namespace value_of_expression_l285_285557

theorem value_of_expression (a : ℚ) (h : a = 1/3) : (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by
  sorry

end value_of_expression_l285_285557


namespace investments_are_beneficial_l285_285752

-- Definitions of examples and their benefits as given in the conditions
def investment_in_education : Prop :=
  ∃ (benefit : String), 
    benefit = "enhances employability and earning potential"

def investment_in_physical_health : Prop :=
  ∃ (benefit : String), 
    benefit = "reduces future healthcare costs and enhances overall well-being"

def investment_in_reading_books : Prop :=
  ∃ (benefit : String), 
    benefit = "cultivates intellectual growth and contributes to personal and professional success"

-- The theorem combining the three investments and their benefits
theorem investments_are_beneficial :
  investment_in_education ∧ investment_in_physical_health ∧ investment_in_reading_books :=
by
  split;
  { 
    existsi "enhances employability and earning potential", sorry <|>
    existsi "reduces future healthcare costs and enhances overall well-being", sorry <|>
    existsi "cultivates intellectual growth and contributes to personal and professional success", sorry
  }

end investments_are_beneficial_l285_285752


namespace find_value_of_fraction_l285_285800

variable {x y : ℝ}

theorem find_value_of_fraction (h1 : x > 0) (h2 : y > x) (h3 : y > 0) (h4 : x / y + y / x = 3) : 
  (x + y) / (y - x) = Real.sqrt 5 := 
by sorry

end find_value_of_fraction_l285_285800


namespace n_minus_m_l285_285925

variable (m n : ℕ)

def is_congruent_to_5_mod_13 (x : ℕ) : Prop := x % 13 = 5
def is_smallest_three_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 100 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 100 → x ≤ y

def is_smallest_four_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 1000 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 1000 → x ≤ y

theorem n_minus_m
  (h₁ : is_smallest_three_digit_integer_congruent_to_5_mod_13 m)
  (h₂ : is_smallest_four_digit_integer_congruent_to_5_mod_13 n) :
  n - m = 897 := sorry

end n_minus_m_l285_285925


namespace original_sum_of_money_l285_285270

theorem original_sum_of_money (P R : ℝ) 
  (h1 : 720 = P + (P * R * 2) / 100) 
  (h2 : 1020 = P + (P * R * 7) / 100) : 
  P = 600 := 
by sorry

end original_sum_of_money_l285_285270


namespace inequality_holds_if_b_greater_than_2_l285_285643

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_b_greater_than_2  :
  (b > 0) → (∃ x, |x-5| + |x-7| < b) ↔ (b > 2) := sorry

end inequality_holds_if_b_greater_than_2_l285_285643


namespace functional_equation_solution_l285_285459

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l285_285459


namespace ellipse_C_properties_l285_285760

open Real

noncomputable def ellipse_eq (b : ℝ) : Prop :=
  (∀ (x y : ℝ), (x = 1 ∧ y = sqrt 3 / 2) → (x^2 / 4 + y^2 / b^2 = 1))

theorem ellipse_C_properties : 
  (∀ (C : ℝ → ℝ → Prop), 
    (C 0 0) ∧ 
    (∀ x y, C x y → (x = 0 ↔ y = 0)) ∧ 
    (∀ x, C x 0) ∧ 
    (∃ x y, C x y ∧ x = 1 ∧ y = sqrt 3 / 2) →
    (∃ b, b > 0 ∧ b^2 = 1 ∧ ellipse_eq b)) ∧
  (∀ P A B : ℝ × ℝ, 
    (P.1 = P.1 ∧ P.1 ≠ 0 ∧ P.2 = 0 ∧ -2 ≤ P.1 ∧ P.1 ≤ 2) →
    (A.2 = 1/2 * (A.1 - P.1) ∧ B.2 = 1/2 * (B.1 - P.1)) →
    ((P.1 - A.1)^2 + A.2^2 + (P.1 - B.1)^2 + B.2^2 = 5)) :=
by sorry

end ellipse_C_properties_l285_285760


namespace probability_log3_N_integer_l285_285431
noncomputable def probability_log3_integer : ℚ :=
  let count := 2
  let total := 900
  count / total

theorem probability_log3_N_integer :
  probability_log3_integer = 1 / 450 :=
sorry

end probability_log3_N_integer_l285_285431


namespace largest_n_for_two_digit_quotient_l285_285687

-- Lean statement for the given problem.
theorem largest_n_for_two_digit_quotient (n : ℕ) (h₀ : 0 ≤ n) (h₃ : n ≤ 9) :
  (10 ≤ (n * 100 + 5) / 5 ∧ (n * 100 + 5) / 5 < 100) ↔ n = 4 :=
by sorry

end largest_n_for_two_digit_quotient_l285_285687


namespace baron_not_boasting_l285_285866

-- Define a function to verify if a given list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- Define a list that represents the sequence given in the solution
def sequence_19 : List ℕ :=
  [9, 18, 7, 16, 5, 14, 3, 12, 1, 10, 11, 2, 13, 4, 15, 6, 17, 8, 19]

-- Prove that the sequence forms a palindrome
theorem baron_not_boasting : is_palindrome sequence_19 :=
by {
  -- Insert actual proof steps here
  sorry
}

end baron_not_boasting_l285_285866


namespace cubic_meter_to_cubic_centimeters_l285_285488

theorem cubic_meter_to_cubic_centimeters : 
  (1 : ℝ)^3 = (100 : ℝ)^3 * (1 : ℝ)^0 := 
by 
  sorry

end cubic_meter_to_cubic_centimeters_l285_285488


namespace find_x_l285_285294

noncomputable def x : ℝ := 80 / 9

theorem find_x
  (hx_pos : 0 < x)
  (hx_condition : x * (⌊x⌋₊ : ℝ) = 80) :
  x = 80 / 9 :=
by
  sorry

end find_x_l285_285294


namespace difference_of_cubes_not_divisible_by_19_l285_285520

theorem difference_of_cubes_not_divisible_by_19 (a b : ℤ) : 
  ¬ (19 ∣ ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3)) := by
  sorry

end difference_of_cubes_not_divisible_by_19_l285_285520


namespace lilly_can_buy_flowers_l285_285350

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end lilly_can_buy_flowers_l285_285350


namespace speed_conversion_l285_285041

theorem speed_conversion (s : ℚ) (h : s = 13 / 48) : 
  ((13 / 48) * 3.6 = 0.975) :=
by
  sorry

end speed_conversion_l285_285041


namespace star_3_5_l285_285055

def star (a b : ℕ) : ℕ := a^2 + 3 * a * b + b^2

theorem star_3_5 : star 3 5 = 79 := 
by
  sorry

end star_3_5_l285_285055


namespace mary_peter_lucy_chestnuts_l285_285933

noncomputable def mary_picked : ℕ := 12
noncomputable def peter_picked : ℕ := mary_picked / 2
noncomputable def lucy_picked : ℕ := peter_picked + 2
noncomputable def total_picked : ℕ := mary_picked + peter_picked + lucy_picked

theorem mary_peter_lucy_chestnuts : total_picked = 26 := by
  sorry

end mary_peter_lucy_chestnuts_l285_285933


namespace congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l285_285826

namespace GeometricPropositions

-- Definitions for congruence in triangles and quadrilaterals:
def congruent_triangles (Δ1 Δ2 : Type) : Prop := sorry
def corresponding_sides_equal (Δ1 Δ2 : Type) : Prop := sorry

def four_equal_sides (Q : Type) : Prop := sorry
def is_square (Q : Type) : Prop := sorry

-- Propositions and their logical forms for triangles
theorem congruent_triangles_implies_corresponding_sides_equal (Δ1 Δ2 : Type) : congruent_triangles Δ1 Δ2 → corresponding_sides_equal Δ1 Δ2 := sorry

theorem corresponding_sides_equal_implies_congruent_triangles (Δ1 Δ2 : Type) : corresponding_sides_equal Δ1 Δ2 → congruent_triangles Δ1 Δ2 := sorry

theorem not_congruent_triangles_implies_not_corresponding_sides_equal (Δ1 Δ2 : Type) : ¬ congruent_triangles Δ1 Δ2 → ¬ corresponding_sides_equal Δ1 Δ2 := sorry

theorem not_corresponding_sides_equal_implies_not_congruent_triangles (Δ1 Δ2 : Type) : ¬ corresponding_sides_equal Δ1 Δ2 → ¬ congruent_triangles Δ1 Δ2 := sorry

-- Propositions and their logical forms for quadrilaterals
theorem four_equal_sides_implies_is_square (Q : Type) : four_equal_sides Q → is_square Q := sorry

theorem is_square_implies_four_equal_sides (Q : Type) : is_square Q → four_equal_sides Q := sorry

theorem not_four_equal_sides_implies_not_is_square (Q : Type) : ¬ four_equal_sides Q → ¬ is_square Q := sorry

theorem not_is_square_implies_not_four_equal_sides (Q : Type) : ¬ is_square Q → ¬ four_equal_sides Q := sorry

end GeometricPropositions

end congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l285_285826


namespace total_earnings_l285_285137

theorem total_earnings (d_a : ℕ) (h : 57 * d_a + 684 + 380 = 1406) : d_a = 6 :=
by {
  -- The proof will involve algebraic manipulations similar to the solution steps
  sorry
}

end total_earnings_l285_285137


namespace kolya_win_l285_285919

theorem kolya_win : ∀ stones : ℕ, stones = 100 → (∃ strategy : (ℕ → ℕ × ℕ), ∀ opponent_strategy : (ℕ → ℕ × ℕ), true → true) :=
by
  sorry

end kolya_win_l285_285919


namespace find_x_plus_y_l285_285636

theorem find_x_plus_y
  (x y : ℤ)
  (hx : |x| = 2)
  (hy : |y| = 3)
  (hxy : x > y) : x + y = -1 := 
sorry

end find_x_plus_y_l285_285636


namespace range_of_m_for_two_solutions_l285_285775

theorem range_of_m_for_two_solutions (x m : ℝ) (h₁ : x > 1) :
  2 * log x / log 2 - log (x - 1) / log 2 = m → (2:ℝ) < m := 
sorry

end range_of_m_for_two_solutions_l285_285775


namespace radio_show_play_song_duration_l285_285433

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end radio_show_play_song_duration_l285_285433


namespace fraction_flower_beds_l285_285031

theorem fraction_flower_beds (length1 length2 height triangle_area yard_area : ℝ) (h1 : length1 = 18) (h2 : length2 = 30) (h3 : height = 10) (h4 : triangle_area = 2 * (1 / 2 * (6 ^ 2))) (h5 : yard_area = ((length1 + length2) / 2) * height) : 
  (triangle_area / yard_area) = 3 / 20 :=
by 
  sorry

end fraction_flower_beds_l285_285031


namespace smallest_sum_of_squares_l285_285368

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 231) :
  x^2 + y^2 ≥ 281 :=
sorry

end smallest_sum_of_squares_l285_285368


namespace average_marks_correct_l285_285100

/-- Define the marks scored by Shekar in different subjects -/
def marks_math : ℕ := 76
def marks_science : ℕ := 65
def marks_social_studies : ℕ := 82
def marks_english : ℕ := 67
def marks_biology : ℕ := 55

/-- Define the total marks scored by Shekar -/
def total_marks : ℕ := marks_math + marks_science + marks_social_studies + marks_english + marks_biology

/-- Define the number of subjects -/
def num_subjects : ℕ := 5

/-- Define the average marks scored by Shekar -/
def average_marks : ℕ := total_marks / num_subjects

theorem average_marks_correct : average_marks = 69 := by
  -- We need to show that the average marks is 69
  sorry

end average_marks_correct_l285_285100


namespace geometric_sum_2015_2016_l285_285889

theorem geometric_sum_2015_2016 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 2)
  (h_a2_a5 : a 2 + a 5 = 0)
  (h_Sn : ∀ n, S n = (1 - (-1)^n)) :
  S 2015 + S 2016 = 2 :=
by sorry

end geometric_sum_2015_2016_l285_285889


namespace emily_disproved_jacob_by_turnover_5_and_7_l285_285951

def is_vowel (c : Char) : Prop :=
  c = 'A'

def is_consonant (c : Char) : Prop :=
  ¬ is_vowel c

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def card_A_is_vowel : Prop := is_vowel 'A'
def card_1_is_odd : Prop := ¬ is_even 1 ∧ ¬ is_prime 1
def card_8_is_even : Prop := is_even 8 ∧ ¬ is_prime 8
def card_R_is_consonant : Prop := is_consonant 'R'
def card_S_is_consonant : Prop := is_consonant 'S'
def card_5_conditions : Prop := ¬ is_even 5 ∧ is_prime 5
def card_7_conditions : Prop := ¬ is_even 7 ∧ is_prime 7

theorem emily_disproved_jacob_by_turnover_5_and_7 :
  card_5_conditions ∧ card_7_conditions →
  (∃ (c : Char), (is_prime 5 ∧ is_consonant c)) ∨
  (∃ (c : Char), (is_prime 7 ∧ is_consonant c)) :=
by sorry

end emily_disproved_jacob_by_turnover_5_and_7_l285_285951


namespace probability_of_triangle_with_nonagon_side_l285_285749

-- Definitions based on the given conditions
def num_vertices : ℕ := 9

def total_triangles : ℕ := Nat.choose num_vertices 3

def favorable_outcomes : ℕ :=
  let one_side_is_side_of_nonagon := num_vertices * 5
  let two_sides_are_sides_of_nonagon := num_vertices
  one_side_is_side_of_nonagon + two_sides_are_sides_of_nonagon

def probability : ℚ := favorable_outcomes / total_triangles

-- Lean 4 statement to prove the equivalence of the probability calculation
theorem probability_of_triangle_with_nonagon_side :
  probability = 9 / 14 :=
by
  sorry

end probability_of_triangle_with_nonagon_side_l285_285749


namespace y_intercept_of_parallel_line_l285_285092

theorem y_intercept_of_parallel_line (m x1 y1 : ℝ) (h_slope : m = -3) (h_point : (x1, y1) = (3, -1))
  (b : ℝ) (h_line_parallel : ∀ x, b = y1 + m * (x - x1)) :
  b = 8 :=
by
  sorry

end y_intercept_of_parallel_line_l285_285092


namespace flowers_in_each_basket_l285_285282

theorem flowers_in_each_basket
  (plants_per_daughter : ℕ)
  (num_daughters : ℕ)
  (grown_flowers : ℕ)
  (died_flowers : ℕ)
  (num_baskets : ℕ)
  (h1 : plants_per_daughter = 5)
  (h2 : num_daughters = 2)
  (h3 : grown_flowers = 20)
  (h4 : died_flowers = 10)
  (h5 : num_baskets = 5) :
  (plants_per_daughter * num_daughters + grown_flowers - died_flowers) / num_baskets = 4 :=
by
  sorry

end flowers_in_each_basket_l285_285282


namespace opposite_of_neg_three_l285_285387

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l285_285387


namespace age_ratio_l285_285109

variables (A B : ℕ)
def present_age_of_A : ℕ := 15
def future_ratio (A B : ℕ) : Prop := (A + 6) / (B + 6) = 7 / 5

theorem age_ratio (A_eq : A = present_age_of_A) (future_ratio_cond : future_ratio A B) : A / B = 5 / 3 :=
sorry

end age_ratio_l285_285109


namespace four_spheres_max_intersections_l285_285748

noncomputable def max_intersection_points (n : Nat) : Nat :=
  if h : n > 0 then n * 2 else 0

theorem four_spheres_max_intersections : max_intersection_points 4 = 8 := by
  sorry

end four_spheres_max_intersections_l285_285748


namespace pow_div_pow_eq_l285_285006

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end pow_div_pow_eq_l285_285006


namespace maria_cookies_left_l285_285346

theorem maria_cookies_left
    (total_cookies : ℕ) -- Maria has 60 cookies
    (friend_share : ℕ) -- 20% of the initial cookies goes to the friend
    (family_share : ℕ) -- 1/3 of the remaining cookies goes to the family
    (eaten_cookies : ℕ) -- Maria eats 4 cookies
    (neighbor_share : ℕ) -- Maria gives 1/6 of the remaining cookies to neighbor
    (initial_cookies : total_cookies = 60)
    (friend_fraction : friend_share = total_cookies * 20 / 100)
    (remaining_after_friend : ℕ := total_cookies - friend_share)
    (family_fraction : family_share = remaining_after_friend / 3)
    (remaining_after_family : ℕ := remaining_after_friend - family_share)
    (eaten : eaten_cookies = 4)
    (remaining_after_eating : ℕ := remaining_after_family - eaten_cookies)
    (neighbor_fraction : neighbor_share = remaining_after_eating / 6)
    (neighbor_integerized : neighbor_share = 4) -- assumed whole number for neighbor's share
    (remaining_after_neighbor : ℕ := remaining_after_eating - neighbor_share) : 
    remaining_after_neighbor = 24 :=
sorry  -- The statement matches the problem, proof is left out

end maria_cookies_left_l285_285346


namespace points_four_units_away_l285_285960

theorem points_four_units_away (x : ℚ) (h : |x| = 4) : x = -4 ∨ x = 4 := 
by 
  sorry

end points_four_units_away_l285_285960


namespace geese_count_l285_285548

-- Define the number of ducks in the marsh
def number_of_ducks : ℕ := 37

-- Define the total number of birds in the marsh
def total_number_of_birds : ℕ := 95

-- Define the number of geese in the marsh
def number_of_geese : ℕ := total_number_of_birds - number_of_ducks

-- Theorem stating the number of geese in the marsh is 58
theorem geese_count : number_of_geese = 58 := by
  sorry

end geese_count_l285_285548


namespace value_of_w_l285_285189

theorem value_of_w (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := 
sorry

end value_of_w_l285_285189


namespace balance_rearrangement_vowels_at_end_l285_285183

theorem balance_rearrangement_vowels_at_end : 
  let vowels := ['A', 'A', 'E'];
  let consonants := ['B', 'L', 'N', 'C'];
  (Nat.factorial 3 / Nat.factorial 2) * Nat.factorial 4 = 72 :=
by
  sorry

end balance_rearrangement_vowels_at_end_l285_285183


namespace prob_all_three_defective_approx_l285_285859

noncomputable def prob_defective (total: ℕ) (defective: ℕ) := (defective : ℚ) / (total : ℚ)

theorem prob_all_three_defective_approx :
  let p_total := 120
  let s_total := 160
  let b_total := 60
  let p_def := 26
  let s_def := 68
  let b_def := 30
  let p_prob := prob_defective p_total p_def
  let s_prob := prob_defective s_total s_def
  let b_prob := prob_defective b_total b_def
  let combined_prob := p_prob * s_prob * b_prob
  abs (combined_prob - (221 / 4800 : ℚ)) < 0.001 :=
by
  sorry

end prob_all_three_defective_approx_l285_285859


namespace arithmetic_mean_of_distribution_l285_285110

-- Defining conditions
def stddev : ℝ := 2.3
def value : ℝ := 11.6

-- Proving the mean (μ) is 16.2
theorem arithmetic_mean_of_distribution : ∃ μ : ℝ, μ = 16.2 ∧ value = μ - 2 * stddev :=
by
  use 16.2
  sorry

end arithmetic_mean_of_distribution_l285_285110


namespace x0_in_M_implies_x0_in_N_l285_285622

def M : Set ℝ := {x | ∃ (k : ℤ), x = k + 1 / 2}
def N : Set ℝ := {x | ∃ (k : ℤ), x = k / 2 + 1}

theorem x0_in_M_implies_x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := 
sorry

end x0_in_M_implies_x0_in_N_l285_285622


namespace geometric_series_sum_l285_285039

def first_term : ℤ := 3
def common_ratio : ℤ := -2
def last_term : ℤ := -1536
def num_terms : ℕ := 10
def sum_of_series (a r : ℤ) (n : ℕ) : ℤ := a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum :
  sum_of_series first_term common_ratio num_terms = -1023 := by
  sorry

end geometric_series_sum_l285_285039


namespace sufficient_and_necessary_condition_l285_285803

def A : Set ℝ := { x | x - 2 > 0 }

def B : Set ℝ := { x | x < 0 }

def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem sufficient_and_necessary_condition :
  ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C :=
sorry

end sufficient_and_necessary_condition_l285_285803


namespace compute_expression_l285_285561

theorem compute_expression : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 :=
by
  sorry

end compute_expression_l285_285561


namespace count_four_digit_numbers_with_repeated_digits_l285_285695

def countDistinctFourDigitNumbersWithRepeatedDigits : Nat :=
  let totalNumbers := 4 ^ 4
  let uniqueNumbers := 4 * 3 * 2 * 1
  totalNumbers - uniqueNumbers

theorem count_four_digit_numbers_with_repeated_digits :
  countDistinctFourDigitNumbersWithRepeatedDigits = 232 := by
  sorry

end count_four_digit_numbers_with_repeated_digits_l285_285695


namespace wickets_before_last_match_l285_285562

-- Define the conditions
variable (W : ℕ)

-- Initial average
def initial_avg : ℝ := 12.4

-- Runs given in the last match
def runs_last_match : ℝ := 26

-- Wickets taken in the last match
def wickets_last_match : ℕ := 4

-- The new average after the last match
def new_avg : ℝ := initial_avg - 0.4

-- Prove the theorem
theorem wickets_before_last_match :
  (12.4 * W + runs_last_match) / (W + wickets_last_match) = new_avg → W = 55 :=
by
  sorry

end wickets_before_last_match_l285_285562


namespace eq_x_in_terms_of_y_l285_285621

theorem eq_x_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : x = (5 - y) / 2 := by
  sorry

end eq_x_in_terms_of_y_l285_285621


namespace white_tiles_count_l285_285339

theorem white_tiles_count (total_tiles yellow_tiles purple_tiles : ℕ)
    (hy : yellow_tiles = 3)
    (hb : ∃ blue_tiles, blue_tiles = yellow_tiles + 1)
    (hp : purple_tiles = 6)
    (ht : total_tiles = 20) : 
    ∃ white_tiles, white_tiles = 7 :=
by
  obtain ⟨blue_tiles, hb_eq⟩ := hb
  let non_white_tiles := yellow_tiles + blue_tiles + purple_tiles
  have hnwt : non_white_tiles = 3 + (3 + 1) + 6,
  {
    rw [hy, hp, hb_eq],
    ring,
  }
  have hwt : total_tiles - non_white_tiles = 7,
  {
    rw ht,
    rw hnwt,
    norm_num,
  }
  use total_tiles - non_white_tiles,
  exact hwt,

end white_tiles_count_l285_285339


namespace create_proper_six_sided_figure_l285_285683

-- Definition of a matchstick configuration
structure MatchstickConfig where
  sides : ℕ
  matchsticks : ℕ

-- Initial configuration: a regular hexagon with 6 matchsticks
def initialConfig : MatchstickConfig := ⟨6, 6⟩

-- Condition: Cannot lay any stick on top of another, no free ends
axiom no_overlap (cfg : MatchstickConfig) : Prop
axiom no_free_ends (cfg : MatchstickConfig) : Prop

-- New configuration after adding 3 matchsticks
def newConfig : MatchstickConfig := ⟨6, 9⟩

-- Theorem stating the possibility to create a proper figure with six sides
theorem create_proper_six_sided_figure : no_overlap newConfig → no_free_ends newConfig → newConfig.sides = 6 :=
by
  sorry

end create_proper_six_sided_figure_l285_285683


namespace recurring_decimal_to_fraction_l285_285397

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l285_285397


namespace solve_system_of_equations_in_nat_numbers_l285_285528

theorem solve_system_of_equations_in_nat_numbers :
  ∃ a b c d : ℕ, a * b = c + d ∧ c * d = a + b ∧ a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 :=
by
  sorry

end solve_system_of_equations_in_nat_numbers_l285_285528


namespace brooke_homework_time_l285_285605

def num_math_problems := 15
def num_social_studies_problems := 6
def num_science_problems := 10

def time_per_math_problem := 2 -- in minutes
def time_per_social_studies_problem := 0.5 -- in minutes (30 seconds)
def time_per_science_problem := 1.5 -- in minutes

def total_time : ℝ :=
  num_math_problems * time_per_math_problem +
  num_social_studies_problems * time_per_social_studies_problem +
  num_science_problems * time_per_science_problem

theorem brooke_homework_time :
  total_time = 48 := by
  sorry

end brooke_homework_time_l285_285605


namespace journey_distance_l285_285707

theorem journey_distance :
  ∃ D : ℝ, (D / 42 + D / 48 = 10) ∧ D = 224 :=
by
  sorry

end journey_distance_l285_285707


namespace fraction_equiv_l285_285401

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l285_285401


namespace number_B_expression_l285_285820

theorem number_B_expression (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4 / 5) :=
sorry

end number_B_expression_l285_285820


namespace greatest_non_sum_complex_l285_285804

def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n

theorem greatest_non_sum_complex : ∀ n : ℕ, (¬ ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ a + b = n) → n ≤ 23 :=
by {
  sorry
}

end greatest_non_sum_complex_l285_285804


namespace casey_stays_for_n_months_l285_285869

-- Definitions based on conditions.
def weekly_cost : ℕ := 280
def monthly_cost : ℕ := 1000
def weeks_per_month : ℕ := 4
def total_savings : ℕ := 360

-- Calculate monthly cost when paying weekly.
def monthly_cost_weekly := weekly_cost * weeks_per_month

-- Calculate savings per month when paying monthly instead of weekly.
def savings_per_month := monthly_cost_weekly - monthly_cost

-- Define the problem statement.
theorem casey_stays_for_n_months :
  (total_savings / savings_per_month) = 3 := by
  -- Proof is omitted.
  sorry

end casey_stays_for_n_months_l285_285869


namespace total_students_l285_285159

theorem total_students (x : ℕ) (h1 : 3 * x + 8 = 3 * x + 5) (h2 : 5 * (x - 1) + 3 > 3 * x + 8) : x = 6 :=
sorry

end total_students_l285_285159


namespace weather_station_accuracy_l285_285529

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

theorem weather_station_accuracy :
  binomial_probability 3 2 0.9 = 0.243 :=
by
  sorry

end weather_station_accuracy_l285_285529


namespace intersection_of_sets_l285_285309

def SetA : Set ℝ := { x | |x| ≤ 1 }
def SetB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_of_sets : (SetA ∩ SetB) = { x | 0 ≤ x ∧ x ≤ 1 } := 
by
  sorry

end intersection_of_sets_l285_285309


namespace solve_eq_l285_285187

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end solve_eq_l285_285187


namespace person_speed_l285_285019

theorem person_speed (d_meters : ℕ) (t_minutes : ℕ) (d_km t_hours : ℝ) :
  (d_meters = 1800) →
  (t_minutes = 12) →
  (d_km = d_meters / 1000) →
  (t_hours = t_minutes / 60) →
  d_km / t_hours = 9 :=
by
  intros
  sorry

end person_speed_l285_285019


namespace percentage_calculation_l285_285406

theorem percentage_calculation :
  ( (2 / 3 * 2432 / 3 + 1 / 6 * 3225) / 450 * 100 ) = 239.54 := 
sorry

end percentage_calculation_l285_285406


namespace roundness_1000000_l285_285816

-- Definitions based on the conditions in the problem
def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  if n = 1 then []
  else [(2, 6), (5, 6)] -- Example specifically for 1,000,000

def roundness (n : ℕ) : ℕ :=
  (prime_factors n).map Prod.snd |>.sum

-- The main theorem
theorem roundness_1000000 : roundness 1000000 = 12 := by
  sorry

end roundness_1000000_l285_285816


namespace value_of_x_l285_285073

theorem value_of_x (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := 
by 
  sorry

end value_of_x_l285_285073


namespace largest_percentage_drop_l285_285126

theorem largest_percentage_drop (jan feb mar apr may jun : ℤ) 
  (h_jan : jan = -10)
  (h_feb : feb = 5)
  (h_mar : mar = -15)
  (h_apr : apr = 10)
  (h_may : may = -30)
  (h_jun : jun = 0) :
  may = -30 ∧ ∀ month, month ≠ may → month ≥ -30 :=
by
  sorry

end largest_percentage_drop_l285_285126


namespace sqrt_meaningful_range_l285_285777

theorem sqrt_meaningful_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end sqrt_meaningful_range_l285_285777


namespace white_tile_count_l285_285336

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l285_285336


namespace problem1_l285_285815

theorem problem1 (x y : ℝ) (h1 : 2^(x + y) = x + 7) (h2 : x + y = 3) : (x = 1 ∧ y = 2) :=
by
  sorry

end problem1_l285_285815


namespace last_colored_cell_is_51_50_l285_285206

def last_spiral_cell (width height : ℕ) : ℕ × ℕ :=
  -- Assuming an external or pre-defined process to calculate the last cell for a spiral pattern
  sorry 

theorem last_colored_cell_is_51_50 :
  last_spiral_cell 200 100 = (51, 50) :=
sorry

end last_colored_cell_is_51_50_l285_285206


namespace contrapositive_example_l285_285956

theorem contrapositive_example (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_example_l285_285956


namespace tan_alpha_plus_pi_over_4_l285_285059

theorem tan_alpha_plus_pi_over_4 
  {α β : ℝ} 
  (h1 : Real.tan (α + β) = 2/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α + π/4) = 9/8 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_l285_285059


namespace circles_intersect_l285_285733

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end circles_intersect_l285_285733


namespace largest_of_nine_consecutive_integers_l285_285545

theorem largest_of_nine_consecutive_integers (sum_eq_99: ∃ (n : ℕ), 99 = (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) : 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end largest_of_nine_consecutive_integers_l285_285545


namespace corveus_sleep_deficit_l285_285453

theorem corveus_sleep_deficit :
  let weekday_sleep := 5 -- 4 hours at night + 1-hour nap
  let weekend_sleep := 5 -- 5 hours at night, no naps
  let total_weekday_sleep := 5 * weekday_sleep
  let total_weekend_sleep := 2 * weekend_sleep
  let total_sleep := total_weekday_sleep + total_weekend_sleep
  let recommended_sleep_per_day := 6
  let total_recommended_sleep := 7 * recommended_sleep_per_day
  let sleep_deficit := total_recommended_sleep - total_sleep
  sleep_deficit = 7 :=
by
  -- Insert proof steps here
  sorry

end corveus_sleep_deficit_l285_285453


namespace count_integers_log_inequality_l285_285467

open Real

theorem count_integers_log_inequality : 
  ∃ (n : ℕ), n = 28 ∧ ∀ (x : ℤ), 50 < x ∧ x < 80 → log 10 ((x - 50) * (80 - x)) < 1.5 :=
by {
  sorry
}

end count_integers_log_inequality_l285_285467


namespace min_B_minus_A_l285_285892

noncomputable def S_n (n : ℕ) : ℚ :=
  let a1 : ℚ := 2
  let r : ℚ := -1 / 3
  a1 * (1 - r ^ n) / (1 - r)

theorem min_B_minus_A :
  ∃ A B : ℚ, 
    (∀ n : ℕ, 1 ≤ n → A ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B) ∧
    ∀ A' B' : ℚ, 
      (∀ n : ℕ, 1 ≤ n → A' ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B') → 
      B' - A' ≥ 9 / 4 ∧ B - A = 9 / 4 :=
sorry

end min_B_minus_A_l285_285892


namespace find_d_l285_285362

noncomputable def single_point_graph (d : ℝ) : Prop :=
  ∃ x y : ℝ, 3 * x^2 + 2 * y^2 + 9 * x - 14 * y + d = 0

theorem find_d : single_point_graph 31.25 :=
sorry

end find_d_l285_285362


namespace opposite_of_neg3_l285_285385

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l285_285385


namespace perpendicular_vectors_implies_m_value_l285_285633

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l285_285633


namespace absolute_value_inequality_l285_285888

theorem absolute_value_inequality (x : ℝ) : 
  (|3 * x + 1| > 2) ↔ (x > 1/3 ∨ x < -1) := by
  sorry

end absolute_value_inequality_l285_285888


namespace ratio_equal_one_of_log_conditions_l285_285089

noncomputable def logBase (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem ratio_equal_one_of_log_conditions
  (p q : ℝ)
  (hp : 0 < p)
  (hq : 0 < q)
  (h : logBase 8 p = logBase 18 q ∧ logBase 18 q = logBase 24 (p + 2 * q)) :
  q / p = 1 :=
by
  sorry

end ratio_equal_one_of_log_conditions_l285_285089


namespace magnitude_of_b_l285_285180

variable (a b : ℝ)

-- Defining the given conditions as hypotheses
def condition1 : Prop := (a - b) * (a - b) = 9
def condition2 : Prop := (a + 2 * b) * (a + 2 * b) = 36
def condition3 : Prop := a^2 + (a * b) - 2 * b^2 = -9

-- Defining the theorem to prove
theorem magnitude_of_b (ha : condition1 a b) (hb : condition2 a b) (hc : condition3 a b) : b^2 = 3 := 
sorry

end magnitude_of_b_l285_285180


namespace invest_in_yourself_examples_l285_285753

theorem invest_in_yourself_examples (example1 example2 example3 : String)
  (benefit1 benefit2 benefit3 : String)
  (h1 : example1 = "Investment in Education")
  (h2 : benefit1 = "Spending money on education improves knowledge and skills, leading to better job opportunities and higher salaries. Education appreciates over time, providing financial stability.")
  (h3 : example2 = "Investment in Physical Health")
  (h4 : benefit2 = "Spending on sports activities, fitness programs, or healthcare prevents chronic diseases, saves future medical expenses, and enhances overall well-being.")
  (h5 : example3 = "Time Spent on Reading Books")
  (h6 : benefit3 = "Reading books expands knowledge, improves vocabulary and cognitive abilities, develops critical thinking and analytical skills, and fosters creativity and empathy."):
  "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." = "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." :=
by
  sorry

end invest_in_yourself_examples_l285_285753


namespace value_of_2_pow_b_l285_285906

theorem value_of_2_pow_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h1 : (2 ^ a) ^ b = 2 ^ 2) (h2 : 2 ^ a * 2 ^ b = 8) : 2 ^ b = 4 :=
by
  sorry

end value_of_2_pow_b_l285_285906


namespace set_characteristics_l285_285245

-- Define the characteristics of elements in a set
def characteristic_definiteness := true
def characteristic_distinctness := true
def characteristic_unorderedness := true
def characteristic_reality := false -- We aim to prove this

-- The problem statement in Lean
theorem set_characteristics :
  ¬ characteristic_reality :=
by
  -- Here would be the proof, but we add sorry as indicated.
  sorry

end set_characteristics_l285_285245


namespace math_problem_l285_285592

theorem math_problem : (-4)^2 * ((-1)^2023 + (3 / 4) + (-1 / 2)^3) = -6 := 
by 
  sorry

end math_problem_l285_285592


namespace number_of_students_taking_math_l285_285273

variable (totalPlayers physicsOnly physicsAndMath mathOnly : ℕ)
variable (h1 : totalPlayers = 15) (h2 : physicsOnly = 9) (h3 : physicsAndMath = 3)

theorem number_of_students_taking_math : mathOnly = 9 :=
by {
  sorry
}

end number_of_students_taking_math_l285_285273


namespace sin_product_identity_sin_cos_fraction_identity_l285_285415

-- First Proof Problem: Proving that the product of sines equals the given value
theorem sin_product_identity :
  (Real.sin (Real.pi * 6 / 180) * 
   Real.sin (Real.pi * 42 / 180) * 
   Real.sin (Real.pi * 66 / 180) * 
   Real.sin (Real.pi * 78 / 180)) = 
  (Real.sqrt 5 - 1) / 32 := 
by 
  sorry

-- Second Proof Problem: Given sin alpha and alpha in the second quadrant, proving the given fraction value
theorem sin_cos_fraction_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + Real.pi / 4)) / 
  (Real.sin (2 * α) + Real.cos (2 * α) + 1) = 
  -Real.sqrt 2 :=
by 
  sorry

end sin_product_identity_sin_cos_fraction_identity_l285_285415


namespace find_percentage_l285_285571

theorem find_percentage (P : ℝ) : 
  0.15 * P * (0.5 * 5600) = 126 → P = 0.3 := 
by 
  sorry

end find_percentage_l285_285571


namespace time_to_empty_l285_285247

-- Definitions for the conditions
def rate_fill_no_leak (R : ℝ) := R = 1 / 2 -- Cistern fills in 2 hours without leak
def effective_fill_rate (R L : ℝ) := R - L = 1 / 4 -- Effective fill rate when leaking
def remember_fill_time_leak (R L : ℝ) := (R - L) * 4 = 1 -- 4 hours to fill with leak

-- Main theorem statement
theorem time_to_empty (R L : ℝ) (h1 : rate_fill_no_leak R) (h2 : effective_fill_rate R L)
  (h3 : remember_fill_time_leak R L) : (1 / L = 4) :=
by
  sorry

end time_to_empty_l285_285247


namespace symmetric_pentominoes_count_l285_285278

-- Assume we have exactly fifteen pentominoes
def num_pentominoes : ℕ := 15

-- Define the number of pentominoes with particular symmetrical properties
def num_reflectional_symmetry : ℕ := 8
def num_rotational_symmetry : ℕ := 3
def num_both_symmetries : ℕ := 2

-- The theorem we wish to prove
theorem symmetric_pentominoes_count 
  (n_p : ℕ) (n_r : ℕ) (n_b : ℕ) (n_tot : ℕ)
  (h1 : n_p = num_pentominoes)
  (h2 : n_r = num_reflectional_symmetry)
  (h3 : n_b = num_both_symmetries)
  (h4 : n_tot = n_r + num_rotational_symmetry - n_b) :
  n_tot = 9 := 
sorry

end symmetric_pentominoes_count_l285_285278


namespace Mark_owes_total_l285_285203

noncomputable def base_fine : ℕ := 50

def additional_fine (speed_over_limit : ℕ) : ℕ :=
  let first_10 := min speed_over_limit 10 * 2
  let next_5 := min (speed_over_limit - 10) 5 * 3
  let next_10 := min (speed_over_limit - 15) 10 * 5
  let remaining := max (speed_over_limit - 25) 0 * 6
  first_10 + next_5 + next_10 + remaining

noncomputable def total_fine (base : ℕ) (additional : ℕ) (school_zone : Bool) : ℕ :=
  let fine := base + additional
  if school_zone then fine * 2 else fine

def court_costs : ℕ := 350

noncomputable def processing_fee (fine : ℕ) : ℕ := fine / 10

def lawyer_fees (hourly_rate : ℕ) (hours : ℕ) : ℕ := hourly_rate * hours

theorem Mark_owes_total :
  let speed_over_limit := 45
  let base := base_fine
  let additional := additional_fine speed_over_limit
  let school_zone := true
  let fine := total_fine base additional school_zone
  let total_fine_with_costs := fine + court_costs
  let processing := processing_fee total_fine_with_costs
  let lawyer := lawyer_fees 100 4
  let total := total_fine_with_costs + processing + lawyer
  total = 1346 := sorry

end Mark_owes_total_l285_285203


namespace evaluate_expression_l285_285173

theorem evaluate_expression (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := 
sorry

end evaluate_expression_l285_285173


namespace fraction_never_simplifiable_l285_285941

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_never_simplifiable_l285_285941


namespace number_of_girls_l285_285227

theorem number_of_girls 
  (B G : ℕ) 
  (h1 : B + G = 480) 
  (h2 : 5 * B = 3 * G) :
  G = 300 := 
sorry

end number_of_girls_l285_285227


namespace sum_of_squares_of_coeffs_l285_285593

theorem sum_of_squares_of_coeffs (a b c : ℕ) : (a = 6) → (b = 24) → (c = 12) → (a^2 + b^2 + c^2 = 756) :=
by
  sorry

end sum_of_squares_of_coeffs_l285_285593


namespace correct_multiplication_result_l285_285992

theorem correct_multiplication_result :
  0.08 * 3.25 = 0.26 :=
by
  -- This is to ensure that the theorem is well-formed and logically connected
  sorry

end correct_multiplication_result_l285_285992


namespace product_of_series_l285_285036

theorem product_of_series :
  (1 - 1/2^2) * (1 - 1/3^2) * (1 - 1/4^2) * (1 - 1/5^2) * (1 - 1/6^2) *
  (1 - 1/7^2) * (1 - 1/8^2) * (1 - 1/9^2) * (1 - 1/10^2) = 11 / 20 :=
by 
  sorry

end product_of_series_l285_285036


namespace league_games_and_weeks_l285_285689

/--
There are 15 teams in a league, and each team plays each of the other teams exactly once.
Due to scheduling limitations, each team can only play one game per week.
Prove that the total number of games played is 105 and the minimum number of weeks needed to complete all the games is 15.
-/
theorem league_games_and_weeks :
  let teams := 15
  let total_games := teams * (teams - 1) / 2
  let games_per_week := Nat.div teams 2
  total_games = 105 ∧ total_games / games_per_week = 15 :=
by
  sorry

end league_games_and_weeks_l285_285689


namespace jim_miles_driven_l285_285917

theorem jim_miles_driven (total_journey : ℕ) (miles_needed : ℕ) (h : total_journey = 1200 ∧ miles_needed = 985) : total_journey - miles_needed = 215 := 
by sorry

end jim_miles_driven_l285_285917


namespace twelfth_term_of_geometric_sequence_l285_285534

theorem twelfth_term_of_geometric_sequence (a : ℕ) (r : ℕ) (h1 : a * r ^ 4 = 8) (h2 : a * r ^ 8 = 128) : 
  a * r ^ 11 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l285_285534


namespace cube_mod7_not_divisible_7_l285_285645

theorem cube_mod7_not_divisible_7 (a : ℤ) (h : ¬ (7 ∣ a)) :
  (a^3 % 7 = 1) ∨ (a^3 % 7 = -1) :=
sorry

end cube_mod7_not_divisible_7_l285_285645


namespace produce_total_worth_l285_285610

/-- Gary is restocking the grocery produce section. He adds 60 bundles of asparagus at $3.00 each, 
40 boxes of grapes at $2.50 each, and 700 apples at $0.50 each. 
This theorem proves that the total worth of all the produce Gary stocked is $630.00. -/
theorem produce_total_worth :
  let asparagus_bundles := 60
  let asparagus_price := 3.00
  let grapes_boxes := 40
  let grapes_price := 2.50
  let apples_count := 700
  let apples_price := 0.50 in
  (asparagus_bundles * asparagus_price) + (grapes_boxes * grapes_price) + (apples_count * apples_price) = 630.00 :=
by
  sorry

end produce_total_worth_l285_285610


namespace larry_substitution_l285_285668

theorem larry_substitution (a b c d e : ℤ)
  (ha : a = 1)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 4)
  (h_ignored : a - b - c - d + e = a - (b - (c - (d + e)))) :
  e = 3 :=
by
  sorry

end larry_substitution_l285_285668


namespace geometric_series_3000_terms_sum_l285_285963

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_3000_terms_sum
    (a r : ℝ)
    (h_r : r ≠ 1)
    (sum_1000 : geometric_sum a r 1000 = 500)
    (sum_2000 : geometric_sum a r 2000 = 950) :
  geometric_sum a r 3000 = 1355 :=
by 
  sorry

end geometric_series_3000_terms_sum_l285_285963


namespace find_value_of_y_l285_285549

variable (p y : ℝ)
variable (h1 : p > 45)
variable (h2 : p * p / 100 = (2 * p / 300) * (p + y))

theorem find_value_of_y (h1 : p > 45) (h2 : p * p / 100 = (2 * p / 300) * (p + y)) : y = p / 2 :=
sorry

end find_value_of_y_l285_285549


namespace less_than_its_reciprocal_l285_285408

-- Define the numbers as constants
def a := -1/3
def b := -3/2
def c := 1/4
def d := 3/4
def e := 4/3 

-- Define the proposition that needs to be proved
theorem less_than_its_reciprocal (n : ℚ) :
  (n = -3/2 ∨ n = 1/4) ↔ (n < 1/n) :=
by
  sorry

end less_than_its_reciprocal_l285_285408


namespace max_S_n_of_arithmetic_seq_l285_285061

theorem max_S_n_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h2 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h3 : a 1 + a 3 + a 5 = 15)
  (h4 : a 2 + a 4 + a 6 = 0) : 
  ∃ n : ℕ, S n = 40 ∧ (∀ m : ℕ, S m ≤ 40) :=
sorry

end max_S_n_of_arithmetic_seq_l285_285061


namespace boundary_of_shadow_of_sphere_l285_285584

theorem boundary_of_shadow_of_sphere (x y : ℝ) :
  let O := (0, 0, 2)
  let P := (1, -2, 3)
  let r := 2
  (∃ T : ℝ × ℝ × ℝ,
    T = (0, -2, 2) ∧
    (∃ g : ℝ → ℝ,
      y = g x ∧
      g x = (x^2 - 2 * x - 11) / 6)) → 
  y = (x^2 - 2 * x - 11) / 6 :=
by
  sorry

end boundary_of_shadow_of_sphere_l285_285584


namespace packages_of_gum_l285_285210

-- Define the conditions
variables (P : Nat) -- Number of packages Robin has

-- State the theorem
theorem packages_of_gum (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end packages_of_gum_l285_285210


namespace tom_speed_first_part_l285_285238

-- Definitions of conditions in Lean
def total_distance : ℕ := 20
def distance_first_part : ℕ := 10
def speed_second_part : ℕ := 10
def average_speed : ℚ := 10.909090909090908
def distance_second_part := total_distance - distance_first_part

-- Lean statement to prove the speed during the first part of the trip
theorem tom_speed_first_part (v : ℚ) :
  (distance_first_part / v + distance_second_part / speed_second_part) = total_distance / average_speed → v = 12 :=
by
  intro h
  sorry

end tom_speed_first_part_l285_285238


namespace students_at_end_of_year_l285_285080

def n_start := 10
def n_left := 4
def n_new := 42

theorem students_at_end_of_year : n_start - n_left + n_new = 48 := by
  sorry

end students_at_end_of_year_l285_285080


namespace scientific_notation_of_8200000_l285_285013

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l285_285013


namespace number_drawn_from_first_group_l285_285002

theorem number_drawn_from_first_group (n: ℕ) (groups: ℕ) (interval: ℕ) (fourth_group_number: ℕ) (total_bags: ℕ) 
    (h1: total_bags = 50) (h2: groups = 5) (h3: interval = total_bags / groups)
    (h4: interval = 10) (h5: fourth_group_number = 36) : n = 6 :=
by
  sorry

end number_drawn_from_first_group_l285_285002


namespace total_lives_after_third_level_l285_285780

def initial_lives : ℕ := 2

def extra_lives_first_level : ℕ := 6
def modifier_first_level (lives : ℕ) : ℕ := lives / 2

def extra_lives_second_level : ℕ := 11
def challenge_second_level (lives : ℕ) : ℕ := lives - 3

def reward_third_level (lives_first_two_levels : ℕ) : ℕ := 2 * lives_first_two_levels

theorem total_lives_after_third_level :
  let lives_first_level := modifier_first_level extra_lives_first_level
  let lives_after_first_level := initial_lives + lives_first_level
  let lives_second_level := challenge_second_level extra_lives_second_level
  let lives_after_second_level := lives_after_first_level + lives_second_level
  let total_gained_lives_first_two_levels := lives_first_level + lives_second_level
  let third_level_reward := reward_third_level total_gained_lives_first_two_levels
  lives_after_second_level + third_level_reward = 35 :=
by
  sorry

end total_lives_after_third_level_l285_285780


namespace div_power_l285_285007

theorem div_power (h : 27 = 3 ^ 3) : 3 ^ 12 / 27 ^ 2 = 729 :=
by {
  calc
    3 ^ 12 / 27 ^ 2 = 3 ^ 12 / (3 ^ 3) ^ 2 : by rw h
               ... = 3 ^ 12 / 3 ^ 6       : by rw pow_mul
               ... = 3 ^ (12 - 6)         : by rw div_eq_sub_pow
               ... = 3 ^ 6                : by rw sub_self_pow
               ... = 729                  : by norm_num,
  sorry
}

end div_power_l285_285007


namespace runners_never_meet_l285_285001

theorem runners_never_meet
    (x : ℕ)  -- Speed of first runner
    (a : ℕ)  -- 1/3 of the circumference of the track
    (C : ℕ)  -- Circumference of the track
    (hC : C = 3 * a)  -- Given that C = 3 * a
    (h_speeds : 1 * x = x ∧ 2 * x = 2 * x ∧ 4 * x = 4 * x)  -- Speed ratios: 1:2:4
    (t : ℕ)  -- Time variable
: ¬(∃ t, (x * t % C = 2 * x * t % C ∧ 2 * x * t % C = 4 * x * t % C)) :=
by sorry

end runners_never_meet_l285_285001


namespace medieval_society_hierarchy_l285_285578

-- Given conditions
def members := 12
def king_choices := members
def remaining_after_king := members - 1
def duke_choices : ℕ := remaining_after_king * (remaining_after_king - 1) * (remaining_after_king - 2)
def knight_choices : ℕ := Nat.choose (remaining_after_king - 2) 2 * Nat.choose (remaining_after_king - 4) 2 * Nat.choose (remaining_after_king - 6) 2

-- The number of ways to establish the hierarchy can be stated as:
def total_ways : ℕ := king_choices * duke_choices * knight_choices

-- Our main theorem
theorem medieval_society_hierarchy : total_ways = 907200 := by
  -- Proof would go here, we skip it with sorry
  sorry

end medieval_society_hierarchy_l285_285578


namespace area_of_X_part_l285_285444

theorem area_of_X_part :
    (∃ s : ℝ, s^2 = 2520 ∧ 
     (∃ E F G H : ℝ, E = F ∧ F = G ∧ G = H ∧ 
         E = s / 4 ∧ F = s / 4 ∧ G = s / 4 ∧ H = s / 4) ∧ 
     2520 * 11 / 24 = 1155) :=
by
  sorry

end area_of_X_part_l285_285444


namespace right_triangle_acute_angle_l285_285498

theorem right_triangle_acute_angle (A B : ℝ) (h₁ : A + B = 90) (h₂ : A = 40) : B = 50 :=
by
  sorry

end right_triangle_acute_angle_l285_285498


namespace enclosed_area_of_curve_l285_285682

/-
  The closed curve in the figure is made up of 9 congruent circular arcs each of length \(\frac{\pi}{2}\),
  where each of the centers of the corresponding circles is among the vertices of a regular hexagon of side 3.
  We want to prove that the area enclosed by the curve is \(\frac{27\sqrt{3}}{2} + \frac{9\pi}{8}\).
-/

theorem enclosed_area_of_curve :
  let side_length := 3
  let arc_length := π / 2
  let num_arcs := 9
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length^2
  let radius := 1 / 2
  let sector_area := (π * radius^2) / 4
  let total_sector_area := num_arcs * sector_area
  let enclosed_area := hexagon_area + total_sector_area
  enclosed_area = (27 * Real.sqrt 3) / 2 + (9 * π) / 8 :=
by
  sorry

end enclosed_area_of_curve_l285_285682


namespace f_of_x_plus_1_f_of_2_f_of_x_l285_285756

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x := sorry

theorem f_of_2 : f 2 = 3 := sorry

theorem f_of_x (x : ℝ) : f x = x^2 - 1 := sorry

end f_of_x_plus_1_f_of_2_f_of_x_l285_285756


namespace hyperbola_equation_l285_285764

noncomputable def sqrt_cubed := Real.sqrt 3

theorem hyperbola_equation
  (P : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (hP : P = (1, sqrt_cubed))
  (hAsymptote : (1 / a)^2 - (sqrt_cubed / b)^2 = 0)
  (hAngle : ∀ F : ℝ × ℝ, ∀ O : ℝ × ℝ, (F.1 - 1)^2 + (F.2 - sqrt_cubed)^2 + F.1^2 + F.2^2 = 16) :
  (a^2 = 4) ∧ (b^2 = 12) ∧ (c = 4) →
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 :=
by
  sorry

end hyperbola_equation_l285_285764


namespace frac_diff_zero_l285_285168

theorem frac_diff_zero (a b : ℝ) (h : a + b = a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a) - (1 / b) = 0 := 
sorry

end frac_diff_zero_l285_285168


namespace fraction_sum_is_half_l285_285446

theorem fraction_sum_is_half :
  (1/5 : ℚ) + (3/10 : ℚ) = 1/2 :=
by linarith

end fraction_sum_is_half_l285_285446


namespace circle_circumference_l285_285580

theorem circle_circumference (a b : ℝ) (h1 : a = 9) (h2 : b = 12) :
  ∃ c : ℝ, c = 15 * Real.pi :=
by
  sorry

end circle_circumference_l285_285580


namespace part_a_part_b_l285_285196

-- This definition states that a number p^m is a divisor of a-1
def divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  (p ^ m) ∣ (a - 1)

-- This definition states that (p^(m+1)) is not a divisor of a-1
def not_divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  ¬ (p ^ (m + 1) ∣ (a - 1))

-- Part (a): Prove divisibility
theorem part_a (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  p ^ (m + n) ∣ a ^ (p ^ n) - 1 := 
sorry

-- Part (b): Prove non-divisibility
theorem part_b (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  ¬ p ^ (m + n + 1) ∣ a ^ (p ^ n) - 1 := 
sorry

end part_a_part_b_l285_285196


namespace min_k_spherical_cap_cylinder_l285_285269

/-- Given a spherical cap and a cylinder sharing a common inscribed sphere with volumes V1 and V2 respectively,
we show that the minimum value of k such that V1 = k * V2 is 4/3. -/
theorem min_k_spherical_cap_cylinder (R : ℝ) (V1 V2 : ℝ) (h1 : V1 = (4/3) * π * R^3) 
(h2 : V2 = 2 * π * R^3) : 
∃ k : ℝ, V1 = k * V2 ∧ k = 4/3 := 
by 
  use (4/3)
  constructor
  . sorry
  . sorry

end min_k_spherical_cap_cylinder_l285_285269


namespace find_p_l285_285043

theorem find_p (p : ℝ) :
  (∀ x : ℝ, x^2 + p * x + p - 1 = 0) →
  ((exists x1 x2 : ℝ, x^2 + p * x + p - 1 = 0 ∧ x1^2 + x1^3 = - (x2^2 + x2^3) ) → (p = 1 ∨ p = 2)) :=
by
  intro h
  sorry

end find_p_l285_285043


namespace none_of_these_true_l285_285799

variable (s r p q : ℝ)
variable (hs : s > 0) (hr : r > 0) (hpq : p * q ≠ 0) (h : s * (p * r) > s * (q * r))

theorem none_of_these_true : ¬ (-p > -q) ∧ ¬ (-p > q) ∧ ¬ (1 > -q / p) ∧ ¬ (1 < q / p) :=
by
  -- The hypothetical theorem to be proven would continue here
  sorry

end none_of_these_true_l285_285799


namespace sum_of_coordinates_point_D_l285_285208

theorem sum_of_coordinates_point_D 
(M : ℝ × ℝ) (C D : ℝ × ℝ) 
(hM : M = (3, 5)) 
(hC : C = (1, 10)) 
(hmid : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
: D.1 + D.2 = 5 :=
sorry

end sum_of_coordinates_point_D_l285_285208


namespace mutually_exclusive_case_B_l285_285973

open Finset

-- Define the events
def atLeastOneHead (outcome : set (Fin 3 × bool)) : Prop :=
  {h | ∃ i : Fin 3, (i, true) ∈ outcome} ⊆ outcome

def atMostOneHead (outcome : set (Fin 3 × bool)) : Prop :=
  {h | ∃ i : Fin 3, ∀ j : Fin 3, j ≠ i → (j, false) ∈ outcome} ⊆ outcome

def atLeastTwoHeads (outcome : set (Fin 3 × bool)) : Prop :=
  have heads : Nat := (count (λ b : bool, b == true) (outcome.toFinMap.range)),
  (heads ≥ 2)

def exactlyTwoHeads (outcome : set (Fin 3 × bool)) : Prop :=
  have heads : Nat := (count (λ b : bool, b == true) (outcome.toFinMap.range)),
  (heads == 2)

-- Main theorem statement
theorem mutually_exclusive_case_B (outcome: set (Fin 3 × bool)): 
  atMostOneHead outcome → ¬ atLeastTwoHeads outcome := by
  sorry

end mutually_exclusive_case_B_l285_285973


namespace find_number_l285_285044

theorem find_number : ∃ x : ℝ, (6 * ((x / 8 + 8) - 30) = 12) ∧ x = 192 :=
by sorry

end find_number_l285_285044


namespace problem1_solution_problem2_solution_l285_285778

noncomputable def problem1 (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) : Real := 
  A

noncomputable def problem2 (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) : Real :=
  a

theorem problem1_solution (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) :
  problem1 a b A B h1 = Real.pi / 4 :=
sorry

theorem problem2_solution (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) :
  problem2 a b c A area h1 h2 h3 = Real.sqrt 2 :=
sorry

end problem1_solution_problem2_solution_l285_285778


namespace acrobat_count_l285_285409

theorem acrobat_count (a e c : ℕ) (h1 : 2 * a + 4 * e + 2 * c = 88) (h2 : a + e + c = 30) : a = 2 :=
by
  sorry

end acrobat_count_l285_285409


namespace r_expansion_l285_285074

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end r_expansion_l285_285074


namespace madison_classes_l285_285805

/-- Madison's classes -/
def total_bell_rings : ℕ := 9

/-- Each class requires two bell rings (one to start, one to end) -/
def bell_rings_per_class : ℕ := 2

/-- The number of classes Madison has on Monday -/
theorem madison_classes (total_bell_rings bell_rings_per_class : ℕ) (last_class_start_only : total_bell_rings % bell_rings_per_class = 1) : 
  (total_bell_rings - 1) / bell_rings_per_class + 1 = 5 :=
by
  sorry

end madison_classes_l285_285805


namespace expand_expression_l285_285874

theorem expand_expression (a b : ℤ) : (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 :=
by sorry

end expand_expression_l285_285874


namespace pen_shorter_than_pencil_l285_285030

-- Definitions of the given conditions
def P (R : ℕ) := R + 3
def L : ℕ := 12
def total_length (R : ℕ) := R + P R + L

-- The theorem to be proven
theorem pen_shorter_than_pencil (R : ℕ) (h : total_length R = 29) : L - P R = 2 :=
by
  sorry

end pen_shorter_than_pencil_l285_285030


namespace base3_addition_l285_285988

theorem base3_addition : 
  Nat.of_digits 3 [2] + Nat.of_digits 3 [1, 2] + Nat.of_digits 3 [0, 1, 1] + Nat.of_digits 3 [2, 0, 2, 2] = Nat.of_digits 3 [0, 0, 0, 0, 1] :=
by
  sorry

end base3_addition_l285_285988


namespace cost_of_graveling_per_sq_meter_l285_285857

theorem cost_of_graveling_per_sq_meter
    (length_lawn : ℝ) (breadth_lawn : ℝ)
    (width_road : ℝ) (total_cost_gravel : ℝ)
    (length_road_area : ℝ) (breadth_road_area : ℝ) (intersection_area : ℝ)
    (total_graveled_area : ℝ) (cost_per_sq_meter : ℝ) :
    length_lawn = 55 →
    breadth_lawn = 35 →
    width_road = 4 →
    total_cost_gravel = 258 →
    length_road_area = length_lawn * width_road →
    intersection_area = width_road * width_road →
    breadth_road_area = breadth_lawn * width_road - intersection_area →
    total_graveled_area = length_road_area + breadth_road_area →
    cost_per_sq_meter = total_cost_gravel / total_graveled_area →
    cost_per_sq_meter = 0.75 :=
by
  intros
  sorry

end cost_of_graveling_per_sq_meter_l285_285857


namespace integral_eq_result_l285_285868

open Real

theorem integral_eq_result:
  ∫ (x: ℝ) in (π/4)..3, (3 * x - x^2) * sin(2 * x) =
    (π - 6 + 2 * cos(6) - 6 * sin(6)) / 8 := 
by
  sorry

end integral_eq_result_l285_285868


namespace seated_students_count_l285_285160

theorem seated_students_count :
  ∀ (S T standing_students total_attendees : ℕ),
    T = 30 →
    standing_students = 25 →
    total_attendees = 355 →
    total_attendees = S + T + standing_students →
    S = 300 :=
by
  intros S T standing_students total_attendees hT hStanding hTotalAttendees hEquation
  sorry

end seated_students_count_l285_285160


namespace prime_sequence_constant_l285_285730

open Nat

-- Define a predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the recurrence relation
def recurrence_relation (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n : ℕ, p (n + 2) = p (n + 1) + p n + k

-- Define the proof problem
theorem prime_sequence_constant (p : ℕ → ℕ) (k : ℤ) : 
  (∀ n, is_prime (p n)) →
  recurrence_relation p k →
  ∃ (q : ℕ), is_prime q ∧ (∀ n, p n = q) ∧ k = -q :=
by
  -- Sorry proof here
  sorry

end prime_sequence_constant_l285_285730


namespace lattice_point_exists_l285_285984

noncomputable def exists_distant_lattice_point : Prop :=
∃ (X Y : ℤ), ∀ (x y : ℤ), gcd x y = 1 → (X - x) ^ 2 + (Y - y) ^ 2 ≥ 1995 ^ 2

theorem lattice_point_exists : exists_distant_lattice_point :=
sorry

end lattice_point_exists_l285_285984


namespace complement_intersection_l285_285311

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {3, 4, 5}) (hN : N = {2, 3}) :
  (U \ N) ∩ M = {4, 5} := by
  sorry

end complement_intersection_l285_285311


namespace fraction_equivalence_l285_285846

theorem fraction_equivalence : (8 : ℝ) / (5 * 48) = 0.8 / (5 * 0.48) :=
  sorry

end fraction_equivalence_l285_285846


namespace carrots_total_l285_285524

theorem carrots_total (sandy_carrots : Nat) (sam_carrots : Nat) (h1 : sandy_carrots = 6) (h2 : sam_carrots = 3) :
  sandy_carrots + sam_carrots = 9 :=
by
  sorry

end carrots_total_l285_285524


namespace opposite_of_neg_three_l285_285377

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l285_285377


namespace distance_from_pole_eq_l285_285484

noncomputable def distance_from_pole_to_line (ρ θ : ℝ) (A B C x0 y0 : ℝ) : ℝ :=
  | A * x0 + B * y0 + C | / Real.sqrt (A^2 + B^2)

theorem distance_from_pole_eq (ρ θ : ℝ)
  (h : ρ * Real.cos (θ + Real.pi / 3) = Real.sqrt 3 / 2) :
  distance_from_pole_to_line ρ θ 1 (-Real.sqrt 3) (-Real.sqrt 3) 0 0 = Real.sqrt 3 / 2 :=
by
  sorry

end distance_from_pole_eq_l285_285484


namespace lasso_success_probability_l285_285138

-- Let p be the probability of successfully placing a lasso in a single throw
def p := 1 / 2

-- Let q be the probability of failure in a single throw
def q := 1 - p

-- Let n be the number of attempts
def n := 4

-- The probability of failing all n times
def probFailAll := q ^ n

-- The probability of succeeding at least once
def probSuccessAtLeastOnce := 1 - probFailAll

-- Theorem statement
theorem lasso_success_probability : probSuccessAtLeastOnce = 15 / 16 := by
  sorry

end lasso_success_probability_l285_285138


namespace test_score_based_on_preparation_l285_285112

theorem test_score_based_on_preparation :
  (grade_varies_directly_with_effective_hours : Prop) →
  (effective_hour_constant : ℝ) →
  (actual_hours_first_test : ℕ) →
  (actual_hours_second_test : ℕ) →
  (score_first_test : ℕ) →
  effective_hour_constant = 0.8 →
  actual_hours_first_test = 5 →
  score_first_test = 80 →
  actual_hours_second_test = 6 →
  grade_varies_directly_with_effective_hours →
  ∃ score_second_test : ℕ, score_second_test = 96 := by
  sorry

end test_score_based_on_preparation_l285_285112


namespace even_combinations_486_l285_285256

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l285_285256


namespace length_of_other_train_l285_285018

variable (L : ℝ)

theorem length_of_other_train
    (train1_length : ℝ := 260)
    (train1_speed_kmh : ℝ := 120)
    (train2_speed_kmh : ℝ := 80)
    (time_to_cross : ℝ := 9)
    (train1_speed : ℝ := train1_speed_kmh * 1000 / 3600)
    (train2_speed : ℝ := train2_speed_kmh * 1000 / 3600)
    (relative_speed : ℝ := train1_speed + train2_speed)
    (total_distance : ℝ := relative_speed * time_to_cross)
    (other_train_length : ℝ := total_distance - train1_length) :
    L = other_train_length := by
  sorry

end length_of_other_train_l285_285018


namespace identically_zero_on_interval_l285_285807

variable (f : ℝ → ℝ) (a b : ℝ)
variable (h_cont : ContinuousOn f (Set.Icc a b))
variable (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0)

theorem identically_zero_on_interval : ∀ x ∈ Set.Icc a b, f x = 0 := 
by 
  sorry

end identically_zero_on_interval_l285_285807


namespace James_total_tabs_l285_285086

theorem James_total_tabs (browsers windows tabs additional_tabs : ℕ) 
  (h_browsers : browsers = 4)
  (h_windows : windows = 5)
  (h_tabs : tabs = 12)
  (h_additional_tabs : additional_tabs = 3) : 
  browsers * (windows * (tabs + additional_tabs)) = 300 := by
  -- Proof goes here
  sorry

end James_total_tabs_l285_285086


namespace probability_xi_eq_2_l285_285235

noncomputable def prob_xi_eq_2 : ℚ := 
  let outcomes := [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)] in
  let favorable := outcomes.filter (λ o, (o.1 * o.2) = 2) in
  favorable.length / outcomes.length

theorem probability_xi_eq_2 (cards : Finset ℕ) (hk : cards = {0, 1, 2}) :
  prob_xi_eq_2 = 2 / 9 :=
by
  sorry

end probability_xi_eq_2_l285_285235


namespace britney_has_more_chickens_l285_285363

theorem britney_has_more_chickens :
  let susie_rhode_island_reds := 11
  let susie_golden_comets := 6
  let britney_rhode_island_reds := 2 * susie_rhode_island_reds
  let britney_golden_comets := susie_golden_comets / 2
  let susie_total := susie_rhode_island_reds + susie_golden_comets
  let britney_total := britney_rhode_island_reds + britney_golden_comets
  britney_total - susie_total = 8 := by
    sorry

end britney_has_more_chickens_l285_285363


namespace man_alone_days_l285_285264

-- Conditions from the problem
variables (M : ℕ) (h1 : (1 / (↑M : ℝ)) + (1 / 12) = 1 / 3)  -- Combined work rate condition

-- The proof statement we need to show
theorem man_alone_days : M = 4 :=
by {
  sorry
}

end man_alone_days_l285_285264


namespace surface_area_after_removing_corners_l285_285877

-- Define the dimensions of the cubes
def original_cube_side : ℝ := 4
def corner_cube_side : ℝ := 2

-- The surface area function for a cube with given side length
def surface_area (side : ℝ) : ℝ := 6 * side * side

theorem surface_area_after_removing_corners :
  surface_area original_cube_side = 96 :=
by
  sorry

end surface_area_after_removing_corners_l285_285877


namespace jugs_needed_to_provide_water_for_students_l285_285983

def jug_capacity : ℕ := 40
def students : ℕ := 200
def cups_per_student : ℕ := 10

def total_cups_needed := students * cups_per_student

theorem jugs_needed_to_provide_water_for_students :
  total_cups_needed / jug_capacity = 50 :=
by
  -- Proof goes here
  sorry

end jugs_needed_to_provide_water_for_students_l285_285983


namespace sin_theta_of_triangle_area_side_median_l285_285862

-- Defining the problem statement and required conditions
theorem sin_theta_of_triangle_area_side_median (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (hA : A = 30)
  (ha : a = 12)
  (hm : m = 8)
  (hTriangleArea : A = 1/2 * a * m * Real.sin θ) :
  Real.sin θ = 5 / 8 :=
by
  -- Proof omitted
  sorry

end sin_theta_of_triangle_area_side_median_l285_285862


namespace ratio_of_inscribed_squares_l285_285032

theorem ratio_of_inscribed_squares (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (hx : x = 60 / 17) (hy : y = 3) :
  x / y = 20 / 17 :=
by
  sorry

end ratio_of_inscribed_squares_l285_285032


namespace fraction_defined_range_l285_285321

theorem fraction_defined_range (x : ℝ) : 
  (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_defined_range_l285_285321


namespace max_vehicles_div_10_l285_285204

-- Each vehicle is 5 meters long
def vehicle_length : ℕ := 5

-- The speed rule condition
def speed_rule (m : ℕ) : ℕ := 20 * m

-- Maximum number of vehicles in one hour
def max_vehicles_per_hour (m : ℕ) : ℕ := 4000 * m / (m + 1)

-- N is the maximum whole number of vehicles
def N : ℕ := 4000

-- The target statement to prove: quotient when N is divided by 10
theorem max_vehicles_div_10 : N / 10 = 400 :=
by
  -- Definitions and given conditions go here
  sorry

end max_vehicles_div_10_l285_285204


namespace joe_lists_count_l285_285848

def num_options (n : ℕ) (k : ℕ) : ℕ := n ^ k

theorem joe_lists_count : num_options 12 3 = 1728 := by
  unfold num_options
  sorry

end joe_lists_count_l285_285848


namespace tank_capacity_l285_285033

theorem tank_capacity (V : ℝ) (initial_fraction final_fraction : ℝ) (added_water : ℝ)
  (h1 : initial_fraction = 1 / 4)
  (h2 : final_fraction = 3 / 4)
  (h3 : added_water = 208)
  (h4 : final_fraction - initial_fraction = 1 / 2)
  (h5 : (1 / 2) * V = added_water) :
  V = 416 :=
by
  -- Given: initial_fraction = 1/4, final_fraction = 3/4, added_water = 208
  -- Difference in fullness: 1/2
  -- Equation for volume: 1/2 * V = 208
  -- Hence, V = 416
  sorry

end tank_capacity_l285_285033


namespace problem1_problem2_l285_285997

-- Problem 1 Lean statement
theorem problem1 (x y : ℝ) (hx : x ≠ 1) (hx' : x ≠ -1) (hy : y ≠ 0) :
    (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) :=
sorry

-- Problem 2 Lean statement
theorem problem2 (m n : ℝ) (hm1 : m ≠ n) (hm2 : m ≠ -n) :
    m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 :=
sorry

end problem1_problem2_l285_285997


namespace solve_system_of_equations_l285_285279

theorem solve_system_of_equations (t : ℝ) (k u v : ℝ) 
  (h1 : t = 75) 
  (h2 : t = 5 / 9 * (k - 32)) 
  (h3 : u = t^2 + 2 * t + 5) 
  (h4 : v = log 3 (u - 9)) : 
  k = 167 ∧ u = 5780 ∧ v ≈ 7.882 :=
by {
  sorry
}

end solve_system_of_equations_l285_285279


namespace fraction_equivalent_l285_285402

theorem fraction_equivalent (x : ℝ) : x = 433 / 990 ↔ x = 0.4 + 37 / 990 * 10 ^ -2 :=
by
  sorry

end fraction_equivalent_l285_285402


namespace salary_increase_l285_285713

theorem salary_increase (new_salary increase : ℝ) (h_new : new_salary = 25000) (h_inc : increase = 5000) : 
  ((increase / (new_salary - increase)) * 100) = 25 :=
by
  -- We will write the proof to satisfy the requirement, but it is currently left out as per the instructions.
  sorry

end salary_increase_l285_285713


namespace total_worth_of_stock_l285_285985

theorem total_worth_of_stock (X : ℝ) (h1 : 0.1 * X * 1.2 - 0.9 * X * 0.95 = -400) : X = 16000 :=
by
  -- actual proof
  sorry

end total_worth_of_stock_l285_285985


namespace probability_at_least_one_white_ball_l285_285420

/-
  We define the conditions:
  - num_white: the number of white balls,
  - num_red: the number of red balls,
  - total_balls: the total number of balls,
  - num_drawn: the number of balls drawn.
-/
def num_white : ℕ := 5
def num_red : ℕ := 4
def total_balls : ℕ := num_white + num_red
def num_drawn : ℕ := 3

/-
  Given the conditions, we need to prove that the probability of drawing at least one white ball is 20/21.
-/
theorem probability_at_least_one_white_ball :
  (1 : ℚ) - (4 / 84) = 20 / 21 :=
by
  sorry

end probability_at_least_one_white_ball_l285_285420


namespace afternoon_sales_l285_285858

theorem afternoon_sales (x : ℕ) (H1 : 2 * x + x = 390) : 2 * x = 260 :=
by
  sorry

end afternoon_sales_l285_285858


namespace int_solutions_to_inequalities_l285_285637

theorem int_solutions_to_inequalities :
  { x : ℤ | -5 * x ≥ 3 * x + 15 } ∩
  { x : ℤ | -3 * x ≤ 9 } ∩
  { x : ℤ | 7 * x ≤ -14 } = { -3, -2 } :=
by {
  sorry
}

end int_solutions_to_inequalities_l285_285637


namespace old_barbell_cost_l285_285504

theorem old_barbell_cost (x : ℝ) (new_barbell_cost : ℝ) (h1 : new_barbell_cost = 1.30 * x) (h2 : new_barbell_cost = 325) : x = 250 :=
by
  sorry

end old_barbell_cost_l285_285504


namespace even_function_increasing_l285_285957

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * b * x + 1

theorem even_function_increasing (h_even : ∀ x : ℝ, f a b x = f a b (-x))
  (h_increasing : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → f a b x < f a b y) :
  f a b (a-2) < f a b (b+1) :=
sorry

end even_function_increasing_l285_285957


namespace fraction_of_marbles_taken_away_l285_285870

theorem fraction_of_marbles_taken_away (Chris_marbles Ryan_marbles remaining_marbles total_marbles taken_away_marbles : ℕ) 
    (hChris : Chris_marbles = 12) 
    (hRyan : Ryan_marbles = 28) 
    (hremaining : remaining_marbles = 20) 
    (htotal : total_marbles = Chris_marbles + Ryan_marbles) 
    (htaken_away : taken_away_marbles = total_marbles - remaining_marbles) : 
    (taken_away_marbles : ℚ) / total_marbles = 1 / 2 := 
by 
  sorry

end fraction_of_marbles_taken_away_l285_285870


namespace determine_time_l285_285854

variable (g a V_0 V S t : ℝ)

def velocity_eq : Prop := V = (g + a) * t + V_0
def displacement_eq : Prop := S = 1 / 2 * (g + a) * t^2 + V_0 * t

theorem determine_time (h1 : velocity_eq g a V_0 V t) (h2 : displacement_eq g a V_0 S t) :
  t = 2 * S / (V + V_0) := 
sorry

end determine_time_l285_285854


namespace find_fraction_B_minus_1_over_A_l285_285174

variable (A B : ℝ) (a_n S_n : ℕ → ℝ)
variable (h1 : ∀ n, a_n n + S_n n = A * (n ^ 2) + B * n + 1)
variable (h2 : A ≠ 0)

theorem find_fraction_B_minus_1_over_A : (B - 1) / A = 3 := by
  sorry

end find_fraction_B_minus_1_over_A_l285_285174


namespace find_side_difference_l285_285790

def triangle_ABC : Type := ℝ
def angle_B := 20
def angle_C := 40
def length_AD := 2

theorem find_side_difference (ABC : triangle_ABC) (B : ℝ) (C : ℝ) (AD : ℝ) (BC AB : ℝ) :
  B = angle_B → C = angle_C → AD = length_AD → BC - AB = 2 :=
by 
  sorry

end find_side_difference_l285_285790


namespace least_integer_gt_sqrt_700_l285_285697

theorem least_integer_gt_sqrt_700 : ∃ n : ℕ, (n - 1) < Real.sqrt 700 ∧ Real.sqrt 700 ≤ n ∧ n = 27 :=
by
  sorry

end least_integer_gt_sqrt_700_l285_285697


namespace ratio_of_speeds_l285_285135

variable (a b : ℝ)

theorem ratio_of_speeds (h1 : b = 1 / 60) (h2 : a + b = 1 / 12) : a / b = 4 := 
sorry

end ratio_of_speeds_l285_285135


namespace cars_meeting_time_l285_285608

def problem_statement (V_A V_B V_C V_D : ℝ) :=
  (V_A ≠ V_B) ∧ (V_A ≠ V_C) ∧ (V_A ≠ V_D) ∧
  (V_B ≠ V_C) ∧ (V_B ≠ V_D) ∧ (V_C ≠ V_D) ∧
  (V_A + V_C = V_B + V_D) ∧
  (53 * (V_A - V_B) / 46 = 7) ∧
  (53 * (V_D - V_C) / 46 = 7)

theorem cars_meeting_time (V_A V_B V_C V_D : ℝ) (h : problem_statement V_A V_B V_C V_D) : 
  ∃ t : ℝ, t = 53 := 
sorry

end cars_meeting_time_l285_285608


namespace line_intersects_plane_at_angle_l285_285304

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end line_intersects_plane_at_angle_l285_285304


namespace sum_of_six_selected_primes_is_even_l285_285361

noncomputable def prob_sum_even_when_selecting_six_primes : ℚ := 
  let first_twenty_primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
  let num_ways_to_choose_6_without_even_sum := Nat.choose 19 6
  let total_num_ways_to_choose_6 := Nat.choose 20 6
  num_ways_to_choose_6_without_even_sum / total_num_ways_to_choose_6

theorem sum_of_six_selected_primes_is_even : 
  prob_sum_even_when_selecting_six_primes = 354 / 505 := 
sorry

end sum_of_six_selected_primes_is_even_l285_285361


namespace distinct_special_sums_l285_285724

def is_special_fraction (a b : ℕ) : Prop := a + b = 18

def is_special_sum (n : ℤ) : Prop :=
  ∃ (a1 b1 a2 b2 : ℕ), is_special_fraction a1 b1 ∧ is_special_fraction a2 b2 ∧ 
  n = (a1 : ℤ) * (b2 : ℤ) * b1 + (a2 : ℤ) * (b1 : ℤ) / a1

theorem distinct_special_sums : 
  (∃ (sums : Finset ℤ), 
    (∀ n, n ∈ sums ↔ is_special_sum n) ∧ 
    sums.card = 7) :=
sorry

end distinct_special_sums_l285_285724


namespace counterexample_to_prime_statement_l285_285040

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem counterexample_to_prime_statement 
  (n : ℕ) 
  (h_n_composite : is_composite n) 
  (h_n_minus_3_not_prime : ¬ is_prime (n - 3)) : 
  n = 18 ∨ n = 24 :=
by 
  sorry

end counterexample_to_prime_statement_l285_285040


namespace ellipse_k_range_ellipse_k_eccentricity_l285_285899

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) ↔ (1 < k ∧ k < 5 ∨ 5 < k ∧ k < 9) := 
sorry

theorem ellipse_k_eccentricity (k : ℝ) (h : ∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) : 
  eccentricity = Real.sqrt (6/7) → (k = 2 ∨ k = 8) := 
sorry

end ellipse_k_range_ellipse_k_eccentricity_l285_285899


namespace sum_3000_l285_285962

-- Definitions based on conditions
def geo_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions
def sum_1000 : Prop := geo_seq_sum a r 1000 = 500
def sum_2000 : Prop := geo_seq_sum a r 2000 = 950

-- The statement to prove
theorem sum_3000 (h1 : sum_1000 a r) (h2 : sum_2000 a r) :
  geo_seq_sum a r 3000 = 1355 :=
sorry

end sum_3000_l285_285962


namespace probability_exactly_one_die_divisible_by_3_l285_285559

noncomputable def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def prob_exactly_one_divisible_by_3 : ℚ :=
  let outcomes := {1, 2, 3, 4, 5, 6}
  let total_outcomes := (outcomes × outcomes × outcomes).card
  let favorable_outcomes := (outcomes × outcomes × outcomes).filter (
    λ x : ℕ × ℕ × ℕ, 
    is_divisible_by_3 (x.1) ∧ ¬is_divisible_by_3 (x.2) ∧ ¬is_divisible_by_3 (x.3) ∨
    ¬is_divisible_by_3 (x.1) ∧ is_divisible_by_3 (x.2) ∧ ¬is_divisible_by_3 (x.3) ∨
    ¬is_divisible_by_3 (x.1) ∧ ¬is_divisible_by_3 (x.2) ∧ is_divisible_by_3 (x.3)
  ).card
  favorable_outcomes / total_outcomes

theorem probability_exactly_one_die_divisible_by_3:
  prob_exactly_one_divisible_by_3 = 4 / 9 := 
by
  sorry

end probability_exactly_one_die_divisible_by_3_l285_285559


namespace question_l285_285980

def N : ℕ := 100101102 -- N should be defined properly but is simplified here for illustration.

theorem question (k : ℕ) (h : N = 100101102502499500) : (3^3 ∣ N) ∧ ¬(3^4 ∣ N) :=
sorry

end question_l285_285980


namespace min_value_of_expr_min_value_achieved_final_statement_l285_285926

theorem min_value_of_expr (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  1 ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem min_value_achieved (x y z : ℝ) (h1 : x = 1) (h2 : y = 1) (h3 : z = 1) :
  1 = (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem final_statement (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  ∃ (x y z : ℝ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x + y + z = 3) ∧ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1) :=
by
  sorry

end min_value_of_expr_min_value_achieved_final_statement_l285_285926


namespace arithmetic_sequence_sum_l285_285330

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ)

def S₁₀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ) : ℕ :=
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀

theorem arithmetic_sequence_sum (h : S₁₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ = 120) :
  a₁ + a₁₀ = 24 :=
by
  sorry

end arithmetic_sequence_sum_l285_285330


namespace abs_diff_l285_285612

theorem abs_diff (a b : ℝ) (h_ab : a < b) (h_a : abs a = 6) (h_b : abs b = 3) :
  a - b = -9 ∨ a - b = 9 :=
by
  sorry

end abs_diff_l285_285612


namespace number_of_routes_from_A_to_B_l285_285726

-- Define the grid dimensions
def grid_rows : ℕ := 3
def grid_columns : ℕ := 2

-- Define the total number of steps needed to travel from A to B
def total_steps : ℕ := grid_rows + grid_columns

-- Define the number of right moves (R) and down moves (D)
def right_moves : ℕ := grid_rows
def down_moves : ℕ := grid_columns

-- Calculate the number of different routes using combination formula
def number_of_routes : ℕ := Nat.choose total_steps right_moves

-- The main statement to be proven
theorem number_of_routes_from_A_to_B : number_of_routes = 10 :=
by sorry

end number_of_routes_from_A_to_B_l285_285726


namespace linear_eq_a_l285_285649

theorem linear_eq_a (a : ℝ) (x y : ℝ) (h1 : (a + 1) ≠ 0) (h2 : |a| = 1) : a = 1 :=
by
  sorry

end linear_eq_a_l285_285649


namespace find_x_l285_285313

theorem find_x (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (3, x)) (h : (a.fst * b.fst + a.snd * b.snd) = 3) : x = 3 :=
by
  sorry

end find_x_l285_285313


namespace divisor_of_number_l285_285708

theorem divisor_of_number : 
  ∃ D, 
    let x := 75 
    let R' := 7 
    let Q := R' + 8 
    x = D * Q + 0 :=
by
  sorry

end divisor_of_number_l285_285708


namespace least_possible_value_of_m_plus_n_l285_285924

theorem least_possible_value_of_m_plus_n 
(m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) 
(hgcd : Nat.gcd (m + n) 210 = 1) 
(hdiv : ∃ k, m^m = k * n^n)
(hnotdiv : ¬ ∃ k, m = k * n) : 
  m + n = 407 := 
sorry

end least_possible_value_of_m_plus_n_l285_285924


namespace find_x_if_vectors_parallel_l285_285623

theorem find_x_if_vectors_parallel (x : ℝ)
  (a : ℝ × ℝ := (x - 1, 2))
  (b : ℝ × ℝ := (2, 1)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → x = 5 :=
by sorry

end find_x_if_vectors_parallel_l285_285623


namespace sum_of_numbers_l285_285699

theorem sum_of_numbers :
  2.12 + 0.004 + 0.345 = 2.469 :=
sorry

end sum_of_numbers_l285_285699


namespace number_of_even_results_l285_285260

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l285_285260


namespace orange_ribbons_count_l285_285653

variable (total_ribbons : ℕ)
variable (orange_ribbons : ℚ)

-- Definitions of the given conditions
def yellow_fraction := (1 : ℚ) / 4
def purple_fraction := (1 : ℚ) / 3
def orange_fraction := (1 : ℚ) / 6
def black_ribbons := 40
def black_fraction := (1 : ℚ) / 4

-- Using the given and derived conditions
theorem orange_ribbons_count
  (hy : yellow_fraction = 1 / 4)
  (hp : purple_fraction = 1 / 3)
  (ho : orange_fraction = 1 / 6)
  (hb : black_ribbons = 40)
  (hbf : black_fraction = 1 / 4)
  (total_eq : total_ribbons = black_ribbons * 4) :
  orange_ribbons = total_ribbons * orange_fraction := by
  -- Proof omitted
  sorry

end orange_ribbons_count_l285_285653


namespace intersecting_lines_l285_285131

theorem intersecting_lines (n c : ℝ) 
  (h1 : (15 : ℝ) = n * 5 + 5)
  (h2 : (15 : ℝ) = 4 * 5 + c) : 
  c + n = -3 := 
by
  sorry

end intersecting_lines_l285_285131


namespace smallest_pos_int_b_for_factorization_l285_285051

theorem smallest_pos_int_b_for_factorization :
  ∃ b : ℤ, 0 < b ∧ ∀ (x : ℤ), ∃ r s : ℤ, r * s = 4032 ∧ r + s = b ∧ x^2 + b * x + 4032 = (x + r) * (x + s) ∧
    (∀ b' : ℤ, 0 < b' → b' ≠ b → ∃ rr ss : ℤ, rr * ss = 4032 ∧ rr + ss = b' ∧ x^2 + b' * x + 4032 = (x + rr) * (x + ss) → b < b') := 
sorry

end smallest_pos_int_b_for_factorization_l285_285051


namespace find_m_value_l285_285650

theorem find_m_value (m : ℤ) (h : (∀ x : ℤ, (x-5)*(x+7) = x^2 - mx - 35)) : m = -2 :=
by sorry

end find_m_value_l285_285650


namespace sum_of_coefficients_l285_285469

noncomputable def polynomial_eq (x : ℝ) : ℝ := 1 + x^5
noncomputable def linear_combination (a0 a1 a2 a3 a4 a5 x : ℝ) : ℝ :=
  a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5

theorem sum_of_coefficients (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial_eq 1 = linear_combination a0 a1 a2 a3 a4 a5 1 →
  polynomial_eq 2 = linear_combination a0 a1 a2 a3 a4 a5 2 →
  a0 = 2 →
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  intros h1 h2 h3
  sorry

end sum_of_coefficients_l285_285469


namespace mobius_round_trip_time_l285_285096

theorem mobius_round_trip_time :
  (let speed_without_load := 13
       miles_per_hour := 1
       speed_with_load := 11
       distance := 143
       rest_time_each_half := 1
       travel_time_with_load := distance / speed_with_load
       travel_time_without_load := distance / speed_without_load
       total_rest_time := rest_time_each_half * 2 in
  travel_time_with_load + travel_time_without_load + total_rest_time = 26) :=
by sorry

end mobius_round_trip_time_l285_285096


namespace point_in_third_quadrant_l285_285647

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (-b < 0 ∧ a - 3 < 0) :=
by sorry

end point_in_third_quadrant_l285_285647


namespace sphere_intersection_circle_radius_l285_285149

theorem sphere_intersection_circle_radius
  (x1 y1 z1: ℝ) (x2 y2 z2: ℝ) (r1 r2: ℝ)
  (hyp1: x1 = 3) (hyp2: y1 = 5) (hyp3: z1 = 0) 
  (hyp4: r1 = 2) 
  (hyp5: x2 = 0) (hyp6: y2 = 5) (hyp7: z2 = -8) :
  r2 = Real.sqrt 59 := 
by
  sorry

end sphere_intersection_circle_radius_l285_285149


namespace mr_william_land_percentage_l285_285164

/--
Given:
1. Farm tax is levied on 90% of the cultivated land.
2. The tax department collected a total of $3840 through the farm tax from the village.
3. Mr. William paid $480 as farm tax.

Prove: The percentage of total land of Mr. William over the total taxable land of the village is 12.5%.
-/
theorem mr_william_land_percentage (T W : ℝ) 
  (h1 : 0.9 * W = 480) 
  (h2 : 0.9 * T = 3840) : 
  (W / T) * 100 = 12.5 :=
by
  sorry

end mr_william_land_percentage_l285_285164


namespace age_of_teacher_l285_285818

theorem age_of_teacher (S : ℕ) (T : Real) (n : ℕ) (average_student_age : Real) (new_average_age : Real) : 
  average_student_age = 14 → 
  new_average_age = 14.66 → 
  n = 45 → 
  S = average_student_age * n → 
  T = 44.7 :=
by
  sorry

end age_of_teacher_l285_285818


namespace vegetable_difference_is_30_l285_285575

def initial_tomatoes : Int := 17
def initial_carrots : Int := 13
def initial_cucumbers : Int := 8
def initial_bell_peppers : Int := 15
def initial_radishes : Int := 0

def picked_tomatoes : Int := 5
def picked_carrots : Int := 6
def picked_cucumbers : Int := 3
def picked_bell_peppers : Int := 8

def given_neighbor1_tomatoes : Int := 3
def given_neighbor1_carrots : Int := 2

def exchanged_neighbor2_tomatoes : Int := 2
def exchanged_neighbor2_cucumbers : Int := 3
def exchanged_neighbor2_radishes : Int := 5

def given_neighbor3_bell_peppers : Int := 3

noncomputable def initial_total := 
  initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers + initial_radishes

noncomputable def remaining_after_picking :=
  (initial_tomatoes - picked_tomatoes) +
  (initial_carrots - picked_carrots) +
  (initial_cucumbers - picked_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers)

noncomputable def remaining_after_exchanges :=
  ((initial_tomatoes - picked_tomatoes - given_neighbor1_tomatoes - exchanged_neighbor2_tomatoes) +
  (initial_carrots - picked_carrots - given_neighbor1_carrots) +
  (initial_cucumbers - picked_cucumbers - exchanged_neighbor2_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers - given_neighbor3_bell_peppers) +
  exchanged_neighbor2_radishes)

noncomputable def remaining_total := remaining_after_exchanges

noncomputable def total_difference := initial_total - remaining_total

theorem vegetable_difference_is_30 : total_difference = 30 := by
  sorry

end vegetable_difference_is_30_l285_285575


namespace train_pass_bridge_time_l285_285021

noncomputable def length_of_train : ℝ := 485
noncomputable def length_of_bridge : ℝ := 140
noncomputable def speed_of_train_kmph : ℝ := 45 
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

theorem train_pass_bridge_time :
  (length_of_train + length_of_bridge) / speed_of_train_mps = 50 :=
by
  sorry

end train_pass_bridge_time_l285_285021


namespace find_value_of_expression_l285_285058

open Polynomial

theorem find_value_of_expression
  {α β : ℝ}
  (h1 : α^2 - 5 * α + 6 = 0)
  (h2 : β^2 - 5 * β + 6 = 0) :
  3 * α^3 + 10 * β^4 = 2305 :=
by
  sorry

end find_value_of_expression_l285_285058


namespace arabella_first_step_time_l285_285864

def time_first_step (x : ℝ) : Prop :=
  let time_second_step := x / 2
  let time_third_step := x + x / 2
  (x + time_second_step + time_third_step = 90)

theorem arabella_first_step_time (x : ℝ) (h : time_first_step x) : x = 30 :=
by
  sorry

end arabella_first_step_time_l285_285864


namespace even_function_a_value_l285_285185

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + (a - 2) * x + a^2 - a - 2 = (a + 1) * x^2 - (a - 2) * x + a^2 - a - 2) → a = 2 := 
by sorry

end even_function_a_value_l285_285185


namespace white_tiles_count_l285_285341

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l285_285341


namespace find_xyz_l285_285167

theorem find_xyz (x y z : ℝ) 
  (h1: 3 * x - y + z = 8)
  (h2: x + 3 * y - z = 2) 
  (h3: x - y + 3 * z = 6) :
  x = 1 ∧ y = 3 ∧ z = 8 := by
  sorry

end find_xyz_l285_285167


namespace opposite_of_neg3_l285_285382

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l285_285382


namespace power_of_p_in_product_l285_285909

theorem power_of_p_in_product (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) 
  (h : (x + 1) * 6 = 30) : x = 4 := 
by sorry

end power_of_p_in_product_l285_285909


namespace range_independent_variable_l285_285542

def domain_of_function (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem range_independent_variable (x : ℝ) :
  domain_of_function x ↔ x ≥ -1 ∧ x ≠ 0 :=
by
  sorry

end range_independent_variable_l285_285542


namespace inequality_solution_l285_285768

open Set

def f (x : ℝ) : ℝ := |x| + x^2 + 2

def solution_set : Set ℝ := { x | x < -2 ∨ x > 4 / 3 }

theorem inequality_solution :
  { x : ℝ | f (2 * x - 1) > f (3 - x) } = solution_set := by
  sorry

end inequality_solution_l285_285768


namespace not_inequality_neg_l285_285317

theorem not_inequality_neg (x y : ℝ) (h : x > y) : ¬ (-x > -y) :=
by {
  sorry
}

end not_inequality_neg_l285_285317


namespace sum_of_roots_l285_285085

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end sum_of_roots_l285_285085


namespace range_of_k_l285_285063

noncomputable def f (k : ℝ) (x : ℝ) := (Real.exp x) / (x^2) + 2 * k * Real.log x - k * x

theorem range_of_k (k : ℝ) (h₁ : ∀ x > 0, (deriv (f k) x = 0) → x = 2) : k < Real.exp 2 / 4 :=
by
  sorry

end range_of_k_l285_285063


namespace percentage_increase_bears_with_assistant_l285_285914

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end percentage_increase_bears_with_assistant_l285_285914


namespace certain_number_division_l285_285842

theorem certain_number_division (x : ℝ) (h : x / 3 + x + 3 = 63) : x = 45 :=
by
  sorry

end certain_number_division_l285_285842


namespace divides_seven_l285_285522

theorem divides_seven (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : Nat.gcd x y = 1) (h5 : x^2 + y^2 = z^4) : 7 ∣ x * y :=
by
  sorry

end divides_seven_l285_285522


namespace cups_of_flour_put_in_l285_285351

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end cups_of_flour_put_in_l285_285351


namespace tangents_of_convex_quad_l285_285496

theorem tangents_of_convex_quad (
  α β γ δ : ℝ
) (m : ℝ) (h₀ : α + β + γ + δ = 2 * Real.pi) (h₁ : 0 < α ∧ α < Real.pi) (h₂ : 0 < β ∧ β < Real.pi) 
  (h₃ : 0 < γ ∧ γ < Real.pi) (h₄ : 0 < δ ∧ δ < Real.pi) (t1 : Real.tan α = m) :
  ¬ (Real.tan β = m ∧ Real.tan γ = m ∧ Real.tan δ = m) :=
sorry

end tangents_of_convex_quad_l285_285496


namespace infinite_colored_points_l285_285457

theorem infinite_colored_points
(P : ℤ → Prop) (red blue : ℤ → Prop)
(h_color : ∀ n : ℤ, (red n ∨ blue n))
(h_red_blue_partition : ∀ n : ℤ, ¬(red n ∧ blue n)) :
  ∃ (C : ℤ → Prop) (k : ℕ), (C = red ∨ C = blue) ∧ ∀ n : ℕ, ∃ m : ℤ, C m ∧ (m % n) = 0 :=
by
  sorry

end infinite_colored_points_l285_285457


namespace max_distance_from_point_on_circle_to_line_l285_285795

noncomputable def center_of_circle : ℝ × ℝ := (5, 3)
noncomputable def radius_of_circle : ℝ := 3
noncomputable def line_eqn (x y : ℝ) : ℝ := 3 * x + 4 * y - 2
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ := (|a * px + b * py + c|) / (Real.sqrt (a * a + b * b))

theorem max_distance_from_point_on_circle_to_line :
  let Cx := (center_of_circle.1)
  let Cy := (center_of_circle.2)
  let d := distance_point_to_line Cx Cy 3 4 (-2)
  d + radius_of_circle = 8 := by
  sorry

end max_distance_from_point_on_circle_to_line_l285_285795


namespace plane_equation_correct_l285_285977

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 1, y := -1, z := 8 }
def B : Point3D := { x := -4, y := -3, z := 10 }
def C : Point3D := { x := -1, y := -1, z := 7 }

-- Define the vector BC
def vecBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

-- Define the equation of the plane
def planeEquation (P : Point3D) (normal : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  (normal.x, normal.y, normal.z, -(normal.x * P.x + normal.y * P.y + normal.z * P.z))

-- Calculate the equation of the plane passing through A and perpendicular to vector BC
def planeThroughAperpToBC : ℝ × ℝ × ℝ × ℝ :=
  let normal := vecBC B C
  planeEquation A normal

-- The expected result
def expectedPlaneEquation : ℝ × ℝ × ℝ × ℝ := (3, 2, -3, 23)

-- The theorem to be proved
theorem plane_equation_correct : planeThroughAperpToBC = expectedPlaneEquation := by
  sorry

end plane_equation_correct_l285_285977


namespace walking_rate_on_escalator_l285_285152

theorem walking_rate_on_escalator 
  (escalator_speed person_time : ℝ) 
  (escalator_length : ℝ) 
  (h1 : escalator_speed = 12) 
  (h2 : person_time = 15) 
  (h3 : escalator_length = 210) 
  : (∃ v : ℝ, escalator_length = (v + escalator_speed) * person_time ∧ v = 2) :=
by
  use 2
  rw [h1, h2, h3]
  sorry

end walking_rate_on_escalator_l285_285152


namespace maximum_value_of_f_minimum_value_of_f_l285_285616

-- Define the function f
def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

-- Define the condition
def condition (x y : ℝ) : Prop := x^2 + y^2 ≤ 5

-- State the maximum value theorem
theorem maximum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 + 6 * Real.sqrt 5 := sorry

-- State the minimum value theorem
theorem minimum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 - 3 * Real.sqrt 10 := sorry

end maximum_value_of_f_minimum_value_of_f_l285_285616


namespace minimum_shift_value_l285_285776

theorem minimum_shift_value
    (m : ℝ) 
    (h1 : m > 0) :
    (∃ (k : ℤ), m = k * π - π / 3 ∧ k > 0) → (m = (2 * π) / 3) :=
sorry

end minimum_shift_value_l285_285776


namespace stable_set_even_subset_count_l285_285793

open Finset

-- Definitions
def is_stable (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ ⦃x y⦄, (x, y) ∈ S → ∀ x' y', x' ≤ x → y' ≤ y → (x', y') ∈ S

-- Main statement
theorem stable_set_even_subset_count (S : Finset (ℕ × ℕ)) (hS : is_stable S):
  (∃ E O : ℕ, E ≥ O ∧ E + O = 2 ^ (S.card)) :=
  sorry

end stable_set_even_subset_count_l285_285793


namespace square_of_neg_3b_l285_285451

theorem square_of_neg_3b (b : ℝ) : (-3 * b)^2 = 9 * b^2 :=
by sorry

end square_of_neg_3b_l285_285451


namespace fraction_calculation_l285_285289

theorem fraction_calculation :
  ( (3 / 7 + 5 / 8 + 1 / 3) / (5 / 12 + 2 / 9) = 2097 / 966 ) :=
by
  sorry

end fraction_calculation_l285_285289


namespace r_expansion_l285_285075

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end r_expansion_l285_285075


namespace z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l285_285619

section
variable (m : ℝ)
def z : ℂ := (m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) * Complex.I

theorem z_is_real_iff_m_values :
  (z m).im = 0 ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_in_third_quadrant_iff_m_interval :
  (z m).re < 0 ∧ (z m).im < 0 ↔ m ∈ Set.Ioo (-3) (-2) :=
by sorry
end

end z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l285_285619


namespace find_x_l285_285140

variable (x : ℤ)

-- Define the conditions based on the problem
def adjacent_sum_condition := 
  (x + 15) + (x + 8) + (x - 7) = x

-- State the goal, which is to prove x = -8
theorem find_x : x = -8 :=
by
  have h : adjacent_sum_condition x := sorry
  sorry

end find_x_l285_285140


namespace polygon_properties_l285_285024

-- Assume n is the number of sides of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_of_exterior_angles : ℝ := 360

-- Given the condition
def given_condition (n : ℕ) : Prop := sum_of_interior_angles n = 5 * sum_of_exterior_angles

theorem polygon_properties (n : ℕ) (h1 : given_condition n) :
  n = 12 ∧ (n * (n - 3)) / 2 = 54 :=
by
  sorry

end polygon_properties_l285_285024


namespace area_of_square_on_AD_l285_285828

theorem area_of_square_on_AD :
  ∃ (AB BC CD AD : ℝ),
    (∃ AB_sq BC_sq CD_sq AD_sq : ℝ,
      AB_sq = 25 ∧ BC_sq = 49 ∧ CD_sq = 64 ∧ 
      AB = Real.sqrt AB_sq ∧ BC = Real.sqrt BC_sq ∧ CD = Real.sqrt CD_sq ∧
      AD_sq = AB^2 + BC^2 + CD^2 ∧ AD = Real.sqrt AD_sq ∧ AD_sq = 138
    ) :=
by
  sorry

end area_of_square_on_AD_l285_285828


namespace can_divide_2007_triangles_can_divide_2008_triangles_l285_285037

theorem can_divide_2007_triangles :
  ∃ k : ℕ, 2007 = 9 + 3 * k :=
by
  sorry

theorem can_divide_2008_triangles :
  ∃ m : ℕ, 2008 = 4 + 3 * m :=
by
  sorry

end can_divide_2007_triangles_can_divide_2008_triangles_l285_285037


namespace men_days_proof_l285_285104

noncomputable def time_to_complete (m d e r : ℕ) : ℕ :=
  (m * d) / (e * (m + r))

theorem men_days_proof (m d e r t : ℕ) (h1 : d = (m * d) / (m * e))
  (h2 : t = (m * d) / (e * (m + r))) :
  t = (m * d) / (e * (m + r)) :=
by
  -- The proof would go here
  sorry

end men_days_proof_l285_285104


namespace sufficiency_condition_a_gt_b_sq_gt_sq_l285_285773

theorem sufficiency_condition_a_gt_b_sq_gt_sq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b → a^2 > b^2) ∧ (∀ (h : a^2 > b^2), ∃ c > 0, ∃ d > 0, c^2 > d^2 ∧ ¬(c > d)) :=
by
  sorry

end sufficiency_condition_a_gt_b_sq_gt_sq_l285_285773


namespace smallest_constant_c_l285_285157

def satisfies_conditions (f : ℝ → ℝ) :=
  ∀ ⦃x : ℝ⦄, (0 ≤ x ∧ x ≤ 1) → (f x ≥ 0 ∧ (x = 1 → f 1 = 1) ∧
  (∀ y, 0 ≤ y → y ≤ 1 → x + y ≤ 1 → f x + f y ≤ f (x + y)))

theorem smallest_constant_c :
  ∀ {f : ℝ → ℝ},
  satisfies_conditions f →
  ∃ c : ℝ, (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ c * x) ∧
  (∀ c', c' < 2 → ∃ x, 0 ≤ x → x ≤ 1 ∧ f x > c' * x) :=
by sorry

end smallest_constant_c_l285_285157


namespace sin_360_eq_0_l285_285154

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end sin_360_eq_0_l285_285154


namespace range_of_a_given_quadratic_condition_l285_285487

theorem range_of_a_given_quadratic_condition:
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 3 * a * x + 9 ≥ 0) → (-2 ≤ a ∧ a ≤ 2)) :=
by
  sorry

end range_of_a_given_quadratic_condition_l285_285487


namespace probability_all_quitters_same_tribe_l285_285543

theorem probability_all_quitters_same_tribe :
  ∀ (people : Finset ℕ) (tribe1 tribe2 : Finset ℕ) (choose : ℕ → ℕ → ℕ) (prob : ℚ),
  people.card = 20 →
  tribe1.card = 10 →
  tribe2.card = 10 →
  tribe1 ∪ tribe2 = people →
  tribe1 ∩ tribe2 = ∅ →
  choose 20 3 = 1140 →
  choose 10 3 = 120 →
  prob = (2 * choose 10 3) / choose 20 3 →
  prob = 20 / 95 :=
by
  intro people tribe1 tribe2 choose prob
  intros hp20 ht1 ht2 hu hi hchoose20 hchoose10 hprob
  sorry

end probability_all_quitters_same_tribe_l285_285543


namespace working_together_time_l285_285248

/-- A is 30% more efficient than B,
and A alone can complete the job in 23 days.
Prove that A and B working together take approximately 13 days to complete the job. -/
theorem working_together_time (Ea Eb : ℝ) (T : ℝ) (h1 : Ea = 1.30 * Eb) 
  (h2 : 1 / 23 = Ea) : T = 13 :=
sorry

end working_together_time_l285_285248


namespace compound_interest_correct_l285_285591

noncomputable def compoundInterest (P: ℝ) (r: ℝ) (n: ℝ) (t: ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_correct :
  compoundInterest 5000 0.04 1 3 - 5000 = 624.32 :=
by
  sorry

end compound_interest_correct_l285_285591


namespace projection_problem_l285_285539

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

theorem projection_problem :
  let v : ℝ × ℝ := (1, -1/2)
  let sum_v := (v.1 + 1, v.2 + 1)
  projection (3, 5) sum_v = (104/17, 26/17) :=
by
  sorry

end projection_problem_l285_285539


namespace positive_integer_solutions_l285_285883

theorem positive_integer_solutions
  (m n k : ℕ)
  (hm : 0 < m) (hn : 0 < n) (hk : 0 < k) :
  3 * m + 4 * n = 5 * k ↔ (m = 1 ∧ n = 2 ∧ k = 2) := 
by
  sorry

end positive_integer_solutions_l285_285883


namespace remainder_divisible_by_4_l285_285686

theorem remainder_divisible_by_4 (z : ℕ) (h : z % 4 = 0) : ((z * (2 + 4 + z) + 3) % 2) = 1 :=
by
  sorry

end remainder_divisible_by_4_l285_285686


namespace tan_a_div_tan_b_l285_285798

variable {a b : ℝ}

-- Conditions
axiom sin_a_plus_b : Real.sin (a + b) = 1/2
axiom sin_a_minus_b : Real.sin (a - b) = 1/4

-- Proof statement (without the explicit proof)
theorem tan_a_div_tan_b : (Real.tan a) / (Real.tan b) = 3 := by
  sorry

end tan_a_div_tan_b_l285_285798


namespace trigonometric_identity_proof_l285_285465

noncomputable def trigonometric_expression : ℝ := 
  (Real.sin (15 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) 
  + Real.cos (165 * Real.pi / 180) * Real.cos (115 * Real.pi / 180)) /
  (Real.sin (35 * Real.pi / 180) * Real.cos (5 * Real.pi / 180) 
  + Real.cos (145 * Real.pi / 180) * Real.cos (85 * Real.pi / 180))

theorem trigonometric_identity_proof : trigonometric_expression = 1 :=
by
  sorry

end trigonometric_identity_proof_l285_285465


namespace fraction_equivalent_l285_285404

theorem fraction_equivalent (x : ℝ) : x = 433 / 990 ↔ x = 0.4 + 37 / 990 * 10 ^ -2 :=
by
  sorry

end fraction_equivalent_l285_285404


namespace reflected_ray_equation_l285_285266

-- Definitions for the given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Problem statement: proving equation of the reflected ray
theorem reflected_ray_equation : 
  ∀ x y : ℝ, incident_line x = y ∧ reflection_line x = y → x - 2*y - 1 = 0 :=
by
  sorry

end reflected_ray_equation_l285_285266


namespace max_area_rectangle_perimeter_156_l285_285916

theorem max_area_rectangle_perimeter_156 (x y : ℕ) 
  (h : 2 * (x + y) = 156) : ∃x y, x * y = 1521 :=
by
  sorry

end max_area_rectangle_perimeter_156_l285_285916


namespace opposite_of_neg_three_l285_285376

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l285_285376


namespace gcd_143_144_l285_285887

def a : ℕ := 143
def b : ℕ := 144

theorem gcd_143_144 : Nat.gcd a b = 1 :=
by
  sorry

end gcd_143_144_l285_285887


namespace linear_eq_solution_l285_285078

theorem linear_eq_solution (m : ℤ) (x : ℝ) (h1 : |m| = 1) (h2 : 1 - m ≠ 0) : x = -1/2 :=
by
  sorry

end linear_eq_solution_l285_285078


namespace chain_of_inequalities_l285_285359

theorem chain_of_inequalities (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  9 / (a + b + c) ≤ (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ∧ 
  (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ≤ (1 / a + 1 / b + 1 / c) := 
by 
  sorry

end chain_of_inequalities_l285_285359


namespace fraction_less_than_40_percent_l285_285142

theorem fraction_less_than_40_percent (x : ℝ) (h1 : x * 180 = 48) (h2 : x < 0.4) : x = 4 / 15 :=
by
  sorry

end fraction_less_than_40_percent_l285_285142


namespace find_polynomial_l285_285882

theorem find_polynomial (P : ℝ → ℝ) (h_poly : ∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) :
  ∃ r s : ℝ, ∀ x : ℝ, P x = r * x^4 + s * x^2 :=
sorry

end find_polynomial_l285_285882


namespace harold_car_payment_l285_285314

variables (C : ℝ)

noncomputable def harold_income : ℝ := 2500
noncomputable def rent : ℝ := 700
noncomputable def groceries : ℝ := 50
noncomputable def remaining_after_retirement : ℝ := 1300

-- Harold's utility cost is half his car payment
noncomputable def utilities (C : ℝ) : ℝ := C / 2

-- Harold's total expenses.
noncomputable def total_expenses (C : ℝ) : ℝ := rent + C + utilities C + groceries

-- Proving that Harold’s car payment \(C\) can be calculated with the remaining money
theorem harold_car_payment : (2500 - total_expenses C = 1300) → (C = 300) :=
by 
  sorry

end harold_car_payment_l285_285314


namespace find_number_l285_285418

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end find_number_l285_285418


namespace probability_of_rectangle_area_greater_than_32_l285_285353

-- Definitions representing the problem conditions
def segment_length : ℝ := 12
def point_C (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ segment_length
def rectangle_area (x : ℝ) : ℝ := x * (segment_length - x)

-- The probability we need to prove. 
noncomputable def desired_probability : ℝ := 1 / 3

theorem probability_of_rectangle_area_greater_than_32 :
  (∀ x, point_C x → rectangle_area x > 32) → (desired_probability = 1 / 3) :=
by
  sorry

end probability_of_rectangle_area_greater_than_32_l285_285353


namespace triangle_area_l285_285667

-- Define the points P, Q, R and the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def PQR_right_triangle (P Q R : Point) : Prop := 
  (P.x - R.x)^2 + (P.y - R.y)^2 = 24^2 ∧  -- Length PR
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 73^2 ∧  -- Length RQ
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = 75^2 ∧  -- Hypotenuse PQ
  (P.y = 3 * P.x + 4) ∧                   -- Median through P
  (Q.y = -Q.x + 5)                        -- Median through Q


noncomputable def area (P Q R : Point) : ℝ := 
  0.5 * abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y))

theorem triangle_area (P Q R : Point) (h : PQR_right_triangle P Q R) : 
  area P Q R = 876 :=
sorry

end triangle_area_l285_285667


namespace total_cans_from_256_l285_285607

-- Define the recursive function to compute the number of new cans produced.
def total_new_cans (n : ℕ) : ℕ :=
  if n < 4 then 0
  else
    let rec_cans := total_new_cans (n / 4)
    (n / 4) + rec_cans

-- Theorem stating the total number of new cans that can be made from 256 initial cans.
theorem total_cans_from_256 : total_new_cans 256 = 85 := by
  sorry

end total_cans_from_256_l285_285607


namespace find_k_l285_285471

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l285_285471


namespace normal_intersects_at_l285_285921

def parabola (x : ℝ) : ℝ := x^2

def slope_of_tangent (x : ℝ) : ℝ := 2 * x

-- C = (2, 4) is a point on the parabola
def C : ℝ × ℝ := (2, parabola 2)

-- Normal to the parabola at C intersects again at point D
-- Prove that D = (-9/4, 81/16)
theorem normal_intersects_at (D : ℝ × ℝ) :
  D = (-9/4, 81/16) :=
sorry

end normal_intersects_at_l285_285921


namespace fraction_value_l285_285184

theorem fraction_value (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : (x + y + z) / (2 * z) = 9 / 8 :=
by
  sorry

end fraction_value_l285_285184


namespace A_B_distance_l285_285394

noncomputable def distance_between_A_and_B 
  (vA: ℕ) (vB: ℕ) (vA_after_return: ℕ) 
  (meet_distance: ℕ) : ℚ := sorry

theorem A_B_distance (distance: ℚ) 
  (hA: vA = 40) (hB: vB = 60) 
  (hA_after_return: vA_after_return = 60) 
  (hmeet: meet_distance = 50) : 
  distance_between_A_and_B vA vB vA_after_return meet_distance = 1000 / 7 := sorry

end A_B_distance_l285_285394


namespace fixed_point_l285_285223

noncomputable def function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (x : ℝ) : ℝ :=
  a ^ (x - 1) + 1

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  function a h_pos h_ne_one 1 = 2 :=
by
  sorry

end fixed_point_l285_285223


namespace tetrahedron_painting_l285_285132

theorem tetrahedron_painting (unique_coloring_per_face : ∀ f : Fin 4, ∃ c : Fin 4, True)
  (rotation_identity : ∀ f g : Fin 4, (f = g → unique_coloring_per_face f = unique_coloring_per_face g))
  : (number_of_distinct_paintings : ℕ) = 2 :=
sorry

end tetrahedron_painting_l285_285132


namespace man_l285_285428

theorem man's_age_ratio_father (M F : ℕ) (hF : F = 60)
  (h_age_relationship : M + 12 = (F + 12) / 2) :
  M / F = 2 / 5 :=
by
  sorry

end man_l285_285428


namespace least_sum_of_exponents_l285_285318

theorem least_sum_of_exponents (a b c : ℕ) (ha : 2^a ∣ 520) (hb : 2^b ∣ 520) (hc : 2^c ∣ 520) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c = 12 :=
by
  sorry

end least_sum_of_exponents_l285_285318


namespace find_m_of_perpendicular_vectors_l285_285634

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l285_285634


namespace no_such_pairs_l285_285169

theorem no_such_pairs :
  ¬ ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4 * c < 0) ∧ (c^2 - 4 * b < 0) := sorry

end no_such_pairs_l285_285169


namespace next_number_in_sequence_is_131_l285_285976

/-- Define the sequence increments between subsequent numbers -/
def sequencePattern : List ℕ := [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]

-- Function to apply a sequence of increments starting from an initial value
def computeNext (initial : ℕ) (increments : List ℕ) : ℕ :=
  increments.foldl (λ acc inc => acc + inc) initial

-- Function to get the sequence's nth element 
def sequenceNthElement (n : ℕ) : ℕ :=
  (computeNext 12 (sequencePattern.take n))

-- Proof that the next number in the sequence is 131 
theorem next_number_in_sequence_is_131 :
  sequenceNthElement 10 = 131 :=
  by
  -- Proof omitted
  sorry

end next_number_in_sequence_is_131_l285_285976


namespace trig_identity_l285_285893

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) : 
  ∃ (res : ℝ), res = 10 / 7 ∧ res = Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) := by
  sorry

end trig_identity_l285_285893


namespace min_value_frac_l285_285303

theorem min_value_frac (x y a b c d : ℝ) (hx : 0 < x) (hy : 0 < y)
  (harith : x + y = a + b) (hgeo : x * y = c * d) : (a + b) ^ 2 / (c * d) ≥ 4 := 
by sorry

end min_value_frac_l285_285303


namespace some_magical_creatures_are_mystical_beings_l285_285907

-- Definitions and conditions based on the problem
def Dragon : Type := sorry
def MagicalCreature : Type := sorry
def MysticalBeing : Type := sorry

-- Main condition: All dragons are magical creatures
axiom (all_dragons_are_magical : ∀ d : Dragon, MagicalCreature)

-- Another condition: Some mystical beings are dragons
axiom (some_mystical_beings_are_dragons : ∃ m : MysticalBeing, ∃ d : Dragon, m = d)

-- Our task is to prove the required statement:
theorem some_magical_creatures_are_mystical_beings : ∃ mc : MagicalCreature, ∃ mb : MysticalBeing, mc = mb :=
begin
  sorry
end

end some_magical_creatures_are_mystical_beings_l285_285907


namespace range_of_k_l285_285898

theorem range_of_k (k : ℝ) :
  ∀ x : ℝ, ∃ a b c : ℝ, (a = k-1) → (b = -2) → (c = 1) → (a ≠ 0) → ((b^2 - 4 * a * c) ≥ 0) → k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_l285_285898


namespace total_profit_l285_285644

theorem total_profit (P Q R : ℝ) (profit : ℝ) 
  (h1 : 4 * P = 6 * Q) 
  (h2 : 6 * Q = 10 * R) 
  (h3 : R = 840 / 6) : 
  profit = 4340 :=
sorry

end total_profit_l285_285644


namespace problem_solution_l285_285489

theorem problem_solution (x : ℝ) (h : 1 - 9 / x + 20 / x^2 = 0) : (2 / x = 1 / 2 ∨ 2 / x = 2 / 5) := 
  sorry

end problem_solution_l285_285489


namespace johnson_family_seating_l285_285530

-- Defining the total number of children:
def total_children := 8

-- Defining the number of sons and daughters:
def sons := 5
def daughters := 3

-- Factoring in the total number of unrestricted seating arrangements:
def total_seating_arrangements : ℕ := Nat.factorial total_children

-- Factoring in the number of non-adjacent seating arrangements for sons:
def non_adjacent_arrangements : ℕ :=
  (Nat.factorial daughters) * (Nat.factorial sons)

-- The lean proof statement to prove:
theorem johnson_family_seating :
  total_seating_arrangements - non_adjacent_arrangements = 39600 :=
by
  sorry

end johnson_family_seating_l285_285530


namespace opposite_of_neg_three_l285_285388

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l285_285388


namespace minimum_value_frac_inverse_l285_285763

theorem minimum_value_frac_inverse (a b c : ℝ) (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b)) + (1 / c) ≥ 4 / 3 :=
by
  sorry

end minimum_value_frac_inverse_l285_285763


namespace probability_divisor_of_12_on_8_sided_die_l285_285424

theorem probability_divisor_of_12_on_8_sided_die :
  let outcomes := finset.range 8
  let divisors_of_12 := {1, 2, 3, 4, 6, 12}.filter (λ n, n ∈ finset.range 8)
  (divisors_of_12.card : ℚ) / (outcomes.card : ℚ) = 5 / 8 :=
by
  sorry

end probability_divisor_of_12_on_8_sided_die_l285_285424


namespace fraction_equiv_l285_285400

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l285_285400


namespace non_officers_count_l285_285564

theorem non_officers_count (avg_salary_all : ℕ) (avg_salary_officers : ℕ) (avg_salary_non_officers : ℕ) (num_officers : ℕ) 
  (N : ℕ) 
  (h_avg_salary_all : avg_salary_all = 120) 
  (h_avg_salary_officers : avg_salary_officers = 430) 
  (h_avg_salary_non_officers : avg_salary_non_officers = 110) 
  (h_num_officers : num_officers = 15) 
  (h_eq : avg_salary_all * (num_officers + N) = avg_salary_officers * num_officers + avg_salary_non_officers * N) 
  : N = 465 :=
by
  -- Proof would be here
  sorry

end non_officers_count_l285_285564


namespace part_I_part_II_l285_285483

def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem part_I (x : ℝ) : (f x > 4) ↔ (x < -1.5 ∨ x > 2.5) :=
by
  sorry

theorem part_II (x : ℝ) : ∀ x : ℝ, f x ≥ 3 :=
by
  sorry

end part_I_part_II_l285_285483


namespace find_k_l285_285473

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l285_285473


namespace product_of_integers_l285_285517

theorem product_of_integers (a b : ℚ) (h1 : a / b = 12) (h2 : a + b = 144) :
  a * b = 248832 / 169 := 
sorry

end product_of_integers_l285_285517


namespace finite_solutions_to_equation_l285_285526

theorem finite_solutions_to_equation :
  ∃ n : ℕ, ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ (1 / (a:ℝ) + 1 / (b:ℝ) + 1 / (c:ℝ) = 1 / 1983) →
    a ≤ n ∧ b ≤ n ∧ c ≤ n :=
sorry

end finite_solutions_to_equation_l285_285526


namespace correct_answer_statement_l285_285246

theorem correct_answer_statement
  (A := "In order to understand the situation of extracurricular reading among middle school students in China, a comprehensive survey should be conducted.")
  (B := "The median and mode of a set of data 1, 2, 5, 5, 5, 3, 3 are both 5.")
  (C := "When flipping a coin 200 times, there will definitely be 100 times when it lands 'heads up.'")
  (D := "If the variance of data set A is 0.03 and the variance of data set B is 0.1, then data set A is more stable than data set B.")
  (correct_answer := "D") : 
  correct_answer = "D" :=
  by sorry

end correct_answer_statement_l285_285246


namespace like_terms_mn_l285_285648

theorem like_terms_mn (m n : ℤ) 
  (H1 : m - 2 = 3) 
  (H2 : n + 2 = 1) : 
  m * n = -5 := 
by
  sorry

end like_terms_mn_l285_285648


namespace diana_total_extra_video_game_time_l285_285876

-- Definitions from the conditions
def minutesPerHourReading := 30
def raisePercent := 20
def choresToMinutes := 10
def maxChoresBonusMinutes := 60
def sportsPracticeHours := 8
def homeworkHours := 4
def totalWeekHours := 24
def readingHours := 8
def choresCompleted := 10

-- Deriving some necessary facts
def baseVideoGameTime := readingHours * minutesPerHourReading
def raiseMinutes := baseVideoGameTime * (raisePercent / 100)
def videoGameTimeWithRaise := baseVideoGameTime + raiseMinutes

def bonusesFromChores := (choresCompleted / 2) * choresToMinutes
def limitedChoresBonus := min bonusesFromChores maxChoresBonusMinutes

-- Total extra video game time
def totalExtraVideoGameTime := videoGameTimeWithRaise + limitedChoresBonus

-- The proof problem
theorem diana_total_extra_video_game_time : totalExtraVideoGameTime = 338 := by
  sorry

end diana_total_extra_video_game_time_l285_285876


namespace arithmetic_mean_difference_l285_285843

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
sorry

end arithmetic_mean_difference_l285_285843


namespace evaluate_expression_at_3_l285_285287

theorem evaluate_expression_at_3 :
  ((3^(3^2))^(3^3)) = 3^(243) := 
by 
  sorry

end evaluate_expression_at_3_l285_285287


namespace white_tile_count_l285_285337

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l285_285337


namespace Rachel_age_when_father_is_60_l285_285946

-- Given conditions
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Proof problem statement
theorem Rachel_age_when_father_is_60 : Rachel_age + (60 - Father_age) = 25 :=
by sorry

end Rachel_age_when_father_is_60_l285_285946


namespace find_t_max_value_of_xyz_l285_285704

-- Problem (1)
theorem find_t (t : ℝ) (x : ℝ) (h1 : |2 * x + t| - t ≤ 8) (sol_set : -5 ≤ x ∧ x ≤ 4) : t = 1 :=
sorry

-- Problem (2)
theorem max_value_of_xyz (x y z : ℝ) (h2 : x^2 + (1/4) * y^2 + (1/9) * z^2 = 2) : x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end find_t_max_value_of_xyz_l285_285704


namespace tom_gave_fred_balloons_l285_285127

variable (initial_balloons : ℕ) (remaining_balloons : ℕ)

def balloons_given (initial remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_fred_balloons (h₀ : initial_balloons = 30) (h₁ : remaining_balloons = 14) :
  balloons_given initial_balloons remaining_balloons = 16 :=
by
  -- Here we are skipping the proof
  sorry

end tom_gave_fred_balloons_l285_285127


namespace stewart_farm_food_l285_285718

variable (S H : ℕ) (HorseFoodPerHorsePerDay : Nat) (TotalSheep : Nat)

theorem stewart_farm_food (ratio_sheep_horses : 6 * H = 7 * S) 
  (total_sheep_count : S = 48) 
  (horse_food : HorseFoodPerHorsePerDay = 230) : 
  HorseFoodPerHorsePerDay * (7 * 48 / 6) = 12880 :=
by
  sorry

end stewart_farm_food_l285_285718


namespace maximum_volume_regular_triangular_pyramid_l285_285618

-- Given values
def R : ℝ := 1

-- Prove the maximum volume
theorem maximum_volume_regular_triangular_pyramid : 
  ∃ (V_max : ℝ), V_max = (8 * Real.sqrt 3) / 27 := 
by 
  sorry

end maximum_volume_regular_triangular_pyramid_l285_285618


namespace find_u_plus_v_l285_285901

theorem find_u_plus_v (u v : ℚ) (h1: 5 * u - 3 * v = 26) (h2: 3 * u + 5 * v = -19) :
  u + v = -101 / 34 :=
sorry

end find_u_plus_v_l285_285901


namespace probability_delegates_adjacent_l285_285128

-- Definitions for the problem's conditions
def total_delegates : ℕ := 12
def delegates_per_country : ℕ := 4
def total_countries : ℕ := 3

-- Statement of the theorem we want to prove
theorem probability_delegates_adjacent : 
  ∃ (m n : ℕ) (rel_prime : Nat.coprime m n), 
  n ≠ 0 ∧ (m * 1.0 / n = 106 * 1.0 / 115) ∧ (m + n = 221) :=
by
  -- This would require a formal proof, omitted here as instructed
  sorry

end probability_delegates_adjacent_l285_285128


namespace inequality_solution_l285_285230

-- Define the problem statement formally
theorem inequality_solution (x : ℝ)
  (h1 : 2 * x > x + 1)
  (h2 : 4 * x - 1 > 7) :
  x > 2 :=
sorry

end inequality_solution_l285_285230


namespace measure_angle_WYZ_l285_285797

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end measure_angle_WYZ_l285_285797


namespace neutral_equilibrium_l285_285688

noncomputable def equilibrium_ratio (r h : ℝ) : ℝ := r / h

theorem neutral_equilibrium (r h : ℝ) (k : ℝ) : (equilibrium_ratio r h = k) → (k = Real.sqrt 2) :=
by
  intro h1
  have h1' : (r / h = k) := h1
  sorry

end neutral_equilibrium_l285_285688


namespace triangle_area_is_24_l285_285696

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (0, 6)
def C : point := (8, 10)

def triangle_area (A B C : point) : ℝ := 
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_24 : triangle_area A B C = 24 :=
by
  -- Insert proof here
  sorry

end triangle_area_is_24_l285_285696


namespace Marcia_wardrobe_cost_l285_285931

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

end Marcia_wardrobe_cost_l285_285931


namespace converse_proposition_l285_285532

-- Define the condition: The equation x^2 + x - m = 0 has real roots
def has_real_roots (a b c : ℝ) : Prop :=
  let Δ := b * b - 4 * a * c
  Δ ≥ 0

theorem converse_proposition (m : ℝ) :
  has_real_roots 1 1 (-m) → m > 0 :=
by
  sorry

end converse_proposition_l285_285532


namespace vanessa_points_record_l285_285830

theorem vanessa_points_record 
  (P : ℕ) 
  (H₁ : P = 48) 
  (O : ℕ) 
  (H₂ : O = 6 * 3.5) : V = (P - O) → V = 27 :=
by
  sorry

end vanessa_points_record_l285_285830


namespace annalise_total_cost_l285_285714

/-- 
Given conditions:
- 25 boxes of tissues.
- Each box contains 18 packs.
- Each pack contains 150 tissues.
- Each tissue costs $0.06.
- A 10% discount on the total price of the packs in each box.

Prove:
The total amount of money Annalise spent is $3645.
-/
theorem annalise_total_cost :
  let boxes := 25
  let packs_per_box := 18
  let tissues_per_pack := 150
  let cost_per_tissue := 0.06
  let discount_rate := 0.10
  let price_per_box := (packs_per_box * tissues_per_pack * cost_per_tissue)
  let discount_per_box := discount_rate * price_per_box
  let discounted_price_per_box := price_per_box - discount_per_box
  let total_cost := discounted_price_per_box * boxes
  total_cost = 3645 :=
by
  sorry

end annalise_total_cost_l285_285714


namespace simplify_expression_l285_285364

variable (b : ℝ)

theorem simplify_expression :
  (2 * b + 6 - 5 * b) / 2 = -3 / 2 * b + 3 :=
sorry

end simplify_expression_l285_285364


namespace rohan_monthly_salary_l285_285947

theorem rohan_monthly_salary :
  ∃ S : ℝ, 
    (0.4 * S) + (0.2 * S) + (0.1 * S) + (0.1 * S) + 1000 = S :=
by
  sorry

end rohan_monthly_salary_l285_285947


namespace last_four_digits_of_m_smallest_l285_285198

theorem last_four_digits_of_m_smallest (m : ℕ) (h1 : m > 0)
  (h2 : m % 6 = 0) (h3 : m % 8 = 0)
  (h4 : ∀ d, d ∈ (m.digits 10) → d = 2 ∨ d = 7)
  (h5 : 2 ∈ (m.digits 10)) (h6 : 7 ∈ (m.digits 10)) :
  (m % 10000) = 2722 :=
sorry

end last_four_digits_of_m_smallest_l285_285198


namespace hex_351_is_849_l285_285268

noncomputable def hex_to_decimal : ℕ := 1 * 16^0 + 5 * 16^1 + 3 * 16^2

-- The following statement is the core of the proof problem
theorem hex_351_is_849 : hex_to_decimal = 849 := by
  -- Here the proof steps would normally go
  sorry

end hex_351_is_849_l285_285268


namespace monotonic_decreasing_interval_l285_285114

noncomputable def f (x : ℝ) := Real.log x + x^2 - 3 * x

theorem monotonic_decreasing_interval :
  (∃ I : Set ℝ, I = Set.Ioo (1 / 2 : ℝ) 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) := 
by
  sorry

end monotonic_decreasing_interval_l285_285114


namespace fraction_subtraction_equivalence_l285_285466

theorem fraction_subtraction_equivalence : 
  (16 / 24 - (1 + 2 / 9)) = -(5 / 9) := by
  -- Simplification of the fractions
  have h1 : 16 / 24 = 2 / 3, by sorry
  have h2 : 1 + 2 / 9 = 11 / 9, by sorry
  -- Conversion to a common denominator
  have h3 : (2 / 3) = 6 / 9, by sorry
  -- Subtraction
  show (6 / 9 - 11 / 9) = -(5 / 9), by sorry

end fraction_subtraction_equivalence_l285_285466


namespace compare_a_b_c_compare_explicitly_defined_a_b_c_l285_285301

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end compare_a_b_c_compare_explicitly_defined_a_b_c_l285_285301


namespace mobius_total_trip_time_l285_285095

-- Define Mobius's top speed without any load
def speed_no_load : ℝ := 13

-- Define Mobius's top speed with a typical load
def speed_with_load : ℝ := 11

-- Define the distance from Florence to Rome
def distance : ℝ := 143

-- Define the number of rest stops per half trip and total rest stops
def rest_stops_per_half_trip : ℕ := 2
def total_rest_stops : ℕ := 2 * rest_stops_per_half_trip

-- Define the rest time per stop in hours
def rest_time_per_stop : ℝ := 0.5

-- Calculate the total rest time
def total_rest_time : ℝ := total_rest_stops * rest_time_per_stop

-- Calculate the total trip time
def total_trip_time : ℝ := (distance / speed_with_load) + (distance / speed_no_load) + total_rest_time

-- The theorem to be proved
theorem mobius_total_trip_time : total_trip_time = 26 := by
  -- definition follows directly from the problem statement
  sorry

end mobius_total_trip_time_l285_285095


namespace final_sum_l285_285719

-- Assuming an initial condition for the values on the calculators
def initial_values : List Int := [2, 1, -1]

-- Defining the operations to be applied on the calculators
def operations (vals : List Int) : List Int :=
  match vals with
  | [a, b, c] => [a * a, b * b * b, -c]
  | _ => vals  -- This case handles unexpected input formats

-- Applying the operations for 43 participants
def final_values (vals : List Int) (n : Nat) : List Int :=
  if n = 0 then vals
  else final_values (operations vals) (n - 1)

-- Prove that the final sum of the values on the calculators equals 2 ^ 2 ^ 43
theorem final_sum : 
  final_values initial_values 43 = [2 ^ 2 ^ 43, 1, -1] → 
  List.sum (final_values initial_values 43) = 2 ^ 2 ^ 43 :=
by
  intro h -- This introduces the hypothesis that the final values list equals the expected values
  sorry   -- Provide an ultimate proof for the statement.

end final_sum_l285_285719


namespace eggs_supplied_l285_285932

-- Define the conditions
def daily_eggs_first_store (D : ℕ) : ℕ := 12 * D
def daily_eggs_second_store : ℕ := 30
def total_weekly_eggs (D : ℕ) : ℕ := 7 * (daily_eggs_first_store D + daily_eggs_second_store)

-- Statement: prove that if the total number of eggs supplied in a week is 630,
-- then Mark supplies 5 dozen eggs to the first store each day.
theorem eggs_supplied (D : ℕ) (h : total_weekly_eggs D = 630) : D = 5 :=
by
  sorry

end eggs_supplied_l285_285932


namespace correct_substitution_l285_285244

theorem correct_substitution (x y : ℤ) (h1 : x = 3 * y - 1) (h2 : x - 2 * y = 4) :
  3 * y - 1 - 2 * y = 4 :=
by
  sorry

end correct_substitution_l285_285244


namespace two_pipes_fill_tank_l285_285555

theorem two_pipes_fill_tank (C : ℝ) (hA : ∀ (t : ℝ), t = 10 → t = C / (C / 10)) (hB : ∀ (t : ℝ), t = 15 → t = C / (C / 15)) :
  ∀ (t : ℝ), t = C / (C / 6) → t = 6 :=
by
  sorry

end two_pipes_fill_tank_l285_285555


namespace probability_of_valid_number_l285_285991

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≠ j → (n % (10^i) / 10^(i-1)) ≠ (n % (10^j) / 10^(j-1))

def digits_in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_number (n : ℕ) : Prop :=
  is_even n ∧ has_distinct_digits n ∧ digits_in_range n

noncomputable def count_valid_numbers : ℕ :=
  2296

noncomputable def total_numbers : ℕ :=
  9000

theorem probability_of_valid_number :
  (count_valid_numbers : ℚ) / total_numbers = 574 / 2250 :=
by sorry

end probability_of_valid_number_l285_285991


namespace binomial_coefficient_proof_l285_285083

open Real BigOperators

-- Definition of region D and the point P(x, y)
def isInRegionD (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Definition of the inequality y ≤ x^2
def satisfiesIneq (x y : ℝ) : Prop := y ≤ x^2

-- The statement we need to prove
theorem binomial_coefficient_proof :
  let a := (∫ x in -1..1, x^2) / 2 in
  a = 1 / 3 ∧
  (∑ r in finset.range 6, if 5 - (3 / 2 : ℝ) * r = 2 then 
    nat.choose 5 r * (-1) ^ r * 3 ^ (5 - r) else 0) = 270 := 
by {
  set a := (\int (x in -1..1), x^2) / 2,
  split,
  { simp [a], 
    sorry },
  { 
    simp [a],
    sorry 
  }
}

end binomial_coefficient_proof_l285_285083


namespace exists_four_digit_number_divisible_by_101_l285_285334

theorem exists_four_digit_number_divisible_by_101 :
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
    b ≠ c ∧ b ≠ d ∧
    c ≠ d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) % 101 = 0 := 
by
  -- To be proven
  sorry

end exists_four_digit_number_divisible_by_101_l285_285334


namespace class_students_count_l285_285452

theorem class_students_count (n : ℕ) 
  (h1 : ∃ k, n = 2 * k + 1 ∧ k + 2 = (k + 3) * 2 - 2) 
  (h2 : ∃ m, n = 2 * (2 * m + 2) - 2 + 3) 
  (h3 : ∃ l, n = 2 * ((2 * l + 3) + 2) - 2) : 
  n = 50 :=
begin
  sorry
end

end class_students_count_l285_285452


namespace gary_current_weekly_eggs_l285_285171

noncomputable def egg_laying_rates : List ℕ := [6, 5, 7, 4]

def total_eggs_per_day (rates : List ℕ) : ℕ :=
  rates.foldl (· + ·) 0

def total_eggs_per_week (eggs_per_day : ℕ) : ℕ :=
  eggs_per_day * 7

theorem gary_current_weekly_eggs : 
  total_eggs_per_week (total_eggs_per_day egg_laying_rates) = 154 :=
by
  sorry

end gary_current_weekly_eggs_l285_285171


namespace larry_expression_correct_l285_285671

theorem larry_expression_correct (a b c d : ℤ) (e : ℤ) :
  (a = 1) → (b = 2) → (c = 3) → (d = 4) →
  (a - b - c - d + e = -2 - e) → (e = 3) :=
by
  intros ha hb hc hd heq
  rw [ha, hb, hc, hd] at heq
  linarith

end larry_expression_correct_l285_285671


namespace prime_9_greater_than_perfect_square_l285_285837

theorem prime_9_greater_than_perfect_square (p : ℕ) (hp : Nat.Prime p) :
  ∃ n m : ℕ, p - 9 = n^2 ∧ p + 2 = m^2 ∧ p = 23 :=
by
  sorry

end prime_9_greater_than_perfect_square_l285_285837


namespace max_value_T_n_l285_285666

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q^n

noncomputable def sum_of_first_n_terms (a₁ q : ℝ) (n : ℕ) :=
  a₁ * (1 - q^(n + 1)) / (1 - q)

noncomputable def T_n (a₁ q : ℝ) (n : ℕ) :=
  (9 * sum_of_first_n_terms a₁ q n - sum_of_first_n_terms a₁ q (2 * n)) /
  geometric_sequence a₁ q (n + 1)

theorem max_value_T_n
  (a₁ : ℝ) (n : ℕ) (h : n > 0) (q : ℝ) (hq : q = 2) :
  ∃ n₀ : ℕ, T_n a₁ q n₀ = 3 := sorry

end max_value_T_n_l285_285666


namespace opposite_of_neg_three_l285_285390

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l285_285390


namespace compute_f3_l285_285513

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 4*n + 3 else 2*n + 1

theorem compute_f3 : f (f (f 3)) = 99 :=
by
  sorry

end compute_f3_l285_285513


namespace impossible_arrangement_of_numbers_l285_285849

theorem impossible_arrangement_of_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) 
(hpos : ∀ i, 0 < a i)
(hdiff : ∃ i, ∀ j ≠ i, a j = a ((j + 1) % n) - a ((j - 1 + n) % n)):
  false :=
by
  sorry

end impossible_arrangement_of_numbers_l285_285849


namespace sufficient_but_not_necessary_l285_285252

theorem sufficient_but_not_necessary (x : ℝ) :
  (x < -1 → x^2 - 1 > 0) ∧ (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l285_285252


namespace tangent_circle_given_r_l285_285903

theorem tangent_circle_given_r (r : ℝ) (h_pos : 0 < r)
    (h_tangent : ∀ x y : ℝ, (2 * x + y = r) → (x^2 + y^2 = 2 * r))
  : r = 10 :=
sorry

end tangent_circle_given_r_l285_285903


namespace knight_king_moves_incompatible_l285_285832

-- Definitions for moves and chessboards
structure Board :=
  (numbering : Fin 64 → Nat)
  (different_board : Prop)

def knights_move (x y : Fin 64) : Prop :=
  (abs (x / 8 - y / 8) = 2 ∧ abs (x % 8 - y % 8) = 1) ∨
  (abs (x / 8 - y / 8) = 1 ∧ abs (x % 8 - y % 8) = 2)

def kings_move (x y : Fin 64) : Prop :=
  abs (x / 8 - y / 8) ≤ 1 ∧ abs (x % 8 - y % 8) ≤ 1 ∧ (x ≠ y)

-- Theorem stating the proof problem
theorem knight_king_moves_incompatible (vlad_board gosha_board : Board) (h_board_diff: vlad_board.different_board):
  ¬ ∀ i j : Fin 64, (knights_move i j ↔ kings_move (vlad_board.numbering i) (vlad_board.numbering j)) :=
by {
  -- Skipping proofs with sorry
  sorry
}

end knight_king_moves_incompatible_l285_285832


namespace arith_seq_ratio_l285_285500

theorem arith_seq_ratio (a_2 a_3 S_4 S_5 : ℕ) 
  (arithmetic_seq : ∀ n : ℕ, ℕ)
  (sum_of_first_n_terms : ∀ n : ℕ, ℕ)
  (h1 : (a_2 : ℚ) / a_3 = 1 / 3) 
  (h2 : S_4 = 4 * (a_2 - (a_3 - a_2)) + ((4 * 3 * (a_3 - a_2)) / 2)) 
  (h3 : S_5 = 5 * (a_2 - (a_3 - a_2)) + ((5 * 4 * (a_3 - a_2)) / 2)) :
  (S_4 : ℚ) / S_5 = 8 / 15 :=
by sorry

end arith_seq_ratio_l285_285500


namespace find_values_l285_285761

theorem find_values (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  3 * Real.sin x + 2 * Real.cos x = ( -21 + 13 * Real.sqrt 145 ) / 58 ∨
  3 * Real.sin x + 2 * Real.cos x = ( -21 - 13 * Real.sqrt 145 ) / 58 := sorry

end find_values_l285_285761


namespace find_length_EF_l285_285788

theorem find_length_EF (DE DF : ℝ) (angle_E : ℝ) (angle_E_val : angle_E = 45) (DE_val : DE = 100) (DF_val : DF = 100 * Real.sqrt 2) : ∃ EF, EF = 141.421 :=
by
  exists 141.421
  sorry

end find_length_EF_l285_285788


namespace percentage_decrease_l285_285226

theorem percentage_decrease (original_price new_price : ℝ) (h1 : original_price = 1400) (h2 : new_price = 1064) :
  ((original_price - new_price) / original_price * 100) = 24 :=
by
  sorry

end percentage_decrease_l285_285226


namespace greatest_servings_l285_285581

def servings (ingredient_amount recipe_amount: ℚ) (recipe_servings: ℕ) : ℚ :=
  (ingredient_amount / recipe_amount) * recipe_servings

theorem greatest_servings (chocolate_new_recipe sugar_new_recipe water_new_recipe milk_new_recipe : ℚ)
                         (servings_new_recipe : ℕ)
                         (chocolate_jordan sugar_jordan milk_jordan : ℚ)
                         (lots_of_water : Prop) :
  chocolate_new_recipe = 3 ∧ sugar_new_recipe = 1/3 ∧ water_new_recipe = 1.5 ∧ milk_new_recipe = 5 ∧
  servings_new_recipe = 6 ∧ chocolate_jordan = 8 ∧ sugar_jordan = 3 ∧ milk_jordan = 12 ∧ lots_of_water →
  max (servings chocolate_jordan chocolate_new_recipe servings_new_recipe)
      (max (servings sugar_jordan sugar_new_recipe servings_new_recipe)
           (servings milk_jordan milk_new_recipe servings_new_recipe)) = 16 :=
by
  sorry

end greatest_servings_l285_285581


namespace ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l285_285690

-- Define the conditions
def W : ℕ := 3
def wait_time_swing : ℕ := 120 * W
def wait_time_slide (S : ℕ) : ℕ := 15 * S
def wait_diff_condition (S : ℕ) : Prop := wait_time_swing - wait_time_slide S = 270

theorem ratio_of_kids_waiting_for_slide_to_swings (S : ℕ) (h : wait_diff_condition S) : S = 6 :=
by
  -- placeholder proof
  sorry

theorem final_ratio_of_kids_waiting (S : ℕ) (h : wait_diff_condition S) : S / W = 2 :=
by
  -- placeholder proof
  sorry

end ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l285_285690


namespace twice_product_of_numbers_l285_285964

theorem twice_product_of_numbers (x y : ℝ) (h1 : x + y = 80) (h2 : x - y = 10) : 2 * (x * y) = 3150 := by
  sorry

end twice_product_of_numbers_l285_285964


namespace pencil_rows_l285_285290

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) : (total_pencils / pencils_per_row) = 7 :=
by
  sorry

end pencil_rows_l285_285290


namespace inequality_problem_l285_285928

noncomputable def nonneg_real := {x : ℝ // 0 ≤ x}

theorem inequality_problem (x y z : nonneg_real) (h : x.val * y.val + y.val * z.val + z.val * x.val = 1) :
  1 / (x.val + y.val) + 1 / (y.val + z.val) + 1 / (z.val + x.val) ≥ 5 / 2 :=
sorry

end inequality_problem_l285_285928


namespace petya_can_restore_numbers_if_and_only_if_odd_l285_285676

def can_restore_numbers (n : ℕ) : Prop :=
  ∀ (V : Fin n → ℕ) (S : ℕ),
    ∃ f : Fin n → ℕ, 
    (∀ i : Fin n, 
      (V i) = f i ∨ 
      (S = f i)) ↔ (n % 2 = 1)

theorem petya_can_restore_numbers_if_and_only_if_odd (n : ℕ) : can_restore_numbers n ↔ n % 2 = 1 := 
by sorry

end petya_can_restore_numbers_if_and_only_if_odd_l285_285676


namespace op_op_k_l285_285729

def op (x y : ℝ) : ℝ := x^3 + x - y

theorem op_op_k (k : ℝ) : op k (op k k) = k := sorry

end op_op_k_l285_285729


namespace fraction_equivalent_l285_285403

theorem fraction_equivalent (x : ℝ) : x = 433 / 990 ↔ x = 0.4 + 37 / 990 * 10 ^ -2 :=
by
  sorry

end fraction_equivalent_l285_285403


namespace line_equation_l285_285464

theorem line_equation
  (x y : ℝ)
  (h1 : 2 * x + y + 2 = 0)
  (h2 : 2 * x - y + 2 = 0)
  (h3 : ∀ x y, x + y = 0 → x - 1 = y): 
  x - y + 1 = 0 :=
sorry

end line_equation_l285_285464


namespace difference_between_possible_x_values_l285_285492

theorem difference_between_possible_x_values :
  ∀ (x : ℝ), (x + 3) ^ 2 / (2 * x + 15) = 3 → (x = 6 ∨ x = -6) →
  (abs (6 - (-6)) = 12) :=
by
  intro x h1 h2
  sorry

end difference_between_possible_x_values_l285_285492


namespace find_m_value_l285_285065

theorem find_m_value (m : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x y : ℝ), x + m * y + 3 - 2 * m = 0) →
  (∃ (y : ℝ), x = 0 ∧ y = -1) →
  m = 1 :=
by
  sorry

end find_m_value_l285_285065


namespace total_oranges_for_philip_l285_285170

-- Define the initial conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def combined_oranges : ℕ := betty_oranges + bill_oranges
def frank_oranges : ℕ := 3 * combined_oranges
def seeds_planted : ℕ := 4 * frank_oranges
def successful_trees : ℕ := (3 / 4) * seeds_planted

-- The ratio of trees with different quantities of oranges
def ratio_parts : ℕ := 2 + 3 + 5
def trees_with_8_oranges : ℕ := (2 * successful_trees) / ratio_parts
def trees_with_10_oranges : ℕ := (3 * successful_trees) / ratio_parts
def trees_with_14_oranges : ℕ := (5 * successful_trees) / ratio_parts

-- Calculate the total number of oranges
def total_oranges : ℕ :=
  (trees_with_8_oranges * 8) +
  (trees_with_10_oranges * 10) +
  (trees_with_14_oranges * 14)

-- Statement to prove
theorem total_oranges_for_philip : total_oranges = 2798 :=
by
  sorry

end total_oranges_for_philip_l285_285170


namespace sqrt_equality_l285_285839

theorem sqrt_equality (m : ℝ) (n : ℝ) (h1 : 0 < m) (h2 : -3 * m ≤ n) (h3 : n ≤ 3 * m) :
    (Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2))
     - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2))
    = 2 * Real.sqrt (3 * m - n)) :=
sorry

end sqrt_equality_l285_285839


namespace trirectangular_tetrahedron_max_volume_l285_285910

noncomputable def max_volume_trirectangular_tetrahedron (S : ℝ) : ℝ :=
  S^3 * (Real.sqrt 2 - 1)^3 / 162

theorem trirectangular_tetrahedron_max_volume
  (a b c : ℝ) (H : a > 0 ∧ b > 0 ∧ c > 0)
  (S : ℝ)
  (edge_sum :
    S = a + b + c + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2))
  : ∃ V, V = max_volume_trirectangular_tetrahedron S :=
by
  sorry

end trirectangular_tetrahedron_max_volume_l285_285910


namespace flowers_per_basket_l285_285280

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end flowers_per_basket_l285_285280


namespace length_of_one_side_nonagon_l285_285691

def total_perimeter (n : ℕ) (side_length : ℝ) : ℝ := n * side_length

theorem length_of_one_side_nonagon (total_perimeter : ℝ) (n : ℕ) (side_length : ℝ) (h1 : n = 9) (h2 : total_perimeter = 171) : side_length = 19 :=
by
  sorry

end length_of_one_side_nonagon_l285_285691


namespace ones_digit_of_8_pow_50_l285_285242

theorem ones_digit_of_8_pow_50 : (8 ^ 50) % 10 = 4 := by
  sorry

end ones_digit_of_8_pow_50_l285_285242


namespace roots_of_polynomial_l285_285603

theorem roots_of_polynomial :
  (∀ x : ℝ, (x^2 - 5 * x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5) :=
by
  sorry

end roots_of_polynomial_l285_285603


namespace problem_statement_l285_285801

theorem problem_statement (a b c d : ℝ) 
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (hcd : c ≤ d)
  (hsum : a + b + c + d = 0)
  (hinv_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 :=
sorry

end problem_statement_l285_285801


namespace geometric_sequence_sum_q_value_l285_285332

theorem geometric_sequence_sum_q_value (q : ℝ) (a S : ℕ → ℝ) :
  a 1 = 4 →
  (∀ n, a (n+1) = a n * q ) →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, (S n + 2) = (S 1 + 2) * (q ^ (n - 1))) →
  q = 3
:= 
by
  sorry

end geometric_sequence_sum_q_value_l285_285332
