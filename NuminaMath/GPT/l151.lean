import Mathlib

namespace range_of_a_l151_15184

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ a > 3 ∨ a < -3 :=
by
  sorry

end range_of_a_l151_15184


namespace call_cost_per_minute_l151_15147

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end call_cost_per_minute_l151_15147


namespace gift_cost_calc_l151_15105

theorem gift_cost_calc (C N : ℕ) (hN : N = 12)
    (h : C / (N - 4) = C / N + 10) : C = 240 := by
  sorry

end gift_cost_calc_l151_15105


namespace curves_intersect_four_points_l151_15138

theorem curves_intersect_four_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = 4 * a^2 ∧ y = x^2 - 2 * a) → (a > 1/3)) :=
sorry

end curves_intersect_four_points_l151_15138


namespace white_roses_count_l151_15186

def total_flowers : ℕ := 6284
def red_roses : ℕ := 1491
def yellow_carnations : ℕ := 3025
def white_roses : ℕ := total_flowers - (red_roses + yellow_carnations)

theorem white_roses_count :
  white_roses = 1768 := by
  sorry

end white_roses_count_l151_15186


namespace total_capacity_l151_15165

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l151_15165


namespace find_m_l151_15133

variable (a : ℝ × ℝ := (2, 3))
variable (b : ℝ × ℝ := (-1, 2))

def isCollinear (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

theorem find_m (m : ℝ) (h : isCollinear (2 * m - 4, 3 * m + 8) (4, -1)) : m = -2 :=
by {
  sorry
}

end find_m_l151_15133


namespace bianca_ate_candies_l151_15127

-- Definitions based on the conditions
def total_candies : ℕ := 32
def pieces_per_pile : ℕ := 5
def number_of_piles : ℕ := 4

-- The statement to prove
theorem bianca_ate_candies : 
  total_candies - (pieces_per_pile * number_of_piles) = 12 := 
by 
  sorry

end bianca_ate_candies_l151_15127


namespace determine_a_l151_15125

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem determine_a (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = 1 / 2 :=
by
  sorry

end determine_a_l151_15125


namespace alpha_beta_value_l151_15169

theorem alpha_beta_value :
  ∃ α β : ℝ, (α^2 - 2 * α - 4 = 0) ∧ (β^2 - 2 * β - 4 = 0) ∧ (α + β = 2) ∧ (α^3 + 8 * β + 6 = 30) :=
by
  sorry

end alpha_beta_value_l151_15169


namespace units_digit_27_mul_46_l151_15171

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l151_15171


namespace convex_polygon_with_tiles_l151_15128

variable (n : ℕ)

def canFormConvexPolygon (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 12

theorem convex_polygon_with_tiles (n : ℕ) 
  (square_internal_angle : ℕ := 90) 
  (equilateral_triangle_internal_angle : ℕ := 60)
  (external_angle_step : ℕ := 30)
  (total_external_angle : ℕ := 360) :
  canFormConvexPolygon n :=
by 
  sorry

end convex_polygon_with_tiles_l151_15128


namespace no_intersections_root_of_quadratic_l151_15177

theorem no_intersections_root_of_quadratic (x : ℝ) :
  ¬(∃ x, (y = x) ∧ (y = x - 3)) ↔ (x^2 - 3 * x = 0) := by
  sorry

end no_intersections_root_of_quadratic_l151_15177


namespace find_principal_l151_15122

theorem find_principal (R : ℝ) (P : ℝ) (h : ((P * (R + 5) * 10) / 100) = ((P * R * 10) / 100 + 600)) : P = 1200 :=
by
  sorry

end find_principal_l151_15122


namespace valentino_chickens_l151_15143

variable (C : ℕ) -- Number of chickens
variable (D : ℕ) -- Number of ducks
variable (T : ℕ) -- Number of turkeys
variable (total_birds : ℕ) -- Total number of birds on the farm

theorem valentino_chickens (h1 : D = 2 * C) 
                            (h2 : T = 3 * D)
                            (h3 : total_birds = C + D + T)
                            (h4 : total_birds = 1800) :
  C = 200 := by
  sorry

end valentino_chickens_l151_15143


namespace toy_spending_ratio_l151_15181

theorem toy_spending_ratio :
  ∃ T : ℝ, 204 - T > 0 ∧ 51 = (204 - T) / 2 ∧ (T / 204) = 1 / 2 :=
by
  sorry

end toy_spending_ratio_l151_15181


namespace difference_mean_median_l151_15129

theorem difference_mean_median :
  let percentage_scored_60 : ℚ := 0.20
  let percentage_scored_70 : ℚ := 0.30
  let percentage_scored_85 : ℚ := 0.25
  let percentage_scored_95 : ℚ := 1 - (percentage_scored_60 + percentage_scored_70 + percentage_scored_85)
  let score_60 : ℚ := 60
  let score_70 : ℚ := 70
  let score_85 : ℚ := 85
  let score_95 : ℚ := 95
  let mean : ℚ := percentage_scored_60 * score_60 + percentage_scored_70 * score_70 + percentage_scored_85 * score_85 + percentage_scored_95 * score_95
  let median : ℚ := 85
  (median - mean) = 7 := 
by 
  sorry

end difference_mean_median_l151_15129


namespace cost_of_5_spoons_l151_15150

theorem cost_of_5_spoons (cost_per_set : ℕ) (num_spoons_per_set : ℕ) (num_spoons_needed : ℕ)
  (h1 : cost_per_set = 21) (h2 : num_spoons_per_set = 7) (h3 : num_spoons_needed = 5) :
  (cost_per_set / num_spoons_per_set) * num_spoons_needed = 15 :=
by
  sorry

end cost_of_5_spoons_l151_15150


namespace find_number_l151_15137

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end find_number_l151_15137


namespace original_price_of_apples_l151_15131

-- Define variables and conditions
variables (P : ℝ)

-- The conditions of the problem
def price_increase_condition := 1.25 * P * 8 = 64

-- The theorem stating the original price per pound of apples
theorem original_price_of_apples (h : price_increase_condition P) : P = 6.40 :=
sorry

end original_price_of_apples_l151_15131


namespace no_nonneg_rational_sol_for_equation_l151_15185

theorem no_nonneg_rational_sol_for_equation :
  ¬ ∃ (x y z : ℚ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^5 + 2 * y^5 + 5 * z^5 = 11 :=
by
  sorry

end no_nonneg_rational_sol_for_equation_l151_15185


namespace arc_PQ_circumference_l151_15156

-- Definitions based on the identified conditions
def radius : ℝ := 24
def angle_PRQ : ℝ := 90

-- The theorem to prove based on the question and correct answer
theorem arc_PQ_circumference : 
  angle_PRQ = 90 → 
  ∃ arc_length : ℝ, arc_length = (2 * Real.pi * radius) / 4 ∧ arc_length = 12 * Real.pi :=
by
  sorry

end arc_PQ_circumference_l151_15156


namespace total_grains_in_grey_regions_l151_15110

def total_grains_circle1 : ℕ := 87
def total_grains_circle2 : ℕ := 110
def white_grains_circle1 : ℕ := 68
def white_grains_circle2 : ℕ := 68

theorem total_grains_in_grey_regions : total_grains_circle1 - white_grains_circle1 + (total_grains_circle2 - white_grains_circle2) = 61 :=
by
  sorry

end total_grains_in_grey_regions_l151_15110


namespace smallest_d_l151_15162

theorem smallest_d (d : ℝ) : 
  (∃ d, 2 * d = Real.sqrt ((4 * Real.sqrt 3) ^ 2 + (d + 4) ^ 2)) →
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
by
  sorry

end smallest_d_l151_15162


namespace ratio_enlarged_by_nine_l151_15189

theorem ratio_enlarged_by_nine (a b : ℕ) (h : b ≠ 0) :
  (3 * a) / (b / 3) = 9 * (a / b) :=
by
  have h1 : b / 3 ≠ 0 := by sorry
  have h2 : a * 3 ≠ 0 := by sorry
  sorry

end ratio_enlarged_by_nine_l151_15189


namespace circle_center_and_radius_l151_15182

theorem circle_center_and_radius :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ 
    (x - C.1)^2 + (y - C.2)^2 = r^2) ∧ C = (1, -2) ∧ r = Real.sqrt 2 :=
by 
  sorry

end circle_center_and_radius_l151_15182


namespace ratio_passengers_i_to_ii_l151_15152

-- Definitions: Conditions from the problem
variables (total_fare : ℕ) (fare_ii_class : ℕ) (fare_i_class_ratio_to_ii : ℕ)

-- Given conditions
axiom total_fare_collected : total_fare = 1325
axiom fare_collected_from_ii_class : fare_ii_class = 1250
axiom i_to_ii_fare_ratio : fare_i_class_ratio_to_ii = 3

-- Define the fare for I class and II class passengers
def fare_i_class := 3 * (fare_ii_class / fare_i_class_ratio_to_ii)

-- Statement of the proof problem translating the question, conditions, and answer
theorem ratio_passengers_i_to_ii (x y : ℕ) (h1 : 3 * fare_i_class * x = total_fare - fare_ii_class)
    (h2 : (fare_ii_class / fare_i_class_ratio_to_ii) * y = fare_ii_class) : x = y / 50 :=
by
  sorry

end ratio_passengers_i_to_ii_l151_15152


namespace remainder_349_div_13_l151_15120

theorem remainder_349_div_13 : 349 % 13 = 11 := 
by 
  sorry

end remainder_349_div_13_l151_15120


namespace find_C_and_D_l151_15113

noncomputable def C : ℚ := 15 / 8
noncomputable def D : ℚ := 17 / 8

theorem find_C_and_D (x : ℚ) (h₁ : x ≠ 9) (h₂ : x ≠ -7) :
  (4 * x - 6) / ((x - 9) * (x + 7)) = C / (x - 9) + D / (x + 7) :=
by sorry

end find_C_and_D_l151_15113


namespace prime_square_minus_five_not_div_by_eight_l151_15164

theorem prime_square_minus_five_not_div_by_eight (p : ℕ) (prime_p : Prime p) (p_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) :=
sorry

end prime_square_minus_five_not_div_by_eight_l151_15164


namespace additional_charge_is_correct_l151_15123

noncomputable def additional_charge_per_segment (initial_fee : ℝ) (total_distance : ℝ) (total_charge : ℝ) (segment_length : ℝ) : ℝ :=
  let segments := total_distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  charge_for_distance / segments

theorem additional_charge_is_correct :
  additional_charge_per_segment 2.0 3.6 5.15 (2/5) = 0.35 :=
by
  sorry

end additional_charge_is_correct_l151_15123


namespace function_properties_l151_15172

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 + x) + log (2 - x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ ⦃a b : ℝ⦄, 0 < a → a < b → b < 2 → f b < f a) := by
  sorry

end function_properties_l151_15172


namespace geometric_sequence_second_term_l151_15118

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l151_15118


namespace fraction_value_l151_15163

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 5) (h3 : (∃ m : ℤ, x = m * y)) : x / y = -2 :=
sorry

end fraction_value_l151_15163


namespace terms_before_one_l151_15168

-- Define the sequence parameters
def a : ℤ := 100
def d : ℤ := -7
def nth_term (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the target term we are interested in
def target_term : ℤ := 1

-- Define the main theorem
theorem terms_before_one : ∃ n : ℕ, nth_term n = target_term ∧ (n - 1) = 14 := by
  sorry

end terms_before_one_l151_15168


namespace intersection_M_N_l151_15132

open Set

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_M_N :
  M ∩ N = { x | -2 ≤ x ∧ x ≤ -1 } := by
  sorry

end intersection_M_N_l151_15132


namespace tree_height_at_3_years_l151_15155

-- Define the conditions as Lean definitions
def tree_height (years : ℕ) : ℕ :=
  2 ^ years

-- State the theorem using the defined conditions
theorem tree_height_at_3_years : tree_height 6 = 32 → tree_height 3 = 4 := by
  intro h
  sorry

end tree_height_at_3_years_l151_15155


namespace area_of_square_l151_15188

theorem area_of_square (a : ℝ) (h : a = 12) : a * a = 144 := by
  rw [h]
  norm_num

end area_of_square_l151_15188


namespace necessary_but_not_sufficient_condition_l151_15108

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b : ℝ × ℝ := (2, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Statement: Prove x > 0 is a necessary but not sufficient condition for the angle between vectors a and b to be acute.
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (dot_product (vector_a x) vector_b > 0) ↔ (x > 0) := 
sorry

end necessary_but_not_sufficient_condition_l151_15108


namespace find_c_l151_15112

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem find_c (a b m c : ℝ) (h1 : ∀ x, f x a b ≥ 0)
  (h2 : ∀ x, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
by
  sorry

end find_c_l151_15112


namespace sum_of_arithmetic_sequence_15_terms_l151_15101

/-- An arithmetic sequence starts at 3 and has a common difference of 4.
    Prove that the sum of the first 15 terms of this sequence is 465. --/
theorem sum_of_arithmetic_sequence_15_terms :
  let a := 3
  let d := 4
  let n := 15
  let aₙ := a + (n - 1) * d
  (n / 2) * (a + aₙ) = 465 :=
by
  sorry

end sum_of_arithmetic_sequence_15_terms_l151_15101


namespace central_number_l151_15107

theorem central_number (C : ℕ) (verts : Finset ℕ) (h : verts = {1, 2, 7, 8, 9, 13, 14}) :
  (∀ T ∈ {t | ∃ a b c, (a + b + c) % 3 = 0 ∧ a ∈ verts ∧ b ∈ verts ∧ c ∈ verts}, (T + C) % 3 = 0) →
  C = 9 :=
by
  sorry

end central_number_l151_15107


namespace money_left_l151_15170

noncomputable def olivia_money : ℕ := 112
noncomputable def nigel_money : ℕ := 139
noncomputable def ticket_cost : ℕ := 28
noncomputable def num_tickets : ℕ := 6

theorem money_left : (olivia_money + nigel_money - ticket_cost * num_tickets) = 83 :=
by
  sorry

end money_left_l151_15170


namespace discount_percentage_l151_15134

theorem discount_percentage (CP MP SP D : ℝ) (cp_value : CP = 100) 
(markup : MP = CP + 0.5 * CP) (profit : SP = CP + 0.35 * CP) 
(discount : D = MP - SP) : (D / MP) * 100 = 10 := 
by 
  sorry

end discount_percentage_l151_15134


namespace sunflower_cans_l151_15148

theorem sunflower_cans (total_seeds seeds_per_can : ℕ) (h_total_seeds : total_seeds = 54) (h_seeds_per_can : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end sunflower_cans_l151_15148


namespace distance_between_A_and_B_is_40_l151_15126

theorem distance_between_A_and_B_is_40
  (v1 v2 : ℝ)
  (h1 : ∃ t: ℝ, t = (40 / 2) / v1 ∧ t = (40 - 24) / v2)
  (h2 : ∃ t: ℝ, t = (40 - 15) / v1 ∧ t = 40 / (2 * v2)) :
  40 = 40 := by
  sorry

end distance_between_A_and_B_is_40_l151_15126


namespace find_positive_number_l151_15146

theorem find_positive_number (x : ℝ) (hx : 0 < x) (h : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := by
  sorry

end find_positive_number_l151_15146


namespace smallest_bisecting_segment_l151_15140

-- Define a structure for a triangle in a plane
structure Triangle (α β γ : Type u) :=
(vertex1 : α) 
(vertex2 : β) 
(vertex3 : γ) 
(area : ℝ)

-- Define a predicate for an excellent line
def is_excellent_line {α β γ : Type u} (T : Triangle α β γ) (A : α) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line excellent here, e.g., dividing area in half
sorry

-- Define a function to get the length of a line segment within the triangle
def length_within_triangle {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : ℝ :=
-- compute the length of the segment within the triangle
sorry

-- Define predicates for triangles with specific properties like medians
def is_median {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line a median
sorry

theorem smallest_bisecting_segment {α β γ : Type u} (T : Triangle α β γ) (A : α) (median : ℝ → ℝ → ℝ) : 
  (∀ line, is_excellent_line T A line → length_within_triangle T line ≥ length_within_triangle T median) →
  median = line  := 
-- show that the median from the vertex opposite the smallest angle has the smallest segment
sorry

end smallest_bisecting_segment_l151_15140


namespace system_of_equations_has_integer_solutions_l151_15199

theorem system_of_equations_has_integer_solutions (a b : ℤ) :
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_of_equations_has_integer_solutions_l151_15199


namespace arithmetic_geometric_mean_inequality_l151_15187

open BigOperators

noncomputable def A (a : Fin n → ℝ) : ℝ := (Finset.univ.sum a) / n

noncomputable def G (a : Fin n → ℝ) : ℝ := (Finset.univ.prod a) ^ (1 / n)

theorem arithmetic_geometric_mean_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) : A a ≥ G a :=
  sorry

end arithmetic_geometric_mean_inequality_l151_15187


namespace percentage_calculation_l151_15178

variable (x : Real)
variable (hx : x > 0)

theorem percentage_calculation : 
  ∃ p : Real, p = (0.18 * x) / (x + 20) * 100 :=
sorry

end percentage_calculation_l151_15178


namespace a_gt_b_l151_15179

variable (n : ℕ) (a b : ℝ)
variable (n_pos : n > 1) (a_pos : 0 < a) (b_pos : 0 < b)
variable (a_eqn : a^n = a + 1)
variable (b_eqn : b^{2 * n} = b + 3 * a)

theorem a_gt_b : a > b :=
by {
  -- Proof is needed here
  sorry
}

end a_gt_b_l151_15179


namespace trader_sold_80_meters_l151_15180

variable (x : ℕ)
variable (selling_price_per_meter profit_per_meter cost_price_per_meter total_selling_price : ℕ)

theorem trader_sold_80_meters
  (h_cost_price : cost_price_per_meter = 118)
  (h_profit : profit_per_meter = 7)
  (h_selling_price : selling_price_per_meter = cost_price_per_meter + profit_per_meter)
  (h_total_selling_price : total_selling_price = 10000)
  (h_eq : selling_price_per_meter * x = total_selling_price) :
  x = 80 := by
    sorry

end trader_sold_80_meters_l151_15180


namespace percentage_of_money_spent_is_80_l151_15166

-- Define the cost of items
def cheeseburger_cost : ℕ := 3
def milkshake_cost : ℕ := 5
def cheese_fries_cost : ℕ := 8

-- Define the amount of money Jim and his cousin brought
def jim_money : ℕ := 20
def cousin_money : ℕ := 10

-- Define the total cost of the meal
def total_cost : ℕ :=
  2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money they brought
def combined_money : ℕ := jim_money + cousin_money

-- Define the percentage of combined money spent
def percentage_spent : ℕ :=
  (total_cost * 100) / combined_money

theorem percentage_of_money_spent_is_80 :
  percentage_spent = 80 :=
by
  -- proof goes here
  sorry

end percentage_of_money_spent_is_80_l151_15166


namespace sequence_two_cases_l151_15198

noncomputable def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≥ a (n-1)) ∧  -- nondecreasing
  (∃ n m, n ≠ m ∧ a n ≠ a m) ∧  -- nonconstant
  (∀ n, a n ∣ n^2)  -- a_n | n^2

theorem sequence_two_cases (a : ℕ → ℕ) :
  sequence_property a →
  (∃ n1, ∀ n ≥ n1, a n = n) ∨ (∃ n2, ∀ n ≥ n2, a n = n^2) :=
by {
  sorry
}

end sequence_two_cases_l151_15198


namespace smallest_number_of_rectangles_needed_l151_15193

-- Define the dimensions of the rectangle
def rectangle_area (length width : ℕ) : ℕ := length * width

-- Define the side length of the square
def square_side_length : ℕ := 12

-- Define the number of rectangles needed to cover the square horizontally
def num_rectangles_to_cover_square : ℕ := (square_side_length / 3) * (square_side_length / 4)

-- The theorem must state the total number of rectangles required
theorem smallest_number_of_rectangles_needed : num_rectangles_to_cover_square = 16 := 
by
  -- Proof details are skipped using sorry
  sorry

end smallest_number_of_rectangles_needed_l151_15193


namespace smallest_b_l151_15139

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7) (h4 : 2 + a ≤ b) : b = 9 / 2 :=
by
  sorry

end smallest_b_l151_15139


namespace length_of_other_parallel_side_l151_15130

theorem length_of_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : 323 = 1/2 * (20 + b) * 17) :
  b = 18 :=
sorry

end length_of_other_parallel_side_l151_15130


namespace find_A_l151_15195

variable (x ω φ b A : ℝ)

-- Given conditions
axiom cos_squared_eq : 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b
axiom A_gt_zero : A > 0

-- Lean 4 statement to prove
theorem find_A : A = Real.sqrt 2 :=
by
  sorry

end find_A_l151_15195


namespace perpendicular_slope_l151_15192

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end perpendicular_slope_l151_15192


namespace colorful_triangle_in_complete_graph_l151_15175

open SimpleGraph

theorem colorful_triangle_in_complete_graph (n : ℕ) (h : n ≥ 3) (colors : Fin n → Fin n → Fin (n - 1)) :
  ∃ (u v w : Fin n), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ colors u v ≠ colors v w ∧ colors v w ≠ colors w u ∧ colors w u ≠ colors u v :=
  sorry

end colorful_triangle_in_complete_graph_l151_15175


namespace boys_without_calculators_l151_15191

theorem boys_without_calculators (total_boys total_students students_with_calculators girls_with_calculators : ℕ) 
    (h1 : total_boys = 20) 
    (h2 : total_students = 40) 
    (h3 : students_with_calculators = 30) 
    (h4 : girls_with_calculators = 18) : 
    (total_boys - (students_with_calculators - girls_with_calculators)) = 8 :=
by
  sorry

end boys_without_calculators_l151_15191


namespace problem_1_problem_2_l151_15194

noncomputable def f (x a : ℝ) : ℝ := abs x + 2 * abs (x - a)

theorem problem_1 (x : ℝ) : (f x 1 ≤ 4) ↔ (- 2 / 3 ≤ x ∧ x ≤ 2) := 
sorry

theorem problem_2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ (4 ≤ a) := 
sorry

end problem_1_problem_2_l151_15194


namespace solve_trig_eq_l151_15159

noncomputable def arccos (x : ℝ) : ℝ := sorry

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  -3 * (Real.cos x) ^ 2 + 5 * (Real.sin x) + 1 = 0 ↔
  (x = Real.arcsin (1 / 3) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (1 / 3) + 2 * k * Real.pi) :=
sorry

end solve_trig_eq_l151_15159


namespace three_digit_cubes_divisible_by_8_l151_15103

theorem three_digit_cubes_divisible_by_8 : ∃ (S : Finset ℕ), S.card = 2 ∧ ∀ x ∈ S, x ^ 3 ≥ 100 ∧ x ^ 3 ≤ 999 ∧ x ^ 3 % 8 = 0 :=
by
  sorry

end three_digit_cubes_divisible_by_8_l151_15103


namespace combined_time_in_pool_l151_15167

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end combined_time_in_pool_l151_15167


namespace average_bmi_is_correct_l151_15141

-- Define Rachel's parameters
def rachel_weight : ℕ := 75
def rachel_height : ℕ := 60  -- in inches

-- Define Jimmy's parameters based on the conditions
def jimmy_weight : ℕ := rachel_weight + 6
def jimmy_height : ℕ := rachel_height + 3

-- Define Adam's parameters based on the conditions
def adam_weight : ℕ := rachel_weight - 15
def adam_height : ℕ := rachel_height - 2

-- Define the BMI formula
def bmi (weight : ℕ) (height : ℕ) : ℚ := (weight * 703 : ℚ) / (height * height)

-- Rachel's, Jimmy's, and Adam's BMIs
def rachel_bmi : ℚ := bmi rachel_weight rachel_height
def jimmy_bmi : ℚ := bmi jimmy_weight jimmy_height
def adam_bmi : ℚ := bmi adam_weight adam_height

-- Proving the average BMI
theorem average_bmi_is_correct : 
  (rachel_bmi + jimmy_bmi + adam_bmi) / 3 = 13.85 := 
by
  sorry

end average_bmi_is_correct_l151_15141


namespace theater_total_seats_l151_15173

theorem theater_total_seats
  (occupied_seats : ℕ) (empty_seats : ℕ) 
  (h1 : occupied_seats = 532) (h2 : empty_seats = 218) :
  occupied_seats + empty_seats = 750 := 
by
  -- This is the placeholder for the proof
  sorry

end theater_total_seats_l151_15173


namespace max_E_l151_15121

def E (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  x₁ + x₂ + x₃ + x₄ -
  x₁ * x₂ - x₁ * x₃ - x₁ * x₄ -
  x₂ * x₃ - x₂ * x₄ - x₃ * x₄ +
  x₁ * x₂ * x₃ + x₁ * x₂ * x₄ +
  x₁ * x₃ * x₄ + x₂ * x₃ * x₄ -
  x₁ * x₂ * x₃ * x₄

theorem max_E (x₁ x₂ x₃ x₄ : ℝ) (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : 0 ≤ x₂) (h₄ : x₂ ≤ 1) (h₅ : 0 ≤ x₃) (h₆ : x₃ ≤ 1) (h₇ : 0 ≤ x₄) (h₈ : x₄ ≤ 1) : 
  E x₁ x₂ x₃ x₄ ≤ 1 :=
sorry

end max_E_l151_15121


namespace inequality_abc_l151_15154

theorem inequality_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1/a + 1/(b * c)) * (1/b + 1/(c * a)) * (1/c + 1/(a * b)) ≥ 1728 :=
by sorry

end inequality_abc_l151_15154


namespace total_spent_target_l151_15106

theorem total_spent_target (face_moisturizer_cost : ℕ) (body_lotion_cost : ℕ) (face_moisturizers_bought : ℕ) (body_lotions_bought : ℕ) (christy_multiplier : ℕ) :
  face_moisturizer_cost = 50 →
  body_lotion_cost = 60 →
  face_moisturizers_bought = 2 →
  body_lotions_bought = 4 →
  christy_multiplier = 2 →
  (face_moisturizers_bought * face_moisturizer_cost + body_lotions_bought * body_lotion_cost) * (1 + christy_multiplier) = 1020 := by
  sorry

end total_spent_target_l151_15106


namespace sally_earnings_l151_15100

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l151_15100


namespace convex_m_gons_two_acute_angles_l151_15161

noncomputable def count_convex_m_gons_with_two_acute_angles (m n : ℕ) (P : Finset ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem convex_m_gons_two_acute_angles {m n : ℕ} {P : Finset ℕ}
  (hP : P.card = 2 * n + 1)
  (hmn : 4 < m ∧ m < n) :
  count_convex_m_gons_with_two_acute_angles m n P = 
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
sorry

end convex_m_gons_two_acute_angles_l151_15161


namespace parabola_no_intersect_l151_15142

theorem parabola_no_intersect (m : ℝ) : 
  (¬ ∃ x : ℝ, -x^2 - 6*x + m = 0 ) ↔ m < -9 :=
by
  sorry

end parabola_no_intersect_l151_15142


namespace circle_area_l151_15109

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = (2 * r) ^ 2) : π * r ^ 2 = π ^ (1 / 3) :=
by
  sorry

end circle_area_l151_15109


namespace find_total_people_find_children_l151_15151

variables (x m : ℕ)

-- Given conditions translated into Lean

def group_b_more_people (x : ℕ) := x + 4
def sum_is_18_times_difference (x : ℕ) := (x + (x + 4)) = 18 * ((x + 4) - x)
def children_b_less_than_three_times (m : ℕ) := (3 * m) - 2
def adult_ticket_price := 100
def children_ticket_price := (100 * 60) / 100
def same_amount_spent (x m : ℕ) := 100 * (x - m) + (100 * 60 / 100) * m = 100 * ((group_b_more_people x) - (children_b_less_than_three_times m)) + (100 * 60 / 100) * (children_b_less_than_three_times m)

-- Proving the two propositions (question == answer given conditions)

theorem find_total_people (x : ℕ) (hx : sum_is_18_times_difference x) : x = 34 ∧ (group_b_more_people x) = 38 :=
by {
  sorry -- proof for x = 34 and group_b_people = 38 given that sum_is_18_times_difference x
}

theorem find_children (m : ℕ) (x : ℕ) (hx : sum_is_18_times_difference x) (hm : same_amount_spent x m) : m = 6 ∧ (children_b_less_than_three_times m) = 16 :=
by {
  sorry -- proof for m = 6 and children_b_people = 16 given sum_is_18_times_difference x and same_amount_spent x m
}

end find_total_people_find_children_l151_15151


namespace secondChapterPages_is_18_l151_15104

-- Define conditions as variables and constants
def thirdChapterPages : ℕ := 3
def additionalPages : ℕ := 15

-- The main statement to prove
theorem secondChapterPages_is_18 : (thirdChapterPages + additionalPages) = 18 := by
  -- Proof would go here, but we skip it with sorry
  sorry

end secondChapterPages_is_18_l151_15104


namespace repeating_decimal_division_l151_15158

def repeating_decimal_142857 : ℚ := 1 / 7
def repeating_decimal_2_857143 : ℚ := 20 / 7

theorem repeating_decimal_division :
  (repeating_decimal_142857 / repeating_decimal_2_857143) = 1 / 20 :=
by
  sorry

end repeating_decimal_division_l151_15158


namespace triangle_angle_sine_identity_l151_15160

theorem triangle_angle_sine_identity (A B C : ℝ) (n : ℤ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n + 1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) :=
by
  sorry

end triangle_angle_sine_identity_l151_15160


namespace cost_of_each_math_book_l151_15124

-- Define the given conditions
def total_books : ℕ := 90
def math_books : ℕ := 53
def history_books : ℕ := total_books - math_books
def history_book_cost : ℕ := 5
def total_price : ℕ := 397

-- The required theorem
theorem cost_of_each_math_book (M : ℕ) (H : 53 * M + history_books * history_book_cost = total_price) : M = 4 :=
by
  sorry

end cost_of_each_math_book_l151_15124


namespace smallest_integer_condition_l151_15196

theorem smallest_integer_condition :
  ∃ (x : ℕ) (d : ℕ) (n : ℕ) (p : ℕ), x = 1350 ∧ d = 1 ∧ n = 450 ∧ p = 2 ∧
  x = 10^p * d + n ∧
  n = x / 19 ∧
  (1 ≤ d ∧ d ≤ 9 ∧ 10^p * d % 18 = 0) :=
sorry

end smallest_integer_condition_l151_15196


namespace find_a_for_inverse_proportion_l151_15136

theorem find_a_for_inverse_proportion (a : ℝ)
  (h_A : ∃ k : ℝ, 4 = k / (-1))
  (h_B : ∃ k : ℝ, 2 = k / a) :
  a = -2 :=
sorry

end find_a_for_inverse_proportion_l151_15136


namespace smallest_x_multiple_of_53_l151_15197

theorem smallest_x_multiple_of_53 :
  ∃ (x : ℕ), (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
sorry

end smallest_x_multiple_of_53_l151_15197


namespace cylinder_radius_range_l151_15115

theorem cylinder_radius_range :
  (V : ℝ) → (h : ℝ) → (r : ℝ) →
  V = 20 * Real.pi →
  h = 2 →
  (V = Real.pi * r^2 * h) →
  3 < r ∧ r < 4 :=
by
  -- Placeholder for the proof
  intro V h r hV hh hV_eq
  sorry

end cylinder_radius_range_l151_15115


namespace least_positive_integer_divisible_by_three_primes_l151_15176

-- Define the next three distinct primes larger than 5
def prime1 := 7
def prime2 := 11
def prime3 := 13

-- Define the product of these primes
def prod := prime1 * prime2 * prime3

-- Statement of the theorem
theorem least_positive_integer_divisible_by_three_primes : prod = 1001 :=
by
  sorry

end least_positive_integer_divisible_by_three_primes_l151_15176


namespace merchant_markup_percentage_l151_15102

theorem merchant_markup_percentage (CP MP SP : ℝ) (x : ℝ) (H_CP : CP = 100)
  (H_MP : MP = CP + (x / 100 * CP)) 
  (H_SP_discount : SP = MP * 0.80) 
  (H_SP_profit : SP = CP * 1.12) : 
  x = 40 := 
by
  sorry

end merchant_markup_percentage_l151_15102


namespace point_in_fourth_quadrant_l151_15157

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l151_15157


namespace tan_22_5_expression_l151_15190

theorem tan_22_5_expression :
  let a := 2
  let b := 1
  let c := 0
  let d := 0
  let t := Real.tan (Real.pi / 8)
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  t = (Real.sqrt a) - (Real.sqrt b) + (Real.sqrt c) - d → 
  a + b + c + d = 3 :=
by
  intros
  exact sorry

end tan_22_5_expression_l151_15190


namespace cost_for_33_people_employees_for_14000_cost_l151_15135

-- Define the conditions for pricing
def price_per_ticket (x : Nat) : Int :=
  if x ≤ 30 then 400
  else max 280 (400 - 5 * (x - 30))

def total_cost (x : Nat) : Int :=
  x * price_per_ticket x

-- Problem Part 1: Proving the total cost for 33 people
theorem cost_for_33_people :
  total_cost 33 = 12705 :=
by
  sorry

-- Problem Part 2: Given a total cost of 14000, finding the number of employees
theorem employees_for_14000_cost :
  ∃ x : Nat, total_cost x = 14000 ∧ price_per_ticket x ≥ 280 :=
by
  sorry

end cost_for_33_people_employees_for_14000_cost_l151_15135


namespace sara_basketball_loss_l151_15174

theorem sara_basketball_loss (total_games : ℕ) (games_won : ℕ) (games_lost : ℕ) 
  (h1 : total_games = 16) 
  (h2 : games_won = 12) 
  (h3 : games_lost = total_games - games_won) : 
  games_lost = 4 :=
by
  sorry

end sara_basketball_loss_l151_15174


namespace value_of_g_neg2_l151_15114

-- Define the function g as given in the conditions
def g (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Statement of the problem: Prove that g(-2) = 11
theorem value_of_g_neg2 : g (-2) = 11 := by
  sorry

end value_of_g_neg2_l151_15114


namespace cube_with_holes_l151_15111

-- Definitions and conditions
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def depth_hole : ℝ := 1
def number_of_holes : ℕ := 6

-- Prove that the total surface area including inside surfaces is 144 square meters
def total_surface_area_including_inside_surfaces : ℝ :=
  let original_surface_area := 6 * (edge_length_cube ^ 2)
  let area_removed_per_hole := side_length_hole ^ 2
  let area_exposed_inside_per_hole := 2 * (side_length_hole * depth_hole) + area_removed_per_hole
  original_surface_area - number_of_holes * area_removed_per_hole + number_of_holes * area_exposed_inside_per_hole

-- Prove that the total volume of material removed is 24 cubic meters
def total_volume_removed : ℝ :=
  number_of_holes * (side_length_hole ^ 2 * depth_hole)

theorem cube_with_holes :
  total_surface_area_including_inside_surfaces = 144 ∧ total_volume_removed = 24 :=
by
  sorry

end cube_with_holes_l151_15111


namespace boat_travel_difference_l151_15144

-- Define the speeds
variables (a b : ℝ) (ha : a > b)

-- Define the travel times
def downstream_time := 3
def upstream_time := 2

-- Define the distances
def downstream_distance := downstream_time * (a + b)
def upstream_distance := upstream_time * (a - b)

-- Prove the mathematical statement
theorem boat_travel_difference : downstream_distance a b - upstream_distance a b = a + 5 * b := by
  -- sorry can be used to skip the proof
  sorry

end boat_travel_difference_l151_15144


namespace additionalPeopleNeededToMowLawn_l151_15153

def numberOfPeopleNeeded (people : ℕ) (hours : ℕ) : ℕ :=
  (people * 8) / hours

theorem additionalPeopleNeededToMowLawn : numberOfPeopleNeeded 4 3 - 4 = 7 :=
by
  sorry

end additionalPeopleNeededToMowLawn_l151_15153


namespace g_diff_eq_neg8_l151_15183

noncomputable def g : ℝ → ℝ := sorry

axiom linear_g : ∀ x y : ℝ, g (x + y) = g x + g y

axiom condition_g : ∀ x : ℝ, g (x + 2) - g x = 4

theorem g_diff_eq_neg8 : g 2 - g 6 = -8 :=
by
  sorry

end g_diff_eq_neg8_l151_15183


namespace prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l151_15117

def prob_has_bio_test : ℚ := 5 / 8
def prob_not_has_chem_test : ℚ := 1 / 2

theorem prob_not_has_bio_test : 1 - 5 / 8 = 3 / 8 := by
  sorry

theorem combined_prob_neither_bio_nor_chem :
  (1 - 5 / 8) * (1 / 2) = 3 / 16 := by
  sorry

end prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l151_15117


namespace taylor_family_reunion_adults_l151_15119

def number_of_kids : ℕ := 45
def number_of_tables : ℕ := 14
def people_per_table : ℕ := 12
def total_people := number_of_tables * people_per_table

theorem taylor_family_reunion_adults : total_people - number_of_kids = 123 := by
  sorry

end taylor_family_reunion_adults_l151_15119


namespace eight_digit_not_perfect_square_l151_15116

theorem eight_digit_not_perfect_square : ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9999 → ¬ ∃ y : ℤ, (99990000 + x) = y * y := 
by
  intros x hx
  intro h
  obtain ⟨y, hy⟩ := h
  sorry

end eight_digit_not_perfect_square_l151_15116


namespace sandy_correct_sums_l151_15149

variable (c i : ℕ)

theorem sandy_correct_sums (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by {
  -- Proof goes here
  sorry
}

end sandy_correct_sums_l151_15149


namespace robin_uploaded_pics_from_camera_l151_15145

-- Definitions of the conditions
def pics_from_phone := 35
def albums := 5
def pics_per_album := 8

-- The statement we want to prove
theorem robin_uploaded_pics_from_camera : (albums * pics_per_album) - pics_from_phone = 5 :=
by
  -- Proof goes here
  sorry

end robin_uploaded_pics_from_camera_l151_15145
