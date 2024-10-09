import Mathlib

namespace harris_flour_amount_l323_32384

noncomputable def flour_needed_by_cakes (cakes : ℕ) : ℕ := cakes * 100

noncomputable def traci_flour : ℕ := 500

noncomputable def total_cakes : ℕ := 9

theorem harris_flour_amount : flour_needed_by_cakes total_cakes - traci_flour = 400 := 
by
  sorry

end harris_flour_amount_l323_32384


namespace zongzi_problem_l323_32387

def zongzi_prices : Prop :=
  ∀ (x y : ℕ), -- x: price of red bean zongzi, y: price of meat zongzi
  10 * x + 12 * y = 136 → -- total cost for the first customer
  y = 2 * x →
  x = 4 ∧ y = 8 -- prices found

def discounted_zongzi_prices : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  20 * a + 30 * b = 270 → -- cost for Xiaohuan's mother
  30 * a + 20 * b = 230 → -- cost for Xiaole's mother
  a = 3 ∧ b = 7 -- discounted prices found

def zongzi_packages (m : ℕ) : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  a = 3 → b = 7 →
  (80 - 4 * m) * (m * a + (40 - m) * b) + (4 * m + 8) * ((40 - m) * a + m * b) = 17280 →
  m ≤ 20 / 2 → -- quantity constraint
  m = 10 -- final m value

-- Statement to prove all together
theorem zongzi_problem :
  zongzi_prices ∧ discounted_zongzi_prices ∧ ∃ (m : ℕ), zongzi_packages m :=
by sorry

end zongzi_problem_l323_32387


namespace roots_quartic_sum_l323_32346

theorem roots_quartic_sum (p q r : ℝ) 
  (h1 : p^3 - 2*p^2 + 3*p - 4 = 0)
  (h2 : q^3 - 2*q^2 + 3*q - 4 = 0)
  (h3 : r^3 - 2*r^2 + 3*r - 4 = 0)
  (h4 : p + q + r = 2)
  (h5 : p*q + q*r + r*p = 3)
  (h6 : p*q*r = 4) :
  p^4 + q^4 + r^4 = 18 := sorry

end roots_quartic_sum_l323_32346


namespace fraction_problem_l323_32342

theorem fraction_problem :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_problem_l323_32342


namespace remainder_of_349_by_17_is_9_l323_32362

theorem remainder_of_349_by_17_is_9 :
  349 % 17 = 9 :=
sorry

end remainder_of_349_by_17_is_9_l323_32362


namespace abs_lt_five_implies_interval_l323_32355

theorem abs_lt_five_implies_interval (x : ℝ) : |x| < 5 → -5 < x ∧ x < 5 := by
  sorry

end abs_lt_five_implies_interval_l323_32355


namespace problem_1_split_terms_problem_2_split_terms_l323_32343

-- Problem 1 Lean statement
theorem problem_1_split_terms :
  (28 + 5/7) + (-25 - 1/7) = 3 + 4/7 := 
  sorry
  
-- Problem 2 Lean statement
theorem problem_2_split_terms :
  (-2022 - 2/7) + (-2023 - 4/7) + 4046 - 1/7 = 0 := 
  sorry

end problem_1_split_terms_problem_2_split_terms_l323_32343


namespace value_of_2p_plus_q_l323_32374

theorem value_of_2p_plus_q (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p :=
by
  sorry

end value_of_2p_plus_q_l323_32374


namespace selene_total_payment_l323_32399

def price_instant_camera : ℝ := 110
def num_instant_cameras : ℕ := 2
def discount_instant_camera : ℝ := 0.07
def price_photo_frame : ℝ := 120
def num_photo_frames : ℕ := 3
def discount_photo_frame : ℝ := 0.05
def sales_tax : ℝ := 0.06

theorem selene_total_payment :
  let total_instant_cameras := num_instant_cameras * price_instant_camera
  let discount_instant := total_instant_cameras * discount_instant_camera
  let discounted_instant := total_instant_cameras - discount_instant
  let total_photo_frames := num_photo_frames * price_photo_frame
  let discount_photo := total_photo_frames * discount_photo_frame
  let discounted_photo := total_photo_frames - discount_photo
  let subtotal := discounted_instant + discounted_photo
  let tax := subtotal * sales_tax
  let total_payment := subtotal + tax
  total_payment = 579.40 :=
by
  sorry

end selene_total_payment_l323_32399


namespace find_parallel_line_l323_32308

def line1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y + 2 = 0
def parallelLine : ℝ → ℝ → Prop := λ x y => 4 * x + y - 4 = 0

theorem find_parallel_line (x y : ℝ) (hx : line1 x y) (hy : line2 x y) : 
  ∃ c : ℝ, (λ x y => 4 * x + y + c = 0) (2:ℝ) (2:ℝ) ∧ 
          ∀ x' y', (λ x' y' => 4 * x' + y' + c = 0) x' y' ↔ 4 * x' + y' - 10 = 0 := 
sorry

end find_parallel_line_l323_32308


namespace problem_statement_l323_32396

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def pow_log2 (x : ℝ) : ℝ := x ^ log2 x

theorem problem_statement (a b c : ℝ)
  (h0 : 1 ≤ a)
  (h1 : 1 ≤ b)
  (h2 : 1 ≤ c)
  (h3 : a * b * c = 10)
  (h4 : pow_log2 a * pow_log2 b * pow_log2 c ≥ 10) :
  a + b + c = 12 := by
  sorry

end problem_statement_l323_32396


namespace tan_sum_identity_l323_32301

theorem tan_sum_identity (α : ℝ) (h : Real.tan α = 1 / 2) : Real.tan (α + π / 4) = 3 := 
by 
  sorry

end tan_sum_identity_l323_32301


namespace monotonic_decreasing_interval_of_f_l323_32398

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0 :=
by
  sorry

end monotonic_decreasing_interval_of_f_l323_32398


namespace rectangle_area_perimeter_l323_32303

/-- 
Given a rectangle with positive integer sides a and b,
let A be the area and P be the perimeter.

A = a * b
P = 2 * a + 2 * b

Prove that 100 cannot be expressed as A + P - 4.
-/
theorem rectangle_area_perimeter (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ)
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : 
  ¬ (A + P - 4 = 100) := 
sorry

end rectangle_area_perimeter_l323_32303


namespace seating_arrangements_l323_32339

theorem seating_arrangements (n : ℕ) (hn : n = 8) : 
  ∃ (k : ℕ), k = 5760 :=
by
  sorry

end seating_arrangements_l323_32339


namespace least_number_to_subtract_l323_32363

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (k : ℕ) (hk : 42398 % 15 = k) : k = 8 :=
by
  sorry

end least_number_to_subtract_l323_32363


namespace B_spends_85_percent_l323_32392

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 4000
def A_savings_percentage : ℝ := 0.05
def A_salary : ℝ := 3000
def B_salary : ℝ := 4000 - A_salary
def equal_savings (S_A S_B : ℝ) : Prop := A_savings_percentage * S_A = (1 - S_B / 100) * B_salary

theorem B_spends_85_percent (S_A S_B : ℝ) (B_spending_percentage : ℝ) :
  combined_salary S_A S_B ∧ S_A = A_salary ∧ equal_savings S_A B_spending_percentage → B_spending_percentage = 0.85 := by
  sorry

end B_spends_85_percent_l323_32392


namespace correct_condition_l323_32376

section proof_problem

variable (a : ℝ)

def cond1 : Prop := (a ^ 6 / a ^ 3 = a ^ 2)
def cond2 : Prop := (2 * a ^ 2 + 3 * a ^ 3 = 5 * a ^ 5)
def cond3 : Prop := (a ^ 4 * a ^ 2 = a ^ 8)
def cond4 : Prop := ((-a ^ 3) ^ 2 = a ^ 6)

theorem correct_condition : cond4 a :=
by
  sorry

end proof_problem

end correct_condition_l323_32376


namespace avg_books_rounded_l323_32385

def books_read : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4)]

noncomputable def total_books_read (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.fst * pair.snd) 0

noncomputable def total_members (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.snd) 0

noncomputable def average_books_read (books : List (ℕ × ℕ)) : ℤ :=
  Int.ofNat (total_books_read books) / Int.ofNat (total_members books)

theorem avg_books_rounded :
  average_books_read books_read = 3 :=
by 
  sorry

end avg_books_rounded_l323_32385


namespace VehicleB_travel_time_l323_32309

theorem VehicleB_travel_time 
    (v_A v_B : ℝ)
    (d : ℝ)
    (h1 : d = 3 * (v_A + v_B))
    (h2 : 3 * v_A = d / 2)
    (h3 : ∀ t ≤ 3.5 , d - t * v_B - 0.5 * v_A = 0)
    : d / v_B = 7.2 :=
by
  sorry

end VehicleB_travel_time_l323_32309


namespace average_difference_is_7_l323_32354

/-- The differences between Mia's and Liam's study times for each day in one week -/
def daily_differences : List ℤ := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week -/
def number_of_days : ℕ := 7

/-- The total difference over the week -/
def total_difference : ℤ := daily_differences.sum

/-- The average difference per day -/
def average_difference_per_day : ℚ := total_difference / number_of_days

theorem average_difference_is_7 : average_difference_per_day = 7 := by 
  sorry

end average_difference_is_7_l323_32354


namespace walter_age_at_2003_l323_32356

theorem walter_age_at_2003 :
  ∀ (w : ℕ),
  (1998 - w) + (1998 - 3 * w) = 3860 → 
  w + 5 = 39 :=
by
  intros w h
  sorry

end walter_age_at_2003_l323_32356


namespace round_310242_to_nearest_thousand_l323_32365

-- Define the conditions and the target statement
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  if (n % 1000) < 500 then (n / 1000) * 1000 else (n / 1000 + 1) * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 :=
by
  sorry

end round_310242_to_nearest_thousand_l323_32365


namespace line_through_point_bisects_chord_l323_32358

theorem line_through_point_bisects_chord 
  (x y : ℝ) 
  (h_parabola : y^2 = 16 * x) 
  (h_point : 8 * 2 - 1 - 15 = 0) :
  8 * x - y - 15 = 0 :=
by
  sorry

end line_through_point_bisects_chord_l323_32358


namespace irrational_sum_floor_eq_iff_l323_32316

theorem irrational_sum_floor_eq_iff (a b c d : ℝ) (h_irr_a : ¬ ∃ (q : ℚ), a = q) 
                                     (h_irr_b : ¬ ∃ (q : ℚ), b = q) 
                                     (h_irr_c : ¬ ∃ (q : ℚ), c = q) 
                                     (h_irr_d : ¬ ∃ (q : ℚ), d = q) 
                                     (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
                                     (h_pos_c : 0 < c) (h_pos_d : 0 < d)
                                     (h_sum_ab : a + b = 1) :
  (c + d = 1) ↔ (∀ (n : ℕ), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
sorry

end irrational_sum_floor_eq_iff_l323_32316


namespace min_varphi_symmetry_l323_32302

theorem min_varphi_symmetry (ϕ : ℝ) (hϕ : ϕ > 0) :
  (∃ k : ℤ, ϕ = (4 * Real.pi) / 3 - k * Real.pi ∧ ϕ > 0 ∧ (∀ x : ℝ, Real.cos (x - ϕ + (4 * Real.pi) / 3) = Real.cos (-x - ϕ + (4 * Real.pi) / 3))) 
  → ϕ = Real.pi / 3 :=
sorry

end min_varphi_symmetry_l323_32302


namespace inverse_var_q_value_l323_32323

theorem inverse_var_q_value (p q : ℝ) (h1 : ∀ p q, (p * q = 400))
(p_init : p = 800) (q_init : q = 0.5) (new_p : p = 400) :
  q = 1 := by
  sorry

end inverse_var_q_value_l323_32323


namespace probability_correct_l323_32386

def total_chips : ℕ := 15
def total_ways_to_draw_2_chips : ℕ := Nat.choose 15 2

def chips_same_color : ℕ := 3 * (Nat.choose 5 2)
def chips_same_number : ℕ := 5 * (Nat.choose 3 2)
def favorable_outcomes : ℕ := chips_same_color + chips_same_number

def probability_same_color_or_number : ℚ := favorable_outcomes / total_ways_to_draw_2_chips

theorem probability_correct :
  probability_same_color_or_number = 3 / 7 :=
by sorry

end probability_correct_l323_32386


namespace spencer_walked_distance_l323_32335

/-- Define the distances involved -/
def total_distance := 0.8
def library_to_post_office := 0.1
def post_office_to_home := 0.4

/-- Define the distance from house to library as a variable to calculate -/
def house_to_library := total_distance - library_to_post_office - post_office_to_home

/-- The theorem states that Spencer walked 0.3 miles from his house to the library -/
theorem spencer_walked_distance : 
  house_to_library = 0.3 :=
by
  -- Proof omitted
  sorry

end spencer_walked_distance_l323_32335


namespace chloe_profit_l323_32332

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end chloe_profit_l323_32332


namespace unique_solution_condition_l323_32326

noncomputable def unique_solution_system (a b c x y z : ℝ) : Prop :=
  (a * x + b * y - b * z = c) ∧ 
  (a * y + b * x - b * z = c) ∧ 
  (a * z + b * y - b * x = c) → 
  (x = y ∧ y = z ∧ x = c / a)

theorem unique_solution_condition (a b c x y z : ℝ) 
  (h1 : a * x + b * y - b * z = c)
  (h2 : a * y + b * x - b * z = c)
  (h3 : a * z + b * y - b * x = c)
  (ha : a ≠ 0)
  (ha_b : a ≠ b)
  (ha_b' : a + b ≠ 0) :
  unique_solution_system a b c x y z :=
by 
  sorry

end unique_solution_condition_l323_32326


namespace hyperbola_asymptote_slope_l323_32372

theorem hyperbola_asymptote_slope :
  ∀ {x y : ℝ}, (x^2 / 144 - y^2 / 81 = 1) → (∃ m : ℝ, ∀ x, y = m * x ∨ y = -m * x ∧ m = 3 / 4) :=
by
  sorry

end hyperbola_asymptote_slope_l323_32372


namespace find_fake_coin_in_two_weighings_l323_32330

theorem find_fake_coin_in_two_weighings (coins : Fin 8 → ℝ) (h : ∃ i : Fin 8, (∀ j ≠ i, coins i < coins j)) : 
  ∃! i : Fin 8, ∀ j ≠ i, coins i < coins j :=
by
  sorry

end find_fake_coin_in_two_weighings_l323_32330


namespace unique_prime_range_start_l323_32328

theorem unique_prime_range_start (N : ℕ) (hN : N = 220) (h1 : ∀ n, N ≥ n → n ≥ 211 → ¬Prime n) (h2 : Prime 211) : N - 8 = 212 :=
by
  sorry

end unique_prime_range_start_l323_32328


namespace f_at_10_l323_32306

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

-- Prove that f(10) = 756
theorem f_at_10 : f 10 = 756 := by
  sorry

end f_at_10_l323_32306


namespace sufficient_but_not_necessary_condition_for_q_l323_32318

theorem sufficient_but_not_necessary_condition_for_q (k : ℝ) :
  (∀ x : ℝ, x ≥ k → x^2 - x > 2) ∧ (∃ x : ℝ, x < k ∧ x^2 - x > 2) ↔ k > 2 :=
sorry

end sufficient_but_not_necessary_condition_for_q_l323_32318


namespace percentage_reduction_l323_32379

theorem percentage_reduction 
  (original_employees : ℝ)
  (new_employees : ℝ)
  (h1 : original_employees = 208.04597701149424)
  (h2 : new_employees = 181) :
  ((original_employees - new_employees) / original_employees) * 100 = 13.00 :=
by
  sorry

end percentage_reduction_l323_32379


namespace quarts_of_water_needed_l323_32325

-- Definitions of conditions
def total_parts := 5 + 2 + 1
def total_gallons := 3
def quarts_per_gallon := 4
def water_parts := 5

-- Lean proof statement
theorem quarts_of_water_needed :
  (water_parts : ℚ) * ((total_gallons * quarts_per_gallon) / total_parts) = 15 / 2 :=
by sorry

end quarts_of_water_needed_l323_32325


namespace find_distance_l323_32378

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_l323_32378


namespace totalShortBushes_l323_32391

namespace ProofProblem

def initialShortBushes : Nat := 37
def additionalShortBushes : Nat := 20

theorem totalShortBushes :
  initialShortBushes + additionalShortBushes = 57 := by
  sorry

end ProofProblem

end totalShortBushes_l323_32391


namespace range_of_a_l323_32375

theorem range_of_a 
{α : Type*} [LinearOrderedField α] (a : α) 
(h : ∃ x, x = 3 ∧ (x - a) * (x + 2 * a - 1) ^ 2 * (x - 3 * a) ≤ 0) :
a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l323_32375


namespace remaining_area_correct_l323_32317

noncomputable def remaining_area_ABHFGD : ℝ :=
  let area_square_ABCD := 25
  let area_square_EFGD := 16
  let side_length_ABCD := Real.sqrt area_square_ABCD
  let side_length_EFGD := Real.sqrt area_square_EFGD
  let overlap_area := 8
  area_square_ABCD + area_square_EFGD - overlap_area

theorem remaining_area_correct :
  let area := remaining_area_ABHFGD
  area = 33 :=
by
  sorry

end remaining_area_correct_l323_32317


namespace count_multiples_3_or_4_but_not_6_l323_32377

def multiples_between (m n k : Nat) : Nat :=
  (k / m) + (k / n) - (k / (m * n))

theorem count_multiples_3_or_4_but_not_6 :
  let count_multiples (d : Nat) := (3000 / d)
  let multiples_of_3 := count_multiples 3
  let multiples_of_4 := count_multiples 4
  let multiples_of_6 := count_multiples 6
  multiples_of_3 + multiples_of_4 - multiples_of_6 = 1250 := by
  sorry

end count_multiples_3_or_4_but_not_6_l323_32377


namespace min_value_expression_l323_32381

theorem min_value_expression : ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by
  intro x y
  sorry

end min_value_expression_l323_32381


namespace certain_event_proof_l323_32344

def Moonlight_in_front_of_bed := "depends_on_time_and_moon_position"
def Lonely_smoke_in_desert := "depends_on_specific_conditions"
def Reach_for_stars_with_hand := "physically_impossible"
def Yellow_River_flows_into_sea := "certain_event"

theorem certain_event_proof : Yellow_River_flows_into_sea = "certain_event" :=
by
  sorry

end certain_event_proof_l323_32344


namespace arithmetic_seq_sum_l323_32373

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h₁ : ∀ n k : ℕ, a (n + k) = a n + k * d) 
  (h₂ : a 5 + a 6 + a 7 + a 8 = 20) : a 1 + a 12 = 10 := 
by 
  sorry

end arithmetic_seq_sum_l323_32373


namespace triangle_angle_C_l323_32315

theorem triangle_angle_C (A B C : Real) (h1 : A - B = 10) (h2 : B = A / 2) :
  C = 150 :=
by
  -- Proof goes here
  sorry

end triangle_angle_C_l323_32315


namespace average_percentage_decrease_l323_32305

theorem average_percentage_decrease
  (original_price final_price : ℕ)
  (h_original_price : original_price = 2000)
  (h_final_price : final_price = 1280) :
  (original_price - final_price) / original_price * 100 / 2 = 18 :=
by 
  sorry

end average_percentage_decrease_l323_32305


namespace points_on_fourth_board_l323_32364

theorem points_on_fourth_board (P_1 P_2 P_3 P_4 : ℕ)
 (h1 : P_1 = 30)
 (h2 : P_2 = 38)
 (h3 : P_3 = 41) :
  P_4 = 34 :=
sorry

end points_on_fourth_board_l323_32364


namespace find_k_for_one_real_solution_l323_32345

theorem find_k_for_one_real_solution (k : ℤ) :
  (∀ x : ℤ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end find_k_for_one_real_solution_l323_32345


namespace time_for_c_l323_32329

   variable (A B C : ℚ)

   -- Conditions
   def condition1 : Prop := (A + B = 1/6)
   def condition2 : Prop := (B + C = 1/8)
   def condition3 : Prop := (C + A = 1/12)

   -- Theorem to be proved
   theorem time_for_c (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
     1 / C = 48 :=
   sorry
   
end time_for_c_l323_32329


namespace quadratic_function_expression_l323_32340

-- Definitions based on conditions
def quadratic (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
def condition1 (f : ℝ → ℝ) : Prop := (f 0 = 1)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) - f x = 4 * x

-- The theorem we want to prove
theorem quadratic_function_expression (f : ℝ → ℝ) 
  (hf_quad : quadratic f)
  (hf_cond1 : condition1 f)
  (hf_cond2 : condition2 f) : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -2 ∧ c = 1 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end quadratic_function_expression_l323_32340


namespace teacher_age_is_56_l323_32314

theorem teacher_age_is_56 (s t : ℝ) (h1 : s = 40 * 15) (h2 : s + t = 41 * 16) : t = 56 := by
  sorry

end teacher_age_is_56_l323_32314


namespace probability_exact_four_out_of_twelve_dice_is_approx_0_089_l323_32338

noncomputable def dice_probability_exact_four_six : ℝ :=
  let p := (1/6 : ℝ)
  let q := (5/6 : ℝ)
  (Nat.choose 12 4) * (p ^ 4) * (q ^ 8)

theorem probability_exact_four_out_of_twelve_dice_is_approx_0_089 :
  abs (dice_probability_exact_four_six - 0.089) < 0.001 :=
sorry

end probability_exact_four_out_of_twelve_dice_is_approx_0_089_l323_32338


namespace quadratic_ineq_solution_l323_32394

theorem quadratic_ineq_solution (x : ℝ) : x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := 
sorry

end quadratic_ineq_solution_l323_32394


namespace card_collection_problem_l323_32360

theorem card_collection_problem 
  (m : ℕ) 
  (h : (2 * m + 1) / 3 = 56) : 
  m = 84 :=
sorry

end card_collection_problem_l323_32360


namespace even_odd_decomposition_exp_l323_32336

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x
def decomposition (f g : ℝ → ℝ) := ∀ x, f x + g x = Real.exp x

-- Main statement to prove
theorem even_odd_decomposition_exp (hf : is_even f) (hg : is_odd g) (hfg : decomposition f g) :
  f (Real.log 2) + g (Real.log (1 / 2)) = 1 / 2 := 
sorry

end even_odd_decomposition_exp_l323_32336


namespace sum_first_100_odd_l323_32348

theorem sum_first_100_odd :
  (Finset.sum (Finset.range 100) (λ x => 2 * (x + 1) - 1)) = 10000 := by
  sorry

end sum_first_100_odd_l323_32348


namespace find_m_l323_32357

theorem find_m (m : ℝ) (h1 : ∀ x y : ℝ, (x ^ 2 + (y - 2) ^ 2 = 1) → (y = x / m ∨ y = -x / m)) (h2 : 0 < m) :
  m = (Real.sqrt 3) / 3 :=
by
  sorry

end find_m_l323_32357


namespace total_number_of_people_on_bus_l323_32337

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l323_32337


namespace equivalent_fraction_l323_32304

theorem equivalent_fraction (b : ℕ) (h : b = 2024) :
  (b^3 - 2 * b^2 * (b + 1) + 3 * b * (b + 1)^2 - (b + 1)^3 + 4) / (b * (b + 1)) = 2022 := by
  rw [h]
  sorry

end equivalent_fraction_l323_32304


namespace jason_seashells_after_giving_l323_32320

-- Define the number of seashells Jason originally found
def original_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given : ℕ := 13

-- Prove that the number of seashells Jason now has is 36
theorem jason_seashells_after_giving : original_seashells - seashells_given = 36 :=
by
  -- This is where the proof would go
  sorry

end jason_seashells_after_giving_l323_32320


namespace sum_three_distinct_zero_l323_32341

variable {R : Type} [Field R]

theorem sum_three_distinct_zero
  (a b c x y : R)
  (h1 : a ^ 3 + a * x + y = 0)
  (h2 : b ^ 3 + b * x + y = 0)
  (h3 : c ^ 3 + c * x + y = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  a + b + c = 0 := by
  sorry

end sum_three_distinct_zero_l323_32341


namespace mean_values_are_two_l323_32319

noncomputable def verify_means (a b : ℝ) : Prop :=
  (a + b) / 2 = 2 ∧ 2 / ((1 / a) + (1 / b)) = 2

theorem mean_values_are_two (a b : ℝ) (h : verify_means a b) : a = 2 ∧ b = 2 :=
  sorry

end mean_values_are_two_l323_32319


namespace negation_of_universal_l323_32390

-- Definitions based on the provided problem
def prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Main proof problem statement
theorem negation_of_universal : 
  ¬ (∀ x : ℝ, x > 0 → x^2 > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0 :=
by sorry

end negation_of_universal_l323_32390


namespace total_questions_reviewed_l323_32389

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l323_32389


namespace sector_radius_l323_32311

theorem sector_radius (r : ℝ) (h1 : r > 0) 
  (h2 : ∀ (l : ℝ), l = r → 
    (3 * r) / (1 / 2 * r^2) = 2) : r = 3 := 
sorry

end sector_radius_l323_32311


namespace investment_initial_amount_l323_32393

noncomputable def initialInvestment (final_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  final_amount / interest_rate^years

theorem investment_initial_amount :
  initialInvestment 705.73 1.12 5 = 400.52 := by
  sorry

end investment_initial_amount_l323_32393


namespace number_of_pizzas_ordered_l323_32310

-- Definitions from conditions
def slices_per_pizza : Nat := 2
def total_slices : Nat := 28

-- Proof that the number of pizzas ordered is 14
theorem number_of_pizzas_ordered : total_slices / slices_per_pizza = 14 := by
  sorry

end number_of_pizzas_ordered_l323_32310


namespace dartboard_odd_score_probability_l323_32397

theorem dartboard_odd_score_probability :
  let π := Real.pi
  let r_outer := 4
  let r_inner := 2
  let area_inner := π * r_inner * r_inner
  let area_outer := π * r_outer * r_outer
  let area_annulus := area_outer - area_inner
  let area_inner_region := area_inner / 3
  let area_outer_region := area_annulus / 3
  let odd_inner_regions := 1
  let even_inner_regions := 2
  let odd_outer_regions := 2
  let even_outer_regions := 1
  let prob_odd_inner := (odd_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_even_inner := (even_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_odd_outer := (odd_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_even_outer := (even_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_odd_region := prob_odd_inner + prob_odd_outer
  let prob_even_region := prob_even_inner + prob_even_outer
  let prob_odd_score := (prob_odd_region * prob_even_region) + (prob_even_region * prob_odd_region)
  prob_odd_score = 5 / 9 :=
by
  -- Proof omitted
  sorry

end dartboard_odd_score_probability_l323_32397


namespace final_cost_is_correct_l323_32359

noncomputable def calculate_final_cost 
  (price_orange : ℕ)
  (price_mango : ℕ)
  (increase_percent : ℕ)
  (bulk_discount_percent : ℕ)
  (sales_tax_percent : ℕ) : ℕ := 
  let new_price_orange := price_orange + (price_orange * increase_percent) / 100
  let new_price_mango := price_mango + (price_mango * increase_percent) / 100
  let total_cost_oranges := 10 * new_price_orange
  let total_cost_mangoes := 10 * new_price_mango
  let total_cost_before_discount := total_cost_oranges + total_cost_mangoes
  let discount_oranges := (total_cost_oranges * bulk_discount_percent) / 100
  let discount_mangoes := (total_cost_mangoes * bulk_discount_percent) / 100
  let total_cost_after_discount := total_cost_before_discount - discount_oranges - discount_mangoes
  let sales_tax := (total_cost_after_discount * sales_tax_percent) / 100
  total_cost_after_discount + sales_tax

theorem final_cost_is_correct :
  calculate_final_cost 40 50 15 10 8 = 100602 :=
by
  sorry

end final_cost_is_correct_l323_32359


namespace find_divisor_l323_32349

theorem find_divisor (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 :=
by
  sorry

end find_divisor_l323_32349


namespace domain_of_g_l323_32380

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊ x^2 - 9 * x + 21 ⌋

theorem domain_of_g :
  { x : ℝ | ∃ y : ℝ, g x = y } = { x : ℝ | x ≤ 4 ∨ x ≥ 5 } :=
by
  sorry

end domain_of_g_l323_32380


namespace son_age_is_26_l323_32367

-- Definitions based on conditions in the problem
variables (S F : ℕ)
axiom cond1 : F = S + 28
axiom cond2 : F + 2 = 2 * (S + 2)

-- Statement to prove that S = 26
theorem son_age_is_26 : S = 26 :=
by 
  -- Proof steps go here
  sorry

end son_age_is_26_l323_32367


namespace find_special_n_l323_32334

open Nat

theorem find_special_n (m : ℕ) (hm : m ≥ 3) :
  ∃ (n : ℕ), 
    (n = m^2 - 2) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k < n ∧ 2 * (Nat.choose n k) = (Nat.choose n (k - 1) + Nat.choose n (k + 1))) :=
by
  sorry

end find_special_n_l323_32334


namespace range_of_a_l323_32361

theorem range_of_a (m : ℝ) (a : ℝ) : 
  m ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  (∀ x₁ x₂ : ℝ, x₁^2 - m * x₁ - 2 = 0 ∧ x₂^2 - m * x₂ - 2 = 0 → a^2 - 5 * a - 3 ≥ |x₁ - x₂|) ↔ (a ≥ 6 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l323_32361


namespace find_m_range_of_x_l323_32395

def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3 * m + 2

theorem find_m (m : ℝ) (H_dec : m^2 - 1 < 0) (H_f1 : f m 1 = 0) : 
  m = 1 / 2 :=
sorry

theorem range_of_x (x : ℝ) :
  f (1 / 2) (x + 1) ≥ x^2 ↔ -3 / 4 ≤ x ∧ x ≤ 0 :=
sorry

end find_m_range_of_x_l323_32395


namespace find_A_from_equation_l323_32352

variable (A B C D : ℕ)
variable (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (eq1 : A * 1000 + B * 100 + 82 - 900 + C * 10 + 9 = 4000 + 900 + 30 + D)

theorem find_A_from_equation (A B C D : ℕ) (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq1 : A * 1000 + B * 100 + 82 - (900 + C * 10 + 9) = 4000 + 900 + 30 + D) : A = 5 :=
by sorry

end find_A_from_equation_l323_32352


namespace geometric_series_S_n_div_a_n_l323_32351

-- Define the conditions and the properties of the geometric sequence
variables (a_3 a_5 a_4 a_6 S_n a_n : ℝ) (n : ℕ)
variable (q : ℝ) -- common ratio of the geometric sequence

-- Conditions given in the problem
axiom h1 : a_3 + a_5 = 5 / 4
axiom h2 : a_4 + a_6 = 5 / 8

-- The value we want to prove
theorem geometric_series_S_n_div_a_n : 
  (a_3 + a_5) * q = 5 / 8 → 
  q = 1 / 2 → 
  S_n = a_n * (2^n - 1) :=
by
  intros h1 h2
  sorry

end geometric_series_S_n_div_a_n_l323_32351


namespace decorations_given_to_friend_l323_32368

-- Definitions of the given conditions
def boxes : ℕ := 6
def decorations_per_box : ℕ := 25
def used_decorations : ℕ := 58
def neighbor_decorations : ℕ := 75

-- The statement of the proof problem
theorem decorations_given_to_friend : 
  (boxes * decorations_per_box) - used_decorations - neighbor_decorations = 17 := 
by 
  sorry

end decorations_given_to_friend_l323_32368


namespace max_value_g_l323_32331

def g (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_value_g : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2 ∧ ∀ y : ℝ, (0 ≤ y ∧ y ≤ 2) → g y ≤ g x) ∧ g x = 3 :=
by
  sorry

end max_value_g_l323_32331


namespace solve_diophantine_l323_32366

theorem solve_diophantine :
  {xy : ℤ × ℤ | 5 * (xy.1 ^ 2) + 5 * xy.1 * xy.2 + 5 * (xy.2 ^ 2) = 7 * xy.1 + 14 * xy.2} = {(-1, 3), (0, 0), (1, 2)} :=
by sorry

end solve_diophantine_l323_32366


namespace union_complement_U_A_B_l323_32371

def U : Set Int := {-1, 0, 1, 2, 3}

def A : Set Int := {-1, 0, 1}

def B : Set Int := {0, 1, 2}

def complement_U_A : Set Int := {u | u ∈ U ∧ u ∉ A}

theorem union_complement_U_A_B : (complement_U_A ∪ B) = {0, 1, 2, 3} :=
by
  sorry

end union_complement_U_A_B_l323_32371


namespace total_molecular_weight_l323_32353

-- Define atomic weights
def atomic_weight (element : String) : Float :=
  match element with
  | "K"  => 39.10
  | "Cr" => 51.996
  | "O"  => 16.00
  | "Fe" => 55.845
  | "S"  => 32.07
  | "Mn" => 54.938
  | _    => 0.0

-- Molecular weights of compounds
def molecular_weight_K2Cr2O7 : Float := 
  2 * atomic_weight "K" + 2 * atomic_weight "Cr" + 7 * atomic_weight "O"

def molecular_weight_Fe2_SO4_3 : Float := 
  2 * atomic_weight "Fe" + 3 * atomic_weight "S" + 12 * atomic_weight "O"

def molecular_weight_KMnO4 : Float := 
  atomic_weight "K" + atomic_weight "Mn" + 4 * atomic_weight "O"

-- Proof statement 
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 + 3 * molecular_weight_Fe2_SO4_3 + 5 * molecular_weight_KMnO4 = 3166.658 :=
by
  sorry

end total_molecular_weight_l323_32353


namespace t_shirt_cost_l323_32322

theorem t_shirt_cost (T : ℕ) 
  (h1 : 3 * T + 50 = 110) : T = 20 := 
by
  sorry

end t_shirt_cost_l323_32322


namespace find_x_complementary_l323_32383

-- Define the conditions.
def are_complementary (a b : ℝ) : Prop := a + b = 90

-- The main theorem statement with the condition and conclusion.
theorem find_x_complementary : ∀ x : ℝ, are_complementary (2*x) (3*x) → x = 18 := 
by
  intros x h
  -- sorry is a placeholder for the proof.
  sorry

end find_x_complementary_l323_32383


namespace james_bags_l323_32300

theorem james_bags (total_marbles : ℕ) (remaining_marbles : ℕ) (b : ℕ) (m : ℕ) 
  (h1 : total_marbles = 28) 
  (h2 : remaining_marbles = 21) 
  (h3 : m = total_marbles - remaining_marbles) 
  (h4 : b = total_marbles / m) : 
  b = 4 :=
by
  sorry

end james_bags_l323_32300


namespace find_breadth_l323_32324

-- Define variables and constants
variables (SA l h w : ℝ)

-- Given conditions
axiom h1 : SA = 2400
axiom h2 : l = 15
axiom h3 : h = 16

-- Define the surface area equation for a cuboid 
def surface_area := 2 * (l * w + l * h + w * h)

-- Statement to prove
theorem find_breadth : surface_area l w h = SA → w = 30.97 := sorry

end find_breadth_l323_32324


namespace a_and_c_can_complete_in_20_days_l323_32327

-- Define the work rates for the pairs given in the conditions.
variables {A B C : ℚ}

-- a and b together can complete the work in 12 days
axiom H1 : A + B = 1 / 12

-- b and c together can complete the work in 15 days
axiom H2 : B + C = 1 / 15

-- a, b, and c together can complete the work in 10 days
axiom H3 : A + B + C = 1 / 10

-- We aim to prove that a and c together can complete the work in 20 days,
-- hence their combined work rate should be 1 / 20.
theorem a_and_c_can_complete_in_20_days : A + C = 1 / 20 :=
by
  -- sorry will be used to skip the proof
  sorry

end a_and_c_can_complete_in_20_days_l323_32327


namespace number_of_green_hats_l323_32350

theorem number_of_green_hats 
  (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) 
  : G = 40 :=
sorry

end number_of_green_hats_l323_32350


namespace divisible_by_units_digit_l323_32312

theorem divisible_by_units_digit :
  ∃ l : List ℕ, l = [21, 22, 24, 25] ∧ l.length = 4 := 
  sorry

end divisible_by_units_digit_l323_32312


namespace domain_of_g_l323_32347

def f (x : ℝ) : Prop := x ∈ Set.Icc (-12.0) 6.0

def g (x : ℝ) : Prop := f (3 * x)

theorem domain_of_g : Set.Icc (-4.0) 2.0 = {x : ℝ | g x} := 
by 
    sorry

end domain_of_g_l323_32347


namespace correct_formula_l323_32321

theorem correct_formula {x y : ℕ} : 
  (x = 0 ∧ y = 100) ∨
  (x = 1 ∧ y = 90) ∨
  (x = 2 ∧ y = 70) ∨
  (x = 3 ∧ y = 40) ∨
  (x = 4 ∧ y = 0) →
  y = 100 - 5 * x - 5 * x^2 :=
by
  sorry

end correct_formula_l323_32321


namespace intersection_complement_eq_C_l323_32370

def A := { x : ℝ | -3 < x ∧ x < 6 }
def B := { x : ℝ | 2 < x ∧ x < 7 }
def complement_B := { x : ℝ | x ≤ 2 ∨ x ≥ 7 }
def C := { x : ℝ | -3 < x ∧ x ≤ 2 }

theorem intersection_complement_eq_C :
  A ∩ complement_B = C :=
sorry

end intersection_complement_eq_C_l323_32370


namespace nancy_hours_to_work_l323_32369

def tuition := 22000
def scholarship := 3000
def hourly_wage := 10
def parents_contribution := tuition / 2
def student_loan := 2 * scholarship
def total_financial_aid := scholarship + student_loan
def remaining_tuition := tuition - parents_contribution - total_financial_aid
def hours_to_work := remaining_tuition / hourly_wage

theorem nancy_hours_to_work : hours_to_work = 200 := by
  -- This by block demonstrates that a proof would go here
  sorry

end nancy_hours_to_work_l323_32369


namespace range_of_a_l323_32382

theorem range_of_a (a : ℝ) (h1 : 0 < a) :
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 - x - 6 ≤ 0) ∧
  (¬ (∀ x : ℝ, x^2 - x - 6 ≤ 0 → x^2 - 4*a*x + 3*a^2 ≤ 0)) →
  0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l323_32382


namespace cubic_ineq_l323_32388

theorem cubic_ineq (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_ineq_l323_32388


namespace change_of_b_l323_32313

variable {t b1 b2 C C_new : ℝ}

theorem change_of_b (hC : C = t * b1^4) 
                   (hC_new : C_new = 16 * C) 
                   (hC_new_eq : C_new = t * b2^4) : 
                   b2 = 2 * b1 :=
by
  sorry

end change_of_b_l323_32313


namespace road_greening_cost_l323_32333

-- Define constants for the conditions
def l_total : ℕ := 1500
def cost_A : ℕ := 22
def cost_B : ℕ := 25

-- Define variables for the cost per stem
variables (x y : ℕ)

-- Define the conditions from Plan A and Plan B
def plan_A (x y : ℕ) : Prop := 2 * x + 3 * y = cost_A
def plan_B (x y : ℕ) : Prop := x + 5 * y = cost_B

-- System of equations to find x and y
def system_of_equations (x y : ℕ) : Prop := plan_A x y ∧ plan_B x y

-- Define the constraint for the length of road greened according to Plan B
def length_constraint (a : ℕ) : Prop := l_total - a ≥ 2 * a

-- Define the total cost function
def total_cost (a : ℕ) (x y : ℕ) : ℕ := 22 * a + (x + 5 * y) * (l_total - a)

-- Prove the cost per stem and the minimized cost
theorem road_greening_cost :
  (∃ x y, system_of_equations x y ∧ x = 5 ∧ y = 4) ∧
  (∃ a : ℕ, length_constraint a ∧ a = 500 ∧ total_cost a 5 4 = 36000) :=
by
  -- This is where the proof would go
  sorry

end road_greening_cost_l323_32333


namespace maximize_distance_l323_32307

theorem maximize_distance (D_F D_R : ℕ) (x y : ℕ) (h1 : D_F = 21000) (h2 : D_R = 28000)
  (h3 : x + y ≤ D_F) (h4 : x + y ≤ D_R) :
  x + y = 24000 :=
sorry

end maximize_distance_l323_32307
