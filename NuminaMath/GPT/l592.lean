import Mathlib

namespace tshirt_more_expensive_l592_59247

-- Definitions based on given conditions
def jeans_price : ℕ := 30
def socks_price : ℕ := 5
def tshirt_price : ℕ := jeans_price / 2

-- Statement to prove (The t-shirt is $10 more expensive than the socks)
theorem tshirt_more_expensive : (tshirt_price - socks_price) = 10 :=
by
  rw [tshirt_price, socks_price]
  sorry  -- proof steps are omitted

end tshirt_more_expensive_l592_59247


namespace tan_double_angle_solution_l592_59246

theorem tan_double_angle_solution (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 :=
sorry

end tan_double_angle_solution_l592_59246


namespace max_rectangle_area_l592_59263

theorem max_rectangle_area (P : ℝ) (hP : 0 < P) : 
  ∃ (x y : ℝ), (2*x + 2*y = P) ∧ (x * y = P ^ 2 / 16) :=
by
  sorry

end max_rectangle_area_l592_59263


namespace greatest_four_digit_number_divisible_by_3_and_4_l592_59275

theorem greatest_four_digit_number_divisible_by_3_and_4 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 12 = 0) ∧ (∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ (m % 12 = 0) → m ≤ 9996) :=
by sorry

end greatest_four_digit_number_divisible_by_3_and_4_l592_59275


namespace radius_of_circle_is_4_l592_59207

noncomputable def circle_radius
  (a : ℝ) 
  (radius : ℝ) 
  (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 9 = 0 ∧ (-a, 0) = (5, 0) ∧ radius = 4

theorem radius_of_circle_is_4 
  (a x y : ℝ) 
  (radius : ℝ) 
  (h : circle_radius a radius x y) : 
  radius = 4 :=
by 
  sorry

end radius_of_circle_is_4_l592_59207


namespace number_of_girls_l592_59219

theorem number_of_girls 
  (B G : ℕ) 
  (h1 : B + G = 480) 
  (h2 : 5 * B = 3 * G) :
  G = 300 := 
sorry

end number_of_girls_l592_59219


namespace perpendicular_lines_slope_l592_59260

theorem perpendicular_lines_slope (a : ℝ) :
  (∀ x1 y1 x2 y2: ℝ, y1 = a * x1 - 2 ∧ y2 = x2 + 1 → (a * 1) = -1) → a = -1 :=
by
  sorry

end perpendicular_lines_slope_l592_59260


namespace maximum_sum_l592_59252

theorem maximum_sum (a b c d : ℕ) (h₀ : a < b ∧ b < c ∧ c < d)
  (h₁ : (c + d) + (a + b + c) = 2017) : a + b + c + d ≤ 806 :=
sorry

end maximum_sum_l592_59252


namespace carlotta_tantrum_time_l592_59256

theorem carlotta_tantrum_time :
  (∀ (T P S : ℕ), 
   S = 6 ∧ T + P + S = 54 ∧ P = 3 * S → T = 5 * S) :=
by
  intro T P S
  rintro ⟨hS, hTotal, hPractice⟩
  sorry

end carlotta_tantrum_time_l592_59256


namespace min_value_of_reciprocal_sum_l592_59214

open Real

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : x + y = 12) (h4 : x * y = 20) : (1 / x + 1 / y) = 3 / 5 :=
sorry

end min_value_of_reciprocal_sum_l592_59214


namespace heptagon_isosceles_same_color_l592_59244

theorem heptagon_isosceles_same_color 
  (color : Fin 7 → Prop) (red blue : Prop)
  (h_heptagon : ∀ i : Fin 7, color i = red ∨ color i = blue) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ color i = color j ∧ color j = color k ∧ ((i + j) % 7 = k ∨ (j + k) % 7 = i ∨ (k + i) % 7 = j) :=
sorry

end heptagon_isosceles_same_color_l592_59244


namespace line_through_points_C_D_has_undefined_slope_and_angle_90_l592_59271

theorem line_through_points_C_D_has_undefined_slope_and_angle_90 (m : ℝ) (n : ℝ) (hn : n ≠ 0) :
  ∃ θ : ℝ, (∀ (slope : ℝ), false) ∧ θ = 90 :=
by { sorry }

end line_through_points_C_D_has_undefined_slope_and_angle_90_l592_59271


namespace root_value_cond_l592_59249

theorem root_value_cond (p q : ℝ) (h₁ : ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = q) (h₂ : q ≠ 0) : p + q = -1 := 
sorry

end root_value_cond_l592_59249


namespace xyz_squared_l592_59234

theorem xyz_squared (x y z p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hxy : x + y = p) (hyz : y + z = q) (hzx : z + x = r) :
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p * q - q * r - r * p) / 2 :=
by
  sorry

end xyz_squared_l592_59234


namespace hockey_championship_max_k_volleyball_championship_max_k_l592_59243

theorem hockey_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 18 :=
by
  -- proof goes here
  sorry

theorem volleyball_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 15 :=
by
  -- proof goes here
  sorry

end hockey_championship_max_k_volleyball_championship_max_k_l592_59243


namespace swim_club_member_count_l592_59204

theorem swim_club_member_count :
  let total_members := 60
  let passed_percentage := 0.30
  let passed_members := total_members * passed_percentage
  let not_passed_members := total_members - passed_members
  let preparatory_course_members := 12
  not_passed_members - preparatory_course_members = 30 :=
by
  sorry

end swim_club_member_count_l592_59204


namespace sophomores_bought_15_more_markers_l592_59251

theorem sophomores_bought_15_more_markers (f_cost s_cost marker_cost : ℕ) (hf: f_cost = 267) (hs: s_cost = 312) (hm: marker_cost = 3) : 
  (s_cost / marker_cost) - (f_cost / marker_cost) = 15 :=
by
  sorry

end sophomores_bought_15_more_markers_l592_59251


namespace graph_of_equation_is_two_lines_l592_59224

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (2 * x - y)^2 = 4 * x^2 - y^2 ↔ (y = 0 ∨ y = 2 * x) :=
by
  sorry

end graph_of_equation_is_two_lines_l592_59224


namespace valid_cone_from_sector_l592_59258

-- Given conditions
def sector_angle : ℝ := 300
def circle_radius : ℝ := 15

-- Definition of correct option E
def base_radius_E : ℝ := 12
def slant_height_E : ℝ := 15

theorem valid_cone_from_sector :
  ( (sector_angle / 360) * (2 * Real.pi * circle_radius) = 25 * Real.pi ) ∧
  (slant_height_E = circle_radius) ∧
  (base_radius_E = 12) ∧
  (15^2 = 12^2 + 9^2) :=
by
  -- This theorem states that given sector angle and circle radius, the valid option is E
  sorry

end valid_cone_from_sector_l592_59258


namespace complement_union_l592_59233

def U : Set ℤ := {x | -3 < x ∧ x ≤ 4}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {1, 2, 3}

def C (U : Set ℤ) (S : Set ℤ) : Set ℤ := {x | x ∈ U ∧ x ∉ S}

theorem complement_union (A B : Set ℤ) (U : Set ℤ) :
  C U (A ∪ B) = {0, 4} :=
by
  sorry

end complement_union_l592_59233


namespace lulu_final_cash_l592_59208

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l592_59208


namespace yz_zx_xy_minus_2xyz_leq_7_27_l592_59269

theorem yz_zx_xy_minus_2xyz_leq_7_27 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  (y * z + z * x + x * y - 2 * x * y * z) ≤ 7 / 27 := 
by 
  sorry

end yz_zx_xy_minus_2xyz_leq_7_27_l592_59269


namespace chess_tournament_l592_59231

theorem chess_tournament (n k : ℕ) (S : ℕ) (m : ℕ) 
  (h1 : S ≤ k * n) 
  (h2 : S ≥ m * n) 
  : m ≤ k := 
by 
  sorry

end chess_tournament_l592_59231


namespace problem_statement_l592_59264

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

theorem problem_statement (f : ℝ → ℝ) :
  is_odd_function f →
  (∀ x : ℝ, f (x + 6) = f (x) + 3) →
  f 1 = 1 →
  f 2015 + f 2016 = 2015 :=
by
  sorry

end problem_statement_l592_59264


namespace apogee_reach_second_stage_model_engine_off_time_l592_59294

-- Given conditions
def altitudes := [(0, 0), (1, 24), (2, 96), (4, 386), (5, 514), (6, 616), (9, 850), (13, 994), (14, 1000), (16, 976), (19, 850), (24, 400)]
def second_stage_curve (x : ℝ) : ℝ := -6 * x^2 + 168 * x - 176

-- Proof problems
theorem apogee_reach : (14, 1000) ∈ altitudes :=
sorry  -- Need to prove the inclusion of the apogee point in the table

theorem second_stage_model : 
    second_stage_curve 14 = 1000 ∧ 
    second_stage_curve 16 = 976 ∧ 
    second_stage_curve 19 = 850 ∧ 
    ∃ n, n = 4 :=
sorry  -- Need to prove the analytical expression is correct and n = 4

theorem engine_off_time : 
    ∃ t : ℝ, t = 14 + 5 * Real.sqrt 6 ∧ second_stage_curve t = 100 :=
sorry  -- Need to prove the engine off time calculation

end apogee_reach_second_stage_model_engine_off_time_l592_59294


namespace shortest_part_is_15_l592_59215

namespace ProofProblem

def rope_length : ℕ := 60
def ratio_part1 : ℕ := 3
def ratio_part2 : ℕ := 4
def ratio_part3 : ℕ := 5

def total_parts := ratio_part1 + ratio_part2 + ratio_part3
def length_per_part := rope_length / total_parts
def shortest_part_length := ratio_part1 * length_per_part

theorem shortest_part_is_15 :
  shortest_part_length = 15 := by
  sorry

end ProofProblem

end shortest_part_is_15_l592_59215


namespace find_real_numbers_l592_59212

theorem find_real_numbers (x : ℝ) :
  (x^3 - x^2 = (x^2 - x)^2) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end find_real_numbers_l592_59212


namespace range_of_y_l592_59210

theorem range_of_y (m n k y : ℝ)
  (h₁ : 0 ≤ m)
  (h₂ : 0 ≤ n)
  (h₃ : 0 ≤ k)
  (h₄ : m - k + 1 = 1)
  (h₅ : 2 * k + n = 1)
  (h₆ : y = 2 * k^2 - 8 * k + 6)
  : 5 / 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end range_of_y_l592_59210


namespace crossing_time_approx_11_16_seconds_l592_59232

noncomputable def length_train_1 : ℝ := 140 -- length of the first train in meters
noncomputable def length_train_2 : ℝ := 170 -- length of the second train in meters
noncomputable def speed_train_1_km_hr : ℝ := 60 -- speed of the first train in km/hr
noncomputable def speed_train_2_km_hr : ℝ := 40 -- speed of the second train in km/hr

noncomputable def speed_conversion_factor : ℝ := 5 / 18 -- conversion factor from km/hr to m/s

-- convert speeds from km/hr to m/s
noncomputable def speed_train_1_m_s : ℝ := speed_train_1_km_hr * speed_conversion_factor
noncomputable def speed_train_2_m_s : ℝ := speed_train_2_km_hr * speed_conversion_factor

-- calculate relative speed in m/s (since they are moving in opposite directions)
noncomputable def relative_speed_m_s : ℝ := speed_train_1_m_s + speed_train_2_m_s

-- total distance to be covered
noncomputable def total_distance : ℝ := length_train_1 + length_train_2

-- calculate the time to cross each other
noncomputable def crossing_time : ℝ := total_distance / relative_speed_m_s

theorem crossing_time_approx_11_16_seconds : abs (crossing_time - 11.16) < 0.01 := by
    sorry

end crossing_time_approx_11_16_seconds_l592_59232


namespace f_1997_leq_666_l592_59266

noncomputable def f : ℕ+ → ℕ := sorry

axiom f_mn_inequality : ∀ (m n : ℕ+), f (m + n) ≥ f m + f n
axiom f_two : f 2 = 0
axiom f_three_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

theorem f_1997_leq_666 : f 1997 ≤ 666 := sorry

end f_1997_leq_666_l592_59266


namespace sum_term_addition_l592_59242

theorem sum_term_addition (k : ℕ) (hk : k ≥ 2) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end sum_term_addition_l592_59242


namespace smallest_N_l592_59240

-- Definitions corresponding to the conditions
def circular_table (chairs : ℕ) : Prop := chairs = 72

def proper_seating (N chairs : ℕ) : Prop :=
  ∀ (new_person : ℕ), new_person < chairs →
    (∃ seated, seated < N ∧ (seated - new_person).gcd chairs = 1)

-- Problem statement
theorem smallest_N (chairs : ℕ) :
  circular_table chairs →
  ∃ N, proper_seating N chairs ∧ (∀ M < N, ¬ proper_seating M chairs) ∧ N = 18 :=
by
  intro h
  sorry

end smallest_N_l592_59240


namespace black_queen_awake_at_10_l592_59265

-- Define the logical context
def king_awake_at_10 (king_asleep : Prop) : Prop :=
  king_asleep -> false

def king_asleep_at_10 (king_asleep : Prop) : Prop :=
  king_asleep

def queen_awake_at_10 (queen_asleep : Prop) : Prop :=
  queen_asleep -> false

-- Define the main theorem
theorem black_queen_awake_at_10 
  (king_asleep : Prop)
  (queen_asleep : Prop)
  (king_belief : king_asleep ↔ (king_asleep ∧ queen_asleep)) :
  queen_awake_at_10 queen_asleep :=
by
  -- Proof is omitted
  sorry

end black_queen_awake_at_10_l592_59265


namespace xy_product_l592_59296

theorem xy_product (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := 
sorry

end xy_product_l592_59296


namespace find_k_l592_59298

-- Definitions of the conditions as given in the problem
def total_amount (A B C : ℕ) : Prop := A + B + C = 585
def c_share (C : ℕ) : Prop := C = 260
def equal_shares (A B C k : ℕ) : Prop := 4 * A = k * C ∧ 6 * B = k * C

-- The theorem we need to prove
theorem find_k (A B C k : ℕ) (h_tot: total_amount A B C)
  (h_c: c_share C) (h_eq: equal_shares A B C k) : k = 3 := by 
  sorry

end find_k_l592_59298


namespace original_salary_l592_59203

theorem original_salary (S : ℝ) (h : 1.10 * S * 0.95 = 3135) : S = 3000 := 
by 
  sorry

end original_salary_l592_59203


namespace product_of_w_and_z_l592_59255

variable (EF FG GH HE : ℕ)
variable (w z : ℕ)

-- Conditions from the problem
def parallelogram_conditions : Prop :=
  EF = 42 ∧ FG = 4 * z^3 ∧ GH = 3 * w + 6 ∧ HE = 32 ∧ EF = GH ∧ FG = HE

-- The proof problem proving the requested product given the conditions
theorem product_of_w_and_z (h : parallelogram_conditions EF FG GH HE w z) : (w * z) = 24 :=
by
  sorry

end product_of_w_and_z_l592_59255


namespace percentage_yield_l592_59257

theorem percentage_yield (market_price annual_dividend : ℝ) (yield : ℝ) 
  (H1 : yield = 0.12)
  (H2 : market_price = 125)
  (H3 : annual_dividend = yield * market_price) :
  (annual_dividend / market_price) * 100 = 12 := 
sorry

end percentage_yield_l592_59257


namespace find_m_l592_59254

def g (x : ℤ) (A : ℤ) (B : ℤ) (C : ℤ) : ℤ := A * x^2 + B * x + C

theorem find_m (A B C m : ℤ) 
  (h1 : g 2 A B C = 0)
  (h2 : 100 < g 9 A B C ∧ g 9 A B C < 110)
  (h3 : 150 < g 10 A B C ∧ g 10 A B C < 160)
  (h4 : 10000 * m < g 200 A B C ∧ g 200 A B C < 10000 * (m + 1)) : 
  m = 16 :=
sorry

end find_m_l592_59254


namespace teacher_earnings_l592_59236

noncomputable def cost_per_half_hour : ℝ := 10
noncomputable def lesson_duration_in_hours : ℝ := 1
noncomputable def lessons_per_week : ℝ := 1
noncomputable def weeks : ℝ := 5

theorem teacher_earnings : 
  2 * cost_per_half_hour * lesson_duration_in_hours * lessons_per_week * weeks = 100 :=
by
  sorry

end teacher_earnings_l592_59236


namespace positive_integer_triplets_l592_59202

theorem positive_integer_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_lcm : a + b + c = Nat.lcm a (Nat.lcm b c)) :
  (∃ k, k ≥ 1 ∧ a = k ∧ b = 2 * k ∧ c = 3 * k) :=
sorry

end positive_integer_triplets_l592_59202


namespace train_speed_l592_59273

noncomputable def train_speed_kmph (L_t L_b : ℝ) (T : ℝ) : ℝ :=
  (L_t + L_b) / T * 3.6

theorem train_speed (L_t L_b : ℝ) (T : ℝ) :
  L_t = 110 ∧ L_b = 190 ∧ T = 17.998560115190784 → train_speed_kmph L_t L_b T = 60 :=
by
  intro h
  sorry

end train_speed_l592_59273


namespace school_club_profit_l592_59221

theorem school_club_profit : 
  let purchase_price_per_bar := 3 / 4
  let selling_price_per_bar := 2 / 3
  let total_bars := 1200
  let bars_with_discount := total_bars - 1000
  let discount_per_bar := 0.10
  let total_cost := total_bars * purchase_price_per_bar
  let total_revenue_without_discount := total_bars * selling_price_per_bar
  let total_discount := bars_with_discount * discount_per_bar
  let adjusted_revenue := total_revenue_without_discount - total_discount
  let profit := adjusted_revenue - total_cost
  profit = -116 :=
by sorry

end school_club_profit_l592_59221


namespace train_or_plane_not_ship_possible_modes_l592_59241

-- Define the probabilities of different modes of transportation
def P_train : ℝ := 0.3
def P_ship : ℝ := 0.2
def P_car : ℝ := 0.1
def P_plane : ℝ := 0.4

-- 1. Proof that probability of train or plane is 0.7
theorem train_or_plane : P_train + P_plane = 0.7 :=
by sorry

-- 2. Proof that probability of not taking a ship is 0.8
theorem not_ship : 1 - P_ship = 0.8 :=
by sorry

-- 3. Proof that if probability is 0.5, the modes are either (ship, train) or (car, plane)
theorem possible_modes (P_value : ℝ) (h1 : P_value = 0.5) :
  (P_ship + P_train = P_value) ∨ (P_car + P_plane = P_value) :=
by sorry

end train_or_plane_not_ship_possible_modes_l592_59241


namespace last_digit_of_power_of_two_l592_59218

theorem last_digit_of_power_of_two (n : ℕ) (h : n ≥ 2) : (2 ^ (2 ^ n) + 1) % 10 = 7 :=
sorry

end last_digit_of_power_of_two_l592_59218


namespace complement_U_M_inter_N_eq_l592_59285

def U : Set ℝ := Set.univ

def M : Set ℝ := { y | ∃ x, y = 2 * x + 1 ∧ -1/2 ≤ x ∧ x ≤ 1/2 }

def N : Set ℝ := { x | ∃ y, y = Real.log (x^2 + 3 * x) ∧ (x < -3 ∨ x > 0) }

def complement_U_M : Set ℝ := U \ M

theorem complement_U_M_inter_N_eq :
  (complement_U_M ∩ N) = ((Set.Iio (-3 : ℝ)) ∪ (Set.Ioi (2 : ℝ))) :=
sorry

end complement_U_M_inter_N_eq_l592_59285


namespace sqrt_sum_fractions_eq_l592_59239

theorem sqrt_sum_fractions_eq :
  (Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30) :=
by
  sorry

end sqrt_sum_fractions_eq_l592_59239


namespace foreign_objects_total_sum_l592_59295

-- define the conditions
def dog_burrs : Nat := 12
def dog_ticks := 6 * dog_burrs
def dog_fleas := 3 * dog_ticks

def cat_burrs := 2 * dog_burrs
def cat_ticks := dog_ticks / 3
def cat_fleas := 4 * cat_ticks

-- calculate the total foreign objects
def total_dog := dog_burrs + dog_ticks + dog_fleas
def total_cat := cat_burrs + cat_ticks + cat_fleas

def total_objects := total_dog + total_cat

-- state the theorem
theorem foreign_objects_total_sum : total_objects = 444 := by
  sorry

end foreign_objects_total_sum_l592_59295


namespace haleys_car_distance_l592_59287

theorem haleys_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) (distance_covered : ℕ) 
   (h_ratio : fuel_ratio = 4) (h_distance_ratio : distance_ratio = 7) (h_fuel_used : fuel_used = 44) :
   distance_covered = 77 := by
  -- Proof to be filled in
  sorry

end haleys_car_distance_l592_59287


namespace find_G16_l592_59205

variable (G : ℝ → ℝ)

def condition1 : Prop := G 8 = 28

def condition2 : Prop := ∀ x : ℝ, 
  (x^2 + 8*x + 16) ≠ 0 → 
  (G (4*x) / G (x + 4) = 16 - (64*x + 80) / (x^2 + 8*x + 16))

theorem find_G16 (h1 : condition1 G) (h2 : condition2 G) : G 16 = 120 :=
sorry

end find_G16_l592_59205


namespace max_x_plus_2y_l592_59261

theorem max_x_plus_2y {x y : ℝ} (h : x^2 - x * y + y^2 = 1) :
  x + 2 * y ≤ (2 * Real.sqrt 21) / 3 :=
sorry

end max_x_plus_2y_l592_59261


namespace euler_quadrilateral_theorem_l592_59201

theorem euler_quadrilateral_theorem (A1 A2 A3 A4 P Q : ℝ) 
  (midpoint_P : P = (A1 + A3) / 2)
  (midpoint_Q : Q = (A2 + A4) / 2) 
  (length_A1A2 length_A2A3 length_A3A4 length_A4A1 length_A1A3 length_A2A4 length_PQ : ℝ)
  (h1 : length_A1A2 = A1A2) (h2 : length_A2A3 = A2A3)
  (h3 : length_A3A4 = A3A4) (h4 : length_A4A1 = A4A1)
  (h5 : length_A1A3 = A1A3) (h6 : length_A2A4 = A2A4)
  (h7 : length_PQ = PQ) :
  length_A1A2^2 + length_A2A3^2 + length_A3A4^2 + length_A4A1^2 = 
  length_A1A3^2 + length_A2A4^2 + 4 * length_PQ^2 := sorry

end euler_quadrilateral_theorem_l592_59201


namespace sphere_tangency_relation_l592_59290

noncomputable def sphere_tangents (r R : ℝ) (h : R > r) :=
  (R >= (2 / (Real.sqrt 3) - 1) * r) ∧
  (∃ x, x = (R * (R + r - Real.sqrt (R^2 + 2 * R * r - r^2 / 3))) /
            (r + Real.sqrt (R^2 + 2 * R * r - r^2 / 3) - R)) 

theorem sphere_tangency_relation (r R: ℝ) (h : R > r) :
  sphere_tangents r R h :=
by
  sorry

end sphere_tangency_relation_l592_59290


namespace egg_production_l592_59259

theorem egg_production (n_chickens1 n_chickens2 n_eggs1 n_eggs2 n_days1 n_days2 : ℕ)
  (h1 : n_chickens1 = 6) (h2 : n_eggs1 = 30) (h3 : n_days1 = 5) (h4 : n_chickens2 = 10) (h5 : n_days2 = 8) :
  n_eggs2 = 80 :=
sorry

end egg_production_l592_59259


namespace total_weight_correct_weight_difference_correct_l592_59206

variables (baskets_of_apples baskets_of_pears : ℕ) (kg_per_basket_of_apples kg_per_basket_of_pears : ℕ)

def total_weight_apples_ppears (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_apples * kg_per_basket_of_apples) + (baskets_of_pears * kg_per_basket_of_pears)

def weight_difference_pears_apples (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_pears * kg_per_basket_of_pears) - (baskets_of_apples * kg_per_basket_of_apples)

theorem total_weight_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  total_weight_apples_ppears baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 11300 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

theorem weight_difference_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  weight_difference_pears_apples baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 1700 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

end total_weight_correct_weight_difference_correct_l592_59206


namespace boat_speed_in_still_water_l592_59216

theorem boat_speed_in_still_water (b s : ℕ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := by
  sorry

end boat_speed_in_still_water_l592_59216


namespace average_height_of_students_l592_59282

theorem average_height_of_students (x : ℕ) (female_height male_height : ℕ) 
  (female_height_eq : female_height = 170) (male_height_eq : male_height = 185) 
  (ratio : 2 * x = x * 2) : 
  ((2 * x * male_height + x * female_height) / (2 * x + x) = 180) := 
by
  sorry

end average_height_of_students_l592_59282


namespace smallest_n_divisible_l592_59292

theorem smallest_n_divisible (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_divisible_l592_59292


namespace find_z_l592_59272

theorem find_z (x y : ℤ) (h1 : x * y + x + y = 106) (h2 : x^2 * y + x * y^2 = 1320) :
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 :=
sorry

end find_z_l592_59272


namespace sum_arithmetic_sequence_l592_59286

noncomputable def is_arithmetic (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∃ a1 : ℚ, ∀ n : ℕ, a n = a1 + n * d

noncomputable def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 0 + a (n - 1)) / 2

theorem sum_arithmetic_sequence (a : ℕ → ℚ) (h_arith : is_arithmetic a)
  (h1 : 2 * a 3 = 5) (h2 : a 4 + a 12 = 9) : sum_of_first_n_terms a 10 = 35 :=
by
  -- Proof omitted
  sorry

end sum_arithmetic_sequence_l592_59286


namespace slower_train_pass_time_l592_59238

noncomputable def relative_speed_km_per_hr (v1 v2 : ℕ) : ℕ :=
v1 + v2

noncomputable def relative_speed_m_per_s (v_km_per_hr : ℕ) : ℝ :=
(v_km_per_hr * 5) / 18

noncomputable def time_to_pass (distance_m : ℕ) (speed_m_per_s : ℝ) : ℝ :=
distance_m / speed_m_per_s

theorem slower_train_pass_time
  (length_train1 length_train2 : ℕ)
  (speed_train1_km_per_hr speed_train2_km_per_hr : ℕ)
  (distance_to_cover : ℕ)
  (h1 : length_train1 = 800)
  (h2 : length_train2 = 600)
  (h3 : speed_train1_km_per_hr = 85)
  (h4 : speed_train2_km_per_hr = 65)
  (h5 : distance_to_cover = length_train2) :
  time_to_pass distance_to_cover (relative_speed_m_per_s (relative_speed_km_per_hr speed_train1_km_per_hr speed_train2_km_per_hr)) = 14.4 := 
sorry

end slower_train_pass_time_l592_59238


namespace integer_solutions_l592_59209

theorem integer_solutions (x : ℝ) (n : ℤ)
  (h1 : ⌊x⌋ = n) :
  3 * x - 2 * n + 4 = 0 ↔
  x = -4 ∨ x = (-14:ℚ)/3 ∨ x = (-16:ℚ)/3 :=
by sorry

end integer_solutions_l592_59209


namespace find_n_solution_l592_59297

theorem find_n_solution : ∃ n : ℤ, (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n : ℝ) / (n + 1 : ℝ) = 3) :=
by
  use 0
  sorry

end find_n_solution_l592_59297


namespace roots_of_quadratic_l592_59268

theorem roots_of_quadratic (p q x1 x2 : ℕ) (hp : p + q = 28) (hroots : ∀ x, x^2 + p * x + q = 0 → (x = x1 ∨ x = x2)) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end roots_of_quadratic_l592_59268


namespace find_a9_a10_l592_59220

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem find_a9_a10 (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 2) :
  a 9 + a 10 = 16 := 
sorry

end find_a9_a10_l592_59220


namespace calculate_price_per_pound_of_meat_l592_59226

noncomputable def price_per_pound_of_meat : ℝ :=
  let total_hours := 50
  let w := 8
  let m_pounds := 20
  let fv_pounds := 15
  let fv_pp := 4
  let b_pounds := 60
  let b_pp := 1.5
  let j_wage := 10
  let j_hours := 10
  let j_rate := 1.5

  -- known costs
  let fv_cost := fv_pounds * fv_pp
  let b_cost := b_pounds * b_pp
  let j_cost := j_hours * j_wage * j_rate

  -- total costs
  let total_cost := total_hours * w
  let known_costs := fv_cost + b_cost + j_cost

  (total_cost - known_costs) / m_pounds

theorem calculate_price_per_pound_of_meat : price_per_pound_of_meat = 5 := by
  sorry

end calculate_price_per_pound_of_meat_l592_59226


namespace arithmetic_sequence_general_formula_l592_59227

theorem arithmetic_sequence_general_formula :
  (∀ n:ℕ, ∃ (a_n : ℕ), ∀ k:ℕ, a_n = 2 * k → k = n)
  ∧ ( 2 * n + 2 * (n + 2) = 8 → 2 * n + 2 * (n + 3) = 12 → a_n = 2 * n )
  ∧ (S_n = (n * (n + 1)) / 2 → S_n = 420 → n = 20) :=
by { sorry }

end arithmetic_sequence_general_formula_l592_59227


namespace net_effect_on_sale_value_l592_59288

theorem net_effect_on_sale_value 
  (P Original_Sales_Volume : ℝ) 
  (reduced_by : ℝ := 0.18) 
  (sales_increase : ℝ := 0.88) 
  (additional_tax : ℝ := 0.12) :
  P * Original_Sales_Volume * ((1 - reduced_by) * (1 + additional_tax) * (1 + sales_increase) - 1) = P * Original_Sales_Volume * 0.7184 :=
  by
  sorry

end net_effect_on_sale_value_l592_59288


namespace gcd_100_450_l592_59291

theorem gcd_100_450 : Int.gcd 100 450 = 50 := 
by sorry

end gcd_100_450_l592_59291


namespace number_of_non_Speedsters_l592_59211

theorem number_of_non_Speedsters (V : ℝ) (h0 : (4 / 15) * V = 12) : (2 / 3) * V = 30 :=
by
  -- The conditions are such that:
  -- V is the total number of vehicles.
  -- (4 / 15) * V = 12 means 4/5 of 1/3 of the total vehicles are convertibles.
  -- We need to prove that 2/3 of the vehicles are not Speedsters.
  sorry

end number_of_non_Speedsters_l592_59211


namespace number_of_women_is_24_l592_59200

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l592_59200


namespace rockham_soccer_league_l592_59213

theorem rockham_soccer_league (cost_socks : ℕ) (cost_tshirt : ℕ) (custom_fee : ℕ) (total_cost : ℕ) :
  cost_socks = 6 →
  cost_tshirt = cost_socks + 7 →
  custom_fee = 200 →
  total_cost = 2892 →
  ∃ members : ℕ, total_cost - custom_fee = members * (2 * (cost_socks + cost_tshirt)) ∧ members = 70 :=
by
  intros
  sorry

end rockham_soccer_league_l592_59213


namespace geometric_sequence_min_value_l592_59293

theorem geometric_sequence_min_value
  (s : ℝ) (b1 b2 b3 : ℝ)
  (h1 : b1 = 2)
  (h2 : b2 = 2 * s)
  (h3 : b3 = 2 * s ^ 2) :
  ∃ (s : ℝ), 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end geometric_sequence_min_value_l592_59293


namespace allocate_plots_l592_59284

theorem allocate_plots (x y : ℕ) (h : x > y) : 
  ∃ u v : ℕ, (u^2 + v^2 = 2 * (x^2 + y^2)) :=
by
  sorry

end allocate_plots_l592_59284


namespace number_of_integer_values_l592_59228

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l592_59228


namespace rate_of_current_in_river_l592_59253

theorem rate_of_current_in_river (b c : ℝ) (h1 : 4 * (b + c) = 24) (h2 : 6 * (b - c) = 24) : c = 1 := by
  sorry

end rate_of_current_in_river_l592_59253


namespace find_smallest_x_l592_59262

theorem find_smallest_x :
  ∃ (x : ℕ), x > 1 ∧ (x^2 % 1000 = x % 1000) ∧ x = 376 := by
  sorry

end find_smallest_x_l592_59262


namespace mary_fruits_l592_59245

noncomputable def totalFruitsLeft 
    (initial_apples: ℕ) (initial_oranges: ℕ) (initial_blueberries: ℕ) (initial_grapes: ℕ) (initial_kiwis: ℕ)
    (salad_apples: ℕ) (salad_oranges: ℕ) (salad_blueberries: ℕ)
    (snack_apples: ℕ) (snack_oranges: ℕ) (snack_kiwis: ℕ)
    (given_apples: ℕ) (given_oranges: ℕ) (given_blueberries: ℕ) (given_grapes: ℕ) (given_kiwis: ℕ) : ℕ :=
  let remaining_apples := initial_apples - salad_apples - snack_apples - given_apples
  let remaining_oranges := initial_oranges - salad_oranges - snack_oranges - given_oranges
  let remaining_blueberries := initial_blueberries - salad_blueberries - given_blueberries
  let remaining_grapes := initial_grapes - given_grapes
  let remaining_kiwis := initial_kiwis - snack_kiwis - given_kiwis
  remaining_apples + remaining_oranges + remaining_blueberries + remaining_grapes + remaining_kiwis

theorem mary_fruits :
    totalFruitsLeft 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end mary_fruits_l592_59245


namespace cube_lateral_surface_area_l592_59270

theorem cube_lateral_surface_area (V : ℝ) (h_V : V = 125) : 
  ∃ A : ℝ, A = 100 :=
by
  sorry

end cube_lateral_surface_area_l592_59270


namespace valid_quadratic_polynomials_l592_59274

theorem valid_quadratic_polynomials (b c : ℤ)
  (h₁ : ∃ x₁ x₂ : ℤ, b = -(x₁ + x₂) ∧ c = x₁ * x₂)
  (h₂ : 1 + b + c = 10) :
  (b = -13 ∧ c = 22) ∨ (b = -9 ∧ c = 18) ∨ (b = 9 ∧ c = 0) ∨ (b = 5 ∧ c = 4) := sorry

end valid_quadratic_polynomials_l592_59274


namespace cosine_sine_inequality_theorem_l592_59248

theorem cosine_sine_inequality_theorem (θ : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
    (π / 12 < θ ∧ θ < 5 * π / 12) :=
by
  sorry

end cosine_sine_inequality_theorem_l592_59248


namespace quadratic_solution_l592_59280

theorem quadratic_solution (m n : ℝ) (h1 : m ≠ 0) (h2 : m * 1^2 + n * 1 - 1 = 0) : m + n = 1 :=
sorry

end quadratic_solution_l592_59280


namespace least_bulbs_needed_l592_59230

/-- Tulip bulbs come in packs of 15, and daffodil bulbs come in packs of 16.
  Rita wants to buy the same number of tulip and daffodil bulbs. 
  The goal is to prove that the least number of bulbs she needs to buy is 240, i.e.,
  the least common multiple of 15 and 16 is 240. -/
theorem least_bulbs_needed : Nat.lcm 15 16 = 240 := 
by
  sorry

end least_bulbs_needed_l592_59230


namespace cube_volume_l592_59289

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l592_59289


namespace find_quotient_l592_59225

theorem find_quotient
  (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 131) (h2 : divisor = 14) (h3 : remainder = 5)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 :=
by
  sorry

end find_quotient_l592_59225


namespace area_of_X_part_l592_59217

theorem area_of_X_part :
    (∃ s : ℝ, s^2 = 2520 ∧ 
     (∃ E F G H : ℝ, E = F ∧ F = G ∧ G = H ∧ 
         E = s / 4 ∧ F = s / 4 ∧ G = s / 4 ∧ H = s / 4) ∧ 
     2520 * 11 / 24 = 1155) :=
by
  sorry

end area_of_X_part_l592_59217


namespace number_of_solutions_l592_59278

theorem number_of_solutions (x y : ℕ) : (3 * x + 2 * y = 1001) → ∃! (n : ℕ), n = 167 := by
  sorry

end number_of_solutions_l592_59278


namespace angle_tuvels_equiv_l592_59283

-- Defining the conditions
def full_circle_tuvels : ℕ := 400
def degree_angle_in_circle : ℕ := 360
def specific_angle_degrees : ℕ := 45

-- Proof statement showing the equivalence
theorem angle_tuvels_equiv :
  (specific_angle_degrees * full_circle_tuvels) / degree_angle_in_circle = 50 :=
by
  sorry

end angle_tuvels_equiv_l592_59283


namespace abs_neg_2023_l592_59222

theorem abs_neg_2023 : |(-2023)| = 2023 :=
by
  sorry

end abs_neg_2023_l592_59222


namespace Norm_photo_count_l592_59223

variables (L M N : ℕ)

-- Conditions from the problem
def cond1 : Prop := L = N - 60
def cond2 : Prop := N = 2 * L + 10

-- Given the conditions, prove N = 110
theorem Norm_photo_count (h1 : cond1 L N) (h2 : cond2 L N) : N = 110 :=
by
  sorry

end Norm_photo_count_l592_59223


namespace compare_negative_sqrt_values_l592_59299

theorem compare_negative_sqrt_values : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := 
sorry

end compare_negative_sqrt_values_l592_59299


namespace calculate_expression_l592_59235

theorem calculate_expression : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by
  sorry

end calculate_expression_l592_59235


namespace units_digit_of_3_pow_1987_l592_59277

theorem units_digit_of_3_pow_1987 : 3 ^ 1987 % 10 = 7 := by
  sorry

end units_digit_of_3_pow_1987_l592_59277


namespace painted_cube_l592_59279

noncomputable def cube_side_length : ℕ :=
  7

theorem painted_cube (painted_faces: ℕ) (one_side_painted_cubes: ℕ) (orig_side_length: ℕ) :
    painted_faces = 6 ∧ one_side_painted_cubes = 54 ∧ (orig_side_length + 2) ^ 2 / 6 = 9 →
    orig_side_length = cube_side_length :=
by
  sorry

end painted_cube_l592_59279


namespace age_of_b_l592_59237

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 42) : b = 16 :=
by
  sorry

end age_of_b_l592_59237


namespace expected_profit_may_is_3456_l592_59267

-- Given conditions as definitions
def february_profit : ℝ := 2000
def april_profit : ℝ := 2880
def growth_rate (x : ℝ) : Prop := (2000 * (1 + x)^2 = 2880)

-- The expected profit in May
def expected_may_profit (x : ℝ) : ℝ := april_profit * (1 + x)

-- The theorem to be proved based on the given conditions
theorem expected_profit_may_is_3456 (x : ℝ) (h : growth_rate x) (h_pos : x = (1:ℝ)/5) : 
    expected_may_profit x = 3456 :=
by sorry

end expected_profit_may_is_3456_l592_59267


namespace campers_difference_l592_59229

theorem campers_difference 
       (total : ℕ)
       (campers_two_weeks_ago : ℕ) 
       (campers_last_week : ℕ) 
       (diff: ℕ)
       (h_total : total = 150)
       (h_two_weeks_ago : campers_two_weeks_ago = 40) 
       (h_last_week : campers_last_week = 80) : 
       diff = campers_two_weeks_ago - (total - campers_two_weeks_ago - campers_last_week) :=
by
  sorry

end campers_difference_l592_59229


namespace chess_tournament_boys_l592_59250

noncomputable def num_boys_in_tournament (n k : ℕ) : Prop :=
  (6 + k * n = (n + 2) * (n + 1) / 2) ∧ (n > 2)

theorem chess_tournament_boys :
  ∃ (n : ℕ), num_boys_in_tournament n (if n = 5 then 3 else if n = 10 then 6 else 0) ∧ (n = 5 ∨ n = 10) :=
by
  sorry

end chess_tournament_boys_l592_59250


namespace prove_intersection_points_l592_59276

noncomputable def sqrt5 := Real.sqrt 5

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5 / 2
def curve2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def curve4 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def line (x y : ℝ) : Prop := x + y = sqrt5

theorem prove_intersection_points :
  (∃! (x y : ℝ), curve1 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve3 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve4 x y ∧ line x y) :=
by
  sorry

end prove_intersection_points_l592_59276


namespace polygon_sides_l592_59281

-- Given conditions
def is_interior_angle (angle : ℝ) : Prop :=
  angle = 150

-- The theorem to prove the number of sides
theorem polygon_sides (h : is_interior_angle 150) : ∃ n : ℕ, n = 12 :=
  sorry

end polygon_sides_l592_59281
