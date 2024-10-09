import Mathlib

namespace buses_needed_l1122_112259

def total_students : ℕ := 111
def seats_per_bus : ℕ := 3

theorem buses_needed : total_students / seats_per_bus = 37 :=
by
  sorry

end buses_needed_l1122_112259


namespace entire_meal_cost_correct_l1122_112271

-- Define given conditions
def appetizer_cost : ℝ := 9.00
def entree_cost : ℝ := 20.00
def num_entrees : ℕ := 2
def dessert_cost : ℝ := 11.00
def tip_percentage : ℝ := 0.30

-- Calculate intermediate values
def total_cost_before_tip : ℝ := appetizer_cost + (entree_cost * num_entrees) + dessert_cost
def tip : ℝ := tip_percentage * total_cost_before_tip
def entire_meal_cost : ℝ := total_cost_before_tip + tip

-- Statement to be proved
theorem entire_meal_cost_correct : entire_meal_cost = 78.00 := by
  -- Proof will go here
  sorry

end entire_meal_cost_correct_l1122_112271


namespace mass_15_implies_age_7_l1122_112289

-- Define the mass function m which depends on age a
variable (m : ℕ → ℕ)

-- Define the condition for the mass to be 15 kg
def is_age_when_mass_is_15 (a : ℕ) : Prop :=
  m a = 15

-- The problem statement to be proven
theorem mass_15_implies_age_7 : ∀ a, is_age_when_mass_is_15 m a → a = 7 :=
by
  -- Proof details would follow here
  sorry

end mass_15_implies_age_7_l1122_112289


namespace find_ab_l1122_112211

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end find_ab_l1122_112211


namespace company_pays_per_box_per_month_l1122_112223

/-
  Given:
  - The dimensions of each box are 15 inches by 12 inches by 10 inches
  - The total volume occupied by all boxes is 1,080,000 cubic inches
  - The total cost for record storage per month is $480

  Prove:
  - The company pays $0.80 per box per month for record storage
-/

theorem company_pays_per_box_per_month :
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  cost_per_box_per_month = 0.80 :=
by
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  sorry

end company_pays_per_box_per_month_l1122_112223


namespace range_of_m_l1122_112220

theorem range_of_m (α : ℝ) (m : ℝ) (h1 : π < α ∧ α < 2 * π ∨ 3 * π < α ∧ α < 4 * π) 
(h2 : Real.sin α = (2 * m - 3) / (4 - m)) : 
  -1 < m ∧ m < (3 : ℝ) / 2 :=
  sorry

end range_of_m_l1122_112220


namespace total_volume_is_correct_l1122_112288

theorem total_volume_is_correct :
  let carl_side := 3
  let carl_count := 3
  let kate_side := 1.5
  let kate_count := 4
  let carl_volume := carl_count * carl_side ^ 3
  let kate_volume := kate_count * kate_side ^ 3
  carl_volume + kate_volume = 94.5 :=
by
  sorry

end total_volume_is_correct_l1122_112288


namespace solid_is_frustum_l1122_112227

-- Definitions for views
def front_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def side_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def top_view_is_concentric_circles (S : Type) : Prop := sorry

-- Define the target solid as a frustum
def is_frustum (S : Type) : Prop := sorry

-- The theorem statement
theorem solid_is_frustum
  (S : Type) 
  (h1 : front_view_is_isosceles_trapezoid S)
  (h2 : side_view_is_isosceles_trapezoid S)
  (h3 : top_view_is_concentric_circles S) :
  is_frustum S :=
sorry

end solid_is_frustum_l1122_112227


namespace linear_function_decreasing_iff_l1122_112292

-- Define the conditions
def linear_function (m b x : ℝ) : ℝ := m * x + b

-- Define the condition for decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

-- The theorem to prove
theorem linear_function_decreasing_iff (m b : ℝ) :
  (is_decreasing (linear_function m b)) ↔ (m < 0) :=
by
  sorry

end linear_function_decreasing_iff_l1122_112292


namespace total_chairs_l1122_112254

/-- Susan loves chairs. In her house, there are red chairs, yellow chairs, blue chairs, and green chairs.
    There are 5 red chairs. There are 4 times as many yellow chairs as red chairs.
    There are 2 fewer blue chairs than yellow chairs. The number of green chairs is half the sum of the number of red chairs and blue chairs (rounded down).
    We want to determine the total number of chairs in Susan's house. -/
theorem total_chairs (r y b g : ℕ) 
  (hr : r = 5)
  (hy : y = 4 * r) 
  (hb : b = y - 2) 
  (hg : g = (r + b) / 2) :
  r + y + b + g = 54 := 
sorry

end total_chairs_l1122_112254


namespace find_vector_b_l1122_112270

def vector_collinear (a b : ℝ × ℝ) : Prop :=
    ∃ k : ℝ, (a.1 = k * b.1 ∧ a.2 = k * b.2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2

theorem find_vector_b (a b : ℝ × ℝ) (h_collinear : vector_collinear a b) (h_dot : dot_product a b = -10) : b = (-4, 2) :=
    by
        sorry

end find_vector_b_l1122_112270


namespace probability_red_ball_distribution_of_X_expected_value_of_X_l1122_112280

theorem probability_red_ball :
  let pB₁ := 2 / 3
  let pB₂ := 1 / 3
  let pA_B₁ := 1 / 2
  let pA_B₂ := 3 / 4
  (pB₁ * pA_B₁ + pB₂ * pA_B₂) = 7 / 12 := by
  sorry

theorem distribution_of_X :
  let p_minus2 := 1 / 12
  let p_0 := 1 / 12
  let p_1 := 11 / 24
  let p_3 := 7 / 48
  let p_4 := 5 / 24
  let p_6 := 1 / 48
  (p_minus2 = 1 / 12) ∧ (p_0 = 1 / 12) ∧ (p_1 = 11 / 24) ∧ (p_3 = 7 / 48) ∧ (p_4 = 5 / 24) ∧ (p_6 = 1 / 48) := by
  sorry

theorem expected_value_of_X :
  let E_X := (-2 * (1 / 12) + 0 * (1 / 12) + 1 * (11 / 24) + 3 * (7 / 48) + 4 * (5 / 24) + 6 * (1 / 48))
  E_X = 27 / 16 := by
  sorry

end probability_red_ball_distribution_of_X_expected_value_of_X_l1122_112280


namespace maria_anna_ages_l1122_112274

theorem maria_anna_ages : 
  ∃ (x y : ℝ), x + y = 44 ∧ x = 2 * (y - (- (1/2) * x + (3/2) * ((2/3) * y))) ∧ x = 27.5 ∧ y = 16.5 := by 
  sorry

end maria_anna_ages_l1122_112274


namespace monotone_f_solve_inequality_range_of_a_l1122_112226

noncomputable def e := Real.exp 1
noncomputable def f (x : ℝ) : ℝ := e^x + 1/(e^x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log ((3 - a) * (f x - 1/e^x) + 1) - Real.log (3 * a) - 2 * x

-- Part 1: Monotonicity of f(x)
theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by sorry

-- Part 2: Solving the inequality f(2x) ≥ f(x + 1)
theorem solve_inequality : ∀ x : ℝ, f (2 * x) ≥ f (x + 1) ↔ x ≥ 1 ∨ x ≤ -1 / 3 :=
by sorry

-- Part 3: Finding the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x → g x a ≤ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
by sorry

end monotone_f_solve_inequality_range_of_a_l1122_112226


namespace compute_expression_l1122_112215

theorem compute_expression (w : ℂ) (hw : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) (hwp : w^11 = 1) :
  (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = -2) :=
sorry

end compute_expression_l1122_112215


namespace gasoline_price_increase_l1122_112241

theorem gasoline_price_increase 
  (P Q : ℝ)
  (h_intends_to_spend : ∃ M, M = P * Q * 1.15)
  (h_reduction : ∃ N, N = Q * (1 - 0.08))
  (h_equation : P * Q * 1.15 = P * (1 + x) * Q * (1 - 0.08)) :
  x = 0.25 :=
by
  sorry

end gasoline_price_increase_l1122_112241


namespace mass_percentage_C_in_butanoic_acid_is_54_50_l1122_112214

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_butanoic_acid : ℝ :=
  (4 * atomic_mass_C) + (8 * atomic_mass_H) + (2 * atomic_mass_O)

noncomputable def mass_of_C_in_butanoic_acid : ℝ :=
  4 * atomic_mass_C

noncomputable def mass_percentage_C : ℝ :=
  (mass_of_C_in_butanoic_acid / molar_mass_butanoic_acid) * 100

theorem mass_percentage_C_in_butanoic_acid_is_54_50 :
  mass_percentage_C = 54.50 := by
  sorry

end mass_percentage_C_in_butanoic_acid_is_54_50_l1122_112214


namespace area_of_triangle_l1122_112278

theorem area_of_triangle (a b : ℝ) 
  (hypotenuse : ℝ) (median : ℝ)
  (h_side : hypotenuse = 2)
  (h_median : median = 1)
  (h_sum : a + b = 1 + Real.sqrt 3) 
  (h_pythagorean :(a^2 + b^2 = 4)): 
  (1/2 * a * b) = (Real.sqrt 3 / 2) := 
sorry

end area_of_triangle_l1122_112278


namespace solve_for_x_l1122_112298

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l1122_112298


namespace distance_of_route_l1122_112231

theorem distance_of_route (Vq : ℝ) (Vy : ℝ) (D : ℝ) (h1 : Vy = 1.5 * Vq) (h2 : D = Vq * 2) (h3 : D = Vy * 1.3333333333333333) : D = 1.5 :=
by
  sorry

end distance_of_route_l1122_112231


namespace parallel_lines_condition_l1122_112232

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y: ℝ, (x + a * y + 6 = 0) ↔ ((a - 2) * x + 3 * y + 2 * a = 0)) ↔ a = -1 :=
by
  sorry

end parallel_lines_condition_l1122_112232


namespace find_angle_C_l1122_112203

theorem find_angle_C (A B C : ℝ) (h1 : |Real.cos A - (Real.sqrt 3 / 2)| + (1 - Real.tan B)^2 = 0) :
  C = 105 :=
by
  sorry

end find_angle_C_l1122_112203


namespace cube_side_length_l1122_112228

theorem cube_side_length (n : ℕ) (h : n^3 - (n-2)^3 = 98) : n = 5 :=
by sorry

end cube_side_length_l1122_112228


namespace total_participants_l1122_112205

theorem total_participants (F M : ℕ)
  (h1 : F / 2 = 130)
  (h2 : F / 2 + M / 4 = (F + M) / 3) : 
  F + M = 780 := 
by 
  sorry

end total_participants_l1122_112205


namespace eval_expression_l1122_112239

theorem eval_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x) (hz' : z ≠ -x) :
  ((x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1) :=
by
  sorry

end eval_expression_l1122_112239


namespace color_cube_color_octahedron_l1122_112250

theorem color_cube (colors : Fin 6) : ∃ (ways : Nat), ways = 30 :=
  sorry

theorem color_octahedron (colors : Fin 8) : ∃ (ways : Nat), ways = 1680 :=
  sorry

end color_cube_color_octahedron_l1122_112250


namespace jacques_suitcase_weight_l1122_112295

noncomputable def suitcase_weight_on_return : ℝ := 
  let initial_weight := 12
  let perfume_weight := (5 * 1.2) / 16
  let chocolate_weight := 4 + 1.5 + 3.25
  let soap_weight := (2 * 5) / 16
  let jam_weight := (8 + 6 + 10 + 12) / 16
  let sculpture_weight := 3.5 * 2.20462
  let shirts_weight := (3 * 300 * 0.03527396) / 16
  let cookies_weight := (450 * 0.03527396) / 16
  let wine_weight := (190 * 0.03527396) / 16
  initial_weight + perfume_weight + chocolate_weight + soap_weight + jam_weight + sculpture_weight + shirts_weight + cookies_weight + wine_weight

theorem jacques_suitcase_weight : suitcase_weight_on_return = 35.111288 := 
by 
  -- Calculation to verify that the total is 35.111288
  sorry

end jacques_suitcase_weight_l1122_112295


namespace maya_additional_cars_l1122_112202

theorem maya_additional_cars : 
  ∃ n : ℕ, 29 + n ≥ 35 ∧ (29 + n) % 7 = 0 ∧ n = 6 :=
by
  sorry

end maya_additional_cars_l1122_112202


namespace terminating_decimal_expansion_of_17_div_625_l1122_112230

theorem terminating_decimal_expansion_of_17_div_625 : 
  ∃ d : ℚ, d = 17 / 625 ∧ d = 0.0272 :=
by
  sorry

end terminating_decimal_expansion_of_17_div_625_l1122_112230


namespace largest_angle_triangl_DEF_l1122_112201

theorem largest_angle_triangl_DEF (d e f : ℝ) (h1 : d + 3 * e + 3 * f = d^2)
  (h2 : d + 3 * e - 3 * f = -8) : 
  ∃ (F : ℝ), F = 109.47 ∧ (F > 90) := by sorry

end largest_angle_triangl_DEF_l1122_112201


namespace cap_to_sunglasses_prob_l1122_112297

-- Define the conditions
def num_people_wearing_sunglasses : ℕ := 60
def num_people_wearing_caps : ℕ := 40
def prob_sunglasses_and_caps : ℚ := 1 / 3

-- Define the statement to prove
theorem cap_to_sunglasses_prob : 
  (num_people_wearing_sunglasses * prob_sunglasses_and_caps) / num_people_wearing_caps = 1 / 2 :=
by
  sorry

end cap_to_sunglasses_prob_l1122_112297


namespace arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l1122_112279

-- Statement for Question 1
theorem arrangement_ways_13_books : 
  (Nat.factorial 13) = 6227020800 := 
sorry

-- Statement for Question 2
theorem arrangement_ways_13_books_with_4_arithmetic_together :
  (Nat.factorial 10) * (Nat.factorial 4) = 87091200 := 
sorry

-- Statement for Question 3
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_together :
  (Nat.factorial 5) * (Nat.factorial 4) * (Nat.factorial 6) = 2073600 := 
sorry

-- Statement for Question 4
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together :
  (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 6) * (Nat.factorial 3) = 622080 := 
sorry

end arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l1122_112279


namespace nick_coin_collection_l1122_112200

theorem nick_coin_collection
  (total_coins : ℕ)
  (quarters_coins : ℕ)
  (dimes_coins : ℕ)
  (nickels_coins : ℕ)
  (state_quarters : ℕ)
  (pa_state_quarters : ℕ)
  (roosevelt_dimes : ℕ)
  (h_total : total_coins = 50)
  (h_quarters : quarters_coins = total_coins * 3 / 10)
  (h_dimes : dimes_coins = total_coins * 40 / 100)
  (h_nickels : nickels_coins = total_coins - (quarters_coins + dimes_coins))
  (h_state_quarters : state_quarters = quarters_coins * 2 / 5)
  (h_pa_state_quarters : pa_state_quarters = state_quarters * 3 / 8)
  (h_roosevelt_dimes : roosevelt_dimes = dimes_coins * 75 / 100) :
  pa_state_quarters = 2 ∧ roosevelt_dimes = 15 ∧ nickels_coins = 15 :=
by
  sorry

end nick_coin_collection_l1122_112200


namespace value_of_expression_l1122_112247

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end value_of_expression_l1122_112247


namespace candies_share_equally_l1122_112273

theorem candies_share_equally (mark_candies : ℕ) (peter_candies : ℕ) (john_candies : ℕ)
  (h_mark : mark_candies = 30) (h_peter : peter_candies = 25) (h_john : john_candies = 35) :
  (mark_candies + peter_candies + john_candies) / 3 = 30 :=
by
  sorry

end candies_share_equally_l1122_112273


namespace nurses_count_l1122_112235

theorem nurses_count (total_medical_staff : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (total_ratio_parts : ℕ) (h1 : total_medical_staff = 200) 
  (h2 : ratio_doctors = 4) (h3 : ratio_nurses = 6) (h4 : total_ratio_parts = ratio_doctors + ratio_nurses) :
  (ratio_nurses * total_medical_staff) / total_ratio_parts = 120 :=
by
  sorry

end nurses_count_l1122_112235


namespace andrew_made_35_sandwiches_l1122_112263

-- Define the number of friends and sandwiches per friend
def num_friends : ℕ := 7
def sandwiches_per_friend : ℕ := 5

-- Define the total number of sandwiches and prove it equals 35
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_made_35_sandwiches : total_sandwiches = 35 := by
  sorry

end andrew_made_35_sandwiches_l1122_112263


namespace train_speed_l1122_112266

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) (h_train_length : train_length = 100) (h_bridge_length : bridge_length = 300) (h_crossing_time : crossing_time = 12) : 
  (train_length + bridge_length) / crossing_time = 33.33 := 
by 
  -- sorry allows us to skip the proof
  sorry

end train_speed_l1122_112266


namespace smallest_prime_x_l1122_112224

-- Define prime number checker
def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem conditions and proof goal
theorem smallest_prime_x 
  (x y z : ℕ) 
  (hx : is_prime x)
  (hy : is_prime y)
  (hz : is_prime z)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hyz : y ≠ z)
  (hd : ∀ d : ℕ, d ∣ (x * x * y * z) ↔ (d = 1 ∨ d = x ∨ d = x * x ∨ d = y ∨ d = x * y ∨ d = x * x * y ∨ d = z ∨ d = x * z ∨ d = x * x * z ∨ d = y * z ∨ d = x * y * z ∨ d = x * x * y * z)) 
  : x = 2 := 
sorry

end smallest_prime_x_l1122_112224


namespace probability_green_dinosaur_or_blue_robot_l1122_112268

theorem probability_green_dinosaur_or_blue_robot (t: ℕ) (blue_dinosaurs green_robots blue_robots: ℕ) 
(h1: blue_dinosaurs = 16) (h2: green_robots = 14) (h3: blue_robots = 36) (h4: t = 93):
  t = 93 → (blue_dinosaurs = 16) → (green_robots = 14) → (blue_robots = 36) → 
  (∃ green_dinosaurs: ℕ, t = blue_dinosaurs + green_robots + blue_robots + green_dinosaurs ∧ 
    (∃ k: ℕ, k = (green_dinosaurs + blue_robots) / (t / 31) ∧ k = 21 / 31)) := sorry

end probability_green_dinosaur_or_blue_robot_l1122_112268


namespace smallest_k_DIVISIBLE_by_3_67_l1122_112249

theorem smallest_k_DIVISIBLE_by_3_67 :
  ∃ k : ℕ, (∀ n : ℕ, (2016^k % 3^67 = 0 ∧ (2016^n % 3^67 = 0 → k ≤ n)) ∧ k = 34) := by
  sorry

end smallest_k_DIVISIBLE_by_3_67_l1122_112249


namespace range_of_a_l1122_112229

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

noncomputable def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem range_of_a : 
  (∀ x ∈ Set.Iio 0, ∃ y, f x = g y a ∧ y = -x) →
  a < Real.sqrt (Real.exp 1) :=
  sorry

end range_of_a_l1122_112229


namespace distance_between_lines_l1122_112282

noncomputable def distance_between_parallel_lines
  (a b m n : ℝ) : ℝ :=
  |m - n| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines
  (a b m n : ℝ) :
  distance_between_parallel_lines a b m n = 
  |m - n| / Real.sqrt (a^2 + b^2) :=
by
  sorry

end distance_between_lines_l1122_112282


namespace distance_travelled_downstream_l1122_112238

theorem distance_travelled_downstream :
  let speed_boat_still_water := 42 -- km/hr
  let rate_current := 7 -- km/hr
  let time_travelled_min := 44 -- minutes
  let time_travelled_hrs := time_travelled_min / 60.0 -- converting minutes to hours
  let effective_speed_downstream := speed_boat_still_water + rate_current -- km/hr
  let distance_downstream := effective_speed_downstream * time_travelled_hrs
  distance_downstream = 35.93 :=
by
  -- Proof will go here
  sorry

end distance_travelled_downstream_l1122_112238


namespace johns_profit_l1122_112251

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end johns_profit_l1122_112251


namespace find_x_l1122_112299

theorem find_x 
  (x y z : ℝ)
  (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
  (h2 : y + z = 110) 
  : x = 106 := 
by 
  sorry

end find_x_l1122_112299


namespace tigers_wins_l1122_112252

def totalGames : ℕ := 56
def losses : ℕ := 12
def ties : ℕ := losses / 2

theorem tigers_wins : totalGames - losses - ties = 38 := by
  sorry

end tigers_wins_l1122_112252


namespace height_of_tower_l1122_112225

-- Definitions for points and distances
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 0, y := 0, z := 0 }
def C : Point := { x := 0, y := 0, z := 129 }
def D : Point := { x := 0, y := 0, z := 258 }
def B : Point  := { x := 0, y := 305, z := 305 }

-- Given conditions
def angle_elevation_A_to_B : ℝ := 45 -- degrees
def angle_elevation_D_to_B : ℝ := 60 -- degrees
def distance_A_to_D : ℝ := 258 -- meters

-- The problem is to prove the height of the tower is 305 meters given the conditions
theorem height_of_tower : B.y = 305 :=
by
  -- This spot would contain the actual proof
  sorry

end height_of_tower_l1122_112225


namespace employed_females_percentage_l1122_112204

-- Definitions of the conditions
def employment_rate : ℝ := 0.60
def male_employment_rate : ℝ := 0.15

-- The theorem to prove
theorem employed_females_percentage : employment_rate - male_employment_rate = 0.45 := by
  sorry

end employed_females_percentage_l1122_112204


namespace geometric_sequence_proof_l1122_112237

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h1 : q > 1) (h2 : a 1 > 0)
    (h3 : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - (a 5)^2 = 9) :
  a 3 - a 7 = -3 :=
by sorry

end geometric_sequence_proof_l1122_112237


namespace number_line_distance_l1122_112293

theorem number_line_distance (x : ℝ) : |x + 1| = 6 ↔ (x = 5 ∨ x = -7) :=
by
  sorry

end number_line_distance_l1122_112293


namespace simplify_expression_l1122_112234

variable {x y : ℝ}

theorem simplify_expression : (x^5 * x^3 * y^2 * y^4) = (x^8 * y^6) := by
  sorry

end simplify_expression_l1122_112234


namespace large_number_divisible_by_12_l1122_112244

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l1122_112244


namespace fraction_multiplication_l1122_112236

theorem fraction_multiplication (x : ℚ) (h : x = 236 / 100) : x * 3 = 177 / 25 :=
by
  sorry

end fraction_multiplication_l1122_112236


namespace ganesh_average_speed_l1122_112210

variable (D : ℝ) -- distance between the two towns in kilometers
variable (V : ℝ) -- average speed from x to y in km/hr

-- Conditions
variable (h1 : V > 0) -- Speed must be positive
variable (h2 : 30 > 0) -- Speed must be positive
variable (h3 : 40 = (2 * D) / ((D / V) + (D / 30))) -- Average speed formula

theorem ganesh_average_speed : V = 60 :=
by {
  sorry
}

end ganesh_average_speed_l1122_112210


namespace find_some_number_l1122_112287

theorem find_some_number (some_number : ℝ) (h : (3.242 * some_number) / 100 = 0.045388) : some_number = 1.400 := 
sorry

end find_some_number_l1122_112287


namespace Jose_share_land_l1122_112208

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end Jose_share_land_l1122_112208


namespace nested_expression_evaluation_l1122_112218

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 :=
by
  sorry

end nested_expression_evaluation_l1122_112218


namespace log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l1122_112285

-- Define irrational numbers in Lean
def irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Prove that log base 2 of 3 is irrational
theorem log_two_three_irrational : irrational (Real.log 3 / Real.log 2) := 
sorry

-- Prove that log base sqrt(2) of 3 is irrational
theorem log_sqrt2_three_irrational : 
  irrational (Real.log 3 / (1/2 * Real.log 2)) := 
sorry

-- Prove that log base (5 + 3sqrt(2)) of (3 + 5sqrt(2)) is irrational
theorem log_five_plus_three_sqrt2_irrational :
  irrational (Real.log (3 + 5 * Real.sqrt 2) / Real.log (5 + 3 * Real.sqrt 2)) := 
sorry

end log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l1122_112285


namespace square_area_l1122_112255

theorem square_area (x : ℚ) (side_length : ℚ) 
  (h1 : side_length = 3 * x - 12) 
  (h2 : side_length = 24 - 2 * x) : 
  side_length ^ 2 = 92.16 := 
by 
  sorry

end square_area_l1122_112255


namespace complement_of_P_in_U_l1122_112216

def universal_set : Set ℝ := Set.univ
def set_P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_in_U (U : Set ℝ) (P : Set ℝ) : Set ℝ := U \ P

theorem complement_of_P_in_U :
  complement_in_U universal_set set_P = { x | -1 < x ∧ x < 6 } :=
by
  sorry

end complement_of_P_in_U_l1122_112216


namespace work_days_for_A_and_B_l1122_112243

theorem work_days_for_A_and_B (W_A W_B : ℝ) (h1 : W_A = (1/2) * W_B) (h2 : W_B = 1/21) : 
  1 / (W_A + W_B) = 14 := by 
  sorry

end work_days_for_A_and_B_l1122_112243


namespace tank_full_capacity_l1122_112248

variable (T : ℝ) -- Define T as a real number representing the total capacity of the tank.

-- The main condition: (3 / 4) * T + 5 = (7 / 8) * T
axiom condition : (3 / 4) * T + 5 = (7 / 8) * T

-- Proof statement: Prove that T = 40
theorem tank_full_capacity : T = 40 :=
by
  -- Using the given condition to derive that T = 40.
  sorry

end tank_full_capacity_l1122_112248


namespace orchid_bushes_after_planting_l1122_112209

def total_orchid_bushes (current_orchids new_orchids : Nat) : Nat :=
  current_orchids + new_orchids

theorem orchid_bushes_after_planting :
  ∀ (current_orchids new_orchids : Nat), current_orchids = 22 → new_orchids = 13 → total_orchid_bushes current_orchids new_orchids = 35 :=
by
  intros current_orchids new_orchids h_current h_new
  rw [h_current, h_new]
  exact rfl

end orchid_bushes_after_planting_l1122_112209


namespace greatest_divisor_of_remainders_l1122_112233

theorem greatest_divisor_of_remainders (x : ℕ) :
  (1442 % x = 12) ∧ (1816 % x = 6) ↔ x = 10 :=
by
  sorry

end greatest_divisor_of_remainders_l1122_112233


namespace ratio_of_girls_to_boys_l1122_112213

theorem ratio_of_girls_to_boys (g b : ℕ) (h1 : g = b + 6) (h2 : g + b = 36) : g / b = 7 / 5 := by sorry

end ratio_of_girls_to_boys_l1122_112213


namespace domain_and_range_of_f_l1122_112240

noncomputable def f (a x : ℝ) : ℝ := Real.log (a - a * x) / Real.log a

theorem domain_and_range_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a - a * x > 0 → x < 1) ∧ 
  (∀ t : ℝ, 0 < t ∧ t < a → ∃ x : ℝ, t = a - a * x) :=
by
  sorry

end domain_and_range_of_f_l1122_112240


namespace range_of_m_l1122_112265

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m-2) * x^2 + 2 * m * x - (3 - m)

theorem range_of_m (m : ℝ) (h_vertex_third_quadrant : (-(m) / (m-2) < 0) ∧ ((-5)*m + 6) / (m-2) < 0)
                   (h_parabola_opens_upwards : m - 2 > 0)
                   (h_intersects_negative_y_axis : m < 3) : 2 < m ∧ m < 3 :=
by {
    sorry
}

end range_of_m_l1122_112265


namespace sum_reciprocals_of_factors_12_l1122_112291

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l1122_112291


namespace Z_4_3_eq_37_l1122_112206

def Z (a b : ℕ) : ℕ :=
  a^2 + a * b + b^2

theorem Z_4_3_eq_37 : Z 4 3 = 37 :=
  by
    sorry

end Z_4_3_eq_37_l1122_112206


namespace principal_amount_borrowed_l1122_112258

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end principal_amount_borrowed_l1122_112258


namespace asymptotes_of_hyperbola_l1122_112260

theorem asymptotes_of_hyperbola (a b : ℝ) (h_cond1 : a > b) (h_cond2 : b > 0) 
  (h_eq_ell : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h_eq_hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h_product : ∀ e1 e2 : ℝ, (e1 = Real.sqrt (1 - (b^2 / a^2))) → 
                (e2 = Real.sqrt (1 + (b^2 / a^2))) → 
                (e1 * e2 = Real.sqrt 3 / 2)) :
  ∀ x y : ℝ, x + Real.sqrt 2 * y = 0 ∨ x - Real.sqrt 2 * y = 0 :=
sorry

end asymptotes_of_hyperbola_l1122_112260


namespace min_value_of_reciprocal_sums_l1122_112284

variable {a b : ℝ}

theorem min_value_of_reciprocal_sums (ha : a ≠ 0) (hb : b ≠ 0) (h : 4 * a ^ 2 + b ^ 2 = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) = 9 := by
  sorry

end min_value_of_reciprocal_sums_l1122_112284


namespace final_limes_count_l1122_112283

def limes_initial : ℕ := 9
def limes_by_Sara : ℕ := 4
def limes_used_for_juice : ℕ := 5
def limes_given_to_neighbor : ℕ := 3

theorem final_limes_count :
  limes_initial + limes_by_Sara - limes_used_for_juice - limes_given_to_neighbor = 5 :=
by
  sorry

end final_limes_count_l1122_112283


namespace two_digit_number_condition_l1122_112257

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end two_digit_number_condition_l1122_112257


namespace third_student_number_l1122_112290

theorem third_student_number (A B C D : ℕ) 
  (h1 : A + B + C + D = 531) 
  (h2 : A + B = C + D + 31) 
  (h3 : C = D + 22) : 
  C = 136 := 
by
  sorry

end third_student_number_l1122_112290


namespace solve_for_x_l1122_112262

theorem solve_for_x (x : ℝ) (h : (x - 5)^4 = (1 / 16)⁻¹) : x = 7 :=
by
  sorry

end solve_for_x_l1122_112262


namespace area_of_yard_l1122_112264

theorem area_of_yard (L W : ℕ) (h1 : L = 40) (h2 : L + 2 * W = 64) : L * W = 480 := by
  sorry

end area_of_yard_l1122_112264


namespace prove_remainder_l1122_112261

def problem_statement : Prop := (33333332 % 8 = 4)

theorem prove_remainder : problem_statement := 
by
  sorry

end prove_remainder_l1122_112261


namespace tabletop_qualification_l1122_112222

theorem tabletop_qualification (length width diagonal : ℕ) :
  length = 60 → width = 32 → diagonal = 68 → (diagonal * diagonal = length * length + width * width) :=
by
  intros
  sorry

end tabletop_qualification_l1122_112222


namespace number_of_teams_l1122_112272

theorem number_of_teams (n : ℕ) (G : ℕ) (h1 : G = 28) (h2 : G = n * (n - 1) / 2) : n = 8 := 
  by
  -- Proof skipped
  sorry

end number_of_teams_l1122_112272


namespace abc_inequality_l1122_112296

theorem abc_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a = 1) :
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) ≥ (a * b + b * c + c * a)^2 :=
sorry

end abc_inequality_l1122_112296


namespace arithmetic_sequence_a4_a5_sum_l1122_112275

theorem arithmetic_sequence_a4_a5_sum
  (a_n : ℕ → ℝ)
  (a1_a2_sum : a_n 1 + a_n 2 = -1)
  (a3_val : a_n 3 = 4)
  (h_arith : ∃ d : ℝ, ∀ (n : ℕ), a_n (n + 1) = a_n n + d) :
  a_n 4 + a_n 5 = 17 := 
by
  sorry

end arithmetic_sequence_a4_a5_sum_l1122_112275


namespace george_hours_tuesday_l1122_112221

def wage_per_hour := 5
def hours_monday := 7
def total_earnings := 45

theorem george_hours_tuesday : ∃ (hours_tuesday : ℕ), 
  hours_tuesday = (total_earnings - (hours_monday * wage_per_hour)) / wage_per_hour := 
by
  sorry

end george_hours_tuesday_l1122_112221


namespace sara_total_spent_l1122_112276

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end sara_total_spent_l1122_112276


namespace length_of_segment_AB_l1122_112245

-- Define the parabola and its properties
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ C.1 = 3

-- Main statement of the problem
theorem length_of_segment_AB
  (A B : ℝ × ℝ)
  (hA : parabola_equation A.1 A.2)
  (hB : parabola_equation B.1 B.2)
  (C : ℝ × ℝ)
  (hfoc : focus (1, 0))
  (hm : midpoint_condition A B C) :
  dist A B = 8 :=
by sorry

end length_of_segment_AB_l1122_112245


namespace solve_fraction_equation_l1122_112277

def fraction_equation (x : ℝ) : Prop :=
  1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1) = 5

theorem solve_fraction_equation (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  fraction_equation x → 
  x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 :=
by
  sorry

end solve_fraction_equation_l1122_112277


namespace proportion_solution_l1122_112269

theorem proportion_solution (x : ℝ) (h : 0.6 / x = 5 / 8) : x = 0.96 :=
by 
  -- The proof will go here
  sorry

end proportion_solution_l1122_112269


namespace problem_counts_correct_pairs_l1122_112212

noncomputable def count_valid_pairs : ℝ :=
  sorry

theorem problem_counts_correct_pairs :
  count_valid_pairs = 128 :=
by
  sorry

end problem_counts_correct_pairs_l1122_112212


namespace find_abc_l1122_112286

theorem find_abc
  {a b c : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30)
  (h2 : 1/a + 1/b + 1/c + 672/(a*b*c) = 1) :
  a * b * c = 2808 :=
sorry

end find_abc_l1122_112286


namespace numbers_divisible_by_three_l1122_112207

theorem numbers_divisible_by_three (a b : ℕ) (h1 : a = 150) (h2 : b = 450) :
  ∃ n : ℕ, ∀ x : ℕ, (a < x) → (x < b) → (x % 3 = 0) → (x = 153 + 3 * (n - 1)) :=
by
  sorry

end numbers_divisible_by_three_l1122_112207


namespace arithmetic_square_root_of_4_l1122_112281

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end arithmetic_square_root_of_4_l1122_112281


namespace a_completes_in_12_days_l1122_112294

def work_rate_a_b (r_A r_B : ℝ) := r_A + r_B = 1 / 3
def work_rate_b_c (r_B r_C : ℝ) := r_B + r_C = 1 / 2
def work_rate_a_c (r_A r_C : ℝ) := r_A + r_C = 1 / 3

theorem a_completes_in_12_days (r_A r_B r_C : ℝ) 
  (h1 : work_rate_a_b r_A r_B)
  (h2 : work_rate_b_c r_B r_C)
  (h3 : work_rate_a_c r_A r_C) : 
  1 / r_A = 12 :=
by
  sorry

end a_completes_in_12_days_l1122_112294


namespace central_angle_remains_unchanged_l1122_112253

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end central_angle_remains_unchanged_l1122_112253


namespace hyperbolas_same_asymptotes_l1122_112217

theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) ↔ (y^2 / 25 - x^2 / M = 1)) → M = 225 / 16 :=
by
  sorry

end hyperbolas_same_asymptotes_l1122_112217


namespace total_area_of_figure_l1122_112242

theorem total_area_of_figure :
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  total_area = 89 := by
  -- Definitions
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  -- Proof
  sorry

end total_area_of_figure_l1122_112242


namespace find_expression_for_f_l1122_112267

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem find_expression_for_f (x : ℝ) (h : x ≠ -1) 
    (hf : f ((1 - x) / (1 + x)) = x) : 
    f x = (1 - x) / (1 + x) :=
sorry

end find_expression_for_f_l1122_112267


namespace cloth_meters_sold_l1122_112219

-- Conditions as definitions
def total_selling_price : ℝ := 4500
def profit_per_meter : ℝ := 14
def cost_price_per_meter : ℝ := 86

-- The statement of the problem
theorem cloth_meters_sold (SP : ℝ := cost_price_per_meter + profit_per_meter) :
  total_selling_price / SP = 45 := by
  sorry

end cloth_meters_sold_l1122_112219


namespace evaluate_expression_l1122_112256

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5 * x + 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end evaluate_expression_l1122_112256


namespace function_identity_l1122_112246

variables {R : Type*} [LinearOrderedField R]

-- Define real-valued functions f, g, h
variables (f g h : R → R)

-- Define function composition and multiplication
def comp (f g : R → R) (x : R) := f (g x)
def mul (f g : R → R) (x : R) := f x * g x

-- The statement to prove
theorem function_identity (x : R) : 
  comp (mul f g) h x = mul (comp f h) (comp g h) x :=
sorry

end function_identity_l1122_112246
