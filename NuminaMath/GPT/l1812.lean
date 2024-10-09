import Mathlib

namespace exists_divisor_between_l1812_181209

theorem exists_divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) 
  (h_div1 : a ∣ n) (h_div2 : b ∣ n) (h_neq : a ≠ b) 
  (h_lt : a < b) (h_eq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end exists_divisor_between_l1812_181209


namespace cake_volume_l1812_181298

theorem cake_volume :
  let thickness := 1 / 2
  let diameter := 16
  let radius := diameter / 2
  let total_volume := Real.pi * radius^2 * thickness
  total_volume / 16 = 2 * Real.pi := by
    sorry

end cake_volume_l1812_181298


namespace similarity_ratio_of_polygons_l1812_181264

theorem similarity_ratio_of_polygons (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : a / (b : ℚ) = 3 / 5 :=
by 
  sorry

end similarity_ratio_of_polygons_l1812_181264


namespace tangent_line_hyperbola_l1812_181252

variable {a b x x₀ y y₀ : ℝ}
variable (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (he : x₀^2 / a^2 + y₀^2 / b^2 = 1)
variable (hh : x₀^2 / a^2 - y₀^2 / b^2 = 1)

theorem tangent_line_hyperbola
  (h_tangent_ellipse : (x₀ * x / a^2 + y₀ * y / b^2 = 1)) :
  (x₀ * x / a^2 - y₀ * y / b^2 = 1) :=
sorry

end tangent_line_hyperbola_l1812_181252


namespace initial_girls_count_l1812_181271

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end initial_girls_count_l1812_181271


namespace quadratic_real_roots_l1812_181234

theorem quadratic_real_roots (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 - 1 + m = 0 ∧ x2^2 + 2 * x2 - 1 + m = 0) ↔ m ≤ 2 :=
by
  sorry

end quadratic_real_roots_l1812_181234


namespace min_fraction_value_l1812_181237

theorem min_fraction_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_tangent : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 :=
by
  sorry

end min_fraction_value_l1812_181237


namespace dice_surface_sum_l1812_181220

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l1812_181220


namespace no_intersection_points_l1812_181243

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := -x^2 + 6 * x - 8

-- The statement asserting that the parabolas do not intersect
theorem no_intersection_points :
  ∀ (x y : ℝ), parabola1 x = y → parabola2 x = y → false :=
by
  -- Introducing x and y as elements of the real numbers
  intros x y h1 h2
  
  -- Since this is only the statement, we use sorry to skip the actual proof
  sorry

end no_intersection_points_l1812_181243


namespace correct_expression_l1812_181238

theorem correct_expression (x : ℝ) :
  (x^3 / x^2 = x) :=
by sorry

end correct_expression_l1812_181238


namespace number_of_intersection_points_l1812_181230

theorem number_of_intersection_points : 
  ∃! (P : ℝ × ℝ), 
    (P.1 ^ 2 + P.2 ^ 2 = 16) ∧ (P.1 = 4) := 
by
  sorry

end number_of_intersection_points_l1812_181230


namespace cows_and_goats_sum_l1812_181286

theorem cows_and_goats_sum (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 4 * x + 2 * y + 4 * z = 18 + 2 * (x + y + z)) 
  : x + z = 9 := by 
  sorry

end cows_and_goats_sum_l1812_181286


namespace mia_min_stamps_l1812_181223

theorem mia_min_stamps (x y : ℕ) (hx : 5 * x + 7 * y = 37) : x + y = 7 :=
sorry

end mia_min_stamps_l1812_181223


namespace percentage_problem_l1812_181204

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by
  sorry

end percentage_problem_l1812_181204


namespace find_retail_price_l1812_181221

-- Define the conditions
def wholesale_price : ℝ := 90
def discount_rate : ℝ := 0.10
def profit_rate : ℝ := 0.20

-- Calculate the necessary values from conditions
def profit : ℝ := profit_rate * wholesale_price
def selling_price : ℝ := wholesale_price + profit
def discount_factor : ℝ := 1 - discount_rate

-- Rewrite the main theorem statement
theorem find_retail_price : ∃ w : ℝ, discount_factor * w = selling_price → w = 120 :=
by sorry

end find_retail_price_l1812_181221


namespace solution_set_of_inequality_l1812_181202

theorem solution_set_of_inequality (x: ℝ) : 
  (1 / x ≤ 1) ↔ (x < 0 ∨ x ≥ 1) :=
sorry

end solution_set_of_inequality_l1812_181202


namespace sports_club_membership_l1812_181269

theorem sports_club_membership (B T Both Neither : ℕ) (hB : B = 17) (hT : T = 19) (hBoth : Both = 11) (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end sports_club_membership_l1812_181269


namespace becky_to_aliyah_ratio_l1812_181256

def total_school_days : ℕ := 180
def days_aliyah_packs_lunch : ℕ := total_school_days / 2
def days_becky_packs_lunch : ℕ := 45

theorem becky_to_aliyah_ratio :
  (days_becky_packs_lunch : ℚ) / days_aliyah_packs_lunch = 1 / 2 := by
  sorry

end becky_to_aliyah_ratio_l1812_181256


namespace average_of_second_set_l1812_181262

open Real

theorem average_of_second_set 
  (avg6 : ℝ)
  (n1 n2 n3 n4 n5 n6 : ℝ)
  (avg1_set : ℝ)
  (avg3_set : ℝ)
  (h1 : avg6 = 3.95)
  (h2 : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = avg6)
  (h3 : (n1 + n2) / 2 = 3.6)
  (h4 : (n5 + n6) / 2 = 4.400000000000001) :
  (n3 + n4) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_l1812_181262


namespace evaluate_expression_l1812_181212

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end evaluate_expression_l1812_181212


namespace rotational_transform_preserves_expression_l1812_181284

theorem rotational_transform_preserves_expression
  (a b c : ℝ)
  (ϕ : ℝ)
  (a1 b1 c1 : ℝ)
  (x' y' x'' y'' : ℝ)
  (h1 : x'' = x' * Real.cos ϕ + y' * Real.sin ϕ)
  (h2 : y'' = -x' * Real.sin ϕ + y' * Real.cos ϕ)
  (def_a1 : a1 = a * (Real.cos ϕ)^2 - 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.sin ϕ)^2)
  (def_b1 : b1 = a * (Real.cos ϕ) * (Real.sin ϕ) + b * ((Real.cos ϕ)^2 - (Real.sin ϕ)^2) - c * (Real.cos ϕ) * (Real.sin ϕ))
  (def_c1 : c1 = a * (Real.sin ϕ)^2 + 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.cos ϕ)^2) :
  a1 * c1 - b1^2 = a * c - b^2 := sorry

end rotational_transform_preserves_expression_l1812_181284


namespace isosceles_triangle_perimeter_l1812_181208

-- Define an isosceles triangle structure
structure IsoscelesTriangle where
  (a b c : ℝ) 
  (isosceles : a = b ∨ a = c ∨ b = c)
  (side_lengths : (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧ (c = 2 ∨ c = 3))
  (valid_triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem to prove the perimeter
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.a + t.b + t.c = 7 ∨ t.a + t.b + t.c = 8 :=
sorry

end isosceles_triangle_perimeter_l1812_181208


namespace books_checked_out_on_Thursday_l1812_181282

theorem books_checked_out_on_Thursday (initial_books : ℕ) (wednesday_checked_out : ℕ) 
                                      (thursday_returned : ℕ) (friday_returned : ℕ) (final_books : ℕ) 
                                      (thursday_checked_out : ℕ) : 
  (initial_books = 98) → 
  (wednesday_checked_out = 43) → 
  (thursday_returned = 23) → 
  (friday_returned = 7) → 
  (final_books = 80) → 
  (initial_books - wednesday_checked_out + thursday_returned - thursday_checked_out + friday_returned = final_books) → 
  (thursday_checked_out = 5) :=
by
  intros
  sorry

end books_checked_out_on_Thursday_l1812_181282


namespace fraction_of_seats_sold_l1812_181201

theorem fraction_of_seats_sold
  (ticket_price : ℕ) (number_of_rows : ℕ) (seats_per_row : ℕ) (total_earnings : ℕ)
  (h1 : ticket_price = 10)
  (h2 : number_of_rows = 20)
  (h3 : seats_per_row = 10)
  (h4 : total_earnings = 1500) :
  (total_earnings / ticket_price : ℕ) / (number_of_rows * seats_per_row : ℕ) = 3 / 4 := by
  sorry

end fraction_of_seats_sold_l1812_181201


namespace evaluate_expression_l1812_181250

theorem evaluate_expression : 3 ^ 123 + 9 ^ 5 / 9 ^ 3 = 3 ^ 123 + 81 :=
by
  -- we add sorry as the proof is not required
  sorry

end evaluate_expression_l1812_181250


namespace age_difference_is_12_l1812_181219

noncomputable def age_difference (x : ℕ) : ℕ :=
  let older := 3 * x
  let younger := 2 * x
  older - younger

theorem age_difference_is_12 :
  ∃ x : ℕ, 3 * x + 2 * x = 60 ∧ age_difference x = 12 :=
by
  sorry

end age_difference_is_12_l1812_181219


namespace unique_solution_of_quadratic_l1812_181281

theorem unique_solution_of_quadratic :
  ∀ (b : ℝ), b ≠ 0 → (∃ x : ℝ, 3 * x^2 + b * x + 12 = 0 ∧ ∀ y : ℝ, 3 * y^2 + b * y + 12 = 0 → y = x) → 
  (b = 12 ∧ ∃ x : ℝ, x = -2 ∧ 3 * x^2 + 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 + 12 * y + 12 = 0 → y = x)) ∨ 
  (b = -12 ∧ ∃ x : ℝ, x = 2 ∧ 3 * x^2 - 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 - 12 * y + 12 = 0 → y = x)) :=
by 
  sorry

end unique_solution_of_quadratic_l1812_181281


namespace target_hit_probability_l1812_181258

/-- 
The probabilities for two shooters to hit a target are 1/2 and 1/3, respectively.
If both shooters fire at the target simultaneously, the probability that the target 
will be hit is 2/3.
-/
theorem target_hit_probability (P₁ P₂ : ℚ) (h₁ : P₁ = 1/2) (h₂ : P₂ = 1/3) :
  1 - ((1 - P₁) * (1 - P₂)) = 2/3 :=
by
  sorry

end target_hit_probability_l1812_181258


namespace min_value_fraction_l1812_181293

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  (∃ y, y > 9 ∧ (∀ z, z > 9 → y ≤ (z^3 / (z - 9)))) ∧ (∀ z, z > 9 → (∃ w, w > 9 ∧ z^3 / (z - 9) = 325)) := 
  sorry

end min_value_fraction_l1812_181293


namespace percentage_less_than_l1812_181228

theorem percentage_less_than (x y : ℝ) (P : ℝ) (h1 : y = 1.6667 * x) (h2 : x = (1 - P / 100) * y) : P = 66.67 :=
sorry

end percentage_less_than_l1812_181228


namespace max_ab_value_l1812_181248

theorem max_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_perpendicular : (2 * a - 1) * b = -1) : ab <= 1 / 8 := by
  sorry

end max_ab_value_l1812_181248


namespace length_of_adult_bed_is_20_decimeters_l1812_181297

-- Define the length of an adult bed as per question context
def length_of_adult_bed := 20

-- Prove that the length of an adult bed in decimeters equals 20
theorem length_of_adult_bed_is_20_decimeters : length_of_adult_bed = 20 :=
by
  -- Proof goes here
  sorry

end length_of_adult_bed_is_20_decimeters_l1812_181297


namespace lcm_18_45_l1812_181253

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l1812_181253


namespace inflation_over_two_years_real_yield_deposit_second_year_l1812_181241

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l1812_181241


namespace algebraic_expression_value_l1812_181283

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b = 2) : 
  (-a * (-2) ^ 2 + b * (-2) + 1) = -1 :=
by
  sorry

end algebraic_expression_value_l1812_181283


namespace midpoint_product_zero_l1812_181206

theorem midpoint_product_zero (x y : ℝ)
  (h_midpoint_x : (2 + x) / 2 = 4)
  (h_midpoint_y : (6 + y) / 2 = 3) :
  x * y = 0 :=
by
  sorry

end midpoint_product_zero_l1812_181206


namespace find_police_stations_in_pittsburgh_l1812_181210

-- Conditions
def stores_in_pittsburgh : ℕ := 2000
def hospitals_in_pittsburgh : ℕ := 500
def schools_in_pittsburgh : ℕ := 200
def total_buildings_in_new_city : ℕ := 2175

-- Define the problem statement and the target proof
theorem find_police_stations_in_pittsburgh (P : ℕ) :
  1000 + 1000 + 150 + (P + 5) = total_buildings_in_new_city → P = 20 :=
by
  sorry

end find_police_stations_in_pittsburgh_l1812_181210


namespace probability_adjacent_vertices_of_octagon_l1812_181205

theorem probability_adjacent_vertices_of_octagon :
  let num_vertices := 8;
  let adjacent_vertices (v1 v2 : Fin num_vertices) : Prop := 
    (v2 = (v1 + 1) % num_vertices) ∨ (v2 = (v1 - 1 + num_vertices) % num_vertices);
  let total_vertices := num_vertices - 1;
  (2 : ℚ) / total_vertices = (2 / 7 : ℚ) :=
by
  -- Proof goes here
  sorry

end probability_adjacent_vertices_of_octagon_l1812_181205


namespace find_k_l1812_181259

theorem find_k (x y k : ℝ) (h1 : x + 2 * y = k + 1) (h2 : 2 * x + y = 1) (h3 : x + y = 3) : k = 7 :=
by
  sorry

end find_k_l1812_181259


namespace initial_winning_percentage_calc_l1812_181203

variable (W : ℝ)
variable (initial_matches : ℝ := 120)
variable (additional_wins : ℝ := 70)
variable (final_matches : ℝ := 190)
variable (final_average : ℝ := 0.52)
variable (initial_wins : ℝ := 29)

noncomputable def winning_percentage_initial :=
  (initial_wins / initial_matches) * 100

theorem initial_winning_percentage_calc :
  (W = initial_wins) →
  ((W + additional_wins) / final_matches = final_average) →
  winning_percentage_initial = 24.17 :=
by
  intros
  sorry

end initial_winning_percentage_calc_l1812_181203


namespace part_I_part_II_l1812_181231

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1))

theorem part_I (a : ℝ) (h_a_pos : a > 0) : (∀ x > 0, (1 / ((2 * x + 1) * (a * (2 * x + 1) - (2 * (a * x + 1) / 2))) ≥ 0) ↔ a ≥ 2) :=
sorry

theorem part_II : ∃ a : ℝ, (∀ x > 0, (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1)) ≥ 1) ∧ (Real.log (a * (Real.sqrt ((2 - a) / (4 * a))) + 1 / 2) + (2 / (2 * (Real.sqrt ((2 - a) / (4 * a))) + 1)) = 1) ∧ a = 1 :=
sorry

end part_I_part_II_l1812_181231


namespace relative_errors_are_equal_l1812_181211

theorem relative_errors_are_equal :
  let e1 := 0.04
  let l1 := 20.0
  let e2 := 0.3
  let l2 := 150.0
  (e1 / l1) = (e2 / l2) :=
by
  sorry

end relative_errors_are_equal_l1812_181211


namespace gamma_start_time_correct_l1812_181276

noncomputable def trisection_points (AB : ℕ) : Prop := AB ≥ 3

structure Walkers :=
  (d : ℕ) -- Total distance AB
  (Vα : ℕ) -- Speed of person α
  (Vβ : ℕ) -- Speed of person β
  (Vγ : ℕ) -- Speed of person γ

def meeting_times (w : Walkers) := 
  w.Vα = w.d / 72 ∧ 
  w.Vβ = w.d / 36 ∧ 
  w.Vγ = w.Vβ

def start_times_correct (startA timeA_meetC : ℕ) (startB timeB_reachesA: ℕ) (startC_latest: ℕ): Prop :=
  startA = 0 ∧ 
  startB = 12 ∧
  timeA_meetC = 24 ∧ 
  timeB_reachesA = 30 ∧
  startC_latest = 16

theorem gamma_start_time_correct (AB : ℕ) (w : Walkers) (t : Walkers → Prop) : 
  trisection_points AB → 
  meeting_times w →
  start_times_correct 0 24 12 30 16 → 
  ∃ tγ_start, tγ_start = 16 :=
sorry

end gamma_start_time_correct_l1812_181276


namespace a6_minus_b6_divisible_by_9_l1812_181200

theorem a6_minus_b6_divisible_by_9 {a b : ℤ} (h₁ : a % 3 ≠ 0) (h₂ : b % 3 ≠ 0) : (a ^ 6 - b ^ 6) % 9 = 0 := 
sorry

end a6_minus_b6_divisible_by_9_l1812_181200


namespace whole_milk_fat_percentage_l1812_181216

def fat_in_some_milk : ℝ := 4
def percentage_less : ℝ := 0.5

theorem whole_milk_fat_percentage : ∃ (x : ℝ), fat_in_some_milk = percentage_less * x ∧ x = 8 :=
sorry

end whole_milk_fat_percentage_l1812_181216


namespace apples_remain_correct_l1812_181278

def total_apples : ℕ := 15
def apples_eaten : ℕ := 7
def apples_remaining : ℕ := total_apples - apples_eaten

theorem apples_remain_correct : apples_remaining = 8 :=
by
  -- Initial number of apples
  let total := total_apples
  -- Number of apples eaten
  let eaten := apples_eaten
  -- Remaining apples
  let remain := total - eaten
  -- Assertion
  have h : remain = 8 := by
      sorry
  exact h

end apples_remain_correct_l1812_181278


namespace hyperbola_eccentricity_l1812_181272

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b = -4 * a / 3)
  (hc : c = (Real.sqrt (a ^ 2 + b ^ 2)))
  (point_on_asymptote : ∃ x y : ℝ, x = 3 ∧ y = -4 ∧ (y = b / a * x ∨ y = -b / a * x)) :
  (c / a) = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l1812_181272


namespace train_A_distance_travelled_l1812_181226

/-- Let Train A and Train B start from opposite ends of a 200-mile route at the same time.
Train A has a constant speed of 20 miles per hour, and Train B has a constant speed of 200 miles / 6 hours (which is approximately 33.33 miles per hour).
Prove that Train A had traveled 75 miles when it met Train B. --/
theorem train_A_distance_travelled:
  ∀ (T : Type) (start_time : T) (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (meeting_time : ℝ),
  distance = 200 ∧ speed_A = 20 ∧ speed_B = 33.33 ∧ meeting_time = 200 / (speed_A + speed_B) → 
  (speed_A * meeting_time = 75) :=
by
  sorry

end train_A_distance_travelled_l1812_181226


namespace bars_not_sold_l1812_181257

-- Definitions for the conditions
def cost_per_bar : ℕ := 3
def total_bars : ℕ := 9
def money_made : ℕ := 18

-- The theorem we need to prove
theorem bars_not_sold : total_bars - (money_made / cost_per_bar) = 3 := sorry

end bars_not_sold_l1812_181257


namespace geometric_progression_solution_l1812_181217

theorem geometric_progression_solution 
  (b₁ q : ℝ)
  (h₁ : b₁^3 * q^3 = 1728)
  (h₂ : b₁ * (1 + q + q^2) = 63) :
  (b₁ = 3 ∧ q = 4) ∨ (b₁ = 48 ∧ q = 1/4) :=
  sorry

end geometric_progression_solution_l1812_181217


namespace books_in_school_libraries_correct_l1812_181229

noncomputable def booksInSchoolLibraries : ℕ :=
  let booksInPublicLibrary := 1986
  let totalBooks := 7092
  totalBooks - booksInPublicLibrary

-- Now we create a theorem to check the correctness of our definition
theorem books_in_school_libraries_correct :
  booksInSchoolLibraries = 5106 := by
  sorry -- We skip the proof, as instructed

end books_in_school_libraries_correct_l1812_181229


namespace severe_flood_probability_next_10_years_l1812_181222

variable (A B C : Prop)
variable (P : Prop → ℝ)
variable (P_A : P A = 0.8)
variable (P_B : P B = 0.85)
variable (thirty_years_no_flood : ¬A)

theorem severe_flood_probability_next_10_years :
  P C = (P B - P A) / (1 - P A) := by
  sorry

end severe_flood_probability_next_10_years_l1812_181222


namespace quadratic_function_incorrect_statement_l1812_181249

theorem quadratic_function_incorrect_statement (x : ℝ) : 
  ∀ y : ℝ, y = -(x + 2)^2 - 1 → ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y = 0 ∧ -(x1 + 2)^2 - 1 = 0 ∧ -(x2 + 2)^2 - 1 = 0) :=
by 
sorry

end quadratic_function_incorrect_statement_l1812_181249


namespace number_of_men_in_first_group_l1812_181236

/-- The number of men in the first group that can complete a piece of work in 5 days alongside 16 boys,
    given that 13 men and 24 boys can complete the same work in 4 days, and the ratio of daily work done 
    by a man to a boy is 2:1, is 12. -/
theorem number_of_men_in_first_group
  (x : ℕ)  -- define x as the amount of work a boy can do in a day
  (m : ℕ)  -- define m as the number of men in the first group
  (h1 : ∀ (x : ℕ), 5 * (m * 2 * x + 16 * x) = 4 * (13 * 2 * x + 24 * x))
  (h2 : 2 * x = x + x) : m = 12 :=
sorry

end number_of_men_in_first_group_l1812_181236


namespace loan_principal_and_repayment_amount_l1812_181244

theorem loan_principal_and_repayment_amount (P R : ℝ) (r : ℝ) (years : ℕ) (total_interest : ℝ)
    (h1: r = 0.12)
    (h2: years = 3)
    (h3: total_interest = 5400)
    (h4: total_interest / years = R)
    (h5: R = P * r) :
    P = 15000 ∧ R = 1800 :=
sorry

end loan_principal_and_repayment_amount_l1812_181244


namespace total_boxes_l1812_181290
namespace AppleBoxes

theorem total_boxes (initial_boxes : ℕ) (apples_per_box : ℕ) (rotten_apples : ℕ)
  (apples_per_bag : ℕ) (bags_per_box : ℕ) (good_apples : ℕ) (final_boxes : ℕ) :
  initial_boxes = 14 →
  apples_per_box = 105 →
  rotten_apples = 84 →
  apples_per_bag = 6 →
  bags_per_box = 7 →
  final_boxes = (initial_boxes * apples_per_box - rotten_apples) / (apples_per_bag * bags_per_box) →
  final_boxes = 33 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  exact h6

end AppleBoxes

end total_boxes_l1812_181290


namespace prime_roots_quadratic_l1812_181246

theorem prime_roots_quadratic (p q : ℕ) (x1 x2 : ℕ) 
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (h_prime_x1 : Nat.Prime x1)
  (h_prime_x2 : Nat.Prime x2)
  (h_eq : p * x1 * x1 + p * x2 * x2 - q * x1 * x2 + 1985 = 0) :
  12 * p * p + q = 414 :=
sorry

end prime_roots_quadratic_l1812_181246


namespace four_digit_unique_count_l1812_181268

theorem four_digit_unique_count : 
  (∃ k : ℕ, k = 14 ∧ ∃ lst : List ℕ, lst.length = 4 ∧ 
    (∀ d ∈ lst, d = 2 ∨ d = 3) ∧ (2 ∈ lst) ∧ (3 ∈ lst)) :=
by
  sorry

end four_digit_unique_count_l1812_181268


namespace least_positive_integer_x_l1812_181227

theorem least_positive_integer_x (x : ℕ) (h : x + 5683 ≡ 420 [MOD 17]) : x = 7 :=
sorry

end least_positive_integer_x_l1812_181227


namespace division_correct_l1812_181251

-- Definitions based on conditions
def expr1 : ℕ := 12 + 15 * 3
def expr2 : ℚ := 180 / expr1

-- Theorem statement using the question and correct answer
theorem division_correct : expr2 = 180 / 57 := by
  sorry

end division_correct_l1812_181251


namespace find_f_neg2_l1812_181254

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Define the conditions and statement to be proved
theorem find_f_neg2 (a b : ℝ) (h1 : f a b 2 = 9) : f a b (-2) = 13 :=
by
  -- Conditions lead to the conclusion to be proved
  sorry

end find_f_neg2_l1812_181254


namespace valid_fractions_l1812_181285

theorem valid_fractions :
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (1 ≤ z ∧ z ≤ 9) ∧
  (10 * x + y) % (10 * y + z) = 0 ∧ (10 * x + y) / (10 * y + z) = x / z :=
sorry

end valid_fractions_l1812_181285


namespace proof_expression_equals_60_times_10_power_1501_l1812_181207

noncomputable def expression_equals_60_times_10_power_1501 : Prop :=
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501

theorem proof_expression_equals_60_times_10_power_1501 :
  expression_equals_60_times_10_power_1501 :=
by 
  sorry

end proof_expression_equals_60_times_10_power_1501_l1812_181207


namespace sum_of_arith_seq_l1812_181215

noncomputable def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arith_seq (a : ℕ → ℝ) (h_a : is_arith_seq a)
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 21 :=
sorry

end sum_of_arith_seq_l1812_181215


namespace correct_operation_l1812_181274
variable (a x y: ℝ)

theorem correct_operation : 
  ¬ (5 * a - 2 * a = 3) ∧
  ¬ ((x + 2 * y)^2 = x^2 + 4 * y^2) ∧
  ¬ (x^8 / x^4 = x^2) ∧
  ((2 * a)^3 = 8 * a^3) :=
by
  sorry

end correct_operation_l1812_181274


namespace radian_measure_of_200_degrees_l1812_181288

theorem radian_measure_of_200_degrees :
  (200 : ℝ) * (Real.pi / 180) = (10 / 9) * Real.pi :=
sorry

end radian_measure_of_200_degrees_l1812_181288


namespace can_weigh_1kg_with_300g_and_650g_weights_l1812_181295

-- Definitions based on conditions
def balance_scale (a b : ℕ) (w₁ w₂ : ℕ) : Prop :=
  a * w₁ + b * w₂ = 1000

-- Statement to prove based on the problem and solution
theorem can_weigh_1kg_with_300g_and_650g_weights (w₁ : ℕ) (w₂ : ℕ) (a b : ℕ)
  (h_w1 : w₁ = 300) (h_w2 : w₂ = 650) (h_a : a = 1) (h_b : b = 1) :
  balance_scale a b w₁ w₂ :=
by 
  -- We are given:
  -- - w1 = 300 g
  -- - w2 = 650 g
  -- - we want to measure 1000 g using these weights
  -- - a = 1
  -- - b = 1
  -- Prove that:
  --   a * w1 + b * w2 = 1000
  -- Which is:
  --   1 * 300 + 1 * 650 = 1000
  sorry

end can_weigh_1kg_with_300g_and_650g_weights_l1812_181295


namespace weeks_per_mouse_correct_l1812_181280

def years_in_decade : ℕ := 10
def weeks_per_year : ℕ := 52
def total_mice : ℕ := 130

def total_weeks_in_decade : ℕ := years_in_decade * weeks_per_year
def weeks_per_mouse : ℕ := total_weeks_in_decade / total_mice

theorem weeks_per_mouse_correct : weeks_per_mouse = 4 := 
sorry

end weeks_per_mouse_correct_l1812_181280


namespace polynomial_coefficients_sum_and_difference_l1812_181270

theorem polynomial_coefficients_sum_and_difference :
  ∀ (a_0 a_1 a_2 a_3 a_4 : ℤ),
  (∀ (x : ℤ), (2 * x - 3)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_1 + a_2 + a_3 + a_4 = -80) ∧ ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625) :=
by
  intros a_0 a_1 a_2 a_3 a_4 h
  sorry

end polynomial_coefficients_sum_and_difference_l1812_181270


namespace find_slope_l1812_181232

theorem find_slope (k b x y y2 : ℝ) (h1 : y = k * x + b) (h2 : y2 = k * (x + 3) + b) (h3 : y2 - y = -2) : k = -2 / 3 := by
  sorry

end find_slope_l1812_181232


namespace converse_proposition_l1812_181287

-- Define the condition: The equation x^2 + x - m = 0 has real roots
def has_real_roots (a b c : ℝ) : Prop :=
  let Δ := b * b - 4 * a * c
  Δ ≥ 0

theorem converse_proposition (m : ℝ) :
  has_real_roots 1 1 (-m) → m > 0 :=
by
  sorry

end converse_proposition_l1812_181287


namespace probability_page_multiple_of_7_l1812_181255

theorem probability_page_multiple_of_7 (total_pages : ℕ) (probability : ℚ)
  (h_total_pages : total_pages = 500) 
  (h_probability : probability = 71 / 500) :
  probability = 0.142 := 
sorry

end probability_page_multiple_of_7_l1812_181255


namespace rental_days_l1812_181291

-- Definitions based on conditions
def daily_rate := 30
def weekly_rate := 190
def total_payment := 310

-- Prove that Jennie rented the car for 11 days
theorem rental_days : ∃ d : ℕ, d = 11 ∧ (total_payment = weekly_rate + (d - 7) * daily_rate) ∨ (d < 7 ∧ total_payment = d * daily_rate) :=
by
  sorry

end rental_days_l1812_181291


namespace smallest_circle_tangent_to_line_and_circle_l1812_181239

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the original circle equation as a condition
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 2 * y = 0

-- Define the smallest circle equation as a condition
def smallest_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- The main lemma to prove that the smallest circle's equation matches the expected result
theorem smallest_circle_tangent_to_line_and_circle :
  (∀ x y, line_eq x y → smallest_circle_eq x y) ∧ (∀ x y, circle_eq x y → smallest_circle_eq x y) :=
by
  sorry -- Proof is omitted, as instructed

end smallest_circle_tangent_to_line_and_circle_l1812_181239


namespace consistent_price_per_kg_l1812_181266

theorem consistent_price_per_kg (m₁ m₂ : ℝ) (p₁ p₂ : ℝ)
  (h₁ : p₁ = 6) (h₂ : m₁ = 2)
  (h₃ : p₂ = 36) (h₄ : m₂ = 12) :
  (p₁ / m₁ = p₂ / m₂) := 
by 
  sorry

end consistent_price_per_kg_l1812_181266


namespace candy_bars_per_bag_l1812_181245

theorem candy_bars_per_bag (total_candy_bars : ℕ) (number_of_bags : ℕ) (h1 : total_candy_bars = 15) (h2 : number_of_bags = 5) : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l1812_181245


namespace lean_proof_problem_l1812_181299

section

variable {R : Type*} [AddCommGroup R]

def is_odd_function (f : ℝ → R) : Prop :=
  ∀ x, f (-x) = -f x

theorem lean_proof_problem (f: ℝ → ℝ) (h_odd: is_odd_function f)
    (h_cond: f 3 + f (-2) = 2) : f 2 - f 3 = -2 :=
by
  sorry

end

end lean_proof_problem_l1812_181299


namespace lowest_temperature_at_noon_l1812_181218

theorem lowest_temperature_at_noon
  (L : ℤ) -- Denote lowest temperature as L
  (avg_temp : ℤ) -- Average temperature from Monday to Friday
  (max_range : ℤ) -- Maximum possible range of the temperature
  (h1 : avg_temp = 50) -- Condition 1: average temperature is 50
  (h2 : max_range = 50) -- Condition 2: maximum range is 50
  (total_temp : ℤ) -- Sum of temperatures from Monday to Friday
  (h3 : total_temp = 250) -- Sum of temperatures equals 5 * 50
  (h4 : total_temp = L + (L + 50) + (L + 50) + (L + 50) + (L + 50)) -- Sum represented in terms of L
  : L = 10 := -- Prove that L equals 10
sorry

end lowest_temperature_at_noon_l1812_181218


namespace minuend_is_12_point_5_l1812_181289

theorem minuend_is_12_point_5 (x y : ℝ) (h : x + y + (x - y) = 25) : x = 12.5 := by
  sorry

end minuend_is_12_point_5_l1812_181289


namespace coordinate_plane_points_l1812_181265

theorem coordinate_plane_points (x y : ℝ) :
    4 * x^2 * y^2 = 4 * x * y + 3 ↔ (x * y = 3 / 2 ∨ x * y = -1 / 2) :=
by 
  sorry

end coordinate_plane_points_l1812_181265


namespace mixture_price_l1812_181233

-- Define constants
noncomputable def V1 (X : ℝ) : ℝ := 3.50 * X
noncomputable def V2 : ℝ := 4.30 * 6.25
noncomputable def W2 : ℝ := 6.25
noncomputable def W1 (X : ℝ) : ℝ := X

-- Define the total mixture weight condition
theorem mixture_price (X : ℝ) (P : ℝ) (h1 : W1 X + W2 = 10) (h2 : 10 * P = V1 X + V2) :
  P = 4 := by
  sorry

end mixture_price_l1812_181233


namespace Durakavalyanie_last_lesson_class_1C_l1812_181225

theorem Durakavalyanie_last_lesson_class_1C :
  ∃ (class_lesson : String × Nat → String), 
  class_lesson ("1B", 1) = "Kurashenie" ∧
  (∃ (k m n : Nat), class_lesson ("1A", k) = "Durakavalyanie" ∧ class_lesson ("1B", m) = "Durakavalyanie" ∧ m > k) ∧
  class_lesson ("1A", 2) ≠ "Nizvedenie" ∧
  class_lesson ("1C", 3) = "Durakavalyanie" :=
sorry

end Durakavalyanie_last_lesson_class_1C_l1812_181225


namespace oranges_thrown_away_l1812_181214

theorem oranges_thrown_away (initial_oranges old_oranges_thrown new_oranges final_oranges : ℕ) 
    (h1 : initial_oranges = 34)
    (h2 : new_oranges = 13)
    (h3 : final_oranges = 27)
    (h4 : initial_oranges - old_oranges_thrown + new_oranges = final_oranges) :
    old_oranges_thrown = 20 :=
by
  sorry

end oranges_thrown_away_l1812_181214


namespace find_largest_integer_solution_l1812_181277

theorem find_largest_integer_solution:
  ∃ x: ℤ, (1/4 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < (7/9 : ℝ) ∧ (x = 4) := by
  sorry

end find_largest_integer_solution_l1812_181277


namespace angle_between_vectors_with_offset_l1812_181275

noncomputable def vector_angle_with_offset : ℝ :=
  let v1 := (4, -1)
  let v2 := (6, 8)
  let dot_product := 4 * 6 + (-1) * 8
  let magnitude_v1 := Real.sqrt (4 ^ 2 + (-1) ^ 2)
  let magnitude_v2 := Real.sqrt (6 ^ 2 + 8 ^ 2)
  let cos_theta := dot_product / (magnitude_v1 * magnitude_v2)
  Real.arccos cos_theta + 30

theorem angle_between_vectors_with_offset :
  vector_angle_with_offset = Real.arccos (8 / (5 * Real.sqrt 17)) + 30 := 
sorry

end angle_between_vectors_with_offset_l1812_181275


namespace green_beans_count_l1812_181235

def total_beans := 572
def red_beans := (1 / 4) * total_beans
def remaining_after_red := total_beans - red_beans
def white_beans := (1 / 3) * remaining_after_red
def remaining_after_white := remaining_after_red - white_beans
def green_beans := (1 / 2) * remaining_after_white

theorem green_beans_count : green_beans = 143 := by
  sorry

end green_beans_count_l1812_181235


namespace sum_of_ages_l1812_181263

/-- Given a woman's age is three years more than twice her son's age, 
and the son is 27 years old, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ)
  (h1 : son_age = 27)
  (h2 : woman_age = 3 + 2 * son_age) :
  son_age + woman_age = 84 := 
sorry

end sum_of_ages_l1812_181263


namespace monotonic_intervals_find_f_max_l1812_181240

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem monotonic_intervals :
  (∀ x, 0 < x → x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
  (∀ x, x > Real.exp 1 → (1 - Real.log x) / x^2 < 0) :=
sorry

theorem find_f_max (m : ℝ) (h : m > 0) :
  if 0 < 2 * m ∧ 2 * m ≤ Real.exp 1 then f (2 * m) = Real.log (2 * m) / (2 * m)
  else if m ≥ Real.exp 1 then f m = Real.log m / m
  else f (Real.exp 1) = 1 / Real.exp 1 :=
sorry

end monotonic_intervals_find_f_max_l1812_181240


namespace distinct_solutions_difference_l1812_181213

theorem distinct_solutions_difference (r s : ℝ) (hr : (r - 5) * (r + 5) = 25 * r - 125)
  (hs : (s - 5) * (s + 5) = 25 * s - 125) (neq : r ≠ s) (hgt : r > s) : r - s = 15 := by
  sorry

end distinct_solutions_difference_l1812_181213


namespace min_value_expression_l1812_181273

theorem min_value_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 :=
by
  sorry

end min_value_expression_l1812_181273


namespace tan_B_eq_one_third_l1812_181279

theorem tan_B_eq_one_third
  (A B : ℝ)
  (h1 : Real.cos A = 4 / 5)
  (h2 : Real.tan (A - B) = 1 / 3) :
  Real.tan B = 1 / 3 := by
  sorry

end tan_B_eq_one_third_l1812_181279


namespace number_of_ordered_tuples_l1812_181294

noncomputable def count_tuples 
  (a1 a2 a3 a4 : ℕ) 
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2): ℕ :=
40

theorem number_of_ordered_tuples 
  (a1 a2 a3 a4 : ℕ)
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2) : 
  count_tuples a1 a2 a3 a4 H_distinct H_range H_eqn = 40 :=
sorry

end number_of_ordered_tuples_l1812_181294


namespace system_has_real_solution_l1812_181296

theorem system_has_real_solution (k : ℝ) : 
  (∃ x y : ℝ, y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by
  sorry

end system_has_real_solution_l1812_181296


namespace find_vanilla_cookies_l1812_181242

variable (V : ℕ)

def num_vanilla_cookies_sold (choc_cookies: ℕ) (vanilla_cookies: ℕ) (total_revenue: ℕ) : Prop :=
  choc_cookies * 1 + vanilla_cookies * 2 = total_revenue

theorem find_vanilla_cookies (h : num_vanilla_cookies_sold 220 V 360) : V = 70 :=
by
  sorry

end find_vanilla_cookies_l1812_181242


namespace mass_percentage_O_in_CaO_l1812_181292

theorem mass_percentage_O_in_CaO :
  let molar_mass_Ca := 40.08
  let molar_mass_O := 16.00
  let molar_mass_CaO := molar_mass_Ca + molar_mass_O
  let mass_percentage_O := (molar_mass_O / molar_mass_CaO) * 100
  mass_percentage_O = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l1812_181292


namespace polygon_area_l1812_181261

-- Definitions and conditions
def side_length (n : ℕ) (p : ℕ) := p / n
def rectangle_area (s : ℕ) := 2 * s * s
def total_area (r : ℕ) (area : ℕ) := r * area

-- Theorem statement with conditions and conclusion
theorem polygon_area (n r p : ℕ) (h1 : n = 24) (h2 : r = 4) (h3 : p = 48) :
  total_area r (rectangle_area (side_length n p)) = 32 := by
  sorry

end polygon_area_l1812_181261


namespace payment_to_C_l1812_181267

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℚ := 3360

def work_done (rate : ℚ) (days : ℕ) : ℚ := rate * days

-- Conditions
def person_A_work_rate := work_rate 6
def person_B_work_rate := work_rate 8
def combined_work_rate := person_A_work_rate + person_B_work_rate
def work_by_A_and_B_in_3_days := work_done combined_work_rate 3
def total_work : ℚ := 1
def work_done_by_C := total_work - work_by_A_and_B_in_3_days

-- Proof problem statement
theorem payment_to_C :
  (work_done_by_C / total_work) * total_payment = 420 := 
sorry

end payment_to_C_l1812_181267


namespace rectangle_vertex_x_coordinate_l1812_181224

theorem rectangle_vertex_x_coordinate
  (x : ℝ)
  (y1 y2 : ℝ)
  (slope : ℝ)
  (h1 : x = 1)
  (h2 : 9 = 9)
  (h3 : slope = 0.2)
  (h4 : y1 = 0)
  (h5 : y2 = 2)
  (h6 : ∀ (x : ℝ), (0.2 * x : ℝ) = 1 → x = 1) :
  x = 1 := 
by sorry

end rectangle_vertex_x_coordinate_l1812_181224


namespace math_problem_l1812_181247

theorem math_problem (x : ℝ) : 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1/2 ∧ (x^2 + x^3 - 2 * x^4) / (x + x^2 - 2 * x^3) ≥ -1 ↔ 
  x ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ioc (-1/2 : ℝ) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 := 
by 
  sorry

end math_problem_l1812_181247


namespace inequality_proof_l1812_181260

variable (x1 x2 y1 y2 z1 z2 : ℝ)
variable (h0 : 0 < x1)
variable (h1 : 0 < x2)
variable (h2 : x1 * y1 > z1^2)
variable (h3 : x2 * y2 > z2^2)

theorem inequality_proof :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l1812_181260
