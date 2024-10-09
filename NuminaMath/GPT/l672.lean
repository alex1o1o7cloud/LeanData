import Mathlib

namespace find_a_b_and_tangent_line_l672_67280

noncomputable def f (a b x : ℝ) := x^3 + 2 * a * x^2 + b * x + a
noncomputable def g (x : ℝ) := x^2 - 3 * x + 2
noncomputable def f' (a b x : ℝ) := 3 * x^2 + 4 * a * x + b
noncomputable def g' (x : ℝ) := 2 * x - 3

theorem find_a_b_and_tangent_line (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f' a b 2 = 1 ∧ g' 2 = 1 → (a = -2 ∧ b = 5 ∧ ∀ x y : ℝ, y = x - 2 ↔ x - y - 2 = 0) :=
by
  intro h
  sorry

end find_a_b_and_tangent_line_l672_67280


namespace same_cost_number_of_guests_l672_67200

theorem same_cost_number_of_guests (x : ℕ) : 
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by {
  sorry
}

end same_cost_number_of_guests_l672_67200


namespace donny_total_spending_l672_67230

noncomputable def total_saving_mon : ℕ := 15
noncomputable def total_saving_tue : ℕ := 28
noncomputable def total_saving_wed : ℕ := 13
noncomputable def total_saving_fri : ℕ := 22

noncomputable def total_savings_mon_to_wed : ℕ := total_saving_mon + total_saving_tue + total_saving_wed
noncomputable def thursday_spending : ℕ := total_savings_mon_to_wed / 2
noncomputable def remaining_savings_after_thursday : ℕ := total_savings_mon_to_wed - thursday_spending
noncomputable def total_savings_before_sat : ℕ := remaining_savings_after_thursday + total_saving_fri
noncomputable def saturday_spending : ℕ := total_savings_before_sat * 40 / 100

theorem donny_total_spending : thursday_spending + saturday_spending = 48 := by sorry

end donny_total_spending_l672_67230


namespace problem1_problem2_l672_67208

-- Problem 1 Definition: Operation ※
def operation (m n : ℚ) : ℚ := 3 * m - n

-- Lean 4 statement: Prove 2※10 = -4
theorem problem1 : operation 2 10 = -4 := by
  sorry

-- Lean 4 statement: Prove that ※ does not satisfy the distributive law
theorem problem2 (a b c : ℚ) : 
  operation a (b + c) ≠ operation a b + operation a c := by
  sorry

end problem1_problem2_l672_67208


namespace solve_equation_l672_67283

theorem solve_equation : ∀ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) → x = 2 :=
by
  intro x h
  sorry

end solve_equation_l672_67283


namespace max_value_of_sum_of_cubes_l672_67216

theorem max_value_of_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end max_value_of_sum_of_cubes_l672_67216


namespace pentagon_area_l672_67223

-- Definitions of the vertices of the pentagon
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 2), (3, 3), (4, 1), (2, 0)]

-- Definition of the number of interior points
def interior_points : ℕ := 7

-- Definition of the number of boundary points
def boundary_points : ℕ := 5

-- Pick's theorem: Area = Interior points + Boundary points / 2 - 1
noncomputable def area : ℝ :=
  interior_points + boundary_points / 2 - 1

-- Theorem to be proved
theorem pentagon_area :
  area = 8.5 :=
by
  sorry

end pentagon_area_l672_67223


namespace work_completion_time_l672_67226

-- Define work rates for workers p, q, and r
def work_rate_p : ℚ := 1 / 12
def work_rate_q : ℚ := 1 / 9
def work_rate_r : ℚ := 1 / 18

-- Define time they work in respective phases
def time_p : ℚ := 2
def time_pq : ℚ := 3

-- Define the total time taken to complete the work
def total_time : ℚ := 6

-- Prove that the total time to complete the work is 6 days
theorem work_completion_time :
  (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq + (1 - (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq)) / (work_rate_p + work_rate_q + work_rate_r)) = total_time :=
by sorry

end work_completion_time_l672_67226


namespace train_trip_length_l672_67201

theorem train_trip_length (v D : ℝ) :
  (3 + (3 * D - 6 * v) / (2 * v) = 4 + D / v) ∧ 
  (2.5 + 120 / v + (6 * D - 12 * v - 720) / (5 * v) = 3.5 + D / v) →
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by
  sorry

end train_trip_length_l672_67201


namespace smallest_n_is_60_l672_67251

def smallest_n (n : ℕ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ (24 ∣ n^2) ∧ (450 ∣ n^3) ∧ ∀ m : ℕ, 24 ∣ m^2 → 450 ∣ m^3 → m ≥ n

theorem smallest_n_is_60 : smallest_n 60 :=
  sorry

end smallest_n_is_60_l672_67251


namespace find_remainder_2500th_term_l672_67207

theorem find_remainder_2500th_term : 
    let seq_position (n : ℕ) := n * (n + 1) / 2 
    let n := ((1 + Int.ofNat 20000).natAbs.sqrt + 1) / 2
    let term_2500 := if seq_position n < 2500 then n + 1 else n
    (term_2500 % 7) = 1 := by 
    sorry

end find_remainder_2500th_term_l672_67207


namespace split_trout_equally_l672_67205

-- Definitions for conditions
def Total_trout : ℕ := 18
def People : ℕ := 2

-- Statement we need to prove
theorem split_trout_equally 
(H1 : Total_trout = 18)
(H2 : People = 2) : 
  (Total_trout / People = 9) :=
by
  sorry

end split_trout_equally_l672_67205


namespace sufficient_but_not_necessary_condition_l672_67272

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^(m - 1) > 0 → m = 2) →
  (|m - 2| < 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l672_67272


namespace brothers_travel_distance_l672_67274

theorem brothers_travel_distance
  (x : ℝ)
  (hb_x : (120 : ℝ) / (x : ℝ) - 4 = (120 : ℝ) / (x + 40))
  (total_time : 2 = 2) :
  x = 20 ∧ (x + 40) = 60 :=
by
  -- we need to prove the distances
  sorry

end brothers_travel_distance_l672_67274


namespace f_monotonically_increasing_on_1_to_infinity_l672_67297

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_monotonically_increasing_on_1_to_infinity :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := 
sorry

end f_monotonically_increasing_on_1_to_infinity_l672_67297


namespace infinite_primes_of_form_2px_plus_1_l672_67235

theorem infinite_primes_of_form_2px_plus_1 (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) : 
  ∃ᶠ (n : ℕ) in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_2px_plus_1_l672_67235


namespace right_triangle_sides_l672_67262

theorem right_triangle_sides {a b c : ℕ} (h1 : a * (b + 2) = 150) (h2 : a^2 + b^2 = c^2) (h3 : a + (1 / 2 : ℤ) * (a * b) = 75) :
  (a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
sorry

end right_triangle_sides_l672_67262


namespace real_part_of_i_squared_times_1_plus_i_l672_67284

noncomputable def imaginary_unit : ℂ := Complex.I

theorem real_part_of_i_squared_times_1_plus_i :
  (Complex.re (imaginary_unit^2 * (1 + imaginary_unit))) = -1 :=
by
  sorry

end real_part_of_i_squared_times_1_plus_i_l672_67284


namespace simplify_expression_l672_67204

variable (x : ℝ)

theorem simplify_expression : (x + 2)^2 - (x + 1) * (x + 3) = 1 := 
by 
  sorry

end simplify_expression_l672_67204


namespace total_cost_pants_and_belt_l672_67211

theorem total_cost_pants_and_belt (P B : ℝ) 
  (hP : P = 34.0) 
  (hCondition : P = B - 2.93) : 
  P + B = 70.93 :=
by
  -- Placeholder for proof
  sorry

end total_cost_pants_and_belt_l672_67211


namespace crayons_total_l672_67214

theorem crayons_total (blue red green : ℕ) 
  (h1 : red = 4 * blue) 
  (h2 : green = 2 * red) 
  (h3 : blue = 3) : 
  blue + red + green = 39 := 
by
  sorry

end crayons_total_l672_67214


namespace units_digit_8421_1287_l672_67249

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_8421_1287 :
  units_digit (8421 ^ 1287) = 1 := 
by
  sorry

end units_digit_8421_1287_l672_67249


namespace num_divisors_1215_l672_67278

theorem num_divisors_1215 : (Finset.filter (λ d => 1215 % d = 0) (Finset.range (1215 + 1))).card = 12 :=
by
  sorry

end num_divisors_1215_l672_67278


namespace remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l672_67282

theorem remainder_of_x_pow_105_div_x_sq_sub_4x_add_3 :
  ∀ (x : ℤ), (x^105) % (x^2 - 4*x + 3) = (3^105 * (x-1) - (x-2)) / 2 :=
by sorry

end remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l672_67282


namespace students_doing_at_least_one_hour_of_homework_l672_67234

theorem students_doing_at_least_one_hour_of_homework (total_angle : ℝ) (less_than_one_hour_angle : ℝ) 
  (h1 : total_angle = 360) (h2 : less_than_one_hour_angle = 90) :
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  at_least_one_hour_percentage = 75 :=
by
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  sorry

end students_doing_at_least_one_hour_of_homework_l672_67234


namespace problem_divisibility_l672_67222

theorem problem_divisibility 
  (m n : ℕ) 
  (a : Fin (mn + 1) → ℕ)
  (h_pos : ∀ i, 0 < a i)
  (h_order : ∀ i j, i < j → a i < a j) :
  (∃ (b : Fin (m + 1) → Fin (mn + 1)), ∀ i j, i ≠ j → ¬(a (b i) ∣ a (b j))) ∨
  (∃ (c : Fin (n + 1) → Fin (mn + 1)), ∀ i, i < n → a (c i) ∣ a (c i.succ)) :=
sorry

end problem_divisibility_l672_67222


namespace sum_of_reciprocals_of_factors_of_13_l672_67217

theorem sum_of_reciprocals_of_factors_of_13 : 
  (1 : ℚ) + (1 / 13) = 14 / 13 :=
by {
  sorry
}

end sum_of_reciprocals_of_factors_of_13_l672_67217


namespace find_value_of_m_l672_67260

theorem find_value_of_m : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 :=
by
  use -12
  sorry

end find_value_of_m_l672_67260


namespace train_length_l672_67265

noncomputable section

-- Define the variables involved in the problem.
def train_length_cross_signal (V : ℝ) : ℝ := V * 18
def train_speed_cross_platform (L : ℝ) (platform_length : ℝ) : ℝ := (L + platform_length) / 40

-- Define the main theorem to prove the length of the train.
theorem train_length (V L : ℝ) (platform_length : ℝ) (h1 : L = V * 18)
(h2 : L + platform_length = V * 40) (h3 : platform_length = 366.67) :
L = 300 := 
by
  sorry

end train_length_l672_67265


namespace expression_parity_l672_67263

variable (o n c : ℕ)

def is_odd (x : ℕ) : Prop := ∃ k, x = 2 * k + 1

theorem expression_parity (ho : is_odd o) (hc : is_odd c) : 
  (o^2 + n * o + c) % 2 = 0 :=
  sorry

end expression_parity_l672_67263


namespace number_of_levels_l672_67299

theorem number_of_levels (total_capacity : ℕ) (additional_cars : ℕ) (already_parked_cars : ℕ) (n : ℕ) :
  total_capacity = 425 →
  additional_cars = 62 →
  already_parked_cars = 23 →
  n = total_capacity / (already_parked_cars + additional_cars) →
  n = 5 :=
by
  intros
  sorry

end number_of_levels_l672_67299


namespace gcd_1729_78945_is_1_l672_67287

theorem gcd_1729_78945_is_1 :
  ∃ m n : ℤ, 1729 * m + 78945 * n = 1 := sorry

end gcd_1729_78945_is_1_l672_67287


namespace christine_savings_l672_67266

/-- Christine's commission rate as a percentage. -/
def commissionRate : ℝ := 0.12

/-- Total sales made by Christine this month in dollars. -/
def totalSales : ℝ := 24000

/-- Percentage of commission allocated to personal needs. -/
def personalNeedsRate : ℝ := 0.60

/-- The amount Christine saved this month. -/
def amountSaved : ℝ := 1152

/--
Given the commission rate, total sales, and personal needs rate,
prove the amount saved is correctly calculated.
-/
theorem christine_savings :
  (1 - personalNeedsRate) * (commissionRate * totalSales) = amountSaved :=
by
  sorry

end christine_savings_l672_67266


namespace ratio_shorter_to_longer_l672_67225

-- Constants for the problem
def total_length : ℝ := 49
def shorter_piece_length : ℝ := 14

-- Definition of longer piece length based on the given conditions
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- The theorem to be proved
theorem ratio_shorter_to_longer : 
  shorter_piece_length / longer_piece_length = 2 / 5 :=
by
  -- This is where the proof would go
  sorry

end ratio_shorter_to_longer_l672_67225


namespace sector_properties_l672_67232

variables (r : ℝ) (alpha l S : ℝ)

noncomputable def arc_length (r alpha : ℝ) : ℝ := alpha * r
noncomputable def sector_area (l r : ℝ) : ℝ := (1/2) * l * r

theorem sector_properties
  (h_r : r = 2)
  (h_alpha : alpha = π / 6) :
  arc_length r alpha = π / 3 ∧ sector_area (arc_length r alpha) r = π / 3 :=
by
  sorry

end sector_properties_l672_67232


namespace total_hours_proof_l672_67250

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end total_hours_proof_l672_67250


namespace harish_ganpat_paint_wall_together_l672_67273

theorem harish_ganpat_paint_wall_together :
  let r_h := 1 / 3 -- Harish's rate of work (walls per hour)
  let r_g := 1 / 6 -- Ganpat's rate of work (walls per hour)
  let combined_rate := r_h + r_g -- Combined rate of work when both work together
  let time_to_paint_one_wall := 1 / combined_rate -- Time to paint one wall together
  time_to_paint_one_wall = 2 :=
by
  sorry

end harish_ganpat_paint_wall_together_l672_67273


namespace problem_statement_l672_67247

noncomputable def f (m x : ℝ) := (m-1) * Real.log x + m * x^2 + 1

theorem problem_statement (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → f m x₁ - f m x₂ > 2 * (x₁ - x₂)) ↔ 
  m ≥ (1 + Real.sqrt 3) / 2 :=
sorry

end problem_statement_l672_67247


namespace find_length_PB_l672_67203

-- Define the conditions of the problem
variables (AC AP PB : ℝ) (x : ℝ)

-- Condition: The length of chord AC is x
def length_AC := AC = x

-- Condition: The length of segment AP is x + 1
def length_AP := AP = x + 1

-- Statement of the theorem to prove the length of segment PB
theorem find_length_PB (h_AC : length_AC AC x) (h_AP : length_AP AP x) :
  PB = 2 * x + 1 :=
sorry

end find_length_PB_l672_67203


namespace distance_covered_l672_67252

theorem distance_covered (t : ℝ) (s_kmph : ℝ) (distance : ℝ) (h1 : t = 180) (h2 : s_kmph = 18) : 
  distance = 900 :=
by 
  sorry

end distance_covered_l672_67252


namespace max_chickens_ducks_l672_67279

theorem max_chickens_ducks (x y : ℕ) 
  (h1 : ∀ (k : ℕ), k = 6 → x + y - 6 ≥ 2) 
  (h2 : ∀ (k : ℕ), k = 9 → y ≥ 1) : 
  x + y ≤ 12 :=
sorry

end max_chickens_ducks_l672_67279


namespace geometric_progression_exists_l672_67295

theorem geometric_progression_exists :
  ∃ (b_1 b_2 b_3 b_4 q : ℚ), 
    b_1 - b_2 = 35 ∧ 
    b_3 - b_4 = 560 ∧ 
    b_2 = b_1 * q ∧ 
    b_3 = b_1 * q^2 ∧ 
    b_4 = b_1 * q^3 ∧ 
    ((b_1 = 7 ∧ q = -4 ∧ b_2 = -28 ∧ b_3 = 112 ∧ b_4 = -448) ∨ 
    (b_1 = -35/3 ∧ q = 4 ∧ b_2 = -140/3 ∧ b_3 = -560/3 ∧ b_4 = -2240/3)) :=
by
  sorry

end geometric_progression_exists_l672_67295


namespace therese_older_than_aivo_l672_67219

-- Definitions based on given conditions
variables {Aivo Jolyn Leon Therese : ℝ}
variables (h1 : Jolyn = Therese + 2)
variables (h2 : Leon = Aivo + 2)
variables (h3 : Jolyn = Leon + 5)

-- Statement to prove
theorem therese_older_than_aivo :
  Therese = Aivo + 5 :=
by
  sorry

end therese_older_than_aivo_l672_67219


namespace probability_of_selection_of_X_l672_67240

theorem probability_of_selection_of_X 
  (P_Y : ℝ)
  (P_X_and_Y : ℝ) :
  P_Y = 2 / 7 →
  P_X_and_Y = 0.05714285714285714 →
  ∃ P_X : ℝ, P_X = 0.2 :=
by
  intro hY hXY
  sorry

end probability_of_selection_of_X_l672_67240


namespace profit_without_discount_l672_67275

theorem profit_without_discount (CP SP_original SP_discount : ℝ) (h1 : CP > 0) (h2 : SP_discount = CP * 1.14) (h3 : SP_discount = SP_original * 0.95) :
  (SP_original - CP) / CP * 100 = 20 :=
by
  have h4 : SP_original = SP_discount / 0.95 := by sorry
  have h5 : SP_original = CP * 1.2 := by sorry
  have h6 : (SP_original - CP) / CP * 100 = (CP * 1.2 - CP) / CP * 100 := by sorry
  have h7 : (SP_original - CP) / CP * 100 = 20 := by sorry
  exact h7

end profit_without_discount_l672_67275


namespace triangular_number_30_l672_67254

theorem triangular_number_30 : 
  (∃ (T : ℕ), T = 30 * (30 + 1) / 2 ∧ T = 465) :=
by 
  sorry

end triangular_number_30_l672_67254


namespace solve_ticket_problem_l672_67290

def ticket_problem : Prop :=
  ∃ S N : ℕ, S + N = 2000 ∧ 9 * S + 11 * N = 20960 ∧ S = 520

theorem solve_ticket_problem : ticket_problem :=
sorry

end solve_ticket_problem_l672_67290


namespace least_three_digit_divisible_3_4_7_is_168_l672_67245

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end least_three_digit_divisible_3_4_7_is_168_l672_67245


namespace find_larger_number_l672_67296

theorem find_larger_number (x y : ℕ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : x + y = 54) : 
  y = 30 :=
sorry

end find_larger_number_l672_67296


namespace Oliver_total_workout_hours_l672_67292

-- Define the working hours for each day
def Monday_hours : ℕ := 4
def Tuesday_hours : ℕ := Monday_hours - 2
def Wednesday_hours : ℕ := 2 * Monday_hours
def Thursday_hours : ℕ := 2 * Tuesday_hours

-- Prove that the total hours Oliver worked out adds up to 18
theorem Oliver_total_workout_hours : Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours = 18 := by
  sorry

end Oliver_total_workout_hours_l672_67292


namespace acres_of_flax_l672_67256

-- Let F be the number of acres of flax
variable (F : ℕ)

-- Condition: The total farm size is 240 acres
def total_farm_size (F : ℕ) := F + (F + 80) = 240

-- Proof statement
theorem acres_of_flax (h : total_farm_size F) : F = 80 :=
sorry

end acres_of_flax_l672_67256


namespace average_monthly_growth_rate_l672_67248

-- Define the conditions
variables (P : ℝ) (r : ℝ)
-- The condition that output in December is P times that of January
axiom growth_rate_condition : (1 + r)^11 = P

-- Define the goal to prove the average monthly growth rate
theorem average_monthly_growth_rate : r = (P^(1/11) - 1) :=
by
  sorry

end average_monthly_growth_rate_l672_67248


namespace eight_mul_eleven_and_one_fourth_l672_67241

theorem eight_mul_eleven_and_one_fourth : 8 * (11 + (1 / 4 : ℝ)) = 90 := by
  sorry

end eight_mul_eleven_and_one_fourth_l672_67241


namespace problem_solution_l672_67285

theorem problem_solution (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 14) (h2 : a = b + c) : ab - bc + ac = 7 :=
  sorry

end problem_solution_l672_67285


namespace tom_saves_money_l672_67202

-- Defining the cost of a normal doctor's visit
def normal_doctor_cost : ℕ := 200

-- Defining the discount percentage for the discount clinic
def discount_percentage : ℕ := 70

-- Defining the cost reduction based on the discount percentage
def discount_amount (cost percentage : ℕ) : ℕ := (percentage * cost) / 100

-- Defining the cost of a visit to the discount clinic
def discount_clinic_cost (normal_cost discount_amount : ℕ ) : ℕ := normal_cost - discount_amount

-- Defining the number of visits to the discount clinic
def discount_clinic_visits : ℕ := 2

-- Defining the total cost for the discount clinic visits
def total_discount_clinic_cost (visit_cost visits : ℕ) : ℕ := visits * visit_cost

-- The final cost savings calculation
def cost_savings (normal_cost total_discount_cost : ℕ) : ℕ := normal_cost - total_discount_cost

-- Proving the amount Tom saves by going to the discount clinic
theorem tom_saves_money : cost_savings normal_doctor_cost (total_discount_clinic_cost (discount_clinic_cost normal_doctor_cost (discount_amount normal_doctor_cost discount_percentage)) discount_clinic_visits) = 80 :=
by
  sorry

end tom_saves_money_l672_67202


namespace smallest_period_of_f_is_pi_div_2_l672_67267

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 4 + (Real.sin x) ^ 2

theorem smallest_period_of_f_is_pi_div_2 : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2 :=
sorry

end smallest_period_of_f_is_pi_div_2_l672_67267


namespace product_of_two_numbers_l672_67213

theorem product_of_two_numbers (a b : ℕ) (h_gcd : Nat.gcd a b = 8) (h_lcm : Nat.lcm a b = 72) : a * b = 576 := 
by
  sorry

end product_of_two_numbers_l672_67213


namespace greatest_divisor_same_remainder_l672_67239

theorem greatest_divisor_same_remainder (a b c : ℕ) (h₁ : a = 54) (h₂ : b = 87) (h₃ : c = 172) : 
  ∃ d, (d ∣ (b - a)) ∧ (d ∣ (c - b)) ∧ (d ∣ (c - a)) ∧ (∀ e, (e ∣ (b - a)) ∧ (e ∣ (c - b)) ∧ (e ∣ (c - a)) → e ≤ d) ∧ d = 1 := 
by 
  sorry

end greatest_divisor_same_remainder_l672_67239


namespace power_function_at_4_l672_67212

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_at_4 {α : ℝ} :
  power_function α 2 = (Real.sqrt 2) / 2 →
  α = -1/2 →
  power_function α 4 = 1 / 2 :=
by
  intros h1 h2
  rw [h2, power_function]
  sorry

end power_function_at_4_l672_67212


namespace probability_correct_l672_67221

noncomputable def probability_point_between_lines : ℝ :=
  let intersection_x_l := 4    -- x-intercept of line l
  let intersection_x_m := 3    -- x-intercept of line m
  let area_under_l := (1 / 2) * intersection_x_l * 8 -- area under line l
  let area_under_m := (1 / 2) * intersection_x_m * 9 -- area under line m
  let area_between := area_under_l - area_under_m    -- area between lines
  (area_between / area_under_l : ℝ)

theorem probability_correct : probability_point_between_lines = 0.16 :=
by
  simp only [probability_point_between_lines]
  sorry

end probability_correct_l672_67221


namespace hours_per_day_l672_67264

theorem hours_per_day
  (num_warehouse : ℕ := 4)
  (num_managers : ℕ := 2)
  (rate_warehouse : ℝ := 15)
  (rate_manager : ℝ := 20)
  (tax_rate : ℝ := 0.10)
  (days_worked : ℕ := 25)
  (total_cost : ℝ := 22000) :
  ∃ h : ℝ, 6 * h * days_worked * (rate_warehouse + rate_manager) * (1 + tax_rate) = total_cost ∧ h = 8 :=
by
  sorry

end hours_per_day_l672_67264


namespace smallest_of_seven_consecutive_even_numbers_l672_67289

theorem smallest_of_seven_consecutive_even_numbers (n : ℤ) :
  (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 448 → 
  (n - 6) = 58 :=
by
  sorry

end smallest_of_seven_consecutive_even_numbers_l672_67289


namespace nate_current_age_l672_67259

open Real

variables (E N : ℝ)

/-- Ember is half as old as Nate, so E = 1/2 * N. -/
def ember_half_nate (h1 : E = 1/2 * N) : Prop := True

/-- The age difference of 7 years remains constant, so 21 - 14 = N - E. -/
def age_diff_constant (h2 : 7 = N - E) : Prop := True

/-- Prove that Nate is currently 14 years old given the conditions. -/
theorem nate_current_age (h1 : E = 1/2 * N) (h2 : 7 = N - E) : N = 14 :=
by sorry

end nate_current_age_l672_67259


namespace paula_aunt_money_l672_67237

theorem paula_aunt_money
  (shirts_cost : ℕ := 2 * 11)
  (pants_cost : ℕ := 13)
  (money_left : ℕ := 74) : 
  shirts_cost + pants_cost + money_left = 109 :=
by
  sorry

end paula_aunt_money_l672_67237


namespace ratio_of_frank_to_joystick_l672_67220

-- Define the costs involved
def cost_table : ℕ := 140
def cost_chair : ℕ := 100
def cost_joystick : ℕ := 20
def diff_spent : ℕ := 30

-- Define the payments
def F_j := 5
def E_j := 15

-- The ratio we need to prove
def ratio_frank_to_total_joystick (F_j : ℕ) (total_joystick : ℕ) : (ℕ × ℕ) :=
  (F_j / Nat.gcd F_j total_joystick, total_joystick / Nat.gcd F_j total_joystick)

theorem ratio_of_frank_to_joystick :
  let F_j := 5
  let total_joystick := 20
  ratio_frank_to_total_joystick F_j total_joystick = (1, 4) := by
  sorry

end ratio_of_frank_to_joystick_l672_67220


namespace mark_collects_money_l672_67258

variable (households_per_day : Nat)
variable (days : Nat)
variable (pair_amount : Nat)
variable (half_factor : Nat)

theorem mark_collects_money
  (h1 : households_per_day = 20)
  (h2 : days = 5)
  (h3 : pair_amount = 40)
  (h4 : half_factor = 2) :
  (households_per_day * days / half_factor) * pair_amount = 2000 :=
by
  sorry

end mark_collects_money_l672_67258


namespace angle_b_is_acute_l672_67210

-- Definitions for angles being right, acute, and sum of angles in a triangle
def is_right_angle (θ : ℝ) : Prop := θ = 90
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_sum_to_180 (α β γ : ℝ) : Prop := α + β + γ = 180

-- Main theorem statement
theorem angle_b_is_acute {α β γ : ℝ} (hC : is_right_angle γ) (hSum : angles_sum_to_180 α β γ) : is_acute_angle β :=
by
  sorry

end angle_b_is_acute_l672_67210


namespace kiwi_count_l672_67298

theorem kiwi_count (o a b k : ℕ) (h1 : o + a + b + k = 540) (h2 : a = 3 * o) (h3 : b = 4 * a) (h4 : k = 5 * b) : k = 420 :=
sorry

end kiwi_count_l672_67298


namespace raspberry_pie_degrees_l672_67246

def total_students : ℕ := 48
def chocolate_preference : ℕ := 18
def apple_preference : ℕ := 10
def blueberry_preference : ℕ := 8
def remaining_students : ℕ := total_students - chocolate_preference - apple_preference - blueberry_preference
def raspberry_preference : ℕ := remaining_students / 2
def pie_chart_degrees : ℕ := (raspberry_preference * 360) / total_students

theorem raspberry_pie_degrees :
  pie_chart_degrees = 45 := by
  sorry

end raspberry_pie_degrees_l672_67246


namespace bees_count_l672_67281

-- Definitions of the conditions
def day1_bees (x : ℕ) := x  -- Number of bees on the first day
def day2_bees (x : ℕ) := 3 * day1_bees x  -- Number of bees on the second day is 3 times that on the first day

theorem bees_count (x : ℕ) (h : day2_bees x = 432) : day1_bees x = 144 :=
by
  dsimp [day1_bees, day2_bees] at h
  have h1 : 3 * x = 432 := h
  sorry

end bees_count_l672_67281


namespace find_pairs_l672_67255

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end find_pairs_l672_67255


namespace cost_difference_is_360_l672_67293

def sailboat_cost_per_day : ℕ := 60
def ski_boat_cost_per_hour : ℕ := 80
def ken_days : ℕ := 2
def aldrich_hours_per_day : ℕ := 3
def aldrich_days : ℕ := 2

theorem cost_difference_is_360 :
  let ken_total_cost := sailboat_cost_per_day * ken_days
  let aldrich_total_cost_per_day := ski_boat_cost_per_hour * aldrich_hours_per_day
  let aldrich_total_cost := aldrich_total_cost_per_day * aldrich_days
  let cost_diff := aldrich_total_cost - ken_total_cost
  cost_diff = 360 :=
by
  sorry

end cost_difference_is_360_l672_67293


namespace inverse_function_log_l672_67231

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem inverse_function_log (a : ℝ) (g : ℝ → ℝ) (x : ℝ) (y : ℝ) :
  (a > 0) → (a ≠ 1) → 
  (f 2 a = 4) → 
  (f y a = x) → 
  (g x = y) → 
  g x = Real.logb 2 x := 
by
  intros ha hn hfx hfy hg
  sorry

end inverse_function_log_l672_67231


namespace highest_score_l672_67244

variables (H L : ℕ) (average_46 : ℕ := 61) (innings_46 : ℕ := 46) 
                (difference : ℕ := 150) (average_44 : ℕ := 58) (innings_44 : ℕ := 44)

theorem highest_score:
  (H - L = difference) →
  (average_46 * innings_46 = average_44 * innings_44 + H + L) →
  H = 202 :=
by
  intros h_diff total_runs_eq
  sorry

end highest_score_l672_67244


namespace direction_vector_of_line_l672_67227

theorem direction_vector_of_line : ∃ Δx Δy : ℚ, y = - (1/2) * x + 1 → Δx = 2 ∧ Δy = -1 :=
sorry

end direction_vector_of_line_l672_67227


namespace tiered_water_pricing_l672_67261

theorem tiered_water_pricing (x : ℝ) (y : ℝ) : 
  (∀ z, 0 ≤ z ∧ z ≤ 12 → y = 3 * z ∨
        12 < z ∧ z ≤ 18 → y = 36 + 6 * (z - 12) ∨
        18 < z → y = 72 + 9 * (z - 18)) → 
  y = 54 → 
  x = 15 :=
by
  sorry

end tiered_water_pricing_l672_67261


namespace fractional_part_lawn_remainder_l672_67257

def mary_mowing_time := 3 -- Mary can mow the lawn in 3 hours
def tom_mowing_time := 6  -- Tom can mow the lawn in 6 hours
def mary_working_hours := 1 -- Mary works for 1 hour alone

theorem fractional_part_lawn_remainder : 
  (1 - mary_working_hours / mary_mowing_time) = 2 / 3 := 
by
  sorry

end fractional_part_lawn_remainder_l672_67257


namespace ratio_of_areas_of_circles_l672_67277

theorem ratio_of_areas_of_circles (C_A C_B C_C : ℝ) (h1 : (60 / 360) * C_A = (40 / 360) * C_B) (h2 : (30 / 360) * C_B = (90 / 360) * C_C) : 
  (C_A / (2 * Real.pi))^2 / (C_C / (2 * Real.pi))^2 = 2 :=
by
  sorry

end ratio_of_areas_of_circles_l672_67277


namespace volume_is_correct_l672_67238

noncomputable def volume_of_rectangular_parallelepiped (a b : ℝ) (h_diag : (2 * a^2 + b^2 = 1)) (h_surface_area : (4 * a * b + 2 * a^2 = 1)) : ℝ :=
  a^2 * b

theorem volume_is_correct (a b : ℝ)
  (h_diag : 2 * a^2 + b^2 = 1)
  (h_surface_area : 4 * a * b + 2 * a^2 = 1) :
  volume_of_rectangular_parallelepiped a b h_diag h_surface_area = (Real.sqrt 2) / 27 :=
sorry

end volume_is_correct_l672_67238


namespace power_sum_inequality_l672_67206

theorem power_sum_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
by sorry

end power_sum_inequality_l672_67206


namespace youngest_child_cakes_l672_67253

theorem youngest_child_cakes : 
  let total_cakes := 60
  let oldest_cakes := (1 / 4 : ℚ) * total_cakes
  let second_oldest_cakes := (3 / 10 : ℚ) * total_cakes
  let middle_cakes := (1 / 6 : ℚ) * total_cakes
  let second_youngest_cakes := (1 / 5 : ℚ) * total_cakes
  let distributed_cakes := oldest_cakes + second_oldest_cakes + middle_cakes + second_youngest_cakes
  let youngest_cakes := total_cakes - distributed_cakes
  youngest_cakes = 5 := 
by
  exact sorry

end youngest_child_cakes_l672_67253


namespace paul_bought_150_books_l672_67288

theorem paul_bought_150_books (initial_books sold_books books_now : ℤ)
  (h1 : initial_books = 2)
  (h2 : sold_books = 94)
  (h3 : books_now = 58) :
  initial_books - sold_books + books_now = 150 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end paul_bought_150_books_l672_67288


namespace tan_angle_equiv_tan_1230_l672_67229

theorem tan_angle_equiv_tan_1230 : ∃ n : ℤ, -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1230 * Real.pi / 180) :=
sorry

end tan_angle_equiv_tan_1230_l672_67229


namespace number_of_divisors_23232_l672_67271

theorem number_of_divisors_23232 : ∀ (n : ℕ), 
    n = 23232 → 
    (∃ k : ℕ, k = 42 ∧ (∀ d : ℕ, (d > 0 ∧ d ∣ n) → (↑d < k + 1))) :=
by
  sorry

end number_of_divisors_23232_l672_67271


namespace interval_second_bell_l672_67276

theorem interval_second_bell 
  (T : ℕ)
  (h1 : ∀ n : ℕ, n ≠ 0 → 630 % n = 0)
  (h2 : gcd T 630 = T)
  (h3 : lcm 9 (lcm 14 18) = lcm 9 (lcm 14 18))
  (h4 : 630 % lcm 9 (lcm 14 18) = 0) : 
  T = 5 :=
sorry

end interval_second_bell_l672_67276


namespace evaluate_expression_l672_67242

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3 / 2 :=
by
  sorry

end evaluate_expression_l672_67242


namespace no_integers_satisfy_eq_l672_67218

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^2 + 1954 = n^2) := 
by
  sorry

end no_integers_satisfy_eq_l672_67218


namespace cube_sum_l672_67236

theorem cube_sum (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end cube_sum_l672_67236


namespace amount_of_second_alloy_used_l672_67286

variable (x : ℝ)

-- Conditions
def chromium_in_first_alloy : ℝ := 0.10 * 15
def chromium_in_second_alloy (x : ℝ) : ℝ := 0.06 * x
def total_weight (x : ℝ) : ℝ := 15 + x
def chromium_in_third_alloy (x : ℝ) : ℝ := 0.072 * (15 + x)

-- Proof statement
theorem amount_of_second_alloy_used :
  1.5 + 0.06 * x = 0.072 * (15 + x) → x = 35 := by
  sorry

end amount_of_second_alloy_used_l672_67286


namespace larry_win_probability_correct_l672_67269

/-- Define the probabilities of knocking off the bottle for both players in the first four turns. -/
structure GameProb (turns : ℕ) :=
  (larry_prob : ℚ)
  (julius_prob : ℚ)

/-- Define the probabilities of knocking off the bottle for both players from the fifth turn onwards. -/
def subsequent_turns_prob : ℚ := 1 / 2
/-- Initial probabilities for the first four turns -/
def initial_prob : GameProb 4 := { larry_prob := 2 / 3, julius_prob := 1 / 3 }
/-- The probability that Larry wins the game -/
def larry_wins (prob : GameProb 4) (subsequent_prob : ℚ) : ℚ :=
  -- Calculation logic goes here resulting in the final probability
  379 / 648

theorem larry_win_probability_correct :
  larry_wins initial_prob subsequent_turns_prob = 379 / 648 :=
sorry

end larry_win_probability_correct_l672_67269


namespace max_min_z_diff_correct_l672_67215

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end max_min_z_diff_correct_l672_67215


namespace average_of_scores_with_average_twice_l672_67209

variable (scores: List ℝ) (A: ℝ) (A': ℝ)
variable (h1: scores.length = 50)
variable (h2: A = (scores.sum) / 50)
variable (h3: A' = ((scores.sum + 2 * A) / 52))

theorem average_of_scores_with_average_twice (h1: scores.length = 50) (h2: A = (scores.sum) / 50) (h3: A' = ((scores.sum + 2 * A) / 52)) :
  A' = A :=
by
  sorry

end average_of_scores_with_average_twice_l672_67209


namespace fraction_of_90_l672_67243

theorem fraction_of_90 : (1 / 2) * (1 / 3) * (1 / 6) * (90 : ℝ) = (5 / 2) := by
  sorry

end fraction_of_90_l672_67243


namespace factor_expression_l672_67294

theorem factor_expression (a : ℝ) : 
  49 * a ^ 3 + 245 * a ^ 2 + 588 * a = 49 * a * (a ^ 2 + 5 * a + 12) :=
by
  sorry

end factor_expression_l672_67294


namespace find_f3_l672_67291

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f3 (h1 : ∀ x : ℝ, f (x + 1) = f (-x - 1))
                (h2 : ∀ x : ℝ, f (2 - x) = -f x) :
  f 3 = 0 := 
sorry

end find_f3_l672_67291


namespace permutation_combination_example_l672_67228

-- Definition of permutation (A) and combination (C) in Lean
def permutation (n k : ℕ): ℕ := Nat.factorial n / Nat.factorial (n - k)
def combination (n k : ℕ): ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The Lean statement of the proof problem
theorem permutation_combination_example : 
3 * permutation 3 2 + 2 * combination 4 2 = 30 := 
by 
  sorry

end permutation_combination_example_l672_67228


namespace fill_boxes_l672_67268

theorem fill_boxes (a b c d e f g : ℤ) 
  (h1 : a + (-1) + 2 = 4)
  (h2 : 2 + 1 + b = 3)
  (h3 : c + (-4) + (-3) = -2)
  (h4 : b - 5 - 4 = -9)
  (h5 : f = d - 3)
  (h6 : g = d + 3)
  (h7 : -8 = 4 + 3 - 9 - 2 + (d - 3) + (d + 3)) : 
  a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 :=
by {
  sorry
}

end fill_boxes_l672_67268


namespace y_relationship_l672_67233

theorem y_relationship :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = (-2)^2 - 4*(-2) - 3) ∧ 
  (y2 = 1^2 - 4*1 - 3) ∧ 
  (y3 = 4^2 - 4*4 - 3) → 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end y_relationship_l672_67233


namespace percentage_less_than_l672_67224

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end percentage_less_than_l672_67224


namespace meal_cost_l672_67270

theorem meal_cost (total_paid change tip_rate : ℝ)
  (h_total_paid : total_paid = 20 - change)
  (h_change : change = 5)
  (h_tip_rate : tip_rate = 0.2) :
  ∃ x, x + tip_rate * x = total_paid ∧ x = 12.5 := 
by
  sorry

end meal_cost_l672_67270
