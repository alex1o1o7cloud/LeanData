import Mathlib

namespace divisors_not_multiples_of_14_l1856_185682

theorem divisors_not_multiples_of_14 (m : ℕ)
  (h1 : ∃ k : ℕ, m = 2 * k ∧ (k : ℕ) * k = m / 2)  
  (h2 : ∃ k : ℕ, m = 3 * k ∧ (k : ℕ) * k * k = m / 3)  
  (h3 : ∃ k : ℕ, m = 7 * k ∧ (k : ℕ) ^ 7 = m / 7) : 
  let total_divisors := (6 + 1) * (10 + 1) * (7 + 1)
  let divisors_divisible_by_14 := (5 + 1) * (10 + 1) * (6 + 1)
  total_divisors - divisors_divisible_by_14 = 154 :=
by
  sorry

end divisors_not_multiples_of_14_l1856_185682


namespace trajectory_is_parabola_l1856_185622

theorem trajectory_is_parabola (C : ℝ × ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ)
  (hM : M = (0, 3)) (hl : ∀ y, l y = -3)
  (h : dist C M = |C.2 + 3|) : C.1^2 = 12 * C.2 := by
  sorry

end trajectory_is_parabola_l1856_185622


namespace largest_circle_radius_l1856_185657

theorem largest_circle_radius 
  (h H : ℝ) (h_pos : h > 0) (H_pos : H > 0) :
  ∃ R, R = (h * H) / (h + H) :=
sorry

end largest_circle_radius_l1856_185657


namespace max_least_integer_l1856_185686

theorem max_least_integer (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2160) (h_order : x ≤ y ∧ y ≤ z) : x ≤ 10 :=
by
  sorry

end max_least_integer_l1856_185686


namespace anthony_total_pencils_l1856_185664

theorem anthony_total_pencils (initial_pencils : ℕ) (pencils_given_by_kathryn : ℕ) (total_pencils : ℕ) :
  initial_pencils = 9 →
  pencils_given_by_kathryn = 56 →
  total_pencils = initial_pencils + pencils_given_by_kathryn →
  total_pencils = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end anthony_total_pencils_l1856_185664


namespace tully_twice_kate_in_three_years_l1856_185691

-- Definitions for the conditions
def tully_was := 60
def kate_is := 29

-- Number of years from now when Tully will be twice as old as Kate
theorem tully_twice_kate_in_three_years : 
  ∃ (x : ℕ), (tully_was + 1 + x = 2 * (kate_is + x)) ∧ x = 3 :=
by
  sorry

end tully_twice_kate_in_three_years_l1856_185691


namespace maximize_abs_sum_solution_problem_l1856_185652

theorem maximize_abs_sum_solution :
ℤ → ℤ → Ennreal := sorry

theorem problem :
  (∃ (x y : ℤ), 6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7 ∧ 
  x = -8 ∧ y = 25 ∧ (maximize_abs_sum_solution x y = 33)) := sorry

end maximize_abs_sum_solution_problem_l1856_185652


namespace find_certain_number_l1856_185625

theorem find_certain_number (n : ℕ)
  (h1 : 3153 + 3 = 3156)
  (h2 : 3156 % 9 = 0)
  (h3 : 3156 % 70 = 0)
  (h4 : 3156 % 25 = 0) :
  3156 % 37 = 0 :=
by
  sorry

end find_certain_number_l1856_185625


namespace negation_universal_proposition_l1856_185636

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_universal_proposition_l1856_185636


namespace meetings_percentage_l1856_185632

-- Define all the conditions given in the problem
def first_meeting := 60 -- duration of first meeting in minutes
def second_meeting := 2 * first_meeting -- duration of second meeting in minutes
def third_meeting := first_meeting / 2 -- duration of third meeting in minutes
def total_meeting_time := first_meeting + second_meeting + third_meeting -- total meeting time
def total_workday := 10 * 60 -- total workday time in minutes

-- Statement to prove that the percentage of workday spent in meetings is 35%
def percent_meetings : Prop := (total_meeting_time / total_workday) * 100 = 35

theorem meetings_percentage :
  percent_meetings :=
by
  sorry

end meetings_percentage_l1856_185632


namespace binom_mult_eq_6720_l1856_185616

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l1856_185616


namespace cube_properties_l1856_185692

theorem cube_properties (y : ℝ) (s : ℝ) 
  (h_volume : s^3 = 6 * y)
  (h_surface_area : 6 * s^2 = 2 * y) :
  y = 5832 :=
by sorry

end cube_properties_l1856_185692


namespace casey_saves_by_paying_monthly_l1856_185635

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_in_a_month := 4
  let number_of_months := 3
  let total_weeks := number_of_months * weeks_in_a_month
  let total_cost_weekly := total_weeks * weekly_rate
  let total_cost_monthly := number_of_months * monthly_rate
  let savings := total_cost_weekly - total_cost_monthly
  savings = 360 :=
by
  sorry

end casey_saves_by_paying_monthly_l1856_185635


namespace range_of_sum_abs_l1856_185693

variable {x y z : ℝ}

theorem range_of_sum_abs : 
  x^2 + y^2 + z = 15 → 
  x + y + z^2 = 27 → 
  xy + yz + zx = 7 → 
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end range_of_sum_abs_l1856_185693


namespace ac_bc_nec_not_suff_l1856_185695

theorem ac_bc_nec_not_suff (a b c : ℝ) : 
  (a = b → a * c = b * c) ∧ (¬(a * c = b * c → a = b)) := by
  sorry

end ac_bc_nec_not_suff_l1856_185695


namespace lcm_18_35_l1856_185658

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l1856_185658


namespace find_largest_even_integer_l1856_185612

-- Define the sum of the first 30 positive even integers
def sum_first_30_even : ℕ := 2 * (30 * 31 / 2)

-- Assume five consecutive even integers and their sum
def consecutive_even_sum (m : ℕ) : ℕ := (m - 8) + (m - 6) + (m - 4) + (m - 2) + m

-- Statement of the theorem to be proven
theorem find_largest_even_integer : ∃ (m : ℕ), consecutive_even_sum m = sum_first_30_even ∧ m = 190 :=
by
  sorry

end find_largest_even_integer_l1856_185612


namespace train_stop_time_per_hour_l1856_185633

theorem train_stop_time_per_hour
    (v1 : ℕ) (v2 : ℕ)
    (h1 : v1 = 45)
    (h2 : v2 = 33) : ∃ (t : ℕ), t = 16 := by
  -- including the proof steps here is unnecessary, so we use sorry
  sorry

end train_stop_time_per_hour_l1856_185633


namespace fixed_point_coordinates_l1856_185619

theorem fixed_point_coordinates (k : ℝ) (M : ℝ × ℝ) (h : ∀ k : ℝ, M.2 - 2 = k * (M.1 + 1)) :
  M = (-1, 2) :=
sorry

end fixed_point_coordinates_l1856_185619


namespace product_remainder_l1856_185613

theorem product_remainder (a b c d : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 4) (hd : d % 7 = 5) :
  (a * b * c * d) % 7 = 1 :=
by
  sorry

end product_remainder_l1856_185613


namespace triangle_base_second_l1856_185647

theorem triangle_base_second (base1 height1 height2 : ℝ) 
  (h_base1 : base1 = 15) (h_height1 : height1 = 12) (h_height2 : height2 = 18) :
  let area1 := (base1 * height1) / 2
  let area2 := 2 * area1
  let base2 := (2 * area2) / height2
  base2 = 20 :=
by
  sorry

end triangle_base_second_l1856_185647


namespace part1_part2_l1856_185615

noncomputable def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

theorem part1 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 2) :
  {x : ℝ | f x a b ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem part2 (a b : ℝ) (h_min_value : ∀ x : ℝ, f x a b ≥ 3) :
  a + b = 3 → (a > 0 ∧ b > 0) →
  (∃ a b : ℝ, a = b ∧ a + b = 3 ∧ (a = b → f x a b = 3)) →
  (∀ a b : ℝ, (a^2/b + b^2/a) ≥ 3) :=
by
  sorry

end part1_part2_l1856_185615


namespace find_x3_minus_y3_l1856_185674

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l1856_185674


namespace exists_word_D_l1856_185675

variable {α : Type} [Inhabited α] [DecidableEq α]

def repeats (D : List α) (w : List α) : Prop :=
  ∃ k : ℕ, w = List.join (List.replicate k D)

theorem exists_word_D (A B C : List α)
  (h : (A ++ A ++ B ++ B) = (C ++ C)) :
  ∃ D : List α, repeats D A ∧ repeats D B ∧ repeats D C :=
sorry

end exists_word_D_l1856_185675


namespace soda_cost_132_cents_l1856_185621

theorem soda_cost_132_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s + 30 = 510)
  (h2 : 2 * b + 3 * s = 540) 
  : s = 132 :=
by
  sorry

end soda_cost_132_cents_l1856_185621


namespace total_cost_verification_l1856_185667

-- Conditions given in the problem
def holstein_cost : ℕ := 260
def jersey_cost : ℕ := 170
def num_hearts_on_card : ℕ := 4
def num_cards_in_deck : ℕ := 52
def cow_ratio_holstein : ℕ := 3
def cow_ratio_jersey : ℕ := 2
def sales_tax : ℝ := 0.05
def transport_cost_per_cow : ℕ := 20

def num_hearts_in_deck := num_cards_in_deck
def total_num_cows := 2 * num_hearts_in_deck
def total_parts_ratio := cow_ratio_holstein + cow_ratio_jersey

-- Total number of cows calculated 
def num_holstein_cows : ℕ := (cow_ratio_holstein * total_num_cows) / total_parts_ratio
def num_jersey_cows : ℕ := (cow_ratio_jersey * total_num_cows) / total_parts_ratio

-- Cost calculations
def holstein_total_cost := num_holstein_cows * holstein_cost
def jersey_total_cost := num_jersey_cows * jersey_cost
def total_cost_before_tax_and_transport := holstein_total_cost + jersey_total_cost
def total_sales_tax := total_cost_before_tax_and_transport * sales_tax
def total_transport_cost := total_num_cows * transport_cost_per_cow
def final_total_cost := total_cost_before_tax_and_transport + total_sales_tax + total_transport_cost

-- Lean statement to prove the result
theorem total_cost_verification : final_total_cost = 26324.50 := by sorry

end total_cost_verification_l1856_185667


namespace cost_of_fencing_per_meter_l1856_185688

theorem cost_of_fencing_per_meter
  (length : ℕ) (breadth : ℕ) (total_cost : ℝ) (cost_per_meter : ℝ)
  (h1 : length = 64) 
  (h2 : length = breadth + 28)
  (h3 : total_cost = 5300)
  (h4 : cost_per_meter = total_cost / (2 * (length + breadth))) :
  cost_per_meter = 26.50 :=
by {
  sorry
}

end cost_of_fencing_per_meter_l1856_185688


namespace count_total_legs_l1856_185677

theorem count_total_legs :
  let tables4 := 4 * 4
  let sofa := 1 * 4
  let chairs4 := 2 * 4
  let tables3 := 3 * 3
  let table1 := 1 * 1
  let rocking_chair := 1 * 2
  let total_legs := tables4 + sofa + chairs4 + tables3 + table1 + rocking_chair
  total_legs = 40 :=
by
  sorry

end count_total_legs_l1856_185677


namespace max_value_7x_10y_z_l1856_185651

theorem max_value_7x_10y_z (x y z : ℝ) 
  (h : x^2 + 2 * x + (1 / 5) * y^2 + 7 * z^2 = 6) : 
  7 * x + 10 * y + z ≤ 55 := 
sorry

end max_value_7x_10y_z_l1856_185651


namespace square_of_1037_l1856_185668

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := 
by {
  -- Proof omitted
  sorry
}

end square_of_1037_l1856_185668


namespace parabola_value_f_l1856_185617

theorem parabola_value_f (d e f : ℝ) :
  (∀ y : ℝ, x = d * y ^ 2 + e * y + f) →
  (∀ x y : ℝ, (x + 3) = d * (y - 1) ^ 2) →
  (x = -1 ∧ y = 3) →
  y = 0 →
  f = -2.5 :=
sorry

end parabola_value_f_l1856_185617


namespace find_first_episode_l1856_185607

variable (x : ℕ)
variable (w y z : ℕ)
variable (total_minutes: ℕ)
variable (h1 : w = 62)
variable (h2 : y = 65)
variable (h3 : z = 55)
variable (h4 : total_minutes = 240)

theorem find_first_episode :
  x + w + y + z = total_minutes → x = 58 := 
by
  intro h
  rw [h1, h2, h3, h4] at h
  linarith

end find_first_episode_l1856_185607


namespace contrapositive_example_l1856_185627

theorem contrapositive_example (x : ℝ) (h : -2 < x ∧ x < 2) : x^2 < 4 :=
sorry

end contrapositive_example_l1856_185627


namespace melinda_doughnuts_picked_l1856_185626

theorem melinda_doughnuts_picked :
  (∀ d h_coffee m_coffee : ℕ, d = 3 → h_coffee = 4 → m_coffee = 6 →
    ∀ cost_d cost_h cost_m : ℝ, cost_d = 0.45 → 
    cost_h = 4.91 → cost_m = 7.59 → 
    ∃ m_doughnuts : ℕ, cost_m - m_coffee * ((cost_h - d * cost_d) / h_coffee) = m_doughnuts * cost_d) → 
  ∃ n : ℕ, n = 5 := 
by sorry

end melinda_doughnuts_picked_l1856_185626


namespace inclination_angle_of_line_l1856_185690

theorem inclination_angle_of_line (θ : Real) 
  (h : θ = Real.tan 45) : θ = 90 :=
sorry

end inclination_angle_of_line_l1856_185690


namespace fraction_division_addition_l1856_185679

theorem fraction_division_addition :
  (3 / 7 / 4) + (2 / 7) = 11 / 28 := by
  sorry

end fraction_division_addition_l1856_185679


namespace father_son_skating_ratio_l1856_185608

theorem father_son_skating_ratio (v_f v_s : ℝ) (h1 : v_f > v_s) (h2 : (v_f + v_s) / (v_f - v_s) = 5) :
  v_f / v_s = 1.5 :=
sorry

end father_son_skating_ratio_l1856_185608


namespace correct_calculation_l1856_185687

theorem correct_calculation (a b : ℝ) : 
  (a + 2 * a = 3 * a) := by
  sorry

end correct_calculation_l1856_185687


namespace total_fish_l1856_185639

-- Definition of the number of fish Lilly has
def lilly_fish : Nat := 10

-- Definition of the number of fish Rosy has
def rosy_fish : Nat := 8

-- Statement to prove
theorem total_fish : lilly_fish + rosy_fish = 18 := 
by
  -- The proof is omitted
  sorry

end total_fish_l1856_185639


namespace count_three_digit_integers_with_tens_7_divisible_by_25_l1856_185620

theorem count_three_digit_integers_with_tens_7_divisible_by_25 :
  ∃ n, n = 33 ∧ ∃ k1 k2 : ℕ, 175 = 25 * k1 ∧ 975 = 25 * k2 ∧ (k2 - k1 + 1 = n) :=
by
  sorry

end count_three_digit_integers_with_tens_7_divisible_by_25_l1856_185620


namespace shaded_area_is_correct_l1856_185663

theorem shaded_area_is_correct : 
  ∀ (leg_length : ℕ) (total_partitions : ℕ) (shaded_partitions : ℕ) 
    (tri_area : ℕ) (small_tri_area : ℕ) (shaded_area : ℕ), 
  leg_length = 10 → 
  total_partitions = 25 →
  shaded_partitions = 15 →
  tri_area = (1 / 2 * leg_length * leg_length) → 
  small_tri_area = (tri_area / total_partitions) →
  shaded_area = (shaded_partitions * small_tri_area) →
  shaded_area = 30 :=
by
  intros leg_length total_partitions shaded_partitions tri_area small_tri_area shaded_area
  intros h_leg_length h_total_partitions h_shaded_partitions h_tri_area h_small_tri_area h_shaded_area
  sorry

end shaded_area_is_correct_l1856_185663


namespace log_lt_x_squared_for_x_gt_zero_l1856_185618

theorem log_lt_x_squared_for_x_gt_zero (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 :=
sorry

end log_lt_x_squared_for_x_gt_zero_l1856_185618


namespace average_fuel_efficiency_l1856_185669

theorem average_fuel_efficiency (d1 d2 : ℝ) (e1 e2 : ℝ) (fuel1 fuel2 : ℝ)
  (h1 : d1 = 150) (h2 : e1 = 35) (h3 : d2 = 180) (h4 : e2 = 18)
  (h_fuel1 : fuel1 = d1 / e1) (h_fuel2 : fuel2 = d2 / e2)
  (total_distance : ℝ := 330)
  (total_fuel : ℝ := fuel1 + fuel2) :
  total_distance / total_fuel = 23 := by
  sorry

end average_fuel_efficiency_l1856_185669


namespace min_sum_of_ab_l1856_185606

theorem min_sum_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = a * b) :
  a + b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_sum_of_ab_l1856_185606


namespace jessica_can_mail_letter_l1856_185653

-- Define the constants
def paper_weight := 1/5 -- each piece of paper weighs 1/5 ounce
def envelope_weight := 2/5 -- envelope weighs 2/5 ounce
def num_papers := 8

-- Calculate the total weight
def total_weight := num_papers * paper_weight + envelope_weight

-- Define stamping rates
def international_rate := 2 -- $2 per ounce internationally

-- Calculate the required postage
def required_postage := total_weight * international_rate

-- Define the available stamp values
inductive Stamp
| one_dollar : Stamp
| fifty_cents : Stamp

-- Function to calculate the total value of a given stamp combination
def stamp_value : List Stamp → ℝ
| [] => 0
| (Stamp.one_dollar :: rest) => 1 + stamp_value rest
| (Stamp.fifty_cents :: rest) => 0.5 + stamp_value rest

-- State the theorem to be proved
theorem jessica_can_mail_letter :
  ∃ stamps : List Stamp, stamp_value stamps = required_postage := by
sorry

end jessica_can_mail_letter_l1856_185653


namespace find_f_8_5_l1856_185666

-- Conditions as definitions in Lean
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def segment_function (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- The main theorem to prove
theorem find_f_8_5 (f : ℝ → ℝ) (h1 : even_function f) (h2 : periodic_function f 3) (h3 : segment_function f)
: f 8.5 = 1.5 :=
sorry

end find_f_8_5_l1856_185666


namespace integer_solutions_eq_l1856_185670

theorem integer_solutions_eq (x y z : ℤ) :
  (x + y + z) ^ 5 = 80 * x * y * z * (x ^ 2 + y ^ 2 + z ^ 2) ↔
  ∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨ (x = -a ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = -a) ∨ (x = -a ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = -a) ∨ (x = 0 ∧ y = -a ∧ z = a) :=
by sorry

end integer_solutions_eq_l1856_185670


namespace expression_equals_16_l1856_185680

open Real

theorem expression_equals_16 (x : ℝ) :
  (x + 1) ^ 2 + 2 * (x + 1) * (3 - x) + (3 - x) ^ 2 = 16 :=
sorry

end expression_equals_16_l1856_185680


namespace min_cost_speed_l1856_185640

noncomputable def fuel_cost (v : ℝ) : ℝ := (1/200) * v^3

theorem min_cost_speed 
  (v : ℝ) 
  (u : ℝ) 
  (other_costs : ℝ) 
  (h1 : u = (1/200) * v^3) 
  (h2 : u = 40) 
  (h3 : v = 20) 
  (h4 : other_costs = 270) 
  (b : ℝ) 
  : ∃ v_min, v_min = 30 ∧ 
    ∀ (v : ℝ), (0 < v ∧ v ≤ b) → 
    ((fuel_cost v / v + other_costs / v) ≥ (fuel_cost v_min / v_min + other_costs / v_min)) := 
sorry

end min_cost_speed_l1856_185640


namespace problem1_problem2_problem3_l1856_185655

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^2 + 0.5 * x

theorem problem1 (h : ∀ x : ℝ, f (x + 1) = f x + x + 1) (h0 : f 0 = 0) : 
  ∀ x : ℝ, f x = 0.5 * x^2 + 0.5 * x := by 
  sorry

noncomputable def g (t : ℝ) : ℝ :=
  if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
  else if -1.5 < t ∧ t < -0.5 then -1 / 8
  else 0.5 * t^2 + 0.5 * t

theorem problem2 (h : ∀ t : ℝ, g t = min (f (t)) (f (t + 1))) : 
  ∀ t : ℝ, g t = 
    if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
    else if -1.5 < t ∧ t < -0.5 then -1 / 8
    else 0.5 * t^2 + 0.5 * t := by 
  sorry

theorem problem3 (m : ℝ) : (∀ t : ℝ, g t + m ≥ 0) → m ≥ 1 / 8 := by 
  sorry

end problem1_problem2_problem3_l1856_185655


namespace solve_inequality_system_l1856_185604

theorem solve_inequality_system (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ (x - 1 ≤ 7 - x) ↔ (2 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequality_system_l1856_185604


namespace enrique_shredder_pages_l1856_185605

theorem enrique_shredder_pages (total_contracts : ℕ) (num_times : ℕ) (pages_per_time : ℕ) :
  total_contracts = 2132 ∧ num_times = 44 → pages_per_time = 48 :=
by
  intros h
  sorry

end enrique_shredder_pages_l1856_185605


namespace gcd_polynomial_l1856_185628

theorem gcd_polynomial (b : ℤ) (h : 2142 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 :=
sorry

end gcd_polynomial_l1856_185628


namespace probability_two_most_expensive_l1856_185699

open Nat

noncomputable def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem probability_two_most_expensive :
  (combination 8 1) / (combination 10 3) = 1 / 15 :=
by
  sorry

end probability_two_most_expensive_l1856_185699


namespace car_cost_l1856_185689

def initial_savings : ℕ := 14500
def charge_per_trip : ℚ := 1.5
def percentage_groceries_earnings : ℚ := 0.05
def number_of_trips : ℕ := 40
def total_value_of_groceries : ℕ := 800

theorem car_cost (initial_savings charge_per_trip percentage_groceries_earnings number_of_trips total_value_of_groceries : ℚ) :
  initial_savings + (charge_per_trip * number_of_trips) + (percentage_groceries_earnings * total_value_of_groceries) = 14600 := 
by
  sorry

end car_cost_l1856_185689


namespace zero_in_interval_l1856_185659

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 2

theorem zero_in_interval : f 1 < 0 ∧ f 2 > 0 → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 := 
by
  intros h
  sorry

end zero_in_interval_l1856_185659


namespace ladybugs_calculation_l1856_185673

def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170
def ladybugs_without_spots : ℕ := 54912

theorem ladybugs_calculation :
  total_ladybugs - ladybugs_with_spots = ladybugs_without_spots :=
by
  sorry

end ladybugs_calculation_l1856_185673


namespace cannot_form_right_triangle_l1856_185645

theorem cannot_form_right_triangle :
  ¬ (6^2 + 7^2 = 8^2) :=
by
  sorry

end cannot_form_right_triangle_l1856_185645


namespace composite_10201_base_n_composite_10101_base_n_l1856_185609

-- 1. Prove that 10201_n is composite given n > 2
theorem composite_10201_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + 2*n^2 + 1 := 
sorry

-- 2. Prove that 10101_n is composite given n > 2.
theorem composite_10101_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + n^2 + 1 := 
sorry

end composite_10201_base_n_composite_10101_base_n_l1856_185609


namespace seokjin_fewer_books_l1856_185685

theorem seokjin_fewer_books (init_books : ℕ) (jungkook_initial : ℕ) (seokjin_initial : ℕ) (jungkook_bought : ℕ) (seokjin_bought : ℕ) :
  jungkook_initial = init_books → seokjin_initial = init_books → jungkook_bought = 18 → seokjin_bought = 11 →
  jungkook_initial + jungkook_bought - (seokjin_initial + seokjin_bought) = 7 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end seokjin_fewer_books_l1856_185685


namespace find_b_value_l1856_185671

theorem find_b_value (a b c : ℝ)
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : b = 15 := by
  sorry

end find_b_value_l1856_185671


namespace find_n_coordinates_l1856_185654

variables {a b : ℝ}

def is_perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_n_coordinates (n : ℝ × ℝ) (h1 : is_perpendicular (a, b) n) (h2 : same_magnitude (a, b) n) :
  n = (b, -a) :=
sorry

end find_n_coordinates_l1856_185654


namespace ab_equals_6_l1856_185656

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l1856_185656


namespace hyperbola_center_l1856_185648

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end hyperbola_center_l1856_185648


namespace dimes_in_piggy_bank_l1856_185631

variable (q d : ℕ)

def total_coins := q + d = 100
def total_amount := 25 * q + 10 * d = 1975

theorem dimes_in_piggy_bank (h1 : total_coins q d) (h2 : total_amount q d) : d = 35 := by
  sorry

end dimes_in_piggy_bank_l1856_185631


namespace who_scored_full_marks_l1856_185600

-- Define students and their statements
inductive Student
| A | B | C

open Student

def scored_full_marks (s : Student) : Prop :=
  match s with
  | A => true
  | B => true
  | C => true

def statement_A : Prop := scored_full_marks A
def statement_B : Prop := ¬ scored_full_marks C
def statement_C : Prop := statement_B

-- Given conditions
def exactly_one_lied (a b c : Prop) : Prop :=
  (a ∧ ¬ b ∧ ¬ c) ∨ (¬ a ∧ b ∧ ¬ c) ∨ (¬ a ∧ ¬ b ∧ c)

-- Main proof statement: Prove that B scored full marks
theorem who_scored_full_marks (h : exactly_one_lied statement_A statement_B statement_C) : scored_full_marks B :=
sorry

end who_scored_full_marks_l1856_185600


namespace skateboarder_speed_l1856_185649

theorem skateboarder_speed :
  let distance := 293.33
  let time := 20
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  let speed_ft_per_sec := distance / time
  let speed_mph := speed_ft_per_sec * (feet_per_mile / seconds_per_hour)
  speed_mph = 21.5 :=
by
  sorry

end skateboarder_speed_l1856_185649


namespace find_a_l1856_185637

def has_root_greater_than_zero (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ ((3 * x - 1) / (x - 3) = a / (3 - x) - 1)

theorem find_a (a : ℝ) : has_root_greater_than_zero a → a = -8 :=
by
  sorry

end find_a_l1856_185637


namespace highest_financial_backing_l1856_185610

-- Let x be the lowest level of financial backing
-- Define the five levels of backing as x, 6x, 36x, 216x, 1296x
-- Given that the total raised is $200,000

theorem highest_financial_backing (x : ℝ) 
  (h₁: 50 * x + 20 * 6 * x + 12 * 36 * x + 7 * 216 * x + 4 * 1296 * x = 200000) : 
  1296 * x = 35534 :=
sorry

end highest_financial_backing_l1856_185610


namespace speed_of_man_correct_l1856_185672

noncomputable def speed_of_man_in_kmph (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_pass_sec : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := (train_length_m / time_pass_sec)
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * 3600 / 1000

theorem speed_of_man_correct : 
  speed_of_man_in_kmph 77.993280537557 140 6 = 6.00871946444388 := 
by simp [speed_of_man_in_kmph]; sorry

end speed_of_man_correct_l1856_185672


namespace minimum_p_for_required_profit_l1856_185634

noncomputable def profit (x p : ℝ) : ℝ := p * x - (0.5 * x^2 - 2 * x - 10)
noncomputable def max_profit (p : ℝ) : ℝ := (p + 2)^2 / 2 + 10

theorem minimum_p_for_required_profit : ∀ (p : ℝ), 3 * max_profit p >= 126 → p >= 6 :=
by
  intro p
  unfold max_profit
  -- Given:  3 * ((p + 2)^2 / 2 + 10) >= 126
  sorry

end minimum_p_for_required_profit_l1856_185634


namespace largest_inscribed_square_size_l1856_185603

noncomputable def side_length_of_largest_inscribed_square : ℝ :=
  6 - 2 * Real.sqrt 3

theorem largest_inscribed_square_size (side_length_of_square : ℝ)
  (equi_triangles_shared_side : ℝ)
  (vertexA_of_square : ℝ)
  (vertexB_of_square : ℝ)
  (vertexC_of_square : ℝ)
  (vertexD_of_square : ℝ)
  (vertexF_of_triangles : ℝ)
  (vertexG_of_triangles : ℝ) :
  side_length_of_square = 12 →
  equi_triangles_shared_side = vertexB_of_square - vertexA_of_square →
  vertexF_of_triangles = vertexD_of_square - vertexC_of_square →
  vertexG_of_triangles = vertexB_of_square - vertexA_of_square →
  side_length_of_largest_inscribed_square = 6 - 2 * Real.sqrt 3 :=
sorry

end largest_inscribed_square_size_l1856_185603


namespace tricycles_count_l1856_185676

-- Define the variables for number of bicycles, tricycles, and scooters.
variables (b t s : ℕ)

-- Define the total number of children and total number of wheels conditions.
def children_condition := b + t + s = 10
def wheels_condition := 2 * b + 3 * t + 2 * s = 27

-- Prove that number of tricycles t is 4 under these conditions.
theorem tricycles_count : children_condition b t s → wheels_condition b t s → t = 4 := by
  sorry

end tricycles_count_l1856_185676


namespace relationship_bx_l1856_185630

variable {a b t x : ℝ}

-- Given conditions
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : t > 0)
variable (h4 : a ^ x = a + t)

theorem relationship_bx (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a ^ x = a + t) : b ^ x > b + t :=
by
  sorry

end relationship_bx_l1856_185630


namespace inequality_not_true_l1856_185646

theorem inequality_not_true (a b : ℝ) (h : a > b) : (a / (-2)) ≤ (b / (-2)) :=
sorry

end inequality_not_true_l1856_185646


namespace div_40_of_prime_ge7_l1856_185698

theorem div_40_of_prime_ge7 (p : ℕ) (hp_prime : Prime p) (hp_ge7 : p ≥ 7) : 40 ∣ (p^2 - 1) :=
sorry

end div_40_of_prime_ge7_l1856_185698


namespace max_cursed_roads_l1856_185642

theorem max_cursed_roads (cities roads N kingdoms : ℕ) (h1 : cities = 1000) (h2 : roads = 2017)
  (h3 : cities = 1 → cities = 1000 → N ≤ 1024 → kingdoms = 7 → True) :
  max_N = 1024 :=
by
  sorry

end max_cursed_roads_l1856_185642


namespace min_value_expression_l1856_185661

theorem min_value_expression (a b : ℝ) : ∃ v : ℝ, ∀ (a b : ℝ), (a^2 + a * b + b^2 - a - 2 * b) ≥ v ∧ v = -1 :=
by
  sorry

end min_value_expression_l1856_185661


namespace tangerines_in_one_box_l1856_185638

theorem tangerines_in_one_box (total_tangerines boxes remaining_tangerines tangerines_per_box : ℕ) 
  (h1 : total_tangerines = 29)
  (h2 : boxes = 8)
  (h3 : remaining_tangerines = 5)
  (h4 : total_tangerines - remaining_tangerines = boxes * tangerines_per_box) :
  tangerines_per_box = 3 :=
by 
  sorry

end tangerines_in_one_box_l1856_185638


namespace find_notebook_price_l1856_185643

noncomputable def notebook_and_pencil_prices : Prop :=
  ∃ (x y : ℝ),
    5 * x + 4 * y = 16.5 ∧
    2 * x + 2 * y = 7 ∧
    x = 2.5

theorem find_notebook_price : notebook_and_pencil_prices :=
  sorry

end find_notebook_price_l1856_185643


namespace xy_value_l1856_185678

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 :=
by
  sorry

end xy_value_l1856_185678


namespace original_price_l1856_185660

theorem original_price (P : ℝ) (h1 : 0.76 * P = 820) : P = 1079 :=
by
  sorry

end original_price_l1856_185660


namespace units_digit_2_pow_2015_l1856_185601

theorem units_digit_2_pow_2015 : ∃ u : ℕ, (2 ^ 2015 % 10) = u ∧ u = 8 := 
by
  sorry

end units_digit_2_pow_2015_l1856_185601


namespace amount_spent_on_food_l1856_185683

-- We define the conditions given in the problem
def Mitzi_brought_money : ℕ := 75
def ticket_cost : ℕ := 30
def tshirt_cost : ℕ := 23
def money_left : ℕ := 9

-- Define the total amount Mitzi spent
def total_spent : ℕ := Mitzi_brought_money - money_left

-- Define the combined cost of the ticket and T-shirt
def combined_cost : ℕ := ticket_cost + tshirt_cost

-- The proof goal
theorem amount_spent_on_food : total_spent - combined_cost = 13 := by
  sorry

end amount_spent_on_food_l1856_185683


namespace total_cost_of_trip_l1856_185644

def totalDistance (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def gallonsUsed (distance miles_per_gallon : ℕ) : ℕ :=
  distance / miles_per_gallon

def totalCost (gallons : ℕ) (cost_per_gallon : ℕ) : ℕ :=
  gallons * cost_per_gallon

theorem total_cost_of_trip :
  (totalDistance 10 6 5 9 = 30) →
  (gallonsUsed 30 15 = 2) →
  totalCost 2 35 = 700 :=
by
  sorry

end total_cost_of_trip_l1856_185644


namespace total_animals_l1856_185665

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end total_animals_l1856_185665


namespace perfect_square_iff_all_perfect_squares_l1856_185611

theorem perfect_square_iff_all_perfect_squares
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0):
  (∃ k : ℕ, (xy + 1) * (yz + 1) * (zx + 1) = k^2) ↔
  (∃ a b c : ℕ, xy + 1 = a^2 ∧ yz + 1 = b^2 ∧ zx + 1 = c^2) := 
sorry

end perfect_square_iff_all_perfect_squares_l1856_185611


namespace helen_needed_gas_l1856_185624

-- Definitions based on conditions
def cuts_per_month_routine_1 : ℕ := 2 -- Cuts per month for March, April, September, October
def cuts_per_month_routine_2 : ℕ := 4 -- Cuts per month for May, June, July, August
def months_routine_1 : ℕ := 4 -- Number of months with routine 1
def months_routine_2 : ℕ := 4 -- Number of months with routine 2
def gas_per_fill : ℕ := 2 -- Gallons of gas used every 4th cut
def cuts_per_fill : ℕ := 4 -- Number of cuts per fill

-- Total number of cuts in routine 1 months
def total_cuts_routine_1 : ℕ := cuts_per_month_routine_1 * months_routine_1

-- Total number of cuts in routine 2 months
def total_cuts_routine_2 : ℕ := cuts_per_month_routine_2 * months_routine_2

-- Total cuts from March to October
def total_cuts : ℕ := total_cuts_routine_1 + total_cuts_routine_2

-- Total fills needed from March to October
def total_fills : ℕ := total_cuts / cuts_per_fill

-- Total gallons of gas needed
def total_gal_of_gas : ℕ := total_fills * gas_per_fill

-- The statement to prove
theorem helen_needed_gas : total_gal_of_gas = 12 :=
by
  -- This would be replaced by our solution steps.
  sorry

end helen_needed_gas_l1856_185624


namespace arithmetic_geometric_progression_l1856_185684

-- Define the arithmetic progression terms
def u (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the property that the squares of the 12th, 13th, and 15th terms form a geometric progression
def geometric_progression (a d : ℝ) : Prop :=
  let u12 := u a d 12
  let u13 := u a d 13
  let u15 := u a d 15
  (u13^2 / u12^2 = u15^2 / u13^2)

-- The main statement
theorem arithmetic_geometric_progression (a d : ℝ) (h : geometric_progression a d) :
  d = 0 ∨ 4 * ((a + 11 * d)^2) = (a + 12 *d)^2 * (a + 14 * d)^2 / (a + 12 * d)^2 ∨ (a + 11 * d) * ((a + 11 * d) - 2 *d) = 0 :=
sorry

end arithmetic_geometric_progression_l1856_185684


namespace investment_recovery_l1856_185694

-- Define the conditions and the goal
theorem investment_recovery (c : ℕ) : 
  (15 * c - 5 * c) ≥ 8000 ↔ c ≥ 800 := 
sorry

end investment_recovery_l1856_185694


namespace part1_part2_l1856_185614

theorem part1 (m : ℝ) (h_m_not_zero : m ≠ 0) : m ≤ 4 / 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

theorem part2 (m : ℕ) (h_m_range : m = 1) :
  ∃ x1 x2 : ℝ, (m * x1^2 - 4 * x1 + 3 = 0) ∧ (m * x2^2 - 4 * x2 + 3 = 0) ∧ x1 = 1 ∧ x2 = 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

end part1_part2_l1856_185614


namespace age_difference_l1856_185662

theorem age_difference (M T J X S : ℕ)
  (hM : M = 3)
  (hT : T = 4 * M)
  (hJ : J = T - 5)
  (hX : X = 2 * J)
  (hS : S = 3 * X - 1) :
  S - M = 38 :=
by
  sorry

end age_difference_l1856_185662


namespace extra_games_needed_l1856_185650

def initial_games : ℕ := 500
def initial_success_rate : ℚ := 0.49
def target_success_rate : ℚ := 0.5

theorem extra_games_needed :
  ∀ (x : ℕ),
  (245 + x) / (initial_games + x) = target_success_rate → x = 10 := 
by
  sorry

end extra_games_needed_l1856_185650


namespace solution_set_of_inequality_l1856_185623

theorem solution_set_of_inequality : {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x^2 + 2 * x < 3} :=
sorry

end solution_set_of_inequality_l1856_185623


namespace smaller_cylinder_diameter_l1856_185602

theorem smaller_cylinder_diameter
  (vol_large : ℝ)
  (height_large : ℝ)
  (diameter_large : ℝ)
  (height_small : ℝ)
  (ratio : ℝ)
  (π : ℝ)
  (volume_large_eq : vol_large = π * (diameter_large / 2)^2 * height_large)  -- Volume formula for the larger cylinder
  (ratio_eq : ratio = 74.07407407407408) -- Given ratio
  (height_large_eq : height_large = 10)  -- Given height of the larger cylinder
  (diameter_large_eq : diameter_large = 20)  -- Given diameter of the larger cylinder
  (height_small_eq : height_small = 6)  -- Given height of smaller cylinders):
  :
  ∃ (diameter_small : ℝ), diameter_small = 3 := 
by
  sorry

end smaller_cylinder_diameter_l1856_185602


namespace cashback_percentage_l1856_185681

theorem cashback_percentage
  (total_cost : ℝ) (rebate : ℝ) (final_cost : ℝ)
  (H1 : total_cost = 150) (H2 : rebate = 25) (H3 : final_cost = 110) :
  (total_cost - rebate - final_cost) / (total_cost - rebate) * 100 = 12 := by
  sorry

end cashback_percentage_l1856_185681


namespace problem_solution_l1856_185697

theorem problem_solution (a : ℝ) : 
  ( ∀ x : ℝ, (ax - 1) * (x + 1) < 0 ↔ (x ∈ Set.Iio (-1) ∨ x ∈ Set.Ioi (-1 / 2)) ) →
  a = -2 :=
by
  sorry

end problem_solution_l1856_185697


namespace perimeter_ABFCDE_l1856_185641

theorem perimeter_ABFCDE 
  (ABCD_perimeter : ℝ)
  (ABCD : ℝ)
  (triangle_BFC : ℝ -> ℝ)
  (translate_BFC : ℝ -> ℝ)
  (ABFCDE : ℝ -> ℝ -> ℝ)
  (h1 : ABCD_perimeter = 40)
  (h2 : ABCD = ABCD_perimeter / 4)
  (h3 : triangle_BFC ABCD = 10 * Real.sqrt 2)
  (h4 : translate_BFC (10 * Real.sqrt 2) = 10 * Real.sqrt 2)
  (h5 : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2)
  : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2 := 
by 
  sorry

end perimeter_ABFCDE_l1856_185641


namespace find_p_from_circle_and_parabola_tangency_l1856_185696

theorem find_p_from_circle_and_parabola_tangency :
  (∃ x y : ℝ, (x^2 + y^2 - 6*x - 7 = 0) ∧ (y^2 = 2*p * x) ∧ p > 0) →
  p = 2 :=
by {
  sorry
}

end find_p_from_circle_and_parabola_tangency_l1856_185696


namespace max_value_of_function_l1856_185629

noncomputable def y (x : ℝ) : ℝ := 
  Real.sin x - Real.cos x - Real.sin x * Real.cos x

theorem max_value_of_function :
  ∃ x : ℝ, y x = (1 / 2) + Real.sqrt 2 :=
sorry

end max_value_of_function_l1856_185629
