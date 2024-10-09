import Mathlib

namespace tan_diff_l1061_106193

variables {α β : ℝ}

theorem tan_diff (h1 : Real.tan α = -3/4) (h2 : Real.tan (Real.pi - β) = 1/2) :
  Real.tan (α - β) = -2/11 :=
by
  sorry

end tan_diff_l1061_106193


namespace find_first_term_l1061_106171

noncomputable def firstTermOfGeometricSeries (S : ℝ) (r : ℝ) : ℝ :=
  S * (1 - r) / (1 - r)

theorem find_first_term
  (S : ℝ)
  (r : ℝ)
  (hS : S = 20)
  (hr : r = -3/7) :
  firstTermOfGeometricSeries S r = 200 / 7 :=
  by
    rw [hS, hr]
    sorry

end find_first_term_l1061_106171


namespace roots_nonpositive_if_ac_le_zero_l1061_106180

theorem roots_nonpositive_if_ac_le_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a * c ≤ 0) :
  ¬ (∀ x : ℝ, x^2 - (b/a)*x + (c/a) = 0 → x > 0) :=
sorry

end roots_nonpositive_if_ac_le_zero_l1061_106180


namespace total_weight_of_10_moles_CaH2_is_420_96_l1061_106109

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008
def molecular_weight_CaH2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_H
def moles_CaH2 : ℝ := 10
def total_weight_CaH2 : ℝ := molecular_weight_CaH2 * moles_CaH2

theorem total_weight_of_10_moles_CaH2_is_420_96 :
  total_weight_CaH2 = 420.96 :=
by
  sorry

end total_weight_of_10_moles_CaH2_is_420_96_l1061_106109


namespace f_g_3_value_l1061_106194

def f (x : ℝ) := x^3 + 1
def g (x : ℝ) := 3 * x + 2

theorem f_g_3_value : f (g 3) = 1332 := by
  sorry

end f_g_3_value_l1061_106194


namespace ratio_of_population_is_correct_l1061_106149

noncomputable def ratio_of_population (M W C : ℝ) : ℝ :=
  (M / (W + C)) * 100

theorem ratio_of_population_is_correct
  (M W C : ℝ) 
  (hW: W = 0.9 * M)
  (hC: C = 0.6 * (M + W)) :
  ratio_of_population M W C = 49.02 := 
by
  sorry

end ratio_of_population_is_correct_l1061_106149


namespace max_marks_l1061_106159

theorem max_marks (M : ℝ) (h : 0.80 * M = 240) : M = 300 :=
sorry

end max_marks_l1061_106159


namespace cube_sum_l1061_106129

theorem cube_sum (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 :=
by
  sorry

end cube_sum_l1061_106129


namespace total_walnut_trees_in_park_l1061_106186

theorem total_walnut_trees_in_park 
  (initial_trees planted_by_first planted_by_second planted_by_third removed_trees : ℕ)
  (h_initial : initial_trees = 22)
  (h_first : planted_by_first = 12)
  (h_second : planted_by_second = 15)
  (h_third : planted_by_third = 10)
  (h_removed : removed_trees = 4) :
  initial_trees + (planted_by_first + planted_by_second + planted_by_third - removed_trees) = 55 :=
by
  sorry

end total_walnut_trees_in_park_l1061_106186


namespace car_travel_distance_20_minutes_l1061_106141

noncomputable def train_speed_in_mph : ℝ := 80
noncomputable def car_speed_ratio : ℝ := 3/4
noncomputable def car_speed_in_mph : ℝ := car_speed_ratio * train_speed_in_mph
noncomputable def travel_time_in_hours : ℝ := 20 / 60
noncomputable def distance_travelled_by_car : ℝ := car_speed_in_mph * travel_time_in_hours

theorem car_travel_distance_20_minutes : distance_travelled_by_car = 20 := 
by 
  sorry

end car_travel_distance_20_minutes_l1061_106141


namespace weight_loss_total_l1061_106111

theorem weight_loss_total :
  ∀ (weight1 weight2 weight3 weight4 : ℕ),
    weight1 = 27 →
    weight2 = weight1 - 7 →
    weight3 = 28 →
    weight4 = 28 →
    weight1 + weight2 + weight3 + weight4 = 103 :=
by
  intros weight1 weight2 weight3 weight4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end weight_loss_total_l1061_106111


namespace find_y_l1061_106162

theorem find_y (x y : ℝ) (h₁ : 1.5 * x = 0.3 * y) (h₂ : x = 20) : y = 100 :=
sorry

end find_y_l1061_106162


namespace factorize_x4_plus_16_l1061_106174

theorem factorize_x4_plus_16 :
  ∀ x : ℝ, (x^4 + 16) = (x^2 - 2 * x + 2) * (x^2 + 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l1061_106174


namespace x_equals_eleven_l1061_106156

theorem x_equals_eleven (x : ℕ) 
  (h : (1 / 8) * 2^36 = 8^x) : x = 11 :=
sorry

end x_equals_eleven_l1061_106156


namespace decimal_sum_sqrt_l1061_106184

theorem decimal_sum_sqrt (a b : ℝ) (h₁ : a = Real.sqrt 5 - 2) (h₂ : b = Real.sqrt 13 - 3) : 
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
  sorry

end decimal_sum_sqrt_l1061_106184


namespace units_digit_sum_of_factorials_50_l1061_106118

def units_digit (n : Nat) : Nat :=
  n % 10

def sum_of_factorials (n : Nat) : Nat :=
  (List.range' 1 n).map Nat.factorial |>.sum

theorem units_digit_sum_of_factorials_50 :
  units_digit (sum_of_factorials 51) = 3 := 
sorry

end units_digit_sum_of_factorials_50_l1061_106118


namespace assembly_line_average_output_l1061_106107

theorem assembly_line_average_output :
  (60 / 90) + (60 / 60) = (5 / 3) →
  60 + 60 = 120 →
  120 / (5 / 3) = 72 :=
by
  intros h1 h2
  -- Proof follows, but we will end with 'sorry' to indicate further proof steps need to be done.
  sorry

end assembly_line_average_output_l1061_106107


namespace polygon_sides_l1061_106133

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l1061_106133


namespace addition_result_l1061_106134

theorem addition_result (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end addition_result_l1061_106134


namespace road_repair_completion_time_l1061_106104

theorem road_repair_completion_time :
  (∀ (r : ℝ), 1 = 45 * r * 3) → (∀ (t : ℝ), (30 * (1 / (3 * 45))) * t = 1) → t = 4.5 :=
by
  intros rate_eq time_eq
  sorry

end road_repair_completion_time_l1061_106104


namespace problem_statement_l1061_106179

theorem problem_statement (x : ℝ) (h : 0 < x) : x + 2016^2016 / x^2016 ≥ 2017 := 
by
  sorry

end problem_statement_l1061_106179


namespace y_increase_by_30_when_x_increases_by_12_l1061_106154

theorem y_increase_by_30_when_x_increases_by_12
  (h : ∀ x y : ℝ, x = 4 → y = 10)
  (x_increase : ℝ := 12) :
  ∃ y_increase : ℝ, y_increase = 30 :=
by
  -- Here we assume the condition h and x_increase
  let ratio := 10 / 4  -- Establish the ratio of increase
  let expected_y_increase := x_increase * ratio
  exact ⟨expected_y_increase, sorry⟩  -- Prove it is 30

end y_increase_by_30_when_x_increases_by_12_l1061_106154


namespace find_pool_length_l1061_106139

noncomputable def pool_length : ℝ :=
  let drain_rate := 60 -- cubic feet per minute
  let width := 40 -- feet
  let depth := 10 -- feet
  let capacity_percent := 0.80
  let drain_time := 800 -- minutes
  let drained_volume := drain_rate * drain_time -- cubic feet
  let full_capacity := drained_volume / capacity_percent -- cubic feet
  let length := full_capacity / (width * depth) -- feet
  length

theorem find_pool_length : pool_length = 150 := by
  sorry

end find_pool_length_l1061_106139


namespace junk_mail_each_house_l1061_106123

def blocks : ℕ := 16
def houses_per_block : ℕ := 17
def total_junk_mail : ℕ := 1088
def total_houses : ℕ := blocks * houses_per_block
def junk_mail_per_house : ℕ := total_junk_mail / total_houses

theorem junk_mail_each_house :
  junk_mail_per_house = 4 :=
by
  sorry

end junk_mail_each_house_l1061_106123


namespace complex_number_quadrant_l1061_106112

theorem complex_number_quadrant (a : ℝ) : 
  (a^2 - 2 = 3 * a - 4) ∧ (a^2 - 2 < 0 ∧ 3 * a - 4 < 0) → a = 1 :=
by
  sorry

end complex_number_quadrant_l1061_106112


namespace original_price_per_pound_l1061_106185

theorem original_price_per_pound (P x : ℝ)
  (h1 : 0.2 * x * P = 0.2 * x)
  (h2 : x * P = x * P)
  (h3 : 1.08 * (0.8 * x) * 1.08 = 1.08 * x * P) :
  P = 1.08 :=
sorry

end original_price_per_pound_l1061_106185


namespace prime_if_and_only_if_digit_is_nine_l1061_106151

theorem prime_if_and_only_if_digit_is_nine (B : ℕ) (h : 0 ≤ B ∧ B < 10) :
  Prime (303200 + B) ↔ B = 9 := 
by
  sorry

end prime_if_and_only_if_digit_is_nine_l1061_106151


namespace perpendicular_lines_condition_l1061_106128

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, x + (m + 1) * y = 2 - m → m * x + 2 * y = -8) ↔ m = -2 / 3 :=
by sorry

end perpendicular_lines_condition_l1061_106128


namespace find_x_l1061_106114

theorem find_x (x : ℝ) (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3) * (0 : ℂ).im) : x = 3 :=
sorry

end find_x_l1061_106114


namespace scientific_notation_of_0_0000003_l1061_106137

theorem scientific_notation_of_0_0000003 : 0.0000003 = 3 * 10^(-7) := by
  sorry

end scientific_notation_of_0_0000003_l1061_106137


namespace sampling_methods_used_l1061_106147

-- Definitions based on problem conditions
def TotalHouseholds : Nat := 2000
def FarmerHouseholds : Nat := 1800
def WorkerHouseholds : Nat := 100
def IntellectualHouseholds : Nat := TotalHouseholds - FarmerHouseholds - WorkerHouseholds
def SampleSize : Nat := 40

-- The statement of the proof problem
theorem sampling_methods_used
  (N : Nat := TotalHouseholds)
  (F : Nat := FarmerHouseholds)
  (W : Nat := WorkerHouseholds)
  (I : Nat := IntellectualHouseholds)
  (S : Nat := SampleSize)
:
  (1 ∈ [1, 2, 3]) ∧ (2 ∈ [1, 2, 3]) ∧ (3 ∈ [1, 2, 3]) :=
by
  -- Add the proof here
  sorry

end sampling_methods_used_l1061_106147


namespace neg_ex_proposition_l1061_106191

open Classical

theorem neg_ex_proposition :
  ¬ (∃ n : ℕ, n^2 > 2^n) ↔ ∀ n : ℕ, n^2 ≤ 2^n :=
by sorry

end neg_ex_proposition_l1061_106191


namespace find_m_l1061_106158

-- Definitions based on conditions
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def are_roots_of_quadratic (b c m : ℝ) : Prop :=
  b * c = 6 - m ∧ b + c = -(m + 2)

-- The theorem statement
theorem find_m {a b c m : ℝ} (h₁ : a = 5) (h₂ : is_isosceles_triangle a b c) (h₃ : are_roots_of_quadratic b c m) : m = -10 :=
sorry

end find_m_l1061_106158


namespace parallel_line_distance_equation_l1061_106182

theorem parallel_line_distance_equation :
  ∃ m : ℝ, (m = -20 ∨ m = 32) ∧
  ∀ x y : ℝ, (5 * x - 12 * y + 6 = 0) → 
            (5 * x - 12 * y + m = 0) :=
by
  sorry

end parallel_line_distance_equation_l1061_106182


namespace additional_people_needed_to_mow_lawn_l1061_106144

theorem additional_people_needed_to_mow_lawn :
  (∀ (k : ℕ), (∀ (n t : ℕ), n * t = k) → (4 * 6 = k) → (∃ (n : ℕ), n * 3 = k) → (8 - 4 = 4)) :=
by sorry

end additional_people_needed_to_mow_lawn_l1061_106144


namespace don_travel_time_to_hospital_l1061_106197

noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def time_to_travel (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem don_travel_time_to_hospital :
  let speed_mary := 60
  let speed_don := 30
  let time_mary_minutes := 15
  let time_mary_hours := time_mary_minutes / 60
  let distance := distance_traveled speed_mary time_mary_hours
  let time_don_hours := time_to_travel distance speed_don
  time_don_hours * 60 = 30 :=
by
  sorry

end don_travel_time_to_hospital_l1061_106197


namespace vehicle_speed_l1061_106152

theorem vehicle_speed (distance : ℝ) (time : ℝ) (h_dist : distance = 150) (h_time : time = 0.75) : distance / time = 200 :=
  by
    sorry

end vehicle_speed_l1061_106152


namespace temperature_difference_l1061_106124

def Shanghai_temp : ℤ := 3
def Beijing_temp : ℤ := -5

theorem temperature_difference :
  Shanghai_temp - Beijing_temp = 8 := by
  sorry

end temperature_difference_l1061_106124


namespace jim_gas_gallons_l1061_106138

theorem jim_gas_gallons (G : ℕ) (C_NC C_VA : ℕ → ℕ) 
  (h₁ : ∀ G, C_NC G = 2 * G)
  (h₂ : ∀ G, C_VA G = 3 * G)
  (h₃ : C_NC G + C_VA G = 50) :
  G = 10 := 
sorry

end jim_gas_gallons_l1061_106138


namespace man_speed_against_current_l1061_106132

-- Definitions for the problem conditions
def man_speed_with_current : ℝ := 21
def current_speed : ℝ := 4.3

-- Main proof statement
theorem man_speed_against_current : man_speed_with_current - 2 * current_speed = 12.4 :=
by
  sorry

end man_speed_against_current_l1061_106132


namespace smallest_pieces_to_remove_l1061_106150

theorem smallest_pieces_to_remove 
  (total_fruit : ℕ)
  (friends : ℕ)
  (h_fruit : total_fruit = 30)
  (h_friends : friends = 4) 
  : ∃ k : ℕ, k = 2 ∧ ((total_fruit - k) % friends = 0) :=
sorry

end smallest_pieces_to_remove_l1061_106150


namespace trigonometric_identity_l1061_106167

noncomputable def point_on_terminal_side (x y : ℝ) : Prop :=
    ∃ α : ℝ, x = Real.cos α ∧ y = Real.sin α

theorem trigonometric_identity (x y : ℝ) (h : point_on_terminal_side 1 3) :
    (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by
  sorry

end trigonometric_identity_l1061_106167


namespace fourth_student_seat_number_l1061_106106

theorem fourth_student_seat_number (n : ℕ) (pop_size sample_size : ℕ)
  (s1 s2 s3 : ℕ)
  (h_pop_size : pop_size = 52)
  (h_sample_size : sample_size = 4)
  (h_6_in_sample : s1 = 6)
  (h_32_in_sample : s2 = 32)
  (h_45_in_sample : s3 = 45)
  : ∃ s4 : ℕ, s4 = 19 :=
by
  sorry

end fourth_student_seat_number_l1061_106106


namespace simplify_expr_l1061_106155

variable (a b c : ℤ)

theorem simplify_expr :
  (15 * a + 45 * b + 20 * c) + (25 * a - 35 * b - 10 * c) - (10 * a + 55 * b + 30 * c) = 30 * a - 45 * b - 20 * c := 
by
  sorry

end simplify_expr_l1061_106155


namespace base7_to_base10_l1061_106196

open Nat

theorem base7_to_base10 : (3 * 7^2 + 5 * 7^1 + 1 * 7^0 = 183) :=
by
  sorry

end base7_to_base10_l1061_106196


namespace BANANA_permutations_l1061_106136

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l1061_106136


namespace cross_section_area_l1061_106113

open Real

theorem cross_section_area (b α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ (area : ℝ), area = - (b^2 * cos α * tan β) / (2 * cos (3 * α)) :=
by
  sorry

end cross_section_area_l1061_106113


namespace find_divisor_l1061_106146

-- Define the conditions
def dividend : ℕ := 22
def quotient : ℕ := 7
def remainder : ℕ := 1

-- The divisor is what we need to find
def divisor : ℕ := 3

-- The proof problem: proving that the given conditions imply the divisor is 3
theorem find_divisor :
  ∃ d : ℕ, dividend = d * quotient + remainder ∧ d = divisor :=
by
  use 3
  -- Replace actual proof with sorry for now
  sorry

end find_divisor_l1061_106146


namespace inequality_proof_l1061_106168

theorem inequality_proof (p : ℝ) (x y z v : ℝ) (hp : p ≥ 2) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y) ^ p + (z + v) ^ p + (x + z) ^ p + (y + v) ^ p ≤ x ^ p + y ^ p + z ^ p + v ^ p + (x + y + z + v) ^ p := 
by sorry

end inequality_proof_l1061_106168


namespace find_c_value_l1061_106189

theorem find_c_value 
  (b : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + b * x + 3 ≥ 0) 
  (h2 : ∀ m c : ℝ, (∀ x : ℝ, x^2 + b * x + 3 < c ↔ m - 8 < x ∧ x < m)) 
  : c = 16 :=
sorry

end find_c_value_l1061_106189


namespace square_side_length_l1061_106145

noncomputable def side_length_square_inscribed_in_hexagon : ℝ :=
  50 * Real.sqrt 3

theorem square_side_length (a b: ℝ) (h1 : a = 50) (h2 : b = 50 * (2 - Real.sqrt 3)) 
(s1 s2 s3 s4 s5 s6: ℝ) (ha : s1 = s2) (hb : s2 = s3) (hc : s3 = s4) 
(hd : s4 = s5) (he : s5 = s6) (hf : s6 = s1) : side_length_square_inscribed_in_hexagon = 50 * Real.sqrt 3 :=
by
  sorry

end square_side_length_l1061_106145


namespace complex_square_eq_l1061_106169

theorem complex_square_eq (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I) : 
  a + b * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end complex_square_eq_l1061_106169


namespace minimum_value_of_function_l1061_106153

theorem minimum_value_of_function (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  ∃ y : ℝ, (∀ z : ℝ, z = (1 / x) + (4 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end minimum_value_of_function_l1061_106153


namespace oranges_left_uneaten_l1061_106116

variable (total_oranges : ℕ)
variable (half_oranges ripe_oranges unripe_oranges eaten_ripe_oranges eaten_unripe_oranges uneaten_ripe_oranges uneaten_unripe_oranges total_uneaten_oranges : ℕ)

axiom h1 : total_oranges = 96
axiom h2 : half_oranges = total_oranges / 2
axiom h3 : ripe_oranges = half_oranges
axiom h4 : unripe_oranges = half_oranges
axiom h5 : eaten_ripe_oranges = ripe_oranges / 4
axiom h6 : eaten_unripe_oranges = unripe_oranges / 8
axiom h7 : uneaten_ripe_oranges = ripe_oranges - eaten_ripe_oranges
axiom h8 : uneaten_unripe_oranges = unripe_oranges - eaten_unripe_oranges
axiom h9 : total_uneaten_oranges = uneaten_ripe_oranges + uneaten_unripe_oranges

theorem oranges_left_uneaten : total_uneaten_oranges = 78 := by
  sorry

end oranges_left_uneaten_l1061_106116


namespace range_of_m_l1061_106160

-- Definitions based on the conditions
def p (m : ℝ) : Prop := 4 - 4 * m > 0
def q (m : ℝ) : Prop := m + 2 > 0

-- Problem statement in Lean 4
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≤ -2 ∨ m ≥ 1 := by
  sorry

end range_of_m_l1061_106160


namespace distinct_integers_problem_l1061_106117

variable (a b c d e : ℤ)

theorem distinct_integers_problem
  (h1 : a ≠ b) 
  (h2 : a ≠ c) 
  (h3 : a ≠ d) 
  (h4 : a ≠ e) 
  (h5 : b ≠ c) 
  (h6 : b ≠ d) 
  (h7 : b ≠ e) 
  (h8 : c ≠ d) 
  (h9 : c ≠ e) 
  (h10 : d ≠ e) 
  (h_prod : (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12) : 
  a + b + c + d + e = 17 := 
sorry

end distinct_integers_problem_l1061_106117


namespace sum_of_numbers_eq_l1061_106190

theorem sum_of_numbers_eq (a b : ℕ) (h1 : a = 64) (h2 : b = 32) (h3 : a = 2 * b) : a + b = 96 := 
by 
  sorry

end sum_of_numbers_eq_l1061_106190


namespace calvin_haircut_goal_percentage_l1061_106192

theorem calvin_haircut_goal_percentage :
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  (completed_haircuts / total_haircuts_needed) * 100 = 80 :=
by
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  show (completed_haircuts / total_haircuts_needed) * 100 = 80
  sorry

end calvin_haircut_goal_percentage_l1061_106192


namespace number_of_marked_points_l1061_106163

theorem number_of_marked_points (S S' : ℤ) (n : ℤ) 
  (h1 : S = 25) 
  (h2 : S' = S - 5 * n) 
  (h3 : S' = -35) : 
  n = 12 := 
  sorry

end number_of_marked_points_l1061_106163


namespace train_speed_l1061_106170

theorem train_speed (train_length : ℕ) (cross_time : ℕ) (speed : ℕ) 
  (h_train_length : train_length = 300)
  (h_cross_time : cross_time = 10)
  (h_speed_eq : speed = train_length / cross_time) : 
  speed = 30 :=
by
  sorry

end train_speed_l1061_106170


namespace farmer_feed_total_cost_l1061_106105

/-- 
A farmer spent $35 on feed for chickens and goats. He spent 40% of the money on chicken feed, which he bought at a 50% discount off the full price, and spent the rest on goat feed, which he bought at full price. Prove that if the farmer had paid full price for both the chicken feed and the goat feed, he would have spent $49.
-/
theorem farmer_feed_total_cost
  (total_spent : ℝ := 35)
  (chicken_feed_fraction : ℝ := 0.40)
  (goat_feed_fraction : ℝ := 0.60)
  (discount : ℝ := 0.50)
  (chicken_feed_discounted : ℝ := chicken_feed_fraction * total_spent)
  (chicken_feed_full_price : ℝ := chicken_feed_discounted / (1 - discount))
  (goat_feed_full_price : ℝ := goat_feed_fraction * total_spent):
  chicken_feed_full_price + goat_feed_full_price = 49 := 
sorry

end farmer_feed_total_cost_l1061_106105


namespace initial_apples_l1061_106148

theorem initial_apples (picked: ℕ) (newly_grown: ℕ) (still_on_tree: ℕ) (initial: ℕ):
  (picked = 7) →
  (newly_grown = 2) →
  (still_on_tree = 6) →
  (still_on_tree + picked - newly_grown = initial) →
  initial = 11 :=
by
  intros hpicked hnewly_grown hstill_on_tree hcalculation
  sorry

end initial_apples_l1061_106148


namespace surface_area_of_circumscribed_sphere_of_triangular_pyramid_l1061_106130

theorem surface_area_of_circumscribed_sphere_of_triangular_pyramid
  (a : ℝ)
  (h₁ : a > 0) : 
  ∃ S, S = (27 * π / 32 * a^2) := 
by
  sorry

end surface_area_of_circumscribed_sphere_of_triangular_pyramid_l1061_106130


namespace pastor_prayer_ratio_l1061_106187

theorem pastor_prayer_ratio 
  (R : ℚ) 
  (paul_prays_per_day : ℚ := 20)
  (paul_sunday_times : ℚ := 2 * paul_prays_per_day)
  (paul_total : ℚ := 6 * paul_prays_per_day + paul_sunday_times)
  (bruce_ratio : ℚ := R)
  (bruce_prays_per_day : ℚ := bruce_ratio * paul_prays_per_day)
  (bruce_sunday_times : ℚ := 2 * paul_sunday_times)
  (bruce_total : ℚ := 6 * bruce_prays_per_day + bruce_sunday_times)
  (condition : paul_total = bruce_total + 20) :
  R = 1/2 :=
sorry

end pastor_prayer_ratio_l1061_106187


namespace train_number_of_cars_l1061_106198

theorem train_number_of_cars (lena_cars : ℕ) (time_counted : ℕ) (total_time : ℕ) 
  (cars_in_train : ℕ)
  (h1 : lena_cars = 8) 
  (h2 : time_counted = 15)
  (h3 : total_time = 210)
  (h4 : (8 / 15 : ℚ) * 210 = 112)
  : cars_in_train = 112 :=
sorry

end train_number_of_cars_l1061_106198


namespace spherical_to_rectangular_coordinates_l1061_106122

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 6 → θ = 7 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (3, -3, 3 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_coordinates_l1061_106122


namespace ticket_representation_l1061_106120

-- Define a structure for representing a movie ticket
structure Ticket where
  rows : Nat
  seats : Nat

-- Define the specific instance of representing 7 rows and 5 seats
def ticket_7_5 : Ticket := ⟨7, 5⟩

-- The theorem stating our problem: the representation of 7 rows and 5 seats is (7,5)
theorem ticket_representation : ticket_7_5 = ⟨7, 5⟩ :=
  by
    -- Proof goes here (omitted as per instructions)
    sorry

end ticket_representation_l1061_106120


namespace initial_quarters_l1061_106195

-- Define the conditions
def quartersAfterLoss (x : ℕ) : ℕ := (4 * x) / 3
def quartersAfterThirdYear (x : ℕ) : ℕ := x - 4
def quartersAfterSecondYear (x : ℕ) : ℕ := x - 36
def quartersAfterFirstYear (x : ℕ) : ℕ := x * 2

-- The main theorem
theorem initial_quarters (x : ℕ) (h1 : quartersAfterLoss x = 140)
    (h2 : quartersAfterThirdYear 140 = 136)
    (h3 : quartersAfterSecondYear 136 = 100)
    (h4 : quartersAfterFirstYear 50 = 100) :
  x = 50 := by
  simp [quartersAfterFirstYear, quartersAfterSecondYear,
        quartersAfterThirdYear, quartersAfterLoss] at *
  sorry

end initial_quarters_l1061_106195


namespace number_of_pickup_trucks_l1061_106161

theorem number_of_pickup_trucks 
  (cars : ℕ) (bicycles : ℕ) (tricycles : ℕ) (total_tires : ℕ)
  (tires_per_car : ℕ) (tires_per_bicycle : ℕ) (tires_per_tricycle : ℕ) (tires_per_pickup : ℕ) :
  cars = 15 →
  bicycles = 3 →
  tricycles = 1 →
  total_tires = 101 →
  tires_per_car = 4 →
  tires_per_bicycle = 2 →
  tires_per_tricycle = 3 →
  tires_per_pickup = 4 →
  ((total_tires - (cars * tires_per_car + bicycles * tires_per_bicycle + tricycles * tires_per_tricycle)) / tires_per_pickup) = 8 :=
by
  sorry

end number_of_pickup_trucks_l1061_106161


namespace big_eighteen_basketball_games_count_l1061_106140

def num_teams_in_division := 6
def num_teams := 18
def games_within_division := 3
def games_between_divisions := 1
def divisions := 3

theorem big_eighteen_basketball_games_count :
  (num_teams * ((num_teams_in_division - 1) * games_within_division + (num_teams - num_teams_in_division) * games_between_divisions)) / 2 = 243 :=
by
  have teams_in_other_divisions : num_teams - num_teams_in_division = 12 := rfl
  have games_per_team_within_division : (num_teams_in_division - 1) * games_within_division = 15 := rfl
  have games_per_team_between_division : 12 * games_between_divisions = 12 := rfl
  sorry

end big_eighteen_basketball_games_count_l1061_106140


namespace min_value_abs_function_l1061_106178

theorem min_value_abs_function : ∀ (x : ℝ), (|x + 1| + |2 - x|) ≥ 3 :=
by
  sorry

end min_value_abs_function_l1061_106178


namespace find_counterfeit_coins_l1061_106165

structure Coins :=
  (a a₁ b b₁ c c₁ : ℝ)
  (genuine_weight : ℝ)
  (counterfeit_weight : ℝ)
  (a_is_genuine_or_counterfeit : a = genuine_weight ∨ a = counterfeit_weight)
  (a₁_is_genuine_or_counterfeit : a₁ = genuine_weight ∨ a₁ = counterfeit_weight)
  (b_is_genuine_or_counterfeit : b = genuine_weight ∨ b = counterfeit_weight)
  (b₁_is_genuine_or_counterfeit : b₁ = genuine_weight ∨ b₁ = counterfeit_weight)
  (c_is_genuine_or_counterfeit : c = genuine_weight ∨ c = counterfeit_weight)
  (c₁_is_genuine_or_counterfeit : c₁ = genuine_weight ∨ c₁ = counterfeit_weight)
  (counterfeit_pair_ends_unit_segment : (a = counterfeit_weight ∧ a₁ = counterfeit_weight) 
                                        ∨ (b = counterfeit_weight ∧ b₁ = counterfeit_weight)
                                        ∨ (c = counterfeit_weight ∧ c₁ = counterfeit_weight))

theorem find_counterfeit_coins (coins : Coins) : 
  (coins.a = coins.genuine_weight ∧ coins.b = coins.genuine_weight → coins.a₁ = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.a < coins.b → coins.a = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.b < coins.a → coins.b = coins.counterfeit_weight ∧ coins.a₁ = coins.counterfeit_weight) := 
by
  sorry

end find_counterfeit_coins_l1061_106165


namespace soccer_ball_selling_price_l1061_106108

theorem soccer_ball_selling_price
  (cost_price_per_ball : ℕ)
  (num_balls : ℕ)
  (total_profit : ℕ)
  (h_cost_price : cost_price_per_ball = 60)
  (h_num_balls : num_balls = 50)
  (h_total_profit : total_profit = 1950) :
  (cost_price_per_ball + (total_profit / num_balls) = 99) :=
by 
  -- Note: Proof can be filled here
  sorry

end soccer_ball_selling_price_l1061_106108


namespace final_quantity_of_milk_l1061_106183

-- Define initial conditions
def initial_volume : ℝ := 60
def removed_volume : ℝ := 9

-- Given the initial conditions, calculate the quantity of milk left after two dilutions
theorem final_quantity_of_milk :
  let first_removal_ratio := initial_volume - removed_volume / initial_volume
  let first_milk_volume := initial_volume * (first_removal_ratio)
  let second_removal_ratio := first_milk_volume / initial_volume
  let second_milk_volume := first_milk_volume * (second_removal_ratio)
  second_milk_volume = 43.35 :=
by
  sorry

end final_quantity_of_milk_l1061_106183


namespace guest_bedroom_ratio_l1061_106131

theorem guest_bedroom_ratio 
  (lr_dr_kitchen : ℝ) (total_house : ℝ) (master_bedroom : ℝ) (guest_bedroom : ℝ) 
  (h1 : lr_dr_kitchen = 1000) 
  (h2 : total_house = 2300)
  (h3 : master_bedroom = 1040)
  (h4 : guest_bedroom = total_house - (lr_dr_kitchen + master_bedroom)) :
  guest_bedroom / master_bedroom = 1 / 4 := 
by
  sorry

end guest_bedroom_ratio_l1061_106131


namespace digit_d_multiple_of_9_l1061_106115

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end digit_d_multiple_of_9_l1061_106115


namespace remaining_money_correct_l1061_106176

open Nat

def initial_money : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def remaining_money : ℕ := initial_money - total_spent

theorem remaining_money_correct : remaining_money = 78 := by
  -- Proof goes here
  sorry

end remaining_money_correct_l1061_106176


namespace cos_C_value_l1061_106101

-- Definitions for the perimeter and sine ratios
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (perimeter : ℝ) (sin_ratio_A sin_ratio_B sin_ratio_C : ℚ)

-- Given conditions
axiom perimeter_condition : perimeter = a + b + c
axiom sine_ratio_condition : (sin_ratio_A / sin_ratio_B / sin_ratio_C) = (3 / 2 / 4)
axiom side_lengths : a = 3 ∧ b = 2 ∧ c = 4

-- To prove

theorem cos_C_value (h1 : sine_ratio_A = 3) (h2 : sine_ratio_B = 2) (h3 : sin_ratio_C = 4) :
  (3^2 + 2^2 - 4^2) / (2 * 3 * 2) = -1 / 4 :=
sorry

end cos_C_value_l1061_106101


namespace jerry_total_cost_correct_l1061_106135

theorem jerry_total_cost_correct :
  let bw_cost := 27
  let bw_discount := 0.1 * bw_cost
  let bw_discounted_price := bw_cost - bw_discount
  let color_cost := 32
  let color_discount := 0.05 * color_cost
  let color_discounted_price := color_cost - color_discount
  let total_color_discounted_price := 3 * color_discounted_price
  let total_discounted_price_before_tax := bw_discounted_price + total_color_discounted_price
  let tax_rate := 0.07
  let tax := total_discounted_price_before_tax * tax_rate
  let total_cost := total_discounted_price_before_tax + tax
  (Float.round (total_cost * 100) / 100) = 123.59 :=
sorry

end jerry_total_cost_correct_l1061_106135


namespace Luke_spent_per_week_l1061_106102

-- Definitions based on the conditions
def money_from_mowing := 9
def money_from_weeding := 18
def total_money := money_from_mowing + money_from_weeding
def weeks := 9
def amount_spent_per_week := total_money / weeks

-- The proof statement
theorem Luke_spent_per_week :
  amount_spent_per_week = 3 := 
  sorry

end Luke_spent_per_week_l1061_106102


namespace mean_equality_l1061_106175

theorem mean_equality (z : ℚ) :
  ((8 + 7 + 28) / 3 : ℚ) = (14 + z) / 2 → z = 44 / 3 :=
by
  sorry

end mean_equality_l1061_106175


namespace atomic_number_order_l1061_106199

-- Define that elements A, B, C, D, and E are in the same period
variable (A B C D E : Type)

-- Define conditions based on the problem
def highest_valence_oxide_basic (x : Type) : Prop := sorry
def basicity_greater (x y : Type) : Prop := sorry
def gaseous_hydride_stability (x y : Type) : Prop := sorry
def smallest_ionic_radius (x : Type) : Prop := sorry

-- Assume conditions given in the problem
axiom basic_oxides : highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
axiom basicity_order : basicity_greater B A
axiom hydride_stabilities : gaseous_hydride_stability C D
axiom smallest_radius : smallest_ionic_radius E

-- Prove that the order of atomic numbers from smallest to largest is B, A, E, D, C
theorem atomic_number_order :
  ∃ (A B C D E : Type), highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
  ∧ basicity_greater B A ∧ gaseous_hydride_stability C D ∧ smallest_ionic_radius E
  ↔ B = B ∧ A = A ∧ E = E ∧ D = D ∧ C = C := sorry

end atomic_number_order_l1061_106199


namespace remainder_is_v_l1061_106121

theorem remainder_is_v (x y u v : ℤ) (hx : x > 0) (hy : y > 0)
  (hdiv : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + (2 * u + 1) * y) % y = v :=
by
  sorry

end remainder_is_v_l1061_106121


namespace initial_discount_percentage_l1061_106103

variable (d : ℝ) (x : ℝ)
variable (h1 : 0 < d) (h2 : 0 ≤ x) (h3 : x ≤ 100)
variable (h4 : (1 - x / 100) * 0.6 * d = 0.33 * d)

theorem initial_discount_percentage : x = 45 :=
by
  sorry

end initial_discount_percentage_l1061_106103


namespace seq_inequality_l1061_106173

variable (a : ℕ → ℝ)
variable (n m : ℕ)

-- Conditions
axiom pos_seq (k : ℕ) : a k ≥ 0
axiom add_condition (i j : ℕ) : a (i + j) ≤ a i + a j

-- Statement to prove
theorem seq_inequality (n m : ℕ) (h : m > 0) (h' : n ≥ m) : 
  a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := sorry

end seq_inequality_l1061_106173


namespace exists_distinct_numbers_divisible_by_3_l1061_106119

-- Define the problem in Lean with the given conditions and goal.
theorem exists_distinct_numbers_divisible_by_3 : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧ d % 3 = 0 ∧
  (a + b + c) % d = 0 ∧ (a + b + d) % c = 0 ∧ (a + c + d) % b = 0 ∧ (b + c + d) % a = 0 :=
by
  sorry

end exists_distinct_numbers_divisible_by_3_l1061_106119


namespace cost_of_10_apples_l1061_106188

-- Define the price for 10 apples as a variable
noncomputable def price_10_apples (P : ℝ) : ℝ := P

-- Theorem stating that the cost for 10 apples is the provided price
theorem cost_of_10_apples (P : ℝ) : price_10_apples P = P :=
  by
    sorry

end cost_of_10_apples_l1061_106188


namespace arithmetic_sequence_common_difference_l1061_106110

theorem arithmetic_sequence_common_difference
    (a : ℕ → ℝ)
    (h1 : a 2 + a 3 = 9)
    (h2 : a 4 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n + d) : d = 3 :=
        sorry

end arithmetic_sequence_common_difference_l1061_106110


namespace minimum_value_l1061_106166

variable (a b : ℝ)

-- Assume a and b are positive real numbers
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)

-- Given the condition a + b = 2
variable (h₂ : a + b = 2)

theorem minimum_value : (1 / a) + (2 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end minimum_value_l1061_106166


namespace triangle_inequality_l1061_106164

theorem triangle_inequality (a : ℝ) (h1 : a + 3 > 5) (h2 : a + 5 > 3) (h3 : 3 + 5 > a) :
  2 < a ∧ a < 8 :=
by {
  sorry
}

end triangle_inequality_l1061_106164


namespace equation_for_number_l1061_106142

variable (a : ℤ)

theorem equation_for_number : 3 * a + 5 = 9 :=
sorry

end equation_for_number_l1061_106142


namespace range_of_x_inequality_l1061_106177

theorem range_of_x_inequality (a : ℝ) (x : ℝ)
  (h : -1 ≤ a ∧ a ≤ 1) : 
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end range_of_x_inequality_l1061_106177


namespace three_x_plus_y_eq_zero_l1061_106100

theorem three_x_plus_y_eq_zero (x y : ℝ) (h : (2 * x + y) ^ 3 + x ^ 3 + 3 * x + y = 0) : 3 * x + y = 0 :=
sorry

end three_x_plus_y_eq_zero_l1061_106100


namespace max_n_l1061_106172

noncomputable def prod := 160 * 170 * 180 * 190

theorem max_n : ∃ n : ℕ, n = 30499 ∧ n^2 ≤ prod := by
  sorry

end max_n_l1061_106172


namespace max_length_PQ_l1061_106126

-- Define the curve in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Definition of points P and Q lying on the curve
def point_on_curve (ρ θ : ℝ) (P : ℝ × ℝ) : Prop :=
  curve ρ θ ∧ P = (ρ * Real.cos θ, ρ * Real.sin θ)

def points_on_curve (P Q : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ ρ₁ ρ₂, point_on_curve ρ₁ θ₁ P ∧ point_on_curve ρ₂ θ₂ Q

-- The theorem stating the maximum length of PQ
theorem max_length_PQ {P Q : ℝ × ℝ} (h : points_on_curve P Q) : dist P Q ≤ 4 :=
sorry

end max_length_PQ_l1061_106126


namespace share_of_a_l1061_106143

def shares_sum (a b c : ℝ) := a + b + c = 366
def share_a (a b c : ℝ) := a = 1/2 * (b + c)
def share_b (a b c : ℝ) := b = 2/3 * (a + c)

theorem share_of_a (a b c : ℝ) 
  (h1 : shares_sum a b c) 
  (h2 : share_a a b c) 
  (h3 : share_b a b c) : 
  a = 122 := 
by 
  -- Proof goes here
  sorry

end share_of_a_l1061_106143


namespace total_pies_sold_l1061_106127

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end total_pies_sold_l1061_106127


namespace solve_sqrt_equation_l1061_106157

theorem solve_sqrt_equation :
  ∀ (x : ℝ), (3 * Real.sqrt x + 3 * x⁻¹/2 = 7) →
  (x = (49 + 14 * Real.sqrt 13 + 13) / 36 ∨ x = (49 - 14 * Real.sqrt 13 + 13) / 36) :=
by
  intro x hx
  sorry

end solve_sqrt_equation_l1061_106157


namespace remainder_17_pow_1499_mod_23_l1061_106125

theorem remainder_17_pow_1499_mod_23 : (17 ^ 1499) % 23 = 11 :=
by
  sorry

end remainder_17_pow_1499_mod_23_l1061_106125


namespace radius_of_scrap_cookie_l1061_106181

theorem radius_of_scrap_cookie :
  ∀ (r : ℝ),
    (∃ (r_dough r_cookie : ℝ),
      r_dough = 6 ∧  -- Radius of the large dough
      r_cookie = 2 ∧  -- Radius of each cookie
      8 * (π * r_cookie^2) ≤ π * r_dough^2 ∧  -- Total area of cookies is less than or equal to area of large dough
      (π * r_dough^2) - (8 * (π * r_cookie^2)) = π * r^2  -- Area of scrap dough forms a circle of radius r
    ) → r = 2 := by
  sorry

end radius_of_scrap_cookie_l1061_106181
