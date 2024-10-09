import Mathlib

namespace route_C_is_quicker_l589_58970

/-
  Define the conditions based on the problem:
  - Route C: 8 miles at 40 mph.
  - Route D: 5 miles at 35 mph and 2 miles at 25 mph with an additional 3 minutes stop.
-/

def time_route_C : ℚ := (8 : ℚ) / (40 : ℚ) * 60  -- in minutes

def time_route_D : ℚ := ((5 : ℚ) / (35 : ℚ) * 60) + ((2 : ℚ) / (25 : ℚ) * 60) + 3  -- in minutes

def time_difference : ℚ := time_route_D - time_route_C  -- difference in minutes

theorem route_C_is_quicker : time_difference = 4.37 := 
by 
  sorry

end route_C_is_quicker_l589_58970


namespace john_total_spent_l589_58902

noncomputable def computer_cost : ℝ := 1500
noncomputable def peripherals_cost : ℝ := (1 / 4) * computer_cost
noncomputable def base_video_card_cost : ℝ := 300
noncomputable def upgraded_video_card_cost : ℝ := 2.5 * base_video_card_cost
noncomputable def discount_on_video_card : ℝ := 0.12 * upgraded_video_card_cost
noncomputable def video_card_cost_after_discount : ℝ := upgraded_video_card_cost - discount_on_video_card
noncomputable def sales_tax_on_peripherals : ℝ := 0.05 * peripherals_cost
noncomputable def total_spent : ℝ := computer_cost + peripherals_cost + video_card_cost_after_discount + sales_tax_on_peripherals

theorem john_total_spent : total_spent = 2553.75 := by
  sorry

end john_total_spent_l589_58902


namespace inequality_proof_l589_58996

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) +
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) +
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 := 
by
  sorry

end inequality_proof_l589_58996


namespace area_of_rhombus_l589_58983

-- Defining the conditions
def diagonal1 : ℝ := 20
def diagonal2 : ℝ := 30

-- Proving the area of the rhombus
theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = diagonal1) (h2 : d2 = diagonal2) : 
  (d1 * d2 / 2) = 300 := by
  sorry

end area_of_rhombus_l589_58983


namespace find_sachin_age_l589_58913

-- Define Sachin's and Rahul's ages as variables
variables (S R : ℝ)

-- Define the conditions
def rahul_age := S + 9
def age_ratio := (S / R) = (7 / 9)

-- State the theorem for Sachin's age
theorem find_sachin_age (h1 : R = rahul_age S) (h2 : age_ratio S R) : S = 31.5 :=
by sorry

end find_sachin_age_l589_58913


namespace geometric_sequence_values_l589_58982

theorem geometric_sequence_values (l a b c : ℝ) (h : ∃ r : ℝ, a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : b = -3 ∧ a * c = 9 :=
by
  sorry

end geometric_sequence_values_l589_58982


namespace problem_l589_58958

theorem problem (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := 
by sorry

end problem_l589_58958


namespace problem_solution_l589_58969

theorem problem_solution (a b c : ℝ) (h : b^2 = a * c) :
  (a^2 * b^2 * c^2 / (a^3 + b^3 + c^3)) * (1 / a^3 + 1 / b^3 + 1 / c^3) = 1 :=
  by sorry

end problem_solution_l589_58969


namespace parallel_lines_sufficient_not_necessary_l589_58923

theorem parallel_lines_sufficient_not_necessary (a : ℝ) :
  ((a = 3) → (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) → (3 * x + (a - 1) * y - 2 = 0)) ∧ 
  (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) ∧ (3 * x + (a - 1) * y - 2 = 0) → (a = 3 ∨ a = -2))) :=
sorry

end parallel_lines_sufficient_not_necessary_l589_58923


namespace log_expression_equals_four_l589_58990

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l589_58990


namespace union_of_M_and_N_l589_58944

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N :
  M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l589_58944


namespace scientific_notation_347000_l589_58919

theorem scientific_notation_347000 :
  347000 = 3.47 * 10^5 :=
by 
  -- Proof will go here
  sorry

end scientific_notation_347000_l589_58919


namespace slope_of_line_l589_58907

variable (x y : ℝ)

def line_equation : Prop := 4 * y = -5 * x + 8

theorem slope_of_line (h : line_equation x y) :
  ∃ m b, y = m * x + b ∧ m = -5/4 :=
by
  sorry

end slope_of_line_l589_58907


namespace chemical_reaction_produces_l589_58962

def balanced_equation : Prop :=
  ∀ {CaCO3 HCl CaCl2 CO2 H2O : ℕ},
    (CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O)

def calculate_final_products (initial_CaCO3 initial_HCl final_CaCl2 final_CO2 final_H2O remaining_HCl : ℕ) : Prop :=
  balanced_equation ∧
  initial_CaCO3 = 3 ∧
  initial_HCl = 8 ∧
  final_CaCl2 = 3 ∧
  final_CO2 = 3 ∧
  final_H2O = 3 ∧
  remaining_HCl = 2

theorem chemical_reaction_produces :
  calculate_final_products 3 8 3 3 3 2 :=
by sorry

end chemical_reaction_produces_l589_58962


namespace transport_equivalence_l589_58921

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end transport_equivalence_l589_58921


namespace correct_statements_l589_58984

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

noncomputable def a_n_sequence (n : ℕ) := a n
noncomputable def Sn_sum (n : ℕ) := S n

axiom Sn_2022_lt_zero : S 2022 < 0
axiom Sn_2023_gt_zero : S 2023 > 0

theorem correct_statements :
  (a 1012 > 0) ∧ ( ∀ n, S n >= S 1011 → n = 1011) :=
  sorry

end correct_statements_l589_58984


namespace smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l589_58929

noncomputable def f (x : Real) : Real :=
  Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, ( ∀ x, f (x + T') = f x) → T ≤ T') := by
  sorry

theorem f_ge_negative_sqrt_3_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), f x ≥ -Real.sqrt 3 := by
  sorry

end smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l589_58929


namespace solve_equation_l589_58922

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 :=
sorry

end solve_equation_l589_58922


namespace weight_of_b_l589_58946

theorem weight_of_b (A B C : ℕ) 
  (h1 : A + B + C = 129) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 37 := 
by 
  sorry

end weight_of_b_l589_58946


namespace result_is_0_85_l589_58954

noncomputable def calc_expression := 1.85 - 1.85 / 1.85

theorem result_is_0_85 : calc_expression = 0.85 :=
by 
  sorry

end result_is_0_85_l589_58954


namespace smallest_m_l589_58927

-- Let n be a positive integer and r be a positive real number less than 1/5000
def valid_r (r : ℝ) : Prop := 0 < r ∧ r < 1 / 5000

def m (n : ℕ) (r : ℝ) := (n + r)^3

theorem smallest_m : (∃ (n : ℕ) (r : ℝ), valid_r r ∧ n ≥ 41 ∧ m n r = 68922) :=
by
  sorry

end smallest_m_l589_58927


namespace roots_numerically_equal_opposite_signs_l589_58967

theorem roots_numerically_equal_opposite_signs
  (a b d: ℝ) 
  (h: ∃ x : ℝ, (x^2 - (a + 1) * x) / ((b + 1) * x - d) = (n - 2) / (n + 2) ∧ x = -x)
  : n = 2 * (b - a) / (a + b + 2) := by
  sorry

end roots_numerically_equal_opposite_signs_l589_58967


namespace center_of_symmetry_l589_58908

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * Real.tan (-7 * x + (Real.pi / 3))

theorem center_of_symmetry : f (Real.pi / 21) = 0 :=
by
  -- Mathematical proof goes here, skipping with sorry.
  sorry

end center_of_symmetry_l589_58908


namespace ratio_both_basketball_volleyball_l589_58959

variable (total_students : ℕ) (play_basketball : ℕ) (play_volleyball : ℕ) (play_neither : ℕ) (play_both : ℕ)

theorem ratio_both_basketball_volleyball (h1 : total_students = 20)
    (h2 : play_basketball = 20 / 2)
    (h3 : play_volleyball = (2 * 20) / 5)
    (h4 : play_neither = 4)
    (h5 : total_students - play_neither = play_basketball + play_volleyball - play_both) :
    play_both / total_students = 1 / 10 :=
by
  sorry

end ratio_both_basketball_volleyball_l589_58959


namespace complex_exp_form_pow_four_l589_58989

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l589_58989


namespace fencing_required_l589_58975

-- Conditions
def L : ℕ := 20
def A : ℕ := 680

-- Statement to prove
theorem fencing_required : ∃ W : ℕ, A = L * W ∧ 2 * W + L = 88 :=
by
  -- Here you would normally need the logical steps to arrive at the proof
  sorry

end fencing_required_l589_58975


namespace train_speed_kmph_l589_58971

def length_of_train : ℝ := 120
def length_of_bridge : ℝ := 255.03
def time_to_cross : ℝ := 30

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross * 3.6 = 45.0036 :=
by
  sorry

end train_speed_kmph_l589_58971


namespace initial_black_water_bottles_l589_58972

-- Define the conditions
variables (red black blue taken left total : ℕ)
variables (hred : red = 2) (hblue : blue = 4) (htaken : taken = 5) (hleft : left = 4)

-- State the theorem with the correct answer given the conditions
theorem initial_black_water_bottles : (red + black + blue = taken + left) → black = 3 :=
by
  intros htotal
  rw [hred, hblue, htaken, hleft] at htotal
  sorry

end initial_black_water_bottles_l589_58972


namespace train_length_l589_58915

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end train_length_l589_58915


namespace binom_divisible_by_prime_l589_58909

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : Nat.choose p k % p = 0 := 
  sorry

end binom_divisible_by_prime_l589_58909


namespace unit_digit_smaller_by_four_l589_58999

theorem unit_digit_smaller_by_four (x : ℤ) : x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4 :=
by
  sorry

end unit_digit_smaller_by_four_l589_58999


namespace abhinav_annual_salary_l589_58991

def RamMontlySalary : ℝ := 25600
def ShyamMontlySalary (A : ℝ) := 2 * A
def AbhinavAnnualSalary (A : ℝ) := 12 * A

theorem abhinav_annual_salary (A : ℝ) : 
  0.10 * RamMontlySalary = 0.08 * ShyamMontlySalary A → 
  AbhinavAnnualSalary A = 192000 :=
by
  sorry

end abhinav_annual_salary_l589_58991


namespace expand_expression_l589_58987

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 :=
  sorry

end expand_expression_l589_58987


namespace pieces_length_l589_58976

theorem pieces_length :
  let total_length_meters := 29.75
  let number_of_pieces := 35
  let length_per_piece_meters := total_length_meters / number_of_pieces
  let length_per_piece_centimeters := length_per_piece_meters * 100
  length_per_piece_centimeters = 85 :=
by
  sorry

end pieces_length_l589_58976


namespace solve_system_l589_58980

theorem solve_system : ∀ (a b : ℝ), (∃ (x y : ℝ), x = 5 ∧ y = b ∧ 2 * x + y = a ∧ 2 * x - y = 12) → (a = 8 ∧ b = -2) :=
by
  sorry

end solve_system_l589_58980


namespace beijing_time_conversion_l589_58988

-- Define the conversion conditions
def new_clock_hours_in_day : Nat := 10
def new_clock_minutes_per_hour : Nat := 100
def new_clock_time_at_5_beijing_time : Nat := 12 * 60  -- converting 12 noon to minutes


-- Define the problem to prove the corresponding Beijing time 
theorem beijing_time_conversion :
  new_clock_minutes_per_hour * 5 = 500 → 
  new_clock_time_at_5_beijing_time = 720 →
  (720 + 175 * 1.44) = 4 * 60 + 12 :=
by
  intros h1 h2
  sorry

end beijing_time_conversion_l589_58988


namespace speed_of_river_l589_58978

theorem speed_of_river :
  ∃ v : ℝ, 
    (∀ d : ℝ, (2 * d = 9.856) → 
              (d = 4.928) ∧ 
              (1 = (d / (10 - v) + d / (10 + v)))) 
    → v = 1.2 :=
sorry

end speed_of_river_l589_58978


namespace cost_of_adult_ticket_l589_58905

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l589_58905


namespace smallest_n_for_divisibility_l589_58952

theorem smallest_n_for_divisibility (n : ℕ) : 
  (∀ m, m > 0 → (315^2 - m^2) ∣ (315^3 - m^3) → m ≥ n) → 
  (315^2 - n^2) ∣ (315^3 - n^3) → 
  n = 90 :=
by
  sorry

end smallest_n_for_divisibility_l589_58952


namespace vector_equivalence_l589_58994

-- Define the vectors a and b
noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)

-- Define the operation 3a - b
noncomputable def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
  (3 * a.1 - b.1, 3 * a.2 - b.2)

-- State that for given vectors a and b, the result of the operation equals (4, 2)
theorem vector_equivalence : vector_operation vector_a vector_b = (4, 2) :=
  sorry

end vector_equivalence_l589_58994


namespace term_37_l589_58917

section GeometricSequence

variable {a b : ℕ → ℝ}
variable (q p : ℝ)

-- Definition of geometric sequences
def is_geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n ≥ 1, a (n + 1) = r * a n

-- Given conditions
axiom a1_25 : a 1 = 25
axiom b1_4 : b 1 = 4
axiom a2b2_100 : a 2 * b 2 = 100

-- Assume a and b are geometric sequences
axiom a_geom_seq : is_geometric_seq a q
axiom b_geom_seq : is_geometric_seq b p

-- Main theorem to prove
theorem term_37 (n : ℕ) (hn : n = 37) : (a n * b n) = 100 :=
sorry

end GeometricSequence

end term_37_l589_58917


namespace square_of_sum_l589_58968

theorem square_of_sum (x y : ℝ) (A B C D : ℝ) :
  A = 2 * x^2 + y^2 →
  B = 2 * (x + y)^2 →
  C = 2 * x + y^2 →
  D = (2 * x + y)^2 →
  D = (2 * x + y)^2 :=
by intros; exact ‹D = (2 * x + y)^2›

end square_of_sum_l589_58968


namespace find_c_l589_58998

theorem find_c (a b c : ℝ) (h1 : ∃ a, ∃ b, ∃ c, 
              ∀ y, (∀ x, (x = a * (y-1)^2 + 4) ↔ (x = -2 → y = 3)) ∧
              (∀ y, x = a * y^2 + b * y + c)) : c = 1 / 2 :=
sorry

end find_c_l589_58998


namespace good_numbers_characterization_l589_58977

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers_characterization :
  {n : ℕ | is_good n} = {1} ∪ {p | Nat.Prime p ∧ p % 2 = 1} :=
by 
  sorry

end good_numbers_characterization_l589_58977


namespace power_division_correct_l589_58985

theorem power_division_correct :
  (∀ x : ℝ, x^4 / x = x^3) ∧ 
  ¬(∀ x : ℝ, 3 * x^2 * 4 * x^2 = 12 * x^2) ∧
  ¬(∀ x : ℝ, (x - 1) * (x - 1) = x^2 - 1) ∧
  ¬(∀ x : ℝ, (x^5)^2 = x^7) := 
by {
  -- Proof would go here
  sorry
}

end power_division_correct_l589_58985


namespace average_score_after_19_innings_l589_58925

/-
  Problem Statement:
  Prove that the cricketer's average score after 19 innings is 24,
  given that scoring 96 runs in the 19th inning increased his average by 4.
-/

theorem average_score_after_19_innings :
  ∀ A : ℕ,
  (18 * A + 96) / 19 = A + 4 → A + 4 = 24 :=
by
  intros A h
  /- Skipping proof by adding "sorry" -/
  sorry

end average_score_after_19_innings_l589_58925


namespace height_of_parallelogram_l589_58937

def area_of_parallelogram (base height : ℝ) : ℝ := base * height

theorem height_of_parallelogram (A B H : ℝ) (hA : A = 33.3) (hB : B = 9) (hAparallelogram : A = area_of_parallelogram B H) :
  H = 3.7 :=
by 
  -- Proof would go here
  sorry

end height_of_parallelogram_l589_58937


namespace toothpicks_required_l589_58955

noncomputable def total_small_triangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def total_initial_toothpicks (n : ℕ) : ℕ :=
  3 * total_small_triangles n

noncomputable def adjusted_toothpicks (n : ℕ) : ℕ :=
  total_initial_toothpicks n / 2

noncomputable def boundary_toothpicks (n : ℕ) : ℕ :=
  2 * n

noncomputable def total_toothpicks (n : ℕ) : ℕ :=
  adjusted_toothpicks n + boundary_toothpicks n

theorem toothpicks_required {n : ℕ} (h : n = 2500) : total_toothpicks n = 4694375 :=
by sorry

end toothpicks_required_l589_58955


namespace original_group_size_l589_58906

theorem original_group_size (M : ℕ) 
  (h1 : ∀ work_done_by_one, work_done_by_one = 1 / (6 * M))
  (h2 : ∀ work_done_by_one, work_done_by_one = 1 / (12 * (M - 4))) : 
  M = 8 :=
by
  sorry

end original_group_size_l589_58906


namespace friends_can_reach_destinations_l589_58942

/-- The distance between Coco da Selva and Quixajuba is 24 km. 
    The walking speed is 6 km/h and the biking speed is 18 km/h. 
    Show that the friends can proceed to reach their destinations in at most 2 hours 40 minutes, with the bicycle initially in Quixajuba. -/
theorem friends_can_reach_destinations (d q c : ℕ) (vw vb : ℕ) (h1 : d = 24) (h2 : vw = 6) (h3 : vb = 18): 
  (∃ ta tb tc : ℕ, ta ≤ 2 * 60 + 40 ∧ tb ≤ 2 * 60 + 40 ∧ tc ≤ 2 * 60 + 40 ∧ 
     True) :=
sorry

end friends_can_reach_destinations_l589_58942


namespace sum_of_ages_is_l589_58997

-- Define the ages of the triplets and twins
def age_triplet (x : ℕ) := x
def age_twin (x : ℕ) := x - 3

-- Define the total age sum
def total_age_sum (x : ℕ) := 3 * age_triplet x + 2 * age_twin x

-- State the theorem
theorem sum_of_ages_is (x : ℕ) (h : total_age_sum x = 89) : ∃ x : ℕ, total_age_sum x = 89 := 
sorry

end sum_of_ages_is_l589_58997


namespace Joey_age_is_six_l589_58943

theorem Joey_age_is_six (ages: Finset ℕ) (a1 a2 a3 a4 : ℕ) (h1: ages = {4, 6, 8, 10})
  (h2: a1 + a2 = 14 ∨ a2 + a3 = 14 ∨ a3 + a4 = 14) (h3: a1 > 7 ∨ a2 > 7 ∨ a3 > 7 ∨ a4 > 7)
  (h4: (6 ∈ ages ∧ a1 ∈ ages) ∨ (6 ∈ ages ∧ a2 ∈ ages) ∨ 
      (6 ∈ ages ∧ a3 ∈ ages) ∨ (6 ∈ ages ∧ a4 ∈ ages)): 
  (a1 = 6 ∨ a2 = 6 ∨ a3 = 6 ∨ a4 = 6) :=
by
  sorry

end Joey_age_is_six_l589_58943


namespace inequality_solution_l589_58965

theorem inequality_solution (x : ℝ) :
  (x / (x^2 - 4) ≥ 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ico 0 2) :=
by sorry

end inequality_solution_l589_58965


namespace coin_difference_l589_58924

theorem coin_difference : ∀ (p : ℕ), 1 ≤ p ∧ p ≤ 999 → (10000 - 9 * 1) - (10000 - 9 * 999) = 8982 :=
by
  intro p
  intro hp
  sorry

end coin_difference_l589_58924


namespace compare_sqrt_expression_l589_58901

theorem compare_sqrt_expression : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := 
sorry

end compare_sqrt_expression_l589_58901


namespace man_mass_calculation_l589_58948

/-- A boat has a length of 4 m, a breadth of 2 m, and a weight of 300 kg.
    The density of the water is 1000 kg/m³.
    When the man gets on the boat, it sinks by 1 cm.
    Prove that the mass of the man is 80 kg. -/
theorem man_mass_calculation :
  let length_boat := 4     -- in meters
  let breadth_boat := 2    -- in meters
  let weight_boat := 300   -- in kg
  let density_water := 1000  -- in kg/m³
  let additional_depth := 0.01 -- in meters (1 cm)
  volume_displaced = length_boat * breadth_boat * additional_depth →
  mass_water_displaced = volume_displaced * density_water →
  mass_of_man = mass_water_displaced →
  mass_of_man = 80 :=
by 
  intros length_boat breadth_boat weight_boat density_water additional_depth volume_displaced
  intros mass_water_displaced mass_of_man
  sorry

end man_mass_calculation_l589_58948


namespace above_line_sign_l589_58979

theorem above_line_sign (A B C x y : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
(h_above : ∃ y₁, Ax + By₁ + C = 0 ∧ y > y₁) : 
  (Ax + By + C > 0 ∧ B > 0) ∨ (Ax + By + C < 0 ∧ B < 0) := 
by
  sorry

end above_line_sign_l589_58979


namespace minimum_cards_to_ensure_60_of_same_color_l589_58936

-- Define the conditions as Lean definitions
def total_cards : ℕ := 700
def ratio_red_orange_yellow : ℕ × ℕ × ℕ := (1, 3, 4)
def ratio_green_blue_white : ℕ × ℕ × ℕ := (3, 1, 6)
def yellow_more_than_blue : ℕ := 50

-- Define the proof goal
theorem minimum_cards_to_ensure_60_of_same_color :
  ∀ (x y : ℕ),
  (total_cards = (1 * x + 3 * x + 4 * x + 3 * y + y + 6 * y)) ∧
  (4 * x = y + yellow_more_than_blue) →
  min_cards :=
  -- Sorry here to indicate that proof is not provided
  sorry

end minimum_cards_to_ensure_60_of_same_color_l589_58936


namespace father_age_l589_58931

variable (F C1 C2 : ℕ)

theorem father_age (h1 : F = 3 * (C1 + C2))
  (h2 : F + 5 = 2 * (C1 + 5 + C2 + 5)) :
  F = 45 := by
  sorry

end father_age_l589_58931


namespace complement_intersection_l589_58928

-- Define the universal set U and sets A and B.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 4, 6}
def B : Set ℕ := {4, 5, 7}

-- Define the complements of A and B in U.
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof problem: Prove that the intersection of the complements of A and B 
-- in the universal set U equals {2, 3, 8}.
theorem complement_intersection :
  (C_UA ∩ C_UB = {2, 3, 8}) := by
  sorry

end complement_intersection_l589_58928


namespace eulers_formula_l589_58947

-- Definitions related to simply connected polyhedra
def SimplyConnectedPolyhedron (V E F : ℕ) : Prop := true  -- Genus 0 implies it is simply connected

-- Euler's characteristic property for simply connected polyhedra
theorem eulers_formula (V E F : ℕ) (h : SimplyConnectedPolyhedron V E F) : V - E + F = 2 := 
by
  sorry

end eulers_formula_l589_58947


namespace total_biking_distance_l589_58981

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end total_biking_distance_l589_58981


namespace evaluate_expression_l589_58964

-- Definitions based on conditions
variables (b : ℤ) (x : ℤ)
def condition := x = 2 * b + 9

-- Statement of the problem
theorem evaluate_expression (b : ℤ) (x : ℤ) (h : condition b x) : x - 2 * b + 5 = 14 :=
by sorry

end evaluate_expression_l589_58964


namespace total_remaining_macaroons_l589_58953

-- Define initial macaroons count
def initial_red_macaroons : ℕ := 50
def initial_green_macaroons : ℕ := 40

-- Define macaroons eaten
def eaten_green_macaroons : ℕ := 15
def eaten_red_macaroons : ℕ := 2 * eaten_green_macaroons

-- Define remaining macaroons
def remaining_red_macaroons : ℕ := initial_red_macaroons - eaten_red_macaroons
def remaining_green_macaroons : ℕ := initial_green_macaroons - eaten_green_macaroons

-- Prove the total remaining macaroons
theorem total_remaining_macaroons : remaining_red_macaroons + remaining_green_macaroons = 45 := 
by
  -- Proof omitted
  sorry

end total_remaining_macaroons_l589_58953


namespace sum_of_consecutive_integers_l589_58911

theorem sum_of_consecutive_integers (n : ℕ) (h : n*(n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end sum_of_consecutive_integers_l589_58911


namespace flour_amount_l589_58950

theorem flour_amount (a b : ℕ) (h₁ : a = 8) (h₂ : b = 2) : a + b = 10 := by
  sorry

end flour_amount_l589_58950


namespace students_per_van_correct_l589_58914

-- Define the conditions.
def num_vans : Nat := 6
def num_minibuses : Nat := 4
def students_per_minibus : Nat := 24
def total_students : Nat := 156

-- Define the number of students on each van is 'V'
def V : Nat := sorry 

-- State the final question/proof.
theorem students_per_van_correct : V = 10 :=
  sorry


end students_per_van_correct_l589_58914


namespace total_wheels_l589_58903

theorem total_wheels (n_bicycles n_tricycles n_unicycles n_four_wheelers : ℕ)
                     (w_bicycle w_tricycle w_unicycle w_four_wheeler : ℕ)
                     (h1 : n_bicycles = 16)
                     (h2 : n_tricycles = 7)
                     (h3 : n_unicycles = 10)
                     (h4 : n_four_wheelers = 5)
                     (h5 : w_bicycle = 2)
                     (h6 : w_tricycle = 3)
                     (h7 : w_unicycle = 1)
                     (h8 : w_four_wheeler = 4)
  : (n_bicycles * w_bicycle + n_tricycles * w_tricycle
     + n_unicycles * w_unicycle + n_four_wheelers * w_four_wheeler) = 83 := by
  sorry

end total_wheels_l589_58903


namespace find_number_l589_58940

theorem find_number : ∃ x : ℝ, 0.35 * x = 0.15 * 40 ∧ x = 120 / 7 :=
by
  sorry

end find_number_l589_58940


namespace artwork_collection_l589_58912

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l589_58912


namespace scientific_notation_integer_l589_58945

theorem scientific_notation_integer (x : ℝ) (h1 : x > 10) :
  ∃ (A : ℝ) (N : ℤ), (1 ≤ A ∧ A < 10) ∧ x = A * 10^N :=
by
  sorry

end scientific_notation_integer_l589_58945


namespace proof_evaluate_expression_l589_58932

def evaluate_expression : Prop :=
  - (18 / 3 * 8 - 72 + 4 * 8) = 8

theorem proof_evaluate_expression : evaluate_expression :=
by 
  sorry

end proof_evaluate_expression_l589_58932


namespace sufficient_not_necessary_condition_l589_58986

noncomputable def has_negative_root (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x < 0

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ x < 0) ↔ (a < 0) :=
sorry

end sufficient_not_necessary_condition_l589_58986


namespace books_leftover_l589_58956

theorem books_leftover (boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) 
  (h1 : boxes = 1575) (h2 : books_per_box = 45) (h3 : new_box_capacity = 50) :
  ((boxes * books_per_box) % new_box_capacity) = 25 :=
by
  sorry

end books_leftover_l589_58956


namespace delta_value_l589_58904

noncomputable def delta : ℝ :=
  Real.arccos (
    (Finset.range 3600).sum (fun k => Real.sin ((2539 + k) * Real.pi / 180)) ^ Real.cos (2520 * Real.pi / 180) +
    (Finset.range 3599).sum (fun k => Real.cos ((2521 + k) * Real.pi / 180)) +
    Real.cos (6120 * Real.pi / 180)
  )

theorem delta_value : delta = 71 :=
by
  sorry

end delta_value_l589_58904


namespace solve_x_l589_58900

theorem solve_x :
  ∃ x : ℝ, 2.5 * ( ( x * 0.48 * 2.50 ) / ( 0.12 * 0.09 * 0.5 ) ) = 2000.0000000000002 ∧ x = 3.6 :=
by sorry

end solve_x_l589_58900


namespace find_natural_n_l589_58935

theorem find_natural_n (a : ℂ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) (h₂ : a ≠ -1)
    (h₃ : a ^ 11 + a ^ 7 + a ^ 3 = 1) : a ^ 4 + a ^ 3 = a ^ 15 + 1 :=
sorry

end find_natural_n_l589_58935


namespace proof_problem_l589_58934

-- Definitions of propositions p and q
def p (a b : ℝ) : Prop := a < b → ∀ c : ℝ, c ≠ 0 → a * c^2 < b * c^2
def q : Prop := ∃ x₀ > 0, x₀ - 1 + Real.log x₀ = 0

-- Conditions for the problem
variable (a b : ℝ)
variable (p_false : ¬ p a b)
variable (q_true : q)

-- Proving which compound proposition is true
theorem proof_problem : (¬ p a b) ∧ q := by
  exact ⟨p_false, q_true⟩

end proof_problem_l589_58934


namespace ladder_base_distance_l589_58918

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l589_58918


namespace symmetric_periodic_l589_58910

theorem symmetric_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, f (a - x) = f (a + x))
  (h3 : ∀ x : ℝ, f (b - x) = f (b + x)) :
  ∀ x : ℝ, f x = f (x + 2 * (b - a)) :=
by
  sorry

end symmetric_periodic_l589_58910


namespace worth_of_presents_is_33536_36_l589_58960

noncomputable def total_worth_of_presents : ℝ :=
  let ring := 4000
  let car := 2000
  let bracelet := 2 * ring
  let gown := bracelet / 2
  let jewelry := 1.2 * ring
  let painting := 3000 * 1.2
  let honeymoon := 180000 / 110
  let watch := 5500
  ring + car + bracelet + gown + jewelry + painting + honeymoon + watch

theorem worth_of_presents_is_33536_36 : total_worth_of_presents = 33536.36 := by
  sorry

end worth_of_presents_is_33536_36_l589_58960


namespace quadratic_switch_real_roots_l589_58961

theorem quadratic_switch_real_roots (a b c u v w : ℝ) (ha : a ≠ u)
  (h_root1 : b^2 - 4 * a * c ≥ 0)
  (h_root2 : v^2 - 4 * u * w ≥ 0)
  (hwc : w * c > 0) :
  (b^2 - 4 * u * c ≥ 0) ∨ (v^2 - 4 * a * w ≥ 0) :=
sorry

end quadratic_switch_real_roots_l589_58961


namespace minimum_notes_to_determine_prize_location_l589_58951

/--
There are 100 boxes, numbered from 1 to 100. A prize is hidden in one of the boxes, 
and the host knows its location. The viewer can send the host a batch of notes 
with questions that require a "yes" or "no" answer. The host shuffles the notes 
in the batch and, without announcing the questions aloud, honestly answers 
all of them. Prove that the minimum number of notes that need to be sent to 
definitely determine where the prize is located is 99.
-/
theorem minimum_notes_to_determine_prize_location : 
  ∀ (boxes : Fin 100 → Prop) (prize_location : ∃ i : Fin 100, boxes i) 
    (batch_size : Nat), 
  (batch_size + 1) ≥ 100 → batch_size = 99 :=
by
  sorry

end minimum_notes_to_determine_prize_location_l589_58951


namespace range_of_a_l589_58916

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt x) :
  (f a < f (a + 1)) ↔ a ∈ Set.Ici (-1) :=
by
  sorry

end range_of_a_l589_58916


namespace Maggie_apples_l589_58949

-- Definition of our problem conditions
def K : ℕ := 28 -- Kelsey's apples
def L : ℕ := 22 -- Layla's apples
def avg : ℕ := 30 -- The average number of apples picked

-- Main statement to prove Maggie's apples
theorem Maggie_apples : (A : ℕ) → (A + K + L) / 3 = avg → A = 40 := by
  intros A h
  -- sorry is added to skip the proof since it's not required here.
  sorry

end Maggie_apples_l589_58949


namespace right_triangle_medians_l589_58995

theorem right_triangle_medians
    (a b c d m : ℝ)
    (h1 : ∀(a b c d : ℝ), 2 * (c/d) = 3)
    (h2 : m = 4 * 3 ∨ m = (3/4)) :
    ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ (m₁ = 12 ∨ m₁ = 3/4) ∧ (m₂ = 12 ∨ m₂ = 3/4) :=
by 
  sorry

end right_triangle_medians_l589_58995


namespace efficiency_ratio_l589_58926

theorem efficiency_ratio (E_A E_B : ℝ) 
  (h1 : E_B = 1 / 18) 
  (h2 : E_A + E_B = 1 / 6) : 
  E_A / E_B = 2 :=
by {
  sorry
}

end efficiency_ratio_l589_58926


namespace max_sum_of_two_integers_l589_58993

theorem max_sum_of_two_integers (x : ℕ) (h : x + 2 * x < 100) : x + 2 * x = 99 :=
sorry

end max_sum_of_two_integers_l589_58993


namespace purchase_probability_l589_58941

/--
A batch of products from a company has packages containing 10 components each.
Each package has either 1 or 2 second-grade components. 10% of the packages
contain 2 second-grade components. Xiao Zhang will decide to purchase
if all 4 randomly selected components from a package are first-grade.

We aim to prove the probability that Xiao Zhang decides to purchase the company's
products is \( \frac{43}{75} \).
-/
theorem purchase_probability : true := sorry

end purchase_probability_l589_58941


namespace annie_gives_mary_25_crayons_l589_58939

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end annie_gives_mary_25_crayons_l589_58939


namespace ellipse_hexagon_proof_l589_58933

noncomputable def m_value : ℝ := 3 + 2 * Real.sqrt 3

theorem ellipse_hexagon_proof (m : ℝ) (k : ℝ) 
  (hk : k ≠ 0) (hm : m > 3) :
  (∀ x y : ℝ, (x / m)^2 + (y / 3)^2 = 1 ∧ (y = k * x ∨ y = -k * x)) →
  k = Real.sqrt 3 →
  (|((4*m)/(m+1)) - (m-3)| = 0) →
  m = m_value :=
by
  sorry

end ellipse_hexagon_proof_l589_58933


namespace total_water_filled_jars_l589_58992

theorem total_water_filled_jars :
  ∃ x : ℕ, 
    16 * (1/4) + 12 * (1/2) + 8 * 1 + 4 * 2 + x * 3 = 56 ∧
    16 + 12 + 8 + 4 + x = 50 :=
by
  sorry

end total_water_filled_jars_l589_58992


namespace determine_ABC_l589_58920

noncomputable def digits_are_non_zero_distinct_and_not_larger_than_5 (A B C : ℕ) : Prop :=
  0 < A ∧ A ≤ 5 ∧ 0 < B ∧ B ≤ 5 ∧ 0 < C ∧ C ≤ 5 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

noncomputable def first_condition (A B : ℕ) : Prop :=
  A * 6 + B + A = B * 6 + A -- AB_6 + A_6 = BA_6 condition translated into arithmetics

noncomputable def second_condition (A B C : ℕ) : Prop :=
  A * 6 + B + B = C * 6 + 1 -- AB_6 + B_6 = C1_6 condition translated into arithmetics

theorem determine_ABC (A B C : ℕ) (h1 : digits_are_non_zero_distinct_and_not_larger_than_5 A B C)
    (h2 : first_condition A B) (h3 : second_condition A B C) :
    A * 100 + B * 10 + C = 5 * 100 + 1 * 10 + 5 := -- Final transformation of ABC to 515
  sorry

end determine_ABC_l589_58920


namespace puppies_per_cage_l589_58963

/-
Theorem: If a pet store had 56 puppies, sold 24 of them, and placed the remaining puppies into 8 cages, then each cage contains 4 puppies.
-/

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (cages : ℕ)
  (remaining_puppies : ℕ)
  (puppies_per_cage : ℕ) :
  initial_puppies = 56 →
  sold_puppies = 24 →
  cages = 8 →
  remaining_puppies = initial_puppies - sold_puppies →
  puppies_per_cage = remaining_puppies / cages →
  puppies_per_cage = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end puppies_per_cage_l589_58963


namespace carol_ate_12_cakes_l589_58973

-- Definitions for conditions
def cakes_per_day : ℕ := 10
def days_baking : ℕ := 5
def cans_per_cake : ℕ := 2
def cans_for_remaining_cakes : ℕ := 76

-- Total cakes baked by Sara
def total_cakes_baked (cakes_per_day days_baking : ℕ) : ℕ :=
  cakes_per_day * days_baking

-- Remaining cakes based on frosting cans needed
def remaining_cakes (cans_for_remaining_cakes cans_per_cake : ℕ) : ℕ :=
  cans_for_remaining_cakes / cans_per_cake

-- Cakes Carol ate
def cakes_carol_ate (total_cakes remaining_cakes : ℕ) : ℕ :=
  total_cakes - remaining_cakes

-- Theorem statement
theorem carol_ate_12_cakes :
  cakes_carol_ate (total_cakes_baked cakes_per_day days_baking) (remaining_cakes cans_for_remaining_cakes cans_per_cake) = 12 :=
by
  sorry

end carol_ate_12_cakes_l589_58973


namespace savings_percentage_l589_58957

variables (I S : ℝ)
-- Conditions
-- A man saves a certain portion S of his income I during the first year.
-- He spends the remaining portion (I - S) on his personal expenses.
-- In the second year, his income increases by 50%, so his new income is 1.5I.
-- His savings increase by 100%, so his new savings are 2S.
-- His total expenditure in 2 years is double his expenditure in the first year.

def first_year_expenditure (I S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (S : ℝ) : ℝ := 2 * S
def second_year_expenditure (I S : ℝ) : ℝ := second_year_income I - second_year_savings S
def total_expenditure (I S : ℝ) : ℝ := first_year_expenditure I S + second_year_expenditure I S

theorem savings_percentage :
  total_expenditure I S = 2 * first_year_expenditure I S → S / I = 0.5 :=
by
  sorry

end savings_percentage_l589_58957


namespace symmetric_to_y_axis_circle_l589_58938

open Real

-- Definition of the original circle's equation
def original_circle (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 3

-- Definition of the symmetric circle's equation with respect to the y-axis
def symmetric_circle (x y : ℝ) : Prop := x^2 + 2 * x + y^2 = 3

-- Theorem stating that the symmetric circle has the given equation
theorem symmetric_to_y_axis_circle (x y : ℝ) : 
  (symmetric_circle x y) ↔ (original_circle ((-x) - 2) y) :=
sorry

end symmetric_to_y_axis_circle_l589_58938


namespace longest_perimeter_l589_58974

theorem longest_perimeter 
  (x : ℝ) (h : x > 1)
  (pA : ℝ := 4 + 6 * x)
  (pB : ℝ := 2 + 10 * x)
  (pC : ℝ := 7 + 5 * x)
  (pD : ℝ := 6 + 6 * x)
  (pE : ℝ := 1 + 11 * x) :
  pE > pA ∧ pE > pB ∧ pE > pC ∧ pE > pD :=
by
  sorry

end longest_perimeter_l589_58974


namespace space_left_each_side_l589_58966

theorem space_left_each_side (wall_width : ℕ) (picture_width : ℕ)
  (picture_centered : wall_width = 2 * ((wall_width - picture_width) / 2) + picture_width) :
  (wall_width - picture_width) / 2 = 9 :=
by
  have h : wall_width = 25 := sorry
  have h2 : picture_width = 7 := sorry
  exact sorry

end space_left_each_side_l589_58966


namespace shaded_square_percentage_l589_58930

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 16) (h2 : shaded_squares = 8) : 
  (shaded_squares : ℚ) / total_squares * 100 = 50 :=
by
  sorry

end shaded_square_percentage_l589_58930
