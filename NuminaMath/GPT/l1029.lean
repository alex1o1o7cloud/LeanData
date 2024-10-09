import Mathlib

namespace John_study_time_second_exam_l1029_102986

variable (StudyTime Score : ℝ)
variable (k : ℝ) (h1 : k = Score / StudyTime)
variable (study_first : ℝ := 3) (score_first : ℝ := 60)
variable (avg_target : ℝ := 75)
variable (total_tests : ℕ := 2)

theorem John_study_time_second_exam :
  (avg_target * total_tests - score_first) / (score_first / study_first) = 4.5 :=
by
  sorry

end John_study_time_second_exam_l1029_102986


namespace number_of_pieces_of_string_l1029_102987

theorem number_of_pieces_of_string (total_length piece_length : ℝ) (h1 : total_length = 60) (h2 : piece_length = 0.6) :
    total_length / piece_length = 100 := by
  sorry

end number_of_pieces_of_string_l1029_102987


namespace area_of_quadrilateral_l1029_102946

/-- The area of the quadrilateral defined by the system of inequalities is 15/7. -/
theorem area_of_quadrilateral : 
  (∃ (x y : ℝ), 3 * x + 2 * y ≤ 6 ∧ x + 3 * y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0) →
  (∃ (area : ℝ), area = 15 / 7) :=
by
  sorry

end area_of_quadrilateral_l1029_102946


namespace greatest_five_digit_common_multiple_l1029_102934

theorem greatest_five_digit_common_multiple (n : ℕ) :
  (n % 18 = 0) ∧ (10000 ≤ n) ∧ (n ≤ 99999) → n = 99990 :=
by
  sorry

end greatest_five_digit_common_multiple_l1029_102934


namespace problem1_problem2_l1029_102998

noncomputable def problem1_solution1 : ℝ := (2 + Real.sqrt 6) / 2
noncomputable def problem1_solution2 : ℝ := (2 - Real.sqrt 6) / 2

theorem problem1 (x : ℝ) : 
  (2 * x ^ 2 - 4 * x - 1 = 0) ↔ (x = problem1_solution1 ∨ x = problem1_solution2) :=
by
  sorry

theorem problem2 : 
  (4 * (x + 2) ^ 2 - 9 * (x - 3) ^ 2 = 0) ↔ (x = 1 ∨ x = 13) :=
by
  sorry

end problem1_problem2_l1029_102998


namespace ruby_height_l1029_102920

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end ruby_height_l1029_102920


namespace bridge_length_l1029_102992

theorem bridge_length 
  (train_length : ℕ) 
  (speed_km_hr : ℕ) 
  (cross_time_sec : ℕ) 
  (conversion_factor_num : ℕ) 
  (conversion_factor_den : ℕ)
  (expected_length : ℕ) 
  (speed_m_s : ℕ := speed_km_hr * conversion_factor_num / conversion_factor_den)
  (total_distance : ℕ := speed_m_s * cross_time_sec) :
  train_length = 150 →
  speed_km_hr = 45 →
  cross_time_sec = 30 →
  conversion_factor_num = 1000 →
  conversion_factor_den = 3600 →
  expected_length = 225 →
  total_distance - train_length = expected_length :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end bridge_length_l1029_102992


namespace side_length_is_36_l1029_102937

variable (a : ℝ)

def side_length_of_largest_square (a : ℝ) := 
  2 * (a / 2) ^ 2 + 2 * (a / 4) ^ 2 = 810

theorem side_length_is_36 (h : side_length_of_largest_square a) : a = 36 :=
by
  sorry

end side_length_is_36_l1029_102937


namespace monotonic_increasing_implies_range_l1029_102985

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x ^ 2 + 2 * x - 2 * log x

theorem monotonic_increasing_implies_range (a : ℝ) :
  (∀ x > (0 : ℝ), deriv f x ≥ 0) → a ≤ 1 :=
  by 
  sorry

end monotonic_increasing_implies_range_l1029_102985


namespace largest_four_digit_negative_congruent_to_1_pmod_17_l1029_102926

theorem largest_four_digit_negative_congruent_to_1_pmod_17 :
  ∃ n : ℤ, 17 * n + 1 < -1000 ∧ 17 * n + 1 ≥ -9999 ∧ 17 * n + 1 ≡ 1 [ZMOD 17] := 
sorry

end largest_four_digit_negative_congruent_to_1_pmod_17_l1029_102926


namespace circle_radius_l1029_102965

theorem circle_radius 
  (x y : ℝ)
  (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = Real.sqrt 117 :=
by 
  sorry

end circle_radius_l1029_102965


namespace months_passed_l1029_102945

-- Let's define our conditions in mathematical terms
def received_bones (months : ℕ) : ℕ := 10 * months
def buried_bones : ℕ := 42
def available_bones : ℕ := 8
def total_bones (months : ℕ) : Prop := received_bones months = buried_bones + available_bones

-- We need to prove that the number of months (x) satisfies the condition
theorem months_passed (x : ℕ) : total_bones x → x = 5 :=
by
  sorry

end months_passed_l1029_102945


namespace mean_of_xyz_l1029_102957

theorem mean_of_xyz (x y z : ℝ) (h1 : 9 * x + 3 * y - 5 * z = -4) (h2 : 5 * x + 2 * y - 2 * z = 13) : 
  (x + y + z) / 3 = 10 := 
sorry

end mean_of_xyz_l1029_102957


namespace greatest_integer_l1029_102902

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 2) (h3 : ∃ l : ℕ, n = 8 * l - 4) : n = 124 :=
by
  sorry

end greatest_integer_l1029_102902


namespace minimal_S_n_l1029_102929

theorem minimal_S_n (a_n : ℕ → ℤ) 
  (h : ∀ n, a_n n = 3 * (n : ℤ) - 23) :
  ∃ n, (∀ m < n, (∀ k ≥ n, a_n k ≤ 0)) → n = 7 :=
by
  sorry

end minimal_S_n_l1029_102929


namespace root_in_interval_sum_eq_three_l1029_102928

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end root_in_interval_sum_eq_three_l1029_102928


namespace arithmetic_sequence_values_l1029_102927

noncomputable def common_difference (a₁ a₂ : ℕ) : ℕ := (a₂ - a₁) / 2

theorem arithmetic_sequence_values (x y z d: ℕ) 
    (h₁: d = common_difference 7 11) 
    (h₂: x = 7 + d) 
    (h₃: y = 11 + d) 
    (h₄: z = y + d): 
    x = 9 ∧ y = 13 ∧ z = 15 :=
by {
  sorry
}

end arithmetic_sequence_values_l1029_102927


namespace sum_consecutive_even_integers_l1029_102911

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end sum_consecutive_even_integers_l1029_102911


namespace triangle_perimeter_l1029_102947

-- Define the given quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - (5 + m) * x + 5 * m

-- Define the isosceles triangle with sides given by the roots of the equation
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

-- Defining the fact that 2 is a root of the given quadratic equation with an unknown m
lemma two_is_root (m : ℝ) : quadratic_equation m 2 = 0 := sorry

-- Prove that the perimeter of triangle ABC is 12 given the conditions
theorem triangle_perimeter (α β γ : ℝ) (m : ℝ) (h1 : quadratic_equation m α = 0) 
  (h2 : quadratic_equation m β = 0) 
  (h3 : is_isosceles_triangle α β γ) : α + β + γ = 12 := sorry

end triangle_perimeter_l1029_102947


namespace percent_of_x_l1029_102940

-- The mathematical equivalent of the problem statement in Lean.
theorem percent_of_x (x : ℝ) (hx : 0 < x) : (x / 10 + x / 25) = 0.14 * x :=
by
  sorry

end percent_of_x_l1029_102940


namespace problem_statement_l1029_102972

def g (x : ℕ) : ℕ := x^2 - 4 * x

theorem problem_statement :
  g (g (g (g (g (g 2))))) = L := sorry

end problem_statement_l1029_102972


namespace cube_add_constant_135002_l1029_102917

theorem cube_add_constant_135002 (n : ℤ) : 
  (∃ m : ℤ, m = n + 1 ∧ m^3 - n^3 = 135002) →
  (n = 149 ∨ n = -151) :=
by
  -- This is where the proof should go
  sorry

end cube_add_constant_135002_l1029_102917


namespace negation_of_exists_leq_l1029_102935

theorem negation_of_exists_leq (
  P : ∃ x : ℝ, x^2 - 2 * x + 4 ≤ 0
) : ∀ x : ℝ, x^2 - 2 * x + 4 > 0 :=
sorry

end negation_of_exists_leq_l1029_102935


namespace solution_of_equation_l1029_102963

def solve_equation (x : ℚ) : Prop := 
  (x^2 + 3 * x + 4) / (x + 5) = x + 6

theorem solution_of_equation : solve_equation (-13/4) := 
by
  sorry

end solution_of_equation_l1029_102963


namespace largest_primes_product_l1029_102913

theorem largest_primes_product : 7 * 97 * 997 = 679679 := by
  sorry

end largest_primes_product_l1029_102913


namespace find_f_2012_l1029_102951

-- Given a function f: ℤ → ℤ that satisfies the functional equation:
def functional_equation (f : ℤ → ℤ) := ∀ m n : ℤ, m + f (m + f (n + f m)) = n + f m

-- Given condition:
def f_6_is_6 (f : ℤ → ℤ) := f 6 = 6

-- We need to prove that f 2012 = -2000 under the given conditions.
theorem find_f_2012 (f : ℤ → ℤ) (hf : functional_equation f) (hf6 : f_6_is_6 f) : f 2012 = -2000 := sorry

end find_f_2012_l1029_102951


namespace max_value_a_plus_b_plus_c_plus_d_eq_34_l1029_102989

theorem max_value_a_plus_b_plus_c_plus_d_eq_34 :
  ∃ (a b c d : ℕ), (∀ (x y: ℝ), 0 < x → 0 < y → x^2 - 2 * x * y + 3 * y^2 = 10 → x^2 + 2 * x * y + 3 * y^2 = (a + b * Real.sqrt c) / d) ∧ a + b + c + d = 34 :=
sorry

end max_value_a_plus_b_plus_c_plus_d_eq_34_l1029_102989


namespace find_k_value_l1029_102975

noncomputable def solve_for_k (k : ℚ) : Prop :=
  ∃ x : ℚ, (x = 1) ∧ (3 * x + (2 * k - 1) = x - 6 * (3 * k + 2))

theorem find_k_value : solve_for_k (-13 / 20) :=
  sorry

end find_k_value_l1029_102975


namespace min_value_b_over_a_l1029_102923

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + (Real.exp 1 - a) * x - b

theorem min_value_b_over_a 
  (a b : ℝ)
  (h_cond : ∀ x > 0, f x a b ≤ 0)
  (h_b : b = -1 - Real.log (a - Real.exp 1)) 
  (h_a_gt_e : a > Real.exp 1) :
  ∃ (x : ℝ), x = 2 * Real.exp 1 ∧ (b / a) = - (1 / Real.exp 1) := 
sorry

end min_value_b_over_a_l1029_102923


namespace hoseok_position_reversed_l1029_102932

def nine_people (P : ℕ → Prop) : Prop :=
  P 1 ∧ P 2 ∧ P 3 ∧ P 4 ∧ P 5 ∧ P 6 ∧ P 7 ∧ P 8 ∧ P 9

variable (h : ℕ → Prop)

def hoseok_front_foremost : Prop :=
  nine_people h ∧ h 1 -- Hoseok is at the forefront and is the shortest

theorem hoseok_position_reversed :
  hoseok_front_foremost h → h 9 :=
by 
  sorry

end hoseok_position_reversed_l1029_102932


namespace proof_problem_l1029_102938

variable (a b c : ℝ)

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : ∀ x, abs (x + a) - abs (x - b) + c ≤ 10) :
  a + b + c = 10 ∧ 
  (∀ (h5 : a + b + c = 10), 
    (∃ a' b' c', a' = 11/3 ∧ b' = 8/3 ∧ c' = 11/3 ∧ 
                (∀ a'' b'' c'', a'' = a ∧ b'' = b ∧ c'' = c → 
                (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2) ≥ 8/3 ∧ 
                (1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) = 8 / 3 ))) := by
  sorry

end proof_problem_l1029_102938


namespace inequality_solution_sets_equivalence_l1029_102956

theorem inequality_solution_sets_equivalence
  (a b : ℝ)
  (h1 : (∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0)) :
  (∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ bx^2 - 5 * x + a > 0) :=
  sorry

end inequality_solution_sets_equivalence_l1029_102956


namespace perfect_cubes_count_l1029_102996

theorem perfect_cubes_count : 
  Nat.card {n : ℕ | n^3 > 500 ∧ n^3 < 2000} = 5 :=
by
  sorry

end perfect_cubes_count_l1029_102996


namespace largest_value_of_x_l1029_102948

theorem largest_value_of_x (x : ℝ) (hx : x / 3 + 1 / (7 * x) = 1 / 2) : 
  x = (21 + Real.sqrt 105) / 28 := 
sorry

end largest_value_of_x_l1029_102948


namespace james_hours_to_work_l1029_102995

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end james_hours_to_work_l1029_102995


namespace derivative_at_2_l1029_102919

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_at_2 : deriv f 2 = Real.sqrt 2 / 4 := by
  sorry

end derivative_at_2_l1029_102919


namespace total_payment_mr_benson_made_l1029_102955

noncomputable def general_admission_ticket_cost : ℝ := 40
noncomputable def num_general_admission_tickets : ℕ := 10
noncomputable def num_vip_tickets : ℕ := 3
noncomputable def num_premium_tickets : ℕ := 2
noncomputable def vip_ticket_rate_increase : ℝ := 0.20
noncomputable def premium_ticket_rate_increase : ℝ := 0.50
noncomputable def discount_rate : ℝ := 0.05
noncomputable def threshold_tickets : ℕ := 10

noncomputable def vip_ticket_cost : ℝ := general_admission_ticket_cost * (1 + vip_ticket_rate_increase)
noncomputable def premium_ticket_cost : ℝ := general_admission_ticket_cost * (1 + premium_ticket_rate_increase)

noncomputable def total_general_admission_cost : ℝ := num_general_admission_tickets * general_admission_ticket_cost
noncomputable def total_vip_cost : ℝ := num_vip_tickets * vip_ticket_cost
noncomputable def total_premium_cost : ℝ := num_premium_tickets * premium_ticket_cost

noncomputable def total_tickets : ℕ := num_general_admission_tickets + num_vip_tickets + num_premium_tickets
noncomputable def tickets_exceeding_threshold : ℕ := if total_tickets > threshold_tickets then total_tickets - threshold_tickets else 0

noncomputable def discounted_vip_cost : ℝ := vip_ticket_cost * (1 - discount_rate)
noncomputable def discounted_premium_cost : ℝ := premium_ticket_cost * (1 - discount_rate)

noncomputable def total_discounted_vip_cost : ℝ :=  num_vip_tickets * discounted_vip_cost
noncomputable def total_discounted_premium_cost : ℝ := num_premium_tickets * discounted_premium_cost

noncomputable def total_cost_with_discounts : ℝ := total_general_admission_cost + total_discounted_vip_cost + total_discounted_premium_cost

theorem total_payment_mr_benson_made : total_cost_with_discounts = 650.80 :=
by
  -- Proof is omitted
  sorry

end total_payment_mr_benson_made_l1029_102955


namespace mutually_exclusive_white_ball_events_l1029_102977

-- Definitions of persons and balls
inductive Person | A | B | C
inductive Ball | red | black | white

-- Definitions of events
def eventA (dist : Person → Ball) : Prop := dist Person.A = Ball.white
def eventB (dist : Person → Ball) : Prop := dist Person.B = Ball.white

theorem mutually_exclusive_white_ball_events (dist : Person → Ball) :
  (eventA dist → ¬eventB dist) :=
by
  sorry

end mutually_exclusive_white_ball_events_l1029_102977


namespace cube_sum_gt_zero_l1029_102924

variable {x y z : ℝ}

theorem cube_sum_gt_zero (h1 : x < y) (h2 : y < z) : 
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 :=
sorry

end cube_sum_gt_zero_l1029_102924


namespace ticket_sales_revenue_l1029_102962

theorem ticket_sales_revenue :
  let student_ticket_price := 4
  let general_admission_ticket_price := 6
  let total_tickets_sold := 525
  let general_admission_tickets_sold := 388
  let student_tickets_sold := total_tickets_sold - general_admission_tickets_sold
  let money_from_student_tickets := student_tickets_sold * student_ticket_price
  let money_from_general_admission_tickets := general_admission_tickets_sold * general_admission_ticket_price
  let total_money_collected := money_from_student_tickets + money_from_general_admission_tickets
  total_money_collected = 2876 :=
by
  sorry

end ticket_sales_revenue_l1029_102962


namespace length_of_bridge_l1029_102943

theorem length_of_bridge (ship_length : ℝ) (ship_speed_kmh : ℝ) (time : ℝ) (bridge_length : ℝ) :
  ship_length = 450 → ship_speed_kmh = 24 → time = 202.48 → bridge_length = (6.67 * 202.48 - 450) → bridge_length = 900.54 :=
by
  intros h1 h2 h3 h4
  sorry

end length_of_bridge_l1029_102943


namespace cos_angle_relation_l1029_102970

theorem cos_angle_relation (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (2 * α - 2 * π / 3) = -7 / 9 := by 
  sorry

end cos_angle_relation_l1029_102970


namespace brian_total_commission_l1029_102984

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end brian_total_commission_l1029_102984


namespace existence_of_special_numbers_l1029_102936

theorem existence_of_special_numbers :
  ∃ (N : Finset ℕ), N.card = 1998 ∧ 
  ∀ (a b : ℕ), a ∈ N → b ∈ N → a ≠ b → a * b ∣ (a - b)^2 :=
sorry

end existence_of_special_numbers_l1029_102936


namespace solution_set_of_quadratic_inequality_l1029_102905

variable {a x : ℝ} (h_neg : a < 0)

theorem solution_set_of_quadratic_inequality :
  (a * x^2 - (a + 2) * x + 2) ≥ 0 ↔ (x ∈ Set.Icc (2 / a) 1) :=
by
  sorry

end solution_set_of_quadratic_inequality_l1029_102905


namespace total_local_percentage_approx_52_74_l1029_102916

-- We provide the conditions as definitions
def total_arts_students : ℕ := 400
def local_arts_percentage : ℝ := 0.50
def total_science_students : ℕ := 100
def local_science_percentage : ℝ := 0.25
def total_commerce_students : ℕ := 120
def local_commerce_percentage : ℝ := 0.85

-- Calculate the expected total percentage of local students
noncomputable def calculated_total_local_percentage : ℝ :=
  let local_arts_students := local_arts_percentage * total_arts_students
  let local_science_students := local_science_percentage * total_science_students
  let local_commerce_students := local_commerce_percentage * total_commerce_students
  let total_local_students := local_arts_students + local_science_students + local_commerce_students
  let total_students := total_arts_students + total_science_students + total_commerce_students
  (total_local_students / total_students) * 100

-- State what we need to prove
theorem total_local_percentage_approx_52_74 :
  abs (calculated_total_local_percentage - 52.74) < 1 :=
sorry

end total_local_percentage_approx_52_74_l1029_102916


namespace number_of_pens_l1029_102990

theorem number_of_pens (x y : ℝ) (h1 : 60 * (x + 2 * y) = 50 * (x + 3 * y)) (h2 : x = 3 * y) : 
  (60 * (x + 2 * y)) / x = 100 :=
by
  sorry

end number_of_pens_l1029_102990


namespace total_candy_eaten_by_bobby_l1029_102914

-- Definitions based on the problem conditions
def candy_eaten_by_bobby_round1 : ℕ := 28
def candy_eaten_by_bobby_round2 : ℕ := 42
def chocolate_eaten_by_bobby : ℕ := 63

-- Define the statement to prove
theorem total_candy_eaten_by_bobby : 
  candy_eaten_by_bobby_round1 + candy_eaten_by_bobby_round2 + chocolate_eaten_by_bobby = 133 :=
  by
  -- Skipping the proof itself
  sorry

end total_candy_eaten_by_bobby_l1029_102914


namespace population_net_increase_in_one_day_l1029_102950

-- Define the problem conditions
def birth_rate : ℕ := 6 / 2  -- births per second
def death_rate : ℕ := 3 / 2  -- deaths per second
def seconds_in_a_day : ℕ := 60 * 60 * 24

-- Define the assertion we want to prove
theorem population_net_increase_in_one_day : 
  ( (birth_rate - death_rate) * seconds_in_a_day ) = 259200 := by
  -- Since 6/2 = 3 and 3/2 = 1.5 is not an integer in Lean, we use ratios directly
  sorry  -- Proof is not required

end population_net_increase_in_one_day_l1029_102950


namespace curve_product_l1029_102910

theorem curve_product (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) : a * b = -3 := by
  sorry

end curve_product_l1029_102910


namespace min_value_f_l1029_102982

noncomputable def f (a b : ℝ) : ℝ := (1 / a^5 + a^5 - 2) * (1 / b^5 + b^5 - 2)

theorem min_value_f :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → f a b ≥ (31^4 / 32^2) :=
by
  intros
  sorry

end min_value_f_l1029_102982


namespace range_of_a_l1029_102999

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + a * x + 2)
  (h2 : ∀ y, (∃ x, y = f (f x)) ↔ (∃ x, y = f x)) : a ≥ 4 ∨ a ≤ -2 := 
sorry

end range_of_a_l1029_102999


namespace gray_area_l1029_102979

-- Given conditions
def rect1_length : ℕ := 8
def rect1_width : ℕ := 10
def rect2_length : ℕ := 12
def rect2_width : ℕ := 9
def black_area : ℕ := 37

-- Define areas based on conditions
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width
def white_area : ℕ := area_rect1 - black_area

-- Theorem to prove the area of the gray part
theorem gray_area : area_rect2 - white_area = 65 :=
by
  sorry

end gray_area_l1029_102979


namespace largest_int_less_than_100_with_remainder_5_l1029_102960

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l1029_102960


namespace mean_median_mode_l1029_102930

theorem mean_median_mode (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : m + 7 < n) 
  (h4 : (m + (m + 3) + (m + 7) + n + (n + 5) + (2 * n - 1)) / 6 = n)
  (h5 : ((m + 7) + n) / 2 = n)
  (h6 : (m+3 < m+7 ∧ m+7 = n ∧ n < n+5 ∧ n+5 < 2*n - 1 )) :
  m+n = 2*n := by
  sorry

end mean_median_mode_l1029_102930


namespace inequality_sin_cos_l1029_102976

theorem inequality_sin_cos 
  (a b : ℝ) (n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) : 
  (a / (Real.sin x)^n) + (b / (Real.cos x)^n) ≥ (a^(2/(n+2)) + b^(2/(n+2)))^((n+2)/2) :=
sorry

end inequality_sin_cos_l1029_102976


namespace angle_measure_l1029_102900

theorem angle_measure (x : ℝ) (h : 180 - x = (90 - x) - 4) : x = 60 := by
  sorry

end angle_measure_l1029_102900


namespace solution_interval_l1029_102997

theorem solution_interval (x : ℝ) : 
  (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7) ∨ (7 < x) ↔ 
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0) := sorry

end solution_interval_l1029_102997


namespace total_drums_l1029_102993

theorem total_drums (x y : ℕ) (hx : 30 * x + 20 * y = 160) : x + y = 7 :=
sorry

end total_drums_l1029_102993


namespace number_of_distinct_cubes_l1029_102909

theorem number_of_distinct_cubes (w b : ℕ) (total_cubes : ℕ) (dim : ℕ) :
  w + b = total_cubes ∧ total_cubes = 8 ∧ dim = 2 ∧ w = 6 ∧ b = 2 →
  (number_of_distinct_orbits : ℕ) = 1 :=
by
  -- Conditions
  intros h
  -- Translation of conditions into a useful form
  let num_cubes := 8
  let distinct_configurations := 1
  -- Burnside's Lemma applied to find the distinct configurations
  sorry

end number_of_distinct_cubes_l1029_102909


namespace mary_initial_amount_l1029_102942

theorem mary_initial_amount (current_amount pie_cost mary_after_pie : ℕ) 
  (h1 : pie_cost = 6) 
  (h2 : mary_after_pie = 52) :
  current_amount = pie_cost + mary_after_pie → 
  current_amount = 58 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end mary_initial_amount_l1029_102942


namespace gwen_received_more_money_from_mom_l1029_102980

theorem gwen_received_more_money_from_mom :
  let mom_money := 8
  let dad_money := 5
  mom_money - dad_money = 3 :=
by
  sorry

end gwen_received_more_money_from_mom_l1029_102980


namespace constant_a_value_l1029_102931

theorem constant_a_value (S : ℕ → ℝ)
  (a : ℝ)
  (h : ∀ n : ℕ, S n = 3 ^ (n + 1) + a) :
  a = -3 :=
sorry

end constant_a_value_l1029_102931


namespace sum_of_remainders_l1029_102973

theorem sum_of_remainders (a b c d : ℕ)
  (ha : a % 17 = 3) (hb : b % 17 = 5) (hc : c % 17 = 7) (hd : d % 17 = 9) :
  (a + b + c + d) % 17 = 7 :=
by
  sorry

end sum_of_remainders_l1029_102973


namespace ellipse_a_plus_k_l1029_102953

theorem ellipse_a_plus_k (f1 f2 p : Real × Real) (a b h k : Real) :
  f1 = (2, 0) →
  f2 = (-2, 0) →
  p = (5, 3) →
  (∀ x y, ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) →
  a > 0 →
  b > 0 →
  h = 0 →
  k = 0 →
  a = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 →
  a + k = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 :=
by
  intros
  sorry

end ellipse_a_plus_k_l1029_102953


namespace B_and_D_know_their_grades_l1029_102969

-- Define the students and their respective grades
inductive Grade : Type
| excellent : Grade
| good : Grade

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the information given in the problem regarding which student sees whose grade
def sees (s1 s2 : Student) : Prop :=
  (s1 = Student.A ∧ (s2 = Student.B ∨ s2 = Student.C)) ∨
  (s1 = Student.B ∧ s2 = Student.C) ∨
  (s1 = Student.D ∧ s2 = Student.A)

-- Define the condition that there are 2 excellent and 2 good grades
def grade_distribution (gA gB gC gD : Grade) : Prop :=
  gA ≠ gB → (gC = gA ∨ gC = gB) ∧ (gD = gA ∨ gD = gB) ∧
  (gA = Grade.excellent ∧ (gB = Grade.good ∨ gC = Grade.good ∨ gD = Grade.good)) ∧
  (gA = Grade.good ∧ (gB = Grade.excellent ∨ gC = Grade.excellent ∨ gD = Grade.excellent))

-- Student A's statement after seeing B and C's grades
def A_statement (gA gB gC : Grade) : Prop :=
  (gB = gA ∨ gC = gA) ∨ (gB ≠ gA ∧ gC ≠ gA)

-- Formal proof goal: Prove that B and D can know their own grades based on the information provided
theorem B_and_D_know_their_grades (gA gB gC gD : Grade)
  (h1 : grade_distribution gA gB gC gD)
  (h2 : A_statement gA gB gC)
  (h3 : sees Student.A Student.B)
  (h4 : sees Student.A Student.C)
  (h5 : sees Student.B Student.C)
  (h6 : sees Student.D Student.A) :
  (gB ≠ Grade.excellent ∨ gB ≠ Grade.good) ∧ (gD ≠ Grade.excellent ∨ gD ≠ Grade.good) :=
by sorry

end B_and_D_know_their_grades_l1029_102969


namespace inverse_square_variation_l1029_102967

theorem inverse_square_variation (k : ℝ) (y x : ℝ) (h1: x = k / y^2) (h2: 0.25 = k / 36) : 
  x = 1 :=
by
  -- Here, you would provide further Lean code to complete the proof
  -- using the given hypothesis h1 and h2, along with some computation.
  sorry

end inverse_square_variation_l1029_102967


namespace sequence_value_2_l1029_102941

/-- 
Given the following sequence:
1 = 6
3 = 18
4 = 24
5 = 30

The sequence follows the pattern that for all n ≠ 6, n is mapped to n * 6.
Prove that the value of the 2nd term in the sequence is 12.
-/

theorem sequence_value_2 (a : ℕ → ℕ) 
  (h1 : a 1 = 6) 
  (h3 : a 3 = 18) 
  (h4 : a 4 = 24) 
  (h5 : a 5 = 30) 
  (h_pattern : ∀ n, n ≠ 6 → a n = n * 6) :
  a 2 = 12 :=
by
  sorry

end sequence_value_2_l1029_102941


namespace value_of_f_2011_l1029_102968

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_2011 (h_even : ∀ x : ℝ, f x = f (-x))
                       (h_sym : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f (2 + x) = f (2 - x))
                       (h_def : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f x = 2^x) : 
  f 2011 = 1 / 2 := 
sorry

end value_of_f_2011_l1029_102968


namespace ball_box_problem_l1029_102901

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l1029_102901


namespace polygon_has_9_diagonals_has_6_sides_l1029_102994

theorem polygon_has_9_diagonals_has_6_sides :
  ∀ (n : ℕ), (∃ D : ℕ, D = n * (n - 3) / 2 ∧ D = 9) → n = 6 := 
by
  sorry

end polygon_has_9_diagonals_has_6_sides_l1029_102994


namespace inequality_system_solution_l1029_102988

theorem inequality_system_solution (x : ℝ) (h1 : 5 - 2 * x ≤ 1) (h2 : x - 4 < 0) : 2 ≤ x ∧ x < 4 :=
  sorry

end inequality_system_solution_l1029_102988


namespace angle_C_max_sum_of_sides_l1029_102907

theorem angle_C (a b c : ℝ) (S : ℝ) (h1 : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.pi / 3 :=
by
  sorry

theorem max_sum_of_sides (a b : ℝ) (c : ℝ) (hC : c = Real.sqrt 3) :
  (a + b) ≤ 2 * Real.sqrt 3 :=
by
  sorry

end angle_C_max_sum_of_sides_l1029_102907


namespace range_of_m_if_p_range_of_m_if_p_and_q_l1029_102978

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  (3 - m > m - 1) ∧ (m - 1 > 0)

def proposition_q (m : ℝ) : Prop :=
  m^2 - 9 / 4 < 0

theorem range_of_m_if_p (m : ℝ) (hp : proposition_p m) : 1 < m ∧ m < 2 :=
  sorry

theorem range_of_m_if_p_and_q (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : 1 < m ∧ m < 3 / 2 :=
  sorry

end range_of_m_if_p_range_of_m_if_p_and_q_l1029_102978


namespace problem_statement_negation_statement_l1029_102906

variable {a b : ℝ}

theorem problem_statement (h : a * b ≤ 0) : a ≤ 0 ∨ b ≤ 0 :=
sorry

theorem negation_statement (h : a * b > 0) : a > 0 ∧ b > 0 :=
sorry

end problem_statement_negation_statement_l1029_102906


namespace find_x_y_l1029_102925

theorem find_x_y (x y : ℤ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 :=
by
  sorry

end find_x_y_l1029_102925


namespace inequality_holds_l1029_102952

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > c) : (a - b) * |c - b| > 0 :=
sorry

end inequality_holds_l1029_102952


namespace sum_of_reciprocals_of_roots_l1029_102904

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 + r2 = 17) (h2 : r1 * r2 = 8) :
  1 / r1 + 1 / r2 = 17 / 8 :=
by
  sorry

end sum_of_reciprocals_of_roots_l1029_102904


namespace natalie_needs_12_bushes_for_60_zucchinis_l1029_102933

-- Definitions based on problem conditions
def bushes_to_containers (bushes : ℕ) : ℕ := bushes * 10
def containers_to_zucchinis (containers : ℕ) : ℕ := (containers * 3) / 6

-- Theorem statement
theorem natalie_needs_12_bushes_for_60_zucchinis : 
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) = 60 ∧ bushes = 12 := by
  sorry

end natalie_needs_12_bushes_for_60_zucchinis_l1029_102933


namespace mrs_hilt_bakes_loaves_l1029_102939

theorem mrs_hilt_bakes_loaves :
  let total_flour := 5
  let flour_per_loaf := 2.5
  (total_flour / flour_per_loaf) = 2 := 
by
  sorry

end mrs_hilt_bakes_loaves_l1029_102939


namespace tom_initial_game_count_zero_l1029_102954

theorem tom_initial_game_count_zero
  (batman_game_cost superman_game_cost total_expenditure initial_game_count : ℝ)
  (h_batman_cost : batman_game_cost = 13.60)
  (h_superman_cost : superman_game_cost = 5.06)
  (h_total_expenditure : total_expenditure = 18.66)
  (h_initial_game_cost : initial_game_count = total_expenditure - (batman_game_cost + superman_game_cost)) :
  initial_game_count = 0 :=
by
  sorry

end tom_initial_game_count_zero_l1029_102954


namespace find_z_l1029_102912

variable (x y z : ℝ)

theorem find_z (h1 : 12 * 40 = 480)
    (h2 : 15 * 50 = 750)
    (h3 : x + y + z = 270)
    (h4 : x + y = 100) :
    z = 170 := by
  sorry

end find_z_l1029_102912


namespace completion_time_C_l1029_102964

theorem completion_time_C (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3) 
  (h2 : r_B + r_C = 1 / 3) 
  (h3 : r_A + r_C = 1 / 3) :
  1 / r_C = 6 :=
by
  sorry

end completion_time_C_l1029_102964


namespace total_cost_of_hotel_stay_l1029_102991

-- Define the necessary conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- State the problem
theorem total_cost_of_hotel_stay :
  (cost_per_night_per_person * number_of_people * number_of_nights) = 360 := by
  sorry

end total_cost_of_hotel_stay_l1029_102991


namespace age_in_1900_l1029_102903

theorem age_in_1900 
  (x y : ℕ)
  (H1 : y = 29 * x)
  (H2 : 1901 ≤ y + x ∧ y + x ≤ 1930) :
  1900 - y = 44 := 
sorry

end age_in_1900_l1029_102903


namespace matt_books_second_year_l1029_102918

-- Definitions based on the conditions
variables (M : ℕ) -- number of books Matt read last year
variables (P : ℕ) -- number of books Pete read last year

-- Pete read twice as many books as Matt last year
def pete_read_last_year (M : ℕ) : ℕ := 2 * M

-- This year, Pete doubles the number of books he read last year
def pete_read_this_year (M : ℕ) : ℕ := 2 * (2 * M)

-- Matt reads 50% more books this year than he did last year
def matt_read_this_year (M : ℕ) : ℕ := M + M / 2

-- Pete read 300 books across both years
def total_books_pete_read_last_and_this_year (M : ℕ) : ℕ :=
  pete_read_last_year M + pete_read_this_year M

-- Prove that Matt read 75 books in his second year
theorem matt_books_second_year (M : ℕ) (h : total_books_pete_read_last_and_this_year M = 300) :
  matt_read_this_year M = 75 :=
by sorry

end matt_books_second_year_l1029_102918


namespace squirrels_more_than_nuts_l1029_102966

theorem squirrels_more_than_nuts (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : nuts = 2) : squirrels - nuts = 2 := by
  sorry

end squirrels_more_than_nuts_l1029_102966


namespace graduation_problem_l1029_102908

def valid_xs : List ℕ :=
  [10, 12, 15, 18, 20, 24, 30]

noncomputable def sum_valid_xs (l : List ℕ) : ℕ :=
  l.foldr (λ x sum => x + sum) 0

theorem graduation_problem :
  sum_valid_xs valid_xs = 129 :=
by
  sorry

end graduation_problem_l1029_102908


namespace sufficient_but_not_necessary_condition_l1029_102944

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : x > 2 ∨ x < -1 :=
by
  sorry

example (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : (x > 2) ∨ (x < -1) := 
by 
  apply sufficient_but_not_necessary_condition; exact h

end sufficient_but_not_necessary_condition_l1029_102944


namespace marcella_pairs_l1029_102958

theorem marcella_pairs (pairs_initial : ℕ) (shoes_lost : ℕ) (h1 : pairs_initial = 50) (h2 : shoes_lost = 15) :
  ∃ pairs_left : ℕ, pairs_left = 35 := 
by
  existsi 35
  sorry

end marcella_pairs_l1029_102958


namespace max_ab_l1029_102961

theorem max_ab (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : 3 * a + 2 * b = 2) :
  ab ≤ 1 / 6 :=
sorry

end max_ab_l1029_102961


namespace percentage_increase_is_correct_l1029_102922

-- Define the original and new weekly earnings
def original_earnings : ℕ := 60
def new_earnings : ℕ := 90

-- Define the percentage increase calculation
def percentage_increase (original new : ℕ) : Rat := ((new - original) / original: Rat) * 100

-- State the theorem that the percentage increase is 50%
theorem percentage_increase_is_correct : percentage_increase original_earnings new_earnings = 50 := 
sorry

end percentage_increase_is_correct_l1029_102922


namespace part1_part2_l1029_102959

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a * sin A * sin B + b * cos A^2 = 4 / 3 * a)
variable (h2 : c^2 = a^2 + (1 / 4) * b^2)

theorem part1 : b = 4 / 3 * a := by sorry

theorem part2 : C = π / 3 := by sorry

end part1_part2_l1029_102959


namespace perpendicular_lines_parallel_lines_l1029_102915

-- Define the lines l1 and l2 in terms of a
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + a ^ 2 - 1 = 0

-- Define the perpendicular condition
def perp (a : ℝ) : Prop :=
  a * 1 + 2 * (a - 1) = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop :=
  a / 1 = 2 / (a - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perp a → a = 2 / 3 := by
  intro h
  sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 := by
  intro h
  sorry

end perpendicular_lines_parallel_lines_l1029_102915


namespace rectangular_solid_volume_l1029_102921

theorem rectangular_solid_volume
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : b = 2 * a) :
  a * b * c = 12 := 
by
  sorry

end rectangular_solid_volume_l1029_102921


namespace original_price_of_shoes_l1029_102983

theorem original_price_of_shoes (P : ℝ) (h1 : 0.25 * P = 51) : P = 204 := 
by 
  sorry

end original_price_of_shoes_l1029_102983


namespace unique_common_element_l1029_102971

variable (A B : Set ℝ)
variable (a : ℝ)

theorem unique_common_element :
  A = {1, 3, a} → 
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := 
by
  intro hA hB hAB
  sorry

end unique_common_element_l1029_102971


namespace inequality1_inequality2_l1029_102974

variable (a b c d : ℝ)

theorem inequality1 : 
  (a + c)^2 * (b + d)^2 ≥ 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) :=
  sorry

theorem inequality2 : 
  (a + c)^2 * (b + d)^2 ≥ 4 * b * c * (c * d + d * a + a * b) :=
  sorry

end inequality1_inequality2_l1029_102974


namespace sum_gn_eq_one_third_l1029_102949

noncomputable def g (n : ℕ) : ℝ :=
  ∑' i : ℕ, if i ≥ 3 then 1 / (i ^ n) else 0

theorem sum_gn_eq_one_third :
  (∑' n : ℕ, if n ≥ 3 then g n else 0) = 1 / 3 := 
by sorry

end sum_gn_eq_one_third_l1029_102949


namespace tina_took_away_2_oranges_l1029_102981

-- Definition of the problem
def oranges_taken_away (x : ℕ) : Prop :=
  let original_oranges := 5
  let tangerines_left := 17 - 10 
  let oranges_left := original_oranges - x
  tangerines_left = oranges_left + 4 

-- The statement that needs to be proven
theorem tina_took_away_2_oranges : oranges_taken_away 2 :=
  sorry

end tina_took_away_2_oranges_l1029_102981
