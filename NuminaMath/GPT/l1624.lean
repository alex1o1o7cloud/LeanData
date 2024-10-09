import Mathlib

namespace number_of_valid_arithmetic_sequences_l1624_162479

theorem number_of_valid_arithmetic_sequences : 
  ∃ S : Finset (Finset ℕ), 
  S.card = 16 ∧ 
  ∀ s ∈ S, s.card = 3 ∧ 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ s = {a, b, c} ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
  (b - a = c - b) ∧ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) := 
sorry

end number_of_valid_arithmetic_sequences_l1624_162479


namespace geom_seq_a_n_l1624_162483

theorem geom_seq_a_n (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -1) 
  (h_a7 : a 7 = -9) :
  a 5 = -3 :=
sorry

end geom_seq_a_n_l1624_162483


namespace larger_number_of_two_l1624_162478

theorem larger_number_of_two
  (HCF : ℕ)
  (factor1 : ℕ)
  (factor2 : ℕ)
  (cond_HCF : HCF = 23)
  (cond_factor1 : factor1 = 15)
  (cond_factor2 : factor2 = 16) :
  ∃ (A : ℕ), A = 23 * 16 := by
  sorry

end larger_number_of_two_l1624_162478


namespace geometric_sequence_sixth_term_correct_l1624_162431

noncomputable def geometric_sequence_sixth_term (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r)
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : ℝ :=
  a * r^5

theorem geometric_sequence_sixth_term_correct (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r) 
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : geometric_sequence_sixth_term a r pos_a pos_r third_term ninth_term = 9 := 
sorry

end geometric_sequence_sixth_term_correct_l1624_162431


namespace cone_volume_l1624_162413

theorem cone_volume :
  ∀ (l h : ℝ) (r : ℝ), l = 15 ∧ h = 9 ∧ h = 3 * r → 
  (1 / 3) * Real.pi * r^2 * h = 27 * Real.pi :=
by
  intros l h r
  intro h_eqns
  sorry

end cone_volume_l1624_162413


namespace remainder_of_9_pow_333_div_50_l1624_162468

theorem remainder_of_9_pow_333_div_50 : (9 ^ 333) % 50 = 29 :=
by
  sorry

end remainder_of_9_pow_333_div_50_l1624_162468


namespace three_pow_sub_two_pow_prime_power_prime_l1624_162433

theorem three_pow_sub_two_pow_prime_power_prime (n : ℕ) (hn : n > 0) (hp : ∃ p k : ℕ, Nat.Prime p ∧ 3^n - 2^n = p^k) : Nat.Prime n := 
sorry

end three_pow_sub_two_pow_prime_power_prime_l1624_162433


namespace total_distance_maria_l1624_162414

theorem total_distance_maria (D : ℝ)
  (half_dist : D/2 + (D/2 - D/8) + 180 = D) :
  3 * D / 8 = 180 → 
  D = 480 :=
by
  sorry

end total_distance_maria_l1624_162414


namespace part1_part2_l1624_162456

-- Definition of the quadratic equation and its real roots condition
def quadratic_has_real_roots (k : ℝ) : Prop :=
  let Δ := (2 * k - 1)^2 - 4 * (k^2 - 1)
  Δ ≥ 0

-- Proving part (1): The range of real number k
theorem part1 (k : ℝ) (hk : quadratic_has_real_roots k) : k ≤ 5 / 4 := 
  sorry

-- Definition using the given condition in part (2)
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 16 + x₁ * x₂

-- Sum and product of roots of the quadratic equation
theorem part2 (k : ℝ) (h : quadratic_has_real_roots k) 
  (hx_sum : ∃ x₁ x₂ : ℝ, x₁ + x₂ = 1 - 2 * k ∧ x₁ * x₂ = k^2 - 1 ∧ roots_condition x₁ x₂) : k = -2 :=
  sorry

end part1_part2_l1624_162456


namespace percentage_students_qualified_school_A_l1624_162490

theorem percentage_students_qualified_school_A 
  (A Q : ℝ)
  (h1 : 1.20 * A = A + 0.20 * A)
  (h2 : 1.50 * Q = Q + 0.50 * Q)
  (h3 : (1.50 * Q / 1.20 * A) * 100 = 87.5) :
  (Q / A) * 100 = 58.33 := sorry

end percentage_students_qualified_school_A_l1624_162490


namespace find_c_l1624_162488

theorem find_c (a b c : ℝ) (h1 : a * 2 = 3 * b / 2) (h2 : a * 2 + 9 = c) (h3 : 4 - 3 * b = -c) : 
  c = 12 :=
by
  sorry

end find_c_l1624_162488


namespace slope_of_line_l1624_162443

theorem slope_of_line (x y : ℝ) : (4 * y = 5 * x - 20) → (y = (5/4) * x - 5) :=
by
  intro h
  sorry

end slope_of_line_l1624_162443


namespace find_integer_pairs_l1624_162430

theorem find_integer_pairs :
  ∀ x y : ℤ, x^2 = 2 + 6 * y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) :=
by {
  sorry
}

end find_integer_pairs_l1624_162430


namespace total_limes_picked_l1624_162437

theorem total_limes_picked (Alyssa_limes Mike_limes : ℕ) 
        (hAlyssa : Alyssa_limes = 25) (hMike : Mike_limes = 32) : 
       Alyssa_limes + Mike_limes = 57 :=
by {
  sorry
}

end total_limes_picked_l1624_162437


namespace max_students_total_l1624_162429

def max_students_class (a b : ℕ) (h : 3 * a + 5 * b = 115) : ℕ :=
  a + b

theorem max_students_total :
  ∃ a b : ℕ, 3 * a + 5 * b = 115 ∧ max_students_class a b (by sorry) = 37 :=
sorry

end max_students_total_l1624_162429


namespace inequality_x2_8_over_xy_y2_l1624_162494

open Real

theorem inequality_x2_8_over_xy_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^2 + 8 / (x * y) + y^2 ≥ 8 := 
sorry

end inequality_x2_8_over_xy_y2_l1624_162494


namespace shopkeeper_loss_percent_l1624_162449

noncomputable def loss_percentage (cost_price profit_percent theft_percent: ℝ) :=
  let selling_price := cost_price * (1 + profit_percent / 100)
  let value_lost := cost_price * (theft_percent / 100)
  let remaining_cost_price := cost_price * (1 - theft_percent / 100)
  (value_lost / remaining_cost_price) * 100

theorem shopkeeper_loss_percent
  (cost_price : ℝ)
  (profit_percent : ℝ := 10)
  (theft_percent : ℝ := 20)
  (expected_loss_percent : ℝ := 25)
  (h1 : profit_percent = 10) (h2 : theft_percent = 20) : 
  loss_percentage cost_price profit_percent theft_percent = expected_loss_percent := 
by
  sorry

end shopkeeper_loss_percent_l1624_162449


namespace inequality_solution_range_4_l1624_162459

theorem inequality_solution_range_4 (a : ℝ) : 
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
sorry

end inequality_solution_range_4_l1624_162459


namespace math_problem_l1624_162407

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l1624_162407


namespace sum_of_exterior_angles_of_triangle_l1624_162441

theorem sum_of_exterior_angles_of_triangle
  {α β γ α' β' γ' : ℝ} 
  (h1 : α + β + γ = 180)
  (h2 : α + α' = 180)
  (h3 : β + β' = 180)
  (h4 : γ + γ' = 180) :
  α' + β' + γ' = 360 := 
by 
sorry

end sum_of_exterior_angles_of_triangle_l1624_162441


namespace largest_number_l1624_162497

theorem largest_number (a b c : ℝ) (h1 : a + b + c = 67) (h2 : c - b = 7) (h3 : b - a = 5) : c = 86 / 3 := 
by sorry

end largest_number_l1624_162497


namespace square_area_problem_l1624_162469

theorem square_area_problem 
  (BM : ℝ) 
  (ABCD_is_divided : Prop)
  (hBM : BM = 4)
  (hABCD_is_divided : ABCD_is_divided) : 
  ∃ (side_length : ℝ), side_length * side_length = 144 := 
by
-- We skip the proof part for this task
sorry

end square_area_problem_l1624_162469


namespace percentage_fescue_in_Y_l1624_162412

-- Define the seed mixtures and their compositions
structure SeedMixture :=
  (ryegrass : ℝ)  -- percentage of ryegrass

-- Seed mixture X
def X : SeedMixture := { ryegrass := 0.40 }

-- Seed mixture Y
def Y : SeedMixture := { ryegrass := 0.25 }

-- Mixture of X and Y contains 32 percent ryegrass
def mixture_percentage := 0.32

-- 46.67 percent of the weight of this mixture is X
def weight_X := 0.4667

-- Question: What percent of seed mixture Y is fescue
theorem percentage_fescue_in_Y : (1 - Y.ryegrass) = 0.75 := by
  sorry

end percentage_fescue_in_Y_l1624_162412


namespace div_condition_l1624_162423

theorem div_condition
  (a b : ℕ)
  (h₁ : a < 1000)
  (h₂ : b ≠ 0)
  (h₃ : b ∣ a ^ 21)
  (h₄ : b ^ 10 ∣ a ^ 21) :
  b ∣ a ^ 2 :=
sorry

end div_condition_l1624_162423


namespace proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l1624_162418

variables (G : Type) [Group G] (kidney testis liver : G)
variables (SudanIII gentianViolet JanusGreenB dissociationFixative : G)

-- Conditions c1, c2, c3
def c1 : Prop := True -- Meiosis occurs in gonads, we simplify this in Lean to a true condition for brevity
def c2 : Prop := True -- Steps for slide preparation
def c3 : Prop := True -- Materials available

-- Questions
def q1 : G := testis
def q2 : G := dissociationFixative
def q3 : G := gentianViolet
def q4 : List G := [kidney, dissociationFixative, gentianViolet] -- Assume these are placeholders for correct cell types

-- Answers
def a1 : G := testis
def a2 : G := dissociationFixative
def a3 : G := gentianViolet
def a4 : List G := [testis, dissociationFixative, gentianViolet] -- Correct cells

-- Proving the equivalence of questions and answers given the conditions
theorem proof_q1_a1 : c1 ∧ c2 ∧ c3 → q1 = a1 := 
by sorry

theorem proof_q2_a2 : c1 ∧ c2 ∧ c3 → q2 = a2 := 
by sorry

theorem proof_q3_a3 : c1 ∧ c2 ∧ c3 → q3 = a3 := 
by sorry

theorem proof_q4_a4 : c1 ∧ c2 ∧ c3 → q4 = a4 := 
by sorry

end proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l1624_162418


namespace second_offset_l1624_162464

theorem second_offset (d : ℝ) (h1 : ℝ) (A : ℝ) (h2 : ℝ) : 
  d = 28 → h1 = 9 → A = 210 → h2 = 6 :=
by
  sorry

end second_offset_l1624_162464


namespace find_number_l1624_162432

theorem find_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end find_number_l1624_162432


namespace reversible_triangle_inequality_l1624_162489

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def reversible_triangle (a b c : ℝ) : Prop :=
  (is_triangle a b c) ∧ 
  (is_triangle (1 / a) (1 / b) (1 / c)) ∧
  (a ≤ b) ∧ (b ≤ c)

theorem reversible_triangle_inequality {a b c : ℝ} (h : reversible_triangle a b c) :
  a > (3 - Real.sqrt 5) / 2 * c :=
sorry

end reversible_triangle_inequality_l1624_162489


namespace average_weight_increase_l1624_162450

theorem average_weight_increase (A : ℝ) :
  let initial_weight := 8 * A
  let new_weight := initial_weight - 65 + 89
  let new_average := new_weight / 8
  let increase := new_average - A
  increase = (89 - 65) / 8 := 
by 
  sorry

end average_weight_increase_l1624_162450


namespace roots_sum_of_squares_l1624_162487

noncomputable def proof_problem (p q r : ℝ) : Prop :=
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 598

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h1 : p + q + r = 18)
  (h2 : p * q + q * r + r * p = 25)
  (h3 : p * q * r = 6) :
  proof_problem p q r :=
by {
  -- Solution steps here (omitted; not needed for the task)
  sorry
}

end roots_sum_of_squares_l1624_162487


namespace range_of_m_l1624_162404

noncomputable def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 - 4 * x + 1 = 0

theorem range_of_m (m : ℝ) (h : intersects_x_axis m) : m ≤ 4 := by
  sorry

end range_of_m_l1624_162404


namespace percentage_defective_meters_l1624_162451

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (percentage : ℚ) :
  total_meters = 2500 →
  defective_meters = 2 →
  percentage = (defective_meters / total_meters) * 100 →
  percentage = 0.08 := 
sorry

end percentage_defective_meters_l1624_162451


namespace trigonometric_identity_l1624_162474

theorem trigonometric_identity :
  (1 / Real.cos (40 * Real.pi / 180) - 2 * Real.sqrt 3 / Real.sin (40 * Real.pi / 180)) = -4 * Real.tan (20 * Real.pi / 180) := 
sorry

end trigonometric_identity_l1624_162474


namespace find_distance_l1624_162411

variable (A B : Point)
variable (distAB : ℝ) -- the distance between A and B
variable (meeting1 : ℝ) -- first meeting distance from A
variable (meeting2 : ℝ) -- second meeting distance from B

-- Conditions
axiom meeting_conditions_1 : meeting1 = 70
axiom meeting_conditions_2 : meeting2 = 90

-- Prove the distance between A and B is 120 km
def distance_from_A_to_B : ℝ := 120

theorem find_distance : distAB = distance_from_A_to_B := 
sorry

end find_distance_l1624_162411


namespace cost_of_three_pencils_and_two_pens_l1624_162480

theorem cost_of_three_pencils_and_two_pens
  (p q : ℝ)
  (h₁ : 8 * p + 3 * q = 5.20)
  (h₂ : 2 * p + 5 * q = 4.40) :
  3 * p + 2 * q = 2.5881 :=
by
  sorry

end cost_of_three_pencils_and_two_pens_l1624_162480


namespace new_computer_price_l1624_162463

theorem new_computer_price (d : ℕ) (h : 2 * d = 560) : d + 3 * d / 10 = 364 :=
by
  sorry

end new_computer_price_l1624_162463


namespace chord_length_of_circle_l1624_162465

theorem chord_length_of_circle (x y : ℝ) :
  (x^2 + y^2 - 4 * x - 4 * y - 1 = 0) ∧ (y = x + 2) → 
  2 * Real.sqrt 7 = 2 * Real.sqrt 7 :=
by sorry

end chord_length_of_circle_l1624_162465


namespace luke_pages_lemma_l1624_162442

def number_of_new_cards : ℕ := 3
def number_of_old_cards : ℕ := 9
def cards_per_page : ℕ := 3
def total_number_of_cards := number_of_new_cards + number_of_old_cards
def total_number_of_pages := total_number_of_cards / cards_per_page

theorem luke_pages_lemma : total_number_of_pages = 4 := by
  sorry

end luke_pages_lemma_l1624_162442


namespace problem_inequality_l1624_162417

open Real

theorem problem_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_prod : x * y * z = 1) :
    1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x :=
sorry

end problem_inequality_l1624_162417


namespace my_current_age_l1624_162453

-- Definitions based on the conditions
def bro_age (x : ℕ) : ℕ := 2 * x - 5

-- Main theorem to prove that my current age is 13 given the conditions
theorem my_current_age 
  (x y : ℕ)
  (h1 : y - 5 = 2 * (x - 5))
  (h2 : (x + 8) + (y + 8) = 50) :
  x = 13 :=
sorry

end my_current_age_l1624_162453


namespace production_value_n_l1624_162435

theorem production_value_n :
  -- Definitions based on conditions:
  (∀ a b : ℝ,
    (120 * a + 120 * b) / 60 = 6 ∧
    (100 * a + 100 * b) / 30 = 30) →
  (∃ n : ℝ, 80 * 3 * (a + b) = 480 * a + n * b) →
  n = 120 :=
by
  sorry

end production_value_n_l1624_162435


namespace find_B_l1624_162424

theorem find_B (A B : ℕ) (h₁ : 6 * A + 10 * B + 2 = 77) (h₂ : A ≤ 9) (h₃ : B ≤ 9) : B = 1 := sorry

end find_B_l1624_162424


namespace symmetric_point_l1624_162400

theorem symmetric_point (x y : ℝ) : 
  (x - 2 * y + 1 = 0) ∧ (y / x * 1 / 2 = -1) → (x = -2/5 ∧ y = 4/5) :=
by 
  sorry

end symmetric_point_l1624_162400


namespace river_width_l1624_162461

theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) 
  (h1 : depth = 2) 
  (h2 : flow_rate = 4000 / 60)  -- Flow rate in meters per minute
  (h3 : volume_per_minute = 6000) :
  volume_per_minute / (flow_rate * depth) = 45 :=
by
  sorry

end river_width_l1624_162461


namespace quadratic_passing_point_calc_l1624_162439

theorem quadratic_passing_point_calc :
  (∀ (x y : ℤ), y = 2 * x ^ 2 - 3 * x + 4 → ∃ (x' y' : ℤ), x' = 2 ∧ y' = 6) →
  (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  -- The corresponding proof would follow by providing the necessary steps.
  -- For now, let's just use sorry to meet the requirement.
  sorry

end quadratic_passing_point_calc_l1624_162439


namespace hank_route_distance_l1624_162403

theorem hank_route_distance 
  (d : ℝ) 
  (h1 : ∃ t1 : ℝ, t1 = d / 70 ∧ t1 = d / 70 + 1 / 60) 
  (h2 : ∃ t2 : ℝ, t2 = d / 75 ∧ t2 = d / 75 - 1 / 60) 
  (time_diff : (d / 70 - d / 75) = 1 / 30) : 
  d = 35 :=
sorry

end hank_route_distance_l1624_162403


namespace gcd_problem_l1624_162408

-- Define the conditions
def a (d : ℕ) : ℕ := d - 3
def b (d : ℕ) : ℕ := d - 2
def c (d : ℕ) : ℕ := d - 1

-- Define the number formed by digits in the specific form
def abcd (d : ℕ) : ℕ := 1000 * a d + 100 * b d + 10 * c d + d
def dcba (d : ℕ) : ℕ := 1000 * d + 100 * c d + 10 * b d + a d

-- Summing the two numbers
def num_sum (d : ℕ) : ℕ := abcd d + dcba d

-- The GCD of all num_sum(d) where d ranges from 3 to 9
def gcd_of_nums : ℕ := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (num_sum 3) (num_sum 4)) (num_sum 5)) (num_sum 6)) (Nat.gcd (num_sum 7) (Nat.gcd (num_sum 8) (num_sum 9)))

theorem gcd_problem : gcd_of_nums = 1111 := sorry

end gcd_problem_l1624_162408


namespace find_y_l1624_162473

theorem find_y 
  (x y : ℕ) 
  (hx : x % y = 9) 
  (hxy : (x : ℝ) / y = 96.12) : y = 75 :=
sorry

end find_y_l1624_162473


namespace real_roots_condition_l1624_162458

theorem real_roots_condition (k m : ℝ) (h : m ≠ 0) : (∃ x : ℝ, x^2 + k * x + m = 0) ↔ (m ≤ k^2 / 4) :=
by
  sorry

end real_roots_condition_l1624_162458


namespace energy_calculation_l1624_162472

noncomputable def stormy_day_energy_production 
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (proportional_increase : ℝ) : ℝ :=
  proportional_increase * (energy_per_day * days * number_of_windmills)

theorem energy_calculation
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (wind_speed_proportion : ℝ)
  (stormy_day_energy_per_windmill : ℝ) (s : ℝ)
  (H1 : energy_per_day = 400) 
  (H2 : days = 2) 
  (H3 : number_of_windmills = 3) 
  (H4 : stormy_day_energy_per_windmill = s * energy_per_day)
  : stormy_day_energy_production energy_per_day days number_of_windmills s = s * (400 * 3 * 2) :=
by
  sorry

end energy_calculation_l1624_162472


namespace alicia_tax_correct_l1624_162457

theorem alicia_tax_correct :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let basic_tax_rate := 0.01
  let additional_tax_rate := 0.0075
  let basic_tax := basic_tax_rate * hourly_wage_cents
  let excess_amount_cents := (hourly_wage_dollars - 20) * 100
  let additional_tax := additional_tax_rate * excess_amount_cents
  basic_tax + additional_tax = 28.75 := 
by
  sorry

end alicia_tax_correct_l1624_162457


namespace compute_n_binom_l1624_162405

-- Definitions based on conditions
def n : ℕ := sorry  -- Assume n is a positive integer defined elsewhere
def k : ℕ := 4

-- The binomial coefficient definition
def binom (n k : ℕ) : ℕ :=
  if h₁ : k ≤ n then
    (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))
  else 0

-- The theorem to prove
theorem compute_n_binom : n * binom k 3 = 4 * n :=
by
  sorry

end compute_n_binom_l1624_162405


namespace repeating_decimal_product_as_fraction_l1624_162452

theorem repeating_decimal_product_as_fraction :
  let x := 37 / 999
  let y := 7 / 9
  x * y = 259 / 8991 := by {
    sorry
  }

end repeating_decimal_product_as_fraction_l1624_162452


namespace find_natural_number_l1624_162486

theorem find_natural_number :
  ∃ x : ℕ, (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → d2 - d1 = 4) ∧
           (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → x - d2 = 308) ∧
           x = 385 :=
by
  sorry

end find_natural_number_l1624_162486


namespace quadratic_inequality_range_l1624_162460

theorem quadratic_inequality_range (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) → a ≤ 0 :=
sorry

end quadratic_inequality_range_l1624_162460


namespace right_triangle_side_length_l1624_162447

theorem right_triangle_side_length
  (c : ℕ) (a : ℕ) (h_c : c = 13) (h_a : a = 12) :
  ∃ b : ℕ, b = 5 ∧ c^2 = a^2 + b^2 :=
by
  -- Definitions from conditions
  have h_c_square : c^2 = 169 := by rw [h_c]; norm_num
  have h_a_square : a^2 = 144 := by rw [h_a]; norm_num
  -- Prove the final result
  sorry

end right_triangle_side_length_l1624_162447


namespace christian_sue_need_more_money_l1624_162471

-- Definitions based on the given conditions
def bottle_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def christian_mowing_rate : ℕ := 5
def christian_mowing_count : ℕ := 4
def sue_walking_rate : ℕ := 2
def sue_walking_count : ℕ := 6

-- Prove that Christian and Sue will need 6 more dollars to buy the bottle of perfume
theorem christian_sue_need_more_money :
  let christian_earning := christian_mowing_rate * christian_mowing_count
  let christian_total := christian_initial + christian_earning
  let sue_earning := sue_walking_rate * sue_walking_count
  let sue_total := sue_initial + sue_earning
  let total_money := christian_total + sue_total
  50 - total_money = 6 :=
by
  sorry

end christian_sue_need_more_money_l1624_162471


namespace jesse_bananas_l1624_162455

def number_of_bananas_shared (friends : ℕ) (bananas_per_friend : ℕ) : ℕ :=
  friends * bananas_per_friend

theorem jesse_bananas :
  number_of_bananas_shared 3 7 = 21 :=
by
  sorry

end jesse_bananas_l1624_162455


namespace first_candidate_percentage_l1624_162416

-- Conditions
def total_votes : ℕ := 600
def second_candidate_votes : ℕ := 240
def first_candidate_votes : ℕ := total_votes - second_candidate_votes

-- Question and correct answer
theorem first_candidate_percentage : (first_candidate_votes * 100) / total_votes = 60 := by
  sorry

end first_candidate_percentage_l1624_162416


namespace kayak_total_until_May_l1624_162415

noncomputable def kayak_number (n : ℕ) : ℕ :=
  if n = 0 then 5
  else 3 * kayak_number (n - 1)

theorem kayak_total_until_May : kayak_number 0 + kayak_number 1 + kayak_number 2 + kayak_number 3 = 200 := by
  sorry

end kayak_total_until_May_l1624_162415


namespace equal_number_of_boys_and_girls_l1624_162485

theorem equal_number_of_boys_and_girls
  (m d M D : ℝ)
  (hm : m ≠ 0)
  (hd : d ≠ 0)
  (avg1 : M / m ≠ D / d)
  (avg2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d :=
by
  sorry

end equal_number_of_boys_and_girls_l1624_162485


namespace min_even_integers_zero_l1624_162427

theorem min_even_integers_zero (x y a b m n : ℤ)
(h1 : x + y = 28) 
(h2 : x + y + a + b = 46) 
(h3 : x + y + a + b + m + n = 64) : 
∃ e, e = 0 :=
by {
  -- The conditions assure the sums of pairs are even including x, y, a, b, m, n.
  sorry
}

end min_even_integers_zero_l1624_162427


namespace trivia_team_absentees_l1624_162428

theorem trivia_team_absentees (total_members : ℕ) (total_points : ℕ) (points_per_member : ℕ) 
  (h1 : total_members = 5) 
  (h2 : total_points = 6) 
  (h3 : points_per_member = 2) : 
  total_members - (total_points / points_per_member) = 2 := 
by 
  sorry

end trivia_team_absentees_l1624_162428


namespace no_positive_n_for_prime_expr_l1624_162422

noncomputable def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ m : ℤ, 1 < m → m < p → ¬ (m ∣ p))

theorem no_positive_n_for_prime_expr : 
  ∀ n : ℕ, 0 < n → ¬ is_prime (n^3 - 9 * n^2 + 23 * n - 17) := by
  sorry

end no_positive_n_for_prime_expr_l1624_162422


namespace sequences_of_lemon_recipients_l1624_162444

theorem sequences_of_lemon_recipients :
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  total_sequences = 759375 :=
by
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  have h : total_sequences = 759375 := by sorry
  exact h

end sequences_of_lemon_recipients_l1624_162444


namespace ball_distribution_l1624_162481

theorem ball_distribution (n m : Nat) (h_n : n = 6) (h_m : m = 2) : 
  ∃ ways, 
    (ways = 2 ^ n - (1 + n)) ∧ ways = 57 :=
by
  sorry

end ball_distribution_l1624_162481


namespace range_of_m_for_circle_l1624_162484

theorem range_of_m_for_circle (m : ℝ) :
  (∃ x y, x^2 + y^2 - 4 * x - 2 * y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_for_circle_l1624_162484


namespace triangle_perimeter_l1624_162492

theorem triangle_perimeter (A r p : ℝ) (hA : A = 60) (hr : r = 2.5) (h_eq : A = r * p / 2) : p = 48 := 
by
  sorry

end triangle_perimeter_l1624_162492


namespace watch_hands_angle_120_l1624_162475

theorem watch_hands_angle_120 (n : ℝ) (h₁ : 0 ≤ n ∧ n ≤ 60) 
    (h₂ : abs ((210 + n / 2) - 6 * n) = 120) : n = 43.64 := sorry

end watch_hands_angle_120_l1624_162475


namespace luke_base_points_per_round_l1624_162426

theorem luke_base_points_per_round
    (total_score : ℕ)
    (rounds : ℕ)
    (bonus : ℕ)
    (penalty : ℕ)
    (adjusted_total : ℕ) :
    total_score = 370 → rounds = 5 → bonus = 50 → penalty = 30 → adjusted_total = total_score + bonus - penalty → (adjusted_total / rounds) = 78 :=
by
  intros
  sorry

end luke_base_points_per_round_l1624_162426


namespace prob1_prob2_prob3_l1624_162410

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 + 2
  else x

theorem prob1 :
  (∀ x, x ≥ 0 → f x = x^2 + 2) ∧
  (∀ x, x < 0 → f x = x) :=
by
  sorry

theorem prob2 : f 5 = 27 :=
by 
  sorry

theorem prob3 : ∀ (x : ℝ), f x = 0 → false :=
by
  sorry

end prob1_prob2_prob3_l1624_162410


namespace cos_half_alpha_l1624_162454

open Real -- open the Real namespace for convenience

theorem cos_half_alpha {α : ℝ} (h1 : cos α = 1 / 5) (h2 : 0 < α ∧ α < π) :
  cos (α / 2) = sqrt (15) / 5 :=
by
  sorry -- Proof is omitted

end cos_half_alpha_l1624_162454


namespace job_completion_time_l1624_162491

theorem job_completion_time (x : ℤ) (hx : (4 : ℝ) / x + (2 : ℝ) / 3 = 1) : x = 12 := by
  sorry

end job_completion_time_l1624_162491


namespace most_stable_performance_l1624_162406

-- Define the variances for each player
def variance_A : ℝ := 0.66
def variance_B : ℝ := 0.52
def variance_C : ℝ := 0.58
def variance_D : ℝ := 0.62

-- State the theorem
theorem most_stable_performance : variance_B < variance_C ∧ variance_C < variance_D ∧ variance_D < variance_A :=
by
  -- Since we are tasked to write only the statement, the proof part is skipped.
  sorry

end most_stable_performance_l1624_162406


namespace sqrt_two_irrational_l1624_162466

def irrational (x : ℝ) := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem sqrt_two_irrational : irrational (Real.sqrt 2) := 
by 
  sorry

end sqrt_two_irrational_l1624_162466


namespace face_value_of_share_l1624_162409

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end face_value_of_share_l1624_162409


namespace sin_seven_pi_div_six_l1624_162445

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l1624_162445


namespace area_of_EFCD_l1624_162477

noncomputable def area_of_quadrilateral (AB CD altitude: ℝ) :=
  let sum_bases_half := (AB + CD) / 2
  let small_altitude := altitude / 2
  small_altitude * (sum_bases_half + CD) / 2

theorem area_of_EFCD
  (AB CD altitude : ℝ)
  (AB_len : AB = 10)
  (CD_len : CD = 24)
  (altitude_len : altitude = 15)
  : area_of_quadrilateral AB CD altitude = 153.75 :=
by
  rw [AB_len, CD_len, altitude_len]
  simp [area_of_quadrilateral]
  sorry

end area_of_EFCD_l1624_162477


namespace positive_number_l1624_162419

theorem positive_number (x : ℝ) (h1 : 0 < x) (h2 : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := sorry

end positive_number_l1624_162419


namespace product_value_l1624_162499

theorem product_value : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := 
by
  sorry

end product_value_l1624_162499


namespace smallest_palindromic_primes_l1624_162462

def is_palindromic (n : ℕ) : Prop :=
  ∀ a b : ℕ, n = 1001 * a + 1010 * b → 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_palindromic_primes :
  ∃ n1 n2 : ℕ, 
  is_palindromic n1 ∧ is_palindromic n2 ∧ is_prime n1 ∧ is_prime n2 ∧ n1 < n2 ∧
  ∀ m : ℕ, (is_palindromic m ∧ is_prime m ∧ m < n2 → m = n1) ∧
           (is_palindromic m ∧ is_prime m ∧ m < n1 → m ≠ n2) ∧ n1 = 1221 ∧ n2 = 1441 := 
sorry

end smallest_palindromic_primes_l1624_162462


namespace exist_a_b_for_every_n_l1624_162495

theorem exist_a_b_for_every_n (n : ℕ) (hn : 0 < n) : 
  ∃ (a b : ℤ), 1 < a ∧ 1 < b ∧ a^2 + 1 = 2 * b^2 ∧ (a - b) % n = 0 := 
sorry

end exist_a_b_for_every_n_l1624_162495


namespace tent_cost_solution_l1624_162496

-- We define the prices of the tents and other relevant conditions.
def tent_costs (m n : ℕ) : Prop :=
  2 * m + 4 * n = 5200 ∧ 3 * m + n = 2800

-- Define the condition for the number of tents and constraints.
def optimal_tent_count (x : ℕ) (w : ℕ) : Prop :=
  x + (20 - x) = 20 ∧ x ≤ (20 - x) / 3 ∧ w = 600 * x + 1000 * (20 - x)

-- The main theorem to be proven in Lean.
theorem tent_cost_solution :
  ∃ m n, tent_costs m n ∧ m = 600 ∧ n = 1000 ∧
  ∃ x, optimal_tent_count x 18000 ∧ x = 5 ∧ (20 - x) = 15 :=
by
  sorry

end tent_cost_solution_l1624_162496


namespace value_of_n_l1624_162421

theorem value_of_n 
  {a b n : ℕ} (ha : a > 0) (hb : b > 0) 
  (h : (1 + b)^n = 243) : 
  n = 5 := by 
  sorry

end value_of_n_l1624_162421


namespace distance_traveled_is_correct_l1624_162434

noncomputable def speed_in_mph : ℝ := 23.863636363636363
noncomputable def seconds : ℝ := 2

-- constants for conversion
def miles_to_feet : ℝ := 5280
def hours_to_seconds : ℝ := 3600

-- speed in feet per second
noncomputable def speed_in_fps : ℝ := speed_in_mph * miles_to_feet / hours_to_seconds

-- distance traveled
noncomputable def distance : ℝ := speed_in_fps * seconds

theorem distance_traveled_is_correct : distance = 69.68 := by
  sorry

end distance_traveled_is_correct_l1624_162434


namespace gino_gave_away_l1624_162420

theorem gino_gave_away (initial_sticks given_away left_sticks : ℝ) 
  (h1 : initial_sticks = 63.0) (h2 : left_sticks = 13.0) 
  (h3 : left_sticks = initial_sticks - given_away) : 
  given_away = 50.0 :=
by
  sorry

end gino_gave_away_l1624_162420


namespace buffy_whiskers_l1624_162470

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l1624_162470


namespace plane_equation_exists_l1624_162448

noncomputable def equation_of_plane (A B C D : ℤ) (hA : A > 0) (hGCD : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) : Prop :=
∃ (x y z : ℤ),
  x = 1 ∧ y = -2 ∧ z = 2 ∧ D = -18 ∧
  (2 * x + (-3) * y + 5 * z + D = 0) ∧  -- Point (2, -3, 5) satisfies equation
  (4 * x + (-3) * y + 6 * z + D = 0) ∧  -- Point (4, -3, 6) satisfies equation
  (6 * x + (-4) * y + 8 * z + D = 0)    -- Point (6, -4, 8) satisfies equation

theorem plane_equation_exists : equation_of_plane 1 (-2) 2 (-18) (by decide) (by decide) :=
by
  -- Proof is omitted
  sorry

end plane_equation_exists_l1624_162448


namespace simplify_expression_correct_l1624_162438

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l1624_162438


namespace cos_105_degree_value_l1624_162401

noncomputable def cos105 : ℝ := Real.cos (105 * Real.pi / 180)

theorem cos_105_degree_value :
  cos105 = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  sorry

end cos_105_degree_value_l1624_162401


namespace bug_twelfth_move_l1624_162402

theorem bug_twelfth_move (Q : ℕ → ℚ)
  (hQ0 : Q 0 = 1)
  (hQ1 : Q 1 = 0)
  (hQ2 : Q 2 = 1/2)
  (h_recursive : ∀ n, Q (n + 1) = 1/2 * (1 - Q n)) :
  let m := 683
  let n := 2048
  (Nat.gcd m n = 1) ∧ (m + n = 2731) :=
by
  sorry

end bug_twelfth_move_l1624_162402


namespace Jason_4week_visits_l1624_162446

-- Definitions
def William_weekly_visits : ℕ := 2
def Jason_weekly_multiplier : ℕ := 4
def weeks_period : ℕ := 4

-- We need to prove that Jason goes to the library 32 times in 4 weeks.
theorem Jason_4week_visits : William_weekly_visits * Jason_weekly_multiplier * weeks_period = 32 := 
by sorry

end Jason_4week_visits_l1624_162446


namespace total_paintable_area_correct_l1624_162440

-- Bedroom dimensions and unoccupied wall space
def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 12
def bedroom1_height : ℕ := 9
def bedroom1_unoccupied : ℕ := 70

def bedroom2_length : ℕ := 12
def bedroom2_width : ℕ := 11
def bedroom2_height : ℕ := 9
def bedroom2_unoccupied : ℕ := 65

def bedroom3_length : ℕ := 13
def bedroom3_width : ℕ := 12
def bedroom3_height : ℕ := 9
def bedroom3_unoccupied : ℕ := 68

-- Total paintable area calculation
def calculate_paintable_area (length width height unoccupied : ℕ) : ℕ :=
  2 * (length * height + width * height) - unoccupied

-- Total paintable area of all bedrooms
def total_paintable_area : ℕ :=
  calculate_paintable_area bedroom1_length bedroom1_width bedroom1_height bedroom1_unoccupied +
  calculate_paintable_area bedroom2_length bedroom2_width bedroom2_height bedroom2_unoccupied +
  calculate_paintable_area bedroom3_length bedroom3_width bedroom3_height bedroom3_unoccupied

theorem total_paintable_area_correct : 
  total_paintable_area = 1129 :=
by
  unfold total_paintable_area
  unfold calculate_paintable_area
  norm_num
  sorry

end total_paintable_area_correct_l1624_162440


namespace zahra_kimmie_money_ratio_l1624_162498

theorem zahra_kimmie_money_ratio (KimmieMoney ZahraMoney : ℕ) (hKimmie : KimmieMoney = 450)
  (totalSavings : ℕ) (hSaving : totalSavings = 375)
  (h : KimmieMoney / 2 + ZahraMoney / 2 = totalSavings) :
  ZahraMoney / KimmieMoney = 2 / 3 :=
by
  -- Conditions to be used in the proof, but skipped for now
  sorry

end zahra_kimmie_money_ratio_l1624_162498


namespace number_divisible_by_11_l1624_162425

theorem number_divisible_by_11 (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by
  sorry

end number_divisible_by_11_l1624_162425


namespace students_count_geometry_history_science_l1624_162436

noncomputable def number_of_students (geometry_only history_only science_only 
                                      geometry_and_history geometry_and_science : ℕ) : ℕ :=
  geometry_only + history_only + science_only

theorem students_count_geometry_history_science (geometry_total history_only science_only 
                                                 geometry_and_history geometry_and_science : ℕ) :
  geometry_total = 30 →
  geometry_and_history = 15 →
  history_only = 15 →
  geometry_and_science = 8 →
  science_only = 10 →
  number_of_students (geometry_total - geometry_and_history - geometry_and_science)
                     history_only
                     science_only = 32 :=
by
  sorry

end students_count_geometry_history_science_l1624_162436


namespace Xiaogang_shooting_probability_l1624_162467

theorem Xiaogang_shooting_probability (total_shots : ℕ) (shots_made : ℕ) (h_total : total_shots = 50) (h_made : shots_made = 38) :
  (shots_made : ℝ) / total_shots = 0.76 :=
by
  sorry

end Xiaogang_shooting_probability_l1624_162467


namespace c_investment_l1624_162493

theorem c_investment 
  (A_investment B_investment : ℝ)
  (C_share total_profit : ℝ)
  (hA : A_investment = 8000)
  (hB : B_investment = 4000)
  (hC_share : C_share = 36000)
  (h_profit : total_profit = 252000) :
  ∃ (x : ℝ), (x / 4000) / (2 + 1 + x / 4000) = (36000 / 252000) ∧ x = 2000 :=
by
  sorry

end c_investment_l1624_162493


namespace unique_arrangements_of_MOON_l1624_162482

open Nat

theorem unique_arrangements_of_MOON : 
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  factorial n / (factorial numO * factorial numM * factorial numN) = 12 :=
by
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  sorry

end unique_arrangements_of_MOON_l1624_162482


namespace solve_triangle_l1624_162476

theorem solve_triangle (a b : ℝ) (A B : ℝ) : ((A + B < π ∧ A > 0 ∧ B > 0 ∧ a > 0) ∨ (a > 0 ∧ b > 0 ∧ (π > A) ∧ (A > 0))) → ∃ c C, c > 0 ∧ (π > C) ∧ C > 0 :=
sorry

end solve_triangle_l1624_162476
