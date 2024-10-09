import Mathlib

namespace radius_of_inscribed_circle_is_three_fourths_l2106_210610

noncomputable def circle_diameter : ℝ := Real.sqrt 12

noncomputable def radius_of_new_inscribed_circle : ℝ :=
  let R := circle_diameter / 2
  let s := R * Real.sqrt 3
  let h := s * Real.sqrt 3 / 2
  let a := Real.sqrt (h^2 - (h/2)^2)
  a * Real.sqrt 3 / 6

theorem radius_of_inscribed_circle_is_three_fourths :
  radius_of_new_inscribed_circle = 3 / 4 := sorry

end radius_of_inscribed_circle_is_three_fourths_l2106_210610


namespace samantha_total_cost_l2106_210671

-- Defining the conditions in Lean
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℕ := 25
def loads : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Proving the total cost Samantha spends is $11
theorem samantha_total_cost : (loads * washer_cost + num_dryers * (dryer_time / 10 * dryer_cost_per_10_min)) = 1100 :=
by
  sorry

end samantha_total_cost_l2106_210671


namespace good_students_l2106_210657

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l2106_210657


namespace part_a_part_b_part_c_l2106_210686

def quadradois (n : ℕ) : Prop :=
  ∃ (S1 S2 : ℕ), S1 ≠ S2 ∧ (S1 * S1 + S2 * S2 ≤ S1 * S1 + S2 * S2 + (n - 2))

theorem part_a : quadradois 6 := 
sorry

theorem part_b : quadradois 2015 := 
sorry

theorem part_c : ∀ (n : ℕ), n > 5 → quadradois n := 
sorry

end part_a_part_b_part_c_l2106_210686


namespace sufficient_prime_logarithms_l2106_210653

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

-- Statement of the properties of logarithms
axiom log_mul (b x y : ℝ) : log_b b (x * y) = log_b b x + log_b b y
axiom log_div (b x y : ℝ) : log_b b (x / y) = log_b b x - log_b b y
axiom log_pow (b x : ℝ) (n : ℝ) : log_b b (x ^ n) = n * log_b b x

-- Main theorem
theorem sufficient_prime_logarithms (b : ℝ) (hb : 1 < b) :
  (∀ p : ℕ, is_prime p → ∃ Lp : ℝ, log_b b p = Lp) →
  ∀ n : ℕ, n > 0 → ∃ Ln : ℝ, log_b b n = Ln :=
by
  sorry

end sufficient_prime_logarithms_l2106_210653


namespace proof_problem_l2106_210609

theorem proof_problem (k m : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hkm : k > m)
  (hdiv : (k * m * (k ^ 2 - m ^ 2)) ∣ (k ^ 3 - m ^ 3)) :
  (k - m) ^ 3 > 3 * k * m :=
sorry

end proof_problem_l2106_210609


namespace mr_blue_carrots_l2106_210631

theorem mr_blue_carrots :
  let steps_length := 3 -- length of each step in feet
  let garden_length_steps := 25 -- length of garden in steps
  let garden_width_steps := 35 -- width of garden in steps
  let length_feet := garden_length_steps * steps_length -- length of garden in feet
  let width_feet := garden_width_steps * steps_length -- width of garden in feet
  let area_feet2 := length_feet * width_feet -- area of garden in square feet
  let yield_rate := 3 / 4 -- yield rate of carrots in pounds per square foot
  let expected_yield := area_feet2 * yield_rate -- expected yield in pounds
  expected_yield = 5906.25
:= by
  sorry

end mr_blue_carrots_l2106_210631


namespace correct_operation_is_B_l2106_210683

-- Definitions of the operations as conditions
def operation_A (x : ℝ) : Prop := 3 * x - x = 3
def operation_B (x : ℝ) : Prop := x^2 * x^3 = x^5
def operation_C (x : ℝ) : Prop := x^6 / x^2 = x^3
def operation_D (x : ℝ) : Prop := (x^2)^3 = x^5

-- Prove that the correct operation is B
theorem correct_operation_is_B (x : ℝ) : operation_B x :=
by
  show x^2 * x^3 = x^5
  sorry

end correct_operation_is_B_l2106_210683


namespace product_of_solutions_eq_zero_l2106_210613

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) → (x = 0 ∨ x = -4 / 7)) → (0 = 0) := 
by
  intros h
  sorry

end product_of_solutions_eq_zero_l2106_210613


namespace triangle_area_l2106_210628

noncomputable def area_ABC (AB BC : ℝ) (angle_B : ℝ) : ℝ :=
  1/2 * AB * BC * Real.sin angle_B

theorem triangle_area
  (A B C : Type)
  (AB : ℝ) (A_eq : ℝ) (B_eq : ℝ)
  (h_AB : AB = 6)
  (h_A : A_eq = Real.pi / 6)
  (h_B : B_eq = 2 * Real.pi / 3) :
  area_ABC AB AB (2 * Real.pi / 3) = 9 * Real.sqrt 3 :=
by
  simp [area_ABC, h_AB, h_A, h_B]
  sorry

end triangle_area_l2106_210628


namespace integer_quotient_is_perfect_square_l2106_210682

theorem integer_quotient_is_perfect_square (a b : ℕ) (h : 0 < a ∧ 0 < b) (h_int : (a + b) ^ 2 % (4 * a * b + 1) = 0) :
  ∃ k : ℕ, (a + b) ^ 2 = k ^ 2 * (4 * a * b + 1) := sorry

end integer_quotient_is_perfect_square_l2106_210682


namespace total_shots_cost_l2106_210673

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l2106_210673


namespace solve_siblings_age_problem_l2106_210647

def siblings_age_problem (x : ℕ) : Prop :=
  let age_eldest := 20
  let age_middle := 15
  let age_youngest := 10
  (age_eldest + x) + (age_middle + x) + (age_youngest + x) = 75 → x = 10

theorem solve_siblings_age_problem : siblings_age_problem 10 :=
by
  sorry

end solve_siblings_age_problem_l2106_210647


namespace gcd_of_1230_and_920_is_10_l2106_210627

theorem gcd_of_1230_and_920_is_10 : Int.gcd 1230 920 = 10 :=
sorry

end gcd_of_1230_and_920_is_10_l2106_210627


namespace find_a_l2106_210680

def setA (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def setB (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (a : ℝ) : 
  (setA a ∩ setB a = {2, 5}) → (a = -1 ∨ a = 2) :=
sorry

end find_a_l2106_210680


namespace y_intercept_of_line_l2106_210644

theorem y_intercept_of_line (m : ℝ) (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : b = 0) (h_slope : m = 3) (h_x_intercept : (a, b) = (4, 0)) :
  ∃ y : ℝ, (0, y) = (0, -12) :=
by 
  sorry

end y_intercept_of_line_l2106_210644


namespace boarders_initial_count_l2106_210661

noncomputable def initial_boarders (x : ℕ) : ℕ := 7 * x

theorem boarders_initial_count (x : ℕ) (h1 : 80 + initial_boarders x = (2 : ℝ) * 16) :
  initial_boarders x = 560 :=
by
  sorry

end boarders_initial_count_l2106_210661


namespace total_hours_until_joy_sees_grandma_l2106_210640

theorem total_hours_until_joy_sees_grandma
  (days_until_grandma: ℕ)
  (hours_in_a_day: ℕ)
  (timezone_difference: ℕ)
  (H_days : days_until_grandma = 2)
  (H_hours : hours_in_a_day = 24)
  (H_timezone : timezone_difference = 3) :
  (days_until_grandma * hours_in_a_day = 48) :=
by
  sorry

end total_hours_until_joy_sees_grandma_l2106_210640


namespace joe_total_cars_l2106_210692

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end joe_total_cars_l2106_210692


namespace sequence_bound_l2106_210630

variable {a : ℕ+ → ℝ}

theorem sequence_bound (h : ∀ k m : ℕ+, |a (k + m) - a k - a m| ≤ 1) :
    ∀ (p q : ℕ+), |a p / p - a q / q| < 1 / p + 1 / q :=
by
  sorry

end sequence_bound_l2106_210630


namespace tangent_line_at_point_l2106_210605

theorem tangent_line_at_point :
  ∀ (x y : ℝ) (h : y = x^3 - 2 * x + 1),
    ∃ (m b : ℝ), (1, 0) = (x, y) → (m = 1) ∧ (b = -1) ∧ (∀ (z : ℝ), z = m * x + b) := sorry

end tangent_line_at_point_l2106_210605


namespace arrangements_with_AB_together_l2106_210638

theorem arrangements_with_AB_together (n : ℕ) (A B: ℕ) (students: Finset ℕ) (h₁ : students.card = 6) (h₂ : A ∈ students) (h₃ : B ∈ students):
  ∃! (count : ℕ), count = 240 :=
by
  sorry

end arrangements_with_AB_together_l2106_210638


namespace area_of_triangle_hyperbola_focus_l2106_210679

theorem area_of_triangle_hyperbola_focus :
  let F₁ := (-Real.sqrt 2, 0)
  let F₂ := (Real.sqrt 2, 0)
  let hyperbola := {p : ℝ × ℝ | p.1 ^ 2 - p.2 ^ 2 = 1}
  let asymptote (p : ℝ × ℝ) := p.1 = p.2
  let circle := {p : ℝ × ℝ | (p.1 - F₁.1 / 2) ^ 2 + (p.2 - F₁.2 / 2) ^ 2 = (Real.sqrt 2) ^ 2}
  let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let area (p1 p2 p3 : ℝ × ℝ) := 0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))
  area F₁ P Q = Real.sqrt 2 := 
sorry

end area_of_triangle_hyperbola_focus_l2106_210679


namespace white_marbles_count_l2106_210655

theorem white_marbles_count (total_marbles blue_marbles red_marbles : ℕ) (probability_red_or_white : ℚ)
    (h_total : total_marbles = 60)
    (h_blue : blue_marbles = 5)
    (h_red : red_marbles = 9)
    (h_probability : probability_red_or_white = 0.9166666666666666) :
    ∃ W : ℕ, W = total_marbles - blue_marbles - red_marbles ∧ probability_red_or_white = (red_marbles + W)/(total_marbles) ∧ W = 46 :=
by
  sorry

end white_marbles_count_l2106_210655


namespace final_pens_count_l2106_210602

-- Define the initial number of pens and subsequent operations
def initial_pens : ℕ := 7
def pens_after_mike (initial : ℕ) : ℕ := initial + 22
def pens_after_cindy (pens : ℕ) : ℕ := pens * 2
def pens_after_sharon (pens : ℕ) : ℕ := pens - 19

-- Prove that the final number of pens is 39
theorem final_pens_count : pens_after_sharon (pens_after_cindy (pens_after_mike initial_pens)) = 39 := 
sorry

end final_pens_count_l2106_210602


namespace B_work_rate_l2106_210666

-- Definitions for the conditions
def A (t : ℝ) := 1 / 15 -- A's work rate per hour
noncomputable def B : ℝ := 1 / 10 - 1 / 15 -- Definition using the condition of the combined work rate

-- Lean 4 statement for the proof problem
theorem B_work_rate : B = 1 / 30 := by sorry

end B_work_rate_l2106_210666


namespace minimum_value_l2106_210639

open Real

variables {A B C M : Type}
variables (AB AC : ℝ) 
variables (S_MBC x y : ℝ)

-- Assume the given conditions
axiom dot_product_AB_AC : AB * AC = 2 * sqrt 3
axiom angle_BAC_30 : (30 : Real) = π / 6
axiom area_MBC : S_MBC = 1/2
axiom area_sum : x + y = 1/2

-- Define the minimum value problem
theorem minimum_value : 
  ∃ m, m = 18 ∧ (∀ x y, (1/x + 4/y) ≥ m) :=
sorry

end minimum_value_l2106_210639


namespace problem_solution_l2106_210656

theorem problem_solution :
  (1/3⁻¹) - Real.sqrt 27 + 3 * Real.tan (Real.pi / 6) + (Real.pi - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end problem_solution_l2106_210656


namespace additional_people_required_l2106_210685

-- Define conditions
def people := 8
def time1 := 3
def total_work := people * time1 -- This gives us the constant k

-- Define the second condition where 12 people are needed to complete in 2 hours
def required_people (t : Nat) := total_work / t

-- The number of additional people required
def additional_people := required_people 2 - people

-- State the theorem
theorem additional_people_required : additional_people = 4 :=
by 
  show additional_people = 4
  sorry

end additional_people_required_l2106_210685


namespace ab_difference_l2106_210620

theorem ab_difference (a b : ℝ) 
  (h1 : 10 = a * 3 + b)
  (h2 : 22 = a * 7 + b) : 
  a - b = 2 := 
  sorry

end ab_difference_l2106_210620


namespace fg_of_3_is_2810_l2106_210670

def f (x : ℕ) : ℕ := x^2 + 1
def g (x : ℕ) : ℕ := 2 * x^3 - 1

theorem fg_of_3_is_2810 : f (g 3) = 2810 := by
  sorry

end fg_of_3_is_2810_l2106_210670


namespace probability_of_three_black_balls_l2106_210618

def total_ball_count : ℕ := 4 + 8

def white_ball_count : ℕ := 4

def black_ball_count : ℕ := 8

def total_combinations : ℕ := Nat.choose total_ball_count 3

def black_combinations : ℕ := Nat.choose black_ball_count 3

def probability_three_black : ℚ := black_combinations / total_combinations

theorem probability_of_three_black_balls : 
  probability_three_black = 14 / 55 := 
sorry

end probability_of_three_black_balls_l2106_210618


namespace arithmetic_sequence_value_l2106_210632

theorem arithmetic_sequence_value (a : ℕ → ℕ) (m : ℕ) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 4) 
  (h_a5 : a 5 = m) 
  (h_a7 : a 7 = 16) : 
  m = 10 := 
by
  sorry

end arithmetic_sequence_value_l2106_210632


namespace smallest_y_in_geometric_sequence_l2106_210600

theorem smallest_y_in_geometric_sequence (x y z r : ℕ) (h1 : y = x * r) (h2 : z = x * r^2) (h3 : xyz = 125) : y = 5 :=
by sorry

end smallest_y_in_geometric_sequence_l2106_210600


namespace revision_cost_is_3_l2106_210668

def cost_first_time (pages : ℕ) : ℝ := 5 * pages

def cost_for_revisions (rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := (rev1 * rev_cost) + (rev2 * 2 * rev_cost)

def total_cost (pages rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := 
  cost_first_time pages + cost_for_revisions rev1 rev2 rev_cost

theorem revision_cost_is_3 :
  ∀ (pages rev1 rev2 : ℕ) (total : ℝ),
      pages = 100 →
      rev1 = 30 →
      rev2 = 20 →
      total = 710 →
      total_cost pages rev1 rev2 3 = total :=
by
  intros pages rev1 rev2 total pages_eq rev1_eq rev2_eq total_eq
  sorry

end revision_cost_is_3_l2106_210668


namespace triangle_XYZ_XY2_XZ2_difference_l2106_210665

-- Define the problem parameters and conditions
def YZ : ℝ := 10
def XM : ℝ := 6
def midpoint_YZ (M : ℝ) := 2 * M = YZ

-- The main theorem to be proved
theorem triangle_XYZ_XY2_XZ2_difference :
  ∀ (XY XZ : ℝ), 
  (∀ (M : ℝ), midpoint_YZ M) →
  ((∃ (x : ℝ), (0 ≤ x ∧ x ≤ 10) ∧ XY^2 + XZ^2 = 2 * x^2 - 20 * x + 2 * (11 * x - x^2 - 11) + 100)) →
  (120 - 100 = 20) :=
by
  sorry

end triangle_XYZ_XY2_XZ2_difference_l2106_210665


namespace sqrt_equality_l2106_210662

theorem sqrt_equality (m : ℝ) (n : ℝ) (h1 : 0 < m) (h2 : -3 * m ≤ n) (h3 : n ≤ 3 * m) :
    (Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2))
     - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2))
    = 2 * Real.sqrt (3 * m - n)) :=
sorry

end sqrt_equality_l2106_210662


namespace annual_depletion_rate_l2106_210629

theorem annual_depletion_rate
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (depletion_rate : ℝ)
  (h_initial_value : initial_value = 40000)
  (h_final_value : final_value = 36100)
  (h_time : time = 2)
  (decay_eq : final_value = initial_value * (1 - depletion_rate)^time) :
  depletion_rate = 0.05 :=
by 
  sorry

end annual_depletion_rate_l2106_210629


namespace probability_no_defective_pens_l2106_210658

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (non_defective_pens : ℕ) (prob_first_non_defective : ℚ) (prob_second_non_defective : ℚ) :
  total_pens = 12 →
  defective_pens = 4 →
  non_defective_pens = total_pens - defective_pens →
  prob_first_non_defective = non_defective_pens / total_pens →
  prob_second_non_defective = (non_defective_pens - 1) / (total_pens - 1) →
  prob_first_non_defective * prob_second_non_defective = 14 / 33 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end probability_no_defective_pens_l2106_210658


namespace jack_christina_speed_l2106_210616

noncomputable def speed_of_jack_christina (d_jack_christina : ℝ) (v_lindy : ℝ) (d_lindy : ℝ) (relative_speed_factor : ℝ := 2) : ℝ :=
d_lindy * relative_speed_factor / d_jack_christina

theorem jack_christina_speed :
  speed_of_jack_christina 240 10 400 = 3 := by
  sorry

end jack_christina_speed_l2106_210616


namespace total_money_l2106_210689

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end total_money_l2106_210689


namespace trout_to_bass_ratio_l2106_210672

theorem trout_to_bass_ratio 
  (bass : ℕ) 
  (trout : ℕ) 
  (blue_gill : ℕ)
  (h1 : bass = 32) 
  (h2 : blue_gill = 2 * bass) 
  (h3 : bass + trout + blue_gill = 104) 
  : (trout / bass) = 1 / 4 :=
by 
  -- intermediate steps can be included here
  sorry

end trout_to_bass_ratio_l2106_210672


namespace at_least_one_greater_than_16000_l2106_210694

open Nat

theorem at_least_one_greater_than_16000 (seq : Fin 20 → ℕ)
  (h_distinct : ∀ i j : Fin 20, i ≠ j → seq i ≠ seq j)
  (h_perfect_square : ∀ i : Fin 19, ∃ k : ℕ, (seq i) * (seq (i + 1)) = k^2)
  (h_first : seq 0 = 42) : ∃ i : Fin 20, seq i > 16000 :=
by
  sorry

end at_least_one_greater_than_16000_l2106_210694


namespace number_of_members_l2106_210603

theorem number_of_members (n : ℕ) (h : n^2 = 5929) : n = 77 :=
sorry

end number_of_members_l2106_210603


namespace find_set_B_l2106_210601

set_option pp.all true

variable (A : Set ℤ) (B : Set ℤ)

theorem find_set_B (hA : A = {-2, 0, 1, 3})
                    (hB : B = {x | -x ∈ A ∧ 1 - x ∉ A}) :
  B = {-3, -1, 2} :=
by
  sorry

end find_set_B_l2106_210601


namespace calc_xy_square_l2106_210691

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end calc_xy_square_l2106_210691


namespace length_of_pipe_is_correct_l2106_210633

-- Definitions of the conditions
def step_length : ℝ := 0.8
def steps_same_direction : ℤ := 210
def steps_opposite_direction : ℤ := 100

-- The distance moved by the tractor in one step
noncomputable def tractor_step_distance : ℝ := (steps_same_direction * step_length - steps_opposite_direction * step_length) / (steps_opposite_direction + steps_same_direction : ℝ)

-- The length of the pipe
noncomputable def length_of_pipe (steps_same_direction steps_opposite_direction : ℤ) (step_length : ℝ) : ℝ :=
 steps_same_direction * (step_length - tractor_step_distance)

-- Proof statement
theorem length_of_pipe_is_correct :
  length_of_pipe steps_same_direction steps_opposite_direction step_length = 108 :=
sorry

end length_of_pipe_is_correct_l2106_210633


namespace abs_a_lt_abs_b_add_abs_c_l2106_210688

theorem abs_a_lt_abs_b_add_abs_c (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_a_lt_abs_b_add_abs_c_l2106_210688


namespace line_intersects_circle_l2106_210693

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  ∃ p : ℝ × ℝ, (p.1 ^ 2 + p.2 ^ 2 = 9) ∧ (a * p.1 - p.2 + 2 * a = 0) :=
by
  sorry

end line_intersects_circle_l2106_210693


namespace relationship_of_y_values_l2106_210695

theorem relationship_of_y_values (m n y1 y2 y3 : ℝ) (h1 : m < 0) (h2 : n > 0) 
  (hA : y1 = m * (-2) + n) (hB : y2 = m * (-3) + n) (hC : y3 = m * 1 + n) :
  y3 < y1 ∧ y1 < y2 := 
by 
  sorry

end relationship_of_y_values_l2106_210695


namespace find_second_liquid_parts_l2106_210619

-- Define the given constants
def first_liquid_kerosene_percentage : ℝ := 0.25
def second_liquid_kerosene_percentage : ℝ := 0.30
def first_liquid_parts : ℝ := 6
def mixture_kerosene_percentage : ℝ := 0.27

-- Define the amount of kerosene from each liquid
def kerosene_from_first_liquid := first_liquid_kerosene_percentage * first_liquid_parts
def kerosene_from_second_liquid (x : ℝ) := second_liquid_kerosene_percentage * x

-- Define the total parts of mixture
def total_mixture_parts (x : ℝ) := first_liquid_parts + x

-- Define the total kerosene in the mixture
def total_kerosene_in_mixture (x : ℝ) := mixture_kerosene_percentage * total_mixture_parts x

-- State the theorem
theorem find_second_liquid_parts (x : ℝ) :
  kerosene_from_first_liquid + kerosene_from_second_liquid x = total_kerosene_in_mixture x → 
  x = 4 :=
by
  sorry

end find_second_liquid_parts_l2106_210619


namespace james_distance_l2106_210636

-- Definitions and conditions
def speed : ℝ := 80.0
def time : ℝ := 16.0

-- Proof problem statement
theorem james_distance : speed * time = 1280.0 := by
  sorry

end james_distance_l2106_210636


namespace walkways_area_l2106_210660

theorem walkways_area (rows cols : ℕ) (bed_length bed_width walkthrough_width garden_length garden_width total_flower_beds bed_area total_bed_area total_garden_area : ℝ) 
  (h1 : rows = 4) (h2 : cols = 3) 
  (h3 : bed_length = 8) (h4 : bed_width = 3) 
  (h5 : walkthrough_width = 2)
  (h6 : garden_length = (cols * bed_length) + ((cols + 1) * walkthrough_width))
  (h7 : garden_width = (rows * bed_width) + ((rows + 1) * walkthrough_width))
  (h8 : total_garden_area = garden_length * garden_width)
  (h9 : total_flower_beds = rows * cols)
  (h10 : bed_area = bed_length * bed_width)
  (h11 : total_bed_area = total_flower_beds * bed_area)
  (h12 : total_garden_area - total_bed_area = 416) : 
  True := 
sorry

end walkways_area_l2106_210660


namespace min_value_of_a_l2106_210641

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + 2 * a * x + 1 ≥ 0) → a ≥ -5 / 4 := 
sorry

end min_value_of_a_l2106_210641


namespace bc_together_l2106_210677

theorem bc_together (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 20) : B + C = 320 :=
by
  sorry

end bc_together_l2106_210677


namespace find_m_l2106_210652

-- Definition of the constraints and the values of x and y that satisfy them
def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y + 1 ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 3

-- Given conditions
def satisfies_constraints (x y : ℝ) : Prop := 
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x

-- The objective to prove
theorem find_m (x y m : ℝ) (h : satisfies_constraints x y) : 
  (∀ x y, satisfies_constraints x y → (- 3 = m * x + y)) → m = -2 / 3 :=
by
  sorry

end find_m_l2106_210652


namespace cost_of_one_lesson_l2106_210649

-- Define the conditions
def total_cost_for_lessons : ℝ := 360
def total_hours_of_lessons : ℝ := 18
def duration_of_one_lesson : ℝ := 1.5

-- Define the theorem statement
theorem cost_of_one_lesson :
  (total_cost_for_lessons / total_hours_of_lessons) * duration_of_one_lesson = 30 := by
  -- Proof goes here
  sorry

end cost_of_one_lesson_l2106_210649


namespace intersection_range_l2106_210623

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- State the problem: Given the line and the curve have a common point, prove the range of m is m >= 3
theorem intersection_range (k m : ℝ) (h : ∃ x y, line k x = y ∧ curve x y m) : m ≥ 3 :=
by {
  sorry
}

end intersection_range_l2106_210623


namespace rectangular_field_perimeter_l2106_210669

variable (length width : ℝ)

theorem rectangular_field_perimeter (h_area : length * width = 50) (h_width : width = 5) : 2 * (length + width) = 30 := by
  sorry

end rectangular_field_perimeter_l2106_210669


namespace complement_union_example_l2106_210663

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 4}

-- State the theorem we want to prove
theorem complement_union_example : (U \ A) ∪ B = {2, 4, 5} :=
by
  sorry

end complement_union_example_l2106_210663


namespace find_x_l2106_210651

theorem find_x (x : ℝ) (h : 5020 - (x / 100.4) = 5015) : x = 502 :=
sorry

end find_x_l2106_210651


namespace swimmers_speed_in_still_water_l2106_210659

theorem swimmers_speed_in_still_water
  (v : ℝ) -- swimmer's speed in still water
  (current_speed : ℝ) -- speed of the water current
  (time : ℝ) -- time taken to swim against the current
  (distance : ℝ) -- distance swum against the current
  (h_current_speed : current_speed = 2)
  (h_time : time = 3.5)
  (h_distance : distance = 7)
  (h_eqn : time = distance / (v - current_speed)) :
  v = 4 :=
by
  sorry

end swimmers_speed_in_still_water_l2106_210659


namespace ratio_of_areas_of_triangles_l2106_210643

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l2106_210643


namespace ratio_removing_middle_digit_l2106_210684

theorem ratio_removing_middle_digit 
  (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9)
  (h1 : 10 * b + c = 8 * a) 
  (h2 : 10 * a + b = 8 * c) : 
  (10 * a + c) / b = 17 :=
by sorry

end ratio_removing_middle_digit_l2106_210684


namespace ab_sum_l2106_210614

theorem ab_sum (a b : ℕ) (h1: (a + b) % 9 = 8) (h2: (a - b) % 11 = 7) : a + b = 8 :=
sorry

end ab_sum_l2106_210614


namespace finiteness_of_triples_l2106_210617

theorem finiteness_of_triples (x : ℚ) : ∃! (a b c : ℤ), a < 0 ∧ b^2 - 4*a*c = 5 ∧ (a*x^2 + b*x + c > 0) := sorry

end finiteness_of_triples_l2106_210617


namespace at_least_one_not_beyond_20m_l2106_210626

variables (p q : Prop)

theorem at_least_one_not_beyond_20m : (¬ p ∨ ¬ q) ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_beyond_20m_l2106_210626


namespace number_of_people_in_group_l2106_210621

theorem number_of_people_in_group (n : ℕ) (h1 : 110 - 60 = 5 * n) : n = 10 :=
by 
  sorry

end number_of_people_in_group_l2106_210621


namespace inequality_proof_l2106_210637

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z ≥ 1/x + 1/y + 1/z) : 
  x/y + y/z + z/x ≥ 1/(x * y) + 1/(y * z) + 1/(z * x) :=
by
  sorry

end inequality_proof_l2106_210637


namespace squared_difference_l2106_210690

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l2106_210690


namespace original_ratio_l2106_210667

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 :=
by sorry

end original_ratio_l2106_210667


namespace max_angle_AFB_l2106_210612

noncomputable def focus_of_parabola := (2, 0)
def parabola (x y : ℝ) := y^2 = 8 * x
def on_parabola (A B : ℝ × ℝ) := parabola A.1 A.2 ∧ parabola B.1 B.2
def condition (x1 x2 : ℝ) (AB : ℝ) := x1 + x2 + 4 = (2 * Real.sqrt 3 / 3) * AB

theorem max_angle_AFB (A B : ℝ × ℝ) (x1 x2 : ℝ) (AB : ℝ)
  (h1 : on_parabola A B)
  (h2 : condition x1 x2 AB)
  (hA : A.1 = x1)
  (hB : B.1 = x2) :
  ∃ θ, θ ≤ Real.pi * 2 / 3 := 
  sorry

end max_angle_AFB_l2106_210612


namespace quadrant_and_terminal_angle_l2106_210635

def alpha : ℝ := -1910 

noncomputable def normalize_angle (α : ℝ) : ℝ := 
  let β := α % 360
  if β < 0 then β + 360 else β

noncomputable def in_quadrant_3 (β : ℝ) : Prop :=
  180 ≤ β ∧ β < 270

noncomputable def equivalent_theta (α : ℝ) (θ : ℝ) : Prop :=
  (α % 360 = θ % 360) ∧ (-720 ≤ θ ∧ θ < 0)

theorem quadrant_and_terminal_angle :
  in_quadrant_3 (normalize_angle alpha) ∧ 
  (equivalent_theta alpha (-110) ∨ equivalent_theta alpha (-470)) :=
by 
  sorry

end quadrant_and_terminal_angle_l2106_210635


namespace tan_of_angle_123_l2106_210615

variable (a : ℝ)
variable (h : Real.sin 123 = a)

theorem tan_of_angle_123 : Real.tan 123 = a / Real.cos 123 :=
by
  sorry

end tan_of_angle_123_l2106_210615


namespace number_of_cows_l2106_210650

/-- 
The number of cows Mr. Reyansh has on his dairy farm 
given the conditions of water consumption and total water used in a week. 
-/
theorem number_of_cows (C : ℕ) 
  (h1 : ∀ (c : ℕ), (c = 80 * 7))
  (h2 : ∀ (s : ℕ), (s = 10 * C))
  (h3 : ∀ (d : ℕ), (d = 20 * 7))
  (h4 : 1960 * C = 78400) : 
  C = 40 :=
sorry

end number_of_cows_l2106_210650


namespace first_night_percentage_is_20_l2106_210697

-- Conditions
variable (total_pages : ℕ) (pages_left : ℕ)
variable (pages_second_night : ℕ)
variable (pages_third_night : ℕ)
variable (first_night_percentage : ℕ)

-- Definitions
def total_read_pages (total_pages pages_left : ℕ) : ℕ := total_pages - pages_left

def pages_first_night (total_pages first_night_percentage : ℕ) : ℕ :=
  (first_night_percentage * total_pages) / 100

def total_read_on_three_nights (total_pages pages_left pages_second_night pages_third_night first_night_percentage : ℕ) : Prop :=
  total_read_pages total_pages pages_left = pages_first_night total_pages first_night_percentage + pages_second_night + pages_third_night

-- Theorem
theorem first_night_percentage_is_20 :
  ∀ total_pages pages_left pages_second_night pages_third_night,
  total_pages = 500 →
  pages_left = 150 →
  pages_second_night = 100 →
  pages_third_night = 150 →
  total_read_on_three_nights total_pages pages_left pages_second_night pages_third_night 20 :=
by
  intros
  sorry

end first_night_percentage_is_20_l2106_210697


namespace greatest_three_digit_multiple_of_17_l2106_210678

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l2106_210678


namespace min_value_of_m_l2106_210642

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_value_of_m {m : ℝ} (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ m) : m = 5 := 
sorry

end min_value_of_m_l2106_210642


namespace opposite_of_neg_three_l2106_210634

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l2106_210634


namespace octahedron_tetrahedron_volume_ratio_l2106_210607

theorem octahedron_tetrahedron_volume_ratio (a : ℝ) :
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3
  V_o / V_t = 1 :=
by 
  -- Definitions from conditions
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3

  -- Proof omitted
  -- Proof goes here
  sorry

end octahedron_tetrahedron_volume_ratio_l2106_210607


namespace p_iff_q_l2106_210664

variables {a b c : ℝ}
def p (a b c : ℝ) : Prop := ∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0
def q (a b c : ℝ) : Prop := a + b + c = 0

theorem p_iff_q (h : a ≠ 0) : p a b c ↔ q a b c :=
sorry

end p_iff_q_l2106_210664


namespace dinner_plates_percentage_l2106_210681

/-- Define the cost of silverware and the total cost of both items -/
def silverware_cost : ℝ := 20
def total_cost : ℝ := 30

/-- Define the percentage of the silverware cost that the dinner plates cost -/
def percentage_of_silverware_cost := 50

theorem dinner_plates_percentage :
  ∃ (P : ℝ) (S : ℝ) (x : ℝ), S = silverware_cost ∧ (P + S = total_cost) ∧ (P = (x / 100) * S) ∧ x = percentage_of_silverware_cost :=
by {
  sorry
}

end dinner_plates_percentage_l2106_210681


namespace input_x_for_y_16_l2106_210699

noncomputable def output_y_from_input_x (x : Int) : Int :=
if x < 0 then (x + 1) * (x + 1)
else (x - 1) * (x - 1)

theorem input_x_for_y_16 (x : Int) (y : Int) (h : y = 16) :
  output_y_from_input_x x = y ↔ (x = 5 ∨ x = -5) :=
by
  sorry

end input_x_for_y_16_l2106_210699


namespace find_k_l2106_210608

def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 6
def g (k x : ℝ) : ℝ := 2 * x^2 - k * x + 2

theorem find_k (k : ℝ) : 
  f 5 - g k 5 = 15 -> k = -15.8 :=
by
  intro h
  sorry

end find_k_l2106_210608


namespace largest_divisor_composite_difference_l2106_210622

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_composite_difference_l2106_210622


namespace cube_surface_area_l2106_210606

-- Define the edge length of the cube.
def edge_length (a : ℝ) : ℝ := 6 * a

-- Define the surface area of a cube given the edge length.
def surface_area (e : ℝ) : ℝ := 6 * (e * e)

-- The theorem to prove.
theorem cube_surface_area (a : ℝ) : surface_area (edge_length a) = 216 * (a * a) := 
  sorry

end cube_surface_area_l2106_210606


namespace problem1_problem2_problem3_l2106_210696

theorem problem1 (a : ℝ) : |a + 2| = 4 → (a = 2 ∨ a = -6) :=
sorry

theorem problem2 (a : ℝ) (h₀ : -4 < a) (h₁ : a < 2) : |a + 4| + |a - 2| = 6 :=
sorry

theorem problem3 (a : ℝ) : ∃ x ∈ Set.Icc (-2 : ℝ) 1, |x-1| + |x+2| = 3 :=
sorry

end problem1_problem2_problem3_l2106_210696


namespace parallel_tangent_line_l2106_210654

theorem parallel_tangent_line (b : ℝ) :
  (∃ b : ℝ, (∀ x y : ℝ, x + 2 * y + b = 0 → (x^2 + y^2 = 5))) →
  (b = 5 ∨ b = -5) :=
by
  sorry

end parallel_tangent_line_l2106_210654


namespace initial_birds_in_tree_l2106_210624

theorem initial_birds_in_tree (x : ℕ) (h : x + 81 = 312) : x = 231 := 
by
  sorry

end initial_birds_in_tree_l2106_210624


namespace simplify_expression_l2106_210604

theorem simplify_expression (x : ℝ) : 
  (12 * x ^ 12 - 3 * x ^ 10 + 5 * x ^ 9) + (-1 * x ^ 12 + 2 * x ^ 10 + x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  11 * x ^ 12 - x ^ 10 + 6 * x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
by
  sorry

end simplify_expression_l2106_210604


namespace total_soccer_balls_donated_l2106_210611

def num_elementary_classes_per_school := 4
def num_middle_classes_per_school := 5
def num_schools := 2
def soccer_balls_per_class := 5

theorem total_soccer_balls_donated : 
  (num_elementary_classes_per_school + num_middle_classes_per_school) * num_schools * soccer_balls_per_class = 90 :=
by
  sorry

end total_soccer_balls_donated_l2106_210611


namespace twice_total_credits_l2106_210648

-- Define the variables and conditions
variables (Aria Emily Spencer Hannah : ℕ)
variables (h1 : Aria = 2 * Emily) 
variables (h2 : Emily = 2 * Spencer)
variables (h3 : Emily = 20)
variables (h4 : Hannah = 3 * Spencer)

-- Proof statement
theorem twice_total_credits : 2 * (Aria + Emily + Spencer + Hannah) = 200 :=
by 
  -- Proof steps are omitted with sorry
  sorry

end twice_total_credits_l2106_210648


namespace number_of_children_bikes_l2106_210625

theorem number_of_children_bikes (c : ℕ) 
  (regular_bikes : ℕ) (wheels_per_regular_bike : ℕ) 
  (wheels_per_children_bike : ℕ) (total_wheels : ℕ)
  (h1 : regular_bikes = 7) 
  (h2 : wheels_per_regular_bike = 2) 
  (h3 : wheels_per_children_bike = 4) 
  (h4 : total_wheels = 58) 
  (h5 : total_wheels = (regular_bikes * wheels_per_regular_bike) + (c * wheels_per_children_bike)) 
  : c = 11 :=
by
  sorry

end number_of_children_bikes_l2106_210625


namespace right_triangle_legs_from_medians_l2106_210698

theorem right_triangle_legs_from_medians
  (a b : ℝ) (x y : ℝ)
  (h1 : x^2 + 4 * y^2 = 4 * a^2)
  (h2 : 4 * x^2 + y^2 = 4 * b^2) :
  y^2 = (16 * a^2 - 4 * b^2) / 15 ∧ x^2 = (16 * b^2 - 4 * a^2) / 15 :=
by
  sorry

end right_triangle_legs_from_medians_l2106_210698


namespace gcd_90_450_l2106_210675

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l2106_210675


namespace green_turtles_1066_l2106_210645

def number_of_turtles (G H : ℕ) : Prop :=
  H = 2 * G ∧ G + H = 3200

theorem green_turtles_1066 : ∃ G : ℕ, number_of_turtles G (2 * G) ∧ G = 1066 :=
by
  sorry

end green_turtles_1066_l2106_210645


namespace multiplicative_inverse_137_391_l2106_210687

theorem multiplicative_inverse_137_391 :
  ∃ (b : ℕ), (b ≤ 390) ∧ (137 * b) % 391 = 1 :=
sorry

end multiplicative_inverse_137_391_l2106_210687


namespace minimum_sugar_correct_l2106_210674

noncomputable def minimum_sugar (f : ℕ) (s : ℕ) : ℕ := 
  if (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) then s else sorry

theorem minimum_sugar_correct (f s : ℕ) : 
  (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) → s ≥ 4 :=
by sorry

end minimum_sugar_correct_l2106_210674


namespace amanda_needs_how_many_bags_of_grass_seeds_l2106_210646

theorem amanda_needs_how_many_bags_of_grass_seeds
    (lot_length : ℕ := 120)
    (lot_width : ℕ := 60)
    (concrete_length : ℕ := 40)
    (concrete_width : ℕ := 40)
    (bag_coverage : ℕ := 56) :
    (lot_length * lot_width - concrete_length * concrete_width) / bag_coverage = 100 := by
  sorry

end amanda_needs_how_many_bags_of_grass_seeds_l2106_210646


namespace perimeter_of_billboard_l2106_210676
noncomputable def perimeter_billboard : ℝ :=
  let width := 8
  let area := 104
  let length := area / width
  let perimeter := 2 * (length + width)
  perimeter

theorem perimeter_of_billboard (width area : ℝ) (P : width = 8 ∧ area = 104) :
    perimeter_billboard = 42 :=
by
  sorry

end perimeter_of_billboard_l2106_210676
