import Mathlib

namespace fraction_addition_l248_248462

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248462


namespace inequality_one_system_of_inequalities_l248_248070

theorem inequality_one (x : ℝ) : 
  (2 * x - 2) / 3 ≤ 2 - (2 * x + 2) / 2 → x ≤ 1 :=
sorry

theorem system_of_inequalities (x : ℝ) : 
  (3 * (x - 2) - 1 ≥ -4 - 2 * (x - 2) → x ≥ 7 / 5) ∧
  ((1 - 2 * x) / 3 > (3 * (2 * x - 1)) / 2 → x < 1 / 2) → false :=
sorry

end inequality_one_system_of_inequalities_l248_248070


namespace add_fractions_l248_248501

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248501


namespace milk_packet_volume_l248_248852

theorem milk_packet_volume :
  ∃ (m : ℕ), (150 * m = 1250 * 30) ∧ m = 250 :=
by
  sorry

end milk_packet_volume_l248_248852


namespace Zilla_savings_l248_248111

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l248_248111


namespace solve_x_l248_248030

theorem solve_x (x y : ℝ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 16) : x = 16 := by
  sorry

end solve_x_l248_248030


namespace inverse_solution_correct_l248_248648

noncomputable def f (a b c x : ℝ) : ℝ :=
  1 / (a * x^2 + b * x + c)

theorem inverse_solution_correct (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c x = 1 ↔ x = (-b + Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) ∨
               x = (-b - Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) :=
by
  sorry

end inverse_solution_correct_l248_248648


namespace increased_cost_is_97_l248_248844

-- Define the original costs and increases due to inflation
def original_cost_lumber := 450
def original_cost_nails := 30
def original_cost_fabric := 80

def increase_percentage_lumber := 0.20
def increase_percentage_nails := 0.10
def increase_percentage_fabric := 0.05

-- Calculate the increased costs
def increase_cost_lumber := increase_percentage_lumber * original_cost_lumber
def increase_cost_nails := increase_percentage_nails * original_cost_nails
def increase_cost_fabric := increase_percentage_fabric * original_cost_fabric

-- Calculate the total increased cost
def total_increased_cost := increase_cost_lumber + increase_cost_nails + increase_cost_fabric

-- The theorem to prove
theorem increased_cost_is_97 : total_increased_cost = 97 :=
by
  sorry

end increased_cost_is_97_l248_248844


namespace complex_roots_sum_condition_l248_248303

theorem complex_roots_sum_condition 
  (z1 z2 : ℂ) 
  (h1 : ∀ z, z ^ 2 + z + 1 = 0) 
  (h2 : z1 ^ 2 + z1 + 1 = 0)
  (h3 : z2 ^ 2 + z2 + 1 = 0) : 
  (z2 / (z1 + 1)) + (z1 / (z2 + 1)) = -2 := 
 sorry

end complex_roots_sum_condition_l248_248303


namespace coin_problem_l248_248958

variable (x y S k : ℕ)

theorem coin_problem
  (h1 : x + y = 14)
  (h2 : 2 * x + 5 * y = S)
  (h3 : S = k + 2 * k)
  (h4 : k * 4 = S) :
  y = 4 ∨ y = 8 ∨ y = 12 :=
by
  sorry

end coin_problem_l248_248958


namespace find_y_l248_248313

theorem find_y (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := 
by sorry

end find_y_l248_248313


namespace meena_cookies_left_l248_248653

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l248_248653


namespace inequality_solution_l248_248827

theorem inequality_solution (x : ℝ) : 2 * x - 1 ≤ 3 → x ≤ 2 :=
by
  intro h
  -- Here we would perform the solution steps, but we'll skip the proof with sorry.
  sorry

end inequality_solution_l248_248827


namespace max_min_of_f_l248_248378

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * Real.pi + x) + 
  Real.sqrt 3 * Real.cos (2 * Real.pi - x) -
  Real.sin (2013 * Real.pi + Real.pi / 6)

theorem max_min_of_f : 
  - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 →
  (-1 / 2) ≤ f x ∧ f x ≤ 5 / 2 :=
sorry

end max_min_of_f_l248_248378


namespace find_common_ratio_l248_248790

-- Declare the sequence and conditions
variables {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions of the problem 
def positive_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ m n : ℕ, a m = a 0 * q ^ m) ∧ q > 0

def third_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 + a 5 = 5

def fifth_term_seventh_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 5 + a 7 = 20

-- The final lean statement proving the common ratio is 2
theorem find_common_ratio 
  (h1 : positive_geometric_sequence a q) 
  (h2 : third_term_condition a q) 
  (h3 : fifth_term_seventh_term_condition a q) : 
  q = 2 :=
sorry

end find_common_ratio_l248_248790


namespace range_of_a_l248_248292

theorem range_of_a (x a : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) : -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l248_248292


namespace nails_no_three_collinear_l248_248275

-- Let's denote the 8x8 chessboard as an 8x8 grid of cells

-- Define a type for positions on the chessboard
def Position := (ℕ × ℕ)

-- Condition: 16 nails should be placed in such a way that no three are collinear. 
-- Let's create an inductive type to capture these conditions

def no_three_collinear (nails : List Position) : Prop :=
  ∀ (p1 p2 p3 : Position), p1 ∈ nails → p2 ∈ nails → p3 ∈ nails → 
  (p1.1 = p2.1 ∧ p2.1 = p3.1) → False ∧
  (p1.2 = p2.2 ∧ p2.2 = p3.2) → False ∧
  (p1.1 - p1.2 = p2.1 - p2.2 ∧ p2.1 - p2.2 = p3.1 - p3.2) → False

-- The main statement to prove
theorem nails_no_three_collinear :
  ∃ nails : List Position, List.length nails = 16 ∧ no_three_collinear nails :=
sorry

end nails_no_three_collinear_l248_248275


namespace length_of_parallelepiped_l248_248997

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248997


namespace angle_same_terminal_side_l248_248834

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, -330 = k * 360 + 30 :=
by
  use -1
  sorry

end angle_same_terminal_side_l248_248834


namespace hyperbola_equation_l248_248620

-- Definitions of the conditions
def is_asymptote_1 (y x : ℝ) : Prop :=
  y = 2 * x

def is_asymptote_2 (y x : ℝ) : Prop :=
  y = -2 * x

def passes_through_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

-- The statement to be proved
theorem hyperbola_equation :
  (∀ x y : ℝ, passes_through_focus x y → x^2 - (y^2 / 4) = 1) :=
sorry

end hyperbola_equation_l248_248620


namespace add_fractions_l248_248581

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248581


namespace therapy_charge_l248_248118

-- Defining the conditions
variables (A F : ℝ)
variables (h1 : F = A + 25)
variables (h2 : F + 4*A = 250)

-- The statement we need to prove
theorem therapy_charge : F + A = 115 := 
by
  -- proof would go here
  sorry

end therapy_charge_l248_248118


namespace jean_initial_stuffies_l248_248329

variable (S : ℕ) (h1 : S * 2 / 3 / 4 = 10)

theorem jean_initial_stuffies : S = 60 :=
by
  sorry

end jean_initial_stuffies_l248_248329


namespace range_of_a_l248_248618

def increasing {α : Type*} [Preorder α] (f : α → α) := ∀ x y, x ≤ y → f x ≤ f y

theorem range_of_a
  (f : ℝ → ℝ)
  (increasing_f : increasing f)
  (h_domain : ∀ x, 1 ≤ x ∧ x ≤ 5 → (f x = f x))
  (h_ineq : ∀ a, 1 ≤ a + 1 ∧ a + 1 ≤ 5 ∧ 1 ≤ 2 * a - 1 ∧ 2 * a - 1 ≤ 5 ∧ f (a + 1) < f (2 * a - 1)) :
  (2 : ℝ) < a ∧ a ≤ (3 : ℝ) := 
by
  sorry

end range_of_a_l248_248618


namespace overtime_hours_l248_248983

theorem overtime_hours (regular_rate: ℝ) (regular_hours: ℝ) (total_payment: ℝ) (overtime_rate_multiplier: ℝ) (overtime_hours: ℝ):
  regular_rate = 3 → regular_hours = 40 → total_payment = 198 → overtime_rate_multiplier = 2 → 
  overtime_hours = (total_payment - (regular_rate * regular_hours)) / (regular_rate * overtime_rate_multiplier) →
  overtime_hours = 13 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overtime_hours_l248_248983


namespace fraction_addition_l248_248568

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248568


namespace fraction_addition_l248_248441

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248441


namespace reciprocal_of_neg_three_l248_248082

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l248_248082


namespace only_odd_digit_squared_n_l248_248286

def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → is_odd_digit d

theorem only_odd_digit_squared_n (n : ℕ) :
  0 < n ∧ has_only_odd_digits (n * n) ↔ n = 1 ∨ n = 3 :=
sorry

end only_odd_digit_squared_n_l248_248286


namespace probability_event_B_l248_248860

-- Define the type of trial outcomes, we're considering binary outcomes for simplicity
inductive Outcome
| win : Outcome
| lose : Outcome

open Outcome

def all_possible_outcomes := [
  [win, win, win],
  [win, win, win, lose],
  [win],
  [win],
  [lose],
  [win, win, lose, lose],
  [win, lose],
  [win, lose, win, lose, win],
  [win],
  [lose],
  [lose],
  [lose],
  [lose, win, win],
  [win, lose, lose, win],
  [lose, win, lose, lose],
  [win],
  [win],
  [lose],
  [lose],
  [lose, lose],
  [lose],
  [lose],
  [],
  [lose, lose, lose, lose]
]

-- Event A is winning a prize
def event_A := [
  [win, win, win],
  [win, win, win, lose],
  [win, win, lose, lose],
  [win, lose, win, lose, win],
  [win, lose, lose, win]
]

-- Event B is satisfying the condition \(a + b + c + d \leq 2\)
def event_B := [
  [lose],
  [win, lose],
  [lose, win],
  [win],
  [lose, lose],
  [lose, win, lose],
  [lose, lose, win],
  [lose, win, win],
  [win, lose, lose],
  [lose, lose, lose],
  []
]

-- Proof that the probability of event B equals 11/16
theorem probability_event_B : (event_B.length / all_possible_outcomes.length) = 11 / 16 := by
  sorry

end probability_event_B_l248_248860


namespace age_of_new_teacher_l248_248362

theorem age_of_new_teacher (sum_of_20_teachers : ℕ)
  (avg_age_20_teachers : ℕ)
  (total_teachers_after_new_teacher : ℕ)
  (new_avg_age_after_new_teacher : ℕ)
  (h1 : sum_of_20_teachers = 20 * 49)
  (h2 : avg_age_20_teachers = 49)
  (h3 : total_teachers_after_new_teacher = 21)
  (h4 : new_avg_age_after_new_teacher = 48) :
  ∃ (x : ℕ), x = 28 :=
by
  sorry

end age_of_new_teacher_l248_248362


namespace black_balls_in_box_l248_248780

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end black_balls_in_box_l248_248780


namespace smallest_y_in_geometric_sequence_l248_248920

theorem smallest_y_in_geometric_sequence (x y z r : ℕ) (h1 : y = x * r) (h2 : z = x * r^2) (h3 : xyz = 125) : y = 5 :=
by sorry

end smallest_y_in_geometric_sequence_l248_248920


namespace find_k_x_l248_248054

-- Define the nonzero polynomial condition
def nonzero_poly (p : Polynomial ℝ) : Prop :=
  ¬ (p = 0)

-- Define the conditions from the problem statement
def conditions (h k : Polynomial ℝ) : Prop :=
  nonzero_poly h ∧ nonzero_poly k ∧ (h.comp k = h * k) ∧ (k.eval 3 = 58)

-- State the main theorem to be proven
theorem find_k_x (h k : Polynomial ℝ) (cond : conditions h k) : 
  k = Polynomial.C 1 + Polynomial.C 49 * Polynomial.X + Polynomial.C (-49) * Polynomial.X^2 :=
sorry

end find_k_x_l248_248054


namespace range_of_a_l248_248028

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l248_248028


namespace remaining_money_correct_l248_248933

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

end remaining_money_correct_l248_248933


namespace woman_speed_in_still_water_l248_248120

theorem woman_speed_in_still_water (V_w V_s : ℝ)
  (downstream_distance upstream_distance downstream_time upstream_time : ℝ)
  (h1 : downstream_distance = 45)
  (h2 : upstream_distance = 15)
  (h3 : downstream_time = 3)
  (h4 : upstream_time = 3)
  (h5 : V_w + V_s = downstream_distance / downstream_time)
  (h6 : V_w - V_s = upstream_distance / upstream_time) :
  V_w = 10 :=
by 
  have h1 : downstream_distance = 45 := by assumption,
  have h2 : upstream_distance = 15 := by assumption,
  have h3 : downstream_time = 3 := by assumption,
  have h4 : upstream_time = 3 := by assumption,
  have h5 : V_w + V_s = 15 := by {
    rw h1 at *,
    rw h3 at *,
    exact h5
  },
  have h6 : V_w - V_s = 5 := by {
    rw h2 at *,
    rw h4 at *,
    exact h6
  },
  sorry

end woman_speed_in_still_water_l248_248120


namespace intersection_M_N_l248_248040

-- Definition of the sets M and N
def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6 * x < 0}

-- Intersection of M and N
def intersection : Set ℝ := {x | 4 < x ∧ x < 6}

-- Theorem statement asserting the equality between the intersection and the desired set
theorem intersection_M_N : ∀ (x : ℝ), x ∈ M ∩ N ↔ x ∈ intersection := by
  sorry

end intersection_M_N_l248_248040


namespace quadratic_inequality_solution_l248_248369

theorem quadratic_inequality_solution :
  { m : ℝ // ∀ x : ℝ, m * x^2 - 6 * m * x + 5 * m + 1 > 0 } = { m : ℝ // 0 ≤ m ∧ m < 1/4 } :=
sorry

end quadratic_inequality_solution_l248_248369


namespace find_abc_l248_248954

theorem find_abc :
  ∃ (N : ℕ), (N > 0 ∧ (N % 10000 = N^2 % 10000) ∧ (N % 1000 > 100)) ∧ (N % 1000 / 100 = 937) :=
sorry

end find_abc_l248_248954


namespace certain_number_is_18_l248_248774

theorem certain_number_is_18 (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : p - q = 0.20833333333333334) : 3 / q = 18 :=
sorry

end certain_number_is_18_l248_248774


namespace geometric_sequence_fifth_term_l248_248005

theorem geometric_sequence_fifth_term : 
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  a₅ = 1 / 2048 :=
by
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  sorry

end geometric_sequence_fifth_term_l248_248005


namespace trigonometric_identity_l248_248254

theorem trigonometric_identity (α : ℝ) :
  (tan (4 * α) + sec (4 * α) = (cos (2 * α) + sin (2 * α)) / (cos (2 * α) - sin (2 * α))) :=
by
  sorry

end trigonometric_identity_l248_248254


namespace zilla_savings_l248_248115

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l248_248115


namespace number_of_combinations_l248_248795

-- Conditions as definitions
def n : ℕ := 9
def k : ℕ := 4

-- Lean statement of the equivalent proof problem
theorem number_of_combinations : (nat.choose n k) = 126 := by
  -- Sorry is used to skip the proof
  sorry

end number_of_combinations_l248_248795


namespace second_cyclist_speed_l248_248963

-- Definitions of the given conditions
def total_course_length : ℝ := 45
def first_cyclist_speed : ℝ := 14
def meeting_time : ℝ := 1.5

-- Lean 4 statement for the proof problem
theorem second_cyclist_speed : 
  ∃ v : ℝ, first_cyclist_speed * meeting_time + v * meeting_time = total_course_length → v = 16 := 
by 
  sorry

end second_cyclist_speed_l248_248963


namespace bijection_condition_l248_248887

variable {n m : ℕ}
variable (f : Fin n → Fin n)

theorem bijection_condition (h_even : m % 2 = 0)
(h_prime : Nat.Prime (n + 1))
(h_bij : Function.Bijective f) :
  ∀ x y : Fin n, (n : ℕ) ∣ (m * x - y : ℕ) → (n + 1) ∣ (f x).val ^ m - (f y).val := sorry

end bijection_condition_l248_248887


namespace employee_pays_204_l248_248262

-- Definitions based on conditions
def wholesale_cost : ℝ := 200
def markup_percent : ℝ := 0.20
def discount_percent : ℝ := 0.15

def retail_price := wholesale_cost * (1 + markup_percent)
def employee_payment := retail_price * (1 - discount_percent)

-- Theorem with the expected result
theorem employee_pays_204 : employee_payment = 204 := by
  -- Proof not required, we add sorry to avoid the proof details
  sorry

end employee_pays_204_l248_248262


namespace number_of_possible_flags_l248_248593

-- Define the number of colors available
def num_colors : ℕ := 3

-- Define the number of stripes on the flag
def num_stripes : ℕ := 3

-- Define the total number of possible flags
def total_flags : ℕ := num_colors ^ num_stripes

-- The statement we need to prove
theorem number_of_possible_flags : total_flags = 27 := by
  sorry

end number_of_possible_flags_l248_248593


namespace power_div_ex_l248_248858

theorem power_div_ex (a b c : ℕ) (h1 : a = 2^4) (h2 : b = 2^3) (h3 : c = 2^2) :
  ((a^4) * (b^6)) / (c^12) = 1024 := 
sorry

end power_div_ex_l248_248858


namespace parallelepiped_length_l248_248995

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248995


namespace triangle_shortest_side_condition_l248_248786

theorem triangle_shortest_side_condition
  (A B C : Type) 
  (r : ℝ) (AF FB : ℝ)
  (P : ℝ)
  (h_AF : AF = 7)
  (h_FB : FB = 9)
  (h_r : r = 5)
  (h_P : P = 46) 
  : (min (min (7 + 9) (2 * 14)) ((7 + 9) - 14)) = 2 := 
by sorry

end triangle_shortest_side_condition_l248_248786


namespace dhoni_dishwasher_spending_l248_248866

noncomputable def percentage_difference : ℝ := 0.25 - 0.225
noncomputable def percentage_less_than : ℝ := (percentage_difference / 0.25) * 100

theorem dhoni_dishwasher_spending :
  (percentage_difference / 0.25) * 100 = 10 :=
by sorry

end dhoni_dishwasher_spending_l248_248866


namespace parallelepiped_length_l248_248996

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248996


namespace money_left_after_shopping_l248_248936

def initial_money : ℕ := 158
def shoe_cost : ℕ := 45
def bag_cost := shoe_cost - 17
def lunch_cost := bag_cost / 4
def total_expenses := shoe_cost + bag_cost + lunch_cost
def remaining_money := initial_money - total_expenses

theorem money_left_after_shopping : remaining_money = 78 := by
  sorry

end money_left_after_shopping_l248_248936


namespace range_f_when_a_1_range_of_a_values_l248_248189

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

theorem range_f_when_a_1 : 
  (∀ x : ℝ, f x 1 ≥ 5) :=
sorry

theorem range_of_a_values :
  (∀ x, f x a ≥ 1) → (a ∈ Set.union (Set.Iic (-5)) (Set.Ici (-3))) :=
sorry

end range_f_when_a_1_range_of_a_values_l248_248189


namespace abs_neg_implies_nonpositive_l248_248703

theorem abs_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  sorry

end abs_neg_implies_nonpositive_l248_248703


namespace _l248_248174

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l248_248174


namespace player_a_winning_strategy_l248_248134

theorem player_a_winning_strategy (P : ℝ) : 
  (∃ m n : ℕ, P = m / (2 ^ n) ∧ m < 2 ^ n)
  ∨ P = 0
  ∨ P = 1 ↔
  (∀ d : ℝ, ∃ d_direction : ℤ, 
    (P + (d * d_direction) = 0) ∨ (P + (d * d_direction) = 1)) :=
sorry

end player_a_winning_strategy_l248_248134


namespace fraction_addition_l248_248550

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248550


namespace fraction_addition_l248_248444

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248444


namespace inequality_solution_set_l248_248607

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ -2 ≤ x ∧ x < 1 := 
sorry

end inequality_solution_set_l248_248607


namespace problem_real_numbers_inequality_l248_248225

open Real

theorem problem_real_numbers_inequality 
  (a1 b1 a2 b2 : ℝ) :
  a1 * b1 + a2 * b2 ≤ sqrt (a1^2 + a2^2) * sqrt (b1^2 + b2^2) :=
by 
  sorry

end problem_real_numbers_inequality_l248_248225


namespace minimize_expression_l248_248841

theorem minimize_expression (a : ℝ) : ∃ c : ℝ, 0 ≤ c ∧ c ≤ a ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → (x^2 + 3 * (a-x)^2) ≥ ((3*a/4)^2 + 3 * (a-3*a/4)^2)) :=
by
  sorry

end minimize_expression_l248_248841


namespace add_fractions_l248_248584

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248584


namespace number_of_students_l248_248242

variable (F S J R T : ℕ)

axiom freshman_more_than_junior : F = (5 * J) / 4
axiom sophomore_fewer_than_freshman : S = 9 * F / 10
axiom total_students : T = F + S + J + R
axiom seniors_total : R = T / 5
axiom given_sophomores : S = 144

theorem number_of_students (T : ℕ) : T = 540 :=
by 
  sorry

end number_of_students_l248_248242


namespace sticks_form_equilateral_triangle_l248_248155

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l248_248155


namespace number_of_liars_on_the_island_l248_248092

-- Definitions for the conditions
def isKnight (person : ℕ) : Prop := sorry -- Placeholder, we know knights always tell the truth
def isLiar (person : ℕ) : Prop := sorry -- Placeholder, we know liars always lie
def population := 1000
def villages := 10
def minInhabitantsPerVillage := 2

-- Definitional property: each islander claims that all other villagers in their village are liars
def claimsAllOthersAreLiars (islander : ℕ) (village : ℕ) : Prop := 
  ∀ (other : ℕ), (other ≠ islander) → (isLiar other)

-- Main statement in Lean
theorem number_of_liars_on_the_island : ∃ liars, liars = 990 :=
by
  have total_population := population
  have number_of_villages := villages
  have min_people_per_village := minInhabitantsPerVillage
  have knight_prop := isKnight
  have liar_prop := isLiar
  have claim_prop := claimsAllOthersAreLiars
  -- Proof will be filled here
  sorry

end number_of_liars_on_the_island_l248_248092


namespace marcy_multiple_tickets_l248_248372

theorem marcy_multiple_tickets (m : ℕ) : 
  (26 + (m * 26 - 6) = 150) → m = 5 :=
by
  intro h
  sorry

end marcy_multiple_tickets_l248_248372


namespace fraction_addition_l248_248438

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248438


namespace expected_distinct_values_sum_l248_248911

theorem expected_distinct_values_sum (m n : ℕ) (h_rel_prime : Int.gcd m n = 1) :
  m = 671 ∧ n = 216 → m + n = 887 := by
  intros h
  cases h
  simp [h_left, h_right]
  sorry

end expected_distinct_values_sum_l248_248911


namespace max_among_l248_248338

theorem max_among (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ (1 / (2 * real.sqrt 3)) :=
  sorry

end max_among_l248_248338


namespace interval_1_5_frequency_is_0_70_l248_248849

-- Define the intervals and corresponding frequencies
def intervals : List (ℤ × ℤ) := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

def frequencies : List ℕ := [1, 1, 2, 3, 1, 2]

-- Sample capacity
def sample_capacity : ℕ := 10

-- Calculate the frequency of the sample in the interval [1,5)
noncomputable def frequency_in_interval_1_5 : ℝ := (frequencies.take 4).sum / sample_capacity

-- Prove that the frequency in the interval [1,5) is 0.70
theorem interval_1_5_frequency_is_0_70 : frequency_in_interval_1_5 = 0.70 := by
  sorry

end interval_1_5_frequency_is_0_70_l248_248849


namespace combined_weight_of_jake_and_sister_l248_248777

theorem combined_weight_of_jake_and_sister
  (J : ℕ) (S : ℕ)
  (h₁ : J = 113)
  (h₂ : J - 33 = 2 * S)
  : J + S = 153 :=
sorry

end combined_weight_of_jake_and_sister_l248_248777


namespace locus_of_midpoint_l248_248756

theorem locus_of_midpoint (x y : ℝ) (h : y ≠ 0) :
  (∃ P : ℝ × ℝ, P = (2*x, 2*y) ∧ ((P.1^2 + (P.2-3)^2 = 9))) →
  (x^2 + (y - 3/2)^2 = 9/4) :=
by
  sorry

end locus_of_midpoint_l248_248756


namespace odd_function_negative_value_l248_248018

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_value {f : ℝ → ℝ} (h_odd : is_odd_function f) :
  (∀ x, 0 < x → f x = x^2 - x - 1) → (∀ x, x < 0 → f x = -x^2 - x + 1) :=
by
  sorry

end odd_function_negative_value_l248_248018


namespace same_grade_percentage_l248_248243

theorem same_grade_percentage (total_students: ℕ)
  (a_students: ℕ) (b_students: ℕ) (c_students: ℕ) (d_students: ℕ)
  (total: total_students = 30)
  (a: a_students = 2) (b: b_students = 4) (c: c_students = 5) (d: d_students = 1)
  : (a_students + b_students + c_students + d_students) * 100 / total_students = 40 := by
  sorry

end same_grade_percentage_l248_248243


namespace area_ratio_of_similar_triangles_l248_248833

noncomputable def similarity_ratio := 3 / 5

theorem area_ratio_of_similar_triangles (k : ℝ) (h_sim : similarity_ratio = k) : (k^2 = 9 / 25) :=
by
  sorry

end area_ratio_of_similar_triangles_l248_248833


namespace fraction_addition_l248_248552

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248552


namespace percentage_increase_l248_248825

variables (a b x m : ℝ) (p : ℝ)
variables (h1 : a / b = 4 / 5)
variables (h2 : x = a + (p / 100) * a)
variables (h3 : m = b - 0.6 * b)
variables (h4 : m / x = 0.4)

theorem percentage_increase (a_pos : 0 < a) (b_pos : 0 < b) : p = 25 :=
by sorry

end percentage_increase_l248_248825


namespace sum_fractions_eq_l248_248476

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248476


namespace fraction_addition_l248_248555

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248555


namespace expression_evaluation_l248_248672

noncomputable def x := Real.sqrt 5 + 1
noncomputable def y := Real.sqrt 5 - 1

theorem expression_evaluation : 
  ( ( (5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2) ) / (1 / (x^2 * y - x * y^2)) ) = 12 := 
by 
  -- Provide a proof here
  sorry

end expression_evaluation_l248_248672


namespace first_driver_spends_less_time_l248_248832

noncomputable def round_trip_time (d : ℝ) (v₁ v₂ : ℝ) : ℝ := (d / v₁) + (d / v₂)

theorem first_driver_spends_less_time (d : ℝ) : 
  round_trip_time d 80 80 < round_trip_time d 90 70 :=
by
  --We skip the proof here
  sorry

end first_driver_spends_less_time_l248_248832


namespace probability_heads_100_l248_248906

noncomputable
def probability_heads_on_100th_toss : Prop :=
  ∀ (coin_toss: ℕ → Prop),
  (∀ n, coin_toss (n - 1) → coin_toss n) →
  (∀ n, (coin_toss n = true ∨ coin_toss n = false)) →
  (coin_toss 100 = true ∨ coin_toss 100 = false) →
  ∀ (p : ℝ), (p = 1/2) →
  (∀ n, coin_toss n = true ∧ p = 1/2) 

theorem probability_heads_100 :
  probability_heads_on_100th_toss :=
sorry

end probability_heads_100_l248_248906


namespace equation_1_solution_1_equation_2_solution_l248_248877

theorem equation_1_solution_1 (x : ℝ) (h : 4 * (x - 1) ^ 2 = 25) : x = 7 / 2 ∨ x = -3 / 2 := by
  sorry

theorem equation_2_solution (x : ℝ) (h : (1 / 3) * (x + 2) ^ 3 - 9 = 0) : x = 1 := by
  sorry

end equation_1_solution_1_equation_2_solution_l248_248877


namespace sum_of_fractions_l248_248436

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248436


namespace perfect_square_conditions_l248_248148

theorem perfect_square_conditions (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101 * k = m^2) ↔ (k = 101 ∨ k = 2601) := 
by 
  sorry

end perfect_square_conditions_l248_248148


namespace find_second_number_l248_248605

theorem find_second_number (n : ℕ) 
  (h1 : Nat.lcm 24 (Nat.lcm n 42) = 504)
  (h2 : 504 = 2^3 * 3^2 * 7) 
  (h3 : Nat.lcm 24 42 = 168) : n = 3 := 
by 
  sorry

end find_second_number_l248_248605


namespace max_f_angle_A_of_triangle_l248_248898

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x - 4 * Real.pi / 3)) + 2 * (Real.cos x)^2

theorem max_f : ∃ x : ℝ, f x = 2 := sorry

theorem angle_A_of_triangle (A B C : ℝ) (h : A + B + C = Real.pi)
  (h2 : f (B + C) = 3 / 2) : A = Real.pi / 3 := sorry

end max_f_angle_A_of_triangle_l248_248898


namespace ratio_of_beef_to_pork_l248_248050

/-- 
James buys 20 pounds of beef. 
James buys an unknown amount of pork. 
James uses 1.5 pounds of meat to make each meal. 
Each meal sells for $20. 
James made $400 from selling meals.
The ratio of the amount of beef to the amount of pork James bought is 2:1.
-/
theorem ratio_of_beef_to_pork (beef pork : ℝ) (meal_weight : ℝ) (meal_price : ℝ) (total_revenue : ℝ)
  (h_beef : beef = 20)
  (h_meal_weight : meal_weight = 1.5)
  (h_meal_price : meal_price = 20)
  (h_total_revenue : total_revenue = 400) :
  (beef / pork) = 2 :=
by
  sorry

end ratio_of_beef_to_pork_l248_248050


namespace add_fractions_l248_248509

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248509


namespace LynsDonation_l248_248060

theorem LynsDonation (X : ℝ)
  (h1 : 1/3 * X + 1/2 * X + 1/4 * (X - (1/3 * X + 1/2 * X)) = 3/4 * X)
  (h2 : (X - 3/4 * X)/4 = 30) :
  X = 240 := by
  sorry

end LynsDonation_l248_248060


namespace initial_people_in_castle_l248_248245

theorem initial_people_in_castle (P : ℕ) (provisions : ℕ → ℕ → ℕ) :
  (provisions P 90) - (provisions P 30) = provisions (P - 100) 90 ↔ P = 300 :=
by
  sorry

end initial_people_in_castle_l248_248245


namespace pie_distribution_l248_248692

theorem pie_distribution (x y : ℕ) (h1 : x + y + 2 * x = 13) (h2 : x < y) (h3 : y < 2 * x) :
  x = 3 ∧ y = 4 ∧ 2 * x = 6 := by
  sorry

end pie_distribution_l248_248692


namespace cyclic_quadrilateral_angles_l248_248144

theorem cyclic_quadrilateral_angles (A B C D : ℝ) (h_cyclic : A + C = 180) (h_diag_bisect : (A = 2 * (B / 5 + B / 5)) ∧ (C = 2 * (D / 5 + D / 5))) (h_ratio : B / D = 2 / 3):
  A = 80 ∨ A = 100 ∨ A = 1080 / 11 ∨ A = 900 / 11 :=
  sorry

end cyclic_quadrilateral_angles_l248_248144


namespace fraction_addition_l248_248562

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248562


namespace fraction_addition_l248_248536

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248536


namespace fraction_addition_l248_248565

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248565


namespace fraction_addition_l248_248560

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248560


namespace sum_of_three_squares_l248_248729

variable (t s : ℝ)

-- Given equations
axiom h1 : 3 * t + 2 * s = 27
axiom h2 : 2 * t + 3 * s = 25

-- What we aim to prove
theorem sum_of_three_squares : 3 * s = 63 / 5 :=
by
  sorry

end sum_of_three_squares_l248_248729


namespace fraction_addition_l248_248535

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248535


namespace Zilla_savings_l248_248109

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l248_248109


namespace find_divisor_l248_248036

theorem find_divisor (x d : ℤ) (h1 : ∃ k : ℤ, x = k * d + 5)
                     (h2 : ∃ n : ℤ, x + 17 = n * 41 + 22) :
    d = 1 :=
by
  sorry

end find_divisor_l248_248036


namespace convert_to_scientific_notation_9600000_l248_248071

theorem convert_to_scientific_notation_9600000 :
  9600000 = 9.6 * 10^6 := 
sorry

end convert_to_scientific_notation_9600000_l248_248071


namespace find_all_functions_l248_248649

theorem find_all_functions (n : ℕ) (h_pos : 0 < n) (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x)^n * f (x + y) = (f x)^(n + 1) + x^n * f y) ↔
  (if n % 2 = 1 then ∀ x, f x = 0 ∨ f x = x else ∀ x, f x = 0 ∨ f x = x ∨ f x = -x) :=
sorry

end find_all_functions_l248_248649


namespace probability_bernardo_larger_than_silvia_l248_248857

open ProbabilityTheory

section probability_of_larger_number

def probability_three_digit_number_larger (bernardo_set : Finset ℕ) (silvia_set : Finset ℕ) : ℚ :=
    let bernardo_pick := ∑ b in bernardo_set.powerset.filter (λ s, s.card = 3), (1 : ℚ) / bernardo_set.powerset.card
    let silvia_pick := ∑ s in silvia_set.powerset.filter (λ s, s.card = 3), (1 : ℚ) / silvia_set.powerset.card
    ((bernardo_pick * (1 - (1 / bernardo_pick)) / 2) + 3 / 10 + (7 / 10 * (1 - (1 / 84) / 2)))

theorem probability_bernardo_larger_than_silvia : probability_three_digit_number_larger 
 (Finset.range 10 + 1).erase 10 
 (Finset.range 9 + 1) = 155 / 240 := sorry
end probability_of_larger_number

end probability_bernardo_larger_than_silvia_l248_248857


namespace identity_holds_for_all_a_b_l248_248317

theorem identity_holds_for_all_a_b (a b : ℝ) :
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  x^2 + y^2 = 2 * (z^2 + t^2) :=
by {
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  sorry
}

end identity_holds_for_all_a_b_l248_248317


namespace average_tickets_sold_by_female_members_l248_248711

theorem average_tickets_sold_by_female_members 
  (average_all : ℕ)
  (ratio_mf : ℕ)
  (average_male : ℕ)
  (h1 : average_all = 66)
  (h2 : ratio_mf = 2)
  (h3 : average_male = 58) :
  ∃ (F : ℕ), F = 70 :=
by
  let M := 1
  let num_female := ratio_mf * M
  let total_tickets_male := average_male * M
  let total_tickets_female := num_female * 70
  have total_all_members : ℕ := M + num_female
  have total_tickets_all : ℕ := total_tickets_male + total_tickets_female
  have average_all_eq : average_all = total_tickets_all / total_all_members
  use 70
  sorry

end average_tickets_sold_by_female_members_l248_248711


namespace probability_of_queen_is_correct_l248_248104

def deck_size : ℕ := 52
def queen_count : ℕ := 4

-- This definition denotes the probability calculation.
def probability_drawing_queen : ℚ := queen_count / deck_size

theorem probability_of_queen_is_correct :
  probability_drawing_queen = 1 / 13 :=
by
  sorry

end probability_of_queen_is_correct_l248_248104


namespace total_value_of_item_l248_248840

variable {V : ℝ}

theorem total_value_of_item (h : 0.07 * (V - 1000) = 109.20) : V = 2560 := 
by
  sorry

end total_value_of_item_l248_248840


namespace total_rooms_to_paint_l248_248131

-- Definitions based on conditions
def hours_per_room : ℕ := 8
def rooms_already_painted : ℕ := 8
def hours_to_paint_rest : ℕ := 16

-- Theorem statement
theorem total_rooms_to_paint :
  rooms_already_painted + hours_to_paint_rest / hours_per_room = 10 :=
  sorry

end total_rooms_to_paint_l248_248131


namespace find_value_of_c_l248_248865

theorem find_value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 8 > 0 ↔ x < -2 ∨ x > 4)) → c = 2 :=
by
  sorry

end find_value_of_c_l248_248865


namespace area_of_intersection_of_circles_l248_248962

theorem area_of_intersection_of_circles :
  let circle1_c : (ℝ × ℝ) := (3, 0),
      radius1  : ℝ := 3,
      circle2_c : (ℝ × ℝ) := (0, 3),
      radius2  : ℝ := 3 in
  (∀ x y : ℝ, (x - circle1_c.1)^2 + y^2 < radius1^2 → 
               x^2 + (y - circle2_c.2)^2 < radius2^2 → 
               ((∃ a b : set ℝ, (a = set_of (λ p, (p.1 - circle1_c.1)^2 + p.2^2 < radius1^2) ∧ 
                                   b = set_of (λ p, p.1^2 + (p.2 - circle2_c.2)^2 < radius2^2))) ∧ 
                measure_theory.measure (@set.inter ℝ (λ p, (p.1 - circle1_c.1)^2 + p.2^2 < radius1^2) 
                                                (λ p, p.1^2 + (p.2 - circle2_c.2)^2 < radius2^2)) = 
                (9 * real.pi - 18) / 2)) :=
sorry

end area_of_intersection_of_circles_l248_248962


namespace find_value_of_a_l248_248905

theorem find_value_of_a (a : ℝ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end find_value_of_a_l248_248905


namespace geometric_series_properties_l248_248020

theorem geometric_series_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  a 3 = 3 ∧ a 10 = 384 → 
  q = 2 ∧ 
  (∀ n, a n = (3 / 4) * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (3 / 4) * (2 ^ n - 1)) :=
by
  intro h
  -- Proofs will go here, if necessary.
  sorry

end geometric_series_properties_l248_248020


namespace axes_of_symmetry_not_coincide_l248_248076

def y₁ (x : ℝ) := (1 / 8) * (x^2 + 6 * x - 25)
def y₂ (x : ℝ) := (1 / 8) * (31 - x^2)

def tangent_y₁ (x : ℝ) := (x + 3) / 4
def tangent_y₂ (x : ℝ) := -x / 4

def axes_symmetry_y₁ := -3
def axes_symmetry_y₂ := 0

theorem axes_of_symmetry_not_coincide :
  (∃ x1 x2 : ℝ, y₁ x1 = y₂ x1 ∧ y₁ x2 = y₂ x2 ∧ tangent_y₁ x1 * tangent_y₂ x1 = -1 ∧ tangent_y₁ x2 * tangent_y₂ x2 = -1) →
  axes_symmetry_y₁ ≠ axes_symmetry_y₂ :=
by sorry

end axes_of_symmetry_not_coincide_l248_248076


namespace cos_B_and_area_of_triangle_l248_248214

theorem cos_B_and_area_of_triangle (A B C : ℝ) (a b c : ℝ)
  (h_sin_A : Real.sin A = Real.sin (2 * B))
  (h_a : a = 4) (h_b : b = 6) :
  Real.cos B = 1 / 3 ∧ ∃ (area : ℝ), area = 8 * Real.sqrt 2 :=
by
  sorry  -- Proof goes here

end cos_B_and_area_of_triangle_l248_248214


namespace add_fractions_l248_248487

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248487


namespace find_second_number_l248_248955

theorem find_second_number 
  (x y z : ℕ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 * y) / 4)
  (h3 : z = (9 * y) / 7) : 
  y = 40 :=
sorry

end find_second_number_l248_248955


namespace initial_oranges_l248_248695

theorem initial_oranges (O : ℕ) (h1 : O + 6 - 3 = 6) : O = 3 :=
by
  sorry

end initial_oranges_l248_248695


namespace units_digit_of_quotient_l248_248139

theorem units_digit_of_quotient : 
  (7 ^ 2023 + 4 ^ 2023) % 9 = 2 → 
  (7 ^ 2023 + 4 ^ 2023) / 9 % 10 = 0 :=
by
  -- condition: calculation of modulo result
  have h1 : (7 ^ 2023 + 4 ^ 2023) % 9 = 2 := sorry

  -- we have the target statement here
  exact sorry

end units_digit_of_quotient_l248_248139


namespace triangle_side_a_l248_248915

theorem triangle_side_a (c b : ℝ) (B : ℝ) (h₁ : c = 2) (h₂ : b = 6) (h₃ : B = 120) : a = 2 :=
by sorry

end triangle_side_a_l248_248915


namespace line_tangent_ellipse_l248_248981

-- Define the conditions of the problem
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 2
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

-- Prove the statement about the intersection of the line and ellipse
theorem line_tangent_ellipse (m : ℝ) :
  (∀ x y, line m x y → ellipse x y → x = 0.0 ∧ y = 2.0)
  ↔ m^2 = 1 / 3 :=
sorry

end line_tangent_ellipse_l248_248981


namespace point_in_second_quadrant_l248_248203

theorem point_in_second_quadrant (m : ℝ) (h : 2 > 0 ∧ m < 0) : m < 0 :=
by
  sorry

end point_in_second_quadrant_l248_248203


namespace part_I_part_II_l248_248922

/-- (I) -/
theorem part_I (x : ℝ) (a : ℝ) (h_a : a = -1) :
  (|2 * x| + |x - 1| ≤ 4) → x ∈ Set.Icc (-1) (5 / 3) :=
by sorry

/-- (II) -/
theorem part_II (x : ℝ) (a : ℝ) (h_eq : |2 * x| + |x + a| = |x - a|) :
  (a > 0 → x ∈ Set.Icc (-a) 0) ∧ (a < 0 → x ∈ Set.Icc 0 (-a)) :=
by sorry

end part_I_part_II_l248_248922


namespace fraction_addition_l248_248534

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248534


namespace count_games_l248_248209

def total_teams : ℕ := 20
def games_per_pairing : ℕ := 7
def total_games := (total_teams * (total_teams - 1)) / 2 * games_per_pairing

theorem count_games : total_games = 1330 := by
  sorry

end count_games_l248_248209


namespace fraction_addition_l248_248458

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248458


namespace sheila_weekly_earnings_l248_248810

theorem sheila_weekly_earnings:
  (∀(m w f : ℕ), (m = 8) → (w = 8) → (f = 8) → 
   ∀(t th : ℕ), (t = 6) → (th = 6) → 
   ∀(h : ℕ), (h = 6) → 
   (m + w + f + t + th) * h = 216) := by
  sorry

end sheila_weekly_earnings_l248_248810


namespace x_varies_z_pow_l248_248201

variable (k j : ℝ)
variable (y z : ℝ)

-- Given conditions
def x_varies_y_squared (x : ℝ) := x = k * y^2
def y_varies_z_cuberoot_squared := y = j * z^(2/3)

-- To prove: 
theorem x_varies_z_pow (x : ℝ) (h1 : x_varies_y_squared k y x) (h2 : y_varies_z_cuberoot_squared j z y) : ∃ m : ℝ, x = m * z^(4/3) :=
by
  sorry

end x_varies_z_pow_l248_248201


namespace rebecca_marbles_l248_248807

theorem rebecca_marbles (M : ℕ) (h1 : 20 = M + 14) : M = 6 :=
by
  sorry

end rebecca_marbles_l248_248807


namespace min_cos_C_l248_248206

theorem min_cos_C (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h1 : (1 / Real.sin A) + (2 / Real.sin B) = 3 * ((1 / Real.tan A) + (1 / Real.tan B))) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := 
sorry

end min_cos_C_l248_248206


namespace milkman_profit_percentage_l248_248971

noncomputable def profit_percentage (x : ℝ) : ℝ :=
  let cp_per_litre := x
  let sp_per_litre := 2 * x
  let mixture_litres := 8
  let milk_litres := 6
  let cost_price := milk_litres * cp_per_litre
  let selling_price := mixture_litres * sp_per_litre
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

theorem milkman_profit_percentage (x : ℝ) 
  (h : x > 0) : 
  profit_percentage x = 166.67 :=
by
  sorry

end milkman_profit_percentage_l248_248971


namespace add_fractions_l248_248582

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248582


namespace Emmy_money_l248_248000

theorem Emmy_money {Gerry_money cost_per_apple number_of_apples Emmy_money : ℕ} 
    (h1 : Gerry_money = 100)
    (h2 : cost_per_apple = 2) 
    (h3 : number_of_apples = 150) 
    (h4 : number_of_apples * cost_per_apple = Gerry_money + Emmy_money) :
    Emmy_money = 200 :=
by
   sorry

end Emmy_money_l248_248000


namespace T_description_l248_248334

-- Definitions of conditions
def T (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y ≤ 9) ∨
  (y - 5 = 4 ∧ x ≤ 1) ∨
  (x + 3 = y - 5 ∧ x ≥ 1)

-- The problem statement in Lean: Prove that T describes three rays with a common point (1, 9)
theorem T_description :
  ∀ x y, T x y ↔ 
    ((x = 1 ∧ y ≤ 9) ∨
     (x ≤ 1 ∧ y = 9) ∨
     (x ≥ 1 ∧ y = x + 8)) :=
by sorry

end T_description_l248_248334


namespace base_eight_to_base_ten_642_l248_248697

theorem base_eight_to_base_ten_642 :
  let d0 := 2
  let d1 := 4
  let d2 := 6
  let base := 8
  d0 * base^0 + d1 * base^1 + d2 * base^2 = 418 := 
by
  sorry

end base_eight_to_base_ten_642_l248_248697


namespace add_fractions_l248_248506

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248506


namespace max_value_of_x2_plus_y2_l248_248624

theorem max_value_of_x2_plus_y2 {x y : ℝ} 
  (h1 : x ≥ 1)
  (h2 : y ≥ x)
  (h3 : x - 2 * y + 3 ≥ 0) : 
  x^2 + y^2 ≤ 18 :=
sorry

end max_value_of_x2_plus_y2_l248_248624


namespace sticks_form_equilateral_triangle_l248_248156

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l248_248156


namespace fraction_addition_l248_248564

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248564


namespace part1_part2_l248_248919

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem part1 : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 := sorry

theorem part2 : (a ^ 2 + c ^ 2) / b + (b ^ 2 + a ^ 2) / c + (c ^ 2 + b ^ 2) / a ≥ 2 := sorry

end part1_part2_l248_248919


namespace sticks_form_equilateral_triangle_l248_248157

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l248_248157


namespace fraction_addition_l248_248540

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248540


namespace initial_rulers_calculation_l248_248094

variable {initial_rulers taken_rulers left_rulers : ℕ}

theorem initial_rulers_calculation 
  (h1 : taken_rulers = 25) 
  (h2 : left_rulers = 21) 
  (h3 : initial_rulers = taken_rulers + left_rulers) : 
  initial_rulers = 46 := 
by 
  sorry

end initial_rulers_calculation_l248_248094


namespace participated_in_both_l248_248320

-- Define the conditions
def total_students := 40
def math_competition := 31
def physics_competition := 20
def not_participating := 8

-- Define number of students participated in both competitions
def both_competitions := 59 - total_students

-- Theorem statement
theorem participated_in_both : both_competitions = 19 := 
sorry

end participated_in_both_l248_248320


namespace fraction_addition_l248_248545

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248545


namespace point_on_opposite_sides_l248_248039

theorem point_on_opposite_sides (y_0 : ℝ) :
  (2 - 2 * 3 + 5 > 0) ∧ (6 - 2 * y_0 < 0) → y_0 > 3 :=
by
  sorry

end point_on_opposite_sides_l248_248039


namespace geometric_sequence_fifth_term_l248_248006

theorem geometric_sequence_fifth_term (a1 a2 : ℝ) (h1 : a1 = 2) (h2 : a2 = 1 / 4) : 
  let r := a2 / a1 in
  let a5 := a1 * r ^ 4 in
  a5 = 1 / 2048 :=
by
  sorry

end geometric_sequence_fifth_term_l248_248006


namespace maximum_x_minus_y_l248_248337

theorem maximum_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end maximum_x_minus_y_l248_248337


namespace num_right_angle_triangles_l248_248789

-- Step d): Lean 4 statement
theorem num_right_angle_triangles {C : ℝ × ℝ} (hC : C.2 = 0) :
  (C = (-2, 0) ∨ C = (4, 0) ∨ C = (1, 0)) ↔ ∃ A B : ℝ × ℝ,
  (A = (-2, 3)) ∧ (B = (4, 3)) ∧ 
  (A.2 = B.2) ∧ (A.1 ≠ B.1) ∧ 
  (((C.1-A.1)*(B.1-A.1) + (C.2-A.2)*(B.2-A.2) = 0) ∨ 
   ((C.1-B.1)*(A.1-B.1) + (C.2-B.2)*(A.2-B.2) = 0)) :=
sorry

end num_right_angle_triangles_l248_248789


namespace ruble_coins_problem_l248_248957

theorem ruble_coins_problem : 
    ∃ y : ℕ, y ∈ {4, 8, 12} ∧ ∃ x : ℕ, x + y = 14 ∧ ∃ S : ℕ, S = 2 * x + 5 * y ∧ S % 4 = 0 :=
by
  sorry

end ruble_coins_problem_l248_248957


namespace mary_income_is_128_percent_of_juan_income_l248_248928

def juan_income : ℝ := sorry
def tim_income : ℝ := 0.80 * juan_income
def mary_income : ℝ := 1.60 * tim_income

theorem mary_income_is_128_percent_of_juan_income
  (J : ℝ) : mary_income = 1.28 * J :=
by
  sorry

end mary_income_is_128_percent_of_juan_income_l248_248928


namespace problem1_problem2_l248_248733

theorem problem1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 :=
by
  sorry

theorem problem2 : Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3 :=
by
  sorry

end problem1_problem2_l248_248733


namespace pebbles_sum_at_12_days_l248_248222

def pebbles_collected (n : ℕ) : ℕ :=
  if n = 0 then 0 else n + pebbles_collected (n - 1)

theorem pebbles_sum_at_12_days : pebbles_collected 12 = 78 := by
  -- This would be the place for the proof, but adding sorry as instructed.
  sorry

end pebbles_sum_at_12_days_l248_248222


namespace fraction_addition_l248_248548

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248548


namespace region_area_l248_248707

/-- 
  Trapezoid has side lengths 10, 10, 10, and 22. 
  Each side of the trapezoid is the diameter of a semicircle 
  with the two semicircles on the two parallel sides of the trapezoid facing outside 
  and the other two semicircles facing inside the trapezoid.
  The region bounded by these four semicircles has area m + nπ, where m and n are positive integers.
  Prove that m + n = 188.5.
-/
theorem region_area (m n : ℝ) (h1: m = 128) (h2: n = 60.5) : m + n = 188.5 :=
by
  rw [h1, h2]
  norm_num -- simplifies the expression and checks it is equal to 188.5

end region_area_l248_248707


namespace determine_initial_fund_l248_248236

def initial_amount_fund (n : ℕ) := 60 * n + 30 - 10

theorem determine_initial_fund (n : ℕ) (h : 50 * n + 110 = 60 * n - 10) : initial_amount_fund n = 740 :=
by
  -- we skip the proof steps here
  sorry

end determine_initial_fund_l248_248236


namespace asymptote_of_hyperbola_l248_248192

theorem asymptote_of_hyperbola (x y : ℝ) :
  (x^2 - (y^2 / 4) = 1) → (y = 2 * x ∨ y = -2 * x) := sorry

end asymptote_of_hyperbola_l248_248192


namespace centroid_coordinates_of_tetrahedron_l248_248617

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (O A B C G G1 : V) (OG1_subdivides : G -ᵥ O = 3 • (G1 -ᵥ G))
variable (A_centroid : G1 -ᵥ O = (1/3 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O))

-- The main proof problem
theorem centroid_coordinates_of_tetrahedron :
  G -ᵥ O = (1/4 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O) :=
sorry

end centroid_coordinates_of_tetrahedron_l248_248617


namespace roots_of_x2_eq_x_l248_248687

theorem roots_of_x2_eq_x : ∀ x : ℝ, x^2 = x ↔ (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_x2_eq_x_l248_248687


namespace collinear_vector_l248_248762

theorem collinear_vector (c R : ℝ) (A B : ℝ × ℝ) (hA: A.1 ^ 2 + A.2 ^ 2 = R ^ 2) (hB: B.1 ^ 2 + B.2 ^ 2 = R ^ 2) 
                         (h_line_A: 2 * A.1 + A.2 = c) (h_line_B: 2 * B.1 + B.2 = c) :
                         ∃ k : ℝ, (4, 2) = (k * (A.1 + B.1), k * (A.2 + B.2)) :=
sorry

end collinear_vector_l248_248762


namespace length_of_segments_equal_d_l248_248205

noncomputable def d_eq (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) : ℝ :=
  if h_eq : AB = 550 ∧ BC = 580 ∧ AC = 620 then 342 else 0

theorem length_of_segments_equal_d (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) :
  d_eq AB BC AC h = 342 :=
by
  sorry

end length_of_segments_equal_d_l248_248205


namespace intersection_M_N_union_complements_M_N_l248_248750

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_M_N :
  M ∩ N = {x | 1 ≤ x ∧ x < 5} :=
by {
  sorry
}

theorem union_complements_M_N :
  (compl M) ∪ (compl N) = {x | x < 1 ∨ x ≥ 5} :=
by {
  sorry
}

end intersection_M_N_union_complements_M_N_l248_248750


namespace johanna_loses_half_turtles_l248_248356

theorem johanna_loses_half_turtles
  (owen_turtles_initial : ℕ)
  (johanna_turtles_fewer : ℕ)
  (owen_turtles_after_month : ℕ)
  (owen_turtles_final : ℕ)
  (johanna_donates_rest_to_owen : ℚ → ℚ)
  (x : ℚ)
  (hx1 : owen_turtles_initial = 21)
  (hx2 : johanna_turtles_fewer = 5)
  (hx3 : owen_turtles_after_month = owen_turtles_initial * 2)
  (hx4 : owen_turtles_final = owen_turtles_after_month + johanna_donates_rest_to_owen (1 - x))
  (hx5 : owen_turtles_final = 50) :
  x = 1 / 2 :=
by
  sorry

end johanna_loses_half_turtles_l248_248356


namespace find_f_of_2_l248_248166

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem find_f_of_2 : f 2 = 5 := by
  sorry

end find_f_of_2_l248_248166


namespace distinct_ints_sum_to_4r_l248_248333

theorem distinct_ints_sum_to_4r 
  (a b c d r : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d := 
by sorry

end distinct_ints_sum_to_4r_l248_248333


namespace Jamie_minimum_4th_quarter_score_l248_248098

theorem Jamie_minimum_4th_quarter_score (q1 q2 q3 : ℤ) (avg : ℤ) (minimum_score : ℤ) :
  q1 = 84 → q2 = 80 → q3 = 83 → avg = 85 → minimum_score = 93 → 4 * avg - (q1 + q2 + q3) = minimum_score :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Jamie_minimum_4th_quarter_score_l248_248098


namespace fraction_addition_l248_248566

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248566


namespace minutes_until_8_00_am_l248_248776

-- Definitions based on conditions
def time_in_minutes (hours : Nat) (minutes : Nat) : Nat := hours * 60 + minutes

def current_time : Nat := time_in_minutes 7 30 + 16

def target_time : Nat := time_in_minutes 8 0

-- The theorem we need to prove
theorem minutes_until_8_00_am : target_time - current_time = 14 :=
by
  sorry

end minutes_until_8_00_am_l248_248776


namespace add_fractions_l248_248524

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248524


namespace angle_420_mod_360_eq_60_l248_248968

def angle_mod_equiv (a b : ℕ) : Prop := a % 360 = b

theorem angle_420_mod_360_eq_60 : angle_mod_equiv 420 60 := 
by
  sorry

end angle_420_mod_360_eq_60_l248_248968


namespace translate_graph_cos_l248_248830

/-- Let f(x) = cos(2x). 
    Translate f(x) to the left by π/6 units to get g(x), 
    then translate g(x) upwards by 1 unit to get h(x). 
    Prove that h(x) = cos(2x + π/3) + 1. -/
theorem translate_graph_cos :
  let f (x : ℝ) := Real.cos (2 * x)
  let g (x : ℝ) := f (x + Real.pi / 6)
  let h (x : ℝ) := g x + 1
  ∀ (x : ℝ), h x = Real.cos (2 * x + Real.pi / 3) + 1 :=
by
  sorry

end translate_graph_cos_l248_248830


namespace add_fractions_l248_248519

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248519


namespace complement_of_intersection_l248_248650

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = -x^2}

theorem complement_of_intersection :
  (Set.compl (A ∩ B) = {x | x < -2 ∨ x > 0 }) :=
by
  sorry

end complement_of_intersection_l248_248650


namespace cream_ratio_l248_248792

def joe_ends_with_cream (start_coffee : ℕ) (drank_coffee : ℕ) (added_cream : ℕ) : ℕ :=
  added_cream

def joann_cream_left (start_coffee : ℕ) (added_cream : ℕ) (drank_mix : ℕ) : ℚ :=
  added_cream - drank_mix * (added_cream / (start_coffee + added_cream))

theorem cream_ratio (start_coffee : ℕ) (joe_drinks : ℕ) (joe_adds : ℕ)
                    (joann_adds : ℕ) (joann_drinks : ℕ) :
  joe_ends_with_cream start_coffee joe_drinks joe_adds / 
  joann_cream_left start_coffee joann_adds joann_drinks = (9 : ℚ) / (7 : ℚ) :=
by
  sorry

end cream_ratio_l248_248792


namespace add_fractions_result_l248_248412

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248412


namespace real_part_of_complex_l248_248621

theorem real_part_of_complex (a : ℝ) (h : a^2 + 2 * a - 15 = 0 ∧ a + 5 ≠ 0) : a = 3 :=
by sorry

end real_part_of_complex_l248_248621


namespace quadratic_distinct_real_roots_l248_248014

theorem quadratic_distinct_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a = 0 → x^2 - 2*x - a = 0 ∧ (∀ y : ℝ, y ≠ x → y^2 - 2*y - a = 0)) → 
  a > -1 :=
by
  sorry

end quadratic_distinct_real_roots_l248_248014


namespace trapezoid_median_l248_248680

theorem trapezoid_median {BC AD : ℝ} (h AC CD : ℝ) (h_nonneg : h = 2) (AC_eq_CD : AC = 4) (BC_eq_0 : BC = 0) 
: (AD = 4 * Real.sqrt 3) → (median = 3 * Real.sqrt 3) := by
  sorry

end trapezoid_median_l248_248680


namespace radius_of_circle_l248_248681

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

-- Prove that given the circle's equation, the radius is 1
theorem radius_of_circle (x y : ℝ) :
  circle_equation x y → ∃ (r : ℝ), r = 1 :=
by
  sorry

end radius_of_circle_l248_248681


namespace range_alpha_sub_beta_l248_248312

theorem range_alpha_sub_beta (α β : ℝ) (h₁ : -π/2 < α) (h₂ : α < β) (h₃ : β < π/2) : -π < α - β ∧ α - β < 0 := by
  sorry

end range_alpha_sub_beta_l248_248312


namespace proof_of_a_neg_two_l248_248167

theorem proof_of_a_neg_two (a : ℝ) (i : ℂ) (h_i : i^2 = -1) (h_real : (1 + i)^2 - a / i = (a + 2) * i → ∃ r : ℝ, (1 + i)^2 - a / i = r) : a = -2 :=
sorry

end proof_of_a_neg_two_l248_248167


namespace range_of_a_l248_248379

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → x^2 - 2*x + a < 0) ↔ a ≤ 0 :=
by sorry

end range_of_a_l248_248379


namespace complement_supplement_angle_l248_248892

theorem complement_supplement_angle (α : ℝ) : 
  ( 180 - α) = 3 * ( 90 - α ) → α = 45 :=
by 
  sorry

end complement_supplement_angle_l248_248892


namespace side_length_of_square_ground_l248_248396

theorem side_length_of_square_ground
    (radius : ℝ)
    (Q_area : ℝ)
    (pi : ℝ)
    (quarter_circle_area : Q_area = (pi * (radius^2) / 4))
    (pi_approx : pi = 3.141592653589793)
    (Q_area_val : Q_area = 15393.804002589986)
    (radius_val : radius = 140) :
    ∃ (s : ℝ), s^2 = radius^2 :=
by
  sorry -- Proof not required per the instructions

end side_length_of_square_ground_l248_248396


namespace distance_BC_l248_248264

variable (AC AB : ℝ) (angleACB : ℝ)
  (hAC : AC = 2)
  (hAB : AB = 3)
  (hAngle : angleACB = 120)

theorem distance_BC (BC : ℝ) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end distance_BC_l248_248264


namespace fraction_addition_l248_248439

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248439


namespace R_and_D_expenditure_l248_248270

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end R_and_D_expenditure_l248_248270


namespace sum_fractions_eq_l248_248477

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248477


namespace number_of_extreme_points_l248_248894

-- Define the function's derivative
def f_derivative (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- State the theorem
theorem number_of_extreme_points : ∃ n : ℕ, n = 2 ∧ 
  (∀ x, (f_derivative x = 0 → ((f_derivative (x - ε) > 0 ∧ f_derivative (x + ε) < 0) ∨ 
                             (f_derivative (x - ε) < 0 ∧ f_derivative (x + ε) > 0))) → 
   (x = 1 ∨ x = 2)) :=
sorry

end number_of_extreme_points_l248_248894


namespace value_of_abs_m_minus_n_l248_248763

theorem value_of_abs_m_minus_n  (m n : ℝ) (h_eq : ∀ x, (x^2 - 2 * x + m) * (x^2 - 2 * x + n) = 0)
  (h_arith_seq : ∀ x₁ x₂ x₃ x₄ : ℝ, x₁ + x₂ = 2 ∧ x₃ + x₄ = 2 ∧ x₁ = 1 / 4 ∧ x₂ = 3 / 4 ∧ x₃ = 5 / 4 ∧ x₄ = 7 / 4) :
  |m - n| = 1 / 2 :=
by
  sorry

end value_of_abs_m_minus_n_l248_248763


namespace students_without_favorite_subject_l248_248782

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end students_without_favorite_subject_l248_248782


namespace tangent_line_equation_l248_248873

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def x0 : ℝ := 2

-- Define the value of function at the point of tangency
def y0 : ℝ := f x0

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem tangent_line_equation : ∃ (m b : ℝ), m = f' x0 ∧ b = y0 - m * x0 ∧ ∀ x, (y = m * x + b) ↔ (x = 2 → y = f x - f' x0 * (x - 2)) :=
by
  sorry

end tangent_line_equation_l248_248873


namespace arithmetic_progression_common_difference_l248_248964

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (h1 : 280 * x^2 - 61 * x * y + 3 * y^2 - 13 = 0) 
  (h2 : ∃ a d : ℤ, x = a + 3 * d ∧ y = a + 8 * d) : 
  ∃ d : ℤ, d = -5 := 
sorry

end arithmetic_progression_common_difference_l248_248964


namespace always_true_inequality_l248_248141

theorem always_true_inequality (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end always_true_inequality_l248_248141


namespace find_x_l248_248106

theorem find_x (x y z w : ℕ) (h1 : x = y + 8) (h2 : y = z + 15) (h3 : z = w + 25) (h4 : w = 90) : x = 138 :=
by
  sorry

end find_x_l248_248106


namespace unique_solution_pair_l248_248736

open Real

theorem unique_solution_pair :
  ∃! (x y : ℝ), y = (x-1)^2 ∧ x * y - y = -3 :=
sorry

end unique_solution_pair_l248_248736


namespace stayed_days_calculation_l248_248072

theorem stayed_days_calculation (total_cost : ℕ) (charge_1st_week : ℕ) (charge_additional_week : ℕ) (first_week_days : ℕ) :
  total_cost = 302 ∧ charge_1st_week = 18 ∧ charge_additional_week = 11 ∧ first_week_days = 7 →
  ∃ D : ℕ, D = 23 :=
by {
  sorry
}

end stayed_days_calculation_l248_248072


namespace range_of_m_l248_248307

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) →
  (1 < m) :=
by
  sorry

end range_of_m_l248_248307


namespace oliver_dishes_count_l248_248987

def total_dishes : ℕ := 42
def mango_salsa_dishes : ℕ := 5
def fresh_mango_dishes : ℕ := total_dishes / 6
def mango_jelly_dishes : ℕ := 2
def strawberry_dishes : ℕ := 3
def pineapple_dishes : ℕ := 5
def kiwi_dishes : ℕ := 4
def mango_dishes_oliver_picks_out : ℕ := 3

def total_mango_dishes : ℕ := mango_salsa_dishes + fresh_mango_dishes + mango_jelly_dishes
def mango_dishes_oliver_wont_eat : ℕ := total_mango_dishes - mango_dishes_oliver_picks_out
def max_strawberry_pineapple_dishes : ℕ := strawberry_dishes

def dishes_left_for_oliver : ℕ := total_dishes - mango_dishes_oliver_wont_eat - max_strawberry_pineapple_dishes

theorem oliver_dishes_count : dishes_left_for_oliver = 28 := 
by 
  sorry

end oliver_dishes_count_l248_248987


namespace lucas_notation_sum_l248_248924

-- Define what each representation in Lucas's notation means
def lucasValue : String → Int
| "0" => 0
| s => -((s.length) - 1)

-- Define the question as a Lean theorem
theorem lucas_notation_sum :
  lucasValue "000" + lucasValue "0000" = lucasValue "000000" :=
by
  sorry

end lucas_notation_sum_l248_248924


namespace calculate_expr_equals_243_l248_248859

theorem calculate_expr_equals_243 :
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by
  sorry

end calculate_expr_equals_243_l248_248859


namespace marathon_end_time_l248_248984

open Nat

def marathonStart := 15 * 60  -- 3:00 p.m. in minutes (15 hours * 60 minutes)
def marathonDuration := 780    -- Duration in minutes

theorem marathon_end_time : marathonStart + marathonDuration = 28 * 60 := -- 4:00 a.m. in minutes (28 hours * 60 minutes)
  sorry

end marathon_end_time_l248_248984


namespace infinite_geometric_sum_l248_248623

noncomputable def geometric_sequence (n : ℕ) : ℝ := 3 * (-1 / 2)^(n - 1)

theorem infinite_geometric_sum :
  ∑' n, geometric_sequence n = 2 :=
sorry

end infinite_geometric_sum_l248_248623


namespace volume_parallelepiped_eq_20_l248_248956

theorem volume_parallelepiped_eq_20 (k : ℝ) (h : k > 0) (hvol : abs (3 * k^2 - 7 * k - 6) = 20) :
  k = 13 / 3 :=
sorry

end volume_parallelepiped_eq_20_l248_248956


namespace count_numbers_between_200_and_500_with_digit_two_l248_248768

noncomputable def count_numbers_with_digit_two (a b : ℕ) : ℕ :=
  let numbers := list.filter (λ n, '2' ∈ n.digits 10) (list.range' a (b - a))
  numbers.length

theorem count_numbers_between_200_and_500_with_digit_two :
  count_numbers_with_digit_two 200 500 = 138 :=
by sorry

end count_numbers_between_200_and_500_with_digit_two_l248_248768


namespace y_capital_l248_248123

theorem y_capital (X Y Z : ℕ) (Pz : ℕ) (Z_months_after_start : ℕ) (total_profit Z_share : ℕ)
    (hx : X = 20000)
    (hz : Z = 30000)
    (hz_profit : Z_share = 14000)
    (htotal_profit : total_profit = 50000)
    (hZ_months : Z_months_after_start = 5)
  : Y = 25000 := 
by
  -- Here we would have a proof, skipped with sorry for now
  sorry

end y_capital_l248_248123


namespace negation_of_universal_l248_248951

theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x ≥ 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0^2 + x_0 < 0) :=
by
  sorry

end negation_of_universal_l248_248951


namespace average_speed_is_40_l248_248351

-- Define the total distance
def total_distance : ℝ := 640

-- Define the distance for the first half
def first_half_distance : ℝ := total_distance / 2

-- Define the average speed for the first half
def first_half_speed : ℝ := 80

-- Define the time taken for the first half
def first_half_time : ℝ := first_half_distance / first_half_speed

-- Define the multiplicative factor for time increase in the second half
def time_increase_factor : ℝ := 3

-- Define the time taken for the second half
def second_half_time : ℝ := first_half_time * time_increase_factor

-- Define the total time for the trip
def total_time : ℝ := first_half_time + second_half_time

-- Define the calculated average speed for the entire trip
def calculated_average_speed : ℝ := total_distance / total_time

-- State the theorem that the average speed for the entire trip is 40 miles per hour
theorem average_speed_is_40 : calculated_average_speed = 40 :=
by
  sorry

end average_speed_is_40_l248_248351


namespace symm_central_origin_l248_248594

noncomputable def f₁ (x : ℝ) : ℝ := 3^x

noncomputable def f₂ (x : ℝ) : ℝ := -3^(-x)

theorem symm_central_origin :
  ∀ x : ℝ, ∃ x' y y' : ℝ, (f₁ x = y) ∧ (f₂ x' = y') ∧ (x' = -x) ∧ (y' = -y) :=
by
  sorry

end symm_central_origin_l248_248594


namespace is_factorization_l248_248730

-- Define the conditions
def A_transformation : Prop := (∀ x : ℝ, (x + 1) * (x - 1) = x ^ 2 - 1)
def B_transformation : Prop := (∀ m : ℝ, m ^ 2 + m - 4 = (m + 3) * (m - 2) + 2)
def C_transformation : Prop := (∀ x : ℝ, x ^ 2 + 2 * x = x * (x + 2))
def D_transformation : Prop := (∀ x : ℝ, 2 * x ^ 2 + 2 * x = 2 * x ^ 2 * (1 + (1 / x)))

-- The goal is to prove that transformation C is a factorization
theorem is_factorization : C_transformation :=
by
  sorry

end is_factorization_l248_248730


namespace polynomial_sum_at_points_l248_248876

def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

theorem polynomial_sum_at_points :
  P 19.1 + P (-19.1) = 5 := by
  sorry

end polynomial_sum_at_points_l248_248876


namespace solve_equation_l248_248815

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end solve_equation_l248_248815


namespace fraction_addition_l248_248546

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248546


namespace trigonometric_identity_l248_248181

noncomputable def cos_alpha (α : ℝ) : ℝ := -Real.sqrt (1 / (1 + (tan α)^2))
noncomputable def sin_alpha (α : ℝ) : ℝ := Real.sqrt (1 - (cos_alpha α)^2)

theorem trigonometric_identity
  (α : ℝ) (h1 : tan α = -2) (h2 : (π / 2) < α ∧ α < π) :
  cos_alpha α + sin_alpha α = Real.sqrt(5) / 5 :=
sorry

end trigonometric_identity_l248_248181


namespace reciprocal_neg3_l248_248087

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l248_248087


namespace y_directly_varies_as_square_l248_248035

theorem y_directly_varies_as_square (k : ℚ) (y : ℚ) (x : ℚ) 
  (h1 : y = k * x ^ 2) (h2 : y = 18) (h3 : x = 3) : 
  ∃ y : ℚ, ∀ x : ℚ, x = 6 → y = 72 :=
by
  sorry

end y_directly_varies_as_square_l248_248035


namespace minimum_value_of_sum_of_squares_l248_248947

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) : 
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end minimum_value_of_sum_of_squares_l248_248947


namespace race_distance_l248_248043

variables (a b c d : ℝ)
variables (h1 : d / a = (d - 30) / b)
variables (h2 : d / b = (d - 15) / c)
variables (h3 : d / a = (d - 40) / c)

theorem race_distance : d = 90 :=
by 
  sorry

end race_distance_l248_248043


namespace additional_grassy_area_l248_248263

theorem additional_grassy_area (r1 r2 : ℝ) (r1_pos : r1 = 10) (r2_pos : r2 = 35) : 
  let A1 := π * r1^2
  let A2 := π * r2^2
  (A2 - A1) = 1125 * π :=
by 
  sorry

end additional_grassy_area_l248_248263


namespace max_value_of_a_l248_248907

noncomputable def maximum_a : ℝ := 1/3

theorem max_value_of_a :
  ∀ x : ℝ, 1 + maximum_a * Real.cos x ≥ (2/3) * Real.sin ((Real.pi / 2) + 2 * x) :=
by 
  sorry

end max_value_of_a_l248_248907


namespace add_fractions_l248_248493

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248493


namespace coconut_grove_l248_248637

theorem coconut_grove (x Y : ℕ) (h1 : 3 * x ≠ 0) (h2 : (x+3) * 60 + x * Y + (x-3) * 180 = 3 * x * 100) (hx : x = 6) : Y = 120 :=
by 
  sorry

end coconut_grove_l248_248637


namespace geometric_sequence_a1_value_l248_248755

variable {a_1 q : ℝ}

theorem geometric_sequence_a1_value
  (h1 : a_1 * q^2 = 1)
  (h2 : a_1 * q^4 + (3 / 2) * a_1 * q^3 = 1) :
  a_1 = 4 := by
  sorry

end geometric_sequence_a1_value_l248_248755


namespace sum_of_fractions_l248_248433

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248433


namespace selling_price_eq_l248_248080

theorem selling_price_eq (cp sp L : ℕ) (h_cp: cp = 47) (h_L : L = cp - 40) (h_profit_loss_eq : sp - cp = L) :
  sp = 54 :=
by
  sorry

end selling_price_eq_l248_248080


namespace fraction_addition_l248_248542

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248542


namespace value_of_m_l248_248758

def p (m : ℝ) : Prop :=
  4 < m ∧ m < 10

def q (m : ℝ) : Prop :=
  8 < m ∧ m < 12

theorem value_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
by
  sorry

end value_of_m_l248_248758


namespace reciprocal_of_neg3_l248_248085

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l248_248085


namespace height_of_tank_B_l248_248361

noncomputable def height_tank_A : ℝ := 5
noncomputable def circumference_tank_A : ℝ := 4
noncomputable def circumference_tank_B : ℝ := 10
noncomputable def capacity_ratio : ℝ := 0.10000000000000002

theorem height_of_tank_B {h_B : ℝ} 
  (h_tank_A : height_tank_A = 5)
  (c_tank_A : circumference_tank_A = 4)
  (c_tank_B : circumference_tank_B = 10)
  (capacity_percentage : capacity_ratio = 0.10000000000000002)
  (V_A : ℝ := π * (2 / π)^2 * height_tank_A)
  (V_B : ℝ := π * (5 / π)^2 * h_B)
  (capacity_relation : V_A = capacity_ratio * V_B) :
  h_B = 8 :=
sorry

end height_of_tank_B_l248_248361


namespace highest_y_coordinate_l248_248863

-- Define the conditions
def ellipse_condition (x y : ℝ) : Prop :=
  (x^2 / 25) + ((y - 3)^2 / 9) = 1

-- The theorem to prove
theorem highest_y_coordinate : ∃ x : ℝ, ∀ y : ℝ, ellipse_condition x y → y ≤ 6 :=
sorry

end highest_y_coordinate_l248_248863


namespace exist_three_integers_l248_248595

theorem exist_three_integers :
  ∃ (a b c : ℤ), a * b - c = 2018 ∧ b * c - a = 2018 ∧ c * a - b = 2018 := 
sorry

end exist_three_integers_l248_248595


namespace solution_set_of_f_gt_0_range_of_m_l248_248923

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem solution_set_of_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 / 3} ∪ {x | x > 3} :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 + 2 * m^2 < 4 * m) ↔ -1 / 2 < m ∧ m < 5 / 2 :=
by sorry

end solution_set_of_f_gt_0_range_of_m_l248_248923


namespace additionalPeopleNeededToMowLawn_l248_248613

def numberOfPeopleNeeded (people : ℕ) (hours : ℕ) : ℕ :=
  (people * 8) / hours

theorem additionalPeopleNeededToMowLawn : numberOfPeopleNeeded 4 3 - 4 = 7 :=
by
  sorry

end additionalPeopleNeededToMowLawn_l248_248613


namespace zilla_savings_l248_248117

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l248_248117


namespace find_positive_n_l248_248289

theorem find_positive_n (n x : ℝ) (h : 16 * x ^ 2 + n * x + 4 = 0) : n = 16 :=
by
  sorry

end find_positive_n_l248_248289


namespace actual_revenue_percentage_l248_248348

def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.25 * R
def actual_revenue (R : ℝ) := 0.75 * R

theorem actual_revenue_percentage (R : ℝ) : 
  (actual_revenue R / projected_revenue R) * 100 = 60 :=
by
  sorry

end actual_revenue_percentage_l248_248348


namespace fraction_addition_l248_248445

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248445


namespace find_angle_A_l248_248024

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) 
  (h : 1 + (Real.tan A / Real.tan B) = 2 * c / b) : 
  A = Real.pi / 3 :=
sorry

end find_angle_A_l248_248024


namespace add_fractions_result_l248_248414

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248414


namespace tangent_line_to_circle_polar_l248_248642

-- Definitions
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def point_polar_coordinates (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4
def tangent_line_polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Theorem Statement
theorem tangent_line_to_circle_polar {ρ θ : ℝ} :
  (∃ ρ θ, polar_circle_equation ρ θ) →
  (∃ ρ θ, point_polar_coordinates ρ θ) →
  tangent_line_polar_equation ρ θ :=
sorry

end tangent_line_to_circle_polar_l248_248642


namespace fraction_addition_l248_248460

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248460


namespace greatest_possible_integer_l248_248048

theorem greatest_possible_integer (m : ℕ) (h1 : m < 150) (h2 : ∃ a : ℕ, m = 10 * a - 2) (h3 : ∃ b : ℕ, m = 9 * b - 4) : m = 68 := 
  by sorry

end greatest_possible_integer_l248_248048


namespace reciprocal_of_neg_three_l248_248081

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l248_248081


namespace find_numbers_with_lcm_gcd_l248_248950

theorem find_numbers_with_lcm_gcd :
  ∃ a b : ℕ, lcm a b = 90 ∧ gcd a b = 6 ∧ ((a = 18 ∧ b = 30) ∨ (a = 30 ∧ b = 18)) :=
by
  sorry

end find_numbers_with_lcm_gcd_l248_248950


namespace mike_total_cans_l248_248803

theorem mike_total_cans (monday_cans : ℕ) (tuesday_cans : ℕ) (total_cans : ℕ) : 
  monday_cans = 71 ∧ tuesday_cans = 27 ∧ total_cans = monday_cans + tuesday_cans → total_cans = 98 :=
by
  sorry

end mike_total_cans_l248_248803


namespace michael_weight_loss_in_may_l248_248929

-- Defining the conditions
def weight_loss_goal : ℕ := 10
def weight_loss_march : ℕ := 3
def weight_loss_april : ℕ := 4

-- Statement of the problem to prove
theorem michael_weight_loss_in_may (weight_loss_goal weight_loss_march weight_loss_april : ℕ) :
  weight_loss_goal - (weight_loss_march + weight_loss_april) = 3 :=
by
  sorry

end michael_weight_loss_in_may_l248_248929


namespace ants_crushed_l248_248722

theorem ants_crushed {original_ants left_ants crushed_ants : ℕ} 
  (h1 : original_ants = 102) 
  (h2 : left_ants = 42) 
  (h3 : crushed_ants = original_ants - left_ants) : 
  crushed_ants = 60 := 
by
  sorry

end ants_crushed_l248_248722


namespace number_of_students_is_20_l248_248677

-- Define the constants and conditions
def average_age_all_students (N : ℕ) : ℕ := 20
def average_age_9_students : ℕ := 11
def average_age_10_students : ℕ := 24
def age_20th_student : ℕ := 61

theorem number_of_students_is_20 (N : ℕ) 
  (h1 : N * average_age_all_students N = 99 + 240 + 61) 
  (h2 : 99 = 9 * average_age_9_students) 
  (h3 : 240 = 10 * average_age_10_students) 
  (h4 : N = 9 + 10 + 1) : N = 20 :=
sorry

end number_of_students_is_20_l248_248677


namespace sandwiches_final_count_l248_248942

def sandwiches_left (initial : ℕ) (eaten_by_ruth : ℕ) (given_to_brother : ℕ) (eaten_by_first_cousin : ℕ) (eaten_by_other_cousins : ℕ) : ℕ :=
  initial - (eaten_by_ruth + given_to_brother + eaten_by_first_cousin + eaten_by_other_cousins)

theorem sandwiches_final_count :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end sandwiches_final_count_l248_248942


namespace add_fractions_l248_248527

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248527


namespace sum_of_fractions_l248_248434

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248434


namespace fraction_addition_l248_248465

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248465


namespace prove_a_plus_b_l248_248200

-- Defining the function f(x)
def f (a b x: ℝ) : ℝ := a * x^2 + b * x

-- The given conditions
variable (a b : ℝ)
variable (h1 : f a b (a - 1) = f a b (2 * a))
variable (h2 : ∀ x : ℝ, f a b x = f a b (-x))

-- The objective is to show a + b = 1/3
theorem prove_a_plus_b (a b : ℝ) (h1 : f a b (a - 1) = f a b (2 * a)) (h2 : ∀ x : ℝ, f a b x = f a b (-x)) :
  a + b = 1 / 3 := 
sorry

end prove_a_plus_b_l248_248200


namespace fraction_addition_l248_248451

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248451


namespace exists_f_prime_eq_inverses_l248_248749

theorem exists_f_prime_eq_inverses (f : ℝ → ℝ) (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : ContinuousOn f (Set.Icc a b))
  (h4 : DifferentiableOn ℝ f (Set.Ioo a b)) :
  ∃ c ∈ Set.Ioo a b, (deriv f c) = (1 / (a - c)) + (1 / (b - c)) + (1 / (a + b)) :=
by
  sorry

end exists_f_prime_eq_inverses_l248_248749


namespace h_f_equals_h_g_l248_248056

def f (x : ℝ) := x^2 - x + 1

def g (x : ℝ) := -x^2 + x + 1

def h (x : ℝ) := (x - 1)^2

theorem h_f_equals_h_g : ∀ x : ℝ, h (f x) = h (g x) :=
by
  intro x
  unfold f g h
  sorry

end h_f_equals_h_g_l248_248056


namespace polar_line_through_center_perpendicular_to_axis_l248_248046

-- We define our conditions
def circle_in_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

def center_of_circle (C : ℝ × ℝ) : Prop := C = (2, 0)

def line_in_rectangular (x : ℝ) : Prop := x = 2

-- We now state the proof problem
theorem polar_line_through_center_perpendicular_to_axis (ρ θ : ℝ) : 
  (∃ C, center_of_circle C ∧ (∃ x, line_in_rectangular x)) →
  (circle_in_polar ρ θ → ρ * Real.cos θ = 2) :=
by
  sorry

end polar_line_through_center_perpendicular_to_axis_l248_248046


namespace sticks_form_equilateral_triangle_l248_248150

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l248_248150


namespace factor_expression_l248_248868

theorem factor_expression (x : ℝ) : 
  3 * x^2 * (x - 5) + 4 * x * (x - 5) + 6 * (x - 5) = (3 * x^2 + 4 * x + 6) * (x - 5) :=
  sorry

end factor_expression_l248_248868


namespace sum_fractions_eq_l248_248481

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248481


namespace shuttle_speed_l248_248119

theorem shuttle_speed (v : ℕ) (h : v = 9) : v * 3600 = 32400 :=
by
  sorry

end shuttle_speed_l248_248119


namespace sum_of_fractions_l248_248427

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248427


namespace range_of_a_l248_248759

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

theorem range_of_a (a : ℝ) : (A ∩ B a) = B a ↔ (2 ≤ a ∧ a < 10) ∨ (a = 1) := by
  sorry

end range_of_a_l248_248759


namespace max_triangle_area_l248_248213

theorem max_triangle_area (AB BC AC : ℝ) (h1 : AB = 16) (h2 : ∃ x, BC = 3 * x ∧ AC = 4 * x ∧ x > 16 / 7 ∧ x < 16) :
  ∃ (x : ℝ), let BC := 3 * x
               let AC := 4 * x
               let s := (AB + BC + AC) / 2
               let area := s * (s - AB) * (s - BC) * (s - AC)
               area ≤ 128^2 :=
by
  sorry

end max_triangle_area_l248_248213


namespace b6_b8_value_l248_248170

def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d
def nonzero_sequence (a : ℕ → ℕ) := ∀ n : ℕ, a n ≠ 0
def geometric_seq (b : ℕ → ℕ) := ∃ r : ℕ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℕ) (d : ℕ) 
  (h_arith : arithmetic_seq a) 
  (h_nonzero : nonzero_sequence a) 
  (h_cond1 : 2 * a 3 = a 1^2) 
  (h_cond2 : a 1 = d)
  (h_geo : geometric_seq b)
  (h_b13 : b 13 = a 2)
  (h_b1 : b 1 = a 1) :
  b 6 * b 8 = 72 := 
sorry

end b6_b8_value_l248_248170


namespace fraction_red_after_tripling_l248_248319

-- Define the initial conditions
def initial_fraction_blue : ℚ := 4 / 7
def initial_fraction_red : ℚ := 1 - initial_fraction_blue
def triple_red_fraction (initial_red : ℚ) : ℚ := 3 * initial_red

-- Theorem statement
theorem fraction_red_after_tripling :
  let x := 1 -- Any number since it will cancel out
  let initial_red_marble := initial_fraction_red * x
  let total_marble := x
  let new_red_marble := triple_red_fraction initial_red_marble
  let new_total_marble := initial_fraction_blue * x + new_red_marble
  (new_red_marble / new_total_marble) = 9 / 13 :=
by
  sorry

end fraction_red_after_tripling_l248_248319


namespace arithmetic_sequence_x_y_sum_l248_248105

theorem arithmetic_sequence_x_y_sum :
  ∀ (a d x y: ℕ), 
  a = 3 → d = 6 → 
  (∀ (n: ℕ), n ≥ 1 → a + (n-1) * d = 3 + (n-1) * 6) →
  (a + 5 * d = x) → (a + 6 * d = y) → 
  (y = 45 - d) → x + y = 72 :=
by
  intros a d x y h_a h_d h_seq h_x h_y h_y_equals
  sorry

end arithmetic_sequence_x_y_sum_l248_248105


namespace maximal_p_sum_consecutive_l248_248872

theorem maximal_p_sum_consecutive (k : ℕ) (h1 : k = 31250) : 
  ∃ p a : ℕ, p * (2 * a + p - 1) = k ∧ ∀ p' a', (p' * (2 * a' + p' - 1) = k) → p' ≤ p := by
  sorry

end maximal_p_sum_consecutive_l248_248872


namespace fraction_addition_l248_248530

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248530


namespace probability_of_distance_less_than_8000_l248_248818

def distance (c1 c2 : String) : ℕ :=
  if (c1 = "Tokyo" ∧ c2 = "Sydney") ∨ (c1 = "Sydney" ∧ c2 = "Tokyo") then 4800 else
  if (c1 = "Tokyo" ∧ c2 = "New York") ∨ (c1 = "New York" ∧ c2 = "Tokyo") then 6760 else
  if (c1 = "Tokyo" ∧ c2 = "Paris") ∨ (c1 = "Paris" ∧ c2 = "Tokyo") then 6037 else
  if (c1 = "Sydney" ∧ c2 = "New York") ∨ (c1 = "New York" ∧ c2 = "Sydney") then 9954 else
  if (c1 = "Sydney" ∧ c2 = "Paris") ∨ (c1 = "Paris" ∧ c2 = "Sydney") then 10560 else
  if (c1 = "New York" ∧ c2 = "Paris") ∨ (c1 = "Paris" ∧ c2 = "New York") then 3624 else 0

def total_pairs := 6
def valid_pairs := 4

theorem probability_of_distance_less_than_8000 :
  valid_pairs / total_pairs = (2 : ℚ) / 3 :=
by
  sorry

end probability_of_distance_less_than_8000_l248_248818


namespace middle_managers_sample_count_l248_248713

def employees_total : ℕ := 1000
def managers_middle_total : ℕ := 150
def sample_total : ℕ := 200

theorem middle_managers_sample_count :
  sample_total * managers_middle_total / employees_total = 30 := by
  sorry

end middle_managers_sample_count_l248_248713


namespace probability_samantha_in_sam_not_l248_248809

noncomputable def probability_in_picture_but_not (time_samantha : ℕ) (lap_samantha : ℕ) (time_sam : ℕ) (lap_sam : ℕ) : ℚ :=
  let seconds_raced := 900
  let samantha_laps := seconds_raced / time_samantha
  let sam_laps := seconds_raced / time_sam
  let start_line_samantha := (samantha_laps - (samantha_laps % 1)) * time_samantha + ((samantha_laps % 1) * lap_samantha)
  let start_line_sam := (sam_laps - (sam_laps % 1)) * time_sam + ((sam_laps % 1) * lap_sam)
  let in_picture_duration := 80
  let overlapping_time := 30
  overlapping_time / in_picture_duration

theorem probability_samantha_in_sam_not : probability_in_picture_but_not 120 60 75 25 = 3 / 8 := by
  sorry

end probability_samantha_in_sam_not_l248_248809


namespace number_division_l248_248702

theorem number_division (n : ℕ) (h1 : n / 25 = 5) (h2 : n % 25 = 2) : n = 127 :=
by
  sorry

end number_division_l248_248702


namespace add_fractions_l248_248510

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248510


namespace zinc_weight_in_mixture_l248_248836

theorem zinc_weight_in_mixture (total_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) (total_parts : ℝ) (fraction_zinc : ℝ) (weight_zinc : ℝ) :
  zinc_ratio = 9 ∧ copper_ratio = 11 ∧ total_weight = 70 ∧ total_parts = zinc_ratio + copper_ratio ∧
  fraction_zinc = zinc_ratio / total_parts ∧ weight_zinc = fraction_zinc * total_weight →
  weight_zinc = 31.5 :=
by
  intros h
  sorry

end zinc_weight_in_mixture_l248_248836


namespace compound_interest_rate_l248_248700

theorem compound_interest_rate (P A : ℝ) (t n : ℝ)
  (hP : P = 5000) 
  (hA : A = 7850)
  (ht : t = 8)
  (hn : n = 1) : 
  ∃ r : ℝ, 0.057373 ≤ (r * 100) ∧ (r * 100) ≤ 5.7373 :=
by
  sorry

end compound_interest_rate_l248_248700


namespace find_angle_B_l248_248207

open Real

theorem find_angle_B (A B : ℝ) 
  (h1 : 0 < B ∧ B < A ∧ A < π/2)
  (h2 : cos A = 1/7) 
  (h3 : cos (A - B) = 13/14) : 
  B = π/3 :=
sorry

end find_angle_B_l248_248207


namespace add_fractions_l248_248492

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248492


namespace sum_fractions_eq_l248_248474

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248474


namespace evaluate_expression_l248_248673

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)
variable (h4 : ∀ x, g (g_inv x) = x)
variable (h5 : ∀ x, g_inv (g x) = x)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 :=
by
  -- The proof is omitted
  sorry

end evaluate_expression_l248_248673


namespace price_reduction_correct_l248_248727

theorem price_reduction_correct (P : ℝ) : 
  let first_reduction := 0.92 * P
  let second_reduction := first_reduction * 0.90
  second_reduction = 0.828 * P := 
by 
  sorry

end price_reduction_correct_l248_248727


namespace cooking_time_eq_80_l248_248247

-- Define the conditions
def hushpuppies_per_guest : Nat := 5
def number_of_guests : Nat := 20
def hushpuppies_per_batch : Nat := 10
def time_per_batch : Nat := 8

-- Calculate total number of hushpuppies needed
def total_hushpuppies : Nat := hushpuppies_per_guest * number_of_guests

-- Calculate number of batches needed
def number_of_batches : Nat := total_hushpuppies / hushpuppies_per_batch

-- Calculate total time needed
def total_time_needed : Nat := number_of_batches * time_per_batch

-- Statement to prove the correctness
theorem cooking_time_eq_80 : total_time_needed = 80 := by
  sorry

end cooking_time_eq_80_l248_248247


namespace add_fractions_l248_248585

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248585


namespace inequality_subtraction_l248_248032

variable (a b : ℝ)

theorem inequality_subtraction (h : a > b) : a - 5 > b - 5 :=
sorry

end inequality_subtraction_l248_248032


namespace reciprocal_of_neg_three_l248_248083

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l248_248083


namespace sum_of_fractions_l248_248437

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248437


namespace expand_product_l248_248002

theorem expand_product (x : ℝ) : 
  5 * (x + 6) * (x^2 + 2 * x + 3) = 5 * x^3 + 40 * x^2 + 75 * x + 90 := 
by 
  sorry

end expand_product_l248_248002


namespace alexis_shirt_expense_l248_248854

theorem alexis_shirt_expense :
  let B := 200
  let E_pants := 46
  let E_coat := 38
  let E_socks := 11
  let E_belt := 18
  let E_shoes := 41
  let L := 16
  let S := B - (E_pants + E_coat + E_socks + E_belt + E_shoes + L)
  S = 30 :=
by
  sorry

end alexis_shirt_expense_l248_248854


namespace time_to_paint_one_house_l248_248047

theorem time_to_paint_one_house (houses : ℕ) (total_time_hours : ℕ) (total_time_minutes : ℕ) 
  (minutes_per_hour : ℕ) (h1 : houses = 9) (h2 : total_time_hours = 3) 
  (h3 : minutes_per_hour = 60) (h4 : total_time_minutes = total_time_hours * minutes_per_hour) : 
  (total_time_minutes / houses) = 20 :=
by
  sorry

end time_to_paint_one_house_l248_248047


namespace factorize_difference_of_squares_l248_248600

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
by 
  sorry

end factorize_difference_of_squares_l248_248600


namespace add_fractions_l248_248483

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248483


namespace product_of_positive_solutions_l248_248161

theorem product_of_positive_solutions :
  ∃ n : ℕ, ∃ p : ℕ, Prime p ∧ (n^2 - 41*n + 408 = p) ∧ (∀ m : ℕ, (Prime p ∧ (m^2 - 41*m + 408 = p)) → m = n) ∧ (n = 406) := 
sorry

end product_of_positive_solutions_l248_248161


namespace interior_angle_of_arithmetic_sequence_triangle_l248_248779

theorem interior_angle_of_arithmetic_sequence_triangle :
  ∀ (α d : ℝ), (α - d) + α + (α + d) = 180 → α = 60 :=
by 
  sorry

end interior_angle_of_arithmetic_sequence_triangle_l248_248779


namespace watch_correction_l248_248991

def watch_loss_per_day : ℚ := 13 / 4

def hours_from_march_15_noon_to_march_22_9am : ℚ := 7 * 24 + 21

def per_hour_loss : ℚ := watch_loss_per_day / 24

def total_loss_in_minutes : ℚ := hours_from_march_15_noon_to_march_22_9am * per_hour_loss

theorem watch_correction :
  total_loss_in_minutes = 2457 / 96 :=
by
  sorry

end watch_correction_l248_248991


namespace selection_ways_l248_248606

def ways_to_select_president_and_secretary (n : Nat) : Nat :=
  n * (n - 1)

theorem selection_ways :
  ways_to_select_president_and_secretary 5 = 20 :=
by
  sorry

end selection_ways_l248_248606


namespace least_prime_value_l248_248057

/-- Let q be a set of 12 distinct prime numbers. If the sum of the integers in q is odd,
the product of all the integers in q is divisible by a perfect square, and the number x is a member of q,
then the least value that x can be is 2. -/
theorem least_prime_value (q : Finset ℕ) (hq_distinct : q.card = 12) (hq_prime : ∀ p ∈ q, Nat.Prime p) 
    (hq_odd_sum : q.sum id % 2 = 1) (hq_perfect_square_div : ∃ k, q.prod id % (k * k) = 0) (x : ℕ)
    (hx : x ∈ q) : x = 2 :=
sorry

end least_prime_value_l248_248057


namespace number_of_students_l248_248846

theorem number_of_students (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 4) : n = 13 :=
by
  sorry

end number_of_students_l248_248846


namespace length_of_parallelepiped_l248_248999

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248999


namespace inequality_holds_l248_248611

theorem inequality_holds (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  (a^3 / (a^3 + 15 * b * c * d))^(1/2) ≥ a^(15/8) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) :=
sorry

end inequality_holds_l248_248611


namespace problem_maximum_marks_l248_248839

theorem problem_maximum_marks (M : ℝ) (h : 0.92 * M = 184) : M = 200 :=
sorry

end problem_maximum_marks_l248_248839


namespace fraction_addition_l248_248544

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248544


namespace sum_of_fractions_l248_248428

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248428


namespace sum_of_coordinates_l248_248037

-- Define the conditions for m and n
def m : ℤ := -3
def n : ℤ := 2

-- State the proposition based on the conditions
theorem sum_of_coordinates : m + n = -1 := 
by 
  -- Provide an incomplete proof skeleton with "sorry" to skip the proof
  sorry

end sum_of_coordinates_l248_248037


namespace four_points_nonexistent_l248_248143

theorem four_points_nonexistent :
  ¬ (∃ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 8 ∧ 
    dist C D = 8 ∧ 
    dist A C = 10 ∧ 
    dist B D = 10 ∧ 
    dist A D = 13 ∧ 
    dist B C = 13) :=
by
  sorry

end four_points_nonexistent_l248_248143


namespace locus_of_circumcenters_is_circle_l248_248193

open Real EuclideanGeometry

-- Given the setup of two intersecting circles and points P, Q, C, A, B as described
variable {C : Point}
variable {P Q : Point}
variable {circle1 circle2 : Circle}

-- Points P and Q lie on both circles
axiom P_on_circles : point_on_circle P circle1 ∧ point_on_circle P circle2
axiom Q_on_circles : point_on_circle Q circle1 ∧ point_on_circle Q circle2

-- Point C is an arbitrary point distinct from P and Q on the first circle
axiom C_on_circle1 : point_on_circle C circle1
axiom C_ne_P : C ≠ P
axiom C_ne_Q : C ≠ Q

-- Points A and B are the second intersections of lines CP and CQ with the second circle
axiom A_on_circle2 : point_on_circle (line_intersection (line_through C P) circle2) circle2
axiom B_on_circle2 : point_on_circle (line_intersection (line_through C Q) circle2) circle2

-- Prove that the locus of the circumcenters of ΔABC is a circle
theorem locus_of_circumcenters_is_circle :
  ∃ (circ_center : Circle), 
  ∀ (C : Point), 
  C ≠ P → C ≠ Q → point_on_circle C circle1 →
  let A := line_intersection (line_through C P) circle2 in
  let B := line_intersection (line_through C Q) circle2 in
  circumcenter A B C ∈ circ_center :=
sorry

end locus_of_circumcenters_is_circle_l248_248193


namespace zilla_savings_l248_248112

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l248_248112


namespace total_air_removed_after_5_strokes_l248_248632

theorem total_air_removed_after_5_strokes:
  let initial_air := 1
  let remaining_air_after_first_stroke := initial_air * (2 / 3)
  let remaining_air_after_second_stroke := remaining_air_after_first_stroke * (3 / 4)
  let remaining_air_after_third_stroke := remaining_air_after_second_stroke * (4 / 5)
  let remaining_air_after_fourth_stroke := remaining_air_after_third_stroke * (5 / 6)
  let remaining_air_after_fifth_stroke := remaining_air_after_fourth_stroke * (6 / 7)
  initial_air - remaining_air_after_fifth_stroke = 5 / 7 := by
  sorry

end total_air_removed_after_5_strokes_l248_248632


namespace add_fractions_l248_248494

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248494


namespace worms_stolen_correct_l248_248346

-- Given conditions translated into Lean statements
def num_babies : ℕ := 6
def worms_per_baby_per_day : ℕ := 3
def papa_bird_worms : ℕ := 9
def mama_bird_initial_worms : ℕ := 13
def additional_worms_needed : ℕ := 34

-- From the conditions, determine the total number of worms needed for 3 days
def total_worms_needed : ℕ := worms_per_baby_per_day * num_babies * 3

-- Calculate how many worms they will have after catching additional worms
def total_worms_after_catching_more : ℕ := papa_bird_worms + mama_bird_initial_worms + additional_worms_needed

-- Amount suspected to be stolen
def worms_stolen : ℕ := total_worms_after_catching_more - total_worms_needed

theorem worms_stolen_correct : worms_stolen = 2 :=
by sorry

end worms_stolen_correct_l248_248346


namespace reciprocal_neg3_l248_248088

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l248_248088


namespace salmon_total_l248_248740

def num_male : ℕ := 712261
def num_female : ℕ := 259378
def num_total : ℕ := 971639

theorem salmon_total :
  num_male + num_female = num_total :=
by
  -- proof will be provided here
  sorry

end salmon_total_l248_248740


namespace solve_for_x_l248_248383

theorem solve_for_x (x : ℝ) (h : 4 / (1 + 3 / x) = 1) : x = 1 :=
sorry

end solve_for_x_l248_248383


namespace bus_fare_with_train_change_in_total_passengers_l248_248972

variables (p : ℝ) (q : ℝ) (TC : ℝ → ℝ)
variables (p_train : ℝ) (train_capacity : ℝ)

-- Demand function
def demand_function (p : ℝ) : ℝ := 4200 - 100 * p

-- Train fare is fixed
def train_fare : ℝ := 4

-- Train capacity
def train_cap : ℝ := 800

-- Bus total cost function
def total_cost (y : ℝ) : ℝ := 10 * y + 225

-- Case when there is competition (train available)
def optimal_bus_fare_with_train : ℝ := 22

-- Case when there is no competition (train service is closed)
def optimal_bus_fare_without_train : ℝ := 26

-- Change in the number of passengers when the train service closes
def change_in_passengers : ℝ := 400

-- Theorems to prove
theorem bus_fare_with_train : optimal_bus_fare_with_train = 22 := sorry
theorem change_in_total_passengers : change_in_passengers = 400 := sorry

end bus_fare_with_train_change_in_total_passengers_l248_248972


namespace inequality_proof_l248_248610

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / Real.sqrt (a^2 + 8 * b * c)) + 
  (b / Real.sqrt (b^2 + 8 * c * a)) + 
  (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proof_l248_248610


namespace math_problem_l248_248011

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 3) : a^(2008 : ℕ) + b^(2008 : ℕ) + c^(2008 : ℕ) = 3 :=
by 
  let h1' : a + b + c = 3 := h1
  let h2' : a^2 + b^2 + c^2 = 3 := h2
  sorry

end math_problem_l248_248011


namespace perpendicular_lines_a_eq_2_l248_248318

/-- Given two lines, ax + 2y + 2 = 0 and x - y - 2 = 0, prove that if these lines are perpendicular, then a = 2. -/
theorem perpendicular_lines_a_eq_2 {a : ℝ} :
  (∃ a, (a ≠ 0)) → (∃ x y, ((ax + 2*y + 2 = 0) ∧ (x - y - 2 = 0)) → - (a / 2) * 1 = -1) → a = 2 :=
by
  sorry

end perpendicular_lines_a_eq_2_l248_248318


namespace factor_diff_of_squares_l248_248598

-- Define the expression t^2 - 49 and show it is factored as (t - 7)(t + 7)
theorem factor_diff_of_squares (t : ℝ) : t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end factor_diff_of_squares_l248_248598


namespace lyssa_fewer_correct_l248_248345

-- Define the total number of items in the exam
def total_items : ℕ := 75

-- Define the number of mistakes made by Lyssa
def lyssa_mistakes : ℕ := total_items * 20 / 100  -- 20% of 75

-- Define the number of correct answers by Lyssa
def lyssa_correct : ℕ := total_items - lyssa_mistakes

-- Define the number of mistakes made by Precious
def precious_mistakes : ℕ := 12

-- Define the number of correct answers by Precious
def precious_correct : ℕ := total_items - precious_mistakes

-- Statement to prove Lyssa got 3 fewer correct answers than Precious
theorem lyssa_fewer_correct : (precious_correct - lyssa_correct) = 3 := by
  sorry

end lyssa_fewer_correct_l248_248345


namespace two_digit_integer_divides_491_remainder_59_l248_248381

theorem two_digit_integer_divides_491_remainder_59 :
  ∃ n Q : ℕ, (n = 10 * x + y) ∧ (0 < x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) ∧ (491 = n * Q + 59) ∧ (n = 72) :=
by
  sorry

end two_digit_integer_divides_491_remainder_59_l248_248381


namespace add_fractions_l248_248505

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248505


namespace factorial_fraction_simplification_l248_248732

theorem factorial_fraction_simplification : 
  (4 * (Nat.factorial 6) + 24 * (Nat.factorial 5)) / (Nat.factorial 7) = 8 / 7 :=
by
  sorry

end factorial_fraction_simplification_l248_248732


namespace number_of_whole_numbers_between_sqrt2_and_3e_is_7_l248_248311

noncomputable def number_of_whole_numbers_between_sqrt2_and_3e : ℕ :=
  let sqrt2 : ℝ := Real.sqrt 2
  let e : ℝ := Real.exp 1
  let small_int := Nat.ceil sqrt2 -- This is 2
  let large_int := Nat.floor (3 * e) -- This is 8
  large_int - small_int + 1 -- The number of integers between small_int and large_int (inclusive)

theorem number_of_whole_numbers_between_sqrt2_and_3e_is_7 :
  number_of_whole_numbers_between_sqrt2_and_3e = 7 := by
  sorry

end number_of_whole_numbers_between_sqrt2_and_3e_is_7_l248_248311


namespace probability_of_same_suit_or_number_but_not_both_l248_248267

def same_suit_or_number_but_not_both : Prop :=
  let total_outcomes := 52 * 52
  let prob_same_suit := 12 / 51
  let prob_same_number := 3 / 51
  let prob_same_suit_and_number := 1 / 51
  (prob_same_suit + prob_same_number - 2 * prob_same_suit_and_number) = 15 / 52

theorem probability_of_same_suit_or_number_but_not_both :
  same_suit_or_number_but_not_both :=
by sorry

end probability_of_same_suit_or_number_but_not_both_l248_248267


namespace prime_sum_is_prime_l248_248368

def prime : ℕ → Prop := sorry 

theorem prime_sum_is_prime (A B : ℕ) (hA : prime A) (hB : prime B) (hAB : prime (A - B)) (hABB : prime (A - B - B)) : prime (A + B + (A - B) + (A - B - B)) :=
sorry

end prime_sum_is_prime_l248_248368


namespace remainder_div_x_minus_2_l248_248380

noncomputable def q (x : ℝ) (A B C : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 10

theorem remainder_div_x_minus_2 (A B C : ℝ) (h : q 2 A B C = 20) : q (-2) A B C = 20 :=
by sorry

end remainder_div_x_minus_2_l248_248380


namespace fraction_addition_l248_248446

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248446


namespace add_fractions_l248_248488

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248488


namespace gcd_ab_l248_248376

def a : ℕ := 130^2 + 215^2 + 310^2
def b : ℕ := 131^2 + 216^2 + 309^2

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l248_248376


namespace range_of_a_l248_248299

theorem range_of_a 
  (a : ℕ) 
  (an : ℕ → ℕ)
  (Sn : ℕ → ℕ)
  (h1 : a_1 = a)
  (h2 : ∀ n : ℕ, n ≥ 2 → Sn n + Sn (n - 1) = 4 * n^2)
  (h3 : ∀ n : ℕ, an n < an (n + 1)) : 
  3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l248_248299


namespace person_saves_2000_l248_248234

variable (income expenditure savings : ℕ)
variable (h_ratio : income / expenditure = 7 / 6)
variable (h_income : income = 14000)

theorem person_saves_2000 (h_ratio : income / expenditure = 7 / 6) (h_income : income = 14000) :
  savings = income - (6 * (14000 / 7)) :=
by
  sorry

end person_saves_2000_l248_248234


namespace fraction_addition_l248_248569

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248569


namespace total_people_at_evening_l248_248662

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l248_248662


namespace add_fractions_l248_248586

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248586


namespace siblings_of_John_l248_248090

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (height : String)

def John : Child := {name := "John", eyeColor := "Brown", hairColor := "Blonde", height := "Tall"}
def Emma : Child := {name := "Emma", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Oliver : Child := {name := "Oliver", eyeColor := "Brown", hairColor := "Black", height := "Short"}
def Mia : Child := {name := "Mia", eyeColor := "Blue", hairColor := "Blonde", height := "Short"}
def Lucas : Child := {name := "Lucas", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Sophia : Child := {name := "Sophia", eyeColor := "Blue", hairColor := "Blonde", height := "Tall"}

theorem siblings_of_John : 
  (John.hairColor = Mia.hairColor ∧ John.hairColor = Sophia.hairColor) ∧
  ((John.eyeColor = Mia.eyeColor ∨ John.eyeColor = Sophia.eyeColor) ∨
   (John.height = Mia.height ∨ John.height = Sophia.height)) ∧
  (Mia.eyeColor = Sophia.eyeColor ∨ Mia.hairColor = Sophia.hairColor ∨ Mia.height = Sophia.height) ∧
  (John.hairColor = "Blonde") ∧
  (John.height = "Tall") ∧
  (Mia.hairColor = "Blonde") ∧
  (Sophia.hairColor = "Blonde") ∧
  (Sophia.height = "Tall") 
  → True := sorry

end siblings_of_John_l248_248090


namespace parabola_focus_l248_248232

theorem parabola_focus (p : ℝ) (hp : p > 0) :
    ∀ (x y : ℝ), (x = 2 * p * y^2) ↔ (x, y) = (1 / (8 * p), 0) :=
by 
  sorry

end parabola_focus_l248_248232


namespace tractor_efficiency_l248_248227

theorem tractor_efficiency (x y : ℝ) (h1 : 18 / x = 24 / y) (h2 : x + y = 7) :
  x = 3 ∧ y = 4 :=
by {
  sorry
}

end tractor_efficiency_l248_248227


namespace area_of_intersection_of_two_circles_l248_248959

open Real

noncomputable def area_intersection (r : ℝ) (c1 c2 : ℝ × ℝ) : ℝ :=
  let quarter_circle_area := (1/4) * π * r^2
  let triangle_area := (1/2) * r^2
  let segment_area := quarter_circle_area - triangle_area
  2 * segment_area

theorem area_of_intersection_of_two_circles :
  area_intersection 3 (3, 0) (0, 3) = (9 * π - 18) / 2 :=
by
  -- This will be proven by the steps of the provided solution.
  sorry

end area_of_intersection_of_two_circles_l248_248959


namespace striped_to_total_ratio_l248_248785

theorem striped_to_total_ratio (total_students shorts_checkered_diff striped_shorts_diff : ℕ)
    (h_total : total_students = 81)
    (h_shorts_checkered : ∃ checkered, shorts_checkered_diff = checkered + 19)
    (h_striped_shorts : ∃ shorts, striped_shorts_diff = shorts + 8) :
    (striped_shorts_diff : ℚ) / total_students = 2 / 3 :=
by sorry

end striped_to_total_ratio_l248_248785


namespace missing_number_l248_248140

theorem missing_number (mean : ℝ) (numbers : List ℝ) (x : ℝ) (h_mean : mean = 14.2) (h_numbers : numbers = [13.0, 8.0, 13.0, 21.0, 23.0]) :
  (numbers.sum + x) / (numbers.length + 1) = mean → x = 7.2 :=
by
  -- states the hypothesis about the mean calculation into the theorem structure
  intro h
  sorry

end missing_number_l248_248140


namespace number_of_distinct_sentences_l248_248310

noncomputable def count_distinct_sentences (phrase : String) : Nat :=
  let I_options := 3 -- absent, partially present, fully present
  let II_options := 2 -- absent, present
  let IV_options := 2 -- incomplete or absent
  let III_mandatory := 1 -- always present
  (III_mandatory * IV_options * I_options * II_options) - 1 -- subtract the original sentence

theorem number_of_distinct_sentences :
  count_distinct_sentences "ранним утром на рыбалку улыбающийся Игорь мчался босиком" = 23 :=
by
  sorry

end number_of_distinct_sentences_l248_248310


namespace rectangle_ratio_l248_248685

theorem rectangle_ratio {l w : ℕ} (h_w : w = 5) (h_A : 50 = l * w) : l / w = 2 := by 
  sorry

end rectangle_ratio_l248_248685


namespace sum_fractions_eq_l248_248468

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248468


namespace add_fractions_l248_248525

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248525


namespace sam_gave_joan_seashells_l248_248643

variable (original_seashells : ℕ) (total_seashells : ℕ)

theorem sam_gave_joan_seashells (h1 : original_seashells = 70) (h2 : total_seashells = 97) :
  total_seashells - original_seashells = 27 :=
by
  sorry

end sam_gave_joan_seashells_l248_248643


namespace dmitriev_is_older_l248_248129

variables (Alekseev Borisov Vasilyev Grigoryev Dima Dmitriev : ℤ)

def Lesha := Alekseev + 1
def Borya := Borisov + 2
def Vasya := Vasilyev + 3
def Grisha := Grigoryev + 4

theorem dmitriev_is_older :
  Dima + 10 = Dmitriev :=
sorry

end dmitriev_is_older_l248_248129


namespace add_fractions_l248_248490

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248490


namespace number_of_shelves_l248_248973

def initial_bears : ℕ := 17
def shipment_bears : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 3 :=
by
  sorry

end number_of_shelves_l248_248973


namespace ratio_ba_in_range_l248_248038

theorem ratio_ba_in_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h1 : a + 2 * b = 7) (h2 : a^2 + b^2 ≤ 25) : 
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ 4 / 3 :=
by {
  sorry
}

end ratio_ba_in_range_l248_248038


namespace monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l248_248622

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l248_248622


namespace nancy_pics_uploaded_l248_248667

theorem nancy_pics_uploaded (a b n : ℕ) (h₁ : a = 11) (h₂ : b = 8) (h₃ : n = 5) : a + b * n = 51 := 
by 
  sorry

end nancy_pics_uploaded_l248_248667


namespace solve_for_x_l248_248251

theorem solve_for_x (x : ℤ) (h : (3012 + x)^2 = x^2) : x = -1506 := 
sorry

end solve_for_x_l248_248251


namespace students_without_favorite_subject_l248_248781

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end students_without_favorite_subject_l248_248781


namespace tiffany_bags_difference_l248_248693

theorem tiffany_bags_difference : 
  ∀ (monday_bags next_day_bags : ℕ), monday_bags = 7 → next_day_bags = 12 → next_day_bags - monday_bags = 5 := 
by
  intros monday_bags next_day_bags h1 h2
  sorry

end tiffany_bags_difference_l248_248693


namespace triangle_is_right_angled_l248_248757

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end triangle_is_right_angled_l248_248757


namespace point_to_real_l248_248597

-- Condition: Real numbers correspond one-to-one with points on the number line.
def real_numbers_correspond (x : ℝ) : Prop :=
  ∃ (p : ℝ), p = x

-- Condition: Any real number can be represented by a point on the number line.
def represent_real_by_point (x : ℝ) : Prop :=
  real_numbers_correspond x

-- Condition: Conversely, any point on the number line represents a real number.
def point_represents_real (p : ℝ) : Prop :=
  ∃ (x : ℝ), x = p

-- Condition: The number represented by any point on the number line is either a rational number or an irrational number.
def rational_or_irrational (p : ℝ) : Prop :=
  (∃ q : ℚ, (q : ℝ) = p) ∨ (¬∃ q : ℚ, (q : ℝ) = p)

theorem point_to_real (p : ℝ) : represent_real_by_point p ∧ point_represents_real p ∧ rational_or_irrational p → real_numbers_correspond p :=
by sorry

end point_to_real_l248_248597


namespace add_fractions_l248_248512

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248512


namespace students_without_favorite_subject_l248_248784

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end students_without_favorite_subject_l248_248784


namespace find_t_l248_248285

theorem find_t :
  ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 ∧ t = 44 :=
by
  sorry

end find_t_l248_248285


namespace sophomores_bought_15_more_markers_l248_248640

theorem sophomores_bought_15_more_markers (f_cost s_cost marker_cost : ℕ) (hf: f_cost = 267) (hs: s_cost = 312) (hm: marker_cost = 3) : 
  (s_cost / marker_cost) - (f_cost / marker_cost) = 15 :=
by
  sorry

end sophomores_bought_15_more_markers_l248_248640


namespace fraction_addition_l248_248538

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248538


namespace fraction_addition_l248_248547

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248547


namespace pancakes_needed_l248_248059

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end pancakes_needed_l248_248059


namespace final_customer_boxes_l248_248917

theorem final_customer_boxes (f1 f2 f3 f4 goal left boxes_first : ℕ) 
  (h1 : boxes_first = 5) 
  (h2 : f2 = 4 * boxes_first) 
  (h3 : f3 = f2 / 2) 
  (h4 : f4 = 3 * f3)
  (h5 : goal = 150) 
  (h6 : left = 75) 
  (h7 : goal - left = f1 + f2 + f3 + f4) : 
  (goal - left - (f1 + f2 + f3 + f4) = 10) := 
sorry

end final_customer_boxes_l248_248917


namespace simplify_expression_l248_248359

theorem simplify_expression : 1 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 1 - 2 * Real.sqrt 5 :=
by
  sorry

end simplify_expression_l248_248359


namespace relationship_between_abc_l248_248753

open Real

-- Define the constants for the problem
noncomputable def a : ℝ := sqrt 2023 - sqrt 2022
noncomputable def b : ℝ := sqrt 2022 - sqrt 2021
noncomputable def c : ℝ := sqrt 2021 - sqrt 2020

-- State the theorem we want to prove
theorem relationship_between_abc : c > b ∧ b > a := 
sorry

end relationship_between_abc_l248_248753


namespace financed_amount_correct_l248_248628

-- Define the conditions
def monthly_payment : ℝ := 150.0
def years : ℝ := 5.0
def months_in_a_year : ℝ := 12.0

-- Define the total number of months
def total_months : ℝ := years * months_in_a_year

-- Define the amount financed
def total_financed : ℝ := monthly_payment * total_months

-- State the theorem
theorem financed_amount_correct :
  total_financed = 9000 :=
by
  -- Provide the proof here
  sorry

end financed_amount_correct_l248_248628


namespace area_of_sector_equals_13_75_cm2_l248_248675

noncomputable def radius : ℝ := 5 -- radius in cm
noncomputable def arc_length : ℝ := 5.5 -- arc length in cm
noncomputable def circumference : ℝ := 2 * Real.pi * radius -- circumference of the circle
noncomputable def area_of_circle : ℝ := Real.pi * radius^2 -- area of the entire circle

theorem area_of_sector_equals_13_75_cm2 :
  (arc_length / circumference) * area_of_circle = 13.75 :=
by sorry

end area_of_sector_equals_13_75_cm2_l248_248675


namespace average_first_two_l248_248363

theorem average_first_two (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) = 16.8)
  (h2 : (c + d) = 4.6)
  (h3 : (e + f) = 7.4) : 
  (a + b) / 2 = 2.4 :=
by
  sorry

end average_first_two_l248_248363


namespace lines_intersect_value_k_l248_248079

theorem lines_intersect_value_k :
  ∀ (x y k : ℝ), (-3 * x + y = k) → (2 * x + y = 20) → (x = -10) → (k = 70) :=
by
  intros x y k h1 h2 h3
  sorry

end lines_intersect_value_k_l248_248079


namespace fraction_addition_l248_248556

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248556


namespace total_players_l248_248257

def kabaddi (K : ℕ) (Kho_only : ℕ) (Both : ℕ) : ℕ :=
  K - Both + Kho_only + Both

theorem total_players (K : ℕ) (Kho_only : ℕ) (Both : ℕ)
  (hK : K = 10)
  (hKho_only : Kho_only = 35)
  (hBoth : Both = 5) :
  kabaddi K Kho_only Both = 45 :=
by
  rw [hK, hKho_only, hBoth]
  unfold kabaddi
  norm_num

end total_players_l248_248257


namespace reciprocal_neg3_l248_248089

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l248_248089


namespace pipe_A_fill_time_l248_248397

theorem pipe_A_fill_time 
  (t : ℝ)
  (ht : (1 / t - 1 / 6) = 4 / 15.000000000000005) : 
  t = 30 / 13 :=  
sorry

end pipe_A_fill_time_l248_248397


namespace estimate_number_of_trees_l248_248210

-- Definitions derived from the conditions
def forest_length : ℝ := 100
def forest_width : ℝ := 0.5
def plot_length : ℝ := 1
def plot_width : ℝ := 0.5
def tree_counts : List ℕ := [65110, 63200, 64600, 64700, 67300, 63300, 65100, 66600, 62800, 65500]

-- The main theorem stating the problem
theorem estimate_number_of_trees :
  let avg_trees_per_plot := tree_counts.sum / tree_counts.length
  let total_plots := (forest_length * forest_width) / (plot_length * plot_width)
  avg_trees_per_plot * total_plots = 6482100 :=
by
  sorry

end estimate_number_of_trees_l248_248210


namespace mia_study_time_l248_248802

-- Let's define the conditions in Lean
def total_hours_in_day : ℕ := 24
def fraction_time_watching_TV : ℚ := 1 / 5
def fraction_time_studying : ℚ := 1 / 4

-- Time remaining after watching TV
def time_left_after_TV (total_hours : ℚ) (fraction_TV : ℚ) : ℚ :=
  total_hours * (1 - fraction_TV)

-- Time spent studying
def time_studying (remaining_time : ℚ) (fraction_studying : ℚ) : ℚ :=
  remaining_time * fraction_studying

-- Convert hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- The theorem statement
theorem mia_study_time :
  let total_hours := (total_hours_in_day : ℚ),
      time_after_TV := time_left_after_TV total_hours fraction_time_watching_TV,
      study_hours := time_studying time_after_TV fraction_time_studying,
      study_minutes := hours_to_minutes study_hours in
  study_minutes = 288 := by
  sorry

end mia_study_time_l248_248802


namespace angle_bisector_coordinates_distance_to_x_axis_l248_248754

structure Point where
  x : ℝ
  y : ℝ

def M (m : ℝ) : Point :=
  ⟨m - 1, 2 * m + 3⟩

theorem angle_bisector_coordinates (m : ℝ) :
  (M m = ⟨-5, -5⟩) ∨ (M m = ⟨-(5/3), 5/3⟩) := sorry

theorem distance_to_x_axis (m : ℝ) :
  (|2 * m + 3| = 1) → (M m = ⟨-2, 1⟩) ∨ (M m = ⟨-3, -1⟩) := sorry

end angle_bisector_coordinates_distance_to_x_axis_l248_248754


namespace math_problem_l248_248296

theorem math_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 * x + y - x * y = 0) : 
  ((9 * x + y) * (9 / y + 1 / x) = x * y) ∧ ¬ ((x / 9) + y = 10) ∧ 
  ((x + y = 16) ↔ (x = 4 ∧ y = 12)) ∧ 
  ((x * y = 36) ↔ (x = 2 ∧ y = 18)) :=
by {
  sorry
}

end math_problem_l248_248296


namespace circle_tangent_and_AB_distance_l248_248306

noncomputable section

-- Definitions based on the problem conditions
def parabola := {p : ℝ × ℝ | p.2 ^ 2 = 4 * (p.1 - 1)}

def vertex : ℝ × ℝ := (1, 0)

def center : ℝ × ℝ := (-1, 0)

def circle (p : ℝ × ℝ) := (p.1 + 1) ^ 2 + p.2 ^ 2 = 1

def tangent_points (p : ℝ × ℝ) := 
  let x0 := p.1
  let y0 := p.2 in
  y0 ^ 2 = 4 * (x0 - 1)

def distance_AB (x0 y0 : ℝ) : ℝ :=
  sqrt ((2 * y0 / (x0 + 2)) ^ 2 + 4 * (x0 / (x0 + 2)))

-- Theorem statement
theorem circle_tangent_and_AB_distance :
  (∀ p : ℝ × ℝ, p ∈ parabola → circle vertex ∧ 
    ∀ x0 y0 : ℝ, y0 ^ 2 = 4 * (x0 - 1) → 
      (distance_AB x0 y0) ∈ Set.Icc (2 * sqrt 3 / 3) (sqrt 39 / 3)) :=
sorry

end circle_tangent_and_AB_distance_l248_248306


namespace steps_left_to_climb_l248_248138

-- Define the conditions
def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

-- The problem: Prove that the number of stairs left to climb is 22
theorem steps_left_to_climb : (total_stairs - climbed_stairs) = 22 :=
by 
  sorry

end steps_left_to_climb_l248_248138


namespace evaluate_expression_l248_248277

theorem evaluate_expression :
  ((gcd 54 42 |> lcm 36) * (gcd 78 66 |> gcd 90) + (lcm 108 72 |> gcd 66 |> gcd 84)) = 24624 := by
  sorry

end evaluate_expression_l248_248277


namespace pencils_per_child_l248_248738

-- Define the conditions
def totalPencils : ℕ := 18
def numberOfChildren : ℕ := 9

-- The proof problem
theorem pencils_per_child : totalPencils / numberOfChildren = 2 := 
by
  sorry

end pencils_per_child_l248_248738


namespace chocolate_bars_percentage_l248_248219

noncomputable def total_chocolate_bars (milk dark almond white caramel : ℕ) : ℕ :=
  milk + dark + almond + white + caramel

noncomputable def percentage (count total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

theorem chocolate_bars_percentage :
  let milk := 36
  let dark := 21
  let almond := 40
  let white := 15
  let caramel := 28
  let total := total_chocolate_bars milk dark almond white caramel
  total = 140 ∧
  percentage milk total = 25.71 ∧
  percentage dark total = 15 ∧
  percentage almond total = 28.57 ∧
  percentage white total = 10.71 ∧
  percentage caramel total = 20 :=
by
  sorry

end chocolate_bars_percentage_l248_248219


namespace sum_of_fractions_l248_248424

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248424


namespace pancakes_needed_l248_248058

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end pancakes_needed_l248_248058


namespace right_triangle_power_inequality_l248_248052

theorem right_triangle_power_inequality {a b c x : ℝ} (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a^2 = b^2 + c^2) (h_longest : a > b ∧ a > c) :
  (x > 2) → (a^x > b^x + c^x) :=
by sorry

end right_triangle_power_inequality_l248_248052


namespace loss_percent_l248_248909

theorem loss_percent (C S : ℝ) (h : 100 * S = 40 * C) : ((C - S) / C) * 100 = 60 :=
by
  sorry

end loss_percent_l248_248909


namespace midpoint_distance_l248_248066

theorem midpoint_distance (a b c d : ℝ) :
  let m := (a + c) / 2
  let n := (b + d) / 2
  let m' := m - 0.5
  let n' := n - 0.5
  dist (m, n) (m', n') = (Real.sqrt 2) / 2 := 
by 
  sorry

end midpoint_distance_l248_248066


namespace reciprocal_of_neg3_l248_248084

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l248_248084


namespace relationship_between_abc_l248_248165

open Real

variables (a b c : ℝ)

def definition_a := a = 1 / 2023
def definition_b := b = tan (exp (1 / 2023) / 2023)
def definition_c := c = sin (exp (1 / 2024) / 2024)

theorem relationship_between_abc (h1 : definition_a a) (h2 : definition_b b) (h3 : definition_c c) : 
  c < a ∧ a < b :=
by
  sorry

end relationship_between_abc_l248_248165


namespace fraction_addition_l248_248551

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248551


namespace count_ordered_pairs_no_distinct_real_solutions_l248_248612

theorem count_ordered_pairs_no_distinct_real_solutions :
  {n : Nat // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (4 * b^2 - 4 * c ≤ 0) ∧ (4 * c^2 - 4 * b ≤ 0) ∧ n = 1} :=
sorry

end count_ordered_pairs_no_distinct_real_solutions_l248_248612


namespace circle_positional_relationship_l248_248908

noncomputable def r1 : ℝ := 2
noncomputable def r2 : ℝ := 3
noncomputable def d : ℝ := 5

theorem circle_positional_relationship :
  d = r1 + r2 → "externally tangent" = "externally tangent" := by
  intro h
  exact rfl

end circle_positional_relationship_l248_248908


namespace add_fractions_l248_248489

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248489


namespace Petya_wins_l248_248952

theorem Petya_wins (n : ℕ) (h₁ : n = 2016) : (∀ m : ℕ, m < n → ∀ k : ℕ, k ∣ m ∧ k ≠ m → m - k = 1 → false) :=
sorry

end Petya_wins_l248_248952


namespace overall_average_correct_l248_248394

noncomputable def overall_average : ℝ :=
  let students1 := 60
  let students2 := 35
  let students3 := 45
  let students4 := 42
  let avgMarks1 := 50
  let avgMarks2 := 60
  let avgMarks3 := 55
  let avgMarks4 := 45
  let total_students := students1 + students2 + students3 + students4
  let total_marks := (students1 * avgMarks1) + (students2 * avgMarks2) + (students3 * avgMarks3) + (students4 * avgMarks4)
  total_marks / total_students

theorem overall_average_correct : overall_average = 52.00 := by
  sorry

end overall_average_correct_l248_248394


namespace complement_intersect_eq_l248_248899

variable (U M N : Set ℕ)

-- The universal set
def U := {0, 1, 2, 3, 4}

-- Subsets M and N
def M := {0, 1, 2}
def N := {2, 3}

-- The main theorem to be proved
theorem complement_intersect_eq :
  (U \ M) ∩ N = {3} :=
by
  sorry

end complement_intersect_eq_l248_248899


namespace add_fractions_l248_248503

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248503


namespace min_value_of_expression_l248_248893

/-- Given the area of △ ABC is 2, and the sides opposite to angles A, B, C are a, b, c respectively,
    prove that the minimum value of a^2 + 2b^2 + 3c^2 is 8 * sqrt(11). -/
theorem min_value_of_expression
  (a b c : ℝ)
  (h₁ : 1/2 * b * c * Real.sin A = 2) :
  a^2 + 2 * b^2 + 3 * c^2 ≥ 8 * Real.sqrt 11 :=
sorry

end min_value_of_expression_l248_248893


namespace parallel_lines_m_eq_neg4_l248_248305

theorem parallel_lines_m_eq_neg4 (m : ℝ) (h1 : (m-2) ≠ -m) 
  (h2 : (m-2) / 3 = -m / (m + 2)) : m = -4 :=
sorry

end parallel_lines_m_eq_neg4_l248_248305


namespace rent_budget_l248_248861

variables (food_per_week : ℝ) (weekly_food_budget : ℝ) (video_streaming : ℝ)
          (cell_phone : ℝ) (savings : ℝ) (rent : ℝ)
          (total_spending : ℝ)

-- Conditions
def food_budget := food_per_week * 4 = weekly_food_budget
def video_streaming_budget := video_streaming = 30
def cell_phone_budget := cell_phone = 50
def savings_budget := savings = 0.1 * total_spending
def savings_amount := savings = 198

-- Prove
theorem rent_budget (h1 : food_budget food_per_week weekly_food_budget)
                    (h2 : video_streaming_budget video_streaming)
                    (h3 : cell_phone_budget cell_phone)
                    (h4 : savings_budget savings total_spending)
                    (h5 : savings_amount savings) :
  rent = 1500 :=
sorry

end rent_budget_l248_248861


namespace cost_of_rusted_side_l248_248986

-- Define the conditions
def perimeter (s : ℕ) (l : ℕ) : ℕ :=
  2 * s + 2 * l

def long_side (s : ℕ) : ℕ :=
  3 * s

def cost_per_foot : ℕ :=
  5

-- Given these conditions, we prove the cost of replacing one short side.
theorem cost_of_rusted_side (s l : ℕ) (h1 : perimeter s l = 640) (h2 : l = long_side s) : 
  5 * s = 400 :=
by 
  sorry

end cost_of_rusted_side_l248_248986


namespace minimum_value_l248_248891

theorem minimum_value (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ z, z = 9 ∧ (forall x y, x > 0 ∧ y > 0 ∧ x + y = 1 → (1/x + 4/y) ≥ z) := 
sorry

end minimum_value_l248_248891


namespace percent_area_shaded_l248_248696

-- Conditions: Square $ABCD$ has a side length of 10, and square $PQRS$ has a side length of 15.
-- The overlap of these squares forms a rectangle $AQRD$ with dimensions $20 \times 25$.

theorem percent_area_shaded 
  (side_ABCD : ℕ := 10) 
  (side_PQRS : ℕ := 15) 
  (dim_AQRD_length : ℕ := 25) 
  (dim_AQRD_width : ℕ := 20) 
  (area_AQRD : ℕ := dim_AQRD_length * dim_AQRD_width)
  (overlap_side : ℕ := 10) 
  (area_shaded : ℕ := overlap_side * overlap_side)
  : (area_shaded * 100) / area_AQRD = 20 := 
by 
  sorry

end percent_area_shaded_l248_248696


namespace max_lattice_points_in_unit_circle_l248_248699

-- Define a point with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the condition for a lattice point to be strictly inside a given circle
def strictly_inside_circle (p : LatticePoint) (center : Prod ℤ ℤ) (r : ℝ) : Prop :=
  let dx := (p.x - center.fst : ℝ)
  let dy := (p.y - center.snd : ℝ)
  dx^2 + dy^2 < r^2

-- Define the problem statement
theorem max_lattice_points_in_unit_circle : ∀ (center : Prod ℤ ℤ) (r : ℝ),
  r = 1 → 
  ∃ (ps : Finset LatticePoint), 
    (∀ p ∈ ps, strictly_inside_circle p center r) ∧ 
    ps.card = 4 :=
by
  sorry

end max_lattice_points_in_unit_circle_l248_248699


namespace average_tickets_sold_by_female_l248_248712

-- Define the conditions as Lean expressions.

def totalMembers (M : ℕ) : ℕ := M + 2 * M
def totalTickets (F : ℕ) (M : ℕ) : ℕ := 58 * M + F * 2 * M
def averageTicketsPerMember (F : ℕ) (M : ℕ) : ℕ := (totalTickets F M) / (totalMembers M)

theorem average_tickets_sold_by_female (F M : ℕ) 
  (h1 : 66 * (totalMembers M) = totalTickets F M) :
  F = 70 :=
by
  sorry

end average_tickets_sold_by_female_l248_248712


namespace total_number_of_people_l248_248388

-- Definitions corresponding to conditions
variables (A C : ℕ)
variables (cost_adult cost_child total_revenue : ℝ)
variables (ratio_child_adult : ℝ)

-- Assumptions given in the problem
axiom cost_adult_def : cost_adult = 7
axiom cost_child_def : cost_child = 3
axiom total_revenue_def : total_revenue = 6000
axiom ratio_def : C = 3 * A
axiom revenue_eq : total_revenue = cost_adult * A + cost_child * C

-- The main statement to prove
theorem total_number_of_people : A + C = 1500 :=
by
  sorry  -- Proof of the theorem

end total_number_of_people_l248_248388


namespace add_fractions_l248_248523

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248523


namespace horses_legs_problem_l248_248978

theorem horses_legs_problem 
    (m h a b : ℕ) 
    (h_eq : h = m) 
    (men_to_A : m = 3 * a) 
    (men_to_B : m = 4 * b) 
    (total_legs : 2 * m + 4 * (h / 2) + 3 * a + 4 * b = 200) : 
    h = 25 :=
  sorry

end horses_legs_problem_l248_248978


namespace add_fractions_l248_248577

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248577


namespace difference_in_soda_bottles_l248_248391

-- Define the given conditions
def regular_soda_bottles : ℕ := 81
def diet_soda_bottles : ℕ := 60

-- Define the difference in the number of bottles
def difference_bottles : ℕ := regular_soda_bottles - diet_soda_bottles

-- The theorem we want to prove
theorem difference_in_soda_bottles : difference_bottles = 21 := by
  sorry

end difference_in_soda_bottles_l248_248391


namespace maximum_bottles_l248_248737

-- Definitions for the number of bottles each shop sells
def bottles_from_shop_A : ℕ := 150
def bottles_from_shop_B : ℕ := 180
def bottles_from_shop_C : ℕ := 220

-- The main statement to prove
theorem maximum_bottles : bottles_from_shop_A + bottles_from_shop_B + bottles_from_shop_C = 550 := 
by 
  sorry

end maximum_bottles_l248_248737


namespace cos_diff_alpha_beta_l248_248890

theorem cos_diff_alpha_beta (α β : ℝ) (h1 : Real.sin α = 2 / 3) (h2 : Real.cos β = -3 / 4)
    (h3 : α ∈ Set.Ioo (π / 2) π) (h4 : β ∈ Set.Ioo π (3 * π / 2)) :
    Real.cos (α - β) = (3 * Real.sqrt 5 - 2 * Real.sqrt 7) / 12 := 
sorry

end cos_diff_alpha_beta_l248_248890


namespace dilation_image_l248_248073

open Complex

theorem dilation_image (z₀ : ℂ) (c : ℂ) (k : ℝ) (z : ℂ)
    (h₀ : z₀ = 0 - 2*I) (h₁ : c = 1 + 2*I) (h₂ : k = 2) :
    z = -1 - 6*I :=
by
  sorry

end dilation_image_l248_248073


namespace monotone_decreasing_intervals_l248_248022

theorem monotone_decreasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, deriv f x = (x - 2) * (x^2 - 1)) :
  ((∀ x : ℝ, x < -1 → deriv f x < 0) ∧ (∀ x : ℝ, 1 < x → x < 2 → deriv f x < 0)) :=
by
  sorry

end monotone_decreasing_intervals_l248_248022


namespace sum_of_roots_eq_9_div_4_l248_248280

-- Define the values for the coefficients
def a : ℝ := -48
def b : ℝ := 108
def c : ℝ := -27

-- Define the quadratic equation and the function that represents the sum of the roots
def quadratic_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement of the problem: Prove the sum of the roots of the quadratic equation equals 9/4
theorem sum_of_roots_eq_9_div_4 : 
  (∀ x y : ℝ, quadratic_eq x = 0 → quadratic_eq y = 0 → x ≠ y → x + y = - (b/a)) → - (b / a) = 9 / 4 :=
by
  sorry

end sum_of_roots_eq_9_div_4_l248_248280


namespace compute_expression_l248_248734

theorem compute_expression : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end compute_expression_l248_248734


namespace add_fractions_result_l248_248411

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248411


namespace fraction_addition_l248_248455

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248455


namespace money_left_after_shopping_l248_248938

def initial_money : ℕ := 158
def shoe_cost : ℕ := 45
def bag_cost := shoe_cost - 17
def lunch_cost := bag_cost / 4
def total_expenses := shoe_cost + bag_cost + lunch_cost
def remaining_money := initial_money - total_expenses

theorem money_left_after_shopping : remaining_money = 78 := by
  sorry

end money_left_after_shopping_l248_248938


namespace gcd_840_1764_l248_248683

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by 
  sorry

end gcd_840_1764_l248_248683


namespace smallest_value_of_a_for_polynomial_l248_248237

theorem smallest_value_of_a_for_polynomial (r1 r2 r3 : ℕ) (h_prod : r1 * r2 * r3 = 30030) :
  (r1 + r2 + r3 = 54) ∧ (r1 * r2 * r3 = 30030) → 
  (∀ a, a = r1 + r2 + r3 → a ≥ 54) :=
by
  sorry

end smallest_value_of_a_for_polynomial_l248_248237


namespace angle_QPS_l248_248373

-- Definitions of the points and angles
variables (P Q R S : Point)
variables (angle : Point → Point → Point → ℝ)

-- Conditions about the isosceles triangles and angles
variables (isosceles_PQR : PQ = QR)
variables (isosceles_PRS : PR = RS)
variables (R_inside_PQS : ¬(R ∈ convex_hull ℝ {P, Q, S}))
variables (angle_PQR : angle P Q R = 50)
variables (angle_PRS : angle P R S = 120)

-- The theorem we want to prove
theorem angle_QPS : angle Q P S = 35 :=
sorry -- Proof goes here

end angle_QPS_l248_248373


namespace solve_for_C_days_l248_248974

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 15
noncomputable def C_work_rate : ℚ := 1 / 50
noncomputable def total_work_done_by_A_B : ℚ := 6 * (A_work_rate + B_work_rate)
noncomputable def remaining_work : ℚ := 1 - total_work_done_by_A_B

theorem solve_for_C_days : ∃ d : ℚ, d * C_work_rate = remaining_work ∧ d = 15 :=
by
  use 15
  simp [C_work_rate, remaining_work, total_work_done_by_A_B, A_work_rate, B_work_rate]
  sorry

end solve_for_C_days_l248_248974


namespace find_f_neg_two_l248_248615

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (h1 : ∀ a b : ℝ, f (a + b) = f a * f b)
variable (h2 : ∀ x : ℝ, f x > 0)
variable (h3 : f 1 = 1 / 2)

-- State the theorem to prove that f(-2) = 4
theorem find_f_neg_two : f (-2) = 4 :=
by
  sorry

end find_f_neg_two_l248_248615


namespace prove_travel_cost_l248_248668

noncomputable def least_expensive_travel_cost
  (a_cost_per_km : ℝ) (a_booking_fee : ℝ) (b_cost_per_km : ℝ)
  (DE DF EF : ℝ) :
  ℝ := by
  let a_cost_DE := DE * a_cost_per_km + a_booking_fee
  let b_cost_DE := DE * b_cost_per_km
  let cheaper_cost_DE := min a_cost_DE b_cost_DE

  let a_cost_EF := EF * a_cost_per_km + a_booking_fee
  let b_cost_EF := EF * b_cost_per_km
  let cheaper_cost_EF := min a_cost_EF b_cost_EF

  let a_cost_DF := DF * a_cost_per_km + a_booking_fee
  let b_cost_DF := DF * b_cost_per_km
  let cheaper_cost_DF := min a_cost_DF b_cost_DF

  exact cheaper_cost_DE + cheaper_cost_EF + cheaper_cost_DF

def travel_problem : Prop :=
  let DE := 5000
  let DF := 4000
  let EF := 2500 -- derived from the Pythagorean theorem
  least_expensive_travel_cost 0.12 120 0.20 DE DF EF = 1740

theorem prove_travel_cost : travel_problem := sorry

end prove_travel_cost_l248_248668


namespace odds_against_C_l248_248901

theorem odds_against_C (pA pB : ℚ) (hA : pA = 1 / 5) (hB : pB = 2 / 3) :
  (1 - (1 - pA + 1 - pB)) / (1 - pA - pB) = 13 / 2 := 
sorry

end odds_against_C_l248_248901


namespace fraction_addition_l248_248572

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248572


namespace radius_range_of_circle_l248_248187

theorem radius_range_of_circle (r : ℝ) :
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
  (abs (4*x - 3*y - 2) = 1)) →
  4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l248_248187


namespace julia_total_food_cost_l248_248330

-- Definitions based on conditions
def weekly_total_cost : ℕ := 30
def rabbit_weeks : ℕ := 5
def rabbit_food_cost : ℕ := 12
def parrot_weeks : ℕ := 3
def parrot_food_cost : ℕ := weekly_total_cost - rabbit_food_cost

-- Proof statement
theorem julia_total_food_cost : 
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost = 114 := 
by 
  sorry

end julia_total_food_cost_l248_248330


namespace sin_of_tan_l248_248635

theorem sin_of_tan (A : ℝ) (hA_acute : 0 < A ∧ A < π / 2) (h_tan_A : Real.tan A = (Real.sqrt 2) / 3) :
  Real.sin A = (Real.sqrt 22) / 11 :=
sorry

end sin_of_tan_l248_248635


namespace fraction_addition_l248_248559

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248559


namespace sum_first_and_third_angle_l248_248240

-- Define the conditions
variable (A : ℕ)
axiom C1 : A + 2 * A + (A - 40) = 180

-- State the theorem to be proven
theorem sum_first_and_third_angle : A + (A - 40) = 70 :=
by
  sorry

end sum_first_and_third_angle_l248_248240


namespace profit_difference_is_50_l248_248714

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end profit_difference_is_50_l248_248714


namespace fraction_addition_l248_248558

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248558


namespace total_people_at_beach_l248_248659

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l248_248659


namespace inappropriate_survey_method_l248_248704

def survey_method_appropriate (method : String) : Bool :=
  method = "sampling" -- only sampling is considered appropriate in this toy model

def survey_approps : Bool :=
  let A := survey_method_appropriate "sampling"
  let B := survey_method_appropriate "sampling"
  let C := ¬ survey_method_appropriate "census"
  let D := survey_method_appropriate "census"
  C

theorem inappropriate_survey_method :
  survey_approps = true :=
by
  sorry

end inappropriate_survey_method_l248_248704


namespace fraction_addition_l248_248554

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248554


namespace add_fractions_l248_248518

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248518


namespace jump_difference_l248_248669

variable (runningRicciana jumpRicciana runningMargarita : ℕ)

theorem jump_difference :
  (runningMargarita + (2 * jumpRicciana - 1)) - (runningRicciana + jumpRicciana) = 1 :=
by
  -- Given conditions
  let runningRicciana := 20
  let jumpRicciana := 4
  let runningMargarita := 18
  -- The proof is omitted (using 'sorry')
  sorry

end jump_difference_l248_248669


namespace volume_of_pyramid_correct_l248_248078

noncomputable def volume_of_pyramid (lateral_surface_area base_area inscribed_circle_area radius : ℝ) : ℝ :=
  if lateral_surface_area = 3 * base_area ∧ inscribed_circle_area = radius then
    (2 * Real.sqrt 6) / (Real.pi ^ 3)
  else
    0

theorem volume_of_pyramid_correct
  (lateral_surface_area base_area inscribed_circle_area radius : ℝ)
  (h1 : lateral_surface_area = 3 * base_area)
  (h2 : inscribed_circle_area = radius) :
  volume_of_pyramid lateral_surface_area base_area inscribed_circle_area radius = (2 * Real.sqrt 6) / (Real.pi ^ 3) :=
by {
  sorry
}

end volume_of_pyramid_correct_l248_248078


namespace remaining_money_correct_l248_248935

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

end remaining_money_correct_l248_248935


namespace circle_intersection_l248_248041

theorem circle_intersection (a : ℝ) :
  ((-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2)) ↔
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 1) :=
sorry

end circle_intersection_l248_248041


namespace find_prime_p_l248_248870

theorem find_prime_p
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h : Nat.Prime (p^3 + p^2 + 11 * p + 2)) :
  p = 3 :=
sorry

end find_prime_p_l248_248870


namespace sum_fractions_eq_l248_248472

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248472


namespace least_value_l248_248698

theorem least_value : ∀ x y : ℝ, (xy + 1)^2 + (x - y)^2 ≥ 1 :=
by
  sorry

end least_value_l248_248698


namespace add_fractions_l248_248484

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248484


namespace initial_walking_speed_l248_248848

open Real

theorem initial_walking_speed :
  ∃ (v : ℝ), (∀ (d : ℝ), d = 9.999999999999998 →
  (∀ (lateness_time : ℝ), lateness_time = 10 / 60 →
  ((d / v) - (d / 15) = lateness_time + lateness_time)) → v = 11.25) :=
by
  sorry

end initial_walking_speed_l248_248848


namespace player_A_prize_received_event_A_not_low_probability_l248_248374

-- Condition Definitions
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3
def a : ℚ := 243

-- Part 1: Player A's Prize
theorem player_A_prize_received :
  (a * (p * p + 3 * p * (1 - p) * p + 3 * (1 - p) * p * p + (1 - p) * (1 - p) * p * p)) = 216 := sorry

-- Part 2: Probability of Event A with Low Probability Conditions
def low_probability_event (prob : ℚ) : Prop := prob < 0.05

-- Probability that player B wins the entire prize
def event_A_probability (p : ℚ) : ℚ :=
  (1 - p) ^ 3 + 3 * p * (1 - p) ^ 3

theorem event_A_not_low_probability (p : ℚ) (hp : p ≥ 3 / 4) :
  ¬ low_probability_event (event_A_probability p) := sorry

end player_A_prize_received_event_A_not_low_probability_l248_248374


namespace quilt_cost_calculation_l248_248644

theorem quilt_cost_calculation :
  let length := 12
  let width := 15
  let cost_per_sq_foot := 70
  let sales_tax_rate := 0.05
  let discount_rate := 0.10
  let area := length * width
  let cost_before_discount := area * cost_per_sq_foot
  let discount_amount := cost_before_discount * discount_rate
  let cost_after_discount := cost_before_discount - discount_amount
  let sales_tax_amount := cost_after_discount * sales_tax_rate
  let total_cost := cost_after_discount + sales_tax_amount
  total_cost = 11907 := by
  {
    sorry
  }

end quilt_cost_calculation_l248_248644


namespace exp_mono_increasing_l248_248855

theorem exp_mono_increasing (x y : ℝ) (h : x ≤ y) : (2:ℝ)^x ≤ (2:ℝ)^y :=
sorry

end exp_mono_increasing_l248_248855


namespace solve_inequality_l248_248228

theorem solve_inequality :
  {x : ℝ | (x - 1) * (2 * x + 1) ≤ 0} = { x : ℝ | -1/2 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l248_248228


namespace biology_exam_students_l248_248805

theorem biology_exam_students :
  let students := 200
  let score_A := (1 / 4) * students
  let remaining_students := students - score_A
  let score_B := (1 / 5) * remaining_students
  let score_C := (1 / 3) * remaining_students
  let score_D := (5 / 12) * remaining_students
  let score_F := students - (score_A + score_B + score_C + score_D)
  let re_assessed_C := (3 / 5) * score_C
  let final_score_B := score_B + re_assessed_C
  let final_score_C := score_C - re_assessed_C
  score_A = 50 ∧ 
  final_score_B = 60 ∧ 
  final_score_C = 20 ∧ 
  score_D = 62 ∧ 
  score_F = 8 :=
by {
  sorry
}

end biology_exam_students_l248_248805


namespace solve_for_x_l248_248813

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 4) * x - 3 = 5 → x = 112 := by
  sorry

end solve_for_x_l248_248813


namespace solve_for_a_l248_248629

theorem solve_for_a (a x : ℝ) (h : x = 3) (eqn : a * x - 5 = x + 1) : a = 3 :=
by
  -- proof omitted
  sorry

end solve_for_a_l248_248629


namespace factorize_x2_minus_9_l248_248602

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end factorize_x2_minus_9_l248_248602


namespace parallelepiped_length_l248_248994

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248994


namespace parabola_tangent_sum_l248_248370

theorem parabola_tangent_sum (m n : ℕ) (hmn_coprime : Nat.gcd m n = 1)
    (h_tangent : ∃ (k : ℝ), ∀ (x y : ℝ), y = 4 * x^2 ↔ x = y^2 + (m / n)) :
    m + n = 19 :=
by
  sorry

end parabola_tangent_sum_l248_248370


namespace remaining_money_correct_l248_248934

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

end remaining_money_correct_l248_248934


namespace fraction_addition_l248_248531

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248531


namespace add_fractions_l248_248511

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248511


namespace max_profit_at_9_l248_248390

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 10.8 - (1 / 30) * x^2
else if h : x > 10 then 108 / x - 1000 / (3 * x^2)
else 0

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
else if h : x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
else 0

theorem max_profit_at_9 : W 9 = 38.6 :=
sorry

end max_profit_at_9_l248_248390


namespace problem_statement_l248_248875

theorem problem_statement : (1021 ^ 1022) % 1023 = 4 := 
by
  sorry

end problem_statement_l248_248875


namespace sum_of_fractions_l248_248425

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248425


namespace min_value_of_c_l248_248301

theorem min_value_of_c (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_ineq1 : a < b) 
  (h_ineq2 : b < 2 * b) 
  (h_ineq3 : 2 * b < c)
  (h_unique_sol : ∃ x : ℝ, 3 * x + (|x - a| + |x - b| + |x - (2 * b)| + |x - c|) = 3000) :
  c = 502 := sorry

end min_value_of_c_l248_248301


namespace simplify_polynomial_l248_248671

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end simplify_polynomial_l248_248671


namespace add_fractions_l248_248514

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248514


namespace add_fractions_l248_248580

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248580


namespace fraction_addition_l248_248461

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248461


namespace find_a_plus_b_l248_248284

theorem find_a_plus_b (a b : ℝ) : (3 = 1/3 * 1 + a) → (1 = 1/3 * 3 + b) → a + b = 8/3 :=
by
  intros h1 h2
  sorry

end find_a_plus_b_l248_248284


namespace shelves_of_picture_books_l248_248900

theorem shelves_of_picture_books
   (total_books : ℕ)
   (books_per_shelf : ℕ)
   (mystery_shelves : ℕ)
   (mystery_books : ℕ)
   (total_mystery_books : mystery_books = mystery_shelves * books_per_shelf)
   (total_books_condition : total_books = 32)
   (mystery_books_condition : mystery_books = 5 * books_per_shelf) :
   (total_books - mystery_books) / books_per_shelf = 3 :=
by
  sorry

end shelves_of_picture_books_l248_248900


namespace add_fractions_l248_248499

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248499


namespace fraction_addition_l248_248553

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248553


namespace min_fence_dimensions_l248_248799

theorem min_fence_dimensions (A : ℝ) (hA : A ≥ 800) (x : ℝ) (hx : 2 * x * x = A) : x = 20 ∧ 2 * x = 40 := by
  sorry

end min_fence_dimensions_l248_248799


namespace problem_result_l248_248102

open Matrix FiniteDimensional

-- Definitions of the initial vectors u₀ and z₀
def u₀ : ℝ^2 := ![2, 1]
def z₀ : ℝ^2 := ![3, 2]

-- Projections in general form for Lean
def proj (a b : ℝ^2) : ℝ^2 := (dot_product a b / dot_product a a) • a

-- Definitions for the vector sequences
def u (n : ℕ) : ℝ^2 :=
  if n = 0 then u₀ else proj u₀ (z (n - 1))

def z (n : ℕ) : ℝ^2 :=
  if n = 0 then z₀ else proj z₀ (u n)

-- The sum of all vectors in the sequence
noncomputable def sequence_sum : ℝ^2 := 
  ∑' n : ℕ, u (n + 1) + z (n + 1)

-- The theorem to prove the desired sum
theorem problem_result :
  sequence_sum = ![520, 312] :=
sorry

end problem_result_l248_248102


namespace Jorge_age_in_2005_l248_248645

theorem Jorge_age_in_2005
  (age_Simon_2010 : ℕ)
  (age_difference : ℕ)
  (age_of_Simon_2010 : age_Simon_2010 = 45)
  (age_difference_Simon_Jorge : age_difference = 24)
  (age_Simon_2005 : ℕ := age_Simon_2010 - 5)
  (age_Jorge_2005 : ℕ := age_Simon_2005 - age_difference) :
  age_Jorge_2005 = 16 := by
  sorry

end Jorge_age_in_2005_l248_248645


namespace intersection_A_complement_B_range_of_a_l248_248162

-- Define sets A and B with their respective conditions
def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Question 1: Prove the intersection when a = 2
theorem intersection_A_complement_B (a : ℝ) (h : a = 2) : 
  A a ∩ (U \ B a) = {x | 2 < x ∧ x ≤ 4} ∪ {x | 5 ≤ x ∧ x < 7} :=
by sorry

-- Question 2: Find the range of a such that A ∪ B = A given a ≠ 1
theorem range_of_a (a : ℝ) (h : a ≠ 1) : 
  (A a ∪ B a = A a) ↔ (1 < a ∧ a ≤ 3 ∨ a = -1) :=
by sorry

end intersection_A_complement_B_range_of_a_l248_248162


namespace num_even_multiple_5_perfect_squares_lt_1000_l248_248197

theorem num_even_multiple_5_perfect_squares_lt_1000 : 
  ∃ n, n = 3 ∧ ∀ x, (x < 1000) ∧ (x > 0) ∧ (∃ k, x = 100 * k^2) → (n = 3) := by 
  sorry

end num_even_multiple_5_perfect_squares_lt_1000_l248_248197


namespace probability_of_sum_six_two_dice_l248_248108

noncomputable def probability_sum_six : ℚ := 5 / 36

theorem probability_of_sum_six_two_dice (dice_faces : ℕ := 6) : 
  ∃ (p : ℚ), p = probability_sum_six :=
by
  sorry

end probability_of_sum_six_two_dice_l248_248108


namespace add_fractions_l248_248500

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248500


namespace sum_of_fractions_l248_248426

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248426


namespace MrMartinSpent_l248_248221

theorem MrMartinSpent : 
  ∀ (C B : ℝ), 
    3 * C + 2 * B = 12.75 → 
    B = 1.5 → 
    2 * C + 5 * B = 14 := 
by
  intros C B h1 h2
  sorry

end MrMartinSpent_l248_248221


namespace school_club_profit_l248_248850

-- Definition of the problem conditions
def candy_bars_bought : ℕ := 800
def cost_per_four_bars : ℚ := 3
def bars_per_four_bars : ℕ := 4
def sell_price_per_three_bars : ℚ := 2
def bars_per_three_bars : ℕ := 3
def sales_fee_per_bar : ℚ := 0.05

-- Definition for cost calculations
def cost_per_bar : ℚ := cost_per_four_bars / bars_per_four_bars
def total_cost : ℚ := candy_bars_bought * cost_per_bar

-- Definition for revenue calculations
def sell_price_per_bar : ℚ := sell_price_per_three_bars / bars_per_three_bars
def total_revenue : ℚ := candy_bars_bought * sell_price_per_bar

-- Definition for total sales fee
def total_sales_fee : ℚ := candy_bars_bought * sales_fee_per_bar

-- Definition of profit
def profit : ℚ := total_revenue - total_cost - total_sales_fee

-- The statement to be proved
theorem school_club_profit : profit = -106.64 := by sorry

end school_club_profit_l248_248850


namespace percent_commute_l248_248775

variable (x : ℝ)

theorem percent_commute (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_commute_l248_248775


namespace find_S2012_l248_248884

section Problem

variable {a : ℕ → ℝ} -- Defining the sequence

-- Conditions
def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (Finset.range n).sum a

axiom a1 : a 1 = 2011
axiom recurrence_relation (n : ℕ) : a n + 2*a (n + 1) + a (n + 2) = 0

-- Proof statement
theorem find_S2012 (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ):
  geometric_sequence a →
  (∀ n, S n = sum_S a n) →
  S 2012 = 0 :=
by
  sorry

end Problem

end find_S2012_l248_248884


namespace tiling_implies_divisibility_l248_248798

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end tiling_implies_divisibility_l248_248798


namespace sum_of_fractions_l248_248435

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248435


namespace loss_percentage_l248_248260

variable (CP SP : ℕ) -- declare the variables for cost price and selling price

theorem loss_percentage (hCP : CP = 1400) (hSP : SP = 1190) : 
  ((CP - SP) / CP * 100) = 15 := by
sorry

end loss_percentage_l248_248260


namespace sticks_form_equilateral_triangle_l248_248151

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l248_248151


namespace add_fractions_l248_248574

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248574


namespace linear_system_incorrect_statement_l248_248068

def is_determinant (a b c d : ℝ) := a * d - b * c

def is_solution_system (a1 b1 c1 a2 b2 c2 D Dx Dy : ℝ) :=
  D = is_determinant a1 b1 a2 b2 ∧
  Dx = is_determinant c1 b1 c2 b2 ∧
  Dy = is_determinant a1 c1 a2 c2

def is_solution_linear_system (a1 b1 c1 a2 b2 c2 x y : ℝ) :=
  a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

theorem linear_system_incorrect_statement :
  ∀ (x y : ℝ),
    is_solution_system 3 (-1) 1 1 3 7 10 10 20 ∧
    is_solution_linear_system 3 (-1) 1 1 3 7 x y →
    x = 1 ∧ y = 2 ∧ ¬(20 = -20) := 
by sorry

end linear_system_incorrect_statement_l248_248068


namespace R_and_D_expenditure_l248_248269

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end R_and_D_expenditure_l248_248269


namespace abs_frac_sqrt_l248_248904

theorem abs_frac_sqrt (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 + b^2 = 9 * a * b) : 
  abs ((a + b) / (a - b)) = Real.sqrt (11 / 7) :=
by
  sorry

end abs_frac_sqrt_l248_248904


namespace grogg_expected_value_l248_248195

theorem grogg_expected_value (n : ℕ) (p : ℝ) (h_n : 2 ≤ n) (h_p : 0 < p ∧ p < 1) :
  (p + n * p^n * (1 - p) = 1) ↔ (p = 1 / n^(1/n:ℝ)) :=
sorry

end grogg_expected_value_l248_248195


namespace mary_chopped_tables_l248_248927

-- Define the constants based on the conditions
def chairs_sticks := 6
def tables_sticks := 9
def stools_sticks := 2
def burn_rate := 5

-- Define the quantities of items Mary chopped up
def chopped_chairs := 18
def chopped_stools := 4
def warm_hours := 34
def sticks_from_chairs := chopped_chairs * chairs_sticks
def sticks_from_stools := chopped_stools * stools_sticks
def total_needed_sticks := warm_hours * burn_rate
def sticks_from_tables (chopped_tables : ℕ) := chopped_tables * tables_sticks

-- Define the proof goal
theorem mary_chopped_tables : ∃ chopped_tables, sticks_from_chairs + sticks_from_stools + sticks_from_tables chopped_tables = total_needed_sticks ∧ chopped_tables = 6 :=
by
  sorry

end mary_chopped_tables_l248_248927


namespace number_of_people_l248_248246

theorem number_of_people (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 2 * x + y / 2 + z / 4 = 12) : 
  x = 5 ∧ y = 1 ∧ z = 6 := 
by
  sorry

end number_of_people_l248_248246


namespace fraction_addition_l248_248557

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248557


namespace range_of_x_l248_248015

def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hpq : p x ∨ q x) (hnq : ¬ q x) : x ≤ 0 ∨ x ≥ 4 :=
by sorry

end range_of_x_l248_248015


namespace unique_scalar_matrix_l248_248744

theorem unique_scalar_matrix (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, Matrix.mulVec N v = 5 • v) → 
  N = !![5, 0, 0; 0, 5, 0; 0, 0, 5] :=
by
  intro hv
  sorry -- Proof omitted as per instructions

end unique_scalar_matrix_l248_248744


namespace max_speed_of_cart_l248_248724

theorem max_speed_of_cart (R a : ℝ) (hR : R > 0) (ha : a > 0) :
  ∃ v_max : ℝ, v_max = sqrt (sqrt ((16 * a^2 * R^2 * Real.pi^2) / (1 + 16 * Real.pi^2))) :=
by
  sorry

end max_speed_of_cart_l248_248724


namespace max_last_digit_of_sequence_l248_248823

theorem max_last_digit_of_sequence :
  ∀ (s : Fin 1001 → ℕ), 
  (s 0 = 2) →
  (∀ (i : Fin 1000), (s i) * 10 + (s i.succ) ∈ {n | n % 17 = 0 ∨ n % 23 = 0}) →
  ∃ (d : ℕ), (d = s ⟨1000, sorry⟩) ∧ (∀ (d' : ℕ), d' = s ⟨1000, sorry⟩ → d' ≤ d) ∧ (d = 2) :=
by
  intros s h1 h2
  use 2
  sorry

end max_last_digit_of_sequence_l248_248823


namespace typing_page_percentage_l248_248853

/--
Given:
- Original sheet dimensions are 20 cm by 30 cm.
- Margins are 2 cm on each side (left and right), and 3 cm on the top and bottom.
Prove that the percentage of the page used by the typist is 64%.
-/
theorem typing_page_percentage (width height margin_lr margin_tb : ℝ)
  (h1 : width = 20) 
  (h2 : height = 30) 
  (h3 : margin_lr = 2) 
  (h4 : margin_tb = 3) : 
  (width - 2 * margin_lr) * (height - 2 * margin_tb) / (width * height) * 100 = 64 :=
by
  sorry

end typing_page_percentage_l248_248853


namespace total_people_at_evening_l248_248663

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l248_248663


namespace profit_ratio_l248_248801

variables (P_s : ℝ)

theorem profit_ratio (h1 : 21 * (7 / 3) + 3 * P_s = 175) : P_s / 21 = 2 :=
by
  sorry

end profit_ratio_l248_248801


namespace odd_function_property_l248_248188

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / x

theorem odd_function_property (a : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2) (h_fa : f a = -4) : f (-a) = 4 :=
by
  sorry

end odd_function_property_l248_248188


namespace fraction_addition_l248_248532

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248532


namespace people_at_the_beach_l248_248666

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l248_248666


namespace complex_number_identity_l248_248182

theorem complex_number_identity (m : ℝ) (h : m + ((m ^ 2 - 4) * Complex.I) = Complex.re 0 + 1 * Complex.I ↔ m > 0): 
  (Complex.mk m 2 * Complex.mk 2 (-2)⁻¹) = Complex.I := sorry

end complex_number_identity_l248_248182


namespace add_fractions_result_l248_248421

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248421


namespace fraction_addition_l248_248443

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248443


namespace max_distance_m_l248_248966

def circle_eq (x y : ℝ) := x^2 + y^2 - 4*x + 6*y - 3 = 0
def line_eq (m x y : ℝ) := m * x + y + m - 1 = 0
def center_circle (x y : ℝ) := circle_eq x y → (x = 2) ∧ (y = -3)

theorem max_distance_m :
  ∃ m : ℝ, line_eq m (-1) 1 ∧ ∀ x y t u : ℝ, center_circle x y → line_eq m t u → 
  -(4 / 3) * -m = -1 → m = -(3 / 4) :=
sorry

end max_distance_m_l248_248966


namespace count_numbers_with_digit_2_from_200_to_499_l248_248769

def count_numbers_with_digit_2 (lower upper : ℕ) : ℕ :=
  let A := 100  -- Numbers of the form 2xx (from 200 to 299)
  let B := 30   -- Numbers of the form x2x (where first digit is 2, 3, or 4, last digit can be any)
  let C := 30   -- Numbers of the form xx2 (similar reasoning as B)
  let A_and_B := 10  -- Numbers of the form 22x
  let A_and_C := 10  -- Numbers of the form 2x2
  let B_and_C := 3   -- Numbers of the form x22
  let A_and_B_and_C := 1  -- The number 222
  A + B + C - A_and_B - A_and_C - B_and_C + A_and_B_and_C

theorem count_numbers_with_digit_2_from_200_to_499 : 
  count_numbers_with_digit_2 200 499 = 138 :=
by
  unfold count_numbers_with_digit_2
  exact rfl

end count_numbers_with_digit_2_from_200_to_499_l248_248769


namespace smallest_integer_in_correct_range_l248_248746

theorem smallest_integer_in_correct_range :
  ∃ (n : ℤ), n > 1 ∧ n % 3 = 1 ∧ n % 5 = 1 ∧ n % 8 = 1 ∧ n % 7 = 2 ∧ 161 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_integer_in_correct_range_l248_248746


namespace original_number_of_people_is_fifteen_l248_248231

/-!
The average age of all the people who gathered at a family celebration was equal to the number of attendees. 
Aunt Beta, who was 29 years old, soon excused herself and left. 
Even after Aunt Beta left, the average age of all the remaining attendees was still equal to their number.
Prove that the original number of people at the celebration is 15.
-/

theorem original_number_of_people_is_fifteen
  (n : ℕ)
  (s : ℕ)
  (h1 : s = n^2)
  (h2 : s - 29 = (n - 1)^2):
  n = 15 :=
by
  sorry

end original_number_of_people_is_fifteen_l248_248231


namespace part1_part2_l248_248218

open Set Real

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | 2 ≤ x ∧ x < 5 }
def setB : Set ℝ := { x | 1 < x ∧ x < 8 }
def setC (a : ℝ) : Set ℝ := { x | x < a - 1 ∨ x > a }

-- Conditions:
-- - Complement of A
def complementA : Set ℝ := { x | x < 2 ∨ x ≥ 5 }

-- Question parts:
-- (1) Finding intersection of complementA and B
theorem part1 : (complementA ∩ setB) = { x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 8) } := sorry

-- (2) Finding range of a for specific condition on C
theorem part2 (a : ℝ) : (setA ∪ setC a = univ) → (a ≤ 2 ∨ a > 6) := sorry

end part1_part2_l248_248218


namespace inequality_inequality_l248_248670

open Real

theorem inequality_inequality (n : ℕ) (k : ℝ) (hn : 0 < n) (hk : 0 < k) : 
  1 - 1/k ≤ n * (k^(1 / n) - 1) ∧ n * (k^(1 / n) - 1) ≤ k - 1 := 
  sorry

end inequality_inequality_l248_248670


namespace fraction_filled_in_5_minutes_l248_248392

-- Conditions
def fill_time : ℕ := 55 -- Total minutes to fill the cistern
def duration : ℕ := 5  -- Minutes we are examining

-- The theorem to prove that the fraction filled in 'duration' minutes is 1/11
theorem fraction_filled_in_5_minutes : (duration : ℚ) / (fill_time : ℚ) = 1 / 11 :=
by
  have fraction_per_minute : ℚ := 1 / fill_time
  have fraction_in_5_minutes : ℚ := duration * fraction_per_minute
  sorry -- Proof steps would go here, if needed.

end fraction_filled_in_5_minutes_l248_248392


namespace marco_older_than_twice_marie_l248_248347

variable (M m x : ℕ)

def marie_age : ℕ := 12
def sum_of_ages : ℕ := 37

theorem marco_older_than_twice_marie :
  m = marie_age → (M = 2 * m + x) → (M + m = sum_of_ages) → x = 1 :=
by
  intros h1 h2 h3
  rw [h1] at h2 h3
  sorry

end marco_older_than_twice_marie_l248_248347


namespace find_number_l248_248631

theorem find_number (x q : ℕ) (h1 : x = 7 * q) (h2 : q + x + 7 = 175) : x = 147 := 
by
  sorry

end find_number_l248_248631


namespace trigonometric_problem_l248_248751

theorem trigonometric_problem
  (α : ℝ)
  (h1 : Real.tan α = Real.sqrt 3)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.cos (2 * π - α) - Real.sin α = (Real.sqrt 3 - 1) / 2 := by
  sorry

end trigonometric_problem_l248_248751


namespace extra_taxes_paid_l248_248128

variables (initial_tax_rate new_tax_rate initial_income new_income : ℝ)

-- Conditions
def initial_tax_rate := 0.2
def new_tax_rate := 0.3
def initial_income := 1000000
def new_income := 1500000

-- The statement about the extra taxes paid
theorem extra_taxes_paid :
  (new_tax_rate * new_income) - (initial_tax_rate * initial_income) = 250000 :=
by
  sorry

end extra_taxes_paid_l248_248128


namespace sum_fractions_eq_l248_248473

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248473


namespace A_leaves_after_one_day_l248_248977

-- Define and state all the conditions
def A_work_rate := 1 / 21
def B_work_rate := 1 / 28
def C_work_rate := 1 / 35
def total_work := 1
def B_time_after_A_leave := 21
def C_intermittent_working_cycle := 3 / 1 -- C works 1 out of every 3 days

-- The statement that needs to be proved
theorem A_leaves_after_one_day :
  ∃ x : ℕ, x = 1 ∧
  (A_work_rate * x + B_work_rate * x + (C_work_rate * (x / C_intermittent_working_cycle)) + (B_work_rate * B_time_after_A_leave) + (C_work_rate * (B_time_after_A_leave / C_intermittent_working_cycle)) = total_work) :=
sorry

end A_leaves_after_one_day_l248_248977


namespace sticks_form_equilateral_triangle_l248_248149

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l248_248149


namespace division_yields_square_l248_248178

theorem division_yields_square (a b : ℕ) (hab : ab + 1 ∣ a^2 + b^2) :
  ∃ m : ℕ, m^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end division_yields_square_l248_248178


namespace bookkeeper_arrangements_l248_248027

theorem bookkeeper_arrangements :
  (fact 10) / ((fact 2) * (fact 2) * (fact 2) * (fact 2)) = 226800 :=
by
  sorry

end bookkeeper_arrangements_l248_248027


namespace add_fractions_l248_248587

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248587


namespace multiplication_result_l248_248701

theorem multiplication_result :
  (500 ^ 50) * (2 ^ 100) = 10 ^ 75 :=
by
  sorry

end multiplication_result_l248_248701


namespace no_integer_roots_if_coefficients_are_odd_l248_248224

theorem no_integer_roots_if_coefficients_are_odd (a b c x : ℤ) 
  (h1 : Odd a) (h2 : Odd b) (h3 : Odd c) (h4 : a * x^2 + b * x + c = 0) : False := 
by
  sorry

end no_integer_roots_if_coefficients_are_odd_l248_248224


namespace problem1_problem3_l248_248895

-- Define the function f(x)
def f (x : ℚ) : ℚ := (1 - x) / (1 + x)

-- Problem 1: Prove f(1/x) = -f(x), given x ≠ -1, x ≠ 0
theorem problem1 (x : ℚ) (hx1 : x ≠ -1) (hx2 : x ≠ 0) : f (1 / x) = -f x :=
by sorry

-- Problem 2: Comment on graph transformations for f(x)
-- This is a conceptual question about graph translation and is not directly translatable to a Lean theorem.

-- Problem 3: Find the minimum value of M - m such that m ≤ f(x) ≤ M for x ∈ ℤ
theorem problem3 : ∃ (M m : ℤ), (∀ x : ℤ, m ≤ f x ∧ f x ≤ M) ∧ (M - m = 4) :=
by sorry

end problem1_problem3_l248_248895


namespace longest_side_of_enclosure_l248_248657

theorem longest_side_of_enclosure (l w : ℝ)
  (h_perimeter : 2 * l + 2 * w = 240)
  (h_area : l * w = 8 * 240) :
  max l w = 80 :=
by
  sorry

end longest_side_of_enclosure_l248_248657


namespace fraction_addition_l248_248464

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248464


namespace sum_of_fractions_l248_248429

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248429


namespace add_fractions_l248_248486

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248486


namespace range_of_values_l248_248897

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem range_of_values (x : ℝ) : f (2 * x) > f (x - 3) ↔ x < -3 ∨ x > 1 := 
by
  sorry

end range_of_values_l248_248897


namespace total_money_amount_l248_248791

-- Define the conditions
def num_bills : ℕ := 3
def value_per_bill : ℕ := 20
def initial_amount : ℕ := 75

-- Define the statement about the total amount of money James has
theorem total_money_amount : num_bills * value_per_bill + initial_amount = 135 := 
by 
  -- Since the proof is not required, we use 'sorry' to skip it
  sorry

end total_money_amount_l248_248791


namespace sum_fractions_eq_l248_248480

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248480


namespace chris_initial_donuts_l248_248276

theorem chris_initial_donuts (D : ℝ) (H1 : D * 0.90 - 4 = 23) : D = 30 := 
by
sorry

end chris_initial_donuts_l248_248276


namespace carl_additional_hours_per_week_l248_248099

def driving_hours_per_day : ℕ := 2

def days_per_week : ℕ := 7

def total_hours_two_weeks_after_promotion : ℕ := 40

def driving_hours_per_week_before_promotion : ℕ := driving_hours_per_day * days_per_week

def driving_hours_per_week_after_promotion : ℕ := total_hours_two_weeks_after_promotion / 2

def additional_hours_per_week : ℕ := driving_hours_per_week_after_promotion - driving_hours_per_week_before_promotion

theorem carl_additional_hours_per_week : 
  additional_hours_per_week = 6 :=
by
  -- Using plain arithmetic based on given definitions
  sorry

end carl_additional_hours_per_week_l248_248099


namespace max_perimeter_right_triangle_l248_248327

theorem max_perimeter_right_triangle (a b : ℝ) (h₁ : a^2 + b^2 = 25) :
  (a + b + 5) ≤ 5 + 5 * Real.sqrt 2 :=
by
  sorry

end max_perimeter_right_triangle_l248_248327


namespace complex_w_sixth_power_l248_248646

theorem complex_w_sixth_power :
  let w := Complex.ofReal (-1) / 2 + Complex.I * Real.sqrt 3 / 2 in
  w ^ 6 = 1 / 4 :=
by
  sorry

end complex_w_sixth_power_l248_248646


namespace circles_internally_tangent_l248_248238

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x - 6)^2 + y^2 = 1 → 
  (x - 3)^2 + (y - 4)^2 = 36 → 
  true := 
by 
  intros x y h1 h2
  sorry

end circles_internally_tangent_l248_248238


namespace probability_of_last_two_marbles_one_green_one_red_l248_248386

theorem probability_of_last_two_marbles_one_green_one_red : 
    let total_marbles := 10
    let blue := 4
    let white := 3
    let red := 2
    let green := 1
    let total_ways := Nat.choose total_marbles 8
    let favorable_ways := Nat.choose (total_marbles - red - green) 6
    total_ways = 45 ∧ favorable_ways = 28 →
    (favorable_ways : ℚ) / total_ways = 28 / 45 :=
by
    intros total_marbles blue white red green total_ways favorable_ways h
    sorry

end probability_of_last_two_marbles_one_green_one_red_l248_248386


namespace sum_of_reciprocal_roots_l248_248344

theorem sum_of_reciprocal_roots (r s α β : ℝ) (h1 : 7 * r^2 - 8 * r + 6 = 0) (h2 : 7 * s^2 - 8 * s + 6 = 0) (h3 : α = 1 / r) (h4 : β = 1 / s) :
  α + β = 4 / 3 := 
sorry

end sum_of_reciprocal_roots_l248_248344


namespace point_a_number_l248_248819

theorem point_a_number (x : ℝ) (h : abs (x - 2) = 6) : x = 8 ∨ x = -4 :=
sorry

end point_a_number_l248_248819


namespace factorial_fraction_l248_248731

theorem factorial_fraction : 
  (4 * nat.factorial 6 + 24 * nat.factorial 5) / nat.factorial 7 = 48 / 7 := 
by
  sorry

end factorial_fraction_l248_248731


namespace length_of_parallelepiped_l248_248998

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248998


namespace find_third_triangle_angles_l248_248842

-- Define the problem context
variables {A B C : ℝ} -- angles of the original triangle

-- Condition: The sum of the angles in a triangle is 180 degrees
axiom sum_of_angles (a b c : ℝ) : a + b + c = 180

-- Given conditions about the triangle and inscribed circles
def original_triangle (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

def inscribed_circle (a b c : ℝ) : Prop :=
original_triangle a b c

def second_triangle (a b c : ℝ) : Prop :=
inscribed_circle a b c

def third_triangle (a b c : ℝ) : Prop :=
second_triangle a b c

-- Goal: Prove that the angles in the third triangle are 60 degrees each
theorem find_third_triangle_angles (a b c : ℝ) (ha : original_triangle a b c)
  (h_inscribed : inscribed_circle a b c)
  (h_second : second_triangle a b c)
  (h_third : third_triangle a b c) : a = 60 ∧ b = 60 ∧ c = 60 := by
sorry

end find_third_triangle_angles_l248_248842


namespace find_number_l248_248259

def incorrect_multiplication (x : ℕ) : ℕ := 394 * x
def correct_multiplication (x : ℕ) : ℕ := 493 * x
def difference (x : ℕ) : ℕ := correct_multiplication x - incorrect_multiplication x
def expected_difference : ℕ := 78426

theorem find_number (x : ℕ) (h : difference x = expected_difference) : x = 792 := by
  sorry

end find_number_l248_248259


namespace nested_fraction_evaluation_l248_248001

def nested_expression := 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))

theorem nested_fraction_evaluation : nested_expression = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l248_248001


namespace set_proof_l248_248710

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem set_proof :
  (U \ A) ∩ (U \ B) = {4, 8} := by
  sorry

end set_proof_l248_248710


namespace remaining_units_correct_l248_248398

-- Definitions based on conditions
def total_units : ℕ := 2000
def fraction_built_in_first_half : ℚ := 3/5
def additional_units_by_october : ℕ := 300

-- Calculate units built in the first half of the year
def units_built_in_first_half : ℚ := fraction_built_in_first_half * total_units

-- Remaining units after the first half of the year
def remaining_units_after_first_half : ℚ := total_units - units_built_in_first_half

-- Remaining units after building additional units by October
def remaining_units_to_be_built : ℚ := remaining_units_after_first_half - additional_units_by_october

-- Theorem statement: Prove remaining units to be built is 500
theorem remaining_units_correct : remaining_units_to_be_built = 500 := by
  sorry

end remaining_units_correct_l248_248398


namespace num_ordered_pairs_solutions_l248_248293

theorem num_ordered_pairs_solutions :
  ∃ (n : ℕ), n = 18 ∧
    (∀ (a b : ℝ), (∃ x y : ℤ , a * (x : ℝ) + b * (y : ℝ) = 1 ∧ (x * x + y * y = 50))) :=
sorry

end num_ordered_pairs_solutions_l248_248293


namespace x_coordinate_at_2005th_stop_l248_248835

theorem x_coordinate_at_2005th_stop :
 (∃ (f : ℕ → ℤ × ℤ),
    f 0 = (0, 0) ∧
    f 1 = (1, 0) ∧
    f 2 = (1, 1) ∧
    f 3 = (0, 1) ∧
    f 4 = (-1, 1) ∧
    f 5 = (-1, 0) ∧
    f 9 = (2, -1))
  → (∃ (f : ℕ → ℤ × ℤ), f 2005 = (3, -n)) := sorry

end x_coordinate_at_2005th_stop_l248_248835


namespace find_x_l248_248261

noncomputable def x : ℝ := 20

def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := x / 100 * 150 - 20 = 10

theorem find_x (x : ℝ) : condition1 x ∧ condition2 x ↔ x = 20 :=
by
  sorry

end find_x_l248_248261


namespace cylinder_height_decrease_l248_248965

/--
Two right circular cylinders have the same volume. The radius of the second cylinder is 20% more than the radius
of the first. Prove that the height of the second cylinder is approximately 30.56% less than the first one's height.
-/
theorem cylinder_height_decrease (r1 h1 r2 h2 : ℝ) (hradius : r2 = 1.2 * r1) (hvolumes : π * r1^2 * h1 = π * r2^2 * h2) :
  h2 = 25 / 36 * h1 :=
by
  sorry

end cylinder_height_decrease_l248_248965


namespace equiangular_polygon_angle_solution_l248_248025

-- Given two equiangular polygons P_1 and P_2 with different numbers of sides
-- Each angle of P_1 is x degrees
-- Each angle of P_2 is k * x degrees where k is an integer greater than 1
-- Prove that the number of valid pairs (x, k) is exactly 1

theorem equiangular_polygon_angle_solution : ∃ x k : ℕ, ( ∀ n m : ℕ, x = 180 - 360 / n ∧ k * x = 180 - 360 / m → (k > 1) → x = 60 ∧ k = 2) := sorry

end equiangular_polygon_angle_solution_l248_248025


namespace proof_a_eq_b_pow_n_l248_248300

theorem proof_a_eq_b_pow_n
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n := 
by sorry

end proof_a_eq_b_pow_n_l248_248300


namespace sticks_form_equilateral_triangle_l248_248152

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l248_248152


namespace initial_caps_correct_l248_248796

variable (bought : ℕ)
variable (total : ℕ)

def initial_bottle_caps (bought : ℕ) (total : ℕ) : ℕ :=
  total - bought

-- Given conditions
def bought_caps : ℕ := 7
def total_caps : ℕ := 47

theorem initial_caps_correct : initial_bottle_caps bought_caps total_caps = 40 :=
by
  -- proof here
  sorry

end initial_caps_correct_l248_248796


namespace find_all_functions_satisfying_functional_equation_l248_248604

theorem find_all_functions_satisfying_functional_equation :
  ∀ (f : ℚ → ℚ), (∀ x y : ℚ, f (x + y) = f x + f y) →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by {
  intro f,
  assume h,
  -- Here, "sorry" is a placeholder.
  sorry,
}

end find_all_functions_satisfying_functional_equation_l248_248604


namespace add_fractions_l248_248579

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248579


namespace sum_fractions_eq_l248_248478

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248478


namespace semicircle_arc_length_l248_248817

theorem semicircle_arc_length (a b : ℝ) (hypotenuse_sum : a + b = 70) (a_eq_30 : a = 30) (b_eq_40 : b = 40) :
  ∃ (R : ℝ), (R = 24) ∧ (π * R = 12 * π) :=
by
  sorry

end semicircle_arc_length_l248_248817


namespace units_digit_of_product_l248_248608

theorem units_digit_of_product : 
  (27 % 10 = 7) ∧ (68 % 10 = 8) → ((27 * 68) % 10 = 6) :=
by sorry

end units_digit_of_product_l248_248608


namespace sin_cos_sum_eq_l248_248183

theorem sin_cos_sum_eq (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan (θ + π / 4) = 1 / 2): 
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := 
  sorry

end sin_cos_sum_eq_l248_248183


namespace probability_target_hit_l248_248931

theorem probability_target_hit {P_A P_B : ℚ}
  (hA : P_A = 1 / 2) 
  (hB : P_B = 1 / 3) 
  : (1 - (1 - P_A) * (1 - P_B)) = 2 / 3 := 
by
  sorry

end probability_target_hit_l248_248931


namespace sum_fractions_eq_l248_248475

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248475


namespace factor_squared_of_symmetric_poly_l248_248709

theorem factor_squared_of_symmetric_poly (P : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ)
  (h_symm : ∀ x y, P x y = P y x)
  (h_factor : ∀ x y, (x - y) ∣ P x y) :
  ∀ x y, (x - y) ^ 2 ∣ P x y := 
sorry

end factor_squared_of_symmetric_poly_l248_248709


namespace radius_of_intersection_l248_248266

noncomputable def sphere_radius := 2 * Real.sqrt 17

theorem radius_of_intersection (s : ℝ) 
  (h1 : (3:ℝ)=(3:ℝ)) (h2 : (5:ℝ)=(5:ℝ)) (h3 : (0-3:ℝ)^2 + (5-5:ℝ)^2 + (s-(-8+8))^2 = sphere_radius^2) :
  s = Real.sqrt 59 :=
by
  sorry

end radius_of_intersection_l248_248266


namespace spherical_coord_plane_l248_248007

-- Let's define spherical coordinates and the condition theta = c.
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def is_plane (c : ℝ) (p : SphericalCoordinates) : Prop :=
  p.θ = c

theorem spherical_coord_plane (c : ℝ) : 
  ∀ p : SphericalCoordinates, is_plane c p → True := 
by
  intros p hp
  sorry

end spherical_coord_plane_l248_248007


namespace three_digit_diff_no_repeated_digits_l248_248249

theorem three_digit_diff_no_repeated_digits :
  let largest := 987
  let smallest := 102
  largest - smallest = 885 := by
  sorry

end three_digit_diff_no_repeated_digits_l248_248249


namespace quadratic_intersects_x_axis_l248_248625

theorem quadratic_intersects_x_axis (a b : ℝ) (h : a ≠ 0) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 - (b^2 / (4 * a)) = 0 ∧ a * x2^2 + b * x2 - (b^2 / (4 * a)) = 0 := by
  sorry

end quadratic_intersects_x_axis_l248_248625


namespace zilla_savings_l248_248113

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l248_248113


namespace find_a_from_function_l248_248896

theorem find_a_from_function (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt (2 * x + 1)) (a : ℝ) (h_a : f a = 5) : a = 12 :=
by
  sorry

end find_a_from_function_l248_248896


namespace R_and_D_per_increase_l248_248272

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end R_and_D_per_increase_l248_248272


namespace probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l248_248851

noncomputable def probability_first_third_fifth_hit : ℚ :=
  (3 / 5) * (2 / 5) * (3 / 5) * (2 / 5) * (3 / 5)

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  ↑(Nat.factorial n) / (↑(Nat.factorial k) * ↑(Nat.factorial (n - k)))

noncomputable def probability_exactly_three_hits : ℚ :=
  binomial_coefficient 5 3 * (3 / 5)^3 * (2 / 5)^2

theorem probability_first_third_fifth_correct :
  probability_first_third_fifth_hit = 108 / 3125 :=
by sorry

theorem probability_exactly_three_hits_correct :
  probability_exactly_three_hits = 216 / 625 :=
by sorry

end probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l248_248851


namespace sum_of_fractions_l248_248430

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248430


namespace factorize_difference_of_squares_l248_248601

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
by 
  sorry

end factorize_difference_of_squares_l248_248601


namespace domain_of_function_l248_248288

def domain_sqrt_log : Set ℝ :=
  {x | (2 - x ≥ 0) ∧ ((2 * x - 1) / (3 - x) > 0)}

theorem domain_of_function :
  domain_sqrt_log = {x | (1/2 < x) ∧ (x ≤ 2)} :=
by
  sorry

end domain_of_function_l248_248288


namespace correct_option_is_optionB_l248_248676

-- Definitions based on conditions
def optionA : ℝ := 0.37 * 1.5
def optionB : ℝ := 3.7 * 1.5
def optionC : ℝ := 0.37 * 1500
def original : ℝ := 0.37 * 15

-- Statement to prove that the correct answer (optionB) yields the same result as the original expression
theorem correct_option_is_optionB : optionB = original :=
sorry

end correct_option_is_optionB_l248_248676


namespace yuna_candy_days_l248_248253

theorem yuna_candy_days (total_candies : ℕ) (daily_candies_week : ℕ) (days_week : ℕ) (remaining_candies : ℕ) (daily_candies_future : ℕ) :
  total_candies = 60 →
  daily_candies_week = 6 →
  days_week = 7 →
  remaining_candies = total_candies - (daily_candies_week * days_week) →
  daily_candies_future = 3 →
  remaining_candies / daily_candies_future = 6 :=
by
  intros h_total h_daily_week h_days_week h_remaining h_daily_future
  sorry

end yuna_candy_days_l248_248253


namespace two_digit_number_representation_l248_248609

theorem two_digit_number_representation (m n : ℕ) (hm : m < 10) (hn : n < 10) : 10 * n + m = m + 10 * n :=
by sorry

end two_digit_number_representation_l248_248609


namespace add_fractions_l248_248517

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248517


namespace min_value_a_plus_b_l248_248055

theorem min_value_a_plus_b (a b : ℕ) (h₁ : 79 ∣ (a + 77 * b)) (h₂ : 77 ∣ (a + 79 * b)) : a + b = 193 :=
by
  sorry

end min_value_a_plus_b_l248_248055


namespace problem1_problem2_l248_248831

section ProofProblems

-- Definitions for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Prove that n! = binom(n, k) * k! * (n-k)!
theorem problem1 (n k : ℕ) : n.factorial = binom n k * k.factorial * (n - k).factorial :=
by sorry

-- Problem 2: Prove that binom(n, k) = binom(n-1, k) + binom(n-1, k-1)
theorem problem2 (n k : ℕ) : binom n k = binom (n-1) k + binom (n-1) (k-1) :=
by sorry

end ProofProblems

end problem1_problem2_l248_248831


namespace trigonometric_identity_l248_248614

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

end trigonometric_identity_l248_248614


namespace axis_of_symmetry_l248_248821

theorem axis_of_symmetry (x : ℝ) (h : x = -Real.pi / 12) :
  ∃ k : ℤ, 2 * x - Real.pi / 3 = k * Real.pi + Real.pi / 2 :=
sorry

end axis_of_symmetry_l248_248821


namespace units_digit_G_100_l248_248065

def G (n : ℕ) : ℕ := 3 ^ (2 ^ n) + 1

theorem units_digit_G_100 : (G 100) % 10 = 2 := 
by
  sorry

end units_digit_G_100_l248_248065


namespace total_surface_area_of_square_pyramid_is_correct_l248_248212

-- Define the base side length and height from conditions
def a : ℝ := 3
def PD : ℝ := 4

-- Conditions
def square_pyramid : Prop :=
  let AD := a
  let PA := Real.sqrt (PD^2 - a^2)
  let Area_PAD := (1 / 2) * AD * PA
  let Area_PCD := Area_PAD
  let Area_base := a * a
  let Total_surface_area := Area_base + 2 * Area_PAD + 2 * Area_PCD
  Total_surface_area = 9 + 6 * Real.sqrt 7

-- Theorem statement
theorem total_surface_area_of_square_pyramid_is_correct : square_pyramid := sorry

end total_surface_area_of_square_pyramid_is_correct_l248_248212


namespace solution_alcohol_content_l248_248812

noncomputable def volume_of_solution_y_and_z (V: ℝ) : Prop :=
  let vol_X := 300.0
  let conc_X := 0.10
  let conc_Y := 0.30
  let conc_Z := 0.40
  let vol_Y := 2 * V
  let vol_new := vol_X + vol_Y + V
  let alcohol_new := conc_X * vol_X + conc_Y * vol_Y + conc_Z * V
  (alcohol_new / vol_new) = 0.22

theorem solution_alcohol_content : volume_of_solution_y_and_z 300.0 :=
by
  sorry

end solution_alcohol_content_l248_248812


namespace remainder_of_product_l248_248708

theorem remainder_of_product (a b n : ℕ) (ha : a % n = 7) (hb : b % n = 1) :
  ((a * b) % n) = 7 :=
by
  -- Definitions as per the conditions
  let a := 63
  let b := 65
  let n := 8
  /- Now prove the statement -/
  sorry

end remainder_of_product_l248_248708


namespace sticks_forming_equilateral_triangle_l248_248153

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l248_248153


namespace value_of_A_l248_248031

def clubsuit (A B : ℕ) := 3 * A + 2 * B + 5

theorem value_of_A (A : ℕ) (h : clubsuit A 7 = 82) : A = 21 :=
by
  sorry

end value_of_A_l248_248031


namespace sum_of_cubes_equality_l248_248159

theorem sum_of_cubes_equality (a b p n : ℕ) (hp : Nat.Prime p) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^3 + b^3 = p^n) ↔ 
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
  (∃ k : ℕ, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end sum_of_cubes_equality_l248_248159


namespace fraction_addition_l248_248442

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248442


namespace sum_of_fractions_l248_248432

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248432


namespace sqrt_simplify_l248_248316

theorem sqrt_simplify (a b x : ℝ) (h : a < b) (hx1 : x + b ≥ 0) (hx2 : x + a ≤ 0) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * (Real.sqrt (-(x + a) * (x + b))) :=
by
  sorry

end sqrt_simplify_l248_248316


namespace total_fishermen_count_l248_248371

theorem total_fishermen_count (F T F1 F2 : ℕ) (hT : T = 10000) (hF1 : F1 = 19 * 400) (hF2 : F2 = 2400) (hTotal : F1 + F2 = T) : F = 20 :=
by
  sorry

end total_fishermen_count_l248_248371


namespace no_such_function_exists_l248_248340

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∃ M > 0, ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧
                    (f 1 = 1) ∧
                    (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end no_such_function_exists_l248_248340


namespace interest_rate_bc_l248_248718

def interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

def gain_b (interest_bc interest_ab : ℝ) : ℝ :=
  interest_bc - interest_ab

theorem interest_rate_bc :
  ∀ (principal : ℝ) (rate_ab rate_bc : ℝ) (time : ℕ) (gain : ℝ),
    principal = 3500 → rate_ab = 0.10 → time = 3 → gain = 525 →
    interest principal rate_ab time = 1050 →
    gain_b (interest principal rate_bc time) (interest principal rate_ab time) = gain →
    rate_bc = 0.15 :=
by
  intros principal rate_ab rate_bc time gain h_principal h_rate_ab h_time h_gain h_interest_ab h_gain_b
  sorry

end interest_rate_bc_l248_248718


namespace cone_base_circumference_l248_248988

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) 
  (hV : V = 18 * Real.pi)
  (hh : h = 6) 
  (hV_cone : V = (1/3) * Real.pi * r^2 * h) :
  C = 2 * Real.pi * r → C = 6 * Real.pi :=
by 
  -- We assume as conditions are only mentioned
  sorry

end cone_base_circumference_l248_248988


namespace triangle_DOE_area_l248_248400

theorem triangle_DOE_area
  (area_ABC : ℝ)
  (DO : ℝ) (OB : ℝ)
  (EO : ℝ) (OA : ℝ)
  (h_area_ABC : area_ABC = 1)
  (h_DO_OB : DO / OB = 1 / 3)
  (h_EO_OA : EO / OA = 4 / 5)
  : (1 / 4) * (4 / 9) * area_ABC = 11 / 135 := 
by 
  sorry

end triangle_DOE_area_l248_248400


namespace expression_value_l248_248107

theorem expression_value (x : ℝ) (h : x = 4) :
  (x^2 - 2*x - 15) / (x - 5) = 7 :=
sorry

end expression_value_l248_248107


namespace length_reduction_by_50_percent_l248_248684

variable (L B L' : ℝ)

def rectangle_dimension_change (L B : ℝ) (perc_area_change : ℝ) (new_breadth_factor : ℝ) : Prop :=
  let original_area := L * B
  let new_breadth := new_breadth_factor * B
  let new_area := L' * new_breadth
  let expected_new_area := (1 + perc_area_change) * original_area
  new_area = expected_new_area

theorem length_reduction_by_50_percent (L B : ℝ) (h1: rectangle_dimension_change L B L' 0.5 3) : 
  L' = 0.5 * L :=
by
  unfold rectangle_dimension_change at h1
  simp at h1
  sorry

end length_reduction_by_50_percent_l248_248684


namespace train_speed_equals_36_0036_l248_248393

noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_equals_36_0036 :
  train_speed 70 6.999440044796416 = 36.0036 :=
by
  unfold train_speed
  sorry

end train_speed_equals_36_0036_l248_248393


namespace sum_of_fractions_l248_248431

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248431


namespace find_a_equiv_l248_248283

theorem find_a_equiv (a x : ℝ) (h : ∀ x, (a * x^2 + 20 * x + 25) = (2 * x + 5) * (2 * x + 5)) : a = 4 :=
by
  sorry

end find_a_equiv_l248_248283


namespace solve_equation_l248_248814

theorem solve_equation (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) → x = -0.5 := 
by
  sorry

end solve_equation_l248_248814


namespace altitude_angle_bisector_inequality_l248_248886

theorem altitude_angle_bisector_inequality
  (h l R r : ℝ) 
  (triangle_condition : ∀ (h l : ℝ) (R r : ℝ), (h > 0 ∧ l > 0 ∧ R > 0 ∧ r > 0)) :
  h / l ≥ Real.sqrt (2 * r / R) :=
by
  sorry

end altitude_angle_bisector_inequality_l248_248886


namespace find_P_l248_248847

noncomputable def parabola_vertex : ℝ × ℝ := (0, 0)
noncomputable def parabola_focus : ℝ × ℝ := (0, -1)
noncomputable def point_P : ℝ × ℝ := (20 * Real.sqrt 6, -120)
noncomputable def PF_distance : ℝ := 121

def parabola_equation (x y : ℝ) : Prop :=
  x^2 = -4 * y

def parabola_condition (x y : ℝ) : Prop :=
  (parabola_equation x y) ∧ 
  (Real.sqrt (x^2 + (y + 1)^2) = PF_distance)

theorem find_P : parabola_condition (point_P.1) (point_P.2) :=
by
  sorry

end find_P_l248_248847


namespace factorize_x_squared_minus_sixteen_l248_248869

theorem factorize_x_squared_minus_sixteen (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) :=
by
  sorry

end factorize_x_squared_minus_sixteen_l248_248869


namespace fraction_addition_l248_248567

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248567


namespace trapezoid_upper_base_BC_l248_248176

theorem trapezoid_upper_base_BC (A B C D M : Point) (d : ℝ)
  (h1 : Trapezoid A B C D)
  (h2 : OnLine M A B)
  (h3 : Perpendicular D M A B)
  (h4 : Distance M C = Distance C D)
  (h5 : Distance A D = d) : Distance B C = d / 2 := 
sorry

end trapezoid_upper_base_BC_l248_248176


namespace tethered_dog_area_comparison_l248_248864

theorem tethered_dog_area_comparison :
  let fence_radius := 20
  let rope_length := 30
  let arrangement1_area := π * (rope_length ^ 2)
  let tether_distance := 12
  let arrangement2_effective_radius := rope_length - tether_distance
  let arrangement2_full_circle_area := π * (arrangement2_effective_radius ^ 2)
  let arrangement2_additional_area := (1 / 4) * π * (tether_distance ^ 2)
  let arrangement2_total_area := arrangement2_full_circle_area + arrangement2_additional_area
  (arrangement1_area - arrangement2_total_area) = 540 * π := 
by
  sorry

end tethered_dog_area_comparison_l248_248864


namespace find_a_max_min_f_l248_248191

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem find_a (a : ℝ) (h : (deriv (f a) 0 = 1)) : a = 1 :=
by sorry

noncomputable def f_one (x : ℝ) : ℝ := f 1 x

theorem max_min_f (h : ∀ x, 0 ≤ x → x ≤ 2 → deriv (f_one) x > 0) :
  (f_one 0 = 0) ∧ (f_one 2 = 2 * Real.exp 2) :=
by sorry

end find_a_max_min_f_l248_248191


namespace point_on_x_axis_l248_248912

theorem point_on_x_axis (m : ℝ) (h : (2 * m + 3) = 0) : m = -3 / 2 :=
sorry

end point_on_x_axis_l248_248912


namespace remainder_when_dividing_sum_l248_248596

theorem remainder_when_dividing_sum (k m : ℤ) (c d : ℤ) (h1 : c = 60 * k + 47) (h2 : d = 42 * m + 17) :
  (c + d) % 21 = 1 :=
by
  sorry

end remainder_when_dividing_sum_l248_248596


namespace sum_fractions_eq_l248_248471

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248471


namespace total_pitches_missed_l248_248926

theorem total_pitches_missed (tokens_to_pitches : ℕ → ℕ) 
  (macy_used : ℕ) (piper_used : ℕ) 
  (macy_hits : ℕ) (piper_hits : ℕ) 
  (h1 : tokens_to_pitches 1 = 15) 
  (h_macy_used : macy_used = 11) 
  (h_piper_used : piper_used = 17) 
  (h_macy_hits : macy_hits = 50) 
  (h_piper_hits : piper_hits = 55) :
  let total_pitches := tokens_to_pitches macy_used + tokens_to_pitches piper_used
  let total_hits := macy_hits + piper_hits
  total_pitches - total_hits = 315 :=
by
  sorry

end total_pitches_missed_l248_248926


namespace intersection_M_N_l248_248626

-- Define the sets based on the given conditions
def M : Set ℝ := {x | x + 2 < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {x | x < -2} := by
sorry

end intersection_M_N_l248_248626


namespace heather_total_distance_l248_248026

theorem heather_total_distance :
  let d1 := 0.3333333333333333
  let d2 := 0.3333333333333333
  let d3 := 0.08333333333333333
  d1 + d2 + d3 = 0.75 :=
by
  sorry

end heather_total_distance_l248_248026


namespace avg_weight_a_b_l248_248949

theorem avg_weight_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 60)
  (h2 : (B + C) / 2 = 50)
  (h3 : B = 60) :
  (A + B) / 2 = 70 := 
sorry

end avg_weight_a_b_l248_248949


namespace fourth_power_square_prime_l248_248250

noncomputable def fourth_smallest_prime := 7

theorem fourth_power_square_prime :
  (fourth_smallest_prime ^ 2) ^ 4 = 5764801 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end fourth_power_square_prime_l248_248250


namespace cart_max_speed_l248_248723

noncomputable def maximum_speed (a R : ℝ) : ℝ :=
  (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4)

theorem cart_max_speed (a R v : ℝ) (h : v = maximum_speed a R) : 
  v = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4) :=
by
  -- Proof is omitted
  sorry

end cart_max_speed_l248_248723


namespace add_fractions_l248_248502

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248502


namespace spaceship_not_moving_time_l248_248395

-- Definitions based on the conditions given
def total_journey_time : ℕ := 3 * 24  -- 3 days in hours

def first_travel_time : ℕ := 10
def first_break_time : ℕ := 3
def second_travel_time : ℕ := 10
def second_break_time : ℕ := 1

def subsequent_travel_period : ℕ := 11  -- 11 hours traveling, then 1 hour break

-- Function to compute total break time
def total_break_time (total_travel_time : ℕ) : ℕ :=
  let remaining_time := total_journey_time - (first_travel_time + first_break_time + second_travel_time + second_break_time)
  let subsequent_breaks := remaining_time / subsequent_travel_period
  first_break_time + second_break_time + subsequent_breaks

theorem spaceship_not_moving_time : total_break_time total_journey_time = 8 := by
  sorry

end spaceship_not_moving_time_l248_248395


namespace ticket_cost_is_nine_l248_248402

theorem ticket_cost_is_nine (bought_tickets : ℕ) (left_tickets : ℕ) (spent_dollars : ℕ) 
  (h1 : bought_tickets = 6) 
  (h2 : left_tickets = 3) 
  (h3 : spent_dollars = 27) : 
  spent_dollars / (bought_tickets - left_tickets) = 9 :=
by
  -- Using the imported library and the given conditions
  sorry

end ticket_cost_is_nine_l248_248402


namespace leaks_drain_time_l248_248132

-- Definitions from conditions
def pump_rate : ℚ := 1 / 2 -- tanks per hour
def leak1_rate : ℚ := 1 / 6 -- tanks per hour
def leak2_rate : ℚ := 1 / 9 -- tanks per hour

-- Proof statement
theorem leaks_drain_time : (leak1_rate + leak2_rate)⁻¹ = 3.6 :=
by
  sorry

end leaks_drain_time_l248_248132


namespace triangle_inequality_from_condition_l248_248837

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end triangle_inequality_from_condition_l248_248837


namespace largest_is_B_l248_248142

noncomputable def A : ℚ := ((2023:ℚ) / 2022) + ((2023:ℚ) / 2024)
noncomputable def B : ℚ := ((2024:ℚ) / 2023) + ((2026:ℚ) / 2023)
noncomputable def C : ℚ := ((2025:ℚ) / 2024) + ((2025:ℚ) / 2026)

theorem largest_is_B : B > A ∧ B > C := by
  sorry

end largest_is_B_l248_248142


namespace factorize_x2_minus_9_l248_248603

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end factorize_x2_minus_9_l248_248603


namespace total_perimeter_l248_248133

/-- 
A rectangular plot where the long sides are three times the length of the short sides. 
One short side is 80 feet. Prove the total perimeter is 640 feet.
-/
theorem total_perimeter (s : ℕ) (h : s = 80) : 8 * s = 640 :=
  by sorry

end total_perimeter_l248_248133


namespace factor_x4_plus_64_monic_real_l248_248741

theorem factor_x4_plus_64_monic_real :
  ∀ x : ℝ, x^4 + 64 = (x^2 + 4 * x + 8) * (x^2 - 4 * x + 8) := 
by
  intros
  sorry

end factor_x4_plus_64_monic_real_l248_248741


namespace find_upper_base_length_l248_248175

variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (AB: line_segment A B) (CD: line_segment C D)
variables (AD : ℝ) (d : ℝ)

noncomputable def upper_base_length (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d)
: Prop :=
BC = d / 2

theorem find_upper_base_length :
∀ (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d),
  upper_base_length A B C D M ABCD DM_perp_AB MC_eq_CD AD_length d_value := sorry

end find_upper_base_length_l248_248175


namespace committee_selection_count_l248_248636

-- Definition of the problem condition: Club of 12 people, one specific person must always be on the committee.
def club_size : ℕ := 12
def committee_size : ℕ := 4
def specific_person_included : ℕ := 1

-- Number of ways to choose 3 members from the other 11 people
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem committee_selection_count : choose 11 3 = 165 := 
  sorry

end committee_selection_count_l248_248636


namespace total_people_at_beach_l248_248658

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l248_248658


namespace symmetric_line_equation_l248_248822

theorem symmetric_line_equation (x y : ℝ) :
  let line_original := x - 2 * y + 1 = 0
  let line_symmetry := x = 1
  let line_symmetric := x + 2 * y - 3 = 0
  ∀ (x y : ℝ), (2 - x - 2 * y + 1 = 0) ↔ (x + 2 * y - 3 = 0) := by
sorry

end symmetric_line_equation_l248_248822


namespace profit_percentage_l248_248705

variable {C S : ℝ}

theorem profit_percentage (h : 19 * C = 16 * S) :
  ((S - C) / C) * 100 = 18.75 := by
  sorry

end profit_percentage_l248_248705


namespace parallel_planes_x_plus_y_l248_248185

def planes_parallel (x y : ℝ) : Prop :=
  ∃ k : ℝ, (x = -k) ∧ (1 = k * y) ∧ (-2 = (1 / 2) * k)

theorem parallel_planes_x_plus_y (x y : ℝ) (h : planes_parallel x y) : x + y = 15 / 4 :=
sorry

end parallel_planes_x_plus_y_l248_248185


namespace january_31_is_friday_l248_248281

theorem january_31_is_friday (h : ∀ (d : ℕ), (d % 7 = 0 → d = 1)) : ∀ d, (d = 31) → (d % 7 = 3) :=
by
  sorry

end january_31_is_friday_l248_248281


namespace total_seeds_gray_sections_combined_l248_248787

noncomputable def total_seeds_first_circle : ℕ := 87
noncomputable def seeds_white_first_circle : ℕ := 68
noncomputable def total_seeds_second_circle : ℕ := 110
noncomputable def seeds_white_second_circle : ℕ := 68

theorem total_seeds_gray_sections_combined :
  (total_seeds_first_circle - seeds_white_first_circle) +
  (total_seeds_second_circle - seeds_white_second_circle) = 61 :=
by
  sorry

end total_seeds_gray_sections_combined_l248_248787


namespace r_and_s_earns_per_day_l248_248706

variable (P Q R S : Real)

-- Conditions as given in the problem
axiom cond1 : P + Q + R + S = 2380 / 9
axiom cond2 : P + R = 600 / 5
axiom cond3 : Q + S = 800 / 6
axiom cond4 : Q + R = 910 / 7
axiom cond5 : P = 150 / 3

theorem r_and_s_earns_per_day : R + S = 143.33 := by
  sorry

end r_and_s_earns_per_day_l248_248706


namespace total_cost_of_items_l248_248233

variable (M R F : ℝ)
variable (h1 : 10 * M = 24 * R)
variable (h2 : F = 2 * R)
variable (h3 : F = 21)

theorem total_cost_of_items : 4 * M + 3 * R + 5 * F = 237.3 :=
by
  sorry

end total_cost_of_items_l248_248233


namespace olivia_possible_amount_l248_248044

theorem olivia_possible_amount (k : ℕ) :
  ∃ k : ℕ, 1 + 79 * k = 1984 :=
by
  -- Prove that there exists a non-negative integer k such that the equation holds
  sorry

end olivia_possible_amount_l248_248044


namespace possible_values_for_D_l248_248211

noncomputable def distinct_digit_values (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  B < 10 ∧ A < 10 ∧ D < 10 ∧ C < 10 ∧ C = 9 ∧ (B + A = 9 + D)

theorem possible_values_for_D :
  ∃ (Ds : Finset Nat), (∀ D ∈ Ds, ∃ A B C, distinct_digit_values A B C D) ∧
  Ds.card = 5 :=
sorry

end possible_values_for_D_l248_248211


namespace add_fractions_result_l248_248418

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248418


namespace fraction_addition_l248_248537

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248537


namespace tan_sum_identity_l248_248881

theorem tan_sum_identity (a b : ℝ) (h₁ : Real.tan a = 1/2) (h₂ : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_identity_l248_248881


namespace line_equation_l248_248160

theorem line_equation 
    (passes_through_intersection : ∃ (P : ℝ × ℝ), P ∈ { (x, y) | 11 * x + 3 * y - 7 = 0 } ∧ P ∈ { (x, y) | 12 * x + y - 19 = 0 })
    (equidistant_from_A_and_B : ∃ (P : ℝ × ℝ), dist P (3, -2) = dist P (-1, 6)) :
    ∃ (a b c : ℝ), (a = 7 ∧ b = 1 ∧ c = -9) ∨ (a = 2 ∧ b = 1 ∧ c = 1) ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
sorry

end line_equation_l248_248160


namespace cos_plus_sin_l248_248180

open Real

theorem cos_plus_sin (α : ℝ) (h₁ : tan α = -2) (h₂ : (π / 2) < α ∧ α < π) : 
  cos α + sin α = (sqrt 5) / 5 :=
sorry

end cos_plus_sin_l248_248180


namespace Stuart_reward_points_l248_248268

theorem Stuart_reward_points (reward_points_per_unit : ℝ) (spending : ℝ) (unit_amount : ℝ) : 
  reward_points_per_unit = 5 → 
  spending = 200 → 
  unit_amount = 25 → 
  (spending / unit_amount) * reward_points_per_unit = 40 :=
by 
  intros h_points h_spending h_unit
  sorry

end Stuart_reward_points_l248_248268


namespace g_of_1_equals_3_l248_248017

theorem g_of_1_equals_3 (f g : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hg_even : ∀ x, g (-x) = g x)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 :=
sorry

end g_of_1_equals_3_l248_248017


namespace sqrt_of_4_l248_248241

theorem sqrt_of_4 :
  {x | x * x = 4} = {2, -2} :=
sorry

end sqrt_of_4_l248_248241


namespace problem_solution_l248_248029

variable (a b : ℝ)

theorem problem_solution (h : 2 * a - 3 * b = 5) : 4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 :=
sorry

end problem_solution_l248_248029


namespace find_x_collinear_l248_248309

theorem find_x_collinear (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, 1)) (h_b : b = (x, -1)) 
  (h_collinear : ∃ k : ℝ, (a.1 - b.1, a.2 - b.2) = (k * b.1, k * b.2)) : x = -2 :=
by 
  -- the proof would go here
  sorry

end find_x_collinear_l248_248309


namespace add_fractions_l248_248515

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248515


namespace arithmetic_sequence_y_l248_248103

theorem arithmetic_sequence_y :
  let a := 3^3
  let c := 3^5
  let y := (a + c) / 2
  y = 135 :=
by
  let a := 27
  let c := 243
  let y := (a + c) / 2
  show y = 135
  sorry

end arithmetic_sequence_y_l248_248103


namespace find_min_value_l248_248752

theorem find_min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / a + 1 / b = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end find_min_value_l248_248752


namespace find_m_value_l248_248315

noncomputable def is_direct_proportion_function (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x

theorem find_m_value (m : ℝ) (hk : ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x) : m = -1 :=
by
  sorry

end find_m_value_l248_248315


namespace add_fractions_result_l248_248416

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248416


namespace find_a_plus_2b_l248_248766

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x + b

noncomputable def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem find_a_plus_2b (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f a b 2 = 9) : a + 2 * b = -24 := 
by sorry

end find_a_plus_2b_l248_248766


namespace sum_of_squares_l248_248689

theorem sum_of_squares (a b c : ℝ)
  (h1 : a + b + c = 19)
  (h2 : a * b + b * c + c * a = 131) :
  a^2 + b^2 + c^2 = 99 :=
by
  sorry

end sum_of_squares_l248_248689


namespace fraction_addition_l248_248452

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248452


namespace fraction_addition_l248_248541

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248541


namespace fraction_addition_l248_248454

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248454


namespace sneakers_cost_l248_248916

theorem sneakers_cost (rate_per_yard : ℝ) (num_yards_cut : ℕ) (total_earnings : ℝ) :
  rate_per_yard = 2.15 ∧ num_yards_cut = 6 ∧ total_earnings = rate_per_yard * num_yards_cut → 
  total_earnings = 12.90 :=
by
  sorry

end sneakers_cost_l248_248916


namespace equal_areas_greater_perimeter_l248_248725

noncomputable def side_length_square := Real.sqrt 3 + 3

noncomputable def length_rectangle := Real.sqrt 72 + 3 * Real.sqrt 6
noncomputable def width_rectangle := Real.sqrt 2

noncomputable def area_square := (side_length_square) ^ 2

noncomputable def area_rectangle := length_rectangle * width_rectangle

noncomputable def perimeter_square := 4 * side_length_square

noncomputable def perimeter_rectangle := 2 * (length_rectangle + width_rectangle)

theorem equal_areas : area_square = area_rectangle := sorry

theorem greater_perimeter : perimeter_square < perimeter_rectangle := sorry

end equal_areas_greater_perimeter_l248_248725


namespace add_fractions_l248_248496

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248496


namespace solve_for_a_l248_248186

theorem solve_for_a (a x : ℝ) (h₁ : 2 * x - 3 = 5 * x - 2 * a) (h₂ : x = 1) : a = 3 :=
by
  sorry

end solve_for_a_l248_248186


namespace y_value_l248_248771

-- Given conditions
variables (x y : ℝ)
axiom h1 : x - y = 20
axiom h2 : x + y = 14

-- Prove that y = -3
theorem y_value : y = -3 :=
by { sorry }

end y_value_l248_248771


namespace people_at_the_beach_l248_248664

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l248_248664


namespace max_value_expression_l248_248034

theorem max_value_expression  
    (x y : ℝ) 
    (h : 2 * x^2 + y^2 = 6 * x) : 
    x^2 + y^2 + 2 * x ≤ 15 :=
sorry

end max_value_expression_l248_248034


namespace parabola_vertex_position_l248_248591

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem parabola_vertex_position (x y : ℝ) :
  (∃ a b : ℝ, f a = y ∧ g b = y ∧ a = 1 ∧ b = -1)
  → (1 > -1) ∧ (f 1 > g (-1)) :=
by
  sorry

end parabola_vertex_position_l248_248591


namespace simplify_expression_l248_248314

variable (x y z : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hne : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := 
by 
  sorry

end simplify_expression_l248_248314


namespace billy_watches_videos_l248_248407

-- Conditions definitions
def num_suggestions_per_list : Nat := 15
def num_iterations : Nat := 5
def pick_index_on_final_list : Nat := 5

-- Main theorem statement
theorem billy_watches_videos : 
  num_suggestions_per_list * num_iterations + (pick_index_on_final_list - 1) = 79 :=
by
  sorry

end billy_watches_videos_l248_248407


namespace find_d_l248_248826

open Real

-- Define the given conditions
variable (a b c d e : ℝ)

axiom cond1 : 3 * (a^2 + b^2 + c^2) + 4 = 2 * d + sqrt (a + b + c - d + e)
axiom cond2 : e = 1

-- Define the theorem stating that d = 7/4 under the given conditions
theorem find_d : d = 7/4 := by
  sorry

end find_d_l248_248826


namespace find_k_l248_248627

-- Definitions of given vectors and the condition that the vectors are parallel.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Condition for vectors to be parallel in 2D is that their cross product is zero.
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_k : ∀ k : ℝ, parallel vector_a (vector_b k) → k = -2 :=
by
  intro k
  intro h
  sorry

end find_k_l248_248627


namespace simplify_expr1_simplify_expr2_l248_248589

-- Definition for the expression (2x - 3y)²
def expr1 (x y : ℝ) : ℝ := (2 * x - 3 * y) ^ 2

-- Theorem to prove that (2x - 3y)² = 4x² - 12xy + 9y²
theorem simplify_expr1 (x y : ℝ) : expr1 x y = 4 * (x ^ 2) - 12 * x * y + 9 * (y ^ 2) := 
sorry

-- Definition for the expression (x + y) * (x + y) * (x² + y²)
def expr2 (x y : ℝ) : ℝ := (x + y) * (x + y) * (x ^ 2 + y ^ 2)

-- Theorem to prove that (x + y) * (x + y) * (x² + y²) = x⁴ + 2x²y² + y⁴ + 2x³y + 2xy³
theorem simplify_expr2 (x y : ℝ) : expr2 x y = x ^ 4 + 2 * (x ^ 2) * (y ^ 2) + y ^ 4 + 2 * (x ^ 3) * y + 2 * x * (y ^ 3) := 
sorry

end simplify_expr1_simplify_expr2_l248_248589


namespace presidency_meeting_ways_l248_248389

theorem presidency_meeting_ways :
  let schools := 4
  let members_per_school := 5
  let host_reps := 3
  let other_reps := 2
  let ways_choose_host := Nat.choose schools 1
  let ways_choose_host_reps := Nat.choose members_per_school host_reps
  let ways_choose_other_reps := Nat.choose members_per_school other_reps
  ways_choose_host * ways_choose_host_reps * ways_choose_other_reps^ (schools - 1) = 40000 :=
by
  sorry

end presidency_meeting_ways_l248_248389


namespace fraction_addition_l248_248533

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248533


namespace remainder_when_dividing_P_by_DDD_l248_248797

variables (P D D' D'' Q Q' Q'' R R' R'' : ℕ)

-- Define the conditions
def condition1 : Prop := P = Q * D + R
def condition2 : Prop := Q = Q' * D' + R'
def condition3 : Prop := Q' = Q'' * D'' + R''

-- Theorem statement asserting the given conclusion
theorem remainder_when_dividing_P_by_DDD' 
  (H1 : condition1 P D Q R)
  (H2 : condition2 Q D' Q' R')
  (H3 : condition3 Q' D'' Q'' R'') : 
  P % (D * D' * D') = R'' * D * D' + R * D' + R := 
sorry

end remainder_when_dividing_P_by_DDD_l248_248797


namespace students_without_an_A_l248_248321

theorem students_without_an_A :
  ∀ (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (computing_A : ℕ)
    (math_and_history_A : ℕ) (history_and_computing_A : ℕ)
    (math_and_computing_A : ℕ) (all_three_A : ℕ),
  total_students = 40 →
  history_A = 10 →
  math_A = 18 →
  computing_A = 9 →
  math_and_history_A = 5 →
  history_and_computing_A = 3 →
  math_and_computing_A = 4 →
  all_three_A = 2 →
  total_students - (history_A + math_A + computing_A - math_and_history_A - history_and_computing_A - math_and_computing_A + all_three_A) = 13 :=
by
  intros total_students history_A math_A computing_A math_and_history_A history_and_computing_A math_and_computing_A all_three_A 
         ht_total_students ht_history_A ht_math_A ht_computing_A ht_math_and_history_A ht_history_and_computing_A ht_math_and_computing_A ht_all_three_A
  sorry

end students_without_an_A_l248_248321


namespace box_volume_is_correct_l248_248130

noncomputable def box_volume (length width cut_side : ℝ) : ℝ :=
  (length - 2 * cut_side) * (width - 2 * cut_side) * cut_side

theorem box_volume_is_correct : box_volume 48 36 5 = 9880 := by
  sorry

end box_volume_is_correct_l248_248130


namespace age_difference_l248_248121

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 := 
sorry

end age_difference_l248_248121


namespace find_value_l248_248012

theorem find_value 
    (x y : ℝ) 
    (hx : x = 1 / (Real.sqrt 2 + 1)) 
    (hy : y = 1 / (Real.sqrt 2 - 1)) : 
    x^2 - 3 * x * y + y^2 = 3 := 
by 
    sorry

end find_value_l248_248012


namespace add_fractions_l248_248485

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248485


namespace add_fractions_l248_248495

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248495


namespace minimum_value_of_f_roots_sum_gt_2_l248_248764

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f 1 = 1 := by
  exists 1
  sorry

theorem roots_sum_gt_2 (a x₁ x₂ : ℝ) (h_f_x₁ : f x₁ = a) (h_f_x₂ : f x₂ = a) (h_x₁_lt_x₂ : x₁ < x₂) :
    x₁ + x₂ > 2 := by
  sorry

end minimum_value_of_f_roots_sum_gt_2_l248_248764


namespace symmetric_points_y_axis_l248_248633

theorem symmetric_points_y_axis (a b : ℝ) (h1 : a - b = -3) (h2 : 2 * a + b = 2) :
  a = -1 / 3 ∧ b = 8 / 3 :=
by
  sorry

end symmetric_points_y_axis_l248_248633


namespace hiker_miles_l248_248196

-- Defining the conditions as a def
def total_steps (flips : ℕ) (additional_steps : ℕ) : ℕ := flips * 100000 + additional_steps

def steps_per_mile : ℕ := 1500

-- The target theorem to prove the number of miles walked
theorem hiker_miles (flips : ℕ) (additional_steps : ℕ) (s_per_mile : ℕ) 
  (h_flips : flips = 72) (h_additional_steps : additional_steps = 25370) 
  (h_s_per_mile : s_per_mile = 1500) : 
  (total_steps flips additional_steps) / s_per_mile = 4817 :=
by
  -- sorry is used to skip the actual proof
  sorry

end hiker_miles_l248_248196


namespace add_fractions_result_l248_248419

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248419


namespace cost_flying_X_to_Y_l248_248357

def distance_XY : ℝ := 4500 -- Distance from X to Y in km
def cost_per_km_flying : ℝ := 0.12 -- Cost per km for flying in dollars
def booking_fee_flying : ℝ := 120 -- Booking fee for flying in dollars

theorem cost_flying_X_to_Y : 
    distance_XY * cost_per_km_flying + booking_fee_flying = 660 := by
  sorry

end cost_flying_X_to_Y_l248_248357


namespace average_rate_640_miles_trip_l248_248350

theorem average_rate_640_miles_trip 
  (total_distance : ℕ) 
  (first_half_distance : ℕ) 
  (first_half_rate : ℕ) 
  (second_half_time_multiplier : ℕ) 
  (first_half_time : ℕ := first_half_distance / first_half_rate)
  (second_half_time : ℕ := second_half_time_multiplier * first_half_time)
  (total_time : ℕ := first_half_time + second_half_time)
  (average_rate : ℕ := total_distance / total_time) : 
  total_distance = 640 ∧ 
  first_half_distance = 320 ∧ 
  first_half_rate = 80 ∧ 
  second_half_time_multiplier = 3 → 
  average_rate = 40 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  rw [h1, h2, h3, h4] at *
  have h5 : first_half_time = 320 / 80 := rfl
  have h6 : second_half_time = 3 * (320 / 80) := rfl
  have h7 : total_time = (320 / 80) + 3 * (320 / 80) := rfl
  have h8 : average_rate = 640 / (4 + 12) := rfl
  have h9 : average_rate = 640 / 16 := rfl
  have average_rate_correct : average_rate = 40 := rfl
  exact average_rate_correct

end average_rate_640_miles_trip_l248_248350


namespace ordered_pairs_count_l248_248772

theorem ordered_pairs_count : 
  (∃ s : Finset (ℕ × ℕ), (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) ∧ s.card = 15) :=
by
  -- The proof would go here
  sorry

end ordered_pairs_count_l248_248772


namespace no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l248_248838

theorem no_integer_for_58th_power_64_digits : ¬ ∃ n : ℤ, 10^63 ≤ n^58 ∧ n^58 < 10^64 :=
sorry

theorem valid_replacement_for_64_digits (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 81) : 
  ¬ ∃ n : ℤ, 10^(k-1) ≤ n^58 ∧ n^58 < 10^k :=
sorry

end no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l248_248838


namespace simplify_product_of_fractions_l248_248226

theorem simplify_product_of_fractions :
  8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end simplify_product_of_fractions_l248_248226


namespace choose_photographers_l248_248788

theorem choose_photographers (n k : ℕ) (h_n : n = 10) (h_k : k = 3) : Nat.choose n k = 120 := by
  -- The proof is omitted
  sorry

end choose_photographers_l248_248788


namespace compare_expressions_l248_248295

theorem compare_expressions (a b : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) :=
by {
  sorry
}

end compare_expressions_l248_248295


namespace fraction_addition_l248_248529

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248529


namespace add_fractions_l248_248583

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248583


namespace celsius_equals_fahrenheit_l248_248353

-- Define the temperature scales.
def celsius_to_fahrenheit (T_C : ℝ) : ℝ := 1.8 * T_C + 32

-- The Lean statement for the problem.
theorem celsius_equals_fahrenheit : ∃ (T : ℝ), T = celsius_to_fahrenheit T ↔ T = -40 :=
by
  sorry -- Proof is not required, just the statement.

end celsius_equals_fahrenheit_l248_248353


namespace number_of_liars_on_island_l248_248091

theorem number_of_liars_on_island :
  ∃ (knights liars : ℕ),
  let Total := 1000 in
  let Villages := 10 in
  let members_per_village := Total / Villages in
  -- Each village has at least 2 members
  (∀ i : ℕ, i < Villages → 2 ≤ members_per_village) ∧
  -- Populations of knights and liars respecting the Total population
  (knights + liars = Total) ∧
  -- Each inhabitant claims all others in their village are liars
  (∀ (i v : ℕ), i < Villages → v < members_per_village → 
     (∃ k l : ℕ, k + l = members_per_village ∧ 
      k = 1 ∧ -- there is exactly one knight in each village
      knights = Villages ∧ liars = Total - knights ∧
      l = members_per_village - 1)) ∧
  liars = 990 := sorry

end number_of_liars_on_island_l248_248091


namespace point_reflection_l248_248326

-- Definition of point and reflection over x-axis
def P : ℝ × ℝ := (-2, 3)

def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Statement to prove
theorem point_reflection : reflect_x_axis P = (-2, -3) :=
by
  -- Proof goes here
  sorry

end point_reflection_l248_248326


namespace additional_people_required_l248_248879

-- Given condition: Four people can mow a lawn in 6 hours
def work_rate: ℕ := 4 * 6

-- New condition: Number of people needed to mow the lawn in 3 hours
def people_required_in_3_hours: ℕ := work_rate / 3

-- Statement: Number of additional people required
theorem additional_people_required : people_required_in_3_hours - 4 = 4 :=
by
  -- Proof would go here
  sorry

end additional_people_required_l248_248879


namespace two_digit_number_tens_place_l248_248990

theorem two_digit_number_tens_place (x y : Nat) (hx1 : 0 ≤ x) (hx2 : x ≤ 9) (hy1 : 0 ≤ y) (hy2 : y ≤ 9)
    (h : (x + y) * 3 = 10 * x + y - 2) : x = 2 := 
sorry

end two_digit_number_tens_place_l248_248990


namespace min_bounces_for_height_less_than_two_l248_248387

theorem min_bounces_for_height_less_than_two : 
  ∃ (k : ℕ), (20 * (3 / 4 : ℝ)^k < 2 ∧ ∀ n < k, ¬(20 * (3 / 4 : ℝ)^n < 2)) :=
sorry

end min_bounces_for_height_less_than_two_l248_248387


namespace sum_fractions_eq_l248_248479

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248479


namespace add_fractions_result_l248_248420

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248420


namespace gnomes_cannot_cross_l248_248067

theorem gnomes_cannot_cross :
  ∀ (gnomes : List ℕ), 
    (∀ g, g ∈ gnomes → g ∈ (List.range 100).map (λ x => x + 1)) →
    List.sum gnomes = 5050 → 
    ∀ (boat_capacity : ℕ), boat_capacity = 100 →
    ∀ (k : ℕ), (200 * (k + 1) - k^2 = 10100) → false :=
by
  intros gnomes H_weights H_sum boat_capacity H_capacity k H_equation
  sorry

end gnomes_cannot_cross_l248_248067


namespace fraction_addition_l248_248528

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248528


namespace vessel_base_length_l248_248726

noncomputable def volume_of_cube (side: ℝ) : ℝ :=
  side ^ 3

noncomputable def volume_displaced (length breadth height: ℝ) : ℝ :=
  length * breadth * height

theorem vessel_base_length
  (breadth : ℝ) 
  (cube_edge : ℝ)
  (water_rise : ℝ)
  (displaced_volume : ℝ) 
  (h1 : breadth = 30) 
  (h2 : cube_edge = 30) 
  (h3 : water_rise = 15) 
  (h4 : volume_of_cube cube_edge = displaced_volume) :
  volume_displaced (displaced_volume / (breadth * water_rise)) breadth water_rise = displaced_volume :=
  by
  sorry

end vessel_base_length_l248_248726


namespace train_speed_l248_248845

/-- 
A man sitting in a train which is traveling at a certain speed observes 
that a goods train, traveling in the opposite direction, takes 9 seconds 
to pass him. The goods train is 280 m long and its speed is 52 kmph. 
Prove that the speed of the train the man is sitting in is 60 kmph.
-/
theorem train_speed (t : ℝ) (h1 : 0 < t)
  (goods_speed_kmph : ℝ := 52)
  (goods_length_m : ℝ := 280)
  (time_seconds : ℝ := 9)
  (h2 : goods_length_m / time_seconds = (t + goods_speed_kmph) * (5 / 18)) :
  t = 60 :=
sorry

end train_speed_l248_248845


namespace percentage_hindus_l248_248322

-- Conditions 
def total_boys : ℕ := 850
def percentage_muslims : ℝ := 0.44
def percentage_sikhs : ℝ := 0.10
def boys_other_communities : ℕ := 272

-- Question and proof statement
theorem percentage_hindus (total_boys : ℕ) (percentage_muslims percentage_sikhs : ℝ) (boys_other_communities : ℕ) : 
  (total_boys = 850) →
  (percentage_muslims = 0.44) →
  (percentage_sikhs = 0.10) →
  (boys_other_communities = 272) →
  ((850 - (374 + 85 + 272)) / 850) * 100 = 14 := 
by
  intros
  sorry

end percentage_hindus_l248_248322


namespace profit_difference_is_50_l248_248715

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end profit_difference_is_50_l248_248715


namespace add_fractions_l248_248521

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248521


namespace sin_six_theta_l248_248770

theorem sin_six_theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (6 * θ) = - (630 * Real.sqrt 8) / 15625 := by
  sorry

end sin_six_theta_l248_248770


namespace min_value_of_expression_l248_248883

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  (1 / x + 4 / y) ≥ 9 :=
by
  sorry

end min_value_of_expression_l248_248883


namespace add_fractions_l248_248497

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248497


namespace parallelepiped_length_l248_248993

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248993


namespace buckets_oranges_l248_248093

theorem buckets_oranges :
  ∀ (a b c : ℕ), 
  a = 22 → 
  b = a + 17 → 
  a + b + c = 89 → 
  b - c = 11 := 
by 
  intros a b c h1 h2 h3 
  sorry

end buckets_oranges_l248_248093


namespace arithmetic_sequence_S7_geometric_sequence_k_l248_248177

noncomputable def S_n (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_S7 (a_4 : ℕ) (h : a_4 = 8) : S_n a_4 1 7 = 56 := by
  sorry

def Sn_formula (k : ℕ) : ℕ := k^2 + k
def a (i d : ℕ) := i * d

theorem geometric_sequence_k (a_1 k : ℕ) (h1 : a_1 = 2) (h2 : (2 * k + 2)^2 = 6 * (k^2 + k)) :
  k = 2 := by
  sorry

end arithmetic_sequence_S7_geometric_sequence_k_l248_248177


namespace expansion_coeff_sum_l248_248735

theorem expansion_coeff_sum
  (a : ℕ → ℤ)
  (h : ∀ x y : ℤ, (x - 2 * y) ^ 5 * (x + 3 * y) ^ 4 = 
    a 9 * x ^ 9 + 
    a 8 * x ^ 8 * y + 
    a 7 * x ^ 7 * y ^ 2 + 
    a 6 * x ^ 6 * y ^ 3 + 
    a 5 * x ^ 5 * y ^ 4 + 
    a 4 * x ^ 4 * y ^ 5 + 
    a 3 * x ^ 3 * y ^ 6 + 
    a 2 * x ^ 2 * y ^ 7 + 
    a 1 * x * y ^ 8 + 
    a 0 * y ^ 9) :
  a 0 + a 8 = -2602 := by
  sorry

end expansion_coeff_sum_l248_248735


namespace max_students_l248_248095

open Nat

theorem max_students (B G : ℕ) (h1 : 11 * B = 7 * G) (h2 : G = B + 72) (h3 : B + G ≤ 550) : B + G = 324 := by
  sorry

end max_students_l248_248095


namespace money_left_correct_l248_248939

def initial_amount : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def amount_left : ℕ := initial_amount - total_spent

theorem money_left_correct :
  amount_left = 78 := by
  sorry

end money_left_correct_l248_248939


namespace find_a4b4_l248_248820

theorem find_a4b4 
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) :
  a4 * b4 = -6 :=
sorry

end find_a4b4_l248_248820


namespace add_fractions_l248_248491

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l248_248491


namespace increase_in_cost_l248_248843

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def initial_total_cost : ℝ := initial_lumber_cost + initial_nails_cost + initial_fabric_cost

def new_lumber_cost : ℝ := initial_lumber_cost * (1 + lumber_inflation_rate)
def new_nails_cost : ℝ := initial_nails_cost * (1 + nails_inflation_rate)
def new_fabric_cost : ℝ := initial_fabric_cost * (1 + fabric_inflation_rate)

def new_total_cost : ℝ := new_lumber_cost + new_nails_cost + new_fabric_cost

theorem increase_in_cost :
  new_total_cost - initial_total_cost = 97 := 
sorry

end increase_in_cost_l248_248843


namespace sum_of_coefficients_of_factorized_polynomial_l248_248074

theorem sum_of_coefficients_of_factorized_polynomial : 
  ∃ (a b c d e : ℕ), 
    (216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 36) :=
sorry

end sum_of_coefficients_of_factorized_polynomial_l248_248074


namespace find_rectangle_pairs_l248_248651

theorem find_rectangle_pairs (w l : ℕ) (hw : w > 0) (hl : l > 0) (h : w * l = 18) : 
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) ∨
  (w, l) = (6, 3) ∨ (w, l) = (9, 2) ∨ (w, l) = (18, 1) :=
by
  sorry

end find_rectangle_pairs_l248_248651


namespace quotient_base4_correct_l248_248003

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 1302 => 1 * 4^3 + 3 * 4^2 + 0 * 4^1 + 2 * 4^0
  | 12 => 1 * 4^1 + 2 * 4^0
  | _ => 0

def base10_to_base4 (n : ℕ) : ℕ :=
  match n with
  | 19 => 1 * 4^2 + 0 * 4^1 + 3 * 4^0
  | _ => 0

theorem quotient_base4_correct : base10_to_base4 (114 / 6) = 103 := 
  by sorry

end quotient_base4_correct_l248_248003


namespace least_k_inequality_l248_248279

theorem least_k_inequality :
  ∃ k : ℝ, (∀ a b c : ℝ, 
    ((2 * a / (a - b)) ^ 2 + (2 * b / (b - c)) ^ 2 + (2 * c / (c - a)) ^ 2 + k 
    ≥ 4 * (2 * a / (a - b) + 2 * b / (b - c) + 2 * c / (c - a)))) ∧ k = 8 :=
by
  sorry  -- proof is omitted

end least_k_inequality_l248_248279


namespace football_team_selection_l248_248355

theorem football_team_selection :
  let team_members : ℕ := 12
  let offensive_lineman_choices : ℕ := 4
  let tight_end_choices : ℕ := 2
  let players_left_after_offensive : ℕ := team_members - 1
  let players_left_after_tightend : ℕ := players_left_after_offensive - 1
  let quarterback_choices : ℕ := players_left_after_tightend
  let players_left_after_quarterback : ℕ := quarterback_choices - 1
  let running_back_choices : ℕ := players_left_after_quarterback
  let players_left_after_runningback : ℕ := running_back_choices - 1
  let wide_receiver_choices : ℕ := players_left_after_runningback
  offensive_lineman_choices * tight_end_choices * 
  quarterback_choices * running_back_choices * 
  wide_receiver_choices = 5760 := 
by 
  sorry

end football_team_selection_l248_248355


namespace total_pitches_missed_l248_248925

theorem total_pitches_missed (tokens_to_pitches : ℕ → ℕ) 
  (macy_used : ℕ) (piper_used : ℕ) 
  (macy_hits : ℕ) (piper_hits : ℕ) 
  (h1 : tokens_to_pitches 1 = 15) 
  (h_macy_used : macy_used = 11) 
  (h_piper_used : piper_used = 17) 
  (h_macy_hits : macy_hits = 50) 
  (h_piper_hits : piper_hits = 55) :
  let total_pitches := tokens_to_pitches macy_used + tokens_to_pitches piper_used
  let total_hits := macy_hits + piper_hits
  total_pitches - total_hits = 315 :=
by
  sorry

end total_pitches_missed_l248_248925


namespace cuberoot_eq_3_implies_cube_eq_19683_l248_248199

theorem cuberoot_eq_3_implies_cube_eq_19683 (x : ℝ) (h : (x + 6)^(1/3) = 3) : (x + 6)^3 = 19683 := by
  sorry

end cuberoot_eq_3_implies_cube_eq_19683_l248_248199


namespace domain_of_transformed_function_l248_248364

theorem domain_of_transformed_function (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → True) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → True :=
sorry

end domain_of_transformed_function_l248_248364


namespace fraction_addition_l248_248466

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248466


namespace probability_top_card_is_5_or_king_l248_248989

theorem probability_top_card_is_5_or_king :
  let total_cards := 52
  let count_fives := 4
  let count_kings := 4
  let total_desirable := count_fives + count_kings
  let probability := total_desirable / total_cards
  probability = 2 / 13 :=
by
  let total_cards := 52
  let count_fives := 4
  let count_kings := 4
  let total_desirable := count_fives + count_kings
  let probability := total_desirable / total_cards
  show probability = 2 / 13
  sorry

end probability_top_card_is_5_or_king_l248_248989


namespace large_circle_diameter_l248_248069

theorem large_circle_diameter (r : ℝ) (R : ℝ) (R' : ℝ) :
  r = 2 ∧ R = 2 * r ∧ R' = R + r → 2 * R' = 12 :=
by
  intros h
  sorry

end large_circle_diameter_l248_248069


namespace sum_of_fractions_l248_248423

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l248_248423


namespace only_solution_l248_248009

theorem only_solution (a : ℤ) : 
  (∀ x : ℤ, x > 0 → 2 * x > 4 * x - 8 → 3 * x - a > -9 → x = 2) →
  (12 ≤ a ∧ a < 15) :=
by
  sorry

end only_solution_l248_248009


namespace add_fractions_l248_248516

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248516


namespace sum_of_digits_l248_248630

theorem sum_of_digits (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h3 : 34 * a + 42 * b = 142) : a + b = 4 := 
by
  sorry

end sum_of_digits_l248_248630


namespace add_fractions_l248_248520

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248520


namespace intersection_count_is_one_l248_248592

theorem intersection_count_is_one :
  (∀ x y : ℝ, y = 2 * x^3 + 6 * x + 1 → y = -3 / x^2) → ∃! p : ℝ × ℝ, p.2 = 2 * p.1^3 + 6 * p.1 + 1 ∧ p.2 = -3 / p.1 :=
sorry

end intersection_count_is_one_l248_248592


namespace tyler_meals_l248_248101

-- Define the types of items Tyler can choose from in terms of meats, vegetables, and desserts.
inductive Meat 
| beef | chicken | pork | turkey

inductive Vegetable 
| baked_beans | corn | potatoes | tomatoes | carrots

inductive Dessert 
| brownies | chocolate_cake | chocolate_pudding | ice_cream | cheesecake

-- Use the relevant combinatorial functions to calculate the number of ways to choose foods

open Nat

def num_meals : ℕ :=
  let meat_choices := choose 4 1 + choose 4 2 -- 4 choose 1 + 4 choose 2
  let veg_choices := choose 5 2 -- 5 choose 2
  let dessert_choices := choose 5 2 -- 5 choose 2
  meat_choices * veg_choices * dessert_choices

theorem tyler_meals : num_meals = 1000 :=
  by
  -- Calculation based on the combinatorial analysis from the problem statement.
  -- The proof will be done step by step to form the full calculation.
  sorry

end tyler_meals_l248_248101


namespace mechanic_worked_days_l248_248719

-- Definitions of conditions as variables
def hourly_rate : ℝ := 60
def hours_per_day : ℝ := 8
def cost_of_parts : ℝ := 2500
def total_amount_paid : ℝ := 9220

-- Definition to calculate the total labor cost
def total_labor_cost : ℝ := total_amount_paid - cost_of_parts

-- Definition to calculate the daily labor cost
def daily_labor_cost : ℝ := hourly_rate * hours_per_day

-- Proof (statement only) that the number of days the mechanic worked on the car is 14
theorem mechanic_worked_days : total_labor_cost / daily_labor_cost = 14 := by
  sorry

end mechanic_worked_days_l248_248719


namespace probability_adjacent_A_before_B_l248_248804

theorem probability_adjacent_A_before_B 
  (total_students : ℕ)
  (A B C D : ℚ)
  (hA : total_students = 8)
  (hB : B = 1/3) : 
  (∃ prob : ℚ, prob = 1/3) :=
by
  sorry

end probability_adjacent_A_before_B_l248_248804


namespace billy_watches_videos_l248_248406

-- Conditions definitions
def num_suggestions_per_list : Nat := 15
def num_iterations : Nat := 5
def pick_index_on_final_list : Nat := 5

-- Main theorem statement
theorem billy_watches_videos : 
  num_suggestions_per_list * num_iterations + (pick_index_on_final_list - 1) = 79 :=
by
  sorry

end billy_watches_videos_l248_248406


namespace trip_time_is_approximate_l248_248063

noncomputable def total_distance : ℝ := 620
noncomputable def half_distance : ℝ := total_distance / 2
noncomputable def speed1 : ℝ := 70
noncomputable def speed2 : ℝ := 85
noncomputable def time1 : ℝ := half_distance / speed1
noncomputable def time2 : ℝ := half_distance / speed2
noncomputable def total_time : ℝ := time1 + time2

theorem trip_time_is_approximate :
  abs (total_time - 8.0757) < 0.0001 :=
sorry

end trip_time_is_approximate_l248_248063


namespace widget_cost_reduction_l248_248223

theorem widget_cost_reduction:
  ∀ (C C_reduced : ℝ), 
  6 * C = 27.60 → 
  8 * C_reduced = 27.60 → 
  C - C_reduced = 1.15 := 
by
  intros C C_reduced h1 h2
  sorry

end widget_cost_reduction_l248_248223


namespace ratio_of_children_l248_248880

theorem ratio_of_children (C H : ℕ) 
  (hC1 : C / 8 = 16)
  (hC2 : C * (C / 8) = 512)
  (hH : H * 16 = 512) :
  H / C = 1 / 2 :=
by
  sorry

end ratio_of_children_l248_248880


namespace remaining_perimeter_l248_248265

-- Definitions based on conditions
noncomputable def GH : ℝ := 2
noncomputable def HI : ℝ := 2
noncomputable def GI : ℝ := Real.sqrt (GH^2 + HI^2)
noncomputable def side_JKL : ℝ := 5
noncomputable def JI : ℝ := side_JKL - GH
noncomputable def IK : ℝ := side_JKL - HI
noncomputable def JK : ℝ := side_JKL

-- Problem statement in Lean 4
theorem remaining_perimeter :
  JI + IK + JK = 11 :=
by
  sorry

end remaining_perimeter_l248_248265


namespace sam_pens_count_l248_248208

-- Lean 4 statement
theorem sam_pens_count :
  ∃ (black_pens blue_pens pencils red_pens : ℕ),
    (black_pens = blue_pens + 10) ∧
    (blue_pens = 2 * pencils) ∧
    (pencils = 8) ∧
    (red_pens = pencils - 2) ∧
    (black_pens + blue_pens + red_pens = 48) :=
by {
  sorry
}

end sam_pens_count_l248_248208


namespace people_at_the_beach_l248_248665

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l248_248665


namespace add_fractions_l248_248507

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248507


namespace sqrt_x_div_sqrt_y_l248_248282

theorem sqrt_x_div_sqrt_y (x y : ℝ)
  (h : ( ( (2/3)^2 + (1/6)^2 ) / ( (1/2)^2 + (1/7)^2 ) ) = 28 * x / (25 * y)) :
  (Real.sqrt x) / (Real.sqrt y) = 5 / 2 :=
sorry

end sqrt_x_div_sqrt_y_l248_248282


namespace annual_increase_rate_l248_248146

theorem annual_increase_rate (r : ℝ) (h : 70400 * (1 + r)^2 = 89100) : r = 0.125 :=
sorry

end annual_increase_rate_l248_248146


namespace money_difference_l248_248401

-- Given conditions
def packs_per_hour_peak : Nat := 6
def packs_per_hour_low : Nat := 4
def price_per_pack : Nat := 60
def hours_per_day : Nat := 15

-- Calculate total sales in peak and low seasons
def total_sales_peak : Nat :=
  packs_per_hour_peak * price_per_pack * hours_per_day

def total_sales_low : Nat :=
  packs_per_hour_low * price_per_pack * hours_per_day

-- The Lean statement proving the correct answer
theorem money_difference :
  total_sales_peak - total_sales_low = 1800 :=
by
  sorry

end money_difference_l248_248401


namespace max_sum_of_factors_l248_248124

theorem max_sum_of_factors (h k : ℕ) (h_even : Even h) (prod_eq : h * k = 24) : h + k ≤ 14 :=
sorry

end max_sum_of_factors_l248_248124


namespace cost_per_serving_l248_248808

-- Define the costs
def pasta_cost : ℝ := 1.00
def sauce_cost : ℝ := 2.00
def meatball_cost : ℝ := 5.00

-- Define the number of servings
def servings : ℝ := 8.0

-- State the theorem
theorem cost_per_serving : (pasta_cost + sauce_cost + meatball_cost) / servings = 1.00 :=
by
  sorry

end cost_per_serving_l248_248808


namespace fraction_addition_l248_248457

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248457


namespace tan_addition_formula_15_30_l248_248867

-- Define tangent function for angles in degrees.
noncomputable def tanDeg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem for the given problem
theorem tan_addition_formula_15_30 :
  tanDeg 15 + tanDeg 30 + tanDeg 15 * tanDeg 30 = 1 :=
by
  -- Here we use the given conditions and properties in solution
  sorry

end tan_addition_formula_15_30_l248_248867


namespace sum_of_palindromic_primes_less_than_70_l248_248290

def is_prime (n : ℕ) : Prop := Nat.Prime n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem sum_of_palindromic_primes_less_than_70 :
  let palindromic_primes := [11, 13, 31, 37]
  (∀ p ∈ palindromic_primes, is_palindromic_prime p ∧ p < 70) →
  palindromic_primes.sum = 92 :=
by
  sorry

end sum_of_palindromic_primes_less_than_70_l248_248290


namespace Zack_kept_5_marbles_l248_248970

-- Define the initial number of marbles Zack had
def Zack_initial_marbles : ℕ := 65

-- Define the number of marbles each friend receives
def marbles_per_friend : ℕ := 20

-- Define the total number of friends
def friends : ℕ := 3

noncomputable def marbles_given_away : ℕ := friends * marbles_per_friend

-- Define the amount of marbles kept by Zack
noncomputable def marbles_kept_by_Zack : ℕ := Zack_initial_marbles - marbles_given_away

-- The theorem to prove
theorem Zack_kept_5_marbles : marbles_kept_by_Zack = 5 := by
  -- Proof skipped with sorry
  sorry

end Zack_kept_5_marbles_l248_248970


namespace pie_shop_revenue_l248_248720

noncomputable def revenue_day1 := 5 * 6 * 12 + 6 * 6 * 8 + 7 * 6 * 10
noncomputable def revenue_day2 := 6 * 6 * 15 + 7 * 6 * 10 + 8 * 6 * 14
noncomputable def revenue_day3 := 4 * 6 * 18 + 7 * 6 * 7 + 9 * 6 * 13
noncomputable def total_revenue := revenue_day1 + revenue_day2 + revenue_day3

theorem pie_shop_revenue : total_revenue = 4128 := by
  sorry

end pie_shop_revenue_l248_248720


namespace fraction_addition_l248_248453

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248453


namespace coffee_equals_milk_l248_248690

theorem coffee_equals_milk (S : ℝ) (h : 0 < S ∧ S < 1/2) :
  let initial_milk := 1 / 2
  let initial_coffee := 1 / 2
  let glass1_initial := initial_milk
  let glass2_initial := initial_coffee
  let glass2_after_first_transfer := glass2_initial + S
  let coffee_transferred_back := (S * initial_coffee) / (initial_coffee + S)
  let milk_transferred_back := (S^2) / (initial_coffee + S)
  let glass1_after_second_transfer := glass1_initial - S + milk_transferred_back
  let glass2_after_second_transfer := glass2_initial + S - coffee_transferred_back
  (glass1_initial - S + milk_transferred_back) = (glass2_initial + S - coffee_transferred_back) :=
sorry

end coffee_equals_milk_l248_248690


namespace min_value_ineq_l248_248194

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 4 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 4 → (1/a + 4/b) ≥ 9/4)

theorem min_value_ineq : problem_statement :=
by
  unfold problem_statement
  sorry

end min_value_ineq_l248_248194


namespace max_ratio_square_l248_248053

variables {a b c x y : ℝ}
-- Assume a, b, c are positive real numbers
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
-- Assume the order of a, b, c: a ≥ b ≥ c
variable (h_order : a ≥ b ∧ b ≥ c)
-- Define the system of equations
variable (h_system : a^2 + y^2 = c^2 + x^2 ∧ c^2 + x^2 = (a - x)^2 + (c - y)^2)
-- Assume the constraints on x and y
variable (h_constraints : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < c)

theorem max_ratio_square :
  ∃ (ρ : ℝ), ρ = (a / c) ∧ ρ^2 = 4 / 3 :=
sorry

end max_ratio_square_l248_248053


namespace fraction_addition_l248_248563

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248563


namespace abc_divides_sum_exp21_l248_248332

theorem abc_divides_sum_exp21
  (a b c : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ∣ b^4)
  (hbc : b ∣ c^4)
  (hca : c ∣ a^4)
  : abc ∣ (a + b + c)^21 :=
by
sorry

end abc_divides_sum_exp21_l248_248332


namespace trigonometric_values_l248_248302

variable (α : ℝ)

theorem trigonometric_values (h : Real.cos (3 * Real.pi + α) = 3 / 5) :
  Real.cos α = -3 / 5 ∧
  Real.cos (Real.pi + α) = 3 / 5 ∧
  Real.sin (3 * Real.pi / 2 - α) = -3 / 5 :=
by
  sorry

end trigonometric_values_l248_248302


namespace number_of_blocks_needed_l248_248328

-- Define the dimensions of the fort
def fort_length : ℕ := 20
def fort_width : ℕ := 15
def fort_height : ℕ := 8

-- Define the thickness of the walls and the floor
def wall_thickness : ℕ := 2
def floor_thickness : ℕ := 1

-- Define the original volume of the fort
def V_original : ℕ := fort_length * fort_width * fort_height

-- Define the interior dimensions of the fort considering the thickness of the walls and floor
def interior_length : ℕ := fort_length - 2 * wall_thickness
def interior_width : ℕ := fort_width - 2 * wall_thickness
def interior_height : ℕ := fort_height - floor_thickness

-- Define the volume of the interior space
def V_interior : ℕ := interior_length * interior_width * interior_height

-- Statement to prove: number of blocks needed equals 1168
theorem number_of_blocks_needed : V_original - V_interior = 1168 := 
by 
  sorry

end number_of_blocks_needed_l248_248328


namespace hyperbola_eccentricity_range_l248_248023

-- Definitions of hyperbola and distance condition
def hyperbola (x y a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def distance_condition (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola x y a b → (b * x + a * y - 2 * a * b) > a

-- The range of the eccentricity
theorem hyperbola_eccentricity_range (a b : ℝ) (h : hyperbola 0 1 a b) 
  (dist_cond : distance_condition a b) : 
  ∃ e : ℝ, e ≥ (2 * Real.sqrt 3 / 3) :=
sorry

end hyperbola_eccentricity_range_l248_248023


namespace problem_solution_l248_248274

theorem problem_solution :
  (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 :=
by sorry

end problem_solution_l248_248274


namespace point_above_line_range_l248_248325

theorem point_above_line_range (a : ℝ) :
  (2 * a - (-1) + 1 < 0) ↔ a < -1 :=
by
  sorry

end point_above_line_range_l248_248325


namespace larger_number_is_42_l248_248384

theorem larger_number_is_42 (x y : ℕ) (h1 : x + y = 77) (h2 : 5 * x = 6 * y) : x = 42 :=
by
  sorry

end larger_number_is_42_l248_248384


namespace no_solution_l248_248871

theorem no_solution : ∀ x : ℝ, ¬ (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 5 * x + 1) :=
by
  intro x
  -- Solve each part of the inequality
  have h1 : ¬ (3 * x + 2 < (x + 2)^2) ↔ x^2 + x + 2 ≤ 0 := by sorry
  have h2 : ¬ ((x + 2)^2 < 5 * x + 1) ↔ x^2 - x + 3 ≥ 0 := by sorry
  -- Combine the results
  exact sorry

end no_solution_l248_248871


namespace coordinates_of_P_l248_248760

theorem coordinates_of_P (a : ℝ) (h : 2 * a - 6 = 0) : (2 * a - 6, a + 1) = (0, 4) :=
by 
  have ha : a = 3 := by linarith
  rw [ha]
  sorry

end coordinates_of_P_l248_248760


namespace lock_code_digits_l248_248688

noncomputable def valid_lock_code_count (n : ℕ) : ℕ :=
  if n < 2 then 0
  else 4 * 4 * Nat.descFactorial 5 (n - 2)

theorem lock_code_digits : ∃ n : ℕ, valid_lock_code_count n = 240 :=
begin
  use 4,
  unfold valid_lock_code_count,
  norm_num,
  rw Nat.descFactorial_eq_factorial_div_factorial,
  norm_num,
end

end lock_code_digits_l248_248688


namespace add_fractions_l248_248504

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248504


namespace simplify_fraction_mul_l248_248943

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : 405 = 27 * a) (h2 : 1215 = 27 * b) (h3 : a / d = 1) (h4 : b / d = 3) : (a / d) * (27 : ℕ) = 9 :=
by
  sorry

end simplify_fraction_mul_l248_248943


namespace max_complete_bouquets_l248_248375

-- Definitions based on conditions
def total_roses := 20
def total_lilies := 15
def total_daisies := 10

def wilted_roses := 12
def wilted_lilies := 8
def wilted_daisies := 5

def roses_per_bouquet := 3
def lilies_per_bouquet := 2
def daisies_per_bouquet := 1

-- Calculation of remaining flowers
def remaining_roses := total_roses - wilted_roses
def remaining_lilies := total_lilies - wilted_lilies
def remaining_daisies := total_daisies - wilted_daisies

-- Proof statement
theorem max_complete_bouquets : 
  min
    (remaining_roses / roses_per_bouquet)
    (min (remaining_lilies / lilies_per_bouquet) (remaining_daisies / daisies_per_bouquet)) = 2 :=
by
  sorry

end max_complete_bouquets_l248_248375


namespace product_defect_rate_correct_l248_248230

-- Definitions for the defect rates of the stages
def defect_rate_stage1 : ℝ := 0.10
def defect_rate_stage2 : ℝ := 0.03

-- Definitions for the probability of passing each stage without defects
def pass_rate_stage1 : ℝ := 1 - defect_rate_stage1
def pass_rate_stage2 : ℝ := 1 - defect_rate_stage2

-- Definition for the overall probability of a product not being defective
def pass_rate_overall : ℝ := pass_rate_stage1 * pass_rate_stage2

-- Definition for the overall defect rate based on the above probabilities
def defect_rate_product : ℝ := 1 - pass_rate_overall

-- The theorem statement to be proved
theorem product_defect_rate_correct : defect_rate_product = 0.127 :=
by
  -- Proof here
  sorry

end product_defect_rate_correct_l248_248230


namespace positive_integer_base_conversion_l248_248679

theorem positive_integer_base_conversion (A B : ℕ) (h1 : A < 9) (h2 : B < 7) 
(h3 : 9 * A + B = 7 * B + A) : 9 * 3 + 4 = 31 :=
by sorry

end positive_integer_base_conversion_l248_248679


namespace tom_caught_16_trout_l248_248064

theorem tom_caught_16_trout (melanie_trout : ℕ) (tom_caught_twice : melanie_trout * 2 = 16) : 
  2 * melanie_trout = 16 :=
by 
  sorry

end tom_caught_16_trout_l248_248064


namespace sum_fractions_eq_l248_248470

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248470


namespace relationship_among_abc_l248_248882

noncomputable def a : ℝ := 2 ^ (3 / 2)
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 0.8 ^ 2

theorem relationship_among_abc : b < c ∧ c < a := 
by
  -- these are conditions directly derived from the problem
  let h1 : a = 2 ^ (3 / 2) := rfl
  let h2 : b = Real.log 0.3 / Real.log 2 := rfl
  let h3 : c = 0.8 ^ 2 := rfl
  sorry

end relationship_among_abc_l248_248882


namespace marissa_sunflower_height_l248_248061

def height_sister_in_inches : ℚ := 4 * 12 + 3
def height_difference_in_inches : ℚ := 21
def inches_to_cm (inches : ℚ) : ℚ := inches * 2.54
def cm_to_m (cm : ℚ) : ℚ := cm / 100

theorem marissa_sunflower_height :
  cm_to_m (inches_to_cm (height_sister_in_inches + height_difference_in_inches)) = 1.8288 :=
by sorry

end marissa_sunflower_height_l248_248061


namespace monotonicity_and_inequality_l248_248190

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem monotonicity_and_inequality (a : ℝ) (p q : ℝ) (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1)
  (h_distinct: p ≠ q) (h_a : a ≥ 10) : 
  (f a (p + 1) - f a (q + 1)) / (p - q) > 1 := by
  sorry

end monotonicity_and_inequality_l248_248190


namespace boat_speed_in_still_water_l248_248324

theorem boat_speed_in_still_water (B S : ℕ) (h1 : B + S = 13) (h2 : B - S = 5) : B = 9 :=
by
  sorry

end boat_speed_in_still_water_l248_248324


namespace acetic_acid_molecular_weight_is_correct_l248_248273

def molecular_weight_acetic_acid : ℝ :=
  let carbon_weight := 12.01
  let hydrogen_weight := 1.008
  let oxygen_weight := 16.00
  let num_carbons := 2
  let num_hydrogens := 4
  let num_oxygens := 2
  num_carbons * carbon_weight + num_hydrogens * hydrogen_weight + num_oxygens * oxygen_weight

theorem acetic_acid_molecular_weight_is_correct : molecular_weight_acetic_acid = 60.052 :=
by 
  unfold molecular_weight_acetic_acid
  sorry

end acetic_acid_molecular_weight_is_correct_l248_248273


namespace add_fractions_l248_248578

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248578


namespace tan_x_value_l248_248021

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_x_value:
  (∀ x : ℝ, deriv f x = 2 * f x) → (∀ x : ℝ, f x = Real.sin x - Real.cos x) → (∀ x : ℝ, Real.tan x = 3) := 
by
  intros h_deriv h_f
  sorry

end tan_x_value_l248_248021


namespace find_tangent_c_l248_248184

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → (-12)^2 - 4 * (1) * (12 * c) = 0) → c = 3 :=
sorry

end find_tangent_c_l248_248184


namespace find_f_7_l248_248336

noncomputable def f (a b c d x : ℝ) : ℝ :=
  a * x^8 + b * x^7 + c * x^3 + d * x - 6

theorem find_f_7 (a b c d : ℝ) (h : f a b c d (-7) = 10) :
  f a b c d 7 = 11529580 * a - 22 :=
sorry

end find_f_7_l248_248336


namespace determine_m_of_monotonically_increasing_function_l248_248343

theorem determine_m_of_monotonically_increasing_function 
  (m n : ℝ)
  (h : ∀ x, 12 * x ^ 2 + 2 * m * x + (m - 3) ≥ 0) :
  m = 6 := 
by 
  sorry

end determine_m_of_monotonically_increasing_function_l248_248343


namespace system_has_two_distinct_solutions_for_valid_a_l248_248287

noncomputable def log_eq (x y a : ℝ) : Prop := 
  Real.log (a * x + 4 * a) / Real.log (abs (x + 3)) = 
  2 * Real.log (x + y) / Real.log (abs (x + 3))

noncomputable def original_system (x y a : ℝ) : Prop :=
  log_eq x y a ∧ (x + 1 + Real.sqrt (x^2 + 2 * x + y - 4) = 0)

noncomputable def valid_range (a : ℝ) : Prop := 
  (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16 / 3)

theorem system_has_two_distinct_solutions_for_valid_a (a : ℝ) :
  valid_range a → 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ original_system x₁ 5 a ∧ original_system x₂ 5 a ∧ (-5 < x₁ ∧ x₁ ≤ -1) ∧ (-5 < x₂ ∧ x₂ ≤ -1) := 
sorry

end system_has_two_distinct_solutions_for_valid_a_l248_248287


namespace orange_slices_needed_l248_248979

theorem orange_slices_needed (total_slices containers_capacity leftover_slices: ℕ) 
(h1 : containers_capacity = 4) 
(h2 : total_slices = 329) 
(h3 : leftover_slices = 1) :
    containers_capacity - leftover_slices = 3 :=
by
  sorry

end orange_slices_needed_l248_248979


namespace fraction_addition_l248_248448

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248448


namespace number_of_students_with_no_pets_l248_248913

-- Define the number of students in the class
def total_students : ℕ := 25

-- Define the number of students with cats
def students_with_cats : ℕ := (3 * total_students) / 5

-- Define the number of students with dogs
def students_with_dogs : ℕ := (20 * total_students) / 100

-- Define the number of students with elephants
def students_with_elephants : ℕ := 3

-- Calculate the number of students with no pets
def students_with_no_pets : ℕ := total_students - (students_with_cats + students_with_dogs + students_with_elephants)

-- Statement to be proved
theorem number_of_students_with_no_pets : students_with_no_pets = 2 :=
sorry

end number_of_students_with_no_pets_l248_248913


namespace min_xy_min_x_plus_y_l248_248013

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x * y ≥ 9 :=
sorry

theorem min_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x + y ≥ 6 :=
sorry

end min_xy_min_x_plus_y_l248_248013


namespace max_xyz_eq_one_l248_248339

noncomputable def max_xyz (x y z : ℝ) : ℝ :=
  if h_cond : 0 < x ∧ 0 < y ∧ 0 < z ∧ (x * y + z ^ 2 = (x + z) * (y + z)) ∧ (x + y + z = 3) then
    x * y * z
  else
    0

theorem max_xyz_eq_one : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x * y + z ^ 2 = (x + z) * (y + z)) → (x + y + z = 3) → max_xyz x y z ≤ 1 :=
by
  intros x y z hx hy hz h1 h2
  -- Proof is omitted here
  sorry

end max_xyz_eq_one_l248_248339


namespace baseball_games_in_season_l248_248691

theorem baseball_games_in_season 
  (games_per_month : ℕ) 
  (months_in_season : ℕ)
  (h1 : games_per_month = 7) 
  (h2 : months_in_season = 2) :
  games_per_month * months_in_season = 14 := by
  sorry


end baseball_games_in_season_l248_248691


namespace find_f_10_l248_248765

def f : ℕ → ℚ := sorry
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = f x / (1 + f x)
axiom f_initial : f 1 = 1

theorem find_f_10 : f 10 = 1 / 10 :=
by
  sorry

end find_f_10_l248_248765


namespace ratio_addition_l248_248385

theorem ratio_addition (x : ℤ) (h : 4 + x = 3 * (15 + x) / 4): x = 29 :=
by
  sorry

end ratio_addition_l248_248385


namespace find_other_x_intercept_l248_248294

theorem find_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 2 → y = -3) (h_x_intercept : ∀ x, x = 5 → y = 0) : 
  ∃ x, x = -1 ∧ y = 0 := 
sorry

end find_other_x_intercept_l248_248294


namespace johnny_marbles_combination_l248_248794

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end johnny_marbles_combination_l248_248794


namespace lcm_of_numbers_l248_248910

theorem lcm_of_numbers (a b : ℕ) (L : ℕ) 
  (h1 : a + b = 55) 
  (h2 : Nat.gcd a b = 5) 
  (h3 : (1 / (a : ℝ)) + (1 / (b : ℝ)) = 0.09166666666666666) : (Nat.lcm a b = 120) := 
sorry

end lcm_of_numbers_l248_248910


namespace fraction_addition_l248_248440

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248440


namespace boxes_per_case_l248_248811

/-- Let's define the variables for the problem.
    We are given that Shirley sold 10 boxes of trefoils,
    and she needs to deliver 5 cases of boxes. --/
def total_boxes : ℕ := 10
def number_of_cases : ℕ := 5

/-- We need to prove that the number of boxes in each case is 2. --/
theorem boxes_per_case :
  total_boxes / number_of_cases = 2 :=
by
  -- Definition step where we specify the calculation
  unfold total_boxes number_of_cases
  -- The problem requires a division operation
  norm_num
  -- The result should be correct according to the solution steps
  done

end boxes_per_case_l248_248811


namespace money_left_correct_l248_248941

def initial_amount : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def amount_left : ℕ := initial_amount - total_spent

theorem money_left_correct :
  amount_left = 78 := by
  sorry

end money_left_correct_l248_248941


namespace add_fractions_l248_248522

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248522


namespace arithmetic_sequence_sum_l248_248366

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the conditions
def a_5 := a 5
def a_6 := a 6
def a_7 := a 7

axiom cond1 : a_5 = 11
axiom cond2 : a_6 = 17
axiom cond3 : a_7 = 23

noncomputable def sum_first_four_terms : ℤ :=
  a 1 + a 2 + a 3 + a 4

theorem arithmetic_sequence_sum :
  a_5 = 11 → a_6 = 17 → a_7 = 23 → sum_first_four_terms a = -16 :=
by
  intros h5 h6 h7
  sorry

end arithmetic_sequence_sum_l248_248366


namespace mean_calculation_incorrect_l248_248969

theorem mean_calculation_incorrect (a b c : ℝ) (h : a < b) (h1 : b < c) :
  let x := (a + b) / 2
  let y := (x + c) / 2
  y < (a + b + c) / 3 :=
by 
  let x := (a + b) / 2
  let y := (x + c) / 2
  sorry

end mean_calculation_incorrect_l248_248969


namespace average_marks_of_all_candidates_l248_248816

def n : ℕ := 120
def p : ℕ := 100
def f : ℕ := n - p
def A_p : ℕ := 39
def A_f : ℕ := 15
def total_marks : ℕ := p * A_p + f * A_f
def average_marks : ℚ := total_marks / n

theorem average_marks_of_all_candidates :
  average_marks = 35 := 
sorry

end average_marks_of_all_candidates_l248_248816


namespace prime_count_between_20_and_40_l248_248198

open Nat

def is_prime_in_range_21_to_39 (n : ℕ) : Prop :=
  n ≥ 21 ∧ n ≤ 39 ∧ Prime n

theorem prime_count_between_20_and_40 :
  {n : ℕ | is_prime_in_range_21_to_39 n}.toFinset.card = 4 := by
  sorry

end prime_count_between_20_and_40_l248_248198


namespace increasing_function_probability_l248_248164

noncomputable def a_vals := {-2, 0, 1, 3, 4}
noncomputable def b_vals := {1, 2}

def is_increasing (a : ℤ) : Prop := (a^2 - 2) > 0

def increasing_probability : ℚ :=
  let suitable_a := {a ∈ a_vals | is_increasing a}
  let total_a := a_vals.card
  let suitable_a_count := suitable_a.card
  suitable_a_count / total_a

theorem increasing_function_probability :
  increasing_probability = 3 / 5 :=
by 
  sorry

end increasing_function_probability_l248_248164


namespace curves_tangent_at_m_eq_two_l248_248590

-- Definitions of the ellipsoid and hyperbola equations.
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

-- The proposition to be proved.
theorem curves_tangent_at_m_eq_two :
  ∃ m : ℝ, (∀ x y : ℝ, ellipse x y ∧ hyperbola x y m → m = 2) :=
sorry

end curves_tangent_at_m_eq_two_l248_248590


namespace meena_cookies_left_l248_248655

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end meena_cookies_left_l248_248655


namespace find_N_l248_248773

theorem find_N (x y N : ℝ) (h1 : 2 * x + y = N) (h2 : x + 2 * y = 5) (h3 : (x + y) / 3 = 1) : N = 4 :=
by
  have h4 : x + y = 3 := by
    linarith [h3]
  have h5 : y = 3 - x := by
    linarith [h4]
  have h6 : x + 2 * (3 - x) = 5 := by
    linarith [h2, h5]
  have h7 : x = 1 := by
    linarith [h6]
  have h8 : y = 2 := by
    linarith [h4, h7]
  have h9 : 2 * x + y = 4 := by
    linarith [h7, h8]
  linarith [h1, h9]

end find_N_l248_248773


namespace profit_difference_l248_248716

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end profit_difference_l248_248716


namespace magnitude_of_complex_l248_248235

def complex_number := Complex.mk 2 3 -- Define the complex number 2+3i

theorem magnitude_of_complex : Complex.abs complex_number = Real.sqrt 13 := by
  sorry

end magnitude_of_complex_l248_248235


namespace meena_cookies_left_l248_248656

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end meena_cookies_left_l248_248656


namespace num_women_in_luxury_suite_l248_248147

theorem num_women_in_luxury_suite (total_passengers : ℕ) (pct_women : ℕ) (pct_women_luxury : ℕ)
  (h_total_passengers : total_passengers = 300)
  (h_pct_women : pct_women = 50)
  (h_pct_women_luxury : pct_women_luxury = 15) :
  (total_passengers * pct_women / 100) * pct_women_luxury / 100 = 23 := 
by
  sorry

end num_women_in_luxury_suite_l248_248147


namespace add_fractions_result_l248_248417

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248417


namespace fraction_addition_l248_248561

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248561


namespace xyz_value_l248_248674

-- Define the basic conditions
variables (x y z : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variables (h1 : x * y = 40 * (4:ℝ)^(1/3))
variables (h2 : x * z = 56 * (4:ℝ)^(1/3))
variables (h3 : y * z = 32 * (4:ℝ)^(1/3))
variables (h4 : x + y = 18)

-- The target theorem
theorem xyz_value : x * y * z = 16 * (895:ℝ)^(1/2) :=
by
  -- Here goes the proof, but we add 'sorry' to end the theorem placeholder
  sorry

end xyz_value_l248_248674


namespace gain_percent_correct_l248_248258

noncomputable def cycleCP : ℝ := 900
noncomputable def cycleSP : ℝ := 1180
noncomputable def gainPercent : ℝ := (cycleSP - cycleCP) / cycleCP * 100

theorem gain_percent_correct :
  gainPercent = 31.11 := by
  sorry

end gain_percent_correct_l248_248258


namespace simplify_expression_l248_248358

variable (x y : ℤ)

theorem simplify_expression : 
  (15 * x + 45 * y) + (7 * x + 18 * y) - (6 * x + 35 * y) = 16 * x + 28 * y :=
by
  sorry

end simplify_expression_l248_248358


namespace julia_spent_on_animals_l248_248331

theorem julia_spent_on_animals 
  (total_weekly_cost : ℕ)
  (weekly_cost_rabbit : ℕ)
  (weeks_rabbit : ℕ)
  (weeks_parrot : ℕ) :
  total_weekly_cost = 30 →
  weekly_cost_rabbit = 12 →
  weeks_rabbit = 5 →
  weeks_parrot = 3 →
  total_weekly_cost * weeks_parrot - weekly_cost_rabbit * weeks_parrot + weekly_cost_rabbit * weeks_rabbit = 114 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  linarith,
end

end julia_spent_on_animals_l248_248331


namespace students_with_dogs_l248_248136

theorem students_with_dogs (total_students : ℕ) (half_students : total_students = 100)
                           (girls_percentage : ℕ) (boys_percentage : ℕ)
                           (girls_dog_percentage : ℕ) (boys_dog_percentage : ℕ)
                           (P1 : half_students / 2 = 50)
                           (P2 : girls_dog_percentage = 20)
                           (P3 : boys_dog_percentage = 10) :
                           (50 * girls_dog_percentage / 100 + 
                            50 * boys_dog_percentage / 100) = 15 :=
by sorry

end students_with_dogs_l248_248136


namespace zilla_savings_l248_248114

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l248_248114


namespace find_m_value_l248_248885

theorem find_m_value (m : ℝ) : (∃ A B : ℝ × ℝ, A = (-2, m) ∧ B = (m, 4) ∧ (∃ k : ℝ, k = (4 - m) / (m + 2) ∧ k = -2) ∧ (∃ l : ℝ, l = -2 ∧ 2 * l + l - 1 = 0)) → m = -8 :=
by
  sorry

end find_m_value_l248_248885


namespace percentage_of_money_spent_l248_248248

theorem percentage_of_money_spent (initial_amount remaining_amount : ℝ) (h_initial : initial_amount = 500) (h_remaining : remaining_amount = 350) :
  (((initial_amount - remaining_amount) / initial_amount) * 100) = 30 :=
by
  -- Start the proof
  sorry

end percentage_of_money_spent_l248_248248


namespace tangent_lines_diff_expected_l248_248019

noncomputable def tangent_lines_diff (a : ℝ) (k1 k2 : ℝ) : Prop :=
  let curve (x : ℝ) := a * x + 2 * Real.log (|x|)
  let deriv (x : ℝ) := a + 2 / x
  -- Tangent conditions at some x1 > 0 for k1
  (∃ x1 : ℝ, 0 < x1 ∧ k1 = deriv x1 ∧ curve x1 = k1 * x1)
  -- Tangent conditions at some x2 < 0 for k2
  ∧ (∃ x2 : ℝ, x2 < 0 ∧ k2 = deriv x2 ∧ curve x2 = k2 * x2)
  -- The lines' slopes relations
  ∧ k1 > k2

theorem tangent_lines_diff_expected (a k1 k2 : ℝ) (h : tangent_lines_diff a k1 k2) :
  k1 - k2 = 4 / Real.exp 1 :=
sorry

end tangent_lines_diff_expected_l248_248019


namespace fraction_addition_l248_248456

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248456


namespace farmer_children_l248_248980

theorem farmer_children (n : ℕ) 
  (h1 : 15 * n - 8 - 7 = 60) : n = 5 := 
by
  sorry

end farmer_children_l248_248980


namespace johns_sixth_quiz_score_l248_248793

theorem johns_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (mean : ℕ) (n : ℕ) :
  s1 = 86 ∧ s2 = 91 ∧ s3 = 83 ∧ s4 = 88 ∧ s5 = 97 ∧ mean = 90 ∧ n = 6 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / n = mean ∧ s6 = 95 :=
by
  intro h
  obtain ⟨hs1, hs2, hs3, hs4, hs5, hmean, hn⟩ := h
  have htotal : s1 + s2 + s3 + s4 + s5 + 95 = 540 := by sorry
  have hmean_eq : (s1 + s2 + s3 + s4 + s5 + 95) / n = mean := by sorry
  exact ⟨95, hmean_eq, rfl⟩

end johns_sixth_quiz_score_l248_248793


namespace pizzeria_provolone_shred_l248_248985

theorem pizzeria_provolone_shred 
    (cost_blend : ℝ) 
    (cost_mozzarella : ℝ) 
    (cost_romano : ℝ) 
    (cost_provolone : ℝ) 
    (prop_mozzarella : ℝ) 
    (prop_romano : ℝ) 
    (prop_provolone : ℝ) 
    (shredded_mozzarella : ℕ) 
    (shredded_romano : ℕ) 
    (shredded_provolone_needed : ℕ) :
  cost_blend = 696.05 ∧ 
  cost_mozzarella = 504.35 ∧ 
  cost_romano = 887.75 ∧ 
  cost_provolone = 735.25 ∧ 
  prop_mozzarella = 2 ∧ 
  prop_romano = 1 ∧ 
  prop_provolone = 2 ∧ 
  shredded_mozzarella = 20 ∧ 
  shredded_romano = 10 → 
  shredded_provolone_needed = 20 :=
by {
  sorry -- proof to be provided
}

end pizzeria_provolone_shred_l248_248985


namespace interval_of_n_l248_248008

theorem interval_of_n (n : ℕ) (h_pos : 0 < n) (h_lt_2000 : n < 2000) 
                      (h_div_99999999 : 99999999 % n = 0) (h_div_999999 : 999999 % (n + 6) = 0) : 
                      801 ≤ n ∧ n ≤ 1200 :=
by {
  sorry
}

end interval_of_n_l248_248008


namespace sum_of_powers_mod_7_l248_248256

theorem sum_of_powers_mod_7 :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7 = 1) := by
  sorry

end sum_of_powers_mod_7_l248_248256


namespace parameterization_properties_l248_248686

theorem parameterization_properties (a b c d : ℚ)
  (h1 : a * (-1) + b = -3)
  (h2 : c * (-1) + d = 5)
  (h3 : a * 2 + b = 4)
  (h4 : c * 2 + d = 15) :
  a^2 + b^2 + c^2 + d^2 = 790 / 9 :=
sorry

end parameterization_properties_l248_248686


namespace numMilkmen_rented_pasture_l248_248945

def cowMonths (cows: ℕ) (months: ℕ) : ℕ := cows * months

def totalCowMonths (a: ℕ) (b: ℕ) (c: ℕ) (d: ℕ) : ℕ := a + b + c + d

noncomputable def rentPerCowMonth (share: ℕ) (cowMonths: ℕ) : ℕ := 
  share / cowMonths

theorem numMilkmen_rented_pasture 
  (a_cows: ℕ) (a_months: ℕ) (b_cows: ℕ) (b_months: ℕ) (c_cows: ℕ) (c_months: ℕ) (d_cows: ℕ) (d_months: ℕ)
  (a_share: ℕ) (total_rent: ℕ) 
  (ha: a_cows = 24) (hma: a_months = 3) 
  (hb: b_cows = 10) (hmb: b_months = 5)
  (hc: c_cows = 35) (hmc: c_months = 4)
  (hd: d_cows = 21) (hmd: d_months = 3)
  (ha_share: a_share = 720) (htotal_rent: total_rent = 3250)
  : 4 = 4 := by
  sorry

end numMilkmen_rented_pasture_l248_248945


namespace axes_of_symmetry_coincide_l248_248075

-- Define the quadratic functions f and g
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := (x + 3) / 4
def g' (x : ℝ) : ℝ := -x / 4

-- Define the axes of symmetry for the functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the intersection points
def intersection_points : List ℝ := [4, -7]

-- State the problem: Do the axes of symmetry coincide?
theorem axes_of_symmetry_coincide :
  (axis_of_symmetry_f = axis_of_symmetry_g) = False :=
by
  sorry

end axes_of_symmetry_coincide_l248_248075


namespace exponent_fraction_simplification_l248_248377

theorem exponent_fraction_simplification :
  (2 ^ 2020 + 2 ^ 2016) / (2 ^ 2020 - 2 ^ 2016) = 17 / 15 :=
by
  sorry

end exponent_fraction_simplification_l248_248377


namespace count_odd_ad_bc_l248_248171

open Finset

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

theorem count_odd_ad_bc :
  let s := {0, 1, 2, 3}
  finset.card {p : ℤ × ℤ × ℤ × ℤ | p.1.1 ∈ s ∧ p.1.2 ∈ s ∧ p.2.1 ∈ s ∧ p.2.2 ∈ s ∧
                                    is_odd (p.1.1 * p.2.2 - p.1.2 * p.2.1) } = 96 :=
by
  sorry

end count_odd_ad_bc_l248_248171


namespace add_fractions_result_l248_248413

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248413


namespace red_balls_removal_condition_l248_248042

theorem red_balls_removal_condition (total_balls : ℕ) (initial_red_balls : ℕ) (r : ℕ) : 
  total_balls = 600 → 
  initial_red_balls = 420 → 
  60 * (total_balls - r) = 100 * (initial_red_balls - r) → 
  r = 150 :=
by
  sorry

end red_balls_removal_condition_l248_248042


namespace ball_bounces_height_l248_248126

theorem ball_bounces_height (initial_height : ℝ) (decay_factor : ℝ) (threshold : ℝ) (n : ℕ) :
  initial_height = 20 →
  decay_factor = 3/4 →
  threshold = 2 →
  n = 9 →
  initial_height * (decay_factor ^ n) < threshold :=
by
  intros
  sorry

end ball_bounces_height_l248_248126


namespace fraction_of_meat_used_for_meatballs_l248_248049

theorem fraction_of_meat_used_for_meatballs
    (initial_meat : ℕ)
    (spring_rolls_meat : ℕ)
    (remaining_meat : ℕ)
    (total_meat_used : ℕ)
    (meatballs_meat : ℕ)
    (h_initial : initial_meat = 20)
    (h_spring_rolls : spring_rolls_meat = 3)
    (h_remaining : remaining_meat = 12) :
    (initial_meat - remaining_meat) = total_meat_used ∧
    (total_meat_used - spring_rolls_meat) = meatballs_meat ∧
    (meatballs_meat / initial_meat) = (1/4 : ℝ) :=
by
  sorry

end fraction_of_meat_used_for_meatballs_l248_248049


namespace axes_of_symmetry_do_not_coincide_l248_248077

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the axes of symmetry for the quadratic functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the slopes of the tangents to the graphs at x = 4 and x = -7
def slope_f (x : ℝ) : ℝ := (2*x + 6) / 8
def slope_g (x : ℝ) : ℝ := -x / 4

-- We need to prove that the axes of symmetry do not coincide
theorem axes_of_symmetry_do_not_coincide :
    axis_of_symmetry_f ≠ axis_of_symmetry_g :=
by {
    sorry
}

end axes_of_symmetry_do_not_coincide_l248_248077


namespace ratio_sub_add_l248_248903

theorem ratio_sub_add (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 :=
sorry

end ratio_sub_add_l248_248903


namespace value_of_expression_l248_248647

theorem value_of_expression (a b : ℝ) (h1 : a ≠ b)
  (h2 : a^2 + 2 * a - 2022 = 0)
  (h3 : b^2 + 2 * b - 2022 = 0) :
  a^2 + 4 * a + 2 * b = 2018 :=
by
  sorry

end value_of_expression_l248_248647


namespace cos_of_sum_eq_one_l248_248297

theorem cos_of_sum_eq_one
  (x y : ℝ)
  (a : ℝ)
  (h1 : x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h2 : y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h3 : x^3 + Real.sin x - 2 * a = 0)
  (h4 : 4 * y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2 * y) = 1 := 
by
  sorry

end cos_of_sum_eq_one_l248_248297


namespace solve_inequality_system_l248_248229

theorem solve_inequality_system (x : ℝ) :
  (x - 1 < 2 * x + 1) ∧ ((2 * x - 5) / 3 ≤ 1) → (-2 < x ∧ x ≤ 4) :=
by
  intro cond
  sorry

end solve_inequality_system_l248_248229


namespace unique_positive_integers_pqr_l248_248341

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 61) / 2 + 5 / 2)

lemma problem_condition (p q r : ℕ) (py : ℝ) :
  py = y^100
  ∧ py = 2 * (y^98)
  ∧ py = 16 * (y^96)
  ∧ py = 13 * (y^94)
  ∧ py = - y^50
  ∧ py = ↑p * y^46
  ∧ py = ↑q * y^44
  ∧ py = ↑r * y^40 :=
sorry

theorem unique_positive_integers_pqr : 
  ∃! (p q r : ℕ), 
    p = 37 ∧ q = 47 ∧ r = 298 ∧ 
    y^100 = 2 * y^98 + 16 * y^96 + 13 * y^94 - y^50 + ↑p * y^46 + ↑q * y^44 + ↑r * y^40 :=
sorry

end unique_positive_integers_pqr_l248_248341


namespace Tiffany_bags_l248_248097

theorem Tiffany_bags (x : ℕ) 
  (h1 : 8 = x + 1) : 
  x = 7 :=
by
  sorry

end Tiffany_bags_l248_248097


namespace concentration_after_dilution_l248_248748

-- Definitions and conditions
def initial_volume : ℝ := 5
def initial_concentration : ℝ := 0.06
def poured_out_volume : ℝ := 1
def added_water_volume : ℝ := 2

-- Theorem statement
theorem concentration_after_dilution : 
  (initial_volume * initial_concentration - poured_out_volume * initial_concentration) / 
  (initial_volume - poured_out_volume + added_water_volume) = 0.04 :=
by 
  sorry

end concentration_after_dilution_l248_248748


namespace find_b_l248_248239

theorem find_b (a b : ℤ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 :=
sorry

end find_b_l248_248239


namespace max_distance_to_pole_l248_248045

noncomputable def max_distance_to_origin (r1 r2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  r1 + r2

theorem max_distance_to_pole (r : ℝ) (c : ℝ) : max_distance_to_origin 2 1 0 0 = 3 := by
  sorry

end max_distance_to_pole_l248_248045


namespace alicia_local_tax_in_cents_l248_248728

theorem alicia_local_tax_in_cents (hourly_wage : ℝ) (tax_rate : ℝ)
  (h_hourly_wage : hourly_wage = 30) (h_tax_rate : tax_rate = 0.021) :
  (hourly_wage * tax_rate * 100) = 63 := by
  sorry

end alicia_local_tax_in_cents_l248_248728


namespace fraction_meaningful_iff_l248_248829

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x + 1)) ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_iff_l248_248829


namespace benjie_is_6_years_old_l248_248856

-- Definitions based on conditions
def margo_age_in_3_years := 4
def years_until_then := 3
def age_difference := 5

-- Current age of Margo
def margo_current_age := margo_age_in_3_years - years_until_then

-- Current age of Benjie
def benjie_current_age := margo_current_age + age_difference

-- The theorem we need to prove
theorem benjie_is_6_years_old : benjie_current_age = 6 :=
by
  -- Proof
  sorry

end benjie_is_6_years_old_l248_248856


namespace profit_difference_l248_248717

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end profit_difference_l248_248717


namespace range_of_t_l248_248204

variable (t : ℝ)

def point_below_line (x y a b c : ℝ) : Prop :=
  a * x - b * y + c < 0

theorem range_of_t (t : ℝ) : point_below_line 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
  sorry

end range_of_t_l248_248204


namespace exists_subset_F_l248_248051

variable (E : Type) [Fintype E]
variable (f : Finset E → ℝ) 
variable (h0 : ∀ A B : Finset E, Disjoint A B → f (A ∪ B) = f A + f B)

theorem exists_subset_F (hf : ∀ A, 0 ≤ f A) : 
∃ F : Finset E, ∀ A : Finset E,
  let A' := A \ F in
  f A = f A' ∧ (f A = 0 ↔ A ⊆ F) :=
by
  sorry

end exists_subset_F_l248_248051


namespace find_ab_l248_248202

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 15 :=
by
  sorry

end find_ab_l248_248202


namespace wang_trip_duration_xiao_travel_times_l248_248354

variables (start_fee : ℝ) (time_fee_per_min : ℝ) (mileage_fee_per_km : ℝ) (long_distance_fee_per_km : ℝ)

-- Conditions
def billing_rules := 
  start_fee = 12 ∧ 
  time_fee_per_min = 0.5 ∧ 
  mileage_fee_per_km = 2.0 ∧ 
  long_distance_fee_per_km = 1.0

-- Proof for Mr. Wang's trip duration
theorem wang_trip_duration
  (x : ℝ) 
  (total_fare : ℝ)
  (distance : ℝ) 
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km) : 
  total_fare = 69.5 ∧ distance = 20 → 0.5 * x = 12.5 :=
by 
  sorry

-- Proof for Xiao Hong's and Xiao Lan's travel times
theorem xiao_travel_times 
  (x : ℝ) 
  (travel_time_multiplier : ℝ)
  (distance_hong : ℝ)
  (distance_lan : ℝ)
  (equal_fares : Prop)
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km)
  (p1 : distance_hong = 14 ∧ distance_lan = 16 ∧ travel_time_multiplier = 1.5) :
  equal_fares → 0.25 * x = 5 :=
by 
  sorry

end wang_trip_duration_xiao_travel_times_l248_248354


namespace relationship_among_abc_l248_248761

theorem relationship_among_abc (e1 e2 : ℝ) (h1 : 0 ≤ e1) (h2 : e1 < 1) (h3 : e2 > 1) :
  let a := 3 ^ e1
  let b := 2 ^ (-e2)
  let c := Real.sqrt 5
  b < c ∧ c < a := by
  sorry

end relationship_among_abc_l248_248761


namespace fraction_addition_l248_248450

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248450


namespace sufficient_but_not_necessary_l248_248382

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x ≥ 2 ∧ y ≥ 2) → x + y ≥ 4 ∧ (¬ (x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_l248_248382


namespace proof_problem_l248_248216

noncomputable def f (a b : ℝ) (x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) : 
  (f a b (11 * π / 12) = 0) ∧
  (|f a b (7 * π / 12)| < |f a b (π / 5)|) ∧
  (¬ (∀ x : ℝ, f a b x = f a b (-x)) ∧ ¬ (∀ x : ℝ, f a b x = -f a b (-x))) := 
sorry

end proof_problem_l248_248216


namespace reciprocal_of_neg3_l248_248086

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l248_248086


namespace max_distinct_prime_factors_of_a_l248_248946

noncomputable def distinct_prime_factors (n : ℕ) : ℕ := sorry -- placeholder for the number of distinct prime factors

theorem max_distinct_prime_factors_of_a (a b : ℕ)
  (ha_pos : a > 0) (hb_pos : b > 0)
  (gcd_ab_primes : distinct_prime_factors (gcd a b) = 5)
  (lcm_ab_primes : distinct_prime_factors (lcm a b) = 18)
  (a_less_than_b : distinct_prime_factors a < distinct_prime_factors b) :
  distinct_prime_factors a = 11 :=
sorry

end max_distinct_prime_factors_of_a_l248_248946


namespace mike_avg_speed_l248_248352

/-
  Given conditions:
  * total distance d = 640 miles
  * half distance h = 320 miles
  * first half average rate r1 = 80 mph
  * time for first half t1 = h / r1 = 4 hours
  * second half time t2 = 3 * t1 = 12 hours
  * total time tt = t1 + t2 = 16 hours
  * total distance d = 640 miles
  * average rate for entire trip should be (d/tt) = 40 mph.
  
  The goal is to prove that the average rate for the entire trip is 40 mph.
-/
theorem mike_avg_speed:
  let d := 640 in
  let h := 320 in
  let r1 := 80 in
  let t1 := h / r1 in
  let t2 := 3 * t1 in
  let tt := t1 + t2 in
  let avg_rate := d / tt in
  avg_rate = 40 := by
  sorry

end mike_avg_speed_l248_248352


namespace add_fractions_l248_248513

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248513


namespace gcd_lcm_product_180_l248_248967

theorem gcd_lcm_product_180 (a b : ℕ) (g l : ℕ) (ha : a > 0) (hb : b > 0) (hg : g > 0) (hl : l > 0) 
  (h₁ : g = gcd a b) (h₂ : l = lcm a b) (h₃ : g * l = 180):
  ∃(n : ℕ), n = 8 :=
by
  sorry

end gcd_lcm_product_180_l248_248967


namespace obtuse_triangle_sum_range_l248_248323

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end obtuse_triangle_sum_range_l248_248323


namespace find_upper_base_length_l248_248173

-- Define the trapezoid and its properties.
variables (d : ℝ)
variables (A D : ℝ × ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the conditions of the problem.
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := 
  ∃ M, M.1 = (A.1 + B.1) / 2 ∧ M.1 = (C.1 + D.1) / 2
  
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def equal_distance (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R

-- Setting the exact locations of points
def coordinates : Prop := 
  D = (0, 0) ∧ A = (d, 0) ∧ perpendicular (M) (D, A)

-- Required proof
theorem find_upper_base_length :
  coordinates d A D M ∧ equal_distance M C D → 
  dist (A, C) = d / 2 :=
by sorry

end find_upper_base_length_l248_248173


namespace area_of_triangle_8_9_9_l248_248004

noncomputable def triangle_area (a b c : ℕ) : Real :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_8_9_9 : triangle_area 8 9 9 = 4 * Real.sqrt 65 :=
by
  sorry

end area_of_triangle_8_9_9_l248_248004


namespace circles_intersect_area_3_l248_248961

def circle_intersection_area (r : ℝ) (c1 c2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := c1
  let (x2, y2) := c2
  let dist_centers := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  if dist_centers > 2 * r then 0
  else if dist_centers = 0 then π * r^2
  else
    let α := 2 * Real.acos (dist_centers / (2 * r))
    let area_segment := r^2 * (α - Real.sin(α)) / 2
    2 * area_segment - r^2 * Real.sin(α)

theorem circles_intersect_area_3 :
  circle_intersection_area 3 (3,0) (0,3) = (9 * π / 2) - 9 :=
by
  sorry

end circles_intersect_area_3_l248_248961


namespace solve_system_l248_248360

theorem solve_system (x1 x2 x3 : ℝ) :
  (x1 - 2 * x2 + 3 * x3 = 5) ∧ 
  (2 * x1 + 3 * x2 - x3 = 7) ∧ 
  (3 * x1 + x2 + 2 * x3 = 12) 
  ↔ (x1, x2, x3) = (7 - 5 * x3, 1 - x3, x3) :=
by
  sorry

end solve_system_l248_248360


namespace simplify_expression_correct_l248_248944

noncomputable def simplify_expression (α : ℝ) : ℝ :=
    (2 * (Real.cos (2 * α))^2 - 1) / 
    (2 * Real.tan ((Real.pi / 4) - 2 * α) * (Real.sin ((3 * Real.pi / 4) - 2 * α))^2) -
    Real.tan (2 * α) + Real.cos (2 * α) - Real.sin (2 * α)

theorem simplify_expression_correct (α : ℝ) : 
    simplify_expression α = 
    (2 * Real.sqrt 2 * Real.sin ((Real.pi / 4) - 2 * α) * (Real.cos α)^2) /
    Real.cos (2 * α) := by
    sorry

end simplify_expression_correct_l248_248944


namespace mrs_evans_class_l248_248930

def students_enrolled_in_class (S Q1 Q2 missing both: ℕ) : Prop :=
  25 = Q1 ∧ 22 = Q2 ∧ 5 = missing ∧ 22 = both → S = Q1 + Q2 - both + missing

theorem mrs_evans_class (S : ℕ) : students_enrolled_in_class S 25 22 5 22 :=
by
  sorry

end mrs_evans_class_l248_248930


namespace initial_observations_l248_248678

theorem initial_observations {n : ℕ} (S : ℕ) (new_observation : ℕ) 
  (h1 : S = 15 * n) (h2 : new_observation = 14 - n)
  (h3 : (S + new_observation) / (n + 1) = 14) : n = 6 :=
sorry

end initial_observations_l248_248678


namespace exponent_zero_value_of_neg_3_raised_to_zero_l248_248244

theorem exponent_zero (x : ℤ) (hx : x ≠ 0) : x ^ 0 = 1 :=
by
  -- Proof goes here
  sorry

theorem value_of_neg_3_raised_to_zero : (-3 : ℤ) ^ 0 = 1 :=
by
  exact exponent_zero (-3) (by norm_num)

end exponent_zero_value_of_neg_3_raised_to_zero_l248_248244


namespace total_amount_spent_l248_248125

theorem total_amount_spent (avg_price_goat : ℕ) (num_goats : ℕ) (avg_price_cow : ℕ) (num_cows : ℕ) (total_spent : ℕ) 
  (h1 : avg_price_goat = 70) (h2 : num_goats = 10) (h3 : avg_price_cow = 400) (h4 : num_cows = 2) :
  total_spent = 1500 :=
by
  have cost_goats := avg_price_goat * num_goats
  have cost_cows := avg_price_cow * num_cows
  have total := cost_goats + cost_cows
  sorry

end total_amount_spent_l248_248125


namespace money_left_correct_l248_248940

def initial_amount : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def amount_left : ℕ := initial_amount - total_spent

theorem money_left_correct :
  amount_left = 78 := by
  sorry

end money_left_correct_l248_248940


namespace find_x_l248_248874

noncomputable def x : ℝ :=
  sorry

theorem find_x (h : ∃ x : ℝ, x > 0 ∧ ⌊x⌋ * x = 48) : x = 8 :=
  sorry

end find_x_l248_248874


namespace add_fractions_l248_248526

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l248_248526


namespace fraction_addition_l248_248459

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248459


namespace weight_of_11m_rebar_l248_248291

theorem weight_of_11m_rebar (w5m : ℝ) (l5m : ℝ) (l11m : ℝ) 
  (h_w5m : w5m = 15.3) (h_l5m : l5m = 5) (h_l11m : l11m = 11) : 
  (w5m / l5m) * l11m = 33.66 := 
by {
  sorry
}

end weight_of_11m_rebar_l248_248291


namespace fraction_addition_l248_248447

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248447


namespace fraction_addition_l248_248467

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248467


namespace total_assembly_time_l248_248652

def chairs := 2
def tables := 2
def bookshelf := 1
def tv_stand := 1

def time_per_chair := 8
def time_per_table := 12
def time_per_bookshelf := 25
def time_per_tv_stand := 35

theorem total_assembly_time : (chairs * time_per_chair) + (tables * time_per_table) + (bookshelf * time_per_bookshelf) + (tv_stand * time_per_tv_stand) = 100 := by
  sorry

end total_assembly_time_l248_248652


namespace solution_set_inequality_l248_248778

theorem solution_set_inequality (m : ℝ) (h : 3 - m < 0) :
  { x : ℝ | (2 - m) * x + 2 > m } = { x : ℝ | x < -1 } :=
sorry

end solution_set_inequality_l248_248778


namespace add_fractions_result_l248_248410

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248410


namespace positive_integers_m_divisors_l248_248878

theorem positive_integers_m_divisors :
  ∃ n, n = 3 ∧ ∀ m : ℕ, (0 < m ∧ ∃ k, 2310 = k * (m^2 + 2)) ↔ m = 1 ∨ m = 2 ∨ m = 3 :=
by
  sorry

end positive_integers_m_divisors_l248_248878


namespace multiple_of_4_and_6_sum_even_l248_248948

theorem multiple_of_4_and_6_sum_even (a b : ℤ) (h₁ : ∃ m : ℤ, a = 4 * m) (h₂ : ∃ n : ℤ, b = 6 * n) : ∃ k : ℤ, (a + b) = 2 * k :=
by
  sorry

end multiple_of_4_and_6_sum_even_l248_248948


namespace find_constant_t_l248_248641

theorem find_constant_t (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t + 5^n) ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ (a 1 = S 1) ∧ 
  (∃ q, ∀ n ≥ 1, a (n + 1) = q * a n) → 
  t = -1 := by
  sorry

end find_constant_t_l248_248641


namespace compound_interest_rate_l248_248255

theorem compound_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Annual interest rate in decimal
  (A2 A3 : ℝ)  -- Amounts after 2 and 3 years
  (h2 : A2 = P * (1 + r)^2)
  (h3 : A3 = P * (1 + r)^3) :
  A2 = 17640 → A3 = 22932 → r = 0.3 := by
  sorry

end compound_interest_rate_l248_248255


namespace find_primes_l248_248743

-- Definition of being a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

-- Lean 4 statement of the problem
theorem find_primes (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 → p = 5 ∧ q = 3 ∧ r = 19 := 
by
  sorry

end find_primes_l248_248743


namespace students_in_fifth_and_sixth_classes_l248_248220

theorem students_in_fifth_and_sixth_classes :
  let c1 := 20
  let c2 := 25
  let c3 := 25
  let c4 := c1 / 2
  let total_students := 136
  let total_first_four_classes := c1 + c2 + c3 + c4
  let c5_and_c6 := total_students - total_first_four_classes
  c5_and_c6 = 56 :=
by
  sorry

end students_in_fifth_and_sixth_classes_l248_248220


namespace remaining_units_l248_248399

theorem remaining_units : 
  ∀ (total_units : ℕ) (first_half_fraction : ℚ) (additional_units : ℕ), 
  total_units = 2000 →
  first_half_fraction = 3 / 5 →
  additional_units = 300 →
  (total_units - (first_half_fraction * total_units).toNat - additional_units) = 500 := by
  intros total_units first_half_fraction additional_units htotal hunits_fraction hadditional
  sorry

end remaining_units_l248_248399


namespace fraction_addition_l248_248570

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248570


namespace total_people_at_evening_l248_248661

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l248_248661


namespace xyz_div_by_27_l248_248367

theorem xyz_div_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) :=
sorry

end xyz_div_by_27_l248_248367


namespace total_pages_in_book_l248_248588

def pages_already_read : ℕ := 147
def pages_left_to_read : ℕ := 416

theorem total_pages_in_book : pages_already_read + pages_left_to_read = 563 := by
  sorry

end total_pages_in_book_l248_248588


namespace add_fractions_l248_248573

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248573


namespace students_with_dogs_l248_248135

theorem students_with_dogs (total_students : ℕ) (half_students : total_students / 2 = 50)
  (percent_girls_with_dogs : ℕ → ℚ) (percent_boys_with_dogs : ℕ → ℚ)
  (girls_with_dogs : ∀ (total_girls: ℕ), percent_girls_with_dogs total_girls = 0.2)
  (boys_with_dogs : ∀ (total_boys: ℕ), percent_boys_with_dogs total_boys = 0.1) :
  ∀ (total_girls total_boys students_with_dogs: ℕ),
  total_students = 100 →
  total_girls = total_students / 2 →
  total_boys = total_students / 2 →
  total_girls = 50 →
  total_boys = 50 →
  students_with_dogs = (percent_girls_with_dogs (total_students / 2) * (total_students / 2) + 
                        percent_boys_with_dogs (total_students / 2) * (total_students / 2)) →
  students_with_dogs = 15 :=
by
  intros total_girls total_boys students_with_dogs h1 h2 h3 h4 h5 h6
  sorry

end students_with_dogs_l248_248135


namespace seashell_count_l248_248800

variable (initial_seashells additional_seashells total_seashells : ℕ)

theorem seashell_count (h1 : initial_seashells = 19) (h2 : additional_seashells = 6) : 
  total_seashells = initial_seashells + additional_seashells → total_seashells = 25 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end seashell_count_l248_248800


namespace area_of_intersection_circles_l248_248960

-- Constants representing the circles and required parameters
def circle1 := {x : ℝ × ℝ // (x.1 - 3)^2 + x.2^2 < 9}
def circle2 := {x : ℝ × ℝ // x.1^2 + (x.2 - 3)^2 < 9}

-- Theorem stating the area of the intersection of the two circles
theorem area_of_intersection_circles :
  (area_of_intersection circle1 circle2) = (9 * (π - 2) / 2) :=
by sorry

end area_of_intersection_circles_l248_248960


namespace tan_2theta_sin_cos_fraction_l248_248016

variable {θ : ℝ} (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1)

-- Part (I)
theorem tan_2theta (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : Real.tan (2 * θ) = 4 / 3 :=
by sorry

-- Part (II)
theorem sin_cos_fraction (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 :=
by sorry

end tan_2theta_sin_cos_fraction_l248_248016


namespace perpendicular_lines_slope_l248_248747

theorem perpendicular_lines_slope (b : ℝ) (h1 : ∀ x : ℝ, 3 * x + 7 = 4 * (-(b / 4) * x + 4)) : b = 4 / 3 :=
by
  sorry

end perpendicular_lines_slope_l248_248747


namespace trapezoid_area_l248_248932

theorem trapezoid_area (EF GH h : ℕ) (hEF : EF = 60) (hGH : GH = 30) (hh : h = 15) : 
  (EF + GH) * h / 2 = 675 := by 
  sorry

end trapezoid_area_l248_248932


namespace john_total_trip_cost_l248_248215

noncomputable def total_trip_cost
  (hotel_nights : ℕ) 
  (hotel_rate_per_night : ℝ) 
  (discount : ℝ) 
  (loyal_customer_discount_rate : ℝ) 
  (service_tax_rate : ℝ) 
  (room_service_cost_per_day : ℝ) 
  (cab_cost_per_ride : ℝ) : ℝ :=
  let hotel_cost := hotel_nights * hotel_rate_per_night
  let cost_after_discount := hotel_cost - discount
  let loyal_customer_discount := loyal_customer_discount_rate * cost_after_discount
  let cost_after_loyalty_discount := cost_after_discount - loyal_customer_discount
  let service_tax := service_tax_rate * cost_after_loyalty_discount
  let final_hotel_cost := cost_after_loyalty_discount + service_tax
  let room_service_cost := hotel_nights * room_service_cost_per_day
  let cab_cost := cab_cost_per_ride * 2 * hotel_nights
  final_hotel_cost + room_service_cost + cab_cost

theorem john_total_trip_cost : total_trip_cost 3 250 100 0.10 0.12 50 30 = 985.20 :=
by 
  -- We are skipping the proof but our focus is the statement
  sorry

end john_total_trip_cost_l248_248215


namespace add_fractions_l248_248575

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248575


namespace max_XYZ_plus_terms_l248_248342

theorem max_XYZ_plus_terms {X Y Z : ℕ} (h : X + Y + Z = 15) :
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 :=
sorry

end max_XYZ_plus_terms_l248_248342


namespace add_fractions_l248_248508

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248508


namespace zilla_savings_l248_248116

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l248_248116


namespace add_fractions_result_l248_248422

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248422


namespace problem_statement_l248_248033

def g (x : ℝ) : ℝ := x ^ 3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem problem_statement : f (g 3) = 53 :=
by
  sorry

end problem_statement_l248_248033


namespace problem_conditions_l248_248888

theorem problem_conditions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 3) :
  (x * y ≤ 9 / 8) ∧ (4 ^ x + 2 ^ y ≥ 4 * Real.sqrt 2) ∧ (x / y + 1 / x ≥ 2 / 3 + 2 * Real.sqrt 3 / 3) :=
by
  -- Proof goes here
  sorry

end problem_conditions_l248_248888


namespace factor_correct_l248_248599

-- Define the polynomial p(x)
def p (x : ℤ) : ℤ := 6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2

-- Define the potential factors of p(x)
def f1 (x : ℤ) : ℤ := 3 * x^2 + 93 * x
def f2 (x : ℤ) : ℤ := 2 * x^2 + 178 * x + 5432

theorem factor_correct : ∀ x : ℤ, p x = f1 x * f2 x := by
  sorry

end factor_correct_l248_248599


namespace watched_videos_correct_l248_248405

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end watched_videos_correct_l248_248405


namespace sticks_forming_equilateral_triangle_l248_248154

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l248_248154


namespace trigonometric_sum_l248_248298

theorem trigonometric_sum (θ : ℝ) (h_tan_θ : Real.tan θ = 5 / 12) (h_range : π ≤ θ ∧ θ ≤ 3 * π / 2) : 
  Real.cos θ + Real.sin θ = -17 / 13 :=
by
  sorry

end trigonometric_sum_l248_248298


namespace sets_relation_l248_248179

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def M : Set ℚ := {x | ∃ (m : ℤ), x = m + 1/6}
def S : Set ℚ := {x | ∃ (s : ℤ), x = s/2 - 1/3}
def P : Set ℚ := {x | ∃ (p : ℤ), x = p/2 + 1/6}

theorem sets_relation : M ⊆ S ∧ S = P := by
  sorry

end sets_relation_l248_248179


namespace students_like_both_l248_248767

variable (total_students : ℕ) 
variable (students_like_sea : ℕ) 
variable (students_like_mountains : ℕ) 
variable (students_like_neither : ℕ) 

theorem students_like_both (h1 : total_students = 500)
                           (h2 : students_like_sea = 337)
                           (h3 : students_like_mountains = 289)
                           (h4 : students_like_neither = 56) :
  (students_like_sea + students_like_mountains - (total_students - students_like_neither)) = 182 :=
sorry

end students_like_both_l248_248767


namespace positive_number_divisible_by_4_l248_248721

theorem positive_number_divisible_by_4 (N : ℕ) (h1 : N % 4 = 0) (h2 : (2 + 4 + N + 3) % 2 = 1) : N = 4 := 
by 
  sorry

end positive_number_divisible_by_4_l248_248721


namespace find_x_l248_248889

theorem find_x (x : ℝ) (h : 128/x + 75/x + 57/x = 6.5) : x = 40 :=
by
  sorry

end find_x_l248_248889


namespace fraction_addition_l248_248449

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l248_248449


namespace max_unsealed_windows_l248_248914

-- Definitions of conditions for the problem
def windows : Nat := 15
def panes : Nat := 15

-- Definition of the matching and selection process conditions
def matched_panes (window pane : Nat) : Prop :=
  pane >= window

-- Proof problem statement
theorem max_unsealed_windows 
  (glazier_approaches_window : ∀ (current_window : Nat), ∃ pane : Nat, pane >= current_window) :
  ∃ (max_unsealed : Nat), max_unsealed = 7 :=
by
  sorry

end max_unsealed_windows_l248_248914


namespace average_rate_of_trip_l248_248349

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l248_248349


namespace complement_of_M_with_respect_to_U_l248_248163

noncomputable def U : Set ℕ := {1, 2, 3, 4}
noncomputable def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l248_248163


namespace area_of_triangle_is_correct_l248_248745

def line_1 (x y : ℝ) : Prop := y - 5 * x = -4
def line_2 (x y : ℝ) : Prop := 4 * y + 2 * x = 16

def y_axis (x y : ℝ) : Prop := x = 0

def satisfies_y_intercepts (f : ℝ → ℝ) : Prop :=
f 0 = -4 ∧ f 0 = 4

noncomputable def area_of_triangle (height base : ℝ) : ℝ :=
(1 / 2) * base * height

theorem area_of_triangle_is_correct :
  ∃ (x y : ℝ), line_1 x y ∧ line_2 x y ∧ y_axis 0 8 ∧ area_of_triangle (16 / 11) 8 = (64 / 11) := 
sorry

end area_of_triangle_is_correct_l248_248745


namespace polygon_perimeter_l248_248634

theorem polygon_perimeter (a b : ℕ) (h : adjacent_sides_perpendicular) :
  perimeter = 2 * (a + b) :=
sorry

end polygon_perimeter_l248_248634


namespace total_games_played_l248_248122

-- Define the function for combinations
def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def teams : ℕ := 20
def games_per_pair : ℕ := 10

-- Proposition stating the target result
theorem total_games_played : 
  (combination teams 2 * games_per_pair) = 1900 :=
by
  sorry

end total_games_played_l248_248122


namespace quadratic_int_roots_iff_n_eq_3_or_4_l248_248168

theorem quadratic_int_roots_iff_n_eq_3_or_4 (n : ℕ) (hn : 0 < n) :
    (∃ m k : ℤ, (m ≠ k) ∧ (m^2 - 4 * m + n = 0) ∧ (k^2 - 4 * k + n = 0)) ↔ (n = 3 ∨ n = 4) := sorry

end quadratic_int_roots_iff_n_eq_3_or_4_l248_248168


namespace money_left_after_shopping_l248_248937

def initial_money : ℕ := 158
def shoe_cost : ℕ := 45
def bag_cost := shoe_cost - 17
def lunch_cost := bag_cost / 4
def total_expenses := shoe_cost + bag_cost + lunch_cost
def remaining_money := initial_money - total_expenses

theorem money_left_after_shopping : remaining_money = 78 := by
  sorry

end money_left_after_shopping_l248_248937


namespace add_fractions_result_l248_248408

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248408


namespace sum_fractions_eq_l248_248469

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248469


namespace seventh_rack_dvds_l248_248252

def rack_dvds : ℕ → ℕ
| 0 => 3
| 1 => 4
| n + 2 => ((rack_dvds (n + 1)) - (rack_dvds n)) * 2 + (rack_dvds (n + 1))

theorem seventh_rack_dvds : rack_dvds 6 = 66 := 
by
  sorry

end seventh_rack_dvds_l248_248252


namespace percentage_of_valid_votes_l248_248639

theorem percentage_of_valid_votes 
  (total_votes : ℕ) 
  (invalid_percentage : ℕ) 
  (candidate_valid_votes : ℕ)
  (percentage_invalid : invalid_percentage = 15)
  (total_votes_eq : total_votes = 560000)
  (candidate_votes_eq : candidate_valid_votes = 380800) 
  : (candidate_valid_votes : ℝ) / (total_votes * (0.85 : ℝ)) * 100 = 80 := 
by 
  sorry

end percentage_of_valid_votes_l248_248639


namespace add_fractions_result_l248_248415

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248415


namespace find_minimum_value_l248_248921

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem find_minimum_value :
  let x := 9
  let y := 2
  (∀ x y : ℝ, f x y ≥ 3) ∧ (f 9 2 = 3) :=
by
  sorry

end find_minimum_value_l248_248921


namespace positive_numbers_l248_248806

theorem positive_numbers {a b c : ℝ} (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end positive_numbers_l248_248806


namespace fraction_addition_l248_248543

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248543


namespace perpendicular_condition_parallel_condition_parallel_opposite_direction_l248_248010

variables (a b : ℝ × ℝ) (k : ℝ)

-- Define the vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-3, 2)

-- Define the given expressions
def expression1 (k : ℝ) : ℝ × ℝ := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
def expression2 : ℝ × ℝ := (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2)

-- Dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Perpendicular condition
theorem perpendicular_condition : (k : ℝ) → dot_product (expression1 k) expression2 = 0 → k = 19 :=
by sorry

-- Parallel and opposite condition
theorem parallel_condition : (k : ℝ) → (∃ m : ℝ, expression1 k = m • expression2) → k = -1 / 3 :=
by sorry

noncomputable def m (k : ℝ) : ℝ × ℝ := 
  let ex1 := expression1 k
  let ex2 := expression2
  (ex2.1 / ex1.1, ex2.2 / ex1.2)

theorem parallel_opposite_direction : (k : ℝ) → expression1 k = -1 / 3 • expression2 → k = -1 / 3 :=
by sorry

end perpendicular_condition_parallel_condition_parallel_opposite_direction_l248_248010


namespace wire_length_before_cutting_l248_248992

theorem wire_length_before_cutting (L S : ℝ) (h1 : S = 40) (h2 : S = 2 / 5 * L) : L + S = 140 :=
by
  sorry

end wire_length_before_cutting_l248_248992


namespace difference_students_rabbits_l248_248739

-- Define the number of students per classroom
def students_per_classroom := 22

-- Define the number of rabbits per classroom
def rabbits_per_classroom := 4

-- Define the number of classrooms
def classrooms := 6

-- Calculate the total number of students
def total_students := students_per_classroom * classrooms

-- Calculate the total number of rabbits
def total_rabbits := rabbits_per_classroom * classrooms

-- Prove the difference between the number of students and rabbits is 108
theorem difference_students_rabbits : total_students - total_rabbits = 108 := by
  sorry

end difference_students_rabbits_l248_248739


namespace add_fractions_l248_248498

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l248_248498


namespace watched_videos_correct_l248_248404

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end watched_videos_correct_l248_248404


namespace roots_quadratic_l248_248335

theorem roots_quadratic (a b : ℝ) (h₁ : a + b = 6) (h₂ : a * b = 8) :
  a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 :=
by
  sorry

end roots_quadratic_l248_248335


namespace non_sibling_probability_l248_248638

-- Conditions outlined in the problem
def num_people : ℕ := 6
def num_sibling_sets : ℕ := 3
def siblings_per_set : ℕ := 2

-- Function to calculate combinations
def combinations (n k : ℕ) : ℕ := n.choose k

-- Main theorem statement
theorem non_sibling_probability : 
  let total_ways := combinations num_people 2,
      sibling_ways := num_sibling_sets,
      non_sibling_ways := total_ways - sibling_ways in
  (rat.mk non_sibling_ways total_ways) = rat.mk 4 5 := sorry

end non_sibling_probability_l248_248638


namespace determinant_A_l248_248862

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 5], ![0, 4, -2], ![3, 0, 1]]

theorem determinant_A : Matrix.det A = -46 := by
  sorry

end determinant_A_l248_248862


namespace tax_increase_proof_l248_248127

variables (old_tax_rate new_tax_rate : ℝ) (old_income new_income : ℝ)

def old_taxes_paid (old_tax_rate old_income : ℝ) : ℝ := old_tax_rate * old_income

def new_taxes_paid (new_tax_rate new_income : ℝ) : ℝ := new_tax_rate * new_income

def increase_in_taxes (old_tax_rate new_tax_rate old_income new_income : ℝ) : ℝ :=
  new_taxes_paid new_tax_rate new_income - old_taxes_paid old_tax_rate old_income

theorem tax_increase_proof :
  increase_in_taxes 0.20 0.30 1000000 1500000 = 250000 := by
  sorry

end tax_increase_proof_l248_248127


namespace R_and_D_per_increase_l248_248271

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end R_and_D_per_increase_l248_248271


namespace ratio_M_N_l248_248217

variables {M Q P N R : ℝ}

-- Conditions
def condition1 : M = 0.40 * Q := sorry
def condition2 : Q = 0.25 * P := sorry
def condition3 : N = 0.75 * R := sorry
def condition4 : R = 0.60 * P := sorry

-- Theorem to prove
theorem ratio_M_N : M / N = 2 / 9 := sorry

end ratio_M_N_l248_248217


namespace meena_cookies_left_l248_248654

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l248_248654


namespace upper_base_length_l248_248172

-- Definitions based on the given conditions
variables (A B C D M : ℝ)
variables (d : ℝ)
variables (h : ℝ) -- height from D to AB

-- Conditions given in the problem
def is_trapezoid (ABCD : ℝ) : Prop := true
def is_perpendicular (DM AB : ℝ) : Prop := true
def point_on_side (M AB : ℝ) : Prop := true
def MC_eq_CD (MC CD : ℝ) : Prop := MC = CD
def AD_length (A D d : ℝ) : Prop := A - D = d -- Assuming some coordinate system 

-- Define the proof statement with conditions and the result
theorem upper_base_length
  (ht : is_trapezoid ABCD)
  (hp : is_perpendicular D M)
  (ps : point_on_side M AB)
  (mc_cd : MC_eq_CD M C D)
  (ad_len : AD_length A D d) :
  BC = d / 2 :=
sorry

end upper_base_length_l248_248172


namespace Zilla_savings_l248_248110

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l248_248110


namespace side_length_of_regular_pentagon_l248_248828

theorem side_length_of_regular_pentagon (perimeter : ℝ) (number_of_sides : ℕ) (h1 : perimeter = 23.4) (h2 : number_of_sides = 5) : 
  perimeter / number_of_sides = 4.68 :=
by
  sorry

end side_length_of_regular_pentagon_l248_248828


namespace min_value_fraction_l248_248169

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 
  ( (x + 1) * (y + 1) / (x * y) ) >= 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_fraction_l248_248169


namespace graph_not_through_third_quadrant_l248_248682

theorem graph_not_through_third_quadrant (k : ℝ) (h_nonzero : k ≠ 0) (h_decreasing : k < 0) : 
  ¬(∃ x y : ℝ, y = k * x - k ∧ x < 0 ∧ y < 0) :=
sorry

end graph_not_through_third_quadrant_l248_248682


namespace good_apples_count_l248_248145

def total_apples : ℕ := 14
def unripe_apples : ℕ := 6

theorem good_apples_count : total_apples - unripe_apples = 8 :=
by
  unfold total_apples unripe_apples
  sorry

end good_apples_count_l248_248145


namespace root_in_interval_l248_248824

noncomputable def f (x: ℝ) : ℝ := x^2 + (Real.log x) - 4

theorem root_in_interval : 
  (∃ ξ ∈ Set.Ioo 1 2, f ξ = 0) :=
by
  sorry

end root_in_interval_l248_248824


namespace problem_solution_l248_248616

noncomputable def a_sequence : ℕ → ℕ := sorry
noncomputable def S_n : ℕ → ℕ := sorry
noncomputable def b_sequence : ℕ → ℕ := sorry
noncomputable def c_sequence : ℕ → ℕ := sorry
noncomputable def T_n : ℕ → ℕ := sorry

theorem problem_solution (n : ℕ) (a_condition : ∀ n : ℕ, 2 * S_n = (n + 1) ^ 2 * a_sequence n - n ^ 2 * a_sequence (n + 1))
                        (b_condition : ∀ n : ℕ, b_sequence 1 = a_sequence 1 ∧ (n ≠ 0 → n * b_sequence (n + 1) = a_sequence n * b_sequence n)) :
  (∀ n, a_sequence n = 2 * n) ∧
  (∀ n, b_sequence n = 2 ^ n) ∧
  (∀ n, T_n n = 2 ^ (n + 1) + n ^ 2 + n - 2) :=
sorry


end problem_solution_l248_248616


namespace number_of_divisors_23232_l248_248953

theorem number_of_divisors_23232 : ∀ (n : ℕ), 
    n = 23232 → 
    (∃ k : ℕ, k = 42 ∧ (∀ d : ℕ, (d > 0 ∧ d ∣ n) → (↑d < k + 1))) :=
by
  sorry

end number_of_divisors_23232_l248_248953


namespace total_books_is_correct_l248_248403

-- Definitions based on the conditions
def initial_books_benny : Nat := 24
def books_given_to_sandy : Nat := 10
def books_tim : Nat := 33

-- Definition based on the computation in the solution
def books_benny_now := initial_books_benny - books_given_to_sandy
def total_books : Nat := books_benny_now + books_tim

-- The statement to be proven
theorem total_books_is_correct : total_books = 47 := by
  sorry

end total_books_is_correct_l248_248403


namespace parallel_lines_slope_l248_248304

theorem parallel_lines_slope (m : ℝ) :
  let l1 := (m - 2) * x - 3 * y - 1 = 0,
      l2 := m * x + (m + 2) * y + 1 = 0 in
  (∀ (x y : ℝ), l1 = 0 → l2 = 0) → m = -4 :=
by
  sorry

end parallel_lines_slope_l248_248304


namespace find_function_solution_l248_248742

noncomputable def function_solution (f : ℤ → ℤ) : Prop :=
∀ x y : ℤ, x ≠ 0 → x * f (2 * f y - x) + y^2 * f (2 * x - f y) = f x ^ 2 / x + f (y * f y)

theorem find_function_solution : 
  ∀ f : ℤ → ℤ, function_solution f → (∀ x : ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end find_function_solution_l248_248742


namespace fraction_addition_l248_248539

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l248_248539


namespace boat_stream_speed_l248_248976

theorem boat_stream_speed (v : ℝ) (h : (60 / (15 - v)) - (60 / (15 + v)) = 2) : v = 3.5 := 
by 
  sorry
 
end boat_stream_speed_l248_248976


namespace total_heads_l248_248982

def number_of_heads := 1
def number_of_feet_hen := 2
def number_of_feet_cow := 4
def total_feet := 144

theorem total_heads (H C : ℕ) (h_hens : H = 24) (h_feet : number_of_feet_hen * H + number_of_feet_cow * C = total_feet) :
  H + C = 48 :=
sorry

end total_heads_l248_248982


namespace students_without_favorite_subject_l248_248783

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end students_without_favorite_subject_l248_248783


namespace fraction_addition_l248_248549

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l248_248549


namespace sticks_form_equilateral_triangle_l248_248158

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l248_248158


namespace bamboo_break_height_l248_248975

theorem bamboo_break_height (x : ℝ) (h₁ : 0 < x) (h₂ : x < 9) (h₃ : x^2 + 3^2 = (9 - x)^2) : x = 4 :=
by
  sorry

end bamboo_break_height_l248_248975


namespace smallest_circle_radius_l248_248096

-- Define the problem as a proposition
theorem smallest_circle_radius (r : ℝ) (R1 R2 : ℝ) (hR1 : R1 = 6) (hR2 : R2 = 4) (h_right_triangle : (r + R2)^2 + (r + R1)^2 = (R2 + R1)^2) : r = 2 := 
sorry

end smallest_circle_radius_l248_248096


namespace fraction_addition_l248_248571

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l248_248571


namespace tangent_line_at_point_l248_248365

theorem tangent_line_at_point (x y : ℝ) (h : y = Real.exp x) (t : x = 2) :
  y = Real.exp 2 * x - 2 * Real.exp 2 :=
by sorry

end tangent_line_at_point_l248_248365


namespace set_union_example_l248_248308

variable (A B : Set ℝ)

theorem set_union_example :
  A = {x | -2 < x ∧ x ≤ 1} ∧ B = {x | -1 ≤ x ∧ x < 2} →
  (A ∪ B) = {x | -2 < x ∧ x < 2} := 
by
  sorry

end set_union_example_l248_248308


namespace add_fractions_result_l248_248409

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l248_248409


namespace minimum_sum_of_areas_l248_248278

theorem minimum_sum_of_areas (x y : ℝ) (hx : x + y = 16) (hx_nonneg : 0 ≤ x) (hy_nonneg : 0 ≤ y) : 
  (x ^ 2 / 16 + y ^ 2 / 16) / 4 ≥ 8 :=
  sorry

end minimum_sum_of_areas_l248_248278


namespace volume_of_solid_l248_248137

noncomputable def s : ℝ := 2 * Real.sqrt 2

noncomputable def h : ℝ := 3 * s

noncomputable def base_area (a b : ℝ) : ℝ := 1 / 2 * a * b

noncomputable def volume (base_area height : ℝ) : ℝ := base_area * height

theorem volume_of_solid : volume (base_area s s) h = 24 * Real.sqrt 2 :=
by
  -- The proof will go here
  sorry

end volume_of_solid_l248_248137


namespace negate_proposition_l248_248619

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- The original proposition p
def p : Prop := ∀ x, is_odd x → is_even (2 * x)

-- The negation of the proposition p
def neg_p : Prop := ∃ x, is_odd x ∧ ¬ is_even (2 * x)

-- Proof problem statement: Prove that the negation of proposition p is as defined in neg_p
theorem negate_proposition :
  (∀ x, is_odd x → is_even (2 * x)) ↔ (∃ x, is_odd x ∧ ¬ is_even (2 * x)) :=
sorry

end negate_proposition_l248_248619


namespace add_fractions_l248_248576

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l248_248576


namespace sum_fractions_eq_l248_248482

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l248_248482


namespace tom_age_ratio_l248_248694

theorem tom_age_ratio (T N : ℕ) (h1 : sum_ages = T) (h2 : T - N = 3 * (sum_ages_N_years_ago))
  (h3 : sum_ages = T) (h4 : sum_ages_N_years_ago = T - 4 * N) :
  T / N = 11 / 2 := 
by
  sorry

end tom_age_ratio_l248_248694


namespace fraction_addition_l248_248463

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l248_248463


namespace karl_present_salary_l248_248918

def original_salary : ℝ := 20000
def reduction_percentage : ℝ := 0.10
def increase_percentage : ℝ := 0.10

theorem karl_present_salary :
  let reduced_salary := original_salary * (1 - reduction_percentage)
  let present_salary := reduced_salary * (1 + increase_percentage)
  present_salary = 19800 :=
by
  sorry

end karl_present_salary_l248_248918


namespace mark_hourly_wage_before_raise_40_l248_248062

-- Mark's hourly wage before the raise
def hourly_wage_before_raise (x : ℝ) : Prop :=
  let weekly_hours := 40
  let raise_percentage := 0.05
  let new_hourly_wage := x * (1 + raise_percentage)
  let new_weekly_earnings := weekly_hours * new_hourly_wage
  let old_bills := 600
  let personal_trainer := 100
  let new_expenses := old_bills + personal_trainer
  let leftover_income := 980
  new_weekly_earnings = new_expenses + leftover_income

-- Proving that Mark's hourly wage before the raise was 40 dollars
theorem mark_hourly_wage_before_raise_40 : hourly_wage_before_raise 40 :=
by
  -- Proof goes here
  sorry

end mark_hourly_wage_before_raise_40_l248_248062


namespace cost_condition_shirt_costs_purchasing_plans_maximize_profit_l248_248100

/-- Define the costs and prices of shirts A and B -/
def cost_A (m : ℝ) : ℝ := m
def cost_B (m : ℝ) : ℝ := m - 10
def price_A : ℝ := 260
def price_B : ℝ := 180

/-- Condition: total cost of 3 A shirts and 2 B shirts is 480 -/
theorem cost_condition (m : ℝ) : 3 * (cost_A m) + 2 * (cost_B m) = 480 := by
  sorry

/-- The cost of each A shirt is 100 and each B shirt is 90 -/
theorem shirt_costs : ∃ m, cost_A m = 100 ∧ cost_B m = 90 := by
  sorry

/-- Number of purchasing plans for at least $34,000 profit with 300 shirts and at most 110 A shirts -/
theorem purchasing_plans : ∃ x, 100 ≤ x ∧ x ≤ 110 ∧ 
  (260 * x + 180 * (300 - x) - 100 * x - 90 * (300 - x) ≥ 34000) := by
  sorry

/- Maximize profit given 60 < a < 80:
   - 60 < a < 70: 110 A shirts, 190 B shirts.
   - a = 70: any combination satisfying conditions.
   - 70 < a < 80: 100 A shirts, 200 B shirts. -/

theorem maximize_profit (a : ℝ) (ha : 60 < a ∧ a < 80) : 
  ∃ x, ((60 < a ∧ a < 70 ∧ x = 110 ∧ (300 - x) = 190) ∨ 
        (a = 70) ∨ 
        (70 < a ∧ a < 80 ∧ x = 100 ∧ (300 - x) = 200)) := by
  sorry

end cost_condition_shirt_costs_purchasing_plans_maximize_profit_l248_248100


namespace max_quotient_l248_248902

theorem max_quotient (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : ∃ q, q = b / a ∧ q ≤ 16 / 3 :=
by 
  sorry

end max_quotient_l248_248902


namespace total_people_at_beach_l248_248660

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l248_248660
