import Mathlib

namespace total_crayons_l983_98357

theorem total_crayons (crayons_per_child : ℕ) (number_of_children : ℕ) (h1 : crayons_per_child = 3) (h2 : number_of_children = 6) : 
  crayons_per_child * number_of_children = 18 := by
  sorry

end total_crayons_l983_98357


namespace find_x_l983_98355

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (x, 2)
def b : vector := (1, -1)

-- Dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Orthogonality condition rewritten in terms of dot product
def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

-- Main theorem to prove
theorem find_x (x : ℝ) (h : orthogonal ((a x).1 - b.1, (a x).2 - b.2) b) : x = 4 :=
by sorry

end find_x_l983_98355


namespace max_value_of_expression_l983_98396

theorem max_value_of_expression (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  x^2 * y + x^2 * z + y * z^2 ≤ 3 := sorry

end max_value_of_expression_l983_98396


namespace circle_center_and_radius_locus_of_midpoint_l983_98368

-- Part 1: Prove the equation of the circle C:
theorem circle_center_and_radius (a b r: ℝ) (hc: a + b = 2):
  (4 - a)^2 + b^2 = r^2 →
  (2 - a)^2 + (2 - b)^2 = r^2 →
  a = 2 ∧ b = 0 ∧ r = 2 := by
  sorry

-- Part 2: Prove the locus of the midpoint M:
theorem locus_of_midpoint (x y : ℝ) :
  ∃ (x1 y1 : ℝ), (x1 - 2)^2 + y1^2 = 4 ∧ x = (x1 + 5) / 2 ∧ y = y1 / 2 →
  x^2 - 7*x + y^2 + 45/4 = 0 := by
  sorry

end circle_center_and_radius_locus_of_midpoint_l983_98368


namespace cricket_initial_avg_runs_l983_98327

theorem cricket_initial_avg_runs (A : ℝ) (h : 11 * (A + 4) = 10 * A + 86) : A = 42 :=
sorry

end cricket_initial_avg_runs_l983_98327


namespace seats_shortage_l983_98393

-- Definitions of the conditions
def children := 52
def adults := 29
def seniors := 15
def pets := 3
def total_seats := 95

-- Theorem statement to prove the number of people and pets without seats
theorem seats_shortage : children + adults + seniors + pets - total_seats = 4 :=
by
  sorry

end seats_shortage_l983_98393


namespace algebraic_expression_value_l983_98373

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : -2 * a^2 + 8 * a - 5 = 1 := 
by 
  sorry 

end algebraic_expression_value_l983_98373


namespace part_length_proof_l983_98304

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end part_length_proof_l983_98304


namespace smallest_base_l983_98365

theorem smallest_base : ∃ b : ℕ, (b^2 ≤ 120 ∧ 120 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 120 ∧ 120 < n^3) → b ≤ n :=
by sorry

end smallest_base_l983_98365


namespace shooting_guard_seconds_l983_98321

-- Define the given conditions
def x_pg := 130
def x_sf := 85
def x_pf := 60
def x_c := 180
def avg_time_per_player := 120
def total_players := 5

-- Define the total footage
def total_footage : Nat := total_players * avg_time_per_player

-- Define the footage for four players
def footage_of_four : Nat := x_pg + x_sf + x_pf + x_c

-- Define the footage of the shooting guard, which is a variable we want to compute
def x_sg := total_footage - footage_of_four

-- The statement we want to prove
theorem shooting_guard_seconds :
  x_sg = 145 := by
  sorry

end shooting_guard_seconds_l983_98321


namespace polynomial_transformable_l983_98375

theorem polynomial_transformable (a b c d : ℝ) :
  (∃ A B : ℝ, ∀ z : ℝ, z^4 + A * z^2 + B = (z + a/4)^4 + a * (z + a/4)^3 + b * (z + a/4)^2 + c * (z + a/4) + d) ↔ a^3 - 4 * a * b + 8 * c = 0 :=
by
  sorry

end polynomial_transformable_l983_98375


namespace radian_measure_of_neg_300_degrees_l983_98397

theorem radian_measure_of_neg_300_degrees : (-300 : ℝ) * (Real.pi / 180) = -5 * Real.pi / 3 :=
by
  sorry

end radian_measure_of_neg_300_degrees_l983_98397


namespace find_teacher_age_l983_98362

theorem find_teacher_age (S T : ℕ) (h1 : S / 19 = 20) (h2 : (S + T) / 20 = 21) : T = 40 :=
sorry

end find_teacher_age_l983_98362


namespace solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l983_98387

def solve_inequality (a x : ℝ) : Prop :=
  a^2 * x - 6 < 4 * x + 3 * a

theorem solution_set_a_eq_2 :
  ∀ x : ℝ, solve_inequality 2 x ↔ true :=
sorry

theorem solution_set_a_eq_neg_2 :
  ∀ x : ℝ, ¬ solve_inequality (-2) x :=
sorry

theorem solution_set_neg_2_lt_a_lt_2 (a : ℝ) (h : -2 < a ∧ a < 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x > 3 / (a - 2) :=
sorry

theorem solution_set_a_lt_neg_2_or_a_gt_2 (a : ℝ) (h : a < -2 ∨ a > 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x < 3 / (a - 2) :=
sorry

end solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l983_98387


namespace circumscribed_circle_area_l983_98300

/-- 
Statement: The area of the circle circumscribed about an equilateral triangle with side lengths of 9 units is 27π square units.
-/
theorem circumscribed_circle_area (s : ℕ) (h : s = 9) : 
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)
  area = 27 * Real.pi :=
by
  -- Axis and conditions definitions
  have := h

  -- Definition for the area based on the radius
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)

  -- Statement of the equality to be proven
  show area = 27 * Real.pi
  sorry

end circumscribed_circle_area_l983_98300


namespace remainder_of_sum_divided_by_14_l983_98322

def consecutive_odds : List ℤ := [12157, 12159, 12161, 12163, 12165, 12167, 12169]

def sum_of_consecutive_odds := consecutive_odds.sum

theorem remainder_of_sum_divided_by_14 :
  (sum_of_consecutive_odds % 14) = 7 := by
  sorry

end remainder_of_sum_divided_by_14_l983_98322


namespace sqrt_of_square_neg_l983_98391

variable {a : ℝ}

theorem sqrt_of_square_neg (h : a < 0) : Real.sqrt (a^2) = -a := 
sorry

end sqrt_of_square_neg_l983_98391


namespace probability_of_D_l983_98354

theorem probability_of_D (pA pB pC pD : ℚ)
  (hA : pA = 1/4)
  (hB : pB = 1/3)
  (hC : pC = 1/6)
  (hTotal : pA + pB + pC + pD = 1) : pD = 1/4 :=
by
  have hTotal_before_D : pD = 1 - (pA + pB + pC) := by sorry
  sorry

end probability_of_D_l983_98354


namespace negation_of_exists_l983_98320

theorem negation_of_exists : (¬ ∃ x : ℝ, x > 0 ∧ x^2 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 ≤ 0 :=
by sorry

end negation_of_exists_l983_98320


namespace sum_of_squares_ge_one_third_l983_98313

theorem sum_of_squares_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1/3 := 
by 
  sorry

end sum_of_squares_ge_one_third_l983_98313


namespace Sandy_age_l983_98361

variable (S M : ℕ)

def condition1 (S M : ℕ) : Prop := M = S + 18
def condition2 (S M : ℕ) : Prop := S * 9 = M * 7

theorem Sandy_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 63 := sorry

end Sandy_age_l983_98361


namespace factorial_expression_l983_98337

open Nat

theorem factorial_expression :
  7 * (6!) + 6 * (5!) + 2 * (5!) = 6000 :=
by
  sorry

end factorial_expression_l983_98337


namespace determine_m_l983_98342

theorem determine_m (x y m : ℝ) 
  (h1 : 3 * x + 2 * y = 4 * m - 5) 
  (h2 : 2 * x + 3 * y = m) 
  (h3 : x + y = 2) : 
  m = 3 :=
sorry

end determine_m_l983_98342


namespace inverse_proposition_l983_98324

theorem inverse_proposition (q_1 q_2 : ℚ) :
  (q_1 ^ 2 = q_2 ^ 2 → q_1 = q_2) ↔ (q_1 = q_2 → q_1 ^ 2 = q_2 ^ 2) :=
sorry

end inverse_proposition_l983_98324


namespace pair_opposites_example_l983_98331

theorem pair_opposites_example :
  (-5)^2 = 25 ∧ -((5)^2) = -25 →
  (∀ a b : ℕ, (|-4|)^2 = 4^2 → 4^2 = 16 → |-4|^2 = 16) →
  (-3)^2 = 9 ∧ 3^2 = 9 →
  (-(|-2|)^2 = -4 ∧ -2^2 = -4) →
  25 = -(-25) :=
by
  sorry

end pair_opposites_example_l983_98331


namespace smallest_perfect_square_4_10_18_l983_98392

theorem smallest_perfect_square_4_10_18 :
  ∃ n : ℕ, (∃ k : ℕ, n = k^2) ∧ (4 ∣ n) ∧ (10 ∣ n) ∧ (18 ∣ n) ∧ n = 900 := 
  sorry

end smallest_perfect_square_4_10_18_l983_98392


namespace sum_of_coefficients_l983_98395

noncomputable def simplify (x : ℝ) : ℝ := 
  (x^3 + 11 * x^2 + 38 * x + 40) / (x + 3)

theorem sum_of_coefficients : 
  (∀ x : ℝ, (x ≠ -3) → (simplify x = x^2 + 8 * x + 14)) ∧
  (1 + 8 + 14 + -3 = 20) :=
by      
  sorry

end sum_of_coefficients_l983_98395


namespace midpoint_coords_l983_98371

noncomputable def F1 : (ℝ × ℝ) := (-2 * Real.sqrt 2, 0)
noncomputable def F2 : (ℝ × ℝ) := (2 * Real.sqrt 2, 0)
def major_axis_length : ℝ := 6
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  let a := 3
  let b := 1
  (x^2) / (a^2) + y^2 / (b^2) = 1

theorem midpoint_coords :
  ∃ (A B : ℝ × ℝ), ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 →
  (A.1 + B.1) / 2 = -9 / 5 ∧ (A.2 + B.2) / 2 = 1 / 5 :=
by
  sorry

end midpoint_coords_l983_98371


namespace cube_inequality_sufficient_and_necessary_l983_98367

theorem cube_inequality_sufficient_and_necessary (a b : ℝ) :
  (a > b ↔ a^3 > b^3) := 
sorry

end cube_inequality_sufficient_and_necessary_l983_98367


namespace jaco_payment_l983_98394

theorem jaco_payment :
  let cost_shoes : ℝ := 74
  let cost_socks : ℝ := 2 * 2
  let cost_bag : ℝ := 42
  let total_cost_before_discount : ℝ := cost_shoes + cost_socks + cost_bag
  let discount_threshold : ℝ := 100
  let discount_rate : ℝ := 0.10
  let amount_exceeding_threshold : ℝ := total_cost_before_discount - discount_threshold
  let discount : ℝ := if amount_exceeding_threshold > 0 then discount_rate * amount_exceeding_threshold else 0
  let final_amount : ℝ := total_cost_before_discount - discount
  final_amount = 118 :=
by
  sorry

end jaco_payment_l983_98394


namespace initial_jellybeans_l983_98336

theorem initial_jellybeans (J : ℕ) :
    (∀ x y : ℕ, x = 24 → y = 12 →
    (J - x - y + ((x + y) / 2) = 72) → J = 90) :=
by
  intros x y hx hy h
  rw [hx, hy] at h
  sorry

end initial_jellybeans_l983_98336


namespace train_cross_time_in_seconds_l983_98339

-- Definitions based on conditions
def train_speed_kph : ℚ := 60
def train_length_m : ℚ := 450

-- Statement: prove that the time to cross the pole is 27 seconds
theorem train_cross_time_in_seconds (train_speed_kph train_length_m : ℚ) :
  train_speed_kph = 60 →
  train_length_m = 450 →
  (train_length_m / (train_speed_kph * 1000 / 3600)) = 27 :=
by
  intros h_speed h_length
  rw [h_speed, h_length]
  sorry

end train_cross_time_in_seconds_l983_98339


namespace student_departments_l983_98399

variable {Student : Type}
variable (Anna Vika Masha : Student)

-- Let Department be an enumeration type representing the three departments
inductive Department
| Literature : Department
| History : Department
| Biology : Department

open Department

variables (isLit : Student → Prop) (isHist : Student → Prop) (isBio : Student → Prop)

-- Conditions
axiom cond1 : isLit Anna → ¬isHist Masha
axiom cond2 : ¬isHist Vika → isLit Anna
axiom cond3 : ¬isLit Masha → isBio Vika

-- Target conclusion
theorem student_departments :
  isHist Vika ∧ isLit Masha ∧ isBio Anna :=
sorry

end student_departments_l983_98399


namespace residue_calculation_l983_98312

theorem residue_calculation :
  (196 * 18 - 21 * 9 + 5) % 18 = 14 := 
by 
  sorry

end residue_calculation_l983_98312


namespace value_of_expression_l983_98348

theorem value_of_expression (a b c : ℝ) (h : a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7) :
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 :=
by sorry

end value_of_expression_l983_98348


namespace represent_same_function_l983_98341

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end represent_same_function_l983_98341


namespace sum_of_three_numbers_l983_98338

theorem sum_of_three_numbers (x y z : ℝ) (h1 : x + y = 31) (h2 : y + z = 41) (h3 : z + x = 55) :
  x + y + z = 63.5 :=
by
  sorry

end sum_of_three_numbers_l983_98338


namespace center_radius_sum_l983_98383

theorem center_radius_sum (a b r : ℝ) (h : ∀ x y : ℝ, (x^2 - 8*x - 4*y = -y^2 + 2*y + 13) ↔ (x - 4)^2 + (y - 3)^2 = 38) :
  a = 4 ∧ b = 3 ∧ r = Real.sqrt 38 → a + b + r = 7 + Real.sqrt 38 :=
by
  sorry

end center_radius_sum_l983_98383


namespace circle_passing_origin_l983_98364

theorem circle_passing_origin (a b r : ℝ) :
  ((a^2 + b^2 = r^2) ↔ (∃ (x y : ℝ), (x-a)^2 + (y-b)^2 = r^2 ∧ x = 0 ∧ y = 0)) :=
by
  sorry

end circle_passing_origin_l983_98364


namespace a_plus_b_plus_c_at_2_l983_98326

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def maximum_value (a b c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic a b c x = 75

def passes_through (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  quadratic a b c p1.1 = p1.2 ∧ quadratic a b c p2.1 = p2.2

theorem a_plus_b_plus_c_at_2 
  (a b c : ℝ)
  (hmax : maximum_value a b c)
  (hpoints : passes_through a b c (-3, 0) (3, 0))
  (hvertex : ∀ x : ℝ, quadratic a 0 c x ≤ quadratic a (2 * b) c 0) : 
  quadratic a b c 2 = 125 / 3 :=
sorry

end a_plus_b_plus_c_at_2_l983_98326


namespace equal_powers_equal_elements_l983_98374

theorem equal_powers_equal_elements
  (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) % 17 = a ((i + 1) % 17) ^ a ((i + 2) % 17) % 17)
  : ∀ i j : Fin 17, a i = a j :=
by
  sorry

end equal_powers_equal_elements_l983_98374


namespace lcm_two_primes_is_10_l983_98305

theorem lcm_two_primes_is_10 (x y : ℕ) (h_prime_x : Nat.Prime x) (h_prime_y : Nat.Prime y) (h_lcm : Nat.lcm x y = 10) (h_gt : x > y) : 2 * x + y = 12 :=
sorry

end lcm_two_primes_is_10_l983_98305


namespace count_three_digit_integers_with_remainder_3_div_7_l983_98345

theorem count_three_digit_integers_with_remainder_3_div_7 :
  ∃ n, (100 ≤ 7 * n + 3 ∧ 7 * n + 3 < 1000) ∧
  ∀ m, (100 ≤ 7 * m + 3 ∧ 7 * m + 3 < 1000) → m - n < 142 - 14 + 1 :=
by
  sorry

end count_three_digit_integers_with_remainder_3_div_7_l983_98345


namespace jacobs_hourly_wage_l983_98370

theorem jacobs_hourly_wage (jake_total_earnings : ℕ) (jake_days : ℕ) (hours_per_day : ℕ) (jake_thrice_jacob : ℕ) 
    (h_total_jake : jake_total_earnings = 720) 
    (h_jake_days : jake_days = 5) 
    (h_hours_per_day : hours_per_day = 8)
    (h_jake_thrice_jacob : jake_thrice_jacob = 3) 
    (jacob_hourly_wage : ℕ) :
  jacob_hourly_wage = 6 := 
by
  sorry

end jacobs_hourly_wage_l983_98370


namespace line_equation_l983_98352

-- Given conditions
variables (k x x0 y y0 : ℝ)
variable (line_passes_through : ∀ x0 y0, y0 = k * x0 + l)
variable (M0 : (ℝ × ℝ))

-- Main statement we need to prove
theorem line_equation (k x x0 y y0 : ℝ) (M0 : (ℝ × ℝ)) (line_passes_through : ∀ x0 y0, y0 = k * x0 + l) :
  y - y0 = k * (x - x0) :=
sorry

end line_equation_l983_98352


namespace apples_more_than_oranges_l983_98366

-- Definitions based on conditions
def total_fruits : ℕ := 301
def apples : ℕ := 164

-- Statement to prove
theorem apples_more_than_oranges : (apples - (total_fruits - apples)) = 27 :=
by
  sorry

end apples_more_than_oranges_l983_98366


namespace fewerEmployeesAbroadThanInKorea_l983_98351

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end fewerEmployeesAbroadThanInKorea_l983_98351


namespace tug_of_war_matches_l983_98309

-- Define the number of classes
def num_classes : ℕ := 7

-- Define the number of matches Grade 3 Class 6 competes in
def matches_class6 : ℕ := num_classes - 1

-- Define the total number of matches
def total_matches : ℕ := (num_classes - 1) * num_classes / 2

-- Main theorem stating the problem
theorem tug_of_war_matches :
  matches_class6 = 6 ∧ total_matches = 21 := by
  sorry

end tug_of_war_matches_l983_98309


namespace hole_digging_problem_l983_98382

theorem hole_digging_problem
  (total_distance : ℕ)
  (original_interval : ℕ)
  (new_interval : ℕ)
  (original_holes : ℕ)
  (new_holes : ℕ)
  (lcm_interval : ℕ)
  (common_holes : ℕ)
  (new_holes_to_be_dug : ℕ)
  (original_holes_discarded : ℕ)
  (h1 : total_distance = 3000)
  (h2 : original_interval = 50)
  (h3 : new_interval = 60)
  (h4 : original_holes = total_distance / original_interval + 1)
  (h5 : new_holes = total_distance / new_interval + 1)
  (h6 : lcm_interval = Nat.lcm original_interval new_interval)
  (h7 : common_holes = total_distance / lcm_interval + 1)
  (h8 : new_holes_to_be_dug = new_holes - common_holes)
  (h9 : original_holes_discarded = original_holes - common_holes) :
  new_holes_to_be_dug = 40 ∧ original_holes_discarded = 50 :=
sorry

end hole_digging_problem_l983_98382


namespace average_price_of_pig_l983_98332

theorem average_price_of_pig :
  ∀ (total_cost total_cost_hens total_cost_pigs : ℕ) (num_hens num_pigs avg_price_hen avg_price_pig : ℕ),
  num_hens = 10 →
  num_pigs = 3 →
  total_cost = 1200 →
  avg_price_hen = 30 →
  total_cost_hens = num_hens * avg_price_hen →
  total_cost_pigs = total_cost - total_cost_hens →
  avg_price_pig = total_cost_pigs / num_pigs →
  avg_price_pig = 300 :=
by
  intros total_cost total_cost_hens total_cost_pigs num_hens num_pigs avg_price_hen avg_price_pig h_num_hens h_num_pigs h_total_cost h_avg_price_hen h_total_cost_hens h_total_cost_pigs h_avg_price_pig
  sorry

end average_price_of_pig_l983_98332


namespace multiply_powers_same_base_l983_98356

theorem multiply_powers_same_base (a : ℝ) : a^3 * a = a^4 :=
by
  sorry

end multiply_powers_same_base_l983_98356


namespace amount_paid_correct_l983_98378

-- Defining the conditions and constants
def hourly_rate : ℕ := 60
def hours_per_day : ℕ := 3
def total_days : ℕ := 14

-- The proof statement
theorem amount_paid_correct : hourly_rate * hours_per_day * total_days = 2520 := by
  sorry

end amount_paid_correct_l983_98378


namespace vicentes_total_cost_l983_98389

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end vicentes_total_cost_l983_98389


namespace daily_earnings_from_oil_refining_l983_98353

-- Definitions based on conditions
def daily_earnings_from_mining : ℝ := 3000000
def monthly_expenses : ℝ := 30000000
def fine : ℝ := 25600000
def profit_percentage : ℝ := 0.01
def months_in_year : ℝ := 12
def days_in_month : ℝ := 30

-- The question translated as a Lean theorem statement
theorem daily_earnings_from_oil_refining : ∃ O : ℝ, O = 5111111.11 ∧ 
  fine = profit_percentage * months_in_year * 
    (days_in_month * (daily_earnings_from_mining + O) - monthly_expenses) :=
sorry

end daily_earnings_from_oil_refining_l983_98353


namespace male_students_outnumber_female_students_l983_98330

-- Define the given conditions
def total_students : ℕ := 928
def male_students : ℕ := 713
def female_students : ℕ := total_students - male_students

-- The theorem to be proven
theorem male_students_outnumber_female_students :
  male_students - female_students = 498 :=
by
  sorry

end male_students_outnumber_female_students_l983_98330


namespace linear_increase_y_l983_98360

-- Progressively increase x and track y

theorem linear_increase_y (Δx Δy : ℝ) (x_increase : Δx = 4) (y_increase : Δy = 10) :
  12 * (Δy / Δx) = 30 := by
  sorry

end linear_increase_y_l983_98360


namespace range_of_x_for_direct_above_inverse_l983_98359

-- The conditions
def is_intersection_point (p : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  let (x, y) := p
  y = k1 * x ∧ y = k2 / x

-- The main proof that we need to show
theorem range_of_x_for_direct_above_inverse :
  (∃ k1 k2 : ℝ, is_intersection_point (2, -1/3) k1 k2) →
  {x : ℝ | -1/6 * x > -2/(3 * x)} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by
  intros
  sorry

end range_of_x_for_direct_above_inverse_l983_98359


namespace range_of_m_l983_98376

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (x^2 : ℝ) / (2 - m) + (y^2 : ℝ) / (m - 1) = 1 → 2 - m < 0 ∧ m - 1 > 0) →
  (∀ Δ : ℝ, Δ = 16 * (m - 2) ^ 2 - 16 → Δ < 0 → 1 < m ∧ m < 3) →
  (∀ (p q : Prop), p ∨ q ∧ ¬ q → p ∧ ¬ q) →
  m ≥ 3 :=
by
  intros h1 h2 h3
  sorry

end range_of_m_l983_98376


namespace minimum_sticks_broken_n12_can_form_square_n15_l983_98381

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l983_98381


namespace tan_product_pi_8_l983_98380

theorem tan_product_pi_8 :
  (Real.tan (π / 8)) * (Real.tan (3 * π / 8)) * (Real.tan (5 * π / 8)) * (Real.tan (7 * π / 8)) = 1 :=
sorry

end tan_product_pi_8_l983_98380


namespace total_sales_15_days_l983_98329

def edgar_sales (n : ℕ) : ℕ := 3 * n - 1

def clara_sales (n : ℕ) : ℕ := 4 * n

def edgar_total_sales (d : ℕ) : ℕ := (d * (2 + (d * 3 - 1))) / 2

def clara_total_sales (d : ℕ) : ℕ := (d * (4 + (d * 4))) / 2

def total_sales (d : ℕ) : ℕ := edgar_total_sales d + clara_total_sales d

theorem total_sales_15_days : total_sales 15 = 810 :=
by
  sorry

end total_sales_15_days_l983_98329


namespace no_real_roots_of_polynomial_l983_98317

noncomputable def p (x : ℝ) : ℝ := sorry

theorem no_real_roots_of_polynomial (p : ℝ → ℝ) (h_deg : ∃ n : ℕ, n ≥ 1 ∧ ∀ x: ℝ, p x = x^n) :
  (∀ x, p x * p (2 * x^2) = p (3 * x^3 + x)) →
  ¬ ∃ α : ℝ, p α = 0 := sorry

end no_real_roots_of_polynomial_l983_98317


namespace total_yards_of_fabric_l983_98377

theorem total_yards_of_fabric (cost_checkered : ℝ) (cost_plain : ℝ) (price_per_yard : ℝ)
  (h1 : cost_checkered = 75) (h2 : cost_plain = 45) (h3 : price_per_yard = 7.50) :
  (cost_checkered / price_per_yard) + (cost_plain / price_per_yard) = 16 := 
by
  sorry

end total_yards_of_fabric_l983_98377


namespace initial_population_l983_98347

theorem initial_population (rate_decrease : ℝ) (population_after_2_years : ℝ) (P : ℝ) : 
  rate_decrease = 0.1 → 
  population_after_2_years = 8100 → 
  ((1 - rate_decrease) ^ 2) * P = population_after_2_years → 
  P = 10000 :=
by
  intros h1 h2 h3
  sorry

end initial_population_l983_98347


namespace students_on_couch_per_room_l983_98385

def total_students : ℕ := 30
def total_rooms : ℕ := 6
def students_per_bed : ℕ := 2
def beds_per_room : ℕ := 2
def students_in_beds_per_room : ℕ := beds_per_room * students_per_bed

theorem students_on_couch_per_room :
  (total_students / total_rooms) - students_in_beds_per_room = 1 := by
  sorry

end students_on_couch_per_room_l983_98385


namespace parakeets_per_cage_l983_98328

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (hcages : num_cages = 6) 
  (hparrots : parrots_per_cage = 6) 
  (htotal : total_birds = 48) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := 
  by
  sorry

end parakeets_per_cage_l983_98328


namespace box_weight_no_apples_l983_98340

variable (initialWeight : ℕ) (halfWeight : ℕ) (totalWeight : ℕ)
variable (boxWeight : ℕ)

-- Given conditions
axiom initialWeight_def : initialWeight = 9
axiom halfWeight_def : halfWeight = 5
axiom appleWeight_consistent : ∃ w : ℕ, ∀ n : ℕ, n * w = totalWeight

-- Question: How many kilograms does the empty box weigh?
theorem box_weight_no_apples : (initialWeight - totalWeight) = boxWeight :=
by
  -- The proof steps are omitted as indicated by the 'sorry' placeholder.
  sorry

end box_weight_no_apples_l983_98340


namespace max_ab_l983_98386

theorem max_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 4 * b = 8) :
  ab ≤ 4 :=
sorry

end max_ab_l983_98386


namespace problem_solution_l983_98301

variable (a b : ℝ)

theorem problem_solution (h : 2 * a - 3 * b = 5) : 4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 :=
sorry

end problem_solution_l983_98301


namespace polynomial_sequence_symmetric_l983_98316

def P : ℕ → ℝ → ℝ → ℝ → ℝ 
| 0, x, y, z => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_sequence_symmetric (m : ℕ) (x y z : ℝ) (σ : ℝ × ℝ × ℝ): 
  P m x y z = P m σ.1 σ.2.1 σ.2.2 :=
sorry

end polynomial_sequence_symmetric_l983_98316


namespace union_of_A_and_B_l983_98384

-- Define the sets A and B
def A := {x : ℝ | 0 < x ∧ x < 16}
def B := {y : ℝ | -1 < y ∧ y < 4}

-- Prove that A ∪ B = (-1, 16)
theorem union_of_A_and_B : A ∪ B = {z : ℝ | -1 < z ∧ z < 16} :=
by sorry

end union_of_A_and_B_l983_98384


namespace factorize_x4_minus_64_l983_98303

theorem factorize_x4_minus_64 (x : ℝ) : (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by sorry

end factorize_x4_minus_64_l983_98303


namespace problem_solution_l983_98349

noncomputable def circle_constant : ℝ := Real.pi
noncomputable def natural_base : ℝ := Real.exp 1

theorem problem_solution (π : ℝ) (e : ℝ) (h₁ : π = Real.pi) (h₂ : e = Real.exp 1) :
  π * Real.log e / Real.log 3 > 3 * Real.log e / Real.log π := by
  sorry

end problem_solution_l983_98349


namespace sale_in_fourth_month_l983_98398

-- Given conditions
def sales_first_month : ℕ := 5266
def sales_second_month : ℕ := 5768
def sales_third_month : ℕ := 5922
def sales_sixth_month : ℕ := 4937
def required_average_sales : ℕ := 5600
def number_of_months : ℕ := 6

-- Sum of the first, second, third, and sixth month's sales
def total_sales_without_fourth_fifth : ℕ := sales_first_month + sales_second_month + sales_third_month + sales_sixth_month

-- Total sales required to achieve the average required
def required_total_sales : ℕ := required_average_sales * number_of_months

-- The sale in the fourth month should be calculated as follows
def sales_fourth_month : ℕ := required_total_sales - total_sales_without_fourth_fifth

-- Proof statement
theorem sale_in_fourth_month :
  sales_fourth_month = 11707 := by
  sorry

end sale_in_fourth_month_l983_98398


namespace chickens_pigs_legs_l983_98369

variable (x : ℕ)

-- Define the conditions
def sum_chickens_pigs (x : ℕ) : Prop := x + (70 - x) = 70
def total_legs (x : ℕ) : Prop := 2 * x + 4 * (70 - x) = 196

-- Main theorem to prove the given mathematical statement
theorem chickens_pigs_legs (x : ℕ) (h1 : sum_chickens_pigs x) (h2 : total_legs x) : (2 * x + 4 * (70 - x) = 196) :=
by sorry

end chickens_pigs_legs_l983_98369


namespace first_term_geometric_progression_l983_98335

theorem first_term_geometric_progression (S : ℝ) (sum_first_two_terms : ℝ) (a : ℝ) (r : ℝ) :
  S = 8 → sum_first_two_terms = 5 →
  (a = 8 * (1 - (Real.sqrt 6) / 4)) ∨ (a = 8 * (1 + (Real.sqrt 6) / 4)) :=
by
  sorry

end first_term_geometric_progression_l983_98335


namespace general_term_min_value_S_n_l983_98346

-- Definitions and conditions according to the problem statement
variable (d : ℤ) (a₁ : ℤ) (n : ℕ)

def a_n (n : ℕ) : ℤ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Given conditions
axiom positive_common_difference : 0 < d
axiom a3_a4_product : a_n 3 * a_n 4 = 117
axiom a2_a5_sum : a_n 2 + a_n 5 = -22

-- Proof 1: General term of the arithmetic sequence
theorem general_term : a_n n = 4 * (n : ℤ) - 25 :=
  by sorry

-- Proof 2: Minimum value of the sum of the first n terms
theorem min_value_S_n : S_n 6 = -66 :=
  by sorry

end general_term_min_value_S_n_l983_98346


namespace integer_roots_condition_l983_98358

theorem integer_roots_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℤ, x^2 - 4 * x + n = 0) ↔ (n = 3 ∨ n = 4) := 
by
  sorry

end integer_roots_condition_l983_98358


namespace largest_n_l983_98344

theorem largest_n (n x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 6 * x + 6 * y + 6 * z - 18 →
  n ≤ 3 := 
by 
  sorry

end largest_n_l983_98344


namespace parker_added_dumbbells_l983_98379

def initial_dumbbells : Nat := 4
def weight_per_dumbbell : Nat := 20
def total_weight_used : Nat := 120

theorem parker_added_dumbbells :
  (total_weight_used - (initial_dumbbells * weight_per_dumbbell)) / weight_per_dumbbell = 2 := by
  sorry

end parker_added_dumbbells_l983_98379


namespace tim_buys_loaves_l983_98343

theorem tim_buys_loaves (slices_per_loaf : ℕ) (paid : ℕ) (change : ℕ) (price_per_slice_cents : ℕ) 
    (h1 : slices_per_loaf = 20) 
    (h2 : paid = 2 * 20) 
    (h3 : change = 16) 
    (h4 : price_per_slice_cents = 40) : 
    (paid - change) / (slices_per_loaf * price_per_slice_cents / 100) = 3 := 
by 
  -- proof omitted 
  sorry

end tim_buys_loaves_l983_98343


namespace split_payment_l983_98325

noncomputable def Rahul_work_per_day := (1 : ℝ) / 3
noncomputable def Rajesh_work_per_day := (1 : ℝ) / 2
noncomputable def Ritesh_work_per_day := (1 : ℝ) / 4

noncomputable def total_work_per_day := Rahul_work_per_day + Rajesh_work_per_day + Ritesh_work_per_day

noncomputable def Rahul_proportion := Rahul_work_per_day / total_work_per_day
noncomputable def Rajesh_proportion := Rajesh_work_per_day / total_work_per_day
noncomputable def Ritesh_proportion := Ritesh_work_per_day / total_work_per_day

noncomputable def total_payment := 510

noncomputable def Rahul_share := Rahul_proportion * total_payment
noncomputable def Rajesh_share := Rajesh_proportion * total_payment
noncomputable def Ritesh_share := Ritesh_proportion * total_payment

theorem split_payment :
  Rahul_share + Rajesh_share + Ritesh_share = total_payment :=
by
  sorry

end split_payment_l983_98325


namespace urn_contains_three_red_three_blue_after_five_operations_l983_98314

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5

noncomputable def calculate_probability (initial_red: ℕ) (initial_blue: ℕ) (operations: ℕ) : ℚ :=
  sorry

theorem urn_contains_three_red_three_blue_after_five_operations :
  calculate_probability initial_red_balls initial_blue_balls total_operations = 8 / 105 :=
by sorry

end urn_contains_three_red_three_blue_after_five_operations_l983_98314


namespace minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l983_98323

theorem minimal_distance_ln_x_x :
  ∀ (x : ℝ), x > 0 → ∃ (d : ℝ), d = |Real.log x - x| → d ≥ 0 :=
by sorry

theorem minimal_distance_graphs_ex_ln_x :
  ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), ∃ (d : ℝ), y = d → d = 2 :=
by sorry

end minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l983_98323


namespace proof_complex_magnitude_z_l983_98334

noncomputable def complex_magnitude_z : Prop :=
  ∀ (z : ℂ),
    (z * (Complex.cos (Real.pi / 9) + Complex.sin (Real.pi / 9) * Complex.I) ^ 6 = 2) →
    Complex.abs z = 2

theorem proof_complex_magnitude_z : complex_magnitude_z :=
by
  intros z h
  sorry

end proof_complex_magnitude_z_l983_98334


namespace tan_triple_angle_formula_l983_98390

variable (θ : ℝ)
variable (h : Real.tan θ = 4)

theorem tan_triple_angle_formula : Real.tan (3 * θ) = 52 / 47 :=
by
  sorry  -- Proof is omitted

end tan_triple_angle_formula_l983_98390


namespace marcia_wardrobe_cost_l983_98307

theorem marcia_wardrobe_cost :
  let skirt_price := 20
  let blouse_price := 15
  let pant_price := 30
  let num_skirts := 3
  let num_blouses := 5
  let num_pants := 2
  let pant_offer := buy_1_get_1_half
  let skirt_cost := num_skirts * skirt_price
  let blouse_cost := num_blouses * blouse_price
  let pant_full_price := pant_price
  let pant_half_price := pant_price / 2
  let pant_cost := pant_full_price + pant_half_price
  let total_cost := skirt_cost + blouse_cost + pant_cost
  total_cost = 180 :=
by
  sorry -- proof is omitted

end marcia_wardrobe_cost_l983_98307


namespace general_term_l983_98388

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S2 : S 2 = 4
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1

theorem general_term (n : ℕ) : a n = 3 ^ (n - 1) :=
by
  sorry

end general_term_l983_98388


namespace smallest_p_condition_l983_98310

theorem smallest_p_condition (n p : ℕ) (hn1 : n % 2 = 1) (hn2 : n % 7 = 5) (hp : (n + p) % 10 = 0) : p = 1 := by
  sorry

end smallest_p_condition_l983_98310


namespace perfect_square_fraction_l983_98302

theorem perfect_square_fraction (n : ℤ) : 
  n < 30 ∧ ∃ k : ℤ, (n / (30 - n)) = k^2 → ∃ cnt : ℕ, cnt = 4 :=
  by
  sorry

end perfect_square_fraction_l983_98302


namespace sqrt_of_16_l983_98372

theorem sqrt_of_16 : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_16_l983_98372


namespace largest_unique_k_l983_98350

theorem largest_unique_k (n : ℕ) :
  (∀ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13 → False) ∧
  (∃ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13) → n = 112 :=
by sorry

end largest_unique_k_l983_98350


namespace dice_probability_divisible_by_three_ge_one_fourth_l983_98311

theorem dice_probability_divisible_by_three_ge_one_fourth
  (p q r : ℝ) 
  (h1 : 0 ≤ p) (h2 : 0 ≤ q) (h3 : 0 ≤ r) 
  (h4 : p + q + r = 1) : 
  p^3 + q^3 + r^3 + 6 * p * q * r ≥ 1 / 4 :=
sorry

end dice_probability_divisible_by_three_ge_one_fourth_l983_98311


namespace area_of_square_field_l983_98333

-- Define side length
def side_length : ℕ := 20

-- Theorem statement about the area of the square field
theorem area_of_square_field : (side_length * side_length) = 400 := by
  sorry

end area_of_square_field_l983_98333


namespace three_digit_divisible_by_11_l983_98363

theorem three_digit_divisible_by_11
  (x y z : ℕ) (h1 : y = x + z) : (100 * x + 10 * y + z) % 11 = 0 :=
by
  sorry

end three_digit_divisible_by_11_l983_98363


namespace center_of_circle_in_second_or_fourth_quadrant_l983_98315

theorem center_of_circle_in_second_or_fourth_quadrant
  (α : ℝ) 
  (hyp1 : ∀ x y : ℝ, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 → Real.cos α * Real.sin α > 0)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x*Real.cos α - 2*y*Real.sin α = 0) :
  (-Real.cos α > 0 ∧ Real.sin α > 0) ∨ (-Real.cos α < 0 ∧ Real.sin α < 0) :=
sorry

end center_of_circle_in_second_or_fourth_quadrant_l983_98315


namespace solution_set_inequality_l983_98318

open Set

theorem solution_set_inequality :
  {x : ℝ | (x+1)/(x-4) ≥ 3} = Iio 4 ∪ Ioo 4 (13/2) ∪ {13/2} :=
by
  sorry

end solution_set_inequality_l983_98318


namespace least_sales_needed_not_lose_money_l983_98306

noncomputable def old_salary : ℝ := 75000
noncomputable def new_salary_base : ℝ := 45000
noncomputable def commission_rate : ℝ := 0.15
noncomputable def sale_amount : ℝ := 750

theorem least_sales_needed_not_lose_money : 
  ∃ (n : ℕ), n * (commission_rate * sale_amount) ≥ (old_salary - new_salary_base) ∧ n = 267 := 
by
  -- The proof will show that n = 267 is the least number of sales needed to not lose money.
  existsi 267
  sorry

end least_sales_needed_not_lose_money_l983_98306


namespace even_abs_func_necessary_not_sufficient_l983_98319

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem even_abs_func_necessary_not_sufficient (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, |f (-x)| = |f x|) ∧ (∃ g : ℝ → ℝ, (∀ x : ℝ, |g (-x)| = |g x|) ∧ ¬(∀ x : ℝ, g (-x) = -g x)) :=
by
  -- Proof omitted.
  sorry

end even_abs_func_necessary_not_sufficient_l983_98319


namespace total_and_per_suitcase_profit_l983_98308

theorem total_and_per_suitcase_profit
  (num_suitcases : ℕ)
  (purchase_price_per_suitcase : ℕ)
  (total_sales_revenue : ℕ)
  (total_profit : ℕ)
  (profit_per_suitcase : ℕ)
  (h_num_suitcases : num_suitcases = 60)
  (h_purchase_price : purchase_price_per_suitcase = 100)
  (h_total_sales : total_sales_revenue = 8100)
  (h_total_profit : total_profit = total_sales_revenue - num_suitcases * purchase_price_per_suitcase)
  (h_profit_per_suitcase : profit_per_suitcase = total_profit / num_suitcases) :
  total_profit = 2100 ∧ profit_per_suitcase = 35 := by
  sorry

end total_and_per_suitcase_profit_l983_98308
