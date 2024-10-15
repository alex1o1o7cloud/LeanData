import Mathlib

namespace NUMINAMATH_GPT_largest_among_trig_expressions_l1153_115331

theorem largest_among_trig_expressions :
  let a := Real.tan 48 + 1 / Real.tan 48
  let b := Real.sin 48 + Real.cos 48
  let c := Real.tan 48 + Real.cos 48
  let d := 1 / Real.tan 48 + Real.sin 48
  a > b ∧ a > c ∧ a > d :=
by
  sorry

end NUMINAMATH_GPT_largest_among_trig_expressions_l1153_115331


namespace NUMINAMATH_GPT_sqrt_three_irrational_l1153_115372

theorem sqrt_three_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a:ℝ) / b = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_sqrt_three_irrational_l1153_115372


namespace NUMINAMATH_GPT_second_rectangle_area_l1153_115380

theorem second_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hbx : x < h):
  2 * b * x * (h - 3 * x) / h = (2 * b * x * (h - 3 * x))/h := 
sorry

end NUMINAMATH_GPT_second_rectangle_area_l1153_115380


namespace NUMINAMATH_GPT_max_product_partition_l1153_115333

theorem max_product_partition (k n : ℕ) (hkn : k ≥ n) 
  (q r : ℕ) (hqr : k = n * q + r) (h_r : 0 ≤ r ∧ r < n) : 
  ∃ (F : ℕ → ℕ), F k = q^(n-r) * (q+1)^r :=
by
  sorry

end NUMINAMATH_GPT_max_product_partition_l1153_115333


namespace NUMINAMATH_GPT_intersection_M_N_l1153_115308

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def N : Set ℝ := { y | 0 < y }

theorem intersection_M_N : (M ∩ N) = { z | 0 < z ∧ z ≤ 2 } :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1153_115308


namespace NUMINAMATH_GPT_tuesday_rainfall_l1153_115321

-- Condition: average rainfall for the whole week is 3 cm
def avg_rainfall_week : ℝ := 3

-- Condition: number of days in a week
def days_in_week : ℕ := 7

-- Condition: total rainfall for the week
def total_rainfall_week : ℝ := avg_rainfall_week * days_in_week

-- Condition: total rainfall is twice the rainfall on Tuesday
def total_rainfall_equals_twice_T (T : ℝ) : ℝ := 2 * T

-- Theorem: Prove that the rainfall on Tuesday is 10.5 cm
theorem tuesday_rainfall : ∃ T : ℝ, total_rainfall_equals_twice_T T = total_rainfall_week ∧ T = 10.5 := by
  sorry

end NUMINAMATH_GPT_tuesday_rainfall_l1153_115321


namespace NUMINAMATH_GPT_age_problem_l1153_115378

theorem age_problem (M D : ℕ) (h1 : M = 40) (h2 : 2 * D + M = 70) : 2 * M + D = 95 := by
  sorry

end NUMINAMATH_GPT_age_problem_l1153_115378


namespace NUMINAMATH_GPT_find_C_given_eq_statement_max_area_triangle_statement_l1153_115357

open Real

noncomputable def find_C_given_eq (a b c A : ℝ) (C : ℝ) : Prop :=
  (2 * a = sqrt 3 * c * sin A - a * cos C) → 
  C = 2 * π / 3

noncomputable def max_area_triangle (a b c : ℝ) (C : ℝ) : Prop :=
  C = 2 * π / 3 →
  c = sqrt 3 →
  ∃ S, S = (sqrt 3 / 4) * a * b ∧ 
  ∀ a b : ℝ, a * b ≤ 1 → S = (sqrt 3 / 4)

-- Lean statements
theorem find_C_given_eq_statement (a b c A C : ℝ) : find_C_given_eq a b c A C := 
by sorry

theorem max_area_triangle_statement (a b c : ℝ) (C : ℝ) : max_area_triangle a b c C := 
by sorry

end NUMINAMATH_GPT_find_C_given_eq_statement_max_area_triangle_statement_l1153_115357


namespace NUMINAMATH_GPT_hannah_highest_score_l1153_115335

-- Definitions based on conditions
def total_questions : ℕ := 40
def wrong_questions : ℕ := 3
def correct_percent_student_1 : ℝ := 0.95

-- The Lean statement representing the proof problem
theorem hannah_highest_score :
  ∃ q : ℕ, (q > (total_questions - wrong_questions) ∧ q > (total_questions * correct_percent_student_1)) ∧ q = 39 :=
by
  sorry

end NUMINAMATH_GPT_hannah_highest_score_l1153_115335


namespace NUMINAMATH_GPT_price_increase_equivalence_l1153_115369

theorem price_increase_equivalence (P : ℝ) : 
  let increase_35 := P * 1.35
  let increase_40 := increase_35 * 1.40
  let increase_20 := increase_40 * 1.20
  let final_increase := increase_20
  final_increase = P * 2.268 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_price_increase_equivalence_l1153_115369


namespace NUMINAMATH_GPT_polar_to_cartesian_l1153_115305

theorem polar_to_cartesian (ρ θ x y : ℝ) (h1 : ρ = 2 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_GPT_polar_to_cartesian_l1153_115305


namespace NUMINAMATH_GPT_triangle_side_relation_l1153_115398

-- Definitions for the conditions
variable {A B C a b c : ℝ}
variable (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variable (sides_rel : a = (B * (1 + 2 * C)).sin)
variable (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin))

-- The statement to be proven
theorem triangle_side_relation (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (sides_rel : a = (B * (1 + 2 * C)).sin)
  (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin)) :
  a = 2 * b := 
sorry

end NUMINAMATH_GPT_triangle_side_relation_l1153_115398


namespace NUMINAMATH_GPT_total_net_worth_after_2_years_l1153_115362

def initial_value : ℝ := 40000
def depreciation_rate : ℝ := 0.05
def initial_maintenance_cost : ℝ := 2000
def inflation_rate : ℝ := 0.03
def years : ℕ := 2

def value_at_end_of_year (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc _ => acc * (1 - rate)) initial_value (List.range years)

def cumulative_maintenance_cost (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc year => acc + initial_maintenance_cost * ((1 + inflation_rate) ^ year)) 0 (List.range years)

def total_net_worth (initial_value : ℝ) (depreciation_rate : ℝ) (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  value_at_end_of_year initial_value depreciation_rate years - cumulative_maintenance_cost initial_maintenance_cost inflation_rate years

theorem total_net_worth_after_2_years : total_net_worth initial_value depreciation_rate initial_maintenance_cost inflation_rate years = 32040 :=
  by
    sorry

end NUMINAMATH_GPT_total_net_worth_after_2_years_l1153_115362


namespace NUMINAMATH_GPT_swimming_speed_solution_l1153_115375

-- Definition of the conditions
def speed_of_water : ℝ := 2
def distance_against_current : ℝ := 10
def time_against_current : ℝ := 5

-- Definition of the person's swimming speed in still water
def swimming_speed_in_still_water (v : ℝ) :=
  distance_against_current = (v - speed_of_water) * time_against_current

-- Main theorem we want to prove
theorem swimming_speed_solution : 
  ∃ v : ℝ, swimming_speed_in_still_water v ∧ v = 4 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_solution_l1153_115375


namespace NUMINAMATH_GPT_children_ages_l1153_115385

-- Define the ages of the four children
variable (a b c d : ℕ)

-- Define the conditions
axiom h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom h2 : a + b + c + d = 31
axiom h3 : (a - 4) + (b - 4) + (c - 4) + (d - 4) = 16
axiom h4 : (a - 7) + (b - 7) + (c - 7) + (d - 7) = 8
axiom h5 : (a - 11) + (b - 11) + (c - 11) + (d - 11) = 1
noncomputable def ages : ℕ × ℕ × ℕ × ℕ := (12, 10, 6, 3)

-- The theorem to prove
theorem children_ages (h1 : a = 12) (h2 : b = 10) (h3 : c = 6) (h4 : d = 3) : a = 12 ∧ b = 10 ∧ c = 6 ∧ d = 3 :=
by sorry

end NUMINAMATH_GPT_children_ages_l1153_115385


namespace NUMINAMATH_GPT_converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l1153_115310

-- Definitions
variables {α : Type} [LinearOrderedField α] {a b : α}
variables {A B C : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]

-- Proof Problem for Question 1
theorem converse_angle_bigger_side (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]
  (angle_C angle_B : A) (side_AB side_AC : B) (h : angle_C > angle_B) : side_AB > side_AC :=
sorry

-- Proof Problem for Question 2
theorem negation_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

-- Proof Problem for Question 3
theorem contrapositive_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end NUMINAMATH_GPT_converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l1153_115310


namespace NUMINAMATH_GPT_find_value_l1153_115353

-- Definitions of the curve and the line
def curve (a b : ℝ) (P : ℝ × ℝ) : Prop := (P.1*P.1) / a - (P.2*P.2) / b = 1
def line (P : ℝ × ℝ) : Prop := P.1 + P.2 - 1 = 0

-- Definition of the dot product condition
def dot_product_zero (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

-- Theorem statement
theorem find_value (a b : ℝ) (P Q : ℝ × ℝ)
  (hc1 : curve a b P)
  (hc2 : curve a b Q)
  (hl1 : line P)
  (hl2 : line Q)
  (h_dot : dot_product_zero P Q) :
  1 / a - 1 / b = 2 :=
sorry

end NUMINAMATH_GPT_find_value_l1153_115353


namespace NUMINAMATH_GPT_fewest_students_possible_l1153_115386

theorem fewest_students_possible : 
  ∃ n : ℕ, n % 3 = 1 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ ∀ m, m % 3 = 1 ∧ m % 6 = 4 ∧ m % 8 = 5 → n ≤ m := 
by
  sorry

end NUMINAMATH_GPT_fewest_students_possible_l1153_115386


namespace NUMINAMATH_GPT_varies_fix_l1153_115312

variable {x y z : ℝ}

theorem varies_fix {k j : ℝ} 
  (h1 : x = k * y^4)
  (h2 : y = j * z^(1/3)) : x = (k * j^4) * z^(4/3) := by
  sorry

end NUMINAMATH_GPT_varies_fix_l1153_115312


namespace NUMINAMATH_GPT_round_trip_time_l1153_115337

variable (dist : ℝ)
variable (speed_to_work : ℝ)
variable (speed_to_home : ℝ)

theorem round_trip_time (h_dist : dist = 24) (h_speed_to_work : speed_to_work = 60) (h_speed_to_home : speed_to_home = 40) :
    (dist / speed_to_work + dist / speed_to_home) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_round_trip_time_l1153_115337


namespace NUMINAMATH_GPT_prime_in_A_l1153_115361

def A (n : ℕ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ n = a^2 + 2 * b^2

theorem prime_in_A {p : ℕ} (h_prime : Nat.Prime p) (h_p2_in_A : A (p^2)) : A p :=
sorry

end NUMINAMATH_GPT_prime_in_A_l1153_115361


namespace NUMINAMATH_GPT_containers_per_truck_l1153_115314

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end NUMINAMATH_GPT_containers_per_truck_l1153_115314


namespace NUMINAMATH_GPT_product_sum_diff_l1153_115307

variable (a b : ℝ) -- Real numbers

theorem product_sum_diff (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) :=
by
  sorry

end NUMINAMATH_GPT_product_sum_diff_l1153_115307


namespace NUMINAMATH_GPT_integer_value_of_expression_l1153_115391

theorem integer_value_of_expression (m n p : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ 9)
  (h3 : 2 ≤ n) (h4 : n ≤ 9) (h5 : 2 ≤ p) (h6 : p ≤ 9)
  (h7 : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  (m + n + p) / (m + n) = 1 :=
sorry

end NUMINAMATH_GPT_integer_value_of_expression_l1153_115391


namespace NUMINAMATH_GPT_percent_of_x_l1153_115364

variable (x : ℝ) (h : x > 0)

theorem percent_of_x (p : ℝ) : 
  (p * x = 0.21 * x + 10) → 
  p = 0.21 + 10 / x :=
sorry

end NUMINAMATH_GPT_percent_of_x_l1153_115364


namespace NUMINAMATH_GPT_ferris_wheel_capacity_l1153_115328

-- Define the conditions
def number_of_seats : ℕ := 14
def people_per_seat : ℕ := 6

-- Theorem to prove the total capacity is 84
theorem ferris_wheel_capacity : number_of_seats * people_per_seat = 84 := sorry

end NUMINAMATH_GPT_ferris_wheel_capacity_l1153_115328


namespace NUMINAMATH_GPT_loss_is_negative_one_point_twenty_seven_percent_l1153_115382

noncomputable def book_price : ℝ := 600
noncomputable def gov_tax_rate : ℝ := 0.05
noncomputable def shipping_fee : ℝ := 20
noncomputable def seller_discount_rate : ℝ := 0.03
noncomputable def selling_price : ℝ := 624

noncomputable def gov_tax : ℝ := gov_tax_rate * book_price
noncomputable def seller_discount : ℝ := seller_discount_rate * book_price
noncomputable def total_cost : ℝ := book_price + gov_tax + shipping_fee - seller_discount
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def loss_percentage : ℝ := (profit / total_cost) * 100

theorem loss_is_negative_one_point_twenty_seven_percent :
  loss_percentage = -1.27 :=
by
  sorry

end NUMINAMATH_GPT_loss_is_negative_one_point_twenty_seven_percent_l1153_115382


namespace NUMINAMATH_GPT_find_moles_of_NaOH_l1153_115340

-- Define the conditions
def reaction (NaOH HClO4 NaClO4 H2O : ℕ) : Prop :=
  NaOH = HClO4 ∧ NaClO4 = HClO4 ∧ H2O = 1

def moles_of_HClO4 := 3
def moles_of_NaClO4 := 3

-- Problem statement
theorem find_moles_of_NaOH : ∃ (NaOH : ℕ), NaOH = moles_of_HClO4 ∧ moles_of_NaClO4 = 3 ∧ NaOH = 3 :=
by sorry

end NUMINAMATH_GPT_find_moles_of_NaOH_l1153_115340


namespace NUMINAMATH_GPT_complex_number_properties_l1153_115373

theorem complex_number_properties (z : ℂ) (h : z^2 = 3 + 4 * Complex.I) : 
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_complex_number_properties_l1153_115373


namespace NUMINAMATH_GPT_only_nice_number_is_three_l1153_115300

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def nice (n : ℕ) : Prop :=
  ∃ (xs ys : ℕ → ℕ), 
    xs 1 = 1 ∧ ys 1 = 3 ∧
    (∀ k, xs (k+1) = P (xs k) ∧ ys (k+1) = Q (ys k) ∨ xs (k+1) = Q (xs k) ∧ ys (k+1) = P (ys k)) ∧
    xs n = ys n

theorem only_nice_number_is_three (n : ℕ) : nice n ↔ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_only_nice_number_is_three_l1153_115300


namespace NUMINAMATH_GPT_football_team_throwers_l1153_115350

theorem football_team_throwers
    (total_players : ℕ)
    (right_handed_players : ℕ)
    (one_third : ℚ)
    (number_throwers : ℕ)
    (number_non_throwers : ℕ)
    (right_handed_non_throwers : ℕ)
    (left_handed_non_throwers : ℕ)
    (h1 : total_players = 70)
    (h2 : right_handed_players = 63)
    (h3 : one_third = 1 / 3)
    (h4 : number_non_throwers = total_players - number_throwers)
    (h5 : right_handed_non_throwers = right_handed_players - number_throwers)
    (h6 : left_handed_non_throwers = one_third * number_non_throwers)
    (h7 : 2 * left_handed_non_throwers = right_handed_non_throwers)
    : number_throwers = 49 := 
by
  sorry

end NUMINAMATH_GPT_football_team_throwers_l1153_115350


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_problem_l1153_115374

variable {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 0 + n * d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (n * (a_n 0 + a_n (n-1))) / 2

def forms_geometric_sequence (a1 a3 a4 : ℝ) :=
  a3^2 = a1 * a4

-- The main proof statement
theorem arithmetic_geometric_sequence_problem
        (h_arith : is_arithmetic_sequence a_n)
        (h_sum : sum_of_first_n_terms a_n S)
        (h_geom : forms_geometric_sequence (a_n 0) (a_n 2) (a_n 3)) :
        (S 3 - S 2) / (S 5 - S 3) = 2 ∨ (S 3 - S 2) / (S 5 - S 3) = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_problem_l1153_115374


namespace NUMINAMATH_GPT_product_of_real_solutions_l1153_115368

theorem product_of_real_solutions :
  (∀ x : ℝ, (x + 1) / (3 * x + 3) = (3 * x + 2) / (8 * x + 2)) →
  x = -1 ∨ x = -4 →
  (-1) * (-4) = 4 := 
sorry

end NUMINAMATH_GPT_product_of_real_solutions_l1153_115368


namespace NUMINAMATH_GPT_division_multiplication_order_l1153_115332

theorem division_multiplication_order : 1100 / 25 * 4 / 11 = 16 := by
  sorry

end NUMINAMATH_GPT_division_multiplication_order_l1153_115332


namespace NUMINAMATH_GPT_girls_in_art_class_l1153_115309

theorem girls_in_art_class (g b : ℕ) (h_ratio : 4 * b = 3 * g) (h_total : g + b = 70) : g = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_girls_in_art_class_l1153_115309


namespace NUMINAMATH_GPT_tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l1153_115339

theorem tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2 (alpha : ℝ) 
  (h1 : Real.sin alpha = - (Real.sqrt 3) / 2) 
  (h2 : 3 * π / 2 < alpha ∧ alpha < 2 * π) : 
  Real.tan alpha = - Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l1153_115339


namespace NUMINAMATH_GPT_complex_numbers_are_real_l1153_115301

theorem complex_numbers_are_real
  (a b c : ℂ)
  (h1 : (a + b) * (a + c) = b)
  (h2 : (b + c) * (b + a) = c)
  (h3 : (c + a) * (c + b) = a) : 
  a.im = 0 ∧ b.im = 0 ∧ c.im = 0 :=
sorry

end NUMINAMATH_GPT_complex_numbers_are_real_l1153_115301


namespace NUMINAMATH_GPT_statement_1_statement_2_statement_3_statement_4_l1153_115330

variables (a b c x0 : ℝ)
noncomputable def P (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Statement ①
theorem statement_1 (h : a - b + c = 0) : P a b c (-1) = 0 := sorry

-- Statement ②
theorem statement_2 (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + c = 0 ∧ a*x2^2 + c = 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 := sorry

-- Statement ③
theorem statement_3 (h : P a b c c = 0) : a*c + b + 1 = 0 := sorry

-- Statement ④
theorem statement_4 (h : P a b c x0 = 0) : b^2 - 4*a*c = (2*a*x0 + b)^2 := sorry

end NUMINAMATH_GPT_statement_1_statement_2_statement_3_statement_4_l1153_115330


namespace NUMINAMATH_GPT_div_polynomial_not_div_l1153_115324

theorem div_polynomial_not_div (n : ℕ) : ¬ (n + 2) ∣ (n^3 - 2 * n^2 - 5 * n + 7) := by
  sorry

end NUMINAMATH_GPT_div_polynomial_not_div_l1153_115324


namespace NUMINAMATH_GPT_find_x_if_perpendicular_l1153_115329

-- Define vectors a and b in the given conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x - 5, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The Lean theorem statement equivalent to the math problem
theorem find_x_if_perpendicular (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : x = 2 := by
  sorry

end NUMINAMATH_GPT_find_x_if_perpendicular_l1153_115329


namespace NUMINAMATH_GPT_pow_mod_l1153_115317

theorem pow_mod (h : 3^3 ≡ 1 [MOD 13]) : 3^21 ≡ 1 [MOD 13] :=
by
sorry

end NUMINAMATH_GPT_pow_mod_l1153_115317


namespace NUMINAMATH_GPT_solution_set_of_fx_eq_zero_l1153_115344

noncomputable def f (x : ℝ) : ℝ :=
if hx : x = 0 then 0 else if 0 < x then Real.log x / Real.log 2 else - (Real.log (-x) / Real.log 2)

lemma f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by sorry

lemma f_is_log_for_positive : ∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2 :=
by sorry

theorem solution_set_of_fx_eq_zero :
  {x : ℝ | f x = 0} = {-1, 0, 1} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_fx_eq_zero_l1153_115344


namespace NUMINAMATH_GPT_beach_weather_condition_l1153_115315

theorem beach_weather_condition
  (T : ℝ) -- Temperature in degrees Fahrenheit
  (sunny : Prop) -- Whether it is sunny
  (crowded : Prop) -- Whether the beach is crowded
  (H1 : ∀ (T : ℝ) (sunny : Prop), (T ≥ 80) ∧ sunny → crowded) -- Condition 1
  (H2 : ¬ crowded) -- Condition 2
  : T < 80 ∨ ¬ sunny := sorry

end NUMINAMATH_GPT_beach_weather_condition_l1153_115315


namespace NUMINAMATH_GPT_truffles_more_than_caramels_l1153_115360

-- Define the conditions
def chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def peanut_clusters := (64 * chocolates) / 100
def truffles := chocolates - (caramels + nougats + peanut_clusters)

-- Define the claim
theorem truffles_more_than_caramels : (truffles - caramels) = 6 := by
  sorry

end NUMINAMATH_GPT_truffles_more_than_caramels_l1153_115360


namespace NUMINAMATH_GPT_incorrect_statement_D_l1153_115355

theorem incorrect_statement_D 
  (population : Set ℕ)
  (time_spent_sample : ℕ → ℕ)
  (sample_size : ℕ)
  (individual : ℕ)
  (h1 : ∀ s, s ∈ population → s ≤ 24)
  (h2 : ∀ i, i < sample_size → population (time_spent_sample i))
  (h3 : sample_size = 300)
  (h4 : ∀ i, i < 300 → time_spent_sample i = individual):
  ¬ (∀ i, i < 300 → time_spent_sample i = individual) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1153_115355


namespace NUMINAMATH_GPT_percentage_neither_bp_nor_ht_l1153_115351

noncomputable def percentage_teachers_neither_condition (total: ℕ) (high_bp: ℕ) (heart_trouble: ℕ) (both: ℕ) : ℚ :=
  let either_condition := high_bp + heart_trouble - both
  let neither_condition := total - either_condition
  (neither_condition * 100 : ℚ) / total

theorem percentage_neither_bp_nor_ht :
  percentage_teachers_neither_condition 150 90 50 30 = 26.67 :=
by
  sorry

end NUMINAMATH_GPT_percentage_neither_bp_nor_ht_l1153_115351


namespace NUMINAMATH_GPT_geometric_seq_arith_seq_problem_l1153_115303

theorem geometric_seq_arith_seq_problem (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n, a (n + 1) = q * a n)
  (h_q_pos : q > 0)
  (h_arith : 2 * (1/2 : ℝ) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1 / 9 := 
sorry

end NUMINAMATH_GPT_geometric_seq_arith_seq_problem_l1153_115303


namespace NUMINAMATH_GPT_potatoes_fraction_l1153_115304

theorem potatoes_fraction (w : ℝ) (x : ℝ) (h_weight : w = 36) (h_fraction : w / x = 36) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_potatoes_fraction_l1153_115304


namespace NUMINAMATH_GPT_cakes_to_make_l1153_115346

-- Define the conditions
def packages_per_cake : ℕ := 2
def cost_per_package : ℕ := 3
def total_cost : ℕ := 12

-- Define the proof problem
theorem cakes_to_make (h1 : packages_per_cake = 2) (h2 : cost_per_package = 3) (h3 : total_cost = 12) :
  (total_cost / cost_per_package) / packages_per_cake = 2 :=
by sorry

end NUMINAMATH_GPT_cakes_to_make_l1153_115346


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1153_115326

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_def : ∀ x : ℝ, x ≤ 0 → f x = x^2 + 2 * x) :
  {x : ℝ | f (x + 2) < 3} = {x : ℝ | -5 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1153_115326


namespace NUMINAMATH_GPT_solution_l1153_115371

noncomputable def prove_a_greater_than_3 : Prop :=
  ∀ (x : ℝ) (a : ℝ), (a > 0) → (|x - 2| + |x - 3| + |x - 4| < a) → a > 3

theorem solution : prove_a_greater_than_3 :=
by
  intros x a h_pos h_ineq
  sorry

end NUMINAMATH_GPT_solution_l1153_115371


namespace NUMINAMATH_GPT_gate_paid_more_l1153_115399

def pre_booked_economy_cost : Nat := 10 * 140
def pre_booked_business_cost : Nat := 10 * 170
def total_pre_booked_cost : Nat := pre_booked_economy_cost + pre_booked_business_cost

def gate_economy_cost : Nat := 8 * 190
def gate_business_cost : Nat := 12 * 210
def gate_first_class_cost : Nat := 10 * 300
def total_gate_cost : Nat := gate_economy_cost + gate_business_cost + gate_first_class_cost

theorem gate_paid_more {gate_paid_more_cost : Nat} :
  total_gate_cost - total_pre_booked_cost = 3940 :=
by
  sorry

end NUMINAMATH_GPT_gate_paid_more_l1153_115399


namespace NUMINAMATH_GPT_emma_average_speed_l1153_115354

-- Define the given conditions
def distance1 : ℕ := 420     -- Distance traveled in the first segment
def time1 : ℕ := 7          -- Time taken in the first segment
def distance2 : ℕ := 480    -- Distance traveled in the second segment
def time2 : ℕ := 8          -- Time taken in the second segment

-- Define the total distance and total time
def total_distance : ℕ := distance1 + distance2
def total_time : ℕ := time1 + time2

-- Define the expected average speed
def expected_average_speed : ℕ := 60

-- Prove that the average speed is 60 miles per hour
theorem emma_average_speed : (total_distance / total_time) = expected_average_speed := by
  sorry

end NUMINAMATH_GPT_emma_average_speed_l1153_115354


namespace NUMINAMATH_GPT_tan_ratio_l1153_115393

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 7 / 3 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l1153_115393


namespace NUMINAMATH_GPT_five_term_geometric_sequence_value_of_b_l1153_115387

theorem five_term_geometric_sequence_value_of_b (a b c : ℝ) (h₁ : b ^ 2 = 81) (h₂ : a ^ 2 = b) (h₃ : 1 * a = a) (h₄ : c * c = c) :
  b = 9 :=
by 
  sorry

end NUMINAMATH_GPT_five_term_geometric_sequence_value_of_b_l1153_115387


namespace NUMINAMATH_GPT_sum_of_coordinates_of_D_is_12_l1153_115370

theorem sum_of_coordinates_of_D_is_12 :
  (exists (x y : ℝ), (5 = (11 + x) / 2) ∧ (9 = (5 + y) / 2) ∧ (x + y = 12)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_D_is_12_l1153_115370


namespace NUMINAMATH_GPT_largest_even_among_consecutives_l1153_115395

theorem largest_even_among_consecutives (x : ℤ) (h : (x + (x + 2) + (x + 4) = x + 18)) : x + 4 = 10 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_among_consecutives_l1153_115395


namespace NUMINAMATH_GPT_problem_equivalence_l1153_115390

variables (P Q : Prop)

theorem problem_equivalence :
  (P ↔ Q) ↔ ((P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q)) :=
by sorry

end NUMINAMATH_GPT_problem_equivalence_l1153_115390


namespace NUMINAMATH_GPT_boxes_of_nerds_l1153_115376

def totalCandies (kitKatBars hersheyKisses lollipops babyRuths reeseCups nerds : Nat) : Nat := 
  kitKatBars + hersheyKisses + lollipops + babyRuths + reeseCups + nerds

def adjustForGivenLollipops (total lollipopsGiven : Nat) : Nat :=
  total - lollipopsGiven

theorem boxes_of_nerds :
  ∀ (kitKatBars hersheyKisses lollipops babyRuths reeseCups lollipopsGiven totalAfterGiving nerds : Nat),
  kitKatBars = 5 →
  hersheyKisses = 3 * kitKatBars →
  lollipops = 11 →
  babyRuths = 10 →
  reeseCups = babyRuths / 2 →
  lollipopsGiven = 5 →
  totalAfterGiving = 49 →
  totalCandies kitKatBars hersheyKisses lollipops babyRuths reeseCups 0 - lollipopsGiven + nerds = totalAfterGiving →
  nerds = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boxes_of_nerds_l1153_115376


namespace NUMINAMATH_GPT_route_down_distance_l1153_115316

theorem route_down_distance
  (rate_up : ℕ)
  (time_up : ℕ)
  (rate_down_rate_factor : ℚ)
  (time_down : ℕ)
  (h1 : rate_up = 4)
  (h2 : time_up = 2)
  (h3 : rate_down_rate_factor = (3 / 2))
  (h4 : time_down = time_up) :
  rate_down_rate_factor * rate_up * time_up = 12 := 
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_route_down_distance_l1153_115316


namespace NUMINAMATH_GPT_pyramid_volume_theorem_l1153_115306

noncomputable def volume_of_regular_square_pyramid : ℝ := 
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * side_edge_length * Real.sqrt 3
  (1 / 3) * base_area * height

theorem pyramid_volume_theorem :
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * (side_edge_length * Real.sqrt 3)
  (1 / 3) * base_area * height = 6 := 
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_theorem_l1153_115306


namespace NUMINAMATH_GPT_find_wsquared_l1153_115383

theorem find_wsquared : 
  (2 * w + 10) ^ 2 = (5 * w + 15) * (w + 6) →
  w ^ 2 = (90 + 10 * Real.sqrt 65) / 4 := 
by 
  intro h₀
  sorry

end NUMINAMATH_GPT_find_wsquared_l1153_115383


namespace NUMINAMATH_GPT_speed_of_canoe_downstream_l1153_115342

-- Definition of the problem conditions
def speed_of_canoe_in_still_water (V_c : ℝ) (V_s : ℝ) (upstream_speed : ℝ) : Prop :=
  V_c - V_s = upstream_speed

def speed_of_stream (V_s : ℝ) : Prop :=
  V_s = 4

-- The statement we want to prove
theorem speed_of_canoe_downstream (V_c V_s : ℝ) (upstream_speed : ℝ) 
  (h1 : speed_of_canoe_in_still_water V_c V_s upstream_speed)
  (h2 : speed_of_stream V_s)
  (h3 : upstream_speed = 4) :
  V_c + V_s = 12 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_canoe_downstream_l1153_115342


namespace NUMINAMATH_GPT_fill_in_the_blank_l1153_115341

-- Definitions of the problem conditions
def parent := "being a parent"
def parent_with_special_needs := "being the parent of a child with special needs"

-- The sentence describing two situations of being a parent
def sentence1 := "Being a parent is not always easy"
def sentence2 := "being the parent of a child with special needs often carries with ___ extra stress."

-- The correct word to fill in the blank.
def correct_answer := "it"

-- Proof problem
theorem fill_in_the_blank : correct_answer = "it" :=
by
  sorry

end NUMINAMATH_GPT_fill_in_the_blank_l1153_115341


namespace NUMINAMATH_GPT_inequality_correctness_l1153_115392

theorem inequality_correctness (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_correctness_l1153_115392


namespace NUMINAMATH_GPT_time_to_fill_tank_l1153_115379

-- Definitions for conditions
def pipe_a := 50
def pipe_b := 75
def pipe_c := 100

-- Definition for the combined rate and time to fill the tank
theorem time_to_fill_tank : 
  (1 / pipe_a + 1 / pipe_b + 1 / pipe_c) * (300 / 13) = 1 := 
by
  sorry

end NUMINAMATH_GPT_time_to_fill_tank_l1153_115379


namespace NUMINAMATH_GPT_new_pressure_of_transferred_gas_l1153_115366

theorem new_pressure_of_transferred_gas (V1 V2 : ℝ) (p1 k : ℝ) 
  (h1 : V1 = 3.5) (h2 : p1 = 8) (h3 : k = V1 * p1) (h4 : V2 = 7) :
  ∃ p2 : ℝ, p2 = 4 ∧ k = V2 * p2 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_new_pressure_of_transferred_gas_l1153_115366


namespace NUMINAMATH_GPT_configuration_of_points_l1153_115381

-- Define a type for points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Assuming general position in the plane
def general_position (points : List Point) : Prop :=
  -- Add definition of general position, skipping exact implementation
  sorry

-- Define the congruence condition
def triangles_congruent (points : List Point) : Prop :=
  -- Add definition of the congruent triangles condition
  sorry

-- Define the vertices of two equilateral triangles inscribed in a circle
def two_equilateral_triangles (points : List Point) : Prop :=
  -- Add definition to check if points form two equilateral triangles in a circle
  sorry

theorem configuration_of_points (points : List Point) (h6 : points.length = 6) :
  general_position points →
  triangles_congruent points →
  two_equilateral_triangles points :=
by
  sorry

end NUMINAMATH_GPT_configuration_of_points_l1153_115381


namespace NUMINAMATH_GPT_average_price_per_dvd_l1153_115352

-- Define the conditions
def num_movies_box1 : ℕ := 10
def price_per_movie_box1 : ℕ := 2
def num_movies_box2 : ℕ := 5
def price_per_movie_box2 : ℕ := 5

-- Define total calculations based on conditions
def total_cost_box1 : ℕ := num_movies_box1 * price_per_movie_box1
def total_cost_box2 : ℕ := num_movies_box2 * price_per_movie_box2

def total_cost : ℕ := total_cost_box1 + total_cost_box2
def total_movies : ℕ := num_movies_box1 + num_movies_box2

-- Define the average price per DVD and prove it to be 3
theorem average_price_per_dvd : total_cost / total_movies = 3 := by
  sorry

end NUMINAMATH_GPT_average_price_per_dvd_l1153_115352


namespace NUMINAMATH_GPT_cost_of_graphing_calculator_l1153_115327

/-
  Everton college paid $1625 for an order of 45 calculators.
  Each scientific calculator costs $10.
  The order included 20 scientific calculators and 25 graphing calculators.
  We need to prove that each graphing calculator costs $57.
-/

namespace EvertonCollege

theorem cost_of_graphing_calculator
  (total_cost : ℕ)
  (cost_scientific : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (cost_graphing : ℕ)
  (h_order : total_cost = 1625)
  (h_cost_scientific : cost_scientific = 10)
  (h_num_scientific : num_scientific = 20)
  (h_num_graphing : num_graphing = 25)
  (h_total_calc : num_scientific + num_graphing = 45)
  (h_pay : total_cost = num_scientific * cost_scientific + num_graphing * cost_graphing) :
  cost_graphing = 57 :=
by
  sorry

end EvertonCollege

end NUMINAMATH_GPT_cost_of_graphing_calculator_l1153_115327


namespace NUMINAMATH_GPT_check_interval_of_quadratic_l1153_115367

theorem check_interval_of_quadratic (z : ℝ) : (z^2 - 40 * z + 344 ≤ 0) ↔ (20 - 2 * Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2 * Real.sqrt 14) :=
sorry

end NUMINAMATH_GPT_check_interval_of_quadratic_l1153_115367


namespace NUMINAMATH_GPT_annular_region_area_l1153_115388

noncomputable def area_annulus (r1 r2 : ℝ) : ℝ :=
  (Real.pi * r2 ^ 2) - (Real.pi * r1 ^ 2)

theorem annular_region_area :
  area_annulus 4 7 = 33 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_annular_region_area_l1153_115388


namespace NUMINAMATH_GPT_sum_ends_in_zero_squares_end_same_digit_l1153_115356

theorem sum_ends_in_zero_squares_end_same_digit (a b : ℕ) (h : (a + b) % 10 = 0) : (a^2 % 10) = (b^2 % 10) := 
sorry

end NUMINAMATH_GPT_sum_ends_in_zero_squares_end_same_digit_l1153_115356


namespace NUMINAMATH_GPT_sin_alpha_value_l1153_115397

-- Given conditions
variables (α : ℝ) (h1 : Real.tan α = -5 / 12) (h2 : π / 2 < α ∧ α < π)

-- Assertion to prove
theorem sin_alpha_value : Real.sin α = 5 / 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sin_alpha_value_l1153_115397


namespace NUMINAMATH_GPT_factor_expression_l1153_115345

theorem factor_expression (y : ℝ) :
  5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1153_115345


namespace NUMINAMATH_GPT_dilution_problem_l1153_115347

theorem dilution_problem
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (desired_concentration : ℝ)
  (initial_alcohol_content : initial_concentration * initial_volume / 100 = 4.8)
  (desired_alcohol_content : 4.8 = desired_concentration * (initial_volume + N) / 100)
  (N : ℝ) :
  N = 11.2 :=
sorry

end NUMINAMATH_GPT_dilution_problem_l1153_115347


namespace NUMINAMATH_GPT_incorrect_comparison_tan_138_tan_143_l1153_115348

theorem incorrect_comparison_tan_138_tan_143 :
  ¬ (Real.tan (Real.pi * 138 / 180) > Real.tan (Real.pi * 143 / 180)) :=
by sorry

end NUMINAMATH_GPT_incorrect_comparison_tan_138_tan_143_l1153_115348


namespace NUMINAMATH_GPT_determine_x0_minus_y0_l1153_115384

theorem determine_x0_minus_y0 
  (x0 y0 : ℝ)
  (data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x0, y0)])
  (regression_eq : ∀ x, (x + 2) = (x + 2)) :
  x0 - y0 = -3 :=
by
  sorry

end NUMINAMATH_GPT_determine_x0_minus_y0_l1153_115384


namespace NUMINAMATH_GPT_hall_length_width_difference_l1153_115377

theorem hall_length_width_difference (L W : ℝ) 
(h1 : W = 1 / 2 * L) 
(h2 : L * W = 200) : L - W = 10 := 
by 
  sorry

end NUMINAMATH_GPT_hall_length_width_difference_l1153_115377


namespace NUMINAMATH_GPT_evaluate_expression_l1153_115311

theorem evaluate_expression (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + (a⁻¹ / 3)) / a = 10 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1153_115311


namespace NUMINAMATH_GPT_Robin_hair_initial_length_l1153_115322

theorem Robin_hair_initial_length (x : ℝ) (h1 : x + 8 - 20 = 2) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_Robin_hair_initial_length_l1153_115322


namespace NUMINAMATH_GPT_merchant_mixture_solution_l1153_115389

variable (P C : ℝ)

def P_price : ℝ := 2.40
def C_price : ℝ := 6.00
def total_weight : ℝ := 60
def total_price_per_pound : ℝ := 3.00
def total_price : ℝ := total_price_per_pound * total_weight

theorem merchant_mixture_solution (h1 : P + C = total_weight)
                                  (h2 : P_price * P + C_price * C = total_price) :
  C = 10 := 
sorry

end NUMINAMATH_GPT_merchant_mixture_solution_l1153_115389


namespace NUMINAMATH_GPT_lines_intersect_at_l1153_115325

noncomputable def L₁ (t : ℝ) : ℝ × ℝ := (2 - t, -3 + 4 * t)
noncomputable def L₂ (u : ℝ) : ℝ × ℝ := (-1 + 5 * u, 6 - 7 * u)
noncomputable def point_of_intersection : ℝ × ℝ := (2 / 13, 69 / 13)

theorem lines_intersect_at :
  ∃ t u : ℝ, L₁ t = point_of_intersection ∧ L₂ u = point_of_intersection := 
sorry

end NUMINAMATH_GPT_lines_intersect_at_l1153_115325


namespace NUMINAMATH_GPT_sum_three_numbers_l1153_115394

theorem sum_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 23 := by
  sorry

end NUMINAMATH_GPT_sum_three_numbers_l1153_115394


namespace NUMINAMATH_GPT_solution_system_of_inequalities_l1153_115313

theorem solution_system_of_inequalities (x : ℝ) : 
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * (x^2) - x - 1 > 0 ↔ (-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_system_of_inequalities_l1153_115313


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_in_arithmetic_sequence_l1153_115318

theorem sum_of_inserted_numbers_in_arithmetic_sequence :
  ∃ a2 a3 : ℤ, 2015 > a2 ∧ a2 > a3 ∧ a3 > 131 ∧ (2015 - a2) = (a2 - a3) ∧ (a2 - a3) = (a3 - 131) ∧ (a2 + a3) = 2146 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_in_arithmetic_sequence_l1153_115318


namespace NUMINAMATH_GPT_sqrt_ab_eq_18_l1153_115359

noncomputable def a := Real.log 9 / Real.log 4
noncomputable def b := 108 * (Real.log 8 / Real.log 3)

theorem sqrt_ab_eq_18 : Real.sqrt (a * b) = 18 := by
  sorry

end NUMINAMATH_GPT_sqrt_ab_eq_18_l1153_115359


namespace NUMINAMATH_GPT_minimum_odd_numbers_in_set_l1153_115302

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end NUMINAMATH_GPT_minimum_odd_numbers_in_set_l1153_115302


namespace NUMINAMATH_GPT_total_units_is_34_l1153_115319

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end NUMINAMATH_GPT_total_units_is_34_l1153_115319


namespace NUMINAMATH_GPT_fraction_of_unoccupied_chairs_is_two_fifths_l1153_115365

noncomputable def fraction_unoccupied_chairs (total_chairs : ℕ) (chair_capacity : ℕ) (attended_board_members : ℕ) : ℚ :=
  let total_capacity := total_chairs * chair_capacity
  let total_board_members := total_capacity
  let unoccupied_members := total_board_members - attended_board_members
  let unoccupied_chairs := unoccupied_members / chair_capacity
  unoccupied_chairs / total_chairs

theorem fraction_of_unoccupied_chairs_is_two_fifths :
  fraction_unoccupied_chairs 40 2 48 = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_unoccupied_chairs_is_two_fifths_l1153_115365


namespace NUMINAMATH_GPT_problem1_l1153_115334

theorem problem1 (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (Real.pi / 2 - α)^2 + 3 * Real.sin (α + Real.pi) * Real.sin (α + Real.pi / 2) = -1 :=
sorry

end NUMINAMATH_GPT_problem1_l1153_115334


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_conclusions_l1153_115338

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set_conclusions (h1 : ∀ x, -1 ≤ x ∧ x ≤ 2 → ax^2 + bx + c ≥ 0)
(h2 : ∀ x, x < -1 ∨ x > 2 → ax^2 + bx + c < 0) :
(a + b = 0) ∧ (a + b + c > 0) ∧ (c > 0) ∧ ¬ (b < 0) := by
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_conclusions_l1153_115338


namespace NUMINAMATH_GPT_largest_of_four_l1153_115396

theorem largest_of_four : 
  let a := 1 
  let b := 0 
  let c := |(-2)| 
  let d := -3 
  max (max (max a b) c) d = c := by
  sorry

end NUMINAMATH_GPT_largest_of_four_l1153_115396


namespace NUMINAMATH_GPT_completing_square_correct_l1153_115323

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end NUMINAMATH_GPT_completing_square_correct_l1153_115323


namespace NUMINAMATH_GPT_proof_problem_l1153_115349

noncomputable def polar_to_cartesian_O1 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = 4 * Real.cos θ → (ρ^2 = 4 * ρ * Real.cos θ)

noncomputable def cartesian_O1 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = 4 * x → x^2 + y^2 - 4 * x = 0

noncomputable def polar_to_cartesian_O2 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = -4 * Real.sin θ → (ρ^2 = -4 * ρ * Real.sin θ)

noncomputable def cartesian_O2 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = -4 * y → x^2 + y^2 + 4 * y = 0

noncomputable def intersections_O1_O2 : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 + 4 * y = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)

noncomputable def line_through_intersections : Prop :=
  ∀ (x y : ℝ), ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)) → y = -x

theorem proof_problem : polar_to_cartesian_O1 ∧ cartesian_O1 ∧ polar_to_cartesian_O2 ∧ cartesian_O2 ∧ intersections_O1_O2 ∧ line_through_intersections :=
  sorry

end NUMINAMATH_GPT_proof_problem_l1153_115349


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1153_115363

noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio 
  (S : ℕ → ℝ) 
  (hS12 : S 12 = 1)
  (hS6 : S 6 = 2)
  (geom_property : ∀ a r, (S n = a * (1 - r^n) / (1 - r))) :
  S 18 / S 6 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1153_115363


namespace NUMINAMATH_GPT_identify_incorrect_calculation_l1153_115343

theorem identify_incorrect_calculation : 
  (∀ x : ℝ, x^2 * x^3 = x^5) ∧ 
  (∀ x : ℝ, x^3 + x^3 = 2 * x^3) ∧ 
  (∀ x : ℝ, x^6 / x^2 = x^4) ∧ 
  ¬ (∀ x : ℝ, (-3 * x)^2 = 6 * x^2) := 
by
  sorry

end NUMINAMATH_GPT_identify_incorrect_calculation_l1153_115343


namespace NUMINAMATH_GPT_addition_amount_first_trial_l1153_115320

theorem addition_amount_first_trial :
  ∀ (a b : ℝ),
  20 ≤ a ∧ a ≤ 30 ∧ 20 ≤ b ∧ b ≤ 30 → (a = 20 + (30 - 20) * 0.618 ∨ b = 30 - (30 - 20) * 0.618) :=
by {
  sorry
}

end NUMINAMATH_GPT_addition_amount_first_trial_l1153_115320


namespace NUMINAMATH_GPT_geometric_sequence_17th_term_l1153_115336

variable {α : Type*} [Field α]

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

theorem geometric_sequence_17th_term :
  ∀ (a r : α),
    a * r ^ 4 = 9 →  -- Fifth term condition
    a * r ^ 12 = 1152 →  -- Thirteenth term condition
    a * r ^ 16 = 36864 :=  -- Seventeenth term conclusion
by
  intros a r h5 h13
  sorry

end NUMINAMATH_GPT_geometric_sequence_17th_term_l1153_115336


namespace NUMINAMATH_GPT_count_non_integer_angles_l1153_115358

open Int

def interior_angle (n : ℕ) : ℕ := 180 * (n - 2) / n

def is_integer_angle (n : ℕ) : Prop := 180 * (n - 2) % n = 0

theorem count_non_integer_angles : ∃ (count : ℕ), count = 2 ∧ ∀ n, 3 ≤ n ∧ n < 12 → is_integer_angle n ↔ ¬ (count = count + 1) :=
sorry

end NUMINAMATH_GPT_count_non_integer_angles_l1153_115358
