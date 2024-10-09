import Mathlib

namespace non_negative_integer_solutions_l2053_205301

theorem non_negative_integer_solutions (x : ℕ) : 3 * x - 2 < 7 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end non_negative_integer_solutions_l2053_205301


namespace no_possible_values_of_k_l2053_205340

theorem no_possible_values_of_k :
  ¬(∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65) :=
by
  sorry

end no_possible_values_of_k_l2053_205340


namespace ratio_triangle_square_l2053_205307

noncomputable def square_area (s : ℝ) : ℝ := s * s

noncomputable def triangle_PTU_area (s : ℝ) : ℝ := 1 / 2 * (s / 2) * (s / 2)

theorem ratio_triangle_square (s : ℝ) (h : s > 0) : 
  triangle_PTU_area s / square_area s = 1 / 8 := 
sorry

end ratio_triangle_square_l2053_205307


namespace michael_initial_money_l2053_205386

theorem michael_initial_money 
  (M B_initial B_left B_spent : ℕ) 
  (h_split : M / 2 = B_initial - B_left + B_spent): 
  (M / 2 + B_left = 17 + 35) → M = 152 :=
by
  sorry

end michael_initial_money_l2053_205386


namespace overall_percentage_support_l2053_205373

theorem overall_percentage_support (p_men : ℕ) (p_women : ℕ) (n_men : ℕ) (n_women : ℕ) : 
  (p_men = 55) → (p_women = 80) → (n_men = 200) → (n_women = 800) → 
  (p_men * n_men + p_women * n_women) / (n_men + n_women) = 75 :=
by
  sorry

end overall_percentage_support_l2053_205373


namespace problem_solution_l2053_205392

-- Define the arithmetic sequence and its sum
def arith_seq_sum (n : ℕ) (a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Define the specific condition for our problem
def a1_a5_equal_six (a1 d : ℕ) : Prop :=
  a1 + (a1 + 4 * d) = 6

-- The target value of S5 that we want to prove
def S5 (a1 d : ℕ) : ℕ :=
  arith_seq_sum 5 a1 d

theorem problem_solution (a1 d : ℕ) (h : a1_a5_equal_six a1 d) : S5 a1 d = 15 :=
by
  sorry

end problem_solution_l2053_205392


namespace integer_ratio_value_l2053_205382

theorem integer_ratio_value {x y : ℝ} (h1 : 3 < (x^2 - y^2) / (x^2 + y^2)) (h2 : (x^2 - y^2) / (x^2 + y^2) < 4) (h3 : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = 2 :=
by
  sorry

end integer_ratio_value_l2053_205382


namespace reducible_fraction_l2053_205302

theorem reducible_fraction (l : ℤ) : ∃ k : ℤ, l = 13 * k + 4 ↔ (∃ d > 1, d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7)) :=
sorry

end reducible_fraction_l2053_205302


namespace chairs_per_row_l2053_205305

-- Definition of the given conditions
def rows : ℕ := 20
def people_per_chair : ℕ := 5
def total_people : ℕ := 600

-- The statement to be proven
theorem chairs_per_row (x : ℕ) (h : rows * (x * people_per_chair) = total_people) : x = 6 := 
by sorry

end chairs_per_row_l2053_205305


namespace simplify_expression_l2053_205396

theorem simplify_expression (w : ℕ) : 
  4 * w + 6 * w + 8 * w + 10 * w + 12 * w + 14 * w + 16 = 54 * w + 16 :=
by 
  sorry

end simplify_expression_l2053_205396


namespace negation_of_exists_l2053_205327

theorem negation_of_exists (h : ¬ (∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0)) : ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_of_exists_l2053_205327


namespace quadratic_has_real_roots_l2053_205336

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, x^2 + x - 4 * m = 0) ↔ m ≥ -1 / 16 :=
by
  sorry

end quadratic_has_real_roots_l2053_205336


namespace inequality_true_l2053_205366

theorem inequality_true (a b : ℝ) (h : a^2 + b^2 > 1) : |a| + |b| > 1 :=
sorry

end inequality_true_l2053_205366


namespace compare_values_l2053_205388

-- Define that f(x) is an even function, periodic and satisfies decrease and increase conditions as given
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

noncomputable def f : ℝ → ℝ := sorry -- the exact definition of f is unknown, so we use sorry for now

-- The conditions of the problem
axiom f_even : is_even_function f
axiom f_period : periodic_function f 2
axiom f_decreasing : decreasing_on_interval f (-1) 0
axiom f_transformation : ∀ x, f (x + 1) = 1 / f x

-- Prove the comparison between a, b, and c under the given conditions
theorem compare_values (a b c : ℝ) (h1 : a = f (Real.log 2 / Real.log 5)) (h2 : b = f (Real.log 4 / Real.log 2)) (h3 : c = f (Real.sqrt 2)) :
  a > c ∧ c > b :=
by
  sorry

end compare_values_l2053_205388


namespace cost_price_computer_table_l2053_205362

theorem cost_price_computer_table (CP SP : ℝ) (h1 : SP = 1.15 * CP) (h2 : SP = 6400) : CP = 5565.22 :=
by sorry

end cost_price_computer_table_l2053_205362


namespace cistern_height_l2053_205393

theorem cistern_height (l w A : ℝ) (h : ℝ) (hl : l = 8) (hw : w = 6) (hA : 48 + 2 * (l * h) + 2 * (w * h) = 99.8) : h = 1.85 := by
  sorry

end cistern_height_l2053_205393


namespace initial_investment_l2053_205318

noncomputable def doubling_period (r : ℝ) : ℝ := 70 / r
noncomputable def investment_after_doubling (P : ℝ) (n : ℝ) : ℝ := P * (2 ^ n)

theorem initial_investment (total_amount : ℝ) (years : ℝ) (rate : ℝ) (initial : ℝ) :
  rate = 8 → total_amount = 28000 → years = 18 → 
  initial = total_amount / (2 ^ (years / (doubling_period rate))) :=
by
  intros hrate htotal hyears
  simp [doubling_period, investment_after_doubling] at *
  rw [hrate, htotal, hyears]
  norm_num
  sorry

end initial_investment_l2053_205318


namespace f_19_eq_2017_l2053_205349

noncomputable def f : ℤ → ℤ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ m n : ℤ, f (m + n) = f m + f n + 3 * (4 * m * n - 1)

theorem f_19_eq_2017 : f 19 = 2017 := by
  sorry

end f_19_eq_2017_l2053_205349


namespace prob_A_two_qualified_l2053_205375

noncomputable def prob_qualified (p : ℝ) : ℝ := p * p

def qualified_rate : ℝ := 0.8

theorem prob_A_two_qualified : prob_qualified qualified_rate = 0.64 :=
by
  sorry

end prob_A_two_qualified_l2053_205375


namespace jason_cards_l2053_205399

theorem jason_cards :
  (initial_cards - bought_cards = remaining_cards) →
  initial_cards = 676 →
  bought_cards = 224 →
  remaining_cards = 452 :=
by
  intros h1 h2 h3
  sorry

end jason_cards_l2053_205399


namespace motorcycle_tire_max_distance_l2053_205312

theorem motorcycle_tire_max_distance :
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  let s := 18750
  wear_front * (s / 2) + wear_rear * (s / 2) = 1 :=
by 
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  sorry

end motorcycle_tire_max_distance_l2053_205312


namespace find_digits_l2053_205325

def five_digit_subtraction (a b c d e : ℕ) : Prop :=
    let n1 := 10000 * a + 1000 * b + 100 * c + 10 * d + e
    let n2 := 10000 * e + 1000 * d + 100 * c + 10 * b + a
    (n1 - n2) % 10 = 2 ∧ (((n1 - n2) / 10) % 10) = 7 ∧ a > e ∧ a - e = 2 ∧ b - a = 7

theorem find_digits 
    (a b c d e : ℕ) 
    (h : five_digit_subtraction a b c d e) :
    a = 9 ∧ e = 7 :=
by 
    sorry

end find_digits_l2053_205325


namespace shaded_area_triangle_l2053_205346

theorem shaded_area_triangle (a b : ℝ) (h1 : a = 5) (h2 : b = 15) :
  let area_shaded : ℝ := (5^2) - (1/2 * ((15 / 4) * 5))
  area_shaded = 175 / 8 := 
by
  sorry

end shaded_area_triangle_l2053_205346


namespace difference_of_numbers_l2053_205356

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : abs (x - y) = 7 :=
sorry

end difference_of_numbers_l2053_205356


namespace age_of_15th_student_is_15_l2053_205328

-- Define the total number of students
def total_students : Nat := 15

-- Define the average age of all 15 students together
def avg_age_all_students : Nat := 15

-- Define the average age of the first group of 7 students
def avg_age_first_group : Nat := 14

-- Define the average age of the second group of 7 students
def avg_age_second_group : Nat := 16

-- Define the total age based on the average age and number of students
def total_age_all_students : Nat := total_students * avg_age_all_students
def total_age_first_group : Nat := 7 * avg_age_first_group
def total_age_second_group : Nat := 7 * avg_age_second_group

-- Define the age of the 15th student
def age_of_15th_student : Nat := total_age_all_students - (total_age_first_group + total_age_second_group)

-- Theorem: prove that the age of the 15th student is 15 years
theorem age_of_15th_student_is_15 : age_of_15th_student = 15 := by
  -- The proof will go here
  sorry

end age_of_15th_student_is_15_l2053_205328


namespace construct_segment_AB_l2053_205326

-- Define the two points A and B and assume the distance between them is greater than 1 meter
variables {A B : Point} (dist_AB_gt_1m : Distance A B > 1)

-- Define the ruler length as 10 cm
def ruler_length : ℝ := 0.1

theorem construct_segment_AB 
  (h : dist_AB_gt_1m) 
  (ruler : ℝ := ruler_length) : ∃ (AB : Segment), Distance A B = AB.length ∧ AB.length > 1 :=
sorry

end construct_segment_AB_l2053_205326


namespace scale_division_remainder_l2053_205300

theorem scale_division_remainder (a b c r : ℕ) (h1 : a = b * c + r) (h2 : 0 ≤ r) (h3 : r < b) :
  (3 * a) % (3 * b) = 3 * r :=
sorry

end scale_division_remainder_l2053_205300


namespace intersection_of_S_and_complement_of_T_in_U_l2053_205338

def U : Set ℕ := { x | 0 ≤ x ∧ x ≤ 8 }
def S : Set ℕ := { 1, 2, 4, 5 }
def T : Set ℕ := { 3, 5, 7 }
def C_U_T : Set ℕ := { x | x ∈ U ∧ x ∉ T }

theorem intersection_of_S_and_complement_of_T_in_U :
  S ∩ C_U_T = { 1, 2, 4 } :=
by
  sorry

end intersection_of_S_and_complement_of_T_in_U_l2053_205338


namespace loaned_out_books_is_50_l2053_205372

-- Define the conditions
def initial_books : ℕ := 75
def end_books : ℕ := 60
def percent_returned : ℝ := 0.70

-- Define the variable to represent the number of books loaned out
noncomputable def loaned_out_books := (15:ℝ) / (1 - percent_returned)

-- The target theorem statement we need to prove
theorem loaned_out_books_is_50 : loaned_out_books = 50 :=
by
  sorry

end loaned_out_books_is_50_l2053_205372


namespace remainder_of_82460_div_8_l2053_205354

theorem remainder_of_82460_div_8 :
  82460 % 8 = 4 :=
sorry

end remainder_of_82460_div_8_l2053_205354


namespace range_of_a_minus_abs_b_l2053_205379

theorem range_of_a_minus_abs_b (a b : ℝ) (h1 : 1 < a ∧ a < 8) (h2 : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 8 :=
sorry

end range_of_a_minus_abs_b_l2053_205379


namespace intersection_point_exists_l2053_205380

theorem intersection_point_exists :
  ∃ t u x y : ℚ,
    (x = 2 + 3 * t) ∧ (y = 3 - 4 * t) ∧
    (x = 4 + 5 * u) ∧ (y = -6 + u) ∧
    (x = 175 / 23) ∧ (y = 19 / 23) :=
by
  sorry

end intersection_point_exists_l2053_205380


namespace men_women_arrangement_l2053_205352

theorem men_women_arrangement :
  let men := 2
  let women := 4
  let slots := 5
  (Nat.choose slots women) * women.factorial * men.factorial = 240 :=
by
  sorry

end men_women_arrangement_l2053_205352


namespace shopkeeper_total_cards_l2053_205377

-- Definition of the number of cards in standard, Uno, and tarot decks.
def std_deck := 52
def uno_deck := 108
def tarot_deck := 78

-- Number of complete decks and additional cards.
def std_decks := 4
def uno_decks := 3
def tarot_decks := 5
def additional_std := 12
def additional_uno := 7
def additional_tarot := 9

-- Calculate the total number of cards.
def total_standard_cards := (std_decks * std_deck) + additional_std
def total_uno_cards := (uno_decks * uno_deck) + additional_uno
def total_tarot_cards := (tarot_decks * tarot_deck) + additional_tarot

def total_cards := total_standard_cards + total_uno_cards + total_tarot_cards

theorem shopkeeper_total_cards : total_cards = 950 := by
  sorry

end shopkeeper_total_cards_l2053_205377


namespace abs_expression_value_l2053_205309

theorem abs_expression_value (x : ℤ) (h : x = -2023) :
  abs (2 * abs (abs x - x) - abs x) - x = 8092 :=
by {
  -- Proof will be provided here
  sorry
}

end abs_expression_value_l2053_205309


namespace lcm_144_132_eq_1584_l2053_205387

theorem lcm_144_132_eq_1584 :
  Nat.lcm 144 132 = 1584 :=
by
  sorry

end lcm_144_132_eq_1584_l2053_205387


namespace radius_of_circle_l2053_205321

open Complex

theorem radius_of_circle (z : ℂ) (h : (z + 2)^4 = 16 * z^4) : abs z = 2 / Real.sqrt 3 :=
sorry

end radius_of_circle_l2053_205321


namespace total_water_hold_l2053_205389

variables
  (first : ℕ := 100)
  (second : ℕ := 150)
  (third : ℕ := 75)
  (total : ℕ := 325)

theorem total_water_hold :
  first + second + third = total := by
  sorry

end total_water_hold_l2053_205389


namespace boiling_point_C_l2053_205367

-- Water boils at 212 °F
def water_boiling_point_F : ℝ := 212
-- Ice melts at 32 °F
def ice_melting_point_F : ℝ := 32
-- Ice melts at 0 °C
def ice_melting_point_C : ℝ := 0
-- The temperature of a pot of water in °C
def pot_water_temp_C : ℝ := 40
-- The temperature of the pot of water in °F
def pot_water_temp_F : ℝ := 104

-- The boiling point of water in Celsius is 100 °C.
theorem boiling_point_C : water_boiling_point_F = 212 ∧ ice_melting_point_F = 32 ∧ ice_melting_point_C = 0 ∧ pot_water_temp_C = 40 ∧ pot_water_temp_F = 104 → exists bp_C : ℝ, bp_C = 100 :=
by
  sorry

end boiling_point_C_l2053_205367


namespace quarters_total_l2053_205323

def initial_quarters : ℕ := 21
def additional_quarters : ℕ := 49
def total_quarters : ℕ := initial_quarters + additional_quarters

theorem quarters_total : total_quarters = 70 := by
  sorry

end quarters_total_l2053_205323


namespace abs_neg_two_eq_two_l2053_205313

theorem abs_neg_two_eq_two : abs (-2) = 2 :=
sorry

end abs_neg_two_eq_two_l2053_205313


namespace Oliver_ferris_wheel_rides_l2053_205368

theorem Oliver_ferris_wheel_rides :
  ∃ (F : ℕ), (4 * 7 + F * 7 = 63) ∧ (F = 5) :=
by
  sorry

end Oliver_ferris_wheel_rides_l2053_205368


namespace quadratic_equation_solution_unique_l2053_205347

noncomputable def b_solution := (-3 + 3 * Real.sqrt 21) / 2
noncomputable def c_solution := (33 - 3 * Real.sqrt 21) / 2

theorem quadratic_equation_solution_unique :
  (∃ (b c : ℝ), 
     (∀ (x : ℝ), 3 * x^2 + b * x + c = 0 → x = b_solution) ∧ 
     b + c = 15 ∧ 3 * c = b^2 ∧
     b = b_solution ∧ c = c_solution) :=
by { sorry }

end quadratic_equation_solution_unique_l2053_205347


namespace f_iterate_result_l2053_205303

def f (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1 else 4*n - 3

theorem f_iterate_result : f (f (f 1)) = 17 :=
by
  sorry

end f_iterate_result_l2053_205303


namespace negation_proof_l2053_205310

theorem negation_proof :
  ¬(∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1 :=
by sorry

end negation_proof_l2053_205310


namespace min_value_expression_l2053_205348

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    9 ≤ (5 * z / (2 * x + y) + 5 * x / (y + 2 * z) + 2 * y / (x + z) + (x + y + z) / (x * y + y * z + z * x)) :=
sorry

end min_value_expression_l2053_205348


namespace molecular_weight_K3AlC2O4_3_l2053_205353

noncomputable def molecularWeightOfCompound : ℝ :=
  let potassium_weight : ℝ := 39.10
  let aluminum_weight  : ℝ := 26.98
  let carbon_weight    : ℝ := 12.01
  let oxygen_weight    : ℝ := 16.00
  let total_potassium_weight : ℝ := 3 * potassium_weight
  let total_aluminum_weight  : ℝ := aluminum_weight
  let total_carbon_weight    : ℝ := 3 * 2 * carbon_weight
  let total_oxygen_weight    : ℝ := 3 * 4 * oxygen_weight
  total_potassium_weight + total_aluminum_weight + total_carbon_weight + total_oxygen_weight

theorem molecular_weight_K3AlC2O4_3 : molecularWeightOfCompound = 408.34 := by
  sorry

end molecular_weight_K3AlC2O4_3_l2053_205353


namespace solve_for_q_l2053_205344

theorem solve_for_q :
  ∀ (q : ℕ), 16^15 = 4^q → q = 30 :=
by
  intro q
  intro h
  sorry

end solve_for_q_l2053_205344


namespace smallest_positive_phi_l2053_205324

open Real

theorem smallest_positive_phi :
  (∃ k : ℤ, (2 * φ + π / 4 = π / 2 + k * π)) →
  (∀ k, φ = π / 8 + k * π / 2) → 
  0 < φ → 
  φ = π / 8 :=
by
  sorry

end smallest_positive_phi_l2053_205324


namespace dave_age_l2053_205363

theorem dave_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end dave_age_l2053_205363


namespace polyhedron_edges_faces_vertices_l2053_205334

theorem polyhedron_edges_faces_vertices
  (E F V n m : ℕ)
  (h1 : n * F = 2 * E)
  (h2 : m * V = 2 * E)
  (h3 : V + F = E + 2) :
  ¬(m * F = 2 * E) :=
sorry

end polyhedron_edges_faces_vertices_l2053_205334


namespace inequality_holds_l2053_205333

theorem inequality_holds 
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b < 0) 
  (h3 : b > c) : 
  (a / (c^2)) > (b / (c^2)) :=
by
  sorry

end inequality_holds_l2053_205333


namespace pears_in_basket_l2053_205397

def TaniaFruits (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 18 ∧ b2 = 12 ∧ b3 = 9 ∧ b4 = b3 ∧ b5 + b1 + b2 + b3 + b4 = 58

theorem pears_in_basket {b1 b2 b3 b4 b5 : ℕ} (h : TaniaFruits b1 b2 b3 b4 b5) : b5 = 10 :=
by 
  sorry

end pears_in_basket_l2053_205397


namespace g_odd_l2053_205315

def g (x : ℝ) : ℝ := x^3 - 2*x

theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_odd_l2053_205315


namespace solution_set_l2053_205319

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 3 * x - 4

-- Define the inequality
def inequality (x : ℝ) : Prop := quadratic_expr x > 0

-- State the theorem
theorem solution_set : ∀ x : ℝ, inequality x ↔ (x > 1 ∨ x < -4) :=
by
  sorry

end solution_set_l2053_205319


namespace horatio_sonnets_count_l2053_205395

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end horatio_sonnets_count_l2053_205395


namespace complex_fraction_l2053_205304

open Complex

/-- The given complex fraction \(\frac{5 - i}{1 - i}\) evaluates to \(3 + 2i\). -/
theorem complex_fraction : (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = ⟨3, 2⟩ :=
  by
  sorry

end complex_fraction_l2053_205304


namespace isosceles_triangle_angles_l2053_205358

theorem isosceles_triangle_angles (a b : ℝ) (h₁ : a = 80 ∨ b = 80) (h₂ : a + b + c = 180) (h_iso : a = b ∨ a = c ∨ b = c) :
  (a = 80 ∧ b = 20 ∧ c = 80)
  ∨ (a = 80 ∧ b = 80 ∧ c = 20)
  ∨ (a = 50 ∧ b = 50 ∧ c = 80) :=
by sorry

end isosceles_triangle_angles_l2053_205358


namespace smaller_triangle_perimeter_l2053_205337

theorem smaller_triangle_perimeter (p : ℕ) (h : p * 3 = 120) : p = 40 :=
sorry

end smaller_triangle_perimeter_l2053_205337


namespace regular_polygon_perimeter_l2053_205317

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end regular_polygon_perimeter_l2053_205317


namespace final_value_of_x_l2053_205351

noncomputable def initial_x : ℝ := 52 * 1.2
noncomputable def decreased_x : ℝ := initial_x * 0.9
noncomputable def final_x : ℝ := decreased_x * 1.15

theorem final_value_of_x : final_x = 64.584 := by
  sorry

end final_value_of_x_l2053_205351


namespace number_of_apples_remaining_l2053_205350

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end number_of_apples_remaining_l2053_205350


namespace solve_inequality_l2053_205381

open Real

noncomputable def expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - 4*x + 3) + 1) * log x / (log 2 * 5) + (1 / x) * (sqrt (8 * x - 2 * x^2 - 6) + 1)

theorem solve_inequality :
  ∃ x : ℝ, x = 1 ∧
    (x > 0) ∧
    (x^2 - 4 * x + 3 ≥ 0) ∧
    (8 * x - 2 * x^2 - 6 ≥ 0) ∧
    expression x ≤ 0 :=
by
  sorry

end solve_inequality_l2053_205381


namespace scooter_gain_percent_l2053_205383

theorem scooter_gain_percent 
  (purchase_price : ℕ) 
  (repair_costs : ℕ) 
  (selling_price : ℕ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by
  sorry

end scooter_gain_percent_l2053_205383


namespace sum_of_c_and_d_l2053_205359

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

theorem sum_of_c_and_d (c d : ℝ) (h_asymptote1 : (2:ℝ)^2 + c * 2 + d = 0) (h_asymptote2 : (-1:ℝ)^2 - c + d = 0) :
  c + d = -3 :=
by
-- theorem body (proof omitted)
sorry

end sum_of_c_and_d_l2053_205359


namespace significant_digits_of_side_length_l2053_205343

noncomputable def num_significant_digits (n : Float) : Nat :=
  -- This is a placeholder function to determine the number of significant digits
  sorry

theorem significant_digits_of_side_length :
  ∀ (A : Float), A = 3.2400 → num_significant_digits (Float.sqrt A) = 5 :=
by
  intro A h
  -- Proof would go here
  sorry

end significant_digits_of_side_length_l2053_205343


namespace find_x_l2053_205394

theorem find_x (x y : ℝ) :
  (x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1)) →
  x = (y^2 + 3*y + 2) / 3 :=
by
  intro h
  sorry

end find_x_l2053_205394


namespace range_alpha_div_three_l2053_205320

open Real

theorem range_alpha_div_three (α : ℝ) (k : ℤ) :
  sin α > 0 → cos α < 0 → sin (α / 3) > cos (α / 3) →
  ∃ k : ℤ,
    (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) ∨
    (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
by
  intros
  sorry

end range_alpha_div_three_l2053_205320


namespace spilled_wax_amount_l2053_205365

-- Definitions based on conditions
def car_wax := 3
def suv_wax := 4
def total_wax := 11
def remaining_wax := 2

-- The theorem to be proved
theorem spilled_wax_amount : car_wax + suv_wax + (total_wax - remaining_wax - (car_wax + suv_wax)) = total_wax - remaining_wax :=
by
  sorry


end spilled_wax_amount_l2053_205365


namespace derivative_y_l2053_205361

noncomputable def y (a α x : ℝ) :=
  (Real.exp (a * x)) * (3 * Real.sin (3 * x) - α * Real.cos (3 * x)) / (a ^ 2 + 9)

theorem derivative_y (a α x : ℝ) :
  (deriv (y a α) x) =
    (Real.exp (a * x)) * ((3 * a + 3 * α) * Real.sin (3 * x) + (9 - a * α) * Real.cos (3 * x)) / (a ^ 2 + 9) := 
sorry

end derivative_y_l2053_205361


namespace determine_irrational_option_l2053_205345

def is_irrational (x : ℝ) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

def option_A : ℝ := 7
def option_B : ℝ := 0.5
def option_C : ℝ := abs (3 / 20 : ℚ)
def option_D : ℝ := 0.5151151115 -- Assume notation describes the stated behavior

theorem determine_irrational_option :
  is_irrational option_D ∧
  ¬ is_irrational option_A ∧
  ¬ is_irrational option_B ∧
  ¬ is_irrational option_C := 
by
  sorry

end determine_irrational_option_l2053_205345


namespace value_of_expression_l2053_205316

variables (m n c d : ℝ)
variables (h1 : m = -n) (h2 : c * d = 1)

theorem value_of_expression : m + n + 3 * c * d - 10 = -7 :=
by sorry

end value_of_expression_l2053_205316


namespace other_number_is_36_l2053_205390

theorem other_number_is_36 (hcf lcm given_number other_number : ℕ) 
  (hcf_val : hcf = 16) (lcm_val : lcm = 396) (given_number_val : given_number = 176) 
  (relation : hcf * lcm = given_number * other_number) : 
  other_number = 36 := 
by 
  sorry

end other_number_is_36_l2053_205390


namespace volume_of_rectangular_solid_l2053_205370

theorem volume_of_rectangular_solid (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 15) 
  (h3 : z * x = 10) : 
  x * y * z = 30 * Real.sqrt 3 := 
sorry

end volume_of_rectangular_solid_l2053_205370


namespace right_triangle_sum_of_legs_l2053_205385

theorem right_triangle_sum_of_legs (a b : ℝ) (h₁ : a^2 + b^2 = 2500) (h₂ : (1 / 2) * a * b = 600) : a + b = 70 :=
sorry

end right_triangle_sum_of_legs_l2053_205385


namespace smaller_angle_at_6_30_l2053_205339
-- Import the Mathlib library

-- Define the conditions as a structure
structure ClockAngleConditions where
  hours_on_clock : ℕ
  degrees_per_hour : ℕ
  minute_hand_position : ℕ
  hour_hand_position : ℕ

-- Initialize the conditions for 6:30
def conditions : ClockAngleConditions := {
  hours_on_clock := 12,
  degrees_per_hour := 30,
  minute_hand_position := 180,
  hour_hand_position := 195
}

-- Define the theorem to be proven
theorem smaller_angle_at_6_30 (c : ClockAngleConditions) : 
  c.hour_hand_position - c.minute_hand_position = 15 :=
by
  -- Skip the proof
  sorry

end smaller_angle_at_6_30_l2053_205339


namespace number_of_chocolate_boxes_l2053_205314

theorem number_of_chocolate_boxes
  (x y p : ℕ)
  (pieces_per_box : ℕ)
  (total_candies : ℕ)
  (h_y : y = 4)
  (h_pieces : pieces_per_box = 9)
  (h_total : total_candies = 90) :
  x = 6 :=
by
  -- Definitions of the conditions
  let caramel_candies := y * pieces_per_box
  let total_chocolate_candies := total_candies - caramel_candies
  let x := total_chocolate_candies / pieces_per_box
  
  -- Main theorem statement: x = 6
  sorry

end number_of_chocolate_boxes_l2053_205314


namespace remainder_when_divided_by_24_l2053_205378

theorem remainder_when_divided_by_24 (m k : ℤ) (h : m = 288 * k + 47) : m % 24 = 23 :=
by
  sorry

end remainder_when_divided_by_24_l2053_205378


namespace equal_area_bisecting_line_slope_l2053_205384

theorem equal_area_bisecting_line_slope 
  (circle1_center circle2_center : ℝ × ℝ) 
  (radius : ℝ) 
  (line_point : ℝ × ℝ) 
  (h1 : circle1_center = (20, 100))
  (h2 : circle2_center = (25, 90))
  (h3 : radius = 4)
  (h4 : line_point = (20, 90))
  : ∃ (m : ℝ), |m| = 2 :=
by
  sorry

end equal_area_bisecting_line_slope_l2053_205384


namespace face_value_of_share_l2053_205398

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end face_value_of_share_l2053_205398


namespace range_of_a_l2053_205308

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by {
  sorry
}

end range_of_a_l2053_205308


namespace right_triangle_count_l2053_205306

theorem right_triangle_count (a b : ℕ) (h1 : b < 100) (h2 : a^2 + b^2 = (b + 2)^2) : 
∃ n, n = 10 :=
by sorry

end right_triangle_count_l2053_205306


namespace find_a_l2053_205335

theorem find_a (a : ℝ) : (∃ x y : ℝ, y = 4 - 3 * x ∧ y = 2 * x - 1 ∧ y = a * x + 7) → a = 6 := 
by
  sorry

end find_a_l2053_205335


namespace acute_triangle_l2053_205330

theorem acute_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
                       (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a)
                       (h7 : a^3 + b^3 = c^3) :
                       c^2 < a^2 + b^2 :=
by {
  sorry
}

end acute_triangle_l2053_205330


namespace find_number_l2053_205329

theorem find_number (x : ℝ) 
  (h1 : 0.15 * 40 = 6) 
  (h2 : 6 = 0.25 * x + 2) : 
  x = 16 := 
sorry

end find_number_l2053_205329


namespace parallel_lines_of_equation_l2053_205332

theorem parallel_lines_of_equation (y : Real) :
  (y - 2) * (y + 3) = 0 → (y = 2 ∨ y = -3) :=
by
  sorry

end parallel_lines_of_equation_l2053_205332


namespace remainder_of_sum_mod_13_l2053_205376

theorem remainder_of_sum_mod_13 (a b c d e : ℕ) 
  (h1: a % 13 = 3) (h2: b % 13 = 5) (h3: c % 13 = 7) (h4: d % 13 = 9) (h5: e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := 
by 
  sorry

end remainder_of_sum_mod_13_l2053_205376


namespace smallest_positive_integer_k_l2053_205371

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end smallest_positive_integer_k_l2053_205371


namespace loop_until_correct_l2053_205357

-- Define the conditions
def num_iterations := 20

-- Define the loop condition
def loop_condition (i : Nat) : Prop := i > num_iterations

-- Theorem: Proof that the loop should continue until the counter i exceeds 20
theorem loop_until_correct (i : Nat) : loop_condition i := by
  sorry

end loop_until_correct_l2053_205357


namespace range_of_a_l2053_205374

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x

theorem range_of_a (a : ℝ) (h : f (2 * a) < f (a - 1)) : a < -1 :=
by
  -- Steps of the proof would be placed here, but we're skipping them for now
  sorry

end range_of_a_l2053_205374


namespace diameter_of_outer_circle_l2053_205369

theorem diameter_of_outer_circle (D d : ℝ) 
  (h1 : d = 24) 
  (h2 : π * (D / 2) ^ 2 - π * (d / 2) ^ 2 = 0.36 * π * (D / 2) ^ 2) : D = 30 := 
by 
  sorry

end diameter_of_outer_circle_l2053_205369


namespace shanghai_team_score_l2053_205342

variables (S B : ℕ)

-- Conditions
def yao_ming_points : ℕ := 30
def point_margin : ℕ := 10
def total_points_minus_10 : ℕ := 5 * yao_ming_points - 10
def combined_total_points : ℕ := total_points_minus_10

-- The system of equations as conditions
axiom condition1 : S - B = point_margin
axiom condition2 : S + B = combined_total_points

-- The proof statement
theorem shanghai_team_score : S = 75 :=
by
  sorry

end shanghai_team_score_l2053_205342


namespace money_conditions_l2053_205322

theorem money_conditions (a b : ℝ) (h1 : 4 * a - b > 32) (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := 
sorry

end money_conditions_l2053_205322


namespace number_of_cars_in_second_box_is_31_l2053_205355

-- Define the total number of toy cars, and the number of toy cars in the first and third boxes
def total_toy_cars : ℕ := 71
def cars_in_first_box : ℕ := 21
def cars_in_third_box : ℕ := 19

-- Define the number of toy cars in the second box
def cars_in_second_box : ℕ := total_toy_cars - cars_in_first_box - cars_in_third_box

-- Theorem stating that the number of toy cars in the second box is 31
theorem number_of_cars_in_second_box_is_31 : cars_in_second_box = 31 :=
by
  sorry

end number_of_cars_in_second_box_is_31_l2053_205355


namespace find_common_ratio_geometric_l2053_205364

variable {α : Type*} [Field α] {a : ℕ → α} {S : ℕ → α} {q : α} (h₁ : a 3 = 2 * S 2 + 1) (h₂ : a 4 = 2 * S 3 + 1)

def common_ratio_geometric : α := 3

theorem find_common_ratio_geometric (ha₃ : a 3 = 2 * S 2 + 1) (ha₄ : a 4 = 2 * S 3 + 1) :
  q = common_ratio_geometric := 
  sorry

end find_common_ratio_geometric_l2053_205364


namespace possible_value_of_a_eq_neg1_l2053_205391

theorem possible_value_of_a_eq_neg1 (a : ℝ) : (-6 * a ^ 2 = 3 * (4 * a + 2)) → (a = -1) :=
by
  intro h
  have H : a^2 + 2*a + 1 = 0
  · sorry
  show a = -1
  · sorry

end possible_value_of_a_eq_neg1_l2053_205391


namespace only_one_passes_prob_l2053_205360

variable (P_A P_B P_C : ℚ)
variable (only_one_passes : ℚ)

def prob_A := 4 / 5 
def prob_B := 3 / 5
def prob_C := 7 / 10

def prob_only_A := prob_A * (1 - prob_B) * (1 - prob_C)
def prob_only_B := (1 - prob_A) * prob_B * (1 - prob_C)
def prob_only_C := (1 - prob_A) * (1 - prob_B) * prob_C

def prob_sum : ℚ := prob_only_A + prob_only_B + prob_only_C

theorem only_one_passes_prob : prob_sum = 47 / 250 := 
by sorry

end only_one_passes_prob_l2053_205360


namespace jar_a_marbles_l2053_205331

theorem jar_a_marbles : ∃ A : ℕ, (∃ B : ℕ, B = A + 12) ∧ (∃ C : ℕ, C = 2 * (A + 12)) ∧ (A + (A + 12) + 2 * (A + 12) = 148) ∧ (A = 28) :=
by
sorry

end jar_a_marbles_l2053_205331


namespace square_line_product_l2053_205341

theorem square_line_product (b : ℝ) 
  (h1 : ∃ y1 y2, y1 = -1 ∧ y2 = 4) 
  (h2 : ∃ x1, x1 = 3) 
  (h3 : (4 - (-1)) = (5 : ℝ)) 
  (h4 : ((∃ b1, b1 = 3 + 5 ∨ b1 = 3 - 5) → b = b1)) :
  b = -2 ∨ b = 8 → b * 8 = -16 :=
by sorry

end square_line_product_l2053_205341


namespace domino_swap_correct_multiplication_l2053_205311

theorem domino_swap_correct_multiplication :
  ∃ (a b c d e f : ℕ), 
    a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 3 ∧ e = 12 ∧ f = 3 ∧ 
    a * b = 6 ∧ c * d = 3 ∧ e * f = 36 ∧
    ∃ (x y : ℕ), x * y = 36 := sorry

end domino_swap_correct_multiplication_l2053_205311
