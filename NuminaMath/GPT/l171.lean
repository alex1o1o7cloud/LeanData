import Mathlib

namespace minimum_value_property_l171_171913

noncomputable def min_value_expression (x : ℝ) (h : x > 10) : ℝ :=
  (x^2 + 36) / (x - 10)

noncomputable def min_value : ℝ := 4 * Real.sqrt 34 + 20

theorem minimum_value_property (x : ℝ) (h : x > 10) :
  min_value_expression x h >= min_value := by
  sorry

end minimum_value_property_l171_171913


namespace range_of_a_l171_171533

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ x^2 + (1 - a) * x + 3 - a > 0) ↔ a < 3 := 
sorry

end range_of_a_l171_171533


namespace simplify_and_evaluate_l171_171257

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l171_171257


namespace min_distance_squared_l171_171710

noncomputable def graph_function1 (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_graph1 (a b : ℝ) : Prop := b = graph_function1 a

noncomputable def graph_function2 (x : ℝ) : ℝ := x + 2

noncomputable def point_on_graph2 (c d : ℝ) : Prop := d = graph_function2 c

theorem min_distance_squared (a b c d : ℝ) 
  (hP : point_on_graph1 a b)
  (hQ : point_on_graph2 c d) :
  (a - c)^2 + (b - d)^2 = 8 := 
sorry

end min_distance_squared_l171_171710


namespace truck_dirt_road_time_l171_171371

noncomputable def time_on_dirt_road (time_paved : ℝ) (speed_increment : ℝ) (total_distance : ℝ) (dirt_speed : ℝ) : ℝ :=
  let paved_speed := dirt_speed + speed_increment
  let distance_paved := paved_speed * time_paved
  let distance_dirt := total_distance - distance_paved
  distance_dirt / dirt_speed

theorem truck_dirt_road_time :
  time_on_dirt_road 2 20 200 32 = 3 :=
by
  sorry

end truck_dirt_road_time_l171_171371


namespace point_p_locus_equation_l171_171412

noncomputable def locus_point_p (x y : ℝ) : Prop :=
  ∀ (k b x1 y1 x2 y2 : ℝ), 
  (x1^2 + y1^2 = 1) ∧ 
  (x2^2 + y2^2 = 1) ∧ 
  (3 * x1 * x + 4 * y1 * y = 12) ∧ 
  (3 * x2 * x + 4 * y2 * y = 12) ∧ 
  (1 + k^2 = b^2) ∧ 
  (y = 3 / b) ∧ 
  (x = -4 * k / (3 * b)) → 
  x^2 / 16 + y^2 / 9 = 1

theorem point_p_locus_equation :
  ∀ (x y : ℝ), locus_point_p x y → (x^2 / 16 + y^2 / 9 = 1) :=
by
  intros x y h
  sorry

end point_p_locus_equation_l171_171412


namespace tomatoes_initially_l171_171996

-- Conditions
def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_left_after_yesterday : ℕ := 104

-- The statement to prove
theorem tomatoes_initially : tomatoes_left_after_yesterday + tomatoes_picked_yesterday + tomatoes_picked_today = 201 :=
  by
  -- Proof steps would go here
  sorry

end tomatoes_initially_l171_171996


namespace find_P_l171_171923

-- We start by defining the cubic polynomial
def cubic_eq (P : ℝ) (x : ℝ) := 5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1

-- Define the condition that all roots are natural numbers
def has_three_natural_roots (P : ℝ) : Prop :=
  ∃ a b c : ℕ, 
    cubic_eq P a = 66 * P ∧ cubic_eq P b = 66 * P ∧ cubic_eq P c = 66 * P

-- Prove the value of P that satisfies the condition
theorem find_P : ∀ P : ℝ, has_three_natural_roots P → P = 76 := 
by
  -- We start the proof here
  sorry

end find_P_l171_171923


namespace find_m_l171_171468

def is_ellipse (x y m : ℝ) : Prop :=
  (x^2 / (m + 1) + y^2 / m = 1)

def has_eccentricity (e : ℝ) (m : ℝ) : Prop :=
  e = Real.sqrt (1 - m / (m + 1))

theorem find_m (m : ℝ) (h_m : m > 0) (h_ellipse : ∀ x y, is_ellipse x y m) (h_eccentricity : has_eccentricity (1 / 2) m) : m = 3 :=
by
  sorry

end find_m_l171_171468


namespace remainder_is_37_l171_171740

theorem remainder_is_37
    (d q v r : ℕ)
    (h1 : d = 15968)
    (h2 : q = 89)
    (h3 : v = 179)
    (h4 : d = q * v + r) :
  r = 37 :=
sorry

end remainder_is_37_l171_171740


namespace total_distance_traveled_l171_171267

theorem total_distance_traveled (d : ℝ) (h1 : d/3 + d/4 + d/5 = 47/60) : 3 * d = 3 :=
by
  sorry

end total_distance_traveled_l171_171267


namespace mike_profit_l171_171346

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l171_171346


namespace parallel_lines_perpendicular_lines_l171_171676

theorem parallel_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 = s2) →
  a = 2 :=
by
  intros
  -- Proof goes here
  sorry

theorem perpendicular_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 * s2 = -1) →
  a = 0 ∨ a = -3 :=
by
  intros
  -- Proof goes here
  sorry

end parallel_lines_perpendicular_lines_l171_171676


namespace two_sin_cos_15_eq_half_l171_171456

open Real

theorem two_sin_cos_15_eq_half : 2 * sin (π / 12) * cos (π / 12) = 1 / 2 :=
by
  sorry

end two_sin_cos_15_eq_half_l171_171456


namespace certain_number_l171_171463

theorem certain_number (N : ℝ) (k : ℝ) 
  (h1 : (1 / 2) ^ 22 * N ^ k = 1 / 18 ^ 22) 
  (h2 : k = 11) 
  : N = 81 := 
by
  sorry

end certain_number_l171_171463


namespace solve_inequality_l171_171795

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem solve_inequality {x : ℝ} (hx : 0 < x) : 
  f (Real.log x / Real.log 2) < f 2 ↔ (0 < x ∧ x < 1) ∨ (4 < x) :=
by
sorry

end solve_inequality_l171_171795


namespace angle_C_side_c_area_of_triangle_l171_171521

open Real

variables (A B C a b c : Real)

noncomputable def acute_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
  (A < π / 2) ∧ (B < π / 2) ∧ (C < π / 2) ∧
  (a^2 - 2 * sqrt 3 * a + 2 = 0) ∧
  (b^2 - 2 * sqrt 3 * b + 2 = 0) ∧
  (2 * sin (A + B) - sqrt 3 = 0)

noncomputable def length_side_c (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2 - 2 * a * b * cos (π / 3))

noncomputable def area_triangle (a b : ℝ) : ℝ := 
  (1 / 2) * a * b * sin (π / 3)

theorem angle_C (h : acute_triangle A B C a b c) : C = π / 3 :=
  sorry

theorem side_c (h : acute_triangle A B C a b c) : c = sqrt 6 :=
  sorry

theorem area_of_triangle (h : acute_triangle A B C a b c) : area_triangle a b = sqrt 3 / 2 :=
  sorry

end angle_C_side_c_area_of_triangle_l171_171521


namespace derivative_y_l171_171540

noncomputable def y (x : ℝ) : ℝ := 
  Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

variable {x : ℝ}

theorem derivative_y :
  2 * x + 3 > 0 → 
  HasDerivAt y (4 * Real.sqrt (x^2 + 3 * x + 2) / (2 * x + 3)) x :=
by 
  sorry

end derivative_y_l171_171540


namespace solve_combinations_l171_171621

-- This function calculates combinations
noncomputable def C (n k : ℕ) : ℕ := if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem solve_combinations (x : ℤ) :
  C 16 (x^2 - x).natAbs = C 16 (5*x - 5).natAbs → x = 1 ∨ x = 3 :=
by
  sorry

end solve_combinations_l171_171621


namespace sin_add_pi_over_4_eq_l171_171762

variable (α : Real)
variables (hα1 : 0 < α ∧ α < Real.pi) (hα2 : Real.tan (α - Real.pi / 4) = 1 / 3)

theorem sin_add_pi_over_4_eq : Real.sin (Real.pi / 4 + α) = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end sin_add_pi_over_4_eq_l171_171762


namespace range_of_m_l171_171826

theorem range_of_m (m : ℝ) : (∃ x1 x2 x3 : ℝ, 
    (x1 - 1) * (x1^2 - 2*x1 + m) = 0 ∧ 
    (x2 - 1) * (x2^2 - 2*x2 + m) = 0 ∧ 
    (x3 - 1) * (x3^2 - 2*x3 + m) = 0 ∧ 
    x1 = 1 ∧ 
    x2^2 - 2*x2 + m = 0 ∧ 
    x3^2 - 2*x3 + m = 0 ∧ 
    x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1 ∧ 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0) ↔ 3 / 4 < m ∧ m ≤ 1 := 
by
  sorry

end range_of_m_l171_171826


namespace police_coverage_l171_171734

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define each street as a set of intersections
def horizontal_streets : List (List Intersection) :=
  [[A, B, C, D], [E, F, G], [H, I, J, K]]

def vertical_streets : List (List Intersection) :=
  [[A, E, H], [B, F, I], [D, G, J]]

def diagonal_streets : List (List Intersection) :=
  [[H, F, C], [C, G, K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the set of police officers' placements
def police_officers : List Intersection := [B, G, H]

-- Check if each street is covered by at least one police officer
def is_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i, i ∈ street ∧ i ∈ officers

-- Define the proof problem statement
theorem police_coverage :
  ∀ street ∈ all_streets, is_covered street police_officers :=
by sorry

end police_coverage_l171_171734


namespace line_through_points_l171_171801

theorem line_through_points (a b : ℝ)
  (h1 : 2 = a * 1 + b)
  (h2 : 14 = a * 5 + b) :
  a - b = 4 := 
  sorry

end line_through_points_l171_171801


namespace total_price_correct_l171_171743

-- Definitions based on given conditions
def basic_computer_price : ℝ := 2125
def enhanced_computer_price : ℝ := 2125 + 500
def printer_price (P : ℝ) := P = 1/8 * (enhanced_computer_price + P)

-- Statement to prove the total price of the basic computer and printer
theorem total_price_correct (P : ℝ) (h : printer_price P) : 
  basic_computer_price + P = 2500 :=
by
  sorry

end total_price_correct_l171_171743


namespace happy_dictionary_problem_l171_171234

def smallest_positive_integer : ℕ := 1
def largest_negative_integer : ℤ := -1
def smallest_abs_rational : ℚ := 0

theorem happy_dictionary_problem : 
  smallest_positive_integer - largest_negative_integer + smallest_abs_rational = 2 := 
by
  sorry

end happy_dictionary_problem_l171_171234


namespace molecular_weight_of_3_moles_l171_171401

def molecular_weight_one_mole : ℝ := 176.14
def number_of_moles : ℝ := 3
def total_weight := number_of_moles * molecular_weight_one_mole

theorem molecular_weight_of_3_moles :
  total_weight = 528.42 := sorry

end molecular_weight_of_3_moles_l171_171401


namespace find_grade_2_l171_171970

-- Definitions for the problem
def grade_1 := 78
def weight_1 := 20
def weight_2 := 30
def grade_3 := 90
def weight_3 := 10
def grade_4 := 85
def weight_4 := 40
def overall_average := 83

noncomputable def calc_weighted_average (G : ℕ) : ℝ :=
  (grade_1 * weight_1 + G * weight_2 + grade_3 * weight_3 + grade_4 * weight_4) / (weight_1 + weight_2 + weight_3 + weight_4)

theorem find_grade_2 (G : ℕ) : calc_weighted_average G = overall_average → G = 81 := sorry

end find_grade_2_l171_171970


namespace mark_total_votes_l171_171419

-- Definitions for the problem conditions
def first_area_registered_voters : ℕ := 100000
def first_area_undecided_percentage : ℕ := 5
def first_area_mark_votes_percentage : ℕ := 70

def remaining_area_increase_percentage : ℕ := 20
def remaining_area_undecided_percentage : ℕ := 7
def multiplier_for_remaining_area_votes : ℕ := 2

-- The Lean statement
theorem mark_total_votes : 
  let first_area_undecided_voters := first_area_registered_voters * first_area_undecided_percentage / 100
  let first_area_votes_cast := first_area_registered_voters - first_area_undecided_voters
  let first_area_mark_votes := first_area_votes_cast * first_area_mark_votes_percentage / 100

  let remaining_area_registered_voters := first_area_registered_voters * (1 + remaining_area_increase_percentage / 100)
  let remaining_area_undecided_voters := remaining_area_registered_voters * remaining_area_undecided_percentage / 100
  let remaining_area_votes_cast := remaining_area_registered_voters - remaining_area_undecided_voters
  let remaining_area_mark_votes := first_area_mark_votes * multiplier_for_remaining_area_votes

  let total_mark_votes := first_area_mark_votes + remaining_area_mark_votes
  total_mark_votes = 199500 := 
by
  -- We skipped the proof (it's not required as per instructions)
  sorry

end mark_total_votes_l171_171419


namespace largest_value_l171_171260

theorem largest_value :
  max (max (max (max (4^2) (4 * 2)) (4 - 2)) (4 / 2)) (4 + 2) = 4^2 :=
by sorry

end largest_value_l171_171260


namespace solve_eq1_solve_eq2_l171_171886

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end solve_eq1_solve_eq2_l171_171886


namespace range_of_a_l171_171179

theorem range_of_a (a : ℝ) :
  (abs (15 - 3 * a) / 5 ≤ 3) → (0 ≤ a ∧ a ≤ 10) :=
by
  intro h
  sorry

end range_of_a_l171_171179


namespace alexei_loss_per_week_l171_171792

-- Definitions
def aleesia_loss_per_week : ℝ := 1.5
def aleesia_total_weeks : ℕ := 10
def total_loss : ℝ := 35
def alexei_total_weeks : ℕ := 8

-- The statement to prove
theorem alexei_loss_per_week :
  (total_loss - aleesia_loss_per_week * aleesia_total_weeks) / alexei_total_weeks = 2.5 := 
by sorry

end alexei_loss_per_week_l171_171792


namespace seventh_graders_count_l171_171054

-- Define the problem conditions
def total_students (T : ℝ) : Prop := 0.38 * T = 76
def seventh_grade_ratio : ℝ := 0.32
def seventh_graders (S : ℝ) (T : ℝ) : Prop := S = seventh_grade_ratio * T

-- The goal statement
theorem seventh_graders_count {T S : ℝ} (h : total_students T) : seventh_graders S T → S = 64 :=
by
  sorry

end seventh_graders_count_l171_171054


namespace roots_quadratic_l171_171741

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end roots_quadratic_l171_171741


namespace imaginary_part_of_fraction_l171_171338

open Complex

theorem imaginary_part_of_fraction :
  ∃ z : ℂ, z = ⟨0, 1⟩ / ⟨1, 1⟩ ∧ z.im = 1 / 2 :=
by
  sorry

end imaginary_part_of_fraction_l171_171338


namespace series_sum_l171_171034

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : b < a)

noncomputable def infinite_series : ℝ := 
∑' n, 1 / ( ((n - 1) * a^2 - (n - 2) * b^2) * (n * a^2 - (n - 1) * b^2) )

theorem series_sum : infinite_series a b = 1 / ((a^2 - b^2) * b^2) := 
by 
  sorry

end series_sum_l171_171034


namespace union_of_A_and_B_l171_171759

variables (A B : Set ℤ)
variable (a : ℤ)
theorem union_of_A_and_B : (A = {4, a^2}) → (B = {a-6, 1+a, 9}) → (A ∩ B = {9}) → (A ∪ B = {-9, -2, 4, 9}) :=
by
  intros hA hB hInt
  sorry

end union_of_A_and_B_l171_171759


namespace phone_numbers_count_l171_171576

theorem phone_numbers_count : (2^5 = 32) :=
by sorry

end phone_numbers_count_l171_171576


namespace maximum_daily_sales_revenue_l171_171635

noncomputable def P (t : ℕ) : ℤ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

noncomputable def Q (t : ℕ) : ℤ :=
  if 0 < t ∧ t ≤ 30 then -t + 40 else 0

noncomputable def y (t : ℕ) : ℤ := P t * Q t

theorem maximum_daily_sales_revenue : 
  ∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ y t = 1125 :=
by
  sorry

end maximum_daily_sales_revenue_l171_171635


namespace sum_of_first_45_natural_numbers_l171_171816

theorem sum_of_first_45_natural_numbers : (45 * (45 + 1)) / 2 = 1035 := by
  sorry

end sum_of_first_45_natural_numbers_l171_171816


namespace optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l171_171122
-- Import necessary libraries

-- Define each of the conditions as Lean definitions
def OptionA (a b c : ℝ) : Prop := a = 1.5 ∧ b = 2 ∧ c = 3
def OptionB (a b c : ℝ) : Prop := a = 7 ∧ b = 24 ∧ c = 25
def OptionC (a b c : ℝ) : Prop := ∃ k : ℕ, a = (3 : ℝ)*k ∧ b = (4 : ℝ)*k ∧ c = (5 : ℝ)*k
def OptionD (a b c : ℝ) : Prop := a = 9 ∧ b = 12 ∧ c = 15

-- Define the Pythagorean theorem predicate
def Pythagorean (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- State the theorem to prove Option A cannot form a right triangle
theorem optionA_not_right_triangle : ¬ Pythagorean 1.5 2 3 := by sorry

-- State the remaining options can form a right triangle
theorem optionB_right_triangle : Pythagorean 7 24 25 := by sorry
theorem optionC_right_triangle (k : ℕ) : Pythagorean (3 * k) (4 * k) (5 * k) := by sorry
theorem optionD_right_triangle : Pythagorean 9 12 15 := by sorry

end optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l171_171122


namespace tax_diminished_percentage_l171_171815

theorem tax_diminished_percentage (T C : ℝ) (x : ℝ) (h : (T * (1 - x / 100)) * (C * 1.10) = T * C * 0.88) :
  x = 20 :=
sorry

end tax_diminished_percentage_l171_171815


namespace three_digit_difference_l171_171964

theorem three_digit_difference (x : ℕ) (a b c : ℕ)
  (h1 : a = x + 2)
  (h2 : b = x + 1)
  (h3 : c = x)
  (h4 : a > b)
  (h5 : b > c) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 :=
by
  sorry

end three_digit_difference_l171_171964


namespace second_pipe_fill_time_l171_171409

theorem second_pipe_fill_time (x : ℝ) :
  (1 / 18) + (1 / x) - (1 / 45) = (1 / 15) → x = 30 :=
by
  intro h
  sorry

end second_pipe_fill_time_l171_171409


namespace fourth_term_in_arithmetic_sequence_l171_171879

theorem fourth_term_in_arithmetic_sequence (a d : ℝ) (h : 2 * a + 6 * d = 20) : a + 3 * d = 10 :=
sorry

end fourth_term_in_arithmetic_sequence_l171_171879


namespace negation_all_dogs_playful_l171_171094

variable {α : Type} (dog playful : α → Prop)

theorem negation_all_dogs_playful :
  (¬ ∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬ playful x) :=
by sorry

end negation_all_dogs_playful_l171_171094


namespace H2O_formed_l171_171884

-- Definition of the balanced chemical equation
def balanced_eqn : Prop :=
  ∀ (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ), HCH3CO2 + NaOH = NaCH3CO2 + H2O

-- Statement of the problem
theorem H2O_formed (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ) 
  (h1 : HCH3CO2 = 1)
  (h2 : NaOH = 1)
  (balanced : balanced_eqn):
  H2O = 1 :=
by sorry

end H2O_formed_l171_171884


namespace eval_g_231_l171_171040

def g (a b c : ℤ) : ℚ :=
  (c ^ 2 + a ^ 2) / (c - b)

theorem eval_g_231 : g 2 (-3) 1 = 5 / 4 :=
by
  sorry

end eval_g_231_l171_171040


namespace sum_of_first_49_primes_l171_171520

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end sum_of_first_49_primes_l171_171520


namespace total_profit_percentage_l171_171411

theorem total_profit_percentage (total_apples : ℕ) (percent_sold_10 : ℝ) (percent_sold_30 : ℝ) (profit_10 : ℝ) (profit_30 : ℝ) : 
  total_apples = 280 → 
  percent_sold_10 = 0.40 → 
  percent_sold_30 = 0.60 → 
  profit_10 = 0.10 → 
  profit_30 = 0.30 → 
  ((percent_sold_10 * total_apples * (1 + profit_10) + percent_sold_30 * total_apples * (1 + profit_30) - total_apples) / total_apples * 100) = 22 := 
by 
  intros; sorry

end total_profit_percentage_l171_171411


namespace total_fish_count_l171_171727

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l171_171727


namespace b_minus_a_l171_171571

theorem b_minus_a (a b : ℕ) : (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) :=
by
  sorry

end b_minus_a_l171_171571


namespace leftover_value_is_correct_l171_171402

def value_of_leftover_coins (total_quarters total_dimes quarters_per_roll dimes_per_roll : ℕ) : ℝ :=
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters * 0.25) + (leftover_dimes * 0.10)

def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40

def total_quarters : ℕ := michael_quarters + anna_quarters
def total_dimes : ℕ := michael_dimes + anna_dimes

theorem leftover_value_is_correct : 
  value_of_leftover_coins total_quarters total_dimes quarters_per_roll dimes_per_roll = 10.65 :=
by
  sorry

end leftover_value_is_correct_l171_171402


namespace rate_times_base_eq_9000_l171_171892

noncomputable def Rate : ℝ := 0.00015
noncomputable def BaseAmount : ℝ := 60000000

theorem rate_times_base_eq_9000 :
  Rate * BaseAmount = 9000 := 
  sorry

end rate_times_base_eq_9000_l171_171892


namespace sum_of_first_ten_nicely_odd_numbers_is_775_l171_171130

def is_nicely_odd (n : ℕ) : Prop :=
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ (Odd p ∧ Odd q) ∧ n = p * q)
  ∨ (∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ n = p ^ 3)

theorem sum_of_first_ten_nicely_odd_numbers_is_775 :
  let nicely_odd_nums := [15, 27, 21, 35, 125, 33, 77, 343, 55, 39]
  ∃ (nums : List ℕ), List.length nums = 10 ∧
  (∀ n ∈ nums, is_nicely_odd n) ∧ List.sum nums = 775 := by
  sorry

end sum_of_first_ten_nicely_odd_numbers_is_775_l171_171130


namespace sum_of_roots_even_l171_171893

theorem sum_of_roots_even (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
    (h_distinct : ∃ x y : ℤ, x ≠ y ∧ (x^2 - 2 * p * x + (p * q) = 0) ∧ (y^2 - 2 * p * y + (p * q) = 0)) :
    Even (2 * p) :=
by 
  sorry

end sum_of_roots_even_l171_171893


namespace weeks_to_buy_iphone_l171_171582

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end weeks_to_buy_iphone_l171_171582


namespace emily_total_cost_l171_171507

-- Definition of the monthly cell phone plan costs and usage details
def base_cost : ℝ := 30
def cost_per_text : ℝ := 0.10
def cost_per_extra_minute : ℝ := 0.15
def cost_per_extra_gb : ℝ := 5
def free_hours : ℝ := 25
def free_gb : ℝ := 15
def texts : ℝ := 150
def hours : ℝ := 26
def gb : ℝ := 16

-- Calculate the total cost
def total_cost : ℝ :=
  base_cost +
  (texts * cost_per_text) +
  ((hours - free_hours) * 60 * cost_per_extra_minute) +
  ((gb - free_gb) * cost_per_extra_gb)

-- The proof statement that Emily had to pay $59
theorem emily_total_cost :
  total_cost = 59 := by
  sorry

end emily_total_cost_l171_171507


namespace time_after_6666_seconds_l171_171922

noncomputable def initial_time : Nat := 3 * 3600
noncomputable def additional_seconds : Nat := 6666

-- Function to convert total seconds to "HH:MM:SS" format
def time_in_seconds (h m s : Nat) : Nat :=
  h*3600 + m*60 + s

noncomputable def new_time : Nat :=
  initial_time + additional_seconds

-- Convert the new total time back to "HH:MM:SS" format (expected: 4:51:06)
def hours (secs : Nat) : Nat := secs / 3600
def minutes (secs : Nat) : Nat := (secs % 3600) / 60
def seconds (secs : Nat) : Nat := (secs % 3600) % 60

theorem time_after_6666_seconds :
  hours new_time = 4 ∧ minutes new_time = 51 ∧ seconds new_time = 6 :=
by
  sorry

end time_after_6666_seconds_l171_171922


namespace range_of_y_l171_171746

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 2 + 2) * (2 * (Real.log x / (2 * Real.log 2)) - 4)

theorem range_of_y :
  (1 ≤ x ∧ x ≤ 8) →
  (∀ t : ℝ, t = Real.log x / Real.log 2 → y x = t^2 - 2 * t - 8 ∧ 0 ≤ t ∧ t ≤ 3) →
  ∃ ymin ymax, (ymin ≤ y x ∧ y x ≤ ymax) ∧ ymin = -9 ∧ ymax = -5 :=
by
  sorry

end range_of_y_l171_171746


namespace cost_price_of_book_l171_171293

-- Define the variables and conditions
variable (C : ℝ)
variable (P : ℝ)
variable (S : ℝ)

-- State the conditions given in the problem
def conditions := S = 260 ∧ P = 0.20 * C ∧ S = C + P

-- State the theorem
theorem cost_price_of_book (h : conditions C P S) : C = 216.67 :=
sorry

end cost_price_of_book_l171_171293


namespace copper_production_is_correct_l171_171800

-- Define the percentages of copper production for each mine
def percentage_copper_mine_a : ℝ := 0.05
def percentage_copper_mine_b : ℝ := 0.10
def percentage_copper_mine_c : ℝ := 0.15

-- Define the daily production of each mine in tons
def daily_production_mine_a : ℕ := 3000
def daily_production_mine_b : ℕ := 4000
def daily_production_mine_c : ℕ := 3500

-- Define the total copper produced from all mines
def total_copper_produced : ℝ :=
  percentage_copper_mine_a * daily_production_mine_a +
  percentage_copper_mine_b * daily_production_mine_b +
  percentage_copper_mine_c * daily_production_mine_c

-- Prove that the total daily copper production is 1075 tons
theorem copper_production_is_correct :
  total_copper_produced = 1075 := 
sorry

end copper_production_is_correct_l171_171800


namespace green_fish_always_15_l171_171531

def total_fish (T : ℕ) : Prop :=
∃ (O B G : ℕ),
B = T / 2 ∧
O = B - 15 ∧
T = B + O + G ∧
G = 15

theorem green_fish_always_15 (T : ℕ) : total_fish T → ∃ G, G = 15 :=
by
  intro h
  sorry

end green_fish_always_15_l171_171531


namespace partial_fraction_series_sum_l171_171731

theorem partial_fraction_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end partial_fraction_series_sum_l171_171731


namespace jill_has_1_more_peach_than_jake_l171_171013

theorem jill_has_1_more_peach_than_jake
    (jill_peaches : ℕ)
    (steven_peaches : ℕ)
    (jake_peaches : ℕ)
    (h1 : jake_peaches = steven_peaches - 16)
    (h2 : steven_peaches = jill_peaches + 15)
    (h3 : jill_peaches = 12) :
    12 - (steven_peaches - 16) = 1 := 
sorry

end jill_has_1_more_peach_than_jake_l171_171013


namespace rectangle_new_area_l171_171489

theorem rectangle_new_area (l w : ℝ) (h_area : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 497 :=
by
  sorry

end rectangle_new_area_l171_171489


namespace time_spent_on_type_a_problems_l171_171477

-- Define the conditions
def total_questions := 200
def examination_duration_hours := 3
def type_a_problems := 100
def type_b_problems := total_questions - type_a_problems
def type_a_time_coeff := 2

-- Convert examination duration to minutes
def examination_duration_minutes := examination_duration_hours * 60

-- Variables for time per problem
variable (x : ℝ)

-- The total time spent
def total_time_spent : ℝ := type_a_problems * (type_a_time_coeff * x) + type_b_problems * x

-- Statement we need to prove
theorem time_spent_on_type_a_problems :
  total_time_spent x = examination_duration_minutes → type_a_problems * (type_a_time_coeff * x) = 120 :=
by
  sorry

end time_spent_on_type_a_problems_l171_171477


namespace simplify_expression_l171_171389

theorem simplify_expression (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 :=
by
  sorry

end simplify_expression_l171_171389


namespace new_train_distance_l171_171113

theorem new_train_distance (old_train_distance : ℕ) (additional_factor : ℕ) (h₀ : old_train_distance = 300) (h₁ : additional_factor = 50) :
  let new_train_distance := old_train_distance + (additional_factor * old_train_distance / 100)
  new_train_distance = 450 :=
by
  sorry

end new_train_distance_l171_171113


namespace at_least_one_gt_one_l171_171131

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : (x > 1) ∨ (y > 1) :=
sorry

end at_least_one_gt_one_l171_171131


namespace evaluate_fraction_l171_171238

theorem evaluate_fraction : (1 - 1/4) / (1 - 1/3) = 9/8 :=
by
  sorry

end evaluate_fraction_l171_171238


namespace a_lt_sqrt3b_l171_171629

open Int

theorem a_lt_sqrt3b (a b : ℤ) (h1 : a > b) (h2 : b > 1) 
    (h3 : a + b ∣ a * b + 1) (h4 : a - b ∣ a * b - 1) : a < sqrt 3 * b :=
  sorry

end a_lt_sqrt3b_l171_171629


namespace valid_combinations_l171_171559

-- Definitions based on conditions
def h : Nat := 4  -- number of herbs
def c : Nat := 6  -- number of crystals
def r : Nat := 3  -- number of negative reactions

-- Theorem statement based on the problem and solution
theorem valid_combinations : (h * c) - r = 21 := by
  sorry

end valid_combinations_l171_171559


namespace unripe_oranges_per_day_l171_171784

/-
Problem: Prove that if after 6 days, they will have 390 sacks of unripe oranges, then the number of sacks of unripe oranges harvested per day is 65.
-/

theorem unripe_oranges_per_day (total_sacks : ℕ) (days : ℕ) (harvest_per_day : ℕ)
  (h1 : days = 6)
  (h2 : total_sacks = 390)
  (h3 : harvest_per_day = total_sacks / days) :
  harvest_per_day = 65 :=
by
  sorry

end unripe_oranges_per_day_l171_171784


namespace joyce_apples_l171_171406

/-- Joyce starts with some apples. She gives 52 apples to Larry and ends up with 23 apples. 
    Prove that Joyce initially had 75 apples. -/
theorem joyce_apples (initial_apples given_apples final_apples : ℕ) 
  (h1 : given_apples = 52) 
  (h2 : final_apples = 23) 
  (h3 : initial_apples = given_apples + final_apples) : 
  initial_apples = 75 := 
by 
  sorry

end joyce_apples_l171_171406


namespace solution_1_solution_2_l171_171481

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |x - 3|

theorem solution_1 (x : ℝ) : (f x (-1) >= 2) ↔ (x >= 2) :=
by
  sorry

theorem solution_2 (a : ℝ) : 
  (∃ x : ℝ, f x a <= -(a / 2)) ↔ (a <= 2 ∨ a >= 6) :=
by
  sorry

end solution_1_solution_2_l171_171481


namespace c_finishes_work_in_18_days_l171_171141

theorem c_finishes_work_in_18_days (A B C : ℝ) 
  (h1 : A = 1 / 12) 
  (h2 : B = 1 / 9) 
  (h3 : A + B + C = 1 / 4) : 
  1 / C = 18 := 
    sorry

end c_finishes_work_in_18_days_l171_171141


namespace tori_passing_question_l171_171828

def arithmetic_questions : ℕ := 20
def algebra_questions : ℕ := 40
def geometry_questions : ℕ := 40
def total_questions : ℕ := arithmetic_questions + algebra_questions + geometry_questions
def arithmetic_correct_pct : ℕ := 80
def algebra_correct_pct : ℕ := 50
def geometry_correct_pct : ℕ := 70
def passing_grade_pct : ℕ := 65

theorem tori_passing_question (questions_needed_to_pass : ℕ) (arithmetic_correct : ℕ) (algebra_correct : ℕ) (geometry_correct : ℕ) : 
  questions_needed_to_pass = 1 :=
by
  let arithmetic_correct : ℕ := (arithmetic_correct_pct * arithmetic_questions / 100)
  let algebra_correct : ℕ := (algebra_correct_pct * algebra_questions / 100)
  let geometry_correct : ℕ := (geometry_correct_pct * geometry_questions / 100)
  let total_correct : ℕ := arithmetic_correct + algebra_correct + geometry_correct
  let passing_grade : ℕ := (passing_grade_pct * total_questions / 100)
  let questions_needed_to_pass : ℕ := passing_grade - total_correct
  exact sorry

end tori_passing_question_l171_171828


namespace anna_should_plant_8_lettuce_plants_l171_171264

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end anna_should_plant_8_lettuce_plants_l171_171264


namespace smallest_four_digit_divisible_by_53_l171_171788

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l171_171788


namespace part1_part2_l171_171143

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) := {x : ℝ | -3 * m + 4 ≤ x ∧ x ≤ 2 * m - 1}

def p (m : ℝ) := ∀ x : ℝ, x ∈ A → x ∈ B m
def q (m : ℝ) := ∃ x : ℝ, x ∈ B m ∧ x ∈ A

theorem part1 (m : ℝ) : p m → m ≥ 4 := by
  sorry

theorem part2 (m : ℝ) : q m → m ≥ 3/2 := by
  sorry

end part1_part2_l171_171143


namespace limes_left_l171_171347

-- Define constants
def num_limes_initial : ℕ := 9
def num_limes_given : ℕ := 4

-- Theorem to be proved
theorem limes_left : num_limes_initial - num_limes_given = 5 :=
by
  sorry

end limes_left_l171_171347


namespace high_school_total_students_l171_171786

theorem high_school_total_students (N_seniors N_sample N_freshmen_sample N_sophomores_sample N_total : ℕ)
  (h_seniors : N_seniors = 1000)
  (h_sample : N_sample = 185)
  (h_freshmen_sample : N_freshmen_sample = 75)
  (h_sophomores_sample : N_sophomores_sample = 60)
  (h_proportion : N_seniors * (N_sample - (N_freshmen_sample + N_sophomores_sample)) = N_total * (N_sample - N_freshmen_sample - N_sophomores_sample)) :
  N_total = 3700 :=
by
  sorry

end high_school_total_students_l171_171786


namespace jack_grassy_time_is_6_l171_171232

def jack_sandy_time := 19
def jill_total_time := 32
def jill_time_delay := 7
def jack_total_time : ℕ := jill_total_time - jill_time_delay
def jack_grassy_time : ℕ := jack_total_time - jack_sandy_time

theorem jack_grassy_time_is_6 : jack_grassy_time = 6 := by 
  have h1: jack_total_time = 25 := by sorry
  have h2: jack_grassy_time = 6 := by sorry
  exact h2

end jack_grassy_time_is_6_l171_171232


namespace boards_nailing_l171_171005

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end boards_nailing_l171_171005


namespace assistant_stop_time_l171_171811

-- Define the start time for the craftsman
def craftsmanStartTime : Nat := 8 * 60 -- in minutes

-- Craftsman starts at 8:00 AM and stops at 12:00 PM
def craftsmanEndTime : Nat := 12 * 60 -- in minutes

-- Craftsman produces 6 bracelets every 20 minutes
def craftsmanProductionPerMinute : Nat := 6 / 20

-- Assistant starts working at 9:00 AM
def assistantStartTime : Nat := 9 * 60 -- in minutes

-- Assistant produces 8 bracelets every 30 minutes
def assistantProductionPerMinute : Nat := 8 / 30

-- Total production duration for craftsman in minutes
def craftsmanWorkDuration : Nat := craftsmanEndTime - craftsmanStartTime

-- Total bracelets produced by craftsman
def totalBraceletsCraftsman : Nat := craftsmanWorkDuration * craftsmanProductionPerMinute

-- Time it takes for the assistant to produce the same number of bracelets
def assistantWorkDuration : Nat := totalBraceletsCraftsman / assistantProductionPerMinute

-- Time the assistant will stop working
def assistantEndTime : Nat := assistantStartTime + assistantWorkDuration

-- Convert time in minutes to hours and minutes format (output as a string for clarity)
def formatTime (timeInMinutes: Nat) : String :=
  let hours := timeInMinutes / 60
  let minutes := timeInMinutes % 60
  s! "{hours}:{if minutes < 10 then "0" else ""}{minutes}"

-- Proof goal: assistant will stop working at "13:30" (or 1:30 PM)
theorem assistant_stop_time : 
  formatTime assistantEndTime = "13:30" := 
by
  sorry

end assistant_stop_time_l171_171811


namespace n_gon_angles_l171_171503

theorem n_gon_angles (n : ℕ) (h1 : n > 7) (h2 : n < 12) : 
  (∃ x : ℝ, (150 * (n - 1) + x = 180 * (n - 2)) ∧ (x < 150)) :=
by {
  sorry
}

end n_gon_angles_l171_171503


namespace solve_pow_problem_l171_171329

theorem solve_pow_problem : (-2)^1999 + (-2)^2000 = 2^1999 := 
sorry

end solve_pow_problem_l171_171329


namespace A_elements_l171_171672

open Set -- Open the Set namespace for easy access to set operations

def A : Set ℕ := {x | ∃ (n : ℕ), 12 = n * (6 - x)}

theorem A_elements : A = {0, 2, 3, 4, 5} :=
by
  -- proof steps here
  sorry

end A_elements_l171_171672


namespace quadratic_inequality_solution_minimum_value_expression_l171_171158

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end quadratic_inequality_solution_minimum_value_expression_l171_171158


namespace solve_y_l171_171265

theorem solve_y 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (remainder_condition : x = (96.12 * y))
  (division_condition : x = (96.0624 * y + 5.76)) : 
  y = 100 := 
 sorry

end solve_y_l171_171265


namespace triangle_inequality_x_not_2_l171_171864

theorem triangle_inequality_x_not_2 (x : ℝ) (h1 : 2 < x) (h2 : x < 8) : x ≠ 2 :=
by 
  sorry

end triangle_inequality_x_not_2_l171_171864


namespace initial_speed_of_car_l171_171575

-- Definition of conditions
def distance_from_A_to_B := 100  -- km
def time_remaining_first_reduction := 30 / 60  -- hours
def speed_reduction_first := 10  -- km/h
def time_remaining_second_reduction := 20 / 60  -- hours
def speed_reduction_second := 10  -- km/h
def additional_time_reduced_speeds := 5 / 60  -- hours

-- Variables for initial speed and intermediate distances
variables (v x : ℝ)

-- Proposition to prove the initial speed
theorem initial_speed_of_car :
  (100 - (v / 2 + x + 20)) / v + 
  (v / 2) / (v - 10) + 
  20 / (v - 20) - 
  20 / (v - 10) 
  = 5 / 60 →
  v = 100 :=
by
  sorry

end initial_speed_of_car_l171_171575


namespace calculate_three_Z_five_l171_171443

def Z (a b : ℤ) : ℤ := b + 15 * a - a^3

theorem calculate_three_Z_five : Z 3 5 = 23 :=
by
  -- The proof goes here
  sorry

end calculate_three_Z_five_l171_171443


namespace population_of_metropolitan_county_l171_171314

theorem population_of_metropolitan_county : 
  let average_population := 5500
  let two_populous_cities_population := 2 * average_population
  let remaining_cities := 25 - 2
  let remaining_population := remaining_cities * average_population
  let total_population := (2 * two_populous_cities_population) + remaining_population
  total_population = 148500 := by
sorry

end population_of_metropolitan_county_l171_171314


namespace triangle_inequality_l171_171822

variable (a b c p : ℝ)
variable (triangle : a + b > c ∧ a + c > b ∧ b + c > a)
variable (h_p : p = (a + b + c) / 2)

theorem triangle_inequality : 2 * Real.sqrt ((p - b) * (p - c)) ≤ a :=
sorry

end triangle_inequality_l171_171822


namespace train_speed_in_kmh_l171_171729

/-- Definition of length of the train in meters. -/
def train_length : ℕ := 200

/-- Definition of time taken to cross the electric pole in seconds. -/
def time_to_cross : ℕ := 20

/-- The speed of the train in km/h is 36 given the length of the train and time to cross. -/
theorem train_speed_in_kmh (length : ℕ) (time : ℕ) (h_len : length = train_length) (h_time: time = time_to_cross) : 
  (length / time : ℚ) * 3.6 = 36 := 
by
  sorry

end train_speed_in_kmh_l171_171729


namespace arc_length_120_degrees_l171_171944

theorem arc_length_120_degrees (π : ℝ) : 
  let R := π
  let n := 120
  (n * π * R) / 180 = (2 * π^2) / 3 := 
by
  let R := π
  let n := 120
  sorry

end arc_length_120_degrees_l171_171944


namespace only_zero_function_satisfies_conditions_l171_171434

def is_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n > m → f n ≥ f m

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, f (n * m) = f n + f m

theorem only_zero_function_satisfies_conditions :
  ∀ f : ℕ → ℕ, 
  (is_increasing f) ∧ (satisfies_functional_equation f) → (∀ n : ℕ, f n = 0) :=
by
  sorry

end only_zero_function_satisfies_conditions_l171_171434


namespace sequence_an_general_formula_and_sum_bound_l171_171462

theorem sequence_an_general_formula_and_sum_bound (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (1 / 4) * (a n + 1) ^ 2)
  (h2 : ∀ n, b n = 1 / (a n * a (n + 1)))
  (h3 : ∀ n, T n = (1 / 2) * (1 - (1 / (2 * n + 1))))
  (h4 : ∀ n, 0 < a n) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n < 1 / 2) := 
by
  sorry

end sequence_an_general_formula_and_sum_bound_l171_171462


namespace count_two_digit_primes_with_units_digit_3_l171_171348

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l171_171348


namespace sum_of_reflection_midpoint_coordinates_l171_171315

theorem sum_of_reflection_midpoint_coordinates (P R : ℝ × ℝ) (M : ℝ × ℝ) (P' R' M' : ℝ × ℝ) :
  P = (2, 1) → R = (12, 15) → 
  M = ((P.fst + R.fst) / 2, (P.snd + R.snd) / 2) →
  P' = (-P.fst, P.snd) → R' = (-R.fst, R.snd) →
  M' = ((P'.fst + R'.fst) / 2, (P'.snd + R'.snd) / 2) →
  (M'.fst + M'.snd) = 1 := 
by 
  intros
  sorry

end sum_of_reflection_midpoint_coordinates_l171_171315


namespace vectors_parallel_solution_l171_171030

theorem vectors_parallel_solution (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (2, x)) (h2 : b = (x, 8)) (h3 : ∃ k, b = (k * 2, k * x)) : x = 4 ∨ x = -4 :=
by
  sorry

end vectors_parallel_solution_l171_171030


namespace max_imag_part_of_roots_l171_171342

noncomputable def polynomial (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

theorem max_imag_part_of_roots :
  ∃ (z : ℂ), polynomial z = 0 ∧ ∀ w, polynomial w = 0 → (z.im ≤ w.im) := sorry

end max_imag_part_of_roots_l171_171342


namespace evaluate_expression_l171_171968

theorem evaluate_expression :
  200 * (200 - 3) + (200 ^ 2 - 8 ^ 2) = 79336 :=
by
  sorry

end evaluate_expression_l171_171968


namespace bottles_of_regular_soda_l171_171659

theorem bottles_of_regular_soda
  (diet_soda : ℕ)
  (apples : ℕ)
  (more_bottles_than_apples : ℕ)
  (R : ℕ)
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : more_bottles_than_apples = 26)
  (h4 : R + diet_soda = apples + more_bottles_than_apples) :
  R = 72 := 
by sorry

end bottles_of_regular_soda_l171_171659


namespace total_journey_time_eq_5_l171_171707

-- Define constants for speed and times
def speed1 : ℕ := 40
def speed2 : ℕ := 60
def total_distance : ℕ := 240
def time1 : ℕ := 3

-- Noncomputable definition to avoid computation issues
noncomputable def journey_time : ℕ :=
  let distance1 := speed1 * time1
  let distance2 := total_distance - distance1
  let time2 := distance2 / speed2
  time1 + time2

-- Theorem to state the total journey time
theorem total_journey_time_eq_5 : journey_time = 5 := by
  sorry

end total_journey_time_eq_5_l171_171707


namespace solution_exists_iff_divisor_form_l171_171235

theorem solution_exists_iff_divisor_form (n : ℕ) (hn_pos : 0 < n) (hn_odd : n % 2 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 4 * x * y = n * (x + y)) ↔
    (∃ k : ℕ, n % (4 * k + 3) = 0) :=
by
  sorry

end solution_exists_iff_divisor_form_l171_171235


namespace amoeba_count_14_l171_171773

noncomputable def amoeba_count (day : ℕ) : ℕ :=
  if day = 1 then 1
  else if day = 2 then 2
  else 2^(day - 3) * 5

theorem amoeba_count_14 : amoeba_count 14 = 10240 := by
  sorry

end amoeba_count_14_l171_171773


namespace kelvin_classes_l171_171983

theorem kelvin_classes (c : ℕ) (h1 : Grant = 4 * c) (h2 : c + Grant = 450) : c = 90 :=
by sorry

end kelvin_classes_l171_171983


namespace complement_computation_l171_171102

open Set

theorem complement_computation (U A : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7} → A = {2, 4, 5} →
  U \ A = {1, 3, 6, 7} :=
by
  intros hU hA
  rw [hU, hA]
  ext
  simp
  sorry

end complement_computation_l171_171102


namespace circle_intersection_unique_point_l171_171418

open Complex

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem circle_intersection_unique_point :
  ∃ k : ℝ, (distance (0, 0) (-5 / 2, 0) - 3 / 2 = k ∨ distance (0, 0) (-5 / 2, 0) + 3 / 2 = k)
  ↔ (k = 2 ∨ k = 5) := sorry

end circle_intersection_unique_point_l171_171418


namespace smallest_x_plus_y_l171_171871

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l171_171871


namespace finite_discrete_points_3_to_15_l171_171876

def goldfish_cost (n : ℕ) : ℕ := 18 * n

theorem finite_discrete_points_3_to_15 : 
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 15 → 
  ∃ (C : ℕ), C = goldfish_cost n ∧ ∃ (x : ℕ), (n, C) = (x, goldfish_cost x) :=
by
  sorry

end finite_discrete_points_3_to_15_l171_171876


namespace distinct_values_of_c_l171_171136

theorem distinct_values_of_c {c p q : ℂ} 
  (h_distinct : p ≠ q) 
  (h_eq : ∀ z : ℂ, (z - p) * (z - q) = (z - c * p) * (z - c * q)) :
  (∃ c_values : ℕ, c_values = 2) :=
sorry

end distinct_values_of_c_l171_171136


namespace solve_inequality_l171_171445

theorem solve_inequality (x : ℝ) :
  (x - 1)^2 < 12 - x ↔ 
  (Real.sqrt 5) ≠ 0 ∧
  (1 - 3 * (Real.sqrt 5)) / 2 < x ∧ 
  x < (1 + 3 * (Real.sqrt 5)) / 2 :=
sorry

end solve_inequality_l171_171445


namespace prove_seq_formula_l171_171730

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 1
| 1     => 5
| n + 2 => (2 * (seq a (n + 1))^2 - 3 * (seq a (n + 1)) - 9) / (2 * (seq a n))

theorem prove_seq_formula : ∀ (n : ℕ), seq a n = 2^(n + 2) - 3 :=
by
  sorry  -- Proof not needed for the mathematical translation

end prove_seq_formula_l171_171730


namespace morgan_change_l171_171565

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end morgan_change_l171_171565


namespace simplify_fraction_l171_171833

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l171_171833


namespace price_change_38_percent_l171_171325

variables (P : ℝ) (x : ℝ)
noncomputable def final_price := P * (1 - (x / 100)^2) * 0.9
noncomputable def target_price := 0.77 * P

theorem price_change_38_percent (h : final_price P x = target_price P):
  x = 38 := sorry

end price_change_38_percent_l171_171325


namespace base8_to_decimal_l171_171750

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end base8_to_decimal_l171_171750


namespace polynomial_root_theorem_l171_171544

theorem polynomial_root_theorem
  (α β γ δ p q : ℝ)
  (h₁ : α + β = -p)
  (h₂ : α * β = 1)
  (h₃ : γ + δ = -q)
  (h₄ : γ * δ = 1) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
by
  sorry

end polynomial_root_theorem_l171_171544


namespace total_students_l171_171241

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l171_171241


namespace parallel_vectors_m_eq_neg3_l171_171779

-- Define vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def OA : ℝ × ℝ := vec 3 (-4)
def OB : ℝ × ℝ := vec 6 (-3)
def OC (m : ℝ) : ℝ × ℝ := vec (2 * m) (m + 1)

-- Define the condition that AB is parallel to OC
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

-- Calculate AB
def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

-- The theorem to prove
theorem parallel_vectors_m_eq_neg3 (m : ℝ) :
  is_parallel AB (OC m) → m = -3 := by
  sorry

end parallel_vectors_m_eq_neg3_l171_171779


namespace frank_total_pages_read_l171_171103

-- Definitions of given conditions
def first_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def second_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def third_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days

-- Given values
def pages_first_book := first_book_pages 22 569
def pages_second_book := second_book_pages 35 315
def pages_third_book := third_book_pages 18 450

-- Total number of pages read by Frank
def total_pages := pages_first_book + pages_second_book + pages_third_book

-- Statement to prove
theorem frank_total_pages_read : total_pages = 31643 := by
  sorry

end frank_total_pages_read_l171_171103


namespace hurdle_distance_l171_171887

theorem hurdle_distance (d : ℝ) : 
  50 + 11 * d + 55 = 600 → d = 45 := by
  sorry

end hurdle_distance_l171_171887


namespace fair_dice_can_be_six_l171_171958

def fair_dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem fair_dice_can_be_six : 6 ∈ fair_dice_outcomes :=
by {
  -- This formally states that 6 is a possible outcome when throwing a fair dice
  sorry
}

end fair_dice_can_be_six_l171_171958


namespace ceil_of_neg_frac_squared_l171_171146

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l171_171146


namespace jan_25_on_thursday_l171_171513

/-- 
  Given that December 25 is on Monday,
  prove that January 25 in the following year falls on Thursday.
-/
theorem jan_25_on_thursday (day_of_week : Fin 7) (h : day_of_week = 0) : 
  ((day_of_week + 31) % 7 + 25) % 7 = 4 := 
sorry

end jan_25_on_thursday_l171_171513


namespace integer_solution_inequalities_l171_171022

theorem integer_solution_inequalities (x : ℤ) (h1 : x + 12 > 14) (h2 : -3 * x > -9) : x = 2 :=
by
  sorry

end integer_solution_inequalities_l171_171022


namespace ad_lt_bc_l171_171657

theorem ad_lt_bc (a b c d : ℝ ) (h1a : a > 0) (h1b : b > 0) (h1c : c > 0) (h1d : d > 0)
  (h2 : a + d = b + c) (h3 : |a - d| < |b - c|) : a * d < b * c :=
  sorry

end ad_lt_bc_l171_171657


namespace pythagorean_theorem_l171_171480

theorem pythagorean_theorem (a b c : ℕ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l171_171480


namespace mountain_hill_school_absent_percentage_l171_171403

theorem mountain_hill_school_absent_percentage :
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := (1 / 7) * boys
  let absent_girls := (1 / 5) * girls
  let absent_students := absent_boys + absent_girls
  let absent_percentage := (absent_students / total_students) * 100
  absent_percentage = 16.67 := sorry

end mountain_hill_school_absent_percentage_l171_171403


namespace football_cost_correct_l171_171681

variable (total_spent_on_toys : ℝ := 12.30)
variable (spent_on_marbles : ℝ := 6.59)

theorem football_cost_correct :
  (total_spent_on_toys - spent_on_marbles = 5.71) :=
by
  sorry

end football_cost_correct_l171_171681


namespace correct_equation_l171_171748

theorem correct_equation (x : ℝ) (h1 : 2000 > 0) (h2 : x > 0) (h3 : x + 40 > 0) :
  (2000 / x) - (2000 / (x + 40)) = 3 :=
by
  sorry

end correct_equation_l171_171748


namespace initial_ants_count_l171_171140

theorem initial_ants_count (n : ℕ) (h1 : ∀ x : ℕ, x ≠ n - 42 → x ≠ 42) : n = 42 :=
sorry

end initial_ants_count_l171_171140


namespace second_hand_bisect_angle_l171_171458

theorem second_hand_bisect_angle :
  ∃ x : ℚ, (6 * x - 360 * (x - 1) = 360 * (x - 1) - 0.5 * x) ∧ (x = 1440 / 1427) :=
by
  sorry

end second_hand_bisect_angle_l171_171458


namespace find_a_l171_171323

theorem find_a (a : ℝ) : (4, -5).2 = (a - 2, a + 1).2 → a = -6 :=
by
  intro h
  sorry

end find_a_l171_171323


namespace larger_number_is_50_l171_171077

variable (a b : ℕ)
-- Conditions given in the problem
axiom cond1 : 4 * b = 5 * a
axiom cond2 : b - a = 10

-- The proof statement
theorem larger_number_is_50 : b = 50 :=
sorry

end larger_number_is_50_l171_171077


namespace common_difference_arithmetic_seq_l171_171345

theorem common_difference_arithmetic_seq (a1 d : ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d) : 
  (S 5 / 5 - S 2 / 2 = 3) → d = 2 :=
by
  intros h1
  sorry

end common_difference_arithmetic_seq_l171_171345


namespace inequality_proof_l171_171065

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)^2 + bc / (b + c)^2 + ca / (c + a)^2) + (3 * (a^2 + b^2 + c^2)) / (a + b + c)^2 ≥ 7 / 4 := 
by
  sorry

end inequality_proof_l171_171065


namespace factor_expression_l171_171939

variable (a : ℤ)

theorem factor_expression : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end factor_expression_l171_171939


namespace cookies_left_l171_171814

def initial_cookies : ℕ := 93
def eaten_cookies : ℕ := 15

theorem cookies_left : initial_cookies - eaten_cookies = 78 := by
  sorry

end cookies_left_l171_171814


namespace ninety_times_ninety_l171_171643

theorem ninety_times_ninety : (90 * 90) = 8100 := by
  let a := 100
  let b := 10
  have h1 : (90 * 90) = (a - b) * (a - b) := by decide
  have h2 : (a - b) * (a - b) = a^2 - 2 * a * b + b^2 := by decide
  have h3 : a = 100 := rfl
  have h4 : b = 10 := rfl
  have h5 : 100^2 - 2 * 100 * 10 + 10^2 = 8100 := by decide
  sorry

end ninety_times_ninety_l171_171643


namespace tangent_line_at_origin_l171_171587

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem tangent_line_at_origin :
  ∃ (m b : ℝ), (m = 2) ∧ (b = 1) ∧ (∀ x, f x - (m * x + b) = 0 → 2 * x - f x + 1 = 0) :=
sorry

end tangent_line_at_origin_l171_171587


namespace students_transferred_l171_171908

theorem students_transferred (students_before : ℕ) (total_students : ℕ) (students_equal : ℕ) 
  (h1 : students_before = 23) (h2 : total_students = 50) (h3 : students_equal = total_students / 2) : 
  (∃ x : ℕ, students_equal = students_before + x) → (∃ x : ℕ, x = 2) :=
by
  -- h1: students_before = 23
  -- h2: total_students = 50
  -- h3: students_equal = total_students / 2
  -- to prove: ∃ x : ℕ, students_equal = students_before + x → ∃ x : ℕ, x = 2
  sorry

end students_transferred_l171_171908


namespace find_number_l171_171379

theorem find_number (x : ℤ) (h : 42 + 3 * x - 10 = 65) : x = 11 := 
by 
  sorry 

end find_number_l171_171379


namespace speed_difference_l171_171053

theorem speed_difference :
  let distance : ℝ := 8
  let zoe_time_hours : ℝ := 2 / 3
  let john_time_hours : ℝ := 1
  let zoe_speed : ℝ := distance / zoe_time_hours
  let john_speed : ℝ := distance / john_time_hours
  zoe_speed - john_speed = 4 :=
by
  sorry

end speed_difference_l171_171053


namespace eval_expression_l171_171865

-- Define the expression to evaluate
def expression : ℚ := 2 * 3 + 4 - (5 / 6)

-- Prove the equivalence of the evaluated expression to the expected result
theorem eval_expression : expression = 37 / 3 :=
by
  -- The detailed proof steps are omitted (relying on sorry)
  sorry

end eval_expression_l171_171865


namespace not_perfect_square_4n_squared_plus_4n_plus_4_l171_171799

theorem not_perfect_square_4n_squared_plus_4n_plus_4 :
  ¬ ∃ m n : ℕ, m^2 = 4 * n^2 + 4 * n + 4 := 
by
  sorry

end not_perfect_square_4n_squared_plus_4n_plus_4_l171_171799


namespace sqrt_18_mul_sqrt_32_eq_24_l171_171440
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l171_171440


namespace final_share_approx_equal_l171_171787

noncomputable def total_bill : ℝ := 211.0
noncomputable def number_of_people : ℝ := 6.0
noncomputable def tip_percentage : ℝ := 0.15
noncomputable def tip_amount : ℝ := tip_percentage * total_bill
noncomputable def total_amount : ℝ := total_bill + tip_amount
noncomputable def each_person_share : ℝ := total_amount / number_of_people

theorem final_share_approx_equal :
  abs (each_person_share - 40.44) < 0.01 :=
by
  sorry

end final_share_approx_equal_l171_171787


namespace islanders_liars_count_l171_171288

def number_of_liars (N : ℕ) : ℕ :=
  if N = 30 then 28 else 0

theorem islanders_liars_count : number_of_liars 30 = 28 :=
  sorry

end islanders_liars_count_l171_171288


namespace solution_set_of_equation_l171_171220

theorem solution_set_of_equation :
  {p : ℝ × ℝ | p.1 * p.2 + 1 = p.1 + p.2} = {p : ℝ × ℝ | p.1 = 1 ∨ p.2 = 1} :=
by 
  sorry

end solution_set_of_equation_l171_171220


namespace fixed_cost_to_break_even_l171_171660

def cost_per_handle : ℝ := 0.6
def selling_price_per_handle : ℝ := 4.6
def num_handles_to_break_even : ℕ := 1910

theorem fixed_cost_to_break_even (F : ℝ) (h : F = num_handles_to_break_even * (selling_price_per_handle - cost_per_handle)) :
  F = 7640 := by
  sorry

end fixed_cost_to_break_even_l171_171660


namespace count_three_digit_multiples_13_and_5_l171_171388

theorem count_three_digit_multiples_13_and_5 : 
  ∃ count : ℕ, count = 14 ∧ 
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 65 = 0) → 
  (∃ k : ℕ, n = k * 65 ∧ 2 ≤ k ∧ k ≤ 15) → count = 14 :=
by
  sorry

end count_three_digit_multiples_13_and_5_l171_171388


namespace geometric_series_sum_l171_171157

theorem geometric_series_sum : 
  let a := 1
  let r := 2
  let n := 21
  a * ((r^n - 1) / (r - 1)) = 2097151 :=
by
  sorry

end geometric_series_sum_l171_171157


namespace charity_delivered_100_plates_l171_171010

variables (cost_rice_per_plate cost_chicken_per_plate total_amount_spent : ℝ)
variable (P : ℝ)

-- Conditions provided
def rice_cost : ℝ := 0.10
def chicken_cost : ℝ := 0.40
def total_spent : ℝ := 50
def total_cost_per_plate : ℝ := rice_cost + chicken_cost

-- Lean 4 statement to prove:
theorem charity_delivered_100_plates :
  total_spent = 50 →
  total_cost_per_plate = rice_cost + chicken_cost →
  rice_cost = 0.10 →
  chicken_cost = 0.40 →
  P = total_spent / total_cost_per_plate →
  P = 100 :=
by
  sorry

end charity_delivered_100_plates_l171_171010


namespace exists_fixed_point_sequence_l171_171162

theorem exists_fixed_point_sequence (N : ℕ) (hN : 0 < N) (a : ℕ → ℕ)
  (ha_conditions : ∀ i < N, a i % 2^(N+1) ≠ 0) :
  ∃ M, ∀ n ≥ M, a n = a M :=
sorry

end exists_fixed_point_sequence_l171_171162


namespace largest_n_divisibility_l171_171702

theorem largest_n_divisibility (n : ℕ) (h : n + 12 ∣ n^3 + 144) : n ≤ 132 :=
  sorry

end largest_n_divisibility_l171_171702


namespace total_fat_l171_171630

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end total_fat_l171_171630


namespace identify_functions_l171_171303

-- Define the first expression
def expr1 (x : ℝ) : ℝ := x - (x - 3)

-- Define the second expression
noncomputable def expr2 (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (1 - x)

-- Define the third expression
noncomputable def expr3 (x : ℝ) : ℝ :=
if x < 0 then x - 1 else x + 1

-- Define the fourth expression
noncomputable def expr4 (x : ℝ) : ℝ :=
if x ∈ Set.Ioo (-1) 1 then 0 else 1

-- Proof statement
theorem identify_functions :
  (∀ x, ∃! y, expr1 x = y) ∧ (∀ x, ∃! y, expr3 x = y) ∧
  (¬ ∃ x, ∃! y, expr2 x = y) ∧ (¬ ∀ x, ∃! y, expr4 x = y) := by
    sorry

end identify_functions_l171_171303


namespace f_at_2_l171_171193

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by 
  sorry

end f_at_2_l171_171193


namespace find_m_of_symmetry_l171_171704

-- Define the conditions for the parabola and the axis of symmetry
theorem find_m_of_symmetry (m : ℝ) :
  let a := (1 : ℝ)
  let b := (m - 2 : ℝ)
  let axis_of_symmetry := (0 : ℝ)
  (-b / (2 * a)) = axis_of_symmetry → m = 2 :=
by
  sorry

end find_m_of_symmetry_l171_171704


namespace color_schemes_equivalence_l171_171597

noncomputable def number_of_non_equivalent_color_schemes (n : Nat) : Nat :=
  let total_ways := Nat.choose (n * n) 2
  -- Calculate the count for non-diametrically opposite positions (4 rotations)
  let non_diametric := (total_ways - 24) / 4
  -- Calculate the count for diametrically opposite positions (2 rotations)
  let diametric := 24 / 2
  -- Sum both counts
  non_diametric + diametric

theorem color_schemes_equivalence (n : Nat) (h : n = 7) : number_of_non_equivalent_color_schemes n = 300 :=
  by
    rw [h]
    sorry

end color_schemes_equivalence_l171_171597


namespace min_positive_d_l171_171765

theorem min_positive_d (a b t d : ℤ) (h1 : 3 * t = 2 * a + 2 * b + 2016)
                                       (h2 : t - a = d)
                                       (h3 : t - b = 2 * d)
                                       (h4 : 2 * a + 2 * b > 0) :
    ∃ d : ℤ, d > 0 ∧ (505 ≤ d ∧ ∀ e : ℤ, e > 0 → 3 * (a + d) = 2 * (b + 2 * e) + 2016 → 505 ≤ e) := 
sorry

end min_positive_d_l171_171765


namespace margarita_vs_ricciana_l171_171726

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end margarita_vs_ricciana_l171_171726


namespace pages_in_first_chapter_l171_171290

theorem pages_in_first_chapter
  (total_pages : ℕ)
  (second_chapter_pages : ℕ)
  (first_chapter_pages : ℕ)
  (h1 : total_pages = 81)
  (h2 : second_chapter_pages = 68) :
  first_chapter_pages = 81 - 68 :=
sorry

end pages_in_first_chapter_l171_171290


namespace greatest_k_for_factorial_div_l171_171177

-- Definitions for conditions in the problem
def a : Nat := Nat.factorial 100
noncomputable def b (k : Nat) : Nat := 100^k

-- Statement to prove the greatest value of k for which b is a factor of a
theorem greatest_k_for_factorial_div (k : Nat) : 
  (∀ m : Nat, (m ≤ k → b m ∣ a) ↔ m ≤ 12) := 
by
  sorry

end greatest_k_for_factorial_div_l171_171177


namespace smaller_triangle_perimeter_l171_171070

theorem smaller_triangle_perimeter (p : ℕ) (p1 : ℕ) (p2 : ℕ) (p3 : ℕ) 
  (h₀ : p = 11)
  (h₁ : p1 = 5)
  (h₂ : p2 = 7)
  (h₃ : p3 = 9) : 
  p1 + p2 + p3 - p = 10 := by
  sorry

end smaller_triangle_perimeter_l171_171070


namespace rhombus_area_of_square_4_l171_171530

theorem rhombus_area_of_square_4 :
  let A := (0, 4)
  let B := (0, 0)
  let C := (4, 0)
  let D := (4, 4)
  let F := (0, 2)  -- Midpoint of AB
  let E := (4, 2)  -- Midpoint of CD
  let FG := 2 -- Half of the side of the square (since F and E are midpoints)
  let GH := 2
  let HE := 2
  let EF := 2
  let rhombus_FGEH_area := 1 / 2 * FG * EH
  rhombus_FGEH_area = 4 := sorry

end rhombus_area_of_square_4_l171_171530


namespace find_radius_l171_171271

theorem find_radius (QP QO r : ℝ) (hQP : QP = 420) (hQO : QO = 427) : r = 77 :=
by
  -- Given QP^2 + r^2 = QO^2
  have h : (QP ^ 2) + (r ^ 2) = (QO ^ 2) := sorry
  -- Calculate the squares
  have h1 : (420 ^ 2) = 176400 := sorry
  have h2 : (427 ^ 2) = 182329 := sorry
  -- r^2 = 182329 - 176400
  have h3 : r ^ 2 = 5929 := sorry
  -- Therefore, r = 77
  exact sorry

end find_radius_l171_171271


namespace geom_seq_sum_six_div_a4_minus_one_l171_171148

theorem geom_seq_sum_six_div_a4_minus_one (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = a 1 * r^n) 
  (h2 : a 1 = 1) 
  (h3 : a 2 * a 6 - 6 * a 4 - 16 = 0) :
  S 6 / (a 4 - 1) = 9 :=
sorry

end geom_seq_sum_six_div_a4_minus_one_l171_171148


namespace min_rectilinear_distance_l171_171297

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem min_rectilinear_distance : ∀ (M : ℝ × ℝ), (M.1 - M.2 + 4 = 0) → rectilinear_distance (1, 1) M ≥ 4 :=
by
  intro M hM
  -- We only need the statement, not the proof
  sorry

end min_rectilinear_distance_l171_171297


namespace cos_alpha_beta_value_l171_171144

noncomputable def cos_alpha_beta (α β : ℝ) : ℝ :=
  Real.cos (α + β)

theorem cos_alpha_beta_value (α β : ℝ)
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  cos_alpha_beta α β = -569/800 :=
by
  sorry

end cos_alpha_beta_value_l171_171144


namespace definite_integral_cos_exp_l171_171605

open Real

theorem definite_integral_cos_exp :
  ∫ x in -π..0, (cos x + exp x) = 1 - (1 / exp π) :=
by
  sorry

end definite_integral_cos_exp_l171_171605


namespace right_triangle_ratio_is_4_l171_171491

noncomputable def right_triangle_rectangle_ratio (b h xy : ℝ) : Prop :=
  (0.4 * (1/2) * b * h = 0.25 * xy) ∧ (xy = b * h) → (b / h = 4)

theorem right_triangle_ratio_is_4 (b h xy : ℝ) (h1 : 0.4 * (1/2) * b * h = 0.25 * xy)
(h2 : xy = b * h) : b / h = 4 :=
sorry

end right_triangle_ratio_is_4_l171_171491


namespace largest_of_three_roots_l171_171501

theorem largest_of_three_roots (p q r : ℝ) (hpqr_sum : p + q + r = 3) 
    (hpqr_prod_sum : p * q + p * r + q * r = -8) (hpqr_prod : p * q * r = -15) :
    max p (max q r) = 3 := 
sorry

end largest_of_three_roots_l171_171501


namespace count_cubes_between_bounds_l171_171184

theorem count_cubes_between_bounds : ∃ (n : ℕ), n = 42 ∧
  ∀ x, 2^9 + 1 ≤ x^3 ∧ x^3 ≤ 2^17 + 1 ↔ 9 ≤ x ∧ x ≤ 50 := 
sorry

end count_cubes_between_bounds_l171_171184


namespace wendy_furniture_time_l171_171860

variable (chairs tables pieces minutes total_time : ℕ)

theorem wendy_furniture_time (h1 : chairs = 4) (h2 : tables = 4) (h3 : pieces = chairs + tables) (h4 : minutes = 6) (h5 : total_time = pieces * minutes) : total_time = 48 :=
by
  sorry

end wendy_furniture_time_l171_171860


namespace select_4_blocks_no_same_row_column_l171_171625

theorem select_4_blocks_no_same_row_column :
  ∃ (n : ℕ), n = (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4) ∧ n = 5400 :=
by
  sorry

end select_4_blocks_no_same_row_column_l171_171625


namespace order_of_values_l171_171243

noncomputable def a : ℝ := 21.2
noncomputable def b : ℝ := Real.sqrt 450 - 0.8
noncomputable def c : ℝ := 2 * Real.logb 5 2

theorem order_of_values : c < b ∧ b < a := by 
  sorry

end order_of_values_l171_171243


namespace expected_number_of_own_hats_l171_171899

-- Define the number of people
def num_people : ℕ := 2015

-- Define the expectation based on the problem description
noncomputable def expected_hats (n : ℕ) : ℝ := 1

-- The main theorem representing the problem statement
theorem expected_number_of_own_hats : expected_hats num_people = 1 := sorry

end expected_number_of_own_hats_l171_171899


namespace horner_method_correct_l171_171221

-- Define the polynomial function using Horner's method
def f (x : ℤ) : ℤ := (((((x - 8) * x + 60) * x + 16) * x + 96) * x + 240) * x + 64

-- Define the value to be plugged into the polynomial
def x_val : ℤ := 2

-- Compute v_0, v_1, and v_2 according to the Horner's method
def v0 : ℤ := 1
def v1 : ℤ := v0 * x_val - 8
def v2 : ℤ := v1 * x_val + 60

-- Formal statement of the proof problem
theorem horner_method_correct :
  v2 = 48 := by
  -- Insert proof here
  sorry

end horner_method_correct_l171_171221


namespace find_code_l171_171889

theorem find_code (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 11 * (A + B + C) = 242) :
  A = 5 ∧ B = 8 ∧ C = 9 ∨ A = 5 ∧ B = 9 ∧ C = 8 :=
by
  sorry

end find_code_l171_171889


namespace find_a_plus_b_l171_171125

theorem find_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 :=
sorry

end find_a_plus_b_l171_171125


namespace find_a_from_conditions_l171_171074

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end find_a_from_conditions_l171_171074


namespace nancy_packs_l171_171398

theorem nancy_packs (total_bars packs_bars : ℕ) (h_total : total_bars = 30) (h_packs : packs_bars = 5) :
  total_bars / packs_bars = 6 :=
by
  sorry

end nancy_packs_l171_171398


namespace adult_ticket_cost_l171_171214

-- Definitions from the conditions
def total_amount : ℕ := 35
def child_ticket_cost : ℕ := 3
def num_children : ℕ := 9

-- The amount spent on children’s tickets
def total_child_ticket_cost : ℕ := num_children * child_ticket_cost

-- The remaining amount after purchasing children’s tickets
def remaining_amount : ℕ := total_amount - total_child_ticket_cost

-- The adult ticket cost should be equal to the remaining amount
theorem adult_ticket_cost : remaining_amount = 8 :=
by sorry

end adult_ticket_cost_l171_171214


namespace sasha_quarters_max_l171_171124

/-- Sasha has \$4.80 in U.S. coins. She has four times as many dimes as she has nickels 
and the same number of quarters as nickels. Prove that the greatest number 
of quarters she could have is 6. -/
theorem sasha_quarters_max (q n d : ℝ) (h1 : 0.25 * q + 0.05 * n + 0.1 * d = 4.80)
  (h2 : n = q) (h3 : d = 4 * n) : q = 6 := 
sorry

end sasha_quarters_max_l171_171124


namespace nina_ants_count_l171_171222

theorem nina_ants_count 
  (spiders : ℕ) 
  (eyes_per_spider : ℕ) 
  (eyes_per_ant : ℕ) 
  (total_eyes : ℕ) 
  (total_spider_eyes : ℕ) 
  (total_ant_eyes : ℕ) 
  (ants : ℕ) 
  (h1 : spiders = 3) 
  (h2 : eyes_per_spider = 8) 
  (h3 : eyes_per_ant = 2) 
  (h4 : total_eyes = 124) 
  (h5 : total_spider_eyes = spiders * eyes_per_spider) 
  (h6 : total_ant_eyes = total_eyes - total_spider_eyes) 
  (h7 : ants = total_ant_eyes / eyes_per_ant) : 
  ants = 50 := by
  sorry

end nina_ants_count_l171_171222


namespace num_subsets_containing_6_l171_171545

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l171_171545


namespace cylinder_has_no_triangular_cross_section_l171_171063

inductive GeometricSolid
  | cylinder
  | cone
  | triangularPrism
  | cube

open GeometricSolid

-- Define the cross section properties
def can_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cone ∨ s = triangularPrism ∨ s = cube

-- Define the property where a solid cannot have a triangular cross-section
def cannot_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cylinder

theorem cylinder_has_no_triangular_cross_section :
  cannot_have_triangular_cross_section cylinder ∧
  ¬ can_have_triangular_cross_section cylinder :=
by
  -- This is where we state the proof goal
  sorry

end cylinder_has_no_triangular_cross_section_l171_171063


namespace michael_passes_donovan_l171_171064

noncomputable def track_length : ℕ := 600
noncomputable def donovan_lap_time : ℕ := 45
noncomputable def michael_lap_time : ℕ := 40

theorem michael_passes_donovan :
  ∃ n : ℕ, michael_lap_time * n > donovan_lap_time * (n - 1) ∧ n = 9 :=
by
  sorry

end michael_passes_donovan_l171_171064


namespace correct_calculation_l171_171514

theorem correct_calculation : ∀ (a : ℝ), a^3 * a^2 = a^5 := 
by
  intro a
  sorry

end correct_calculation_l171_171514


namespace susie_investment_l171_171722

theorem susie_investment :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 ∧
  (x * 1.04 + (2000 - x) * 1.06 = 2120) → (x = 0) :=
by
  sorry

end susie_investment_l171_171722


namespace transport_tax_correct_l171_171060

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end transport_tax_correct_l171_171060


namespace area_of_circle_l171_171180

theorem area_of_circle (r : ℝ) : 
  (S = π * r^2) :=
sorry

end area_of_circle_l171_171180


namespace play_area_l171_171549

theorem play_area (posts : ℕ) (space : ℝ) (extra_posts : ℕ) (short_posts long_posts : ℕ) (short_spaces long_spaces : ℕ) 
  (short_length long_length area : ℝ)
  (h1 : posts = 24) 
  (h2 : space = 5)
  (h3 : extra_posts = 6)
  (h4 : long_posts = short_posts + extra_posts)
  (h5 : 2 * short_posts + 2 * long_posts - 4 = posts)
  (h6 : short_spaces = short_posts - 1)
  (h7 : long_spaces = long_posts - 1)
  (h8 : short_length = short_spaces * space)
  (h9 : long_length = long_spaces * space)
  (h10 : area = short_length * long_length) :
  area = 675 := 
sorry

end play_area_l171_171549


namespace women_science_majors_is_30_percent_l171_171877

noncomputable def percentage_women_science_majors (ns_percent : ℝ) (m_percent : ℝ) (m_sci_percent : ℝ) : ℝ :=
  let w_percent := 1 - m_percent
  let m_sci_total := m_percent * m_sci_percent
  let total_sci := 1 - ns_percent
  let w_sci_total := total_sci - m_sci_total
  (w_sci_total / w_percent) * 100

theorem women_science_majors_is_30_percent :
  percentage_women_science_majors 0.60 0.40 0.55 = 30 := by
  sorry

end women_science_majors_is_30_percent_l171_171877


namespace find_m_value_l171_171058

theorem find_m_value (m : ℤ) (h1 : m - 2 ≠ 0) (h2 : |m| = 2) : m = -2 :=
by {
  sorry
}

end find_m_value_l171_171058


namespace laura_five_dollar_bills_l171_171649

theorem laura_five_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 40) 
  (h2 : x + 2 * y + 5 * z = 120) 
  (h3 : y = 2 * x) : 
  z = 16 := 
by
  sorry

end laura_five_dollar_bills_l171_171649


namespace greatest_integer_gcd_is_4_l171_171217

theorem greatest_integer_gcd_is_4 : 
  ∀ (n : ℕ), n < 150 ∧ (Nat.gcd n 24 = 4) → n ≤ 148 := 
by
  sorry

end greatest_integer_gcd_is_4_l171_171217


namespace math_proof_problem_l171_171607

noncomputable def proof_problem : Prop :=
  ∃ (p : ℝ) (k m : ℝ), 
    (∀ (x y : ℝ), y^2 = 2 * p * x) ∧
    (p > 0) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (y1 * y2 = -8) ∧
      (x1 = 4 ∧ y1 = 0 ∨ x2 = 4 ∧ y2 = 0)) ∧
    (p = 1) ∧ 
    (∀ x0 : ℝ, 
      (2 * k * m = 1) ∧
      (∀ (x y : ℝ), y = k * x + m) ∧ 
      (∃ (r : ℝ), 
        ((x0 - r + 1 = 0) ∧
         (x0 - r * x0 + r^2 = 0))) ∧ 
       x0 = -1 / 2 )

theorem math_proof_problem : proof_problem := 
  sorry

end math_proof_problem_l171_171607


namespace find_mystery_number_l171_171224

theorem find_mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 :=
sorry

end find_mystery_number_l171_171224


namespace minimum_value_of_f_l171_171658

noncomputable def f (x y z : ℝ) : ℝ := (1 / (x + y)) + (1 / (x + z)) + (1 / (y + z)) - (x * y * z)

theorem minimum_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → f x y z = 1 / 2 :=
by
  sorry

end minimum_value_of_f_l171_171658


namespace legs_per_bee_l171_171086

def number_of_bees : ℕ := 8
def total_legs : ℕ := 48

theorem legs_per_bee : (total_legs / number_of_bees) = 6 := by
  sorry

end legs_per_bee_l171_171086


namespace find_tents_l171_171825

theorem find_tents (x y : ℕ) (hx : x + y = 600) (hy : 1700 * x + 1300 * y = 940000) : x = 400 ∧ y = 200 :=
by
  sorry

end find_tents_l171_171825


namespace incorrect_conclusion_l171_171538

theorem incorrect_conclusion (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a < 1/b ∧ 1/b < 0) : ¬ (ab > b^2) :=
by
  { sorry }

end incorrect_conclusion_l171_171538


namespace log_base_half_cuts_all_horizontal_lines_l171_171987

theorem log_base_half_cuts_all_horizontal_lines (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_eq : y = Real.logb 0.5 x) : ∃ x, ∀ k, k = Real.logb 0.5 x ↔ x > 0 := 
sorry

end log_base_half_cuts_all_horizontal_lines_l171_171987


namespace alcohol_solution_problem_l171_171796

theorem alcohol_solution_problem (x_vol y_vol : ℚ) (x_alcohol y_alcohol target_alcohol : ℚ) (target_vol : ℚ) :
  x_vol = 250 ∧ x_alcohol = 10/100 ∧ y_alcohol = 30/100 ∧ target_alcohol = 25/100 ∧ target_vol = 250 + y_vol →
  (x_alcohol * x_vol + y_alcohol * y_vol = target_alcohol * target_vol) →
  y_vol = 750 :=
by
  sorry

end alcohol_solution_problem_l171_171796


namespace find_b_l171_171331

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
by {
  -- Proof will be filled in here
  sorry
}

end find_b_l171_171331


namespace problem_gcf_lcm_sum_l171_171984

-- Let A be the GCF of {15, 20, 30}
def A : ℕ := Nat.gcd (Nat.gcd 15 20) 30

-- Let B be the LCM of {15, 20, 30}
def B : ℕ := Nat.lcm (Nat.lcm 15 20) 30

-- We need to prove that A + B = 65
theorem problem_gcf_lcm_sum :
  A + B = 65 :=
by
  sorry

end problem_gcf_lcm_sum_l171_171984


namespace yogurt_cost_l171_171205

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l171_171205


namespace rope_length_eqn_l171_171410

theorem rope_length_eqn (x : ℝ) : 8^2 + (x - 3)^2 = x^2 := 
by 
  sorry

end rope_length_eqn_l171_171410


namespace monotonic_increasing_range_l171_171843

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (x + a) / x

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → (∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 < x2 → f x1 a ≤ f x2 a)) ↔ -4 ≤ a ∧ a ≤ 0 :=
sorry

end monotonic_increasing_range_l171_171843


namespace journey_speed_condition_l171_171072

theorem journey_speed_condition (v : ℝ) :
  (10 : ℝ) = 112 / v + 112 / 24 → (224 / 2 = 112) → v = 21 := by
  intros
  apply sorry

end journey_speed_condition_l171_171072


namespace geom_seq_a_sum_first_n_terms_l171_171126

noncomputable def a (n : ℕ) : ℕ := 2^(n + 1)

def b (n : ℕ) : ℕ := 3 * (n + 1) - 2

def a_b_product (n : ℕ) : ℕ := (3 * (n + 1) - 2) * 2^(n + 1)

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => a_b_product k)

theorem geom_seq_a (n : ℕ) : a (n + 1) = 2 * a n :=
by sorry

theorem sum_first_n_terms (n : ℕ) : S n = 10 + (3 * n - 5) * 2^(n + 1) :=
by sorry

end geom_seq_a_sum_first_n_terms_l171_171126


namespace flour_per_cake_l171_171804

theorem flour_per_cake (traci_flour harris_flour : ℕ) (cakes_each : ℕ)
  (h_traci_flour : traci_flour = 500)
  (h_harris_flour : harris_flour = 400)
  (h_cakes_each : cakes_each = 9) :
  (traci_flour + harris_flour) / (2 * cakes_each) = 50 := by
  sorry

end flour_per_cake_l171_171804


namespace find_teachers_and_students_l171_171483

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end find_teachers_and_students_l171_171483


namespace number_difference_l171_171550

theorem number_difference (a b : ℕ) (h1 : a + b = 25650) (h2 : a % 100 = 0) (h3 : b = a / 100) :
  a - b = 25146 :=
sorry

end number_difference_l171_171550


namespace sum_slope_y_intercept_eq_l171_171665

noncomputable def J : ℝ × ℝ := (0, 8)
noncomputable def K : ℝ × ℝ := (0, 0)
noncomputable def L : ℝ × ℝ := (10, 0)
noncomputable def G : ℝ × ℝ := ((J.1 + K.1) / 2, (J.2 + K.2) / 2)

theorem sum_slope_y_intercept_eq :
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  slope + y_intercept = 18 / 5 :=
by
  -- Place the conditions and setup here
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  -- Proof will be provided here eventually
  sorry

end sum_slope_y_intercept_eq_l171_171665


namespace largest_sum_fraction_l171_171782

theorem largest_sum_fraction :
  max (max (max (max ((1/3) + (1/2)) ((1/3) + (1/4))) ((1/3) + (1/5))) ((1/3) + (1/7))) ((1/3) + (1/9)) = 5/6 :=
by
  sorry

end largest_sum_fraction_l171_171782


namespace lines_intersect_at_same_points_l171_171936

-- Definitions of linear equations in system 1 and system 2
def line1 (a1 b1 c1 x y : ℝ) := a1 * x + b1 * y = c1
def line2 (a2 b2 c2 x y : ℝ) := a2 * x + b2 * y = c2
def line3 (a3 b3 c3 x y : ℝ) := a3 * x + b3 * y = c3
def line4 (a4 b4 c4 x y : ℝ) := a4 * x + b4 * y = c4

-- Equivalence condition of the systems
def systems_equivalent (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :=
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y)

-- Proof statement that the four lines intersect at the same set of points
theorem lines_intersect_at_same_points (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :
  systems_equivalent a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 →
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y) :=
by
  intros h_equiv x y
  exact h_equiv x y

end lines_intersect_at_same_points_l171_171936


namespace set_union_inter_proof_l171_171147

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem set_union_inter_proof : A ∪ B = {0, 1, 2, 3} ∧ A ∩ B = {1, 2} := by
  sorry

end set_union_inter_proof_l171_171147


namespace triangle_square_ratio_l171_171532

theorem triangle_square_ratio :
  ∀ (x y : ℝ), (x = 60 / 17) → (y = 780 / 169) → (x / y = 78 / 102) :=
by
  intros x y hx hy
  rw [hx, hy]
  -- the proof is skipped, as instructed
  sorry

end triangle_square_ratio_l171_171532


namespace ball_travel_distance_fourth_hit_l171_171869

theorem ball_travel_distance_fourth_hit :
  let initial_height := 150
  let rebound_ratio := 1 / 3
  let distances := [initial_height, 
                    initial_height * rebound_ratio, 
                    initial_height * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio]
  distances.sum = 294 + 1 / 3 := by
  sorry

end ball_travel_distance_fourth_hit_l171_171869


namespace initially_calculated_average_l171_171598

theorem initially_calculated_average 
  (correct_sum : ℤ)
  (incorrect_diff : ℤ)
  (num_numbers : ℤ)
  (correct_average : ℤ)
  (h1 : correct_sum = correct_average * num_numbers)
  (h2 : incorrect_diff = 20)
  (h3 : num_numbers = 10)
  (h4 : correct_average = 18) :
  (correct_sum - incorrect_diff) / num_numbers = 16 := by
  sorry

end initially_calculated_average_l171_171598


namespace maria_final_bottle_count_l171_171376

-- Define the initial conditions
def initial_bottles : ℕ := 14
def bottles_drunk : ℕ := 8
def bottles_bought : ℕ := 45

-- State the theorem to prove
theorem maria_final_bottle_count : initial_bottles - bottles_drunk + bottles_bought = 51 :=
by
  sorry

end maria_final_bottle_count_l171_171376


namespace next_chime_time_l171_171928

theorem next_chime_time (chime1_interval : ℕ) (chime2_interval : ℕ) (chime3_interval : ℕ) (start_time : ℕ) 
  (h1 : chime1_interval = 18) (h2 : chime2_interval = 24) (h3 : chime3_interval = 30) (h4 : start_time = 9) : 
  ((start_time * 60 + 6 * 60) % (24 * 60)) / 60 = 15 :=
by
  sorry

end next_chime_time_l171_171928


namespace train_departure_time_l171_171172

theorem train_departure_time 
(distance speed : ℕ) (arrival_time_chicago difference : ℕ) (arrival_time_new_york departure_time : ℕ) 
(h_dist : distance = 480) 
(h_speed : speed = 60)
(h_arrival_chicago : arrival_time_chicago = 17) 
(h_difference : difference = 1)
(h_arrival_new_york : arrival_time_new_york = arrival_time_chicago + difference) : 
  departure_time = arrival_time_new_york - distance / speed :=
by
  sorry

end train_departure_time_l171_171172


namespace brown_eyed_brunettes_count_l171_171357

-- Definitions of conditions
variables (total_students blue_eyed_blondes brunettes brown_eyed_students : ℕ)
variable (brown_eyed_brunettes : ℕ)

-- Initial conditions
axiom h1 : total_students = 60
axiom h2 : blue_eyed_blondes = 18
axiom h3 : brunettes = 40
axiom h4 : brown_eyed_students = 24

-- Proof objective
theorem brown_eyed_brunettes_count :
  brown_eyed_brunettes = 24 - (24 - (20 - (20 - 18))) := sorry

end brown_eyed_brunettes_count_l171_171357


namespace three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l171_171695

theorem three_times_two_to_the_n_minus_one_gt_n_squared_plus_three (n : ℕ) (h : n ≥ 4) : 3 * 2^(n-1) > n^2 + 3 := by
  sorry

end three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l171_171695


namespace seven_nat_sum_divisible_by_5_l171_171414

theorem seven_nat_sum_divisible_by_5 
  (a b c d e f g : ℕ)
  (h1 : (b + c + d + e + f + g) % 5 = 0)
  (h2 : (a + c + d + e + f + g) % 5 = 0)
  (h3 : (a + b + d + e + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + c + d + f + g) % 5 = 0)
  (h6 : (a + b + c + d + e + g) % 5 = 0)
  (h7 : (a + b + c + d + e + f) % 5 = 0)
  : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end seven_nat_sum_divisible_by_5_l171_171414


namespace different_prime_factors_of_factorial_eq_10_l171_171245

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l171_171245


namespace sum_of_non_common_roots_zero_l171_171364

theorem sum_of_non_common_roots_zero (m α β γ : ℝ) 
  (h1 : α + β = -(m + 1))
  (h2 : α * β = -3)
  (h3 : α + γ = 4)
  (h4 : α * γ = -m)
  (h_common : α^2 + (m + 1)*α - 3 = 0)
  (h_common2 : α^2 - 4*α - m = 0)
  : β + γ = 0 := sorry

end sum_of_non_common_roots_zero_l171_171364


namespace exists_prime_q_not_div_n_p_minus_p_l171_171646

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem exists_prime_q_not_div_n_p_minus_p :
  ∃ q : ℕ, Nat.Prime q ∧ q ≠ p ∧ ∀ n : ℕ, ¬ q ∣ (n ^ p - p) :=
sorry

end exists_prime_q_not_div_n_p_minus_p_l171_171646


namespace find_number_lemma_l171_171499

theorem find_number_lemma (x : ℝ) (a b c d : ℝ) (h₁ : x = 5) 
  (h₂ : a = 0.47 * 1442) (h₃ : b = 0.36 * 1412) 
  (h₄ : c = a - b) (h₅ : d + c = x) : 
  d = -164.42 :=
by
  sorry

end find_number_lemma_l171_171499


namespace regina_has_20_cows_l171_171754

theorem regina_has_20_cows (C P : ℕ)
  (h1 : P = 4 * C)
  (h2 : 400 * P + 800 * C = 48000) :
  C = 20 :=
by
  sorry

end regina_has_20_cows_l171_171754


namespace mass_percentage_of_carbon_in_ccl4_l171_171041

-- Define the atomic masses
def atomic_mass_c : Float := 12.01
def atomic_mass_cl : Float := 35.45

-- Define the molecular composition of Carbon Tetrachloride (CCl4)
def mol_mass_ccl4 : Float := (1 * atomic_mass_c) + (4 * atomic_mass_cl)

-- Theorem to prove the mass percentage of carbon in Carbon Tetrachloride is 7.81%
theorem mass_percentage_of_carbon_in_ccl4 : 
  (atomic_mass_c / mol_mass_ccl4) * 100 = 7.81 := by
  sorry

end mass_percentage_of_carbon_in_ccl4_l171_171041


namespace full_capacity_l171_171194

def oil_cylinder_capacity (C : ℝ) :=
  (4 / 5) * C - (3 / 4) * C = 4

theorem full_capacity : oil_cylinder_capacity 80 :=
by
  simp [oil_cylinder_capacity]
  sorry

end full_capacity_l171_171194


namespace product_of_five_integers_l171_171803

theorem product_of_five_integers (E F G H I : ℚ)
  (h1 : E + F + G + H + I = 110)
  (h2 : E / 2 = F / 3 ∧ F / 3 = G * 4 ∧ G * 4 = H * 2 ∧ H * 2 = I - 5) :
  E * F * G * H * I = 623400000 / 371293 := by
  sorry

end product_of_five_integers_l171_171803


namespace gp_condition_necessity_l171_171430

theorem gp_condition_necessity {a b c : ℝ} 
    (h_gp: ∃ r: ℝ, b = a * r ∧ c = a * r^2 ) : b^2 = a * c :=
by
  sorry

end gp_condition_necessity_l171_171430


namespace positivity_of_xyz_l171_171648

variable {x y z : ℝ}

theorem positivity_of_xyz
  (h1 : x + y + z > 0)
  (h2 : xy + yz + zx > 0)
  (h3 : xyz > 0) :
  x > 0 ∧ y > 0 ∧ z > 0 := 
sorry

end positivity_of_xyz_l171_171648


namespace find_value_of_a_l171_171977

noncomputable def f (x a : ℝ) : ℝ := (3 * Real.log x - x^2 - a - 2)^2 + (x - a)^2

theorem find_value_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ 8) ↔ a = -1 :=
by
  sorry

end find_value_of_a_l171_171977


namespace series_sum_eq_l171_171378

noncomputable def sum_series (k : ℝ) : ℝ :=
  (∑' n : ℕ, (4 * (n + 1) + k) / 3^(n + 1))

theorem series_sum_eq (k : ℝ) : sum_series k = 3 + k / 2 := 
  sorry

end series_sum_eq_l171_171378


namespace real_number_a_value_l171_171766

open Set

variable {a : ℝ}

theorem real_number_a_value (A B : Set ℝ) (hA : A = {-1, 1, 3}) (hB : B = {a + 2, a^2 + 4}) (hAB : A ∩ B = {3}) : a = 1 := 
by 
-- Step proof will be here
sorry

end real_number_a_value_l171_171766


namespace cisco_spots_difference_l171_171018

theorem cisco_spots_difference :
  ∃ C G R : ℕ, R = 46 ∧ G = 5 * C ∧ G + C = 108 ∧ (23 - C) = 5 :=
by
  sorry

end cisco_spots_difference_l171_171018


namespace problem_statement_l171_171446

noncomputable def count_valid_numbers : Nat :=
  let digits := [1, 2, 3, 4, 5]
  let repeated_digit_choices := 5
  let positions_for_repeated_digits := Nat.choose 5 2
  let cases_for_tens_and_hundreds :=
    2 * 3 + 2 + 1
  let two_remaining_digits_permutations := 2
  repeated_digit_choices * positions_for_repeated_digits * cases_for_tens_and_hundreds * two_remaining_digits_permutations

theorem problem_statement : count_valid_numbers = 800 := by
  sorry

end problem_statement_l171_171446


namespace simplified_expression_is_one_l171_171612

-- Define the specific mathematical expressions
def expr1 := -1 ^ 2023
def expr2 := (-2) ^ 3
def expr3 := (-2) * (-3)

-- Construct the full expression
def full_expr := expr1 - expr2 - expr3

-- State the theorem that this full expression equals 1
theorem simplified_expression_is_one : full_expr = 1 := by
  sorry

end simplified_expression_is_one_l171_171612


namespace consecutive_rolls_probability_l171_171560

theorem consecutive_rolls_probability : 
  let total_outcomes := 36
  let consecutive_events := 10
  (consecutive_events / total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end consecutive_rolls_probability_l171_171560


namespace parabola_c_value_l171_171427

theorem parabola_c_value {b c : ℝ} :
  (1:ℝ)^2 + b * (1:ℝ) + c = 2 → 
  (4:ℝ)^2 + b * (4:ℝ) + c = 5 → 
  (7:ℝ)^2 + b * (7:ℝ) + c = 2 →
  c = 9 :=
by
  intros h1 h2 h3
  sorry

end parabola_c_value_l171_171427


namespace range_of_x_l171_171719

variable {x p : ℝ}

theorem range_of_x (H : 0 ≤ p ∧ p ≤ 4) : 
  (x^2 + p * x > 4 * x + p - 3) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 3) := 
by
  sorry

end range_of_x_l171_171719


namespace positive_iff_triangle_l171_171619

def is_triangle_inequality (x y z : ℝ) : Prop :=
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x)

noncomputable def poly (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem positive_iff_triangle (x y z : ℝ) : 
  poly |x| |y| |z| > 0 ↔ is_triangle_inequality |x| |y| |z| :=
sorry

end positive_iff_triangle_l171_171619


namespace students_with_both_uncool_parents_l171_171319

theorem students_with_both_uncool_parents :
  let total_students := 35
  let cool_dads := 18
  let cool_moms := 22
  let both_cool := 11
  total_students - (cool_dads + cool_moms - both_cool) = 6 := by
sorry

end students_with_both_uncool_parents_l171_171319


namespace expression_evaluate_l171_171283

theorem expression_evaluate (a b c : ℤ) (h1 : b = a + 2) (h2 : c = b - 10) (ha : a = 4)
(h3 : a ≠ -1) (h4 : b ≠ 2) (h5 : b ≠ -4) (h6 : c ≠ -6) : (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 :=
by
  sorry

end expression_evaluate_l171_171283


namespace find_f_function_l171_171935

def oddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem find_f_function (f : ℝ → ℝ) (h_odd : oddFunction f) (h_pos : ∀ x, 0 < x → f x = x * (1 + x)) :
  ∀ x, x < 0 → f x = -x - x^2 :=
by
  sorry

end find_f_function_l171_171935


namespace batsman_average_l171_171394

theorem batsman_average (A : ℝ) (h1 : 24 * A < 95) 
                        (h2 : 24 * A + 95 = 25 * (A + 3.5)) : A + 3.5 = 11 :=
by
  sorry

end batsman_average_l171_171394


namespace coffee_containers_used_l171_171933

theorem coffee_containers_used :
  let Suki_coffee := 6.5 * 22
  let Jimmy_coffee := 4.5 * 18
  let combined_coffee := Suki_coffee + Jimmy_coffee
  let containers := combined_coffee / 8
  containers = 28 := 
by
  sorry

end coffee_containers_used_l171_171933


namespace speed_of_water_l171_171049

theorem speed_of_water (v : ℝ) :
  (∀ (distance time : ℝ), distance = 16 ∧ time = 8 → distance = (4 - v) * time) → 
  v = 2 :=
by
  intro h
  have h1 : 16 = (4 - v) * 8 := h 16 8 (by simp)
  sorry

end speed_of_water_l171_171049


namespace distinct_roots_of_transformed_polynomial_l171_171548

theorem distinct_roots_of_transformed_polynomial
  (a b c : ℝ)
  (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
                    (a * x^5 + b * x^4 + c = 0) ∧ 
                    (a * y^5 + b * y^4 + c = 0) ∧ 
                    (a * z^5 + b * z^4 + c = 0)) :
  ∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
               (c * u^5 + b * u + a = 0) ∧ 
               (c * v^5 + b * v + a = 0) ∧ 
               (c * w^5 + b * w + a = 0) :=
  sorry

end distinct_roots_of_transformed_polynomial_l171_171548


namespace unique_solution_l171_171387

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l171_171387


namespace compute_expression_l171_171749

theorem compute_expression : 42 * 52 + 48 * 42 = 4200 :=
by sorry

end compute_expression_l171_171749


namespace find_new_length_l171_171310

def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 24
def final_length_cm : ℕ := 76

theorem find_new_length : initial_length_cm - erased_length_cm = final_length_cm := by
  sorry

end find_new_length_l171_171310


namespace proof_equivalence_l171_171012

variable {x y : ℝ}

theorem proof_equivalence (h : x - y = 1) : x^3 - 3 * x * y - y^3 = 1 := by
  sorry

end proof_equivalence_l171_171012


namespace max_hours_worked_l171_171543

theorem max_hours_worked
  (r : ℝ := 8)  -- Regular hourly rate
  (h_r : ℝ := 20)  -- Hours at regular rate
  (r_o : ℝ := r + 0.25 * r)  -- Overtime hourly rate
  (E : ℝ := 410)  -- Total weekly earnings
  : (h_r + (E - r * h_r) / r_o) = 45 :=
by
  sorry

end max_hours_worked_l171_171543


namespace complex_fraction_expression_equals_half_l171_171616

theorem complex_fraction_expression_equals_half :
  ((2 / (3 + 1/5)) + (((3 + 1/4) / 13) / (2 / 3)) + (((2 + 5/18) - (17/36)) * (18 / 65))) * (1 / 3) = 0.5 :=
by
  sorry

end complex_fraction_expression_equals_half_l171_171616


namespace problem_solution_l171_171097

theorem problem_solution (a b c : ℝ)
  (h₁ : 10 = (6 / 100) * a)
  (h₂ : 6 = (10 / 100) * b)
  (h₃ : c = b / a) : c = 0.36 :=
by sorry

end problem_solution_l171_171097


namespace find_units_digit_l171_171534

theorem find_units_digit (A : ℕ) (h : 10 * A + 2 = 20 + A + 9) : A = 3 :=
by
  sorry

end find_units_digit_l171_171534


namespace daniel_video_games_l171_171269

/--
Daniel has a collection of some video games. 80 of them, Daniel bought for $12 each.
Of the rest, 50% were bought for $7. All others had a price of $3 each.
Daniel spent $2290 on all the games in his collection.
Prove that the total number of video games in Daniel's collection is 346.
-/
theorem daniel_video_games (n : ℕ) (r : ℕ)
    (h₀ : 80 * 12 = 960)
    (h₁ : 2290 - 960 = 1330)
    (h₂ : r / 2 * 7 + r / 2 * 3 = 1330):
    n = 80 + r → n = 346 :=
by
  intro h_total
  have r_eq : r = 266 := by sorry
  rw [r_eq] at h_total
  exact h_total

end daniel_video_games_l171_171269


namespace pages_with_same_units_digit_count_l171_171954

theorem pages_with_same_units_digit_count {n : ℕ} (h1 : n = 67) :
  ∃ k : ℕ, k = 13 ∧
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ n → 
    (x ≡ (n + 1 - x) [MOD 10] ↔ 
     (x % 10 = 4 ∨ x % 10 = 9))) :=
by
  sorry

end pages_with_same_units_digit_count_l171_171954


namespace find_a9_l171_171441

variable (S : ℕ → ℚ) (a : ℕ → ℚ) (n : ℕ) (d : ℚ)

-- Conditions
axiom sum_first_six : S 6 = 3
axiom sum_first_eleven : S 11 = 18
axiom Sn_definition : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n)
axiom arithmetic_sequence : ∀ n, a (n + 1) = a 1 + n * d

-- Problem statement
theorem find_a9 : a 9 = 3 := sorry

end find_a9_l171_171441


namespace meaningful_range_l171_171160

theorem meaningful_range (x : ℝ) : (x < 4) ↔ (4 - x > 0) := 
by sorry

end meaningful_range_l171_171160


namespace minimum_bailing_rate_l171_171123

-- Conditions
def distance_to_shore : ℝ := 2 -- miles
def rowing_speed : ℝ := 3 -- miles per hour
def water_intake_rate : ℝ := 15 -- gallons per minute
def max_water_capacity : ℝ := 50 -- gallons

-- Result to prove
theorem minimum_bailing_rate (r : ℝ) : 
  (distance_to_shore / rowing_speed * 60 * water_intake_rate - distance_to_shore / rowing_speed * 60 * r) ≤ max_water_capacity →
  r ≥ 13.75 :=
by
  sorry

end minimum_bailing_rate_l171_171123


namespace smallest_sum_of_squares_value_l171_171004

noncomputable def collinear_points_min_value (A B C D E P : ℝ): Prop :=
  let AB := 3
  let BC := 2
  let CD := 5
  let DE := 4
  let pos_A := 0
  let pos_B := pos_A + AB
  let pos_C := pos_B + BC
  let pos_D := pos_C + CD
  let pos_E := pos_D + DE
  let P := P
  let AP := (P - pos_A)
  let BP := (P - pos_B)
  let CP := (P - pos_C)
  let DP := (P - pos_D)
  let EP := (P - pos_E)
  let sum_squares := AP^2 + BP^2 + CP^2 + DP^2 + EP^2
  (sum_squares = 85.2)

theorem smallest_sum_of_squares_value : ∃ (A B C D E P : ℝ), collinear_points_min_value A B C D E P :=
sorry

end smallest_sum_of_squares_value_l171_171004


namespace h_evaluation_l171_171819

variables {a b c : ℝ}

-- Definitions and conditions
def p (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c
def h (x : ℝ) : ℝ := sorry -- Definition of h(x) in terms of the roots of p(x)

theorem h_evaluation (ha : a < b) (hb : b < c) : h 2 = (2 + 2 * a + 3 * b + c) / (c^2) :=
sorry

end h_evaluation_l171_171819


namespace calculation_equality_l171_171008

theorem calculation_equality : ((8^5 / 8^2) * 4^4) = 2^17 := by
  sorry

end calculation_equality_l171_171008


namespace number_of_classes_l171_171360

theorem number_of_classes (x : ℕ) (total_games : ℕ) (h : total_games = 45) :
  (x * (x - 1)) / 2 = total_games → x = 10 :=
by
  sorry

end number_of_classes_l171_171360


namespace compute_LM_length_l171_171042

-- Definitions of lengths and equidistant property
variables (GH JK LM : ℝ) 
variables (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
variables (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK)

-- State the theorem to prove lengths
theorem compute_LM_length (GH JD LM : ℝ) (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
  (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK) :
  LM = (2 / 3) * 80 := 
sorry

end compute_LM_length_l171_171042


namespace stratified_sampling_elderly_employees_l171_171632

-- Definitions for the conditions
def total_employees : ℕ := 430
def young_employees : ℕ := 160
def middle_aged_employees : ℕ := 180
def elderly_employees : ℕ := 90
def sample_young_employees : ℕ := 32

-- The property we want to prove
theorem stratified_sampling_elderly_employees :
  (sample_young_employees / young_employees) * elderly_employees = 18 :=
by
  sorry

end stratified_sampling_elderly_employees_l171_171632


namespace probability_first_queen_second_diamond_l171_171080

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l171_171080


namespace find_k_l171_171460

-- Define the vectors a, b, and c
def vecA : ℝ × ℝ := (2, -1)
def vecB : ℝ × ℝ := (1, 1)
def vecC : ℝ × ℝ := (-5, 1)

-- Define the condition for two vectors being parallel
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

-- Define the target statement to be proven
theorem find_k : ∃ k : ℝ, parallel (vecA.1 + k * vecB.1, vecA.2 + k * vecB.2) vecC ∧ k = 1/2 := 
sorry

end find_k_l171_171460


namespace probability_of_winning_exactly_once_l171_171090

-- Define the probability of player A winning a match
def prob_win_A (p : ℝ) : Prop := (1 - p) ^ 3 = 1 - 63 / 64

-- Define the binomial probability for exactly one win in three matches
def binomial_prob (p : ℝ) : ℝ := 3 * p * (1 - p) ^ 2

theorem probability_of_winning_exactly_once (p : ℝ) (h : prob_win_A p) : binomial_prob p = 9 / 64 :=
sorry

end probability_of_winning_exactly_once_l171_171090


namespace intersection_empty_l171_171142

open Set

-- Definition of set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 2 * x + 3) }

-- Definition of set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 4 * x + 1) }

-- The proof problem statement in Lean
theorem intersection_empty : A ∩ B = ∅ := sorry

end intersection_empty_l171_171142


namespace Sasha_added_digit_l171_171056

noncomputable def Kolya_number : Nat := 45 -- Sum of all digits 0 to 9

theorem Sasha_added_digit (d x : Nat) (h : 0 ≤ d ∧ d ≤ 9) (h1 : 0 ≤ x ∧ x ≤ 9) (condition : Kolya_number - d + x ≡ 0 [MOD 9]) : x = 0 ∨ x = 9 := 
sorry

end Sasha_added_digit_l171_171056


namespace price_of_turban_l171_171082

theorem price_of_turban (T : ℝ) (h1 : ∀ (T : ℝ), 3 / 4 * (90 + T) = 40 + T) : T = 110 :=
by
  sorry

end price_of_turban_l171_171082


namespace smallest_n_l171_171023

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def condition_for_n (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → ∀ x : ℕ, x ∈ M k → ∃ y : ℕ, y ∈ M k ∧ y ≠ x ∧ is_perfect_square (x + y)
  where M (k : ℕ) := { m : ℕ | m > 0 ∧ m ≤ k }

theorem smallest_n : ∃ n : ℕ, (condition_for_n n) ∧ (∀ m < n, ¬ condition_for_n m) :=
  sorry

end smallest_n_l171_171023


namespace quadratic_value_at_5_l171_171664

-- Define the conditions provided in the problem
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Create a theorem that states that if a quadratic with given conditions has its vertex at (2, 7) and passes through (0, -7), then passing through (5, n) means n = -24.5
theorem quadratic_value_at_5 (a b c n : ℝ)
  (h1 : quadratic a b c 2 = 7)
  (h2 : quadratic a b c 0 = -7)
  (h3 : quadratic a b c 5 = n) :
  n = -24.5 :=
by
  sorry

end quadratic_value_at_5_l171_171664


namespace Portia_school_students_l171_171966

theorem Portia_school_students:
  ∃ (P L : ℕ), P = 2 * L ∧ P + L = 3000 ∧ P = 2000 :=
by
  sorry

end Portia_school_students_l171_171966


namespace hyperbola_transformation_l171_171897

def equation_transform (x y : ℝ) : Prop :=
  y = (1 - 3 * x) / (2 * x - 1)

def coordinate_shift (x y X Y : ℝ) : Prop :=
  X = x - 0.5 ∧ Y = y + 1.5

theorem hyperbola_transformation (x y X Y : ℝ) :
  equation_transform x y →
  coordinate_shift x y X Y →
  (X * Y = -0.25) :=
by
  sorry

end hyperbola_transformation_l171_171897


namespace a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l171_171294

theorem a_sq_greater_than_b_sq_neither_sufficient_nor_necessary 
  (a b : ℝ) : ¬ ((a^2 > b^2) → (a > b)) ∧  ¬ ((a > b) → (a^2 > b^2)) := sorry

end a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l171_171294


namespace combined_work_time_l171_171117

def Worker_A_time : ℝ := 10
def Worker_B_time : ℝ := 15

theorem combined_work_time :
  (1 / Worker_A_time + 1 / Worker_B_time)⁻¹ = 6 := by
  sorry

end combined_work_time_l171_171117


namespace sum_equality_l171_171763

-- Define the conditions and hypothesis
variables (x y z : ℝ)
axiom condition : (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0

-- State the theorem
theorem sum_equality : x + y + z = 21 :=
by sorry

end sum_equality_l171_171763


namespace John_scored_24point5_goals_l171_171926

theorem John_scored_24point5_goals (T G : ℝ) (n : ℕ) (A : ℝ)
  (h1 : T = 65)
  (h2 : n = 9)
  (h3 : A = 4.5) :
  G = T - (n * A) :=
by
  sorry

end John_scored_24point5_goals_l171_171926


namespace time_for_B_alone_to_complete_work_l171_171687

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end time_for_B_alone_to_complete_work_l171_171687


namespace no_solution_of_abs_sum_l171_171578

theorem no_solution_of_abs_sum (a : ℝ) : (∀ x : ℝ, |x - 2| + |x + 3| < a → false) ↔ a ≤ 5 := sorry

end no_solution_of_abs_sum_l171_171578


namespace steven_apples_peaches_difference_l171_171881

def steven_apples := 19
def jake_apples (steven_apples : ℕ) := steven_apples + 4
def jake_peaches (steven_peaches : ℕ) := steven_peaches - 3

theorem steven_apples_peaches_difference (P : ℕ) :
  19 - P = steven_apples - P :=
by
  sorry

end steven_apples_peaches_difference_l171_171881


namespace coffee_ratio_is_one_to_five_l171_171079

-- Given conditions
def thermos_capacity : ℕ := 20 -- capacity in ounces
def times_filled_per_day : ℕ := 2
def school_days_per_week : ℕ := 5
def new_weekly_coffee_consumption : ℕ := 40 -- in ounces

-- Definitions based on the conditions
def old_daily_coffee_consumption := thermos_capacity * times_filled_per_day
def old_weekly_coffee_consumption := old_daily_coffee_consumption * school_days_per_week

-- Theorem: The ratio of the new weekly coffee consumption to the old weekly coffee consumption is 1:5
theorem coffee_ratio_is_one_to_five : 
  new_weekly_coffee_consumption / old_weekly_coffee_consumption = 1 / 5 := 
by
  -- Proof is omitted
  sorry

end coffee_ratio_is_one_to_five_l171_171079


namespace ratio_problem_l171_171311

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l171_171311


namespace triangle_height_from_area_l171_171703

theorem triangle_height_from_area {A b h : ℝ} (hA : A = 36) (hb : b = 8) 
    (formula : A = 1 / 2 * b * h) : h = 9 := 
by
  sorry

end triangle_height_from_area_l171_171703


namespace partnership_profit_l171_171306

noncomputable def totalProfit (P Q R : ℕ) (unit_value_per_share : ℕ) : ℕ :=
  let profit_p := 36 * 2 + 18 * 10
  let profit_q := 24 * 12
  let profit_r := 36 * 12
  (profit_p + profit_q + profit_r) * unit_value_per_share

theorem partnership_profit (P Q R : ℕ) (unit_value_per_share : ℕ) :
  (P / Q = 3 / 2) → (Q / R = 4 / 3) → 
  (unit_value_per_share = 144 / 288) → 
  totalProfit P Q R (unit_value_per_share * 1) = 486 := 
by
  intros h1 h2 h3
  sorry

end partnership_profit_l171_171306


namespace dandelion_seed_production_l171_171522

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end dandelion_seed_production_l171_171522


namespace average_weight_of_whole_class_l171_171099

theorem average_weight_of_whole_class (n_a n_b : ℕ) (w_a w_b : ℕ) (avg_w_a avg_w_b : ℕ)
  (h_a : n_a = 36) (h_b : n_b = 24) (h_avg_a : avg_w_a = 30) (h_avg_b : avg_w_b = 30) :
  ((n_a * avg_w_a + n_b * avg_w_b) / (n_a + n_b) = 30) := 
by
  sorry

end average_weight_of_whole_class_l171_171099


namespace bakery_made_muffins_l171_171747

-- Definitions based on conditions
def muffins_per_box : ℕ := 5
def available_boxes : ℕ := 10
def additional_boxes_needed : ℕ := 9

-- Theorem statement
theorem bakery_made_muffins :
  (available_boxes * muffins_per_box) + (additional_boxes_needed * muffins_per_box) = 95 := 
by
  sorry

end bakery_made_muffins_l171_171747


namespace isosceles_triangle_perimeter_l171_171487

-- Define the lengths of the sides of the isosceles triangle
def side1 : ℕ := 12
def side2 : ℕ := 12
def base : ℕ := 17

-- Define the perimeter as the sum of all three sides
def perimeter : ℕ := side1 + side2 + base

-- State the theorem that needs to be proved
theorem isosceles_triangle_perimeter : perimeter = 41 := by
  -- Insert the proof here
  sorry

end isosceles_triangle_perimeter_l171_171487


namespace value_of_expression_l171_171591

theorem value_of_expression (x y : ℝ) (hy : y > 0) (h : x = 3 * y) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end value_of_expression_l171_171591


namespace cannot_be_sum_of_six_consecutive_odds_l171_171821

def is_sum_of_six_consecutive_odds (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (6 * k + 30)

theorem cannot_be_sum_of_six_consecutive_odds :
  ¬ is_sum_of_six_consecutive_odds 198 ∧ ¬ is_sum_of_six_consecutive_odds 390 := 
sorry

end cannot_be_sum_of_six_consecutive_odds_l171_171821


namespace total_sales_l171_171515

theorem total_sales (S : ℕ) (h1 : (1 / 3 : ℚ) * S + (1 / 4 : ℚ) * S = (1 - (1 / 3 + 1 / 4)) * S + 15) : S = 36 :=
by
  sorry

end total_sales_l171_171515


namespace art_museum_visitors_l171_171880

theorem art_museum_visitors 
  (V : ℕ)
  (H1 : ∃ (d : ℕ), d = 130)
  (H2 : ∃ (e u : ℕ), e = u)
  (H3 : ∃ (x : ℕ), x = (3 * V) / 4)
  (H4 : V = (3 * V) / 4 + 130) :
  V = 520 :=
sorry

end art_museum_visitors_l171_171880


namespace monomial_completes_square_l171_171085

variable (x : ℝ)

theorem monomial_completes_square :
  ∃ (m : ℝ), ∀ (x : ℝ), ∃ (a b : ℝ), (16 * x^2 + 1 + m) = (a * x + b)^2 :=
sorry

end monomial_completes_square_l171_171085


namespace ceil_floor_difference_is_3_l171_171187

noncomputable def ceil_floor_difference : ℤ :=
  Int.ceil ((14:ℚ) / 5 * (-31 / 3)) - Int.floor ((14 / 5) * Int.floor ((-31:ℚ) / 3))

theorem ceil_floor_difference_is_3 : ceil_floor_difference = 3 :=
  sorry

end ceil_floor_difference_is_3_l171_171187


namespace max_sum_of_squares_l171_171327

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 86) 
  (h3 : ad + bc = 180) 
  (h4 : cd = 110) : 
  a^2 + b^2 + c^2 + d^2 ≤ 258 :=
sorry

end max_sum_of_squares_l171_171327


namespace find_constant_a_l171_171045

theorem find_constant_a (a : ℚ) (S : ℕ → ℚ) (hS : ∀ n, S n = (a - 2) * 3^(n + 1) + 2) : a = 4 / 3 :=
by
  sorry

end find_constant_a_l171_171045


namespace sub_neg_eq_add_pos_l171_171044

theorem sub_neg_eq_add_pos : 0 - (-2) = 2 := 
by
  sorry

end sub_neg_eq_add_pos_l171_171044


namespace cube_side_length_l171_171771

def cube_volume (side : ℝ) : ℝ := side ^ 3

theorem cube_side_length (volume : ℝ) (h : volume = 729) : ∃ (side : ℝ), side = 9 ∧ cube_volume side = volume :=
by
  sorry

end cube_side_length_l171_171771


namespace percentage_by_which_x_is_more_than_y_l171_171156

variable {z : ℝ} 

-- Define x and y based on the given conditions
def x (z : ℝ) : ℝ := 0.78 * z
def y (z : ℝ) : ℝ := 0.60 * z

-- The main theorem we aim to prove
theorem percentage_by_which_x_is_more_than_y (z : ℝ) : x z = y z + 0.30 * y z := by
  sorry

end percentage_by_which_x_is_more_than_y_l171_171156


namespace problem_1_problem_2_problem_3_l171_171564

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem problem_1 : f (Real.pi / 2) = 1 := 
sorry

theorem problem_2 : (∃ p > 0, ∀ x, f (x + p) = f x) ∧ (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi) := 
sorry

theorem problem_3 : ∃ x : ℝ, g x = -2 := 
sorry

end problem_1_problem_2_problem_3_l171_171564


namespace total_books_count_l171_171624

theorem total_books_count (books_read : ℕ) (books_unread : ℕ) (h1 : books_read = 13) (h2 : books_unread = 8) : books_read + books_unread = 21 := 
by
  -- Proof omitted
  sorry

end total_books_count_l171_171624


namespace percentage_error_in_area_l171_171300

theorem percentage_error_in_area (S : ℝ) (h : S > 0) :
  let S' := S * 1.06
  let A := S^2
  let A' := (S')^2
  (A' - A) / A * 100 = 12.36 := by
  sorry

end percentage_error_in_area_l171_171300


namespace min_distance_PQ_l171_171457

theorem min_distance_PQ :
  let P_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = 4
  let Q_circle (ρ θ : ℝ) := ρ^2 = 4 * ρ * Real.cos θ - 3
  ∃ (P Q : ℝ × ℝ), 
    (∃ ρP θP, P = (ρP * Real.cos θP, ρP * Real.sin θP) ∧ P_line ρP θP) ∧
    (∃ ρQ θQ, Q = (ρQ * Real.cos θQ, ρQ * Real.sin θQ) ∧ Q_circle ρQ θQ) ∧
    ∀ R S : ℝ × ℝ, 
      (∃ ρR θR, R = (ρR * Real.cos θR, ρR * Real.sin θR) ∧ P_line ρR θR) →
      (∃ ρS θS, S = (ρS * Real.cos θS, ρS * Real.sin θS) ∧ Q_circle ρS θS) →
      dist P Q ≤ dist R S :=
  sorry

end min_distance_PQ_l171_171457


namespace geometric_seq_sum_l171_171318

noncomputable def a_n (n : ℕ) : ℤ :=
  (-3)^(n-1)

theorem geometric_seq_sum :
  let a1 := a_n 1
  let a2 := a_n 2
  let a3 := a_n 3
  let a4 := a_n 4
  let a5 := a_n 5
  a1 + |a2| + a3 + |a4| + a5 = 121 :=
by
  sorry

end geometric_seq_sum_l171_171318


namespace alice_operations_terminate_l171_171208

theorem alice_operations_terminate (a : List ℕ) (h_pos : ∀ x ∈ a, x > 0) : 
(∀ x y z, (x, y) = (y + 1, x) ∨ (x, y) = (x - 1, x) → ∃ n, (x :: y :: z).sum ≤ n) :=
by sorry

end alice_operations_terminate_l171_171208


namespace number_of_B_students_l171_171091

-- Conditions
def prob_A (prob_B : ℝ) := 0.6 * prob_B
def prob_C (prob_B : ℝ) := 1.6 * prob_B
def prob_D (prob_B : ℝ) := 0.3 * prob_B

-- Total students
def total_students : ℝ := 50

-- Main theorem statement
theorem number_of_B_students (x : ℝ) (h1 : prob_A x + x + prob_C x + prob_D x = total_students) :
  x = 14 :=
  by
-- Proof skipped
  sorry

end number_of_B_students_l171_171091


namespace cost_of_double_room_l171_171558

theorem cost_of_double_room (total_rooms : ℕ) (cost_single_room : ℕ) (total_revenue : ℕ) 
  (double_rooms_booked : ℕ) (single_rooms_booked := total_rooms - double_rooms_booked) 
  (total_single_revenue := single_rooms_booked * cost_single_room) : 
  total_rooms = 260 → cost_single_room = 35 → total_revenue = 14000 → double_rooms_booked = 196 → 
  196 * 60 + 64 * 35 = total_revenue :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_double_room_l171_171558


namespace solve_system_l171_171263

theorem solve_system :
  ∃! (x y : ℝ), (2 * x + y + 8 ≤ 0) ∧ (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) ∧ (x = -3 ∧ y = -2) := 
  by
  sorry

end solve_system_l171_171263


namespace a_finishes_job_in_60_days_l171_171298

theorem a_finishes_job_in_60_days (A B : ℝ)
  (h1 : A + B = 1 / 30)
  (h2 : 20 * (A + B) = 2 / 3)
  (h3 : 20 * A = 1 / 3) :
  1 / A = 60 :=
by sorry

end a_finishes_job_in_60_days_l171_171298


namespace percentage_neither_l171_171677

theorem percentage_neither (total_teachers high_blood_pressure heart_trouble both_conditions : ℕ)
  (h1 : total_teachers = 150)
  (h2 : high_blood_pressure = 90)
  (h3 : heart_trouble = 60)
  (h4 : both_conditions = 30) :
  100 * (total_teachers - (high_blood_pressure + heart_trouble - both_conditions)) / total_teachers = 20 :=
by
  sorry

end percentage_neither_l171_171677


namespace part_I_part_II_l171_171302

noncomputable def f (x : ℝ) (m : ℝ) := m - |x - 2|

theorem part_I (m : ℝ) : (∀ x, f (x + 1) m >= 0 → 0 <= x ∧ x <= 2) ↔ m = 1 := by
  sorry

theorem part_II (a b c : ℝ) (m : ℝ) : (1 / a + 1 / (2 * b) + 1 / (3 * c) = m) → (m = 1) → (a + 2 * b + 3 * c >= 9) := by
  sorry

end part_I_part_II_l171_171302


namespace LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l171_171910

noncomputable section

-- Problem 1: Prove length ratios for simultaneous ignition
def LengthRatioSimultaneous (t : ℝ) : Prop :=
  let LA := 1 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosSimultaneous (t : ℝ) : LengthRatioSimultaneous t := sorry

-- Problem 2: Prove length ratios when one candle is lit 30 minutes earlier
def LengthRatioNonSimultaneous (t : ℝ) : Prop :=
  let LA := 5 / 6 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosNonSimultaneous (t : ℝ) : LengthRatioNonSimultaneous t := sorry

end LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l171_171910


namespace find_a_l171_171429

theorem find_a (a : ℂ) (h : a / (1 - I) = (1 + I) / I) : a = -2 * I := 
by
  sorry

end find_a_l171_171429


namespace fill_tank_in_6_hours_l171_171198

theorem fill_tank_in_6_hours (A B : ℝ) (hA : A = 1 / 10) (hB : B = 1 / 15) : (1 / (A + B)) = 6 :=
by 
  sorry

end fill_tank_in_6_hours_l171_171198


namespace prism_volume_l171_171813

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end prism_volume_l171_171813


namespace flower_cost_l171_171915

theorem flower_cost (F : ℕ) (h1 : F + (F + 20) + (F - 2) = 45) : F = 9 :=
by
  sorry

end flower_cost_l171_171915


namespace divisibility_criterion_l171_171313

theorem divisibility_criterion (x y : ℕ) (h_two_digit : 10 ≤ x ∧ x < 100) :
  (1207 % x = 0) ↔ (x = 10 * (x / 10) + (x % 10) ∧ (x / 10)^3 + (x % 10)^3 = 344) :=
by
  sorry

end divisibility_criterion_l171_171313


namespace tom_batteries_used_total_l171_171874

def batteries_used_in_flashlights : Nat := 2 * 3
def batteries_used_in_toys : Nat := 4 * 5
def batteries_used_in_controllers : Nat := 2 * 6
def total_batteries_used : Nat := batteries_used_in_flashlights + batteries_used_in_toys + batteries_used_in_controllers

theorem tom_batteries_used_total : total_batteries_used = 38 :=
by
  sorry

end tom_batteries_used_total_l171_171874


namespace sum_of_digits_82_l171_171175

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_82 : sum_of_digits 82 = 10 := by
  sorry

end sum_of_digits_82_l171_171175


namespace currency_exchange_rate_l171_171484

theorem currency_exchange_rate (b g x : ℕ) (h1 : 1 * b * g = b * g) (h2 : 1 = 1) :
  (b + g) ^ 2 + 1 = b * g * x → x = 5 :=
sorry

end currency_exchange_rate_l171_171484


namespace february_five_sundays_in_twenty_first_century_l171_171999

/-- 
  Define a function to check if a year is a leap year
-/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- 
  Define the specific condition for the problem: 
  Given a year, whether February 1st for that year is a Sunday
-/
def february_first_is_sunday (year : ℕ) : Prop :=
  -- This is a placeholder logic. In real applications, you would
  -- calculate the exact weekday of February 1st for the provided year.
  sorry

/-- 
  The list of years in the 21st century where February has 5 Sundays is 
  exactly {2004, 2032, 2060, and 2088}.
-/
theorem february_five_sundays_in_twenty_first_century :
  {year : ℕ | is_leap_year year ∧ february_first_is_sunday year ∧ (2001 ≤ year ∧ year ≤ 2100)} =
  {2004, 2032, 2060, 2088} := sorry

end february_five_sundays_in_twenty_first_century_l171_171999


namespace chocolate_bar_weight_l171_171335

theorem chocolate_bar_weight :
  let square_weight := 6
  let triangles_count := 16
  let squares_count := 32
  let triangle_weight := square_weight / 2
  let total_square_weight := squares_count * square_weight
  let total_triangles_weight := triangles_count * triangle_weight
  total_square_weight + total_triangles_weight = 240 := 
by
  sorry

end chocolate_bar_weight_l171_171335


namespace tangent_line_at_slope_two_l171_171975

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end tangent_line_at_slope_two_l171_171975


namespace minimal_primes_ensuring_first_player_win_l171_171216

-- Define primes less than or equal to 100
def primes_le_100 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Define function to get the last digit of a number
def last_digit (n : Nat) : Nat := n % 10

-- Define function to get the first digit of a number
def first_digit (n : Nat) : Nat :=
  let rec first_digit_aux (m : Nat) :=
    if m < 10 then m else first_digit_aux (m / 10)
  first_digit_aux n

-- Define a condition that checks if a prime number follows the game rule
def follows_rule (a b : Nat) : Bool :=
  last_digit a = first_digit b

theorem minimal_primes_ensuring_first_player_win :
  ∃ (p1 p2 p3 : Nat),
  p1 ∈ primes_le_100 ∧
  p2 ∈ primes_le_100 ∧
  p3 ∈ primes_le_100 ∧
  follows_rule p1 p2 ∧
  follows_rule p2 p3 ∧
  p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
sorry

end minimal_primes_ensuring_first_player_win_l171_171216


namespace zero_lies_in_interval_l171_171581

def f (x : ℝ) : ℝ := -|x - 5| + 2 * x - 1

theorem zero_lies_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 2 := 
sorry

end zero_lies_in_interval_l171_171581


namespace complement_of_beta_l171_171631

theorem complement_of_beta (α β : ℝ) (h₀ : α + β = 180) (h₁ : α > β) : 
  90 - β = 1/2 * (α - β) :=
by
  sorry

end complement_of_beta_l171_171631


namespace arithmetic_geometric_mean_l171_171998

theorem arithmetic_geometric_mean (a b m : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a + b) / 2 = m * Real.sqrt (a * b)) :
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) :=
by
  sorry

end arithmetic_geometric_mean_l171_171998


namespace net_effect_on_sale_l171_171384

variable (P S : ℝ) (orig_revenue : ℝ := P * S) (new_revenue : ℝ := 0.7 * P * 1.8 * S)

theorem net_effect_on_sale : new_revenue = orig_revenue * 1.26 := by
  sorry

end net_effect_on_sale_l171_171384


namespace apple_tree_bears_fruit_in_7_years_l171_171213

def age_planted : ℕ := 4
def age_eats : ℕ := 11
def time_to_bear_fruit : ℕ := age_eats - age_planted

theorem apple_tree_bears_fruit_in_7_years :
  time_to_bear_fruit = 7 :=
by
  sorry

end apple_tree_bears_fruit_in_7_years_l171_171213


namespace JameMade112kProfit_l171_171720

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l171_171720


namespace chef_michel_total_pies_l171_171191

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l171_171191


namespace blue_paint_cans_needed_l171_171600

theorem blue_paint_cans_needed (ratio_bg : ℤ × ℤ) (total_cans : ℤ) (r : ratio_bg = (4, 3)) (t : total_cans = 42) :
  let ratio_bw : ℚ := 4 / (4 + 3) 
  let blue_cans : ℚ := ratio_bw * total_cans 
  blue_cans = 24 :=
by
  sorry

end blue_paint_cans_needed_l171_171600


namespace find_alpha_l171_171637

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = - (Real.sqrt 3 / 2)) (h_range : 0 < α ∧ α < Real.pi) : α = 5 * Real.pi / 6 :=
sorry

end find_alpha_l171_171637


namespace largest_angle_of_triangle_l171_171705

theorem largest_angle_of_triangle (A B C : ℝ) :
  A + B + C = 180 ∧ A + B = 126 ∧ abs (A - B) = 45 → max A (max B C) = 85.5 :=
by sorry

end largest_angle_of_triangle_l171_171705


namespace diamonds_in_G_15_l171_171415

/-- Define the number of diamonds in G_n -/
def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1
  else 3 * n ^ 2 - 3 * n + 1

/-- Theorem to prove the number of diamonds in G_15 is 631 -/
theorem diamonds_in_G_15 : diamonds_in_G 15 = 631 :=
by
  -- The proof is omitted
  sorry

end diamonds_in_G_15_l171_171415


namespace b_should_pay_348_48_l171_171991

/-- Definitions for the given conditions --/

def horses_a : ℕ := 12
def months_a : ℕ := 8

def horses_b : ℕ := 16
def months_b : ℕ := 9

def horses_c : ℕ := 18
def months_c : ℕ := 6

def total_rent : ℕ := 841

/-- Calculate the individual and total contributions in horse-months --/

def contribution_a : ℕ := horses_a * months_a
def contribution_b : ℕ := horses_b * months_b
def contribution_c : ℕ := horses_c * months_c

def total_contributions : ℕ := contribution_a + contribution_b + contribution_c

/-- Calculate cost per horse-month and b's share of the rent --/

def cost_per_horse_month : ℚ := total_rent / total_contributions
def b_share : ℚ := contribution_b * cost_per_horse_month

/-- Lean statement to check b's share --/

theorem b_should_pay_348_48 : b_share = 348.48 := by
  sorry

end b_should_pay_348_48_l171_171991


namespace union_M_N_eq_N_l171_171994

def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

theorem union_M_N_eq_N : M ∪ N = N := by
  sorry

end union_M_N_eq_N_l171_171994


namespace no_strictly_greater_polynomials_l171_171554

noncomputable def transformation (P : Polynomial ℝ) (k : ℕ) (a : ℝ) : Polynomial ℝ := 
  P + Polynomial.monomial k (2 * a) - Polynomial.monomial (k + 1) a

theorem no_strictly_greater_polynomials (P Q : Polynomial ℝ) 
  (H1 : ∃ (n : ℕ) (a : ℝ), Q = transformation P n a)
  (H2 : ∃ (n : ℕ) (a : ℝ), P = transformation Q n a) : 
  ∃ x : ℝ, P.eval x = Q.eval x :=
sorry

end no_strictly_greater_polynomials_l171_171554


namespace simplify_polynomial_subtraction_l171_171812

variable (x : ℝ)

def P1 : ℝ := 2*x^6 + x^5 + 3*x^4 + x^3 + 5
def P2 : ℝ := x^6 + 2*x^5 + x^4 - x^3 + 7
def P3 : ℝ := x^6 - x^5 + 2*x^4 + 2*x^3 - 2

theorem simplify_polynomial_subtraction : (P1 x - P2 x) = P3 x :=
by
  sorry

end simplify_polynomial_subtraction_l171_171812


namespace gasoline_price_april_l171_171756

theorem gasoline_price_april (P₀ : ℝ) (P₁ P₂ P₃ P₄ : ℝ) (x : ℝ)
  (h₁ : P₁ = P₀ * 1.20)  -- Price after January's increase
  (h₂ : P₂ = P₁ * 0.80)  -- Price after February's decrease
  (h₃ : P₃ = P₂ * 1.25)  -- Price after March's increase
  (h₄ : P₄ = P₃ * (1 - x / 100))  -- Price after April's decrease
  (h₅ : P₄ = P₀)  -- Price at the end of April equals the initial price
  : x = 17 := 
by
  sorry

end gasoline_price_april_l171_171756


namespace triangle_inequality_l171_171002
open Real

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_abc : a + b > c) (h_acb : a + c > b) (h_bca : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l171_171002


namespace find_number_l171_171438

theorem find_number :
  (∃ x : ℝ, x * (3 + Real.sqrt 5) = 1) ∧ (x = (3 - Real.sqrt 5) / 4) :=
sorry

end find_number_l171_171438


namespace probability_no_physics_and_chemistry_l171_171744

-- Define the probabilities for the conditions
def P_physics : ℚ := 5 / 8
def P_no_physics : ℚ := 1 - P_physics
def P_chemistry_given_no_physics : ℚ := 2 / 3

-- Define the theorem we want to prove
theorem probability_no_physics_and_chemistry :
  P_no_physics * P_chemistry_given_no_physics = 1 / 4 :=
by sorry

end probability_no_physics_and_chemistry_l171_171744


namespace find_t_l171_171451

def utility (hours_math hours_reading hours_painting : ℕ) : ℕ :=
  hours_math^2 + hours_reading * hours_painting

def utility_wednesday (t : ℕ) : ℕ :=
  utility 4 t (12 - t)

def utility_thursday (t : ℕ) : ℕ :=
  utility 3 (t + 1) (11 - t)

theorem find_t (t : ℕ) (h : utility_wednesday t = utility_thursday t) : t = 2 :=
by
  sorry

end find_t_l171_171451


namespace christopher_age_l171_171218

variables (C G : ℕ)

theorem christopher_age :
  (C = 2 * G) ∧ (C - 9 = 5 * (G - 9)) → C = 24 :=
by
  intro h
  sorry

end christopher_age_l171_171218


namespace shaded_region_area_l171_171459

theorem shaded_region_area (r : ℝ) (π : ℝ) (h1 : r = 5) : 
  4 * ((1/2 * π * r * r) - (1/2 * r * r)) = 50 * π - 50 :=
by 
  sorry

end shaded_region_area_l171_171459


namespace complement_intersection_l171_171449

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) (hM : M = {1, 3, 5, 7}) (hN : N = {2, 5, 8}) :
  (U \ M) ∩ N = {2, 8} :=
by
  sorry

end complement_intersection_l171_171449


namespace soccer_field_solution_l171_171617

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l171_171617


namespace dog_weight_ratio_l171_171502

theorem dog_weight_ratio :
  ∀ (brown black white grey : ℕ),
    brown = 4 →
    black = brown + 1 →
    grey = black - 2 →
    (brown + black + white + grey) / 4 = 5 →
    white / brown = 2 :=
by
  intros brown black white grey h_brown h_black h_grey h_avg
  sorry

end dog_weight_ratio_l171_171502


namespace find_digit_for_multiple_of_3_l171_171506

theorem find_digit_for_multiple_of_3 (d : ℕ) (h : d < 10) : 
  (56780 + d) % 3 = 0 ↔ d = 1 :=
by sorry

end find_digit_for_multiple_of_3_l171_171506


namespace symmetric_point_l171_171473

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1: P = (2, 1)) (h2 : x - y + 1 = 0) :
  (b - 1) = -(a - 2) ∧ (a + 2) / 2 - (b + 1) / 2 + 1 = 0 → (a, b) = (0, 3) := 
sorry

end symmetric_point_l171_171473


namespace parabola_line_slope_l171_171678

theorem parabola_line_slope (y1 y2 x1 x2 : ℝ) (h1 : y1 ^ 2 = 6 * x1) (h2 : y2 ^ 2 = 6 * x2) 
    (midpoint_condition : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  (y1 - y2) / (x1 - x2) = 3 / 2 :=
by
  -- here will be the actual proof using the given hypothesis
  sorry

end parabola_line_slope_l171_171678


namespace find_m_plus_n_l171_171517

theorem find_m_plus_n (PQ QR RP : ℕ) (x y : ℕ) 
  (h1 : PQ = 26) 
  (h2 : QR = 29) 
  (h3 : RP = 25) 
  (h4 : PQ = x + y) 
  (h5 : QR = x + (QR - x))
  (h6 : RP = x + (RP - x)) : 
  30 = 29 + 1 :=
by
  -- assumptions already provided in problem statement
  sorry

end find_m_plus_n_l171_171517


namespace ab_equals_4_l171_171069

theorem ab_equals_4 (a b : ℝ) (h_pos : a > 0 ∧ b > 0)
  (h_area : (1/2) * (12 / a) * (8 / b) = 12) : a * b = 4 :=
by
  sorry

end ab_equals_4_l171_171069


namespace total_height_increase_l171_171258

def height_increase_per_decade : ℕ := 90
def decades_in_two_centuries : ℕ := (2 * 100) / 10

theorem total_height_increase :
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end total_height_increase_l171_171258


namespace exists_x_y_mod_p_l171_171642

theorem exists_x_y_mod_p (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : ∃ x y : ℤ, (x^2 + y^3) % p = a % p :=
by
  sorry

end exists_x_y_mod_p_l171_171642


namespace fx_greater_than_2_l171_171150

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem fx_greater_than_2 :
  ∀ x : ℝ, x > 0 → f x > 2 :=
by {
  sorry
}

end fx_greater_than_2_l171_171150


namespace compare_expressions_l171_171873

theorem compare_expressions (x y : ℝ) (h1: x * y > 0) (h2: x ≠ y) : 
  x^4 + 6 * x^2 * y^2 + y^4 > 4 * x * y * (x^2 + y^2) :=
by
  sorry

end compare_expressions_l171_171873


namespace base_seven_to_ten_l171_171882

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l171_171882


namespace brianna_books_l171_171416

theorem brianna_books :
  ∀ (books_per_month : ℕ) (given_books : ℕ) (bought_books : ℕ) (borrowed_books : ℕ) (total_books_needed : ℕ),
    (books_per_month = 2) →
    (given_books = 6) →
    (bought_books = 8) →
    (borrowed_books = bought_books - 2) →
    (total_books_needed = 12 * books_per_month) →
    (total_books_needed - (given_books + bought_books + borrowed_books)) = 4 :=
by
  intros
  sorry

end brianna_books_l171_171416


namespace average_output_assembly_line_l171_171368

theorem average_output_assembly_line
  (initial_rate : ℕ) (initial_cogs : ℕ) 
  (increased_rate : ℕ) (increased_cogs : ℕ)
  (h1 : initial_rate = 15)
  (h2 : initial_cogs = 60)
  (h3 : increased_rate = 60)
  (h4 : increased_cogs = 60) :
  (initial_cogs + increased_cogs) / (initial_cogs / initial_rate + increased_cogs / increased_rate) = 24 := 
by sorry

end average_output_assembly_line_l171_171368


namespace triangle_right_angle_l171_171628

theorem triangle_right_angle
  (a b m : ℝ)
  (h1 : 0 < b)
  (h2 : b < m)
  (h3 : a^2 + b^2 = m^2) :
  a^2 + b^2 = m^2 :=
by sorry

end triangle_right_angle_l171_171628


namespace change_combinations_l171_171067

def isValidCombination (nickels dimes quarters : ℕ) : Prop :=
  nickels * 5 + dimes * 10 + quarters * 25 = 50 ∧ quarters ≤ 1

theorem change_combinations : {n // ∃ (combinations : ℕ) (nickels dimes quarters : ℕ), 
  n = combinations ∧ isValidCombination nickels dimes quarters ∧ 
  ((nickels, dimes, quarters) = (10, 0, 0) ∨
   (nickels, dimes, quarters) = (8, 1, 0) ∨
   (nickels, dimes, quarters) = (6, 2, 0) ∨
   (nickels, dimes, quarters) = (4, 3, 0) ∨
   (nickels, dimes, quarters) = (2, 4, 0) ∨
   (nickels, dimes, quarters) = (0, 5, 0) ∨
   (nickels, dimes, quarters) = (5, 0, 1) ∨
   (nickels, dimes, quarters) = (3, 1, 1) ∨
   (nickels, dimes, quarters) = (1, 2, 1))}
  :=
  ⟨9, sorry⟩

end change_combinations_l171_171067


namespace probability_exactly_one_instrument_l171_171369

-- Definitions of the conditions
def total_people : ℕ := 800
def frac_one_instrument : ℚ := 1 / 5
def people_two_or_more_instruments : ℕ := 64

-- Statement of the problem
theorem probability_exactly_one_instrument :
  let people_at_least_one_instrument := frac_one_instrument * total_people
  let people_exactly_one_instrument := people_at_least_one_instrument - people_two_or_more_instruments
  let probability := people_exactly_one_instrument / total_people
  probability = 3 / 25 :=
by
  -- Definitions
  let people_at_least_one_instrument : ℚ := frac_one_instrument * total_people
  let people_exactly_one_instrument : ℚ := people_at_least_one_instrument - people_two_or_more_instruments
  let probability : ℚ := people_exactly_one_instrument / total_people
  
  -- Sorry statement to skip the proof
  exact sorry

end probability_exactly_one_instrument_l171_171369


namespace square_perimeter_l171_171453

theorem square_perimeter (x : ℝ) (h : x * x + x * x = (2 * Real.sqrt 2) * (2 * Real.sqrt 2)) :
    4 * x = 8 :=
by
  sorry

end square_perimeter_l171_171453


namespace solve_inequality_l171_171971

theorem solve_inequality : 
  {x : ℝ | -x^2 - 2*x + 3 ≤ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end solve_inequality_l171_171971


namespace simplify_expression_l171_171168

theorem simplify_expression (α : ℝ) (h_sin_ne_zero : Real.sin α ≠ 0) :
    (1 / Real.sin α + 1 / Real.tan α) * (1 - Real.cos α) = Real.sin α := 
sorry

end simplify_expression_l171_171168


namespace evaluate_f_l171_171048

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem evaluate_f : f (f (f (-1))) = Real.pi + 1 :=
by
  -- Proof goes here
  sorry

end evaluate_f_l171_171048


namespace solve_dividend_and_divisor_l171_171307

-- Definitions for base, digits, and mathematical relationships
def base := 5
def P := 1
def Q := 2
def R := 3
def S := 4
def T := 0
def Dividend := 1 * base^6 + 2 * base^5 + 3 * base^4 + 4 * base^3 + 3 * base^2 + 2 * base^1 + 1 * base^0
def Divisor := 2 * base^2 + 3 * base^1 + 2 * base^0

-- The conditions given in the math problem
axiom condition_1 : Q + R = base
axiom condition_2 : P + 1 = Q
axiom condition_3 : Q + P = R
axiom condition_4 : S = 2 * Q
axiom condition_5 : Q^2 = S
axiom condition_6 : Dividend = 24336
axiom condition_7 : Divisor = 67

-- The goal
theorem solve_dividend_and_divisor : Dividend = 24336 ∧ Divisor = 67 :=
by {
  sorry
}

end solve_dividend_and_divisor_l171_171307


namespace probability_phone_not_answered_l171_171920

noncomputable def P_first_ring : ℝ := 0.1
noncomputable def P_second_ring : ℝ := 0.3
noncomputable def P_third_ring : ℝ := 0.4
noncomputable def P_fourth_ring : ℝ := 0.1

theorem probability_phone_not_answered : 
  1 - P_first_ring - P_second_ring - P_third_ring - P_fourth_ring = 0.1 := 
by
  sorry

end probability_phone_not_answered_l171_171920


namespace wendy_furniture_time_l171_171633

theorem wendy_furniture_time (chairs tables minutes_per_piece : ℕ) 
    (h_chairs : chairs = 4) 
    (h_tables : tables = 4) 
    (h_minutes_per_piece : minutes_per_piece = 6) : 
    chairs + tables * minutes_per_piece = 48 := 
by 
    sorry

end wendy_furniture_time_l171_171633


namespace tan_double_angle_l171_171435

theorem tan_double_angle (α : ℝ) (h1 : Real.sin (5 * Real.pi / 6) = 1 / 2)
  (h2 : Real.cos (5 * Real.pi / 6) = -Real.sqrt 3 / 2) : 
  Real.tan (2 * α) = Real.sqrt 3 := 
sorry

end tan_double_angle_l171_171435


namespace total_pieces_l171_171281

-- Define the given conditions
def pieces_eaten_per_person : ℕ := 4
def num_people : ℕ := 3

-- Theorem stating the result
theorem total_pieces (h : num_people > 0) : (num_people * pieces_eaten_per_person) = 12 := 
by
  sorry

end total_pieces_l171_171281


namespace arithmetic_sequence_theorem_l171_171584

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h_a1_pos : a 1 > 0)
  (h_condition : -1 < a 7 / a 6 ∧ a 7 / a 6 < 0) :
  (∃ d, d < 0) ∧ (∀ n, S n > 0 → n ≤ 12) :=
sorry

end arithmetic_sequence_theorem_l171_171584


namespace linda_savings_l171_171770

theorem linda_savings (S : ℝ) (h : (1 / 2) * S = 300) : S = 600 :=
sorry

end linda_savings_l171_171770


namespace add_in_base_7_l171_171537

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end add_in_base_7_l171_171537


namespace five_digit_numbers_with_4_or_5_l171_171752

theorem five_digit_numbers_with_4_or_5 : 
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  total_five_digit - without_4_or_5 = 61328 :=
by
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  have h : total_five_digit - without_4_or_5 = 61328 := by sorry
  exact h

end five_digit_numbers_with_4_or_5_l171_171752


namespace tangent_line_at_1_f_geq_x_minus_1_min_value_a_l171_171817

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- 1. Proof that the equation of the tangent line at the point (1, f(1)) is y = x - 1
theorem tangent_line_at_1 :
  ∃ k b, (k = 1 ∧ b = -1 ∧ (∀ x, (f x - k * x - b) = 0)) :=
sorry

-- 2. Proof that f(x) ≥ x - 1 for all x in (0, +∞)
theorem f_geq_x_minus_1 :
  ∀ x, 0 < x → f x ≥ x - 1 :=
sorry

-- 3. Proof that the minimum value of a such that f(x) ≥ ax² + 2/a for all x in (0, +∞) is -e³
theorem min_value_a :
  ∃ a, (∀ x, 0 < x → f x ≥ a * x^2 + 2 / a) ∧ (a = -Real.exp 3) :=
sorry

end tangent_line_at_1_f_geq_x_minus_1_min_value_a_l171_171817


namespace salary_reduction_l171_171106

theorem salary_reduction (S : ℝ) (R : ℝ) :
  ((S - (R / 100 * S)) * 1.25 = S) → (R = 20) :=
by
  sorry

end salary_reduction_l171_171106


namespace early_time_l171_171442

noncomputable def speed1 : ℝ := 5 -- km/hr
noncomputable def timeLate : ℝ := 5 / 60 -- convert minutes to hours
noncomputable def speed2 : ℝ := 10 -- km/hr
noncomputable def distance : ℝ := 2.5 -- km

theorem early_time (speed1 speed2 distance : ℝ) (timeLate : ℝ) :
  (distance / speed1 - timeLate) * 60 - (distance / speed2) * 60 = 10 :=
by
  sorry

end early_time_l171_171442


namespace loan_amount_l171_171640

theorem loan_amount (R T SI : ℕ) (hR : R = 7) (hT : T = 7) (hSI : SI = 735) : 
  ∃ P : ℕ, P = 1500 := 
by 
  sorry

end loan_amount_l171_171640


namespace circle_complete_the_square_l171_171490

/-- Given the equation x^2 - 6x + y^2 - 10y + 18 = 0, show that it can be transformed to  
    (x - 3)^2 + (y - 5)^2 = 4^2 -/
theorem circle_complete_the_square :
  ∀ x y : ℝ, x^2 - 6 * x + y^2 - 10 * y + 18 = 0 ↔ (x - 3)^2 + (y - 5)^2 = 4^2 :=
by
  sorry

end circle_complete_the_square_l171_171490


namespace total_students_at_gathering_l171_171358

theorem total_students_at_gathering (x : ℕ) 
  (h1 : ∃ x : ℕ, 0 < x)
  (h2 : (x + 6) / (2 * x + 6) = 2 / 3) : 
  (2 * x + 6) = 18 := 
  sorry

end total_students_at_gathering_l171_171358


namespace housewife_spent_on_oil_l171_171661

-- Define the conditions
variables (P A : ℝ)
variables (h_price_reduced : 0.7 * P = 70)
variables (h_more_oil : A / 70 = A / P + 3)

-- Define the theorem to be proven
theorem housewife_spent_on_oil : A = 700 :=
by
  sorry

end housewife_spent_on_oil_l171_171661


namespace quadratic_roots_range_l171_171450

theorem quadratic_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - (2 * m + 1) * x + m^2 = 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - (2 * m + 1) * y + m^2 = 0)) ↔ m > -1 / 4 :=
by sorry

end quadratic_roots_range_l171_171450


namespace largest_x_by_equation_l171_171330

theorem largest_x_by_equation : ∃ x : ℚ, 
  (∀ y : ℚ, 6 * (12 * y^2 + 12 * y + 11) = y * (12 * y - 44) → y ≤ x) 
  ∧ 6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44) 
  ∧ x = -1 := 
sorry

end largest_x_by_equation_l171_171330


namespace infinite_sequence_domain_l171_171525

def seq_domain (f : ℕ → ℕ) : Set ℕ := {n | 0 < n}

theorem infinite_sequence_domain (f : ℕ → ℕ) (a_n : ℕ → ℕ)
   (h : ∀ (n : ℕ), a_n n = f n) : 
   seq_domain f = {n | 0 < n} :=
sorry

end infinite_sequence_domain_l171_171525


namespace geom_progr_sum_eq_l171_171280

variable (a b q : ℝ) (n p : ℕ)

theorem geom_progr_sum_eq (h : a * (1 - q ^ (n * p)) / (1 - q) = b * (1 - q ^ (n * p)) / (1 - q ^ p)) :
  b = a * (1 - q ^ p) / (1 - q) :=
by
  sorry

end geom_progr_sum_eq_l171_171280


namespace Ryan_hours_learning_Spanish_is_4_l171_171152

-- Definitions based on conditions
def hoursLearningChinese : ℕ := 5
def hoursLearningSpanish := ∃ x : ℕ, hoursLearningChinese = x + 1

-- Proof Statement
theorem Ryan_hours_learning_Spanish_is_4 : ∃ x : ℕ, hoursLearningSpanish ∧ x = 4 :=
by
  sorry

end Ryan_hours_learning_Spanish_is_4_l171_171152


namespace saved_percentage_this_year_l171_171145

variable (S : ℝ) -- Annual salary last year

-- Conditions
def saved_last_year := 0.06 * S
def salary_this_year := 1.20 * S
def saved_this_year := saved_last_year

-- The goal is to prove that the percentage saved this year is 5%
theorem saved_percentage_this_year :
  (saved_this_year / salary_this_year) * 100 = 5 :=
by sorry

end saved_percentage_this_year_l171_171145


namespace find_d_in_polynomial_l171_171618

theorem find_d_in_polynomial 
  (a b c d : ℤ) 
  (x1 x2 x3 x4 : ℤ)
  (roots_neg : x1 < 0 ∧ x2 < 0 ∧ x3 < 0 ∧ x4 < 0)
  (h_poly : ∀ x, 
    (x + x1) * (x + x2) * (x + x3) * (x + x4) = 
    x^4 + a * x^3 + b * x^2 + c * x + d)
  (h_sum_eq : a + b + c + d = 2009) :
  d = (x1 * x2 * x3 * x4) :=
by
  sorry

end find_d_in_polynomial_l171_171618


namespace solution_set_of_quadratic_inequality_l171_171857

theorem solution_set_of_quadratic_inequality (a : ℝ) (x : ℝ) :
  (∀ x, 0 < x - 0.5 ∧ x < 2 → ax^2 + 5 * x - 2 > 0) ∧ a = -2 →
  (∀ x, -3 < x ∧ x < 0.5 → a * x^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end solution_set_of_quadratic_inequality_l171_171857


namespace matching_pair_probability_l171_171092

theorem matching_pair_probability :
  let gray_socks := 12
  let white_socks := 10
  let black_socks := 6
  let total_socks := gray_socks + white_socks + black_socks
  let total_ways := total_socks.choose 2
  let gray_matching := gray_socks.choose 2
  let white_matching := white_socks.choose 2
  let black_matching := black_socks.choose 2
  let matching_ways := gray_matching + white_matching + black_matching
  let probability := matching_ways / total_ways
  probability = 1 / 3 :=
by sorry

end matching_pair_probability_l171_171092


namespace fraction_subtraction_l171_171989

theorem fraction_subtraction : (5 / 6) - (1 / 12) = (3 / 4) := 
by 
  sorry

end fraction_subtraction_l171_171989


namespace simplify_expression_l171_171405

variable (x : ℝ)

theorem simplify_expression :
  3 * x^3 + 4 * x + 5 * x^2 + 2 - (7 - 3 * x^3 - 4 * x - 5 * x^2) =
  6 * x^3 + 10 * x^2 + 8 * x - 5 :=
by
  sorry

end simplify_expression_l171_171405


namespace valid_twenty_letter_words_l171_171066

noncomputable def number_of_valid_words : ℕ := sorry

theorem valid_twenty_letter_words :
  number_of_valid_words = 3 * 2^18 := sorry

end valid_twenty_letter_words_l171_171066


namespace limit_example_l171_171853

open Real

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ → abs ((2 * x^2 + 6 * x - 8) / (x + 4) + 10) < ε) :=
by
  sorry

end limit_example_l171_171853


namespace total_stickers_purchased_l171_171680

-- Definitions for the number of sheets and stickers per sheet for each folder
def num_sheets_per_folder := 10
def stickers_per_sheet_red := 3
def stickers_per_sheet_green := 2
def stickers_per_sheet_blue := 1

-- Theorem stating that the total number of stickers is 60
theorem total_stickers_purchased : 
  num_sheets_per_folder * (stickers_per_sheet_red + stickers_per_sheet_green + stickers_per_sheet_blue) = 60 := 
  by
  -- Skipping the proof
  sorry

end total_stickers_purchased_l171_171680


namespace length_of_AB_l171_171870

-- Conditions:
-- The radius of the inscribed circle is 6 cm.
-- The triangle is a right triangle with a 60 degree angle at one vertex.
-- Question: Prove that the length of AB is 12 + 12√3 cm.

theorem length_of_AB (r : ℝ) (angle : ℝ) (h_radius : r = 6) (h_angle : angle = 60) :
  ∃ (AB : ℝ), AB = 12 + 12 * Real.sqrt 3 :=
by
  sorry

end length_of_AB_l171_171870


namespace num_male_students_selected_l171_171778

def total_students := 220
def male_students := 60
def selected_female_students := 32

def selected_male_students (total_students male_students selected_female_students : Nat) : Nat :=
  (selected_female_students * male_students) / (total_students - male_students)

theorem num_male_students_selected : selected_male_students total_students male_students selected_female_students = 12 := by
  unfold selected_male_students
  sorry

end num_male_students_selected_l171_171778


namespace smallest_angle_in_right_triangle_l171_171121

noncomputable def is_consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ r, Nat.Prime r → p < r → r < q → False

theorem smallest_angle_in_right_triangle : ∃ p : ℕ, ∃ q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p + q = 90 ∧ is_consecutive_primes p q ∧ p = 43 :=
by
  sorry

end smallest_angle_in_right_triangle_l171_171121


namespace problem_statement_l171_171895

def approx_digit_place (num : ℕ) : ℕ :=
if num = 3020000 then 0 else sorry

theorem problem_statement :
  approx_digit_place (3 * 10^6 + 2 * 10^4) = 0 :=
by
  sorry

end problem_statement_l171_171895


namespace trig_identity_l171_171202

theorem trig_identity : 
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (72 * Real.pi / 180)) * (Real.sin (84 * Real.pi / 180)) = 1 / 32 :=
by sorry

end trig_identity_l171_171202


namespace find_a_l171_171047

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

def monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

def valid_interval (a : ℝ) : Prop :=
  monotonically_decreasing f (Set.Icc (a-1) (a+1))

theorem find_a :
  {a : ℝ | valid_interval a} = {a : ℝ | 1 < a ∧ a ≤ 2} :=
by
  sorry

end find_a_l171_171047


namespace trig_expression_simplification_l171_171589

theorem trig_expression_simplification :
  ∃ a b : ℕ, 
  0 < b ∧ b < 90 ∧ 
  (1000 * Real.sin (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) = ↑a * Real.sin (b * Real.pi / 180)) ∧ 
  (100 * a + b = 12560) :=
sorry

end trig_expression_simplification_l171_171589


namespace polynomial_divisibility_l171_171724

theorem polynomial_divisibility (A B : ℝ)
  (h: ∀ (x : ℂ), x^2 + x + 1 = 0 → x^104 + A * x^3 + B * x = 0) :
  A + B = 0 :=
by
  sorry

end polynomial_divisibility_l171_171724


namespace repeating_block_length_7_div_13_l171_171603

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l171_171603


namespace avg_weight_increase_l171_171668

theorem avg_weight_increase (A : ℝ) (X : ℝ) (hp1 : 8 * A - 65 + 105 = 8 * A + 40)
  (hp2 : 8 * (A + X) = 8 * A + 40) : X = 5 := 
by sorry

end avg_weight_increase_l171_171668


namespace linear_function_change_l171_171848

-- Define a linear function g
variable (g : ℝ → ℝ)

-- Define and assume the conditions
def linear_function (g : ℝ → ℝ) : Prop := ∀ x y, g (x + y) = g x + g y ∧ g (x - y) = g x - g y
def condition_g_at_points : Prop := g 3 - g (-1) = 20

-- Prove that g(10) - g(2) = 40
theorem linear_function_change (g : ℝ → ℝ) 
  (linear_g : linear_function g) 
  (cond_g : condition_g_at_points g) : 
  g 10 - g 2 = 40 :=
sorry

end linear_function_change_l171_171848


namespace number_of_possible_values_l171_171829

-- Define the decimal number s and its representation
def s (e f g h : ℕ) : ℚ := e / 10 + f / 100 + g / 1000 + h / 10000

-- Define the condition that the closest fraction is 2/9
def closest_to_2_9 (s : ℚ) : Prop :=
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 1 / 6)) ∧
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 2 / 11))

-- The main theorem stating the number of possible values for s
theorem number_of_possible_values :
  (∃ e f g h : ℕ, 0 ≤ e ∧ e ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9 ∧ 0 ≤ g ∧ g ≤ 9 ∧ 0 ≤ h ∧ h ≤ 9 ∧
    closest_to_2_9 (s e f g h)) → (∃ n : ℕ, n = 169) :=
by
  sorry

end number_of_possible_values_l171_171829


namespace boat_downstream_travel_time_l171_171236

theorem boat_downstream_travel_time (D : ℝ) (V_b : ℝ) (T_u : ℝ) (V_c : ℝ) (T_d : ℝ) : 
  D = 300 ∧ V_b = 105 ∧ T_u = 5 ∧ (300 = (105 - V_c) * 5) ∧ (300 = (105 + V_c) * T_d) → T_d = 2 :=
by
  sorry

end boat_downstream_travel_time_l171_171236


namespace least_integer_value_x_l171_171859

theorem least_integer_value_x (x : ℤ) : (3 * |2 * (x : ℤ) - 1| + 6 < 24) → x = -2 :=
by
  sorry

end least_integer_value_x_l171_171859


namespace solution_count_l171_171949

noncomputable def equation_has_one_solution : Prop :=
∀ x : ℝ, (x - (8 / (x - 2))) = (4 - (8 / (x - 2))) → x = 4

theorem solution_count : equation_has_one_solution :=
by
  sorry

end solution_count_l171_171949


namespace age_of_B_l171_171553

theorem age_of_B (A B C : ℕ) (h1 : A = 2 * C + 2) (h2 : B = 2 * C) (h3 : A + B + C = 27) : B = 10 :=
by
  sorry

end age_of_B_l171_171553


namespace ratio_water_duck_to_pig_l171_171166

theorem ratio_water_duck_to_pig :
  let gallons_per_minute := 3
  let pumping_minutes := 25
  let total_gallons := gallons_per_minute * pumping_minutes
  let corn_rows := 4
  let plants_per_row := 15
  let gallons_per_corn_plant := 0.5
  let total_corn_plants := corn_rows * plants_per_row
  let total_corn_water := total_corn_plants * gallons_per_corn_plant
  let pig_count := 10
  let gallons_per_pig := 4
  let total_pig_water := pig_count * gallons_per_pig
  let duck_count := 20
  let total_duck_water := total_gallons - total_corn_water - total_pig_water
  let gallons_per_duck := total_duck_water / duck_count
  let ratio := gallons_per_duck / gallons_per_pig
  ratio = 1 / 16 := 
by
  sorry

end ratio_water_duck_to_pig_l171_171166


namespace third_smallest_abc_sum_l171_171006

-- Define the necessary conditions and properties
def isIntegerRoots (a b c : ℕ) : Prop :=
  ∃ r1 r2 r3 r4 : ℤ, 
    a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 - c = 0 ∧ 
    a * r3^2 - b * r3 + c = 0 ∧ a * r4^2 - b * r4 - c = 0

-- State the main theorem
theorem third_smallest_abc_sum : ∃ a b c : ℕ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ isIntegerRoots a b c ∧ 
  (a + b + c = 35 ∧ a = 1 ∧ b = 10 ∧ c = 24) :=
by sorry

end third_smallest_abc_sum_l171_171006


namespace raritet_meets_ferries_l171_171132

theorem raritet_meets_ferries :
  (∀ (n : ℕ), ∃ (ferry_departure : Nat), ferry_departure = n ∧ ferry_departure + 8 = 8) →
  (∀ (m : ℕ), ∃ (raritet_departure : Nat), raritet_departure = m ∧ raritet_departure + 8 = 8) →
  ∃ (total_meetings : Nat), total_meetings = 17 := 
by
  sorry

end raritet_meets_ferries_l171_171132


namespace total_birds_correct_l171_171652

def numPairs : Nat := 3
def birdsPerPair : Nat := 2
def totalBirds : Nat := numPairs * birdsPerPair

theorem total_birds_correct : totalBirds = 6 :=
by
  -- proof goes here
  sorry

end total_birds_correct_l171_171652


namespace largest_possible_d_plus_r_l171_171246

theorem largest_possible_d_plus_r :
  ∃ d r : ℕ, 0 < d ∧ 468 % d = r ∧ 636 % d = r ∧ 867 % d = r ∧ d + r = 27 := by
  sorry

end largest_possible_d_plus_r_l171_171246


namespace complement_of_A_relative_to_I_l171_171176

def I : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

def complement_I_A : Set ℤ := {x ∈ I | x ∉ A}

theorem complement_of_A_relative_to_I :
  complement_I_A = {-2, 2} := by
  sorry

end complement_of_A_relative_to_I_l171_171176


namespace value_of_m_l171_171700

theorem value_of_m (a b m : ℚ) (h1 : 2 * a = m) (h2 : 5 * b = m) (h3 : a + b = 2) : m = 20 / 7 :=
by
  sorry

end value_of_m_l171_171700


namespace sachin_age_is_49_l171_171667

open Nat

-- Let S be Sachin's age and R be Rahul's age
def Sachin_age : ℕ := 49
def Rahul_age (S : ℕ) := S + 14

theorem sachin_age_is_49 (S R : ℕ) (h1 : R = S + 14) (h2 : S * 9 = R * 7) : S = 49 :=
by sorry

end sachin_age_is_49_l171_171667


namespace abc_cubic_sum_identity_l171_171352

theorem abc_cubic_sum_identity (a b c : ℂ) 
  (M : Matrix (Fin 3) (Fin 3) ℂ)
  (h1 : M = fun i j => if i = 0 then (if j = 0 then a else if j = 1 then b else c)
                      else if i = 1 then (if j = 0 then b else if j = 1 then c else a)
                      else (if j = 0 then c else if j = 1 then a else b))
  (h2 : M ^ 3 = 1)
  (h3 : a * b * c = -1) :
  a^3 + b^3 + c^3 = 4 := sorry

end abc_cubic_sum_identity_l171_171352


namespace three_correct_deliveries_probability_l171_171161

theorem three_correct_deliveries_probability (n : ℕ) (h1 : n = 5) :
  (∃ p : ℚ, p = 1/6 ∧ 
   (∃ choose3 : ℕ, choose3 = Nat.choose n 3) ∧ 
   (choose3 * 1/5 * 1/4 * 1/3 = p)) :=
by 
  sorry

end three_correct_deliveries_probability_l171_171161


namespace geometric_sequence_sixth_term_l171_171988

variable (q : ℕ) (a_2 a_6 : ℕ)

-- Given conditions:
axiom h1 : q = 2
axiom h2 : a_2 = 8

-- Prove that a_6 = 128 where a_n = a_2 * q^(n-2)
theorem geometric_sequence_sixth_term : a_6 = a_2 * q^4 → a_6 = 128 :=
by sorry

end geometric_sequence_sixth_term_l171_171988


namespace madeline_refills_l171_171098

theorem madeline_refills :
  let total_water := 100
  let bottle_capacity := 12
  let remaining_to_drink := 16
  let already_drank := total_water - remaining_to_drink
  let initial_refills := already_drank / bottle_capacity
  let refills := initial_refills + 1
  refills = 8 :=
by
  sorry

end madeline_refills_l171_171098


namespace health_risk_probability_l171_171918

theorem health_risk_probability :
  let a := 0.08 * 500
  let b := 0.08 * 500
  let c := 0.08 * 500
  let d := 0.18 * 500
  let e := 0.18 * 500
  let f := 0.18 * 500
  let g := 0.05 * 500
  let h := 500 - (3 * 40 + 3 * 90 + 25)
  let q := 500 - (a + d + e + g)
  let p := 1
  let q := 3
  p + q = 4 := sorry

end health_risk_probability_l171_171918


namespace dodgeball_cost_l171_171356

theorem dodgeball_cost (B : ℝ) 
  (hb1 : 1.20 * B = 90) 
  (hb2 : B / 15 = 5) :
  ∃ (cost_per_dodgeball : ℝ), cost_per_dodgeball = 5 := by
sorry

end dodgeball_cost_l171_171356


namespace find_number_l171_171301

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l171_171301


namespace simplify_expression_l171_171583

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 :=
by
  sorry

end simplify_expression_l171_171583


namespace total_number_of_girls_in_school_l171_171805

theorem total_number_of_girls_in_school 
  (students_sampled : ℕ) 
  (students_total : ℕ) 
  (sample_girls : ℕ) 
  (sample_boys : ℕ)
  (h_sample_size : students_sampled = 200)
  (h_total_students : students_total = 2000)
  (h_diff_girls_boys : sample_boys = sample_girls + 6)
  (h_stratified_sampling : students_sampled / students_total = 200 / 2000) :
  sample_girls * (students_total / students_sampled) = 970 :=
by
  sorry

end total_number_of_girls_in_school_l171_171805


namespace not_perfect_square_l171_171227

-- Definitions and Conditions
def N (k : ℕ) : ℕ := (10^300 - 1) / 9 * 10^k

-- Proof Statement
theorem not_perfect_square (k : ℕ) : ¬∃ (m: ℕ), m * m = N k := 
sorry

end not_perfect_square_l171_171227


namespace minimally_intersecting_triples_modulo_1000_eq_344_l171_171295

def minimally_intersecting_triples_count_modulo : ℕ :=
  let total_count := 57344
  total_count % 1000

theorem minimally_intersecting_triples_modulo_1000_eq_344 :
  minimally_intersecting_triples_count_modulo = 344 := by
  sorry

end minimally_intersecting_triples_modulo_1000_eq_344_l171_171295


namespace meaningful_if_and_only_if_l171_171546

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end meaningful_if_and_only_if_l171_171546


namespace part1_part2_l171_171789

variable (x : ℝ)

def A := {x : ℝ | 1 < x ∧ x < 3}
def B := {x : ℝ | x < -3 ∨ 2 < x}

theorem part1 : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

theorem part2 (a b : ℝ) : (∀ x, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) → a = -5 ∧ b = 6 := by
  sorry

end part1_part2_l171_171789


namespace exists_plane_through_point_parallel_to_line_at_distance_l171_171929

-- Definitions of the given entities
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Line :=
(point : Point)
(direction : Point) -- Considering direction as a point vector for simplicity

def distance (P : Point) (L : Line) : ℝ := 
  -- Define the distance from point P to line L
  sorry

noncomputable def construct_plane (P : Point) (L : Line) (d : ℝ) : Prop :=
  -- Define when a plane can be constructed as stated in the problem.
  sorry

-- The main proof problem statement without the solution steps
theorem exists_plane_through_point_parallel_to_line_at_distance (P : Point) (L : Line) (d : ℝ) (h : distance P L > d) :
  construct_plane P L d :=
sorry

end exists_plane_through_point_parallel_to_line_at_distance_l171_171929


namespace symmetry_center_l171_171745

theorem symmetry_center {φ : ℝ} (hφ : |φ| < Real.pi / 2) (h : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ x : ℝ, 2 * Real.sin (2 * x + φ) = 2 * Real.sin (- (2 * x + φ)) ∧ x = -Real.pi / 6 :=
by
  sorry

end symmetry_center_l171_171745


namespace abigail_collected_43_l171_171087

noncomputable def cans_needed : ℕ := 100
noncomputable def collected_by_alyssa : ℕ := 30
noncomputable def more_to_collect : ℕ := 27
noncomputable def collected_by_abigail : ℕ := cans_needed - (collected_by_alyssa + more_to_collect)

theorem abigail_collected_43 : collected_by_abigail = 43 := by
  sorry

end abigail_collected_43_l171_171087


namespace belindas_age_l171_171181

theorem belindas_age (T B : ℕ) (h1 : T + B = 56) (h2 : B = 2 * T + 8) (h3 : T = 16) : B = 40 :=
by
  sorry

end belindas_age_l171_171181


namespace relationship_between_a_b_c_l171_171289

noncomputable def a : ℝ := 81 ^ 31
noncomputable def b : ℝ := 27 ^ 41
noncomputable def c : ℝ := 9 ^ 61

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_between_a_b_c_l171_171289


namespace find_second_projection_l171_171650

noncomputable def second_projection (plane : Prop) (first_proj : Prop) (distance : ℝ) : Prop :=
∃ second_proj : Prop, true

theorem find_second_projection 
  (plane : Prop) 
  (first_proj : Prop) 
  (distance : ℝ) :
  ∃ second_proj : Prop, true :=
sorry

end find_second_projection_l171_171650


namespace roots_solution_l171_171068

theorem roots_solution (p q : ℝ) (h1 : (∀ x : ℝ, (x - 3) * (3 * x + 8) = x^2 - 5 * x + 6 → (x = p ∨ x = q)))
  (h2 : p + q = 0) (h3 : p * q = -9) : (p + 4) * (q + 4) = 7 :=
by
  sorry

end roots_solution_l171_171068


namespace quadratic_condition_l171_171475

theorem quadratic_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 3 = 0) → a ≠ 1 :=
by
  sorry

end quadratic_condition_l171_171475


namespace promotional_price_difference_l171_171135

theorem promotional_price_difference
  (normal_price : ℝ)
  (months : ℕ)
  (issues_per_month : ℕ)
  (discount_per_issue : ℝ)
  (h1 : normal_price = 34)
  (h2 : months = 18)
  (h3 : issues_per_month = 2)
  (h4 : discount_per_issue = 0.25) : 
  normal_price - (months * issues_per_month * discount_per_issue) = 9 := 
by 
  sorry

end promotional_price_difference_l171_171135


namespace wizard_achievable_for_odd_n_l171_171602

-- Define what it means for the wizard to achieve his goal
def wizard_goal_achievable (n : ℕ) : Prop :=
  ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = 2 * n ∧ 
    ∀ (sorcerer_breaks : Finset (ℕ × ℕ)), sorcerer_breaks.card = n → 
      ∃ (dwarves : Finset ℕ), dwarves.card = 2 * n ∧
      ∀ k ∈ dwarves, ((k, (k + 1) % n) ∈ pairs ∨ ((k + 1) % n, k) ∈ pairs) ∧
                     (∀ i j, (i, j) ∈ sorcerer_breaks → ¬((i, j) ∈ pairs ∨ (j, i) ∈ pairs))

theorem wizard_achievable_for_odd_n (n : ℕ) (h : Odd n) : wizard_goal_achievable n := sorry

end wizard_achievable_for_odd_n_l171_171602


namespace basketball_player_height_l171_171500

noncomputable def player_height (H : ℝ) : Prop :=
  let reach := 22 / 12
  let jump := 32 / 12
  let total_rim_height := 10 + (6 / 12)
  H + reach + jump = total_rim_height

theorem basketball_player_height : ∃ H : ℝ, player_height H → H = 6 :=
by
  use 6
  sorry

end basketball_player_height_l171_171500


namespace possible_values_of_K_l171_171383

theorem possible_values_of_K (K N : ℕ) (h1 : K * (K + 1) = 2 * N^2) (h2 : N < 100) :
  K = 1 ∨ K = 8 ∨ K = 49 :=
sorry

end possible_values_of_K_l171_171383


namespace max_students_l171_171151

-- Define the constants for pens and pencils
def pens : ℕ := 1802
def pencils : ℕ := 1203

-- State that the GCD of pens and pencils is 1
theorem max_students : Nat.gcd pens pencils = 1 :=
by sorry

end max_students_l171_171151


namespace expand_expression_l171_171760

variable (x y : ℝ)

theorem expand_expression :
  12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 :=
by
  sorry

end expand_expression_l171_171760


namespace plates_count_l171_171986

variable (x : ℕ)
variable (first_taken : ℕ)
variable (second_taken : ℕ)
variable (remaining_plates : ℕ := 9)

noncomputable def plates_initial : ℕ :=
  let first_batch := (x - 2) / 3
  let remaining_after_first := x - 2 - first_batch
  let second_batch := remaining_after_first / 2
  let remaining_after_second := remaining_after_first - second_batch
  remaining_after_second

theorem plates_count (x : ℕ) (h : plates_initial x = remaining_plates) : x = 29 := sorry

end plates_count_l171_171986


namespace john_spent_at_candy_store_l171_171351

noncomputable def johns_allowance : ℝ := 2.40
noncomputable def arcade_spending : ℝ := (3 / 5) * johns_allowance
noncomputable def remaining_after_arcade : ℝ := johns_allowance - arcade_spending
noncomputable def toy_store_spending : ℝ := (1 / 3) * remaining_after_arcade
noncomputable def remaining_after_toy_store : ℝ := remaining_after_arcade - toy_store_spending
noncomputable def candy_store_spending : ℝ := remaining_after_toy_store

theorem john_spent_at_candy_store : candy_store_spending = 0.64 := by sorry

end john_spent_at_candy_store_l171_171351


namespace even_and_period_pi_l171_171590

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem even_and_period_pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by
  -- First, prove that f(x) is an even function: ∀ x, f(-x) = f(x)
  -- Next, find the smallest positive period T: ∃ T > 0, ∀ x, f(x + T) = f(x)
  -- Finally, show that this period is pi: T = π
  sorry

end even_and_period_pi_l171_171590


namespace initial_number_of_girls_l171_171133

theorem initial_number_of_girls (n : ℕ) (A : ℝ) 
  (h1 : (n + 1) * (A + 3) - 70 = n * A + 94) :
  n = 8 :=
by {
  sorry
}

end initial_number_of_girls_l171_171133


namespace calculate_difference_l171_171772

def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem calculate_difference :
  f (g 5) - g (f 5) = -2 := by
  sorry

end calculate_difference_l171_171772


namespace peter_money_l171_171609

theorem peter_money (cost_per_ounce : ℝ) (amount_bought : ℝ) (leftover_money : ℝ) (total_money : ℝ) :
  cost_per_ounce = 0.25 ∧ amount_bought = 6 ∧ leftover_money = 0.50 → total_money = 2 :=
by
  intros h
  let h1 := h.1
  let h2 := h.2.1
  let h3 := h.2.2
  sorry

end peter_money_l171_171609


namespace probability_of_x_gt_5y_l171_171666

theorem probability_of_x_gt_5y :
  let rectangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y ≤ 2500}
  let area_of_rectangle := 3000 * 2500
  let triangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y < x / 5}
  let area_of_triangle := (3000 * 600) / 2
  ∃ prob : ℚ, (area_of_triangle / area_of_rectangle = prob) ∧ prob = 3 / 25 := by
  sorry

end probability_of_x_gt_5y_l171_171666


namespace number_of_dogs_l171_171990

theorem number_of_dogs (cost_price selling_price total_amount : ℝ) (profit_percentage : ℝ)
    (h1 : cost_price = 1000)
    (h2 : profit_percentage = 0.30)
    (h3 : total_amount = 2600)
    (h4 : selling_price = cost_price + (profit_percentage * cost_price)) :
    total_amount / selling_price = 2 :=
by
  sorry

end number_of_dogs_l171_171990


namespace sum_of_squares_of_consecutive_integers_l171_171509

theorem sum_of_squares_of_consecutive_integers (a : ℝ) (h : (a-1)*a*(a+1) = 36*a) :
  (a-1)^2 + a^2 + (a+1)^2 = 77 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l171_171509


namespace monthly_installment_amount_l171_171399

variable (cashPrice : ℕ) (deposit : ℕ) (monthlyInstallments : ℕ) (savingsIfCash : ℕ)

-- Defining the conditions
def conditions := 
  cashPrice = 8000 ∧ 
  deposit = 3000 ∧ 
  monthlyInstallments = 30 ∧ 
  savingsIfCash = 4000

-- Proving the amount of each monthly installment
theorem monthly_installment_amount (h : conditions cashPrice deposit monthlyInstallments savingsIfCash) : 
  (12000 - deposit) / monthlyInstallments = 300 :=
sorry

end monthly_installment_amount_l171_171399


namespace find_slope_of_line_q_l171_171669

theorem find_slope_of_line_q
  (k : ℝ)
  (h₁ : ∀ (x y : ℝ), (y = 3 * x + 5) → (y = k * x + 3) → (x = -4 ∧ y = -7))
  : k = 2.5 :=
sorry

end find_slope_of_line_q_l171_171669


namespace ratio_of_josh_to_brad_l171_171256

theorem ratio_of_josh_to_brad (J D B : ℝ) (h1 : J + D + B = 68) (h2 : J = (3 / 4) * D) (h3 : D = 32) :
  (J / B) = 2 :=
by
  sorry

end ratio_of_josh_to_brad_l171_171256


namespace find_d_minus_c_l171_171089

variable (c d : ℝ)

def rotate180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (2 * cx - x, 2 * cy - y)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

def transformations (q : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (rotate180 q (2, 3))

theorem find_d_minus_c :
  transformations (c, d) = (1, -4) → d - c = 7 :=
by
  intro h
  sorry

end find_d_minus_c_l171_171089


namespace last_two_digits_of_17_pow_17_l171_171111

theorem last_two_digits_of_17_pow_17 : (17 ^ 17) % 100 = 77 := 
by sorry

end last_two_digits_of_17_pow_17_l171_171111


namespace work_rate_problem_l171_171736

theorem work_rate_problem (A B : ℚ) (h1 : A + B = 1/8) (h2 : A = 1/12) : B = 1/24 :=
sorry

end work_rate_problem_l171_171736


namespace evaluate_101_times_101_l171_171237

theorem evaluate_101_times_101 : 101 * 101 = 10201 :=
by sorry

end evaluate_101_times_101_l171_171237


namespace chocolates_sold_l171_171189

theorem chocolates_sold (C S : ℝ) (n : ℝ)
  (h1 : 65 * C = n * S)
  (h2 : S = 1.3 * C) :
  n = 50 :=
by
  sorry

end chocolates_sold_l171_171189


namespace binom_1300_2_eq_844350_l171_171471

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l171_171471


namespace find_x_given_y_l171_171029

theorem find_x_given_y (x y : ℤ) (h1 : 16 * (4 : ℝ)^x = 3^(y + 2)) (h2 : y = -2) : x = -2 := by
  sorry

end find_x_given_y_l171_171029


namespace total_number_of_girls_l171_171385

-- Define the given initial number of girls and the number of girls joining the school
def initial_girls : Nat := 732
def girls_joined : Nat := 682
def total_girls : Nat := 1414

-- Formalize the problem
theorem total_number_of_girls :
  initial_girls + girls_joined = total_girls :=
by
  -- placeholder for proof
  sorry

end total_number_of_girls_l171_171385


namespace Sara_spent_on_each_movie_ticket_l171_171081

def Sara_spent_on_each_movie_ticket_correct : Prop :=
  let T := 36.78
  let R := 1.59
  let B := 13.95
  (T - R - B) / 2 = 10.62

theorem Sara_spent_on_each_movie_ticket : 
  Sara_spent_on_each_movie_ticket_correct :=
by
  sorry

end Sara_spent_on_each_movie_ticket_l171_171081


namespace geometric_progression_common_ratio_l171_171535

theorem geometric_progression_common_ratio (y r : ℝ) (h : (40 + y)^2 = (10 + y) * (90 + y)) :
  r = (40 + y) / (10 + y) → r = (90 + y) / (40 + y) → r = 5 / 3 :=
by
  sorry

end geometric_progression_common_ratio_l171_171535


namespace sum_of_three_quadratics_no_rot_l171_171009

def quad_poly_sum_no_root (p q : ℝ -> ℝ) : Prop :=
  ∀ x : ℝ, (p x + q x ≠ 0)

theorem sum_of_three_quadratics_no_rot (a b c d e f : ℝ)
    (h1 : quad_poly_sum_no_root (λ x => x^2 + a*x + b) (λ x => x^2 + c*x + d))
    (h2 : quad_poly_sum_no_root (λ x => x^2 + c*x + d) (λ x => x^2 + e*x + f))
    (h3 : quad_poly_sum_no_root (λ x => x^2 + e*x + f) (λ x => x^2 + a*x + b)) :
    quad_poly_sum_no_root (λ x => x^2 + a*x + b) 
                         (λ x => x^2 + c*x + d + x^2 + e*x + f) :=
sorry

end sum_of_three_quadratics_no_rot_l171_171009


namespace no_elimination_method_l171_171497

theorem no_elimination_method
  (x y : ℤ)
  (h1 : x + 3 * y = 4)
  (h2 : 2 * x - y = 1) :
  ¬ (∀ z : ℤ, z = x + 3 * y - 3 * (2 * x - y)) →
  ∃ x y : ℤ, x + 3 * y - 3 * (2 * x - y) ≠ 0 := sorry

end no_elimination_method_l171_171497


namespace odd_function_condition_l171_171732

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x) / ((x - a) * (x + 1))

theorem odd_function_condition (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 := 
sorry

end odd_function_condition_l171_171732


namespace hollis_student_loan_l171_171426

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end hollis_student_loan_l171_171426


namespace cyclic_quadrilateral_AD_correct_l171_171690

noncomputable def cyclic_quadrilateral_AD_length : ℝ :=
  let R := 200 * Real.sqrt 2
  let AB := 200
  let BC := 200
  let CD := 200
  let AD := 500
  sorry

theorem cyclic_quadrilateral_AD_correct (R AB BC CD AD : ℝ) (hR : R = 200 * Real.sqrt 2) 
  (hAB : AB = 200) (hBC : BC = 200) (hCD : CD = 200) : AD = 500 :=
by
  have hRABBCDC: R = 200 * Real.sqrt 2 ∧ AB = 200 ∧ BC = 200 ∧ CD = 200 := ⟨hR, hAB, hBC, hCD⟩
  sorry

end cyclic_quadrilateral_AD_correct_l171_171690


namespace city_division_exists_l171_171835

-- Define the problem conditions and prove the required statement
theorem city_division_exists (squares : Type) (streets : squares → squares → Prop)
  (h_outgoing: ∀ (s : squares), ∃ t u : squares, streets s t ∧ streets s u) :
  ∃ (districts : squares → ℕ), (∀ (s t : squares), districts s ≠ districts t → streets s t ∨ streets t s) ∧
  (∀ (i j : ℕ), i ≠ j → ∀ (s t : squares), districts s = i → districts t = j → streets s t ∨ streets t s) ∧
  (∃ m : ℕ, m = 1014) :=
sorry

end city_division_exists_l171_171835


namespace area_H1H2H3_eq_four_l171_171195

section TriangleArea

variables {P D E F H1 H2 H3 : Type*}

-- Definitions of midpoints, centroid, etc. can be implicit in Lean's formalism if necessary
-- We'll represent the area relation directly

-- Assume P is inside triangle DEF
def point_inside_triangle (P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Assume H1, H2, H3 are centroids of triangles PDE, PEF, PFD respectively
def is_centroid (H1 H2 H3 P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Given the area of triangle DEF
def area_DEF : ℝ := 12

-- Define the area function for the triangle formed by specific points
def area_triangle (A B C : Type*) : ℝ :=
sorry  -- Actual computation is abstracted

-- Mathematical statement to be proven
theorem area_H1H2H3_eq_four (P D E F H1 H2 H3 : Type*)
  (h_inside : point_inside_triangle P D E F)
  (h_centroid : is_centroid H1 H2 H3 P D E F)
  (h_area_DEF : area_triangle D E F = area_DEF) :
  area_triangle H1 H2 H3 = 4 :=
sorry

end TriangleArea

end area_H1H2H3_eq_four_l171_171195


namespace existence_of_positive_numbers_l171_171921

open Real

theorem existence_of_positive_numbers {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 > 2 ∧ a^3 + b^3 + c^3 < 2 ∧ a^4 + b^4 + c^4 > 2 :=
sorry

end existence_of_positive_numbers_l171_171921


namespace number_of_triangles_l171_171186

/-- 
  This statement defines and verifies the number of triangles 
  in the given geometric figure.
-/
theorem number_of_triangles (rectangle : Set ℝ) : 
  (exists lines : Set (List (ℝ × ℝ)), -- assuming a set of lines dividing the rectangle
    let small_right_triangles := 40
    let intermediate_isosceles_triangles := 8
    let intermediate_triangles := 10
    let larger_right_triangles := 20
    let largest_isosceles_triangles := 5
    small_right_triangles + intermediate_isosceles_triangles + intermediate_triangles + larger_right_triangles + largest_isosceles_triangles = 83) :=
sorry

end number_of_triangles_l171_171186


namespace part1_part2_1_part2_2_l171_171839

theorem part1 (n : ℚ) :
  (2 / 2 + n / 5 = (2 + n) / 7) → n = -25 / 2 :=
by sorry

theorem part2_1 (m n : ℚ) :
  (m / 2 + n / 5 = (m + n) / 7) → m = -4 / 25 * n :=
by sorry

theorem part2_2 (m n: ℚ) :
  (m = -4 / 25 * n) → (25 * m + n = 6) → (m = 8 / 25 ∧ n = -2) :=
by sorry

end part1_part2_1_part2_2_l171_171839


namespace directrix_parabola_y_eq_2x2_l171_171878

theorem directrix_parabola_y_eq_2x2 : (∃ y : ℝ, y = 2 * x^2) → (∃ y : ℝ, y = -1/8) :=
by
  sorry

end directrix_parabola_y_eq_2x2_l171_171878


namespace vector_at_t5_l171_171110

theorem vector_at_t5 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h1 : (a, b) = (2, 5)) 
  (h2 : (a + 3 * c, b + 3 * d) = (8, -7)) :
  (a + 5 * c, b + 5 * d) = (10, -11) :=
by
  sorry

end vector_at_t5_l171_171110


namespace total_photos_l171_171439

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end total_photos_l171_171439


namespace students_in_A_and_D_combined_l171_171163

theorem students_in_A_and_D_combined (AB BC CD : ℕ) (hAB : AB = 83) (hBC : BC = 86) (hCD : CD = 88) : (AB + CD - BC = 85) :=
by
  sorry

end students_in_A_and_D_combined_l171_171163


namespace least_four_digit_divisible_1_2_4_8_l171_171375

theorem least_four_digit_divisible_1_2_4_8 : ∃ n : ℕ, ∀ d1 d2 d3 d4 : ℕ, 
  n = d1*1000 + d2*100 + d3*10 + d4 ∧
  1000 ≤ n ∧ n < 10000 ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4 ∧
  n % 1 = 0 ∧
  n % 2 = 0 ∧
  n % 4 = 0 ∧
  n % 8 = 0 ∧
  n = 1248 :=
by
  sorry

end least_four_digit_divisible_1_2_4_8_l171_171375


namespace correct_student_mark_l171_171321

theorem correct_student_mark (x : ℕ) : 
  (∀ (n : ℕ), n = 30) →
  (∀ (avg correct_avg wrong_mark correct_mark : ℕ), 
    avg = 100 ∧ 
    correct_avg = 98 ∧ 
    wrong_mark = 70 ∧ 
    (n * avg) - wrong_mark + correct_mark = n * correct_avg) →
  x = 10 := by
  intros
  sorry

end correct_student_mark_l171_171321


namespace part_3_l171_171374

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) := (1/2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) := f x m + g x m

theorem part_3 (x₁ x₂ : ℝ) (m : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hm : m = -2)
  (hF : F x₁ m + F x₂ m + x₁ * x₂ = 0) : x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end part_3_l171_171374


namespace reading_schedule_l171_171100

-- Definitions of reading speeds and conditions
def total_pages := 910
def alice_speed := 30  -- seconds per page
def bob_speed := 60    -- seconds per page
def chandra_speed := 45  -- seconds per page

-- Mathematical problem statement
theorem reading_schedule :
  ∃ (x y : ℕ), 
    (x < y) ∧ 
    (y ≤ total_pages) ∧ 
    (30 * x = 45 * (y - x) ∧ 45 * (y - x) = 60 * (total_pages - y)) ∧ 
    x = 420 ∧ 
    y = 700 :=
  sorry

end reading_schedule_l171_171100


namespace find_weight_of_silver_in_metal_bar_l171_171523

noncomputable def weight_loss_ratio_tin : ℝ := 1.375 / 10
noncomputable def weight_loss_ratio_silver : ℝ := 0.375
noncomputable def ratio_tin_silver : ℝ := 0.6666666666666664

theorem find_weight_of_silver_in_metal_bar (T S : ℝ)
  (h1 : T + S = 70)
  (h2 : T / S = ratio_tin_silver)
  (h3 : weight_loss_ratio_tin * T + weight_loss_ratio_silver * S = 7) :
  S = 15 :=
by
  sorry

end find_weight_of_silver_in_metal_bar_l171_171523


namespace average_weight_decrease_l171_171119

theorem average_weight_decrease 
  (A1 : ℝ) (new_person_weight : ℝ) (num_initial : ℕ) (num_total : ℕ) 
  (hA1 : A1 = 55) (hnew_person_weight : new_person_weight = 50) 
  (hnum_initial : num_initial = 20) (hnum_total : num_total = 21) :
  A1 - ((A1 * num_initial + new_person_weight) / num_total) = 0.24 :=
by
  rw [hA1, hnew_person_weight, hnum_initial, hnum_total]
  -- Further proof steps would go here
  sorry

end average_weight_decrease_l171_171119


namespace flour_ratio_correct_l171_171183

-- Definitions based on conditions
def initial_sugar : ℕ := 13
def initial_flour : ℕ := 25
def initial_baking_soda : ℕ := 35
def initial_cocoa_powder : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa_powder : ℕ := 15

-- Calculate remaining ingredients
def remaining_flour : ℕ := initial_flour - added_flour
def remaining_sugar : ℕ := initial_sugar - added_sugar
def remaining_cocoa_powder : ℕ := initial_cocoa_powder - added_cocoa_powder

-- Calculate ratio
def total_remaining_sugar_and_cocoa : ℕ := remaining_sugar + remaining_cocoa_powder
def flour_to_sugar_cocoa_ratio : ℕ × ℕ := (remaining_flour, total_remaining_sugar_and_cocoa)

-- Proposition stating the desired ratio
theorem flour_ratio_correct : flour_to_sugar_cocoa_ratio = (17, 46) := by
  sorry

end flour_ratio_correct_l171_171183


namespace triangle_side_length_c_l171_171107

theorem triangle_side_length_c
  (a b A B C : ℝ)
  (ha : a = Real.sqrt 3)
  (hb : b = 1)
  (hA : A = 2 * B)
  (hAngleSum : A + B + C = Real.pi) :
  ∃ c : ℝ, c = 2 := 
by
  sorry

end triangle_side_length_c_l171_171107


namespace fewer_vip_tickets_sold_l171_171359

-- Definitions based on the conditions
variables (V G : ℕ)
def tickets_sold := V + G = 320
def total_cost := 40 * V + 10 * G = 7500

-- The main statement to prove
theorem fewer_vip_tickets_sold :
  tickets_sold V G → total_cost V G → G - V = 34 := 
by
  intros h1 h2
  sorry

end fewer_vip_tickets_sold_l171_171359


namespace max_value_exponential_and_power_functions_l171_171780

variable (a b : ℝ)

-- Given conditions
axiom condition : 0 < b ∧ b < a ∧ a < 1

-- Problem statement
theorem max_value_exponential_and_power_functions : 
  a^b = max (max (a^b) (b^a)) (max (a^a) (b^b)) :=
by
  sorry

end max_value_exponential_and_power_functions_l171_171780


namespace percentage_increase_l171_171230

theorem percentage_increase (R W : ℕ) (hR : R = 36) (hW : W = 20) : 
  ((R - W) / W : ℚ) * 100 = 80 := 
by 
  sorry

end percentage_increase_l171_171230


namespace total_leaves_l171_171014

theorem total_leaves (ferns fronds leaves : ℕ) (h1 : ferns = 12) (h2 : fronds = 15) (h3 : leaves = 45) :
  ferns * fronds * leaves = 8100 :=
by
  sorry

end total_leaves_l171_171014


namespace sufficient_not_necessary_condition_l171_171109

-- Define the condition on a
def condition (a : ℝ) : Prop := a > 0

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) : Prop := a^2 + a ≥ 0

-- The proof statement that "a > 0" is a sufficient but not necessary condition for "a^2 + a ≥ 0"
theorem sufficient_not_necessary_condition (a : ℝ) : condition a → quadratic_inequality a :=
by
    intro ha
    -- [The remaining part of the proof is skipped.]
    sorry

end sufficient_not_necessary_condition_l171_171109


namespace bill_cooking_time_l171_171026

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end bill_cooking_time_l171_171026


namespace total_steps_to_times_square_l171_171420

-- Define the conditions
def steps_to_rockefeller : ℕ := 354
def steps_to_times_square_from_rockefeller : ℕ := 228

-- State the theorem using the conditions
theorem total_steps_to_times_square : 
  steps_to_rockefeller + steps_to_times_square_from_rockefeller = 582 := 
  by 
    -- We skip the proof for now
    sorry

end total_steps_to_times_square_l171_171420


namespace calculate_expression_l171_171914

theorem calculate_expression : 2 * (-2) + (-3) = -7 := 
  sorry

end calculate_expression_l171_171914


namespace ratio_of_areas_l171_171976

-- Definitions based on the conditions given
variables (A B M N P Q O : Type) 
variables (AB BM BP : ℝ)

-- Assumptions
axiom hAB : AB = 6
axiom hBM : BM = 9
axiom hBP : BP = 5

-- Theorem statement
theorem ratio_of_areas (hMN : M ≠ N) (hPQ : P ≠ Q) :
  (1 / 121 : ℝ) = sorry :=
by sorry

end ratio_of_areas_l171_171976


namespace Yura_catches_up_in_five_minutes_l171_171073

-- Define the speeds and distances
variables (v_Lena v_Yura d_Lena d_Yura : ℝ)
-- Assume v_Yura = 2 * v_Lena (Yura is twice as fast)
axiom h1 : v_Yura = 2 * v_Lena 
-- Assume Lena walks for 5 minutes before Yura starts
axiom h2 : d_Lena = v_Lena * 5
-- Assume they walk at constant speeds
noncomputable def t_to_catch_up := 10 / 2 -- time Yura takes to catch up Lena

-- Define the proof problem
theorem Yura_catches_up_in_five_minutes :
    t_to_catch_up = 5 :=
by
    sorry

end Yura_catches_up_in_five_minutes_l171_171073


namespace find_original_number_l171_171164

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_original_number_l171_171164


namespace total_skateboarding_distance_l171_171902

def skateboarded_to_park : ℕ := 16
def skateboarded_back_home : ℕ := 9

theorem total_skateboarding_distance : 
  skateboarded_to_park + skateboarded_back_home = 25 := by 
  sorry

end total_skateboarding_distance_l171_171902


namespace emma_harry_weight_l171_171367

theorem emma_harry_weight (e f g h : ℕ) 
  (h1 : e + f = 280) 
  (h2 : f + g = 260) 
  (h3 : g + h = 290) : 
  e + h = 310 := 
sorry

end emma_harry_weight_l171_171367


namespace yanni_paintings_l171_171758

theorem yanni_paintings
  (total_area : ℤ)
  (painting1 : ℕ → ℤ × ℤ)
  (painting2 : ℤ × ℤ)
  (painting3 : ℤ × ℤ)
  (num_paintings : ℕ) :
  total_area = 200
  → painting1 1 = (5, 5)
  → painting1 2 = (5, 5)
  → painting1 3 = (5, 5)
  → painting2 = (10, 8)
  → painting3 = (5, 9)
  → num_paintings = 5 := 
by
  sorry

end yanni_paintings_l171_171758


namespace total_books_correct_l171_171036

-- Define the number of books each person has
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45

-- Calculate the total number of books they have together
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books

-- State the theorem that needs to be proved
theorem total_books_correct : total_books = 120 :=
by
  sorry

end total_books_correct_l171_171036


namespace max_sum_of_factors_l171_171344

theorem max_sum_of_factors (p q : ℕ) (hpq : p * q = 100) : p + q ≤ 101 :=
sorry

end max_sum_of_factors_l171_171344


namespace roots_quadratic_eq_k_l171_171827

theorem roots_quadratic_eq_k (k : ℝ) :
  (∀ x : ℝ, (5 * x^2 + 20 * x + k = 0) ↔ (x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10)) →
  k = 17 := by
  intro h
  sorry

end roots_quadratic_eq_k_l171_171827


namespace exists_nat_n_gt_one_sqrt_expr_nat_l171_171173

theorem exists_nat_n_gt_one_sqrt_expr_nat (n : ℕ) : ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7 / 8) = m :=
by
  sorry

end exists_nat_n_gt_one_sqrt_expr_nat_l171_171173


namespace inverse_mod_53_l171_171461

theorem inverse_mod_53 (h : 17 * 13 % 53 = 1) : 36 * 40 % 53 = 1 :=
by
  -- Given condition: 17 * 13 % 53 = 1
  -- Derived condition: (-17) * -13 % 53 = 1 which is equivalent to 17 * 13 % 53 = 1
  -- So we need to find: 36 * x % 53 = 1 where x = -13 % 53 => x = 40
  sorry

end inverse_mod_53_l171_171461


namespace cube_cross_section_area_l171_171993

def cube_edge_length (a : ℝ) := a > 0

def plane_perpendicular_body_diagonal := 
  ∃ (p : ℝ × ℝ × ℝ), ∀ (x y z : ℝ), 
  p = (x / 2, y / 2, z / 2) ∧ 
  (x + y + z) = (1 : ℝ)

theorem cube_cross_section_area
  (a : ℝ) 
  (h : cube_edge_length a) 
  (plane : plane_perpendicular_body_diagonal) : 
  ∃ (A : ℝ), 
  A = (3 * a^2 * Real.sqrt 3 / 4) := sorry

end cube_cross_section_area_l171_171993


namespace cost_of_birthday_gift_l171_171972

theorem cost_of_birthday_gift 
  (boss_contrib : ℕ)
  (todd_contrib : ℕ)
  (employee_contrib : ℕ)
  (num_employees : ℕ)
  (h1 : boss_contrib = 15)
  (h2 : todd_contrib = 2 * boss_contrib)
  (h3 : employee_contrib = 11)
  (h4 : num_employees = 5) :
  boss_contrib + todd_contrib + num_employees * employee_contrib = 100 := by
  sorry

end cost_of_birthday_gift_l171_171972


namespace circle_k_range_l171_171802

theorem circle_k_range {k : ℝ}
  (h : ∀ x y : ℝ, x^2 + y^2 - 2*x + y + k = 0) :
  k < 5 / 4 :=
sorry

end circle_k_range_l171_171802


namespace ball_bounce_height_l171_171973

noncomputable def height_after_bounces (h₀ : ℝ) (r : ℝ) (b : ℕ) : ℝ :=
  h₀ * (r ^ b)

theorem ball_bounce_height
  (h₀ : ℝ) (r : ℝ) (hb : ℕ) (h₀_pos : h₀ > 0) (r_pos : 0 < r ∧ r < 1) (h₀_val : h₀ = 320) (r_val : r = 3 / 4) (height_limit : ℝ) (height_limit_val : height_limit = 40):
  (hb ≥ 6) ∧ height_after_bounces h₀ r hb < height_limit :=
by
  sorry

end ball_bounce_height_l171_171973


namespace find_fraction_of_cistern_l171_171866

noncomputable def fraction_initially_full (x : ℝ) : Prop :=
  let rateA := (1 - x) / 12
  let rateB := (1 - x) / 8
  let combined_rate := 1 / 14.4
  combined_rate = rateA + rateB

theorem find_fraction_of_cistern {x : ℝ} (h : fraction_initially_full x) : x = 2 / 3 :=
by
  sorry

end find_fraction_of_cistern_l171_171866


namespace fair_coin_flip_difference_l171_171380

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l171_171380


namespace rebus_solution_l171_171820

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end rebus_solution_l171_171820


namespace bouquet_cost_l171_171823

theorem bouquet_cost (c : ℕ) : (c / 25 = 30 / 15) → c = 50 := by
  sorry

end bouquet_cost_l171_171823


namespace moles_Cl2_combined_l171_171244

-- Condition Definitions
def moles_C2H6 := 2
def moles_HCl_formed := 2
def balanced_reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) : Prop :=
  C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Mathematical Equivalent Proof Problem Statement
theorem moles_Cl2_combined (C2H6 Cl2 HCl C2H4Cl2 : ℝ) (h1 : C2H6 = 2) 
(h2 : HCl = 2) (h3 : balanced_reaction C2H6 Cl2 C2H4Cl2 HCl) :
  Cl2 = 1 :=
by
  -- The proof is stated here.
  sorry

end moles_Cl2_combined_l171_171244


namespace problem_statement_l171_171206

namespace ProofProblems

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 5, 6}

theorem problem_statement : M ∪ N = U := sorry

end ProofProblems

end problem_statement_l171_171206


namespace most_probable_light_l171_171425

theorem most_probable_light (red_duration : ℕ) (yellow_duration : ℕ) (green_duration : ℕ) :
  red_duration = 30 ∧ yellow_duration = 5 ∧ green_duration = 40 →
  (green_duration / (red_duration + yellow_duration + green_duration) > red_duration / (red_duration + yellow_duration + green_duration)) ∧
  (green_duration / (red_duration + yellow_duration + green_duration) > yellow_duration / (red_duration + yellow_duration + green_duration)) :=
by
  sorry

end most_probable_light_l171_171425


namespace lemons_per_glass_l171_171604

theorem lemons_per_glass (lemons glasses : ℕ) (h : lemons = 18 ∧ glasses = 9) : lemons / glasses = 2 :=
by
  sorry

end lemons_per_glass_l171_171604


namespace regular_polygon_sides_l171_171043

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l171_171043


namespace small_gifts_combinations_large_gifts_combinations_l171_171577

/-
  Definitions based on the given conditions:
  - 12 varieties of wrapping paper.
  - 3 colors of ribbon.
  - 6 types of gift cards.
  - Small gifts can use only 2 out of the 3 ribbon colors.
-/

def wrapping_paper_varieties : ℕ := 12
def ribbon_colors : ℕ := 3
def gift_card_types : ℕ := 6
def small_gift_ribbon_colors : ℕ := 2

/-
  Proof problems:

  - For small gifts, there are 12 * 2 * 6 combinations.
  - For large gifts, there are 12 * 3 * 6 combinations.
-/

theorem small_gifts_combinations :
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types = 144 :=
by
  sorry

theorem large_gifts_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types = 216 :=
by
  sorry

end small_gifts_combinations_large_gifts_combinations_l171_171577


namespace solution_interval_log_eq_l171_171832

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) + x - 3

theorem solution_interval_log_eq (h_mono : ∀ x y, (0 < x ∧ x < y) → f x < f y)
  (h_f2 : f 2 = 0)
  (h_f3 : f 3 > 0) :
  ∃ x, (2 ≤ x ∧ x < 3 ∧ f x = 0) :=
by
  sorry

end solution_interval_log_eq_l171_171832


namespace value_of_y_at_48_l171_171261

open Real

noncomputable def collinear_points (x : ℝ) : ℝ :=
  if x = 2 then 5
  else if x = 6 then 17
  else if x = 10 then 29
  else if x = 48 then 143
  else 0 -- placeholder value for other x (not used in proof)

theorem value_of_y_at_48 :
  (∀ (x1 x2 x3 : ℝ), x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → 
    ∃ (m : ℝ), m = (collinear_points x2 - collinear_points x1) / (x2 - x1) ∧ 
               m = (collinear_points x3 - collinear_points x2) / (x3 - x2)) →
  collinear_points 48 = 143 :=
by
  sorry

end value_of_y_at_48_l171_171261


namespace cos_alpha_value_l171_171639

noncomputable def cos_alpha (α : ℝ) : ℝ :=
  (3 - 4 * Real.sqrt 3) / 10

theorem cos_alpha_value (α : ℝ) (h1 : Real.sin (Real.pi / 6 + α) = 3 / 5) (h2 : Real.pi / 3 < α ∧ α < 5 * Real.pi / 6) :
  Real.cos α = cos_alpha α :=
by
sorry

end cos_alpha_value_l171_171639


namespace solve_inequalities_l171_171579

theorem solve_inequalities :
  {x : ℝ | 4 ≤ (2*x) / (3*x - 7) ∧ (2*x) / (3*x - 7) < 9} = {x : ℝ | (63 / 25) < x ∧ x ≤ 2.8} :=
by
  sorry

end solve_inequalities_l171_171579


namespace height_of_parallelogram_l171_171959

theorem height_of_parallelogram
  (A B H : ℝ)
  (h1 : A = 480)
  (h2 : B = 32)
  (h3 : A = B * H) : 
  H = 15 := sorry

end height_of_parallelogram_l171_171959


namespace cos420_add_sin330_l171_171840

theorem cos420_add_sin330 : Real.cos (420 * Real.pi / 180) + Real.sin (330 * Real.pi / 180) = 0 := 
by
  sorry

end cos420_add_sin330_l171_171840


namespace gain_percentage_is_66_67_l171_171051

variable (C S : ℝ)
variable (cost_price_eq : 20 * C = 12 * S)

theorem gain_percentage_is_66_67 (h : 20 * C = 12 * S) : (((5 / 3) * C - C) / C) * 100 = 66.67 := by
  sorry

end gain_percentage_is_66_67_l171_171051


namespace value_of_expression_eq_33_l171_171675

theorem value_of_expression_eq_33 : (3^2 + 7^2 - 5^2 = 33) := by
  sorry

end value_of_expression_eq_33_l171_171675


namespace eval_f_3_minus_f_neg_3_l171_171863

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 7 * x

-- State the theorem
theorem eval_f_3_minus_f_neg_3 : f 3 - f (-3) = 690 := by
  sorry

end eval_f_3_minus_f_neg_3_l171_171863


namespace area_of_rectangle_l171_171270

theorem area_of_rectangle (s : ℝ) (h1 : 4 * s = 100) : 2 * s * 2 * s = 2500 := by
  sorry

end area_of_rectangle_l171_171270


namespace roots_equality_l171_171761

variable {α β p q : ℝ}

theorem roots_equality (h1 : α ≠ β)
    (h2 : α * α + p * α + q = 0 ∧ β * β + p * β + q = 0)
    (h3 : α^3 - α^2 * β - α * β^2 + β^3 = 0) : 
  p = 0 ∧ q < 0 :=
by 
  sorry

end roots_equality_l171_171761


namespace seashells_total_l171_171738

theorem seashells_total (x y z T : ℕ) (m k : ℝ) 
  (h₁ : x = 2) 
  (h₂ : y = 5) 
  (h₃ : z = 9) 
  (h₄ : x + y = T) 
  (h₅ : m * x + k * y = z) : 
  T = 7 :=
by
  -- This is where the proof would go.
  sorry

end seashells_total_l171_171738


namespace incorrect_statement_C_l171_171024

theorem incorrect_statement_C :
  (∀ b h : ℕ, (2 * b) * h = 2 * (b * h)) ∧
  (∀ b h : ℕ, (1 / 2) * b * (2 * h) = 2 * ((1 / 2) * b * h)) ∧
  (∀ r : ℕ, (π * (2 * r) ^ 2 ≠ 2 * (π * r ^ 2))) ∧
  (∀ a b : ℕ, (a / 2) / (2 * b) ≠ a / b) ∧
  (∀ x : ℤ, x < 0 -> 2 * x < x) →
  false :=
by
  intros h
  sorry

end incorrect_statement_C_l171_171024


namespace sufficient_condition_l171_171644

variable (x : ℝ) (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, |x| + |x - 1| ≥ 1) : a < 1 → ∀ x : ℝ, a ≤ |x| + |x - 1| :=
by
  sorry

end sufficient_condition_l171_171644


namespace platform_length_correct_l171_171392

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_cross_platform : ℝ := 30
noncomputable def time_cross_man : ℝ := 19
noncomputable def length_train : ℝ := train_speed_mps * time_cross_man
noncomputable def total_distance_cross_platform : ℝ := train_speed_mps * time_cross_platform
noncomputable def length_platform : ℝ := total_distance_cross_platform - length_train

theorem platform_length_correct : length_platform = 220 := by
  sorry

end platform_length_correct_l171_171392


namespace find_speed_of_P_l171_171701

noncomputable def walking_speeds (v_P v_Q : ℝ) : Prop :=
  let distance_XY := 90
  let distance_meet_from_Y := 15
  let distance_P := distance_XY - distance_meet_from_Y
  let distance_Q := distance_XY + distance_meet_from_Y
  (v_Q = v_P + 3) ∧
  (distance_P / v_P = distance_Q / v_Q)

theorem find_speed_of_P : ∃ v_P : ℝ, walking_speeds v_P (v_P + 3) ∧ v_P = 7.5 :=
by
  sorry

end find_speed_of_P_l171_171701


namespace geometric_sequence_b_l171_171849

theorem geometric_sequence_b (b : ℝ) (h : b > 0) (s : ℝ) 
  (h1 : 30 * s = b) (h2 : b * s = 15 / 4) : 
  b = 15 * Real.sqrt 2 / 2 := 
by
  sorry

end geometric_sequence_b_l171_171849


namespace keith_total_spent_l171_171868

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tire_cost : ℝ := 112.46
def num_tires : ℕ := 4
def printer_cable_cost : ℝ := 14.85
def num_printer_cables : ℕ := 2
def blank_cd_pack_cost : ℝ := 0.98
def num_blank_cds : ℕ := 10
def sales_tax_rate : ℝ := 0.0825

theorem keith_total_spent : 
  speakers_cost +
  cd_player_cost +
  (num_tires * tire_cost) +
  (num_printer_cables * printer_cable_cost) +
  (num_blank_cds * blank_cd_pack_cost) *
  (1 + sales_tax_rate) = 827.87 := 
sorry

end keith_total_spent_l171_171868


namespace number_of_students_with_at_least_two_pets_l171_171674

-- Definitions for the sets of students
def total_students := 50
def dog_students := 35
def cat_students := 40
def rabbit_students := 10
def dog_and_cat_students := 20
def dog_and_rabbit_students := 5
def cat_and_rabbit_students := 0  -- Assuming minimal overlap

-- Problem Statement
theorem number_of_students_with_at_least_two_pets :
  (dog_and_cat_students + dog_and_rabbit_students + cat_and_rabbit_students) = 25 :=
by
  sorry

end number_of_students_with_at_least_two_pets_l171_171674


namespace geometric_sequence_sum_l171_171737

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 1) (h3 : a 4 * a 5 * a 6 = 8) :
  a 2 + a 5 + a 8 + a 11 = 15 :=
by
  sorry

end geometric_sequence_sum_l171_171737


namespace profit_made_after_two_years_l171_171570

variable (present_value : ℝ) (depreciation_rate : ℝ) (selling_price : ℝ) 

def value_after_one_year (present_value depreciation_rate : ℝ) : ℝ :=
  present_value - (depreciation_rate * present_value)

def value_after_two_years (value_after_one_year : ℝ) (depreciation_rate : ℝ) : ℝ :=
  value_after_one_year - (depreciation_rate * value_after_one_year)

def profit (selling_price value_after_two_years : ℝ) : ℝ :=
  selling_price - value_after_two_years

theorem profit_made_after_two_years
  (h_present_value : present_value = 150000)
  (h_depreciation_rate : depreciation_rate = 0.22)
  (h_selling_price : selling_price = 115260) :
  profit selling_price (value_after_two_years (value_after_one_year present_value depreciation_rate) depreciation_rate) = 24000 := 
by
  sorry

end profit_made_after_two_years_l171_171570


namespace boat_goes_6_km_upstream_l171_171556

variable (speed_in_still_water : ℕ) (distance_downstream : ℕ) (time_downstream : ℕ) (effective_speed_downstream : ℕ) (speed_of_stream : ℕ)

-- Given conditions
def condition1 : Prop := speed_in_still_water = 11
def condition2 : Prop := distance_downstream = 16
def condition3 : Prop := time_downstream = 1
def condition4 : Prop := effective_speed_downstream = speed_in_still_water + speed_of_stream
def condition5 : Prop := effective_speed_downstream = 16

-- Prove that the boat goes 6 km against the stream in one hour.
theorem boat_goes_6_km_upstream : speed_of_stream = 5 →
  11 - 5 = 6 :=
by
  intros
  sorry

end boat_goes_6_km_upstream_l171_171556


namespace greatest_k_inequality_l171_171777

theorem greatest_k_inequality :
  ∃ k : ℕ, k = 13 ∧ ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a * b * c = 1 → 
  (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4) :=
sorry

end greatest_k_inequality_l171_171777


namespace maximize_net_income_l171_171316

-- Define the conditions of the problem
def bicycles := 50
def management_cost := 115

def rental_income (x : ℕ) : ℕ :=
if x ≤ 6 then bicycles * x
else (bicycles - 3 * (x - 6)) * x

def net_income (x : ℕ) : ℤ :=
rental_income x - management_cost

-- Define the domain of the function
def domain (x : ℕ) : Prop := 3 ≤ x ∧ x ≤ 20

-- Define the piecewise function for y = f(x)
def f (x : ℕ) : ℤ :=
if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
else if 6 < x ∧ x ≤ 20 then -3 * x * x + 68 * x - 115
else 0  -- Out of domain

-- The theorem that we need to prove
theorem maximize_net_income :
  (∀ x, domain x → net_income x = f x) ∧
  (∃ x, domain x ∧ (∀ y, domain y → net_income y ≤ net_income x) ∧ x = 11) :=
by
  sorry

end maximize_net_income_l171_171316


namespace frederick_final_amount_l171_171721

-- Definitions of conditions
def P : ℝ := 2000
def r : ℝ := 0.05
def n : ℕ := 18

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Theorem stating the question's answer
theorem frederick_final_amount : compound_interest P r n = 4813.24 :=
by
  sorry

end frederick_final_amount_l171_171721


namespace directrix_of_parabola_l171_171797

theorem directrix_of_parabola :
  ∀ (a h k : ℝ), (a < 0) → (∀ x, y = a * (x - h) ^ 2 + k) → (h = 0) → (k = 0) → 
  (directrix = 1 / (4 * a)) → (directrix = 1 / 4) :=
by
  sorry

end directrix_of_parabola_l171_171797


namespace solve_for_S_l171_171927

variable (D S : ℝ)
variable (h1 : D > 0)
variable (h2 : S > 0)
variable (h3 : ((0.75 * D) / 50 + (0.25 * D) / S) / D = 1 / 50)

theorem solve_for_S :
  S = 50 :=
by
  sorry

end solve_for_S_l171_171927


namespace number_of_friends_l171_171393

theorem number_of_friends (total_bill : ℝ) (discount_rate : ℝ) (paid_amount : ℝ) (n : ℝ) 
  (h_total_bill : total_bill = 400) 
  (h_discount_rate : discount_rate = 0.05)
  (h_paid_amount : paid_amount = 63.59) 
  (h_total_paid : n * paid_amount = total_bill * (1 - discount_rate)) : n = 6 := 
by
  -- proof goes here
  sorry

end number_of_friends_l171_171393


namespace binomial_sum_of_coefficients_l171_171620

theorem binomial_sum_of_coefficients (n : ℕ) (h₀ : (1 - 2)^n = 8) :
  (1 - 2)^n = -1 :=
sorry

end binomial_sum_of_coefficients_l171_171620


namespace unique_solution_l171_171965

theorem unique_solution (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  n^2 = m^4 + m^3 + m^2 + m + 1 ↔ (n, m) = (11, 3) :=
by sorry

end unique_solution_l171_171965


namespace factorization_correct_l171_171470

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by
  sorry

end factorization_correct_l171_171470


namespace rational_values_of_expressions_l171_171071

theorem rational_values_of_expressions {x : ℚ} :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by
  sorry

end rational_values_of_expressions_l171_171071


namespace at_least_one_worker_must_wait_l171_171093

/-- 
Given five workers who collectively have a salary of 1500 rubles, 
and each tape recorder costs 320 rubles, we need to prove that 
at least one worker will not be able to buy a tape recorder immediately. 
-/
theorem at_least_one_worker_must_wait 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (tape_recorder_cost : ℕ) 
  (h_workers : num_workers = 5) 
  (h_salary : total_salary = 1500) 
  (h_cost : tape_recorder_cost = 320) :
  ∀ (tape_recorders_required : ℕ), 
    tape_recorders_required = num_workers → total_salary < tape_recorder_cost * tape_recorders_required → ∃ (k : ℕ), 1 ≤ k ∧ k ≤ num_workers ∧ total_salary < k * tape_recorder_cost :=
by 
  intros tape_recorders_required h_required h_insufficient
  sorry

end at_least_one_worker_must_wait_l171_171093


namespace multiply_fractions_l171_171938

theorem multiply_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = (1 / 7) := by
  sorry

end multiply_fractions_l171_171938


namespace income_increase_by_parental_support_l171_171751

variables (a b c S : ℝ)

theorem income_increase_by_parental_support 
  (h1 : S = a + b + c)
  (h2 : 2 * a + b + c = 1.05 * S)
  (h3 : a + 2 * b + c = 1.15 * S) :
  (a + b + 2 * c) = 1.8 * S :=
sorry

end income_increase_by_parental_support_l171_171751


namespace evaluate_fraction_l171_171599

theorem evaluate_fraction (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x - 1 / y ≠ 0) :
  (y - 1 / x) / (x - 1 / y) + y / x = 2 * y / x :=
by sorry

end evaluate_fraction_l171_171599


namespace total_distance_12_hours_l171_171262

-- Define the initial conditions for the speed and distance calculation
def speed_increase : ℕ → ℕ
  | 0 => 50
  | n + 1 => speed_increase n + 2

def distance_in_hour (n : ℕ) : ℕ := speed_increase n

-- Define the total distance traveled in 12 hours
def total_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => total_distance n + distance_in_hour n

theorem total_distance_12_hours :
  total_distance 12 = 732 := by
  sorry

end total_distance_12_hours_l171_171262


namespace correct_judgments_about_f_l171_171516

-- Define the function f with its properties
variable {f : ℝ → ℝ} 

-- f is an even function
axiom even_function : ∀ x, f (-x) = f x

-- f satisfies f(x + 1) = -f(x)
axiom function_property : ∀ x, f (x + 1) = -f x

-- f is increasing on [-1, 0]
axiom increasing_on_interval : ∀ x y, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y

theorem correct_judgments_about_f :
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, f x = f (-x + 2)) ∧
  (f 2 = f 0) :=
by 
  sorry

end correct_judgments_about_f_l171_171516


namespace boys_bought_balloons_l171_171215

def initial_balloons : ℕ := 3 * 12  -- Clown initially has 3 dozen balloons, i.e., 36 balloons
def girls_balloons : ℕ := 12        -- 12 girls buy a balloon each
def balloons_remaining : ℕ := 21     -- Clown is left with 21 balloons

def boys_balloons : ℕ :=
  initial_balloons - balloons_remaining - girls_balloons

theorem boys_bought_balloons :
  boys_balloons = 3 :=
by
  sorry

end boys_bought_balloons_l171_171215


namespace hash_op_example_l171_171912

def hash_op (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem hash_op_example : hash_op 2 5 3 = 1 := 
by 
  sorry

end hash_op_example_l171_171912


namespace kira_travel_time_l171_171408

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end kira_travel_time_l171_171408


namespace esteban_exercise_each_day_l171_171836

theorem esteban_exercise_each_day (natasha_daily : ℕ) (natasha_days : ℕ) (esteban_days : ℕ) (total_hours : ℕ) :
  let total_minutes := total_hours * 60
  let natasha_total := natasha_daily * natasha_days
  let esteban_total := total_minutes - natasha_total
  esteban_days ≠ 0 →
  natasha_daily = 30 →
  natasha_days = 7 →
  esteban_days = 9 →
  total_hours = 5 →
  esteban_total / esteban_days = 10 := 
by
  intros
  sorry

end esteban_exercise_each_day_l171_171836


namespace expected_value_of_unfair_die_l171_171337

-- Define the probabilities for each face of the die.
def prob_face (n : ℕ) : ℚ :=
  if n = 8 then 5/14 else 1/14

-- Define the expected value of a roll of this die.
def expected_value : ℚ :=
  (1 / 14) * 1 + (1 / 14) * 2 + (1 / 14) * 3 + (1 / 14) * 4 + (1 / 14) * 5 + (1 / 14) * 6 + (1 / 14) * 7 + (5 / 14) * 8

-- The statement to prove: the expected value of a roll of this die is 4.857.
theorem expected_value_of_unfair_die : expected_value = 4.857 := by
  sorry

end expected_value_of_unfair_die_l171_171337


namespace min_people_like_mozart_bach_not_beethoven_l171_171963

-- Define the initial conditions
variables {n a b c : ℕ}
variables (total_people := 150)
variables (likes_mozart := 120)
variables (likes_bach := 105)
variables (likes_beethoven := 45)

theorem min_people_like_mozart_bach_not_beethoven : 
  ∃ (x : ℕ), 
    total_people = 150 ∧ 
    likes_mozart = 120 ∧ 
    likes_bach = 105 ∧ 
    likes_beethoven = 45 ∧ 
    x = (likes_mozart + likes_bach - total_people) := 
    sorry

end min_people_like_mozart_bach_not_beethoven_l171_171963


namespace number_of_buckets_l171_171033

-- Defining the conditions
def total_mackerels : ℕ := 27
def mackerels_per_bucket : ℕ := 3

-- The theorem to prove
theorem number_of_buckets :
  total_mackerels / mackerels_per_bucket = 9 :=
sorry

end number_of_buckets_l171_171033


namespace complex_division_l171_171808

-- Define i as the imaginary unit
def i : Complex := Complex.I

-- Define the problem statement to prove that 2i / (1 - i) equals -1 + i
theorem complex_division : (2 * i) / (1 - i) = -1 + i :=
by
  -- Since we are focusing on the statement, we use sorry to skip the proof
  sorry

end complex_division_l171_171808


namespace sum_of_four_interior_edges_l171_171834

-- Define the given conditions
def is_two_inch_frame (w : ℕ) := w = 2
def frame_area (A : ℕ) := A = 68
def outer_edge_length (L : ℕ) := L = 15

-- Define the inner dimensions calculation function
def inner_dimensions (outerL outerH frameW : ℕ) := 
  (outerL - 2 * frameW, outerH - 2 * frameW)

-- Define the final question in Lean 4 reflective of the equivalent proof problem
theorem sum_of_four_interior_edges (w A L y : ℕ) 
  (h1 : is_two_inch_frame w) 
  (h2 : frame_area A)
  (h3 : outer_edge_length L)
  (h4 : 15 * y - (15 - 2 * w) * (y - 2 * w) = A)
  : 2 * (15 - 2 * w) + 2 * (y - 2 * w) = 26 := 
sorry

end sum_of_four_interior_edges_l171_171834


namespace checkerboard_contains_5_black_squares_l171_171636

def is_checkerboard (x y : ℕ) : Prop := 
  x < 8 ∧ y < 8 ∧ (x + y) % 2 = 0

def contains_5_black_squares (x y n : ℕ) : Prop :=
  ∃ k l : ℕ, k ≤ n ∧ l ≤ n ∧ (x + k + y + l) % 2 = 0 ∧ k * l >= 5

theorem checkerboard_contains_5_black_squares :
  ∃ num, num = 73 ∧
  (∀ x y n, contains_5_black_squares x y n → num = 73) :=
by
  sorry

end checkerboard_contains_5_black_squares_l171_171636


namespace smallest_integer_b_gt_4_base_b_perfect_square_l171_171031

theorem smallest_integer_b_gt_4_base_b_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ n : ℕ, 2 * b + 5 = n^2 ∧ b = 10 :=
by
  sorry

end smallest_integer_b_gt_4_base_b_perfect_square_l171_171031


namespace carter_has_255_cards_l171_171961

-- Definition of the number of baseball cards Marcus has.
def marcus_cards : ℕ := 350

-- Definition of the number of more cards Marcus has than Carter.
def difference : ℕ := 95

-- Definition of the number of baseball cards Carter has.
def carter_cards : ℕ := marcus_cards - difference

-- Theorem stating that Carter has 255 baseball cards.
theorem carter_has_255_cards : carter_cards = 255 :=
sorry

end carter_has_255_cards_l171_171961


namespace largest_unattainable_sum_l171_171831

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end largest_unattainable_sum_l171_171831


namespace intersection_M_N_l171_171588

-- Definitions based on the conditions
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

-- Theorem asserting the intersection of sets M and N
theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} := 
by
  sorry

end intersection_M_N_l171_171588


namespace negation_of_P_l171_171037

open Classical

variable (x : ℝ)

def P (x : ℝ) : Prop :=
  x^2 + 2 > 2 * x

theorem negation_of_P : (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_P_l171_171037


namespace gnollish_valid_sentences_l171_171536

def valid_sentences_count : ℕ :=
  let words := ["splargh", "glumph", "amr", "krack"]
  let total_words := 4
  let total_sentences := total_words ^ 3
  let invalid_splargh_glumph := 2 * total_words
  let invalid_amr_krack := 2 * total_words
  let total_invalid := invalid_splargh_glumph + invalid_amr_krack
  total_sentences - total_invalid

theorem gnollish_valid_sentences : valid_sentences_count = 48 :=
by
  sorry

end gnollish_valid_sentences_l171_171536


namespace solve_problem_l171_171572

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end solve_problem_l171_171572


namespace part1_part2_l171_171855

noncomputable def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x < 4 ↔ -1 < x ∧ x < (5:ℝ)/3 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l171_171855


namespace grill_runtime_l171_171165

theorem grill_runtime
    (burn_rate : ℕ)
    (burn_time : ℕ)
    (bags : ℕ)
    (coals_per_bag : ℕ)
    (total_burnt_coals : ℕ)
    (total_time : ℕ)
    (h1 : burn_rate = 15)
    (h2 : burn_time = 20)
    (h3 : bags = 3)
    (h4 : coals_per_bag = 60)
    (h5 : total_burnt_coals = bags * coals_per_bag)
    (h6 : total_time = (total_burnt_coals / burn_rate) * burn_time) :
    total_time = 240 :=
by sorry

end grill_runtime_l171_171165


namespace total_balls_in_bag_l171_171204

theorem total_balls_in_bag (R G B T : ℕ) 
  (hR : R = 907) 
  (hRatio : 15 * T = 15 * R + 13 * R + 17 * R)
  : T = 2721 :=
sorry

end total_balls_in_bag_l171_171204


namespace rectangle_area_l171_171134

theorem rectangle_area (P l w : ℕ) (h_perimeter: 2 * l + 2 * w = 60) (h_aspect: l = 3 * w / 2) : l * w = 216 :=
sorry

end rectangle_area_l171_171134


namespace max_profit_l171_171974

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end max_profit_l171_171974


namespace min_value_arithmetic_sequence_l171_171714

theorem min_value_arithmetic_sequence (d : ℝ) (n : ℕ) (hd : d ≠ 0) (a1 : ℝ) (ha1 : a1 = 1)
(geo : (1 + 2 * d)^2 = 1 + 12 * d) (Sn : ℝ) (hSn : Sn = n^2) (an : ℝ) (han : an = 2 * n - 1) :
  ∀ (n : ℕ), n > 0 → (2 * Sn + 8) / (an + 3) ≥ 5 / 2 :=
by sorry

end min_value_arithmetic_sequence_l171_171714


namespace find_total_amount_l171_171767

noncomputable def total_amount (a b c : ℕ) : Prop :=
  a = 3 * b ∧ b = c + 25 ∧ b = 134 ∧ a + b + c = 645

theorem find_total_amount : ∃ a b c, total_amount a b c :=
by
  sorry

end find_total_amount_l171_171767


namespace quotient_is_seven_l171_171585

def dividend : ℕ := 22
def divisor : ℕ := 3
def remainder : ℕ := 1

theorem quotient_is_seven : ∃ quotient : ℕ, dividend = (divisor * quotient) + remainder ∧ quotient = 7 := by
  sorry

end quotient_is_seven_l171_171585


namespace minimum_questions_to_find_number_l171_171391

theorem minimum_questions_to_find_number (n : ℕ) (h : n ≤ 2020) :
  ∃ m, m = 64 ∧ (∀ (strategy : ℕ → ℕ), ∃ questions : ℕ, questions ≤ m ∧ (strategy questions = n)) :=
sorry

end minimum_questions_to_find_number_l171_171391


namespace max_volume_48cm_square_l171_171212

def volume_of_box (x : ℝ) := x * (48 - 2 * x)^2

theorem max_volume_48cm_square : 
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → volume_of_box x ≥ volume_of_box y) ∧ x = 8 :=
sorry

end max_volume_48cm_square_l171_171212


namespace table_length_is_77_l171_171167

theorem table_length_is_77 :
  ∀ (x : ℕ), (∀ (sheets: ℕ), sheets = 72 → x = (5 + sheets)) → x = 77 :=
by {
  sorry
}

end table_length_is_77_l171_171167


namespace log2_ratio_squared_l171_171723

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log2_ratio_squared :
  ∀ (x y : ℝ), x ≠ 1 → y ≠ 1 → log_base 2 x = log_base y 25 → x * y = 81
  → (log_base 2 (x / y))^2 = 5.11 :=
by
  intros x y hx hy hlog hxy
  sorry

end log2_ratio_squared_l171_171723


namespace samia_walking_distance_l171_171433

theorem samia_walking_distance
  (speed_bike : ℝ)
  (speed_walk : ℝ)
  (total_time : ℝ) 
  (fraction_bike : ℝ) 
  (d : ℝ)
  (walking_distance : ℝ) :
  speed_bike = 15 ∧ 
  speed_walk = 4 ∧ 
  total_time = 1 ∧ 
  fraction_bike = 2/3 ∧ 
  walking_distance = (1/3) * d ∧ 
  (53 * d / 180 = total_time) → 
  walking_distance = 1.1 := 
by 
  sorry

end samia_walking_distance_l171_171433


namespace sum_of_reciprocals_of_roots_l171_171883

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (hroots : ∀ x, (x = p ∨ x = q ∨ x = r) ↔ (30*x^3 - 50*x^2 + 22*x - 1 = 0)) 
  (h0 : 0 < p ∧ p < 1) (h1 : 0 < q ∧ q < 1) (h2 : 0 < r ∧ r < 1) 
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - r)) = 12 := 
by 
  sorry

end sum_of_reciprocals_of_roots_l171_171883


namespace find_perp_line_eq_l171_171898

-- Line equation in the standard form
def line_eq (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Equation of the required line that is perpendicular to the given line and has the same y-intercept
def perp_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 16 = 0

theorem find_perp_line_eq (x y : ℝ) :
  (∃ k : ℝ, line_eq 0 k ∧ perp_line_eq 0 k) →
  (∃ a b c : ℝ, perp_line_eq a b) :=
by
  sorry

end find_perp_line_eq_l171_171898


namespace average_speed_comparison_l171_171673

variables (u v : ℝ) (hu : u > 0) (hv : v > 0)

theorem average_speed_comparison (x y : ℝ) 
  (hx : x = 2 * u * v / (u + v)) 
  (hy : y = (u + v) / 2) : x ≤ y := 
sorry

end average_speed_comparison_l171_171673


namespace angle_difference_l171_171596

theorem angle_difference (A B : ℝ) 
  (h1 : A = 85) 
  (h2 : A + B = 180) : B - A = 10 := 
by sorry

end angle_difference_l171_171596


namespace part1_part2_l171_171907

-- Definitions
def A (x : ℝ) : Prop := (x + 2) / (x - 3 / 2) < 0
def B (x : ℝ) (m : ℝ) : Prop := x^2 - (m + 1) * x + m ≤ 0

-- Part (1): when m = 2, find A ∪ B
theorem part1 :
  (∀ x, A x ∨ B x 2) ↔ ∀ x, -2 < x ∧ x ≤ 2 := sorry

-- Part (2): find the range of real number m
theorem part2 :
  (∀ x, A x → B x m) ↔ (-2 < m ∧ m < 3 / 2) := sorry

end part1_part2_l171_171907


namespace find_c_l171_171508

theorem find_c (c : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (hf : ∀ x, f x = 2 / (3 * x + c))
  (hfinv : ∀ x, f_inv x = (2 - 3 * x) / (3 * x)) :
  c = 3 :=
by
  sorry

end find_c_l171_171508


namespace probability_black_ball_l171_171662

variable (total_balls : ℕ)
variable (red_balls : ℕ)
variable (white_probability : ℝ)

def number_of_balls : Prop := total_balls = 100
def red_ball_count : Prop := red_balls = 45
def white_ball_probability : Prop := white_probability = 0.23

theorem probability_black_ball 
  (h1 : number_of_balls total_balls)
  (h2 : red_ball_count red_balls)
  (h3 : white_ball_probability white_probability) :
  let white_balls := white_probability * total_balls 
  let black_balls := total_balls - red_balls - white_balls
  let black_ball_prob := black_balls / total_balls
  black_ball_prob = 0.32 :=
sorry

end probability_black_ball_l171_171662


namespace banana_permutations_l171_171670

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l171_171670


namespace find_multiplier_l171_171781

theorem find_multiplier (x : ℕ) (h₁ : 3 * x = (26 - x) + 26) (h₂ : x = 13) : 3 = 3 := 
by 
  sorry

end find_multiplier_l171_171781


namespace spherical_to_rectangular_coords_l171_171485

theorem spherical_to_rectangular_coords
  (ρ θ φ : ℝ)
  (hρ : ρ = 6)
  (hθ : θ = 7 * Real.pi / 4)
  (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = -3 * Real.sqrt 6 ∧ y = -3 * Real.sqrt 6 ∧ z = 3 :=
by
  sorry

end spherical_to_rectangular_coords_l171_171485


namespace original_rent_l171_171715

theorem original_rent {avg_rent_before avg_rent_after : ℝ} (total_before total_after increase_percentage diff_increase : ℝ) :
  avg_rent_before = 800 → 
  avg_rent_after = 880 → 
  total_before = 4 * avg_rent_before → 
  total_after = 4 * avg_rent_after → 
  diff_increase = total_after - total_before → 
  increase_percentage = 0.20 → 
  diff_increase = increase_percentage * R → 
  R = 1600 :=
by sorry

end original_rent_l171_171715


namespace value_x_when_y2_l171_171254

theorem value_x_when_y2 (x : ℝ) (h1 : ∃ (x : ℝ), y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 := by
  sorry

end value_x_when_y2_l171_171254


namespace congruence_solution_count_l171_171562

theorem congruence_solution_count :
  ∃! x : ℕ, x < 50 ∧ x + 20 ≡ 75 [MOD 43] := 
by
  sorry

end congruence_solution_count_l171_171562


namespace fireworks_display_l171_171610

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end fireworks_display_l171_171610


namespace how_many_ducks_did_john_buy_l171_171201

def cost_price_per_duck : ℕ := 10
def weight_per_duck : ℕ := 4
def selling_price_per_pound : ℕ := 5
def profit : ℕ := 300

theorem how_many_ducks_did_john_buy (D : ℕ) (h : 10 * D - 10 * D + 10 * D = profit) : D = 30 :=
by 
  sorry

end how_many_ducks_did_john_buy_l171_171201


namespace find_a_l171_171611

theorem find_a (a : ℝ) : (∃ (p : ℝ × ℝ), p = (3, -9) ∧ (3 * a * p.1 + (2 * a + 1) * p.2 = 3 * a + 3)) → a = -1 :=
by
  sorry

end find_a_l171_171611


namespace tan_subtraction_l171_171783

theorem tan_subtraction (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 :=
by
  sorry

end tan_subtraction_l171_171783


namespace mean_of_xyz_l171_171386

theorem mean_of_xyz (x y z : ℝ) (seven_mean : ℝ)
  (h1 : seven_mean = 45)
  (h2 : (7 * seven_mean + x + y + z) / 10 = 58) :
  (x + y + z) / 3 = 265 / 3 :=
by
  sorry

end mean_of_xyz_l171_171386


namespace exists_monotonic_subsequence_l171_171698

open Function -- For function related definitions
open Finset -- For finite set operations

-- Defining the theorem with the given conditions and the goal to be proved
theorem exists_monotonic_subsequence (a : Fin 10 → ℝ) (h : ∀ i j : Fin 10, i ≠ j → a i ≠ a j) :
  ∃ (i1 i2 i3 i4 : Fin 10), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
  ((a i1 < a i2 ∧ a i2 < a i3 ∧ a i3 < a i4) ∨ (a i1 > a i2 ∧ a i2 > a i3 ∧ a i3 > a i4)) :=
by
  sorry -- Proof is omitted as per the instructions

end exists_monotonic_subsequence_l171_171698


namespace range_of_a_l171_171192

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → (x - 1) ^ 2 < Real.log x / Real.log a) → a ∈ Set.Ioc 1 2 :=
by
  sorry

end range_of_a_l171_171192


namespace combined_weight_of_boxes_l171_171207

def weight_box1 : ℝ := 2
def weight_box2 : ℝ := 11
def weight_box3 : ℝ := 5

theorem combined_weight_of_boxes : weight_box1 + weight_box2 + weight_box3 = 18 := by
  sorry

end combined_weight_of_boxes_l171_171207


namespace find_extreme_value_number_of_zeros_l171_171626

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + (a - 2) * x - Real.log x

-- Math proof problem I
theorem find_extreme_value (a : ℝ) (h : (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f a x > f a 1)) : a = 1 := 
sorry

-- Math proof problem II
theorem number_of_zeros (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := 
sorry

end find_extreme_value_number_of_zeros_l171_171626


namespace grace_age_l171_171188

theorem grace_age 
  (H : ℕ) 
  (I : ℕ) 
  (J : ℕ) 
  (G : ℕ)
  (h1 : H = I - 5)
  (h2 : I = J + 7)
  (h3 : G = 2 * J)
  (h4 : H = 18) : 
  G = 32 := 
sorry

end grace_age_l171_171188


namespace solve_equation1_solve_equation2_l171_171027

theorem solve_equation1 :
  ∀ x : ℝ, ((x-1) * (x-1) = 3 * (x-1)) ↔ (x = 1 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_equation2 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by
  intro x
  sorry

end solve_equation1_solve_equation2_l171_171027


namespace share_price_increase_l171_171850

theorem share_price_increase
  (P : ℝ)
  -- At the end of the first quarter, the share price was 20% higher than at the beginning of the year.
  (end_of_first_quarter : ℝ := 1.20 * P)
  -- The percent increase from the end of the first quarter to the end of the second quarter was 25%.
  (percent_increase_second_quarter : ℝ := 0.25)
  -- At the end of the second quarter, the share price
  (end_of_second_quarter : ℝ := end_of_first_quarter + percent_increase_second_quarter * end_of_first_quarter) :
  -- What is the percent increase in share price at the end of the second quarter compared to the beginning of the year?
  end_of_second_quarter = 1.50 * P :=
by
  sorry

end share_price_increase_l171_171850


namespace crayons_left_is_4_l171_171846

-- Define initial number of crayons in the drawer
def initial_crayons : Nat := 7

-- Define number of crayons Mary took out
def taken_by_mary : Nat := 3

-- Define the number of crayons left in the drawer
def crayons_left (initial : Nat) (taken : Nat) : Nat :=
  initial - taken

-- Prove the number of crayons left in the drawer is 4
theorem crayons_left_is_4 : crayons_left initial_crayons taken_by_mary = 4 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end crayons_left_is_4_l171_171846


namespace jenna_interest_l171_171903

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

def interest_earned (P r : ℝ) (n : ℕ) : ℝ :=
  compound_interest P r n - P

theorem jenna_interest :
  interest_earned 1500 0.05 5 = 414.42 :=
by
  sorry

end jenna_interest_l171_171903


namespace total_flowers_l171_171547

theorem total_flowers (initial_rosas_flowers andre_gifted_flowers : ℝ) 
  (h1 : initial_rosas_flowers = 67.0) 
  (h2 : andre_gifted_flowers = 90.0) : 
  initial_rosas_flowers + andre_gifted_flowers = 157.0 :=
  by
  sorry

end total_flowers_l171_171547


namespace tan_difference_l171_171159

variable (α β : ℝ)
variable (tan_α : ℝ := 3)
variable (tan_β : ℝ := 4 / 3)

theorem tan_difference (h₁ : Real.tan α = tan_α) (h₂ : Real.tan β = tan_β) : 
  Real.tan (α - β) = (tan_α - tan_β) / (1 + tan_α * tan_β) := by
  sorry

end tan_difference_l171_171159


namespace average_score_of_class_l171_171708

-- Definitions based on the conditions
def class_size : ℕ := 20
def group1_size : ℕ := 10
def group2_size : ℕ := 10
def group1_avg_score : ℕ := 80
def group2_avg_score : ℕ := 60

-- Average score of the whole class
theorem average_score_of_class : 
  (group1_size * group1_avg_score + group2_size * group2_avg_score) / class_size = 70 := 
by sorry

end average_score_of_class_l171_171708


namespace y2_over_x2_plus_x2_over_y2_eq_9_over_4_l171_171225

theorem y2_over_x2_plus_x2_over_y2_eq_9_over_4 (x y : ℝ) 
  (h : (1 / x) - (1 / (2 * y)) = (1 / (2 * x + y))) : 
  (y^2 / x^2) + (x^2 / y^2) = 9 / 4 := 
by 
  sorry

end y2_over_x2_plus_x2_over_y2_eq_9_over_4_l171_171225


namespace estimate_red_balls_l171_171370

-- Define the conditions in Lean 4
def total_balls : ℕ := 15
def freq_red_ball : ℝ := 0.4

-- Define the proof statement without proving it
theorem estimate_red_balls (x : ℕ) 
  (h1 : x ≤ total_balls) 
  (h2 : ∃ (p : ℝ), p = x / total_balls ∧ p = freq_red_ball) :
  x = 6 :=
sorry

end estimate_red_balls_l171_171370


namespace jesse_max_correct_answers_l171_171052

theorem jesse_max_correct_answers :
  ∃ a b c : ℕ, a + b + c = 60 ∧ 5 * a - 2 * c = 150 ∧ a ≤ 38 :=
sorry

end jesse_max_correct_answers_l171_171052


namespace new_apples_grew_l171_171957

-- The number of apples originally on the tree.
def original_apples : ℕ := 11

-- The number of apples picked by Rachel.
def picked_apples : ℕ := 7

-- The number of apples currently on the tree.
def current_apples : ℕ := 6

-- The number of apples left on the tree after picking.
def remaining_apples : ℕ := original_apples - picked_apples

-- The number of new apples that grew on the tree.
def new_apples : ℕ := current_apples - remaining_apples

-- The theorem we need to prove.
theorem new_apples_grew :
  new_apples = 2 := by
    sorry

end new_apples_grew_l171_171957


namespace find_base_of_exponential_l171_171693

theorem find_base_of_exponential (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a ≠ 1) 
  (h₃ : a ^ 2 = 1 / 16) : 
  a = 1 / 4 := 
sorry

end find_base_of_exponential_l171_171693


namespace sufficient_not_necessary_condition_l171_171716

variable (a : ℝ)

theorem sufficient_not_necessary_condition :
  (1 < a ∧ a < 2) → (a^2 - 3 * a ≤ 0) := by
  intro h
  sorry

end sufficient_not_necessary_condition_l171_171716


namespace equation_solution_l171_171381

theorem equation_solution (x y z : ℕ) :
  x^2 + y^2 = 2^z ↔ ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 1 := 
sorry

end equation_solution_l171_171381


namespace find_possible_y_values_l171_171775

noncomputable def validYValues (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) : Set ℝ :=
  { y | y = (x - 3)^2 * (x + 4) / (2 * x - 4) }

theorem find_possible_y_values (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) :
  validYValues x hx = {39, 6} :=
sorry

end find_possible_y_values_l171_171775


namespace apples_needed_l171_171654

-- Define a simple equivalence relation between the weights of oranges and apples.
def weight_equivalent (oranges apples : ℕ) : Prop :=
  8 * apples = 6 * oranges
  
-- State the main theorem based on the given conditions
theorem apples_needed (oranges_count : ℕ) (h : weight_equivalent 1 1) : oranges_count = 32 → ∃ apples_count, apples_count = 24 :=
by
  sorry

end apples_needed_l171_171654


namespace range_alpha_sub_beta_l171_171128

theorem range_alpha_sub_beta (α β : ℝ) (h₁ : -π/2 < α) (h₂ : α < β) (h₃ : β < π/2) : -π < α - β ∧ α - β < 0 := by
  sorry

end range_alpha_sub_beta_l171_171128


namespace stones_in_pile_l171_171285

theorem stones_in_pile (initial_stones : ℕ) (final_stones_A : ℕ) (final_stones_B_min final_stones_B_max final_stones_B : ℕ) (operations : ℕ) :
  initial_stones = 2006 ∧ final_stones_A = 1990 ∧ final_stones_B_min = 2080 ∧ final_stones_B_max = 2100 ∧ operations < 20 ∧ (final_stones_B_min ≤ final_stones_B ∧ final_stones_B ≤ final_stones_B_max) 
  → final_stones_B = 2090 :=
by
  sorry

end stones_in_pile_l171_171285


namespace max_sum_abc_l171_171569

theorem max_sum_abc
  (a b c : ℤ)
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (hA1 : A = (1/7 : ℚ) • ![![(-5 : ℚ), a], ![b, c]])
  (hA2 : A * A = 2 • (1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  a + b + c ≤ 79 :=
by
  sorry

end max_sum_abc_l171_171569


namespace circle_area_difference_l171_171472

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (A1 A2 diff : ℝ) 
  (hr1 : r1 = 30)
  (hd2 : 2 * r2 = 30)
  (hA1 : A1 = π * r1^2)
  (hA2 : A2 = π * r2^2)
  (hdiff : diff = A1 - A2) :
  diff = 675 * π :=
by 
  sorry

end circle_area_difference_l171_171472


namespace ratio_of_volumes_l171_171050

theorem ratio_of_volumes (rC hC rD hD : ℝ) (h1 : rC = 10) (h2 : hC = 25) (h3 : rD = 25) (h4 : hD = 10) : 
  (1/3 * Real.pi * rC^2 * hC) / (1/3 * Real.pi * rD^2 * hD) = 2 / 5 :=
by
  sorry

end ratio_of_volumes_l171_171050


namespace number_of_days_worked_l171_171203

-- Definitions based on the given conditions and question
def total_hours_worked : ℕ := 15
def hours_worked_each_day : ℕ := 3

-- The statement we need to prove:
theorem number_of_days_worked : 
  (total_hours_worked / hours_worked_each_day) = 5 :=
by
  sorry

end number_of_days_worked_l171_171203


namespace max_value_of_f_f_lt_x3_minus_2x2_l171_171287

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + Real.log x + b

theorem max_value_of_f (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) :
  f a b (Real.sqrt 2 / 2) = - (3 + 2 * Real.log 2) / 4 := by
  sorry

theorem f_lt_x3_minus_2x2 (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) (x : ℝ) (hx : 0 < x) :
  f a b x < x^3 - 2 * x^2 := by
  sorry

end max_value_of_f_f_lt_x3_minus_2x2_l171_171287


namespace exists_zero_in_interval_l171_171308

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

theorem exists_zero_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- This is just the Lean statement, no proof is provided
  sorry

end exists_zero_in_interval_l171_171308


namespace value_of_expression_l171_171108

theorem value_of_expression (a : ℝ) (h : a = 1/2) : 
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end value_of_expression_l171_171108


namespace range_of_a_l171_171247

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 + 2 * (2 - a) * x + 4 ≥ 0) → (-2 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l171_171247


namespace ratio_sheila_purity_l171_171025

theorem ratio_sheila_purity (rose_share : ℕ) (total_rent : ℕ) (purity_share : ℕ) (sheila_share : ℕ) 
  (h1 : rose_share = 1800) 
  (h2 : total_rent = 5400) 
  (h3 : rose_share = 3 * purity_share)
  (h4 : total_rent = purity_share + rose_share + sheila_share) : 
  sheila_share / purity_share = 5 :=
by
  -- Proof will be here
  sorry

end ratio_sheila_purity_l171_171025


namespace min_ball_count_required_l171_171875

def is_valid_ball_count (n : ℕ) : Prop :=
  n >= 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_list (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nthLe i sorry ≠ l.nthLe j sorry

def valid_ball_counts_list (l : List ℕ) : Prop :=
  (l.length = 10) ∧ distinct_list l ∧ (∀ n ∈ l, is_valid_ball_count n)

theorem min_ball_count_required : ∃ l, valid_ball_counts_list l ∧ l.sum = 174 := sorry

end min_ball_count_required_l171_171875


namespace eval_expr_l171_171362

namespace ProofProblem

variables (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d = a + b + c)

theorem eval_expr :
  d = a + b + c →
  (a^3 + b^3 + c^3 - 3 * a * b * c) / (a * b * c) = (d * (a^2 + b^2 + c^2 - a * b - a * c - b * c)) / (a * b * c) :=
by
  intros hd
  sorry

end ProofProblem

end eval_expr_l171_171362


namespace five_fold_function_application_l171_171980

def f (x : ℤ) : ℤ :=
if x ≥ 0 then -x^2 + 1 else x + 9

theorem five_fold_function_application : f (f (f (f (f 2)))) = -17 :=
by
  sorry

end five_fold_function_application_l171_171980


namespace area_ratio_of_squares_l171_171169

theorem area_ratio_of_squares (hA : ∃ sA : ℕ, 4 * sA = 16)
                             (hB : ∃ sB : ℕ, 4 * sB = 20)
                             (hC : ∃ sC : ℕ, 4 * sC = 40) :
  (∃ aB aC : ℕ, aB = sB * sB ∧ aC = sC * sC ∧ aB * 4 = aC) := by
  sorry

end area_ratio_of_squares_l171_171169


namespace complex_number_in_second_quadrant_l171_171011

open Complex

theorem complex_number_in_second_quadrant (z : ℂ) :
  (Complex.abs z = Real.sqrt 7) →
  (z.re < 0 ∧ z.im > 0) →
  z = -2 + Real.sqrt 3 * Complex.I :=
by
  intros h1 h2
  sorry

end complex_number_in_second_quadrant_l171_171011


namespace find_y_when_x_4_l171_171223

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l171_171223


namespace parabola_focus_l171_171595

-- Definitions and conditions from the original problem
def parabola_eq (x y : ℝ) : Prop := x^2 = (1/2) * y 

-- Define the problem to prove the coordinates of the focus
theorem parabola_focus (x y : ℝ) (h : parabola_eq x y) : (x = 0 ∧ y = 1/8) :=
sorry

end parabola_focus_l171_171595


namespace economy_value_after_two_years_l171_171312

/--
Given an initial amount A₀ = 3200,
that increases annually by 1/8th of itself,
with an inflation rate of 3% in the first year and 4% in the second year,
prove that the value of the amount after two years is 3771.36
-/
theorem economy_value_after_two_years :
  let A₀ := 3200 
  let increase_rate := 1 / 8
  let inflation_rate_year_1 := 0.03
  let inflation_rate_year_2 := 0.04
  let A₁ := A₀ * (1 + increase_rate)
  let V₁ := A₁ * (1 - inflation_rate_year_1)
  let A₂ := V₁ * (1 + increase_rate)
  let V₂ := A₂ * (1 - inflation_rate_year_2)
  V₂ = 3771.36 :=
by
  simp only []
  sorry

end economy_value_after_two_years_l171_171312


namespace compute_expression_l171_171350

theorem compute_expression : (3 + 5) ^ 2 + (3 ^ 2 + 5 ^ 2) = 98 := by
  sorry

end compute_expression_l171_171350


namespace part1_part2_part3_l171_171862

variables (a b c : ℤ)
-- Condition: For all integer values of x, (ax^2 + bx + c) is a square number 
def quadratic_is_square_for_any_x (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

-- Question (1): Prove that 2a, 2b, c are all integers
theorem part1 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ m n : ℤ, 2 * a = m ∧ 2 * b = n ∧ ∃ k₁ : ℤ, c = k₁ :=
sorry

-- Question (2): Prove that a, b, c are all integers, and c is a square number
theorem part2 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2 :=
sorry

-- Question (3): Prove that if (2) holds, it does not necessarily mean that 
-- for all integer values of x, (ax^2 + bx + c) is always a square number.
theorem part3 (a b c : ℤ) (h : ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2) : 
  ¬ quadratic_is_square_for_any_x a b c :=
sorry

end part1_part2_part3_l171_171862


namespace pencils_per_row_l171_171127

def total_pencils : ℕ := 32
def rows : ℕ := 4

theorem pencils_per_row : total_pencils / rows = 8 := by
  sorry

end pencils_per_row_l171_171127


namespace simplify_expression_l171_171266

theorem simplify_expression (x : ℝ) : 3 * x + 4 - x + 8 = 2 * x + 12 :=
by
  sorry

end simplify_expression_l171_171266


namespace profit_benny_wants_to_make_l171_171809

noncomputable def pumpkin_pies : ℕ := 10
noncomputable def cherry_pies : ℕ := 12
noncomputable def cost_pumpkin_pie : ℝ := 3
noncomputable def cost_cherry_pie : ℝ := 5
noncomputable def price_per_pie : ℝ := 5

theorem profit_benny_wants_to_make : 5 * (pumpkin_pies + cherry_pies) - (pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) = 20 :=
by
  sorry

end profit_benny_wants_to_make_l171_171809


namespace surface_area_is_726_l171_171035

def edge_length : ℝ := 11

def surface_area_of_cube (e : ℝ) : ℝ := 6 * (e * e)

theorem surface_area_is_726 (h : edge_length = 11) : surface_area_of_cube edge_length = 726 := by
  sorry

end surface_area_is_726_l171_171035


namespace total_pieces_of_gum_l171_171622

theorem total_pieces_of_gum (packages pieces_per_package : ℕ) 
  (h_packages : packages = 9)
  (h_pieces_per_package : pieces_per_package = 15) : 
  packages * pieces_per_package = 135 := by
  subst h_packages
  subst h_pieces_per_package
  exact Nat.mul_comm 9 15 ▸ rfl

end total_pieces_of_gum_l171_171622


namespace volume_of_rectangular_solid_l171_171541

theorem volume_of_rectangular_solid (a b c : ℝ) (h1 : a * b = Real.sqrt 2) (h2 : b * c = Real.sqrt 3) (h3 : c * a = Real.sqrt 6) : a * b * c = Real.sqrt 6 :=
sorry

end volume_of_rectangular_solid_l171_171541


namespace largest_int_less_than_100_with_remainder_5_l171_171992

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end largest_int_less_than_100_with_remainder_5_l171_171992


namespace strictly_increasing_range_l171_171296

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x + 1 else a ^ x

theorem strictly_increasing_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 < a ∧ a ≤ 2) :=
sorry

end strictly_increasing_range_l171_171296


namespace jake_total_work_hours_l171_171495

def initial_debt_A := 150
def payment_A := 60
def hourly_rate_A := 15
def remaining_debt_A := initial_debt_A - payment_A
def hours_to_work_A := remaining_debt_A / hourly_rate_A

def initial_debt_B := 200
def payment_B := 80
def hourly_rate_B := 20
def remaining_debt_B := initial_debt_B - payment_B
def hours_to_work_B := remaining_debt_B / hourly_rate_B

def initial_debt_C := 250
def payment_C := 100
def hourly_rate_C := 25
def remaining_debt_C := initial_debt_C - payment_C
def hours_to_work_C := remaining_debt_C / hourly_rate_C

def total_hours_to_work := hours_to_work_A + hours_to_work_B + hours_to_work_C

theorem jake_total_work_hours :
  total_hours_to_work = 18 :=
sorry

end jake_total_work_hours_l171_171495


namespace chocolate_candy_pieces_l171_171324

-- Define the initial number of boxes and the boxes given away
def initial_boxes : Nat := 12
def boxes_given : Nat := 7

-- Define the number of remaining boxes
def remaining_boxes := initial_boxes - boxes_given

-- Define the number of pieces per box
def pieces_per_box : Nat := 6

-- Calculate the total pieces Tom still has
def total_pieces := remaining_boxes * pieces_per_box

-- State the theorem
theorem chocolate_candy_pieces : total_pieces = 30 :=
by
  -- proof steps would go here
  sorry

end chocolate_candy_pieces_l171_171324


namespace ten_men_ten_boys_work_time_l171_171474

theorem ten_men_ten_boys_work_time :
  (∀ (total_work : ℝ) (man_work_rate boy_work_rate : ℝ),
    15 * 10 * man_work_rate = total_work ∧
    20 * 15 * boy_work_rate = total_work →
    (10 * man_work_rate + 10 * boy_work_rate) * 10 = total_work) :=
by
  sorry

end ten_men_ten_boys_work_time_l171_171474


namespace power_mod_l171_171997

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end power_mod_l171_171997


namespace smallest_integer_20p_larger_and_19p_smaller_l171_171934

theorem smallest_integer_20p_larger_and_19p_smaller :
  ∃ (N x y : ℕ), N = 162 ∧ N = 12 / 10 * x ∧ N = 81 / 100 * y :=
by
  sorry

end smallest_integer_20p_larger_and_19p_smaller_l171_171934


namespace volume_of_region_l171_171255

theorem volume_of_region :
    ∀ (x y z : ℝ), 
    |x - y + z| + |x - y - z| ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 
    → true := by
    sorry

end volume_of_region_l171_171255


namespace recording_incorrect_l171_171684

-- Definitions for given conditions
def qualifying_standard : ℝ := 1.5
def xiao_ming_jump : ℝ := 1.95
def xiao_liang_jump : ℝ := 1.23
def xiao_ming_recording : ℝ := 0.45
def xiao_liang_recording : ℝ := -0.23

-- The proof statement to verify the correctness of the recordings
theorem recording_incorrect :
  (xiao_ming_jump - qualifying_standard = xiao_ming_recording) ∧ 
  (xiao_liang_jump - qualifying_standard ≠ xiao_liang_recording) :=
by
  sorry

end recording_incorrect_l171_171684


namespace fruit_count_correct_l171_171268

def george_oranges := 45
def amelia_oranges := george_oranges - 18
def amelia_apples := 15
def george_apples := amelia_apples + 5

def olivia_orange_rate := 3
def olivia_apple_rate := 2
def olivia_minutes := 30
def olivia_cycle_minutes := 5
def olivia_cycles := olivia_minutes / olivia_cycle_minutes
def olivia_oranges := olivia_orange_rate * olivia_cycles
def olivia_apples := olivia_apple_rate * olivia_cycles

def total_oranges := george_oranges + amelia_oranges + olivia_oranges
def total_apples := george_apples + amelia_apples + olivia_apples
def total_fruits := total_oranges + total_apples

theorem fruit_count_correct : total_fruits = 137 := by
  sorry

end fruit_count_correct_l171_171268


namespace birth_rate_calculation_l171_171574

theorem birth_rate_calculation (D : ℕ) (G : ℕ) (P : ℕ) (NetGrowth : ℕ) (B : ℕ) (h1 : D = 16) (h2 : G = 12) (h3 : P = 3000) (h4 : NetGrowth = G * P / 100) (h5 : NetGrowth = B - D) : B = 52 := by
  sorry

end birth_rate_calculation_l171_171574


namespace sum_of_cubes_eq_neg_27_l171_171354

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l171_171354


namespace millennium_run_time_l171_171798

theorem millennium_run_time (M A B : ℕ) (h1 : B = 100) (h2 : B = A + 10) (h3 : A = M - 30) : M = 120 := by
  sorry

end millennium_run_time_l171_171798


namespace matching_function_l171_171930

open Real

def table_data : List (ℝ × ℝ) := [(1, 4), (2, 2), (4, 1)]

theorem matching_function :
  ∃ a b c : ℝ, a > 0 ∧ 
               (∀ x y, (x, y) ∈ table_data → y = a * x^2 + b * x + c) := 
sorry

end matching_function_l171_171930


namespace nat_number_36_sum_of_digits_l171_171699

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l171_171699


namespace range_of_a_when_min_f_ge_neg_a_l171_171919

noncomputable def f (a x : ℝ) := a * Real.log x + 2 * x

theorem range_of_a_when_min_f_ge_neg_a (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x > 0, f a x ≥ -a) :
  -2 ≤ a ∧ a < 0 :=
sorry

end range_of_a_when_min_f_ge_neg_a_l171_171919


namespace smallest_number_of_people_l171_171608

open Nat

theorem smallest_number_of_people (x : ℕ) :
  (∃ x, x % 18 = 0 ∧ x % 50 = 0 ∧
  (∀ y, y % 18 = 0 ∧ y % 50 = 0 → x ≤ y)) → x = 450 :=
by
  sorry

end smallest_number_of_people_l171_171608


namespace batsman_average_excluding_highest_and_lowest_l171_171539

theorem batsman_average_excluding_highest_and_lowest (average : ℝ) (innings : ℕ) (highest_score : ℝ) (score_difference : ℝ) :
  average = 63 →
  innings = 46 →
  highest_score = 248 →
  score_difference = 150 →
  (average * innings - highest_score - (highest_score - score_difference)) / (innings - 2) = 58 :=
by
  intros h_average h_innings h_highest h_difference
  simp [h_average, h_innings, h_highest, h_difference]
  -- Here the detailed steps from the solution would come in to verify the simplification
  sorry

end batsman_average_excluding_highest_and_lowest_l171_171539


namespace no_solution_exists_l171_171039

theorem no_solution_exists : ¬ ∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by sorry

end no_solution_exists_l171_171039


namespace arrangement_count_l171_171277

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l171_171277


namespace siamese_cats_initial_l171_171653

theorem siamese_cats_initial (S : ℕ) (h1 : 20 + S - 20 = 12) : S = 12 :=
by
  sorry

end siamese_cats_initial_l171_171653


namespace find_number_l171_171810

theorem find_number (x : ℝ) (h : x * 2 + (12 + 4) * (1/8) = 602) : x = 300 :=
by
  sorry

end find_number_l171_171810


namespace find_ks_l171_171404

theorem find_ks (n : ℕ) (h_pos : 0 < n) :
  ∀ k, k ∈ (Finset.range (2 * n * n + 1)).erase 0 ↔ (n^2 - n + 1 ≤ k ∧ k ≤ n^2) ∨ (2*n ∣ k ∧ k ≥ n^2 - n + 1) :=
sorry

end find_ks_l171_171404


namespace students_like_basketball_l171_171062

variable (B C B_inter_C B_union_C : ℕ)

theorem students_like_basketball (hC : C = 8) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 17) 
    (h_incl_excl : B_union_C = B + C - B_inter_C) : B = 12 := by 
  -- Given: 
  --   C = 8
  --   B_inter_C = 3
  --   B_union_C = 17
  --   B_union_C = B + C - B_inter_C
  -- Prove: 
  --   B = 12
  sorry

end students_like_basketball_l171_171062


namespace no_prime_sum_10003_l171_171901

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l171_171901


namespace fraction_reducible_to_17_l171_171953

theorem fraction_reducible_to_17 (m n : ℕ) (h_coprime : Nat.gcd m n = 1)
  (h_reducible : ∃ d : ℕ, d ∣ (3 * m - n) ∧ d ∣ (5 * n + 2 * m)) :
  ∃ k : ℕ, (3 * m - n) / k = 17 ∧ (5 * n + 2 * m) / k = 17 :=
by
  have key : Nat.gcd (3 * m - n) (5 * n + 2 * m) = 17 := sorry
  -- using the result we need to construct our desired k
  use 17 / (Nat.gcd (3 * m - n) (5 * n + 2 * m))
  -- rest of intimate proof here
  sorry

end fraction_reducible_to_17_l171_171953


namespace find_b_l171_171320

theorem find_b (a b : ℝ) (h1 : a * (a - 4) = 21) (h2 : b * (b - 4) = 21) (h3 : a + b = 4) (h4 : a ≠ b) :
  b = -3 :=
sorry

end find_b_l171_171320


namespace prob_2_lt_X_lt_4_l171_171182

noncomputable def normal_dist_p (μ σ : ℝ) (x : ℝ) : ℝ := sorry -- Assume this computes the CDF at x for a normal distribution

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Condition: X follows a normal distribution with mean 3 and variance σ^2
axiom normal_distribution_X : ∀ x, X x = normal_dist_p 3 σ x

-- Condition: P(X ≤ 4) = 0.84
axiom prob_X_leq_4 : normal_dist_p 3 σ 4 = 0.84

-- Goal: Prove P(2 < X < 4) = 0.68
theorem prob_2_lt_X_lt_4 : normal_dist_p 3 σ 4 - normal_dist_p 3 σ 2 = 0.68 := by
  sorry

end prob_2_lt_X_lt_4_l171_171182


namespace number_of_right_triangles_with_hypotenuse_is_12_l171_171696

theorem number_of_right_triangles_with_hypotenuse_is_12 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b : ℕ), 
     (b < 150) →
     (a^2 + b^2 = (b + 2)^2) →
     ∃ (k : ℕ), a = 2 * k ∧ k^2 = b + 1) := 
  sorry

end number_of_right_triangles_with_hypotenuse_is_12_l171_171696


namespace ad_space_length_l171_171995

theorem ad_space_length 
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (width : ℝ)
  (cost_per_sq_ft : ℝ)
  (total_cost : ℝ) 
  (H1 : num_companies = 3)
  (H2 : ads_per_company = 10)
  (H3 : width = 5)
  (H4 : cost_per_sq_ft = 60)
  (H5 : total_cost = 108000) :
  ∃ L : ℝ, (num_companies * ads_per_company * width * L * cost_per_sq_ft = total_cost) ∧ (L = 12) :=
by
  sorry

end ad_space_length_l171_171995


namespace solve_ratios_l171_171199

theorem solve_ratios (q m n : ℕ) (h1 : 7 / 9 = n / 108) (h2 : 7 / 9 = (m + n) / 126) (h3 : 7 / 9 = (q - m) / 162) : q = 140 :=
by
  sorry

end solve_ratios_l171_171199


namespace calculate_distance_to_friend_l171_171469

noncomputable def distance_to_friend (d t : ℝ) : Prop :=
  (d = 45 * (t + 1)) ∧ (d = 45 + 65 * (t - 0.75))

theorem calculate_distance_to_friend : ∃ d t: ℝ, distance_to_friend d t ∧ d = 155 :=
by
  exists 155
  exists 2.4375
  sorry

end calculate_distance_to_friend_l171_171469


namespace part1_part2_l171_171075

-- Statement for Part 1
theorem part1 : 
  ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 11) := sorry

-- Statement for Part 2
theorem part2 : 
  ¬ ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 13) := sorry

end part1_part2_l171_171075


namespace rational_solutions_of_quadratic_l171_171946

theorem rational_solutions_of_quadratic (k : ℕ) (hk : 0 < k ∧ k ≤ 10) :
  ∃ (x : ℚ), k * x^2 + 20 * x + k = 0 ↔ (k = 6 ∨ k = 8 ∨ k = 10) :=
by sorry

end rational_solutions_of_quadratic_l171_171946


namespace sequence_a4_eq_15_l171_171421

theorem sequence_a4_eq_15 (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * a n + 1) → a 4 = 15 :=
by
  sorry

end sequence_a4_eq_15_l171_171421


namespace solve_quadratic_equation_1_solve_quadratic_equation_2_l171_171038

theorem solve_quadratic_equation_1 (x : ℝ) :
  3 * x^2 + 2 * x - 1 = 0 ↔ x = 1/3 ∨ x = -1 :=
by sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  (x + 2) * (x - 3) = 5 * x - 15 ↔ x = 3 :=
by sorry

end solve_quadratic_equation_1_solve_quadratic_equation_2_l171_171038


namespace speed_increase_impossible_l171_171519

theorem speed_increase_impossible (v : ℝ) : v = 60 → (¬ ∃ v', (1 / (v' / 60) = 0)) :=
by sorry

end speed_increase_impossible_l171_171519


namespace intersection_of_M_and_N_l171_171233

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end intersection_of_M_and_N_l171_171233


namespace logically_equivalent_to_original_l171_171498

def original_statement (E W : Prop) : Prop := E → ¬ W
def statement_I (E W : Prop) : Prop := W → E
def statement_II (E W : Prop) : Prop := ¬ E → ¬ W
def statement_III (E W : Prop) : Prop := W → ¬ E
def statement_IV (E W : Prop) : Prop := ¬ E ∨ ¬ W

theorem logically_equivalent_to_original (E W : Prop) :
  (original_statement E W ↔ statement_III E W) ∧
  (original_statement E W ↔ statement_IV E W) :=
  sorry

end logically_equivalent_to_original_l171_171498


namespace sequence_properties_l171_171911

-- Define the sequence formula
def a_n (n : ℤ) : ℤ := n^2 - 5 * n + 4

-- State the theorem about the sequence
theorem sequence_properties :
  -- Part 1: The number of negative terms in the sequence
  (∃ (S : Finset ℤ), ∀ n ∈ S, a_n n < 0 ∧ S.card = 2) ∧
  -- Part 2: The minimum value of the sequence and the value of n at minimum
  (∀ n : ℤ, (a_n n ≥ -9 / 4) ∧ (a_n (5 / 2) = -9 / 4)) :=
by {
  sorry
}

end sequence_properties_l171_171911


namespace binary_to_decimal_101101_l171_171096

def binary_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (λ (digit : ℕ) (acc : ℕ × ℕ) => (acc.1 + digit * 2 ^ acc.2, acc.2 + 1)) (0, 0) |>.1

theorem binary_to_decimal_101101 : binary_to_decimal [1, 0, 1, 1, 0, 1] = 45 :=
by
  -- Proof is needed but here we use sorry as placeholder.
  sorry

end binary_to_decimal_101101_l171_171096


namespace original_class_size_l171_171242

/-- Let A be the average age of the original adult class, which is 40 years. -/
def A : ℕ := 40

/-- Let B be the average age of the 8 new students, which is 32 years. -/
def B : ℕ := 32

/-- Let C be the decreased average age of the class after the new students join, which is 36 years. -/
def C : ℕ := 36

/-- The original number of students in the adult class is N. -/
def N : ℕ := 8

/-- The equation representing the total age of the class after the new students join. -/
theorem original_class_size :
  (A * N) + (B * 8) = C * (N + 8) ↔ N = 8 := by
  sorry

end original_class_size_l171_171242


namespace white_washing_cost_correct_l171_171967

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

def door_length : ℝ := 6
def door_width : ℝ := 3

def window_length : ℝ := 4
def window_width : ℝ := 3

def cost_per_sq_ft : ℝ := 8

def calculate_white_washing_cost : ℝ :=
  let total_wall_area := 2 * (room_length * room_height) + 2 * (room_width * room_height)
  let door_area := door_length * door_width
  let window_area := 3 * (window_length * window_width)
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sq_ft

theorem white_washing_cost_correct : calculate_white_washing_cost = 7248 := by
  sorry

end white_washing_cost_correct_l171_171967


namespace abby_and_damon_weight_l171_171755

variables {a b c d : ℝ}

theorem abby_and_damon_weight (h1 : a + b = 260) (h2 : b + c = 245) 
(h3 : c + d = 270) (h4 : a + c = 220) : a + d = 285 := 
by 
  sorry

end abby_and_damon_weight_l171_171755


namespace probability_of_drawing_1_red_1_white_l171_171931

-- Definitions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Probabilities
def p_red_first_white_second : ℚ := (red_balls / total_balls : ℚ) * (white_balls / total_balls : ℚ)
def p_white_first_red_second : ℚ := (white_balls / total_balls : ℚ) * (red_balls / total_balls : ℚ)

-- Total probability
def total_probability : ℚ := p_red_first_white_second + p_white_first_red_second

theorem probability_of_drawing_1_red_1_white :
  total_probability = 12 / 25 := by
  sorry

end probability_of_drawing_1_red_1_white_l171_171931


namespace parallel_line_through_point_l171_171566

-- Problem: Prove the equation of the line that passes through the point (1, 1)
-- and is parallel to the line 2x - y + 1 = 0 is 2x - y - 1 = 0.

theorem parallel_line_through_point (x y : ℝ) (c : ℝ) :
  (2*x - y + 1 = 0) → (x = 1) → (y = 1) → (2*1 - 1 + c = 0) → c = -1 → (2*x - y - 1 = 0) :=
by
  sorry

end parallel_line_through_point_l171_171566


namespace problem_statement_l171_171250

open Complex

theorem problem_statement (x y : ℝ) (i : ℂ) (h_i : i = Complex.I) (h : x + (y - 2) * i = 2 / (1 + i)) : x + y = 2 :=
by
  sorry

end problem_statement_l171_171250


namespace abc_value_l171_171465

theorem abc_value (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 30) 
  (h5 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := 
sorry

end abc_value_l171_171465


namespace horizontal_length_of_monitor_l171_171078

def monitor_diagonal := 32
def aspect_ratio_horizontal := 16
def aspect_ratio_height := 9

theorem horizontal_length_of_monitor :
  ∃ (horizontal_length : ℝ), horizontal_length = 512 / Real.sqrt 337 := by
  sorry

end horizontal_length_of_monitor_l171_171078


namespace general_form_of_quadratic_equation_l171_171228

noncomputable def quadratic_equation_general_form (x : ℝ) : Prop :=
  (x + 3) * (x - 1) = 2 * x - 4

theorem general_form_of_quadratic_equation (x : ℝ) :
  quadratic_equation_general_form x → x^2 + 1 = 0 :=
sorry

end general_form_of_quadratic_equation_l171_171228


namespace distinct_solutions_l171_171505

theorem distinct_solutions : 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 - 7| = 2 * |x1 + 1| + |x1 - 3| ∧ |x2 - 7| = 2 * |x2 + 1| + |x2 - 3|) := 
by
  sorry

end distinct_solutions_l171_171505


namespace sufficient_and_necessary_condition_l171_171447

variable {a : ℕ → ℝ}
variable {a1 a2 : ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  (∀ n, a n = a1 * q ^ n)

noncomputable def increasing (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1))

theorem sufficient_and_necessary_condition
  (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h_geom : geometric_sequence a a1 q)
  (h_a1_pos : a1 > 0)
  (h_a1_lt_a2 : a1 < a1 * q) :
  increasing a ↔ a1 < a1 * q := 
sorry

end sufficient_and_necessary_condition_l171_171447


namespace ratio_of_perimeters_of_squares_l171_171688

theorem ratio_of_perimeters_of_squares (A B : ℝ) (h: A / B = 16 / 25) : ∃ (P1 P2 : ℝ), P1 / P2 = 4 / 5 :=
by
  sorry

end ratio_of_perimeters_of_squares_l171_171688


namespace Amanda_money_left_l171_171372

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l171_171372


namespace score_of_B_is_correct_l171_171858

theorem score_of_B_is_correct (A B C D E : ℝ)
  (h1 : (A + B + C + D + E) / 5 = 90)
  (h2 : (A + B + C) / 3 = 86)
  (h3 : (B + D + E) / 3 = 95) : 
  B = 93 := 
by 
  sorry

end score_of_B_is_correct_l171_171858


namespace product_of_two_numbers_l171_171717

theorem product_of_two_numbers (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 3)
  : x * y = 154 := by
  sorry

end product_of_two_numbers_l171_171717


namespace find_f_2008_l171_171718

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero (f : ℝ → ℝ) : f 0 = 2008
axiom f_inequality_1 (f : ℝ → ℝ) (x : ℝ) : f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality_2 (f : ℝ → ℝ) (x : ℝ) : f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 (f : ℝ → ℝ) : f 2008 = 2^2008 + 2007 :=
by
  apply sorry

end find_f_2008_l171_171718


namespace shaded_rectangle_area_l171_171697

-- Define the square PQRS and its properties
def is_square (s : ℝ) := ∃ (PQ QR RS SP : ℝ), PQ = s ∧ QR = s ∧ RS = s ∧ SP = s

-- Define the conditions for the side lengths and segments
def side_length := 11
def top_left_height := 6
def top_right_height := 2
def width_bottom_right := 11 - 10
def width_top_right := 8

-- Calculate necessary dimensions
def shaded_rectangle_height := top_left_height - top_right_height
def shaded_rectangle_width := width_top_right - width_bottom_right

-- Proof statement
theorem shaded_rectangle_area (s : ℝ) (h1 : is_square s)
  (h2 : s = side_length)
  (h3 : shaded_rectangle_height = 4)
  (h4 : shaded_rectangle_width = 7) :
  4 * 7 = 28 := by
  sorry

end shaded_rectangle_area_l171_171697


namespace simplify_and_evaluate_expr_l171_171757

noncomputable def a : ℝ := 3 + Real.sqrt 5
noncomputable def b : ℝ := 3 - Real.sqrt 5

theorem simplify_and_evaluate_expr : 
  (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expr_l171_171757


namespace no_exact_cover_l171_171361

theorem no_exact_cover (large_w : ℕ) (large_h : ℕ) (small_w : ℕ) (small_h : ℕ) (n : ℕ) :
  large_w = 13 → large_h = 7 → small_w = 2 → small_h = 3 → n = 15 →
  ¬ (small_w * small_h * n = large_w * large_h) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end no_exact_cover_l171_171361


namespace determine_x_l171_171791

variable {x y : ℝ}

theorem determine_x (h : (x - 1) / x = (y^3 + 3 * y^2 - 4) / (y^3 + 3 * y^2 - 5)) : 
  x = y^3 + 3 * y^2 - 5 := 
sorry

end determine_x_l171_171791


namespace cracked_to_broken_eggs_ratio_l171_171945

theorem cracked_to_broken_eggs_ratio (total_eggs : ℕ) (broken_eggs : ℕ) (P C : ℕ)
  (h1 : total_eggs = 24)
  (h2 : broken_eggs = 3)
  (h3 : P - C = 9)
  (h4 : P + C = 21) :
  (C : ℚ) / (broken_eggs : ℚ) = 2 :=
by
  sorry

end cracked_to_broken_eggs_ratio_l171_171945


namespace ratio_soda_water_l171_171512

variables (W S : ℕ) (k : ℕ)

-- Conditions of the problem
def condition1 : Prop := S = k * W - 6
def condition2 : Prop := W + S = 54
def positive_integer_k : Prop := k > 0

-- The theorem we want to prove
theorem ratio_soda_water (h1 : condition1 W S k) (h2 : condition2 W S) (h3 : positive_integer_k k) : S / gcd S W = 4 ∧ W / gcd S W = 5 :=
sorry

end ratio_soda_water_l171_171512


namespace correct_sequence_of_linear_regression_analysis_l171_171979

def linear_regression_steps : List ℕ := [2, 4, 3, 1]

theorem correct_sequence_of_linear_regression_analysis :
  linear_regression_steps = [2, 4, 3, 1] :=
by
  sorry

end correct_sequence_of_linear_regression_analysis_l171_171979


namespace arithmetic_sequence_sum_product_l171_171020

noncomputable def a := 13 / 2
def d := 3 / 2

theorem arithmetic_sequence_sum_product (a d : ℚ) (h1 : 4 * a = 26) (h2 : a^2 - d^2 = 40) :
  (a - 3 * d, a - d, a + d, a + 3 * d) = (2, 5, 8, 11) ∨
  (a - 3 * d, a - d, a + d, a + 3 * d) = (11, 8, 5, 2) :=
  sorry

end arithmetic_sequence_sum_product_l171_171020


namespace determine_missing_digits_l171_171032

theorem determine_missing_digits :
  (237 * 0.31245 = 7430.65) := 
by 
  sorry

end determine_missing_digits_l171_171032


namespace cube_painting_distinct_ways_l171_171365

theorem cube_painting_distinct_ways : ∃ n : ℕ, n = 7 := sorry

end cube_painting_distinct_ways_l171_171365


namespace Q_at_one_is_zero_l171_171282

noncomputable def Q (x : ℚ) : ℚ := x^4 - 2 * x^2 + 1

theorem Q_at_one_is_zero :
  Q 1 = 0 :=
by
  -- Here we would put the formal proof in Lean language
  sorry

end Q_at_one_is_zero_l171_171282


namespace original_price_l171_171985

theorem original_price (x : ℝ) (h1 : 0.75 * x + 12 = x - 12) (h2 : 0.90 * x - 42 = x - 12) : x = 360 :=
by
  sorry

end original_price_l171_171985


namespace plane_through_Ox_and_point_plane_parallel_Oz_and_points_l171_171015

-- Definitions for first plane problem
def plane1_through_Ox_axis (y z : ℝ) : Prop := 3 * y + 2 * z = 0

-- Definitions for second plane problem
def plane2_parallel_Oz (x y : ℝ) : Prop := x + 3 * y - 1 = 0

theorem plane_through_Ox_and_point : plane1_through_Ox_axis 2 (-3) := 
by {
  -- Hint: Prove that substituting y = 2 and z = -3 in the equation results in LHS equals RHS.
  -- proof
  sorry 
}

theorem plane_parallel_Oz_and_points : 
  plane2_parallel_Oz 1 0 ∧ plane2_parallel_Oz (-2) 1 :=
by {
  -- Hint: Prove that substituting the points (1, 0) and (-2, 1) in the equation results in LHS equals RHS.
  -- proof
  sorry
}

end plane_through_Ox_and_point_plane_parallel_Oz_and_points_l171_171015


namespace girls_more_than_boys_l171_171638

theorem girls_more_than_boys (total_students boys : ℕ) (h1 : total_students = 650) (h2 : boys = 272) :
  (total_students - boys) - boys = 106 :=
by
  sorry

end girls_more_than_boys_l171_171638


namespace central_angle_of_sector_l171_171753

-- Define the given conditions
def radius : ℝ := 10
def area : ℝ := 100

-- The statement to be proved
theorem central_angle_of_sector (α : ℝ) (h : area = (1 / 2) * α * radius ^ 2) : α = 2 :=
by
  sorry

end central_angle_of_sector_l171_171753


namespace polynomial_roots_l171_171861

theorem polynomial_roots : (∃ x : ℝ, (4 * x ^ 4 + 11 * x ^ 3 - 37 * x ^ 2 + 18 * x = 0) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = 3 / 2 ∨ x = -6)) :=
by 
  sorry

end polynomial_roots_l171_171861


namespace amount_p_l171_171353

variable (P : ℚ)

/-- p has $42 more than what q and r together would have had if both q and r had 1/8 of what p has.
    We need to prove that P = 56. -/
theorem amount_p (h : P = (1/8 : ℚ) * P + (1/8) * P + 42) : P = 56 :=
by
  sorry

end amount_p_l171_171353


namespace total_water_heaters_l171_171735

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end total_water_heaters_l171_171735


namespace max_planes_determined_by_15_points_l171_171510

theorem max_planes_determined_by_15_points : ∃ (n : ℕ), n = 455 ∧ 
  ∀ (P : Finset (Fin 15)), (15 + 1) ∣ (P.card * (P.card - 1) * (P.card - 2) / 6) → n = (15 * 14 * 13) / 6 :=
by
  sorry

end max_planes_determined_by_15_points_l171_171510


namespace find_k_l171_171494

theorem find_k (k : ℚ) :
  (5 + ∑' n : ℕ, (5 + 2*k*(n+1)) / 4^n) = 10 → k = 15/4 :=
by
  sorry

end find_k_l171_171494


namespace higher_profit_percentage_l171_171095

theorem higher_profit_percentage (P : ℝ) :
  (P / 100 * 800 = 144) ↔ (P = 18) :=
by
  sorry

end higher_profit_percentage_l171_171095


namespace tyrone_money_l171_171333

def bill_value (count : ℕ) (val : ℝ) : ℝ :=
  count * val

def total_value : ℝ :=
  bill_value 2 1 + bill_value 1 5 + bill_value 13 0.25 + bill_value 20 0.10 + bill_value 8 0.05 + bill_value 35 0.01

theorem tyrone_money : total_value = 13 := by 
  sorry

end tyrone_money_l171_171333


namespace solve_equation_l171_171428

theorem solve_equation :
  ∀ x y : ℝ, (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 :=
by
  intro x y
  sorry

end solve_equation_l171_171428


namespace class_average_score_l171_171561

theorem class_average_score (n_boys n_girls : ℕ) (avg_score_boys avg_score_girls : ℕ) 
  (h_nb : n_boys = 12)
  (h_ng : n_girls = 4)
  (h_ab : avg_score_boys = 84)
  (h_ag : avg_score_girls = 92) : 
  (n_boys * avg_score_boys + n_girls * avg_score_girls) / (n_boys + n_girls) = 86 := 
by 
  sorry

end class_average_score_l171_171561


namespace sequence_bounds_l171_171104

theorem sequence_bounds (n : ℕ) (hpos : 0 < n) :
  ∃ (a : ℕ → ℝ), (a 0 = 1/2) ∧
  (∀ k < n, a (k + 1) = a k + (1/n) * (a k)^2) ∧
  (1 - 1 / n < a n ∧ a n < 1) :=
sorry

end sequence_bounds_l171_171104


namespace dot_product_value_l171_171916

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (3, 1)

theorem dot_product_value :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 :=
by
  -- Proof goes here
  sorry

end dot_product_value_l171_171916


namespace path_problem_l171_171276

noncomputable def path_bounds (N : ℕ) (h : 0 < N) : Prop :=
  ∃ p : ℕ, 4 * N ≤ p ∧ p ≤ 2 * N^2 + 2 * N

theorem path_problem (N : ℕ) (h : 0 < N) : path_bounds N h :=
  sorry

end path_problem_l171_171276


namespace apples_on_tree_l171_171794

-- Defining initial number of apples on the tree
def initial_apples : ℕ := 4

-- Defining apples picked from the tree
def apples_picked : ℕ := 2

-- Defining new apples grown on the tree
def new_apples : ℕ := 3

-- Prove the final number of apples on the tree is 5
theorem apples_on_tree : initial_apples - apples_picked + new_apples = 5 :=
by
  -- This is where the proof would go
  sorry

end apples_on_tree_l171_171794


namespace seq_nat_eq_n_l171_171001

theorem seq_nat_eq_n (a : ℕ → ℕ) (h_inc : ∀ n, a n < a (n + 1))
  (h_le : ∀ n, a n ≤ n + 2020)
  (h_div : ∀ n, a (n + 1) ∣ (n^3 * a n - 1)) :
  ∀ n, a n = n :=
by
  sorry

end seq_nat_eq_n_l171_171001


namespace simplify_expression_l171_171138

variables (a b : ℝ)

theorem simplify_expression : 
  (2 * a^2 - 3 * a * b + 8) - (-a * b - a^2 + 8) = 3 * a^2 - 2 * a * b :=
by sorry

-- Note:
-- ℝ denotes real numbers. Adjust types accordingly if using different numerical domains (e.g., ℚ, ℂ).

end simplify_expression_l171_171138


namespace hemisphere_surface_area_l171_171088

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 225 * π) : 2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l171_171088


namespace ratio_platform_to_train_length_l171_171317

variable (L P t : ℝ)

-- Definitions based on conditions
def train_has_length (L : ℝ) : Prop := true
def train_constant_velocity : Prop := true
def train_passes_pole_in_t_seconds (L t : ℝ) : Prop := L / t = L
def train_passes_platform_in_4t_seconds (L P t : ℝ) : Prop := L / t = (L + P) / (4 * t)

-- Theorem statement: ratio of the length of the platform to the length of the train is 3:1
theorem ratio_platform_to_train_length (h1 : train_has_length L) 
                                      (h2 : train_constant_velocity) 
                                      (h3 : train_passes_pole_in_t_seconds L t)
                                      (h4 : train_passes_platform_in_4t_seconds L P t) :
  P / L = 3 := 
by sorry

end ratio_platform_to_train_length_l171_171317


namespace largest_of_five_consecutive_non_primes_under_40_l171_171061

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n 

theorem largest_of_five_consecutive_non_primes_under_40 :
  ∃ x, (x > 9) ∧ (x + 4 < 40) ∧ 
       (¬ is_prime x) ∧
       (¬ is_prime (x + 1)) ∧
       (¬ is_prime (x + 2)) ∧
       (¬ is_prime (x + 3)) ∧
       (¬ is_prime (x + 4)) ∧
       (x + 4 = 36) :=
sorry

end largest_of_five_consecutive_non_primes_under_40_l171_171061


namespace z_when_y_six_l171_171219

theorem z_when_y_six
    (k : ℝ)
    (h1 : ∀ y (z : ℝ), y^2 * Real.sqrt z = k)
    (h2 : ∃ (y : ℝ) (z : ℝ), y = 3 ∧ z = 4 ∧ y^2 * Real.sqrt z = k) :
  ∃ z : ℝ, y = 6 ∧ z = 1 / 4 := 
sorry

end z_when_y_six_l171_171219


namespace numOxygenAtoms_l171_171366

-- Define the conditions as hypothesis
def numCarbonAtoms : ℕ := 4
def numHydrogenAtoms : ℕ := 8
def molecularWeight : ℕ := 88
def atomicWeightCarbon : ℕ := 12
def atomicWeightHydrogen : ℕ := 1
def atomicWeightOxygen : ℕ := 16

-- The statement to be proved
theorem numOxygenAtoms :
  let totalWeightC := numCarbonAtoms * atomicWeightCarbon
  let totalWeightH := numHydrogenAtoms * atomicWeightHydrogen
  let totalWeightCH := totalWeightC + totalWeightH
  let weightOxygenAtoms := molecularWeight - totalWeightCH
  let numOxygenAtoms := weightOxygenAtoms / atomicWeightOxygen
  numOxygenAtoms = 2 :=
by {
  sorry
}

end numOxygenAtoms_l171_171366


namespace baker_sold_more_cakes_than_pastries_l171_171872

theorem baker_sold_more_cakes_than_pastries (cakes_sold pastries_sold : ℕ) 
  (h_cakes_sold : cakes_sold = 158) (h_pastries_sold : pastries_sold = 147) : 
  (cakes_sold - pastries_sold) = 11 := by
  sorry

end baker_sold_more_cakes_than_pastries_l171_171872


namespace positive_integers_satisfy_l171_171774

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end positive_integers_satisfy_l171_171774


namespace inequality_proof_l171_171504

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) :
  (x / (y + 1) + y / (x + 1)) ≥ (2 / 3) ∧ (x = 1 / 2 ∧ y = 1 / 2 → x / (y + 1) + y / (x + 1) = 2 / 3) := by
  sorry

end inequality_proof_l171_171504


namespace voronovich_inequality_l171_171950

theorem voronovich_inequality (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6 * a * b * c ≥ a * b + b * c + c * a :=
by
  sorry

end voronovich_inequality_l171_171950


namespace angle_SQR_measure_l171_171417

theorem angle_SQR_measure
    (angle_PQR : ℝ)
    (angle_PQS : ℝ)
    (h1 : angle_PQR = 40)
    (h2 : angle_PQS = 15) : 
    angle_PQR - angle_PQS = 25 := 
by
    sorry

end angle_SQR_measure_l171_171417


namespace distance_between_bus_stops_l171_171841

theorem distance_between_bus_stops (d : ℕ) (unit : String) 
  (h: d = 3000 ∧ unit = "meters") : unit = "C" := 
by 
  sorry

end distance_between_bus_stops_l171_171841


namespace total_bees_in_colony_l171_171776

def num_bees_in_hive_after_changes (initial_bees : ℕ) (bees_in : ℕ) (bees_out : ℕ) : ℕ :=
  initial_bees + bees_in - bees_out

theorem total_bees_in_colony :
  let hive1 := num_bees_in_hive_after_changes 45 12 8
  let hive2 := num_bees_in_hive_after_changes 60 15 20
  let hive3 := num_bees_in_hive_after_changes 75 10 5
  hive1 + hive2 + hive3 = 184 :=
by
  sorry

end total_bees_in_colony_l171_171776


namespace second_solution_salt_percent_l171_171304

theorem second_solution_salt_percent (S : ℝ) (x : ℝ) 
  (h1 : 0.14 * S - 0.14 * (S / 4) + (x / 100) * (S / 4) = 0.16 * S) : 
  x = 22 :=
by 
  -- Proof omitted
  sorry

end second_solution_salt_percent_l171_171304


namespace sin_45_eq_sqrt2_div_2_l171_171645

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = (Real.sqrt 2) / 2 := 
  sorry

end sin_45_eq_sqrt2_div_2_l171_171645


namespace smallest_number_divided_into_18_and_60_groups_l171_171711

theorem smallest_number_divided_into_18_and_60_groups : ∃ n : ℕ, (∀ m : ℕ, (m % 18 = 0 ∧ m % 60 = 0) → n ≤ m) ∧ (n % 18 = 0 ∧ n % 60 = 0) ∧ n = 180 :=
by
  use 180
  sorry

end smallest_number_divided_into_18_and_60_groups_l171_171711


namespace impossible_arrangement_of_numbers_l171_171373

theorem impossible_arrangement_of_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) 
(hpos : ∀ i, 0 < a i)
(hdiff : ∃ i, ∀ j ≠ i, a j = a ((j + 1) % n) - a ((j - 1 + n) % n)):
  false :=
by
  sorry

end impossible_arrangement_of_numbers_l171_171373


namespace carla_marbles_l171_171885

theorem carla_marbles (before now bought : ℝ) (h_before : before = 187.0) (h_now : now = 321) : bought = 134 :=
by
  sorry

end carla_marbles_l171_171885


namespace remaining_pieces_l171_171363

theorem remaining_pieces (initial_pieces : ℕ) (arianna_lost : ℕ) (samantha_lost : ℕ) (diego_lost : ℕ) (lucas_lost : ℕ) :
  initial_pieces = 128 → arianna_lost = 3 → samantha_lost = 9 → diego_lost = 5 → lucas_lost = 7 →
  initial_pieces - (arianna_lost + samantha_lost + diego_lost + lucas_lost) = 104 := by
  sorry

end remaining_pieces_l171_171363


namespace carol_initial_cupcakes_l171_171941

variable (x : ℕ)

theorem carol_initial_cupcakes (h : (x - 9) + 28 = 49) : x = 30 := 
  sorry

end carol_initial_cupcakes_l171_171941


namespace percent_asian_population_in_West_l171_171464

-- Define the populations in different regions
def population_NE := 2
def population_MW := 3
def population_South := 4
def population_West := 10

-- Define the total population
def total_population := population_NE + population_MW + population_South + population_West

-- Calculate the percentage of the population in the West
def percentage_in_West := (population_West * 100) / total_population

-- The proof statement
theorem percent_asian_population_in_West : percentage_in_West = 53 := by
  sorry -- proof to be completed

end percent_asian_population_in_West_l171_171464


namespace arithmetic_sequence_value_l171_171423

theorem arithmetic_sequence_value (a_1 d : ℤ) (h : (a_1 + 2 * d) + (a_1 + 7 * d) = 10) : 
  3 * (a_1 + 4 * d) + (a_1 + 6 * d) = 20 :=
by
  sorry

end arithmetic_sequence_value_l171_171423


namespace total_dogs_barking_l171_171343

theorem total_dogs_barking 
  (initial_dogs : ℕ)
  (new_dogs : ℕ)
  (h1 : initial_dogs = 30)
  (h2 : new_dogs = 3 * initial_dogs) :
  initial_dogs + new_dogs = 120 :=
by
  sorry

end total_dogs_barking_l171_171343


namespace min_value_of_quadratic_l171_171000

theorem min_value_of_quadratic :
  (∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896) ∧ (∃ x : ℝ, 3 * x^2 - 12 * x + 908 = 896) :=
by
  sorry

end min_value_of_quadratic_l171_171000


namespace jose_bottle_caps_l171_171890

def jose_start : ℕ := 7
def rebecca_gives : ℕ := 2
def final_bottle_caps : ℕ := 9

theorem jose_bottle_caps :
  jose_start + rebecca_gives = final_bottle_caps :=
by
  sorry

end jose_bottle_caps_l171_171890


namespace value_of_x_squared_plus_inverse_squared_l171_171962

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x ≠ 0) (h : x^4 + (1 / x^4) = 2) : x^2 + (1 / x^2) = 2 :=
sorry

end value_of_x_squared_plus_inverse_squared_l171_171962


namespace preferred_order_for_boy_l171_171275

variable (p q : ℝ)
variable (h : p < q)

theorem preferred_order_for_boy (p q : ℝ) (h : p < q) : 
  (2 * p * q - p^2 * q) > (2 * p * q - p * q^2) := 
sorry

end preferred_order_for_boy_l171_171275


namespace evaluate_expression_at_2_l171_171891

-- Define the quadratic and linear components of the expression
def quadratic (x : ℝ) := 3 * x ^ 2 - 8 * x + 5
def linear (x : ℝ) := 4 * x - 7

-- State the proposition to evaluate the given expression at x = 2
theorem evaluate_expression_at_2 : quadratic 2 * linear 2 = 1 := by
  -- The proof is skipped by using sorry
  sorry

end evaluate_expression_at_2_l171_171891


namespace value_of_g_at_2_l171_171057

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  sorry

end value_of_g_at_2_l171_171057


namespace vector_addition_l171_171252

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 3)

-- Stating the problem: proving the sum of vectors a and b
theorem vector_addition : a + b = (3, 4) := 
by 
  -- Proof is not required as per the instructions
  sorry

end vector_addition_l171_171252


namespace solve_equation_l171_171623

theorem solve_equation (x : ℝ) (h1 : x + 1 ≠ 0) (h2 : 2 * x - 1 ≠ 0) :
  (2 / (x + 1) = 3 / (2 * x - 1)) ↔ (x = 5) := 
sorry

end solve_equation_l171_171623


namespace problem_I_problem_II_l171_171007

-- Problem I statement
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 := 
by
  sorry

-- Problem II statement
theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (2 * x - 1) ≥ 2) →
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (5/2) :=
by 
  sorry

end problem_I_problem_II_l171_171007


namespace opposite_numbers_add_l171_171691

theorem opposite_numbers_add : ∀ {a b : ℤ}, a + b = 0 → a + b + 3 = 3 :=
by
  intros
  sorry

end opposite_numbers_add_l171_171691


namespace parabola_tangents_coprime_l171_171601

theorem parabola_tangents_coprime {d e f : ℤ} (hd : d ≠ 0) (he : e ≠ 0)
  (h_coprime: Int.gcd (Int.gcd d e) f = 1)
  (h_tangent1 : d^2 - 4 * e * (2 * e - f) = 0)
  (h_tangent2 : (e + d)^2 - 4 * d * (8 * d - f) = 0) :
  d + e + f = 8 := by
  sorry

end parabola_tangents_coprime_l171_171601


namespace smallest_hamburger_packages_l171_171567

theorem smallest_hamburger_packages (h_num : ℕ) (b_num : ℕ) (h_bag_num : h_num = 10) (b_bag_num : b_num = 15) :
  ∃ (n : ℕ), n = 3 ∧ (n * h_num) = (2 * b_num) := by
  sorry

end smallest_hamburger_packages_l171_171567


namespace total_parcel_boxes_l171_171249

theorem total_parcel_boxes (a b c d : ℕ) (row_boxes column_boxes total_boxes : ℕ)
  (h_left : a = 7) (h_right : b = 13)
  (h_front : c = 8) (h_back : d = 14)
  (h_row : row_boxes = a - 1 + 1 + b) -- boxes in a row: (a - 1) + 1 (parcel itself) + b
  (h_column : column_boxes = c - 1 + 1 + d) -- boxes in a column: (c -1) + 1(parcel itself) + d
  (h_total : total_boxes = row_boxes * column_boxes) :
  total_boxes = 399 := by
  sorry

end total_parcel_boxes_l171_171249


namespace find_y_l171_171114

theorem find_y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) (h4 : z = 2) : y = 3 :=
    sorry

end find_y_l171_171114


namespace abs_pi_sub_abs_pi_sub_three_l171_171647

theorem abs_pi_sub_abs_pi_sub_three (h : Real.pi > 3) : 
  abs (Real.pi - abs (Real.pi - 3)) = 2 * Real.pi - 3 := 
by
  sorry

end abs_pi_sub_abs_pi_sub_three_l171_171647


namespace probability_of_winning_pair_is_correct_l171_171542

noncomputable def probability_of_winning_pair : ℚ :=
  let total_cards := 10
  let red_cards := 5
  let blue_cards := 5
  let total_ways := Nat.choose total_cards 2 -- Combination C(10,2)
  let same_color_ways := Nat.choose red_cards 2 + Nat.choose blue_cards 2 -- Combination C(5,2) for each color
  let consecutive_pairs_per_color := 4
  let consecutive_ways := 2 * consecutive_pairs_per_color -- Two colors
  let favorable_ways := same_color_ways + consecutive_ways
  favorable_ways / total_ways

theorem probability_of_winning_pair_is_correct : 
  probability_of_winning_pair = 28 / 45 := sorry

end probability_of_winning_pair_is_correct_l171_171542


namespace surface_area_after_removing_corners_l171_171170

-- Define the dimensions of the cubes
def original_cube_side : ℝ := 4
def corner_cube_side : ℝ := 2

-- The surface area function for a cube with given side length
def surface_area (side : ℝ) : ℝ := 6 * side * side

theorem surface_area_after_removing_corners :
  surface_area original_cube_side = 96 :=
by
  sorry

end surface_area_after_removing_corners_l171_171170


namespace jessica_not_work_days_l171_171259

theorem jessica_not_work_days:
  ∃ (x y z : ℕ), 
    (x + y + z = 30) ∧
    (80 * x - 40 * y + 40 * z = 1600) ∧
    (z = 5) ∧
    (y = 5) :=
by
  sorry

end jessica_not_work_days_l171_171259


namespace calc_expression_l171_171248

theorem calc_expression : 
  (Real.sqrt 16 - 4 * (Real.sqrt 2) / 2 + abs (- (Real.sqrt 3 * Real.sqrt 6)) + (-1) ^ 2023) = 
  (3 + Real.sqrt 2) :=
by
  sorry

end calc_expression_l171_171248


namespace cubic_expression_value_l171_171390

theorem cubic_expression_value (m : ℝ) (h : m^2 + 3 * m - 2023 = 0) :
  m^3 + 2 * m^2 - 2026 * m - 2023 = -4046 :=
by
  sorry

end cubic_expression_value_l171_171390


namespace relationship_abc_l171_171240

variables {a b c : ℝ}

-- Given conditions
def condition1 (a b c : ℝ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ (11/6 : ℝ) * c < a + b ∧ a + b < 2 * c
def condition2 (a b c : ℝ) : Prop := (3/2 : ℝ) * a < b + c ∧ b + c < (5/3 : ℝ) * a
def condition3 (a b c : ℝ) : Prop := (5/2 : ℝ) * b < a + c ∧ a + c < (11/4 : ℝ) * b

-- Proof statement
theorem relationship_abc (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  b < c ∧ c < a :=
by
  sorry

end relationship_abc_l171_171240


namespace evaluate_binom_mul_factorial_l171_171299

theorem evaluate_binom_mul_factorial (n : ℕ) (h : n > 0) :
  (Nat.choose (n + 2) n) * n! = ((n + 2) * (n + 1) * n!) / 2 := by
  sorry

end evaluate_binom_mul_factorial_l171_171299


namespace poly_divisible_by_seven_l171_171496

-- Define the given polynomial expression
def poly_expr (x n : ℕ) : ℕ := (1 + x)^n - 1

-- Define the proof statement
theorem poly_divisible_by_seven :
  ∀ x n : ℕ, x = 5 ∧ n = 4 → poly_expr x n % 7 = 0 :=
by
  intro x n h
  cases h
  sorry

end poly_divisible_by_seven_l171_171496


namespace magnitude_of_difference_is_3sqrt5_l171_171555

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem magnitude_of_difference_is_3sqrt5 (x : ℝ) (h_parallel : parallel vector_a (vector_b x)) :
  (Real.sqrt ((vector_a.1 - (vector_b x).1) ^ 2 + (vector_a.2 - (vector_b x).2) ^ 2)) = 3 * Real.sqrt 5 :=
sorry

end magnitude_of_difference_is_3sqrt5_l171_171555


namespace clothing_store_earnings_l171_171900

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end clothing_store_earnings_l171_171900


namespace radius_of_circle_is_zero_l171_171400

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 2 * x^2 - 8 * x + 2 * y^2 - 4 * y + 10 = 0

-- Define the goal: To prove that given this equation, the radius of the circle is 0
theorem radius_of_circle_is_zero :
  ∀ x y : ℝ, circle_eq x y → (x - 2)^2 + (y - 1)^2 = 0 :=
sorry

end radius_of_circle_is_zero_l171_171400


namespace distance_to_school_l171_171960

variables (d : ℝ)
def jog_rate := 5
def bus_rate := 30
def total_time := 1 

theorem distance_to_school :
  (d / jog_rate) + (d / bus_rate) = total_time ↔ d = 30 / 7 :=
by
  sorry

end distance_to_school_l171_171960


namespace no_such_k_l171_171476

theorem no_such_k (u : ℕ → ℝ) (v : ℕ → ℝ)
  (h1 : u 0 = 6) (h2 : v 0 = 4)
  (h3 : ∀ n, u (n + 1) = (3 / 5) * u n - (4 / 5) * v n)
  (h4 : ∀ n, v (n + 1) = (4 / 5) * u n + (3 / 5) * v n) :
  ¬ ∃ k, u k = 7 ∧ v k = 2 :=
by
  sorry

end no_such_k_l171_171476


namespace find_m_value_l171_171982

def quadratic_inequality_solution_set (a b c : ℝ) (m : ℝ) := {x : ℝ | 0 < x ∧ x < 2}

theorem find_m_value (a b c : ℝ) (m : ℝ) 
  (h1 : a = -1/2) 
  (h2 : b = 2) 
  (h3 : c = m) 
  (h4 : quadratic_inequality_solution_set a b c m = {x : ℝ | 0 < x ∧ x < 2}) : 
  m = 1 := 
sorry

end find_m_value_l171_171982


namespace minimum_disks_needed_l171_171292

-- Definition of the conditions
def disk_capacity : ℝ := 2.88
def file_sizes : List (ℝ × ℕ) := [(1.2, 5), (0.9, 10), (0.6, 8), (0.3, 7)]

/-- 
Theorem: Given the capacity of each disk and the sizes and counts of different files,
we can prove that the minimum number of disks needed to store all the files without 
splitting any file is 14.
-/
theorem minimum_disks_needed (capacity : ℝ) (files : List (ℝ × ℕ)) : 
  capacity = disk_capacity ∧ files = file_sizes → ∃ m : ℕ, m = 14 :=
by
  sorry

end minimum_disks_needed_l171_171292


namespace sequence_formula_l171_171526

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = -2) (h2 : a 2 = -1.2) :
  ∀ n, a n = 0.8 * n - 2.8 :=
by
  sorry

end sequence_formula_l171_171526


namespace units_digit_of_product_l171_171952

theorem units_digit_of_product : 
  (27 % 10 = 7) ∧ (68 % 10 = 8) → ((27 * 68) % 10 = 6) :=
by sorry

end units_digit_of_product_l171_171952


namespace compute_expression_l171_171017

theorem compute_expression : 85 * 1500 + (1 / 2) * 1500 = 128250 :=
by
  sorry

end compute_expression_l171_171017


namespace monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l171_171563

noncomputable def f (x m : ℝ) : ℝ := x - m * (x + 1) * Real.log (x + 1)

theorem monotonicity_intervals_m0 :
  ∀ x : ℝ, x > -1 → f x 0 = x - 0 * (x + 1) * Real.log (x + 1) ∧ f x 0 > 0 := 
sorry

theorem monotonicity_intervals_m_positive (m : ℝ) (hm : m > 0) :
  ∀ x : ℝ, x > -1 → 
  (f x m > f (x + e ^ ((1 - m) / m) - 1) m ∧ 
  f (x + e ^ ((1 - m) / m) - 1) m < f (x + e ^ ((1 - m) / m) - 1 + 1) m) :=
sorry

theorem intersection_points_m1 (t : ℝ) (hx_rng : -1 / 2 ≤ t ∧ t < 1) :
  (∃ x1 x2 : ℝ, x1 > -1/2 ∧ x1 ≤ 1 ∧ x2 > -1/2 ∧ x2 ≤ 1 ∧ f x1 1 = t ∧ f x2 1 = t) ↔ 
  (-1 / 2 + 1 / 2 * Real.log 2 ≤ t ∧ t < 0) :=
sorry

theorem inequality_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (1 + a) ^ b < (1 + b) ^ a :=
sorry

end monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l171_171563


namespace simplify_and_evaluate_l171_171284

variable (x y : ℤ)

noncomputable def given_expr := (x + y) ^ 2 - 3 * x * (x + y) + (x + 2 * y) * (x - 2 * y)

theorem simplify_and_evaluate : given_expr 1 (-1) = -3 :=
by
  -- The proof is to be completed here
  sorry

end simplify_and_evaluate_l171_171284


namespace choir_average_age_l171_171725

theorem choir_average_age :
  let num_females := 10
  let avg_age_females := 32
  let num_males := 18
  let avg_age_males := 35
  let num_people := num_females + num_males
  let sum_ages_females := avg_age_females * num_females
  let sum_ages_males := avg_age_males * num_males
  let total_sum_ages := sum_ages_females + sum_ages_males
  let avg_age := (total_sum_ages : ℚ) / num_people
  avg_age = 33.92857 := by
  sorry

end choir_average_age_l171_171725


namespace money_problem_l171_171210

variable {c d : ℝ}

theorem money_problem (h1 : 3 * c - 2 * d < 30) (h2 : 4 * c + d = 60) : 
  c < 150 / 11 ∧ d > 60 / 11 := 
by 
  sorry

end money_problem_l171_171210


namespace evaluate_expression_l171_171785

noncomputable def w := Complex.exp (2 * Real.pi * Complex.I / 11)

theorem evaluate_expression : (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) = 88573 := 
by 
  sorry

end evaluate_expression_l171_171785


namespace rehabilitation_centers_l171_171948

def Lisa : ℕ := 6 
def Jude : ℕ := Lisa / 2
def Han : ℕ := 2 * Jude - 2
def Jane : ℕ := 27 - Lisa - Jude - Han
def x : ℕ := 2

theorem rehabilitation_centers:
  Jane = x * Han + 6 := 
by
  -- Proof goes here (not required)
  sorry

end rehabilitation_centers_l171_171948


namespace relationship_between_x_b_a_l171_171200

variable {x b a : ℝ}

theorem relationship_between_x_b_a 
  (hx : x < 0) (hb : b < 0) (ha : a < 0)
  (hxb : x < b) (hba : b < a) : x^2 > b * x ∧ b * x > b^2 :=
by sorry

end relationship_between_x_b_a_l171_171200


namespace point_in_at_least_15_circles_l171_171059

theorem point_in_at_least_15_circles
  (C : Fin 100 → Set (ℝ × ℝ))
  (h1 : ∀ i j, ∃ p, p ∈ C i ∧ p ∈ C j)
  : ∃ p, ∃ S : Finset (Fin 100), S.card ≥ 15 ∧ ∀ i ∈ S, p ∈ C i :=
sorry

end point_in_at_least_15_circles_l171_171059


namespace charcoal_amount_l171_171627

theorem charcoal_amount (water_per_charcoal : ℕ) (charcoal_ratio : ℕ) (water_added : ℕ) (charcoal_needed : ℕ) 
  (h1 : water_per_charcoal = 30) (h2 : charcoal_ratio = 2) (h3 : water_added = 900) : charcoal_needed = 60 :=
by
  sorry

end charcoal_amount_l171_171627


namespace mistaken_fraction_l171_171272

theorem mistaken_fraction (n correct_result student_result : ℕ) (h1 : n = 384)
  (h2 : correct_result = (5 * n) / 16) (h3 : student_result = correct_result + 200) : 
  (student_result / n : ℚ) = 5 / 6 :=
by
  sorry

end mistaken_fraction_l171_171272


namespace problem_l171_171906

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end problem_l171_171906


namespace therapy_charge_l171_171845

-- Defining the conditions
variables (A F : ℝ)
variables (h1 : F = A + 25)
variables (h2 : F + 4*A = 250)

-- The statement we need to prove
theorem therapy_charge : F + A = 115 := 
by
  -- proof would go here
  sorry

end therapy_charge_l171_171845


namespace necessary_and_sufficient_condition_l171_171924

variables (x y : ℝ)

theorem necessary_and_sufficient_condition (h1 : x > y) (h2 : 1/x > 1/y) : x * y < 0 :=
sorry

end necessary_and_sufficient_condition_l171_171924


namespace RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l171_171309

-- Definitions representing the conditions
def TotalCrates : ℕ := 50
def PriceGrapes : ℕ := 15
def PriceMangoes : ℕ := 20
def PricePassionFruits : ℕ := 25
def CratesGrapes : ℕ := 13
def CratesMangoes : ℕ := 20
def CratesPassionFruits : ℕ := TotalCrates - CratesGrapes - CratesMangoes

def RevenueWithoutDiscounts : ℕ :=
  (CratesGrapes * PriceGrapes) +
  (CratesMangoes * PriceMangoes) +
  (CratesPassionFruits * PricePassionFruits)

def DiscountGrapes : Float := if CratesGrapes > 10 then 0.10 else 0.0
def DiscountMangoes : Float := if CratesMangoes > 15 then 0.15 else 0.0
def DiscountPassionFruits : Float := if CratesPassionFruits > 5 then 0.20 else 0.0

def DiscountedPrice (price : ℕ) (discount : Float) : Float := 
  price.toFloat * (1.0 - discount)

def RevenueWithDiscounts : Float :=
  (CratesGrapes.toFloat * DiscountedPrice PriceGrapes DiscountGrapes) +
  (CratesMangoes.toFloat * DiscountedPrice PriceMangoes DiscountMangoes) +
  (CratesPassionFruits.toFloat * DiscountedPrice PricePassionFruits DiscountPassionFruits)

-- Proof problems
theorem RevenueWithoutDiscounts_is_1020 : RevenueWithoutDiscounts = 1020 := sorry
theorem RevenueWithDiscounts_is_855_5 : RevenueWithDiscounts = 855.5 := sorry
theorem Difference_is_164_5 : (RevenueWithoutDiscounts.toFloat - RevenueWithDiscounts) = 164.5 := sorry

end RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l171_171309


namespace james_speed_is_16_l171_171019

theorem james_speed_is_16
  (distance : ℝ)
  (time : ℝ)
  (distance_eq : distance = 80)
  (time_eq : time = 5) :
  (distance / time = 16) :=
by
  rw [distance_eq, time_eq]
  norm_num

end james_speed_is_16_l171_171019


namespace Toph_caught_12_fish_l171_171818

-- Define the number of fish each person caught
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def average_fish : ℕ := 8
def num_people : ℕ := 3

-- The total number of fish based on the average
def total_fish : ℕ := average_fish * num_people

-- Define the number of fish Toph caught
def Toph_fish : ℕ := total_fish - Aang_fish - Sokka_fish

-- Prove that Toph caught the correct number of fish
theorem Toph_caught_12_fish : Toph_fish = 12 := sorry

end Toph_caught_12_fish_l171_171818


namespace total_fish_caught_l171_171493

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l171_171493


namespace sequence_fifth_number_l171_171154

theorem sequence_fifth_number : (5^2 - 1) = 24 :=
by {
  sorry
}

end sequence_fifth_number_l171_171154


namespace valid_configuration_exists_l171_171448

noncomputable def unique_digits (digits: List ℕ) := (digits.length = List.length (List.eraseDup digits)) ∧ ∀ (d : ℕ), d ∈ digits ↔ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem valid_configuration_exists :
  ∃ a b c d e f g h i j : ℕ,
  unique_digits [a, b, c, d, e, f, g, h, i, j] ∧
  a * (100 * b + 10 * c + d) * (100 * e + 10 * f + g) = 1000 * h + 100 * i + 10 * 9 + 71 := 
by
  sorry

end valid_configuration_exists_l171_171448


namespace smallest_largest_sum_l171_171382

theorem smallest_largest_sum (a b c : ℝ) (m M : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : m = (1/3))
  (h4 : M = 1) :
  (m + M) = 4 / 3 := by
sorry

end smallest_largest_sum_l171_171382


namespace function_value_at_minus_one_l171_171328

theorem function_value_at_minus_one :
  ( -(1:ℝ)^4 + -(1:ℝ)^3 + (1:ℝ) ) / ( -(1:ℝ)^2 + (1:ℝ) ) = 1 / 2 :=
by sorry

end function_value_at_minus_one_l171_171328


namespace xiaohua_distance_rounds_l171_171713

def length := 5
def width := 3
def perimeter (a b : ℕ) := (a + b) * 2
def total_distance (perimeter : ℕ) (laps : ℕ) := perimeter * laps

theorem xiaohua_distance_rounds :
  total_distance (perimeter length width) 3 = 30 :=
by sorry

end xiaohua_distance_rounds_l171_171713


namespace store_A_cheaper_than_store_B_l171_171580

noncomputable def store_A_full_price : ℝ := 125
noncomputable def store_A_discount_pct : ℝ := 0.08
noncomputable def store_B_full_price : ℝ := 130
noncomputable def store_B_discount_pct : ℝ := 0.10

noncomputable def final_price_A : ℝ :=
  store_A_full_price * (1 - store_A_discount_pct)

noncomputable def final_price_B : ℝ :=
  store_B_full_price * (1 - store_B_discount_pct)

theorem store_A_cheaper_than_store_B :
  final_price_B - final_price_A = 2 :=
by
  sorry

end store_A_cheaper_than_store_B_l171_171580


namespace player_A_min_score_l171_171655

theorem player_A_min_score (A B : ℕ) (hA_first_move : A = 1) (hB_next_move : B = 2) : 
  ∃ k : ℕ, k = 64 :=
by
  sorry

end player_A_min_score_l171_171655


namespace total_weight_of_packages_l171_171614

theorem total_weight_of_packages (x y z w : ℕ) (h1 : x + y + z = 150) (h2 : y + z + w = 160) (h3 : z + w + x = 170) :
  x + y + z + w = 160 :=
by sorry

end total_weight_of_packages_l171_171614


namespace find_speed_of_stream_l171_171663

-- Definitions of the conditions:
def downstream_equation (b s : ℝ) : Prop := b + s = 60
def upstream_equation (b s : ℝ) : Prop := b - s = 30

-- Theorem stating the speed of the stream given the conditions:
theorem find_speed_of_stream (b s : ℝ) (h1 : downstream_equation b s) (h2 : upstream_equation b s) : s = 15 := 
sorry

end find_speed_of_stream_l171_171663


namespace ryan_chinese_learning_hours_l171_171291

theorem ryan_chinese_learning_hours : 
    ∀ (h_english : ℕ) (diff : ℕ), 
    h_english = 7 → 
    h_english = 2 + (h_english - diff) → 
    diff = 5 := by
  intros h_english diff h_english_eq h_english_diff_eq
  sorry

end ryan_chinese_learning_hours_l171_171291


namespace negation_exists_x_squared_lt_zero_l171_171956

open Classical

theorem negation_exists_x_squared_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by 
  sorry

end negation_exists_x_squared_lt_zero_l171_171956


namespace rational_non_positive_l171_171712

variable (a : ℚ)

theorem rational_non_positive (h : ∃ a : ℚ, True) : 
  -a^2 ≤ 0 :=
by
  sorry

end rational_non_positive_l171_171712


namespace find_x_values_l171_171557

theorem find_x_values (x : ℝ) : 
  ((x + 1)^2 = 36 ∨ (x + 10)^3 = -27) ↔ (x = 5 ∨ x = -7 ∨ x = -13) :=
by
  sorry

end find_x_values_l171_171557


namespace equal_cubes_l171_171488

theorem equal_cubes (r s : ℤ) (hr : 0 ≤ r) (hs : 0 ≤ s)
  (h : |r^3 - s^3| = |6 * r^2 - 6 * s^2|) : r = s :=
by
  sorry

end equal_cubes_l171_171488


namespace box_surface_area_l171_171422

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end box_surface_area_l171_171422


namespace alpha_in_second_quadrant_l171_171793

theorem alpha_in_second_quadrant (α : ℝ) 
  (h1 : Real.sin α > Real.cos α)
  (h2 : Real.sin α * Real.cos α < 0) : 
  (Real.sin α > 0) ∧ (Real.cos α < 0) :=
by 
  -- Proof omitted
  sorry

end alpha_in_second_quadrant_l171_171793


namespace inequality_proof_l171_171733

theorem inequality_proof (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt ( (a * x) / (a * x + 8) ) ≤ 2 := 
by {
  sorry
}

end inequality_proof_l171_171733


namespace solution_positive_then_opposite_signs_l171_171806

theorem solution_positive_then_opposite_signs
  (a b : ℝ) (h : a ≠ 0) (x : ℝ) (hx : ax + b = 0) (x_pos : x > 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) :=
by
  sorry

end solution_positive_then_opposite_signs_l171_171806


namespace min_x2_y2_l171_171551

theorem min_x2_y2 (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x * y) : 
  (∃ x y, x = 0 ∧ y = 0) ∨ x^2 + y^2 >= 1 := 
sorry

end min_x2_y2_l171_171551


namespace minimum_passing_rate_l171_171171

-- Define the conditions as hypotheses
variable (total_students : ℕ)
variable (correct_q1 : ℕ)
variable (correct_q2 : ℕ)
variable (correct_q3 : ℕ)
variable (correct_q4 : ℕ)
variable (correct_q5 : ℕ)
variable (pass_threshold : ℕ)

-- Assume all percentages are converted to actual student counts based on total_students
axiom students_answered_q1_correctly : correct_q1 = total_students * 81 / 100
axiom students_answered_q2_correctly : correct_q2 = total_students * 91 / 100
axiom students_answered_q3_correctly : correct_q3 = total_students * 85 / 100
axiom students_answered_q4_correctly : correct_q4 = total_students * 79 / 100
axiom students_answered_q5_correctly : correct_q5 = total_students * 74 / 100
axiom passing_criteria : pass_threshold = 3

-- Define the main theorem statement to be proven
theorem minimum_passing_rate (total_students : ℕ) :
  (total_students - (total_students * 19 / 100 + total_students * 9 / 100 + 
  total_students * 15 / 100 + total_students * 21 / 100 + 
  total_students * 26 / 100) / pass_threshold) / total_students * 100 ≥ 70 :=
  by sorry

end minimum_passing_rate_l171_171171


namespace sum_of_coordinates_l171_171396

-- Define the points C and D and the conditions
def point_C : ℝ × ℝ := (0, 0)

def point_D (x : ℝ) : ℝ × ℝ := (x, 5)

def slope_CD (x : ℝ) : Prop :=
  (5 - 0) / (x - 0) = 3 / 4

-- The required theorem to be proved
theorem sum_of_coordinates (D : ℝ × ℝ)
  (hD : D.snd = 5)
  (h_slope : slope_CD D.fst) :
  D.fst + D.snd = 35 / 3 :=
sorry

end sum_of_coordinates_l171_171396


namespace boat_speed_in_still_water_l171_171830

-- Definitions of the conditions
def with_stream_speed : ℝ := 36
def against_stream_speed : ℝ := 8

-- Let Vb be the speed of the boat in still water, and Vs be the speed of the stream.
variable (Vb Vs : ℝ)

-- Conditions given in the problem
axiom h1 : Vb + Vs = with_stream_speed
axiom h2 : Vb - Vs = against_stream_speed

-- The statement to prove: the speed of the boat in still water is 22 km/h.
theorem boat_speed_in_still_water : Vb = 22 := by
  sorry

end boat_speed_in_still_water_l171_171830


namespace range_of_x_l171_171084

noncomputable def is_valid_x (x : ℝ) : Prop :=
  x ≥ 0 ∧ x ≠ 4

theorem range_of_x (x : ℝ) : 
  is_valid_x x ↔ x ≥ 0 ∧ x ≠ 4 :=
by sorry

end range_of_x_l171_171084


namespace door_cranking_time_l171_171239

-- Define the given conditions
def run_time_with_backpack : ℝ := 7 * 60 + 23  -- 443 seconds
def run_time_without_backpack : ℝ := 5 * 60 + 58  -- 358 seconds
def total_time : ℝ := 874  -- 874 seconds

-- Define the Lean statement of the proof
theorem door_cranking_time :
  (run_time_with_backpack + run_time_without_backpack) + (total_time - (run_time_with_backpack + run_time_without_backpack)) = total_time ∧
  (total_time - (run_time_with_backpack + run_time_without_backpack)) = 73 :=
by
  sorry

end door_cranking_time_l171_171239


namespace avg_and_variance_decrease_l171_171586

noncomputable def original_heights : List ℝ := [180, 184, 188, 190, 192, 194]
noncomputable def new_heights : List ℝ := [180, 184, 188, 190, 192, 188]

noncomputable def avg (heights : List ℝ) : ℝ :=
  heights.sum / heights.length

noncomputable def variance (heights : List ℝ) (mean : ℝ) : ℝ :=
  (heights.map (λ h => (h - mean) ^ 2)).sum / heights.length

theorem avg_and_variance_decrease :
  let original_mean := avg original_heights
  let new_mean := avg new_heights
  let original_variance := variance original_heights original_mean
  let new_variance := variance new_heights new_mean
  new_mean < original_mean ∧ new_variance < original_variance :=
by
  sorry

end avg_and_variance_decrease_l171_171586


namespace painting_time_l171_171253

-- Definitions based on the conditions
def num_people1 := 8
def num_houses1 := 3
def time1 := 12
def num_people2 := 9
def num_houses2 := 4
def k := (num_people1 * time1) / num_houses1

-- The statement we want to prove
theorem painting_time : (num_people2 * t = k * num_houses2) → (t = 128 / 9) :=
by sorry

end painting_time_l171_171253


namespace ceil_square_of_neg_fraction_l171_171969

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l171_171969


namespace condition_sufficient_but_not_necessary_l171_171932
noncomputable def sufficient_but_not_necessary (a b : ℝ) : Prop :=
∀ (a b : ℝ), a < 0 → -1 < b ∧ b < 0 → a + a * b < 0

-- Define the theorem stating the proof problem
theorem condition_sufficient_but_not_necessary (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧ 
  (a + a * b < 0 → a < 0 ∧ 1 + b > 0 ∨ a > 0 ∧ 1 + b < 0) :=
sorry

end condition_sufficient_but_not_necessary_l171_171932


namespace rationalize_denominator_l171_171683

theorem rationalize_denominator :
  ∃ (A B C : ℤ), 
  (A + B * Real.sqrt C) = (2 + Real.sqrt 5) / (3 - Real.sqrt 5) 
  ∧ A = 11 ∧ B = 5 ∧ C = 5 ∧ A * B * C = 275 := by
  sorry

end rationalize_denominator_l171_171683


namespace min_z_value_l171_171003

variable (x y z : ℝ)

theorem min_z_value (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  z = x - y → z = -1 :=
by sorry

end min_z_value_l171_171003


namespace evaluate_expression_l171_171905

theorem evaluate_expression :
  500 * 997 * 0.0997 * 10^2 = 5 * (997:ℝ)^2 :=
by
  sorry

end evaluate_expression_l171_171905


namespace david_has_15_shells_l171_171055

-- Definitions from the conditions
def mia_shells (david_shells : ℕ) : ℕ := 4 * david_shells
def ava_shells (david_shells : ℕ) : ℕ := mia_shells david_shells + 20
def alice_shells (david_shells : ℕ) : ℕ := (ava_shells david_shells) / 2

-- Total number of shells
def total_shells (david_shells : ℕ) : ℕ := david_shells + mia_shells david_shells + ava_shells david_shells + alice_shells david_shells

-- Proving the number of shells David has is 15 given the total number of shells is 195
theorem david_has_15_shells : total_shells 15 = 195 :=
by
  sorry

end david_has_15_shells_l171_171055


namespace g_interval_l171_171115

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem g_interval (a b c : ℝ) (ha : 0 < a) (hb: 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
sorry

end g_interval_l171_171115


namespace billy_reads_books_l171_171129

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l171_171129


namespace cars_meet_in_two_hours_l171_171274

theorem cars_meet_in_two_hours (t : ℝ) (d : ℝ) (v1 v2 : ℝ) (h1 : d = 60) (h2 : v1 = 13) (h3 : v2 = 17) (h4 : v1 * t + v2 * t = d) : t = 2 := 
by
  sorry

end cars_meet_in_two_hours_l171_171274


namespace six_digit_start_5_not_possible_l171_171682

theorem six_digit_start_5_not_possible :
  ∀ n : ℕ, (n ≥ 500000 ∧ n < 600000) → (¬ ∃ m : ℕ, (n * 10^6 + m) ^ 2 < 10^12 ∧ (n * 10^6 + m) ^ 2 ≥ 5 * 10^11 ∧ (n * 10^6 + m) ^ 2 < 6 * 10^11) :=
sorry

end six_digit_start_5_not_possible_l171_171682


namespace determine_b_l171_171209

theorem determine_b (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq_radicals: Real.sqrt (4 * a + 4 * b / c) = 2 * a * Real.sqrt (b / c)) : 
  b = c + 1 :=
sorry

end determine_b_l171_171209


namespace compute_value_l171_171606

noncomputable def repeating_decimal_31 : ℝ := 31 / 100000
noncomputable def repeating_decimal_47 : ℝ := 47 / 100000
def term : ℝ := 10^5 - 10^3

theorem compute_value : (term * repeating_decimal_31 + term * repeating_decimal_47) = 77.22 := 
by
  sorry

end compute_value_l171_171606


namespace sum_series_eq_two_l171_171904

noncomputable def series_term (n : ℕ) : ℚ := (3 * n - 2) / (n * (n + 1) * (n + 2))

theorem sum_series_eq_two :
  ∑' n : ℕ, series_term (n + 1) = 2 :=
sorry

end sum_series_eq_two_l171_171904


namespace employees_count_l171_171842

theorem employees_count (E M : ℝ) (h1 : M = 0.99 * E) (h2 : M - 299.9999999999997 = 0.98 * E) :
  E = 30000 :=
by sorry

end employees_count_l171_171842


namespace medians_concurrent_l171_171479

/--
For any triangle ABC, there exists a point G, known as the centroid, such that
the sum of the vectors from G to each of the vertices A, B, and C is the zero vector.
-/
theorem medians_concurrent 
  (A B C : ℝ×ℝ) : 
  ∃ G : ℝ×ℝ, (G -ᵥ A) + (G -ᵥ B) + (G -ᵥ C) = (0, 0) := 
by 
  -- proof will go here
  sorry 

end medians_concurrent_l171_171479


namespace problem1_l171_171454

theorem problem1
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l171_171454


namespace new_average_weight_l171_171940

-- Statement only
theorem new_average_weight (avg_weight_29: ℝ) (weight_new_student: ℝ) (total_students: ℕ) 
  (h1: avg_weight_29 = 28) (h2: weight_new_student = 22) (h3: total_students = 29) : 
  (avg_weight_29 * total_students + weight_new_student) / (total_students + 1) = 27.8 :=
by
  -- declare local variables for simpler proof
  let total_weight := avg_weight_29 * total_students
  let new_total_weight := total_weight + weight_new_student
  let new_total_students := total_students + 1
  have t_weight : total_weight = 812 := by sorry
  have new_t_weight : new_total_weight = 834 := by sorry
  have n_total_students : new_total_students = 30 := by sorry
  exact sorry

end new_average_weight_l171_171940


namespace probability_two_points_square_l171_171467

def gcd (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c  

theorem probability_two_points_square {a b c : ℕ} (hx : gcd a b c = 1)
  (h : (26 - Real.pi) / 32 = (a - b * Real.pi) / c) : a + b + c = 59 :=
by
  sorry

end probability_two_points_square_l171_171467


namespace domain_w_l171_171852

noncomputable def w (y : ℝ) : ℝ := (y - 3)^(1/3) + (15 - y)^(1/3)

theorem domain_w : ∀ y : ℝ, ∃ x : ℝ, w y = x := by
  sorry

end domain_w_l171_171852


namespace solid_is_cone_l171_171174

-- Define what it means for a solid to have a given view as an isosceles triangle or a circle.
structure Solid :=
(front_view : ℝ → ℝ → Prop)
(left_view : ℝ → ℝ → Prop)
(top_view : ℝ → ℝ → Prop)

-- Definition of isosceles triangle view
def isosceles_triangle (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Definition of circle view with a center
def circle_with_center (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Define the solid that satisfies the conditions in the problem
def specified_solid (s : Solid) : Prop :=
  (∀ x y, s.front_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.left_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.top_view x y → circle_with_center x y)

-- Given proof problem statement
theorem solid_is_cone (s : Solid) (h : specified_solid s) : 
  ∃ cone, cone = s :=
sorry

end solid_is_cone_l171_171174


namespace log_equality_ineq_l171_171947

--let a = \log_{\sqrt{5x-1}}(4x+1)
--let b = \log_{4x+1}\left(\frac{x}{2} + 2\right)^2
--let c = \log_{\frac{x}{2} + 2}(5x-1)

noncomputable def a (x : ℝ) : ℝ := 
  Real.log (4 * x + 1) / Real.log (Real.sqrt (5 * x - 1))

noncomputable def b (x : ℝ) : ℝ := 
  2 * (Real.log ((x / 2) + 2) / Real.log (4 * x + 1))

noncomputable def c (x : ℝ) : ℝ := 
  Real.log (5 * x - 1) / Real.log ((x / 2) + 2)

theorem log_equality_ineq (x : ℝ) : 
  a x = b x ∧ c x = a x - 1 ↔ x = 2 := 
by
  sorry

end log_equality_ineq_l171_171947


namespace cubic_root_sqrt_equation_l171_171226

theorem cubic_root_sqrt_equation (x : ℝ) (h1 : 3 - x = y^3) (h2 : x - 2 = z^2) (h3 : y + z = 1) : 
  x = 3 ∨ x = 2 ∨ x = 11 :=
sorry

end cubic_root_sqrt_equation_l171_171226


namespace juliet_age_l171_171524

theorem juliet_age
    (M J R : ℕ)
    (h1 : J = M + 3)
    (h2 : J = R - 2)
    (h3 : M + R = 19) : J = 10 := by
  sorry

end juliet_age_l171_171524


namespace probability_ratio_l171_171137

-- Conditions definitions
def total_choices := Nat.choose 50 5
def p := 10 / total_choices
def q := (Nat.choose 10 2 * Nat.choose 5 2 * Nat.choose 5 3) / total_choices

-- Statement to prove
theorem probability_ratio : q / p = 450 := by
  sorry  -- proof is omitted

end probability_ratio_l171_171137


namespace gain_percent_l171_171527

theorem gain_percent (C S S_d : ℝ) 
  (h1 : 50 * C = 20 * S) 
  (h2 : S_d = S * (1 - 0.15)) : 
  ((S_d - C) / C) * 100 = 112.5 := 
by 
  sorry

end gain_percent_l171_171527


namespace actual_average_speed_l171_171573

theorem actual_average_speed 
  (v t : ℝ)
  (h : v * t = (v + 21) * (2/3) * t) : 
  v = 42 :=
by
  sorry

end actual_average_speed_l171_171573


namespace problem_l171_171844

theorem problem (f : ℝ → ℝ) (h : ∀ x, (x - 3) * (deriv f x) ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := 
sorry

end problem_l171_171844


namespace ratio_Sarah_to_Eli_is_2_l171_171838

variable (Kaylin_age : ℕ := 33)
variable (Freyja_age : ℕ := 10)
variable (Eli_age : ℕ := Freyja_age + 9)
variable (Sarah_age : ℕ := Kaylin_age + 5)

theorem ratio_Sarah_to_Eli_is_2 : (Sarah_age : ℚ) / Eli_age = 2 := 
by 
  -- Proof would go here
  sorry

end ratio_Sarah_to_Eli_is_2_l171_171838


namespace b_l171_171955

def initial_marbles : Nat := 24
def lost_through_hole : Nat := 4
def given_away : Nat := 2 * lost_through_hole
def eaten_by_dog : Nat := lost_through_hole / 2

theorem b {m : Nat} (h₁ : m = initial_marbles - lost_through_hole)
  (h₂ : m - given_away = m₁)
  (h₃ : m₁ - eaten_by_dog = 10) :
  m₁ - eaten_by_dog = 10 := sorry

end b_l171_171955


namespace grid_satisfies_conditions_l171_171444

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l171_171444


namespace prob_even_heads_40_l171_171764

noncomputable def probability_even_heads (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (1/2) * (1 + (2/5) ^ n)

theorem prob_even_heads_40 :
  probability_even_heads 40 = 1/2 * (1 + (2/5) ^ 40) :=
by {
  sorry
}

end prob_even_heads_40_l171_171764


namespace findPositiveRealSolutions_l171_171021

noncomputable def onlySolutions (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a^2 - b * d) / (b + 2 * c + d) +
  (b^2 - c * a) / (c + 2 * d + a) +
  (c^2 - d * b) / (d + 2 * a + b) +
  (d^2 - a * c) / (a + 2 * b + c) = 0

theorem findPositiveRealSolutions :
  ∀ a b c d : ℝ,
  onlySolutions a b c d →
  ∃ k m : ℝ, k > 0 ∧ m > 0 ∧ a = k ∧ b = m ∧ c = k ∧ d = m :=
by
  intros a b c d h
  -- proof steps (if required) go here
  sorry

end findPositiveRealSolutions_l171_171021


namespace pyramid_base_is_octagon_l171_171424
-- Import necessary library

-- Declare the problem
theorem pyramid_base_is_octagon (A : Nat) (h : A = 8) : A = 8 :=
by
  -- Proof goes here
  sorry

end pyramid_base_is_octagon_l171_171424


namespace max_value_of_expression_l171_171739

-- We have three nonnegative real numbers a, b, and c,
-- such that a + b + c = 3.
def nonnegative (x : ℝ) := x ≥ 0

theorem max_value_of_expression (a b c : ℝ) (h1 : nonnegative a) (h2 : nonnegative b) (h3 : nonnegative c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 :=
  sorry

end max_value_of_expression_l171_171739


namespace find_n_l171_171492

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def twin_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ q = p + 2

def is_twins_prime_sum (n p q : ℕ) : Prop :=
  twin_primes p q ∧ is_prime (2^n + p) ∧ is_prime (2^n + q)

theorem find_n :
  ∀ (n : ℕ), (∃ (p q : ℕ), is_twins_prime_sum n p q) → (n = 1 ∨ n = 3) :=
sorry

end find_n_l171_171492


namespace soldier_initial_consumption_l171_171709

theorem soldier_initial_consumption :
  ∀ (s d1 n : ℕ) (c2 d2 : ℝ), 
    s = 1200 → d1 = 30 → n = 528 → c2 = 2.5 → d2 = 25 → 
    36000 * (x : ℝ) = 108000 → x = 3 := 
by {
  sorry
}

end soldier_initial_consumption_l171_171709


namespace ratio_evaluation_l171_171651

theorem ratio_evaluation : (5^3003 * 2^3005) / (10^3004) = 2 / 5 := by
  sorry

end ratio_evaluation_l171_171651


namespace exists_negative_value_of_f_l171_171694

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotonic (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f x < f y
axiom f_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f ((2 * x * y) / (x + y)) ≥ (f x + f y) / 2

theorem exists_negative_value_of_f : ∃ x > 0, f x < 0 := 
sorry

end exists_negative_value_of_f_l171_171694


namespace Tom_time_to_complete_wall_after_one_hour_l171_171917

noncomputable def avery_rate : ℝ := 1 / 2
noncomputable def tom_rate : ℝ := 1 / 4
noncomputable def combined_rate : ℝ := avery_rate + tom_rate
noncomputable def wall_built_in_first_hour : ℝ := combined_rate * 1
noncomputable def remaining_wall : ℝ := 1 - wall_built_in_first_hour 
noncomputable def tom_time_to_complete_remaining_wall : ℝ := remaining_wall / tom_rate

theorem Tom_time_to_complete_wall_after_one_hour : 
  tom_time_to_complete_remaining_wall = 1 :=
by
  sorry

end Tom_time_to_complete_wall_after_one_hour_l171_171917


namespace smallest_positive_leading_coefficient_l171_171613

variable {a b c : ℚ} -- Define variables a, b, c that are rational numbers
variable (P : ℤ → ℚ) -- Define the polynomial P as a function from integers to rationals

-- State that P(x) is in the form of ax^2 + bx + c
def is_quadratic_polynomial (P : ℤ → ℚ) (a b c : ℚ) :=
  ∀ x : ℤ, P x = a * x^2 + b * x + c

-- State that P(x) takes integer values for all integer x
def takes_integer_values (P : ℤ → ℚ) :=
  ∀ x : ℤ, ∃ k : ℤ, P x = k

-- The statement we want to prove
theorem smallest_positive_leading_coefficient (h1 : is_quadratic_polynomial P a b c)
                                              (h2 : takes_integer_values P) :
  ∃ a : ℚ, 0 < a ∧ ∀ b c : ℚ, is_quadratic_polynomial P a b c → takes_integer_values P → a = 1/2 :=
sorry

end smallest_positive_leading_coefficient_l171_171613


namespace A_alone_days_l171_171942

variable (x : ℝ) -- Number of days A takes to do the work alone
variable (B_rate : ℝ := 1 / 12) -- Work rate of B
variable (Together_rate : ℝ := 1 / 4) -- Combined work rate of A and B

theorem A_alone_days :
  (1 / x + B_rate = Together_rate) → (x = 6) := by
  intro h
  sorry

end A_alone_days_l171_171942


namespace esther_biking_speed_l171_171452

theorem esther_biking_speed (d x : ℝ)
  (h_bike_speed : x > 0)
  (h_average_speed : 5 = 2 * d / (d / x + d / 3)) :
  x = 15 :=
by
  sorry

end esther_biking_speed_l171_171452


namespace percentage_broken_in_second_set_l171_171190

-- Define the given conditions
def first_set_total : ℕ := 50
def first_set_broken_percent : ℚ := 0.10
def second_set_total : ℕ := 60
def total_broken : ℕ := 17

-- The proof problem statement
theorem percentage_broken_in_second_set :
  let first_set_broken := first_set_broken_percent * first_set_total
  let second_set_broken := total_broken - first_set_broken
  (second_set_broken / second_set_total) * 100 = 20 := 
sorry

end percentage_broken_in_second_set_l171_171190


namespace gcd_12a_20b_min_value_l171_171854

-- Define the conditions
def is_positive_integer (x : ℕ) : Prop := x > 0

def gcd_condition (a b d : ℕ) : Prop := gcd a b = d

-- State the problem
theorem gcd_12a_20b_min_value (a b : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_gcd_ab : gcd_condition a b 10) :
  ∃ (k : ℕ), k = gcd (12 * a) (20 * b) ∧ k = 40 :=
by
  sorry

end gcd_12a_20b_min_value_l171_171854


namespace arithmetic_sum_sequences_l171_171528

theorem arithmetic_sum_sequences (a b : ℕ → ℕ) (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h2 : ∀ n, b n = b 0 + n * (b 1 - b 0)) (h3 : a 2 + b 2 = 3) (h4 : a 4 + b 4 = 5): a 7 + b 7 = 8 := by
  sorry

end arithmetic_sum_sequences_l171_171528


namespace range_of_m_l171_171326

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0
def proposition_q (m : ℝ) : Prop := 5 - 2*m > 1

theorem range_of_m (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : m ≤ 1 :=
sorry

end range_of_m_l171_171326


namespace range_of_m_max_value_of_t_l171_171397

-- Define the conditions for the quadratic equation problem
def quadratic_eq_has_real_roots (m n : ℝ) := 
  m^2 - 4 * n ≥ 0

def roots_are_negative (m : ℝ) := 
  2 ≤ m ∧ m < 3

-- Question 1: Prove range of m
theorem range_of_m (m : ℝ) (h1 : quadratic_eq_has_real_roots m (3 - m)) : 
  roots_are_negative m :=
sorry

-- Define the conditions for the inequality problem
def quadratic_inequality (m n : ℝ) (t : ℝ) := 
  t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Question 2: Prove maximum value of t
theorem max_value_of_t (m n t : ℝ) (h1 : quadratic_eq_has_real_roots m n) : 
  quadratic_inequality m n t -> t ≤ 9/8 :=
sorry

end range_of_m_max_value_of_t_l171_171397


namespace avg_visitors_per_day_l171_171211

theorem avg_visitors_per_day :
  let visitors := [583, 246, 735, 492, 639]
  (visitors.sum / visitors.length) = 539 := by
  sorry

end avg_visitors_per_day_l171_171211


namespace general_term_of_arithmetic_seq_l171_171641

variable {a : ℕ → ℕ} 
variable {S : ℕ → ℕ}

/-- Definition of sum of first n terms of an arithmetic sequence -/
def sum_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n, a (n + 1) = a n + d

theorem general_term_of_arithmetic_seq
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 6 = 12)
  (h3 : S 3 = 12)
  (h4 : sum_of_arithmetic_sequence S a) :
  ∀ n, a n = 2 * n := 
sorry

end general_term_of_arithmetic_seq_l171_171641


namespace sum_lent_is_3000_l171_171769

noncomputable def principal_sum (P : ℕ) : Prop :=
  let R := 5
  let T := 5
  let SI := (P * R * T) / 100
  SI = P - 2250

theorem sum_lent_is_3000 : ∃ (P : ℕ), principal_sum P ∧ P = 3000 :=
by
  use 3000
  unfold principal_sum
  -- The following are the essential parts
  sorry

end sum_lent_is_3000_l171_171769


namespace not_all_angles_less_than_60_l171_171689

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end not_all_angles_less_than_60_l171_171689


namespace smallest_solution_l171_171118

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l171_171118


namespace minimum_participants_l171_171925

theorem minimum_participants
  (correct_first : ℕ)
  (correct_second : ℕ)
  (correct_third : ℕ)
  (correct_fourth : ℕ)
  (H_first : correct_first = 90)
  (H_second : correct_second = 50)
  (H_third : correct_third = 40)
  (H_fourth : correct_fourth = 20)
  (H_max_two : ∀ p : ℕ, 1 ≤ p ∧ p ≤ correct_first + correct_second + correct_third + correct_fourth → p ≤ 2 * (correct_first + correct_second + correct_third + correct_fourth))
  : ∃ n : ℕ, (correct_first + correct_second + correct_third + correct_fourth) / 2 = 100 :=
by
  sorry

end minimum_participants_l171_171925


namespace jolyn_older_than_leon_l171_171706

open Nat

def Jolyn := Nat
def Therese := Nat
def Aivo := Nat
def Leon := Nat

-- Conditions
variable (jolyn therese aivo leon : Nat)
variable (h1 : jolyn = therese + 2) -- Jolyn is 2 months older than Therese
variable (h2 : therese = aivo + 5) -- Therese is 5 months older than Aivo
variable (h3 : leon = aivo + 2) -- Leon is 2 months older than Aivo

theorem jolyn_older_than_leon :
  jolyn = leon + 5 := by
  sorry

end jolyn_older_than_leon_l171_171706


namespace fraction_to_terminating_decimal_l171_171016

-- Lean statement for the mathematical problem
theorem fraction_to_terminating_decimal: (13 : ℚ) / 200 = 0.26 := 
sorry

end fraction_to_terminating_decimal_l171_171016


namespace num_perfect_squares_diff_consecutive_under_20000_l171_171685

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end num_perfect_squares_diff_consecutive_under_20000_l171_171685


namespace count_integers_between_sqrt5_and_sqrt50_l171_171197

theorem count_integers_between_sqrt5_and_sqrt50 
  (h1 : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3)
  (h2 : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8) : 
  ∃ n : ℕ, n = 5 := 
sorry

end count_integers_between_sqrt5_and_sqrt50_l171_171197


namespace min_max_pieces_three_planes_l171_171888

theorem min_max_pieces_three_planes : 
  ∃ (min max : ℕ), (min = 4) ∧ (max = 8) := by
  sorry

end min_max_pieces_three_planes_l171_171888


namespace distance_traveled_l171_171909

theorem distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 12) : D = 100 := 
sorry

end distance_traveled_l171_171909


namespace prove_a1_geq_2k_l171_171407

variable (n k : ℕ) (a : ℕ → ℕ)
variable (h1: ∀ i, 1 ≤ i → i ≤ n → 1 < a i)
variable (h2: ∀ i j, 1 ≤ i → i < j → j ≤ n → ¬ (a i ∣ a j))
variable (h3: 3^k < 2*n ∧ 2*n < 3^(k + 1))

theorem prove_a1_geq_2k : a 1 ≥ 2^k :=
by
  sorry

end prove_a1_geq_2k_l171_171407


namespace total_amount_paid_correct_l171_171028

-- Definitions of quantities and rates
def quantity_grapes := 3
def rate_grapes := 70
def quantity_mangoes := 9
def rate_mangoes := 55

-- Total amount calculation
def total_amount_paid := quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes

-- Theorem to prove total amount paid is 705
theorem total_amount_paid_correct : total_amount_paid = 705 :=
by
  sorry

end total_amount_paid_correct_l171_171028


namespace number_of_ways_difference_of_squares_l171_171951

-- Lean statement
theorem number_of_ways_difference_of_squares (n k : ℕ) (h1 : n > 10^k) (h2 : n % 10^k = 0) (h3 : k ≥ 2) :
  ∃ D, D = k^2 - 1 ∧ ∀ (a b : ℕ), n = a^2 - b^2 → D = k^2 - 1 :=
by
  sorry

end number_of_ways_difference_of_squares_l171_171951


namespace quadratic_eq_l171_171634

noncomputable def roots (r s : ℝ): Prop := r + s = 12 ∧ r * s = 27 ∧ (r = 2 * s ∨ s = 2 * r)

theorem quadratic_eq (r s : ℝ) (h : roots r s) : 
   Polynomial.C 1 * (X^2 - Polynomial.C (r + s) * X + Polynomial.C (r * s)) = X ^ 2 - 12 * X + 27 := 
sorry

end quadratic_eq_l171_171634


namespace solve_for_a_l171_171482

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x >= 0 then 4 ^ x else 2 ^ (a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 := 
by {
  sorry
}

end solve_for_a_l171_171482


namespace rational_roots_iff_a_eq_b_l171_171395

theorem rational_roots_iff_a_eq_b (a b : ℤ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℚ, x^2 + (a + b)^2 * x + 4 * a * b = 1) ↔ a = b :=
by
  sorry

end rational_roots_iff_a_eq_b_l171_171395


namespace gcd_of_XY_is_6_l171_171105

theorem gcd_of_XY_is_6 (X Y : ℕ) (h1 : Nat.lcm X Y = 180)
  (h2 : X * 6 = Y * 5) : Nat.gcd X Y = 6 :=
sorry

end gcd_of_XY_is_6_l171_171105


namespace directrix_of_parabola_l171_171486

-- Define the parabola and the line conditions
def parabola (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def focus_line (x y : ℝ) := 2 * x + 3 * y - 8 = 0

-- Theorem stating that the directrix of the parabola is x = -4
theorem directrix_of_parabola (p : ℝ) (hx : ∃ x, ∃ y, focus_line x y) (hp : parabola p) :
  ∃ k : ℝ, k = 4 → ∀ x y : ℝ, (-x) = -4 :=
by
  sorry

end directrix_of_parabola_l171_171486


namespace combined_area_win_bonus_l171_171894

theorem combined_area_win_bonus (r : ℝ) (P_win P_bonus : ℝ) : 
  r = 8 → P_win = 1 / 4 → P_bonus = 1 / 8 → 
  (P_win * (Real.pi * r^2) + P_bonus * (Real.pi * r^2) = 24 * Real.pi) :=
by
  intro h_r h_Pwin h_Pbonus
  rw [h_r, h_Pwin, h_Pbonus]
  -- Calculation is skipped as per the instructions
  sorry

end combined_area_win_bonus_l171_171894


namespace no_rational_roots_l171_171139

theorem no_rational_roots (p q : ℤ) (h1 : p % 3 = 2) (h2 : q % 3 = 2) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ a * a = b * b * (p^2 - 4 * q) :=
by
  sorry

end no_rational_roots_l171_171139


namespace find_X_value_l171_171552

-- Given definitions and conditions
def X (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def S (n : ℕ) : ℕ := n * (n + 2)

-- Proposition we need to prove
theorem find_X_value : ∃ n : ℕ, S n ≥ 10000 ∧ X n = 201 :=
by
  -- Placeholder for proof
  sorry

end find_X_value_l171_171552


namespace probability_red_side_given_observed_l171_171592

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end probability_red_side_given_observed_l171_171592


namespace mutually_exclusive_but_not_opposite_l171_171341

-- Define the cards and the people
inductive Card
| Red
| Black
| Blue
| White

inductive Person
| A
| B
| C
| D

-- Define the events
def eventA_gets_red (distribution : Person → Card) : Prop :=
distribution Person.A = Card.Red

def eventB_gets_red (distribution : Person → Card) : Prop :=
distribution Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Prop) : Prop :=
P → ¬ Q

-- Statement of the problem
theorem mutually_exclusive_but_not_opposite :
  ∀ (distribution : Person → Card), 
    mutually_exclusive (eventA_gets_red distribution) (eventB_gets_red distribution) ∧ 
    ¬ (eventA_gets_red distribution ↔ eventB_gets_red distribution) :=
by sorry

end mutually_exclusive_but_not_opposite_l171_171341


namespace original_number_increase_l171_171046

theorem original_number_increase (x : ℝ) (h : 1.20 * x = 1800) : x = 1500 :=
by
  sorry

end original_number_increase_l171_171046


namespace solution_set_of_inequality_l171_171251

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) > -2) ↔ (x < 3 / 2 ∨ x > 2) :=
by sorry

end solution_set_of_inequality_l171_171251


namespace nuts_in_tree_l171_171437

theorem nuts_in_tree (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : squirrels = nuts + 2) : nuts = 2 :=
by
  sorry

end nuts_in_tree_l171_171437


namespace will_jogged_for_30_minutes_l171_171978

theorem will_jogged_for_30_minutes 
  (calories_before : ℕ)
  (calories_per_minute : ℕ)
  (net_calories_after : ℕ)
  (h1 : calories_before = 900)
  (h2 : calories_per_minute = 10)
  (h3 : net_calories_after = 600) :
  let calories_burned := calories_before - net_calories_after
  let jogging_time := calories_burned / calories_per_minute
  jogging_time = 30 := by
  sorry

end will_jogged_for_30_minutes_l171_171978


namespace proof_quotient_l171_171229

/-- Let x be in the form (a + b * sqrt c) / d -/
def x_form (a b c d : ℤ) (x : ℝ) : Prop := x = (a + b * Real.sqrt c) / d

/-- Main theorem -/
theorem proof_quotient (a b c d : ℤ) (x : ℝ) (h_eq : 4 * x / 5 + 2 = 5 / x) (h_form : x_form a b c d x) : (a * c * d) / b = -20 := by
  sorry

end proof_quotient_l171_171229


namespace john_walks_further_than_nina_l171_171355

theorem john_walks_further_than_nina :
  let john_distance := 0.7
  let nina_distance := 0.4
  john_distance - nina_distance = 0.3 :=
by
  sorry

end john_walks_further_than_nina_l171_171355


namespace mooncake_packaging_problem_l171_171656

theorem mooncake_packaging_problem
  (x y : ℕ)
  (L : ℕ := 9)
  (S : ℕ := 4)
  (M : ℕ := 35)
  (h1 : L = 9)
  (h2 : S = 4)
  (h3 : M = 35) :
  9 * x + 4 * y = 35 ∧ x + y = 5 := 
by
  sorry

end mooncake_packaging_problem_l171_171656


namespace product_of_repeating_decimal_l171_171083

theorem product_of_repeating_decimal 
  (t : ℚ) 
  (h : t = 456 / 999) : 
  8 * t = 1216 / 333 :=
by
  sorry

end product_of_repeating_decimal_l171_171083


namespace simplify_div_expression_evaluate_at_2_l171_171807

variable (a : ℝ)

theorem simplify_div_expression (h0 : a ≠ 0) (h1 : a ≠ 1) :
  (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) :=
by
  sorry

theorem evaluate_at_2 : (1 - 1 / 2) / ((2^2 - 2 * 2 + 1) / 2) = 1 :=
by 
  sorry

end simplify_div_expression_evaluate_at_2_l171_171807


namespace actual_average_height_is_correct_l171_171116

-- Definitions based on given conditions
def number_of_students : ℕ := 20
def incorrect_average_height : ℝ := 175.0
def incorrect_height_of_student : ℝ := 151.0
def actual_height_of_student : ℝ := 136.0

-- Prove that the actual average height is 174.25 cm
theorem actual_average_height_is_correct :
  (incorrect_average_height * number_of_students - (incorrect_height_of_student - actual_height_of_student)) / number_of_students = 174.25 :=
sorry

end actual_average_height_is_correct_l171_171116


namespace find_b_l171_171867

def oscillation_period (a b c d : ℝ) (oscillations : ℝ) : Prop :=
  oscillations = 5 * (2 * Real.pi) / b

theorem find_b
  (a b c d : ℝ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0)
  (osc_complexity: oscillation_period a b c d 5):
  b = 5 := by
  sorry

end find_b_l171_171867


namespace tan_half_angle_is_two_l171_171334

-- Define the setup
variables (α : ℝ) (H1 : α ∈ Icc (π/2) π) (H2 : 3 * Real.sin α + 4 * Real.cos α = 0)

-- Define the main theorem
theorem tan_half_angle_is_two : Real.tan (α / 2) = 2 :=
sorry

end tan_half_angle_is_two_l171_171334


namespace equilateral_triangle_bound_l171_171728

theorem equilateral_triangle_bound (n k : ℕ) (h_n_gt_3 : n > 3) 
  (h_k_triangles : ∃ T : Finset (Finset (ℝ × ℝ)), T.card = k ∧ ∀ t ∈ T, 
  ∃ a b c : (ℝ × ℝ), t = {a, b, c} ∧ dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1) :
  k < (2 * n) / 3 :=
by
  sorry

end equilateral_triangle_bound_l171_171728


namespace number_is_24point2_l171_171529

noncomputable def certain_number (x : ℝ) : Prop :=
  0.12 * x = 2.904

theorem number_is_24point2 : certain_number 24.2 :=
by
  unfold certain_number
  sorry

end number_is_24point2_l171_171529


namespace problem_statement_l171_171185

theorem problem_statement (x : ℝ) (h : x^3 - 3 * x = 7) : x^7 + 27 * x^2 = 76 * x^2 + 270 * x + 483 :=
sorry

end problem_statement_l171_171185


namespace triangle_isosceles_l171_171856

theorem triangle_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) :
  b = c → IsoscelesTriangle := 
by
  sorry

end triangle_isosceles_l171_171856


namespace income_of_person_l171_171231

theorem income_of_person (x: ℝ) (h : 9 * x - 8 * x = 2000) : 9 * x = 18000 :=
by
  sorry

end income_of_person_l171_171231


namespace power_function_properties_l171_171112

def power_function (f : ℝ → ℝ) (x : ℝ) (a : ℝ) : Prop :=
  f x = x ^ a

theorem power_function_properties :
  ∃ (f : ℝ → ℝ) (a : ℝ), power_function f 2 a ∧ f 2 = 1/2 ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → 
    (f x1 + f x2) / 2 > f ((x1 + x2) / 2)) :=
sorry

end power_function_properties_l171_171112


namespace find_puppy_weight_l171_171155

noncomputable def weight_problem (a b c : ℕ) : Prop :=
  a + b + c = 36 ∧ a + c = 3 * b ∧ a + b = c + 6

theorem find_puppy_weight (a b c : ℕ) (h : weight_problem a b c) : a = 12 :=
sorry

end find_puppy_weight_l171_171155


namespace arctan_combination_l171_171149

noncomputable def find_m : ℕ :=
  133

theorem arctan_combination :
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/find_m)) = (Real.pi / 4) :=
by
  sorry

end arctan_combination_l171_171149


namespace bounded_sequence_range_l171_171615

theorem bounded_sequence_range (a : ℝ) (a_n : ℕ → ℝ) (h1 : a_n 1 = a)
    (hrec : ∀ n : ℕ, a_n (n + 1) = 3 * (a_n n)^3 - 7 * (a_n n)^2 + 5 * (a_n n))
    (bounded : ∃ M : ℝ, ∀ n : ℕ, abs (a_n n) ≤ M) :
    0 ≤ a ∧ a ≤ 4/3 :=
by
  sorry

end bounded_sequence_range_l171_171615


namespace sufficient_but_not_necessary_condition_l171_171837

theorem sufficient_but_not_necessary_condition (x : ℝ) : (0 < x ∧ x < 5) → |x - 2| < 3 :=
by
  sorry

end sufficient_but_not_necessary_condition_l171_171837


namespace inequality_solution_set_system_of_inequalities_solution_set_l171_171340

theorem inequality_solution_set (x : ℝ) (h : 3 * x - 5 > 5 * x + 3) : x < -4 :=
by sorry

theorem system_of_inequalities_solution_set (x : ℤ) 
  (h₁ : x - 1 ≥ 1 - x) 
  (h₂ : x + 8 > 4 * x - 1) : x = 1 ∨ x = 2 :=
by sorry

end inequality_solution_set_system_of_inequalities_solution_set_l171_171340


namespace age_difference_l171_171076

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end age_difference_l171_171076


namespace time_for_c_l171_171679

theorem time_for_c (a b work_completion: ℝ) (ha : a = 16) (hb : b = 6) (habc : work_completion = 3.2) : 
  (12 : ℝ) = 
  (48 * work_completion - 48) / 4 := 
sorry

end time_for_c_l171_171679


namespace fraction_evaluation_l171_171273

theorem fraction_evaluation : (1 / 2) + (1 / 2 * 1 / 2) = 3 / 4 := by
  sorry

end fraction_evaluation_l171_171273


namespace greatest_q_minus_r_l171_171466

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1001 = 17 * q + r ∧ q - r = 43 :=
by
  sorry

end greatest_q_minus_r_l171_171466


namespace tan_alpha_expression_l171_171432

theorem tan_alpha_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 :=
by
  sorry

end tan_alpha_expression_l171_171432


namespace rate_of_current_l171_171120

theorem rate_of_current : 
  ∀ (v c : ℝ), v = 3.3 → (∀ d: ℝ, d > 0 → (d / (v - c) = 2 * (d / (v + c))) → c = 1.1) :=
by
  intros v c hv h
  sorry

end rate_of_current_l171_171120


namespace pentagon_same_parity_l171_171851

open Classical

theorem pentagon_same_parity (vertices : Fin 5 → ℤ × ℤ) : 
  ∃ i j : Fin 5, i ≠ j ∧ (vertices i).1 % 2 = (vertices j).1 % 2 ∧ (vertices i).2 % 2 = (vertices j).2 % 2 :=
by
  sorry

end pentagon_same_parity_l171_171851


namespace pq_even_impossible_l171_171594

theorem pq_even_impossible {p q : ℤ} (h : (p^2 + q^2 + p*q) % 2 = 1) : ¬(p % 2 = 0 ∧ q % 2 = 0) :=
by
  sorry

end pq_even_impossible_l171_171594


namespace circles_radii_divide_regions_l171_171847

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l171_171847


namespace problem_statement_l171_171686

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end problem_statement_l171_171686


namespace sequence_area_formula_l171_171332

open Real

noncomputable def S_n (n : ℕ) : ℝ := (8 / 5) - (3 / 5) * (4 / 9) ^ n

theorem sequence_area_formula (n : ℕ) :
  S_n n = (8 / 5) - (3 / 5) * (4 / 9) ^ n := sorry

end sequence_area_formula_l171_171332


namespace neither_sufficient_nor_necessary_l171_171692

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0) ↔ (ab > 0)) := 
sorry

end neither_sufficient_nor_necessary_l171_171692


namespace paul_money_duration_l171_171943

theorem paul_money_duration
  (mow_earnings : ℕ)
  (weed_earnings : ℕ)
  (weekly_expenses : ℕ)
  (earnings_mow : mow_earnings = 3)
  (earnings_weed : weed_earnings = 3)
  (expenses : weekly_expenses = 3) :
  (mow_earnings + weed_earnings) / weekly_expenses = 2 := 
by
  sorry

end paul_money_duration_l171_171943


namespace half_abs_diff_squares_l171_171455

/-- Half of the absolute value of the difference of the squares of 23 and 19 is 84. -/
theorem half_abs_diff_squares : (1 / 2 : ℝ) * |(23^2 : ℝ) - (19^2 : ℝ)| = 84 :=
by
  sorry

end half_abs_diff_squares_l171_171455


namespace ceil_evaluation_l171_171305

theorem ceil_evaluation : 
  (Int.ceil (4 * (8 - 1 / 3 : ℚ))) = 31 :=
by
  sorry

end ceil_evaluation_l171_171305


namespace quadratic_real_root_iff_b_range_l171_171101

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l171_171101


namespace proposition_4_correct_l171_171478

section

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Definitions of perpendicular and parallel relationships
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (x y : Line) : Prop := sorry

theorem proposition_4_correct (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end

end proposition_4_correct_l171_171478


namespace unique_root_iff_k_eq_4_l171_171322

theorem unique_root_iff_k_eq_4 (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4 * x + k = 0) ↔ k = 4 := 
by {
  sorry
}

end unique_root_iff_k_eq_4_l171_171322


namespace length_of_midsegment_l171_171436

/-- Given a quadrilateral ABCD where sides AB and CD are parallel with lengths 7 and 3 
    respectively, and the other sides BC and DA are of lengths 5 and 4 respectively, 
    prove that the length of the segment joining the midpoints of sides BC and DA is 5. -/
theorem length_of_midsegment (A B C D : ℝ × ℝ)
  (HAB : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 7 ∧ B.2 = 0)
  (HBC : dist B C = 5)
  (HCD : dist C D = 3)
  (HDA : dist D A = 4)
  (Hparallel : B.2 = 0 ∧ D.2 ≠ 0 → C.2 = D.2) :
  dist ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ((A.1 + D.1) / 2, (A.2 + D.2) / 2) = 5 :=
sorry

end length_of_midsegment_l171_171436


namespace total_loss_is_correct_l171_171413

variable (A P : ℝ)
variable (Ashok_loss Pyarelal_loss : ℝ)

-- Condition 1: Ashok's capital is 1/9 of Pyarelal's capital
def ashokCapital (A P : ℝ) : Prop :=
  A = (1 / 9) * P

-- Condition 2: Pyarelal's loss was Rs 1800
def pyarelalLoss (Pyarelal_loss : ℝ) : Prop :=
  Pyarelal_loss = 1800

-- Question: What was the total loss in the business?
def totalLoss (Ashok_loss Pyarelal_loss : ℝ) : ℝ :=
  Ashok_loss + Pyarelal_loss

-- The mathematically equivalent proof problem statement
theorem total_loss_is_correct (P A : ℝ) (Ashok_loss Pyarelal_loss : ℝ)
  (h1 : ashokCapital A P)
  (h2 : pyarelalLoss Pyarelal_loss)
  (h3 : Ashok_loss = (1 / 9) * Pyarelal_loss) :
  totalLoss Ashok_loss Pyarelal_loss = 2000 := by
  sorry

end total_loss_is_correct_l171_171413


namespace torn_pages_are_112_and_113_l171_171196

theorem torn_pages_are_112_and_113 (n k : ℕ) (S S' : ℕ) 
  (h1 : S = n * (n + 1) / 2)
  (h2 : S' = S - (k - 1) - k)
  (h3 : S' = 15000) :
  (k = 113) ∧ (k - 1 = 112) :=
by
  sorry

end torn_pages_are_112_and_113_l171_171196


namespace ratio_y_to_x_l171_171349

variable (x y z : ℝ)

-- Conditions
def condition1 (x y z : ℝ) := 0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)
def condition2 (y z : ℝ) := ∃ k : ℝ, z = k * y
def condition3 (y z : ℝ) := z = 7 * y
def condition4 (x y : ℝ) := y = 5 * x / 7

theorem ratio_y_to_x (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 y z) (h3 : condition3 y z) (h4 : condition4 x y) : y / x = 5 / 7 :=
by
  sorry

end ratio_y_to_x_l171_171349


namespace sequence_properties_l171_171981

noncomputable def a (n : ℕ) : ℕ := 3 * n - 1

def S (n : ℕ) : ℕ := n * (2 + 3 * n - 1) / 2

theorem sequence_properties :
  a 5 + a 7 = 34 ∧ ∀ n, S n = (3 * n ^ 2 + n) / 2 :=
by
  sorry

end sequence_properties_l171_171981


namespace determine_x_l171_171593

variable (A B C x : ℝ)
variable (hA : A = x)
variable (hB : B = 2 * x)
variable (hC : C = 45)
variable (hSum : A + B + C = 180)

theorem determine_x : x = 45 := 
by
  -- proof steps would go here
  sorry

end determine_x_l171_171593


namespace sector_area_l171_171336

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = π / 3) (hr : r = 4) : 
  (1/2) * (r * θ) * r = 8 * π / 3 :=
by
  -- Implicitly use the given values of θ and r by substituting them in the expression.
  sorry

end sector_area_l171_171336


namespace ratio_of_width_to_length_l171_171377

theorem ratio_of_width_to_length (w l : ℕ) (h1 : w * l = 800) (h2 : l - w = 20) : w / l = 1 / 2 :=
by sorry

end ratio_of_width_to_length_l171_171377


namespace intersection_of_A_and_B_l171_171824

def A : Set ℝ := { x | x^2 - x > 0 }
def B : Set ℝ := { x | Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B : A ∩ B = { x | 1 < x ∧ x < 4 } :=
by sorry

end intersection_of_A_and_B_l171_171824


namespace value_of_f_g_6_squared_l171_171286

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem value_of_f_g_6_squared : (f (g 6))^2 = 26569 :=
by
  -- Place your proof here
  sorry

end value_of_f_g_6_squared_l171_171286


namespace parabola_tangent_line_l171_171511

noncomputable def gcd (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem parabola_tangent_line (a b c : ℕ) (h1 : a^2 + (104 / 5) * a * b - 4 * b * c = 0)
  (h2 : b^2 - 5 * a^2 + 4 * a * c = 0) (hgcd : gcd a b c = 1) :
  a + b + c = 17 := by
  sorry

end parabola_tangent_line_l171_171511


namespace toms_weekly_earnings_l171_171278

variable (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def total_money_per_week (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_per_week

theorem toms_weekly_earnings :
  total_money_per_week 8 12 5 7 = 3360 :=
by
  sorry

end toms_weekly_earnings_l171_171278


namespace largest_possible_value_b_l171_171153

theorem largest_possible_value_b : 
  ∃ b : ℚ, (3 * b + 7) * (b - 2) = 4 * b ∧ b = 40 / 15 := 
by 
  sorry

end largest_possible_value_b_l171_171153


namespace aeroplane_speed_l171_171431

theorem aeroplane_speed (D : ℝ) (S : ℝ) (h1 : D = S * 6) (h2 : D = 540 * (14 / 3)) :
  S = 420 := by
  sorry

end aeroplane_speed_l171_171431


namespace measure_of_RPS_l171_171790

-- Assume the elements of the problem
variables {Q R P S : Type}

-- Angles in degrees
def angle_PQS := 35
def angle_QPR := 80
def angle_PSQ := 40

-- Define the angles and the straight line condition
def QRS_straight_line : Prop := true  -- This definition is trivial for a straight line

-- Measure of angle QPS using sum of angles in triangle
noncomputable def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Measure of angle RPS derived from the previous steps
noncomputable def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The statement of the problem in Lean
theorem measure_of_RPS : angle_RPS = 25 := by
  sorry

end measure_of_RPS_l171_171790


namespace apples_sold_by_noon_l171_171768

theorem apples_sold_by_noon 
  (k g c l : ℕ) 
  (hk : k = 23) 
  (hg : g = 37) 
  (hc : c = 14) 
  (hl : l = 38) :
  k + g + c - l = 36 := 
by
  -- k = 23
  -- g = 37
  -- c = 14
  -- l = 38
  -- k + g + c - l = 36

  sorry

end apples_sold_by_noon_l171_171768


namespace ribbon_left_l171_171671

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end ribbon_left_l171_171671


namespace arithmetic_progression_squares_l171_171568

theorem arithmetic_progression_squares :
  ∃ (n : ℤ), ((3 * n^2 + 8 = 1111 * 5) ∧ (n-2, n, n+2) = (41, 43, 45)) :=
by
  sorry

end arithmetic_progression_squares_l171_171568


namespace find_a_l171_171178

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a = 0 }

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end find_a_l171_171178


namespace always_odd_l171_171339

theorem always_odd (p m : ℕ) (hp : p % 2 = 1) : (p^3 + 3*p*m^2 + 2*m) % 2 = 1 := 
by sorry

end always_odd_l171_171339


namespace hannah_jerry_difference_l171_171279

-- Define the calculations of Hannah (H) and Jerry (J)
def H : Int := 10 - (3 * 4)
def J : Int := 10 - 3 + 4

-- Prove that H - J = -13
theorem hannah_jerry_difference : H - J = -13 := by
  sorry

end hannah_jerry_difference_l171_171279


namespace soccer_league_teams_l171_171742

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 :=
by
  -- Proof will go here
  sorry

end soccer_league_teams_l171_171742


namespace increase_80_by_135_percent_l171_171896

theorem increase_80_by_135_percent : 
  let original := 80 
  let increase := 1.35 
  original + (increase * original) = 188 := 
by
  sorry

end increase_80_by_135_percent_l171_171896


namespace max_triangle_area_l171_171518

theorem max_triangle_area :
  ∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 2 ≤ c ∧ c ≤ 3 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧ (1 ≤ 0.5 * a * b) := sorry

end max_triangle_area_l171_171518


namespace ratio_black_haired_children_l171_171937

theorem ratio_black_haired_children 
  (n_red : ℕ) (n_total : ℕ) (ratio_red : ℕ) (ratio_blonde : ℕ) (ratio_black : ℕ)
  (h_ratio : ratio_red / ratio_red = 1 ∧ ratio_blonde / ratio_red = 2 ∧ ratio_black / ratio_red = 7 / 3)
  (h_n_red : n_red = 9)
  (h_n_total : n_total = 48) :
  (7 : ℚ) / (16 : ℚ) = (n_total * 7 / 16 : ℚ) :=
sorry

end ratio_black_haired_children_l171_171937
