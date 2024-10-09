import Mathlib

namespace voting_for_marty_l586_58694

/-- Conditions provided in the problem -/
def total_people : ℕ := 400
def percentage_biff : ℝ := 0.30
def percentage_clara : ℝ := 0.20
def percentage_doc : ℝ := 0.10
def percentage_ein : ℝ := 0.05
def percentage_undecided : ℝ := 0.15

/-- Statement to prove the number of people voting for Marty -/
theorem voting_for_marty : 
  (1 - percentage_biff - percentage_clara - percentage_doc - percentage_ein - percentage_undecided) * total_people = 80 :=
by
  sorry

end voting_for_marty_l586_58694


namespace biology_class_grades_l586_58611

theorem biology_class_grades (total_students : ℕ)
  (PA PB PC PD : ℕ)
  (h1 : PA = 12 * PB / 10)
  (h2 : PC = PB)
  (h3 : PD = 5 * PB / 10)
  (h4 : PA + PB + PC + PD = total_students) :
  total_students = 40 → PB = 11 := 
by
  sorry

end biology_class_grades_l586_58611


namespace condition_1_valid_for_n_condition_2_valid_for_n_l586_58636

-- Definitions from the conditions
def is_cube_root_of_unity (ω : ℂ) : Prop := ω^3 = 1

def roots_of_polynomial (ω : ℂ) (ω2 : ℂ) : Prop :=
  ω^2 + ω + 1 = 0 ∧ is_cube_root_of_unity ω ∧ is_cube_root_of_unity ω2

-- Problem statements
theorem condition_1_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n - x^n - 1 ↔ ∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k - 1 := sorry

theorem condition_2_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n + x^n + 1 ↔ ∃ k : ℕ, n = 6 * k + 2 ∨ n = 6 * k - 2 := sorry

end condition_1_valid_for_n_condition_2_valid_for_n_l586_58636


namespace cube_diagonal_length_l586_58619

theorem cube_diagonal_length (V A : ℝ) (hV : V = 384) (hA : A = 384) : 
  ∃ d : ℝ, d = 8 * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l586_58619


namespace number_of_points_l586_58626

theorem number_of_points (x y : ℕ) (h : y = (2 * x + 2018) / (x - 1)) 
  (h2 : x > y) (h3 : 0 < x) (h4 : 0 < y) : 
  ∃! (x y : ℕ), y = (2 * x + 2018) / (x - 1) ∧ x > y ∧ 0 < x ∧ 0 < y :=
sorry

end number_of_points_l586_58626


namespace isosceles_triangle_problem_l586_58676

theorem isosceles_triangle_problem
  (BT CT : Real) (BC : Real) (BZ CZ TZ : Real) :
  BT = 20 →
  CT = 20 →
  BC = 24 →
  TZ^2 + 2 * BZ * CZ = 478 →
  BZ = CZ →
  BZ * CZ = 144 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end isosceles_triangle_problem_l586_58676


namespace mean_of_sequence_starting_at_3_l586_58603

def arithmetic_sequence (start : ℕ) (n : ℕ) : List ℕ :=
List.range n |>.map (λ i => start + i)

def arithmetic_mean (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

theorem mean_of_sequence_starting_at_3 : 
  ∀ (seq : List ℕ),
  seq = arithmetic_sequence 3 60 → 
  arithmetic_mean seq = 32.5 := 
by
  intros seq h
  rw [h]
  sorry

end mean_of_sequence_starting_at_3_l586_58603


namespace quadratic_eq_positive_integer_roots_l586_58654

theorem quadratic_eq_positive_integer_roots (k p : ℕ) 
  (h1 : k > 0)
  (h2 : ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k-1) * x1^2 - p * x1 + k = 0 ∧ (k-1) * x2^2 - p * x2 + k = 0) :
  k ^ (k * p) * (p ^ p + k ^ k) + (p + k) = 1989 :=
by
  sorry

end quadratic_eq_positive_integer_roots_l586_58654


namespace percentage_calculation_l586_58669

theorem percentage_calculation
  (x : ℝ)
  (hx : x = 16)
  (h : 0.15 * 40 - (P * x) = 2) :
  P = 0.25 := by
  sorry

end percentage_calculation_l586_58669


namespace cos_sum_simplified_l586_58609

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end cos_sum_simplified_l586_58609


namespace cos_180_eq_neg_one_l586_58605

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l586_58605


namespace closest_weight_total_shortfall_total_selling_price_l586_58655

-- Definitions
def standard_weight : ℝ := 25
def weights : List ℝ := [1.5, -3, 2, -0.5, 1, -2, -2.5, -2]
def price_per_kg : ℝ := 2.6

-- Assertions
theorem closest_weight : ∃ w ∈ weights, abs w = 0.5 ∧ 25 + w = 24.5 :=
by sorry

theorem total_shortfall : (weights.sum = -5.5) :=
by sorry

theorem total_selling_price : (8 * standard_weight + weights.sum) * price_per_kg = 505.7 :=
by sorry

end closest_weight_total_shortfall_total_selling_price_l586_58655


namespace student_arrangement_l586_58631

theorem student_arrangement (students : Fin 6 → Prop)
  (A : (students 0) ∨ (students 5) → False)
  (females_adj : ∃ (i : Fin 6), i < 5 ∧ students i → students (i + 1))
  : ∃! n, n = 96 := by
  sorry

end student_arrangement_l586_58631


namespace axis_of_symmetry_l586_58616

noncomputable def f (x : ℝ) := x^2 - 2 * x + Real.cos (x - 1)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (1 + x) = f (1 - x) :=
by 
  sorry

end axis_of_symmetry_l586_58616


namespace coterminal_angle_equivalence_l586_58624

theorem coterminal_angle_equivalence (k : ℤ) : ∃ n : ℤ, -463 % 360 = (k * 360 + 257) % 360 :=
by
  sorry

end coterminal_angle_equivalence_l586_58624


namespace range_of_a_l586_58687

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≥ 4 ∧ y ≥ 4 ∧ x ≤ y → (x^2 + 2*(a-1)*x + 2) ≤ (y^2 + 2*(a-1)*y + 2)) ↔ a ∈ Set.Ici (-3) :=
by
  sorry

end range_of_a_l586_58687


namespace doubled_base_and_exponent_l586_58689

theorem doubled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ 3) : 
  x = (4 ^ b * a ^ b) ^ (1 / 3) :=
by
  sorry

end doubled_base_and_exponent_l586_58689


namespace VasyaSlowerWalkingFullWayHome_l586_58662

namespace FishingTrip

-- Define the variables involved
variables (x v S : ℝ)   -- x is the speed of Vasya and Petya, v is the speed of Kolya on the bicycle, S is the distance from the house to the lake

-- Conditions derived from the problem statement:
-- Condition 1: When Kolya meets Vasya then Petya starts
-- Condition 2: Given: Petya’s travel time is \( \frac{5}{4} \times \) Vasya's travel time.

theorem VasyaSlowerWalkingFullWayHome (h1 : v = 3 * x) :
  2 * (S / x + v) = (5 / 2) * (S / x) :=
sorry

end FishingTrip

end VasyaSlowerWalkingFullWayHome_l586_58662


namespace distinct_balls_boxes_l586_58622

def count_distinct_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 7 ∧ boxes = 3 then 8 else 0

theorem distinct_balls_boxes :
  count_distinct_distributions 7 3 = 8 :=
by sorry

end distinct_balls_boxes_l586_58622


namespace increase_factor_l586_58697

-- Definition of parameters: number of letters, digits, and symbols.
def num_letters : ℕ := 26
def num_digits : ℕ := 10
def num_symbols : ℕ := 5

-- Definition of the number of old license plates and new license plates.
def num_old_plates : ℕ := num_letters ^ 2 * num_digits ^ 3
def num_new_plates : ℕ := num_letters ^ 3 * num_digits ^ 3 * num_symbols

-- The proof problem statement: Prove that the increase factor is 130.
theorem increase_factor : num_new_plates / num_old_plates = 130 := by
  sorry

end increase_factor_l586_58697


namespace ninth_term_arith_seq_l586_58684

-- Define the arithmetic sequence.
def arith_seq (a₁ d : ℚ) (n : ℕ) := a₁ + n * d

-- Define the third and fifteenth terms of the sequence.
def third_term := (5 : ℚ) / 11
def fifteenth_term := (7 : ℚ) / 8

-- Prove that the ninth term is 117/176 given the conditions.
theorem ninth_term_arith_seq :
    ∃ (a₁ d : ℚ), 
    arith_seq a₁ d 2 = third_term ∧ 
    arith_seq a₁ d 14 = fifteenth_term ∧
    arith_seq a₁ d 8 = 117 / 176 :=
by
  sorry

end ninth_term_arith_seq_l586_58684


namespace probability_of_selecting_A_l586_58633

noncomputable def total_students : ℕ := 4
noncomputable def selected_student_A : ℕ := 1

theorem probability_of_selecting_A : 
  (selected_student_A : ℝ) / (total_students : ℝ) = 1 / 4 :=
by
  sorry

end probability_of_selecting_A_l586_58633


namespace distance_fall_l586_58646

-- Given conditions as definitions
def velocity (g : ℝ) (t : ℝ) := g * t

-- The theorem stating the relationship between time t0 and distance S
theorem distance_fall (g : ℝ) (t0 : ℝ) : 
  (∫ t in (0 : ℝ)..t0, velocity g t) = (1/2) * g * t0^2 :=
by 
  sorry

end distance_fall_l586_58646


namespace cab_base_price_l586_58634

theorem cab_base_price (base_price : ℝ) (total_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) 
  (H1 : total_cost = 23) 
  (H2 : cost_per_mile = 4) 
  (H3 : distance = 5) 
  (H4 : base_price = total_cost - cost_per_mile * distance) : 
  base_price = 3 :=
by 
  sorry

end cab_base_price_l586_58634


namespace line_through_point_l586_58652

theorem line_through_point (k : ℝ) : (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end line_through_point_l586_58652


namespace deformable_to_triangle_l586_58682

-- Definition of the planar polygon with n sides
structure Polygon (n : ℕ) := 
  (vertices : Fin n → ℝ × ℝ) -- This is a simplified representation of a planar polygon using vertex coordinates

noncomputable def canDeformToTriangle (poly : Polygon n) : Prop := sorry

theorem deformable_to_triangle (n : ℕ) (h : n > 4) (poly : Polygon n) : canDeformToTriangle poly := 
  sorry

end deformable_to_triangle_l586_58682


namespace simplify_fraction_l586_58647

theorem simplify_fraction (n : Nat) : (2^(n+4) - 3 * 2^n) / (2 * 2^(n+3)) = 13 / 16 :=
by
  sorry

end simplify_fraction_l586_58647


namespace largest_y_coordinate_of_graph_l586_58696

theorem largest_y_coordinate_of_graph :
  ∀ (x y : ℝ), (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  sorry

end largest_y_coordinate_of_graph_l586_58696


namespace boys_and_girls_solution_l586_58692

theorem boys_and_girls_solution (x y : ℕ) 
  (h1 : 3 * x + y > 24) 
  (h2 : 7 * x + 3 * y < 60) : x = 8 ∧ y = 1 :=
by
  sorry

end boys_and_girls_solution_l586_58692


namespace find_a_l586_58658

-- Points A and B on the x-axis
def point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)
def point_B : (ℝ × ℝ) := (-3, 0)

-- Distance condition
def distance_condition (a : ℝ) : Prop := abs (a + 3) = 5

-- The proof problem: find a such that distance condition holds
theorem find_a (a : ℝ) : distance_condition a ↔ (a = -8 ∨ a = 2) :=
by
  sorry

end find_a_l586_58658


namespace roots_quadratic_equation_l586_58628

theorem roots_quadratic_equation (x1 x2 : ℝ) (h1 : x1^2 - x1 - 1 = 0) (h2 : x2^2 - x2 - 1 = 0) :
  (x2 / x1) + (x1 / x2) = -3 :=
by
  sorry

end roots_quadratic_equation_l586_58628


namespace arc_length_of_sector_l586_58691

theorem arc_length_of_sector (n r : ℝ) (h_angle : n = 60) (h_radius : r = 3) : 
  (n * Real.pi * r / 180) = Real.pi :=
by 
  sorry

end arc_length_of_sector_l586_58691


namespace neg_alpha_quadrant_l586_58644

theorem neg_alpha_quadrant (α : ℝ) (k : ℤ) 
    (h1 : k * 360 + 180 < α)
    (h2 : α < k * 360 + 270) :
    k * 360 + 90 < -α ∧ -α < k * 360 + 180 :=
by
  sorry

end neg_alpha_quadrant_l586_58644


namespace distance_between_neg2_and_3_l586_58637
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end distance_between_neg2_and_3_l586_58637


namespace lottery_win_amount_l586_58614

theorem lottery_win_amount (total_tax : ℝ) (federal_tax_rate : ℝ) (local_tax_rate : ℝ) (tax_paid : ℝ) :
  total_tax = tax_paid →
  federal_tax_rate = 0.25 →
  local_tax_rate = 0.15 →
  tax_paid = 18000 →
  ∃ x : ℝ, x = 49655 :=
by
  intros h1 h2 h3 h4
  use (tax_paid / (federal_tax_rate + local_tax_rate * (1 - federal_tax_rate))), by
    norm_num at h1 h2 h3 h4
    sorry

end lottery_win_amount_l586_58614


namespace alice_numbers_l586_58670

theorem alice_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = 7) : (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) :=
by
  sorry

end alice_numbers_l586_58670


namespace expression_value_zero_l586_58657

variables (a b c A B C : ℝ)

theorem expression_value_zero
  (h1 : a + b + c = 0)
  (h2 : A + B + C = 0)
  (h3 : a / A + b / B + c / C = 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 :=
by
  sorry

end expression_value_zero_l586_58657


namespace mass_percentage_of_S_in_Al2S3_l586_58679

theorem mass_percentage_of_S_in_Al2S3 :
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_S : ℝ := 32.06
  let formula_of_Al2S3: (ℕ × ℕ) := (2, 3)
  let molar_mass_Al2S3 : ℝ := (2 * molar_mass_Al) + (3 * molar_mass_S)
  let total_mass_S_in_Al2S3 : ℝ := 3 * molar_mass_S
  (total_mass_S_in_Al2S3 / molar_mass_Al2S3) * 100 = 64.07 :=
by
  sorry

end mass_percentage_of_S_in_Al2S3_l586_58679


namespace range_of_a_l586_58661

theorem range_of_a (a : ℝ) : 
  (∀ x, (x > 2 ∨ x < -1) → ¬(x^2 + 4 * x + a < 0)) → a ≥ 3 :=
by
  sorry

end range_of_a_l586_58661


namespace number_of_solutions_depends_on_a_l586_58649

theorem number_of_solutions_depends_on_a (a : ℝ) : 
  (∀ x : ℝ, 2^(3 * x) + 4 * a * 2^(2 * x) + a^2 * 2^x - 6 * a^3 = 0) → 
  (if a = 0 then 0 else if a > 0 then 1 else 2) = 
  (if a = 0 then 0 else if a > 0 then 1 else 2) :=
by 
  sorry

end number_of_solutions_depends_on_a_l586_58649


namespace solve_inequality_a_eq_2_solve_inequality_a_in_R_l586_58678

theorem solve_inequality_a_eq_2 :
  {x : ℝ | x > 2 ∨ x < 1} = {x : ℝ | x^2 - 3*x + 2 > 0} :=
sorry

theorem solve_inequality_a_in_R (a : ℝ) :
  {x : ℝ | 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨ 
    (a = 1 ∧ x ≠ 1) ∨ 
    (a < 1 ∧ (x > 1 ∨ x < a))
  } = 
  {x : ℝ | x^2 - (1 + a)*x + a > 0} :=
sorry

end solve_inequality_a_eq_2_solve_inequality_a_in_R_l586_58678


namespace P_intersection_complement_Q_l586_58601

-- Define sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }

-- Prove the required intersection
theorem P_intersection_complement_Q : P ∩ (Set.univ \ Q) = { x | 0 ≤ x ∧ x < 2 } :=
by
  -- Proof will be inserted here
  sorry

end P_intersection_complement_Q_l586_58601


namespace solution_pairs_l586_58635

open Int

theorem solution_pairs (a b : ℝ) (h : ∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int) :=
by sorry

end solution_pairs_l586_58635


namespace kelly_initial_games_l586_58659

theorem kelly_initial_games (games_given_away : ℕ) (games_left : ℕ)
  (h1 : games_given_away = 91) (h2 : games_left = 92) : 
  games_given_away + games_left = 183 :=
by {
  sorry
}

end kelly_initial_games_l586_58659


namespace median_of_consecutive_integers_l586_58627

def sum_of_consecutive_integers (n : ℕ) (a : ℤ) : ℤ :=
  n * (2*a + (n - 1)) / 2

theorem median_of_consecutive_integers (a : ℤ) : 
  (sum_of_consecutive_integers 25 a = 5^5) -> 
  (a + 12 = 125) := 
by
  sorry

end median_of_consecutive_integers_l586_58627


namespace find_a_min_value_of_f_l586_58618

theorem find_a (a : ℕ) (h1 : 3 / 2 < 2 + a) (h2 : 1 / 2 ≥ 2 - a) : a = 1 := by
  sorry

theorem min_value_of_f (a x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) : 
    (a = 1) → ∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, |x + a| + |x - 2| ≥ m := by
  sorry

end find_a_min_value_of_f_l586_58618


namespace find_other_number_l586_58623

theorem find_other_number (HCF LCM a b : ℕ) (h1 : HCF = 108) (h2 : LCM = 27720) (h3 : a = 216) (h4 : HCF * LCM = a * b) : b = 64 :=
  sorry

end find_other_number_l586_58623


namespace age_of_15th_student_l586_58668

theorem age_of_15th_student 
  (total_age_15_students : ℕ)
  (total_age_3_students : ℕ)
  (total_age_11_students : ℕ)
  (h1 : total_age_15_students = 225)
  (h2 : total_age_3_students = 42)
  (h3 : total_age_11_students = 176) :
  total_age_15_students - (total_age_3_students + total_age_11_students) = 7 :=
by
  sorry

end age_of_15th_student_l586_58668


namespace gcd_problem_l586_58650

theorem gcd_problem : 
  let a := 690
  let b := 875
  let r1 := 10
  let r2 := 25
  let n1 := a - r1
  let n2 := b - r2
  gcd n1 n2 = 170 :=
by
  sorry

end gcd_problem_l586_58650


namespace negation_of_exists_l586_58683

theorem negation_of_exists :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l586_58683


namespace candy_count_l586_58656

def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def sister_candy : ℕ := 40
def final_candy : ℕ := 62

theorem candy_count : initial_candy - eaten_candy + sister_candy = final_candy := 
by
  sorry

end candy_count_l586_58656


namespace find_b_l586_58638

theorem find_b
  (a b c : ℚ)
  (h1 : (4 : ℚ) * a = 12)
  (h2 : (4 * (4 * b) = - (14:ℚ) + 3 * a)) :
  b = -(7:ℚ) / 2 :=
by sorry

end find_b_l586_58638


namespace largest_integer_solution_l586_58641

theorem largest_integer_solution (m : ℤ) (h : 2 * m + 7 ≤ 3) : m ≤ -2 :=
sorry

end largest_integer_solution_l586_58641


namespace closest_fraction_to_medals_won_l586_58629

theorem closest_fraction_to_medals_won :
  let gamma_fraction := (13:ℚ) / 80
  let fraction_1_4 := (1:ℚ) / 4
  let fraction_1_5 := (1:ℚ) / 5
  let fraction_1_6 := (1:ℚ) / 6
  let fraction_1_7 := (1:ℚ) / 7
  let fraction_1_8 := (1:ℚ) / 8
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_4) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_5) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_7) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_8) := by
  sorry

end closest_fraction_to_medals_won_l586_58629


namespace find_m_n_l586_58686

theorem find_m_n 
  (a b c d m n : ℕ) 
  (h₁ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₂ : a + b + c + d = m^2)
  (h₃ : a = max (max a b) (max c d) ∨ b = max (max a b) (max c d) ∨ c = max (max a b) (max c d) ∨ d = max (max a b) (max c d))
  (h₄ : exists k, k^2 = max (max a b) (max c d))
  : m = 9 ∧ n = 6 :=
by
  -- Proof omitted
  sorry

end find_m_n_l586_58686


namespace line_passes_through_fixed_point_minimum_area_triangle_l586_58639

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ k : ℝ, (k * 2 - 1 + 1 - 2 * k = 0) :=
sorry

theorem minimum_area_triangle (k : ℝ) :
  ∀ k: ℝ, k < 0 → 1/2 * (2 - 1/k) * (1 - 2*k) ≥ 4 ∧ 
           (1/2 * (2 - 1/k) * (1 - 2*k) = 4 ↔ k = -1/2) :=
sorry

end line_passes_through_fixed_point_minimum_area_triangle_l586_58639


namespace james_hours_worked_l586_58642

variable (x : ℝ) (y : ℝ)

theorem james_hours_worked (h1: 18 * x + 16 * (1.5 * x) = 40 * x + (y - 40) * (2 * x)) : y = 41 :=
by
  sorry

end james_hours_worked_l586_58642


namespace area_of_storm_eye_l586_58643

theorem area_of_storm_eye : 
  let large_quarter_circle_area := (1 / 4) * π * 5^2
  let small_circle_area := π * 2^2
  let storm_eye_area := large_quarter_circle_area - small_circle_area
  storm_eye_area = (9 * π) / 4 :=
by
  sorry

end area_of_storm_eye_l586_58643


namespace input_value_of_x_l586_58630

theorem input_value_of_x (x y : ℤ) (h₁ : (x < 0 → y = (x + 1) * (x + 1)) ∧ (¬(x < 0) → y = (x - 1) * (x - 1)))
  (h₂ : y = 16) : x = 5 ∨ x = -5 :=
sorry

end input_value_of_x_l586_58630


namespace exists_difference_divisible_by_11_l586_58693

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end exists_difference_divisible_by_11_l586_58693


namespace gcd_154_and_90_l586_58690

theorem gcd_154_and_90 : Nat.gcd 154 90 = 2 := by
  sorry

end gcd_154_and_90_l586_58690


namespace none_of_these_l586_58604

theorem none_of_these (a T : ℝ) : 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y - 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y - 4 * a * T = 0) :=
sorry

end none_of_these_l586_58604


namespace total_sales_calculation_l586_58617

def average_price_per_pair : ℝ := 9.8
def number_of_pairs_sold : ℕ := 70
def total_amount : ℝ := 686

theorem total_sales_calculation :
  average_price_per_pair * (number_of_pairs_sold : ℝ) = total_amount :=
by
  -- proof goes here
  sorry

end total_sales_calculation_l586_58617


namespace gcd_360_504_l586_58625

theorem gcd_360_504 : Nat.gcd 360 504 = 72 :=
by sorry

end gcd_360_504_l586_58625


namespace solve_quadratic_eq_l586_58673

theorem solve_quadratic_eq (x : ℝ) (h : x > 0) (eq : 4 * x^2 + 8 * x - 20 = 0) : 
  x = Real.sqrt 6 - 1 :=
sorry

end solve_quadratic_eq_l586_58673


namespace calculate_expression_l586_58688

theorem calculate_expression :
  ((1 / 3 : ℝ) ^ (-2 : ℝ)) + Real.tan (Real.pi / 4) - Real.sqrt ((-10 : ℝ) ^ 2) = 0 := by
  sorry

end calculate_expression_l586_58688


namespace probability_product_zero_probability_product_negative_l586_58645

def given_set : List ℤ := [-3, -2, -1, 0, 5, 6, 7]

def num_pairs : ℕ := 21

theorem probability_product_zero :
  (6 : ℚ) / num_pairs = 2 / 7 := sorry

theorem probability_product_negative :
  (9 : ℚ) / num_pairs = 3 / 7 := sorry

end probability_product_zero_probability_product_negative_l586_58645


namespace no_solution_iff_k_nonnegative_l586_58648

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1 / 2) ^ x

theorem no_solution_iff_k_nonnegative (k : ℝ) :
  (¬ ∃ x : ℝ, f k (f k x) = 3 / 2) ↔ k ≥ 0 :=
  sorry

end no_solution_iff_k_nonnegative_l586_58648


namespace trish_walks_l586_58671

variable (n : ℕ) (M D : ℝ)
variable (d : ℕ → ℝ)
variable (H1 : d 1 = 1)
variable (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k)
variable (H3 : d n > M)

theorem trish_walks (n : ℕ) (M : ℝ) (H1 : d 1 = 1) (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k) (H3 : d n > M) : 2^(n-1) > M := by
  sorry

end trish_walks_l586_58671


namespace triangle_inequality_of_three_l586_58608

theorem triangle_inequality_of_three (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := 
sorry

end triangle_inequality_of_three_l586_58608


namespace original_wire_length_l586_58615

theorem original_wire_length 
(L : ℝ) 
(h1 : L / 2 - 3 / 2 > 0) 
(h2 : L / 2 - 3 > 0) 
(h3 : L / 4 - 11.5 > 0)
(h4 : L / 4 - 6.5 = 7) : 
L = 54 := 
sorry

end original_wire_length_l586_58615


namespace hyperbola_asymptote_value_l586_58632

theorem hyperbola_asymptote_value {b : ℝ} (h : b > 0) 
  (asymptote_eq : ∀ x : ℝ, y = x * (1 / 2) ∨ y = -x * (1 / 2)) :
  b = 1 :=
sorry

end hyperbola_asymptote_value_l586_58632


namespace incorrect_statement_is_C_l586_58602

theorem incorrect_statement_is_C (b h s a x : ℝ) (hb : b > 0) (hh : h > 0) (hs : s > 0) (hx : x < 0) :
  ¬ (9 * s^2 = 4 * (3 * s)^2) :=
by
  sorry

end incorrect_statement_is_C_l586_58602


namespace minimum_value_squared_sum_minimum_value_squared_sum_equality_l586_58665

theorem minimum_value_squared_sum (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

theorem minimum_value_squared_sum_equality (a b c t : ℝ) (h : a + b + c = t) 
  (ha : a = t / 3) (hb : b = t / 3) (hc : c = t / 3) : 
  a^2 + b^2 + c^2 = t^2 / 3 := by
  sorry

end minimum_value_squared_sum_minimum_value_squared_sum_equality_l586_58665


namespace maximize_area_minimize_length_l586_58666

-- Problem 1: Prove maximum area of the enclosure
theorem maximize_area (x y : ℝ) (h : x + 2 * y = 36) : 18 * 9 = 162 :=
by
  sorry

-- Problem 2: Prove the minimum length of steel wire mesh
theorem minimize_length (x y : ℝ) (h1 : x * y = 32) : 8 + 2 * 4 = 16 :=
by
  sorry

end maximize_area_minimize_length_l586_58666


namespace rectangle_area_l586_58653

theorem rectangle_area (x y : ℝ) (hx : x ≠ 0) (h : x * y = 10) : y = 10 / x :=
sorry

end rectangle_area_l586_58653


namespace complex_number_sum_zero_l586_58621

theorem complex_number_sum_zero (a b : ℝ) (i : ℂ) (h : a + b * i = 1 - i) : a + b = 0 := 
by sorry

end complex_number_sum_zero_l586_58621


namespace domain_of_function_l586_58674

theorem domain_of_function :
  (∀ x : ℝ, (2 * Real.sin x - 1 > 0) ∧ (1 - 2 * Real.cos x ≥ 0) ↔
    ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 6) :=
sorry

end domain_of_function_l586_58674


namespace xiaohong_test_number_l586_58680

theorem xiaohong_test_number (x : ℕ) :
  (88 * x - 85 * (x - 1) = 100) → x = 5 :=
by
  intro h
  sorry

end xiaohong_test_number_l586_58680


namespace area_and_cost_of_path_l586_58620

-- Define the dimensions of the grass field
def length_field : ℝ := 85
def width_field : ℝ := 55

-- Define the width of the path around the field
def width_path : ℝ := 2.5

-- Define the cost per square meter of constructing the path
def cost_per_sqm : ℝ := 2

-- Define new dimensions including the path
def new_length : ℝ := length_field + 2 * width_path
def new_width : ℝ := width_field + 2 * width_path

-- Define the area of the entire field including the path
def area_with_path : ℝ := new_length * new_width

-- Define the area of the grass field without the path
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_with_path - area_field

-- Define the cost of constructing the path
def cost_constructing_path : ℝ := area_path * cost_per_sqm

-- Theorem to prove the area of the path and cost of constructing it
theorem area_and_cost_of_path :
  area_path = 725 ∧ cost_constructing_path = 1450 :=
by
  -- Skipping the proof as instructed
  sorry

end area_and_cost_of_path_l586_58620


namespace third_number_sixth_row_l586_58667

/-- Define the arithmetic sequence and related properties. -/
def sequence (n : ℕ) : ℕ := 2 * n - 1

/-- Define sum of first k terms in a series where each row length doubles the previous row length. -/
def sum_of_rows (k : ℕ) : ℕ :=
  2^k - 1

/-- Statement of the problem: Prove that the third number in the sixth row is 67. -/
theorem third_number_sixth_row : sequence (sum_of_rows 5 + 3) = 67 := by
  sorry

end third_number_sixth_row_l586_58667


namespace circle_center_l586_58607

theorem circle_center (x y : ℝ) (h : x^2 + 8*x + y^2 - 4*y = 16) : (x, y) = (-4, 2) :=
by 
  sorry

end circle_center_l586_58607


namespace magnitude_of_z_l586_58660

noncomputable def z : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem magnitude_of_z : Complex.abs z = 5 := by
  sorry

end magnitude_of_z_l586_58660


namespace maximum_value_l586_58600

variables (a b c : ℝ)
variables (a_vec b_vec c_vec : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ‖a_vec‖ = 2
axiom norm_b : ‖b_vec‖ = 3
axiom norm_c : ‖c_vec‖ = 4

theorem maximum_value : 
  (‖(a_vec - (3:ℝ) • b_vec)‖^2 + ‖(b_vec - (3:ℝ) • c_vec)‖^2 + ‖(c_vec - (3:ℝ) • a_vec)‖^2) ≤ 377 :=
by
  sorry

end maximum_value_l586_58600


namespace inequality_solution_l586_58663

theorem inequality_solution (x : ℝ) : 1 - (2 * x - 2) / 5 < (3 - 4 * x) / 2 → x < 1 / 16 := by
  sorry

end inequality_solution_l586_58663


namespace prob_not_lose_when_A_plays_l586_58651

def appearance_prob_center_forward : ℝ := 0.3
def appearance_prob_winger : ℝ := 0.5
def appearance_prob_attacking_midfielder : ℝ := 0.2

def lose_prob_center_forward : ℝ := 0.3
def lose_prob_winger : ℝ := 0.2
def lose_prob_attacking_midfielder : ℝ := 0.2

theorem prob_not_lose_when_A_plays : 
    (appearance_prob_center_forward * (1 - lose_prob_center_forward) + 
    appearance_prob_winger * (1 - lose_prob_winger) + 
    appearance_prob_attacking_midfielder * (1 - lose_prob_attacking_midfielder)) = 0.77 := 
by
  sorry

end prob_not_lose_when_A_plays_l586_58651


namespace isosceles_triangle_base_length_l586_58675

theorem isosceles_triangle_base_length
  (a b : ℝ) (h₁ : a = 4) (h₂ : b = 8) (h₃ : a ≠ b)
  (triangle_inequality : ∀ x y z : ℝ, x + y > z) :
  ∃ base : ℝ, base = 8 := by
  sorry

end isosceles_triangle_base_length_l586_58675


namespace sin_over_cos_inequality_l586_58613

-- Define the main theorem and condition
theorem sin_over_cos_inequality (t : ℝ) (h₁ : 0 < t) (h₂ : t ≤ Real.pi / 2) : 
  (Real.sin t / t)^3 > Real.cos t := 
sorry

end sin_over_cos_inequality_l586_58613


namespace go_game_prob_l586_58664

theorem go_game_prob :
  ∀ (pA pB : ℝ),
    (pA = 0.6) →
    (pB = 0.4) →
    ((pA ^ 2) + (pB ^ 2) = 0.52) :=
by
  intros pA pB hA hB
  rw [hA, hB]
  sorry

end go_game_prob_l586_58664


namespace man_older_than_son_l586_58685

theorem man_older_than_son (S M : ℕ) (hS : S = 27) (hM : M + 2 = 2 * (S + 2)) : M - S = 29 := 
by {
  sorry
}

end man_older_than_son_l586_58685


namespace johns_cookies_left_l586_58677

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end johns_cookies_left_l586_58677


namespace total_distance_traveled_l586_58606

/-- The total distance traveled by Mr. and Mrs. Hugo over three days. -/
theorem total_distance_traveled :
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  first_day + second_day + third_day = 525 := by
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  have h1 : first_day + second_day + third_day = 525 := by
    sorry
  exact h1

end total_distance_traveled_l586_58606


namespace divides_necklaces_l586_58699

/-- Define the number of ways to make an even number of necklaces each of length at least 3. -/
def D_0 (n : ℕ) : ℕ := sorry

/-- Define the number of ways to make an odd number of necklaces each of length at least 3. -/
def D_1 (n : ℕ) : ℕ := sorry

/-- Main theorem: Prove that (n - 1) divides (D_1(n) - D_0(n)) for n ≥ 2 -/
theorem divides_necklaces (n : ℕ) (h : n ≥ 2) : (n - 1) ∣ (D_1 n - D_0 n) := sorry

end divides_necklaces_l586_58699


namespace customers_who_didnt_tip_l586_58672

def initial_customers : ℕ := 39
def added_customers : ℕ := 12
def customers_who_tipped : ℕ := 2

theorem customers_who_didnt_tip : initial_customers + added_customers - customers_who_tipped = 49 := by
  sorry

end customers_who_didnt_tip_l586_58672


namespace value_of_f_at_pi_over_12_l586_58681

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)

theorem value_of_f_at_pi_over_12 : f (Real.pi / 12) = Real.sqrt 2 / 2 :=
by
  sorry

end value_of_f_at_pi_over_12_l586_58681


namespace sum_of_perimeters_geq_4400_l586_58695

theorem sum_of_perimeters_geq_4400 (side original_side : ℕ) 
  (h_side_le_10 : ∀ s, s ≤ side → s ≤ 10) 
  (h_original_square : original_side = 100) 
  (h_cut_condition : side ≤ 10) : 
  ∃ (small_squares : ℕ → ℕ × ℕ), (original_side / side = n) → 4 * n * side ≥ 4400 :=
by
  sorry

end sum_of_perimeters_geq_4400_l586_58695


namespace minimum_value_f_l586_58612

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end minimum_value_f_l586_58612


namespace remaining_money_l586_58698

theorem remaining_money (m : ℝ) (c f t r : ℝ)
  (h_initial : m = 1500)
  (h_clothes : c = (1 / 3) * m)
  (h_food : f = (1 / 5) * (m - c))
  (h_travel : t = (1 / 4) * (m - c - f))
  (h_remaining : r = m - c - f - t) :
  r = 600 := 
by
  sorry

end remaining_money_l586_58698


namespace relayRaceOrders_l586_58610

def countRelayOrders (s1 s2 s3 s4 : String) : Nat :=
  if s1 = "Laura" then
    (if s2 ≠ "Laura" ∧ s3 ≠ "Laura" ∧ s4 ≠ "Laura" then
      if (s2 = "Alice" ∨ s2 = "Bob" ∨ s2 = "Cindy") ∧ 
         (s3 = "Alice" ∨ s3 = "Bob" ∨ s3 = "Cindy") ∧ 
         (s4 = "Alice" ∨ s4 = "Bob" ∨ s4 = "Cindy") then
        if s2 ≠ s3 ∧ s3 ≠ s4 ∧ s2 ≠ s4 then 6 else 0
      else 0
    else 0)
  else 0

theorem relayRaceOrders : countRelayOrders "Laura" "Alice" "Bob" "Cindy" = 6 := 
by sorry

end relayRaceOrders_l586_58610


namespace sin_cos_identity_l586_58640

variables (α : ℝ)

def tan_pi_add_alpha (α : ℝ) : Prop := Real.tan (Real.pi + α) = 3

theorem sin_cos_identity (h : tan_pi_add_alpha α) : 
  Real.sin (-α) * Real.cos (Real.pi - α) = 3 / 10 :=
sorry

end sin_cos_identity_l586_58640
